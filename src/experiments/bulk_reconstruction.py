import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from braket.circuits import Circuit, FreeParameter
from braket.devices import LocalSimulator
from utils.experiment_logger import PhysicsExperimentLogger
from scipy.linalg import sqrtm
from scipy.stats import entropy as scipy_entropy

class BulkReconstructionExperiment:
    def __init__(self, device):
        self.device = device
        self.num_qubits = 7  # 6 boundary + 1 bulk
        self.logger = PhysicsExperimentLogger("bulk_reconstruction")
        self.alpha_values = [1, 2, 3]  # For Renyi entropies

    def shannon_entropy(self, probs):
        probs = np.array(probs)
        probs = probs / np.sum(probs)
        return -np.sum(probs * np.log2(probs + 1e-12))

    def renyi_entropy(self, probs, alpha):
        probs = np.array(probs)
        probs = probs / np.sum(probs)
        if alpha == 1:
            return self.shannon_entropy(probs)
        return 1/(1-alpha) * np.log2(np.sum(probs**alpha) + 1e-12)

    def quantum_fisher_information(self, state, param):
        """Calculate quantum Fisher information for a parameterized state"""
        # Numerical derivative
        eps = 1e-6
        state_plus = self.parameterize_state(state, param + eps)
        state_minus = self.parameterize_state(state, param - eps)
        derivative = (state_plus - state_minus) / (2 * eps)
        
        # Fubini-Study metric
        metric = np.abs(np.vdot(derivative, derivative) - 
                       np.abs(np.vdot(derivative, state))**2)
        return 4 * metric

    def geometric_phase(self, states):
        """Calculate geometric phase for a sequence of states"""
        phase = 0
        for i in range(len(states)-1):
            overlap = np.vdot(states[i], states[i+1])
            phase += np.angle(overlap)
        return phase

    def build_happy_circuit(self, logical_state=0):
        circ = Circuit()
        # Create perfect tensor (6 qubits)
        for i in [0, 2, 4]:
            circ.h(i)
            circ.cnot(i, i+1)
        
        # Entangle across pairs
        circ.cnot(0, 2).rz(2, np.pi).cnot(0, 2)
        circ.cnot(1, 4).rz(4, np.pi).cnot(1, 4)
        circ.cnot(3, 5).rz(5, np.pi).cnot(3, 5)
        
        # Add bulk qubit (index 6) and encode logical state
        circ.h(6)
        if logical_state:
            circ.x(6)
        circ.cnot(6, 2)  # Entangle bulk with boundary
        
        # Add parameterized rotation for Fisher information
        phi = FreeParameter("phi")
        circ.rz(6, phi)
        
        circ.probability()
        return circ

    def parameterize_state(self, state, param):
        """Apply parameterized rotation to state"""
        # Apply Rz rotation to the bulk qubit (index 6)
        rotation = np.exp(1j * param / 2)
        state_rotated = state.copy()
        for i in range(len(state)):
            if i & (1 << 6):  # If bulk qubit is 1
                state_rotated[i] *= rotation
        return state_rotated

    def von_neumann_entropy(self, rho):
        """Calculate von Neumann entropy of a density matrix"""
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical noise
        return -np.sum(eigenvalues * np.log2(eigenvalues))

    def entanglement_spectrum(self, state, partition):
        """Calculate entanglement spectrum for a given partition"""
        # Reshape state into matrix form
        dim = 2**len(partition)
        state_matrix = state.reshape(dim, -1)
        # Calculate reduced density matrix
        rho = state_matrix @ state_matrix.conj().T
        # Get eigenvalues
        eigenvalues = np.linalg.eigvalsh(rho)
        return eigenvalues[eigenvalues > 1e-10]

    def run(self):
        results = []
        states = []
        
        # Test different logical states
        for logical_state in [0, 1]:
            circ = self.build_happy_circuit(logical_state)
            
            # Run with different parameter values for Fisher information
            phi_values = np.linspace(0, 2*np.pi, 10)
            for phi_val in phi_values:
                task = self.device.run(circ, inputs={"phi": phi_val}, shots=2048)
                result = task.result()
                probs = np.array(result.values).reshape(-1)
                state = np.sqrt(probs)  # Amplitude
                states.append(state)
                
                # Calculate reduced density matrix for bulk qubit
                rho_bulk = np.zeros((2, 2), dtype=complex)
                for i in range(len(state)):
                    bulk_bit = (i >> 6) & 1
                    rho_bulk[bulk_bit, bulk_bit] += abs(state[i])**2
                
                # Calculate entanglement spectrum for bulk-boundary partition
                ent_spectrum = self.entanglement_spectrum(state, [6])
                
                # Calculate metrics
                metrics = {
                    "logical_state": logical_state,
                    "phi": phi_val,
                    "shannon_entropy": self.shannon_entropy(probs),
                    "renyi_entropies": {f"S_{alpha}": self.renyi_entropy(probs, alpha) 
                                      for alpha in self.alpha_values},
                    "fisher_info": self.quantum_fisher_information(state, phi_val),
                    "von_neumann_entropy": self.von_neumann_entropy(rho_bulk),
                    "entanglement_spectrum": ent_spectrum.tolist()
                }
                
                # Validate metrics
                valid = all(not np.isnan(v) for v in metrics.values() if isinstance(v, (int, float)))
                print(f"Logical state {logical_state}, φ={phi_val:.2f}:")
                print(f"  Shannon entropy: {metrics['shannon_entropy']:.4f}")
                print(f"  Renyi entropies: {metrics['renyi_entropies']}")
                print(f"  Fisher info: {metrics['fisher_info']:.4f}")
                print(f"  Von Neumann entropy: {metrics['von_neumann_entropy']:.4f}")
                print(f"  Entanglement spectrum: {metrics['entanglement_spectrum']}")
                print(f"  {'[VALID]' if valid else '[INVALID]'}")
                
                results.append(metrics)
                self.logger.log_result(metrics)
        
        # Calculate geometric phase
        geo_phase = self.geometric_phase(states)
        print(f"\nGeometric phase: {geo_phase:.4f}")
        
        self.plot_results(results)
        return results

    def plot_results(self, results):
        # Plot entropy vs parameter
        plt.figure(figsize=(15, 10))
        
        # Shannon entropy
        plt.subplot(231)
        for state in [0, 1]:
            data = [r for r in results if r['logical_state'] == state]
            phis = [r['phi'] for r in data]
            entropies = [r['shannon_entropy'] for r in data]
            plt.plot(phis, entropies, marker='o', label=f'Logical {state}')
        plt.xlabel('Parameter φ')
        plt.ylabel('Shannon Entropy')
        plt.title('Entropy vs Parameter')
        plt.legend()
        
        # Renyi entropies
        plt.subplot(232)
        for alpha in self.alpha_values:
            data = [r for r in results if r['logical_state'] == 0]
            phis = [r['phi'] for r in data]
            entropies = [r['renyi_entropies'][f'S_{alpha}'] for r in data]
            plt.plot(phis, entropies, marker='o', label=f'α={alpha}')
        plt.xlabel('Parameter φ')
        plt.ylabel('Renyi Entropy')
        plt.title('Renyi Entropies vs Parameter')
        plt.legend()
        
        # Fisher information
        plt.subplot(233)
        for state in [0, 1]:
            data = [r for r in results if r['logical_state'] == state]
            phis = [r['phi'] for r in data]
            fisher = [r['fisher_info'] for r in data]
            plt.plot(phis, fisher, marker='o', label=f'Logical {state}')
        plt.xlabel('Parameter φ')
        plt.ylabel('Quantum Fisher Information')
        plt.title('Fisher Information vs Parameter')
        plt.legend()
        
        # Von Neumann entropy
        plt.subplot(234)
        for state in [0, 1]:
            data = [r for r in results if r['logical_state'] == state]
            phis = [r['phi'] for r in data]
            vn_entropy = [r['von_neumann_entropy'] for r in data]
            plt.plot(phis, vn_entropy, marker='o', label=f'Logical {state}')
        plt.xlabel('Parameter φ')
        plt.ylabel('Von Neumann Entropy')
        plt.title('Von Neumann Entropy vs Parameter')
        plt.legend()
        
        # Entanglement spectrum
        plt.subplot(235)
        data = [r for r in results if r['logical_state'] == 0]
        phis = [r['phi'] for r in data]
        ent_spectra = [r['entanglement_spectrum'] for r in data]
        for i in range(len(ent_spectra[0])):
            spectrum = [spec[i] for spec in ent_spectra]
            plt.plot(phis, spectrum, marker='o', label=f'λ_{i+1}')
        plt.xlabel('Parameter φ')
        plt.ylabel('Eigenvalue')
        plt.title('Entanglement Spectrum vs Parameter')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('plots/bulk_reconstruction_metrics.png')
        plt.close()

if __name__ == "__main__":
    device = LocalSimulator()
    experiment = BulkReconstructionExperiment(device)
    results = experiment.run()
    print("Bulk Reconstruction experiment completed. Results logged and plots saved.") 