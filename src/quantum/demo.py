from braket.circuits import Circuit, FreeParameter
from braket.devices import LocalSimulator
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from pathlib import Path

class HolographicDemo:
    def __init__(self, device):
        self.device = device
        self.timesteps = np.linspace(0, 3 * np.pi, 15)
        self.results = []

    def shannon_entropy(self, probs):
        """Calculate Shannon entropy of a probability distribution."""
        probs = np.array(probs)
        probs = probs / np.sum(probs)  # Normalize
        return -np.sum(probs * np.log2(probs + 1e-12))

    def marginal_probs(self, probs, total_qubits, keep):
        """Calculate marginal probabilities for specified qubits."""
        marginal = {}
        for idx, p in enumerate(probs):
            b = format(idx, f"0{total_qubits}b")
            key = ''.join([b[i] for i in keep])
            marginal[key] = marginal.get(key, 0) + p
        return np.array(list(marginal.values()))

    def compute_mi(self, probs, qA, qB, total_qubits):
        """Compute mutual information between two qubits."""
        AB = self.marginal_probs(probs, total_qubits, [qA, qB])
        A = self.marginal_probs(probs, total_qubits, [qA])
        B = self.marginal_probs(probs, total_qubits, [qB])
        return self.shannon_entropy(A) + self.shannon_entropy(B) - self.shannon_entropy(AB)

    def build_circuit(self, phi):
        """Build the holographic demo circuit."""
        circ = Circuit()
        # Initialize boundary state
        circ.h(0)
        circ.h(1)
        
        # Create bulk-boundary entanglement
        circ.cnot(0, 2)
        circ.cnot(1, 3)
        
        # Apply holographic dynamics
        circ.rx(0, phi)
        circ.rx(1, phi)
        circ.cz(0, 1)
        circ.cnot(2, 3)
        
        # Measure all qubits
        circ.probability()
        return circ

    def run(self):
        """Run the holographic demo experiment."""
        for phi_val in self.timesteps:
            # Build and run circuit
            circ = self.build_circuit(phi_val)
            task = self.device.run(circ, shots=1024)
            result = task.result()
            probs = np.array(result.values).reshape(-1)
            
            # Calculate metrics
            entropy = self.shannon_entropy(probs)
            
            # Compute mutual information matrix
            mi_matrix = np.zeros((4, 4))
            for i in range(4):
                for j in range(i + 1, 4):
                    mi = self.compute_mi(probs, i, j, 4)
                    mi_matrix[i, j] = mi_matrix[j, i] = mi
            
            # Store results
            self.results.append({
                "phase": phi_val,
                "entropy": entropy,
                "mi_matrix": mi_matrix,
                "circuit_depth": len(circ.instructions),
                "success_rate": 1.0  # Local simulator is perfect
            })
            
            print(f"Phase {phi_val:.2f}: Entropy = {entropy:.3f}")
        
        # Plot results
        self.plot_results()
        return self.results

    def plot_results(self):
        """Plot the experiment results."""
        phases = [r["phase"] for r in self.results]
        entropies = [r["entropy"] for r in self.results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(phases, entropies, 'o-', label='Entropy')
        plt.xlabel('Phase φ')
        plt.ylabel('Entropy (bits)')
        plt.title('Holographic Demo: Entropy vs Phase')
        plt.grid(True)
        plt.legend()
        
        # Save plot
        plot_dir = Path("plots")
        plot_dir.mkdir(exist_ok=True)
        plt.savefig(plot_dir / "holographic_demo.png")
        plt.close()
        print("Analysis complete. Results saved to holographic_demo.png") 