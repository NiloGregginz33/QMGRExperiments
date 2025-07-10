import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeJakartaV2
from qiskit_ibm_runtime import QiskitRuntimeService
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) )
try:
    from src.CGPTFactory import run as cgpt_run
except ImportError:
    cgpt_run = None
from qiskit.quantum_info import partial_trace, entropy
from pathlib import Path
import argparse
import json

# Logger stub (replace with your logger if needed)
def log_result(metrics, log_dir="experiment_logs/bulk_reconstruction_qiskit"):
    os.makedirs(log_dir, exist_ok=True)
    idx = len([f for f in os.listdir(log_dir) if f.startswith('result_') and f.endswith('.json')]) + 1
    with open(os.path.join(log_dir, f"result_{idx}.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

class BulkReconstructionQiskit:
    def __init__(self, backend=None, shots=2048):
        self.num_qubits = 7  # 6 boundary + 1 bulk
        self.alpha_values = [1, 2, 3]
        self.backend = backend if backend is not None else FakeManilaV2()
        self.shots = shots

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

    def quantum_fisher_information(self, statevec_fn, param, eps=1e-6):
        state_plus = statevec_fn(param + eps)
        state_minus = statevec_fn(param - eps)
        derivative = (state_plus - state_minus) / (2 * eps)
        metric = np.abs(np.vdot(derivative, derivative) - np.abs(np.vdot(derivative, statevec_fn(param)))**2)
        return 4 * metric

    def geometric_phase(self, states):
        phase = 0
        for i in range(len(states)-1):
            overlap = np.vdot(states[i], states[i+1])
            phase += np.angle(overlap)
        return phase

    def marginal_fisher_info(self, p_bulk_current, p_bulk_prev, phi_current, phi_prev):
        delta_phi = phi_current - phi_prev
        if delta_phi == 0:
            return 0
        derivative = (np.array(p_bulk_current) - np.array(p_bulk_prev)) / delta_phi
        fisher = np.sum((derivative ** 2) / (np.array(p_bulk_current) + 1e-12))
        return fisher

    def build_happy_circuit(self, logical_state=0, phi=0):
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        # Create perfect tensor (6 qubits)
        for i in [0, 2, 4]:
            qc.h(i)
            qc.cx(i, i+1)
        # Entangle across pairs
        qc.cx(0, 2)
        qc.rz(np.pi, 2)
        qc.cx(0, 2)
        qc.cx(1, 4)
        qc.rz(np.pi, 4)
        qc.cx(1, 4)
        qc.cx(3, 5)
        qc.rz(np.pi, 5)
        qc.cx(3, 5)
        # Add bulk qubit (index 6) and encode logical state
        qc.h(6)
        if logical_state:
            qc.x(6)
        qc.cx(6, 2)
        qc.rz(phi, 6)
        # Add measurement to all qubits
        qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc

    def run(self):
        results = []
        phi_values = np.linspace(0, 2*np.pi, 10)
        for logical_state in [0, 1]:
            state_results = []
            prev_bulk = None
            prev_phi = None
            for phi_idx, phi_val in enumerate(phi_values):
                qc = self.build_happy_circuit(logical_state, phi_val)
                tqc = transpile(qc, self.backend)
                if cgpt_run is not None:
                    counts = cgpt_run(tqc, backend=self.backend, shots=self.shots)
                else:
                    # fallback: direct run (should not be used in production)
                    job = self.backend.run(tqc, shots=self.shots)
                    result = job.result()
                    counts = result.get_counts()
                # Convert counts to probabilities
                probs = np.zeros(2**self.num_qubits)
                for bitstring, count in counts.items():
                    idx = int(bitstring.replace(' ', ''), 2)
                    probs[idx] = count / self.shots
                # Reduced density matrix for bulk qubit (index 6) via classical marginalization
                # (approximate: sum probabilities where bulk qubit is 0 or 1)
                p_bulk0 = sum(probs[i] for i in range(len(probs)) if (i >> 6) & 1 == 0)
                p_bulk1 = sum(probs[i] for i in range(len(probs)) if (i >> 6) & 1 == 1)
                rho_bulk = np.array([[p_bulk0, 0], [0, p_bulk1]])
                ent_spectrum = [p_bulk0, p_bulk1]
                # Compute metrics
                shannon_entropy = self.shannon_entropy(probs)
                renyi_entropies = {f"S_{alpha}": self.renyi_entropy(probs, alpha) for alpha in self.alpha_values}
                vn_entropy = self.shannon_entropy([p_bulk0, p_bulk1])
                ent_spectrum = [p_bulk0, p_bulk1]
                # Bulk qubit probability for Fisher info
                bulk_prob = p_bulk0 + p_bulk1
                # Fisher info: marginal Fisher information on bulk qubit
                if prev_bulk is not None and prev_phi is not None:
                    fisher_info = self.marginal_fisher_info([p_bulk0, p_bulk1], prev_bulk, phi_val, prev_phi)
                else:
                    fisher_info = 0
                metrics = {
                    "logical_state": logical_state,
                    "phi": phi_val,
                    "shannon_entropy": shannon_entropy,
                    "renyi_entropies": renyi_entropies,
                    "von_neumann_entropy": vn_entropy,
                    "entanglement_spectrum": ent_spectrum,
                    "fisher_info": fisher_info
                }
                print(f"Logical state {logical_state}, phi={phi_val:.2f}:")
                print(f"  Shannon entropy: {metrics['shannon_entropy']:.4f}")
                print(f"  Renyi entropies: {metrics['renyi_entropies']}")
                print(f"  Von Neumann entropy: {metrics['von_neumann_entropy']:.4f}")
                print(f"  Entanglement spectrum: {metrics['entanglement_spectrum']}")
                log_result(metrics)
                state_results.append(metrics)
                prev_bulk = [p_bulk0, p_bulk1]
                prev_phi = phi_val
            results.append(state_results)
        # Flatten results to a single list of dicts, as in the original experiment
        flat_results = [item for sublist in results for item in sublist]
        # Save all results as a single results.json file
        log_dir = "experiment_logs/bulk_reconstruction_qiskit"
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "results.json"), 'w') as f:
            json.dump(flat_results, f, indent=2)
        # Write a summary.txt file
        with open(os.path.join(log_dir, "summary.txt"), 'w') as f:
            f.write(f"Bulk Reconstruction Qiskit Experiment\n")
            f.write(f"Backend: {self.backend}\n")
            f.write(f"Shots: {self.shots}\n")
            f.write(f"Num Qubits: {self.num_qubits}\n")
            f.write(f"Alpha values: {self.alpha_values}\n")
            f.write(f"Phi values: {list(phi_values)}\n\n")
            for state in [0, 1]:
                state_results = [r for r in flat_results if r['logical_state'] == state]
                f.write(f"Logical state {state}:\n")
                for r in state_results:
                    f.write(f"  phi={r['phi']:.2f}, Shannon Entropy={r['shannon_entropy']:.4f}, Renyi S_2={r['renyi_entropies']['S_2']:.4f}, Fisher Info={r['fisher_info']:.4f}\n")
                f.write("\n")
        # Save plots in the same log_dir
        self.plot_results(flat_results, log_dir)
        return results

    def plot_results(self, results, log_dir):
        # Shannon Entropy
        plt.figure()
        for state in [0, 1]:
            data = [r for r in results if r['logical_state'] == state]
            phis = [r['phi'] for r in data]
            entropies = [r['shannon_entropy'] for r in data]
            plt.plot(phis, entropies, marker='o', label=f'Logical {state}')
        plt.xlabel('Parameter phi')
        plt.ylabel('Shannon Entropy')
        plt.title('Entropy vs Parameter')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'shannon_entropy.png'))
        plt.close()

        # Renyi Entropies
        plt.figure()
        for state in [0, 1]:
            data = [r for r in results if r['logical_state'] == state]
            phis = [r['phi'] for r in data]
            renyis = [r['renyi_entropies']['S_2'] for r in data]  # Use string key as in metrics
            plt.plot(phis, renyis, marker='o', label=f'Logical {state}')
        plt.xlabel('Parameter phi')
        plt.ylabel('Renyi Entropy (alpha=2)')
        plt.title('Renyi Entropies vs Parameter')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'renyi_entropy.png'))
        plt.close()

        # Fisher Information (if present)
        if all('fisher_info' in r for r in results):
            plt.figure()
            for state in [0, 1]:
                data = [r for r in results if r['logical_state'] == state]
                phis = [r['phi'] for r in data]
                fishers = [r['fisher_info'] for r in data]
                plt.plot(phis, fishers, marker='o', label=f'Logical {state}')
            plt.xlabel('Parameter phi')
            plt.ylabel('Fisher Information')
            plt.title('Fisher Information vs Parameter')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, 'fisher_info.png'))
            plt.close()
        else:
            print("Fisher information not present in all results; skipping Fisher info plot.")

# Function for integration with runners
def run_bulk_reconstruction_qiskit(backend=None, shots=2048):
    experiment = BulkReconstructionQiskit(backend=backend, shots=shots)
    results = experiment.run()
    print("Bulk Reconstruction Qiskit experiment completed. Results logged and plots saved.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Qiskit bulk reconstruction experiment")
    parser.add_argument('--device', type=str, default='simulator', help='Device to use: "simulator" or IBMQ backend name')
    parser.add_argument('--shots', type=int, default=2048, help='Number of shots')
    args = parser.parse_args()

    if args.device.lower() == 'simulator':
        backend = FakeJakartaV2()
    else:
        service = QiskitRuntimeService()
        backend = service.backend(args.device)
    run_bulk_reconstruction_qiskit(backend=backend, shots=args.shots) 