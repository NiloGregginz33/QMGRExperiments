import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import matplotlib.pyplot as plt
from braket.circuits import Circuit
from braket.devices import LocalSimulator
from utils.experiment_logger import PhysicsExperimentLogger

class EntanglementRobustnessExperiment:
    def __init__(self, device):
        self.device = device
        self.num_qubits = 6
        self.logger = PhysicsExperimentLogger("entanglement_robustness")

    def shannon_entropy(self, probs):
        probs = np.array(probs)
        probs = probs / np.sum(probs)
        return -np.sum(probs * np.log2(probs + 1e-12))

    def marginal_probs(self, probs, total_qubits, target_idxs):
        marginal = {}
        for idx, p in enumerate(probs):
            b = format(idx, f"0{total_qubits}b")
            key = ''.join([b[i] for i in target_idxs])
            marginal[key] = marginal.get(key, 0) + p
        return np.array(list(marginal.values()))

    def compute_mi(self, probs, qA, qB, total_qubits):
        AB = self.marginal_probs(probs, total_qubits, [qA, qB])
        A = self.marginal_probs(probs, total_qubits, [qA])
        B = self.marginal_probs(probs, total_qubits, [qB])
        return self.shannon_entropy(A) + self.shannon_entropy(B) - self.shannon_entropy(AB)

    def build_circuit(self, num_cnots):
        circ = Circuit()
        circ.h(0)
        for i in range(1, num_cnots + 1):
            circ.cnot(0, i)
        circ.probability()
        return circ

    def run(self):
        results = []
        entropies = []
        mi_matrices = []
        for num_cnots in range(self.num_qubits):
            circ = self.build_circuit(num_cnots)
            task = self.device.run(circ, shots=1024)
            result = task.result()
            probs = np.array(result.values).reshape(-1)

            entropy = self.shannon_entropy(probs)
            mi_matrix = np.zeros((self.num_qubits, self.num_qubits))
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    mi = self.compute_mi(probs, i, j, self.num_qubits)
                    mi_matrix[i, j] = mi_matrix[j, i] = mi

            avg_mi = np.mean(mi_matrix)

            # Validation
            valid = True
            if np.isnan(entropy) or entropy < -1e-6:
                print(f"[WARNING] Step {num_cnots}: Invalid entropy value: {entropy}")
                valid = False
            if np.isnan(avg_mi) or avg_mi < -1e-6 or avg_mi > np.log2(self.num_qubits) + 1:
                print(f"[WARNING] Step {num_cnots}: Invalid average MI value: {avg_mi}")
                valid = False

            print(f"Step {num_cnots}: Entropy = {entropy:.4f}, Average MI = {avg_mi:.4f} {'[VALID]' if valid else '[INVALID]'}")

            result_data = {
                "num_cnots": num_cnots,
                "entropy": entropy,
                "mi_matrix": mi_matrix.tolist(),
                "avg_mi": avg_mi,
                "valid": valid
            }
            results.append(result_data)
            self.logger.log_result(result_data)
            entropies.append(entropy)
            mi_matrices.append(mi_matrix)

        self.plot_results(entropies, mi_matrices)
        return results

    def plot_results(self, entropies, mi_matrices):
        # Plot entropy vs. number of CNOTs
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(self.num_qubits), entropies, marker='o')
        plt.xlabel('Number of CNOTs')
        plt.ylabel('Entropy (bits)')
        plt.title('Entropy vs. Entanglement Robustness')
        plt.grid(True)

        # Plot average mutual information vs. number of CNOTs
        avg_mi = [np.mean(mi) for mi in mi_matrices]
        plt.subplot(1, 2, 2)
        plt.plot(range(self.num_qubits), avg_mi, marker='o')
        plt.xlabel('Number of CNOTs')
        plt.ylabel('Average Mutual Information (bits)')
        plt.title('Average MI vs. Entanglement Robustness')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('plots/entanglement_robustness.png')
        plt.close()

if __name__ == "__main__":
    device = LocalSimulator()
    experiment = EntanglementRobustnessExperiment(device)
    results = experiment.run()
    print("Entanglement Robustness Test completed. Results logged and plots saved.") 