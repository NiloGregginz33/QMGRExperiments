import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from braket.circuits import Circuit
from braket.devices import LocalSimulator
from utils.experiment_logger import PhysicsExperimentLogger

class BoundaryVsBulkEntropyExperiment:
    def __init__(self, device):
        self.device = device
        self.num_qubits = 6
        self.logger = PhysicsExperimentLogger("boundary_vs_bulk_entropy")

    def shannon_entropy(self, probs):
        probs = np.array(probs)
        probs = probs / np.sum(probs)
        return -np.sum(probs * np.log2(probs + 1e-12))

    def marginal_probs(self, probs, total_qubits, keep):
        marginal = {}
        for idx, p in enumerate(probs):
            b = format(idx, f"0{total_qubits}b")
            key = ''.join([b[i] for i in keep])
            marginal[key] = marginal.get(key, 0) + p
        return np.array(list(marginal.values()))

    def build_perfect_tensor(self):
        circ = Circuit()
        # Create 3 GHZ pairs: (0,1), (2,3), (4,5)
        for i in [0, 2, 4]:
            circ.h(i)
            circ.cnot(i, i+1)
        # Entangle across pairs (CZ gates)
        circ.cnot(0, 2).rz(2, np.pi).cnot(0, 2)
        circ.cnot(1, 4).rz(4, np.pi).cnot(1, 4)
        circ.cnot(3, 5).rz(5, np.pi).cnot(3, 5)
        # Optional RX rotation to break symmetry
        for q in range(6):
            circ.rx(q, np.pi / 4)
        circ.probability()
        return circ

    def run(self):
        circ = self.build_perfect_tensor()
        task = self.device.run(circ, shots=2048)
        result = task.result()
        probs = np.array(result.values).reshape(-1)

        entropies = []
        for cut_size in range(1, self.num_qubits):
            keep = list(range(cut_size))
            marg = self.marginal_probs(probs, self.num_qubits, keep)
            entropy = self.shannon_entropy(marg)
            valid = not np.isnan(entropy) and entropy >= -1e-6 and entropy <= cut_size
            print(f"Cut size {cut_size}: Entropy = {entropy:.4f} {'[VALID]' if valid else '[INVALID]'}")
            entropies.append(entropy)
            self.logger.log_result({
                "cut_size": cut_size,
                "entropy": entropy,
                "valid": bool(valid)
            })

        self.plot_results(entropies)
        return entropies

    def plot_results(self, entropies):
        plt.figure(figsize=(7, 5))
        plt.plot(range(1, self.num_qubits), entropies, marker='o')
        plt.xlabel('Boundary Cut Size (qubits)')
        plt.ylabel('Entropy (bits)')
        plt.title('Boundary vs. Bulk Entropy Scaling (Perfect Tensor)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plots/boundary_vs_bulk_entropy.png')
        plt.close()

if __name__ == "__main__":
    device = LocalSimulator()
    experiment = BoundaryVsBulkEntropyExperiment(device)
    entropies = experiment.run()
    print("Boundary vs. Bulk Entropy experiment completed. Results logged and plot saved.") 