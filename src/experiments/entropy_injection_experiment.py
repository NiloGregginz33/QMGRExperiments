# Entropy Injection Experiment

import sys
import os

# Define the directory for logging
log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'experiment_logs', 'entropy_injection_experiment')
os.makedirs(log_dir, exist_ok=True)

# Adjust Python path to include the Factory directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
import numpy as np
import argparse
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from CGPTFactory import run
import json
from qiskit.quantum_info import partial_trace, entropy
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
# Remove the execute import as it's not needed
# from qiskit import execute
import matplotlib.pyplot as plt
import seaborn as sns

# Helper function to convert complex numbers to a serializable format
def complex_to_serializable(c):
    return {'real': c.real, 'imag': c.imag}

class EntropyInjectionExperiment:
    def __init__(self, num_qubits, entangle_pairs, trace_out_qubits, curvature):
        self.num_qubits = num_qubits
        self.entangle_pairs = entangle_pairs
        self.trace_out_qubits = trace_out_qubits
        self.curvature = curvature
        self.circuit = QuantumCircuit(num_qubits)

    def apply_entanglement(self):
        # Chain entanglers:
        weights = [0.5, 0.8][:self.num_qubits-1]
        for i, w in enumerate(weights):
            self.circuit.rzz(self.curvature * w, i, i+1)

        # **New**: direct entangler between qubit 0 and 2
        if self.num_qubits >= 3:
            self.circuit.rzz(self.curvature * 0.6, 0, 2)

    def run_experiment(self):
        self.apply_entanglement()
        # Simulate the statevector
        state = Statevector.from_instruction(self.circuit)

        # Partial trace out specified qubits
        reduced_state = partial_trace(state, self.trace_out_qubits)

        # Calculate entropy of each subsystem
        entropies = [entropy(partial_trace(state, [i])) for i in range(self.num_qubits)]

        # Compute mutual information matrix
        I_matrix = np.zeros((self.num_qubits, self.num_qubits))
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                I_matrix[i, j] = entropy(partial_trace(state, [i])) + entropy(partial_trace(state, [j])) - entropy(partial_trace(state, [i, j]))
                I_matrix[j, i] = I_matrix[i, j]

        # Normalize and convert to distance matrix
        I_matrix /= I_matrix.max()
        distances = -np.log(I_matrix + 1e-6)

        return reduced_state, entropies, state, distances

    def visualize_entropy_vs_time(self, entropies_over_time):
        plt.figure(figsize=(10, 6))
        for i, entropies in enumerate(entropies_over_time):
            plt.plot(entropies, label=f'Qubit {i}')
        plt.xlabel('Time Step')
        plt.ylabel('Entropy')
        plt.title('Entropy vs Time')
        plt.legend()
        plt.show()

    def plot_mutual_information_heatmap(self, mutual_information_matrix):
        plt.figure(figsize=(8, 6))
        sns.heatmap(mutual_information_matrix, annot=True, cmap='coolwarm', cbar=True)
        plt.title('Mutual Information Heatmap')
        plt.xlabel('Qubit Index')
        plt.ylabel('Qubit Index')
        plt.show()

    def measure_causal_witness(self):
        # Placeholder for causal witness measurement logic
        # Implement logic to measure causal witness values
        return "Causal witness measured"

# Example usage
experiment = EntropyInjectionExperiment(num_qubits=5, entangle_pairs=[(0, 1), (1, 2), (0, 2)], trace_out_qubits=[1], curvature=1.0)
experiment.apply_entanglement()
reduced_state, entropies, state, distances = experiment.run_experiment()
print("Reduced State:", reduced_state)
print("Entropies:", entropies)
print("Full Statevector:", state)
print("Distances:", distances)

# Log results to JSON with complex number handling
results = {
    "reduced_state": [[complex_to_serializable(c) for c in row] for row in reduced_state.data],
    "entropies": entropies,
    "full_statevector": [complex_to_serializable(c) for c in state.data],
    "distances": distances.tolist()
}

with open(os.path.join(log_dir, 'results.json'), 'w') as f:
    json.dump(results, f, indent=4)

# Save summary to text file
summary = """
Entropy Injection Experiment Summary

Reduced State:
{}

Entropies:
{}

Full Statevector:
{}

Distances:
{}
""".format(reduced_state, entropies, state, distances)

with open(os.path.join(log_dir, 'summary.txt'), 'w') as f:
    f.write(summary)

# Remove the clamping mechanism for cos(theta)
# Revert to the previous state without the placeholder calculation

# Example visualization
entropies_over_time = [[0.5, 0.6, 0.7], [0.4, 0.5, 0.6], [0.3, 0.4, 0.5]]
experiment.visualize_entropy_vs_time(entropies_over_time)

mutual_information_matrix = [[0, 0.1, 0.2], [0.1, 0, 0.3], [0.2, 0.3, 0]]
experiment.plot_mutual_information_heatmap(mutual_information_matrix)