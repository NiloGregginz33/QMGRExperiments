# Entropy Injection Experiment

import sys
import os

# Adjust Python path to include the Factory directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Factory'))

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

class EntropyInjectionExperiment:
    def __init__(self, num_qubits, entangle_pairs, trace_out_qubits):
        self.num_qubits = num_qubits
        self.entangle_pairs = entangle_pairs
        self.trace_out_qubits = trace_out_qubits
        self.circuit = QuantumCircuit(num_qubits)

    def create_bell_pairs(self):
        for q1, q2 in self.entangle_pairs:
            self.circuit.h(q1)
            self.circuit.cx(q1, q2)

    def add_gates(self, gates):
        for gate in gates:
            self.circuit.append(gate)

    def run_experiment(self):
        # Simulate the statevector
        state = Statevector.from_instruction(self.circuit)

        # Partial trace out specified qubits
        reduced_state = partial_trace(state, self.trace_out_qubits)

        # Calculate entropy of each subsystem
        entropies = [entropy(partial_trace(state, [i])) for i in range(self.num_qubits)]

        return reduced_state, entropies, state

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
experiment = EntropyInjectionExperiment(num_qubits=3, entangle_pairs=[(0, 1)], trace_out_qubits=[1])
experiment.create_bell_pairs()
reduced_state, entropies, state = experiment.run_experiment()
print("Reduced State:", reduced_state)
print("Entropies:", entropies)
print("Full Statevector:", state)

# Example visualization
entropies_over_time = [[0.5, 0.6, 0.7], [0.4, 0.5, 0.6], [0.3, 0.4, 0.5]]
experiment.visualize_entropy_vs_time(entropies_over_time)

mutual_information_matrix = [[0, 0.1, 0.2], [0.1, 0, 0.3], [0.2, 0.3, 0]]
experiment.plot_mutual_information_heatmap(mutual_information_matrix) 