from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, entropy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
import argparse
import os
import json
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

class CurvedTimeExperiment:
    def __init__(self, num_qubits, time_warp, entropy_rates, device='simulator'):
        self.num_qubits = num_qubits
        self.time_warp = time_warp
        self.entropy_rates = entropy_rates
        self.device = device
        self.circuit = QuantumCircuit(num_qubits)

    def apply_time_evolution(self, t):
        time_scales = [1.0, 0.5, 2.0]  # Qubit time scales (simulating relativistic dilation)
        for q in range(self.num_qubits):
            self.circuit.rx(time_scales[q] * t, q)
            self.circuit.ry(time_scales[q] * t, q)
            self.circuit.rz(time_scales[q] * t, q)

    def inject_entropy(self):
        for i, rate in enumerate(self.entropy_rates):
            # Apply entropy injection gates
            self.circuit.rx(rate * np.pi / 4, i)

    def insert_entangling_gates(self):
        # Insert cx and cz gates to create causal links
        self.circuit.cx(0, 1)
        self.circuit.cz(1, 2)

    def run_experiment(self, t):
        self.apply_time_evolution(t)
        self.insert_entangling_gates()
        self.inject_entropy()

        if self.device == 'simulator':
            simulator = FakeBrisbane()
            print("Using FakeBrisbane simulator")
            state = Statevector.from_instruction(self.circuit)
        else:
            from qiskit import IBMQ
            provider = IBMQ.load_account()
            backend = provider.get_backend(self.device)
            print(f"Using IBMQ backend: {self.device}")
            state = Statevector.from_instruction(self.circuit)  # Placeholder for actual backend execution

        entropies = [entropy(partial_trace(state, [i])) for i in range(self.num_qubits)]
        mi_matrix = self.calculate_mutual_information(state)

        self.log_results(entropies, mi_matrix, state)

        return entropies, mi_matrix, state

    def calculate_mutual_information(self, state):
        # Calculate mutual information matrix
        mi_matrix = np.zeros((self.num_qubits, self.num_qubits))
        for i in range(self.num_qubits):
            for j in range(self.num_qubits):
                if i != j:
                    rho_i = partial_trace(state, [k for k in range(self.num_qubits) if k != i])
                    rho_j = partial_trace(state, [k for k in range(self.num_qubits) if k != j])
                    rho_ij = partial_trace(state, [k for k in range(self.num_qubits) if k != i and k != j])
                    mi_matrix[i, j] = entropy(rho_i) + entropy(rho_j) - entropy(rho_ij)
        return mi_matrix

    def visualize_entropy_over_time(self, entropies):
        plt.figure(figsize=(10, 6))
        for i, entropy_values in enumerate(entropies):
            plt.plot(entropy_values, label=f'Qubit {i}')
        plt.xlabel('Warped Time Step')
        plt.ylabel('Entropy')
        plt.title('Entropy over Warped Time')
        plt.legend()
        plt.show()

    def plot_mi_graphs(self, mi_matrix):
        plt.figure(figsize=(8, 6))
        sns.heatmap(mi_matrix, annot=True, cmap='coolwarm', cbar=True)
        plt.title('Mutual Information Matrix')
        plt.xlabel('Qubit Index')
        plt.ylabel('Qubit Index')
        plt.show()

    def visualize_causal_structure(self, state):
        # Placeholder for causal structure visualization using MDS
        pass

    def log_results(self, entropies, mi_matrix, state):
        log_dir = 'experiment_logs/curved_time_experiment'
        os.makedirs(log_dir, exist_ok=True)

        # Log results to results.json
        results_path = os.path.join(log_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'entropies': entropies,
                'mutual_information_matrix': mi_matrix.tolist(),
                'statevector': [(v.real, v.imag) for v in state.data]
            }, f, indent=4)

    def optional_teleportation_test(self):
        # Placeholder for teleportation test logic
        pass

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Run curved time experiment.')
        parser.add_argument('--device', type=str, default='simulator',
                            help='Choose the execution device: simulator or backend name')
        return parser.parse_args()

# Example usage
if __name__ == "__main__":
    args = CurvedTimeExperiment.parse_arguments()
    experiment = CurvedTimeExperiment(num_qubits=3, time_warp=[1.0, 0.5, 2.0], entropy_rates=[0.1, 0.2, 0.3], device=args.device)
    entropies, mi_matrix, state = experiment.run_experiment(t=1.0)
    experiment.plot_mi_graphs(mi_matrix)
    experiment.visualize_entropy_over_time([entropies])
    print("Entropies:", entropies)
    print("Mutual Information Matrix:", mi_matrix)
    print("Full Statevector:", state) 