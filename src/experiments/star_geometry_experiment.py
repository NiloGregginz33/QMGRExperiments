import numpy as np
from braket.circuits import Circuit
from braket.devices import LocalSimulator
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from braket.registers import Qubit
import os
from datetime import datetime

def shannon_entropy(probs):
    probs = np.array(probs)
    probs = probs / np.sum(probs)
    return -np.sum(probs * np.log2(probs + 1e-12))

def marginal_probs(probs, total_qubits, keep):
    marginal = {}
    for idx, p in enumerate(probs):
        b = format(idx, f"0{total_qubits}b")
        key = ''.join([b[i] for i in keep])
        marginal[key] = marginal.get(key, 0) + p
    return np.array(list(marginal.values()))

def compute_mi(probs, qA, qB, total_qubits):
    AB = marginal_probs(probs, total_qubits, [qA, qB])
    A = marginal_probs(probs, total_qubits, [qA])
    B = marginal_probs(probs, total_qubits, [qB])
    return shannon_entropy(A) + shannon_entropy(B) - shannon_entropy(AB)

def run_star_geometry_experiment():
    # Create a quantum circuit
    circ = Circuit()
    
    # Apply Hadamard gates to create superposition
    circ.h(0)
    circ.h(1)
    circ.h(2)
    circ.h(3)
    
    # Apply controlled rotations to create star geometry
    circ.rx(0, np.pi/4)  # First qubit rotation
    circ.rx(1, np.pi/4)  # Second qubit rotation
    circ.rx(2, np.pi/4)  # Third qubit rotation
    circ.rx(3, np.pi/4)  # Fourth qubit rotation
    
    # Apply CNOT gates to create entanglement
    circ.cnot(0, 1)
    circ.cnot(1, 2)
    circ.cnot(2, 3)
    circ.cnot(3, 0)
    
    # Create a local simulator
    device = LocalSimulator()
    
    # Run the circuit
    result = device.run(circ, shots=1000).result()
    
    # Get measurement counts
    counts = result.measurement_counts
    
    # Create timestamp for unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder = f"experiment_logs/star_geometry_experiment_{timestamp}"
    os.makedirs(experiment_folder, exist_ok=True)
    
    # Save results to a log file
    log_file = os.path.join(experiment_folder, "star_geometry_experiment_log.txt")
    with open(log_file, "w", encoding='utf-8') as f:
        f.write("Star Geometry Experiment Results\n")
        f.write("==============================\n\n")
        f.write("Circuit:\n")
        f.write(str(circ))
        f.write("\n\nMeasurement Results:\n")
        for state, count in counts.items():
            f.write(f"{state}: {count}\n")
    
    # Create and save visualization
    plt.figure(figsize=(10, 6))
    plt.bar(counts.keys(), counts.values())
    plt.title("Star Geometry Experiment Results")
    plt.xlabel("State")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(experiment_folder, "star_geometry_results.png")
    plt.savefig(plot_file)
    plt.close()
    
    return counts

if __name__ == "__main__":
    run_star_geometry_experiment() 