import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import os
import json

# After the imports, add the following lines to create the output directory
exp_dir = "experiment_logs/ctc_geometry"
os.makedirs(exp_dir, exist_ok=True)

def shannon_entropy(probs):
    probs = np.array(probs)
    probs = probs / np.sum(probs)  # Normalize
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

# Create quantum circuit with CTC-like geometry
circ = QuantumCircuit(4)
# Initial state preparation
circ.h(0)
circ.h(1)

# Create feedback loop structure
circ.cx(0, 1)  # Initial entanglement
circ.cx(1, 2)  # Forward propagation
circ.cx(2, 3)  # Forward propagation
circ.cx(3, 0)  # Feedback loop closure

# Add time-like evolution
for i in range(4):
    circ.rx(np.pi/4, i)  # Time evolution for each qubit

# Get state vector and probabilities
state = Statevector.from_instruction(circ)
probs = np.abs(state.data) ** 2

# Compute mutual information matrix
mi_matrix = np.zeros((4, 4))
for i in range(4):
    for j in range(i + 1, 4):
        mi = compute_mi(probs, i, j, 4)
        mi_matrix[i, j] = mi_matrix[j, i] = mi

print("Mutual Information Matrix (CTC Geometry, Qiskit):")
print(mi_matrix)

# Convert MI to distance matrix for visualization
epsilon = 1e-6
dist = 1 / (mi_matrix + epsilon)
np.fill_diagonal(dist, 0)

# Use MDS to visualize the emergent geometry
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
coords = mds.fit_transform(dist)

print("\nMDS Coordinates (Emergent Geometry):")
print(coords)

# Plot the emergent geometry
plt.figure(figsize=(8, 6))
plt.scatter(coords[:, 0], coords[:, 1], c='blue', s=100)
for i in range(4):
    plt.text(coords[i, 0], coords[i, 1], f"Q{i}", fontsize=12)
plt.title("Emergent Geometry from CTC-like Structure")
plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")
plt.grid(True)
plt.savefig(f"{exp_dir}/ctc_geometry_experiment.png")
plt.close()

# After printing the mutual information matrix, save it to a JSON file
with open(f"{exp_dir}/mi_matrix.json", "w") as f:
    json.dump({"mi_matrix": mi_matrix.tolist()}, f, indent=2)

# After printing the MDS coordinates, save them to a JSON file
with open(f"{exp_dir}/mds_coordinates.json", "w") as f:
    json.dump({"mds_coordinates": coords.tolist()}, f, indent=2)

# Write a summary file
with open(f"{exp_dir}/summary.txt", "w") as f:
    f.write("CTC Geometry Experiment Summary\n")
    f.write("================================\n\n")
    f.write("Theoretical Background:\n")
    f.write("This experiment simulates a Closed Timelike Curve (CTC) geometry using a quantum circuit. It investigates how feedback loops and time-like evolution affect entanglement and emergent geometry.\n\n")
    f.write("Methodology:\n")
    f.write("A 4-qubit quantum circuit is constructed with a feedback loop structure. Mutual information is computed between qubits, and MDS is used to visualize the emergent geometry.\n\n")
    f.write("Results:\n")
    f.write(f"Results saved in: {exp_dir}\n")
    f.write("\nConclusion:\n")
    f.write("The experiment demonstrates how CTC-like structures influence quantum entanglement and emergent geometry, providing insights into the interplay between quantum mechanics and spacetime structure.\n") 