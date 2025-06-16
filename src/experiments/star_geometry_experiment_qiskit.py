import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

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

# Create quantum circuit with star geometry
circ = QuantumCircuit(4)
# Central qubit (0) entangled with all others
circ.h(0)  # Prepare central qubit in superposition
for i in range(1, 4):
    circ.cx(0, i)  # Entangle central qubit with each peripheral qubit
    circ.rx(np.pi/4, i)  # Add some rotation to break symmetry

# Get state vector and probabilities
state = Statevector.from_instruction(circ)
probs = np.abs(state.data) ** 2

# Compute mutual information matrix
mi_matrix = np.zeros((4, 4))
for i in range(4):
    for j in range(i + 1, 4):
        mi = compute_mi(probs, i, j, 4)
        mi_matrix[i, j] = mi_matrix[j, i] = mi

print("Mutual Information Matrix (Star Geometry, Qiskit):")
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
plt.title("Emergent Geometry from Star Entanglement")
plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")
plt.grid(True)
plt.savefig('plots/star_geometry_experiment.png')
plt.close() 