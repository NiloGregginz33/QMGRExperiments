import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

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

num_qubits = 4
circ = QuantumCircuit(num_qubits)
circ.h(0)
circ.cx(0, 1)
circ.cx(1, 2)
circ.cx(2, 3)
circ.cx(3, 0)  # Close the ring
circ.rx(np.pi/4, 0)
circ.rx(np.pi/4, 1)
circ.rx(np.pi/4, 2)
circ.rx(np.pi/4, 3)

# Transpile the circuit
from qiskit.providers.basic_provider import BasicSimulator
backend = BasicSimulator()
transpiled_circ = transpile(circ, backend)

state = Statevector.from_instruction(transpiled_circ)
probs = np.abs(state.data) ** 2

mi_matrix = np.zeros((num_qubits, num_qubits))
for i in range(num_qubits):
    for j in range(i + 1, num_qubits):
        mi = compute_mi(probs, i, j, num_qubits)
        mi_matrix[i, j] = mi_matrix[j, i] = mi

print("Mutual Information Matrix (Ring Geometry, Qiskit):\n", mi_matrix)

epsilon = 1e-6
dist = 1 / (mi_matrix + epsilon)
np.fill_diagonal(dist, 0)

mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
coords = mds.fit_transform(dist)

print("MDS Coordinates (Emergent Geometry):\n", coords)

plt.figure(figsize=(5, 5))
plt.scatter(coords[:, 0], coords[:, 1], c='blue')
for i in range(num_qubits):
    plt.text(coords[i, 0], coords[i, 1], f"Q{i}", fontsize=12)
plt.title('Ring Geometry from MI (Qiskit)')
plt.axis('equal')
plt.grid(True)
plt.savefig('plots/ring_geometry_experiment_qiskit.png')
plt.show() 