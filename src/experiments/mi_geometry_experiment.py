import numpy as np
from braket.circuits import Circuit
from braket.devices import LocalSimulator
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

USE_LOCAL_SIM = True
if USE_LOCAL_SIM:
    device = LocalSimulator()
else:
    from braket.aws import AwsDevice
    arn = "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1"
    device = AwsDevice(arn)

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
circ = Circuit().h(0)
circ.cnot(0, 2)
circ.cnot(0, 3)
circ.rx(0, np.pi/4)
circ.cz(0, 1)
circ.cnot(1, 2)
circ.rx(2, np.pi/4)
circ.cz(1, 3)
circ.probability()
task = device.run(circ, shots=2048)
probs = np.array(task.result().values).reshape(-1)

mi_matrix = np.zeros((num_qubits, num_qubits))
for i in range(num_qubits):
    for j in range(i + 1, num_qubits):
        mi = compute_mi(probs, i, j, num_qubits)
        mi_matrix[i, j] = mi_matrix[j, i] = mi

epsilon = 1e-6
dist = 1 / (mi_matrix + epsilon)
np.fill_diagonal(dist, 0)

mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
coords = mds.fit_transform(dist)

plt.figure(figsize=(5, 5))
plt.scatter(coords[:, 0], coords[:, 1], c='blue')
for i in range(num_qubits):
    plt.text(coords[i, 0], coords[i, 1], f"Q{i}", fontsize=12)
plt.title('Emergent Geometry from MI')
plt.axis('equal')
plt.grid(True)
plt.savefig('plots/mi_geometry_experiment.png')
plt.show() 