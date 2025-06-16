import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt

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

num_qubits = 4
entropies = []
cut_sizes = [1, 2, 3]

for cut in cut_sizes:
    circ = QuantumCircuit(num_qubits)
    circ.h(0)
    for i in range(1, num_qubits):
        circ.cx(0, i)
    state = Statevector.from_instruction(circ)
    probs = np.abs(state.data) ** 2
    marg = marginal_probs(probs, num_qubits, list(range(cut)))
    S = shannon_entropy(marg)
    entropies.append(S)
    print(f"Cut size {cut}: Entropy = {S:.4f}")

plt.plot(cut_sizes, entropies, marker='o')
plt.xlabel('Subsystem size')
plt.ylabel('Entropy (bits)')
plt.title('Area Law Scaling (Qiskit)')
plt.savefig('plots/area_law_experiment_qiskit.png')
plt.show() 