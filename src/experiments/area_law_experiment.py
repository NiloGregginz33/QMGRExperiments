import numpy as np
from braket.circuits import Circuit
from braket.devices import LocalSimulator
import matplotlib.pyplot as plt

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

num_qubits = 4
entropies = []
cut_sizes = [1, 2, 3]

for cut in cut_sizes:
    circ = Circuit().h(0)
    for i in range(1, num_qubits):
        circ.cnot(0, i)
    circ.probability()
    task = device.run(circ, shots=2048)
    probs = np.array(task.result().values).reshape(-1)
    marg = marginal_probs(probs, num_qubits, list(range(cut)))
    S = shannon_entropy(marg)
    entropies.append(S)
    print(f"Cut size {cut}: Entropy = {S:.4f}")

plt.plot(cut_sizes, entropies, marker='o')
plt.xlabel('Subsystem size')
plt.ylabel('Entropy (bits)')
plt.title('Area Law Scaling')
plt.savefig('plots/area_law_experiment.png')
plt.show() 