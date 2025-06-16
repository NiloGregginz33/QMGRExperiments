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
timesteps = np.linspace(0, 3 * np.pi, 30)
entropies = []

for phi_val in timesteps:
    circ = Circuit().h(0)
    circ.cnot(0, 2)
    circ.cnot(0, 3)
    circ.rx(0, phi_val)
    circ.cz(0, 1)
    circ.cnot(1, 2)
    circ.rx(2, phi_val)
    circ.cz(1, 3)
    circ.probability()
    task = device.run(circ, shots=2048)
    probs = np.array(task.result().values).reshape(-1)
    marg = marginal_probs(probs, num_qubits, [2, 3])
    S = shannon_entropy(marg)
    entropies.append(S)
    print(f"Phase {phi_val:.2f}: Entropy = {S:.4f}")

plt.plot(timesteps, entropies, marker='o')
plt.xlabel('Evaporation Phase φ(t)')
plt.ylabel('Entropy (bits)')
plt.title('Page Curve Simulation')
plt.savefig('plots/page_curve_experiment.png')
plt.show() 