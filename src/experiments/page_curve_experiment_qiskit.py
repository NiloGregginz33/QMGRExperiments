"""
PROPRIETARY SOFTWARE - PAGE CURVE EXPERIMENT

Copyright (c) 2024-2025 Matrix Solutions LLC. All rights reserved.

This file contains proprietary research algorithms and experimental protocols
for quantum holographic geometry experiments. This software is proprietary and
confidential to Matrix Solutions LLC.

SPECIFIC LICENSE TERMS:
- Use for academic peer review purposes is permitted
- Academic research and educational use is allowed with proper attribution
- Commercial use is strictly prohibited without written permission
- Redistribution, modification, or derivative works are not permitted
- Reverse engineering or decompilation is prohibited

For licensing inquiries: manavnaik123@gmail.com

By using this file, you acknowledge and agree to be bound by these terms.
"""

import numpy as np
from qiskit import QuantumCircuit, execute, Aer
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
timesteps = np.linspace(0, 3 * np.pi, 30)
entropies = []

for phi_val in timesteps:
    circ = QuantumCircuit(num_qubits)
    circ.h(0)
    circ.cx(0, 2)
    circ.cx(0, 3)
    circ.rx(phi_val, 0)
    circ.cz(0, 1)
    circ.cx(1, 2)
    circ.rx(phi_val, 2)
    circ.cz(1, 3)
    circ.measure_all()
    backend = Aer.get_backend('qasm_simulator')
    job = execute(circ, backend, shots=2048)
    result = job.result()
    counts = result.get_counts()
    probs = np.array([counts.get(format(i, f"0{num_qubits}b"), 0) for i in range(2**num_qubits)]) / 2048
    marg = marginal_probs(probs, num_qubits, [2, 3])
    S = shannon_entropy(marg)
    entropies.append(S)
    print(f"Phase {phi_val:.2f}: Entropy = {S:.4f}")

plt.plot(timesteps, entropies, marker='o')
plt.xlabel('Evaporation Phase φ(t)')
plt.ylabel('Entropy (bits)')
plt.title('Page Curve Simulation (Qiskit)')
plt.savefig('plots/page_curve_experiment_qiskit.png')
plt.show() 