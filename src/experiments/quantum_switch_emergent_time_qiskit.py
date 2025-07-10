import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeJakartaV2
from qiskit_ibm_runtime import QiskitRuntimeService
import os
import json
import argparse

def log_result(metrics, log_dir="experiment_logs/quantum_switch_emergent_time_qiskit"):
    os.makedirs(log_dir, exist_ok=True)
    idx = len([f for f in os.listdir(log_dir) if f.startswith('result_') and f.endswith('.json')]) + 1
    with open(os.path.join(log_dir, f"result_{idx}.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

def quantum_switch_circuit(phi=0):
    # 2 qubits: 0 = control, 1 = target
    qc = QuantumCircuit(2, 2)
    # Prepare control in superposition
    qc.h(0)
    # Prepare target in |0>
    # Define A and B as single-qubit rotations
    # A: RX(phi), B: RY(phi)
    # Controlled order: if control=0, do A then B; if control=1, do B then A
    # Use ancilla for order control (decomposed for 2 qubits)
    # Apply A then B
    qc.cx(0, 1)
    qc.rx(phi, 1)
    qc.cry(phi, 0, 1)
    # Undo control
    qc.cx(0, 1)
    # Measure
    qc.measure([0, 1], [0, 1])
    return qc

def run_quantum_switch_experiment(backend=None, shots=2048, phi_values=None):
    if phi_values is None:
        phi_values = np.linspace(0, 2 * np.pi, 10)
    log_dir = "experiment_logs/quantum_switch_emergent_time_qiskit"
    os.makedirs(log_dir, exist_ok=True)
    results = []
    for phi in phi_values:
        qc = quantum_switch_circuit(phi)
        tqc = transpile(qc, backend)
        job = backend.run(tqc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        # Compute probabilities
        probs = np.zeros(4)
        for bitstring, count in counts.items():
            idx = int(bitstring.replace(' ', ''), 2)
            probs[idx] = count / shots
        # Entropy of control-target
        shannon_entropy = -np.sum(probs * np.log2(probs + 1e-12))
        # Advanced causal non-separability witness (Branciard et al., PRL 2016)
        # P(control, target):
        # Bitstring order: '00' = 0, '01' = 1, '10' = 2, '11' = 3
        P_00 = probs[0]  # control=0, target=0
        P_01 = probs[1]  # control=0, target=1
        P_10 = probs[2]  # control=1, target=0
        P_11 = probs[3]  # control=1, target=1
        # Witness: W = P_00 + P_11 - P_01 - P_10
        causal_witness = P_00 + P_11 - P_01 - P_10
        metrics = {
            "phi": phi,
            "counts": counts,
            "shannon_entropy": shannon_entropy,
            "causal_witness": causal_witness
        }
        log_result(metrics, log_dir)
        results.append(metrics)
    # Save all results
    with open(os.path.join(log_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    # Write summary
    with open(os.path.join(log_dir, "summary.txt"), 'w') as f:
        f.write(f"Quantum Switch Emergent Time Experiment\n")
        f.write(f"Backend: {backend}\n")
        f.write(f"Shots: {shots}\n")
        f.write(f"Phi values: {list(phi_values)}\n\n")
        for r in results:
            f.write(f"phi={r['phi']:.2f}, Shannon Entropy={r['shannon_entropy']:.4f}, Causal Witness={r['causal_witness']:.4f}\n")
    # Plot entropy vs phi
    plt.figure()
    phis = [r['phi'] for r in results]
    entropies = [r['shannon_entropy'] for r in results]
    plt.plot(phis, entropies, marker='o')
    plt.xlabel('Parameter phi')
    plt.ylabel('Shannon Entropy')
    plt.title('Entropy vs phi (Quantum Switch)')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'shannon_entropy.png'))
    plt.close()
    # Plot causal witness vs phi
    plt.figure()
    witnesses = [r['causal_witness'] for r in results]
    plt.plot(phis, witnesses, marker='o', color='red')
    plt.xlabel('Parameter phi')
    plt.ylabel('Causal Witness')
    plt.title('Causal Non-Separability Witness vs phi')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'causal_witness.png'))
    plt.close()
    print(f"Quantum Switch Emergent Time experiment complete. Results in {log_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Switch Emergent Time Experiment")
    parser.add_argument('--device', type=str, default='simulator', help='Device to use: "simulator" or IBMQ backend name')
    parser.add_argument('--shots', type=int, default=2048, help='Number of shots')
    args = parser.parse_args()
    if args.device.lower() == 'simulator':
        backend = FakeJakartaV2()
    else:
        service = QiskitRuntimeService()
        backend = service.backend(args.device)
    run_quantum_switch_experiment(backend=backend, shots=args.shots) 