import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import argparse
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService
from sklearn.manifold import MDS
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from src.CGPTFactory import run as cgpt_run
except ImportError:
    cgpt_run = None

def shannon_entropy(probs):
    """
    Compute the Shannon entropy of a probability distribution.
    """
    probs = np.array(probs)
    probs = probs / np.sum(probs)
    return -np.sum(probs * np.log2(probs + 1e-12))

def marginal_probs(counts, total_qubits, target_idxs, shots):
    """
    Compute marginal probabilities for a subset of qubits from measurement counts.
    """
    marginal = {}
    for bitstring, count in counts.items():
        b = bitstring.zfill(total_qubits)
        key = ''.join([b[-(i+1)] for i in target_idxs])
        marginal[key] = marginal.get(key, 0) + count
    probs = np.array(list(marginal.values())) / shots
    return probs

def compute_mi_counts(counts, qA, qB, total_qubits, shots):
    """
    Compute the mutual information between two qubits from measurement counts.
    """
    AB = marginal_probs(counts, total_qubits, [qA, qB], shots)
    A = marginal_probs(counts, total_qubits, [qA], shots)
    B = marginal_probs(counts, total_qubits, [qB], shots)
    return shannon_entropy(A) + shannon_entropy(B) - shannon_entropy(AB)

def build_ctc_circuit(phi, perturbation='none', target=1, collapse_test=False, nloops=1):
    """
    Build a 4-qubit CTC circuit with optional perturbation (X, RX(pi), or CX(0,target)), repeated for nloops.
    If collapse_test is True, use CX(0, target) as the perturbation.
    Perturbation is applied at every loop if specified.
    """
    circ = QuantumCircuit(4)
    circ.h(0)
    circ.h(1)
    for _ in range(nloops):
        circ.cx(0, 1)
        circ.cx(1, 2)
        circ.cx(2, 3)
        circ.cx(3, 0)
        if perturbation == 'x':
            circ.x(target)
        elif perturbation == 'rx':
            circ.rx(np.pi, target)
        elif perturbation == 'cx' or collapse_test:
            circ.cx(0, target)
    circ.measure_all()
    return circ

def run_ctc_conditional_perturbation_experiment(device_name=None, shots=1024, phi_list=None, perturbation='none', target=1, collapse_test=False, nloops=1):
    """
    For each phi, run both standard and perturbed CTC circuits, saving paired results.
    nloops: number of times to repeat the CTC loop structure (perturbation applied at every loop for perturbed case).
    """
    if phi_list is None:
        phi_list = np.linspace(0, 2 * np.pi, 8)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = f"experiment_logs/ctc_conditional_perturbation_qiskit_{device_name or 'statevector'}_{timestamp}_nloops{nloops}"
    os.makedirs(exp_dir, exist_ok=True)
    results = {'standard': [], 'perturbed': []}
    for phi in phi_list:
        # Standard CTC (no perturbation)
        circ_std = build_ctc_circuit(phi, perturbation='none', nloops=nloops)
        # Perturbed CTC (perturbation at every loop)
        circ_pert = build_ctc_circuit(phi, perturbation=perturbation, target=target, collapse_test=collapse_test, nloops=nloops)
        for label, circ in [('standard', circ_std), ('perturbed', circ_pert)]:
            if device_name is not None:
                service = QiskitRuntimeService()
                backend = service.backend(device_name)
                tqc = transpile(circ, backend, optimization_level=3)
                if cgpt_run is not None:
                    counts = cgpt_run(tqc, backend=backend, shots=shots)
                else:
                    from qiskit_ibm_runtime import SamplerV2 as Sampler, Session
                    with Session(service=service, backend=backend) as session:
                        sampler = Sampler(session=session)
                        job = sampler.run(tqc, shots=shots)
                        result = job.result()
                        counts = result.quasi_dists[0] if hasattr(result, 'quasi_dists') else result.quasi_distribution[0]
                counts = {k: int(v) for k, v in counts.items()}
                mi_matrix = np.zeros((4, 4))
                for i in range(4):
                    for j in range(i + 1, 4):
                        mi = compute_mi_counts(counts, i, j, 4, shots)
                        mi_matrix[i, j] = mi_matrix[j, i] = mi
            else:
                circ_no_meas = QuantumCircuit(4)
                circ_no_meas.h(0)
                circ_no_meas.h(1)
                circ_no_meas.cx(0, 1)
                circ_no_meas.cx(1, 2)
                circ_no_meas.cx(2, 3)
                circ_no_meas.cx(3, 0)
                if label == 'perturbed':
                    if perturbation == 'x':
                        circ_no_meas.x(target)
                    elif perturbation == 'rx':
                        circ_no_meas.rx(np.pi, target)
                    elif perturbation == 'cx' or collapse_test:
                        circ_no_meas.cx(0, target)
                state = Statevector.from_instruction(circ_no_meas)
                probs = np.abs(state.data) ** 2
                shots_sim = 8192
                counts = {}
                for idx, p in enumerate(probs):
                    if p > 0:
                        b = format(idx, "04b")
                        counts[b] = int(round(p * shots_sim))
                mi_matrix = np.zeros((4, 4))
                for i in range(4):
                    for j in range(i + 1, 4):
                        mi = compute_mi_counts(counts, i, j, 4, shots_sim)
                        mi_matrix[i, j] = mi_matrix[j, i] = mi
            epsilon = 1e-6
            dist = 1 / (mi_matrix + epsilon)
            np.fill_diagonal(dist, 0)
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            coords = mds.fit_transform(dist)
            entropy = shannon_entropy(np.array(list(counts.values())) / sum(counts.values()))
            results[label].append({
                'phi': float(phi),
                'mi_matrix': mi_matrix.tolist(),
                'mds_coords': coords.tolist(),
                'entropy': float(entropy)
            })
            plt.figure(figsize=(8, 6))
            plt.scatter(coords[:, 0], coords[:, 1], c='blue', s=100)
            for i in range(4):
                plt.text(coords[i, 0], coords[i, 1], f"Q{i}", fontsize=12)
            plt.title(f"CTC {'Perturbed' if label=='perturbed' else 'Standard'} (phi={phi:.2f})")
            plt.xlabel("MDS Dimension 1")
            plt.ylabel("MDS Dimension 2")
            plt.grid(True)
            plt.savefig(f"{exp_dir}/ctc_{label}_phi_{phi:.2f}.png")
            plt.close()
    with open(f"{exp_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{exp_dir}/summary.txt", "w") as f:
        f.write("CTC Conditional Perturbation Experiment Summary\n")
        f.write("==============================================\n\n")
        f.write(f"Device: {device_name or 'statevector'}\n")
        f.write(f"Shots: {shots}\n\n")
        f.write("Theoretical Background:\n")
        f.write("This experiment simulates a Closed Timelike Curve (CTC) geometry with and without a conditional or unconditional perturbation. It investigates how anomalies or interventions affect the entanglement and emergent geometry of the CTC loop.\n\n")
        f.write("Methodology:\n")
        f.write("A 4-qubit quantum circuit is constructed with a feedback loop structure. A perturbation (X, RX(pi), or CX(0,target)) is optionally applied. Mutual information, MDS geometry, and entropy are computed for both standard and perturbed cases.\n\n")
        f.write("Results:\n")
        f.write(f"Results saved in: {exp_dir}\n")
        f.write("\nConclusion:\n")
        f.write("The experiment demonstrates how local interventions in a CTC-like structure can alter the entanglement and emergent geometry, and tests the loop's sensitivity to collapse and self-consistency.\n")
    print(f"Experiment completed. Results saved in {exp_dir}")
    return exp_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CTC conditional perturbation experiment (Qiskit)')
    parser.add_argument('--device', type=str, default=None, help='IBM device name (or None for statevector sim)')
    parser.add_argument('--shots', type=int, default=1024, help='Number of shots (if using sampler)')
    parser.add_argument('--nphi', type=int, default=8, help='Number of phi values to sweep')
    parser.add_argument('--perturbation', type=str, default='none', choices=['none', 'x', 'rx', 'cx'], help='Type of perturbation: none, x, rx, cx')
    parser.add_argument('--target', type=int, default=1, help='Qubit index to perturb (default 1)')
    parser.add_argument('--collapse_test', action='store_true', help='If set, use CX(0,target) as the perturbation (collapse sensitivity test)')
    parser.add_argument('--nloops', type=int, default=1, help='Number of CTC loops (perturbation applied at every loop for perturbed case)')
    args = parser.parse_args()
    phi_list = np.linspace(0, 2 * np.pi, args.nphi)
    run_ctc_conditional_perturbation_experiment(device_name=args.device, shots=args.shots, phi_list=phi_list, perturbation=args.perturbation, target=args.target, collapse_test=args.collapse_test, nloops=args.nloops) 