import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeJakartaV2
from qiskit_ibm_runtime import QiskitRuntimeService
import os
import json
import argparse
from datetime import datetime

# --- Quantum Switch Subcircuit (2 qubits) ---
def quantum_switch_circuit(phi=0):
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rx(phi, 1)
    qc.cry(phi, 0, 1)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc

def quantum_switch_metrics(counts, shots):
    probs = np.zeros(4)
    for bitstring, count in counts.items():
        idx = int(bitstring.replace(' ', ''), 2)
        probs[idx] = count / shots
    shannon_entropy = -np.sum(probs * np.log2(probs + 1e-12))
    P_00, P_01, P_10, P_11 = probs[0], probs[1], probs[2], probs[3]
    causal_witness = P_00 + P_11 - P_01 - P_10
    return shannon_entropy, causal_witness, probs

# --- Emergent Spacetime Subcircuit (4 qubits) ---
def build_spacetime_circuit(phi):
    qc = QuantumCircuit(4, 4)
    qc.h(0)
    qc.cx(0, 2)
    qc.cx(0, 3)
    qc.rx(phi, 0)
    qc.cx(0, 1)
    qc.rz(np.pi, 1)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.rx(phi, 2)
    qc.cx(1, 3)
    qc.rz(np.pi, 3)
    qc.cx(1, 3)
    qc.measure([0,1,2,3], [0,1,2,3])
    return qc

def shannon_entropy(probs):
    probs = np.array(probs)
    return -np.sum(probs * np.log2(probs + 1e-12))

def marginal_probs(counts, total_qubits, target_idxs, shots):
    marginal = {}
    for bitstring, count in counts.items():
        b = bitstring[::-1]
        key = ''.join([b[i] for i in target_idxs])
        marginal[key] = marginal.get(key, 0) + count
    probs = np.array(list(marginal.values())) / shots
    return probs

def compute_mi(counts, qA, qB, total_qubits, shots):
    AB = marginal_probs(counts, total_qubits, [qA, qB], shots)
    A = marginal_probs(counts, total_qubits, [qA], shots)
    B = marginal_probs(counts, total_qubits, [qB], shots)
    return shannon_entropy(A) + shannon_entropy(B) - shannon_entropy(AB)

def compute_mi_matrix(counts, total_qubits, shots):
    mi_matrix = np.zeros((total_qubits, total_qubits))
    for a in range(total_qubits):
        for b in range(a+1, total_qubits):
            mi = compute_mi(counts, a, b, total_qubits, shots)
            mi_matrix[a, b] = mi_matrix[b, a] = mi
    return mi_matrix

def mds_embedding(mi_matrix):
    dist = np.exp(-mi_matrix)
    np.fill_diagonal(dist, 0)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(dist)
    return coords

def main():
    parser = argparse.ArgumentParser(description="Unified Causal Geometry Experiment (Qiskit)")
    parser.add_argument('--device', type=str, default='simulator', help='Device: "simulator" or IBMQ backend name')
    parser.add_argument('--shots', type=int, default=2048, help='Number of shots')
    parser.add_argument('--nsteps', type=int, default=10, help='Number of phi steps')
    args = parser.parse_args()

    if args.device.lower() == 'simulator':
        backend = FakeJakartaV2()
    else:
        service = QiskitRuntimeService()
        backend = service.backend(args.device)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('experiment_logs', f'unified_causal_geometry_qiskit_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)

    phi_values = np.linspace(0, 2 * np.pi, args.nsteps)
    results = []

    for phi in phi_values:
        # --- Quantum Switch ---
        qs_qc = quantum_switch_circuit(phi)
        qs_tqc = transpile(qs_qc, backend)
        qs_job = backend.run(qs_tqc, shots=args.shots)
        qs_result = qs_job.result()
        qs_counts = qs_result.get_counts()
        qs_entropy, qs_witness, qs_probs = quantum_switch_metrics(qs_counts, args.shots)

        # --- Emergent Spacetime ---
        st_qc = build_spacetime_circuit(phi)
        st_tqc = transpile(st_qc, backend)
        st_job = backend.run(st_tqc, shots=args.shots)
        st_result = st_job.result()
        st_counts = st_result.get_counts()
        st_entropy = shannon_entropy(np.array(list(st_counts.values())) / args.shots)
        mi_matrix = compute_mi_matrix(st_counts, 4, args.shots)
        mds_coords = mds_embedding(mi_matrix)

        metrics = {
            "phi": float(phi),
            "quantum_switch": {
                "shannon_entropy": float(qs_entropy),
                "causal_witness": float(qs_witness),
                "counts": qs_counts
            },
            "spacetime": {
                "shannon_entropy": float(st_entropy),
                "counts": st_counts,
                "mi_matrix": mi_matrix.tolist(),
                "mds_coords": mds_coords.tolist()
            }
        }
        results.append(metrics)

    # Save results
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Write summary
    with open(os.path.join(log_dir, 'summary.txt'), 'w') as f:
        f.write(f"Unified Causal Geometry Experiment\n")
        f.write(f"Backend: {backend}\n")
        f.write(f"Shots: {args.shots}\n")
        f.write(f"Phi values: {list(phi_values)}\n\n")
        for r in results:
            f.write(f"phi={r['phi']:.2f}, QS Entropy={r['quantum_switch']['shannon_entropy']:.4f}, "
                    f"QS Witness={r['quantum_switch']['causal_witness']:.4f}, "
                    f"ST Entropy={r['spacetime']['shannon_entropy']:.4f}\n")

    # Plot causal witness vs phi
    plt.figure()
    phis = [r['phi'] for r in results]
    witnesses = [r['quantum_switch']['causal_witness'] for r in results]
    plt.plot(phis, witnesses, marker='o', color='red')
    plt.xlabel('Parameter phi')
    plt.ylabel('Causal Witness')
    plt.title('Causal Non-Separability Witness vs phi')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'causal_witness.png'))
    plt.close()

    # Plot entropy vs phi
    plt.figure()
    qs_entropies = [r['quantum_switch']['shannon_entropy'] for r in results]
    st_entropies = [r['spacetime']['shannon_entropy'] for r in results]
    plt.plot(phis, qs_entropies, marker='o', label='Quantum Switch')
    plt.plot(phis, st_entropies, marker='s', label='Spacetime')
    plt.xlabel('Parameter phi')
    plt.ylabel('Shannon Entropy')
    plt.title('Entropy vs phi')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'shannon_entropy.png'))
    plt.close()

    # Plot geometry snapshots (MDS)
    for i, r in enumerate(results):
        coords = np.array(r['spacetime']['mds_coords'])
        plt.figure()
        plt.scatter(coords[:,0], coords[:,1], c='b')
        for j, (x, y) in enumerate(coords):
            plt.text(x, y, str(j), fontsize=12, ha='center', va='center', color='white', bbox=dict(facecolor='blue', alpha=0.5, boxstyle='circle'))
        plt.title(f'MDS Geometry (phi={r["phi"]:.2f})')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f'mds_geometry_{i+1:02d}.png'))
        plt.close()

    # Plot correlation between witness and average MI/entropy
    avg_mi = [np.nanmean(r['spacetime']['mi_matrix']) for r in results]
    plt.figure()
    plt.scatter(witnesses, avg_mi)
    plt.xlabel('Causal Witness')
    plt.ylabel('Average MI (Spacetime)')
    plt.title('Correlation: Witness vs Avg MI')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'witness_vs_avg_mi.png'))
    plt.close()

    plt.figure()
    plt.scatter(witnesses, st_entropies)
    plt.xlabel('Causal Witness')
    plt.ylabel('Spacetime Entropy')
    plt.title('Correlation: Witness vs Spacetime Entropy')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'witness_vs_st_entropy.png'))
    plt.close()

    print(f"Unified Causal Geometry experiment complete. Results in {log_dir}")

if __name__ == "__main__":
    main() 