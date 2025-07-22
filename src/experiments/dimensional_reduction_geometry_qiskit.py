import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeJakartaV2
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
import os
import json
import argparse
from datetime import datetime

def build_geometry_circuit(n_qubits, phi):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(0)
    for i in range(1, n_qubits):
        qc.cx(0, i)
    for i in range(n_qubits):
        qc.rx(phi, i)
    for i in range(1, n_qubits):
        qc.cx(i-1, i)
        qc.rz(np.pi, i)
        qc.cx(i-1, i)
    qc.measure(range(n_qubits), range(n_qubits))
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

def mds_embedding(mi_matrix, n_qubits):
    dist = np.exp(-mi_matrix)
    np.fill_diagonal(dist, 0)
    n_components = min(n_qubits, 4)
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(dist)
    return coords, mds.stress_, mds.embedding_, mds

def main():
    parser = argparse.ArgumentParser(description="Dimensional Reduction Geometry Experiment (Qiskit)")
    parser.add_argument('--device', type=str, default='simulator', help='Device: "simulator" or IBMQ backend name')
    parser.add_argument('--shots', type=int, default=1024, help='Number of shots')
    parser.add_argument('--min_qubits', type=int, default=3, help='Minimum boundary qubits')
    parser.add_argument('--max_qubits', type=int, default=7, help='Maximum boundary qubits')
    parser.add_argument('--phi', type=float, default=np.pi/4, help='Geometry parameter phi')
    args = parser.parse_args()

    if args.device.lower() == 'simulator':
        backend = FakeJakartaV2()
        # For simulator, use direct execution
        use_sampler = False
    else:
        service = QiskitRuntimeService()
        backend = service.backend(args.device)
        use_sampler = True

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('experiment_logs', f'dimensional_reduction_geometry_qiskit_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)

    results = []
    for n_qubits in range(args.min_qubits, args.max_qubits+1):
        print(f"Running circuit for {n_qubits} qubits...")
        qc = build_geometry_circuit(n_qubits, args.phi)
        tqc = transpile(qc, backend, optimization_level=0)
        
        if use_sampler:
            # Use modern Sampler primitive
            sampler = Sampler(backend)
            job = sampler.run([tqc], shots=args.shots)
            result = job.result()
            counts = result.quasi_dists[0]
            # Convert quasi_dists to counts format
            counts_dict = {}
            for bitstring, probability in counts.items():
                count = int(probability * args.shots)
                if count > 0:
                    counts_dict[format(bitstring, f'0{n_qubits}b')] = count
        else:
            # For simulator, use direct execution
            job = backend.run(tqc, shots=args.shots)
            result = job.result()
            counts_dict = result.get_counts()
        
        mi_matrix = compute_mi_matrix(counts_dict, n_qubits, args.shots)
        coords, stress, embedding, mds = mds_embedding(mi_matrix, n_qubits)
        # Eigenvalues of the Gram matrix (covariance of embedding)
        gram = np.cov(coords.T)
        eigvals = np.sort(np.linalg.eigvalsh(gram))[::-1]
        # Bulk volume: sum of pairwise distances
        dists = np.sqrt(((coords[:,None,:] - coords[None,:,:])**2).sum(-1))
        bulk_volume = np.sum(dists) / 2
        metrics = {
            "n_qubits": n_qubits,
            "mi_matrix": mi_matrix.tolist(),
            "mds_coords": coords.tolist(),
            "mds_eigenvalues": eigvals.tolist(),
            "bulk_volume": float(bulk_volume),
            "stress": float(stress)
        }
        results.append(metrics)
        # Plot geometry snapshot
        plt.figure()
        plt.scatter(coords[:,0], coords[:,1], c='b')
        for j in range(coords.shape[0]):
            x, y = coords[j, 0], coords[j, 1]
            plt.text(x, y, str(j), fontsize=12, ha='center', va='center', color='white', bbox=dict(facecolor='blue', alpha=0.5, boxstyle='circle'))
        plt.title(f'MDS Geometry (n_qubits={n_qubits})')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f'mds_geometry_{n_qubits}q.png'))
        plt.close()

    # Save results
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Write summary
    with open(os.path.join(log_dir, 'summary.txt'), 'w') as f:
        f.write(f"Dimensional Reduction Geometry Experiment\n")
        f.write(f"Backend: {backend}\n")
        f.write(f"Phi: {args.phi}\n")
        f.write(f"Boundary sizes: {list(range(args.min_qubits, args.max_qubits+1))}\n")
        f.write(f"Results: See results.json and geometry plots.\n")

    # Plot eigenvalue spectrum vs boundary size
    plt.figure()
    for r in results:
        plt.plot(range(1, len(r['mds_eigenvalues'])+1), r['mds_eigenvalues'], marker='o', label=f"{r['n_qubits']}q")
    plt.xlabel('MDS Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.title('MDS Eigenvalue Spectrum vs Boundary Size')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'mds_eigenvalue_spectrum.png'))
    plt.close()

    # Plot bulk volume vs boundary size
    plt.figure()
    plt.plot([r['n_qubits'] for r in results], [r['bulk_volume'] for r in results], marker='o')
    plt.xlabel('Boundary Qubits')
    plt.ylabel('Bulk Volume (sum of pairwise distances)')
    plt.title('Bulk Volume vs Boundary Size')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'bulk_volume_vs_boundary.png'))
    plt.close()

    print(f"Dimensional Reduction Geometry experiment complete. Results in {log_dir}")

if __name__ == "__main__":
    main() 