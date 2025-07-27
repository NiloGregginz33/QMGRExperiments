import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from qiskit import QuantumCircuit
from itertools import combinations
import argparse
from datetime import datetime
import random
# Add Factory to sys.path and import run
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from CGPTFactory import run
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from sklearn.manifold import MDS

def build_circuit_with_charge(num_qubits, depth, charge):
    qc = QuantumCircuit(num_qubits)
    # Inject charge: set first Q qubits to |1>, rest to |0>
    for i in range(charge):
        qc.x(i)
    # Entangling layer
    for d in range(depth):
        for i in range(num_qubits-1):
            qc.cx(i, i+1)
        for i in range(num_qubits):
            qc.ry(np.pi/4, i)
    qc.measure_all()
    return qc

def compute_mi_matrix(counts, num_qubits):
    shots = sum(counts.values())
    probs = np.zeros(2**num_qubits)
    for bitstr, c in counts.items():
        idx = int(bitstr.replace(' ', ''), 2)
        probs[idx] = c / shots
    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log2(p))
    mi_matrix = np.zeros((num_qubits, num_qubits))
    for i, j in combinations(range(num_qubits), 2):
        marg_ij = np.zeros(4)
        marg_i = np.zeros(2)
        marg_j = np.zeros(2)
        for idx, p in enumerate(probs):
            b = format(idx, f'0{num_qubits}b')
            marg_ij[int(b[i]+b[j], 2)] += p
            marg_i[int(b[i])] += p
            marg_j[int(b[j])] += p
        mi = entropy(marg_i) + entropy(marg_j) - entropy(marg_ij)
        mi_matrix[i, j] = mi_matrix[j, i] = mi
    return mi_matrix

def safe_acos(x):
    return np.arccos(np.clip(x, -1.0, 1.0))

def mi_to_distance(mi):
    return -np.log(mi + 1e-6)

def estimate_curvature(mi_matrix):
    # Use average triangle defect as curvature estimator (with log-distance)
    n = mi_matrix.shape[0]
    defects = []
    for i, j, k in combinations(range(n), 3):
        a = mi_to_distance(mi_matrix[i, j])
        b = mi_to_distance(mi_matrix[j, k])
        c = mi_to_distance(mi_matrix[i, k])
        # Skip triangles with any infinite or NaN side
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
            continue
        # Law of cosines for triangle angles, clamp argument to [-1, 1] using safe_acos
        try:
            cosA = (b**2 + c**2 - a**2) / (2*b*c + 1e-12)
            cosB = (a**2 + c**2 - b**2) / (2*a*c + 1e-12)
            cosC = (a**2 + b**2 - c**2) / (2*a*b + 1e-12)
            angleA = safe_acos(cosA)
            angleB = safe_acos(cosB)
            angleC = safe_acos(cosC)
            defect = angleA + angleB + angleC - np.pi
            defects.append(defect)
        except Exception:
            continue
    if len(defects) == 0:
        return np.nan
    return float(np.mean(defects))

def bootstrap_curvature(counts, num_qubits, n_bootstrap=200, shots=512):
    """
    Estimate the standard error of curvature by bootstrapping the measurement counts.
    """
    keys = list(counts.keys())
    values = np.array([counts[k] for k in keys])
    total_shots = int(np.sum(values))
    if total_shots == 0:
        return np.nan
    curvatures = []
    for _ in range(n_bootstrap):
        # Resample bitstrings with replacement
        resampled = random.choices(keys, weights=values, k=shots)
        resampled_counts = {k: 0 for k in keys}
        for k in resampled:
            resampled_counts[k] += 1
        mi_matrix = compute_mi_matrix(resampled_counts, num_qubits)
        curv = estimate_curvature(mi_matrix)
        if not np.isnan(curv):
            curvatures.append(curv)
    if len(curvatures) == 0:
        return np.nan
    return float(np.std(curvatures))

def curvature_from_mi(mi, Q=None, eps=1e-4, embed_dim=2):
    mi = np.array(mi, dtype=float)
    n  = mi.shape[0]
    mi[mi < eps] = eps
    dist = -np.log(mi)
    if Q == 2:
        print(f"charge={Q}  dist_matrix:\n{dist.round(3)}")
    coords = MDS(n_components=embed_dim, dissimilarity='precomputed', random_state=0, max_iter=3000, eps=1e-9).fit_transform(dist)
    def safe_acos(x):
        return np.arccos(np.clip(x, -1.0, 1.0))
    defects = []
    from itertools import combinations
    for i, j, k in combinations(range(n), 3):
        a = np.linalg.norm(coords[j] - coords[k])
        b = np.linalg.norm(coords[i] - coords[k])
        c = np.linalg.norm(coords[i] - coords[j])
        if a < 1e-9 or b < 1e-9 or c < 1e-9:
            continue
        A = safe_acos((b**2 + c**2 - a**2) / (2*b*c))
        B = safe_acos((a**2 + c**2 - b**2) / (2*a*c))
        C = safe_acos((a**2 + b**2 - c**2) / (2*a*b))
        defects.append(A + B + C - np.pi)
    print(f"charge={Q} triangles_kept={len(defects)} first_defect={defects[0] if defects else 'NA'} (dim={embed_dim}D)")
    # 3D embedding diagnostics
    if embed_dim == 2 and Q is not None:
        coords3 = MDS(n_components=3, dissimilarity='precomputed', random_state=0, max_iter=3000, eps=1e-9).fit_transform(dist)
        defects3 = []
        for i, j, k in combinations(range(n), 3):
            a = np.linalg.norm(coords3[j] - coords3[k])
            b = np.linalg.norm(coords3[i] - coords3[k])
            c = np.linalg.norm(coords3[i] - coords3[j])
            if a < 1e-9 or b < 1e-9 or c < 1e-9:
                continue
            A = safe_acos((b**2 + c**2 - a**2) / (2*b*c))
            B = safe_acos((a**2 + c**2 - b**2) / (2*a*c))
            C = safe_acos((a**2 + b**2 - c**2) / (2*a*b))
            defects3.append(A + B + C - np.pi)
        print(f"charge={Q} triangles_kept={len(defects3)} first_defect={defects3[0] if defects3 else 'NA'} (dim=3D)")
    return (float(np.mean(defects)) if defects else np.nan), coords

def save_results(result, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    fname = f"{log_dir}/result_Q{result['charge']}.json"
    with open(fname, 'w') as f:
        json.dump(result, f, indent=2)

def save_summary(results, log_dir, num_qubits, depth, device):
    summary_path = os.path.join(log_dir, 'summary.txt')
    Qs = [r['charge'] for r in results]
    Ks = [r['curvature'] for r in results]
    K_errs = [r.get('curvature_error', float('nan')) for r in results]
    with open(summary_path, 'w') as f:
        f.write(f"Charge Sweep Experiment\n")
        f.write(f"Num Qubits: {num_qubits}, Depth: {depth}, Device: {device}\n")
        f.write(f"\nTheoretical Background:\n")
        f.write("This experiment injects varying charge into a quantum circuit and measures the resulting curvature using mutual information. The curvature is estimated via triangle defects in the MI matrix. Error bars are estimated by bootstrapping the measurement counts.\n")
        f.write(f"\nMethodology:\n")
        f.write(f"For each charge Q, a circuit is prepared with Q qubits in |1> and the rest in |0>. The circuit is entangled and measured. Mutual information matrices are computed from the measurement results, and curvature is estimated from triangle defects. Error bars are computed by resampling the measurement counts and recalculating curvature.\n")
        f.write(f"\nKey Metrics:\n")
        f.write(f"Curvature (average triangle defect) as a function of injected charge, with error bars.\n")
        f.write(f"\nResults:\n")
        for q, k, kerr in zip(Qs, Ks, K_errs):
            f.write(f"Q={q}, Curvature={k:.4f} ± {kerr:.4f}\n")
        f.write(f"\nAnalysis & Interpretation:\n")
        f.write(f"The curvature metric and its error bars reveal how charge injection affects the emergent geometry.\n")
        f.write(f"\nConclusion:\n")
        f.write(f"This experiment demonstrates a relationship between injected charge and emergent curvature in quantum circuits. Error bars provide a measure of statistical uncertainty.\n")

def charge_sweep_experiment(num_qubits=5, depth=3, charge_range=None, device='simulator', shots=2048, log_dir=None, n_bootstrap=200):
    if charge_range is None:
        charge_range = range(0, num_qubits+1)
    if log_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f"experiment_logs/charge_sweep_experiment_{timestamp}/"
    results = []
    # Backend selection
    if device == 'simulator':
        backend = FakeBrisbane()
    else:
        backend = device  # Should be IBM provider name
    curvature_errors = []
    for Q in charge_range:
        qc = build_circuit_with_charge(num_qubits, depth, Q)
        counts = run(qc, backend=backend, shots=shots)
        mi_matrix = compute_mi_matrix(counts, num_qubits)
        # Compute curvature and get MDS coordinates
        curvature, mds_coords = curvature_from_mi(mi_matrix, Q=Q)
        # Bootstrap error estimation
        boot_curvatures = []
        for _ in range(n_bootstrap):
            # Resample counts, recompute MI matrix, then curvature
            resampled = random.choices(list(counts.keys()), weights=list(counts.values()), k=shots)
            resampled_counts = {k: 0 for k in counts.keys()}
            for k in resampled:
                resampled_counts[k] += 1
            mi_boot = compute_mi_matrix(resampled_counts, num_qubits)
            curv_boot, _ = curvature_from_mi(mi_boot, Q=Q)
            boot_curvatures.append(curv_boot)
        curvature_err = float(np.nanstd(boot_curvatures))
        result = {
            'num_qubits': num_qubits,
            'depth': depth,
            'charge': Q,
            'curvature': curvature,
            'curvature_error': curvature_err,
            'mi_matrix': mi_matrix.tolist(),
            'mds_coords': mds_coords.tolist(),
            'device': device
        }
        results.append(result)
        save_results(result, log_dir)
        print(f"Q={Q}, K={curvature:.4f} ± {curvature_err:.4f}")
    # Plot K(Q) with error bars
    Qs = [r['charge'] for r in results]
    Ks = [r['curvature'] for r in results]
    K_errs = [r['curvature_error'] for r in results]
    plt.errorbar(Qs, Ks, yerr=K_errs, fmt='o-', capsize=5, label='Curvature')
    plt.xlabel('Injected Charge Q')
    plt.ylabel('Estimated Curvature (avg. triangle defect)')
    plt.title(f'Charge–Curvature Map (N={num_qubits}, D={depth}, device={device})')
    plt.grid(True)
    plt.legend()
    os.makedirs(log_dir, exist_ok=True)
    plt.savefig(f"{log_dir}/charge_curvature_map_{device}.png")
    plt.close()
    # Save summary
    save_summary(results, log_dir, num_qubits, depth, device)
    # Save all results in a single results.json
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Charge Sweep Experiment')
    parser.add_argument('--num_qubits', type=int, default=5, help='Number of qubits')
    parser.add_argument('--depth', type=int, default=3, help='Circuit depth')
    parser.add_argument('--device', type=str, default='simulator', help="'simulator' for FakeBrisbane or IBM provider name for hardware")
    parser.add_argument('--shots', type=int, default=2048, help='Number of shots')
    args = parser.parse_args()
    charge_sweep_experiment(num_qubits=args.num_qubits, depth=args.depth, device=args.device, shots=args.shots) 