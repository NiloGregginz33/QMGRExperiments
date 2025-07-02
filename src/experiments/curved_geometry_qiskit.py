import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import argparse
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.result import marginal_counts
from scipy.stats import pearsonr
from sklearn.manifold import MDS
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) )
from src.CGPTFactory import run as cgpt_run


# --- Utility Functions ---
def shannon_entropy(probs):
    probs = np.array(probs)
    return -np.sum(probs * np.log2(probs + 1e-12))

def marginal_probs(counts, total_qubits, target_idxs, shots):
    marginal = {}
    for bitstring, count in counts.items():
        b = bitstring.zfill(total_qubits)
        key = ''.join([b[-(i+1)] for i in target_idxs])
        marginal[key] = marginal.get(key, 0) + count
    probs = np.array(list(marginal.values())) / shots
    return probs

def compute_mi(counts, qA, qB, total_qubits, shots):
    AB = marginal_probs(counts, total_qubits, [qA, qB], shots)
    A = marginal_probs(counts, total_qubits, [qA], shots)
    B = marginal_probs(counts, total_qubits, [qB], shots)
    return shannon_entropy(A) + shannon_entropy(B) - shannon_entropy(AB)

def estimate_local_curvature(coords, triplet):
    from numpy.linalg import norm
    i, j, k = triplet
    a = norm(coords[j] - coords[k])
    b = norm(coords[i] - coords[k])
    c = norm(coords[i] - coords[j])
    def safe_acos(x):
        return np.arccos(np.clip(x, -1.0, 1.0))
    angle_i = safe_acos((b**2 + c**2 - a**2) / (2 * b * c))
    angle_j = safe_acos((a**2 + c**2 - b**2) / (2 * a * c))
    angle_k = safe_acos((a**2 + b**2 - c**2) / (2 * a * b))
    return (angle_i + angle_j + angle_k) - np.pi

# --- Circuit Construction ---
def build_circuit(mode, phi):
    qc = QuantumCircuit(6)
    qc.h(0)
    qc.cx(0, 2)
    qc.cx(0, 3)
    if mode == "flat":
        qc.rx(phi, 0)
        qc.cz(0, 1)
        qc.cx(1, 2)
        qc.rx(phi, 2)
        qc.cz(1, 3)
        qc.cx(3, 4)
        qc.rx(phi, 4)
        qc.cx(4, 5)
    elif mode == "curved":
        qc.rx(phi, 0)
        qc.rx(phi, 1)
        qc.rx(phi, 2)
        qc.cz(0, 3)
        qc.cz(1, 4)
        qc.cz(2, 5)
        qc.cx(0, 5)
        qc.cx(5, 3)
        qc.cz(3, 4)
        qc.cz(4, 1)
        qc.cx(4, 2)
    qc.measure_all()

    return qc

# --- Main Experiment Function ---
def run_curved_geometry_qiskit(device_name=None, shots=1024):
    service = QiskitRuntimeService()
    if device_name is None:
        # Default to a real hardware backend (choose a 6+ qubit device)
        backends = [b for b in service.backends(simulator=False) if b.configuration().n_qubits >= 6 and b.status().operational]
        if not backends:
            raise RuntimeError("No suitable IBM hardware backend available.")
        backend = sorted(backends, key=lambda b: b.status().pending_jobs)[0]
        device_name = backend.name
    else:
        backend = service.backend(device_name)
    print(f"Using backend: {device_name}")

    exp_dir = f"experiment_logs/curved_geometry_qiskit_{device_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(exp_dir, exist_ok=True)

    timesteps = np.linspace(0, 3 * np.pi, 9)
    results = {}
    rt_data = []
    coords_list_2d = []
    coords_list_3d = []
    n_qubits = 6

    for mode in ["flat", "curved"]:
        mode_results = []
        for phi_val in timesteps:
            print(f"Running {mode} mode, phi={phi_val:.4f}")
            qc = build_circuit(mode, phi_val)
            tqc = transpile(qc, backend, optimization_level=3)
            try:
                counts = cgpt_run(tqc, backend=backend, shots=shots)
                if counts is None or not isinstance(counts, dict):
                    print(f"[WARNING] cgpt_run returned unexpected value: {counts}. Skipping this step.")
                    continue
            except Exception as e:
                print(f"[ERROR] cgpt_run failed: {e}")
                continue
            mi_matrix = np.zeros((n_qubits, n_qubits))
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    mi = compute_mi(counts, i, j, n_qubits, shots)
                    mi_matrix[i, j] = mi_matrix[j, i] = mi
            rad_probs = marginal_probs(counts, n_qubits, [3, 4], shots)
            S_rad = shannon_entropy(rad_probs)
            epsilon = 1e-6
            dist = np.exp(-mi_matrix)
            dist[dist > 1e4] = 1e4
            np.fill_diagonal(dist, 0)
            coords2 = MDS(n_components=2, dissimilarity='precomputed').fit_transform(dist)
            coords3 = MDS(n_components=3, dissimilarity='precomputed').fit_transform(dist)
            d_Q34 = np.linalg.norm(coords2[3] - coords2[4])
            rt_data.append((phi_val, S_rad, d_Q34))
            coords_list_2d.append(coords2)
            coords_list_3d.append(coords3)
            mode_results.append({
                "phi": float(phi_val),
                "S_rad": float(S_rad),
                "d_Q34": float(d_Q34),
                "mi_matrix": mi_matrix.tolist()
            })
            # Save MI matrix heatmap
            try:
                plt.figure(figsize=(5, 4))
                plt.imshow(mi_matrix, cmap='viridis', aspect='auto')
                plt.colorbar(label='Mutual Information')
                plt.title(f'MI Matrix {mode} φ={phi_val:.2f}')
                plt.xlabel('Qubit')
                plt.ylabel('Qubit')
                plt.savefig(f'{exp_dir}/mi_matrix_{mode}_{phi_val:.2f}.png')
                plt.close()
                print(f"Saved MI matrix heatmap for {mode} φ={phi_val:.2f}")
            except Exception as e:
                print(f"[ERROR] Failed to save MI matrix heatmap: {e}")
        results[mode] = mode_results

    # Save results
    print(f"Writing results to {exp_dir}/results.json")
    with open(f"{exp_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save summary
    print(f"Writing summary to {exp_dir}/summary.txt")
    with open(f"{exp_dir}/summary.txt", "w") as f:
        f.write("Curved Geometry Experiment (Qiskit/IBM)\n")
        f.write("==================================\n\n")
        f.write(f"Device: {device_name}\n")
        f.write(f"Shots: {shots}\n\n")
        f.write("Theoretical Background:\n")
        f.write("This experiment investigates how quantum information and entanglement behave in curved spacetime geometries. It compares the behavior of quantum systems in flat vs curved geometries to understand the influence of spacetime curvature on quantum information.\n\n")
        f.write("Methodology:\n")
        f.write("Quantum circuits are constructed to simulate both flat and curved spacetime geometries. The experiments analyze mutual information, entanglement patterns, and geometric features in both scenarios.\n\n")
        f.write("Results:\n")
        f.write(f"Results saved in: {exp_dir}\n")
        f.write("\nConclusion:\n")
        f.write("The experiment reveals how curvature affects quantum information distribution and entanglement patterns, providing insights into the relationship between quantum mechanics and spacetime geometry.\n")
    print(f"Experiment completed. Results saved in {exp_dir}")
    return exp_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run curved geometry experiment (Qiskit/IBM)')
    parser.add_argument('--device', type=str, default=None, help='IBM Quantum backend to use (default: real hardware)')
    parser.add_argument('--shots', type=int, default=1024, help='Number of shots for quantum measurements')
    parser.add_argument('--mode', type=str, choices=['flat', 'curved', 'both'], default='both', help="Which geometry mode to run: 'flat', 'curved', or 'both' (default: both)")
    args = parser.parse_args()

    def run_with_mode(device, shots, mode):
        service = QiskitRuntimeService()
        if device is None:
            backends = [b for b in service.backends(simulator=False) if b.configuration().n_qubits >= 6 and b.status().operational]
            if not backends:
                raise RuntimeError("No suitable IBM hardware backend available.")
            backend = sorted(backends, key=lambda b: b.status().pending_jobs)[0]
            device = backend.name
        else:
            backend = service.backend(device)
        print(f"Using backend: {device}")

        exp_dir = f"experiment_logs/curved_geometry_qiskit_{device}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(exp_dir, exist_ok=True)

        timesteps = np.linspace(0, 3 * np.pi, 6)
        results = {}
        rt_data = []
        coords_list_2d = []
        coords_list_3d = []
        n_qubits = 6

        modes = []
        if mode == 'both':
            modes = ['flat', 'curved']
        else:
            modes = [mode]

        for mode_val in modes:
            mode_results = []
            for phi_val in timesteps:
                print(f"Running {mode_val} mode, phi={phi_val:.4f}")
                qc = build_circuit(mode_val, phi_val)
                tqc = transpile(qc, backend, initial_layout=list(range(6)))
                try:
                    counts = cgpt_run(tqc, backend=backend, shots=shots)
                    if counts is None or not isinstance(counts, dict):
                        print(f"[WARNING] cgpt_run returned unexpected value: {counts}. Skipping this step.")
                        continue
                except Exception as e:
                    print(f"[ERROR] cgpt_run failed: {e}")
                    continue
                mi_matrix = np.zeros((n_qubits, n_qubits))
                for i in range(n_qubits):
                    for j in range(i+1, n_qubits):
                        mi = compute_mi(counts, i, j, n_qubits, shots)
                        mi_matrix[i, j] = mi_matrix[j, i] = mi
                rad_probs = marginal_probs(counts, n_qubits, [3, 4], shots)
                S_rad = shannon_entropy(rad_probs)
                epsilon = 1e-6
                dist = np.exp(-mi_matrix)
                dist[dist > 1e4] = 1e4
                np.fill_diagonal(dist, 0)
                coords2 = MDS(n_components=2, dissimilarity='precomputed').fit_transform(dist)
                coords3 = MDS(n_components=3, dissimilarity='precomputed').fit_transform(dist)
                d_Q34 = np.linalg.norm(coords2[3] - coords2[4])
                rt_data.append((phi_val, S_rad, d_Q34))
                coords_list_2d.append(coords2)
                coords_list_3d.append(coords3)
                mode_results.append({
                    "phi": float(phi_val),
                    "S_rad": float(S_rad),
                    "d_Q34": float(d_Q34),
                    "mi_matrix": mi_matrix.tolist()
                })
                # Save MI matrix heatmap
                try:
                    plt.figure(figsize=(5, 4))
                    plt.imshow(mi_matrix, cmap='viridis', aspect='auto')
                    plt.colorbar(label='Mutual Information')
                    plt.title(f'MI Matrix {mode_val} φ={phi_val:.2f}')
                    plt.xlabel('Qubit')
                    plt.ylabel('Qubit')
                    plt.savefig(f'{exp_dir}/mi_matrix_{mode_val}_{phi_val:.2f}.png')
                    plt.close()
                    print(f"Saved MI matrix heatmap for {mode_val} φ={phi_val:.2f}")
                except Exception as e:
                    print(f"[ERROR] Failed to save MI matrix heatmap: {e}")
            results[mode_val] = mode_results

        # Save results
        print(f"Writing results to {exp_dir}/results.json")
        with open(f"{exp_dir}/results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Save summary
        print(f"Writing summary to {exp_dir}/summary.txt")
        with open(f"{exp_dir}/summary.txt", "w") as f:
            f.write("Curved Geometry Experiment (Qiskit/IBM)\n")
            f.write("==================================\n\n")
            f.write(f"Device: {device}\n")
            f.write(f"Shots: {shots}\n\n")
            f.write("Theoretical Background:\n")
            f.write("This experiment investigates how quantum information and entanglement behave in curved spacetime geometries. It compares the behavior of quantum systems in flat vs curved geometries to understand the influence of spacetime curvature on quantum information.\n\n")
            f.write("Methodology:\n")
            f.write("Quantum circuits are constructed to simulate both flat and curved spacetime geometries. The experiments analyze mutual information, entanglement patterns, and geometric features in both scenarios.\n\n")
            f.write("Results:\n")
            f.write(f"Results saved in: {exp_dir}\n")
            f.write("\nConclusion:\n")
            f.write("The experiment reveals how curvature affects quantum information distribution and entanglement patterns, providing insights into the relationship between quantum mechanics and spacetime geometry.\n")
        print(f"Experiment completed. Results saved in {exp_dir}")
        return exp_dir

    run_with_mode(args.device, args.shots, args.mode) 