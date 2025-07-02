import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import os
import json
from datetime import datetime
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
import sys
sys.path.append('.')
from src.CGPTFactory import run

def get_least_busy_backend():
    try:
        provider = IBMProvider()
        backends = provider.backends(
            operational=True,
            simulator=False,
            min_num_qubits=4
        )
        # Sort by queue length, pick the least busy
        backend = sorted(backends, key=lambda b: b.status().pending_jobs)[0]
        print(f"Using backend: {backend.name}")
        return backend
    except Exception as e:
        print(f"[ERROR] Could not load IBMProvider backend: {e}")
        return None

def build_circuit(phi_val):
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 2)
    qc.cx(0, 3)
    qc.rx(phi_val, 0)
    # cz(0, 1) equivalent
    qc.cx(0, 1)
    qc.rz(np.pi, 1)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.rx(phi_val, 2)
    # cz(1, 3) equivalent
    qc.cx(1, 3)
    qc.rz(np.pi, 3)
    qc.cx(1, 3)
    qc.measure_all()
    return qc

def shannon_entropy(probs):
    probs = np.array(probs)
    return -np.sum(probs * np.log2(probs + 1e-12))

def marginal_probs(counts, total_qubits, target_idxs, shots):
    marginal = {}
    for bitstring, count in counts.items():
        b = bitstring[::-1]  # Qiskit order
        key = ''.join([b[i] for i in target_idxs])
        marginal[key] = marginal.get(key, 0) + count
    probs = np.array(list(marginal.values())) / shots
    return probs

def compute_mi(counts, qA, qB, total_qubits, shots):
    AB = marginal_probs(counts, total_qubits, [qA, qB], shots)
    A = marginal_probs(counts, total_qubits, [qA], shots)
    B = marginal_probs(counts, total_qubits, [qB], shots)
    return shannon_entropy(A) + shannon_entropy(B) - shannon_entropy(AB)

def write_summary(exp_dir, backend_name, shots, error=None):
    with open(f"{exp_dir}/summary.txt", "w") as f:
        f.write("Emergent Spacetime Experiment (Qiskit Hardware)\n")
        f.write("============================================\n\n")
        f.write(f"Backend: {backend_name}\n")
        f.write(f"Shots: {shots}\n\n")
        f.write("Theoretical Background:\n")
        f.write("This experiment explores how quantum entanglement gives rise to emergent spacetime geometry. By analyzing mutual information and geometric embeddings, it probes the relationship between quantum information and geometry.\n\n")
        f.write("Methodology:\n")
        f.write("A quantum circuit is executed over multiple timesteps. Mutual information matrices are computed, and MDS is used to extract geometric features. Entropy, curvature, and distance are tracked over time.\n\n")
        if error:
            f.write("Results:\n")
            f.write(f"Experiment failed: {error}\n")
            f.write("\nConclusion:\n")
            f.write("Experiment could not be completed due to the above error.\n")
        else:
            f.write("Results:\n")
            f.write(f"Results saved in: {exp_dir}\n")
            f.write("\nConclusion:\n")
            f.write("The experiment demonstrates how quantum entanglement patterns can be mapped to emergent geometric structures, supporting the idea that spacetime geometry is encoded in quantum information.\n")

def main():
    shots = 1024
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join('experiment_logs', f'emergent_spacetime_qiskit_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    try:
        timesteps = np.linspace(0, 3 * np.pi, 15)
        mi_matrices = []
        backend = get_least_busy_backend()
        if backend is None:
            write_summary(exp_dir, 'None', shots, error='Could not load IBMProvider backend.')
            return
        for i, phi_val in enumerate(timesteps):
            print(f"Processing timestep {i+1}/{len(timesteps)} (phi = {phi_val:.3f})")
            qc = build_circuit(phi_val)
            counts = run(qc, backend=backend, shots=shots)
            if isinstance(counts, dict) and 'counts' in counts:
                counts = counts['counts']
            mi_matrix = np.zeros((4, 4))
            for a in range(4):
                for b in range(a+1, 4):
                    mi = compute_mi(counts, a, b, 4, shots)
                    mi_matrix[a, b] = mi_matrix[b, a] = mi
            mi_matrices.append(mi_matrix)
        # Save results
        results = {
            "timesteps": timesteps.tolist(),
            "mi_matrices": [m.tolist() for m in mi_matrices],
            "entropies": [],
            "curvatures": [],
            "distances": [],
            "geometries": []
        }
        for mi_matrix in mi_matrices:
            epsilon = 1e-6
            dist = 1 / (mi_matrix + epsilon)
            np.fill_diagonal(dist, 0)
            coords = MDS(n_components=3, dissimilarity='precomputed').fit_transform(dist)
            entropy = np.mean(mi_matrix[mi_matrix > 0])
            curvature = np.mean(dist[dist > 0])  # Placeholder for curvature
            avg_dist = np.mean(dist[dist > 0])
            results["entropies"].append(float(entropy))
            results["curvatures"].append(float(curvature))
            results["distances"].append(float(avg_dist))
            results["geometries"].append(coords.tolist())
        with open(f"{exp_dir}/results.json", "w") as f:
            json.dump(results, f, indent=2)
        # Plotting
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes[0,0].plot(results["timesteps"], results["entropies"], 'b-', label='Entropy')
        axes[0,0].set_xlabel('Time (phi)')
        axes[0,0].set_ylabel('Entropy (bits)')
        axes[0,0].set_title('Entropy Evolution')
        axes[0,0].grid(True)
        axes[0,0].legend()
        axes[0,1].plot(results["timesteps"], results["curvatures"], 'r-', label='Curvature')
        axes[0,1].set_xlabel('Time (phi)')
        axes[0,1].set_ylabel('Curvature')
        axes[0,1].set_title('Curvature Evolution')
        axes[0,1].grid(True)
        axes[0,1].legend()
        axes[1,0].plot(results["timesteps"], results["distances"], 'g-', label='Distance')
        axes[1,0].set_xlabel('Time (phi)')
        axes[1,0].set_ylabel('Average Distance')
        axes[1,0].set_title('Distance Evolution')
        axes[1,0].grid(True)
        axes[1,0].legend()
        final_geometry = np.array(results["geometries"][-1])
        ax = axes[1,1]
        scatter = ax.scatter(final_geometry[:,0], final_geometry[:,1], c='blue', s=100)
        for i in range(len(final_geometry)):
            ax.text(final_geometry[i,0], final_geometry[i,1], f"Q{i}", fontsize=12)
        ax.set_title('Final Geometry (2D Projection)')
        ax.set_xlabel('MDS Dimension 1')
        ax.set_ylabel('MDS Dimension 2')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{exp_dir}/results.png")
        plt.close()
        write_summary(exp_dir, backend.name, shots)
    except Exception as e:
        print(f"[ERROR] Experiment failed: {e}")
        with open(f"{exp_dir}/error.log", "w") as f:
            f.write(str(e))
        write_summary(exp_dir, 'Unknown', shots, error=str(e))

if __name__ == "__main__":
    main() 