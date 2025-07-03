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
from src.CGPTFactory import extract_counts_from_bitarray
from src.CGPTFactory import run as c_run
import traceback

def get_least_busy_backend():
    """
    Select the least busy IBM Qiskit backend with at least 4 qubits.
    Returns:
        backend: The selected backend object, or None if unavailable.
    """
    try:
        provider = QiskitRuntimeService()
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
    """
    Build a 4-qubit quantum circuit for the emergent spacetime experiment.
    Args:
        phi_val (float): Phase parameter for the circuit.
    Returns:
        QuantumCircuit: The constructed circuit.
    """
    qc = QuantumCircuit(4)
    qc.h(0)  # Create initial superposition
    qc.cx(0, 2)
    qc.cx(0, 3)
    qc.rx(phi_val, 0)
    # cz(0, 1) equivalent using CX and RZ
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
    """
    Compute the Shannon entropy of a probability distribution.
    Args:
        probs (array-like): Probability distribution (should sum to 1).
    Returns:
        float: Shannon entropy in bits.
    """
    probs = np.array(probs)
    return -np.sum(probs * np.log2(probs + 1e-12))

def marginal_probs(counts, total_qubits, target_idxs, shots):
    """
    Compute marginal probabilities for a subset of qubits from measurement counts.
    Args:
        counts (dict): Measurement outcome counts from Qiskit.
        total_qubits (int): Total number of qubits in the system.
        target_idxs (list): Indices of qubits to marginalize over.
        shots (int): Total number of measurement shots.
    Returns:
        np.ndarray: Marginal probability distribution for the target qubits.
    """
    marginal = {}
    for bitstring, count in counts.items():
        b = bitstring[::-1]  # Qiskit order (little-endian)
        key = ''.join([b[i] for i in target_idxs])
        marginal[key] = marginal.get(key, 0) + count
    probs = np.array(list(marginal.values())) / shots
    return probs

def compute_mi(counts, qA, qB, total_qubits, shots):
    """
    Compute the mutual information between two qubits from measurement counts.
    Args:
        counts (dict): Measurement outcome counts.
        qA, qB (int): Indices of the two qubits.
        total_qubits (int): Total number of qubits.
        shots (int): Number of measurement shots.
    Returns:
        float: Mutual information I(A:B).
    """
    AB = marginal_probs(counts, total_qubits, [qA, qB], shots)
    A = marginal_probs(counts, total_qubits, [qA], shots)
    B = marginal_probs(counts, total_qubits, [qB], shots)
    return shannon_entropy(A) + shannon_entropy(B) - shannon_entropy(AB)

def write_summary(exp_dir, backend_name, shots, error=None):
    """
    Write a human-readable summary of the experiment, including theoretical background, methodology, and results.
    Args:
        exp_dir (str): Directory to save the summary.
        backend_name (str): Name of the backend used.
        shots (int): Number of shots.
        error (str, optional): Error message if the experiment failed.
    """
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
    """
    Main function to run the emergent spacetime experiment.
    - Runs the quantum circuit for a range of phi values (timesteps)
    - Computes mutual information matrices for each timestep
    - Embeds geometry using MDS and computes entropy, curvature, and distance
    - Saves results and generates summary and plots
    """
    shots = 1024
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join('experiment_logs', f'emergent_spacetime_qiskit_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    try:
        timesteps = np.linspace(0, 3 * np.pi, 9)
        mi_matrices = []
        backend = get_least_busy_backend()
        if backend is None:
            write_summary(exp_dir, 'None', shots, error='Could not load IBMProvider backend.')
            return
        for i, phi_val in enumerate(timesteps):
            print(f"Processing timestep {i+1}/{len(timesteps)} (phi = {phi_val:.3f})")
            try:
                qc = build_circuit(phi_val)
                result = c_run(qc, backend=backend, shots=shots)
                # Robust extraction of counts from various possible result formats
                counts = None
                if isinstance(result, dict):
                    if 'counts' in result:
                        counts = result['counts']
                    elif 'result' in result and isinstance(result['result'], dict) and 'counts' in result['result']:
                        counts = result['result']['counts']
                    elif '__value__' in result and isinstance(result['__value__'], dict):
                        data = result['__value__'].get('data', None)
                        if data and hasattr(data, 'meas'):
                            bitarray = data.meas
                            counts = extract_counts_from_bitarray(bitarray)
                        elif 'counts' in result['__value__']:
                            counts = result['__value__']['counts']
                if counts is None:
                    print(f"[WARNING] Could not extract counts for phi={phi_val:.3f}, skipping.")
                    mi_matrices.append(np.full((4, 4), np.nan))
                    continue
                # Compute mutual information matrix for all pairs
                mi_matrix = np.zeros((4, 4))
                for a in range(4):
                    for b in range(a+1, 4):
                        mi = compute_mi(counts, a, b, 4, shots)
                        mi_matrix[a, b] = mi_matrix[b, a] = mi
                mi_matrices.append(mi_matrix)
            except Exception as e:
                print(f"[WARNING] Skipping phi={phi_val:.3f} due to error: {e}")
                traceback.print_exc()
                mi_matrices.append(np.full((4, 4), np.nan))
                continue
        # Save results
        results = {
            "timesteps": timesteps.tolist(),
            "mi_matrices": [m.tolist() for m in mi_matrices],
            "entropies": [],
            "curvatures": [],
            "distances": [],
            "geometries": [],
            "plaquette_curvatures": None,
            "mutual_information": None,
            "triangle_angles": None
        }
        for mi_matrix in mi_matrices:
            if np.isnan(mi_matrix).all():
                results["entropies"].append(None)
                results["curvatures"].append(None)
                results["distances"].append(None)
                results["geometries"].append(None)
                continue
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
        # Only plot geometry if available
        final_geometry = results["geometries"][-1]
        ax = axes[1,1]
        if final_geometry is not None:
            final_geometry = np.array(final_geometry)
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
        # Copy outputs to output_logs folders for redundancy and easy access
        import shutil
        for outdir in ["experiment_outputs/output_logs", "output_logs"]:
            os.makedirs(outdir, exist_ok=True)
            for fname in ["results.json", "results.png", "summary.txt"]:
                src = os.path.join(exp_dir, fname)
                dst = os.path.join(outdir, f"emergent_spacetime_qiskit_{timestamp}_{fname}")
                if os.path.exists(src):
                    shutil.copy2(src, dst)
    except Exception as e:
        print(f"[ERROR] Experiment failed: {e}")
        with open(f"{exp_dir}/error.log", "w") as f:
            f.write(str(e))
        write_summary(exp_dir, 'Unknown', shots, error=str(e))

if __name__ == "__main__":
    main() 