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

# After the imports, add the following lines to create the output directory
exp_dir = "experiment_logs/ctc_geometry"
os.makedirs(exp_dir, exist_ok=True)

def shannon_entropy(probs):
    """
    Compute the Shannon entropy of a probability distribution.
    Args:
        probs (array-like): Probability distribution (should sum to 1).
    Returns:
        float: Shannon entropy in bits.
    """
    probs = np.array(probs)
    probs = probs / np.sum(probs)
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
        b = bitstring.zfill(total_qubits)
        key = ''.join([b[-(i+1)] for i in target_idxs])
        marginal[key] = marginal.get(key, 0) + count
    probs = np.array(list(marginal.values())) / shots
    return probs

def compute_mi_counts(counts, qA, qB, total_qubits, shots):
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

def build_ctc_circuit(phi):
    """
    Build a 4-qubit quantum circuit with a CTC-like feedback loop and RX(phi) evolution.
    Args:
        phi (float): RX rotation angle for all qubits.
    Returns:
        QuantumCircuit: The constructed circuit.
    """
    circ = QuantumCircuit(4)
    circ.h(0)
    circ.h(1)
    circ.cx(0, 1)
    circ.cx(1, 2)
    circ.cx(2, 3)
    circ.cx(3, 0)
    for i in range(4):
        circ.rx(phi, i)
    circ.measure_all()
    return circ

def run_ctc_geometry_experiment(device_name=None, shots=1024, phi_list=None):
    """
    Run the CTC geometry experiment for a sweep of RX angles (phi).
    Args:
        device_name (str): Name of the IBM backend to use (None for statevector sim).
        shots (int): Number of measurement shots (if using sampler).
        phi_list (list): List of RX angles to sweep.
    Returns:
        str: Path to experiment log directory.
    """
    if phi_list is None:
        phi_list = np.linspace(0, 2 * np.pi, 8)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = f"experiment_logs/ctc_geometry_qiskit_{device_name or 'statevector'}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    mi_matrices = []
    mds_coords = []
    for phi in phi_list:
        circ = build_ctc_circuit(phi)
        if device_name is not None:
            # Run on real backend or simulator using cgpt_run or Qiskit Sampler
            service = QiskitRuntimeService()
            backend = service.backend(device_name)
            tqc = transpile(circ, backend, optimization_level=3)
            if cgpt_run is not None:
                counts = cgpt_run(tqc, backend=backend, shots=shots)
            else:
                # Fallback: use Qiskit SamplerV2 if available
                from qiskit_ibm_runtime import SamplerV2 as Sampler, Session
                with Session(service=service, backend=backend) as session:
                    sampler = Sampler(session=session)
                    job = sampler.run(tqc, shots=shots)
                    result = job.result()
                    counts = result.quasi_dists[0] if hasattr(result, 'quasi_dists') else result.quasi_distribution[0]
            # Convert counts to int if needed
            counts = {k: int(v) for k, v in counts.items()}
            mi_matrix = np.zeros((4, 4))
            for i in range(4):
                for j in range(i + 1, 4):
                    mi = compute_mi_counts(counts, i, j, 4, shots)
                    mi_matrix[i, j] = mi_matrix[j, i] = mi
        else:
            # Use statevector simulation
            circ_no_meas = QuantumCircuit(4)
            circ_no_meas.h(0)
            circ_no_meas.h(1)
            circ_no_meas.cx(0, 1)
            circ_no_meas.cx(1, 2)
            circ_no_meas.cx(2, 3)
            circ_no_meas.cx(3, 0)
            for i in range(4):
                circ_no_meas.rx(phi, i)
            state = Statevector.from_instruction(circ_no_meas)
            probs = np.abs(state.data) ** 2
            # Convert probabilities to pseudo-counts for MI calculation
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
        mi_matrices.append(mi_matrix)
        # Convert MI to distance matrix for visualization
        epsilon = 1e-6
        dist = 1 / (mi_matrix + epsilon)
        np.fill_diagonal(dist, 0)
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coords = mds.fit_transform(dist)
        mds_coords.append(coords)
        # Plot the emergent geometry for this phi
        plt.figure(figsize=(8, 6))
        plt.scatter(coords[:, 0], coords[:, 1], c='blue', s=100)
        for i in range(4):
            plt.text(coords[i, 0], coords[i, 1], f"Q{i}", fontsize=12)
        plt.title(f"Emergent Geometry from CTC-like Structure (phi={phi:.2f})")
        plt.xlabel("MDS Dimension 1")
        plt.ylabel("MDS Dimension 2")
        plt.grid(True)
        plt.savefig(f"{exp_dir}/ctc_geometry_experiment_phi_{phi:.2f}.png")
        plt.close()
    # Save MI matrices and MDS coordinates
    with open(f"{exp_dir}/mi_matrix.json", "w") as f:
        json.dump({"phi_list": list(map(float, phi_list)), "mi_matrices": [m.tolist() for m in mi_matrices]}, f, indent=2)
    with open(f"{exp_dir}/mds_coordinates.json", "w") as f:
        json.dump({"phi_list": list(map(float, phi_list)), "mds_coordinates": [c.tolist() for c in mds_coords]}, f, indent=2)
    # Write a summary file
    with open(f"{exp_dir}/summary.txt", "w") as f:
        f.write("CTC Geometry Experiment Summary\n")
        f.write("================================\n\n")
        f.write(f"Device: {device_name or 'statevector'}\n")
        f.write(f"Shots: {shots}\n\n")
        f.write("Theoretical Background:\n")
        f.write("This experiment simulates a Closed Timelike Curve (CTC) geometry using a quantum circuit. It investigates how feedback loops and time-like evolution affect entanglement and emergent geometry.\n\n")
        f.write("Methodology:\n")
        f.write("A 4-qubit quantum circuit is constructed with a feedback loop structure. Mutual information is computed between qubits, and MDS is used to visualize the emergent geometry.\n\n")
        f.write("Results:\n")
        f.write(f"Results saved in: {exp_dir}\n")
        f.write("\nConclusion:\n")
        f.write("The experiment demonstrates how CTC-like structures influence quantum entanglement and emergent geometry, providing insights into the interplay between quantum mechanics and spacetime structure.\n")
    print(f"Experiment completed. Results saved in {exp_dir}")
    return exp_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CTC geometry experiment (Qiskit)')
    parser.add_argument('--device', type=str, default=None, help='IBM device name (or None for statevector sim)')
    parser.add_argument('--shots', type=int, default=1024, help='Number of shots (if using sampler)')
    parser.add_argument('--nphi', type=int, default=8, help='Number of phi values to sweep')
    args = parser.parse_args()
    phi_list = np.linspace(0, 2 * np.pi, args.nphi)
    run_ctc_geometry_experiment(device_name=args.device, shots=args.shots, phi_list=phi_list) 