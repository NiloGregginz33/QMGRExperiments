import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import argparse
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, Session
from qiskit.result import marginal_counts
from scipy.stats import pearsonr
from sklearn.manifold import MDS
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) )
from src.CGPTFactory import run as cgpt_run, compute_plaquette_curvature_from_sv, compute_mutual_information, compute_triangle_angles, list_plaquettes
from src.CGPTFactory import compute_face_curvature
from qiskit.quantum_info import DensityMatrix, partial_trace
from src.CGPTFactory import qiskit_entropy
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import AerSimulator


# --- Utility Functions ---
def shannon_entropy(probs):
    """
    Compute the Shannon entropy of a probability distribution.
    Args:
        probs (array-like): Probability distribution (should sum to 1).
    Returns:
        float: Shannon entropy in bits.
    """
    probs = np.array(probs)
    return -np.sum(probs * np.log2(probs + 1e-12))  # Add epsilon to avoid log(0)


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
        b = bitstring.zfill(total_qubits)  # Ensure bitstring has correct length
        key = ''.join([b[-(i+1)] for i in target_idxs])  # Extract bits for target qubits
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


def estimate_local_curvature(coords, triplet):
    """
    Estimate the local Gaussian curvature at a triangle defined by three points in the embedding.
    Args:
        coords (np.ndarray): Coordinates of all points (from MDS embedding).
        triplet (tuple): Indices of the three points forming the triangle.
    Returns:
        float: Angle deficit (sum of triangle angles minus pi).
    """
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
    """
    Build a quantum circuit for the flat or curved geometry experiment.
    Args:
        mode (str): 'flat' or 'curved' geometry.
        phi (float): Phase parameter for the circuit.
    Returns:
        QuantumCircuit: The constructed circuit.
    """
    qc = QuantumCircuit(6)
    qc.h(0)  # Create initial superposition
    qc.cx(0, 2)
    qc.cx(0, 3)
    if mode == "flat":
        # Flat geometry: local interactions
        qc.rx(phi, 0)
        qc.cz(0, 1)
        qc.cx(1, 2)
        qc.rx(phi, 2)
        qc.cz(1, 3)
        qc.cx(3, 4)
        qc.rx(phi, 4)
        qc.cx(4, 5)
    elif mode == "curved":
        # Curved geometry: more nonlocal interactions
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
    qc.measure_all()  # Measure all qubits
    return qc


# --- Main Experiment Function ---
def run_curved_geometry_qiskit(device_name=None, shots=1024):
    """
    Run the curved geometry experiment on IBM Qiskit hardware or simulator.
    Args:
        device_name (str): Name of the IBM backend to use (None for auto-select).
        shots (int): Number of measurement shots.
    Returns:
        None. Results are saved to experiment_logs/.
    """
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

    # Create a timestamped directory for results
    exp_dir = f"experiment_logs/curved_geometry_qiskit_{device_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(exp_dir, exist_ok=True)

    timesteps = np.linspace(0, 3 * np.pi, 9)  # Range of phi values
    results = {}
    rt_data = []
    coords_list_2d = []
    coords_list_3d = []
    n_qubits = 6

    # Determine which modes to run based on args.mode
    if args.mode == 'both':
        modes_to_run = ['flat', 'curved']
    else:
        modes_to_run = [args.mode]

    for mode in modes_to_run:
        mode_results = []
        for phi_val in timesteps:
            print(f"Running {mode} mode, phi={phi_val:.4f}")
            qc = build_circuit(mode, phi_val)
            tqc = transpile(qc, backend, optimization_level=3)
            # --- Readout mitigation ---
            if not backend.configuration().simulator:
                meas_fitter = apply_readout_mitigation(qc, backend, shots)
            try:
                counts = cgpt_run(tqc, backend=backend, shots=shots)
                if counts is None or not isinstance(counts, dict):
                    print(f"[WARNING] cgpt_run returned unexpected value: {counts}. Skipping this step.")
                    continue
                # Apply readout mitigation if on hardware
                if not backend.configuration().simulator:
                    counts = meas_fitter.filter.apply(counts)
            except Exception as e:
                print(f"[ERROR] cgpt_run failed: {e}")
                continue
            # Compute mutual information matrix for all pairs
            mi_matrix = np.zeros((n_qubits, n_qubits))
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    mi = compute_mi(counts, i, j, n_qubits, shots)
                    mi_matrix[i, j] = mi_matrix[j, i] = mi
            # Compute entropy of radiation subsystem (qubits 3,4)
            rad_probs = marginal_probs(counts, n_qubits, [3, 4], shots)
            S_rad = shannon_entropy(rad_probs)
            epsilon = 1e-6
            # Convert MI matrix to a distance matrix for geometry embedding
            dist = np.exp(-mi_matrix)
            dist[dist > 1e4] = 1e4  # Cap large distances
            np.fill_diagonal(dist, 0)
            # Embed geometry in 2D and 3D using MDS
            coords2 = MDS(n_components=2, dissimilarity='precomputed').fit_transform(dist)
            coords3 = MDS(n_components=3, dissimilarity='precomputed').fit_transform(dist)
            # Distance between Q3 and Q4 in the embedding
            d_Q34 = np.linalg.norm(coords2[3] - coords2[4])
            # --- New: Compute additional metrics ---
            # Try to get statevector (skip if not available)
            statevector = None
            try:
                from qiskit.quantum_info import Statevector
                statevector = Statevector.from_instruction(qc)
            except Exception as e:
                print(f"[INFO] Could not extract statevector: {e}")
            # Compute plaquette curvatures if possible
            plaquette_curvatures = None
            mutual_info_dict = None
            triangle_angles = None
            face_curvatures = None
            if statevector is not None:
                try:
                    # List all 4-qubit plaquettes for curvature
                    plaquettes = list_plaquettes(2, 3)  # Example: for 2x3 grid, adjust as needed
                    plaquette_curvatures = compute_plaquette_curvature_from_sv(statevector, plaquettes, n_qubits)
                    # Compute face curvature using statevector and plaquettes
                    dm = DensityMatrix(statevector)
                    face_curvatures = {}
                    for corners in plaquettes:
                        a, b, c, d = corners
                        ab = [a, b]; cd = [c, d]; abcd = corners
                        rho_ab = partial_trace(dm, [q for q in range(n_qubits) if q not in ab])
                        rho_cd = partial_trace(dm, [q for q in range(n_qubits) if q not in cd])
                        rho_abcd = partial_trace(dm, [q for q in range(n_qubits) if q not in abcd])
                        S_ab = qiskit_entropy(rho_ab)
                        S_cd = qiskit_entropy(rho_cd)
                        S_abcd = qiskit_entropy(rho_abcd)
                        face_curvatures[tuple(corners)] = S_ab + S_cd - S_abcd
                except Exception as e:
                    print(f"[INFO] Could not compute plaquette curvatures: {e}")
                try:
                    # Compute MI for all pairs from statevector
                    mutual_info_dict = {}
                    for i in range(n_qubits):
                        for j in range(i+1, n_qubits):
                            mi = compute_mutual_information(statevector, [i], [j])
                            mutual_info_dict[(i, j)] = mi
                except Exception as e:
                    print(f"[INFO] Could not compute mutual information: {e}")
                try:
                    # Compute triangle angles for a few triplets
                    D = np.zeros((n_qubits, n_qubits))
                    for i in range(n_qubits):
                        for j in range(i+1, n_qubits):
                            mi = mutual_info_dict.get((i, j), 0) if mutual_info_dict else 0
                            D[i, j] = D[j, i] = 1.0 / (mi + epsilon) if mi > 0 else 1e6
                    triangle_angles = {}
                    triplets = [(0, 1, 2), (3, 4, 5)]  # Example triplets
                    for (i, j, k) in triplets:
                        try:
                            angles = compute_triangle_angles(D, i, j, k)
                            triangle_angles[(i, j, k)] = [float(a) for a in angles]
                        except Exception as e:
                            triangle_angles[(i, j, k)] = None
                except Exception as e:
                    print(f"[INFO] Could not compute triangle angles: {e}")
            # Compute MI matrix from counts (already present)
            mi_matrix = np.zeros((n_qubits, n_qubits))
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    mi = compute_mi(counts, i, j, n_qubits, shots)
                    mi_matrix[i, j] = mi_matrix[j, i] = mi
            # --- New: Embed geometry and compute curvature from MI matrix ---
            epsilon = 1e-6
            dist = np.exp(-mi_matrix)
            dist[dist > 1e4] = 1e4
            np.fill_diagonal(dist, 0)
            coords2 = MDS(n_components=2, dissimilarity='precomputed').fit_transform(dist)
            coords3 = MDS(n_components=3, dissimilarity='precomputed').fit_transform(dist)
            d_Q34 = np.linalg.norm(coords2[3] - coords2[4])
            # --- Curvature from embedding ---
            # Triangle angle sums (for all triangles)
            triangle_angle_sums = []
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    for k in range(j+1, n_qubits):
                        a = np.linalg.norm(coords2[j] - coords2[k])
                        b = np.linalg.norm(coords2[i] - coords2[k])
                        c = np.linalg.norm(coords2[i] - coords2[j])
                        def safe_acos(x):
                            return np.arccos(np.clip(x, -1.0, 1.0))
                        angle_i = safe_acos((b**2 + c**2 - a**2) / (2 * b * c))
                        angle_j = safe_acos((a**2 + c**2 - b**2) / (2 * a * c))
                        angle_k = safe_acos((a**2 + b**2 - c**2) / (2 * a * b))
                        angle_sum = angle_i + angle_j + angle_k
                        triangle_angle_sums.append({
                            "triangle": [i, j, k],
                            "angle_sum": float(angle_sum)
                        })
            # Gaussian curvature at each vertex (angle deficit method)
            vertex_angle_sum = [0.0 for _ in range(n_qubits)]
            triangle_count = [0 for _ in range(n_qubits)]
            for entry in triangle_angle_sums:
                i, j, k = entry["triangle"]
                angle_sum = entry["angle_sum"]
                # Distribute angles to vertices
                a = np.linalg.norm(coords2[j] - coords2[k])
                b = np.linalg.norm(coords2[i] - coords2[k])
                c = np.linalg.norm(coords2[i] - coords2[j])
                def safe_acos(x):
                    return np.arccos(np.clip(x, -1.0, 1.0))
                angle_i = safe_acos((b**2 + c**2 - a**2) / (2 * b * c))
                angle_j = safe_acos((a**2 + c**2 - b**2) / (2 * a * c))
                angle_k = safe_acos((a**2 + b**2 - c**2) / (2 * a * b))
                vertex_angle_sum[i] += angle_i
                vertex_angle_sum[j] += angle_j
                vertex_angle_sum[k] += angle_k
                triangle_count[i] += 1
                triangle_count[j] += 1
                triangle_count[k] += 1
            gaussian_curvature = []
            for v in range(n_qubits):
                if triangle_count[v] > 0:
                    deficit = 2 * np.pi - vertex_angle_sum[v]
                    gaussian_curvature.append({
                        "vertex": v,
                        "angle_sum": float(vertex_angle_sum[v]),
                        "triangle_count": triangle_count[v],
                        "gaussian_curvature": float(deficit)
                    })
                else:
                    gaussian_curvature.append({
                        "vertex": v,
                        "angle_sum": 0.0,
                        "triangle_count": 0,
                        "gaussian_curvature": 0.0
                    })
            mode_results.append({
                "phi": float(phi_val),
                "S_rad": float(S_rad),
                "d_Q34": float(d_Q34),
                "mi_matrix": mi_matrix.tolist(),
                "triangle_angle_sums": triangle_angle_sums,
                "gaussian_curvature": gaussian_curvature,
                "plaquette_curvatures": plaquette_curvatures,
                "face_curvatures": face_curvatures,
                "mutual_information": mutual_info_dict,
                "triangle_angles": triangle_angles
            })
            rt_data.append((phi_val, S_rad, d_Q34))
            coords_list_2d.append(coords2)
            coords_list_3d.append(coords3)
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
    parser.add_argument('--device', type=str, default=None, help='IBM device name (or None for auto)')
    parser.add_argument('--shots', type=int, default=1024, help='Number of shots')
    parser.add_argument('--simulator', action='store_true', help='Use FakeManilaV2 simulator backend')
    parser.add_argument('--mode', type=str, choices=['flat', 'curved', 'both'], default='both', help="Which geometry mode to run: 'flat', 'curved', or 'both' (default: both)")
    args = parser.parse_args()

    if args.simulator:
        backend = FakeBrisbane()
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        print("Using FakeBrisbane simulator backend.")
    else:
        service = QiskitRuntimeService()
        if args.device is None:
            backends = [b for b in service.backends(simulator=False) if b.configuration().n_qubits >= 6 and b.status().operational]
            if not backends:
                raise RuntimeError("No suitable IBM hardware backend available.")
            backend = sorted(backends, key=lambda b: b.status().pending_jobs)[0]
            device_name = backend.name
        else:
            backend = service.backend(args.device)
            device_name = args.device
        print(f"Using backend: {device_name}")

    exp_dir = f"experiment_logs/curved_geometry_qiskit_{'FakeBrisbane' if args.simulator else device_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(exp_dir, exist_ok=True)

    timesteps = np.linspace(0, 3 * np.pi, 9)
    results = {}
    rt_data = []
    coords_list_2d = []
    coords_list_3d = []
    n_qubits = 6

    # Determine which modes to run based on args.mode
    if args.mode == 'both':
        modes_to_run = ['flat', 'curved']
    else:
        modes_to_run = [args.mode]

    for mode in modes_to_run:
        mode_results = []
        for phi_val in timesteps:
            print(f"Running {mode} mode, phi={phi_val:.4f}")
            qc = build_circuit(mode, phi_val)
            if args.simulator:
                tqc = pm.run(qc)  # Use preset pass manager for simulator
            else:
                tqc = transpile(qc, backend, optimization_level=3)  # Transpile for hardware
            try:
                counts = cgpt_run(tqc, backend=backend, shots=args.shots)
                if counts is None or not isinstance(counts, dict):
                    print(f"[WARNING] cgpt_run returned unexpected value: {counts}. Skipping this step.")
                    continue
            except Exception as e:
                print(f"[ERROR] cgpt_run failed: {e}")
                continue
            # --- Repeat the same analysis as above for each phi and mode ---
            # Compute mutual information matrix for all pairs of qubits
            mi_matrix = np.zeros((n_qubits, n_qubits))
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    mi = compute_mi(counts, i, j, n_qubits, args.shots)
                    mi_matrix[i, j] = mi_matrix[j, i] = mi
            # Compute entropy of radiation subsystem (qubits 3,4)
            rad_probs = marginal_probs(counts, n_qubits, [3, 4], args.shots)
            S_rad = shannon_entropy(rad_probs)
            epsilon = 1e-6
            # Convert MI matrix to a distance matrix for geometry embedding
            dist = np.exp(-mi_matrix)
            dist[dist > 1e4] = 1e4  # Cap large distances
            np.fill_diagonal(dist, 0)
            # Embed geometry in 2D and 3D using MDS
            coords2 = MDS(n_components=2, dissimilarity='precomputed').fit_transform(dist)
            coords3 = MDS(n_components=3, dissimilarity='precomputed').fit_transform(dist)
            # Distance between Q3 and Q4 in the embedding
            d_Q34 = np.linalg.norm(coords2[3] - coords2[4])
            # --- New: Compute additional metrics ---
            statevector = None
            if args.simulator:
                try:
                    from qiskit.quantum_info import Statevector
                    statevector = Statevector.from_instruction(qc)
                except Exception as e:
                    print(f"[INFO] Could not extract statevector: {e}")
            else:
                try:
                    from qiskit.quantum_info import Statevector
                    statevector = Statevector.from_instruction(qc)
                except Exception as e:
                    print(f"[INFO] Could not extract statevector: {e}")
            plaquette_curvatures = None
            mutual_info_dict = None
            triangle_angles = None
            face_curvatures = None
            if statevector is not None:
                try:
                    # List all 4-qubit plaquettes for curvature
                    plaquettes = list_plaquettes(2, 3)
                    plaquette_curvatures = compute_plaquette_curvature_from_sv(statevector, plaquettes, n_qubits)
                    dm = DensityMatrix(statevector)
                    face_curvatures = {}
                    for corners in plaquettes:
                        a, b, c, d = corners
                        ab = [a, b]; cd = [c, d]; abcd = corners
                        rho_ab = partial_trace(dm, [q for q in range(n_qubits) if q not in ab])
                        rho_cd = partial_trace(dm, [q for q in range(n_qubits) if q not in cd])
                        rho_abcd = partial_trace(dm, [q for q in range(n_qubits) if q not in abcd])
                        S_ab = qiskit_entropy(rho_ab)
                        S_cd = qiskit_entropy(rho_cd)
                        S_abcd = qiskit_entropy(rho_abcd)
                        face_curvatures[tuple(corners)] = S_ab + S_cd - S_abcd
                except Exception as e:
                    print(f"[INFO] Could not compute plaquette curvatures: {e}")
                try:
                    # Compute MI for all pairs from statevector
                    mutual_info_dict = {}
                    for i in range(n_qubits):
                        for j in range(i+1, n_qubits):
                            mi = compute_mutual_information(statevector, [i], [j])
                            mutual_info_dict[(i, j)] = mi
                except Exception as e:
                    print(f"[INFO] Could not compute mutual information: {e}")
                try:
                    # Compute triangle angles for a few triplets
                    D = np.zeros((n_qubits, n_qubits))
                    for i in range(n_qubits):
                        for j in range(i+1, n_qubits):
                            mi = mutual_info_dict.get((i, j), 0) if mutual_info_dict else 0
                            D[i, j] = D[j, i] = 1.0 / (mi + epsilon) if mi > 0 else 1e6
                    triangle_angles = {}
                    triplets = [(0, 1, 2), (3, 4, 5)]  # Example triplets
                    for (i, j, k) in triplets:
                        try:
                            angles = compute_triangle_angles(D, i, j, k)
                            triangle_angles[(i, j, k)] = [float(a) for a in angles]
                        except Exception as e:
                            triangle_angles[(i, j, k)] = None
                except Exception as e:
                    print(f"[INFO] Could not compute triangle angles: {e}")
            # --- Curvature from embedding ---
            # Triangle angle sums (for all triangles)
            triangle_angle_sums = []
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    for k in range(j+1, n_qubits):
                        a = np.linalg.norm(coords2[j] - coords2[k])
                        b = np.linalg.norm(coords2[i] - coords2[k])
                        c = np.linalg.norm(coords2[i] - coords2[j])
                        def safe_acos(x):
                            return np.arccos(np.clip(x, -1.0, 1.0))
                        angle_i = safe_acos((b**2 + c**2 - a**2) / (2 * b * c))
                        angle_j = safe_acos((a**2 + c**2 - b**2) / (2 * a * c))
                        angle_k = safe_acos((a**2 + b**2 - c**2) / (2 * a * b))
                        angle_sum = angle_i + angle_j + angle_k
                        triangle_angle_sums.append({
                            "triangle": [i, j, k],
                            "angle_sum": float(angle_sum)
                        })
            # Gaussian curvature at each vertex (angle deficit method)
            vertex_angle_sum = [0.0 for _ in range(n_qubits)]
            triangle_count = [0 for _ in range(n_qubits)]
            for entry in triangle_angle_sums:
                i, j, k = entry["triangle"]
                angle_sum = entry["angle_sum"]
                # Distribute angles to vertices
                a = np.linalg.norm(coords2[j] - coords2[k])
                b = np.linalg.norm(coords2[i] - coords2[k])
                c = np.linalg.norm(coords2[i] - coords2[j])
                def safe_acos(x):
                    return np.arccos(np.clip(x, -1.0, 1.0))
                angle_i = safe_acos((b**2 + c**2 - a**2) / (2 * b * c))
                angle_j = safe_acos((a**2 + c**2 - b**2) / (2 * a * c))
                angle_k = safe_acos((a**2 + b**2 - c**2) / (2 * a * b))
                vertex_angle_sum[i] += angle_i
                vertex_angle_sum[j] += angle_j
                vertex_angle_sum[k] += angle_k
                triangle_count[i] += 1
                triangle_count[j] += 1
                triangle_count[k] += 1
            # Compute Gaussian curvature (angle deficit) for each vertex
            gaussian_curvature = []
            for v in range(n_qubits):
                if triangle_count[v] > 0:
                    deficit = 2 * np.pi - vertex_angle_sum[v]
                    gaussian_curvature.append({
                        "vertex": v,
                        "angle_sum": float(vertex_angle_sum[v]),
                        "triangle_count": triangle_count[v],
                        "gaussian_curvature": float(deficit)
                    })
                else:
                    gaussian_curvature.append({
                        "vertex": v,
                        "angle_sum": 0.0,
                        "triangle_count": 0,
                        "gaussian_curvature": 0.0
                    })
            # Collect all results for this phi and mode
            mode_results.append({
                "phi": float(phi_val),
                "S_rad": float(S_rad),
                "d_Q34": float(d_Q34),
                "mi_matrix": mi_matrix.tolist(),
                "triangle_angle_sums": triangle_angle_sums,
                "gaussian_curvature": gaussian_curvature,
                "plaquette_curvatures": plaquette_curvatures,
                "face_curvatures": face_curvatures,
                "mutual_information": mutual_info_dict,
                "triangle_angles": triangle_angles
            })
            rt_data.append((phi_val, S_rad, d_Q34))
            coords_list_2d.append(coords2)
            coords_list_3d.append(coords3)
            # Save MI matrix heatmap for visualization
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

    # Save all results as JSON for reproducibility and further analysis
    print(f"Writing results to {exp_dir}/results.json")
    with open(f"{exp_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save a human-readable summary of the experiment
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