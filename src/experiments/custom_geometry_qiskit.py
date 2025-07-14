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
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.CGPTFactory import (
    run as cgpt_run,
    compute_plaquette_curvature_from_sv,
    compute_mutual_information,
    compute_triangle_angles,
    list_plaquettes,
    compute_face_curvature,
    qiskit_entropy
)
from qiskit.quantum_info import DensityMatrix, partial_trace
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.exceptions import QiskitBackendNotFoundError

# ───────────────────────────────────────────────────────────────────
def embed_geometry(D, model='euclidean', curvature=1.0):
    """
    D: (N×N) distance matrix
    model: 'euclidean', 'hyperbolic', or 'spherical'
    curvature: positive float |κ|
    returns: (N×2) embedded coords
    """
    D = np.array(D)
    N = D.shape[0]
    if model == 'euclidean':
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        return mds.fit_transform(D)

    elif model == 'hyperbolic':
        if N != 3:
            raise NotImplementedError("Hyperbolic embed only for N=3")
        d01, d02, d12 = D[0,1], D[0,2], D[1,2]
        r01 = np.tanh(np.sqrt(curvature) * d01 / 2)
        r02 = np.tanh(np.sqrt(curvature) * d02 / 2)
        p0 = np.array([0.0, 0.0])
        p1 = np.array([r01, 0.0])
        cosθ = (np.cosh(np.sqrt(curvature)*d01)*np.cosh(np.sqrt(curvature)*d02)
               - np.cosh(np.sqrt(curvature)*d12)) \
              / (np.sinh(np.sqrt(curvature)*d01)*np.sinh(np.sqrt(curvature)*d02))
        θ = np.arccos(np.clip(cosθ, -1.0, 1.0))
        p2 = np.array([r02*np.cos(θ), r02*np.sin(θ)])
        return np.vstack([p0,p1,p2])

    elif model == 'spherical':
        if N != 3:
            raise NotImplementedError("Spherical embed only for N=3")
        d01, d02, d12 = D[0,1], D[0,2], D[1,2]
        α, β, γ = d01*np.sqrt(curvature), d02*np.sqrt(curvature), d12*np.sqrt(curvature)
        p0 = np.array([0,0,1])
        p1 = np.array([np.sin(α), 0, np.cos(α)])
        cosφ = (np.cos(γ) - np.cos(α)*np.cos(β)) / (np.sin(α)*np.sin(β))
        φ = np.arccos(np.clip(cosφ, -1.0, 1.0))
        p2 = np.array([np.sin(β)*np.cos(φ),
                       np.sin(β)*np.sin(φ),
                       np.cos(β)])
        return np.vstack([p0[:2], p1[:2], p2[:2]])

    else:
        raise ValueError(f"Unknown geometry model '{model}'")
# ───────────────────────────────────────────────────────────────────

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

# Implement a function to compute mutual information from probabilities
def compute_mutual_information_from_probs(probs, i, j, n_qubits):
    # Calculate marginal probabilities
    p_i = sum(p for k, p in probs.items() if k[i] == '1')
    p_j = sum(p for k, p in probs.items() if k[j] == '1')
    p_ij = sum(p for k, p in probs.items() if k[i] == '1' and k[j] == '1')
    
    # Calculate entropies
    S_i = -p_i * np.log2(p_i + 1e-12) - (1 - p_i) * np.log2(1 - p_i + 1e-12)
    S_j = -p_j * np.log2(p_j + 1e-12) - (1 - p_j) * np.log2(1 - p_j + 1e-12)
    S_ij = -p_ij * np.log2(p_ij + 1e-12) - (1 - p_ij) * np.log2(1 - p_ij + 1e-12)
    
    # Compute mutual information
    return S_i + S_j - S_ij

# Implement the shannon_entropy function
def shannon_entropy(probs, shots):
    H = 0.0
    for p in probs.values():
        if p > 0:
            H -= p * np.log2(p)
    return H

def run_custom_geometry_qiskit(device_name=None, shots=1024, geometry='euclidean', curvature=1.0):
    service = QiskitRuntimeService()
    # select backend...
    # create exp_dir...
    timesteps = np.linspace(0, 3 * np.pi, 9)
    results = {}
    n_qubits = 6

    # Update backend selection logic
    if device_name == 'simulator':
        backend = FakeBrisbane()
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    else:
        if device_name is None:
            backends = [b for b in service.backends(simulator=False) if b.configuration().n_qubits >= 6 and b.status().operational]
            if not backends:
                raise RuntimeError("No suitable IBM hardware backend available.")
            backend = sorted(backends, key=lambda b: b.status().pending_jobs)[0]
        else:
            try:
                backend = service.get_backend(device_name)
            except QiskitBackendNotFoundError:
                raise RuntimeError(f"Backend {device_name} not found.")

    exp_dir = f"experiment_logs/custom_geometry_qiskit_{('FakeBrisbane' if device_name=='simulator' else device_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(exp_dir, exist_ok=True)

    # Adjust execution logic to use the selected backend
    for mode in (['flat','curved'] if args.mode=='both' else [args.mode]):
        mode_results = []
        for phi_val in timesteps:
            qc = build_circuit(mode, phi_val)
            if device_name == 'simulator':
                tqc = pm.run(qc)
            else:
                tqc = transpile(qc, backend, optimization_level=3)
            counts = cgpt_run(tqc, backend=backend, shots=shots)  # assume dict

            # --- Mutual information matrix ---
            mi_matrix = np.zeros((n_qubits,n_qubits))
            for i in range(n_qubits):
                for j in range(i+1,n_qubits):
                    # Compute mutual information directly from counts using probabilities
                    probs = {k: v / shots for k, v in counts.items()}
                    mi = compute_mutual_information_from_probs(probs, i, j, n_qubits)
                    mi_matrix[i,j] = mi_matrix[j,i] = mi

            # --- Radiation entropy & other metrics (unchanged) ---
            # Calculate probabilities from marginal counts
            rad_counts = marginal_counts(counts, [3, 4], n_qubits)
            rad_probs = {k: v / shots for k, v in rad_counts.items()}
            S_rad = shannon_entropy(rad_probs, shots)

            # --- Build true distance matrix D = -ln(MI + eps) ---
            eps = 1e-12
            MI_safe = np.clip(mi_matrix, eps, None)
            D = -np.log(MI_safe)
            np.fill_diagonal(D,0.0)

            # --- Embed in chosen geometry ---
            coords2 = embed_geometry(D, model=geometry, curvature=curvature)
            coords3 = MDS(n_components=3, dissimilarity='precomputed', random_state=42).fit_transform(D)

            # --- Save, plot, collect results ---
            d_Q34 = np.linalg.norm(coords2[3]-coords2[4])
            mode_results.append({
                'phi': float(phi_val),
                'S_rad': float(S_rad),
                'd_Q34': float(d_Q34),
                'mi_matrix': mi_matrix.tolist(),
                'coords2': coords2.tolist(),
                'coords3': coords3.tolist(),
                # ... other metrics ...
            })

            # Plot:
            plt.figure(figsize=(8,6))
            plt.scatter(coords2[:,0], coords2[:,1], c='C0', s=100)
            for i,(x,y) in enumerate(coords2):
                plt.text(x,y,f"Q{i}")
            plt.title(f"{mode.capitalize()} φ={phi_val:.2f} ({geometry}, κ={curvature})")
            plt.savefig(f"{exp_dir}/{mode}_phi_{phi_val:.2f}_{geometry}.png")
            plt.close()

        results[mode] = mode_results

    # ... save JSON and summary as before ...

    # Enhance data logging
    # Collect all relevant data
    # results = {
    #     'statevector': statevector,
    #     'mutual_information': mutual_information,
    #     'entropy': entropy,
    #     'curvature': args.curvature,
    #     'geometry': args.geometry,
    #     'device': args.device,
    #     'shots': args.shots
    # }

    # Create a comprehensive summary
    summary = """
    Theoretical Background:
    This experiment explores the embedding of qubit entanglement networks in hyperbolic space, simulating curved geometries.

    Methodology:
    The experiment uses a quantum circuit to simulate entanglement and measure mutual information across qubits.

    Key Metrics:
    - Statevector: {statevector}
    - Mutual Information: {mutual_information}
    - Entropy: {entropy}

    Analysis:
    The results indicate...

    Conclusions:
    The experiment demonstrates...
    """.format(statevector=statevector, mutual_information=mutual_information, entropy=entropy)

    # Correct the logic to ensure results and summary are constructed using collected data

    # Implement the save_results_and_summary function
    def save_results_and_summary(results, summary, log_dir):
        # Ensure the log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Save results to results.json
        results_path = os.path.join(log_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save summary to summary.txt
        summary_path = os.path.join(log_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(summary)

    # Ensure results and summary are constructed using collected data

    # Construct results using collected data
    results = {
        'flat': results.get('flat', []),
        'curved': results.get('curved', []),
        'curvature': curvature,
        'geometry': geometry,
        'device': device_name,
        'shots': shots
    }

    # Construct summary using collected data
    summary = """
Theoretical Background:
This experiment explores the embedding of qubit entanglement networks in hyperbolic space, simulating curved geometries.

Methodology:
The experiment uses a quantum circuit to simulate entanglement and measure mutual information across qubits.

Key Metrics:
- Curvature: {curvature}
- Geometry: {geometry}
- Device: {device_name}
- Shots: {shots}

Analysis:
The results indicate...

Conclusions:
The experiment demonstrates...
""".format(curvature=curvature, geometry=geometry, device_name=device_name, shots=shots)

    # Ensure the directory is created before saving files
    os.makedirs(exp_dir, exist_ok=True)

    # Save results and summary
    save_results_and_summary(results, summary, exp_dir)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',    type=str, default=None)
    parser.add_argument('--shots',     type=int, default=1024)
    parser.add_argument('--simulator', action='store_true')
    parser.add_argument('--mode',      choices=['flat','curved','both'], default='both')
    parser.add_argument('--geometry',  choices=['euclidean','hyperbolic','spherical'],
                        default='euclidean',
                        help="2D embedding geometry")
    parser.add_argument('--curvature', type=float, default=1.0,
                        help="|κ| for hyperbolic/spherical")
    args = parser.parse_args()

    run_custom_geometry_qiskit(
        device_name=args.device,
        shots=args.shots,
        geometry=args.geometry,
        curvature=args.curvature
    ) 