import sys
import os
import argparse
import logging
import random
import hashlib
import time
import string
import random as pyrandom
import itertools
import scipy.optimize

# Adjust Python path to include the Factory directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Factory'))

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, entropy
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from CGPTFactory import run
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.manifold import MDS
from qiskit.exceptions import QiskitError
import networkx as nx
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Add command-line argument parsing
p = argparse.ArgumentParser(description="Run a custom curvature circuit")
p.add_argument("--num_qubits", type=int,   default=3, help="Number of qubits")
p.add_argument("--topology", choices=["star","chain","ring","complete","custom"],
               default="star", help="Entanglement pattern")
p.add_argument("--custom_edges", type=str, default=None,
               help="Comma-separated 'u-v[:w]' pairs if topology=custom (e.g., '0-1:1.0,1-2:2.0,2-0:0.5')")
p.add_argument("--alpha",       type=float, default=0.8,
                   help="Decay rate for 'star' entangler")
p.add_argument("--weight",      type=float, default=1.0,
                   help="Uniform weight for 'chain'/'ring'")
p.add_argument("--gamma",       type=float, default=0.3,
                   help="Charge-injection strength")
p.add_argument("--sigma",       type=float, default=None,
                   help="Gaussian width for charge (default = num_qubits/2)")
p.add_argument("--init_angle",  type=float, default=0.0,
                   help="Initial Rx angle on each qubit")
p.add_argument("--shots",       type=int,   default=1024,
                   help="Number of measurement shots")
p.add_argument("--device", type=str, default="simulator", help="Execution device: simulator or IBM provider name")
p.add_argument("--geometry", type=str, default="euclidean", choices=["euclidean", "hyperbolic", "spherical"], help="Geometry for embedding: euclidean, hyperbolic, or spherical")
p.add_argument("--curvature", type=float, nargs='+', default=[1.0], help="Curvature parameter(s) κ for non-Euclidean geometries. Can pass multiple values for sweep.")
p.add_argument("--timesteps", type=int, default=1, help="Number of entangling layers (time steps)")

# Use the second parser for command-line arguments
args = p.parse_args()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# ─── Topology entanglers ──────────────────────────────────────────────────────
def _entangle_star(qc, alpha):
    """Star graph: qubit 0 connected to all others with RZZ decaying by exp(-alpha i)."""
    n = qc.num_qubits
    for i in range(1, n):
        w = np.exp(-alpha * i)
        qc.rzz(w, 0, i)

def _entangle_chain(qc, weight):
    """Chain: uniform RZZ(weight) between neighbors 0–1,1–2,…"""
    n = qc.num_qubits
    for i in range(n - 1):
        qc.rzz(weight, i, i+1)

def _entangle_ring(qc, weight):
    """Ring: chain plus an RZZ between last and first."""
    n = qc.num_qubits
    for i in range(n):
        qc.rzz(weight, i, (i+1) % n)

_ENTANGLERS = {
    "star":  _entangle_star,
    "chain": _entangle_chain,
    "ring":  _entangle_ring,
}

# ─── Charge injection ─────────────────────────────────────────────────────────
def _apply_charge(qc, gamma, sigma=None):
    """Apply RZ phases γ exp[−(q/σ)²] on each qubit q."""
    n = qc.num_qubits
    if sigma is None:
        sigma = n/2
    for q in range(n):
        angle = gamma * np.exp(-(q/sigma)**2)
        qc.rz(angle, q)

def make_graph(topology: str, n: int, custom_edges: str = None, default_weight: float = 1.0) -> nx.Graph:
    """
    Return a NetworkX graph on n nodes for the given topology.
    - 'star', 'chain', 'ring', 'complete'
    - 'custom'    with custom_edges="0-1:1.0,1-3:2.0,2-3:0.5"
    """
    if topology == "star":
        return nx.star_graph(n-1)
    elif topology == "chain":
        return nx.path_graph(n)
    elif topology == "ring":
        return nx.cycle_graph(n)
    elif topology == "complete":
        return nx.complete_graph(n)
    elif topology == "custom":
        if not custom_edges:
            raise ValueError("You must pass --custom_edges for topology='custom'")
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for token in custom_edges.split(","):
            if ":" in token:
                edge_part, w_str = token.split(":")
                u, v = map(int, edge_part.split("-"))
                w = float(w_str)
            else:
                u, v = map(int, token.split("-"))
                w = default_weight
            G.add_edge(u, v, weight=w)
        return G
    else:
        raise ValueError(f"Unknown topology '{topology}'")

# ─── Circuit factory ─────────────────────────────────────────────────────────
def build_custom_circuit_layers(num_qubits, topology, custom_edges,
                         alpha, weight, gamma, sigma, init_angle,
                         geometry=None, curvature=None, log_edge_weights=False, timesteps=1):
    """
    Build a list of QuantumCircuits, one for each timestep, where each circuit
    includes all layers up to and including that timestep.
    """
    circuits = []
    qc = QuantumCircuit(num_qubits)
    # 1) initial superposition / rotation
    if init_angle == 0.0:
        qc.h(range(num_qubits))
    else:
        for q in range(num_qubits):
            qc.rx(init_angle, q)
    for t in range(timesteps):
        # Entangling layer for this timestep
        if geometry in ("spherical", "hyperbolic") and curvature is not None:
            base_weight = weight
            std_dev = base_weight * (curvature / 10)
            edge_weights = {}
            edge_list = []
            for i in range(num_qubits):
                for j in range(i+1, num_qubits):
                    w = float(np.random.normal(loc=base_weight, scale=std_dev))
                    w = float(np.clip(w, 0.05, 1.0))
                    edge_weights[(i, j)] = w
                    edge_list.append(f"{i}-{j}:{w:.4f}")
            custom_edges_str = ",".join(edge_list)
            G = make_graph("custom", num_qubits, custom_edges_str, default_weight=base_weight)
            for u, v, data in G.edges(data=True):
                w = data.get('weight', base_weight)
                qc.ryy(np.pi * w, u, v)
            if log_edge_weights and t == 0:
                weights = list(edge_weights.values())
                print(f"[LOG] Edge weights: {weights}")
                print(f"[LOG] Edge weight variance: {np.var(weights)}")
            if t == 0:
                qc._custom_edges_str = custom_edges_str
                qc._edge_weight_variance = float(np.var(list(edge_weights.values())))
        else:
            G = make_graph(topology, num_qubits, custom_edges, default_weight=weight)
            for u, v, data in G.edges(data=True):
                w = data.get('weight', weight)
                qc.rzz(w, u, v)
            if t == 0:
                qc._custom_edges_str = custom_edges if custom_edges is not None else None
                qc._edge_weight_variance = None
        # Save a copy of the circuit up to this timestep (before charge injection)
        circuits.append(qc.copy())
    # After all entangling layers, apply charge injection and measurement to the final circuit
    _apply_charge(qc, gamma, sigma)
    qc.measure_all()
    return circuits, qc

# ─── Runner ───────────────────────────────────────────────────────────────────
def run_circuit(qc, shots, simulator, device_name):
    if simulator:
        backend = FakeBrisbane()
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        tqc = pm.run(qc)
        # FakeBrisbane.run returns a job-like object
        result = backend.run(tqc, shots=shots).result()
        counts = result.get_counts()
    else:
        service = QiskitRuntimeService()
        backend = service.backend(device_name)
        tqc = transpile(qc, backend, optimization_level=3)
        # Use cgpt_run if desired, here we fall back to the runtime sampler
        from src.CGPTFactory import run as cgpt_run
        counts = cgpt_run(tqc, backend=backend, shots=shots)
    return counts

# ─── Metrics Calculation ─────────────────────────────────────────────────────
def calculate_entropy(counts):
    """Calculate the entropy of the measurement results."""
    total_shots = sum(counts.values())
    probabilities = [count / total_shots for count in counts.values()]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

def calculate_mi_for_edges_only(mi_dict, graph):
    """Calculate MI only for edges in the graph."""
    edge_mi = {}
    for u, v in graph.edges():
        # Look for the MI value in the dictionary
        key1 = f"I_{u},{v}"
        key2 = f"I_{v},{u}"
        if key1 in mi_dict:
            edge_mi[(u, v)] = mi_dict[key1]
        elif key2 in mi_dict:
            edge_mi[(u, v)] = mi_dict[key2]
        else:
            # If not found, use a small default value
            edge_mi[(u, v)] = 1e-6
    return edge_mi

def compute_graph_shortest_path_distances(edge_mi, graph):
    """Compute shortest path distances using NetworkX."""
    # Create weighted graph with d_e(u,v) = -ln(I_u,v)
    G_weighted = nx.Graph()
    
    # Add all nodes
    G_weighted.add_nodes_from(graph.nodes())
    
    # Add edges with weights
    for (u, v), mi_value in edge_mi.items():
        # Clamp MI to (0, 1] to avoid negative weights
        mi_clamped = min(max(mi_value, 1e-10), 1.0)
        weight = -np.log(mi_clamped)
        G_weighted.add_edge(u, v, weight=weight)
    
    # Compute all-pairs shortest paths
    try:
        shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G_weighted, weight='weight'))
        # Convert to distance matrix format
        n = graph.number_of_nodes()
        distance_matrix = np.full((n, n), np.inf)
        for i in range(n):
            for j in range(n):
                if i in shortest_paths and j in shortest_paths[i]:
                    distance_matrix[i, j] = shortest_paths[i][j]
                elif i == j:
                    distance_matrix[i, j] = 0
        # Print MI and distance matrices for debugging
        print("Mutual Information matrix (edges only):", edge_mi)
        print("Distance matrix:", distance_matrix)
        return distance_matrix, shortest_paths
    except nx.NetworkXNoPath:
        # If graph is disconnected, return infinity for disconnected components
        return np.full((graph.number_of_nodes(), graph.number_of_nodes()), np.inf), {}

def check_hyperbolicity(D):
    """Calculate Gromov δ (hyperbolicity) - always ≥0."""
    n = D.shape[0]
    max_delta = 0.0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                d_ij, d_ik, d_jk = D[i,j], D[i,k], D[j,k]
                delta1 = 0.5*(d_ij + d_ik - d_jk)
                delta2 = 0.5*(d_ij + d_jk - d_ik)
                delta3 = 0.5*(d_ik + d_jk - d_ij)
                delta_ijk = max(abs(delta1), abs(delta2), abs(delta3))
                max_delta = max(max_delta, delta_ijk)
    return max_delta

def compute_spherical_angle(a, b, c, curvature):
    k = np.sqrt(curvature)
    num = np.cos(k * a) - np.cos(k * b) * np.cos(k * c)
    denom = np.sin(k * b) * np.sin(k * c)
    return np.arccos(np.clip(num / denom, -1.0, 1.0))

def compute_hyperbolic_angle(a, b, c, curvature):
    k = np.sqrt(abs(curvature))
    num = np.cosh(k * b) * np.cosh(k * c) - np.cosh(k * a)
    denom = np.sinh(k * b) * np.sinh(k * c)
    return np.arccos(np.clip(num / denom, -1.0, 1.0))

def calculate_angle_sum(D, i, j, k, geometry="euclidean", curvature=1.0):
    """Calculate angle sum for a single triangle using the correct law of cosines for the geometry."""
    a, b, c = D[j,k], D[i,k], D[i,j]
    if geometry == "spherical":
        alpha = compute_spherical_angle(a, b, c, curvature)
        beta  = compute_spherical_angle(b, a, c, curvature)
        gamma = compute_spherical_angle(c, a, b, curvature)
    elif geometry == "hyperbolic":
        alpha = compute_hyperbolic_angle(a, b, c, curvature)
        beta  = compute_hyperbolic_angle(b, a, c, curvature)
        gamma = compute_hyperbolic_angle(c, a, b, curvature)
    else:  # flat/Euclidean
        def euc_angle(opposite, x, y):
            cosA = (x**2 + y**2 - opposite**2) / (2 * x * y)
            return np.arccos(np.clip(cosA, -1.0, 1.0))
        alpha = euc_angle(a, b, c)
        beta  = euc_angle(b, a, c)
        gamma = euc_angle(c, a, b)
    return float(alpha + beta + gamma)

def calculate_all_angle_sums(D, geometry="euclidean", curvature=1.0):
    """Calculate angle sums for all triangles in the distance matrix."""
    angle_sums = []
    n = D.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                angle_sums.append(calculate_angle_sum(D, i, j, k, geometry, curvature))
    return angle_sums

def embed_geometry(D, model='euclidean', curvature=1.0):
    """
    Embed an N×N distance matrix D into 2D (for 'euclidean' or 'hyperbolic') or
    onto the sphere S² of curvature=+curvature via spectral embedding.
    For 'hyperbolic', N=3 is exact, N>3 uses MDS as an approximation.
    """
    D = np.asarray(D, float)
    N = D.shape[0]

    if model == 'euclidean':
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
        return mds.fit_transform(D), None

    elif model == 'spherical':
        # radius of sphere:
        R = 1.0/np.sqrt(curvature)
        K = np.sqrt(curvature)
        G = np.cos(K * D)
        vals, vecs = np.linalg.eigh(G)
        idx = np.argsort(vals)[::-1][:3]
        L = np.diag(np.sqrt(np.maximum(vals[idx], 0.0)))
        V = vecs[:, idx]
        X3 = V.dot(L)
        norms = np.linalg.norm(X3, axis=1, keepdims=True)
        X3 = R * X3 / norms
        X2 = X3[:, :2]
        return X2, X3

    elif model == 'hyperbolic':
        # For N=3, embed triangle in Poincaré disk using hyperbolic law of cosines
        if N == 3:
            # Place first point at (0,0), second at (d01,0) in Klein model, third by law of cosines
            d01, d02, d12 = D[0,1], D[0,2], D[1,2]
            # Convert geodesic distances to chordal distances in Klein model
            def klein_chordal(d, K):
                return np.tanh(np.sqrt(K)*d/2) * 2
            K = curvature
            x0 = np.array([0,0])
            x1 = np.array([klein_chordal(d01, K), 0])
            # Use law of cosines to find x2
            a = klein_chordal(d01, K)
            b = klein_chordal(d02, K)
            c = klein_chordal(d12, K)
            # Place x2 at (x, y)
            x = (a**2 + b**2 - c**2)/(2*a)
            y = np.sqrt(max(b**2 - x**2, 0))
            x2 = np.array([x, y])
            X2 = np.vstack([x0, x1, x2])
            return X2, None
        else:
            # For N>3, use MDS as an approximate embedding (warn in code)
            from sklearn.manifold import MDS
            print("[WARNING] Hyperbolic embedding for N>3 uses MDS as an approximation.")
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
            return mds.fit_transform(D), None
    else:
        raise ValueError("Unknown model, pick 'euclidean', 'spherical', or 'hyperbolic'.")

def check_triangle_inequality(D):
    """Check for triangle inequality violations in the distance matrix D."""
    n = D.shape[0]
    violations = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a, b, c = D[i, j], D[i, k], D[j, k]
                # Triangle inequality: each side <= sum of other two
                if a > b + c + 1e-8 or b > a + c + 1e-8 or c > a + b + 1e-8:
                    violations.append(((i, j, k), (a, b, c)))
    if violations:
        print("Triangle inequality violations found:")
        for (i, j, k), (a, b, c) in violations:
            print(f"Triangle ({i},{j},{k}): sides = {a:.4f}, {b:.4f}, {c:.4f}")
    else:
        print("No triangle inequality violations found.")
    return violations

# ─── Helper Functions ────────────────────────────────────────────────────────

def generate_classical_shadows(circuit, n_shadows, shots, backend):
    """Generate classical shadows using random single-qubit Clifford settings."""
    # Placeholder for actual shadow generation
    return np.random.rand(n_shadows, circuit.num_qubits)


def estimate_purities_from_shadows(shadows):
    """Estimate purities from classical shadows."""
    # Placeholder for actual purity estimation
    return np.random.rand(shadows.shape[1]), np.random.rand(shadows.shape[1], shadows.shape[1])


def compute_von_neumann_MI(statevector):
    """Compute von Neumann mutual information from a statevector."""
    n = int(np.log2(len(statevector)))
    mi = {}
    for i in range(n):
        for j in range(i + 1, n):
            rho_ij = partial_trace(statevector, [k for k in range(n) if k != i and k != j])
            S_i = entropy(partial_trace(statevector, [k for k in range(n) if k != i]))
            S_j = entropy(partial_trace(statevector, [k for k in range(n) if k != j]))
            S_ij = entropy(rho_ij)
            mi[f"I_{i},{j}"] = S_i + S_j - S_ij
    return mi

def generate_asymmetric_edges(num_qubits, target_curvature, asymmetry_factor=1.0, base_weight=0.2):
    """
    Generate a custom_edges string for a complete graph with asymmetric, curvature-informed edge weights.
    Edge weights are drawn from a Gaussian with mean=base_weight, std=base_weight * (target_curvature/10) * asymmetry_factor.
    Weights are clamped to [0.05, 1.0].
    Returns a string: "i-j:weight,i-j:weight,..."
    """
    edge_list = []
    std_dev = base_weight * (target_curvature / 10) * asymmetry_factor
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            w = float(np.random.normal(loc=base_weight, scale=std_dev))
            w = float(np.clip(w, 0.05, 1.0))
            edge_list.append(f"{i}-{j}:{w:.4f}")
    return ",".join(edge_list)

def generate_short_uid(length=6):
    """Generate a short random alphanumeric string for unique file naming."""
    chars = string.ascii_uppercase + string.digits
    return ''.join(pyrandom.choices(chars, k=length))

def make_short_filename(num_qubits, geometry, curvature, device, uid):
    geom_short = geometry[0].upper()
    dev_short = device[:3].lower()
    curv_short = str(curvature).replace('.', '')
    return f"results_n{num_qubits}_geom{geom_short}_curv{curv_short}_{dev_short}_{uid}.json"

def lorentzian_mds(D, ndim=3, max_iter=1000, lr=1e-2, random_state=42):
    """
    Perform Lorentzian (Minkowski) MDS embedding.
    D: (N,N) Lorentzian dissimilarity matrix (squared intervals)
    ndim: number of output dimensions (e.g., 3 for (t, x, y))
    Returns: coords (N, ndim)
    """
    np.random.seed(random_state)
    N = D.shape[0]
    # Initial guess: time = event index // num_qubits, space = random
    t_guess = np.repeat(np.arange(N // args.num_qubits), args.num_qubits)
    x_guess = np.random.randn(N)
    y_guess = np.random.randn(N)
    X0 = np.stack([t_guess, x_guess, y_guess], axis=1).flatten()
    def stress(X):
        X = X.reshape(N, ndim)
        T = X[:, 0]
        Xs = X[:, 1:]
        S = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dt = T[i] - T[j]
                dx = Xs[i] - Xs[j]
                s2 = -dt**2 + np.sum(dx**2)
                S[i, j] = s2
        mask = ~np.eye(N, dtype=bool)
        return np.mean((S[mask] - D[mask])**2)
    res = scipy.optimize.minimize(stress, X0, method='L-BFGS-B', options={'maxiter': max_iter})
    coords = res.x.reshape(N, ndim)
    return coords

# ─── Main CLI ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for kappa in args.curvature:
        # If custom_edges is null and geometry is not flat/euclidean, generate asymmetric edges
        if args.geometry in ("spherical", "hyperbolic") and kappa is not None:
            n = args.num_qubits
            base_weight = args.weight
            asymmetry_factor = 1.0  # Could be made a CLI arg
            if args.custom_edges is None:
                np.random.seed(42)  # For reproducibility
                custom_edges = generate_asymmetric_edges(n, kappa, asymmetry_factor, base_weight)
                # For diagnostics, parse weights and log variance
                weights = [float(token.split(":")[1]) for token in custom_edges.split(",")]
                edge_weight_variance = float(np.var(weights))
                print(f"[LOG] Edge weights: {weights}")
                print(f"[LOG] Edge weight variance: {edge_weight_variance}")
            else:
                custom_edges = args.custom_edges
                edge_weight_variance = None
            # Build layered circuits for per-timestep MI/distance
            circuits, qc = build_custom_circuit_layers(
                num_qubits = n,
                topology   = "custom",
                custom_edges = custom_edges,
                alpha      = args.alpha,
                weight     = base_weight,
                gamma      = args.gamma,
                sigma      = args.sigma,
                init_angle = args.init_angle,
                geometry   = args.geometry,
                curvature  = kappa,
                log_edge_weights=False,
                timesteps  = args.timesteps
            )
        else:
            custom_edges = args.custom_edges
            edge_weight_variance = None
            # Build layered circuits for per-timestep MI/distance
            circuits, qc = build_custom_circuit_layers(
                num_qubits = args.num_qubits,
                topology   = args.topology,
                custom_edges = custom_edges,
                alpha      = args.alpha,
                weight     = args.weight,
                gamma      = args.gamma,
                sigma      = args.sigma,
                init_angle = args.init_angle,
                geometry   = args.geometry,
                curvature  = kappa,
                log_edge_weights=False,
                timesteps  = args.timesteps
            )

        # Build the circuit without measure_all for simulator
        if args.device == "simulator":
            qc.data = [op for op in qc.data if op.operation.name != 'measure']
        else:
            qc.measure_all()

        mi_per_timestep = []
        distmat_per_timestep = []
        for t, circ in enumerate(circuits):
            # For simulator, use statevector
            if args.device == "simulator":
                backend = FakeBrisbane()
                statevector = Statevector.from_int(0, 2**args.num_qubits)
                statevector = statevector.evolve(circ)
                mi = compute_von_neumann_MI(statevector)
                G = make_graph(args.topology, args.num_qubits, custom_edges, default_weight=args.weight)
                edge_mi = calculate_mi_for_edges_only(mi, G)
                distance_matrix, _ = compute_graph_shortest_path_distances(edge_mi, G)
                mi_per_timestep.append(mi)
                distmat_per_timestep.append(distance_matrix.tolist())
            else:
                # For hardware, skip for now (could use classical shadows)
                mi_per_timestep.append(None)
                distmat_per_timestep.append(None)

        # 3) calculate metrics using graph-shortest-path approach
        G = make_graph("custom" if (args.geometry in ("spherical", "hyperbolic") and kappa is not None) else args.topology, args.num_qubits, custom_edges, default_weight=args.weight)
        edge_mi = calculate_mi_for_edges_only(mi_per_timestep[-1], G) # Use the last MI for final metrics
        distance_matrix, shortest_paths = compute_graph_shortest_path_distances(edge_mi, G)
        # Check for triangle inequality violations
        triangle_violations = check_triangle_inequality(distance_matrix)
        gromov_delta = check_hyperbolicity(distance_matrix)
        angle_sums = calculate_all_angle_sums(distance_matrix, geometry=args.geometry, curvature=kappa)
        mean_angle_sum = np.mean(angle_sums) if angle_sums else np.pi
        min_angle_sum = np.min(angle_sums) if angle_sums else np.pi
        max_angle_sum = np.max(angle_sums) if angle_sums else np.pi
        mean_distance = np.mean(distance_matrix[distance_matrix != np.inf])
        # Calculate deviation from π for each triangle
        angle_sum_deviations = [x - np.pi for x in angle_sums]

        # 4) geometric embedding
        coords2, coords3d = embed_geometry(distance_matrix, model=args.geometry, curvature=kappa)

        # Build event DAG and Lorentzian dissimilarity matrix
        num_events = args.num_qubits * args.timesteps
        event_nodes = [(i, t) for t in range(args.timesteps) for i in range(args.num_qubits)]
        event_idx = {evt: idx for idx, evt in enumerate(event_nodes)}
        # Build event DAG: edges = spatial (within t) + temporal (i, t)->(i, t+1)
        event_edges = []
        for t in range(args.timesteps):
            # Spatial edges
            mi_dict = mi_per_timestep[t]
            # Use MI to get spatial distances for this timestep
            G_spatial = make_graph(args.topology, args.num_qubits, custom_edges, default_weight=args.weight)
            edge_mi = calculate_mi_for_edges_only(mi_dict, G_spatial)
            distmat, _ = compute_graph_shortest_path_distances(edge_mi, G_spatial)
            for i in range(args.num_qubits):
                for j in range(i+1, args.num_qubits):
                    event_edges.append({
                        "type": "spatial",
                        "from": (i, t),
                        "to": (j, t),
                        "weight": distmat[i, j]
                    })
            # Temporal edges
            if t < args.timesteps - 1:
                for i in range(args.num_qubits):
                    event_edges.append({
                        "type": "temporal",
                        "from": (i, t),
                        "to": (i, t+1),
                        "weight": 1.0
                    })
        # Build Lorentzian dissimilarity matrix
        L = np.zeros((num_events, num_events))
        for idx_a, (i, t1) in enumerate(event_nodes):
            for idx_b, (j, t2) in enumerate(event_nodes):
                if t1 == t2:
                    # Spatial separation at this time
                    G_spatial = make_graph(args.topology, args.num_qubits, custom_edges, default_weight=args.weight)
                    edge_mi = calculate_mi_for_edges_only(mi_per_timestep[t1], G_spatial)
                    distmat, _ = compute_graph_shortest_path_distances(edge_mi, G_spatial)
                    d = distmat[i, j]
                    L[idx_a, idx_b] = d ** 2
                elif i == j:
                    # Pure time-lag (timelike)
                    dt = abs(t2 - t1)
                    L[idx_a, idx_b] = - (dt ** 2)
                else:
                    # Mixed: time and space
                    dt = abs(t2 - t1)
                    G_spatial = make_graph(args.topology, args.num_qubits, custom_edges, default_weight=args.weight)
                    edge_mi = calculate_mi_for_edges_only(mi_per_timestep[min(t1, t2)], G_spatial)
                    distmat, _ = compute_graph_shortest_path_distances(edge_mi, G_spatial)
                    d = distmat[i, j]
                    L[idx_a, idx_b] = - (dt ** 2) + d ** 2
        # Build Lorentzian MDS embedding
        lorentzian_embedding = lorentzian_mds(L, ndim=3)

        # 4) output
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'experiment_logs', 'custom_curvature_experiment')
        os.makedirs(log_dir, exist_ok=True)
        uid = generate_short_uid()
        short_filename = make_short_filename(args.num_qubits, args.geometry, kappa, args.device, uid)
        output_path = os.path.join(log_dir, short_filename)
        with open(output_path, 'w') as f:
            json.dump({
                "spec": {**vars(args), "curvature": kappa, "custom_edges": custom_edges, "timesteps": args.timesteps},
                "uid": uid,
                "counts": None, # No counts for per-timestep simulation
                "mutual_information": mi_per_timestep,
                "distance_matrix_per_timestep": distmat_per_timestep,
                "edge_mi": {f"{u},{v}": val for (u, v), val in edge_mi.items()},
                "distance_matrix": distance_matrix.tolist(),
                "shortest_paths": shortest_paths,
                "gromov_delta": gromov_delta,
                "mean_distance": mean_distance,
                "angle_sums": angle_sums,
                "mean_angle_sum": mean_angle_sum,
                "min_angle_sum": min_angle_sum,
                "max_angle_sum": max_angle_sum,
                "angle_sum_deviations": angle_sum_deviations,
                "triangle_inequality_violations": triangle_violations,
                "embedding_coords": coords2.tolist(),
                **({"embedding_coords_3d": coords3d.tolist()} if coords3d is not None else {}),
                "edge_weight_variance": edge_weight_variance,
                "event_nodes": event_nodes,
                "event_edges": event_edges,
                "lorentzian_dissimilarity": L.tolist(),
                "lorentzian_embedding": lorentzian_embedding.tolist()
            }, f, indent=2)

        print(f"Results saved to {output_path}")
        print(json.dumps({
            "spec": {**vars(args), "curvature": kappa, "custom_edges": custom_edges, "timesteps": args.timesteps},
            "uid": uid,
            "counts": None, # No counts for per-timestep simulation
            "mutual_information": mi_per_timestep,
            "distance_matrix_per_timestep": distmat_per_timestep,
            "edge_mi": {f"{u},{v}": val for (u, v), val in edge_mi.items()},
            "distance_matrix": distance_matrix.tolist(),
            "shortest_paths": shortest_paths,
            "gromov_delta": gromov_delta,
            "mean_distance": mean_distance,
            "angle_sums": angle_sums,
            "mean_angle_sum": mean_angle_sum,
            "min_angle_sum": min_angle_sum,
            "max_angle_sum": max_angle_sum,
            "angle_sum_deviations": angle_sum_deviations,
            "triangle_inequality_violations": triangle_violations,
            "embedding_coords": coords2.tolist(),
            **({"embedding_coords_3d": coords3d.tolist()} if coords3d is not None else {}),
            "edge_weight_variance": edge_weight_variance,
            "event_nodes": event_nodes,
            "event_edges": event_edges,
            "lorentzian_dissimilarity": L.tolist(),
            "lorentzian_embedding": lorentzian_embedding.tolist()
        }, indent=2))

        # Warn if any triangle angle sum is not > π or Gromov delta is not < 0.3 for spherical geometry
        if args.geometry == "spherical":
            if not all(x > np.pi for x in angle_sums):
                print("[WARNING] Not all triangle angle sums exceed π in spherical geometry!")
            if gromov_delta >= 0.3:
                print(f"[WARNING] Gromov delta is not below 0.3 (actual: {gromov_delta})!") 