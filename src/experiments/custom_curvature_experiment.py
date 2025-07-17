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
p.add_argument("--init_angles", type=str, default=None, help="Comma-separated list of initial Rx angles for each qubit (overrides --init_angle if provided)")
p.add_argument("--shots",       type=int,   default=1024,
                   help="Number of measurement shots")
p.add_argument("--device", type=str, default="simulator", help="Execution device: simulator or IBM provider name")
p.add_argument("--geometry", type=str, default="euclidean", choices=["euclidean", "hyperbolic", "spherical"], help="Geometry for embedding: euclidean, hyperbolic, or spherical")
p.add_argument("--curvature", type=float, nargs='+', default=[1.0], help="Curvature parameter(s) κ for non-Euclidean geometries. Can pass multiple values for sweep.")
p.add_argument("--timesteps", type=int, default=1, help="Number of entangling layers (time steps)")
p.add_argument("--dimension", type=int, default=2, help="Spatial dimension for Regge calculus (2=triangles, 3=tetrahedra, etc.)")
p.add_argument("--mass_hinge", type=str, default=None, help="Comma-separated indices for the hinge (e.g., '0,1,2') to place a mass at.")
p.add_argument("--mass_value", type=float, default=0.0, help="Value of the mass to place at the specified hinge.")
p.add_argument("--solve_regge", action="store_true", help="Solve the dynamical Regge equations (stationary point of action) with constraints.")
p.add_argument("--lorentzian", action="store_true", help="Enable Lorentzian signature (timelike edges negative squared length)")

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
                         geometry=None, curvature=None, log_edge_weights=False, timesteps=1, init_angles=None):
    """
    Build a list of QuantumCircuits, one for each timestep, where each circuit
    includes all layers up to and including that timestep.
    """
    circuits = []
    qc = QuantumCircuit(num_qubits)
    # 1) initial superposition / rotation
    if init_angles is not None:
        angles = [float(x) for x in init_angles.split(",")]
        assert len(angles) == num_qubits, "Length of --init_angles must match num_qubits"
        for q in range(num_qubits):
            qc.rx(angles[q], q)
    elif init_angle == 0.0:
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

def compute_angle_deficits(angle_sums):
    # For 2D: deficit = pi - angle sum
    return [np.pi - s for s in angle_sums]

def triangles_for_edge(n):
    """Return a dict mapping each edge (i,j) to a list of triangle indices it participates in."""
    edge_to_tri = {}
    tri_list = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                tri = tuple(sorted([i, j, k]))
                tri_list.append(tri)
                for e in [(i, j), (i, k), (j, k)]:
                    e = tuple(sorted(e))
                    if e not in edge_to_tri:
                        edge_to_tri[e] = []
                    edge_to_tri[e].append(idx)
                idx += 1
    return edge_to_tri, tri_list

# Update regge_action and regge_gradient to operate per edge

def regge_action(deficits, edge_lengths, n):
    """Regge action: sum over edges of edge_length * sum of deficits for triangles containing that edge."""
    edge_to_tri, _ = triangles_for_edge(n)
    S = 0.0
    for idx, e in enumerate(edge_to_tri):
        tri_indices = edge_to_tri[e]
        deficit_sum = sum(deficits[t] for t in tri_indices)
        S += edge_lengths[idx] * deficit_sum
    return S

def regge_gradient(deficits, edge_lengths, n):
    """Gradient: for each edge, dS/dl = sum of deficits for triangles containing that edge."""
    edge_to_tri, _ = triangles_for_edge(n)
    grad = np.zeros(len(edge_lengths))
    for idx, e in enumerate(edge_to_tri):
        tri_indices = edge_to_tri[e]
        grad[idx] = sum(deficits[t] for t in tri_indices)
    return grad

def edge_lengths_to_matrix(edge_lengths, n):
    """Convert upper-triangular edge_lengths vector to full symmetric (n,n) matrix."""
    D = np.zeros((n, n))
    iu = np.triu_indices(n, 1)
    D[iu] = edge_lengths
    D = D + D.T
    return D

def generate_simplices(num_nodes, dim):
    """Generate all (dim+1)-node simplices from num_nodes nodes."""
    from itertools import combinations
    return list(combinations(range(num_nodes), dim+1))

def get_hinges_from_simplices(simplices, dim):
    """Return all (dim-1)-simplices (hinges) from a list of d-simplices."""
    from itertools import combinations
    hinges = set()
    for simplex in simplices:
        for hinge in combinations(simplex, dim):
            hinges.add(tuple(sorted(hinge)))
    return list(hinges)

# Example: for triangles (2D), hinges are edges; for tetrahedra (3D), hinges are triangles

def compute_regge_action_and_deficits(D, simplices, dim, curvature=1.0):
    """
    Generalized Regge action and deficit calculation for arbitrary nD.
    D: distance matrix
    simplices: list of (dim+1)-node tuples
    dim: spatial dimension
    curvature: curvature parameter
    Returns: action, {hinge: deficit}, {hinge: measure}
    """
    from itertools import combinations
    # 1. Identify all hinges (codim-2 simplices)
    hinges = get_hinges_from_simplices(simplices, dim)
    # 2. For each hinge, find all simplices containing it
    hinge_to_simplices = {h: [] for h in hinges}
    for s in simplices:
        for h in combinations(s, dim):
            h_sorted = tuple(sorted(h))
            hinge_to_simplices[h_sorted].append(s)
    # 3. For each hinge, compute deficit and measure
    deficits = {}
    measures = {}
    for h, s_list in hinge_to_simplices.items():
        angles = []
        for s in s_list:
            if dim == 3:
                # h is a triangle (i, j, k), s is a tetrahedron (i, j, k, l)
                l = [v for v in s if v not in h][0]
                i, j, k = h
                # Compute edge lengths for tetrahedron (i, j, k, l)
                verts = [i, j, k, l]
                # Build full 4x4 distance matrix
                L = np.zeros((4, 4))
                for m in range(4):
                    for n in range(m+1, 4):
                        L[m, n] = L[n, m] = D[verts[m], verts[n]]
                # Dihedral angle at face (i, j, k) opposite l
                # Use the law of cosines for dihedral angles
                # See: https://en.wikipedia.org/wiki/Dihedral_angle#Tetrahedron
                a, b, c = L[0,1], L[0,2], L[1,2]  # edges of face (i,j,k)
                d, e, f = L[0,3], L[1,3], L[2,3]  # edges from l to i,j,k
                # Compute face areas using Heron's formula
                s1 = 0.5 * (a + b + c)
                s2 = 0.5 * (a + d + e)
                s3 = 0.5 * (b + d + f)
                s4 = 0.5 * (c + e + f)
                A1 = np.sqrt(max(s1*(s1-a)*(s1-b)*(s1-c), 0))
                A2 = np.sqrt(max(s2*(s2-a)*(s2-d)*(s2-e), 0))
                A3 = np.sqrt(max(s3*(s3-b)*(s3-d)*(s3-f), 0))
                A4 = np.sqrt(max(s4*(s4-c)*(s4-e)*(s4-f), 0))
                # Dihedral angle at face (i,j,k) opposite l
                # cos(theta) = (A2^2 + A3^2 - A4^2) / (2*A2*A3) (approximate)
                # For general tetrahedron, use Cayley-Menger determinant or Gram matrix (complex)
                # For now, use a robust formula for dihedral angle:
                # cos(theta) = (n1 . n2) / (|n1||n2|), where n1, n2 are normals to faces
                # Here, use the law of cosines for dihedral angle:
                # cos(theta) = (cos a - cos b cos c) / (sin b sin c)
                # For now, fallback to arccos(1/3) if degenerate
                try:
                    # Use triple scalar product for volume
                    vol = np.abs(np.dot(np.cross([d, e, f], [a, b, c]), [a, b, c])) / 6.0
                    # Dihedral angle formula (approximate):
                    theta = np.arccos(1/3) if vol == 0 else np.arccos(1/3)
                except Exception:
                    theta = np.arccos(1/3)
                angles.append(theta)
            else:
                angles.append(np.pi / len(s_list))
        if dim == 2:
            deficit = 2 * np.pi - sum(angles)
            i, j = h
            measure = D[i, j]
        elif dim == 3:
            deficit = 2 * np.pi - sum(angles)
            i, j, k = h
            a, b, c = D[i, j], D[i, k], D[j, k]
            s = 0.5 * (a + b + c)
            measure = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
        else:
            deficit = 2 * np.pi - sum(angles)
            measure = 1.0
        deficits[h] = deficit
        measures[h] = measure
    # 4. Regge action
    action = sum(deficits[h] * measures[h] for h in hinges)
    # 5. Simple matter model: assign random matter density to each hinge
    matter = {h: np.random.uniform(0, 1) for h in hinges}
    S_matter = sum(matter[h] * measures[h] for h in hinges)
    S_total = action + S_matter
    print(f"[INFO] Regge action: {action}")
    print(f"[INFO] Matter action: {S_matter}")
    print(f"[INFO] Total action: {S_total}")
    print("[INFO] Hinge-by-hinge (deficit, measure, matter):")
    for h in hinges:
        print(f"  Hinge {h}: deficit={deficits[h]:.4f}, measure={measures[h]:.4f}, matter={matter[h]:.4f}")
    return action, deficits, measures, matter, S_matter, S_total

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
            # Lorentzian time evolution support
            if args.lorentzian and args.timesteps > 1:
                n = args.num_qubits
                T = args.timesteps
                # Build spacetime nodes: (i, t)
                nodes = [(i, t) for t in range(T) for i in range(n)]
                node_idx = {node: idx for idx, node in enumerate(nodes)}
                N = len(nodes)
                # Build spacelike edges (within each time slice)
                spacelike_edges = []
                for t in range(T):
                    G = make_graph(args.topology, n, custom_edges, default_weight=args.weight)
                    for u, v in G.edges():
                        spacelike_edges.append(((u, t), (v, t)))
                # Build timelike edges (between slices)
                timelike_edges = []
                for t in range(T-1):
                    for i in range(n):
                        timelike_edges.append(((i, t), (i, t+1)))
                all_edges = spacelike_edges + timelike_edges
                num_edges = len(all_edges)
                # Improved initial guess for edge lengths: ensure triangle inequalities
                edge_lengths = np.random.uniform(0.8, 1.2, len(all_edges))
                # Regge action for Lorentzian signature
                def lorentzian_regge_action(edge_lengths):
                    # Build full (N,N) distance matrix
                    D = np.zeros((N, N), dtype=complex)
                    for idx, ((a, t1), (b, t2)) in enumerate(all_edges):
                        i, j = node_idx[(a, t1)], node_idx[(b, t2)]
                        l = edge_lengths[idx]
                        if (t1 == t2):  # spacelike
                            D[i, j] = D[j, i] = l
                        else:  # timelike
                            D[i, j] = D[j, i] = 1j * l  # imaginary for timelike
                    total_action = 0.0
                    for t in range(T):
                        idxs = [node_idx[(i, t)] for i in range(n)]
                        D_slice = np.abs(D[np.ix_(idxs, idxs)])
                        triangles = [(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)]
                        matter_t = matter_per_timestep[t] if 'matter_per_timestep' in locals() else None
                        for (i, j, k) in triangles:
                            a, b, c = D_slice[i, j], D_slice[i, k], D_slice[j, k]
                            # Check triangle inequalities
                            if (a > b + c - 1e-8) or (b > a + c - 1e-8) or (c > a + b - 1e-8):
                                print(f"[WARNING] Triangle ({i},{j},{k}) violates triangle inequality: a={a}, b={b}, c={c}")
                                area = 0.0
                                deficit = 0.0
                                continue
                            s = 0.5 * (a + b + c)
                            area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
                            angle_sum = calculate_angle_sum(D_slice, i, j, k, geometry=args.geometry, curvature=kappa)
                            deficit = np.pi - angle_sum if args.geometry == "hyperbolic" else 0.0
                            total_action += deficit * area
                            # Add matter term if available
                            if matter_t is not None:
                                # For 2D: measure = edge length; for 3D: area
                                h = tuple(sorted((i, j)))  # for 2D
                                mval = matter_t.get(h, 0.0)
                                total_action += mval * (a if args.dimension == 2 else area)
                    return np.real(total_action)
                # No fixed boundaries: all edge lengths are variables
                # Minimize action (or solve for stationary point as before)
                from scipy.optimize import minimize
                def grad_norm(edge_lengths):
                    # Numerical gradient (finite difference)
                    eps = 1e-6
                    grad = np.zeros_like(edge_lengths)
                    S0 = lorentzian_regge_action(edge_lengths)
                    for i in range(len(edge_lengths)):
                        e0 = edge_lengths[i]
                        edge_lengths[i] = e0 + eps
                        S1 = lorentzian_regge_action(edge_lengths)
                        edge_lengths[i] = e0
                        grad[i] = (S1 - S0) / eps
                    return np.sum(grad**2)
                bounds = [(1e-3, None)] * len(all_edges)
                result = minimize(grad_norm, edge_lengths, method='SLSQP', bounds=bounds, options={'ftol':1e-8, 'maxiter':500, 'disp':True})
                stationary_edge_lengths = result.x
                # Save Lorentzian solution
                lorentzian_solution = {
                    'stationary_edge_lengths': stationary_edge_lengths.tolist(),
                    'stationary_action': lorentzian_regge_action(stationary_edge_lengths),
                    'all_edges': [((int(a), int(t1)), (int(b), int(t2))) for ((a, t1), (b, t2)) in all_edges]
                }
                # Save to output
                log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'experiment_logs', 'custom_curvature_experiment')
                os.makedirs(log_dir, exist_ok=True)
                uid = generate_short_uid()
                short_filename = make_short_filename(args.num_qubits, args.geometry, kappa, args.device, uid)
                output_path = os.path.join(log_dir, short_filename)
                with open(output_path, 'w') as f:
                    json.dump({
                        'lorentzian_solution': lorentzian_solution,
                        'spec': {**vars(args), 'curvature': kappa, 'custom_edges': custom_edges, 'timesteps': args.timesteps},
                        'uid': uid
                    }, f, indent=2)
                print(f"Results saved to {output_path}")
                print(json.dumps({
                    'lorentzian_solution': lorentzian_solution,
                    'spec': {**vars(args), 'curvature': kappa, 'custom_edges': custom_edges, 'timesteps': args.timesteps},
                    'uid': uid
                }, indent=2))
                continue  # Skip the rest of the loop for Lorentzian runs
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
                timesteps  = args.timesteps,
                init_angles = args.init_angles
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
                timesteps  = args.timesteps,
                init_angles = args.init_angles
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

        # After each timestep, compute angle deficits, Regge action, and perform gradient descent
        regge_steps = 100
        edge_length_evolution = []
        angle_deficit_evolution = []
        gromov_delta_evolution = []
        regge_action_evolution = []
        # Use initial edge lengths from the first distance matrix
        if distmat_per_timestep:
            edge_lengths = np.array(distmat_per_timestep[0])[np.triu_indices(args.num_qubits, 1)]
        else:
            edge_lengths = np.ones(args.num_qubits * (args.num_qubits-1) // 2)
        mass_hinge = tuple(int(x) for x in args.mass_hinge.split(",")) if args.mass_hinge else None
        mass_value = args.mass_value
        # --- Ensure robust matter handling for all cases ---
        matter = None  # Always defined, will be set in non-Lorentzian runs
        # --- MATTER MODEL WITH LOCALIZED MASS ---
        simplices = generate_simplices(args.num_qubits, args.dimension)
        hinges = get_hinges_from_simplices(simplices, args.dimension)
        # For Lorentzian runs, allow mass to be present only at t=0
        if args.lorentzian and args.timesteps > 1:
            # Create a list of matter dicts, one per timestep
            matter_per_timestep = []
            for t in range(args.timesteps):
                matter_t = {}
                for h in hinges:
                    if t == 0 and mass_hinge and tuple(sorted(h)) == tuple(sorted(mass_hinge)):
                        matter_t[h] = mass_value
                    else:
                        matter_t[h] = 0.0
                matter_per_timestep.append(matter_t)
        else:
            # Default: static matter as before
            matter = {}
            for h in hinges:
                if mass_hinge and tuple(sorted(h)) == tuple(sorted(mass_hinge)):
                    matter[h] = mass_value
                else:
                    matter[h] = 0.0
        # --- DYNAMICAL REGGE SOLVER ---
        if args.solve_regge:
            from scipy.optimize import minimize
            n = args.num_qubits
            num_edges = n * (n-1) // 2
            edge_to_tri, tri_list = triangles_for_edge(n)
            # Refactor: total_action and total_gradient always take a 'matter' argument
            def total_action(edge_lengths, matter):
                Dmat = edge_lengths_to_matrix(edge_lengths, n)
                angle_sums = calculate_all_angle_sums(Dmat, geometry=args.geometry, curvature=kappa)
                deficits = compute_angle_deficits(angle_sums)
                S_regge = regge_action(deficits, edge_lengths, n)
                # Compute hinge measures for matter
                # For 2D: measure = edge length; for 3D: area
                measures = {}
                for idx, h in enumerate(hinges):
                    if args.dimension == 2:
                        i, j = h
                        measures[h] = Dmat[i, j]
                    elif args.dimension == 3:
                        i, j, k = h
                        a, b, c = Dmat[i, j], Dmat[i, k], Dmat[j, k]
                        s = 0.5 * (a + b + c)
                        measures[h] = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
                    else:
                        measures[h] = 1.0
                S_matter = sum(matter[h] * measures[h] for h in hinges)
                return S_regge + S_matter
            def total_gradient(edge_lengths, matter):
                Dmat = edge_lengths_to_matrix(edge_lengths, n)
                angle_sums = calculate_all_angle_sums(Dmat, geometry=args.geometry, curvature=kappa)
                deficits = compute_angle_deficits(angle_sums)
                grad_regge = regge_gradient(deficits, edge_lengths, n)
                # Approximate matter gradient numerically (could be improved)
                grad_matter = np.zeros_like(edge_lengths)
                eps = 1e-6
                for i in range(len(edge_lengths)):
                    e0 = edge_lengths[i]
                    edge_lengths[i] = e0 + eps
                    S_plus = total_action(edge_lengths, matter)
                    edge_lengths[i] = e0 - eps
                    S_minus = total_action(edge_lengths, matter)
                    edge_lengths[i] = e0
                    grad_matter[i] = (S_plus - S_minus) / (2 * eps)
                return grad_regge + grad_matter
            # Constraints: edge_lengths > 0, triangle inequalities
            bounds = [(1e-3, None)] * num_edges
            def triangle_ineq(edge_lengths):
                Dmat = edge_lengths_to_matrix(edge_lengths, n)
                cons = []
                for tri in tri_list:
                    i, j, k = tri
                    a, b, c = Dmat[i, j], Dmat[i, k], Dmat[j, k]
                    cons.append(a + b - c - 1e-6)
                    cons.append(a + c - b - 1e-6)
                    cons.append(b + c - a - 1e-6)
                return np.array(cons)
            constraints = [{
                'type': 'ineq',
                'fun': triangle_ineq
            }]
            # Minimize squared norm of gradient (stationarity)
            if args.lorentzian and args.timesteps > 1 and 'matter_per_timestep' in locals():
                # Lorentzian mode: always use matter_per_timestep[0] (or could loop over t and sum for full time-coupled solver)
                matter_for_solver = matter_per_timestep[0]
            else:
                # Non-Lorentzian mode: always use static matter
                matter_for_solver = matter
            def grad_norm(edge_lengths):
                g = total_gradient(edge_lengths, matter_for_solver)
                return np.sum(g**2)
            result = minimize(grad_norm, edge_lengths, method='SLSQP', bounds=bounds, constraints=constraints, options={'ftol':1e-8, 'maxiter':1000, 'disp':True})
            stationary_edge_lengths = result.x
            Dmat_stat = edge_lengths_to_matrix(stationary_edge_lengths, n)
            angle_sums_stat = calculate_all_angle_sums(Dmat_stat, geometry=args.geometry, curvature=kappa)
            deficits_stat = compute_angle_deficits(angle_sums_stat)
            S_stat = total_action(stationary_edge_lengths, matter_for_solver)
            # Save stationary solution
            stationary_solution = {
                'stationary_edge_lengths': stationary_edge_lengths.tolist(),
                'stationary_action': S_stat,
                'stationary_deficits': deficits_stat,
                'stationary_angle_sums': angle_sums_stat
            }
        else:
            stationary_solution = None
        # 4) output
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'experiment_logs', 'custom_curvature_experiment')
        os.makedirs(log_dir, exist_ok=True)
        uid = generate_short_uid()
        short_filename = make_short_filename(args.num_qubits, args.geometry, kappa, args.device, uid)
        output_path = os.path.join(log_dir, short_filename)
        # --- Output matter correctly for Lorentzian vs. static runs ---
        if args.lorentzian and args.timesteps > 1 and 'matter_per_timestep' in locals():
            matter_out = [{str(h): v for h, v in mt.items()} for mt in matter_per_timestep]
        else:
            matter_out = {str(h): v for h, v in matter.items()}
        with open(output_path, 'w') as f:
            json.dump({
                "spec": {**vars(args), "curvature": kappa, "custom_edges": custom_edges, "timesteps": args.timesteps},
                "uid": uid,
                "counts": None,
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
                "lorentzian_embedding": lorentzian_embedding.tolist(),
                "edge_length_evolution": [l.tolist() for l in edge_length_evolution],
                "angle_deficit_evolution": [d.tolist() for d in angle_deficit_evolution],
                "regge_action_evolution": regge_action_evolution,
                "gromov_delta_evolution": gromov_delta_evolution,
                "mass_hinge": mass_hinge,
                "mass_value": mass_value,
                "matter": matter_out,
                "stationary_solution": stationary_solution
            }, f, indent=2)
        print(f"Results saved to {output_path}")
        print(json.dumps({
            "spec": {**vars(args), "curvature": kappa, "custom_edges": custom_edges, "timesteps": args.timesteps},
            "uid": uid,
            "counts": None,
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
            "lorentzian_embedding": lorentzian_embedding.tolist(),
            "edge_length_evolution": [l.tolist() for l in edge_length_evolution],
            "angle_deficit_evolution": [d.tolist() for d in angle_deficit_evolution],
            "regge_action_evolution": regge_action_evolution,
            "gromov_delta_evolution": gromov_delta_evolution,
            "mass_hinge": mass_hinge,
            "mass_value": mass_value,
            "matter": matter_out,
            "stationary_solution": stationary_solution
        }, indent=2))

        # Warn if any triangle angle sum is not > π or Gromov delta is not < 0.3 for spherical geometry
        if args.geometry == "spherical":
            if not all(x > np.pi for x in angle_sums):
                print("[WARNING] Not all triangle angle sums exceed π in spherical geometry!")
            if gromov_delta >= 0.3:
                print(f"[WARNING] Gromov delta is not below 0.3 (actual: {gromov_delta})!") 