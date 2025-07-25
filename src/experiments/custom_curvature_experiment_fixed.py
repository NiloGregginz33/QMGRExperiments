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
from tqdm import tqdm

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
from qiskit.circuit.library import CXGate

# Add error mitigation imports
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import Batch
from qiskit_ibm_runtime import Options
from qiskit_ibm_runtime import Session

# Custom JSON encoder to handle numpy types and booleans
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'bool':
            return bool(obj)
        return super().default(obj)

# Helper functions for extracting results from Sampler primitive
def extract_bitarray_from_primitive(result):
    """Extract bitarray from Sampler primitive result"""
    try:
        # For SamplerV2, the result has a different structure
        if hasattr(result, 'quasi_dists'):
            # Old Sampler format
            quasi_dists = result.quasi_dists
            if quasi_dists and len(quasi_dists) > 0:
                shots = result.metadata[0].get('shots', 1024)
                bitstrings = []
                for bitstring, prob in quasi_dists[0].items():
                    count = int(prob * shots)
                    for _ in range(count):
                        bitstrings.append(bitstring)
                return 'quasi_dists', bitstrings
        elif hasattr(result, '_pub_results'):
            # SamplerV2 format
            pub_result = result._pub_results[0]
            if hasattr(pub_result, 'data'):
                data = pub_result.data
                if hasattr(data, 'meas') and hasattr(data.meas, 'get_bitstrings'):
                    # This is the correct SamplerV2 format
                    bitstrings = data.meas.get_bitstrings()
                    return 'bitstrings', bitstrings
                elif hasattr(data, 'get_bitstrings'):
                    bitstrings = data.get_bitstrings()
                    return 'bitstrings', bitstrings
                elif hasattr(data, 'quasi_dists'):
                    # Alternative SamplerV2 format
                    quasi_dists = data.quasi_dists
                    if quasi_dists and len(quasi_dists) > 0:
                        shots = result.metadata[0].get('shots', 1024)
                        bitstrings = []
                        for bitstring, prob in quasi_dists[0].items():
                            count = int(prob * shots)
                            for _ in range(count):
                                bitstrings.append(bitstring)
                        return 'quasi_dists', bitstrings
        
        print(f"Could not extract bitstrings from result. Available attributes: {dir(result)}")
        if hasattr(result, '_pub_results'):
            print(f"Pub result attributes: {dir(result._pub_results[0])}")
            if hasattr(result._pub_results[0], 'data'):
                print(f"Data attributes: {dir(result._pub_results[0].data)}")
        return None, None
    except Exception as e:
        print(f"Error extracting bitarray: {e}")
        import traceback
        traceback.print_exc()
        return None, None

from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import CXGate
from qiskit.transpiler import PassManager

# Add command-line argument parsing
p = argparse.ArgumentParser(description="Run a custom curvature circuit")
p.add_argument("--num_qubits", type=int, default=3, help="Number of qubits")
p.add_argument("--topology", choices=["star","chain","ring","complete","triangulated","custom"], 
               default="triangulated", help="Entanglement pattern (triangulated recommended for angle-sum curvature)")
p.add_argument("--custom_edges", type=str, default=None, 
               help="Comma-separated 'u-v[:w]' pairs if topology=custom (e.g., '0-1:1.0,1-2:2.0,2-0:0.5')")
p.add_argument("--alpha", type=float, default=0.8, help="Decay rate for 'star' entangler")
p.add_argument("--weight", type=float, default=1.0, help="Uniform weight for 'chain'/'ring'")
p.add_argument("--gamma", type=float, default=0.3, help="Charge-injection strength")
p.add_argument("--sigma", type=float, default=None, help="Gaussian width for charge (default = num_qubits/2)")
p.add_argument("--init_angle", type=float, default=0.0, help="Initial Rx angle on each qubit")
p.add_argument("--init_angles", type=str, default=None, 
               help="Comma-separated list of initial Rx angles for each qubit (overrides --init_angle if provided)")
p.add_argument("--shots", type=int, default=1024, help="Number of measurement shots")
p.add_argument("--device", type=str, default="simulator", help="Execution device: simulator or IBM provider name")
p.add_argument("--geometry", type=str, default="hyperbolic", 
               choices=["euclidean", "spherical", "hyperbolic", "lorentzian"], help="Geometry type")
p.add_argument("--curvature", type=float, nargs='+', 
               default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], 
               help="Curvature parameter(s) κ for non-Euclidean geometries. Can pass multiple values for sweep.")
p.add_argument("--timesteps", type=int, default=5, help="Number of timesteps for evolution")
p.add_argument("--dimension", type=int, default=2, help="Spatial dimension for Regge calculus (2=triangles, 3=tetrahedra, etc.)")
p.add_argument("--mass_hinge", type=str, default=None, 
               help="Comma-separated indices for the hinge (e.g., '0,1,2') to place a mass at.")
p.add_argument("--mass_value", type=float, default=0.0, help="Value of the mass to place at the specified hinge.")
p.add_argument("--solve_regge", action="store_true", 
               help="Solve the dynamical Regge equations (stationary point of action) with constraints. (REQUIRED for time series of scalar curvature R(t))")
p.add_argument("--lorentzian", action="store_true", 
               help="Enable Lorentzian signature (timelike edges negative squared length)")
p.add_argument("--excite", action="store_true", help="Enable bulk excitation analysis (X gate on bulk point)")
p.add_argument("--fast", action="store_true", 
               help="Fast mode: skip expensive computations (geometric embedding, Lorentzian MDS, Regge evolution)")
p.add_argument("--strong_curvature", action="store_true", 
               help="Apply stronger curvature effects for cleaner negative-curvature signals")
p.add_argument("--charge_injection", action="store_true", 
               help="Enable charge injection for stronger bulk-boundary coupling (REQUIRED for stress-energy sourcing)")
p.add_argument("--charge_strength", type=float, default=1.0, help="Strength of charge injection (default: 1.0)")
p.add_argument("--charge_location", type=int, default=3, help="Location for charge injection (default: 3)")
p.add_argument("--spin_injection", action="store_true", 
               help="Enable spin injection for magnetic bulk-boundary coupling")
p.add_argument("--spin_strength", type=float, default=1.0, help="Strength of spin injection (default: 1.0)")
p.add_argument("--spin_location", type=int, default=3, help="Location for spin injection (default: 3)")
p.add_argument("--edge_floor", type=float, default=0.001, 
               help="Minimum edge length floor for Lorentzian solver (default: 0.001)")
p.add_argument("--compute_entropies", action="store_true", 
               help="Enable boundary entropy computation for RT relation testing (S(A) ∝ Area_RT(A)) (REQUIRED for dynamic entropy evolution)")
p.add_argument("--hyperbolic_triangulation", action="store_true", 
               help="Use proper hyperbolic triangulation circuit with RZZ gates and Trotterized evolution")
p.add_argument("--trotter_steps", type=int, default=4, 
               help="Number of Trotter steps per timestep for hyperbolic triangulation (default: 4)")
p.add_argument("--dt", type=float, default=0.1, help="Time step size for Trotter evolution (default: 0.1)")
p.add_argument("--analyze_curvature", action="store_true", 
               help="Enable entanglement-to-curvature analysis using MDS embedding")
p.add_argument("--gravitational_waves", action="store_true", help="Enable gravitational wave detection")
p.add_argument("--einstein_analog", action="store_true", help="Enable Einstein equation analogs")

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
    "star": _entangle_star,
    "chain": _entangle_chain,
    "ring": _entangle_ring,
}

# ─── Charge injection ─────────────────────────────────────────────────────────
def _apply_charge(qc, gamma, sigma=None):
    """Apply RZ phases γ exp[−(q/σ)²] on each qubit q."""
    n = qc.num_qubits
    if sigma is None:
        sigma = n/2
    for q in range(n):
        angle = gamma * np.exp(-(q/sigma)**2)
        qc.rz(angle, q)

def make_graph(topology: str, n: int, custom_edges: str = None, default_weight: float = 1.0) -> nx.Graph:
    """
    Return a NetworkX graph on n nodes for the given topology.
    - 'star', 'chain', 'ring', 'complete', 'triangulated'
    - 'custom' with custom_edges="0-1:1.0,1-3:2.0,2-3:0.5"
    """
    if topology == "star":
        return nx.star_graph(n-1)
    elif topology == "chain":
        return nx.path_graph(n)
    elif topology == "ring":
        return nx.cycle_graph(n)
    elif topology == "complete":
        return nx.complete_graph(n)
    elif topology == "triangulated":
        # Create a triangulated graph that supports angle-sum curvature
        G = nx.Graph()
        G.add_nodes_from(range(n))
        if n < 3:
            if n == 2:
                G.add_edge(0, 1, weight=default_weight)
            return G
        
        # Base ring structure
        for i in range(n):
            G.add_edge(i, (i+1) % n, weight=default_weight)
        
        # Add cross-connections to create triangles
        if n >= 4:
            # Connect every other vertex to create triangles
            for i in range(0, n, 2):
                G.add_edge(i, (i+2) % n, weight=default_weight)
        
        # Add additional connections for more triangulation
        if n >= 6:
            # Connect vertices with distance 3 to create more triangles
            for i in range(n):
                G.add_edge(i, (i+3) % n, weight=default_weight)
        
        # For n=7, this creates a rich triangulated structure with many triangles
        # Each vertex will have multiple triangles incident to it
        return G
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
def build_custom_circuit_layers(num_qubits, topology, custom_edges, alpha, weight, gamma, sigma, 
                                init_angle, geometry=None, curvature=None, log_edge_weights=False, 
                                timesteps=1, init_angles=None):
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
        if geometry in ("spherical", "hyperbolic") and curvature is not None and topology != "triangulated":
            # For non-Euclidean geometries, use custom edges with curvature-dependent weights
            # UNLESS using triangulated topology, which should preserve its structure
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
            # Use the specified topology (including triangulated) with curvature-adjusted weights
            if geometry in ("spherical", "hyperbolic") and curvature is not None and topology == "triangulated":
                # For triangulated topology with non-Euclidean geometry, adjust weights based on curvature
                base_weight = weight
                std_dev = base_weight * (curvature / 10)
                
                # Create triangulated graph first
                G = make_graph(topology, num_qubits, custom_edges, default_weight=base_weight)
                
                # Then adjust weights based on curvature
                edge_weights = {}
                for u, v in G.edges():
                    w = float(np.random.normal(loc=base_weight, scale=std_dev))
                    w = float(np.clip(w, 0.05, 1.0))
                    edge_weights[(u, v)] = w
                    qc.ryy(np.pi * w, u, v)
                
                if log_edge_weights and t == 0:
                    weights = list(edge_weights.values())
                    print(f"[LOG] Triangulated edge weights: {weights}")
                    print(f"[LOG] Edge weight variance: {np.var(weights)}")
                
                if t == 0:
                    qc._custom_edges_str = f"triangulated_with_{geometry}_curvature_{curvature}"
                    qc._edge_weight_variance = float(np.var(list(edge_weights.values())))
            
            else:
                # Standard case: use topology as specified
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
    
    # Add measurements to all circuits in the list
    for circ in circuits:
        circ.measure_all()
    
    return circuits, qc

def build_hyperbolic_triangulation_circuit(num_qubits, custom_edges, weight, gamma, sigma, 
                                          init_angle, geometry="hyperbolic", curvature=1.0, 
                                          timesteps=1, init_angles=None, trotter_steps=4, dt=0.1):
    """
    Build quantum circuit with proper hyperbolic triangulation using RZZ gates and Trotterized evolution.
    This implements the Hamiltonian H = Σ⟨i,j⟩ J_ij Z_i Z_j + Σ_i h_i X_i
    where J_ij are set by the edge weights of the hyperbolic triangulation.
    
    Minimal recipe:
    - RZZ gates entangle each pair according to hyperbolic graph
    - RX gates inject the "field" that drives dynamics
    - Multiple layers let the state pick out minimal surfaces in the bulk
    """
    circuits = []
    qc = QuantumCircuit(num_qubits)
    
    # 1) Initial state preparation
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
    
    # 2) Build hyperbolic triangulation graph and extract edges
    G = make_graph("triangulated", num_qubits, custom_edges, default_weight=weight)
    edges = list(G.edges())
    
    # 3) Set parameters for minimal recipe
    h = np.pi/4  # single-qubit field strength
    J = weight   # coupling constant from edge weights
    
    print(f"[LOG] Hyperbolic triangulation parameters:")
    print(f"[LOG] - dt = {dt}")
    print(f"[LOG] - h = {h} (transverse field)")
    print(f"[LOG] - J = {J} (coupling)")
    print(f"[LOG] - edges = {edges}")
    print(f"[LOG] - trotter_steps = {trotter_steps}")
    print(f"[LOG] - timesteps = {timesteps}")
    
    # 4) Define Trotter step function
    def trotter_step(qc):
        # 1) Apply ZZ couplings along triangulation edges
        for i, j in edges:
            qc.rzz(2 * J * dt, i, j)
        # 2) Apply X-field rotations
        for q in range(num_qubits):
            qc.rx(2 * h * dt, q)
    
    # 5) Build circuit with multiple Trotter steps per timestep
    for t in range(timesteps):
        print(f"[LOG] Building timestep {t+1}/{timesteps} with {trotter_steps} Trotter steps")
        # Apply multiple Trotter steps for this timestep
        for step in range(trotter_steps):
            trotter_step(qc)
        # Save circuit at this timestep
        circuits.append(qc.copy())
    
    # 6) Apply charge injection and measurement
    _apply_charge(qc, gamma, sigma)
    qc.measure_all()
    
    # Add measurements to all circuits
    for circ in circuits:
        circ.measure_all()
    
    # Store metadata
    qc._custom_edges_str = custom_edges
    qc._hyperbolic_triangulation = True
    qc._edges = edges
    qc._trotter_steps = trotter_steps
    qc._dt = dt
    qc._h = h
    qc._J = J
    
    return circuits, qc

# ─── Runner ───────────────────────────────────────────────────────────────────
def run_circuit(qc, shots, simulator, device_name):
    if simulator:
        backend = FakeBrisbane()
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        tqc = pm.run(qc)
        
        # Use the run function from CGPTFactory
        result = run(tqc, backend=backend, shots=shots)
        return result
    else:
        # Hardware execution
        service = QiskitRuntimeService()
        backend = service.get_backend(device_name)
        
        with Session(service=service, backend=backend) as session:
            sampler = Sampler(session=session)
            job = sampler.run([qc], shots=shots)
            result = job.result()
            return result

# Continue with the rest of the functions...
# (This is where the rest of your corrected code would go)

if __name__ == "__main__":
    print("Custom curvature experiment - Fixed version")
    print("This is a placeholder for the complete fixed implementation") 