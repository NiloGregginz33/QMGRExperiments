from __future__ import annotations
import sys
import os
import argparse
import logging
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, entropy
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
import numpy as np
import json
from typing import Dict, List, Tuple, Union, Optional, Callable, Literal
from qiskit import QuantumCircuit, transpile
from scipy.optimize import minimize
import ast

# Ensure build_circuit is defined at the top of the script

def build_circuit(n: int,
                  theta_dict: dict[tuple[int, int], float],
                  init: callable | None = None,
                  measure: bool = False) -> QuantumCircuit:
    """
    Build a Qiskit QuantumCircuit given a θ-dictionary:
        θ_dict[(i,j)] = CP phase applied with control=i, target=j  (radians)

    Parameters
    ----------
    n : int
        Total number of qubits.
    theta_dict : dict[(int,int), float]
        Mapping edge → phase angle (radians).
    init : callable[[QuantumCircuit], None] | None
        Optional callback that takes the circuit and adds gates BEFORE the CP
        network (e.g. custom initial state).  If None, default is Hadamard on
        every qubit.
    measure : bool
        If True, append computational-basis measurements to classical bits of
        the same size.

    Returns
    -------
    qc : QuantumCircuit
    """
    qc = QuantumCircuit(n, n if measure else 0)

    # --- 1.  state preparation -------------------------------------------
    if init is None:
        for q in range(n):
            qc.h(q)
    else:
        init(qc)

    # --- 2.  spacetime / geometry layer ----------------------------------
    for (ctrl, tgt), phase in theta_dict.items():
        # keep angles inside [0, 2π) to avoid numerical wrap-arounds
        qc.cp(float(phase) % (2 * np.pi), ctrl, tgt)

    # --- 3.  optional measurement ----------------------------------------
    if measure:
        qc.barrier()
        qc.measure(range(n), range(n))

    return qc

# Add the curvature helper function below build_circuit
def curvature(dm, edge_list, entropy_fn=lambda dm: entropy(dm, base=2)):
    """
    Return the information-geometric curvature R_i for every qubit i.

    Parameters
    ----------
    dm : DensityMatrix
        Global density matrix of the n-qubit state.
    edge_list : list[(int,int)]
        Graph edges (i,j).  Each contributes the mutual information I_ij
        to the two incident nodes.
    entropy_fn : callable
        Function that returns entropy of a DensityMatrix (von Neumann by
        default, but you can pass the renyi2 helper).

    Returns
    -------
    dict[int,float]
        Mapping qubit index -> curvature value R_i.
    """
    n = dm.num_qubits
    R = {i: 0.0 for i in range(n)}

    for (i, j) in edge_list:
        dm_ij = partial_trace(dm, [q for q in range(n) if q not in (i, j)])

        # single-qubit entropies
        Si = entropy_fn(partial_trace(dm, [q for q in range(n) if q != i]))
        Sj = entropy_fn(partial_trace(dm, [q for q in range(n) if q != j]))

        # two-qubit entropy
        Sij = entropy_fn(dm_ij)

        # mutual information on that edge
        Iij = Si + Sj - Sij

        # add to curvature of the two incident nodes
        R[i] += Iij
        R[j] += Iij

    return R

# Function to create a quantum circuit with any CP angle combination
def create_cp_circuit(num_qubits: int, cp_angles: dict[tuple[int, int], float]) -> QuantumCircuit:
    """
    Create a quantum circuit with specified CP angles.

    Parameters
    ----------
    num_qubits : int
        Total number of qubits in the circuit.
    cp_angles : dict[(int, int), float]
        Dictionary mapping qubit pairs to CP angles (in radians).

    Returns
    -------
    qc : QuantumCircuit
        The constructed quantum circuit with CP gates applied.
    """
    qc = QuantumCircuit(num_qubits)

    # Apply CP gates according to the specified angles
    for (ctrl, tgt), angle in cp_angles.items():
        qc.cp(angle, ctrl, tgt)

    return qc

# Example usage:
# cp_angles = {(0, 1): 0.5, (1, 2): 1.0}
# qc = create_cp_circuit(3, cp_angles)
# print(qc)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Custom Spacetime Qiskit Experiment')
parser.add_argument('--num_qubits', type=int, required=True, help='Number of qubits')
parser.add_argument('--edge_list', type=str, required=True, help='Edge list for the graph')
parser.add_argument('--target_curvature', type=str, required=True, help='Target curvature')
parser.add_argument('--device', type=str, default='simulator', help='Device to run the experiment on')
args = parser.parse_args()

# 4 Canonical parse of CLI inputs (place near the top)
edge_list = ast.literal_eval(args.edge_list)
target_curvature = ast.literal_eval(args.target_curvature)

# 2 Wire the analytic seed before the baseline snapshot
# ---- analytic warm-start --------------------------------
theta = dict(θ_seed)  # 0.374, 0.224, ...

# ---- baseline curvature  -------------------------------
dm0 = DensityMatrix(Statevector.from_instruction(build_circuit(args.num_qubits, theta)))
R_baseline = curvature(dm0, edge_list)  # should be O(0.1–0.3) on endpoints

# 1 Check the three critical inputs to loss_function
# Add print statements inside loss_function to verify inputs

def loss_function(theta_dict, target_curvature):
    print(list(theta_dict.values())[:4])  # Debug: Check theta_dict values
    dm = DensityMatrix(Statevector.from_instruction(build_circuit(args.num_qubits, theta_dict)))
    R = curvature(dm, edge_list)
    R = {i: R_baseline[i] - R[i] for i in R}
    return sum((R[i] - target_curvature[i])**2 for i in R)

# 3 Confirm the optimiser loop actually runs
for t in range(1, steps + 1):
    if t % 100 == 0:
        print(f"step {t}: θ(0,1) = {theta[(0,1)]:.3f}, loss = {loss_function(theta, target_curvature):.4f}")
    # ... existing code for parameter-shift and Adam optimizer ...

# 5 Sanity print right before saving results
print("Final θ:", theta)
print("Final loss:", loss_function(theta, target_curvature))

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# ----- analytic small-angle seed ---------------------------------------
T = [-0.2, -0.1, 0.0, 0.1, 0.2]
θ_seed = {
    (0,1): np.sqrt(abs(T[0])+abs(T[1]))/np.sqrt(2),   # ≈ 0.374
    (1,2): np.sqrt(abs(T[1])+abs(T[2]))/np.sqrt(2),   # ≈ 0.224
    (2,3): np.sqrt(abs(T[2])+abs(T[3]))/np.sqrt(2),   # ≈ 0.224
    (3,4): np.sqrt(abs(T[3])+abs(T[4]))/np.sqrt(2)    # ≈ 0.374
}

# Parameter-shift gradient + Adam optimizer
shift = np.pi/2         # exact for CP gates
β1, β2 = 0.9, 0.999
m, v   = {e:0 for e in edge_list}, {e:0 for e in edge_list}
eps    = 1e-8
lr0, lr_decay = 0.05, 0.92
steps        = 1200

# 1. analytic warm-start
theta = dict(θ_seed)

# 2. baseline curvature for signed loss
dm0 = DensityMatrix(Statevector.from_instruction(build_circuit(args.num_qubits, theta)))
R_baseline = curvature(dm0, edge_list, lambda dm: entropy(dm, base=2))

# 3. signed MSE loss – define *here*, before the optimiser loop
# Update loss_function to accept target_curvature as an argument
def loss_function(theta_dict, target_curvature):
    dm = DensityMatrix(Statevector.from_instruction(build_circuit(args.num_qubits, theta_dict)))
    R  = curvature(dm, edge_list)
    R  = {i: R_baseline[i] - R[i] for i in R}
    return sum((R[i] - target_curvature[i])**2 for i in R)

# 4. parameter-shift + Adam loop
for t in range(1, steps + 1):
    lr_t = lr0 * (lr_decay ** (t//120))

    # --- exact gradient -------------------------------------------------
    grad = {}
    for e in edge_list:
        θp, θm = dict(theta), dict(theta)
        θp[e] += shift
        θm[e] -= shift
        grad[e] = 0.5 * (loss_function(θp, target_curvature) - loss_function(θm, target_curvature))

    # --- Adam update ----------------------------------------------------
    for e in edge_list:
        m[e] = β1*m[e] + (1-β1)*grad[e]
        v[e] = β2*v[e] + (1-β2)*grad[e]**2
        m_hat = m[e]/(1-β1**t)
        v_hat = v[e]/(1-β2**t)
        theta[e] -= lr_t * m_hat/(np.sqrt(v_hat)+eps)

# --- helper: Rényi-2 for a *DensityMatrix* ----------
def renyi2(dm: DensityMatrix) -> float:
    return -np.log2(np.real(np.trace(dm.data @ dm.data)))

# -------------- main routine ------------------------
def generate_spacetime(n: int,
                       edge_list: List[Tuple[int, int]],
                       T_target: Dict[int, float],
                       *,
                       steps: int = 200,
                       lr: float = 0.05,
                       eps: float = 1e-3,
                       entropy_order: Literal[1, 2] = 1,
                       backend_sim: str | None = "aer_simulator_statevector",
                       init_state: Callable[[QuantumCircuit], None] | None = None,
                       seed: Optional[int] = None,
                       transpile_opts: Optional[dict] = None,
                       export_backend=None,
                       shots: int = 8192) -> Tuple[Dict, QuantumCircuit]:
    """
    Optimises θ_ij so that the information-geometric curvature R_i matches T_target.
    Returns (theta_dict, final QuantumCircuit).  If *export_backend* is given,
    the circuit is also executed there and the Job object is attached as
    qc.metadata['job'].

    • entropy_order = 1 uses von Neumann entropy (as before)
    • entropy_order = 2 uses Rényi-2 (−log₂ Tr(ρ²))
    """
    rng = np.random.default_rng(seed)
    theta = {edge: 0.1 * rng.uniform(0.8, 1.2) for edge in edge_list}

    # choose entropy functional ------------------------------------------------
    _entropy = (lambda dm: entropy(dm, base=2)) if entropy_order == 1 else renyi2

    # fast statevector simulator for gradients ---------------------------------
    aer = FakeBrisbane()

    def density_matrix(th_dict):
        qc = build_circuit(n, th_dict)
        sv = Statevector.from_instruction(qc)
        return DensityMatrix(sv)

    # ---------------- gradient-descent loop -----------------------------------
    for _ in range(steps):
        dm = density_matrix(theta)

        # curvature R_i
        R = {i: 0.0 for i in range(n)}
        for (i, j) in edge_list:
            dm_ij = partial_trace(dm, [q for q in range(n) if q not in (i, j)])
            Si  = _entropy(partial_trace(dm, [q for q in range(n) if q != i]))
            Sj  = _entropy(partial_trace(dm, [q for q in range(n) if q != j]))
            Sij = _entropy(dm_ij)
            Iij = Si + Sj - Sij
            R[i] += Iij
            R[j] += Iij
        # loss
        L = sum((R[i] - T_target[i]) ** 2 for i in range(n))

        # finite-difference gradient
        grad = {}
        for edge in edge_list:
            orig = theta[edge]
            for sign in (+1, -1):
                theta[edge] = orig + sign * eps
                dm_d = density_matrix(theta)
                R_d = {k: 0.0 for k in range(n)}
                for (a, b) in edge_list:
                    dm_ab = partial_trace(dm_d, [q for q in range(n) if q not in (a, b)])
                    Sa  = _entropy(partial_trace(dm_d, [q for q in range(n) if q != a]))
                    Sb  = _entropy(partial_trace(dm_d, [q for q in range(n) if q != b]))
                    Sab = _entropy(dm_ab)
                    Iab = Sa + Sb - Sab
                    R_d[a] += Iab
                    R_d[b] += Iab
                if sign == +1:
                    Lp = sum((R_d[k] - T_target[k]) ** 2 for k in range(n))
                else:
                    Lm = sum((R_d[k] - T_target[k]) ** 2 for k in range(n))
            grad[edge] = (Lp - Lm) / (2 * eps)
            theta[edge] = orig

        # update
        for edge in edge_list:
            theta[edge] -= lr * grad[edge]

    # ---------------- build final circuit -------------------------------------
    qc_final = build_circuit(n, theta)

    # optional hardware / backend execution (no tomography required) ----------
    if export_backend is not None:
        qc_run = transpile(qc_final, export_backend, **(transpile_opts or {}))
        job = export_backend.run(qc_run, shots=shots)
        qc_final.metadata = qc_final.metadata or {}
        qc_final.metadata["job"] = job

    return theta, qc_final

# --- main routine: optim_signed_curvature ----
INIT_ANGLE = 0.1          # original baseline
STEPS      = 1200
LR0        = 0.05
LR_DECAY   = 0.92

# Use Adam optimizer
# optimizer = ADAM(maxiter=STEPS, lr=LR0)

# Initialize Adam parameters
beta1 = 0.9
beta2 = 0.999
shift = np.pi / 2

# Ensure edge_list is used consistently in functions and optimizer
# Assuming edge_list is passed as an argument or defined earlier in the script

# Example definition (replace with actual source of edge_list)
# edge_list = [(0,1), (1,2), (2,3), (3,4)]  # Define edge_list if not already defined

def optim_signed_curvature(n, edge_list, T_target,
                           steps=STEPS, lr=LR0, eps=1e-3, seed=None):
    rng = np.random.default_rng(seed)
    theta_init = {e: 0.05 for e in edge_list}  # uniform tiny seed
    entropy_fn = lambda dm: entropy(dm, base=2)

    # Initialize momentum and velocity inside the function
    m = {e: 0 for e in edge_list}
    v = {e: 0 for e in edge_list}

    # Initialize every edge to the same small phase, e.g., θ=0.1 rad
    # Assuming theta_init is a dictionary of edge phases

    def initialize_phases(edges):
        return {edge: 0.1 for edge in edges}

    # Compute the baseline curvature R_i(0)
    def compute_baseline_curvature(theta_init):
        # Placeholder for actual curvature computation
        # Replace with actual logic to compute curvature
        return {edge: 0.0 for edge in theta_init}

    # Quick feasibility scan
    edges = theta_init.keys()
    theta_init = initialize_phases(edges)
    R0 = compute_baseline_curvature(theta_init)
    max_R0 = max(abs(R) for R in R0.values())

    # Check to prevent division by zero
    if max_R0 == 0:
        print("Warning: Baseline curvature is zero, skipping scaling.")
    else:
        alpha = max(abs(T_i) for T_i in T_target.values()) / max_R0
        if alpha > 1:
            theta_init = {e: alpha * th for e, th in theta_init.items()}

    # Proceed with optimization using the adjusted theta_init
    # ---- baseline (all θ = 0.05) ---------------------------------------------
    dm0 = DensityMatrix(Statevector.from_instruction(build_circuit(n, theta_init)))
    R_baseline = curvature(dm0, edge_list, entropy_fn)

    # ---- signed curvature optimiser -----------------------------------------
    def loss_function(theta_dict):
        dm = DensityMatrix(Statevector.from_instruction(build_circuit(n, theta_dict)))
        R = curvature(dm, edge_list, entropy_fn)
        R = {i: R_baseline[i] - R[i] for i in R}
        return sum((R[i] - T_target[i])**2 for i in R)

    # Define gradient function
    def grad_edge(e):
        theta_plus, theta_minus = theta_init.copy(), theta_init.copy()
        theta_plus[e] += shift
        theta_minus[e] -= shift
        L_plus = loss_function(theta_plus)
        L_minus = loss_function(theta_minus)
        return 0.5 * (L_plus - L_minus)

    # Adam loop
    for t in range(1, steps + 1):
        for e in edge_list:
            g = grad_edge(e)
            m[e] = beta1 * m[e] + (1 - beta1) * g
            v[e] = beta2 * v[e] + (1 - beta2) * g**2
            theta_update = lr * (m[e] / (np.sqrt(v[e]) + 1e-8))
            theta_init[e] -= theta_update

    # final curvature
    dm  = DensityMatrix(Statevector.from_instruction(build_circuit(n, theta_init)))
    R   = curvature(dm, edge_list, entropy_fn)
    R   = {i: R_baseline[i] - R[i] for i in R}

    return theta_init, R

# Run the spacetime generation
# theta_dict, final_circuit = generate_spacetime(args.num_qubits, edge_list, target_curvature)
theta_dict, R_final = optim_signed_curvature(args.num_qubits, edge_list, target_curvature)

# Log the results
logging.info(f"Generated spacetime angles: {theta_dict}")
logging.info(f"Achieved curvature R_i: {R_final}")

# Print results
print("\nOptimised θ_ij (rad):")
for e, th in theta_dict.items():
    print(f"  {e}: {th:.4f}")

print("\nAchieved curvature  R_i:")
for i in range(args.num_qubits):
    print(f"  qubit {i}: {R_final[i]:+.4f}   (target {target_curvature[i]:+.2f})")

print("\nResidual RMS error:",
      np.sqrt(sum((R_final[i]-target_curvature[i])**2 for i in range(args.num_qubits))/args.num_qubits))

# Convert tuple keys to strings for JSON compatibility
string_theta_dict = {str(k): v for k, v in theta_dict.items()}

# Prepare results with input parameters and achieved curvature
results = {
    "input_parameters": {
        "num_qubits": args.num_qubits,
        "edge_list": args.edge_list,
        "target_curvature": args.target_curvature,
        "device": args.device
    },
    "theta_dict": string_theta_dict,
    "achieved_curvature": R_final
}

# Save results to experiment_logs
log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'experiment_logs', 'custom_spacetime_qiskit')
os.makedirs(log_dir, exist_ok=True)
with open(os.path.join(log_dir, 'results.json'), 'w') as f:
    json.dump(results, f, indent=4)

# Save plots if any
# Example: plt.savefig(os.path.join(log_dir, 'plot.png'))

logging.info(f"Results saved to {log_dir}") 