"""
custom_spacetime_demo.py
------------------------
• Records a positive-curvature *baseline* from initial angles θ=0.1 rad.
• Defines signed curvature  R_i = (baseline – current).
• Fits a wavy target profile:  [-0.2, -0.1, 0.0, +0.1, +0.2].
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, entropy

# ---------------- helper functions -------------------------------------------
def build_circuit(n, theta_dict):
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    for (i, j), th in theta_dict.items():
        qc.cp(th, i, j)
    return qc

def curvature(dm, edge_list, entropy_fn):
    """Return dictionary  R_i  given a global density matrix."""
    n = dm.num_qubits
    R = {i: 0.0 for i in range(n)}
    for (i, j) in edge_list:
        dm_ij = partial_trace(dm, [q for q in range(n) if q not in (i, j)])
        Si  = entropy_fn(partial_trace(dm, [q for q in range(n) if q != i]))
        Sj  = entropy_fn(partial_trace(dm, [q for q in range(n) if q != j]))
        Sij = entropy_fn(dm_ij)
        Iij = Si + Sj - Sij
        R[i] += Iij
        R[j] += Iij
    return R

def optim_signed_curvature(n, edge_list, T_target,
                           steps=200, lr=0.05, eps=1e-3, seed=None):
    rng = np.random.default_rng(seed)
    theta = {e: 0.1 * rng.uniform(0.8, 1.2) for e in edge_list}
    entropy_fn = lambda dm: entropy(dm, base=2)

    # ---- baseline (all θ = 0.1) ---------------------------------------------
    dm0 = DensityMatrix(Statevector.from_instruction(build_circuit(n, theta)))
    R_baseline = curvature(dm0, edge_list, entropy_fn)

    # ---- signed curvature optimiser -----------------------------------------
    for _ in range(steps):
        # current curvature
        dm  = DensityMatrix(Statevector.from_instruction(build_circuit(n, theta)))
        R   = curvature(dm, edge_list, entropy_fn)
        R   = {i: R_baseline[i] - R[i] for i in R}          # signed version
        L   = sum((R[i] - T_target[i])**2 for i in R)

        # finite-difference gradient
        grad = {}
        for e in edge_list:
            orig = theta[e]
            thetas_shift = {}
            for sign in (+1, -1):
                theta[e] = orig + sign*eps
                dm_d  = DensityMatrix(
                        Statevector.from_instruction(build_circuit(n, theta)))
                R_d   = curvature(dm_d, edge_list, entropy_fn)
                R_d   = {i: R_baseline[i] - R_d[i] for i in R_d}
                L_d   = sum((R_d[i] - T_target[i])**2 for i in R_d)
                thetas_shift[sign] = L_d
            grad[e] = (thetas_shift[+1] - thetas_shift[-1]) / (2*eps)
            theta[e] = orig

        # gradient step
        for e in edge_list:
            theta[e] -= lr * grad[e]

    # final curvature
    dm  = DensityMatrix(Statevector.from_instruction(build_circuit(n, theta)))
    R   = curvature(dm, edge_list, entropy_fn)
    R   = {i: R_baseline[i] - R[i] for i in R}

    return theta, R

# ---------------- run demo ----------------------------------------------------
if __name__ == "__main__":
    n         = 5
    edge_list = [(0,1),(1,2),(2,3),(3,4)]
    T_target  = {0:-0.2, 1:-0.1, 2:0.0, 3:+0.1, 4:+0.2}

    theta_opt, R_final = optim_signed_curvature(n, edge_list, T_target,
                                                steps=250, lr=0.04, seed=42)

    print("\nOptimised θ_ij (rad):")
    for e, th in theta_opt.items():
        print(f"  {e}: {th:.4f}")

    print("\nAchieved curvature  R_i:")
    for i in range(n):
        print(f"  qubit {i}: {R_final[i]:+.4f}   (target {T_target[i]:+.2f})")

    print("\nResidual RMS error:",
          np.sqrt(sum((R_final[i]-T_target[i])**2 for i in range(n))/n)) 