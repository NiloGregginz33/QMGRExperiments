import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix
from qiskit_ibm_runtime.fake_provider import FakeJakartaV2
from qiskit_ibm_runtime import QiskitRuntimeService
import os
import json
import argparse
from datetime import datetime
from scipy.linalg import expm

# --- Geometry Circuit (4 qubits) ---
def build_geometry_circuit(phi):
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 2)
    qc.cx(0, 3)
    qc.rx(phi, 0)
    qc.cx(0, 1)
    qc.rz(np.pi, 1)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.rx(phi, 2)
    qc.cx(1, 3)
    qc.rz(np.pi, 3)
    qc.cx(1, 3)
    return qc

def shannon_entropy(probs):
    probs = np.array(probs)
    return -np.sum(probs * np.log2(probs + 1e-12))

def compute_mi(rho, qA, qB, total_qubits):
    # Compute mutual information I(A:B) for qubits qA, qB from density matrix
    subsys = [qA, qB]
    rho_AB = partial_trace(rho, [i for i in range(total_qubits) if i not in subsys])
    rho_A = partial_trace(rho, [i for i in range(total_qubits) if i != qA])
    rho_B = partial_trace(rho, [i for i in range(total_qubits) if i != qB])
    S_A = -np.trace(rho_A.data @ np.log2(rho_A.data + 1e-12))
    S_B = -np.trace(rho_B.data @ np.log2(rho_B.data + 1e-12))
    S_AB = -np.trace(rho_AB.data @ np.log2(rho_AB.data + 1e-12))
    return S_A + S_B - S_AB

def compute_mi_matrix(rho, total_qubits):
    mi_matrix = np.zeros((total_qubits, total_qubits))
    for a in range(total_qubits):
        for b in range(a+1, total_qubits):
            mi = compute_mi(rho, a, b, total_qubits)
            mi_matrix[a, b] = mi_matrix[b, a] = mi
    return mi_matrix

def mds_embedding(mi_matrix):
    dist = np.exp(-mi_matrix)
    np.fill_diagonal(dist, 0)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(dist)
    return coords

def modular_flow(rho_A, alpha):
    # Modular Hamiltonian: K_A = -log(rho_A)
    evals, evecs = np.linalg.eigh(rho_A.data)
    # Avoid log(0) by thresholding
    evals = np.clip(evals, 1e-12, None)
    K_A = -np.dot(evecs, np.dot(np.diag(np.log(evals)), evecs.conj().T))
    U = expm(-1j * alpha * K_A)
    return U

def apply_modular_flow(rho, A, alpha):
    # Apply modular flow to subsystem A (list of qubit indices)
    # rho: DensityMatrix of full system
    # Returns: new DensityMatrix after modular flow
    n = int(np.log2(rho.data.shape[0]))
    dA = 2 ** len(A)
    dB = 2 ** (n - len(A))
    # Reshape to (A, B, A, B)
    rho_reshaped = rho.data.reshape([dA, dB, dA, dB])
    # Partial trace to get rho_A
    rho_A = np.trace(rho_reshaped, axis1=1, axis2=3)
    U = modular_flow(DensityMatrix(rho_A), alpha)
    # Apply U to subsystem A
    U_full = np.kron(U, np.eye(dB))
    rho_new = U_full @ rho.data @ U_full.conj().T
    return DensityMatrix(rho_new)

def main():
    parser = argparse.ArgumentParser(description="Modular Flow Geometry Experiment (Qiskit)")
    parser.add_argument('--device', type=str, default='simulator', help='Device: "simulator" or IBMQ backend name')
    parser.add_argument('--shots', type=int, default=1024, help='Number of shots (for hardware, not used in statevector)')
    parser.add_argument('--nsteps', type=int, default=6, help='Number of phi steps')
    parser.add_argument('--alphasteps', type=int, default=6, help='Number of modular flow steps')
    parser.add_argument('--subsystem', type=str, default='0,1', help='Subsystem for modular flow (comma-separated qubit indices)')
    args = parser.parse_args()

    if args.device.lower() == 'simulator':
        backend = FakeJakartaV2()
        use_statevector = True
    else:
        service = QiskitRuntimeService()
        backend = service.backend(args.device)
        use_statevector = False

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('experiment_logs', f'modular_flow_geometry_qiskit_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)

    phi_values = np.linspace(0, 2 * np.pi, args.nsteps)
    alpha_values = np.linspace(0, 2 * np.pi, args.alphasteps)
    A = [int(x) for x in args.subsystem.split(',')]
    results = []

    for phi in phi_values:
        qc = build_geometry_circuit(phi)
        if use_statevector:
            sv = Statevector.from_instruction(qc)
            rho = DensityMatrix(sv)
        else:
            tqc = transpile(qc, backend)
            job = backend.run(tqc, shots=args.shots)
            result = job.result()
            counts = result.get_counts()
            # Approximate statevector from counts (not exact)
            probs = np.array([counts.get(format(i, '04b'), 0) for i in range(16)]) / args.shots
            sv = np.sqrt(probs)
            rho = DensityMatrix(np.outer(sv, sv.conj()))
        for alpha in alpha_values:
            # Apply modular flow to subsystem A
            rho_mod = apply_modular_flow(rho, A, alpha)
            mi_matrix = compute_mi_matrix(rho_mod, 4)
            mds_coords = mds_embedding(mi_matrix)
            metrics = {
                "phi": float(phi),
                "alpha": float(alpha),
                "subsystem": A,
                "mi_matrix": mi_matrix.tolist(),
                "mds_coords": mds_coords.tolist()
            }
            results.append(metrics)

    # Save results
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Write summary
    with open(os.path.join(log_dir, 'summary.txt'), 'w') as f:
        f.write(f"Modular Flow Geometry Experiment\n")
        f.write(f"Backend: {backend}\n")
        f.write(f"Subsystem: {A}\n")
        f.write(f"Phi values: {list(phi_values)}\n")
        f.write(f"Alpha values: {list(alpha_values)}\n\n")
        f.write(f"Results: See results.json and geometry plots.\n")

    # Plot geometry snapshots for each phi, alpha
    for i, r in enumerate(results):
        coords = np.array(r['mds_coords'])
        plt.figure()
        plt.scatter(coords[:,0], coords[:,1], c='b')
        for j, (x, y) in enumerate(coords):
            plt.text(x, y, str(j), fontsize=12, ha='center', va='center', color='white', bbox=dict(facecolor='blue', alpha=0.5, boxstyle='circle'))
        plt.title(f'MDS Geometry (phi={r["phi"]:.2f}, alpha={r["alpha"]:.2f})')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f'mds_geometry_phi{int(100*r["phi"]):04d}_alpha{int(100*r["alpha"]):04d}.png'))
        plt.close()

    print(f"Modular Flow Geometry experiment complete. Results in {log_dir}")

if __name__ == "__main__":
    main() 