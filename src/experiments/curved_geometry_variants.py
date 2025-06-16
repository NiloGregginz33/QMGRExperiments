import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
import seaborn as sns
from itertools import combinations
from braket.circuits import Circuit, FreeParameter
from braket.devices import LocalSimulator
from sklearn.manifold import MDS
import json

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.AWSFactory import AdSGeometryAnalyzer6Q

def run_geometry_variants():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiment_logs/curved_geometry_variants_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)

    results = {
        "variant": [],
        "phi": [],
        "entropy": [],
        "distance": [],
        "curvature": [],
        "mi_matrix": []
    }

    # Define geometry variants as functions
    def variant_original(circ, phi):
        circ.h(0)
        circ.cnot(0, 2)
        circ.cnot(0, 3)
        circ.rx(0, phi)
        circ.cz(0, 1)
        circ.cnot(1, 2)
        circ.rx(2, phi)
        circ.cz(1, 3)
        circ.cnot(3, 4)
        circ.rx(4, phi)
        circ.cnot(4, 5)
        return circ

    def variant_nonlocal(circ, phi):
        circ.h(0)
        circ.cnot(0, 2)
        circ.cnot(0, 3)
        circ.rx(0, phi)
        circ.rx(1, phi)
        circ.rx(2, phi)
        circ.cz(0, 3)
        circ.cz(1, 4)
        circ.cz(2, 5)
        circ.cnot(0, 5)
        circ.cnot(5, 3)
        circ.cz(3, 4)
        circ.cz(4, 1)
        circ.cnot(4, 2)
        return circ

    def variant_ring(circ, phi):
        circ.h(0)
        for i in range(6):
            circ.cnot(i, (i+1)%6)
        for i in range(6):
            circ.rx(i, phi)
        return circ

    def variant_star(circ, phi):
        circ.h(0)
        for i in range(1, 6):
            circ.cnot(0, i)
        for i in range(6):
            circ.rx(i, phi)
        return circ

    def variant_alternating(circ, phi):
        circ.h(0)
        for i in range(0, 6, 2):
            circ.cnot(i, (i+1)%6)
        for i in range(6):
            circ.rx(i, phi)
        return circ

    variants = {
        "original": variant_original,
        "nonlocal": variant_nonlocal,
        "ring": variant_ring,
        "star": variant_star,
        "alternating": variant_alternating
    }

    timesteps = np.linspace(0, 2*np.pi, 12)
    device = LocalSimulator()

    for vname, vfunc in variants.items():
        print(f"\nRunning variant: {vname}")
        for phi_val in timesteps:
            phi = FreeParameter("phi")
            circ = Circuit()
            circ = vfunc(circ, phi)
            circ.probability()
            task = device.run(circ, inputs={"phi": phi_val}, shots=1024)
            result = task.result()
            probs = np.array(result.values).reshape(-1)

            # Compute MI matrix
            n_qubits = 6
            mi_matrix = np.zeros((n_qubits, n_qubits))
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    # Use AdSGeometryAnalyzer6Q's MI computation
                    AB = AdSGeometryAnalyzer6Q.marginal_probs(None, probs, n_qubits, [i, j])
                    A = AdSGeometryAnalyzer6Q.marginal_probs(None, probs, n_qubits, [i])
                    B = AdSGeometryAnalyzer6Q.marginal_probs(None, probs, n_qubits, [j])
                    mi = AdSGeometryAnalyzer6Q.shannon_entropy(None, A) + AdSGeometryAnalyzer6Q.shannon_entropy(None, B) - AdSGeometryAnalyzer6Q.shannon_entropy(None, AB)
                    mi_matrix[i, j] = mi_matrix[j, i] = mi

            # Emergent geometry
            dist = np.exp(-mi_matrix)
            np.fill_diagonal(dist, 0)
            coords = MDS(n_components=3, dissimilarity='precomputed').fit_transform(dist)

            # Estimate curvature (average over all triplets)
            curvatures = []
            for triplet in combinations(range(n_qubits), 3):
                curv = AdSGeometryAnalyzer6Q.estimate_local_curvature(None, coords, triplet)
                curvatures.append(curv)
            avg_curvature = float(np.mean(curvatures))

            # Entropy of a subsystem (e.g., qubits 3,4)
            rad_probs = AdSGeometryAnalyzer6Q.marginal_probs(None, probs, n_qubits, [3, 4])
            S_rad = AdSGeometryAnalyzer6Q.shannon_entropy(None, rad_probs)

            # Distance between Q3 and Q4
            d_Q34 = float(np.linalg.norm(coords[3] - coords[4]))

            # Log results
            results["variant"].append(vname)
            results["phi"].append(float(phi_val))
            results["entropy"].append(float(S_rad))
            results["distance"].append(d_Q34)
            results["curvature"].append(avg_curvature)
            results["mi_matrix"].append(mi_matrix.tolist())

            print(f"Variant: {vname}, φ={phi_val:.2f}, S_rad={S_rad:.4f}, d(Q3,Q4)={d_Q34:.4f}, K={avg_curvature:.4f}")

    # Save results
    with open(f"{exp_dir}/geometry_variants_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot summary for each variant
    for vname in variants.keys():
        mask = [i for i, v in enumerate(results["variant"]) if v == vname]
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(np.array(results["phi"])[mask], np.array(results["entropy"])[mask], label="Entropy")
        plt.xlabel("φ")
        plt.ylabel("S_rad")
        plt.title(f"Entropy - {vname}")
        plt.grid(True)
        plt.subplot(1, 3, 2)
        plt.plot(np.array(results["phi"])[mask], np.array(results["distance"])[mask], label="d(Q3,Q4)")
        plt.xlabel("φ")
        plt.ylabel("Distance")
        plt.title(f"d(Q3,Q4) - {vname}")
        plt.grid(True)
        plt.subplot(1, 3, 3)
        plt.plot(np.array(results["phi"])[mask], np.array(results["curvature"])[mask], label="Curvature")
        plt.xlabel("φ")
        plt.ylabel("Curvature")
        plt.title(f"Curvature - {vname}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{exp_dir}/summary_{vname}.png")
        plt.close()

    # Theoretical analysis
    analysis = {}
    for vname in variants.keys():
        mask = [i for i, v in enumerate(results["variant"]) if v == vname]
        entropy = np.array(results["entropy"])[mask]
        curvature = np.array(results["curvature"])[mask]
        distance = np.array(results["distance"])[mask]
        # Correlations
        entropy_curv_corr = float(np.corrcoef(entropy, curvature)[0, 1])
        entropy_dist_corr = float(np.corrcoef(entropy, distance)[0, 1])
        analysis[vname] = {
            "entropy_curvature_corr": entropy_curv_corr,
            "entropy_distance_corr": entropy_dist_corr,
            "max_curvature": float(np.max(curvature)),
            "min_curvature": float(np.min(curvature)),
            "mean_curvature": float(np.mean(curvature))
        }
    with open(f"{exp_dir}/theoretical_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print("\nAnalysis saved. Check the experiment_logs folder for results and plots.")

if __name__ == "__main__":
    run_geometry_variants() 