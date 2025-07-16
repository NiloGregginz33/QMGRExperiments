import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import itertools
import networkx as nx
from scipy.stats import linregress

def load_distance_matrix(mi_dict, num_qubits):
    MI = np.zeros((num_qubits, num_qubits))
    for k, v in mi_dict.items():
        i, j = map(int, k.split('_')[1].split(','))
        MI[i, j] = v
        MI[j, i] = v
    # Clamp MI to avoid log(0)
    MI = np.clip(MI, 1e-10, 1.0)
    D = -np.log(MI)
    return D

def calculate_angle_sum_and_angles(D, i, j, k, curvature=1.0):
    a, b, c = D[j, k], D[i, k], D[i, j]
    # Always compute all three
    def hyp_angle(opposite, x, y, kappa):
        num = np.cosh(x) * np.cosh(y) - np.cosh(opposite)
        denom = np.sinh(x) * np.sinh(y)
        cosA = num / denom if denom != 0 else 1.0
        return np.arccos(np.clip(cosA, -1.0, 1.0))
    def sph_angle(opposite, x, y, kappa):
        K = np.sqrt(curvature)
        num = np.cos(K * opposite) - np.cos(K * x) * np.cos(K * y)
        denom = np.sin(K * x) * np.sin(K * y)
        cosA = num / denom if denom != 0 else 1.0
        return np.arccos(np.clip(cosA, -1.0, 1.0))
    def euc_angle(opposite, x, y):
        cosA = (x**2 + y**2 - opposite**2) / (2 * x * y)
        return np.arccos(np.clip(cosA, -1.0, 1.0))
    # Try all three, return all
    angles_hyp = [hyp_angle(a, b, c, curvature), hyp_angle(b, a, c, curvature), hyp_angle(c, a, b, curvature)]
    angles_sph = [sph_angle(a, b, c, curvature), sph_angle(b, a, c, curvature), sph_angle(c, a, b, curvature)]
    angles_euc = [euc_angle(a, b, c), euc_angle(b, a, c), euc_angle(c, a, b)]
    return angles_hyp, angles_sph, angles_euc

def pick_result_file(default_dir):
    files = [f for f in os.listdir(default_dir) if f.endswith('.json')]
    # Filter files to only those containing 'mutual_information'
    valid_files = []
    for f in files:
        try:
            with open(os.path.join(default_dir, f)) as jf:
                data = json.load(jf)
            if 'mutual_information' in data:
                valid_files.append(f)
        except Exception:
            continue
    if not valid_files:
        print(f"No valid .json files with 'mutual_information' found in {default_dir}")
        exit(1)
    print("Select a result file to analyze:")
    for idx, fname in enumerate(valid_files):
        print(f"[{idx}] {fname}")
    while True:
        try:
            choice = int(input("Enter file number: "))
            if 0 <= choice < len(valid_files):
                return os.path.join(default_dir, valid_files[choice])
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a valid number.")

def spectral_dimension_analysis(D, S_max=10, num_walks=1000, seed=42):
    np.random.seed(seed)
    n = D.shape[0]
    # Build adjacency graph: connect nodes with finite, nonzero distance
    G = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            if 0 < D[i, j] < np.inf:
                G.add_edge(i, j)
    if not nx.is_connected(G):
        print("[WARNING] Adjacency graph is not connected. Spectral dimension may be ill-defined.")
    P_s = []
    for s in range(1, S_max+1):
        returns = 0
        for start in range(n):
            for _ in range(num_walks // n):
                node = start
                for _ in range(s):
                    nbrs = list(G.neighbors(node))
                    if not nbrs:
                        break
                    node = np.random.choice(nbrs)
                if node == start:
                    returns += 1
        P_s.append(returns / num_walks)
    print(f"Return probabilities P(s): {P_s}")
    s_vals = np.arange(1, S_max+1)
    # Filter out s where P(s) == 0
    mask = np.array(P_s) > 0
    if not np.any(mask):
        print("[WARNING] All return probabilities are zero. Cannot fit spectral dimension.")
        return
    log_s = np.log(s_vals[mask])
    log_P = np.log(np.array(P_s)[mask])
    if len(log_s) < 2:
        print("[WARNING] Not enough nonzero P(s) values to fit spectral dimension.")
        return
    slope, intercept, r, p, stderr = linregress(log_s, log_P)
    d_spectral = -2 * slope
    plt.figure()
    plt.plot(log_s, log_P, 'o-', label='Data')
    plt.plot(log_s, slope*log_s + intercept, '--', label=f'Fit: slope={slope:.3f}')
    plt.xlabel('log s')
    plt.ylabel('log P(s)')
    plt.title(f'Spectral Dimension Estimate: d_spectral={d_spectral:.2f}')
    plt.legend()
    plt.savefig(os.path.join("plots", "spectral_dimension_fit.png"))
    plt.show()
    print(f"Spectral dimension d_spectral ≈ {d_spectral:.2f} (fit r={r:.3f})")

def laplacian_spectral_dimension(D, S_max=10, s_min=0.1, s_max=10, num_s=10):
    n = D.shape[0]
    # Build adjacency graph: connect nodes with finite, nonzero distance
    G = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            if 0 < D[i, j] < np.inf:
                G.add_edge(i, j)
    if not nx.is_connected(G):
        print("[WARNING] Adjacency graph is not connected. Laplacian spectrum may be ill-defined.")
        return
    L = nx.laplacian_matrix(G).toarray()
    evals = np.linalg.eigvalsh(L)
    s_vals = np.logspace(np.log10(s_min), np.log10(s_max), num_s)
    K_s = []
    for s in s_vals:
        K = np.sum(np.exp(-s * evals))
        K_s.append(K)
    log_s = np.log(s_vals)
    log_K = np.log(K_s)
    from scipy.stats import linregress
    slope, intercept, r, p, stderr = linregress(log_s, log_K)
    d_spectral = -2 * slope
    plt.figure()
    plt.plot(log_s, log_K, 'o-', label='Data')
    plt.plot(log_s, slope*log_s + intercept, '--', label=f'Fit: slope={slope:.3f}')
    plt.xlabel('log s')
    plt.ylabel('log K(s)')
    plt.title(f'Laplacian Spectral Dimension: d_spectral={d_spectral:.2f}')
    plt.legend()
    plt.savefig(os.path.join("plots", "laplacian_spectral_dimension_fit.png"))
    plt.show()
    print(f"[Laplacian] Spectral dimension d_spectral ≈ {d_spectral:.2f} (fit r={r:.3f})")

def main():
    parser = argparse.ArgumentParser(description="Analyze emergent metric and curvature from MI data.")
    parser.add_argument("result_json", type=str, nargs='?', default=None, help="Path to result JSON file (if not specified, pick from list)")
    parser.add_argument("--geometry", type=str, default=None, choices=["euclidean", "hyperbolic", "spherical"], help="Override geometry type for angle calculation")
    parser.add_argument("--curvature", type=float, default=None, help="Curvature parameter (for hyperbolic/spherical geometry); if not set, use experiment value")
    parser.add_argument("--kappa_fit", type=float, default=None, help="Target kappa for area fit; if not set, use experiment value")
    parser.add_argument("--logdir", type=str, default="experiment_logs/custom_curvature_experiment", help="Directory to search for result files")
    args = parser.parse_args()
    # Ensure plots directory exists
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    # If no result_json specified, pick from list
    if args.result_json is None:
        result_json = pick_result_file(args.logdir)
    else:
        result_json = args.result_json
    with open(result_json) as jf:
        data = json.load(jf)
    if 'mutual_information' not in data:
        print(f"Error: 'mutual_information' key not found in {result_json}. Skipping analysis.")
        return
    num_qubits = data["spec"]["num_qubits"]
    mi_dict = data["mutual_information"][-1] if isinstance(data["mutual_information"], list) else data["mutual_information"]
    D = load_distance_matrix(mi_dict, num_qubits)
    # Print experiment parameters and summary metrics
    print("\n=== Experiment Parameters ===")
    for k, v in data["spec"].items():
        print(f"{k}: {v}")
    print(f"uid: {data.get('uid', 'N/A')}")
    print("\n=== Key Metrics ===")
    for key in ["gromov_delta", "mean_distance", "mean_angle_sum", "min_angle_sum", "max_angle_sum", "edge_weight_variance"]:
        if key in data:
            print(f"{key}: {data[key]}")
    # Plot MI and distance matrix heatmaps for each timestep
    if "mutual_information" in data and isinstance(data["mutual_information"], list):
        for t, mi_t in enumerate(data["mutual_information"]):
            if mi_t is None:
                continue
            D_t = load_distance_matrix(mi_t, num_qubits)
            plt.figure()
            plt.imshow(D_t, cmap='viridis')
            plt.colorbar()
            plt.title(f"Distance Matrix (timestep {t})")
            plt.savefig(os.path.join(plots_dir, f"distance_matrix_t{t}.png"))
            plt.show()
            plt.figure()
            MI_t = np.exp(-D_t)
            plt.imshow(MI_t, cmap='hot')
            plt.colorbar()
            plt.title(f"Mutual Information (timestep {t})")
            plt.savefig(os.path.join(plots_dir, f"mi_matrix_t{t}.png"))
            plt.show()
    # Plot Regge action, edge length, angle deficit, and Gromov delta evolution if present
    if "regge_action_evolution" in data:
        plt.figure()
        plt.plot(data["regge_action_evolution"])
        plt.xlabel("Regge step")
        plt.ylabel("Regge action")
        plt.title("Regge Action Evolution")
        plt.savefig(os.path.join(plots_dir, "regge_action_evolution.png"))
        plt.show()
    if "edge_length_evolution" in data:
        plt.figure()
        arr = np.array(data["edge_length_evolution"])
        plt.plot(arr)
        plt.xlabel("Regge step")
        plt.ylabel("Edge lengths")
        plt.title("Edge Length Evolution")
        plt.savefig(os.path.join(plots_dir, "edge_length_evolution.png"))
        plt.show()
    if "angle_deficit_evolution" in data:
        plt.figure()
        arr = np.array(data["angle_deficit_evolution"])
        plt.plot(arr)
        plt.xlabel("Regge step")
        plt.ylabel("Angle deficits")
        plt.title("Angle Deficit Evolution")
        plt.savefig(os.path.join(plots_dir, "angle_deficit_evolution.png"))
        plt.show()
    if "gromov_delta_evolution" in data:
        plt.figure()
        plt.plot(data["gromov_delta_evolution"])
        plt.xlabel("Regge step")
        plt.ylabel("Gromov delta")
        plt.title("Gromov Delta Evolution")
        plt.savefig(os.path.join(plots_dir, "gromov_delta_evolution.png"))
        plt.show()
    # Print triangle inequality violations if present
    if "triangle_inequality_violations" in data and data["triangle_inequality_violations"]:
        print("\nTriangle inequality violations:")
        for v in data["triangle_inequality_violations"]:
            print(v)
    # Print matter action if present
    if "S_matter" in data:
        print(f"Matter action: {data['S_matter']}")
    # Plot 2D/3D embeddings if available
    if "embedding_coords" in data:
        coords2 = np.array(data["embedding_coords"])
        plt.figure()
        plt.scatter(coords2[:,0], coords2[:,1])
        plt.title("2D Embedding of Geometry")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(os.path.join(plots_dir, "embedding_2d.png"))
        plt.show()
    if "embedding_coords_3d" in data:
        coords3 = np.array(data["embedding_coords_3d"])
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coords3[:,0], coords3[:,1], coords3[:,2])
        plt.title("3D Embedding of Geometry")
        plt.savefig(os.path.join(plots_dir, "embedding_3d.png"))
        plt.show()
    # Optionally plot Lorentzian embedding if present
    if "lorentzian_embedding" in data:
        coordsL = np.array(data["lorentzian_embedding"])
        if coordsL.shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(coordsL[:,0], coordsL[:,1], coordsL[:,2])
            plt.title("Lorentzian MDS Embedding")
            plt.savefig(os.path.join(plots_dir, "lorentzian_embedding.png"))
            plt.show()
        else:
            plt.figure()
            plt.scatter(coordsL[:,0], coordsL[:,1])
            plt.title("Lorentzian MDS Embedding (2D)")
            plt.savefig(os.path.join(plots_dir, "lorentzian_embedding_2d.png"))
            plt.show()
    # Use experiment curvature/geometry as default
    exp_curvature = data["spec"].get("curvature", 1.0)
    exp_geometry = data["spec"].get("geometry", None)
    kappa = args.curvature if args.curvature is not None else exp_curvature
    kappa_fit = args.kappa_fit if args.kappa_fit is not None else kappa
    geometry = args.geometry if args.geometry is not None else exp_geometry
    # If geometry is still None, infer from first triangle
    inferred = False
    deficits = []
    areas = []
    area_delta_pairs = []
    angle_sums = []
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            for k in range(j+1, num_qubits):
                angles_hyp, angles_sph, angles_euc = calculate_angle_sum_and_angles(D, i, j, k, curvature=kappa)
                # Infer geometry if needed
                if geometry is None and not inferred:
                    sum_hyp = sum(angles_hyp)
                    sum_sph = sum(angles_sph)
                    sum_euc = sum(angles_euc)
                    if sum_hyp < np.pi:
                        geometry = "hyperbolic"
                    elif sum_sph > np.pi:
                        geometry = "spherical"
                    else:
                        geometry = "euclidean"
                    print(f"[Auto-detected geometry: {geometry}]")
                    inferred = True
                # Use the right angles for the chosen geometry
                if geometry == "hyperbolic":
                    angles = angles_hyp
                    deficit = np.pi - sum(angles)
                    area = deficit / kappa_fit
                elif geometry == "spherical":
                    angles = angles_sph
                    deficit = sum(angles) - np.pi
                    area = deficit / kappa_fit
                else:
                    angles = angles_euc
                    deficit = 0.0
                    area = 0.0
                deficits.append(deficit)
                areas.append(area)
                area_delta_pairs.append((area, deficit))
                angle_sums.append(sum(angles))
    deficits = np.array(deficits)
    areas = np.array(areas)
    # Plot histogram of triangle deficits
    plt.figure()
    plt.hist(deficits, bins=50, alpha=0.7)
    plt.xlabel("Triangle deficit (Δ or δ_sph)")
    plt.ylabel("Count")
    plt.title(f"Distribution of Triangle Deficits ({geometry.capitalize()} Curvature)")
    plt.savefig(os.path.join(plots_dir, "triangle_deficit_histogram.png"))
    plt.show()
    # Print summary
    print(f"Triangle deficit: mean={np.mean(deficits):.4f}, std={np.std(deficits):.4f}, min={np.min(deficits):.4f}, max={np.max(deficits):.4f}")
    # Δ vs. area sanity check
    if geometry in ("hyperbolic", "spherical"):
        plt.figure()
        plt.scatter(areas, deficits, alpha=0.7)
        plt.xlabel(f"{'Hyperbolic' if geometry=='hyperbolic' else 'Spherical'} triangle area (kappa={kappa_fit})")
        plt.ylabel(f"Triangle deficit {'Δ' if geometry=='hyperbolic' else 'δ_sph'}")
        plt.title(f"Deficit vs. Area ({geometry.capitalize()}, should be linear, slope~{kappa_fit})")
        plt.savefig(os.path.join(plots_dir, "delta_vs_area.png"))
        plt.show()
        # Linear fit
        slope, intercept = np.polyfit(areas, deficits, 1)
        print(f"Deficit vs. area data (area, deficit):")
        for pair in area_delta_pairs:
            print(pair)
        print(f"Linear fit: deficit = {slope:.4f} * area + {intercept:.4f}")
        print(f"Fitted κ ≃ {slope:.4f} (should be ~{kappa_fit} for constant curvature)")
    else:
        print("Euclidean geometry detected: deficits and areas are zero.")

    # After main analysis, run spectral dimension analyses
    print("\n=== Spectral Dimension Analysis (Random Walk) ===")
    spectral_dimension_analysis(D, S_max=min(10, num_qubits*2), num_walks=1000)
    print("\n=== Spectral Dimension Analysis (Laplacian Spectrum) ===")
    laplacian_spectral_dimension(D, S_max=10, s_min=0.1, s_max=10, num_s=10)

if __name__ == "__main__":
    main() 