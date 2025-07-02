import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

EXPERIMENT_LOGS_DIR = 'experiment_logs'

# --- Utility Functions ---
def list_curved_geometry_logs():
    return [d for d in os.listdir(EXPERIMENT_LOGS_DIR)
            if d.startswith('curved_geometry_qiskit_') and os.path.isdir(os.path.join(EXPERIMENT_LOGS_DIR, d))]

def load_results(log_dir):
    path = os.path.join(EXPERIMENT_LOGS_DIR, log_dir, 'results.json')
    with open(path, 'r') as f:
        return json.load(f)

def select_from_list(options, prompt):
    for i, opt in enumerate(options):
        print(f"  {i+1}. {opt}")
    while True:
        try:
            idx = int(input(prompt)) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except Exception:
            pass
        print("Invalid selection. Try again.")

def get_phi_options(results, geometry):
    return [entry['phi'] for entry in results[geometry]]

def get_mi_matrix(results, geometry, phi):
    for entry in results[geometry]:
        if np.isclose(entry['phi'], phi):
            return np.array(entry['mi_matrix'])
    raise ValueError("Phi value not found.")

def get_qubit_count(mi_matrix):
    return mi_matrix.shape[0]

def build_distance_graph(mi_matrix):
    dist = np.exp(-mi_matrix)
    np.fill_diagonal(dist, 0)
    G = nx.Graph()
    n = dist.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            G.add_edge(i, j, weight=dist[i, j])
    return G, dist

def compute_geodesic(G, A, B):
    min_path = None
    min_length = float('inf')
    for a in A:
        for b in B:
            try:
                length, path = nx.single_source_dijkstra(G, a, b)
                if length < min_length:
                    min_length = length
                    min_path = path
            except nx.NetworkXNoPath:
                continue
    return min_path, min_length

def compute_mds(dist):
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(dist)
    return coords

def plot_geodesic(coords, path, A, B, geometry, phi):
    plt.figure(figsize=(6,6))
    plt.scatter(coords[:,0], coords[:,1], c='b', label='Qubits')
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i), fontsize=12, ha='center', va='center', color='white', bbox=dict(facecolor='blue', alpha=0.5, boxstyle='circle'))
    if path is not None:
        path_coords = coords[path]
        plt.plot(path_coords[:,0], path_coords[:,1], 'r-', linewidth=2, label='Entanglement Geodesic')
        plt.scatter(coords[A,0], coords[A,1], c='g', s=100, label='Subsystem A')
        plt.scatter(coords[B,0], coords[B,1], c='orange', s=100, label='Subsystem B')
    plt.title(f'Entanglement Geodesic ({geometry}, phi={phi:.2f})')
    plt.legend()
    plt.axis('equal')
    plt.show()

def parse_range_input(input_str, max_val):
    input_str = input_str.strip().lower()
    if input_str == 'all':
        return list(range(max_val))
    result = set()
    for part in input_str.split(','):
        if '-' in part:
            start, end = part.split('-')
            result.update(range(int(start), int(end)+1))
        else:
            result.add(int(part))
    return sorted(result)

def parse_subsystem_pairs(input_str, n_qubits):
    input_str = input_str.strip().lower()
    if input_str == 'all':
        # All pairs of single qubits
        pairs = []
        for i in range(n_qubits):
            for j in range(n_qubits):
                if i != j:
                    pairs.append(([i], [j]))
        return pairs
    pairs = []
    for pair_str in input_str.split(';'):
        if not pair_str.strip():
            continue
        try:
            A_str, B_str = pair_str.split('->')
            A = [int(x) for x in A_str.split(',') if x.strip()]
            B = [int(x) for x in B_str.split(',') if x.strip()]
            if all(0 <= x < n_qubits for x in A+B):
                pairs.append((A, B))
        except Exception:
            pass
    return pairs

def main():
    print("\n=== Entanglement Geodesic Analysis for Curved Geometry Experiment (Batch Mode) ===\n")
    logs = list_curved_geometry_logs()
    if not logs:
        print("No curved_geometry_qiskit experiment logs found.")
        return
    log_dir = select_from_list(logs, "Select experiment log directory: ")
    results = load_results(log_dir)
    geometry = select_from_list(list(results.keys()), "Select geometry (flat/curved): ")
    phi_options = get_phi_options(results, geometry)
    print("Available phi values:")
    for i, phi in enumerate(phi_options):
        print(f"  {i+1}. {phi:.4f}")
    phi_input = input("Select phi values (e.g. all, 1-3, 1,3,5): ").strip().lower()
    if phi_input == 'all':
        phi_indices = list(range(len(phi_options)))
    else:
        phi_indices = []
        for part in phi_input.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                phi_indices.extend(range(start-1, end))
            else:
                phi_indices.append(int(part)-1)
        phi_indices = [i for i in phi_indices if 0 <= i < len(phi_options)]
    selected_phis = [phi_options[i] for i in phi_indices]
    mi_matrix = get_mi_matrix(results, geometry, selected_phis[0])
    n_qubits = get_qubit_count(mi_matrix)
    print(f"\nSystem has {n_qubits} qubits (0 to {n_qubits-1}).")
    print("Define subsystem pairs to analyze:")
    print("  - Format: A->B;C->D (e.g. 0->5;1,2->3,4)")
    print("  - Use 'all' for all single-qubit pairs.")
    pairs_input = input("Subsystem pairs: ").strip().lower()
    pairs = parse_subsystem_pairs(pairs_input, n_qubits)
    plot_each = input("Plot each geodesic? (y/n): ").strip().lower().startswith('y')
    summary = []
    for phi in selected_phis:
        mi_matrix = get_mi_matrix(results, geometry, phi)
        G, dist = build_distance_graph(mi_matrix)
        coords = compute_mds(dist)
        for (A, B) in pairs:
            path, length = compute_geodesic(G, A, B)
            summary.append({'phi': phi, 'A': A, 'B': B, 'path': path, 'length': length})
            print(f"\nphi={phi:.4f}, A={A}, B={B}")
            print(f"  Minimal entanglement geodesic: {path}")
            print(f"  Geodesic length: {length:.4f}")
            if plot_each:
                plot_geodesic(coords, path, np.array(A), np.array(B), geometry, phi)
    print("\n--- Summary Table ---")
    print(f"{'phi':>8} | {'A':>8} | {'B':>8} | {'length':>10}")
    print('-'*44)
    for row in summary:
        print(f"{row['phi']:8.4f} | {str(row['A']):>8} | {str(row['B']):>8} | {row['length']:10.4f}")
    print("\n--- Interpretation ---")
    print("If the geodesic path is 'straight' in the MDS embedding, this supports the idea that entanglement encodes geometric distance (holography).")
    print("In curved cases, geodesics may bend or take longer paths, reflecting emergent curvature from entanglement structure. Compare flat vs curved for the same subsystems and phi.")

if __name__ == "__main__":
    main() 