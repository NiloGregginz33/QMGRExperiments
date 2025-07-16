import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import itertools
import random

parser = argparse.ArgumentParser(description="Analyze custom curvature experiment result JSON.")
parser.add_argument("result_json", type=str, help="Path to result JSON file")
parser.add_argument("--plot", type=str, nargs='+', choices=["mi_heatmap", "mi_graph", "euclidean", "lorentzian", "regge_edges", "regge_deficits", "regge_gromov", "area_entropy"], required=True, help="Which plot(s) to generate")
parser.add_argument("--timestep", type=int, default=None, help="Timestep for MI heatmap (default: last)")
parser.add_argument("--top_frac", type=float, default=0.1, help="Fraction of strongest MI edges to show in graph overlay")
args = parser.parse_args()

with open(args.result_json, 'r') as f:
    data = json.load(f)

num_qubits = data['spec']['num_qubits']
mi_per_timestep = data.get('mutual_information', [])
embedding_coords = data.get('embedding_coords', None)
lorentzian_embedding = data.get('lorentzian_embedding', None)
edge_length_evolution = data.get('edge_length_evolution', None)
angle_deficit_evolution = data.get('angle_deficit_evolution', None)
gromov_delta_evolution = data.get('gromov_delta_evolution', None)

# 1. MI heatmap
def plot_mi_heatmap(mi_per_timestep, timestep=None):
    if not mi_per_timestep:
        print("No MI data available.")
        return
    if timestep is None:
        timestep = len(mi_per_timestep) - 1
    mi_dict = mi_per_timestep[timestep]
    mi_matrix = np.zeros((num_qubits, num_qubits))
    for k, v in mi_dict.items():
        i, j = map(int, k.split('_')[1].split(','))
        mi_matrix[i, j] = v
        mi_matrix[j, i] = v
    plt.figure(figsize=(7,6))
    sns.heatmap(mi_matrix, annot=True, cmap='viridis')
    plt.title(f"Mutual Information Heatmap (timestep {timestep})")
    plt.xlabel("Qubit")
    plt.ylabel("Qubit")
    plt.show()

# 2. MI graph overlay
def plot_mi_graph(mi_per_timestep, top_frac=0.1, timestep=None):
    if not mi_per_timestep:
        print("No MI data available.")
        return
    if timestep is None:
        timestep = len(mi_per_timestep) - 1
    mi_dict = mi_per_timestep[timestep]
    G = nx.complete_graph(num_qubits)
    for k, v in mi_dict.items():
        i, j = map(int, k.split('_')[1].split(','))
        G[i][j]['weight'] = v
    # Get top edges
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    cutoff = np.quantile(weights, 1 - top_frac)
    strong_edges = [(u, v) for u, v in G.edges() if G[u][v]['weight'] >= cutoff]
    pos = nx.circular_layout(G)
    plt.figure(figsize=(7,7))
    nx.draw(G, pos, node_color='lightblue', with_labels=True, edge_color='gray', alpha=0.3)
    nx.draw_networkx_edges(G, pos, edgelist=strong_edges, edge_color='red', width=2)
    plt.title(f"Strongest MI Edges (top {int(top_frac*100)}%) at timestep {timestep}")
    plt.show()

# 3. 2D Euclidean embedding
def plot_euclidean_embedding(coords):
    if coords is None:
        print("No Euclidean embedding available.")
        return
    coords = np.array(coords)
    plt.figure(figsize=(7,7))
    plt.scatter(coords[:,0], coords[:,1], c='b', s=80)
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i), fontsize=12, ha='center', va='center', color='white', bbox=dict(facecolor='black', alpha=0.5, boxstyle='circle'))
    plt.title("2D Euclidean Embedding")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.show()

# 4. 3D Lorentzian embedding
def plot_lorentzian_embedding(coords):
    if coords is None:
        print("No Lorentzian embedding available.")
        return
    coords = np.array(coords)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=coords[:,0], cmap='cool', s=80)
    for i, (t, x, y) in enumerate(coords):
        ax.text(t, x, y, str(i), fontsize=10)
    ax.set_xlabel('t (time)')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    plt.title("3D Lorentzian (Minkowski) Embedding: (t, x, y)")
    plt.show()

# 5. Regge edge length evolution
def plot_regge_edges(edge_length_evolution):
    if edge_length_evolution is None:
        print("No Regge edge length evolution data available.")
        return
    arr = np.array(edge_length_evolution)
    plt.figure(figsize=(10,6))
    for i in range(arr.shape[1]):
        plt.plot(arr[:,i], alpha=0.7)
    plt.title("Regge Edge Length Evolution (all edges)")
    plt.xlabel("Gradient Descent Step")
    plt.ylabel("Edge Length")
    plt.show()
# 6. Regge angle deficit evolution
def plot_regge_deficits(angle_deficit_evolution):
    if angle_deficit_evolution is None:
        print("No Regge angle deficit evolution data available.")
        return
    arr = np.array(angle_deficit_evolution)
    plt.figure(figsize=(10,6))
    for i in range(arr.shape[1]):
        plt.plot(arr[:,i], alpha=0.7)
    plt.title("Regge Angle Deficit Evolution (all triangles)")
    plt.xlabel("Gradient Descent Step")
    plt.ylabel("Angle Deficit (radians)")
    plt.show()
# 7. Regge Gromov delta evolution
def plot_regge_gromov(gromov_delta_evolution):
    if gromov_delta_evolution is None:
        print("No Regge Gromov delta evolution data available.")
        return
    arr = np.array(gromov_delta_evolution)
    plt.figure(figsize=(8,5))
    plt.plot(arr, 'o-')
    plt.title("Gromov Delta Evolution (Regge Gradient Descent)")
    plt.xlabel("Gradient Descent Step")
    plt.ylabel("Mean |Angle Deficit|")
    plt.show()

# 6. Area-entropy law
def plot_area_entropy(data, timestep=None, max_samples=256):
    mi_list = data["mutual_information"]
    n = len(mi_list[-1]) if mi_list[-1] is not None else 0
    num_qubits = data["spec"]["num_qubits"]
    if timestep is None:
        timestep = len(mi_list) - 1
    mi = mi_list[timestep]
    if mi is None:
        print("No MI data for this timestep.")
        return
    # Build MI matrix
    MI = np.zeros((num_qubits, num_qubits))
    for k, v in mi.items():
        i, j = map(int, k.split('_')[1].split(','))
        MI[i, j] = v
        MI[j, i] = v
    # All bipartitions (sample if too many)
    all_A = []
    for r in range(1, num_qubits//2+1):
        all_A += list(itertools.combinations(range(num_qubits), r))
    if len(all_A) > max_samples:
        all_A = random.sample(all_A, max_samples)
    S_list = []
    area_list = []
    for A in all_A:
        B = [i for i in range(num_qubits) if i not in A]
        # Area: sum of MI across the cut
        area = sum(MI[i, j] for i in A for j in B)
        area_list.append(area)
        # Entropy: sum of MI for all pairs in A (approximate S_A)
        S = sum(MI[i, j] for i in A for j in A if i < j)
        S_list.append(S)
    plt.figure()
    plt.scatter(area_list, S_list, alpha=0.5)
    plt.xlabel("Boundary area (sum of MI across cut)")
    plt.ylabel("Entanglement entropy S_A (approximate)")
    plt.title(f"Area-Entropy Law (timestep {timestep})")
    # Fit line
    if len(area_list) > 1:
        m, b = np.polyfit(area_list, S_list, 1)
        xfit = np.linspace(min(area_list), max(area_list), 100)
        plt.plot(xfit, m*xfit + b, 'r--', label=f"Fit: S = {m:.2f}*Area + {b:.2f}")
        plt.legend()
    plt.show()

# Dispatch
if "mi_heatmap" in args.plot:
    plot_mi_heatmap(mi_per_timestep, timestep=args.timestep)
if "mi_graph" in args.plot:
    plot_mi_graph(mi_per_timestep, top_frac=args.top_frac, timestep=args.timestep)
if "euclidean" in args.plot:
    plot_euclidean_embedding(embedding_coords)
if "lorentzian" in args.plot:
    plot_lorentzian_embedding(lorentzian_embedding)
if "regge_edges" in args.plot:
    plot_regge_edges(edge_length_evolution)
if "regge_deficits" in args.plot:
    plot_regge_deficits(angle_deficit_evolution)
if "regge_gromov" in args.plot:
    plot_regge_gromov(gromov_delta_evolution)
if "area_entropy" in args.plot:
    plot_area_entropy(data, timestep=args.timestep) 