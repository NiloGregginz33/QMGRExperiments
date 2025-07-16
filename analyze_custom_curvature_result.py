import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser(description="Analyze custom curvature experiment result JSON.")
parser.add_argument("result_json", type=str, help="Path to result JSON file")
parser.add_argument("--plot", type=str, nargs='+', choices=["mi_heatmap", "mi_graph", "euclidean", "lorentzian"], required=True, help="Which plot(s) to generate")
parser.add_argument("--timestep", type=int, default=None, help="Timestep for MI heatmap (default: last)")
parser.add_argument("--top_frac", type=float, default=0.1, help="Fraction of strongest MI edges to show in graph overlay")
args = parser.parse_args()

with open(args.result_json, 'r') as f:
    data = json.load(f)

num_qubits = data['spec']['num_qubits']
mi_per_timestep = data.get('mutual_information', [])
embedding_coords = data.get('embedding_coords', None)
lorentzian_embedding = data.get('lorentzian_embedding', None)

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

# Dispatch
if "mi_heatmap" in args.plot:
    plot_mi_heatmap(mi_per_timestep, timestep=args.timestep)
if "mi_graph" in args.plot:
    plot_mi_graph(mi_per_timestep, top_frac=args.top_frac, timestep=args.timestep)
if "euclidean" in args.plot:
    plot_euclidean_embedding(embedding_coords)
if "lorentzian" in args.plot:
    plot_lorentzian_embedding(lorentzian_embedding) 