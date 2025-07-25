#!/usr/bin/env python3
"""
Analyze curvature results from real hardware data
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Usage: python analyze_curvature_results.py <path_to_results_json>
def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_curvature_results.py <path_to_results_json>")
        sys.exit(1)
    results_path = sys.argv[1]
    if not os.path.isfile(results_path):
        print(f"File not found: {results_path}")
        sys.exit(1)
    with open(results_path, 'r') as f:
        data = json.load(f)
    # Determine output directory
    out_dir = os.path.dirname(results_path)
    # 1. Plot entropy vs. timestep
    entropy = None
    timesteps = None
    if 'entropy_per_timestep' in data and data['entropy_per_timestep']:
        entropy = data['entropy_per_timestep']
        timesteps = list(range(1, len(entropy)+1))
    elif 'boundary_entropies_per_timestep' in data and data['boundary_entropies_per_timestep']:
        entropy = [b.get('entropy_A', None) for b in data['boundary_entropies_per_timestep']]
        timesteps = list(range(1, len(entropy)+1))
    if entropy:
        plt.figure()
        plt.plot(timesteps, entropy, marker='o')
        plt.xlabel('Timestep')
        plt.ylabel('Entropy')
        plt.title('Entropy vs. Timestep')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'entropy_vs_timestep.png'))
        plt.close()
        print(f"Saved: {os.path.join(out_dir, 'entropy_vs_timestep.png')}")
    else:
        print("No entropy data found.")
    # 2. Plot MI for selected pairs over time
    mi_per_timestep = data.get('mutual_information_per_timestep', None)
    if mi_per_timestep:
        # Select a few pairs to plot (e.g., (0,1), (1,2), (0,6))
        pairs = [(0,1), (1,2), (0,6)]
        plt.figure()
        for i,j in pairs:
            mi_vals = []
            for mi_dict in mi_per_timestep:
                key1 = f"I_{i},{j}"
                key2 = f"I_{j},{i}"
                val = mi_dict.get(key1) or mi_dict.get(key2) or np.nan
                mi_vals.append(val)
            plt.plot(timesteps, mi_vals, marker='o', label=f'Qubits {i}-{j}')
        plt.xlabel('Timestep')
        plt.ylabel('Mutual Information')
        plt.title('Mutual Information vs. Timestep')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'mi_vs_timestep.png'))
        plt.close()
        print(f"Saved: {os.path.join(out_dir, 'mi_vs_timestep.png')}")
    else:
        print("No mutual_information_per_timestep data found.")
    # 3. Plot 2D and 3D embeddings
    coords2 = data.get('embedding_coords', None)
    if coords2:
        coords2 = np.array(coords2)
        plt.figure()
        plt.scatter(coords2[:,0], coords2[:,1], c='b')
        for idx, (x, y) in enumerate(coords2):
            plt.text(x, y, str(idx), fontsize=10, ha='right')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2D Hyperbolic Embedding')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'embedding_2d.png'))
        plt.close()
        print(f"Saved: {os.path.join(out_dir, 'embedding_2d.png')}")
    else:
        print("No embedding_coords (2D) found.")
    coords3 = data.get('embedding_coords_3d', None)
    if coords3:
        coords3 = np.array(coords3)
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coords3[:,0], coords3[:,1], coords3[:,2], c='r')
        for idx, (x, y, z) in enumerate(coords3):
            ax.text(x, y, z, str(idx), fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Hyperbolic Embedding')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'embedding_3d.png'))
        plt.close()
        print(f"Saved: {os.path.join(out_dir, 'embedding_3d.png')}")
    else:
        print("No embedding_coords_3d (3D) found.")

if __name__ == '__main__':
    main() 