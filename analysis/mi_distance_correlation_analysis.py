#!/usr/bin/env python3
"""
MI-Distance Correlation Analysis
===============================

This script analyzes the correlation between mutual information and geometric distance
in quantum geometry experiments, providing insights into the boundary-bulk correspondence.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.sparse.csgraph import shortest_path
from sklearn.manifold import MDS
import pandas as pd
from pathlib import Path

def load_experiment_data(filepath):
    """Load experiment data from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def compute_mutual_information_matrix(counts, num_qubits):
    """Compute mutual information matrix from quantum measurement counts"""
    
    def mutual_information(counts, num_qubits, qubits_a, qubits_b):
        """Compute mutual information between two sets of qubits"""
        total_shots = sum(counts.values())
        if total_shots == 0:
            return 0.0
        
        # Compute marginal probabilities
        p_a = {}
        p_b = {}
        p_ab = {}
        
        for bitstring, count in counts.items():
            prob = count / total_shots
            
            # Extract bits for qubits_a and qubits_b
            bits_a = ''.join([bitstring[i] for i in qubits_a])
            bits_b = ''.join([bitstring[i] for i in qubits_b])
            bits_ab = bits_a + bits_b
            
            # Update joint probability
            p_ab[bits_ab] = p_ab.get(bits_ab, 0) + prob
            
            # Update marginal probabilities
            p_a[bits_a] = p_a.get(bits_a, 0) + prob
            p_b[bits_b] = p_b.get(bits_b, 0) + prob
        
        # Compute entropies
        def entropy(probs):
            return -sum(p * np.log2(p) for p in probs.values() if p > 0)
        
        H_a = entropy(p_a)
        H_b = entropy(p_b)
        H_ab = entropy(p_ab)
        
        # Mutual information: I(A;B) = H(A) + H(B) - H(A,B)
        mi = H_a + H_b - H_ab
        return max(0, mi)  # Ensure non-negative
    
    # Compute MI for all pairs
    mi_matrix = np.zeros((num_qubits, num_qubits))
    
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            mi = mutual_information(counts, num_qubits, [i], [j])
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi
    
    return mi_matrix

def compute_geometric_distances(mi_matrix):
    """Compute geometric distances from MI matrix using MDS"""
    # Convert MI to distance: higher MI = shorter distance
    eps = 1e-8
    distance_matrix = -np.log(mi_matrix + eps)
    
    # Normalize distance matrix
    dist_max = np.max(distance_matrix)
    if dist_max > 0:
        distance_matrix = distance_matrix / dist_max
    
    # Use MDS to embed in 2D
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coordinates = mds.fit_transform(distance_matrix)
    
    return coordinates, distance_matrix

def compute_graph_distances(mi_matrix):
    """Compute shortest path distances on the MI-weighted graph"""
    # Convert MI to edge weights (higher MI = lower weight for shortest path)
    eps = 1e-8
    edge_weights = -np.log(mi_matrix + eps)
    
    # Compute shortest paths
    distances, predecessors = shortest_path(edge_weights, directed=False, return_predecessors=True)
    
    return distances

def analyze_mi_distance_correlation(results_file, output_dir):
    """Main analysis function"""
    
    # Load data
    data = load_experiment_data(results_file)
    
    # Extract counts and parameters
    counts_per_timestep = data.get('counts_per_timestep', [])
    if not counts_per_timestep:
        print("No counts_per_timestep data found!")
        return
    
    # Use the first timestep for analysis
    counts = counts_per_timestep[0] if counts_per_timestep else {}
    if not counts:
        print("No counts data found in first timestep!")
        return
    
    num_qubits = data.get('spec', {}).get('num_qubits', 0)
    if num_qubits == 0:
        print("No qubit count found!")
        return
    
    print(f"Analyzing MI-Distance correlation for {num_qubits} qubits...")
    
    # Compute MI matrix
    mi_matrix = compute_mutual_information_matrix(counts, num_qubits)
    print(f"✓ MI matrix computed - Max MI: {np.max(mi_matrix):.6f}")
    
    # Compute geometric distances
    coordinates, mi_distance_matrix = compute_geometric_distances(mi_matrix)
    print(f"✓ Geometric embedding computed")
    
    # Compute graph distances
    graph_distances = compute_graph_distances(mi_matrix)
    print(f"✓ Graph distances computed")
    
    # Prepare data for analysis
    mi_values = []
    geometric_distances = []
    graph_distances_flat = []
    
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            mi_values.append(mi_matrix[i, j])
            geometric_distances.append(mi_distance_matrix[i, j])
            graph_distances_flat.append(graph_distances[i, j])
    
    # Convert to numpy arrays
    mi_values = np.array(mi_values)
    geometric_distances = np.array(geometric_distances)
    graph_distances_flat = np.array(graph_distances_flat)
    
    # Filter out zero MI values
    non_zero_mask = mi_values > 0
    mi_non_zero = mi_values[non_zero_mask]
    geo_dist_non_zero = geometric_distances[non_zero_mask]
    graph_dist_non_zero = graph_distances_flat[non_zero_mask]
    
    # Compute correlations
    geo_corr, geo_p = pearsonr(mi_non_zero, geo_dist_non_zero)
    graph_corr, graph_p = pearsonr(mi_non_zero, graph_dist_non_zero)
    
    geo_spearman, geo_spearman_p = spearmanr(mi_non_zero, geo_dist_non_zero)
    graph_spearman, graph_spearman_p = spearmanr(mi_non_zero, graph_dist_non_zero)
    
    print(f"✓ Geometric distance correlation: {geo_corr:.6f} (p={geo_p:.2e})")
    print(f"✓ Graph distance correlation: {graph_corr:.6f} (p={graph_p:.2e})")
    
    # Create comprehensive plots
    create_mi_distance_plots(mi_non_zero, geo_dist_non_zero, graph_dist_non_zero,
                           geo_corr, graph_corr, geo_spearman, graph_spearman,
                           coordinates, mi_matrix, output_dir)
    
    # Save analysis results
    save_analysis_results(mi_non_zero, geo_dist_non_zero, graph_dist_non_zero,
                         geo_corr, graph_corr, geo_spearman, graph_spearman,
                         output_dir)
    
    return {
        'mi_matrix': mi_matrix,
        'coordinates': coordinates,
        'geo_correlation': geo_corr,
        'graph_correlation': graph_corr,
        'geo_spearman': geo_spearman,
        'graph_spearman': graph_spearman
    }

def create_mi_distance_plots(mi_values, geo_distances, graph_distances,
                           geo_corr, graph_corr, geo_spearman, graph_spearman,
                           coordinates, mi_matrix, output_dir):
    """Create comprehensive MI-Distance correlation plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('MI-Distance Correlation Analysis: Boundary-Bulk Correspondence', fontsize=16, fontweight='bold')
    
    # Plot 1: MI vs Geometric Distance
    ax1 = axes[0, 0]
    ax1.scatter(geo_distances, mi_values, alpha=0.6, s=50)
    ax1.set_xlabel('Geometric Distance (MDS)')
    ax1.set_ylabel('Mutual Information')
    ax1.set_title(f'MI vs Geometric Distance\nPearson: {geo_corr:.4f}, Spearman: {geo_spearman:.4f}')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    if len(geo_distances) > 1:
        z = np.polyfit(geo_distances, mi_values, 1)
        p = np.poly1d(z)
        ax1.plot(geo_distances, p(geo_distances), "r--", alpha=0.8)
    
    # Plot 2: MI vs Graph Distance
    ax2 = axes[0, 1]
    ax2.scatter(graph_distances, mi_values, alpha=0.6, s=50, color='green')
    ax2.set_xlabel('Graph Distance (Shortest Path)')
    ax2.set_ylabel('Mutual Information')
    ax2.set_title(f'MI vs Graph Distance\nPearson: {graph_corr:.4f}, Spearman: {graph_spearman:.4f}')
    ax2.grid(True, alpha=0.3)
    
    # Add trend line
    if len(graph_distances) > 1:
        z = np.polyfit(graph_distances, mi_values, 1)
        p = np.poly1d(z)
        ax2.plot(graph_distances, p(graph_distances), "r--", alpha=0.8)
    
    # Plot 3: Geometric vs Graph Distance
    ax3 = axes[0, 2]
    ax3.scatter(graph_distances, geo_distances, alpha=0.6, s=50, color='purple')
    ax3.set_xlabel('Graph Distance')
    ax3.set_ylabel('Geometric Distance')
    ax3.set_title('Geometric vs Graph Distance')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: MI Matrix Heatmap
    ax4 = axes[1, 0]
    im = ax4.imshow(mi_matrix, cmap='viridis', aspect='auto')
    ax4.set_title('Mutual Information Matrix')
    ax4.set_xlabel('Qubit Index')
    ax4.set_ylabel('Qubit Index')
    plt.colorbar(im, ax=ax4)
    
    # Plot 5: 2D Embedding with MI-weighted edges
    ax5 = axes[1, 1]
    ax5.scatter(coordinates[:, 0], coordinates[:, 1], s=100, c='red', alpha=0.7)
    
    # Add MI-weighted edges
    for i in range(len(coordinates)):
        for j in range(i+1, len(coordinates)):
            mi_val = mi_matrix[i, j]
            if mi_val > 0:
                # Edge thickness and alpha based on MI strength
                thickness = mi_val * 10  # Scale for visibility
                alpha = min(0.8, mi_val * 20)  # Scale for visibility
                ax5.plot([coordinates[i, 0], coordinates[j, 0]], 
                        [coordinates[i, 1], coordinates[j, 1]], 
                        'b-', alpha=alpha, linewidth=thickness)
    
    ax5.set_title('2D Embedding with MI-Weighted Edges')
    ax5.set_xlabel('X Coordinate')
    ax5.set_ylabel('Y Coordinate')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Distance correlation comparison
    ax6 = axes[1, 2]
    correlations = [geo_corr, graph_corr, geo_spearman, graph_spearman]
    labels = ['Geo Pearson', 'Graph Pearson', 'Geo Spearman', 'Graph Spearman']
    colors = ['blue', 'green', 'orange', 'red']
    
    bars = ax6.bar(labels, correlations, color=colors, alpha=0.7)
    ax6.set_ylabel('Correlation Coefficient')
    ax6.set_title('Distance Correlation Comparison')
    ax6.set_ylim(-1, 1)
    ax6.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{corr:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / 'mi_distance_correlation_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ MI-Distance correlation plot saved to: {plot_path}")
    plt.show()
    
    # Create additional detailed plot
    create_detailed_mi_analysis(mi_values, geo_distances, graph_distances, output_dir)

def create_detailed_mi_analysis(mi_values, geo_distances, graph_distances, output_dir):
    """Create detailed MI analysis plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed MI-Distance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Log-log analysis
    ax1 = axes[0, 0]
    # Filter out very small values for log plot
    mask = (mi_values > 1e-6) & (geo_distances > 1e-6)
    if np.sum(mask) > 5:
        log_mi = np.log(mi_values[mask])
        log_geo = np.log(geo_distances[mask])
        
        ax1.scatter(log_geo, log_mi, alpha=0.6, s=50)
        ax1.set_xlabel('ln(Geometric Distance)')
        ax1.set_ylabel('ln(Mutual Information)')
        ax1.set_title('Log-Log Analysis: MI vs Geometric Distance')
        ax1.grid(True, alpha=0.3)
        
        # Fit power law
        try:
            z = np.polyfit(log_geo, log_mi, 1)
            p = np.poly1d(z)
            ax1.plot(log_geo, p(log_geo), "r--", alpha=0.8, 
                    label=f'Power law: y ∝ x^{z[0]:.2f}')
            ax1.legend()
        except:
            pass
    
    # Plot 2: MI distribution
    ax2 = axes[0, 1]
    ax2.hist(mi_values, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Mutual Information')
    ax2.set_ylabel('Frequency')
    ax2.set_title('MI Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distance distribution comparison
    ax3 = axes[1, 0]
    ax3.hist(geo_distances, bins=20, alpha=0.7, label='Geometric', edgecolor='black')
    ax3.hist(graph_distances, bins=20, alpha=0.7, label='Graph', edgecolor='black')
    ax3.set_xlabel('Distance')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distance Distribution Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Correlation scatter matrix
    ax4 = axes[1, 1]
    data = pd.DataFrame({
        'MI': mi_values,
        'Geo_Dist': geo_distances,
        'Graph_Dist': graph_distances
    })
    
    # Create correlation matrix
    corr_matrix = data.corr()
    im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax4.set_xticks(range(len(corr_matrix.columns)))
    ax4.set_yticks(range(len(corr_matrix.columns)))
    ax4.set_xticklabels(corr_matrix.columns)
    ax4.set_yticklabels(corr_matrix.columns)
    ax4.set_title('Correlation Matrix')
    
    # Add correlation values
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / 'detailed_mi_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Detailed MI analysis plot saved to: {plot_path}")
    plt.show()

def save_analysis_results(mi_values, geo_distances, graph_distances,
                         geo_corr, graph_corr, geo_spearman, graph_spearman,
                         output_dir):
    """Save analysis results to files"""
    
    # Save summary statistics
    summary = {
        'total_pairs': len(mi_values),
        'non_zero_pairs': len(mi_values),
        'mi_statistics': {
            'mean': float(np.mean(mi_values)),
            'std': float(np.std(mi_values)),
            'min': float(np.min(mi_values)),
            'max': float(np.max(mi_values)),
            'median': float(np.median(mi_values))
        },
        'geometric_distance_statistics': {
            'mean': float(np.mean(geo_distances)),
            'std': float(np.std(geo_distances)),
            'min': float(np.min(geo_distances)),
            'max': float(np.max(geo_distances)),
            'median': float(np.median(geo_distances))
        },
        'graph_distance_statistics': {
            'mean': float(np.mean(graph_distances)),
            'std': float(np.std(graph_distances)),
            'min': float(np.min(graph_distances)),
            'max': float(np.max(graph_distances)),
            'median': float(np.median(graph_distances))
        },
        'correlations': {
            'geometric_pearson': float(geo_corr),
            'graph_pearson': float(graph_corr),
            'geometric_spearman': float(geo_spearman),
            'graph_spearman': float(graph_spearman)
        }
    }
    
    # Save to JSON
    summary_path = Path(output_dir) / 'mi_distance_correlation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save raw data
    data_df = pd.DataFrame({
        'mutual_information': mi_values,
        'geometric_distance': geo_distances,
        'graph_distance': graph_distances
    })
    
    data_path = Path(output_dir) / 'mi_distance_correlation_data.csv'
    data_df.to_csv(data_path, index=False)
    
    # Create text summary
    text_summary = f"""MI-DISTANCE CORRELATION ANALYSIS SUMMARY
{'='*60}

CORRELATION RESULTS:
-------------------
Geometric Distance (Pearson): {geo_corr:.6f}
Graph Distance (Pearson): {graph_corr:.6f}
Geometric Distance (Spearman): {geo_spearman:.6f}
Graph Distance (Spearman): {graph_spearman:.6f}

MI STATISTICS:
--------------
Mean MI: {np.mean(mi_values):.6f}
Std MI: {np.std(mi_values):.6f}
Min MI: {np.min(mi_values):.6f}
Max MI: {np.max(mi_values):.6f}

DISTANCE STATISTICS:
-------------------
Geometric Distance:
  Mean: {np.mean(geo_distances):.6f}
  Std: {np.std(geo_distances):.6f}
  Range: [{np.min(geo_distances):.6f}, {np.max(geo_distances):.6f}]

Graph Distance:
  Mean: {np.mean(graph_distances):.6f}
  Std: {np.std(graph_distances):.6f}
  Range: [{np.min(graph_distances):.6f}, {np.max(graph_distances):.6f}]

INTERPRETATION:
--------------
This analysis tests the boundary-bulk correspondence by examining the relationship
between mutual information (boundary observable) and geometric distance (bulk property).

Strong negative correlations indicate that:
- Higher mutual information corresponds to shorter geometric distances
- The bulk geometry emerges from boundary entanglement patterns
- The holographic principle is manifest in the quantum simulation

The comparison between geometric and graph distances reveals:
- How well the emergent geometry captures the entanglement structure
- Whether the bulk geometry is locally consistent with boundary correlations
- The effectiveness of the MDS embedding in preserving distance relationships
"""
    
    text_path = Path(output_dir) / 'mi_distance_correlation_summary.txt'
    with open(text_path, 'w') as f:
        f.write(text_summary)
    
    print(f"✓ Analysis results saved to:")
    print(f"  - {summary_path}")
    print(f"  - {data_path}")
    print(f"  - {text_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python mi_distance_correlation_analysis.py <results_file> <output_dir>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    analyze_mi_distance_correlation(results_file, output_dir) 