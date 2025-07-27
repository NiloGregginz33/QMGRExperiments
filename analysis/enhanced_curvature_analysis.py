#!/usr/bin/env python3
"""
Enhanced Curvature Analysis Script

This script generates comprehensive visualizations for custom curvature experiments:
1. 2D Embedding Scatter (Annotated) with neighbor connections
2. 3D Lorentzian Embedding (3-axis scatter) with color coding
3. Edge MI vs. Graph Distance scatter plot

All plots are saved to the instance subdirectory of the experiment being analyzed.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from pathlib import Path
import argparse
import sys
import os

def load_experiment_results(json_path):
    """Load experiment results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def create_2d_embedding_scatter(data, output_dir):
    """
    Create 2D embedding scatter plot with qubit labels and mutual information weighted edges.
    
    Args:
        data: Experiment results dictionary
        output_dir: Directory to save the plot
    """
    if 'embedding_coords' not in data:
        print("⚠️  No 2D embedding coordinates found in data")
        return
    
    coords = np.array(data['embedding_coords'])
    num_qubits = len(coords)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot qubit positions
    ax.scatter(coords[:, 0], coords[:, 1], c='blue', s=200, alpha=0.7, edgecolors='black', linewidth=2)
    
    # Add qubit labels
    for i in range(num_qubits):
        ax.annotate(f'Q{i}', (coords[i, 0], coords[i, 1]), 
                   xytext=(5, 5), textcoords='offset points', 
                   fontsize=12, fontweight='bold')
    
    # Draw mutual information weighted edges if MI data is available
    mi_data = None
    if 'mi_matrix' in data:
        mi_data = np.array(data['mi_matrix'])
    elif 'edge_mi_per_timestep' in data:
        # Use the last timestep's MI data
        edge_mi_dict = data['edge_mi_per_timestep'][-1]
        # Convert edge MI dictionary to matrix format
        mi_data = np.zeros((num_qubits, num_qubits))
        for edge_key, mi_val in edge_mi_dict.items():
            i, j = map(int, edge_key.split(','))
            mi_data[i, j] = mi_val
            mi_data[j, i] = mi_val  # Make it symmetric
    
    if mi_data is not None:
        # Get all pairwise MI values
        mi_values = []
        edge_pairs = []
        
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                mi_val = mi_data[i, j]
                if mi_val > 0:  # Only plot edges with positive MI
                    mi_values.append(mi_val)
                    edge_pairs.append((i, j))
        
        if mi_values:
            # Normalize MI values for line width and alpha
            mi_min, mi_max = min(mi_values), max(mi_values)
            mi_range = mi_max - mi_min if mi_max > mi_min else 1
            
            # Plot MI-weighted edges
            for (i, j), mi_val in zip(edge_pairs, mi_values):
                # Normalize MI value for visualization
                norm_mi = (mi_val - mi_min) / mi_range if mi_range > 0 else 0.5
                
                # Line width scales with MI strength (min 0.5, max 4.0)
                linewidth = 0.5 + 3.5 * norm_mi
                
                # Alpha scales with MI strength (min 0.1, max 0.8)
                alpha = 0.1 + 0.7 * norm_mi
                
                # Color from blue (weak) to red (strong)
                color = plt.cm.coolwarm(norm_mi)
                
                ax.plot([coords[i, 0], coords[j, 0]], 
                       [coords[i, 1], coords[j, 1]], 
                       color=color, alpha=alpha, linewidth=linewidth)
    
    # Draw neighbor connections if graph data is available (as dashed lines)
    if 'graph_edges' in data:
        edges = data['graph_edges']
        for edge in edges:
            i, j = edge[0], edge[1]
            if i < len(coords) and j < len(coords):
                ax.plot([coords[i, 0], coords[j, 0]], 
                       [coords[i, 1], coords[j, 1]], 
                       'k--', alpha=0.4, linewidth=1.0)
    
    # Add experiment info
    title_parts = []
    if 'num_qubits' in data:
        title_parts.append(f"N={data['num_qubits']}")
    if 'geometry' in data:
        title_parts.append(f"Geometry: {data['geometry']}")
    if 'curvature' in data:
        title_parts.append(f"κ={data['curvature']:.2f}")
    
    title = f"2D Embedding with MI-Weighted Edges: {' | '.join(title_parts)}"
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add colorbar for MI values
    if 'mi_matrix' in data and mi_values:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(mi_min, mi_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Mutual Information', fontsize=12)
    
    # Add legend
    ax.scatter([], [], c='blue', s=200, alpha=0.7, edgecolors='black', linewidth=2, label='Qubits')
    if 'mi_matrix' in data and mi_values:
        ax.plot([], [], color=plt.cm.coolwarm(0.5), alpha=0.5, linewidth=2, label='MI-Weighted Edges')
    if 'graph_edges' in data:
        ax.plot([], [], 'k--', alpha=0.4, linewidth=1.0, label='Neighbor Connections')
    ax.legend(loc='upper right')
    
    # Save plot
    output_path = Path(output_dir) / "2d_embedding_scatter_annotated.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 2D embedding scatter plot with MI-weighted edges saved to: {output_path}")

    # In create_2d_embedding_scatter, after plotting MI-weighted edges, also save as '2d_embedding_scatter_mi_weighted.png'
    # In create_edge_mi_vs_distance_plot, ensure the plot is always saved as 'edge_mi_vs_distance.png' and not skipped if MI data is present.
    # In create_comprehensive_summary, mention both files in the output section.

    # After the existing plt.savefig(output_path, ...):
    output_path_mi = Path(output_dir) / "2d_embedding_scatter_mi_weighted.png"
    plt.savefig(output_path_mi, dpi=300, bbox_inches='tight')
    print(f"✅ 2D embedding scatter plot with MI-weighted edges saved to: {output_path_mi}")

def create_3d_lorentzian_embedding(data, output_dir):
    """
    Create 3D Lorentzian embedding scatter plot with color coding.
    
    Args:
        data: Experiment results dictionary
        output_dir: Directory to save the plot
    """
    if 'lorentzian_embedding' not in data:
        print("⚠️  No Lorentzian embedding coordinates found in data")
        return
    
    lorentzian_coords = np.array(data['lorentzian_embedding'])
    num_qubits = lorentzian_coords.shape[0]
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color code by qubit index
    colors = plt.cm.viridis(np.linspace(0, 1, num_qubits))
    
    # Plot 3D scatter
    scatter = ax.scatter(lorentzian_coords[:, 0], 
                        lorentzian_coords[:, 1], 
                        lorentzian_coords[:, 2], 
                        c=colors, s=100, alpha=0.8, edgecolors='black', linewidth=1)
    
    # Add qubit labels
    for i in range(num_qubits):
        ax.text(lorentzian_coords[i, 0], lorentzian_coords[i, 1], lorentzian_coords[i, 2], 
                f'Q{i}', fontsize=10, fontweight='bold')
    
    # Add experiment info
    title_parts = []
    if 'num_qubits' in data:
        title_parts.append(f"N={data['num_qubits']}")
    if 'geometry' in data:
        title_parts.append(f"Geometry: {data['geometry']}")
    if 'curvature' in data:
        title_parts.append(f"κ={data['curvature']:.2f}")
    
    title = f"3D Lorentzian Embedding: {' | '.join(title_parts)}"
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('X (Spatial)', fontsize=12)
    ax.set_ylabel('Y (Spatial)', fontsize=12)
    ax.set_zlabel('T (Temporal)', fontsize=12)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, num_qubits-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Qubit Index', fontsize=12)
    
    # Save plot
    output_path = Path(output_dir) / "3d_lorentzian_embedding.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 3D Lorentzian embedding plot saved to: {output_path}")

def create_edge_mi_vs_distance_plot(data, output_dir):
    """
    Create scatter plot of edge MI vs. graph distance.
    
    Args:
        data: Experiment results dictionary
        output_dir: Directory to save the plot
    """
    # Check for different MI data formats
    mi_data = None
    if 'mi_matrix' in data:
        mi_data = np.array(data['mi_matrix'])
    elif 'edge_mi_per_timestep' in data:
        # Use the last timestep's MI data
        edge_mi_dict = data['edge_mi_per_timestep'][-1]
        # Convert edge MI dictionary to matrix format
        if 'spec' in data and 'num_qubits' in data['spec']:
            num_qubits = data['spec']['num_qubits']
        else:
            num_qubits = max([int(i) for k in edge_mi_dict.keys() for i in k.split(',')]) + 1
        mi_data = np.zeros((num_qubits, num_qubits))
        for edge_key, mi_val in edge_mi_dict.items():
            i, j = map(int, edge_key.split(','))
            mi_data[i, j] = mi_val
            mi_data[j, i] = mi_val  # Make it symmetric
    else:
        print("⚠️  No MI matrix or edge_mi_per_timestep found in data")
        # Plot a blank graph with a message
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title('Edge MI vs. Graph Distance: No MI data found', fontsize=16, fontweight='bold')
        ax.set_xlabel('Shortest Path Distance', fontsize=12)
        ax.set_ylabel('Mutual Information', fontsize=12)
        ax.text(0.5, 0.5, 'No MI data available', fontsize=18, ha='center', va='center', transform=ax.transAxes)
        plt.tight_layout()
        output_path = Path(output_dir) / "edge_mi_vs_distance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"⚠️  No MI data: blank plot saved to: {output_path}")
        return
    
    # Get graph edges from custom_edges if available
    edges = []
    if 'graph_edges' in data:
        edges = data['graph_edges']
    elif 'spec' in data and 'custom_edges' in data['spec']:
        # Parse custom_edges string
        custom_edges_str = data['spec']['custom_edges']
        edges = []
        for edge_str in custom_edges_str.split(','):
            if ':' in edge_str:
                edge_part = edge_str.split(':')[0]
                i, j = map(int, edge_part.split('-'))
                edges.append([i, j])
    
    if not edges:
        print("⚠️  No graph edges found in data")
        # Plot a blank graph with a message
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title('Edge MI vs. Graph Distance: No graph edges', fontsize=16, fontweight='bold')
        ax.set_xlabel('Shortest Path Distance', fontsize=12)
        ax.set_ylabel('Mutual Information', fontsize=12)
        ax.text(0.5, 0.5, 'No graph edges available', fontsize=18, ha='center', va='center', transform=ax.transAxes)
        plt.tight_layout()
        output_path = Path(output_dir) / "edge_mi_vs_distance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"⚠️  No graph edges: blank plot saved to: {output_path}")
        return
    
    # Determine num_qubits from spec if available
    if 'spec' in data and 'num_qubits' in data['spec']:
        num_qubits = data['spec']['num_qubits']
    else:
        num_qubits = len(mi_data)
    
    # Calculate shortest path distances
    G = nx.Graph()
    for i in range(num_qubits):
        G.add_node(i)
    
    for edge in edges:
        i, j = edge[0], edge[1]
        if i < num_qubits and j < num_qubits:
            G.add_edge(i, j)
    
    # Get all pairwise distances
    distances = []
    mi_values = []
    
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            try:
                dist = nx.shortest_path_length(G, i, j)
                mi_val = mi_data[i, j]
                distances.append(dist)
                mi_values.append(mi_val)
            except nx.NetworkXNoPath:
                # No path exists between these nodes
                continue
    
    # Debug output
    print(f"[DEBUG] Number of points to plot: {len(distances)}")
    if distances:
        print(f"[DEBUG] Sample distances: {distances[:10]}")
        print(f"[DEBUG] Sample MI values: {mi_values[:10]}")
    else:
        print("[DEBUG] No distances to plot.")
    
    # If all MI values are zero or nearly zero, plot a message
    if not distances or np.allclose(mi_values, 0):
        print("⚠️  All MI values are zero or no valid data. Plotting message.")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title('Edge MI vs. Graph Distance', fontsize=16, fontweight='bold')
        ax.set_xlabel('Shortest Path Distance', fontsize=12)
        ax.set_ylabel('Mutual Information', fontsize=12)
        ax.text(0.5, 0.5, 'All MI values are zero or no valid data', fontsize=18, ha='center', va='center', transform=ax.transAxes)
        plt.tight_layout()
        output_path = Path(output_dir) / "edge_mi_vs_distance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"⚠️  All MI zero: blank plot saved to: {output_path}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot
    scatter = ax.scatter(distances, mi_values, c=mi_values, cmap='viridis', 
                        s=100, alpha=0.7, edgecolors='black', linewidth=1)
    
    # Add trend line
    if len(distances) > 1:
        z = np.polyfit(distances, mi_values, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(distances), max(distances), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, 
                label=f'Trend: y = {z[0]:.3f}x + {z[1]:.3f}')
    
    # Add experiment info
    title_parts = []
    if 'num_qubits' in data:
        title_parts.append(f"N={data['num_qubits']}")
    if 'geometry' in data:
        title_parts.append(f"Geometry: {data['geometry']}")
    if 'curvature' in data:
        title_parts.append(f"κ={data['curvature']:.2f}")
    
    title = f"Edge MI vs. Graph Distance: {' | '.join(title_parts)}"
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Shortest Path Distance', fontsize=12)
    ax.set_ylabel('Mutual Information', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('MI Value', fontsize=12)
    
    # Add legend if trend line was plotted
    if len(distances) > 1:
        ax.legend(loc='upper right')
    
    # Add statistics
    if len(distances) > 0:
        correlation = np.corrcoef(distances, mi_values)[0, 1]
        ax.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
                transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    # If very few points, add a note
    if len(distances) < 5:
        ax.text(0.5, 0.1, 'Few data points', fontsize=14, ha='center', va='center', transform=ax.transAxes, color='red')
    
    # Save plot
    output_path = Path(output_dir) / "edge_mi_vs_distance.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Edge MI vs. distance plot saved to: {output_path}")

def create_comprehensive_summary(data, output_dir):
    """
    Create a comprehensive summary of all visualizations.
    
    Args:
        data: Experiment results dictionary
        output_dir: Directory to save the summary
    """
    summary_path = Path(output_dir) / "enhanced_analysis_summary.txt"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("ENHANCED CURVATURE ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        # Experiment parameters
        f.write("EXPERIMENT PARAMETERS:\n")
        f.write("-" * 25 + "\n")
        if 'num_qubits' in data:
            f.write(f"Number of qubits: {data['num_qubits']}\n")
        if 'geometry' in data:
            f.write(f"Geometry: {data['geometry']}\n")
        if 'curvature' in data:
            f.write(f"Curvature (κ): {data['curvature']:.4f}\n")
        if 'topology' in data:
            f.write(f"Topology: {data['topology']}\n")
        if 'timesteps' in data:
            f.write(f"Timesteps: {data['timesteps']}\n")
        f.write("\n")
        
        # Available data
        f.write("AVAILABLE DATA:\n")
        f.write("-" * 15 + "\n")
        f.write(f"2D Embedding: {'✓' if 'embedding_coords' in data else '✗'}\n")
        f.write(f"3D Lorentzian: {'✓' if 'lorentzian_embedding' in data else '✗'}\n")
        f.write(f"MI Matrix: {'✓' if 'mi_matrix' in data or 'edge_mi_per_timestep' in data else '✗'}\n")
        f.write(f"Graph Edges: {'✓' if 'graph_edges' in data or ('spec' in data and 'custom_edges' in data['spec']) else '✗'}\n")
        f.write(f"Entropy Data: {'✓' if 'entropy_per_timestep' in data else '✗'}\n")
        f.write("\n")
        
        # Key metrics
        if 'mi_matrix' in data:
            mi_matrix = np.array(data['mi_matrix'])
            f.write("KEY METRICS:\n")
            f.write("-" * 12 + "\n")
            f.write(f"Average MI: {np.mean(mi_matrix):.4f}\n")
            f.write(f"Max MI: {np.max(mi_matrix):.4f}\n")
            f.write(f"Min MI: {np.min(mi_matrix):.4f}\n")
            f.write(f"MI Standard Deviation: {np.std(mi_matrix):.4f}\n")
            f.write("\n")
        
        # Generated plots
        f.write("GENERATED PLOTS:\n")
        f.write("-" * 16 + "\n")
        f.write("1. 2D Embedding Scatter with MI-Weighted Edges - 2d_embedding_scatter_mi_weighted.png\n")
        f.write("   - Shows MI-weighted edges between all qubit pairs in 2D embedding\n")
        f.write("   - Color and thickness indicate MI strength\n\n")
        f.write("2. Edge MI vs. Graph Distance - edge_mi_vs_distance.png\n")
        f.write("   - Scatter plot of MI vs. shortest path distance\n")
        f.write("   - Tests bulk locality hypothesis\n\n")
        
        f.write("ANALYSIS NOTES:\n")
        f.write("-" * 16 + "\n")
        f.write("- These visualizations help understand emergent geometry from quantum entanglement\n")
        f.write("- The 2D embedding shows the spatial structure of the quantum system\n")
        f.write("- The 3D Lorentzian embedding reveals causal structure and temporal evolution\n")
        f.write("- The MI vs. distance plot tests whether closer qubits have higher entanglement\n")
        f.write("- All plots are saved in the experiment instance directory for easy reference\n")
    
    print(f"✅ Enhanced analysis summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced curvature analysis with advanced visualizations')
    parser.add_argument('json_path', help='Path to experiment results JSON file')
    parser.add_argument('--output-dir', help='Output directory (defaults to same directory as JSON file)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"📁 Loading experiment results from: {args.json_path}")
    try:
        data = load_experiment_results(args.json_path)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.json_path).parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Generate visualizations
    print("\n🎨 Generating enhanced visualizations...")
    
    create_2d_embedding_scatter(data, output_dir)
    create_3d_lorentzian_embedding(data, output_dir)
    create_edge_mi_vs_distance_plot(data, output_dir)
    create_comprehensive_summary(data, output_dir)
    
    print(f"\n✅ Enhanced analysis complete! All plots saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - 2d_embedding_scatter_annotated.png")
    print("  - 2d_embedding_scatter_mi_weighted.png")
    print("  - 3d_lorentzian_embedding.png") 
    print("  - edge_mi_vs_distance.png")
    print("  - enhanced_analysis_summary.txt")

if __name__ == "__main__":
    main() 