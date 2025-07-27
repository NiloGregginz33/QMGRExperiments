#!/usr/bin/env python3
"""
Spherical Geometry Analysis
Analyzes spherical geometry experiment results with focus on curvature detection and geometric properties.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy import stats

def load_experiment_data(json_path):
    """Load experiment data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def analyze_spherical_curvature(data):
    """Analyze spherical curvature characteristics."""
    print("=" * 70)
    print("SPHERICAL GEOMETRY ANALYSIS")
    print("=" * 70)
    
    spec = data['spec']
    geometry = spec['geometry']
    curvature = spec['curvature']
    num_qubits = spec['num_qubits']
    
    print(f"\nEXPERIMENT PARAMETERS:")
    print(f"  Geometry: {geometry}")
    print(f"  Curvature: {curvature}")
    print(f"  Number of qubits: {num_qubits}")
    print(f"  Device: {spec['device']}")
    print(f"  Timesteps: {spec['timesteps']}")
    
    # Analyze embedding coordinates if available
    if 'embedding_coords' in data:
        coords = np.array(data['embedding_coords'])
        print(f"\nEMBEDDING ANALYSIS:")
        print(f"  2D embedding shape: {coords.shape}")
        print(f"  Coordinate range: [{coords.min():.4f}, {coords.max():.4f}]")
        
        # Calculate distances between points
        distances = []
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])
                distances.append(dist)
        
        distances = np.array(distances)
        print(f"  Mean distance: {np.mean(distances):.4f}")
        print(f"  Distance std: {np.std(distances):.4f}")
        print(f"  Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
    
    # Analyze evolution data if available
    if 'evolution_summary' in data:
        evolution = data['evolution_summary']
        print(f"\nEVOLUTION ANALYSIS:")
        print(f"  Total timesteps: {evolution['total_timesteps']}")
        print(f"  Gromov delta range: {evolution['gromov_delta_range']}")
        print(f"  Mean distance range: {evolution['mean_distance_range']}")
        print(f"  Triangle violations: {evolution['total_triangle_violations']}")
        
        # Check if hyperbolic evolution was detected
        if 'hyperbolic_evolution' in evolution:
            hyperbolic_steps = sum(evolution['hyperbolic_evolution'])
            print(f"  Hyperbolic evolution detected in {hyperbolic_steps}/{evolution['total_timesteps']} timesteps")
    
    # Analyze Lorentzian embedding if available
    if 'lorentzian_embedding' in data:
        lorentzian = np.array(data['lorentzian_embedding'])
        print(f"\nLORENTZIAN EMBEDDING ANALYSIS:")
        print(f"  3D Lorentzian shape: {lorentzian.shape}")
        
        # Calculate time-like and space-like separations
        time_component = lorentzian[:, 0]  # First component is time
        space_components = lorentzian[:, 1:]  # Last two components are space
        
        print(f"  Time component range: [{time_component.min():.4f}, {time_component.max():.4f}]")
        print(f"  Space component range: [{space_components.min():.4f}, {space_components.max():.4f}]")
    
    return data

def create_spherical_analysis_plots(data, output_dir):
    """Create analysis plots for spherical geometry."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Embedding coordinates if available
    if 'embedding_coords' in data:
        coords = np.array(data['embedding_coords'])
        plt.figure(figsize=(10, 8))
        plt.scatter(coords[:, 0], coords[:, 1], c=range(len(coords)), cmap='viridis', s=100)
        plt.colorbar(label='Qubit Index')
        
        # Add qubit labels
        for i, (x, y) in enumerate(coords):
            plt.annotate(f'Q{i}', (x, y), xytext=(5, 5), textcoords='offset points')
        
        plt.title(f'Spherical Geometry 2D Embedding\nCurvature: {data["spec"]["curvature"]}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'spherical_embedding_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Distance distribution
    if 'embedding_coords' in data:
        coords = np.array(data['embedding_coords'])
        distances = []
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])
                distances.append(dist)
        
        plt.figure(figsize=(10, 6))
        plt.hist(distances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(distances), color='red', linestyle='--', label=f'Mean: {np.mean(distances):.4f}')
        plt.title(f'Distance Distribution in Spherical Geometry\nCurvature: {data["spec"]["curvature"]}')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'spherical_distance_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Evolution analysis if available
    if 'evolution_summary' in data:
        evolution = data['evolution_summary']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gromov delta evolution
        if 'gromov_delta_range' in evolution:
            ax1.plot([0, 1], evolution['gromov_delta_range'], 'o-', color='blue', linewidth=2, markersize=8)
            ax1.set_title('Gromov Delta Evolution')
            ax1.set_xlabel('Timestep (Normalized)')
            ax1.set_ylabel('Gromov Delta')
            ax1.grid(True, alpha=0.3)
        
        # Mean distance evolution
        if 'mean_distance_range' in evolution:
            ax2.plot([0, 1], evolution['mean_distance_range'], 'o-', color='red', linewidth=2, markersize=8)
            ax2.set_title('Mean Distance Evolution')
            ax2.set_xlabel('Timestep (Normalized)')
            ax2.set_ylabel('Mean Distance')
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Spherical Geometry Evolution Analysis\nCurvature: {data["spec"]["curvature"]}')
        plt.tight_layout()
        plt.savefig(output_dir / 'spherical_evolution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_comprehensive_summary(data, output_dir):
    """Create a comprehensive analysis summary."""
    output_dir = Path(output_dir)
    
    summary_path = output_dir / 'spherical_analysis_summary.txt'
    
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SPHERICAL GEOMETRY COMPREHENSIVE ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        
        # Experiment parameters
        spec = data['spec']
        f.write("EXPERIMENT PARAMETERS:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Geometry: {spec['geometry']}\n")
        f.write(f"Curvature: {spec['curvature']}\n")
        f.write(f"Number of qubits: {spec['num_qubits']}\n")
        f.write(f"Device: {spec['device']}\n")
        f.write(f"Timesteps: {spec['timesteps']}\n")
        f.write(f"Topology: {spec['topology']}\n\n")
        
        # Available data
        f.write("AVAILABLE DATA:\n")
        f.write("-" * 15 + "\n")
        f.write(f"2D Embedding: {'✓' if 'embedding_coords' in data else '✗'}\n")
        f.write(f"3D Lorentzian: {'✓' if 'lorentzian_embedding' in data else '✗'}\n")
        f.write(f"Evolution Summary: {'✓' if 'evolution_summary' in data else '✗'}\n")
        f.write(f"Regge Data: {'✓' if 'regge_evolution_data' in data else '✗'}\n\n")
        
        # Key findings
        f.write("KEY FINDINGS:\n")
        f.write("-" * 12 + "\n")
        
        if 'embedding_coords' in data:
            coords = np.array(data['embedding_coords'])
            f.write(f"• 2D embedding successfully generated with {coords.shape[0]} points\n")
            f.write(f"• Coordinate range: [{coords.min():.4f}, {coords.max():.4f}]\n")
        
        if 'evolution_summary' in data:
            evolution = data['evolution_summary']
            f.write(f"• Evolution tracked over {evolution['total_timesteps']} timesteps\n")
            f.write(f"• Gromov delta range: {evolution['gromov_delta_range']}\n")
            f.write(f"• Mean distance range: {evolution['mean_distance_range']}\n")
            f.write(f"• Triangle violations: {evolution['total_triangle_violations']}\n")
        
        f.write("\nSPHERICAL GEOMETRY CHARACTERISTICS:\n")
        f.write("-" * 35 + "\n")
        f.write("• Positive curvature (κ > 0) indicates spherical geometry\n")
        f.write("• Angle sums in triangles should be > π\n")
        f.write("• Distances should follow spherical trigonometry\n")
        f.write("• Gromov delta should reflect positive curvature\n\n")
        
        f.write("ANALYSIS NOTES:\n")
        f.write("-" * 16 + "\n")
        f.write("• This analysis focuses on spherical geometry characteristics\n")
        f.write("• The embedding shows the spatial structure of the quantum system\n")
        f.write("• Evolution data reveals how geometry changes over time\n")
        f.write("• All plots are saved in the experiment instance directory\n")
    
    print(f"✅ Comprehensive summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze spherical geometry experiment results')
    parser.add_argument('json_path', help='Path to experiment results JSON file')
    parser.add_argument('--output-dir', help='Output directory (defaults to same directory as JSON file)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"📁 Loading experiment results from: {args.json_path}")
    try:
        data = load_experiment_data(args.json_path)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.json_path).parent
    
    # Analyze spherical curvature
    analyze_spherical_curvature(data)
    
    # Create plots
    print("\n🎨 Generating spherical geometry analysis plots...")
    create_spherical_analysis_plots(data, output_dir)
    
    # Create summary
    create_comprehensive_summary(data, output_dir)
    
    print(f"\n✅ Spherical geometry analysis complete! All results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - spherical_embedding_analysis.png")
    print("  - spherical_distance_distribution.png")
    print("  - spherical_evolution_analysis.png")
    print("  - spherical_analysis_summary.txt")

if __name__ == "__main__":
    main() 