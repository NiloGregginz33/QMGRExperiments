#!/usr/bin/env python3
"""
Simple RT Surface vs. Boundary Entropy Plotter
Plots S(A) vs Area_RT(A) from experiment results to test holographic principle
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os

def plot_rt_surface_entropy(results_file):
    """Plot S(A) vs Area_RT(A) from experiment results"""
    
    print(f"Loading results from: {results_file}")
    
    # Load the results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract boundary entropy data
    boundary_entropies_data = results.get('boundary_entropies_per_timestep', [])
    if not boundary_entropies_data:
        print("No boundary entropy data found!")
        return
    
    # Use the last timestep for analysis
    final_boundary_data = boundary_entropies_data[-1]
    multiple_regions = final_boundary_data.get('multiple_regions', {})
    
    if not multiple_regions:
        print("No multiple regions data found!")
        return
    
    # Extract data for plotting
    entropies = []
    rt_areas = []
    region_sizes = []
    region_labels = []
    
    for region_key, region_data in multiple_regions.items():
        entropy = region_data.get('entropy', 0.0)
        rt_area = region_data.get('rt_area', 0.0)
        region = region_data.get('region', [])
        size = region_data.get('size', 0)
        
        if entropy > 0 and rt_area > 0:
            entropies.append(entropy)
            rt_areas.append(rt_area)
            region_sizes.append(size)
            region_labels.append(f"Size {size}: {region}")
    
    if len(entropies) < 3:
        print(f"Not enough data points ({len(entropies)}) for analysis")
        return
    
    print(f"Found {len(entropies)} data points for analysis")
    
    # Fit linear relationship S(A) ∝ Area_RT(A)
    slope, intercept, r_value, p_value, std_err = stats.linregress(rt_areas, entropies)
    r_squared = r_value ** 2
    
    # Calculate confidence intervals
    n = len(entropies)
    t_critical = stats.t.ppf(0.975, n-2)  # 95% confidence
    slope_ci = t_critical * std_err
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Main plot: S(A) vs Area_RT(A)
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(rt_areas, entropies, c=region_sizes, cmap='viridis', s=100, alpha=0.7)
    
    # Add linear fit line
    x_fit = np.linspace(min(rt_areas), max(rt_areas), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, 'r--', linewidth=2, 
            label=f"Fit: S(A) = {slope:.4f} × Area_RT(A) + {intercept:.4f}\nR² = {r_squared:.4f}")
    
    plt.xlabel('RT Surface Area')
    plt.ylabel('Boundary Entropy S(A)')
    plt.title('Holographic Principle: S(A) ∝ Area_RT(A)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Colorbar for region sizes
    cbar = plt.colorbar(scatter)
    cbar.set_label('Region Size')
    
    # Add data point labels
    for i, (x, y, label) in enumerate(zip(rt_areas, entropies, region_labels)):
        plt.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Region size vs entropy
    plt.subplot(2, 2, 2)
    plt.scatter(region_sizes, entropies, alpha=0.7, s=100)
    plt.xlabel('Region Size')
    plt.ylabel('Boundary Entropy S(A)')
    plt.title('Entropy vs Region Size')
    plt.grid(True, alpha=0.3)
    
    # RT area vs region size
    plt.subplot(2, 2, 3)
    plt.scatter(region_sizes, rt_areas, alpha=0.7, s=100)
    plt.xlabel('Region Size')
    plt.ylabel('RT Surface Area')
    plt.title('RT Area vs Region Size')
    plt.grid(True, alpha=0.3)
    
    # Statistics table
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    stats_text = f"""
    Linear Fit Results:
    
    Equation: S(A) = {slope:.4f} × Area_RT(A) + {intercept:.4f}
    R² = {r_squared:.4f}
    p-value = {p_value:.2e}
    Slope CI (95%): ±{slope_ci:.4f}
    
    Data Points: {len(entropies)}
    Entropy Range: [{min(entropies):.3f}, {max(entropies):.3f}]
    RT Area Range: [{min(rt_areas):.1f}, {max(rt_areas):.1f}]
    Region Sizes: {min(region_sizes)}-{max(region_sizes)}
    
    Holographic Support: {'Yes' if r_squared > 0.7 else 'Weak'}
    """
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = "rt_entropy_analysis"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'rt_surface_entropy_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    # Print detailed results
    print(f"\n=== RT Surface vs. Boundary Entropy Analysis ===")
    print(f"Linear fit: S(A) = {slope:.4f} × Area_RT(A) + {intercept:.4f}")
    print(f"R² = {r_squared:.4f}")
    print(f"p-value = {p_value:.2e}")
    print(f"95% CI for slope: ±{slope_ci:.4f}")
    print(f"\nData points:")
    for i, (entropy, rt_area, size, label) in enumerate(zip(entropies, rt_areas, region_sizes, region_labels)):
        print(f"  {i+1}. {label}: S(A)={entropy:.3f}, Area_RT(A)={rt_area:.1f}")
    
    # Test pure state condition
    print(f"\n=== Pure State Test ===")
    pure_state_tests = []
    for i, (entropy, rt_area, size, label) in enumerate(zip(entropies, rt_areas, region_sizes, region_labels)):
        # Find complementary region
        total_qubits = max(region_sizes) + 1  # Estimate total qubits
        complement_size = total_qubits - size
        
        # Find entropy of complementary region
        complement_entropy = None
        for j, (other_entropy, other_rt_area, other_size, other_label) in enumerate(zip(entropies, rt_areas, region_sizes, region_labels)):
            if other_size == complement_size:
                complement_entropy = other_entropy
                break
        
        if complement_entropy is not None:
            entropy_diff = abs(entropy - complement_entropy)
            is_pure = entropy_diff < 0.1
            pure_state_tests.append(is_pure)
            print(f"  {label}: S(A)={entropy:.3f}, S(B)={complement_entropy:.3f}, diff={entropy_diff:.3f}, pure={is_pure}")
    
    if pure_state_tests:
        pure_fraction = sum(pure_state_tests) / len(pure_state_tests)
        print(f"Pure state fraction: {pure_fraction:.2f}")
    
    plt.show()
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'p_value': p_value,
        'slope_ci': slope_ci,
        'entropies': entropies,
        'rt_areas': rt_areas,
        'region_sizes': region_sizes
    }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_rt_entropy.py <results_file>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    if not os.path.exists(results_file):
        print(f"Error: File {results_file} not found!")
        sys.exit(1)
    
    plot_rt_surface_entropy(results_file) 