#!/usr/bin/env python3
"""
Add Error Bars to Existing Instance Directory Graphs
==================================================

This script adds error bars to existing graphs in the instance directory and replaces them
with enhanced versions that include error bars, confidence intervals, and statistical annotations.

Features:
- Reads error analysis results from error_analysis_results.json
- Enhances existing graphs with error bars and confidence intervals
- Replaces original graphs with enhanced versions
- Maintains publication-quality formatting
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

def load_error_analysis_data(instance_dir: str) -> Dict:
    """Load error analysis data from the instance directory."""
    error_file = Path(instance_dir) / "error_analysis_results.json"
    
    if not error_file.exists():
        print(f"Error: {error_file} not found")
        return None
    
    with open(error_file, 'r') as f:
        return json.load(f)

def enhance_mi_plot(instance_dir: str, error_data: Dict):
    """Enhance existing MI plot with error bars."""
    mi_analysis = error_data['mi_analysis']
    spec = error_data['experiment_spec']
    
    timesteps = mi_analysis['timesteps']
    mi_means = mi_analysis['mi_means']
    mi_errors = mi_analysis['mi_errors']
    ci_lower = mi_analysis['mi_ci_lower']
    ci_upper = mi_analysis['mi_ci_upper']
    confidence_level = mi_analysis['confidence_level']
    
    plt.figure(figsize=(12, 8))
    
    # Main plot with error bars
    plt.errorbar(timesteps, mi_means, yerr=mi_errors, 
                fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8,
                label=f'MI ± {confidence_level*100:.0f}% CI', color='#2E86AB')
    
    # Confidence interval shading
    plt.fill_between(timesteps, ci_lower, ci_upper, alpha=0.3, 
                    color='#2E86AB', label=f'{confidence_level*100:.0f}% Confidence Interval')
    
    plt.xlabel('Timestep', fontsize=14)
    plt.ylabel('Mutual Information', fontsize=14)
    plt.title(f'Mutual Information Evolution with Error Analysis\n'
              f'{spec["num_qubits"]} qubits, {spec["geometry"]} geometry, κ={spec["curvature"]}, {spec["device"]}', 
              fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistical summary
    mi_mean_overall = np.mean(mi_means)
    mi_std_overall = np.std(mi_means)
    plt.text(0.02, 0.98, f'Mean MI: {mi_mean_overall:.6f} ± {mi_std_overall:.6f}', 
            transform=plt.gca().transAxes, fontsize=12, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save enhanced plot (replace existing)
    output_path = Path(instance_dir) / "mi_evolution_with_errors.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced MI plot saved to: {output_path}")

def enhance_entropy_plot(instance_dir: str, error_data: Dict):
    """Enhance existing entropy plot with error bars."""
    entropy_analysis = error_data['entropy_analysis']
    spec = error_data['experiment_spec']
    
    timesteps = entropy_analysis['timesteps']
    entropy_means = entropy_analysis['entropy_means']
    entropy_errors = entropy_analysis['entropy_errors']
    ci_lower = entropy_analysis['entropy_ci_lower']
    ci_upper = entropy_analysis['entropy_ci_upper']
    
    plt.figure(figsize=(12, 8))
    
    # Main plot with error bars
    plt.errorbar(timesteps, entropy_means, yerr=entropy_errors, 
                fmt='s-', capsize=5, capthick=2, linewidth=2, markersize=8,
                label=f'Entropy ± 95% CI', color='#A23B72')
    
    # Confidence interval shading
    plt.fill_between(timesteps, ci_lower, ci_upper, alpha=0.3, 
                    color='#A23B72', label='95% Confidence Interval')
    
    # Add physical constraint line at y=0
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1,
               label='Physical Constraint (Entropy ≥ 0)')
    
    plt.xlabel('Timestep', fontsize=14)
    plt.ylabel('Entropy (bits)', fontsize=14)
    plt.title(f'Entropy Evolution with Error Analysis\n'
              f'{spec["num_qubits"]} qubits, {spec["geometry"]} geometry, κ={spec["curvature"]}, {spec["device"]}', 
              fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistical summary
    entropy_mean_overall = np.mean(entropy_means)
    entropy_std_overall = np.std(entropy_means)
    plt.text(0.02, 0.98, f'Mean Entropy: {entropy_mean_overall:.4f} ± {entropy_std_overall:.4f}', 
            transform=plt.gca().transAxes, fontsize=12, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save enhanced plot (replace existing)
    output_path = Path(instance_dir) / "entropy_evolution_with_errors.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced entropy plot saved to: {output_path}")

def enhance_correlation_plot(instance_dir: str, error_data: Dict):
    """Enhance existing correlation plot with error bars."""
    # Check if correlation data exists
    correlation_file = Path(instance_dir) / "mi_distance_correlation_data.csv"
    
    if not correlation_file.exists():
        print(f"MI-distance correlation data not found: {correlation_file}")
        return
    
    # Load correlation data
    import pandas as pd
    df = pd.read_csv(correlation_file)
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    plt.scatter(df['geometric_distance'], df['mutual_information'], 
               s=50, alpha=0.7, color='#F18F01', label='Data points')
    
    # Fit line if enough data points
    if len(df) > 1:
        from scipy import stats
        
        x_col = 'geometric_distance'
        y_col = 'mutual_information'
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(df[x_col], df[y_col])
        
        x_fit = np.linspace(df[x_col].min(), df[x_col].max(), 100)
        y_fit = slope * x_fit + intercept
        
        plt.plot(x_fit, y_fit, 'r--', linewidth=2, 
                label=f'Linear fit (R² = {r_value**2:.3f}, p = {p_value:.3f})')
    
    plt.xlabel('Geometric Distance', fontsize=14)
    plt.ylabel('Mutual Information', fontsize=14)
    plt.title('MI vs Distance Correlation Analysis', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save enhanced plot (replace existing)
    output_path = Path(instance_dir) / "mi_distance_correlation_with_errors.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced MI-distance correlation plot saved to: {output_path}")

def main():
    """Main function to add error bars to existing instance directory graphs."""
    if len(sys.argv) != 2:
        print("Usage: python add_error_bars_to_existing_graphs.py <instance_directory>")
        print("Example: python add_error_bars_to_existing_graphs.py experiment_logs/custom_curvature_experiment/instance_20250726_153536")
        return
    
    instance_dir = sys.argv[1]
    
    if not Path(instance_dir).exists():
        print(f"Error: Directory {instance_dir} does not exist")
        return
    
    print(f"Adding error bars to existing graphs in: {instance_dir}")
    
    # Load error analysis data
    error_data = load_error_analysis_data(instance_dir)
    if error_data is None:
        return
    
    print("Loaded error analysis data successfully")
    
    # Create enhanced plots
    enhance_mi_plot(instance_dir, error_data)
    enhance_entropy_plot(instance_dir, error_data)
    enhance_correlation_plot(instance_dir, error_data)
    
    print("\n✅ Successfully enhanced all graphs with error bars!")
    print("Enhanced plots saved in the instance directory")

if __name__ == "__main__":
    main()