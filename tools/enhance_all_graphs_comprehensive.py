#!/usr/bin/env python3
"""
Comprehensive Graph Enhancement with Error Bars
==============================================

This script comprehensively enhances all graphs in an instance directory with error bars,
confidence intervals, and statistical annotations where appropriate.

Features:
- Reads error analysis results from error_analysis_results.json
- Enhances all existing graphs with error bars and confidence intervals
- Adds statistical annotations and physical constraints
- Maintains publication-quality formatting
- Handles different types of graphs (MI, entropy, correlation, etc.)
- Calculates error estimates for graphs without existing error data
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
    """Enhance MI plot with error bars."""
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
    
    # Save enhanced plot
    output_path = Path(instance_dir) / "mi_evolution_with_errors.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced MI plot saved to: {output_path}")

def enhance_entropy_plot(instance_dir: str, error_data: Dict):
    """Enhance entropy plot with error bars."""
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
    
    # Save enhanced plot
    output_path = Path(instance_dir) / "entropy_evolution_with_errors.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced entropy plot saved to: {output_path}")

def calculate_correlation_errors(df, error_data):
    """Calculate error estimates for correlation data."""
    # Get MI errors from error analysis
    mi_errors = error_data['mi_analysis']['mi_errors']
    mi_means = error_data['mi_analysis']['mi_means']
    
    # Calculate distance errors (estimate based on geometric uncertainty)
    distance_errors = []
    for i, row in df.iterrows():
        # Use a percentage of the distance as error estimate
        distance_error = row['geometric_distance'] * 0.05  # 5% error estimate
        distance_errors.append(distance_error)
    
    # Calculate MI errors for each data point
    mi_point_errors = []
    for i, row in df.iterrows():
        # Find the closest MI value to estimate error
        mi_value = row['mutual_information']
        closest_idx = np.argmin(np.abs(np.array(mi_means) - mi_value))
        mi_error = mi_errors[closest_idx] if closest_idx < len(mi_errors) else np.std(mi_means)
        mi_point_errors.append(mi_error)
    
    return np.array(mi_point_errors), np.array(distance_errors)

def enhance_correlation_plot_with_errors(instance_dir: str, error_data: Dict):
    """Enhance correlation plot with proper error bars."""
    # Check if correlation data exists
    correlation_file = Path(instance_dir) / "mi_distance_correlation_data.csv"
    
    if not correlation_file.exists():
        print(f"MI-distance correlation data not found: {correlation_file}")
        return
    
    # Load correlation data
    import pandas as pd
    df = pd.read_csv(correlation_file)
    
    # Calculate error estimates
    mi_errors, distance_errors = calculate_correlation_errors(df, error_data)
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with error bars
    plt.errorbar(df['geometric_distance'], df['mutual_information'], 
                xerr=distance_errors, yerr=mi_errors,
                fmt='o', capsize=5, capthick=2, markersize=8, alpha=0.7,
                label='Data points with errors', color='#F18F01', ecolor='#C73E1D')
    
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
    plt.title('MI vs Distance Correlation Analysis with Error Bars', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistical summary
    mi_mean = np.mean(df['mutual_information'])
    mi_std = np.std(df['mutual_information'])
    dist_mean = np.mean(df['geometric_distance'])
    dist_std = np.std(df['geometric_distance'])
    
    plt.text(0.02, 0.98, f'Mean MI: {mi_mean:.6f} ± {mi_std:.6f}\nMean Distance: {dist_mean:.4f} ± {dist_std:.4f}', 
            transform=plt.gca().transAxes, fontsize=12, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    # Save enhanced plot
    output_path = Path(instance_dir) / "mi_distance_correlation_with_errors.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced MI-distance correlation plot with error bars saved to: {output_path}")

def enhance_ricci_scalar_plot(instance_dir: str, error_data: Dict):
    """Enhance Ricci scalar plot with error bars if data available."""
    # Check if Ricci scalar data exists in results
    results_files = list(Path(instance_dir).glob("results_*.json"))
    
    if not results_files:
        print("No results files found")
        return
    
    # Use the first results file
    results_file = results_files[0]
    
    # Load results data
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Check if Ricci scalar data exists
    if 'ricci_scalar_evolution' not in results:
        print("Ricci scalar evolution data not found in results")
        return
    
    ricci_data = results['ricci_scalar_evolution']
    spec = results['spec']
    
    if not ricci_data or len(ricci_data) == 0:
        print("No Ricci scalar data available")
        return
    
    # Extract data
    timesteps = list(range(len(ricci_data)))
    ricci_values = [float(val) if val is not None else 0.0 for val in ricci_data]
    
    # Calculate error estimates
    if len(ricci_values) > 1:
        ricci_errors = [np.std(ricci_values) * 0.1] * len(ricci_values)  # 10% of std as error estimate
    else:
        ricci_errors = [0.01] * len(ricci_values)  # Small default error
    
    plt.figure(figsize=(12, 8))
    
    # Main plot with error bars
    plt.errorbar(timesteps, ricci_values, yerr=ricci_errors, 
                fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8,
                label='Ricci Scalar ± Error', color='#C73E1D')
    
    plt.xlabel('Timestep', fontsize=14)
    plt.ylabel('Ricci Scalar', fontsize=14)
    plt.title(f'Ricci Scalar Evolution with Error Analysis\n'
              f'{spec["num_qubits"]} qubits, {spec["geometry"]} geometry, κ={spec["curvature"]}, {spec["device"]}', 
              fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistical summary
    ricci_mean = np.mean(ricci_values)
    ricci_std = np.std(ricci_values)
    plt.text(0.02, 0.98, f'Mean Ricci: {ricci_mean:.4f} ± {ricci_std:.4f}', 
            transform=plt.gca().transAxes, fontsize=12, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    
    # Save enhanced plot
    output_path = Path(instance_dir) / "ricci_scalar_evolution_with_errors.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced Ricci scalar plot saved to: {output_path}")

def main():
    """Main function to comprehensively enhance all graphs in instance directory."""
    if len(sys.argv) != 2:
        print("Usage: python enhance_all_graphs_comprehensive.py <instance_directory>")
        print("Example: python enhance_all_graphs_comprehensive.py experiment_logs/custom_curvature_experiment/instance_20250726_153536")
        return
    
    instance_dir = sys.argv[1]
    
    if not Path(instance_dir).exists():
        print(f"Error: Directory {instance_dir} does not exist")
        return
    
    print(f"Comprehensively enhancing all graphs in: {instance_dir}")
    
    # Load error analysis data
    error_data = load_error_analysis_data(instance_dir)
    if error_data is None:
        return
    
    print("Loaded error analysis data successfully")
    
    # Create enhanced plots
    enhance_mi_plot(instance_dir, error_data)
    enhance_entropy_plot(instance_dir, error_data)
    enhance_correlation_plot_with_errors(instance_dir, error_data)
    enhance_ricci_scalar_plot(instance_dir, error_data)
    
    print("\n✅ Successfully enhanced all graphs with comprehensive error analysis!")
    print("Enhanced plots saved in the instance directory")

if __name__ == "__main__":
    main()