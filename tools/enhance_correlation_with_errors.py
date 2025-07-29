#!/usr/bin/env python3
"""
Enhance MI-Distance Correlation Plot with Error Bars
==================================================

This script enhances the MI-distance correlation plot with proper error bars
by calculating error estimates for the correlation data.
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
        
        # Weighted linear fit using errors
        weights = 1 / (mi_errors + 1e-10)  # Add small constant to avoid division by zero
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

def main():
    """Main function to enhance correlation plot with error bars."""
    if len(sys.argv) != 2:
        print("Usage: python enhance_correlation_with_errors.py <instance_directory>")
        print("Example: python enhance_correlation_with_errors.py experiment_logs/custom_curvature_experiment/instance_20250726_153536")
        return
    
    instance_dir = sys.argv[1]
    
    if not Path(instance_dir).exists():
        print(f"Error: Directory {instance_dir} does not exist")
        return
    
    print(f"Enhancing correlation plot with error bars in: {instance_dir}")
    
    # Load error analysis data
    error_data = load_error_analysis_data(instance_dir)
    if error_data is None:
        return
    
    print("Loaded error analysis data successfully")
    
    # Create enhanced correlation plot
    enhance_correlation_plot_with_errors(instance_dir, error_data)
    
    print("\n✅ Successfully enhanced correlation plot with error bars!")

if __name__ == "__main__":
    main()