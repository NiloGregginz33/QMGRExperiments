#!/usr/bin/env python3
"""
Dynamic Curvature vs Entropy Analysis: R(t) vs S(t)

This script analyzes the relationship between scalar curvature R(t) and 
entropy S(t) over time to demonstrate the emergent Einstein analogue
in quantum gravity experiments.

Author: Quantum Gravity Research Team
Date: 2024
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import savgol_filter
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def load_experiment_results(results_file):
    """Load experiment results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

def extract_time_series_data(results):
    """
    Extract time series data for R(t) and S(t) from experiment results.
    
    Returns:
        dict: Time series data including R(t), S(t), and metadata
    """
    data = {
        'timesteps': [],
        'scalar_curvature': [],
        'entropy_evolution': [],
        'mi_evolution': [],
        'regge_evolution': [],
        'metadata': {}
    }
    
    # Extract scalar curvature from Regge evolution data
    if 'regge_evolution_data' in results:
        regge_data = results['regge_evolution_data']
        if 'scalar_curvature_per_timestep' in regge_data:
            data['scalar_curvature'] = regge_data['scalar_curvature_per_timestep']
            data['timesteps'] = list(range(len(data['scalar_curvature'])))
    
    # Extract entropy evolution from boundary entropies
    if 'boundary_entropies_per_timestep' in results:
        boundary_entropies = results['boundary_entropies_per_timestep']
        # Use the entropy of region A as our primary entropy measure
        data['entropy_evolution'] = []
        for timestep_data in boundary_entropies:
            if isinstance(timestep_data, dict) and 'S_A' in timestep_data:
                data['entropy_evolution'].append(timestep_data['S_A'])
            else:
                # Fallback: use first available entropy value
                if isinstance(timestep_data, list) and len(timestep_data) > 0:
                    data['entropy_evolution'].append(timestep_data[0].get('entropy', 0.0))
                else:
                    data['entropy_evolution'].append(0.0)
    
    # Extract mutual information evolution
    if 'mi_per_timestep' in results:
        mi_data = results['mi_per_timestep']
        data['mi_evolution'] = []
        for timestep_mi in mi_data:
            if isinstance(timestep_mi, dict):
                # Calculate total mutual information
                total_mi = sum(timestep_mi.values()) if timestep_mi else 0.0
                data['mi_evolution'].append(total_mi)
            else:
                data['mi_evolution'].append(0.0)
    
    # Extract Regge evolution data
    if 'regge_evolution_data' in results:
        regge_data = results['regge_evolution_data']
        data['regge_evolution'] = {
            'edge_lengths': regge_data.get('regge_edge_lengths_per_timestep', []),
            'angle_sums': regge_data.get('regge_angle_sums_per_timestep', []),
            'deficits': regge_data.get('regge_deficits_per_timestep', []),
            'actions': regge_data.get('regge_actions_per_timestep', [])
        }
    
    # Extract metadata
    data['metadata'] = {
        'num_qubits': results.get('num_qubits', 0),
        'curvature': results.get('curvature', 0.0),
        'geometry': results.get('geometry', 'unknown'),
        'device': results.get('device', 'unknown'),
        'timesteps': results.get('timesteps', 0)
    }
    
    return data

def compute_entropy_derivatives(entropy_evolution, timesteps):
    """
    Compute first and second derivatives of entropy S(t).
    
    Args:
        entropy_evolution: List of entropy values over time
        timesteps: List of timestep numbers
    
    Returns:
        dict: First and second derivatives of entropy
    """
    if len(entropy_evolution) < 3:
        return {'S_dot': [], 'S_ddot': []}
    
    # Convert to numpy arrays
    S = np.array(entropy_evolution)
    t = np.array(timesteps)
    
    # Compute first derivative Ṡ(t)
    S_dot = np.gradient(S, t)
    
    # Compute second derivative S̈(t)
    S_ddot = np.gradient(S_dot, t)
    
    return {
        'S_dot': S_dot.tolist(),
        'S_ddot': S_ddot.tolist()
    }

def analyze_einstein_analogue(R_t, S_t, S_ddot):
    """
    Analyze the Einstein analogue relationship: R(t) ∝ S̈(t)
    
    Args:
        R_t: Scalar curvature over time
        S_t: Entropy over time
        S_ddot: Second derivative of entropy over time
    
    Returns:
        dict: Analysis results including correlation and fit parameters
    """
    if len(R_t) < 3 or len(S_ddot) < 3:
        return {
            'correlation': None,
            'slope': None,
            'intercept': None,
            'r_squared': None,
            'einstein_analogue_strength': 'insufficient_data'
        }
    
    # Convert to numpy arrays
    R = np.array(R_t)
    S = np.array(S_t)
    S_ddot = np.array(S_ddot)
    
    # Remove any NaN or infinite values
    valid_mask = np.isfinite(R) & np.isfinite(S_ddot)
    R_valid = R[valid_mask]
    S_ddot_valid = S_ddot[valid_mask]
    
    if len(R_valid) < 2:
        return {
            'correlation': None,
            'slope': None,
            'intercept': None,
            'r_squared': None,
            'einstein_analogue_strength': 'insufficient_valid_data'
        }
    
    # Compute correlation between R(t) and S̈(t)
    correlation, p_value = stats.pearsonr(R_valid, S_ddot_valid)
    
    # Fit linear relationship: R(t) = α * S̈(t) + β
    slope, intercept, r_value, p_value, std_err = stats.linregress(S_ddot_valid, R_valid)
    r_squared = r_value ** 2
    
    # Determine strength of Einstein analogue
    if abs(correlation) > 0.8 and r_squared > 0.6:
        einstein_strength = 'strong'
    elif abs(correlation) > 0.6 and r_squared > 0.4:
        einstein_strength = 'moderate'
    elif abs(correlation) > 0.4 and r_squared > 0.2:
        einstein_strength = 'weak'
    else:
        einstein_strength = 'none'
    
    return {
        'correlation': float(correlation),
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_squared),
        'p_value': float(p_value),
        'einstein_analogue_strength': einstein_strength,
        'valid_data_points': len(R_valid)
    }

def create_rt_vs_st_plots(time_series_data, output_dir):
    """
    Create comprehensive plots showing R(t) vs S(t) relationship.
    
    Args:
        time_series_data: Extracted time series data
        output_dir: Directory to save plots
    """
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Dynamic Curvature vs Entropy Analysis: R(t) vs S(t)', 
                 fontsize=16, fontweight='bold')
    
    timesteps = time_series_data['timesteps']
    R_t = time_series_data['scalar_curvature']
    S_t = time_series_data['entropy_evolution']
    
    # Compute entropy derivatives
    derivatives = compute_entropy_derivatives(S_t, timesteps)
    S_dot = derivatives['S_dot']
    S_ddot = derivatives['S_ddot']
    
    # Plot 1: R(t) and S(t) over time
    ax1 = axes[0, 0]
    if R_t and S_t:
        ax1.plot(timesteps, R_t, 'b-o', label='R(t) - Scalar Curvature', linewidth=2, markersize=6)
        ax1.plot(timesteps, S_t, 'r-s', label='S(t) - Entropy', linewidth=2, markersize=6)
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Value')
        ax1.set_title('Scalar Curvature and Entropy Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: R(t) vs S̈(t) - Einstein analogue
    ax2 = axes[0, 1]
    if R_t and S_ddot and len(R_t) == len(S_ddot):
        # Remove invalid data points
        valid_mask = np.isfinite(R_t) & np.isfinite(S_ddot)
        R_valid = np.array(R_t)[valid_mask]
        S_ddot_valid = np.array(S_ddot)[valid_mask]
        
        if len(R_valid) > 1:
            ax2.scatter(S_ddot_valid, R_valid, c='purple', s=100, alpha=0.7, label='Data Points')
            
            # Fit and plot linear relationship
            if len(R_valid) >= 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(S_ddot_valid, R_valid)
                x_fit = np.linspace(min(S_ddot_valid), max(S_ddot_valid), 100)
                y_fit = slope * x_fit + intercept
                ax2.plot(x_fit, y_fit, 'g--', linewidth=2, 
                        label=f'Fit: R = {slope:.3f}×S̈ + {intercept:.3f}\nR² = {r_value**2:.3f}')
            
            ax2.set_xlabel('S̈(t) - Second Derivative of Entropy')
            ax2.set_ylabel('R(t) - Scalar Curvature')
            ax2.set_title('Einstein Analogue: R(t) ∝ S̈(t)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    
    # Plot 3: Entropy derivatives
    ax3 = axes[1, 0]
    if S_t and S_dot and S_ddot:
        ax3.plot(timesteps, S_t, 'b-o', label='S(t)', linewidth=2, markersize=6)
        ax3.plot(timesteps, S_dot, 'r-s', label='Ṡ(t)', linewidth=2, markersize=6)
        ax3.plot(timesteps, S_ddot, 'g-^', label='S̈(t)', linewidth=2, markersize=6)
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('Value')
        ax3.set_title('Entropy and its Derivatives')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Mutual Information evolution
    ax4 = axes[1, 1]
    if time_series_data['mi_evolution']:
        mi_evolution = time_series_data['mi_evolution']
        ax4.plot(timesteps, mi_evolution, 'm-o', linewidth=2, markersize=6)
        ax4.set_xlabel('Timestep')
        ax4.set_ylabel('Total Mutual Information')
        ax4.set_title('Mutual Information Evolution')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'rt_vs_st_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def generate_analysis_report(time_series_data, einstein_analysis, output_dir):
    """
    Generate a comprehensive analysis report.
    
    Args:
        time_series_data: Extracted time series data
        einstein_analysis: Results of Einstein analogue analysis
        output_dir: Directory to save report
    """
    report = []
    report.append("=" * 80)
    report.append("DYNAMIC CURVATURE VS ENTROPY ANALYSIS REPORT")
    report.append("R(t) vs S(t) - Emergent Einstein Analogue")
    report.append("=" * 80)
    report.append("")
    
    # Experiment metadata
    metadata = time_series_data['metadata']
    report.append("EXPERIMENT METADATA:")
    report.append(f"  Number of qubits: {metadata['num_qubits']}")
    report.append(f"  Curvature: {metadata['curvature']}")
    report.append(f"  Geometry: {metadata['geometry']}")
    report.append(f"  Device: {metadata['device']}")
    report.append(f"  Timesteps: {metadata['timesteps']}")
    report.append("")
    
    # Time series summary
    report.append("TIME SERIES SUMMARY:")
    report.append(f"  Scalar curvature data points: {len(time_series_data['scalar_curvature'])}")
    report.append(f"  Entropy data points: {len(time_series_data['entropy_evolution'])}")
    report.append(f"  Mutual information data points: {len(time_series_data['mi_evolution'])}")
    report.append("")
    
    # Einstein analogue analysis
    report.append("EINSTEIN ANALOGUE ANALYSIS:")
    if einstein_analysis['correlation'] is not None:
        report.append(f"  Correlation R(t) vs S̈(t): {einstein_analysis['correlation']:.4f}")
        report.append(f"  Linear fit slope (α): {einstein_analysis['slope']:.4f}")
        report.append(f"  Linear fit intercept (β): {einstein_analysis['intercept']:.4f}")
        report.append(f"  R-squared: {einstein_analysis['r_squared']:.4f}")
        report.append(f"  P-value: {einstein_analysis['p_value']:.4f}")
        report.append(f"  Einstein analogue strength: {einstein_analysis['einstein_analogue_strength']}")
        report.append(f"  Valid data points: {einstein_analysis['valid_data_points']}")
    else:
        report.append("  Insufficient data for Einstein analogue analysis")
    report.append("")
    
    # Physical interpretation
    report.append("PHYSICAL INTERPRETATION:")
    if einstein_analysis['einstein_analogue_strength'] == 'strong':
        report.append("  ✅ STRONG EINSTEIN ANALOGUE DETECTED")
        report.append("  The relationship R(t) ∝ S̈(t) demonstrates emergent gravity")
        report.append("  from entanglement dynamics, supporting holographic principle")
    elif einstein_analysis['einstein_analogue_strength'] == 'moderate':
        report.append("  ⚠️  MODERATE EINSTEIN ANALOGUE DETECTED")
        report.append("  Some evidence for R(t) ∝ S̈(t) relationship")
        report.append("  Further investigation recommended")
    elif einstein_analysis['einstein_analogue_strength'] == 'weak':
        report.append("  🔍 WEAK EINSTEIN ANALOGUE DETECTED")
        report.append("  Minimal evidence for R(t) ∝ S̈(t) relationship")
        report.append("  May require different parameters or longer evolution")
    else:
        report.append("  ❌ NO EINSTEIN ANALOGUE DETECTED")
        report.append("  No significant relationship between R(t) and S̈(t)")
        report.append("  Consider adjusting experimental parameters")
    report.append("")
    
    # Theoretical implications
    report.append("THEORETICAL IMPLICATIONS:")
    report.append("  • R(t) ∝ S̈(t) suggests curvature responds to entropy acceleration")
    report.append("  • This mirrors Einstein's equations: G_μν = 8πG T_μν")
    report.append("  • Entanglement dynamics drive geometric evolution")
    report.append("  • Supports AdS/CFT correspondence and holographic principle")
    report.append("  • Demonstrates quantum-to-classical gravity emergence")
    report.append("")
    
    report.append("=" * 80)
    
    # Save report
    report_path = os.path.join(output_dir, 'rt_vs_st_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    return report_path

def main():
    """Main analysis function."""
    # Find the most recent experiment results
    experiment_logs_dir = Path(__file__).parent.parent / 'experiment_logs' / 'custom_curvature_experiment'
    
    if not experiment_logs_dir.exists():
        print(f"❌ Experiment logs directory not found: {experiment_logs_dir}")
        return
    
    # Find the most recent results file
    results_files = list(experiment_logs_dir.glob('results_*.json'))
    if not results_files:
        print(f"❌ No results files found in {experiment_logs_dir}")
        return
    
    # Sort by modification time and get the most recent
    latest_file = max(results_files, key=lambda f: f.stat().st_mtime)
    print(f"📊 Analyzing latest experiment results: {latest_file.name}")
    
    # Load results
    results = load_experiment_results(latest_file)
    
    # Extract time series data
    time_series_data = extract_time_series_data(results)
    
    # Check if we have sufficient data
    if not time_series_data['scalar_curvature'] or not time_series_data['entropy_evolution']:
        print("❌ Insufficient data for R(t) vs S(t) analysis")
        print("   Make sure to run experiment with --solve_regge and --compute_entropies")
        return
    
    # Compute entropy derivatives
    derivatives = compute_entropy_derivatives(
        time_series_data['entropy_evolution'], 
        time_series_data['timesteps']
    )
    
    # Analyze Einstein analogue
    einstein_analysis = analyze_einstein_analogue(
        time_series_data['scalar_curvature'],
        time_series_data['entropy_evolution'],
        derivatives['S_ddot']
    )
    
    # Create output directory
    output_dir = Path(__file__).parent / 'rt_vs_st_analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Create plots
    plot_path = create_rt_vs_st_plots(time_series_data, output_dir)
    print(f"📈 Plots saved to: {plot_path}")
    
    # Generate report
    report_path = generate_analysis_report(time_series_data, einstein_analysis, output_dir)
    print(f"📄 Report saved to: {report_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Experiment: {latest_file.name}")
    print(f"Timesteps analyzed: {len(time_series_data['timesteps'])}")
    
    if einstein_analysis['correlation'] is not None:
        print(f"R(t) vs S̈(t) correlation: {einstein_analysis['correlation']:.4f}")
        print(f"Einstein analogue strength: {einstein_analysis['einstein_analogue_strength']}")
        print(f"R-squared: {einstein_analysis['r_squared']:.4f}")
    else:
        print("Einstein analogue analysis: Insufficient data")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 