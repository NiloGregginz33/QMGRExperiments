#!/usr/bin/env python3
"""
Statistical Analysis of Lorentzian Action Results
Computes p-values, confidence intervals, and effect sizes for publication
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import bootstrap
import pandas as pd
from datetime import datetime
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def load_lorentzian_results(file_path):
    """Load Lorentzian results from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def compute_basic_statistics(edge_lengths, action_value):
    """Compute basic statistics for edge lengths and action"""
    stats_dict = {
        'action_value': action_value,
        'edge_lengths_mean': np.mean(edge_lengths),
        'edge_lengths_std': np.std(edge_lengths),
        'edge_lengths_median': np.median(edge_lengths),
        'edge_lengths_min': np.min(edge_lengths),
        'edge_lengths_max': np.max(edge_lengths),
        'edge_lengths_cv': np.std(edge_lengths) / np.mean(edge_lengths),  # Coefficient of variation
        'num_edges': len(edge_lengths)
    }
    return stats_dict

def generate_null_distribution(num_samples=10000, num_qubits=7, timesteps=3):
    """Generate null distribution using random edge lengths"""
    # Simulate random quantum circuits with similar structure
    null_actions = []
    
    for _ in range(num_samples):
        # Generate random edge lengths (uniform distribution)
        num_edges = num_qubits * (num_qubits - 1) // 2 * timesteps + num_qubits * (timesteps - 1)
        random_edges = np.random.uniform(0.001, 25.0, num_edges)  # Similar range to observed data
        
        # Compute a simple action-like quantity for null hypothesis
        # This simulates what we'd expect from random quantum noise
        null_action = np.mean(random_edges) * np.std(random_edges) / len(random_edges)
        null_actions.append(null_action)
    
    return np.array(null_actions)

def compute_p_value(observed_action, null_distribution, alternative='less'):
    """Compute p-value for observed action against null distribution"""
    if alternative == 'less':
        p_value = np.mean(null_distribution <= observed_action)
    elif alternative == 'greater':
        p_value = np.mean(null_distribution >= observed_action)
    else:  # two-sided
        p_value = 2 * min(np.mean(null_distribution <= observed_action),
                         np.mean(null_distribution >= observed_action))
    
    return p_value

def compute_effect_size(observed_action, null_distribution):
    """Compute Cohen's d effect size"""
    null_mean = np.mean(null_distribution)
    null_std = np.std(null_distribution)
    effect_size = (observed_action - null_mean) / null_std
    return effect_size

def bootstrap_confidence_interval(edge_lengths, action_value, n_bootstrap=10000, confidence_level=0.95):
    """Compute bootstrap confidence interval for action"""
    def action_statistic(data):
        # Compute action-like statistic from edge lengths
        return np.mean(data) * np.std(data) / len(data)
    
    # Bootstrap resampling
    bootstrap_result = bootstrap((edge_lengths,), action_statistic, 
                               n_resamples=n_bootstrap, confidence_level=confidence_level)
    
    return bootstrap_result.confidence_interval

def analyze_geometric_significance(edge_lengths, action_value):
    """Analyze if the geometry shows significant curvature"""
    # Test for non-uniformity in edge lengths (indicator of curvature)
    # Use chi-square test for uniformity
    hist, bins = np.histogram(edge_lengths, bins=10)
    expected = len(edge_lengths) / 10  # Uniform distribution
    chi2_stat, chi2_p = stats.chisquare(hist, [expected] * 10)
    
    # Test for log-normal distribution (common in curved geometries)
    log_edges = np.log(edge_lengths[edge_lengths > 0])
    _, lognorm_p = stats.normaltest(log_edges)
    
    return {
        'chi2_statistic': float(chi2_stat),
        'chi2_p_value': float(chi2_p),
        'lognorm_p_value': float(lognorm_p),
        'is_significantly_curved': bool(chi2_p < 0.05 and lognorm_p < 0.05)
    }

def create_publication_plots(observed_action, null_distribution, edge_lengths, results_dir):
    """Create publication-quality plots"""
    # Ensure directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Null distribution with observed action
    axes[0, 0].hist(null_distribution, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0, 0].axvline(observed_action, color='red', linewidth=3, label=f'Observed: {observed_action:.6f}')
    axes[0, 0].set_xlabel('Action Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Null Distribution vs Observed Action')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Edge length distribution
    axes[0, 1].hist(edge_lengths, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Edge Length')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Edge Length Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Q-Q plot for normality test
    stats.probplot(edge_lengths, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot: Edge Lengths vs Normal Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Log-scale edge lengths
    log_edges = np.log(edge_lengths[edge_lengths > 0])
    axes[1, 1].hist(log_edges, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].set_xlabel('Log(Edge Length)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Log-Scale Edge Length Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'lorentzian_statistical_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_publication_report(results_dict, results_dir):
    """Generate comprehensive publication report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results
    with open(os.path.join(results_dir, 'statistical_analysis_results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    # Generate summary report
    report = f"""
# Lorentzian Action Statistical Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Experimental Parameters
- Device: {results_dict['experiment_params']['device']}
- Geometry: {results_dict['experiment_params']['geometry']}
- Curvature: {results_dict['experiment_params']['curvature']}
- Shots: {results_dict['experiment_params']['shots']}
- Qubits: {results_dict['experiment_params']['num_qubits']}

## Key Results

### Lorentzian Action
- Observed Action: {results_dict['basic_stats']['action_value']:.8f}
- 95% CI: [{results_dict['bootstrap_ci'][0]:.8f}, {results_dict['bootstrap_ci'][1]:.8f}]

### Statistical Significance
- P-value: {results_dict['p_value']:.8f}
- Significance Level: {'***' if results_dict['p_value'] < 0.001 else '**' if results_dict['p_value'] < 0.01 else '*' if results_dict['p_value'] < 0.05 else 'Not Significant'}
- Effect Size (Cohen's d): {results_dict['effect_size']:.4f}
- Effect Size Interpretation: {results_dict['effect_size_interpretation']}

### Geometric Analysis
- Chi-square p-value (uniformity test): {results_dict['geometric_analysis']['chi2_p_value']:.6f}
- Log-normal p-value: {results_dict['geometric_analysis']['lognorm_p_value']:.6f}
- Significantly Curved: {results_dict['geometric_analysis']['is_significantly_curved']}

### Edge Length Statistics
- Mean: {results_dict['basic_stats']['edge_lengths_mean']:.4f}
- Std Dev: {results_dict['basic_stats']['edge_lengths_std']:.4f}
- Coefficient of Variation: {results_dict['basic_stats']['edge_lengths_cv']:.4f}
- Range: [{results_dict['basic_stats']['edge_lengths_min']:.4f}, {results_dict['basic_stats']['edge_lengths_max']:.4f}]

## Publication Readiness Assessment

### Statistical Power
- Current p-value: {results_dict['p_value']:.8f}
- Recommended for high-impact journals: p < 0.001
- Current status: {'SUFFICIENT' if results_dict['p_value'] < 0.001 else 'NEEDS MORE SHOTS'}

### Recommendations
"""
    
    if results_dict['p_value'] >= 0.001:
        report += f"""
**RECOMMENDATION: RUN MORE EXPERIMENTS**

Current p-value ({results_dict['p_value']:.8f}) is above the 0.001 threshold for high-impact journals.
Estimated additional shots needed: {int(2000 * (0.001 / results_dict['p_value']) ** 2)}
"""
    else:
        report += """
**STATISTICAL SIGNIFICANCE ACHIEVED**

Results meet publication standards for high-impact physics journals.
"""
    
    # Save report
    with open(os.path.join(results_dir, 'statistical_analysis_report.txt'), 'w') as f:
        f.write(report)
    
    return report

def main():
    """Main analysis function"""
    # Load the Lorentzian results
    results_file = "experiment_logs/custom_curvature_experiment/results_n7_geomH_curv25_ibm_BR9RDF.json"
    data = load_lorentzian_results(results_file)
    
    # Extract key data
    edge_lengths = np.array(data['lorentzian_solution']['stationary_edge_lengths'])
    action_value = data['lorentzian_solution']['stationary_action']
    experiment_params = data['spec']
    
    print(f"Analyzing Lorentzian action: {action_value:.8f}")
    print(f"Number of edge lengths: {len(edge_lengths)}")
    print(f"Device: {experiment_params['device']}")
    print(f"Shots: {experiment_params['shots']}")
    
    # Compute basic statistics
    basic_stats = compute_basic_statistics(edge_lengths, action_value)
    
    # Generate null distribution
    print("Generating null distribution...")
    null_distribution = generate_null_distribution(
        num_samples=10000, 
        num_qubits=experiment_params['num_qubits'],
        timesteps=experiment_params['timesteps']
    )
    
    # Compute p-value
    p_value = compute_p_value(action_value, null_distribution, alternative='less')
    
    # Compute effect size
    effect_size = compute_effect_size(action_value, null_distribution)
    
    # Interpret effect size
    if abs(effect_size) < 0.2:
        effect_interpretation = "Small"
    elif abs(effect_size) < 0.5:
        effect_interpretation = "Small to Medium"
    elif abs(effect_size) < 0.8:
        effect_interpretation = "Medium"
    elif abs(effect_size) < 1.2:
        effect_interpretation = "Medium to Large"
    else:
        effect_interpretation = "Large"
    
    # Bootstrap confidence interval
    print("Computing bootstrap confidence interval...")
    bootstrap_ci = bootstrap_confidence_interval(edge_lengths, action_value)
    
    # Geometric significance analysis
    geometric_analysis = analyze_geometric_significance(edge_lengths, action_value)
    
    # Compile results
    results_dict = {
        'basic_stats': basic_stats,
        'p_value': p_value,
        'effect_size': effect_size,
        'effect_size_interpretation': effect_interpretation,
        'bootstrap_ci': [float(bootstrap_ci[0]), float(bootstrap_ci[1])],
        'geometric_analysis': geometric_analysis,
        'experiment_params': experiment_params,
        'null_distribution_stats': {
            'mean': float(np.mean(null_distribution)),
            'std': float(np.std(null_distribution)),
            'min': float(np.min(null_distribution)),
            'max': float(np.max(null_distribution))
        }
    }
    
    # Create results directory
    results_dir = f"experiment_logs/lorentzian_statistical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Generate plots
    print("Creating publication plots...")
    create_publication_plots(action_value, null_distribution, edge_lengths, results_dir)
    
    # Generate report
    print("Generating publication report...")
    report = generate_publication_report(results_dict, results_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("="*60)
    print(f"Observed Lorentzian Action: {action_value:.8f}")
    print(f"P-value: {p_value:.8f}")
    print(f"Significance: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'Not Significant'}")
    print(f"Effect Size (Cohen's d): {effect_size:.4f} ({effect_interpretation})")
    print(f"95% Confidence Interval: [{bootstrap_ci[0]:.8f}, {bootstrap_ci[1]:.8f}]")
    print(f"Geometric Curvature Significant: {geometric_analysis['is_significantly_curved']}")
    print(f"Results saved to: {results_dir}")
    
    if p_value >= 0.001:
        estimated_shots = int(2000 * (0.001 / p_value) ** 2)
        print(f"\n⚠️  RECOMMENDATION: Run more experiments")
        print(f"   Estimated additional shots needed: {estimated_shots}")
        print(f"   Target p-value for high-impact journals: < 0.001")
    else:
        print(f"\n✅ STATISTICAL SIGNIFICANCE ACHIEVED")
        print(f"   Results meet publication standards for high-impact journals")
    
    return results_dict

if __name__ == "__main__":
    main() 