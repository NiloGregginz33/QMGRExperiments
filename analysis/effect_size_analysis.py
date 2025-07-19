#!/usr/bin/env python3
"""
Effect Size Analysis: Is the low p-value due to large sample size or real effect?
Addresses the concern about statistical vs practical significance.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

def analyze_effect_size_and_sample_size(results_file):
    """Analyze whether the low p-value is due to large sample size or real effect."""
    
    print("=" * 80)
    print("EFFECT SIZE AND SAMPLE SIZE ANALYSIS")
    print("=" * 80)
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Extract data
    angle_sums = np.array(data['angle_sums'])
    mean_angle_sum = np.mean(angle_sums)
    std_angle_sum = np.std(angle_sums)
    pi = np.pi
    
    print(f"\nOBSERVED DATA:")
    print(f"  Mean angle sum: {mean_angle_sum:.6f} radians")
    print(f"  Expected (flat): π ≈ {pi:.6f} radians")
    print(f"  Standard deviation: {std_angle_sum:.6f} radians")
    print(f"  Sample size: {len(angle_sums)} triangles")
    
    # Calculate effect size measures
    deviation = pi - mean_angle_sum
    cohens_d = deviation / std_angle_sum
    percent_deviation = (deviation / pi) * 100
    
    print(f"\nEFFECT SIZE MEASURES:")
    print(f"  Absolute deviation: {deviation:.6f} radians")
    print(f"  Percent deviation: {percent_deviation:.2f}%")
    print(f"  Cohen's d: {cohens_d:.2f}")
    
    # Interpret Cohen's d
    if cohens_d >= 0.8:
        effect_size_interpretation = "LARGE"
    elif cohens_d >= 0.5:
        effect_size_interpretation = "MEDIUM"
    elif cohens_d >= 0.2:
        effect_size_interpretation = "SMALL"
    else:
        effect_size_interpretation = "NEGLIGIBLE"
    
    print(f"  Effect size interpretation: {effect_size_interpretation}")
    
    # Calculate eta-squared (proportion of variance explained)
    ss_total = np.sum((angle_sums - pi) ** 2)
    ss_between = len(angle_sums) * (mean_angle_sum - pi) ** 2
    ss_within = np.sum((angle_sums - mean_angle_sum) ** 2)
    eta_squared = ss_between / ss_total
    
    print(f"  Eta-squared: {eta_squared:.4f} ({eta_squared*100:.2f}% of variance explained)")
    
    # Simulate p-values with smaller sample sizes
    print(f"\nSAMPLE SIZE SIMULATION:")
    print(f"  Original sample size: {len(angle_sums)} triangles")
    print(f"  Original p-value: {stats.ttest_1samp(angle_sums, pi)[1]:.10f}")
    
    # Simulate with smaller samples
    sample_sizes = [10, 20, 35, 50, 100, 200, 500, 1000]
    simulated_p_values = []
    
    np.random.seed(42)  # For reproducibility
    
    for n in sample_sizes:
        # Bootstrap sampling to simulate smaller sample sizes
        p_values = []
        for _ in range(100):  # 100 simulations per sample size
            sample = np.random.choice(angle_sums, size=min(n, len(angle_sums)), replace=True)
            _, p_val = stats.ttest_1samp(sample, pi)
            p_values.append(p_val)
        
        mean_p = np.mean(p_values)
        simulated_p_values.append(mean_p)
        
        significance = "***" if mean_p < 0.001 else "**" if mean_p < 0.01 else "*" if mean_p < 0.05 else ""
        print(f"  Sample size {n:4d}: p = {mean_p:.8f} {significance}")
    
    # Calculate minimum sample size needed for significance
    print(f"\nMINIMUM SAMPLE SIZE ANALYSIS:")
    
    # Use power analysis to find minimum sample size
    from scipy.stats import norm
    
    # For 80% power, alpha = 0.05
    alpha = 0.05
    power = 0.80
    
    # Z-scores
    z_alpha = norm.ppf(1 - alpha/2)  # Two-tailed
    z_beta = norm.ppf(power)
    
    # Minimum sample size formula for one-sample t-test
    min_n = ((z_alpha + z_beta) ** 2) / (cohens_d ** 2)
    
    print(f"  Minimum sample size for 80% power: {min_n:.1f} triangles")
    print(f"  This means the effect would be detectable with just {int(min_n)} triangles!")
    
    # Practical significance assessment
    print(f"\nPRACTICAL SIGNIFICANCE ASSESSMENT:")
    
    if cohens_d > 2.0:
        print(f"  ✅ PRACTICALLY SIGNIFICANT")
        print(f"     - Cohen's d = {cohens_d:.2f} is extremely large")
        print(f"     - {percent_deviation:.1f}% deviation from expected value")
        print(f"     - Effect would be obvious even with small samples")
    elif cohens_d > 0.8:
        print(f"  ⚠️  LIKELY PRACTICALLY SIGNIFICANT")
        print(f"     - Cohen's d = {cohens_d:.2f} is large")
        print(f"     - {percent_deviation:.1f}% deviation from expected value")
    else:
        print(f"  ❌ MAY NOT BE PRACTICALLY SIGNIFICANT")
        print(f"     - Cohen's d = {cohens_d:.2f} is small")
        print(f"     - Statistical significance may be due to large sample size")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Effect size visualization
    plt.subplot(2, 3, 1)
    x = np.linspace(0, 4, 1000)
    y1 = stats.norm.pdf(x, pi, std_angle_sum)  # Expected distribution
    y2 = stats.norm.pdf(x, mean_angle_sum, std_angle_sum)  # Observed distribution
    
    plt.plot(x, y1, 'r-', linewidth=2, label=f'Expected (π = {pi:.2f})')
    plt.plot(x, y2, 'b-', linewidth=2, label=f'Observed (μ = {mean_angle_sum:.2f})')
    plt.fill_between(x, y1, alpha=0.3, color='red')
    plt.fill_between(x, y2, alpha=0.3, color='blue')
    plt.xlabel('Angle Sum (radians)')
    plt.ylabel('Probability Density')
    plt.title(f'Effect Size: Cohen\'s d = {cohens_d:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Sample size vs p-value
    plt.subplot(2, 3, 2)
    plt.semilogy(sample_sizes, simulated_p_values, 'bo-', linewidth=2, markersize=8)
    plt.axhline(0.05, color='red', linestyle='--', label='α = 0.05')
    plt.axhline(0.01, color='orange', linestyle='--', label='α = 0.01')
    plt.axhline(0.001, color='green', linestyle='--', label='α = 0.001')
    plt.xlabel('Sample Size')
    plt.ylabel('P-value (log scale)')
    plt.title('P-value vs Sample Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Effect size comparison
    plt.subplot(2, 3, 3)
    effect_sizes = [0.2, 0.5, 0.8, 2.0, cohens_d]
    labels = ['Small', 'Medium', 'Large', 'Very Large', 'Observed']
    colors = ['lightblue', 'lightgreen', 'orange', 'red', 'purple']
    
    bars = plt.bar(labels, effect_sizes, color=colors, alpha=0.7, edgecolor='black')
    plt.axhline(0.8, color='black', linestyle='--', alpha=0.5, label='Large effect threshold')
    plt.ylabel("Cohen's d")
    plt.title('Effect Size Comparison')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, effect_sizes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Variance explained
    plt.subplot(2, 3, 4)
    labels = ['Explained by Effect', 'Unexplained Variance']
    sizes = [eta_squared, 1 - eta_squared]
    colors = ['lightcoral', 'lightblue']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title(f'Variance Explained\n(η² = {eta_squared:.3f})')
    
    # Plot 5: Sample size power analysis
    plt.subplot(2, 3, 5)
    sample_sizes_power = np.arange(5, 100, 5)
    powers = []
    
    for n in sample_sizes_power:
        # Calculate power for this sample size
        z_critical = norm.ppf(1 - alpha/2)
        z_effect = cohens_d * np.sqrt(n)
        power_val = 1 - norm.cdf(z_critical - z_effect) + norm.cdf(-z_critical - z_effect)
        powers.append(power_val)
    
    plt.plot(sample_sizes_power, powers, 'g-', linewidth=2)
    plt.axhline(0.8, color='red', linestyle='--', label='80% power')
    plt.axhline(0.95, color='orange', linestyle='--', label='95% power')
    plt.xlabel('Sample Size')
    plt.ylabel('Statistical Power')
    plt.title('Power Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Practical vs statistical significance
    plt.subplot(2, 3, 6)
    x = np.linspace(0, 5, 100)
    y_effect = 1 / (1 + np.exp(-(x - 2)))  # Sigmoid for practical significance
    y_stat = 1 / (1 + np.exp(-(x - 0.5)))  # Sigmoid for statistical significance
    
    plt.plot(x, y_stat, 'b-', linewidth=2, label='Statistical Significance')
    plt.plot(x, y_effect, 'r-', linewidth=2, label='Practical Significance')
    plt.axvline(cohens_d, color='purple', linestyle='--', linewidth=2, 
                label=f'Observed d = {cohens_d:.1f}')
    plt.xlabel("Cohen's d")
    plt.ylabel('Probability')
    plt.title('Significance Types')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"effect_size_analysis.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {plot_filename}")
    
    # Final conclusion
    print(f"\n" + "=" * 80)
    print("FINAL ASSESSMENT")
    print("=" * 80)
    
    if cohens_d > 2.0:
        print(f"✅ THE LOW P-VALUE IS JUSTIFIED")
        print(f"   - Cohen's d = {cohens_d:.2f} is extremely large")
        print(f"   - {percent_deviation:.1f}% deviation is practically significant")
        print(f"   - Effect would be detectable with just {int(min_n)} triangles")
        print(f"   - This is a REAL effect, not a statistical artifact")
    elif cohens_d > 0.8:
        print(f"⚠️  THE LOW P-VALUE IS LIKELY JUSTIFIED")
        print(f"   - Cohen's d = {cohens_d:.2f} is large")
        print(f"   - Effect size suggests practical significance")
        print(f"   - But consider the context carefully")
    else:
        print(f"❌ THE LOW P-VALUE MAY BE DUE TO LARGE SAMPLE SIZE")
        print(f"   - Cohen's d = {cohens_d:.2f} is small")
        print(f"   - Statistical significance without practical significance")
        print(f"   - Large sample size may be creating false significance")
    
    return {
        'cohens_d': cohens_d,
        'eta_squared': eta_squared,
        'percent_deviation': percent_deviation,
        'min_sample_size': min_n,
        'practically_significant': cohens_d > 0.8
    }

def main():
    """Main function to run effect size analysis."""
    results_file = "experiment_logs/custom_curvature_experiment/results_n7_geomH_curv05_ibm_60JXA8.json"
    
    if not Path(results_file).exists():
        print(f"File not found: {results_file}")
        return
    
    results = analyze_effect_size_and_sample_size(results_file)
    
    print(f"\nSUMMARY:")
    print(f"  Cohen's d: {results['cohens_d']:.2f}")
    print(f"  Eta-squared: {results['eta_squared']:.4f}")
    print(f"  Percent deviation: {results['percent_deviation']:.1f}%")
    print(f"  Minimum sample size: {results['min_sample_size']:.1f}")
    print(f"  Practically significant: {'YES' if results['practically_significant'] else 'NO'}")

if __name__ == "__main__":
    main() 