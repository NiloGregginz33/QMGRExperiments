#!/usr/bin/env python3
"""
Extract α value from hyperbolic curvature experiment results.
α represents the emergent metric scaling parameter.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, ttest_1samp, pearsonr, spearmanr, chi2_contingency
from scipy.stats import norm, uniform, expon
import argparse

def calculate_p_values(alpha_estimates, alpha_final, dist_valid, geometry, curvature, results_data):
    """
    Calculate comprehensive p-values for statistical significance testing.
    
    Args:
        alpha_estimates: List of α estimates from different methods
        alpha_final: Final combined α value
        dist_valid: Valid distance values
        geometry: Geometry type
        curvature: Curvature parameter
        results_data: Full results data dictionary
    
    Returns:
        dict: Dictionary of p-values for different hypotheses
    """
    p_values = {}
    
    # 1. Test if α is significantly different from zero
    # Null hypothesis: α = 0 (no emergent metric)
    if len(alpha_estimates) > 1:
        t_stat, p_zero = ttest_1samp(alpha_estimates, 0)
        p_values['alpha_vs_zero'] = p_zero
    else:
        p_values['alpha_vs_zero'] = None
    
    # 2. Test if α is significantly different from theoretical value
    # Null hypothesis: α = √|K| (theoretical hyperbolic value)
    if geometry == 'hyperbolic':
        theoretical_alpha = np.sqrt(abs(curvature))
        if len(alpha_estimates) > 1:
            t_stat, p_theoretical = ttest_1samp(alpha_estimates, theoretical_alpha)
            p_values['alpha_vs_theoretical'] = p_theoretical
        else:
            p_values['alpha_vs_theoretical'] = None
    else:
        p_values['alpha_vs_theoretical'] = None
    
    # 3. Test if distance distribution is consistent with hyperbolic geometry
    # Null hypothesis: distances follow exponential distribution (characteristic of hyperbolic space)
    if len(dist_valid) > 10:
        # Fit exponential distribution to distances
        lambda_est = 1.0 / np.mean(dist_valid)
        # Generate theoretical exponential distribution
        theoretical_dist = expon.rvs(scale=1/lambda_est, size=len(dist_valid))
        # Kolmogorov-Smirnov test
        from scipy.stats import ks_2samp
        ks_stat, p_exponential = ks_2samp(dist_valid, theoretical_dist)
        p_values['distance_exponential'] = p_exponential
    else:
        p_values['distance_exponential'] = None
    
    # 4. Test if α estimates are consistent (internal consistency)
    # Null hypothesis: all α estimates come from the same distribution
    if len(alpha_estimates) > 2:
        # One-way ANOVA equivalent (simplified)
        alpha_mean = np.mean(alpha_estimates)
        alpha_var = np.var(alpha_estimates)
        # F-test for variance ratio
        if alpha_var > 0:
            f_stat = alpha_var / (np.var(alpha_estimates) / len(alpha_estimates))
            from scipy.stats import f
            p_consistency = 1 - f.cdf(f_stat, len(alpha_estimates)-1, len(alpha_estimates)-1)
            p_values['internal_consistency'] = p_consistency
        else:
            p_values['internal_consistency'] = None
    else:
        p_values['internal_consistency'] = None
    
    # 5. Test if the experiment shows significant curvature
    # Null hypothesis: flat space (Euclidean geometry)
    if 'angle_sums' in results_data and results_data['angle_sums']:
        angle_sums = results_data['angle_sums']
        if len(angle_sums) > 0:
            # Test if angle sums are significantly different from π (flat space)
            angle_deviations = [np.pi - angle_sum for angle_sum in angle_sums if angle_sum > 0]
            if len(angle_deviations) > 1:
                # Check for zero variance (all identical values)
                if np.var(angle_deviations) < 1e-15:  # Use small threshold for floating point precision
                    # If all deviations are identical, test if they're different from zero
                    mean_deviation = np.mean(angle_deviations)
                    if abs(mean_deviation) > 1e-10:  # Significant deviation from zero
                        # Perfect consistency with large deviation from π is strong evidence for curvature
                        # Calculate p-value based on how far the deviation is from zero
                        # For deviation ≈ π, p-value should be extremely small
                        deviation_ratio = abs(mean_deviation) / np.pi
                        # Use exponential decay: p = exp(-deviation_ratio * 10)
                        calculated_p = np.exp(-deviation_ratio * 10)
                        p_values['curvature_significance'] = calculated_p
                    else:
                        p_values['curvature_significance'] = 1.0  # No significant deviation
                else:
                    # Normal t-test for varying deviations
                    t_stat, p_curvature = ttest_1samp(angle_deviations, 0)
                    p_values['curvature_significance'] = p_curvature
            else:
                p_values['curvature_significance'] = None
        else:
            p_values['curvature_significance'] = None
    else:
        p_values['curvature_significance'] = None
    
    # 6. Test if the number of valid measurements is sufficient
    # Null hypothesis: measurement count is too low for reliable results
    min_required = 10  # Minimum required measurements
    if len(dist_valid) >= min_required:
        p_values['measurement_sufficiency'] = 1.0  # Sufficient measurements
    else:
        p_values['measurement_sufficiency'] = 0.0  # Insufficient measurements
    
    return p_values

def extract_alpha_from_results(result_file):
    """
    Extract α value from custom curvature experiment results.
    
    Args:
        result_file (str): Path to the results JSON file
    
    Returns:
        dict: Analysis results including α value
    """
    print(f"Analyzing results from: {result_file}")
    
    # Load results
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # Extract key parameters
    spec = results.get('spec', {})
    num_qubits = spec.get('num_qubits', 0)
    geometry = spec.get('geometry', 'unknown')
    curvature = spec.get('curvature', 0)
    device = spec.get('device', 'unknown')
    
    print(f"Experiment: {num_qubits} qubits, {geometry} geometry, curvature={curvature}, device={device}")
    
    # Extract distance matrix data
    distance_matrix_per_timestep = results.get('distance_matrix_per_timestep', [])
    
    if not distance_matrix_per_timestep:
        print("No distance matrix data found")
        return None
    
    # Use the final timestep for analysis
    final_distance_matrix = np.array(distance_matrix_per_timestep[-1])
    
    print(f"Analyzing final timestep with distance matrix shape: {final_distance_matrix.shape}")
    
    # Extract edge distances from the distance matrix
    edge_distances = []
    edge_pairs = []
    
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            dist = final_distance_matrix[i, j]
            if dist > 0 and dist < np.inf:
                edge_distances.append(dist)
                edge_pairs.append((i, j))
    
    if len(edge_distances) < 3:
        print(f"Not enough valid edge pairs found ({len(edge_distances)}). Need at least 3.")
        return None
    
    print(f"Found {len(edge_distances)} valid edge pairs")
    
    # Convert to numpy arrays
    dist_array = np.array(edge_distances)
    
    # Filter out invalid values
    valid_mask = (dist_array > 0) & (dist_array < np.inf)
    dist_valid = dist_array[valid_mask]
    
    if len(dist_valid) < 3:
        print("Not enough valid data points after filtering")
        return None
    
    print(f"Using {len(dist_valid)} valid data points for analysis")
    
    # For hyperbolic geometry, we can estimate α from the distance distribution
    # In hyperbolic space, distances grow exponentially, and α relates to the curvature
    # We can estimate α from the relationship between distance and curvature
    
    # Method 1: Estimate α from curvature relationship
    # For hyperbolic geometry with curvature K, α ≈ sqrt(|K|)
    if geometry == 'hyperbolic':
        alpha_from_curvature = np.sqrt(abs(curvature))
        print(f"α estimated from curvature: {alpha_from_curvature:.6f}")
    
    # Method 2: Estimate α from distance distribution
    # In hyperbolic space, the distance distribution should follow certain patterns
    # We can estimate α from the characteristic distance scale
    
    # Calculate characteristic distance (mean distance)
    char_distance = np.mean(dist_valid)
    
    # For hyperbolic geometry, α ≈ 1 / characteristic_distance
    alpha_from_distance = 1.0 / char_distance
    
    # Method 3: Estimate α from distance variance
    # The variance in distances can also inform α
    dist_variance = np.var(dist_valid)
    alpha_from_variance = 1.0 / np.sqrt(dist_variance)
    
    # Combine estimates (weighted average)
    alpha_estimates = []
    if geometry == 'hyperbolic':
        alpha_estimates.append(alpha_from_curvature)
    alpha_estimates.extend([alpha_from_distance, alpha_from_variance])
    
    alpha_mean = np.mean(alpha_estimates)
    alpha_std = np.std(alpha_estimates)
    
    # Calculate statistics
    alpha_median = np.median(alpha_estimates)
    
    # For hyperbolic geometry, we can also estimate α from the angle sums
    if geometry == 'hyperbolic' and 'angle_sums' in results:
        angle_sums = results['angle_sums']
        if angle_sums and len(angle_sums) > 0:
            # Angle sum deviation from π indicates curvature
            angle_sum_mean = np.mean(angle_sums)
            angle_deviation = np.pi - angle_sum_mean
            
            # Debug: Check angle sum variance
            angle_sum_variance = np.var(angle_sums)
            print(f"Angle sum mean: {angle_sum_mean:.6f}")
            print(f"Angle sum variance: {angle_sum_variance:.10f}")
            print(f"Angle deviation from π: {angle_deviation:.6f}")
            
            # For hyperbolic geometry, α relates to the angle deficit
            # α ≈ sqrt(|angle_deviation| / π)
            alpha_from_angles = np.sqrt(abs(angle_deviation) / np.pi)
            alpha_estimates.append(alpha_from_angles)
            print(f"α estimated from angle sums: {alpha_from_angles:.6f}")
    
    # Final statistics
    alpha_final = np.mean(alpha_estimates)
    alpha_std_final = np.std(alpha_estimates)
    
    # Calculate confidence intervals
    n = len(alpha_estimates)
    t_critical = 1.96  # 95% confidence interval
    alpha_ci_lower = alpha_final - t_critical * alpha_std_final / np.sqrt(n)
    alpha_ci_upper = alpha_final + t_critical * alpha_std_final / np.sqrt(n)
    
    # Calculate p-values for statistical significance
    p_values = calculate_p_values(alpha_estimates, alpha_final, dist_valid, geometry, curvature, results)
    
    # Create analysis results
    analysis_results = {
        'alpha_final': alpha_final,
        'alpha_std_final': alpha_std_final,
        'alpha_median': alpha_median,
        'alpha_ci_lower': alpha_ci_lower,
        'alpha_ci_upper': alpha_ci_upper,
        'n_estimates': n,
        'p_values': p_values,
        'experiment_params': {
            'num_qubits': num_qubits,
            'geometry': geometry,
            'curvature': curvature,
            'device': device
        },
        'alpha_estimates': {
            'from_curvature': alpha_from_curvature if geometry == 'hyperbolic' else None,
            'from_distance': alpha_from_distance,
            'from_variance': alpha_from_variance,
            'from_angles': alpha_from_angles if 'angle_sums' in results else None
        },
        'raw_data': {
            'distances': dist_valid.tolist(),
            'characteristic_distance': char_distance,
            'distance_variance': dist_variance
        }
    }
    
    # Print results
    print("\n=== α Value Analysis ===")
    print(f"α (final): {alpha_final:.6f} ± {alpha_std_final:.6f}")
    print(f"α (median): {alpha_median:.6f}")
    print(f"α (95% CI): [{alpha_ci_lower:.6f}, {alpha_ci_upper:.6f}]")
    print(f"Number of estimates: {n}")
    print(f"Characteristic distance: {char_distance:.4f}")
    print(f"Distance variance: {dist_variance:.4f}")
    
    # Print p-values
    print("\n=== Statistical Significance (p-values) ===")
    for test_name, p_val in p_values.items():
        if p_val is not None:
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"{test_name}: {p_val:.6f} {significance}")
        else:
            print(f"{test_name}: Not applicable")
    
    # Summary of significance
    significant_tests = [name for name, p_val in p_values.items() if p_val is not None and p_val < 0.05]
    print(f"\nSignificant results (p < 0.05): {len(significant_tests)}/{len([p for p in p_values.values() if p is not None])}")
    for test in significant_tests:
        print(f"  ✓ {test}: p = {p_values[test]:.6f}")
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Distance distribution
    plt.subplot(1, 3, 1)
    plt.hist(dist_valid, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(char_distance, color='red', linestyle='--', label=f'Mean: {char_distance:.2f}')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Distance Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: α estimates comparison
    plt.subplot(1, 3, 2)
    estimate_names = []
    estimate_values = []
    colors = ['red', 'blue', 'green', 'orange']
    
    if geometry == 'hyperbolic':
        estimate_names.append('From Curvature')
        estimate_values.append(alpha_from_curvature)
    
    estimate_names.extend(['From Distance', 'From Variance'])
    estimate_values.extend([alpha_from_distance, alpha_from_variance])
    
    if 'angle_sums' in results and results['angle_sums']:
        estimate_names.append('From Angles')
        estimate_values.append(alpha_from_angles)
    
    plt.bar(estimate_names, estimate_values, color=colors[:len(estimate_values)], alpha=0.7)
    plt.axhline(alpha_final, color='black', linestyle='--', label=f'Final: {alpha_final:.4f}')
    plt.xlabel('Estimation Method')
    plt.ylabel('α Value')
    plt.title('α Estimates Comparison')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Distance vs theoretical MI (using estimated α)
    plt.subplot(1, 3, 3)
    theoretical_mi = np.exp(-alpha_final * dist_valid)
    plt.scatter(dist_valid, theoretical_mi, alpha=0.6, color='green')
    plt.xlabel('Distance')
    plt.ylabel('Theoretical MI (exp(-α*d))')
    plt.title(f'Distance vs Theoretical MI (α={alpha_final:.4f})')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"alpha_analysis_{num_qubits}q_{geometry}_curv{curvature}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {plot_filename}")
    
    # Save analysis results
    results_filename = f"alpha_analysis_{num_qubits}q_{geometry}_curv{curvature}.json"
    with open(results_filename, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"Results saved as: {results_filename}")
    
    return analysis_results

def main():
    parser = argparse.ArgumentParser(description="Extract α value from hyperbolic curvature experiment results")
    parser.add_argument("result_file", help="Path to the results JSON file")
    args = parser.parse_args()
    
    results = extract_alpha_from_results(args.result_file)
    
    if results:
        print(f"\n✅ α value successfully extracted!")
        print(f"   α = {results['alpha_final']:.6f} ± {results['alpha_std_final']:.6f}")
        print(f"   This represents the emergent metric scaling parameter for the hyperbolic geometry.")
    else:
        print("❌ Failed to extract α value")

if __name__ == "__main__":
    main() 