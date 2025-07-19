#!/usr/bin/env python3
"""
Curvature Analysis: Does the data show emergent curvature?
Analyzes angle sums to determine if hyperbolic curvature is detected.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

def analyze_curvature_detection(results_file):
    """Analyze whether the data shows emergent curvature."""
    
    print("=" * 70)
    print("EMERGENT CURVATURE ANALYSIS")
    print("=" * 70)
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Extract key parameters
    spec = data['spec']
    geometry = spec['geometry']
    curvature = spec['curvature']
    angle_sums = data['angle_sums']
    
    print(f"\nEXPERIMENT PARAMETERS:")
    print(f"  Geometry: {geometry}")
    print(f"  Curvature: {curvature}")
    print(f"  Number of triangles: {len(angle_sums)}")
    
    # Calculate key statistics
    mean_angle_sum = np.mean(angle_sums)
    std_angle_sum = np.std(angle_sums)
    min_angle_sum = np.min(angle_sums)
    max_angle_sum = np.max(angle_sums)
    
    print(f"\nANGLE SUMS STATISTICS:")
    print(f"  Mean: {mean_angle_sum:.6f} radians")
    print(f"  Standard deviation: {std_angle_sum:.6f} radians")
    print(f"  Minimum: {min_angle_sum:.6f} radians")
    print(f"  Maximum: {max_angle_sum:.6f} radians")
    
    # Expected values for different geometries
    pi = np.pi
    expected_flat = pi
    expected_spherical = pi + 0.1  # Example for positive curvature
    expected_hyperbolic = pi - 0.1  # Example for negative curvature
    
    print(f"\nEXPECTED VALUES:")
    print(f"  Flat (Euclidean) geometry: π ≈ {pi:.6f} radians")
    print(f"  Spherical geometry: > π radians")
    print(f"  Hyperbolic geometry: < π radians")
    
    # Calculate deviation from flat geometry
    deviation_from_flat = pi - mean_angle_sum
    percent_deviation = (deviation_from_flat / pi) * 100
    
    print(f"\nDEVIATION FROM FLAT GEOMETRY:")
    print(f"  Absolute deviation: {deviation_from_flat:.6f} radians")
    print(f"  Percent deviation: {percent_deviation:.2f}%")
    
    # Statistical significance test
    # Null hypothesis: angle sums are from flat geometry (mean = π)
    t_stat, p_value = stats.ttest_1samp(angle_sums, pi)
    
    print(f"\nSTATISTICAL SIGNIFICANCE:")
    print(f"  t-statistic: {t_stat:.6f}")
    print(f"  p-value: {p_value:.10f}")
    print(f"  Significance level: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'Not significant'}")
    
    # Check consistency across triangles
    triangles_below_pi = sum(1 for angle in angle_sums if angle < pi)
    triangles_above_pi = sum(1 for angle in angle_sums if angle > pi)
    triangles_at_pi = sum(1 for angle in angle_sums if abs(angle - pi) < 0.01)
    
    print(f"\nCONSISTENCY ANALYSIS:")
    print(f"  Triangles with angle sum < π: {triangles_below_pi}/{len(angle_sums)} ({triangles_below_pi/len(angle_sums)*100:.1f}%)")
    print(f"  Triangles with angle sum > π: {triangles_above_pi}/{len(angle_sums)} ({triangles_above_pi/len(angle_sums)*100:.1f}%)")
    print(f"  Triangles with angle sum ≈ π: {triangles_at_pi}/{len(angle_sums)} ({triangles_at_pi/len(angle_sums)*100:.1f}%)")
    
    # Determine curvature type
    print(f"\nCURVATURE DETECTION:")
    
    if p_value < 0.001:  # Highly significant
        if mean_angle_sum < pi:
            print(f"  ✅ STRONG EVIDENCE FOR HYPERBOLIC CURVATURE")
            print(f"     - Mean angle sum ({mean_angle_sum:.6f}) < π ({pi:.6f})")
            print(f"     - {triangles_below_pi}/{len(angle_sums)} triangles show hyperbolic behavior")
            print(f"     - p-value ({p_value:.10f}) indicates high statistical significance")
        elif mean_angle_sum > pi:
            print(f"  ✅ STRONG EVIDENCE FOR SPHERICAL CURVATURE")
            print(f"     - Mean angle sum ({mean_angle_sum:.6f}) > π ({pi:.6f})")
            print(f"     - {triangles_above_pi}/{len(angle_sums)} triangles show spherical behavior")
            print(f"     - p-value ({p_value:.10f}) indicates high statistical significance")
    elif p_value < 0.05:  # Significant
        if mean_angle_sum < pi:
            print(f"  ⚠️  EVIDENCE FOR HYPERBOLIC CURVATURE")
        elif mean_angle_sum > pi:
            print(f"  ⚠️  EVIDENCE FOR SPHERICAL CURVATURE")
    else:
        print(f"  ❌ NO SIGNIFICANT EVIDENCE FOR CURVATURE")
        print(f"     - Angle sums consistent with flat geometry")
    
    # Compare with experiment design
    print(f"\nCOMPARISON WITH EXPERIMENT DESIGN:")
    print(f"  Designed geometry: {geometry}")
    print(f"  Designed curvature: {curvature}")
    
    if geometry == 'hyperbolic' and mean_angle_sum < pi:
        print(f"  ✅ EXPERIMENTAL RESULTS MATCH DESIGN INTENT")
        print(f"     - Designed for hyperbolic geometry")
        print(f"     - Observed hyperbolic behavior (angle sums < π)")
    elif geometry == 'spherical' and mean_angle_sum > pi:
        print(f"  ✅ EXPERIMENTAL RESULTS MATCH DESIGN INTENT")
        print(f"     - Designed for spherical geometry")
        print(f"     - Observed spherical behavior (angle sums > π)")
    else:
        print(f"  ⚠️  EXPERIMENTAL RESULTS DIFFER FROM DESIGN INTENT")
        print(f"     - Designed for {geometry} geometry")
        print(f"     - Observed different behavior")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Angle sums distribution
    plt.subplot(2, 2, 1)
    plt.hist(angle_sums, bins=15, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(pi, color='red', linestyle='--', linewidth=2, label=f'π = {pi:.2f}')
    plt.axvline(mean_angle_sum, color='green', linestyle='-', linewidth=2, label=f'Mean = {mean_angle_sum:.3f}')
    plt.xlabel('Angle Sum (radians)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Triangle Angle Sums')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Deviation from flat geometry
    plt.subplot(2, 2, 2)
    deviations = [pi - angle for angle in angle_sums]
    plt.hist(deviations, bins=15, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Flat geometry')
    plt.axvline(np.mean(deviations), color='green', linestyle='-', linewidth=2, label=f'Mean deviation = {np.mean(deviations):.3f}')
    plt.xlabel('Deviation from π (radians)')
    plt.ylabel('Frequency')
    plt.title('Deviation from Flat Geometry')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Angle sums vs triangle index
    plt.subplot(2, 2, 3)
    plt.plot(range(len(angle_sums)), angle_sums, 'bo-', alpha=0.7, markersize=4)
    plt.axhline(pi, color='red', linestyle='--', linewidth=2, label=f'π = {pi:.2f}')
    plt.axhline(mean_angle_sum, color='green', linestyle='-', linewidth=2, label=f'Mean = {mean_angle_sum:.3f}')
    plt.xlabel('Triangle Index')
    plt.ylabel('Angle Sum (radians)')
    plt.title('Angle Sums by Triangle')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    plt.subplot(2, 2, 4)
    categories = ['< π', '≈ π', '> π']
    counts = [triangles_below_pi, triangles_at_pi, triangles_above_pi]
    colors = ['blue', 'gray', 'red']
    plt.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Number of Triangles')
    plt.title('Triangle Classification')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, count in enumerate(counts):
        plt.text(i, count + 0.5, str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"curvature_analysis_{geometry}_{curvature}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {plot_filename}")
    
    # Final conclusion
    print(f"\n" + "=" * 70)
    print("FINAL CONCLUSION")
    print("=" * 70)
    
    if p_value < 0.001 and mean_angle_sum < pi:
        print(f"✅ YES - The data shows STRONG EVIDENCE of emergent hyperbolic curvature")
        print(f"   - Statistical significance: p < 0.001")
        print(f"   - Mean angle sum: {mean_angle_sum:.6f} radians (expected π ≈ {pi:.6f})")
        print(f"   - Deviation: {deviation_from_flat:.6f} radians ({percent_deviation:.1f}% reduction)")
        print(f"   - Consistency: {triangles_below_pi}/{len(angle_sums)} triangles show hyperbolic behavior")
    elif p_value < 0.05 and mean_angle_sum < pi:
        print(f"⚠️  YES - The data shows EVIDENCE of emergent hyperbolic curvature")
        print(f"   - Statistical significance: p < 0.05")
        print(f"   - Mean angle sum: {mean_angle_sum:.6f} radians")
    else:
        print(f"❌ NO - The data does not show significant evidence of emergent curvature")
        print(f"   - Statistical significance: p = {p_value:.6f}")
        print(f"   - Mean angle sum: {mean_angle_sum:.6f} radians")
    
    return {
        'mean_angle_sum': mean_angle_sum,
        'deviation_from_flat': deviation_from_flat,
        'p_value': p_value,
        'triangles_below_pi': triangles_below_pi,
        'total_triangles': len(angle_sums),
        'shows_curvature': p_value < 0.05 and mean_angle_sum < pi
    }

def main():
    """Main function to run curvature analysis."""
    results_file = "experiment_logs/custom_curvature_experiment/results_n7_geomH_curv05_ibm_60JXA8.json"
    
    if not Path(results_file).exists():
        print(f"File not found: {results_file}")
        return
    
    results = analyze_curvature_detection(results_file)
    
    # Print summary
    print(f"\nSUMMARY:")
    print(f"  Shows emergent curvature: {'YES' if results['shows_curvature'] else 'NO'}")
    print(f"  Curvature type: Hyperbolic (if detected)")
    print(f"  Statistical significance: p = {results['p_value']:.10f}")

if __name__ == "__main__":
    main() 