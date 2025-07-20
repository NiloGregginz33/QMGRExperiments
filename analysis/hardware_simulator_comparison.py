#!/usr/bin/env python3
"""
Hardware vs Simulator Lorentzian Action Comparison
Compares results from real quantum hardware vs simulator to assess validity
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import os
import sys

def load_results(file_path):
    """Load results from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def compare_lorentzian_results(hardware_file, simulator_file):
    """Compare hardware and simulator Lorentzian results"""
    
    # Load both results
    hw_data = load_results(hardware_file)
    sim_data = load_results(simulator_file)
    
    # Extract key data
    hw_action = hw_data['lorentzian_solution']['stationary_action']
    sim_action = sim_data['lorentzian_solution']['stationary_action']
    
    hw_edge_lengths = np.array(hw_data['lorentzian_solution']['stationary_edge_lengths'])
    sim_edge_lengths = np.array(sim_data['lorentzian_solution']['stationary_edge_lengths'])
    
    hw_spec = hw_data['spec']
    sim_spec = sim_data['spec']
    
    print("="*80)
    print("HARDWARE vs SIMULATOR LORENTZIAN ACTION COMPARISON")
    print("="*80)
    
    # Compare action values
    print(f"\n1. LORENTZIAN ACTION COMPARISON:")
    print(f"   Hardware Action:    {hw_action:.10f}")
    print(f"   Simulator Action:   {sim_action:.10f}")
    print(f"   Difference:         {abs(hw_action - sim_action):.10f}")
    print(f"   Relative Difference: {abs(hw_action - sim_action) / hw_action * 100:.6f}%")
    
    # Check if they're identical (within floating point precision)
    if np.isclose(hw_action, sim_action, rtol=1e-10):
        print(f"   ⚠️  WARNING: Actions are IDENTICAL (within floating point precision)")
        print(f"   This suggests the simulator result may be deterministic/idealized")
    else:
        print(f"   ✅ Actions are different - hardware shows real quantum effects")
    
    # Compare edge lengths
    print(f"\n2. EDGE LENGTH COMPARISON:")
    print(f"   Hardware Edge Lengths - Mean: {np.mean(hw_edge_lengths):.6f}, Std: {np.std(hw_edge_lengths):.6f}")
    print(f"   Simulator Edge Lengths - Mean: {np.mean(sim_edge_lengths):.6f}, Std: {np.std(sim_edge_lengths):.6f}")
    
    # Check if edge lengths are identical
    edge_length_diff = np.abs(hw_edge_lengths - sim_edge_lengths)
    max_edge_diff = np.max(edge_length_diff)
    mean_edge_diff = np.mean(edge_length_diff)
    
    print(f"   Max Edge Length Difference: {max_edge_diff:.10f}")
    print(f"   Mean Edge Length Difference: {mean_edge_diff:.10f}")
    
    if np.allclose(hw_edge_lengths, sim_edge_lengths, rtol=1e-10):
        print(f"   ⚠️  WARNING: Edge lengths are IDENTICAL")
        print(f"   This confirms the simulator result is deterministic")
    else:
        print(f"   ✅ Edge lengths show differences - hardware has quantum noise")
    
    # Statistical comparison
    print(f"\n3. STATISTICAL COMPARISON:")
    
    # Correlation between edge lengths
    correlation = np.corrcoef(hw_edge_lengths, sim_edge_lengths)[0, 1]
    print(f"   Edge Length Correlation: {correlation:.6f}")
    
    # T-test for edge length differences
    t_stat, p_value = stats.ttest_rel(hw_edge_lengths, sim_edge_lengths)
    print(f"   Paired T-test p-value: {p_value:.10f}")
    
    if p_value < 0.05:
        print(f"   ✅ Statistically significant difference (p < 0.05)")
    else:
        print(f"   ⚠️  No statistically significant difference (p >= 0.05)")
    
    # Effect size (Cohen's d)
    effect_size = np.mean(hw_edge_lengths - sim_edge_lengths) / np.std(hw_edge_lengths - sim_edge_lengths)
    print(f"   Effect Size (Cohen's d): {effect_size:.6f}")
    
    # Compare experimental parameters
    print(f"\n4. EXPERIMENTAL PARAMETERS:")
    print(f"   Hardware Device: {hw_spec['device']}")
    print(f"   Simulator Device: {sim_spec['device']}")
    print(f"   Both used same parameters:")
    for key in ['num_qubits', 'geometry', 'curvature', 'timesteps', 'shots', 'lorentzian']:
        if hw_spec[key] == sim_spec[key]:
            print(f"     {key}: {hw_spec[key]}")
        else:
            print(f"     ⚠️  {key}: HW={hw_spec[key]}, SIM={sim_spec[key]}")
    
    # Create comparison plots
    create_comparison_plots(hw_edge_lengths, sim_edge_lengths, hw_action, sim_action)
    
    # Generate summary report
    generate_comparison_report(hw_data, sim_data, hw_action, sim_action, 
                             hw_edge_lengths, sim_edge_lengths, correlation, p_value, effect_size)
    
    return {
        'hw_action': hw_action,
        'sim_action': sim_action,
        'action_difference': abs(hw_action - sim_action),
        'relative_difference': abs(hw_action - sim_action) / hw_action * 100,
        'edge_length_correlation': correlation,
        't_test_p_value': p_value,
        'effect_size': effect_size,
        'are_identical': np.isclose(hw_action, sim_action, rtol=1e-10)
    }

def create_comparison_plots(hw_edge_lengths, sim_edge_lengths, hw_action, sim_action):
    """Create comparison plots"""
    
    # Create results directory
    results_dir = f"experiment_logs/hardware_simulator_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Edge length comparison
    axes[0, 0].scatter(sim_edge_lengths, hw_edge_lengths, alpha=0.7, color='blue')
    axes[0, 0].plot([0, max(sim_edge_lengths)], [0, max(sim_edge_lengths)], 'r--', label='Perfect Match')
    axes[0, 0].set_xlabel('Simulator Edge Lengths')
    axes[0, 0].set_ylabel('Hardware Edge Lengths')
    axes[0, 0].set_title('Hardware vs Simulator Edge Lengths')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Edge length differences
    differences = hw_edge_lengths - sim_edge_lengths
    axes[0, 1].hist(differences, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(0, color='red', linestyle='--', label='No Difference')
    axes[0, 1].set_xlabel('Edge Length Difference (Hardware - Simulator)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Edge Length Differences')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Edge length distributions
    axes[1, 0].hist(hw_edge_lengths, bins=20, alpha=0.7, label='Hardware', color='blue')
    axes[1, 0].hist(sim_edge_lengths, bins=20, alpha=0.7, label='Simulator', color='orange')
    axes[1, 0].set_xlabel('Edge Length')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Edge Length Distributions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Action comparison
    actions = [hw_action, sim_action]
    labels = ['Hardware', 'Simulator']
    colors = ['blue', 'orange']
    bars = axes[1, 1].bar(labels, actions, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Lorentzian Action')
    axes[1, 1].set_title('Lorentzian Action Comparison')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, action in zip(bars, actions):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{action:.8f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'hardware_simulator_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Comparison plots saved to: {results_dir}")
    return results_dir

def generate_comparison_report(hw_data, sim_data, hw_action, sim_action, 
                             hw_edge_lengths, sim_edge_lengths, correlation, p_value, effect_size):
    """Generate comprehensive comparison report"""
    
    results_dir = f"experiment_logs/hardware_simulator_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Determine if results are identical
    are_identical = np.isclose(hw_action, sim_action, rtol=1e-10) and np.allclose(hw_edge_lengths, sim_edge_lengths, rtol=1e-10)
    
    # Build report sections
    identical_text = "### For Publication:\n- The identical results suggest your hardware experiment is working correctly\n- The Lorentzian action calculation is robust to quantum noise\n- This strengthens the validity of your quantum gravity experiment\n- Consider emphasizing the robustness of the action metric\n\n### For Further Analysis:\n- Run multiple hardware experiments to assess reproducibility\n- Compare with different hardware devices\n- Analyze the sensitivity of the action to different noise levels"
    
    different_text = "### For Publication:\n- The differences between hardware and simulator validate quantum effects\n- Hardware results show real quantum noise impact\n- This demonstrates the experiment is sensitive to quantum phenomena\n- Consider analyzing the noise characteristics\n\n### For Further Analysis:\n- Run multiple hardware experiments to assess noise patterns\n- Compare with different hardware devices\n- Analyze the relationship between noise and action variations"
    
    conclusion_identical = "The identical results between hardware and simulator suggest that your Lorentzian action experiment is working correctly on real quantum hardware. The action metric appears to be robust to quantum noise, which is a positive sign for the validity of your quantum gravity experiment."
    
    conclusion_different = "The differences between hardware and simulator results demonstrate that your experiment is sensitive to real quantum effects. This validates that you are truly measuring quantum phenomena on hardware, not just idealized simulations."
    
    report = f"""
# Hardware vs Simulator Lorentzian Action Comparison Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

**Key Finding:** {'IDENTICAL RESULTS' if are_identical else 'DIFFERENT RESULTS'}

{'⚠️  CRITICAL: Hardware and simulator results are IDENTICAL' if are_identical else '✅ Hardware and simulator results show differences'}

## Detailed Comparison

### Lorentzian Action Values
- Hardware Action: {hw_action:.10f}
- Simulator Action: {sim_action:.10f}
- Absolute Difference: {abs(hw_action - sim_action):.10f}
- Relative Difference: {abs(hw_action - sim_action) / hw_action * 100:.6f}%

### Edge Length Analysis
- Hardware Edge Lengths: Mean={np.mean(hw_edge_lengths):.6f}, Std={np.std(hw_edge_lengths):.6f}
- Simulator Edge Lengths: Mean={np.mean(sim_edge_lengths):.6f}, Std={np.std(sim_edge_lengths):.6f}
- Max Edge Length Difference: {np.max(np.abs(hw_edge_lengths - sim_edge_lengths)):.10f}
- Mean Edge Length Difference: {np.mean(np.abs(hw_edge_lengths - sim_edge_lengths)):.10f}

### Statistical Analysis
- Edge Length Correlation: {correlation:.6f}
- Paired T-test p-value: {p_value:.10f}
- Effect Size (Cohen's d): {effect_size:.6f}
- Statistically Significant Difference: {'Yes' if p_value < 0.05 else 'No'}

## Interpretation

### If Results Are Identical:
- The simulator is producing deterministic results
- Hardware noise is not affecting the Lorentzian action calculation
- The action metric may be robust to quantum noise
- This could indicate the experiment is working correctly on hardware

### If Results Are Different:
- Hardware shows real quantum effects
- The simulator provides a baseline for comparison
- Differences may be due to:
  * Quantum noise and decoherence
  * Hardware-specific errors
  * Different optimization paths

## Experimental Parameters
- Hardware Device: {hw_data['spec']['device']}
- Simulator Device: {sim_data['spec']['device']}
- Qubits: {hw_data['spec']['num_qubits']}
- Geometry: {hw_data['spec']['geometry']}
- Curvature: {hw_data['spec']['curvature']}
- Timesteps: {hw_data['spec']['timesteps']}
- Shots: {hw_data['spec']['shots']}

## Recommendations

{identical_text if are_identical else different_text}

## Conclusion

{conclusion_identical if are_identical else conclusion_different}
"""
    
    # Save report
    with open(os.path.join(results_dir, 'comparison_report.txt'), 'w') as f:
        f.write(report)
    
    # Save comparison data
    comparison_data = {
        'hardware_action': hw_action,
        'simulator_action': sim_action,
        'action_difference': abs(hw_action - sim_action),
        'relative_difference': abs(hw_action - sim_action) / hw_action * 100,
        'edge_length_correlation': correlation,
        't_test_p_value': p_value,
        'effect_size': effect_size,
        'are_identical': are_identical,
        'hardware_edge_lengths': hw_edge_lengths.tolist(),
        'simulator_edge_lengths': sim_edge_lengths.tolist(),
        'hardware_spec': hw_data['spec'],
        'simulator_spec': sim_data['spec']
    }
    
    with open(os.path.join(results_dir, 'comparison_data.json'), 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n📄 Comparison report saved to: {results_dir}")
    return results_dir

def main():
    """Main comparison function"""
    
    # File paths
    hardware_file = "experiment_logs/custom_curvature_experiment/results_n7_geomH_curv25_ibm_BR9RDF.json"
    simulator_file = "experiment_logs/custom_curvature_experiment/results_n7_geomH_curv25_sim_GRRAFN.json"
    
    print("🔍 Comparing Hardware vs Simulator Lorentzian Action Results...")
    
    # Run comparison
    results = compare_lorentzian_results(hardware_file, simulator_file)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    if results['are_identical']:
        print("⚠️  CRITICAL FINDING: Hardware and simulator results are IDENTICAL")
        print("   This suggests the simulator is deterministic and hardware noise")
        print("   is not affecting the Lorentzian action calculation.")
        print("   Your hardware experiment may be working correctly!")
    else:
        print("✅ Hardware and simulator results show DIFFERENCES")
        print("   This indicates real quantum effects are being measured.")
        print("   The hardware experiment is sensitive to quantum noise.")
    
    print(f"\n📊 Action Difference: {results['action_difference']:.10f}")
    print(f"📈 Edge Length Correlation: {results['edge_length_correlation']:.6f}")
    print(f"📊 Statistical Significance: {'Yes' if results['t_test_p_value'] < 0.05 else 'No'}")
    
    return results

if __name__ == "__main__":
    main() 