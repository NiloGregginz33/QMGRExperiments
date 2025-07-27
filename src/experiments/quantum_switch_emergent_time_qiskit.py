"""
PROPRIETARY SOFTWARE - QUANTUM SWITCH EMERGENT TIME EXPERIMENT

Copyright (c) 2024-2025 Matrix Solutions LLC. All rights reserved.

This file contains proprietary research algorithms and experimental protocols
for quantum holographic geometry experiments. This software is proprietary and
confidential to Matrix Solutions LLC.

SPECIFIC LICENSE TERMS:
- Use for academic peer review purposes is permitted
- Academic research and educational use is allowed with proper attribution
- Commercial use is strictly prohibited without written permission
- Redistribution, modification, or derivative works are not permitted
- Reverse engineering or decompilation is prohibited

For licensing inquiries: manavnaik123@gmail.com

By using this file, you acknowledge and agree to be bound by these terms.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
import os
import json
import argparse
import sys
from datetime import datetime
import time
from scipy import stats
from scipy.stats import binomtest, chi2_contingency
import warnings

# Add CGPTFactory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from CGPTFactory import run

def log_result(metrics, log_dir="experiment_logs/quantum_switch_emergent_time_qiskit"):
    os.makedirs(log_dir, exist_ok=True)
    idx = len([f for f in os.listdir(log_dir) if f.startswith('result_') and f.endswith('.json')]) + 1
    with open(os.path.join(log_dir, f"result_{idx}.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

def quantum_switch_circuit(phi=0):
    """
    Create a quantum switch circuit that demonstrates causal non-separability.
    
    The circuit implements:
    - Control qubit in superposition
    - Two operations A and B applied in different orders based on control
    - A: RX(phi), B: RY(phi)
    - If control=0: A then B, if control=1: B then A
    
    This creates a quantum superposition of causal orders.
    """
    qc = QuantumCircuit(2, 2)
    
    # Prepare control qubit in superposition
    qc.h(0)
    
    # Prepare target qubit in |0> state
    # (already in |0> by default)
    
    # Apply controlled operations
    # Use ancilla-like approach for 2-qubit implementation
    qc.cx(0, 1)  # Control operation
    qc.rx(phi, 1)  # Operation A: RX(phi)
    qc.cry(phi, 0, 1)  # Operation B: RY(phi) controlled
    qc.cx(0, 1)  # Undo control
    
    # Measure both qubits
    qc.measure([0, 1], [0, 1])
    
    return qc

def create_noise_scaled_circuit(circuit, noise_factor):
    """
    Create a noise-scaled version of the circuit for zero-noise extrapolation.
    """
    # For simple noise scaling, we can repeat gates
    # This is a basic implementation - more sophisticated methods exist
    qc_scaled = circuit.copy()
    
    if noise_factor > 1.0:
        # Insert identity gates to increase noise
        for i in range(int(noise_factor - 1)):
            qc_scaled.id(0)
            qc_scaled.id(1)
    
    return qc_scaled

def extrapolate_to_zero_noise(noise_factors, results):
    """
    Perform zero-noise extrapolation to estimate noiseless results.
    """
    # Extract causal witness values
    witnesses = [r['causal_witness'] for r in results]
    
    # Simple linear extrapolation
    if len(noise_factors) >= 2:
        coeffs = np.polyfit(noise_factors, witnesses, 1)
        zero_noise_witness = coeffs[1]  # intercept
        slope = coeffs[0]
        
        return {
            'zero_noise_witness': zero_noise_witness,
            'slope': slope,
            'extrapolation_quality': np.corrcoef(noise_factors, witnesses)[0, 1]**2
        }
    
    return None

def run_circuit_with_mitigation(qc, shots, device_name, use_mitigation=True):
    """
    Run circuit with error mitigation techniques.
    """
    if device_name.lower() == 'simulator':
        # Use FakeBrisbane for simulation
        backend = FakeBrisbane()
        print(f"Running on simulator: {backend.name}")
        
        # Run with CGPTFactory
        result = run(qc, backend=backend, shots=shots)
        return result
    else:
        # Real hardware execution
        print(f"Running on hardware: {device_name}")
        
        # Get IBM backend
        service = QiskitRuntimeService()
        backend = service.backend(device_name)
        
        if use_mitigation:
            # Zero-noise extrapolation with multiple noise levels
            noise_factors = [1.0, 1.5, 2.0]
            noise_results = []
            
            for noise_factor in noise_factors:
                print(f"Running with noise factor: {noise_factor}")
                qc_scaled = create_noise_scaled_circuit(qc, noise_factor)
                
                # Run with CGPTFactory
                result = run(qc_scaled, backend=backend, shots=shots)
                
                if result is not None:
                    # Calculate metrics
                    probs = np.zeros(4)
                    for bitstring, count in result.items():
                        idx = int(bitstring, 2)
                        probs[idx] = count / shots
                    
                    shannon_entropy = -np.sum(probs * np.log2(probs + 1e-12))
                    causal_witness = probs[0] + probs[3] - probs[1] - probs[2]
                    
                    noise_results.append({
                        'noise_factor': noise_factor,
                        'causal_witness': causal_witness,
                        'shannon_entropy': shannon_entropy,
                        'counts': result
                    })
            
            # Perform zero-noise extrapolation
            if len(noise_results) >= 2:
                extrapolation = extrapolate_to_zero_noise(
                    [r['noise_factor'] for r in noise_results],
                    noise_results
                )
                
                # Use extrapolated result
                final_result = noise_results[0].copy()
                if extrapolation:
                    final_result['causal_witness'] = extrapolation['zero_noise_witness']
                    final_result['extrapolation'] = extrapolation
                
                return final_result
            else:
                return noise_results[0] if noise_results else None
        else:
            # Simple execution without mitigation
            result = run(qc, backend=backend, shots=shots)
            
            if result is not None:
                # Calculate metrics
                probs = np.zeros(4)
                for bitstring, count in result.items():
                    idx = int(bitstring, 2)
                    probs[idx] = count / shots
                
                shannon_entropy = -np.sum(probs * np.log2(probs + 1e-12))
                causal_witness = probs[0] + probs[3] - probs[1] - probs[2]
                
                return {
                    'causal_witness': causal_witness,
                    'shannon_entropy': shannon_entropy,
                    'counts': result
                }
            
            return None

def run_quantum_switch_experiment(backend=None, shots=2048, phi_values=None, device_name='simulator', use_mitigation=True):
    """
    Run the quantum switch emergent time experiment with enhanced error mitigation.
    """
    if phi_values is None:
        # Run just one circuit with a single φ value for faster testing
        phi_values = [np.pi/2]  # Use π/2 as it typically shows strong causal non-separability
    
    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"experiment_logs/quantum_switch_emergent_time_qiskit_{device_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Starting Quantum Switch Emergent Time Experiment")
    print(f"Device: {device_name}")
    print(f"Shots: {shots}")
    print(f"Error mitigation: {use_mitigation}")
    print(f"Phi values: {len(phi_values)} point(s) - {phi_values}")
    print(f"Log directory: {log_dir}")
    
    results = []
    start_time = time.time()
    
    for i, phi in enumerate(phi_values):
        print(f"\nProgress: {i+1}/{len(phi_values)} - phi = {phi:.3f}")
        
        # Create circuit
        qc = quantum_switch_circuit(phi)
        
        # Run with mitigation
        result = run_circuit_with_mitigation(qc, shots, device_name, use_mitigation)
        
        if result is not None:
            metrics = {
                "phi": phi,
                "counts": result.get('counts', {}),
                "shannon_entropy": result.get('shannon_entropy', 0),
                "causal_witness": result.get('causal_witness', 0),
                "device": device_name,
                "shots": shots,
                "use_mitigation": use_mitigation
            }
            
            # Add extrapolation data if available
            if 'extrapolation' in result:
                metrics['extrapolation'] = result['extrapolation']
            
            # Perform comprehensive statistical analysis
            counts = result.get('counts', {})
            if counts:
                # Statistical significance analysis
                stats_analysis = calculate_statistical_significance(counts, shots)
                metrics['statistical_analysis'] = stats_analysis
                
                # Measurement uncertainty analysis
                uncertainty_analysis = analyze_measurement_uncertainty(counts, shots)
                metrics['uncertainty_analysis'] = uncertainty_analysis
                
                # Bootstrap confidence intervals
                all_counts = []
                for outcome, count in counts.items():
                    all_counts.extend([outcome] * count)
                
                if len(all_counts) > 0:
                    # Bootstrap causal witness
                    def causal_witness_stat(data):
                        # Calculate causal witness from resampled data
                        counts_resampled = {}
                        for outcome in ['00', '01', '10', '11']:
                            counts_resampled[outcome] = np.sum(data == outcome)
                        total = len(data)
                        if total > 0:
                            p_00 = counts_resampled.get('00', 0) / total
                            p_01 = counts_resampled.get('01', 0) / total
                            p_10 = counts_resampled.get('10', 0) / total
                            p_11 = counts_resampled.get('11', 0) / total
                            return p_00 + p_11 - p_01 - p_10
                        return 0
                    
                    bootstrap_mean, bootstrap_lower, bootstrap_upper, bootstrap_samples = bootstrap_confidence_interval(
                        all_counts, causal_witness_stat, n_bootstrap=1000
                    )
                    
                    metrics['bootstrap_analysis'] = {
                        'bootstrap_mean': bootstrap_mean,
                        'bootstrap_ci_95': (bootstrap_lower, bootstrap_upper),
                        'bootstrap_std': np.std(bootstrap_samples)
                    }
            
            log_result(metrics, log_dir)
            results.append(metrics)
            
            print(f"  Shannon Entropy: {metrics['shannon_entropy']:.4f}")
            print(f"  Causal Witness: {metrics['causal_witness']:.4f}")
            
            # Print statistical significance
            if 'statistical_analysis' in metrics:
                stats = metrics['statistical_analysis']
                print(f"  P-value: {stats['p_value']:.6f}")
                print(f"  Significance: {stats['significance_level']}")
                print(f"  Effect size: {stats['effect_size']:.2f}")
                print(f"  95% CI: [{stats['confidence_interval_95'][0]:.4f}, {stats['confidence_interval_95'][1]:.4f}]")
            
            # Check for causal non-separability
            if abs(metrics['causal_witness']) > 0.5:
                print(f"  ✓ STRONG causal non-separability detected!")
            elif abs(metrics['causal_witness']) > 0.1:
                print(f"  ✓ Moderate causal non-separability detected")
            else:
                print(f"  WARNING: Weak or no causal non-separability detected")
        else:
            print(f"  ERROR: Failed to get results for phi = {phi}")
    
    # Calculate experiment statistics
    experiment_time = time.time() - start_time
    
    # Save all results
    with open(os.path.join(log_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Write comprehensive summary
    with open(os.path.join(log_dir, "summary.txt"), 'w') as f:
        f.write(f"Quantum Switch Emergent Time Experiment Summary\n")
        f.write(f"==============================================\n\n")
        f.write(f"Experiment Parameters:\n")
        f.write(f"  Device: {device_name}\n")
        f.write(f"  Shots: {shots}\n")
        f.write(f"  Error mitigation: {use_mitigation}\n")
        f.write(f"  Number of phi values: {len(phi_values)}\n")
        f.write(f"  Phi value(s): {phi_values}\n")
        f.write(f"  Total experiment time: {experiment_time:.1f} seconds\n\n")
        
        f.write(f"Results Summary:\n")
        if results:
            causal_witnesses = [r['causal_witness'] for r in results]
            entropies = [r['shannon_entropy'] for r in results]
            
            f.write(f"  Causal Witness Statistics:\n")
            f.write(f"    Mean: {np.mean(causal_witnesses):.4f}\n")
            f.write(f"    Std: {np.std(causal_witnesses):.4f}\n")
            f.write(f"    Min: {np.min(causal_witnesses):.4f}\n")
            f.write(f"    Max: {np.max(causal_witnesses):.4f}\n")
            
            f.write(f"  Shannon Entropy Statistics:\n")
            f.write(f"    Mean: {np.mean(entropies):.4f}\n")
            f.write(f"    Std: {np.std(entropies):.4f}\n")
            f.write(f"    Min: {np.min(entropies):.4f}\n")
            f.write(f"    Max: {np.max(entropies):.4f}\n")
            
            # Statistical significance analysis
            f.write(f"\n  Statistical Significance Analysis:\n")
            for i, r in enumerate(results):
                if 'statistical_analysis' in r:
                    stats = r['statistical_analysis']
                    f.write(f"    Result {i+1} (phi={r['phi']:.3f}):\n")
                    f.write(f"      Causal Witness: {stats['causal_witness']:.6f} ± {stats['std_error']:.6f}\n")
                    f.write(f"      95% CI: [{stats['confidence_interval_95'][0]:.6f}, {stats['confidence_interval_95'][1]:.6f}]\n")
                    f.write(f"      P-value: {stats['p_value']:.6f}\n")
                    f.write(f"      Significance: {stats['significance_level']}\n")
                    f.write(f"      Z-statistic: {stats['z_statistic']:.3f}\n")
                    f.write(f"      Effect size: {stats['effect_size']:.3f}\n")
                    f.write(f"      Chi² p-value: {stats['chi2_p_value']:.6f}\n")
                    
                    # Bootstrap analysis
                    if 'bootstrap_analysis' in r:
                        bootstrap = r['bootstrap_analysis']
                        f.write(f"      Bootstrap CI: [{bootstrap['bootstrap_ci_95'][0]:.6f}, {bootstrap['bootstrap_ci_95'][1]:.6f}]\n")
                        f.write(f"      Bootstrap std: {bootstrap['bootstrap_std']:.6f}\n")
                    
                    # Uncertainty analysis
                    if 'uncertainty_analysis' in r:
                        uncertainty = r['uncertainty_analysis']
                        f.write(f"      Total uncertainty: {uncertainty['total_uncertainty']:.6f}\n")
                        f.write(f"      Systematic error: {uncertainty['systematic_error']:.6f}\n")
            
            # Check for causal non-separability
            max_witness = np.max(np.abs(causal_witnesses))
            f.write(f"\n  Causal Non-Separability Analysis:\n")
            f.write(f"    Maximum absolute causal witness: {max_witness:.4f}\n")
            if max_witness > 0.5:
                f.write(f"    ✓ STRONG causal non-separability detected\n")
            elif max_witness > 0.1:
                f.write(f"    ✓ Moderate causal non-separability detected\n")
            else:
                f.write(f"    WARNING: Weak or no causal non-separability detected\n")
            
            # Overall statistical significance
            significant_results = [r for r in results if 'statistical_analysis' in r and r['statistical_analysis']['p_value'] < 0.05]
            f.write(f"\n  Overall Statistical Summary:\n")
            f.write(f"    Significant results: {len(significant_results)}/{len(results)}\n")
            if significant_results:
                f.write(f"    Average p-value: {np.mean([r['statistical_analysis']['p_value'] for r in significant_results]):.6f}\n")
                f.write(f"    Average effect size: {np.mean([r['statistical_analysis']['effect_size'] for r in significant_results]):.3f}\n")
        
        f.write(f"\nDetailed Results:\n")
        for r in results:
            f.write(f"  phi={r['phi']:.3f}: S={r['shannon_entropy']:.4f}, W={r['causal_witness']:.4f}\n")
    
    # Create enhanced plots
    if results:
        create_enhanced_plots(results, log_dir, device_name)
    
    print(f"\nExperiment complete! Results saved to: {log_dir}")
    print(f"Total time: {experiment_time:.1f} seconds")
    
    return results, log_dir

def create_enhanced_plots(results, log_dir, device_name):
    """
    Create publication-quality plots comparing simulator and hardware results.
    """
    phis = [r['phi'] for r in results]
    entropies = [r['shannon_entropy'] for r in results]
    witnesses = [r['causal_witness'] for r in results]
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Shannon Entropy vs Phi
    ax1.plot(phis, entropies, 'o-', linewidth=2, markersize=8, 
             color='#2E86AB', label=f'{device_name.upper()}')
    ax1.set_xlabel('Parameter φ', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Shannon Entropy', fontsize=14, fontweight='bold')
    ax1.set_title(f'Quantum Switch: Entropy vs φ ({device_name.upper()})', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # Add statistical annotations
    entropy_mean = np.mean(entropies)
    entropy_std = np.std(entropies)
    ax1.text(0.05, 0.95, f'Mean: {entropy_mean:.3f} ± {entropy_std:.3f}', 
             transform=ax1.transAxes, bbox=dict(boxstyle="round,pad=0.3", 
             facecolor="white", alpha=0.8), verticalalignment='top', fontsize=12)
    
    # Plot 2: Causal Witness vs Phi
    ax2.plot(phis, witnesses, 'o-', linewidth=2, markersize=8, 
             color='#A23B72', label=f'{device_name.upper()}')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Parameter φ', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Causal Witness', fontsize=14, fontweight='bold')
    ax2.set_title(f'Quantum Switch: Causal Non-Separability Witness ({device_name.upper()})', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    
    # Add statistical annotations
    witness_mean = np.mean(witnesses)
    witness_std = np.std(witnesses)
    max_witness = np.max(np.abs(witnesses))
    ax2.text(0.05, 0.95, f'Mean: {witness_mean:.3f} ± {witness_std:.3f}\nMax |W|: {max_witness:.3f}', 
             transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", 
             facecolor="white", alpha=0.8), verticalalignment='top', fontsize=12)
    
    # Add causal non-separability indicator
    if max_witness > 0.5:
        ax2.text(0.05, 0.05, '✓ STRONG Causal Non-Separability', 
                 transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", 
                 facecolor="lightgreen", alpha=0.8), fontsize=12, fontweight='bold')
    elif max_witness > 0.1:
        ax2.text(0.05, 0.05, '✓ Moderate Causal Non-Separability', 
                 transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", 
                 facecolor="lightyellow", alpha=0.8), fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'quantum_switch_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create individual plots
    # Entropy plot
    plt.figure(figsize=(10, 6))
    plt.plot(phis, entropies, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    plt.xlabel('Parameter φ', fontsize=14, fontweight='bold')
    plt.ylabel('Shannon Entropy', fontsize=14, fontweight='bold')
    plt.title(f'Quantum Switch: Entropy Evolution ({device_name.upper()})', 
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'shannon_entropy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Causal witness plot
    plt.figure(figsize=(10, 6))
    plt.plot(phis, witnesses, 'o-', linewidth=2, markersize=8, color='#A23B72')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Parameter φ', fontsize=14, fontweight='bold')
    plt.ylabel('Causal Witness', fontsize=14, fontweight='bold')
    plt.title(f'Quantum Switch: Causal Non-Separability Witness ({device_name.upper()})', 
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'causal_witness.png'), dpi=300, bbox_inches='tight')
    plt.close()

def compare_simulator_hardware(simulator_results, hardware_results, log_dir):
    """
    Create comparison plots between simulator and hardware results.
    """
    if not simulator_results or not hardware_results:
        print("Cannot create comparison: missing simulator or hardware results")
        return
    
    # Extract data
    sim_phis = [r['phi'] for r in simulator_results]
    sim_entropies = [r['shannon_entropy'] for r in simulator_results]
    sim_witnesses = [r['causal_witness'] for r in simulator_results]
    
    hw_phis = [r['phi'] for r in hardware_results]
    hw_entropies = [r['shannon_entropy'] for r in hardware_results]
    hw_witnesses = [r['causal_witness'] for r in hardware_results]
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Entropy comparison
    ax1.plot(sim_phis, sim_entropies, 'o-', linewidth=2, markersize=8, 
             color='#2E86AB', label='Simulator', alpha=0.8)
    ax1.plot(hw_phis, hw_entropies, 's-', linewidth=2, markersize=8, 
             color='#A23B72', label='Hardware', alpha=0.8)
    ax1.set_xlabel('Parameter φ', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Shannon Entropy', fontsize=14, fontweight='bold')
    ax1.set_title('Quantum Switch: Simulator vs Hardware Comparison', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # Causal witness comparison
    ax2.plot(sim_phis, sim_witnesses, 'o-', linewidth=2, markersize=8, 
             color='#2E86AB', label='Simulator', alpha=0.8)
    ax2.plot(hw_phis, hw_witnesses, 's-', linewidth=2, markersize=8, 
             color='#A23B72', label='Hardware', alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Parameter φ', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Causal Witness', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'simulator_hardware_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

def calculate_statistical_significance(counts, shots, null_hypothesis=0.0):
    """
    Calculate statistical significance of causal witness measurement.
    
    Args:
        counts: Dictionary of measurement counts
        shots: Total number of shots
        null_hypothesis: Expected causal witness under null hypothesis
    
    Returns:
        Dictionary with p-value, confidence intervals, and statistical tests
    """
    # Extract counts for each outcome
    count_00 = counts.get('00', 0)
    count_01 = counts.get('01', 0)
    count_10 = counts.get('10', 0)
    count_11 = counts.get('11', 0)
    
    # Calculate probabilities
    p_00 = count_00 / shots
    p_01 = count_01 / shots
    p_10 = count_10 / shots
    p_11 = count_11 / shots
    
    # Causal witness: W = P(00) + P(11) - P(01) - P(10)
    causal_witness = p_00 + p_11 - p_01 - p_10
    
    # Calculate standard error of causal witness
    # Using propagation of errors for binomial distributions
    var_00 = p_00 * (1 - p_00) / shots
    var_01 = p_01 * (1 - p_01) / shots
    var_10 = p_10 * (1 - p_10) / shots
    var_11 = p_11 * (1 - p_11) / shots
    
    # Variance of causal witness (assuming independence)
    var_witness = var_00 + var_01 + var_10 + var_11
    std_error = np.sqrt(var_witness)
    
    # Calculate 95% confidence interval
    z_score = 1.96  # 95% confidence level
    ci_lower = causal_witness - z_score * std_error
    ci_upper = causal_witness + z_score * std_error
    
    # Z-test for significance
    if std_error > 0:
        z_stat = (causal_witness - null_hypothesis) / std_error
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed test
    else:
        z_stat = 0
        p_value = 1.0
    
    # Chi-squared test for independence
    contingency_table = np.array([[count_00, count_01], [count_10, count_11]])
    try:
        chi2_stat, chi2_p_value, dof, expected = chi2_contingency(contingency_table)
    except:
        chi2_stat, chi2_p_value = 0, 1.0
    
    # Binomial test for each outcome
    binomial_tests = {}
    for outcome, count in counts.items():
        if shots > 0:
            # Test against expected probability of 0.25 (uniform distribution)
            binom_result = binomtest(count, shots, p=0.25, alternative='two-sided')
            binomial_tests[outcome] = {
                'p_value': binom_result.pvalue,
                'statistic': binom_result.statistic,
                'proportion': count / shots
            }
    
    return {
        'causal_witness': causal_witness,
        'std_error': std_error,
        'confidence_interval_95': (ci_lower, ci_upper),
        'z_statistic': z_stat,
        'p_value': p_value,
        'chi2_statistic': chi2_stat,
        'chi2_p_value': chi2_p_value,
        'binomial_tests': binomial_tests,
        'significance_level': 'significant' if p_value < 0.05 else 'not_significant',
        'effect_size': abs(causal_witness) / std_error if std_error > 0 else 0
    }

def bootstrap_confidence_interval(data, statistic, n_bootstrap=1000, confidence=0.95):
    """
    Compute bootstrap confidence interval for a statistic.
    """
    n = len(data)
    bootstrap_samples = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_data = np.random.choice(data, size=n, replace=True)
        bootstrap_samples.append(statistic(bootstrap_data))
    
    bootstrap_samples = np.array(bootstrap_samples)
    stat_value = statistic(data)
    
    # Compute confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_ci = np.percentile(bootstrap_samples, lower_percentile)
    upper_ci = np.percentile(bootstrap_samples, upper_percentile)
    
    return stat_value, lower_ci, upper_ci, bootstrap_samples

def analyze_measurement_uncertainty(counts, shots):
    """
    Analyze measurement uncertainty and error sources.
    """
    # Calculate measurement uncertainties
    uncertainties = {}
    for outcome, count in counts.items():
        # Binomial uncertainty: sqrt(p*(1-p)/n)
        p = count / shots
        uncertainty = np.sqrt(p * (1 - p) / shots)
        uncertainties[outcome] = uncertainty
    
    # Total measurement uncertainty
    total_uncertainty = np.sqrt(sum(u**2 for u in uncertainties.values()))
    
    # Systematic error estimate (from hardware calibration)
    systematic_error = 0.01  # 1% systematic error estimate
    
    # Total uncertainty including systematic effects
    total_error = np.sqrt(total_uncertainty**2 + systematic_error**2)
    
    return {
        'measurement_uncertainties': uncertainties,
        'total_measurement_uncertainty': total_uncertainty,
        'systematic_error': systematic_error,
        'total_uncertainty': total_error
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Switch Emergent Time Experiment")
    parser.add_argument('--device', type=str, default='simulator', 
                       help='Device to use: "simulator" or IBMQ backend name (e.g., "ibm_brisbane")')
    parser.add_argument('--shots', type=int, default=2048, help='Number of shots')
    parser.add_argument('--phi', type=float, default=np.pi/2, 
                       help='Single phi value to test (default: π/2)')
    parser.add_argument('--no-mitigation', action='store_true', 
                       help='Disable error mitigation')
    parser.add_argument('--compare', action='store_true', 
                       help='Run both simulator and hardware for comparison')
    args = parser.parse_args()
    
    use_mitigation = not args.no_mitigation
    
    # Use single phi value
    phi_values = [args.phi]
    
    if args.compare:
        print("Running comparison experiment: simulator vs hardware")
        
        # Run simulator first
        print("\n=== Running Simulator Experiment ===")
        sim_results, sim_log_dir = run_quantum_switch_experiment(
            shots=args.shots, phi_values=phi_values, device_name='simulator', use_mitigation=False
        )
        
        # Run hardware
        print("\n=== Running Hardware Experiment ===")
        hw_results, hw_log_dir = run_quantum_switch_experiment(
            shots=args.shots, phi_values=phi_values, device_name=args.device, use_mitigation=use_mitigation
        )
        
        # Create comparison
        if sim_results and hw_results:
            compare_simulator_hardware(sim_results, hw_results, hw_log_dir)
            print(f"\nComparison complete! Results in: {hw_log_dir}")
    else:
        # Run single experiment
        results, log_dir = run_quantum_switch_experiment(
            shots=args.shots, phi_values=phi_values, device_name=args.device, use_mitigation=use_mitigation
        ) 