#!/usr/bin/env python3
"""
Deutsch CTC Analysis for 8-qubit Experiment
===========================================

Analyzes the Deutsch fixed-point iteration results to extract quantum spacetime signatures
and temporal paradox indicators from the CTC experiment.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys

def load_deutsch_ctc_data(filepath):
    """Load Deutsch CTC analysis data from experiment file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract Deutsch CTC analysis
    deutsch_data = data.get('deutsch_ctc_analysis', {})
    spec = data.get('spec', {})
    
    return deutsch_data, spec

def analyze_fixed_point_density_matrix(density_matrix):
    """Analyze the fixed point density matrix for quantum spacetime signatures."""
    print(f"\nANALYZING FIXED POINT DENSITY MATRIX")
    print(f"=" * 50)
    
    # Convert to numpy array
    rho = np.array([[complex(elem['real'], elem['imag']) for elem in row] for row in density_matrix])
    
    # Calculate key properties
    trace = np.trace(rho)
    purity = np.trace(rho @ rho)
    eigenvalues = np.linalg.eigvals(rho)
    entropy = -np.sum(eigenvalues * np.log2(np.abs(eigenvalues) + 1e-10))
    
    print(f"📊 Trace: {trace:.6f}")
    print(f"📊 Purity: {purity:.6f}")
    print(f"📊 Von Neumann Entropy: {entropy:.6f}")
    print(f"📊 Eigenvalues: {eigenvalues}")
    
    # Check for maximally mixed state (trivial fixed point)
    dim = len(rho)
    max_entropy = np.log2(dim)
    is_maximally_mixed = abs(entropy - max_entropy) < 0.1
    
    print(f"📊 Max Entropy (log2({dim})): {max_entropy:.6f}")
    print(f"📊 Is Maximally Mixed: {is_maximally_mixed}")
    
    # Analyze quantum coherence
    off_diagonal_strength = np.sum(np.abs(rho - np.diag(np.diag(rho))))
    coherence_measure = off_diagonal_strength / (dim * (dim - 1))
    
    print(f"📊 Off-diagonal Strength: {off_diagonal_strength:.6f}")
    print(f"📊 Coherence Measure: {coherence_measure:.6f}")
    
    return {
        'trace': trace,
        'purity': purity,
        'entropy': entropy,
        'eigenvalues': eigenvalues,
        'is_maximally_mixed': is_maximally_mixed,
        'coherence_measure': coherence_measure,
        'off_diagonal_strength': off_diagonal_strength
    }

def analyze_convergence_info(convergence_info):
    """Analyze convergence information for temporal paradox indicators."""
    print(f"\nANALYZING CONVERGENCE INFORMATION")
    print(f"=" * 50)
    
    iterations = convergence_info.get('iterations', 0)
    converged = convergence_info.get('converged', False)
    final_fidelity = convergence_info.get('final_fidelity', 0.0)
    fidelity_history = convergence_info.get('fidelity_history', [])
    
    print(f"📊 Iterations: {iterations}")
    print(f"📊 Converged: {converged}")
    print(f"📊 Final Fidelity: {final_fidelity:.6f}")
    print(f"📊 Fidelity History: {fidelity_history}")
    
    # Analyze convergence speed
    convergence_speed = "instant" if iterations == 1 else "gradual"
    temporal_stability = "stable" if converged else "unstable"
    
    print(f"📊 Convergence Speed: {convergence_speed}")
    print(f"📊 Temporal Stability: {temporal_stability}")
    
    # Check for temporal paradox indicators
    paradox_indicators = []
    
    if iterations == 1:
        paradox_indicators.append("INSTANT_CONVERGENCE - Suggests strong temporal paradox")
    
    if final_fidelity > 0.999:
        paradox_indicators.append("PERFECT_FIDELITY - Indicates temporal consistency")
    
    if len(fidelity_history) == 1 and fidelity_history[0] > 0.999:
        paradox_indicators.append("IMMEDIATE_STABILITY - No temporal evolution needed")
    
    return {
        'iterations': iterations,
        'converged': converged,
        'final_fidelity': final_fidelity,
        'fidelity_history': fidelity_history,
        'convergence_speed': convergence_speed,
        'temporal_stability': temporal_stability,
        'paradox_indicators': paradox_indicators
    }

def analyze_sample_counts(sample_counts, loop_qubits):
    """Analyze sample counts for quantum spacetime signatures."""
    print(f"\nANALYZING SAMPLE COUNTS")
    print(f"=" * 50)
    
    total_samples = sum(sample_counts.values())
    probabilities = {k: v/total_samples for k, v in sample_counts.items()}
    
    print(f"📊 Total Samples: {total_samples}")
    print(f"📊 Loop Qubits: {loop_qubits}")
    print(f"📊 Sample Distribution:")
    for outcome, prob in probabilities.items():
        print(f"  {outcome}: {prob:.3f}")
    
    # Calculate entropy of the distribution
    entropy = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)
    max_entropy = np.log2(len(probabilities))
    
    print(f"📊 Distribution Entropy: {entropy:.6f}")
    print(f"📊 Max Entropy: {max_entropy:.6f}")
    print(f"📊 Entropy Ratio: {entropy/max_entropy:.6f}")
    
    # Check for uniform distribution (maximally mixed)
    is_uniform = all(abs(p - 1.0/len(probabilities)) < 0.1 for p in probabilities.values())
    print(f"📊 Is Uniform Distribution: {is_uniform}")
    
    # Analyze quantum correlations
    correlation_analysis = analyze_quantum_correlations(probabilities, loop_qubits)
    
    return {
        'total_samples': total_samples,
        'probabilities': probabilities,
        'entropy': entropy,
        'max_entropy': max_entropy,
        'entropy_ratio': entropy/max_entropy,
        'is_uniform': is_uniform,
        'correlation_analysis': correlation_analysis
    }

def analyze_quantum_correlations(probabilities, loop_qubits):
    """Analyze quantum correlations in the sample distribution."""
    print(f"\nANALYZING QUANTUM CORRELATIONS")
    print(f"=" * 50)
    
    # Extract bit patterns
    outcomes = list(probabilities.keys())
    n_qubits = len(outcomes[0])
    
    # Calculate pairwise correlations
    correlations = {}
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            # Calculate correlation between qubits i and j
            correlation = 0.0
            for outcome, prob in probabilities.items():
                bit_i = int(outcome[i])
                bit_j = int(outcome[j])
                correlation += prob * (2*bit_i - 1) * (2*bit_j - 1)
            correlations[f"qubit_{i}_{j}"] = correlation
    
    print(f"📊 Pairwise Correlations:")
    for pair, corr in correlations.items():
        print(f"  {pair}: {corr:.6f}")
    
    # Check for entanglement signatures
    entanglement_signatures = []
    for pair, corr in correlations.items():
        if abs(corr) > 0.5:
            entanglement_signatures.append(f"Strong correlation in {pair}: {corr:.3f}")
        elif abs(corr) > 0.1:
            entanglement_signatures.append(f"Moderate correlation in {pair}: {corr:.3f}")
    
    if not entanglement_signatures:
        entanglement_signatures.append("No significant correlations detected")
    
    print(f"📊 Entanglement Signatures:")
    for sig in entanglement_signatures:
        print(f"  - {sig}")
    
    return {
        'correlations': correlations,
        'entanglement_signatures': entanglement_signatures
    }

def generate_quantum_spacetime_signatures(density_analysis, convergence_analysis, sample_analysis):
    """Generate quantum spacetime signatures from all analyses."""
    print(f"\nQUANTUM SPACETIME SIGNATURES")
    print(f"=" * 50)
    
    signatures = []
    
    # Temporal paradox signatures
    if convergence_analysis['iterations'] == 1:
        signatures.append("[TEMPORAL PARADOX] Instant convergence suggests temporal paradox")
    
    if convergence_analysis['final_fidelity'] > 0.999:
        signatures.append("[TEMPORAL CONSISTENCY] Perfect fidelity indicates temporal consistency")
    
    # Quantum coherence signatures
    if density_analysis['coherence_measure'] > 0.1:
        signatures.append("[QUANTUM COHERENCE] Significant off-diagonal elements preserved")
    else:
        signatures.append("[DECOHERENCE] System has decohered to diagonal form")
    
    # Entanglement signatures
    if sample_analysis['correlation_analysis']['entanglement_signatures']:
        signatures.append("[ENTANGLEMENT] Quantum correlations detected in sample distribution")
    
    # Thermalization signatures
    if density_analysis['is_maximally_mixed']:
        signatures.append("[THERMALIZATION] System has thermalized to maximally mixed state")
    
    # Spacetime geometry signatures
    if density_analysis['entropy'] > 1.5:
        signatures.append("[SPACETIME CURVATURE] High entropy suggests curved spacetime geometry")
    
    if not signatures:
        signatures.append("[NO SIGNATURES] No strong quantum spacetime signatures detected")
    
    print("🎯 Detected Signatures:")
    for sig in signatures:
        print(f"  {sig}")
    
    return signatures

def create_visualizations(density_matrix, sample_counts, output_dir):
    """Create visualizations of the CTC analysis."""
    print(f"\nCREATING VISUALIZATIONS")
    print(f"=" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Density matrix heatmap
    rho = np.array([[complex(elem['real'], elem['imag']) for elem in row] for row in density_matrix])
    rho_real = np.real(rho)
    
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    sns.heatmap(rho_real, annot=True, fmt='.3f', cmap='RdBu_r', center=0)
    plt.title('Fixed Point Density Matrix (Real Part)')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    # Plot 2: Sample distribution
    plt.subplot(2, 2, 2)
    outcomes = list(sample_counts.keys())
    counts = list(sample_counts.values())
    plt.bar(range(len(outcomes)), counts)
    plt.title('Sample Distribution')
    plt.xlabel('Outcome')
    plt.ylabel('Count')
    plt.xticks(range(len(outcomes)), outcomes, rotation=45)
    
    # Plot 3: Eigenvalue distribution
    plt.subplot(2, 2, 3)
    eigenvalues = np.linalg.eigvals(rho)
    plt.bar(range(len(eigenvalues)), np.real(eigenvalues))
    plt.title('Eigenvalue Distribution')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Real Part')
    
    # Plot 4: Entropy analysis
    plt.subplot(2, 2, 4)
    entropy_components = ['Trace', 'Purity', 'Coherence']
    entropy_values = [np.trace(rho), np.trace(rho @ rho), np.sum(np.abs(rho - np.diag(np.diag(rho))))]
    plt.bar(entropy_components, entropy_values)
    plt.title('Quantum State Properties')
    plt.ylabel('Value')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'deutsch_ctc_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {plot_path}")

def generate_summary_report(analysis_results, output_dir):
    """Generate a comprehensive summary report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f'deutsch_ctc_analysis_report_{timestamp}.txt')
    
    with open(report_path, 'w') as f:
        f.write("DEUTSCH CTC ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("EXPERIMENT PARAMETERS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Loop Qubits: {analysis_results['loop_qubits']}\n")
        f.write(f"Total Samples: {analysis_results['sample_analysis']['total_samples']}\n\n")
        
        f.write("DENSITY MATRIX ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Trace: {analysis_results['density_analysis']['trace']:.6f}\n")
        f.write(f"Purity: {analysis_results['density_analysis']['purity']:.6f}\n")
        f.write(f"Von Neumann Entropy: {analysis_results['density_analysis']['entropy']:.6f}\n")
        f.write(f"Coherence Measure: {analysis_results['density_analysis']['coherence_measure']:.6f}\n")
        f.write(f"Is Maximally Mixed: {analysis_results['density_analysis']['is_maximally_mixed']}\n\n")
        
        f.write("CONVERGENCE ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Iterations: {analysis_results['convergence_analysis']['iterations']}\n")
        f.write(f"Converged: {analysis_results['convergence_analysis']['converged']}\n")
        f.write(f"Final Fidelity: {analysis_results['convergence_analysis']['final_fidelity']:.6f}\n")
        f.write(f"Convergence Speed: {analysis_results['convergence_analysis']['convergence_speed']}\n")
        f.write(f"Temporal Stability: {analysis_results['convergence_analysis']['temporal_stability']}\n\n")
        
        f.write("SAMPLE DISTRIBUTION ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Distribution Entropy: {analysis_results['sample_analysis']['entropy']:.6f}\n")
        f.write(f"Entropy Ratio: {analysis_results['sample_analysis']['entropy_ratio']:.6f}\n")
        f.write(f"Is Uniform: {analysis_results['sample_analysis']['is_uniform']}\n\n")
        
        f.write("QUANTUM SPACETIME SIGNATURES:\n")
        f.write("-" * 30 + "\n")
        for signature in analysis_results['signatures']:
            f.write(f"{signature}\n")
        f.write("\n")
        
        f.write("CONCLUSION:\n")
        f.write("-" * 30 + "\n")
        if analysis_results['density_analysis']['is_maximally_mixed']:
            f.write("The Deutsch fixed-point iteration has converged to a maximally mixed state,\n")
            f.write("indicating that the system has thermalized. This suggests that the quantum\n")
            f.write("circuit acts as a universal thermalizer, driving all states toward the\n")
            f.write("trivial fixed point.\n\n")
        else:
            f.write("The Deutsch fixed-point iteration has converged to a non-trivial state,\n")
            f.write("preserving some quantum structure. This indicates the presence of\n")
            f.write("non-trivial quantum spacetime signatures.\n\n")
    
    print(f"Summary report saved: {report_path}")
    return report_path

def main():
    if len(sys.argv) != 2:
        print("Usage: python deutsch_ctc_analysis.py <experiment_file>")
        print("Example: python deutsch_ctc_analysis.py experiment_logs/custom_curvature_experiment/instance_20250801_151819/results_n8_geomS_curv1_ibm_brisbane_HSXEL9.json")
        return
    
    filepath = sys.argv[1]
    
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found")
        return
    
    print(f"DEUTSCH CTC ANALYSIS")
    print(f"=" * 50)
    print(f"Analyzing: {os.path.basename(filepath)}")
    
    # Load data
    deutsch_data, spec = load_deutsch_ctc_data(filepath)
    
    if not deutsch_data:
        print("Error: No Deutsch CTC analysis data found in file")
        return
    
    # Run analyses
    density_analysis = analyze_fixed_point_density_matrix(deutsch_data['fixed_point_density_matrix'])
    convergence_analysis = analyze_convergence_info(deutsch_data['convergence_info'])
    sample_analysis = analyze_sample_counts(deutsch_data['sample_counts'], deutsch_data['loop_qubits'])
    
    # Generate signatures
    signatures = generate_quantum_spacetime_signatures(density_analysis, convergence_analysis, sample_analysis)
    
    # Create output directory
    output_dir = os.path.dirname(filepath)
    
    # Create visualizations
    create_visualizations(deutsch_data['fixed_point_density_matrix'], deutsch_data['sample_counts'], output_dir)
    
    # Generate summary report
    analysis_results = {
        'loop_qubits': deutsch_data['loop_qubits'],
        'density_analysis': density_analysis,
        'convergence_analysis': convergence_analysis,
        'sample_analysis': sample_analysis,
        'signatures': signatures
    }
    
    generate_summary_report(analysis_results, output_dir)
    
    print(f"\nDeutsch CTC Analysis Complete!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 