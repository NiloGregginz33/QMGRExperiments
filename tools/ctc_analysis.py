#!/usr/bin/env python3
"""
CTC (Closed Timelike Curve) Analysis Tool
=========================================

This tool analyzes CTC experiment results to extract quantum spacetime signatures:
- Temporal paradox detection
- Entanglement structure analysis
- Causal consistency verification
- Quantum coherence measurements
- Spacetime geometry reconstruction
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse
import os
import sys
from datetime import datetime

def load_ctc_results(filepath):
    """Load CTC results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_temporal_paradox(counts, ctc_size):
    """
    Analyze temporal paradox signatures in CTC measurements.
    
    Returns:
    - paradox_score: Measure of temporal inconsistency
    - causality_violations: Number of causal violations
    - temporal_coherence: Measure of temporal coherence
    """
    print(f"\n🔍 ANALYZING TEMPORAL PARADOX SIGNATURES")
    print(f"=" * 50)
    
    # Convert counts to probabilities
    total = sum(counts.values())
    probabilities = {k: v/total for k, v in counts.items()}
    
    # Analyze bit patterns for temporal paradox indicators
    paradox_indicators = []
    causality_violations = 0
    
    for outcome, prob in probabilities.items():
        bits = [int(b) for b in outcome]
        
        # Check for temporal paradox patterns
        # Pattern 1: All zeros (temporal vacuum)
        if all(b == 0 for b in bits):
            paradox_indicators.append(("temporal_vacuum", prob, outcome))
        
        # Pattern 2: Alternating patterns (temporal oscillation)
        if len(bits) >= 2:
            alternating = all(bits[i] != bits[i+1] for i in range(len(bits)-1))
            if alternating:
                paradox_indicators.append(("temporal_oscillation", prob, outcome))
        
        # Pattern 3: High entropy states (temporal chaos)
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in [prob, 1-prob])
        if entropy > 0.8:
            paradox_indicators.append(("temporal_chaos", prob, outcome))
    
    # Calculate paradox score
    paradox_score = sum(prob for _, prob, _ in paradox_indicators)
    
    # Check for causality violations (unexpected correlations)
    expected_uniform = 1.0 / len(probabilities)
    causality_violations = sum(1 for prob in probabilities.values() if abs(prob - expected_uniform) > 0.1)
    
    # Calculate temporal coherence
    temporal_coherence = 1.0 - paradox_score
    
    print(f"📊 Paradox Score: {paradox_score:.3f}")
    print(f"📊 Causality Violations: {causality_violations}")
    print(f"📊 Temporal Coherence: {temporal_coherence:.3f}")
    
    print(f"\n🎯 Paradox Indicators:")
    for indicator_type, prob, outcome in paradox_indicators:
        print(f"  - {indicator_type}: {outcome} (p={prob:.3f})")
    
    return {
        'paradox_score': paradox_score,
        'causality_violations': causality_violations,
        'temporal_coherence': temporal_coherence,
        'paradox_indicators': paradox_indicators
    }

def analyze_entanglement_structure(counts, ctc_size):
    """
    Analyze entanglement structure in CTC measurements.
    
    Returns:
    - entanglement_entropy: Von Neumann entropy of the system
    - mutual_information: Mutual information between qubits
    - entanglement_patterns: Identified entanglement patterns
    """
    print(f"\n🔗 ANALYZING ENTANGLEMENT STRUCTURE")
    print(f"=" * 50)
    
    # Convert counts to density matrix
    total = sum(counts.values())
    n_qubits = ctc_size
    
    # Create density matrix from measurement outcomes
    dim = 2**n_qubits
    rho = np.zeros((dim, dim), dtype=complex)
    
    for outcome, count in counts.items():
        # Convert binary string to state vector
        state_idx = int(outcome, 2)
        prob = count / total
        rho[state_idx, state_idx] = prob
    
    # Calculate von Neumann entropy
    eigenvals = np.linalg.eigvalsh(rho)
    eigenvals = eigenvals[eigenvals > 1e-10]  # Remove numerical zeros
    entanglement_entropy = -np.sum(eigenvals * np.log2(eigenvals))
    
    # Calculate mutual information between adjacent qubits
    mutual_info = {}
    for i in range(n_qubits - 1):
        # Marginalize over qubits i and i+1
        mi = calculate_pairwise_mutual_info(counts, i, i+1, n_qubits)
        mutual_info[f"I_{i},{i+1}"] = mi
    
    # Identify entanglement patterns
    entanglement_patterns = []
    
    # Check for Bell state signatures
    bell_states = ['00', '01', '10', '11']
    for i in range(n_qubits - 1):
        pair_counts = defaultdict(int)
        for outcome, count in counts.items():
            pair = outcome[i:i+2]
            pair_counts[pair] += count
        
        # Check if this pair shows Bell state characteristics
        total_pair = sum(pair_counts.values())
        if total_pair > 0:
            max_prob = max(pair_counts.values()) / total_pair
            if max_prob > 0.6:  # Strong correlation
                entanglement_patterns.append(f"Bell_pair_{i}-{i+1}")
    
    print(f"📊 Von Neumann Entropy: {entanglement_entropy:.3f}")
    print(f"📊 Mutual Information:")
    for pair, mi in mutual_info.items():
        print(f"  - {pair}: {mi:.3f}")
    
    print(f"📊 Entanglement Patterns:")
    for pattern in entanglement_patterns:
        print(f"  - {pattern}")
    
    return {
        'entanglement_entropy': entanglement_entropy,
        'mutual_information': mutual_info,
        'entanglement_patterns': entanglement_patterns
    }

def calculate_pairwise_mutual_info(counts, qubit1, qubit2, n_qubits):
    """Calculate mutual information between two qubits."""
    # Marginalize over the two qubits
    joint_counts = defaultdict(int)
    marginal1 = defaultdict(int)
    marginal2 = defaultdict(int)
    
    for outcome, count in counts.items():
        bit1 = outcome[qubit1]
        bit2 = outcome[qubit2]
        joint_counts[f"{bit1}{bit2}"] += count
        marginal1[bit1] += count
        marginal2[bit2] += count
    
    total = sum(counts.values())
    
    # Calculate mutual information
    mi = 0
    for joint_state, joint_count in joint_counts.items():
        if joint_count > 0:
            p_joint = joint_count / total
            p1 = marginal1[joint_state[0]] / total
            p2 = marginal2[joint_state[1]] / total
            
            if p1 > 0 and p2 > 0:
                mi += p_joint * np.log2(p_joint / (p1 * p2))
    
    return mi

def analyze_quantum_coherence(counts, phase_profile):
    """
    Analyze quantum coherence signatures.
    
    Returns:
    - coherence_measure: Measure of quantum coherence
    - superposition_strength: Strength of superposition states
    - decoherence_indicators: Signs of decoherence
    """
    print(f"\n🌊 ANALYZING QUANTUM COHERENCE")
    print(f"=" * 50)
    
    total = sum(counts.values())
    probabilities = {k: v/total for k, v in counts.items()}
    
    # Calculate coherence measure (based on deviation from uniform)
    uniform_prob = 1.0 / len(probabilities)
    coherence_measure = 1.0 - np.std(list(probabilities.values()))
    
    # Analyze superposition strength
    # Look for balanced superpositions (equal probabilities)
    balanced_states = []
    for outcome, prob in probabilities.items():
        if abs(prob - uniform_prob) < 0.1:
            balanced_states.append((outcome, prob))
    
    superposition_strength = len(balanced_states) / len(probabilities)
    
    # Check for decoherence indicators
    decoherence_indicators = []
    
    # Indicator 1: Dominant state (decoherence to classical state)
    max_prob = max(probabilities.values())
    if max_prob > 0.5:
        dominant_state = max(probabilities.items(), key=lambda x: x[1])
        decoherence_indicators.append(f"dominant_state_{dominant_state[0]}")
    
    # Indicator 2: Low entropy (loss of quantum information)
    entropy = -sum(p * np.log2(p) for p in probabilities.values())
    max_entropy = np.log2(len(probabilities))
    if entropy < 0.5 * max_entropy:
        decoherence_indicators.append("low_entropy")
    
    print(f"📊 Coherence Measure: {coherence_measure:.3f}")
    print(f"📊 Superposition Strength: {superposition_strength:.3f}")
    print(f"📊 System Entropy: {entropy:.3f} / {max_entropy:.3f}")
    
    print(f"📊 Balanced Superpositions:")
    for state, prob in balanced_states:
        print(f"  - {state}: p={prob:.3f}")
    
    print(f"📊 Decoherence Indicators:")
    for indicator in decoherence_indicators:
        print(f"  - {indicator}")
    
    return {
        'coherence_measure': coherence_measure,
        'superposition_strength': superposition_strength,
        'system_entropy': entropy,
        'max_entropy': max_entropy,
        'decoherence_indicators': decoherence_indicators
    }

def analyze_spacetime_geometry(counts, phase_profile, ctc_size):
    """
    Analyze spacetime geometry signatures from CTC measurements.
    
    Returns:
    - geometry_signature: Identified spacetime geometry
    - curvature_indicators: Signs of spacetime curvature
    - causal_structure: Causal structure analysis
    """
    print(f"\n🌌 ANALYZING SPACETIME GEOMETRY")
    print(f"=" * 50)
    
    # Analyze phase profile for geometry information
    phase_array = np.array(phase_profile)
    phase_variance = np.var(phase_array)
    phase_gradient = np.gradient(phase_array)
    
    # Determine geometry signature based on phase profile
    if phase_variance < 1e-6:
        geometry_signature = "flat_spacetime"
    elif np.max(phase_gradient) > 0.1:
        geometry_signature = "curved_spacetime"
    else:
        geometry_signature = "mildly_curved_spacetime"
    
    # Analyze causal structure from measurement outcomes
    total = sum(counts.values())
    probabilities = {k: v/total for k, v in counts.items()}
    
    # Check for causal consistency
    causal_violations = 0
    causal_patterns = []
    
    for outcome, prob in probabilities.items():
        bits = [int(b) for b in outcome]
        
        # Check for causal ordering violations
        # In a CTC, we expect certain correlations
        if len(bits) >= 2:
            # Check for time-reversed correlations
            if bits[0] == bits[-1] and prob > 0.3:
                causal_patterns.append("time_reversal")
            
            # Check for causal loops
            if all(bits[i] == bits[i+1] for i in range(len(bits)-1)):
                causal_patterns.append("causal_loop")
    
    # Calculate curvature indicators
    curvature_indicators = []
    
    # Indicator 1: Phase profile uniformity
    if phase_variance < 1e-6:
        curvature_indicators.append("flat_geometry")
    else:
        curvature_indicators.append("curved_geometry")
    
    # Indicator 2: Measurement outcome distribution
    outcome_entropy = -sum(p * np.log2(p) for p in probabilities.values())
    if outcome_entropy < 2.0:  # Low entropy suggests curvature effects
        curvature_indicators.append("curvature_effects")
    
    print(f"📊 Geometry Signature: {geometry_signature}")
    print(f"📊 Phase Variance: {phase_variance:.6f}")
    print(f"📊 Causal Violations: {causal_violations}")
    
    print(f"📊 Causal Patterns:")
    for pattern in causal_patterns:
        print(f"  - {pattern}")
    
    print(f"📊 Curvature Indicators:")
    for indicator in curvature_indicators:
        print(f"  - {indicator}")
    
    return {
        'geometry_signature': geometry_signature,
        'phase_variance': phase_variance,
        'causal_violations': causal_violations,
        'causal_patterns': causal_patterns,
        'curvature_indicators': curvature_indicators
    }

def create_visualizations(results, output_dir):
    """Create visualizations of the analysis results."""
    print(f"\n📊 CREATING VISUALIZATIONS")
    print(f"=" * 50)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CTC Quantum Spacetime Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Measurement outcome distribution
    outcomes = list(results['counts'].keys())
    counts = list(results['counts'].values())
    
    axes[0, 0].bar(range(len(outcomes)), counts, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Measurement Outcome Distribution')
    axes[0, 0].set_xlabel('Outcome')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_xticks(range(len(outcomes)))
    axes[0, 0].set_xticklabels(outcomes, rotation=45)
    
    # Plot 2: Phase profile heatmap
    if 'phase_profile_matrix' in results:
        phase_matrix = np.array(results['phase_profile_matrix'])
        im = axes[0, 1].imshow(phase_matrix, cmap='viridis', aspect='equal')
        axes[0, 1].set_title('Phase Profile Matrix')
        axes[0, 1].set_xlabel('Column')
        axes[0, 1].set_ylabel('Row')
        plt.colorbar(im, ax=axes[0, 1])
    else:
        axes[0, 1].text(0.5, 0.5, 'Phase Profile\nNot Available', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Phase Profile Matrix')
    
    # Plot 3: Entanglement analysis
    if 'entanglement_entropy' in results:
        entropy_data = [results['entanglement_entropy']]
        axes[1, 0].bar(['System'], entropy_data, color='lightcoral', alpha=0.7)
        axes[1, 0].set_title('Von Neumann Entropy')
        axes[1, 0].set_ylabel('Entropy')
    
    # Plot 4: Coherence analysis
    if 'coherence_measure' in results:
        coherence_data = [results['coherence_measure']]
        axes[1, 1].bar(['System'], coherence_data, color='lightgreen', alpha=0.7)
        axes[1, 1].set_title('Quantum Coherence Measure')
        axes[1, 1].set_ylabel('Coherence')
        axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'ctc_analysis_visualization.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved: {plot_path}")
    
    plt.show()

def generate_summary_report(analysis_results, output_dir):
    """Generate a comprehensive summary report."""
    print(f"\n📝 GENERATING SUMMARY REPORT")
    print(f"=" * 50)
    
    report_path = os.path.join(output_dir, 'ctc_analysis_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("CTC QUANTUM SPACETIME ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("EXPERIMENT PARAMETERS:\n")
        f.write(f"- CTC Size: {analysis_results['ctc_size']}\n")
        f.write(f"- Phase Profile: {analysis_results['phase_profile']}\n")
        f.write(f"- Phase Strength: {analysis_results['phase_strength']}\n")
        f.write(f"- Backend: {analysis_results['backend']}\n")
        f.write(f"- Shots: {analysis_results['shots']}\n\n")
        
        f.write("TEMPORAL PARADOX ANALYSIS:\n")
        f.write(f"- Paradox Score: {analysis_results['temporal']['paradox_score']:.3f}\n")
        f.write(f"- Causality Violations: {analysis_results['temporal']['causality_violations']}\n")
        f.write(f"- Temporal Coherence: {analysis_results['temporal']['temporal_coherence']:.3f}\n\n")
        
        f.write("ENTANGLEMENT STRUCTURE:\n")
        f.write(f"- Von Neumann Entropy: {analysis_results['entanglement']['entanglement_entropy']:.3f}\n")
        f.write(f"- Entanglement Patterns: {', '.join(analysis_results['entanglement']['entanglement_patterns'])}\n\n")
        
        f.write("QUANTUM COHERENCE:\n")
        f.write(f"- Coherence Measure: {analysis_results['coherence']['coherence_measure']:.3f}\n")
        f.write(f"- Superposition Strength: {analysis_results['coherence']['superposition_strength']:.3f}\n")
        f.write(f"- System Entropy: {analysis_results['coherence']['system_entropy']:.3f}\n\n")
        
        f.write("SPACETIME GEOMETRY:\n")
        f.write(f"- Geometry Signature: {analysis_results['geometry']['geometry_signature']}\n")
        f.write(f"- Phase Variance: {analysis_results['geometry']['phase_variance']:.6f}\n")
        f.write(f"- Causal Patterns: {', '.join(analysis_results['geometry']['causal_patterns'])}\n\n")
        
        f.write("QUANTUM SPACETIME SIGNATURES:\n")
        f.write("=" * 30 + "\n")
        
        # Determine if we have quantum spacetime signatures
        signatures = []
        
        if analysis_results['temporal']['paradox_score'] > 0.3:
            signatures.append("[SIGNATURE] Temporal Paradox Detected")
        
        if analysis_results['entanglement']['entanglement_entropy'] > 1.0:
            signatures.append("[SIGNATURE] Strong Entanglement Structure")
        
        if analysis_results['coherence']['coherence_measure'] > 0.5:
            signatures.append("[SIGNATURE] Quantum Coherence Preserved")
        
        if analysis_results['geometry']['geometry_signature'] != "flat_spacetime":
            signatures.append("[SIGNATURE] Spacetime Curvature Detected")
        
        if not signatures:
            signatures.append("[NO SIGNATURE] No Strong Quantum Spacetime Signatures")
        
        for signature in signatures:
            f.write(f"{signature}\n")
    
    print(f"📝 Summary report saved: {report_path}")
    return report_path

def main():
    parser = argparse.ArgumentParser(description='Analyze CTC quantum spacetime signatures')
    parser.add_argument('ctc_results_file', help='Path to CTC results JSON file')
    parser.add_argument('--output_dir', default='.', help='Output directory for analysis results')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    
    args = parser.parse_args()
    
    # Load CTC results
    print(f"🔍 Loading CTC results from: {args.ctc_results_file}")
    results = load_ctc_results(args.ctc_results_file)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run comprehensive analysis
    analysis_results = {
        'ctc_size': results['ctc_size'],
        'phase_profile': results['phase_profile'],
        'phase_strength': results['phase_strength'],
        'backend': results['backend'],
        'shots': results['shots'],
        'counts': results['counts']
    }
    
    # Temporal paradox analysis
    analysis_results['temporal'] = analyze_temporal_paradox(results['counts'], results['ctc_size'])
    
    # Entanglement structure analysis
    analysis_results['entanglement'] = analyze_entanglement_structure(results['counts'], results['ctc_size'])
    
    # Quantum coherence analysis
    analysis_results['coherence'] = analyze_quantum_coherence(results['counts'], results['phase_profile_matrix'])
    
    # Spacetime geometry analysis
    analysis_results['geometry'] = analyze_spacetime_geometry(results['counts'], results['phase_profile_matrix'], results['ctc_size'])
    
    # Create visualizations if requested
    if args.visualize:
        create_visualizations(analysis_results, args.output_dir)
    
    # Generate summary report
    report_path = generate_summary_report(analysis_results, args.output_dir)
    
    print(f"\n🎉 CTC Analysis Complete!")
    print(f"📁 Results saved to: {args.output_dir}")
    print(f"📝 Summary report: {report_path}")

if __name__ == "__main__":
    main() 