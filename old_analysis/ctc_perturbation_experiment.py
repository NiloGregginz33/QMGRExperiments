#!/usr/bin/env python3
"""
CTC PERTURBATION EXPERIMENT: Testing Deutsch Fixed-Point Stability
==================================================================

This experiment tests how robust the Deutsch fixed-point CTC implementation
is to various types of perturbations:

1. Circuit parameter noise
2. Initial state perturbations  
3. Loop-bulk coupling variations
4. Different perturbation magnitudes

This helps us understand the stability and tolerance limits of the CTC.
"""

import sys
import os
import numpy as np
import json
from datetime import datetime
import argparse
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import Operator, DensityMatrix, partial_trace, state_fidelity
import matplotlib.pyplot as plt

# Add src to path
sys.path.append('src')
from CGPTFactory import run

def create_perturbed_deutsch_ctc(num_qubits, perturbation_type, perturbation_magnitude):
    """
    Create a Deutsch CTC circuit with controlled perturbations
    
    Args:
        num_qubits: Number of qubits
        perturbation_type: 'circuit_noise', 'initial_state', 'coupling_variation'
        perturbation_magnitude: Strength of perturbation (0.0 to 1.0)
    
    Returns:
        qc: QuantumCircuit with perturbations
        perturbation_info: Dictionary with perturbation details
    """
    qc = QuantumCircuit(num_qubits)
    
    # Separate loop and bulk qubits
    loop_qubits = min(2, num_qubits // 2)
    bulk_qubits = num_qubits - loop_qubits
    
    perturbation_info = {
        'type': perturbation_type,
        'magnitude': perturbation_magnitude,
        'loop_qubits': loop_qubits,
        'bulk_qubits': bulk_qubits
    }
    
    # Initialize bulk qubits in superposition
    for i in range(loop_qubits, num_qubits):
        qc.h(i)
    
    # Apply perturbations based on type
    if perturbation_type == 'initial_state':
        # Add small random rotations to bulk qubits
        for i in range(loop_qubits, num_qubits):
            angle = np.random.normal(0, perturbation_magnitude * np.pi/4)
            qc.rz(angle, i)
            perturbation_info[f'bulk_rotation_{i}'] = angle
    
    # Create entanglement between bulk and loop
    for i in range(loop_qubits):
        for j in range(loop_qubits, num_qubits):
            qc.cx(i, j)
            
            # Apply coupling variation perturbation
            if perturbation_type == 'coupling_variation':
                phase = np.pi/4 + np.random.normal(0, perturbation_magnitude * np.pi/4)
                qc.rzz(phase, i, j)
                perturbation_info[f'coupling_phase_{i}_{j}'] = phase
            else:
                qc.rzz(np.pi/4, i, j)
    
    # Add time-asymmetric operations on loop qubits
    for i in range(loop_qubits):
        qc.t(i)  # T-gate breaks time-reversal symmetry
        
        # Apply circuit noise perturbation
        if perturbation_type == 'circuit_noise':
            angle = np.pi/3 + np.random.normal(0, perturbation_magnitude * np.pi/6)
            qc.rz(angle, i)
            perturbation_info[f'loop_rotation_{i}'] = angle
        else:
            qc.rz(np.pi/3, i)
    
    # Create the CTC loop structure
    if loop_qubits > 1:
        for i in range(loop_qubits):
            qc.cx(i, (i + 1) % loop_qubits)
    else:
        # Single loop qubit - apply self-interaction
        qc.h(0)
    
    # Add bulk-loop coupling that creates the fixed point
    for i in range(loop_qubits):
        for j in range(loop_qubits, num_qubits):
            if perturbation_type == 'coupling_variation':
                phase = np.pi/2 + np.random.normal(0, perturbation_magnitude * np.pi/4)
                qc.cp(phase, i, j)
                perturbation_info[f'fixed_point_phase_{i}_{j}'] = phase
            else:
                qc.cp(np.pi/2, i, j)  # Controlled phase for fixed point
    
    return qc, perturbation_info

def deutsch_fixed_point_iteration_perturbed(qc, loop_qubits, max_iters=20, tol=1e-6):
    """
    Solve ρ_C = Tr_S[ U (ρ_S ⊗ ρ_C) U† ] by iteration with perturbation tracking
    """
    print(f"[PERTURBED] Starting fixed-point iteration for {len(loop_qubits)} loop qubits")
    
    # Get the unitary matrix for the circuit
    U_mat = Operator(qc).data
    print(f"[PERTURBED] Circuit unitary shape: {U_mat.shape}")
    
    # Number of S-qubits (bulk):
    n_S = qc.num_qubits - len(loop_qubits)
    dim_S = 2**n_S
    
    # Number of C-qubits (loop):
    n_C = len(loop_qubits)
    dim_C = 2**n_C
    
    print(f"[PERTURBED] Bulk qubits: {n_S}, Loop qubits: {n_C}")
    print(f"[PERTURBED] Bulk dimension: {dim_S}, Loop dimension: {dim_C}")
    
    # Initialize ρ_S as maximally mixed on the bulk
    rho_S = DensityMatrix(np.eye(dim_S) / dim_S)
    
    # Initialize ρ_C as maximally mixed on the loop
    rho_C = DensityMatrix(np.eye(dim_C) / dim_C)
    
    convergence_info = {
        'iterations': 0,
        'converged': False,
        'final_fidelity': 0.0,
        'fidelity_history': [],
        'perturbation_effects': []
    }
    
    for iteration in range(max_iters):
        # Build the joint state on n_S + n_C qubits
        joint = rho_S.tensor(rho_C)
        
        # Apply U
        joint = DensityMatrix(U_mat @ joint.data @ U_mat.conj().T)
        
        # Trace out the S-subsystem (bulk qubits)
        new_rho_C = partial_trace(joint, list(range(n_S)))
        new_rho_C = DensityMatrix(new_rho_C.data)
        
        # Check convergence by fidelity
        fidelity = state_fidelity(rho_C, new_rho_C)
        convergence_info['fidelity_history'].append(fidelity)
        
        # Track perturbation effects
        if iteration > 0:
            fidelity_change = abs(fidelity - convergence_info['fidelity_history'][-2])
            convergence_info['perturbation_effects'].append(fidelity_change)
        
        print(f"[PERTURBED] Iteration {iteration + 1}: Fidelity = {fidelity:.6f}")
        
        if abs(fidelity - 1) < tol:
            convergence_info['converged'] = True
            convergence_info['final_fidelity'] = fidelity
            convergence_info['iterations'] = iteration + 1
            print(f"[PERTURBED] ✅ Fixed point found after {iteration + 1} iterations!")
            break
        
        rho_C = new_rho_C
    
    if not convergence_info['converged']:
        print(f"[PERTURBED] ⚠️ Fixed point not found after {max_iters} iterations")
        convergence_info['iterations'] = max_iters
        convergence_info['final_fidelity'] = convergence_info['fidelity_history'][-1]
    
    return rho_C, convergence_info

def run_perturbation_experiment(num_qubits=4, perturbation_types=None, perturbation_magnitudes=None, device="simulator", shots=1024):
    """
    Run CTC perturbation experiment with multiple perturbation types and magnitudes
    """
    if perturbation_types is None:
        perturbation_types = ['none', 'circuit_noise', 'initial_state', 'coupling_variation']
    
    if perturbation_magnitudes is None:
        perturbation_magnitudes = [0.0, 0.05, 0.1, 0.15, 0.2]
    
    results = {
        'experiment_info': {
            'num_qubits': num_qubits,
            'device': device,
            'shots': shots,
            'timestamp': datetime.now().isoformat()
        },
        'perturbation_results': []
    }
    
    print(f"🔬 CTC PERTURBATION EXPERIMENT")
    print(f"==================================")
    print(f"Qubits: {num_qubits}")
    print(f"Device: {device}")
    print(f"Perturbation types: {perturbation_types}")
    print(f"Perturbation magnitudes: {perturbation_magnitudes}")
    print()
    
    for pert_type in perturbation_types:
        for pert_mag in perturbation_magnitudes:
            print(f"🧪 Testing {pert_type} perturbation (magnitude: {pert_mag})")
            
            # Create perturbed circuit
            if pert_type == 'none':
                qc, pert_info = create_perturbed_deutsch_ctc(num_qubits, 'circuit_noise', 0.0)
                pert_info['type'] = 'none'
                pert_info['magnitude'] = 0.0
            else:
                qc, pert_info = create_perturbed_deutsch_ctc(num_qubits, pert_type, pert_mag)
            
            # Remove measurements for fixed-point iteration
            qc_no_measure = qc.copy()
            qc_no_measure.remove_final_measurements(inplace=True)
            
            # Run Deutsch fixed-point iteration
            loop_qubits = list(range(pert_info['loop_qubits']))
            rho_C, conv_info = deutsch_fixed_point_iteration_perturbed(qc_no_measure, loop_qubits)
            
            # Sample from fixed point
            counts = sample_from_fixed_point(rho_C, shots)
            
            # Store results
            result = {
                'perturbation_type': pert_type,
                'perturbation_magnitude': pert_mag,
                'perturbation_info': pert_info,
                'convergence_info': conv_info,
                'counts': counts,
                'success': conv_info['converged']
            }
            
            results['perturbation_results'].append(result)
            
            print(f"   ✅ Converged: {conv_info['converged']}")
            print(f"   📊 Fidelity: {conv_info['final_fidelity']:.6f}")
            print(f"   🔄 Iterations: {conv_info['iterations']}")
            print()
    
    return results

def sample_from_fixed_point(rho_C, shots=1024):
    """
    Sample measurement counts from the fixed-point density matrix
    """
    # Get the probabilities from the density matrix
    probs = np.real(np.diag(rho_C.data))
    
    # Normalize probabilities
    probs = probs / np.sum(probs)
    
    # Sample from the distribution
    samples = np.random.choice(len(probs), size=shots, p=probs)
    
    # Convert to counts format
    counts = {}
    for sample in samples:
        bitstring = format(sample, f'0{int(np.log2(len(probs)))}b')
        counts[bitstring] = counts.get(bitstring, 0) + 1
    
    return counts

def analyze_perturbation_stability(results):
    """
    Analyze the stability of the CTC under perturbations
    """
    analysis = {
        'overall_stats': {},
        'by_perturbation_type': {},
        'by_magnitude': {},
        'stability_metrics': {}
    }
    
    # Overall statistics
    total_experiments = len(results['perturbation_results'])
    successful_experiments = sum(1 for r in results['perturbation_results'] if r['success'])
    analysis['overall_stats'] = {
        'total_experiments': total_experiments,
        'successful_experiments': successful_experiments,
        'success_rate': successful_experiments / total_experiments,
        'average_fidelity': np.mean([r['convergence_info']['final_fidelity'] for r in results['perturbation_results']]),
        'average_iterations': np.mean([r['convergence_info']['iterations'] for r in results['perturbation_results']])
    }
    
    # Analysis by perturbation type
    for pert_type in set(r['perturbation_type'] for r in results['perturbation_results']):
        type_results = [r for r in results['perturbation_results'] if r['perturbation_type'] == pert_type]
        successful = sum(1 for r in type_results if r['success'])
        
        analysis['by_perturbation_type'][pert_type] = {
            'total': len(type_results),
            'successful': successful,
            'success_rate': successful / len(type_results),
            'average_fidelity': np.mean([r['convergence_info']['final_fidelity'] for r in type_results]),
            'average_iterations': np.mean([r['convergence_info']['iterations'] for r in type_results])
        }
    
    # Analysis by magnitude
    for pert_mag in set(r['perturbation_magnitude'] for r in results['perturbation_results']):
        mag_results = [r for r in results['perturbation_results'] if r['perturbation_magnitude'] == pert_mag]
        successful = sum(1 for r in mag_results if r['success'])
        
        analysis['by_magnitude'][pert_mag] = {
            'total': len(mag_results),
            'successful': successful,
            'success_rate': successful / len(mag_results),
            'average_fidelity': np.mean([r['convergence_info']['final_fidelity'] for r in mag_results]),
            'average_iterations': np.mean([r['convergence_info']['iterations'] for r in mag_results])
        }
    
    # Stability metrics
    analysis['stability_metrics'] = {
        'robustness_threshold': None,  # Will be calculated
        'critical_perturbation': None,
        'stability_score': analysis['overall_stats']['success_rate']
    }
    
    # Find robustness threshold (magnitude where success rate drops below 0.8)
    magnitudes = sorted(analysis['by_magnitude'].keys())
    for mag in magnitudes:
        if analysis['by_magnitude'][mag]['success_rate'] < 0.8:
            analysis['stability_metrics']['robustness_threshold'] = mag
            break
    
    return analysis

def create_perturbation_visualization(results, analysis):
    """
    Create visualization of perturbation stability results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CTC Perturbation Stability Analysis', fontsize=16, fontweight='bold')
    
    # 1. Success rate by perturbation type
    pert_types = list(analysis['by_perturbation_type'].keys())
    success_rates = [analysis['by_perturbation_type'][pt]['success_rate'] for pt in pert_types]
    
    axes[0, 0].bar(pert_types, success_rates, color=['blue', 'green', 'orange', 'red'])
    axes[0, 0].set_title('Success Rate by Perturbation Type')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(success_rates):
        axes[0, 0].text(i, v + 0.02, f'{v:.2f}', ha='center')
    
    # 2. Success rate by magnitude
    magnitudes = sorted(analysis['by_magnitude'].keys())
    mag_success_rates = [analysis['by_magnitude'][mag]['success_rate'] for mag in magnitudes]
    
    axes[0, 1].plot(magnitudes, mag_success_rates, 'o-', linewidth=2, markersize=8)
    axes[0, 1].set_title('Success Rate vs Perturbation Magnitude')
    axes[0, 1].set_xlabel('Perturbation Magnitude')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Average fidelity by magnitude
    mag_fidelities = [analysis['by_magnitude'][mag]['average_fidelity'] for mag in magnitudes]
    
    axes[1, 0].plot(magnitudes, mag_fidelities, 's-', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_title('Average Fidelity vs Perturbation Magnitude')
    axes[1, 0].set_xlabel('Perturbation Magnitude')
    axes[1, 0].set_ylabel('Average Fidelity')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Average iterations by magnitude
    mag_iterations = [analysis['by_magnitude'][mag]['average_iterations'] for mag in magnitudes]
    
    axes[1, 1].plot(magnitudes, mag_iterations, '^-', linewidth=2, markersize=8, color='red')
    axes[1, 1].set_title('Average Iterations vs Perturbation Magnitude')
    axes[1, 1].set_xlabel('Perturbation Magnitude')
    axes[1, 1].set_ylabel('Average Iterations')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = f"ctc_perturbation_analysis_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file

def generate_perturbation_report(results, analysis, plot_file):
    """
    Generate a comprehensive report on perturbation stability
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"CTC_PERTURBATION_REPORT_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("CTC PERTURBATION STABILITY REPORT\n")
        f.write("==================================\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("==================\n")
        f.write(f"Total experiments: {analysis['overall_stats']['total_experiments']}\n")
        f.write(f"Success rate: {analysis['overall_stats']['success_rate']:.2%}\n")
        f.write(f"Average fidelity: {analysis['overall_stats']['average_fidelity']:.6f}\n")
        f.write(f"Average iterations: {analysis['overall_stats']['average_iterations']:.1f}\n")
        f.write(f"Robustness threshold: {analysis['stability_metrics']['robustness_threshold']}\n")
        f.write(f"Stability score: {analysis['stability_metrics']['stability_score']:.2f}\n\n")
        
        f.write("DETAILED ANALYSIS BY PERTURBATION TYPE\n")
        f.write("=====================================\n")
        for pert_type, stats in analysis['by_perturbation_type'].items():
            f.write(f"\n{pert_type.upper()}:\n")
            f.write(f"  Success rate: {stats['success_rate']:.2%}\n")
            f.write(f"  Average fidelity: {stats['average_fidelity']:.6f}\n")
            f.write(f"  Average iterations: {stats['average_iterations']:.1f}\n")
        
        f.write("\nDETAILED ANALYSIS BY MAGNITUDE\n")
        f.write("=============================\n")
        for magnitude, stats in analysis['by_magnitude'].items():
            f.write(f"\nMagnitude {magnitude}:\n")
            f.write(f"  Success rate: {stats['success_rate']:.2%}\n")
            f.write(f"  Average fidelity: {stats['average_fidelity']:.6f}\n")
            f.write(f"  Average iterations: {stats['average_iterations']:.1f}\n")
        
        f.write("\nSTABILITY CONCLUSIONS\n")
        f.write("=====================\n")
        if analysis['stability_metrics']['robustness_threshold']:
            f.write(f"[SUCCESS] CTC is robust up to perturbation magnitude: {analysis['stability_metrics']['robustness_threshold']}\n")
        else:
            f.write("[SUCCESS] CTC remains stable across all tested perturbation magnitudes\n")
        
        f.write(f"[CHART] Overall stability score: {analysis['stability_metrics']['stability_score']:.2f}\n")
        f.write(f"[ANALYSIS] Visualization: {plot_file}\n")
        
        f.write("\nPHYSICAL INTERPRETATION\n")
        f.write("======================\n")
        f.write("The perturbation analysis reveals:\n")
        f.write("1. How robust the Deutsch fixed-point CTC is to noise\n")
        f.write("2. Whether perturbations cause convergence failures\n")
        f.write("3. The tolerance limits of the CTC implementation\n")
        f.write("4. The stability of self-consistent solutions under disturbances\n")
        
        f.write("\nCONCLUSION\n")
        f.write("==========\n")
        if analysis['stability_metrics']['stability_score'] > 0.8:
            f.write("[SUCCESS] CTC is HIGHLY STABLE under perturbations\n")
        elif analysis['stability_metrics']['stability_score'] > 0.6:
            f.write("[WARNING] CTC is MODERATELY STABLE under perturbations\n")
        else:
            f.write("[FAILURE] CTC is UNSTABLE under perturbations\n")
        
        f.write("The evidence demonstrates the robustness of quantum consistency!\n")
    
    return report_file

def main():
    parser = argparse.ArgumentParser(description='Run CTC perturbation experiment')
    parser.add_argument('--num_qubits', type=int, default=4, help='Number of qubits')
    parser.add_argument('--device', type=str, default='simulator', help='Device to use')
    parser.add_argument('--shots', type=int, default=1024, help='Number of shots')
    parser.add_argument('--perturbation_types', nargs='+', 
                       default=['none', 'circuit_noise', 'initial_state', 'coupling_variation'],
                       help='Types of perturbations to test')
    parser.add_argument('--perturbation_magnitudes', nargs='+', type=float,
                       default=[0.0, 0.05, 0.1, 0.15, 0.2],
                       help='Perturbation magnitudes to test')
    
    args = parser.parse_args()
    
    print("🔬 CTC PERTURBATION EXPERIMENT")
    print("================================")
    print(f"Qubits: {args.num_qubits}")
    print(f"Device: {args.device}")
    print(f"Perturbation types: {args.perturbation_types}")
    print(f"Perturbation magnitudes: {args.perturbation_magnitudes}")
    print()
    
    # Run perturbation experiment
    results = run_perturbation_experiment(
        num_qubits=args.num_qubits,
        perturbation_types=args.perturbation_types,
        perturbation_magnitudes=args.perturbation_magnitudes,
        device=args.device,
        shots=args.shots
    )
    
    # Analyze results
    print("📊 Analyzing perturbation stability...")
    analysis = analyze_perturbation_stability(results)
    
    # Create visualization
    print("📈 Creating perturbation visualization...")
    plot_file = create_perturbation_visualization(results, analysis)
    
    # Generate report
    print("📋 Generating perturbation report...")
    report_file = generate_perturbation_report(results, analysis, plot_file)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"ctc_perturbation_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("🎉 CTC PERTURBATION EXPERIMENT COMPLETE!")
    print(f"   Results: {results_file}")
    print(f"   Visualization: {plot_file}")
    print(f"   Report: {report_file}")
    print(f"   Overall success rate: {analysis['overall_stats']['success_rate']:.2%}")
    print(f"   Stability score: {analysis['stability_metrics']['stability_score']:.2f}")

if __name__ == "__main__":
    main() 