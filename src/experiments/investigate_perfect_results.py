#!/usr/bin/env python3
"""
INVESTIGATE PERFECT RESULTS: Are our CTC results real or numerical artifacts?
================================================================================

This script investigates whether our suspiciously perfect results are:
1. Real fixed points (good)
2. Numerical precision artifacts (bad)
3. Trivial solutions (bad)
4. Algorithm bugs (bad)

We'll test:
- Much larger perturbation magnitudes
- Examine actual density matrices
- Check for trivial vs interesting solutions
- Test numerical precision limits
- Compare with analytical expectations
"""

import sys
import os
import numpy as np
import json
from datetime import datetime
import argparse
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import Operator, DensityMatrix, partial_trace, state_fidelity, entropy
import matplotlib.pyplot as plt

# Add src to path
sys.path.append('src')
from CGPTFactory import run

def create_perturbed_deutsch_ctc(num_qubits, perturbation_type, perturbation_magnitude):
    """Create a Deutsch CTC circuit with controlled perturbations"""
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

def deutsch_fixed_point_iteration_detailed(qc, loop_qubits, max_iters=20, tol=1e-6):
    """
    Solve ρ_C = Tr_S[ U (ρ_S ⊗ ρ_C) U† ] by iteration with detailed analysis
    """
    print(f"[DETAILED] Starting fixed-point iteration for {len(loop_qubits)} loop qubits")
    
    # Get the unitary matrix for the circuit
    U_mat = Operator(qc).data
    print(f"[DETAILED] Circuit unitary shape: {U_mat.shape}")
    
    # Number of S-qubits (bulk):
    n_S = qc.num_qubits - len(loop_qubits)
    dim_S = 2**n_S
    
    # Number of C-qubits (loop):
    n_C = len(loop_qubits)
    dim_C = 2**n_C
    
    print(f"[DETAILED] Bulk qubits: {n_S}, Loop qubits: {n_C}")
    print(f"[DETAILED] Bulk dimension: {dim_S}, Loop dimension: {dim_C}")
    
    # Initialize ρ_S as maximally mixed on the bulk
    rho_S = DensityMatrix(np.eye(dim_S) / dim_S)
    
    # Initialize ρ_C as maximally mixed on the loop
    rho_C = DensityMatrix(np.eye(dim_C) / dim_C)
    
    convergence_info = {
        'iterations': 0,
        'converged': False,
        'final_fidelity': 0.0,
        'fidelity_history': [],
        'density_matrices': [],
        'entropies': [],
        'purities': [],
        'numerical_issues': []
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
        
        # Store density matrix for analysis
        convergence_info['density_matrices'].append(new_rho_C.data.copy())
        
        # Calculate entropy and purity
        entropy_val = entropy(new_rho_C)
        purity = np.trace(new_rho_C.data @ new_rho_C.data)
        convergence_info['entropies'].append(entropy_val)
        convergence_info['purities'].append(purity)
        
        # Check for numerical issues
        if np.any(np.isnan(new_rho_C.data)) or np.any(np.isinf(new_rho_C.data)):
            convergence_info['numerical_issues'].append(f"Iteration {iteration}: NaN/Inf detected")
        
        # Check if density matrix is Hermitian (should be)
        if not np.allclose(new_rho_C.data, new_rho_C.data.conj().T):
            convergence_info['numerical_issues'].append(f"Iteration {iteration}: Non-Hermitian matrix")
        
        # Check if trace is 1 (should be)
        if not np.isclose(np.trace(new_rho_C.data), 1.0, atol=1e-10):
            convergence_info['numerical_issues'].append(f"Iteration {iteration}: Trace != 1")
        
        print(f"[DETAILED] Iteration {iteration + 1}: Fidelity = {fidelity:.10f}, Entropy = {entropy_val:.6f}, Purity = {purity:.6f}")
        
        if abs(fidelity - 1) < tol:
            convergence_info['converged'] = True
            convergence_info['final_fidelity'] = fidelity
            convergence_info['iterations'] = iteration + 1
            print(f"[DETAILED] [SUCCESS] Fixed point found after {iteration + 1} iterations!")
            break
        
        rho_C = new_rho_C
    
    if not convergence_info['converged']:
        print(f"[DETAILED] [WARNING] Fixed point not found after {max_iters} iterations")
        convergence_info['iterations'] = max_iters
        convergence_info['final_fidelity'] = convergence_info['fidelity_history'][-1]
    
    return rho_C, convergence_info

def analyze_density_matrix(rho_C, iteration_num):
    """
    Analyze a density matrix to determine if it's interesting or trivial
    """
    analysis = {
        'iteration': iteration_num,
        'is_trivial': False,
        'is_maximally_mixed': False,
        'is_pure': False,
        'entropy': entropy(DensityMatrix(rho_C)),
        'purity': np.trace(rho_C @ rho_C),
        'eigenvalues': np.linalg.eigvals(rho_C),
        'max_eigenvalue': np.max(np.real(np.linalg.eigvals(rho_C))),
        'min_eigenvalue': np.min(np.real(np.linalg.eigvals(rho_C))),
        'condition_number': np.linalg.cond(rho_C),
        'numerical_issues': []
    }
    
    # Check if it's maximally mixed (trivial)
    dim = rho_C.shape[0]
    maximally_mixed = np.eye(dim) / dim
    if np.allclose(rho_C, maximally_mixed, atol=1e-10):
        analysis['is_maximally_mixed'] = True
        analysis['is_trivial'] = True
    
    # Check if it's pure (eigenvalue = 1, rest = 0)
    eigenvalues = np.real(np.linalg.eigvals(rho_C))
    if np.any(eigenvalues > 0.999) and np.sum(eigenvalues > 0.001) == 1:
        analysis['is_pure'] = True
    
    # Check for numerical issues
    if np.any(np.isnan(rho_C)) or np.any(np.isinf(rho_C)):
        analysis['numerical_issues'].append("NaN/Inf detected")
    
    if not np.allclose(rho_C, rho_C.conj().T):
        analysis['numerical_issues'].append("Non-Hermitian")
    
    if not np.isclose(np.trace(rho_C), 1.0, atol=1e-10):
        analysis['numerical_issues'].append("Trace != 1")
    
    return analysis

def run_investigation_experiment(num_qubits=4, perturbation_types=None, perturbation_magnitudes=None):
    """
    Run investigation experiment with much larger perturbations
    """
    if perturbation_types is None:
        perturbation_types = ['circuit_noise', 'initial_state', 'coupling_variation']
    
    if perturbation_magnitudes is None:
        perturbation_magnitudes = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    results = {
        'experiment_info': {
            'num_qubits': num_qubits,
            'timestamp': datetime.now().isoformat(),
            'investigation_type': 'perfect_results_analysis'
        },
        'investigation_results': []
    }
    
    print(f"🔍 INVESTIGATING PERFECT RESULTS")
    print(f"=================================")
    print(f"Qubits: {num_qubits}")
    print(f"Perturbation types: {perturbation_types}")
    print(f"Perturbation magnitudes: {perturbation_magnitudes}")
    print()
    
    for pert_type in perturbation_types:
        for pert_mag in perturbation_magnitudes:
            print(f"🔬 Testing {pert_type} perturbation (magnitude: {pert_mag})")
            
            # Create perturbed circuit
            qc, pert_info = create_perturbed_deutsch_ctc(num_qubits, pert_type, pert_mag)
            
            # Remove measurements for fixed-point iteration
            qc_no_measure = qc.copy()
            qc_no_measure.remove_final_measurements(inplace=True)
            
            # Run detailed Deutsch fixed-point iteration
            loop_qubits = list(range(pert_info['loop_qubits']))
            rho_C, conv_info = deutsch_fixed_point_iteration_detailed(qc_no_measure, loop_qubits)
            
            # Analyze the final density matrix
            final_analysis = analyze_density_matrix(rho_C.data, conv_info['iterations'])
            
            # Analyze all density matrices in the iteration
            iteration_analyses = []
            for i, rho_data in enumerate(conv_info['density_matrices']):
                iter_analysis = analyze_density_matrix(rho_data, i+1)
                iteration_analyses.append(iter_analysis)
            
            # Store results
            result = {
                'perturbation_type': pert_type,
                'perturbation_magnitude': pert_mag,
                'perturbation_info': pert_info,
                'convergence_info': conv_info,
                'final_analysis': final_analysis,
                'iteration_analyses': iteration_analyses,
                'success': conv_info['converged'],
                'is_trivial': final_analysis['is_trivial'],
                'numerical_issues': len(conv_info['numerical_issues']) > 0
            }
            
            results['investigation_results'].append(result)
            
            print(f"   [SUCCESS] Converged: {conv_info['converged']}")
            print(f"   [ANALYSIS] Fidelity: {conv_info['final_fidelity']:.10f}")
            print(f"   [ANALYSIS] Iterations: {conv_info['iterations']}")
            print(f"   [ANALYSIS] Is trivial: {final_analysis['is_trivial']}")
            print(f"   [ANALYSIS] Is maximally mixed: {final_analysis['is_maximally_mixed']}")
            print(f"   [ANALYSIS] Is pure: {final_analysis['is_pure']}")
            print(f"   [ANALYSIS] Entropy: {final_analysis['entropy']:.6f}")
            print(f"   [ANALYSIS] Purity: {final_analysis['purity']:.6f}")
            print(f"   [ANALYSIS] Numerical issues: {len(conv_info['numerical_issues'])}")
            if conv_info['numerical_issues']:
                print(f"   [WARNING] Issues: {conv_info['numerical_issues']}")
            print()
    
    return results

def analyze_investigation_results(results):
    """
    Analyze the investigation results to determine if our perfect results are real
    """
    analysis = {
        'summary': {},
        'by_magnitude': {},
        'trivial_solutions': 0,
        'numerical_issues': 0,
        'real_fixed_points': 0,
        'conclusion': ""
    }
    
    total_experiments = len(results['investigation_results'])
    successful_experiments = sum(1 for r in results['investigation_results'] if r['success'])
    trivial_solutions = sum(1 for r in results['investigation_results'] if r['is_trivial'])
    numerical_issues = sum(1 for r in results['investigation_results'] if r['numerical_issues'])
    
    analysis['summary'] = {
        'total_experiments': total_experiments,
        'successful_experiments': successful_experiments,
        'success_rate': successful_experiments / total_experiments,
        'trivial_solutions': trivial_solutions,
        'trivial_rate': trivial_solutions / total_experiments,
        'numerical_issues': numerical_issues,
        'numerical_issue_rate': numerical_issues / total_experiments,
        'real_fixed_points': successful_experiments - trivial_solutions - numerical_issues
    }
    
    # Analysis by magnitude
    for pert_mag in set(r['perturbation_magnitude'] for r in results['investigation_results']):
        mag_results = [r for r in results['investigation_results'] if r['perturbation_magnitude'] == pert_mag]
        successful = sum(1 for r in mag_results if r['success'])
        trivial = sum(1 for r in mag_results if r['is_trivial'])
        issues = sum(1 for r in mag_results if r['numerical_issues'])
        
        analysis['by_magnitude'][pert_mag] = {
            'total': len(mag_results),
            'successful': successful,
            'success_rate': successful / len(mag_results),
            'trivial': trivial,
            'trivial_rate': trivial / len(mag_results),
            'numerical_issues': issues,
            'issue_rate': issues / len(mag_results)
        }
    
    # Determine conclusion
    if analysis['summary']['trivial_rate'] > 0.8:
        analysis['conclusion'] = "MOSTLY TRIVIAL SOLUTIONS - Our perfect results are mostly finding maximally mixed states"
    elif analysis['summary']['numerical_issue_rate'] > 0.5:
        analysis['conclusion'] = "NUMERICAL ISSUES - Our perfect results are due to numerical precision problems"
    elif analysis['summary']['success_rate'] < 0.5:
        analysis['conclusion'] = "BREAKS UNDER LARGE PERTURBATIONS - Our stability was only apparent for small perturbations"
    else:
        analysis['conclusion'] = "REAL FIXED POINTS - Our perfect results represent genuine quantum fixed points"
    
    return analysis

def create_investigation_visualization(results, analysis):
    """
    Create visualization of investigation results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Investigation of Perfect CTC Results', fontsize=16, fontweight='bold')
    
    # 1. Success rate by magnitude
    magnitudes = sorted(analysis['by_magnitude'].keys())
    success_rates = [analysis['by_magnitude'][mag]['success_rate'] for mag in magnitudes]
    
    axes[0, 0].plot(magnitudes, success_rates, 'o-', linewidth=2, markersize=8, color='blue')
    axes[0, 0].set_title('Success Rate vs Perturbation Magnitude')
    axes[0, 0].set_xlabel('Perturbation Magnitude')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Trivial solution rate by magnitude
    trivial_rates = [analysis['by_magnitude'][mag]['trivial_rate'] for mag in magnitudes]
    
    axes[0, 1].plot(magnitudes, trivial_rates, 's-', linewidth=2, markersize=8, color='red')
    axes[0, 1].set_title('Trivial Solution Rate vs Perturbation Magnitude')
    axes[0, 1].set_xlabel('Perturbation Magnitude')
    axes[0, 1].set_ylabel('Trivial Solution Rate')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Numerical issues by magnitude
    issue_rates = [analysis['by_magnitude'][mag]['issue_rate'] for mag in magnitudes]
    
    axes[1, 0].plot(magnitudes, issue_rates, '^-', linewidth=2, markersize=8, color='orange')
    axes[1, 0].set_title('Numerical Issues Rate vs Perturbation Magnitude')
    axes[1, 0].set_xlabel('Perturbation Magnitude')
    axes[1, 0].set_ylabel('Numerical Issue Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Summary pie chart
    labels = ['Real Fixed Points', 'Trivial Solutions', 'Numerical Issues', 'Failed']
    sizes = [
        analysis['summary']['real_fixed_points'],
        analysis['summary']['trivial_solutions'],
        analysis['summary']['numerical_issues'],
        analysis['summary']['total_experiments'] - analysis['summary']['successful_experiments']
    ]
    colors = ['green', 'red', 'orange', 'gray']
    
    axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Overall Results Breakdown')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = f"investigation_results_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file

def generate_investigation_report(results, analysis, plot_file):
    """
    Generate a comprehensive investigation report
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"INVESTIGATION_REPORT_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("INVESTIGATION OF PERFECT CTC RESULTS\n")
        f.write("====================================\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("==================\n")
        f.write(f"Total experiments: {analysis['summary']['total_experiments']}\n")
        f.write(f"Success rate: {analysis['summary']['success_rate']:.2%}\n")
        f.write(f"Trivial solutions: {analysis['summary']['trivial_solutions']} ({analysis['summary']['trivial_rate']:.2%})\n")
        f.write(f"Numerical issues: {analysis['summary']['numerical_issues']} ({analysis['summary']['numerical_issue_rate']:.2%})\n")
        f.write(f"Real fixed points: {analysis['summary']['real_fixed_points']}\n")
        f.write(f"CONCLUSION: {analysis['conclusion']}\n\n")
        
        f.write("DETAILED ANALYSIS BY MAGNITUDE\n")
        f.write("=============================\n")
        for magnitude, stats in analysis['by_magnitude'].items():
            f.write(f"\nMagnitude {magnitude}:\n")
            f.write(f"  Success rate: {stats['success_rate']:.2%}\n")
            f.write(f"  Trivial rate: {stats['trivial_rate']:.2%}\n")
            f.write(f"  Numerical issues: {stats['issue_rate']:.2%}\n")
        
        f.write("\nCRITICAL FINDINGS\n")
        f.write("=================\n")
        if analysis['summary']['trivial_rate'] > 0.5:
            f.write("[WARNING] Most solutions are trivial (maximally mixed states)\n")
            f.write("This suggests our algorithm is not finding interesting quantum states\n")
        
        if analysis['summary']['numerical_issue_rate'] > 0.3:
            f.write("[WARNING] Significant numerical issues detected\n")
            f.write("Our perfect results may be due to floating-point precision problems\n")
        
        if analysis['summary']['success_rate'] < 0.5:
            f.write("[WARNING] Success rate drops significantly with large perturbations\n")
            f.write("Our stability was only apparent for small perturbations\n")
        
        if analysis['summary']['real_fixed_points'] > analysis['summary']['total_experiments'] * 0.3:
            f.write("[SUCCESS] Found significant number of real fixed points\n")
            f.write("Our perfect results represent genuine quantum phenomena\n")
        
        f.write(f"\n[ANALYSIS] Visualization: {plot_file}\n")
        
        f.write("\nRECOMMENDATIONS\n")
        f.write("===============\n")
        if analysis['conclusion'].startswith("MOSTLY TRIVIAL"):
            f.write("1. Modify algorithm to avoid trivial solutions\n")
            f.write("2. Use different initial conditions\n")
            f.write("3. Implement constraints to find non-trivial fixed points\n")
        elif analysis['conclusion'].startswith("NUMERICAL"):
            f.write("1. Use higher precision arithmetic\n")
            f.write("2. Implement better numerical stability checks\n")
            f.write("3. Use different convergence criteria\n")
        elif analysis['conclusion'].startswith("BREAKS"):
            f.write("1. Our stability was only apparent for small perturbations\n")
            f.write("2. Need to understand the stability limits better\n")
            f.write("3. Consider different perturbation types\n")
        else:
            f.write("1. Our results are genuine - proceed with confidence\n")
            f.write("2. Consider publishing these findings\n")
            f.write("3. Test on larger systems\n")
    
    return report_file

def main():
    parser = argparse.ArgumentParser(description='Investigate perfect CTC results')
    parser.add_argument('--num_qubits', type=int, default=4, help='Number of qubits')
    parser.add_argument('--perturbation_types', nargs='+', 
                       default=['circuit_noise', 'initial_state', 'coupling_variation'],
                       help='Types of perturbations to test')
    parser.add_argument('--perturbation_magnitudes', nargs='+', type=float,
                       default=[0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
                       help='Perturbation magnitudes to test')
    
    args = parser.parse_args()
    
    print("🔍 INVESTIGATING PERFECT CTC RESULTS")
    print("=====================================")
    print(f"Qubits: {args.num_qubits}")
    print(f"Perturbation types: {args.perturbation_types}")
    print(f"Perturbation magnitudes: {args.perturbation_magnitudes}")
    print()
    
    # Run investigation experiment
    results = run_investigation_experiment(
        num_qubits=args.num_qubits,
        perturbation_types=args.perturbation_types,
        perturbation_magnitudes=args.perturbation_magnitudes
    )
    
    # Analyze results
    print("📊 Analyzing investigation results...")
    analysis = analyze_investigation_results(results)
    
    # Create visualization
    print("📈 Creating investigation visualization...")
    plot_file = create_investigation_visualization(results, analysis)
    
    # Generate report
    print("📋 Generating investigation report...")
    report_file = generate_investigation_report(results, analysis, plot_file)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"investigation_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("🎉 INVESTIGATION COMPLETE!")
    print(f"   Results: {results_file}")
    print(f"   Visualization: {plot_file}")
    print(f"   Report: {report_file}")
    print(f"   Conclusion: {analysis['conclusion']}")
    print(f"   Trivial solutions: {analysis['summary']['trivial_solutions']}/{analysis['summary']['total_experiments']}")
    print(f"   Numerical issues: {analysis['summary']['numerical_issues']}/{analysis['summary']['total_experiments']}")

if __name__ == "__main__":
    main() 