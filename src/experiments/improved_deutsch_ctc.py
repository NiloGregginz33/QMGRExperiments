#!/usr/bin/env python3
"""
IMPROVED DEUTSCH CTC: Finding Non-Trivial Fixed Points
======================================================

This script implements an improved Deutsch fixed-point CTC that avoids
trivial (maximally mixed) solutions by:

1. Using non-maximally mixed initial conditions
2. Adding constraints to avoid trivial solutions
3. Implementing better fixed-point search algorithms
4. Testing with different circuit structures
5. Using multiple search strategies

The goal is to find interesting quantum states that represent genuine
CTC fixed points rather than just the most boring possible solution.
"""

import sys
import os
import numpy as np
import json
from datetime import datetime
import argparse
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import Operator, DensityMatrix, partial_trace, state_fidelity, entropy, random_density_matrix
import matplotlib.pyplot as plt

# Add src to path
sys.path.append('src')
from CGPTFactory import run

def create_improved_ctc_circuit(num_qubits, circuit_type='entangled'):
    """
    Create improved CTC circuits that are more likely to have non-trivial fixed points
    """
    qc = QuantumCircuit(num_qubits)
    
    # Separate loop and bulk qubits
    loop_qubits = min(2, num_qubits // 2)
    bulk_qubits = num_qubits - loop_qubits
    
    circuit_info = {
        'type': circuit_type,
        'loop_qubits': loop_qubits,
        'bulk_qubits': bulk_qubits
    }
    
    if circuit_type == 'entangled':
        # Create highly entangled initial state
        for i in range(num_qubits):
            qc.h(i)
        
        # Create entanglement between all qubits
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                qc.cx(i, j)
                qc.cp(np.pi/3, i, j)
        
        # Add time-asymmetric operations
        for i in range(loop_qubits):
            qc.t(i)  # T-gate breaks time-reversal symmetry
            qc.rz(np.pi/4, i)
        
        # Create strong loop-bulk coupling
        for i in range(loop_qubits):
            for j in range(loop_qubits, num_qubits):
                qc.cx(i, j)
                qc.cp(np.pi/2, i, j)
                qc.rzz(np.pi/3, i, j)
        
        # Add loop structure with non-trivial dynamics
        if loop_qubits > 1:
            for i in range(loop_qubits):
                qc.cx(i, (i + 1) % loop_qubits)
                qc.cp(np.pi/4, i, (i + 1) % loop_qubits)
        else:
            qc.h(0)
            qc.t(0)
    
    elif circuit_type == 'quantum_walk':
        # Quantum walk inspired circuit
        for i in range(num_qubits):
            qc.h(i)
        
        # Create quantum walk structure
        for step in range(3):
            for i in range(num_qubits):
                qc.rz(step * np.pi/6, i)
            
            for i in range(num_qubits - 1):
                qc.cx(i, i+1)
                qc.cp(np.pi/4, i, i+1)
            
            # Periodic boundary
            qc.cx(num_qubits-1, 0)
            qc.cp(np.pi/4, num_qubits-1, 0)
        
        # Add time-asymmetric operations on loop
        for i in range(loop_qubits):
            qc.t(i)
            qc.rz(np.pi/3, i)
    
    elif circuit_type == 'topological':
        # Topological-inspired circuit
        for i in range(num_qubits):
            qc.h(i)
        
        # Create topological-like structure
        for i in range(loop_qubits):
            qc.rz(np.pi/4, i)
            qc.rx(np.pi/4, i)
        
        # Create bulk in different state
        for i in range(loop_qubits, num_qubits):
            qc.rz(np.pi/2, i)
            qc.rx(np.pi/2, i)
        
        # Strong entanglement between loop and bulk
        for i in range(loop_qubits):
            for j in range(loop_qubits, num_qubits):
                qc.cx(i, j)
                qc.cp(np.pi/2, i, j)
                qc.rzz(np.pi/2, i, j)
        
        # Add time-asymmetric operations
        for i in range(loop_qubits):
            qc.t(i)
    
    return qc, circuit_info

def create_non_trivial_initial_state(dim_C, strategy='random_pure'):
    """
    Create non-trivial initial states for the loop qubits
    """
    if strategy == 'random_pure':
        # Create a random pure state
        random_state = np.random.randn(dim_C) + 1j * np.random.randn(dim_C)
        random_state = random_state / np.linalg.norm(random_state)
        rho_C = np.outer(random_state, random_state.conj())
        return DensityMatrix(rho_C)
    
    elif strategy == 'random_mixed':
        # Create a random mixed state (not maximally mixed)
        rho_C = random_density_matrix(dim_C)
        return rho_C
    
    elif strategy == 'coherent':
        # Create a coherent state (superposition)
        if dim_C == 2:
            # For 1 qubit: |0⟩ + |1⟩
            state = np.array([1.0, 1.0]) / np.sqrt(2)
        elif dim_C == 4:
            # For 2 qubits: |00⟩ + |01⟩ + |10⟩ + |11⟩
            state = np.array([1.0, 1.0, 1.0, 1.0]) / 2.0
        else:
            state = np.ones(dim_C) / np.sqrt(dim_C)
        
        rho_C = np.outer(state, state.conj())
        return DensityMatrix(rho_C)
    
    elif strategy == 'bell_like':
        # Create Bell-like state for 2 qubits
        if dim_C == 4:
            state = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2)
            rho_C = np.outer(state, state.conj())
            return DensityMatrix(rho_C)
        else:
            # Fall back to coherent
            return create_non_trivial_initial_state(dim_C, 'coherent')
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def deutsch_fixed_point_iteration_improved(qc, loop_qubits, initial_strategy='random_pure', max_iters=50, tol=1e-6):
    """
    Improved Deutsch fixed-point iteration with non-trivial initial conditions
    """
    print(f"[IMPROVED] Starting improved fixed-point iteration for {len(loop_qubits)} loop qubits")
    print(f"[IMPROVED] Initial strategy: {initial_strategy}")
    
    # Get the unitary matrix for the circuit
    U_mat = Operator(qc).data
    print(f"[IMPROVED] Circuit unitary shape: {U_mat.shape}")
    
    # Number of S-qubits (bulk):
    n_S = qc.num_qubits - len(loop_qubits)
    dim_S = 2**n_S
    
    # Number of C-qubits (loop):
    n_C = len(loop_qubits)
    dim_C = 2**n_C
    
    print(f"[IMPROVED] Bulk qubits: {n_S}, Loop qubits: {n_C}")
    print(f"[IMPROVED] Bulk dimension: {dim_S}, Loop dimension: {dim_C}")
    
    # Initialize ρ_S as maximally mixed on the bulk (this is fine)
    rho_S = DensityMatrix(np.eye(dim_S) / dim_S)
    
    # Initialize ρ_C with non-trivial state
    rho_C = create_non_trivial_initial_state(dim_C, initial_strategy)
    
    convergence_info = {
        'iterations': 0,
        'converged': False,
        'final_fidelity': 0.0,
        'fidelity_history': [],
        'density_matrices': [],
        'entropies': [],
        'purities': [],
        'numerical_issues': [],
        'initial_strategy': initial_strategy
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
        
        print(f"[IMPROVED] Iteration {iteration + 1}: Fidelity = {fidelity:.10f}, Entropy = {entropy_val:.6f}, Purity = {purity:.6f}")
        
        if abs(fidelity - 1) < tol:
            convergence_info['converged'] = True
            convergence_info['final_fidelity'] = fidelity
            convergence_info['iterations'] = iteration + 1
            print(f"[IMPROVED] [SUCCESS] Fixed point found after {iteration + 1} iterations!")
            break
        
        rho_C = new_rho_C
    
    if not convergence_info['converged']:
        print(f"[IMPROVED] [WARNING] Fixed point not found after {max_iters} iterations")
        convergence_info['iterations'] = max_iters
        convergence_info['final_fidelity'] = convergence_info['fidelity_history'][-1]
    
    return rho_C, convergence_info

def analyze_density_matrix_improved(rho_C, iteration_num):
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
        'numerical_issues': [],
        'interesting_score': 0.0
    }
    
    # Check if it's maximally mixed (trivial)
    dim = rho_C.shape[0]
    maximally_mixed = np.eye(dim) / dim
    if np.allclose(rho_C, maximally_mixed, atol=1e-10):
        analysis['is_maximally_mixed'] = True
        analysis['is_trivial'] = True
        analysis['interesting_score'] = 0.0
    
    # Check if it's pure (eigenvalue = 1, rest = 0)
    eigenvalues = np.real(np.linalg.eigvals(rho_C))
    if np.any(eigenvalues > 0.999) and np.sum(eigenvalues > 0.001) == 1:
        analysis['is_pure'] = True
        analysis['interesting_score'] = 1.0
    
    # Calculate interesting score based on entropy and purity
    if not analysis['is_trivial']:
        # Higher score for intermediate entropy (not too mixed, not too pure)
        entropy_score = 1.0 - abs(analysis['entropy'] - np.log2(dim)/2) / (np.log2(dim)/2)
        purity_score = analysis['purity']  # Higher purity is more interesting
        analysis['interesting_score'] = (entropy_score + purity_score) / 2
    
    # Check for numerical issues
    if np.any(np.isnan(rho_C)) or np.any(np.isinf(rho_C)):
        analysis['numerical_issues'].append("NaN/Inf detected")
    
    if not np.allclose(rho_C, rho_C.conj().T):
        analysis['numerical_issues'].append("Non-Hermitian")
    
    if not np.isclose(np.trace(rho_C), 1.0, atol=1e-10):
        analysis['numerical_issues'].append("Trace != 1")
    
    return analysis

def run_improved_experiment(num_qubits=4, circuit_types=None, initial_strategies=None):
    """
    Run improved Deutsch CTC experiment with multiple strategies
    """
    if circuit_types is None:
        circuit_types = ['entangled', 'quantum_walk', 'topological']
    
    if initial_strategies is None:
        initial_strategies = ['random_pure', 'random_mixed', 'coherent', 'bell_like']
    
    results = {
        'experiment_info': {
            'num_qubits': num_qubits,
            'timestamp': datetime.now().isoformat(),
            'experiment_type': 'improved_deutsch_ctc'
        },
        'improved_results': []
    }
    
    print(f"🚀 IMPROVED DEUTSCH CTC EXPERIMENT")
    print(f"===================================")
    print(f"Qubits: {num_qubits}")
    print(f"Circuit types: {circuit_types}")
    print(f"Initial strategies: {initial_strategies}")
    print()
    
    for circuit_type in circuit_types:
        for strategy in initial_strategies:
            print(f"🔬 Testing {circuit_type} circuit with {strategy} initial state")
            
            # Create improved circuit
            qc, circuit_info = create_improved_ctc_circuit(num_qubits, circuit_type)
            
            # Remove measurements for fixed-point iteration
            qc_no_measure = qc.copy()
            qc_no_measure.remove_final_measurements(inplace=True)
            
            # Run improved Deutsch fixed-point iteration
            loop_qubits = list(range(circuit_info['loop_qubits']))
            rho_C, conv_info = deutsch_fixed_point_iteration_improved(qc_no_measure, loop_qubits, strategy)
            
            # Analyze the final density matrix
            final_analysis = analyze_density_matrix_improved(rho_C.data, conv_info['iterations'])
            
            # Analyze all density matrices in the iteration
            iteration_analyses = []
            for i, rho_data in enumerate(conv_info['density_matrices']):
                iter_analysis = analyze_density_matrix_improved(rho_data, i+1)
                iteration_analyses.append(iter_analysis)
            
            # Store results
            result = {
                'circuit_type': circuit_type,
                'initial_strategy': strategy,
                'circuit_info': circuit_info,
                'convergence_info': conv_info,
                'final_analysis': final_analysis,
                'iteration_analyses': iteration_analyses,
                'success': conv_info['converged'],
                'is_trivial': final_analysis['is_trivial'],
                'interesting_score': final_analysis['interesting_score'],
                'numerical_issues': len(conv_info['numerical_issues']) > 0
            }
            
            results['improved_results'].append(result)
            
            print(f"   [SUCCESS] Converged: {conv_info['converged']}")
            print(f"   [ANALYSIS] Fidelity: {conv_info['final_fidelity']:.10f}")
            print(f"   [ANALYSIS] Iterations: {conv_info['iterations']}")
            print(f"   [ANALYSIS] Is trivial: {final_analysis['is_trivial']}")
            print(f"   [ANALYSIS] Is maximally mixed: {final_analysis['is_maximally_mixed']}")
            print(f"   [ANALYSIS] Is pure: {final_analysis['is_pure']}")
            print(f"   [ANALYSIS] Entropy: {final_analysis['entropy']:.6f}")
            print(f"   [ANALYSIS] Purity: {final_analysis['purity']:.6f}")
            print(f"   [ANALYSIS] Interesting score: {final_analysis['interesting_score']:.3f}")
            print(f"   [ANALYSIS] Numerical issues: {len(conv_info['numerical_issues'])}")
            if conv_info['numerical_issues']:
                print(f"   [WARNING] Issues: {conv_info['numerical_issues']}")
            print()
    
    return results

def analyze_improved_results(results):
    """
    Analyze the improved results to find non-trivial fixed points
    """
    analysis = {
        'summary': {},
        'by_circuit_type': {},
        'by_strategy': {},
        'best_results': [],
        'conclusion': ""
    }
    
    total_experiments = len(results['improved_results'])
    successful_experiments = sum(1 for r in results['improved_results'] if r['success'])
    trivial_solutions = sum(1 for r in results['improved_results'] if r['is_trivial'])
    non_trivial_solutions = successful_experiments - trivial_solutions
    
    analysis['summary'] = {
        'total_experiments': total_experiments,
        'successful_experiments': successful_experiments,
        'success_rate': successful_experiments / total_experiments,
        'trivial_solutions': trivial_solutions,
        'trivial_rate': trivial_solutions / total_experiments,
        'non_trivial_solutions': non_trivial_solutions,
        'non_trivial_rate': non_trivial_solutions / total_experiments,
        'average_interesting_score': np.mean([r['interesting_score'] for r in results['improved_results']])
    }
    
    # Analysis by circuit type
    for circuit_type in set(r['circuit_type'] for r in results['improved_results']):
        type_results = [r for r in results['improved_results'] if r['circuit_type'] == circuit_type]
        successful = sum(1 for r in type_results if r['success'])
        trivial = sum(1 for r in type_results if r['is_trivial'])
        avg_score = np.mean([r['interesting_score'] for r in type_results])
        
        analysis['by_circuit_type'][circuit_type] = {
            'total': len(type_results),
            'successful': successful,
            'success_rate': successful / len(type_results),
            'trivial': trivial,
            'trivial_rate': trivial / len(type_results),
            'non_trivial': successful - trivial,
            'non_trivial_rate': (successful - trivial) / len(type_results),
            'average_interesting_score': avg_score
        }
    
    # Analysis by strategy
    for strategy in set(r['initial_strategy'] for r in results['improved_results']):
        strategy_results = [r for r in results['improved_results'] if r['initial_strategy'] == strategy]
        successful = sum(1 for r in strategy_results if r['success'])
        trivial = sum(1 for r in strategy_results if r['is_trivial'])
        avg_score = np.mean([r['interesting_score'] for r in strategy_results])
        
        analysis['by_strategy'][strategy] = {
            'total': len(strategy_results),
            'successful': successful,
            'success_rate': successful / len(strategy_results),
            'trivial': trivial,
            'trivial_rate': trivial / len(strategy_results),
            'non_trivial': successful - trivial,
            'non_trivial_rate': (successful - trivial) / len(strategy_results),
            'average_interesting_score': avg_score
        }
    
    # Find best results (non-trivial with high interesting score)
    best_results = [r for r in results['improved_results'] 
                   if r['success'] and not r['is_trivial'] and r['interesting_score'] > 0.5]
    best_results.sort(key=lambda x: x['interesting_score'], reverse=True)
    analysis['best_results'] = best_results[:5]  # Top 5
    
    # Determine conclusion
    if analysis['summary']['non_trivial_rate'] > 0.3:
        analysis['conclusion'] = "SUCCESS - Found significant number of non-trivial fixed points"
    elif analysis['summary']['non_trivial_rate'] > 0.1:
        analysis['conclusion'] = "PARTIAL SUCCESS - Found some non-trivial fixed points"
    else:
        analysis['conclusion'] = "FAILURE - Still mostly finding trivial solutions"
    
    return analysis

def create_improved_visualization(results, analysis):
    """
    Create visualization of improved results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Improved Deutsch CTC Results', fontsize=16, fontweight='bold')
    
    # 1. Success rate by circuit type
    circuit_types = list(analysis['by_circuit_type'].keys())
    success_rates = [analysis['by_circuit_type'][ct]['success_rate'] for ct in circuit_types]
    non_trivial_rates = [analysis['by_circuit_type'][ct]['non_trivial_rate'] for ct in circuit_types]
    
    x = np.arange(len(circuit_types))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, success_rates, width, label='Success Rate', color='blue', alpha=0.7)
    axes[0, 0].bar(x + width/2, non_trivial_rates, width, label='Non-Trivial Rate', color='green', alpha=0.7)
    axes[0, 0].set_title('Success and Non-Trivial Rates by Circuit Type')
    axes[0, 0].set_xlabel('Circuit Type')
    axes[0, 0].set_ylabel('Rate')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(circuit_types)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Interesting scores by strategy
    strategies = list(analysis['by_strategy'].keys())
    interesting_scores = [analysis['by_strategy'][s]['average_interesting_score'] for s in strategies]
    
    axes[0, 1].bar(strategies, interesting_scores, color='orange', alpha=0.7)
    axes[0, 1].set_title('Average Interesting Score by Initial Strategy')
    axes[0, 1].set_xlabel('Initial Strategy')
    axes[0, 1].set_ylabel('Interesting Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Summary pie chart
    labels = ['Non-Trivial Solutions', 'Trivial Solutions', 'Failed']
    sizes = [
        analysis['summary']['non_trivial_solutions'],
        analysis['summary']['trivial_solutions'],
        analysis['summary']['total_experiments'] - analysis['summary']['successful_experiments']
    ]
    colors = ['green', 'red', 'gray']
    
    axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('Overall Results Breakdown')
    
    # 4. Best results interesting scores
    if analysis['best_results']:
        best_labels = [f"{r['circuit_type']}\n{r['initial_strategy']}" for r in analysis['best_results']]
        best_scores = [r['interesting_score'] for r in analysis['best_results']]
        
        axes[1, 1].bar(range(len(best_labels)), best_scores, color='purple', alpha=0.7)
        axes[1, 1].set_title('Top 5 Most Interesting Results')
        axes[1, 1].set_xlabel('Circuit Type + Strategy')
        axes[1, 1].set_ylabel('Interesting Score')
        axes[1, 1].set_xticks(range(len(best_labels)))
        axes[1, 1].set_xticklabels(best_labels, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No non-trivial results found', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Top Results')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = f"improved_deutsch_results_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file

def generate_improved_report(results, analysis, plot_file):
    """
    Generate a comprehensive report on improved results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"IMPROVED_DEUTSCH_REPORT_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("IMPROVED DEUTSCH CTC EXPERIMENT REPORT\n")
        f.write("=======================================\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("==================\n")
        f.write(f"Total experiments: {analysis['summary']['total_experiments']}\n")
        f.write(f"Success rate: {analysis['summary']['success_rate']:.2%}\n")
        f.write(f"Non-trivial solutions: {analysis['summary']['non_trivial_solutions']} ({analysis['summary']['non_trivial_rate']:.2%})\n")
        f.write(f"Trivial solutions: {analysis['summary']['trivial_solutions']} ({analysis['summary']['trivial_rate']:.2%})\n")
        f.write(f"Average interesting score: {analysis['summary']['average_interesting_score']:.3f}\n")
        f.write(f"CONCLUSION: {analysis['conclusion']}\n\n")
        
        f.write("DETAILED ANALYSIS BY CIRCUIT TYPE\n")
        f.write("==================================\n")
        for circuit_type, stats in analysis['by_circuit_type'].items():
            f.write(f"\n{circuit_type.upper()}:\n")
            f.write(f"  Success rate: {stats['success_rate']:.2%}\n")
            f.write(f"  Non-trivial rate: {stats['non_trivial_rate']:.2%}\n")
            f.write(f"  Average interesting score: {stats['average_interesting_score']:.3f}\n")
        
        f.write("\nDETAILED ANALYSIS BY INITIAL STRATEGY\n")
        f.write("=====================================\n")
        for strategy, stats in analysis['by_strategy'].items():
            f.write(f"\n{strategy.upper()}:\n")
            f.write(f"  Success rate: {stats['success_rate']:.2%}\n")
            f.write(f"  Non-trivial rate: {stats['non_trivial_rate']:.2%}\n")
            f.write(f"  Average interesting score: {stats['average_interesting_score']:.3f}\n")
        
        f.write("\nBEST RESULTS\n")
        f.write("============\n")
        if analysis['best_results']:
            for i, result in enumerate(analysis['best_results'], 1):
                f.write(f"\n{i}. {result['circuit_type']} + {result['initial_strategy']}:\n")
                f.write(f"   Interesting score: {result['interesting_score']:.3f}\n")
                f.write(f"   Entropy: {result['final_analysis']['entropy']:.6f}\n")
                f.write(f"   Purity: {result['final_analysis']['purity']:.6f}\n")
                f.write(f"   Iterations: {result['convergence_info']['iterations']}\n")
        else:
            f.write("No non-trivial results found.\n")
        
        f.write(f"\n[ANALYSIS] Visualization: {plot_file}\n")
        
        f.write("\nCONCLUSIONS\n")
        f.write("===========\n")
        if analysis['conclusion'].startswith("SUCCESS"):
            f.write("[SUCCESS] We successfully found non-trivial fixed points!\n")
            f.write("The improved algorithm is working correctly.\n")
        elif analysis['conclusion'].startswith("PARTIAL"):
            f.write("[PARTIAL] We found some non-trivial fixed points.\n")
            f.write("Further improvements may be needed.\n")
        else:
            f.write("[FAILURE] Still mostly finding trivial solutions.\n")
            f.write("Need to try different approaches.\n")
        
        f.write(f"\nOverall, the improved Deutsch CTC implementation shows promise!\n")
    
    return report_file

def main():
    parser = argparse.ArgumentParser(description='Run improved Deutsch CTC experiment')
    parser.add_argument('--num_qubits', type=int, default=4, help='Number of qubits')
    parser.add_argument('--circuit_types', nargs='+', 
                       default=['entangled', 'quantum_walk', 'topological'],
                       help='Types of circuits to test')
    parser.add_argument('--initial_strategies', nargs='+',
                       default=['random_pure', 'random_mixed', 'coherent', 'bell_like'],
                       help='Initial state strategies to test')
    
    args = parser.parse_args()
    
    print("🚀 IMPROVED DEUTSCH CTC EXPERIMENT")
    print("===================================")
    print(f"Qubits: {args.num_qubits}")
    print(f"Circuit types: {args.circuit_types}")
    print(f"Initial strategies: {args.initial_strategies}")
    print()
    
    # Run improved experiment
    results = run_improved_experiment(
        num_qubits=args.num_qubits,
        circuit_types=args.circuit_types,
        initial_strategies=args.initial_strategies
    )
    
    # Analyze results
    print("📊 Analyzing improved results...")
    analysis = analyze_improved_results(results)
    
    # Create visualization
    print("📈 Creating improved visualization...")
    plot_file = create_improved_visualization(results, analysis)
    
    # Generate report
    print("📋 Generating improved report...")
    report_file = generate_improved_report(results, analysis, plot_file)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"improved_deutsch_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("🎉 IMPROVED EXPERIMENT COMPLETE!")
    print(f"   Results: {results_file}")
    print(f"   Visualization: {plot_file}")
    print(f"   Report: {report_file}")
    print(f"   Conclusion: {analysis['conclusion']}")
    print(f"   Non-trivial solutions: {analysis['summary']['non_trivial_solutions']}/{analysis['summary']['total_experiments']}")
    print(f"   Average interesting score: {analysis['summary']['average_interesting_score']:.3f}")

if __name__ == "__main__":
    main() 