#!/usr/bin/env python3
"""
FIND BEST INITIAL CONDITIONS: Systematic Search for Non-Trivial Fixed Points
============================================================================

This script systematically searches for the best initial conditions that lead
to non-trivial Deutsch fixed points. Based on our previous results, we focus on:

1. Quantum walk circuits (showed 50% non-trivial rate)
2. Systematic variation of initial state parameters
3. Different random seeds and distributions
4. Targeted search for high interesting scores

The goal is to find initial conditions that consistently produce non-trivial
fixed points with high interesting scores.
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

def create_quantum_walk_circuit(num_qubits, steps=3):
    """
    Create quantum walk circuit that showed promise in previous experiments
    """
    qc = QuantumCircuit(num_qubits)
    
    # Separate loop and bulk qubits
    loop_qubits = min(2, num_qubits // 2)
    bulk_qubits = num_qubits - loop_qubits
    
    circuit_info = {
        'type': 'quantum_walk',
        'steps': steps,
        'loop_qubits': loop_qubits,
        'bulk_qubits': bulk_qubits
    }
    
    # Create quantum walk structure
    for i in range(num_qubits):
        qc.h(i)
    
    # Create quantum walk structure
    for step in range(steps):
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
    
    return qc, circuit_info

def create_systematic_initial_state(dim_C, strategy, params=None):
    """
    Create initial states with systematic parameter variation
    """
    if params is None:
        params = {}
    
    if strategy == 'random_pure':
        # Random pure state with controlled randomness
        seed = params.get('seed', None)
        if seed is not None:
            np.random.seed(seed)
        
        # Control the distribution
        distribution = params.get('distribution', 'normal')
        if distribution == 'normal':
            random_state = np.random.randn(dim_C) + 1j * np.random.randn(dim_C)
        elif distribution == 'uniform':
            random_state = np.random.uniform(-1, 1, dim_C) + 1j * np.random.uniform(-1, 1, dim_C)
        elif distribution == 'sparse':
            # Create sparse random state
            random_state = np.zeros(dim_C, dtype=complex)
            num_nonzero = params.get('num_nonzero', dim_C // 2)
            indices = np.random.choice(dim_C, num_nonzero, replace=False)
            random_state[indices] = np.random.randn(num_nonzero) + 1j * np.random.randn(num_nonzero)
        
        random_state = random_state / np.linalg.norm(random_state)
        rho_C = np.outer(random_state, random_state.conj())
        return DensityMatrix(rho_C)
    
    elif strategy == 'random_mixed':
        # Random mixed state with controlled parameters
        seed = params.get('seed', None)
        if seed is not None:
            np.random.seed(seed)
        
        # Control the mixedness
        mixedness = params.get('mixedness', 0.5)  # 0 = pure, 1 = maximally mixed
        rho_C = random_density_matrix(dim_C)
        
        if mixedness < 1.0:
            # Make it less mixed by moving towards a pure state
            eigenvalues = np.real(np.linalg.eigvals(rho_C.data))
            # Increase the largest eigenvalue
            max_idx = np.argmax(eigenvalues)
            eigenvalues[max_idx] = eigenvalues[max_idx] + (1 - mixedness) * (1 - eigenvalues[max_idx])
            # Renormalize
            eigenvalues = eigenvalues / np.sum(eigenvalues)
            
            # Reconstruct density matrix
            U = np.linalg.eigh(rho_C.data)[1]
            rho_C = DensityMatrix(U @ np.diag(eigenvalues) @ U.conj().T)
        
        return rho_C
    
    elif strategy == 'coherent_superposition':
        # Create coherent superposition with controlled phases
        phases = params.get('phases', np.random.uniform(0, 2*np.pi, dim_C))
        amplitudes = params.get('amplitudes', np.ones(dim_C))
        
        state = amplitudes * np.exp(1j * phases)
        state = state / np.linalg.norm(state)
        rho_C = np.outer(state, state.conj())
        return DensityMatrix(rho_C)
    
    elif strategy == 'bell_like':
        # Create Bell-like state with controlled entanglement
        entanglement = params.get('entanglement', 0.5)  # 0 = separable, 1 = maximally entangled
        
        if dim_C == 4:  # 2 qubits
            # Create |00⟩ + α|11⟩ state
            alpha = np.sqrt(entanglement)
            state = np.array([1.0, 0.0, 0.0, alpha]) / np.sqrt(1 + alpha**2)
            rho_C = np.outer(state, state.conj())
            return DensityMatrix(rho_C)
        else:
            # Fall back to coherent superposition
            return create_systematic_initial_state(dim_C, 'coherent_superposition', params)
    
    elif strategy == 'thermal':
        # Create thermal state with controlled temperature
        temperature = params.get('temperature', 1.0)  # Higher = more mixed
        
        if temperature == 0:
            # Pure ground state
            state = np.zeros(dim_C)
            state[0] = 1.0
            rho_C = np.outer(state, state.conj())
        else:
            # Thermal state
            energies = np.arange(dim_C)
            boltzmann = np.exp(-energies / temperature)
            rho_C = np.diag(boltzmann / np.sum(boltzmann))
        
        return DensityMatrix(rho_C)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def deutsch_fixed_point_iteration_systematic(qc, loop_qubits, initial_strategy, initial_params, max_iters=50, tol=1e-6):
    """
    Systematic Deutsch fixed-point iteration with detailed tracking
    """
    print(f"[SYSTEMATIC] Starting systematic fixed-point iteration")
    print(f"[SYSTEMATIC] Initial strategy: {initial_strategy}")
    print(f"[SYSTEMATIC] Initial params: {initial_params}")
    
    # Get the unitary matrix for the circuit
    U_mat = Operator(qc).data
    
    # Number of S-qubits (bulk):
    n_S = qc.num_qubits - len(loop_qubits)
    dim_S = 2**n_S
    
    # Number of C-qubits (loop):
    n_C = len(loop_qubits)
    dim_C = 2**n_C
    
    # Initialize ρ_S as maximally mixed on the bulk
    rho_S = DensityMatrix(np.eye(dim_S) / dim_S)
    
    # Initialize ρ_C with systematic state
    rho_C = create_systematic_initial_state(dim_C, initial_strategy, initial_params)
    
    convergence_info = {
        'iterations': 0,
        'converged': False,
        'final_fidelity': 0.0,
        'fidelity_history': [],
        'density_matrices': [],
        'entropies': [],
        'purities': [],
        'numerical_issues': [],
        'initial_strategy': initial_strategy,
        'initial_params': initial_params
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
        
        if abs(fidelity - 1) < tol:
            convergence_info['converged'] = True
            convergence_info['final_fidelity'] = fidelity
            convergence_info['iterations'] = iteration + 1
            break
        
        rho_C = new_rho_C
    
    if not convergence_info['converged']:
        convergence_info['iterations'] = max_iters
        convergence_info['final_fidelity'] = convergence_info['fidelity_history'][-1]
    
    return rho_C, convergence_info

def analyze_density_matrix_systematic(rho_C, iteration_num):
    """
    Analyze a density matrix with systematic metrics
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
        'interesting_score': 0.0,
        'non_triviality_score': 0.0
    }
    
    # Check if it's maximally mixed (trivial)
    dim = rho_C.shape[0]
    maximally_mixed = np.eye(dim) / dim
    if np.allclose(rho_C, maximally_mixed, atol=1e-10):
        analysis['is_maximally_mixed'] = True
        analysis['is_trivial'] = True
        analysis['interesting_score'] = 0.0
        analysis['non_triviality_score'] = 0.0
    
    # Check if it's pure (eigenvalue = 1, rest = 0)
    eigenvalues = np.real(np.linalg.eigvals(rho_C))
    if np.any(eigenvalues > 0.999) and np.sum(eigenvalues > 0.001) == 1:
        analysis['is_pure'] = True
        analysis['interesting_score'] = 1.0
        analysis['non_triviality_score'] = 1.0
    
    # Calculate interesting score based on entropy and purity
    if not analysis['is_trivial']:
        # Higher score for intermediate entropy (not too mixed, not too pure)
        entropy_score = 1.0 - abs(analysis['entropy'] - np.log2(dim)/2) / (np.log2(dim)/2)
        purity_score = analysis['purity']  # Higher purity is more interesting
        analysis['interesting_score'] = (entropy_score + purity_score) / 2
        
        # Non-triviality score: how far from maximally mixed
        max_entropy = np.log2(dim)
        analysis['non_triviality_score'] = 1.0 - analysis['entropy'] / max_entropy
    
    # Check for numerical issues
    if np.any(np.isnan(rho_C)) or np.any(np.isinf(rho_C)):
        analysis['numerical_issues'].append("NaN/Inf detected")
    
    if not np.allclose(rho_C, rho_C.conj().T):
        analysis['numerical_issues'].append("Non-Hermitian")
    
    if not np.isclose(np.trace(rho_C), 1.0, atol=1e-10):
        analysis['numerical_issues'].append("Trace != 1")
    
    return analysis

def run_systematic_search(num_qubits=4, num_trials=20):
    """
    Run systematic search for best initial conditions
    """
    results = {
        'experiment_info': {
            'num_qubits': num_qubits,
            'num_trials': num_trials,
            'timestamp': datetime.now().isoformat(),
            'experiment_type': 'systematic_initial_conditions'
        },
        'systematic_results': []
    }
    
    print(f"🔍 SYSTEMATIC SEARCH FOR BEST INITIAL CONDITIONS")
    print(f"================================================")
    print(f"Qubits: {num_qubits}")
    print(f"Trials: {num_trials}")
    print()
    
    # Create quantum walk circuit
    qc, circuit_info = create_quantum_walk_circuit(num_qubits)
    loop_qubits = list(range(circuit_info['loop_qubits']))
    
    # Remove measurements for fixed-point iteration
    qc_no_measure = qc.copy()
    qc_no_measure.remove_final_measurements(inplace=True)
    
    # Define systematic parameter variations
    strategies_and_params = [
        # Random pure with different distributions
        ('random_pure', {'distribution': 'normal', 'seed': None}),
        ('random_pure', {'distribution': 'uniform', 'seed': None}),
        ('random_pure', {'distribution': 'sparse', 'num_nonzero': 2, 'seed': None}),
        ('random_pure', {'distribution': 'sparse', 'num_nonzero': 1, 'seed': None}),
        
        # Random mixed with different mixedness
        ('random_mixed', {'mixedness': 0.1, 'seed': None}),
        ('random_mixed', {'mixedness': 0.3, 'seed': None}),
        ('random_mixed', {'mixedness': 0.5, 'seed': None}),
        ('random_mixed', {'mixedness': 0.7, 'seed': None}),
        
        # Coherent superposition with different phases
        ('coherent_superposition', {'phases': np.array([0, np.pi/2, np.pi, 3*np.pi/2]), 'amplitudes': np.array([1, 1, 1, 1])}),
        ('coherent_superposition', {'phases': np.array([0, np.pi/4, np.pi/2, 3*np.pi/4]), 'amplitudes': np.array([1, 0.5, 0.5, 1])}),
        
        # Bell-like with different entanglement
        ('bell_like', {'entanglement': 0.2}),
        ('bell_like', {'entanglement': 0.5}),
        ('bell_like', {'entanglement': 0.8}),
        ('bell_like', {'entanglement': 1.0}),
        
        # Thermal states with different temperatures
        ('thermal', {'temperature': 0.1}),
        ('thermal', {'temperature': 0.5}),
        ('thermal', {'temperature': 1.0}),
        ('thermal', {'temperature': 2.0}),
    ]
    
    trial_count = 0
    for strategy, params in strategies_and_params:
        for trial in range(num_trials // len(strategies_and_params)):
            trial_count += 1
            
            # Add random seed for this trial
            trial_params = params.copy()
            if 'seed' in trial_params and trial_params['seed'] is None:
                trial_params['seed'] = np.random.randint(0, 10000)
            
            print(f"🔬 Trial {trial_count}/{num_trials}: {strategy} with params {trial_params}")
            
            # Run systematic Deutsch fixed-point iteration
            rho_C, conv_info = deutsch_fixed_point_iteration_systematic(
                qc_no_measure, loop_qubits, strategy, trial_params
            )
            
            # Analyze the final density matrix
            final_analysis = analyze_density_matrix_systematic(rho_C.data, conv_info['iterations'])
            
            # Store results
            result = {
                'trial': trial_count,
                'strategy': strategy,
                'params': trial_params,
                'circuit_info': circuit_info,
                'convergence_info': conv_info,
                'final_analysis': final_analysis,
                'success': conv_info['converged'],
                'is_trivial': final_analysis['is_trivial'],
                'interesting_score': final_analysis['interesting_score'],
                'non_triviality_score': final_analysis['non_triviality_score'],
                'numerical_issues': len(conv_info['numerical_issues']) > 0
            }
            
            results['systematic_results'].append(result)
            
            print(f"   [SUCCESS] Converged: {conv_info['converged']}")
            print(f"   [ANALYSIS] Fidelity: {conv_info['final_fidelity']:.10f}")
            print(f"   [ANALYSIS] Iterations: {conv_info['iterations']}")
            print(f"   [ANALYSIS] Is trivial: {final_analysis['is_trivial']}")
            print(f"   [ANALYSIS] Entropy: {final_analysis['entropy']:.6f}")
            print(f"   [ANALYSIS] Purity: {final_analysis['purity']:.6f}")
            print(f"   [ANALYSIS] Interesting score: {final_analysis['interesting_score']:.3f}")
            print(f"   [ANALYSIS] Non-triviality score: {final_analysis['non_triviality_score']:.3f}")
            print()
    
    return results

def analyze_systematic_results(results):
    """
    Analyze systematic search results to find best initial conditions
    """
    analysis = {
        'summary': {},
        'by_strategy': {},
        'best_conditions': [],
        'conclusion': ""
    }
    
    total_experiments = len(results['systematic_results'])
    successful_experiments = sum(1 for r in results['systematic_results'] if r['success'])
    trivial_solutions = sum(1 for r in results['systematic_results'] if r['is_trivial'])
    non_trivial_solutions = successful_experiments - trivial_solutions
    
    analysis['summary'] = {
        'total_experiments': total_experiments,
        'successful_experiments': successful_experiments,
        'success_rate': successful_experiments / total_experiments,
        'trivial_solutions': trivial_solutions,
        'trivial_rate': trivial_solutions / total_experiments,
        'non_trivial_solutions': non_trivial_solutions,
        'non_trivial_rate': non_trivial_solutions / total_experiments,
        'average_interesting_score': np.mean([r['interesting_score'] for r in results['systematic_results']]),
        'average_non_triviality_score': np.mean([r['non_triviality_score'] for r in results['systematic_results']])
    }
    
    # Analysis by strategy
    for strategy in set(r['strategy'] for r in results['systematic_results']):
        strategy_results = [r for r in results['systematic_results'] if r['strategy'] == strategy]
        successful = sum(1 for r in strategy_results if r['success'])
        trivial = sum(1 for r in strategy_results if r['is_trivial'])
        avg_interesting = np.mean([r['interesting_score'] for r in strategy_results])
        avg_non_triviality = np.mean([r['non_triviality_score'] for r in strategy_results])
        
        analysis['by_strategy'][strategy] = {
            'total': len(strategy_results),
            'successful': successful,
            'success_rate': successful / len(strategy_results),
            'trivial': trivial,
            'trivial_rate': trivial / len(strategy_results),
            'non_trivial': successful - trivial,
            'non_trivial_rate': (successful - trivial) / len(strategy_results),
            'average_interesting_score': avg_interesting,
            'average_non_triviality_score': avg_non_triviality
        }
    
    # Find best conditions (non-trivial with high scores)
    best_results = [r for r in results['systematic_results'] 
                   if r['success'] and not r['is_trivial'] and 
                   r['interesting_score'] > 0.3 and r['non_triviality_score'] > 0.3]
    best_results.sort(key=lambda x: x['interesting_score'] + x['non_triviality_score'], reverse=True)
    analysis['best_conditions'] = best_results[:10]  # Top 10
    
    # Determine conclusion
    if analysis['summary']['non_trivial_rate'] > 0.4:
        analysis['conclusion'] = "EXCELLENT - Found many non-trivial fixed points"
    elif analysis['summary']['non_trivial_rate'] > 0.2:
        analysis['conclusion'] = "GOOD - Found significant number of non-trivial fixed points"
    elif analysis['summary']['non_trivial_rate'] > 0.1:
        analysis['conclusion'] = "MODERATE - Found some non-trivial fixed points"
    else:
        analysis['conclusion'] = "POOR - Still mostly finding trivial solutions"
    
    return analysis

def create_systematic_visualization(results, analysis):
    """
    Create visualization of systematic search results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Systematic Search for Best Initial Conditions', fontsize=16, fontweight='bold')
    
    # 1. Success and non-trivial rates by strategy
    strategies = list(analysis['by_strategy'].keys())
    success_rates = [analysis['by_strategy'][s]['success_rate'] for s in strategies]
    non_trivial_rates = [analysis['by_strategy'][s]['non_trivial_rate'] for s in strategies]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, success_rates, width, label='Success Rate', color='blue', alpha=0.7)
    axes[0, 0].bar(x + width/2, non_trivial_rates, width, label='Non-Trivial Rate', color='green', alpha=0.7)
    axes[0, 0].set_title('Success and Non-Trivial Rates by Strategy')
    axes[0, 0].set_xlabel('Strategy')
    axes[0, 0].set_ylabel('Rate')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(strategies, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Average scores by strategy
    interesting_scores = [analysis['by_strategy'][s]['average_interesting_score'] for s in strategies]
    non_triviality_scores = [analysis['by_strategy'][s]['average_non_triviality_score'] for s in strategies]
    
    axes[0, 1].bar(x - width/2, interesting_scores, width, label='Interesting Score', color='orange', alpha=0.7)
    axes[0, 1].bar(x + width/2, non_triviality_scores, width, label='Non-Triviality Score', color='red', alpha=0.7)
    axes[0, 1].set_title('Average Scores by Strategy')
    axes[0, 1].set_xlabel('Strategy')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(strategies, rotation=45, ha='right')
    axes[0, 1].legend()
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
    
    # 4. Best conditions scatter plot
    if analysis['best_conditions']:
        interesting_scores = [r['interesting_score'] for r in analysis['best_conditions']]
        non_triviality_scores = [r['non_triviality_score'] for r in analysis['best_conditions']]
        strategies = [r['strategy'] for r in analysis['best_conditions']]
        
        # Color by strategy
        unique_strategies = list(set(strategies))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_strategies)))
        color_map = dict(zip(unique_strategies, colors))
        
        for i, (x, y, strategy) in enumerate(zip(interesting_scores, non_triviality_scores, strategies)):
            axes[1, 1].scatter(x, y, c=[color_map[strategy]], s=100, alpha=0.7, label=strategy if strategy not in [s[0] for s in axes[1, 1].get_children() if hasattr(s, 'get_label')] else "")
        
        axes[1, 1].set_xlabel('Interesting Score')
        axes[1, 1].set_ylabel('Non-Triviality Score')
        axes[1, 1].set_title('Best Initial Conditions')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        axes[1, 1].text(0.5, 0.5, 'No good conditions found', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Best Conditions')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = f"systematic_search_results_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file

def generate_systematic_report(results, analysis, plot_file):
    """
    Generate comprehensive report on systematic search
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"SYSTEMATIC_SEARCH_REPORT_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("SYSTEMATIC SEARCH FOR BEST INITIAL CONDITIONS REPORT\n")
        f.write("====================================================\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("==================\n")
        f.write(f"Total experiments: {analysis['summary']['total_experiments']}\n")
        f.write(f"Success rate: {analysis['summary']['success_rate']:.2%}\n")
        f.write(f"Non-trivial solutions: {analysis['summary']['non_trivial_solutions']} ({analysis['summary']['non_trivial_rate']:.2%})\n")
        f.write(f"Trivial solutions: {analysis['summary']['trivial_solutions']} ({analysis['summary']['trivial_rate']:.2%})\n")
        f.write(f"Average interesting score: {analysis['summary']['average_interesting_score']:.3f}\n")
        f.write(f"Average non-triviality score: {analysis['summary']['average_non_triviality_score']:.3f}\n")
        f.write(f"CONCLUSION: {analysis['conclusion']}\n\n")
        
        f.write("DETAILED ANALYSIS BY STRATEGY\n")
        f.write("============================\n")
        for strategy, stats in analysis['by_strategy'].items():
            f.write(f"\n{strategy.upper()}:\n")
            f.write(f"  Success rate: {stats['success_rate']:.2%}\n")
            f.write(f"  Non-trivial rate: {stats['non_trivial_rate']:.2%}\n")
            f.write(f"  Average interesting score: {stats['average_interesting_score']:.3f}\n")
            f.write(f"  Average non-triviality score: {stats['average_non_triviality_score']:.3f}\n")
        
        f.write("\nBEST INITIAL CONDITIONS FOUND\n")
        f.write("=============================\n")
        if analysis['best_conditions']:
            for i, result in enumerate(analysis['best_conditions'], 1):
                f.write(f"\n{i}. {result['strategy']} with params {result['params']}:\n")
                f.write(f"   Interesting score: {result['interesting_score']:.3f}\n")
                f.write(f"   Non-triviality score: {result['non_triviality_score']:.3f}\n")
                f.write(f"   Entropy: {result['final_analysis']['entropy']:.6f}\n")
                f.write(f"   Purity: {result['final_analysis']['purity']:.6f}\n")
                f.write(f"   Iterations: {result['convergence_info']['iterations']}\n")
        else:
            f.write("No good conditions found.\n")
        
        f.write(f"\n[ANALYSIS] Visualization: {plot_file}\n")
        
        f.write("\nRECOMMENDATIONS\n")
        f.write("===============\n")
        if analysis['best_conditions']:
            f.write("RECOMMENDED INITIAL CONDITIONS:\n")
            for i, result in enumerate(analysis['best_conditions'][:3], 1):
                f.write(f"{i}. Strategy: {result['strategy']}\n")
                f.write(f"   Parameters: {result['params']}\n")
                f.write(f"   Combined Score: {result['interesting_score'] + result['non_triviality_score']:.3f}\n")
        else:
            f.write("No specific recommendations - need to try different approaches.\n")
        
        f.write(f"\nOverall, the systematic search provides clear guidance for future experiments!\n")
    
    return report_file

def main():
    parser = argparse.ArgumentParser(description='Run systematic search for best initial conditions')
    parser.add_argument('--num_qubits', type=int, default=4, help='Number of qubits')
    parser.add_argument('--num_trials', type=int, default=20, help='Number of trials per strategy')
    
    args = parser.parse_args()
    
    print("🔍 SYSTEMATIC SEARCH FOR BEST INITIAL CONDITIONS")
    print("================================================")
    print(f"Qubits: {args.num_qubits}")
    print(f"Trials: {args.num_trials}")
    print()
    
    # Run systematic search
    results = run_systematic_search(
        num_qubits=args.num_qubits,
        num_trials=args.num_trials
    )
    
    # Analyze results
    print("📊 Analyzing systematic results...")
    analysis = analyze_systematic_results(results)
    
    # Create visualization
    print("📈 Creating systematic visualization...")
    plot_file = create_systematic_visualization(results, analysis)
    
    # Generate report
    print("📋 Generating systematic report...")
    report_file = generate_systematic_report(results, analysis, plot_file)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"systematic_search_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("🎉 SYSTEMATIC SEARCH COMPLETE!")
    print(f"   Results: {results_file}")
    print(f"   Visualization: {plot_file}")
    print(f"   Report: {report_file}")
    print(f"   Conclusion: {analysis['conclusion']}")
    print(f"   Non-trivial solutions: {analysis['summary']['non_trivial_solutions']}/{analysis['summary']['total_experiments']}")
    print(f"   Average interesting score: {analysis['summary']['average_interesting_score']:.3f}")
    print(f"   Average non-triviality score: {analysis['summary']['average_non_triviality_score']:.3f}")
    
    if analysis['best_conditions']:
        print(f"\n🏆 TOP 3 RECOMMENDED INITIAL CONDITIONS:")
        for i, result in enumerate(analysis['best_conditions'][:3], 1):
            print(f"   {i}. {result['strategy']} with params {result['params']}")
            print(f"      Combined score: {result['interesting_score'] + result['non_triviality_score']:.3f}")

if __name__ == "__main__":
    main() 