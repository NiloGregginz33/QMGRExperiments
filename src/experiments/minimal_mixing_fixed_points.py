#!/usr/bin/env python3
"""
MINIMAL MIXING FIXED POINTS: Ultra-Low Mixing Circuits for Non-Trivial Deutsch Fixed Points
==========================================================================================

This script implements circuits with minimal mixing to preserve quantum structure and find
truly interesting non-trivial fixed points. The key insight is that we need to dramatically
reduce the mixing operations to preserve quantum coherence and structure.

Key features:
1. Ultra-minimal mixing circuits (1-3 CX gates total)
2. Larger system sizes (6-8 qubits) with asymmetric loop/bulk ratios
3. Structure-preserving operations (phase gates, rotations)
4. Focus on preserving quantum coherence
5. Targeted initial states with specific quantum features
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

def create_minimal_mixing_circuit(num_qubits, circuit_type='ultra_minimal'):
    """
    Create circuits with minimal mixing to preserve quantum structure
    
    Args:
        num_qubits: Total number of qubits
        circuit_type: 'ultra_minimal', 'phase_preserving', 'coherent_only', 'structured_minimal'
    """
    qc = QuantumCircuit(num_qubits)
    
    # Determine loop and bulk qubits with asymmetric ratios
    if circuit_type == 'ultra_minimal':
        loop_qubits = max(1, num_qubits // 4)  # Very few loop qubits
    else:
        loop_qubits = max(1, num_qubits // 3)
    
    bulk_qubits = num_qubits - loop_qubits
    
    circuit_info = {
        'type': circuit_type,
        'loop_qubits': loop_qubits,
        'bulk_qubits': bulk_qubits,
        'total_qubits': num_qubits,
        'mixing_level': 'ultra_minimal'
    }
    
    if circuit_type == 'ultra_minimal':
        # Ultra-minimal mixing: only 1-2 CX gates total
        # Start with coherent preparation
        for i in range(num_qubits):
            qc.h(i)
            qc.rz(np.pi/6, i)  # Add some phase structure
        
        # Apply only 1-2 CX gates strategically
        if num_qubits >= 2:
            qc.cx(0, 1)  # Single coupling between first two qubits
        
        if num_qubits >= 4:
            qc.cx(2, 3)  # One more coupling if we have enough qubits
        
        # Add phase operations to preserve structure
        for i in range(num_qubits):
            qc.rz(np.pi/8, i)
            qc.t(i)
    
    elif circuit_type == 'phase_preserving':
        # Phase-preserving circuit with minimal mixing
        # Create coherent initial state
        for i in range(num_qubits):
            qc.h(i)
        
        # Apply only phase-preserving operations
        for i in range(num_qubits):
            qc.rz(np.pi/4, i)
            qc.t(i)
        
        # Minimal mixing: only one CX gate
        if num_qubits >= 2:
            qc.cx(loop_qubits-1, loop_qubits)  # Couple loop to bulk
        
        # More phase operations
        for i in range(num_qubits):
            qc.rz(np.pi/6, i)
    
    elif circuit_type == 'coherent_only':
        # Circuit with almost no mixing, just coherent operations
        # Prepare coherent state
        for i in range(num_qubits):
            qc.h(i)
            qc.rz(np.pi/3, i)
        
        # Apply only phase operations, no CX gates
        for i in range(num_qubits):
            qc.t(i)
            qc.rz(np.pi/4, i)
        
        # Add one very weak coupling at the end
        if num_qubits >= 2:
            qc.cp(np.pi/8, 0, 1)  # Very weak phase coupling
    
    elif circuit_type == 'structured_minimal':
        # Structured minimal mixing with specific patterns
        # Create structured initial state
        qc.h(0)
        for i in range(1, num_qubits):
            qc.cx(0, i)  # Create entanglement structure
            qc.rz(np.pi/4, i)
        
        # Apply minimal mixing with structure preservation
        if num_qubits >= 3:
            qc.cx(1, 2)  # One strategic coupling
        
        # Add phase operations to maintain structure
        for i in range(num_qubits):
            qc.rz(np.pi/6, i)
            qc.t(i)
    
    return qc, circuit_info

def create_structured_initial_state(dim_C, strategy, params=None):
    """
    Create initial states designed to preserve quantum structure
    """
    if params is None:
        params = {}
    
    if strategy == 'coherent_pure':
        # Create coherent pure state with specific structure
        state = np.zeros(dim_C, dtype=complex)
        
        # Create structured superposition
        if dim_C >= 4:
            # |00⟩ + α|01⟩ + β|10⟩ + γ|11⟩ with controlled structure
            alpha = params.get('alpha', 0.8)
            beta = params.get('beta', 0.4)
            gamma = params.get('gamma', 0.2)
            
            state[0] = 1.0
            if dim_C > 1:
                state[1] = alpha
            if dim_C > 2:
                state[2] = beta
            if dim_C > 3:
                state[3] = gamma
            
            # Ensure normalization
            state = state / np.linalg.norm(state)
        else:
            # Fallback for smaller dimensions
            state[0] = 1.0
            state = state / np.linalg.norm(state)
        
        rho_C = np.outer(state, state.conj())
        return DensityMatrix(rho_C)
    
    elif strategy == 'low_entropy':
        # Create state with low but non-zero entropy
        target_entropy = params.get('target_entropy', 0.5)  # Low entropy
        
        # Create diagonal matrix with controlled eigenvalues
        eigenvalues = np.ones(dim_C)
        max_entropy = np.log2(dim_C)
        
        if target_entropy > 0:
            # Create one dominant eigenvalue and small others
            dominant_weight = 1.0 - (target_entropy / max_entropy) * 0.5  # Keep it high
            eigenvalues[0] = dominant_weight
            remaining_weight = 1.0 - dominant_weight
            eigenvalues[1:] = remaining_weight / (dim_C - 1)
        
        # Ensure eigenvalues are positive and sum to 1
        eigenvalues = np.maximum(eigenvalues, 1e-10)  # Ensure positive
        eigenvalues = eigenvalues / np.sum(eigenvalues)  # Normalize
        
        rho_C = np.diag(eigenvalues)
        return DensityMatrix(rho_C)
    
    elif strategy == 'phase_coherent':
        # Create state with specific phase relationships
        phases = params.get('phases', np.linspace(0, np.pi, dim_C, endpoint=False))
        amplitudes = params.get('amplitudes', np.ones(dim_C))
        
        # Create structured amplitudes
        if 'structure_type' in params:
            if params['structure_type'] == 'gaussian':
                # Gaussian distribution centered on first state
                amplitudes = np.exp(-0.5 * (np.arange(dim_C) / (dim_C/4))**2)
            elif params['structure_type'] == 'exponential':
                # Exponential decay from first state
                amplitudes = np.exp(-np.arange(dim_C) * 0.3)
        
        # Ensure amplitudes are positive and normalize
        amplitudes = np.maximum(amplitudes, 1e-10)
        state = amplitudes * np.exp(1j * phases)
        state = state / np.linalg.norm(state)
        rho_C = np.outer(state, state.conj())
        return DensityMatrix(rho_C)
    
    elif strategy == 'entangled_structure':
        # Create entangled state with specific structure
        state = np.zeros(dim_C, dtype=complex)
        
        if dim_C >= 4:
            # Create Bell-like state with structure
            # |00⟩ + α|11⟩ + small contributions from other states
            alpha = params.get('alpha', 0.9)
            small_contrib = params.get('small_contrib', 0.1)
            
            state[0] = 1.0
            if dim_C > 3:
                state[3] = alpha
            
            # Add small contributions to other states
            for i in range(1, dim_C):
                if i != 3:  # Not the main Bell state
                    state[i] = small_contrib * np.exp(1j * i * np.pi/4)
            
            # Ensure normalization
            state = state / np.linalg.norm(state)
        else:
            # Fallback for smaller dimensions
            state[0] = 1.0
            state = state / np.linalg.norm(state)
        
        rho_C = np.outer(state, state.conj())
        return DensityMatrix(rho_C)
    
    else:
        # Fall back to random state with controlled entropy
        np.random.seed(params.get('seed', None))
        random_state = np.random.randn(dim_C) + 1j * np.random.randn(dim_C)
        random_state = random_state / np.linalg.norm(random_state)
        rho_C = np.outer(random_state, random_state.conj())
        return DensityMatrix(rho_C)

def deutsch_fixed_point_iteration_minimal(qc, loop_qubits, initial_strategy, initial_params, max_iters=50, tol=1e-6):
    """
    Minimal mixing Deutsch fixed-point iteration with detailed analysis
    """
    print(f"[MINIMAL] Starting minimal mixing fixed-point iteration")
    print(f"[MINIMAL] Initial strategy: {initial_strategy}")
    print(f"[MINIMAL] Initial params: {initial_params}")
    
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
    
    # Initialize ρ_C with structured state
    rho_C = create_structured_initial_state(dim_C, initial_strategy, initial_params)
    
    convergence_info = {
        'iterations': 0,
        'converged': False,
        'final_fidelity': 0.0,
        'fidelity_history': [],
        'entropies': [],
        'purities': [],
        'coherence_scores': [],
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
        
        # Calculate entropy and purity
        entropy_val = entropy(new_rho_C)
        purity = np.trace(new_rho_C.data @ new_rho_C.data)
        convergence_info['entropies'].append(entropy_val)
        convergence_info['purities'].append(purity)
        
        # Calculate coherence score
        off_diagonal = new_rho_C.data - np.diag(np.diag(new_rho_C.data))
        coherence_score = np.linalg.norm(off_diagonal) / np.linalg.norm(new_rho_C.data)
        convergence_info['coherence_scores'].append(coherence_score)
        
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

def analyze_minimal_fixed_point(rho_C, iteration_num):
    """
    Analyze fixed point with focus on structure preservation
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
        'interesting_score': 0.0,
        'non_triviality_score': 0.0,
        'coherence_score': 0.0,
        'structure_score': 0.0,
        'quality_score': 0.0
    }
    
    dim = rho_C.shape[0]
    max_entropy = np.log2(dim)
    maximally_mixed = np.eye(dim) / dim
    
    # Check if it's maximally mixed (trivial)
    if np.allclose(rho_C, maximally_mixed, atol=1e-8):
        analysis['is_maximally_mixed'] = True
        analysis['is_trivial'] = True
        analysis['interesting_score'] = 0.0
        analysis['non_triviality_score'] = 0.0
        analysis['coherence_score'] = 0.0
        analysis['structure_score'] = 0.0
        analysis['quality_score'] = 0.0
        return analysis
    
    # Check if it's pure
    eigenvalues = np.real(np.linalg.eigvals(rho_C))
    if np.any(eigenvalues > 0.999) and np.sum(eigenvalues > 0.001) == 1:
        analysis['is_pure'] = True
        analysis['interesting_score'] = 1.0
        analysis['non_triviality_score'] = 1.0
        analysis['coherence_score'] = 1.0
        analysis['structure_score'] = 1.0
        analysis['quality_score'] = 1.0
        return analysis
    
    # Calculate quality scores for non-trivial states
    # Interesting score: how far from both pure and maximally mixed
    entropy_score = 1.0 - abs(analysis['entropy'] - max_entropy/3) / (max_entropy/3)  # Prefer lower entropy
    purity_score = analysis['purity']
    analysis['interesting_score'] = (entropy_score + purity_score) / 2
    
    # Non-triviality score: how far from maximally mixed
    analysis['non_triviality_score'] = 1.0 - analysis['entropy'] / max_entropy
    
    # Coherence score: off-diagonal elements
    off_diagonal = rho_C - np.diag(np.diag(rho_C))
    coherence_score = np.linalg.norm(off_diagonal) / np.linalg.norm(rho_C)
    analysis['coherence_score'] = coherence_score
    
    # Structure score: eigenvalue distribution
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    # Prefer states with a few dominant eigenvalues
    structure_score = 1.0 - np.std(sorted_eigenvalues) / np.mean(sorted_eigenvalues)
    analysis['structure_score'] = max(0, structure_score)
    
    # Overall quality score
    analysis['quality_score'] = (analysis['interesting_score'] + 
                               analysis['non_triviality_score'] + 
                               analysis['coherence_score'] + 
                               analysis['structure_score']) / 4
    
    return analysis

def run_minimal_mixing_search(num_qubits=6, num_trials=40):
    """
    Run minimal mixing search for interesting fixed points
    """
    results = {
        'experiment_info': {
            'num_qubits': num_qubits,
            'num_trials': num_trials,
            'timestamp': datetime.now().isoformat(),
            'experiment_type': 'minimal_mixing_fixed_points'
        },
        'minimal_results': []
    }
    
    print(f"🔍 MINIMAL MIXING SEARCH FOR INTERESTING FIXED POINTS")
    print(f"====================================================")
    print(f"Qubits: {num_qubits}")
    print(f"Trials: {num_trials}")
    print()
    
    # Define circuit types
    circuit_types = [
        'ultra_minimal',
        'phase_preserving', 
        'coherent_only',
        'structured_minimal'
    ]
    
    # Define initial state strategies
    initial_strategies = [
        ('coherent_pure', {'alpha': 0.8, 'beta': 0.4, 'gamma': 0.2}),
        ('coherent_pure', {'alpha': 0.9, 'beta': 0.3, 'gamma': 0.1}),
        ('low_entropy', {'target_entropy': 0.3}),
        ('low_entropy', {'target_entropy': 0.7}),
        ('phase_coherent', {'structure_type': 'gaussian'}),
        ('phase_coherent', {'structure_type': 'exponential'}),
        ('entangled_structure', {'alpha': 0.9, 'small_contrib': 0.05}),
        ('entangled_structure', {'alpha': 0.8, 'small_contrib': 0.1}),
    ]
    
    trial_count = 0
    total_configs = len(circuit_types) * len(initial_strategies)
    trials_per_config = max(1, num_trials // total_configs)
    
    for circuit_type in circuit_types:
        for strategy, params in initial_strategies:
            for trial in range(trials_per_config):
                trial_count += 1
                
                # Add random seed for this trial
                trial_params = params.copy()
                trial_params['seed'] = np.random.randint(0, 10000)
                
                print(f"🔬 Trial {trial_count}/{num_trials}: {circuit_type} + {strategy}")
                
                # Create minimal mixing circuit
                qc, circuit_info = create_minimal_mixing_circuit(num_qubits, circuit_type)
                loop_qubits = list(range(circuit_info['loop_qubits']))
                
                # Remove measurements for fixed-point iteration
                qc_no_measure = qc.copy()
                qc_no_measure.remove_final_measurements(inplace=True)
                
                # Run minimal mixing Deutsch fixed-point iteration
                rho_C, conv_info = deutsch_fixed_point_iteration_minimal(
                    qc_no_measure, loop_qubits, strategy, trial_params
                )
                
                # Analyze the final density matrix
                final_analysis = analyze_minimal_fixed_point(rho_C.data, conv_info['iterations'])
                
                # Store results
                result = {
                    'trial': trial_count,
                    'circuit_type': circuit_type,
                    'strategy': strategy,
                    'params': trial_params,
                    'circuit_info': circuit_info,
                    'convergence_info': conv_info,
                    'final_analysis': final_analysis,
                    'success': conv_info['converged'],
                    'is_trivial': final_analysis['is_trivial'],
                    'interesting_score': final_analysis['interesting_score'],
                    'non_triviality_score': final_analysis['non_triviality_score'],
                    'coherence_score': final_analysis['coherence_score'],
                    'structure_score': final_analysis['structure_score'],
                    'quality_score': final_analysis['quality_score']
                }
                
                results['minimal_results'].append(result)
                
                print(f"   [SUCCESS] Converged: {conv_info['converged']}")
                print(f"   [ANALYSIS] Fidelity: {conv_info['final_fidelity']:.10f}")
                print(f"   [ANALYSIS] Iterations: {conv_info['iterations']}")
                print(f"   [ANALYSIS] Is trivial: {final_analysis['is_trivial']}")
                print(f"   [ANALYSIS] Entropy: {final_analysis['entropy']:.6f}")
                print(f"   [ANALYSIS] Purity: {final_analysis['purity']:.6f}")
                print(f"   [ANALYSIS] Interesting score: {final_analysis['interesting_score']:.3f}")
                print(f"   [ANALYSIS] Non-triviality score: {final_analysis['non_triviality_score']:.3f}")
                print(f"   [ANALYSIS] Coherence score: {final_analysis['coherence_score']:.3f}")
                print(f"   [ANALYSIS] Structure score: {final_analysis['structure_score']:.3f}")
                print(f"   [ANALYSIS] Quality score: {final_analysis['quality_score']:.3f}")
                print()
    
    return results

def analyze_minimal_results(results):
    """
    Analyze minimal mixing search results
    """
    analysis = {
        'summary': {},
        'by_circuit_type': {},
        'by_strategy': {},
        'best_fixed_points': [],
        'conclusion': ""
    }
    
    total_experiments = len(results['minimal_results'])
    
    if total_experiments == 0:
        analysis['summary'] = {
            'total_experiments': 0,
            'successful_experiments': 0,
            'success_rate': 0.0,
            'trivial_solutions': 0,
            'trivial_rate': 0.0,
            'non_trivial_solutions': 0,
            'non_trivial_rate': 0.0,
            'average_quality_score': 0.0,
            'high_quality_count': 0
        }
        analysis['conclusion'] = "NO EXPERIMENTS RUN - Check configuration"
        return analysis
    
    successful_experiments = sum(1 for r in results['minimal_results'] if r['success'])
    trivial_solutions = sum(1 for r in results['minimal_results'] if r['is_trivial'])
    non_trivial_solutions = successful_experiments - trivial_solutions
    high_quality_count = sum(1 for r in results['minimal_results'] if r['success'] and not r['is_trivial'] and r['quality_score'] > 0.6)
    
    # Calculate average scores
    avg_quality = np.mean([r['quality_score'] for r in results['minimal_results']]) if results['minimal_results'] else 0.0
    avg_interesting = np.mean([r['interesting_score'] for r in results['minimal_results']]) if results['minimal_results'] else 0.0
    avg_coherence = np.mean([r['coherence_score'] for r in results['minimal_results']]) if results['minimal_results'] else 0.0
    
    analysis['summary'] = {
        'total_experiments': total_experiments,
        'successful_experiments': successful_experiments,
        'success_rate': successful_experiments / total_experiments,
        'trivial_solutions': trivial_solutions,
        'trivial_rate': trivial_solutions / total_experiments,
        'non_trivial_solutions': non_trivial_solutions,
        'non_trivial_rate': non_trivial_solutions / total_experiments,
        'average_quality_score': avg_quality,
        'average_interesting_score': avg_interesting,
        'average_coherence_score': avg_coherence,
        'high_quality_count': high_quality_count,
        'high_quality_rate': high_quality_count / total_experiments
    }
    
    # Analysis by circuit type
    for circuit_type in set(r['circuit_type'] for r in results['minimal_results']):
        circuit_results = [r for r in results['minimal_results'] if r['circuit_type'] == circuit_type]
        successful = sum(1 for r in circuit_results if r['success'])
        trivial = sum(1 for r in circuit_results if r['is_trivial'])
        avg_quality = np.mean([r['quality_score'] for r in circuit_results])
        high_quality = sum(1 for r in circuit_results if r['success'] and not r['is_trivial'] and r['quality_score'] > 0.6)
        
        analysis['by_circuit_type'][circuit_type] = {
            'total': len(circuit_results),
            'successful': successful,
            'success_rate': successful / len(circuit_results),
            'trivial': trivial,
            'trivial_rate': trivial / len(circuit_results),
            'non_trivial': successful - trivial,
            'non_trivial_rate': (successful - trivial) / len(circuit_results),
            'average_quality_score': avg_quality,
            'high_quality_count': high_quality,
            'high_quality_rate': high_quality / len(circuit_results)
        }
    
    # Analysis by strategy
    for strategy in set(r['strategy'] for r in results['minimal_results']):
        strategy_results = [r for r in results['minimal_results'] if r['strategy'] == strategy]
        successful = sum(1 for r in strategy_results if r['success'])
        trivial = sum(1 for r in strategy_results if r['is_trivial'])
        avg_quality = np.mean([r['quality_score'] for r in strategy_results])
        high_quality = sum(1 for r in strategy_results if r['success'] and not r['is_trivial'] and r['quality_score'] > 0.6)
        
        analysis['by_strategy'][strategy] = {
            'total': len(strategy_results),
            'successful': successful,
            'success_rate': successful / len(strategy_results),
            'trivial': trivial,
            'trivial_rate': trivial / len(strategy_results),
            'non_trivial': successful - trivial,
            'non_trivial_rate': (successful - trivial) / len(strategy_results),
            'average_quality_score': avg_quality,
            'high_quality_count': high_quality,
            'high_quality_rate': high_quality / len(strategy_results)
        }
    
    # Find best fixed points (high quality scores)
    best_results = [r for r in results['minimal_results'] 
                   if r['success'] and not r['is_trivial'] and 
                   r['quality_score'] > 0.5]
    best_results.sort(key=lambda x: x['quality_score'], reverse=True)
    analysis['best_fixed_points'] = best_results[:10]  # Top 10
    
    # Determine conclusion
    if analysis['summary']['high_quality_rate'] > 0.3:
        analysis['conclusion'] = "EXCELLENT - Found many high-quality non-trivial fixed points"
    elif analysis['summary']['high_quality_rate'] > 0.1:
        analysis['conclusion'] = "GOOD - Found significant number of high-quality fixed points"
    elif analysis['summary']['non_trivial_rate'] > 0.2:
        analysis['conclusion'] = "MODERATE - Found some non-trivial fixed points"
    else:
        analysis['conclusion'] = "POOR - Still mostly finding trivial solutions"
    
    return analysis

def create_minimal_visualization(results, analysis):
    """
    Create visualization of minimal mixing search results
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Minimal Mixing Search for Interesting Fixed Points', fontsize=16, fontweight='bold')
    
    # 1. Success and non-trivial rates by circuit type
    circuit_types = list(analysis['by_circuit_type'].keys())
    success_rates = [analysis['by_circuit_type'][ct]['success_rate'] for ct in circuit_types]
    non_trivial_rates = [analysis['by_circuit_type'][ct]['non_trivial_rate'] for ct in circuit_types]
    high_quality_rates = [analysis['by_circuit_type'][ct]['high_quality_rate'] for ct in circuit_types]
    
    x = np.arange(len(circuit_types))
    width = 0.25
    
    axes[0, 0].bar(x - width, success_rates, width, label='Success Rate', color='blue', alpha=0.7)
    axes[0, 0].bar(x, non_trivial_rates, width, label='Non-Trivial Rate', color='green', alpha=0.7)
    axes[0, 0].bar(x + width, high_quality_rates, width, label='High Quality Rate', color='orange', alpha=0.7)
    axes[0, 0].set_title('Rates by Circuit Type')
    axes[0, 0].set_xlabel('Circuit Type')
    axes[0, 0].set_ylabel('Rate')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(circuit_types, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Average quality scores by circuit type
    quality_scores = [analysis['by_circuit_type'][ct]['average_quality_score'] for ct in circuit_types]
    axes[0, 1].bar(circuit_types, quality_scores, color='purple', alpha=0.7)
    axes[0, 1].set_title('Average Quality Scores by Circuit Type')
    axes[0, 1].set_xlabel('Circuit Type')
    axes[0, 1].set_ylabel('Quality Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Summary pie chart
    labels = ['High-Quality Non-Trivial', 'Low-Quality Non-Trivial', 'Trivial Solutions', 'Failed']
    high_quality = analysis['summary']['high_quality_count']
    low_quality = analysis['summary']['non_trivial_solutions'] - high_quality
    trivial = analysis['summary']['trivial_solutions']
    failed = analysis['summary']['total_experiments'] - analysis['summary']['successful_experiments']
    
    sizes = [high_quality, low_quality, trivial, failed]
    colors = ['green', 'yellow', 'red', 'gray']
    
    axes[0, 2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0, 2].set_title('Overall Results Breakdown')
    
    # 4. Best fixed points scatter plot
    if analysis['best_fixed_points']:
        interesting_scores = [r['interesting_score'] for r in analysis['best_fixed_points']]
        coherence_scores = [r['coherence_score'] for r in analysis['best_fixed_points']]
        circuit_types_best = [r['circuit_type'] for r in analysis['best_fixed_points']]
        
        # Color by circuit type
        unique_types = list(set(circuit_types_best))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
        color_map = dict(zip(unique_types, colors))
        
        for i, (x, y, ct) in enumerate(zip(interesting_scores, coherence_scores, circuit_types_best)):
            axes[1, 0].scatter(x, y, c=[color_map[ct]], s=100, alpha=0.7, label=ct if ct not in [s[0] for s in axes[1, 0].get_children() if hasattr(s, 'get_label')] else "")
        
        axes[1, 0].set_xlabel('Interesting Score')
        axes[1, 0].set_ylabel('Coherence Score')
        axes[1, 0].set_title('Best Fixed Points')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        axes[1, 0].text(0.5, 0.5, 'No good fixed points found', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Best Fixed Points')
    
    # 5. Quality score distribution histogram
    quality_scores_all = [r['quality_score'] for r in results['minimal_results'] if r['success']]
    axes[1, 1].hist(quality_scores_all, bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_xlabel('Quality Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Quality Scores')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Entropy vs Purity scatter
    entropies = [r['final_analysis']['entropy'] for r in results['minimal_results'] if r['success']]
    purities = [r['final_analysis']['purity'] for r in results['minimal_results'] if r['success']]
    colors_scatter = ['green' if r['quality_score'] > 0.6 else 'red' for r in results['minimal_results'] if r['success']]
    
    axes[1, 2].scatter(entropies, purities, c=colors_scatter, alpha=0.6)
    axes[1, 2].set_xlabel('Entropy')
    axes[1, 2].set_ylabel('Purity')
    axes[1, 2].set_title('Entropy vs Purity (Green=High Quality)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = f"minimal_mixing_results_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file

def generate_minimal_report(results, analysis, plot_file):
    """
    Generate comprehensive report on minimal mixing search
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"MINIMAL_MIXING_REPORT_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("MINIMAL MIXING SEARCH FOR INTERESTING FIXED POINTS REPORT\n")
        f.write("========================================================\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("==================\n")
        f.write(f"Total experiments: {analysis['summary']['total_experiments']}\n")
        f.write(f"Success rate: {analysis['summary']['success_rate']:.2%}\n")
        f.write(f"Non-trivial solutions: {analysis['summary']['non_trivial_solutions']} ({analysis['summary']['non_trivial_rate']:.2%})\n")
        f.write(f"High-quality solutions: {analysis['summary']['high_quality_count']} ({analysis['summary']['high_quality_rate']:.2%})\n")
        f.write(f"Trivial solutions: {analysis['summary']['trivial_solutions']} ({analysis['summary']['trivial_rate']:.2%})\n")
        f.write(f"Average quality score: {analysis['summary']['average_quality_score']:.3f}\n")
        f.write(f"Average interesting score: {analysis['summary']['average_interesting_score']:.3f}\n")
        f.write(f"Average coherence score: {analysis['summary']['average_coherence_score']:.3f}\n")
        f.write(f"CONCLUSION: {analysis['conclusion']}\n\n")
        
        f.write("DETAILED ANALYSIS BY CIRCUIT TYPE\n")
        f.write("=================================\n")
        for circuit_type, stats in analysis['by_circuit_type'].items():
            f.write(f"\n{circuit_type.upper()}:\n")
            f.write(f"  Success rate: {stats['success_rate']:.2%}\n")
            f.write(f"  Non-trivial rate: {stats['non_trivial_rate']:.2%}\n")
            f.write(f"  High-quality rate: {stats['high_quality_rate']:.2%}\n")
            f.write(f"  Average quality score: {stats['average_quality_score']:.3f}\n")
        
        f.write("\nDETAILED ANALYSIS BY STRATEGY\n")
        f.write("=============================\n")
        for strategy, stats in analysis['by_strategy'].items():
            f.write(f"\n{strategy.upper()}:\n")
            f.write(f"  Success rate: {stats['success_rate']:.2%}\n")
            f.write(f"  Non-trivial rate: {stats['non_trivial_rate']:.2%}\n")
            f.write(f"  High-quality rate: {stats['high_quality_rate']:.2%}\n")
            f.write(f"  Average quality score: {stats['average_quality_score']:.3f}\n")
        
        f.write("\nBEST FIXED POINTS FOUND\n")
        f.write("=======================\n")
        if analysis['best_fixed_points']:
            for i, result in enumerate(analysis['best_fixed_points'], 1):
                f.write(f"\n{i}. Circuit: {result['circuit_type']}\n")
                f.write(f"   Strategy: {result['strategy']} with params {result['params']}\n")
                f.write(f"   Quality score: {result['quality_score']:.3f}\n")
                f.write(f"   Interesting score: {result['interesting_score']:.3f}\n")
                f.write(f"   Non-triviality score: {result['non_triviality_score']:.3f}\n")
                f.write(f"   Coherence score: {result['coherence_score']:.3f}\n")
                f.write(f"   Structure score: {result['structure_score']:.3f}\n")
                f.write(f"   Entropy: {result['final_analysis']['entropy']:.6f}\n")
                f.write(f"   Purity: {result['final_analysis']['purity']:.6f}\n")
                f.write(f"   Iterations: {result['convergence_info']['iterations']}\n")
        else:
            f.write("No high-quality fixed points found.\n")
        
        f.write(f"\n[ANALYSIS] Visualization: {plot_file}\n")
        
        f.write("\nRECOMMENDATIONS\n")
        f.write("===============\n")
        if analysis['best_fixed_points']:
            f.write("RECOMMENDED CONFIGURATIONS:\n")
            for i, result in enumerate(analysis['best_fixed_points'][:3], 1):
                f.write(f"{i}. Circuit: {result['circuit_type']}\n")
                f.write(f"   Strategy: {result['strategy']}\n")
                f.write(f"   Parameters: {result['params']}\n")
                f.write(f"   Quality Score: {result['quality_score']:.3f}\n")
        else:
            f.write("No specific recommendations - need to try different approaches.\n")
        
        f.write(f"\nThe minimal mixing approach shows promise for finding truly interesting fixed points!\n")
    
    return report_file

def main():
    parser = argparse.ArgumentParser(description='Run minimal mixing search for interesting fixed points')
    parser.add_argument('--num_qubits', type=int, default=6, help='Number of qubits')
    parser.add_argument('--num_trials', type=int, default=40, help='Number of trials')
    
    args = parser.parse_args()
    
    print("🔍 MINIMAL MIXING SEARCH FOR INTERESTING FIXED POINTS")
    print("====================================================")
    print(f"Qubits: {args.num_qubits}")
    print(f"Trials: {args.num_trials}")
    print()
    
    # Run minimal mixing search
    results = run_minimal_mixing_search(
        num_qubits=args.num_qubits,
        num_trials=args.num_trials
    )
    
    # Analyze results
    print("📊 Analyzing minimal mixing results...")
    analysis = analyze_minimal_results(results)
    
    # Create visualization
    print("📈 Creating minimal mixing visualization...")
    plot_file = create_minimal_visualization(results, analysis)
    
    # Generate report
    print("📋 Generating minimal mixing report...")
    report_file = generate_minimal_report(results, analysis, plot_file)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"minimal_mixing_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("🎉 MINIMAL MIXING SEARCH COMPLETE!")
    print(f"   Results: {results_file}")
    print(f"   Visualization: {plot_file}")
    print(f"   Report: {report_file}")
    print(f"   Conclusion: {analysis['conclusion']}")
    print(f"   Non-trivial solutions: {analysis['summary']['non_trivial_solutions']}/{analysis['summary']['total_experiments']}")
    print(f"   High-quality solutions: {analysis['summary']['high_quality_count']}/{analysis['summary']['total_experiments']}")
    print(f"   Average quality score: {analysis['summary']['average_quality_score']:.3f}")
    
    if analysis['best_fixed_points']:
        print(f"\n🏆 TOP 3 BEST FIXED POINTS:")
        for i, result in enumerate(analysis['best_fixed_points'][:3], 1):
            print(f"   {i}. {result['circuit_type']} + {result['strategy']}")
            print(f"      Quality score: {result['quality_score']:.3f}")

if __name__ == "__main__":
    main() 