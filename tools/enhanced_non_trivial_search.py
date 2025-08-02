#!/usr/bin/env python3
"""
ENHANCED NON-TRIVIAL FIXED POINTS: Force Structure to Emerge
============================================================

This script implements the user's specific modifications to force non-trivial fixed points:

1. RAISE MIXING LEVEL: Change from "ultra_minimal" to "minimal" or "structured_minimal"
2. ANTI-TRIVIAL SCORING: Penalize states with entropy near maximal value
3. TOPOLOGY VARIATION: Random connectivity and edge weights
4. STRUCTURED SEEDS: Graph states, GHZ, W states as initial conditions
5. CONVERGENCE LIMITS: Force minimum iteration count before acceptance
6. ENHANCED CONSTRAINTS: Discard maximally mixed outputs automatically
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

def create_enhanced_circuit(num_qubits, circuit_type='minimal', topology_variation=True):
    """
    Create circuits with enhanced mixing and topology variation
    """
    qc = QuantumCircuit(num_qubits)
    
    # Determine loop and bulk qubits with asymmetric ratios
    loop_qubits = max(1, num_qubits // 3)  # More loop qubits than ultra_minimal
    bulk_qubits = num_qubits - loop_qubits
    
    # Add topology variation with random connectivity
    if topology_variation:
        np.random.seed(np.random.randint(0, 10000))
        edge_weights = np.random.uniform(0.1, 1.0, num_qubits)
        connectivity_pattern = np.random.choice([0, 1], size=(num_qubits, num_qubits), p=[0.7, 0.3])
    else:
        edge_weights = np.ones(num_qubits)
        connectivity_pattern = np.zeros((num_qubits, num_qubits))
    
    circuit_info = {
        'type': circuit_type,
        'loop_qubits': loop_qubits,
        'bulk_qubits': bulk_qubits,
        'total_qubits': num_qubits,
        'mixing_level': 'enhanced_minimal',
        'topology_variation': topology_variation,
        'edge_weights': edge_weights.tolist() if topology_variation else None
    }
    
    if circuit_type == 'minimal':
        # Enhanced minimal mixing: more gates than ultra_minimal
        # Start with coherent preparation
        for i in range(num_qubits):
            qc.h(i)
            qc.rz(np.pi/4 * edge_weights[i], i)  # Variable phase structure
        
        # Apply 3-4 CX gates strategically
        if num_qubits >= 2:
            qc.cx(0, 1)
        if num_qubits >= 3:
            qc.cx(1, 2)
        if num_qubits >= 4:
            qc.cx(2, 3)
        if num_qubits >= 5:
            qc.cx(3, 4)
        
        # Add controlled phase operations
        for i in range(num_qubits - 1):
            qc.cp(np.pi/6 * edge_weights[i], i, i+1)
        
        # Add phase operations to preserve structure
        for i in range(num_qubits):
            qc.rz(np.pi/8 * edge_weights[i], i)
            qc.t(i)
    
    elif circuit_type == 'structured_minimal':
        # Structured minimal mixing with specific patterns
        # Create structured initial state
        qc.h(0)
        for i in range(1, num_qubits):
            qc.cx(0, i)  # Create entanglement structure
            qc.rz(np.pi/4 * edge_weights[i], i)
        
        # Apply structured mixing with more gates
        if num_qubits >= 3:
            qc.cx(1, 2)
        if num_qubits >= 4:
            qc.cx(2, 3)
        if num_qubits >= 5:
            qc.cx(3, 4)
        
        # Add controlled operations
        for i in range(num_qubits - 1):
            qc.cp(np.pi/4 * edge_weights[i], i, i+1)
        
        # Add phase operations to maintain structure
        for i in range(num_qubits):
            qc.rz(np.pi/6 * edge_weights[i], i)
            qc.t(i)
    
    elif circuit_type == 'graph_state':
        # Graph state inspired circuit
        # Create graph state structure
        for i in range(num_qubits):
            qc.h(i)
        
        # Apply controlled-Z operations in graph pattern
        for i in range(num_qubits):
            for j in range(i+1, min(i+3, num_qubits)):  # Connect to next 2 qubits
                if connectivity_pattern[i, j] or np.random.random() < 0.3:
                    qc.cz(i, j)
        
        # Add some mixing gates
        for i in range(num_qubits - 1):
            qc.cx(i, i+1)
            qc.rz(np.pi/6 * edge_weights[i], i)
    
    elif circuit_type == 'ghz_inspired':
        # GHZ state inspired circuit
        qc.h(0)
        for i in range(1, num_qubits):
            qc.cx(0, i)
        
        # Add mixing while preserving structure
        for i in range(num_qubits - 1):
            qc.cx(i, i+1)
            qc.rz(np.pi/4 * edge_weights[i], i)
        
        # Add phase operations
        for i in range(num_qubits):
            qc.t(i)
    
    return qc, circuit_info

def create_structured_initial_state(dim_C, strategy, params=None):
    """
    Create initial states including known structured quantum states
    """
    if params is None:
        params = {}
    
    if strategy == 'graph_state':
        # Create graph state density matrix
        state = np.zeros(dim_C, dtype=complex)
        state[0] = 1.0  # Start with |0...0⟩
        
        # Apply graph state structure
        if dim_C >= 4:
            # |00⟩ + |01⟩ + |10⟩ - |11⟩ pattern
            state[0] = 1.0
            if dim_C > 1:
                state[1] = 1.0
            if dim_C > 2:
                state[2] = 1.0
            if dim_C > 3:
                state[3] = -1.0
        
        state = state / np.linalg.norm(state)
        rho_C = np.outer(state, state.conj())
        return DensityMatrix(rho_C)
    
    elif strategy == 'ghz_state':
        # Create GHZ state density matrix
        state = np.zeros(dim_C, dtype=complex)
        state[0] = 1.0  # |0...0⟩
        if dim_C > 1:
            state[-1] = 1.0  # |1...1⟩ (last state)
        
        state = state / np.linalg.norm(state)
        rho_C = np.outer(state, state.conj())
        return DensityMatrix(rho_C)
    
    elif strategy == 'w_state':
        # Create W state density matrix
        state = np.zeros(dim_C, dtype=complex)
        n_qubits = int(np.log2(dim_C))
        
        # W state: (|100...0⟩ + |010...0⟩ + ... + |000...1⟩) / √n
        for i in range(min(n_qubits, dim_C)):
            state[2**i] = 1.0
        
        state = state / np.linalg.norm(state)
        rho_C = np.outer(state, state.conj())
        return DensityMatrix(rho_C)
    
    elif strategy == 'coherent_pure':
        # Enhanced coherent pure state
        state = np.zeros(dim_C, dtype=complex)
        
        if dim_C >= 4:
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
            
            state = state / np.linalg.norm(state)
        else:
            state[0] = 1.0
            state = state / np.linalg.norm(state)
        
        rho_C = np.outer(state, state.conj())
        return DensityMatrix(rho_C)
    
    elif strategy == 'low_entropy':
        # Create state with low but non-zero entropy
        target_entropy = params.get('target_entropy', 0.3)  # Lower entropy
        
        eigenvalues = np.ones(dim_C)
        max_entropy = np.log2(dim_C)
        
        if target_entropy > 0:
            dominant_weight = 1.0 - (target_entropy / max_entropy) * 0.3  # Keep it very high
            eigenvalues[0] = dominant_weight
            remaining_weight = 1.0 - dominant_weight
            eigenvalues[1:] = remaining_weight / (dim_C - 1)
        
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        eigenvalues = eigenvalues / np.sum(eigenvalues)
        
        rho_C = np.diag(eigenvalues)
        return DensityMatrix(rho_C)
    
    else:
        # Fall back to random state with controlled entropy
        np.random.seed(params.get('seed', None))
        random_state = np.random.randn(dim_C) + 1j * np.random.randn(dim_C)
        random_state = random_state / np.linalg.norm(random_state)
        rho_C = np.outer(random_state, random_state.conj())
        return DensityMatrix(rho_C)

def deutsch_fixed_point_iteration_enhanced(qc, loop_qubits, initial_strategy, initial_params, max_iters=50, tol=1e-6, min_iters=5):
    """
    Enhanced Deutsch fixed-point iteration with minimum iteration requirement
    """
    print(f"[ENHANCED] Starting enhanced fixed-point iteration")
    print(f"[ENHANCED] Initial strategy: {initial_strategy}")
    print(f"[ENHANCED] Min iterations: {min_iters}")
    
    U_mat = Operator(qc).data
    
    n_S = qc.num_qubits - len(loop_qubits)
    dim_S = 2**n_S
    
    n_C = len(loop_qubits)
    dim_C = 2**n_C
    
    rho_S = DensityMatrix(np.eye(dim_S) / dim_S)
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
        joint = rho_S.tensor(rho_C)
        joint = DensityMatrix(U_mat @ joint.data @ U_mat.conj().T)
        new_rho_C = partial_trace(joint, list(range(n_S)))
        new_rho_C = DensityMatrix(new_rho_C.data)
        
        fidelity = state_fidelity(rho_C, new_rho_C)
        convergence_info['fidelity_history'].append(fidelity)
        
        entropy_val = entropy(new_rho_C)
        purity = np.trace(new_rho_C.data @ new_rho_C.data)
        convergence_info['entropies'].append(entropy_val)
        convergence_info['purities'].append(purity)
        
        off_diagonal = new_rho_C.data - np.diag(np.diag(new_rho_C.data))
        coherence_score = np.linalg.norm(off_diagonal) / np.linalg.norm(new_rho_C.data)
        convergence_info['coherence_scores'].append(coherence_score)
        
        # ENHANCED: Force minimum iterations before accepting convergence
        if abs(fidelity - 1) < tol and iteration >= min_iters:
            convergence_info['converged'] = True
            convergence_info['final_fidelity'] = fidelity
            convergence_info['iterations'] = iteration + 1
            break
        
        rho_C = new_rho_C
    
    if not convergence_info['converged']:
        convergence_info['iterations'] = max_iters
        convergence_info['final_fidelity'] = convergence_info['fidelity_history'][-1]
    
    return rho_C, convergence_info

def analyze_enhanced_fixed_point(rho_C, iteration_num):
    """
    Analyze fixed point with ANTI-TRIVIAL scoring
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
    
    # ENHANCED: ANTI-TRIVIAL SCORING
    # Penalize states where entropy is within ε of maximal value
    epsilon = 0.05
    anti_trivial_penalty = 0.0
    if abs(analysis['entropy'] - max_entropy) < epsilon:
        anti_trivial_penalty = 1.0  # Heavy penalty for near-trivial states
    
    # Calculate quality scores for non-trivial states
    entropy_score = 1.0 - abs(analysis['entropy'] - max_entropy/4) / (max_entropy/4)  # Prefer even lower entropy
    purity_score = analysis['purity']
    analysis['interesting_score'] = (entropy_score + purity_score) / 2 - anti_trivial_penalty
    
    # Non-triviality score: how far from maximally mixed
    analysis['non_triviality_score'] = 1.0 - analysis['entropy'] / max_entropy - anti_trivial_penalty
    
    # Coherence score: off-diagonal elements
    off_diagonal = rho_C - np.diag(np.diag(rho_C))
    coherence_score = np.linalg.norm(off_diagonal) / np.linalg.norm(rho_C)
    analysis['coherence_score'] = coherence_score
    
    # Structure score: eigenvalue distribution
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    structure_score = 1.0 - np.std(sorted_eigenvalues) / np.mean(sorted_eigenvalues)
    analysis['structure_score'] = max(0, structure_score)
    
    # Overall quality score with anti-trivial penalty
    analysis['quality_score'] = (analysis['interesting_score'] + 
                               analysis['non_triviality_score'] + 
                               analysis['coherence_score'] + 
                               analysis['structure_score']) / 4 - anti_trivial_penalty
    
    return analysis

def run_enhanced_search(num_qubits=6, num_trials=40, min_iters=5):
    """
    Run enhanced search for non-trivial fixed points
    """
    results = {
        'experiment_info': {
            'num_qubits': num_qubits,
            'num_trials': num_trials,
            'min_iters': min_iters,
            'timestamp': datetime.now().isoformat(),
            'experiment_type': 'enhanced_non_trivial_search'
        },
        'enhanced_results': []
    }
    
    print(f"🚀 ENHANCED NON-TRIVIAL FIXED POINT SEARCH")
    print(f"==========================================")
    print(f"Qubits: {num_qubits}")
    print(f"Trials: {num_trials}")
    print(f"Min iterations: {min_iters}")
    print()
    
    # Enhanced circuit types with more mixing
    circuit_types = [
        'minimal',
        'structured_minimal', 
        'graph_state',
        'ghz_inspired'
    ]
    
    # Enhanced initial state strategies including known structured states
    initial_strategies = [
        ('graph_state', {}),
        ('ghz_state', {}),
        ('w_state', {}),
        ('coherent_pure', {'alpha': 0.9, 'beta': 0.3, 'gamma': 0.1}),
        ('low_entropy', {'target_entropy': 0.2}),
        ('low_entropy', {'target_entropy': 0.4}),
    ]
    
    trial_count = 0
    total_configs = len(circuit_types) * len(initial_strategies)
    trials_per_config = max(1, num_trials // total_configs)
    
    for circuit_type in circuit_types:
        for strategy, params in initial_strategies:
            for trial in range(trials_per_config):
                trial_count += 1
                
                trial_params = params.copy()
                trial_params['seed'] = np.random.randint(0, 10000)
                
                print(f"🔬 Trial {trial_count}/{num_trials}: {circuit_type} + {strategy}")
                
                # Create enhanced circuit with topology variation
                qc, circuit_info = create_enhanced_circuit(num_qubits, circuit_type, topology_variation=True)
                loop_qubits = list(range(circuit_info['loop_qubits']))
                
                qc_no_measure = qc.copy()
                qc_no_measure.remove_final_measurements(inplace=True)
                
                # Run enhanced Deutsch fixed-point iteration
                rho_C, conv_info = deutsch_fixed_point_iteration_enhanced(
                    qc_no_measure, loop_qubits, strategy, trial_params, min_iters=min_iters
                )
                
                # Analyze the final density matrix
                final_analysis = analyze_enhanced_fixed_point(rho_C.data, conv_info['iterations'])
                
                # ENHANCED: Discard maximally mixed outputs automatically
                if final_analysis['is_trivial']:
                    print(f"   [DISCARDED] Trivial solution - continuing search...")
                    continue
                
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
                
                results['enhanced_results'].append(result)
                
                print(f"   [SUCCESS] Converged: {conv_info['converged']}")
                print(f"   [ANALYSIS] Fidelity: {conv_info['final_fidelity']:.10f}")
                print(f"   [ANALYSIS] Iterations: {conv_info['iterations']}")
                print(f"   [ANALYSIS] Is trivial: {final_analysis['is_trivial']}")
                print(f"   [ANALYSIS] Entropy: {final_analysis['entropy']:.6f}")
                print(f"   [ANALYSIS] Purity: {final_analysis['purity']:.6f}")
                print(f"   [ANALYSIS] Quality score: {final_analysis['quality_score']:.3f}")
                print()
    
    return results

def analyze_enhanced_results(results):
    """
    Analyze enhanced search results
    """
    analysis = {
        'summary': {},
        'by_circuit_type': {},
        'by_strategy': {},
        'best_fixed_points': [],
        'conclusion': ""
    }
    
    total_experiments = len(results['enhanced_results'])
    
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
        analysis['conclusion'] = "NO NON-TRIVIAL SOLUTIONS FOUND - All solutions were trivial and discarded"
        return analysis
    
    successful_experiments = sum(1 for r in results['enhanced_results'] if r['success'])
    trivial_solutions = sum(1 for r in results['enhanced_results'] if r['is_trivial'])
    non_trivial_solutions = successful_experiments - trivial_solutions
    high_quality_count = sum(1 for r in results['enhanced_results'] if r['success'] and not r['is_trivial'] and r['quality_score'] > 0.6)
    
    avg_quality = np.mean([r['quality_score'] for r in results['enhanced_results']]) if results['enhanced_results'] else 0.0
    avg_interesting = np.mean([r['interesting_score'] for r in results['enhanced_results']]) if results['enhanced_results'] else 0.0
    avg_coherence = np.mean([r['coherence_score'] for r in results['enhanced_results']]) if results['enhanced_results'] else 0.0
    
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
    for circuit_type in set(r['circuit_type'] for r in results['enhanced_results']):
        circuit_results = [r for r in results['enhanced_results'] if r['circuit_type'] == circuit_type]
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
    for strategy in set(r['strategy'] for r in results['enhanced_results']):
        strategy_results = [r for r in results['enhanced_results'] if r['strategy'] == strategy]
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
    
    # Find best fixed points
    best_results = [r for r in results['enhanced_results'] 
                   if r['success'] and not r['is_trivial'] and 
                   r['quality_score'] > 0.5]
    best_results.sort(key=lambda x: x['quality_score'], reverse=True)
    analysis['best_fixed_points'] = best_results[:10]
    
    # Determine conclusion
    if analysis['summary']['high_quality_rate'] > 0.3:
        analysis['conclusion'] = "EXCELLENT - Found many high-quality non-trivial fixed points!"
    elif analysis['summary']['high_quality_rate'] > 0.1:
        analysis['conclusion'] = "GOOD - Found significant number of high-quality fixed points"
    elif analysis['summary']['non_trivial_rate'] > 0.2:
        analysis['conclusion'] = "MODERATE - Found some non-trivial fixed points"
    else:
        analysis['conclusion'] = "POOR - Still mostly finding trivial solutions"
    
    return analysis

def main():
    parser = argparse.ArgumentParser(description='Run enhanced non-trivial fixed point search')
    parser.add_argument('--num_qubits', type=int, default=6, help='Number of qubits')
    parser.add_argument('--num_trials', type=int, default=40, help='Number of trials')
    parser.add_argument('--min_iters', type=int, default=5, help='Minimum iterations before convergence')
    
    args = parser.parse_args()
    
    print("🚀 ENHANCED NON-TRIVIAL FIXED POINT SEARCH")
    print("==========================================")
    print(f"Qubits: {args.num_qubits}")
    print(f"Trials: {args.num_trials}")
    print(f"Min iterations: {args.min_iters}")
    print()
    
    # Run enhanced search
    results = run_enhanced_search(
        num_qubits=args.num_qubits,
        num_trials=args.num_trials,
        min_iters=args.min_iters
    )
    
    # Analyze results
    print("📊 Analyzing enhanced results...")
    analysis = analyze_enhanced_results(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"enhanced_non_trivial_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("🎉 ENHANCED SEARCH COMPLETE!")
    print(f"   Results: {results_file}")
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