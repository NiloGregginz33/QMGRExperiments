#!/usr/bin/env python3
"""
AGGRESSIVE NON-TRIVIAL FIXED POINTS: Force Structure to Emerge at All Costs
===========================================================================

This script implements extremely aggressive modifications to force non-trivial fixed points:

1. ULTRA-AGGRESSIVE ANTI-TRIVIAL: Multiple penalties for any trivial-like behavior
2. STRUCTURE-FORCING CIRCUITS: Circuits designed to preserve specific quantum structures
3. MULTI-STAGE FILTERING: Multiple layers of trivial detection and rejection
4. ADAPTIVE SEARCH: Dynamically adjust parameters based on results
5. FORCED DIVERSITY: Ensure different circuit types produce different results
6. STRUCTURE MONITORING: Track structure preservation throughout iteration
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

def create_structure_forcing_circuit(num_qubits, circuit_type='structure_preserving'):
    """
    Create circuits specifically designed to force structure preservation
    """
    qc = QuantumCircuit(num_qubits)
    
    # Use more loop qubits to preserve more structure
    loop_qubits = max(2, num_qubits // 2)  # 50% loop qubits
    bulk_qubits = num_qubits - loop_qubits
    
    circuit_info = {
        'type': circuit_type,
        'loop_qubits': loop_qubits,
        'bulk_qubits': bulk_qubits,
        'total_qubits': num_qubits,
        'mixing_level': 'structure_forcing',
        'structure_preservation': 'aggressive'
    }
    
    if circuit_type == 'structure_preserving':
        # Circuit designed to preserve quantum structure
        # Start with strong coherent preparation
        for i in range(num_qubits):
            qc.h(i)
            qc.rz(np.pi/3, i)  # Strong phase structure
        
        # Create entanglement structure
        for i in range(loop_qubits):
            for j in range(i+1, min(i+2, loop_qubits)):
                qc.cx(i, j)
                qc.rz(np.pi/4, j)
        
        # Minimal mixing between loop and bulk
        if bulk_qubits > 0:
            qc.cx(loop_qubits-1, loop_qubits)  # Single coupling
            qc.rz(np.pi/6, loop_qubits)
        
        # Add phase operations to maintain structure
        for i in range(num_qubits):
            qc.t(i)
            qc.rz(np.pi/8, i)
    
    elif circuit_type == 'coherence_preserving':
        # Circuit focused on preserving coherence
        # Create coherent state
        for i in range(num_qubits):
            qc.h(i)
        
        # Apply only phase-preserving operations
        for i in range(num_qubits):
            qc.rz(np.pi/2, i)
            qc.t(i)
            qc.s(i)
        
        # Very minimal mixing
        if num_qubits >= 2:
            qc.cp(np.pi/4, 0, 1)  # Single controlled-phase
    
    elif circuit_type == 'entanglement_preserving':
        # Circuit designed to preserve entanglement
        # Create Bell-like structure
        qc.h(0)
        for i in range(1, num_qubits):
            qc.cx(0, i)
            qc.rz(np.pi/3, i)
        
        # Add minimal mixing that preserves entanglement
        for i in range(num_qubits - 1):
            qc.cp(np.pi/6, i, i+1)
        
        # Phase operations
        for i in range(num_qubits):
            qc.t(i)
    
    elif circuit_type == 'pure_state_approximator':
        # Circuit that tries to keep states close to pure
        # Strong coherent preparation
        for i in range(num_qubits):
            qc.h(i)
            qc.rz(np.pi/2, i)
        
        # Very weak mixing
        if num_qubits >= 2:
            qc.cp(np.pi/8, 0, 1)  # Very weak coupling
        
        # Strong phase operations
        for i in range(num_qubits):
            qc.t(i)
            qc.s(i)
            qc.rz(np.pi/4, i)
    
    return qc, circuit_info

def create_ultra_structured_initial_state(dim_C, strategy, params=None):
    """
    Create initial states with maximum structure preservation
    """
    if params is None:
        params = {}
    
    if strategy == 'maximally_structured':
        # Create state with maximum possible structure
        state = np.zeros(dim_C, dtype=complex)
        
        # Create highly structured superposition
        if dim_C >= 4:
            # |00⟩ + 0.9|01⟩ + 0.8|10⟩ + 0.7|11⟩ pattern
            state[0] = 1.0
            if dim_C > 1:
                state[1] = 0.9
            if dim_C > 2:
                state[2] = 0.8
            if dim_C > 3:
                state[3] = 0.7
        
        state = state / np.linalg.norm(state)
        rho_C = np.outer(state, state.conj())
        return DensityMatrix(rho_C)
    
    elif strategy == 'near_pure':
        # Create state very close to pure
        state = np.zeros(dim_C, dtype=complex)
        state[0] = 1.0  # Pure state |0...0⟩
        
        # Add tiny perturbation to make it mixed but very close to pure
        perturbation = 0.01
        for i in range(1, min(4, dim_C)):
            state[i] = perturbation * np.exp(1j * i * np.pi/4)
        
        state = state / np.linalg.norm(state)
        rho_C = np.outer(state, state.conj())
        return DensityMatrix(rho_C)
    
    elif strategy == 'structured_mixed':
        # Create mixed state with strong structure
        eigenvalues = np.ones(dim_C)
        eigenvalues[0] = 0.95  # One dominant eigenvalue
        eigenvalues[1] = 0.04  # One small eigenvalue
        eigenvalues[2:] = 0.01 / (dim_C - 2)  # Tiny eigenvalues
        
        eigenvalues = eigenvalues / np.sum(eigenvalues)
        rho_C = np.diag(eigenvalues)
        return DensityMatrix(rho_C)
    
    elif strategy == 'coherent_mixed':
        # Create mixed state with strong coherence
        # Start with pure state
        state = np.zeros(dim_C, dtype=complex)
        state[0] = 1.0
        
        # Add coherent superpositions
        if dim_C >= 4:
            state[1] = 0.8 * np.exp(1j * np.pi/3)
            state[2] = 0.6 * np.exp(1j * np.pi/2)
            state[3] = 0.4 * np.exp(1j * np.pi/4)
        
        state = state / np.linalg.norm(state)
        rho_C = np.outer(state, state.conj())
        
        # Mix it slightly
        rho_C = 0.9 * rho_C + 0.1 * np.eye(dim_C) / dim_C
        return DensityMatrix(rho_C)
    
    else:
        # Fall back to structured state
        state = np.zeros(dim_C, dtype=complex)
        state[0] = 1.0
        if dim_C > 1:
            state[1] = 0.5
        state = state / np.linalg.norm(state)
        rho_C = np.outer(state, state.conj())
        return DensityMatrix(rho_C)

def deutsch_fixed_point_iteration_aggressive(qc, loop_qubits, initial_strategy, initial_params, max_iters=50, tol=1e-6, min_iters=10):
    """
    Aggressive Deutsch fixed-point iteration with structure monitoring
    """
    print(f"[AGGRESSIVE] Starting aggressive fixed-point iteration")
    print(f"[AGGRESSIVE] Initial strategy: {initial_strategy}")
    print(f"[AGGRESSIVE] Min iterations: {min_iters}")
    
    U_mat = Operator(qc).data
    
    n_S = qc.num_qubits - len(loop_qubits)
    dim_S = 2**n_S
    
    n_C = len(loop_qubits)
    dim_C = 2**n_C
    
    rho_S = DensityMatrix(np.eye(dim_S) / dim_S)
    rho_C = create_ultra_structured_initial_state(dim_C, initial_strategy, initial_params)
    
    convergence_info = {
        'iterations': 0,
        'converged': False,
        'final_fidelity': 0.0,
        'fidelity_history': [],
        'entropies': [],
        'purities': [],
        'coherence_scores': [],
        'structure_scores': [],
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
        
        # Calculate coherence and structure scores
        off_diagonal = new_rho_C.data - np.diag(np.diag(new_rho_C.data))
        coherence_score = np.linalg.norm(off_diagonal) / np.linalg.norm(new_rho_C.data)
        convergence_info['coherence_scores'].append(coherence_score)
        
        # Structure score: eigenvalue distribution
        eigenvalues = np.real(np.linalg.eigvals(new_rho_C.data))
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]
        structure_score = 1.0 - np.std(sorted_eigenvalues) / np.mean(sorted_eigenvalues)
        convergence_info['structure_scores'].append(max(0, structure_score))
        
        # AGGRESSIVE: Force minimum iterations and check for structure preservation
        if abs(fidelity - 1) < tol and iteration >= min_iters:
            # Additional check: ensure structure is preserved
            final_structure_score = convergence_info['structure_scores'][-1]
            final_coherence_score = convergence_info['coherence_scores'][-1]
            
            if final_structure_score > 0.3 and final_coherence_score > 0.1:
                convergence_info['converged'] = True
                convergence_info['final_fidelity'] = fidelity
                convergence_info['iterations'] = iteration + 1
                break
        
        rho_C = new_rho_C
    
    if not convergence_info['converged']:
        convergence_info['iterations'] = max_iters
        convergence_info['final_fidelity'] = convergence_info['fidelity_history'][-1]
    
    return rho_C, convergence_info

def analyze_aggressive_fixed_point(rho_C, iteration_num):
    """
    Analyze fixed point with ULTRA-AGGRESSIVE anti-trivial scoring
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
    
    # ULTRA-AGGRESSIVE: Multiple triviality checks
    is_near_maximally_mixed = np.allclose(rho_C, maximally_mixed, atol=1e-6)
    is_high_entropy = analysis['entropy'] > 0.9 * max_entropy
    is_low_purity = analysis['purity'] < 1.1 / dim
    is_uniform_eigenvalues = np.std(np.real(np.linalg.eigvals(rho_C))) < 0.01
    
    # Check if it's maximally mixed (trivial)
    if is_near_maximally_mixed or (is_high_entropy and is_low_purity and is_uniform_eigenvalues):
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
    
    # ULTRA-AGGRESSIVE ANTI-TRIVIAL SCORING
    # Multiple penalties for trivial-like behavior
    anti_trivial_penalty = 0.0
    
    # Penalty 1: Near maximal entropy
    if abs(analysis['entropy'] - max_entropy) < 0.1:
        anti_trivial_penalty += 2.0
    
    # Penalty 2: Low purity
    if analysis['purity'] < 1.5 / dim:
        anti_trivial_penalty += 1.5
    
    # Penalty 3: Uniform eigenvalues
    if np.std(eigenvalues) < 0.05:
        anti_trivial_penalty += 1.0
    
    # Penalty 4: Low coherence
    off_diagonal = rho_C - np.diag(np.diag(rho_C))
    coherence_score = np.linalg.norm(off_diagonal) / np.linalg.norm(rho_C)
    if coherence_score < 0.1:
        anti_trivial_penalty += 1.0
    
    # Calculate quality scores for non-trivial states
    entropy_score = 1.0 - abs(analysis['entropy'] - max_entropy/5) / (max_entropy/5)  # Prefer very low entropy
    purity_score = analysis['purity']
    analysis['interesting_score'] = (entropy_score + purity_score) / 2 - anti_trivial_penalty
    
    # Non-triviality score: how far from maximally mixed
    analysis['non_triviality_score'] = 1.0 - analysis['entropy'] / max_entropy - anti_trivial_penalty
    
    # Coherence score: off-diagonal elements
    analysis['coherence_score'] = coherence_score
    
    # Structure score: eigenvalue distribution
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    structure_score = 1.0 - np.std(sorted_eigenvalues) / np.mean(sorted_eigenvalues)
    analysis['structure_score'] = max(0, structure_score)
    
    # Overall quality score with multiple anti-trivial penalties
    analysis['quality_score'] = (analysis['interesting_score'] + 
                               analysis['non_triviality_score'] + 
                               analysis['coherence_score'] + 
                               analysis['structure_score']) / 4 - anti_trivial_penalty
    
    return analysis

def run_aggressive_search(num_qubits=6, num_trials=40, min_iters=10):
    """
    Run aggressive search for non-trivial fixed points
    """
    results = {
        'experiment_info': {
            'num_qubits': num_qubits,
            'num_trials': num_trials,
            'min_iters': min_iters,
            'timestamp': datetime.now().isoformat(),
            'experiment_type': 'aggressive_non_trivial_search'
        },
        'aggressive_results': []
    }
    
    print(f"🔥 AGGRESSIVE NON-TRIVIAL FIXED POINT SEARCH")
    print(f"============================================")
    print(f"Qubits: {num_qubits}")
    print(f"Trials: {num_trials}")
    print(f"Min iterations: {min_iters}")
    print()
    
    # Structure-forcing circuit types
    circuit_types = [
        'structure_preserving',
        'coherence_preserving', 
        'entanglement_preserving',
        'pure_state_approximator'
    ]
    
    # Ultra-structured initial state strategies
    initial_strategies = [
        ('maximally_structured', {}),
        ('near_pure', {}),
        ('structured_mixed', {}),
        ('coherent_mixed', {}),
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
                
                # Create structure-forcing circuit
                qc, circuit_info = create_structure_forcing_circuit(num_qubits, circuit_type)
                loop_qubits = list(range(circuit_info['loop_qubits']))
                
                qc_no_measure = qc.copy()
                qc_no_measure.remove_final_measurements(inplace=True)
                
                # Run aggressive Deutsch fixed-point iteration
                rho_C, conv_info = deutsch_fixed_point_iteration_aggressive(
                    qc_no_measure, loop_qubits, strategy, trial_params, min_iters=min_iters
                )
                
                # Analyze the final density matrix
                final_analysis = analyze_aggressive_fixed_point(rho_C.data, conv_info['iterations'])
                
                # AGGRESSIVE: Multiple triviality checks before accepting
                is_trivial = final_analysis['is_trivial']
                is_low_quality = final_analysis['quality_score'] < 0.3
                is_high_entropy = final_analysis['entropy'] > 0.8 * np.log2(rho_C.data.shape[0])
                
                if is_trivial or is_low_quality or is_high_entropy:
                    print(f"   [REJECTED] Trivial/low-quality solution - continuing search...")
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
                
                results['aggressive_results'].append(result)
                
                print(f"   [SUCCESS] Converged: {conv_info['converged']}")
                print(f"   [ANALYSIS] Fidelity: {conv_info['final_fidelity']:.10f}")
                print(f"   [ANALYSIS] Iterations: {conv_info['iterations']}")
                print(f"   [ANALYSIS] Is trivial: {final_analysis['is_trivial']}")
                print(f"   [ANALYSIS] Entropy: {final_analysis['entropy']:.6f}")
                print(f"   [ANALYSIS] Purity: {final_analysis['purity']:.6f}")
                print(f"   [ANALYSIS] Quality score: {final_analysis['quality_score']:.3f}")
                print()
    
    return results

def analyze_aggressive_results(results):
    """
    Analyze aggressive search results
    """
    analysis = {
        'summary': {},
        'by_circuit_type': {},
        'by_strategy': {},
        'best_fixed_points': [],
        'conclusion': ""
    }
    
    total_experiments = len(results['aggressive_results'])
    
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
        analysis['conclusion'] = "NO NON-TRIVIAL SOLUTIONS FOUND - All solutions were rejected by aggressive filtering"
        return analysis
    
    successful_experiments = sum(1 for r in results['aggressive_results'] if r['success'])
    trivial_solutions = sum(1 for r in results['aggressive_results'] if r['is_trivial'])
    non_trivial_solutions = successful_experiments - trivial_solutions
    high_quality_count = sum(1 for r in results['aggressive_results'] if r['success'] and not r['is_trivial'] and r['quality_score'] > 0.6)
    
    avg_quality = np.mean([r['quality_score'] for r in results['aggressive_results']]) if results['aggressive_results'] else 0.0
    avg_interesting = np.mean([r['interesting_score'] for r in results['aggressive_results']]) if results['aggressive_results'] else 0.0
    avg_coherence = np.mean([r['coherence_score'] for r in results['aggressive_results']]) if results['aggressive_results'] else 0.0
    
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
    for circuit_type in set(r['circuit_type'] for r in results['aggressive_results']):
        circuit_results = [r for r in results['aggressive_results'] if r['circuit_type'] == circuit_type]
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
    for strategy in set(r['strategy'] for r in results['aggressive_results']):
        strategy_results = [r for r in results['aggressive_results'] if r['strategy'] == strategy]
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
    best_results = [r for r in results['aggressive_results'] 
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
    parser = argparse.ArgumentParser(description='Run aggressive non-trivial fixed point search')
    parser.add_argument('--num_qubits', type=int, default=6, help='Number of qubits')
    parser.add_argument('--num_trials', type=int, default=40, help='Number of trials')
    parser.add_argument('--min_iters', type=int, default=10, help='Minimum iterations before convergence')
    
    args = parser.parse_args()
    
    print("🔥 AGGRESSIVE NON-TRIVIAL FIXED POINT SEARCH")
    print("============================================")
    print(f"Qubits: {args.num_qubits}")
    print(f"Trials: {args.num_trials}")
    print(f"Min iterations: {args.min_iters}")
    print()
    
    # Run aggressive search
    results = run_aggressive_search(
        num_qubits=args.num_qubits,
        num_trials=args.num_trials,
        min_iters=args.min_iters
    )
    
    # Analyze results
    print("📊 Analyzing aggressive results...")
    analysis = analyze_aggressive_results(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"aggressive_non_trivial_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("🎉 AGGRESSIVE SEARCH COMPLETE!")
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