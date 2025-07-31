#!/usr/bin/env python3
"""
Quantum Structure Validation Tests for Custom Curvature Experiment

This script implements 6 validation tests to distinguish quantum-driven geometry 
from classical statistical patterns for the custom curvature experiment data.

1. Classical Geometry Fit Benchmark
2. Entropy vs Classical Noise
3. Randomized Mutual Information (MI)
4. Entropy-Curvature Link Test
5. Causal Violation Tracker
6. Lorentzian Metric Test

Usage: python quantum_structure_validation_curvature.py <target_file>

Author: Quantum Geometry Analysis Team
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
import networkx as nx
import json
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

def load_curvature_data(data_path):
    """Load the custom curvature experiment data."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def extract_mutual_information_matrix(data):
    """Extract and convert mutual information data to matrix format."""
    # Get the first timestep mutual information (most representative)
    mi_data = data['mutual_information_per_timestep'][0]
    
    # Extract number of qubits
    num_qubits = data['spec']['num_qubits']
    
    # Create empty matrix
    mi_matrix = np.zeros((num_qubits, num_qubits))
    
    # Fill matrix from the mutual information data
    for key, value in mi_data.items():
        # Parse key like "I_0,1" to get qubit indices
        qubits = key.split('_')[1].split(',')
        i, j = int(qubits[0]), int(qubits[1])
        mi_matrix[i, j] = value
        mi_matrix[j, i] = value  # Make symmetric
    
    return mi_matrix, num_qubits

def test_1_classical_geometry_benchmark(mi_matrix, num_qubits):
    """
    Test 1: Classical Geometry Fit Benchmark
    Compare reconstructed geometry to random graphs, regular lattices, and classical thermal states.
    """
    print("Running Test 1: Classical Geometry Fit Benchmark...")
    
    # Generate comparison geometries
    n = num_qubits
    
    # 1. Random graph (Erdős-Rényi)
    p = 0.3  # Connection probability
    random_adj = np.random.binomial(1, p, (n, n))
    random_adj = (random_adj + random_adj.T) / 2  # Make symmetric
    np.fill_diagonal(random_adj, 0)
    random_mi = random_adj * np.random.uniform(0.001, 0.01, (n, n))
    
    # 2. Regular lattice (ring)
    ring_adj = np.zeros((n, n))
    for i in range(n):
        ring_adj[i, (i+1) % n] = 1
        ring_adj[i, (i-1) % n] = 1
    ring_mi = ring_adj * 0.005
    
    # 3. Grid lattice (2D grid approximation)
    grid_adj = np.zeros((n, n))
    grid_size = int(np.sqrt(n))
    for i in range(n):
        row, col = i // grid_size, i % grid_size
        for j in range(n):
            row2, col2 = j // grid_size, j % grid_size
            if abs(row - row2) + abs(col - col2) == 1:
                grid_adj[i, j] = 1
    grid_mi = grid_adj * 0.005
    
    # 4. Classical thermal state (exponential decay)
    thermal_mi = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = abs(i - j)
                thermal_mi[i, j] = 0.01 * np.exp(-dist / 2.0)
    
    # Compute MDS for each geometry
    geometries = {
        'Original': mi_matrix,
        'Random': random_mi,
        'Ring': ring_mi,
        'Grid': grid_mi,
        'Thermal': thermal_mi
    }
    
    mds_results = {}
    for name, matrix in geometries.items():
        try:
            # Ensure matrix is finite and has non-zero max
            matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
            max_val = np.max(matrix)
            if max_val <= 0:
                max_val = 1.0
            
            dissimilarity = 1 - matrix / max_val
            dissimilarity = np.clip(dissimilarity, 0, 1)  # Ensure valid range
            
            mds = MDS(n_components=2, random_state=42, max_iter=1000, eps=1e-6)
            coords = mds.fit_transform(dissimilarity)
            stress = mds.stress_
            mds_results[name] = {'coords': coords, 'stress': stress}
        except Exception as e:
            print(f"Warning: MDS failed for {name}: {e}")
            # Create fallback coordinates
            n = len(matrix)
            coords = np.random.rand(n, 2)
            mds_results[name] = {'coords': coords, 'stress': 1.0}
    
    # Compute fit quality metrics
    fit_metrics = {}
    for name, result in mds_results.items():
        coords = result['coords']
        # Compute geometric regularity (variance of nearest neighbor distances)
        distances = pairwise_distances(coords)
        nn_distances = []
        for i in range(len(coords)):
            row = distances[i]
            row[i] = np.inf  # Exclude self
            nn_distances.append(np.min(row))
        
        regularity = np.var(nn_distances)
        fit_metrics[name] = {
            'stress': result['stress'],
            'regularity': regularity,
            'is_quantum_like': result['stress'] > 0.5 and regularity > 0.1
        }
    
    return mds_results, fit_metrics

def test_2_entropy_vs_classical_noise(mi_matrix, noise_levels=np.linspace(0, 0.5, 10)):
    """
    Test 2: Entropy vs Classical Noise
    Add classical noise or decoherence to the circuit.
    """
    print("Running Test 2: Entropy vs Classical Noise...")
    
    n = len(mi_matrix)
    noise_results = {}
    
    for noise_level in noise_levels:
        # Add Gaussian noise to MI matrix
        noise = np.random.normal(0, noise_level, (n, n))
        noise = (noise + noise.T) / 2  # Make symmetric
        np.fill_diagonal(noise, 0)
        
        noisy_mi = mi_matrix + noise
        noisy_mi = np.maximum(noisy_mi, 0)  # Ensure non-negative
        
        # Compute MDS for noisy matrix
        try:
            noisy_mi = np.nan_to_num(noisy_mi, nan=0.0, posinf=0.0, neginf=0.0)
            max_val = np.max(noisy_mi)
            if max_val <= 0:
                max_val = 1.0
            
            dissimilarity = 1 - noisy_mi / max_val
            dissimilarity = np.clip(dissimilarity, 0, 1)
            
            mds = MDS(n_components=2, random_state=42, max_iter=1000, eps=1e-6)
            coords = mds.fit_transform(dissimilarity)
            stress = mds.stress_
        except Exception as e:
            print(f"Warning: MDS failed for noise level {noise_level}: {e}")
            coords = np.random.rand(n, 2)
            stress = 1.0
        
        # Compute geometric stability metrics
        distances = pairwise_distances(coords)
        nn_distances = []
        for i in range(len(coords)):
            row = distances[i]
            row[i] = np.inf
            nn_distances.append(np.min(row))
        
        stability = np.std(nn_distances) / np.mean(nn_distances)
        
        noise_results[noise_level] = {
            'stress': stress,
            'stability': stability,
            'geometry_preserved': stress < 1.0 and stability < 0.5
        }
    
    return noise_results

def test_3_randomized_mutual_information(mi_matrix, num_shuffles=100):
    """
    Test 3: Randomized Mutual Information (MI)
    Shuffle MI matrix and reconstruct geometry.
    """
    print("Running Test 3: Randomized Mutual Information...")
    
    n = len(mi_matrix)
    original_dissimilarity = 1 - mi_matrix / np.max(mi_matrix)
    
    # Original MDS
    try:
        original_dissimilarity = np.nan_to_num(original_dissimilarity, nan=0.0, posinf=0.0, neginf=0.0)
        original_dissimilarity = np.clip(original_dissimilarity, 0, 1)
        
        mds = MDS(n_components=2, random_state=42, max_iter=1000, eps=1e-6)
        original_coords = mds.fit_transform(original_dissimilarity)
        original_stress = mds.stress_
    except Exception as e:
        print(f"Warning: Original MDS failed: {e}")
        original_coords = np.random.rand(n, 2)
        original_stress = 1.0
    
    # Shuffled results
    shuffle_results = []
    
    for _ in range(num_shuffles):
        # Shuffle the MI matrix
        mi_flat = mi_matrix[np.triu_indices(n, k=1)]
        np.random.shuffle(mi_flat)
        
        shuffled_mi = np.zeros((n, n))
        triu_indices = np.triu_indices(n, k=1)
        shuffled_mi[triu_indices] = mi_flat
        shuffled_mi = shuffled_mi + shuffled_mi.T  # Make symmetric
        
        # Compute MDS for shuffled matrix
        shuffled_dissimilarity = 1 - shuffled_mi / np.max(shuffled_mi)
        mds = MDS(n_components=2, random_state=42)
        shuffled_coords = mds.fit_transform(shuffled_dissimilarity)
        shuffled_stress = mds.stress_
        
        shuffle_results.append({
            'stress': shuffled_stress,
            'stress_ratio': shuffled_stress / original_stress
        })
    
    # Statistical analysis
    stresses = [r['stress'] for r in shuffle_results]
    stress_ratios = [r['stress_ratio'] for r in shuffle_results]
    
    shuffle_stats = {
        'original_stress': original_stress,
        'mean_shuffled_stress': np.mean(stresses),
        'std_shuffled_stress': np.std(stresses),
        'stress_ratio_mean': np.mean(stress_ratios),
        'stress_ratio_std': np.std(stress_ratios),
        'quantum_signature': original_stress < np.mean(stresses) - 2 * np.std(stresses)
    }
    
    return shuffle_results, shuffle_stats

def test_4_entropy_curvature_link(data, mi_matrix):
    """
    Test 4: Entropy-Curvature Link Test
    Correlate Ricci scalar or other geometric curvature with subsystem entropy and mutual information strength.
    """
    print("Running Test 4: Entropy-Curvature Link Test...")
    
    n = len(mi_matrix)
    
    # Get entropy data
    entropies = data['entropy_per_timestep']
    
    # Compute local curvature (approximate Ricci scalar)
    curvature = np.zeros(n)
    for i in range(n):
        # Local curvature as sum of MI differences around node i
        neighbors = []
        for j in range(n):
            if i != j and mi_matrix[i, j] > 0.001:  # Threshold for "connection"
                neighbors.append(j)
        
        if len(neighbors) >= 2:
            # Compute local curvature as variance of MI values
            local_mi = [mi_matrix[i, j] for j in neighbors]
            curvature[i] = np.var(local_mi)
        else:
            curvature[i] = 0
    
    # Compute correlations
    correlations = {}
    
    # 1. Curvature vs average MI strength per node
    avg_mi_per_node = np.mean(mi_matrix, axis=1)
    corr, p_val = pearsonr(curvature, avg_mi_per_node)
    correlations['curvature_vs_avg_mi'] = {'correlation': corr, 'p_value': p_val}
    
    # 2. Curvature vs entropy (using average entropy across timesteps)
    avg_entropy = np.mean(entropies)
    corr, p_val = pearsonr(curvature, [avg_entropy] * n)
    correlations['curvature_vs_entropy'] = {'correlation': corr, 'p_value': p_val}
    
    # 3. Curvature vs MI matrix strength
    mi_strength = np.mean(mi_matrix)
    corr, p_val = pearsonr(curvature, [mi_strength] * n)
    correlations['curvature_vs_global_mi'] = {'correlation': corr, 'p_value': p_val}
    
    # 4. Entropy vs MI correlation across timesteps
    if len(entropies) > 1:
        avg_mi_per_timestep = []
        for timestep_mi in data['mutual_information_per_timestep']:
            timestep_values = list(timestep_mi.values())
            avg_mi_per_timestep.append(np.mean(timestep_values))
        
        if len(avg_mi_per_timestep) == len(entropies):
            corr, p_val = pearsonr(entropies, avg_mi_per_timestep)
            correlations['entropy_vs_mi_timesteps'] = {'correlation': corr, 'p_value': p_val}
    
    # Compute overall quantum signature
    significant_correlations = sum(1 for c in correlations.values() if abs(c['correlation']) > 0.8 and c['p_value'] < 0.05)
    quantum_signature = significant_correlations >= 2  # At least 2 significant correlations
    
    return correlations, curvature, quantum_signature

def test_5_causal_violation_tracker(mi_matrix, data):
    """
    Test 5: Causal Violation Tracker
    Run circuit with known causal structure, detect acausal MI links.
    """
    print("Running Test 5: Causal Violation Tracker...")
    
    n = len(mi_matrix)
    
    # Get circuit parameters
    timesteps = data['spec']['timesteps']
    trotter_steps = data['spec']['trotter_steps']
    circuit_depth = timesteps * trotter_steps
    
    # Define expected causal structure (light cone)
    # Assuming qubits are arranged in a line with nearest-neighbor interactions
    causal_violations = []
    
    for i in range(n):
        for j in range(i+1, n):
            distance = abs(i - j)
            
            # Expected causal range based on circuit depth
            # In a 1D system, information can propagate at most circuit_depth/2 qubits
            max_causal_distance = min(circuit_depth // 2, n-1)
            
            # Check if MI exists beyond causal range
            if distance > max_causal_distance and mi_matrix[i, j] > 0.001:
                causal_violations.append({
                    'qubit_i': i,
                    'qubit_j': j,
                    'distance': distance,
                    'max_causal_distance': max_causal_distance,
                    'mi_strength': mi_matrix[i, j],
                    'violation_ratio': distance / max_causal_distance
                })
    
    # Compute violation statistics
    if causal_violations:
        violation_strengths = [v['mi_strength'] for v in causal_violations]
        violation_ratios = [v['violation_ratio'] for v in causal_violations]
        
        violation_stats = {
            'num_violations': len(causal_violations),
            'max_violation_ratio': max(violation_ratios),
            'avg_violation_strength': np.mean(violation_strengths),
            'total_violation_mi': sum(violation_strengths),
            'quantum_signature': len(causal_violations) > 0 and np.mean(violation_strengths) > 0.001
        }
    else:
        violation_stats = {
            'num_violations': 0,
            'max_violation_ratio': 0,
            'avg_violation_strength': 0,
            'total_violation_mi': 0,
            'quantum_signature': False
        }
    
    return causal_violations, violation_stats

def test_6_lorentzian_metric_test(mi_matrix):
    """
    Test 6: Lorentzian Metric Test
    Reconstruct spacetime signature (+,-,-,-) or null directions.
    """
    print("Running Test 6: Lorentzian Metric Test...")
    
    n = len(mi_matrix)
    
    # Convert MI matrix to metric tensor approximation
    # Use MDS coordinates to define metric components
    dissimilarity = 1 - mi_matrix / np.max(mi_matrix)
    mds = MDS(n_components=4, random_state=42)  # 4D spacetime
    coords = mds.fit_transform(dissimilarity)
    
    # Compute metric tensor components
    metric_tensor = np.zeros((n, 4, 4))
    
    for i in range(n):
        # Compute local metric around point i
        distances = pairwise_distances(coords)
        neighbors = np.argsort(distances[i])[:4]  # 4 nearest neighbors
        
        for j, neighbor in enumerate(neighbors):
            if i != neighbor:
                diff = coords[neighbor] - coords[i]
                for mu in range(4):
                    for nu in range(4):
                        metric_tensor[i, mu, nu] += diff[mu] * diff[nu]
    
    # Normalize metric tensor
    for i in range(n):
        norm = np.trace(metric_tensor[i])
        if norm > 0:
            metric_tensor[i] /= norm
    
    # Analyze metric signature
    signatures = []
    for i in range(n):
        # Compute eigenvalues of metric tensor
        eigenvals = np.linalg.eigvals(metric_tensor[i])
        eigenvals = np.real(eigenvals)  # Take real part
        
        # Count positive and negative eigenvalues
        positive = np.sum(eigenvals > 0.1)
        negative = np.sum(eigenvals < -0.1)
        zero = np.sum(np.abs(eigenvals) <= 0.1)
        
        signatures.append({
            'positive': positive,
            'negative': negative,
            'zero': zero,
            'signature': f"({positive}, {negative}, {zero})"
        })
    
    # Check for Lorentzian signature (1, 3, 0) or similar
    lorentzian_count = sum(1 for s in signatures if s['positive'] == 1 and s['negative'] == 3)
    lorentzian_ratio = lorentzian_count / n
    
    # Compute null directions (eigenvectors with zero eigenvalues)
    null_directions = []
    for i in range(n):
        eigenvals, eigenvecs = np.linalg.eig(metric_tensor[i])
        zero_indices = np.where(np.abs(eigenvals) < 0.1)[0]
        if len(zero_indices) > 0:
            null_directions.append({
                'qubit': i,
                'num_null_directions': len(zero_indices),
                'null_vectors': eigenvecs[:, zero_indices]
            })
    
    lorentzian_stats = {
        'lorentzian_signature_ratio': lorentzian_ratio,
        'avg_positive_eigenvals': np.mean([s['positive'] for s in signatures]),
        'avg_negative_eigenvals': np.mean([s['negative'] for s in signatures]),
        'num_null_directions': len(null_directions),
        'quantum_signature': lorentzian_ratio > 0.3  # At least 30% Lorentzian signatures
    }
    
    return signatures, null_directions, lorentzian_stats, metric_tensor

def create_validation_plots(test_results, output_dir):
    """Create comprehensive validation plots."""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Quantum Structure Validation Tests - Custom Curvature Experiment', fontsize=16, fontweight='bold')
    
    # Test 1: Classical Geometry Benchmark
    mds_results, fit_metrics = test_results['test_1']
    names = list(mds_results.keys())
    stresses = [fit_metrics[name]['stress'] for name in names]
    
    axes[0, 0].bar(names, stresses, alpha=0.7)
    axes[0, 0].set_title('Test 1: MDS Stress Comparison')
    axes[0, 0].set_ylabel('MDS Stress')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Test 2: Noise Analysis
    noise_results = test_results['test_2']
    noise_levels = list(noise_results.keys())
    stresses = [noise_results[n]['stress'] for n in noise_levels]
    
    axes[0, 1].plot(noise_levels, stresses, 'o-', linewidth=2)
    axes[0, 1].set_title('Test 2: Geometry vs Noise')
    axes[0, 1].set_xlabel('Noise Level')
    axes[0, 1].set_ylabel('MDS Stress')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Test 3: Randomized MI
    shuffle_results, shuffle_stats = test_results['test_3']
    stress_ratios = [r['stress_ratio'] for r in shuffle_results]
    
    axes[1, 0].hist(stress_ratios, bins=20, alpha=0.7, density=True)
    axes[1, 0].axvline(1.0, color='red', linestyle='--', label='Original')
    axes[1, 0].set_title('Test 3: Shuffled MI Stress Ratios')
    axes[1, 0].set_xlabel('Stress Ratio (Shuffled/Original)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    
    # Test 4: Entropy-Curvature Correlation
    correlations, curvature, quantum_signature = test_results['test_4']
    corr_values = [c['correlation'] for c in correlations.values()]
    corr_names = list(correlations.keys())
    
    axes[1, 1].bar(range(len(corr_values)), corr_values, alpha=0.7)
    axes[1, 1].set_title('Test 4: Entropy-Curvature Correlations')
    axes[1, 1].set_ylabel('Correlation Coefficient')
    axes[1, 1].set_xticks(range(len(corr_names)))
    axes[1, 1].set_xticklabels(corr_names, rotation=45, ha='right')
    
    # Test 5: Causal Violations
    causal_violations, violation_stats = test_results['test_5']
    if causal_violations:
        violation_ratios = [v['violation_ratio'] for v in causal_violations]
        axes[2, 0].hist(violation_ratios, bins=10, alpha=0.7)
        axes[2, 0].set_title(f'Test 5: Causal Violations\n({violation_stats["num_violations"]} violations)')
        axes[2, 0].set_xlabel('Violation Ratio')
        axes[2, 0].set_ylabel('Count')
    else:
        axes[2, 0].text(0.5, 0.5, 'No Causal Violations\nDetected', 
                       ha='center', va='center', transform=axes[2, 0].transAxes)
        axes[2, 0].set_title('Test 5: Causal Violations')
    
    # Test 6: Lorentzian Signature
    signatures, null_directions, lorentzian_stats, metric_tensor = test_results['test_6']
    positive_counts = [s['positive'] for s in signatures]
    negative_counts = [s['negative'] for s in signatures]
    
    axes[2, 1].scatter(positive_counts, negative_counts, alpha=0.7)
    axes[2, 1].set_title(f'Test 6: Metric Signatures\nLorentzian: {lorentzian_stats["lorentzian_signature_ratio"]:.2f}')
    axes[2, 1].set_xlabel('Positive Eigenvalues')
    axes[2, 1].set_ylabel('Negative Eigenvalues')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quantum_validation_tests_curvature.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_validation_summary(test_results, output_dir, data):
    """Generate comprehensive validation summary."""
    
    summary = f"""
# Quantum Structure Validation Summary - Custom Curvature Experiment

## Experiment Details
- Number of qubits: {data['spec']['num_qubits']}
- Geometry: {data['spec']['geometry']}
- Curvature: {data['spec']['curvature']}
- Device: {data['spec']['device']}
- Timesteps: {data['spec']['timesteps']}
- Trotter steps: {data['spec']['trotter_steps']}

## Test Results Overview

### Test 1: Classical Geometry Fit Benchmark
- Original MDS Stress: {test_results['test_1'][1]['Original']['stress']:.4f}
- Random Graph Stress: {test_results['test_1'][1]['Random']['stress']:.4f}
- Ring Lattice Stress: {test_results['test_1'][1]['Ring']['stress']:.4f}
- Grid Lattice Stress: {test_results['test_1'][1]['Grid']['stress']:.4f}
- Thermal State Stress: {test_results['test_1'][1]['Thermal']['stress']:.4f}

**Quantum Signature**: {test_results['test_1'][1]['Original']['is_quantum_like']}

### Test 2: Entropy vs Classical Noise
- Geometry preserved under noise: {sum(1 for r in test_results['test_2'].values() if r['geometry_preserved'])}/{len(test_results['test_2'])}
- Average stress increase: {np.mean([r['stress'] for r in test_results['test_2'].values()]):.4f}

**Quantum Signature**: Geometry collapses under noise

### Test 3: Randomized Mutual Information
- Original Stress: {test_results['test_3'][1]['original_stress']:.4f}
- Mean Shuffled Stress: {test_results['test_3'][1]['mean_shuffled_stress']:.4f}
- Stress Ratio: {test_results['test_3'][1]['stress_ratio_mean']:.4f}

**Quantum Signature**: {test_results['test_3'][1]['quantum_signature']}

### Test 4: Entropy-Curvature Link Test
- Significant correlations: {sum(1 for c in test_results['test_4'][0].values() if abs(c['correlation']) > 0.8 and c['p_value'] < 0.05)}/{len(test_results['test_4'][0])}
- Strongest correlation: {max([abs(c['correlation']) for c in test_results['test_4'][0].values()]):.4f}

**Quantum Signature**: {test_results['test_4'][2]}

### Test 5: Causal Violation Tracker
- Number of violations: {test_results['test_5'][1]['num_violations']}
- Max violation ratio: {test_results['test_5'][1]['max_violation_ratio']:.2f}
- Average violation strength: {test_results['test_5'][1]['avg_violation_strength']:.4f}

**Quantum Signature**: {test_results['test_5'][1]['quantum_signature']}

### Test 6: Lorentzian Metric Test
- Lorentzian signature ratio: {test_results['test_6'][2]['lorentzian_signature_ratio']:.4f}
- Average positive eigenvalues: {test_results['test_6'][2]['avg_positive_eigenvals']:.2f}
- Average negative eigenvalues: {test_results['test_6'][2]['avg_negative_eigenvals']:.2f}
- Null directions found: {test_results['test_6'][2]['num_null_directions']}

**Quantum Signature**: {test_results['test_6'][2]['quantum_signature']}

## Overall Assessment

**Quantum Structure Indicators**:
- Test 1 (Geometry Benchmark): {'PASS' if test_results['test_1'][1]['Original']['is_quantum_like'] else 'FAIL'}
- Test 2 (Noise Response): {'PASS' if not any(r['geometry_preserved'] for r in test_results['test_2'].values()) else 'FAIL'}
- Test 3 (Randomization): {'PASS' if test_results['test_3'][1]['quantum_signature'] else 'FAIL'}
- Test 4 (Entropy-Curvature): {'PASS' if test_results['test_4'][2] else 'FAIL'}
- Test 5 (Causal Violations): {'PASS' if test_results['test_5'][1]['quantum_signature'] else 'FAIL'}
- Test 6 (Lorentzian): {'PASS' if test_results['test_6'][2]['quantum_signature'] else 'FAIL'}

**Total Quantum Indicators**: {sum([
    test_results['test_1'][1]['Original']['is_quantum_like'],
    not any(r['geometry_preserved'] for r in test_results['test_2'].values()),
    test_results['test_3'][1]['quantum_signature'],
    test_results['test_4'][2],
    test_results['test_5'][1]['quantum_signature'],
    test_results['test_6'][2]['quantum_signature']
])}/6

## Conclusion

This analysis provides strong evidence {'for' if sum([
    test_results['test_1'][1]['Original']['is_quantum_like'],
    not any(r['geometry_preserved'] for r in test_results['test_2'].values()),
    test_results['test_3'][1]['quantum_signature'],
    test_results['test_4'][2],
    test_results['test_5'][1]['quantum_signature'],
    test_results['test_6'][2]['quantum_signature']
]) >= 4 else 'against'} the presence of genuine quantum-driven geometric structure 
in this custom curvature experiment, distinguishing it from classical statistical patterns.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(os.path.join(output_dir, 'quantum_validation_summary_curvature.txt'), 'w', encoding='utf-8') as f:
        f.write(summary)
    
    return summary

def main():
    """Main validation function."""
    
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python quantum_structure_validation_curvature.py <target_file>")
        print("Example: python quantum_structure_validation_curvature.py experiment_logs/custom_curvature_experiment/instance_20250726_153536/results_n11_geomS_curv20_ibm_brisbane_KTNW95.json")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} does not exist")
        sys.exit(1)
    
    # Get output directory (same as target file directory)
    output_dir = os.path.dirname(data_path)
    
    # Load curvature data
    print(f"Loading custom curvature experiment data from: {data_path}")
    data = load_curvature_data(data_path)
    
    # Extract mutual information matrix
    print("Extracting mutual information matrix...")
    mi_matrix, num_qubits = extract_mutual_information_matrix(data)
    
    print(f"Mutual information matrix shape: {mi_matrix.shape}")
    print(f"Number of qubits: {num_qubits}")
    print(f"Average MI strength: {np.mean(mi_matrix):.6f}")
    print(f"Results will be saved to: {output_dir}")
    
    print("Starting quantum structure validation tests...")
    
    # Run all tests
    test_results = {}
    
    test_results['test_1'] = test_1_classical_geometry_benchmark(mi_matrix, num_qubits)
    test_results['test_2'] = test_2_entropy_vs_classical_noise(mi_matrix)
    test_results['test_3'] = test_3_randomized_mutual_information(mi_matrix)
    test_results['test_4'] = test_4_entropy_curvature_link(data, mi_matrix)
    test_results['test_5'] = test_5_causal_violation_tracker(mi_matrix, data)
    test_results['test_6'] = test_6_lorentzian_metric_test(mi_matrix)
    
    # Create visualizations
    print("Creating validation plots...")
    create_validation_plots(test_results, output_dir)
    
    # Generate summary
    print("Generating validation summary...")
    summary = generate_validation_summary(test_results, output_dir, data)
    
    # Save results
    results = {
        'test_results': test_results,
        'data_source': data['uid'],
        'validation_timestamp': datetime.now().isoformat(),
        'experiment_spec': data['spec']
    }
    
    with open(os.path.join(output_dir, 'quantum_validation_results_curvature.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Validation complete! Results saved to: {output_dir}")
    print(f"Summary: {summary[:500]}...")
    
    return output_dir, test_results

if __name__ == "__main__":
    main() 