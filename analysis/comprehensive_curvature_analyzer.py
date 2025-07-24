#!/usr/bin/env python3
"""
Comprehensive Curvature Analyzer
Combines RT surface approximation, Lorentzian geometry analysis, and advanced holographic analyses
for any custom curvature experiment results.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import shortest_path
from scipy.stats import pearsonr, bootstrap, norm
from sklearn.manifold import MDS
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_experiment_results(results_file):
    """Load experiment results from JSON file"""
    print(f"Loading results from: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Handle different result formats
    if 'lorentzian_solution' in results:
        # Custom curvature experiment format
        lorentzian_data = results['lorentzian_solution']
        return {
            'lorentzian_action': lorentzian_data.get('stationary_action'),
            'edge_lengths': lorentzian_data.get('edge_lengths', []),
            'counts': results.get('counts'),  # May be None
            'num_qubits': results.get('spec', {}).get('num_qubits', 0),
            'geometry': results.get('spec', {}).get('geometry', 'unknown'),
            'curvature': results.get('spec', {}).get('curvature', 0.0),
            'device': results.get('spec', {}).get('device', 'unknown'),
            'shots': results.get('spec', {}).get('shots', 0),
            'source_file': results_file
        }
    elif 'counts_per_timestep' in results:
        # Custom curvature experiment with timesteps format
        # Use the first timestep for analysis
        counts = results['counts_per_timestep'][0] if results['counts_per_timestep'] else {}
        return {
            'lorentzian_action': None,  # Not computed in this format
            'edge_lengths': [],  # Not available in this format
            'counts': counts,
            'num_qubits': results.get('spec', {}).get('num_qubits', 0),
            'geometry': results.get('spec', {}).get('geometry', 'unknown'),
            'curvature': results.get('spec', {}).get('curvature', 0.0),
            'device': results.get('spec', {}).get('device', 'unknown'),
            'shots': results.get('spec', {}).get('shots', 0),
            'source_file': results_file
        }
    elif 'counts' in results:
        # Bulk reconstruction format
        return {
            'lorentzian_action': None,
            'edge_lengths': [],
            'counts': results['counts'],
            'num_qubits': results.get('num_qubits', 0),
            'geometry': results.get('geometry_type', 'unknown'),
            'curvature': results.get('curvature_strength', 0.0),
            'device': results.get('device', 'unknown'),
            'shots': results.get('shots', 0),
            'source_file': results_file
        }
    else:
        # Fallback for other formats
        return {
            'lorentzian_action': results.get('lorentzian_action'),
            'edge_lengths': results.get('edge_lengths', []),
            'counts': results.get('counts'),
            'num_qubits': results.get('num_qubits', 0),
            'geometry': results.get('geometry', 'unknown'),
            'curvature': results.get('curvature', 0.0),
            'device': results.get('device', 'unknown'),
            'shots': results.get('shots', 0),
            'source_file': results_file
        }

def compute_rt_surface_approximation(mi_matrix):
    """Compute Ryu-Takayanagi surface approximation using graph-theoretic approach"""
    num_qubits = len(mi_matrix)
    
    # Normalize MI matrix to [0,1] range for better scaling
    mi_max = np.max(mi_matrix)
    if mi_max > 0:
        mi_normalized = mi_matrix / mi_max
    else:
        mi_normalized = mi_matrix
    
    # Convert MI to distance: higher MI = shorter distance
    eps = 1e-8
    distance_matrix = -np.log(mi_normalized + eps)
    
    # Normalize distance matrix to reasonable range [0, 10]
    dist_max = np.max(distance_matrix)
    if dist_max > 0:
        distance_matrix = 10 * distance_matrix / dist_max
    
    np.fill_diagonal(distance_matrix, 0)
    
    # Use Dijkstra's algorithm to find shortest paths
    distances, predecessors = shortest_path(distance_matrix, directed=False, return_predecessors=True)
    
    # Compute RT surface lengths for different boundary regions
    rt_surfaces = {}
    
    # Test RT formula for various boundary regions
    for size in range(1, num_qubits//2 + 1):
        # Sample random boundary regions of size 'size'
        import random
        for _ in range(min(10, num_qubits)):  # Sample up to 10 regions per size
            boundary_region = random.sample(range(num_qubits), size)
            complement_region = [i for i in range(num_qubits) if i not in boundary_region]
            
            # Compute entanglement entropy of boundary region
            boundary_mi_values = []
            for i in boundary_region:
                for j in boundary_region:
                    if i != j and mi_matrix[i, j] > 0:
                        boundary_mi_values.append(mi_matrix[i, j])
            
            boundary_mi = np.mean(boundary_mi_values) if boundary_mi_values else 0.0
            
            # RT surface length: sum of edge weights within the boundary region
            rt_length = 0.0
            edge_count = 0
            for i in boundary_region:
                for j in boundary_region:
                    if i != j and distance_matrix[i, j] < np.inf:
                        rt_length += distance_matrix[i, j]
                        edge_count += 1
            
            # Average RT length per edge
            avg_rt_length = rt_length / edge_count if edge_count > 0 else 0.0
            
            # RT ratio: boundary MI / RT length (should be roughly constant if RT holds)
            rt_ratio = boundary_mi / (avg_rt_length + eps) if avg_rt_length > 0 else 0.0
            
            rt_surfaces[f"size_{size}_region_{tuple(boundary_region)}"] = {
                'boundary_mi': float(boundary_mi),
                'rt_length': float(avg_rt_length),
                'rt_ratio': float(rt_ratio),
                'edge_count': int(edge_count)
            }
    
    return rt_surfaces, distances

def analyze_lorentzian_geometry(results):
    """Analyze Lorentzian geometry from experiment results"""
    if 'lorentzian_action' not in results:
        return None
    
    lorentzian_action = results['lorentzian_action']
    edge_lengths = results.get('edge_lengths', [])
    
    # Statistical analysis
    if edge_lengths:
        edge_lengths = np.array(edge_lengths)
        mean_length = np.mean(edge_lengths)
        std_length = np.std(edge_lengths)
        
        # Bootstrap confidence interval for Lorentzian action
        def lorentzian_statistic(data):
            return np.mean(data)
        
        bootstrap_ci = bootstrap((edge_lengths,), lorentzian_statistic, confidence_level=0.95)
        
        # Monte Carlo simulation for p-value
        n_simulations = 10000
        random_actions = []
        for _ in range(n_simulations):
            random_lengths = np.random.normal(mean_length, std_length, len(edge_lengths))
            random_action = np.mean(random_lengths)
            random_actions.append(random_action)
        
        random_actions = np.array(random_actions)
        p_value = np.sum(random_actions <= lorentzian_action) / n_simulations
        
        return {
            'lorentzian_action': float(lorentzian_action),
            'mean_edge_length': float(mean_length),
            'std_edge_length': float(std_length),
            'bootstrap_ci': [float(bootstrap_ci.confidence_interval[0]), float(bootstrap_ci.confidence_interval[1])],
            'p_value': float(p_value),
            'is_significant': bool(p_value < 0.05)
        }
    
    return None

def compute_mutual_information_matrix(counts, num_qubits):
    """Compute mutual information matrix from measurement counts"""
    def mutual_information(counts, num_qubits, qubits_a, qubits_b):
        """Compute mutual information between two sets of qubits"""
        total_shots = sum(counts.values())
        if total_shots == 0:
            return 0.0
        
        # Compute marginal probabilities
        p_a = {}
        p_b = {}
        p_ab = {}
        
        for bitstring, count in counts.items():
            prob = count / total_shots
            
            # Extract bits for qubits A and B
            bits_a = ''.join([bitstring[i] for i in qubits_a])
            bits_b = ''.join([bitstring[i] for i in qubits_b])
            
            # Update joint probability
            if (bits_a, bits_b) not in p_ab:
                p_ab[(bits_a, bits_b)] = 0
            p_ab[(bits_a, bits_b)] += prob
            
            # Update marginal probabilities
            if bits_a not in p_a:
                p_a[bits_a] = 0
            p_a[bits_a] += prob
            
            if bits_b not in p_b:
                p_b[bits_b] = 0
            p_b[bits_b] += prob
        
        # Compute entropies
        def entropy(probs):
            return -sum(p * np.log2(p) for p in probs.values() if p > 0)
        
        h_a = entropy(p_a)
        h_b = entropy(p_b)
        h_ab = entropy(p_ab)
        
        # Mutual information
        mi = h_a + h_b - h_ab
        return max(0, mi)  # MI should be non-negative
    
    mi_matrix = np.zeros((num_qubits, num_qubits))
    
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            mi = mutual_information(counts, num_qubits, [i], [j])
            if mi > 0.001:  # Threshold for meaningful MI
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
    
    return mi_matrix

def rt_verification_analysis(mi_matrix, coordinates):
    """RT Verification: Show that MI between boundary regions tracks with minimal paths in emergent bulk geometry"""
    num_qubits = len(mi_matrix)
    
    # Compute minimal paths in bulk geometry
    distances_3d = np.zeros((num_qubits, num_qubits))
    for i in range(num_qubits):
        for j in range(num_qubits):
            if i != j:
                # Euclidean distance in embedded space
                distances_3d[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])
    
    # Collect MI vs minimal path data
    mi_values = []
    path_lengths = []
    
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            if mi_matrix[i, j] > 0:
                mi_values.append(mi_matrix[i, j])
                path_lengths.append(distances_3d[i, j])
    
    if len(mi_values) > 1:
        correlation, p_value = pearsonr(mi_values, path_lengths)
        
        return {
            'mi_values': mi_values,
            'path_lengths': path_lengths,
            'correlation': float(correlation),
            'p_value': float(p_value),
            'num_pairs': len(mi_values)
        }
    
    return None

def holographic_entropy_scaling_analysis(mi_matrix, num_qubits):
    """Dimension Scaling: Show holographic scaling of entropy (area, not volume)"""
    # Compute entanglement entropy for different boundary region sizes
    entropy_data = {}
    
    for region_size in range(1, num_qubits):
        entropies = []
        
        # Sample different regions of this size
        import random
        for _ in range(min(20, num_qubits)):  # Sample up to 20 regions per size
            region = random.sample(range(num_qubits), region_size)
            
            # Compute entanglement entropy of this region
            region_mi_values = []
            for i in region:
                for j in region:
                    if i != j and mi_matrix[i, j] > 0:
                        region_mi_values.append(mi_matrix[i, j])
            
            # Entropy is related to mutual information
            if region_mi_values:
                entropy = np.mean(region_mi_values)
                entropies.append(entropy)
        
        if entropies:
            entropy_data[region_size] = {
                'mean_entropy': float(np.mean(entropies)),
                'std_entropy': float(np.std(entropies)),
                'num_samples': len(entropies)
            }
    
    # Fit scaling law
    if len(entropy_data) > 2:
        sizes = list(entropy_data.keys())
        entropies = [entropy_data[size]['mean_entropy'] for size in sizes]
        
        # Try different scaling laws
        # Area scaling: entropy ∝ region_size
        # Volume scaling: entropy ∝ region_size^2 (for 2D boundary)
        
        # Linear fit (area scaling)
        area_coeff = np.polyfit(sizes, entropies, 1)
        area_r_squared = 1 - np.sum((entropies - np.polyval(area_coeff, sizes))**2) / np.sum((entropies - np.mean(entropies))**2)
        
        # Quadratic fit (volume scaling)
        volume_coeff = np.polyfit(sizes, entropies, 2)
        volume_r_squared = 1 - np.sum((entropies - np.polyval(volume_coeff, sizes))**2) / np.sum((entropies - np.mean(entropies))**2)
        
        return {
            'entropy_data': entropy_data,
            'area_scaling': {
                'coefficients': area_coeff.tolist(),
                'r_squared': float(area_r_squared)
            },
            'volume_scaling': {
                'coefficients': volume_coeff.tolist(),
                'r_squared': float(volume_r_squared)
            },
            'holographic_support': bool(area_r_squared > volume_r_squared)
        }
    
    return None

def boundary_dynamics_analysis(results, mi_matrix, coordinates):
    """Boundary Dynamics: Correlate time evolution of qubit observables with geometry deformation"""
    # This analysis requires time-resolved data
    # For now, we'll analyze spatial correlations between boundary observables and bulk geometry
    
    num_qubits = len(mi_matrix)
    
    # Compute boundary observables (using MI as proxy for entanglement)
    boundary_observables = []
    bulk_geometry_features = []
    
    for i in range(num_qubits):
        # Boundary observable: average MI with other qubits
        observable = np.mean([mi_matrix[i, j] for j in range(num_qubits) if i != j])
        boundary_observables.append(observable)
        
        # Bulk geometry feature: distance from center of mass
        center_of_mass = np.mean(coordinates, axis=0)
        distance_from_center = np.linalg.norm(coordinates[i] - center_of_mass)
        bulk_geometry_features.append(distance_from_center)
    
    if len(boundary_observables) > 1:
        correlation, p_value = pearsonr(boundary_observables, bulk_geometry_features)
        
        return {
            'boundary_observables': boundary_observables,
            'bulk_geometry_features': bulk_geometry_features,
            'correlation': float(correlation),
            'p_value': float(p_value)
        }
    
    return None

def rt_surface_entropy_analysis(results, mi_matrix, coordinates):
    """
    RT Surface Analysis: Test S(A) ∝ Area_RT(A) relation
    
    This function analyzes the relationship between boundary entropy S(A) and 
    the area of the corresponding RT surface Area_RT(A) to verify the holographic
    principle prediction.
    """
    print("🔍 Performing RT Surface vs. Boundary Entropy Analysis...")
    
    # Extract boundary entropies from results
    boundary_entropies_data = results.get('boundary_entropies_per_timestep', [])
    if not boundary_entropies_data:
        print("⚠️  No boundary entropy data found in results")
        return None
    
    # Use the last timestep for analysis (most evolved state)
    final_boundary_data = boundary_entropies_data[-1]
    
    # Extract multiple region analysis
    multiple_regions = final_boundary_data.get('multiple_regions', {})
    if not multiple_regions:
        print("⚠️  No multiple region analysis found in boundary entropy data")
        return None
    
    # Prepare data for RT surface analysis
    rt_analysis_data = []
    
    for region_key, region_data in multiple_regions.items():
        region = region_data.get('region', [])
        entropy = region_data.get('entropy', 0.0)
        
        if not region or entropy <= 0:
            continue
            
        # Calculate RT surface area for this region
        # RT surface is the minimal surface separating this region from its complement
        complement = [i for i in range(len(mi_matrix)) if i not in region]
        
        # Find edges crossing between region and complement
        rt_edges = []
        for i in region:
            for j in complement:
                if mi_matrix[i, j] > 0:  # Edge exists
                    rt_edges.append((i, j))
        
        # Calculate RT surface area (sum of edge weights)
        rt_area = sum(mi_matrix[i, j] for i, j in rt_edges)
        
        if rt_area > 0:
            rt_analysis_data.append({
                'region': region,
                'region_size': len(region),
                'entropy': entropy,
                'rt_area': rt_area,
                'rt_edges': rt_edges
            })
    
    if len(rt_analysis_data) < 3:
        print("⚠️  Insufficient data points for RT surface analysis")
        return None
    
    # Extract data for fitting
    entropies = [data['entropy'] for data in rt_analysis_data]
    rt_areas = [data['rt_area'] for data in rt_analysis_data]
    region_sizes = [data['region_size'] for data in rt_analysis_data]
    
    # Fit linear relationship S(A) ∝ Area_RT(A)
    from scipy import stats
    
    # Linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(rt_areas, entropies)
    r_squared = r_value ** 2
    
    # Calculate confidence intervals
    n = len(rt_analysis_data)
    t_critical = stats.t.ppf(0.975, n-2)  # 95% confidence
    slope_ci = t_critical * std_err
    
    # Test for pure state condition: S(A) = S(B) for complementary regions
    pure_state_tests = []
    for data in rt_analysis_data:
        region = data['region']
        complement = [i for i in range(len(mi_matrix)) if i not in region]
        
        # Find entropy of complement
        complement_entropy = None
        for other_data in rt_analysis_data:
            if set(other_data['region']) == set(complement):
                complement_entropy = other_data['entropy']
                break
        
        if complement_entropy is not None:
            entropy_diff = abs(data['entropy'] - complement_entropy)
            pure_state_tests.append({
                'region': region,
                'complement': complement,
                'entropy_A': data['entropy'],
                'entropy_B': complement_entropy,
                'entropy_difference': entropy_diff,
                'is_pure_state': entropy_diff < 0.1  # Threshold for pure state
            })
    
    # Calculate pure state statistics
    pure_state_fraction = sum(1 for test in pure_state_tests if test['is_pure_state']) / len(pure_state_tests) if pure_state_tests else 0
    
    analysis_results = {
        'rt_analysis_data': rt_analysis_data,
        'linear_fit': {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_squared),
            'p_value': float(p_value),
            'slope_confidence_interval': float(slope_ci),
            'equation': f"S(A) = {slope:.4f} × Area_RT(A) + {intercept:.4f}"
        },
        'pure_state_analysis': {
            'pure_state_tests': pure_state_tests,
            'pure_state_fraction': float(pure_state_fraction),
            'mean_entropy_difference': float(np.mean([test['entropy_difference'] for test in pure_state_tests])) if pure_state_tests else 0
        },
        'statistics': {
            'num_regions_analyzed': len(rt_analysis_data),
            'entropy_range': [float(min(entropies)), float(max(entropies))],
            'rt_area_range': [float(min(rt_areas)), float(max(rt_areas))],
            'region_size_range': [min(region_sizes), max(region_sizes)]
        }
    }
    
    print(f"✅ RT Surface Analysis Complete:")
    print(f"   - Linear fit: {analysis_results['linear_fit']['equation']}")
    print(f"   - R² = {analysis_results['linear_fit']['r_squared']:.4f}")
    print(f"   - Pure state fraction: {analysis_results['pure_state_analysis']['pure_state_fraction']:.2f}")
    
    return analysis_results

def comprehensive_analysis(results_file, output_dir=None):
    """Run comprehensive analysis on experiment results"""
    print(f"Loading results from: {results_file}")
    results = load_experiment_results(results_file)
    
    if output_dir is None:
        output_dir = f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract filename for analysis
    filename = os.path.basename(results_file)
    
    analysis_results = {
        'experiment_info': {
            'source_file': filename,
            'device': results.get('device', 'unknown'),
            'shots': results.get('shots', 0),
            'num_qubits': results.get('num_qubits', 0),
            'geometry_type': results.get('geometry', 'unknown'),
            'curvature_strength': results.get('curvature', 0)
        }
    }
    
    # 1. Lorentzian Geometry Analysis
    print("Analyzing Lorentzian geometry...")
    lorentzian_analysis = analyze_lorentzian_geometry(results)
    if lorentzian_analysis:
        analysis_results['lorentzian_analysis'] = lorentzian_analysis
        print(f"✓ Lorentzian action: {lorentzian_analysis['lorentzian_action']:.6f}")
        print(f"✓ P-value: {lorentzian_analysis['p_value']:.6f}")
        print(f"✓ Significant: {lorentzian_analysis['is_significant']}")
    
    # 2. Check if we have counts data or boundary entropy data for advanced analyses
    has_counts_data = 'counts' in results and results['counts'] is not None and len(results['counts']) > 0
    has_boundary_entropy_data = 'boundary_entropies_per_timestep' in results and results['boundary_entropies_per_timestep'] and len(results['boundary_entropies_per_timestep']) > 0
    
    print(f"DEBUG: has_counts_data = {has_counts_data}")
    print(f"DEBUG: has_boundary_entropy_data = {has_boundary_entropy_data}")
    print(f"DEBUG: 'counts' in results = {'counts' in results}")
    print(f"DEBUG: 'boundary_entropies_per_timestep' in results = {'boundary_entropies_per_timestep' in results}")
    if 'boundary_entropies_per_timestep' in results:
        print(f"DEBUG: len(boundary_entropies_per_timestep) = {len(results['boundary_entropies_per_timestep'])}")
    
    if has_counts_data:
        print("✓ Found quantum measurement counts - running advanced holographic analyses...")
        
        # 2. Mutual Information Analysis
        print("Computing mutual information matrix...")
        mi_matrix = compute_mutual_information_matrix(results['counts'], results['num_qubits'])
        analysis_results['mi_analysis'] = {
            'max_mi': float(np.max(mi_matrix)),
            'mean_mi': float(np.mean(mi_matrix)),
            'non_zero_pairs': int(np.sum(mi_matrix > 0)),
            'mi_matrix': mi_matrix.tolist()
        }
        print(f"✓ Max MI: {np.max(mi_matrix):.6f}")
        print(f"✓ Non-zero MI pairs: {np.sum(mi_matrix > 0)}")
        
        # 3. RT Surface Approximation
        print("Computing RT surface approximation...")
        rt_surfaces, distances = compute_rt_surface_approximation(mi_matrix)
        
        # Analyze RT consistency
        rt_ratios = [data['rt_ratio'] for data in rt_surfaces.values() if data['rt_ratio'] > 0]
        rt_consistency = np.std(rt_ratios) if rt_ratios else float('inf')
        
        analysis_results['rt_analysis'] = {
            'rt_surfaces': rt_surfaces,
            'rt_consistency': float(rt_consistency),
            'num_rt_surfaces': len(rt_surfaces),
            'mean_rt_ratio': float(np.mean(rt_ratios)) if rt_ratios else 0.0
        }
        print(f"✓ RT surfaces computed: {len(rt_surfaces)}")
        print(f"✓ RT consistency: {rt_consistency:.6f}")
        
        # 4. Geometry Embedding
        print("Embedding geometry via MDS...")
        coordinates = MDS(n_components=2, dissimilarity='precomputed', random_state=42).fit_transform(distances)
        
        # Analyze geometry
        geometry_radius = np.max(np.linalg.norm(coordinates, axis=1))
        correlation = pearsonr(distances.flatten(), np.linalg.norm(coordinates[:, None] - coordinates, axis=2).flatten())[0]
        
        analysis_results['geometry_analysis'] = {
            'coordinates': coordinates.tolist(),
            'geometry_radius': float(geometry_radius),
            'mi_distance_correlation': float(correlation),
            'distances': distances.tolist()
        }
        print(f"✓ Geometry radius: {geometry_radius:.6f}")
        print(f"✓ MI-Distance correlation: {correlation:.6f}")
        
        # 5. Advanced Holographic Analyses
        print("\n" + "="*60)
        print("ADVANCED HOLOGRAPHIC ANALYSES")
        print("="*60)
        
        # RT Verification Analysis
        rt_results = rt_verification_analysis(mi_matrix, coordinates)
        analysis_results['rt_verification'] = rt_results
        
        # Holographic Entropy Scaling Analysis
        entropy_scaling_results = holographic_entropy_scaling_analysis(mi_matrix, results['num_qubits'])
        analysis_results['entropy_scaling'] = entropy_scaling_results
        
        # Boundary Dynamics Analysis
        boundary_dynamics_results = boundary_dynamics_analysis(results, mi_matrix, coordinates)
        analysis_results['boundary_dynamics'] = boundary_dynamics_results
        
        # RT Surface vs. Boundary Entropy Analysis
        rt_entropy_results = rt_surface_entropy_analysis(results, mi_matrix, coordinates)
        analysis_results['rt_entropy_analysis'] = rt_entropy_results
        
    elif has_boundary_entropy_data:
        print("✓ Found boundary entropy data - running RT surface entropy analysis...")
        
        # RT Surface vs. Boundary Entropy Analysis (using mutual information from results)
        if 'mutual_information_per_timestep' in results and results['mutual_information_per_timestep']:
            # Convert mutual information dict to matrix
            mi_dict = results['mutual_information_per_timestep'][-1]  # Use last timestep
            num_qubits = results['num_qubits']
            mi_matrix = np.zeros((num_qubits, num_qubits))
            
            for key, value in mi_dict.items():
                if key.startswith('I_'):
                    i, j = map(int, key[2:].split(','))
                    mi_matrix[i, j] = value
                    mi_matrix[j, i] = value  # Symmetric
            
            # Create dummy coordinates for analysis
            coordinates = np.random.rand(num_qubits, 2)  # Placeholder coordinates
            
            rt_entropy_results = rt_surface_entropy_analysis(results, mi_matrix, coordinates)
            analysis_results['rt_entropy_analysis'] = rt_entropy_results
        

        
        # 8. Generate robustness plots
        print("\n5. Generating Robustness Analysis Plots...")
        
        # Bootstrap confidence intervals plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 3, 1)
        mi_lower = np.array(bootstrap_results['mi_confidence_intervals']['lower'])
        mi_upper = np.array(bootstrap_results['mi_confidence_intervals']['upper'])
        mi_mean = np.array(bootstrap_results['mi_confidence_intervals']['mean'])
        
        # Plot MI matrix with confidence intervals
        im = plt.imshow(mi_mean, cmap='viridis', aspect='auto')
        plt.colorbar(im)
        plt.title('MI Matrix (Bootstrap Mean)')
        plt.xlabel('Qubit Index')
        plt.ylabel('Qubit Index')
        
        plt.subplot(2, 3, 2)
        # Plot confidence interval width
        ci_width = mi_upper - mi_lower
        im = plt.imshow(ci_width, cmap='plasma', aspect='auto')
        plt.colorbar(im)
        plt.title('MI Confidence Interval Width')
        plt.xlabel('Qubit Index')
        plt.ylabel('Qubit Index')
        
        plt.subplot(2, 3, 3)
        # Scrambling test results
        plt.bar(['Original', 'Scrambled'], 
               [scrambling_results['original_mi_mean'], scrambling_results['scrambled_mi_mean']],
               yerr=[scrambling_results['original_mi_std'], scrambling_results['scrambled_mi_std']],
               capsize=5)
        plt.title('MI: Original vs Scrambled')
        plt.ylabel('Mean MI')
        if scrambling_results['significantly_different']:
            plt.text(0.5, 0.9, 'SIGNIFICANT DIFFERENCE', 
                    transform=plt.gca().transAxes, ha='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.7))
        
        plt.subplot(2, 3, 4)
        # Embedding quality metrics
        methods = list(embedding_robustness.keys())
        stresses = [embedding_robustness[m]['stress'] for m in methods if not np.isnan(embedding_robustness[m]['stress'])]
        valid_methods = [m for m in methods if not np.isnan(embedding_robustness[m]['stress'])]
        
        plt.bar(range(len(valid_methods)), stresses)
        plt.xticks(range(len(valid_methods)), valid_methods, rotation=45)
        plt.title('Embedding Stress Scores')
        plt.ylabel('Stress')
        
        plt.subplot(2, 3, 5)
        # Curvature confidence intervals
        curvature_ci = bootstrap_results['curvature_confidence_intervals']
        plt.errorbar([1], [curvature_ci['mean']], 
                    yerr=[[curvature_ci['mean'] - curvature_ci['lower']], 
                          [curvature_ci['upper'] - curvature_ci['mean']]], 
                    fmt='o', capsize=5, markersize=10)
        plt.title('Curvature Estimate\nwith 95% CI')
        plt.ylabel('Curvature')
        plt.xticks([1], ['Estimated'])
        
        plt.subplot(2, 3, 6)
        # Embedding quality summary
        quality_metrics = [
            embedding_quality['stress'],
            embedding_quality['shepard_correlation'],
            embedding_quality['local_global_ratio']
        ]
        metric_names = ['Stress', 'Shepard Corr', 'Local/Global Ratio']
        
        plt.bar(metric_names, quality_metrics)
        plt.title('Embedding Quality Metrics')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'robustness_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Robustness analysis plots saved to {output_dir}/robustness_analysis.png")
    
    else:
        print("⚠️  No quantum measurement counts found - skipping advanced holographic analyses")
        print("   (This is normal for experiments that only compute Lorentzian geometry)")
        analysis_results['analysis_note'] = "Advanced holographic analyses require quantum measurement counts, which are not available in this experiment."
    
    # 6. Generate plots
    print("Generating analysis plots...")
    generate_analysis_plots(analysis_results, output_dir)
    
    # 7. Save comprehensive results
    results_file = os.path.join(output_dir, 'comprehensive_analysis.json')
    
    # Make results JSON serializable with debugging
    def make_json_serializable(obj, path=""):
        try:
            if isinstance(obj, dict):
                return {k: make_json_serializable(v, f"{path}.{k}" if path else k) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(v, f"{path}[{i}]" if path else f"[{i}]") for i, v in enumerate(obj)]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, bool):
                return bool(obj)  # Ensure it's a Python bool
            elif obj is None:
                return None
            elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'bool':
                print(f"Found problematic bool at {path}: {obj} (type: {type(obj)})")
                return bool(obj)
            else:
                return obj
        except Exception as e:
            print(f"Error serializing object at {path}: {obj} (type: {type(obj)}) - {e}")
            return str(obj)
    
    print("Converting results to JSON serializable format...")
    serializable_results = make_json_serializable(analysis_results)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # 8. Generate summary
    summary_file = os.path.join(output_dir, 'analysis_summary.txt')
    generate_analysis_summary(analysis_results, output_dir)
    
    print(f"\n✓ Comprehensive analysis completed!")
    print(f"✓ Results saved in: {output_dir}")
    print(f"✓ Analysis file: {results_file}")
    print(f"✓ Summary file: {summary_file}")
    
    return analysis_results

def generate_analysis_plots(analysis_results, output_dir):
    """Generate comprehensive analysis plots"""
    
    # 1. MI Matrix Heatmap
    if 'mi_analysis' in analysis_results:
        plt.figure(figsize=(10, 8))
        mi_matrix = np.array(analysis_results['mi_analysis']['mi_matrix'])
        plt.imshow(mi_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label='Mutual Information')
        plt.title('Mutual Information Matrix')
        plt.xlabel('Qubit Index')
        plt.ylabel('Qubit Index')
        plt.savefig(os.path.join(output_dir, 'mi_matrix_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. RT Surface Analysis
    if 'rt_analysis' in analysis_results:
        rt_surfaces = analysis_results['rt_analysis']['rt_surfaces']
        if rt_surfaces:
            rt_ratios = [data['rt_ratio'] for data in rt_surfaces.values() if data['rt_ratio'] > 0]
            rt_lengths = [data['rt_length'] for data in rt_surfaces.values() if data['rt_length'] > 0]
            
            # Ensure we have matching data for plotting
            if len(rt_ratios) == len(rt_lengths) and len(rt_ratios) > 0:
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.hist(rt_ratios, bins=20, alpha=0.7, color='blue')
                plt.xlabel('RT Ratio (Boundary MI / RT Length)')
                plt.ylabel('Frequency')
                plt.title('RT Surface Ratio Distribution')
                
                plt.subplot(1, 2, 2)
                plt.scatter(rt_lengths, rt_ratios, alpha=0.6)
                plt.xlabel('RT Surface Length')
                plt.ylabel('RT Ratio')
                plt.title('RT Length vs RT Ratio')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'rt_surface_analysis.png'), dpi=300, bbox_inches='tight')
                plt.close()
    
    # 3. Geometry Embedding
    if 'geometry_analysis' in analysis_results:
        coordinates = np.array(analysis_results['geometry_analysis']['coordinates'])
        
        plt.figure(figsize=(10, 8))
        plt.scatter(coordinates[:, 0], coordinates[:, 1], s=100, alpha=0.7)
        for i, (x, y) in enumerate(coordinates):
            plt.annotate(f'q{i}', (x, y), xytext=(5, 5), textcoords='offset points')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Reconstructed Geometry via MDS')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'geometry_embedding.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. RT Verification Plot
    if 'rt_verification' in analysis_results:
        rt_verification = analysis_results['rt_verification']
        plt.figure(figsize=(10, 6))
        plt.scatter(rt_verification['path_lengths'], rt_verification['mi_values'], alpha=0.7)
        plt.xlabel('Minimal Path Length in Bulk Geometry')
        plt.ylabel('Mutual Information')
        plt.title(f'RT Verification: MI vs Minimal Path\nCorrelation: {rt_verification["correlation"]:.4f}')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'rt_verification.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Holographic Entropy Scaling Plot
    if 'entropy_scaling' in analysis_results:
        entropy_scaling = analysis_results['entropy_scaling']
        entropy_data = entropy_scaling['entropy_data']
        
        sizes = list(entropy_data.keys())
        entropies = [entropy_data[size]['mean_entropy'] for size in sizes]
        errors = [entropy_data[size]['std_entropy'] for size in sizes]
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(sizes, entropies, yerr=errors, marker='o', capsize=5)
        plt.xlabel('Boundary Region Size')
        plt.ylabel('Entanglement Entropy')
        plt.title('Holographic Entropy Scaling')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'entropy_scaling.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Boundary Dynamics Plot
    if 'boundary_dynamics' in analysis_results:
        boundary_dynamics = analysis_results['boundary_dynamics']
        plt.figure(figsize=(10, 6))
        plt.scatter(boundary_dynamics['bulk_geometry_features'], boundary_dynamics['boundary_observables'], alpha=0.7)
        plt.xlabel('Bulk Geometry Feature (Distance from Center)')
        plt.ylabel('Boundary Observable (Average MI)')
        plt.title(f'Boundary Dynamics: Observables vs Bulk Geometry\nCorrelation: {boundary_dynamics["correlation"]:.4f}')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'boundary_dynamics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 7. RT Surface vs. Boundary Entropy Plot
    if 'rt_entropy_analysis' in analysis_results:
        rt_entropy = analysis_results['rt_entropy_analysis']
        rt_data = rt_entropy['rt_analysis_data']
        
        if rt_data:
            entropies = [data['entropy'] for data in rt_data]
            rt_areas = [data['rt_area'] for data in rt_data]
            region_sizes = [data['region_size'] for data in rt_data]
            
            plt.figure(figsize=(12, 8))
            
            # Main plot: S(A) vs Area_RT(A)
            plt.subplot(2, 2, 1)
            plt.scatter(rt_areas, entropies, c=region_sizes, cmap='viridis', s=100, alpha=0.7)
            
            # Add linear fit line
            linear_fit = rt_entropy['linear_fit']
            x_fit = np.linspace(min(rt_areas), max(rt_areas), 100)
            y_fit = linear_fit['slope'] * x_fit + linear_fit['intercept']
            plt.plot(x_fit, y_fit, 'r--', linewidth=2, 
                    label=f"Fit: {linear_fit['equation']}\nR² = {linear_fit['r_squared']:.4f}")
            
            plt.xlabel('RT Surface Area')
            plt.ylabel('Boundary Entropy S(A)')
            plt.title('Holographic Principle: S(A) ∝ Area_RT(A)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Colorbar for region sizes
            cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'))
            cbar.set_label('Region Size')
            
            # Pure state test plot
            plt.subplot(2, 2, 2)
            pure_state_tests = rt_entropy['pure_state_analysis']['pure_state_tests']
            if pure_state_tests:
                entropy_A = [test['entropy_A'] for test in pure_state_tests]
                entropy_B = [test['entropy_B'] for test in pure_state_tests]
                colors = ['green' if test['is_pure_state'] else 'red' for test in pure_state_tests]
                
                plt.scatter(entropy_A, entropy_B, c=colors, s=100, alpha=0.7)
                plt.plot([0, max(max(entropy_A), max(entropy_B))], [0, max(max(entropy_A), max(entropy_B))], 
                        'k--', alpha=0.5, label='S(A) = S(B)')
                plt.xlabel('Entropy S(A)')
                plt.ylabel('Entropy S(B)')
                plt.title(f'Pure State Test\nFraction: {rt_entropy["pure_state_analysis"]["pure_state_fraction"]:.2f}')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Region size vs entropy
            plt.subplot(2, 2, 3)
            plt.scatter(region_sizes, entropies, alpha=0.7)
            plt.xlabel('Region Size')
            plt.ylabel('Boundary Entropy S(A)')
            plt.title('Entropy vs Region Size')
            plt.grid(True, alpha=0.3)
            
            # RT area vs region size
            plt.subplot(2, 2, 4)
            plt.scatter(region_sizes, rt_areas, alpha=0.7)
            plt.xlabel('Region Size')
            plt.ylabel('RT Surface Area')
            plt.title('RT Area vs Region Size')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'rt_surface_entropy_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ RT Surface vs. Boundary Entropy plot saved")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'boundary_dynamics.png'), dpi=300, bbox_inches='tight')
        plt.close()

def generate_analysis_summary(analysis_results, output_dir):
    """Generate a comprehensive analysis summary"""
    summary = f"""
Comprehensive Curvature Analysis Summary
==================================================

Source File: {analysis_results['experiment_info']['source_file']}
Device: {analysis_results['experiment_info']['device']}
Shots: {analysis_results['experiment_info']['shots']}
Qubits: {analysis_results['experiment_info']['num_qubits']}
Geometry Type: {analysis_results['experiment_info']['geometry_type']}
Curvature Strength: {analysis_results['experiment_info']['curvature_strength']}

LORENTZIAN GEOMETRY ANALYSIS:
----------------------------
"""
    
    if 'lorentzian_analysis' in analysis_results:
        lorentzian = analysis_results['lorentzian_analysis']
        summary += f"""
Lorentzian Action: {lorentzian.get('lorentzian_action', 'N/A')}
Chi-squared p-value: {lorentzian.get('chi2_p_value', 'N/A')}
Lognormal p-value: {lorentzian.get('lognorm_p_value', 'N/A')}
Significantly Curved: {lorentzian.get('is_significantly_curved', 'N/A')}
"""

    summary += """
ADVANCED HOLOGRAPHIC ANALYSES:
------------------------------
"""
    
    if 'mi_analysis' in analysis_results:
        mi = analysis_results['mi_analysis']
        summary += f"""
Mutual Information Analysis:
- Max MI: {mi.get('max_mi', 'N/A')}
- Non-zero MI pairs: {mi.get('non_zero_pairs', 'N/A')}
- Average MI: {mi.get('average_mi', 'N/A')}
"""

    if 'rt_analysis' in analysis_results:
        rt = analysis_results['rt_analysis']
        summary += f"""
RT Surface Analysis:
- RT surfaces computed: {rt.get('num_rt_surfaces', 'N/A')}
- RT consistency: {rt.get('rt_consistency', 'N/A')}
- RT ratio range: {rt.get('mean_rt_ratio', 'N/A')}
"""

    if 'geometry_analysis' in analysis_results:
        geom = analysis_results['geometry_analysis']
        summary += f"""
Geometry Embedding:
- Geometry radius: {geom.get('geometry_radius', 'N/A')}
- MI-Distance correlation: {geom.get('mi_distance_correlation', 'N/A')}
- MDS stress: {geom.get('mds_stress', 'N/A')}
"""

    if 'entropy_scaling' in analysis_results:
        entropy_scaling = analysis_results['entropy_scaling']
        summary += f"""
Holographic Entropy Scaling:
- Area Scaling R²: {entropy_scaling.get('area_scaling', {}).get('r_squared', 'N/A')}
- Volume Scaling R²: {entropy_scaling.get('volume_scaling', {}).get('r_squared', 'N/A')}
- Supports Holographic Principle: {entropy_scaling.get('holographic_support', 'N/A')}
"""

    if 'boundary_dynamics' in analysis_results:
        boundary_dynamics = analysis_results['boundary_dynamics']
        summary += f"""
Boundary Dynamics:
- Boundary-Bulk Correlation: {boundary_dynamics.get('correlation', 'N/A')}
- P-value: {boundary_dynamics.get('p_value', 'N/A')}
"""

    if 'rt_verification' in analysis_results:
        rt_verification = analysis_results['rt_verification']
        summary += f"""
RT Verification:
- MI vs Minimal Path Correlation: {rt_verification.get('correlation', 'N/A')}
- P-value: {rt_verification.get('p_value', 'N/A')}
- Number of pairs analyzed: {rt_verification.get('num_pairs', 'N/A')}
"""

    if 'bootstrap_analysis' in analysis_results:
        robustness = analysis_results['bootstrap_analysis']
        scrambling_results = analysis_results['scrambling_test']
        embedding_quality = analysis_results['embedding_quality']
        
        summary += """
ROBUSTNESS ANALYSIS SUMMARY:
-----------------------------
"""

        summary += f"""
Bootstrap Resampling Results:
- Bootstrap samples: {robustness.get('bootstrap_samples', 'N/A')}
- Confidence level: {robustness.get('confidence_level', 'N/A')}

Curvature confidence interval:
- Curvature estimate: {robustness.get('curvature_confidence_intervals', {}).get('mean', 'N/A')}
- 95% CI: [{robustness.get('curvature_confidence_intervals', {}).get('lower', 'N/A')}, {robustness.get('curvature_confidence_intervals', {}).get('upper', 'N/A')}]

Stress confidence interval:
- MDS stress score: {robustness.get('stress_confidence_intervals', {}).get('mean', 'N/A')}
- Stress 95% CI: [{robustness.get('stress_confidence_intervals', {}).get('lower', 'N/A')}, {robustness.get('stress_confidence_intervals', {}).get('upper', 'N/A')}]

Scrambling Test Results:
- Original MI mean: {scrambling_results.get('original_mi_mean', 'N/A')}
- Scrambled MI mean: {scrambling_results.get('scrambled_mi_mean', 'N/A')}
- Z-score: {scrambling_results.get('z_score', 'N/A')}
- P-value: {scrambling_results.get('p_value', 'N/A')}
- Significantly different: {scrambling_results.get('significantly_different', 'N/A')}

Embedding Quality Assessment:
- MDS stress score: {embedding_quality.get('stress', 'N/A')}
- Shepard correlation: {embedding_quality.get('shepard_correlation', 'N/A')}
- Local/global ratio: {embedding_quality.get('local_global_ratio', 'N/A')}
- Overall quality: {embedding_quality.get('embedding_quality', 'N/A')}

Embedding Robustness Across Methods:
"""
        for method, data in analysis_results['embedding_robustness'].items():
            if not np.isnan(data['stress']):
                summary += f"- {method}: Stress = {data['stress']:.6f}\n"

        summary += """
Robustness Conclusions:
"""
        if scrambling_results['significantly_different'] and embedding_quality['stress'] < 0.3:
            summary += "STRONG EVIDENCE: Geometric structure is robust and statistically significant\n"
            summary += "The observed geometry survives noise and is not an artifact\n"
            summary += "Confidence intervals provide reliable estimates of geometric properties\n"
        elif scrambling_results['significantly_different']:
            summary += "MODERATE EVIDENCE: Structure is real but embedding quality could be improved\n"
            summary += "Consider using different embedding techniques or more qubits\n"
        else:
            summary += "WEAK EVIDENCE: Structure may be fragile or due to noise\n"
            summary += "Results should be interpreted with caution\n"
            summary += "Consider increasing shot count or improving circuit design\n"

    summary += """
OVERALL ASSESSMENT:
"""
    
    if 'lorentzian_analysis' in analysis_results:
        if analysis_results['lorentzian_analysis']['is_significant']:
            summary += "+ Strong evidence for Lorentzian geometry\n"
        else:
            summary += "- Weak evidence for Lorentzian geometry\n"
    
    if 'rt_analysis' in analysis_results:
        if analysis_results['rt_analysis']['rt_consistency'] < 0.1:
            summary += "+ RT surface consistency high, supporting holographic duality\n"
        else:
            summary += "- RT surface consistency low, may need more entanglement\n"
    
    if 'geometry_analysis' in analysis_results:
        correlation = analysis_results['geometry_analysis']['mi_distance_correlation']
        if correlation > 0.7:
            summary += "+ Strong correlation between MI and geometric distance\n"
        elif correlation > 0.3:
            summary += "+ Moderate correlation between MI and geometric distance\n"
        else:
            summary += "- Weak correlation, may indicate insufficient entanglement\n"
    
    if 'entropy_scaling' in analysis_results:
        if analysis_results['entropy_scaling']['holographic_support']:
            summary += "+ Entropy scaling supports holographic principle (area > volume)\n"
        else:
            summary += "- Entropy scaling does not clearly support holographic principle\n"
    
    if 'boundary_dynamics' in analysis_results:
        boundary_corr = analysis_results['boundary_dynamics']['correlation']
        if abs(boundary_corr) > 0.5:
            summary += "+ Strong boundary-bulk dynamics correlation\n"
        elif abs(boundary_corr) > 0.3:
            summary += "+ Moderate boundary-bulk dynamics correlation\n"
        else:
            summary += "- Weak boundary-bulk dynamics correlation\n"

    # Save summary
    summary_file = os.path.join(output_dir, 'analysis_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"✓ Analysis summary saved to {summary_file}")
    
    # Save detailed results as JSON
    # Fix JSON serialization by converting numpy types and ensuring all values are JSON serializable
    def make_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return bool(obj)  # Ensure it's a Python bool
        elif obj is None:
            return None
        else:
            return obj
    
    serializable_results = make_json_serializable(analysis_results)
    
    json_file = os.path.join(output_dir, 'comprehensive_analysis.json')
    with open(json_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"✓ Detailed results saved to {json_file}")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def bootstrap_mi_analysis(counts, num_qubits, n_bootstrap=1000, confidence_level=0.95):
    """
    Bootstrap resampling of mutual information matrix to compute confidence intervals.
    
    Args:
        counts: Dictionary of measurement counts
        num_qubits: Number of qubits
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals (0.95 = 95%)
    
    Returns:
        Dictionary with bootstrap statistics
    """
    print(f"Running bootstrap resampling with {n_bootstrap} samples...")
    
    # Convert counts to probability distribution
    total_shots = sum(counts.values())
    prob_dist = {k: v/total_shots for k, v in counts.items()}
    
    # Bootstrap samples
    bootstrap_mi_matrices = []
    bootstrap_curvatures = []
    bootstrap_stress_scores = []
    
    for i in range(n_bootstrap):
        if i % 100 == 0:
            print(f"  Bootstrap sample {i}/{n_bootstrap}")
        
        # Resample with replacement
        resampled_counts = {}
        for _ in range(total_shots):
            # Sample from probability distribution
            bitstring = np.random.choice(list(prob_dist.keys()), p=list(prob_dist.values()))
            resampled_counts[bitstring] = resampled_counts.get(bitstring, 0) + 1
        
        # Compute MI matrix for resampled data
        mi_matrix = compute_mutual_information_matrix(resampled_counts, num_qubits)
        bootstrap_mi_matrices.append(mi_matrix)
        
        # Compute geometric metrics
        try:
            # MDS embedding
            dissimilarity_matrix = 1 - mi_matrix  # Convert MI to dissimilarity
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            coords = mds.fit_transform(dissimilarity_matrix)
            stress = mds.stress_
            bootstrap_stress_scores.append(stress)
            
            # Estimate curvature from coordinates (simplified)
            if len(coords) >= 3:
                # Compute triangle areas to estimate curvature
                areas = []
                for i in range(len(coords)):
                    for j in range(i+1, len(coords)):
                        for k in range(j+1, len(coords)):
                            # Triangle area
                            area = 0.5 * abs(np.cross(coords[j] - coords[i], coords[k] - coords[i]))
                            areas.append(area)
                
                if areas:
                    # Curvature estimate (inverse of average area)
                    curvature = 1.0 / (np.mean(areas) + 1e-8)
                    bootstrap_curvatures.append(curvature)
                else:
                    bootstrap_curvatures.append(0.0)
            else:
                bootstrap_curvatures.append(0.0)
                
        except Exception as e:
            print(f"    Warning: Bootstrap sample {i} failed: {e}")
            bootstrap_stress_scores.append(np.nan)
            bootstrap_curvatures.append(np.nan)
    
    # Compute confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    # MI matrix confidence intervals
    mi_matrices_array = np.array(bootstrap_mi_matrices)
    mi_lower = np.percentile(mi_matrices_array, lower_percentile, axis=0)
    mi_upper = np.percentile(mi_matrices_array, upper_percentile, axis=0)
    mi_mean = np.mean(mi_matrices_array, axis=0)
    
    # Curvature confidence intervals
    valid_curvatures = [c for c in bootstrap_curvatures if not np.isnan(c)]
    if valid_curvatures:
        curvature_lower = np.percentile(valid_curvatures, lower_percentile)
        curvature_upper = np.percentile(valid_curvatures, upper_percentile)
        curvature_mean = np.mean(valid_curvatures)
    else:
        curvature_lower = curvature_upper = curvature_mean = 0.0
    
    # Stress score confidence intervals
    valid_stress = [s for s in bootstrap_stress_scores if not np.isnan(s)]
    if valid_stress:
        stress_lower = np.percentile(valid_stress, lower_percentile)
        stress_upper = np.percentile(valid_stress, upper_percentile)
        stress_mean = np.mean(valid_stress)
    else:
        stress_lower = stress_upper = stress_mean = np.nan
    
    return {
        'mi_confidence_intervals': {
            'lower': mi_lower.tolist(),
            'upper': mi_upper.tolist(),
            'mean': mi_mean.tolist()
        },
        'curvature_confidence_intervals': {
            'lower': float(curvature_lower),
            'upper': float(curvature_upper),
            'mean': float(curvature_mean)
        },
        'stress_confidence_intervals': {
            'lower': float(stress_lower) if not np.isnan(stress_lower) else None,
            'upper': float(stress_upper) if not np.isnan(stress_upper) else None,
            'mean': float(stress_mean) if not np.isnan(stress_mean) else None
        },
        'bootstrap_samples': n_bootstrap,
        'confidence_level': confidence_level
    }

def scrambling_test(counts, num_qubits, n_scrambles=100):
    """
    Test robustness by randomly shuffling measurement outcomes across qubits.
    
    Args:
        counts: Dictionary of measurement counts
        num_qubits: Number of qubits
        n_scrambles: Number of scrambling tests
    
    Returns:
        Dictionary with scrambling test results
    """
    print(f"Running scrambling test with {n_scrambles} random shuffles...")
    
    # Original MI matrix
    original_mi = compute_mutual_information_matrix(counts, num_qubits)
    original_mi_mean = np.mean(original_mi)
    original_mi_std = np.std(original_mi)
    
    # Scrambled MI matrices
    scrambled_mi_means = []
    scrambled_mi_stds = []
    
    for i in range(n_scrambles):
        if i % 20 == 0:
            print(f"  Scramble test {i}/{n_scrambles}")
        
        # Create scrambled counts by shuffling qubit positions
        scrambled_counts = {}
        for bitstring, count in counts.items():
            # Convert bitstring to list of bits
            bits = list(bitstring)
            # Shuffle the bits
            np.random.shuffle(bits)
            # Convert back to string
            scrambled_bitstring = ''.join(bits)
            scrambled_counts[scrambled_bitstring] = count
        
        # Compute MI for scrambled data
        scrambled_mi = compute_mutual_information_matrix(scrambled_counts, num_qubits)
        scrambled_mi_means.append(np.mean(scrambled_mi))
        scrambled_mi_stds.append(np.std(scrambled_mi))
    
    # Statistical test: is original MI significantly different from scrambled?
    scrambled_mean = np.mean(scrambled_mi_means)
    scrambled_std = np.std(scrambled_mi_means)
    
    # Z-score for original vs scrambled
    z_score = (original_mi_mean - scrambled_mean) / (scrambled_std + 1e-8)
    p_value = 2 * (1 - norm.cdf(abs(z_score)))  # Two-tailed test
    
    return {
        'original_mi_mean': float(original_mi_mean),
        'original_mi_std': float(original_mi_std),
        'scrambled_mi_mean': float(scrambled_mean),
        'scrambled_mi_std': float(scrambled_std),
        'z_score': float(z_score),
        'p_value': float(p_value),
        'significantly_different': p_value < 0.05,
        'n_scrambles': n_scrambles
    }

def embedding_robustness_analysis(mi_matrix, num_qubits):
    """
    Test robustness across different embedding techniques and dimensionalities.
    
    Args:
        mi_matrix: Mutual information matrix
        num_qubits: Number of qubits
    
    Returns:
        Dictionary with embedding robustness results
    """
    print("Testing embedding robustness across different techniques...")
    
    dissimilarity_matrix = 1 - mi_matrix
    
    results = {}
    
    # Test different embedding techniques
    embedding_methods = {
        'MDS_2D': MDS(n_components=2, dissimilarity='precomputed', random_state=42),
        'MDS_3D': MDS(n_components=3, dissimilarity='precomputed', random_state=42),
        'TSNE_2D': TSNE(n_components=2, metric='precomputed', random_state=42),
        'Isomap_2D': Isomap(n_components=2, metric='precomputed'),
        'PCA_2D': PCA(n_components=2)
    }
    
    for name, method in embedding_methods.items():
        try:
            if 'PCA' in name:
                # PCA works on the original data, not dissimilarity
                coords = method.fit_transform(mi_matrix)
            else:
                coords = method.fit_transform(dissimilarity_matrix)
            
            # Compute stress score (for MDS) or reconstruction error
            if 'MDS' in name:
                stress = method.stress_
                results[name] = {
                    'stress': float(stress),
                    'coordinates': coords.tolist()
                }
            else:
                # For other methods, compute reconstruction error
                if 'PCA' in name:
                    reconstructed = method.inverse_transform(coords)
                    error = np.mean((mi_matrix - reconstructed) ** 2)
                else:
                    # Approximate stress for non-MDS methods
                    coords_dist = pairwise_distances(coords)
                    stress = np.mean((dissimilarity_matrix - coords_dist) ** 2)
                
                results[name] = {
                    'stress': float(stress),
                    'coordinates': coords.tolist()
                }
                
        except Exception as e:
            print(f"    Warning: {name} failed: {e}")
            results[name] = {'stress': np.nan, 'coordinates': None}
    
    return results

def quantify_embedding_error(mi_matrix, coordinates):
    """
    Quantify the quality of the embedding using various metrics.
    
    Args:
        mi_matrix: Mutual information matrix
        coordinates: 2D coordinates from embedding
    
    Returns:
        Dictionary with embedding quality metrics
    """
    dissimilarity_matrix = 1 - mi_matrix
    
    # Ensure diagonal is zero for distance matrix
    np.fill_diagonal(dissimilarity_matrix, 0.0)
    
    # MDS stress (how well dissimilarities are preserved)
    coords_dist = pairwise_distances(coordinates)
    stress = np.mean((dissimilarity_matrix - coords_dist) ** 2)
    
    # Shepard plot correlation
    try:
        flat_dissimilarity = squareform(dissimilarity_matrix)
        flat_coords_dist = squareform(coords_dist)
        correlation = np.corrcoef(flat_dissimilarity, flat_coords_dist)[0, 1]
    except Exception as e:
        print(f"    Warning: Could not compute Shepard correlation: {e}")
        correlation = 0.0
    
    # Local vs global structure preservation
    # Compute ratio of local to global distances
    local_distances = []
    global_distances = []
    
    for i in range(len(coordinates)):
        for j in range(i+1, len(coordinates)):
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            if dist < np.median(coords_dist):  # Local
                local_distances.append(dist)
            else:  # Global
                global_distances.append(dist)
    
    if local_distances and global_distances:
        local_global_ratio = np.mean(local_distances) / (np.mean(global_distances) + 1e-8)
    else:
        local_global_ratio = 1.0
    
    return {
        'stress': float(stress),
        'shepard_correlation': float(correlation) if not np.isnan(correlation) else 0.0,
        'local_global_ratio': float(local_global_ratio),
        'embedding_quality': 'good' if stress < 0.1 else 'moderate' if stress < 0.3 else 'poor'
    }

def analyze_curvature_phase_diagram(experiment_logs_dir):
    """
    Analyze phase diagram across multiple curvature values to establish systematic trends.
    This addresses reviewer concerns about isolated data points vs. systematic behavior.
    """
    print("=== CURVATURE PHASE DIAGRAM ANALYSIS ===")
    print("Establishing systematic trends across curvature values...")
    
    # Find all custom curvature experiment results
    results_files = []
    for filename in os.listdir(experiment_logs_dir):
        if filename.startswith('results_') and filename.endswith('.json'):
            filepath = os.path.join(experiment_logs_dir, filename)
            results_files.append(filepath)
    
    print(f"Found {len(results_files)} result files")
    
    # Extract curvature values and organize data
    curvature_data = {}
    
    for filepath in results_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Extract curvature value from filename or data
            curvature = None
            if 'spec' in data and 'curvature' in data['spec']:
                curvature = data['spec']['curvature']
            else:
                # Try to extract from filename
                import re
                match = re.search(r'curv(\d+(?:\.\d+)?)', filename)
                if match:
                    curvature = float(match.group(1))
            
            if curvature is not None:
                if curvature not in curvature_data:
                    curvature_data[curvature] = []
                curvature_data[curvature].append({
                    'filepath': filepath,
                    'data': data,
                    'filename': filename
                })
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    print(f"Organized data by curvature: {sorted(curvature_data.keys())}")
    
    # Analyze each curvature value
    phase_diagram_results = {}
    
    for curvature in sorted(curvature_data.keys()):
        print(f"\nAnalyzing curvature κ = {curvature}")
        
        curvature_results = []
        
        for result_info in curvature_data[curvature]:
            data = result_info['data']
            
            # Extract key metrics
            metrics = {
                'curvature': curvature,
                'filename': result_info['filename'],
                'device': data.get('spec', {}).get('device', 'unknown'),
                'num_qubits': data.get('spec', {}).get('num_qubits', 0),
                'shots': data.get('spec', {}).get('shots', 0),
                'geometry': data.get('spec', {}).get('geometry', 'unknown')
            }
            
            # Extract Lorentzian action if available
            if 'lorentzian_solution' in data:
                lorentzian_sol = data['lorentzian_solution']
                metrics['lorentzian_action'] = lorentzian_sol.get('stationary_action')
                metrics['edge_lengths'] = lorentzian_sol.get('stationary_edge_lengths', [])
                
                # Compute edge length statistics
                if metrics['edge_lengths']:
                    edge_lengths = np.array(metrics['edge_lengths'])
                    metrics['mean_edge_length'] = np.mean(edge_lengths)
                    metrics['std_edge_length'] = np.std(edge_lengths)
                    metrics['min_edge_length'] = np.min(edge_lengths)
                    metrics['max_edge_length'] = np.max(edge_lengths)
            
            # Extract mutual information if available
            if 'mutual_information_per_timestep' in data:
                mi_data = data['mutual_information_per_timestep']
                if mi_data and len(mi_data) > 0:
                    # Use first timestep for analysis
                    mi_timestep = mi_data[0]
                    
                    # Convert to matrix format
                    num_qubits = metrics['num_qubits']
                    mi_matrix = np.zeros((num_qubits, num_qubits))
                    
                    for key, value in mi_timestep.items():
                        if key.startswith('I_'):
                            # Parse qubit indices
                            parts = key[2:].split(',')
                            if len(parts) == 2:
                                i, j = int(parts[0]), int(parts[1])
                                if 0 <= i < num_qubits and 0 <= j < num_qubits:
                                    mi_matrix[i, j] = mi_matrix[j, i] = value
                    
                    # Compute MI statistics
                    mask = ~np.eye(num_qubits, dtype=bool)
                    mi_values = mi_matrix[mask]
                    
                    metrics['mean_mi'] = np.mean(mi_values)
                    metrics['std_mi'] = np.std(mi_values)
                    metrics['min_mi'] = np.min(mi_values)
                    metrics['max_mi'] = np.max(mi_values)
                    
                    # Compute Gromov delta (hyperbolicity measure)
                    if num_qubits >= 4:
                        # Use MI as distance metric (inverse relationship)
                        eps = 1e-8
                        mi_max = np.max(mi_matrix)
                        if mi_max > 0:
                            mi_normalized = mi_matrix / mi_max
                        else:
                            mi_normalized = mi_matrix
                        
                        distance_matrix = -np.log(mi_normalized + eps)
                        
                        # Compute Gromov delta
                        gromov_deltas = []
                        for i in range(num_qubits):
                            for j in range(i+1, num_qubits):
                                for k in range(j+1, num_qubits):
                                    for l in range(k+1, num_qubits):
                                        d_ij = distance_matrix[i, j]
                                        d_kl = distance_matrix[k, l]
                                        d_ik = distance_matrix[i, k]
                                        d_jl = distance_matrix[j, l]
                                        d_il = distance_matrix[i, l]
                                        d_jk = distance_matrix[j, k]
                                        
                                        # Gromov delta condition
                                        delta1 = (d_ij + d_kl) - max(d_ik + d_jl, d_il + d_jk)
                                        delta2 = (d_ik + d_jl) - max(d_ij + d_kl, d_il + d_jk)
                                        delta3 = (d_il + d_jk) - max(d_ij + d_kl, d_ik + d_jl)
                                        
                                        gromov_delta = max(delta1, delta2, delta3)
                                        if gromov_delta > 0:
                                            gromov_deltas.append(gromov_delta)
                        
                        if gromov_deltas:
                            metrics['gromov_delta_mean'] = np.mean(gromov_deltas)
                            metrics['gromov_delta_std'] = np.std(gromov_deltas)
                            metrics['gromov_delta_max'] = np.max(gromov_deltas)
                        else:
                            metrics['gromov_delta_mean'] = 0.0
                            metrics['gromov_delta_std'] = 0.0
                            metrics['gromov_delta_max'] = 0.0
            
            curvature_results.append(metrics)
        
        # Aggregate statistics for this curvature value
        if curvature_results:
            # Separate hardware and simulator results
            hardware_results = [r for r in curvature_results if 'ibm_' in r.get('device', '')]
            simulator_results = [r for r in curvature_results if 'sim' in r.get('device', '')]
            
            phase_diagram_results[curvature] = {
                'hardware_results': hardware_results,
                'simulator_results': simulator_results,
                'total_experiments': len(curvature_results),
                'hardware_experiments': len(hardware_results),
                'simulator_experiments': len(simulator_results)
            }
            
            print(f"  Curvature κ = {curvature}: {len(curvature_results)} experiments")
            print(f"    Hardware: {len(hardware_results)}, Simulator: {len(simulator_results)}")
    
    return phase_diagram_results

def generate_phase_diagram_plots(phase_diagram_results, output_dir):
    """Generate phase diagram plots showing systematic trends"""
    print("\n=== GENERATING PHASE DIAGRAM PLOTS ===")
    
    curvatures = sorted(phase_diagram_results.keys())
    
    # Prepare data for plotting
    curvature_values = []
    lorentzian_actions_hw = []
    lorentzian_actions_sim = []
    gromov_deltas_hw = []
    gromov_deltas_sim = []
    mean_mi_hw = []
    mean_mi_sim = []
    
    for curvature in curvatures:
        data = phase_diagram_results[curvature]
        
        # Hardware results
        if data['hardware_results']:
            hw_actions = [r.get('lorentzian_action', 0) for r in data['hardware_results'] if r.get('lorentzian_action') is not None]
            hw_gromov = [r.get('gromov_delta_mean', 0) for r in data['hardware_results'] if r.get('gromov_delta_mean') is not None]
            hw_mi = [r.get('mean_mi', 0) for r in data['hardware_results'] if r.get('mean_mi') is not None]
            
            if hw_actions:
                curvature_values.append(curvature)
                lorentzian_actions_hw.append(np.mean(hw_actions))
                gromov_deltas_hw.append(np.mean(hw_gromov) if hw_gromov else 0)
                mean_mi_hw.append(np.mean(hw_mi) if hw_mi else 0)
        
        # Simulator results
        if data['simulator_results']:
            sim_actions = [r.get('lorentzian_action', 0) for r in data['simulator_results'] if r.get('lorentzian_action') is not None]
            sim_gromov = [r.get('gromov_delta_mean', 0) for r in data['simulator_results'] if r.get('gromov_delta_mean') is not None]
            sim_mi = [r.get('mean_mi', 0) for r in data['simulator_results'] if r.get('mean_mi') is not None]
            
            if sim_actions:
                if curvature not in curvature_values:
                    curvature_values.append(curvature)
                lorentzian_actions_sim.append(np.mean(sim_actions))
                gromov_deltas_sim.append(np.mean(sim_gromov) if sim_gromov else 0)
                mean_mi_sim.append(np.mean(sim_mi) if sim_mi else 0)
    
    # Create phase diagram plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Curvature Phase Diagram: Systematic Trends Across κ Values', fontsize=16, fontweight='bold')
    
    # Plot 1: Lorentzian Action vs Curvature
    ax1 = axes[0, 0]
    if lorentzian_actions_hw:
        ax1.plot(curvature_values[:len(lorentzian_actions_hw)], lorentzian_actions_hw, 'o-', 
                color='red', linewidth=2, markersize=8, label='Hardware (IBM Brisbane)')
    if lorentzian_actions_sim:
        ax1.plot(curvature_values[:len(lorentzian_actions_sim)], lorentzian_actions_sim, 's-', 
                color='blue', linewidth=2, markersize=8, label='Simulator (FakeBrisbane)')
    
    ax1.set_xlabel('Curvature κ', fontsize=12)
    ax1.set_ylabel('Lorentzian Action', fontsize=12)
    ax1.set_title('Lorentzian Action vs Curvature', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Gromov Delta vs Curvature
    ax2 = axes[0, 1]
    if gromov_deltas_hw:
        ax2.plot(curvature_values[:len(gromov_deltas_hw)], gromov_deltas_hw, 'o-', 
                color='red', linewidth=2, markersize=8, label='Hardware')
    if gromov_deltas_sim:
        ax2.plot(curvature_values[:len(gromov_deltas_sim)], gromov_deltas_sim, 's-', 
                color='blue', linewidth=2, markersize=8, label='Simulator')
    
    ax2.set_xlabel('Curvature κ', fontsize=12)
    ax2.set_ylabel('Gromov Delta (Hyperbolicity)', fontsize=12)
    ax2.set_title('Hyperbolicity vs Curvature', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Mean MI vs Curvature
    ax3 = axes[1, 0]
    if mean_mi_hw:
        ax3.plot(curvature_values[:len(mean_mi_hw)], mean_mi_hw, 'o-', 
                color='red', linewidth=2, markersize=8, label='Hardware')
    if mean_mi_sim:
        ax3.plot(curvature_values[:len(mean_mi_sim)], mean_mi_sim, 's-', 
                color='blue', linewidth=2, markersize=8, label='Simulator')
    
    ax3.set_xlabel('Curvature κ', fontsize=12)
    ax3.set_ylabel('Mean Mutual Information', fontsize=12)
    ax3.set_title('Entanglement vs Curvature', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Hardware vs Simulator Comparison
    ax4 = axes[1, 1]
    if lorentzian_actions_hw and lorentzian_actions_sim:
        min_len = min(len(lorentzian_actions_hw), len(lorentzian_actions_sim))
        hw_vals = lorentzian_actions_hw[:min_len]
        sim_vals = lorentzian_actions_sim[:min_len]
        curv_vals = curvature_values[:min_len]
        
        ax4.scatter(hw_vals, sim_vals, c=curv_vals, cmap='viridis', s=100, alpha=0.7)
        ax4.plot([0, max(max(hw_vals), max(sim_vals))], [0, max(max(hw_vals), max(sim_vals))], 
                'k--', alpha=0.5, label='Perfect Agreement')
        
        ax4.set_xlabel('Hardware Lorentzian Action', fontsize=12)
        ax4.set_ylabel('Simulator Lorentzian Action', fontsize=12)
        ax4.set_title('Hardware vs Simulator Agreement', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label('Curvature κ', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'curvature_phase_diagram.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Phase diagram saved to: {plot_path}")
    
    plt.show()
    
    return plot_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comprehensive Curvature Analyzer')
    parser.add_argument('results_file', help='Path to experiment results JSON file')
    parser.add_argument('--output_dir', help='Output directory for analysis results')
    parser.add_argument('--phase_diagram', action='store_true', 
                       help='Generate phase diagram analysis across all curvature values')
    
    args = parser.parse_args()
    
    if args.phase_diagram:
        # Run phase diagram analysis
        experiment_logs_dir = os.path.join(os.path.dirname(args.results_file), '..', 'custom_curvature_experiment')
        if not os.path.exists(experiment_logs_dir):
            experiment_logs_dir = os.path.join(os.path.dirname(args.results_file), '..', '..', 'experiment_logs', 'custom_curvature_experiment')
        
        if os.path.exists(experiment_logs_dir):
            phase_results = analyze_curvature_phase_diagram(experiment_logs_dir)
            
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                generate_phase_diagram_plots(phase_results, args.output_dir)
                
                # Save phase diagram data
                phase_data_path = os.path.join(args.output_dir, 'phase_diagram_data.json')
                with open(phase_data_path, 'w') as f:
                    json.dump(phase_results, f, indent=2, default=str)
                print(f"Phase diagram data saved to: {phase_data_path}")
        else:
            print(f"Experiment logs directory not found: {experiment_logs_dir}")
    else:
        # Run standard analysis
        if not os.path.exists(args.results_file):
            print(f"Error: Results file {args.results_file} not found!")
            exit(1)
        
        comprehensive_analysis(args.results_file, args.output_dir) 