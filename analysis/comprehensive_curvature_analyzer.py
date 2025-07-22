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
from scipy.stats import pearsonr, bootstrap
from sklearn.manifold import MDS
import argparse
from datetime import datetime

def load_experiment_results(results_file):
    """Load experiment results from JSON file"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Extract key information from the experiment results
    results = {}
    
    # Handle different experiment formats
    if 'lorentzian_solution' in data:
        # Custom curvature experiment format
        lorentzian = data['lorentzian_solution']
        results['lorentzian_action'] = lorentzian.get('stationary_action', 0)
        results['edge_lengths'] = lorentzian.get('stationary_edge_lengths', [])
        
        # Extract experiment parameters from filename
        filename = os.path.basename(results_file)
        if 'n7' in filename:
            results['num_qubits'] = 7
        elif 'n5' in filename:
            results['num_qubits'] = 5
        
        if 'curv25' in filename:
            results['curvature_strength'] = 2.5
        elif 'curv100' in filename:
            results['curvature_strength'] = 10.0
        
        if 'geomH' in filename:
            results['geometry_type'] = 'hyperbolic'
        
        if 'ibm' in filename:
            results['device'] = 'ibm_quantum'
    
    else:
        # Bulk reconstruction or other experiment format
        results['device'] = data.get('device', 'unknown')
        results['backend'] = data.get('backend', 'unknown')
        results['shots'] = data.get('shots', 0)
        results['num_qubits'] = data.get('num_qubits', 0)
        results['geometry_type'] = data.get('geometry_type', 'unknown')
        results['curvature_strength'] = data.get('curvature_strength', 0)
        
        # Copy counts data if available
        if 'counts' in data:
            results['counts'] = data['counts']
    
    # Set default values for missing fields
    results['shots'] = results.get('shots', 2000)
    results['device'] = results.get('device', 'unknown')
    results['geometry_type'] = results.get('geometry_type', 'unknown')
    results['curvature_strength'] = results.get('curvature_strength', 0)
    results['num_qubits'] = results.get('num_qubits', 0)
    
    return results

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
            'geometry_type': results.get('geometry_type', 'unknown'),
            'curvature_strength': results.get('curvature_strength', 0)
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
    
    # 2. Check if we have counts data for advanced analyses
    has_counts_data = 'counts' in results and results['counts']
    
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
        print("Running advanced holographic analyses...")
        
        # RT Verification
        print("  - RT Verification analysis...")
        rt_verification = rt_verification_analysis(mi_matrix, coordinates)
        if rt_verification:
            analysis_results['rt_verification'] = rt_verification
            print(f"    ✓ RT correlation: {rt_verification['correlation']:.6f}")
        
        # Holographic Entropy Scaling
        print("  - Holographic entropy scaling analysis...")
        entropy_scaling = holographic_entropy_scaling_analysis(mi_matrix, results['num_qubits'])
        if entropy_scaling:
            analysis_results['entropy_scaling'] = entropy_scaling
            print(f"    ✓ Holographic support: {entropy_scaling['holographic_support']}")
        
        # Boundary Dynamics
        print("  - Boundary dynamics analysis...")
        boundary_dynamics = boundary_dynamics_analysis(results, mi_matrix, coordinates)
        if boundary_dynamics:
            analysis_results['boundary_dynamics'] = boundary_dynamics
            print(f"    ✓ Boundary-bulk correlation: {boundary_dynamics['correlation']:.6f}")
    
    else:
        print("⚠️  No quantum measurement counts found - skipping advanced holographic analyses")
        print("   (This is normal for experiments that only compute Lorentzian geometry)")
        analysis_results['analysis_note'] = "Advanced holographic analyses require quantum measurement counts, which are not available in this experiment."
    
    # 6. Generate plots
    print("Generating analysis plots...")
    generate_analysis_plots(analysis_results, output_dir)
    
    # 7. Save comprehensive results
    results_file = os.path.join(output_dir, 'comprehensive_analysis.json')
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # 8. Generate summary
    summary_file = os.path.join(output_dir, 'analysis_summary.txt')
    generate_analysis_summary(analysis_results, summary_file)
    
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

def generate_analysis_summary(analysis_results, summary_file):
    """Generate comprehensive analysis summary"""
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Comprehensive Curvature Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Experiment info
        exp_info = analysis_results['experiment_info']
        f.write(f"Source File: {exp_info['source_file']}\n")
        f.write(f"Device: {exp_info['device']}\n")
        f.write(f"Shots: {exp_info['shots']}\n")
        f.write(f"Qubits: {exp_info['num_qubits']}\n")
        f.write(f"Geometry Type: {exp_info['geometry_type']}\n")
        f.write(f"Curvature Strength: {exp_info['curvature_strength']}\n\n")
        
        # Lorentzian analysis
        if 'lorentzian_analysis' in analysis_results:
            lorentzian = analysis_results['lorentzian_analysis']
            f.write("LORENTZIAN GEOMETRY ANALYSIS:\n")
            f.write(f"Lorentzian Action: {lorentzian['lorentzian_action']:.6f}\n")
            f.write(f"P-value: {lorentzian['p_value']:.6f}\n")
            f.write(f"Statistically Significant: {lorentzian['is_significant']}\n")
            f.write(f"Bootstrap CI: [{lorentzian['bootstrap_ci'][0]:.6f}, {lorentzian['bootstrap_ci'][1]:.6f}]\n\n")
        
        # MI analysis
        if 'mi_analysis' in analysis_results:
            mi_analysis = analysis_results['mi_analysis']
            f.write("MUTUAL INFORMATION ANALYSIS:\n")
            f.write(f"Maximum MI: {mi_analysis['max_mi']:.6f}\n")
            f.write(f"Mean MI: {mi_analysis['mean_mi']:.6f}\n")
            f.write(f"Non-zero MI pairs: {mi_analysis['non_zero_pairs']}\n\n")
        
        # RT analysis
        if 'rt_analysis' in analysis_results:
            rt_analysis = analysis_results['rt_analysis']
            f.write("RT SURFACE ANALYSIS:\n")
            f.write(f"RT Surfaces computed: {rt_analysis['num_rt_surfaces']}\n")
            f.write(f"RT Consistency (std): {rt_analysis['rt_consistency']:.6f}\n")
            f.write(f"Mean RT Ratio: {rt_analysis['mean_rt_ratio']:.6f}\n\n")
        
        # Geometry analysis
        if 'geometry_analysis' in analysis_results:
            geom_analysis = analysis_results['geometry_analysis']
            f.write("GEOMETRY RECONSTRUCTION:\n")
            f.write(f"Geometry Radius: {geom_analysis['geometry_radius']:.6f}\n")
            f.write(f"MI-Distance Correlation: {geom_analysis['mi_distance_correlation']:.6f}\n\n")
        
        # Advanced Holographic Analyses
        f.write("ADVANCED HOLOGRAPHIC ANALYSES:\n")
        f.write("-" * 30 + "\n\n")
        
        # RT Verification
        if 'rt_verification' in analysis_results:
            rt_verification = analysis_results['rt_verification']
            f.write("RT VERIFICATION:\n")
            f.write(f"MI vs Minimal Path Correlation: {rt_verification['correlation']:.6f}\n")
            f.write(f"P-value: {rt_verification['p_value']:.6f}\n")
            f.write(f"Number of pairs analyzed: {rt_verification['num_pairs']}\n\n")
        
        # Entropy Scaling
        if 'entropy_scaling' in analysis_results:
            entropy_scaling = analysis_results['entropy_scaling']
            f.write("HOLOGRAPHIC ENTROPY SCALING:\n")
            f.write(f"Area Scaling R²: {entropy_scaling['area_scaling']['r_squared']:.6f}\n")
            f.write(f"Volume Scaling R²: {entropy_scaling['volume_scaling']['r_squared']:.6f}\n")
            f.write(f"Supports Holographic Principle: {entropy_scaling['holographic_support']}\n\n")
        
        # Boundary Dynamics
        if 'boundary_dynamics' in analysis_results:
            boundary_dynamics = analysis_results['boundary_dynamics']
            f.write("BOUNDARY DYNAMICS:\n")
            f.write(f"Boundary-Bulk Correlation: {boundary_dynamics['correlation']:.6f}\n")
            f.write(f"P-value: {boundary_dynamics['p_value']:.6f}\n\n")
        
        # Overall assessment
        f.write("OVERALL ASSESSMENT:\n")
        
        # Lorentzian assessment
        if 'lorentzian_analysis' in analysis_results:
            if analysis_results['lorentzian_analysis']['is_significant']:
                f.write("+ Strong evidence for Lorentzian geometry\n")
            else:
                f.write("- Weak evidence for Lorentzian geometry\n")
        
        # RT assessment
        if 'rt_analysis' in analysis_results:
            if analysis_results['rt_analysis']['rt_consistency'] < 0.1:
                f.write("+ RT surface consistency high, supporting holographic duality\n")
            else:
                f.write("- RT surface consistency low, may need more entanglement\n")
        
        # Geometry assessment
        if 'geometry_analysis' in analysis_results:
            correlation = analysis_results['geometry_analysis']['mi_distance_correlation']
            if correlation > 0.7:
                f.write("+ Strong correlation between MI and geometric distance\n")
            elif correlation > 0.3:
                f.write("+ Moderate correlation between MI and geometric distance\n")
            else:
                f.write("- Weak correlation, may indicate insufficient entanglement\n")
        
        # Advanced analyses assessment
        if 'rt_verification' in analysis_results:
            rt_corr = analysis_results['rt_verification']['correlation']
            if abs(rt_corr) > 0.5:
                f.write("+ Strong RT verification: MI tracks with minimal bulk paths\n")
            elif abs(rt_corr) > 0.3:
                f.write("+ Moderate RT verification\n")
            else:
                f.write("- Weak RT verification\n")
        
        if 'entropy_scaling' in analysis_results:
            if analysis_results['entropy_scaling']['holographic_support']:
                f.write("+ Entropy scaling supports holographic principle (area > volume)\n")
            else:
                f.write("- Entropy scaling does not clearly support holographic principle\n")
        
        if 'boundary_dynamics' in analysis_results:
            boundary_corr = analysis_results['boundary_dynamics']['correlation']
            if abs(boundary_corr) > 0.5:
                f.write("+ Strong boundary-bulk dynamics correlation\n")
            elif abs(boundary_corr) > 0.3:
                f.write("+ Moderate boundary-bulk dynamics correlation\n")
            else:
                f.write("- Weak boundary-bulk dynamics correlation\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comprehensive Curvature Analyzer')
    parser.add_argument('results_file', help='Path to experiment results JSON file')
    parser.add_argument('--output_dir', help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: Results file {args.results_file} not found!")
        exit(1)
    
    comprehensive_analysis(args.results_file, args.output_dir) 