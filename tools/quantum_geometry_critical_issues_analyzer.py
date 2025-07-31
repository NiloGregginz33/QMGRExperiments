#!/usr/bin/env python3
"""
Quantum Geometry Critical Issues Analyzer
========================================

This script analyzes quantum geometry experiments for 5 critical issues:

1. Continuum Limit: "Discrete Signatures" - Detects when system behaves too discretely
2. Classical Benchmarks: "Low Classical Agreement" - Compares against known geometries
3. Decoherence Sensitivity: Tests robustness against shot noise and backend variation
4. Causal Structure: Detects causal violations in information flow
5. Holographic Consistency: Evaluates bulk-boundary entanglement correspondence

Features:
- Systematic detection of each critical issue
- Specific recommendations with command-line flags
- Visual analysis of problems and solutions
- Statistical validation of findings
- Comprehensive reporting for peer review
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys
import os
from typing import Dict, List, Optional, Tuple, Any
import warnings
from scipy import stats, optimize, interpolate
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.manifold import MDS
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class QuantumGeometryCriticalAnalyzer:
    """Analyzer for critical issues in quantum geometry experiments."""
    
    def __init__(self, instance_dir: str):
        """Initialize analyzer with experiment instance directory."""
        self.instance_dir = Path(instance_dir)
        self.data = self._load_experiment_data()
        self.analysis_results = {}
        
    def _load_experiment_data(self) -> Dict:
        """Load all experiment data from instance directory."""
        # Load results data
        results_files = list(self.instance_dir.glob("results_*.json"))
        if not results_files:
            raise FileNotFoundError(f"No results files found in {self.instance_dir}")
        
        # Use the largest results file (most complete data)
        results_file = max(results_files, key=lambda x: x.stat().st_size)
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        # Load summary if available
        summary_file = self.instance_dir / "summary.txt"
        summary_data = None
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary_data = f.read()
        
        return {
            'results_data': results_data,
            'summary_data': summary_data,
            'instance_dir': str(self.instance_dir)
        }
    
    def analyze_continuum_limit(self) -> Dict:
        """
        Issue 1: Continuum Limit - "Discrete Signatures"
        
        Problem: System behaves too discretely — not approaching smooth manifold geometry.
        Solution: Add interpolation between MI distances using differentiable embedding.
        """
        print("🔍 Analyzing Continuum Limit Issues...")
        
        results = self.data['results_data']
        spec = results['spec']
        
        # Extract MI matrix and distances
        if 'mi_matrix' not in results:
            return {'error': 'No MI matrix found in results'}
        
        mi_matrix = np.array(results['mi_matrix'])
        num_qubits = spec['num_qubits']
        
        # Calculate distance matrix from MI
        distances = -np.log(mi_matrix + 1e-10)  # Avoid log(0)
        
        # Test for discrete signatures
        discrete_indicators = {}
        
        # 1. Check for quantization in distance values
        unique_distances = np.unique(distances)
        quantization_ratio = len(unique_distances) / (num_qubits * (num_qubits - 1) / 2)
        discrete_indicators['quantization_ratio'] = quantization_ratio
        discrete_indicators['unique_distance_count'] = len(unique_distances)
        
        # 2. Check for step-like behavior in distance distribution
        distance_hist, _ = np.histogram(distances.flatten(), bins=20)
        step_detection = np.std(distance_hist) / np.mean(distance_hist)
        discrete_indicators['step_detection'] = step_detection
        
        # 3. Test smoothness via gradient analysis
        if 'embedding_coords' in results:
            coords = np.array(results['embedding_coords'])
            if len(coords) > 0:
                # Calculate gradients between adjacent points
                gradients = []
                for i in range(len(coords)):
                    for j in range(i+1, len(coords)):
                        if distances[i, j] > 0:
                            grad = np.linalg.norm(coords[i] - coords[j]) / distances[i, j]
                            gradients.append(grad)
                
                gradient_variation = np.std(gradients) / np.mean(gradients) if gradients else 0
                discrete_indicators['gradient_variation'] = gradient_variation
        
        # 4. Test for interpolation potential
        try:
            # Try to fit a smooth function to the distance data
            x_coords = np.arange(num_qubits)
            y_coords = np.arange(num_qubits)
            X, Y = np.meshgrid(x_coords, y_coords)
            
            # Fit Gaussian Process for smoothness
            gp = GaussianProcessRegressor(
                kernel=RBF(length_scale=1.0),
                random_state=42
            )
            
            # Sample points for fitting
            sample_indices = np.random.choice(num_qubits**2, min(100, num_qubits**2), replace=False)
            X_sample = np.column_stack([X.flatten()[sample_indices], Y.flatten()[sample_indices]])
            y_sample = distances.flatten()[sample_indices]
            
            gp.fit(X_sample, y_sample)
            y_pred = gp.predict(X_sample)
            interpolation_score = r2_score(y_sample, y_pred)
            discrete_indicators['interpolation_score'] = interpolation_score
            
        except Exception as e:
            discrete_indicators['interpolation_score'] = 0
            discrete_indicators['interpolation_error'] = str(e)
        
        # Determine if continuum limit is violated
        continuum_violated = (
            quantization_ratio < 0.3 or  # Too few unique distances
            step_detection > 2.0 or      # Too step-like
            discrete_indicators.get('interpolation_score', 0) < 0.5  # Poor interpolation
        )
        
        analysis_result = {
            'continuum_violated': continuum_violated,
            'discrete_indicators': discrete_indicators,
            'recommendations': [
                "Add interpolation between MI distances using differentiable embedding (e.g. RBF kernel or MDS smoothing)",
                "Test for continuity in MI-derived distances over adjacent nodes",
                "Increase number of qubits or smooth out charge injection"
            ],
            'command_line_flags': [
                "--interpolate_geometry true",
                "--smooth_charge_injection true", 
                "--min_qubits_for_continuum 10"
            ]
        }
        
        self.analysis_results['continuum_limit'] = analysis_result
        return analysis_result
    
    def analyze_classical_benchmarks(self) -> Dict:
        """
        Issue 2: Classical Benchmarks - "Low Classical Agreement"
        
        Problem: Reconstructed geometry diverges from known classical spacetime models.
        Solution: Compare against classical curved space embeddings and report deviation metrics.
        """
        print("🔍 Analyzing Classical Benchmark Agreement...")
        
        results = self.data['results_data']
        spec = results['spec']
        
        geometry = spec.get('geometry', 'unknown')
        curvature = spec.get('curvature', 0.0)
        num_qubits = spec['num_qubits']
        
        if 'embedding_coords' not in results:
            return {'error': 'No embedding coordinates found for benchmark comparison'}
        
        coords = np.array(results['embedding_coords'])
        if len(coords) == 0:
            return {'error': 'Empty embedding coordinates'}
        
        # Generate classical benchmark geometries
        benchmark_geometries = {}
        
        # 1. Spherical geometry benchmark
        if geometry == 'spherical' or curvature > 0:
            spherical_coords = self._generate_spherical_benchmark(num_qubits, curvature)
            benchmark_geometries['spherical'] = spherical_coords
        
        # 2. Hyperbolic geometry benchmark  
        if geometry == 'hyperbolic' or curvature < 0:
            hyperbolic_coords = self._generate_hyperbolic_benchmark(num_qubits, abs(curvature))
            benchmark_geometries['hyperbolic'] = hyperbolic_coords
        
        # 3. Euclidean geometry benchmark
        euclidean_coords = self._generate_euclidean_benchmark(num_qubits)
        benchmark_geometries['euclidean'] = euclidean_coords
        
        # Calculate deviation metrics
        deviation_metrics = {}
        best_fit_geometry = None
        best_fit_score = float('inf')
        
        for geom_name, benchmark_coords in benchmark_geometries.items():
            # Calculate distances for both geometries
            exp_distances = squareform(pdist(coords))
            bench_distances = squareform(pdist(benchmark_coords))
            
            # Normalize distances for comparison
            exp_dist_norm = exp_distances / np.max(exp_distances)
            bench_dist_norm = bench_distances / np.max(bench_distances)
            
            # Calculate multiple deviation metrics
            metrics = {}
            
            # 1. Wasserstein distance
            try:
                wasserstein_dist = wasserstein_distance(
                    exp_dist_norm.flatten(), 
                    bench_dist_norm.flatten()
                )
                metrics['wasserstein_distance'] = wasserstein_dist
            except:
                metrics['wasserstein_distance'] = float('inf')
            
            # 2. KL divergence (with smoothing)
            try:
                exp_hist, _ = np.histogram(exp_dist_norm.flatten(), bins=20, density=True)
                bench_hist, _ = np.histogram(bench_dist_norm.flatten(), bins=20, density=True)
                
                # Add small epsilon to avoid log(0)
                epsilon = 1e-10
                exp_hist += epsilon
                bench_hist += epsilon
                
                kl_div = np.sum(exp_hist * np.log(exp_hist / bench_hist))
                metrics['kl_divergence'] = kl_div
            except:
                metrics['kl_divergence'] = float('inf')
            
            # 3. Euclidean distance between distance matrices
            euclidean_dist = np.linalg.norm(exp_dist_norm - bench_dist_norm)
            metrics['euclidean_distance'] = euclidean_dist
            
            # 4. Correlation coefficient
            correlation = np.corrcoef(exp_dist_norm.flatten(), bench_dist_norm.flatten())[0, 1]
            metrics['correlation'] = correlation if not np.isnan(correlation) else 0
            
            deviation_metrics[geom_name] = metrics
            
            # Track best fit
            overall_score = metrics.get('wasserstein_distance', float('inf'))
            if overall_score < best_fit_score:
                best_fit_score = overall_score
                best_fit_geometry = geom_name
        
        # Determine if classical agreement is poor
        expected_geometry = geometry if geometry in ['spherical', 'hyperbolic', 'euclidean'] else 'unknown'
        if expected_geometry in deviation_metrics:
            expected_metrics = deviation_metrics[expected_geometry]
            classical_agreement_poor = (
                expected_metrics.get('wasserstein_distance', float('inf')) > 0.5 or
                expected_metrics.get('correlation', 0) < 0.3
            )
        else:
            classical_agreement_poor = True
        
        analysis_result = {
            'classical_agreement_poor': classical_agreement_poor,
            'deviation_metrics': deviation_metrics,
            'best_fit_geometry': best_fit_geometry,
            'expected_geometry': expected_geometry,
            'recommendations': [
                "Compare reconstructed geometry against classical curved space embeddings (spherical, hyperbolic)",
                "Report deviation metrics for each classical geometry",
                "Implement geometry-specific validation tests"
            ],
            'command_line_flags': [
                "--benchmark_against_classical_geometry true",
                "--geometry_fit_metric wasserstein,kl,euclidean"
            ]
        }
        
        self.analysis_results['classical_benchmarks'] = analysis_result
        return analysis_result
    
    def analyze_decoherence_sensitivity(self) -> Dict:
        """
        Issue 3: Decoherence Sensitivity
        
        Problem: Too sensitive to shot noise or backend variation → might be a quantum effect, but need to verify robustness.
        Solution: Repeat experiment with different backend seeds and increased shots.
        """
        print("🔍 Analyzing Decoherence Sensitivity...")
        
        results = self.data['results_data']
        spec = results['spec']
        
        shots = spec.get('shots', 1000)
        device = spec.get('device', 'unknown')
        
        # Analyze shot noise sensitivity
        shot_noise_analysis = {}
        
        if 'counts_per_timestep' in results:
            counts_data = results['counts_per_timestep']
            
            # Calculate shot noise metrics
            total_counts = sum(sum(counts.values()) for counts in counts_data)
            shot_noise_analysis['total_shots'] = total_counts
            
            # Analyze count distribution variance
            count_variances = []
            for timestep_counts in counts_data:
                if timestep_counts:
                    counts_array = np.array(list(timestep_counts.values()))
                    variance = np.var(counts_array)
                    count_variances.append(variance)
            
            if count_variances:
                shot_noise_analysis['count_variance_mean'] = np.mean(count_variances)
                shot_noise_analysis['count_variance_std'] = np.std(count_variances)
                shot_noise_analysis['relative_variance'] = np.mean(count_variances) / (total_counts / len(counts_data))
        
        # Analyze MI matrix stability
        mi_stability_analysis = {}
        
        if 'mi_matrix' in results:
            mi_matrix = np.array(results['mi_matrix'])
            
            # Calculate MI matrix condition number
            try:
                condition_number = np.linalg.cond(mi_matrix)
                mi_stability_analysis['condition_number'] = condition_number
            except:
                mi_stability_analysis['condition_number'] = float('inf')
            
            # Analyze MI value distribution
            mi_values = mi_matrix.flatten()
            mi_stability_analysis['mi_mean'] = np.mean(mi_values)
            mi_stability_analysis['mi_std'] = np.std(mi_values)
            mi_stability_analysis['mi_cv'] = np.std(mi_values) / np.mean(mi_values) if np.mean(mi_values) > 0 else 0
        
        # Check for hardware-specific issues
        hardware_sensitivity = {}
        if 'ibm' in device.lower():
            hardware_sensitivity['is_hardware'] = True
            hardware_sensitivity['device_name'] = device
            
            # Check if shots are sufficient for hardware
            if shots < 4000:
                hardware_sensitivity['insufficient_shots'] = True
                hardware_sensitivity['recommended_shots'] = 8192
            else:
                hardware_sensitivity['insufficient_shots'] = False
        else:
            hardware_sensitivity['is_hardware'] = False
        
        # Determine if decoherence sensitivity is high
        decoherence_sensitive = (
            shot_noise_analysis.get('relative_variance', 0) > 0.1 or
            mi_stability_analysis.get('condition_number', float('inf')) > 1e6 or
            mi_stability_analysis.get('mi_cv', 0) > 0.5 or
            hardware_sensitivity.get('insufficient_shots', False)
        )
        
        analysis_result = {
            'decoherence_sensitive': decoherence_sensitive,
            'shot_noise_analysis': shot_noise_analysis,
            'mi_stability_analysis': mi_stability_analysis,
            'hardware_sensitivity': hardware_sensitivity,
            'recommendations': [
                "Repeat the experiment with different backend seeds and increased shots",
                "Report standard deviation across MI and entropy metrics",
                "Test on multiple backends to verify consistency"
            ],
            'command_line_flags': [
                "--verify_noise_robustness true",
                "--shots 8192",
                "--run_on_multiple_backends true"
            ]
        }
        
        self.analysis_results['decoherence_sensitivity'] = analysis_result
        return analysis_result
    
    def analyze_causal_structure(self) -> Dict:
        """
        Issue 4: Causal Structure - 278 Causal Violations
        
        Problem: Information flow violates expected causality (e.g., future nodes influencing past).
        Solution: Trace MI flow and identify feedback loops violating lightcone constraints.
        """
        print("🔍 Analyzing Causal Structure Violations...")
        
        results = self.data['results_data']
        spec = results['spec']
        
        num_qubits = spec['num_qubits']
        timesteps = spec.get('timesteps', 1)
        
        causal_violations = {}
        
        # Analyze MI matrix for causal violations
        if 'mi_matrix' in results:
            mi_matrix = np.array(results['mi_matrix'])
            
            # Create directed graph from MI matrix
            G = nx.DiGraph()
            
            # Add nodes
            for i in range(num_qubits):
                G.add_node(i)
            
            # Add edges based on MI values (threshold-based)
            mi_threshold = np.percentile(mi_matrix.flatten(), 75)  # Top 25% connections
            
            for i in range(num_qubits):
                for j in range(num_qubits):
                    if i != j and mi_matrix[i, j] > mi_threshold:
                        G.add_edge(i, j, weight=mi_matrix[i, j])
            
            # Detect cycles (causal violations)
            try:
                cycles = list(nx.simple_cycles(G))
                causal_violations['cycle_count'] = len(cycles)
                causal_violations['total_cycle_length'] = sum(len(cycle) for cycle in cycles)
                
                # Analyze cycle characteristics
                if cycles:
                    cycle_lengths = [len(cycle) for cycle in cycles]
                    causal_violations['avg_cycle_length'] = np.mean(cycle_lengths)
                    causal_violations['max_cycle_length'] = max(cycle_lengths)
                    causal_violations['cycle_length_std'] = np.std(cycle_lengths)
                else:
                    causal_violations['avg_cycle_length'] = 0
                    causal_violations['max_cycle_length'] = 0
                    causal_violations['cycle_length_std'] = 0
                    
            except Exception as e:
                causal_violations['cycle_detection_error'] = str(e)
                causal_violations['cycle_count'] = 0
        
        # Analyze temporal evolution for causal violations
        temporal_violations = {}
        
        if 'counts_per_timestep' in results and timesteps > 1:
            counts_data = results['counts_per_timestep']
            
            # Calculate state evolution
            state_evolution = []
            for timestep_counts in counts_data:
                if timestep_counts:
                    # Calculate state vector from counts
                    total_counts = sum(timestep_counts.values())
                    state_vector = {}
                    for bitstring, count in timestep_counts.items():
                        state_vector[bitstring] = count / total_counts
                    state_evolution.append(state_vector)
            
            # Check for backward influence (future affecting past)
            if len(state_evolution) > 1:
                backward_influence_score = 0
                for t in range(1, len(state_evolution)):
                    current_state = state_evolution[t]
                    previous_state = state_evolution[t-1]
                    
                    # Calculate state difference
                    common_keys = set(current_state.keys()) & set(previous_state.keys())
                    if common_keys:
                        differences = [abs(current_state[k] - previous_state[k]) for k in common_keys]
                        backward_influence_score += np.mean(differences)
                
                temporal_violations['backward_influence_score'] = backward_influence_score
                temporal_violations['state_evolution_steps'] = len(state_evolution)
        
        # Check for lightcone violations
        lightcone_violations = {}
        
        if 'embedding_coords' in results and 'mi_matrix' in results:
            coords = np.array(results['embedding_coords'])
            mi_matrix = np.array(results['mi_matrix'])
            if len(coords) > 0:
                # Calculate spatial distances
                spatial_distances = squareform(pdist(coords))
                
                # Check for faster-than-light information transfer
                # (assuming unit speed of light and unit time steps)
                lightcone_violations_count = 0
                mi_threshold = np.percentile(mi_matrix.flatten(), 75)  # Top 25% connections
                
                for i in range(num_qubits):
                    for j in range(i+1, num_qubits):
                        if i < len(spatial_distances) and j < len(spatial_distances) and i < mi_matrix.shape[0] and j < mi_matrix.shape[1]:
                            spatial_dist = spatial_distances[i, j]
                            mi_strength = mi_matrix[i, j]
                            
                            # If MI is high but spatial distance is large, potential lightcone violation
                            if mi_strength > mi_threshold and spatial_dist > 2.0:  # Arbitrary threshold
                                lightcone_violations_count += 1
                
                lightcone_violations['violation_count'] = lightcone_violations_count
                lightcone_violations['violation_ratio'] = lightcone_violations_count / (num_qubits * (num_qubits - 1) / 2)
        
        # Determine if causal violations are significant
        significant_causal_violations = (
            causal_violations.get('cycle_count', 0) > 10 or
            temporal_violations.get('backward_influence_score', 0) > 0.1 or
            lightcone_violations.get('violation_ratio', 0) > 0.1
        )
        
        analysis_result = {
            'significant_causal_violations': significant_causal_violations,
            'causal_violations': causal_violations,
            'temporal_violations': temporal_violations,
            'lightcone_violations': lightcone_violations,
            'recommendations': [
                "Trace MI flow and identify feedback loops violating lightcone constraints",
                "Suppress or tag non-causal paths",
                "Implement causal ordering in quantum circuit design"
            ],
            'command_line_flags': [
                "--detect_and_flag_causal_loops true",
                "--restrict_information_flow_direction forward",
                "--filter_noncausal_edges true"
            ]
        }
        
        self.analysis_results['causal_structure'] = analysis_result
        return analysis_result
    
    def analyze_holographic_consistency(self) -> Dict:
        """
        Issue 5: Holographic Consistency Score: 0.565
        
        Problem: Bulk-boundary entanglement correspondence is weak.
        Solution: Refine Ryu-Takayanagi estimates using refined mutual information surfaces.
        """
        print("🔍 Analyzing Holographic Consistency...")
        
        results = self.data['results_data']
        spec = results['spec']
        
        num_qubits = spec['num_qubits']
        
        holographic_analysis = {}
        
        # Calculate holographic consistency score
        if 'mi_matrix' in results:
            mi_matrix = np.array(results['mi_matrix'])
            
            # 1. Calculate boundary entropy estimates
            boundary_entropies = []
            for i in range(num_qubits):
                # Single qubit entropy (von Neumann entropy)
                if i < len(mi_matrix):
                    # Approximate single qubit entropy from diagonal elements
                    single_qubit_mi = mi_matrix[i, i] if i < mi_matrix.shape[0] else 0
                    boundary_entropies.append(single_qubit_mi)
            
            # 2. Calculate bulk geometry areas
            bulk_areas = []
            if 'embedding_coords' in results:
                coords = np.array(results['embedding_coords'])
                if len(coords) > 0:
                    # Calculate areas of triangles formed by qubits
                    for i in range(num_qubits):
                        for j in range(i+1, num_qubits):
                            for k in range(j+1, num_qubits):
                                if i < len(coords) and j < len(coords) and k < len(coords):
                                    # Calculate triangle area
                                    v1 = coords[j] - coords[i]
                                    v2 = coords[k] - coords[i]
                                    area = 0.5 * np.linalg.norm(np.cross(v1, v2))
                                    bulk_areas.append(area)
            
            # 3. Calculate Ryu-Takayanagi correspondence
            if boundary_entropies and bulk_areas:
                # Normalize both quantities
                norm_boundary = np.array(boundary_entropies) / np.max(boundary_entropies)
                norm_bulk = np.array(bulk_areas) / np.max(bulk_areas)
                
                # Calculate correlation between boundary entropy and bulk area
                min_len = min(len(norm_boundary), len(norm_bulk))
                if min_len > 1:
                    correlation = np.corrcoef(norm_boundary[:min_len], norm_bulk[:min_len])[0, 1]
                    holographic_analysis['ryu_takayanagi_correlation'] = correlation if not np.isnan(correlation) else 0
                else:
                    holographic_analysis['ryu_takayanagi_correlation'] = 0
            else:
                holographic_analysis['ryu_takayanagi_correlation'] = 0
            
            # 4. Calculate mutual information surface areas
            mi_surface_analysis = {}
            
            # Calculate MI-weighted surface areas
            mi_surfaces = []
            for i in range(num_qubits):
                for j in range(i+1, num_qubits):
                    if i < mi_matrix.shape[0] and j < mi_matrix.shape[1]:
                        mi_value = mi_matrix[i, j]
                        if mi_value > 0:
                            # Approximate surface area as MI-weighted distance
                            if 'embedding_coords' in results and len(coords) > max(i, j):
                                distance = np.linalg.norm(coords[i] - coords[j])
                                surface_area = mi_value * distance
                                mi_surfaces.append(surface_area)
            
            if mi_surfaces:
                mi_surface_analysis['mean_surface_area'] = np.mean(mi_surfaces)
                mi_surface_analysis['surface_area_std'] = np.std(mi_surfaces)
                mi_surface_analysis['total_surface_area'] = np.sum(mi_surfaces)
            
            holographic_analysis['mi_surface_analysis'] = mi_surface_analysis
            
            # 5. Calculate subsystem entropy vs MI comparison
            subsystem_analysis = {}
            
            # Calculate subsystem entropies for different partition sizes
            subsystem_entropies = []
            for size in range(1, min(5, num_qubits // 2 + 1)):
                # Approximate subsystem entropy using MI
                subsystem_mi = []
                for i in range(num_qubits - size + 1):
                    subsystem = list(range(i, i + size))
                    # Calculate average MI within subsystem
                    if len(subsystem) > 1:
                        subsystem_mi_values = []
                        for j in subsystem:
                            for k in subsystem:
                                if j != k and j < mi_matrix.shape[0] and k < mi_matrix.shape[1]:
                                    subsystem_mi_values.append(mi_matrix[j, k])
                        if subsystem_mi_values:
                            subsystem_mi.append(np.mean(subsystem_mi_values))
                
                if subsystem_mi:
                    subsystem_entropies.append(np.mean(subsystem_mi))
            
            if subsystem_entropies:
                subsystem_analysis['subsystem_entropy_evolution'] = subsystem_entropies
                subsystem_analysis['entropy_scaling'] = np.polyfit(range(1, len(subsystem_entropies) + 1), subsystem_entropies, 1)[0]
            
            holographic_analysis['subsystem_analysis'] = subsystem_analysis
        
        # Calculate overall holographic consistency score
        consistency_score = 0.0
        if holographic_analysis.get('ryu_takayanagi_correlation', 0) > 0:
            consistency_score += 0.4 * holographic_analysis['ryu_takayanagi_correlation']
        
        if holographic_analysis.get('mi_surface_analysis', {}).get('total_surface_area', 0) > 0:
            consistency_score += 0.3
        
        if holographic_analysis.get('subsystem_analysis', {}).get('entropy_scaling', 0) > 0:
            consistency_score += 0.3
        
        holographic_analysis['consistency_score'] = consistency_score
        
        # Determine if holographic consistency is weak
        holographic_consistency_weak = consistency_score < 0.7
        
        analysis_result = {
            'holographic_consistency_weak': holographic_consistency_weak,
            'holographic_analysis': holographic_analysis,
            'consistency_score': consistency_score,
            'recommendations': [
                "Refine Ryu-Takayanagi estimates using refined mutual information surfaces",
                "Compare to exact subsystem entropy instead of approximated MI-only methods",
                "Implement boundary entropy embedding in geometry reconstruction"
            ],
            'command_line_flags': [
                "--use_ryu_takayanagi_test true",
                "--compare_MI_vs_subsystem_entropy true",
                "--embed_boundary_entropy_in_geometry true"
            ]
        }
        
        self.analysis_results['holographic_consistency'] = analysis_result
        return analysis_result
    
    def _generate_spherical_benchmark(self, num_qubits: int, curvature: float) -> np.ndarray:
        """Generate spherical geometry benchmark coordinates."""
        coords = []
        for i in range(num_qubits):
            # Generate points on a sphere
            phi = 2 * np.pi * i / num_qubits
            theta = np.arccos(2 * (i % (num_qubits // 2)) / (num_qubits // 2) - 1)
            
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            
            # Scale by curvature
            scale = 1.0 / np.sqrt(curvature) if curvature > 0 else 1.0
            coords.append([x * scale, y * scale, z * scale])
        
        return np.array(coords)
    
    def _generate_hyperbolic_benchmark(self, num_qubits: int, curvature: float) -> np.ndarray:
        """Generate hyperbolic geometry benchmark coordinates."""
        coords = []
        for i in range(num_qubits):
            # Generate points in hyperbolic space (Poincaré disk model)
            r = np.sqrt(i / num_qubits)  # Radial coordinate
            phi = 2 * np.pi * i / num_qubits  # Angular coordinate
            
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            
            # Scale by curvature
            scale = 1.0 / np.sqrt(curvature) if curvature > 0 else 1.0
            coords.append([x * scale, y * scale])
        
        return np.array(coords)
    
    def _generate_euclidean_benchmark(self, num_qubits: int) -> np.ndarray:
        """Generate Euclidean geometry benchmark coordinates."""
        coords = []
        for i in range(num_qubits):
            # Generate points in a square grid
            x = (i % int(np.sqrt(num_qubits))) / np.sqrt(num_qubits)
            y = (i // int(np.sqrt(num_qubits))) / np.sqrt(num_qubits)
            coords.append([x, y])
        
        return np.array(coords)
    
    def generate_analysis_report(self) -> str:
        """Generate comprehensive analysis report."""
        report = []
        report.append("=" * 80)
        report.append("QUANTUM GEOMETRY CRITICAL ISSUES ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Instance Directory: {self.instance_dir}")
        report.append(f"Analysis Date: {pd.Timestamp.now()}")
        report.append("")
        
        # Summary of critical issues
        critical_issues = []
        if self.analysis_results.get('continuum_limit', {}).get('continuum_violated', False):
            critical_issues.append("🚧 1. Continuum Limit: Discrete signatures detected")
        
        if self.analysis_results.get('classical_benchmarks', {}).get('classical_agreement_poor', False):
            critical_issues.append("🚧 2. Classical Benchmarks: Low agreement with expected geometry")
        
        if self.analysis_results.get('decoherence_sensitivity', {}).get('decoherence_sensitive', False):
            critical_issues.append("🚧 3. Decoherence Sensitivity: High sensitivity to noise detected")
        
        if self.analysis_results.get('causal_structure', {}).get('significant_causal_violations', False):
            critical_issues.append("🚧 4. Causal Structure: Significant causal violations found")
        
        if self.analysis_results.get('holographic_consistency', {}).get('holographic_consistency_weak', False):
            critical_issues.append("🚧 5. Holographic Consistency: Weak bulk-boundary correspondence")
        
        if critical_issues:
            report.append("🚨 CRITICAL ISSUES DETECTED:")
            for issue in critical_issues:
                report.append(f"  {issue}")
        else:
            report.append("✅ No critical issues detected - experiment appears robust")
        
        report.append("")
        
        # Detailed analysis for each issue
        for issue_name, analysis in self.analysis_results.items():
            report.append(f"📊 {issue_name.upper().replace('_', ' ')} ANALYSIS:")
            report.append("-" * 50)
            
            if 'error' in analysis:
                report.append(f"Error: {analysis['error']}")
            else:
                # Add key metrics
                for key, value in analysis.items():
                    if key not in ['recommendations', 'command_line_flags']:
                        if isinstance(value, dict):
                            report.append(f"{key}:")
                            for subkey, subvalue in value.items():
                                report.append(f"  {subkey}: {subvalue}")
                        else:
                            report.append(f"{key}: {value}")
                
                # Add recommendations
                if 'recommendations' in analysis:
                    report.append("")
                    report.append("💡 RECOMMENDATIONS:")
                    for i, rec in enumerate(analysis['recommendations'], 1):
                        report.append(f"  {i}. {rec}")
                
                # Add command line flags
                if 'command_line_flags' in analysis:
                    report.append("")
                    report.append("🔧 COMMAND LINE FLAGS:")
                    for flag in analysis['command_line_flags']:
                        report.append(f"  {flag}")
            
            report.append("")
        
        return "\n".join(report)
    
    def create_visualization_plots(self, output_dir: str = None):
        """Create visualization plots for critical issues."""
        if output_dir is None:
            output_dir = self.instance_dir
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Quantum Geometry Critical Issues Analysis', fontsize=16, fontweight='bold')
        
        # 1. Continuum Limit Analysis
        if 'continuum_limit' in self.analysis_results:
            ax = axes[0, 0]
            analysis = self.analysis_results['continuum_limit']
            
            if 'discrete_indicators' in analysis:
                indicators = analysis['discrete_indicators']
                labels = list(indicators.keys())
                values = list(indicators.values())
                
                # Filter out non-numeric values
                numeric_data = [(l, v) for l, v in zip(labels, values) if isinstance(v, (int, float))]
                if numeric_data:
                    labels, values = zip(*numeric_data)
                    ax.bar(labels, values, color='red' if analysis.get('continuum_violated', False) else 'green')
                    ax.set_title('Continuum Limit Indicators')
                    ax.tick_params(axis='x', rotation=45)
        
        # 2. Classical Benchmark Comparison
        if 'classical_benchmarks' in self.analysis_results:
            ax = axes[0, 1]
            analysis = self.analysis_results['classical_benchmarks']
            
            if 'deviation_metrics' in analysis:
                geometries = list(analysis['deviation_metrics'].keys())
                wasserstein_distances = [analysis['deviation_metrics'][g].get('wasserstein_distance', 0) for g in geometries]
                
                ax.bar(geometries, wasserstein_distances, 
                      color='red' if analysis.get('classical_agreement_poor', False) else 'green')
                ax.set_title('Classical Geometry Deviations')
                ax.set_ylabel('Wasserstein Distance')
        
        # 3. Decoherence Sensitivity
        if 'decoherence_sensitivity' in self.analysis_results:
            ax = axes[0, 2]
            analysis = self.analysis_results['decoherence_sensitivity']
            
            if 'mi_stability_analysis' in analysis:
                mi_analysis = analysis['mi_stability_analysis']
                metrics = ['mi_mean', 'mi_std', 'mi_cv']
                values = [mi_analysis.get(m, 0) for m in metrics]
                
                ax.bar(metrics, values, color='red' if analysis.get('decoherence_sensitive', False) else 'green')
                ax.set_title('MI Stability Metrics')
                ax.tick_params(axis='x', rotation=45)
        
        # 4. Causal Structure
        if 'causal_structure' in self.analysis_results:
            ax = axes[1, 0]
            analysis = self.analysis_results['causal_structure']
            
            if 'causal_violations' in analysis:
                violations = analysis['causal_violations']
                metrics = ['cycle_count', 'avg_cycle_length', 'max_cycle_length']
                values = [violations.get(m, 0) for m in metrics]
                
                ax.bar(metrics, values, color='red' if analysis.get('significant_causal_violations', False) else 'green')
                ax.set_title('Causal Violations')
                ax.tick_params(axis='x', rotation=45)
        
        # 5. Holographic Consistency
        if 'holographic_consistency' in self.analysis_results:
            ax = axes[1, 1]
            analysis = self.analysis_results['holographic_consistency']
            
            consistency_score = analysis.get('consistency_score', 0)
            ax.bar(['Holographic\nConsistency'], [consistency_score], 
                  color='red' if analysis.get('holographic_consistency_weak', False) else 'green')
            ax.set_title('Holographic Consistency Score')
            ax.set_ylim(0, 1)
        
        # 6. Overall Summary
        ax = axes[1, 2]
        issue_names = ['Continuum\nLimit', 'Classical\nBenchmarks', 'Decoherence\nSensitivity', 
                      'Causal\nStructure', 'Holographic\nConsistency']
        
        issue_status = []
        for issue in ['continuum_limit', 'classical_benchmarks', 'decoherence_sensitivity', 
                     'causal_structure', 'holographic_consistency']:
            if issue in self.analysis_results:
                analysis = self.analysis_results[issue]
                # Check if issue is problematic
                problematic = False
                if issue == 'continuum_limit':
                    problematic = analysis.get('continuum_violated', False)
                elif issue == 'classical_benchmarks':
                    problematic = analysis.get('classical_agreement_poor', False)
                elif issue == 'decoherence_sensitivity':
                    problematic = analysis.get('decoherence_sensitive', False)
                elif issue == 'causal_structure':
                    problematic = analysis.get('significant_causal_violations', False)
                elif issue == 'holographic_consistency':
                    problematic = analysis.get('holographic_consistency_weak', False)
                
                issue_status.append(1 if problematic else 0)
            else:
                issue_status.append(0)
        
        colors = ['red' if status else 'green' for status in issue_status]
        ax.bar(issue_names, issue_status, color=colors)
        ax.set_title('Critical Issues Summary')
        ax.set_ylabel('Issue Detected (1=Yes, 0=No)')
        ax.set_ylim(0, 1.2)
        
        # Add text labels
        for i, (name, status) in enumerate(zip(issue_names, issue_status)):
            ax.text(i, status + 0.05, 'ISSUE' if status else 'OK', 
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path / 'critical_issues_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved to: {output_path / 'critical_issues_analysis.png'}")
    
    def save_analysis_results(self, output_dir: str = None):
        """Save analysis results to files."""
        if output_dir is None:
            output_dir = self.instance_dir
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save detailed analysis as JSON
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = output_path / f"critical_issues_analysis_{timestamp}.json"
        
        with open(analysis_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        # Save report as text
        report_file = output_path / f"critical_issues_report_{timestamp}.txt"
        report = self.generate_analysis_report()
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Analysis results saved to:")
        print(f"  JSON: {analysis_file}")
        print(f"  Report: {report_file}")
        
        return str(analysis_file), str(report_file)

def main():
    """Main function to run the critical issues analyzer."""
    if len(sys.argv) < 2:
        print("Usage: python quantum_geometry_critical_issues_analyzer.py <instance_directory>")
        print("Example: python quantum_geometry_critical_issues_analyzer.py experiment_logs/custom_curvature_experiment/instance_20250730_190246")
        sys.exit(1)
    
    instance_dir = sys.argv[1]
    
    try:
        # Initialize analyzer
        analyzer = QuantumGeometryCriticalAnalyzer(instance_dir)
        
        # Run all analyses
        print("🚀 Starting Quantum Geometry Critical Issues Analysis...")
        print(f"📁 Analyzing instance: {instance_dir}")
        print()
        
        analyzer.analyze_continuum_limit()
        analyzer.analyze_classical_benchmarks()
        analyzer.analyze_decoherence_sensitivity()
        analyzer.analyze_causal_structure()
        analyzer.analyze_holographic_consistency()
        
        # Generate and display report
        report = analyzer.generate_analysis_report()
        print(report)
        
        # Create visualizations
        analyzer.create_visualization_plots()
        
        # Save results
        analyzer.save_analysis_results()
        
        print("\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 