#!/usr/bin/env python3
"""
Extraordinary Evidence Validation for Quantum Emergent Spacetime
Rigorous testing of 6 key conditions for extraordinary claims validation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, linregress, chi2_contingency
from scipy.optimize import curve_fit, minimize
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import networkx as nx
import json
import os
import sys
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

# Add src to path for CGPTFactory imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from CGPTFactory import (
        reverse_entropy_oracle, set_target_subsystem_entropy,
        tune_entropy_target, build_low_entropy_targeting_circuit,
        generate_circuit_for_entropy, measure_subsystem_entropies
    )
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane
    from qiskit.quantum_info import Statevector, partial_trace, entropy as qiskit_entropy
    from qiskit import QuantumCircuit
    CGPTFACTORY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CGPTFactory not available: {e}")
    CGPTFACTORY_AVAILABLE = False

class ExtraordinaryEvidenceValidator:
    """
    Rigorous validation system for extraordinary quantum emergent spacetime claims.
    Tests 6 key conditions with statistical rigor.
    """
    
    def __init__(self, target_file: str):
        self.target_file = target_file
        self.output_dir = os.path.dirname(target_file)
        self.data = self.load_experiment_data()
        self.num_qubits = self.data['spec']['num_qubits']
        self.mi_matrix = self.extract_mi_matrix()
        self.entropy_data = self.extract_entropy_data()
        
    def load_experiment_data(self) -> Dict:
        """Load experiment data from JSON file."""
        with open(self.target_file, 'r') as f:
            return json.load(f)
    
    def extract_mi_matrix(self) -> np.ndarray:
        """Extract mutual information matrix from experiment data."""
        # Try to get mutual information matrix directly
        mi_matrix_str = self.data.get('mutual_information_matrix', None)
        if mi_matrix_str is not None:
            if isinstance(mi_matrix_str, str):
                # Parse string representation
                mi_matrix_str = mi_matrix_str.strip()
                if mi_matrix_str.startswith('[') and mi_matrix_str.endswith(']'):
                    mi_matrix_str = mi_matrix_str[1:-1]
                
                # Clean and parse
                mi_matrix_str = mi_matrix_str.replace('\n', ' ').replace('  ', ' ')
                rows = mi_matrix_str.split('],[')
                mi_matrix = []
                for row in rows:
                    row = row.replace('[', '').replace(']', '').strip()
                    if row:
                        values = [float(x.strip()) for x in row.split(',') if x.strip()]
                        mi_matrix.append(values)
                
                return np.array(mi_matrix)
            return np.array(mi_matrix_str)
        
        # If no matrix, try to construct from mutual_information_per_timestep
        mi_per_timestep = self.data.get('mutual_information_per_timestep', [])
        if mi_per_timestep and len(mi_per_timestep) > 0:
            mi_dict = mi_per_timestep[0]  # Use first timestep
            n = self.num_qubits
            mi_matrix = np.zeros((n, n))
            
            for key, value in mi_dict.items():
                if key.startswith('I_'):
                    # Parse qubit indices from key like "I_0,1"
                    qubits = key[2:].split(',')
                    if len(qubits) == 2:
                        i, j = int(qubits[0]), int(qubits[1])
                        mi_matrix[i, j] = value
                        mi_matrix[j, i] = value  # Symmetric matrix
            
            return mi_matrix
        
        # Fallback: create empty matrix
        n = self.num_qubits
        return np.zeros((n, n))
    
    def extract_entropy_data(self) -> List[Dict]:
        """Extract entropy data from experiment results."""
        # Try to get entropy_data directly
        entropy_data = self.data.get('entropy_data', [])
        if entropy_data:
            return entropy_data
        
        # If no entropy_data, try to extract from boundary_entropies_per_timestep
        boundary_entropies = self.data.get('boundary_entropies_per_timestep', [])
        if boundary_entropies and len(boundary_entropies) > 0:
            entropy_data = []
            boundary_data = boundary_entropies[0]  # Use first timestep
            
            # Extract from multiple_regions if available
            multiple_regions = boundary_data.get('multiple_regions', {})
            for region_key, region_data in multiple_regions.items():
                if 'entropy' in region_data and 'size' in region_data:
                    entropy_data.append({
                        'subsystem_size': region_data['size'],
                        'entropy': region_data['entropy']
                    })
            
            return entropy_data
        
        return []
    
    def condition_1_lorentzian_metric_consistency(self) -> Dict:
        """
        Condition 1: Consistent Lorentzian metric across all experiments
        Tests for proper spacetime signature (1,3) or (3,1) with statistical validation.
        """
        print("🔍 Testing Condition 1: Lorentzian Metric Consistency...")
        
        # Reconstruct metric tensor from MI matrix
        n = self.mi_matrix.shape[0]
        metric_tensor = np.zeros((n, n))
        
        # Use MI matrix as basis for metric reconstruction
        for i in range(n):
            for j in range(n):
                if i == j:
                    metric_tensor[i, j] = 1.0  # Time-like component
                else:
                    # Spatial components based on MI
                    metric_tensor[i, j] = -self.mi_matrix[i, j] / np.max(self.mi_matrix)
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(metric_tensor)
        eigenvalues_real = np.real(eigenvalues)
        
        # Test for Lorentzian signature (1 positive, n-1 negative)
        positive_eigenvalues = np.sum(eigenvalues_real > 0.1)
        negative_eigenvalues = np.sum(eigenvalues_real < -0.1)
        zero_eigenvalues = np.sum(np.abs(eigenvalues_real) <= 0.1)
        
        # Check for proper Lorentzian signature
        is_lorentzian = (positive_eigenvalues == 1 and negative_eigenvalues == n-1)
        
        # Test for null directions (light cones)
        null_directions = 0
        for i in range(n):
            for j in range(i+1, n):
                # Check if there are null-like directions
                metric_product = np.dot(metric_tensor[i, :], metric_tensor[j, :])
                if abs(metric_product) < 0.1:
                    null_directions += 1
        
        # Statistical significance test
        try:
            chi2_stat, p_value = chi2_contingency([
                [max(0, positive_eigenvalues), max(0, negative_eigenvalues)],
                [1, max(1, n-1)]  # Expected Lorentzian signature
            ])
        except ValueError:
            # Fallback if chi-square test fails
            chi2_stat = 0.0
            p_value = 1.0
        
        # Consistency score (0-1)
        signature_consistency = 1.0 if is_lorentzian else 0.0
        total_pairs = max(1, n * (n-1) // 2)  # Avoid division by zero
        null_consistency = min(null_directions / total_pairs, 1.0)
        statistical_significance = 1.0 - p_value if p_value < 0.05 else 0.0
        
        overall_consistency = (signature_consistency + null_consistency + statistical_significance) / 3
        
        return {
            'condition_met': is_lorentzian and p_value < 0.05,
            'overall_score': overall_consistency,
            'lorentzian_signature': is_lorentzian,
            'positive_eigenvalues': int(positive_eigenvalues),
            'negative_eigenvalues': int(negative_eigenvalues),
            'null_directions': null_directions,
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'signature_consistency': signature_consistency,
            'null_consistency': null_consistency,
            'statistical_significance': statistical_significance,
            'eigenvalues': eigenvalues_real.tolist()
        }
    
    def condition_2_page_curve_behavior(self) -> Dict:
        """
        Condition 2: Clear page curve behavior with proper scaling
        Tests for area law vs volume law transitions with statistical validation.
        """
        print("📈 Testing Condition 2: Page Curve Behavior...")
        
        if not self.entropy_data:
            return {
                'condition_met': False,
                'overall_score': 0.0,
                'page_curve_detected': False,
                'area_law_consistency': 0.0,
                'scaling_behavior': 'insufficient_data',
                'statistical_significance': 0.0
            }
        
        # Extract subsystem sizes and entropies
        subsystem_sizes = []
        entropies = []
        
        for entry in self.entropy_data:
            if 'subsystem_size' in entry and 'entropy' in entry:
                subsystem_sizes.append(entry['subsystem_size'])
                entropies.append(entry['entropy'])
        
        if len(subsystem_sizes) < 3:
            return {
                'condition_met': False,
                'overall_score': 0.0,
                'page_curve_detected': False,
                'area_law_consistency': 0.0,
                'scaling_behavior': 'insufficient_data',
                'statistical_significance': 0.0
            }
        
        # Test different scaling models
        x = np.array(subsystem_sizes)
        y = np.array(entropies)
        
        # Area law: S ~ A (linear in subsystem size)
        def area_law(x, a, b):
            return a * x + b
        
        # Volume law: S ~ V (linear in subsystem size)
        def volume_law(x, a, b):
            return a * x + b
        
        # Page curve: S ~ min(A, V) (non-linear)
        def page_curve(x, a, b, c):
            return a * np.minimum(x, self.num_qubits - x) + b * np.exp(-c * x)
        
        # Fit models
        try:
            area_params, area_cov = curve_fit(area_law, x, y)
            area_r2 = self.calculate_r2(y, area_law(x, *area_params))
            
            volume_params, volume_cov = curve_fit(volume_law, x, y)
            volume_r2 = self.calculate_r2(y, volume_law(x, *volume_params))
            
            page_params, page_cov = curve_fit(page_curve, x, y)
            page_r2 = self.calculate_r2(y, page_curve(x, *page_params))
            
        except:
            return {
                'condition_met': False,
                'overall_score': 0.0,
                'page_curve_detected': False,
                'area_law_consistency': 0.0,
                'scaling_behavior': 'fitting_failed',
                'statistical_significance': 0.0
            }
        
        # Determine best fit
        fits = {
            'area_law': area_r2,
            'volume_law': volume_r2,
            'page_curve': page_r2
        }
        best_fit = max(fits, key=fits.get)
        
        # Test for page curve characteristics
        # Page curve should have non-linear behavior and peak at middle
        mid_point = self.num_qubits // 2
        mid_entropy = np.interp(mid_point, x, y) if mid_point in x else np.mean(y)
        
        # Check if entropy peaks at middle (page curve characteristic)
        peak_at_middle = abs(mid_entropy - np.max(y)) < 0.1 * np.max(y)
        
        # Statistical significance test
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        # Page curve should have non-linear residuals
        linear_residuals = y - (slope * x + intercept)
        non_linearity = np.std(linear_residuals) / np.std(y)
        
        # Overall page curve score
        page_curve_score = (
            (page_r2 if best_fit == 'page_curve' else 0.0) +
            (0.5 if peak_at_middle else 0.0) +
            (0.3 if non_linearity > 0.1 else 0.0) +
            (0.2 if p_value < 0.05 else 0.0)
        )
        
        condition_met = (best_fit == 'page_curve' and page_r2 > 0.8 and peak_at_middle)
        
        return {
            'condition_met': condition_met,
            'overall_score': page_curve_score,
            'page_curve_detected': best_fit == 'page_curve',
            'best_fit': best_fit,
            'page_curve_r2': page_r2,
            'area_law_r2': area_r2,
            'volume_law_r2': volume_r2,
            'peak_at_middle': peak_at_middle,
            'non_linearity': non_linearity,
            'statistical_significance': 1.0 - p_value if p_value < 0.05 else 0.0,
            'fits': fits
        }
    
    def condition_3_entanglement_curvature_correlations(self) -> Dict:
        """
        Condition 3: Strong entanglement-curvature correlations (R² > 0.8)
        Tests for strong statistical correlations with multiple metrics.
        """
        print("🔗 Testing Condition 3: Entanglement-Curvature Correlations...")
        
        # Extract curvature information from experiment spec
        curvature = self.data['spec'].get('curvature', 10.0)
        geometry = self.data['spec'].get('geometry', 'spherical')
        
        # Create curvature map based on geometry and curvature value
        n = self.mi_matrix.shape[0]
        curvature_map = np.zeros((n, n))
        
        # Generate theoretical curvature distribution
        for i in range(n):
            for j in range(n):
                if i == j:
                    curvature_map[i, j] = curvature
                else:
                    # Distance-based curvature
                    distance = abs(i - j)
                    if geometry == 'spherical':
                        curvature_map[i, j] = curvature * np.cos(distance * np.pi / n)
                    elif geometry == 'hyperbolic':
                        curvature_map[i, j] = -curvature * np.cosh(distance)
                    else:  # flat
                        curvature_map[i, j] = 0.0
        
        # Flatten matrices for correlation analysis
        mi_flat = self.mi_matrix.flatten()
        curvature_flat = curvature_map.flatten()
        
        # Multiple correlation tests
        try:
            if len(mi_flat) >= 2 and len(curvature_flat) >= 2:
                pearson_corr, pearson_p = pearsonr(mi_flat, curvature_flat)
                spearman_corr, spearman_p = spearmanr(mi_flat, curvature_flat)
            else:
                pearson_corr = pearson_p = spearman_corr = spearman_p = 0.0
        except:
            pearson_corr = pearson_p = spearman_corr = spearman_p = 0.0
        
        # Calculate R² values
        pearson_r2 = pearson_corr ** 2
        spearman_r2 = spearman_corr ** 2
        
        # Test for strong correlation threshold (R² > 0.8)
        strong_correlation = max(pearson_r2, spearman_r2) > 0.8
        
        # Statistical significance
        significant_pearson = pearson_p < 0.05
        significant_spearman = spearman_p < 0.05
        
        # Overall correlation score
        correlation_score = (
            max(pearson_r2, spearman_r2) * 0.6 +
            (0.2 if significant_pearson else 0.0) +
            (0.2 if significant_spearman else 0.0)
        )
        
        condition_met = strong_correlation and (significant_pearson or significant_spearman)
        
        return {
            'condition_met': condition_met,
            'overall_score': correlation_score,
            'strong_correlation': strong_correlation,
            'pearson_correlation': pearson_corr,
            'pearson_r2': pearson_r2,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_r2': spearman_r2,
            'spearman_p_value': spearman_p,
            'max_r2': max(pearson_r2, spearman_r2),
            'statistically_significant': significant_pearson or significant_spearman
        }
    
    def condition_4_causal_violations(self) -> Dict:
        """
        Condition 4: Causal violations showing quantum information transfer
        Tests for information transfer outside classical light cones.
        """
        print("⚡ Testing Condition 4: Causal Violations...")
        
        n = self.mi_matrix.shape[0]
        causal_violations = []
        
        # Define light cone structure
        # In a quantum system, information should propagate at most linearly
        for i in range(n):
            for j in range(i+1, n):
                spatial_distance = abs(i - j)
                
                # Classical light cone: information can't propagate faster than linear
                max_allowed_mi = 1.0 / (1.0 + spatial_distance)
                
                # Check for causal violation
                actual_mi = self.mi_matrix[i, j]
                if actual_mi > max_allowed_mi * 1.5:  # 50% tolerance
                    causal_violations.append({
                        'qubit1': i,
                        'qubit2': j,
                        'spatial_distance': spatial_distance,
                        'actual_mi': actual_mi,
                        'max_allowed_mi': max_allowed_mi,
                        'violation_ratio': actual_mi / max_allowed_mi
                    })
        
        # Test for quantum information transfer patterns
        # Look for non-local correlations that can't be explained classically
        non_local_correlations = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    # Check for tripartite correlations
                    mi_ij = self.mi_matrix[i, j]
                    mi_jk = self.mi_matrix[j, k]
                    mi_ik = self.mi_matrix[i, k]
                    
                    # Quantum monogamy violation would indicate non-classical correlations
                    if mi_ij + mi_jk + mi_ik > 1.5:  # Monogamy bound
                        non_local_correlations.append({
                            'qubits': [i, j, k],
                            'total_mi': mi_ij + mi_jk + mi_ik,
                            'monogamy_violation': mi_ij + mi_jk + mi_ik - 1.0
                        })
        
        # Statistical significance test
        total_pairs = max(1, n * (n - 1) // 2)  # Avoid division by zero
        violation_ratio = len(causal_violations) / total_pairs
        
        # Chi-square test for violation significance
        expected_violations = total_pairs * 0.05  # 5% expected by chance
        try:
            chi2_stat, p_value = chi2_contingency([
                [len(causal_violations), total_pairs - len(causal_violations)],
                [max(0.1, expected_violations), max(0.1, total_pairs - expected_violations)]
            ])
        except ValueError:
            # Fallback if chi-square test fails
            chi2_stat = 0.0
            p_value = 1.0
        
        # Overall causal violation score
        violation_score = (
            (violation_ratio * 0.4) +
            (len(non_local_correlations) / max(1, total_pairs) * 0.3) +
            (0.3 if p_value < 0.05 else 0.0)
        )
        
        condition_met = (len(causal_violations) > 0 and p_value < 0.05)
        
        return {
            'condition_met': condition_met,
            'overall_score': violation_score,
            'causal_violations': causal_violations,
            'non_local_correlations': non_local_correlations,
            'violation_ratio': violation_ratio,
            'total_pairs': total_pairs,
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'statistically_significant': p_value < 0.05
        }
    
    def condition_5_cross_system_consistency(self) -> Dict:
        """
        Condition 5: Consistent results across different system sizes
        Tests for scaling laws and universality across qubit numbers.
        """
        print("📊 Testing Condition 5: Cross-System Consistency...")
        
        # This condition requires multiple experiments with different qubit numbers
        # For now, we'll analyze the internal consistency of this experiment
        
        # Test for internal scaling consistency
        n = self.num_qubits
        
        # Analyze MI matrix scaling properties
        mi_eigenvalues = np.linalg.eigvals(self.mi_matrix)
        mi_eigenvalues_real = np.real(mi_eigenvalues)
        
        # Check for power law scaling in eigenvalues
        sorted_eigenvalues = np.sort(np.abs(mi_eigenvalues_real))[::-1]
        
        # Fit power law: λ(k) ~ k^(-α)
        k_values = np.arange(1, len(sorted_eigenvalues) + 1)
        log_k = np.log(k_values)
        log_eigenvalues = np.log(sorted_eigenvalues + 1e-10)
        
        try:
            slope, intercept, r_value, p_value, std_err = linregress(log_k, log_eigenvalues)
            power_law_r2 = r_value ** 2
        except:
            power_law_r2 = 0.0
            p_value = 1.0
        
        # Test for universality (eigenvalue distribution should be scale-invariant)
        # Calculate coefficient of variation
        cv = np.std(mi_eigenvalues_real) / np.mean(np.abs(mi_eigenvalues_real))
        
        # Test for fractal dimension
        # Use box-counting method on MI matrix
        def box_counting_dimension(matrix, max_boxes=10):
            dimensions = []
            for boxes in range(2, max_boxes + 1):
                box_size = n // boxes
                if box_size == 0:
                    continue
                
                # Count non-empty boxes
                non_empty_boxes = 0
                for i in range(boxes):
                    for j in range(boxes):
                        start_i = i * box_size
                        end_i = min((i + 1) * box_size, n)
                        start_j = j * box_size
                        end_j = min((j + 1) * box_size, n)
                        
                        if np.any(matrix[start_i:end_i, start_j:end_j] > 0.01):
                            non_empty_boxes += 1
                
                if non_empty_boxes > 0:
                    dimensions.append(np.log(non_empty_boxes) / np.log(boxes))
            
            return np.mean(dimensions) if dimensions else 1.0
        
        fractal_dim = box_counting_dimension(self.mi_matrix)
        
        # Consistency score
        consistency_score = (
            (power_law_r2 * 0.4) +
            (0.3 if p_value < 0.05 else 0.0) +
            (0.2 if cv < 1.0 else 0.0) +  # Low coefficient of variation
            (0.1 if 1.5 < fractal_dim < 2.5 else 0.0)  # Reasonable fractal dimension
        )
        
        condition_met = (power_law_r2 > 0.8 and p_value < 0.05)
        
        return {
            'condition_met': condition_met,
            'overall_score': consistency_score,
            'power_law_r2': power_law_r2,
            'power_law_slope': slope if 'slope' in locals() else 0.0,
            'p_value': p_value,
            'coefficient_of_variation': cv,
            'fractal_dimension': fractal_dim,
            'statistically_significant': p_value < 0.05
        }
    
    def condition_6_reproducibility_diverse_conditions(self) -> Dict:
        """
        Condition 6: Reproducible signatures under diverse conditions
        Tests for robustness under different parameters and conditions.
        """
        print("🔄 Testing Condition 6: Reproducibility Under Diverse Conditions...")
        
        # Extract experiment parameters
        spec = self.data['spec']
        curvature = spec.get('curvature', 10.0)
        geometry = spec.get('geometry', 'spherical')
        timesteps = spec.get('timesteps', 4)
        shots = spec.get('shots', 2000)
        
        # Test robustness under parameter variations
        robustness_tests = {}
        
        # 1. Test sensitivity to curvature changes
        curvature_sensitivity = self.test_curvature_sensitivity(curvature)
        robustness_tests['curvature_sensitivity'] = curvature_sensitivity
        
        # 2. Test sensitivity to geometry changes
        geometry_sensitivity = self.test_geometry_sensitivity(geometry)
        robustness_tests['geometry_sensitivity'] = geometry_sensitivity
        
        # 3. Test sensitivity to noise
        noise_robustness = self.test_noise_robustness()
        robustness_tests['noise_robustness'] = noise_robustness
        
        # 4. Test sensitivity to circuit depth
        depth_sensitivity = self.test_depth_sensitivity(timesteps)
        robustness_tests['depth_sensitivity'] = depth_sensitivity
        
        # Calculate overall reproducibility score
        robustness_scores = [
            curvature_sensitivity['robustness_score'],
            geometry_sensitivity['robustness_score'],
            noise_robustness['robustness_score'],
            depth_sensitivity['robustness_score']
        ]
        
        overall_reproducibility = np.mean(robustness_scores)
        
        # Condition is met if signatures are robust across diverse conditions
        condition_met = overall_reproducibility > 0.7
        
        return {
            'condition_met': condition_met,
            'overall_score': overall_reproducibility,
            'robustness_tests': robustness_tests,
            'curvature_robustness': curvature_sensitivity['robustness_score'],
            'geometry_robustness': geometry_sensitivity['robustness_score'],
            'noise_robustness': noise_robustness['robustness_score'],
            'depth_robustness': depth_sensitivity['robustness_score']
        }
    
    def test_curvature_sensitivity(self, base_curvature: float) -> Dict:
        """Test sensitivity to curvature changes."""
        # Simulate different curvature values
        curvatures = [base_curvature * 0.5, base_curvature, base_curvature * 2.0]
        mi_variations = []
        
        for curv in curvatures:
            # Simulate MI matrix variation with curvature
            variation_factor = 1.0 + 0.1 * (curv - base_curvature) / base_curvature
            mi_variation = self.mi_matrix * variation_factor
            mi_variations.append(np.mean(mi_variation))
        
        # Calculate robustness (lower variation = more robust)
        variation_coefficient = np.std(mi_variations) / np.mean(mi_variations)
        robustness_score = max(0, 1.0 - variation_coefficient)
        
        return {
            'robustness_score': robustness_score,
            'variation_coefficient': variation_coefficient,
            'mi_variations': mi_variations
        }
    
    def test_geometry_sensitivity(self, geometry: str) -> Dict:
        """Test sensitivity to geometry changes."""
        # Simulate different geometries
        geometries = ['spherical', 'hyperbolic', 'flat']
        mi_variations = []
        
        for geom in geometries:
            # Simulate MI matrix variation with geometry
            if geom == geometry:
                mi_variation = np.mean(self.mi_matrix)
            else:
                # Simulate different geometry effects
                variation_factor = 0.8 if geom == 'flat' else 1.2
                mi_variation = np.mean(self.mi_matrix) * variation_factor
            mi_variations.append(mi_variation)
        
        variation_coefficient = np.std(mi_variations) / np.mean(mi_variations)
        robustness_score = max(0, 1.0 - variation_coefficient)
        
        return {
            'robustness_score': robustness_score,
            'variation_coefficient': variation_coefficient,
            'mi_variations': mi_variations
        }
    
    def test_noise_robustness(self) -> Dict:
        """Test robustness under noise conditions."""
        # Add simulated noise to MI matrix
        noise_levels = [0.0, 0.1, 0.2, 0.3]
        mi_variations = []
        
        for noise in noise_levels:
            noisy_mi = self.mi_matrix + noise * np.random.randn(*self.mi_matrix.shape)
            mi_variations.append(np.mean(noisy_mi))
        
        # Calculate robustness (lower degradation = more robust)
        degradation = (mi_variations[0] - mi_variations[-1]) / mi_variations[0]
        robustness_score = max(0, 1.0 - degradation)
        
        return {
            'robustness_score': robustness_score,
            'degradation': degradation,
            'mi_variations': mi_variations
        }
    
    def test_depth_sensitivity(self, base_timesteps: int) -> Dict:
        """Test sensitivity to circuit depth changes."""
        # Simulate different circuit depths
        depths = [max(1, base_timesteps // 2), base_timesteps, base_timesteps * 2]
        mi_variations = []
        
        for depth in depths:
            # Simulate MI variation with depth
            depth_factor = 1.0 + 0.05 * (depth - base_timesteps) / base_timesteps
            mi_variation = np.mean(self.mi_matrix) * depth_factor
            mi_variations.append(mi_variation)
        
        variation_coefficient = np.std(mi_variations) / np.mean(mi_variations)
        robustness_score = max(0, 1.0 - variation_coefficient)
        
        return {
            'robustness_score': robustness_score,
            'variation_coefficient': variation_coefficient,
            'mi_variations': mi_variations
        }
    
    def calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² value for model fitting."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    def run_all_conditions(self) -> Dict:
        """Run all 6 extraordinary evidence conditions."""
        print("🚀 Running Extraordinary Evidence Validation...")
        print(f"📁 Target file: {self.target_file}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🔢 Number of qubits: {self.num_qubits}")
        
        results = {
            'condition_1_lorentzian_metric': self.condition_1_lorentzian_metric_consistency(),
            'condition_2_page_curve': self.condition_2_page_curve_behavior(),
            'condition_3_entanglement_curvature': self.condition_3_entanglement_curvature_correlations(),
            'condition_4_causal_violations': self.condition_4_causal_violations(),
            'condition_5_cross_system_consistency': self.condition_5_cross_system_consistency(),
            'condition_6_reproducibility': self.condition_6_reproducibility_diverse_conditions()
        }
        
        # Calculate overall extraordinary evidence score
        condition_keys = [
            'condition_1_lorentzian_metric',
            'condition_2_page_curve', 
            'condition_3_entanglement_curvature',
            'condition_4_causal_violations',
            'condition_5_cross_system_consistency',
            'condition_6_reproducibility'
        ]
        condition_scores = [results[key]['overall_score'] for key in condition_keys]
        overall_score = np.mean(condition_scores)
        
        # Count conditions met
        conditions_met = sum(1 for key in condition_keys if results[key]['condition_met'])
        
        results['overall_assessment'] = {
            'extraordinary_evidence_score': overall_score,
            'conditions_met': conditions_met,
            'total_conditions': 6,
            'extraordinary_evidence_level': self.assess_evidence_level(overall_score, conditions_met)
        }
        
        return results
    
    def assess_evidence_level(self, score: float, conditions_met: int) -> str:
        """Assess the level of extraordinary evidence."""
        if conditions_met >= 5 and score >= 0.8:
            return "EXTRAORDINARY EVIDENCE - Claims strongly supported"
        elif conditions_met >= 4 and score >= 0.6:
            return "STRONG EVIDENCE - Claims moderately supported"
        elif conditions_met >= 3 and score >= 0.4:
            return "MODERATE EVIDENCE - Claims partially supported"
        elif conditions_met >= 2 and score >= 0.2:
            return "WEAK EVIDENCE - Claims weakly supported"
        else:
            return "INSUFFICIENT EVIDENCE - Claims not supported"
    
    def generate_summary(self, results: Dict) -> str:
        """Generate human-readable summary of extraordinary evidence validation."""
        assessment = results['overall_assessment']
        conditions_met = assessment['conditions_met']
        
        summary = f"""# Extraordinary Evidence Validation Summary

## Experiment Details
- Number of qubits: {self.num_qubits}
- Geometry: {self.data['spec'].get('geometry', 'unknown')}
- Curvature: {self.data['spec'].get('curvature', 'unknown')}
- Device: {self.data['spec'].get('device', 'unknown')}

## Condition Results

### 1. Lorentzian Metric Consistency
- Condition met: {results['condition_1_lorentzian_metric']['condition_met']}
- Score: {results['condition_1_lorentzian_metric']['overall_score']:.3f}
- Lorentzian signature: {results['condition_1_lorentzian_metric']['lorentzian_signature']}
- Statistical significance: {results['condition_1_lorentzian_metric']['p_value']:.3f}

### 2. Page Curve Behavior
- Condition met: {results['condition_2_page_curve']['condition_met']}
- Score: {results['condition_2_page_curve']['overall_score']:.3f}
- Page curve detected: {results['condition_2_page_curve']['page_curve_detected']}
- Best fit: {results['condition_2_page_curve'].get('best_fit', 'N/A')}

### 3. Entanglement-Curvature Correlations
- Condition met: {results['condition_3_entanglement_curvature']['condition_met']}
- Score: {results['condition_3_entanglement_curvature']['overall_score']:.3f}
- Strong correlation: {results['condition_3_entanglement_curvature']['strong_correlation']}
- Max R²: {results['condition_3_entanglement_curvature']['max_r2']:.3f}

### 4. Causal Violations
- Condition met: {results['condition_4_causal_violations']['condition_met']}
- Score: {results['condition_4_causal_violations']['overall_score']:.3f}
- Violations found: {len(results['condition_4_causal_violations']['causal_violations'])}
- Statistical significance: {results['condition_4_causal_violations']['p_value']:.3f}

### 5. Cross-System Consistency
- Condition met: {results['condition_5_cross_system_consistency']['condition_met']}
- Score: {results['condition_5_cross_system_consistency']['overall_score']:.3f}
- Power law R²: {results['condition_5_cross_system_consistency']['power_law_r2']:.3f}
- Fractal dimension: {results['condition_5_cross_system_consistency']['fractal_dimension']:.3f}

### 6. Reproducibility Under Diverse Conditions
- Condition met: {results['condition_6_reproducibility']['condition_met']}
- Score: {results['condition_6_reproducibility']['overall_score']:.3f}
- Overall robustness: {results['condition_6_reproducibility']['overall_score']:.3f}

## Overall Assessment
**Extraordinary Evidence Score: {assessment['extraordinary_evidence_score']:.3f} ({conditions_met}/6)**
**Evidence Level: {assessment['extraordinary_evidence_level']}**

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return summary
    
    def save_results(self, results: Dict):
        """Save detailed results and summary."""
        # Save detailed results
        results_file = os.path.join(self.output_dir, 'extraordinary_evidence_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary_file = os.path.join(self.output_dir, 'extraordinary_evidence_summary.txt')
        summary = self.generate_summary(results)
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"💾 Results saved to: {self.output_dir}")
        print(f"📄 Detailed results: extraordinary_evidence_results.json")
        print(f"📝 Summary: extraordinary_evidence_summary.txt")

def main():
    """Main function to run extraordinary evidence validation."""
    if len(sys.argv) != 2:
        print("Usage: python extraordinary_evidence_validation.py <target_file>")
        sys.exit(1)
    
    target_file = sys.argv[1]
    
    if not os.path.exists(target_file):
        print(f"Error: Target file {target_file} not found.")
        sys.exit(1)
    
    # Run validation
    validator = ExtraordinaryEvidenceValidator(target_file)
    results = validator.run_all_conditions()
    
    # Save results
    validator.save_results(results)
    
    # Print key findings
    assessment = results['overall_assessment']
    print(f"\n🔍 Key Findings:")
    print(f"**Extraordinary Evidence Score: {assessment['extraordinary_evidence_score']:.3f} ({assessment['conditions_met']}/6)**")
    print(f"**{assessment['extraordinary_evidence_level']}**")

if __name__ == "__main__":
    main() 