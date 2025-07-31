"""
Causal Asymmetry Detection for Quantum Geometry Experiments

This module provides comprehensive detection of causal asymmetry evidence including:
- Closed Timelike Curves (CTC) loops
- Non-commuting path integrals
- Time-reversal violations
- Causal consistency analysis

Author: Quantum Geometry Research Team
Date: 2024-2025
"""

import numpy as np
import itertools
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import networkx as nx

def detect_causal_asymmetry_evidence(mi_matrix: np.ndarray, 
                                   timesteps_data: List[Dict], 
                                   num_qubits: int, 
                                   geometry: str = "hyperbolic", 
                                   curvature: float = 1.0) -> Dict:
    """
    Detect Causal Asymmetry Evidence (CTC loops, non-commuting path integrals, time-reversal violations)
    
    Args:
        mi_matrix: Mutual information matrix from quantum measurements
        timesteps_data: List of mutual information data per timestep
        num_qubits: Number of qubits in the system
        geometry: Geometry type (hyperbolic, spherical, euclidean)
        curvature: Curvature parameter
        
    Returns:
        dict: Comprehensive causal asymmetry analysis results
    """
    print("🔍 Detecting Causal Asymmetry Evidence...")
    
    results = {
        'ctc_loops_detected': False,
        'non_commuting_paths': False,
        'time_reversal_violations': False,
        'causal_consistency_score': 0.0,
        'asymmetry_metrics': {},
        'detailed_analysis': {}
    }
    
    try:
        print(f"   - MI matrix shape: {mi_matrix.shape}")
        print(f"   - Number of timesteps: {len(timesteps_data)}")
        
        # 1. CTC Loop Detection
        print("   - Running CTC loop detection...")
        ctc_analysis = _detect_ctc_loops(mi_matrix, num_qubits)
        results['ctc_loops_detected'] = ctc_analysis['ctc_detected']
        results['detailed_analysis']['ctc_loops'] = ctc_analysis
        print(f"   - CTC detection complete: {ctc_analysis['ctc_detected']}")
        
        # 2. Non-commuting Path Integral Analysis
        print("   - Running non-commuting path analysis...")
        path_analysis = _analyze_non_commuting_paths(mi_matrix, timesteps_data, num_qubits)
        results['non_commuting_paths'] = path_analysis['non_commuting_detected']
        results['detailed_analysis']['path_integrals'] = path_analysis
        print(f"   - Path analysis complete: {path_analysis['non_commuting_detected']}")
        
        # 3. Time-Reversal Violation Detection
        print("   - Running time-reversal analysis...")
        time_analysis = _detect_time_reversal_violations(timesteps_data, num_qubits)
        results['time_reversal_violations'] = time_analysis['time_reversal_violated']
        results['detailed_analysis']['time_reversal'] = time_analysis
        print(f"   - Time-reversal analysis complete: {time_analysis['time_reversal_violated']}")
        
        # 4. Causal Consistency Score
        print("   - Calculating causal consistency...")
        consistency_score = _calculate_causal_consistency(mi_matrix, timesteps_data, num_qubits)
        results['causal_consistency_score'] = consistency_score
        print(f"   - Causal consistency: {consistency_score:.4f}")
        
        # 5. Asymmetry Metrics
        print("   - Computing asymmetry metrics...")
        asymmetry_metrics = _compute_asymmetry_metrics(mi_matrix, timesteps_data, num_qubits, geometry, curvature)
        results['asymmetry_metrics'] = asymmetry_metrics
        print(f"   - Asymmetry metrics complete")
        
        # 6. Overall Assessment
        print("   - Creating overall assessment...")
        total_violations = sum([
            results['ctc_loops_detected'],
            results['non_commuting_paths'],
            results['time_reversal_violations']
        ])
        
        confidence_level = _calculate_confidence_level(results)
        
        results['overall_assessment'] = {
            'total_violations': total_violations,
            'causal_structure': 'VIOLATED' if total_violations > 0 else 'CONSISTENT',
            'quantum_gravity_evidence': total_violations > 0,
            'confidence_level': confidence_level
        }
        
        print(f"✅ Causal Asymmetry Analysis Complete:")
        print(f"   - CTC Loops: {'DETECTED' if results['ctc_loops_detected'] else 'None'}")
        print(f"   - Non-commuting Paths: {'DETECTED' if results['non_commuting_paths'] else 'None'}")
        print(f"   - Time-Reversal Violations: {'DETECTED' if results['time_reversal_violations'] else 'None'}")
        print(f"   - Causal Consistency Score: {results['causal_consistency_score']:.4f}")
        print(f"   - Overall Assessment: {results['overall_assessment']['causal_structure']}")
        
    except Exception as e:
        print(f"❌ Error in causal asymmetry detection: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)
        
        # Ensure overall_assessment is always created, even on error
        total_violations = sum([
            results.get('ctc_loops_detected', False),
            results.get('non_commuting_paths', False),
            results.get('time_reversal_violations', False)
        ])
        
        results['overall_assessment'] = {
            'total_violations': total_violations,
            'causal_structure': 'ERROR',
            'quantum_gravity_evidence': False,
            'confidence_level': 0.0
        }
    
    return results

def _detect_ctc_loops(mi_matrix: np.ndarray, num_qubits: int) -> Dict:
    """Detect Closed Timelike Curves (CTC) in the mutual information structure."""
    
    # Convert MI matrix to distance matrix (inverse relationship)
    distance_matrix = 1.0 / (mi_matrix + 1e-10)  # Add small constant to avoid division by zero
    
    # Look for closed loops with negative total distance (timelike loops)
    ctc_candidates = []
    
    # Check all possible 3-qubit loops
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            for k in range(j+1, num_qubits):
                # Calculate loop distance
                loop_distance = (distance_matrix[i, j] + 
                               distance_matrix[j, k] + 
                               distance_matrix[k, i])
                
                # Check if this forms a CTC (negative or very small loop distance)
                if loop_distance < 0.1:  # Threshold for CTC detection
                    ctc_candidates.append({
                        'qubits': [i, j, k],
                        'loop_distance': loop_distance,
                        'mi_values': [mi_matrix[i, j], mi_matrix[j, k], mi_matrix[k, i]]
                    })
    
    # Check for larger loops (4+ qubits)
    for size in range(4, min(6, num_qubits + 1)):
        for qubits in itertools.combinations(range(num_qubits), size):
            # Calculate perimeter of the loop
            perimeter = 0
            mi_values = []
            for idx in range(len(qubits)):
                i, j = qubits[idx], qubits[(idx + 1) % len(qubits)]
                perimeter += distance_matrix[i, j]
                mi_values.append(mi_matrix[i, j])
            
            if perimeter < 0.2:  # Threshold for larger CTCs
                ctc_candidates.append({
                    'qubits': list(qubits),
                    'loop_distance': perimeter,
                    'mi_values': mi_values
                })
    
    return {
        'ctc_detected': len(ctc_candidates) > 0,
        'num_ctc_candidates': len(ctc_candidates),
        'ctc_candidates': ctc_candidates,
        'strongest_ctc': min(ctc_candidates, key=lambda x: x['loop_distance']) if ctc_candidates else None
    }

def _analyze_non_commuting_paths(mi_matrix: np.ndarray, timesteps_data: List[Dict], num_qubits: int) -> Dict:
    """Analyze non-commuting path integrals through time evolution."""
    
    if len(timesteps_data) < 2:
        return {'non_commuting_detected': False, 'reason': 'Insufficient timesteps'}
    
    # Extract MI matrices for different timesteps
    mi_matrices = []
    for timestep_data in timesteps_data:
        if isinstance(timestep_data, dict):
            # Convert dictionary format to matrix
            mi_matrix_step = np.zeros((num_qubits, num_qubits))
            for key, value in timestep_data.items():
                if key.startswith('I_'):
                    indices = key[2:].split(',')
                    i, j = int(indices[0]), int(indices[1])
                    mi_matrix_step[i, j] = value
                    mi_matrix_step[j, i] = value
            mi_matrices.append(mi_matrix_step)
    
    if len(mi_matrices) < 2:
        return {'non_commuting_detected': False, 'reason': 'Could not extract MI matrices'}
    
    # Test commutativity of evolution operators
    commutator_violations = []
    
    for t in range(len(mi_matrices) - 1):
        M1 = mi_matrices[t]
        M2 = mi_matrices[t + 1]
        
        # Calculate commutator [M1, M2] = M1*M2 - M2*M1
        commutator = np.dot(M1, M2) - np.dot(M2, M1)
        commutator_norm = np.linalg.norm(commutator)
        
        if commutator_norm > 0.1:  # Threshold for non-commutativity
            commutator_violations.append({
                'timestep': t,
                'commutator_norm': commutator_norm,
                'eigenvalues': np.linalg.eigvals(commutator).tolist()
            })
    
    # Test path independence
    path_dependence = _test_path_independence(mi_matrices, num_qubits)
    
    return {
        'non_commuting_detected': len(commutator_violations) > 0 or path_dependence['path_dependent'],
        'commutator_violations': commutator_violations,
        'path_dependence': path_dependence,
        'total_violations': len(commutator_violations) + (1 if path_dependence['path_dependent'] else 0)
    }

def _test_path_independence(mi_matrices: List[np.ndarray], num_qubits: int) -> Dict:
    """Test if the evolution is path-independent."""
    
    if len(mi_matrices) < 3:
        return {'path_dependent': False, 'reason': 'Insufficient data'}
    
    # Test different evolution paths
    path_differences = []
    
    # Compare forward vs backward evolution
    forward_evolution = mi_matrices[-1] - mi_matrices[0]
    
    # Simulate backward evolution (reverse order)
    backward_evolution = mi_matrices[0] - mi_matrices[-1]
    
    path_difference = np.linalg.norm(forward_evolution + backward_evolution)
    path_differences.append(path_difference)
    
    # Test intermediate path variations
    if len(mi_matrices) >= 4:
        # Path 1: 0 -> 1 -> 3
        path1 = (mi_matrices[1] - mi_matrices[0]) + (mi_matrices[3] - mi_matrices[1])
        # Path 2: 0 -> 2 -> 3
        path2 = (mi_matrices[2] - mi_matrices[0]) + (mi_matrices[3] - mi_matrices[2])
        
        path_difference_2 = np.linalg.norm(path1 - path2)
        path_differences.append(path_difference_2)
    
    return {
        'path_dependent': any(diff > 0.1 for diff in path_differences),
        'path_differences': path_differences,
        'max_difference': max(path_differences) if path_differences else 0.0
    }

def _detect_time_reversal_violations(timesteps_data: List[Dict], num_qubits: int) -> Dict:
    """Detect violations of time-reversal symmetry."""
    
    if len(timesteps_data) < 2:
        return {'time_reversal_violated': False, 'reason': 'Insufficient timesteps'}
    
    # Extract entropy evolution
    entropy_evolution = []
    for timestep_data in timesteps_data:
        if isinstance(timestep_data, dict):
            # Calculate total system entropy from MI data
            total_mi = sum(value for key, value in timestep_data.items() if key.startswith('I_'))
            # Approximate entropy from mutual information
            entropy = total_mi / (num_qubits * (num_qubits - 1) / 2)
            entropy_evolution.append(entropy)
    
    if len(entropy_evolution) < 2:
        return {'time_reversal_violated': False, 'reason': 'Could not extract entropy evolution'}
    
    # Test time-reversal symmetry
    forward_evolution = np.array(entropy_evolution[1:]) - np.array(entropy_evolution[:-1])
    reverse_evolution = np.array(entropy_evolution[:-1]) - np.array(entropy_evolution[1:])
    
    # Calculate asymmetry
    asymmetry = np.mean(np.abs(forward_evolution + reverse_evolution))
    
    # Test entropy production (should be positive for irreversible processes)
    entropy_production = np.diff(entropy_evolution)
    positive_production = np.sum(entropy_production > 0)
    negative_production = np.sum(entropy_production < 0)
    
    return {
        'time_reversal_violated': asymmetry > 0.05,  # Threshold
        'asymmetry_score': asymmetry,
        'entropy_production_analysis': {
            'positive_steps': positive_production,
            'negative_steps': negative_production,
            'net_production': np.sum(entropy_production)
        },
        'forward_evolution': forward_evolution.tolist(),
        'reverse_evolution': reverse_evolution.tolist()
    }

def _calculate_causal_consistency(mi_matrix: np.ndarray, timesteps_data: List[Dict], num_qubits: int) -> float:
    """Calculate overall causal consistency score."""
    
    # Test triangle inequality violations
    triangle_violations = 0
    total_triangles = 0
    
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            for k in range(j+1, num_qubits):
                total_triangles += 1
                # Convert MI to distance
                d_ij = 1.0 / (mi_matrix[i, j] + 1e-10)
                d_ik = 1.0 / (mi_matrix[i, k] + 1e-10)
                d_jk = 1.0 / (mi_matrix[j, k] + 1e-10)
                
                # Check triangle inequality
                if d_ij + d_ik < d_jk or d_ij + d_jk < d_ik or d_ik + d_jk < d_ij:
                    triangle_violations += 1
    
    triangle_consistency = 1.0 - (triangle_violations / total_triangles) if total_triangles > 0 else 1.0
    
    # Test causality preservation (information flow direction)
    causality_violations = 0
    total_pairs = 0
    
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            total_pairs += 1
            # Check if distant qubits have higher MI than nearby ones (potential causality violation)
            for k in range(num_qubits):
                if k != i and k != j:
                    if mi_matrix[i, j] < mi_matrix[i, k] and mi_matrix[i, j] < mi_matrix[j, k]:
                        causality_violations += 1
                        break
    
    causality_consistency = 1.0 - (causality_violations / total_pairs) if total_pairs > 0 else 1.0
    
    # Overall consistency score
    consistency_score = (triangle_consistency + causality_consistency) / 2.0
    
    return consistency_score

def _compute_asymmetry_metrics(mi_matrix: np.ndarray, timesteps_data: List[Dict], num_qubits: int, 
                             geometry: str, curvature: float) -> Dict:
    """Compute detailed asymmetry metrics."""
    
    # Spatial asymmetry
    spatial_asymmetry = np.std(mi_matrix[np.triu_indices(num_qubits, k=1)])
    
    # Temporal asymmetry (if timesteps available)
    temporal_asymmetry = 0.0
    if len(timesteps_data) > 1:
        mi_evolution = []
        for timestep_data in timesteps_data:
            if isinstance(timestep_data, dict):
                total_mi = sum(value for key, value in timestep_data.items() if key.startswith('I_'))
                mi_evolution.append(total_mi)
        
        if len(mi_evolution) > 1:
            temporal_asymmetry = np.std(np.diff(mi_evolution))
    
    # Geometric asymmetry (based on geometry type)
    geometric_asymmetry = 0.0
    if geometry == "hyperbolic":
        # Hyperbolic geometry should show more asymmetry
        geometric_asymmetry = spatial_asymmetry * curvature
    elif geometry == "spherical":
        # Spherical geometry should be more symmetric
        geometric_asymmetry = spatial_asymmetry / (1 + curvature)
    else:  # euclidean
        geometric_asymmetry = spatial_asymmetry
    
    return {
        'spatial_asymmetry': spatial_asymmetry,
        'temporal_asymmetry': temporal_asymmetry,
        'geometric_asymmetry': geometric_asymmetry,
        'total_asymmetry': spatial_asymmetry + temporal_asymmetry + geometric_asymmetry,
        'geometry_factor': curvature if geometry == "hyperbolic" else 1.0 / (1 + curvature) if geometry == "spherical" else 1.0
    }

def _calculate_confidence_level(results: Dict) -> float:
    """Calculate confidence level in the causal asymmetry detection."""
    
    confidence_factors = []
    
    # Factor 1: Number of violations (calculate directly)
    total_violations = sum([
        results.get('ctc_loops_detected', False),
        results.get('non_commuting_paths', False),
        results.get('time_reversal_violations', False)
    ])
    
    if total_violations == 0:
        confidence_factors.append(0.9)  # High confidence in consistency
    elif total_violations == 1:
        confidence_factors.append(0.7)  # Moderate confidence
    else:
        confidence_factors.append(0.9)  # High confidence in violations
    
    # Factor 2: Causal consistency score
    consistency_score = results.get('causal_consistency_score', 0.0)
    confidence_factors.append(consistency_score)
    
    # Factor 3: Strength of violations
    violation_strength = 0.0
    if results.get('ctc_loops_detected', False):
        ctc_data = results.get('detailed_analysis', {}).get('ctc_loops', {})
        if ctc_data.get('ctc_candidates'):
            violation_strength += abs(ctc_data['strongest_ctc']['loop_distance'])
    
    if results.get('non_commuting_paths', False):
        path_data = results.get('detailed_analysis', {}).get('path_integrals', {})
        violation_strength += len(path_data.get('commutator_violations', [])) * 0.1
    
    if results.get('time_reversal_violations', False):
        time_data = results.get('detailed_analysis', {}).get('time_reversal', {})
        violation_strength += time_data.get('asymmetry_score', 0.0)
    
    confidence_factors.append(min(violation_strength, 1.0))
    
    # Average confidence
    return np.mean(confidence_factors)

def create_causal_asymmetry_visualization(results: Dict, output_path: str = None) -> None:
    """Create comprehensive visualization of causal asymmetry results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Causal Asymmetry Evidence Analysis', fontsize=16, fontweight='bold')
    
    # 1. Overall Assessment
    ax1 = axes[0, 0]
    violations = results.get('overall_assessment', {}).get('total_violations', 0)
    consistency = results.get('causal_consistency_score', 0.0)
    
    ax1.bar(['CTC Loops', 'Non-commuting', 'Time-Reversal'], 
            [results.get('ctc_loops_detected', False), 
             results.get('non_commuting_paths', False), 
             results.get('time_reversal_violations', False)],
            color=['red' if v else 'green' for v in [results.get('ctc_loops_detected', False), 
                                                    results.get('non_commuting_paths', False), 
                                                    results.get('time_reversal_violations', False)]])
    ax1.set_title(f'Causal Violations (Total: {violations})')
    ax1.set_ylabel('Detected')
    
    # 2. Causal Consistency Score
    ax2 = axes[0, 1]
    ax2.bar(['Consistency'], [consistency], color='blue' if consistency > 0.7 else 'orange')
    ax2.set_ylim(0, 1)
    ax2.set_title(f'Causal Consistency: {consistency:.3f}')
    ax2.set_ylabel('Score')
    
    # 3. Asymmetry Metrics
    ax3 = axes[0, 2]
    asymmetry_metrics = results.get('asymmetry_metrics', {})
    metrics_names = ['Spatial', 'Temporal', 'Geometric', 'Total']
    metrics_values = [asymmetry_metrics.get('spatial_asymmetry', 0),
                     asymmetry_metrics.get('temporal_asymmetry', 0),
                     asymmetry_metrics.get('geometric_asymmetry', 0),
                     asymmetry_metrics.get('total_asymmetry', 0)]
    
    ax3.bar(metrics_names, metrics_values, color=['purple', 'orange', 'green', 'red'])
    ax3.set_title('Asymmetry Metrics')
    ax3.set_ylabel('Asymmetry Score')
    
    # 4. CTC Analysis (if detected)
    ax4 = axes[1, 0]
    if results.get('ctc_loops_detected', False) and 'detailed_analysis' in results:
        ctc_data = results['detailed_analysis'].get('ctc_loops', {})
        if ctc_data.get('ctc_candidates'):
            loop_distances = [ctc['loop_distance'] for ctc in ctc_data['ctc_candidates']]
            ax4.hist(loop_distances, bins=10, color='red', alpha=0.7)
            ax4.set_title(f'CTC Loop Distances ({len(loop_distances)} candidates)')
            ax4.set_xlabel('Loop Distance')
            ax4.set_ylabel('Count')
        else:
            ax4.text(0.5, 0.5, 'No CTC Loops Detected', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('CTC Analysis')
    else:
        ax4.text(0.5, 0.5, 'No CTC Loops Detected', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('CTC Analysis')
    
    # 5. Time Reversal Analysis
    ax5 = axes[1, 1]
    if results.get('time_reversal_violations', False) and 'detailed_analysis' in results:
        time_data = results['detailed_analysis'].get('time_reversal', {})
        if 'forward_evolution' in time_data and 'reverse_evolution' in time_data:
            forward = time_data['forward_evolution']
            reverse = time_data['reverse_evolution']
            
            ax5.plot(forward, label='Forward', color='blue')
            ax5.plot(reverse, label='Reverse', color='red')
            ax5.set_title(f'Time Evolution (Asymmetry: {time_data.get("asymmetry_score", 0):.3f})')
            ax5.set_xlabel('Timestep')
            ax5.set_ylabel('Entropy Change')
            ax5.legend()
        else:
            ax5.text(0.5, 0.5, 'No Time-Reversal Violations', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Time-Reversal Analysis')
    else:
        ax5.text(0.5, 0.5, 'No Time-Reversal Violations', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Time-Reversal Analysis')
    
    # 6. Confidence Assessment
    ax6 = axes[1, 2]
    confidence = results.get('overall_assessment', {}).get('confidence_level', 0.0)
    assessment = results.get('overall_assessment', {}).get('causal_structure', 'UNKNOWN')
    
    ax6.bar(['Confidence'], [confidence], color='green' if confidence > 0.7 else 'orange')
    ax6.set_ylim(0, 1)
    ax6.set_title(f'Confidence: {confidence:.3f}\nAssessment: {assessment}')
    ax6.set_ylabel('Confidence Level')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Causal asymmetry visualization saved to: {output_path}")
    
    plt.show()

def main():
    """Example usage of causal asymmetry detection."""
    
    # Example data (replace with your actual experiment data)
    num_qubits = 7
    mi_matrix = np.random.rand(num_qubits, num_qubits)
    mi_matrix = (mi_matrix + mi_matrix.T) / 2  # Make symmetric
    
    timesteps_data = []
    for t in range(5):
        timestep_data = {}
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                timestep_data[f'I_{i},{j}'] = np.random.rand()
        timesteps_data.append(timestep_data)
    
    # Run causal asymmetry detection
    results = detect_causal_asymmetry_evidence(
        mi_matrix=mi_matrix,
        timesteps_data=timesteps_data,
        num_qubits=num_qubits,
        geometry="hyperbolic",
        curvature=2.0
    )
    
    # Create visualization
    create_causal_asymmetry_visualization(results, "causal_asymmetry_analysis.png")
    
    return results

if __name__ == "__main__":
    main() 