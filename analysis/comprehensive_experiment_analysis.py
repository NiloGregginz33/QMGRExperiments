#!/usr/bin/env python3
"""
Comprehensive Experiment Analysis for "Undeniable" Evidence
Analyzes current experiment results and generates specific recommendations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

def analyze_experiment_file(file_path: str) -> Dict:
    """Comprehensive analysis of a single experiment file."""
    print(f"🔍 Analyzing: {os.path.basename(file_path)}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    spec = data.get('spec', {})
    results = {
        'file_path': file_path,
        'spec': spec,
        'issues': [],
        'recommendations': [],
        'metrics': {}
    }
    
    # Extract key parameters
    num_qubits = spec.get('num_qubits', 0)
    curvature = spec.get('curvature', 0.0)
    geometry = spec.get('geometry', 'unknown')
    timesteps = spec.get('timesteps', 0)
    device = spec.get('device', 'unknown')
    
    print(f"  📊 Parameters: {num_qubits} qubits, curvature={curvature}, {geometry} geometry, {timesteps} timesteps")
    
    # Analyze mutual information evolution
    mi_evolution = data.get('mutual_information_per_timestep', [])
    if mi_evolution:
        mi_analysis = analyze_mutual_information_evolution(mi_evolution)
        results['metrics']['mi_evolution'] = mi_analysis
        
        # Check for static MI (all values identical)
        if mi_analysis['is_static']:
            results['issues'].append("CRITICAL: Mutual information is static (no evolution)")
            results['recommendations'].append("Increase entanglement strength and circuit depth")
            results['recommendations'].append("Use more timesteps (8-12 minimum)")
            results['recommendations'].append("Enable stronger charge injection")
    
    # Analyze entropy evolution
    entropy_evolution = data.get('entropy_per_timestep', [])
    if entropy_evolution:
        entropy_analysis = analyze_entropy_evolution(entropy_evolution)
        results['metrics']['entropy_evolution'] = entropy_analysis
    
    # Analyze boundary entropies
    boundary_entropies = data.get('boundary_entropies_per_timestep', [])
    if boundary_entropies:
        boundary_analysis = analyze_boundary_entropies(boundary_entropies)
        results['metrics']['boundary_entropies'] = boundary_analysis
    
    # Generate specific recommendations
    recommendations = generate_specific_recommendations(spec, results['metrics'])
    results['recommendations'].extend(recommendations)
    
    return results

def analyze_mutual_information_evolution(mi_evolution: List[Dict]) -> Dict:
    """Analyze mutual information evolution across timesteps."""
    if not mi_evolution:
        return {'error': 'No MI evolution data'}
    
    # Convert to matrix format
    mi_matrices = []
    for timestep_data in mi_evolution:
        if isinstance(timestep_data, dict):
            # Convert dict to matrix
            n = int(np.sqrt(len(timestep_data) * 2))  # Estimate matrix size
            matrix = np.zeros((n, n))
            
            for key, value in timestep_data.items():
                if key.startswith('I_'):
                    # Parse "I_i,j" format
                    try:
                        i, j = map(int, key[2:].split(','))
                        matrix[i, j] = value
                        matrix[j, i] = value  # Symmetric
                    except:
                        continue
            
            mi_matrices.append(matrix)
    
    # Analyze evolution
    if len(mi_matrices) < 2:
        return {'error': 'Insufficient timesteps for evolution analysis'}
    
    # Check if MI is static
    first_matrix = mi_matrices[0]
    is_static = all(np.allclose(matrix, first_matrix, atol=1e-6) for matrix in mi_matrices)
    
    # Calculate evolution metrics
    changes = []
    for i in range(len(mi_matrices) - 1):
        change = np.mean(np.abs(mi_matrices[i+1] - mi_matrices[i]))
        changes.append(change)
    
    return {
        'num_timesteps': len(mi_matrices),
        'is_static': is_static,
        'mean_change': np.mean(changes) if changes else 0.0,
        'max_change': np.max(changes) if changes else 0.0,
        'evolution_variance': np.var(changes) if changes else 0.0,
        'mi_matrices': mi_matrices
    }

def analyze_entropy_evolution(entropy_evolution: List) -> Dict:
    """Analyze entropy evolution across timesteps."""
    if not entropy_evolution or all(e is None for e in entropy_evolution):
        return {'error': 'No valid entropy data'}
    
    # Filter out None values
    valid_entropies = [e for e in entropy_evolution if e is not None]
    
    if len(valid_entropies) < 2:
        return {'error': 'Insufficient entropy data for evolution analysis'}
    
    changes = np.diff(valid_entropies)
    
    return {
        'num_timesteps': len(valid_entropies),
        'mean_entropy': np.mean(valid_entropies),
        'entropy_variance': np.var(valid_entropies),
        'mean_change': np.mean(changes),
        'change_variance': np.var(changes),
        'is_monotonic': np.all(changes >= 0) or np.all(changes <= 0)
    }

def analyze_boundary_entropies(boundary_entropies: List[Dict]) -> Dict:
    """Analyze boundary entropy patterns."""
    if not boundary_entropies:
        return {'error': 'No boundary entropy data'}
    
    # Extract key metrics
    entropy_A_values = []
    entropy_B_values = []
    mi_AB_values = []
    
    for timestep_data in boundary_entropies:
        if isinstance(timestep_data, dict):
            entropy_A_values.append(timestep_data.get('entropy_A', 0))
            entropy_B_values.append(timestep_data.get('entropy_B', 0))
            mi_AB_values.append(timestep_data.get('mi_AB', 0))
    
    if not entropy_A_values:
        return {'error': 'No valid boundary entropy data'}
    
    return {
        'num_timesteps': len(entropy_A_values),
        'mean_entropy_A': np.mean(entropy_A_values),
        'mean_entropy_B': np.mean(entropy_B_values),
        'mean_mi_AB': np.mean(mi_AB_values),
        'entropy_correlation': pearsonr(entropy_A_values, entropy_B_values)[0] if len(entropy_A_values) > 1 else 0.0,
        'entropy_difference': np.mean(np.abs(np.array(entropy_A_values) - np.array(entropy_B_values)))
    }

def generate_specific_recommendations(spec: Dict, metrics: Dict) -> List[str]:
    """Generate specific recommendations based on analysis."""
    recommendations = []
    
    # Parameter analysis
    num_qubits = spec.get('num_qubits', 0)
    curvature = spec.get('curvature', 0.0)
    timesteps = spec.get('timesteps', 0)
    geometry = spec.get('geometry', 'unknown')
    
    # Qubit count recommendations
    if num_qubits < 8:
        recommendations.append(f"Increase qubit count from {num_qubits} to 8-12 for better geometric resolution")
    
    # Curvature recommendations
    if curvature < 10.0:
        recommendations.append(f"Increase curvature from {curvature} to 10.0-20.0 for stronger geometric effects")
    
    # Timestep recommendations
    if timesteps < 6:
        recommendations.append(f"Increase timesteps from {timesteps} to 8-12 for better evolution tracking")
    
    # Geometry recommendations
    if geometry == 'spherical':
        recommendations.append("Consider using hyperbolic geometry for stronger causal asymmetry")
    
    # MI evolution recommendations
    if 'mi_evolution' in metrics:
        mi_analysis = metrics['mi_evolution']
        if mi_analysis.get('is_static', False):
            recommendations.append("CRITICAL: Enable stronger entanglement with --entanglement_strength 8.0")
            recommendations.append("CRITICAL: Increase charge injection strength to 5.0")
            recommendations.append("CRITICAL: Use longer circuit depth with --quantum_circuit_depth 20")
    
    # Specific parameter recommendations
    recommendations.extend([
        "Use --strong_curvature flag for enhanced geometric effects",
        "Enable --charge_injection with --charge_strength 5.0",
        "Enable --spin_injection with --spin_strength 3.0",
        "Use --lorentzian flag for temporal signature",
        "Increase --shots to 8192 for better statistics",
        "Enable --error_mitigation for hardware runs"
    ])
    
    return recommendations

def generate_improved_experiment_config(spec: Dict, recommendations: List[str]) -> Dict:
    """Generate an improved experiment configuration."""
    improved_spec = spec.copy()
    
    # Apply specific improvements
    improved_spec['num_qubits'] = max(spec.get('num_qubits', 6), 8)
    improved_spec['curvature'] = max(spec.get('curvature', 8.0), 15.0)
    improved_spec['timesteps'] = max(spec.get('timesteps', 3), 8)
    improved_spec['shots'] = max(spec.get('shots', 4096), 8192)
    improved_spec['charge_strength'] = max(spec.get('charge_strength', 2.5), 5.0)
    improved_spec['spin_strength'] = max(spec.get('spin_strength', 2.0), 3.0)
    improved_spec['entanglement_strength'] = max(spec.get('entanglement_strength', 0.35), 8.0)
    improved_spec['quantum_circuit_depth'] = max(spec.get('quantum_circuit_depth', 12), 20)
    
    # Enable critical flags
    improved_spec['strong_curvature'] = True
    improved_spec['charge_injection'] = True
    improved_spec['spin_injection'] = True
    improved_spec['lorentzian'] = True
    improved_spec['error_mitigation'] = True
    
    # Change geometry if needed
    if spec.get('geometry') == 'spherical':
        improved_spec['geometry'] = 'hyperbolic'
    
    return improved_spec

def main():
    """Main analysis function."""
    if len(sys.argv) < 2:
        print("Usage: python comprehensive_experiment_analysis.py <experiment_file>")
        return
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return
    
    # Analyze the experiment
    results = analyze_experiment_file(file_path)
    
    # Generate improved configuration
    improved_spec = generate_improved_experiment_config(results['spec'], results['recommendations'])
    
    # Print results
    print("\n" + "="*60)
    print("🔍 COMPREHENSIVE EXPERIMENT ANALYSIS")
    print("="*60)
    
    print(f"\n📊 EXPERIMENT PARAMETERS:")
    spec = results['spec']
    print(f"  Qubits: {spec.get('num_qubits', 'N/A')}")
    print(f"  Curvature: {spec.get('curvature', 'N/A')}")
    print(f"  Geometry: {spec.get('geometry', 'N/A')}")
    print(f"  Timesteps: {spec.get('timesteps', 'N/A')}")
    print(f"  Device: {spec.get('device', 'N/A')}")
    
    print(f"\n🚨 IDENTIFIED ISSUES:")
    if results['issues']:
        for issue in results['issues']:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ No critical issues identified")
    
    print(f"\n💡 SPECIFIC RECOMMENDATIONS:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print(f"\n🚀 IMPROVED EXPERIMENT CONFIGURATION:")
    print("  Use these parameters for 'undeniable' evidence:")
    print(f"  --num_qubits {improved_spec['num_qubits']}")
    print(f"  --curvature {improved_spec['curvature']}")
    print(f"  --timesteps {improved_spec['timesteps']}")
    print(f"  --geometry {improved_spec['geometry']}")
    print(f"  --charge_strength {improved_spec['charge_strength']}")
    print(f"  --spin_strength {improved_spec['spin_strength']}")
    print(f"  --entanglement_strength {improved_spec['entanglement_strength']}")
    print(f"  --quantum_circuit_depth {improved_spec['quantum_circuit_depth']}")
    print(f"  --shots {improved_spec['shots']}")
    print("  --strong_curvature --charge_injection --spin_injection --lorentzian --error_mitigation")
    
    # Save detailed results
    output_dir = os.path.dirname(file_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Clean results for JSON serialization
    clean_results = {
        'file_path': results['file_path'],
        'spec': results['spec'],
        'issues': results['issues'],
        'recommendations': results['recommendations'],
        'metrics': {}
    }
    
    # Clean metrics for JSON serialization
    for key, value in results['metrics'].items():
        if isinstance(value, dict):
            clean_metrics = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    clean_metrics[k] = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    clean_metrics[k] = float(v)
                else:
                    clean_metrics[k] = v
            clean_results['metrics'][key] = clean_metrics
        else:
            clean_results['metrics'][key] = value
    
    results_file = os.path.join(output_dir, f"comprehensive_analysis_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump({
            'analysis_results': clean_results,
            'improved_configuration': improved_spec,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Generate command for improved experiment
    improved_cmd = f"""python run_experiment.py --args \\
  --num_qubits {improved_spec['num_qubits']} \\
  --curvature {improved_spec['curvature']} \\
  --timesteps {improved_spec['timesteps']} \\
  --geometry {improved_spec['geometry']} \\
  --charge_strength {improved_spec['charge_strength']} \\
  --spin_strength {improved_spec['spin_strength']} \\
  --entanglement_strength {improved_spec['entanglement_strength']} \\
  --quantum_circuit_depth {improved_spec['quantum_circuit_depth']} \\
  --shots {improved_spec['shots']} \\
  --strong_curvature --charge_injection --spin_injection --lorentzian --error_mitigation"""
    
    print(f"\n🎯 COMMAND FOR IMPROVED EXPERIMENT:")
    print(improved_cmd)

if __name__ == "__main__":
    main() 