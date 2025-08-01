#!/usr/bin/env python3
"""
Causal Asymmetry Analysis - Detect emergent time structure and directional entropy transfer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

def load_experiment_data(file_path: str) -> Optional[Dict]:
    """Load experiment data and extract mutual information evolution."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract basic parameters
        spec = data.get('spec', {})
        num_qubits = spec.get('num_qubits', 0)
        curvature = spec.get('curvature', 0.0)
        geometry = spec.get('geometry', 'unknown')
        device = spec.get('device', 'unknown')
        
        # Extract mutual information evolution
        mi_evolution = data.get('mutual_information_per_timestep', [])
        if not mi_evolution:
            print(f"  ❌ No mutual information evolution data found")
            return None
        
        # Convert dictionary format to numpy matrices
        mi_matrices = []
        for timestep_data in mi_evolution:
            if isinstance(timestep_data, dict):
                # Convert dictionary format {"I_0,1": 0.1, "I_0,2": 0.1, ...} to matrix
                matrix = np.zeros((num_qubits, num_qubits))
                for key, value in timestep_data.items():
                    if key.startswith('I_'):
                        # Parse "I_0,1" -> (0, 1)
                        try:
                            qubits = key[2:].split(',')
                            i, j = int(qubits[0]), int(qubits[1])
                            matrix[i, j] = value
                            matrix[j, i] = value  # Symmetric
                        except:
                            continue
                mi_matrices.append(matrix)
            elif isinstance(timestep_data, str):
                # Parse string representation
                try:
                    # Remove brackets and split
                    clean_str = timestep_data.strip('[]').replace('\n', '')
                    rows = clean_str.split(']')[:-1]  # Remove empty last element
                    matrix = []
                    for row in rows:
                        row_clean = row.strip('[').strip()
                        if row_clean:
                            row_values = [float(x.strip()) for x in row_clean.split() if x.strip()]
                            matrix.append(row_values)
                    mi_matrices.append(np.array(matrix))
                except:
                    print(f"  ❌ Failed to parse MI matrix string")
                    return None
            elif isinstance(timestep_data, list):
                # Already in matrix format
                mi_matrices.append(np.array(timestep_data))
            else:
                print(f"  ❌ Unknown MI data format: {type(timestep_data)}")
                return None
        
        return {
            'file_path': file_path,
            'num_qubits': num_qubits,
            'curvature': curvature,
            'geometry': geometry,
            'device': device,
            'mi_evolution': mi_matrices,
            'timesteps': len(mi_matrices)
        }
        
    except Exception as e:
        print(f"  ❌ Error loading {file_path}: {e}")
        return None

def analyze_causal_asymmetry(mi_evolution: List[np.ndarray]) -> Dict:
    """Analyze causal asymmetry in mutual information evolution."""
    if len(mi_evolution) < 1:
        return {}
    
    results = {}
    
    # Handle single timestep case
    if len(mi_evolution) == 1:
        print("  📊 Single timestep analysis (no evolution data)")
        mi_matrix = mi_evolution[0]
        n = mi_matrix.shape[0]
        
        # Analyze spatial structure
        spatial_asymmetry = analyze_spatial_asymmetry(mi_matrix)
        results['spatial_asymmetry'] = spatial_asymmetry
        
        # Analyze connectivity patterns
        connectivity = analyze_connectivity_patterns(mi_matrix)
        results['connectivity_patterns'] = connectivity
        
        return results
    
    # Multi-timestep analysis
    if len(mi_evolution) < 2:
        print("  ⚠️  Insufficient timesteps for evolution analysis")
        return results
    
    # 1. Temporal asymmetry analysis
    print("  📊 Analyzing temporal asymmetry...")
    
    # Calculate MI change between consecutive timesteps
    mi_changes = []
    for i in range(len(mi_evolution) - 1):
        change = mi_evolution[i + 1] - mi_evolution[i]
        mi_changes.append(change)
    
    # Analyze asymmetry in MI changes
    forward_changes = []
    backward_changes = []
    
    for change_matrix in mi_changes:
        # Forward: upper triangle (future influence)
        # Backward: lower triangle (past influence)
        n = change_matrix.shape[0]
        forward = []
        backward = []
        
        for i in range(n):
            for j in range(i+1, n):
                    forward.append(change_matrix[i, j])
                backward.append(change_matrix[j, i])
        
        forward_changes.extend(forward)
        backward_changes.extend(backward)
    
    # Calculate temporal asymmetry
    if forward_changes and backward_changes:
        forward_mean = np.mean(forward_changes)
        backward_mean = np.mean(backward_changes)
        asymmetry_ratio = abs(forward_mean - backward_mean) / (abs(forward_mean) + abs(backward_mean) + 1e-10)
        
            results['temporal_asymmetry'] = {
            'forward_mean': float(forward_mean),
            'backward_mean': float(backward_mean),
            'asymmetry_ratio': float(asymmetry_ratio),
            'forward_std': float(np.std(forward_changes)),
            'backward_std': float(np.std(backward_changes))
        }
    
    # 2. Spatial asymmetry analysis (for each timestep)
    spatial_asymmetries = []
    for mi_matrix in mi_evolution:
        spatial = analyze_spatial_asymmetry(mi_matrix)
        spatial_asymmetries.append(spatial)
    
    results['spatial_asymmetry_evolution'] = spatial_asymmetries
    
    # 3. Connectivity pattern analysis
    connectivity_evolution = []
    for mi_matrix in mi_evolution:
        connectivity = analyze_connectivity_patterns(mi_matrix)
        connectivity_evolution.append(connectivity)
    
    results['connectivity_evolution'] = connectivity_evolution
    
    # 4. Causal violation analysis
    causal_violations = analyze_causal_violations(mi_evolution)
    results['causal_violations'] = causal_violations
    
    return results

def analyze_spatial_asymmetry(mi_matrix: np.ndarray) -> Dict:
    """Analyze spatial asymmetry in mutual information matrix."""
    n = mi_matrix.shape[0]
    
    # Calculate asymmetry between different spatial regions
    # For simplicity, divide into quadrants
    mid = n // 2
    
    # Top-left vs bottom-right
    tl = mi_matrix[:mid, :mid]
    br = mi_matrix[mid:, mid:]
    
    # Top-right vs bottom-left
    tr = mi_matrix[:mid, mid:]
    bl = mi_matrix[mid:, :mid]
    
    asymmetry_score = abs(np.mean(tl) - np.mean(br)) + abs(np.mean(tr) - np.mean(bl))
    asymmetry_score = asymmetry_score / (np.mean(mi_matrix) + 1e-10)
    
    return {
        'asymmetry_score': float(asymmetry_score),
        'top_left_mean': float(np.mean(tl)),
        'bottom_right_mean': float(np.mean(br)),
        'top_right_mean': float(np.mean(tr)),
        'bottom_left_mean': float(np.mean(bl)),
        'overall_mean': float(np.mean(mi_matrix))
    }

def analyze_connectivity_patterns(mi_matrix: np.ndarray) -> Dict:
    """Analyze connectivity patterns in mutual information matrix."""
    n = mi_matrix.shape[0]
    
    # Calculate average connectivity
    avg_connectivity = np.mean(mi_matrix)
    
    # Calculate clustering coefficient (simplified)
    # Count triangles in the graph
    triangles = 0
    total_triplets = 0
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if mi_matrix[i, j] > 0 and mi_matrix[j, k] > 0 and mi_matrix[i, k] > 0:
                    triangles += 1
                total_triplets += 1
    
    clustering_coeff = triangles / (total_triplets + 1e-10)
    
    return {
        'avg_connectivity': float(avg_connectivity),
        'clustering_coefficient': float(clustering_coeff),
        'total_connections': int(np.sum(mi_matrix > 0)),
        'max_connection': float(np.max(mi_matrix)),
        'min_connection': float(np.min(mi_matrix))
    }

def analyze_causal_violations(mi_evolution: List[np.ndarray]) -> Dict:
    """Analyze potential causal violations in MI evolution."""
    violations = []
    
    for i in range(len(mi_evolution) - 1):
        current_mi = mi_evolution[i]
        next_mi = mi_evolution[i + 1]
        
        # Check for violations of causality
        # This is a simplified check - in reality, we'd need more sophisticated analysis
        n = current_mi.shape[0]
        
        for j in range(n):
            for k in range(n):
                if j != k:
                    # Check if MI increases too dramatically (potential violation)
                    mi_change = next_mi[j, k] - current_mi[j, k]
                    if abs(mi_change) > 0.5:  # Threshold for violation
                        violations.append({
                            'timestep': i,
                            'qubits': (j, k),
                            'change': float(mi_change)
                        })
    
    return {
        'total_violations': len(violations),
        'violations': violations,
        'violation_rate': len(violations) / (len(mi_evolution) - 1) if len(mi_evolution) > 1 else 0
    }

def analyze_multiple_experiments(file_paths: List[str], min_timesteps: int = 6) -> Dict:
    """Analyze multiple experiments for causal asymmetry."""
    print("🔬 Causal Asymmetry Analysis")
    print("=" * 50)
    print(f"📊 Analyzing {len(file_paths)} experiments...")
    
    results = []
    experiments_analyzed = 0
    experiments_with_evolution = 0
    
    for i, file_path in enumerate(file_paths):
        print(f"[{i+1}/{len(file_paths)}] Analyzing {os.path.basename(file_path)}...")
        
        # Load experiment data
        data = load_experiment_data(file_path)
        if not data:
            continue
        
        # Check timestep requirement
        if data['timesteps'] < min_timesteps:
            print(f"  ⚠️  Insufficient timesteps ({data['timesteps']} < {min_timesteps})")
            # Still analyze if we have at least 1 timestep
            if data['timesteps'] >= 1:
                analysis_results = analyze_causal_asymmetry(data['mi_evolution'])
                if analysis_results:
                    results.append({
                        'file_path': data['file_path'],
                        'num_qubits': data['num_qubits'],
                        'curvature': data['curvature'],
                        'geometry': data['geometry'],
                        'device': data['device'],
                        'timesteps': data['timesteps'],
                        'analysis_results': analysis_results,
                        'has_evolution': data['timesteps'] > 1
                    })
                    experiments_analyzed += 1
                    if data['timesteps'] > 1:
                        experiments_with_evolution += 1
            continue
        
        # Analyze causal asymmetry
        analysis_results = analyze_causal_asymmetry(data['mi_evolution'])
        
        if analysis_results:
            results.append({
                'file_path': data['file_path'],
                'num_qubits': data['num_qubits'],
                'curvature': data['curvature'],
                'geometry': data['geometry'],
                'device': data['device'],
                'timesteps': data['timesteps'],
                'analysis_results': analysis_results,
                'has_evolution': data['timesteps'] > 1
            })
            experiments_analyzed += 1
            if data['timesteps'] > 1:
                experiments_with_evolution += 1
        else:
            print(f"  ❌ Analysis failed")
    
    print(f"\n📊 Analysis Complete:")
    print(f"  ✅ Experiments analyzed: {experiments_analyzed}")
    print(f"  📈 Experiments with evolution: {experiments_with_evolution}")
    print(f"  📊 Experiments with single timestep: {experiments_analyzed - experiments_with_evolution}")
    
    return {
        'results': results,
        'experiments_analyzed': experiments_analyzed,
        'experiments_with_evolution': experiments_with_evolution,
        'min_timesteps_required': min_timesteps
    }

def generate_visualizations(results: Dict, output_dir: str):
    """Generate visualization plots for causal asymmetry analysis."""
    print("📊 Generating visualizations...")
    
    if not results['results']:
        print("  ❌ No results to visualize")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Temporal asymmetry evolution plot
    temporal_data = []
    for result in results['results']:
        if result.get('has_evolution', False):
            temporal = result['analysis_results'].get('temporal_asymmetry', {})
            if temporal:
                temporal_data.append({
                    'file': os.path.basename(result['file_path']),
                    'num_qubits': result['num_qubits'],
                    'curvature': result['curvature'],
                    'geometry': result['geometry'],
                    'device': result['device'],
                    'asymmetry_ratio': temporal.get('asymmetry_ratio', 0),
                    'forward_mean': temporal.get('forward_mean', 0),
                    'backward_mean': temporal.get('backward_mean', 0)
                })
    
    if temporal_data:
        plt.figure(figsize=(12, 8))
        
        # Plot asymmetry ratios
        plt.subplot(2, 2, 1)
        curvatures = [d['curvature'] for d in temporal_data]
        asymmetry_ratios = [d['asymmetry_ratio'] for d in temporal_data]
        plt.scatter(curvatures, asymmetry_ratios, alpha=0.7)
        plt.xlabel('Curvature')
        plt.ylabel('Temporal Asymmetry Ratio')
        plt.title('Temporal Asymmetry vs Curvature')
        plt.grid(True, alpha=0.3)
        
        # Plot forward vs backward means
        plt.subplot(2, 2, 2)
        forward_means = [d['forward_mean'] for d in temporal_data]
        backward_means = [d['backward_mean'] for d in temporal_data]
        plt.scatter(forward_means, backward_means, alpha=0.7)
        plt.xlabel('Forward Mean')
        plt.ylabel('Backward Mean')
        plt.title('Forward vs Backward MI Changes')
        plt.plot([min(forward_means), max(forward_means)], [min(forward_means), max(forward_means)], 'r--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        # Plot by device type
        plt.subplot(2, 2, 3)
        devices = {}
        for d in temporal_data:
            device = d['device']
            if device not in devices:
                devices[device] = []
            devices[device].append(d['asymmetry_ratio'])
        
        device_names = list(devices.keys())
        device_means = [np.mean(devices[d]) for d in device_names]
        plt.bar(device_names, device_means)
        plt.ylabel('Mean Asymmetry Ratio')
        plt.title('Asymmetry by Device Type')
        plt.xticks(rotation=45)
        
        # Plot by geometry
        plt.subplot(2, 2, 4)
        geometries = {}
        for d in temporal_data:
            geom = d['geometry']
            if geom not in geometries:
                geometries[geom] = []
            geometries[geom].append(d['asymmetry_ratio'])
        
        geom_names = list(geometries.keys())
        geom_means = [np.mean(geometries[g]) for g in geom_names]
        plt.bar(geom_names, geom_means)
        plt.ylabel('Mean Asymmetry Ratio')
        plt.title('Asymmetry by Geometry')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/temporal_asymmetry_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Temporal asymmetry plot saved")
    
    # 2. Spatial asymmetry analysis
    spatial_data = []
    for result in results['results']:
        spatial_evolution = result['analysis_results'].get('spatial_asymmetry_evolution', [])
        if spatial_evolution:
            # Take the first timestep for spatial analysis
            spatial = spatial_evolution[0]
            spatial_data.append({
                'file': os.path.basename(result['file_path']),
                'num_qubits': result['num_qubits'],
                'curvature': result['curvature'],
                'geometry': result['geometry'],
                'asymmetry_score': spatial.get('asymmetry_score', 0),
                'overall_mean': spatial.get('overall_mean', 0)
            })
    
    if spatial_data:
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        curvatures = [d['curvature'] for d in spatial_data]
        asymmetry_scores = [d['asymmetry_score'] for d in spatial_data]
        plt.scatter(curvatures, asymmetry_scores, alpha=0.7)
        plt.xlabel('Curvature')
        plt.ylabel('Spatial Asymmetry Score')
        plt.title('Spatial Asymmetry vs Curvature')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        qubit_counts = [d['num_qubits'] for d in spatial_data]
        plt.scatter(qubit_counts, asymmetry_scores, alpha=0.7)
        plt.xlabel('Number of Qubits')
        plt.ylabel('Spatial Asymmetry Score')
        plt.title('Spatial Asymmetry vs Qubit Count')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/spatial_asymmetry_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Spatial asymmetry plot saved")
    
    # 3. Connectivity analysis
    connectivity_data = []
    for result in results['results']:
        connectivity_evolution = result['analysis_results'].get('connectivity_evolution', [])
        if connectivity_evolution:
            # Take the first timestep
            connectivity = connectivity_evolution[0]
            connectivity_data.append({
                'file': os.path.basename(result['file_path']),
                'num_qubits': result['num_qubits'],
                'curvature': result['curvature'],
                'avg_connectivity': connectivity.get('avg_connectivity', 0),
                'clustering_coefficient': connectivity.get('clustering_coefficient', 0),
                'total_connections': connectivity.get('total_connections', 0)
            })
    
    if connectivity_data:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        curvatures = [d['curvature'] for d in connectivity_data]
        avg_connectivity = [d['avg_connectivity'] for d in connectivity_data]
        plt.scatter(curvatures, avg_connectivity, alpha=0.7)
        plt.xlabel('Curvature')
        plt.ylabel('Average Connectivity')
        plt.title('Connectivity vs Curvature')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        clustering_coeffs = [d['clustering_coefficient'] for d in connectivity_data]
        plt.scatter(curvatures, clustering_coeffs, alpha=0.7)
        plt.xlabel('Curvature')
        plt.ylabel('Clustering Coefficient')
        plt.title('Clustering vs Curvature')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        total_connections = [d['total_connections'] for d in connectivity_data]
        plt.scatter(curvatures, total_connections, alpha=0.7)
        plt.xlabel('Curvature')
        plt.ylabel('Total Connections')
        plt.title('Total Connections vs Curvature')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/connectivity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Connectivity analysis plot saved")
    
    # 4. Causal violations summary
    violation_data = []
    for result in results['results']:
        violations = result['analysis_results'].get('causal_violations', {})
        if violations:
            violation_data.append({
                'file': os.path.basename(result['file_path']),
                'num_qubits': result['num_qubits'],
                'curvature': result['curvature'],
                'total_violations': violations.get('total_violations', 0),
                'violation_rate': violations.get('violation_rate', 0)
            })
    
    if violation_data:
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        curvatures = [d['curvature'] for d in violation_data]
        total_violations = [d['total_violations'] for d in violation_data]
        plt.scatter(curvatures, total_violations, alpha=0.7)
        plt.xlabel('Curvature')
        plt.ylabel('Total Causal Violations')
        plt.title('Causal Violations vs Curvature')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        violation_rates = [d['violation_rate'] for d in violation_data]
        plt.scatter(curvatures, violation_rates, alpha=0.7)
        plt.xlabel('Curvature')
        plt.ylabel('Violation Rate')
        plt.title('Violation Rate vs Curvature')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
        plt.savefig(f"{output_dir}/causal_violations_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
        print(f"  ✅ Causal violations plot saved")
    
    print(f"  ✅ All visualizations saved to {output_dir}")

def generate_summary_report(results: Dict) -> str:
    """Generate a comprehensive summary report of the causal asymmetry analysis."""
    print("📝 Generating Summary Report...")
    
    if not results['results']:
        return "❌ No results to summarize"
    
    report = f"""
# Causal Asymmetry Analysis Summary Report

## Analysis Overview
- **Total experiments analyzed**: {results['experiments_analyzed']}
- **Experiments with evolution data**: {results['experiments_with_evolution']}
- **Experiments with single timestep**: {results['experiments_analyzed'] - results['experiments_with_evolution']}
- **Minimum timesteps required**: {results['min_timesteps_required']}

## Key Findings

### Temporal Asymmetry Analysis
"""
    
    # Extract temporal asymmetry data
    temporal_results = []
    for result in results['results']:
        if result.get('has_evolution', False):
            temporal = result['analysis_results'].get('temporal_asymmetry', {})
            if temporal:
                temporal_results.append({
                    'file': os.path.basename(result['file_path']),
                    'num_qubits': result['num_qubits'],
                    'curvature': result['curvature'],
                    'geometry': result['geometry'],
                    'device': result['device'],
                    'asymmetry_ratio': temporal.get('asymmetry_ratio', 0),
                    'forward_mean': temporal.get('forward_mean', 0),
                    'backward_mean': temporal.get('backward_mean', 0)
                })
    
    if temporal_results:
        asymmetry_ratios = [r['asymmetry_ratio'] for r in temporal_results]
        mean_asymmetry = np.mean(asymmetry_ratios)
        max_asymmetry = np.max(asymmetry_ratios)
        
        report += f"""
- **Mean temporal asymmetry ratio**: {mean_asymmetry:.4f}
- **Maximum temporal asymmetry**: {max_asymmetry:.4f}
- **Experiments with significant asymmetry (>0.1)**: {sum(1 for r in asymmetry_ratios if r > 0.1)}

**Top 5 experiments by temporal asymmetry:**
"""
        
        # Sort by asymmetry ratio
        sorted_results = sorted(temporal_results, key=lambda x: x['asymmetry_ratio'], reverse=True)
        for i, result in enumerate(sorted_results[:5]):
            report += f"{i+1}. {result['file']}: {result['asymmetry_ratio']:.4f} (curvature={result['curvature']}, {result['device']})\n"
    else:
        report += "- **No temporal evolution data available**\n"
    
    # Spatial asymmetry analysis
    report += f"""
### Spatial Asymmetry Analysis
"""
    
    spatial_results = []
    for result in results['results']:
        spatial_evolution = result['analysis_results'].get('spatial_asymmetry_evolution', [])
        if spatial_evolution:
            spatial = spatial_evolution[0]  # Take first timestep
            spatial_results.append({
                'file': os.path.basename(result['file_path']),
                'num_qubits': result['num_qubits'],
                'curvature': result['curvature'],
                'geometry': result['geometry'],
                'asymmetry_score': spatial.get('asymmetry_score', 0)
            })
    
    if spatial_results:
        asymmetry_scores = [r['asymmetry_score'] for r in spatial_results]
        mean_spatial_asymmetry = np.mean(asymmetry_scores)
        max_spatial_asymmetry = np.max(asymmetry_scores)
        
        report += f"""
- **Mean spatial asymmetry score**: {mean_spatial_asymmetry:.4f}
- **Maximum spatial asymmetry**: {max_spatial_asymmetry:.4f}
- **Experiments with significant spatial asymmetry (>0.1)**: {sum(1 for s in asymmetry_scores if s > 0.1)}

**Top 5 experiments by spatial asymmetry:**
"""
        
        sorted_spatial = sorted(spatial_results, key=lambda x: x['asymmetry_score'], reverse=True)
        for i, result in enumerate(sorted_spatial[:5]):
            report += f"{i+1}. {result['file']}: {result['asymmetry_score']:.4f} (curvature={result['curvature']}, {result['geometry']})\n"
    else:
        report += "- **No spatial asymmetry data available**\n"
    
    # Connectivity analysis
    report += f"""
### Connectivity Analysis
"""
    
    connectivity_results = []
    for result in results['results']:
        connectivity_evolution = result['analysis_results'].get('connectivity_evolution', [])
        if connectivity_evolution:
            connectivity = connectivity_evolution[0]  # Take first timestep
            connectivity_results.append({
                'file': os.path.basename(result['file_path']),
                'num_qubits': result['num_qubits'],
                'curvature': result['curvature'],
                'avg_connectivity': connectivity.get('avg_connectivity', 0),
                'clustering_coefficient': connectivity.get('clustering_coefficient', 0)
            })
    
    if connectivity_results:
        avg_connectivities = [r['avg_connectivity'] for r in connectivity_results]
        clustering_coeffs = [r['clustering_coefficient'] for r in connectivity_results]
        
        report += f"""
- **Mean average connectivity**: {np.mean(avg_connectivities):.4f}
- **Mean clustering coefficient**: {np.mean(clustering_coeffs):.4f}
- **Experiments with high connectivity (>0.5)**: {sum(1 for c in avg_connectivities if c > 0.5)}
- **Experiments with high clustering (>0.3)**: {sum(1 for c in clustering_coeffs if c > 0.3)}
"""
    else:
        report += "- **No connectivity data available**\n"
    
    # Causal violations
    report += f"""
### Causal Violations Analysis
"""
    
    violation_results = []
    for result in results['results']:
        violations = result['analysis_results'].get('causal_violations', {})
        if violations:
            violation_results.append({
                'file': os.path.basename(result['file_path']),
                'num_qubits': result['num_qubits'],
                'curvature': result['curvature'],
                'total_violations': violations.get('total_violations', 0),
                'violation_rate': violations.get('violation_rate', 0)
            })
    
    if violation_results:
        total_violations = sum(r['total_violations'] for r in violation_results)
        violation_rates = [r['violation_rate'] for r in violation_results]
        
        report += f"""
- **Total causal violations across all experiments**: {total_violations}
- **Mean violation rate**: {np.mean(violation_rates):.4f}
- **Experiments with violations**: {sum(1 for r in violation_results if r['total_violations'] > 0)}
"""
        
        if total_violations > 0:
            report += f"""
**Experiments with causal violations:**
"""
            for result in violation_results:
                if result['total_violations'] > 0:
                    report += f"- {result['file']}: {result['total_violations']} violations (rate={result['violation_rate']:.4f})\n"
    else:
        report += "- **No causal violations detected**\n"
    
    # Device and geometry analysis
    report += f"""
### Device and Geometry Analysis
"""
    
    device_stats = {}
    geometry_stats = {}
    
    for result in results['results']:
        device = result['device']
        geometry = result['geometry']
        
        if device not in device_stats:
            device_stats[device] = {'count': 0, 'asymmetries': []}
        if geometry not in geometry_stats:
            geometry_stats[geometry] = {'count': 0, 'asymmetries': []}
        
        device_stats[device]['count'] += 1
        geometry_stats[geometry]['count'] += 1
        
        # Add temporal asymmetry if available
        if result.get('has_evolution', False):
            temporal = result['analysis_results'].get('temporal_asymmetry', {})
            if temporal:
                asymmetry = temporal.get('asymmetry_ratio', 0)
                device_stats[device]['asymmetries'].append(asymmetry)
                geometry_stats[geometry]['asymmetries'].append(asymmetry)
    
    report += f"""
**Device Statistics:**
"""
    for device, stats in device_stats.items():
        if stats['asymmetries']:
            mean_asymmetry = np.mean(stats['asymmetries'])
            report += f"- {device}: {stats['count']} experiments, mean asymmetry={mean_asymmetry:.4f}\n"
        else:
            report += f"- {device}: {stats['count']} experiments, no evolution data\n"
    
    report += f"""
**Geometry Statistics:**
"""
    for geometry, stats in geometry_stats.items():
        if stats['asymmetries']:
            mean_asymmetry = np.mean(stats['asymmetries'])
            report += f"- {geometry}: {stats['count']} experiments, mean asymmetry={mean_asymmetry:.4f}\n"
        else:
            report += f"- {geometry}: {stats['count']} experiments, no evolution data\n"
    
    # Conclusions
    report += f"""
## Conclusions

### Emergent Time Structure Assessment
"""
    
    if temporal_results:
        mean_asymmetry = np.mean([r['asymmetry_ratio'] for r in temporal_results])
        if mean_asymmetry > 0.1:
            report += f"""
✅ **EMERGENT TIME STRUCTURE DETECTED**
- Strong temporal asymmetry observed (mean ratio: {mean_asymmetry:.4f})
- Evidence for directional information flow
- Quantum spacetime signatures present
"""
        elif mean_asymmetry > 0.05:
            report += f"""
⚠️ **WEAK EMERGENT TIME STRUCTURE**
- Moderate temporal asymmetry observed (mean ratio: {mean_asymmetry:.4f})
- Some evidence for directional effects
- Further investigation recommended
"""
        else:
        report += f"""
❌ **NO EMERGENT TIME STRUCTURE DETECTED**
- Low temporal asymmetry (mean ratio: {mean_asymmetry:.4f})
- No significant directional effects
- Consider increasing curvature or entanglement strength
"""
    else:
        report += f"""
❌ **INSUFFICIENT DATA FOR TIME STRUCTURE ANALYSIS**
- No temporal evolution data available
- Need experiments with multiple timesteps
- Consider running experiments with --timesteps 6 or higher
"""
    
    # Recommendations
    report += f"""
## Recommendations

### For Stronger Quantum Spacetime Signatures:
1. **Increase curvature values** (try 10.0-20.0)
2. **Use more timesteps** (6-12 timesteps minimum)
3. **Enhance entanglement strength** (weight=8.0, gamma=5.0)
4. **Use larger qubit counts** (8-12 qubits)
5. **Enable charge injection** for stronger perturbations
6. **Run on real hardware** for quantum noise effects

### Next Steps:
1. Run experiments with enhanced parameters
2. Analyze specific experiments with high asymmetry
3. Investigate spatial-temporal correlations
4. Validate with quantum coherence tests
"""
    
    report += f"""
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return report

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Causal Asymmetry Analysis')
    parser.add_argument('file_patterns', nargs='+', help='File patterns to analyze')
    parser.add_argument('--timesteps', type=int, default=6, help='Minimum timesteps required')
    parser.add_argument('--plot_evolution', action='store_true', help='Generate evolution plots')
    parser.add_argument('--metric_evolution', action='store_true', help='Analyze metric evolution')
    parser.add_argument('--output_dir', default='experiment_logs/causal_asymmetry_analysis', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Find all matching files
    file_paths = []
    for pattern in args.file_patterns:
        if os.path.isfile(pattern):
            file_paths.append(pattern)
        elif os.path.isdir(pattern):
            # Search directory for JSON files
            for root, dirs, files in os.walk(pattern):
                for file in files:
                    if file.startswith('results_') and file.endswith('.json'):
                        file_paths.append(os.path.join(root, file))
        else:
            # Try glob pattern
            import glob
            file_paths.extend(glob.glob(pattern))
    
    if not file_paths:
        print("❌ No files found matching the patterns")
        return
    
    print(f"🔍 Found {len(file_paths)} files to analyze")
    
    # Run analysis
    results = analyze_multiple_experiments(file_paths, args.timesteps)
    
    if not results['results']:
        print("❌ No experiments could be analyzed!")
        return
    
    # Generate visualizations
    generate_visualizations(results, args.output_dir)
    
    # Generate and save summary report
    report = generate_summary_report(results)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_file = f"{args.output_dir}/causal_asymmetry_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save summary report
    report_file = f"{args.output_dir}/causal_asymmetry_summary_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n💾 Results saved to {args.output_dir}")
    print(f"📄 Detailed results: {os.path.basename(results_file)}")
    print(f"📝 Summary report: {os.path.basename(report_file)}")
    
    # Print key findings
    print(f"\n🔍 Key Findings:")
    print(f"📊 Experiments analyzed: {results['experiments_analyzed']}")
    
    if results['results']:
        temporal_asymmetries = []
        causal_violations = []
        for result in results['results']:
            temporal = result['analysis_results'].get('temporal_asymmetry', {})
            temporal_asymmetries.append(temporal.get('asymmetry_ratio', 0))
            
            violations = result['analysis_results'].get('causal_violations', {})
            causal_violations.append(violations.get('total_violations', 0))
        
        print(f"📈 Mean temporal asymmetry: {np.mean(temporal_asymmetries):.4f}")
        print(f"🚨 Total causal violations: {sum(causal_violations)}")
        print(f"🎯 Emergent time structure: {'DETECTED' if np.mean(temporal_asymmetries) > 0.05 else 'NOT DETECTED'}")

if __name__ == "__main__":
    main() 