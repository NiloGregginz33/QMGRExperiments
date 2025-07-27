#!/usr/bin/env python3
"""
Controlled Curvature Analysis
============================

This script performs controlled comparisons by grouping experiments with identical parameters
and analyzing curvature effects only within each controlled group.

The analysis:
1. Groups experiments by identical parameters (n_qubits, device, geometry, etc.)
2. Only compares curvature effects within each group
3. Identifies which parameter combinations show strongest curvature-geometry relationships
4. Creates a more accurate mapping by eliminating confounding factors
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from pathlib import Path
import ast
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

def parse_dict_string(dict_str):
    """Parse dictionary string from CSV"""
    if pd.isna(dict_str) or dict_str == '{}':
        return {}
    try:
        dict_str = dict_str.replace('np.float64(', '').replace(')', '')
        return ast.literal_eval(dict_str)
    except:
        return {}

def extract_parameters_from_filename(filename):
    """Extract parameters from filename pattern"""
    # Example: results_n7_geomH_curv12_ibm_brisbane_ZW7YZA.json
    parts = filename.replace('.json', '').split('_')
    
    params = {}
    
    # Extract n_qubits
    for i, part in enumerate(parts):
        if part.startswith('n') and part[1:].isdigit():
            params['n_qubits'] = int(part[1:])
            break
    
    # Extract geometry
    for i, part in enumerate(parts):
        if part.startswith('geom'):
            geom_code = part[4:]
            if geom_code == 'H':
                params['geometry'] = 'hyperbolic'
            elif geom_code == 'S':
                params['geometry'] = 'spherical'
            elif geom_code == 'E':
                params['geometry'] = 'euclidean'
            break
    
    # Extract curvature
    for i, part in enumerate(parts):
        if part.startswith('curv'):
            try:
                params['curvature'] = float(part[4:])
            except:
                pass
            break
    
    # Extract device
    device_keywords = ['ibm_brisbane', 'ibm_sherbrooke', 'simulator', 'sim']
    for i, part in enumerate(parts):
        if part in device_keywords:
            params['device'] = part
            break
    
    return params

def load_and_group_data():
    """Load data and group by identical parameters"""
    
    # Load the existing data
    df = pd.read_csv('analysis/curvature_mapping_data.csv')
    
    # Parse dictionary columns
    df['mi_stats_parsed'] = df['mi_stats'].apply(parse_dict_string)
    df['distance_stats_parsed'] = df['distance_stats'].apply(parse_dict_string)
    
    # Extract key metrics
    df['mi_variance'] = df['mi_stats_parsed'].apply(lambda x: x.get('mi_variance', np.nan))
    df['mi_range'] = df['mi_stats_parsed'].apply(lambda x: x.get('mi_range', np.nan))
    df['mean_mi'] = df['mi_stats_parsed'].apply(lambda x: x.get('mean_mi', np.nan))
    df['distance_variance'] = df['distance_stats_parsed'].apply(lambda x: x.get('distance_variance', np.nan))
    df['mean_distance'] = df['distance_stats_parsed'].apply(lambda x: x.get('mean_distance', np.nan))
    
    # Extract parameters from filename
    df['params'] = df['filename'].apply(extract_parameters_from_filename)
    
    # Create parameter groups
    groups = defaultdict(list)
    
    for idx, row in df.iterrows():
        if pd.notna(row['input_curvature']) and pd.notna(row['mi_variance']):
            params = row['params']
            if 'n_qubits' in params and 'geometry' in params and 'device' in params:
                # Create a unique key for each parameter combination
                key = f"n{params['n_qubits']}_{params['geometry']}_{params['device']}"
                groups[key].append({
                    'curvature': row['input_curvature'],
                    'mi_variance': row['mi_variance'],
                    'mi_range': row['mi_range'],
                    'distance_variance': row['distance_variance'],
                    'mean_distance': row['mean_distance'],
                    'filename': row['filename'],
                    'params': params
                })
    
    # Filter groups with multiple curvature values
    controlled_groups = {}
    for key, group_data in groups.items():
        curvatures = [item['curvature'] for item in group_data]
        unique_curvatures = len(set(curvatures))
        
        if unique_curvatures >= 2:  # At least 2 different curvature values
            controlled_groups[key] = group_data
    
    return controlled_groups, df

def analyze_controlled_groups(controlled_groups):
    """Analyze each controlled group for curvature effects"""
    
    print("="*80)
    print("CONTROLLED CURVATURE ANALYSIS")
    print("="*80)
    
    results = {}
    
    for group_key, group_data in controlled_groups.items():
        print(f"\n{'='*60}")
        print(f"GROUP: {group_key}")
        print(f"{'='*60}")
        
        # Extract data
        curvatures = [item['curvature'] for item in group_data]
        mi_variances = [item['mi_variance'] for item in group_data]
        mi_ranges = [item['mi_range'] for item in group_data if pd.notna(item['mi_range'])]
        distance_variances = [item['distance_variance'] for item in group_data if pd.notna(item['distance_variance'])]
        
        print(f"Parameter combination: {group_key}")
        print(f"Number of experiments: {len(group_data)}")
        print(f"Curvature range: {min(curvatures):.1f} - {max(curvatures):.1f}")
        print(f"Unique curvature values: {len(set(curvatures))}")
        
        # Analyze MI variance vs curvature
        if len(curvatures) >= 3:  # Need at least 3 points for correlation
            r_mi, p_mi = stats.pearsonr(curvatures, mi_variances)
            spearman_r_mi, spearman_p_mi = stats.spearmanr(curvatures, mi_variances)
            
            print(f"\nMI Variance vs Curvature:")
            print(f"  Pearson r: {r_mi:.3f} (p={p_mi:.3e})")
            print(f"  Spearman r: {spearman_r_mi:.3f} (p={spearman_p_mi:.3e})")
            
            if p_mi < 0.05:
                print(f"  ✅ Statistically significant correlation!")
            else:
                print(f"  ❌ No significant correlation")
            
            # Analyze MI range vs curvature
            if len(mi_ranges) >= 3:
                r_range, p_range = stats.pearsonr(curvatures[:len(mi_ranges)], mi_ranges)
                print(f"\nMI Range vs Curvature:")
                print(f"  Pearson r: {r_range:.3f} (p={p_range:.3e})")
                
                if p_range < 0.05:
                    print(f"  ✅ Statistically significant correlation!")
                else:
                    print(f"  ❌ No significant correlation")
            
            # Analyze distance variance vs curvature
            if len(distance_variances) >= 3:
                r_dist, p_dist = stats.pearsonr(curvatures[:len(distance_variances)], distance_variances)
                print(f"\nDistance Variance vs Curvature:")
                print(f"  Pearson r: {r_dist:.3f} (p={p_dist:.3e})")
                
                if p_dist < 0.05:
                    print(f"  ✅ Statistically significant correlation!")
                else:
                    print(f"  ❌ No significant correlation")
            
            # Store results
            results[group_key] = {
                'n_experiments': len(group_data),
                'curvature_range': [min(curvatures), max(curvatures)],
                'unique_curvatures': len(set(curvatures)),
                'mi_variance_correlation': {
                    'pearson_r': r_mi,
                    'pearson_p': p_mi,
                    'spearman_r': spearman_r_mi,
                    'spearman_p': spearman_p_mi,
                    'significant': p_mi < 0.05
                },
                'mi_range_correlation': {
                    'pearson_r': r_range if len(mi_ranges) >= 3 else None,
                    'pearson_p': p_range if len(mi_ranges) >= 3 else None,
                    'significant': p_range < 0.05 if len(mi_ranges) >= 3 else None
                },
                'distance_variance_correlation': {
                    'pearson_r': r_dist if len(distance_variances) >= 3 else None,
                    'pearson_p': p_dist if len(distance_variances) >= 3 else None,
                    'significant': p_dist < 0.05 if len(distance_variances) >= 3 else None
                },
                'data': group_data
            }
        else:
            print(f"  ⚠️  Insufficient data for correlation analysis (need ≥3 points)")
    
    return results

def create_controlled_plots(controlled_results):
    """Create plots for controlled groups with significant correlations"""
    
    print("\n" + "="*80)
    print("CREATING CONTROLLED ANALYSIS PLOTS")
    print("="*80)
    
    # Find groups with significant correlations
    significant_groups = []
    for group_key, result in controlled_results.items():
        if result['mi_variance_correlation']['significant']:
            significant_groups.append((group_key, result))
    
    if not significant_groups:
        print("No groups found with statistically significant correlations")
        return
    
    print(f"Found {len(significant_groups)} groups with significant correlations")
    
    # Create plots
    n_groups = len(significant_groups)
    fig, axes = plt.subplots(2, min(n_groups, 3), figsize=(15, 10))
    if n_groups == 1:
        axes = axes.reshape(1, -1)
    
    for i, (group_key, result) in enumerate(significant_groups[:3]):  # Plot first 3
        if n_groups == 1:
            ax = axes[i]
        else:
            ax = axes[0, i] if i < 3 else axes[1, i-3]
        
        # Extract data
        curvatures = [item['curvature'] for item in result['data']]
        mi_variances = [item['mi_variance'] for item in result['data']]
        
        # Plot
        ax.scatter(curvatures, mi_variances, alpha=0.7, s=50)
        
        # Add trend line
        if len(curvatures) >= 2:
            z = np.polyfit(curvatures, mi_variances, 1)
            p = np.poly1d(z)
            ax.plot(curvatures, p(curvatures), "r--", alpha=0.8)
        
        ax.set_xlabel('Input Curvature k')
        ax.set_ylabel('MI Variance')
        ax.set_title(f'{group_key}\nr={result["mi_variance_correlation"]["pearson_r"]:.3f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis/controlled_curvature_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create summary plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    group_names = []
    correlation_strengths = []
    p_values = []
    
    for group_key, result in controlled_results.items():
        if result['mi_variance_correlation']['significant']:
            group_names.append(group_key)
            correlation_strengths.append(abs(result['mi_variance_correlation']['pearson_r']))
            p_values.append(result['mi_variance_correlation']['pearson_p'])
    
    if group_names:
        bars = ax.bar(range(len(group_names)), correlation_strengths, alpha=0.7)
        
        # Color bars by significance
        for i, p_val in enumerate(p_values):
            if p_val < 0.01:
                bars[i].set_color('green')
            elif p_val < 0.05:
                bars[i].set_color('orange')
            else:
                bars[i].set_color('red')
        
        ax.set_xlabel('Parameter Groups')
        ax.set_ylabel('|Correlation Strength|')
        ax.set_title('Curvature-Geometry Correlation Strength by Parameter Group')
        ax.set_xticks(range(len(group_names)))
        ax.set_xticklabels(group_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('analysis/controlled_correlation_summary.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_controlled_summary(controlled_results):
    """Create a summary of controlled analysis results"""
    
    print("\n" + "="*80)
    print("CONTROLLED ANALYSIS SUMMARY")
    print("="*80)
    
    # Count significant correlations
    total_groups = len(controlled_results)
    significant_mi = sum(1 for r in controlled_results.values() if r['mi_variance_correlation']['significant'])
    significant_range = sum(1 for r in controlled_results.values() if r['mi_range_correlation']['significant'])
    significant_dist = sum(1 for r in controlled_results.values() if r['distance_variance_correlation']['significant'])
    
    print(f"\nTotal controlled groups: {total_groups}")
    print(f"Groups with significant MI variance correlation: {significant_mi}/{total_groups} ({significant_mi/total_groups*100:.1f}%)")
    print(f"Groups with significant MI range correlation: {significant_range}/{total_groups} ({significant_range/total_groups*100:.1f}%)")
    print(f"Groups with significant distance variance correlation: {significant_dist}/{total_groups} ({significant_dist/total_groups*100:.1f}%)")
    
    # Find strongest correlations
    if controlled_results:
        strongest_mi = max(controlled_results.items(), 
                          key=lambda x: abs(x[1]['mi_variance_correlation']['pearson_r']) if x[1]['mi_variance_correlation']['significant'] else 0)
        
        print(f"\nStrongest MI variance correlation:")
        print(f"  Group: {strongest_mi[0]}")
        print(f"  Correlation: {strongest_mi[1]['mi_variance_correlation']['pearson_r']:.3f}")
        print(f"  P-value: {strongest_mi[1]['mi_variance_correlation']['pearson_p']:.3e}")
        print(f"  Curvature range: {strongest_mi[1]['curvature_range'][0]:.1f} - {strongest_mi[1]['curvature_range'][1]:.1f}")
    
    # Parameter analysis
    print(f"\nParameter Analysis:")
    
    # Analyze by number of qubits
    n_qubits_groups = defaultdict(list)
    for group_key, result in controlled_results.items():
        if result['mi_variance_correlation']['significant']:
            n_qubits = int(group_key.split('_')[0][1:])  # Extract n from "n7_..."
            n_qubits_groups[n_qubits].append(result['mi_variance_correlation']['pearson_r'])
    
    for n_qubits, correlations in n_qubits_groups.items():
        print(f"  n={n_qubits} qubits: {len(correlations)} significant groups, avg |r|={np.mean([abs(r) for r in correlations]):.3f}")
    
    # Analyze by geometry
    geometry_groups = defaultdict(list)
    for group_key, result in controlled_results.items():
        if result['mi_variance_correlation']['significant']:
            geometry = group_key.split('_')[1]  # Extract geometry
            geometry_groups[geometry].append(result['mi_variance_correlation']['pearson_r'])
    
    for geometry, correlations in geometry_groups.items():
        print(f"  {geometry} geometry: {len(correlations)} significant groups, avg |r|={np.mean([abs(r) for r in correlations]):.3f}")
    
    # Analyze by device
    device_groups = defaultdict(list)
    for group_key, result in controlled_results.items():
        if result['mi_variance_correlation']['significant']:
            device = group_key.split('_')[2]  # Extract device
            device_groups[device].append(result['mi_variance_correlation']['pearson_r'])
    
    for device, correlations in device_groups.items():
        print(f"  {device} device: {len(correlations)} significant groups, avg |r|={np.mean([abs(r) for r in correlations]):.3f}")
    
    # Save detailed results
    summary_data = {
        'total_groups': total_groups,
        'significant_mi_groups': significant_mi,
        'significant_range_groups': significant_range,
        'significant_dist_groups': significant_dist,
        'strongest_correlation': {
            'group': strongest_mi[0] if controlled_results else None,
            'correlation': strongest_mi[1]['mi_variance_correlation']['pearson_r'] if controlled_results else None,
            'p_value': strongest_mi[1]['mi_variance_correlation']['pearson_p'] if controlled_results else None
        },
        'parameter_analysis': {
            'n_qubits': {str(k): len(v) for k, v in n_qubits_groups.items()},
            'geometry': {k: len(v) for k, v in geometry_groups.items()},
            'device': {k: len(v) for k, v in device_groups.items()}
        },
        'detailed_results': controlled_results
    }
    
    with open('analysis/controlled_analysis_results.json', 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: analysis/controlled_analysis_results.json")
    
    return summary_data

def main():
    """Main analysis function"""
    
    print("CONTROLLED CURVATURE ANALYSIS")
    print("="*80)
    
    # Load and group data
    controlled_groups, full_df = load_and_group_data()
    
    print(f"Found {len(controlled_groups)} controlled groups with multiple curvature values")
    
    if not controlled_groups:
        print("No controlled groups found. All experiments have different parameters.")
        return
    
    # Analyze each controlled group
    controlled_results = analyze_controlled_groups(controlled_groups)
    
    # Create plots
    create_controlled_plots(controlled_results)
    
    # Create summary
    summary = create_controlled_summary(controlled_results)
    
    print("\nControlled curvature analysis complete!")

if __name__ == "__main__":
    main() 