#!/usr/bin/env python3
"""
Curvature Mapping Analysis
==========================

This script analyzes the relationship between the input curvature parameter k
and the reconstructed geometric properties from custom_curvature_experiment results.

The mapping explores:
1. Input k → Edge weight variance
2. Input k → Reconstructed Ricci scalar
3. Input k → Angle deficits (Regge calculus)
4. Input k → Gromov delta (geometric distortion)
5. Input k → Mutual information patterns
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import glob
import re

def extract_curvature_from_filename(filename):
    """Extract curvature value from filename pattern like 'curv20'"""
    match = re.search(r'curv(\d+(?:\.\d+)?)', filename)
    if match:
        return float(match.group(1))
    return None

def load_experiment_data(filepath):
    """Load experiment data from JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_geometric_properties(data):
    """Extract geometric properties from experiment data"""
    if not data:
        return None
    
    spec = data.get('spec', {})
    input_curvature = spec.get('curvature', None)
    geometry = spec.get('geometry', 'unknown')
    
    # Extract edge weight variance if available
    edge_weight_variance = None
    if hasattr(data, '_edge_weight_variance'):
        edge_weight_variance = data._edge_weight_variance
    
    # Extract mutual information statistics
    mi_data = data.get('mutual_information_per_timestep', [])
    mi_stats = {}
    if mi_data:
        # Flatten all MI values across timesteps
        all_mi_values = []
        for timestep_mi in mi_data:
            if isinstance(timestep_mi, dict):
                all_mi_values.extend(timestep_mi.values())
        
        if all_mi_values:
            mi_stats = {
                'mean_mi': np.mean(all_mi_values),
                'std_mi': np.std(all_mi_values),
                'max_mi': np.max(all_mi_values),
                'min_mi': np.min(all_mi_values),
                'mi_range': np.max(all_mi_values) - np.min(all_mi_values),
                'mi_variance': np.var(all_mi_values)
            }
    
    # Extract distance matrix statistics
    distance_data = data.get('distance_matrix_per_timestep', [])
    distance_stats = {}
    if distance_data:
        # Use the first timestep for analysis
        first_distance_matrix = distance_data[0]
        if isinstance(first_distance_matrix, list) and len(first_distance_matrix) > 0:
            # Flatten the distance matrix
            distances = []
            for row in first_distance_matrix:
                if isinstance(row, list):
                    distances.extend([d for d in row if isinstance(d, (int, float)) and d > 0])
            
            if distances:
                distance_stats = {
                    'mean_distance': np.mean(distances),
                    'std_distance': np.std(distances),
                    'max_distance': np.max(distances),
                    'min_distance': np.min(distances),
                    'distance_range': np.max(distances) - np.min(distances),
                    'distance_variance': np.var(distances)
                }
    
    # Extract evolution summary if available
    evolution_summary = data.get('evolution_summary', {})
    gromov_delta_range = evolution_summary.get('gromov_delta_range', [])
    mean_distance_range = evolution_summary.get('mean_distance_range', [])
    
    return {
        'input_curvature': input_curvature,
        'geometry': geometry,
        'edge_weight_variance': edge_weight_variance,
        'mi_stats': mi_stats,
        'distance_stats': distance_stats,
        'gromov_delta_range': gromov_delta_range,
        'mean_distance_range': mean_distance_range
    }

def analyze_curvature_mapping():
    """Main analysis function"""
    # Find all experiment result files
    experiment_dir = Path("experiment_logs/custom_curvature_experiment")
    result_files = []
    
    # Search in older_results and recent instances
    for pattern in ["older_results/*.json", "instance_*/results_*.json"]:
        result_files.extend(experiment_dir.glob(pattern))
    
    print(f"Found {len(result_files)} experiment result files")
    
    # Extract data from all files
    all_data = []
    for filepath in result_files:
        # Extract curvature from filename as backup
        filename_curvature = extract_curvature_from_filename(filepath.name)
        
        data = load_experiment_data(filepath)
        if data:
            properties = extract_geometric_properties(data)
            if properties:
                # Use filename curvature if not found in data
                if properties['input_curvature'] is None:
                    properties['input_curvature'] = filename_curvature
                
                properties['filename'] = filepath.name
                all_data.append(properties)
    
    print(f"Successfully processed {len(all_data)} experiments")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_data)
    
    # Filter out experiments without curvature data
    df = df.dropna(subset=['input_curvature'])
    df = df[df['input_curvature'] > 0]
    
    print(f"Final dataset: {len(df)} experiments with valid curvature data")
    
    # Create mapping analysis
    create_curvature_mapping_plots(df)
    
    # Perform statistical analysis
    perform_statistical_analysis(df)
    
    return df

def create_curvature_mapping_plots(df):
    """Create plots showing the mapping between input curvature and geometric properties"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Mapping Between Input Curvature k and Reconstructed Geometric Properties', fontsize=16)
    
    # 1. MI Variance vs Input Curvature
    ax1 = axes[0, 0]
    valid_mi = df.dropna(subset=['mi_stats'])
    if len(valid_mi) > 0:
        mi_variances = []
        curvatures = []
        for idx, row in valid_mi.iterrows():
            if row['mi_stats'] and 'mi_variance' in row['mi_stats']:
                mi_variances.append(row['mi_stats']['mi_variance'])
                curvatures.append(row['input_curvature'])
        
        if mi_variances:
            ax1.scatter(curvatures, mi_variances, alpha=0.6)
            ax1.set_xlabel('Input Curvature k')
            ax1.set_ylabel('MI Variance')
            ax1.set_title('MI Variance vs Input Curvature')
            ax1.grid(True, alpha=0.3)
    
    # 2. MI Range vs Input Curvature
    ax2 = axes[0, 1]
    if len(valid_mi) > 0:
        mi_ranges = []
        curvatures = []
        for idx, row in valid_mi.iterrows():
            if row['mi_stats'] and 'mi_range' in row['mi_stats']:
                mi_ranges.append(row['mi_stats']['mi_range'])
                curvatures.append(row['input_curvature'])
        
        if mi_ranges:
            ax2.scatter(curvatures, mi_ranges, alpha=0.6)
            ax2.set_xlabel('Input Curvature k')
            ax2.set_ylabel('MI Range (max - min)')
            ax2.set_title('MI Range vs Input Curvature')
            ax2.grid(True, alpha=0.3)
    
    # 3. Distance Variance vs Input Curvature
    ax3 = axes[0, 2]
    valid_dist = df.dropna(subset=['distance_stats'])
    if len(valid_dist) > 0:
        dist_variances = []
        curvatures = []
        for idx, row in valid_dist.iterrows():
            if row['distance_stats'] and 'distance_variance' in row['distance_stats']:
                dist_variances.append(row['distance_stats']['distance_variance'])
                curvatures.append(row['input_curvature'])
        
        if dist_variances:
            ax3.scatter(curvatures, dist_variances, alpha=0.6)
            ax3.set_xlabel('Input Curvature k')
            ax3.set_ylabel('Distance Variance')
            ax3.set_title('Distance Variance vs Input Curvature')
            ax3.grid(True, alpha=0.3)
    
    # 4. Gromov Delta vs Input Curvature
    ax4 = axes[1, 0]
    valid_gromov = df[df['gromov_delta_range'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]
    if len(valid_gromov) > 0:
        gromov_max = [max(delta_range) for delta_range in valid_gromov['gromov_delta_range']]
        ax4.scatter(valid_gromov['input_curvature'], gromov_max, alpha=0.6)
        ax4.set_xlabel('Input Curvature k')
        ax4.set_ylabel('Max Gromov Delta')
        ax4.set_title('Geometric Distortion vs Input Curvature')
        ax4.grid(True, alpha=0.3)
    
    # 5. Mean Distance vs Input Curvature
    ax5 = axes[1, 1]
    if len(valid_dist) > 0:
        mean_distances = []
        curvatures = []
        for idx, row in valid_dist.iterrows():
            if row['distance_stats'] and 'mean_distance' in row['distance_stats']:
                mean_distances.append(row['distance_stats']['mean_distance'])
                curvatures.append(row['input_curvature'])
        
        if mean_distances:
            ax5.scatter(curvatures, mean_distances, alpha=0.6)
            ax5.set_xlabel('Input Curvature k')
            ax5.set_ylabel('Mean Distance')
            ax5.set_title('Mean Distance vs Input Curvature')
            ax5.grid(True, alpha=0.3)
    
    # 6. Curvature distribution
    ax6 = axes[1, 2]
    ax6.hist(df['input_curvature'], bins=20, alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Input Curvature k')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Distribution of Input Curvature Values')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis/curvature_mapping_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create detailed analysis plots
    create_detailed_analysis_plots(df)

def create_detailed_analysis_plots(df):
    """Create more detailed analysis plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Curvature Mapping Analysis', fontsize=16)
    
    # 1. Log-log plot of MI variance vs curvature
    ax1 = axes[0, 0]
    valid_mi = df.dropna(subset=['mi_stats'])
    if len(valid_mi) > 0:
        mi_variances = []
        curvatures = []
        for idx, row in valid_mi.iterrows():
            if row['mi_stats'] and 'mi_variance' in row['mi_stats']:
                mi_variances.append(row['mi_stats']['mi_variance'])
                curvatures.append(row['input_curvature'])
        
        # Filter out zero values for log plot
        valid_indices = [i for i, v in enumerate(mi_variances) if v > 0]
        if valid_indices:
            log_curv = np.log([curvatures[i] for i in valid_indices])
            log_mi_var = np.log([mi_variances[i] for i in valid_indices])
            
            ax1.scatter(log_curv, log_mi_var, alpha=0.6)
            
            # Fit power law
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_curv, log_mi_var)
                x_fit = np.linspace(min(log_curv), max(log_curv), 100)
                y_fit = slope * x_fit + intercept
                ax1.plot(x_fit, y_fit, 'r--', label=f'Power law: y ∝ x^{slope:.2f}\nR² = {r_value**2:.3f}')
                ax1.legend()
            except:
                pass
            
            ax1.set_xlabel('ln(Input Curvature k)')
            ax1.set_ylabel('ln(MI Variance)')
            ax1.set_title('Power Law Analysis: MI Variance vs Curvature')
            ax1.grid(True, alpha=0.3)
    
    # 2. Geometry comparison
    ax2 = axes[0, 1]
    geometry_counts = df['geometry'].value_counts()
    ax2.pie(geometry_counts.values, labels=geometry_counts.index, autopct='%1.1f%%')
    ax2.set_title('Distribution of Geometry Types')
    
    # 3. Curvature vs MI statistics by geometry
    ax3 = axes[1, 0]
    valid_mi = df.dropna(subset=['mi_stats'])
    if len(valid_mi) > 0:
        colors = {'hyperbolic': 'red', 'spherical': 'blue', 'euclidean': 'green'}
        for geometry in valid_mi['geometry'].unique():
            geom_data = valid_mi[valid_mi['geometry'] == geometry]
            if len(geom_data) > 0:
                mi_means = []
                geom_curvatures = []
                for idx, row in geom_data.iterrows():
                    if row['mi_stats'] and 'mean_mi' in row['mi_stats']:
                        mi_means.append(row['mi_stats']['mean_mi'])
                        geom_curvatures.append(row['input_curvature'])
                
                if mi_means:
                    color = colors.get(geometry, 'gray')
                    ax3.scatter(geom_curvatures, mi_means, 
                               alpha=0.6, label=geometry, color=color)
        
        ax3.set_xlabel('Input Curvature k')
        ax3.set_ylabel('Mean Mutual Information')
        ax3.set_title('Mean MI vs Curvature by Geometry')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Curvature threshold analysis
    ax4 = axes[1, 1]
    if len(valid_mi) > 0:
        mi_ranges = []
        curvatures = []
        for idx, row in valid_mi.iterrows():
            if row['mi_stats'] and 'mi_range' in row['mi_stats']:
                mi_ranges.append(row['mi_stats']['mi_range'])
                curvatures.append(row['input_curvature'])
        
        # Find threshold where MI range becomes significant
        threshold = np.percentile(mi_ranges, 75)  # 75th percentile as threshold
        high_mi = [i for i, r in enumerate(mi_ranges) if r > threshold]
        low_mi = [i for i, r in enumerate(mi_ranges) if r <= threshold]
        
        if high_mi and low_mi:
            ax4.scatter([curvatures[i] for i in low_mi], 
                       [mi_ranges[i] for i in low_mi], 
                       alpha=0.6, color='blue', label='Low MI Range')
            ax4.scatter([curvatures[i] for i in high_mi], 
                       [mi_ranges[i] for i in high_mi], 
                       alpha=0.6, color='red', label='High MI Range')
            
            # Find transition point
            high_curvatures = [curvatures[i] for i in high_mi]
            if high_curvatures:
                transition_curvature = min(high_curvatures)
                ax4.axvline(transition_curvature, color='green', linestyle='--', 
                           label=f'Transition at k={transition_curvature:.1f}')
            
            ax4.set_xlabel('Input Curvature k')
            ax4.set_ylabel('MI Range')
            ax4.set_title('Curvature Threshold Analysis')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis/detailed_curvature_mapping.png', dpi=300, bbox_inches='tight')
    plt.show()

def perform_statistical_analysis(df):
    """Perform statistical analysis of the curvature mapping"""
    
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS OF CURVATURE MAPPING")
    print("="*60)
    
    # Basic statistics
    print(f"\nDataset Summary:")
    print(f"Total experiments: {len(df)}")
    print(f"Curvature range: {df['input_curvature'].min():.1f} to {df['input_curvature'].max():.1f}")
    print(f"Mean curvature: {df['input_curvature'].mean():.2f}")
    print(f"Geometry distribution: {df['geometry'].value_counts().to_dict()}")
    
    # Analyze MI variance relationship
    valid_mi = df.dropna(subset=['mi_stats'])
    if len(valid_mi) > 0:
        mi_variances = []
        curvatures = []
        for idx, row in valid_mi.iterrows():
            if row['mi_stats'] and 'mi_variance' in row['mi_stats']:
                mi_variances.append(row['mi_stats']['mi_variance'])
                curvatures.append(row['input_curvature'])
        
        # Correlation analysis
        correlation, p_value = stats.pearsonr(curvatures, mi_variances)
        print(f"\nMI Variance vs Curvature Correlation:")
        print(f"Pearson correlation: {correlation:.3f}")
        print(f"P-value: {p_value:.3e}")
        
        # Power law fit
        try:
            valid_indices = [i for i, v in enumerate(mi_variances) if v > 0]
            if valid_indices:
                log_curv = np.log([curvatures[i] for i in valid_indices])
                log_mi_var = np.log([mi_variances[i] for i in valid_indices])
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_curv, log_mi_var)
                print(f"\nPower Law Analysis:")
                print(f"Exponent: {slope:.3f} ± {std_err:.3f}")
                print(f"R²: {r_value**2:.3f}")
                print(f"P-value: {p_value:.3e}")
        except Exception as e:
            print(f"Power law analysis failed: {e}")
    
    # Threshold analysis
    if len(valid_mi) > 0:
        mi_ranges = []
        curvatures = []
        for idx, row in valid_mi.iterrows():
            if row['mi_stats'] and 'mi_range' in row['mi_stats']:
                mi_ranges.append(row['mi_stats']['mi_range'])
                curvatures.append(row['input_curvature'])
        
        # Find transition point
        threshold = np.percentile(mi_ranges, 75)
        high_mi_indices = [i for i, r in enumerate(mi_ranges) if r > threshold]
        
        if high_mi_indices:
            transition_curvature = min([curvatures[i] for i in high_mi_indices])
            print(f"\nThreshold Analysis:")
            print(f"MI Range threshold (75th percentile): {threshold:.3f}")
            print(f"Transition curvature: k = {transition_curvature:.1f}")
            print(f"Experiments above threshold: {len(high_mi_indices)}/{len(valid_mi)}")
    
    # Geometry-specific analysis
    print(f"\nGeometry-Specific Analysis:")
    for geometry in df['geometry'].unique():
        geom_data = df[df['geometry'] == geometry]
        print(f"\n{geometry.upper()} geometry:")
        print(f"  Count: {len(geom_data)}")
        print(f"  Curvature range: {geom_data['input_curvature'].min():.1f} - {geom_data['input_curvature'].max():.1f}")
        print(f"  Mean curvature: {geom_data['input_curvature'].mean():.2f}")

def save_analysis_results(df):
    """Save analysis results to files"""
    
    # Save processed data
    df.to_csv('analysis/curvature_mapping_data.csv', index=False)
    
    # Save summary statistics
    summary_stats = {
        'total_experiments': len(df),
        'curvature_range': [df['input_curvature'].min(), df['input_curvature'].max()],
        'mean_curvature': df['input_curvature'].mean(),
        'geometry_distribution': df['geometry'].value_counts().to_dict()
    }
    
    with open('analysis/curvature_mapping_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"\nAnalysis results saved to:")
    print(f"- analysis/curvature_mapping_data.csv")
    print(f"- analysis/curvature_mapping_summary.json")
    print(f"- analysis/curvature_mapping_analysis.png")
    print(f"- analysis/detailed_curvature_mapping.png")

if __name__ == "__main__":
    # Create analysis directory if it doesn't exist
    Path("analysis").mkdir(exist_ok=True)
    
    # Run the analysis
    df = analyze_curvature_mapping()
    
    # Save results
    save_analysis_results(df)
    
    print("\nCurvature mapping analysis complete!") 