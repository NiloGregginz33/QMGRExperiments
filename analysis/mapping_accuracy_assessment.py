#!/usr/bin/env python3
"""
Mapping Accuracy Assessment
==========================

This script assesses the accuracy of our curvature mapping by analyzing:
1. Statistical significance of correlations
2. Predictive power of the mapping
3. Consistency across different geometric properties
4. Error analysis and uncertainty quantification
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import ast
import warnings
warnings.filterwarnings('ignore')

def parse_dict_string(dict_str):
    """Parse dictionary string from CSV"""
    if pd.isna(dict_str) or dict_str == '{}':
        return {}
    try:
        # Convert numpy types to regular Python types
        dict_str = dict_str.replace('np.float64(', '').replace(')', '')
        return ast.literal_eval(dict_str)
    except:
        return {}

def load_and_clean_data():
    """Load and clean the curvature mapping data"""
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
    
    return df

def assess_correlation_accuracy(df):
    """Assess the accuracy of correlations between input curvature and geometric properties"""
    
    print("="*60)
    print("CORRELATION ACCURACY ASSESSMENT")
    print("="*60)
    
    # Filter valid data
    valid_data = df.dropna(subset=['input_curvature', 'mi_variance'])
    
    if len(valid_data) < 10:
        print("Insufficient data for correlation analysis")
        return {}
    
    results = {}
    
    # 1. MI Variance vs Curvature
    curvatures = valid_data['input_curvature']
    mi_variances = valid_data['mi_variance']
    
    # Remove zero/negative values for log analysis
    positive_mask = (mi_variances > 0) & (curvatures > 0)
    if positive_mask.sum() > 5:
        log_curv = np.log(curvatures[positive_mask])
        log_mi_var = np.log(mi_variances[positive_mask])
        
        # Linear correlation
        pearson_r, pearson_p = stats.pearsonr(curvatures, mi_variances)
        spearman_r, spearman_p = stats.spearmanr(curvatures, mi_variances)
        
        # Log-log correlation
        log_pearson_r, log_pearson_p = stats.pearsonr(log_curv, log_mi_var)
        
        results['mi_variance_correlation'] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'log_pearson_r': log_pearson_r,
            'log_pearson_p': log_pearson_p,
            'n_samples': len(valid_data),
            'positive_samples': positive_mask.sum()
        }
        
        print(f"\nMI Variance vs Curvature Correlation:")
        print(f"  Pearson r: {pearson_r:.3f} (p={pearson_p:.3e})")
        print(f"  Spearman r: {spearman_r:.3f} (p={spearman_p:.3e})")
        print(f"  Log-Pearson r: {log_pearson_r:.3f} (p={log_pearson_p:.3e})")
        print(f"  Sample size: {len(valid_data)}")
        print(f"  Positive samples: {positive_mask.sum()}")
        
        # Significance assessment
        if pearson_p < 0.05:
            print(f"  ✅ Statistically significant correlation (p < 0.05)")
        else:
            print(f"  ❌ No statistically significant correlation (p >= 0.05)")
    
    # 2. MI Range vs Curvature
    valid_range = df.dropna(subset=['input_curvature', 'mi_range'])
    if len(valid_range) > 5:
        range_pearson_r, range_pearson_p = stats.pearsonr(valid_range['input_curvature'], valid_range['mi_range'])
        results['mi_range_correlation'] = {
            'pearson_r': range_pearson_r,
            'pearson_p': range_pearson_p,
            'n_samples': len(valid_range)
        }
        
        print(f"\nMI Range vs Curvature Correlation:")
        print(f"  Pearson r: {range_pearson_r:.3f} (p={range_pearson_p:.3e})")
        print(f"  Sample size: {len(valid_range)}")
        
        if range_pearson_p < 0.05:
            print(f"  ✅ Statistically significant correlation")
        else:
            print(f"  ❌ No statistically significant correlation")
    
    # 3. Distance Variance vs Curvature
    valid_dist = df.dropna(subset=['input_curvature', 'distance_variance'])
    if len(valid_dist) > 5:
        dist_pearson_r, dist_pearson_p = stats.pearsonr(valid_dist['input_curvature'], valid_dist['distance_variance'])
        results['distance_variance_correlation'] = {
            'pearson_r': dist_pearson_r,
            'pearson_p': dist_pearson_p,
            'n_samples': len(valid_dist)
        }
        
        print(f"\nDistance Variance vs Curvature Correlation:")
        print(f"  Pearson r: {dist_pearson_r:.3f} (p={dist_pearson_p:.3e})")
        print(f"  Sample size: {len(valid_dist)}")
        
        if dist_pearson_p < 0.05:
            print(f"  ✅ Statistically significant correlation")
        else:
            print(f"  ❌ No statistically significant correlation")
    
    return results

def assess_predictive_power(df):
    """Assess the predictive power of the curvature mapping"""
    
    print("\n" + "="*60)
    print("PREDICTIVE POWER ASSESSMENT")
    print("="*60)
    
    results = {}
    
    # 1. Linear regression for MI variance prediction
    valid_data = df.dropna(subset=['input_curvature', 'mi_variance'])
    if len(valid_data) > 10:
        X = valid_data['input_curvature'].values.reshape(-1, 1)
        y = valid_data['mi_variance'].values
        
        # Split data for cross-validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        results['mi_variance_prediction'] = {
            'r2_score': r2,
            'mse': mse,
            'mae': mae,
            'slope': model.coef_[0],
            'intercept': model.intercept_,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
        
        print(f"\nMI Variance Prediction (Linear Regression):")
        print(f"  R² Score: {r2:.3f}")
        print(f"  Mean Squared Error: {mse:.6f}")
        print(f"  Mean Absolute Error: {mae:.6f}")
        print(f"  Slope: {model.coef_[0]:.6f}")
        print(f"  Intercept: {model.intercept_:.6f}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        
        if r2 > 0.5:
            print(f"  ✅ Good predictive power (R² > 0.5)")
        elif r2 > 0.1:
            print(f"  ⚠️  Moderate predictive power (0.1 < R² < 0.5)")
        else:
            print(f"  ❌ Poor predictive power (R² < 0.1)")
    
    # 2. Power law fitting
    positive_mask = (valid_data['mi_variance'] > 0) & (valid_data['input_curvature'] > 0)
    if positive_mask.sum() > 5:
        log_curv = np.log(valid_data.loc[positive_mask, 'input_curvature'])
        log_mi_var = np.log(valid_data.loc[positive_mask, 'mi_variance'])
        
        # Power law fit: y = ax^b
        try:
            popt, pcov = curve_fit(lambda x, a, b: a * np.exp(b * x), log_curv, log_mi_var)
            a, b = popt
            a_err, b_err = np.sqrt(np.diag(pcov))
            
            # Predictions
            y_pred_power = a * np.exp(b * log_curv)
            r2_power = r2_score(log_mi_var, y_pred_power)
            
            results['power_law_fit'] = {
                'exponent': b,
                'exponent_error': b_err,
                'amplitude': a,
                'amplitude_error': a_err,
                'r2_score': r2_power,
                'n_samples': positive_mask.sum()
            }
            
            print(f"\nPower Law Fit (y = ax^b):")
            print(f"  Exponent b: {b:.3f} ± {b_err:.3f}")
            print(f"  Amplitude a: {a:.6f} ± {a_err:.6f}")
            print(f"  R² Score: {r2_power:.3f}")
            print(f"  Sample size: {positive_mask.sum()}")
            
            if r2_power > 0.5:
                print(f"  ✅ Good power law fit (R² > 0.5)")
            elif r2_power > 0.1:
                print(f"  ⚠️  Moderate power law fit (0.1 < R² < 0.5)")
            else:
                print(f"  ❌ Poor power law fit (R² < 0.1)")
                
        except Exception as e:
            print(f"\nPower law fitting failed: {e}")
    
    return results

def assess_consistency(df):
    """Assess consistency across different geometric properties"""
    
    print("\n" + "="*60)
    print("CONSISTENCY ASSESSMENT")
    print("="*60)
    
    results = {}
    
    # 1. Geometry-specific analysis
    geometries = df['geometry'].unique()
    
    for geometry in geometries:
        geom_data = df[df['geometry'] == geometry]
        valid_geom = geom_data.dropna(subset=['input_curvature', 'mi_variance'])
        
        if len(valid_geom) > 3:
            r, p = stats.pearsonr(valid_geom['input_curvature'], valid_geom['mi_variance'])
            
            results[f'{geometry}_correlation'] = {
                'pearson_r': r,
                'pearson_p': p,
                'n_samples': len(valid_geom),
                'curvature_range': [valid_geom['input_curvature'].min(), valid_geom['input_curvature'].max()]
            }
            
            print(f"\n{geometry.upper()} Geometry:")
            print(f"  Pearson r: {r:.3f} (p={p:.3e})")
            print(f"  Sample size: {len(valid_geom)}")
            print(f"  Curvature range: {valid_geom['input_curvature'].min():.1f} - {valid_geom['input_curvature'].max():.1f}")
            
            if p < 0.05:
                print(f"  ✅ Consistent correlation")
            else:
                print(f"  ❌ Inconsistent correlation")
    
    # 2. Curvature regime analysis
    low_curv = df[df['input_curvature'] < 3.0].dropna(subset=['mi_variance'])
    med_curv = df[(df['input_curvature'] >= 3.0) & (df['input_curvature'] < 10.0)].dropna(subset=['mi_variance'])
    high_curv = df[df['input_curvature'] >= 10.0].dropna(subset=['mi_variance'])
    
    regimes = [
        ('Low Curvature (k < 3)', low_curv),
        ('Medium Curvature (3 ≤ k < 10)', med_curv),
        ('High Curvature (k ≥ 10)', high_curv)
    ]
    
    for regime_name, regime_data in regimes:
        if len(regime_data) > 3:
            mean_mi_var = regime_data['mi_variance'].mean()
            std_mi_var = regime_data['mi_variance'].std()
            
            results[f'{regime_name}_stats'] = {
                'mean_mi_variance': mean_mi_var,
                'std_mi_variance': std_mi_var,
                'n_samples': len(regime_data)
            }
            
            print(f"\n{regime_name}:")
            print(f"  Mean MI variance: {mean_mi_var:.6f}")
            print(f"  Std MI variance: {std_mi_var:.6f}")
            print(f"  Sample size: {len(regime_data)}")
    
    return results

def assess_uncertainty(df):
    """Assess uncertainty and error in the mapping"""
    
    print("\n" + "="*60)
    print("UNCERTAINTY ASSESSMENT")
    print("="*60)
    
    results = {}
    
    # 1. Data quality assessment
    total_experiments = len(df)
    valid_mi = df.dropna(subset=['mi_variance']).shape[0]
    valid_dist = df.dropna(subset=['distance_variance']).shape[0]
    
    data_quality = {
        'total_experiments': total_experiments,
        'valid_mi_data': valid_mi,
        'valid_distance_data': valid_dist,
        'mi_data_completeness': valid_mi / total_experiments,
        'distance_data_completeness': valid_dist / total_experiments
    }
    
    results['data_quality'] = data_quality
    
    print(f"\nData Quality:")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Valid MI data: {valid_mi} ({valid_mi/total_experiments*100:.1f}%)")
    print(f"  Valid distance data: {valid_dist} ({valid_dist/total_experiments*100:.1f}%)")
    
    # 2. Measurement uncertainty
    valid_data = df.dropna(subset=['mi_variance'])
    if len(valid_data) > 0:
        # Coefficient of variation
        cv = valid_data['mi_variance'].std() / valid_data['mi_variance'].mean()
        
        # Bootstrap confidence intervals
        bootstrap_means = []
        for _ in range(1000):
            sample = valid_data['mi_variance'].sample(n=len(valid_data), replace=True)
            bootstrap_means.append(sample.mean())
        
        ci_95 = np.percentile(bootstrap_means, [2.5, 97.5])
        
        uncertainty_stats = {
            'coefficient_of_variation': cv,
            'bootstrap_ci_95': ci_95.tolist(),
            'mean_mi_variance': valid_data['mi_variance'].mean(),
            'std_mi_variance': valid_data['mi_variance'].std()
        }
        
        results['measurement_uncertainty'] = uncertainty_stats
        
        print(f"\nMeasurement Uncertainty:")
        print(f"  Coefficient of variation: {cv:.3f}")
        print(f"  Mean MI variance: {valid_data['mi_variance'].mean():.6f}")
        print(f"  Std MI variance: {valid_data['mi_variance'].std():.6f}")
        print(f"  95% CI: [{ci_95[0]:.6f}, {ci_95[1]:.6f}]")
    
    return results

def create_accuracy_summary(correlation_results, predictive_results, consistency_results, uncertainty_results):
    """Create a comprehensive accuracy summary"""
    
    print("\n" + "="*60)
    print("COMPREHENSIVE ACCURACY SUMMARY")
    print("="*60)
    
    # Overall accuracy score
    accuracy_metrics = []
    
    # 1. Correlation strength
    if 'mi_variance_correlation' in correlation_results:
        corr = correlation_results['mi_variance_correlation']
        if corr['pearson_p'] < 0.05:
            accuracy_metrics.append(min(abs(corr['pearson_r']), 1.0))
        else:
            accuracy_metrics.append(0.0)
    
    # 2. Predictive power
    if 'mi_variance_prediction' in predictive_results:
        pred = predictive_results['mi_variance_prediction']
        accuracy_metrics.append(max(pred['r2_score'], 0.0))
    
    # 3. Data quality
    if 'data_quality' in uncertainty_results:
        quality = uncertainty_results['data_quality']
        accuracy_metrics.append(quality['mi_data_completeness'])
    
    # 4. Consistency across geometries
    geometry_correlations = [v for k, v in consistency_results.items() if 'correlation' in k]
    if geometry_correlations:
        significant_correlations = sum(1 for corr in geometry_correlations if corr['pearson_p'] < 0.05)
        consistency_score = significant_correlations / len(geometry_correlations)
        accuracy_metrics.append(consistency_score)
    
    # Calculate overall accuracy
    if accuracy_metrics:
        overall_accuracy = np.mean(accuracy_metrics)
        
        print(f"\nOverall Mapping Accuracy: {overall_accuracy:.1%}")
        print(f"  Correlation strength: {accuracy_metrics[0]:.1%}")
        if len(accuracy_metrics) > 1:
            print(f"  Predictive power: {accuracy_metrics[1]:.1%}")
        if len(accuracy_metrics) > 2:
            print(f"  Data quality: {accuracy_metrics[2]:.1%}")
        if len(accuracy_metrics) > 3:
            print(f"  Consistency: {accuracy_metrics[3]:.1%}")
        
        # Accuracy assessment
        if overall_accuracy >= 0.7:
            print(f"\n✅ HIGH ACCURACY: The mapping shows strong predictive power and consistency")
        elif overall_accuracy >= 0.4:
            print(f"\n⚠️  MODERATE ACCURACY: The mapping shows some predictive power but with limitations")
        else:
            print(f"\n❌ LOW ACCURACY: The mapping has limited predictive power and consistency")
    
    # Save comprehensive results
    all_results = {
        'correlation_analysis': correlation_results,
        'predictive_analysis': predictive_results,
        'consistency_analysis': consistency_results,
        'uncertainty_analysis': uncertainty_results,
        'overall_accuracy': overall_accuracy if accuracy_metrics else None
    }
    
    with open('analysis/mapping_accuracy_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: analysis/mapping_accuracy_results.json")
    
    return all_results

def main():
    """Main analysis function"""
    
    print("MAPPING ACCURACY ASSESSMENT")
    print("="*60)
    
    # Load data
    df = load_and_clean_data()
    print(f"Loaded {len(df)} experiments")
    
    # Perform assessments
    correlation_results = assess_correlation_accuracy(df)
    predictive_results = assess_predictive_power(df)
    consistency_results = assess_consistency(df)
    uncertainty_results = assess_uncertainty(df)
    
    # Create summary
    all_results = create_accuracy_summary(
        correlation_results, 
        predictive_results, 
        consistency_results, 
        uncertainty_results
    )
    
    print("\nMapping accuracy assessment complete!")

if __name__ == "__main__":
    main() 