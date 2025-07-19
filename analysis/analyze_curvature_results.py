#!/usr/bin/env python3
"""
Analyze curvature results from real hardware data
"""

import json
import numpy as np
from scipy import stats

def analyze_curvature_results(json_file):
    """Analyze curvature results from JSON file"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print("=== CURVATURE ANALYSIS RESULTS ===")
    print(f"Experiment: {data['experiment_params']['num_qubits']} qubits, {data['experiment_params']['geometry']} geometry")
    print(f"Target curvature: {data['experiment_params']['curvature']}")
    print(f"Device: {data['experiment_params']['device']}")
    print()
    
    # Alpha estimates
    print("=== α ESTIMATES ===")
    alpha_ests = data['alpha_estimates']
    print(f"α from curvature (theoretical): {alpha_ests['from_curvature']:.6f}")
    print(f"α from distance matrix: {alpha_ests['from_distance']:.6f}")
    print(f"α from variance: {alpha_ests['from_variance']:.6f}")
    print(f"α from angle sums: {alpha_ests['from_angles']:.6f}")
    print(f"α final (weighted): {data['alpha_final']:.6f} ± {data['alpha_std_final']:.6f}")
    print()
    
    # Statistical significance
    print("=== STATISTICAL SIGNIFICANCE ===")
    pvals = data['p_values']
    print(f"Curvature significance: p = {pvals['curvature_significance']:.6f}")
    print(f"Distance exponential: p = {pvals['distance_exponential']:.6f}")
    print(f"α vs zero: p = {pvals['alpha_vs_zero']:.6f}")
    print(f"α vs theoretical: p = {pvals['alpha_vs_theoretical']:.6f}")
    print()
    
    # Interpretation
    print("=== CURVATURE DETECTION ===")
    
    # Check if curvature is significantly detected
    if pvals['curvature_significance'] < 0.05:
        print("✅ CURVATURE DETECTED: p < 0.05")
        print(f"   The data shows significant deviation from flat geometry")
    else:
        print("❌ NO CURVATURE DETECTED: p >= 0.05")
        print(f"   The data is consistent with flat geometry")
    
    # Check if distance exponentiality is significant
    if pvals['distance_exponential'] < 0.05:
        print("✅ EXPONENTIAL DISTANCE RELATION: p < 0.05")
        print(f"   The distance-weight relationship follows exponential scaling")
    else:
        print("❌ NO EXPONENTIAL DISTANCE RELATION: p >= 0.05")
    
    # Check if α is significantly different from zero
    if pvals['alpha_vs_zero'] < 0.05:
        print("✅ α SIGNIFICANTLY NON-ZERO: p < 0.05")
        print(f"   The emergent metric scaling parameter is meaningful")
    else:
        print("❌ α NOT SIGNIFICANTLY DIFFERENT FROM ZERO: p >= 0.05")
    
    # Check if α matches theoretical expectation
    theoretical_alpha = alpha_ests['from_curvature']
    if pvals['alpha_vs_theoretical'] < 0.05:
        print("✅ α SIGNIFICANTLY DIFFERENT FROM THEORETICAL: p < 0.05")
        print(f"   The measured α differs from theoretical expectation")
    else:
        print("✅ α CONSISTENT WITH THEORETICAL: p >= 0.05")
        print(f"   The measured α is consistent with theoretical prediction")
    
    print()
    print("=== PHYSICAL INTERPRETATION ===")
    
    # Physical interpretation
    if pvals['curvature_significance'] < 0.05:
        print("The quantum circuit successfully encoded hyperbolic curvature.")
        print("The measured angle sums deviate significantly from π (flat space).")
        print("This indicates the emergence of curved geometry in the quantum state.")
        
        if data['alpha_final'] > 0:
            print(f"The emergent metric scaling parameter α = {data['alpha_final']:.3f}")
            print("indicates how strongly the curvature affects the quantum geometry.")
        else:
            print("The negative α value suggests anti-correlation with curvature.")
    else:
        print("The quantum circuit did not produce significant curvature effects.")
        print("The angle sums are consistent with flat (Euclidean) geometry.")
    
    if pvals['distance_exponential'] < 0.05:
        print("The exponential distance-weight relationship confirms")
        print("that the quantum state encodes geometric structure.")
    
    return data

if __name__ == "__main__":
    analyze_curvature_results("alpha_analysis_7q_hyperbolic_curv0.5.json") 