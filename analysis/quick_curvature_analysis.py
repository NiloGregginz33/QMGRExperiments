#!/usr/bin/env python3
"""
Quick Curvature Analysis - Analyze specific experiment files with known analysis results
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

def analyze_specific_files():
    """Analyze specific experiment files that we know have analysis results."""
    
    # Files we know have analysis results
    files_to_analyze = [
        "experiment_logs/custom_curvature_experiment/instance_20250730_231421/results_n4_geomS_curv10_ibm_brisbane_QK2NTB.json",
        "experiment_logs/custom_curvature_experiment/instance_20250730_231845/results_n10_geomS_curv10_ibm_brisbane_89S63U.json",
        "experiment_logs/custom_curvature_experiment/instance_20250730_230624/results_n4_geomS_curv10_ibm_brisbane_OJWM4J.json",
        "experiment_logs/custom_curvature_experiment/instance_20250731_005039/results_n7_geomS_curv2_ibm_brisbane_5PVWS5.json",
        "experiment_logs/custom_curvature_experiment/instance_20250731_012542/results_n12_geomS_curv2_ibm_brisbane_SNDD23.json"
    ]
    
    results = []
    
    print("🔍 Analyzing specific experiment files...")
    
    for i, file_path in enumerate(files_to_analyze, 1):
        if not os.path.exists(file_path):
            print(f"[{i}/{len(files_to_analyze)}] ❌ File not found: {file_path}")
            continue
            
        print(f"[{i}/{len(files_to_analyze)}] Analyzing {os.path.basename(file_path)}...")
        
        try:
            # Read the experiment data
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract basic parameters
            spec = data.get('spec', {})
            num_qubits = spec.get('num_qubits', 0)
            curvature = spec.get('curvature', 0.0)
            geometry = spec.get('geometry', 'unknown')
            
            # Check if analysis results exist
            output_dir = os.path.dirname(file_path)
            
            # Look for extraordinary evidence results
            ee_results_file = os.path.join(output_dir, 'extraordinary_evidence_results.json')
            qes_results_file = os.path.join(output_dir, 'quantum_emergent_spacetime_results.json')
            
            ee_results = None
            qes_results = None
            
            if os.path.exists(ee_results_file):
                with open(ee_results_file, 'r') as f:
                    ee_results = json.load(f)
                print(f"  ✅ Found extraordinary evidence results")
            
            if os.path.exists(qes_results_file):
                with open(qes_results_file, 'r') as f:
                    qes_results = json.load(f)
                print(f"  ✅ Found quantum emergent spacetime results")
            
            # Extract metrics
            analysis_result = {
                'file_path': file_path,
                'num_qubits': num_qubits,
                'curvature': curvature,
                'geometry': geometry,
                'extraordinary_evidence_score': ee_results.get('overall_assessment', {}).get('extraordinary_evidence_score', 0) if ee_results else 0,
                'conditions_met': ee_results.get('overall_assessment', {}).get('conditions_met', 0) if ee_results else 0,
                'quantum_spacetime_score': qes_results.get('overall_assessment', {}).get('quantum_emergent_spacetime_score', 0) if qes_results else 0,
                'features_detected': qes_results.get('overall_assessment', {}).get('features_detected', 0) if qes_results else 0,
                'page_curve_detected': ee_results.get('condition_2_page_curve', {}).get('page_curve_detected', False) if ee_results else False,
                'entanglement_curvature_r2': ee_results.get('condition_3_entanglement_curvature', {}).get('max_r2', 0) if ee_results else 0,
                'causal_violations': len(ee_results.get('condition_4_causal_violations', {}).get('causal_violations', [])) if ee_results else 0,
                'power_law_r2': ee_results.get('condition_5_cross_system_consistency', {}).get('power_law_r2', 0) if ee_results else 0,
                'reproducibility_score': ee_results.get('condition_6_reproducibility', {}).get('overall_score', 0) if ee_results else 0,
                'time_asymmetry': qes_results.get('charge_injection_analysis', {}).get('total_time_asymmetry', 0.0) if qes_results else 0.0,
                'asymmetry_measure': qes_results.get('charge_injection_analysis', {}).get('asymmetry_measure', 0.0) if qes_results else 0.0,
                'rt_geometry_consistent': qes_results.get('rt_geometry_analysis', {}).get('enhanced_area_law_consistent', False) if qes_results else False
            }
            
            results.append(analysis_result)
            print(f"  ✅ {num_qubits} qubits, curvature {curvature}, QES Score: {analysis_result['quantum_spacetime_score']:.3f}, EE Score: {analysis_result['extraordinary_evidence_score']:.3f}")
            
        except Exception as e:
            print(f"  ❌ Error analyzing {file_path}: {e}")
    
    return results

def generate_quick_summary(results):
    """Generate a quick summary of the analysis."""
    if not results:
        print("❌ No results to analyze!")
        return
    
    print(f"\n📊 Quick Analysis Summary")
    print(f"=========================")
    print(f"Total experiments analyzed: {len(results)}")
    
    # Extract data
    curvatures = [r['curvature'] for r in results]
    qubits = [r['num_qubits'] for r in results]
    ee_scores = [r['extraordinary_evidence_score'] for r in results]
    qes_scores = [r['quantum_spacetime_score'] for r in results]
    
    print(f"\n📈 Key Metrics:")
    print(f"Curvature range: {min(curvatures):.1f} - {max(curvatures):.1f}")
    print(f"Qubit range: {min(qubits)} - {max(qubits)}")
    print(f"Best Quantum Spacetime Score: {max(qes_scores):.3f}")
    print(f"Best Extraordinary Evidence Score: {max(ee_scores):.3f}")
    
    # Find best performing experiment
    best_qes_idx = np.argmax(qes_scores)
    best_ee_idx = np.argmax(ee_scores)
    
    print(f"\n🏆 Best Quantum Spacetime Experiment:")
    best_qes = results[best_qes_idx]
    print(f"  File: {os.path.basename(best_qes['file_path'])}")
    print(f"  Qubits: {best_qes['num_qubits']}")
    print(f"  Curvature: {best_qes['curvature']}")
    print(f"  QES Score: {best_qes['quantum_spacetime_score']:.3f}")
    print(f"  EE Score: {best_qes['extraordinary_evidence_score']:.3f}")
    print(f"  Conditions Met: {best_qes['conditions_met']}/6")
    
    print(f"\n🏆 Best Extraordinary Evidence Experiment:")
    best_ee = results[best_ee_idx]
    print(f"  File: {os.path.basename(best_ee['file_path'])}")
    print(f"  Qubits: {best_ee['num_qubits']}")
    print(f"  Curvature: {best_ee['curvature']}")
    print(f"  QES Score: {best_ee['quantum_spacetime_score']:.3f}")
    print(f"  EE Score: {best_ee['extraordinary_evidence_score']:.3f}")
    print(f"  Conditions Met: {best_ee['conditions_met']}/6")
    
    # Calculate correlations
    from scipy.stats import pearsonr
    
    print(f"\n📊 Correlations:")
    try:
        corr, p_val = pearsonr(curvatures, qes_scores)
        print(f"  Curvature vs QES Score: {corr:.3f} (p={p_val:.3f})")
    except:
        print(f"  Curvature vs QES Score: Could not calculate")
    
    try:
        corr, p_val = pearsonr(qubits, qes_scores)
        print(f"  Qubits vs QES Score: {corr:.3f} (p={p_val:.3f})")
    except:
        print(f"  Qubits vs QES Score: Could not calculate")
    
    try:
        corr, p_val = pearsonr(curvatures, ee_scores)
        print(f"  Curvature vs EE Score: {corr:.3f} (p={p_val:.3f})")
    except:
        print(f"  Curvature vs EE Score: Could not calculate")
    
    try:
        corr, p_val = pearsonr(qubits, ee_scores)
        print(f"  Qubits vs EE Score: {corr:.3f} (p={p_val:.3f})")
    except:
        print(f"  Qubits vs EE Score: Could not calculate")
    
    # Group by curvature
    print(f"\n📋 Results by Curvature:")
    curvature_groups = {}
    for result in results:
        curvature = result['curvature']
        if curvature not in curvature_groups:
            curvature_groups[curvature] = []
        curvature_groups[curvature].append(result)
    
    for curvature in sorted(curvature_groups.keys()):
        group = curvature_groups[curvature]
        avg_qes = np.mean([r['quantum_spacetime_score'] for r in group])
        avg_ee = np.mean([r['extraordinary_evidence_score'] for r in group])
        print(f"  Curvature {curvature}: {len(group)} experiments, Avg QES: {avg_qes:.3f}, Avg EE: {avg_ee:.3f}")
    
    # Group by qubit count
    print(f"\n📋 Results by Qubit Count:")
    qubit_groups = {}
    for result in results:
        qubits = result['num_qubits']
        if qubits not in qubit_groups:
            qubit_groups[qubits] = []
        qubit_groups[qubits].append(result)
    
    for qubits in sorted(qubit_groups.keys()):
        group = qubit_groups[qubits]
        avg_qes = np.mean([r['quantum_spacetime_score'] for r in group])
        avg_ee = np.mean([r['extraordinary_evidence_score'] for r in group])
        print(f"  {qubits} qubits: {len(group)} experiments, Avg QES: {avg_qes:.3f}, Avg EE: {avg_ee:.3f}")

def main():
    """Main function."""
    print("🚀 Quick Curvature Analysis")
    print("=" * 40)
    
    results = analyze_specific_files()
    
    if results:
        generate_quick_summary(results)
        
        # Save results
        output_dir = "experiment_logs/curvature_system_size_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{output_dir}/quick_curvature_analysis_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'results': results,
                'timestamp': timestamp,
                'summary': {
                    'total_experiments': len(results),
                    'curvature_range': [min([r['curvature'] for r in results]), max([r['curvature'] for r in results])],
                    'qubit_range': [min([r['num_qubits'] for r in results]), max([r['num_qubits'] for r in results])],
                    'best_qes_score': max([r['quantum_spacetime_score'] for r in results]),
                    'best_ee_score': max([r['extraordinary_evidence_score'] for r in results])
                }
            }, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to {results_file}")
    else:
        print("❌ No results to analyze!")

if __name__ == "__main__":
    main() 