#!/usr/bin/env python3
"""
Curvature and System Size Analysis for Quantum Emergent Spacetime
Analyzes individual experiments to quantify the relationship between curvature, qubit count, and quantum signatures.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def analyze_single_experiment(file_path: str) -> Optional[Dict]:
    """Analyze a single experiment file and return key metrics."""
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
        
        if os.path.exists(qes_results_file):
            with open(qes_results_file, 'r') as f:
                qes_results = json.load(f)
        
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
        
        return analysis_result
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None

def find_experiment_files(base_dir: str = "experiment_logs/custom_curvature_experiment") -> List[str]:
    """Find all experiment result files in the custom curvature experiment directory."""
    experiment_files = []
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found")
        return experiment_files
    
    for instance_dir in os.listdir(base_dir):
        instance_path = os.path.join(base_dir, instance_dir)
        if not os.path.isdir(instance_path):
            continue
        
        # Look for result files
        for file in os.listdir(instance_path):
            if file.startswith('results_') and file.endswith('.json'):
                file_path = os.path.join(instance_path, file)
                experiment_files.append(file_path)
    
    return experiment_files

def analyze_experiments(experiment_files: List[str]) -> List[Dict]:
    """Analyze all experiment files and return results."""
    results = []
    
    print(f"🔍 Analyzing {len(experiment_files)} experiments...")
    
    for i, file_path in enumerate(experiment_files, 1):
        print(f"[{i}/{len(experiment_files)}] Analyzing {os.path.basename(file_path)}...")
        result = analyze_single_experiment(file_path)
        if result:
            results.append(result)
            print(f"  ✅ {result['num_qubits']} qubits, curvature {result['curvature']}, QES Score: {result['quantum_spacetime_score']:.3f}")
        else:
            print(f"  ❌ Failed to analyze")
    
    return results

def perform_statistical_analysis(results: List[Dict]) -> Dict:
    """Perform statistical analysis of the relationships."""
    if not results:
        return {}
    
    # Extract data
    curvatures = [r['curvature'] for r in results]
    qubits = [r['num_qubits'] for r in results]
    ee_scores = [r['extraordinary_evidence_score'] for r in results]
    qes_scores = [r['quantum_spacetime_score'] for r in results]
    entanglement_r2 = [r['entanglement_curvature_r2'] for r in results]
    time_asymmetry = [r['time_asymmetry'] for r in results]
    
    # Calculate correlations
    from scipy.stats import pearsonr, spearmanr
    
    correlations = {}
    
    # Curvature correlations
    try:
        corr, p_val = pearsonr(curvatures, ee_scores)
        correlations['curvature_ee_pearson'] = {'correlation': corr, 'p_value': p_val}
    except:
        correlations['curvature_ee_pearson'] = {'correlation': 0, 'p_value': 1}
        
    try:
        corr, p_val = pearsonr(curvatures, qes_scores)
        correlations['curvature_qes_pearson'] = {'correlation': corr, 'p_value': p_val}
    except:
        correlations['curvature_qes_pearson'] = {'correlation': 0, 'p_value': 1}
        
    try:
        corr, p_val = pearsonr(curvatures, entanglement_r2)
        correlations['curvature_entanglement_pearson'] = {'correlation': corr, 'p_value': p_val}
    except:
        correlations['curvature_entanglement_pearson'] = {'correlation': 0, 'p_value': 1}
    
    # System size correlations
    try:
        corr, p_val = pearsonr(qubits, ee_scores)
        correlations['qubits_ee_pearson'] = {'correlation': corr, 'p_value': p_val}
    except:
        correlations['qubits_ee_pearson'] = {'correlation': 0, 'p_value': 1}
        
    try:
        corr, p_val = pearsonr(qubits, qes_scores)
        correlations['qubits_qes_pearson'] = {'correlation': corr, 'p_value': p_val}
    except:
        correlations['qubits_qes_pearson'] = {'correlation': 0, 'p_value': 1}
    
    # Combined effect (curvature * qubits)
    curvature_qubit_product = [c * q for c, q in zip(curvatures, qubits)]
    try:
        corr, p_val = pearsonr(curvature_qubit_product, qes_scores)
        correlations['curvature_qubit_product_qes_pearson'] = {'correlation': corr, 'p_value': p_val}
    except:
        correlations['curvature_qubit_product_qes_pearson'] = {'correlation': 0, 'p_value': 1}
    
    # Group by curvature and system size
    curvature_groups = {}
    qubit_groups = {}
    
    for result in results:
        curvature = result['curvature']
        num_qubits = result['num_qubits']
        
        if curvature not in curvature_groups:
            curvature_groups[curvature] = []
        curvature_groups[curvature].append(result['quantum_spacetime_score'])
        
        if num_qubits not in qubit_groups:
            qubit_groups[num_qubits] = []
        qubit_groups[num_qubits].append(result['quantum_spacetime_score'])
    
    # Calculate group statistics
    curvature_stats = {k: {'mean': np.mean(v), 'std': np.std(v), 'count': len(v)} 
                      for k, v in curvature_groups.items()}
    qubit_stats = {k: {'mean': np.mean(v), 'std': np.std(v), 'count': len(v)} 
                  for k, v in qubit_groups.items()}
    
    return {
        'correlations': correlations,
        'curvature_stats': curvature_stats,
        'qubit_stats': qubit_stats,
        'total_experiments': len(results),
        'curvature_range': [min(curvatures), max(curvatures)] if curvatures else [0, 0],
        'qubit_range': [min(qubits), max(qubits)] if qubits else [0, 0],
        'best_quantum_score': max(qes_scores) if qes_scores else 0,
        'best_curvature': curvatures[np.argmax(qes_scores)] if qes_scores and len(curvatures) > 0 else 0,
        'best_qubits': qubits[np.argmax(qes_scores)] if qes_scores and len(qubits) > 0 else 0
    }

def generate_visualizations(results: List[Dict], analysis_summary: Dict):
    """Generate comprehensive visualizations."""
    if not results:
        return
        
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Curvature and System Size Analysis for Quantum Emergent Spacetime', fontsize=16, fontweight='bold')
    
    # Extract data
    curvatures = [r['curvature'] for r in results]
    qubits = [r['num_qubits'] for r in results]
    ee_scores = [r['extraordinary_evidence_score'] for r in results]
    qes_scores = [r['quantum_spacetime_score'] for r in results]
    entanglement_r2 = [r['entanglement_curvature_r2'] for r in results]
    time_asymmetry = [r['time_asymmetry'] for r in results]
    
    # 1. Curvature vs Quantum Spacetime Score
    axes[0, 0].scatter(curvatures, qes_scores, alpha=0.7, s=100)
    axes[0, 0].set_xlabel('Curvature')
    axes[0, 0].set_ylabel('Quantum Emergent Spacetime Score')
    axes[0, 0].set_title('Curvature vs Quantum Spacetime Score')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. System Size vs Quantum Spacetime Score
    axes[0, 1].scatter(qubits, qes_scores, alpha=0.7, s=100)
    axes[0, 1].set_xlabel('Number of Qubits')
    axes[0, 1].set_ylabel('Quantum Emergent Spacetime Score')
    axes[0, 1].set_title('System Size vs Quantum Spacetime Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Curvature vs Entanglement-Curvature Correlation
    axes[0, 2].scatter(curvatures, entanglement_r2, alpha=0.7, s=100)
    axes[0, 2].set_xlabel('Curvature')
    axes[0, 2].set_ylabel('Entanglement-Curvature R²')
    axes[0, 2].set_title('Curvature vs Entanglement-Curvature Correlation')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 2D Scatter: Curvature vs Qubits, colored by score
    scatter = axes[1, 0].scatter(curvatures, qubits, c=qes_scores, cmap='viridis', s=100, alpha=0.8)
    axes[1, 0].set_xlabel('Curvature')
    axes[1, 0].set_ylabel('Number of Qubits')
    axes[1, 0].set_title('Curvature vs System Size (colored by QES Score)')
    plt.colorbar(scatter, ax=axes[1, 0], label='Quantum Spacetime Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Time Asymmetry vs Curvature
    axes[1, 1].scatter(curvatures, time_asymmetry, alpha=0.7, s=100)
    axes[1, 1].set_xlabel('Curvature')
    axes[1, 1].set_ylabel('Time Asymmetry')
    axes[1, 1].set_title('Curvature vs Time Asymmetry')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Extraordinary Evidence vs Quantum Spacetime Score
    axes[1, 2].scatter(ee_scores, qes_scores, alpha=0.7, s=100)
    axes[1, 2].set_xlabel('Extraordinary Evidence Score')
    axes[1, 2].set_ylabel('Quantum Emergent Spacetime Score')
    axes[1, 2].set_title('Extraordinary Evidence vs Quantum Spacetime')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = "experiment_logs/curvature_system_size_analysis"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/curvature_system_size_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizations saved to {output_dir}/curvature_system_size_analysis.png")

def save_results(results: List[Dict], analysis_summary: Dict):
    """Save analysis results to files."""
    output_dir = "experiment_logs/curvature_system_size_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_file = f"{output_dir}/curvature_system_size_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'analysis_summary': analysis_summary,
            'detailed_results': results,
            'timestamp': timestamp
        }, f, indent=2, default=str)
    
    # Generate summary report
    summary_file = f"{output_dir}/curvature_system_size_summary_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write(generate_summary_report(analysis_summary, results))
    
    print(f"💾 Results saved to {output_dir}")
    print(f"📄 Detailed results: {os.path.basename(results_file)}")
    print(f"📝 Summary: {os.path.basename(summary_file)}")

def generate_summary_report(analysis_summary: Dict, results: List[Dict]) -> str:
    """Generate a human-readable summary report."""
    correlations = analysis_summary.get('correlations', {})
    
    report = f"""# Curvature and System Size Analysis Summary

## Analysis Overview
- Total experiments analyzed: {analysis_summary.get('total_experiments', 0)}
- Curvature range: {analysis_summary.get('curvature_range', [0, 0])}
- Qubit range: {analysis_summary.get('qubit_range', [0, 0])}

## Key Findings

### Best Performing Configuration
- Best Quantum Spacetime Score: {analysis_summary.get('best_quantum_score', 0):.3f}
- Optimal Curvature: {analysis_summary.get('best_curvature', 0):.1f}
- Optimal Qubit Count: {analysis_summary.get('best_qubits', 0)}

### Statistical Correlations

#### Curvature Effects
- Curvature vs Quantum Spacetime Score: {correlations.get('curvature_qes_pearson', {}).get('correlation', 0):.3f} (p={correlations.get('curvature_qes_pearson', {}).get('p_value', 1):.3f})
- Curvature vs Entanglement-Curvature R²: {correlations.get('curvature_entanglement_pearson', {}).get('correlation', 0):.3f} (p={correlations.get('curvature_entanglement_pearson', {}).get('p_value', 1):.3f})
- Curvature vs Extraordinary Evidence: {correlations.get('curvature_ee_pearson', {}).get('correlation', 0):.3f} (p={correlations.get('curvature_ee_pearson', {}).get('p_value', 1):.3f})

#### System Size Effects
- Qubit Count vs Quantum Spacetime Score: {correlations.get('qubits_qes_pearson', {}).get('correlation', 0):.3f} (p={correlations.get('qubits_qes_pearson', {}).get('p_value', 1):.3f})
- Qubit Count vs Extraordinary Evidence: {correlations.get('qubits_ee_pearson', {}).get('correlation', 0):.3f} (p={correlations.get('qubits_ee_pearson', {}).get('p_value', 1):.3f})

#### Combined Effects
- Curvature × Qubits vs Quantum Spacetime: {correlations.get('curvature_qubit_product_qes_pearson', {}).get('correlation', 0):.3f} (p={correlations.get('curvature_qubit_product_qes_pearson', {}).get('p_value', 1):.3f})

## Curvature Group Statistics
"""
    
    for curvature, stats in analysis_summary.get('curvature_stats', {}).items():
        report += f"- Curvature {curvature}: Mean QES Score = {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})\n"
    
    report += "\n## Qubit Count Group Statistics\n"
    for qubits, stats in analysis_summary.get('qubit_stats', {}).items():
        report += f"- {qubits} qubits: Mean QES Score = {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})\n"
    
    report += "\n## Individual Experiment Results\n"
    for result in sorted(results, key=lambda x: (x['curvature'], x['num_qubits'])):
        report += f"- {result['num_qubits']} qubits, curvature {result['curvature']}: QES Score = {result['quantum_spacetime_score']:.3f}, EE Score = {result['extraordinary_evidence_score']:.3f}\n"
    
    report += f"""

## Conclusions

### Strongest Correlations
1. {'Curvature' if abs(correlations.get('curvature_qes_pearson', {}).get('correlation', 0)) > abs(correlations.get('qubits_qes_pearson', {}).get('correlation', 0)) else 'System Size'} has the strongest effect on quantum emergent spacetime signatures.

### Optimal Parameters
- For maximum quantum signatures, use curvature around {analysis_summary.get('best_curvature', 0):.1f} and {analysis_summary.get('best_qubits', 0)} qubits.

### Recommendations
- {'Higher curvature enhances quantum signatures' if correlations.get('curvature_qes_pearson', {}).get('correlation', 0) > 0.3 else 'Curvature has limited effect on quantum signatures'}
- {'Smaller systems show stronger quantum effects' if correlations.get('qubits_qes_pearson', {}).get('correlation', 0) < -0.3 else 'System size has limited effect on quantum signatures'}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return report

def main():
    """Main function to run the curvature-system size analysis."""
    print("🔬 Running Curvature-System Size Analysis...")
    
    # Find experiment files
    experiment_files = find_experiment_files()
    print(f"📊 Found {len(experiment_files)} experiment files")
    
    if not experiment_files:
        print("❌ No experiment files found!")
        return
    
    # Analyze experiments
    results = analyze_experiments(experiment_files)
    
    if not results:
        print("❌ No experiments could be analyzed!")
        return
    
    print(f"\n✅ Successfully analyzed {len(results)} experiments")
    
    # Perform statistical analysis
    analysis_summary = perform_statistical_analysis(results)
    
    # Generate visualizations
    generate_visualizations(results, analysis_summary)
    
    # Save results
    save_results(results, analysis_summary)
    
    print("\n🔍 Analysis Complete!")
    print(f"📊 Analyzed {analysis_summary.get('total_experiments', 0)} experiments")
    print(f"🏆 Best Quantum Score: {analysis_summary.get('best_quantum_score', 0):.3f}")
    print(f"🎯 Optimal Curvature: {analysis_summary.get('best_curvature', 0):.1f}")
    print(f"🎯 Optimal Qubits: {analysis_summary.get('best_qubits', 0)}")

if __name__ == "__main__":
    main() 