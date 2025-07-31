"""
Run Causal Asymmetry Analysis on Custom Curvature Experiment Results

This script demonstrates how to integrate causal asymmetry detection into your
custom curvature experiments to detect CTC loops, non-commuting path integrals,
and time-reversal violations.

Usage:
    python tools/run_causal_asymmetry_analysis.py --experiment_file path/to/results.json
"""

import argparse
import json
import numpy as np
import os
import sys
from datetime import datetime

# Add the tools directory to the path
sys.path.append(os.path.dirname(__file__))

from causal_asymmetry_detection import detect_causal_asymmetry_evidence, create_causal_asymmetry_visualization

def load_experiment_data(file_path):
    """Load experiment data from JSON file."""
    print(f"📂 Loading experiment data from: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract key information
    if 'spec' in data:
        spec = data['spec']
        num_qubits = spec['num_qubits']
        geometry = spec.get('geometry', 'unknown')
        curvature = spec.get('curvature', 0)
    else:
        num_qubits = data.get('num_qubits', 0)
        geometry = data.get('geometry_type', 'unknown')
        curvature = data.get('curvature', 0)
    
    # Extract mutual information data
    if 'mutual_information_per_timestep' in data:
        timesteps_data = data['mutual_information_per_timestep']
        
        # Convert the last timestep to a matrix for analysis
        if timesteps_data:
            last_timestep = timesteps_data[-1]
            mi_matrix = np.zeros((num_qubits, num_qubits))
            
            for key, value in last_timestep.items():
                if key.startswith('I_'):
                    indices = key[2:].split(',')
                    i, j = int(indices[0]), int(indices[1])
                    mi_matrix[i, j] = value
                    mi_matrix[j, i] = value  # Symmetric matrix
            
            # Fill diagonal with mean of off-diagonal values
            off_diag_mean = np.mean(mi_matrix[np.triu_indices(num_qubits, k=1)])
            np.fill_diagonal(mi_matrix, off_diag_mean)
        else:
            raise ValueError("No mutual information data found")
    else:
        raise ValueError("No mutual_information_per_timestep found in data")
    
    print(f"✅ Loaded data:")
    print(f"   - Qubits: {num_qubits}")
    print(f"   - Geometry: {geometry}")
    print(f"   - Curvature: {curvature}")
    print(f"   - Timesteps: {len(timesteps_data)}")
    
    return {
        'mi_matrix': mi_matrix,
        'timesteps_data': timesteps_data,
        'num_qubits': num_qubits,
        'geometry': geometry,
        'curvature': curvature,
        'raw_data': data
    }

def save_causal_asymmetry_results(results, experiment_data, output_dir):
    """Save causal asymmetry analysis results."""
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = os.path.join(output_dir, f"causal_asymmetry_analysis_{timestamp}")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Save detailed results
    results_file = os.path.join(analysis_dir, "causal_asymmetry_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create visualization
    viz_file = os.path.join(analysis_dir, "causal_asymmetry_visualization.png")
    create_causal_asymmetry_visualization(results, viz_file)
    
    # Create summary report
    summary_file = os.path.join(analysis_dir, "causal_asymmetry_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("CAUSAL ASYMMETRY EVIDENCE ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Experiment Details:\n")
        f.write(f"- Qubits: {experiment_data['num_qubits']}\n")
        f.write(f"- Geometry: {experiment_data['geometry']}\n")
        f.write(f"- Curvature: {experiment_data['curvature']}\n")
        f.write(f"- Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Key Findings:\n")
        f.write(f"- CTC Loops Detected: {'YES' if results['ctc_loops_detected'] else 'NO'}\n")
        f.write(f"- Non-commuting Paths: {'YES' if results['non_commuting_paths'] else 'NO'}\n")
        f.write(f"- Time-Reversal Violations: {'YES' if results['time_reversal_violations'] else 'NO'}\n")
        f.write(f"- Causal Consistency Score: {results['causal_consistency_score']:.4f}\n")
        f.write(f"- Overall Assessment: {results['overall_assessment']['causal_structure']}\n")
        f.write(f"- Confidence Level: {results['overall_assessment']['confidence_level']:.4f}\n\n")
        
        f.write(f"Quantum Gravity Evidence:\n")
        f.write(f"- Evidence for Quantum Gravity: {'YES' if results['overall_assessment']['quantum_gravity_evidence'] else 'NO'}\n")
        f.write(f"- Total Violations: {results['overall_assessment']['total_violations']}\n\n")
        
        f.write(f"Detailed Analysis:\n")
        
        if results['ctc_loops_detected']:
            ctc_data = results['detailed_analysis']['ctc_loops']
            f.write(f"- CTC Loops: {ctc_data['num_ctc_candidates']} candidates found\n")
            if ctc_data['strongest_ctc']:
                strongest = ctc_data['strongest_ctc']
                f.write(f"  Strongest CTC: qubits {strongest['qubits']}, distance {strongest['loop_distance']:.6f}\n")
        
        if results['non_commuting_paths']:
            path_data = results['detailed_analysis']['path_integrals']
            f.write(f"- Non-commuting Paths: {len(path_data['commutator_violations'])} commutator violations\n")
            f.write(f"  Path Dependence: {'YES' if path_data['path_dependence']['path_dependent'] else 'NO'}\n")
        
        if results['time_reversal_violations']:
            time_data = results['detailed_analysis']['time_reversal']
            f.write(f"- Time-Reversal Violations: asymmetry score {time_data['asymmetry_score']:.4f}\n")
            f.write(f"  Entropy Production: {time_data['entropy_production_analysis']['net_production']:.4f}\n")
        
        f.write(f"\nAsymmetry Metrics:\n")
        asymmetry = results['asymmetry_metrics']
        f.write(f"- Spatial Asymmetry: {asymmetry.get('spatial_asymmetry', 0):.4f}\n")
        f.write(f"- Temporal Asymmetry: {asymmetry.get('temporal_asymmetry', 0):.4f}\n")
        f.write(f"- Geometric Asymmetry: {asymmetry.get('geometric_asymmetry', 0):.4f}\n")
        f.write(f"- Total Asymmetry: {asymmetry.get('total_asymmetry', 0):.4f}\n")
        
        f.write(f"\nPhysics Interpretation:\n")
        if results['overall_assessment']['quantum_gravity_evidence']:
            f.write(f"This experiment shows evidence of causal asymmetry that could indicate\n")
            f.write(f"emergent quantum gravity effects. The violations of classical causality\n")
            f.write(f"suggest the presence of quantum correlations that transcend classical\n")
            f.write(f"spacetime structure.\n")
        else:
            f.write(f"This experiment shows consistent causal structure, suggesting classical\n")
            f.write(f"or semi-classical behavior without strong quantum gravity effects.\n")
    
    print(f"📁 Results saved to: {analysis_dir}")
    print(f"   - JSON results: {results_file}")
    print(f"   - Visualization: {viz_file}")
    print(f"   - Summary report: {summary_file}")
    
    return analysis_dir

def main():
    parser = argparse.ArgumentParser(description="Run causal asymmetry analysis on custom curvature experiment results")
    parser.add_argument("--experiment_file", type=str, required=True,
                       help="Path to the experiment results JSON file")
    parser.add_argument("--output_dir", type=str, default="experiment_logs/causal_asymmetry_analysis",
                       help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.experiment_file):
        print(f"❌ Error: Experiment file not found: {args.experiment_file}")
        return
    
    try:
        # Load experiment data
        experiment_data = load_experiment_data(args.experiment_file)
        
        # Run causal asymmetry detection
        print("\n🔍 Running Causal Asymmetry Detection...")
        results = detect_causal_asymmetry_evidence(
            mi_matrix=experiment_data['mi_matrix'],
            timesteps_data=experiment_data['timesteps_data'],
            num_qubits=experiment_data['num_qubits'],
            geometry=experiment_data['geometry'],
            curvature=experiment_data['curvature']
        )
        
        # Save results
        output_dir = save_causal_asymmetry_results(results, experiment_data, args.output_dir)
        
        # Print summary
        print("\n" + "=" * 60)
        print("CAUSAL ASYMMETRY ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"CTC Loops: {'DETECTED' if results['ctc_loops_detected'] else 'None'}")
        print(f"Non-commuting Paths: {'DETECTED' if results['non_commuting_paths'] else 'None'}")
        print(f"Time-Reversal Violations: {'DETECTED' if results['time_reversal_violations'] else 'None'}")
        print(f"Causal Consistency: {results['causal_consistency_score']:.4f}")
        print(f"Overall Assessment: {results['overall_assessment']['causal_structure']}")
        print(f"Quantum Gravity Evidence: {'YES' if results['overall_assessment']['quantum_gravity_evidence'] else 'NO'}")
        print(f"Confidence Level: {results['overall_assessment']['confidence_level']:.4f}")
        print(f"\nResults saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 