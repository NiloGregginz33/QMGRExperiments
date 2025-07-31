"""
Integration Guide: Causal Asymmetry Detection in Custom Curvature Experiments

This script shows how to integrate causal asymmetry detection into your
custom curvature experiment workflow.

Usage:
    # Add this to your custom curvature experiment after running the circuit
    python tools/integrate_causal_asymmetry.py --experiment_results path/to/results.json
"""

import argparse
import json
import numpy as np
import os
import sys
from datetime import datetime

# Add the tools directory to the path
sys.path.append(os.path.dirname(__file__))

from causal_asymmetry_detection import detect_causal_asymmetry_evidence

def integrate_causal_asymmetry_into_experiment(experiment_results_path, output_dir=None):
    """
    Integrate causal asymmetry detection into your custom curvature experiment.
    
    This function should be called after your experiment completes and saves results.
    """
    
    print("🔗 Integrating Causal Asymmetry Detection...")
    
    # Load your experiment results
    with open(experiment_results_path, 'r') as f:
        experiment_data = json.load(f)
    
    # Extract experiment parameters
    if 'spec' in experiment_data:
        spec = experiment_data['spec']
        num_qubits = spec['num_qubits']
        geometry = spec.get('geometry', 'unknown')
        curvature = spec.get('curvature', 0)
    else:
        num_qubits = experiment_data.get('num_qubits', 0)
        geometry = experiment_data.get('geometry_type', 'unknown')
        curvature = experiment_data.get('curvature', 0)
    
    # Extract mutual information data
    if 'mutual_information_per_timestep' not in experiment_data:
        print("❌ No mutual information data found in experiment results")
        return None
    
    timesteps_data = experiment_data['mutual_information_per_timestep']
    
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
        print("❌ No timestep data found")
        return None
    
    # Run causal asymmetry detection
    print(f"🔍 Running Causal Asymmetry Detection on {num_qubits}-qubit {geometry} geometry...")
    
    causal_results = detect_causal_asymmetry_evidence(
        mi_matrix=mi_matrix,
        timesteps_data=timesteps_data,
        num_qubits=num_qubits,
        geometry=geometry,
        curvature=curvature
    )
    
    # Add causal asymmetry results to your experiment data
    experiment_data['causal_asymmetry_analysis'] = causal_results
    
    # Save updated experiment results
    if output_dir is None:
        output_dir = os.path.dirname(experiment_results_path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    updated_results_path = os.path.join(output_dir, f"results_with_causal_asymmetry_{timestamp}.json")
    
    with open(updated_results_path, 'w') as f:
        json.dump(experiment_data, f, indent=2, default=str)
    
    print(f"✅ Causal asymmetry analysis integrated and saved to: {updated_results_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("CAUSAL ASYMMETRY INTEGRATION SUMMARY")
    print("=" * 60)
    print(f"Experiment: {num_qubits} qubits, {geometry} geometry, curvature {curvature}")
    print(f"CTC Loops: {'DETECTED' if causal_results['ctc_loops_detected'] else 'None'}")
    print(f"Non-commuting Paths: {'DETECTED' if causal_results['non_commuting_paths'] else 'None'}")
    print(f"Time-Reversal Violations: {'DETECTED' if causal_results['time_reversal_violations'] else 'None'}")
    print(f"Causal Consistency: {causal_results['causal_consistency_score']:.4f}")
    print(f"Overall Assessment: {causal_results['overall_assessment']['causal_structure']}")
    print(f"Quantum Gravity Evidence: {'YES' if causal_results['overall_assessment']['quantum_gravity_evidence'] else 'NO'}")
    
    return causal_results

def add_to_custom_curvature_experiment():
    """
    Example of how to add this to your custom curvature experiment script.
    
    Add this code to your main experiment function after the circuit execution:
    """
    
    example_code = '''
    # === ADD THIS TO YOUR CUSTOM CURVATURE EXPERIMENT ===
    
    # After your experiment completes and saves results:
    if args.detect_causal_asymmetry:  # Add this flag to your argparse
        print("🔗 Running causal asymmetry detection...")
        
        # Import the detection function
        from tools.causal_asymmetry_detection import detect_causal_asymmetry_evidence
        
        # Extract mutual information data from your results
        if 'mutual_information_per_timestep' in experiment_results:
            timesteps_data = experiment_results['mutual_information_per_timestep']
            
            # Convert to matrix format
            mi_matrix = np.zeros((args.num_qubits, args.num_qubits))
            last_timestep = timesteps_data[-1]
            
            for key, value in last_timestep.items():
                if key.startswith('I_'):
                    indices = key[2:].split(',')
                    i, j = int(indices[0]), int(indices[1])
                    mi_matrix[i, j] = value
                    mi_matrix[j, i] = value
            
            # Fill diagonal
            off_diag_mean = np.mean(mi_matrix[np.triu_indices(args.num_qubits, k=1)])
            np.fill_diagonal(mi_matrix, off_diag_mean)
            
            # Run causal asymmetry detection
            causal_results = detect_causal_asymmetry_evidence(
                mi_matrix=mi_matrix,
                timesteps_data=timesteps_data,
                num_qubits=args.num_qubits,
                geometry=args.geometry,
                curvature=kappa  # Current curvature being tested
            )
            
            # Add to experiment results
            experiment_results['causal_asymmetry_analysis'] = causal_results
            
            # Save updated results
            with open(output_path, 'w') as f:
                json.dump(experiment_results, f, indent=2, cls=CustomJSONEncoder)
            
            print(f"✅ Causal asymmetry analysis complete:")
            print(f"   - Assessment: {causal_results['overall_assessment']['causal_structure']}")
            print(f"   - Quantum Gravity Evidence: {causal_results['overall_assessment']['quantum_gravity_evidence']}")
    
    # === END OF INTEGRATION CODE ===
    '''
    
    print("Example integration code:")
    print(example_code)

def main():
    parser = argparse.ArgumentParser(description="Integrate causal asymmetry detection into custom curvature experiments")
    parser.add_argument("--experiment_results", type=str, required=False,
                       help="Path to the experiment results JSON file")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for updated results")
    parser.add_argument("--show_integration", action="store_true",
                       help="Show example integration code")
    
    args = parser.parse_args()
    
    if args.show_integration:
        add_to_custom_curvature_experiment()
        return
    
    if not args.experiment_results:
        print("❌ Error: --experiment_results is required unless --show_integration is used")
        return
    
    # Check if file exists
    if not os.path.exists(args.experiment_results):
        print(f"❌ Error: Experiment file not found: {args.experiment_results}")
        return
    
    try:
        # Integrate causal asymmetry detection
        results = integrate_causal_asymmetry_into_experiment(args.experiment_results, args.output_dir)
        
        if results:
            print(f"\n🎉 Integration successful!")
            print(f"Your experiment now includes causal asymmetry analysis.")
            print(f"Check the updated results file for detailed analysis.")
        
    except Exception as e:
        print(f"❌ Error during integration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 