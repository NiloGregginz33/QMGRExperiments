#!/usr/bin/env python3
"""
Verification script for quantum experiment results.
Checks shot counts, angle sums, mutual information, and alpha values.
"""

import json
import numpy as np
from pathlib import Path

def verify_shot_counts(results_file):
    """Verify that shot counts add up to the declared number of shots."""
    print(f"\n=== SHOT COUNTS VERIFICATION ===")
    print(f"File: {results_file}")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    declared_shots = data['spec']['shots']
    num_qubits = data['spec']['num_qubits']
    timesteps = data['spec']['timesteps']
    
    print(f"Declared shots: {declared_shots}")
    print(f"Number of qubits: {num_qubits}")
    print(f"Number of timesteps: {timesteps}")
    
    counts_per_timestep = data['counts_per_timestep']
    
    for t, counts in enumerate(counts_per_timestep):
        total_count = sum(counts.values())
        print(f"\nTimestep {t}:")
        print(f"  Total count: {total_count}")
        print(f"  Expected: {declared_shots}")
        print(f"  Match: {'✓' if total_count == declared_shots else '✗'}")
        
        # Check bitstring lengths
        bitstring_lengths = [len(bitstring) for bitstring in counts.keys()]
        expected_length = num_qubits
        all_correct_length = all(length == expected_length for length in bitstring_lengths)
        print(f"  Bitstring lengths correct: {'✓' if all_correct_length else '✗'}")
        
        if not all_correct_length:
            print(f"  Expected length: {expected_length}")
            print(f"  Found lengths: {set(bitstring_lengths)}")

def verify_angle_sums(results_file):
    """Verify angle sums array length and structure."""
    print(f"\n=== ANGLE SUMS VERIFICATION ===")
    print(f"File: {results_file}")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    num_qubits = data['spec']['num_qubits']
    timesteps = data['spec']['timesteps']
    angle_sums = data['angle_sums']
    
    print(f"Number of qubits: {num_qubits}")
    print(f"Number of timesteps: {timesteps}")
    print(f"Expected angle sums (qubits × timesteps): {num_qubits * timesteps}")
    print(f"Actual angle sums length: {len(angle_sums)}")
    print(f"Match: {'✓' if len(angle_sums) == num_qubits * timesteps else '✗'}")
    
    if len(angle_sums) != num_qubits * timesteps:
        print(f"  Extra entries: {len(angle_sums) - num_qubits * timesteps}")
        print(f"  Missing entries: {num_qubits * timesteps - len(angle_sums)}")
    
    # Check if angle sums might be per triangle instead
    # For n qubits, number of triangles = C(n,3) = n!/(3!(n-3)!)
    if num_qubits >= 3:
        num_triangles = (num_qubits * (num_qubits - 1) * (num_qubits - 2)) // 6
        print(f"Number of possible triangles: {num_triangles}")
        print(f"Triangles × timesteps: {num_triangles * timesteps}")
        print(f"Match with triangles: {'✓' if len(angle_sums) == num_triangles * timesteps else '✗'}")

def verify_mutual_information(results_file):
    """Verify mutual information data and check for non-zero values."""
    print(f"\n=== MUTUAL INFORMATION VERIFICATION ===")
    print(f"File: {results_file}")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    if 'mutual_information_per_timestep' not in data:
        print("No mutual information data found!")
        return
    
    mi_data = data['mutual_information_per_timestep']
    print(f"Number of timesteps with MI data: {len(mi_data)}")
    
    for t, timestep_mi in enumerate(mi_data):
        print(f"\nTimestep {t}:")
        
        # Find non-zero MI values
        non_zero_mi = {edge: value for edge, value in timestep_mi.items() if value > 1e-10}
        
        if non_zero_mi:
            print(f"  Non-zero MI values found: {len(non_zero_mi)}")
            max_mi = max(non_zero_mi.values())
            max_edge = max(non_zero_mi, key=non_zero_mi.get)
            print(f"  Maximum MI: {max_mi:.6f} on edge {max_edge}")
            
            # Show all non-zero values
            for edge, value in sorted(non_zero_mi.items(), key=lambda x: x[1], reverse=True):
                print(f"    {edge}: {value:.6f}")
        else:
            print("  All MI values are essentially zero (< 1e-10)")

def verify_alpha_values(results_file):
    """Verify alpha parameter values and clarify input vs emergent."""
    print(f"\n=== ALPHA PARAMETER VERIFICATION ===")
    print(f"File: {results_file}")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Input alpha from spec
    input_alpha = data['spec']['alpha']
    print(f"Input alpha (hyperparameter): {input_alpha}")
    
    # Check for emergent alpha in results
    if 'emergent_alpha' in data:
        emergent_alpha = data['emergent_alpha']
        print(f"Emergent alpha (posterior fit): {emergent_alpha}")
    else:
        print("No emergent alpha found in results")
    
    # Check for any other alpha-related values
    alpha_keys = [key for key in data.keys() if 'alpha' in key.lower()]
    if alpha_keys:
        print(f"Other alpha-related keys: {alpha_keys}")

def main():
    """Main verification function."""
    # Check the specific file mentioned in the user's query
    results_file = "experiment_logs/custom_curvature_experiment/results_n7_geomH_curv05_ibm_60JXA8.json"
    
    if not Path(results_file).exists():
        print(f"File not found: {results_file}")
        return
    
    print("QUANTUM EXPERIMENT RESULTS VERIFICATION")
    print("=" * 50)
    
    verify_shot_counts(results_file)
    verify_angle_sums(results_file)
    verify_mutual_information(results_file)
    verify_alpha_values(results_file)
    
    print(f"\n" + "=" * 50)
    print("VERIFICATION COMPLETE")

if __name__ == "__main__":
    main() 