#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

from experiments.custom_curvature_experiment import run_superposition_gravity_experiment
import argparse

# Create a simple args object for testing
class Args:
    def __init__(self):
        self.num_qubits = 4
        self.device = "simulator"
        self.shots = 100
        self.massive_bulk_mass_hinge = "2"
        self.massive_bulk_mass_value = 1.0
        self.massless_bulk_mass_hinge = "None"
        self.massless_bulk_mass_value = 0.0
        self.superposition_control_qubit = 0
        self.superposition_phase = 0.0
        self.classical_mixture_comparison = True
        self.interference_analysis = True
        self.bulk_reconstruction_method = "mi_embedding"
        self.coherence_preservation = False  # Add missing attribute

# Test the superposition experiment
if __name__ == "__main__":
    args = Args()
    experiment_log_dir = "test_superposition_debug"
    
    # Create directory if it doesn't exist
    os.makedirs(experiment_log_dir, exist_ok=True)
    
    print("Testing superposition gravity experiment...")
    try:
        results = run_superposition_gravity_experiment(args, experiment_log_dir)
        print(f"Results: {results}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 