#!/usr/bin/env python3
"""
Debug script for entropy optimization
"""

import sys
import os
sys.path.append('src')

# Import the function we want to test
from experiments.custom_curvature_experiment import set_target_subsystem_entropy

def test_entropy_optimization():
    """Test the entropy optimization function"""
    print("Testing entropy optimization...")
    
    # Simple test case
    target_entropies = [0.5, 1.0, 1.5]  # Target entropies for subsystems of size 1, 2, 3
    num_qubits = 4
    
    print(f"Target entropies: {target_entropies}")
    print(f"Number of qubits: {num_qubits}")
    
    # Run optimization
    result = set_target_subsystem_entropy(target_entropies, num_qubits, max_iter=20)
    
    print(f"Optimization result: {result}")
    
    if result['success']:
        print(f"Success! Achieved entropies: {result['achieved_entropies']}")
        print(f"Loss: {result['loss']}")
        print(f"Iterations: {result['iterations']}")
    else:
        print(f"Failed: {result['error']}")

if __name__ == "__main__":
    test_entropy_optimization() 