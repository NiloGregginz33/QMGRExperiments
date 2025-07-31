#!/usr/bin/env python3
"""
Quantum Diagnostic Tool
=======================
Simple script to diagnose why quantum metrics are all returning 0.
"""

import json
import numpy as np
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit import QuantumCircuit
import sys
import os

def load_experiment_results(filepath):
    """Load experiment results and print diagnostic info."""
    print(f"🔍 Loading experiment results from: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded JSON data")
        return data
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None

def analyze_quantum_state_outputs(data):
    """Analyze the quantum_state_outputs section."""
    print("\n🔬 ANALYZING QUANTUM STATE OUTPUTS")
    print("=" * 50)
    
    if 'quantum_state_outputs' not in data:
        print("❌ No 'quantum_state_outputs' found in data")
        return
    
    outputs = data['quantum_state_outputs']
    print(f"✅ Found {len(outputs)} quantum state outputs")
    
    for i, output in enumerate(outputs):
        print(f"\n--- Timestep {i+1} ---")
        
        # Check statevector
        if 'statevector' in output:
            statevector_data = output['statevector']
            print(f"✅ Statevector found with {len(statevector_data)} elements")
            
            # Convert to numpy array
            try:
                # Handle complex number format
                if isinstance(statevector_data[0], dict) and 'real' in statevector_data[0]:
                    # Format: [{"real": 0.1, "imag": 0.2}, ...]
                    complex_array = np.array([complex(item['real'], item['imag']) for item in statevector_data])
                else:
                    # Format: [0.1+0.2j, ...]
                    complex_array = np.array(statevector_data)
                
                print(f"✅ Converted to complex array: shape {complex_array.shape}")
                print(f"   First few elements: {complex_array[:3]}")
                
                # Check normalization
                norm = np.linalg.norm(complex_array)
                print(f"   Norm: {norm:.6f} (should be ~1.0)")
                
                # Create Qiskit Statevector
                statevector = Statevector(complex_array)
                print(f"✅ Created Qiskit Statevector successfully")
                
                # Test basic quantum properties
                density_matrix = DensityMatrix(statevector)
                purity = density_matrix.purity()
                print(f"   Purity: {purity:.6f}")
                
                # Test von Neumann entropy
                # Calculate entropy manually from eigenvalues
                density_array = density_matrix.data
                eigenvalues = np.linalg.eigvals(density_array)
                entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
                print(f"   Von Neumann entropy: {entropy:.6f}")
                
            except Exception as e:
                print(f"❌ Error processing statevector: {e}")
        else:
            print("❌ No statevector found")
        
        # Check mutual information matrix
        if 'mutual_information_matrix' in output:
            mi_data = output['mutual_information_matrix']
            print(f"✅ Mutual information matrix found")
            print(f"   Type: {type(mi_data)}")
            print(f"   Content: {mi_data}")
        else:
            print("❌ No mutual information matrix found")

def test_simple_quantum_circuit():
    """Test a simple quantum circuit to verify our analysis works."""
    print("\n🧪 TESTING SIMPLE QUANTUM CIRCUIT")
    print("=" * 50)
    
    # Create a simple Bell state
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    
    print(f"✅ Created Bell state circuit")
    print(f"   Circuit depth: {qc.depth()}")
    print(f"   Number of gates: {len(qc.data)}")
    
    # Get statevector
    statevector = Statevector.from_instruction(qc)
    print(f"✅ Statevector: {statevector}")
    
    # Test quantum properties
    density_matrix = DensityMatrix(statevector)
    purity = density_matrix.purity()
    # Calculate entropy manually from eigenvalues
    density_array = density_matrix.data
    eigenvalues = np.linalg.eigvals(density_array)
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
    
    print(f"   Purity: {purity:.6f}")
    print(f"   Von Neumann entropy: {entropy:.6f}")
    
    # Test Bell inequality violation
    # For Bell state |00⟩ + |11⟩, we expect strong correlations
    print(f"   Bell state correlations should be strong")

def main():
    """Main diagnostic function."""
    if len(sys.argv) != 2:
        print("Usage: python quantum_diagnostic.py <experiment_results.json>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    print("🚀 QUANTUM DIAGNOSTIC TOOL")
    print("=" * 60)
    
    # Test simple quantum circuit first
    test_simple_quantum_circuit()
    
    # Analyze experiment results
    data = load_experiment_results(filepath)
    if data:
        analyze_quantum_state_outputs(data)
    
    print("\n🎯 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print("This tool helps identify why quantum metrics are returning 0.")
    print("Check the output above for any ❌ errors or missing data.")

if __name__ == "__main__":
    main() 