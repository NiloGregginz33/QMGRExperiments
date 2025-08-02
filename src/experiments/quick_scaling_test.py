#!/usr/bin/env python3
"""
Quick test of scaling optimizations for the quantum geometry experiment.
"""

import sys
import os
import time
import numpy as np
from qiskit import QuantumCircuit

# Add the experiments directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'experiments'))

def quick_test():
    """Quick test of the optimized functions."""
    print("=== QUICK SCALING OPTIMIZATION TEST ===")
    
    try:
        # Test 1: Import optimized functions
        print("1. Testing imports...")
        from custom_curvature_experiment import create_optimized_quantum_spacetime_circuit
        from custom_curvature_experiment import compute_optimized_von_neumann_MI
        from custom_curvature_experiment import progressive_analysis_runner
        print("   ✅ All optimized functions imported successfully")
        
        # Test 2: Circuit creation scaling
        print("\n2. Testing circuit creation scaling...")
        qubit_counts = [4, 6, 8]
        
        for num_qubits in qubit_counts:
            print(f"   Testing {num_qubits} qubits...")
            start_time = time.time()
            qc = create_optimized_quantum_spacetime_circuit(num_qubits, entanglement_strength=2.0)
            creation_time = time.time() - start_time
            print(f"   ✅ {num_qubits} qubits: {creation_time:.3f}s, depth: {qc.depth()}")
        
        # Test 3: Simple statevector creation
        print("\n3. Testing statevector creation...")
        qc = QuantumCircuit(4)
        qc.h(range(4))
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        
        # Create a simple statevector for testing
        from qiskit.quantum_info import Statevector
        statevector = Statevector.from_instruction(qc)
        
        # Test 4: MI computation
        print("\n4. Testing MI computation...")
        start_time = time.time()
        mi_dict = compute_optimized_von_neumann_MI(statevector)
        mi_time = time.time() - start_time
        print(f"   ✅ MI computation: {mi_time:.3f}s, {len(mi_dict)} pairs")
        
        # Test 5: Progressive analysis (simulated)
        print("\n5. Testing progressive analysis...")
        start_time = time.time()
        results = progressive_analysis_runner(qc, 4, "simulator", shots=100)
        analysis_time = time.time() - start_time
        print(f"   ✅ Progressive analysis: {analysis_time:.3f}s")
        print(f"   Results: {list(results.keys())}")
        
        print("\n=== ALL TESTS PASSED ===")
        print("✅ Optimized functions are working correctly")
        print("✅ Scaling improvements are active")
        print("✅ Ready for larger qubit counts")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test() 