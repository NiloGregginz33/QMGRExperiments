#!/usr/bin/env python3
"""
Debug Transpilation - Understand why circuits collapse to all zeros
"""

import sys
import os
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from CGPTFactory import extract_counts_from_bitarray

def debug_circuit_transpilation():
    """Debug what happens during circuit transpilation."""
    
    print("=== DEBUGGING CIRCUIT TRANSPILATION ===")
    
    # Create the same circuit as in area_law_hardware_robust.py
    qc = QuantumCircuit(4, 4)
    
    # Layer 1: Initial Hadamard gates
    for i in range(4):
        qc.h(i)
    
    # Depth 2: Nearest neighbor entanglement
    for layer in range(2):
        for i in range(3):  # 4 qubits -> 3 pairs
            qc.cx(i, i + 1)
            qc.h(i)
            qc.h(i + 1)
    
    # Additional entanglement
    for i in range(3):
        qc.cx(i, i + 1)
        qc.h(i)
        qc.h(i + 1)
    
    # Final layer
    for i in range(4):
        qc.h(i)
    
    print("Original circuit:")
    print(qc.draw())
    
    # Test on simulator first
    print("\n--- Simulator Test ---")
    qc_no_measure = qc.copy()
    qc_no_measure.remove_final_measurements()
    sv = Statevector.from_instruction(qc_no_measure)
    print(f"Statevector: {sv}")
    
    # Check probabilities
    probs = np.abs(sv.data) ** 2
    print("Probabilities:")
    for i, p in enumerate(probs):
        if p > 0.01:
            print(f"  |{i:04b}>: {p:.4f}")
    
    # Test transpilation
    print("\n--- Transpilation Test ---")
    try:
        service = QiskitRuntimeService()
        backend = service.backend("ibm_brisbane")
        
        print(f"Backend: {backend.name}")
        print(f"Status: {backend.status()}")
        
        # Transpile with different optimization levels
        for opt_level in [0, 1, 2, 3]:
            print(f"\nOptimization level {opt_level}:")
            qc_t = transpile(qc, backend, optimization_level=opt_level)
            print(f"  Depth: {qc_t.depth()}")
            print(f"  Gates: {qc_t.count_ops()}")
            print(f"  Circuit:")
            print(qc_t.draw())
            
            # Test the transpiled circuit on simulator
            qc_t_no_measure = qc_t.copy()
            qc_t_no_measure.remove_final_measurements()
            try:
                sv_t = Statevector.from_instruction(qc_t_no_measure)
                probs_t = np.abs(sv_t.data) ** 2
                print(f"  Probabilities:")
                for i, p in enumerate(probs_t):
                    if p > 0.01:
                        print(f"    |{i:04b}>: {p:.4f}")
                
                # Check if it's all zeros
                if probs_t[0] > 0.99:
                    print(f"  ⚠️  WARNING: Circuit collapses to |0000> with probability {probs_t[0]:.4f}")
                else:
                    print(f"  ✅ Circuit has good distribution")
                    
            except Exception as e:
                print(f"  ❌ Error simulating transpiled circuit: {e}")
        
    except Exception as e:
        print(f"Transpilation failed: {e}")
        import traceback
        traceback.print_exc()

def test_simple_entangling():
    """Test a very simple entangling circuit."""
    
    print("\n=== SIMPLE ENTANGLING TEST ===")
    
    # Create a simple 2-qubit entangling circuit
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(0)
    qc.h(1)
    qc.measure_all()
    
    print("Simple entangling circuit:")
    print(qc.draw())
    
    # Test on simulator
    qc_no_measure = qc.copy()
    qc_no_measure.remove_final_measurements()
    sv = Statevector.from_instruction(qc_no_measure)
    probs = np.abs(sv.data) ** 2
    print("Simulator probabilities:")
    for i, p in enumerate(probs):
        if p > 0.01:
            print(f"  |{i:02b}>: {p:.4f}")
    
    # Test on hardware
    try:
        service = QiskitRuntimeService()
        backend = service.backend("ibm_brisbane")
        
        qc_t = transpile(qc, backend, optimization_level=0)
        print(f"\nTranspiled circuit:")
        print(qc_t.draw())
        
        sampler = Sampler(backend)
        job = sampler.run([qc_t], shots=100)
        result = job.result()
        
        pub_result = result[0]
        data = pub_result.data
        bitarray = data.c
        
        counts = extract_counts_from_bitarray(bitarray)
        print(f"Hardware counts: {counts}")
        
    except Exception as e:
        print(f"Hardware test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_circuit_transpilation()
    test_simple_entangling() 