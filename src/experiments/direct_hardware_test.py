#!/usr/bin/env python3
"""
Direct Hardware Test - Use Qiskit Runtime directly
"""

import sys
import os
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Factory'))
from CGPTFactory import extract_counts_from_bitarray

def test_direct_hardware():
    """Test hardware directly with Qiskit Runtime."""
    
    print("=== DIRECT HARDWARE TEST ===")
    
    # Create a simple circuit
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    
    print("Circuit:")
    print(qc.draw())
    
    # Simulator test
    print("\n--- Simulator Test ---")
    qc_no_measure = qc.copy()
    qc_no_measure.remove_final_measurements()
    sv = Statevector.from_instruction(qc_no_measure)
    print(f"Statevector: {sv}")
    
    # Direct hardware test
    print("\n--- Direct Hardware Test ---")
    try:
        service = QiskitRuntimeService()
        backend = service.backend("ibm_brisbane")
        
        print(f"Backend: {backend.name}")
        print(f"Status: {backend.status()}")
        
        # Transpile
        qc_t = transpile(qc, backend, optimization_level=0)
        print(f"Transpiled circuit depth: {qc_t.depth()}")
        print(f"Transpiled gates: {qc_t.count_ops()}")
        
        # Run directly with SamplerV2
        sampler = Sampler(backend)
        job = sampler.run([qc_t], shots=100)
        result = job.result()
        
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        
        # Extract counts
        pub_result = result[0]
        print(f"Pub result type: {type(pub_result)}")
        print(f"Pub result: {pub_result}")
        
        # Try to get counts
        if hasattr(pub_result, 'data'):
            data = pub_result.data
            print(f"Data type: {type(data)}")
            print(f"Data: {data}")
            
            if hasattr(data, 'c'):
                bitarray = data.c
                print(f"BitArray: {bitarray}")
                print(f"First 10 bitstrings: {bitarray[:10]}")
                
                # Convert to counts using CGPTFactory function
                counts = extract_counts_from_bitarray(bitarray)
                
                print(f"Counts: {counts}")
                
                # Calculate entropy
                total = sum(counts.values())
                probs = {k: v/total for k, v in counts.items()}
                entropy = -sum(p * np.log2(p) for p in probs.values() if p > 0)
                print(f"Entropy: {entropy:.4f}")
        
    except Exception as e:
        print(f"Hardware execution failed: {e}")
        import traceback
        traceback.print_exc()

def test_simple_hadamard():
    """Test just a Hadamard gate directly."""
    
    print("\n=== SIMPLE HADAMARD TEST ===")
    
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    
    print("Hadamard circuit:")
    print(qc.draw())
    
    try:
        service = QiskitRuntimeService()
        backend = service.backend("ibm_brisbane")
        
        # Transpile
        qc_t = transpile(qc, backend, optimization_level=0)
        print(f"Transpiled circuit:")
        print(qc_t.draw())
        
        # Run
        sampler = Sampler(backend)
        job = sampler.run([qc_t], shots=100)
        result = job.result()
        
        # Extract counts
        pub_result = result[0]
        data = pub_result.data
        bitarray = data.c
        
        counts = extract_counts_from_bitarray(bitarray)
        
        print(f"Hardware counts: {counts}")
        
    except Exception as e:
        print(f"Hardware execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_hardware()
    test_simple_hadamard() 