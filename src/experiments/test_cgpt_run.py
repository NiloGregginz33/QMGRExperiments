#!/usr/bin/env python3
"""
Test CGPTFactory run function with IBM hardware
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
from Factory.CGPTFactory import run

def test_simple_circuit():
    """Test a simple 3-qubit circuit on IBM hardware"""
    
    # Create a simple circuit
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    
    print("Circuit:")
    print(qc)
    
    # Get IBM backend
    service = QiskitRuntimeService()
    backend = service.backend("ibm_brisbane")
    
    print(f"Using backend: {backend}")
    
    # Run the circuit
    try:
        counts = run(qc, backend=backend, shots=1000)
        print(f"Success! Got counts: {counts}")
        
        if counts and len(counts) > 0:
            print("✅ Hardware execution working!")
            print(f"Total shots: {sum(counts.values())}")
            print(f"Number of unique outcomes: {len(counts)}")
        else:
            print("❌ No counts returned")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_circuit() 