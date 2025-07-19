#!/usr/bin/env python3
"""
Simple Hardware Test - Verify IBM Quantum hardware execution
"""

import sys
import os
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_ibm_runtime import QiskitRuntimeService

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Factory'))
from CGPTFactory import run

def test_simple_circuit():
    """Test a very simple circuit on hardware."""
    
    print("Testing simple hardware execution...")
    
    # Create a simple 2-qubit circuit
    qc = QuantumCircuit(2, 2)
    
    # Simple operations that should give non-zero results
    qc.h(0)  # Hadamard on qubit 0
    qc.cx(0, 1)  # CNOT from 0 to 1
    qc.h(1)  # Hadamard on qubit 1
    
    qc.measure_all()
    
    print("Circuit:")
    print(qc.draw())
    
    # Test on simulator first
    print("\n=== SIMULATOR TEST ===")
    # Remove measurements for statevector calculation
    qc_no_measure = qc.copy()
    qc_no_measure.remove_final_measurements()
    sv = Statevector.from_instruction(qc_no_measure)
    probs = np.abs(sv.data) ** 2
    print("Statevector probabilities:")
    for i, p in enumerate(probs):
        if p > 0.01:
            print(f"  |{i:02b}>: {p:.4f}")
    
    # Test on hardware
    print("\n=== HARDWARE TEST ===")
    try:
        service = QiskitRuntimeService()
        backend = service.backend("ibm_brisbane")
        
        result = run(qc, backend, shots=1000)
        print(f"Hardware result type: {type(result)}")
        
        if isinstance(result, dict):
            counts = result
        elif hasattr(result, 'get_counts'):
            counts = result.get_counts()
        else:
            print(f"Unexpected result format: {result}")
            return
        
        print("Hardware counts:")
        for bitstring, count in counts.items():
            print(f"  {bitstring}: {count}")
        
        # Calculate entropy
        total = sum(counts.values())
        probs = {k: v/total for k, v in counts.items()}
        entropy = -sum(p * np.log2(p) for p in probs.values() if p > 0)
        print(f"Hardware entropy: {entropy:.4f}")
        
    except Exception as e:
        print(f"Hardware execution failed: {e}")

def test_bell_state():
    """Test a Bell state circuit."""
    
    print("\n=== BELL STATE TEST ===")
    
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    
    print("Bell state circuit:")
    print(qc.draw())
    
    # Simulator
    print("\nSimulator result:")
    qc_no_measure = qc.copy()
    qc_no_measure.remove_final_measurements()
    sv = Statevector.from_instruction(qc_no_measure)
    probs = np.abs(sv.data) ** 2
    for i, p in enumerate(probs):
        if p > 0.01:
            print(f"  |{i:02b}>: {p:.4f}")
    
    # Hardware
    print("\nHardware result:")
    try:
        service = QiskitRuntimeService()
        backend = service.backend("ibm_brisbane")
        
        result = run(qc, backend, shots=1000)
        
        if isinstance(result, dict):
            counts = result
        elif hasattr(result, 'get_counts'):
            counts = result.get_counts()
        else:
            print(f"Unexpected result format: {result}")
            return
        
        for bitstring, count in counts.items():
            print(f"  {bitstring}: {count}")
        
    except Exception as e:
        print(f"Hardware execution failed: {e}")

if __name__ == "__main__":
    test_simple_circuit()
    test_bell_state() 