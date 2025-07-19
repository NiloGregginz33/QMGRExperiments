#!/usr/bin/env python3
"""
Debug Hardware Execution - Understand why hardware returns all zeros
"""

import sys
import os
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_ibm_runtime import QiskitRuntimeService

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Factory'))
from CGPTFactory import run

def debug_bell_state():
    """Debug Bell state execution on hardware."""
    
    print("=== DEBUGGING BELL STATE ===")
    
    # Create Bell state circuit
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    
    print("Original circuit:")
    print(qc.draw())
    
    # Get backend
    service = QiskitRuntimeService()
    backend = service.backend("ibm_brisbane")
    
    print(f"\nBackend: {backend.name}")
    print(f"Basis gates: {backend.configuration().basis_gates}")
    
    # Transpile circuit
    qc_t = transpile(qc, backend, optimization_level=0)
    print(f"\nTranspiled circuit (optimization_level=0):")
    print(qc_t.draw())
    
    # Check if transpiled circuit is different
    qc_no_measure = qc.copy()
    qc_no_measure.remove_final_measurements()
    sv_original = Statevector.from_instruction(qc_no_measure)
    
    qc_t_no_measure = qc_t.copy()
    qc_t_no_measure.remove_final_measurements()
    sv_transpiled = Statevector.from_instruction(qc_t_no_measure)
    
    print(f"\nOriginal statevector: {sv_original}")
    print(f"Transpiled statevector: {sv_transpiled}")
    print(f"Fidelity: {abs(sv_original.inner(sv_transpiled)):.6f}")
    
    # Try different optimization levels
    for opt_level in [0, 1, 2, 3]:
        print(f"\n--- Optimization Level {opt_level} ---")
        qc_opt = transpile(qc, backend, optimization_level=opt_level)
        print(f"Circuit depth: {qc_opt.depth()}")
        print(f"Gate count: {qc_opt.count_ops()}")
        
        # Simulate the transpiled circuit
        qc_opt_no_measure = qc_opt.copy()
        qc_opt_no_measure.remove_final_measurements()
        sv_opt = Statevector.from_instruction(qc_opt_no_measure)
        print(f"Statevector: {sv_opt}")
        
        # Test on hardware
        try:
            result = run(qc_opt, backend, shots=100)
            if isinstance(result, dict):
                counts = result
            else:
                counts = result.get_counts()
            
            print(f"Hardware counts: {counts}")
            
        except Exception as e:
            print(f"Hardware execution failed: {e}")

def test_simple_rotation():
    """Test a simple rotation to see if hardware responds."""
    
    print("\n=== TESTING SIMPLE ROTATION ===")
    
    qc = QuantumCircuit(1, 1)
    qc.rx(np.pi/2, 0)  # 90-degree rotation around X
    qc.measure(0, 0)
    
    print("Simple rotation circuit:")
    print(qc.draw())
    
    # Simulator
    qc_no_measure = qc.copy()
    qc_no_measure.remove_final_measurements()
    sv = Statevector.from_instruction(qc_no_measure)
    print(f"Simulator statevector: {sv}")
    
    # Hardware
    try:
        service = QiskitRuntimeService()
        backend = service.backend("ibm_brisbane")
        
        result = run(qc, backend, shots=100)
        if isinstance(result, dict):
            counts = result
        else:
            counts = result.get_counts()
        
        print(f"Hardware counts: {counts}")
        
    except Exception as e:
        print(f"Hardware execution failed: {e}")

def test_hadamard_only():
    """Test just a Hadamard gate."""
    
    print("\n=== TESTING HADAMARD ONLY ===")
    
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    
    print("Hadamard circuit:")
    print(qc.draw())
    
    # Simulator
    qc_no_measure = qc.copy()
    qc_no_measure.remove_final_measurements()
    sv = Statevector.from_instruction(qc_no_measure)
    print(f"Simulator statevector: {sv}")
    
    # Hardware
    try:
        service = QiskitRuntimeService()
        backend = service.backend("ibm_brisbane")
        
        result = run(qc, backend, shots=100)
        if isinstance(result, dict):
            counts = result
        else:
            counts = result.get_counts()
        
        print(f"Hardware counts: {counts}")
        
    except Exception as e:
        print(f"Hardware execution failed: {e}")

if __name__ == "__main__":
    debug_bell_state()
    test_simple_rotation()
    test_hadamard_only() 