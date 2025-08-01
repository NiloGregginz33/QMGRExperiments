#!/usr/bin/env python3
"""
Ultimate fix for quantum execution with correct circuit wrapping
"""

import sys
import os

def fix_quantum_execution():
    """Fix quantum execution with correct circuit wrapping"""
    
    print("=== ULTIMATE QUANTUM FIX ===")
    
    # Read the file
    with open('src/experiments/custom_curvature_experiment.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the quantum execution section
    old_pattern = "sampler.run(circ, shots=args.shots)"
    new_pattern = "sampler.run([circ], shots=args.shots)"
    
    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        
        with open('src/experiments/custom_curvature_experiment.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Quantum execution fixed with correct circuit wrapping!")
        return True
    else:
        print("❌ Could not find quantum execution pattern")
        return False

def test_quantum_execution():
    """Test if quantum execution works now"""
    
    print("\n=== TESTING QUANTUM EXECUTION ===")
    
    try:
        from qiskit import QuantumCircuit
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit.primitives import BackendSamplerV2
        
        print("✅ Correct imports successful")
        
        # Create test circuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        # Test execution with correct wrapping
        service = QiskitRuntimeService()
        backend = service.backend("ibm_brisbane")
        sampler = BackendSamplerV2(backend=backend)
        
        job = sampler.run([qc], shots=100)  # Note the list wrapping
        result = job.result()
        
        print("✅ Quantum execution successful!")
        print(f"✅ Got counts: {result.get_counts()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Still failing: {e}")
        return False

if __name__ == "__main__":
    # Apply fix
    fix_quantum_execution()
    
    # Test
    test_quantum_execution() 