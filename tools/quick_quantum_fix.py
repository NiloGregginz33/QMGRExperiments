#!/usr/bin/env python3
"""
Quick fix for quantum execution import issue
"""

import sys
import os

def fix_quantum_imports():
    """Fix the quantum execution imports"""
    
    print("=== QUICK QUANTUM FIX ===")
    
    # Read the file
    with open('src/experiments/custom_curvature_experiment.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the import issue
    old_import = "from qiskit.primitives import Sampler"
    new_import = "from qiskit.primitives import Sampler as QiskitSampler"
    
    # Replace the import
    if old_import in content:
        content = content.replace(old_import, new_import)
        
        # Also fix the usage
        content = content.replace("sampler = Sampler(backend=backend)", "sampler = QiskitSampler(backend=backend)")
        
        # Write back
        with open('src/experiments/custom_curvature_experiment.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Quantum imports fixed!")
        return True
    else:
        print("❌ Could not find import to fix")
        return False

def test_quantum_execution():
    """Test if quantum execution works now"""
    
    print("\n=== TESTING QUANTUM EXECUTION ===")
    
    try:
        from qiskit import QuantumCircuit
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit.primitives import Sampler as QiskitSampler
        
        print("✅ Fixed imports successful")
        
        # Create test circuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        # Test execution
        service = QiskitRuntimeService()
        backend = service.backend("ibm_brisbane")
        sampler = QiskitSampler(backend=backend)
        
        job = sampler.run(qc, shots=100)
        result = job.result()
        
        print("✅ Quantum execution successful!")
        print(f"✅ Got counts: {result.get_counts()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Still failing: {e}")
        return False

if __name__ == "__main__":
    # Apply fix
    fix_quantum_imports()
    
    # Test
    test_quantum_execution() 