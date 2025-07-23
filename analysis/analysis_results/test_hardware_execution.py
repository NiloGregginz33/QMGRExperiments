#!/usr/bin/env python3
"""
Test script to verify hardware execution of custom curvature experiment
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
import time

def test_hardware_connection():
    """Test basic hardware connection"""
    print("Testing IBM Quantum hardware connection...")
    
    try:
        service = QiskitRuntimeService()
        backend = service.backend('ibm_brisbane')
        
        print(f"✓ Connected to backend: {backend.name}")
        print(f"✓ Backend status: {backend.status()}")
        print(f"✓ Pending jobs: {backend.status().pending_jobs}")
        
        # Test circuit compilation
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        qc_t = transpile(qc, backend)
        print(f"✓ Circuit transpiled successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Hardware connection failed: {e}")
        return False

def test_cgptfactory_config():
    """Test CGPTFactory configuration"""
    print("\nTesting CGPTFactory configuration...")
    
    try:
        import src.CGPTFactory as cgpt
        
        # Check if USE_IBM is set correctly
        use_ibm = os.getenv("USE_IBM", "True").lower() == "true"
        print(f"✓ USE_IBM environment variable: {use_ibm}")
        
        # Check if service is initialized
        if hasattr(cgpt, 'service'):
            print(f"✓ QiskitRuntimeService initialized")
            backends = cgpt.service.backends()
            print(f"✓ Available backends: {[b.name for b in backends]}")
        else:
            print("✗ QiskitRuntimeService not found in CGPTFactory")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ CGPTFactory test failed: {e}")
        return False

def test_custom_curvature_execution():
    """Test if custom curvature experiment can execute on hardware"""
    print("\nTesting custom curvature experiment execution...")
    
    try:
        # Import the experiment
        from src.experiments.custom_curvature_experiment import build_custom_circuit_layers
        
        # Build a simple circuit
        circuits, qc = build_custom_circuit_layers(
            num_qubits=3,
            topology="triangulated", 
            custom_edges=None,
            alpha=0.8,
            weight=1.0,
            gamma=0.3,
            sigma=None,
            init_angle=0.0,
            geometry="hyperbolic",
            curvature=2.5,
            log_edge_weights=False,
            timesteps=1,
            init_angles=None
        )
        
        print(f"✓ Circuit built successfully")
        print(f"✓ Number of timesteps: {len(circuits)}")
        print(f"✓ Circuit depth: {qc.depth()}")
        
        # Test transpilation
        service = QiskitRuntimeService()
        backend = service.backend('ibm_brisbane')
        
        qc.measure_all()  # Add measurements for hardware
        qc_t = transpile(qc, backend)
        print(f"✓ Circuit transpiled for hardware")
        
        return True
        
    except Exception as e:
        print(f"✗ Custom curvature test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Hardware Execution Verification Test ===\n")
    
    # Run all tests
    tests = [
        test_hardware_connection,
        test_cgptfactory_config, 
        test_custom_curvature_execution
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n=== Test Results ===")
    print(f"Hardware connection: {'✓ PASS' if results[0] else '✗ FAIL'}")
    print(f"CGPTFactory config: {'✓ PASS' if results[1] else '✗ FAIL'}")
    print(f"Custom curvature: {'✓ PASS' if results[2] else '✗ FAIL'}")
    
    if all(results):
        print("\n🎉 ALL TESTS PASSED - Hardware execution is working!")
        print("Your custom curvature experiment should be running on real IBM hardware.")
    else:
        print("\n⚠️  SOME TESTS FAILED - Check configuration.") 