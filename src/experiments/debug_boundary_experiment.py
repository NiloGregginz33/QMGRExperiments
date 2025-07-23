#!/usr/bin/env python3
"""
Debug script for boundary vs bulk entropy experiment
"""

import sys
import os
sys.path.append(os.path.abspath('.'))
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

def test_circuit_building():
    """Test if the circuit can be built"""
    print("=== Testing Circuit Building ===")
    try:
        qc = QuantumCircuit(6, 6)
        
        # Create 3 GHZ pairs: (0,1), (2,3), (4,5)
        for i in [0, 2, 4]:
            qc.h(i)
            qc.cx(i, i+1)
        
        # Entangle across pairs (CZ gates)
        qc.cz(0, 2)
        qc.cz(1, 4)
        qc.cz(3, 5)
        
        # Optional RX rotation to break symmetry
        for q in range(6):
            qc.rx(np.pi / 4, q)
        
        # Add measurements
        qc.measure_all()
        
        print("✓ Circuit built successfully")
        print(f"Circuit depth: {qc.depth()}")
        print(f"Circuit size: {qc.num_qubits} qubits, {qc.num_clbits} classical bits")
        return qc
    except Exception as e:
        print(f"✗ Circuit building failed: {e}")
        return None

def test_backend_connection():
    """Test if we can connect to IBM backends"""
    print("\n=== Testing Backend Connection ===")
    try:
        service = QiskitRuntimeService(channel="ibm_cloud")
        print("✓ QiskitRuntimeService created successfully")
        
        # List available backends
        backends = service.backends()
        print(f"Available backends: {len(backends)}")
        for backend in backends[:5]:  # Show first 5
            print(f"  - {backend.name}")
        
        # Try to get IBM Torino specifically
        try:
            torino = service.backend("ibm_torino")
            print(f"✓ IBM Torino backend found: {torino.name}")
            print(f"  Status: {torino.status()}")
            print(f"  Pending jobs: {torino.status().pending_jobs}")
            return torino
        except Exception as e:
            print(f"✗ IBM Torino not found: {e}")
            return None
            
    except Exception as e:
        print(f"✗ Backend connection failed: {e}")
        return None

def test_transpilation(qc, backend):
    """Test if the circuit can be transpiled"""
    print("\n=== Testing Circuit Transpilation ===")
    try:
        tqc = transpile(qc, backend, optimization_level=3)
        print("✓ Circuit transpiled successfully")
        print(f"Transpiled depth: {tqc.depth()}")
        print(f"Basis gates: {backend.operation_names}")
        return tqc
    except Exception as e:
        print(f"✗ Transpilation failed: {e}")
        return None

def test_cgpt_factory():
    """Test if CGPTFactory can be imported and used"""
    print("\n=== Testing CGPTFactory ===")
    try:
        from src.CGPTFactory import run as cgpt_run
        print("✓ CGPTFactory imported successfully")
        return cgpt_run
    except Exception as e:
        print(f"✗ CGPTFactory import failed: {e}")
        return None

def test_simulator_run():
    """Test if the experiment works on simulator"""
    print("\n=== Testing Simulator Run ===")
    try:
        qc = test_circuit_building()
        if qc is None:
            return False
            
        backend = FakeBrisbane()
        print(f"✓ Using FakeBrisbane simulator: {backend.name}")
        
        tqc = transpile(qc, backend, optimization_level=3)
        print("✓ Circuit transpiled for simulator")
        
        # Test with CGPTFactory
        cgpt_run = test_cgpt_factory()
        if cgpt_run is None:
            return False
            
        counts = cgpt_run(tqc, backend=backend, shots=100)
        print("✓ Simulator execution successful")
        print(f"Counts: {dict(counts)}")
        return True
        
    except Exception as e:
        print(f"✗ Simulator run failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Debugging Boundary vs Bulk Entropy Experiment")
    print("=" * 50)
    
    # Test 1: Circuit building
    qc = test_circuit_building()
    
    # Test 2: Backend connection
    backend = test_backend_connection()
    
    # Test 3: Transpilation (if backend available)
    if qc and backend:
        tqc = test_transpilation(qc, backend)
    
    # Test 4: CGPTFactory
    cgpt_run = test_cgpt_factory()
    
    # Test 5: Simulator run
    simulator_success = test_simulator_run()
    
    print("\n=== Summary ===")
    print(f"Circuit building: {'✓' if qc else '✗'}")
    print(f"Backend connection: {'✓' if backend else '✗'}")
    print(f"CGPTFactory: {'✓' if cgpt_run else '✗'}")
    print(f"Simulator run: {'✓' if simulator_success else '✗'}")
    
    if not simulator_success:
        print("\n❌ The experiment will likely fail on hardware if it fails on simulator")
    else:
        print("\n✅ Basic functionality works, hardware issues may be related to:")
        print("  - IBM Torino availability/queue")
        print("  - Circuit complexity for hardware")
        print("  - Authentication/permissions")

if __name__ == "__main__":
    main() 