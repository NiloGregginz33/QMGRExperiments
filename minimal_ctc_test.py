#!/usr/bin/env python3
"""
Minimal CTC test to identify where hanging occurs.
"""

import sys
import os
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print(f"[{datetime.now()}] Starting minimal CTC test...")

# Test 1: Import and basic setup
print(f"[{datetime.now()}] Test 1: Import and setup...")
try:
    from experiments.custom_curvature_experiment import build_custom_circuit_layers
    print(f"[{datetime.now()}] ✅ Import successful")
except Exception as e:
    print(f"[{datetime.now()}] ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Build minimal circuit
print(f"[{datetime.now()}] Test 2: Build minimal circuit...")
start_time = time.time()

try:
    # Create minimal args
    class MockArgs:
        def __init__(self):
            self.device = "simulator"
            self.num_qubits = 3
            self.timesteps = 1
            self.alpha = 1.0
            self.weight = 1.0
            self.gamma = 0.1
            self.sigma = 0.1
            self.init_angle = 0.5
            self.geometry = "euclidean"
            self.curvature = 1.0
            self.ctc_type = "standard"
            self.fast = True  # Enable fast mode
    
    args = MockArgs()
    
    print(f"[{datetime.now()}] Building circuit...")
    circuits, qc = build_custom_circuit_layers(
        num_qubits=3,
        topology="ring",
        custom_edges=None,
        alpha=1.0,
        weight=1.0,
        gamma=0.1,
        sigma=0.1,
        init_angle=0.5,
        geometry="euclidean",
        curvature=1.0,
        timesteps=1,
        args=args
    )
    
    build_time = time.time() - start_time
    print(f"[{datetime.now()}] ✅ Circuit built in {build_time:.2f}s")
    print(f"[{datetime.now()}] Circuit depth: {qc.depth()}")
    
except Exception as e:
    print(f"[{datetime.now()}] ❌ Circuit building failed: {e}")
    sys.exit(1)

# Test 3: Run minimal simulation
print(f"[{datetime.now()}] Test 3: Run minimal simulation...")
start_time = time.time()

try:
    from qiskit import QuantumCircuit
    from qiskit.primitives import Sampler
    
    print(f"[{datetime.now()}] Running simulation...")
    
    # Use a very simple circuit for testing
    test_qc = QuantumCircuit(3)
    test_qc.h([0, 1, 2])
    test_qc.cx(0, 1)
    test_qc.cx(1, 2)
    test_qc.measure_all()
    
    sampler = Sampler()
    job = sampler.run(test_qc, shots=10)
    result = job.result()
    
    sim_time = time.time() - start_time
    print(f"[{datetime.now()}] ✅ Simulation completed in {sim_time:.2f}s")
    print(f"[{datetime.now()}] Result: {result.quasi_dists[0]}")
    
except Exception as e:
    print(f"[{datetime.now()}] ❌ Simulation failed: {e}")
    sys.exit(1)

# Test 4: Test CTC-specific functions
print(f"[{datetime.now()}] Test 4: Test CTC functions...")
start_time = time.time()

try:
    from experiments.custom_curvature_experiment import _apply_ctc_circuit_structure
    
    print(f"[{datetime.now()}] Testing CTC circuit structure...")
    
    test_qc = QuantumCircuit(3)
    _apply_ctc_circuit_structure(test_qc, 3, ctc_type="standard")
    
    ctc_time = time.time() - start_time
    print(f"[{datetime.now()}] ✅ CTC structure applied in {ctc_time:.2f}s")
    print(f"[{datetime.now()}] CTC circuit depth: {test_qc.depth()}")
    
except Exception as e:
    print(f"[{datetime.now()}] ❌ CTC function failed: {e}")
    sys.exit(1)

print(f"[{datetime.now()}] ✅ All minimal CTC tests passed!")
print(f"[{datetime.now()}] The hanging issue is likely in the main experiment execution, not in these basic functions.") 