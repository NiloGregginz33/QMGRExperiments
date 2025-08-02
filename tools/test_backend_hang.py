#!/usr/bin/env python3
"""
Test to check if backend detection is causing the hanging issue.
"""

import time
import sys
import os
from datetime import datetime

print(f"[{datetime.now()}] Starting backend hang test...")

# Test 1: Basic Qiskit imports
print(f"[{datetime.now()}] Test 1: Qiskit imports...")
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    print(f"[{datetime.now()}] ✅ QiskitRuntimeService imported")
except Exception as e:
    print(f"[{datetime.now()}] ❌ QiskitRuntimeService failed: {e}")

# Test 2: Service initialization with timeout
print(f"[{datetime.now()}] Test 2: Service initialization...")
start_time = time.time()

try:
    print(f"[{datetime.now()}] Creating QiskitRuntimeService...")
    service = QiskitRuntimeService(channel="ibm_cloud")
    init_time = time.time() - start_time
    print(f"[{datetime.now()}] ✅ Service created in {init_time:.2f}s")
except Exception as e:
    print(f"[{datetime.now()}] ❌ Service creation failed: {e}")
    print(f"[{datetime.now()}] This might be where the hanging occurs!")

# Test 3: Backend detection
print(f"[{datetime.now()}] Test 3: Backend detection...")
start_time = time.time()

try:
    print(f"[{datetime.now()}] Getting backends...")
    backends = service.backends()
    backend_time = time.time() - start_time
    print(f"[{datetime.now()}] ✅ Got {len(backends)} backends in {backend_time:.2f}s")
    
    # List first few backends
    for i, backend in enumerate(backends[:5]):
        print(f"[{datetime.now()}] Backend {i}: {backend.name}")
        
except Exception as e:
    print(f"[{datetime.now()}] ❌ Backend detection failed: {e}")
    print(f"[{datetime.now()}] This is likely where the hanging occurs!")

# Test 4: Best backend selection
print(f"[{datetime.now()}] Test 4: Best backend selection...")
start_time = time.time()

try:
    print(f"[{datetime.now()}] Selecting best backend...")
    
    def get_best_backend(service, min_qubits=3, max_queue=10):
        backends = service.backends()
        suitable_backends = [
            b for b in backends if b.configuration().n_qubits >= min_qubits and b.status().pending_jobs <= max_queue
        ]
        if not suitable_backends:
            print("No suitable backends found. Using default: ibm_brisbane")
            return service.backend("ibm_brisbane")
        
        best_backend = sorted(suitable_backends, key=lambda b: b.status().pending_jobs)[0]
        print(f"Best backend chosen: {best_backend.name}")
        return best_backend
    
    backend = get_best_backend(service)
    selection_time = time.time() - start_time
    print(f"[{datetime.now()}] ✅ Backend selected in {selection_time:.2f}s")
    print(f"[{datetime.now()}] Selected backend: {backend.name}")
    
except Exception as e:
    print(f"[{datetime.now()}] ❌ Backend selection failed: {e}")
    print(f"[{datetime.now()}] This is likely where the hanging occurs!")

print(f"[{datetime.now()}] Backend hang test completed!") 