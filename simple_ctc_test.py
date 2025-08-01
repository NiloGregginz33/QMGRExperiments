#!/usr/bin/env python3
"""
Simple test to isolate where CTC experiment hangs.
"""

import sys
import os
import time
from datetime import datetime

print(f"[{datetime.now()}] Starting simple CTC test...")

# Test 1: Basic imports
print(f"[{datetime.now()}] Test 1: Basic imports...")
try:
    import numpy as np
    print(f"[{datetime.now()}] ✅ numpy imported")
except Exception as e:
    print(f"[{datetime.now()}] ❌ numpy failed: {e}")

try:
    import qiskit
    print(f"[{datetime.now()}] ✅ qiskit imported")
except Exception as e:
    print(f"[{datetime.now()}] ❌ qiskit failed: {e}")

# Test 2: Add src to path
print(f"[{datetime.now()}] Test 2: Adding src to path...")
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
print(f"[{datetime.now()}] ✅ src added to path")

# Test 3: Try importing specific modules
print(f"[{datetime.now()}] Test 3: Importing specific modules...")

try:
    from experiments import custom_curvature_experiment
    print(f"[{datetime.now()}] ✅ custom_curvature_experiment imported")
except Exception as e:
    print(f"[{datetime.now()}] ❌ custom_curvature_experiment failed: {e}")
    print(f"[{datetime.now()}] This is likely where the hanging occurs!")

# Test 4: If we get here, try importing specific functions
print(f"[{datetime.now()}] Test 4: Importing specific functions...")

try:
    from experiments.custom_curvature_experiment import build_custom_circuit_layers
    print(f"[{datetime.now()}] ✅ build_custom_circuit_layers imported")
except Exception as e:
    print(f"[{datetime.now()}] ❌ build_custom_circuit_layers failed: {e}")

print(f"[{datetime.now()}] Simple test completed!") 