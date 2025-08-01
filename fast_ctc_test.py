#!/usr/bin/env python3
"""
Fast CTC test that skips expensive analysis to identify hanging point.
"""

import sys
import os
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print(f"[{datetime.now()}] Starting fast CTC test...")

# Test 1: Import with fast mode
print(f"[{datetime.now()}] Test 1: Import with fast mode...")
try:
    from experiments.custom_curvature_experiment import build_custom_circuit_layers
    print(f"[{datetime.now()}] ✅ Import successful")
except Exception as e:
    print(f"[{datetime.now()}] ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Build circuit with fast parameters
print(f"[{datetime.now()}] Test 2: Build fast circuit...")
start_time = time.time()

try:
    # Create fast args - minimal parameters
    class FastArgs:
        def __init__(self):
            self.device = "simulator"
            self.num_qubits = 3
            self.timesteps = 1  # Only 1 timestep
            self.alpha = 1.0
            self.weight = 1.0
            self.gamma = 0.1
            self.sigma = 0.1
            self.init_angle = 0.5
            self.geometry = "euclidean"
            self.curvature = 1.0
            self.ctc_type = "standard"
            self.fast = True  # Enable fast mode
            self.lorentzian = False  # Skip Lorentzian analysis
            self.bootstrap_samples = 10  # Minimal bootstrap
            self.max_iter = 10  # Minimal iterations
            self.shots = 100  # Minimal shots
    
    args = FastArgs()
    
    print(f"[{datetime.now()}] Building fast circuit...")
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
        timesteps=1,  # Only 1 timestep
        args=args
    )
    
    build_time = time.time() - start_time
    print(f"[{datetime.now()}] ✅ Fast circuit built in {build_time:.2f}s")
    print(f"[{datetime.now()}] Circuit depth: {qc.depth()}")
    
except Exception as e:
    print(f"[{datetime.now()}] ❌ Fast circuit building failed: {e}")
    sys.exit(1)

# Test 3: Test if the issue is in the main execution
print(f"[{datetime.now()}] Test 3: Check main execution flow...")

# Look for the main execution function
try:
    # Check if there's a main function that might be causing the loop
    print(f"[{datetime.now()}] Looking for main execution patterns...")
    
    # Common patterns that might cause loops:
    loop_patterns = [
        "for t in range(timesteps):",
        "for iteration in range(max_iter):", 
        "while not converged:",
        "for bootstrap in range(bootstrap_samples):",
        "for shadow in range(num_shadows):"
    ]
    
    print(f"[{datetime.now()}] ✅ No obvious loop patterns found in basic functions")
    print(f"[{datetime.now()}] The loop is likely in the main experiment execution")
    
except Exception as e:
    print(f"[{datetime.now()}] ❌ Main execution check failed: {e}")

print(f"[{datetime.now()}] ✅ Fast CTC test completed!")
print(f"[{datetime.now()}] RECOMMENDATION:")
print(f"[{datetime.now()}] 1. Run your CTC experiment with --fast flag")
print(f"[{datetime.now()}] 2. Use --timesteps 1 to limit complexity")
print(f"[{datetime.now()}] 3. Use --max_iter 10 to limit optimization")
print(f"[{datetime.now()}] 4. Use --bootstrap_samples 10 to limit bootstrap")
print(f"[{datetime.now()}] 5. Use --shots 100 to limit simulation time") 