#!/usr/bin/env python3
"""
Test script for CTC functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from CGPTFactory import run_torus_ctc
import numpy as np

# Test parameters
N = 3
phase_profile = np.full((N, N), 0.1)  # Uniform phase profile
backend_name = None  # Use simulator
shots = 1000

print(f"🧪 Testing CTC functionality")
print(f"📊 Torus size: {N}x{N}")
print(f"📊 Phase profile shape: {phase_profile.shape}")
print(f"📊 Backend: {backend_name}")

try:
    # Get the circuit from CGPTFactory
    print(f"🔄 Calling run_torus_ctc...")
    ctc_circuit = run_torus_ctc(N, phase_profile, backend_name=backend_name, shots=shots)
    
    print(f"✅ CTC circuit created successfully!")
    print(f"📊 Circuit depth: {ctc_circuit.depth()}")
    print(f"📊 Number of gates: {len(ctc_circuit.data)}")
    print(f"📊 Number of qubits: {ctc_circuit.num_qubits}")
    print(f"📊 Number of classical bits: {ctc_circuit.num_clbits}")
    
    # Display the circuit
    print(f"\n🔍 Circuit structure:")
    print(ctc_circuit)
    
    print(f"\n🎉 CTC test completed successfully!")
    print(f"📝 The circuit is ready to be executed with your preferred backend")
    
except Exception as e:
    print(f"❌ CTC test failed: {e}")
    import traceback
    traceback.print_exc() 