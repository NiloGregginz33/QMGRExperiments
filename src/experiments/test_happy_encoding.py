#!/usr/bin/env python3
"""
Test script to understand HaPPY encoding.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from quantum.happy_code import HaPPYCode
from qiskit.quantum_info import Statevector

def test_happy_encoding():
    """Test HaPPY encoding to understand how it works."""
    print("=== TESTING HAPPY ENCODING ===")
    
    # Create HaPPY code
    happy_code = HaPPYCode(num_boundary_qubits=12)
    
    # Get logical operators
    logical_x = happy_code.logical_x
    logical_z = happy_code.logical_z
    
    print(f"Logical X: {logical_x.to_label()}")
    print(f"Logical Z: {logical_z.to_label()}")
    
    # Create encoding circuits
    encoding_0 = happy_code.create_encoding_circuit(logical_state='0')
    encoding_1 = happy_code.create_encoding_circuit(logical_state='1')
    
    print(f"Encoding |0⟩ circuit depth: {encoding_0.depth()}")
    print(f"Encoding |1⟩ circuit depth: {encoding_1.depth()}")
    
    # Get encoded states
    state_0 = Statevector.from_instruction(encoding_0)
    state_1 = Statevector.from_instruction(encoding_1)
    
    print(f"State |0⟩ size: {len(state_0.data)}")
    print(f"State |1⟩ size: {len(state_1.data)}")
    
    # Check if states are different
    overlap = np.abs(np.dot(state_0.data.conj(), state_1.data))
    print(f"Overlap between |0⟩ and |1⟩: {overlap:.6f}")
    
    # Analyze the states
    print(f"\nState Analysis:")
    print(f"|0⟩ state norm: {np.linalg.norm(state_0.data):.6f}")
    print(f"|1⟩ state norm: {np.linalg.norm(state_1.data):.6f}")
    
    # Check specific qubits
    print(f"\nQubit Analysis:")
    for qubit in [0, 4, 8]:  # Logical X qubits
        prob_0 = 0.0
        prob_1 = 0.0
        
        for i, amp in enumerate(state_0.data):
            binary = format(i, '012b')
            if binary[qubit] == '1':
                prob_0 += abs(amp)**2
        
        for i, amp in enumerate(state_1.data):
            binary = format(i, '012b')
            if binary[qubit] == '1':
                prob_1 += abs(amp)**2
        
        print(f"Qubit {qubit}: |0⟩ prob |1⟩ = {prob_0:.4f}, |1⟩ prob |1⟩ = {prob_1:.4f}")
    
    # Check Region B qubits
    print(f"\nRegion B Analysis (qubits 4,5,6,7):")
    for qubit in [4, 5, 6, 7]:
        prob_0 = 0.0
        prob_1 = 0.0
        
        for i, amp in enumerate(state_0.data):
            binary = format(i, '012b')
            if binary[qubit] == '1':
                prob_0 += abs(amp)**2
        
        for i, amp in enumerate(state_1.data):
            binary = format(i, '012b')
            if binary[qubit] == '1':
                prob_1 += abs(amp)**2
        
        print(f"Qubit {qubit}: |0⟩ prob |1⟩ = {prob_0:.4f}, |1⟩ prob |1⟩ = {prob_1:.4f}")
    
    return state_0, state_1

if __name__ == "__main__":
    state_0, state_1 = test_happy_encoding()
    print(f"\n=== SUMMARY ===")
    print("HaPPY encoding test completed!") 