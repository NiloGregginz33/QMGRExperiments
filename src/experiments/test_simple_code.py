#!/usr/bin/env python3
"""
Test script for simple stabilizer code.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from quantum.simple_stabilizer_code import SimpleStabilizerCode
from qiskit.quantum_info import Statevector

def test_simple_code():
    """Test simple stabilizer code."""
    print("=== TESTING SIMPLE STABILIZER CODE ===")
    
    # Create simple stabilizer code
    code = SimpleStabilizerCode(num_qubits=4)
    
    # Get logical operators
    logical_x = code.logical_x
    logical_z = code.logical_z
    
    print(f"Logical X: {logical_x.to_label()}")
    print(f"Logical Z: {logical_z.to_label()}")
    
    # Create encoding circuits
    encoding_0 = code.create_encoding_circuit(logical_state='0')
    encoding_1 = code.create_encoding_circuit(logical_state='1')
    
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
    
    # Test decoders
    print(f"\nTesting Decoders:")
    
    # Test effective decoder (region contains bulk)
    effective_decoder = code.create_decoder_circuit(region_qubits=[0, 1], rt_contains_bulk=True)
    
    # Test with encoded states
    combined_0 = encoding_0.compose(effective_decoder)
    combined_1 = encoding_1.compose(effective_decoder)
    
    state_0_decoded = Statevector.from_instruction(combined_0)
    state_1_decoded = Statevector.from_instruction(combined_1)
    
    # Calculate success probabilities
    prob_0 = 0.0
    prob_1 = 0.0
    
    for i, amp in enumerate(state_0_decoded.data):
        binary = format(i, '05b')  # 4 qubits + 1 output
        if binary[0] == '1':  # Output qubit is |1⟩
            prob_0 += abs(amp)**2
    
    for i, amp in enumerate(state_1_decoded.data):
        binary = format(i, '05b')  # 4 qubits + 1 output
        if binary[0] == '1':  # Output qubit is |1⟩
            prob_1 += abs(amp)**2
    
    print(f"Effective decoder:")
    print(f"  |0⟩ success probability: {prob_0:.4f}")
    print(f"  |1⟩ success probability: {prob_1:.4f}")
    print(f"  Difference: {abs(prob_1 - prob_0):.4f}")
    
    # Test ineffective decoder (region doesn't contain bulk)
    ineffective_decoder = code.create_decoder_circuit(region_qubits=[0, 1], rt_contains_bulk=False)
    
    combined_0_ineff = encoding_0.compose(ineffective_decoder)
    combined_1_ineff = encoding_1.compose(ineffective_decoder)
    
    state_0_ineff = Statevector.from_instruction(combined_0_ineff)
    state_1_ineff = Statevector.from_instruction(combined_1_ineff)
    
    # Calculate success probabilities
    prob_0_ineff = 0.0
    prob_1_ineff = 0.0
    
    for i, amp in enumerate(state_0_ineff.data):
        binary = format(i, '05b')
        if binary[0] == '1':
            prob_0_ineff += abs(amp)**2
    
    for i, amp in enumerate(state_1_ineff.data):
        binary = format(i, '05b')
        if binary[0] == '1':
            prob_1_ineff += abs(amp)**2
    
    print(f"Ineffective decoder:")
    print(f"  |0⟩ success probability: {prob_0_ineff:.4f}")
    print(f"  |1⟩ success probability: {prob_1_ineff:.4f}")
    print(f"  Difference: {abs(prob_1_ineff - prob_0_ineff):.4f}")
    
    return {
        'effective_diff': abs(prob_1 - prob_0),
        'ineffective_diff': abs(prob_1_ineff - prob_0_ineff)
    }

if __name__ == "__main__":
    results = test_simple_code()
    print(f"\n=== SUMMARY ===")
    print(f"Effective decoder difference: {results['effective_diff']:.4f}")
    print(f"Ineffective decoder difference: {results['ineffective_diff']:.4f}")
    
    if results['effective_diff'] > 0.3:
        print("✓ Effective decoder is working!")
    else:
        print("✗ Effective decoder needs fixing")
    
    if results['ineffective_diff'] < 0.1:
        print("✓ Ineffective decoder is working!")
    else:
        print("✗ Ineffective decoder needs fixing") 