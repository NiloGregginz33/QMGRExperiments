#!/usr/bin/env python3
"""
Debug script to analyze Region B decoder failure.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from quantum.rt_surface_oracle import RTSurfaceOracle, create_boundary_graph
from quantum.happy_code import HaPPYCode
from quantum.region_decoders import DecoderSynthesizer

def debug_region_b():
    """Debug Region B decoder failure."""
    print("=== DEBUGGING REGION B DECODER ===")
    
    # Initialize components
    boundary_graph = create_boundary_graph(12, 'linear')
    rt_oracle = RTSurfaceOracle(boundary_graph)
    happy_code = HaPPYCode(num_boundary_qubits=12)
    
    # Define regions
    regions = {
        'A': [0, 1, 2, 3],
        'B': [4, 5, 6, 7],
        'C': [8, 9, 10, 11]
    }
    
    # Compute RT surface results
    rt_results = rt_oracle.compute_all_rt_surfaces(regions, bulk_node=6)
    print(f"RT Results: {rt_results}")
    
    # Create decoder synthesizer
    synthesizer = DecoderSynthesizer(happy_code)
    decoders = synthesizer.synthesize_region_decoders(regions, rt_results)
    
    # Get Region B decoder
    region_b_decoder = decoders['B']
    print(f"\nRegion B Decoder Info:")
    info = region_b_decoder.get_decoder_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Create encoding circuit
    encoding_circuit = happy_code.create_encoding_circuit(logical_state='1')
    print(f"\nEncoding Circuit Depth: {encoding_circuit.depth()}")
    
    # Get the encoded state
    from qiskit.quantum_info import Statevector
    encoded_state = Statevector.from_instruction(encoding_circuit)
    print(f"Encoded State Size: {len(encoded_state.data)}")
    
    # Get the decoder preparation circuit
    prep_circuit = region_b_decoder.decoder_circuit['preparation']
    print(f"Decoder Preparation Circuit Depth: {prep_circuit.depth()}")
    print(f"Decoder Preparation Circuit Qubits: {prep_circuit.num_qubits}")
    
    # Analyze the decoder circuit
    print(f"\nDecoder Circuit Analysis:")
    print(f"  Region qubits: {region_b_decoder.region_qubits}")
    print(f"  RT contains bulk: {region_b_decoder.rt_contains_bulk}")
    
    # Check if the decoder is actually effective
    if region_b_decoder.rt_contains_bulk:
        print("  ✓ Region B should succeed (RT surface contains bulk)")
    else:
        print("  ✗ Region B should fail (RT surface doesn't contain bulk)")
    
    # Test the decoder with a simple state
    print(f"\nTesting Decoder with Simple State:")
    
    # Create a simple test state (all qubits in |0⟩)
    from qiskit import QuantumCircuit
    test_circuit = QuantumCircuit(prep_circuit.num_qubits)
    
    # Apply the decoder preparation
    combined_circuit = test_circuit.compose(prep_circuit)
    
    # Get the final state
    final_state = Statevector.from_instruction(combined_circuit)
    
    # Calculate success probability (probability of measuring |1⟩ on output qubit)
    output_qubit = prep_circuit.num_qubits - 1  # Last qubit is output
    prob_1 = 0.0
    
    for i, amp in enumerate(final_state.data):
        binary = format(i, f'0{int(np.log2(len(final_state.data)))}b')
        if binary[output_qubit] == '1':  # Output qubit is |1⟩
            prob_1 += abs(amp)**2
    
    print(f"  Success probability with |0⟩ input: {prob_1:.4f}")
    
    # Test with encoded state
    print(f"\nTesting Decoder with Encoded State:")
    
    # Create a proper test by combining encoding + decoding
    # We need to ensure the encoded state is properly mapped to the decoder
    
    # Create a circuit that combines encoding + decoding
    total_qubits = max(encoding_circuit.num_qubits, prep_circuit.num_qubits)
    combined_circuit = QuantumCircuit(total_qubits)
    
    # Apply encoding to the first 12 qubits
    combined_circuit.compose(encoding_circuit, qubits=list(range(12)), inplace=True)
    
    # Apply decoder preparation to the full circuit
    combined_circuit.compose(prep_circuit, inplace=True)
    
    # Get final state
    final_state = Statevector.from_instruction(combined_circuit)
    
    # Calculate success probability
    output_qubit = prep_circuit.num_qubits - 1
    prob_1 = 0.0
    
    for i, amp in enumerate(final_state.data):
        binary = format(i, f'0{int(np.log2(len(final_state.data)))}b')
        if binary[output_qubit] == '1':  # Output qubit is |1⟩
            prob_1 += abs(amp)**2
    
    print(f"  Success probability with encoded input: {prob_1:.4f}")
    
    # Also test with logical |0⟩ state
    print(f"\nTesting Decoder with Logical |0⟩ State:")
    encoding_circuit_0 = happy_code.create_encoding_circuit(logical_state='0')
    
    combined_circuit_0 = QuantumCircuit(total_qubits)
    combined_circuit_0.compose(encoding_circuit_0, qubits=list(range(12)), inplace=True)
    combined_circuit_0.compose(prep_circuit, inplace=True)
    
    final_state_0 = Statevector.from_instruction(combined_circuit_0)
    
    prob_1_0 = 0.0
    for i, amp in enumerate(final_state_0.data):
        binary = format(i, f'0{int(np.log2(len(final_state_0.data)))}b')
        if binary[output_qubit] == '1':  # Output qubit is |1⟩
            prob_1_0 += abs(amp)**2
    
    print(f"  Success probability with logical |0⟩ input: {prob_1_0:.4f}")
    
    # Calculate the difference (should be large for effective decoder)
    difference = abs(prob_1 - prob_1_0)
    print(f"  Difference between |1⟩ and |0⟩: {difference:.4f}")
    
    if difference > 0.3:
        print("  ✓ Decoder is distinguishing between logical states!")
    else:
        print("  ✗ Decoder is not distinguishing between logical states")
    
    return prob_1

if __name__ == "__main__":
    success_prob = debug_region_b()
    print(f"\n=== FINAL RESULT ===")
    print(f"Region B Success Probability: {success_prob:.4f}")
    if success_prob > 0.5:
        print("✓ Region B decoder is working correctly!")
    else:
        print("✗ Region B decoder needs fixing!") 