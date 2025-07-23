#!/usr/bin/env python3
"""
Debug script for simple EWR experiment.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from quantum.rt_surface_oracle import RTSurfaceOracle, create_boundary_graph
from quantum.simple_stabilizer_code import SimpleStabilizerCode
from qiskit.quantum_info import Statevector

def debug_simple_ewr():
    """Debug simple EWR experiment."""
    print("=== DEBUGGING SIMPLE EWR ===")
    
    # Initialize components
    boundary_graph = create_boundary_graph(4, 'linear')
    rt_oracle = RTSurfaceOracle(boundary_graph)
    code = SimpleStabilizerCode(num_qubits=4)
    
    # Define regions
    regions = {
        'A': [0, 1],
        'B': [2, 3]
    }
    
    # Compute RT surface results
    rt_results = rt_oracle.compute_all_rt_surfaces(regions, bulk_node=2)
    print(f"RT Results: {rt_results}")
    
    # Test each region
    for region_name, region_qubits in regions.items():
        print(f"\nTesting Region {region_name} (qubits {region_qubits}):")
        print(f"  RT contains bulk: {rt_results[region_name]}")
        
        # Create decoder
        decoder = code.create_decoder_circuit(
            region_qubits=region_qubits,
            rt_contains_bulk=rt_results[region_name]
        )
        
        print(f"  Decoder circuit depth: {decoder.depth()}")
        print(f"  Decoder circuit: {decoder}")
        
        # Test with logical |0⟩ and |1⟩ states
        encoding_0 = code.create_encoding_circuit(logical_state='0')
        encoding_1 = code.create_encoding_circuit(logical_state='1')
        
        # Combine encoding + decoding
        combined_0 = encoding_0.compose(decoder)
        combined_1 = encoding_1.compose(decoder)
        
        # Get final states (before measurement)
        # Remove measurement from circuit for statevector simulation
        prep_0 = combined_0.copy()
        prep_1 = combined_1.copy()
        
        # Remove the measurement instruction
        if prep_0.data[-1][0].name == 'measure':
            prep_0.data.pop()
        if prep_1.data[-1][0].name == 'measure':
            prep_1.data.pop()
        
        state_0 = Statevector.from_instruction(prep_0)
        state_1 = Statevector.from_instruction(prep_1)
        
        # Calculate success probabilities
        prob_0 = 0.0
        prob_1 = 0.0
        
        print(f"  State |0⟩ size: {len(state_0.data)}")
        print(f"  State |1⟩ size: {len(state_1.data)}")
        
        # Check if there are any non-zero amplitudes
        non_zero_0 = np.count_nonzero(state_0.data)
        non_zero_1 = np.count_nonzero(state_1.data)
        print(f"  Non-zero amplitudes |0⟩: {non_zero_0}")
        print(f"  Non-zero amplitudes |1⟩: {non_zero_1}")
        
        # Print first few amplitudes
        print(f"  First 5 amplitudes |0⟩: {state_0.data[:5]}")
        print(f"  First 5 amplitudes |1⟩: {state_1.data[:5]}")
        
        # Check all qubits for measurement
        for qubit in range(4):
            prob_0_qubit = 0.0
            prob_1_qubit = 0.0
            
            for i, amp in enumerate(state_0.data):
                binary = format(i, '04b')  # 4 qubits
                if binary[qubit] == '1':
                    prob_0_qubit += abs(amp)**2
            
            for i, amp in enumerate(state_1.data):
                binary = format(i, '04b')  # 4 qubits
                if binary[qubit] == '1':
                    prob_1_qubit += abs(amp)**2
            
            print(f"  Qubit {qubit}: |0⟩ prob |1⟩ = {prob_0_qubit:.4f}, |1⟩ prob |1⟩ = {prob_1_qubit:.4f}")
        
        # Always measure qubit 0 (the output qubit in the circuit)
        output_qubit = 0
        
        prob_0 = 0.0
        prob_1 = 0.0
        
        for i, amp in enumerate(state_0.data):
            binary = format(i, '04b')  # 4 qubits
            if binary[output_qubit] == '1':  # Use the correct output qubit
                prob_0 += abs(amp)**2
        
        for i, amp in enumerate(state_1.data):
            binary = format(i, '04b')  # 4 qubits
            if binary[output_qubit] == '1':  # Use the correct output qubit
                prob_1 += abs(amp)**2
        
        print(f"  |0⟩ success probability: {prob_0:.4f}")
        print(f"  |1⟩ success probability: {prob_1:.4f}")
        print(f"  Difference: {abs(prob_1 - prob_0):.4f}")
        
        if rt_results[region_name]:
            if abs(prob_1 - prob_0) > 0.3:
                print("  ✓ Effective decoder working!")
            else:
                print("  ✗ Effective decoder not working")
        else:
            if abs(prob_1 - prob_0) < 0.1:
                print("  ✓ Ineffective decoder working!")
            else:
                print("  ✗ Ineffective decoder not working")

if __name__ == "__main__":
    debug_simple_ewr() 