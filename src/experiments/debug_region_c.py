#!/usr/bin/env python3
"""
Debug script for Region C decoding in EWR experiment
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from quantum.ewr_circuits import (
    create_bulk_logical_qubit,
    define_boundary_regions,
    create_holographic_mapping,
    create_decoding_circuit
)

def debug_region_c():
    """Debug Region C decoding specifically"""
    
    print("=== REGION C DECODING DEBUG ===")
    
    # Create the full EWR circuit
    num_qubits = 12
    bulk_point_location = 6
    
    print(f"Creating bulk logical qubit with {num_qubits} qubits...")
    bulk_circuit = create_bulk_logical_qubit(num_qubits)
    print(f"Bulk circuit depth: {bulk_circuit.depth()}")
    
    print(f"Defining boundary regions...")
    regions = define_boundary_regions(num_qubits)
    print(f"Regions: {regions}")
    
    print(f"Applying holographic mapping...")
    mapped_circuit = create_holographic_mapping(bulk_circuit, regions)
    print(f"Mapped circuit depth: {mapped_circuit.depth()}")
    
    # Focus on Region C
    region_c_qubits = regions['C']
    print(f"\nRegion C qubits: {region_c_qubits}")
    print(f"Bulk point location: {bulk_point_location}")
    print(f"Is bulk point in Region C? {bulk_point_location in region_c_qubits}")
    
    # Create decoding circuit for Region C
    print(f"\nCreating decoding circuit for Region C...")
    decoding_circuits = create_decoding_circuit(region_c_qubits)
    
    print(f"Preparation circuit depth: {decoding_circuits['preparation'].depth()}")
    print(f"Measurement circuit depth: {decoding_circuits['measurement'].depth()}")
    print(f"Full circuit depth: {decoding_circuits['full'].depth()}")
    
    # Test the full pipeline
    print(f"\nTesting full pipeline...")
    
    # Step 1: Get the state after bulk preparation and mapping
    print("Step 1: Getting mapped state...")
    mapped_state = Statevector.from_instruction(mapped_circuit)
    print(f"Mapped state shape: {mapped_state.data.shape}")
    print(f"Mapped state norm: {np.linalg.norm(mapped_state.data):.6f}")
    
    # Step 2: Apply decoding to get the decoded state
    print("Step 2: Applying decoding circuit...")
    decoded_state = Statevector.from_instruction(decoding_circuits['preparation'])
    print(f"Decoded state shape: {decoded_state.data.shape}")
    print(f"Decoded state norm: {np.linalg.norm(decoded_state.data):.6f}")
    
    # Step 3: Analyze the decoded state
    print("Step 3: Analyzing decoded state...")
    
    # Get the measurement circuit to understand what we're measuring
    meas_circuit = decoding_circuits['measurement']
    print(f"Measurement circuit:")
    print(meas_circuit)
    
    # Check the target qubit (should be the first qubit in the region)
    target_qubit = region_c_qubits[0]
    print(f"Target qubit: {target_qubit}")
    
    # Analyze the decoded state for the target qubit
    print(f"\nAnalyzing target qubit {target_qubit} state...")
    
    # Calculate the probability of measuring |1⟩ on the target qubit
    prob_1 = 0.0
    for i, amp in enumerate(decoded_state.data):
        # Convert to binary representation for the 4-qubit decoded state
        binary = format(i, '04b')  # 4 qubits in the decoded state
        # Check if the target qubit (index 0 in the decoded circuit) is |1⟩
        if binary[0] == '1':  # First qubit in the decoded circuit
            prob_1 += abs(amp)**2
    
    print(f"Probability of measuring |1⟩ on target qubit: {prob_1:.6f}")
    
    # Also check the full state to see if there are any patterns
    print(f"\nFull decoded state analysis:")
    print(f"Number of non-zero amplitudes: {np.count_nonzero(decoded_state.data)}")
    print(f"Max amplitude magnitude: {np.max(np.abs(decoded_state.data)):.6f}")
    print(f"Min amplitude magnitude: {np.min(np.abs(decoded_state.data)):.6f}")
    
    # Check if the state has the expected structure for a logical qubit
    logical_1_prob = 0.0
    for i, amp in enumerate(decoded_state.data):
        if abs(amp) > 1e-6:  # Non-zero amplitude
            # Check if this basis state corresponds to logical |1⟩
            binary = format(i, '04b')  # 4 qubits
            parity = sum(int(bit) for bit in binary) % 2
            if parity == 1:  # Odd parity
                logical_1_prob += abs(amp)**2
    
    print(f"Probability of logical |1⟩ state: {logical_1_prob:.6f}")
    
    return {
        'target_qubit_prob_1': prob_1,
        'logical_1_prob': logical_1_prob,
        'mapped_state_norm': np.linalg.norm(mapped_state.data),
        'decoded_state_norm': np.linalg.norm(decoded_state.data)
    }

if __name__ == "__main__":
    results = debug_region_c()
    print(f"\n=== DEBUG SUMMARY ===")
    print(f"Target qubit |1⟩ probability: {results['target_qubit_prob_1']:.6f}")
    print(f"Logical |1⟩ probability: {results['logical_1_prob']:.6f}")
    print(f"Mapped state norm: {results['mapped_state_norm']:.6f}")
    print(f"Decoded state norm: {results['decoded_state_norm']:.6f}") 