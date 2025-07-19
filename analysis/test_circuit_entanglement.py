#!/usr/bin/env python3
"""
Test script to verify entanglement in the quantum circuit
"""

import sys
import os
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, entropy

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_simple_entanglement():
    """Test a simple Bell state to verify mutual information calculation"""
    print("=== Testing Simple Bell State ===")
    
    # Create Bell state |00⟩ + |11⟩
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    
    # Get statevector
    state = Statevector.from_int(0, 2**2)
    state = state.evolve(qc)
    
    # Calculate mutual information manually
    rho_01 = partial_trace(state, [])
    rho_0 = partial_trace(rho_01, [1])
    rho_1 = partial_trace(rho_01, [0])
    
    S_01 = entropy(rho_01)
    S_0 = entropy(rho_0)
    S_1 = entropy(rho_1)
    
    MI = S_0 + S_1 - S_01
    print(f"Bell state mutual information: {MI:.6f}")
    print(f"Expected: ~1.0 (maximal entanglement)")
    
    return MI

def test_custom_circuit_entanglement():
    """Test the custom curvature circuit for entanglement"""
    print("\n=== Testing Custom Curvature Circuit ===")
    
    # Import the circuit building function
    from experiments.custom_curvature_experiment import build_custom_circuit_layers
    
    # Use the same parameters as the experiment
    num_qubits = 7
    topology = "triangulated"
    custom_edges = "0-1:1.0000,0-2:0.9931,0-3:1.0000,0-4:1.0000,0-5:0.9883,0-6:0.9883,1-2:1.0000,1-3:1.0000,1-4:0.9765,1-5:1.0000,1-6:0.9768,2-3:0.9767,2-4:1.0000,2-5:0.9043,2-6:0.9138,3-4:0.9719,3-5:0.9494,3-6:1.0000,4-5:0.9546,4-6:0.9294,5-6:1.0000"
    alpha = 0.8
    weight = 1.0
    gamma = 0.3
    sigma = None
    init_angle = 0.0
    geometry = "hyperbolic"
    curvature = 0.5
    timesteps = 3
    
    # Build the circuit
    circuits, final_circuit = build_custom_circuit_layers(
        num_qubits, topology, custom_edges, alpha, weight, gamma, sigma, init_angle,
        geometry=geometry, curvature=curvature, timesteps=timesteps
    )
    
    print(f"Circuit depth: {final_circuit.depth()}")
    print(f"Number of gates: {final_circuit.count_ops()}")
    
    # Test each timestep
    for t, circuit in enumerate(circuits):
        print(f"\n--- Timestep {t+1} ---")
        
        # Remove measurements for statevector calculation
        test_circuit = circuit.copy()
        test_circuit.data = [op for op in test_circuit.data if op.operation.name != 'measure']
        
        # Get statevector
        state = Statevector.from_int(0, 2**num_qubits)
        state = state.evolve(test_circuit)
        
        # Calculate mutual information for all pairs
        mi_values = []
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                # Trace out all qubits except i and j
                qubits_to_trace = list(range(num_qubits))
                qubits_to_trace.remove(i)
                qubits_to_trace.remove(j)
                
                rho_ij = partial_trace(state, qubits_to_trace)
                rho_i = partial_trace(rho_ij, [1])
                rho_j = partial_trace(rho_ij, [0])
                
                # Calculate entropies
                S_ij = entropy(rho_ij)
                S_i = entropy(rho_i)
                S_j = entropy(rho_j)
                
                # Mutual information: I(A;B) = S(A) + S(B) - S(AB)
                mi = S_i + S_j - S_ij
                mi_values.append(mi)
                
                if mi > 0.01:  # Only print significant MI
                    print(f"  MI({i},{j}): {mi:.6f}")
        
        avg_mi = np.mean(mi_values)
        max_mi = np.max(mi_values)
        print(f"  Average MI: {avg_mi:.6f}")
        print(f"  Max MI: {max_mi:.6f}")
        print(f"  Non-zero MI pairs: {sum(1 for mi in mi_values if mi > 0.01)}/{len(mi_values)}")

def test_entangling_gates():
    """Test different entangling gates"""
    print("\n=== Testing Different Entangling Gates ===")
    
    # Test RZZ gate
    print("Testing RZZ gate:")
    qc_rzz = QuantumCircuit(2)
    qc_rzz.h(0)
    qc_rzz.rzz(np.pi/2, 0, 1)
    
    state_rzz = Statevector.from_int(0, 4)
    state_rzz = state_rzz.evolve(qc_rzz)
    
    rho_rzz = partial_trace(state_rzz, [])
    rho_0_rzz = partial_trace(rho_rzz, [1])
    rho_1_rzz = partial_trace(rho_rzz, [0])
    
    S_rzz = entropy(rho_rzz)
    S_0_rzz = entropy(rho_0_rzz)
    S_1_rzz = entropy(rho_1_rzz)
    
    MI_rzz = S_0_rzz + S_1_rzz - S_rzz
    print(f"  RZZ mutual information: {MI_rzz:.6f}")
    
    # Test RYY gate
    print("Testing RYY gate:")
    qc_ryy = QuantumCircuit(2)
    qc_ryy.h(0)
    qc_ryy.ryy(np.pi/2, 0, 1)
    
    state_ryy = Statevector.from_int(0, 4)
    state_ryy = state_ryy.evolve(qc_ryy)
    
    rho_ryy = partial_trace(state_ryy, [])
    rho_0_ryy = partial_trace(rho_ryy, [1])
    rho_1_ryy = partial_trace(rho_ryy, [0])
    
    S_ryy = entropy(rho_ryy)
    S_0_ryy = entropy(rho_0_ryy)
    S_1_ryy = entropy(rho_1_ryy)
    
    MI_ryy = S_0_ryy + S_1_ryy - S_ryy
    print(f"  RYY mutual information: {MI_ryy:.6f}")

if __name__ == "__main__":
    test_simple_entanglement()
    test_entangling_gates()
    test_custom_circuit_entanglement() 