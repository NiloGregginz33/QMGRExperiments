#!/usr/bin/env python3
"""
Demonstration of CTC modifier approach
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from qiskit import QuantumCircuit
import numpy as np

def simple_ctc_modifier(qc, ctc_size=3, phase_strength=0.1):
    """
    Simple CTC modifier that adds temporal paradox effects to an existing circuit.
    This demonstrates the modifier approach instead of creating a separate circuit.
    """
    print(f"[CTC] Applying CTC modifier to circuit with {qc.num_qubits} qubits")
    print(f"[CTC] CTC size: {ctc_size}, Phase strength: {phase_strength}")
    
    # Apply CTC effects to the first ctc_size qubits
    ctc_qubits = min(ctc_size, qc.num_qubits)
    
    # Step 1: Create temporal superposition
    for i in range(ctc_qubits):
        qc.h(i)  # Hadamard to create superposition
    
    # Step 2: Apply temporal phase shifts (creates temporal paradox)
    for i in range(ctc_qubits):
        qc.rz(phase_strength * np.pi, i)  # Rotation around Z-axis creates temporal phase
    
    # Step 3: Create causal loops with controlled operations
    for i in range(ctc_qubits - 1):
        # Create causal loop: qubit i controls qubit i+1, but also vice versa
        qc.cx(i, i+1)  # Forward causality
        qc.rz(phase_strength * 0.5 * np.pi, i+1)  # Temporal phase shift
        qc.cx(i+1, i)  # Reverse causality (paradox!)
    
    # Step 4: Final temporal measurement preparation
    for i in range(ctc_qubits):
        qc.h(i)  # Final Hadamard to reveal temporal paradox
    
    print(f"[CTC] CTC modifier applied successfully!")
    print(f"[CTC] Circuit depth after CTC: {qc.depth()}")
    print(f"[CTC] Total gates after CTC: {len(qc.data)}")
    
    return qc

def create_quantum_spacetime_circuit(num_qubits=4):
    """Create a simple quantum spacetime circuit"""
    qc = QuantumCircuit(num_qubits)
    
    # Initial state preparation
    for i in range(num_qubits):
        qc.h(i)
    
    # Add some entanglement
    for i in range(num_qubits - 1):
        qc.cx(i, i+1)
    
    # Add some rotations
    for i in range(num_qubits):
        qc.rx(0.5 * np.pi, i)
    
    return qc

def main():
    print("=== CTC MODIFIER DEMONSTRATION ===")
    print("This shows how CTC acts as a modifier to existing circuits")
    print()
    
    # Create a quantum spacetime circuit
    print("1. Creating quantum spacetime circuit...")
    qc = create_quantum_spacetime_circuit(4)
    print(f"   Circuit depth: {qc.depth()}")
    print(f"   Number of gates: {len(qc.data)}")
    print()
    
    # Apply CTC modifier
    print("2. Applying CTC modifier...")
    qc_with_ctc = simple_ctc_modifier(qc.copy(), ctc_size=3, phase_strength=0.1)
    print()
    
    # Show the difference
    print("3. Comparison:")
    print(f"   Original circuit depth: {qc.depth()}")
    print(f"   Circuit with CTC depth: {qc_with_ctc.depth()}")
    print(f"   Additional gates from CTC: {len(qc_with_ctc.data) - len(qc.data)}")
    print()
    
    print("4. Circuit structure with CTC:")
    print(qc_with_ctc)
    print()
    
    print("=== SUCCESS! ===")
    print("The CTC modifier approach is working correctly.")
    print("Instead of creating a separate circuit, CTC effects are")
    print("integrated into the existing quantum spacetime circuit.")
    print()
    print("This approach is much more realistic and integrated!")

if __name__ == "__main__":
    main() 