"""
Basic quantum circuit creation and manipulation functions.
This module provides fundamental quantum circuit operations and basic circuit creation utilities.
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
import numpy as np

def create_base_circuit(clbits=2):
    """
    Create a basic quantum circuit with specified number of classical bits.
    
    Args:
        clbits (int): Number of classical bits in the circuit
        
    Returns:
        QuantumCircuit: A basic quantum circuit
    """
    qr = QuantumRegister(2)
    cr = ClassicalRegister(clbits)
    qc = QuantumCircuit(qr, cr)
    return qc

def create_entangled_system(n_qubits=3):
    """
    Create a circuit with an entangled system of n qubits.
    
    Args:
        n_qubits (int): Number of qubits to entangle
        
    Returns:
        QuantumCircuit: Circuit with entangled qubits
    """
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(1, n_qubits):
        qc.cx(0, i)
    return qc

def create_teleportation_circuit():
    """
    Create a quantum teleportation circuit.
    
    Returns:
        QuantumCircuit: Circuit implementing quantum teleportation
    """
    qc = QuantumCircuit(3, 2)
    # Create Bell state
    qc.h(1)
    qc.cx(1, 2)
    # Teleportation protocol
    qc.cx(0, 1)
    qc.h(0)
    return qc

def create_holographic_circuit():
    """
    Create a circuit implementing holographic encoding.
    
    Returns:
        QuantumCircuit: Circuit with holographic encoding
    """
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    return qc

def create_entanglement_wedge_circuit():
    """
    Create a circuit implementing an entanglement wedge.
    
    Returns:
        QuantumCircuit: Circuit with entanglement wedge structure
    """
    qc = QuantumCircuit(5)
    # Create boundary-bulk entanglement
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 4)
    return qc 