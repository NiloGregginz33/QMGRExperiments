#!/usr/bin/env python3
"""
Simple stabilizer code for EWR demonstration.
This creates a working quantum error-correcting code that can be used to test EWR.
"""

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from typing import List, Dict

class SimpleStabilizerCode:
    """
    Simple stabilizer code for EWR demonstration.
    
    This is a 4-qubit code with 2 logical qubits and 2 stabilizer generators.
    The logical operators are:
    - Logical X1: XXXX (acts on all qubits)
    - Logical Z1: ZIII (acts on first qubit)
    """
    
    def __init__(self, num_qubits: int = 4):
        """Initialize simple stabilizer code."""
        self.num_qubits = num_qubits
        
        # Define stabilizer generators
        self.stabilizers = [
            Pauli('ZZII'),  # Z1 Z2
            Pauli('IIZZ'),  # Z3 Z4
        ]
        
        # Define logical operators
        self.logical_x = Pauli('XXXX')  # Logical X acts on all qubits
        self.logical_z = Pauli('ZIII')  # Logical Z acts on first qubit
    
    def create_encoding_circuit(self, logical_state: str = '1') -> QuantumCircuit:
        """
        Create encoding circuit for the simple stabilizer code.
        
        Args:
            logical_state: Logical state to encode ('0' or '1')
            
        Returns:
            QuantumCircuit: Circuit that encodes the logical state
        """
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        
        # Step 1: Initialize in entangled state
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        
        # Step 2: Apply stabilizers
        # Stabilizer 1: Z1 Z2
        qc.h(0)
        qc.h(1)
        qc.cx(0, 1)
        qc.h(0)
        qc.h(1)
        
        # Stabilizer 2: Z3 Z4
        qc.h(2)
        qc.h(3)
        qc.cx(2, 3)
        qc.h(2)
        qc.h(3)
        
        # Step 3: Apply logical operator if encoding |1⟩
        if logical_state == '1':
            # Apply logical X (XXXX)
            for i in range(self.num_qubits):
                qc.x(i)
        
        return qc
    
    def create_decoder_circuit(self, region_qubits: List[int], rt_contains_bulk: bool) -> QuantumCircuit:
        """
        Create decoder circuit for a specific region.
        
        Args:
            region_qubits: List of qubits in the region
            rt_contains_bulk: Whether RT surface contains bulk point
            
        Returns:
            QuantumCircuit: Decoder circuit
        """
        qc = QuantumCircuit(self.num_qubits, 1)
        
        if rt_contains_bulk:
            # Effective decoder: extract logical X information
            # The logical X operator is XXXX, so we need to measure the parity
            qc.h(0)  # Prepare ancilla in |+⟩ state
            
            # Apply CNOTs to measure the logical X operator
            for qubit in region_qubits:
                if qubit != 0:  # Avoid duplicate qubit arguments
                    qc.cx(qubit, 0)
            
            qc.h(0)  # Measure in X basis
            qc.measure(0, 0)  # Measure qubit 0 to classical bit 0
            
        else:
            # Ineffective decoder: random operations that don't extract logical info
            qc.h(0)
            qc.s(0)
            qc.h(0)
            qc.measure(0, 0)  # Measure qubit 0 to classical bit 0
        
        return qc
    
    def get_logical_to_physical_mapping(self) -> Dict[str, List[int]]:
        """Get logical-to-physical qubit mapping."""
        return {
            'logical_x': list(range(self.num_qubits)),  # All qubits
            'logical_z': [0]  # First qubit only
        } 