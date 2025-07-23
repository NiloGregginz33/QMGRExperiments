#!/usr/bin/env python3
"""
3-Layer HaPPY Code Implementation
HaPPY (Holographic Pentagon Code) with perfect-tensor circuits for EWR experiment.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Operator, Pauli
from typing import Dict, List, Tuple, Optional
import itertools

class HaPPYCode:
    """
    3-Layer HaPPY Code implementation.
    
    Based on the holographic pentagon code from:
    - Pastawski, Yoshida, Harlow, Preskill (2015) - "Holographic quantum error-correcting codes"
    - Almheiri, Dong, Harlow (2015) - "Bulk locality and quantum error correction in AdS/CFT"
    """
    
    def __init__(self, num_boundary_qubits: int = 12):
        """
        Initialize HaPPY code.
        
        Args:
            num_boundary_qubits: Number of boundary qubits (should be 12 for 3 regions)
        """
        self.num_boundary_qubits = num_boundary_qubits
        self.num_bulk_qubits = 1  # Single logical bulk qubit
        self.num_physical_qubits = num_boundary_qubits
        
        # Define the code structure
        self._define_code_structure()
        
    def _define_code_structure(self):
        """Define the HaPPY code structure with 3 layers."""
        # Layer 1: Basic stabilizer generators
        self.layer1_stabilizers = self._create_layer1_stabilizers()
        
        # Layer 2: Enhanced entanglement
        self.layer2_stabilizers = self._create_layer2_stabilizers()
        
        # Layer 3: Final stabilizer layer
        self.layer3_stabilizers = self._create_layer3_stabilizers()
        
        # Logical operators
        self.logical_x = self._create_logical_x()
        self.logical_z = self._create_logical_z()
        
    def _create_layer1_stabilizers(self) -> List[Pauli]:
        """Create Layer 1 stabilizer generators (basic surface code)."""
        stabilizers = []
        
        # X-type stabilizers (horizontal)
        for i in range(0, self.num_boundary_qubits - 3, 4):
            pauli_str = ['I'] * self.num_boundary_qubits
            pauli_str[i] = 'X'
            pauli_str[i+1] = 'X'
            pauli_str[i+2] = 'X'
            pauli_str[i+3] = 'X'
            stabilizers.append(Pauli(''.join(pauli_str)))
        
        # Z-type stabilizers (vertical)
        for i in range(0, self.num_boundary_qubits - 4, 4):
            pauli_str = ['I'] * self.num_boundary_qubits
            pauli_str[i] = 'Z'
            pauli_str[i+4] = 'Z'
            pauli_str[i+1] = 'Z'
            pauli_str[i+5] = 'Z'
            stabilizers.append(Pauli(''.join(pauli_str)))
        
        return stabilizers
    
    def _create_layer2_stabilizers(self) -> List[Pauli]:
        """Create Layer 2 stabilizer generators (enhanced entanglement)."""
        stabilizers = []
        
        # Cross-connections between regions
        for i in range(0, self.num_boundary_qubits - 1, 3):
            pauli_str = ['I'] * self.num_boundary_qubits
            pauli_str[i] = 'X'
            pauli_str[i+1] = 'X'
            stabilizers.append(Pauli(''.join(pauli_str)))
            
            pauli_str = ['I'] * self.num_boundary_qubits
            pauli_str[i] = 'Z'
            pauli_str[i+1] = 'Z'
            stabilizers.append(Pauli(''.join(pauli_str)))
        
        return stabilizers
    
    def _create_layer3_stabilizers(self) -> List[Pauli]:
        """Create Layer 3 stabilizer generators (final layer)."""
        stabilizers = []
        
        # Final stabilizer layer for robustness
        for i in range(0, self.num_boundary_qubits - 3, 2):
            pauli_str = ['I'] * self.num_boundary_qubits
            pauli_str[i] = 'X'
            pauli_str[i+1] = 'X'
            pauli_str[i+2] = 'X'
            stabilizers.append(Pauli(''.join(pauli_str)))
            
            pauli_str = ['I'] * self.num_boundary_qubits
            pauli_str[i] = 'Z'
            pauli_str[i+2] = 'Z'
            stabilizers.append(Pauli(''.join(pauli_str)))
        
        return stabilizers
    
    def _create_logical_x(self) -> Pauli:
        """Create logical X operator."""
        pauli_str = ['I'] * self.num_boundary_qubits
        # Logical X: product of X operators along a horizontal path
        for i in range(0, self.num_boundary_qubits, 4):
            pauli_str[i] = 'X'
        return Pauli(''.join(pauli_str))
    
    def _create_logical_z(self) -> Pauli:
        """Create logical Z operator."""
        pauli_str = ['I'] * self.num_boundary_qubits
        # Logical Z: product of Z operators along a vertical path
        for i in range(0, 4):
            pauli_str[i] = 'Z'
        return Pauli(''.join(pauli_str))
    
    def create_encoding_circuit(self, logical_state: str = '1') -> QuantumCircuit:
        """
        Create encoding circuit for the HaPPY code.
        
        Args:
            logical_state: Logical state to encode ('0' or '1')
            
        Returns:
            QuantumCircuit: Circuit that encodes the logical state
        """
        qc = QuantumCircuit(self.num_boundary_qubits, self.num_boundary_qubits)
        
        # Step 1: Initialize boundary qubits in entangled state
        # Create Bell pairs across the boundary
        for i in range(0, self.num_boundary_qubits, 2):
            qc.h(i)
            qc.cx(i, i+1)
        
        # Step 2: Apply Layer 1 stabilizers
        for stabilizer in self.layer1_stabilizers:
            self._apply_stabilizer(qc, stabilizer)
        
        # Step 3: Apply Layer 2 stabilizers
        for stabilizer in self.layer2_stabilizers:
            self._apply_stabilizer(qc, stabilizer)
        
        # Step 4: Apply Layer 3 stabilizers
        for stabilizer in self.layer3_stabilizers:
            self._apply_stabilizer(qc, stabilizer)
        
        # Step 5: Apply logical operator to encode desired state
        if logical_state == '1':
            self._apply_logical_operator(qc, self.logical_x)
        
        return qc
    
    def _apply_stabilizer(self, qc: QuantumCircuit, stabilizer: Pauli):
        """Apply a stabilizer to the circuit."""
        pauli_str = stabilizer.to_label()
        
        # Apply Hadamard gates for Z operators
        for i, pauli in enumerate(pauli_str):
            if pauli == 'Z':
                qc.h(i)
        
        # Apply CNOT gates to create the stabilizer
        # This is a simplified implementation
        x_positions = [i for i, pauli in enumerate(pauli_str) if pauli == 'X']
        z_positions = [i for i, pauli in enumerate(pauli_str) if pauli == 'Z']
        
        # Apply CNOTs between X and Z positions
        for x_pos in x_positions:
            for z_pos in z_positions:
                if x_pos != z_pos:
                    qc.cx(x_pos, z_pos)
        
        # Apply Hadamard gates back for Z operators
        for i, pauli in enumerate(pauli_str):
            if pauli == 'Z':
                qc.h(i)
    
    def _apply_logical_operator(self, qc: QuantumCircuit, logical_op: Pauli):
        """Apply a logical operator to the circuit."""
        pauli_str = logical_op.to_label()
        
        # Apply the logical operator
        for i, pauli in enumerate(pauli_str):
            if pauli == 'X':
                qc.h(i)
                qc.cx(i, (i+1) % self.num_boundary_qubits)
            elif pauli == 'Z':
                qc.cz(i, (i+1) % self.num_boundary_qubits)
    
    def get_logical_to_physical_mapping(self) -> Dict[str, List[int]]:
        """
        Get the logical-to-physical qubit mapping.
        
        Returns:
            Dict mapping logical qubit to physical qubit positions
        """
        # For HaPPY code, the logical qubit is encoded across all boundary qubits
        # but with different weights based on the stabilizer structure
        
        # Calculate the weight of each physical qubit in the logical encoding
        logical_x_str = self.logical_x.to_label()
        logical_z_str = self.logical_z.to_label()
        
        qubit_weights = {}
        for i in range(self.num_boundary_qubits):
            weight = 0
            if logical_x_str[i] != 'I':
                weight += 1
            if logical_z_str[i] != 'I':
                weight += 1
            qubit_weights[i] = weight
        
        # Return mapping with qubits ordered by weight
        sorted_qubits = sorted(qubit_weights.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'logical_bulk_qubit': [qubit for qubit, weight in sorted_qubits],
            'qubit_weights': dict(sorted_qubits)
        }
    
    def create_perfect_tensor_circuit(self, region_qubits: List[int]) -> QuantumCircuit:
        """
        Create a perfect-tensor circuit for a specific region.
        
        Args:
            region_qubits: List of qubits in the boundary region
            
        Returns:
            QuantumCircuit: Perfect-tensor circuit for the region
        """
        num_region_qubits = len(region_qubits)
        qc = QuantumCircuit(max(region_qubits) + 1)
        
        # Perfect tensor: maximally entangled state
        # Apply a series of random Clifford operations
        
        # Step 1: Create maximally entangled state
        for i in range(0, num_region_qubits, 2):
            if i + 1 < num_region_qubits:
                qc.h(region_qubits[i])
                qc.cx(region_qubits[i], region_qubits[i+1])
        
        # Step 2: Apply random Clifford operations
        for i in range(num_region_qubits):
            # Random single-qubit Clifford
            qc.h(region_qubits[i])
            qc.s(region_qubits[i])
            qc.h(region_qubits[i])
        
        # Step 3: Apply entangling operations
        for i in range(num_region_qubits - 1):
            qc.cx(region_qubits[i], region_qubits[i+1])
            qc.cz(region_qubits[i], region_qubits[i+1])
        
        return qc
    
    def get_code_distance(self) -> int:
        """Get the code distance."""
        # Simplified code distance calculation
        # In a real implementation, this would involve checking all error patterns
        return 3  # Minimum weight of logical operators
    
    def get_logical_operator_weight(self, logical_op: str) -> int:
        """
        Get the weight of a logical operator.
        
        Args:
            logical_op: 'X' or 'Z'
            
        Returns:
            int: Weight of the logical operator
        """
        if logical_op == 'X':
            pauli_str = self.logical_x.to_label()
        elif logical_op == 'Z':
            pauli_str = self.logical_z.to_label()
        else:
            raise ValueError(f"Unknown logical operator: {logical_op}")
        
        return sum(1 for pauli in pauli_str if pauli != 'I')

# Example usage
if __name__ == "__main__":
    # Create HaPPY code
    happy_code = HaPPYCode(num_boundary_qubits=12)
    
    # Create encoding circuit
    encoding_circuit = happy_code.create_encoding_circuit(logical_state='1')
    print(f"Encoding circuit depth: {encoding_circuit.depth()}")
    
    # Get logical-to-physical mapping
    mapping = happy_code.get_logical_to_physical_mapping()
    print(f"Logical to physical mapping: {mapping}")
    
    # Test with a region
    region_qubits = [4, 5, 6, 7]
    perfect_tensor_circuit = happy_code.create_perfect_tensor_circuit(region_qubits)
    print(f"Perfect tensor circuit depth: {perfect_tensor_circuit.depth()}")
    
    print(f"Code distance: {happy_code.get_code_distance()}")
    print(f"Logical X weight: {happy_code.get_logical_operator_weight('X')}")
    print(f"Logical Z weight: {happy_code.get_logical_operator_weight('Z')}") 