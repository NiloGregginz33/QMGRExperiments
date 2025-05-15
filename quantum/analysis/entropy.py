"""
Quantum entropy analysis functions.
This module provides tools for calculating and analyzing various types of quantum entropy.
"""

from qiskit.quantum_info import Statevector, entropy
import numpy as np
from typing import Dict, List, Tuple

def qiskit_entropy(rho):
    """
    Calculate von Neumann entropy using Qiskit's built-in function.
    
    Args:
        rho: Density matrix or statevector
        
    Returns:
        float: von Neumann entropy
    """
    return entropy(rho)

def shannon_entropy(probs):
    """
    Calculate Shannon entropy for a probability distribution.
    
    Args:
        probs: Probability distribution
        
    Returns:
        float: Shannon entropy
    """
    probs = np.array(probs)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def measure_subsystem_entropies(state: Statevector, num_qubits: int) -> Dict[int, float]:
    """
    Measure entropies of all possible subsystems.
    
    Args:
        state: Quantum state
        num_qubits: Total number of qubits
        
    Returns:
        Dict[int, float]: Dictionary mapping subsystem size to entropy
    """
    entropies = {}
    for i in range(1, num_qubits):
        subsystem = list(range(i))
        entropies[i] = entropy(state, subsystem)
    return entropies

def calculate_von_neumann_entropy(qc, num_radiation_qubits):
    """
    Calculate von Neumann entropy for a quantum circuit with radiation qubits.
    
    Args:
        qc: Quantum circuit
        num_radiation_qubits: Number of radiation qubits
        
    Returns:
        float: von Neumann entropy
    """
    state = Statevector.from_instruction(qc)
    return entropy(state, list(range(num_radiation_qubits)))

def analyze_subsystem_qiskit_entropy(statevector):
    """
    Analyze subsystem entropies using Qiskit's entropy function.
    
    Args:
        statevector: Quantum state vector
        
    Returns:
        Dict[int, float]: Dictionary of subsystem entropies
    """
    n_qubits = int(np.log2(len(statevector)))
    entropies = {}
    for i in range(1, n_qubits):
        subsystem = list(range(i))
        entropies[i] = entropy(statevector, subsystem)
    return entropies

def compute_mutual_information(statevector, subsystem_a, subsystem_b):
    """
    Compute mutual information between two subsystems.
    
    Args:
        statevector: Quantum state vector
        subsystem_a: First subsystem indices
        subsystem_b: Second subsystem indices
        
    Returns:
        float: Mutual information
    """
    S_a = entropy(statevector, subsystem_a)
    S_b = entropy(statevector, subsystem_b)
    S_ab = entropy(statevector, subsystem_a + subsystem_b)
    return S_a + S_b - S_ab 