"""
Quantum utility functions.
This module provides various utility functions for quantum circuit manipulation and analysis.
"""

from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import Statevector
import numpy as np
from typing import Dict, List, Tuple

def get_best_backend(service, min_qubits=3, max_queue=10):
    """
    Get the best available quantum backend.
    
    Args:
        service: Qiskit service instance
        min_qubits (int): Minimum number of qubits required
        max_queue (int): Maximum acceptable queue length
        
    Returns:
        Backend: Best available quantum backend
    """
    backends = service.backends()
    available = [b for b in backends if b.configuration().n_qubits >= min_qubits]
    if not available:
        return Aer.get_backend('aer_simulator')
    return min(available, key=lambda b: b.status().pending_jobs)

def is_simulator(backend):
    """
    Check if a backend is a simulator.
    
    Args:
        backend: Qiskit backend
        
    Returns:
        bool: True if backend is a simulator
    """
    return backend.name().startswith('aer') or backend.name().startswith('qasm')

def select_backend(use_simulator=True, hardware_backend=None):
    """
    Select an appropriate quantum backend.
    
    Args:
        use_simulator (bool): Whether to use a simulator
        hardware_backend (str): Name of hardware backend to use
        
    Returns:
        Backend: Selected quantum backend
    """
    if use_simulator:
        return Aer.get_backend('aer_simulator')
    return hardware_backend

def initialize_qubits(num_qubits):
    """
    Initialize a quantum circuit with specified number of qubits.
    
    Args:
        num_qubits (int): Number of qubits to initialize
        
    Returns:
        QuantumCircuit: Initialized quantum circuit
    """
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.h(i)
    return qc

def apply_entanglement(qc, qr):
    """
    Apply entanglement to a quantum circuit.
    
    Args:
        qc (QuantumCircuit): Circuit to modify
        qr (QuantumRegister): Quantum register to entangle
        
    Returns:
        QuantumCircuit: Circuit with entanglement
    """
    for i in range(len(qr)-1):
        qc.cx(qr[i], qr[i+1])
    return qc

def measure_and_reset(qc, qr, cr):
    """
    Measure qubits and reset them to |0⟩.
    
    Args:
        qc (QuantumCircuit): Circuit to modify
        qr (QuantumRegister): Quantum register to measure
        cr (ClassicalRegister): Classical register for results
        
    Returns:
        QuantumCircuit: Circuit with measurements and resets
    """
    for i in range(len(qr)):
        qc.measure(qr[i], cr[i])
        qc.reset(qr[i])
    return qc

def run_circuit_statevector(qc):
    """
    Run a circuit and return its statevector.
    
    Args:
        qc (QuantumCircuit): Circuit to run
        
    Returns:
        Statevector: Final statevector of the circuit
    """
    backend = Aer.get_backend('statevector_simulator')
    job = backend.run(qc)
    return job.result().get_statevector()

def process_sampler_result(result, shots=8192):
    """
    Process results from a quantum circuit execution.
    
    Args:
        result: Result from circuit execution
        shots (int): Number of measurement shots
        
    Returns:
        Dict: Processed results
    """
    counts = result.get_counts()
    total = sum(counts.values())
    return {k: v/total for k, v in counts.items()} 