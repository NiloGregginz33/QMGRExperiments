# Small Entropy Injection Experiment

import sys
import os

# Adjust Python path to include the Factory directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
import numpy as np
import argparse
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from CGPTFactory import run
import json
from qiskit.quantum_info import partial_trace, entropy
from qiskit_aer import Aer

# Function to create a quantum circuit for small entropy injection
def create_small_entropy_injection_circuit(num_qubits):
    # Create a quantum circuit with ancilla bits
    circuit = QuantumCircuit(num_qubits + 1, num_qubits)  # Adjusted classical bits
    
    # Create a Bell pair between the first qubit and the ancilla
    circuit.h(0)  # Apply Hadamard gate to the first qubit
    circuit.cx(0, num_qubits)  # Apply CNOT gate to entangle with ancilla
    
    # Trace out the ancilla to introduce entropy
    # (This is a conceptual step; actual tracing out is done in analysis)
    
    return circuit

# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Run small entropy injection experiment.')
    parser.add_argument('--device', type=str, default='simulator', choices=['simulator', 'hardware'],
                        help='Choose the execution device: simulator or hardware')
    return parser.parse_args()

# Function to log results
def log_results(counts, num_qubits, device, statevector, entropies, causal_witness_before, causal_witness_after):
    # Create a directory for logs if it doesn't exist
    log_dir = 'experiment_logs/small_entropy_injection_experiment'
    os.makedirs(log_dir, exist_ok=True)
    
    # Convert statevector's complex numbers to a serializable format
    statevector_serializable = [(v.real, v.imag) for v in statevector.data]
    
    # Log results to results.json
    results_path = os.path.join(log_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'counts': counts,
            'num_qubits': num_qubits,
            'device': device,
            'statevector': statevector_serializable,  # Use serializable format
            'entropies': entropies,
            'causal_witness_before': causal_witness_before,
            'causal_witness_after': causal_witness_after
        }, f, indent=4)
    
    # Log summary to summary.txt
    summary_path = os.path.join(log_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Experiment: Small Entropy Injection\n")
        f.write(f"Number of Qubits: {num_qubits}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Counts: {counts}\n")
        f.write(f"Statevector: {statevector}\n")
        f.write(f"Entropies: {entropies}\n")
        f.write(f"Causal Witness Before: {causal_witness_before}\n")
        f.write(f"Causal Witness After: {causal_witness_after}\n")

# Function to add entropy deliberately
def add_entropy(circuit, num_qubits):
    # Example: Apply decoherence to ancilla
    circuit.reset(num_qubits)  # Reset ancilla to simulate decoherence
    return circuit

# Function to measure causal witness
def measure_causal_witness(circuit):
    # Placeholder for causal witness measurement logic
    # (To be implemented)
    return "Causal witness measured"

# Function to compare entropies over time
def compare_entropies(circuit, num_qubits):
    # Create a copy of the circuit
    circuit_no_measure = circuit.copy()
    
    # Correctly remove measurement operations
    circuit_no_measure.data = [instr for instr in circuit_no_measure.data if instr[0].name != 'measure']
    
    # Use a statevector simulator to get the state
    from qiskit_aer import AerSimulator
    statevector_simulator = AerSimulator(method='statevector')
    result = run(circuit_no_measure, statevector_simulator, shots=1, rs=True)
    state = result  # Use the result directly as the statevector
    
    # Partial trace out the ancilla (last qubit)
    reduced_state = partial_trace(state, [num_qubits])
    
    # Compute entropy of the main qubits
    entropies = [entropy(reduced_state) for _ in range(num_qubits)]
    return entropies

# Function to run the experiment with device selection
def run_experiment(num_qubits, device):
    circuit = create_small_entropy_injection_circuit(num_qubits)
    
    # Add entropy deliberately
    circuit = add_entropy(circuit, num_qubits)
    
    # Measure causal witness before
    causal_witness_before = measure_causal_witness(circuit)
    
    if device == 'simulator':
        # Remove measurement operations for statevector simulation
        circuit_no_measure = circuit.copy()
        circuit_no_measure.data = [instr for instr in circuit_no_measure.data if instr[0].name != 'measure']
        
        # Use a statevector simulator
        from qiskit_aer import AerSimulator
        simulator = AerSimulator(method='statevector')
        print("Using AerSimulator with statevector method")
        result = run(circuit_no_measure, simulator, shots=1, rs=True)
        # Directly use the result as the statevector
        statevector = result
        counts = None  # No counts available from statevector simulation
    else:
        # Use real hardware
        from qiskit import IBMQ
        provider = IBMQ.load_account()
        backend = provider.get_backend('ibmq_manila')  # Example hardware backend
        print("Using IBMQ hardware backend")
        result = run(circuit, backend, shots=1024)
        statevector = None  # Statevector not available from hardware run
        counts = result.get_counts()
    
    if counts is not None:
        print("Counts:", counts)
        plot_histogram(counts)
    
    # Measure causal witness after
    causal_witness_after = measure_causal_witness(circuit)
    
    # Compare entropies over time using statevector
    entropies = compare_entropies(circuit, num_qubits)
    print("Entropies:", entropies)
    
    # Log results
    log_results(counts, num_qubits, device, statevector, entropies, causal_witness_before, causal_witness_after)
    
    # Log additional data
    with open('experiment_logs/small_entropy_injection_experiment/summary.txt', 'a') as f:
        f.write(f"Causal Witness Before: {causal_witness_before}\n")
        f.write(f"Causal Witness After: {causal_witness_after}\n")
        f.write(f"Entropies: {entropies}\n")

# Example usage
if __name__ == "__main__":
    args = parse_arguments()
    run_experiment(3, args.device)  # Run with 3 qubits and selected device 

def minimal_working_example():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, partial_trace, entropy
    
    # Create a quantum circuit with 2 qubits
    qc = QuantumCircuit(2)  # qubit 0: system, qubit 1: ancilla
    qc.h(0)                 # Hadamard → superposition
    qc.cx(0, 1)             # Entangle → Bell state
    
    # Simulate the statevector
    state = Statevector.from_instruction(qc)
    
    # Trace out ancilla (qubit 1)
    reduced = partial_trace(state, [1])
    
    # Calculate entropy of system qubit
    S = entropy(reduced)
    print("Entropy of system qubit:", S)

# Run the minimal working example
minimal_working_example() 