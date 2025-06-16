#This is the same as the last experiment (ex2), however but now i am more unsure of the time-independency,
#and this reason might just be because the black hole is kind of a quantum particle
#and can be described by a handful of properties. What if its just the most dominant
#charge that gets emitted over and over because that last experiment was weird

from qiskit_aer import Aer
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister
from qiskit.quantum_info import Statevector, partial_trace, entropy
from qiskit.visualization import plot_histogram
import numpy as np

# Function to create the quantum circuit (no classical bits for Statevector)
def create_circuit(apply_positive_charge, apply_negative_charge):
    qc = QuantumCircuit(2)  # Create a new circuit
    qc.h(0)  # Superposition on Black Hole qubit
    qc.cx(0, 1)  # Entangle Black Hole and Radiation qubits

    # Simulate Charge Pulses
    if apply_positive_charge:
        qc.x(0)  # Positive charge (Pauli-X gate on Black Hole)
    if apply_negative_charge:
        qc.z(0)  # Negative charge (Pauli-Z gate on Black Hole)
    
    return qc

# Analyze phase shifts in the quantum state
def analyze_phases(qc):
    state = Statevector.from_instruction(qc)  # Get the statevector
    phases = np.angle(state.data)  # Extract the phases
    return phases

# Simulate black hole evaporation
def simulate_evaporation(charge_state, num_radiation_qubits):
    qc = QuantumCircuit(num_radiation_qubits + 1)  # One black hole + radiation qubits
    if charge_state == "positive":
        qc.x(0)
    elif charge_state == "negative":
        qc.z(0)

    # Entangle black hole with radiation qubits sequentially
    for i in range(1, num_radiation_qubits + 1):
        qc.h(0)  # Superposition on Black Hole
        qc.cx(0, i)  # Entangle with radiation qubit i

    return qc

# Function to add measurements
def add_measurements(qc, measure_qubits):
    measured_circuit = qc.copy()  # Create a fresh copy
    measured_circuit.add_register(ClassicalRegister(len(measure_qubits)))  # Add classical register
    measured_circuit.measure(measure_qubits, range(len(measure_qubits)))  # Measure specified qubits
    return measured_circuit


# Function to create the circuit with a specific charge state
def create_circuit_with_charge(charge_state):
    qc = QuantumCircuit(2)  # Create a new 2-qubit circuit
    if charge_state == "positive":
        qc.x(0)  # Set the black hole qubit to |1⟩ (positive charge)
    elif charge_state == "negative":
        qc.z(0)  # Introduce a phase flip (negative charge)
    elif charge_state == "neutral":
        pass  # Default to |0⟩ (neutral charge)
    
    # Step 2: Entangle the black hole and radiation qubits
    qc.h(0)  # Superposition on Black Hole qubit
    qc.cx(0, 1)  # Entangle Black Hole (Register A) and Radiation (Register B)
    return qc

# Function to create the circuit with charge injections
def create_circuit_with_alternating_charges(num_injections, num_radiation_qubits):
    qc = QuantumCircuit(num_radiation_qubits + 1)  # Black hole + radiation qubits

    # Alternate between injecting positive (X) and negative (Z) charge
    for i in range(num_injections):
        if i % 2 == 0:
            qc.x(0)  # Positive charge injection
        else:
            qc.z(0)  # Negative charge injection

        # Entangle black hole with radiation qubits sequentially
        for j in range(1, num_radiation_qubits + 1):
            qc.h(0)  # Superposition on Black Hole
            qc.cx(0, j)  # Entangle with radiation qubit j

    return qc

def create_1_qubit_circuit():
    qc = QuantumCircuit(1)
    qc.h(0) #Puts the state in superposition
    qc.cx(0, 1)
    retutun qc


# Function to create a circuit with prolonged charge injection
def create_circuit_with_prolonged_charges(num_iterations, cycle_length, num_radiation_qubits):
    qc = QuantumCircuit(num_radiation_qubits + 1)  # Black hole + radiation qubits

    for iteration in range(num_iterations):
        # Determine current charge based on cycle
        if (iteration // cycle_length) % 2 == 0:
            qc.x(0)  # Positive charge injection
        else:
            qc.z(0)  # Negative charge injection

        # Entangle black hole with radiation qubits sequentially
        for j in range(1, num_radiation_qubits + 1):
            qc.h(0)  # Superposition on Black Hole
            qc.cx(0, j)  # Entangle with radiation qubit j

    return qc

def create_circuit_with_time_gaps(num_injections, num_radiation_qubits, gap_cycles):
    qc = QuantumCircuit(num_radiation_qubits + 1)  # Black hole + radiation qubits

    for i in range(num_injections):
        # Inject charge
        if i % 2 == 0:
            qc.x(0)  # Positive charge injection
        else:
            qc.z(0)  # Negative charge injection

        # Entangle black hole with radiation qubits sequentially
        for j in range(1, num_radiation_qubits + 1):
            qc.h(0)  # Superposition on Black Hole
            qc.cx(0, j)  # Entangle with radiation qubit j

        # Add "time gap" as idle cycles
        qc.barrier()
        for _ in range(gap_cycles):
            qc.id(0)  # Idle gate to simulate a time gap

    return qc

def interpret_results(counts):
    print("\nInterpretation of Measurement Results:")
    
    # Identify the most and least common states
    dominant_state = max(counts, key=counts.get)
    least_common_state = min(counts, key=counts.get)
    
    print(f"Dominant state: {dominant_state} occurred {counts[dominant_state]} times.")
    print(f"Least common state: {least_common_state} occurred {counts[least_common_state]} times.")
    
    # Assess the distribution of measurements
    total_shots = sum(counts.values())
    entropy = -sum((count / total_shots) * np.log2(count / total_shots) for count in counts.values())
    
    print(f"Entropy of the distribution: {entropy:.4f} bits")
    print("Low entropy suggests a more deterministic system, potentially reflecting dominant charge patterns.")
    print("Higher entropy would indicate more randomness in radiation emission, aligning with time-dependency hypotheses.")


# Parameters
# Use Aer simulator
simulator = Aer.get_backend('aer_simulator')
shots = 8192
num_injections = 10  # Total charge injections
num_radiation_qubits = 4  # Number of radiation qubits
gap_cycles = 100  # Idle cycles between injections

# Create and run the circuit
qc = create_circuit_with_time_gaps(num_injections, num_radiation_qubits, gap_cycles)
qc_with_measurements = add_measurements(qc, range(1, num_radiation_qubits + 1))

# Run the simulation
job = simulator.run(qc_with_measurements, shots=8192)
result = job.result()
counts = result.get_counts()

# Analyze Results
print("Measurement Results:")
print(counts)

interpret_results(counts)
