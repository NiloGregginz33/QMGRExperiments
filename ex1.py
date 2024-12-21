# This is basically to test if charge is included in the information leaked by hawking radiation
# We do this by "injecting" charge of an "uncollapsable" wavefunction into a simulated black holes then seeing if it affects the content
# of the information in the hawking radiation over many runs and see what comes out. We are going to be seeing
# if bells inequality or the entanglement entropy.

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

# Use Aer simulator for results
simulator = Aer.get_backend('aer_simulator')
iterations = 10
shots = 8192
charge_states = ["positive", "negative", "neutral"]
phase_results = {}
retrieval_results = {}
evaporation_results = {}

# Results storage
entropy_results = []
counts_results = {}

i = 0

for iteration in range(iterations):
    for charge in charge_states:
        print(f"Analyzing charge state: {charge}")

        # Phase Evolution Analysis
        qc = create_circuit_with_charge(charge)
        phases = analyze_phases(qc)
        phase_results[charge] = phases

        # Information Retrieval (Measure only Radiation Qubit)
        qc_with_measurement = add_measurements(qc, [1])  # Measure only qubit 1 (radiation)
        job = simulator.run(qc_with_measurement, shots=shots)
        result = job.result()
        retrieval_counts = result.get_counts()
        retrieval_results[charge] = retrieval_counts

        # Black Hole Evaporation Simulation
        qc_evaporation = simulate_evaporation(charge, num_radiation_qubits=3)
        qc_with_measurement_evap = add_measurements(qc_evaporation, range(1, 4))  # Measure all radiation qubits
        job = simulator.run(qc_with_measurement_evap, shots=shots)
        result = job.result()
        evaporation_counts = result.get_counts()
        evaporation_results[charge] = evaporation_counts

    i = 0

# Debugging: Check final dictionary contents
print("Counts Results Keys:", counts_results.keys())
print("Counts Results Values:", counts_results)

print("\nPhase Results:")
for charge, phases in phase_results.items():
    print(f"Charge: {charge}, Phases: {phases}")

print("\nRetrieval Results:")
for charge, counts in retrieval_results.items():
    print(f"Charge: {charge}, Counts: {counts}")

print("\nEvaporation Results:")
for charge, counts in evaporation_results.items():
    print(f"Charge: {charge}, Counts: {counts}")
