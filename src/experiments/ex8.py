#Same experiment as we did for charge but for spin.
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, partial_trace, entropy
from qiskit.visualization import plot_histogram
import numpy as np

# Function to create a quantum circuit simulating black hole with spin conservation
def create_spin_circuit(spin_state):
    """
    Creates a quantum circuit representing a black hole and radiation system
    with initial spin injection.
    
    spin_state: "up" or "down"
    """
    qc = QuantumCircuit(2)  # 2 qubits: black hole and radiation

    # Initialize spin state for black hole
    if spin_state == "up":
        qc.initialize([1, 0], 0)  # Spin up state |0>
    elif spin_state == "down":
        qc.initialize([0, 1], 0)  # Spin down state |1>

    # Entangle black hole with radiation
    qc.h(0)  # Superposition for black hole qubit
    qc.cx(0, 1)  # Entangle black hole with radiation

    return qc

# Function to add measurement to the circuit
def add_measurements(qc):
    measured_circuit = qc.copy()
    measured_circuit.measure_all()
    return measured_circuit

# Analyze entanglement entropy
def analyze_entanglement(qc):
    state = Statevector.from_instruction(qc)  # Get statevector

    # Compute partial traces to analyze subsystems
    black_hole_state = partial_trace(state, [1])  # Trace out radiation qubit
    radiation_state = partial_trace(state, [0])  # Trace out black hole qubit

    # Compute von Neumann entropy for subsystems
    bh_entropy = entropy(black_hole_state, base=2)
    rad_entropy = entropy(radiation_state, base=2)

    return bh_entropy, rad_entropy

# Main experiment
def run_experiment(spin_states, shots=8192):
    simulator = AerSimulator()

    results = {}
    entropies = {}

    for spin_state in spin_states:
        print(f"Testing spin state: {spin_state}")

        # Create circuit for given spin state
        qc = create_spin_circuit(spin_state)

        # Analyze entanglement entropies
        bh_entropy, rad_entropy = analyze_entanglement(qc)
        entropies[spin_state] = {
            "Black Hole Entropy": bh_entropy,
            "Radiation Entropy": rad_entropy
        }

        # Add measurements and run simulation
        qc_with_measurements = add_measurements(qc)
        job = simulator.run(qc_with_measurements, shots=shots)
        result = job.result()

        # Record measurement results
        results[spin_state] = result.get_counts()

    return results, entropies

# Define spin states to test
spin_states = ["up", "down"]

# Run the experiment
results, entropies = run_experiment(spin_states)

# Display results
print("Measurement Results:")
for spin_state, counts in results.items():
    print(f"Spin State: {spin_state}, Counts: {counts}")

print("\nSubsystem Entropies:")
for spin_state, entropy_data in entropies.items():
    print(f"Spin State: {spin_state}, Entropies: {entropy_data}")
