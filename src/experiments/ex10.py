# Testing an idea the radiation must be red-blue bc why is green so weird?
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, partial_trace, entropy
import numpy as np

# Function to create a quantum circuit simulating black hole with color charge conservation
def create_color_charge_circuit(color_charge):
    """
    Creates a quantum circuit representing a black hole and radiation system
    with initial color charge injection.

    color_charge: "red", "green", or "blue"
    """
    qc = QuantumCircuit(2)  # 2 qubits: black hole and radiation

    # Initialize color charge state for black hole
    if color_charge == "red":
        qc.rx(np.pi / 2, 0)  # Rotate to simulate "red" charge
    elif color_charge == "green":
        qc.ry(np.pi / 3.5, 0)
        qc.rz(np.pi / 3.5, 0)
        qc.rx(np.pi / 3.5, 0)# Rotate to simulate "green" charge
        state = Statevector.from_instruction(qc)  # Debugging statevector
        print(f"Statevector for {color_charge} charge:", state)
    elif color_charge == "blue":
        qc.rz(np.pi / 2, 0)  # Rotate to simulate "blue" charge

    # Entangle black hole with radiation
    qc.h(0)  # Superposition for black hole qubit
    qc.cx(0, 1)  # Entangle black hole with radiation

    return qc

# Function to analyze entanglement entropies
def analyze_entanglement(qc):
    """
    Analyze entanglement entropy of the system
    """
    state = Statevector.from_instruction(qc)  # Get statevector

    # Compute partial traces to analyze subsystems
    black_hole_state = partial_trace(state, [1])  # Trace out radiation qubit
    radiation_state = partial_trace(state, [0])  # Trace out black hole qubit

    # Compute von Neumann entropy for subsystems
    bh_entropy = entropy(black_hole_state, base=2)
    rad_entropy = entropy(radiation_state, base=2)

    return bh_entropy, rad_entropy

# Function to add measurement to the circuit
def add_measurements(qc):
    """
    Add measurements to the circuit
    """
    measured_circuit = qc.copy()
    measured_circuit.measure_all()
    return measured_circuit

# Main experiment
def run_experiment(color_charges, shots=8192):
    """
    Run the experiment for multiple color charges
    """
    simulator = AerSimulator()

    results = {}
    entropies = {}

    for color_charge in color_charges:
        print(f"Testing color charge: {color_charge}")

        # Create circuit for given color charge
        qc = create_color_charge_circuit(color_charge)

        # Analyze entanglement entropies
        bh_entropy, rad_entropy = analyze_entanglement(qc)
        entropies[color_charge] = {
            "Black Hole Entropy": bh_entropy,
            "Radiation Entropy": rad_entropy
        }

        # Add measurements and run simulation
        qc_with_measurements = add_measurements(qc)
        job = simulator.run(qc_with_measurements, shots=shots)
        result = job.result()

        # Record measurement results
        results[color_charge] = result.get_counts()

    return results, entropies

# Define color charges to test
color_charges = ["red", "green", "blue"]

# Run the experiment
results, entropies = run_experiment(color_charges)

# Display results
print("Measurement Results:")
for color_charge, counts in results.items():
    print(f"Color Charge: {color_charge}, Counts: {counts}")

print("\nSubsystem Entropies:")
for color_charge, entropy_data in entropies.items():
    print(f"Color Charge: {color_charge}, Entropies: {entropy_data}")
