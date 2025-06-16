#Want to see if time dilation affects this all
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.circuit.library import RZGate
from qiskit.quantum_info import Statevector, partial_trace, entropy
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np

def create_black_hole_circuit(time_dilation_phase):
    qc = QuantumCircuit(4)
    
    qc.h(0)
    qc.h(1)  # Black hole 2 in superposition

    # Step 2: Entangle black holes (mimic initial quantum connection)
    qc.cx(0, 1)

    # Step 3: Apply time dilation effect to Black Hole 2
    qc.append(RZGate(time_dilation_phase), [1])  # Add time dilation as phase rotation

    # Step 4: Emit entangled radiation
    qc.cx(0, 2)  # Radiation from Black Hole 1
    qc.cx(1, 3)  # Radiation from Black Hole 2

    # Step 5: Entangle radiation qubits to simulate shared information
    qc.cx(2, 3)

    return qc

# Function to analyze the entanglement and radiation results
def analyze_results(qc, shots=8192):
    # Create a copy of the circuit without measurement for statevector analysis
    qc_no_measurement = qc.copy()
    
    # Calculate the statevector
    state = Statevector.from_instruction(qc_no_measurement)
    subsystems = [partial_trace(state, [i]) for i in range(4)]  # Trace out each qubit
    entropies = [entropy(subsystem) for subsystem in subsystems]
    
    # Add measurements to a new circuit for simulation
    qc_with_measurement = qc.copy()
    qc_with_measurement.measure_all()
    
    simulator = Aer.get_backend('aer_simulator')
    job = simulator.run(qc_with_measurement, shots=shots)
    result = job.result()
    counts = result.get_counts()

    return counts, entropies

# Experiment parameters
time_dilation_phase = np.pi / 2  # Simulate moderate time dilation
qc = create_black_hole_circuit(time_dilation_phase)
counts, entropies = analyze_results(qc)

# Display results
print("Measurement Results:")
print(counts)
print("\nSubsystem Entropies:")
for i, ent in enumerate(entropies):
    print(f"Qubit {i}: Entropy = {ent:.4f}")

# Visualize the measurement results
plot_histogram(counts)
plt.show()
