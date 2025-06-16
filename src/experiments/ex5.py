#Test for negative energy from the half of hawking radiation that fell into the black hole
#and how it changes over time
from qiskit_aer import Aer
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister
from qiskit.quantum_info import Statevector, partial_trace, entropy
from qiskit.visualization import plot_histogram
import numpy as np

# Function to create the initial quantum circuit
def create_black_hole_system(num_radiation_qubits):
    # Create a circuit with one black hole qubit, one negative energy qubit, and radiation qubits
    qc = QuantumCircuit(2 + num_radiation_qubits)  # Black Hole, Negative Energy, Radiation Qubits
    qc.h(0)  # Black Hole starts in superposition
    return qc

# Function to simulate Hawking radiation emission with a negative energy component
def simulate_hawking_evaporation(qc, num_radiation_qubits):
    for i in range(num_radiation_qubits):
        # Step 1: Introduce entanglement between black hole and radiation
        qc.h(0)  # Black Hole superposition
        qc.cx(0, 2 + i)  # Entangle Black Hole with Radiation Qubit
        
        # Step 2: Entangle Black Hole with Negative Energy Qubit
        qc.cx(0, 1)  # Couple Black Hole with Negative Energy
        
        # Step 3: Phase shift to simulate mass reduction (negative energy effect)
        qc.p(np.pi / (i + 1), 0)  # Gradually reduce the black hole's state
        
    return qc

# Function to add measurements
def add_measurements(qc, num_radiation_qubits):
    measured_circuit = qc.copy()  # Copy the circuit
    classical_bits = ClassicalRegister(num_radiation_qubits + 1)  # Add classical bits for radiation and black hole
    measured_circuit.add_register(classical_bits)
    measured_circuit.measure(range(2, 2 + num_radiation_qubits), range(num_radiation_qubits))  # Measure radiation
    measured_circuit.measure(0, num_radiation_qubits)  # Measure black hole
    return measured_circuit

def interpret_results(counts, final_state):
    print("\nInterpretation of Measurement Results:")
    
    # Identify the most and least common states
    dominant_state = max(counts, key=counts.get)
    least_common_state = min(counts, key=counts.get)
    total_shots = sum(counts.values())
    
    # Calculate entropy
    entropy = -sum((count / total_shots) * np.log2(count / total_shots) for count in counts.values())
    
    # Display results
    print(f"Dominant state: {dominant_state} occurred {counts[dominant_state]} times.")
    print(f"Least common state: {least_common_state} occurred {counts[least_common_state]} times.")
    print(f"Entropy of the distribution: {entropy:.4f} bits")
    print("- Low entropy suggests a more deterministic radiation emission process.")
    print("- Near-maximal entropy would indicate a more random distribution of states.")
    
    print("\nInterpretation of Final Statevector:")
    # Find the state with the highest amplitude
    max_amplitude_index = np.argmax(np.abs(final_state.data))
    max_amplitude_state = f"{max_amplitude_index:0{final_state.num_qubits}b}"
    max_amplitude_value = final_state.data[max_amplitude_index]
    
    print(f"State with the highest amplitude: |{max_amplitude_state}> with amplitude {max_amplitude_value}")
    print("- The final statevector reflects the combined effect of entanglement and negative energy influence.")
    print("- Observing patterns in the amplitudes can reveal the interplay between radiation and black hole dynamics.")


# Initialize the system
num_radiation_qubits = 4
qc = create_black_hole_system(num_radiation_qubits)

# Simulate Hawking radiation with negative energy
qc = simulate_hawking_evaporation(qc, num_radiation_qubits)

# Add measurements
qc_with_measurements = add_measurements(qc, num_radiation_qubits)

# Simulate using Aer
simulator = Aer.get_backend('aer_simulator')
job = simulator.run(qc_with_measurements, shots=8192)
result = job.result()

# Extract counts
counts = result.get_counts()
print("Measurement Results:")
print(counts)

# Analyze the statevector
final_state = Statevector.from_instruction(qc)
print("\nFinal Statevector:")
print(final_state)

interpret_results(counts, final_state)
