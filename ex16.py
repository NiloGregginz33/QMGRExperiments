import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, entropy, partial_trace

# Define optimal angles for red and blue
optimal_red_angle = 0.3307
optimal_blue_angle = 0.6614

# Function to create circuit with fixed angles for red and blue, varying green
def create_circuit(red_angle, green_angle, blue_angle):
    qc = QuantumCircuit(3)
    
    # Apply rotations for each color charge
    qc.rx(red_angle, 0)  # Red charge
    qc.ry(green_angle, 1)  # Green charge
    qc.rz(blue_angle, 2)  # Blue charge

    # Entangle the qubits
    qc.cx(0, 1)
    qc.cx(1, 2)

    return qc

# Function to calculate combined entropy for the circuit
def calculate_combined_entropy(qc):
    state = Statevector.from_instruction(qc)
    
    # Trace out one subsystem to calculate entropy of remaining subsystems
    reduced_state_black_hole = partial_trace(state, [1, 2])  # Trace out green and blue
    reduced_state_radiation = partial_trace(state, [0, 2])  # Trace out red and blue

    black_hole_entropy = entropy(reduced_state_black_hole)
    radiation_entropy = entropy(reduced_state_radiation)

    return black_hole_entropy, radiation_entropy

# Exploration range for the green angle
green_angles = np.linspace(0, 2 * np.pi, 100)

# Store results
results = []

for green_angle in green_angles:
    # Create circuit
    qc = create_circuit(optimal_red_angle, green_angle, optimal_blue_angle)
    
    # Calculate entropies
    black_hole_entropy, radiation_entropy = calculate_combined_entropy(qc)
    
    results.append((green_angle, black_hole_entropy, radiation_entropy))

# Print results
for angle, black_entropy, rad_entropy in results:
    print(f"Green Angle: {angle:.4f}, Black Hole Entropy: {black_entropy:.4f}, Radiation Entropy: {rad_entropy:.4f}")
