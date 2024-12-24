import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector, entropy, partial_trace
from qiskit import QuantumCircuit

# Function to prepare a state with given angles for red, green, and blue
def prepare_combined_state(red_angle, green_angle, blue_angle):
    qc = QuantumCircuit(3)
    qc.rx(red_angle, 0)  # Red charge
    qc.ry(green_angle, 1)  # Green charge
    qc.rz(blue_angle, 2)  # Blue charge
    return Statevector.from_instruction(qc)

# Function to compute entropies for a given state
def compute_entropies(state):
    black_hole_entropy = entropy(partial_trace(state, [1, 2]))  # Trace out green and blue
    radiation_entropy = entropy(partial_trace(state, [0]))  # Trace out red
    return black_hole_entropy, radiation_entropy

# Parameters
num_points = 20  # Resolution for each angle
angles = np.linspace(0, 2 * np.pi, num_points)

# Initialize results
entropies = np.zeros((num_points, num_points, num_points))

# Iterate over all combinations of angles
for i, red_angle in enumerate(angles):
    for j, green_angle in enumerate(angles):
        for k, blue_angle in enumerate(angles):
            state = prepare_combined_state(red_angle, green_angle, blue_angle)
            black_hole_entropy, radiation_entropy = compute_entropies(state)
            entropies[i, j, k] = black_hole_entropy + radiation_entropy

# Find maximum entropy and corresponding angles
max_entropy = np.max(entropies)
max_indices = np.unravel_index(np.argmax(entropies), entropies.shape)
optimal_red_angle = angles[max_indices[0]]
optimal_green_angle = angles[max_indices[1]]
optimal_blue_angle = angles[max_indices[2]]

print(f"Optimal Angles:")
print(f"  Red Angle: {optimal_red_angle:.4f} radians")
print(f"  Green Angle: {optimal_green_angle:.4f} radians")
print(f"  Blue Angle: {optimal_blue_angle:.4f} radians")
print(f"Maximum Combined Entropy: {max_entropy:.4f}")

# Visualization (Slice through green vs. blue for optimal red)
slice_index = max_indices[0]
slice_data = entropies[slice_index, :, :]

plt.figure(figsize=(8, 6))
plt.imshow(slice_data, extent=[0, 2 * np.pi, 0, 2 * np.pi], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='Entropy')
plt.title('Entropy Variation with Three Angles (Green vs. Blue, Optimal Red)')
plt.xlabel('Green Angle (rad)')
plt.ylabel('Blue Angle (rad)')
plt.show()
