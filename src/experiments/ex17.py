from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, entropy, partial_trace
import numpy as np
import matplotlib.pyplot as plt

# Function to test entropy for two varying angles
def test_two_angle_variation(fixed_color, fixed_angle, var1, var2, resolution=50):
    angles_var1 = np.linspace(0, 2 * np.pi, resolution)
    angles_var2 = np.linspace(0, 2 * np.pi, resolution)
    entropies = np.zeros((resolution, resolution))

    for i, angle1 in enumerate(angles_var1):
        for j, angle2 in enumerate(angles_var2):
            qc = QuantumCircuit(2)

            # Apply rotations for fixed color
            if fixed_color == "red":
                qc.rx(fixed_angle, 0)
            elif fixed_color == "green":
                qc.ry(fixed_angle, 0)
            elif fixed_color == "blue":
                qc.rz(fixed_angle, 0)

            # Apply rotations for varying angles
            qc.ry(angle1, 1)  # Assume first varying angle is green
            qc.rz(angle2, 1)  # Assume second varying angle is blue

            state = Statevector.from_instruction(qc)
            bh_entropy = entropy(partial_trace(state, [1]))  # Black hole entropy
            entropies[i, j] = bh_entropy

    return angles_var1, angles_var2, entropies

# Run experiment for fixed red angle and varying green and blue angles
fixed_color = "red"
fixed_angle = np.pi / 3  # Use the optimal angle from previous experiments
var1, var2, entropies = test_two_angle_variation(fixed_color, fixed_angle, "green", "blue")

# Plotting the results
plt.figure(figsize=(8, 6))
plt.contourf(var1, var2, entropies, cmap='viridis', levels=50)
plt.colorbar(label="Entropy")
plt.title("Entropy Variation with Two Angles (Green and Blue)")
plt.xlabel("Green Angle (rad)")
plt.ylabel("Blue Angle (rad)")
plt.show()
