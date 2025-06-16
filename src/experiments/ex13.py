from qiskit.quantum_info import entropy, Statevector, partial_trace
import numpy as np
from qiskit import QuantumCircuit

# Function to test a range of rotation angles
def find_optimal_angle(step=0.1):
    optimal_angle = 0
    max_entropy = 0
    angles = np.arange(0, 2 * np.pi, step)
    entropies = []

    for angle in angles:
        qc = QuantumCircuit(2)
        qc.h(0)  # Initialize black hole in superposition
        qc.cx(0, 1)  # Entangle black hole with radiation

        # Apply green charge rotations
        qc.ry(angle, 0)
        qc.rz(angle, 0)
        qc.rx(angle, 0)

        # Get statevector and calculate entropy
        state = Statevector.from_instruction(qc)
        bh_entropy = entropy(partial_trace(state, [1]))  # Entropy of black hole
        entropies.append(bh_entropy)

        print(f"Angle: {angle:.4f}, Statevector: {state}, Entropy: {bh_entropy:.4f}")

        # Update optimal angle if entropy is higher
        if bh_entropy > max_entropy:
            max_entropy = bh_entropy
            optimal_angle = angle

    return optimal_angle, max_entropy, angles, entropies

# Run the angle search
optimal_angle, max_entropy, angles, entropies = find_optimal_angle()

# Print results
print(f"Optimal Angle: {optimal_angle:.4f} radians")
print(f"Maximum Entropy: {max_entropy:.4f}")

# Optional: Plot the results
import matplotlib.pyplot as plt
plt.plot(angles, entropies)
plt.xlabel("Rotation Angle (radians)")
plt.ylabel("Entropy")
plt.title("Entropy vs Rotation Angle for Green Charge")
plt.show()
