import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, entropy, partial_trace

def find_perfect_angles():
    """Find the perfect angles for red, green, and blue to minimize combined entropy."""
    angles = np.linspace(0, 2 * np.pi, 20)
    best_angles = None
    min_entropy = float('inf')

    for red_angle in angles:
        for green_angle in angles:
            for blue_angle in angles:
                qc = QuantumCircuit(3)
                qc.rx(red_angle, 0)
                qc.rx(green_angle, 1)
                qc.rx(blue_angle, 2)

                # Measure entropies
                state = Statevector.from_instruction(qc)
                bh_entropy = entropy(partial_trace(state, [1, 2]))
                rad_entropy = entropy(partial_trace(state, [0]))

                combined_entropy = bh_entropy + rad_entropy

                if combined_entropy < min_entropy:
                    min_entropy = combined_entropy
                    best_angles = (red_angle, green_angle, blue_angle)

    return best_angles, min_entropy

def simulate_bell_test(angles, perturbation=np.pi/4):
    """Simulate a Bell-like test with perturbed angles."""
    red_angle, green_angle, blue_angle = angles
    perturb_angles = [red_angle + perturbation, green_angle + perturbation, blue_angle + perturbation]

    # Original setup
    qc_original = QuantumCircuit(3)
    qc_original.rx(red_angle, 0)
    qc_original.rx(green_angle, 1)
    qc_original.rx(blue_angle, 2)

    # Perturbed setup
    qc_perturbed = QuantumCircuit(3)
    qc_perturbed.rx(perturb_angles[0], 0)
    qc_perturbed.rx(perturb_angles[1], 1)
    qc_perturbed.rx(perturb_angles[2], 2)

    # Measure entropy correlations
    original_state = Statevector.from_instruction(qc_original)
    perturbed_state = Statevector.from_instruction(qc_perturbed)

    bh_entropy_original = entropy(partial_trace(original_state, [1, 2]))
    rad_entropy_original = entropy(partial_trace(original_state, [0]))

    bh_entropy_perturbed = entropy(partial_trace(perturbed_state, [1, 2]))
    rad_entropy_perturbed = entropy(partial_trace(perturbed_state,[0]))

    return {
        "original": (bh_entropy_original, rad_entropy_original),
        "perturbed": (bh_entropy_perturbed, rad_entropy_perturbed),
    }

# Main execution
optimal_angles, min_entropy = find_perfect_angles()
simulator = Aer.get_backend('aer_simulator')
shots = 8192
print(f"Optimal Angles: Red: {optimal_angles[0]:.4f}, Green: {optimal_angles[1]:.4f}, Blue: {optimal_angles[2]:.4f}")
print(f"Minimum Combined Entropy: {min_entropy:.4f}")

results = simulate_bell_test(optimal_angles)
print("Original Entropies:", results["original"])
print("Perturbed Entropies:", results["perturbed"])
