from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, entropy
import numpy as np

# Function to calculate entropies for a given set of angles
def calculate_combined_entropy(red_angle, green_angle, blue_angle):
    qc = QuantumCircuit(3)
    
    # Apply rotation angles for charges
    qc.rx(red_angle, 0)  # Red charge
    qc.ry(green_angle, 1)  # Green charge
    qc.rz(blue_angle, 2)  # Blue charge
    
    # Entangle with radiation (simulating black hole interaction)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    
    state = Statevector.from_instruction(qc)
    
    # Calculate individual and combined entropies
    red_entropy = entropy(partial_trace(state, [1, 2]))
    green_entropy = entropy(partial_trace(state, [0, 2]))
    blue_entropy = entropy(partial_trace(state, [0, 1]))
    combined_entropy = entropy(partial_trace(state, [0]))
    
    return red_entropy, green_entropy, blue_entropy, combined_entropy

# Grid search over angles
def test_dimensionality():
    angles = np.linspace(0, 2 * np.pi, 20)  # Test 50 angles from 0 to 2π
    results = []
    
    for red_angle in angles:
        for green_angle in angles:
            for blue_angle in angles:
                red_entropy, green_entropy, blue_entropy, combined_entropy = calculate_combined_entropy(
                    red_angle, green_angle, blue_angle)
                results.append((red_angle, green_angle, blue_angle, combined_entropy))
    
    # Find the configuration with maximum combined entropy
    optimal_config = max(results, key=lambda x: x[3])
    return optimal_config, results

# Main execution
if __name__ == "__main__":
    optimal_config, results = test_dimensionality()
    red_angle, green_angle, blue_angle, max_entropy = optimal_config
    
    print(f"Optimal Angles:")
    print(f"  Red Angle: {red_angle:.4f} radians")
    print(f"  Green Angle: {green_angle:.4f} radians")
    print(f"  Blue Angle: {blue_angle:.4f} radians")
    print(f"Maximum Combined Entropy: {max_entropy:.4f}")
