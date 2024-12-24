# Seeing if its a rotation issue for the green charge and if so by what angle
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, entropy, partial_trace
from qiskit_aer import AerSimulator

# Initialize Aer simulator
simulator = AerSimulator()

# Function to test green charge with varying angles
def test_green_charge(angles):
    results = {}
    for angle in angles:
        qc = QuantumCircuit(2)  # Create a 2-qubit circuit
        qc.ry(angle, 0)  # Apply rotation for green charge
        qc.h(0)  # Entangle black hole with radiation
        qc.cx(0, 1)
        
        # Measure the statevector
        state = Statevector.from_instruction(qc)
        ent = entropy(partial_trace(state,[1]))  # Entropy of black hole qubit
        results[angle] = ent
    return results

# Range of angles to test
angles = np.linspace(0, np.pi, 20)  # Test 20 angles from 0 to π
green_results = test_green_charge(angles)

# Print results
for angle, entropy_value in green_results.items():
    print(f"Angle: {angle:.2f}, Entropy: {entropy_value:.4f}")
