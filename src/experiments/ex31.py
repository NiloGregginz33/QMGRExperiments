from qiskit import QuantumCircuit, execute
from qiskit_aer import Aer
from qiskit.circuit.library import RZGate, CXGate
import numpy as np

def simulate_quantum_gravity(num_qubits, rotation_angle, curvature_effect):
    """
    Simulate a black hole system with quantum gravity-inspired modifications.
    """
    qc = QuantumCircuit(num_qubits + 1, num_qubits)  # +1 for black hole qubit

    # Initialize black hole qubit in superposition
    qc.h(0)  # Black hole in superposition
    qc.append(RzGate(rotation_angle), [0])  # Simulate black hole rotation

    # Entangle radiation qubits with black hole, incorporating curvature effects
    for i in range(1, num_qubits + 1):
        qc.cx(0, i)  # Entangle black hole with radiation
        qc.append(RzGate(curvature_effect * (1 / (i + 1))), [i])  # Curvature gradient

    # Measure radiation qubits
    qc.measure(range(1, num_qubits + 1), range(num_qubits))

    return qc

def calculate_entropy(counts):
    """
    Calculate subsystem entropy based on measurement results.
    """
    total_shots = sum(counts.values())
    probs = np.array(list(counts.values())) / total_shots
    entropy = -np.sum(probs * np.log2(probs + 1e-12))  # Avoid log(0)
    return entropy

# Parameters
num_qubits = 4  # Number of radiation qubits
rotation_angle = np.pi / 4  # Simulated rotation of black hole
curvature_effect = 0.1  # Gravitational curvature effect

# Simulate the quantum gravity circuit
qc = simulate_quantum_gravity(num_qubits, rotation_angle, curvature_effect)
backend = Aer.get_backend('aer_simulator')
job = backend.run(qc, shots=1024)
results = job.result().get_counts()

# Calculate entropy for the radiation subsystem
radiation_entropy = calculate_entropy(results)

# Display results
print("Quantum Gravity Simulation Results:")
print(f"Measurement Counts: {results}")
print(f"Radiation Subsystem Entropy: {radiation_entropy:.3f}")

# Optional: Visualize the circuit
print("\nQuantum Circuit:")
print(qc.draw())
