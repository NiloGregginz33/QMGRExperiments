# test of green color charge and bells theorem
from qiskit import QuantumCircuit
from qiskit_aer import Aer, AerSimulator
from qiskit.circuit.library import RYGate
import numpy as np

# Function to create a Bell test circuit for green color charge
def create_bell_test_circuit(angle):
    qc = QuantumCircuit(2, 2)

    # Create entanglement
    qc.h(0)
    qc.cx(0, 1)

    # Simulate green color charge effect with a rotation
    qc.append(RYGate(angle), [0])

    # Measurement basis for testing Bell inequality
    qc.h(1)
    qc.measure([0, 1], [0, 1])
    
    return qc

# Function to simulate the Bell test
def run_bell_test(angle):
    # Create the circuit
    qc = create_bell_test_circuit(angle)

    # Simulated backend
    backend = Aer.get_backend('aer_simulator')

    # Execute the circuit
    job = backend.run(qc, shots=1000)
    result = job.result()

    # Get measurement results
    counts = result.get_counts()

    # Calculate correlation
    p00 = counts.get('00', 0) / 1000
    p01 = counts.get('01', 0) / 1000
    p10 = counts.get('10', 0) / 1000
    p11 = counts.get('11', 0) / 1000
    
    correlation = (p00 + p11) - (p01 + p10)

    return counts, correlation

# Test with different angles simulating green color charge effect
angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]

print("Running Bell test simulations...")
for angle in angles:
    counts, correlation = run_bell_test(angle)
    print(f"Angle: {angle:.2f} rad")
    print(f"Counts: {counts}")
    print(f"Correlation: {correlation:.2f}\n")
