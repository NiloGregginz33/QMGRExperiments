from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
import numpy as np

# Define Bell inequality measurement circuit
def create_bell_test_circuit(color_charge, angle_a, angle_b):
    qc = QuantumCircuit(2)  # Two qubits: Black Hole and Radiation

    # Prepare the entangled Bell state
    qc.h(0)
    qc.cx(0, 1)

    # Apply color charge rotation
    if color_charge == "red":
        qc.rx(np.pi / 4, 0)  # Rotate Black Hole by pi/4
    elif color_charge == "green":
        qc.rx(np.pi / 3, 0)  # Rotate Black Hole by pi/3
    elif color_charge == "blue":
        qc.rx(np.pi / 6, 0)  # Rotate Black Hole by pi/6

    # Apply measurement basis rotations
    qc.ry(angle_a, 0)
    qc.ry(angle_b, 1)
    qc.measure_all()

    return qc

# Function to compute correlation coefficient
def compute_correlation(results, shots):
    p_00 = results.get('00', 0) / shots
    p_01 = results.get('01', 0) / shots
    p_10 = results.get('10', 0) / shots
    p_11 = results.get('11', 0) / shots

    correlation = (p_00 + p_11 - p_01 - p_10)
    return correlation

# Function to find optimal angles for correlation
def find_optimal_angles(color_charge, shots):
    max_correlation = -1
    optimal_angles = (0, 0)

    for angle_a in np.linspace(0, 2 * np.pi, 20):
        for angle_b in np.linspace(0, 2 * np.pi, 20):
            qc = create_bell_test_circuit(color_charge, angle_a, angle_b)
            simulator = Aer.get_backend('aer_simulator')
            job = simulator.run(qc, shots=shots)
            results = job.result().get_counts()

            correlation = compute_correlation(results, shots)
            if abs(correlation) > max_correlation:
                max_correlation = abs(correlation)
                optimal_angles = (angle_a, angle_b)

    return optimal_angles, max_correlation

# Run Bell inequality tests for each color charge
shots = 8192
color_charges = ["red", "green", "blue"]

for color in color_charges:
    print(f"Testing color charge: {color}")

    # Find optimal angles
    optimal_angles, max_correlation = find_optimal_angles(color, shots)
    print(f"Optimal angles for {color}: {optimal_angles}")
    print(f"Maximum correlation: {max_correlation}")

    # Test with optimal angles
    qc = create_bell_test_circuit(color, optimal_angles[0], optimal_angles[1])
    simulator = Aer.get_backend('aer_simulator')
    job = simulator.run(qc, shots=shots)
    results = job.result().get_counts()

    # Analyze results
    correlation = compute_correlation(results, shots)
    print(f"Measured correlation for {color}: {correlation}")
    print(f"Measurement results: {results}")
    print("\n")
