
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
import numpy as np

# Define the hybrid color charge circuit
def create_hybrid_circuit(color_charge_1, color_charge_2, angle_a, angle_b, with_measurements = True):
    qc = QuantumCircuit(2)  # Two qubits: Black Hole and Radiation

    # Prepare the entangled Bell state
    qc.h(0)
    qc.cx(0, 1)

    # Apply first color charge rotation
    if color_charge_1 == "red":
        qc.rx(np.pi / 4, 0)
    elif color_charge_1 == "green":
        qc.rx(np.pi / 3, 0)
    elif color_charge_1 == "blue":
        qc.rx(np.pi / 6, 0)

    # Apply second color charge rotation
    if color_charge_2 == "red":
        qc.rx(np.pi / 4, 1)
    elif color_charge_2 == "green":
        qc.rx(np.pi / 3, 1)
    elif color_charge_2 == "blue":
        qc.rx(np.pi / 6, 1)

    # Apply measurement basis rotations
    qc.ry(angle_a, 0)
    qc.ry(angle_b, 1)

    if with_measurements:
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
def find_optimal_angles(color_charge_1, color_charge_2, shots):
    max_correlation = -1
    optimal_angles = (0, 0)

    for angle_a in np.linspace(0, 2 * np.pi, 20):
        for angle_b in np.linspace(0, 2 * np.pi, 20):
            qc = create_hybrid_circuit(color_charge_1, color_charge_2, angle_a, angle_b)
            simulator = Aer.get_backend('aer_simulator')
            job = simulator.run(qc, shots=shots, experiment_label=f"{color_charge_1}_{color_charge_2}_{angle_a:.2f}_{angle_b:.2f}")
            results = job.result().get_counts()
            if not results:
                print(f"No results for angles: {angle_a}, {angle_b}")
                continue


            correlation = compute_correlation(results, shots)
            if abs(correlation) > max_correlation:
                max_correlation = abs(correlation)
                optimal_angles = (angle_a, angle_b)

    return optimal_angles, max_correlation

# Function to analyze phases
def analyze_phases(color_charge_1, color_charge_2, angle_a, angle_b):
    qc = create_hybrid_circuit(color_charge_1, color_charge_2, angle_a, angle_b, with_measurements=False)
    state = Statevector.from_instruction(qc)  # Get the statevector after the circuit
    phases = np.angle(state.data)  # Extract the phases
    return phases
O
# Run Bell inequality tests for hybrid states with phase analysis
shots = 8192
color_charge_pairs = [("red", "green"), ("green", "blue"), ("red", "blue")]

for pair in color_charge_pairs:
    color_charge_1, color_charge_2 = pair
    print(f"Testing hybrid color charge: {color_charge_1} and {color_charge_2}")

    # Find optimal angles
    optimal_angles, max_correlation = find_optimal_angles(color_charge_1, color_charge_2, shots)
    print(f"Optimal angles for {color_charge_1} and {color_charge_2}: {optimal_angles}")
    print(f"Maximum correlation: {max_correlation}")

    # Test with optimal angles
    qc = create_hybrid_circuit(color_charge_1, color_charge_2, optimal_angles[0], optimal_angles[1])
    try:
        simulator = Aer.get_backend('aer_simulator')
    except Exception as e:
        print(f"Error initializing simulation: {e}")
        
    job = simulator.run(qc, shots=shots)
    results = job.result().get_counts()

    # Analyze results
    correlation = compute_correlation(results, shots)
    phases = analyze_phases(color_charge_1, color_charge_2, optimal_angles[0], optimal_angles[1])
    print(f"Measured correlation for {color_charge_1} and {color_charge_2}: {correlation}")
    print(f"Phases (Pre-Measurement): {phases}")
    print(f"Measurement results: {results}")
    print("\n")
