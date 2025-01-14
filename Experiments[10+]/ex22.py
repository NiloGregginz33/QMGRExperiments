from qiskit.quantum_info import Statevector
from qiskit_aer import Aer
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
import numpy as np

# Define the hybrid color charge circuit
def create_hybrid_circuit(color_charge_1, color_charge_2, angle_a, angle_b):
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
            job = simulator.run(qc, shots=shots)
            results = job.result().get_counts()

            correlation = compute_correlation(results, shots)
            if abs(correlation) > max_correlation:
                max_correlation = abs(correlation)
                optimal_angles = (angle_a, angle_b)

    return optimal_angles, max_correlation

# Function to analyze the phase evolution of the statevector
def analyze_phases(qc):
    # Ensure the circuit has no measurements or classical registers
    clean_circuit = qc.remove_final_measurements(inplace=False)  # Remove any measurement operations
    state = Statevector.from_instruction(clean_circuit)  # Get the statevector without measurement
    amplitudes = state.data  # Extract the amplitudes
    phases = np.angle(amplitudes)  # Compute the phases (in radians)
    return phases

def test_hybrid_with_phase_analysis(color_charge_1, color_charge_2, angle_a, angle_b, shots):
    # Create circuit without measurement for phase analysis
    qc = create_hybrid_circuit(color_charge_1, color_charge_2, angle_a, angle_b)
    phases = analyze_phases(qc)  # Analyze phases
    
    # Create circuit with measurement for correlation analysis
    qc_with_measurements = qc.copy()
    qc_with_measurements.measure_all()

    # Simulate measurement results
    simulator = Aer.get_backend('aer_simulator')
    job = simulator.run(qc_with_measurements, shots=shots)
    results = job.result().get_counts()

    correlation = compute_correlation(results, shots)

    print(f"Hybrid Pair: {color_charge_1} and {color_charge_2}")
    print(f"Measured Correlation: {correlation}")
    print(f"Phases (Pre-Measurement): {phases}")
    print(f"Measurement Results: {results}")
    print("\n")

def create_hybrid_circuit_with_adjustments(color_charge_1, color_charge_2, angle_a, angle_b):
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    # Adjusted encoding for green charge
    if color_charge_1 == "green":
        qc.rx(np.pi / 3, 0)
        qc.rz(np.pi / 6, 0)  # Additional phase rotation
    elif color_charge_2 == "green":
        qc.rx(np.pi / 3, 1)
        qc.rz(np.pi / 6, 1)

    qc.ry(angle_a, 0)
    qc.ry(angle_b, 1)
    qc.measure_all()

    return qc

# Define hybrid color charge pairs
color_charge_pairs = [
    ("red", "green"),
    ("green", "blue"),
    ("red", "blue")
]

# Run phase analysis for hybrid pairs
shots = 8192
for pair in color_charge_pairs:
    color_charge_1, color_charge_2 = pair
    optimal_angles, _ = find_optimal_angles(color_charge_1, color_charge_2, shots)
    test_hybrid_with_phase_analysis(color_charge_1, color_charge_2, optimal_angles[0], optimal_angles[1], shots)
    create_hybrid_circuit_with_adjustments(color_charge_1, color_charge_2, optimal_angles[0], optimal_angles[1])
