import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram

# Function to create a quantum circuit for radiation with a rotation angle
def create_circuit(rotation_angle, num_qubits):
    qc = QuantumCircuit(num_qubits + 1, num_qubits + 1)  # Add 1 extra qubit for rotation control
    
    # Apply Hadamard to the first qubit
    qc.h(0)

    # Apply parameterized phase and controlled-X gates
    for i in range(1, num_qubits + 1):
        qc.p(rotation_angle, 0)
        qc.cx(0, i)

    # Measure all qubits
    qc.measure_all()
    
    return qc

# Function to normalize measurement counts into probabilities
def calculate_probabilities(counts, shots):
    probabilities = {state: count / shots for state, count in counts.items()}
    return probabilities

# Function to analyze results and correlate with theoretical expectations
def analyze_results(rotation_angles, results, num_qubits):
    print("\nAnalysis Results:")
    
    for angle, counts in zip(rotation_angles, results):
        print(f"\nRotation Angle: {angle}")
        print(f"Measurement Counts: {counts}")

        # Calculate normalized probabilities
        total_shots = sum(counts.values())
        probabilities = calculate_probabilities(counts, total_shots)
        print(f"Normalized Probabilities: {probabilities}")

        # Find symmetrical properties
        symmetries = check_symmetry(probabilities, num_qubits)
        print(f"Symmetries Detected: {symmetries}")

# Function to check symmetry in probabilities (e.g., balanced states)
def check_symmetry(probabilities, num_qubits):
    symmetric_states = []
    for state, prob in probabilities.items():
        reversed_state = state[::-1]  # Reverse the bitstring
        if reversed_state in probabilities and np.isclose(prob, probabilities[reversed_state]):
            symmetric_states.append(state)
    return symmetric_states

# Main code
if __name__ == "__main__":
    num_qubits = 4  # Number of radiation qubits
    rotation_angles = [0, 0.1, 0.5, 1.0, 2.0]  # Rotation angles to test

    # Generate and analyze circuits for each rotation angle
    results = []
    for angle in rotation_angles:
        print(f"Running simulation with rotation angle: {angle}...")
        qc = create_circuit(angle, num_qubits)
        
        # Debug: Display circuit
        print("Circuit created successfully!")
        print(qc)

        # Transpile circuit (no backend simulation here)
        transpiled_qc = transpile(qc)
        print("Circuit transpiled successfully!")

        # Simulate results (mockup results for now)
        mock_counts = {
            '0' * num_qubits + '1' * num_qubits: np.random.randint(1000, 5000),
            '1' * num_qubits + '0' * num_qubits: np.random.randint(1000, 5000)
        }
        results.append(mock_counts)

    # Analyze all results
    analyze_results(rotation_angles, results, num_qubits)
