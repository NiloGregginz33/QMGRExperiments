from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, partial_trace, entropy

def analyze_entanglement(qc):
    """
    Analyzes the survival of entanglement without classical measurements.
    """
    # Get the quantum statevector
    state = Statevector.from_instruction(qc)
    
    # Trace out subsystems to calculate entropies
    subsystems = [partial_trace(state, [1, 2]),  # Trace out boundary & outside
                  partial_trace(state, [0, 2]),  # Trace out center & outside
                  partial_trace(state, [0, 1])]  # Trace out center & boundary

    # Calculate entropy for each subsystem
    entropies = [entropy(sub, base=2) for sub in subsystems]

    print("Subsystem Entropies:")
    print(f"Center (Qubit 0): {entropies[0]:.4f}")
    print(f"Boundary (Qubit 1): {entropies[1]:.4f}")
    print(f"Outside (Qubit 2): {entropies[2]:.4f}")

    return entropies



def warp_information_circuit():
    """
    Creates a 3-qubit circuit to test entanglement survival under spacetime distortions.
    """
    qc = QuantumCircuit(3)  # Three qubits

    # Step 1: Initialize entanglement
    qc.h(0)  # Superposition for the "center"
    qc.cx(0, 1)  # Entangle "center" with "boundary"
    qc.cx(1, 2)  # Extend entanglement to "outside"

    # Step 2: Apply spacetime distortions
    qc.rz(np.pi / 4, 0)  # Phase shift on the "center"
    qc.cp(np.pi / 3, 0, 1)  # Controlled phase between "center" and "boundary"
    qc.cp(np.pi / 6, 1, 2)  # Controlled phase between "boundary" and "outside"
    qc.rx(np.pi / 8, 2)  # Additional distortion on "outside"

    return qc

def run_with_measurement(qc):
    """
    Runs the circuit with measurements and visualizes results.
    """
    qc_with_measure = qc.copy()
    qc_with_measure.measure_all()  # Add measurements to a copy of the circuit

    simulator = Aer.get_backend('aer_simulator')
    transpiled = transpile(qc_with_measure, simulator)
    result = simulator.run(transpiled, shots=1024).result()
    counts = result.get_counts()

    # Display measurement results
    plot_histogram(counts)
    plt.title("Warp Information Transfer Results")
    plt.show()

    return counts

# Run the circuit
# Main execution
if __name__ == "__main__":
    qc = warp_information_circuit()

    # Analyze entanglement without measurements
    entropies = analyze_entanglement(qc)

    # Run circuit with measurements
    results = run_with_measurement(qc)
    print("Measurement Results:", results)
