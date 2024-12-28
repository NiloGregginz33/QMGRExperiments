from qiskit import QuantumCircuit, Aer, transpile
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector, partial_trace, entropy
import matplotlib.pyplot as plt
import numpy as np

def create_holographic_timeline_circuit():
    """
    Creates a quantum circuit to simulate holographic timeline interactions.
    """
    qc = QuantumCircuit(2, 2)  # Two qubits (Black Hole and Radiation) with classical bits for measurement
    qc.h(0)  # Apply superposition to the Black Hole qubit
    qc.cx(0, 1)  # Entangle Black Hole with Radiation
    qc.rz(np.pi / 3, 0)  # Simulate timeline distortion on the Black Hole qubit
    qc.rx(np.pi / 4, 1)  # Simulate holographic interaction on the Radiation qubit
    qc.measure([0, 1], [0, 1])  # Measure both qubits
    return qc

def run_holographic_timeline_circuit(qc):
    """
    Runs the holographic timeline circuit and visualizes results.
    """
    # Use Aer simulator
    simulator = Aer.get_backend('aer_simulator')
    transpiled_circuit = transpile(qc, simulator)
    job = simulator.run(transpiled_circuit, shots=1024)
    result = job.result()
    counts = result.get_counts()
    
    # Plot the histogram of results
    plot_histogram(counts)
    plt.title("Holographic Timeline Interaction Results")
    plt.show()

    # Analyze statevector
    state = Statevector.from_instruction(qc)
    entropies = analyze_subsystem_entropy(state)
    
    print("Subsystem Entropies:", entropies)
    return counts, entropies

def analyze_subsystem_entropy(statevector):
    """
    Analyzes the entropy of subsystems (Black Hole and Radiation).
    """
    black_hole_state = partial_trace(statevector, [1])  # Trace out Radiation
    radiation_state = partial_trace(statevector, [0])  # Trace out Black Hole
    bh_entropy = entropy(black_hole_state, base=2)
    rad_entropy = entropy(radiation_state, base=2)
    return {
        "black_hole_entropy": bh_entropy,
        "radiation_entropy": rad_entropy
    }

# Example Execution
if __name__ == "__main__":
    holographic_circuit = create_holographic_timeline_circuit()
    results, entropies = run_holographic_timeline_circuit(holographic_circuit)
    print("Results:", results)
    print("Subsystem Entropies:", entropies)
