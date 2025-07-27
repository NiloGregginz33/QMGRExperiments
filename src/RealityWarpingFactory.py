from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector, partial_trace, entropy
import matplotlib.pyplot as plt
import numpy as np
from qiskit_aer import Aer

def calculate_entropies(state):
    """
    Calculates entropy for subsystems in the quantum state.
    """
    # Trace out specific qubits to isolate subsystems
    subsystem_0 = partial_trace(state, [1, 2])  # Isolate Qubit 0
    subsystem_1 = partial_trace(state, [0, 2])  # Isolate Qubit 1
    subsystem_2 = partial_trace(state, [0, 1])  # Isolate Qubit 2

    # Calculate Von Neumann entropy
    entropy_0 = entropy(subsystem_0, base=2)
    entropy_1 = entropy(subsystem_1, base=2)
    entropy_2 = entropy(subsystem_2, base=2)

    return {"Qubit 0": entropy_0, "Qubit 1": entropy_1, "Qubit 2": entropy_2}

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
    plt.show(block=False)

    # Analyze statevector
    state = Statevector.from_instruction(qc)
    entropies = analyze_subsystem_entropy(state)
    
    print("Subsystem Entropies:", entropies)
    return counts, entropies

def warp_hologram_circuit():
    """
    Creates a simple circuit to simulate holographic warp effects.
    """
    # Create a quantum circuit with 3 qubits and 3 classical bits
    qc = QuantumCircuit(3, 3)

    # Initialize the hologram (entanglement and superposition)
    qc.h(0)  # Superposition for the "center" of the hologram
    qc.cx(0, 1)  # Entangle "center" with "outside"
    qc.cx(1, 2)  # Extend entanglement to the third qubit

    # Apply warp-like effects with phase shifts
    qc.rz(np.pi / 2, 0)  # Phase shift on the "center"
    qc.cp(np.pi / 4, 0, 1)  # Controlled phase shift between qubits

    # Measure all qubits
    qc.measure([0, 1, 2], [0, 1, 2])

    return qc

def dynamic_warp_circuit(t_steps):
    """
    Simulates a dynamically evolving warp hologram.
    """
    qc = QuantumCircuit(3, 3)

    # Initialize entanglement
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)

    # Time evolution with phase shifts
    for t in range(1, t_steps + 1):
        qc.rz(np.pi * t / 10, 0)  # Time-dependent phase shift
        qc.rx(np.pi * t / 15, 1)  # Rotation on Qubit 1
        qc.cp(np.pi * t / 20, 1, 2)  # Controlled phase interaction

    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def run_warp_simulation(qc):
    """
    Runs the warp simulation circuit and displays results.
    """
    # Use the Aer simulator
    simulator = Aer.get_backend('aer_simulator')
    transpiled = transpile(qc, simulator)
    result = simulator.run(transpiled, shots=1024).result()
    counts = result.get_counts()

    # Display results as a histogram
    plot_histogram(counts)
    plt.title("Warp Hologram Simulation Results")
    plt.show()

    return counts


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

t = 8

qc = create_holographic_timeline_circuit()
counts, entropies = run_holographic_timeline_circuit(qc)

print("Results:", counts, " Entropies: ", entropies)
