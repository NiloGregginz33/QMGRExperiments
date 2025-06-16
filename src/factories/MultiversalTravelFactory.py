from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, partial_trace, entropy
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import mutual_information

def multiverse_warp_circuit():
    """
    Creates a circuit to simulate a warp bubble interacting with alternate timelines.
    """
    # Create a quantum circuit with 5 qubits: 3 for the bubble, 2 for alternate timelines
    qc = QuantumCircuit(5)

    # Step 1: Initialize entanglement in the warp bubble
    qc.h(0)  # Superposition for the "center"
    qc.cx(0, 1)  # Entangle "center" with "boundary"
    qc.cx(1, 2)  # Extend entanglement to "outside"

    # Step 2: Initialize alternate timelines
    qc.h(3)  # Superposition for timeline 1
    qc.h(4)  # Superposition for timeline 2

    # Step 3: Apply interactions between bubble and timelines
    qc.cp(np.pi / 3, 0, 3)  # Controlled phase between center and timeline 1
    qc.cp(np.pi / 4, 1, 4)  # Controlled phase between boundary and timeline 2
    qc.cx(2, 3)  # Entangle outer region with timeline 1
    qc.cx(3, 4)  # Entangle timeline 1 with timeline 2

    # Step 4: Apply distortions to the timelines
    qc.rz(np.pi / 6, 3)  # Phase shift on timeline 1
    qc.rx(np.pi / 8, 4)  # Rotation on timeline 2

    return qc

def analyze_multiverse_entanglement(qc):
    """
    Analyzes the survival of entanglement and mutual information in a multiverse scenario.
    """
    # Get the quantum statevector
    state = Statevector.from_instruction(qc)

    # Trace out subsystems to calculate entropies
    subsystems = [
        partial_trace(state, [1, 2, 3, 4]),  # Center
        partial_trace(state, [0, 2, 3, 4]),  # Boundary
        partial_trace(state, [0, 1, 3, 4]),  # Outside
        partial_trace(state, [0, 1, 2, 4]),  # Timeline 1
        partial_trace(state, [0, 1, 2, 3])   # Timeline 2
    ]
    entropies = [entropy(sub, base=2) for sub in subsystems]

    # Combined subsystems for mutual information
    combined_center_timeline1 = partial_trace(state, [1, 2, 4])  # Center + Timeline 1
    combined_boundary_timeline2 = partial_trace(state, [0, 2, 3])  # Boundary + Timeline 2

    # Calculate mutual information
    mi_center_timeline1 = (
        entropies[0] + entropies[3] - entropy(combined_center_timeline1, base=2)
    )
    mi_boundary_timeline2 = (
        entropies[1] + entropies[4] - entropy(combined_boundary_timeline2, base=2)
    )

    # Print subsystem entropies
    print("Subsystem Entropies:")
    print(f"Center (Qubit 0): {entropies[0]:.4f}")
    print(f"Boundary (Qubit 1): {entropies[1]:.4f}")
    print(f"Outside (Qubit 2): {entropies[2]:.4f}")
    print(f"Timeline 1 (Qubit 3): {entropies[3]:.4f}")
    print(f"Timeline 2 (Qubit 4): {entropies[4]:.4f}")

    # Print mutual information
    print("\nMutual Information:")
    print(f"Center ↔ Timeline 1: {mi_center_timeline1:.4f}")
    print(f"Boundary ↔ Timeline 2: {mi_boundary_timeline2:.4f}")

    return entropies, mi_center_timeline1, mi_boundary_timeline2

def multiversal_decision_circuit():
    """
    Creates a quantum circuit where decisions in one timeline influence another.
    """
    # Create a quantum circuit with 5 qubits: 3 for the bubble, 2 for alternate timelines
    qc = QuantumCircuit(5)

    # Step 1: Initialize entanglement in the warp bubble
    qc.h(0)  # Superposition for the "center"
    qc.cx(0, 1)  # Entangle "center" with "boundary"
    qc.cx(1, 2)  # Extend entanglement to "outside"

    # Step 2: Initialize alternate timelines
    qc.h(3)  # Superposition for timeline 1
    qc.h(4)  # Superposition for timeline 2

    # Step 3: Apply interactions between bubble and timelines
    qc.cp(np.pi / 3, 0, 3)  # Controlled phase between center and timeline 1
    qc.cp(np.pi / 4, 1, 4)  # Controlled phase between boundary and timeline 2
    qc.cx(2, 3)  # Entangle outer region with timeline 1
    qc.cx(3, 4)  # Entangle timeline 1 with timeline 2

    # Step 4: Apply feedback loops based on measurements (influence Timeline 2)
    qc.measure(3, 0)  # Measure timeline 1 (classical communication)
    qc.cx(0, 4)  # Apply controlled-X operation to Timeline 2 based on Timeline 1's state
    qc.cz(1, 4)  # Apply controlled-Z operation to Timeline 2 based on Timeline 1's state

    # Step 5: Apply distortions to the timelines
    qc.rz(np.pi / 6, 3)  # Phase shift on timeline 1
    qc.rx(np.pi / 8, 4)  # Rotation on timeline 2

    # Measure the final states of the timelines
    qc.measure(3, 3)  # Measure timeline 1
    qc.measure(4, 4)  # Measure timeline 2

    return qc


# Main execution
if __name__ == "__main__":
    qc = multiverse_warp_circuit()

    # Simulate the quantum circuit on a qasm simulator
    backend = Aer.get_backend('qasm_simulator')
    job = backend.run(qc, shots=1024)
    result = job.result()

    # Get the results and visualize the outcomes
    counts = result.get_counts(qc)
    plot_histogram(counts)

