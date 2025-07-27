from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, partial_trace, entropy
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt

def multiverse_warp_circuit(t_steps):
    """
    Creates a dynamically evolving circuit to simulate a warp bubble interacting with alternate timelines.
    """
    qc = QuantumCircuit(5)  # 3 qubits for bubble, 2 for alternate timelines

    # Step 1: Initialize entanglement in the warp bubble
    qc.h(0)  # Superposition for the "center"
    qc.cx(0, 1)  # Entangle "center" with "boundary"
    qc.cx(1, 2)  # Extend entanglement to "outside"

    # Step 2: Initialize alternate timelines
    qc.h(3)  # Superposition for timeline 1
    qc.h(4)  # Superposition for timeline 2

    # Dynamic evolution over time
    for t in range(1, t_steps + 1):
        # Apply time-dependent interactions between bubble and timelines
        qc.cp(np.pi * t / 20, 0, 3)  # Center interacts with Timeline 1
        qc.cp(np.pi * t / 30, 1, 4)  # Boundary interacts with Timeline 2
        qc.rx(np.pi * t / 40, 3)  # Rotate Timeline 1
        qc.rz(np.pi * t / 50, 4)  # Phase shift Timeline 2

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
    print(f"Center \u2194 Timeline 1: {mi_center_timeline1:.4f}")
    print(f"Boundary \u2194 Timeline 2: {mi_boundary_timeline2:.4f}")

    return entropies, mi_center_timeline1, mi_boundary_timeline2

def visualize_results(t_steps, entropies, mi_values):
    """
    Visualizes subsystem entropies and mutual information over time.
    """
    entropies = np.array(entropies)
    mi_values = np.array(mi_values)

    # Plot entropies over time
    plt.figure(figsize=(12, 6))
    for i, label in enumerate(["Center", "Boundary", "Outside", "Timeline 1", "Timeline 2"]):
        plt.plot(range(1, t_steps + 1), entropies[:, i], label=f"{label} Entropy")
    plt.xlabel("Time Steps")
    plt.ylabel("Entropy")
    plt.title("Subsystem Entropies Over Time")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot mutual information over time
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, t_steps + 1), mi_values[:, 0], label="Center \u2194 Timeline 1")
    plt.plot(range(1, t_steps + 1), mi_values[:, 1], label="Boundary \u2194 Timeline 2")
    plt.xlabel("Time Steps")
    plt.ylabel("Mutual Information")
    plt.title("Mutual Information Over Time")
    plt.legend()
    plt.grid()
    plt.show()

# Main execution
if __name__ == "__main__":
    t_steps = 20  # Number of time steps for dynamic evolution
    qc = multiverse_warp_circuit(t_steps)

    # Analyze entanglement and mutual information dynamically
    entropies_over_time = []
    mi_values_over_time = []

    for t in range(1, t_steps + 1):
        step_qc = multiverse_warp_circuit(t)  # Create circuit for each time step
        entropies, mi_c_t1, mi_b_t2 = analyze_multiverse_entanglement(step_qc)
        entropies_over_time.append(entropies)
        mi_values_over_time.append((mi_c_t1, mi_b_t2))

    # Visualize results
    visualize_results(t_steps, entropies_over_time, mi_values_over_time)
