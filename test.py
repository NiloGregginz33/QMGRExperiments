from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, partial_trace, entropy
import numpy as np
import matplotlib.pyplot as plt
from qiskit_aer import Aer

def enhanced_multi_bubble_circuit(t_steps):
    """
    Creates a circuit with enhanced inter-bubble entanglement and nonlinear dynamics.
    """
    # 3 qubits per bubble, 2 timelines per bubble
    qc = QuantumCircuit(10)

    # Initialize Bubble 1
    qc.h(0)  # Center
    qc.cx(0, 1)  # Center ↔ Boundary
    qc.cx(1, 2)  # Boundary ↔ Outside

    # Initialize Bubble 2
    qc.h(5)  # Center
    qc.cx(5, 6)  # Center ↔ Boundary
    qc.cx(6, 7)  # Boundary ↔ Outside

    # Initialize Timelines
    qc.h(3)  # Timeline 1 (Bubble 1)
    qc.h(4)  # Timeline 2 (Bubble 1)
    qc.h(8)  # Timeline 1 (Bubble 2)
    qc.h(9)  # Timeline 2 (Bubble 2)

    # Enhanced entanglement during initialization
    qc.cx(0, 5)  # Bubble 1 Center ↔ Bubble 2 Center
    qc.cx(2, 7)  # Bubble 1 Outside ↔ Bubble 2 Outside
    qc.cx(1, 6)  # Bubble 1 Boundary ↔ Bubble 2 Boundary
    qc.cx(3, 8)  # Timeline 1 ↔ Timeline 1
    qc.cx(4, 9)  # Timeline 2 ↔ Timeline 2

    # Time-dependent, nonlinear interactions
    for t in range(1, t_steps + 1):
        distortion = np.exp(-t / 5)  # Exponential decay factor
        qc.cp(np.pi * distortion, 0, 3)  # Bubble 1 Center ↔ Timeline 1
        qc.cp(np.pi * distortion / 2, 1, 4)  # Bubble 1 Boundary ↔ Timeline 2
        qc.cp(np.pi * distortion, 5, 8)  # Bubble 2 Center ↔ Timeline 1
        qc.cp(np.pi * distortion / 2, 6, 9)  # Bubble 2 Boundary ↔ Timeline 2

        # Strengthen inter-bubble interactions
        qc.cp(np.pi / (t + 1), 2, 5)  # Bubble 1 Outside ↔ Bubble 2 Center
        qc.cp(np.pi / (t + 1), 7, 0)  # Bubble 2 Outside ↔ Bubble 1 Center

    return qc

def analyze_enhanced_bubble(qc):
    """
    Analyzes entanglement and mutual information for enhanced multi-bubble systems.
    """
    state = Statevector.from_instruction(qc)

    # Define subsystems
    subsystems = [
        partial_trace(state, list(range(1, 10))),  # Bubble 1 Center
        partial_trace(state, [0] + list(range(2, 10))),  # Bubble 1 Boundary
        partial_trace(state, list(range(3, 10))),  # Bubble 1 Outside
        partial_trace(state, list(range(0, 5)) + list(range(6, 10))),  # Bubble 2 Center
        partial_trace(state, list(range(0, 6)) + list(range(7, 10))),  # Bubble 2 Boundary
        partial_trace(state, list(range(0, 7)) + list(range(8, 10)))   # Bubble 2 Outside
    ]

    # Calculate entropies
    entropies = [entropy(sub, base=2) for sub in subsystems]

    # Mutual Information
    combined_b1_c_b2_c = partial_trace(state, [1, 2, 4, 5])
    combined_b1_o_b2_o = partial_trace(state, [0, 3, 6, 7])
    mi_b1_c_b2_c = entropies[0] + entropies[3] - entropy(combined_b1_c_b2_c, base=2)
    mi_b1_o_b2_o = entropies[2] + entropies[5] - entropy(combined_b1_o_b2_o, base=2)

    print("Subsystem Entropies:")
    print(f"Bubble 1 Center: {entropies[0]:.4f}")
    print(f"Bubble 1 Boundary: {entropies[1]:.4f}")
    print(f"Bubble 1 Outside: {entropies[2]:.4f}")
    print(f"Bubble 2 Center: {entropies[3]:.4f}")
    print(f"Bubble 2 Boundary: {entropies[4]:.4f}")
    print(f"Bubble 2 Outside: {entropies[5]:.4f}")

    print("\nMutual Information:")
    print(f"Bubble 1 Center ↔ Bubble 2 Center: {mi_b1_c_b2_c:.4f}")
    print(f"Bubble 1 Outside ↔ Bubble 2 Outside: {mi_b1_o_b2_o:.4f}")

    return entropies, mi_b1_c_b2_c, mi_b1_o_b2_o

# Main Execution
if __name__ == "__main__":
    t_steps = 10
    qc = enhanced_multi_bubble_circuit(t_steps)

    # Analyze enhanced multi-bubble system
    entropies, mi_b1_c_b2_c, mi_b1_o_b2_o = analyze_enhanced_bubble(qc)

    # Visualize results over time
    entropies_over_time = []
    mi_values_over_time = []

    for t in range(1, t_steps + 1):
        step_qc = enhanced_multi_bubble_circuit(t)
        entropies, mi_c_c, mi_o_o = analyze_enhanced_bubble(step_qc)
        entropies_over_time.append(entropies)
        mi_values_over_time.append((mi_c_c, mi_o_o))

    # Plot Entropies
    plt.figure(figsize=(12, 6))
    for i, label in enumerate(["B1 Center", "B1 Boundary", "B1 Outside", "B2 Center", "B2 Boundary", "B2 Outside"]):
        plt.plot(range(1, t_steps + 1), [e[i] for e in entropies_over_time], label=f"{label}")
    plt.xlabel("Time Steps")
    plt.ylabel("Entropy")
    plt.title("Subsystem Entropies Over Time")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Mutual Information
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, t_steps + 1), [mi[0] for mi in mi_values_over_time], label="B1 Center ↔ B2 Center")
    plt.plot(range(1, t_steps + 1), [mi[1] for mi in mi_values_over_time], label="B1 Outside ↔ B2 Outside")
    plt.xlabel("Time Steps")
    plt.ylabel("Mutual Information")
    plt.title("Mutual Information Over Time")
    plt.legend()
    plt.grid()
    plt.show()
