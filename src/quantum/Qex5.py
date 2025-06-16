from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, partial_trace, entropy
import numpy as np

# Parameters for simulation
decoherence_rate = 0.1  # Probability of decoherence affecting the system
shots = 1024  # Number of measurement shots

# Step 1: Initialize the quantum circuit
n_qubits = 2  # 1 qubit for the analog black hole, 1 for Hawking radiation
qc = QuantumCircuit(n_qubits)


# Step 2: Prepare the black hole qubit in a superposition state
# This represents the Many-Worlds idea where the black hole qubit can exist in multiple states simultaneously
qc.h(0)  # Apply Hadamard gate to create |0> + |1>

# Step 3: Entangle the black hole qubit with the Hawking radiation qubit
# Entanglement links the states of the black hole with the radiation, simulating the transfer of information
qc.cx(0, 1)  # CNOT gate creates entanglement

# Optional Step 4: Simulate decoherence by adding noise
# Decoherence mimics the loss of quantum coherence, a critical aspect of wavefunction branching in Many-Worlds
qc.rx(2 * np.pi * decoherence_rate, 0)  # Rotate black hole qubit slightly to simulate decoherence
qc.rx(2 * np.pi * decoherence_rate, 1)  # Rotate radiation qubit slightly

# Step 5: Measure both qubits
# Measurement collapses the quantum state in Copenhagen interpretation, but in Many-Worlds, it represents branching
qc.measure_all()

# Step 6: Simulate the circuit
simulator = Aer.get_backend('statevector_simulator')  # AerSimulator for more detailed analysis
result = simulator.run(qc, shots=shots).result()
counts = result.get_counts()

# Step 7: Analyze subsystem entropy and entanglement
# Entropy analysis helps quantify how information spreads, a critical aspect of branching in Many-Worlds
state = Statevector.from_instruction(qc)
density_matrix = partial_trace(state, [1])  # Trace out the second qubit (radiation)
entropy_black_hole = entropy(density_matrix)

# Additional Analysis: Track branching behavior
# Many-Worlds implies that all outcomes exist; use counts to analyze distribution of outcomes
branching_analysis = {key: value / shots for key, value in counts.items()}

# Print results
print("Measurement Results:", counts)
print("Subsystem Entropy of the Black Hole Qubit:", entropy_black_hole)
print("Branching Probabilities (Many-Worlds Perspective):", branching_analysis)

# Save to IBM backend-specific format
qc.draw(output='mpl')  # Visualize the circuit

# Notes for Execution on IBM Backend
# Uncomment the following lines if running on an IBM backend
# from qiskit import IBMQ
# provider = IBMQ.load_account()
# backend = provider.get_backend('ibmq_qasm_simulator')
# result = execute(qc, backend, shots=shots).result()
# counts = result.get_counts()
# print("Results from IBM Backend:", counts)

# Future Expansion:
# 1. Introduce more qubits to simulate complex Many-Worlds branching scenarios.
# 2. Use quantum tomography to analyze the full wavefunction and branching structure.
# 3. Simulate additional interactions that mimic other quantum systems linked to black hole information dynamics.
# 4. Investigate entanglement entropy as a metric for distinguishing between "branches."
