from qiskit import execute
from qiskit_aer import Aer

# Debugging: Print the created quantum circuit
qc = create_hybrid_circuit("red", "green", 0, 0, with_measurements=True)
print("Generated Circuit:")
print(qc)

# Execute the circuit
simulator = Aer.get_backend('aer_simulator')
job = simulator.run(qc, shots=shots)

# Fetch and debug results
try:
    results = job.result().get_counts()
    print(f"Simulation results: {results}")
except Exception as e:
    print(f"Error fetching results: {e}")
