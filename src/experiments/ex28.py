from qiskit import QuantumCircuit
from qiskit_ibm_runtime import Sampler, Session

# Debug function for constructing circuits with rotation
def create_circuit_with_rotation(num_radiation_qubits, rotation_angle):
    try:
        qc = QuantumCircuit(num_radiation_qubits + 1)  # Black hole + radiation qubits
        qc.h(0)  # Initialize black hole qubit in superposition
        
        for i in range(num_radiation_qubits):
            qc.p(rotation_angle, 0)  # Apply phase rotation
            qc.cx(0, i + 1)  # Entangle with radiation qubits

        qc.measure_all()  # Add measurements
        print("Circuit created successfully!")
        print(qc)
        return qc
    except Exception as e:
        print(f"Error creating circuit: {e}")
        return None

# Backend and simulation setup
try:
    with Session(backend=best_backend) as session:
        sampler = Sampler(session=session)
        
        for rotation_angle in [0, 0.1, 0.5, 1.0, 2.0]:
            print(f"Running simulation with rotation angle: {rotation_angle}...")
            circuit = create_circuit_with_rotation(4, rotation_angle)
            transpiled_qc = transpile(circuit, backend=best_backend)

            try:
                job = sampler.run(circuits=[transpiled_qc], shots=8192)
                result = job.result()
                print("Job completed successfully!")
                print("Measurement results:", result)
            except Exception as sim_error:
                print(f"Simulation failed for rotation angle {rotation_angle} with error: {sim_error}")
except Exception as backend_error:
    print(f"Error with backend or session: {backend_error}")
