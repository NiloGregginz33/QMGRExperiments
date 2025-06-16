from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
from qiskit.quantum_info import Statevector, entropy

# Initialize IBM Runtime Service
service = QiskitRuntimeService()

def get_backend(service, min_qubits=2):
    print("Fetching available backends...")
    backends = service.backends()
    for backend in backends:
        print(f"Backend: {backend.name} - Qubits: {backend.configuration().n_qubits}")
    suitable_backends = [
        b for b in backends if b.configuration().n_qubits >= min_qubits
    ]
    if suitable_backends:
        print(f"Selected Backend: {suitable_backends[0].name}")
        return suitable_backends[0]
    else:
        print("No suitable backends found.")
        return None


backend = get_backend(service)

# Circuit to simulate Kerr Black Hole with Rotation
def create_rotation_circuit(num_radiation_qubits, rotation_angle):
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

# Run simulation and collect results
def run_simulation(rotation_angle, num_radiation_qubits=4, shots=8192):
    print(f"Running simulation with rotation angle: {rotation_angle}...")
    qc = create_rotation_circuit(num_radiation_qubits, rotation_angle)
    transpiled_qc = transpile(qc, backend=backend)
    print("Circuit transpiled successfully!")

    # Submit the circuit to the backend
    try:
        with Session(backend=backend) as session:
            print(f"Executing circuit on backend {backend.name()}...")
            sampler = Sampler()


            job = sampler.run([transpiled_qc], shots=shots)
            result = job.result()
            
            # Debugging output
            print("Raw Result Object:", result)
            
            # Extract counts and entropy
            counts = result.get_counts(0)
            entropy_value = entropy(Statevector.from_counts(counts))
            print(f"Counts: {counts}")
            print(f"Subsystem Entropy: {entropy_value}")
            return counts, entropy_value
    except Exception as e:
        print(f"Error during simulation: {e}")
        return None, None

# Parameters
# Use Aer simulator
simulator = Aer.get_backend('aer_simulator')
shots = 8192
num_injections = 10  # Total charge injections
num_radiation_qubits = 4  # Number of radiation qubits
gap_cycles = 100  # Idle cycles between injections

# Create and run the circuit
qc = create_circuit_with_rotation(num_injections, num_radiation_qubits, gap_cycles)
qc_with_measurements = add_measurements(qc, range(1, num_radiation_qubits + 1))

# Run the simulation
job = simulator.run(qc_with_measurements, shots=8192)
result = job.result()
counts = result.get_counts()

# Analyze Results
print("Measurement Results:")
print(counts)
