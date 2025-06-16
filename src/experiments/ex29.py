from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import RZGate
from qiskit.visualization import plot_histogram
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.circuit import ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService

# Initialize Qiskit Runtime Service
try:
    service = QiskitRuntimeService()
    print("Qiskit Runtime Service initialized successfully!")
except Exception as e:
    print(f"Error initializing Qiskit Runtime Service: {e}")
    exit(1)
    
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

# Function to create a circuit with rotating black hole simulation
def create_rotating_black_hole_circuit(num_radiation_qubits, rotation_angle):
    qc = QuantumCircuit(num_radiation_qubits + 1, num_radiation_qubits)

    # Black hole qubit
    qc.h(0)
    qc.append(RZGate(rotation_angle), [0])  # Simulating black hole rotation

    # Radiation qubits entangled with the black hole
    for i in range(1, num_radiation_qubits + 1):
        qc.cx(0, i)
        qc.append(RZGate(rotation_angle / (i + 1)), [i])  # Modulate with decreasing impact

    # Add measurement
    qc.measure(range(num_radiation_qubits), range(num_radiation_qubits))
    return qc

# Function to execute the simulation and retrieve results
def execute_simulation(qc, shots=8192):
    simulator = Aer.get_backend('qasm_simulator')
    transpiled_qc = transpile(qc, simulator)
    job = execute(transpiled_qc, backend=simulator, shots=shots)
    result = job.result()
    return result

# Function to analyze results
def analyze_results(result, num_radiation_qubits):
    counts = result.get_counts()
    print("Measurement Counts:", counts)

    # Normalize probabilities
    total_shots = sum(counts.values())
    probabilities = {state: count / total_shots for state, count in counts.items()}
    print("Normalized Probabilities:", probabilities)

    # Symmetry detection (example logic)
    detected_symmetries = []
    for state in probabilities.keys():
        if state[::-1] == state:  # Palindromic symmetry as an example
            detected_symmetries.append(state)
    print("Detected Symmetries:", detected_symmetries)

    # Plot the results for better visualization
    plot_histogram(counts).show()

    return probabilities, detected_symmetries


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
   
### Backend and simulation setup
##try:
##    with Session(backend=best_backend as session:
##        sampler = Sampler(session=session)
##        
##        for rotation_angle in [0, 0.1, 0.5, 1.0, 2.0]:
##            print(f"Running simulation with rotation angle: {rotation_angle}...")
##            circuit = create_circuit_with_rotation(4, rotation_angle)
##            transpiled_qc = transpile(circuit, backend=best_backend)
##
##            try:
##                job = sampler.run([transpiled_qc], shots=8192)
##                result = job.result()
##                print("Job completed successfully!")
##                print("Measurement results:", result)
##            except Exception as sim_error:
##                print(f"Simulation failed for rotation angle {rotation_angle} with error: {sim_error}")
##                
##except Exception as backend_error:
##    print(f"Error with backend or session: {backend_error}")
##
    
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

# Function to add measurements
def add_measurements(qc, measure_qubits):
    measured_circuit = qc.copy()  # Create a fresh copy
    measured_circuit.add_register(ClassicalRegister(len(measure_qubits)))  # Add classical register
    measured_circuit.measure(measure_qubits, range(len(measure_qubits)))  # Measure specified qubits
    return measured_circuit

# Parameters
# Use Aer simulator
simulator = Aer.get_backend('aer_simulator')
shots = 8192
num_injections = 10  # Total charge injections
num_radiation_qubits = 4  # Number of radiation qubits
gap_cycles = 100  # Idle cycles between injections

# Create and run the circuit
qc = create_rotating_black_hole_circuit(num_injections, num_radiation_qubits)
qc_with_measurements = add_measurements(qc, range(1, num_radiation_qubits + 1))

# Run the simulation
job = simulator.run(qc_with_measurements, shots=8192)
result = job.result()
counts = result.get_counts()

# Analyze Results
print("Measurement Results:")
print(counts)

