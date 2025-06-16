#This is the same as the last experiment (ex3), which was interesting because of the more complex wavestate patterns, not
#only did it show a diversity in outcomes, but also that shifting balances in the outcomes
#may mean a more complex interplay at work. This version of the experiment is attempted on a real quantum computer because idk
#the less abstractions you need to make the better. I need to be sure of this

from qiskit_aer import Aer
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister
from qiskit.quantum_info import Statevector, partial_trace, entropy
from qiskit.visualization import plot_histogram
import numpy as np
from qiskit import transpile
from qiskit_ibm_runtime import Sampler, Session
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import transpile
from qiskit import QuantumCircuit

# Load IBM Quantum account
service = QiskitRuntimeService()

# Fetch all available backends
backends = service.backends()

# Define your preferred criteria
min_qubits = 2              # Minimum number of qubits
max_queue = 10
min_t1_time = 10e-6          # Relaxed T1 time (e.g., 10 microseconds)
min_t2_time = 10e-6          # Relaxed T2 time (e.g., 10 microseconds)

# Function to evaluate backend suitability
def evaluate_backend(backend):
    try:
        props = backend.properties()
        num_qubits = backend.num_qubits
        queue_length = backend.status().pending_jobs

        # Calculate average T1 and T2 times
        t1_times = [qubit_data[0].t1 for qubit_data in props.qubits if 't1' in qubit_data[0].__dict__]
        t2_times = [qubit_data[0].t2 for qubit_data in props.qubits if 't2' in qubit_data[0].__dict__]

        t1_avg = sum(t1_times) / len(t1_times) if t1_times else 0
        t2_avg = sum(t2_times) / len(t2_times) if t2_times else 0

        return (
            num_qubits >= min_qubits and
            queue_length <= max_queue and
            t1_avg >= min_t1_time and
            t2_avg >= min_t2_time
        )
    except Exception as e:
        print(f"Error processing backend {backend.name}: {e}")
        return False

# Filter suitable backends
suitable_backends = [
    backend for backend in backends if evaluate_backend(backend)
]

# Auto-choose the best backend
if suitable_backends:
    # Sort by queue length (ascending) and qubit count (descending)
    suitable_backends.sort(
        key=lambda b: (b.status().pending_jobs, -b.num_qubits)
    )
    try:
        best_backend = suitable_backends[0]
    except Exception as e:
        print(f"Error fetching backend: {e}")
        best_backend = service.backend("ibmq_qasm_simulator")
    print(f"Best Backend Chosen: {best_backend.name}")
    print(f"Using backend: {best_backend.name}")
    print(f"  Qubits: {best_backend.num_qubits}")
    print(f"  Pending Jobs: {best_backend.status().pending_jobs}")
else:
    print("No suitable backends found based on your criteria.")
    best_backend = service.backend("ibmq_qasm_simulator")


print("Transpiled circuit ready for execution.")

# Function to create the quantum circuit (no classical bits for Statevector)
def create_circuit(apply_positive_charge, apply_negative_charge):
    qc = QuantumCircuit(2)  # Create a new circuit
    qc.h(0)  # Superposition on Black Hole qubit
    qc.cx(0, 1)  # Entangle Black Hole and Radiation qubits

    # Simulate Charge Pulses
    if apply_positive_charge:
        qc.x(0)  # Positive charge (Pauli-X gate on Black Hole)
    if apply_negative_charge:
        qc.z(0)  # Negative charge (Pauli-Z gate on Black Hole)
    
    return qc

# Analyze phase shifts in the quantum state
def analyze_phases(qc):
    state = Statevector.from_instruction(qc)  # Get the statevector
    phases = np.angle(state.data)  # Extract the phases
    return phases

# Simulate black hole evaporation
def simulate_evaporation(charge_state, num_radiation_qubits):
    qc = QuantumCircuit(num_radiation_qubits + 1)  # One black hole + radiation qubits
    if charge_state == "positive":
        qc.x(0)
    elif charge_state == "negative":
        qc.z(0)

    # Entangle black hole with radiation qubits sequentially
    for i in range(1, num_radiation_qubits + 1):
        qc.h(0)  # Superposition on Black Hole
        qc.cx(0, i)  # Entangle with radiation qubit i

    return qc

# Function to add measurements
def add_measurements(qc, measure_qubits):
    measured_circuit = qc.copy()  # Create a fresh copy
    measured_circuit.add_register(ClassicalRegister(len(measure_qubits)))  # Add classical register
    measured_circuit.measure(measure_qubits, range(len(measure_qubits)))  # Measure specified qubits
    return measured_circuit


# Function to create the circuit with a specific charge state
def create_circuit_with_charge(charge_state):
    qc = QuantumCircuit(2)  # Create a new 2-qubit circuit
    if charge_state == "positive":
        qc.x(0)  # Set the black hole qubit to |1⟩ (positive charge)
    elif charge_state == "negative":
        qc.z(0)  # Introduce a phase flip (negative charge)
    elif charge_state == "neutral":
        pass  # Default to |0⟩ (neutral charge)
    
    # Step 2: Entangle the black hole and radiation qubits
    qc.h(0)  # Superposition on Black Hole qubit
    qc.cx(0, 1)  # Entangle Black Hole (Register A) and Radiation (Register B)
    return qc

# Function to create the circuit with charge injections
def create_circuit_with_alternating_charges(num_injections, num_radiation_qubits):
    qc = QuantumCircuit(num_radiation_qubits + 1)  # Black hole + radiation qubits

    # Alternate between injecting positive (X) and negative (Z) charge
    for i in range(num_injections):
        if i % 2 == 0:
            qc.x(0)  # Positive charge injection
        else:
            qc.z(0)  # Negative charge injection

        # Entangle black hole with radiation qubits sequentially
        for j in range(1, num_radiation_qubits + 1):
            qc.h(0)  # Superposition on Black Hole
            qc.cx(0, j)  # Entangle with radiation qubit j

    return qc

# Function to create a circuit with prolonged charge injection
def create_circuit_with_prolonged_charges(num_iterations, cycle_length, num_radiation_qubits):
    qc = QuantumCircuit(num_radiation_qubits + 1)  # Black hole + radiation qubits

    for iteration in range(num_iterations):
        # Determine current charge based on cycle
        if (iteration // cycle_length) % 2 == 0:
            qc.x(0)  # Positive charge injection
        else:
            qc.z(0)  # Negative charge injection

        # Entangle black hole with radiation qubits sequentially
        for j in range(1, num_radiation_qubits + 1):
            qc.h(0)  # Superposition on Black Hole
            qc.cx(0, j)  # Entangle with radiation qubit j

    return qc

def create_circuit_with_time_gaps(num_injections, num_radiation_qubits, gap_cycles):
    qc = QuantumCircuit(num_radiation_qubits + 1)  # Black hole + radiation qubits

    for i in range(num_injections):
        # Inject charge
        if i % 2 == 0:
            qc.x(0)  # Positive charge injection
        else:
            qc.z(0)  # Negative charge injection

        # Entangle black hole with radiation qubits sequentially
        for j in range(1, num_radiation_qubits + 1):
            qc.h(0)  # Superposition on Black Hole
            qc.cx(0, j)  # Entangle with radiation qubit j

        # Add "time gap" as idle cycles
        qc.barrier()
        for _ in range(gap_cycles):
            qc.id(0)  # Idle gate to simulate a time gap

    return qc

# Add measurements to the circuit
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

# Create and transpile the circuit
qc = create_circuit_with_time_gaps(num_injections, num_radiation_qubits, gap_cycles)
qc_with_measurements = add_measurements(qc, range(1, num_radiation_qubits + 1))
transpiled_qc = transpile(qc_with_measurements, backend=best_backend)

# Execute on the real backend using IBM Runtime Sampler
with Session(service=service, backend=backend) as session:
    sampler = Sampler(session=session)
    job = sampler.run(circuits=transpiled_qc, shots=8192)
    result = job.result()

# Analyze Results
counts = result.quasi_dists[0].binary_probabilities()  # Get quasi-probabilities
counts_int = {k: int(v * 8192) for k, v in counts.items()}  # Convert to counts assuming shots = 8192

print("Measurement Results (Counts):", counts_int)
