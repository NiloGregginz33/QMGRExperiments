from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
from collections import Counter
from qiskit.quantum_info import Statevector, partial_trace, entropy
import numpy as np

# Extract counts from BitArray
def extract_counts_from_bitarray(bit_array):
    try:
        # Attempt to use `get_counts` or related methods
        if hasattr(bit_array, "get_counts"):
            counts = bit_array.get_counts()
            print("Counts (get_counts):", counts)
            return counts

        if hasattr(bit_array, "get_int_counts"):
            int_counts = bit_array.get_int_counts()
            print("Integer Counts (get_int_counts):", int_counts)
            return int_counts

        if hasattr(bit_array, "get_bitstrings"):
            bitstrings = bit_array.get_bitstrings()
            counts = Counter(bitstrings)
            print("Bitstrings (Counter):", counts)
            return counts

        # Manual decoding if above methods are unavailable
        print("No direct methods worked; attempting manual decoding.")
        raw_data = str(bit_array)
        counts = Counter(raw_data.split())
        return counts

    except Exception as e:
        print(f"Error processing BitArray: {e}")
        return {}

# Initialize IBM Quantum service
service = QiskitRuntimeService(channel="ibm_quantum")

# Select a backend
def get_best_backend(service, min_qubits=2, max_queue=10):
    backends = service.backends()
    suitable_backends = [
        b for b in backends if b.configuration().n_qubits >= min_qubits and b.status().pending_jobs <= max_queue
    ]
    if not suitable_backends:
        print("No suitable backends found. Using default: ibmq_qasm_simulator")
        return service.backend("ibmq_qasm_simulator")
    
    best_backend = sorted(suitable_backends, key=lambda b: b.status().pending_jobs)[0]
    print(f"Best backend chosen: {best_backend.name}")
    return best_backend

try:
    best_backend = get_best_backend(service)
except Exception as e:
    print(f"Error fetching backend: {e}")
    best_backend = service.backend("ibmq_qasm_simulator")

# Create Experiment 1 circuit
def create_ex1_circuit():
    qc = QuantumCircuit(2)  # Black hole (q0) and radiation (q1)
    qc.h(0)  # Superposition on the black hole qubit
    qc.cx(0, 1)  # Entangle black hole and radiation qubits
    qc.measure_all()  # Measure both qubits
    return qc

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

# Function to calculate Shannon entropy
def calculate_shannon_entropy(counts, num_shots):
    probabilities = {key: value / num_shots for key, value in counts.items()}
    entropy = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)
    return entropy

# Function to calculate von Neumann entropy
def calculate_von_neumann_entropy(qc, num_radiation_qubits):
    state = Statevector.from_instruction(qc)  # Get statevector
    total_entropy = entropy(state)  # Full system entropy

    # Calculate entanglement entropy for subsystems
    black_hole_entropy = entropy(partial_trace(state, range(1, num_radiation_qubits + 1)))  # Trace out radiation
    radiation_entropy = entropy(partial_trace(state, [0]))  # Trace out black hole

    return total_entropy, black_hole_entropy, radiation_entropy

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

    
    # Add classical registers for measurement
    classical_register = ClassicalRegister(num_radiation_qubits + 1)
    qc.add_register(classical_register)
    qc.measure(range(num_radiation_qubits + 1), range(num_radiation_qubits + 1))

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

    # Add classical registers for measurement
    classical_register = ClassicalRegister(num_radiation_qubits + 1)
    qc.add_register(classical_register)
    qc.measure(range(num_radiation_qubits + 1), range(num_radiation_qubits + 1))


    return qc

# Function to calculate von Neumann entropy for the unmeasured circuit
def calculate_von_neumann_entropy_unmeasured(qc, num_radiation_qubits):
    try:
        # Get the statevector from the unmeasured circuit
        state = Statevector.from_instruction(qc)
        
        # Compute total system entropy
        total_entropy = entropy(state)

        # Compute entropies of subsystems
        black_hole_entropy = entropy(partial_trace(state, range(1, num_radiation_qubits + 1)))  # Trace out radiation
        radiation_entropy = entropy(partial_trace(state, [0]))  # Trace out black hole

        return total_entropy, black_hole_entropy, radiation_entropy
    except Exception as e:
        print(f"Error calculating von Neumann entropy: {e}")
        return None, None, None

def process_sampler_result(result, shots=8192):
    """
    Processes the result from the Sampler and converts probabilities into counts.

    Args:
        result: The result object from the Sampler.
        shots: The number of shots used in the experiment (default: 8192).

    Returns:
        A dictionary mapping measurement outcomes (bitstrings) to their counts.
    """
    try:
        # Debug: Print the result object to inspect its structure
        print("Raw Result Object:", result)
        # Debug: Check if result.values exists and is iterable
        if hasattr(result, "values") and isinstance(result.values, list):
            probabilities = result.values[0]  # Extract the probabilities for the first circuit
            print("Extracted Probabilities:", probabilities)
        else:
            raise AttributeError("Result object does not have 'values' or it's not a list.")
        
        # probabilities = result[0].data
        # print(f"Probabilities: ", probabilities)
        # Access the probabilities for the first circuit
            # Extract and process measurement data
        try:
            raw_data = result[0].data  # Updated for structured outputs
            counts = print(f"{key: int(value * 8192) for key, value in raw_data.items()}")  # Convert to counts
        except Exception as e:
            print(f"Error processing sampler result: {e}")
            counts = {}
            
        num_qubits = qc.num_clbits  # Infer number of qubits
        counts = {
            f"{i:0{num_qubits}b}": int(prob * 8192) for i, prob in enumerate(raw_data)
                    }
        return counts
    except Exception as e:
        print(f"Error processing sampler result: {e}")
        return {}

# Parameters for the circuit
num_injections = 5  # 5 charge injections
num_radiation_qubits = 3  # 3 radiation qubits entangled with the black hole
gap_cycles = 50  # 50 idle cycles between injections

# Create the quantum circuit
qc = create_circuit_with_time_gaps(num_injections, num_radiation_qubits, gap_cycles)


# Transpile the circuit for the backend
transpiled_qc = transpile(qc, backend=best_backend)

# Execute the circuit using the Session and Sampler
with Session(backend=best_backend) as session:  # Attach backend to session
    sampler = Sampler()
    job = sampler.run([transpiled_qc], shots=8192)
    result = job.result()

    # Debug: Inspect raw result
    print("Raw Result Object:", result)

    # Extract counts from the nested structure
    try:
        # Navigate to the `BitArray`
        pub_result = result._pub_results[0]  # Access the first `SamplerPubResult`
        data_bin = pub_result.data  # Access `DataBin`
        bit_array = data_bin.c0  # Access `BitArray`

        
        # Use the function to extract counts
        counts = extract_counts_from_bitarray(bit_array)

        
        print("Measurement Results (Counts):")
        for bitstring, count in counts.items():
            print(f"{bitstring}: {count}")


        if hasattr(bit_array, "binary_probabilities"):

            probabilities = bit_array.binary_probabilities()
            print("Binary Probabilities:", probabilities)

        if hasattr(bit_array, "to_dict"):
            data_dict = bit_array.to_dict()
            print("Data as Dictionary:", data_dict)


        # Debug: Inspect the `BitArray` structure
        print("BitArray Attributes:", dir(bit_array))
        print("BitArray Content:", bit_array)

        if hasattr(bit_array, "to_list"):
            data_list = bit_array.to_list()
            print("Data as List:", data_list)


        # Attempt to decode or convert the `BitArray`
        # Replace with actual method based on debug output
        if hasattr(bit_array, "to_counts"):
            counts = bit_array.to_counts()
        elif hasattr(bit_array, "to_dict"):
            counts = bit_array.to_dict()
        elif hasattr(bit_array, "get_counts"):
            counts = bit_array.get_counts()

        # Display the counts
        print("Measurement Results (Counts):")
        for bitstring, count in counts.items():
            print(f"{bitstring}: {count}")

        print("BitArray Internal Variables:", bit_array.__dict__)
        # Total number of shots
        num_shots = sum(counts.values())

        # Shannon Entropy
        shannon_entropy = calculate_shannon_entropy(counts, num_shots)
        print(f"Shannon Entropy (Measurement Distribution): {shannon_entropy:.4f}")
        # Von Neumann Entropy for unmeasured circuit
        unmeasured_qc = qc.copy()  # Copy the circuit before adding measurements
        unmeasured_qc.data = [instr for instr in unmeasured_qc.data if instr[0].name != "measure"]  # Remove measurement ops

        total_entropy, black_hole_entropy, radiation_entropy = calculate_von_neumann_entropy_unmeasured(unmeasured_qc, num_radiation_qubits)
        if total_entropy is not None:
            print(f"Total Entropy (Full System): {total_entropy:.4f}")
            print(f"Entropy (Black Hole Subsystem): {black_hole_entropy:.4f}")
            print(f"Entropy (Radiation Subsystem): {radiation_entropy:.4f}")
        else:
            print("Von Neumann entropy calculation failed.")

    except Exception as e:
        print(f"Error processing result for entropy analysis: {e}")
