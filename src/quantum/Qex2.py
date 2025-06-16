# This replicates the experiments with the green color charge to see if the information is "destroyed"
from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
from collections import Counter

# Step 1: Initialize Quantum Service
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

# Step 2: Create the Green Color Charge Circuit
def create_green_charge_circuit():
    qc = QuantumCircuit(2)  # Example: Black hole qubit + radiation qubit
    qc.h(0)  # Superposition on Black Hole
    qc.cx(0, 1)  # Entangle Black Hole and Radiation
    qc.ry(1.047, 0)  # Apply a Y-rotation gate to simulate green charge
    return qc

# Function to add measurements
def add_measurements(qc):
    """
    Adds classical registers and measurement instructions to the circuit.
    """
    # Determine the number of qubits in the circuit
    num_qubits = qc.num_qubits

    # Create a classical register
    classical_reg = ClassicalRegister(num_qubits, "meas")

    # Add the classical register to the circuit
    qc.add_register(classical_reg)

    # Add measurements to all qubits
    qc.measure(range(num_qubits), range(num_qubits))

    return qc

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


# Step 3: Transpile and Run the Circuit
qc = create_green_charge_circuit()
add_measurements(qc)
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
        bit_array = data_bin # Access `BitArray`

        
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


    except Exception as e:
        print(f"Error extracting counts from result: {e}")

