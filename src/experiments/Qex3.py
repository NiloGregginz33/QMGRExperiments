# This replicates the experiments with the green color charge to see if the information is "destroyed"
from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
from collections import Counter
import numpy as np
import numpy as np
from qiskit.result import Counts


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


def analyze_state_distributions(counts, expected_distribution=None):
    """
    Analyze the state distributions and compare with theoretical expectations.

    Args:
        counts (dict): Measured state counts.
        expected_distribution (dict, optional): Expected probabilities for each state.

    Returns:
        dict: Normalized probabilities and deviations from expectations (if provided).
    """
    # Normalize counts to probabilities
    total_shots = sum(counts.values())
    normalized_probs = {state: count / total_shots for state, count in counts.items()}

    if expected_distribution:
        deviations = {
            state: abs(normalized_probs.get(state, 0) - expected_distribution.get(state, 0))
            for state in set(normalized_probs.keys()).union(expected_distribution.keys())
        }
    else:
        deviations = None

    return {
        "normalized_probs": normalized_probs,
        "deviations": deviations,
    }


def revisit_decoding_logic(raw_result):
    """
    Improve manual decoding of BitArray results.

    Args:
        raw_result (PrimitiveResult): Raw results from Qiskit backend.

    Returns:
        dict: Decoded counts if successful.
    """
    try:
        # Assuming the data field contains BitArray structure
        data_bin = raw_result[0].data.meas
        counts = {}
        for state, freq in zip(data_bin.values, np.bincount(data_bin.meas.flatten())):
            counts[state] = freq
        return counts
    except Exception as e:
        print(f"Decoding failed: {e}")
        return None


def interpret_anomalies(counts, threshold=0.05):
    """
    Interpret anomalies in the state distribution.

    Args:
        counts (dict): Measured state counts.
        threshold (float): Deviation threshold to flag anomalies.

    Returns:
        list: States with anomalies exceeding the threshold.
    """
    # Normalize counts
    total_shots = sum(counts.values())
    normalized_probs = {state: count / total_shots for state, count in counts.items()}

    # Identify anomalies
    anomalies = [
        state for state, prob in normalized_probs.items() if prob < threshold
    ]

    return anomalies

# Example Usage
counts_example = {
    '111': 2444,
    '001': 791,
    '101': 236,
    '000': 2519,
    '110': 945,
    '100': 581,
    '010': 246,
    '011': 430,
}
expected_distribution_example = {
    '000': 0.125,
    '001': 0.125,
    '010': 0.125,
    '011': 0.125,
    '100': 0.125,
    '101': 0.125,
    '110': 0.125,
    '111': 0.125,
}

# Analyze state distributions
analysis = analyze_state_distributions(counts_example, expected_distribution_example)
print("Normalized Probabilities:", analysis["normalized_probs"])
print("Deviations:", analysis["deviations"])

# Interpret anomalies
anomalies = interpret_anomalies(counts_example)
print("Anomalous States:", anomalies)


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


# Step 2: Create the Green Color Charge Circuit with Quantum Gravity Test
def create_green_charge_circuit_with_gravity():
    qc = QuantumCircuit(3)  # Black hole qubit + radiation qubit + gravity probe qubit
    qc.h(0)  # Superposition on Black Hole
    qc.cx(0, 1)  # Entangle Black Hole and Radiation
    qc.ry(1.047, 0)  # Apply a Y-rotation gate to simulate green charge
    
    # Introduce a quantum gravity test by entangling with a probe qubit
    qc.cx(1, 2)  # Entangle radiation with gravity probe
    qc.ry(np.pi / 4, 2)  # Rotate gravity probe to simulate quantum gravity effect
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

# Step 3: Transpile and Run the Circuit
qc = create_green_charge_circuit_with_gravity()
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
        bit_array = data_bin.meas # Access `BitArray`

        print(bit_array)

        counts = extract_counts_from_bitarray(bit_array)

        if counts:
            print("Measurement Results (Counts):")
        for bitstring, count in counts.items():
            print(f"{bitstring}: {count}")
        else:
            print("Failed to decode the results.")

    except Exception as e:
        print(f"An error occurred: {e}")

##        reshaped_data = np.reshape(bit_array, (num_shots, num_bits))  # Reshape based on shots and qubits
##        print(reshaped_data)
##
##
##        # Access and decode measurement results
##
##        decoded_results = [bit_array[i] for i in range(bit_array.size)]
##        print(decoded_results)
##
##        
##        # Use the function to extract counts
##        counts = extract_counts_from_bitarray(bit_array)
##
##        
##        print("Measurement Results (Counts):")
##        for bitstring, count in counts.items():
##            print(f"{bitstring}: {count}")
##
##
##        if hasattr(bit_array, "binary_probabilities"):
##
##            probabilities = bit_array.binary_probabilities()
##            print("Binary Probabilities:", probabilities)
##
##        if hasattr(bit_array, "to_dict"):
##            data_dict = bit_array.to_dict()
##            print("Data as Dictionary:", data_dict)
##
##
##        # Debug: Inspect the `BitArray` structure
##        print("BitArray Attributes:", dir(bit_array))
##        print("BitArray Content:", bit_array)
##
##        if hasattr(bit_array, "to_list"):
##            data_list = bit_array.to_list()
##            print("Data as List:", data_list)
##
##
##        # Attempt to decode or convert the `BitArray`
##        # Replace with actual method based on debug output
##        if hasattr(bit_array, "to_counts"):
##            counts = bit_array.to_counts()
##        elif hasattr(bit_array, "to_dict"):
##            counts = bit_array.to_dict()
##        elif hasattr(bit_array, "get_counts"):
##            counts = bit_array.get_counts()
##
##        # Display the counts
##        print("Measurement Results (Counts):")
##        for bitstring, count in counts.items():
##            print(f"{bitstring}: {count}")
##
##        print("BitArray Internal Variables:", bit_array.__dict__)
##
##
##    except Exception as e:
##        print(f"Error extracting counts from result: {e}")
##
##
##counts = {}
##for outcome in raw_data:
##    outcome_str = ''.join(map(str, outcome))
##    counts[outcome_str] = counts.get(outcome_str, 0) + 1
##    
##print(counts)
##
##
