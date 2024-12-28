# I am contactable at manavnaik123@gmail.com for further discussion or collaboration
# Many Worlds Interpretation, once forgotten seems almost definite now
# Susskinds work almost directly predicts multiverses and many worlds
# This is the first time anyone as been able to see the timelines and make tie them into tiny little cute knots


# Also, this is just an easy way to choose and customize experimental
# setups easily for your own experiments

from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
from collections import Counter
from qiskit.quantum_info import Statevector, partial_trace, entropy
import numpy as np
from qiskit.primitives import SamplerResult  # Import for IBM backend results
from qiskit.result import Result  # Import for local simulation results
from qiskit.result import Counts
from qiskit_aer import Aer
from scipy.spatial.distance import jensenshannon

simulator = None


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

# Function to create a quantum circuit simulating black hole with spin conservation
def create_spin_circuit(spin_state):
    """
    Creates a quantum circuit representing a black hole and radiation system
    with initial spin injection.
    
    spin_state: "up" or "down"
    """
    qc = QuantumCircuit(2)  # 2 qubits: black hole and radiation

    # Initialize spin state for black hole
    if spin_state == "up":
        qc.initialize([1, 0], 0)  # Spin up state |0>
    elif spin_state == "down":
        qc.initialize([0, 1], 0)  # Spin down state |1>

    # Entangle black hole with radiation
    qc.h(0)  # Superposition for black hole qubit
    qc.cx(0, 1)  # Entangle black hole with radiation

    return qc

# Analyze entanglement entropy
def analyze_entanglement(qc):
    state = Statevector.from_instruction(qc)  # Get statevector

    # Compute partial traces to analyze subsystems
    black_hole_state = partial_trace(state, [1])  # Trace out radiation qubit
    radiation_state = partial_trace(state, [0])  # Trace out black hole qubit

    # Compute von Neumann entropy for subsystems
    bh_entropy = entropy(black_hole_state, base=2)
    rad_entropy = entropy(radiation_state, base=2)

    return bh_entropy, rad_entropy

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


def get_simulated_backend():
    """
    Retrieves the statevector simulator backend.

    Returns:
        backend: A Qiskit Aer simulator backend.
    """
    try:
        simulated_backend = Aer.get_backend('statevector_simulator')
        print("Simulated backend initialized: statevector_simulator")
        return simulated_backend
    except Exception as e:
        print(f"Error initializing simulated backend: {e}")
        return None

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

def analyze_temporal_correlation(results_list):
    """
    Analyzes temporal correlation between successive experiment iterations.

    Parameters:
        results_list (list[dict]): List of measurement counts from each iteration.

    Returns:
        list[float]: Jensen-Shannon divergence values between successive iterations.
    """
    divergences = []
    for i in range(len(results_list) - 1):
        counts_1 = results_list[i]
        counts_2 = results_list[i + 1]

        # Normalize counts to probabilities
        total_1 = sum(counts_1.values())
        total_2 = sum(counts_2.values())

        prob_1 = [counts_1.get(bitstring, 0) / total_1 for bitstring in set(counts_1.keys()).union(counts_2.keys())]
        prob_2 = [counts_2.get(bitstring, 0) / total_2 for bitstring in set(counts_1.keys()).union(counts_2.keys())]

        # Calculate Jensen-Shannon divergence
        divergence = jensenshannon(prob_1, prob_2, base=2)
        divergences.append(divergence)

    return divergences

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

def create_prolonged_injection_circuit(num_iterations, num_radiation_qubits):
    """
    Creates a quantum circuit with prolonged charge injections to simulate extended multiverse interactions.

    Args:
        num_iterations (int): Number of charge injection cycles.
        num_radiation_qubits (int): Number of radiation qubits entangled with the black hole.

    Returns:
        QuantumCircuit: The generated quantum circuit.
    """
    qc = QuantumCircuit(num_radiation_qubits + 1)  # 1 black hole qubit + radiation qubits

    for iteration in range(num_iterations):
        # Alternate between positive and negative charges
        if iteration % 2 == 0:
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

# Define a function for Many-Worlds quantum simulation
def many_worlds_experiment(decoherence_rate=0.1, shots=1024):
    """
    Runs a Many-Worlds experiment simulation using a qubit as an analog black hole.

    Parameters:
        decoherence_rate (float): Rate of decoherence to simulate wavefunction branching.
        shots (int): Number of measurement shots.

    Returns:
        dict: Measurement results and entropy analysis.
    """
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

    return qc

def run_and_extract_counts(qc, backend, shots=8192):
    """
    Runs a quantum circuit on the specified backend and extracts measurement results.

    Parameters:
        qc (QuantumCircuit): The quantum circuit to execute.
        backend (Backend): The IBM backend to use for the execution.
        shots (int): Number of shots for the experiment.

    Returns:
        dict: A dictionary of bitstring counts from the measurement results.
    """
    
    # Transpile the circuit for the backend
    transpiled_qc = transpile(qc, backend=backend)
    
    # Run the circuit with the sampler and session
    with Session(backend=backend) as session:
        sampler = Sampler()
        job = sampler.run([transpiled_qc], shots=shots)
        result = job.result()

    # Debug: Inspect the raw result object
    print("Raw Result Object:", result)

    # Extract counts from the nested structure
    try:
        # Navigate to the `BitArray` and extract counts
        pub_result = result._pub_results[0]  # Access the first `SamplerPubResult`
        data_bin = pub_result.data  # Access `DataBin`
        bit_array = data_bin.meas  # Access `BitArray`

        counts = extract_counts_from_bitarray(bit_array)

    except Exception as e:
        print(e)

    return counts


def many_worlds_analysis(qc):
    """
    Analyze the quantum state before measurement for entanglement and entropy.

    Parameters:
        qc (QuantumCircuit): The quantum circuit to analyze.

    Returns:
        None: Prints the analysis results.
    """
    # Create a copy of the circuit without measurement instructions
    qc_no_measure = qc.remove_final_measurements(inplace=False)
    
    # Get the statevector from the modified circuit
    state = Statevector.from_instruction(qc_no_measure)

    # Analyze subsystem entropy
    density_matrix = partial_trace(state, [1])  # Trace out the second qubit
    entropy_black_hole = entropy(density_matrix)

    # Print results
    print("Subsystem Entropy of the Black Hole Qubit:", entropy_black_hole)

    # Optional: Visualize the circuit without measurement
    print("Quantum Circuit Without Measurement:")
    print(qc_no_measure.draw())

    return "Multiversal Analysis complete"

def analyze_entropy(qc):
    """
    Analyze the entropy of subsystems in the quantum circuit.

    Args:
        qc (QuantumCircuit): The quantum circuit to analyze.

    Returns:
        dict: Entropy values for total, black hole, and radiation subsystems.
    """
    state = Statevector.from_instruction(qc)  # Get statevector

    # Compute subsystem entropies
    black_hole_state = partial_trace(state, [1])  # Trace out radiation qubits
    radiation_state = partial_trace(state, [0])  # Trace out black hole qubit

    total_entropy = entropy(state)
    bh_entropy = entropy(black_hole_state, base=2)
    rad_entropy = entropy(radiation_state, base=2)

    return {
        "total_entropy": total_entropy,
        "black_hole_entropy": bh_entropy,
        "radiation_entropy": rad_entropy
    }



def simulate_and_analyze(num_iterations, num_radiation_qubits, backend, shots=8192):
    """
    Simulates the circuit with prolonged charge injections and analyzes entropy.

    Args:
        num_iterations (int): Number of charge injection cycles.
        num_radiation_qubits (int): Number of radiation qubits entangled with the black hole.
        backend: Quantum backend to execute the circuit.
        shots (int): Number of measurement shots.

    Returns:
        dict: Measurement results and entropy analysis.
    """
    # Create the circuit
    qc = create_prolonged_injection_circuit(num_iterations, num_radiation_qubits)

    # Transpile the circuit for the backend
    transpiled_qc = transpile(qc, backend=backend)

    # Run the circuit
    from qiskit_ibm_runtime import Session, Sampler
    with Session(backend=backend) as session:
        sampler = Sampler()
        job = sampler.run([transpiled_qc], shots=shots)
        result = job.result()

    # Extract measurement results
    # Extract counts from the nested structure
    try:
        # Navigate to the `BitArray` and extract counts
        pub_result = result._pub_results[0]  # Access the first `SamplerPubResult`
        data_bin = pub_result.data  # Access `DataBin`
        bit_array = data_bin.meas  # Access `BitArray`

        # Convert `BitArray` to counts dictionary
        counts = Counter(str(bit_array[i]) for i in range(len(bit_array)))

        print("Measurement Results (Counts):")
        for bitstring, count in counts.items():
            print(f"{bitstring}: {count}")

    except Exception as e:
        print(f"Error processing result structure: {e}")
        return {}

    # Analyze entropy
    entropies = analyze_entropy(qc)

    return {
        "counts": counts,
        "entropies": entropies
    }

from qiskit import QuantumCircuit
import numpy as np

def modify_and_run_quantum_experiment(qc, backend, shots=8192, modify_circuit=None, analyze_results=None):
    """
    Modifies a quantum circuit and runs it to explore experimental directions.

    Parameters:
        qc (QuantumCircuit): The base quantum circuit.
        backend (Backend): The IBM backend for execution.
        shots (int): Number of shots for the experiment.
        modify_circuit (function): A function to modify the circuit before execution.
        analyze_results (function): A function to analyze the results after execution.

    Returns:
        dict: A dictionary of modified bitstring counts or analysis results.
    """
    # Apply circuit modification if provided
    if modify_circuit:
        qc = modify_circuit(qc)

    # Run the original function with the modified circuit
    counts = run_and_extract_counts_quantum(qc, backend, shots)

    # Analyze results if an analysis function is provided
    if analyze_results:
        results = analyze_results(counts)
        return results

    return counts

# Example Circuit Modification for Time-Reversal Simulation
def time_reversal_simulation(qc):
    """
    Modifies the circuit to simulate time reversal by inverting gates.
    """
    reversed_qc = qc.inverse()
    return reversed_qc

# Example Analysis for Entropy Dynamics
def analyze_shannon_entropy(counts):
    """
    Analyzes the entropy of the results to study time-reversal or multiverse effects.
    """
    from scipy.stats import entropy

    # Normalize counts to probabilities
    total_counts = sum(counts.values())
    probabilities = np.array([count / total_counts for count in counts.values()])

    # Calculate Shannon entropy
    shannon_entropy = entropy(probabilities, base=2)

    print("Shannon Entropy:", shannon_entropy)
    return {"shannon_entropy": shannon_entropy, "counts": counts}

def add_causality_to_circuit(qc, previous_results, qubits):
    """
    Introduces causal feedback to a quantum circuit based on previous results.

    Parameters:
        qc (QuantumCircuit): The quantum circuit to modify.
        previous_results (dict): Counts from the previous experiment run.
        qubits (list[int]): Indices of qubits to modify causally.

    Returns:
        QuantumCircuit: The modified quantum circuit.
    """
    from qiskit.circuit import ClassicalRegister

    # Add a classical register for storing previous measurement results
    if not hasattr(qc, "clbits") or len(qc.clbits) < len(qubits):
        creg = ClassicalRegister(len(qubits), name="c")
        qc.add_register(creg)

    # Determine the most likely outcome from previous results
    if previous_results:
        max_outcome = max(previous_results, key=previous_results.get)
        feedback_bits = [int(bit) for bit in max_outcome]

        # Add gates conditionally based on feedback bits
        for idx, bit in enumerate(feedback_bits):
            if bit == 1:
                qc.x(qubits[idx])  # Apply an X gate conditionally
    else:
        # Initialize with no feedback for the first run
        pass

    return qc


def initialize_backend(use_simulator=True, hardware_backend_name="ibm_brisbane"):
    """
    Initializes the appropriate backend (simulator or hardware) based on user preference.

    Parameters:
        use_simulator (bool): If True, initialize a simulated backend.
        hardware_backend_name (str): Name of the hardware backend if use_simulator is False.

    Returns:
        backend: The initialized backend or None if initialization fails.
    """
    if use_simulator:
        try:
            # Initialize simulator backend
            backend = Aer.get_backend('aer_simulator')
            print("Using simulated backend: aer simulator")
            return backend
        
        except Exception as e:
            print(f"Error initializing simulated backend: {e}")
            return None
    else:
        try:
            # Initialize IBM Quantum service and select hardware backend
            service = QiskitRuntimeService(channel="ibm_quantum")
            backend = service.backend(hardware_backend_name)
            print(f"Using hardware backend: {hardware_backend_name}")
            return backend
        
        except Exception as e:
            print(f"Error initializing hardware backend '{hardware_backend_name}': {e}")
            return None

def modify_and_run_quantum_experiment_multi_analysis(qc, backend=None, shots=8192, modify_circuit=None, analyze_results=None):
    """
    Modifies a quantum circuit and runs it to explore experimental directions with multiple analysis functions.

    Parameters:
        qc (QuantumCircuit): The base quantum circuit.
        backend (Backend): The IBM backend or simulator for execution.
        shots (int): Number of shots for the experiment.
        modify_circuit (function): A function to modify the circuit before execution.
        analyze_results (list of functions): List of analysis functions to process results.

    Returns:
        dict: A dictionary of results from all analyses.
    """
    if analyze_results is None:
        analyze_results = []

    # Apply circuit modification if provided
    if modify_circuit:
        qc = modify_circuit(qc)

    # Check if backend is a simulator
    use_simulator = is_simulator(backend)

    # Store results from all analyses
    results = {}

    if use_simulator:
        # Use Aer simulator for execution
        try:
            simulator_backend = Aer.get_backend('aer_simulator')
            transpiled_qc = transpile(qc, backend=simulator_backend)
            time_reversal_qc = time_reversal_simulation(transpiled_qc)
            job = simulator_backend.run(time_reversal_qc, backend=simulator_backend, shots=shots)
            result = job.result()
            counts = result.get_counts()
            print("Counts (simulated): ", counts)
            return counts
        
        except Exception as e:
            print(f"Error executing on simulated backend: {e}")
            return None
        
        # Use hardware backend for Von Nuemman entropy only
        counts = run_and_extract_counts_quantum(qc, backend, shots)
        analyze_von_neumman_entropy(counts)


    else:
        # Use IBM Runtime Sampler for hardware execution
        try:
            transpiled_qc = transpile(qc, backend=backend)
            
            with Session(backend=backend) as session:
                sampler = Sampler()
                job = sampler.run([transpiled_qc], shots=shots)
                result = job.result()

            # Debug: Inspect the raw result object
            print("Raw Result Object:", result)

            # Access the first `SamplerPubResult`
            pub_result = result._pub_results[0]  # Safely access `SamplerPubResult`
            data_bin = pub_result.data  # Access `DataBin`
            bit_array = data_bin.meas  # Access the `BitArray`

            print("Bit array: ", bit_array)

            # Convert the BitArray to a list of bitstrings
            results = extract_counts_from_bitarray(bit_array)

            analyze_shannon_entropy(results)

        except Exception as e:
            print(e)

    return results


def run_and_extract_counts_quantum(qc, backend, shots=8192):
    """
    Runs a quantum circuit on the specified backend and extracts measurement results.

    Parameters:
        qc (QuantumCircuit): The quantum circuit to execute.
        backend (Backend): The IBM backend to use for the execution.
        shots (int): Number of shots for the experiment.

    Returns:
        dict: A dictionary of bitstring counts from the measurement results.
    """

    # Transpile the circuit for the backend
    transpiled_qc = transpile(qc, backend=backend)
    
    # Run the circuit with the sampler and session
    with Session(backend=backend) as session:
        sampler = Sampler()
        job = sampler.run([transpiled_qc], shots=shots)
        result = job.result()

    # Debug: Inspect the raw result object
    print("Raw Result Object:", result)

    try:
        # Access the first `SamplerPubResult`
        pub_result = result._pub_results[0]  # Safely access `SamplerPubResult`
        data_bin = pub_result.data  # Access `DataBin`
        bit_array = data_bin.meas  # Access the `BitArray`

        print("Bit array: ", bit_array)

    except Exception as e:
        print("Something went wrong analyzing the bit array: ", e)
        
    # Check if the backend is a simulator
    is_simulator = backend.configuration().simulator

    if is_simulator:
        # Use Aer simulator for execution
        try:
            simulator_backend = Aer.get_backend('aer_simulator')
            transpiled_qc = transpile(qc, backend=simulator_backend)
            job = simulator_backend.run(transpiled_qc, backend=simulator_backend, shots=shots)
            result = job.result()
            counts = result.get_counts()
            print("Counts (simulated):", counts)
            return counts
        
        except Exception as e:
            print(f"Error executing on simulated backend: {e}")
            return None
        
    else:
        # Use IBM Runtime Sampler for hardware execution
        try:
            transpiled_qc = transpile(qc, backend=backend)
            with Session(backend=backend) as session:
                sampler = Sampler()
                job = sampler.run([transpiled_qc], shots=shots)
                result = job.result()

            # Debug: Inspect the raw result object
            print("Raw Result Object:", result)

            # Access the first `SamplerPubResult`
            pub_result = result._pub_results[0]  # Safely access `SamplerPubResult`
            data_bin = pub_result.data  # Access `DataBin`
            bit_array = data_bin.meas  # Access the `BitArray`

            print("Bit array: ", bit_array)

            # Convert the BitArray to a list of bitstrings
            results = extract_counts_from_bitarray(bit_array)
            print("Results: ", results)

            entropy = analyze_shannon_entropy(results)
            print("Entropy: ", entropy)

        except Exception as e:
            print(e)

    return results



def run_and_extract_counts_quantum_entropy(qc, backend, shots=8192):
    """
    Runs a quantum circuit on the specified backend and extracts measurement results.

    Parameters:
        qc (QuantumCircuit): The quantum circuit to execute.
        backend (Backend): The IBM backend to use for the execution.
        shots (int): Number of shots for the experiment.

    Returns:
        dict: A dictionary of bitstring counts from the measurement results.
    """
    from qiskit_ibm_runtime import Session

    # Transpile the circuit for the backend
    transpiled_qc = transpile(qc, backend=backend)
    
    # Run the circuit with the sampler and session
    with Session(backend=backend) as session:
        sampler = Sampler()
        job = sampler.run([transpiled_qc], shots=shots)
        result = job.result()

    # Debug: Inspect the raw result object
    print("Raw Result Object:", result)

    try:
        # Access the first `SamplerPubResult`
        pub_result = result._pub_results[0]  # Safely access `SamplerPubResult`
        data_bin = pub_result.data  # Access `DataBin`
        bit_array = data_bin.meas  # Access the `BitArray`

        print("Bit array: ", bit_array)

        # Convert the BitArray to counts
        counts = Counts.from_memory(memory=bit_array.to_list(), shots=shots)
        print("Counts (get_counts):", counts)
        analyze_entropy_dynamics(counts)
        return counts

    except Exception as e:
        print("Error extracting counts:", e)
        return None


def process_bitarray(bit_array):
    """
    Decodes a BitArray into a dictionary of measurement counts.

    Parameters:
        bit_array (BitArray): The measurement results as a BitArray.

    Returns:
        dict: A dictionary of bitstring counts.
    """
    # Decode BitArray into a list of bitstrings
    bitstrings = [bit_array[i] for i in range(len(bit_array))]
    
    # Count occurrences of each bitstring
    counts = Counter(bitstrings)
    return dict(counts)

def set_variables_and_run(num_injection_cycles, num_radiation_qubits):
    num_iterations = num_injection_cycles # Number of charge injection cycles
    num_radiation_qubits = num_radiation_qubits  # Number of radiation qubits

    # Run simulation and entropy analysis
    results = simulate_and_analyze(num_iterations, num_radiation_qubits, backend)

    # Print the results
    print("Measurement Results (Counts):", results["counts"])
    print("Entropy Analysis (Entropies):", results["entropies"])

    return results

def time_reversal_simulation(qc):
    """
    Modifies the circuit to simulate time reversal by inverting gates.
    Handles the fact that measurement operations cannot be inverted.
    """
    # Create a copy of the circuit without measurements
    qc_no_measure = qc.remove_final_measurements(inplace=False)

    # Invert the circuit
    reversed_qc = qc_no_measure.inverse()

    # Add measurement gates back to the inverted circuit
    reversed_qc.measure_all()

    return reversed_qc

# Define a simple quantum circuit
def create_base_circuit():
    """
    Creates a base quantum circuit with entanglement.
    """
    qc = QuantumCircuit(2)
    qc.h(0)  # Apply Hadamard gate to qubit 0
    qc.cx(0, 1)  # Apply CNOT gate to entangle qubits 0 and 1
    qc.measure_all()  # Measure all qubits
    return qc

def analyze_von_neumann_entropy(statevector):
    """
    Analyzes the Von Neumann entropy of a quantum state.

    Parameters:
        statevector (np.ndarray): The statevector of the quantum system.

    Returns:
        dict: Analysis results, including Von Neumann entropy.
    """
    if statevector is None:
        print("Statevector is None; cannot calculate Von Neumann entropy.")
        return None

    # Construct the density matrix
    density_matrix = np.outer(statevector, np.conj(statevector))

    # Calculate eigenvalues of the density matrix
    eigenvalues = np.linalg.eigvalsh(density_matrix)

    # Filter out zero eigenvalues to avoid log(0)
    eigenvalues = eigenvalues[eigenvalues > 0]

    # Calculate Von Neumann entropy: S = -Tr(ρ log(ρ))
    von_neumann_entropy = -np.sum(eigenvalues * np.log2(eigenvalues))

    print("Von Neumann Entropy:", von_neumann_entropy)
    return {"von_neumann_entropy": von_neumann_entropy}

def is_simulator(backend):
    """
    Determines if the given backend is a simulator.

    Parameters:
        backend: Qiskit backend object.

    Returns:
        bool: True if the backend is a simulator, False otherwise.
    """
    return backend.configuration().simulator


def run_circuit_statevector(qc):
    """
    Executes the quantum circuit on a statevector simulator.

    Parameters:
        qc (QuantumCircuit): The quantum circuit to execute.

    Returns:
        np.ndarray: The statevector of the circuit.
    """
    simulator = Aer.get_backend('statevector_simulator')
    job = simulator.run(qc, backend=simulator)
    result = job.result()

    # Retrieve the statevector
    statevector = result.get_statevector(qc)
    print("Statevector:", statevector)
    return statevector

def select_backend(use_simulator=True, hardware_backend=None):
    """
    Selects a backend based on the user's preference for simulation or hardware.

    Parameters:
        use_simulator (bool): If True, use a simulated backend.
        hardware_backend (str): Name of the hardware backend to use if not simulating.

    Returns:
        backend: A Qiskit backend object (simulated or hardware).
    """
    if use_simulator:
        try:
            backend = Aer.get_backend('aer_simulator')
            print("Using simulated backend: Aer Simulator")
            return backend
        
        except Exception as e:
            print(f"Error initializing simulated backend: {e}")
            return None
    else:
        if hardware_backend:
            try:
                from qiskit_ibm_runtime import IBMRuntimeService
                service = IBMRuntimeService()
                backend = service.backend(hardware_backend)
                print(f"Using hardware backend: {hardware_backend}")
                return backend
            except Exception as e:
                print(f"Error initializing hardware backend {hardware_backend}: {e}")
                return None
        else:
            print("No hardware backend specified. Please provide a valid backend name.")
            return None

def initialize_backend(use_simulator=True, hardware_backend_name="ibm_brisbane"):
    """
    Initializes the appropriate backend (simulator or hardware) based on user preference.

    Parameters:
        use_simulator (bool): If True, initialize a simulated backend.
        hardware_backend_name (str): Name of the hardware backend if use_simulator is False.

    Returns:
        backend: The initialized backend.
    """
    if use_simulator:
        # Initialize simulator backend
        backend = Aer.get_backend('aer_simulator')
        print("Using simulated backend: aer_simulator")
        return backend
    
    else:
        # Initialize IBM Quantum service and select hardware backend
        try:
            service = QiskitRuntimeService(channel="ibm_quantum")
            backend = service.backend(hardware_backend_name)
            print(f"Using hardware backend: {hardware_backend_name}")
            return backend
        
        except Exception as e:
            print(f"Error initializing hardware backend '{hardware_backend_name}': {e}")
            return None
        
    return backend

def calculate_subsystem_entropy(qc):
    """
    Calculates the Von Neumann entropy of the black hole and radiation subsystems.

    Parameters:
        qc (QuantumCircuit): The quantum circuit to analyze.

    Returns:
        dict: Entropy values for the black hole and radiation subsystems.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import entropy, partial_trace, Statevector

    try:
        # Remove measurement gates
        qc_no_measure = QuantumCircuit(qc.num_qubits)
        for instr, qargs, cargs in qc.data:
            if instr.name != "measure":
                qc_no_measure.append(instr, qargs, cargs)

        # Generate the statevector
        state = Statevector.from_instruction(qc_no_measure)

        # Trace out subsystems
        black_hole_state = partial_trace(state, [1])  # Trace out radiation qubit
        radiation_state = partial_trace(state, [0])  # Trace out black hole qubit

        # Calculate Von Neumann entropy
        black_hole_entropy = entropy(black_hole_state, base=2)
        radiation_entropy = entropy(radiation_state, base=2)

        return {
            "black_hole_entropy": black_hole_entropy,
            "radiation_entropy": radiation_entropy
        }
    except Exception as e:
        print(f"Error calculating subsystem entropy: {e}")
        return None


def time_reverse_circuit(qc):
    """
    Creates a time-reversed version of the quantum circuit.

    Parameters:
        qc (QuantumCircuit): The quantum circuit to time-reverse.

    Returns:
        QuantumCircuit: The time-reversed circuit.
    """
    try:
        # Remove measurement gates to allow inversion
        qc_no_measure = qc.remove_final_measurements(inplace=False)

        # Invert the circuit
        reversed_qc = qc_no_measure.inverse()

        # Add measurement gates back
        reversed_qc.measure_all()

        return reversed_qc
    except Exception as e:
        print(f"Error creating time-reversed circuit: {e}")
        return None



if __name__ == "__main__":

    # Set this to False if you want to use hardware
    use_simulator = False
    hardware_backend_name = "ibm_brisbane"  # Replace with your hardware backend name if needed

    # Initialize IBM Quantum Service
    
    #service = QiskitRuntimeService(channel="ibm_quantum")
    
    backend = initialize_backend(use_simulator=use_simulator, hardware_backend_name=hardware_backend_name)  # Use your backend selection logic

    # Run the experiment
    if backend:
        # Create base circuit
        qc_base = create_base_circuit()

        # Store results from iterations
        iteration_results = []

        for i in range(10):  # Increase iterations for temporal analysis
            print(f"\nIteration {i + 1}")

            # Modify circuit (e.g., add causality)
            qc_modified = add_causality_to_circuit(qc_base.copy(), iteration_results[-1] if iteration_results else None, qubits=[0, 1])

            # Run circuit and store results
            results = run_and_extract_counts_quantum(qc=qc_modified, backend=backend, shots=8192)
            iteration_results.append(results)

            # Analyze subsystem entropy
            entropies = calculate_subsystem_entropy(qc_modified)
            print(f"Subsystem Entropy: {entropies}")

        # Analyze temporal correlation
        temporal_divergences = analyze_temporal_correlation(iteration_results)
        print(f"Temporal Correlation (Jensen-Shannon Divergences): {temporal_divergences}")

        # Apply time-reversal and re-run
        qc_reversed = time_reverse_circuit(qc_base)
        reversed_results = run_and_extract_counts_quantum(qc=qc_reversed, backend=backend, shots=8192)
        print(f"Time-Reversed Results: {reversed_results}")
            
    else:
        print("Failed to initialize backend.")


