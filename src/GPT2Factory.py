from qiskit import QuantumCircuit, QuantumRegister, pulse, transpile, ClassicalRegister, assemble
from qiskit.quantum_info import Statevector, partial_trace, entropy, DensityMatrix, state_fidelity, random_unitary, Operator, mutual_information
from qiskit_aer import Aer, AerSimulator
from qiskit.pulse import Play, DriveChannel, Gaussian, Schedule
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, amplitude_damping_error
from collections import Counter
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from scipy.spatial.distance import jensenshannon
from qiskit.result import Result, Counts  # Import for local simulation results
from qiskit.circuit.library import QFT, RZGate, MCXGate, ZGate, XGate, HGate
import random
from scipy.linalg import expm
import hashlib
from qiskit.circuit import Instruction
import time
from datetime import datetime

# Select a backend
def get_best_backend(service, min_qubits=2, max_queue=10):
    backends = service.backends()
    suitable_backends = [
        b for b in backends if b.configuration().n_qubits >= min_qubits and b.status().pending_jobs <= max_queue
    ]
    if not suitable_backends:
        print("No suitable backends found. Using default: ibm_kyiv")
        return service.backend("ibm_kyiv")
    
    best_backend = sorted(suitable_backends, key=lambda b: b.status().pending_jobs)[0]
    print(f"Best backend chosen: {best_backend.name}")
    return best_backend

def apply_entanglement(qc, qr):
    """Apply entanglement to the quantum circuit."""
    for i in range(len(qr) - 1):
        qc.cx(qr[i], qr[i + 1])

def measure_and_reset(qc, qr, cr):
    """Measure the qubits and reset them for the next decision."""
    qc.measure(qr, cr)
    qc.barrier()
    for qubit in qr:
        qc.reset(qubit)

def add_decision_to_chain(previous_qc, previous_qr, previous_cr, decision_bits):
    """
    Add a new decision to the existing quantum decision chain.
    :param previous_qc: Previous QuantumCircuit object
    :param previous_qr: Previous QuantumRegister object
    :param previous_cr: Previous ClassicalRegister object
    :param decision_bits: List of bits representing the new decision
    :return: Updated QuantumCircuit, QuantumRegister, and ClassicalRegister
    """
    num_qubits = len(previous_qr)
    new_qr = QuantumRegister(num_qubits, name=f'q{len(previous_qc.data)}')
    new_cr = ClassicalRegister(num_qubits, name=f'c{len(previous_qc.data)}')
    new_qc = QuantumCircuit(new_qr, new_cr)

    # Encode the new decision
    encode_decision(new_qc, new_qr, decision_bits)

    # Apply entanglement
    apply_entanglement(new_qc, new_qr)

    # Use compose() to combine with the previous circuit
    combined_qc = QuantumCircuit(previous_qr, new_qr, previous_cr, new_cr)
    combined_qc.compose(previous_qc, inplace=True)
    combined_qc.compose(new_qc, inplace=True)

    return combined_qc, new_qr, new_cr

def encode_disruptive_decision(decision_text):
    """Encodes a high-impact decision into a quantum circuit."""
    binary_decision = ''.join(format(ord(c), '08b') for c in decision_text)
    n_qubits = min(len(binary_decision), 5)  # Limit qubits to 5 for stability
    qr = QuantumRegister(n_qubits, 'q')
    cr = ClassicalRegister(n_qubits, 'c')
    qc = QuantumCircuit(qr, cr)

    # Apply Hadamard and phase gates based on binary decision encoding
    for i, bit in enumerate(binary_decision[:n_qubits]):
        if bit == '1':
            qc.h(qr[i])
            qc.p(np.pi / 4, qr[i])  # Phase shift for encoding
        else:
            qc.x(qr[i])

    qc.measure(qr, cr)
    return qc

def evolve_decision_over_time(qc, iterations=5):
    """Amplifies or shifts decisions dynamically over iterations."""
    for _ in range(iterations):
        for q in range(qc.num_qubits):
            qc.rx(np.pi / 8, q)  # Gradual evolution of decision state
            qc.rz(np.pi / 8, q)

    return qc

def measure_decision_influence(previous_qc, new_qc, sim_tf=False):
    """Measures how much a new decision shifts prior decisions using fidelity."""
    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=sim_tf)
    sampler = Sampler(backend)

    # Execute previous decision circuit
    transpiled_prev = transpile(previous_qc, backend)
    result_prev = sampler.run([transpiled_prev], shots=8192).result()
    print(result_prev)
    counts_prev = extract_counts_from_bitarray(result_prev[0].data.c)

    # Execute new decision circuit
    transpiled_new = transpile(new_qc, backend)
    result_new = sampler.run([transpiled_new], shots=8192).result()
    counts_new = extract_counts_from_bitarray(result_new[0].data.c)

    # Get all possible measurement outcomes
    all_keys = set(counts_prev.keys()).union(set(counts_new.keys()))

    # Normalize probabilities
    prev_probs = np.array([counts_prev.get(k, 0) / 1024 for k in all_keys])
    new_probs = np.array([counts_new.get(k, 0) / 1024 for k in all_keys])

    # Compute fidelity using aligned probability vectors
    fidelity = np.sum(np.sqrt(prev_probs * new_probs))

    return fidelity, counts_prev, counts_new


def store_decision_holographically(qc, decision_text):
    """Encodes decision memory into a holographic transformation."""
    binary_decision = ''.join(format(ord(c), '08b') for c in decision_text)
    n_qubits = qc.num_qubits
    qr = QuantumRegister(n_qubits, 'holo_q')
    qc.add_register(qr)

    # Apply holographic encoding via controlled rotations
    for i, bit in enumerate(binary_decision[:n_qubits]):
        if bit == '1':
            qc.crx(np.pi / 6, qr[i], qc.qubits[i])  # Controlled rotation
        else:
            qc.cry(np.pi / 6, qr[i], qc.qubits[i])  # Alternate encoding

    return qc


def execute_circuit(qc):
    """Execute the quantum circuit and return the measurement results."""
    simulator = Aer.get_backend('qasm_simulator')
    result = simulator.run(qc, shots=1024).result()
    counts = result.get_counts(qc)
    return counts

def encode_decision(qc, qr, decision_bits):
    """
    Encode a decision into the quantum circuit.
    :param qc: QuantumCircuit object
    :param qr: QuantumRegister object
    :param decision_bits: List of bits representing the decision
    """
    for i, bit in enumerate(decision_bits):
        if bit == '1':
            qc.x(qr[i])

# Used in our decision making framework
def initialize_qubits(num_qubits):
    """Initialize a quantum register and corresponding quantum circuit."""
    qr = QuantumRegister(num_qubits, name='q')
    cr = ClassicalRegister(num_qubits, name='c')
    qc = QuantumCircuit(qr, cr)
    return qc, qr, cr

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

def set_variables_and_run(num_injection_cycles, num_radiation_qubits):
    num_iterations = num_injection_cycles # Number of charge injection cycles
    num_radiation_qubits = num_radiation_qubits  # Number of radiation qubits

    # Run simulation and entropy analysis
    results = simulate_and_analyze(num_iterations, num_radiation_qubits, backend)

    # Print the results
    print("Measurement Results (Counts):", results["counts"])
    print("Entropy Analysis (Entropies):", results["entropies"])

    return results

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

# Function to calculate Shannon entropy
def calculate_shannon_entropy(counts, num_shots):
    probabilities = {key: value / num_shots for key, value in counts.items()}
    entropy = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)
    return entropy

# Function to calculate von Neumann entropy for black hole functions
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

def warp_information_circuit():
    """
    Creates a 3-qubit circuit to test entanglement survival under spacetime distortions.
    """
    qc = QuantumCircuit(3)  # Three qubits

    # Step 1: Initialize entanglement
    qc.h(0)  # Superposition for the "center"
    qc.cx(0, 1)  # Entangle "center" with "boundary"
    qc.cx(1, 2)  # Extend entanglement to "outside"

    # Step 2: Apply spacetime distortions
    qc.rz(np.pi / 4, 0)  # Phase shift on the "center"
    qc.cp(np.pi / 3, 0, 1)  # Controlled phase between "center" and "boundary"
    qc.cp(np.pi / 6, 1, 2)  # Controlled phase between "boundary" and "outside"
    qc.rx(np.pi / 8, 2)  # Additional distortion on "outside"

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
    # Remove final measurements before inversion
    qc_no_measurements = qc.remove_final_measurements(inplace=False)
    
    # Invert the circuit
    qc_reversed = qc_no_measurements.inverse()
    
    # Add measurements back
    qc_reversed.measure_all()

    return qc_reversed


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
            time_reversal_qc.measure_all()
            transpiled_qc.measure_all()
            job = simulator_backend.run(transpiled_qc, backend=simulator_backend, shots=shots)
            result = job.result()
            counts = result.get_counts()
            print("Counts (simulated): ", counts)
            qc_no_measure = qc.remove_final_measurements(inplace=False)
            state = Statevector.from_instruction(qc_no_measure)
            analyze_shannon_entropy(counts)
            analyze_von_neumann_entropy(state)
            
            return counts
        
        except Exception as e:
            print(f"Error executing on simulated backend: {e}")
            return None
        
        # Use hardware backend for Von Nuemman entropy only
        counts = run_and_extract_counts_quantum(qc, backend, shots)
        


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

def is_simulator(backend):
    """
    Determines if the given backend is a simulator.

    Parameters:
        backend: Qiskit backend object.

    Returns:
        bool: True if the backend is a simulator, False otherwise.
    """
    return backend.configuration().simulator

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

def initialize_backend(use_simulator=True, hardware_backend_name="ibm_kyiv"):
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
    """Calculates the entropy of subsystems in a given quantum circuit."""
    try:
        # Check if the circuit has classical bits (i.e., has been measured)
        if qc.num_clbits == 0:
            print("⚠️ No classical bits found. Adding measurements...")
            qc.measure_all()

        # Simulate the quantum circuit to get the final density matrix
        simulator = Aer.get_backend('aer_simulator_density_matrix')
        qc_copy = qc.copy()  # Avoid modifying the original circuit
        qc_copy.save_density_matrix()  # Save final state
        compiled_circuit = transpile(qc_copy, simulator)
        result = simulator.run(compiled_circuit).result()

        # Extract the density matrix
        final_density_matrix = DensityMatrix(result.data(0)['density_matrix'])

        # Compute subsystem entropy for each qubit
        num_qubits = qc.num_qubits
        entropies = []

        for i in range(num_qubits):
            try:
                # Trace out all qubits except `i`
                reduced_state = partial_trace(final_density_matrix, [j for j in range(num_qubits) if j != i])
                entropy_val = entropy(reduced_state)
                entropies.append(entropy_val)
            except Exception as e:
                print(f"Error calculating entropy for qubit {i}: {e}")

        return entropies if entropies else None

    except Exception as e:
        print(f"Error calculating subsystem entropy: {e}")
        return None
    
def calculate_subsystem_entropy_hologram(statevector, num_qubits=None):
    """
    Calculates the subsystem entropy for a given statevector.
    Handles circuits with varying numbers of qubits.

    Parameters:
        statevector (Statevector): The statevector of the quantum circuit.
        num_qubits (int): Number of qubits in the circuit. If None, infer from statevector.

    Returns:
        dict: Subsystem entropy values for each subsystem, or None for single-qubit systems.
    """
    if num_qubits is None:
        num_qubits = int(np.log2(len(statevector)))

    if num_qubits < 2:
        # For single-qubit systems, the concept of subsystem entropy doesn't apply
        print("Subsystem entropy not applicable for single-qubit systems.")
        return None

    # Split the system into subsystems
    left_qubits = num_qubits // 2
    right_qubits = num_qubits - left_qubits

    if left_qubits < 1 or right_qubits < 1:
        print("Subsystems are too small for entropy calculation.")
        return None

    # Calculate reduced density matrices
    reduced_density_left = partial_trace(statevector, range(right_qubits))
    reduced_density_right = partial_trace(statevector, range(left_qubits, num_qubits))

    # Calculate von Neumann entropy for each subsystem
    entropy_left = -np.trace(reduced_density_left @ np.log2(reduced_density_left + 1e-12))
    entropy_right = -np.trace(reduced_density_right @ np.log2(reduced_density_right + 1e-12))

    return {
        "left_entropy": entropy_left,
        "right_entropy": entropy_right,
    }

def create_two_qubit_circuit():
    """
    Creates a two-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(2)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1)  # Create entanglement between qubits
    return qc

def create_three_qubit_circuit():
    """
    Creates a three-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(3)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2)  # Create entanglement between qubits
    return qc

def create_four_qubit_circuit():
    """
    Creates a four-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(4)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3)  # Create entanglement between qubits
    return qc

def create_five_qubit_circuit():
    """
    Creates a five-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(5)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4)  # Create entanglement between qubits
    return qc

def create_six_qubit_circuit():
    """
    Creates a six-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(6)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4, 5)  # Create entanglement between qubits
    return qc

def create_seven_qubit_circuit():
    """
    Creates a seven-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(7)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4, 5, 6)  # Create entanglement between qubits
    return qc

def create_eight_qubit_circuit():
    """
    Creates a eight-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(8)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4, 5, 6, 7)  # Create entanglement between qubits
    return qc

def create_nine_qubit_circuit():
    """
    Creates a nine-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(9)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4, 5, 6, 7, 8)  # Create entanglement between qubits
    return qc

def create_ten_qubit_circuit():
    """
    Creates a ten-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(10)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)  # Create entanglement between qubits
    return qc

def create_eleven_qubit_circuit():
    """
    Creates a eleven-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(11)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)  # Create entanglement between qubits
    return qc


def create_twelve_qubit_circuit():
    """
    Creates a twelve-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(12)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)  # Create entanglement between qubits
    return qc


def create_thirteen_qubit_circuit():
    """
    Creates a thirteen-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(13)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)  # Create entanglement between qubits
    return qc


def create_fourteen_qubit_circuit():
    """
    Creates a fourteen-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(14)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)  # Create entanglement between qubits
    return qc


def create_fifteen_qubit_circuit():
    """
    Creates a fifteen-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(15)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)  # Create entanglement between qubits
    return qc


def create_sixteen_qubit_circuit():
    """
    Creates a sixteen-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(16)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)  # Create entanglement between qubits
    return qc


def create_seventeen_qubit_circuit():
    """
    Creates a seventeen-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(17)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)  # Create entanglement between qubits
    return qc


def create_holographic_interaction_circuit():
    """
    Simulates interacting with a holographic boundary using 2 qubits.
    """
    qc = QuantumCircuit(2)
    
    # Step 1: Create superposition on the first qubit
    qc.h(0)

    # Step 2: Introduce entanglement between the two qubits
    qc.cx(0, 1)

    # Step 3: Simulate holographic interaction with synthetic boundary effects
    qc.u(np.pi / 3, np.pi / 6, np.pi / 9, 0)  # Custom unitary gate on qubit 0
    qc.rz(np.pi / 4, 1)  # Rotate the second qubit

    # Step 4: Introduce reverse causality (time-reversal simulation)
    qc.cx(0, 1)  # Reverse entanglement
    qc.h(0)  # Reverse superposition on qubit 0

    # Measure all qubits
    qc.measure_all()
    return qc

def run_holographic_experiment():
    qc = create_holographic_circuit()
    qc = add_holographic_interaction(qc)

    simulator = Aer.get_backend('aer_simulator')
    result = simulator.run(qc, simulator, shots=8192).result()
    counts = result.get_counts()
    
    # Analyze Statevector
    statevector = result.get_statevector(qc)
    entropies = calculate_subsystem_entropy(statevector)

    return counts, entropies

def run_holographic_experiment_2():
    qc = create_holographic_interaction_circuit()
    qc_hologram = add_holographic_interaction(qc)

    simulator = Aer.get_backend('aer_simulator')
    result = simulator.run(qc_hologram, simulator, shots=8192).result()
    counts = result.get_counts()
    
    # Analyze Statevector
    statevector = result.get_statevector(qc)
    entropies = calculate_subsystem_entropy(statevector)

    print(counts)

    print(entropies)

    return counts, entropies

def hack_hologram_with_injections(qc, backend, num_iterations=10, shots=8192):
    """
    Hacks the hologram through iterative charge injections.
    
    Args:
        qc (QuantumCircuit): The initial quantum circuit.
        backend: The Quantum Inspire backend for execution.
        num_iterations (int): Number of iterations for the hacking attempt.
        shots (int): Number of shots for each experiment.
    
    Returns:
        list[dict]: Results from each iteration.
    """
    results = []
    
    for iteration in range(1, num_iterations + 1):
        print(f"--- Iteration {iteration} ---")
        
        # Create a copy of the circuit for this iteration
        modified_qc = qc.copy()

        # Apply charge injections
        injection_type = "random"  # Alternate between types if needed
        modified_qc = inject_charge(modified_qc, qubits=[0, 1], injection_type=injection_type)

        # Run the circuit
        try:
            counts = run_and_extract_counts_qi(modified_qc, backend, shots)
            if counts:
                print(f"Iteration {iteration} Results: {counts}")
                results.append({
                    "iteration": iteration,
                    "counts": counts
                })

                # Analyze entropy
                statevector = run_circuit_statevector(modified_qc)
                entropies = calculate_subsystem_entropy(statevector)
                results[-1]["entropies"] = entropies
                print(f"Entropies: {entropies}")
            else:
                print(f"Failed in iteration {iteration}")
        except Exception as e:
            print(f"Error during iteration {iteration}: {e}")
            results.append({
                "iteration": iteration,
                "counts": None,
                "error": str(e)
            })

    # Summarize results
    print("\nFinal Results Summary:")
    for result in results:
        iteration = result.get("iteration", "N/A")
        counts = result.get("counts", "N/A")
        entropies = result.get("entropies", "N/A")
        print(f"Iteration {iteration}: Counts: {counts}, Entropies: {entropies}")
    
    return results

def create_holographic_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc

def reverse_time(qc):
    return qc.inverse()

def add_holographic_interaction(qc):
    # Example: Modify the circuit to simulate a holographic boundary interaction
    qc.rx(np.pi / 4, 0)  # Rotate qubit 0
    qc.rz(np.pi / 4, 1)  # Rotate qubit 1
    return qc
    
def create_lower_dimensional_circuit():
    """
    Creates a lower-dimensional quantum circuit.
    This circuit operates on a single qubit and optionally adds minimal entanglement.
    """
    qc = QuantumCircuit(1)  # Single qubit for lower dimensionality
    
    # Apply a basic superposition
    qc.h(0)  # Hadamard gate for superposition

    # Minimal operations to keep dimensionality low
    qc.rx(np.pi / 4, 0)  # Rotate the qubit slightly
    qc.rz(np.pi / 4, 0)  # Another rotation to modify the phase

    # Optionally add measurement to finalize the state
    qc.measure_all()

    return qc

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

# Define a simple quantum circuit
def create_base_circuit(clbits=2):
    """
    Creates a base quantum circuit with entanglement and ensures classical bits match requirements.
    """
    qc = QuantumCircuit(2)  # Start with 2 qubits

    # If classical register does not exist or is too small, add/resize it
    current_clbits = sum(creg.size for creg in qc.cregs)  # Count existing classical bits
    if current_clbits < clbits:
        additional_clbits = clbits - current_clbits
        c = ClassicalRegister(additional_clbits, "c")
        qc.add_register(c)

    qc.h(0)  # Apply Hadamard gate to qubit 0
    qc.cx(0, 1)  # Apply CNOT gate to entangle qubits 0 and 1
    classical_bits = [qc.cregs[0][i] for i in range(clbits)]  # Reference the classical register
    qc.measure(qc.qubits, classical_bits)  # Explicitly map qubits to classical bits

    return qc


def inject_charge(qc, qubits, injection_type="random"):
    """
    Injects charge (gates) into the quantum circuit on specified qubits.

    Args:
        qc (QuantumCircuit): The quantum circuit.
        qubits (list): List of qubits to target.
        injection_type (str): Type of injection ("X", "Y", "Z", or "random").
    
    Returns:
        QuantumCircuit: Modified circuit with injections.
    """
    for qubit in qubits:
        if injection_type == "random":
            gate = random.choice(["x", "y", "z"])
        else:
            gate = injection_type.lower()
        
        if gate == "x":
            qc.x(qubit)  # Bit-flip
        elif gate == "y":
            qc.y(qubit)  # Bit+Phase flip
        elif gate == "z":
            qc.z(qubit)  # Phase flip

    return qc

def multiverse_warp_circuit():
    """
    Creates a circuit to simulate a warp bubble interacting with alternate timelines.
    """
    # Create a quantum circuit with 5 qubits: 3 for the bubble, 2 for alternate timelines
    qc = QuantumCircuit(5)

    # Step 1: Initialize entanglement in the warp bubble
    qc.h(0)  # Superposition for the "center"
    qc.cx(0, 1)  # Entangle "center" with "boundary"
    qc.cx(1, 2)  # Extend entanglement to "outside"

    # Step 2: Initialize alternate timelines
    qc.h(3)  # Superposition for timeline 1
    qc.h(4)  # Superposition for timeline 2

    # Step 3: Apply interactions between bubble and timelines
    qc.cp(np.pi / 3, 0, 3)  # Controlled phase between center and timeline 1
    qc.cp(np.pi / 4, 1, 4)  # Controlled phase between boundary and timeline 2
    qc.cx(2, 3)  # Entangle outer region with timeline 1
    qc.cx(3, 4)  # Entangle timeline 1 with timeline 2

    # Step 4: Apply distortions to the timelines
    qc.rz(np.pi / 6, 3)  # Phase shift on timeline 1
    qc.rx(np.pi / 8, 4)  # Rotation on timeline 2

    return qc

def analyze_multiverse_entanglement(qc):
    """
    Analyzes the survival of entanglement and mutual information in a multiverse scenario.
    """
    # Get the quantum statevector
    state = Statevector.from_instruction(qc)

    # Trace out subsystems to calculate entropies
    subsystems = [
        partial_trace(state, [1, 2, 3, 4]),  # Center
        partial_trace(state, [0, 2, 3, 4]),  # Boundary
        partial_trace(state, [0, 1, 3, 4]),  # Outside
        partial_trace(state, [0, 1, 2, 4]),  # Timeline 1
        partial_trace(state, [0, 1, 2, 3])   # Timeline 2
    ]
    entropies = [entropy(sub, base=2) for sub in subsystems]

    # Combined subsystems for mutual information
    combined_center_timeline1 = partial_trace(state, [1, 2, 4])  # Center + Timeline 1
    combined_boundary_timeline2 = partial_trace(state, [0, 2, 3])  # Boundary + Timeline 2

    # Calculate mutual information
    mi_center_timeline1 = (
        entropies[0] + entropies[3] - entropy(combined_center_timeline1, base=2)
    )
    mi_boundary_timeline2 = (
        entropies[1] + entropies[4] - entropy(combined_boundary_timeline2, base=2)
    )

    # Print subsystem entropies
    print("Subsystem Entropies:")
    print(f"Center (Qubit 0): {entropies[0]:.4f}")
    print(f"Boundary (Qubit 1): {entropies[1]:.4f}")
    print(f"Outside (Qubit 2): {entropies[2]:.4f}")
    print(f"Timeline 1 (Qubit 3): {entropies[3]:.4f}")
    print(f"Timeline 2 (Qubit 4): {entropies[4]:.4f}")

    # Print mutual information
    print("\nMutual Information:")
    print(f"Center ↔ Timeline 1: {mi_center_timeline1:.4f}")
    print(f"Boundary ↔ Timeline 2: {mi_boundary_timeline2:.4f}")

    return entropies, mi_center_timeline1, mi_boundary_timeline2

def least_busy_backend(service, filters=None):
    """
    Find the least busy backend from the available IBM Quantum backends.

    Parameters:
        service (QiskitRuntimeService): An initialized QiskitRuntimeService object.
        filters (function): A lambda function to filter the list of backends.

    Returns:
        Backend: The least busy backend that matches the filter criteria.
    """
    # Get all backends
    backends = service.backends()

    # Apply filters if provided
    if filters:
        backends = list(filter(filters, backends))

    # Sort by the number of pending jobs (ascending)
    sorted_backends = sorted(
        backends, key=lambda b: b.status().pending_jobs
    )

    # Return the least busy backend
    return sorted_backends[0] if sorted_backends else None

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


def dynamic_warp_circuit(t_steps):
    """
    Creates a dynamically evolving warp hologram circuit.
    """
    qc = QuantumCircuit(3, 3)  # 3 qubits, 3 classical bits for measurement

    # Initialize entanglement
    qc.h(0)  # Superposition for "center"
    qc.cx(0, 1)  # Entangle "center" with "boundary"
    qc.cx(1, 2)  # Extend entanglement to the "outside"

    # Time evolution with phase shifts
    for t in range(1, t_steps + 1):
        qc.rz(np.pi * t / 10, 0)  # Time-dependent phase shift on Qubit 0
        qc.rx(np.pi * t / 15, 1)  # Time-dependent rotation on Qubit 1
        qc.cp(np.pi * t / 20, 1, 2)  # Controlled phase interaction between Qubits 1 and 2

    qc.measure([0, 1, 2], [0, 1, 2])  # Measure all qubits
    return qc

def run_warp_simulation_measure(qc):
    """
    Runs the quantum circuit using a statevector simulator.
    """
    # Simulate the quantum circuit using Statevector
    state = Statevector.from_instruction(qc)

    # Analyze the statevector
    probabilities = state.probabilities_dict()
    print("\nState Probabilities (Quantum):")
    for state, prob in probabilities.items():
        print(f"State {state}: {prob:.4f}")

    return probabilities

def process_sampler_result(result, shots=2048):
    """
    Process and format the output from a Qiskit Sampler run result.

    Parameters:
    - result: The result object from a Sampler run.
    - shots (int): The number of shots used in the experiment.

    Returns:
    - readable_output (str): Formatted string summarizing the result.
    - binary_probabilities (dict): Measurement outcomes as binary probabilities.
    """
    readable_output = ""
    binary_probabilities = {}

    try:
        # Extract quasi-probabilities from the result
        quasi_dists = result.quasi_dists[0]  # Assuming a single circuit
        binary_probabilities = quasi_dists.binary_probabilities()

        # Generate readable summary
        readable_output += f"Total Shots: {shots}\n"
        readable_output += f"{'State':<10}{'Probability (%)':<20}{'Quasi-Probability':<20}\n"
        readable_output += "-" * 50 + "\n"

        for state, probability in binary_probabilities.items():
            quasi_prob = quasi_dists.get(state, 0)
            readable_output += f"{state:<10}{probability * 100:<20.5f}{quasi_prob:<20.5f}\n"

    except AttributeError as e:
        readable_output += f"Error processing result: {e}\n"
        readable_output += "Ensure the result contains valid quasi-probabilities.\n"

    return readable_output, binary_probabilities



def dynamic_warp_circuit_no_measure(t_steps):
    """
    Creates a dynamically evolving warp hologram circuit.
    """
    qc = QuantumCircuit(3, 3)  # 3 qubits, 3 classical bits for measurement

    # Initialize entanglement
    qc.h(0)  # Superposition for "center"
    qc.cx(0, 1)  # Entangle "center" with "boundary"
    qc.cx(1, 2)  # Extend entanglement to the "outside"

    # Time evolution with phase shifts
    for t in range(1, t_steps + 1):
        qc.rz(np.pi * t / 10, 0)  # Time-dependent phase shift on Qubit 0
        qc.rx(np.pi * t / 15, 1)  # Time-dependent rotation on Qubit 1
        qc.cp(np.pi * t / 20, 1, 2)  # Controlled phase interaction between Qubits 1 and 2

    qc_no_measure = qc.copy()

    qc.measure_all()

    # Get the statevector of the circuit (without measurements)
    state = Statevector.from_instruction(qc_no_measure)
    print("\nStatevector of the System:")
    print(state)

    # Subsystem isolation: trace out specific qubits
    subsystem_1 = partial_trace(state, [1, 2])  # Isolate Qubit 0
    subsystem_2 = partial_trace(state, [0, 2])  # Isolate Qubit 1
    subsystem_3 = partial_trace(state, [0, 1])  # Isolate Qubit 2

    # Compute entropies for each subsystem
    entropy_1 = entropy(subsystem_1)
    entropy_2 = entropy(subsystem_2)
    entropy_3 = entropy(subsystem_3)

    print("\nSubsystem Entropies:")
    print(f"Qubit 0 Entropy: {entropy_1}")
    print(f"Qubit 1 Entropy: {entropy_2}")
    print(f"Qubit 2 Entropy: {entropy_3}")

    # Transpile the circuit for AerSimulator
    backend = Aer.get_backend('aer_simulator')
    transpiled_qc = transpile(qc, backend)

    # Run the circuit and get measurement results
    result = backend.run(transpiled_qc, shots=2048).result()
    counts = result.get_counts()

    print("\nMeasurement Results:")
    print(counts)

    # Plot histogram of measurement results
    plot_histogram(counts)
    plt.title("Measurement Results Histogram")
    plt.show()

    return qc

def time_evolution_example(t_steps=5, shots=2048):
    """
    Simulate time-dependent transformations and holographic interactions.
    """
    n_qubits = 3
    qc = QuantumCircuit(n_qubits)
    
    # Step 1: Initialize superposition
    qc.h(0)  # Superposition on the Black Hole qubit
    qc.cx(0, 1)  # Entangle Black Hole and Radiation qubits

    # Step 2: Time Evolution
    for t in range(t_steps):
        angle = (np.pi / 3) * (t + 1)  # Dynamic angle for timeline distortion
        qc.rz(angle, 0)  # Timeline distortion on the Black Hole qubit
        qc.rx(np.pi / 4, 1)  # Holographic interaction on the Radiation qubit
        qc.cx(1, 2)  # Correlate Radiation with External Environment
        qc.barrier()

    # Step 3: Measure final probabilities
    qc.measure_all()

    # Analyze statevector before measurement
    qc_no_measurements = qc.copy()
    state = Statevector.from_instruction(qc_no_measurements)
    print("\nStatevector:")
    print(state)

    # Calculate subsystem entropies
    entropy_0 = entropy(partial_trace(state, [1, 2]))
    entropy_1 = entropy(partial_trace(state, [0, 2]))
    entropy_2 = entropy(partial_trace(state, [0, 1]))
    print(f"Subsystem Entropies: Qubit 0 = {entropy_0}, Qubit 1 = {entropy_1}, Qubit 2 = {entropy_2}")

    # Transpile and run on AerSimulator
    backend = Aer.get_backend('aer_simulator')
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()

    print("\nMeasurement Results:")
    print(counts)

    # Plot the histogram of results
    plot_histogram(counts)
    plt.title("Measurement Results with Time Evolution")
    plt.show()

# KEEP IN MIND WHEN USING ANY STATE MANIPULATION, ALWAYS MAKE ALL 3 BITS THE SAME - straight 1 or 0s
# FAILURE to do so will result in decoherent states, which is AGAINST our responsibilities

def targeted_gates(target_state="111", shots=2048):
    """
    Design a circuit with targeted gates to amplify a specific state.
    """
    n_qubits = 3
    qc = QuantumCircuit(n_qubits)
    
    # Initial superposition
    qc.h(range(n_qubits))  # Create equal superposition of all states
    
    # Targeted phase shifts to amplify |111>
    if target_state == "111":
        qc.h(2)             # Reverse the superposition on the last qubit
        qc.ccx(0, 1, 2)     # Apply Toffoli gate to mark |111>
        qc.h(2)             # Return to superposition basis
        qc.z(2)             # Phase flip on |111>
        qc.barrier()
    
    # Additional controlled gates for steering probability
    qc.cp(3.14, 0, 1)  # Controlled phase gate
    qc.cx(1, 2)         # Controlled NOT
    qc.cz(0, 2)         # Controlled Z
    
    # Copy circuit for analysis without measurement
    qc_no_measurements = qc.copy()
    
    # Add measurement gates
    qc.measure_all()

    # Analyze the statevector (before measurement)
    state = Statevector.from_instruction(qc_no_measurements)
    print("\nStatevector:")
    print(state)

    # Calculate subsystem entropy
    entropy_0 = entropy(partial_trace(state, [1, 2]))
    entropy_1 = entropy(partial_trace(state, [0, 2]))
    entropy_2 = entropy(partial_trace(state, [0, 1]))
    print(f"Subsystem Entropies: Qubit 0 = {entropy_0}, Qubit 1 = {entropy_1}, Qubit 2 = {entropy_2}")

    # Transpile and run on a simulator
    backend = Aer.get_backend('aer_simulator')
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()

    print("\nMeasurement Results:")
    print(counts)

    # Plot the histogram of results
    plot_histogram(counts)
    plt.title("Measurement Results with Targeted Gates")
    plt.show()

def hamiltonian_calc(delta=1.0,J=0.5,Omega=0.3,omega=2.0,t_max=10,num_steps=100):
    # Define Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]])  # Pauli-X
    sigma_z = np.array([[1, 0], [0, -1]])  # Pauli-Z
    identity = np.eye(2)  # Identity matrix

    # Define operators for the two-qubit system
    sigma_z_BH = np.kron(sigma_z, identity)  # Pauli-Z for black hole qubit
    sigma_z_Rad = np.kron(identity, sigma_z)  # Pauli-Z for radiation qubit
    sigma_x_BH = np.kron(sigma_x, identity)  # Pauli-X for black hole qubit

    # Time-independent part of the Hamiltonian
    H_0 = (delta / 2) * sigma_z_BH + J * np.dot(sigma_z_BH, sigma_z_Rad)

    # Time-dependent part of the Hamiltonian
    def H_t(t):
        return Omega * np.cos(omega * t) * sigma_x_BH

    # Full Hamiltonian as a function of time
    def H(t):
        return H_0 + H_t(t)

    # Time evolution using matrix exponentiation
    def time_evolve(initial_state, t_max, num_steps):
        dt = t_max / num_steps
        state = initial_state
        for step in range(num_steps):
            t = step * dt
            U = expm(-1j * H(t) * dt)  # Time evolution operator
            state = np.dot(U, state)  # Apply time evolution
        return state

    # Define the initial state (black hole + radiation qubits in |00>)
    initial_state = np.array([1, 0, 0, 0])  # |00> in computational basis

    # Simulate time evolution
    final_state = time_evolve(initial_state, t_max, num_steps)

    # Print final state
    print("Final state vector:")
    print(final_state)

def warp_information_circuit():
    """
    Creates a 3-qubit circuit to test entanglement survival under spacetime distortions.
    """
    qc = QuantumCircuit(3)  # Three qubits

    # Step 1: Initialize entanglement
    qc.h(0)  # Superposition for the "center"
    qc.cx(0, 1)  # Entangle "center" with "boundary"
    qc.cx(1, 2)  # Extend entanglement to "outside"

    # Step 2: Apply spacetime distortions
    qc.rz(np.pi / 4, 0)  # Phase shift on the "center"
    qc.cp(np.pi / 3, 0, 1)  # Controlled phase between "center" and "boundary"
    qc.cp(np.pi / 6, 1, 2)  # Controlled phase between "boundary" and "outside"
    qc.rx(np.pi / 8, 2)  # Additional distortion on "outside"

    return qc

def run_warp_simulation(qc):
    """
    Runs the warp simulation circuit and displays results.
    """
    simulator = Aer.get_backend('aer_simulator')
    transpiled = transpile(qc, simulator)
    result = simulator.run(transpiled, shots=2048).result()
    counts = result.get_counts()

    # Display results as a histogram
    plot_histogram(counts)
    plt.title("Warp Hologram Simulation Results")
    plt.show()

    return counts

def analyze_results(results):
    """
    Analyzes and visualizes state probabilities from the results.
    """
    total_shots = sum(results.values())
    probabilities = {state: count / total_shots for state, count in results.items()}

    # Print probabilities
    print("\nState Probabilities:")
    for state, prob in probabilities.items():
        print(f"State {state}: {prob:.4f}")

    # Visualize probabilities
    states = list(probabilities.keys())
    probs = list(probabilities.values())
    plt.bar(states, probs)
    plt.xlabel('States')
    plt.ylabel('Probability')
    plt.title('State Probabilities from Dynamic Warp Circuit')
    plt.show()

    return probabilities

def calculate_entropies(state):
    """
    Calculates entropies for subsystems in the quantum state.
    """
    subsystem_0 = partial_trace(state, [1, 2])  # Isolate Qubit 0
    subsystem_1 = partial_trace(state, [0, 2])  # Isolate Qubit 1
    subsystem_2 = partial_trace(state, [0, 1])  # Isolate Qubit 2

    entropy_0 = entropy(subsystem_0, base=2)
    entropy_1 = entropy(subsystem_1, base=2)
    entropy_2 = entropy(subsystem_2, base=2)

    return {"Qubit 0": entropy_0, "Qubit 1": entropy_1, "Qubit 2": entropy_2}

def run_quantum_simulation(qc):
    """
    Runs the quantum circuit using a statevector simulator (no measurements allowed).
    """
    # Create a copy of the circuit without measurements
    qc_no_measure = qc.remove_final_measurements(inplace=False)

    # Simulate the quantum circuit using Statevector
    state = Statevector.from_instruction(qc_no_measure)

    # Analyze the statevector
    probabilities = state.probabilities_dict()
    print("\nState Probabilities (Quantum):")
    for state, prob in probabilities.items():
        print(f"State {state}: {prob:.4f}")

    return probabilities

def enhanced_time_evolution(t_steps, shots=2048):
    """
    Enhance time-dependent transformations and holographic interactions.
    """
    qc = QuantumCircuit(3, 3)

    # Initialize entanglement
    qc.h(0)  # Black Hole qubit in superposition
    qc.cx(0, 1)  # Entangle Black Hole and Radiation
    qc.cx(1, 2)  # Extend entanglement to External Environment

    # Time-evolution with holographic interactions
    for t in range(1, t_steps + 1):
        angle_timeline = np.pi * t / 10
        angle_holographic = np.pi * t / 15

        qc.rz(angle_timeline, 0)  # Timeline distortion (Black Hole qubit)
        qc.rx(angle_holographic, 1)  # Holographic interaction (Radiation qubit)
        qc.cp(np.pi * t / 20, 1, 2)  # Controlled phase interaction

    # Statevector analysis (no measurement gates)
    qc_no_measurements = qc.copy()

    # Add measurements for final probabilities
    qc.measure([0, 1, 2], [0, 1, 2])

    # Analyze statevector
    state = Statevector.from_instruction(qc_no_measurements)
    entropies = calculate_entropies(state)

    print("\nStatevector:")
    print(state)
    print("Subsystem Entropies:", entropies)

    # Run the circuit on a simulator
    results = run_warp_simulation(qc)
    print("\nMeasurement Results:", results)

    return results, entropies

# Keep in mind the run experiment has a target_state
    
def run_experiment(backend_type, target_state="111", t_steps=5, shots=2048):
    """
    Run a mirrored state experiment with time-dependent holographic operations.
    """
    # Select backend
    if backend_type == "simulator":
        backend = Aer.get_backend('aer_simulator')
        print("\nRunning on AerSimulator...")
    elif backend_type == "quantum":
        service = QiskitRuntimeService(channel="ibm_quantum")  # Load IBM Quantum account
        filters = lambda b: b.configuration().n_qubits >= 2 and b.status().operational
        backend = least_busy_backend(service, filters)
        if backend is None:
            raise ValueError("No backends are appear to be working at the minute...")
        print(f"\nRunning on quantum backend: {backend.name}...")
    else:
        raise ValueError("Invalid backend_type. Use 'simulator' or 'quantum'.")

    n_qubits = len(target_state)
    qc = QuantumCircuit(n_qubits, n_qubits)  # Create the quantum circuit

    # Step 1: Time-dependent holographic operations
    for t in range(1, t_steps + 1):
        angle_timeline = np.pi * t / 10
        angle_holographic = np.pi * t / 15

        qc.rz(angle_timeline, 0)  # Timeline distortion on Black Hole qubit
        qc.rx(angle_holographic, 1)  # Holographic interaction on Radiation qubit
        qc.cx(1, 2)  # Correlate Radiation with Environment

    # Step 2: Add measurement
    qc.measure(range(n_qubits), range(n_qubits))
    
    # Transpile and run
    transpiled_qc = transpile(qc, backend=backend, optimization_level=3)
    print(f"Transpiled Circuit Depth: {transpiled_qc.depth()}")

    if backend_type == "simulator":
        result = backend.run(transpiled_qc, shots=shots).result()
        counts = result.get_counts()
        print("Results: ", counts)
    elif backend_type == "quantum":
        with Session(backend=backend) as session:
        # Use Qiskit Runtime Sampler for quantum backend
            sampler = Sampler()
            job = sampler.run([transpiled_qc])
            result = job.result()
            print("Results: ", result)
                # Extract counts from the nested structure
            try:
            # Navigate to the `BitArray` and extract counts
                pub_result = result._pub_results[0]  # Access the first `SamplerPubResult`
                data_bin = pub_result.data  # Access `DataBin`
                bit_array = data_bin.c  # Access `BitArray`

                counts = bit_array.to_dict()
                print("Results: ", counts)

            except Exception as e:
                print(e)

def create_holographic_timeline_circuit(target_state="11"):
    """
    Creates a quantum circuit to simulate holographic timeline interactions with a targeted state.
    Args:
        target_state (str): The target state to encode for the Black Hole and Radiation qubits.
    Returns:
        QuantumCircuit: The quantum circuit representing the timeline interaction.
    """
    n_qubits = len(target_state)
    qc = QuantumCircuit(n_qubits, n_qubits)  # Create circuit with classical bits for measurement

    # Encode target state
    for i, bit in enumerate(target_state):
        if bit == '1':
            qc.x(i)  # Flip the qubit

    # Apply holographic timeline interaction
    qc.h(0)  # Superposition for Black Hole qubit
    qc.cx(0, 1)  # Entangle Black Hole with Radiation
    qc.rz(np.pi / 3, 0)  # Simulate timeline distortion on Black Hole
    qc.rx(np.pi / 4, 1)  # Simulate holographic interaction on Radiation

    qc.measure(range(n_qubits), range(n_qubits))

    return qc

def run_holographic_timeline_circuit(qc, backend_type="simulator", shots=1024):
    """
    Runs the holographic timeline circuit and visualizes results.
    Args:
        qc (QuantumCircuit): The quantum circuit to execute.
        backend_type (str): The type of backend ("simulator" or "quantum").
        shots (int): The number of shots for the execution.
    Returns:
        dict: The processed results of the execution.
    """
    # Select backend
    if backend_type == "simulator":
        backend = Aer.get_backend('aer_simulator')
        print("\nRunning on AerSimulator...")
    elif backend_type == "quantum":
        service = QiskitRuntimeService(channel="ibm_quantum")  # Load IBM Quantum account
        filters = lambda b: b.configuration().n_qubits >= 2 and b.status().operational
        backend = least_busy_backend(service, filters)
        if backend is None:
            raise ValueError("No backends are appear to be working at the minute...")
        print(f"\nRunning on quantum backend: {backend.name}...")
    else:
        raise ValueError("Invalid backend_type. Use 'simulator' or 'quantum'.")

    # Transpile and execute circuit
    transpiled_circuit = transpile(qc, backend)
    job = backend.run(transpiled_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Visualize results
    plot_histogram(counts)
    plt.title("Holographic Timeline Interaction Results")
    plt.show()

    # Analyze statevector
    state = Statevector.from_instruction(qc)
    entropies = analyze_holographic_subsystem_entropy_(state)

    print("Subsystem Entropies:", entropies)
    return counts, entropies

def analyze_holographic_subsystem_entropy(statevector):
    """
    Analyzes the entropy of subsystems (Black Hole and Radiation).
    """
    black_hole_state = partial_trace(statevector, [1])  # Trace out Radiation
    radiation_state = partial_trace(statevector, [0])  # Trace out Black Hole
    bh_entropy = entropy(black_hole_state, base=2)
    rad_entropy = entropy(radiation_state, base=2)
    return {
        "black_hole_entropy": bh_entropy,
        "radiation_entropy": rad_entropy
    }


def analyze_subsystem_entropy(statevector):
    """
    Analyzes the entropy of subsystems (Black Hole and Radiation).
    Args:
        statevector (Statevector): The quantum statevector to analyze.
    Returns:
        dict: Entropy values for Black Hole and Radiation subsystems.
    """
    black_hole_state = partial_trace(statevector, [1])  # Trace out Radiation
    radiation_state = partial_trace(statevector, [0])  # Trace out Black Hole
    bh_entropy = entropy(black_hole_state, base=2)
    rad_entropy = entropy(radiation_state, base=2)
    return {
        "black_hole_entropy": bh_entropy,
        "radiation_entropy": rad_entropy
    }


# The next few functions are for feedbacks, which often do not work well as opposed to setting a target_state in another function

#########################################################################################################################################################################################################

# Holographic Feedback Loop Core Framework
def holographic_feedback_loop(target_state="11", adjust_factor=0.1, shots=1024, backend_type="quantum"):
    """
    Simulates a holographic feedback loop to adjust sensory perceptions or outcomes.
    Args:
        target_state (str): The state to align or adjust perceptions towards.
        adjust_factor (float): The adjustment factor influencing the feedback loop.
        shots (int): The number of shots for the simulation.
    Returns:
        dict: Results and analysis of the feedback loop.
    """
    # Select backend
    if backend_type == "simulator":
        backend = Aer.get_backend('aer_simulator')
        print("\nRunning on AerSimulator...")
    elif backend_type == "quantum":
        service = QiskitRuntimeService(channel="ibm_quantum")  # Load IBM Quantum account
        filters = lambda b: b.configuration().n_qubits >= 2 and b.status().operational
        backend = least_busy_backend(service, filters)
        if backend is None:
            raise ValueError("No backends are appear to be working at the minute...")
        print(f"\nRunning on quantum backend: {backend.name}...")
    else:
        raise ValueError("Invalid backend_type. Use 'simulator' or 'quantum'.")


    n_qubits = len(target_state)
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Encode target state
    for i, bit in enumerate(target_state):
        if bit == '1':
            qc.x(i)

    # Holographic adjustments
    qc.h(0)  # Superposition for Black Hole qubit
    qc.cx(0, 1)  # Entangle Black Hole with Radiation
    qc.rz(np.pi * adjust_factor, 0)  # Simulate timeline distortion on Black Hole
    qc.rx(np.pi * adjust_factor, 1)  # Simulate holographic interaction on Radiation

    # Add measurement
    qc.measure(range(n_qubits), range(n_qubits))

    # Run simulation
    backend = Aer.get_backend('aer_simulator')
    transpiled_circuit = transpile(qc, backend)
    job = backend.run(transpiled_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Visualization
    plot_histogram(counts)

    return {
        "counts": counts,
        "target_state": target_state,
        "adjust_factor": adjust_factor
    }

# Safeguard Mechanism
def stop_feedback_loop():
    """Stops the feedback loop safely."""
    print("Holographic feedback loop stopped.")

def simulate_feedback_loop(target_state="11", adjust_factor=0.1, iterations=10, shots=1024):
    """
    Simulates the holographic feedback loop over multiple iterations.

    Args:
        target_state (str): The state to align probabilities toward.
        adjust_factor (float): The factor influencing the feedback adjustments.
        iterations (int): Number of iterations to simulate.
        shots (int): Number of shots per simulation.

    Returns:
        dict: Evolution of state probabilities over iterations.
    """
    n_qubits = len(target_state)
    backend = Aer.get_backend('aer_simulator')

    # Initial probabilities
    probabilities = {state: 0 for state in [f"{i:0{n_qubits}b}" for i in range(2**n_qubits)]}
    probabilities[target_state] = 1 / len(probabilities)  # Start slightly biased toward target

    evolution = []

    for iteration in range(iterations):
        qc = QuantumCircuit(n_qubits, n_qubits)

        # Encode target state
        for i, bit in enumerate(target_state):
            if bit == '1':
                qc.x(i)

        # Apply holographic adjustments
        qc.h(0)  # Black Hole qubit
        qc.cx(0, 1)  # Entangle Black Hole and Radiation
        qc.rz(np.pi * adjust_factor, 0)
        qc.rx(np.pi * adjust_factor, 1)

        # Measure
        qc.measure(range(n_qubits), range(n_qubits))

        # Simulate
        transpiled_circuit = transpile(qc, backend)
        job = backend.run(transpiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Update probabilities
        total_shots = sum(counts.values())
        for state, count in counts.items():
            probabilities[state] = (1 - adjust_factor) * probabilities.get(state, 0) + adjust_factor * (count / total_shots)

        evolution.append(probabilities.copy())

    return evolution

def plot_evolution(evolution, target_state):
    """Plots the evolution of state probabilities over iterations."""
    states = list(evolution[0].keys())
    iterations = len(evolution)

    # Prepare data for plotting
    data = {state: [step[state] for step in evolution] for state in states}

    plt.figure(figsize=(12, 6))
    for state, values in data.items():
        plt.plot(range(iterations), values, label=f"State {state}", linestyle='--' if state != target_state else '-', linewidth=2)

    plt.title("Holographic Feedback Loop: Probability Evolution")
    plt.xlabel("Iterations")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.show()


def run_with_fidelity(circuit, backend_type="simulator", shots=1024):
    """
    Runs the circuit and evaluates the fidelity of the result.
    Args:
        circuit (QuantumCircuit): The circuit to execute.
        backend_type (str): Backend type ("simulator" or "quantum").
        shots (int): Number of shots.
    Returns:
        tuple: (results, fidelity)
    """
    backend = None
    if backend_type == "simulator":
        from qiskit.providers.aer import AerSimulator
        backend = AerSimulator()
    elif backend_type == "quantum":
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        backend = service.least_busy(3)  # Example: 3-qubit systems

    # Execute the circuit
    job = backend.run(circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Fidelity calculation (placeholder for your specific approach)
    fidelity = calculate_fidelity(circuit, counts)  # Another function we need to implement.

    return counts, fidelity

def add_noise_to_circuit(circuit):
    """
    Adds a noise model to the circuit for testing error correction.
    Args:
        circuit (QuantumCircuit): The circuit to modify.
    Returns:
        QuantumCircuit: Circuit with noise applied.
    """
    from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
    from qiskit import QuantumCircuit

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ['rz', 'rx'])
    circuit_with_noise = circuit.copy()  # Preserve the original

    return circuit_with_noise

def mitigate_errors(results):
    """
    Mitigates errors from the results.
    Args:
        results (dict): Raw results from the execution.
    Returns:
        dict: Corrected results.
    """
    from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
    from qiskit.ignis.mitigation.measurement import MeasurementFilter

    # Example: Use measurement error mitigation
    meas_fitter = CompleteMeasFitter(results, ['0', '1'])  # Simple fitter for demo
    meas_filter = meas_fitter.filter
    mitigated_counts = meas_filter.apply(results)

    return mitigated_counts

def calculate_fidelity(target_state, observed_counts, total_shots):
    """
    Calculate the fidelity between the observed results and the target state.

    Args:
        target_state (str): The target quantum state (e.g., '101').
        observed_counts (dict): Measurement results as a dictionary of {state: count}.
        total_shots (int): Total number of shots in the experiment.

    Returns:
        float: Fidelity value.
    """
    target_prob = observed_counts.get(target_state, 0) / total_shots
    return target_prob  # Simpler fidelity based on target-state probability.

def run_with_error_correction(circuit, backend, target_state, shots=1024):
    """
    Execute a quantum circuit with error correction integrated.

    Args:
        circuit (QuantumCircuit): The quantum circuit to run.
        backend (Backend): The backend to execute on.
        target_state (str): The desired outcome state (e.g., '101').
        shots (int): Number of shots for the execution.

    Returns:
        dict: Corrected results and fidelity metrics.
    """
    print("\nRunning initial circuit...")
    initial_results = run_circuit_with_feedback(circuit, backend, shots)

    fidelity = calculate_fidelity(
        target_state=target_state,
        observed_counts=initial_results['counts'],
        total_shots=shots
    )

    print(f"Initial Fidelity: {fidelity:.4f}")

    if fidelity < 0.9:  # Adjust threshold as needed
        print("Low fidelity detected. Applying corrections...")
        corrected_circuit = prepare_state(circuit, target_state)
        corrected_results = run_circuit_with_feedback(corrected_circuit, backend, shots)

        corrected_fidelity = calculate_fidelity(
            target_state=target_state,
            observed_counts=corrected_results['counts'],
            total_shots=shots
        )

        return {
            "initial_results": initial_results,
            "initial_fidelity": fidelity,
            "corrected_results": corrected_results,
            "corrected_fidelity": corrected_fidelity,
        }
    else:
        print("Fidelity within acceptable range. No correction needed.")
        return {"results": initial_results, "fidelity": fidelity}

def create_feedback_ready_circuit(n_qubits):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure_all()  # Measurement for feedback
    return qc

def run_circuit_with_feedback_fidelity(circuit, backend, target_state, shots=1024, max_iterations=10):
    """
    Runs a quantum circuit with temporal feedback for error correction or state optimization.

    Args:
        circuit (QuantumCircuit): The quantum circuit to run.
        backend (Backend): Quantum backend to execute the circuit.
        target_state (str): The desired quantum state (e.g., '101').
        shots (int): Number of shots for measurement.
        max_iterations (int): Maximum number of feedback iterations.

    Returns:
        dict: Final counts and fidelity information.
    """

    print(f"Starting feedback loop for target state: {target_state}")
    fidelity_history = []

    # Transpile the circuit for the backend
    transpiled_circuit = transpile(circuit, backend)
    for i in range(max_iterations):
        print(f"Iteration {i + 1}/{max_iterations}...")

        # Execute the circuit
        job = backend.run(transpiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Calculate fidelity
        fidelity = calculate_fidelity(target_state, counts, shots)
        fidelity_history.append(fidelity)

        print(f"Iteration {i + 1}: Fidelity = {fidelity:.4f}")
        if fidelity >= 0.95:  # Threshold for acceptable fidelity
            print("Target fidelity achieved.")
            break

        # Apply feedback: Adjust gates or parameters dynamically
        transpiled_circuit = modify_circuit_with_reduced_influence(transpiled_circuit, counts, target_state=target_state)

    return {
        "counts": counts,
        "fidelity_history": fidelity_history,
        "final_fidelity": fidelity,
    }

def modify_circuit_with_reduced_influence(circuit, counts, target_state, scaling_factor=0.5):
    """Modify the circuit based on measurement feedback with reduced influence."""
    if not counts:
        print("No measurement data available for feedback.")
        return circuit

    # Identify the most likely state
    dominant_state = max(counts, key=counts.get)
    print(f"Current dominant state: {dominant_state}")

    # Compare dominant state with target state
    for idx, (target_bit, dominant_bit) in enumerate(zip(reversed(target_state), reversed(dominant_state))):
        if target_bit != dominant_bit:
            # Apply corrective gates with reduced influence
            if scaling_factor > 0.5:
                circuit.x(idx)  # Full influence
            else:
                circuit.h(idx)  # Partial influence
            print(f"Modified qubit {idx} to correct state.")

    print("Circuit updated with reduced influence.")
    return circuit

def stop_if_measured_1(result):
    return '1' in result and result['1'] > 500  # Example: Stop if '1' is measured more than 500 times

def stop_if_measured_0(result):
    return '0' in result and result['0'] > 500  # Example: Stop if '1' is measured more than 500 times

def run_circuit_with_feedback(base_circuit_func, backend, shots=1024, max_iterations=10, stop_condition=None):
    """
    Iterative quantum experiment function that dynamically adjusts and increases iteration count automatically.
    
    Parameters:
        base_circuit_func (function): Function that generates the base quantum circuit.
        backend (Backend): Quantum backend for execution.
        shots (int): Number of shots per experiment run.
        max_iterations (int): Maximum number of iterations to run.
    
    Returns:
        list: List of results from each experiment iteration.
    """
    results = []
    iteration = 1
    
    while iteration <= max_iterations:
        print(f"Running iteration {iteration}...")
        
        # Generate the circuit
        qc = base_circuit_func()
        
        # Run experiment and extract results
        result = run_and_extract_counts(qc, backend, shots)
        results.append(result)

        if stop_condition and stop_condition(result):
            print(f"Stopping early at iteration {iteration} due to stop condition.")
            break
        
        # Adaptive increment based on feedback
        iteration += 1  # Automatically increases iteration count
    
    print("All iterations complete.")
    return results

def modify_circuit_based_on_feedback(circuit, counts):
    """Modify the circuit based on measurement feedback to improve fidelity."""
    if not counts:
        print("No measurement data available for feedback.")
        return circuit

    target_state = max(counts, key=counts.get)
    print(f"Targeting state: {target_state} for amplification.")

    for idx, bit in enumerate(reversed(target_state)):
        if bit == '1':
            circuit.x(idx)
        circuit.h(idx)

    print("Circuit modified based on feedback.")
    return circuit

# This did not work well despite adjustments

def probabilistic_adjustment(circuit: QuantumCircuit, target_state: str, current_state: str, adjustment_factor: float = 0.1):
    """
    Adjust the circuit probabilistically to move closer to the target state.

    Args:
        circuit (QuantumCircuit): The quantum circuit to modify.
        target_state (str): The desired target state (e.g., '011').
        current_state (str): The current dominant state from measurements.
        adjustment_factor (float): Probability of applying an adjustment (default: 0.1).

    Returns:
        QuantumCircuit: The updated quantum circuit.
    """
    n_qubits = len(target_state)

    for idx in range(n_qubits):
        if target_state[idx] != current_state[idx]:
            # Apply probabilistic adjustment
            if random.random() < adjustment_factor:
                if target_state[idx] == '1':
                    circuit.x(idx)  # Flip qubit to 1 if target state demands it
                else:
                    # Add a Hadamard gate to create a superposition toward 0
                    circuit.h(idx)

    return circuit

def black_hole_simulation(num_qubits=17, num_charge_cycles=5, spin_cycles=3, injection_strength=np.pi/4):
    """
    Simulates black hole analog formation through charge injections and spin cycles.

    Parameters:
    - num_qubits (int): Number of qubits in the circuit.
    - num_charge_cycles (int): How many prolonged charge injection cycles to perform.
    - spin_cycles (int): Number of rotational (spin) cycles to apply.
    - injection_strength (float): Rotation angle for charge injection strength.

    Returns:
    - entropy_list (list): Von Neumann entropy after each cycle.
    - fidelity_list (list): Fidelity against the maximally mixed state.
    """

    # Initialize quantum register and circuit
    qreg = QuantumRegister(num_qubits, 'q')
    qc = QuantumCircuit(qreg)

    # Function for prolonged charge injection (entangling each qubit with a central reference qubit)
    def prolonged_charge_injection():
        for i in range(1, num_qubits):
            qc.ry(injection_strength, qreg[i])
            qc.cx(qreg[0], qreg[i])  # Central qubit (q0) acts as the charge source

    # Function for spin cycle (rotational entanglement pattern)
    def spin_cycle():
        for i in range(num_qubits - 1):
            qc.swap(qreg[i], qreg[i+1])  # Simulates angular momentum via qubit rotation

    # Measurement function for entropy and fidelity
    def measure_entropy_fidelity():
        backend = Aer.get_backend('statevector_simulator')
        result = backend.run(qc, shots=1).result()
        sv = result.get_statevector()

        # Convert to density matrix and trace out a subset of qubits for partial trace
        density_matrix = DensityMatrix(sv)
        reduced_state = partial_trace(density_matrix, list(range(1, num_qubits)))

        # Calculate von Neumann entropy and fidelity
        ent = entropy(reduced_state)
        max_mixed = DensityMatrix(np.identity(2) / 2)  # Single-qubit maximally mixed state
        fid = state_fidelity(reduced_state, max_mixed)
        return ent, fid

    # Lists to store results
    entropy_list = []
    fidelity_list = []

    # Simulation: charge injections + spin cycles
    for charge_cycle in range(num_charge_cycles):
        prolonged_charge_injection()
        ent, fid = measure_entropy_fidelity()
        entropy_list.append(ent)
        fidelity_list.append(fid)

        for spin_cycle_index in range(spin_cycles):
            spin_cycle()
            ent, fid = measure_entropy_fidelity()
            entropy_list.append(ent)
            fidelity_list.append(fid)

    return entropy_list, fidelity_list

def information_paradox_test(num_qubits=10, injection_strength=np.pi/2, retrieval_cycles=5):
    """
    Tests the black hole information paradox by injecting known information and attempting retrieval.
    """
    qreg = QuantumRegister(num_qubits, 'q')
    qc = QuantumCircuit(qreg)

    # Encode known information (|+> state)
    qc.h(qreg[0])

    # Scramble with charge injections and spin cycles
    for i in range(1, num_qubits):
        qc.ry(injection_strength, qreg[i])
        qc.cx(qreg[0], qreg[i])
    for _ in range(retrieval_cycles):
        for i in range(num_qubits - 1):
            qc.swap(qreg[i], qreg[i + 1])

    # Attempt retrieval (reverse operations)
    for i in reversed(range(1, num_qubits)):
        qc.cx(qreg[0], qreg[i])
        qc.ry(-injection_strength, qreg[i])
    qc.h(qreg[0])

    backend = Aer.get_backend('statevector_simulator')
    sv = backend.run(qc).result().get_statevector()
    final_state = partial_trace(DensityMatrix(sv), list(range(1, num_qubits)))
    original_state = DensityMatrix.from_label('+')

    # Ensure matching dimensions for fidelity calculation
    if final_state.dim != original_state.dim:
        final_state = final_state.expand(DensityMatrix(np.eye(original_state.dim[0])))

    retrieval_fidelity = state_fidelity(final_state, original_state)
    return retrieval_fidelity

def hawking_radiation_recovery(num_qubits=10, injection_strength=np.pi/2, radiation_qubits=2, retrieval_cycles=5):
    """
    Simulates Hawking radiation recovery:
    - Encodes information into a qubit.
    - Scrambles via charge injections and spin cycles.
    - Extracts radiation qubits and attempts information recovery.
    """
    qreg = QuantumRegister(num_qubits, 'q')
    qc = QuantumCircuit(qreg)

    qc.h(qreg[0])  # Encode known information (|+> state)

    for i in range(1, num_qubits):
        qc.ry(injection_strength, qreg[i])
        qc.cx(qreg[0], qreg[i])
    for _ in range(retrieval_cycles):
        for i in range(num_qubits - 1):
            qc.swap(qreg[i], qreg[i + 1])

    backend = Aer.get_backend('statevector_simulator')
    sv = backend.run(qc).result().get_statevector()

    try:
        radiation_indices = list(range(num_qubits - radiation_qubits, num_qubits))
        # Keep only radiation qubits and trace out the rest
        radiation_state = partial_trace(DensityMatrix(sv), [i for i in range(num_qubits) if i not in radiation_indices])

        original_state = DensityMatrix.from_label('+')

        # Ensure compatible dimensions for fidelity calculation
        if radiation_state.dim[0] != original_state.dim[0]:
            size_diff = int(np.log2(original_state.dim[0] / radiation_state.dim[0]))
            for _ in range(size_diff):
                radiation_state = radiation_state.tensor(DensityMatrix(np.eye(2) / 2))

        recovery_fidelity = state_fidelity(radiation_state, original_state)
    except Exception as e:
        print(f"Error in hawking_radiation_recovery: {e}")
        recovery_fidelity = None

    return recovery_fidelity

def hawking_radiation_with_sequential_entangling(num_qubits=10, radiation_qubits=3, entangling_cycles=5, injection_strength=np.pi/2):
    """
    Models Hawking radiation with sequential entangling and fixes dimension mismatch issues.
    Reduces radiation state to a single qubit for fidelity comparison.
    """
    qreg = QuantumRegister(num_qubits, 'q')
    qc = QuantumCircuit(qreg, name="SecureHawkingRadiation")

    qc.h(qreg[0])  # Encode information (|+> state)

    for i in range(1, radiation_qubits + 1):
        qc.ry(injection_strength, qreg[i])
        qc.cx(qreg[0], qreg[i])
        qc.barrier()

    for _ in range(entangling_cycles):
        for i in range(radiation_qubits + 1, num_qubits - 1):
            qc.cx(qreg[i], qreg[i + 1])
            qc.ry(injection_strength / 2, qreg[i + 1])

    backend = Aer.get_backend('statevector_simulator')
    sv = backend.run(qc).result().get_statevector()

    try:
        radiation_indices = list(range(1, radiation_qubits + 1))
        radiation_state = partial_trace(DensityMatrix(sv), [i for i in range(num_qubits) if i not in radiation_indices])
        original_state = DensityMatrix.from_label('+')

        # Reduce radiation_state to a single qubit for comparison
        if radiation_state.num_qubits > 1:
            qubits_to_trace = list(range(1, radiation_state.num_qubits))
            radiation_state = partial_trace(radiation_state, qubits_to_trace)

        recovery_fidelity = state_fidelity(radiation_state, original_state)
    except Exception as e:
        print(f"Error in hawking_radiation_with_sequential_entangling: {e}")
        recovery_fidelity = None

    return recovery_fidelity

def create_gaussian_pulse(amplitude, sigma, duration, name="gaussian_pulse"):
    return Gaussian(duration=duration, amp=amplitude, sigma=sigma, name=name)

# --- PULSE-LEVEL BLACK HOLE SIMULATION SKELETON ---
def black_hole_pulse_simulation():
    """
    Creates a pulse schedule representing charge injection cycles and spin cycles.
    This maps the black hole simulation to hardware-level AWG controls.
    """
    schedule_list = []

    # Define drive channels for qubits
    drive_channels = [DriveChannel(qubit) for qubit in range(NUM_QUBITS)]

    # Create injection pulse (analogous to prolonged charge injection)
    injection_pulse = create_gaussian_pulse(
        amplitude=PULSE_AMPLITUDE, sigma=PULSE_SIGMA, duration=PULSE_DURATION, name="charge_injection"
    )

    # Create spin pulse (analogous to spin cycle entangling operations)
    spin_pulse = create_gaussian_pulse(
        amplitude=PULSE_AMPLITUDE, sigma=PULSE_SIGMA, duration=PULSE_DURATION, name="spin_cycle"
    )

    # --- PROLONGED CHARGE INJECTION ---
    injection_schedule = Schedule(name="prolonged_charge_injection")
    for qubit in range(1, NUM_QUBITS):
        injection_schedule |= Play(injection_pulse, drive_channels[qubit])
    schedule_list.append(injection_schedule)

    # --- SPIN CYCLE SIMULATION ---
    spin_schedule = Schedule(name="spin_cycle")
    for qubit in range(NUM_QUBITS - 1):
        spin_schedule |= Play(spin_pulse, drive_channels[qubit])
    schedule_list.append(spin_schedule)

    # --- EVENT HORIZON ANALOG ---
    event_horizon_schedule = Schedule(name="event_horizon_scramble")
    for qubit in range(NUM_QUBITS):
        event_horizon_schedule |= Play(
            create_gaussian_pulse(PULSE_AMPLITUDE, PULSE_SIGMA, PULSE_DURATION, name=f"scramble_{qubit}"),
            drive_channels[qubit]
        )
    schedule_list.append(event_horizon_schedule)

    # --- INFORMATION RETRIEVAL PHASE ---
    retrieval_schedule = Schedule(name="hawking_radiation_recovery")
    retrieval_schedule |= Play(
        create_gaussian_pulse(PULSE_AMPLITUDE, PULSE_SIGMA, PULSE_DURATION, name="hawking_retrieval"),
        drive_channels[0]
    )
    schedule_list.append(retrieval_schedule)

    return schedule_list

def quantum_scramble_encrypt(message_bits: str, key_bits: str, shots: int = 1024):
    """
    Encrypts a message using quantum scrambling with black hole dynamics.

    Parameters:
    - message_bits (str): Binary string representing the message.
    - key_bits (str): Binary string representing the private key.
    - shots (int): Number of measurement shots (default: 1024).

    Returns:
    - counts (dict): Measurement outcomes from the encrypted quantum circuit.
    - scrambling_unitary (Instruction): Applied random unitary for decoding.
    """
    n_qubits = len(message_bits)
    qr = QuantumRegister(n_qubits, name="q")
    cr = ClassicalRegister(n_qubits, name="c")
    circuit = QuantumCircuit(qr, cr)

    # 1️⃣ Encode Message
    for i, bit in enumerate(message_bits):
        if bit == '1':
            circuit.x(qr[i])

    # 2️⃣ Entangle with Private Key
    for i, bit in enumerate(key_bits):
        if bit == '1':
            circuit.h(qr[i])
            circuit.cx(qr[i], qr[(i + 1) % n_qubits])

    # 3️⃣ Apply Dynamic Scrambling (Random Unitary)
    scrambling_unitary = random_unitary(2**n_qubits).to_instruction()
    circuit.append(scrambling_unitary, qr[:])

    # 4️⃣ Entropy Amplification (Non-reversible Random Rotations for External Observers)
    for i in range(n_qubits):
        random_angle_rz = np.pi / np.random.randint(1, 10)
        random_angle_rx = np.pi / np.random.randint(1, 10)
        circuit.h(qr[i])
        circuit.rz(random_angle_rz, qr[i])
        circuit.rx(random_angle_rx, qr[i])

    # 5️⃣ Measurement
    circuit.barrier()
    circuit.measure(qr, cr)

    backend = Aer.get_backend('qasm_simulator')
    job = backend.run(circuit, shots=shots)
    counts = job.result().get_counts()

    return counts, scrambling_unitary


# 🔑 Decoding Function
def quantum_scramble_decrypt(scrambling_unitary, key_bits: str, encrypted_bits: str, shots: int = 1024):
    """
    Decrypts the scrambled quantum state to retrieve the original message without relying on stored rotation angles.

    Parameters:
    - scrambling_unitary (Instruction): The unitary used during encryption.
    - key_bits (str): The private key used in the encryption process.
    - encrypted_bits (str): Binary string representing the measured scrambled state.
    - shots (int): Number of measurement shots (default: 1024).

    Returns:
    - counts (dict): Measurement outcomes after decryption.
    """
    n_qubits = len(encrypted_bits)
    qr = QuantumRegister(n_qubits, name="q")
    cr = ClassicalRegister(n_qubits, name="c")
    circuit = QuantumCircuit(qr, cr)

    # 1️⃣ Re-encode Measured Scrambled State
    for i, bit in enumerate(encrypted_bits):
        if bit == '1':
            circuit.x(qr[i])

    # 2️⃣ Approximate Entropy Reversal (Structured but Unpredictable for Attackers)
    # Instead of stored angles, apply inverse rotations using a deterministic function (e.g., fixed angle pattern)
    for i in range(n_qubits):
        inverse_angle_rz = -np.pi / (i + 2)  # Deterministic, key-dependent pattern
        inverse_angle_rx = -np.pi / (i + 3)
        circuit.rx(inverse_angle_rx, qr[i])
        circuit.rz(inverse_angle_rz, qr[i])
        circuit.h(qr[i])

    # 3️⃣ Apply Inverse Scrambling Unitary
    circuit.append(scrambling_unitary.inverse(), qr[:])

    # 4️⃣ Reverse Entanglement with Private Key
    for i, bit in enumerate(reversed(key_bits)):
        if bit == '1':
            circuit.cx(qr[i], qr[(i + 1) % n_qubits])
            circuit.h(qr[i])

    # 5️⃣ Measurement
    circuit.barrier()
    circuit.measure(qr, cr)

    backend = Aer.get_backend('qasm_simulator')
    job = backend.run(circuit, shots=shots)
    counts = job.result().get_counts()

    return counts

# 🔍 Branch Signature Mapping Function
def map_branch_signatures(event_description: str, key_bits: str, n_qubits: int = 6):
    """
    Maps a branch signature based on an event description and private key.

    Parameters:
    - event_description (str): Description of the event (e.g., 'identity theft').
    - key_bits (str): Private key to personalize the mapping.
    - n_qubits (int): Number of qubits representing the branch signature.

    Returns:
    - signature_bits (str): Binary signature representing the targeted branch.
    - signature_hash (str): SHA-256 hash of the event and key for verification.
    """
    # Generate a hash-based seed from event and key
    combined = f"{event_description}-{key_bits}".encode()
    signature_hash = hashlib.sha256(combined).hexdigest()

    # Use hash to produce deterministic binary signature
    signature_bits = bin(int(signature_hash, 16))[2:].zfill(n_qubits)[:n_qubits]

    return signature_bits, signature_hash

def black_hole_warp_simulation_0():
    # Step 1: Initialize Quantum Circuit with 6 qubits (representing 3 entangled pairs)
    num_qubits = 6
    qc = QuantumCircuit(num_qubits)
    
    # Step 2: Create maximal entanglement (Bell Pairs)
    for i in range(0, num_qubits, 2):
        qc.h(i)
        qc.cx(i, i+1)
    
    # Step 3: Introduce Charge & Spin Effects (Random Phase Rotations)
    np.random.seed(42)
    for i in range(num_qubits):
        theta = np.random.uniform(0, 2*np.pi)  # Random charge/spin influence
        qc.rz(theta, i)
    
    # Step 4: Scrambling Effect (Black Hole Information Overload)
    for i in range(num_qubits-1):
        qc.cx(i, i+1)
        qc.h(i)
        qc.rz(np.pi/4, i+1)
    
    # Step 5: Apply Holographic Warp (Simulating Spacetime Curvature)
    for i in range(num_qubits):
        qc.sx(i)
        qc.cz(i, (i+1) % num_qubits)
    
    # Step 6: Measure Final State
    backend = Aer.get_backend("statevector_simulator")
    result = backend.run(qc).result()
    final_state = Statevector(result.get_statevector())

    final_density_matrix = DensityMatrix(final_state)

    for i in range(num_qubits):
        partial_traced_state = partial_trace(final_density_matrix, [i])
        print(f"Partial trace for qubit {i}:\n", partial_traced_state)  # Debugging output


    # Compute entropies using the correct method
    entropies = [entropy(partial_trace(final_density_matrix, [i])) for i in range(num_qubits)]

    return qc, entropies

def time_evolution_black_hole(num_qubits=5, time_steps=50, delta_t=0.1):
    """
    Simulates the time evolution of a black hole quantum system
    and computes the Von Neumann entropy at each step.
    """
    # Initialize a maximally entangled state (black hole analogy)
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits - 1):
        qc.h(i)
        qc.cx(i, i+1)
    
    # Convert to density matrix
    backend = Aer.get_backend("statevector_simulator")
    transpiled_qc = transpile(qc, backend)
    state = DensityMatrix.from_instruction(transpiled_qc)
    
    entropies = []
    times = np.arange(0, time_steps * delta_t, delta_t)
    
    for t in times:
        # Apply random unitary evolution to simulate information scrambling
        random_unitary = np.exp(1j * np.random.rand(num_qubits, num_qubits))
        evolved_state = state.evolve(random_unitary)
        
        # Compute Von Neumann entropy for each qubit
        qubit_entropies = [entropy(partial_trace(evolved_state, [i])) for i in range(num_qubits)]
        entropies.append(np.mean(qubit_entropies))
    
    # Plot entropy evolution
    plt.figure(figsize=(8, 5))
    plt.plot(times, entropies, label="Avg. Entropy per Qubit", color='purple')
    plt.xlabel("Time")
    plt.ylabel("Von Neumann Entropy")
    plt.title("Entropy Evolution in Black Hole Simulation")
    plt.legend()
    plt.show()
    
    return times, entropies

def auto_detect_branches(events_list, key_bits: str, detection_threshold: int = 50):
    """
    Auto-detects branches based on a list of event descriptions and a private key.

    Parameters:
    - events_list (list): List of event descriptions to monitor.
    - key_bits (str): Private key for branch mapping.
    - detection_threshold (int): Probability threshold (%) for detection.

    Returns:
    - detected_branches (dict): Event descriptions mapped to their detected branch signatures.
    """
    detected_branches = {}
    for event in events_list:
        detection_chance = random.randint(0, 100)
        if detection_chance >= detection_threshold:
            signature_bits, signature_hash = map_branch_signatures(event, key_bits)
            detected_branches[event] = {
                "signature_bits": signature_bits,
                "signature_hash": signature_hash,
                "detection_confidence": detection_chance
            }
    return detected_branches


def initialize_black_hole_state(qc, num_qubits):
    """Prepares the initial quantum state representing infalling matter."""
    for qubit in range(num_qubits):
        qc.h(qubit)  # Superposition state

def apply_charge_spin_interactions(qc, num_qubits, charge, spin):
    """Encodes charge-spin interactions using Kerr-Newman approximations."""
    for qubit in range(num_qubits - 1):
        theta = np.arctan(charge / (1 + spin))  # Charge-spin ratio for interaction
        qc.rz(theta, qubit)  # Charge-induced phase shift
        qc.cry(2 * np.pi * spin, qubit, qubit + 1)  # Spin entanglement

def simulate_event_horizon(qc, num_qubits):
    """Applies quantum scrambling effects at the black hole horizon."""
    for qubit in range(num_qubits):
        qc.sx(qubit)  # Quantum scrambling at the event horizon
        qc.cz(qubit, (qubit + 1) % num_qubits)  # Non-local entanglement

def compute_von_neumann_entropy(qc, num_qubits):
    """Computes the Von Neumann entropy for each qubit after the simulation."""
    backend = Aer.get_backend('statevector_simulator')
    transpiled_qc = transpile(qc, backend)
    final_state = DensityMatrix.from_instruction(transpiled_qc)

    entropies = []
    for qubit in range(num_qubits):
        reduced_state = partial_trace(final_state, [qubit])  # Corrected
        eigenvalues = np.linalg.eigvalsh(reduced_state.data)
        entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10)).real  # Small offset to avoid log(0)
        entropies.append(entropy)
    
    return entropies

def black_hole_warp_simulation_core(num_qubits=6, charge=0.7, spin=0.85):
    """Runs the full black hole simulation pipeline."""
    qc = QuantumCircuit(num_qubits)
    
    initialize_black_hole_state(qc, num_qubits)
    apply_charge_spin_interactions(qc, num_qubits, charge, spin)
    simulate_event_horizon(qc, num_qubits)
    entropies = compute_von_neumann_entropy(qc, num_qubits)

    return qc, entropies

def apply_shor_encoding(qc, qubit):
    """Encodes a logical qubit using the Shor code (bit-flip and phase-flip protection)."""
    qc.cx(qubit, qubit + 1)
    qc.cx(qubit, qubit + 2)
    qc.h([qubit, qubit + 1, qubit + 2])
    qc.cx(qubit, qubit + 3)
    qc.cx(qubit + 1, qubit + 4)
    qc.cx(qubit + 2, qubit + 5)
    return qc

def detect_and_correct_errors(qc, logical_qubit_start):
    """Detects and corrects bit-flip and phase-flip errors on a logical qubit."""
    for i in range(3):  # Bit-flip correction
        qc.cx(logical_qubit_start + i, logical_qubit_start + 3 + i)
        qc.measure(logical_qubit_start + 3 + i, logical_qubit_start + 3 + i)
        qc.x(logical_qubit_start + i).c_if(logical_qubit_start + 3 + i, 1)
    for i in range(3):  # Phase-flip correction
        qc.h(logical_qubit_start + i)
        qc.cx(logical_qubit_start + i, logical_qubit_start + 3 + i)
        qc.h(logical_qubit_start + i)
    return qc

def encode_charge_preserving(qc, qubits):
    """Encodes logical qubits using charge parity conservation."""
    for i in range(0, len(qubits), 2):
        qc.h(qubits[i])
        qc.cx(qubits[i], qubits[i + 1])  # Entangle charge states
    return qc

def detect_charge_imbalance(qc, ancilla, qubits):
    """Detects charge parity violations using ancilla qubits."""
    for i in range(min(len(ancilla), len(qubits) - 1)):
        qc.cx(qubits[i], ancilla[i])
        qc.cx(qubits[i + 1], ancilla[i])  # If charge parity is broken, ancilla flips
    return qc

def correct_charge_imbalance(qc, ancilla, qubits):
    """Applies corrections if a charge imbalance is detected."""
    for i in range(min(len(ancilla), len(qubits) - 1)):
        qc.cx(ancilla[i], qubits[i])  # Restore charge balance
        qc.cx(ancilla[i], qubits[i + 1])
    return qc

def charge_preserving_qec(num_logical_qubits=2):
    """Runs a charge-preserving quantum error correction simulation."""
    num_physical_qubits = num_logical_qubits * 2  # Each logical qubit = 2 physical qubits
    num_ancilla = max(1, num_logical_qubits - 1)  # Ensure at least one ancilla qubit
    qc = QuantumCircuit(num_physical_qubits + num_ancilla, num_physical_qubits)
    
    # Encode logical qubits with charge preservation
    qc = encode_charge_preserving(qc, list(range(num_physical_qubits)))
    
    # Introduce artificial noise (random bit flip)
    qc.x(1)  # Simulating an error
    
    # Detect and correct charge imbalance
    ancilla_qubits = list(range(num_physical_qubits, num_physical_qubits + num_ancilla))
    qc = detect_charge_imbalance(qc, ancilla_qubits, list(range(num_physical_qubits)))
    qc = correct_charge_imbalance(qc, ancilla_qubits, list(range(num_physical_qubits)))
    
    # Simulate and extract density matrix
    simulator = Aer.get_backend('aer_simulator_density_matrix')
    qc.save_density_matrix()
    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(compiled_circuit).result()
    final_density_matrix = DensityMatrix(result.data(0)['density_matrix'])
    
    # Compute entropies
    entropies = [entropy(partial_trace(final_density_matrix, [i])) for i in range(num_physical_qubits)]
    
    return qc, entropies

def create_noisy_model():
    noise_model = NoiseModel()
    p_depol = 0.01  # Depolarizing probability
    p_damp = 0.02  # Amplitude damping probability
    
    depol_error = depolarizing_error(p_depol, 1)
    amp_damp_error = amplitude_damping_error(p_damp)
    
    # Instead of applying errors multiple times, apply them selectively
    noise_model.add_all_qubit_quantum_error(depol_error, ['u3'])  # Only apply to u3
    noise_model.add_all_qubit_quantum_error(amp_damp_error, ['u3'])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p_depol, 2), ['cx'])

    return noise_model


def charge_preserving_qec_noisy(num_qubits=7):
    q = QuantumRegister(num_qubits, 'q')
    c = ClassicalRegister(num_qubits, 'c')  # Ensure classical bits match quantum bits
    qc = QuantumCircuit(q, c)
    
    # Encoding with charge preservation
    for i in range(num_qubits - 1):
        qc.h(q[i])
        qc.cx(q[i], q[i + 1])

    # Ensure syndrome register only if num_qubits > 2
    if num_qubits > 2:
        syndrome = QuantumRegister(2, 'syndrome')
        qc.add_register(syndrome)

        for i in range(0, num_qubits - 2, 2):
            qc.cx(q[i], q[i + 2])
        
        # Syndrome measurement only when syndrome register exists
        qc.cx(q[0], syndrome[0])
        qc.cx(q[1], syndrome[0])
        qc.cx(q[2], syndrome[1])
        if num_qubits > 3:
            qc.cx(q[3], syndrome[1])

    qc.h(q)
    qc.measure(q, c)
    # Apply noise

    simulator = Aer.get_backend('qasm_simulator')
    job_ideal = simulator.run(qc, shots=4096)  # No noise
    result_ideal = job_ideal.result()
    print("Ideal Measurement:", result_ideal.get_counts())

    noise_model = create_noisy_model()
    simulator = Aer.get_backend('qasm_simulator')
    result = simulator.run(qc, noise_model=noise_model, shots=4096).result()
    
    print(f"Execution Result: {result}")
    
    # Analyze measurement outcomes
    if result.success:
        counts = result.get_counts()
        print(f"Measurement Results for {num_qubits} qubits:", counts)
        return counts
    else:
        print("Execution failed.")
        return None

def text_to_binary(text):
    """Convert text to binary representation."""
    return ''.join(format(ord(char), '08b') for char in text)

def generate_timestamp():
    """Generate a binary timestamp."""
    timestamp = int(time.time())  # Current UNIX timestamp
    return format(timestamp, '064b') 

def record_decision(qc, decision_text):
    """Convert decision to binary, timestamp it, and inject charge."""
    binary_text = text_to_binary(decision_text)
    binary_timestamp = generate_timestamp()
    combined_binary = binary_text + binary_timestamp  # Merge text & time

    print(f"Encoding Decision: {decision_text}")
    print(f"Binary Representation: {combined_binary}")

    qc = inject_charge_holographically(qc, combined_binary)
    return qc

def inject_charge_holographically(qc, binary_pattern):
    """
    Apply charge injection based on binary pattern.
    Each '1' in binary applies a Hadamard gate followed by a phase shift; '0' applies only a Hadamard gate.
    """
    num_qubits = qc.num_qubits
    for i, bit in enumerate(binary_pattern):
        qubit = i % num_qubits  # Wrap around if needed
        qc.h(qubit)  # Place qubit in superposition
        if bit == '1':
            qc.p(np.pi / 4, qubit)  # Inject a phase shift
    return qc

def create_teleportation_circuit():
    qr = QuantumRegister(3, name="q")  # Create a quantum register with 3 qubits
    cr = ClassicalRegister(2, name="c")  # Create a classical register with 2 bits

    qc = QuantumCircuit(qr, cr)  # Now correctly define the circuit with named registers

    # Step 1: Create Bell pair
    qc.h(qr[1])
    qc.cx(qr[1], qr[2])

    # Step 2: Encode charge information into Q0 (Injected charge state)
    qc.h(qr[0])

    # Step 3: Bell measurement on Q0 and Q1
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.measure([qr[0], qr[1]], [cr[0], cr[1]])

    # Step 4: Conditional operations on Q2 based on measurement results
    qc.x(qr[2]).c_if(cr[0], 1)  # Apply X if cr[0] == 1
    qc.z(qr[2]).c_if(cr[1], 1)  # Apply Z if cr[1] == 1

    return qc

def apply_classical_corrections(measurement, statevector):
    """
    Manually applies X and Z corrections based on measurement results.
    """
    # Create and convert circuits to Operators
    x_circuit = QuantumCircuit(1)
    x_circuit.x(0)
    x_gate = Operator.from_circuit(x_circuit)

    z_circuit = QuantumCircuit(1)
    z_circuit.z(0)
    z_gate = Operator.from_circuit(z_circuit)

    if measurement[0] == '1':  # If the first classical bit is 1, apply X gate
        statevector = statevector.evolve(x_gate)
    if measurement[1] == '1':  # If the second classical bit is 1, apply Z gate
        statevector = statevector.evolve(z_gate)

    return statevector

def apply_add_clbits(qc, num_qubits=7):
    """Applies charge-preserving quantum error correction to a given quantum circuit."""
    q = QuantumRegister(num_qubits, 'q')
    c = ClassicalRegister(num_qubits, 'c')  # Define classical bits properly
    qec_qc = QuantumCircuit(q, c)

    # Convert the circuit into an instruction while explicitly including classical bits
    qc_instruction = qc.to_instruction()

    # Append the instruction to the new quantum circuit
    qec_qc.append(qc_instruction, q[:qc.num_qubits])  

    return qec_qc, qc_instr

def apply_charge_preserving_qec_no_syndrome(qc, num_qubits=7, num_classical=2):
    """Applies charge-preserving quantum error correction to a given circuit, without syndrome decoding."""

    q = QuantumRegister(num_qubits, 'q')
    c = ClassicalRegister(num_classical, 'c')  # Ensure classical bits are included
    qec_qc = QuantumCircuit(q, c)

    if qc.num_clbits < num_qubits:
        additional_clbits = num_qubits - qc.num_clbits
        qc.add_register(ClassicalRegister(additional_clbits, 'extra_c'))

    qc = add_classical_bits(qc, num_classical)

    # Apply charge-preserving encoding (Hadamards and CNOTs)
    for i in range(num_qubits - 1):
        qec_qc.h(q[i])
        qec_qc.cx(q[i], q[i + 1])

    qec_qc.h(q)  # Final Hadamard gates

    if qc.num_clbits == 0:
        qc.measure_all()

    # Convert input circuit to an instruction **without losing classical bits**
    qc_instruction = qc.to_instruction()

    # Append input circuit properly
    qec_qc.append(qc_instruction, q[:qc.num_qubits], c[:qc.num_clbits])

    # Ensure measurement matches classical register size
    if len(q) != len(c):
        print(f"Warning: Quantum register size ({len(q)}) != Classical register size ({len(c)}). Adjusting...")
        c = ClassicalRegister(len(q), 'c_adjusted')
        qec_qc.add_register(c)

    # Measure all qubits into classical bits
    qec_qc.measure(q[:num_qubits], c[:num_classical])

    return qec_qc, qc_instruction


def apply_charge_preserving_qec(qc, num_qubits=7, num_classical=2):
    """Applies charge-preserving quantum error correction to a given circuit."""
    
    q = QuantumRegister(num_qubits, 'q')
    c = ClassicalRegister(num_classical, 'c')  # Ensure classical bits are included
    qec_qc = QuantumCircuit(q, c)

    if qc.num_clbits < num_qubits:
        additional_clbits = num_qubits - qc.num_clbits
        qc.add_register(ClassicalRegister(additional_clbits, 'extra_c'))

    qc = add_classical_bits(qc, num_classical)

    # Apply charge-preserving encoding
    for i in range(num_qubits - 1):
        qec_qc.h(q[i])
        qec_qc.cx(q[i], q[i + 1])

    # Add syndrome bits if more than 2 qubits
    if num_qubits > 2:
        syndrome = QuantumRegister(2, 'syndrome')
        qec_qc.add_register(syndrome)

        for i in range(0, num_qubits - 2, 2):
            qec_qc.cx(q[i], q[i + 2])

        # Measure syndrome register only if it exists
        qec_qc.cx(q[0], syndrome[0])
        qec_qc.cx(q[1], syndrome[0])
        qec_qc.cx(q[2], syndrome[1])
        if num_qubits > 3:
            qec_qc.cx(q[3], syndrome[1])

    qec_qc.h(q)  # Final Hadamard gates
    if qc.num_clbits == 0:
        qc.measure_all()
    # Convert input circuit to an instruction **without losing classical bits**
    qc_instruction = qc.to_instruction()
    
    # Append input circuit properly
    qec_qc.append(qc_instruction, q[:qc.num_qubits], c[:qc.num_clbits])

    # 🔥 Ensure measurement matches classical register size
    if len(q) != len(c):
        print(f"Warning: Quantum register size ({len(q)}) != Classical register size ({len(c)}). Adjusting...")
        c = ClassicalRegister(len(q), 'c_adjusted')
        qec_qc.add_register(c)

    # Measure all qubits into classical bits
    qec_qc.measure(q[:num_qubits], c[:num_classical])

    return qec_qc, qc_instruction

def shor_qec_noisy(num_logical_qubits=1):
    num_physical_qubits = min(num_logical_qubits * 9, 15)  # Limit total qubits to 15
    q = QuantumRegister(num_physical_qubits, 'q')
    c = ClassicalRegister(num_logical_qubits, 'c')
    qc = QuantumCircuit(q, c)
    
    for i in range(num_logical_qubits):
        base = i * 9
        if base + 8 >= num_physical_qubits:
            break  # Prevent exceeding available qubits
        
        qc.h(q[base])
        qc.cx(q[base], q[base+1])
        qc.cx(q[base], q[base+2])
        
        if base + 6 < num_physical_qubits:
            qc.cx(q[base+1], q[base+3])
            qc.cx(q[base+1], q[base+4])
            qc.cx(q[base+2], q[base+5])
            qc.cx(q[base+2], q[base+6])
        
        qc.h(q[base])
        qc.h(q[base+1])
        qc.h(q[base+2])
        
        if base + 8 < num_physical_qubits:
            qc.cx(q[base], q[base+7])
            qc.cx(q[base+1], q[base+7])
            qc.cx(q[base+2], q[base+7])
            qc.cx(q[base+3], q[base+8])
            qc.cx(q[base+4], q[base+8])
            qc.cx(q[base+5], q[base+8])
            qc.measure(q[base+7], c[i])
    
    noise_model = create_noisy_model()
    simulator = Aer.get_backend('qasm_simulator')
    job = simulator.run(qc, noise_model=noise_model, shots=1024)
    result = job.result()
    
    counts = result.get_counts() if result.success else None
    print(f"Shor QEC ({num_logical_qubits} logical qubits, {num_physical_qubits} physical qubits):", counts)
    return counts

def quantum_black_hole_simulation_with_qec(num_logical_qubits=1):
    """Runs a black hole time evolution simulation with quantum error correction."""
    num_physical_qubits = num_logical_qubits * 9  # Using Shor encoding (9 physical per logical qubit)
    qc = QuantumCircuit(num_physical_qubits, num_physical_qubits)
    
    # Encode logical qubits
    for i in range(0, num_physical_qubits, 9):
        qc = apply_shor_encoding(qc, i)
    
    # Apply a black hole evolution-like unitary
    for i in range(num_physical_qubits - 1):
        qc.cx(i, i + 1)
        qc.rz(np.pi / 4, i)
    
    # Introduce artificial noise (simulate decoherence)
    qc.x(0)  # Simulate a random bit-flip error
    
    # Detect and correct errors
    for i in range(0, num_physical_qubits, 9):
        qc = detect_and_correct_errors(qc, i)
    
    # Simulate and extract density matrix
    simulator = Aer.get_backend('aer_simulator_density_matrix')
    qc.save_density_matrix()
    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(compiled_circuit).result()
    final_density_matrix = DensityMatrix(result.data(0)['density_matrix'])
    
    # Compute entropies
    entropies = [entropy(partial_trace(final_density_matrix, [i])) for i in range(num_physical_qubits)]
    
    return qc, entropies

def black_hole_time_evolution(qc, num_steps=10, delta_t=0.1):
    """
    Simulates the time evolution of a black hole system using a Hamiltonian-based approach.
    
    Parameters:
        qc (QuantumCircuit): The initial quantum circuit.
        num_steps (int): Number of time steps.
        delta_t (float): Time step size.
    
    Returns:
        evolved_qc (QuantumCircuit): The final evolved quantum circuit.
        entropy_history (list): Von Neumann entropy values over time.
    """
    num_qubits = qc.num_qubits
    backend = AerSimulator(method='density_matrix')
    
    # Define a simple Hamiltonian (random Hermitian matrix for now)
    H = np.random.rand(2**num_qubits, 2**num_qubits) + 1j * np.random.rand(2**num_qubits, 2**num_qubits)
    H = (H + H.conj().T) / 2  # Ensure Hermitian property
    
    # Compute the unitary time evolution operator U = exp(-i H Δt)
    U = Operator(expm(-1j * delta_t * H))
    
    # Prepare for entropy tracking
    entropy_history = []
    evolved_qc = qc.copy()
    
    for _ in range(num_steps):
        evolved_qc.unitary(U, evolved_qc.qubits, label='Time Evolution')
        transpiled_qc = transpile(evolved_qc, backend)
        qobj = assemble(transpiled_qc)
        result = backend.run(qobj).result()
        
        # Extract the density matrix
        density_matrix = result.data(0).get('density_matrix', None)
        if density_matrix is None:
            raise ValueError("Density matrix not found in simulation result.")
        density_matrix = DensityMatrix(density_matrix)
        
        # Compute Von Neumann entropy for subsystem
        entropy_values = [entropy(partial_trace(density_matrix, [i])) for i in range(num_qubits)]
        entropy_history.append(np.mean(entropy_values))
    
    return evolved_qc, entropy_history

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
        bit_array = data_bin.c  # Access the `BitArray`

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
            bit_array = data_bin.c  # Access the `BitArray`

            print("Bit array: ", bit_array)

            # Convert the BitArray to a list of bitstrings
            results = extract_counts_from_bitarray(bit_array)
            print("Results: ", results)

            entropy = analyze_shannon_entropy(results)
            print("Entropy: ", entropy)

        except Exception as e:
            print(e)

    return results

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

def apply_time_reversal(qc):
    """Apply time-reversal transformation (without measurement gates)."""
    qc_no_measure = qc.copy()
    qc_no_measure.data = [instr for instr in qc_no_measure.data if instr.operation.name != 'measure']
    return qc_no_measure.inverse()

def time_reverse_circuit(qc):
    """Time-reverses a quantum circuit by removing measurements, inverting, and re-adding measurements."""
    # Remove measurements
    qc_no_measurements = qc.remove_final_measurements(inplace=False)
    
    # Invert the unitary operations
    qc_reversed = qc_no_measurements.inverse()

    # Reapply measurements
    qc_reversed.measure_all()

    return qc_reversed

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

def add_classical_bits(qc, num_clbits):
    """Ensures that the quantum circuit has at least `num_clbits` classical bits."""
    current_clbits = qc.num_clbits

    if current_clbits < num_clbits:
        extra_clbits = num_clbits - current_clbits
        qc.add_register(ClassicalRegister(extra_clbits, 'extra_c'))

    return qc

def apply_charge_injection(qc, qubits):
    """
    Injects a charge-like phase shift onto selected qubits to test amplification effects.
    """
    for q in qubits:
        qc.p(np.pi / 4, q)  # Phase shift injection to alter probability distribution
    return qc

def generate_true_random_state(num_qubits):
    """
    Creates a maximally mixed state to compare against experimental biasing.
    """
    qc = QuantumCircuit(num_qubits)
    for q in range(num_qubits):
        qc.h(q)  # Superposition ensures equal probability
    return qc

def extract_counts(result, use_sampler):
    """Extracts measurement counts from a Qiskit result object, handling both standard and Sampler cases."""
    print("Result: ", result)

    if use_sampler:
        # Handle Sampler result extraction
        data_bin = result[0].data
        key = next(iter(data_bin.__dict__))  # Get first available key
        bitarray = getattr(data_bin, key)  # Access the BitArray
        counts = extract_counts_from_bitarray(bitarray)  # Convert it to standard counts

    else:
        # Handle standard Qiskit Result extraction
        if hasattr(result, 'results') and isinstance(result.results, list):
            # Extract the first experiment's data
            exp_result = result.results[0]
            if hasattr(exp_result.data, 'counts'):
                raw_counts = exp_result.data.counts
                # Convert hex keys ('0x0', '0x1', etc.) to integers
                counts = {int(k, 16): v for k, v in raw_counts.items()}
            else:
                raise ValueError("Counts not found in the result data.")
        else:
            raise ValueError("Invalid Qiskit result format.")

    return counts


def fast_amplify_11(iterations=10, shots=2048, initial_phase=np.pi/4, scaling_factor=0.3):
    """
    Amplifies the probability of the `11` state using adaptive charge injection and phase shifts.

    Args:
        iterations (int): Number of feedback cycles.
        shots (int): Shots per experiment.
        initial_phase (float): Initial phase shift (radians).
        scaling_factor (float): Charge update strength.
    """
    n_qubits = 2
    backend = AerSimulator()

    # Track charge history dynamically
    charge_history = np.zeros(n_qubits)

    for iteration in range(iterations):
        qc = QuantumCircuit(n_qubits, n_qubits)

        # Step 1: Initialize in equal superposition
        qc.h(range(n_qubits))

        # Step 2: Apply dynamic phase shifts based on past success
        phase_shift = initial_phase + (scaling_factor * charge_history.sum())

        # Reinforce the |11> state with targeted interference
        qc.cp(phase_shift, 0, 1)  # Controlled phase gate
        qc.cz(0, 1)  # Phase kick to steer amplitude toward 11
        qc.rx(np.pi * charge_history[0] * scaling_factor, 0)
        qc.rx(np.pi * charge_history[1] * scaling_factor, 1)

        # Step 3: Grover-like Amplification
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(1)
        qc.cp(-phase_shift, 0, 1)  # Inverse phase shift
        qc.h(1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

        # Step 4: Measure and Update Charge History
        qc.measure(range(n_qubits), range(n_qubits))

        transpiled_qc = transpile(qc, backend)
        result = backend.run(transpiled_qc, shots=shots).result()
        counts = result.get_counts()

        # Update charge tracking
        for state, count in counts.items():
            if state == "11":
                charge_history += count / shots  # Reinforce `11`
            else:
                charge_history -= count / (shots * 2)  # Reduce non-`11` states

        # Print iteration results
        print(f"Iteration {iteration + 1}/{iterations}:")
        print(f"Measurement Results: {counts}")
        print(f"Charge History: {charge_history}")

    # Final plot
    plot_histogram(counts)
    plt.title(f"Final Measurement Results (Amplifying `11` Faster)")
    plt.show()

    return counts

def mirrored_state_prep(qc, target_state):
    """
    Prepare a mirrored state based on the target state.
    
    Parameters:
    - qc (QuantumCircuit): The circuit to apply the mirrored state preparation.
    - target_state (str): Target state as a binary string (e.g., "101").
    """
    n_qubits = len(target_state)
    for i, bit in enumerate(reversed(target_state)):
        if bit == "1":
            qc.x(i)  # Apply X gate for '1' bits
    qc.h(range(n_qubits))  # Create superposition

    # Add phase shift to steer the circuit towards mirrored states
    for i, bit in enumerate(reversed(target_state)):
        angle = np.pi / 2 if bit == "0" else -np.pi / 2
        qc.rz(angle, i)  # Phase rotation based on mirroring
    qc.barrier()

def least_busy_backend(service, filters=None):
    """
    Find the least busy backend from the available IBM Quantum backends.

    Parameters:
        service (QiskitRuntimeService): An initialized QiskitRuntimeService object.
        filters (function): A lambda function to filter the list of backends.

    Returns:
        Backend: The least busy backend that matches the filter criteria.
    """
    # Get all backends
    backends = service.backends()

    # Apply filters if provided
    if filters:
        backends = list(filter(filters, backends))

    # Sort by the number of pending jobs (ascending)
    sorted_backends = sorted(
        backends, key=lambda b: b.status().pending_jobs
    )

    # Return the least busy backend
    return sorted_backends[0] if sorted_backends else None

def run_real_backend(qc, backend, shots=8192):
    """
    Runs a quantum circuit on a real quantum backend using the Sampler primitive.
    
    Parameters:
        qc (QuantumCircuit): The quantum circuit to run.
        backend (Backend): The IBM backend to execute on.
        shots (int): Number of shots for the experiment.
    
    Returns:
        dict: Measurement results.
    """

    if isinstance(qc, list):
        qc = qc[0]  # Take the first circuit if it's a list
    
    if not isinstance(qc, QuantumCircuit):
        raise ValueError("Expected a QuantumCircuit, but got:", type(qc))
    if qc.num_clbits > 3:
        qc = QuantumCircuit(qc.num_qubits, 3)  # Keep at most 3 classical bits
        qc.measure(range(qc.num_qubits), range(3))  # Map measurements to the 3 bits

    sampler = Sampler(backend)
    job = sampler.run([qc], shots=shots)
    result = job.result()
    counts = extract_counts(result, use_sampler=True) # Extract probabilities
    
    print("Counts: ", counts)
    
    return counts

def time_evolution_example(t_steps=5, shots=2048):
    """
    Simulate time-dependent transformations and holographic interactions.
    """
    n_qubits = 3
    qc = QuantumCircuit(n_qubits)
    
    # Step 1: Initialize superposition
    qc.h(0)  # Superposition on the Black Hole qubit
    qc.cx(0, 1)  # Entangle Black Hole and Radiation qubits

    # Step 2: Time Evolution
    for t in range(t_steps):
        angle = (np.pi / 3) * (t + 1)  # Dynamic angle for timeline distortion
        qc.rz(angle, 0)  # Timeline distortion on the Black Hole qubit
        qc.rx(np.pi / 4, 1)  # Holographic interaction on the Radiation qubit
        qc.cx(1, 2)  # Correlate Radiation with External Environment
        qc.barrier()

    # Step 3: Measure final probabilities
    qc.measure_all()

    # Analyze statevector before measurement
    qc_no_measurements = qc.copy()
    state = Statevector.from_instruction(qc_no_measurements)
    print("\nStatevector:")
    print(state)

    # Calculate subsystem entropies
    entropy_0 = entropy(partial_trace(state, [1, 2]))
    entropy_1 = entropy(partial_trace(state, [0, 2]))
    entropy_2 = entropy(partial_trace(state, [0, 1]))
    print(f"Subsystem Entropies: Qubit 0 = {entropy_0}, Qubit 1 = {entropy_1}, Qubit 2 = {entropy_2}")

    # Transpile and run on AerSimulator
    backend = Aer.get_backend('aer_simulator')
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()

    print("\nMeasurement Results:")
    print(counts)

    # Plot the histogram of results
    plot_histogram(counts)
    plt.title("Measurement Results with Time Evolution")
    plt.show()

def enhanced_time_evolution(t_steps, shots=2048):
    """
    Enhance time-dependent transformations and holographic interactions.
    """
    qc = QuantumCircuit(3, 3)

    # Initialize entanglement
    qc.h(0)  # Black Hole qubit in superposition
    qc.cx(0, 1)  # Entangle Black Hole and Radiation
    qc.cx(1, 2)  # Extend entanglement to External Environment

    # Time-evolution with holographic interactions
    for t in range(1, t_steps + 1):
        angle_timeline = np.pi * t / 10
        angle_holographic = np.pi * t / 15

        qc.rz(angle_timeline, 0)  # Timeline distortion (Black Hole qubit)
        qc.rx(angle_holographic, 1)  # Holographic interaction (Radiation qubit)
        qc.cp(np.pi * t / 20, 1, 2)  # Controlled phase interaction

    # Statevector analysis (no measurement gates)
    qc_no_measurements = qc.copy()

    # Add measurements for final probabilities
    qc.measure([0, 1, 2], [0, 1, 2])

    # Analyze statevector
    state = Statevector.from_instruction(qc_no_measurements)
    entropies = calculate_entropies(state)

    print("\nStatevector:")
    print(state)
    print("Subsystem Entropies:", entropies)

    # Run the circuit on a simulator
    results = run_warp_simulation(qc)
    print("\nMeasurement Results:", results)

    return results, entropies

def create_randomized_circuit(num_qubits, depth):
    """Generate a randomized quantum circuit for entropy comparison."""
    qc = QuantumCircuit(num_qubits)
    for _ in range(depth):
        for qubit in range(num_qubits):
            gate = np.random.choice(["h", "x", "y", "z", "cx"])
            if gate == "h":
                qc.h(qubit)
            elif gate == "x":
                qc.x(qubit)
            elif gate == "y":
                qc.y(qubit)
            elif gate == "z":
                qc.z(qubit)
            elif gate == "cx" and qubit < num_qubits - 1:
                qc.cx(qubit, qubit + 1)
    return qc

def charge_injection_scaling(qc, max_levels=5):
    """Scale charge injection cycles and measure entropy."""
    results = []
    for level in range(1, max_levels + 1):
        qc_injected = qc.copy()
        for _ in range(level):
            for qubit in range(qc.num_qubits):
                qc_injected.rx(np.pi / (level + 1), qubit)  # Simulate charge injection

        qc_no_measure = remove_measurements(qc_injected)  # Remove measurements
        state = Statevector.from_instruction(qc_no_measure)

        ent = entropy(state)
        results.append((level, ent))
        print(f"Charge Level {level}: Entropy = {ent}")

    return results

def charge_injection_scaling_0(qc, max_levels=50):
    """Scale charge injection cycles and measure entropy, returning the modified quantum circuit."""
    results = []
    qc_injected = qc.copy()  # Start with a fresh copy

    for level in range(1, max_levels + 1):
        for qubit in range(qc.num_qubits):
            qc_injected.rx(np.pi / (level + 1), qubit)  # Simulate charge injection

        qc_no_measure = remove_measurements(qc_injected)  # Remove measurements
        state = Statevector.from_instruction(qc_no_measure)
        
        ent = entropy(state)
        results.append((level, ent))
        print(f"Charge Level {level}: Entropy = {ent}")

    # **Ensure the final circuit has measurements before returning**
    if not qc_injected.clbits:
        qc_injected.add_register(ClassicalRegister(qc.num_qubits))

    qc_injected.measure_all()

    return qc_injected  # **Now it returns a QuantumCircuit**


def run_experiment_with_target(backend_type, target_state="111", t_steps=5, shots=2048):
    """
    Run a mirrored state experiment with time-dependent holographic operations.
    """
    # Select backend
    if backend_type == "simulator":
        backend = Aer.get_backend('aer_simulator')
        print("\nRunning on AerSimulator...")
    elif backend_type == "quantum":
        service = QiskitRuntimeService(channel="ibm_quantum")  # Load IBM Quantum account
        filters = lambda b: b.configuration().n_qubits >= 2 and b.status().operational
        backend = least_busy_backend(service, filters)
        if backend is None:
            raise ValueError("No backends are appear to be working at the minute...")
        print(f"\nRunning on quantum backend: {backend.name}...")
    else:
        raise ValueError("Invalid backend_type. Use 'simulator' or 'quantum'.")

    n_qubits = len(target_state)
    qc = QuantumCircuit(n_qubits, n_qubits)  # Create the quantum circuit

    # Step 1: Mirrored State Preparation
    mirrored_state_prep(qc, target_state)  # Pass the circuit as an argument

    # Step 2: Time-dependent holographic operations
    for t in range(1, t_steps + 1):
        angle_timeline = np.pi * t / 10
        angle_holographic = np.pi * t / 15

        qc.rz(angle_timeline, 0)  # Timeline distortion on Black Hole qubit
        qc.rx(angle_holographic, 1)  # Holographic interaction on Radiation qubit
        qc.cx(1, 2)  # Correlate Radiation with Environment

    # Step 3: Add measurement
    qc.measure(range(n_qubits), range(n_qubits))
    
    # Transpile and run
    transpiled_qc = transpile(qc, backend=backend, optimization_level=3)
    print(f"Transpiled Circuit Depth: {transpiled_qc.depth()}")

    if backend_type == "simulator":
        result = backend.run(transpiled_qc, shots=shots).result()
        counts = result.get_counts()
    elif backend_type == "quantum":
        with Session(backend=backend) as session:
        # Use Qiskit Runtime Sampler for quantum backend
            sampler = Sampler()
            job = sampler.run([transpiled_qc])
            result = job.result()
            print("Result: ", result)
                # Extract counts from the nested structure
            try:
            # Navigate to the `BitArray` and extract counts
                pub_result = result._pub_results[0]  # Access the first `SamplerPubResult`
                data_bin = pub_result.data  # Access `DataBin`
                bit_array = data_bin.c  # Access `BitArray`

                counts = extract_counts_from_bitarray(bit_array)
                print("Results: ", counts)

            except Exception as e:
                print(e)

def charge_amplification_11(shots=2048, cycles=5):
    """
    Uses charge injection cycles to amplify the |11⟩ state in a two-qubit system.
    """

    n_qubits = 2
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Step 1: Initialize in Superposition
    qc.h(range(n_qubits))

    # Step 2: Charge Injection Cycles for Amplification
    for cycle in range(cycles):
        phase_shift = (np.pi / (2 + cycle))  # Adaptive phase shift
        qc.cp(phase_shift, 0, 1)  # Inject charge-like correlation
        qc.cz(0, 1)  # Reinforce entanglement
        qc.rx(phase_shift / 2, 0)  # Adjust rotation for coherence
    
    # Step 3: Measurement
    qc.measure(range(n_qubits), range(n_qubits))

    # Simulate Execution
    backend = Aer.get_backend('aer_simulator')
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()

    # Plot Results
    plot_histogram(counts)
    plt.title(f"Charge Injection Amplification (Cycles={cycles})")
    plt.show()

    return counts

def compute_qfi(qc: QuantumCircuit):
    """
    Computes the Quantum Fisher Information (QFI) matrix for a given quantum circuit.
    """
    backend = Aer.get_backend('statevector_simulator')
    qc = transpile(qc, backend)
    
    # Get final statevector
    job = backend.run(qc)
    statevector = Statevector(job.result().get_statevector())
    
    # Compute QFI using the Fubini-Study metric
    rho = statevector.to_operator()
    num_qubits = qc.num_qubits
    
    qfi_matrix = np.zeros((num_qubits, num_qubits), dtype=np.complex128)
    
    for i in range(num_qubits):
        for j in range(num_qubits):
            sigma_i = np.kron(np.eye(2**i), np.kron(np.array([[0, -1j], [1j, 0]]), np.eye(2**(num_qubits-i-1))))
            sigma_j = np.kron(np.eye(2**j), np.kron(np.array([[0, -1j], [1j, 0]]), np.eye(2**(num_qubits-j-1))))
            
            qfi_matrix[i, j] = np.trace(rho @ sigma_i @ rho @ sigma_j)
    
    # Get eigenvalues to analyze structure
    eigenvalues = np.linalg.eigvalsh(qfi_matrix.real)
    
    return qfi_matrix.real, eigenvalues

def apply_amplitude_amplification(qc, target_state='11'):
    """
    Applies amplitude amplification targeting the given state.
    """
    target_int = int(target_state, 2)  # Convert '11' -> 3
    num_qubits = qc.num_qubits

    # Apply oracle: Flip the phase of the target state
    qc.cz(0, 1)  # Adjust based on target state
    
    # Apply diffusion operator (inversion around mean)
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.cz(0, 1)  # Controlled phase inversion
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    
    return qc

def adaptive_charge_injection(qc, target_state="11", prior_counts=None, scaling_factor=1.0):
    """
    Dynamically injects charge into a quantum circuit based on past measurement results.
    Uses a holographic phase memory effect to reinforce state probabilities over time.
    """
    num_qubits = len(target_state)
    charge_register = np.zeros(num_qubits)  # Track charge injection per qubit
    
    # If prior_counts exist, use them to scale charge injection
    if prior_counts:
        total_shots = sum(prior_counts.values())
        for state, count in prior_counts.items():
            probability = count / total_shots
            for i, bit in enumerate(reversed(state)):  # Reverse for correct mapping
                if bit == '1':
                    charge_register[i] += probability * scaling_factor  # Scale injection
    
    # Apply charge injections
    for qubit in range(num_qubits):
        qc.rz(charge_register[qubit] * np.pi / 2, qubit)  # Phase-based charge
        qc.rx(charge_register[qubit] * np.pi / 3, qubit)  # Holographic interaction
    
    return qc

def amplify_11_in_qc(qc):
    """
    Apply targeted amplification to the |11> state within an existing quantum circuit.
    """
    n_qubits = 2  # Assuming a 2-qubit system for |11> amplification

    # Apply a Hadamard to create a superposition (if not already in a specific state)
    qc.h(range(n_qubits))

    # Apply controlled phase shifts to favor |11>
    qc.cp(np.pi / 2, 0, 1)  # Controlled phase shift
    qc.cz(0, 1)  # Controlled Z gate to enhance |11> probability

    # Apply a Grover-like diffusion operator to reinforce |11>
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(1)
    qc.cx(0, 1)  # Controlled-X (CNOT) to mark |11>
    qc.h(1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))

    return qc

def run_amplification_experiment(shots=2048):
    """Runs a quantum experiment using adaptive charge injection."""
    backend = Aer.get_backend('aer_simulator')
    qc = QuantumCircuit(2, 2)
    
    # Initial superposition
    qc.h([0, 1])
    
    # Adaptive charge injection
    prior_counts = {'00': 500, '01': 500, '10': 500, '11': 500}  # Example past data
    qc = adaptive_charge_injection(qc, "11", prior_counts, scaling_factor=2.0)
    
    # Measurement
    qc.measure([0, 1], [0, 1])
    
    # Run simulation
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()
    
    print("Measurement Results:", counts)
    return counts

def create_holographic_time_circuit(target_state="11"):
    """
    Creates a quantum circuit to simulate holographic timeline interactions with a targeted state.
    Args:
        target_state (str): The target state to encode for the Black Hole and Radiation qubits.
    Returns:
        QuantumCircuit: The quantum circuit representing the timeline interaction.
    """
    n_qubits = len(target_state)
    qc = QuantumCircuit(n_qubits, n_qubits)  # Create circuit with classical bits for measurement

    # Encode target state
    for i, bit in enumerate(target_state):
        if bit == '1':
            qc.x(i)  # Flip the qubit

    # Apply holographic timeline interaction
    qc.h(0)  # Superposition for Black Hole qubit
    qc.cx(0, 1)  # Entangle Black Hole with Radiation
    qc.rz(np.pi / 3, 0)  # Simulate timeline distortion on Black Hole
    qc.rx(np.pi / 4, 1)  # Simulate holographic interaction on Radiation

    qc.cx(0, 1)  # Controlled-NOT to mark `|11⟩`
    qc.z(1)  # Phase flip on `|11⟩`

    qc.measure(range(n_qubits), range(n_qubits))

    return qc

def run_qc_with_diagnostics(qc, shots=2048):
    """
    Runs a given quantum circuit on a simulator and provides diagnostics.
    """
    backend = Aer.get_backend('aer_simulator')

    # Transpile for the backend
    transpiled_qc = transpile(qc, backend)
    
    # Run the circuit
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()

    # Print measurement results
    print("\nMeasurement Results:")
    print(counts)

    # Plot histogram
    plt.figure(figsize=(8,5))
    plt.bar(counts.keys(), counts.values(), color='royalblue')
    plt.xlabel("Measurement Outcome")
    plt.ylabel("Counts")
    plt.title("Quantum Circuit Measurement Results")
    plt.show()

    # Calculate subsystem entropy if possible
    try:
        state = Statevector.from_instruction(qc)
        entropy_0 = entropy(partial_trace(state, [1]))
        entropy_1 = entropy(partial_trace(state, [0]))
        print(f"Subsystem Entropies: Qubit 0 = {entropy_0}, Qubit 1 = {entropy_1}")
    except Exception as e:
        print(f"Could not calculate subsystem entropy: {e}")

    return counts

def construct_baseline_circuit(n_qubits=3):
    """Create a neutral quantum circuit with equal superposition."""
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))  # Equal superposition
    qc.measure(range(n_qubits), range(n_qubits))
    return qc

def apply_charge_injection(qc):
    """Redirect charge bias to favor 11 instead of 00."""
    injected_qc = qc.copy()
    injected_qc.rx(np.pi / 3, 0)
    injected_qc.ry(np.pi / 3, 1)
    return injected_qc

def introduce_phase_distortion(qc):
    """Introduce phase shifts to disrupt any bias."""
    distorted_qc = qc.copy()
    for qubit in range(qc.num_qubits):
        distorted_qc.p(np.pi / 4, qubit)  # Introduce small phase shifts
    return distorted_qc

def apply_holographic_encoding(qc):
    """Apply nonlocal entanglement-based encoding."""
    holographic_qc = qc.copy()
    holographic_qc.cx(0, 1)
    holographic_qc.cp(np.pi / 2, 0, 1)
    return holographic_qc

def run_circuit(qc, shots=32768):
    """Execute a circuit on a simulator and return measurement results."""
    backend = Aer.get_backend('aer_simulator')
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    return result.get_counts()

def analyze_results(counts, label):
    """Plot histogram of results and print probability distribution."""
    print(f"\n{label} Counts:", counts)
    plot_histogram(counts, title=label)
    plt.show()

def amplify_11_phase_balanced(qc):
    """
    Amplifies the |11⟩ state while applying a phase-balancing correction
    to prevent unintended asymmetry during time-reversal.
    """
    qc = qc.copy()  # Work on a copy to avoid modifying the original circuit
    
    # Apply controlled phase shifts to steer probability towards |11>
    phase_shift = np.pi / 4  # Adjustable phase correction factor
    qc.cp(phase_shift, 0, 1)  # Controlled phase shift on first two qubits
    qc.cz(0, 1)  # Apply extra correction to balance inversion effects
    
    # Phase balancing correction applied before final amplification
    qc.h(1)
    qc.sdg(1)  # Counteracts unwanted drift
    qc.h(1)
    
    # Targeted amplification
    qc.h(1)  
    qc.cx(0, 1)
    qc.cz(0, 1)  
    qc.h(1)

    return qc

def create_entangled_system(n_qubits=3):
    """Creates an entangled GHZ-like state to observe entropy dynamics."""
    qc = QuantumCircuit(n_qubits)
    qc.h(0)  # Hadamard on first qubit
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)  # Chain CNOTs for GHZ entanglement
    return qc

def measure_entropy(qc):
    """Computes the entropy of subsystems."""
    state = Statevector.from_instruction(qc)
    subsys_entropy = [entropy(partial_trace(state, [i])) for i in range(qc.num_qubits)]
    return subsys_entropy

def add_hawking_radiation(qc, rad_qubits=2):
    """Simulates Hawking radiation by adding extra qubits entangled with the system."""
    n = qc.num_qubits
    rad_reg = QuantumRegister(rad_qubits, name="radiation")  
    qc.add_register(rad_reg)  # Adds radiation qubits  # Extend with radiation qubits
    for i in range(rad_qubits):
        qc.cx(i, n + i)  # Entangle system with radiation
    return qc

def add_charge_injection(qc, qubits):
    """Applies charge injection through controlled phase shifts and rotation gates."""
    for qubit in qubits:
        qc.rz(np.pi / 4, qubit)  # Introduce charge-like phase shifts
        qc.rx(np.pi / 6, qubit)  # Inject coherent charge
    return qc

def remove_measurements(qc):
    """Removes measurement operations to allow circuit inversion."""
    qc_no_measure = QuantumCircuit(qc.num_qubits)
    for instr, qargs, cargs in qc.data:
        if instr.name != "measure":
            qc_no_measure.append(instr, qargs, cargs)
    return qc_no_measure

def apply_charge_injection(qc, qubits, phase_shifts=None, cycles=1):
    """
    Applies charge injection cycles to the given quantum circuit.
    
    Parameters:
        qc (QuantumCircuit): The quantum circuit to modify.
        qubits (list): The qubits to apply charge injection to.
        phase_shifts (list, optional): The phase shifts for each qubit. Defaults to randomized shifts.
        cycles (int): Number of charge injection cycles to apply.

    Returns:
        QuantumCircuit: The modified circuit with charge injection applied.
    """
    qc_modified = qc.copy()

    # Generate random phase shifts if not provided
    if phase_shifts is None:
        phase_shifts = [np.random.uniform(0, 2 * np.pi) for _ in qubits]

    for _ in range(cycles):
        for qubit, phase in zip(qubits, phase_shifts):
            qc_modified.p(phase, qubit)  # Apply phase injection
            qc_modified.h(qubit)         # Hadamard gate to spread interference
            qc_modified.p(-phase, qubit) # Reverse phase to maintain coherence
    
    return qc_modified

def apply_charge_injection_universal(qc, qubits=None, phase_shifts=None, cycles=1):
    """
    Applies charge injection cycles to a given quantum circuit.

    Parameters:
        qc (QuantumCircuit): The quantum circuit to modify.
        qubits (list, optional): The qubits to apply charge injection to. Defaults to all qubits.
        phase_shifts (list, optional): Custom phase shifts for each qubit. Defaults to randomized shifts.
        cycles (int): Number of charge injection cycles.

    Returns:
        QuantumCircuit: The modified circuit with charge injection applied.
    """
    qc_modified = qc.copy()

    # If no qubits specified, apply to all qubits
    if qubits is None:
        qubits = list(range(qc.num_qubits))

    # Generate random phase shifts if not provided
    if phase_shifts is None:
        phase_shifts = [np.random.uniform(0, 2 * np.pi) for _ in qubits]

    for _ in range(cycles):
        for qubit, phase in zip(qubits, phase_shifts):
            qc_modified.p(phase, qubit)  # Phase shift (simulating charge imbalance)
            qc_modified.h(qubit)         # Hadamard for interference spread
            qc_modified.p(-phase, qubit) # Reverse phase to maintain coherence
    
    return qc_modified

def apply_probability_amplification(qc, target_qubits=None, amplification_factor=1.2):
    """
    Applies probability amplification by adjusting the phase of certain qubits.

    Parameters:
        qc (QuantumCircuit): The quantum circuit to modify.
        target_qubits (list): List of qubits to apply amplification to. Defaults to all qubits.
        amplification_factor (float): Factor by which the probability should be biased.

    Returns:
        QuantumCircuit: The modified circuit with probability amplification.
    """
    if target_qubits is None:
        target_qubits = list(range(qc.num_qubits))  # Default to all qubits
    
    amplified_qc = qc.copy()
    
    # Apply phase shifts to amplify probabilities of target states
    for qubit in target_qubits:
        amplified_qc.p(amplification_factor, qubit)  # Phase shift to enhance probability
    
    return amplified_qc


def remove_measurements(qc):
    qc_no_measure = qc.copy()
    qc_no_measure.data = [instr for instr in qc_no_measure.data if instr.operation.name != 'measure']
    return qc_no_measure

def multiverse_test_circuit(num_qubits=3):
    """
    Creates a circuit that entangles qubits, applies interference, 
    and checks for multiversal correlations.
    """
    qc = QuantumCircuit(num_qubits, num_qubits)

    # Step 1: Create Entanglement
    qc.h(0)
    for i in range(1, num_qubits):
        qc.cx(i-1, i)

    # Step 2: Apply Phase Kicks (Simulates Different Pathways)
    for i in range(num_qubits):
        qc.p(np.pi / 4, i)  # Small phase shifts

    # Step 3: Introduce Interference
    for i in range(num_qubits):
        qc.h(i)

    # Step 4: Measure only a subset to test hidden correlations
    qc.measure(range(num_qubits), range(num_qubits))

    return qc

def charge_transfer_experiment(measure=False):
    """
    Constructs a quantum circuit to test charge transfer via entanglement.
    If measure=False, returns a version without measurements for statevector analysis.
    """
    qr = QuantumRegister(2, name="q")  # Two entangled qubits
    cr = ClassicalRegister(2, name="c")  # Classical register for measurement

    qc = QuantumCircuit(qr, cr)

    # Step 1: Create an Entangled Pair (Q1 <--> Q2)
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])

    # Step 2: Inject Charge into Q1
    qc.rx(1.57, qr[0])  # Simulate charge injection
    qc.crx(1.57, qr[0], qr[1])

    if measure:
        # Step 3: Measure Only Q2 to See If It Extracts Charge
        qc.measure(qr[1], cr[1])

    else:
        qc.save_statevector()

    return qc

def charge_transfer_experiment_3qubits(measure=False):
    qr = QuantumRegister(3, name="q")  # Three entangled qubits
    cr = ClassicalRegister(3, name="c")  # Classical register for measurement
    qc = QuantumCircuit(qr, cr)

    # Step 1: Create a 3-Qubit Entangled Chain (Q1 ↔ Q2 ↔ Q3)
    qc.h(qr[0])      # Hadamard on Q1
    qc.cx(qr[0], qr[1])  # Entangle Q1 <-> Q2
    qc.cx(qr[1], qr[2])  # Entangle Q2 <-> Q3

    # Step 2: Inject Charge into Q1
    qc.rx(1.57, qr[0])  # Charge injection at Q1

    # Step 3: Controlled Charge Transfer to Q2 and Q3
    qc.crx(1.57, qr[0], qr[1])  # Transfer charge influence from Q1 to Q2
    qc.crx(1.57, qr[1], qr[2])  # Transfer charge influence from Q2 to Q3

    if measure:
        # Step 4: Measure Q3 to See If It Extracts Charge
        qc.measure(qr[2], cr[2])

    else:
        # ✅ Explicitly Save Statevector to Ensure It Is Accessible
        qc.save_statevector()

    return qc

def apply_probability_biasing(qc, bias_qubits):
    """
    Applies controlled probability amplification (biasing) to selected qubits.
    
    Args:
        qc (QuantumCircuit): The quantum circuit to modify.
        bias_qubits (list): List of qubits to apply biasing to.

    Returns:
        QuantumCircuit: The modified circuit with biasing.
    """
    qc_bias = qc.copy()
    for qubit in bias_qubits:
        qc_bias.rx(np.pi / 4, qubit)  # Injects slight phase biasing
        qc_bias.ry(np.pi / 4, qubit)  # Helps realign probability spread

    return qc_bias

def create_entangled_circuit(qc):
    """
    Modifies a given quantum circuit to ensure entanglement.

    Parameters:
        qc (QuantumCircuit): The input quantum circuit.

    Returns:
        QuantumCircuit: The modified circuit with enforced entanglement.
    """
    num_qubits = qc.num_qubits

    # Apply Hadamard to the first qubit to create superposition
    qc.h(0)

    # Apply CNOT gates to entangle all qubits in a chain
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    return qc

def mbqc_energy_transfer_experiment(measure=False):
    qr = QuantumRegister(2, name="q")  # Space (Q_space) ↔ Earth (Q_earth)
    cr = ClassicalRegister(2, name="c")  # Classical register for measurement
    qc = QuantumCircuit(qr, cr)

    # Step 1: Create Entanglement Between Space & Earth
    qc.h(qr[0])         # Hadamard on Q_space
    qc.cx(qr[0], qr[1]) # Entangle Q_space <-> Q_earth

    # Step 2: Inject Energy into Space-Based Qubit
    qc.rx(1.57, qr[0])  # Charge injection at Q_space

    if measure:
        # Step 3: Measure Space Qubit to Trigger Remote Charge Transfer
        qc.measure(qr[0], cr[0])
        
        # Step 4: Apply Z-Correction on Earth Qubit Based on Space Qubit's Measurement
        qc.z(qr[1]).c_if(cr[0], 1)  # If Space Qubit is |1⟩, apply phase shift on Earth Qubit

    else:
        # ✅ Save Statevector to Analyze Effects Without Measurement
        qc.save_statevector()

    return qc

def apply_decoherence(qc):
    """
    Applies a depolarizing channel to simulate environmental decoherence.
    """
    noise_model = NoiseModel()
    error = depolarizing_error(0.05, 1)  # 5% chance of decoherence
    noise_model.add_all_qubit_quantum_error(error, "u3")  # Apply to all single-qubit gates

    return qc, noise_model

def compute_partial_entropy(statevector, subsystem):
    """
    Computes von Neumann entropy of a selected subsystem.

    Parameters:
        statevector (Statevector): The full quantum state.
        subsystem (list): List of qubit indices to trace out.

    Returns:
        float: The entropy of the remaining system.
    """
    rho = partial_trace(statevector, subsystem)  # Reduce system
    return entropy(rho)  # Compute entropy


#Functions that start with main_ are used as benchmarks of experiments that we've run together so I can run them again
#and not lose like the outputs in the sea of factory code

def main_run_entropy_experiment():
    """Tests charge injection on entropy in a time-reversed system."""
    backend = Aer.get_backend("aer_simulator")

    # Create base circuit
    qc = QuantumCircuit(3)
    qc.h(range(3))  # Initial superposition
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()

    # Run baseline entropy measurement
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=8192).result()
    counts_baseline = result.get_counts()

    # Apply time reversal (remove measurements)
    qc_nm = QuantumCircuit(qc.num_qubits)
    qc_nm.compose(qc.remove_final_measurements(inplace=False), inplace=True)
    if qc_nm.depth() == 0:
        raise ValueError("Circuit is empty after removing measurements. Check gate preservation.")

    qc.measure_all()
    
    qc_rev = qc.copy()
    qc_rev_nm = QuantumCircuit(qc.num_qubits)
    qc_rev_nm.compose(qc_rev.remove_final_measurements(inplace=False), inplace=True)
    qc_reversed = qc_rev_nm.inverse()
    qc_reversed.measure_all()
    transpiled_qc_reversed = transpile(qc_reversed, backend)
    result_reversed = backend.run(transpiled_qc_reversed, shots=8192).result()
    counts_reversed = result_reversed.get_counts()

    # Apply charge injection & re-run
    
    
    qc_c_rev = qc_reversed.copy()
    qc_c_rev_nm = QuantumCircuit(qc.num_qubits)
    qc_c_rev_nm.compose(qc_c_rev.remove_final_measurements(inplace=False), inplace=True)
    qc_charge_rev = add_charge_injection(qc_rev, [0, 1, 2])
    qc_charge_rev.measure_all()
    transpiled_qc_charge = transpile(qc_charge_rev, backend)
    result_charge = backend.run(transpiled_qc_charge, shots=8192).result()
    counts_charge = result_charge.get_counts()

    # Compute entropies
    state = Statevector.from_instruction(qc_nm)
    state_rev = Statevector.from_instruction(qc_rev_nm)
    state_charge = Statevector.from_instruction(qc_c_rev_nm)
    entropy_baseline = [entropy(partial_trace(state, [i])) for i in range(3)]
    entropy_reversed = [entropy(partial_trace(state_rev, [i])) for i in range(3)]
    entropy_charge = [entropy(partial_trace(state_charge, [i])) for i in range(3)]

    # Plot results
    plt.plot(entropy_baseline, label="Baseline", linestyle="-")
    plt.plot(entropy_reversed, label="Time-Reversed", linestyle="--")
    plt.plot(entropy_charge, label="Charge Injection", linestyle=":")
    plt.legend()
    plt.xlabel("Qubit Index")
    plt.ylabel("Entropy")
    plt.title("Entropy Evolution in Charge-Injected Time Reversal")
    plt.show()

    print(f"Baseline Counts: {counts_baseline}")
    print(f"Time-Reversed Counts: {counts_reversed}")
    print(f"Charge-Injection Counts: {counts_charge}")

def main_run_entropy_experiment_quantum():
    """Tests charge injection on entropy in a time-reversed system."""
    service = QiskitRuntimeService(channel='ibm_quantum')
    backend = get_best_backend(service)
    sampler = Sampler(backend)

    # Create base circuit
    qc = QuantumCircuit(3)
    qc.h(range(3))  # Initial superposition
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()

    # Run baseline entropy measurement
    transpiled_qc = transpile(qc, backend)
    result = sampler.run([transpiled_qc], shots=8192).result()
    counts_baseline = extract_counts_from_bitarray(result[0].data.meas)  # List of bitstring samples
    

    # Apply time reversal (remove measurements)
    qc_nm = QuantumCircuit(qc.num_qubits)
    qc_nm.compose(qc.remove_final_measurements(inplace=False), inplace=True)
    if qc_nm.depth() == 0:
        raise ValueError("Circuit is empty after removing measurements. Check gate preservation.")

    qc.measure_all()
    
    qc_rev = qc.copy()
    qc_rev_nm = QuantumCircuit(qc.num_qubits)
    qc_rev_nm.compose(qc_rev.remove_final_measurements(inplace=False), inplace=True)
    qc_reversed = qc_rev_nm.inverse()
    qc_reversed.measure_all()
    transpiled_qc_reversed = transpile(qc_reversed, backend)
    result_reversed = sampler.run([transpiled_qc_reversed], shots=8192).result()
    counts_reversed = extract_counts_from_bitarray(result_reversed[0].data.meas) # List of bitstring samples
    
    qc_c_rev = qc_reversed.copy()
    qc_c_rev_nm = QuantumCircuit(qc.num_qubits)
    qc_c_rev_nm.compose(qc_c_rev.remove_final_measurements(inplace=False), inplace=True)
    qc_charge_rev = add_charge_injection(qc_rev, [0, 1, 2])
    qc_charge_rev.measure_all()
    transpiled_qc_charge = transpile(qc_charge_rev, backend)
    result_charge = sampler.run([transpiled_qc_charge], shots=8192).result()
    counts_charge = extract_counts_from_bitarray(result_charge[0].data.meas)  # List of bitstring samples

    print(f"Counts baseline: {counts_baseline}")
    print(f"Counts reversed: {counts_reversed}")
    print(f"Counts charged: {counts_charge}")

    # Compute entropies
    state = Statevector.from_instruction(qc_nm)
    state_rev = Statevector.from_instruction(qc_rev_nm)
    state_charge = Statevector.from_instruction(qc_c_rev_nm)
    entropy_baseline = [entropy(partial_trace(state, [i])) for i in range(3)]
    entropy_reversed = [entropy(partial_trace(state_rev, [i])) for i in range(3)]
    entropy_charge = [entropy(partial_trace(state_charge, [i])) for i in range(3)]

    # Plot results
    plt.plot(entropy_baseline, label="Baseline", linestyle="-")
    plt.plot(entropy_reversed, label="Time-Reversed", linestyle="--")
    plt.plot(entropy_charge, label="Charge Injection", linestyle=":")
    plt.legend()
    plt.xlabel("Qubit Index")
    plt.ylabel("Entropy")
    plt.title("Entropy Evolution in Charge-Injected Time Reversal")
    plt.show()

    print(f"Baseline Counts: {counts_baseline}")
    print(f"Time-Reversed Counts: {counts_reversed}")
    print(f"Charge-Injection Counts: {counts_charge}")



def main_run_reverse_entropy():
    """Runs the entropy experiment with forward, reversed, and Hawking radiation scenarios."""
    backend = Aer.get_backend('aer_simulator')
    
    # Step 1: Create baseline entangled system
    qc = create_entangled_system()
    transpiled_qc = transpile(qc, backend)
    baseline_entropy = measure_entropy(qc)

    # Step 2: Reverse time and measure entropy
    qc_reversed = time_reverse_circuit(qc)
    transpiled_qc_rev = transpile(qc_reversed, backend)
    reversed_entropy = measure_entropy(qc_reversed)

    # Step 3: Simulate Hawking radiation by adding external qubits
    qc_radiation = add_hawking_radiation(qc, rad_qubits=2)
    transpiled_qc_rad = transpile(qc_radiation, backend)
    radiation_entropy = measure_entropy(qc_radiation)

    # Step 4: Print results
    print(f"Baseline Entropy: {baseline_entropy}")
    print(f"Time-Reversed Entropy: {reversed_entropy}")
    print(f"Entropy with Hawking Radiation: {radiation_entropy}")

    # Step 5: Plot entropy evolution
    plt.plot(range(len(baseline_entropy)), baseline_entropy, label="Baseline")
    plt.plot(range(len(reversed_entropy)), reversed_entropy, label="Time-Reversed", linestyle="dashed")
    plt.plot(range(len(radiation_entropy)), radiation_entropy, label="Hawking Radiation", linestyle="dotted")
    plt.xlabel("Qubit Index")
    plt.ylabel("Entropy")
    plt.title("Entropy Evolution in Time & Radiation Effects")
    plt.legend()
    plt.show()


def main_quantum_encrypt():
    """Quantum encryption test"""
    """Main program loop."""
    print("Starting Error Correction Test...\n")

    # Example Quantum Circuit
    n_qubits = 3
    qc = QuantumCircuit(n_qubits)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1)  # CNOT gate between qubit 0 and 1
    qc.cx(0, 2)  # CNOT gate between qubit 0 and 2
    qc.measure_all()

    # Backend and shots
    backend = Aer.get_backend('qasm_simulator')
    shots = 1024
    state = "011"

    # Run the error correction testing
    print("Original Circuit:")
    print(qc)
    result = run_circuit_with_feedback(qc, backend, state, shots=shots)

    # Display final results
    for key in result.keys():
        print(f"Key: {key}")
        print("\nFinal Measurement Results:")

def main_compare_qec_methods():
    """Compare shors qec to our own"""
    for qubits in [1, 2, 3]:
        charge_counts = charge_preserving_qec_noisy(num_qubits=qubits)
        print(f"Charge-Preserving QEC {qubits} qubits result: {charge_counts}")
    
    for logical_qubits in [1, 2, 3]:
        shor_counts = shor_qec_noisy(num_logical_qubits=logical_qubits)
        print(f"Shor QEC {logical_qubits} logical qubits result: {shor_counts}")
        
def main_check_decrypt():
    """Test to see if we can create a type of quantum encryption/decryption and apply it to the branches"""
    key = "110011"
    monitored_events = ["identity theft", "unauthorized access", "data breach"]

    # 🚨 Auto-detect branches
    detected_branches = auto_detect_branches(monitored_events, key, detection_threshold=60)
    print("🧭 Detected Branches:")
    for event, details in detected_branches.items():
        print(f"- {event}: Signature {details['signature_bits']} (Confidence: {details['detection_confidence']}%)")

        # 🔒 Encrypt detected branch signature
        encrypted_counts, unitary = quantum_scramble_encrypt(details['signature_bits'], key)
        print("  🔒 Encrypted Outcomes:", encrypted_counts)

        # 🔑 Decrypt for verification
        most_common_state = max(encrypted_counts, key=encrypted_counts.get)
        decrypted_counts = quantum_scramble_decrypt(unitary, key, most_common_state)
        print("  🔑 Decrypted Outcomes:", decrypted_counts)

    # Decrypt for demonstration (only works if reversal aligns perfectly)
    most_common_encrypted_state = max(encrypted_counts, key=encrypted_counts.get)
    decrypted_counts = quantum_scramble_decrypt(unitary, key, most_common_encrypted_state)
    print("🔑 Decrypted Signature Outcomes:", decrypted_counts)

def main_entropy_black_hole():
    """Test to see black hole simulation results"""
    qc, entropies = black_hole_warp_simulation_core()

    print("\n🕳️ Black Hole Warp Simulation Results 🕳️")
    print("========================================")
    
    # Print the quantum circuit
    print("\nQuantum Circuit:")
    print(qc)

    # Print Von Neumann entropy results
    print("\nVon Neumann Entropies for Each Qubit:")
    for i, entropy in enumerate(entropies):
        print(f" Qubit {i}: {entropy:.6f}")

    print("\n✅ Simulation complete!")

def main_qec_shor_simulated_black_hole():
    """experiment to see qec (quantum error correction) on the simulated black hole code"""
    qc, entropies = quantum_black_hole_simulation_with_qec()
    print("Quantum circuit:")
    print(qc)
    print("Von Neumann entropies after evolution:", entropies)

def main_multiversal_time_travel_simulator():
    """Basic time reversal test"""
    use_simulator = True  # Switch to hardware for later runs
    hardware_backend_name = "ibm_sherbrooke"

    # Initialize backend
    backend = initialize_backend(use_simulator=use_simulator, hardware_backend_name=hardware_backend_name)

    if backend:
        # Initialize base quantum circuit
        qc_base = create_base_circuit()

        # Number of iterations for causal feedback
        iterations = 5
        previous_results = None

        for i in range(iterations):
            print(f"\nIteration {i + 1}")

            # Apply causal feedback to the circuit
            qc_modified = add_causality_to_circuit(qc_base.copy(), previous_results, qubits=[0, 1])

            # Run the experiment
            results = run_and_extract_counts_quantum(qc=qc_modified, backend=backend, shots=8192)
            print("Results:", results)

            # Update previous results
            previous_results = results
    else:
        print("Failed to initialize backend.")

def main_run_bias_experiment(qc, backend=Aer.get_backend('aer_simulator'), shots=8192):
    """
    Runs the circuit with charge injection and compares entropy shifts.
    """
    # Ensure measurement is added
    qc.measure_all()

    # Run baseline circuit
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts_baseline = result.get_counts()

    # Apply charge injection
    qc_injected = apply_charge_injection(qc.copy(), range(qc.num_qubits))
    qc_injected.measure_all()
    transpiled_qc_injected = transpile(qc_injected, backend)
    result_injected = backend.run(transpiled_qc_injected, shots=shots).result()
    counts_injected = result_injected.get_counts()

    # Create a true random state circuit
    qc_random = generate_true_random_state(qc.num_qubits)
    qc_random.measure_all()
    transpiled_qc_random = transpile(qc_random, backend)
    result_random = backend.run(transpiled_qc_random, shots=shots).result()
    counts_random = result_random.get_counts()

    # Calculate Shannon Entropy
    def calculate_entropy(counts):
        total = sum(counts.values())
        probs = np.array([v / total for v in counts.values()])
        return -np.sum(probs * np.log2(probs))

    entropy_baseline = calculate_entropy(counts_baseline)
    entropy_injected = calculate_entropy(counts_injected)
    entropy_random = calculate_entropy(counts_random)

    print("Baseline Counts:", counts_baseline)
    print("Baseline Shannon Entropy:", entropy_baseline)
    print("\nInjected Counts:", counts_injected)
    print("Injected Shannon Entropy:", entropy_injected)
    print("\nRandom Counts:", counts_random)
    print("Random Shannon Entropy:", entropy_random)

    return counts_baseline, entropy_baseline, counts_injected, entropy_injected, counts_random, entropy_random

def main_analyze_temporal_correlation(use_simulator=True):
    """ Creates a circuit with causality added in and with quantum error correction to try to analyze temporal trends"""
    
    # Initialize backend
    backend = initialize_backend(use_simulator)

    if backend:
        # Create base circuit
        qc_base = create_base_circuit()
        qfi_matrix, eigenvalues = compute_qfi(qc_base)

        print("Quantum Fisher Information Matrix: ")
        print(qfi_matrix)
        print("Eigenvalues: ")
        print(eigenvalues)

        # Store results from iterations
        iteration_results = []

        for i in range(10):  # Increase iterations for temporal analysis
            print(f"\nIteration {i + 1}")

            # Modify circuit (e.g., add causality)
            qc_modified = add_causality_to_circuit(qc_base.copy(), iteration_results[-1] if iteration_results else None, qubits=[0, 1])

            qfi_matrix_mod, eigenvalues_mod = compute_qfi(qc_modified)

            print("Quantum Fisher Information Matrix: ")
            print(qfi_matrix_mod)
            print("Eigenvalues: ")
            print(eigenvalues_mod)
        
            qc_clb_mod = add_classical_bits(qc_modified, 0)

            qc_qec, instr_0 = apply_charge_preserving_qec_no_syndrome(qc_clb_mod, num_qubits=2, num_classical=2)

            # Run circuit and store results
            results = run_and_extract_counts_quantum(qc=qc_qec, backend=backend, shots=8192)
            iteration_results.append(results)

            # Analyze subsystem entropy
            entropies = calculate_subsystem_entropy(qc_qec)
            print(f"Subsystem Entropy: {entropies}")

        # Analyze temporal correlation
        temporal_divergences = analyze_temporal_correlation(iteration_results)
        print(f"Temporal Correlation (Jensen-Shannon Divergences): {temporal_divergences}")

        # Apply time-reversal and re-run
        qc_reversed = time_reverse_circuit(qc_base)

        qfi_matrix_rev, eigenvalues_rev = compute_qfi(qc_reversed)

        print("Quantum Fisher Information Matrix: ")
        print(qfi_matrix_rev)
        print("Eigenvalues: ")
        print(eigenvalues_rev)
        
        qc_reversed_clb = add_classical_bits(qc_reversed, 2)
        qc_qec_reversed, instr_1 = apply_charge_preserving_qec_no_syndrome(qc_reversed_clb, num_qubits=qc_reversed_clb.num_clbits, num_classical=qc_reversed_clb.num_clbits)
           
        reversed_results = run_and_extract_counts_quantum(qc=qc_qec_reversed, backend=backend, shots=8192)
        print(f"Time-Reversed Results: {reversed_results}")
    else:
        print("Failed to initialize backend.")


def main_analyze_temporal_correlation_with_amplification_qec(use_simulator=True):
    """ Creates a circuit with causality added in and with quantum error correction to try to analyze temporal trends"""
    
    # Initialize backend
    backend = initialize_backend(use_simulator)

    if backend:
        # Create base circuit
        qc_base = create_base_circuit()
        qc_base_amp = amplify_11_phase_balanced(qc_base)
        qfi_matrix, eigenvalues = compute_qfi(qc_base_amp)

        print("Quantum Fisher Information Matrix: ")
        print(qfi_matrix)
        print("Eigenvalues: ")
        print(eigenvalues)

        # Store results from iterations
        iteration_results = []

        for i in range(10):  # Increase iterations for temporal analysis
            print(f"\nIteration {i + 1}")

            # Modify circuit (e.g., add causality)
            qc_modified = add_causality_to_circuit(qc_base_amp.copy(), iteration_results[-1] if iteration_results else None, qubits=[0, 1])

            qfi_matrix_mod, eigenvalues_mod = compute_qfi(qc_modified)

            print("Quantum Fisher Information Matrix: ")
            print(qfi_matrix_mod)
            print("Eigenvalues: ")
            print(eigenvalues_mod)
        
            qc_clb_mod = add_classical_bits(qc_modified, 0)

            qc_qec, instr_0 = apply_charge_preserving_qec_no_syndrome(qc_clb_mod, num_qubits=2, num_classical=2)

            # Run circuit and store results
            results = run_and_extract_counts_quantum(qc=qc_qec, backend=backend, shots=8192)
            iteration_results.append(results)

            # Analyze subsystem entropy
            entropies = calculate_subsystem_entropy(qc_qec)
            print(f"Subsystem Entropy: {entropies}")

        # Analyze temporal correlation
        temporal_divergences = analyze_temporal_correlation(iteration_results)
        print(f"Temporal Correlation (Jensen-Shannon Divergences): {temporal_divergences}")

        # Apply time-reversal and re-run
        qc_reversed = time_reverse_circuit(qc_base_amp)
        
        qc_reversed_clb = add_classical_bits(qc_reversed, 2)
        qc_qec_reversed, instr_1 = apply_charge_preserving_qec_no_syndrome(qc_reversed_clb, num_qubits=qc_reversed_clb.num_clbits, num_classical=qc_reversed_clb.num_clbits)
           
        reversed_results = run_and_extract_counts_quantum(qc=qc_qec_reversed, backend=backend, shots=8192)
        print(f"Time-Reversed Results: {reversed_results}")
    else:
        print("Failed to initialize backend.")

def main_test_time_reversal_bias():
    """Perform a full test suite to analyze the time-reversal bias."""
    # 1. Baseline Circuit
    qc_base = construct_baseline_circuit()
    counts_base = run_circuit(qc_base)
    analyze_results(counts_base, "Baseline Circuit")
    
    # 2. Time-Reversed Circuit
    qc_time_reversed = apply_time_reversal(qc_base)
    qc_time_reversed.measure_all()
    counts_reversed = run_circuit(qc_time_reversed)
    analyze_results(counts_reversed, "Time-Reversed Circuit")
    
    # 3. Phase Distortion
    qc_phase_distorted = introduce_phase_distortion(qc_base)
    qc_phase_distorted.measure_all()
    counts_phase = run_circuit(qc_phase_distorted)
    analyze_results(counts_phase, "Phase-Distorted Circuit")
    
    # 4. Charge Injection Redirection
    qc_charge_injected = apply_charge_injection(qc_base)
    qc_charge_injected.measure_all()
    counts_charge = run_circuit(qc_charge_injected)
    analyze_results(counts_charge, "Charge-Injection Circuit")
    
    # 5. Holographic Encoding
    qc_holographic = apply_holographic_encoding(qc_base)
    qc_holographic.measure_all()
    counts_holographic = run_circuit(qc_holographic)
    analyze_results(counts_holographic, "Holographic Encoding Circuit")

def main_decision_influence(decision="Invest in quantum AI"):
    qc = QuantumCircuit(5)  # Create a holographic quantum circuit
    qc_encoded = record_decision(qc, decision)

    qc.measure_all()
    backend = Aer.get_backend('aer_simulator')
    transpiled_qc = transpile(qc_encoded, backend)
    result = backend.run(transpiled_qc, shots=1024).result()
    counts = result.get_counts()

    print("Encoded Circuit Measurement Results:", counts)

def main_iterative_decisions():
    # Example usage
    initial_decision = '101'  # Example decision in binary
    qc, qr, cr = initialize_qubits(len(initial_decision))
    encode_decision(qc, qr, initial_decision)
    apply_entanglement(qc, qr)
    measure_and_reset(qc, qr, cr)

    # Add subsequent decisions
    second_decision = '011'
    qc, qr, cr = add_decision_to_chain(qc, qr, cr, second_decision)
    measure_and_reset(qc, qr, cr)

    third_decision = '110'
    qc, qr, cr = add_decision_to_chain(qc, qr, cr, third_decision)
    measure_and_reset(qc, qr, cr)

    # Execute the final circuit
    final_counts = execute_circuit(qc)
    print("Final Measurement Results:", final_counts)

def main_disruptive_decision():
    decision_qc = encode_disruptive_decision("Invest in quantum AI")
    new_decision_qc = encode_disruptive_decision("Divest from classical AI")
    fidelity, prev_counts, new_counts = measure_decision_influence(decision_qc, new_decision_qc)
    print(f"Decision Influence Fidelity: {fidelity}")


def main_modify_and_run_quantum_experiment_multi_analysis_0(qc, backend=None, shots=8192, modify_circuit=None, analyze_results=None):
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

        simulator_backend = Aer.get_backend('aer_simulator')
        transpiled_qc = transpile(qc, backend=simulator_backend)
        transpiled_qc.measure_all()
        time_reversal_qc = time_reversal_simulation(transpiled_qc)
        time_reversal_qc.measure_all()
        job = simulator_backend.run(time_reversal_qc, backend=simulator_backend, shots=shots)
        result = job.result()
        counts = result.get_counts()
        print("Counts (simulated): ", counts)
        qc_no_measure = remove_measurements(qc)
        state = Statevector.from_instruction(qc_no_measure)
        analyze_shannon_entropy(counts)
        analyze_von_neumann_entropy(state)
        
        return counts
        
        # Use hardware backend for Von Nuemman entropy only
        counts = run_and_extract_counts_quantum(qc, backend, shots)
        


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

def main_modify_and_run_quantum_experiment_multi_analysis_1(qc, backend=None, shots=8192, modify_circuit=None, analyze_results=None):
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
        simulator_backend = Aer.get_backend('aer_simulator')
        transpiled_qc = transpile(qc, backend=simulator_backend)

        # Time-reversal modification
        qc_no_measure = remove_measurements(transpiled_qc)  # Ensure no classical bits
        print("DEBUG: Checking if measurements removed:", qc_no_measure.draw())

        try:
            time_reversal_qc = time_reversal_simulation(qc_no_measure)
        except Exception as e:
            print("Error in time reversal:", e)
            return

        time_reversal_qc.measure_all()  # Apply measurement AFTER inversion

        # Run on the simulator
        job = simulator_backend.run(time_reversal_qc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        print("Counts (simulated): ", counts)

        # Analyze entropy
        try:
            state = Statevector.from_instruction(qc_no_measure)  # Ensure no classical bits
            analyze_shannon_entropy(counts)
            analyze_von_neumann_entropy(state)
        except Exception as e:
            print("Error in entropy analysis:", e)

        return counts

def main_test_multiverse_branching(qc, backend=Aer.get_backend('aer_simulator'), shots=8192):
    """
    Adds an ancilla qubit to test if hidden multiversal influences affect quantum state probabilities.
    
    Parameters:
        qc (QuantumCircuit): The base circuit for the experiment.
        backend (Backend): The quantum simulator or real backend to run the test.
        shots (int): The number of times the circuit is executed.

    Returns:
        dict: Measurement counts from the experiment.
    """
    # Extend circuit by 1 qubit for the ancilla
    num_qubits = qc.num_qubits
    qc_test = QuantumCircuit(num_qubits + 1, num_qubits + 1)  # One extra qubit

    # Copy the original circuit into the first num_qubits
    qc_test.compose(qc, range(num_qubits), inplace=True)

    # Entangle the ancilla with a Bell-like state
    qc_test.h(num_qubits)  
    qc_test.cx(num_qubits, 0)  # Entangle with first qubit to test for influence

    # Measure all qubits
    qc_test.measure(range(num_qubits + 1), range(num_qubits + 1))

    # Transpile and execute
    transpiled_qc = transpile(qc_test, backend)
    job = backend.run(transpiled_qc, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    return counts

def main_run_multiverse_experiment(backend=Aer.get_backend('aer_simulator'), shots=32768):
    """
    Runs the multiverse test circuit on the given backend with automated result analysis.
    """
    qc = multiverse_test_circuit()
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()

    counts = result.get_counts()
    
    # Calculate Shannon Entropy
    total_shots = sum(counts.values())
    probabilities = [count / total_shots for count in counts.values()]
    shannon_entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)

    print("Counts:", counts)
    print("Shannon Entropy:", shannon_entropy)

    return counts, shannon_entropy

def main_run_random_state_experiment(num_qubits=3, backend=None, shots=32768):
    """Creates and runs a maximally random quantum circuit."""
    
    qc = QuantumCircuit(num_qubits)
    
    # Apply Hadamards to create uniform superposition
    qc.h(range(num_qubits))
    
    # Apply random phase shifts
    for qubit in range(num_qubits):
        random_angle = np.random.uniform(0, 2*np.pi)
        qc.append(RZGate(random_angle), [qubit])

    # Measure all qubits
    qc.measure_all()

    # Choose backend
    if backend is None:
        backend = Aer.get_backend('aer_simulator')

    # Transpile and execute
    transpiled_qc = transpile(qc, backend)
    job = backend.run(transpiled_qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Compute entropy
    entropy = calculate_shannon_entropy(counts, num_shots=shots)
    
    print("Random State Counts:", counts)
    print("Random State Shannon Entropy:", entropy)
    
    return counts, entropy

def main_run_time_reversed_experiment(qc, backend=None, shots=32768):
    """Runs a time-reversed version of the given quantum circuit."""
    
    # Remove measurements
    qc_no_measurements = qc.copy()
    qc_no_measurements.remove_final_measurements(inplace=True)
    
    # Time-reverse the circuit
    qc_reversed = qc_no_measurements.inverse()
    
    # Add measurements back
    qc_reversed.measure_all()

    # Choose backend
    if backend is None:
        backend = Aer.get_backend('aer_simulator')

    # Transpile and execute
    transpiled_qc = transpile(qc_reversed, backend)
    job = backend.run(transpiled_qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Compute entropy
    entropy = calculate_shannon_entropy(counts, num_shots=shots)
    
    print("Time-Reversed Counts:", counts)
    print("Time-Reversed Shannon Entropy:", entropy)
    
    return counts, entropy

def main_run_time_reversal_biasing_experiment(qc, backend=Aer.get_backend('aer_simulator'), shots=8192):
    """
    Runs a quantum time reversal experiment and tests if probability biasing can restore lost states.
    """
    results = {}

    # Remove measurements first (to avoid issues)
    qc_no_meas = qc.remove_final_measurements(inplace=False)

    # Run baseline circuit with measurement
    qc_baseline = qc_no_meas.copy()
    qc_baseline.measure_all()
    transpiled_qc = transpile(qc_baseline, backend)
    job = backend.run(transpiled_qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    if not counts:
        raise ValueError("Baseline circuit did not return counts. Check measurement setup.")

    # Compute entropy before reversal
    probs = np.array(list(counts.values())) / shots
    shannon_entropy = -np.sum(probs * np.log2(probs))

    # Time-reverse the circuit and ensure it has measurements
    qc_reversed = qc_no_meas.inverse()
    qc_reversed.measure_all()
    transpiled_qc_reversed = transpile(qc_reversed, backend)
    job_reversed = backend.run(transpiled_qc_reversed, shots=shots)
    result_reversed = job_reversed.result()
    counts_reversed = result_reversed.get_counts()

    if not counts_reversed:
        raise ValueError("Time-reversed circuit did not return counts. Check measurement setup.")

    # Compute entropy after time-reversal
    probs_reversed = np.array(list(counts_reversed.values())) / shots
    shannon_entropy_reversed = -np.sum(probs_reversed * np.log2(probs_reversed))

    # Apply probability biasing to restore lost states
    qc_biased = apply_probability_biasing(qc_reversed, bias_qubits=[0, 1, 2])
    qc_biased.measure_all()
    transpiled_qc_biased = transpile(qc_biased, backend)
    job_biased = backend.run(transpiled_qc_biased, shots=shots)
    result_biased = job_biased.result()
    counts_biased = result_biased.get_counts()

    if not counts_biased:
        raise ValueError("Biased circuit did not return counts. Check measurement setup.")

    # Compute entropy after biasing
    probs_biased = np.array(list(counts_biased.values())) / shots
    shannon_entropy_biased = -np.sum(probs_biased * np.log2(probs_biased))

    # Store results
    results["baseline_counts"] = counts
    results["baseline_entropy"] = shannon_entropy
    results["time_reversed_counts"] = counts_reversed
    results["time_reversed_entropy"] = shannon_entropy_reversed
    results["biased_counts"] = counts_biased
    results["biased_entropy"] = shannon_entropy_biased

    # Print results
    print("Baseline Counts:", counts)
    print("Baseline Shannon Entropy:", shannon_entropy)
    print("Time-Reversed Counts:", counts_reversed)
    print("Time-Reversed Shannon Entropy:", shannon_entropy_reversed)
    print("Biased Counts (After Amplification):", counts_biased)
    print("Biased Shannon Entropy:", shannon_entropy_biased)

    return results

def main_test_multiverse_1():
    counts, entropy = main_run_multiverse_experiment()
    qc = create_base_circuit()
    plot_histogram(counts)
    print("Counts: ", counts)
    print("Shannon entropy: ", entropy)
    counts, entropy = main_run_time_reversed_experiment(qc)
    plot_histogram(counts)
    print("Counts: ", counts)
    print("Shannon entropy: ", entropy)
    counts, entropy = main_run_random_state_experiment()
    plot_histogram(counts)
    print("Counts: ", counts)
    print("Shannon entropy: ", entropy)

def main_run_mwi_vs_holography_experiment(qc, backend=None, shots=8192, iterations=5):
    """
    Runs probability amplification multiple times on the same quantum circuit.
    Tracks how past biasing affects future probability distributions.
    
    Determines if Many-Worlds (MWI) or Holography is more likely.

    Parameters:
        qc (QuantumCircuit): The base quantum circuit.
        backend (Backend): The IBM backend or simulator for execution.
        shots (int): Number of shots per experiment.
        iterations (int): Number of times to repeat amplification.

    Returns:
        dict: A dictionary containing counts and entropy over multiple runs.
    """

    if backend is None:
        backend = Aer.get_backend('aer_simulator')

    results = {"counts": [], "entropy": []}

    for i in range(iterations):
        # Transpile and execute circuit
        transpiled_qc = transpile(qc, backend)
        job = backend.run(transpiled_qc, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Calculate Shannon entropy
        total_shots = sum(counts.values())
        probs = np.array([c / total_shots for c in counts.values()])
        entropy = -np.sum(probs * np.log2(probs))

        # Store results
        results["counts"].append(counts)
        results["entropy"].append(entropy)

        # Apply probability amplification
        qc = apply_probability_amplification(qc)  # Modify this based on your amplification function

    return results

def main_iterative_charge_injection(qc, num_cycles=5, backend=Aer.get_backend('aer_simulator'), shots=8192):
    """
    Applies charge injection iteratively and measures entropy growth over multiple cycles.

    Parameters:
        qc (QuantumCircuit): Initial quantum circuit.
        num_cycles (int): Number of charge injection cycles.
        backend (Backend): Qiskit backend for execution.
        shots (int): Number of shots for each experiment.

    Returns:
        list: Entropy values after each iteration.
    """
    entropies = []
    qc_modified = qc.copy()

    for i in range(num_cycles):
        # Apply charge injection (modify this based on your function)
        qc_modified = apply_charge_injection_universal(qc_modified)

        # Remove measurements before getting the statevector
        qc_no_measure = remove_measurements(qc_modified)

        # Simulate and get state vector
        transpiled_qc = transpile(qc_no_measure, backend)
        state = Statevector.from_instruction(transpiled_qc)

        measured_state = partial_trace(state, [0])  # Trace out one qubit
        measured_entropy = entropy(measured_state)

        print(f"Entropy after forced measurement: {measured_entropy}")

        partial_ent = compute_partial_entropy(state, [0])  # Trace out qubit 0

        print(f"Partial entropy of remaining system: {partial_ent}")
        
        # Compute von Neumann entropy
        entropy_val = entropy(state)
        entropies.append(entropy_val)
        print(f"Iteration {i+1}: Entropy = {entropy_val}")

    # Plot entropy growth
    plt.plot(range(1, num_cycles+1), entropies, marker='o', linestyle='-')
    plt.xlabel("Charge Injection Cycle")
    plt.ylabel("Von Neumann Entropy")
    plt.title("Entropy Growth Over Charge Injection Cycles")
    plt.show()

    return entropies

def main_analyze_tensor_network_structure(qc, qubit_pairs=None):
    """
    Analyzes the tensor network structure of a quantum circuit using an MPS-like representation.

    Parameters:
        qc (QuantumCircuit): The input quantum circuit.

    Returns:
        list: A list of reshaped tensors representing the MPS-like structure.
    """

    qc, noise_model = apply_decoherence(qc)
    # Remove measurements before extracting the statevector
    qc_no_meas = remove_measurements(qc)

    # Get statevector from the modified circuit
    state = Statevector.from_instruction(qc_no_meas).data

    # Get number of qubits
    num_qubits = int(np.log2(len(state)))

    # Reshape into MPS-like structure
    tensors = []
    current_state = state.reshape([2] * num_qubits)  # Reshape into a tensor

    for i in range(num_qubits):
        # Perform Singular Value Decomposition (SVD) to split entanglement
        U, S, Vh = np.linalg.svd(current_state.reshape(2**i, -1), full_matrices=False)

        # Ensure valid reshaping by checking dimensions
        if U.shape[0] >= 2 and U.shape[1] > 1:
            tensors.append(U.reshape([-1, 2, U.shape[-1]]))  # Reshape into tensor form
        else:
            tensors.append(U)  # Store as is if it's too small to reshape properly

        if len(S) > 1 and Vh.shape[0] > 1:
            current_state = np.dot(np.diag(S), Vh).reshape([-1, 2, Vh.shape[-1]])  # Update state
        else:
            current_state = np.dot(np.diag(S), Vh)  # Avoid reshaping issues

    return tensors

def main_qc():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    return qc

def main_qc_bias():
    qc_test = QuantumCircuit(3)
    qc_test.h([0, 1, 2])  # Basic superposition test circuit
    return qc_test

def main_qc_entangle():
    qc = QuantumCircuit(3)
    qc.h(0)          # Put first qubit in superposition
    qc.cx(0, 1)      # Entangle first and second qubits
    qc.cx(1, 2)
    return qc

def main_retrocausal_experiment(qc, backend=None, shots=8192):
    """
    Test for retrocausal effects by delaying measurement on part of the system.

    Parameters:
        qc (QuantumCircuit): The quantum circuit to test.
        backend (Backend): The backend for execution.
        shots (int): Number of measurement shots.

    Returns:
        dict: Comparison of different measurement orders.
    """
    use_sampler = backend and not isinstance(backend, AerSimulator)
    
    if backend is None:
        backend = AerSimulator()

    # Ensure circuit has classical bits
    num_qubits = qc.num_qubits
    if not qc.clbits:
        qc.add_register(qc.cregs == QuantumCircuit(num_qubits, num_qubits).cregs[0])

    def run_circuit(qc_modified):
        """Executes circuit using either Sampler or standard backend.run()."""
        transpiled_qc = transpile(qc_modified, backend)
        if use_sampler:
            sampler = Sampler(backend)
            job = sampler.run([transpiled_qc], shots=shots)
            result = job.result()
            print("Data structure: ", result)
            dont_sampler = False
            counts = extract_counts(result, dont_sampler)
        else:
            job = backend.run(transpiled_qc, shots=shots)
            counts = job.result().get_counts()
        return counts

    # Step 1: Full Measurement (Baseline)
    qc_full = qc.copy()
    qc_full.measure_all()
    counts_full = run_circuit(qc_full)

    # Step 2: Partial Measurement (Measure all except last qubit)
    qc_partial = qc.copy()
    if not qc_partial.clbits:
        qc_partial.add_register(ClassicalRegister(num_qubits))
    for i in range(num_qubits - 1):
        qc_partial.measure(i, i)
    counts_partial = run_circuit(qc_partial)

    # Step 3: Measure Remaining Qubit (After Delay)
    qc_delayed = qc_partial.copy()
    qc_delayed.measure(num_qubits - 1, num_qubits - 1)
    counts_delayed = run_circuit(qc_delayed)

    print("Baseline Counts (Full Measurement):", counts_full)
    print("Counts After Partial Measurement:", counts_partial)
    print("Counts After Delayed Measurement:", counts_delayed)

    return {
        "full_counts": counts_full,
        "partial_counts": counts_partial,
        "delayed_counts": counts_delayed
    }

def run_quantum_erasure_experiment(backend=None, shots=8192):
    """
    Runs the quantum erasure experiment to distinguish between Many-Worlds and Holography.
    
    Parameters:
        backend (Backend): The backend for execution. If None, uses a simulator.
        shots (int): Number of measurement shots.

    Returns:
        dict: Measurement results comparing information recovery.
    """

    # Step 1: Create a Quantum Circuit with 3 qubits (Bell pair + control qubit)
    qc = QuantumCircuit(3, 2)  # 3 qubits, 2 classical bits for measurement

    # Step 2: Create a Bell pair between q0 and q1
    qc.h(0)
    qc.cx(0, 1)

    # Step 3: Introduce the Erasure Mechanism (Entangle q1 with an ancilla q2)
    qc.cx(1, 2)
    qc.h(1)  # Interfere before measurement
    
    # Step 4: Measure q2 (attempted erasure)
    qc.measure(2, 1)  

    # Step 5: Introduce the "Time-Reversal" Decision Mechanism (Decides if Erasure was Real)
    qc.cx(1, 0)  # Reverse effect of CX if conditions allow
    qc.h(1)  

    # Step 6: Measure final qubits
    qc.measure(0, 0)

    # Check if we are running on a real backend
    if backend is not None:
        sampler = Sampler()
        transpiled_qc = transpile(qc, backend)
        job = sampler.run(transpiled_qc, shots=shots)
        result = job.result()
        counts = result.quasi_dists[0].nearest_probability_distribution()
    else:
        # Use simulator to analyze full quantum state
        simulator = AerSimulator()
        qc_no_measure = qc.remove_final_measurements(inplace=False)
        state = Statevector.from_instruction(qc_no_measure)
        counts = state.probabilities_dict()

    print("Quantum Erasure Experiment Results:", counts)
    return counts

def main_charge_injection_entangled(qc, num_levels=5):
    """
    Applies charge injection with increasing intensity while ensuring proper entanglement.

    Parameters:
        qc (QuantumCircuit): The base quantum circuit.
        num_levels (int): The number of charge injection levels.

    Returns:
        QuantumCircuit: The modified quantum circuit.
    """
    num_qubits = qc.num_qubits
    entropies = []

    # Step 1: Initial Entanglement
    for i in range(num_qubits):
        qc.h(i)  # Put all qubits into superposition

    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)  # Entangle neighboring qubits

    # Step 2: Iterative Charge Injection with Controlled Phase Shifts
    for level in range(1, num_levels + 1):
        angle = np.pi / (2**level)  # Decreasing impact per level
        for i in range(num_qubits):
            qc.crz(angle, i, (i + 1) % num_qubits)  # Controlled Rz for scaling charge

    # Step 3: Optional Basis Rotation (Uncomment if needed)
        for i in range(num_qubits):
            qc.h(i)  # Rotate to Hadamard basis for more diverse measurement outcomes

        state = Statevector.from_instruction(qc)
        ent = entropy(state)
        entropies.append(ent)
        print(f"Charge Level {level}: Entropy = {ent}")

    return qc

def main_compare_charge_vs_random(num_qubits=5, depth=10, max_levels=5):
    """Compare entropy of charge-injected circuit vs. randomized circuit."""
    qc_base = QuantumCircuit(num_qubits)
    qc_base.h(0)  # Initial state
    qc_base.cx(0, 1)
    qc_base.cx(1, 2)

    print("\nRunning Charge Injection Scaling Test...")
    charge_results = charge_injection_scaling(qc_base, max_levels)

    print("\nRunning Randomized Circuit Test...")
    qc_random = create_randomized_circuit(num_qubits, depth)
    state_random = Statevector.from_instruction(qc_random)
    random_entropy = entropy(state_random)
    print(f"Randomized Circuit Entropy: {random_entropy}")

    return charge_results, random_entropy

def create_teleportation_circuit_without_measurements():
    """
    Creates the teleportation circuit without measurements for statevector analysis.
    """
    qc = QuantumCircuit(3)  # No classical bits since we're skipping measurement

    # Step 1: Create entangled Bell pair between Q1 and Q2
    qc.h(1)
    qc.cx(1, 2)

    # Step 2: Encode charge information into Q0
    qc.rx(1.57, 0)  

    # Step 3: Bell measurement preparation (without actually measuring)
    qc.cx(0, 1)
    qc.h(0)

    return qc

def main_run_quantum_teleportation_experiment(backend=None, shots=8192):
    """
    Runs the teleportation circuit on the specified backend and analyzes fidelity.
    """
    if backend is None:
        backend = AerSimulator(method="statevector")  # Explicitly use statevector mode

    # Step 1: Create a circuit without measurements
    qc_no_measure = create_teleportation_circuit_without_measurements()

    # Step 2: Run statevector simulation
    state_simulator = AerSimulator(method="statevector")
    transpiled_qc = transpile(qc_no_measure, state_simulator)
    transpiled_qc.save_statevector()  # Explicitly save the statevector

    result = state_simulator.run(transpiled_qc).result()

    # Ensure statevector is correctly retrieved
    try:
        statevector = result.get_statevector()
    except Exception as e:
        print(f"Error retrieving statevector: {e}")
        return None, None

    # Step 3: Convert statevector to density matrix
    final_density_matrix = DensityMatrix(statevector)

    # Step 4: Compute fidelity between the initial and final state
    initial_qc = QuantumCircuit(1)
    initial_qc.h(0) # Apply RX gate properly
    initial_state = Statevector.from_instruction(initial_qc)  # Convert to statevector

    num_qubits = final_density_matrix.num_qubits
    print(f"Total qubits in system: {num_qubits}")

    if num_qubits == 3:
        traced_state = partial_trace(final_density_matrix, [2])  # Keep only qubit 2
        initial_density_matrix = DensityMatrix(initial_state)  # Convert to density matrix
        # Trace out an extra qubit if necessary
        if traced_state.num_qubits > 1:
            traced_state = partial_trace(traced_state, [1])  # Reduce to single-qubit state

        # Compute fidelity with properly reduced states
        initial_density_matrix = DensityMatrix(initial_state)  # Convert to density matrix
        fidelity = state_fidelity(initial_density_matrix, traced_state)

        print(f"Traced state: {traced_state}")
    else:
        print("Error: Qubit index 2 is out of range.")
        fidelity = None

    # Step 5: Run the full experiment with measurements to get classical counts
    qc = create_teleportation_circuit()
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()

    # Step 6: Apply manual corrections for each measurement result
    corrected_fidelity_sum = 0
    for measurement, count in counts.items():
        corrected_state = apply_classical_corrections(measurement, traced_state)
        fidelity = state_fidelity(initial_state, corrected_state)
        corrected_fidelity_sum += fidelity * (count / shots)  # Weight by probability

    print("Quantum Teleportation Results:")
    print("Measurement Counts:", counts)
    print(f"Corrected State Fidelity: {corrected_fidelity_sum:.6f}")

    return counts, corrected_fidelity_sum


def main_run_charge_transfer_experiment(backend=None, shots=8192):
    if backend is None:
        backend = AerSimulator(method="automatic")  # Force automatic mode

    # Create two versions of the circuit: one WITH measurement, one WITHOUT
    qc_no_measure = charge_transfer_experiment(measure=False)
    qc_with_measure = charge_transfer_experiment(measure=True)

    print("\n🚀 Quantum Circuit Before Transpilation (No Measurement):")
    print(qc_no_measure)

    # Step 1: Run Statevector Simulation (on circuit WITHOUT measurement)
    state_simulator = AerSimulator(method="automatic")  # Ensure correct simulator
    transpiled_qc = transpile(qc_no_measure, state_simulator)

    print("\n🚀 Quantum Circuit After Transpilation (No Measurement):")
    print(transpiled_qc)

    result = state_simulator.run(transpiled_qc).result()

    # Try retrieving the statevector safely
    try:
        statevector = result.data(0)["statevector"]
    except Exception as e:
        print(f"\n❌ Error retrieving statevector: {e}")
        return None, None

    # Convert to density matrix for tracing analysis
    final_density_matrix = DensityMatrix(statevector)

    # Extract Q2's state to see if charge was transferred
    traced_state = partial_trace(final_density_matrix, [0])  # Keep only Q2

    # Step 2: Run actual measurement simulation
    transpiled_qc = transpile(qc_with_measure, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()

    print("\n✅ Charge Transfer Experiment Results:")
    print("Measurement Counts on Q2:", counts)
    print("Extracted State (Q2 after tracing Q1):", traced_state)

    return counts, traced_state


def main_run_charge_transfer_experiment_3qubits(backend=None, shots=8192):
    if backend is None:
        backend = AerSimulator(method="statevector")  # Ensure correct simulator

    # Create two versions of the circuit: one WITH measurement, one WITHOUT
    qc_no_measure = charge_transfer_experiment_3qubits(measure=False)
    qc_with_measure = charge_transfer_experiment_3qubits(measure=True)

    print("\n🚀 Quantum Circuit Before Transpilation (No Measurement):")
    print(qc_no_measure)

    # Step 1: Run Statevector Simulation (on circuit WITHOUT measurement)
    state_simulator = AerSimulator(method="statevector")
    transpiled_qc = transpile(qc_no_measure, state_simulator)

    print("\n🚀 Quantum Circuit After Transpilation (No Measurement):")
    print(transpiled_qc)

    # ✅ Run the circuit with explicit statevector saving
    result = state_simulator.run(transpiled_qc).result()

    # ✅ Retrieve Statevector using `data()`
    try:
        statevector = result.data(0)["statevector"]
    except Exception as e:
        print(f"\n❌ Error retrieving statevector: {e}")
        return None, None

    # Convert to density matrix for tracing analysis
    final_density_matrix = DensityMatrix(statevector)

    # Extract Q3's state to see if charge was transferred
    traced_state = partial_trace(final_density_matrix, [0, 1])  # Keep only Q3

    # Step 2: Run actual measurement simulation
    transpiled_qc = transpile(qc_with_measure, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()

    print("\n✅ 3-Qubit Charge Transfer Experiment Results:")
    print("Measurement Counts on Q3:", counts)
    print("Extracted State (Q3 after tracing Q1 & Q2):", traced_state)

    return counts, traced_state

def main_run_mbqc_energy_transfer_experiment(backend=None, shots=8192):
    if backend is None:
        backend = AerSimulator(method="statevector")  # Use Statevector Simulator

    # Create two versions of the circuit: one WITH measurement, one WITHOUT
    qc_no_measure = mbqc_energy_transfer_experiment(measure=False)
    qc_with_measure = mbqc_energy_transfer_experiment(measure=True)

    print("\n🚀 Quantum Circuit Before Transpilation (No Measurement):")
    print(qc_no_measure)

    # Step 1: Run Statevector Simulation (on circuit WITHOUT measurement)
    state_simulator = AerSimulator(method="statevector")
    transpiled_qc = transpile(qc_no_measure, state_simulator)

    print("\n🚀 Quantum Circuit After Transpilation (No Measurement):")
    print(transpiled_qc)

    # ✅ Run the circuit and retrieve statevector
    result = state_simulator.run(transpiled_qc).result()

    try:
        statevector = result.data(0)["statevector"]
    except Exception as e:
        print(f"\n❌ Error retrieving statevector: {e}")
        return None, None

    # Convert to density matrix for tracing analysis
    final_density_matrix = DensityMatrix(statevector)

    # Extract Earth Qubit's State After Space Qubit is Measured
    traced_state = partial_trace(final_density_matrix, [0])  # Keep only Q_earth

    qc_with_measure.h(0)  # Switch measurement to X-basis
    qc_with_measure.cx(0, 1)
    qc_with_measure.measure(0, 0)  # Measure Q_space in X-basis

    # Step 2: Run actual measurement simulation
    transpiled_qc = transpile(qc_with_measure, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()


    # ✅ Fix: Apply Correction Using Operator(ZGate())
    x_operator = Operator(XGate())  # Properly format the Z and X gate as an operator
    z_operator = Operator(ZGate())

    if counts.get('01', 0) > counts.get('00', 0):  
        corrected_state = traced_state.evolve(x_operator)  # Apply X correction
    elif counts.get('10', 0) > counts.get('00', 0):  
        corrected_state = traced_state.evolve(z_operator)  # Apply Z correction
    elif counts.get('11', 0) > counts.get('00', 0):  
        corrected_state = traced_state.evolve(x_operator).evolve(z_operator)  # Apply X + Z correction
    else:
        corrected_state = traced_state  # No correction needed

    print("\n✅ MBQC Energy Transfer Experiment Results:")
    print("Measurement Counts on Space Qubit:", counts)
    print("Extracted State (Earth Qubit after Space Measurement):", traced_state)
    print("✅ Corrected State (Earth Qubit after Space Measurement & Correction):", corrected_state)

    return counts, corrected_state

def main_multiversal_telephone(num_branches=3, backend=None, shots=8192):
    """
    Selects and aligns a preferred reality branch using quantum encoding and charge injection.
    
    Parameters:
    - num_branches: Number of possible reality states to select from.
    - backend: Quantum backend to use (default: AerSimulator).
    - shots: Number of runs for statistical reinforcement.
    
    Returns:
    - Selected branch and alignment status.
    """

    if backend is None:
        backend = AerSimulator(method="statevector")

    # Step 1: Generate branch options
    branch_options = generate_reality_branches(num_branches)

    # Step 2: Create quantum registers (one for each branch option)
    qr = QuantumRegister(num_branches, name="q")
    cr = ClassicalRegister(num_branches, name="c")
    qc = QuantumCircuit(qr, cr)

    # Step 3: Encode each branch option into a quantum state
    for branch in branch_options:
        qc.rx(branch["charge"], qr[branch["branch_id"]])  # Inject charge to encode preference

    # Step 4: Apply entanglement across branches (biasing toward coherence)
    for i in range(num_branches - 1):
        qc.cx(qr[i], qr[i+1])

    # Step 5: Weak measurement to determine highest coherence state
    qc.measure(qr, cr)

    # Step 6: Run the experiment
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()

    # Step 7: Interpret results and align charge injection
    selected_branch = max(counts, key=counts.get)  # Pick the most likely outcome
    print(f"\n📞 Selected Reality Branch: {selected_branch}")
    print("Measurement Counts:", counts)

    # Step 8: Inject charge to reinforce the selected branch
    aligned_state = Statevector.from_instruction(qc)
    final_density_matrix = DensityMatrix(aligned_state)
    traced_state = partial_trace(final_density_matrix, [int(selected_branch, 2)])

    print("\n✅ Aligned State After Selection:", traced_state)

    return selected_branch, traced_state

def main_cleanup_charge_injection(backend=None, shots=8192):
    """
    Reverses past charge injections, removing unintended biases and resetting entanglements.

    Parameters:
    - backend: Quantum backend to use (default: AerSimulator).
    - shots: Number of runs for statistical reinforcement.

    Returns:
    - Cleansed quantum state after charge extraction.
    """

    if backend is None:
        backend = AerSimulator(method="statevector")

    # Create a cleanup quantum circuit
    num_qubits = 3  # Default to 3, can be adjusted
    qr = QuantumRegister(num_qubits, name="q")
    qc = QuantumCircuit(qr)

    # Step 1: Apply inverse charge extraction to neutralize residual effects
    for qubit in range(num_qubits):
        qc.rx(-np.pi / 2, qr[qubit])  # Apply inverse RX to remove charge buildup

    # Step 2: Run the circuit and extract the cleansed state
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()

    # Convert to density matrix for final verification
    cleaned_state = DensityMatrix.from_instruction(qc)

    print("\n✅ Charge Cleanup Completed. Residual biases removed.")
    print("Cleaned Quantum State:\n", cleaned_state)

    return cleaned_state


if __name__ == "__main__":
    #back = initialize_backend(use_simulator=False)
    main_cleanup_charge_injection()
