from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, partial_trace, entropy
from qiskit_aer import Aer
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram


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


def targeted_gates_111(target_state="111", shots=2048):
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

def mirrored_state_prep(qc, target_state):
    """
    Prepare a mirrored state based on the target state.
    
    Parameters:
    - qc (QuantumCircuit): The circuit to apply the mirrored state preparation.
    - target_state (str): Target state as a binary string (e.g., "101").
    """
    for i, bit in enumerate(reversed(target_state)):
        if bit == "1":
            qc.x(i)  # Apply X gate for '1' bits
    qc.h(range(len(target_state)))  # Create superposition

def run_ejection_tool(backend_type="simulator", target_state="101", shots=1024):
    """
    Runs the mirrored ejection tool with specified backend and target state.
    Args:
        backend_type (str): Type of backend ("simulator" or "quantum").
        target_state (str): Target state for the ejection.
        shots (int): Number of shots for the execution.
    """
    n_qubits = len(target_state)
    qc = QuantumCircuit(n_qubits)

    # Apply mirrored ejection logic
    qc = mirrored_eject(qc, target_state)

    # Select backend
    if backend_type == "simulator":
        backend = AerSimulator()
        print("Running on AerSimulator...")
    elif backend_type == "quantum":
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub="ibm-q")
        backend = provider.get_backend("ibm_nairobi")  # Example quantum backend
        print(f"Running on quantum backend: {backend.name()}...")
    else:
        raise ValueError("Invalid backend_type. Use 'simulator' or 'quantum'.")

    # Execute ejection
    results = execute_eject(qc, backend, shots)
    print("Ejection Results:", results)

def mirrored_eject(qc, target_state):
    """
    Ejects a multiversal communication chain by collapsing mirrored states.
    Args:
        qc (QuantumCircuit): The quantum circuit representing the interaction.
        target_state (str): The state to anchor the ejection logic on.
    Returns:
        QuantumCircuit: The modified circuit with ejection logic applied.
    """
    n_qubits = len(target_state)
    for i, bit in enumerate(target_state):
        if bit == '1':
            qc.x(i)  # Apply X gate to flip the qubit
        qc.h(i)  # Apply Hadamard to balance probabilities
        qc.rz(-1 * (3.14 / 2), i)  # Simulate mirrored collapse effect

    # Add measurements to observe the collapse
    qc.measure_all()
    return qc
    
def run_experiment(backend_type, target_state="101", t_steps=5, shots=2048):
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

def run_mirrored_experiment(backend_type, target_state="101", t_steps=5, shots=2048):
    """
    Run a mirrored state experiment with time-dependent holographic operations.
    """
    # Select backend
    if backend_type == "simulator":
        backend = Aer.get_backend('aer_simulator')
        print("\nRunning on AerSimulator...")
    elif backend_type == "quantum":
        service = QiskitRuntimeService(channel="ibm_quantum")  # Load IBM Quantum account
        filters = lambda b: b.configuration().n_qubits >= 3 and b.status().operational
        backend = service.backend("ibm_brisbane")
        if backend is None:
            raise ValueError("No backends are available at the moment...")
        print(f"\nRunning on quantum backend: {backend.name}...")
    else:
        raise ValueError("Invalid backend_type. Use 'simulator' or 'quantum'.")

    n_qubits = len(target_state)
    qc_um = QuantumCircuit(n_qubits, n_qubits)  # Create the quantum circuit

    # Step 1: Mirrored State Preparation
    mirrored_state_prep(qc_um, target_state)

    # Step 2: Time-dependent holographic operations
    for t in range(1, t_steps + 1):
        angle_timeline = np.pi * t / 10
        angle_holographic = np.pi * t / 15

        qc_um.rz(angle_timeline, 0)  # Timeline distortion on Black Hole qubit
        qc_um.rx(angle_holographic, 1)  # Holographic interaction on Radiation qubit
        qc_um.cx(1, 2)  # Correlate Radiation with Environment

    # Step 3: Add measurement
    qc_measured = qc_um.copy()
    qc_measured.measure(range(n_qubits), range(n_qubits))

    # Transpile and run
    transpiled_qc = transpile(qc_um, backend=backend, optimization_level=3)
    print(f"Transpiled Circuit Depth: {transpiled_qc.depth()}")

    counts = {}

    if backend_type == "simulator":
        result = backend.run(transpiled_qc, shots=shots).result()
        counts = result.get_counts()
    elif backend_type == "quantum":
        with Session(backend=backend) as session:
            sampler = Sampler()
            job = sampler.run([transpiled_qc])
            result = job.result()
            try:
            # Navigate to the `BitArray` and extract counts
                pub_result = result._pub_results[0]  # Access the first `SamplerPubResult`
                data_bin = pub_result.data  # Access `DataBin`
                bit_array = data_bin.c
                print("Bit array: ", bit_array)
                count = extract_counts_from_bitarray(bit_array)
                for raw_outcome in bit_array:
                    bitstring = f"{raw_outcome:0{num_bits}b}"
                    counts[bitstring] = counts.get(bitstring, 0) + 1

            # Debugging output
            except Exception as e:
                print(e)
                
    print("Processed Counts:", counts)

    # Ensure array compatibility
    array = np.array(list(counts.values()))
    if array.ndim == 1:
        array = np.expand_dims(array, axis=0) 

    print("Unmeasured Circuit:\n", qc_um.draw())
    print("Measured Circuit:\n", qc_measured.draw())


# Main Execution
if __name__ == "__main__":
##    t_steps = 5  # Number of time steps for dynamic evolution
##    qc = dynamic_warp_circuit_no_measure(t_steps)
##
##    # Run the simulation and get results
##    results = run_quantum_simulation(qc)
##    print("Results:", results)
##
##    # Analyze the probabilities
##    probabilities = analyze_results(results)
##
##    # Analyze entropies
##    state = Statevector.from_instruction(qc)  # Get statevector of the circuit
##    entropies = calculate_entropies(state)
##    print("\nSubsystem Entropies:", entropies)

    run_experiment(backend_type="quantum", target_state="111", t_steps=6, shots=2048)
