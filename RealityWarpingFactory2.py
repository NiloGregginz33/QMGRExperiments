from qiskit import QuantumCircuit, transpile, ClassicalRegister
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

def targeted_gates_111_2(target_state="111", shots=2048):
    """
    Design a circuit with targeted gates to amplify a specific state.
    """
    n_qubits = 3
    qc = QuantumCircuit(n_qubits)

    # Initial superposition
    qc.h(range(n_qubits))  # Create equal superposition of all states

    # Targeted phase shifts to amplify |111>
    if target_state == "111":
        qc.ccx(0, 1, 2)     # Apply Toffoli gate to mark |111>
        qc.mcp(np.pi, [0, 1], 2)  # Multi-controlled phase to reinforce |111|

    # Additional controlled gates for steering probability
    qc.cp(np.pi, 0, 1)  # Controlled phase gate
    qc.cx(1, 2)         # Controlled NOT
    qc.cz(0, 2)         # Controlled Z

    # Copy circuit for analysis without measurement
    qc_no_measurements = qc.copy()

    # Add measurement gates
    qc.add_register(ClassicalRegister(n_qubits))
    qc.measure_all()

    # Analyze the statevector (before measurement)
    state = Statevector.from_instruction(qc_no_measurements)
    print("\nStatevector:")
    print(state)

    # Calculate subsystem entropy
    rho_0 = partial_trace(state, [1, 2]).to_operator().data
    rho_1 = partial_trace(state, [0, 2]).to_operator().data
    rho_2 = partial_trace(state, [0, 1]).to_operator().data

    entropy_0 = entropy(rho_0, base=2)
    entropy_1 = entropy(rho_1, base=2)
    entropy_2 = entropy(rho_2, base=2)
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

    return counts  # Return counts for further analysis if needed

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
 # Pass the circuit as an argument

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

                counts = bit_array.to_dict()
                print("Results: ", counts)

            except Exception as e:
                print(e)

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


def run_experiment_2(backend_type, target_state="111", t_steps=5, shots=2048):
    """
    Run a mirrored state experiment with time-dependent holographic operations and targeted state amplification.
    """
    # Select backend
    if backend_type == "simulator":
        backend = Aer.get_backend('aer_simulator')
        print("\nRunning on AerSimulator...")
    elif backend_type == "quantum":
        service = QiskitRuntimeService(channel="ibm_quantum")
        filters = lambda b: b.configuration().n_qubits >= 3 and b.status().operational
        backend = least_busy_backend(service, filters)
        if backend is None:
            raise ValueError("No quantum backends available.")
        print(f"\nRunning on quantum backend: {backend.name}...")

    n_qubits = len(target_state)
    qc = QuantumCircuit(n_qubits, n_qubits)  

    # Step 1: Superposition (Equal Probability)
    qc.h(range(n_qubits))

    # Step 2: Time-dependent holographic operations
    for t in range(1, t_steps + 1):
        angle_timeline = np.pi * t / 10
        angle_holographic = np.pi * t / 15

        qc.rz(angle_timeline, 0)  # Timeline distortion on Black Hole qubit
        qc.rx(angle_holographic, 1)  # Holographic interaction on Radiation qubit
        qc.cx(1, 2)  # Correlate Radiation with Environment

    # Step 3: Explicit `|111⟩` Amplification
    qc.mcx([0, 1], 2)  # Toffoli gate: Marks `111`
    qc.z(2)  # Phase flip on `111`
    
    # Step 4: Grover Diffusion (Pulls Probability to `111`)
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(2)
    qc.mcx([0, 1], 2)  # Reflect over `|000>`
    qc.h(2)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))

    # Step 5: Add Measurement
    qc.measure(range(n_qubits), range(n_qubits))

    # Transpile and Run
    transpiled_qc = transpile(qc, backend=backend, optimization_level=3)
    print(f"Transpiled Circuit Depth: {transpiled_qc.depth()}")

    if backend_type == "simulator":
        result = backend.run(transpiled_qc, shots=shots).result()
        counts = result.get_counts()
    elif backend_type == "quantum":
        with Session(backend=backend) as session:
            sampler = Sampler()
            job = sampler.run([transpiled_qc])
            result = job.result()
            print("Result: ", result)
            try:
                pub_result = result._pub_results[0]
                data_bin = pub_result.data
                bit_array = data_bin.c
                counts = bit_array.to_dict()
                print("Results: ", counts)
            except Exception as e:
                print(e)

    # Plot and Return Results
    plot_histogram(counts)
    plt.title(f"Measurement Results (Targeting {target_state})")
    plt.show()

    return counts

def run_experiment_3(backend_type, target_state="11", t_steps=5, shots=2048):
    """
    Run a mirrored state experiment with time-dependent holographic operations 
    and targeted state amplification for `|11⟩`.
    """
    # Select backend
    if backend_type == "simulator":
        backend = Aer.get_backend('aer_simulator')
        print("\nRunning on AerSimulator...")
    elif backend_type == "quantum":
        service = QiskitRuntimeService(channel="ibm_quantum")
        filters = lambda b: b.configuration().n_qubits >= 2 and b.status().operational
        backend = least_busy_backend(service, filters)
        if backend is None:
            raise ValueError("No quantum backends available.")
        print(f"\nRunning on quantum backend: {backend.name}...")

    # Adjust qubit count for `|11⟩` (2-qubit target state)
    n_qubits = len(target_state)
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Step 1: Superposition (Equal Probability)
    qc.h(range(n_qubits))

    # Step 2: Time-dependent holographic operations
    for t in range(1, t_steps + 1):
        angle_timeline = np.pi * t / 10
        angle_holographic = np.pi * t / 15

        qc.rz(angle_timeline, 0)  # Timeline distortion on Qubit 0
        qc.rx(angle_holographic, 1)  # Holographic interaction on Qubit 1
        qc.cx(0, 1)  # Correlate the two qubits

    # Step 3: Explicit `|11⟩` Amplification
    qc.cx(0, 1)  # Controlled-NOT to mark `|11⟩`
    qc.z(1)  # Phase flip on `|11⟩`
    
    # Step 4: Grover Diffusion (Pull Probability to `|11⟩`)
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(1)
    qc.cz(0, 1)  # Reflect over `|00>` in 2-qubit space
    qc.h(1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))

    # Step 5: Add Measurement
    qc.measure(range(n_qubits), range(n_qubits))

    # Transpile and Run
    transpiled_qc = transpile(qc, backend=backend, optimization_level=3)
    print(f"Transpiled Circuit Depth: {transpiled_qc.depth()}")

    if backend_type == "simulator":
        result = backend.run(transpiled_qc, shots=shots).result()
        counts = result.get_counts()
    elif backend_type == "quantum":
        with Session(backend=backend) as session:
            sampler = Sampler()
            job = sampler.run([transpiled_qc])
            result = job.result()
            print("Result: ", result)
            try:
                pub_result = result._pub_results[0]
                data_bin = pub_result.data
                bit_array = data_bin.c
                counts = bit_array.to_dict()
                print("Results: ", counts)
            except Exception as e:
                print(e)

    # Plot and Return Results
    plot_histogram(counts)
    plt.title(f"Measurement Results (Targeting {target_state})")
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

def run_mirrored_experiment(backend_type, target_state="111", t_steps=5, shots=2048):
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
        print("Running on AerSimulator...")
    elif backend_type == "quantum":
        QiskitRuntimeService.save_account(channel="ibm_quantum")
        service = QiskitRuntimeService()
        backend = service.least_busy(
            filters=lambda b: b.configuration().n_qubits >= 2 and b.status().operational
        )
        print(f"Running on quantum backend: {backend.name}...")
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
    entropies = analyze_subsystem_entropy(state)

    print("Subsystem Entropies:", entropies)
    return counts, entropies

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

# Main Execution
if __name__ == "__main__":

    t_steps = 5  # Number of time steps for dynamic evolution
##    qc = dynamic_warp_circuit_no_measure(t_steps)
    qc = create_holographic_timeline_circuit()
    # Run the simulation and get results
    results = run_holographic_timeline_circuit(qc)
    print("Results:", results)

    # Analyze the probabilities
    probabilities = analyze_results(results)

    # Analyze entropies
##    state = Statevector.from_instruction(qc)  # Get statevector of the circuit
##    entropies = calculate_entropies(state)
##    print("\nSubsystem Entropies:", entropies)

    run_experiment_3(backend_type="simulator", target_state="111", t_steps=6, shots=2048)

##    # Example Execution
##    target_state = "11"  # Define the target state
##    holographic_circuit = create_holographic_timeline_circuit(target_state=target_state)
##    results, entropies = run_holographic_timeline_circuit(holographic_circuit, backend_type="simulator", shots=1024)
##    print("Results:", results)
##    print("Subsystem Entropies:", entropies)
