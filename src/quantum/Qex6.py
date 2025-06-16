#Many worlds
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
from qiskit.quantum_info import Statevector, partial_trace, entropy
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np

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

def print_experiment_goals():
    """
    Prints the goals and purpose of the experiment.
    """
    print("\nExperiment Goals:")
    print("- Simulate the Many-Worlds interpretation of quantum mechanics.")
    print("- Use a black hole qubit in superposition, entangled with Hawking radiation.")
    print("- Introduce decoherence to mimic wavefunction branching.")
    print("- Measure branching probabilities and analyze subsystem entropy.")
    print("- Investigate the role of entanglement in preserving information transfer.\n")

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

def run_experiment(backend_type="simulator", shots=8192):
    """
    Run the Many-Worlds experiment on either a simulator or a real quantum backend.

    Parameters:
        backend_type (str): "simulator" to use AerSimulator, "quantum" for a real quantum backend.
        shots (int): Number of measurement shots.

    Returns:
        None: Displays results and analysis.
    """
    # Step 1: Initialize the quantum circuit
    n_qubits = 2  # 1 qubit for the analog black hole, 1 for Hawking radiation
    qc = QuantumCircuit(n_qubits)

    # Step 2: Prepare the black hole qubit in a superposition state
    qc.h(0)  # Apply Hadamard gate to create |0> + |1>

    # Step 3: Entangle the black hole qubit with the Hawking radiation qubit
    qc.cx(0, 1)  # CNOT gate creates entanglement

    # Optional Step 4: Simulate decoherence by adding noise
    decoherence_rate = 0.1
    qc.rx(2 * np.pi * decoherence_rate, 0)  # Rotate black hole qubit slightly to simulate decoherence
    qc.rx(2 * np.pi * decoherence_rate, 1)  # Rotate radiation qubit slightly

    state = Statevector.from_instruction(qc)

    # Calculate subsystem entropy
    subsystem_entropy = entropy(state, [0])  # Entropy of the black hole qubit
    print("\nSubsystem Entropy of the Black Hole Qubit:")
    print(subsystem_entropy)
    

    # Step 5: Measure both qubits
    qc_with_measurements = qc.copy()
    qc_with_measurements.measure_all()

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

    # Transpile the circuit
    transpiled_qc = transpile(qc_with_measurements, backend)

    # Run the circuit
    if backend_type == "simulator":
        result = backend.run(transpiled_qc, shots=shots).result()
        counts = result.get_counts()
    else:
        with Session(backend=backend) as session:
        # Use Qiskit Runtime Sampler for quantum backend
            sampler = Sampler()
            job = sampler.run([transpiled_qc])
            result = job.result()
            counts = process_sampler_result(result, shots)

    # Display results
    print("\nMeasurement Results:")
    print(counts)

    print("\nSubsystem Entropy of the Black Hole Qubit:")
    print(entropy(state, [0]))

    print("\nBranching Probabilities (Many-Worlds Perspective):")
    branching_analysis = {key: value / shots for key, value in counts.items()}
    for state, prob in branching_analysis.items():
        print(f"State {state}: Probability {prob:.4f}")


    # Plot histogram
##    plot_histogram(counts)
##    plt.title("Measurement Results Histogram")
##    plt.show()
##
##    # Visualize the circuit
##    print("\nQuantum Circuit:")
##    qc.draw('mpl')
##    plt.show()


if __name__ == "__main__":
    print_experiment_goals()
    backend_type = "quantum"  # Change to "quantum" to run on a real device
    run_experiment(backend_type=backend_type, shots=1024)
