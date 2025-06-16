from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
import numpy as np
import json
from qiskit import transpile

# Initialize IBM Quantum runtime service
service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token='f56367772bb8e1ccbbd9abcbed8d6a18a4633173cd6d04da31f6e709e866b4c7ab101bcc1154ca5587b97ce24240aa2f84bb990956be3a7f34b30c1fc4ea8765'
)

#QiskitRuntimeService.save_account(channel='ibm_quantum', instance='ibm-q/open/main', token='f56367772bb8e1ccbbd9abcbed8d6a18a4633173cd6d04da31f6e709e866b4c7ab101bcc1154ca5587b97ce24240aa2f84bb990956be3a7f34b30c1fc4ea8765')
# List available real quantum hardware
available_backends = service.backends(filters=lambda x: x.configuration().n_qubits >= 2 and not x.configuration().simulator)
print("Available hardware backends:", [(backend.name, backend.status()) for backend in available_backends])
hardware_backends = service.backends(
    filters=lambda b: not b.configuration().simulator and b.status().operational and b.configuration().n_qubits >= 2
)
print("Filtered backends:", [b.name for b in hardware_backends])

# Or save your credentials on disk.
# QiskitRuntimeService.save_account(channel='ibm_quantum', instance='ibm-q/open/main', token='f56367772bb8e1ccbbd9abcbed8d6a18a4633173cd6d04da31f6e709e866b4c7ab101bcc1154ca5587b97ce24240aa2f84bb990956be3a7f34b30c1fc4ea8765'
# Select a backend (real quantum hardware)
backend_name = "ibm_kyiv"  # Replace with the desired backend name
print(f"Using backend: {backend_name}")
print(f"Service backends available: ", service.backends())
print(f"Configuration: ", [backend.name for backend in available_backends])

# Select the backend
try:
    backend = service.backend(backend_name)
    print(f"Using backend: {backend.name}")
except Exception as e:
    print(f"Error selecting backend: {e}")

print("Basis gates of circuit are: ", backend.configuration().basis_gates)
print("Coupling map of backend configuration: ", backend.configuration().coupling_map)


# Function to execute circuit using the IBM Quantum runtime
def execute_circuit_on_hardware(qc, backend_name, shots):
    with Session(backend=backend_name) as session:
        sampler = Sampler(session=session)
        job = sampler.run(circuits=qc, shots=shots)
        result = job.result()
        counts = result.probabilities() # Access quasi-probabilities/regular probabilities
        return {key: int(value * shots) for key, value in counts.items()}  # Convert to raw counts


# Create a quantum circuit for hybrid charge experiments
def create_hybrid_circuit(angle_a, angle_b, with_measurements=False):
    qc = QuantumCircuit(2)

    # Prepare entangled state
    qc.h(0)
    qc.cx(0, 1)

    # Apply measurement basis rotations
    qc.ry(angle_a, 0)
    qc.ry(angle_b, 1)

    if with_measurements:
        qc.measure_all()

    return qc

# Function to compute correlation
def compute_correlation(results, shots):
    p_00 = results.get('00', 0) / shots
    p_01 = results.get('01', 0) / shots
    p_10 = results.get('10', 0) / shots
    p_11 = results.get('11', 0) / shots

    correlation = (p_00 + p_11 - p_01 - p_10)
    return correlation

# Experiment loop with error handling
def run_experiments(shots):
    angles = np.linspace(0, 2 * np.pi, 20)
    results = {}
    max_correlation = -1
    best_angles = None

    try:
        sampler = Sampler()

        for angle_a in angles:
            for angle_b in angles:
                qc = create_hybrid_circuit(angle_a, angle_b, with_measurements=True)
                # Transpile the circuit for the specific backend
                transpiled_qc = transpile(qc, backend=backend)
                try:
                    # Run the circuit and get results
                    job = sampler.run([transpiled_qc], shots=shots)
                    job_result = job.result()
                    counts = {key: int(value * shots) for key, value in job_result.probabilities.items()}

                    # Compute correlation
                    correlation = compute_correlation(counts, shots)

                    # Save results
                    results[(angle_a, angle_b)] = {
                        "counts": counts,
                        "correlation": correlation
                    }

                    # Check for maximum correlation
                    if correlation > max_correlation:
                        max_correlation = correlation
                        best_angles = (angle_a, angle_b)

                    print(f"Angles: ({angle_a:.2f}, {angle_b:.2f}) | Correlation: {correlation:.4f}")

                except Exception as e:
                    print(f"Error executing circuit at angles ({angle_a:.2f}, {angle_b:.2f}): {e}")

        # Save results to a JSON file
        with open(f"experiment_results.json", "w") as f:
            json.dump(results, f, indent=4)

        # Display summary of results
        print("\nExperiment Summary:")
        print(f"Total angles tested: {len(angles) ** 2}")
        print(f"Maximum correlation: {max_correlation:.4f}")
        
        if best_angles:
            print(f"Best angles: ({best_angles[0]:.2f}, {best_angles[1]:.2f})")
    except Exception as e:
        print(f"Error during sampler initialization or execution: {e}")

# Function to find optimal angles for correlation
def find_optimal_angles(color_charge_1, color_charge_2, shots):
    max_correlation = -1
    optimal_angles = (0, 0)
    best_angles = (1 / np.pi, 1 / np.pi)

    for angle_a in np.linspace(0, 2 * np.pi, 20):
        for angle_b in np.linspace(0, 2 * np.pi, 20):
            qc = create_hybrid_circuit(color_charge_1, color_charge_2, angle_a, angle_b)

            sampler = Sampler(backend=backend)
            job = sampler.run(circuits=[qc], shots=shots)
            results = job.result().get_counts(0)
            if not results:
                print(f"No results for angles: {angle_a}, {angle_b}")
                continue


            correlation = compute_correlation(results, shots)
            if abs(correlation) > max_correlation:
                max_correlation = abs(correlation)
                optimal_angles = (angle_a, angle_b)

    return optimal_angles, max_correlation

# Function to execute circuit using the IBM Quantum runtime
def execute_circuit_on_hardware(qc, shots):
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler

    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.backend("ibmq_qasm_simulator")

    sampler = Sampler(backend=backend)
    job = sampler.run(circuits=[qc], shots=shots)
    result = job.result()
    counts = result.get_counts(0)  # Retrieve counts for the first circuit
    return counts


# Run the experiment
with Session(backend=backend) as session:
    shots = 8192
    run_experiments(shots)
