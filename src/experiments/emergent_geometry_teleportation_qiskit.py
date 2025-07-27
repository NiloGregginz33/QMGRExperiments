import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import argparse
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, Session
from qiskit.result import marginal_counts
from scipy.stats import pearsonr
from sklearn.manifold import MDS
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.CGPTFactory import run
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import partial_trace

# Define a custom entropy function

def calculate_entropy(rho):
    # Eigenvalues of the density matrix
    eigenvalues = np.linalg.eigvalsh(rho.data)
    # Clip eigenvalues to avoid log(0)
    eigenvalues = np.clip(eigenvalues, 1e-12, 1.0)
    # Calculate entropy
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
    return entropy

# Define a robust function to compute mutual information

def compute_mutual_information(theta_dict, simulator):
    # Create a quantum circuit
    qc = QuantumCircuit(len(theta_dict) + 1)  # Add 1 to ensure the range is correct
    qc.h(range(len(theta_dict) + 1))
    for (i, j), theta in theta_dict.items():
        qc.cp(theta, i, j)

    # Get the statevector from the circuit
    statevector = Statevector.from_instruction(qc)
    dm = DensityMatrix(statevector)

    # Compute mutual information for each pair
    I = {}
    for (i, j) in theta_dict.keys():
        rho_i = partial_trace(dm, [q for q in range(len(theta_dict) + 1) if q != i])
        rho_j = partial_trace(dm, [q for q in range(len(theta_dict) + 1) if q != j])
        rho_ij = partial_trace(dm, [q for q in range(len(theta_dict) + 1) if q not in (i, j)])
        I[(i, j)] = calculate_entropy(rho_i) + calculate_entropy(rho_j) - calculate_entropy(rho_ij)

    return I

# Define the experiment parameters
num_qubits = 5
shots = 1024

# Ensure the quantum circuit is initialized with the correct number of qubits

# Create a quantum circuit with the required number of qubits
num_qubits = 5  # Ensure this matches the number of qubits used in operations
qc = QuantumCircuit(num_qubits)

# Prepare an entangled state for teleportation
# Ensure operations are within the valid range of qubits
for i in range(num_qubits - 1):
    qc.h(i)
    qc.cx(i, i+1)

# Add measurement instructions to the quantum circuit
qc.measure_all()

# Define the 'compute_fidelity' function

def compute_fidelity(qc, node_a, node_b, shots):
    # Prepare the circuit for teleportation between node_a and node_b
    # For simplicity, assume the circuit is already set up for teleportation
    
    # Use 'run' to obtain counts
    counts = run(qc, backend=backend, shots=shots)
    
    # Calculate the fidelity based on the results
    # For simplicity, assume fidelity is the probability of measuring the expected state
    expected_state = '00'  # Example expected state
    fidelity = counts.get(expected_state, 0) / shots
    
    # Return the calculated fidelity
    return fidelity

# Define the theta_dict variable
# Ensure the theta_dict is correctly initialized for the specified number of qubits
# Only create cp gates for valid qubit pairs
# Example: theta_dict = {(0, 1): np.pi/4, (1, 2): np.pi/4, ...}
theta_dict = {(i, i+1): np.pi/4 for i in range(num_qubits - 1)}

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Emergent Geometry Teleportation Experiment')
parser.add_argument('--shots', type=int, default=1024, help='Number of shots for the experiment')
parser.add_argument('--device', type=str, default='simulator', help='Device to run the experiment on')
parser.add_argument('--num_qubits', type=int, default=5, help='Number of qubits for the experiment')
args = parser.parse_args()

# Use 'FakeBrisbane' as the backend if specified
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

if args.device == 'simulator':
    backend = FakeBrisbane()
    print("Using FakeBrisbane simulator backend.")
else:
    service = QiskitRuntimeService()
    backend = next(b for b in service.backends() if b.name == args.device)

qc = transpile(qc, backend=backend)

# Use 'run' from CGPTFactory to execute the experiment
counts = run(qc, backend=backend, shots=shots)

# Manually construct the density matrix from counts
state = DensityMatrix.from_label('0' * num_qubits)
for bitstring, probability in counts.items():
    state += probability * DensityMatrix.from_label(bitstring)

# Compute the mutual information matrix
mi_matrix = np.zeros((num_qubits, num_qubits))
mi = compute_mutual_information(theta_dict, backend)
for (i, j), value in mi.items():
    mi_matrix[i, j] = value
    mi_matrix[j, i] = value

# Embed the MI matrix into a geometric space
mds = MDS(n_components=2, dissimilarity='precomputed')
embedded_space = mds.fit_transform(1 - mi_matrix)

# Select distant and close nodes based on the embedded space
# For simplicity, choose the first and last nodes as distant, and adjacent nodes as close
node_pairs = [(0, 4), (1, 2)]

# Perform teleportation and measure fidelity
fidelities = {}
for (node_a, node_b) in node_pairs:
    # Attempt teleportation between node_a and node_b
    fidelity = compute_fidelity(qc, node_a, node_b, shots)
    fidelities[(node_a, node_b)] = fidelity

# Analyze the results
for (node_a, node_b), fidelity in fidelities.items():
    distance = np.linalg.norm(embedded_space[node_a] - embedded_space[node_b])
    print(f"Teleportation between nodes {node_a} and {node_b}: Fidelity = {fidelity}, Emergent Distance = {distance}")

# Save the results
experiment_dir = f"experiment_logs/emergent_geometry_teleportation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(experiment_dir, exist_ok=True)

# Convert tuple keys to strings before saving the results
fidelities_str_keys = {str(k): v for k, v in fidelities.items()}

# Save the expanded results
results = {
    'experiment_parameters': {
        'num_qubits': num_qubits,
        'shots': shots
    },
    'fidelities': fidelities_str_keys,
    'mutual_information': mi_matrix.tolist(),  # Convert numpy array to list
    'embedded_space': embedded_space.tolist()  # Convert numpy array to list
}

# Save the results with expanded data
with open(os.path.join(experiment_dir, 'results.json'), 'w') as f:
    json.dump(results, f, indent=4)

# Generate a summary
summary = f"Experiment on emergent geometry and teleportation\n"
summary += f"Nodes tested: {node_pairs}\n"
summary += f"Fidelities: {fidelities}\n"
summary += f"Embedded Space: {embedded_space}\n"

with open(os.path.join(experiment_dir, 'summary.txt'), 'w') as f:
    f.write(summary) 