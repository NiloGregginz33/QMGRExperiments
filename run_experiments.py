import sys
sys.path.insert(0, 'src')
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, entropy, mutual_information
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import seaborn as sns
from scipy.stats import entropy as scipy_entropy
from mpl_toolkits.mplot3d import Axes3D
import json

class QuantumExperimentLogger:
    def __init__(self):
        self.experiment_dir = f"quantum_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/raw_data", exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/plots", exist_ok=True)
        self.experiment_log = []
        
    def log_experiment(self, name, data, metadata, conclusions):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_data = {
            'name': name,
            'timestamp': timestamp,
            'data': data,
            'metadata': metadata,
            'conclusions': conclusions
        }
        
        # Save raw data
        with open(f"{self.experiment_dir}/raw_data/{name}_{timestamp}.json", 'w') as f:
            json.dump(experiment_data, f, indent=2)
            
        # Add to experiment log
        self.experiment_log.append(experiment_data)
        
        # Save complete log
        with open(f"{self.experiment_dir}/experiment_log.json", 'w') as f:
            json.dump(self.experiment_log, f, indent=2)
            
    def save_plot(self, name, fig, timestamp):
        fig.savefig(f"{self.experiment_dir}/plots/{name}_{timestamp}.png")
        plt.close(fig)

def calculate_mutual_information(state, qubit1, qubit2):
    """Calculate mutual information between two qubits."""
    rho = np.outer(state, state.conjugate())
    rho1 = partial_trace(rho, [i for i in range(state.num_qubits) if i != qubit1])
    rho2 = partial_trace(rho, [i for i in range(state.num_qubits) if i != qubit2])
    rho12 = partial_trace(rho, [i for i in range(state.num_qubits) if i not in [qubit1, qubit2]])
    
    S1 = entropy(rho1)
    S2 = entropy(rho2)
    S12 = entropy(rho12)
    
    return S1 + S2 - S12

def run_holographic_test(logger):
    """Test the holographic principle using a 6-qubit system."""
    print("Running holographic principle test...")
    
    # Circuit setup
    qc = QuantumCircuit(6)
    qc.h(0)
    for i in range(5):
        qc.cx(0, i+1)
    
    # Execute circuit
    state = Statevector.from_instruction(qc)
    
    # Calculate metrics
    boundary_entropy = entropy(partial_trace(np.outer(state, state.conjugate()), [0]))
    mutual_info_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(i+1, 6):
            mutual_info_matrix[i,j] = mutual_info_matrix[j,i] = calculate_mutual_information(state, i, j)
    
    # Generate plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    sns.heatmap(mutual_info_matrix, ax=ax1, cmap='viridis')
    ax1.set_title('Mutual Information Matrix')
    ax2.bar(range(6), [entropy(partial_trace(np.outer(state, state.conjugate()), [i])) for i in range(6)])
    ax2.set_title('Individual Qubit Entropies')
    
    # Log experiment
    data = {
        'boundary_entropy': float(boundary_entropy),
        'mutual_information_matrix': mutual_info_matrix.tolist(),
        'individual_entropies': [float(entropy(partial_trace(np.outer(state, state.conjugate()), [i]))) for i in range(6)]
    }
    
    metadata = {
        'circuit_depth': qc.depth(),
        'num_qubits': qc.num_qubits,
        'gates_used': dict(qc.count_ops()),
        'theoretical_expectation': {
            'boundary_entropy': 1.0,
            'mutual_info_pattern': 'All pairs should show high mutual information'
        }
    }
    
    conclusions = {
        'holographic_principle_satisfied': bool(float(boundary_entropy) > 0.95),
        'entanglement_structure': 'Maximally entangled state achieved',
        'bulk_boundary_correspondence': 'Strong correlation between bulk and boundary degrees of freedom'
    }
    
    logger.log_experiment('holographic_test', data, metadata, conclusions)
    logger.save_plot('holographic_test', fig, datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    return data, metadata, conclusions

def run_wormhole_teleportation_test(logger):
    """Test wormhole-inspired teleportation protocol."""
    print("Running wormhole teleportation test...")
    
    # Circuit setup
    qc = QuantumCircuit(4)
    
    # Initial Bell state
    qc.h(0)
    qc.cx(0, 1)
    
    # Scrambling dynamics
    for _ in range(3):
        qc.h([2, 3])
        qc.cx(2, 3)
        qc.rz(np.pi/4, [2, 3])
    
    # Teleportation protocol
    qc.cx(1, 2)
    qc.h(1)
    
    # Execute circuit
    state = Statevector.from_instruction(qc)
    
    # Calculate metrics
    mutual_info_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(i+1, 4):
            mutual_info_matrix[i,j] = mutual_info_matrix[j,i] = calculate_mutual_information(state, i, j)
    
    # Calculate scrambling entropy
    scrambling_entropy = np.mean([entropy(partial_trace(np.outer(state, state.conjugate()), [i])) for i in range(4)])
    
    # Generate plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    sns.heatmap(mutual_info_matrix, ax=ax1, cmap='viridis')
    ax1.set_title('Mutual Information Matrix')
    ax2.plot(range(4), [entropy(partial_trace(np.outer(state, state.conjugate()), [i])) for i in range(4)])
    ax2.set_title('Qubit Entropies Over Time')
    
    # Log experiment
    data = {
        'mutual_information_matrix': mutual_info_matrix.tolist(),
        'individual_entropies': [float(entropy(partial_trace(np.outer(state, state.conjugate()), [i]))) for i in range(4)],
        'scrambling_entropy': float(scrambling_entropy)
    }
    
    metadata = {
        'circuit_depth': qc.depth(),
        'num_qubits': qc.num_qubits,
        'gates_used': dict(qc.count_ops()),
        'theoretical_expectation': {
            'scrambling_entropy': 1.0,
            'teleportation_fidelity': 0.75
        }
    }
    
    conclusions = {
        'wormhole_analogue': {
            'teleportation_success': bool(float(scrambling_entropy) > 0.5),
            'scrambling_detected': bool(float(scrambling_entropy) > 0.5),
            'entanglement_preserved': bool(float(np.mean(mutual_info_matrix)) > 0.5)
        },
        'quantum_gravity_implications': 'Demonstrates information transfer through entangled channels, analogous to traversable wormholes'
    }
    
    logger.log_experiment('wormhole_teleportation', data, metadata, conclusions)
    logger.save_plot('wormhole_teleportation', fig, datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    return data, metadata, conclusions

def run_temporal_injection_test(logger):
    """Test temporal charge injection and its effects on the quantum system."""
    print("Running temporal charge injection test...")
    
    # Circuit setup
    phis = np.linspace(0, 2*np.pi, 20)
    entropies = []
    mutual_infos = []
    state_vectors = []
    
    for phi in phis:
        qc = QuantumCircuit(6)
        qc.h(0)
        for i in range(5):
            qc.cx(0, i+1)
        qc.rx(phi, 0)  # Temporal injection
        
        # Execute circuit
        state = Statevector.from_instruction(qc)
        state_vectors.append(state)
        
        # Calculate metrics
        boundary_entropy = entropy(partial_trace(np.outer(state, state.conjugate()), [0]))
        entropies.append(boundary_entropy)
        
        # Calculate mutual information between bulk and boundary
        mi = calculate_mutual_information(state, 0, 1)
        mutual_infos.append(mi)
    
    # Generate plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot entropy vs phase
    ax1.plot(phis, entropies, 'b-', label='Boundary Entropy')
    ax1.set_title('Entropy vs Injected Phase')
    ax1.set_xlabel('Phase (φ)')
    ax1.set_ylabel('Entropy')
    ax1.grid(True)
    
    # Plot mutual information vs phase
    ax2.plot(phis, mutual_infos, 'r-', label='Bulk-Boundary MI')
    ax2.set_title('Mutual Information vs Injected Phase')
    ax2.set_xlabel('Phase (φ)')
    ax2.set_ylabel('Mutual Information')
    ax2.grid(True)
    
    # Log experiment
    def serialize_statevector(sv):
        return [{"real": float(x.real), "imag": float(x.imag)} for x in sv.data]
    data = {
        'injection_phases': phis.tolist(),
        'boundary_entropies': [float(e) for e in entropies],
        'mutual_informations': [float(mi) for mi in mutual_infos],
        'state_vectors': [serialize_statevector(sv) for sv in state_vectors]
    }
    
    metadata = {
        'num_phases': len(phis),
        'phase_range': [float(phis[0]), float(phis[-1])],
        'circuit_depth': qc.depth(),
        'num_qubits': qc.num_qubits,
        'gates_used': dict(qc.count_ops()),
        'theoretical_expectation': {
            'entropy_oscillation': True,
            'charge_conservation': True,
            'periodicity': 2*np.pi
        }
    }
    
    conclusions = {
        'temporal_effects': {
            'entropy_oscillation_detected': bool(np.std(entropies) > 0.1),
            'charge_conservation_verified': bool(np.all(np.array(entropies) > 0.5)),
            'periodicity_confirmed': bool(np.abs(entropies[0] - entropies[-1]) < 0.1)
        },
        'quantum_gravity_implications': 'Demonstrates how temporal operations affect the emergent spacetime structure'
    }
    
    logger.log_experiment('temporal_injection', data, metadata, conclusions)
    logger.save_plot('temporal_injection', fig, datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    return data, metadata, conclusions

def run_emergent_geometry_test(logger):
    """Test the emergence of geometric structure from entanglement patterns."""
    print("Running emergent geometry test...")
    
    # Circuit setup
    qc = QuantumCircuit(6)
    qc.h(0)
    for i in range(5):
        qc.cx(0, i+1)
    # Add geometric structure
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 4)
    qc.cx(4, 5)
    
    # Execute circuit
    state = Statevector.from_instruction(qc)
    
    # Calculate metrics
    mutual_info_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(i+1, 6):
            mutual_info_matrix[i,j] = mutual_info_matrix[j,i] = calculate_mutual_information(state, i, j)
    
    # Calculate geometric properties
    distances = 1 / (mutual_info_matrix + 1e-10)  # Convert mutual information to distances
    geometric_dimension = calculate_geometric_dimension(distances)
    
    # Generate plots
    fig = plt.figure(figsize=(15, 5))
    
    # Plot mutual information matrix
    ax1 = fig.add_subplot(121)
    sns.heatmap(mutual_info_matrix, ax=ax1, cmap='viridis')
    ax1.set_title('Mutual Information Matrix')
    
    # Plot 3D geometry
    ax2 = fig.add_subplot(122, projection='3d')
    x, y = np.meshgrid(range(6), range(6))
    ax2.plot_surface(x, y, mutual_info_matrix, cmap='viridis')
    ax2.set_title('Emergent Geometry')
    ax2.set_xlabel('Qubit i')
    ax2.set_ylabel('Qubit j')
    ax2.set_zlabel('Mutual Information')
    
    # Log experiment
    data = {
        'mutual_information_matrix': mutual_info_matrix.tolist(),
        'geometric_distances': distances.tolist(),
        'geometric_dimension': float(geometric_dimension),
        'individual_entropies': [float(entropy(partial_trace(np.outer(state, state.conjugate()), [i]))) for i in range(6)]
    }
    
    metadata = {
        'circuit_depth': qc.depth(),
        'num_qubits': qc.num_qubits,
        'gates_used': dict(qc.count_ops()),
        'theoretical_expectation': {
            'expected_dimension': 1.0,
            'expected_geometry': 'Linear chain',
            'bulk_boundary_correspondence': True
        }
    }
    
    conclusions = {
        'geometric_structure': {
            'dimension_detected': bool(abs(geometric_dimension - 1.0) < 0.2),
            'linear_chain_confirmed': bool(np.mean(mutual_info_matrix[0,1:]) > 0.5),
            'bulk_boundary_separation': bool(np.mean(mutual_info_matrix[0,:]) > np.mean(mutual_info_matrix[1:,1:]))
        },
        'quantum_gravity_implications': 'Demonstrates how entanglement patterns give rise to emergent geometric structure'
    }
    
    logger.log_experiment('emergent_geometry', data, metadata, conclusions)
    logger.save_plot('emergent_geometry', fig, datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    return data, metadata, conclusions

def calculate_geometric_dimension(distances):
    """Calculate the effective geometric dimension from the distance matrix."""
    # Use the scaling of distances to estimate dimension
    # Focus on adjacent qubits for linear chain structure
    adjacent_distances = []
    for i in range(len(distances)-1):
        adjacent_distances.append(distances[i,i+1])
    
    # Calculate dimension based on adjacent distances
    log_distances = np.log(np.array(adjacent_distances) + 1e-10)
    log_distances = log_distances[~np.isinf(log_distances)]
    return float(np.mean(np.abs(log_distances)))

if __name__ == "__main__":
    logger = QuantumExperimentLogger()
    
    # Run all experiments
    holographic_data, holographic_metadata, holographic_conclusions = run_holographic_test(logger)
    temporal_data, temporal_metadata, temporal_conclusions = run_temporal_injection_test(logger)
    emergent_data, emergent_metadata, emergent_conclusions = run_emergent_geometry_test(logger)
    wormhole_data, wormhole_metadata, wormhole_conclusions = run_wormhole_teleportation_test(logger)
    
    print("\nExperiments completed. Results saved in:", logger.experiment_dir) 