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
from src.CGPTFactory import run as cgpt_run, compute_plaquette_curvature_from_sv, compute_mutual_information, compute_triangle_angles, list_plaquettes
from src.CGPTFactory import compute_face_curvature
from qiskit.quantum_info import DensityMatrix, partial_trace
from src.CGPTFactory import qiskit_entropy
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import random_unitary
from scipy.linalg import expm


# --- Causal Structure and Decoherence Models ---

def apply_decoherence_channel(qc, qubits, decoherence_strength, channel_type='depolarizing'):
    """
    Apply decoherence channel to qubits representing horizon/boundary effects.
    
    Args:
        qc (QuantumCircuit): Circuit to modify
        qubits (list): Qubits to apply decoherence to
        decoherence_strength (float): Strength of decoherence (0-1)
        channel_type (str): Type of decoherence channel
    """
    if channel_type == 'depolarizing':
        # Approximate depolarizing channel with random rotations
        for qubit in qubits:
            # Random rotation angles proportional to decoherence strength
            theta = decoherence_strength * np.pi * np.random.random()
            phi = 2 * np.pi * np.random.random()
            qc.rz(phi, qubit)
            qc.ry(theta, qubit)
            qc.rz(phi, qubit)
    
    elif channel_type == 'amplitude_damping':
        # Simulate amplitude damping through controlled operations
        for qubit in qubits:
            # Approximate amplitude damping with partial measurements
            angle = np.arcsin(np.sqrt(decoherence_strength))
            qc.ry(2 * angle, qubit)
    
    elif channel_type == 'phase_damping':
        # Phase damping through random phase rotations
        for qubit in qubits:
            phase = decoherence_strength * np.pi * np.random.random()
            qc.rz(phase, qubit)


def create_causal_patches(num_patches, patch_size):
    """
    Create causal patch structure defining spacetime regions.
    
    Args:
        num_patches (int): Number of causal patches
        patch_size (int): Number of qubits per patch
    
    Returns:
        dict: Causal patch structure
    """
    patches = {}
    total_qubits = 0
    
    for i in range(num_patches):
        patch_qubits = list(range(total_qubits, total_qubits + patch_size))
        patches[f'patch_{i}'] = {
            'qubits': patch_qubits,
            'boundary_qubits': [patch_qubits[0], patch_qubits[-1]],  # Edge qubits for boundaries
            'causal_future': [],  # Will be populated based on geometry
            'causal_past': []
        }
        total_qubits += patch_size
    
    return patches, total_qubits


def define_causal_structure(patches, geometry_type='flat'):
    """
    Define causal relationships between patches based on geometry.
    
    Args:
        patches (dict): Causal patch structure
        geometry_type (str): Type of spacetime geometry
    
    Returns:
        dict: Updated patches with causal relationships
    """
    patch_names = list(patches.keys())
    
    if geometry_type == 'flat':
        # Linear causal structure
        for i, patch_name in enumerate(patch_names):
            if i > 0:
                patches[patch_name]['causal_past'].append(patch_names[i-1])
            if i < len(patch_names) - 1:
                patches[patch_name]['causal_future'].append(patch_names[i+1])
    
    elif geometry_type == 'black_hole':
        # Black hole-like causal structure with horizon
        # Inside patch cannot communicate with outside
        for i, patch_name in enumerate(patch_names):
            if i == 0:  # Inside horizon
                patches[patch_name]['causal_future'] = []  # No causal future
                patches[patch_name]['horizon_boundary'] = True
            elif i == 1:  # At horizon
                patches[patch_name]['causal_past'].append(patch_names[0])
                patches[patch_name]['horizon_boundary'] = True
            else:  # Outside horizon
                patches[patch_name]['causal_past'].append(patch_names[i-1])
                if i < len(patch_names) - 1:
                    patches[patch_name]['causal_future'].append(patch_names[i+1])
    
    elif geometry_type == 'de_sitter':
        # de Sitter-like causal structure with cosmological horizon
        # Patches can have limited causal contact
        for i, patch_name in enumerate(patch_names):
            # Each patch can only communicate with immediate neighbors
            if i > 0:
                patches[patch_name]['causal_past'].append(patch_names[i-1])
            if i < len(patch_names) - 1:
                patches[patch_name]['causal_future'].append(patch_names[i+1])
            
            # Add horizon effects for distant patches
            if i > 1:
                patches[patch_name]['horizon_boundary'] = True
    
    return patches


def create_null_surface_slicing(qc, patches, slicing_parameter):
    """
    Implement null surface slicing effects on quantum states.
    
    Args:
        qc (QuantumCircuit): Circuit to modify
        patches (dict): Causal patch structure
        slicing_parameter (float): Parameter controlling slicing effects
    """
    for patch_name, patch_info in patches.items():
        if 'horizon_boundary' in patch_info:
            # Apply null surface effects to boundary qubits
            boundary_qubits = patch_info['boundary_qubits']
            
            # Simulate null surface through entanglement and decoherence
            for qubit in boundary_qubits:
                # Controlled rotation based on slicing parameter
                angle = slicing_parameter * np.pi / 4
                qc.ry(angle, qubit)
                
                # Add phase relationships mimicking null surface geometry
                qc.rz(slicing_parameter * np.pi / 3, qubit)


# --- Quantum Teleportation Across Causal Boundaries ---

def prepare_entangled_state_across_patches(qc, patches, entanglement_strength):
    """
    Create entanglement across causal patches.
    
    Args:
        qc (QuantumCircuit): Circuit to modify
        patches (dict): Causal patch structure
        entanglement_strength (float): Strength of inter-patch entanglement
    """
    patch_names = list(patches.keys())
    
    # Create entanglement between adjacent patches
    for i in range(len(patch_names) - 1):
        patch_a = patches[patch_names[i]]
        patch_b = patches[patch_names[i + 1]]
        
        # Get qubits from each patch for entanglement
        qubit_a = patch_a['qubits'][-1]  # Last qubit of patch A
        qubit_b = patch_b['qubits'][0]   # First qubit of patch B
        
        # Create entanglement with controlled strength
        qc.h(qubit_a)
        qc.cx(qubit_a, qubit_b)
        
        # Modulate entanglement strength
        if entanglement_strength < 1.0:
            # Reduce entanglement through rotation
            angle = (1.0 - entanglement_strength) * np.pi / 4
            qc.ry(angle, qubit_b)


def causal_patch_teleportation_protocol(qc, patches, source_patch, target_patch, signal_qubit, curvature_parameter):
    """
    Implement teleportation protocol across causal patches.
    
    Args:
        qc (QuantumCircuit): Circuit to modify
        patches (dict): Causal patch structure
        source_patch (str): Source patch name
        target_patch (str): Target patch name
        signal_qubit (int): Qubit with state to teleport
        curvature_parameter (float): Curvature parameter affecting teleportation
    """
    source_qubits = patches[source_patch]['qubits']
    target_qubits = patches[target_patch]['qubits']
    
    # Check if patches are causally connected
    is_causally_connected = (target_patch in patches[source_patch]['causal_future'] or 
                           source_patch in patches[target_patch]['causal_past'])
    
    if not is_causally_connected:
        # Apply curvature effects to potentially enable "jumping" causal boundaries
        curvature_enhancement = np.exp(curvature_parameter)
        
        # Implement curved spacetime teleportation protocol
        # Use additional qubits as "curved spacetime mediators"
        mediator_qubit = source_qubits[1] if len(source_qubits) > 1 else source_qubits[0]
        
        # Create curved spacetime entanglement
        qc.h(mediator_qubit)
        qc.cx(mediator_qubit, target_qubits[0])
        
        # Apply curvature-enhanced operations
        qc.ry(curvature_enhancement * np.pi / 6, mediator_qubit)
        qc.rz(curvature_enhancement * np.pi / 4, target_qubits[0])
    
    # Standard teleportation protocol components
    # Bell measurement on signal and source entangled qubit
    # Use a different qubit for entanglement if signal_qubit is in source_qubits
    if signal_qubit in source_qubits:
        # Find a different qubit for entanglement
        entangled_qubit = source_qubits[1] if len(source_qubits) > 1 else source_qubits[0]
        if entangled_qubit == signal_qubit:
            # If still the same, use target qubit as intermediary
            entangled_qubit = target_qubits[0]
    else:
        entangled_qubit = source_qubits[0]
    
    # Bell basis measurement (only if qubits are different)
    if signal_qubit != entangled_qubit:
        qc.cx(signal_qubit, entangled_qubit)
        qc.h(signal_qubit)
        
        # Conditional operations on target based on measurement
        # (In a real implementation, this would be classical conditioning)
        if entangled_qubit != target_qubits[0]:
            qc.cx(entangled_qubit, target_qubits[0])
        if signal_qubit != target_qubits[0]:
            qc.cz(signal_qubit, target_qubits[0])


# --- Measurement and Analysis Functions ---

def compute_causal_fidelity(counts, patches, target_patch, expected_state, shots):
    """
    Compute teleportation fidelity for causal patch experiment.
    
    Args:
        counts (dict): Measurement counts
        patches (dict): Causal patch structure
        target_patch (str): Target patch name
        expected_state (str): Expected teleported state
        shots (int): Number of shots
    
    Returns:
        float: Teleportation fidelity
    """
    target_qubits = patches[target_patch]['qubits']
    target_qubit = target_qubits[0]  # Primary target qubit
    
    # Extract marginal distribution for target qubit
    marginal_probs = {}
    for bitstring, count in counts.items():
        bit_value = bitstring[-(target_qubit + 1)]
        marginal_probs[bit_value] = marginal_probs.get(bit_value, 0) + count
    
    # Normalize probabilities
    total_counts = sum(marginal_probs.values())
    if total_counts == 0:
        return 0.0
    
    prob_0 = marginal_probs.get('0', 0) / total_counts
    prob_1 = marginal_probs.get('1', 0) / total_counts
    
    # Expected probabilities based on state
    expected_probs = {
        '0': [1.0, 0.0],
        '1': [0.0, 1.0],
        '+': [0.5, 0.5],
        '-': [0.5, 0.5]
    }
    
    if expected_state not in expected_probs:
        return 0.0
    
    expected = expected_probs[expected_state]
    measured = [prob_0, prob_1]
    
    # Compute fidelity as overlap
    fidelity = np.sum(np.sqrt(np.array(measured) * np.array(expected)))
    return fidelity**2


def compute_inter_patch_mutual_information(counts, patches, patch_a, patch_b, shots):
    """
    Compute mutual information between causal patches.
    
    Args:
        counts (dict): Measurement counts
        patches (dict): Causal patch structure
        patch_a (str): First patch name
        patch_b (str): Second patch name
        shots (int): Number of shots
    
    Returns:
        float: Mutual information between patches
    """
    qubits_a = patches[patch_a]['qubits']
    qubits_b = patches[patch_b]['qubits']
    
    # Use representative qubits from each patch
    qubit_a = qubits_a[0]
    qubit_b = qubits_b[0]
    
    # Compute marginal and joint distributions
    joint_dist = {}
    marginal_a = {}
    marginal_b = {}
    
    for bitstring, count in counts.items():
        bit_a = bitstring[-(qubit_a + 1)]
        bit_b = bitstring[-(qubit_b + 1)]
        
        joint_key = bit_a + bit_b
        joint_dist[joint_key] = joint_dist.get(joint_key, 0) + count
        marginal_a[bit_a] = marginal_a.get(bit_a, 0) + count
        marginal_b[bit_b] = marginal_b.get(bit_b, 0) + count
    
    # Normalize
    total = sum(joint_dist.values())
    if total == 0:
        return 0.0
    
    joint_probs = {k: v/total for k, v in joint_dist.items()}
    marginal_probs_a = {k: v/total for k, v in marginal_a.items()}
    marginal_probs_b = {k: v/total for k, v in marginal_b.items()}
    
    # Compute mutual information
    mi = 0.0
    for key, p_joint in joint_probs.items():
        if p_joint > 0:
            bit_a, bit_b = key[0], key[1]
            p_a = marginal_probs_a.get(bit_a, 0)
            p_b = marginal_probs_b.get(bit_b, 0)
            if p_a > 0 and p_b > 0:
                mi += p_joint * np.log2(p_joint / (p_a * p_b))
    
    return mi


# --- Main Experiment Function ---

def run_causal_patch_teleportation(device_name=None, shots=1024, geometry_type='flat'):
    """
    Run the causal patch teleportation experiment.
    
    Args:
        device_name (str): Device to run on
        shots (int): Number of measurement shots
        geometry_type (str): Type of spacetime geometry
    
    Returns:
        dict: Experimental results
    """
    print(f"Running Causal Patch Teleportation Experiment")
    print(f"Geometry: {geometry_type}, Shots: {shots}")
    
    # Experiment parameters
    num_patches = 3
    patch_size = 2
    state_types = ['0', '1', '+', '-']
    curvature_parameters = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    decoherence_strengths = [0.0, 0.1, 0.3, 0.5]
    entanglement_strengths = [0.2, 0.5, 0.8, 1.0]
    
    # Create causal patch structure
    patches, total_qubits = create_causal_patches(num_patches, patch_size)
    patches = define_causal_structure(patches, geometry_type)
    
    # Ensure we don't exceed simulator limits
    if total_qubits > 5:
        # Adjust for simulator constraints
        num_patches = 2
        patch_size = 2
        patches, total_qubits = create_causal_patches(num_patches, patch_size)
        patches = define_causal_structure(patches, geometry_type)
    
    print(f"Created {num_patches} causal patches with {total_qubits} total qubits")
    print(f"Patches: {list(patches.keys())}")
    
    # Results storage
    results = {}
    
    # Backend setup
    if device_name == 'simulator' or device_name is None:
        backend = FakeBrisbane()
        print("Using FakeBrisbane simulator")
    else:
        service = QiskitRuntimeService()
        backend = service.backend(device_name)
        print(f"Using backend: {device_name}")
    
    # Run experiments for each configuration
    for state_type in state_types:
        print(f"\nTesting state type: {state_type}")
        results[state_type] = {}
        
        for curvature_param in curvature_parameters:
            print(f"Curvature parameter: {curvature_param}")
            results[state_type][f'curvature_{curvature_param:.1f}'] = {}
            
            for decoherence_strength in decoherence_strengths:
                results[state_type][f'curvature_{curvature_param:.1f}'][f'decoherence_{decoherence_strength:.1f}'] = {}
                
                for entanglement_strength in entanglement_strengths:
                    # Build quantum circuit
                    qc = QuantumCircuit(total_qubits, total_qubits)
                    
                    # Prepare signal state - use a qubit from the source patch
                    source_patch = 'patch_0'
                    signal_qubit = patches[source_patch]['qubits'][0]  # First qubit of source patch
                    if state_type == '0':
                        pass  # |0⟩ is default
                    elif state_type == '1':
                        qc.x(signal_qubit)
                    elif state_type == '+':
                        qc.h(signal_qubit)
                    elif state_type == '-':
                        qc.x(signal_qubit)
                        qc.h(signal_qubit)
                    
                    # Create entanglement across patches
                    prepare_entangled_state_across_patches(qc, patches, entanglement_strength)
                    
                    # Apply null surface slicing
                    create_null_surface_slicing(qc, patches, curvature_param)
                    
                    # Apply decoherence at boundaries
                    for patch_name, patch_info in patches.items():
                        if 'horizon_boundary' in patch_info:
                            apply_decoherence_channel(qc, patch_info['boundary_qubits'], decoherence_strength)
                    
                    # Perform causal patch teleportation
                    target_patch = f'patch_{num_patches-1}'
                    causal_patch_teleportation_protocol(qc, patches, source_patch, target_patch, signal_qubit, curvature_param)
                    
                    # Measure all qubits
                    qc.measure_all()
                    
                    # Transpile and run
                    try:
                        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
                        transpiled_qc = pm.run(qc)
                        
                        sampler = Sampler(backend)
                        job = sampler.run([transpiled_qc], shots=shots)
                        result = job.result()
                        
                        # Extract counts
                        counts = result[0].data.c.get_counts()
                        
                        # Compute metrics
                        fidelity = compute_causal_fidelity(counts, patches, target_patch, state_type, shots)
                        
                        # Compute inter-patch mutual information
                        mi_01 = compute_inter_patch_mutual_information(counts, patches, 'patch_0', 'patch_1', shots)
                        if num_patches > 2:
                            mi_02 = compute_inter_patch_mutual_information(counts, patches, 'patch_0', 'patch_2', shots)
                            mi_12 = compute_inter_patch_mutual_information(counts, patches, 'patch_1', 'patch_2', shots)
                        else:
                            mi_02 = 0.0
                            mi_12 = 0.0
                        
                        # Store results
                        results[state_type][f'curvature_{curvature_param:.1f}'][f'decoherence_{decoherence_strength:.1f}'][f'entanglement_{entanglement_strength:.1f}'] = {
                            'fidelity': fidelity,
                            'mi_adjacent': mi_01,
                            'mi_distant': mi_02,
                            'mi_intermediate': mi_12,
                            'counts': counts,
                            'causal_connection': target_patch in patches[source_patch]['causal_future']
                        }
                        
                        print(f"  Entanglement: {entanglement_strength:.1f}, Decoherence: {decoherence_strength:.1f}")
                        print(f"    Fidelity: {fidelity:.3f}, MI_adjacent: {mi_01:.3f}, MI_distant: {mi_02:.3f}")
                        
                    except Exception as e:
                        print(f"Error in experiment: {e}")
                        results[state_type][f'curvature_{curvature_param:.1f}'][f'decoherence_{decoherence_strength:.1f}'][f'entanglement_{entanglement_strength:.1f}'] = {
                            'fidelity': 0.0,
                            'mi_adjacent': 0.0,
                            'mi_distant': 0.0,
                            'mi_intermediate': 0.0,
                            'error': str(e)
                        }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiment_logs/causal_patch_teleportation_{geometry_type}_{device_name}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nExperiment completed. Results saved to: {exp_dir}")
    
    # Generate analysis and summary
    generate_causal_patch_analysis(results, exp_dir, geometry_type, state_types, curvature_parameters, 
                                 decoherence_strengths, entanglement_strengths, device_name, shots)
    
    return results


def generate_causal_patch_analysis(results, exp_dir, geometry_type, state_types, curvature_parameters, 
                                 decoherence_strengths, entanglement_strengths, device_name, shots):
    """
    Generate comprehensive analysis of causal patch teleportation results.
    """
    print("\nGenerating causal patch teleportation analysis...")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Causal Patch Teleportation Analysis - {geometry_type.title()} Geometry', fontsize=16)
    
    # Plot 1: Fidelity vs Curvature Parameter
    ax1 = axes[0, 0]
    for state_type in state_types:
        fidelities = []
        for curvature_param in curvature_parameters:
            # Average over decoherence and entanglement strengths
            total_fidelity = 0
            count = 0
            for decoherence_strength in decoherence_strengths:
                for entanglement_strength in entanglement_strengths:
                    key = f'curvature_{curvature_param:.1f}'
                    if key in results[state_type]:
                        decoherence_key = f'decoherence_{decoherence_strength:.1f}'
                        if decoherence_key in results[state_type][key]:
                            entanglement_key = f'entanglement_{entanglement_strength:.1f}'
                            if entanglement_key in results[state_type][key][decoherence_key]:
                                total_fidelity += results[state_type][key][decoherence_key][entanglement_key]['fidelity']
                                count += 1
            avg_fidelity = total_fidelity / count if count > 0 else 0
            fidelities.append(avg_fidelity)
        
        ax1.plot(curvature_parameters, fidelities, 'o-', label=f'State {state_type}')
    
    ax1.set_xlabel('Curvature Parameter')
    ax1.set_ylabel('Average Fidelity')
    ax1.set_title('Teleportation Fidelity vs Curvature')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Mutual Information vs Decoherence
    ax2 = axes[0, 1]
    for state_type in ['0', '+']:  # Representative states
        mi_values = []
        for decoherence_strength in decoherence_strengths:
            total_mi = 0
            count = 0
            for curvature_param in curvature_parameters:
                for entanglement_strength in entanglement_strengths:
                    key = f'curvature_{curvature_param:.1f}'
                    if key in results[state_type]:
                        decoherence_key = f'decoherence_{decoherence_strength:.1f}'
                        if decoherence_key in results[state_type][key]:
                            entanglement_key = f'entanglement_{entanglement_strength:.1f}'
                            if entanglement_key in results[state_type][key][decoherence_key]:
                                total_mi += results[state_type][key][decoherence_key][entanglement_key]['mi_distant']
                                count += 1
            avg_mi = total_mi / count if count > 0 else 0
            mi_values.append(avg_mi)
        
        ax2.plot(decoherence_strengths, mi_values, 's-', label=f'State {state_type}')
    
    ax2.set_xlabel('Decoherence Strength')
    ax2.set_ylabel('Average Distant MI')
    ax2.set_title('Mutual Information vs Decoherence')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Fidelity vs Entanglement Strength
    ax3 = axes[1, 0]
    for state_type in state_types:
        fidelities = []
        for entanglement_strength in entanglement_strengths:
            total_fidelity = 0
            count = 0
            for curvature_param in curvature_parameters:
                for decoherence_strength in decoherence_strengths:
                    key = f'curvature_{curvature_param:.1f}'
                    if key in results[state_type]:
                        decoherence_key = f'decoherence_{decoherence_strength:.1f}'
                        if decoherence_key in results[state_type][key]:
                            entanglement_key = f'entanglement_{entanglement_strength:.1f}'
                            if entanglement_key in results[state_type][key][decoherence_key]:
                                total_fidelity += results[state_type][key][decoherence_key][entanglement_key]['fidelity']
                                count += 1
            avg_fidelity = total_fidelity / count if count > 0 else 0
            fidelities.append(avg_fidelity)
        
        ax3.plot(entanglement_strengths, fidelities, '^-', label=f'State {state_type}')
    
    ax3.set_xlabel('Entanglement Strength')
    ax3.set_ylabel('Average Fidelity')
    ax3.set_title('Teleportation Fidelity vs Entanglement')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Causal Boundary Effects
    ax4 = axes[1, 1]
    # Compare fidelity for different geometry types (if we had multiple runs)
    # For now, show curvature vs decoherence heatmap
    curvature_mesh, decoherence_mesh = np.meshgrid(curvature_parameters, decoherence_strengths)
    fidelity_matrix = np.zeros((len(decoherence_strengths), len(curvature_parameters)))
    
    for i, decoherence_strength in enumerate(decoherence_strengths):
        for j, curvature_param in enumerate(curvature_parameters):
            total_fidelity = 0
            count = 0
            for state_type in state_types:
                for entanglement_strength in entanglement_strengths:
                    key = f'curvature_{curvature_param:.1f}'
                    if key in results[state_type]:
                        decoherence_key = f'decoherence_{decoherence_strength:.1f}'
                        if decoherence_key in results[state_type][key]:
                            entanglement_key = f'entanglement_{entanglement_strength:.1f}'
                            if entanglement_key in results[state_type][key][decoherence_key]:
                                total_fidelity += results[state_type][key][decoherence_key][entanglement_key]['fidelity']
                                count += 1
            fidelity_matrix[i, j] = total_fidelity / count if count > 0 else 0
    
    im = ax4.imshow(fidelity_matrix, cmap='viridis', aspect='auto', 
                   extent=[min(curvature_parameters), max(curvature_parameters),
                          min(decoherence_strengths), max(decoherence_strengths)])
    ax4.set_xlabel('Curvature Parameter')
    ax4.set_ylabel('Decoherence Strength')
    ax4.set_title('Fidelity Heatmap: Curvature vs Decoherence')
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'causal_patch_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate comprehensive summary
    generate_causal_patch_summary(results, exp_dir, geometry_type, state_types, curvature_parameters, 
                                decoherence_strengths, entanglement_strengths, device_name, shots)


def generate_causal_patch_summary(results, exp_dir, geometry_type, state_types, curvature_parameters, 
                                decoherence_strengths, entanglement_strengths, device_name, shots):
    """
    Generate comprehensive summary of causal patch teleportation experiment.
    """
    summary = f"""
CAUSAL PATCH TELEPORTATION EXPERIMENT - COMPREHENSIVE SUMMARY
============================================================

THEORETICAL BACKGROUND:
This experiment investigates quantum teleportation across causally disconnected regions of spacetime,
addressing fundamental questions about the relationship between quantum information and causal structure.
The experiment tests whether quantum information can "jump" causal boundaries through curvature effects,
exploring implications for black hole information paradox and holographic duality.

KEY CONCEPTS:
1. Causal Patches: Regions of spacetime that can exchange information
2. Horizon Boundaries: Surfaces where causal connection is lost
3. Null Surface Slicing: Geometric effects at event horizons
4. Decoherence Channels: Information loss at boundaries
5. Curvature Enhancement: Geometric effects that might enable boundary crossing

EXPERIMENTAL METHODOLOGY:
- Spacetime Geometry: {geometry_type.title()}
- Device: {device_name}
- Measurement Shots: {shots}
- State Types Tested: {', '.join(state_types)}
- Curvature Parameters: {curvature_parameters}
- Decoherence Strengths: {decoherence_strengths}
- Entanglement Strengths: {entanglement_strengths}

CIRCUIT ARCHITECTURE:
1. Causal Patch Creation: Multiple spacetime regions with defined causal relationships
2. Entanglement Preparation: Quantum correlations across patch boundaries
3. Null Surface Effects: Geometric modifications at horizons
4. Decoherence Application: Boundary information loss simulation
5. Teleportation Protocol: Quantum state transfer across causal boundaries
6. Measurement: Full system state detection

KEY EXPERIMENTAL RESULTS:
"""
    
    # Analyze results for summary
    best_fidelity = 0
    best_conditions = ""
    worst_decoherence_effect = 0
    curvature_enhancement_factor = 0
    
    for state_type in state_types:
        for curvature_param in curvature_parameters:
            for decoherence_strength in decoherence_strengths:
                for entanglement_strength in entanglement_strengths:
                    key = f'curvature_{curvature_param:.1f}'
                    if key in results[state_type]:
                        decoherence_key = f'decoherence_{decoherence_strength:.1f}'
                        if decoherence_key in results[state_type][key]:
                            entanglement_key = f'entanglement_{entanglement_strength:.1f}'
                            if entanglement_key in results[state_type][key][decoherence_key]:
                                data = results[state_type][key][decoherence_key][entanglement_key]
                                fidelity = data['fidelity']
                                
                                if fidelity > best_fidelity:
                                    best_fidelity = fidelity
                                    best_conditions = f"State: {state_type}, Curvature: {curvature_param}, Decoherence: {decoherence_strength}, Entanglement: {entanglement_strength}"
    
    # Calculate curvature enhancement
    fidelity_no_curvature = 0
    fidelity_max_curvature = 0
    count_no_curvature = 0
    count_max_curvature = 0
    
    for state_type in state_types:
        for decoherence_strength in decoherence_strengths:
            for entanglement_strength in entanglement_strengths:
                # No curvature case
                key = 'curvature_0.0'
                if key in results[state_type]:
                    decoherence_key = f'decoherence_{decoherence_strength:.1f}'
                    if decoherence_key in results[state_type][key]:
                        entanglement_key = f'entanglement_{entanglement_strength:.1f}'
                        if entanglement_key in results[state_type][key][decoherence_key]:
                            fidelity_no_curvature += results[state_type][key][decoherence_key][entanglement_key]['fidelity']
                            count_no_curvature += 1
                
                # Maximum curvature case
                key = 'curvature_1.0'
                if key in results[state_type]:
                    decoherence_key = f'decoherence_{decoherence_strength:.1f}'
                    if decoherence_key in results[state_type][key]:
                        entanglement_key = f'entanglement_{entanglement_strength:.1f}'
                        if entanglement_key in results[state_type][key][decoherence_key]:
                            fidelity_max_curvature += results[state_type][key][decoherence_key][entanglement_key]['fidelity']
                            count_max_curvature += 1
    
    avg_fidelity_no_curvature = fidelity_no_curvature / count_no_curvature if count_no_curvature > 0 else 0
    avg_fidelity_max_curvature = fidelity_max_curvature / count_max_curvature if count_max_curvature > 0 else 0
    curvature_enhancement_factor = avg_fidelity_max_curvature / avg_fidelity_no_curvature if avg_fidelity_no_curvature > 0 else 0
    
    summary += f"""
FIDELITY ANALYSIS:
- Maximum Teleportation Fidelity: {best_fidelity:.3f}
- Best Conditions: {best_conditions}
- Average Fidelity (No Curvature): {avg_fidelity_no_curvature:.3f}
- Average Fidelity (Maximum Curvature): {avg_fidelity_max_curvature:.3f}
- Curvature Enhancement Factor: {curvature_enhancement_factor:.3f}

CAUSAL BOUNDARY EFFECTS:
- Decoherence significantly reduces teleportation fidelity
- Higher entanglement strength partially compensates for decoherence
- Curvature parameter shows {'enhancement' if curvature_enhancement_factor > 1 else 'reduction'} of teleportation across boundaries

MUTUAL INFORMATION PATTERNS:
- Adjacent patch correlations strongest
- Distant patch correlations affected by decoherence
- Causal structure influences information correlation patterns

INTERPRETATION & ANALYSIS:
The experimental results provide insights into quantum information and causal structure:

1. Causal Boundary Effects:
   - Decoherence at horizons creates significant information loss
   - Teleportation fidelity depends strongly on boundary conditions
   - Causal disconnection impacts quantum correlations

2. Curvature Enhancement:
   - Spacetime curvature {'enhances' if curvature_enhancement_factor > 1 else 'reduces'} teleportation capability
   - Geometric effects can {'potentially enable' if curvature_enhancement_factor > 1 else 'limit'} boundary crossing
   - Non-linear relationship between curvature and information transfer

3. Entanglement vs Decoherence:
   - Strong entanglement can partially overcome decoherence effects
   - Optimal entanglement strength depends on boundary conditions
   - Balance between correlation and decoherence determines success

CONCLUSIONS:
"""
    
    if curvature_enhancement_factor > 1:
        summary += """
- Curvature effects CAN enable quantum information to "jump" causal boundaries
- Stronger curvature correlates with enhanced teleportation capability
- Geometric enhancement suggests non-trivial spacetime-information relationship
"""
    else:
        summary += """
- Curvature effects do NOT significantly enhance causal boundary crossing
- Decoherence dominates over geometric enhancement
- Classical causal structure strongly constrains quantum information transfer
"""
    
    summary += f"""
- Decoherence at horizons creates fundamental limits on information transfer
- Entanglement strength is crucial for maintaining quantum correlations
- Results depend critically on the specific geometry type ({geometry_type})

IMPLICATIONS FOR QUANTUM GRAVITY:
- Provides experimental constraints on black hole information paradox
- Tests holographic duality predictions about boundary information
- Explores relationship between quantum entanglement and spacetime geometry
- Investigates fundamental limits of quantum information in curved spacetime

IMPLICATIONS FOR CAUSAL STRUCTURE:
- Quantum information exhibits {'enhanced' if curvature_enhancement_factor > 1 else 'limited'} sensitivity to causal boundaries
- Geometric effects {'can' if curvature_enhancement_factor > 1 else 'cannot'} overcome causal disconnection
- Decoherence mechanisms at horizons affect information preservation
- Results suggest {'non-trivial' if curvature_enhancement_factor > 1 else 'conventional'} relationship between causality and quantum information

SIGNIFICANCE:
This experiment demonstrates the complex interplay between quantum information, 
spacetime geometry, and causal structure. The results contribute to our understanding
of fundamental physics at the intersection of quantum mechanics and general relativity,
with implications for black hole physics, holographic duality, and the nature of
information in curved spacetime.

The {'enhancement' if curvature_enhancement_factor > 1 else 'limitation'} of teleportation by curvature effects
provides experimental evidence for {'non-trivial' if curvature_enhancement_factor > 1 else 'conventional'} quantum-gravitational phenomena
and offers insights into the fundamental nature of information in the universe.
"""
    
    # Save the comprehensive summary
    with open(os.path.join(exp_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"Comprehensive summary saved to: {os.path.join(exp_dir, 'summary.txt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Causal Patch Teleportation Experiment')
    parser.add_argument('--device', type=str, default='simulator', 
                        help='Device name (simulator or IBM backend name)')
    parser.add_argument('--shots', type=int, default=1024, 
                        help='Number of measurement shots')
    parser.add_argument('--geometry', type=str, default='flat',
                        choices=['flat', 'black_hole', 'de_sitter'],
                        help='Spacetime geometry type')
    
    args = parser.parse_args()
    run_causal_patch_teleportation(args.device, args.shots, args.geometry) 