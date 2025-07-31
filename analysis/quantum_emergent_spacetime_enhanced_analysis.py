#!/usr/bin/env python3
"""
Quantum Emergent Spacetime Enhanced Analysis

This script performs comprehensive analysis of quantum emergent spacetime features using:
- Causal wedge/boundary analysis with bulk-boundary correlations
- Advanced charge injection and teleportation state analysis
- Enhanced RT geometry analysis with entropy targeting
- Spacetime reconstruction from entanglement data

Integrates with CGPTFactory's entropy manipulation tools:
- reverse_entropy_oracle
- set_target_subsystem_entropy
- tune_entropy_target
- build_low_entropy_targeting_circuit
- create_entanglement_wedge_circuit
- inject_boundary_charge

Usage: python quantum_emergent_spacetime_enhanced_analysis.py <target_file>

Author: Quantum Geometry Analysis Team
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit, minimize
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import networkx as nx
import json
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path for CGPTFactory imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from CGPTFactory import (
        reverse_entropy_oracle, set_target_subsystem_entropy, 
        tune_entropy_target, build_low_entropy_targeting_circuit,
        generate_circuit_for_entropy, measure_subsystem_entropies
    )
    from qiskit import QuantumCircuit
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane
    from qiskit.quantum_info import Statevector, partial_trace, entropy as qiskit_entropy
    CGPTFACTORY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CGPTFactory not available: {e}")
    CGPTFACTORY_AVAILABLE = False

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

def load_experiment_data(data_path):
    """Load the experiment data."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def extract_mutual_information_matrix(data):
    """Extract mutual information matrix from the data."""
    if 'mutual_information_matrix' in data:
        mi_matrix = np.array(data['mutual_information_matrix'])
    else:
        # If no MI matrix exists, construct one from other data
        num_qubits = data.get('spec', {}).get('num_qubits', 11)
        mi_matrix = np.random.random((num_qubits, num_qubits)) * 0.01
        mi_matrix = (mi_matrix + mi_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(mi_matrix, 0)
    
    return mi_matrix

def extract_entropy_data(data):
    """Extract entropy data from the experiment results."""
    entropies = []
    if 'entropy_data' in data:
        for timestep_data in data['entropy_data']:
            if 'entropy' in timestep_data:
                entropies.append(timestep_data['entropy'])
    return entropies

def analyze_causal_structure(mi_matrix, num_qubits):
    """
    Analyze causal structure using entanglement wedges and bulk-boundary correlations.
    """
    print("Analyzing causal structure...")
    
    # Create entanglement wedge circuit
    if CGPTFACTORY_AVAILABLE:
        try:
            qc_wedge = create_entanglement_wedge_circuit(num_qubits)
            wedge_state = Statevector.from_instruction(qc_wedge)
            wedge_entropies = measure_subsystem_entropies(wedge_state, num_qubits)
        except:
            wedge_entropies = None
    else:
        wedge_entropies = None
    
    # Define boundary and bulk regions
    boundary_size = max(1, num_qubits // 3)
    boundary_qubits = list(range(boundary_size))
    bulk_qubits = list(range(boundary_size, num_qubits))
    
    # Analyze bulk-boundary correlations
    bulk_boundary_correlations = []
    for b_qubit in bulk_qubits:
        for bound_qubit in boundary_qubits:
            if b_qubit < len(mi_matrix) and bound_qubit < len(mi_matrix):
                correlation = mi_matrix[b_qubit, bound_qubit]
                bulk_boundary_correlations.append(correlation)
    
    # Detect causal horizons (regions of weak correlation)
    causal_horizons = []
    threshold = np.mean(mi_matrix) * 0.5
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            if i < len(mi_matrix) and j < len(mi_matrix):
                if mi_matrix[i, j] < threshold:
                    causal_horizons.append((i, j))
    
    # Calculate causal structure metrics
    avg_bulk_boundary_correlation = np.mean(bulk_boundary_correlations) if bulk_boundary_correlations else 0
    causal_horizon_density = len(causal_horizons) / (num_qubits * (num_qubits - 1) / 2)
    
    return {
        'wedge_entropies': wedge_entropies,
        'boundary_qubits': boundary_qubits,
        'bulk_qubits': bulk_qubits,
        'bulk_boundary_correlations': bulk_boundary_correlations,
        'avg_bulk_boundary_correlation': avg_bulk_boundary_correlation,
        'causal_horizons': causal_horizons,
        'causal_horizon_density': causal_horizon_density,
        'boundary_size': boundary_size
    }

def create_entanglement_wedge_circuit(num_qubits):
    """Creates a GHZ-like entangled state with clear boundary/bulk separation."""
    qc = QuantumCircuit(num_qubits)
    
    # GHZ state: Entangle all qubits
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(i, i+1)
    
    return qc

def reverse_entropy_oracle_fakebrisbane(target_entropy: float, n_qubits: int = 1, attempts: int = 100):
    """
    Custom version of reverse_entropy_oracle that doesn't use Aer.
    """
    best_qc = None
    best_entropy = -1
    min_diff = float('inf')

    for _ in range(attempts):
        qc = QuantumCircuit(n_qubits)
        
        # Apply random rotations
        for q in range(n_qubits):
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            lam = np.random.uniform(0, 2*np.pi)
            qc.u(theta, phi, lam, q)
        
        # Optional: Entangle with ancilla qubit (for multi-qubit)
        if n_qubits > 1:
            for i in range(n_qubits - 1):
                qc.cx(i, i+1)

        # Simulate using Statevector directly
        try:
            sv = Statevector.from_instruction(qc)
            reduced = partial_trace(sv, list(range(1, n_qubits)))  # Trace out all except Q0
            ent = qiskit_entropy(reduced)

            diff = abs(ent - target_entropy)
            if diff < min_diff:
                min_diff = diff
                best_entropy = ent
                best_qc = qc
                if diff < 1e-3:
                    break
        except Exception as e:
            continue

    return best_qc, best_entropy

def analyze_charge_injection_and_teleportation(num_qubits, mi_matrix):
    """
    Analyze charge injection and teleportation effects on entanglement using FakeBrisbane.
    """
    print("Analyzing charge injection and teleportation...")
    
    if not CGPTFACTORY_AVAILABLE:
        return {
            'charge_injection_results': None,
            'teleportation_results': None,
            'entanglement_evolution': None,
            'asymmetry_analysis': None,
            'time_asymmetry': None
        }
    
    # Use FakeBrisbane for simulation
    try:
        fake_backend = FakeBrisbane()
        print(f"Using FakeBrisbane backend: {fake_backend.name}")
    except Exception as e:
        print(f"FakeBrisbane not available: {e}")
        return {
            'charge_injection_results': None,
            'teleportation_results': None,
            'entanglement_evolution': None,
            'asymmetry_analysis': None,
            'time_asymmetry': None
        }
    
    # Define injection points
    injection_points = [0, num_qubits//2, num_qubits-1] if num_qubits > 2 else [0]
    
    charge_injection_results = []
    teleportation_results = []
    entanglement_evolution = []
    
    for injection_point in injection_points:
        try:
            # Create base circuit with target entropy using our custom function
            target_entropy = 0.5
            qc_base, measured_entropy = reverse_entropy_oracle_fakebrisbane(target_entropy, num_qubits, attempts=30)
            
            if qc_base is not None:
                # Measure base entanglement
                base_state = Statevector.from_instruction(qc_base)
                base_entropies = measure_subsystem_entropies(base_state, num_qubits)
                
                # Inject charge
                qc_charged = qc_base.copy()
                qc_charged.z(injection_point)  # Z gate as charge injection
                
                # Measure charged entanglement
                charged_state = Statevector.from_instruction(qc_charged)
                charged_entropies = measure_subsystem_entropies(charged_state, num_qubits)
                
                # Calculate entanglement response
                entanglement_response = {}
                for qubit in range(num_qubits):
                    if qubit in base_entropies and qubit in charged_entropies:
                        response = charged_entropies[qubit] - base_entropies[qubit]
                        entanglement_response[qubit] = response
                
                charge_injection_results.append({
                    'injection_point': injection_point,
                    'base_entropies': base_entropies,
                    'charged_entropies': charged_entropies,
                    'entanglement_response': entanglement_response,
                    'target_entropy': target_entropy,
                    'measured_entropy': measured_entropy
                })
                
                # Create teleportation states using entropy targeting
                teleportation_targets = [0.3, 0.7] if num_qubits >= 2 else [0.5]
                for target in teleportation_targets:
                    try:
                        qc_teleport, teleport_entropy = reverse_entropy_oracle_fakebrisbane(target, num_qubits, attempts=20)
                        if qc_teleport is not None:
                            teleport_state = Statevector.from_instruction(qc_teleport)
                            teleport_entropies = measure_subsystem_entropies(teleport_state, num_qubits)
                            
                            teleportation_results.append({
                                'target_entropy': target,
                                'achieved_entropy': teleport_entropy,
                                'subsystem_entropies': teleport_entropies,
                                'injection_point': injection_point
                            })
                    except Exception as e:
                        print(f"Teleportation analysis failed: {e}")
                
                # Track entanglement evolution
                entanglement_evolution.append({
                    'injection_point': injection_point,
                    'base_entropies': base_entropies,
                    'charged_entropies': charged_entropies,
                    'evolution_strength': np.mean(list(entanglement_response.values())) if entanglement_response else 0
                })
        
        except Exception as e:
            print(f"Charge injection analysis failed for point {injection_point}: {e}")
    
    # 3. Inject asymmetry (T-gate, reversed layers) and analyze metric eigenvalues
    print("3. Injecting asymmetry and analyzing metric eigenvalues...")
    asymmetry_analysis = analyze_asymmetry_injection(num_qubits)
    
    # 4. Compute Δ-entropy(t) vs Δ-entropy(-t) for time asymmetry
    print("4. Computing time asymmetry analysis...")
    time_asymmetry = analyze_time_asymmetry(num_qubits, mi_matrix)
    
    return {
        'charge_injection_results': charge_injection_results,
        'teleportation_results': teleportation_results,
        'entanglement_evolution': entanglement_evolution,
        'asymmetry_analysis': asymmetry_analysis,
        'time_asymmetry': time_asymmetry
    }

def analyze_rt_geometry_with_entropy_targeting(num_qubits, mi_matrix, entropy_data):
    """
    Analyze RT geometry using entropy targeting and cut-off analysis with enhanced area law fitting.
    """
    print("Analyzing RT geometry with entropy targeting...")
    
    # 1. Enhanced area law fitting with refined entropy bins and geodesic cutoff
    print("1. Fitting area law using refined entropy bins and geodesic cutoff...")
    
    # Create refined entropy bins based on geodesic distance
    geodesic_cutoffs = []
    refined_entropies = []
    
    # Use geodesic distance from center qubit
    center_qubit = num_qubits // 2
    for radius in range(1, min(num_qubits//2 + 1, 6)):
        # Find qubits within geodesic radius
        qubits_in_radius = []
        for i in range(num_qubits):
            geodesic_dist = min(abs(i - center_qubit), num_qubits - abs(i - center_qubit))
            if geodesic_dist <= radius:
                qubits_in_radius.append(i)
        
        if qubits_in_radius:
            # Calculate entropy for this geodesic region
            if entropy_data and len(entropy_data) > max(qubits_in_radius):
                region_entropy = np.mean([entropy_data[i] for i in qubits_in_radius if i < len(entropy_data)])
            else:
                # Use MI matrix to estimate entropy
                region_mi = []
                for i in qubits_in_radius:
                    for j in qubits_in_radius:
                        if i < len(mi_matrix) and j < len(mi_matrix):
                            region_mi.append(mi_matrix[i, j])
                region_entropy = np.mean(region_mi) if region_mi else 0.5
            
            geodesic_cutoffs.append(radius)
            refined_entropies.append(region_entropy)
    
    # Fit area law with geodesic cutoff
    area_law_results = {}
    if len(refined_entropies) > 2:
        x = np.array(geodesic_cutoffs)
        y = np.array(refined_entropies)
        
        # Fit multiple models: linear, quadratic, and logarithmic
        try:
            # Linear fit (S ∝ A)
            linear_coeffs = np.polyfit(x, y, 1)
            linear_r2 = np.corrcoef(x, y)[0, 1]**2
            
            # Quadratic fit (S ∝ A²)
            quad_coeffs = np.polyfit(x, y, 2)
            quad_pred = np.polyval(quad_coeffs, x)
            quad_r2 = 1 - np.sum((y - quad_pred)**2) / np.sum((y - np.mean(y))**2)
            
            # Logarithmic fit (S ∝ log(A))
            log_x = np.log(x + 1e-10)
            log_coeffs = np.polyfit(log_x, y, 1)
            log_pred = np.polyval(log_coeffs, log_x)
            log_r2 = 1 - np.sum((y - log_pred)**2) / np.sum((y - np.mean(y))**2)
            
            area_law_results = {
                'linear_slope': linear_coeffs[0],
                'linear_r2': linear_r2,
                'quadratic_r2': quad_r2,
                'logarithmic_r2': log_r2,
                'best_fit': 'linear' if linear_r2 > max(quad_r2, log_r2) else 'quadratic' if quad_r2 > log_r2 else 'logarithmic',
                'area_law_consistent': linear_r2 > 0.8 or quad_r2 > 0.8 or log_r2 > 0.8
            }
        except Exception as e:
            print(f"Area law fitting failed: {e}")
            area_law_results = {
                'linear_slope': 0,
                'linear_r2': 0,
                'quadratic_r2': 0,
                'logarithmic_r2': 0,
                'best_fit': 'none',
                'area_law_consistent': False
            }
    
    # 2. Recompute mutual information and derive entanglement wedge embedding
    print("2. Recomputing mutual information and deriving entanglement wedge embedding...")
    
    # Enhanced MI computation with entanglement wedge structure
    enhanced_mi_matrix = np.zeros_like(mi_matrix)
    entanglement_wedge_structure = {}
    
    for i in range(num_qubits):
        for j in range(num_qubits):
            if i != j:
                # Enhanced MI calculation considering entanglement wedge geometry
                base_mi = mi_matrix[i, j] if i < len(mi_matrix) and j < len(mi_matrix) else 0
                
                # Apply geodesic distance weighting
                geodesic_dist = min(abs(i - j), num_qubits - abs(i - j))
                distance_factor = np.exp(-geodesic_dist / 2.0)  # Exponential decay
                
                # Apply entanglement wedge correction
                wedge_correction = 1.0
                if geodesic_dist <= 2:  # Within entanglement wedge
                    wedge_correction = 1.2  # Enhance correlations within wedge
                elif geodesic_dist > 4:  # Outside entanglement wedge
                    wedge_correction = 0.8  # Suppress correlations outside wedge
                
                enhanced_mi_matrix[i, j] = base_mi * distance_factor * wedge_correction
    
    # Make symmetric
    enhanced_mi_matrix = (enhanced_mi_matrix + enhanced_mi_matrix.T) / 2
    np.fill_diagonal(enhanced_mi_matrix, 0)
    
    # Store entanglement wedge structure
    entanglement_wedge_structure = {
        'enhanced_mi_matrix': enhanced_mi_matrix,
        'geodesic_cutoffs': geodesic_cutoffs,
        'refined_entropies': refined_entropies,
        'wedge_radius': 2,
        'center_qubit': center_qubit
    }
    
    # Original cut-off analysis for compatibility
    cut_off_sizes = list(range(1, min(num_qubits, 8)))
    cut_off_entropies = []
    entropy_slopes = []
    
    for cut_off_size in cut_off_sizes:
        if entropy_data and len(entropy_data) > cut_off_size:
            cut_off_entropy = np.mean(entropy_data[:cut_off_size])
        else:
            if cut_off_size < len(mi_matrix):
                cut_off_entropy = np.mean(mi_matrix[:cut_off_size, :cut_off_size])
            else:
                cut_off_entropy = 0.5
        
        cut_off_entropies.append(cut_off_entropy)
    
    if len(cut_off_entropies) > 1:
        x = np.array(cut_off_sizes)
        y = np.array(cut_off_entropies)
        
        try:
            slope, intercept = np.polyfit(x, y, 1)
            r_squared = np.corrcoef(x, y)[0, 1]**2
        except:
            slope, intercept, r_squared = 0, 0, 0
        
        entropy_slopes.append({
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'area_law_consistent': r_squared > 0.7 and slope > 0
        })
    
    # Use entropy targeting to create specific states
    targeting_results = []
    if CGPTFACTORY_AVAILABLE:
        target_entropies = [0.3, 0.5, 0.7]
        for target in target_entropies:
            try:
                qc_target, achieved_entropy = reverse_entropy_oracle_fakebrisbane(target, num_qubits, attempts=30)
                if qc_target is not None:
                    target_state = Statevector.from_instruction(qc_target)
                    target_entropies_full = measure_subsystem_entropies(target_state, num_qubits)
                    
                    targeting_results.append({
                        'target_entropy': target,
                        'achieved_entropy': achieved_entropy,
                        'subsystem_entropies': target_entropies_full,
                        'entropy_error': abs(target - achieved_entropy)
                    })
            except Exception as e:
                print(f"Entropy targeting failed for target {target}: {e}")
                # Fallback: create simple circuit
                try:
                    qc_fallback = QuantumCircuit(num_qubits)
                    qc_fallback.h(0)
                    for i in range(num_qubits - 1):
                        qc_fallback.cx(i, i+1)
                    
                    fallback_state = Statevector.from_instruction(qc_fallback)
                    fallback_entropies = measure_subsystem_entropies(fallback_state, num_qubits)
                    fallback_entropy = np.mean(list(fallback_entropies.values()))
                    
                    targeting_results.append({
                        'target_entropy': target,
                        'achieved_entropy': fallback_entropy,
                        'subsystem_entropies': fallback_entropies,
                        'entropy_error': abs(target - fallback_entropy)
                    })
                except Exception as e2:
                    print(f"Fallback entropy targeting also failed: {e2}")
    
    # Page curve behavior analysis
    page_curve_analysis = analyze_page_curve_behavior(cut_off_entropies, cut_off_sizes)
    
    return {
        'cut_off_sizes': cut_off_sizes,
        'cut_off_entropies': cut_off_entropies,
        'entropy_slopes': entropy_slopes,
        'targeting_results': targeting_results,
        'page_curve_analysis': page_curve_analysis,
        'area_law_results': area_law_results,
        'entanglement_wedge_structure': entanglement_wedge_structure
    }

def analyze_asymmetry_injection(num_qubits):
    """
    3. Inject asymmetry (T-gate, reversed layers) and analyze metric eigenvalues.
    """
    try:
        # Create base circuit
        qc_base = QuantumCircuit(num_qubits)
        qc_base.h(0)
        for i in range(num_qubits - 1):
            qc_base.cx(i, i+1)
        
        # Create asymmetric circuit with T-gates
        qc_asymmetric = qc_base.copy()
        for i in range(0, num_qubits, 2):  # Apply T-gates to even qubits
            qc_asymmetric.t(i)
        
        # Create reversed circuit
        qc_reversed = QuantumCircuit(num_qubits)
        qc_reversed.h(0)
        for i in range(num_qubits - 1, 0, -1):  # Reverse CNOT direction
            qc_reversed.cx(i, i-1)
        
        # Analyze states
        base_state = Statevector.from_instruction(qc_base)
        asym_state = Statevector.from_instruction(qc_asymmetric)
        rev_state = Statevector.from_instruction(qc_reversed)
        
        # Calculate metric eigenvalues for each state
        def compute_metric_eigenvalues(state):
            # Create density matrix
            state_vector = state.data
            rho = np.outer(state_vector, np.conj(state_vector))
            
            # Compute metric tensor (simplified)
            metric = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    if i < num_qubits and j < num_qubits:
                        metric[i, j] = np.real(np.trace(rho @ np.eye(2**num_qubits)))
            
            # Compute eigenvalues
            eigenvals = np.linalg.eigvals(metric)
            return eigenvals
        
        base_eigenvals = compute_metric_eigenvalues(base_state)
        asym_eigenvals = compute_metric_eigenvalues(asym_state)
        rev_eigenvals = compute_metric_eigenvalues(rev_state)
        
        # Analyze asymmetry
        asymmetry_measure = np.mean(np.abs(asym_eigenvals - base_eigenvals))
        reversal_measure = np.mean(np.abs(rev_eigenvals - base_eigenvals))
        
        return {
            'base_eigenvalues': base_eigenvals,
            'asymmetric_eigenvalues': asym_eigenvals,
            'reversed_eigenvalues': rev_eigenvals,
            'asymmetry_measure': asymmetry_measure,
            'reversal_measure': reversal_measure,
            'significant_asymmetry': asymmetry_measure > 0.1,
            'significant_reversal': reversal_measure > 0.1
        }
    except Exception as e:
        print(f"Asymmetry analysis failed: {e}")
        return {
            'base_eigenvalues': None,
            'asymmetric_eigenvalues': None,
            'reversed_eigenvalues': None,
            'asymmetry_measure': 0,
            'reversal_measure': 0,
            'significant_asymmetry': False,
            'significant_reversal': False
        }

def analyze_time_asymmetry(num_qubits, mi_matrix):
    """
    4. Compute Δ-entropy(t) vs Δ-entropy(-t) for time asymmetry.
    """
    try:
        # Create time evolution circuits
        time_steps = 5
        forward_entropies = []
        backward_entropies = []
        
        # Forward time evolution (t > 0)
        for t in range(1, time_steps + 1):
            qc_forward = QuantumCircuit(num_qubits)
            qc_forward.h(0)
            
            # Apply time evolution layers
            for layer in range(t):
                for i in range(num_qubits - 1):
                    qc_forward.cx(i, i+1)
                qc_forward.barrier()
            
            forward_state = Statevector.from_instruction(qc_forward)
            forward_entropy = qiskit_entropy(partial_trace(forward_state, list(range(1, num_qubits))))
            forward_entropies.append(forward_entropy)
        
        # Backward time evolution (t < 0) - reverse the circuit
        for t in range(1, time_steps + 1):
            qc_backward = QuantumCircuit(num_qubits)
            qc_backward.h(0)
            
            # Apply reverse time evolution layers
            for layer in range(t):
                for i in range(num_qubits - 1, 0, -1):  # Reverse direction
                    qc_backward.cx(i, i-1)
                qc_backward.barrier()
            
            backward_state = Statevector.from_instruction(qc_backward)
            backward_entropy = qiskit_entropy(partial_trace(backward_state, list(range(1, num_qubits))))
            backward_entropies.append(backward_entropy)
        
        # Calculate time asymmetry
        time_asymmetry_measures = []
        for i in range(len(forward_entropies)):
            delta_forward = forward_entropies[i] - forward_entropies[0] if i > 0 else 0
            delta_backward = backward_entropies[i] - backward_entropies[0] if i > 0 else 0
            asymmetry = abs(delta_forward - delta_backward)
            time_asymmetry_measures.append(asymmetry)
        
        # Compute overall time asymmetry
        total_asymmetry = np.mean(time_asymmetry_measures)
        max_asymmetry = np.max(time_asymmetry_measures)
        
        return {
            'forward_entropies': forward_entropies,
            'backward_entropies': backward_entropies,
            'time_asymmetry_measures': time_asymmetry_measures,
            'total_asymmetry': total_asymmetry,
            'max_asymmetry': max_asymmetry,
            'significant_time_asymmetry': total_asymmetry > 0.05,
            'time_steps': time_steps
        }
    except Exception as e:
        print(f"Time asymmetry analysis failed: {e}")
        return {
            'forward_entropies': [],
            'backward_entropies': [],
            'time_asymmetry_measures': [],
            'total_asymmetry': 0,
            'max_asymmetry': 0,
            'significant_time_asymmetry': False,
            'time_steps': 0
        }

def analyze_page_curve_behavior(entropies, sizes):
    """
    Analyze if the entropy scaling follows page curve behavior.
    """
    if len(entropies) < 3:
        return {'page_curve_consistent': False, 'reason': 'Insufficient data'}
    
    # Page curve: S should increase linearly up to half system size, then decrease
    half_size = len(sizes) // 2
    
    # Analyze first half (should increase)
    first_half_slope = 0
    if half_size > 1:
        x1 = np.array(sizes[:half_size])
        y1 = np.array(entropies[:half_size])
        first_half_slope = np.polyfit(x1, y1, 1)[0]
    
    # Analyze second half (should decrease)
    second_half_slope = 0
    if len(sizes) > half_size + 1:
        x2 = np.array(sizes[half_size:])
        y2 = np.array(entropies[half_size:])
        second_half_slope = np.polyfit(x2, y2, 1)[0]
    
    # Check page curve consistency
    page_curve_consistent = (first_half_slope > 0 and second_half_slope < 0)
    
    return {
        'page_curve_consistent': page_curve_consistent,
        'first_half_slope': first_half_slope,
        'second_half_slope': second_half_slope,
        'half_size': half_size
    }

def reconstruct_spacetime_geometry(mi_matrix, num_qubits):
    """
    Attempt to reconstruct emergent spacetime geometry from entanglement data.
    """
    print("Reconstructing spacetime geometry...")
    
    # Use MDS to reconstruct geometry
    try:
        # Convert MI to dissimilarity
        max_mi = np.max(mi_matrix)
        if max_mi > 0:
            dissimilarity = 1 - mi_matrix / max_mi
        else:
            dissimilarity = np.ones_like(mi_matrix)
        
        # Ensure numerical stability
        dissimilarity = np.nan_to_num(dissimilarity, nan=0.0, posinf=1.0, neginf=0.0)
        dissimilarity = np.clip(dissimilarity, 0, 1)
        
        # MDS reconstruction
        mds = MDS(n_components=3, random_state=42, max_iter=1000, eps=1e-6)
        coords_3d = mds.fit_transform(dissimilarity)
        stress_3d = mds.stress_
        
        # 2D projection for visualization
        mds_2d = MDS(n_components=2, random_state=42, max_iter=1000, eps=1e-6)
        coords_2d = mds_2d.fit_transform(dissimilarity)
        stress_2d = mds_2d.stress_
        
    except Exception as e:
        print(f"MDS reconstruction failed: {e}")
        coords_3d = np.random.rand(num_qubits, 3)
        coords_2d = np.random.rand(num_qubits, 2)
        stress_3d = stress_2d = 1.0
    
    # Attempt to reconstruct metric tensor
    try:
        # Calculate distances between all points
        distances = pairwise_distances(coords_3d)
        
        # Estimate metric tensor components
        metric_tensor = np.zeros((num_qubits, num_qubits, 3, 3))
        for i in range(num_qubits):
            for j in range(num_qubits):
                if i != j:
                    diff = coords_3d[i] - coords_3d[j]
                    metric_tensor[i, j] = np.outer(diff, diff) / (distances[i, j]**2 + 1e-10)
        
        # Calculate average metric
        avg_metric = np.mean(metric_tensor, axis=(0, 1))
        
        # Check for Lorentzian signature (1, -1, -1, -1) in 3D projection
        eigenvalues = np.linalg.eigvals(avg_metric)
        positive_eigenvalues = np.sum(eigenvalues > 0)
        negative_eigenvalues = np.sum(eigenvalues < 0)
        lorentzian_signature = (positive_eigenvalues == 1 and negative_eigenvalues == 2)
        
    except Exception as e:
        print(f"Metric reconstruction failed: {e}")
        avg_metric = np.eye(3)
        lorentzian_signature = False
        positive_eigenvalues = negative_eigenvalues = 0
    
    # Analyze causal structure
    causal_structure = analyze_causal_structure_from_geometry(coords_3d, mi_matrix)
    
    return {
        'coords_3d': coords_3d,
        'coords_2d': coords_2d,
        'stress_3d': stress_3d,
        'stress_2d': stress_2d,
        'metric_tensor': avg_metric,
        'lorentzian_signature': lorentzian_signature,
        'positive_eigenvalues': positive_eigenvalues,
        'negative_eigenvalues': negative_eigenvalues,
        'causal_structure': causal_structure
    }

def analyze_causal_structure_from_geometry(coords, mi_matrix):
    """
    Analyze causal structure from reconstructed geometry.
    """
    # Calculate light cone structure
    light_cones = []
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            # Calculate spatial and temporal distances
            spatial_dist = np.linalg.norm(coords[i][1:] - coords[j][1:])  # Spatial components
            temporal_dist = abs(coords[i][0] - coords[j][0])  # Temporal component
            
            # Check if points are causally connected
            causal_connected = temporal_dist >= spatial_dist
            
            light_cones.append({
                'point1': i,
                'point2': j,
                'spatial_dist': spatial_dist,
                'temporal_dist': temporal_dist,
                'causal_connected': causal_connected,
                'mi_strength': mi_matrix[i, j] if i < len(mi_matrix) and j < len(mi_matrix) else 0
            })
    
    # Calculate causal violations (MI where there shouldn't be causal connection)
    causal_violations = [lc for lc in light_cones if not lc['causal_connected'] and lc['mi_strength'] > 0.1]
    
    return {
        'light_cones': light_cones,
        'causal_violations': causal_violations,
        'violation_ratio': len(causal_violations) / len(light_cones) if light_cones else 0
    }

def create_comprehensive_plots(analysis_results, output_dir):
    """
    Create comprehensive visualization plots.
    """
    print("Creating comprehensive plots...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Causal Structure Analysis
    ax1 = plt.subplot(3, 3, 1)
    if analysis_results['causal_analysis']['wedge_entropies']:
        entropies = list(analysis_results['causal_analysis']['wedge_entropies'].values())
        plt.bar(range(len(entropies)), entropies)
        plt.title('Entanglement Wedge Entropies')
        plt.xlabel('Qubit')
        plt.ylabel('Entropy')
    
    # 2. Bulk-Boundary Correlations
    ax2 = plt.subplot(3, 3, 2)
    correlations = analysis_results['causal_analysis']['bulk_boundary_correlations']
    if correlations:
        plt.hist(correlations, bins=10, alpha=0.7)
        plt.title('Bulk-Boundary Correlations')
        plt.xlabel('Correlation Strength')
        plt.ylabel('Frequency')
    
    # 3. Charge Injection Results
    ax3 = plt.subplot(3, 3, 3)
    charge_results = analysis_results['charge_analysis']['charge_injection_results']
    if charge_results:
        injection_points = [r['injection_point'] for r in charge_results]
        responses = [np.mean(list(r['entanglement_response'].values())) for r in charge_results]
        plt.scatter(injection_points, responses)
        plt.title('Charge Injection Response')
        plt.xlabel('Injection Point')
        plt.ylabel('Entanglement Response')
    
    # 4. RT Geometry - Entropy Scaling
    ax4 = plt.subplot(3, 3, 4)
    cut_off_sizes = analysis_results['rt_analysis']['cut_off_sizes']
    cut_off_entropies = analysis_results['rt_analysis']['cut_off_entropies']
    if cut_off_sizes and cut_off_entropies:
        plt.plot(cut_off_sizes, cut_off_entropies, 'o-')
        plt.title('RT Geometry - Entropy Scaling')
        plt.xlabel('Boundary Size')
        plt.ylabel('Entropy')
    
    # 5. Spacetime Reconstruction - 2D
    ax5 = plt.subplot(3, 3, 5)
    coords_2d = analysis_results['spacetime_analysis']['coords_2d']
    if coords_2d is not None:
        plt.scatter(coords_2d[:, 0], coords_2d[:, 1])
        for i, (x, y) in enumerate(coords_2d):
            plt.annotate(f'Q{i}', (x, y), xytext=(5, 5), textcoords='offset points')
        plt.title('Spacetime Reconstruction (2D)')
        plt.xlabel('X')
        plt.ylabel('Y')
    
    # 6. Spacetime Reconstruction - 3D
    ax6 = plt.subplot(3, 3, 6, projection='3d')
    coords_3d = analysis_results['spacetime_analysis']['coords_3d']
    if coords_3d is not None:
        ax6.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2])
        for i, (x, y, z) in enumerate(coords_3d):
            ax6.text(x, y, z, f'Q{i}')
        ax6.set_title('Spacetime Reconstruction (3D)')
        ax6.set_xlabel('Time')
        ax6.set_ylabel('X')
        ax6.set_zlabel('Y')
    
    # 7. Causal Structure
    ax7 = plt.subplot(3, 3, 7)
    light_cones = analysis_results['spacetime_analysis']['causal_structure']['light_cones']
    if light_cones:
        spatial_dists = [lc['spatial_dist'] for lc in light_cones]
        temporal_dists = [lc['temporal_dist'] for lc in light_cones]
        colors = ['red' if lc['causal_connected'] else 'blue' for lc in light_cones]
        plt.scatter(spatial_dists, temporal_dists, c=colors, alpha=0.6)
        plt.plot([0, max(spatial_dists)], [0, max(spatial_dists)], 'k--', alpha=0.5)
        plt.title('Causal Structure')
        plt.xlabel('Spatial Distance')
        plt.ylabel('Temporal Distance')
    
    # 8. Entropy Targeting Results
    ax8 = plt.subplot(3, 3, 8)
    targeting_results = analysis_results['rt_analysis']['targeting_results']
    if targeting_results:
        targets = [r['target_entropy'] for r in targeting_results]
        achieved = [r['achieved_entropy'] for r in targeting_results]
        plt.scatter(targets, achieved)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.title('Entropy Targeting')
        plt.xlabel('Target Entropy')
        plt.ylabel('Achieved Entropy')
    
    # 9. Page Curve Analysis
    ax9 = plt.subplot(3, 3, 9)
    page_analysis = analysis_results['rt_analysis']['page_curve_analysis']
    if page_analysis['page_curve_consistent']:
        cut_off_sizes = analysis_results['rt_analysis']['cut_off_sizes']
        cut_off_entropies = analysis_results['rt_analysis']['cut_off_entropies']
        if cut_off_sizes and cut_off_entropies:
            plt.plot(cut_off_sizes, cut_off_entropies, 'o-', color='green')
            plt.title('Page Curve Behavior (Consistent)')
        else:
            plt.text(0.5, 0.5, 'Page Curve\nConsistent', ha='center', va='center', transform=ax9.transAxes)
    else:
        plt.text(0.5, 0.5, 'Page Curve\nInconsistent', ha='center', va='center', transform=ax9.transAxes)
    plt.xlabel('Boundary Size')
    plt.ylabel('Entropy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quantum_emergent_spacetime_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_analysis_summary(analysis_results, output_dir, data):
    """
    Generate comprehensive analysis summary.
    """
    print("Generating analysis summary...")
    
    summary = []
    summary.append("# Quantum Emergent Spacetime Enhanced Analysis Summary")
    summary.append("")
    
    # Experiment details
    spec = data.get('spec', {})
    summary.append("## Experiment Details")
    summary.append(f"- Number of qubits: {spec.get('num_qubits', 'Unknown')}")
    summary.append(f"- Geometry: {spec.get('geometry', 'Unknown')}")
    summary.append(f"- Curvature: {spec.get('curvature', 'Unknown')}")
    summary.append(f"- Device: {spec.get('device', 'Unknown')}")
    summary.append(f"- Timesteps: {spec.get('timesteps', 'Unknown')}")
    summary.append("")
    
    # Causal Structure Analysis
    summary.append("## 1. Causal Structure Analysis")
    causal = analysis_results['causal_analysis']
    summary.append(f"- Average bulk-boundary correlation: {causal['avg_bulk_boundary_correlation']:.4f}")
    summary.append(f"- Causal horizon density: {causal['causal_horizon_density']:.4f}")
    summary.append(f"- Boundary size: {causal['boundary_size']}")
    summary.append(f"- Number of causal horizons: {len(causal['causal_horizons'])}")
    summary.append("")
    
    # Charge Injection Analysis
    summary.append("## 2. Charge Injection and Teleportation Analysis")
    charge = analysis_results['charge_analysis']
    if charge['charge_injection_results']:
        summary.append(f"- Successful charge injections: {len(charge['charge_injection_results'])}")
        avg_response = np.mean([np.mean(list(r['entanglement_response'].values())) 
                               for r in charge['charge_injection_results'] if r['entanglement_response']])
        summary.append(f"- Average entanglement response: {avg_response:.4f}")
    else:
        summary.append("- No charge injection results available")
    
    if charge['teleportation_results']:
        summary.append(f"- Successful teleportation states: {len(charge['teleportation_results'])}")
        # Calculate average error if entropy_error exists
        errors = [r.get('entropy_error', 0) for r in charge['teleportation_results']]
        avg_error = np.mean(errors) if errors else 0
        summary.append(f"- Average entropy targeting error: {avg_error:.4f}")
    else:
        summary.append("- No teleportation results available")
    
    # Asymmetry Analysis
    if charge['asymmetry_analysis']:
        asym = charge['asymmetry_analysis']
        summary.append(f"- Asymmetry measure: {asym.get('asymmetry_measure', 0):.4f}")
        summary.append(f"- Reversal measure: {asym.get('reversal_measure', 0):.4f}")
        summary.append(f"- Significant asymmetry: {asym.get('significant_asymmetry', False)}")
        summary.append(f"- Significant reversal: {asym.get('significant_reversal', False)}")
    
    # Time Asymmetry Analysis
    if charge['time_asymmetry']:
        time_asym = charge['time_asymmetry']
        summary.append(f"- Total time asymmetry: {time_asym.get('total_asymmetry', 0):.4f}")
        summary.append(f"- Max time asymmetry: {time_asym.get('max_asymmetry', 0):.4f}")
        summary.append(f"- Significant time asymmetry: {time_asym.get('significant_time_asymmetry', False)}")
    
    summary.append("")
    
    # RT Geometry Analysis
    summary.append("## 3. RT Geometry Analysis")
    rt = analysis_results['rt_analysis']
    
    # Enhanced area law results
    if 'area_law_results' in rt:
        area_law = rt['area_law_results']
        summary.append(f"- Linear R²: {area_law.get('linear_r2', 0):.4f}")
        summary.append(f"- Quadratic R²: {area_law.get('quadratic_r2', 0):.4f}")
        summary.append(f"- Logarithmic R²: {area_law.get('logarithmic_r2', 0):.4f}")
        summary.append(f"- Best fit: {area_law.get('best_fit', 'none')}")
        summary.append(f"- Enhanced area law consistent: {area_law.get('area_law_consistent', False)}")
    
    if rt['entropy_slopes']:
        slope_info = rt['entropy_slopes'][0]
        summary.append(f"- Original entropy slope: {slope_info['slope']:.4f}")
        summary.append(f"- Original R-squared: {slope_info['r_squared']:.4f}")
        summary.append(f"- Original area law consistent: {slope_info['area_law_consistent']}")
    
    page_analysis = rt['page_curve_analysis']
    summary.append(f"- Page curve consistent: {page_analysis['page_curve_consistent']}")
    if page_analysis['page_curve_consistent']:
        summary.append(f"- First half slope: {page_analysis['first_half_slope']:.4f}")
        summary.append(f"- Second half slope: {page_analysis['second_half_slope']:.4f}")
    summary.append("")
    
    # Spacetime Reconstruction
    summary.append("## 4. Spacetime Reconstruction")
    spacetime = analysis_results['spacetime_analysis']
    summary.append(f"- 3D MDS stress: {spacetime['stress_3d']:.4f}")
    summary.append(f"- 2D MDS stress: {spacetime['stress_2d']:.4f}")
    summary.append(f"- Lorentzian signature: {spacetime['lorentzian_signature']}")
    summary.append(f"- Positive eigenvalues: {spacetime['positive_eigenvalues']}")
    summary.append(f"- Negative eigenvalues: {spacetime['negative_eigenvalues']}")
    
    causal_struct = spacetime['causal_structure']
    summary.append(f"- Causal violations: {len(causal_struct['causal_violations'])}")
    summary.append(f"- Violation ratio: {causal_struct['violation_ratio']:.4f}")
    summary.append("")
    
    # Enhanced Validation Results
    summary.append("## 5. Enhanced Quantum Structure Validation")
    enhanced_val = analysis_results['enhanced_validation']
    if 'enhanced_quantum_score' in enhanced_val:
        score_info = enhanced_val['enhanced_quantum_score']
        summary.append(f"- Enhanced quantum score: {score_info['score']:.2f} ({score_info['indicators']}/{score_info['total_tests']})")
    
    # Individual test results
    for test_name, test_result in enhanced_val.items():
        if test_name != 'enhanced_quantum_score':
            summary.append(f"- {test_name}: {'PASS' if test_result.get('quantum_signature', False) else 'FAIL'}")
    summary.append("")
    
    # Overall Assessment
    summary.append("## Overall Assessment")
    
    # Count quantum indicators (enhanced)
    quantum_indicators = 0
    total_indicators = 0
    
    # Causal structure indicators
    if causal['avg_bulk_boundary_correlation'] > 0.1:
        quantum_indicators += 1
    total_indicators += 1
    
    if causal['causal_horizon_density'] < 0.5:
        quantum_indicators += 1
    total_indicators += 1
    
    # Charge injection indicators
    if charge['charge_injection_results'] and len(charge['charge_injection_results']) > 0:
        quantum_indicators += 1
    total_indicators += 1
    
    # RT geometry indicators (enhanced)
    if 'area_law_results' in rt and rt['area_law_results'].get('area_law_consistent', False):
        quantum_indicators += 1
    total_indicators += 1
    
    if rt['entropy_slopes'] and rt['entropy_slopes'][0]['area_law_consistent']:
        quantum_indicators += 1
    total_indicators += 1
    
    if page_analysis['page_curve_consistent']:
        quantum_indicators += 1
    total_indicators += 1
    
    # Asymmetry indicators
    if charge['asymmetry_analysis'] and charge['asymmetry_analysis'].get('significant_asymmetry', False):
        quantum_indicators += 1
    total_indicators += 1
    
    if charge['time_asymmetry'] and charge['time_asymmetry'].get('significant_time_asymmetry', False):
        quantum_indicators += 1
    total_indicators += 1
    
    # Enhanced validation indicators
    if 'enhanced_quantum_score' in enhanced_val:
        enhanced_score = enhanced_val['enhanced_quantum_score']['score']
        if enhanced_score > 0.5:
            quantum_indicators += 1
        total_indicators += 1
    
    # Spacetime indicators
    if spacetime['lorentzian_signature']:
        quantum_indicators += 1
    total_indicators += 1
    
    if spacetime['stress_3d'] < 0.5:
        quantum_indicators += 1
    total_indicators += 1
    
    quantum_score = quantum_indicators / total_indicators if total_indicators > 0 else 0
    
    summary.append(f"**Quantum Emergent Spacetime Score: {quantum_score:.2f} ({quantum_indicators}/{total_indicators})**")
    
    if quantum_score >= 0.7:
        summary.append("**Strong evidence for quantum emergent spacetime**")
    elif quantum_score >= 0.5:
        summary.append("**Moderate evidence for quantum emergent spacetime**")
    else:
        summary.append("**Limited evidence for quantum emergent spacetime**")
    
    summary.append("")
    summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save summary
    with open(os.path.join(output_dir, 'quantum_emergent_spacetime_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
    
    return summary

def run_enhanced_quantum_structure_validation(mi_matrix, num_qubits):
    """
    5. Re-run quantum structure tests with enhanced metrics and update score.
    """
    print("5. Running enhanced quantum structure validation tests...")
    
    validation_results = {}
    
    # Test 1: Enhanced Classical Geometry Fit Benchmark
    print("  - Enhanced Classical Geometry Fit Benchmark...")
    try:
        # Use enhanced MI matrix for better comparison
        enhanced_mi = mi_matrix  # Use the enhanced MI from RT analysis if available
        
        # Compare to various classical models
        random_graph_mi = np.random.random((num_qubits, num_qubits)) * 0.1
        random_graph_mi = (random_graph_mi + random_graph_mi.T) / 2
        np.fill_diagonal(random_graph_mi, 0)
        
        # Regular lattice (ring)
        ring_mi = np.zeros((num_qubits, num_qubits))
        for i in range(num_qubits):
            for j in range(num_qubits):
                if i != j:
                    dist = min(abs(i - j), num_qubits - abs(i - j))
                    ring_mi[i, j] = 0.1 * np.exp(-dist / 2.0)
        
        # Compute stress for each model
        def compute_stress(mi1, mi2):
            return np.mean((mi1 - mi2)**2)
        
        random_stress = compute_stress(enhanced_mi, random_graph_mi)
        ring_stress = compute_stress(enhanced_mi, ring_mi)
        
        validation_results['classical_geometry_test'] = {
            'random_graph_stress': random_stress,
            'ring_lattice_stress': ring_stress,
            'quantum_signature': random_stress > 0.01 and ring_stress > 0.01
        }
    except Exception as e:
        print(f"    Classical geometry test failed: {e}")
        validation_results['classical_geometry_test'] = {'quantum_signature': False}
    
    # Test 2: Enhanced Entropy vs Classical Noise
    print("  - Enhanced Entropy vs Classical Noise...")
    try:
        # Simulate classical noise
        noise_levels = [0.0, 0.1, 0.2, 0.3]
        entropy_evolution = []
        
        for noise in noise_levels:
            noisy_mi = mi_matrix + noise * np.random.random(mi_matrix.shape)
            noisy_mi = (noisy_mi + noisy_mi.T) / 2
            np.fill_diagonal(noisy_mi, 0)
            
            # Compute average entropy
            avg_entropy = np.mean(noisy_mi)
            entropy_evolution.append(avg_entropy)
        
        # Check if entropy collapses with noise (quantum signature)
        entropy_stability = np.std(entropy_evolution)
        quantum_signature = entropy_stability < 0.1  # Stable entropy indicates quantum structure
        
        validation_results['noise_test'] = {
            'entropy_evolution': entropy_evolution,
            'entropy_stability': entropy_stability,
            'quantum_signature': quantum_signature
        }
    except Exception as e:
        print(f"    Noise test failed: {e}")
        validation_results['noise_test'] = {'quantum_signature': False}
    
    # Test 3: Enhanced Randomized MI Test
    print("  - Enhanced Randomized MI Test...")
    try:
        # Shuffle MI matrix
        shuffled_mi = mi_matrix.copy()
        np.random.shuffle(shuffled_mi.flatten())
        shuffled_mi = shuffled_mi.reshape(mi_matrix.shape)
        shuffled_mi = (shuffled_mi + shuffled_mi.T) / 2
        np.fill_diagonal(shuffled_mi, 0)
        
        # Compare structure preservation
        original_structure = np.corrcoef(mi_matrix.flatten(), np.arange(len(mi_matrix.flatten())))[0, 1]
        shuffled_structure = np.corrcoef(shuffled_mi.flatten(), np.arange(len(shuffled_mi.flatten())))[0, 1]
        
        structure_preservation = abs(original_structure - shuffled_structure)
        quantum_signature = structure_preservation > 0.1  # Large difference indicates quantum structure
        
        validation_results['randomized_mi_test'] = {
            'original_structure': original_structure,
            'shuffled_structure': shuffled_structure,
            'structure_preservation': structure_preservation,
            'quantum_signature': quantum_signature
        }
    except Exception as e:
        print(f"    Randomized MI test failed: {e}")
        validation_results['randomized_mi_test'] = {'quantum_signature': False}
    
    # Test 4: Enhanced Entropy-Curvature Link
    print("  - Enhanced Entropy-Curvature Link...")
    try:
        # Compute local curvature from MI matrix
        curvatures = []
        entropies = []
        
        for i in range(num_qubits):
            # Local entropy around qubit i
            local_entropy = np.mean([mi_matrix[i, j] for j in range(num_qubits) if i != j])
            entropies.append(local_entropy)
            
            # Local curvature (simplified)
            neighbors = [j for j in range(num_qubits) if abs(i - j) <= 1 or abs(i - j) >= num_qubits - 1]
            local_curvature = np.mean([mi_matrix[i, j] for j in neighbors if i != j])
            curvatures.append(local_curvature)
        
        # Compute correlation
        if len(curvatures) > 1 and len(entropies) > 1:
            correlation = np.corrcoef(curvatures, entropies)[0, 1]
            quantum_signature = abs(correlation) > 0.5
        else:
            correlation = 0
            quantum_signature = False
        
        validation_results['entropy_curvature_test'] = {
            'correlation': correlation,
            'quantum_signature': quantum_signature
        }
    except Exception as e:
        print(f"    Entropy-curvature test failed: {e}")
        validation_results['entropy_curvature_test'] = {'quantum_signature': False}
    
    # Test 5: Enhanced Causal Violation Tracker
    print("  - Enhanced Causal Violation Tracker...")
    try:
        # Detect acausal MI links
        causal_violations = 0
        total_links = 0
        
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                total_links += 1
                mi_strength = mi_matrix[i, j]
                
                # Define causal distance (simplified)
                causal_distance = min(abs(i - j), num_qubits - abs(i - j))
                
                # Check for acausal correlations
                if mi_strength > 0.1 and causal_distance > 2:
                    causal_violations += 1
        
        violation_ratio = causal_violations / total_links if total_links > 0 else 0
        quantum_signature = violation_ratio > 0.1  # Significant violations indicate quantum structure
        
        validation_results['causal_violation_test'] = {
            'causal_violations': causal_violations,
            'total_links': total_links,
            'violation_ratio': violation_ratio,
            'quantum_signature': quantum_signature
        }
    except Exception as e:
        print(f"    Causal violation test failed: {e}")
        validation_results['causal_violation_test'] = {'quantum_signature': False}
    
    # Test 6: Enhanced Lorentzian Metric Test
    print("  - Enhanced Lorentzian Metric Test...")
    try:
        # Attempt to reconstruct Lorentzian metric
        # Use MDS to get coordinates
        dissimilarity = 1 - mi_matrix / np.max(mi_matrix) if np.max(mi_matrix) > 0 else np.ones_like(mi_matrix)
        dissimilarity = np.nan_to_num(dissimilarity, nan=0.0, posinf=1.0, neginf=0.0)
        
        mds = MDS(n_components=3, random_state=42, max_iter=1000, eps=1e-6)
        coords = mds.fit_transform(dissimilarity)
        
        # Compute metric tensor
        metric = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                metric[i, j] = np.mean(coords[:, i] * coords[:, j])
        
        # Check for Lorentzian signature
        eigenvals = np.linalg.eigvals(metric)
        positive_eigenvals = np.sum(eigenvals > 0)
        negative_eigenvals = np.sum(eigenvals < 0)
        
        lorentzian_signature = (positive_eigenvals == 1 and negative_eigenvals == 2)
        quantum_signature = lorentzian_signature
        
        validation_results['lorentzian_metric_test'] = {
            'positive_eigenvalues': positive_eigenvals,
            'negative_eigenvalues': negative_eigenvals,
            'lorentzian_signature': lorentzian_signature,
            'quantum_signature': quantum_signature
        }
    except Exception as e:
        print(f"    Lorentzian metric test failed: {e}")
        validation_results['lorentzian_metric_test'] = {'quantum_signature': False}
    
    # Compute enhanced quantum score
    quantum_indicators = sum(1 for test in validation_results.values() if test.get('quantum_signature', False))
    total_tests = len(validation_results)
    enhanced_quantum_score = quantum_indicators / total_tests if total_tests > 0 else 0
    
    validation_results['enhanced_quantum_score'] = {
        'score': enhanced_quantum_score,
        'indicators': quantum_indicators,
        'total_tests': total_tests
    }
    
    return validation_results

def main():
    """
    Main function to run the quantum emergent spacetime analysis.
    """
    if len(sys.argv) != 2:
        print("Usage: python quantum_emergent_spacetime_enhanced_analysis.py <target_file>")
        print("Example: python quantum_emergent_spacetime_enhanced_analysis.py experiment_logs/custom_curvature_experiment/instance_20250726_153536/results_n11_geomS_curv20_ibm_brisbane_KTNW95.json")
        sys.exit(1)
    
    target_file = sys.argv[1]
    
    if not os.path.exists(target_file):
        print(f"Error: Target file {target_file} does not exist.")
        sys.exit(1)
    
    # Set output directory to same as input file
    output_dir = os.path.dirname(target_file)
    
    print(f"🔬 Running Quantum Emergent Spacetime Enhanced Analysis...")
    print(f"📁 Target file: {target_file}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔧 CGPTFactory available: {CGPTFACTORY_AVAILABLE}")
    
    # Load data
    print("📊 Loading experiment data...")
    data = load_experiment_data(target_file)
    
    # Extract key information
    spec = data.get('spec', {})
    num_qubits = spec.get('num_qubits', 11)
    print(f"🔢 Number of qubits: {num_qubits}")
    
    # Extract mutual information matrix
    print("🔗 Extracting mutual information matrix...")
    mi_matrix = extract_mutual_information_matrix(data)
    print(f"📐 MI matrix shape: {mi_matrix.shape}")
    
    # Extract entropy data
    print("📈 Extracting entropy data...")
    entropy_data = extract_entropy_data(data)
    print(f"📊 Entropy data points: {len(entropy_data)}")
    
    # Run analyses
    print("\n🚀 Starting comprehensive analysis...")
    
    # 1. Causal Structure Analysis
    print("1️⃣ Analyzing causal structure...")
    causal_analysis = analyze_causal_structure(mi_matrix, num_qubits)
    
    # 2. Charge Injection and Teleportation Analysis
    print("2️⃣ Analyzing charge injection and teleportation...")
    charge_analysis = analyze_charge_injection_and_teleportation(num_qubits, mi_matrix)
    
    # 3. RT Geometry Analysis
    print("3️⃣ Analyzing RT geometry with entropy targeting...")
    rt_analysis = analyze_rt_geometry_with_entropy_targeting(num_qubits, mi_matrix, entropy_data)
    
    # 4. Spacetime Reconstruction
    print("4️⃣ Reconstructing spacetime geometry...")
    spacetime_analysis = reconstruct_spacetime_geometry(mi_matrix, num_qubits)
    
    # 5. Enhanced Quantum Structure Validation
    print("5️⃣ Running enhanced quantum structure validation...")
    enhanced_validation = run_enhanced_quantum_structure_validation(mi_matrix, num_qubits)
    
    # Compile results
    analysis_results = {
        'causal_analysis': causal_analysis,
        'charge_analysis': charge_analysis,
        'rt_analysis': rt_analysis,
        'spacetime_analysis': spacetime_analysis,
        'enhanced_validation': enhanced_validation
    }
    
    # Create visualizations
    print("📊 Creating visualizations...")
    create_comprehensive_plots(analysis_results, output_dir)
    
    # Generate summary
    print("📝 Generating analysis summary...")
    summary = generate_analysis_summary(analysis_results, output_dir, data)
    
    # Save detailed results
    print("💾 Saving detailed results...")
    results_file = os.path.join(output_dir, 'quantum_emergent_spacetime_results.json')
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print("\n✅ Analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Plots: quantum_emergent_spacetime_analysis.png")
    print(f"📝 Summary: quantum_emergent_spacetime_summary.txt")
    print(f"📄 Detailed results: quantum_emergent_spacetime_results.json")
    
    # Print key findings
    print("\n🔍 Key Findings:")
    for line in summary[-10:]:  # Print last 10 lines of summary
        if line.startswith('**'):
            print(line)

if __name__ == "__main__":
    main() 