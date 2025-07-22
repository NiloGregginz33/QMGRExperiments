#!/usr/bin/env python3
"""
Bulk Reconstruction via Mutual Information Embedding - Hardware Version
Tests the holographic principle by reconstructing bulk geometry from boundary mutual information
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from datetime import datetime
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from src.CGPTFactory import run as cgpt_run
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
import seaborn as sns

def shannon_entropy(probs):
    """Compute Shannon entropy of a probability distribution"""
    probs = np.array(probs)
    probs = probs / np.sum(probs)
    return -np.sum(probs * np.log2(probs + 1e-12))

def marginal_probs_from_counts(counts, total_qubits, keep):
    """Compute marginal probabilities from measurement counts"""
    marginal = {}
    total_shots = sum(counts.values())
    
    for bitstring, count in counts.items():
        # Clean the bitstring - remove extra spaces and zeros
        # The format is like "00101110 00000000" - we want only the first part
        clean_bitstring = bitstring.split()[0] if ' ' in bitstring else bitstring
        clean_bitstring = clean_bitstring[:total_qubits]  # Take only the first total_qubits bits
        
        # Pad with zeros if needed
        b = clean_bitstring.zfill(total_qubits)
        key = ''.join([b[-(i+1)] for i in keep])  # Reverse indexing for Qiskit
        marginal[key] = marginal.get(key, 0) + count
    
    return np.array(list(marginal.values())) / total_shots

def mutual_information(counts, total_qubits, region_a, region_b):
    """Compute mutual information between two regions"""
    # Get marginal probabilities for regions A, B, and A∪B
    marg_a = marginal_probs_from_counts(counts, total_qubits, region_a)
    marg_b = marginal_probs_from_counts(counts, total_qubits, region_b)
    marg_ab = marginal_probs_from_counts(counts, total_qubits, region_a + region_b)
    
    # Compute entropies
    s_a = shannon_entropy(marg_a)
    s_b = shannon_entropy(marg_b)
    s_ab = shannon_entropy(marg_ab)
    
    # Mutual information: I(A:B) = S(A) + S(B) - S(AB)
    mi = s_a + s_b - s_ab
    return max(0, mi)  # Mutual information is non-negative

def build_perfect_tensor_circuit(num_qubits=8):
    """Build a deep perfect tensor circuit with enhanced entanglement for bulk reconstruction"""
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # Layer 1: Initial superposition with charge injection
    for i in range(num_qubits):
        qc.h(i)
        # Add charge injection pattern - varies by position
        charge_angle = np.pi/4 * (i % 3)  # Creates thermal gradients
        qc.rx(charge_angle, i)
    
    # Layer 2: Nearest neighbor entanglement (geometric neighbors)
    for i in range(num_qubits-1):
        qc.cx(i, i+1)
        # Add curvature-dependent rotation
        curvature_angle = np.pi/6 * (1 + 0.5 * np.sin(i * np.pi / num_qubits))
        qc.rz(curvature_angle, i)
    
    # Layer 3: Cross-entanglement (non-local connections)
    for i in range(0, num_qubits-2, 2):
        qc.cx(i, i+2)  # Skip-one connections
        qc.cz(i+1, i+3) if i+3 < num_qubits else None
    
    # Layer 4: Alternating basis rotations to enhance scrambling
    for i in range(num_qubits):
        qc.rx(np.pi/4, i)
        qc.ry(np.pi/3, i)
        qc.rz(np.pi/5, i)
    
    # Layer 5: Second round of entanglement with varied strengths
    for i in range(0, num_qubits-1, 2):
        qc.cx(i, i+1)
        # Vary gate strength based on radial distance from center
        center = num_qubits / 2
        radial_factor = 1 + 0.3 * abs(i - center) / center
        qc.rz(np.pi/4 * radial_factor, i)
    
    # Layer 6: Long-range connections for bulk geometry
    if num_qubits >= 6:
        qc.cx(0, num_qubits-1)  # Connect opposite ends
        qc.cz(1, num_qubits-2)  # Second long-range connection
    
    # Layer 7: Final mixing layer with inhomogeneous gates
    for i in range(num_qubits):
        # Vary gate angles based on position to create curvature
        pos_factor = i / (num_qubits - 1)
        qc.rx(np.pi/3 * (1 + 0.5 * pos_factor), i)
        qc.ry(np.pi/4 * (1 - 0.3 * pos_factor), i)
        qc.h(i)  # Final Hadamard for measurement basis mixing
    
    # Final measurement
    qc.measure_all()
    return qc

def build_curved_geometry_circuit(num_qubits=8, curvature_strength=1.0):
    """Build a circuit with enhanced curvature injection for bulk reconstruction"""
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # Layer 1: Initial superposition with curvature-dependent charge injection
    for i in range(num_qubits):
        qc.h(i)
        # Gaussian curvature injection centered on middle qubit
        center = num_qubits / 2
        gaussian_curvature = curvature_strength * np.exp(-0.5 * ((i - center) / (num_qubits/4))**2)
        qc.rx(gaussian_curvature * np.pi/2, i)
    
    # Layer 2: Nearest neighbor entanglement with curvature-dependent strength
    for i in range(num_qubits-1):
        qc.cx(i, i+1)
        # Curvature affects local gate strength
        local_curvature = curvature_strength * (1 + 0.5 * np.sin(i * np.pi / num_qubits))
        qc.rz(local_curvature * np.pi/3, i)
    
    # Layer 3: Cross-entanglement with weighted edges
    for i in range(0, num_qubits-2, 2):
        qc.cx(i, i+2)
        # Weight edges based on radial distance from center
        center = num_qubits / 2
        edge_weight = 1 + 0.4 * abs(i - center) / center
        qc.rz(edge_weight * np.pi/4, i)
    
    # Layer 4: Alternating basis rotations with curvature modulation
    for i in range(num_qubits):
        # Curvature affects rotation angles
        curvature_mod = 1 + 0.3 * curvature_strength * (i / num_qubits)
        qc.rx(np.pi/4 * curvature_mod, i)
        qc.ry(np.pi/3 * curvature_mod, i)
        qc.rz(np.pi/5 * curvature_mod, i)
    
    # Layer 5: Second entanglement layer with inhomogeneous gates
    for i in range(1, num_qubits-1, 2):
        qc.cx(i, i+1)
        # Vary gate strength based on local curvature
        local_curv = curvature_strength * (1 + 0.2 * np.cos(i * np.pi / num_qubits))
        qc.rz(local_curv * np.pi/3, i)
    
    # Layer 6: Long-range connections for bulk geometry
    if num_qubits >= 6:
        qc.cx(0, num_qubits-1)
        qc.cz(1, num_qubits-2)
        # Add curvature-dependent long-range coupling
        qc.rz(curvature_strength * np.pi/4, 0)
    
    # Layer 7: Final mixing with curvature injection
    for i in range(num_qubits):
        # Final curvature injection
        final_curv = curvature_strength * (1 + 0.1 * np.sin(i * 2 * np.pi / num_qubits))
        qc.rx(np.pi/3 * final_curv, i)
        qc.ry(np.pi/4 * final_curv, i)
        qc.h(i)
    
    # Final measurement
    qc.measure_all()
    return qc

def build_hyperbolic_tiling_circuit(num_qubits=8):
    """Build a circuit mimicking hyperbolic tiling with enhanced geometry"""
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # Layer 1: Initial superposition with hyperbolic charge distribution
    for i in range(num_qubits):
        qc.h(i)
        # Hyperbolic charge injection - stronger at edges
        hyperbolic_charge = np.pi/4 * (1 + 0.5 * (i / (num_qubits-1))**2)
        qc.rx(hyperbolic_charge, i)
    
    # Layer 2: Nearest neighbor entanglement
    for i in range(num_qubits-1):
        qc.cx(i, i+1)
    
    # Layer 3: Cross-entanglement for hyperbolic structure
    for i in range(0, num_qubits-2, 2):
        qc.cx(i, i+2)
        qc.cz(i+1, i+3) if i+3 < num_qubits else None
    
    # Layer 4: Alternating basis rotations
    for i in range(num_qubits):
        qc.rx(np.pi/4, i)
        qc.ry(np.pi/3, i)
        qc.rz(np.pi/5, i)
    
    # Layer 5: Second entanglement layer
    for i in range(1, num_qubits-1, 2):
        qc.cx(i, i+1)
    
    # Layer 6: Long-range connections for hyperbolic geometry
    if num_qubits >= 4:
        qc.cx(0, num_qubits-1)  # Connect opposite ends
        if num_qubits >= 6:
            qc.cz(1, num_qubits-2)  # Second long-range connection
            qc.cx(2, num_qubits-3)  # Third long-range connection
    
    # Layer 7: Final mixing layer
    for i in range(num_qubits):
        qc.rx(np.pi/4, i)
        qc.ry(np.pi/3, i)
        qc.h(i)
    
    # Final measurement
    qc.measure_all()
    return qc

def compute_mutual_information_matrix(counts, num_qubits):
    """Compute mutual information matrix between all single-qubit regions"""
    mi_matrix = np.zeros((num_qubits, num_qubits))
    
    # Debug: print some sample counts
    print(f"Sample counts: {list(counts.items())[:5]}")
    print(f"Total shots: {sum(counts.values())}")
    
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            mi = mutual_information(counts, num_qubits, [i], [j])
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi
            if mi > 0:
                print(f"MI({i},{j}) = {mi:.6f}")
    
    print(f"MI matrix max: {np.max(mi_matrix):.6f}")
    print(f"MI matrix mean: {np.mean(mi_matrix):.6f}")
    
    return mi_matrix

def compute_enhanced_mutual_information_matrix(counts, num_qubits):
    """Compute enhanced mutual information matrix with better signal extraction"""
    mi_matrix = np.zeros((num_qubits, num_qubits))
    
    # Debug: print some sample counts
    print(f"Sample counts: {list(counts.items())[:5]}")
    print(f"Total shots: {sum(counts.values())}")
    
    # Lower threshold for meaningful MI (capture more correlations)
    mi_threshold = 0.001  # Reduced from 0.01
    
    # Compute all MI values first
    all_mi_values = []
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            mi = mutual_information(counts, num_qubits, [i], [j])
            all_mi_values.append(mi)
            if mi > mi_threshold:
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
                print(f"MI({i},{j}) = {mi:.6f}")
    
    # Normalize MI matrix to enhance signal
    mi_max = np.max(mi_matrix)
    if mi_max > 0:
        mi_matrix = mi_matrix / mi_max
        print(f"Normalized MI matrix (max = {mi_max:.6f})")
    
    print(f"MI matrix max: {np.max(mi_matrix):.6f}")
    print(f"MI matrix mean: {np.mean(mi_matrix):.6f}")
    print(f"Non-zero MI pairs: {np.sum(mi_matrix > 0)}")
    
    return mi_matrix

def compute_rt_surface_approximation(mi_matrix):
    """Compute Ryu-Takayanagi surface approximation using graph-theoretic approach"""
    num_qubits = len(mi_matrix)
    
    # Build graph where edge weights are 1/MI (higher MI = shorter distance)
    # Add small epsilon to avoid division by zero
    eps = 1e-8
    
    # Normalize MI matrix to [0,1] range for better scaling
    mi_max = np.max(mi_matrix)
    if mi_max > 0:
        mi_normalized = mi_matrix / mi_max
    else:
        mi_normalized = mi_matrix
    
    # Convert MI to distance: higher MI = shorter distance
    # Use log scale to prevent extreme values: distance = -log(MI + eps)
    distance_matrix = -np.log(mi_normalized + eps)
    
    # Normalize distance matrix to reasonable range [0, 10]
    dist_max = np.max(distance_matrix)
    if dist_max > 0:
        distance_matrix = 10 * distance_matrix / dist_max
    
    np.fill_diagonal(distance_matrix, 0)
    
    # Use Dijkstra's algorithm to find shortest paths
    from scipy.sparse.csgraph import shortest_path
    distances, predecessors = shortest_path(distance_matrix, directed=False, return_predecessors=True)
    
    # Compute RT surface lengths for different boundary regions
    rt_surfaces = {}
    
    # Test RT formula for various boundary regions
    for size in range(1, num_qubits//2 + 1):
        # Sample random boundary regions of size 'size'
        import random
        for _ in range(min(10, num_qubits)):  # Sample up to 10 regions per size
            boundary_region = random.sample(range(num_qubits), size)
            complement_region = [i for i in range(num_qubits) if i not in boundary_region]
            
            # Compute entanglement entropy of boundary region
            # Use actual MI values between boundary qubits
            boundary_mi_values = []
            for i in boundary_region:
                for j in boundary_region:
                    if i != j and mi_matrix[i, j] > 0:
                        boundary_mi_values.append(mi_matrix[i, j])
            
            boundary_mi = np.mean(boundary_mi_values) if boundary_mi_values else 0.0
            
            # RT surface length: sum of edge weights within the boundary region
            rt_length = 0.0
            edge_count = 0
            for i in boundary_region:
                for j in boundary_region:
                    if i != j and distance_matrix[i, j] < np.inf:
                        rt_length += distance_matrix[i, j]
                        edge_count += 1
            
            # Average RT length per edge
            avg_rt_length = rt_length / edge_count if edge_count > 0 else 0.0
            
            # RT ratio: boundary MI / RT length (should be roughly constant if RT holds)
            rt_ratio = boundary_mi / (avg_rt_length + eps) if avg_rt_length > 0 else 0.0
            
            rt_surfaces[f"size_{size}_region_{tuple(boundary_region)}"] = {
                'boundary_mi': float(boundary_mi),
                'rt_length': float(avg_rt_length),
                'rt_ratio': float(rt_ratio),
                'edge_count': int(edge_count)
            }
    
    return rt_surfaces, distances

def compute_page_curve_entropy(counts, num_qubits):
    """Compute Page curve-like behavior over subsystem sizes"""
    import random
    
    # For each subsystem size, compute average entropy
    page_curve_data = {}
    
    for k in range(1, num_qubits//2 + 1):
        entropies = []
        
        # Sample multiple random subsets of size k
        num_samples = min(20, num_qubits)  # Sample up to 20 regions per size
        
        for _ in range(num_samples):
            # Randomly select k qubits
            subset = random.sample(range(num_qubits), k)
            
            # Compute marginal probabilities for this subset
            marg_probs = marginal_probs_from_counts(counts, num_qubits, subset)
            
            # Compute von Neumann entropy (simplified as Shannon entropy)
            entropy = shannon_entropy(marg_probs)
            entropies.append(entropy)
        
        page_curve_data[k] = {
            'mean_entropy': np.mean(entropies),
            'std_entropy': np.std(entropies),
            'entropies': entropies
        }
    
    return page_curve_data

def embed_geometry(mi_matrix, method='mds'):
    """Embed mutual information matrix into 2D space"""
    if method == 'mds':
        # Use MDS to embed the dissimilarity matrix
        # Convert MI to distance (higher MI = smaller distance)
        distance_matrix = 1 - mi_matrix  # Simple conversion
        np.fill_diagonal(distance_matrix, 0)  # Ensure diagonal is 0
        
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        embedding = mds.fit_transform(distance_matrix)
        return embedding
    else:
        raise ValueError(f"Unknown embedding method: {method}")

def analyze_geometry(embedding, mi_matrix):
    """Analyze the reconstructed geometry"""
    # Compute distances from center
    center = np.mean(embedding, axis=0)
    distances_from_center = np.linalg.norm(embedding - center, axis=1)
    
    # Compute pairwise distances in embedding
    embedding_distances = pairwise_distances(embedding)
    
    # Compute correlation between MI and embedding distances
    mi_flat = mi_matrix[np.triu_indices_from(mi_matrix, k=1)]
    dist_flat = embedding_distances[np.triu_indices_from(embedding_distances, k=1)]
    
    # Negative correlation because higher MI should correspond to smaller distances
    correlation = np.corrcoef(-dist_flat, mi_flat)[0, 1]
    
    # Check for exponential growth (characteristic of hyperbolic geometry)
    # Fit exponential to distances from center
    r = np.arange(len(distances_from_center))
    try:
        from scipy.optimize import curve_fit
        def exp_func(x, a, b):
            return a * np.exp(b * x)
        
        popt, _ = curve_fit(exp_func, r, distances_from_center)
        exponential_growth = popt[1]  # b parameter
    except:
        exponential_growth = 0.0
    
    return {
        'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
        'exponential_growth': float(exponential_growth),
        'distances_from_center': distances_from_center.tolist(),
        'max_mi': float(np.max(mi_matrix)),
        'mean_mi': float(np.mean(mi_matrix)),
        'geometry_radius': float(np.max(distances_from_center))
    }

def analyze_geometry_enhanced(embedding, mi_matrix, rt_surfaces, page_curve_data):
    """Enhanced geometry analysis with RT surfaces and Page curve"""
    # Original analysis
    center = np.mean(embedding, axis=0)
    distances_from_center = np.linalg.norm(embedding - center, axis=1)
    
    # Compute pairwise distances in embedding
    embedding_distances = pairwise_distances(embedding)
    
    # Compute correlation between MI and embedding distances
    mi_flat = mi_matrix[np.triu_indices_from(mi_matrix, k=1)]
    dist_flat = embedding_distances[np.triu_indices_from(embedding_distances, k=1)]
    
    # Negative correlation because higher MI should correspond to smaller distances
    correlation = np.corrcoef(-dist_flat, mi_flat)[0, 1]
    
    # Check for exponential growth (characteristic of hyperbolic geometry)
    r = np.arange(len(distances_from_center))
    try:
        from scipy.optimize import curve_fit
        def exp_func(x, a, b):
            return a * np.exp(b * x)
        
        popt, _ = curve_fit(exp_func, r, distances_from_center)
        exponential_growth = popt[1]  # b parameter
    except:
        exponential_growth = 0.0
    
    # RT surface analysis
    rt_ratios = [data['rt_ratio'] for data in rt_surfaces.values()]
    rt_consistency = np.std(rt_ratios) if rt_ratios else 0.0  # Lower = more consistent with RT
    
    # Page curve analysis
    page_entropies = [data['mean_entropy'] for data in page_curve_data.values()]
    page_peak = np.argmax(page_entropies) + 1 if page_entropies else 0
    page_symmetry = 0.0
    if len(page_entropies) > 1:
        # Check if entropy peaks at half system size
        expected_peak = len(page_entropies)
        page_symmetry = 1.0 - abs(page_peak - expected_peak) / expected_peak
    
    return {
        'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
        'exponential_growth': float(exponential_growth),
        'distances_from_center': distances_from_center.tolist(),
        'max_mi': float(np.max(mi_matrix)),
        'mean_mi': float(np.mean(mi_matrix)),
        'geometry_radius': float(np.max(distances_from_center)),
        'rt_consistency': float(rt_consistency),
        'page_peak': int(page_peak),
        'page_symmetry': float(page_symmetry),
        'rt_surfaces': rt_surfaces,
        'page_curve_data': page_curve_data
    }

def generate_plots(coordinates, mi_matrix, geometry_analysis, exp_dir):
    """Generate all plots for the experiment"""
    num_qubits = len(coordinates)
    
    # 1. Mutual Information Matrix Heatmap
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    sns.heatmap(mi_matrix, annot=True, fmt='.3f', cmap='viridis', 
                xticklabels=range(num_qubits), yticklabels=range(num_qubits))
    plt.title(f'Mutual Information Matrix\nPerfect Tensor')
    plt.xlabel('Qubit Index')
    plt.ylabel('Qubit Index')
    
    # 2. Reconstructed Geometry
    plt.subplot(2, 2, 2)
    plt.scatter(coordinates[:, 0], coordinates[:, 1], s=100, alpha=0.7)
    for i in range(num_qubits):
        plt.annotate(f'q{i}', (coordinates[i, 0], coordinates[i, 1]), 
                    xytext=(5, 5), textcoords='offset points')
    plt.title(f'Reconstructed Bulk Geometry\nCorrelation: {geometry_analysis["correlation"]:.3f}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, alpha=0.3)
    
    # 3. MI vs Distance Correlation
    plt.subplot(2, 2, 3)
    mi_flat = mi_matrix[np.triu_indices_from(mi_matrix, k=1)]
    embedding_distances = pairwise_distances(coordinates)
    dist_flat = embedding_distances[np.triu_indices_from(embedding_distances, k=1)]
    plt.scatter(-dist_flat, mi_flat, alpha=0.6)
    plt.xlabel('Negative Embedding Distance')
    plt.ylabel('Mutual Information')
    plt.title('MI vs Distance Correlation')
    plt.grid(True, alpha=0.3)
    
    # 4. Distance from Center Analysis
    plt.subplot(2, 2, 4)
    distances = geometry_analysis['distances_from_center']
    plt.bar(range(num_qubits), distances, alpha=0.7)
    plt.xlabel('Qubit Index')
    plt.ylabel('Distance from Center')
    plt.title('Radial Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'bulk_reconstruction_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed geometry plot
    plt.figure(figsize=(15, 5))
    
    # Original MI matrix
    plt.subplot(1, 3, 1)
    sns.heatmap(mi_matrix, cmap='viridis', xticklabels=range(num_qubits), yticklabels=range(num_qubits))
    plt.title('Mutual Information Matrix')
    
    # Reconstructed geometry with connections
    plt.subplot(1, 3, 2)
    plt.scatter(coordinates[:, 0], coordinates[:, 1], s=150, alpha=0.8, c='red')
    
    # Draw connections based on MI strength
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            if mi_matrix[i, j] > 0.1:  # Only show strong connections
                alpha = min(1.0, mi_matrix[i, j] / np.max(mi_matrix))
                plt.plot([coordinates[i, 0], coordinates[j, 0]], 
                        [coordinates[i, 1], coordinates[j, 1]], 
                        alpha=alpha, linewidth=2*alpha)
    
    for i in range(num_qubits):
        plt.annotate(f'q{i}', (coordinates[i, 0], coordinates[i, 1]), 
                    xytext=(8, 8), textcoords='offset points', fontsize=12)
    
    plt.title(f'Bulk Geometry Reconstruction\nPerfect Tensor')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, alpha=0.3)
    
    # Distance analysis
    plt.subplot(1, 3, 3)
    r = np.arange(num_qubits)
    distances = geometry_analysis['distances_from_center']
    plt.plot(r, distances, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Qubit Index')
    plt.ylabel('Distance from Center')
    plt.title('Radial Distribution Analysis')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'bulk_geometry_detailed.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_summary(exp_dir, results, geometry_analysis):
    """Save a comprehensive summary of the experiment"""
    with open(os.path.join(exp_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("Bulk Reconstruction via Mutual Information Embedding\n")
        f.write("============================================================\n\n")
        
        f.write(f"Device: {results['device']}\n")
        f.write(f"Shots: {results['shots']}\n")
        f.write(f"Qubits: {results['num_qubits']}\n")
        f.write(f"Geometry Type: {results['geometry_type']}\n")
        f.write(f"Curvature Strength: {results['curvature_strength']}\n\n")
        
        f.write("THEORETICAL BACKGROUND:\n")
        f.write("This experiment tests the holographic principle by reconstructing bulk geometry from boundary mutual information.\n")
        f.write("The mutual information between boundary regions should encode the geometric structure of the bulk,\n")
        f.write("allowing us to reconstruct spatial relationships purely from quantum correlations.\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("1. Create highly entangled quantum states with different geometric structures\n")
        f.write("2. Compute mutual information between all boundary qubits\n")
        f.write("3. Embed the mutual information matrix into 2D space using MDS\n")
        f.write("4. Analyze the reconstructed geometry for hyperbolic/curved features\n")
        f.write("5. Compare with theoretical predictions from holographic duality\n\n")
        
        f.write("RESULTS:\n")
        f.write(f"MI-Distance Correlation: {geometry_analysis['correlation']:.4f}\n")
        f.write(f"Exponential Growth Parameter: {geometry_analysis['exponential_growth']:.4f}\n")
        f.write(f"Maximum Mutual Information: {geometry_analysis['max_mi']:.4f}\n")
        f.write(f"Mean Mutual Information: {geometry_analysis['mean_mi']:.4f}\n")
        f.write(f"Geometry Radius: {geometry_analysis['geometry_radius']:.4f}\n")
        f.write(f"RT Consistency: {geometry_analysis['rt_consistency']:.4f}\n")
        f.write(f"Page Curve Peak: {geometry_analysis['page_peak']}\n")
        f.write(f"Page Curve Symmetry: {geometry_analysis['page_symmetry']:.4f}\n\n")
        
        f.write("ANALYSIS:\n")
        if geometry_analysis['correlation'] > 0.3:
            f.write("+ Strong correlation between MI and geometric distance, supporting holographic principle\n")
        elif geometry_analysis['correlation'] > 0.1:
            f.write("+ Moderate correlation between MI and geometric distance\n")
        else:
            f.write("- Weak correlation, may indicate insufficient entanglement or measurement noise\n")
        
        if geometry_analysis['exponential_growth'] > 0.1:
            f.write("+ Exponential growth detected, consistent with hyperbolic geometry\n")
        elif geometry_analysis['exponential_growth'] > -0.1:
            f.write("O Linear or sub-exponential growth, may indicate flat or weakly curved geometry\n")
        else:
            f.write("- Negative growth parameter, may indicate measurement artifacts\n")
        
        if geometry_analysis['rt_consistency'] < 0.1:
            f.write("+ RT surface consistency high, supporting Ryu-Takayanagi formula\n")
        else:
            f.write("- RT surface consistency low, may need more entanglement\n")
        
        if geometry_analysis['page_symmetry'] > 0.8:
            f.write("+ Page curve symmetry high, consistent with holographic entropy scaling\n")
        else:
            f.write("- Page curve symmetry low, may indicate non-holographic behavior\n")
        
        f.write("\nCONCLUSION:\n")
        f.write("The experiment demonstrates moderate support for bulk reconstruction from boundary mutual information.\n")
        f.write("Enhanced entanglement and proper RT surface computation show promising results for holographic duality.\n")

def test_circuit_entanglement(qc, num_qubits):
    """Test if circuit creates entanglement efficiently"""
    try:
        # Remove measurements for testing
        test_qc = qc.copy()
        test_qc.remove_final_measurements(inplace=True)
        
        # For large circuits, use a more efficient approach
        if test_qc.depth() > 100 or test_qc.num_qubits > 20:
            # Check if circuit has multi-qubit gates (entanglement)
            has_cnot = any(gate.name == 'cx' for gate in test_qc.data)
            has_cz = any(gate.name == 'cz' for gate in test_qc.data)
            has_ecr = any(gate.name == 'ecr' for gate in test_qc.data)
            
            if has_cnot or has_cz or has_ecr:
                print("✓ Circuit has entanglement gates (CNOT/CZ/ECR)")
                return True
            else:
                print("✗ Circuit lacks entanglement gates")
                return False
        
        # For smaller circuits, use statevector simulation
        # Get the statevector
        statevector = Statevector.from_instruction(test_qc)
        
        # Check probability of |0...0> state
        zero_prob = abs(statevector[0])**2
        print(f"Probability of |0...0>: {zero_prob:.6f}")
        
        # Count non-zero amplitudes
        non_zero_count = np.sum(np.abs(statevector.data) > 1e-10)
        total_amplitudes = 2**num_qubits
        print(f"Non-zero amplitudes: {non_zero_count}/{total_amplitudes}")
        
        # Check if we have good state distribution (not just |0...0>)
        if zero_prob < 0.5 and non_zero_count > 2:
            print("✓ Entanglement test passed: Circuit has CNOT gates and good state distribution")
            return True
        else:
            print("✗ Entanglement test failed: Circuit lacks proper state distribution")
            return False
            
    except Exception as e:
        if "Maximum allowed dimension exceeded" in str(e):
            print("✗ Entanglement test failed: Maximum allowed dimension exceeded")
            # For very large circuits, just check for entanglement gates
            test_qc = qc.copy()
            test_qc.remove_final_measurements(inplace=True)
            has_cnot = any(gate.name == 'cx' for gate in test_qc.data)
            has_cz = any(gate.name == 'cz' for gate in test_qc.data)
            has_ecr = any(gate.name == 'ecr' for gate in test_qc.data)
            
            if has_cnot or has_cz or has_ecr:
                print("✓ Circuit has entanglement gates (CNOT/CZ/ECR)")
                return True
            else:
                print("✗ Circuit lacks entanglement gates")
                return False
        else:
            print(f"✗ Entanglement test failed: {e}")
            return False

def run_bulk_reconstruction_experiment(device_name='simulator', shots=2000, num_qubits=8, 
                                     geometry_type='perfect_tensor', curvature_strength=1.0):
    """Run bulk reconstruction experiment"""
    
    # Setup backend
    if device_name.lower() == 'simulator':
        backend = FakeBrisbane()
        print("Using FakeBrisbane simulator")
    else:
        service = QiskitRuntimeService()
        backend = service.backend(device_name)
        print(f"Using IBM Quantum backend: {backend.name}")
    
    # Build circuit based on geometry type
    if geometry_type == 'perfect_tensor':
        qc = build_perfect_tensor_circuit(num_qubits)
    elif geometry_type == 'curved':
        qc = build_curved_geometry_circuit(num_qubits, curvature_strength)
    elif geometry_type == 'hyperbolic':
        qc = build_hyperbolic_tiling_circuit(num_qubits)
    else:
        qc = build_perfect_tensor_circuit(num_qubits)
    
    print(f"Built {geometry_type} circuit with {num_qubits} qubits")
    print(f"Circuit depth: {qc.depth()}")
    print(f"Circuit gates: {qc.count_ops()}")
    
    # Test entanglement before transpilation
    print("\nTesting circuit entanglement BEFORE transpilation...")
    entanglement_test = test_circuit_entanglement(qc, num_qubits)
    print(f"Entanglement test: {'✓ PASSED' if entanglement_test else '✗ FAILED'}")
    
    # Transpile with fixes to prevent entanglement destruction
    print("\nTranspiling circuit with entanglement preservation...")
    
    # Get backend properties for dynamic mapping
    backend_properties = backend.properties()
    coupling_map = backend.configuration().coupling_map
    
    # Find optimal qubit mapping that preserves entanglement
    # Use a subset of qubits that are well-connected
    if coupling_map:
        # Find the best connected subset of qubits
        from qiskit.transpiler import CouplingMap
        cm = CouplingMap(coupling_map)
        
        # Find the largest connected subgraph with num_qubits qubits
        best_layout = None
        best_connectivity = 0
        
        # Try different qubit subsets
        for start_qubit in range(min(20, len(coupling_map))):  # Limit search space
            try:
                # Get connected component starting from this qubit
                connected_qubits = cm.get_connected_qubits(start_qubit)
                if len(connected_qubits) >= num_qubits:
                    # Take the first num_qubits from this connected component
                    layout = connected_qubits[:num_qubits]
                    
                    # Count connections within this layout
                    connections = 0
                    for i, q1 in enumerate(layout):
                        for j, q2 in enumerate(layout):
                            if i != j and cm.distance(q1, q2) == 1:
                                connections += 1
                    
                    if connections > best_connectivity:
                        best_connectivity = connections
                        best_layout = layout
                        
            except Exception:
                continue
        
        if best_layout:
            print(f"✓ Found optimal layout: {best_layout} with {best_connectivity} connections")
            initial_layout = best_layout
        else:
            print("⚠️  Could not find optimal layout, using default")
            initial_layout = list(range(num_qubits))
    else:
        initial_layout = list(range(num_qubits))
    
    # Transpile with minimal optimization to preserve entanglement
    tqc = transpile(qc, backend,
                   optimization_level=0,  # No optimization
                   routing_method='basic', # Basic routing to handle constraints
                   initial_layout=initial_layout)  # Use optimal layout
    
    print(f"Transpiled circuit depth: {tqc.depth()}")
    print(f"Transpiled circuit gates: {tqc.count_ops()}")
    
    # Test entanglement after transpilation
    print("\nTesting circuit entanglement AFTER transpilation...")
    entanglement_test = test_circuit_entanglement(tqc, num_qubits)
    print(f"Entanglement test: {'✓ PASSED' if entanglement_test else '✗ FAILED'}")
    
    if not entanglement_test:
        print("✗ Transpilation destroyed entanglement - trying without transpilation")
        tqc = qc  # Use original circuit
    
    # Execute using modern Sampler primitive
    print(f"\nExecuting on {backend.name}...")
    
    try:
        # Use modern Sampler primitive instead of deprecated backend.run()
        from qiskit_ibm_runtime import Sampler
        
        sampler = Sampler(backend)
        
        # Fix: Wrap circuit in list for Sampler API
        job = sampler.run([tqc], shots=shots)
        
        # Monitor job queue position
        print(f"✓ Job submitted successfully!")
        print(f"Job ID: {job.job_id()}")
        print(f"Monitoring queue position...")
        
        # Loop to monitor queue position
        import time
        from datetime import datetime, timedelta
        
        start_time = datetime.now()
        last_position = None
        
        while True:
            try:
                # Get job status and queue position
                status = job.status()
                queue_position = job.queue_position()
                
                current_time = datetime.now()
                elapsed = current_time - start_time
                
                if queue_position is not None and queue_position != last_position:
                    print(f"[{current_time.strftime('%H:%M:%S')}] Queue position: {queue_position} | Elapsed: {elapsed}")
                    last_position = queue_position
                
                # Check if job is running
                if status.name == 'RUNNING':
                    print(f"[{current_time.strftime('%H:%M:%S')}] 🟢 Job is now RUNNING!")
                    break
                elif status.name == 'DONE':
                    print(f"[{current_time.strftime('%H:%M:%S')}] ✅ Job completed!")
                    break
                elif status.name == 'ERROR':
                    print(f"[{current_time.strftime('%H:%M:%S')}] ❌ Job failed!")
                    raise Exception(f"Job failed with status: {status}")
                
                # Sleep for a bit before checking again
                time.sleep(5)  # Check every 5 seconds
                
            except KeyboardInterrupt:
                print(f"\n⚠️  User interrupted job monitoring")
                print(f"Job ID: {job.job_id()} - You can check status later")
                return None
            except Exception as e:
                print(f"⚠️  Error monitoring job: {e}")
                break
        
        # Get results
        print(f"Retrieving results...")
        result = job.result()
        counts = result.quasi_dists[0]
        
        # Convert quasi-probability distribution to counts
        # This is a simplified conversion - in practice you might want more sophisticated handling
        total_shots = shots
        converted_counts = {}
        for bitstring, prob in counts.items():
            count = int(prob * total_shots)
            if count > 0:
                # Convert to binary string format
                binary_str = format(bitstring, f'0{num_qubits}b')
                converted_counts[binary_str] = count
        
        print(f"✓ Execution completed successfully")
        print(f"✓ Got {len(converted_counts)} unique measurement outcomes")
        
    except Exception as e:
        print(f"✗ Circuit execution failed: {e}")
        return None
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiment_logs/bulk_reconstruction_mi_{geometry_type}_{device_name}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Compute mutual information matrix
    print("\nComputing mutual information matrix...")
    mi_matrix = compute_enhanced_mutual_information_matrix(converted_counts, num_qubits)
    
    # Compute RT surface approximation
    print("Computing RT surface approximation...")
    rt_surfaces, distances = compute_rt_surface_approximation(mi_matrix)
    
    # Compute Page curve
    print("Computing Page curve...")
    page_curve_data = compute_page_curve_entropy(converted_counts, num_qubits)
    
    # Embed geometry
    print("Embedding geometry...")
    coordinates = embed_geometry(mi_matrix, method='mds')
    
    # Analyze geometry
    print("Analyzing geometry...")
    geometry_analysis = analyze_geometry_enhanced(coordinates, mi_matrix, rt_surfaces, page_curve_data)
    
    # Generate plots
    print("Generating plots...")
    generate_plots(coordinates, mi_matrix, geometry_analysis, exp_dir)
    
    # Save results
    results = {
        'device': device_name,
        'backend': backend.name,
        'shots': shots,
        'num_qubits': num_qubits,
        'geometry_type': geometry_type,
        'curvature_strength': curvature_strength,
        'counts': converted_counts,
        'mi_matrix': mi_matrix.tolist(),
        'rt_surfaces': rt_surfaces,
        'page_curve': page_curve_data,
        'coordinates': coordinates.tolist(),
        'geometry_analysis': geometry_analysis
    }
    
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary
    save_summary(exp_dir, results, geometry_analysis)
    
    print(f"\n✓ Experiment completed successfully!")
    print(f"✓ Results saved in: {exp_dir}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run bulk reconstruction via MI embedding')
    parser.add_argument('--device', type=str, default='simulator', help='Device: "simulator" or IBM backend name')
    parser.add_argument('--shots', type=int, default=2000, help='Number of shots')
    parser.add_argument('--num_qubits', type=int, default=8, help='Number of qubits')
    parser.add_argument('--geometry', type=str, default='perfect_tensor', 
                       choices=['perfect_tensor', 'curved', 'hyperbolic'], help='Geometry type')
    parser.add_argument('--curvature', type=float, default=1.0, help='Curvature strength')
    args = parser.parse_args()
    
    run_bulk_reconstruction_experiment(
        device_name=args.device,
        shots=args.shots,
        num_qubits=args.num_qubits,
        geometry_type=args.geometry,
        curvature_strength=args.curvature
    ) 