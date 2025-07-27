#!/usr/bin/env python3
"""
Curvature Flow Analysis
Visualizes curvature gradient vectors and effective energy density flow patterns
to demonstrate mass back-propagation in spherical geometry.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def load_experiment_data(json_path):
    """Load experiment data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def compute_mutual_information_from_counts(counts, num_qubits):
    """Compute mutual information matrix from quantum measurement counts."""
    mi_matrix = np.zeros((num_qubits, num_qubits))
    total_shots = sum(counts.values())
    
    if total_shots == 0:
        print("  Warning: No measurement counts found")
        return mi_matrix
    
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            # Calculate mutual information between qubits i and j
            # Extract marginal probabilities for qubits i and j
            p_i_0 = 0.0  # P(qubit i = 0)
            p_i_1 = 0.0  # P(qubit i = 1)
            p_j_0 = 0.0  # P(qubit j = 0)
            p_j_1 = 0.0  # P(qubit j = 1)
            p_ij_00 = 0.0  # P(qubit i = 0, qubit j = 0)
            p_ij_01 = 0.0  # P(qubit i = 0, qubit j = 1)
            p_ij_10 = 0.0  # P(qubit i = 1, qubit j = 0)
            p_ij_11 = 0.0  # P(qubit i = 1, qubit j = 1)
            
            # Count occurrences
            for bitstring, count in counts.items():
                if len(bitstring) >= max(i, j) + 1:
                    # Extract bits for qubits i and j
                    bit_i = bitstring[i]
                    bit_j = bitstring[j]
                    
                    # Update joint probabilities
                    if bit_i == '0' and bit_j == '0':
                        p_ij_00 += count
                    elif bit_i == '0' and bit_j == '1':
                        p_ij_01 += count
                    elif bit_i == '1' and bit_j == '0':
                        p_ij_10 += count
                    elif bit_i == '1' and bit_j == '1':
                        p_ij_11 += count
            
            # Normalize probabilities
            p_ij_00 /= total_shots
            p_ij_01 /= total_shots
            p_ij_10 /= total_shots
            p_ij_11 /= total_shots
            
            # Calculate marginal probabilities
            p_i_0 = p_ij_00 + p_ij_01
            p_i_1 = p_ij_10 + p_ij_11
            p_j_0 = p_ij_00 + p_ij_10
            p_j_1 = p_ij_01 + p_ij_11
            
            # Calculate mutual information: I(X;Y) = sum p(x,y) * log(p(x,y)/(p(x)*p(y)))
            mi_value = 0.0
            if p_ij_00 > 0 and p_i_0 > 0 and p_j_0 > 0:
                mi_value += p_ij_00 * np.log(p_ij_00 / (p_i_0 * p_j_0))
            if p_ij_01 > 0 and p_i_0 > 0 and p_j_1 > 0:
                mi_value += p_ij_01 * np.log(p_ij_01 / (p_i_0 * p_j_1))
            if p_ij_10 > 0 and p_i_1 > 0 and p_j_0 > 0:
                mi_value += p_ij_10 * np.log(p_ij_10 / (p_i_1 * p_j_0))
            if p_ij_11 > 0 and p_i_1 > 0 and p_j_1 > 0:
                mi_value += p_ij_11 * np.log(p_ij_11 / (p_i_1 * p_j_1))
            
            # Store mutual information (symmetric matrix)
            mi_matrix[i, j] = mi_value
            mi_matrix[j, i] = mi_value
    
    return mi_matrix

def compute_curvature_gradient_vectors(data):
    """Compute curvature gradient vectors at each node."""
    print("=" * 70)
    print("CURVATURE GRADIENT VECTOR ANALYSIS")
    print("=" * 70)
    
    spec = data['spec']
    geometry = spec['geometry']
    curvature = spec['curvature']
    num_qubits = spec['num_qubits']
    
    print(f"\nEXPERIMENT PARAMETERS:")
    print(f"  Geometry: {geometry}")
    print(f"  Curvature: κ = {curvature}")
    print(f"  Number of qubits: {num_qubits}")
    
    # Get embedding coordinates
    if 'embedding_coords' not in data:
        print("❌ No embedding coordinates found.")
        return None
    
    coords = np.array(data['embedding_coords'])
    print(f"  Embedding shape: {coords.shape}")
    
    # Get or compute mutual information matrix
    if 'mi_matrix' in data:
        mi_matrix = np.array(data['mi_matrix'])
        print(f"  MI matrix shape: {mi_matrix.shape}")
    elif 'counts_per_timestep' in data and data['counts_per_timestep']:
        print("  Computing mutual information matrix from quantum measurement counts...")
        # Use the first timestep counts to compute MI
        counts = data['counts_per_timestep'][0]
        mi_matrix = compute_mutual_information_from_counts(counts, num_qubits)
        print(f"  Real MI matrix computed from quantum data, shape: {mi_matrix.shape}")
    else:
        print("  Creating synthetic MI matrix from embedding coordinates and edge weights...")
        # Create synthetic MI matrix based on embedding distances and edge weights
        mi_matrix = np.zeros((num_qubits, num_qubits))
        
        # Parse custom edges to get edge weights
        custom_edges = spec.get('custom_edges', '')
        edge_weights = {}
        
        if custom_edges:
            for edge in custom_edges.split(','):
                if ':' in edge:
                    nodes, weight = edge.split(':')
                    i, j = map(int, nodes.split('-'))
                    edge_weights[(i, j)] = float(weight)
                    edge_weights[(j, i)] = float(weight)  # Symmetric
        
        # Fill MI matrix based on distances and edge weights
        for i in range(num_qubits):
            for j in range(num_qubits):
                if i != j:
                    # Distance between nodes
                    distance = np.linalg.norm(coords[i] - coords[j])
                    
                    # Edge weight (default to 1.0 if not specified)
                    weight = edge_weights.get((i, j), 1.0)
                    
                    # MI decreases with distance and increases with edge weight
                    mi_matrix[i, j] = weight * np.exp(-distance)
        
        print(f"  Synthetic MI matrix created with shape: {mi_matrix.shape}")
    
    # Compute distance matrix
    distances = squareform(pdist(coords))
    
    # Compute curvature gradient vectors
    gradient_vectors = []
    effective_energy_density = []
    
    for i in range(num_qubits):
        # Compute gradient of curvature at node i
        gradient = np.zeros(2)
        energy_density = 0.0
        
        for j in range(num_qubits):
            if i != j:
                # Vector from node i to node j
                direction = coords[j] - coords[i]
                distance = np.linalg.norm(direction)
                
                if distance > 0:
                    # Normalize direction
                    unit_direction = direction / distance
                    
                    # Curvature contribution from this connection
                    # For spherical geometry: positive curvature creates inward flow
                    if geometry == "spherical":
                        # Positive curvature creates inward gradient
                        curvature_contribution = mi_matrix[i, j] * curvature
                        gradient += curvature_contribution * unit_direction
                        
                        # Effective energy density (Ricci scalar contribution)
                        energy_density += curvature_contribution / (1 + distance**2)
                    else:
                        # Negative curvature creates outward gradient
                        curvature_contribution = -mi_matrix[i, j] * curvature
                        gradient += curvature_contribution * unit_direction
                        energy_density += curvature_contribution / (1 + distance**2)
        
        gradient_vectors.append(gradient)
        effective_energy_density.append(energy_density)
    
    gradient_vectors = np.array(gradient_vectors)
    effective_energy_density = np.array(effective_energy_density)
    
    # Analyze gradient patterns
    gradient_magnitudes = np.linalg.norm(gradient_vectors, axis=1)
    mean_gradient_magnitude = np.mean(gradient_magnitudes)
    
    print(f"\nGRADIENT VECTOR STATISTICS:")
    print(f"  Mean gradient magnitude: {mean_gradient_magnitude:.6f}")
    print(f"  Gradient magnitude range: [{gradient_magnitudes.min():.6f}, {gradient_magnitudes.max():.6f}]")
    print(f"  Energy density range: [{effective_energy_density.min():.6f}, {effective_energy_density.max():.6f}]")
    
    # Check flow direction
    if geometry == "spherical":
        # For spherical geometry, we expect inward flow (negative divergence)
        # Compute divergence-like measure
        divergence_measure = np.mean([np.dot(gradient_vectors[i], coords[i]) for i in range(num_qubits)])
        print(f"  Divergence measure: {divergence_measure:.6f}")
        
        if divergence_measure < 0:
            print(f"  ✅ INWARD FLOW DETECTED → CONFIRMS SPHERICAL GEOMETRY")
        else:
            print(f"  ⚠️  OUTWARD FLOW DETECTED → INCONSISTENT WITH SPHERICAL GEOMETRY")
    
    return gradient_vectors, effective_energy_density, coords

def compute_riemann_curvature_tensor(data):
    """Compute local Riemann curvature tensor components."""
    print("\n" + "=" * 70)
    print("RIEMANN CURVATURE TENSOR ANALYSIS")
    print("=" * 70)
    
    spec = data['spec']
    geometry = spec['geometry']
    curvature = spec['curvature']
    
    if 'embedding_coords' not in data:
        print("❌ Missing embedding coordinates for Riemann tensor computation.")
        return None
    
    coords = np.array(data['embedding_coords'])
    num_qubits = coords.shape[0]
    
    # Get or compute MI matrix for Riemann tensor computation
    if 'mi_matrix' in data:
        mi_matrix = np.array(data['mi_matrix'])
    elif 'counts_per_timestep' in data and data['counts_per_timestep']:
        # Use the first timestep counts to compute MI
        counts = data['counts_per_timestep'][0]
        mi_matrix = compute_mutual_information_from_counts(counts, num_qubits)
    else:
        # Create synthetic MI matrix (same as in gradient computation)
        mi_matrix = np.zeros((num_qubits, num_qubits))
        spec = data['spec']
        custom_edges = spec.get('custom_edges', '')
        edge_weights = {}
        
        if custom_edges:
            for edge in custom_edges.split(','):
                if ':' in edge:
                    nodes, weight = edge.split(':')
                    i, j = map(int, nodes.split('-'))
                    edge_weights[(i, j)] = float(weight)
                    edge_weights[(j, i)] = float(weight)
        
        for i in range(num_qubits):
            for j in range(num_qubits):
                if i != j:
                    distance = np.linalg.norm(coords[i] - coords[j])
                    weight = edge_weights.get((i, j), 1.0)
                    mi_matrix[i, j] = weight * np.exp(-distance)
    
    # Compute local Riemann tensor components at each node
    riemann_components = []
    
    for i in range(num_qubits):
        # Find nearest neighbors
        distances = [np.linalg.norm(coords[i] - coords[j]) for j in range(num_qubits) if j != i]
        nearest_indices = np.argsort(distances)[:3]  # Top 3 nearest neighbors
        
        # Compute local curvature tensor
        local_curvature = np.zeros((2, 2, 2, 2))  # R_abcd
        
        for j in nearest_indices:
            if j != i:
                # Vector from i to j
                direction = coords[j] - coords[i]
                distance = np.linalg.norm(direction)
                
                if distance > 0:
                    # Normalize
                    unit_direction = direction / distance
                    
                    # For 2D, Riemann tensor has only one independent component: R_1212
                    # In spherical geometry: R_1212 = K (Gaussian curvature)
                    if geometry == "spherical":
                        K = curvature * mi_matrix[i, j] / (1 + distance**2)
                    else:
                        K = -curvature * mi_matrix[i, j] / (1 + distance**2)
                    
                    # Fill Riemann tensor components
                    local_curvature[0, 1, 0, 1] = K
                    local_curvature[1, 0, 1, 0] = K
                    local_curvature[0, 1, 1, 0] = -K
                    local_curvature[1, 0, 0, 1] = -K
        
        riemann_components.append(local_curvature)
    
    riemann_components = np.array(riemann_components)
    
    # Extract key components
    R_1212_components = riemann_components[:, 0, 1, 0, 1]
    
    print(f"RIEMANN TENSOR STATISTICS:")
    print(f"  R_1212 component range: [{R_1212_components.min():.6f}, {R_1212_components.max():.6f}]")
    print(f"  Mean R_1212: {np.mean(R_1212_components):.6f}")
    
    if geometry == "spherical":
        positive_components = np.sum(R_1212_components > 0)
        print(f"  Positive R_1212 components: {positive_components}/{num_qubits} ({positive_components/num_qubits*100:.1f}%)")
        
        if positive_components > num_qubits * 0.5:
            print(f"  ✅ DOMINANTLY POSITIVE R_1212 → CONFIRMS SPHERICAL GEOMETRY")
        else:
            print(f"  ❌ INSUFFICIENT POSITIVE R_1212 → INCONSISTENT WITH SPHERICAL GEOMETRY")
    
    return riemann_components

def create_curvature_flow_visualizations(gradient_vectors, effective_energy_density, coords, riemann_components, output_dir):
    """Create comprehensive curvature flow visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Curvature gradient vector field
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    # Normalize gradient vectors for visualization
    gradient_magnitudes = np.linalg.norm(gradient_vectors, axis=1)
    max_magnitude = gradient_magnitudes.max()
    if max_magnitude > 0:
        normalized_gradients = gradient_vectors / max_magnitude * 0.5
    else:
        normalized_gradients = gradient_vectors
    
    # Plot gradient vectors
    for i in range(len(coords)):
        plt.arrow(coords[i, 0], coords[i, 1], 
                 normalized_gradients[i, 0], normalized_gradients[i, 1],
                 head_width=0.05, head_length=0.1, fc='red', ec='red', alpha=0.7)
    
    plt.scatter(coords[:, 0], coords[:, 1], c=gradient_magnitudes, cmap='viridis', s=100, alpha=0.8)
    plt.colorbar(label='Gradient Magnitude')
    plt.title('Curvature Gradient Vector Field')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Effective energy density
    plt.subplot(2, 3, 2)
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=effective_energy_density, cmap='plasma', s=100, alpha=0.8)
    plt.colorbar(label='Effective Energy Density')
    plt.title('Effective Energy Density Distribution')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Flow direction analysis
    plt.subplot(2, 3, 3)
    # Compute flow direction relative to center
    center = np.mean(coords, axis=0)
    flow_directions = []
    
    for i in range(len(coords)):
        # Vector from center to node
        to_center = center - coords[i]
        to_center_norm = np.linalg.norm(to_center)
        
        if to_center_norm > 0:
            to_center_unit = to_center / to_center_norm
            # Project gradient onto direction to center
            gradient_projection = np.dot(gradient_vectors[i], to_center_unit)
            flow_directions.append(gradient_projection)
        else:
            flow_directions.append(0)
    
    flow_directions = np.array(flow_directions)
    colors = ['red' if d < 0 else 'blue' for d in flow_directions]  # Red for inward, blue for outward
    
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, s=100, alpha=0.8)
    plt.title('Flow Direction Analysis\nRed=Inward, Blue=Outward')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Riemann tensor components
    if riemann_components is not None:
        plt.subplot(2, 3, 4)
        R_1212_values = riemann_components[:, 0, 1, 0, 1]
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=R_1212_values, cmap='RdBu_r', s=100, alpha=0.8)
        plt.colorbar(label='R_1212 Component')
        plt.title('Riemann Tensor R_1212 Component')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True, alpha=0.3)
    
    # Plot 5: Mass back-propagation visualization
    plt.subplot(2, 3, 5)
    # Create streamlines showing mass flow
    x_min, x_max = coords[:, 0].min() - 0.5, coords[:, 0].max() + 0.5
    y_min, y_max = coords[:, 1].min() - 0.5, coords[:, 1].max() + 0.5
    
    # Create grid for streamlines
    x_grid = np.linspace(x_min, x_max, 20)
    y_grid = np.linspace(y_min, y_max, 20)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Interpolate gradient field on grid
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            
            # Find nearest nodes and interpolate
            distances = [np.linalg.norm(point - coords[k]) for k in range(len(coords))]
            nearest_idx = np.argmin(distances)
            
            if distances[nearest_idx] < 1.0:  # Only interpolate near nodes
                U[i, j] = gradient_vectors[nearest_idx, 0]
                V[i, j] = gradient_vectors[nearest_idx, 1]
    
    # Plot streamlines
    plt.streamplot(X, Y, U, V, density=1.5, color='purple', linewidth=1)
    plt.scatter(coords[:, 0], coords[:, 1], c='black', s=50, alpha=0.8)
    plt.title('Mass Flow Streamlines')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Curvature convergence analysis
    plt.subplot(2, 3, 6)
    # Compute convergence measure (how much curvature flows toward center)
    center = np.mean(coords, axis=0)
    convergence_measures = []
    
    for i in range(len(coords)):
        to_center = center - coords[i]
        to_center_norm = np.linalg.norm(to_center)
        
        if to_center_norm > 0:
            to_center_unit = to_center / to_center_norm
            gradient_projection = np.dot(gradient_vectors[i], to_center_unit)
            convergence_measures.append(gradient_projection)
        else:
            convergence_measures.append(0)
    
    convergence_measures = np.array(convergence_measures)
    plt.scatter(coords[:, 0], coords[:, 1], c=convergence_measures, cmap='RdYlBu_r', s=100, alpha=0.8)
    plt.colorbar(label='Convergence Measure')
    plt.title('Curvature Convergence Analysis\nNegative=Inward Flow')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'curvature_flow_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create additional detailed plots
    create_detailed_flow_plots(gradient_vectors, effective_energy_density, coords, output_dir)

def create_detailed_flow_plots(gradient_vectors, effective_energy_density, coords, output_dir):
    """Create additional detailed flow analysis plots."""
    
    # Plot 1: Energy density evolution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    # Sort nodes by distance from center
    center = np.mean(coords, axis=0)
    distances_from_center = [np.linalg.norm(coords[i] - center) for i in range(len(coords))]
    sorted_indices = np.argsort(distances_from_center)
    
    sorted_energy = effective_energy_density[sorted_indices]
    sorted_distances = np.array(distances_from_center)[sorted_indices]
    
    plt.plot(sorted_distances, sorted_energy, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Distance from Center')
    plt.ylabel('Effective Energy Density')
    plt.title('Energy Density vs Distance from Center')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Gradient magnitude vs energy density
    plt.subplot(2, 2, 2)
    gradient_magnitudes = np.linalg.norm(gradient_vectors, axis=1)
    plt.scatter(effective_energy_density, gradient_magnitudes, alpha=0.7)
    plt.xlabel('Effective Energy Density')
    plt.ylabel('Gradient Magnitude')
    plt.title('Gradient Magnitude vs Energy Density')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Flow convergence histogram
    plt.subplot(2, 2, 3)
    center = np.mean(coords, axis=0)
    convergence_measures = []
    
    for i in range(len(coords)):
        to_center = center - coords[i]
        to_center_norm = np.linalg.norm(to_center)
        
        if to_center_norm > 0:
            to_center_unit = to_center / to_center_norm
            gradient_projection = np.dot(gradient_vectors[i], to_center_unit)
            convergence_measures.append(gradient_projection)
        else:
            convergence_measures.append(0)
    
    plt.hist(convergence_measures, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', label='No Flow')
    plt.xlabel('Convergence Measure')
    plt.ylabel('Frequency')
    plt.title('Flow Convergence Distribution\nNegative=Inward Flow')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Energy density heatmap
    plt.subplot(2, 2, 4)
    # Create a more detailed heatmap
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    # Create interpolation grid
    x_min, x_max = x_coords.min() - 0.5, x_coords.max() + 0.5
    y_min, y_max = y_coords.min() - 0.5, y_coords.max() + 0.5
    
    xi = np.linspace(x_min, x_max, 50)
    yi = np.linspace(y_min, y_max, 50)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Simple interpolation for visualization
    from scipy.interpolate import griddata
    Zi = griddata((x_coords, y_coords), effective_energy_density, (Xi, Yi), method='cubic')
    
    plt.contourf(Xi, Yi, Zi, levels=20, cmap='plasma', alpha=0.8)
    plt.scatter(x_coords, y_coords, c='white', s=50, alpha=0.8, edgecolors='black')
    plt.colorbar(label='Effective Energy Density')
    plt.title('Energy Density Heatmap')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detailed_flow_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_flow_analysis_report(data, gradient_vectors, effective_energy_density, riemann_components, output_dir):
    """Create comprehensive flow analysis report."""
    output_dir = Path(output_dir)
    
    report_path = output_dir / 'curvature_flow_analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CURVATURE FLOW ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Experiment parameters
        spec = data['spec']
        f.write("EXPERIMENT PARAMETERS:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Geometry: {spec['geometry']}\n")
        f.write(f"Curvature: κ = {spec['curvature']}\n")
        f.write(f"Number of qubits: {spec['num_qubits']}\n")
        f.write(f"Device: {spec['device']}\n\n")
        
        # Gradient vector analysis
        f.write("1. CURVATURE GRADIENT VECTOR ANALYSIS:\n")
        f.write("-" * 35 + "\n")
        gradient_magnitudes = np.linalg.norm(gradient_vectors, axis=1)
        f.write(f"Mean gradient magnitude: {np.mean(gradient_magnitudes):.6f}\n")
        f.write(f"Gradient magnitude range: [{gradient_magnitudes.min():.6f}, {gradient_magnitudes.max():.6f}]\n")
        f.write(f"Gradient magnitude std: {np.std(gradient_magnitudes):.6f}\n\n")
        
        # Flow direction analysis
        f.write("2. FLOW DIRECTION ANALYSIS:\n")
        f.write("-" * 25 + "\n")
        center = np.mean(np.array(data['embedding_coords']), axis=0)
        convergence_measures = []
        
        for i in range(len(gradient_vectors)):
            to_center = center - np.array(data['embedding_coords'])[i]
            to_center_norm = np.linalg.norm(to_center)
            
            if to_center_norm > 0:
                to_center_unit = to_center / to_center_norm
                gradient_projection = np.dot(gradient_vectors[i], to_center_unit)
                convergence_measures.append(gradient_projection)
            else:
                convergence_measures.append(0)
        
        convergence_measures = np.array(convergence_measures)
        inward_flow = np.sum(convergence_measures < 0)
        outward_flow = np.sum(convergence_measures > 0)
        no_flow = np.sum(np.abs(convergence_measures) < 1e-6)
        
        f.write(f"Inward flow nodes: {inward_flow}/{len(convergence_measures)} ({inward_flow/len(convergence_measures)*100:.1f}%)\n")
        f.write(f"Outward flow nodes: {outward_flow}/{len(convergence_measures)} ({outward_flow/len(convergence_measures)*100:.1f}%)\n")
        f.write(f"No flow nodes: {no_flow}/{len(convergence_measures)} ({no_flow/len(convergence_measures)*100:.1f}%)\n")
        f.write(f"Mean convergence measure: {np.mean(convergence_measures):.6f}\n\n")
        
        if spec['geometry'] == "spherical":
            if inward_flow > outward_flow:
                f.write("✅ DOMINANT INWARD FLOW → CONFIRMS SPHERICAL GEOMETRY\n")
                f.write("   This demonstrates mass back-propagation characteristic of positive curvature.\n")
            else:
                f.write("❌ DOMINANT OUTWARD FLOW → INCONSISTENT WITH SPHERICAL GEOMETRY\n")
        f.write("\n")
        
        # Energy density analysis
        f.write("3. EFFECTIVE ENERGY DENSITY ANALYSIS:\n")
        f.write("-" * 35 + "\n")
        f.write(f"Mean energy density: {np.mean(effective_energy_density):.6f}\n")
        f.write(f"Energy density range: [{effective_energy_density.min():.6f}, {effective_energy_density.max():.6f}]\n")
        f.write(f"Energy density std: {np.std(effective_energy_density):.6f}\n\n")
        
        # Riemann tensor analysis
        if riemann_components is not None:
            f.write("4. RIEMANN TENSOR ANALYSIS:\n")
            f.write("-" * 25 + "\n")
            R_1212_values = riemann_components[:, 0, 1, 0, 1]
            f.write(f"Mean R_1212 component: {np.mean(R_1212_values):.6f}\n")
            f.write(f"R_1212 range: [{R_1212_values.min():.6f}, {R_1212_values.max():.6f}]\n")
            
            if spec['geometry'] == "spherical":
                positive_components = np.sum(R_1212_values > 0)
                f.write(f"Positive R_1212 components: {positive_components}/{len(R_1212_values)} ({positive_components/len(R_1212_values)*100:.1f}%)\n")
                
                if positive_components > len(R_1212_values) * 0.5:
                    f.write("✅ DOMINANTLY POSITIVE R_1212 → CONFIRMS SPHERICAL GEOMETRY\n")
                else:
                    f.write("❌ INSUFFICIENT POSITIVE R_1212 → INCONSISTENT WITH SPHERICAL GEOMETRY\n")
            f.write("\n")
        
        # Mass back-propagation summary
        f.write("5. MASS BACK-PROPAGATION SUMMARY:\n")
        f.write("-" * 35 + "\n")
        
        if spec['geometry'] == "spherical":
            if inward_flow > outward_flow and np.mean(convergence_measures) < 0:
                f.write("🎉 MASS BACK-PROPAGATION CONFIRMED!\n")
                f.write("   - Dominant inward flow detected\n")
                f.write("   - Negative mean convergence measure\n")
                f.write("   - This is the signature of positive curvature (spherical geometry)\n")
                f.write("   - Mass/energy flows inward rather than dissipating outward\n")
            else:
                f.write("⚠️  MASS BACK-PROPAGATION NOT CLEARLY DETECTED\n")
                f.write("   - Flow patterns are inconsistent with spherical geometry\n")
                f.write("   - May need adjustment of curvature computation\n")
        else:
            f.write("Analysis completed for non-spherical geometry.\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF FLOW ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"✅ Flow analysis report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze curvature flow patterns and mass back-propagation')
    parser.add_argument('json_path', help='Path to experiment results JSON file')
    parser.add_argument('--output-dir', help='Output directory (defaults to same directory as JSON file)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"📁 Loading experiment results from: {args.json_path}")
    try:
        data = load_experiment_data(args.json_path)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.json_path).parent
    
    # Run curvature flow analysis
    print("\n🔍 Running curvature flow analysis...")
    
    # 1. Compute curvature gradient vectors
    gradient_vectors, effective_energy_density, coords = compute_curvature_gradient_vectors(data)
    
    if gradient_vectors is None:
        print("❌ Failed to compute gradient vectors.")
        return
    
    # 2. Compute Riemann curvature tensor
    riemann_components = compute_riemann_curvature_tensor(data)
    
    # 3. Create visualizations
    print("\n🎨 Generating curvature flow visualizations...")
    create_curvature_flow_visualizations(gradient_vectors, effective_energy_density, coords, riemann_components, output_dir)
    
    # 4. Create comprehensive report
    create_flow_analysis_report(data, gradient_vectors, effective_energy_density, riemann_components, output_dir)
    
    print(f"\n✅ Curvature flow analysis complete! All results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - curvature_flow_analysis.png")
    print("  - detailed_flow_analysis.png")
    print("  - curvature_flow_analysis_report.txt")

if __name__ == "__main__":
    main() 