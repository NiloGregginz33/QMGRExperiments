import numpy as np
import itertools
from scipy.spatial.distance import squareform
from scipy.sparse.csgraph import floyd_warshall
from scipy.spatial import Delaunay

"""
geometry_analysis_tools.py

This module provides:
- Quantum information-derived distance functions
- Geodesic path extraction
- Curvature estimation from triangle angle sums

Supports quantum circuit-based exploration of emergent geometry from entanglement.
Author: Manav Naik
"""

# -------------------------------
# Quantum Distance Functions
# -------------------------------

def mi_inverse(mi_matrix, epsilon=1e-10):
    return 1.0 / (mi_matrix + epsilon)

def mi_neglog(mi_matrix, epsilon=1e-10):
    return -np.log(mi_matrix + epsilon)

def entropy_gradient(entropy_vector):
    entropy_vector = np.array(entropy_vector)
    n = len(entropy_vector)
    d_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d_matrix[i, j] = abs(entropy_vector[i] - entropy_vector[j])
    return d_matrix

def combined_metric(mi_matrix, entropy_vector, alpha=1.0, beta=1.0, epsilon=1e-10):
    mi_term = (1.0 / (mi_matrix + epsilon)) ** alpha
    entropy_term = entropy_gradient(entropy_vector)
    return mi_term + beta * entropy_term

def normalize_matrix(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    return (matrix - min_val) / (max_val - min_val + 1e-10)

# -------------------------------
# Geodesic Path Extraction
# -------------------------------

def compute_geodesics(dist_matrix):
    """
    Uses Floyd-Warshall algorithm to compute shortest paths between all node pairs.
    Returns:
        geodesic_matrix: 2D numpy array of minimal path distances
    """
    return floyd_warshall(dist_matrix, directed=False)

# -------------------------------
# Curvature Estimation
# -------------------------------

def triangle_angle_sum(a, b, c):
    """
    Compute angle sum from side lengths (Law of Cosines).
    """
    def angle(opposite, side1, side2):
        cos_theta = (side1**2 + side2**2 - opposite**2) / (2 * side1 * side2)
        return np.arccos(np.clip(cos_theta, -1.0, 1.0))

    A = angle(a, b, c)
    B = angle(b, a, c)
    C = angle(c, a, b)
    return A + B + C

def estimate_curvature_from_angles(points_2d, dist_matrix):
    """
    Estimate Gaussian curvature by checking triangle angle sums
    using Delaunay triangulation on 2D MDS embedding.
    """
    triangles = Delaunay(points_2d).simplices
    curvatures = []
    for tri in triangles:
        i, j, k = tri
        a = dist_matrix[i, j]
        b = dist_matrix[j, k]
        c = dist_matrix[k, i]
        angle_sum = triangle_angle_sum(a, b, c)
        curvature = angle_sum - np.pi
        curvatures.append(curvature)
    return np.array(curvatures)

# -------------------------------
# Mutual Information from Density Matrices
# -------------------------------

def von_neumann_entropy(rho):
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-12]
    return -np.sum(evals * np.log2(evals))

def compute_mutual_information(rho_ij, rho_i, rho_j):
    return (von_neumann_entropy(rho_i)
            + von_neumann_entropy(rho_j)
            - von_neumann_entropy(rho_ij))

def triangle_inequality_check(dist_matrix, tol=1e-8):
    """
    Checks the triangle inequality for all triplets in the distance matrix.
    Returns a list of violations and summary statistics.
    """
    n = dist_matrix.shape[0]
    violations = []
    for i, j, k in itertools.combinations(range(n), 3):
        dij, djk, dki = dist_matrix[i, j], dist_matrix[j, k], dist_matrix[k, i]
        # Check all three triangle inequalities
        if dij + djk < dki - tol:
            violations.append(((i, j, k), dij, djk, dki, 'dij+djk < dki'))
        if djk + dki < dij - tol:
            violations.append(((j, k, i), djk, dki, dij, 'djk+dki < dij'))
        if dki + dij < djk - tol:
            violations.append(((k, i, j), dki, dij, djk, 'dki+dij < djk'))
    return violations, len(violations)

def gaussian_curvature_angle_deficit(points_2d, dist_matrix):
    """
    Computes Gaussian curvature at each vertex using the angle deficit method
    on the Delaunay triangulation of the 2D embedding.
    Returns a dict: vertex -> angle deficit (curvature)
    """
    triangles = Delaunay(points_2d).simplices
    n = points_2d.shape[0]
    angle_sums = np.zeros(n)
    for tri in triangles:
        i, j, k = tri
        a = dist_matrix[j, k]
        b = dist_matrix[i, k]
        c = dist_matrix[i, j]
        def angle(opposite, side1, side2):
            cos_theta = (side1**2 + side2**2 - opposite**2) / (2 * side1 * side2)
            return np.arccos(np.clip(cos_theta, -1.0, 1.0))
        angles = [angle(a, b, c), angle(b, a, c), angle(c, a, b)]
        angle_sums[i] += angles[0]
        angle_sums[j] += angles[1]
        angle_sums[k] += angles[2]
    # Each vertex may appear in multiple triangles; angle deficit = 2pi - sum(angles at vertex)
    curvature = {v: 2 * np.pi - angle_sums[v] for v in range(n)}
    return curvature 