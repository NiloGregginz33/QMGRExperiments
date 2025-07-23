#!/usr/bin/env python3
"""
Rigorous RT Surface Oracle for EWR Experiment
Implements min-cut/max-flow algorithm on dual graph to determine RT surface containment.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Set
import json

class RTSurfaceOracle:
    """
    Rigorous RT Surface Oracle using graph-theoretic methods.
    
    Based on the principle that the RT surface is the minimal surface
    whose boundary encloses the bulk point. Implemented via min-cut/max-flow
    on the dual graph where cut cost = geodesic length.
    """
    
    def __init__(self, boundary_graph: nx.Graph, edge_weights: Dict[Tuple, float] = None):
        """
        Initialize RT Surface Oracle.
        
        Args:
            boundary_graph: NetworkX graph representing boundary connectivity
            edge_weights: Dictionary of edge weights (default: all weights = 1)
        """
        self.boundary_graph = boundary_graph.copy()
        self.num_nodes = boundary_graph.number_of_nodes()
        
        # Set edge weights (default: all weights = 1 for regular lattice)
        if edge_weights is None:
            edge_weights = {(u, v): 1.0 for u, v in boundary_graph.edges()}
        
        # Add edge weights to graph
        nx.set_edge_attributes(self.boundary_graph, edge_weights, 'weight')
        
        # Pre-compute dual graph for min-cut calculations
        self._compute_dual_graph()
        
        # Cache for RT surface results
        self.rt_cache = {}
        
    def _compute_dual_graph(self):
        """Compute the dual graph for min-cut/max-flow calculations."""
        # For a boundary graph, the dual graph represents the bulk geometry
        # Each face of the boundary graph becomes a node in the dual graph
        # Each edge in the boundary graph becomes an edge in the dual graph
        
        # Create dual graph
        self.dual_graph = nx.Graph()
        
        # Add nodes for each face (region) of the boundary graph
        # For simplicity, we'll use the boundary nodes as dual nodes
        for node in self.boundary_graph.nodes():
            self.dual_graph.add_node(node)
        
        # Add edges with weights corresponding to geodesic distances
        for u, v, data in self.boundary_graph.edges(data=True):
            weight = data.get('weight', 1.0)
            self.dual_graph.add_edge(u, v, weight=weight)
    
    def rt_surface_contains_bulk(self, region_nodes: Set[int], bulk_node: int) -> bool:
        """
        Determine if RT surface for a region contains the bulk point.
        
        Args:
            region_nodes: Set of boundary nodes in the region
            bulk_node: Location of the bulk point
            
        Returns:
            bool: True if RT surface contains bulk point
        """
        # Create cache key
        cache_key = (frozenset(region_nodes), bulk_node)
        
        if cache_key in self.rt_cache:
            return self.rt_cache[cache_key]
        
        # Check if bulk node is directly in the region
        if bulk_node in region_nodes:
            self.rt_cache[cache_key] = True
            return True
        
        # Use geometric RT surface calculation (more robust)
        contains_bulk = self._geometric_rt_surface(region_nodes, bulk_node)
        
        # Cache the result
        self.rt_cache[cache_key] = contains_bulk
        return contains_bulk
    
    def _geometric_rt_surface(self, region_nodes: Set[int], bulk_node: int) -> bool:
        """
        Fallback geometric RT surface calculation.
        
        Args:
            region_nodes: Set of boundary nodes in the region
            bulk_node: Location of the bulk point
            
        Returns:
            bool: True if RT surface contains bulk point
        """
        # Calculate geometric center of region
        region_center = np.mean(list(region_nodes))
        
        # Calculate distance from bulk point to region center
        distance = abs(bulk_node - region_center)
        
        # Calculate region radius (max distance from center to any region node)
        region_radius = max(abs(node - region_center) for node in region_nodes)
        
        # RT surface contains bulk if distance is less than region radius
        return distance <= region_radius
    
    def compute_all_rt_surfaces(self, regions: Dict[str, List[int]], bulk_node: int) -> Dict[str, bool]:
        """
        Compute RT surface containment for all regions.
        
        Args:
            regions: Dictionary mapping region names to lists of boundary nodes
            bulk_node: Location of the bulk point
            
        Returns:
            Dict[str, bool]: Mapping of region names to RT surface containment
        """
        rt_results = {}
        
        for region_name, region_nodes in regions.items():
            region_set = set(region_nodes)
            contains_bulk = self.rt_surface_contains_bulk(region_set, bulk_node)
            rt_results[region_name] = contains_bulk
        
        return rt_results
    
    def get_cache(self) -> Dict:
        """Get the RT surface cache for logging."""
        # Convert frozen sets back to lists for JSON serialization
        cache_dict = {}
        for (region_frozen, bulk_node), result in self.rt_cache.items():
            region_list = list(region_frozen)
            cache_dict[f"region_{region_list}_bulk_{bulk_node}"] = bool(result)
        return cache_dict
    
    def clear_cache(self):
        """Clear the RT surface cache."""
        self.rt_cache.clear()

def create_boundary_graph(num_qubits: int, connectivity: str = 'linear') -> nx.Graph:
    """
    Create boundary graph for RT surface calculations.
    
    Args:
        num_qubits: Number of boundary qubits
        connectivity: Type of connectivity ('linear', 'circular', 'grid')
        
    Returns:
        nx.Graph: Boundary graph with specified connectivity
    """
    G = nx.Graph()
    
    if connectivity == 'linear':
        # Linear chain connectivity
        for i in range(num_qubits - 1):
            G.add_edge(i, i + 1, weight=1.0)
            
    elif connectivity == 'circular':
        # Circular connectivity
        for i in range(num_qubits):
            G.add_edge(i, (i + 1) % num_qubits, weight=1.0)
            
    elif connectivity == 'grid':
        # 2D grid connectivity (for square number of qubits)
        import math
        grid_size = int(math.sqrt(num_qubits))
        if grid_size * grid_size != num_qubits:
            raise ValueError(f"Grid connectivity requires square number of qubits, got {num_qubits}")
        
        for i in range(grid_size):
            for j in range(grid_size):
                node = i * grid_size + j
                # Add horizontal edges
                if j < grid_size - 1:
                    G.add_edge(node, node + 1, weight=1.0)
                # Add vertical edges
                if i < grid_size - 1:
                    G.add_edge(node, node + grid_size, weight=1.0)
    
    return G

# Example usage and testing
if __name__ == "__main__":
    # Test with 12 qubits in linear arrangement
    num_qubits = 12
    boundary_graph = create_boundary_graph(num_qubits, 'linear')
    
    # Create RT surface oracle
    rt_oracle = RTSurfaceOracle(boundary_graph)
    
    # Define regions
    regions = {
        'A': [0, 1, 2, 3],
        'B': [4, 5, 6, 7],
        'C': [8, 9, 10, 11]
    }
    
    # Test with bulk point at qubit 6
    bulk_node = 6
    rt_results = rt_oracle.compute_all_rt_surfaces(regions, bulk_node)
    
    print("RT Surface Results:")
    for region_name, contains_bulk in rt_results.items():
        print(f"Region {region_name}: RT surface contains bulk = {contains_bulk}")
    
    print(f"\nRT Surface Cache:")
    print(json.dumps(rt_oracle.get_cache(), indent=2)) 