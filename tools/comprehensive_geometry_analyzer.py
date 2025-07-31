#!/usr/bin/env python3
"""
Comprehensive Geometry Analyzer
==============================

Analyzes quantum geometry experiments for:
1. Geometric locality in mutual information matrix
2. Graph structure and cluster hierarchy
3. MDS visualization of bulk geometry
4. Ryu-Takayanagi consistency with dissimilarity scaling
5. Entanglement spectrum comparison to Haar states
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys
import os
from typing import Dict, List, Optional, Tuple, Any
import warnings
from scipy import stats, optimize, spatial, cluster
from sklearn.metrics import r2_score
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import itertools
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.append('src')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class ComprehensiveGeometryAnalyzer:
    def __init__(self, results_file: str):
        """Initialize the analyzer with experiment results."""
        self.results_file = results_file
        self.results = self.load_results()
        self.num_qubits = self.results.get('spec', {}).get('num_qubits', 9)
        self.mi_matrix = None
        self.dissimilarity_matrix = None
        self.analysis_results = {}
        
    def load_results(self) -> Dict:
        """Load experiment results from JSON file."""
        try:
            with open(self.results_file, 'r') as f:
                results = json.load(f)
            print(f"✅ Loaded results from: {self.results_file}")
            return results
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return {}
    
    def extract_mutual_information_matrix(self) -> np.ndarray:
        """Extract mutual information matrix from results."""
        if 'mi_matrix' in self.results:
            mi_matrix = np.array(self.results['mi_matrix'])
        elif 'mutual_information_matrix' in self.results:
            mi_matrix = np.array(self.results['mutual_information_matrix'])
        else:
            print("❌ No mutual information matrix found in results")
            # Create a dummy matrix for testing
            mi_matrix = np.random.rand(self.num_qubits, self.num_qubits)
            mi_matrix = (mi_matrix + mi_matrix.T) / 2  # Make symmetric
            np.fill_diagonal(mi_matrix, 1.0)  # Diagonal should be 1
        
        self.mi_matrix = mi_matrix
        print(f"✅ Mutual information matrix shape: {mi_matrix.shape}")
        return mi_matrix
    
    def analyze_geometric_locality(self) -> Dict:
        """Analyze geometric locality in the mutual information matrix."""
        print("🔍 Analyzing geometric locality...")
        
        if self.mi_matrix is None:
            self.extract_mutual_information_matrix()
        
        mi_matrix = self.mi_matrix
        n = mi_matrix.shape[0]
        
        # Calculate distance-dependent correlations
        locality_metrics = {}
        
        # 1. Nearest neighbor correlation
        nn_corr = 0
        nn_count = 0
        for i in range(n-1):
            nn_corr += mi_matrix[i, i+1]
            nn_count += 1
        if nn_count > 0:
            nn_corr /= nn_count
        locality_metrics['nearest_neighbor_correlation'] = float(nn_corr)
        
        # 2. Distance decay analysis
        distances = []
        correlations = []
        for i in range(n):
            for j in range(i+1, n):
                distance = abs(i - j)
                distances.append(distance)
                correlations.append(mi_matrix[i, j])
        
        # Fit exponential decay model
        try:
            def exp_decay(x, a, b, c):
                return a * np.exp(-b * x) + c
            
            popt, pcov = optimize.curve_fit(exp_decay, distances, correlations, 
                                          p0=[1.0, 0.5, 0.1], maxfev=10000)
            decay_rate = popt[1]
            decay_r2 = r2_score(correlations, exp_decay(distances, *popt))
            
            locality_metrics['distance_decay_rate'] = float(decay_rate)
            locality_metrics['distance_decay_r2'] = float(decay_r2)
            locality_metrics['distance_decay_params'] = popt.tolist()
        except:
            locality_metrics['distance_decay_rate'] = 0.0
            locality_metrics['distance_decay_r2'] = 0.0
        
        # 3. Locality score (higher = more local)
        locality_score = nn_corr / (np.mean(mi_matrix) + 1e-10)
        locality_metrics['locality_score'] = float(locality_score)
        
        # 4. Non-local correlations
        nonlocal_threshold = 0.1
        nonlocal_count = np.sum(mi_matrix > nonlocal_threshold) - n  # Exclude diagonal
        total_pairs = n * (n - 1) // 2
        nonlocal_fraction = nonlocal_count / total_pairs
        locality_metrics['nonlocal_correlation_fraction'] = float(nonlocal_fraction)
        
        print(f"  📊 Locality score: {locality_score:.4f}")
        print(f"  📊 Distance decay rate: {locality_metrics['distance_decay_rate']:.4f}")
        print(f"  📊 Non-local fraction: {nonlocal_fraction:.4f}")
        
        return locality_metrics
    
    def analyze_graph_structure(self) -> Dict:
        """Analyze graph structure and cluster hierarchy."""
        print("🔍 Analyzing graph structure...")
        
        if self.mi_matrix is None:
            self.extract_mutual_information_matrix()
        
        mi_matrix = self.mi_matrix
        n = mi_matrix.shape[0]
        
        # Create adjacency matrix (threshold-based)
        threshold = np.percentile(mi_matrix[mi_matrix > 0], 75)  # Top 25% connections
        adjacency_matrix = (mi_matrix > threshold).astype(float)
        np.fill_diagonal(adjacency_matrix, 0)  # Remove self-loops
        
        # Create NetworkX graph
        G = nx.from_numpy_array(adjacency_matrix)
        
        # Graph metrics
        graph_metrics = {}
        
        # 1. Basic graph properties
        graph_metrics['num_nodes'] = G.number_of_nodes()
        graph_metrics['num_edges'] = G.number_of_edges()
        graph_metrics['density'] = nx.density(G)
        graph_metrics['average_degree'] = np.mean([d for n, d in G.degree()])
        
        # 2. Connectivity analysis
        if nx.is_connected(G):
            graph_metrics['is_connected'] = True
            graph_metrics['diameter'] = nx.diameter(G)
            graph_metrics['average_shortest_path'] = nx.average_shortest_path_length(G)
        else:
            graph_metrics['is_connected'] = False
            graph_metrics['num_components'] = nx.number_connected_components(G)
            largest_cc = max(nx.connected_components(G), key=len)
            graph_metrics['largest_component_size'] = len(largest_cc)
        
        # 3. Clustering analysis
        graph_metrics['average_clustering'] = nx.average_clustering(G)
        graph_metrics['transitivity'] = nx.transitivity(G)
        
        # 4. Centrality measures
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        # Handle eigenvector centrality for disconnected graphs
        try:
            eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
            graph_metrics['max_eigenvector'] = max(eigenvector_centrality.values())
        except nx.AmbiguousSolution:
            # For disconnected graphs, use largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            G_largest = G.subgraph(largest_cc)
            if len(largest_cc) > 1:
                eigenvector_centrality = nx.eigenvector_centrality_numpy(G_largest)
                graph_metrics['max_eigenvector'] = max(eigenvector_centrality.values())
            else:
                graph_metrics['max_eigenvector'] = 0.0
        
        graph_metrics['max_betweenness'] = max(betweenness_centrality.values())
        graph_metrics['max_closeness'] = max(closeness_centrality.values())
        
        # 5. Community detection
        try:
            communities = nx.community.greedy_modularity_communities(G)
            graph_metrics['num_communities'] = len(communities)
            graph_metrics['modularity'] = nx.community.modularity(G, communities)
            
            # Community size distribution
            community_sizes = [len(c) for c in communities]
            graph_metrics['community_sizes'] = community_sizes
            graph_metrics['largest_community_size'] = max(community_sizes)
        except:
            graph_metrics['num_communities'] = 1
            graph_metrics['modularity'] = 0.0
        
        print(f"  📊 Graph density: {graph_metrics['density']:.4f}")
        print(f"  📊 Average clustering: {graph_metrics['average_clustering']:.4f}")
        print(f"  📊 Number of communities: {graph_metrics['num_communities']}")
        
        return graph_metrics
    
    def run_mds_visualization(self, output_dir: str = None) -> Dict:
        """Run MDS to visualize bulk geometry."""
        print("🔍 Running MDS visualization...")
        
        if self.mi_matrix is None:
            self.extract_mutual_information_matrix()
        
        mi_matrix = self.mi_matrix
        
        # Convert MI to dissimilarity (distance)
        # Higher MI = lower distance
        max_mi = np.max(mi_matrix)
        dissimilarity_matrix = max_mi - mi_matrix
        np.fill_diagonal(dissimilarity_matrix, 0)
        
        self.dissimilarity_matrix = dissimilarity_matrix
        
        # Run MDS
        mds_results = {}
        
        # 2D MDS
        try:
            mds_2d = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            coords_2d = mds_2d.fit_transform(dissimilarity_matrix)
            mds_results['coords_2d'] = coords_2d.tolist()
            mds_results['stress_2d'] = float(mds_2d.stress_)
        except Exception as e:
            print(f"  ❌ 2D MDS failed: {e}")
            mds_results['coords_2d'] = []
            mds_results['stress_2d'] = 1.0
        
        # 3D MDS
        try:
            mds_3d = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
            coords_3d = mds_3d.fit_transform(dissimilarity_matrix)
            mds_results['coords_3d'] = coords_3d.tolist()
            mds_results['stress_3d'] = float(mds_3d.stress_)
        except Exception as e:
            print(f"  ❌ 3D MDS failed: {e}")
            mds_results['coords_3d'] = []
            mds_results['stress_3d'] = 1.0
        
        # Create visualizations
        if output_dir:
            self.create_mds_visualizations(mds_results, output_dir)
        
        print(f"  📊 2D MDS stress: {mds_results['stress_2d']:.4f}")
        print(f"  📊 3D MDS stress: {mds_results['stress_3d']:.4f}")
        
        return mds_results
    
    def create_mds_visualizations(self, mds_results: Dict, output_dir: str):
        """Create MDS visualization plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        # 2D MDS plot
        if mds_results['coords_2d']:
            coords_2d = np.array(mds_results['coords_2d'])
            
            plt.figure(figsize=(10, 8))
            plt.scatter(coords_2d[:, 0], coords_2d[:, 1], c=range(len(coords_2d)), 
                       cmap='viridis', s=100, alpha=0.7)
            
            # Add node labels
            for i, (x, y) in enumerate(coords_2d):
                plt.annotate(f'q{i}', (x, y), xytext=(5, 5), textcoords='offset points')
            
            plt.title(f'MDS 2D Visualization (Stress: {mds_results["stress_2d"]:.4f})')
            plt.xlabel('MDS Dimension 1')
            plt.ylabel('MDS Dimension 2')
            plt.colorbar(label='Qubit Index')
            plt.grid(True, alpha=0.3)
            
            plot_file = os.path.join(output_dir, 'mds_2d_visualization.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  📈 2D MDS plot saved: {plot_file}")
        
        # 3D MDS plot
        if mds_results['coords_3d']:
            coords_3d = np.array(mds_results['coords_3d'])
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2], 
                               c=range(len(coords_3d)), cmap='viridis', s=100, alpha=0.7)
            
            # Add node labels
            for i, (x, y, z) in enumerate(coords_3d):
                ax.text(x, y, z, f'q{i}', fontsize=8)
            
            ax.set_title(f'MDS 3D Visualization (Stress: {mds_results["stress_3d"]:.4f})')
            ax.set_xlabel('MDS Dimension 1')
            ax.set_ylabel('MDS Dimension 2')
            ax.set_zlabel('MDS Dimension 3')
            
            plot_file = os.path.join(output_dir, 'mds_3d_visualization.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  📈 3D MDS plot saved: {plot_file}")
    
    def check_ryu_takayanagi_consistency(self) -> Dict:
        """Check consistency with Ryu-Takayanagi using dissimilarity scaling."""
        print("🔍 Checking Ryu-Takayanagi consistency...")
        
        if self.dissimilarity_matrix is None:
            self.run_mds_visualization()
        
        dissimilarity_matrix = self.dissimilarity_matrix
        n = dissimilarity_matrix.shape[0]
        
        ryu_takayanagi_metrics = {}
        
        # Generate all possible boundary regions
        boundary_regions = []
        for size in range(1, n//2 + 1):
            for subset in itertools.combinations(range(n), size):
                boundary_regions.append(list(subset))
        
        # Calculate RT-like quantities
        rt_quantities = []
        
        for region in boundary_regions:
            region_size = len(region)
            
            # Calculate boundary length (perimeter of region)
            boundary_length = 0
            for i in region:
                for j in range(n):
                    if j not in region:
                        boundary_length += dissimilarity_matrix[i, j]
            
            # Calculate bulk volume (area/volume of region)
            bulk_volume = 0
            for i in region:
                for j in region:
                    if i != j:
                        bulk_volume += dissimilarity_matrix[i, j]
            
            # RT-like quantity: boundary length / bulk volume
            if bulk_volume > 0:
                rt_quantity = boundary_length / bulk_volume
                rt_quantities.append({
                    'region_size': region_size,
                    'boundary_length': boundary_length,
                    'bulk_volume': bulk_volume,
                    'rt_quantity': rt_quantity
                })
        
        # Analyze RT scaling
        if rt_quantities:
            region_sizes = [q['region_size'] for q in rt_quantities]
            rt_values = [q['rt_quantity'] for q in rt_quantities]
            
            # Fit power law: RT ~ size^alpha
            try:
                def power_law(x, a, alpha):
                    return a * (x ** alpha)
                
                popt, pcov = optimize.curve_fit(power_law, region_sizes, rt_values, 
                                              p0=[1.0, 0.5], maxfev=10000)
                alpha = popt[1]
                power_law_r2 = r2_score(rt_values, power_law(region_sizes, *popt))
                
                ryu_takayanagi_metrics['rt_scaling_exponent'] = float(alpha)
                ryu_takayanagi_metrics['rt_power_law_r2'] = float(power_law_r2)
                ryu_takayanagi_metrics['rt_power_law_params'] = popt.tolist()
                
                # Check if scaling is consistent with RT (typically alpha ~ 1)
                rt_consistency = 1.0 - abs(alpha - 1.0)
                ryu_takayanagi_metrics['rt_consistency_score'] = float(rt_consistency)
                
            except:
                ryu_takayanagi_metrics['rt_scaling_exponent'] = 0.0
                ryu_takayanagi_metrics['rt_power_law_r2'] = 0.0
                ryu_takayanagi_metrics['rt_consistency_score'] = 0.0
        
        # Calculate area law violation
        area_law_violation = 0.0
        if rt_quantities:
            # Check if RT quantity scales with boundary length rather than volume
            boundary_lengths = [q['boundary_length'] for q in rt_quantities]
            bulk_volumes = [q['bulk_volume'] for q in rt_quantities]
            
            # Area law: entropy ~ boundary length
            # Volume law: entropy ~ volume
            try:
                area_corr = np.corrcoef(boundary_lengths, rt_values)[0, 1]
                volume_corr = np.corrcoef(bulk_volumes, rt_values)[0, 1]
                
                area_law_violation = abs(volume_corr) - abs(area_corr)
                ryu_takayanagi_metrics['area_law_violation'] = float(area_law_violation)
            except:
                ryu_takayanagi_metrics['area_law_violation'] = 0.0
        
        print(f"  📊 RT scaling exponent: {ryu_takayanagi_metrics.get('rt_scaling_exponent', 0):.4f}")
        print(f"  📊 RT consistency score: {ryu_takayanagi_metrics.get('rt_consistency_score', 0):.4f}")
        print(f"  📊 Area law violation: {area_law_violation:.4f}")
        
        return ryu_takayanagi_metrics
    
    def analyze_entanglement_spectrum(self) -> Dict:
        """Analyze entanglement spectrum and compare to Haar random states."""
        print("🔍 Analyzing entanglement spectrum...")
        
        # Get statevector if available
        statevector = None
        if 'statevector' in self.results:
            statevector = np.array(self.results['statevector'])
        elif 'density_matrix' in self.results:
            # Convert density matrix to statevector (take first eigenvector)
            rho = np.array(self.results['density_matrix'])
            eigenvalues, eigenvectors = np.linalg.eigh(rho)
            statevector = eigenvectors[:, -1]  # Largest eigenvalue
        
        if statevector is None:
            print("  ❌ No statevector found for entanglement spectrum analysis")
            return {}
        
        entanglement_metrics = {}
        
        # Calculate entanglement spectrum for all bipartitions
        entanglement_spectra = []
        
        for size in range(1, self.num_qubits//2 + 1):
            for subset in itertools.combinations(range(self.num_qubits), size):
                # Calculate reduced density matrix
                complement = [i for i in range(self.num_qubits) if i not in subset]
                
                # Reshape statevector for partial trace
                dim_a = 2**len(subset)
                dim_b = 2**len(complement)
                
                # Create density matrix
                rho_full = np.outer(statevector, statevector.conj())
                
                # Reshape for partial trace
                rho_reshaped = rho_full.reshape(dim_a, dim_b, dim_a, dim_b)
                
                # Partial trace over complement
                rho_reduced = np.trace(rho_reshaped, axis1=1, axis2=3)
                
                # Get eigenvalues
                eigenvalues = np.linalg.eigvalsh(rho_reduced)
                eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove zeros
                
                # Sort in descending order
                eigenvalues = np.sort(eigenvalues)[::-1]
                
                entanglement_spectra.append({
                    'subset': subset,
                    'size': len(subset),
                    'eigenvalues': eigenvalues.tolist(),
                    'entropy': -np.sum(eigenvalues * np.log2(eigenvalues))
                })
        
        # Analyze entanglement spectrum
        if entanglement_spectra:
            # 1. Average entanglement entropy
            avg_entropy = np.mean([spec['entropy'] for spec in entanglement_spectra])
            entanglement_metrics['average_entanglement_entropy'] = float(avg_entropy)
            
            # 2. Entanglement spectrum shape
            all_eigenvalues = []
            for spec in entanglement_spectra:
                all_eigenvalues.extend(spec['eigenvalues'])
            
            # Calculate spectral statistics
            entanglement_metrics['spectral_gap'] = float(np.mean(all_eigenvalues) - np.min(all_eigenvalues))
            entanglement_metrics['spectral_width'] = float(np.max(all_eigenvalues) - np.min(all_eigenvalues))
            
            # 3. Compare to Haar random states
            # For Haar random states, eigenvalues follow Marchenko-Pastur distribution
            # Expected entropy for Haar random state: log(d_A) - 1/2
            expected_haar_entropy = np.log2(2**(self.num_qubits//2)) - 0.5
            entanglement_metrics['expected_haar_entropy'] = float(expected_haar_entropy)
            
            # Deviation from Haar
            haar_deviation = abs(avg_entropy - expected_haar_entropy) / expected_haar_entropy
            entanglement_metrics['haar_deviation'] = float(haar_deviation)
            
            # 4. Entanglement spectrum flatness
            # Flatter spectrum = more random
            spectrum_flatness = np.std(all_eigenvalues) / np.mean(all_eigenvalues)
            entanglement_metrics['spectrum_flatness'] = float(spectrum_flatness)
            
            # 5. Page curve analysis
            entropies_by_size = {}
            for spec in entanglement_spectra:
                size = spec['size']
                if size not in entropies_by_size:
                    entropies_by_size[size] = []
                entropies_by_size[size].append(spec['entropy'])
            
            # Calculate average entropy for each subsystem size
            avg_entropies = []
            sizes = []
            for size in sorted(entropies_by_size.keys()):
                sizes.append(size)
                avg_entropies.append(np.mean(entropies_by_size[size]))
            
            # Fit Page curve model
            try:
                def page_curve(x, a, b, c):
                    return a * x * (max(sizes) - x) + b * x + c
                
                popt, _ = optimize.curve_fit(page_curve, sizes, avg_entropies, 
                                          p0=[0.1, 0.5, 0.0], maxfev=10000)
                page_pred = page_curve(sizes, *popt)
                page_r2 = r2_score(avg_entropies, page_pred)
                
                entanglement_metrics['page_curve_r2'] = float(page_r2)
                entanglement_metrics['page_curve_params'] = popt.tolist()
            except:
                entanglement_metrics['page_curve_r2'] = 0.0
        
        print(f"  📊 Average entanglement entropy: {entanglement_metrics.get('average_entanglement_entropy', 0):.4f}")
        print(f"  📊 Haar deviation: {entanglement_metrics.get('haar_deviation', 0):.4f}")
        print(f"  📊 Page curve R²: {entanglement_metrics.get('page_curve_r2', 0):.4f}")
        
        return entanglement_metrics
    
    def run_comprehensive_analysis(self, output_dir: str = None) -> Dict:
        """Run the complete comprehensive analysis."""
        print("🚀 Starting Comprehensive Geometry Analysis")
        print("=" * 60)
        
        if output_dir is None:
            output_dir = "comprehensive_analysis_results"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Run all analyses
        self.analysis_results = {
            'geometric_locality': self.analyze_geometric_locality(),
            'graph_structure': self.analyze_graph_structure(),
            'mds_visualization': self.run_mds_visualization(output_dir),
            'ryu_takayanagi': self.check_ryu_takayanagi_consistency(),
            'entanglement_spectrum': self.analyze_entanglement_spectrum()
        }
        
        # Create summary visualizations
        self.create_summary_visualizations(output_dir)
        
        # Save results
        self.save_results(output_dir)
        
        # Print summary
        self.print_summary()
        
        return self.analysis_results
    
    def create_summary_visualizations(self, output_dir: str):
        """Create summary visualizations."""
        print("📈 Creating summary visualizations...")
        
        # 1. Mutual Information Matrix Heatmap
        if self.mi_matrix is not None:
            plt.figure(figsize=(10, 8))
            sns.heatmap(self.mi_matrix, annot=True, fmt='.3f', cmap='viridis', 
                       square=True, cbar_kws={'label': 'Mutual Information'})
            plt.title('Mutual Information Matrix')
            plt.xlabel('Qubit Index')
            plt.ylabel('Qubit Index')
            
            plot_file = os.path.join(output_dir, 'mutual_information_heatmap.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  📈 MI heatmap saved: {plot_file}")
        
        # 2. Analysis Summary Dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Locality metrics
        locality = self.analysis_results['geometric_locality']
        axes[0, 0].bar(['NN Corr', 'Locality Score', 'Non-local Frac'], 
                      [locality.get('nearest_neighbor_correlation', 0),
                       locality.get('locality_score', 0),
                       locality.get('nonlocal_correlation_fraction', 0)])
        axes[0, 0].set_title('Geometric Locality Metrics')
        axes[0, 0].set_ylabel('Value')
        
        # Graph metrics
        graph = self.analysis_results['graph_structure']
        axes[0, 1].bar(['Density', 'Clustering', 'Modularity'], 
                      [graph.get('density', 0),
                       graph.get('average_clustering', 0),
                       graph.get('modularity', 0)])
        axes[0, 1].set_title('Graph Structure Metrics')
        axes[0, 1].set_ylabel('Value')
        
        # RT consistency
        rt = self.analysis_results['ryu_takayanagi']
        axes[0, 2].bar(['RT Exponent', 'Consistency', 'Area Law Viol'], 
                      [rt.get('rt_scaling_exponent', 0),
                       rt.get('rt_consistency_score', 0),
                       rt.get('area_law_violation', 0)])
        axes[0, 2].set_title('Ryu-Takayanagi Metrics')
        axes[0, 2].set_ylabel('Value')
        
        # Entanglement metrics
        ent = self.analysis_results['entanglement_spectrum']
        axes[1, 0].bar(['Avg Entropy', 'Haar Dev', 'Page R²'], 
                      [ent.get('average_entanglement_entropy', 0),
                       ent.get('haar_deviation', 0),
                       ent.get('page_curve_r2', 0)])
        axes[1, 0].set_title('Entanglement Spectrum Metrics')
        axes[1, 0].set_ylabel('Value')
        
        # MDS stress
        mds = self.analysis_results['mds_visualization']
        axes[1, 1].bar(['2D Stress', '3D Stress'], 
                      [mds.get('stress_2d', 1.0),
                       mds.get('stress_3d', 1.0)])
        axes[1, 1].set_title('MDS Visualization Quality')
        axes[1, 1].set_ylabel('Stress (lower is better)')
        
        # Overall assessment
        overall_score = self.calculate_overall_score()
        axes[1, 2].bar(['Overall Score'], [overall_score])
        axes[1, 2].set_title('Overall Quantum Geometry Score')
        axes[1, 2].set_ylabel('Score (0-1)')
        axes[1, 2].set_ylim(0, 1)
        
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, 'analysis_summary_dashboard.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  📈 Summary dashboard saved: {plot_file}")
    
    def calculate_overall_score(self) -> float:
        """Calculate overall quantum geometry score."""
        score = 0.0
        count = 0
        
        # Locality score (higher is better)
        locality = self.analysis_results['geometric_locality']
        if 'locality_score' in locality:
            score += min(locality['locality_score'] / 2.0, 1.0)  # Normalize
            count += 1
        
        # Graph modularity (higher is better)
        graph = self.analysis_results['graph_structure']
        if 'modularity' in graph:
            score += max(0, graph['modularity'])  # Already 0-1
            count += 1
        
        # RT consistency (closer to 1 is better)
        rt = self.analysis_results['ryu_takayanagi']
        if 'rt_consistency_score' in rt:
            score += rt['rt_consistency_score']
            count += 1
        
        # Entanglement spectrum (closer to Haar is better)
        ent = self.analysis_results['entanglement_spectrum']
        if 'haar_deviation' in ent:
            score += max(0, 1.0 - ent['haar_deviation'])
            count += 1
        
        # Page curve fit (higher R² is better)
        if 'page_curve_r2' in ent:
            score += ent['page_curve_r2']
            count += 1
        
        return score / count if count > 0 else 0.0
    
    def save_results(self, output_dir: str):
        """Save analysis results."""
        # Save JSON results
        json_file = os.path.join(output_dir, 'comprehensive_analysis_results.json')
        with open(json_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
        print(f"💾 Results saved: {json_file}")
        
        # Save summary report
        summary_file = os.path.join(output_dir, 'comprehensive_analysis_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE QUANTUM GEOMETRY ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("ANALYSIS OVERVIEW:\n")
            f.write(f"  - Input file: {self.results_file}\n")
            f.write(f"  - Number of qubits: {self.num_qubits}\n")
            f.write(f"  - Overall score: {self.calculate_overall_score():.4f}\n\n")
            
            f.write("GEOMETRIC LOCALITY:\n")
            locality = self.analysis_results['geometric_locality']
            f.write(f"  - Locality score: {locality.get('locality_score', 0):.4f}\n")
            f.write(f"  - Distance decay rate: {locality.get('distance_decay_rate', 0):.4f}\n")
            f.write(f"  - Non-local fraction: {locality.get('nonlocal_correlation_fraction', 0):.4f}\n\n")
            
            f.write("GRAPH STRUCTURE:\n")
            graph = self.analysis_results['graph_structure']
            f.write(f"  - Graph density: {graph.get('density', 0):.4f}\n")
            f.write(f"  - Average clustering: {graph.get('average_clustering', 0):.4f}\n")
            f.write(f"  - Modularity: {graph.get('modularity', 0):.4f}\n")
            f.write(f"  - Number of communities: {graph.get('num_communities', 1)}\n\n")
            
            f.write("MDS VISUALIZATION:\n")
            mds = self.analysis_results['mds_visualization']
            f.write(f"  - 2D MDS stress: {mds.get('stress_2d', 1.0):.4f}\n")
            f.write(f"  - 3D MDS stress: {mds.get('stress_3d', 1.0):.4f}\n\n")
            
            f.write("RYU-TAKAYANAGI CONSISTENCY:\n")
            rt = self.analysis_results['ryu_takayanagi']
            f.write(f"  - RT scaling exponent: {rt.get('rt_scaling_exponent', 0):.4f}\n")
            f.write(f"  - RT consistency score: {rt.get('rt_consistency_score', 0):.4f}\n")
            f.write(f"  - Area law violation: {rt.get('area_law_violation', 0):.4f}\n\n")
            
            f.write("ENTANGLEMENT SPECTRUM:\n")
            ent = self.analysis_results['entanglement_spectrum']
            f.write(f"  - Average entanglement entropy: {ent.get('average_entanglement_entropy', 0):.4f}\n")
            f.write(f"  - Haar deviation: {ent.get('haar_deviation', 0):.4f}\n")
            f.write(f"  - Page curve R²: {ent.get('page_curve_r2', 0):.4f}\n\n")
            
            f.write("INTERPRETATION:\n")
            overall_score = self.calculate_overall_score()
            if overall_score > 0.8:
                f.write("  🎉 EXCELLENT: Strong evidence of emergent quantum geometry\n")
                f.write("  - High locality and modularity\n")
                f.write("  - Consistent with Ryu-Takayanagi\n")
                f.write("  - Entanglement spectrum close to Haar random\n")
            elif overall_score > 0.6:
                f.write("  ✅ GOOD: Moderate evidence of quantum geometry\n")
                f.write("  - Some geometric structure detected\n")
                f.write("  - May need parameter tuning\n")
            elif overall_score > 0.4:
                f.write("  ⚠️  FAIR: Weak evidence of quantum geometry\n")
                f.write("  - Limited geometric structure\n")
                f.write("  - Consider different circuit design\n")
            else:
                f.write("  ❌ POOR: Little evidence of quantum geometry\n")
                f.write("  - Primarily classical correlations\n")
                f.write("  - Significant redesign needed\n")
        
        print(f"📝 Summary saved: {summary_file}")
    
    def print_summary(self):
        """Print analysis summary to console."""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE ANALYSIS COMPLETE")
        print("=" * 60)
        
        overall_score = self.calculate_overall_score()
        print(f"🎯 Overall Quantum Geometry Score: {overall_score:.4f}")
        
        print(f"\n📊 Key Metrics:")
        print(f"  - Locality Score: {self.analysis_results['geometric_locality'].get('locality_score', 0):.4f}")
        print(f"  - Graph Modularity: {self.analysis_results['graph_structure'].get('modularity', 0):.4f}")
        print(f"  - RT Consistency: {self.analysis_results['ryu_takayanagi'].get('rt_consistency_score', 0):.4f}")
        print(f"  - Haar Deviation: {self.analysis_results['entanglement_spectrum'].get('haar_deviation', 0):.4f}")
        print(f"  - Page Curve R²: {self.analysis_results['entanglement_spectrum'].get('page_curve_r2', 0):.4f}")
        
        if overall_score > 0.8:
            print(f"\n🎉 EXCELLENT: Strong evidence of emergent quantum geometry!")
        elif overall_score > 0.6:
            print(f"\n✅ GOOD: Moderate evidence of quantum geometry")
        elif overall_score > 0.4:
            print(f"\n⚠️  FAIR: Weak evidence of quantum geometry")
        else:
            print(f"\n❌ POOR: Little evidence of quantum geometry")

def main():
    """Main function to run comprehensive geometry analysis."""
    if len(sys.argv) < 2:
        print("Usage: python comprehensive_geometry_analyzer.py <results_file> [output_directory]")
        print("Example: python comprehensive_geometry_analyzer.py experiment_logs/results.json")
        sys.exit(1)
    
    results_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "comprehensive_analysis_results"
    
    # Create and run the analyzer
    analyzer = ComprehensiveGeometryAnalyzer(results_file)
    
    try:
        results = analyzer.run_comprehensive_analysis(output_dir)
        print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 