#!/usr/bin/env python3
"""
Page Curve Geometric Analysis
Analyzes page curve data for geometric structure, MDS visualization, 
Ryu-Takayanagi consistency, and entanglement spectrum analysis.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from datetime import datetime
import seaborn as sns

class PageCurveGeometricAnalyzer:
    def __init__(self, data_path):
        """Initialize analyzer with page curve data."""
        self.data_path = Path(data_path)
        self.data = None
        self.n_qubits = None
        self.entanglement_entropies = None
        self.bipartitions = None
        self.mutual_info_matrix = None
        self.distance_matrix = None
        
    def load_data(self):
        """Load page curve experiment data."""
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
        
        self.n_qubits = self.data['num_qubits']
        self.entanglement_entropies = np.array(self.data['page_curve_analysis']['entanglement_entropies'])
        self.bipartitions = self.data['page_curve_analysis']['bipartitions']
        
        print(f"Loaded data for {self.n_qubits} qubits")
        print(f"Number of bipartitions: {len(self.bipartitions)}")
        print(f"Entanglement entropy range: {self.entanglement_entropies.min():.3f} - {self.entanglement_entropies.max():.3f}")
        
    def construct_mutual_information_matrix(self):
        """Construct mutual information matrix from entanglement entropies."""
        n_partitions = len(self.bipartitions)
        self.mutual_info_matrix = np.zeros((n_partitions, n_partitions))
        
        # Calculate mutual information between all pairs of bipartitions
        for i, part1 in enumerate(self.bipartitions):
            for j, part2 in enumerate(self.bipartitions):
                if i == j:
                    # Self mutual information is just the entropy
                    self.mutual_info_matrix[i, j] = self.entanglement_entropies[i]
                else:
                    # Calculate mutual information using S(A) + S(B) - S(A∪B)
                    # For bipartitions, we need to be careful about the union
                    union_part = list(set(part1 + part2))
                    union_idx = self.bipartitions.index(union_part) if union_part in self.bipartitions else None
                    
                    if union_idx is not None:
                        mi = (self.entanglement_entropies[i] + 
                              self.entanglement_entropies[j] - 
                              self.entanglement_entropies[union_idx])
                        self.mutual_info_matrix[i, j] = max(0, mi)  # MI should be non-negative
                    else:
                        # Approximate using distance-based mutual information
                        self.mutual_info_matrix[i, j] = self._approximate_mutual_info(part1, part2)
        
        print(f"Constructed mutual information matrix: {self.mutual_info_matrix.shape}")
        print(f"MI range: {self.mutual_info_matrix.min():.3f} - {self.mutual_info_matrix.max():.3f}")
        
    def _approximate_mutual_info(self, part1, part2):
        """Approximate mutual information based on partition overlap."""
        overlap = len(set(part1) & set(part2))
        total = len(set(part1) | set(part2))
        if total == 0:
            return 0
        # Use Jaccard similarity as a proxy for mutual information
        jaccard = overlap / total
        return jaccard * min(len(part1), len(part2)) * 0.1  # Scale factor
        
    def analyze_geometric_locality(self):
        """Analyze geometric locality in the mutual information matrix."""
        # Convert MI to distance matrix (inverse relationship)
        max_mi = self.mutual_info_matrix.max()
        self.distance_matrix = max_mi - self.mutual_info_matrix + 1e-6  # Avoid zeros
        
        # Calculate locality metrics
        locality_scores = []
        for i in range(len(self.bipartitions)):
            # Find nearest neighbors
            distances = self.distance_matrix[i, :]
            nearest_indices = np.argsort(distances)[1:6]  # Top 5 nearest (excluding self)
            
            # Calculate locality score based on partition similarity
            locality_score = 0
            for j in nearest_indices:
                part1, part2 = self.bipartitions[i], self.bipartitions[j]
                overlap = len(set(part1) & set(part2))
                total = len(set(part1) | set(part2))
                if total > 0:
                    locality_score += overlap / total
            
            locality_scores.append(locality_score / len(nearest_indices))
        
        avg_locality = np.mean(locality_scores)
        print(f"Average geometric locality score: {avg_locality:.3f}")
        
        return {
            'locality_scores': locality_scores,
            'average_locality': avg_locality,
            'distance_matrix': self.distance_matrix
        }
        
    def analyze_graph_structure(self):
        """Analyze graph structure from mutual information matrix."""
        # Create adjacency matrix (threshold-based)
        threshold = np.percentile(self.mutual_info_matrix, 75)
        adjacency_matrix = (self.mutual_info_matrix > threshold).astype(int)
        
        # Create NetworkX graph
        G = nx.from_numpy_array(adjacency_matrix)
        
        # Calculate graph metrics
        metrics = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'average_clustering': nx.average_clustering(G),
            'average_shortest_path': nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf'),
            'num_connected_components': nx.number_connected_components(G),
            'largest_component_size': len(max(nx.connected_components(G), key=len)) if nx.number_connected_components(G) > 0 else 0
        }
        
        print("Graph Structure Analysis:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
            
        return metrics, G
        
    def analyze_cluster_hierarchy(self):
        """Analyze cluster hierarchy using hierarchical clustering."""
        # Use distance matrix for clustering
        clustering = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=0.5,
            linkage='ward'
        )
        
        # Fit clustering
        cluster_labels = clustering.fit_predict(self.distance_matrix)
        
        # Analyze cluster properties
        unique_labels = np.unique(cluster_labels)
        cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels]
        cluster_entropies = []
        
        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_entropy = np.mean(self.entanglement_entropies[cluster_indices])
            cluster_entropies.append(cluster_entropy)
        
        hierarchy_metrics = {
            'num_clusters': len(unique_labels),
            'cluster_sizes': cluster_sizes,
            'cluster_entropies': cluster_entropies,
            'cluster_labels': cluster_labels,
            'size_entropy_correlation': np.corrcoef(cluster_sizes, cluster_entropies)[0, 1] if len(cluster_sizes) > 1 else 0
        }
        
        print(f"Cluster Hierarchy Analysis:")
        print(f"  Number of clusters: {hierarchy_metrics['num_clusters']}")
        print(f"  Cluster sizes: {hierarchy_metrics['cluster_sizes']}")
        print(f"  Size-entropy correlation: {hierarchy_metrics['size_entropy_correlation']:.3f}")
        
        return hierarchy_metrics
        
    def run_mds_visualization(self, n_components=3):
        """Run MDS to visualize bulk geometry."""
        # Use distance matrix for MDS
        mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
        embedding = mds.fit_transform(self.distance_matrix)
        
        # Calculate stress (goodness of fit)
        stress = mds.stress_
        
        print(f"MDS embedding stress: {stress:.3f}")
        
        # Create visualization
        fig = plt.figure(figsize=(15, 5))
        
        if n_components >= 2:
            # 2D projection
            ax1 = fig.add_subplot(131)
            scatter = ax1.scatter(embedding[:, 0], embedding[:, 1], 
                                c=self.entanglement_entropies, cmap='viridis', s=50)
            ax1.set_xlabel('MDS Dimension 1')
            ax1.set_ylabel('MDS Dimension 2')
            ax1.set_title('2D MDS Projection')
            plt.colorbar(scatter, ax=ax1, label='Entanglement Entropy')
            
        if n_components >= 3:
            # 3D projection
            ax2 = fig.add_subplot(132, projection='3d')
            scatter = ax2.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                                c=self.entanglement_entropies, cmap='viridis', s=50)
            ax2.set_xlabel('MDS Dimension 1')
            ax2.set_ylabel('MDS Dimension 2')
            ax2.set_zlabel('MDS Dimension 3')
            ax2.set_title('3D MDS Projection')
            
        # Entropy vs MDS coordinates
        ax3 = fig.add_subplot(133)
        ax3.scatter(embedding[:, 0], self.entanglement_entropies, alpha=0.6)
        ax3.set_xlabel('MDS Dimension 1')
        ax3.set_ylabel('Entanglement Entropy')
        ax3.set_title('Entropy vs MDS Coordinate')
        
        plt.tight_layout()
        
        return {
            'embedding': embedding,
            'stress': stress,
            'figure': fig
        }
        
    def check_ryu_takayanagi_consistency(self):
        """Check consistency with Ryu-Takayanagi formula using dissimilarity scaling."""
        # RT formula: S(A) = Area(γ_A) / 4G_N
        # For our purposes, we'll check if entropy scales with boundary size
        
        boundary_sizes = [len(part) for part in self.bipartitions]
        
        # Calculate area-law scaling
        area_law_correlation = np.corrcoef(boundary_sizes, self.entanglement_entropies)[0, 1]
        
        # Check for logarithmic corrections (expected for 1D systems)
        log_corrections = np.corrcoef(np.log(np.array(boundary_sizes) + 1), self.entanglement_entropies)[0, 1]
        
        # Analyze RT surface properties
        rt_metrics = []
        for i, part in enumerate(self.bipartitions):
            if len(part) <= self.n_qubits // 2:  # Only consider smaller subsystems
                # Calculate complement
                complement = list(set(range(self.n_qubits)) - set(part))
                
                # Find complement index
                try:
                    complement_idx = self.bipartitions.index(complement)
                    # RT predicts S(A) = S(A^c) for pure states
                    rt_violation = abs(self.entanglement_entropies[i] - self.entanglement_entropies[complement_idx])
                    rt_metrics.append(rt_violation)
                except ValueError:
                    continue
        
        rt_consistency = {
            'area_law_correlation': area_law_correlation,
            'log_correction_correlation': log_corrections,
            'rt_violation_mean': np.mean(rt_metrics) if rt_metrics else 0,
            'rt_violation_std': np.std(rt_metrics) if rt_metrics else 0,
            'rt_violations': rt_metrics
        }
        
        print("Ryu-Takayanagi Consistency Analysis:")
        print(f"  Area law correlation: {rt_consistency['area_law_correlation']:.3f}")
        print(f"  Log correction correlation: {rt_consistency['log_correction_correlation']:.3f}")
        print(f"  RT violation mean: {rt_consistency['rt_violation_mean']:.3f}")
        print(f"  RT violation std: {rt_consistency['rt_violation_std']:.3f}")
        
        return rt_consistency
        
    def analyze_entanglement_spectrum(self):
        """Analyze entanglement spectrum and compare to random Haar states."""
        # Calculate entanglement spectrum properties
        spectrum_properties = {
            'mean_entropy': np.mean(self.entanglement_entropies),
            'std_entropy': np.std(self.entanglement_entropies),
            'entropy_distribution': self.entanglement_entropies,
            'max_entropy': np.max(self.entanglement_entropies),
            'min_entropy': np.min(self.entanglement_entropies)
        }
        
        # Compare to random Haar state expectations
        # For random Haar states, Page's formula gives:
        # S(A) ≈ min(|A|, |A^c|) * log(2) - 1/2 for large systems
        
        haar_predictions = []
        for part in self.bipartitions:
            size_a = len(part)
            size_ac = self.n_qubits - size_a
            min_size = min(size_a, size_ac)
            
            # Page's formula (approximate for finite systems)
            if min_size > 0:
                haar_entropy = min_size * np.log(2) - 0.5
                haar_predictions.append(haar_entropy)
            else:
                haar_predictions.append(0)
        
        # Calculate deviation from Haar predictions
        deviations = np.array(self.entanglement_entropies) - np.array(haar_predictions)
        
        spectrum_analysis = {
            'spectrum_properties': spectrum_properties,
            'haar_predictions': haar_predictions,
            'deviations_from_haar': deviations,
            'mean_deviation': np.mean(deviations),
            'std_deviation': np.std(deviations),
            'haar_correlation': np.corrcoef(self.entanglement_entropies, haar_predictions)[0, 1]
        }
        
        print("Entanglement Spectrum Analysis:")
        print(f"  Mean entropy: {spectrum_analysis['spectrum_properties']['mean_entropy']:.3f}")
        print(f"  Entropy std: {spectrum_analysis['spectrum_properties']['std_entropy']:.3f}")
        print(f"  Mean deviation from Haar: {spectrum_analysis['mean_deviation']:.3f}")
        print(f"  Haar correlation: {spectrum_analysis['haar_correlation']:.3f}")
        
        return spectrum_analysis
        
    def create_comprehensive_plots(self, mds_results, graph_metrics, rt_consistency, spectrum_analysis):
        """Create comprehensive visualization plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Mutual Information Matrix Heatmap
        im1 = axes[0, 0].imshow(self.mutual_info_matrix, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Mutual Information Matrix')
        axes[0, 0].set_xlabel('Partition Index')
        axes[0, 0].set_ylabel('Partition Index')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. Entanglement Entropy vs Subsystem Size
        boundary_sizes = [len(part) for part in self.bipartitions]
        axes[0, 1].scatter(boundary_sizes, self.entanglement_entropies, alpha=0.6)
        axes[0, 1].set_xlabel('Subsystem Size')
        axes[0, 1].set_ylabel('Entanglement Entropy')
        axes[0, 1].set_title('Area Law Scaling')
        
        # Add theoretical curves
        sizes = np.array(boundary_sizes)
        axes[0, 1].plot(sizes, sizes * np.log(2), 'r--', label='Volume Law')
        axes[0, 1].plot(sizes, np.log(sizes + 1), 'g--', label='Log Law')
        axes[0, 1].legend()
        
        # 3. MDS 2D Projection
        if 'embedding' in mds_results:
            embedding = mds_results['embedding']
            scatter = axes[0, 2].scatter(embedding[:, 0], embedding[:, 1], 
                                       c=self.entanglement_entropies, cmap='viridis', s=50)
            axes[0, 2].set_xlabel('MDS Dimension 1')
            axes[0, 2].set_ylabel('MDS Dimension 2')
            axes[0, 2].set_title('MDS Bulk Geometry')
            plt.colorbar(scatter, ax=axes[0, 2])
        
        # 4. Entanglement Spectrum Distribution
        axes[1, 0].hist(self.entanglement_entropies, bins=20, alpha=0.7, density=True)
        axes[1, 0].set_xlabel('Entanglement Entropy')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Entanglement Spectrum')
        
        # 5. RT Consistency Check
        if rt_consistency['rt_violations']:
            axes[1, 1].hist(rt_consistency['rt_violations'], bins=15, alpha=0.7)
            axes[1, 1].set_xlabel('RT Violation')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Ryu-Takayanagi Violations')
        
        # 6. Deviation from Haar States
        axes[1, 2].scatter(spectrum_analysis['haar_predictions'], 
                          self.entanglement_entropies, alpha=0.6)
        axes[1, 2].plot([0, max(spectrum_analysis['haar_predictions'])], 
                       [0, max(spectrum_analysis['haar_predictions'])], 'r--', label='Perfect Haar')
        axes[1, 2].set_xlabel('Haar Prediction')
        axes[1, 2].set_ylabel('Actual Entropy')
        axes[1, 2].set_title('Haar State Comparison')
        axes[1, 2].legend()
        
        plt.tight_layout()
        return fig
        
    def run_comprehensive_analysis(self):
        """Run all analyses and generate comprehensive report."""
        print("=" * 60)
        print("PAGE CURVE GEOMETRIC ANALYSIS")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Construct mutual information matrix
        self.construct_mutual_information_matrix()
        
        # Run all analyses
        locality_results = self.analyze_geometric_locality()
        graph_results, graph = self.analyze_graph_structure()
        cluster_results = self.analyze_cluster_hierarchy()
        mds_results = self.run_mds_visualization()
        rt_results = self.check_ryu_takayanagi_consistency()
        spectrum_results = self.analyze_entanglement_spectrum()
        
        # Create comprehensive plots
        comprehensive_fig = self.create_comprehensive_plots(
            mds_results, graph_results, rt_results, spectrum_results
        )
        
        # Compile results
        analysis_results = {
            'experiment_info': {
                'data_path': str(self.data_path),
                'n_qubits': self.n_qubits,
                'num_bipartitions': len(self.bipartitions),
                'analysis_timestamp': datetime.now().isoformat()
            },
            'locality_analysis': locality_results,
            'graph_analysis': graph_results,
            'cluster_analysis': cluster_results,
            'mds_analysis': {
                'stress': mds_results['stress'],
                'embedding_shape': mds_results['embedding'].shape
            },
            'rt_consistency': rt_results,
            'spectrum_analysis': spectrum_results,
            'summary_metrics': {
                'geometric_locality_score': locality_results['average_locality'],
                'graph_connectivity': graph_results['density'],
                'rt_consistency_score': 1 - rt_results['rt_violation_mean'],
                'haar_correlation': spectrum_results['haar_correlation']
            }
        }
        
        return analysis_results, comprehensive_fig, mds_results['figure']
        
    def save_results(self, analysis_results, comprehensive_fig, mds_fig):
        """Save analysis results and plots."""
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"experiment_logs/page_curve_geometric_analysis/instance_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        results_file = output_dir / "geometric_analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Save plots
        comprehensive_fig.savefig(output_dir / "comprehensive_analysis.png", dpi=300, bbox_inches='tight')
        mds_fig.savefig(output_dir / "mds_visualization.png", dpi=300, bbox_inches='tight')
        
        # Save summary
        summary_file = output_dir / "analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("PAGE CURVE GEOMETRIC ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Experiment: {analysis_results['experiment_info']['data_path']}\n")
            f.write(f"Qubits: {analysis_results['experiment_info']['n_qubits']}\n")
            f.write(f"Bipartitions: {analysis_results['experiment_info']['num_bipartitions']}\n\n")
            
            f.write("KEY METRICS:\n")
            f.write(f"  Geometric Locality Score: {analysis_results['summary_metrics']['geometric_locality_score']:.3f}\n")
            f.write(f"  Graph Connectivity: {analysis_results['summary_metrics']['graph_connectivity']:.3f}\n")
            f.write(f"  RT Consistency Score: {analysis_results['summary_metrics']['rt_consistency_score']:.3f}\n")
            f.write(f"  Haar Correlation: {analysis_results['summary_metrics']['haar_correlation']:.3f}\n\n")
            
            f.write("INTERPRETATION:\n")
            f.write("- Geometric Locality: Measures how well the system exhibits spatial locality\n")
            f.write("- Graph Connectivity: Indicates the connectivity structure of the quantum system\n")
            f.write("- RT Consistency: How well the system obeys Ryu-Takayanagi formula\n")
            f.write("- Haar Correlation: How close the system is to random Haar states\n")
        
        print(f"\nResults saved to: {output_dir}")
        print(f"  - JSON results: {results_file}")
        print(f"  - Comprehensive plots: {output_dir}/comprehensive_analysis.png")
        print(f"  - MDS visualization: {output_dir}/mds_visualization.png")
        print(f"  - Summary: {summary_file}")
        
        return output_dir

def main():
    """Main function to run the analysis."""
    # Path to the new experiment data
    data_path = "experiment_logs/custom_curvature_experiment/instance_20250730_224708/results_n12_geomS_curv10_ibm_brisbane_KJRBYG.json"
    
    # Create analyzer and run analysis
    analyzer = PageCurveGeometricAnalyzer(data_path)
    analysis_results, comprehensive_fig, mds_fig = analyzer.run_comprehensive_analysis()
    
    # Save results
    output_dir = analyzer.save_results(analysis_results, comprehensive_fig, mds_fig)
    
    # Display summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Geometric Locality: {analysis_results['summary_metrics']['geometric_locality_score']:.3f}")
    print(f"Graph Connectivity: {analysis_results['summary_metrics']['graph_connectivity']:.3f}")
    print(f"RT Consistency: {analysis_results['summary_metrics']['rt_consistency_score']:.3f}")
    print(f"Haar Correlation: {analysis_results['summary_metrics']['haar_correlation']:.3f}")
    
    plt.show()

if __name__ == "__main__":
    main() 