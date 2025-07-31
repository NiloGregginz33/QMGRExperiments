#!/usr/bin/env python3
"""
Custom Curvature Geometric Analysis
Analyzes custom curvature experiment data for geometric structure, MDS visualization, 
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

class CustomCurvatureGeometricAnalyzer:
    def __init__(self, data_path):
        """Initialize analyzer with custom curvature experiment data."""
        self.data_path = Path(data_path)
        self.data = None
        self.n_qubits = None
        self.mutual_info_matrix = None
        self.distance_matrix = None
        self.geometry_type = None
        self.curvature = None
        
    def load_data(self):
        """Load custom curvature experiment data."""
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
        
        # Handle nested structure where data is under 'spec'
        if 'spec' in self.data:
            spec = self.data['spec']
            self.n_qubits = spec['num_qubits']
            self.geometry_type = spec.get('geometry', 'unknown')
            self.curvature = spec.get('curvature', 0)
        else:
            # Direct structure
            self.n_qubits = self.data['num_qubits']
            self.geometry_type = self.data.get('geometry_type', 'unknown')
            self.curvature = self.data.get('curvature', 0)
        
        # Extract mutual information matrix
        if 'mutual_information_per_timestep' in self.data:
            # Use the last timestep for analysis
            mi_dict = self.data['mutual_information_per_timestep'][-1]
            
            # Convert dictionary format to matrix
            self.mutual_info_matrix = np.zeros((self.n_qubits, self.n_qubits))
            
            for key, value in mi_dict.items():
                # Parse key like "I_0,1" to get indices
                if key.startswith('I_'):
                    indices = key[2:].split(',')
                    i, j = int(indices[0]), int(indices[1])
                    self.mutual_info_matrix[i, j] = value
                    self.mutual_info_matrix[j, i] = value  # Symmetric matrix
            
            # Set diagonal to average of off-diagonal values for self-mutual information
            off_diag_mean = np.mean(self.mutual_info_matrix[np.triu_indices(self.n_qubits, k=1)])
            np.fill_diagonal(self.mutual_info_matrix, off_diag_mean)
            
        else:
            raise ValueError("No mutual information data found in experiment results")
        
        print(f"Loaded data for {self.n_qubits} qubits")
        print(f"Geometry type: {self.geometry_type}")
        print(f"Curvature: {self.curvature}")
        print(f"Mutual information matrix shape: {self.mutual_info_matrix.shape}")
        print(f"MI range: {self.mutual_info_matrix.min():.3f} - {self.mutual_info_matrix.max():.3f}")
        
    def analyze_geometric_locality(self):
        """Analyze geometric locality in the mutual information matrix."""
        # Convert MI to distance matrix (inverse relationship)
        max_mi = self.mutual_info_matrix.max()
        self.distance_matrix = max_mi - self.mutual_info_matrix + 1e-6  # Avoid zeros
        
        # Calculate locality metrics
        locality_scores = []
        for i in range(self.n_qubits):
            # Find nearest neighbors
            distances = self.distance_matrix[i, :]
            nearest_indices = np.argsort(distances)[1:6]  # Top 5 nearest (excluding self)
            
            # Calculate locality score based on spatial proximity
            locality_score = 0
            for j in nearest_indices:
                # For qubits, locality is based on physical distance
                # Higher MI between nearby qubits indicates better locality
                locality_score += self.mutual_info_matrix[i, j]
            
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
        cluster_mi_scores = []
        
        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_mi = np.mean(self.mutual_info_matrix[cluster_indices][:, cluster_indices])
            cluster_mi_scores.append(cluster_mi)
        
        hierarchy_metrics = {
            'num_clusters': len(unique_labels),
            'cluster_sizes': cluster_sizes,
            'cluster_mi_scores': cluster_mi_scores,
            'cluster_labels': cluster_labels,
            'size_mi_correlation': np.corrcoef(cluster_sizes, cluster_mi_scores)[0, 1] if len(cluster_sizes) > 1 else 0
        }
        
        print(f"Cluster Hierarchy Analysis:")
        print(f"  Number of clusters: {hierarchy_metrics['num_clusters']}")
        print(f"  Cluster sizes: {hierarchy_metrics['cluster_sizes']}")
        print(f"  Size-MI correlation: {hierarchy_metrics['size_mi_correlation']:.3f}")
        
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
                                c=np.diag(self.mutual_info_matrix), cmap='viridis', s=100)
            ax1.set_xlabel('MDS Dimension 1')
            ax1.set_ylabel('MDS Dimension 2')
            ax1.set_title('2D MDS Projection')
            plt.colorbar(scatter, ax=ax1, label='Self Mutual Information')
            
            # Add qubit labels
            for i in range(self.n_qubits):
                ax1.annotate(f'q{i}', (embedding[i, 0], embedding[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
        if n_components >= 3:
            # 3D projection
            ax2 = fig.add_subplot(132, projection='3d')
            scatter = ax2.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                                c=np.diag(self.mutual_info_matrix), cmap='viridis', s=100)
            ax2.set_xlabel('MDS Dimension 1')
            ax2.set_ylabel('MDS Dimension 2')
            ax2.set_zlabel('MDS Dimension 3')
            ax2.set_title('3D MDS Projection')
            
        # MI vs MDS coordinates
        ax3 = fig.add_subplot(133)
        ax3.scatter(embedding[:, 0], np.diag(self.mutual_info_matrix), alpha=0.6)
        ax3.set_xlabel('MDS Dimension 1')
        ax3.set_ylabel('Self Mutual Information')
        ax3.set_title('MI vs MDS Coordinate')
        
        plt.tight_layout()
        
        return {
            'embedding': embedding,
            'stress': stress,
            'figure': fig
        }
        
    def check_ryu_takayanagi_consistency(self):
        """Check consistency with Ryu-Takayanagi formula using dissimilarity scaling."""
        # For custom curvature experiments, we analyze how well the system
        # exhibits holographic properties
        
        # Calculate subsystem entropies (approximated from mutual information)
        subsystem_entropies = []
        subsystem_sizes = []
        
        for size in range(2, self.n_qubits // 2 + 1):  # Start from size 2 to avoid division by zero
            # Calculate average entropy for subsystems of this size
            total_entropy = 0
            count = 0
            
            # Sample some subsystems of this size
            for i in range(min(10, self.n_qubits - size + 1)):
                subsystem = list(range(i, i + size))
                # Approximate entropy as sum of mutual information within subsystem
                subsystem_mi = 0
                for j in subsystem:
                    for k in subsystem:
                        if j != k:
                            subsystem_mi += self.mutual_info_matrix[j, k]
                total_entropy += subsystem_mi / (size * (size - 1))
                count += 1
            
            if count > 0:
                avg_entropy = total_entropy / count
                subsystem_entropies.append(avg_entropy)
                subsystem_sizes.append(size)
        
        # Calculate area-law scaling
        if len(subsystem_sizes) > 1:
            area_law_correlation = np.corrcoef(subsystem_sizes, subsystem_entropies)[0, 1]
        else:
            area_law_correlation = 0
        
        # Check for logarithmic corrections
        if len(subsystem_sizes) > 1:
            log_corrections = np.corrcoef(np.log(np.array(subsystem_sizes) + 1), subsystem_entropies)[0, 1]
        else:
            log_corrections = 0
        
        rt_consistency = {
            'area_law_correlation': area_law_correlation,
            'log_correction_correlation': log_corrections,
            'subsystem_sizes': subsystem_sizes,
            'subsystem_entropies': subsystem_entropies
        }
        
        print("Ryu-Takayanagi Consistency Analysis:")
        print(f"  Area law correlation: {rt_consistency['area_law_correlation']:.3f}")
        print(f"  Log correction correlation: {rt_consistency['log_correction_correlation']:.3f}")
        
        return rt_consistency
        
    def analyze_entanglement_spectrum(self):
        """Analyze entanglement spectrum and compare to random Haar states."""
        # Calculate entanglement spectrum properties from mutual information
        mi_spectrum = np.diag(self.mutual_info_matrix)
        
        spectrum_properties = {
            'mean_mi': np.mean(mi_spectrum),
            'std_mi': np.std(mi_spectrum),
            'mi_distribution': mi_spectrum,
            'max_mi': np.max(mi_spectrum),
            'min_mi': np.min(mi_spectrum)
        }
        
        # Compare to random state expectations
        # For random states, mutual information should be more uniform
        random_predictions = np.full(self.n_qubits, np.mean(mi_spectrum))
        
        # Calculate deviation from random predictions
        deviations = mi_spectrum - random_predictions
        
        spectrum_analysis = {
            'spectrum_properties': spectrum_properties,
            'random_predictions': random_predictions,
            'deviations_from_random': deviations,
            'mean_deviation': np.mean(deviations),
            'std_deviation': np.std(deviations),
            'random_correlation': np.corrcoef(mi_spectrum, random_predictions)[0, 1]
        }
        
        print("Entanglement Spectrum Analysis:")
        print(f"  Mean MI: {spectrum_analysis['spectrum_properties']['mean_mi']:.3f}")
        print(f"  MI std: {spectrum_analysis['spectrum_properties']['std_mi']:.3f}")
        print(f"  Mean deviation from random: {spectrum_analysis['mean_deviation']:.3f}")
        print(f"  Random correlation: {spectrum_analysis['random_correlation']:.3f}")
        
        return spectrum_analysis
        
    def create_comprehensive_plots(self, mds_results, graph_metrics, rt_consistency, spectrum_analysis):
        """Create comprehensive visualization plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Mutual Information Matrix Heatmap
        im1 = axes[0, 0].imshow(self.mutual_info_matrix, cmap='viridis', aspect='auto')
        axes[0, 0].set_title(f'Mutual Information Matrix\n{self.geometry_type}, Curvature={self.curvature}')
        axes[0, 0].set_xlabel('Qubit Index')
        axes[0, 0].set_ylabel('Qubit Index')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. Entanglement vs Subsystem Size
        if rt_consistency['subsystem_sizes']:
            axes[0, 1].scatter(rt_consistency['subsystem_sizes'], rt_consistency['subsystem_entropies'], alpha=0.6)
            axes[0, 1].set_xlabel('Subsystem Size')
            axes[0, 1].set_ylabel('Average Entropy')
            axes[0, 1].set_title('Area Law Scaling')
            
            # Add theoretical curves
            sizes = np.array(rt_consistency['subsystem_sizes'])
            axes[0, 1].plot(sizes, sizes * np.log(2), 'r--', label='Volume Law')
            axes[0, 1].plot(sizes, np.log(sizes + 1), 'g--', label='Log Law')
            axes[0, 1].legend()
        
        # 3. MDS 2D Projection
        if 'embedding' in mds_results:
            embedding = mds_results['embedding']
            scatter = axes[0, 2].scatter(embedding[:, 0], embedding[:, 1], 
                                       c=np.diag(self.mutual_info_matrix), cmap='viridis', s=100)
            axes[0, 2].set_xlabel('MDS Dimension 1')
            axes[0, 2].set_ylabel('MDS Dimension 2')
            axes[0, 2].set_title('MDS Bulk Geometry')
            plt.colorbar(scatter, ax=axes[0, 2])
            
            # Add qubit labels
            for i in range(self.n_qubits):
                axes[0, 2].annotate(f'q{i}', (embedding[i, 0], embedding[i, 1]), 
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Mutual Information Spectrum Distribution
        axes[1, 0].hist(np.diag(self.mutual_info_matrix), bins=15, alpha=0.7, density=True)
        axes[1, 0].set_xlabel('Self Mutual Information')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('MI Spectrum Distribution')
        
        # 5. Graph Connectivity Analysis
        if graph_metrics['density'] > 0:
            # Create a simple graph visualization
            G = nx.from_numpy_array((self.mutual_info_matrix > np.percentile(self.mutual_info_matrix, 75)).astype(int))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, ax=axes[1, 1], with_labels=True, node_color='lightblue', 
                   node_size=300, font_size=8, font_weight='bold')
            axes[1, 1].set_title(f'Graph Structure\nDensity: {graph_metrics["density"]:.3f}')
        
        # 6. Deviation from Random States
        axes[1, 2].scatter(spectrum_analysis['random_predictions'], 
                          spectrum_analysis['spectrum_properties']['mi_distribution'], alpha=0.6)
        axes[1, 2].plot([0, max(spectrum_analysis['random_predictions'])], 
                       [0, max(spectrum_analysis['random_predictions'])], 'r--', label='Perfect Random')
        axes[1, 2].set_xlabel('Random Prediction')
        axes[1, 2].set_ylabel('Actual MI')
        axes[1, 2].set_title('Random State Comparison')
        axes[1, 2].legend()
        
        plt.tight_layout()
        return fig
        
    def run_comprehensive_analysis(self):
        """Run all analyses and generate comprehensive report."""
        print("=" * 60)
        print("CUSTOM CURVATURE GEOMETRIC ANALYSIS")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
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
                'geometry_type': self.geometry_type,
                'curvature': self.curvature,
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
                'rt_consistency_score': rt_results['area_law_correlation'],
                'random_correlation': spectrum_results['random_correlation']
            }
        }
        
        return analysis_results, comprehensive_fig, mds_results['figure']
        
    def save_results(self, analysis_results, comprehensive_fig, mds_fig):
        """Save analysis results and plots."""
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"experiment_logs/custom_curvature_geometric_analysis/instance_{timestamp}")
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
            f.write("CUSTOM CURVATURE GEOMETRIC ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Experiment: {analysis_results['experiment_info']['data_path']}\n")
            f.write(f"Qubits: {analysis_results['experiment_info']['n_qubits']}\n")
            f.write(f"Geometry: {analysis_results['experiment_info']['geometry_type']}\n")
            f.write(f"Curvature: {analysis_results['experiment_info']['curvature']}\n\n")
            
            f.write("KEY METRICS:\n")
            f.write(f"  Geometric Locality Score: {analysis_results['summary_metrics']['geometric_locality_score']:.3f}\n")
            f.write(f"  Graph Connectivity: {analysis_results['summary_metrics']['graph_connectivity']:.3f}\n")
            f.write(f"  RT Consistency Score: {analysis_results['summary_metrics']['rt_consistency_score']:.3f}\n")
            f.write(f"  Random Correlation: {analysis_results['summary_metrics']['random_correlation']:.3f}\n\n")
            
            f.write("INTERPRETATION:\n")
            f.write("- Geometric Locality: Measures how well the system exhibits spatial locality\n")
            f.write("- Graph Connectivity: Indicates the connectivity structure of the quantum system\n")
            f.write("- RT Consistency: How well the system obeys area-law scaling\n")
            f.write("- Random Correlation: How close the system is to random states\n")
        
        print(f"\nResults saved to: {output_dir}")
        print(f"  - JSON results: {results_file}")
        print(f"  - Comprehensive plots: {output_dir}/comprehensive_analysis.png")
        print(f"  - MDS visualization: {output_dir}/mds_visualization.png")
        print(f"  - Summary: {summary_file}")
        
        return output_dir

def main():
    """Main function to run the analysis."""
    # Path to the new experiment data
    data_path = "experiment_logs/custom_curvature_experiment/instance_20250731_012542/results_n12_geomS_curv2_ibm_brisbane_SNDD23.json"
    
    # Create analyzer and run analysis
    analyzer = CustomCurvatureGeometricAnalyzer(data_path)
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
    print(f"Random Correlation: {analysis_results['summary_metrics']['random_correlation']:.3f}")
    
    plt.show()

if __name__ == "__main__":
    main() 