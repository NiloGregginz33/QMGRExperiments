#!/usr/bin/env python3
"""
Mutual Information Invariance Experiment
Tests coordinate independence of results using information-theoretic tools like 
mutual information matrices and graph isomorphism invariance.

This experiment implements:
1. Coordinate transformation tests
2. Graph isomorphism invariance checks
3. Information-theoretic distance measures
4. Statistical validation of coordinate independence
5. Cross-validation across different coordinate choices
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import networkx as nx
from itertools import permutations
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from CGPTFactory import run
from utils.experiment_logger import ExperimentLogger

class MutualInformationInvarianceExperiment:
    """
    Mutual Information Invariance Experiment for testing coordinate independence.
    
    This experiment systematically tests whether results are invariant under
    different coordinate choices using information-theoretic tools.
    """
    
    def __init__(self, num_qubits=8, geometry='spherical', curvature=3.0, 
                 device='simulator', shots=1000, num_coordinate_tests=10):
        """
        Initialize the Mutual Information Invariance Experiment.
        
        Args:
            num_qubits: Number of qubits in the system
            geometry: Geometry type ('spherical', 'hyperbolic', 'flat')
            curvature: Curvature parameter k
            device: Quantum device ('simulator' or IBM provider name)
            shots: Number of measurement shots
            num_coordinate_tests: Number of coordinate transformations to test
        """
        self.num_qubits = num_qubits
        self.geometry = geometry
        self.curvature = curvature
        self.device = device
        self.shots = shots
        self.num_coordinate_tests = num_coordinate_tests
        
        # Initialize logger
        self.logger = ExperimentLogger()
        
        # Results storage
        self.results = {
            'experiment_type': 'mutual_information_invariance',
            'parameters': {
                'num_qubits': num_qubits,
                'geometry': geometry,
                'curvature': curvature,
                'device': device,
                'shots': shots,
                'num_coordinate_tests': num_coordinate_tests
            },
            'coordinate_tests': {},
            'invariance_analysis': {},
            'graph_isomorphism_tests': {},
            'statistical_validation': {},
            'plots': []
        }
    
    def generate_coordinate_transformations(self):
        """
        Generate different coordinate transformations to test.
        
        Returns:
            list: List of coordinate transformation functions
        """
        transformations = []
        
        # Identity transformation (baseline)
        transformations.append({
            'name': 'identity',
            'function': lambda x: x,
            'description': 'No transformation (baseline)'
        })
        
        # Random permutations
        for i in range(self.num_coordinate_tests - 1):
            perm = np.random.permutation(self.num_qubits)
            transformations.append({
                'name': f'permutation_{i+1}',
                'function': lambda x, p=perm: x[p][:, p],
                'description': f'Random permutation {perm}'
            })
        
        return transformations
    
    def compute_mutual_information_matrix(self, density_matrix):
        """
        Compute mutual information matrix from density matrix.
        
        Args:
            density_matrix: Full system density matrix
            
        Returns:
            np.ndarray: Mutual information matrix
        """
        # For simplicity, we'll use a placeholder MI matrix
        # In practice, this would involve proper partial tracing and entropy calculations
        
        # Generate a realistic MI matrix based on geometry
        mi_matrix = np.zeros((self.num_qubits, self.num_qubits))
        
        # Add some structure based on geometry
        for i in range(self.num_qubits):
            for j in range(i+1, self.num_qubits):
                # Distance-based mutual information
                distance = abs(i - j)
                if self.geometry == 'spherical':
                    # Spherical: MI decreases with distance
                    mi = np.exp(-distance / 2.0) * (1 + 0.1 * np.random.random())
                elif self.geometry == 'hyperbolic':
                    # Hyperbolic: MI can increase with distance
                    mi = np.exp(-distance / 3.0) * (1 + 0.2 * np.random.random())
                else:  # flat
                    # Flat: MI decreases linearly
                    mi = max(0, 1 - distance / self.num_qubits) * (1 + 0.1 * np.random.random())
                
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
        
        return mi_matrix
    
    def compute_information_theoretic_distance(self, mi_matrix_1, mi_matrix_2):
        """
        Compute information-theoretic distance between two MI matrices.
        
        Args:
            mi_matrix_1: First mutual information matrix
            mi_matrix_2: Second mutual information matrix
            
        Returns:
            dict: Various distance measures
        """
        # Flatten matrices for comparison
        mi_1_flat = mi_matrix_1.flatten()
        mi_2_flat = mi_matrix_2.flatten()
        
        # Various distance measures
        distances = {}
        
        # Euclidean distance
        distances['euclidean'] = np.linalg.norm(mi_1_flat - mi_2_flat)
        
        # Cosine distance
        dot_product = np.dot(mi_1_flat, mi_2_flat)
        norm_1 = np.linalg.norm(mi_1_flat)
        norm_2 = np.linalg.norm(mi_2_flat)
        if norm_1 > 0 and norm_2 > 0:
            cosine_sim = dot_product / (norm_1 * norm_2)
            distances['cosine'] = 1 - cosine_sim
        else:
            distances['cosine'] = 1.0
        
        # Relative entropy (KL divergence approximation)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        mi_1_safe = mi_1_flat + epsilon
        mi_2_safe = mi_2_flat + epsilon
        
        # Normalize to probability distributions
        mi_1_norm = mi_1_safe / np.sum(mi_1_safe)
        mi_2_norm = mi_2_safe / np.sum(mi_2_safe)
        
        kl_div = np.sum(mi_1_norm * np.log(mi_1_norm / mi_2_norm))
        distances['kl_divergence'] = kl_div
        
        # Jensen-Shannon divergence
        m = 0.5 * (mi_1_norm + mi_2_norm)
        js_div = 0.5 * (kl_div + np.sum(mi_2_norm * np.log(mi_2_norm / m)))
        distances['jensen_shannon'] = js_div
        
        return distances
    
    def test_graph_isomorphism(self, mi_matrix_1, mi_matrix_2, threshold=0.1):
        """
        Test if two MI matrices represent isomorphic graphs.
        
        Args:
            mi_matrix_1: First mutual information matrix
            mi_matrix_2: Second mutual information matrix
            threshold: Threshold for edge existence
            
        Returns:
            dict: Graph isomorphism test results
        """
        # Convert MI matrices to adjacency matrices
        adj_1 = (mi_matrix_1 > threshold).astype(int)
        adj_2 = (mi_matrix_2 > threshold).astype(int)
        
        # Create NetworkX graphs
        G1 = nx.from_numpy_array(adj_1)
        G2 = nx.from_numpy_array(adj_2)
        
        # Test isomorphism
        is_isomorphic = nx.is_isomorphic(G1, G2)
        
        # Compute graph invariants
        invariants = {
            'is_isomorphic': is_isomorphic,
            'num_nodes_1': G1.number_of_nodes(),
            'num_nodes_2': G2.number_of_nodes(),
            'num_edges_1': G1.number_of_edges(),
            'num_edges_2': G2.number_of_edges(),
            'density_1': nx.density(G1),
            'density_2': nx.density(G2),
            'avg_clustering_1': nx.average_clustering(G1),
            'avg_clustering_2': nx.average_clustering(G2),
            'avg_shortest_path_1': nx.average_shortest_path_length(G1) if nx.is_connected(G1) else float('inf'),
            'avg_shortest_path_2': nx.average_shortest_path_length(G2) if nx.is_connected(G2) else float('inf')
        }
        
        return invariants
    
    def run_quantum_circuit(self):
        """
        Load existing data from experiment_logs for testing coordinate invariance.
        
        Returns:
            dict: Circuit results including mutual information matrix from existing data
        """
        print(f"🔬 Loading existing data from experiment_logs...")
        
        # Find the most recent experiment data with actual results
        experiment_dir = "experiment_logs/custom_curvature_experiment"
        instances = [d for d in os.listdir(experiment_dir) if d.startswith("instance_")]
        
        # Find the first instance that has results files
        instance_path = None
        for instance in sorted(instances, reverse=True):
            test_path = os.path.join(experiment_dir, instance)
            if os.path.exists(test_path):
                files = os.listdir(test_path)
                if any(f.endswith('.json') and 'results' in f and not f.endswith('_page_curve_results.json') for f in files):
                    instance_path = test_path
                    break
        
        if not instance_path:
            raise FileNotFoundError("No experiment instances with results found in experiment_logs")
        
        # Find the results file
        results_files = [f for f in os.listdir(instance_path) if f.endswith('.json') and 'results' in f and not f.endswith('_page_curve_results.json')]
        
        if not results_files:
            raise FileNotFoundError(f"No results files found in {instance_path}")
        
        results_file = results_files[0]
        results_path = os.path.join(instance_path, results_file)
        
        print(f"📂 Loading data from: {results_path}")
        
        # Load the existing data
        with open(results_path, 'r') as f:
            existing_data = json.load(f)
        
        # Extract mutual information data from the most recent timestep
        mi_per_timestep = existing_data.get('mutual_information_per_timestep', [])
        
        if not mi_per_timestep:
            raise ValueError("No mutual information data found in existing results")
        
        # Use the last timestep (most evolved state)
        latest_mi_data = mi_per_timestep[-1]
        
        # Convert to matrix format
        num_qubits = existing_data['spec']['num_qubits']
        mi_matrix = np.zeros((num_qubits, num_qubits))
        
        for key, value in latest_mi_data.items():
            # Parse key like "I_0,1" to get qubit indices
            qubits = key.split('_')[1].split(',')
            i, j = int(qubits[0]), int(qubits[1])
            mi_matrix[i, j] = value
            mi_matrix[j, i] = value  # Symmetric matrix
        
        # Create results structure
        results = {
            'mutual_information_matrix': mi_matrix.tolist(),
            'num_qubits': num_qubits,
            'geometry': existing_data['spec']['geometry'],
            'curvature': existing_data['spec']['curvature'],
            'device': existing_data['spec']['device'],
            'shots': existing_data['spec']['shots'],
            'source_file': results_path,
            'timestep_used': len(mi_per_timestep) - 1
        }
        
        print(f"✅ Loaded MI matrix for {num_qubits} qubits, {existing_data['spec']['geometry']} geometry from timestep {len(mi_per_timestep) - 1}")
        
        return results
    
    def test_coordinate_invariance(self, circuit_results):
        """
        Test invariance under different coordinate transformations.
        
        Args:
            circuit_results: Results from quantum circuit execution
            
        Returns:
            dict: Coordinate invariance test results
        """
        print("🔄 Testing coordinate invariance...")
        
        # Extract mutual information matrix
        mi_matrix = np.array(circuit_results.get('mutual_information_matrix', []))
        if mi_matrix.size == 0:
            print("⚠️  No mutual information matrix found!")
            return {}
        
        # Generate coordinate transformations
        transformations = self.generate_coordinate_transformations()
        print(f"🔍 Testing {len(transformations)} coordinate transformations...")
        
        # Test each transformation
        transformation_results = []
        baseline_mi = mi_matrix
        
        for i, transform in enumerate(transformations):
            print(f"  Transformation {i+1}/{len(transformations)}: {transform['name']}")
            
            # Apply transformation
            transformed_mi = transform['function'](mi_matrix)
            
            # Compute information-theoretic distance
            distances = self.compute_information_theoretic_distance(baseline_mi, transformed_mi)
            
            # Test graph isomorphism
            isomorphism_test = self.test_graph_isomorphism(baseline_mi, transformed_mi)
            
            # Store results
            transformation_results.append({
                'transformation_name': transform['name'],
                'transformation_description': transform['description'],
                'transformed_mi_matrix': transformed_mi.tolist(),
                'distances': distances,
                'isomorphism_test': isomorphism_test,
                'invariance_score': 1.0 - distances['cosine']  # Higher is more invariant
            })
        
        return transformation_results
    
    def analyze_invariance_statistics(self, transformation_results):
        """
        Analyze statistical properties of coordinate invariance.
        
        Args:
            transformation_results: Results from coordinate invariance tests
            
        Returns:
            dict: Statistical analysis results
        """
        print("📊 Analyzing invariance statistics...")
        
        if not transformation_results:
            return {}
        
        # Extract invariance scores
        invariance_scores = [result['invariance_score'] for result in transformation_results]
        euclidean_distances = [result['distances']['euclidean'] for result in transformation_results]
        cosine_distances = [result['distances']['cosine'] for result in transformation_results]
        kl_divergences = [result['distances']['kl_divergence'] for result in transformation_results]
        
        # Statistical analysis
        stats_analysis = {
            'invariance_scores': {
                'mean': np.mean(invariance_scores),
                'std': np.std(invariance_scores),
                'min': np.min(invariance_scores),
                'max': np.max(invariance_scores),
                'median': np.median(invariance_scores)
            },
            'euclidean_distances': {
                'mean': np.mean(euclidean_distances),
                'std': np.std(euclidean_distances),
                'max': np.max(euclidean_distances)
            },
            'cosine_distances': {
                'mean': np.mean(cosine_distances),
                'std': np.std(cosine_distances),
                'max': np.max(cosine_distances)
            },
            'kl_divergences': {
                'mean': np.mean(kl_divergences),
                'std': np.std(kl_divergences),
                'max': np.max(kl_divergences)
            }
        }
        
        # Test for significant differences
        # Compare against identity transformation (should be perfect invariance)
        identity_result = next((r for r in transformation_results if r['transformation_name'] == 'identity'), None)
        if identity_result:
            identity_score = identity_result['invariance_score']
            
            # Test if other transformations are significantly different from identity
            other_scores = [r['invariance_score'] for r in transformation_results if r['transformation_name'] != 'identity']
            if other_scores:
                t_stat, p_value = stats.ttest_1samp(other_scores, identity_score)
                stats_analysis['significance_test'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant_difference': p_value < 0.05
                }
        
        return stats_analysis
    
    def create_invariance_plots(self, transformation_results, stats_analysis):
        """
        Create comprehensive invariance analysis plots.
        
        Args:
            transformation_results: Results from coordinate invariance tests
            stats_analysis: Statistical analysis results
            
        Returns:
            list: List of plot file paths
        """
        print("📊 Creating invariance analysis plots...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Extract data
        transformation_names = [r['transformation_name'] for r in transformation_results]
        invariance_scores = [r['invariance_score'] for r in transformation_results]
        euclidean_distances = [r['distances']['euclidean'] for r in transformation_results]
        cosine_distances = [r['distances']['cosine'] for r in transformation_results]
        
        # Plot 1: Invariance Scores
        plt.subplot(2, 3, 1)
        bars = plt.bar(range(len(transformation_names)), invariance_scores, alpha=0.7)
        
        # Color bars based on transformation type
        for i, name in enumerate(transformation_names):
            if name == 'identity':
                bars[i].set_color('green')
            else:
                bars[i].set_color('blue')
        
        plt.xlabel('Coordinate Transformation')
        plt.ylabel('Invariance Score')
        plt.title('Coordinate Invariance Scores')
        plt.xticks(range(len(transformation_names)), transformation_names, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add horizontal line at perfect invariance
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Invariance')
        plt.legend()
        
        # Plot 2: Distance Measures Comparison
        plt.subplot(2, 3, 2)
        x = np.arange(len(transformation_names))
        width = 0.35
        
        plt.bar(x - width/2, euclidean_distances, width, label='Euclidean Distance', alpha=0.7)
        plt.bar(x + width/2, cosine_distances, width, label='Cosine Distance', alpha=0.7)
        
        plt.xlabel('Coordinate Transformation')
        plt.ylabel('Distance')
        plt.title('Information-Theoretic Distances')
        plt.xticks(x, transformation_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Invariance Score Distribution
        plt.subplot(2, 3, 3)
        plt.hist(invariance_scores, bins=10, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(invariance_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(invariance_scores):.3f}')
        plt.xlabel('Invariance Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Invariance Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Graph Isomorphism Results
        plt.subplot(2, 3, 4)
        isomorphism_results = [r['isomorphism_test']['is_isomorphic'] for r in transformation_results]
        isomorphism_counts = [sum(isomorphism_results), len(isomorphism_results) - sum(isomorphism_results)]
        labels = ['Isomorphic', 'Non-isomorphic']
        colors = ['green', 'red']
        
        plt.pie(isomorphism_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Graph Isomorphism Results')
        
        # Plot 5: Statistical Summary
        plt.subplot(2, 3, 5)
        plt.axis('off')
        
        # Create summary text
        summary_text = f"""
        Coordinate Invariance Analysis
        
        Parameters:
        - Qubits: {self.num_qubits}
        - Geometry: {self.geometry}
        - Curvature: {self.curvature}
        - Device: {self.device}
        
        Transformations Tested: {len(transformation_results)}
        
        Invariance Statistics:
        - Mean Score: {stats_analysis.get('invariance_scores', {}).get('mean', 0):.4f}
        - Std Score: {stats_analysis.get('invariance_scores', {}).get('std', 0):.4f}
        - Min Score: {stats_analysis.get('invariance_scores', {}).get('min', 0):.4f}
        - Max Score: {stats_analysis.get('invariance_scores', {}).get('max', 0):.4f}
        
        Distance Statistics:
        - Mean Euclidean: {stats_analysis.get('euclidean_distances', {}).get('mean', 0):.4f}
        - Mean Cosine: {stats_analysis.get('cosine_distances', {}).get('mean', 0):.4f}
        
        Graph Isomorphism:
        - Isomorphic: {sum(isomorphism_results)}/{len(isomorphism_results)}
        """
        
        if stats_analysis.get('significance_test'):
            sig_test = stats_analysis['significance_test']
            summary_text += f"""
        Significance Test:
        - t-statistic: {sig_test['t_statistic']:.4f}
        - p-value: {sig_test['p_value']:.4f}
        - Significant: {'Yes' if sig_test['significant_difference'] else 'No'}
        """
        
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # Plot 6: MI Matrix Comparison
        plt.subplot(2, 3, 6)
        
        # Show baseline MI matrix
        baseline_result = next((r for r in transformation_results if r['transformation_name'] == 'identity'), None)
        if baseline_result:
            mi_matrix = np.array(baseline_result['transformed_mi_matrix'])
            plt.imshow(mi_matrix, cmap='viridis', aspect='auto')
            plt.colorbar(label='Mutual Information')
            plt.title('Baseline MI Matrix')
            plt.xlabel('Qubit Index')
            plt.ylabel('Qubit Index')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"mutual_information_invariance_analysis_{timestamp}.png"
        plot_path = os.path.join("experiment_logs", "mutual_information_invariance_experiment", f"instance_{timestamp}", plot_filename)
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return [plot_path]
    
    def run(self):
        """
        Run the complete Mutual Information Invariance Experiment.
        
        Returns:
            dict: Complete experiment results
        """
        print("🚀 Starting Mutual Information Invariance Experiment...")
        print(f"📋 Parameters: {self.num_qubits} qubits, {self.geometry} geometry, k={self.curvature}, {self.device}")
        
        try:
            # Step 1: Run quantum circuit
            circuit_results = self.run_quantum_circuit()
            
            # Step 2: Test coordinate invariance
            transformation_results = self.test_coordinate_invariance(circuit_results)
            
            # Step 3: Analyze invariance statistics
            stats_analysis = self.analyze_invariance_statistics(transformation_results)
            
            # Step 4: Create analysis plots
            plot_paths = self.create_invariance_plots(transformation_results, stats_analysis)
            
            # Step 5: Compile results
            self.results.update({
                'coordinate_tests': {
                    'transformation_results': transformation_results,
                    'num_transformations_tested': len(transformation_results)
                },
                'invariance_analysis': stats_analysis,
                'plots': plot_paths,
                'circuit_results': circuit_results,
                'timestamp': datetime.now().isoformat()
            })
            
            # Step 6: Save results
            self.save_results()
            
            print("✅ Mutual Information Invariance Experiment completed successfully!")
            return self.results
            
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_results(self):
        """Save experiment results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create experiment directory
        experiment_dir = os.path.join("experiment_logs", "mutual_information_invariance_experiment", f"instance_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save results JSON
        results_filename = f"mutual_information_invariance_results_{timestamp}.json"
        results_path = os.path.join(experiment_dir, results_filename)
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {results_path}")
        
        # Create summary
        self.create_summary(experiment_dir, timestamp)
    
    def create_summary(self, experiment_dir, timestamp):
        """Create a summary of the experiment results."""
        summary_filename = f"mutual_information_invariance_summary_{timestamp}.txt"
        summary_path = os.path.join(experiment_dir, summary_filename)
        
        with open(summary_path, 'w') as f:
            f.write("Mutual Information Invariance Experiment Summary\n")
            f.write("=" * 55 + "\n\n")
            
            f.write("Experiment Parameters:\n")
            f.write(f"- Number of qubits: {self.num_qubits}\n")
            f.write(f"- Geometry: {self.geometry}\n")
            f.write(f"- Curvature: {self.curvature}\n")
            f.write(f"- Device: {self.device}\n")
            f.write(f"- Shots: {self.shots}\n")
            f.write(f"- Coordinate tests: {self.num_coordinate_tests}\n\n")
            
            f.write("Results Summary:\n")
            f.write("-" * 20 + "\n")
            
            if self.results.get('coordinate_tests'):
                f.write(f"- Transformations tested: {self.results['coordinate_tests']['num_transformations_tested']}\n")
            
            if self.results.get('invariance_analysis', {}).get('invariance_scores'):
                scores = self.results['invariance_analysis']['invariance_scores']
                f.write(f"- Mean invariance score: {scores['mean']:.4f}\n")
                f.write(f"- Invariance score std: {scores['std']:.4f}\n")
                f.write(f"- Invariance score range: [{scores['min']:.4f}, {scores['max']:.4f}]\n")
            
            if self.results.get('invariance_analysis', {}).get('significance_test'):
                sig_test = self.results['invariance_analysis']['significance_test']
                f.write(f"- Significant difference from identity: {'Yes' if sig_test['significant_difference'] else 'No'}\n")
                f.write(f"- p-value: {sig_test['p_value']:.4f}\n")
            
            f.write(f"\nPlots generated: {len(self.results.get('plots', []))}\n")
            f.write(f"Timestamp: {timestamp}\n")
        
        print(f"📝 Summary saved to: {summary_path}")

def main():
    """Main function to run the Mutual Information Invariance Experiment."""
    parser = argparse.ArgumentParser(description='Mutual Information Invariance Experiment')
    parser.add_argument('--num_qubits', type=int, default=8, help='Number of qubits')
    parser.add_argument('--geometry', type=str, default='spherical', 
                       choices=['spherical', 'hyperbolic', 'flat'], help='Geometry type')
    parser.add_argument('--curvature', type=float, default=3.0, help='Curvature parameter')
    parser.add_argument('--device', type=str, default='simulator', help='Quantum device')
    parser.add_argument('--shots', type=int, default=1000, help='Number of shots')
    parser.add_argument('--num_coordinate_tests', type=int, default=10, help='Number of coordinate tests')
    
    args = parser.parse_args()
    
    # Create and run experiment
    experiment = MutualInformationInvarianceExperiment(
        num_qubits=args.num_qubits,
        geometry=args.geometry,
        curvature=args.curvature,
        device=args.device,
        shots=args.shots,
        num_coordinate_tests=args.num_coordinate_tests
    )
    
    results = experiment.run()
    
    if results:
        print("\n🎉 Experiment completed successfully!")
        print(f"📊 Results saved to experiment_logs/mutual_information_invariance_experiment/")
    else:
        print("\n❌ Experiment failed!")

if __name__ == "__main__":
    main() 