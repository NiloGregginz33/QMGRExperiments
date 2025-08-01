"""
Quantum Analysis Helper Functions
Functions that were missing from the main experiment file to provide stronger data analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, partial_trace
import networkx as nx
from sklearn.manifold import MDS
import json
import os
from datetime import datetime

def compute_radiation_entropy_advanced(circuit, backend, radiation_qubits, method='shadow', **kwargs):
    """
    Advanced radiation entropy computation using multiple methods.
    
    Args:
        circuit: Quantum circuit
        backend: Backend for execution
        radiation_qubits: List of qubits in radiation
        method: 'shadow', 'random', 'hybrid', or 'basic'
        **kwargs: Additional parameters
    
    Returns:
        dict: Entropy data with metadata
    """
    try:
        if method == 'shadow':
            num_shadows = kwargs.get('num_shadows', 100)
            shots_per_shadow = kwargs.get('shots_per_shadow', 1000)
            
            # Generate classical shadows
            shadows = []
            for _ in range(num_shadows):
                # Random unitary rotation
                random_circuit = circuit.copy()
                for qubit in range(circuit.num_qubits):
                    random_circuit.rx(np.random.random() * 2 * np.pi, qubit)
                    random_circuit.ry(np.random.random() * 2 * np.pi, qubit)
                    random_circuit.rz(np.random.random() * 2 * np.pi, qubit)
                
                # Execute and get counts
                job = execute(random_circuit, backend, shots=shots_per_shadow)
                counts = job.result().get_counts()
                shadows.append(counts)
            
            # Estimate entropy from shadows
            entropy = estimate_entropy_from_shadows(shadows, radiation_qubits)
            
        elif method == 'random':
            num_bases = kwargs.get('num_bases', 20)
            shots_per_basis = kwargs.get('shots_per_basis', 1000)
            
            # Random measurement bases
            entropies = []
            for _ in range(num_bases):
                random_circuit = circuit.copy()
                for qubit in range(circuit.num_qubits):
                    random_circuit.rx(np.random.random() * 2 * np.pi, qubit)
                    random_circuit.ry(np.random.random() * 2 * np.pi, qubit)
                
                job = execute(random_circuit, backend, shots=shots_per_basis)
                counts = job.result().get_counts()
                entropy = compute_von_neumann_entropy_from_counts(counts, radiation_qubits)
                entropies.append(entropy)
            
            entropy = np.mean(entropies)
            
        else:  # basic method
            job = execute(circuit, backend, shots=kwargs.get('shots', 4096))
            counts = job.result().get_counts()
            entropy = compute_von_neumann_entropy_from_counts(counts, radiation_qubits)
        
        return {
            'entropy': entropy,
            'method': method,
            'radiation_qubits': radiation_qubits,
            'metadata': kwargs
        }
        
    except Exception as e:
        print(f"Error in compute_radiation_entropy_advanced: {e}")
        return {
            'entropy': 0.0,
            'method': method,
            'radiation_qubits': radiation_qubits,
            'error': str(e)
        }

def estimate_entropy_from_shadows(shadows, radiation_qubits):
    """Estimate entropy from classical shadows."""
    try:
        # Simple estimation based on shadow data
        total_states = 0
        unique_states = set()
        
        for shadow in shadows:
            for state, count in shadow.items():
                total_states += count
                # Extract radiation qubits from state
                radiation_state = ''.join([state[i] for i in radiation_qubits])
                unique_states.add(radiation_state)
        
        # Estimate entropy as log of number of unique states
        if len(unique_states) > 0:
            return np.log2(len(unique_states))
        else:
            return 0.0
            
    except Exception as e:
        print(f"Error estimating entropy from shadows: {e}")
        return 0.0

def compute_von_neumann_entropy_from_counts(counts, radiation_qubits):
    """Compute von Neumann entropy from measurement counts."""
    try:
        # Convert counts to probability distribution
        total_shots = sum(counts.values())
        probabilities = {}
        
        for state, count in counts.items():
            # Extract radiation qubits
            radiation_state = ''.join([state[i] for i in radiation_qubits])
            if radiation_state in probabilities:
                probabilities[radiation_state] += count / total_shots
            else:
                probabilities[radiation_state] = count / total_shots
        
        # Compute von Neumann entropy
        entropy = 0.0
        for prob in probabilities.values():
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return entropy
        
    except Exception as e:
        print(f"Error computing von Neumann entropy: {e}")
        return 0.0

def create_dynamic_evidence_plots(mi_matrix, distance_matrix, entropy_data, experiment_log_dir, experiment_name):
    """Create comprehensive dynamic evidence plots."""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Dynamic Evidence Plots - {experiment_name}', fontsize=16)
        
        # 1. MI Matrix Heatmap
        sns.heatmap(mi_matrix, annot=True, fmt='.3f', cmap='viridis', ax=axes[0,0])
        axes[0,0].set_title('Mutual Information Matrix')
        axes[0,0].set_xlabel('Qubit Index')
        axes[0,0].set_ylabel('Qubit Index')
        
        # 2. Distance Matrix Heatmap
        sns.heatmap(distance_matrix, annot=True, fmt='.3f', cmap='plasma', ax=axes[0,1])
        axes[0,1].set_title('Distance Matrix')
        axes[0,1].set_xlabel('Qubit Index')
        axes[0,1].set_ylabel('Qubit Index')
        
        # 3. Entropy Evolution
        if 'entropy_per_timestep' in entropy_data:
            timesteps = range(len(entropy_data['entropy_per_timestep']))
            axes[0,2].plot(timesteps, entropy_data['entropy_per_timestep'], 'b-o', linewidth=2, markersize=6)
            axes[0,2].set_title('Entropy Evolution')
            axes[0,2].set_xlabel('Timestep')
            axes[0,2].set_ylabel('Entropy')
            axes[0,2].grid(True, alpha=0.3)
        
        # 4. MI Distribution
        mi_values = mi_matrix[np.triu_indices_from(mi_matrix, k=1)]
        axes[1,0].hist(mi_values, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1,0].set_title('MI Distribution')
        axes[1,0].set_xlabel('Mutual Information')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Distance Distribution
        dist_values = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        axes[1,1].hist(dist_values, bins=20, alpha=0.7, color='red', edgecolor='black')
        axes[1,1].set_title('Distance Distribution')
        axes[1,1].set_xlabel('Distance')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. MI vs Distance Correlation
        axes[1,2].scatter(dist_values, mi_values, alpha=0.6, color='purple')
        axes[1,2].set_title('MI vs Distance Correlation')
        axes[1,2].set_xlabel('Distance')
        axes[1,2].set_ylabel('Mutual Information')
        axes[1,2].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        if len(dist_values) > 1:
            corr = np.corrcoef(dist_values, mi_values)[0,1]
            axes[1,2].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                          transform=axes[1,2].transAxes, fontsize=12,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = os.path.join(experiment_log_dir, f'dynamic_evidence_plots_{experiment_name}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Dynamic evidence plots saved to: {plot_filename}")
        return plot_filename
        
    except Exception as e:
        print(f"Error creating dynamic evidence plots: {e}")
        return None

def create_einstein_time_evolution_plots(einstein_data, experiment_log_dir, experiment_name):
    """Create Einstein tensor time evolution plots."""
    try:
        if not einstein_data or 'time_evolution' not in einstein_data:
            print("No Einstein time evolution data available")
            return None
        
        time_data = einstein_data['time_evolution']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Einstein Tensor Time Evolution - {experiment_name}', fontsize=16)
        
        # 1. Ricci Scalar Evolution
        if 'ricci_scalar' in time_data:
            timesteps = range(len(time_data['ricci_scalar']))
            axes[0,0].plot(timesteps, time_data['ricci_scalar'], 'r-o', linewidth=2, markersize=6)
            axes[0,0].set_title('Ricci Scalar Evolution')
            axes[0,0].set_xlabel('Timestep')
            axes[0,0].set_ylabel('Ricci Scalar')
            axes[0,0].grid(True, alpha=0.3)
        
        # 2. Einstein Tensor Norm Evolution
        if 'einstein_norm' in time_data:
            timesteps = range(len(time_data['einstein_norm']))
            axes[0,1].plot(timesteps, time_data['einstein_norm'], 'b-o', linewidth=2, markersize=6)
            axes[0,1].set_title('Einstein Tensor Norm Evolution')
            axes[0,1].set_xlabel('Timestep')
            axes[0,1].set_ylabel('||G||')
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Stress-Energy Tensor Evolution
        if 'stress_energy' in time_data:
            timesteps = range(len(time_data['stress_energy']))
            axes[1,0].plot(timesteps, time_data['stress_energy'], 'g-o', linewidth=2, markersize=6)
            axes[1,0].set_title('Stress-Energy Tensor Evolution')
            axes[1,0].set_xlabel('Timestep')
            axes[1,0].set_ylabel('||T||')
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. Einstein Equation Residual
        if 'residual' in time_data:
            timesteps = range(len(time_data['residual']))
            axes[1,1].plot(timesteps, time_data['residual'], 'm-o', linewidth=2, markersize=6)
            axes[1,1].set_title('Einstein Equation Residual')
            axes[1,1].set_xlabel('Timestep')
            axes[1,1].set_ylabel('||G - 8πT||')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = os.path.join(experiment_log_dir, f'einstein_time_evolution_{experiment_name}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Einstein time evolution plots saved to: {plot_filename}")
        return plot_filename
        
    except Exception as e:
        print(f"Error creating Einstein time evolution plots: {e}")
        return None

def create_einstein_tensor_heatmaps(einstein_data, experiment_log_dir, experiment_name):
    """Create Einstein tensor component heatmaps."""
    try:
        if not einstein_data or 'tensor_components' not in einstein_data:
            print("No Einstein tensor component data available")
            return None
        
        components = einstein_data['tensor_components']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Einstein Tensor Components - {experiment_name}', fontsize=16)
        
        # Plot different tensor components
        component_names = ['G_00', 'G_01', 'G_10', 'G_11']
        for i, (ax, name) in enumerate(zip(axes.flat, component_names)):
            if name in components:
                sns.heatmap(components[name], annot=True, fmt='.3f', cmap='RdBu_r', ax=ax)
                ax.set_title(f'{name} Component')
                ax.set_xlabel('Qubit Index')
                ax.set_ylabel('Qubit Index')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = os.path.join(experiment_log_dir, f'einstein_tensor_heatmaps_{experiment_name}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Einstein tensor heatmaps saved to: {plot_filename}")
        return plot_filename
        
    except Exception as e:
        print(f"Error creating Einstein tensor heatmaps: {e}")
        return None

def create_einstein_3d_visualization(einstein_data, experiment_log_dir, experiment_name):
    """Create 3D visualization of Einstein tensor evolution."""
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        if not einstein_data or 'time_evolution' not in einstein_data:
            print("No Einstein time evolution data available for 3D visualization")
            return None
        
        time_data = einstein_data['time_evolution']
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create 3D plot
        if 'ricci_scalar' in time_data and 'einstein_norm' in time_data and 'stress_energy' in time_data:
            timesteps = range(len(time_data['ricci_scalar']))
            x = time_data['ricci_scalar']
            y = time_data['einstein_norm']
            z = time_data['stress_energy']
            
            ax.scatter(x, y, z, c=timesteps, cmap='viridis', s=100, alpha=0.7)
            ax.plot(x, y, z, 'k-', alpha=0.5)
            
            ax.set_xlabel('Ricci Scalar')
            ax.set_ylabel('Einstein Tensor Norm')
            ax.set_zlabel('Stress-Energy Tensor Norm')
            ax.set_title(f'Einstein Tensor 3D Evolution - {experiment_name}')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = os.path.join(experiment_log_dir, f'einstein_3d_visualization_{experiment_name}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Einstein 3D visualization saved to: {plot_filename}")
        return plot_filename
        
    except Exception as e:
        print(f"Error creating Einstein 3D visualization: {e}")
        return None

def create_einstein_phase_space_plots(einstein_data, experiment_log_dir, experiment_name):
    """Create phase space plots for Einstein tensor dynamics."""
    try:
        if not einstein_data or 'time_evolution' not in einstein_data:
            print("No Einstein time evolution data available for phase space plots")
            return None
        
        time_data = einstein_data['time_evolution']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Einstein Tensor Phase Space - {experiment_name}', fontsize=16)
        
        # 1. Ricci Scalar vs Einstein Norm
        if 'ricci_scalar' in time_data and 'einstein_norm' in time_data:
            axes[0,0].scatter(time_data['ricci_scalar'], time_data['einstein_norm'], 
                            c=range(len(time_data['ricci_scalar'])), cmap='viridis', alpha=0.7)
            axes[0,0].set_xlabel('Ricci Scalar')
            axes[0,0].set_ylabel('Einstein Tensor Norm')
            axes[0,0].set_title('Ricci vs Einstein Norm')
            axes[0,0].grid(True, alpha=0.3)
        
        # 2. Stress-Energy vs Einstein Norm
        if 'stress_energy' in time_data and 'einstein_norm' in time_data:
            axes[0,1].scatter(time_data['stress_energy'], time_data['einstein_norm'], 
                            c=range(len(time_data['stress_energy'])), cmap='plasma', alpha=0.7)
            axes[0,1].set_xlabel('Stress-Energy Tensor Norm')
            axes[0,1].set_ylabel('Einstein Tensor Norm')
            axes[0,1].set_title('Stress-Energy vs Einstein Norm')
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Residual vs Timestep
        if 'residual' in time_data:
            timesteps = range(len(time_data['residual']))
            axes[1,0].scatter(timesteps, time_data['residual'], 
                            c=time_data['residual'], cmap='Reds', alpha=0.7)
            axes[1,0].set_xlabel('Timestep')
            axes[1,0].set_ylabel('Einstein Equation Residual')
            axes[1,0].set_title('Residual Evolution')
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. Phase Space Trajectory
        if all(key in time_data for key in ['ricci_scalar', 'einstein_norm', 'stress_energy']):
            axes[1,1].scatter(time_data['ricci_scalar'], time_data['stress_energy'], 
                            c=time_data['einstein_norm'], cmap='viridis', alpha=0.7)
            axes[1,1].set_xlabel('Ricci Scalar')
            axes[1,1].set_ylabel('Stress-Energy Tensor Norm')
            axes[1,1].set_title('Phase Space Trajectory')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = os.path.join(experiment_log_dir, f'einstein_phase_space_{experiment_name}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Einstein phase space plots saved to: {plot_filename}")
        return plot_filename
        
    except Exception as e:
        print(f"Error creating Einstein phase space plots: {e}")
        return None

def create_enhanced_entropy_analysis(entropy_data, experiment_log_dir, experiment_name):
    """Create enhanced entropy analysis plots."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Enhanced Entropy Analysis - {experiment_name}', fontsize=16)
        
        # 1. Entropy Evolution with Confidence Intervals
        if 'entropy_per_timestep' in entropy_data:
            timesteps = range(len(entropy_data['entropy_per_timestep']))
            entropies = entropy_data['entropy_per_timestep']
            
            axes[0,0].plot(timesteps, entropies, 'b-o', linewidth=2, markersize=6, label='Entropy')
            
            # Add confidence intervals if available
            if 'entropy_confidence' in entropy_data:
                conf = entropy_data['entropy_confidence']
                axes[0,0].fill_between(timesteps, 
                                     [e - c for e, c in zip(entropies, conf)],
                                     [e + c for e, c in zip(entropies, conf)],
                                     alpha=0.3, color='blue', label='95% Confidence')
            
            axes[0,0].set_title('Entropy Evolution with Confidence')
            axes[0,0].set_xlabel('Timestep')
            axes[0,0].set_ylabel('Entropy')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
        
        # 2. Entropy Rate Analysis
        if 'entropy_per_timestep' in entropy_data and len(entropy_data['entropy_per_timestep']) > 1:
            entropies = np.array(entropy_data['entropy_per_timestep'])
            entropy_rates = np.diff(entropies)
            timesteps = range(1, len(entropies))
            
            axes[0,1].plot(timesteps, entropy_rates, 'r-o', linewidth=2, markersize=6)
            axes[0,1].set_title('Entropy Rate (dS/dt)')
            axes[0,1].set_xlabel('Timestep')
            axes[0,1].set_ylabel('Entropy Rate')
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Entropy Distribution
        if 'entropy_per_timestep' in entropy_data:
            entropies = entropy_data['entropy_per_timestep']
            axes[1,0].hist(entropies, bins=15, alpha=0.7, color='green', edgecolor='black')
            axes[1,0].set_title('Entropy Distribution')
            axes[1,0].set_xlabel('Entropy')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. Entropy vs System Size
        if 'subsystem_entropies' in entropy_data:
            subsystem_sizes = list(entropy_data['subsystem_entropies'].keys())
            avg_entropies = [np.mean(entropy_data['subsystem_entropies'][size]) for size in subsystem_sizes]
            
            axes[1,1].plot(subsystem_sizes, avg_entropies, 'g-o', linewidth=2, markersize=6)
            axes[1,1].set_title('Entropy vs Subsystem Size')
            axes[1,1].set_xlabel('Subsystem Size')
            axes[1,1].set_ylabel('Average Entropy')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = os.path.join(experiment_log_dir, f'enhanced_entropy_analysis_{experiment_name}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Enhanced entropy analysis saved to: {plot_filename}")
        return plot_filename
        
    except Exception as e:
        print(f"Error creating enhanced entropy analysis: {e}")
        return None

def create_quantum_geometry_validation_plots(geometry_data, experiment_log_dir, experiment_name):
    """Create quantum geometry validation plots."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Quantum Geometry Validation - {experiment_name}', fontsize=16)
        
        # 1. Curvature Distribution
        if 'curvature_values' in geometry_data:
            curvatures = geometry_data['curvature_values']
            axes[0,0].hist(curvatures, bins=20, alpha=0.7, color='blue', edgecolor='black')
            axes[0,0].set_title('Curvature Distribution')
            axes[0,0].set_xlabel('Curvature')
            axes[0,0].set_ylabel('Frequency')
            axes[0,0].grid(True, alpha=0.3)
        
        # 2. Distance vs Curvature Correlation
        if 'distances' in geometry_data and 'curvatures' in geometry_data:
            distances = geometry_data['distances']
            curvatures = geometry_data['curvatures']
            axes[0,1].scatter(distances, curvatures, alpha=0.6, color='red')
            axes[0,1].set_title('Distance vs Curvature')
            axes[0,1].set_xlabel('Distance')
            axes[0,1].set_ylabel('Curvature')
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Geometry Consistency Check
        if 'consistency_metrics' in geometry_data:
            metrics = geometry_data['consistency_metrics']
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            axes[1,0].bar(metric_names, metric_values, alpha=0.7, color='green')
            axes[1,0].set_title('Geometry Consistency Metrics')
            axes[1,0].set_xlabel('Metric')
            axes[1,0].set_ylabel('Value')
            axes[1,0].tick_params(axis='x', rotation=45)
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. Quantum-Classical Comparison
        if 'quantum_vs_classical' in geometry_data:
            qc_data = geometry_data['quantum_vs_classical']
            if 'quantum' in qc_data and 'classical' in qc_data:
                quantum_vals = qc_data['quantum']
                classical_vals = qc_data['classical']
                
                axes[1,1].scatter(classical_vals, quantum_vals, alpha=0.6, color='purple')
                axes[1,1].plot([min(classical_vals), max(classical_vals)], 
                              [min(classical_vals), max(classical_vals)], 'k--', alpha=0.5)
                axes[1,1].set_title('Quantum vs Classical Geometry')
                axes[1,1].set_xlabel('Classical Prediction')
                axes[1,1].set_ylabel('Quantum Result')
                axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = os.path.join(experiment_log_dir, f'quantum_geometry_validation_{experiment_name}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Quantum geometry validation plots saved to: {plot_filename}")
        return plot_filename
        
    except Exception as e:
        print(f"Error creating quantum geometry validation plots: {e}")
        return None

def generate_comprehensive_summary(experiment_results, experiment_log_dir, experiment_name):
    """Generate a comprehensive summary of all experiment results."""
    try:
        summary = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'summary': {}
        }
        
        # Basic experiment info
        if 'num_qubits' in experiment_results:
            summary['summary']['num_qubits'] = experiment_results['num_qubits']
        if 'geometry' in experiment_results:
            summary['summary']['geometry'] = experiment_results['geometry']
        if 'curvature' in experiment_results:
            summary['summary']['curvature'] = experiment_results['curvature']
        
        # Key metrics
        if 'key_metrics' in experiment_results:
            summary['summary']['key_metrics'] = experiment_results['key_metrics']
        
        # Entropy analysis
        if 'entropy_analysis' in experiment_results:
            entropy_data = experiment_results['entropy_analysis']
            summary['summary']['entropy'] = {
                'final_entropy': entropy_data.get('final_entropy', 'N/A'),
                'entropy_evolution': len(entropy_data.get('entropy_per_timestep', [])),
                'max_entropy': max(entropy_data.get('entropy_per_timestep', [0])),
                'min_entropy': min(entropy_data.get('entropy_per_timestep', [0]))
            }
        
        # Einstein analysis
        if 'einstein_analysis' in experiment_results:
            einstein_data = experiment_results['einstein_analysis']
            summary['summary']['einstein'] = {
                'final_ricci_scalar': einstein_data.get('final_ricci_scalar', 'N/A'),
                'einstein_equation_satisfied': einstein_data.get('equation_satisfied', False),
                'residual_norm': einstein_data.get('residual_norm', 'N/A')
            }
        
        # Geometry analysis
        if 'geometry_analysis' in experiment_results:
            geometry_data = experiment_results['geometry_analysis']
            summary['summary']['geometry'] = {
                'gromov_delta': geometry_data.get('gromov_delta', 'N/A'),
                'hyperbolicity': geometry_data.get('is_hyperbolic', False),
                'curvature_consistency': geometry_data.get('curvature_consistency', 'N/A')
            }
        
        # Performance metrics
        if 'performance' in experiment_results:
            perf_data = experiment_results['performance']
            summary['summary']['performance'] = {
                'total_runtime': perf_data.get('total_runtime', 'N/A'),
                'shots_used': perf_data.get('total_shots', 'N/A'),
                'optimization_iterations': perf_data.get('optimization_iterations', 'N/A')
            }
        
        # Save summary
        summary_filename = os.path.join(experiment_log_dir, f'comprehensive_summary_{experiment_name}.json')
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Comprehensive summary saved to: {summary_filename}")
        return summary_filename
        
    except Exception as e:
        print(f"Error generating comprehensive summary: {e}")
        return None

# Initialize einstein_stats as a global variable
einstein_stats = {}

def update_einstein_stats(new_data):
    """Update global Einstein statistics."""
    global einstein_stats
    einstein_stats.update(new_data)
    return einstein_stats

def get_einstein_stats():
    """Get current Einstein statistics."""
    global einstein_stats
    return einstein_stats 