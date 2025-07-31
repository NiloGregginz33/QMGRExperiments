#!/usr/bin/env python3
"""
Enhanced Quantum Geometry Analysis with Page Curve Test
======================================================

A comprehensive quantum geometry analysis tool that implements:
1. Area law fitting using refined entropy bins and geodesic cutoff
2. Mutual information matrix computation and entanglement wedge embedding
3. Asymmetry injection (T-gate, reversed layers) and metric eigenvalue analysis
4. Time evolution tracking: Δ-entropy(t) vs Δ-entropy(-t) for time asymmetry
5. Comprehensive quantum structure validation and scoring
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
from scipy import stats, optimize
from sklearn.metrics import r2_score
from sklearn.manifold import MDS
import itertools
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.append('src')

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, Operator

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class EnhancedQuantumGeometryAnalyzer:
    def __init__(self, num_qubits: int = 9, entanglement_strength: float = 3.0, 
                 timesteps: int = 12, shots: int = 10000, asymmetry_strength: float = 1.0):
        self.num_qubits = num_qubits
        self.entanglement_strength = entanglement_strength
        self.timesteps = timesteps
        self.shots = shots
        self.asymmetry_strength = asymmetry_strength
        self.results = {}
        self.time_evolution_data = []

    def create_enhanced_circuit_with_asymmetry(self) -> QuantumCircuit:
        """Create a circuit with enhanced entanglement and time asymmetry."""
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        
        # Initialize in a superposition state
        for i in range(self.num_qubits):
            qc.h(i)
        
        # Track time evolution
        self.time_evolution_data = []
        
        # Apply multiple layers with asymmetry
        for step in range(self.timesteps):
            # Store intermediate state for time evolution analysis
            if step > 0:
                try:
                    intermediate_sv = Statevector.from_instruction(qc)
                    self.time_evolution_data.append({
                        'step': step,
                        'statevector': intermediate_sv.data.copy()
                    })
                except:
                    pass
            
            # Layer 1: Nearest neighbor entanglement (enhanced)
            for i in range(self.num_qubits - 1):
                qc.rzz(self.entanglement_strength * 1.0, i, i+1)
                qc.ryy(self.entanglement_strength * 0.7, i, i+1)
                qc.rxx(self.entanglement_strength * 0.5, i, i+1)
            
            # Layer 2: Long-range entanglement (crucial for Page curve)
            for i in range(self.num_qubits):
                for j in range(i+2, self.num_qubits):
                    distance = abs(i - j)
                    strength = self.entanglement_strength * np.exp(-distance / (self.num_qubits / 4))
                    if strength > 0.05:
                        qc.rzz(strength, i, j)
                        qc.ryy(strength * 0.8, i, j)
                        qc.rxx(strength * 0.6, i, j)
            
            # Layer 3: Asymmetry injection (T-gates for time-reversal breaking)
            if step % 2 == 0:  # Every other step
                for i in range(self.num_qubits):
                    qc.t(i)  # T-gate breaks time-reversal symmetry
                    qc.rz(self.asymmetry_strength * np.pi/4, i)  # Additional phase
            
            # Layer 4: Reversed layers for asymmetry (every 3rd step)
            if step % 3 == 0:
                # Apply gates in reverse order for this step
                for i in range(self.num_qubits - 1, -1, -1):
                    qc.rx(self.asymmetry_strength * np.pi/6, i)
                    qc.ry(self.asymmetry_strength * np.pi/6, i)
                    qc.rz(self.asymmetry_strength * np.pi/6, i)
            
            # Layer 5: All-to-all entanglement with distance-dependent strength
            for i in range(self.num_qubits):
                for j in range(i+1, self.num_qubits):
                    if (i + j) % 2 == 0:
                        qc.rzz(self.entanglement_strength * 1.2, i, j)
                        qc.ryy(self.entanglement_strength * 0.9, i, j)
                        qc.rxx(self.entanglement_strength * 0.6, i, j)
            
            # Layer 6: Non-local operations (simulating quantum gravity effects)
            if step % 2 == 0:
                for i in range(0, self.num_qubits, 2):
                    if i + 1 < self.num_qubits:
                        qc.cx(i, i+1)
                        qc.h(i)
                        qc.h(i+1)
                        qc.cz(i, i+1)
            
            # Layer 7: State purification and mixing (every 4th step)
            if step % 4 == 0:
                for i in range(self.num_qubits):
                    qc.rx(np.random.uniform(0, 2*np.pi), i)
                    qc.ry(np.random.uniform(0, 2*np.pi), i)
                    qc.rz(np.random.uniform(0, 2*np.pi), i)
                
                for i in range(self.num_qubits):
                    for j in range(i+1, self.num_qubits):
                        qc.rzz(self.entanglement_strength * 0.3, i, j)
        
        return qc

    def compute_subsystem_entropy(self, statevector: np.ndarray, subsystem_qubits: List[int]) -> float:
        """Compute von Neumann entropy of a subsystem using manual calculation."""
        try:
            sv = Statevector(statevector)
            all_qubits = list(range(self.num_qubits))
            complement_qubits = [q for q in all_qubits if q not in subsystem_qubits]
            
            if complement_qubits:
                reduced_state = partial_trace(sv, complement_qubits)
            else:
                reduced_state = sv
            
            if hasattr(reduced_state, 'data'):
                rho = reduced_state.data
            else:
                rho = np.array(reduced_state)
            
            if rho.ndim == 1:
                rho = np.outer(rho, rho.conj())
            
            eigenvalues = np.linalg.eigvalsh(rho)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            
            entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
            return float(entropy)
                
        except Exception as e:
            print(f"Error computing subsystem entropy: {e}")
            return 0.0

    def compute_mutual_information_matrix(self, statevector: np.ndarray) -> np.ndarray:
        """Compute mutual information matrix between all qubit pairs."""
        print("🔍 Computing mutual information matrix...")
        mi_matrix = np.zeros((self.num_qubits, self.num_qubits))
        
        for i in range(self.num_qubits):
            for j in range(i+1, self.num_qubits):
                # Compute entropy of individual qubits
                s_i = self.compute_subsystem_entropy(statevector, [i])
                s_j = self.compute_subsystem_entropy(statevector, [j])
                
                # Compute entropy of combined subsystem
                s_ij = self.compute_subsystem_entropy(statevector, [i, j])
                
                # Mutual information: I(A:B) = S(A) + S(B) - S(AB)
                mi = s_i + s_j - s_ij
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi  # Symmetric
        
        return mi_matrix

    def fit_area_law_with_refined_bins(self, entropies: List[float], subsystem_sizes: List[int]) -> Dict:
        """Fit area law using refined entropy bins and geodesic cutoff."""
        print("🔍 Fitting area law with refined bins...")
        
        # Create refined entropy bins
        unique_sizes = sorted(set(subsystem_sizes))
        binned_data = {}
        
        for size in unique_sizes:
            size_entropies = [entropies[i] for i, s in enumerate(subsystem_sizes) if s == size]
            binned_data[size] = {
                'mean': np.mean(size_entropies),
                'std': np.std(size_entropies),
                'count': len(size_entropies)
            }
        
        # Fit area law: S(A) ∝ |∂A|^α
        sizes = list(binned_data.keys())
        mean_entropies = [binned_data[s]['mean'] for s in sizes]
        
        # Area law model: S = a * (size)^α
        def area_law_model(x, a, alpha):
            return a * np.power(x, alpha)
        
        try:
            popt, pcov = optimize.curve_fit(area_law_model, sizes, mean_entropies, 
                                         p0=[0.5, 1.0], maxfev=10000)
            area_law_pred = area_law_model(sizes, *popt)
            area_law_r2 = r2_score(mean_entropies, area_law_pred)
            
            # Geodesic cutoff analysis
            geodesic_cutoff = self.analyze_geodesic_cutoff(sizes, mean_entropies)
            
            return {
                'area_law_params': {'a': popt[0], 'alpha': popt[1]},
                'area_law_r2': area_law_r2,
                'geodesic_cutoff': geodesic_cutoff,
                'binned_data': binned_data,
                'predictions': area_law_pred.tolist()
            }
        except:
            return {
                'area_law_params': {'a': 0.0, 'alpha': 0.0},
                'area_law_r2': 0.0,
                'geodesic_cutoff': 0.0,
                'binned_data': binned_data,
                'predictions': [0.0] * len(sizes)
            }

    def analyze_geodesic_cutoff(self, sizes: List[int], entropies: List[float]) -> float:
        """Analyze geodesic cutoff for boundary effects."""
        if len(sizes) < 3:
            return 0.0
        
        # Find where entropy growth changes slope (potential cutoff)
        slopes = []
        for i in range(1, len(sizes)):
            slope = (entropies[i] - entropies[i-1]) / (sizes[i] - sizes[i-1])
            slopes.append(slope)
        
        # Find the point where slope changes significantly
        if len(slopes) > 1:
            slope_changes = [abs(slopes[i] - slopes[i-1]) for i in range(1, len(slopes))]
            max_change_idx = np.argmax(slope_changes)
            cutoff_size = sizes[max_change_idx + 1]
            return float(cutoff_size)
        
        return 0.0

    def analyze_time_asymmetry(self) -> Dict:
        """Analyze time asymmetry: Δ-entropy(t) vs Δ-entropy(-t)."""
        print("🔍 Analyzing time asymmetry...")
        
        if len(self.time_evolution_data) < 2:
            return {'time_asymmetry_detected': False, 'asymmetry_score': 0.0}
        
        # Compute entropy evolution
        entropy_evolution = []
        for data in self.time_evolution_data:
            # Compute total system entropy at each step
            total_entropy = self.compute_subsystem_entropy(data['statevector'], list(range(self.num_qubits)))
            entropy_evolution.append({
                'step': data['step'],
                'entropy': total_entropy
            })
        
        # Compute Δ-entropy for forward and reverse evolution
        forward_deltas = []
        reverse_deltas = []
        
        for i in range(1, len(entropy_evolution)):
            # Forward: ΔS(t) = S(t) - S(t-1)
            forward_delta = entropy_evolution[i]['entropy'] - entropy_evolution[i-1]['entropy']
            forward_deltas.append(forward_delta)
            
            # Reverse: ΔS(-t) = S(t-1) - S(t) (time-reversed)
            reverse_delta = entropy_evolution[i-1]['entropy'] - entropy_evolution[i]['entropy']
            reverse_deltas.append(reverse_delta)
        
        # Compute asymmetry score
        if forward_deltas and reverse_deltas:
            asymmetry_score = np.mean(np.abs(np.array(forward_deltas) - np.array(reverse_deltas)))
            time_asymmetry_detected = asymmetry_score > 0.1  # Threshold
        else:
            asymmetry_score = 0.0
            time_asymmetry_detected = False
        
        return {
            'time_asymmetry_detected': time_asymmetry_detected,
            'asymmetry_score': float(asymmetry_score),
            'forward_deltas': forward_deltas,
            'reverse_deltas': reverse_deltas,
            'entropy_evolution': entropy_evolution
        }

    def analyze_metric_eigenvalues(self, mi_matrix: np.ndarray) -> Dict:
        """Analyze metric eigenvalues for geometric structure."""
        print("🔍 Analyzing metric eigenvalues...")
        
        # Use MI matrix as a metric tensor
        metric_tensor = mi_matrix + np.eye(self.num_qubits) * 1e-6  # Add small diagonal for stability
        
        try:
            eigenvalues = np.linalg.eigvalsh(metric_tensor)
            eigenvectors = np.linalg.eigh(metric_tensor)[1]
            
            # Analyze eigenvalue spectrum
            positive_eigenvalues = eigenvalues[eigenvalues > 1e-10]
            negative_eigenvalues = eigenvalues[eigenvalues < -1e-10]
            
            # Compute geometric invariants
            determinant = np.prod(positive_eigenvalues) if len(positive_eigenvalues) > 0 else 1.0
            trace = np.sum(eigenvalues)
            condition_number = np.max(np.abs(eigenvalues)) / (np.min(np.abs(eigenvalues)) + 1e-10)
            
            # Check for Lorentzian signature (negative eigenvalues)
            lorentzian_signature = len(negative_eigenvalues) > 0
            
            return {
                'eigenvalues': eigenvalues.tolist(),
                'eigenvectors': eigenvectors.tolist(),
                'determinant': float(determinant),
                'trace': float(trace),
                'condition_number': float(condition_number),
                'lorentzian_signature': lorentzian_signature,
                'num_negative_eigenvalues': len(negative_eigenvalues),
                'num_positive_eigenvalues': len(positive_eigenvalues)
            }
        except Exception as e:
            print(f"Error in metric eigenvalue analysis: {e}")
            return {
                'eigenvalues': [],
                'eigenvectors': [],
                'determinant': 0.0,
                'trace': 0.0,
                'condition_number': 0.0,
                'lorentzian_signature': False,
                'num_negative_eigenvalues': 0,
                'num_positive_eigenvalues': 0
            }

    def compute_quantum_geometry_score(self, analysis_results: Dict) -> float:
        """Compute comprehensive quantum geometry score."""
        score_components = []
        
        # 1. Page curve score (0-1)
        page_curve_r2 = analysis_results.get('page_curve_r2', 0.0)
        score_components.append(min(page_curve_r2, 1.0))
        
        # 2. Area law score (0-1)
        area_law_r2 = analysis_results.get('area_law_r2', 0.0)
        score_components.append(min(area_law_r2, 1.0))
        
        # 3. Time asymmetry score (0-1)
        asymmetry_score = analysis_results.get('asymmetry_score', 0.0)
        score_components.append(min(asymmetry_score * 10, 1.0))  # Scale up
        
        # 4. Metric eigenvalue score (0-1)
        lorentzian = analysis_results.get('lorentzian_signature', False)
        condition_number = analysis_results.get('condition_number', 0.0)
        metric_score = 0.5 if lorentzian else 0.0
        if condition_number > 1.0:
            metric_score += 0.5 * min(condition_number / 10, 1.0)
        score_components.append(metric_score)
        
        # 5. Mutual information locality score (0-1)
        mi_locality = analysis_results.get('mi_locality_score', 0.0)
        score_components.append(min(mi_locality, 1.0))
        
        # Average all components
        final_score = np.mean(score_components)
        return float(final_score)

    def analyze_page_curve(self, statevector: np.ndarray) -> Dict:
        """Analyze the Page curve behavior of the quantum state."""
        print("🔬 Analyzing Page curve behavior...")
        
        all_qubits = list(range(self.num_qubits))
        bipartitions = []
        entropies = []
        
        max_size = self.num_qubits // 2
        
        for size in range(1, max_size + 1):
            for subset in itertools.combinations(all_qubits, size):
                bipartitions.append(list(subset))
                entropy = self.compute_subsystem_entropy(statevector, list(subset))
                entropies.append(entropy)
                print(f"  Subsystem {subset}: S = {entropy:.4f}")
        
        full_entropy = self.compute_subsystem_entropy(statevector, all_qubits)
        bipartitions.append(all_qubits)
        entropies.append(full_entropy)
        
        subsystem_sizes = [len(b) for b in bipartitions]
        
        # Fit models
        models = {}
        
        # Linear model
        try:
            coeffs = np.polyfit(subsystem_sizes, entropies, 1)
            linear_pred = np.polyval(coeffs, subsystem_sizes)
            linear_r2 = r2_score(entropies, linear_pred)
            models['linear'] = {
                'coefficients': coeffs.tolist(),
                'r_squared': linear_r2,
                'predictions': linear_pred.tolist()
            }
        except:
            models['linear'] = {'r_squared': 0.0}
        
        # Page curve model
        try:
            def page_curve_model(x, a, b, c):
                return a * x * (max(subsystem_sizes) - x) + b * x + c
            
            popt, _ = optimize.curve_fit(page_curve_model, subsystem_sizes, entropies, 
                                       p0=[0.1, 0.5, 0.0], maxfev=10000)
            page_pred = page_curve_model(subsystem_sizes, *popt)
            page_r2 = r2_score(entropies, page_pred)
            models['page_curve'] = {
                'parameters': popt.tolist(),
                'r_squared': page_r2,
                'predictions': page_pred.tolist()
            }
        except:
            models['page_curve'] = {'r_squared': 0.0}
        
        page_curve_detected = (
            models['page_curve']['r_squared'] > models['linear']['r_squared'] and
            models['page_curve']['r_squared'] > 0.8
        )
        
        return {
            'subsystem_sizes': subsystem_sizes,
            'entropies': entropies,
            'bipartitions': bipartitions,
            'models': models,
            'page_curve_detected': page_curve_detected,
            'linear_r2': models['linear']['r_squared'],
            'page_curve_r2': models['page_curve']['r_squared']
        }

    def run_enhanced_experiment(self) -> Dict:
        """Run the enhanced quantum geometry experiment."""
        print(f"🚀 Starting Enhanced Quantum Geometry Analysis")
        print(f"📊 Parameters: {self.num_qubits} qubits, strength={self.entanglement_strength}, timesteps={self.timesteps}")
        
        try:
            # Create the circuit
            qc = self.create_enhanced_circuit_with_asymmetry()
            print(f"✅ Circuit created with depth {qc.depth()}")
            
            # Get the final statevector
            try:
                statevector = Statevector.from_instruction(qc)
                statevector = statevector.data
                print(f"✅ Statevector obtained: shape {statevector.shape}")
            except Exception as e:
                print(f"❌ Error getting statevector: {e}")
                return {}
            
            # 1. Page curve analysis
            page_curve_analysis = self.analyze_page_curve(statevector)
            
            # 2. Area law fitting with refined bins
            area_law_analysis = self.fit_area_law_with_refined_bins(
                page_curve_analysis['entropies'], 
                page_curve_analysis['subsystem_sizes']
            )
            
            # 3. Mutual information matrix
            mi_matrix = self.compute_mutual_information_matrix(statevector)
            
            # 4. Time asymmetry analysis
            time_asymmetry_analysis = self.analyze_time_asymmetry()
            
            # 5. Metric eigenvalue analysis
            metric_analysis = self.analyze_metric_eigenvalues(mi_matrix)
            
            # 6. Compute MI locality score
            mi_locality_score = self.compute_mi_locality_score(mi_matrix)
            
            # 7. Compute overall quantum geometry score
            analysis_results = {
                'page_curve_r2': page_curve_analysis['page_curve_r2'],
                'area_law_r2': area_law_analysis['area_law_r2'],
                'asymmetry_score': time_asymmetry_analysis['asymmetry_score'],
                'mi_locality_score': mi_locality_score
            }
            quantum_geometry_score = self.compute_quantum_geometry_score(analysis_results)
            
            # Store comprehensive results
            self.results = {
                'num_qubits': self.num_qubits,
                'entanglement_strength': self.entanglement_strength,
                'timesteps': self.timesteps,
                'asymmetry_strength': self.asymmetry_strength,
                'circuit_depth': qc.depth(),
                'page_curve_analysis': page_curve_analysis,
                'area_law_analysis': area_law_analysis,
                'mutual_information_matrix': mi_matrix.tolist(),
                'time_asymmetry_analysis': time_asymmetry_analysis,
                'metric_analysis': metric_analysis,
                'quantum_geometry_score': quantum_geometry_score,
                'statevector_shape': statevector.shape,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            print(f"✅ Enhanced experiment completed successfully!")
            print(f"🎯 Quantum Geometry Score: {quantum_geometry_score:.4f}")
            return self.results
            
        except Exception as e:
            print(f"❌ Error during experiment: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def compute_mi_locality_score(self, mi_matrix: np.ndarray) -> float:
        """Compute locality score from mutual information matrix."""
        # Compute average MI for nearest neighbors vs long-range
        nearest_neighbor_mi = []
        long_range_mi = []
        
        for i in range(self.num_qubits):
            for j in range(i+1, self.num_qubits):
                distance = abs(i - j)
                mi_value = mi_matrix[i, j]
                
                if distance == 1:  # Nearest neighbors
                    nearest_neighbor_mi.append(mi_value)
                elif distance > 2:  # Long-range
                    long_range_mi.append(mi_value)
        
        if nearest_neighbor_mi and long_range_mi:
            avg_nn = np.mean(nearest_neighbor_mi)
            avg_lr = np.mean(long_range_mi)
            locality_score = avg_nn / (avg_lr + 1e-10)
            return float(locality_score)
        
        return 0.0

    def create_comprehensive_visualization(self, output_dir: str = None):
        """Create comprehensive visualization plots."""
        if not self.results:
            print("❌ No results to visualize")
            return
        
        if output_dir is None:
            output_dir = "."
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a comprehensive dashboard
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Page curve analysis
        ax1 = fig.add_subplot(gs[0, 0])
        page_analysis = self.results['page_curve_analysis']
        subsystem_sizes = page_analysis['subsystem_sizes']
        entropies = page_analysis['entropies']
        
        ax1.scatter(subsystem_sizes, entropies, color='blue', alpha=0.7, s=100, label='Data')
        if 'page_curve' in page_analysis['models'] and 'predictions' in page_analysis['models']['page_curve']:
            ax1.plot(subsystem_sizes, page_analysis['models']['page_curve']['predictions'], 
                    'g-', linewidth=2, label=f'Page Curve (R²={page_analysis["page_curve_r2"]:.3f})')
        ax1.set_xlabel('Subsystem Size')
        ax1.set_ylabel('Von Neumann Entropy')
        ax1.set_title('Page Curve Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Area law analysis
        ax2 = fig.add_subplot(gs[0, 1])
        area_analysis = self.results['area_law_analysis']
        binned_data = area_analysis['binned_data']
        sizes = list(binned_data.keys())
        means = [binned_data[s]['mean'] for s in sizes]
        stds = [binned_data[s]['std'] for s in sizes]
        
        ax2.errorbar(sizes, means, yerr=stds, fmt='o-', color='red', capsize=5, label='Binned Data')
        if 'predictions' in area_analysis:
            ax2.plot(sizes, area_analysis['predictions'], 'b--', linewidth=2, 
                    label=f'Area Law (R²={area_analysis["area_law_r2"]:.3f})')
        ax2.set_xlabel('Subsystem Size')
        ax2.set_ylabel('Average Entropy')
        ax2.set_title('Area Law Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Mutual information heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        mi_matrix = np.array(self.results['mutual_information_matrix'])
        im = ax3.imshow(mi_matrix, cmap='viridis', aspect='auto')
        ax3.set_title('Mutual Information Matrix')
        ax3.set_xlabel('Qubit Index')
        ax3.set_ylabel('Qubit Index')
        plt.colorbar(im, ax=ax3)
        
        # 4. Time asymmetry analysis
        ax4 = fig.add_subplot(gs[1, 0])
        time_analysis = self.results['time_asymmetry_analysis']
        if 'entropy_evolution' in time_analysis:
            steps = [e['step'] for e in time_analysis['entropy_evolution']]
            entropies = [e['entropy'] for e in time_analysis['entropy_evolution']]
            ax4.plot(steps, entropies, 'o-', color='purple', linewidth=2)
            ax4.set_xlabel('Circuit Step')
            ax4.set_ylabel('Total System Entropy')
            ax4.set_title(f'Time Evolution (Asymmetry: {time_analysis["asymmetry_score"]:.4f})')
            ax4.grid(True, alpha=0.3)
        
        # 5. Metric eigenvalues
        ax5 = fig.add_subplot(gs[1, 1])
        metric_analysis = self.results['metric_analysis']
        if 'eigenvalues' in metric_analysis:
            eigenvalues = metric_analysis['eigenvalues']
            ax5.bar(range(len(eigenvalues)), eigenvalues, color='orange', alpha=0.7)
            ax5.set_xlabel('Eigenvalue Index')
            ax5.set_ylabel('Eigenvalue')
            ax5.set_title('Metric Eigenvalues')
            ax5.grid(True, alpha=0.3)
        
        # 6. Time asymmetry deltas
        ax6 = fig.add_subplot(gs[1, 2])
        if 'forward_deltas' in time_analysis and 'reverse_deltas' in time_analysis:
            forward = time_analysis['forward_deltas']
            reverse = time_analysis['reverse_deltas']
            x = range(len(forward))
            ax6.plot(x, forward, 'o-', color='green', label='Forward ΔS(t)', linewidth=2)
            ax6.plot(x, reverse, 's-', color='red', label='Reverse ΔS(-t)', linewidth=2)
            ax6.set_xlabel('Time Step')
            ax6.set_ylabel('Δ Entropy')
            ax6.set_title('Time Asymmetry Analysis')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Quantum geometry score breakdown
        ax7 = fig.add_subplot(gs[2, :])
        score = self.results['quantum_geometry_score']
        components = ['Page Curve', 'Area Law', 'Time Asymmetry', 'Metric Structure', 'MI Locality']
        values = [
            page_analysis['page_curve_r2'],
            area_analysis['area_law_r2'],
            min(time_analysis['asymmetry_score'] * 10, 1.0),
            metric_analysis.get('condition_number', 0.0) / 10,
            self.results.get('mi_locality_score', 0.0)
        ]
        
        bars = ax7.bar(components, values, color=['blue', 'red', 'purple', 'orange', 'green'], alpha=0.7)
        ax7.set_ylabel('Score Component')
        ax7.set_title(f'Quantum Geometry Score Breakdown (Overall: {score:.4f})')
        ax7.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 8. Summary text
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')
        
        summary_text = f"""
        ENHANCED QUANTUM GEOMETRY ANALYSIS SUMMARY
        ==========================================
        
        🎯 Overall Quantum Geometry Score: {score:.4f}
        
        📊 Key Metrics:
        • Page Curve R²: {page_analysis['page_curve_r2']:.4f}
        • Area Law R²: {area_analysis['area_law_r2']:.4f}
        • Time Asymmetry Score: {time_analysis['asymmetry_score']:.4f}
        • Metric Condition Number: {metric_analysis.get('condition_number', 0.0):.4f}
        • MI Locality Score: {self.results.get('mi_locality_score', 0.0):.4f}
        
        🔍 Key Findings:
        • Page Curve Detected: {'YES' if page_analysis['page_curve_detected'] else 'NO'}
        • Lorentzian Signature: {'YES' if metric_analysis.get('lorentzian_signature', False) else 'NO'}
        • Time Asymmetry Detected: {'YES' if time_analysis['time_asymmetry_detected'] else 'NO'}
        • Geodesic Cutoff: {area_analysis.get('geodesic_cutoff', 0.0):.1f}
        
        🎉 Assessment: {'STRONG EVIDENCE OF QUANTUM GEOMETRY' if score > 0.7 else 'MODERATE EVIDENCE' if score > 0.4 else 'WEAK EVIDENCE'}
        """
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        # Save the comprehensive dashboard
        dashboard_file = os.path.join(output_dir, f"enhanced_quantum_geometry_dashboard_{self.num_qubits}q.png")
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        print(f"📈 Comprehensive dashboard saved: {dashboard_file}")
        
        plt.show()

    def save_enhanced_results(self, output_dir: str = None):
        """Save the enhanced experiment results."""
        if not self.results:
            print("❌ No results to save")
            return
        
        if output_dir is None:
            output_dir = "."
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert complex numbers to real for JSON serialization
        def convert_complex(obj):
            if isinstance(obj, complex):
                return float(obj.real)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, bool):
                return obj  # Keep boolean values as-is
            elif isinstance(obj, dict):
                return {k: convert_complex(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_complex(item) for item in obj]
            elif isinstance(obj, (int, float, str)):
                return obj
            else:
                return str(obj)  # Convert any other types to string
        
        serializable_results = convert_complex(self.results)
        
        # Save JSON results
        json_file = os.path.join(output_dir, f"enhanced_quantum_geometry_results_{self.num_qubits}q.json")
        with open(json_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        print(f"💾 Enhanced results saved: {json_file}")
        
        # Save comprehensive summary
        summary_file = os.path.join(output_dir, f"enhanced_quantum_geometry_summary_{self.num_qubits}q.txt")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("ENHANCED QUANTUM GEOMETRY ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"EXPERIMENT PARAMETERS:\n")
            f.write(f"  - Number of qubits: {self.num_qubits}\n")
            f.write(f"  - Entanglement strength: {self.entanglement_strength}\n")
            f.write(f"  - Timesteps: {self.timesteps}\n")
            f.write(f"  - Asymmetry strength: {self.asymmetry_strength}\n")
            f.write(f"  - Circuit depth: {self.results['circuit_depth']}\n\n")
            
            f.write(f"QUANTUM GEOMETRY SCORE: {self.results['quantum_geometry_score']:.4f}\n\n")
            
            f.write(f"ANALYSIS RESULTS:\n")
            f.write(f"  1. Page Curve Analysis:\n")
            f.write(f"     - Page curve detected: {'YES' if self.results['page_curve_analysis']['page_curve_detected'] else 'NO'}\n")
            f.write(f"     - Linear model R²: {self.results['page_curve_analysis']['linear_r2']:.4f}\n")
            f.write(f"     - Page curve model R²: {self.results['page_curve_analysis']['page_curve_r2']:.4f}\n\n")
            
            f.write(f"  2. Area Law Analysis:\n")
            f.write(f"     - Area law R²: {self.results['area_law_analysis']['area_law_r2']:.4f}\n")
            f.write(f"     - Geodesic cutoff: {self.results['area_law_analysis']['geodesic_cutoff']:.1f}\n")
            f.write(f"     - Area law parameters: a={self.results['area_law_analysis']['area_law_params']['a']:.4f}, α={self.results['area_law_analysis']['area_law_params']['alpha']:.4f}\n\n")
            
            f.write(f"  3. Time Asymmetry Analysis:\n")
            f.write(f"     - Time asymmetry detected: {'YES' if self.results['time_asymmetry_analysis']['time_asymmetry_detected'] else 'NO'}\n")
            f.write(f"     - Asymmetry score: {self.results['time_asymmetry_analysis']['asymmetry_score']:.4f}\n\n")
            
            f.write(f"  4. Metric Eigenvalue Analysis:\n")
            f.write(f"     - Lorentzian signature: {'YES' if self.results['metric_analysis']['lorentzian_signature'] else 'NO'}\n")
            f.write(f"     - Condition number: {self.results['metric_analysis']['condition_number']:.4f}\n")
            f.write(f"     - Number of negative eigenvalues: {self.results['metric_analysis']['num_negative_eigenvalues']}\n")
            f.write(f"     - Number of positive eigenvalues: {self.results['metric_analysis']['num_positive_eigenvalues']}\n\n")
            
            f.write(f"  5. Mutual Information Analysis:\n")
            f.write(f"     - MI locality score: {self.results.get('mi_locality_score', 0.0):.4f}\n\n")
            
            f.write(f"PHYSICAL INTERPRETATION:\n")
            score = self.results['quantum_geometry_score']
            if score > 0.7:
                f.write(f"  🎉 STRONG EVIDENCE OF EMERGENT QUANTUM SPACETIME\n")
                f.write(f"  - High quantum geometry score indicates non-classical correlations\n")
                f.write(f"  - Page curve behavior suggests holographic entanglement structure\n")
                f.write(f"  - Time asymmetry indicates quantum dynamics beyond classical evolution\n")
                f.write(f"  - Metric structure consistent with emergent geometry\n")
            elif score > 0.4:
                f.write(f"  🔍 MODERATE EVIDENCE OF QUANTUM GEOMETRY\n")
                f.write(f"  - Some quantum features detected but not conclusive\n")
                f.write(f"  - Consider increasing entanglement strength or circuit depth\n")
                f.write(f"  - May need more sophisticated analysis techniques\n")
            else:
                f.write(f"  ❌ WEAK EVIDENCE OF QUANTUM GEOMETRY\n")
                f.write(f"  - Results suggest classical correlations dominate\n")
                f.write(f"  - Consider fundamental changes to circuit design\n")
                f.write(f"  - May need different entanglement patterns or more qubits\n")
        
        print(f"📝 Enhanced summary saved: {summary_file}")

def main():
    """Main function to run the enhanced quantum geometry analysis."""
    if len(sys.argv) < 2:
        print("Usage: python simple_page_curve_robust.py <output_directory>")
        print("Example: python simple_page_curve_robust.py experiment_logs/enhanced_quantum_geometry")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    # Create and run the enhanced experiment
    experiment = EnhancedQuantumGeometryAnalyzer(
        num_qubits=9,
        entanglement_strength=3.0,
        timesteps=12,
        shots=10000,
        asymmetry_strength=1.0
    )
    
    try:
        results = experiment.run_enhanced_experiment()
        
        if results:
            # Create comprehensive visualization
            experiment.create_comprehensive_visualization(output_dir)
            
            # Save enhanced results
            experiment.save_enhanced_results(output_dir)
            
            # Print summary
            score = results['quantum_geometry_score']
            print(f"\n📊 ENHANCED ANALYSIS SUMMARY:")
            print(f"  🎯 Quantum Geometry Score: {score:.4f}")
            print(f"  📈 Page Curve R²: {results['page_curve_analysis']['page_curve_r2']:.4f}")
            print(f"  📐 Area Law R²: {results['area_law_analysis']['area_law_r2']:.4f}")
            print(f"  ⏰ Time Asymmetry: {results['time_asymmetry_analysis']['asymmetry_score']:.4f}")
            print(f"  🌌 Lorentzian Signature: {'YES' if results['metric_analysis']['lorentzian_signature'] else 'NO'}")
            
            if score > 0.7:
                print(f"  🎉 EXCELLENT: Strong evidence of emergent quantum spacetime!")
            elif score > 0.4:
                print(f"  🔍 GOOD: Moderate evidence of quantum geometry")
            else:
                print(f"  ⚠️  WEAK: Limited evidence of quantum geometry")
        
    except Exception as e:
        print(f"❌ Error during enhanced experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 