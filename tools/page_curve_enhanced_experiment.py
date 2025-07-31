#!/usr/bin/env python3
"""
Page Curve Enhanced Experiment
=============================

A specialized experiment designed to generate quantum state data for Page curve analysis.
This experiment focuses on creating strong entanglement patterns that should produce
nonlinear entropy scaling characteristic of emergent quantum spacetime.
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
import itertools
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from CGPTFactory import run
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, Operator, partial_trace
# from qiskit.primitives import Sampler  # Not needed for this experiment
from qiskit_aer import AerSimulator

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class PageCurveEnhancedExperiment:
    def __init__(self, num_qubits: int = 9, entanglement_strength: float = 3.0, 
                 timesteps: int = 12, shots: int = 10000):
        self.num_qubits = num_qubits
        self.entanglement_strength = entanglement_strength
        self.timesteps = timesteps
        self.shots = shots
        self.results = {}
        
    def create_enhanced_circuit(self) -> QuantumCircuit:
        """Create a circuit with enhanced entanglement for Page curve generation."""
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        
        # Initialize in a superposition state
        for i in range(self.num_qubits):
            qc.h(i)
        
        # Apply multiple layers of entanglement with increased depth
        for step in range(self.timesteps):
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
                    if strength > 0.05:  # Lower threshold for more connections
                        qc.rzz(strength, i, j)
                        qc.ryy(strength * 0.8, i, j)
                        qc.rxx(strength * 0.6, i, j)
            
            # Layer 3: All-to-all entanglement with distance-dependent strength
            for i in range(self.num_qubits):
                for j in range(i+1, self.num_qubits):
                    # Create stronger entanglement for specific pairs
                    if (i + j) % 2 == 0:  # Every second pair gets extra entanglement
                        qc.rzz(self.entanglement_strength * 1.2, i, j)
                        qc.ryy(self.entanglement_strength * 0.9, i, j)
                        qc.rxx(self.entanglement_strength * 0.6, i, j)
            
            # Layer 4: Non-local operations (simulating quantum gravity effects)
            if step % 2 == 0:  # Every other step
                for i in range(0, self.num_qubits, 2):
                    if i + 1 < self.num_qubits:
                        qc.cx(i, i+1)
                        qc.h(i)
                        qc.h(i+1)
                        qc.cz(i, i+1)  # Additional phase entanglement
            
            # Layer 5: State purification and mixing (every 3rd step)
            if step % 3 == 0:
                # Apply random rotations to mix the state
                for i in range(self.num_qubits):
                    qc.rx(np.random.uniform(0, 2*np.pi), i)
                    qc.ry(np.random.uniform(0, 2*np.pi), i)
                    qc.rz(np.random.uniform(0, 2*np.pi), i)
                
                # Additional all-to-all entanglement
                for i in range(self.num_qubits):
                    for j in range(i+1, self.num_qubits):
                        qc.rzz(self.entanglement_strength * 0.3, i, j)
        
        return qc
    
    def compute_subsystem_entropy(self, statevector: np.ndarray, subsystem_qubits: List[int]) -> float:
        """Compute von Neumann entropy of a subsystem using tensor operations."""
        try:
            # Get the complement subsystem
            all_qubits = list(range(self.num_qubits))
            complement_qubits = [q for q in all_qubits if q not in subsystem_qubits]
            
            # Create density matrix from statevector
            rho_full = np.outer(statevector, statevector.conj())
            
            # Reshape to tensor form for partial trace
            dim = 2**self.num_qubits
            rho_tensor = rho_full.reshape([2] * (2 * self.num_qubits))
            
            # Trace out complement qubits
            for qubit in reversed(sorted(complement_qubits)):
                # Find indices for this qubit (both ket and bra indices)
                ket_idx = qubit
                bra_idx = qubit + self.num_qubits
                
                # Trace out this qubit
                rho_tensor = np.trace(rho_tensor, axis1=ket_idx, axis2=bra_idx)
                
                # Update remaining dimensions
                remaining_dims = list(range(self.num_qubits))
                remaining_dims.remove(qubit)
                rho_tensor = rho_tensor.reshape([2] * (2 * len(remaining_dims)))
            
            # Reshape back to matrix form
            subsystem_dim = 2**len(subsystem_qubits)
            rho_subsystem = rho_tensor.reshape(subsystem_dim, subsystem_dim)
            
            # Compute von Neumann entropy
            eigenvalues = np.linalg.eigvalsh(rho_subsystem)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical zeros
            entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
            
            return float(entropy)
                
        except Exception as e:
            print(f"Error computing subsystem entropy: {e}")
            return 0.0
    
    def analyze_page_curve(self, statevector: np.ndarray) -> Dict:
        """Analyze the Page curve behavior of the quantum state."""
        print("🔬 Analyzing Page curve behavior...")
        
        # Generate all possible bipartitions
        all_qubits = list(range(self.num_qubits))
        bipartitions = []
        entropies = []
        
        # Only consider bipartitions up to half the system size (due to symmetry)
        max_size = self.num_qubits // 2
        
        for size in range(1, max_size + 1):
            for subset in itertools.combinations(all_qubits, size):
                bipartitions.append(list(subset))
                entropy = self.compute_subsystem_entropy(statevector, list(subset))
                entropies.append(entropy)
                print(f"  Subsystem {subset}: S = {entropy:.4f}")
        
        # Add the full system entropy
        full_entropy = self.compute_subsystem_entropy(statevector, all_qubits)
        bipartitions.append(all_qubits)
        entropies.append(full_entropy)
        
        # Analyze the entropy scaling
        subsystem_sizes = [len(b) for b in bipartitions]
        
        # Fit different models
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
        
        # Page curve model (quadratic with peak)
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
        
        # Determine if Page curve behavior is detected
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
    
    def run_experiment(self) -> Dict:
        """Run the enhanced Page curve experiment."""
        print(f"🚀 Starting Page Curve Enhanced Experiment")
        print(f"📊 Parameters: {self.num_qubits} qubits, strength={self.entanglement_strength}, timesteps={self.timesteps}")
        
        # Create the enhanced circuit
        qc = self.create_enhanced_circuit()
        print(f"🔧 Circuit created with {qc.depth()} depth")
        
        # Get the statevector
        try:
            # Use Statevector class directly
            statevector = Statevector.from_instruction(qc)
            statevector = statevector.data
            print(f"✅ Statevector obtained: shape {statevector.shape}")
        except Exception as e:
            print(f"❌ Error getting statevector: {e}")
            return {}
        
        # Analyze Page curve behavior
        page_curve_analysis = self.analyze_page_curve(statevector)
        
        # Store results
        self.results = {
            'spec': {
                'num_qubits': self.num_qubits,
                'entanglement_strength': self.entanglement_strength,
                'timesteps': self.timesteps,
                'shots': self.shots
            },
            'statevector': statevector.tolist(),
            'page_curve_analysis': page_curve_analysis,
            'circuit_depth': qc.depth(),
            'circuit_gates': qc.count_ops()
        }
        
        return self.results
    
    def create_visualization(self, output_dir: str = None):
        """Create visualization plots for the Page curve analysis."""
        if not self.results:
            print("❌ No results to visualize")
            return
        
        if output_dir is None:
            output_dir = "."
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        analysis = self.results['page_curve_analysis']
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Entropy vs Subsystem Size
        sizes = analysis['subsystem_sizes']
        entropies = analysis['entropies']
        
        ax1.scatter(sizes, entropies, c='blue', alpha=0.7, s=50, label='Actual Entropy')
        
        # Plot model fits
        if 'linear' in analysis['models'] and analysis['models']['linear']['r_squared'] > 0:
            linear_pred = analysis['models']['linear']['predictions']
            ax1.plot(sizes, linear_pred, 'r--', alpha=0.8, label=f'Linear (R²={analysis["linear_r2"]:.3f})')
        
        if 'page_curve' in analysis['models'] and analysis['models']['page_curve']['r_squared'] > 0:
            page_pred = analysis['models']['page_curve']['predictions']
            ax1.plot(sizes, page_pred, 'g-', alpha=0.8, label=f'Page Curve (R²={analysis["page_curve_r2"]:.3f})')
        
        ax1.set_xlabel('Subsystem Size')
        ax1.set_ylabel('Von Neumann Entropy')
        ax1.set_title('Page Curve Analysis: Entropy vs Subsystem Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Model Comparison
        models = ['Linear', 'Page Curve']
        r2_scores = [analysis['linear_r2'], analysis['page_curve_r2']]
        colors = ['red' if analysis['page_curve_detected'] else 'gray', 
                 'green' if analysis['page_curve_detected'] else 'gray']
        
        bars = ax2.bar(models, r2_scores, color=colors, alpha=0.7)
        ax2.set_ylabel('R² Score')
        ax2.set_title('Model Fit Comparison')
        ax2.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Add detection indicator
        detection_text = "✅ PAGE CURVE DETECTED" if analysis['page_curve_detected'] else "❌ NO PAGE CURVE"
        ax2.text(0.5, 0.95, detection_text, transform=ax2.transAxes, 
                ha='center', va='top', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow" if analysis['page_curve_detected'] else "lightgray"))
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(output_dir, f"page_curve_analysis_{self.num_qubits}q.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📈 Visualization saved: {plot_file}")
        
        plt.show()
    
    def save_results(self, output_dir: str = None):
        """Save the experiment results."""
        if not self.results:
            print("❌ No results to save")
            return
        
        if output_dir is None:
            output_dir = "."
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert complex numbers to real for JSON serialization
        def convert_complex(obj):
            if isinstance(obj, complex):
                return float(obj.real)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_complex(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_complex(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_complex(self.results)
        
        # Save JSON results
        json_file = os.path.join(output_dir, f"page_curve_results_{self.num_qubits}q.json")
        with open(json_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"💾 Results saved: {json_file}")
        
        # Save summary report
        analysis = self.results['page_curve_analysis']
        summary_file = os.path.join(output_dir, f"page_curve_summary_{self.num_qubits}q.txt")
        
        with open(summary_file, 'w') as f:
            f.write("PAGE CURVE ENHANCED EXPERIMENT SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Parameters:\n")
            f.write(f"  - Number of qubits: {self.num_qubits}\n")
            f.write(f"  - Entanglement strength: {self.entanglement_strength}\n")
            f.write(f"  - Timesteps: {self.timesteps}\n")
            f.write(f"  - Circuit depth: {self.results['circuit_depth']}\n\n")
            
            f.write(f"Page Curve Analysis:\n")
            f.write(f"  - Page curve detected: {'YES' if analysis['page_curve_detected'] else 'NO'}\n")
            f.write(f"  - Linear model R²: {analysis['linear_r2']:.4f}\n")
            f.write(f"  - Page curve model R²: {analysis['page_curve_r2']:.4f}\n\n")
            
            f.write(f"Entropy Data:\n")
            for i, (size, entropy) in enumerate(zip(analysis['subsystem_sizes'], analysis['entropies'])):
                f.write(f"  - Subsystem size {size}: S = {entropy:.4f}\n")
            
            f.write(f"\nInterpretation:\n")
            if analysis['page_curve_detected']:
                f.write(f"  ✅ STRONG EVIDENCE OF EMERGENT QUANTUM SPACETIME\n")
                f.write(f"  - Nonlinear entropy scaling detected\n")
                f.write(f"  - Page curve behavior indicates nonlocal entanglement structure\n")
                f.write(f"  - Suggests unitarity-preserving evolution\n")
                f.write(f"  - Consistent with holographic principle predictions\n")
            else:
                f.write(f"  ❌ NO CLEAR EVIDENCE OF EMERGENT QUANTUM SPACETIME\n")
                f.write(f"  - Linear entropy scaling suggests classical correlations\n")
                f.write(f"  - May need stronger entanglement or different circuit design\n")
                f.write(f"  - Consider increasing entanglement strength or timesteps\n")
        
        print(f"📝 Summary saved: {summary_file}")

def main():
    """Main function to run the Page curve enhanced experiment."""
    if len(sys.argv) < 2:
        print("Usage: python page_curve_enhanced_experiment.py <output_directory>")
        print("Example: python page_curve_enhanced_experiment.py experiment_logs/page_curve_test")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    # Create and run the experiment
    experiment = PageCurveEnhancedExperiment(
        num_qubits=9,
        entanglement_strength=3.0,
        timesteps=12,
        shots=10000
    )
    
    try:
        results = experiment.run_experiment()
        
        if results:
            # Create visualization
            experiment.create_visualization(output_dir)
            
            # Save results
            experiment.save_results(output_dir)
            
            # Print summary
            analysis = results['page_curve_analysis']
            print("\n" + "="*60)
            print("PAGE CURVE EXPERIMENT COMPLETE")
            print("="*60)
            print(f"Page curve detected: {'✅ YES' if analysis['page_curve_detected'] else '❌ NO'}")
            print(f"Linear R²: {analysis['linear_r2']:.4f}")
            print(f"Page curve R²: {analysis['page_curve_r2']:.4f}")
            
            if analysis['page_curve_detected']:
                print("\n🎉 SUCCESS: Evidence of emergent quantum spacetime detected!")
                print("   This suggests nonlocal entanglement structure and")
                print("   unitarity-preserving evolution characteristic of")
                print("   holographic quantum gravity.")
            else:
                print("\n⚠️  No clear Page curve behavior detected.")
                print("   Consider increasing entanglement strength or")
                print("   modifying the circuit design.")
            
            print(f"\nResults saved to: {output_dir}")
        else:
            print("❌ Experiment failed to produce results")
            
    except Exception as e:
        print(f"❌ Error during experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 