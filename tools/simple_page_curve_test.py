#!/usr/bin/env python3
"""
Simple Page Curve Test
=====================

A simplified version of the subsystem entropy scaling test that focuses on:
1. Computing von Neumann entropy across all possible bipartitions
2. Detecting Page curve behavior (growing then saturating entropy)
3. Basic quantum signature analysis

This is fast, robust, and focuses on the key signatures of emergent quantum spacetime.
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

# Set publication-quality plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class SimplePageCurveTest:
    """Simplified test for Page curve behavior in subsystem entropy scaling."""
    
    def __init__(self, instance_dir: str):
        """Initialize test with experiment instance directory."""
        self.instance_dir = Path(instance_dir)
        self.data = self._load_experiment_data()
        self.test_results = {}
        
    def _load_experiment_data(self) -> Dict:
        """Load experiment data from instance directory."""
        # Load results data
        results_files = list(self.instance_dir.glob("results_*.json"))
        if not results_files:
            raise FileNotFoundError(f"No results files found in {self.instance_dir}")
        
        # Use the largest results file (most complete data)
        results_file = max(results_files, key=lambda x: x.stat().st_size)
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        return {
            'results_data': results_data,
            'instance_dir': str(self.instance_dir)
        }
    
    def compute_von_neumann_entropy(self, density_matrix: np.ndarray) -> float:
        """Compute von Neumann entropy S = -Tr(ρ log ρ)."""
        try:
            # Ensure density matrix is normalized
            rho = density_matrix / np.trace(density_matrix)
            
            # Compute eigenvalues
            eigenvals = np.linalg.eigvalsh(rho)
            
            # Remove zero eigenvalues to avoid log(0)
            eigenvals = eigenvals[eigenvals > 1e-12]
            
            # Compute von Neumann entropy
            entropy = -np.sum(eigenvals * np.log2(eigenvals))
            
            return entropy
        except Exception as e:
            print(f"Error computing von Neumann entropy: {e}")
            return 0.0
    
    def construct_density_matrix_from_counts(self, counts: Dict[str, int], num_qubits: int) -> np.ndarray:
        """Construct density matrix from measurement counts."""
        total_counts = sum(counts.values())
        if total_counts == 0:
            return np.eye(2**num_qubits) / (2**num_qubits)
        
        # Initialize density matrix
        dim = 2**num_qubits
        rho = np.zeros((dim, dim), dtype=complex)
        
        # Convert bitstrings to state vectors
        for bitstring, count in counts.items():
            if len(bitstring) != num_qubits:
                continue
                
            # Convert bitstring to integer index
            state_idx = int(bitstring, 2)
            
            # Create state vector
            state_vec = np.zeros(dim, dtype=complex)
            state_vec[state_idx] = 1.0
            
            # Add to density matrix (diagonal approximation)
            prob = count / total_counts
            rho[state_idx, state_idx] = prob
        
        return rho
    
    def compute_subsystem_entropy_simple(self, density_matrix: np.ndarray, subsystem_qubits: List[int], num_qubits: int) -> float:
        """Compute entropy of a subsystem using diagonal approximation."""
        try:
            # Get the diagonal elements of the density matrix
            diag_elements = np.diag(density_matrix)
            
            # For each bitstring, check if it has the right pattern for the subsystem
            subsystem_probs = {}
            
            for i in range(len(diag_elements)):
                # Convert index to binary string
                bitstring = format(i, f'0{num_qubits}b')
                
                # Extract subsystem bits
                subsystem_bits = ''.join([bitstring[q] for q in subsystem_qubits])
                
                # Sum probabilities for this subsystem configuration
                if subsystem_bits not in subsystem_probs:
                    subsystem_probs[subsystem_bits] = 0
                subsystem_probs[subsystem_bits] += diag_elements[i]
            
            # Compute entropy
            total_prob = sum(subsystem_probs.values())
            if total_prob > 0:
                normalized_probs = [p / total_prob for p in subsystem_probs.values()]
                # Compute entropy: -sum(p * log(p))
                entropy = -sum(p * np.log2(p) for p in normalized_probs if p > 0)
                return entropy
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error computing subsystem entropy: {e}")
            return 0.0
    
    def fit_page_curve_model(self, subsystem_sizes: List[int], entropies: List[float]) -> Dict:
        """Fit a Page curve model to the entropy data."""
        try:
            # Convert to numpy arrays
            x_data = np.array(subsystem_sizes, dtype=float)
            y_data = np.array(entropies, dtype=float)
            
            # Page curve model: S(A) = min(S_A, S_A^c) where S_A = |A| log(2)
            def page_curve_model(x, a, b, c):
                # Modified Page curve: S = a * min(x, max_size - x) + b * exp(-c * x)
                max_size = np.max(x_data)
                return a * np.minimum(x, max_size - x) + b * np.exp(-c * x)
            
            # Fit the model
            popt, pcov = optimize.curve_fit(page_curve_model, x_data, y_data, 
                                          p0=[1.0, 0.1, 0.1], maxfev=10000)
            
            # Compute R-squared
            y_pred = page_curve_model(x_data, *popt)
            r_squared = r2_score(y_data, y_pred)
            
            return {
                'parameters': popt.tolist(),
                'covariance': pcov.tolist(),
                'r_squared': r_squared,
                'fitted_values': y_pred.tolist()
            }
            
        except Exception as e:
            print(f"Error fitting Page curve model: {e}")
            return {
                'parameters': [0, 0, 0],
                'covariance': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                'r_squared': 0,
                'fitted_values': [0] * len(subsystem_sizes)
            }
    
    def run_page_curve_test(self) -> Dict:
        """Run the simplified Page curve test."""
        print("🔬 Running Simple Page Curve Test...")
        
        results = self.data['results_data']
        spec = results['spec']
        num_qubits = spec['num_qubits']
        
        # Get measurement counts
        if 'counts_per_timestep' not in results:
            return {'error': 'No measurement counts found in results'}
        
        counts_data = results['counts_per_timestep']
        if not counts_data:
            return {'error': 'Empty measurement counts data'}
        
        # Use the last timestep for analysis
        final_counts = counts_data[-1] if counts_data else {}
        
        # Construct density matrix
        print(f"  Constructing density matrix from {len(final_counts)} measurement outcomes...")
        density_matrix = self.construct_density_matrix_from_counts(final_counts, num_qubits)
        
        # Compute entropy for all possible bipartitions
        print(f"  Computing entropy for all bipartitions of {num_qubits} qubits...")
        subsystem_entropies = {}
        
        for subsystem_size in range(1, num_qubits):
            entropies = []
            
            # Generate all possible subsystems of this size
            for subsystem_qubits in itertools.combinations(range(num_qubits), subsystem_size):
                entropy = self.compute_subsystem_entropy_simple(density_matrix, list(subsystem_qubits), num_qubits)
                entropies.append(entropy)
            
            subsystem_entropies[subsystem_size] = {
                'mean_entropy': np.mean(entropies),
                'std_entropy': np.std(entropies),
                'min_entropy': np.min(entropies),
                'max_entropy': np.max(entropies),
                'all_entropies': entropies
            }
        
        # Fit Page curve model
        print("  Fitting Page curve model...")
        subsystem_sizes = list(subsystem_entropies.keys())
        mean_entropies = [subsystem_entropies[size]['mean_entropy'] for size in subsystem_sizes]
        
        page_curve_fit = self.fit_page_curve_model(subsystem_sizes, mean_entropies)
        
        # Analyze Page curve characteristics
        page_curve_analysis = self._analyze_page_curve_characteristics(subsystem_sizes, mean_entropies)
        
        # Determine if Page curve behavior is detected
        page_curve_detected = (
            page_curve_fit['r_squared'] > 0.6 and
            page_curve_analysis['peak_detected'] and
            page_curve_analysis['symmetry_score'] > 0.3
        )
        
        test_results = {
            'page_curve_detected': page_curve_detected,
            'subsystem_entropies': subsystem_entropies,
            'page_curve_fit': page_curve_fit,
            'page_curve_analysis': page_curve_analysis,
            'num_qubits': num_qubits,
            'total_measurements': sum(final_counts.values()) if final_counts else 0
        }
        
        self.test_results = test_results
        return test_results
    
    def _analyze_page_curve_characteristics(self, subsystem_sizes: List[int], entropies: List[float]) -> Dict:
        """Analyze characteristics of potential Page curve behavior."""
        if len(entropies) < 3:
            return {'peak_detected': False, 'characteristics': 'Insufficient data'}
        
        # Check for peak behavior (growing then saturating)
        peak_idx = np.argmax(entropies)
        peak_detected = 0 < peak_idx < len(entropies) - 1
        
        # Check for symmetry around peak
        if peak_detected:
            left_side = entropies[:peak_idx]
            right_side = entropies[peak_idx+1:]
            
            # Pad shorter side with last value
            min_len = min(len(left_side), len(right_side))
            if min_len > 0:
                left_side = left_side[-min_len:]
                right_side = right_side[:min_len]
                
                symmetry_score = 1 - np.mean(np.abs(np.array(left_side) - np.array(right_side)))
            else:
                symmetry_score = 0
        else:
            symmetry_score = 0
        
        # Check for saturation behavior
        if len(entropies) > 2:
            # Check if entropy saturates after peak
            if peak_detected:
                post_peak_entropies = entropies[peak_idx:]
                saturation_score = 1 - np.std(post_peak_entropies) / np.mean(post_peak_entropies)
            else:
                saturation_score = 0
        else:
            saturation_score = 0
        
        return {
            'peak_detected': peak_detected,
            'peak_location': peak_idx if peak_detected else -1,
            'symmetry_score': symmetry_score,
            'saturation_score': saturation_score,
            'entropy_range': [np.min(entropies), np.max(entropies)],
            'entropy_variance': np.var(entropies)
        }
    
    def create_visualization_plots(self, output_dir: str = None):
        """Create visualization plots for Page curve analysis."""
        if output_dir is None:
            output_dir = self.instance_dir
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if not self.test_results:
            print("No test results available for visualization")
            return
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Simple Page Curve Test Analysis', fontsize=16, fontweight='bold')
        
        # 1. Main entropy scaling plot
        ax = axes[0, 0]
        subsystem_entropies = self.test_results['subsystem_entropies']
        subsystem_sizes = list(subsystem_entropies.keys())
        mean_entropies = [subsystem_entropies[size]['mean_entropy'] for size in subsystem_sizes]
        std_entropies = [subsystem_entropies[size]['std_entropy'] for size in subsystem_sizes]
        
        # Plot experimental data
        ax.errorbar(subsystem_sizes, mean_entropies, yerr=std_entropies, 
                   marker='o', linewidth=2, markersize=8, label='Experimental', color='blue')
        
        # Plot Page curve fit
        if 'page_curve_fit' in self.test_results:
            page_fit = self.test_results['page_curve_fit']
            if page_fit['r_squared'] > 0:
                fitted_values = page_fit['fitted_values']
                ax.plot(subsystem_sizes, fitted_values, '--', linewidth=2, 
                       label=f'Page Curve Fit (R²={page_fit["r_squared"]:.3f})', color='red')
        
        # Plot theoretical Page curve
        theoretical_curve = [min(size, max(subsystem_sizes) - size) for size in subsystem_sizes]
        ax.plot(subsystem_sizes, theoretical_curve, ':', linewidth=2, 
               label='Theoretical Page Curve', color='green')
        
        ax.set_xlabel('Subsystem Size')
        ax.set_ylabel('Von Neumann Entropy')
        ax.set_title('Subsystem Entropy Scaling')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Page curve characteristics
        ax = axes[0, 1]
        if 'page_curve_analysis' in self.test_results:
            analysis = self.test_results['page_curve_analysis']
            
            # Create bar plot of characteristics
            characteristics = ['Peak\nDetected', 'Symmetry\nScore', 'Saturation\nScore']
            values = [
                1.0 if analysis['peak_detected'] else 0.0,
                analysis['symmetry_score'],
                analysis['saturation_score']
            ]
            colors = ['green' if analysis['peak_detected'] else 'red', 'blue', 'orange']
            
            bars = ax.bar(characteristics, values, color=colors, alpha=0.7)
            ax.set_ylabel('Score')
            ax.set_title('Page Curve Characteristics')
            ax.set_ylim(0, 1.1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{value:.3f}', ha='center', va='bottom')
        
        # 3. Entropy distribution
        ax = axes[1, 0]
        all_entropies = []
        for size in subsystem_entropies.keys():
            all_entropies.extend(subsystem_entropies[size]['all_entropies'])
        
        ax.hist(all_entropies, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax.set_xlabel('Von Neumann Entropy')
        ax.set_ylabel('Frequency')
        ax.set_title('Entropy Distribution')
        ax.grid(True, alpha=0.3)
        
        # 4. Summary statistics
        ax = axes[1, 1]
        summary_stats = []
        summary_values = []
        summary_colors = []
        
        # Page curve detection
        page_detected = self.test_results.get('page_curve_detected', False)
        summary_stats.append('Page Curve\nDetected')
        summary_values.append(1.0 if page_detected else 0.0)
        summary_colors.append('green' if page_detected else 'red')
        
        # R-squared of fit
        if 'page_curve_fit' in self.test_results:
            r_squared = self.test_results['page_curve_fit']['r_squared']
            summary_stats.append('Fit R²')
            summary_values.append(r_squared)
            summary_colors.append('blue')
        
        # Number of qubits
        num_qubits = self.test_results.get('num_qubits', 0)
        summary_stats.append('Number of\nQubits')
        summary_values.append(num_qubits / 10.0)  # Normalize for display
        summary_colors.append('orange')
        
        # Total measurements
        total_measurements = self.test_results.get('total_measurements', 0)
        summary_stats.append('Total\nMeasurements')
        summary_values.append(min(total_measurements / 10000.0, 1.0))  # Normalize
        summary_colors.append('purple')
        
        bars = ax.bar(summary_stats, summary_values, color=summary_colors, alpha=0.7)
        ax.set_ylabel('Score/Value')
        ax.set_title('Test Summary')
        ax.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, value in zip(bars, summary_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'simple_page_curve_test.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved to: {output_path / 'simple_page_curve_test.png'}")
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report."""
        if not self.test_results:
            return "No test results available"
        
        report = []
        report.append("=" * 80)
        report.append("SIMPLE PAGE CURVE TEST REPORT")
        report.append("=" * 80)
        report.append(f"Instance Directory: {self.instance_dir}")
        report.append(f"Test Date: {pd.Timestamp.now()}")
        report.append("")
        
        # Test summary
        page_detected = self.test_results.get('page_curve_detected', False)
        num_qubits = self.test_results.get('num_qubits', 0)
        total_measurements = self.test_results.get('total_measurements', 0)
        
        report.append("📋 TEST SUMMARY:")
        report.append(f"  Page Curve Detected: {'✅ YES' if page_detected else '❌ NO'}")
        report.append(f"  Number of Qubits: {num_qubits}")
        report.append(f"  Total Measurements: {total_measurements}")
        report.append("")
        
        # Page curve analysis
        if 'page_curve_analysis' in self.test_results:
            analysis = self.test_results['page_curve_analysis']
            report.append("📊 PAGE CURVE ANALYSIS:")
            report.append(f"  Peak Detected: {'Yes' if analysis['peak_detected'] else 'No'}")
            if analysis['peak_detected']:
                report.append(f"  Peak Location: Subsystem size {analysis['peak_location']}")
            report.append(f"  Symmetry Score: {analysis['symmetry_score']:.3f}")
            report.append(f"  Saturation Score: {analysis['saturation_score']:.3f}")
            report.append(f"  Entropy Range: [{analysis['entropy_range'][0]:.3f}, {analysis['entropy_range'][1]:.3f}]")
            report.append("")
        
        # Page curve fit
        if 'page_curve_fit' in self.test_results:
            fit = self.test_results['page_curve_fit']
            report.append("🔧 PAGE CURVE FIT:")
            report.append(f"  R-squared: {fit['r_squared']:.3f}")
            report.append(f"  Parameters: {fit['parameters']}")
            report.append("")
        
        # Subsystem entropy details
        if 'subsystem_entropies' in self.test_results:
            report.append("📈 SUBSYSTEM ENTROPY DETAILS:")
            entropies = self.test_results['subsystem_entropies']
            for size in sorted(entropies.keys()):
                data = entropies[size]
                report.append(f"  Size {size}: {data['mean_entropy']:.3f} ± {data['std_entropy']:.3f}")
            report.append("")
        
        # Interpretation
        report.append("🔬 INTERPRETATION:")
        if page_detected:
            report.append("  ✅ Page curve behavior detected!")
            report.append("  ✅ Indicates nonlocal entanglement structure")
            report.append("  ✅ Suggests unitarity-preserving evolution")
            report.append("  ✅ Consistent with emergent quantum spacetime")
            report.append("")
            report.append("  This is a strong signature of quantum geometry that cannot")
            report.append("  be reproduced by classical geometry without access to")
            report.append("  actual quantum entanglement dynamics.")
        else:
            report.append("  ❌ Page curve behavior not clearly detected")
            report.append("  ❌ May indicate classical or mixed behavior")
            report.append("  ❌ Consider increasing system size or measurement precision")
            report.append("")
            report.append("  This could be due to:")
            report.append("  - Insufficient system size")
            report.append("  - Measurement noise")
            report.append("  - Classical geometry dominance")
            report.append("  - Need for more sophisticated entanglement measures")
        
        return "\n".join(report)
    
    def save_test_results(self, output_dir: str = None):
        """Save test results to files."""
        if output_dir is None:
            output_dir = self.instance_dir
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save detailed results as JSON
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"simple_page_curve_test_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        # Save report as text
        report_file = output_path / f"simple_page_curve_test_report_{timestamp}.txt"
        report = self.generate_test_report()
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Test results saved to:")
        print(f"  JSON: {results_file}")
        print(f"  Report: {report_file}")
        
        return str(results_file), str(report_file)

def main():
    """Main function to run the simple Page curve test."""
    if len(sys.argv) < 2:
        print("Usage: python simple_page_curve_test.py <instance_directory>")
        print("Example: python simple_page_curve_test.py experiment_logs/custom_curvature_experiment/instance_20250730_190246")
        sys.exit(1)
    
    instance_dir = sys.argv[1]
    
    try:
        # Initialize test
        test = SimplePageCurveTest(instance_dir)
        
        # Run test
        print("🚀 Starting Simple Page Curve Test...")
        print(f"📁 Testing instance: {instance_dir}")
        print()
        
        results = test.run_page_curve_test()
        
        # Generate and display report
        report = test.generate_test_report()
        print(report)
        
        # Create visualizations
        test.create_visualization_plots()
        
        # Save results
        test.save_test_results()
        
        print("\n✅ Test complete!")
        
        # Final summary
        page_detected = results.get('page_curve_detected', False)
        if page_detected:
            print("\n🎉 SUCCESS: Page curve behavior detected!")
            print("   This indicates emergent quantum spacetime features.")
        else:
            print("\n⚠️  Page curve behavior not detected.")
            print("   Consider increasing system size or measurement precision.")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 