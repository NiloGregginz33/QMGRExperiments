#!/usr/bin/env python3
"""
Simple Area Law Entropy Experiment - Hardware Version
Uses only basic gates to avoid transpilation issues.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
from sklearn.utils import resample
from scipy.stats import ttest_1samp, ttest_ind, pearsonr, linregress
import argparse

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Factory'))
from utils.experiment_logger import PhysicsExperimentLogger
from CGPTFactory import run, extract_counts_from_bitarray

def build_simple_area_law_circuit(num_qubits=4, depth=2):
    """
    Build a very simple circuit that should exhibit area law scaling.
    Uses only H and CX gates to avoid transpilation issues.
    
    Args:
        num_qubits: Number of qubits
        depth: Circuit depth
    
    Returns:
        QuantumCircuit: The simple area law circuit
    """
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # Layer 1: Initial Hadamard gates
    for i in range(num_qubits):
        qc.h(i)
    
    # Multiple layers of simple entangling gates
    for layer in range(depth):
        # Nearest neighbor entanglement only
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
            qc.h(i)  # Add some randomness
            qc.h(i + 1)
    
    # Final layer of Hadamard gates
    for i in range(num_qubits):
        qc.h(i)
    
    return qc

def calculate_shannon_entropy(counts, total_shots):
    """Calculate Shannon entropy from measurement counts."""
    if total_shots == 0:
        return 0.0
    
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total_shots
            entropy -= p * np.log2(p)
    
    return entropy

def calculate_mutual_information(counts, total_shots, subsystem_a, subsystem_b):
    """Calculate mutual information between two subsystems."""
    # Calculate entropies for individual subsystems
    counts_a = {}
    counts_b = {}
    counts_ab = {}
    
    for bitstring, count in counts.items():
        # Extract subsystem bits
        bits_a = ''.join(bitstring[i] for i in subsystem_a)
        bits_b = ''.join(bitstring[i] for i in subsystem_b)
        
        # Count for subsystem A
        counts_a[bits_a] = counts_a.get(bits_a, 0) + count
        # Count for subsystem B
        counts_b[bits_b] = counts_b.get(bits_b, 0) + count
        # Count for combined system
        counts_ab[bitstring] = count
    
    # Calculate entropies
    S_a = calculate_shannon_entropy(counts_a, total_shots)
    S_b = calculate_shannon_entropy(counts_b, total_shots)
    S_ab = calculate_shannon_entropy(counts_ab, total_shots)
    
    # Mutual information: I(A:B) = S(A) + S(B) - S(AB)
    mutual_info = S_a + S_b - S_ab
    
    return mutual_info

def bootstrap_confidence_interval(data, confidence=0.95, n_bootstrap=1000):
    """Calculate bootstrap confidence interval."""
    if len(data) == 0:
        return 0.0, 0.0
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = resample(data, replace=True, n_samples=len(data))
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    # Calculate confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    
    return lower_bound, upper_bound

def calculate_p_value(data, null_hypothesis=0.0):
    """Calculate p-value using t-test."""
    if len(data) == 0:
        return np.nan
    
    # One-sample t-test
    t_stat, p_value = ttest_1samp(data, null_hypothesis)
    
    return p_value

def run_experiment(device='simulator', shots=1000, num_qubits=4, depth=2, num_runs=3):
    """
    Run the simple area law entropy experiment.
    
    Args:
        device: 'simulator' or IBM backend name
        shots: Number of shots per run
        num_qubits: Number of qubits
        depth: Circuit depth
        num_runs: Number of experimental runs
    
    Returns:
        dict: Experimental results
    """
    print(f"Starting Simple Area Law Entropy Experiment")
    print(f"Device: {device}")
    print(f"Shots: {shots}")
    print(f"Qubits: {num_qubits}")
    print(f"Circuit depth: {depth}")
    print(f"Number of runs: {num_runs}")
    
    # Build circuit
    qc = build_simple_area_law_circuit(num_qubits, depth)
    print(f"Circuit depth: {qc.depth()}")
    print(f"Circuit gates: {qc.count_ops()}")
    
    # Determine backend
    if device == 'simulator':
        backend_info = {'type': 'simulator', 'name': 'FakeBrisbane'}
        backend = FakeBrisbane()
    else:
        backend_info = {'type': 'hardware', 'name': device}
        service = QiskitRuntimeService()
        backend = service.backend(device)
    
    print(f"Using backend: {backend.name}")
    
    # Store results
    all_entropies = []
    all_mutual_infos = []
    run_results = []
    
    # Multiple experimental runs
    for run_idx in range(num_runs):
        print(f"\nRun {run_idx + 1}/{num_runs}")
        
        # Execute circuit
        if backend_info['type'] == 'simulator':
            # Use statevector for simulator
            qc_no_measure = qc.copy()
            qc_no_measure.remove_final_measurements()
            sv = Statevector.from_instruction(qc_no_measure)
            probs = np.abs(sv.data) ** 2
            
            # Convert to counts
            counts = {}
            for i, p in enumerate(probs):
                if p > 0:
                    bitstring = format(i, f'0{num_qubits}b')
                    counts[bitstring] = int(p * shots)
            
            print("Simulator execution completed!")
            
        else:
            # Use direct hardware execution
            try:
                print(f"Executing on hardware directly...")
                
                # Transpile with minimal optimization
                qc_t = transpile(qc, backend, optimization_level=0)
                print(f"Transpiled circuit depth: {qc_t.depth()}")
                print(f"Transpiled gates: {qc_t.count_ops()}")
                
                # Run directly with SamplerV2
                sampler = Sampler(backend)
                job = sampler.run([qc_t], shots=shots)
                result = job.result()
                
                print(f"Hardware execution completed!")
                print(f"Result type: {type(result)}")
                
                # Extract counts
                pub_result = result[0]
                data = pub_result.data
                bitarray = data.c
                
                print(f"BitArray: {bitarray}")
                
                # Convert to counts using CGPTFactory function
                counts = extract_counts_from_bitarray(bitarray)
                print(f"Counts: {counts}")
                
            except Exception as e:
                print(f"Hardware execution failed: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        # Calculate entropies for different cut sizes
        cut_entropies = []
        for cut_size in range(1, num_qubits + 1):
            # Calculate entropy for this cut size
            # We'll use the first 'cut_size' qubits as our subsystem
            subsystem_counts = {}
            total_count = 0
            
            for bitstring, count in counts.items():
                if len(bitstring) >= cut_size:
                    subsystem_bitstring = bitstring[:cut_size]
                    subsystem_counts[subsystem_bitstring] = subsystem_counts.get(subsystem_bitstring, 0) + count
                    total_count += count
            
            entropy = calculate_shannon_entropy(subsystem_counts, total_count)
            cut_entropies.append(entropy)
            print(f"  Cut size {cut_size}: Entropy = {entropy:.6f}")
        
        # Calculate mutual information between adjacent qubits
        mutual_infos = []
        for i in range(num_qubits - 1):
            mi = calculate_mutual_information(counts, shots, [i], [i + 1])
            mutual_infos.append(mi)
            print(f"  Mutual info qubits {i}-{i+1}: {mi:.6f}")
        
        # Store results
        run_result = {
            'run': run_idx + 1,
            'counts': counts,
            'cut_entropies': cut_entropies,
            'mutual_infos': mutual_infos,
            'total_shots': shots
        }
        run_results.append(run_result)
        
        all_entropies.append(cut_entropies)
        all_mutual_infos.append(mutual_infos)
    
    # Statistical analysis
    print(f"\nPerforming statistical analysis...")
    
    # Convert to numpy arrays
    all_entropies = np.array(all_entropies)
    all_mutual_infos = np.array(all_mutual_infos)
    
    # Calculate statistics for each cut size
    cut_sizes = list(range(1, num_qubits + 1))
    entropy_stats = []
    
    for i, cut_size in enumerate(cut_sizes):
        entropies = all_entropies[:, i]
        
        # Basic statistics
        mean_entropy = np.mean(entropies)
        std_entropy = np.std(entropies)
        
        # Bootstrap confidence interval
        ci_lower, ci_upper = bootstrap_confidence_interval(entropies)
        
        # P-value (test against null hypothesis of zero entropy)
        p_value = calculate_p_value(entropies, null_hypothesis=0.0)
        
        # Effect size (Cohen's d)
        if std_entropy > 0:
            effect_size = mean_entropy / std_entropy
        else:
            effect_size = np.inf if mean_entropy > 0 else -np.inf
        
        entropy_stats.append({
            'cut_size': cut_size,
            'mean': mean_entropy,
            'std': std_entropy,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'effect_size': effect_size,
            'n_runs': len(entropies)
        })
    
    # Area law analysis
    print(f"\nAnalyzing area law scaling...")
    
    # Fit area law: S = A * log(cut_size) + B
    def area_law_fit(x, A, B):
        return A * np.log(x) + B
    
    def volume_law_fit(x, A, B):
        return A * x + B
    
    # Get mean entropies for fitting
    mean_entropies = [stats['mean'] for stats in entropy_stats]
    entropy_errors = [stats['std'] for stats in entropy_stats]
    
    # Fit area law
    try:
        from scipy.optimize import curve_fit
        popt_area, pcov_area = curve_fit(area_law_fit, cut_sizes, mean_entropies, sigma=entropy_errors)
        y_pred_area = area_law_fit(np.array(cut_sizes), *popt_area)
        
        # Calculate R² for area law
        ss_res_area = np.sum((mean_entropies - y_pred_area) ** 2)
        ss_tot_area = np.sum((mean_entropies - np.mean(mean_entropies)) ** 2)
        r2_area = 1 - (ss_res_area / ss_tot_area) if ss_tot_area > 0 else 0
        
        # Chi-squared test
        chi2_area = np.sum(((mean_entropies - y_pred_area) / np.array(entropy_errors)) ** 2) if np.any(np.array(entropy_errors) > 0) else np.inf
        
        area_law_fit_result = {
            'A': popt_area[0],
            'B': popt_area[1],
            'r2': r2_area,
            'chi2': chi2_area,
            'p_value': 1 - chi2.cdf(chi2_area, len(cut_sizes) - 2) if chi2_area != np.inf else 0
        }
    except:
        area_law_fit_result = None
    
    # Fit volume law
    try:
        popt_volume, pcov_volume = curve_fit(volume_law_fit, cut_sizes, mean_entropies, sigma=entropy_errors)
        y_pred_volume = volume_law_fit(np.array(cut_sizes), *popt_volume)
        
        # Calculate R² for volume law
        ss_res_volume = np.sum((mean_entropies - y_pred_volume) ** 2)
        ss_tot_volume = np.sum((mean_entropies - np.mean(mean_entropies)) ** 2)
        r2_volume = 1 - (ss_res_volume / ss_tot_volume) if ss_tot_volume > 0 else 0
        
        volume_law_fit_result = {
            'A': popt_volume[0],
            'B': popt_volume[1],
            'r2': r2_volume
        }
    except:
        volume_law_fit_result = None
    
    # Compile results
    results = {
        'experiment_info': {
            'device': device,
            'shots': shots,
            'num_qubits': num_qubits,
            'depth': depth,
            'num_runs': num_runs,
            'timestamp': datetime.now().isoformat()
        },
        'entropy_statistics': entropy_stats,
        'area_law_analysis': {
            'area_law_fit': area_law_fit_result,
            'volume_law_fit': volume_law_fit_result,
            'cut_sizes': cut_sizes,
            'mean_entropies': mean_entropies,
            'entropy_errors': entropy_errors
        },
        'run_results': run_results
    }
    
    return results

def create_plots(results, save_path):
    """Create and save plots."""
    entropy_stats = results['entropy_statistics']
    area_law_analysis = results['area_law_analysis']
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Entropy vs cut size with error bars
    cut_sizes = [stats['cut_size'] for stats in entropy_stats]
    mean_entropies = [stats['mean'] for stats in entropy_stats]
    entropy_errors = [stats['std'] for stats in entropy_stats]
    p_values = [stats['p_value'] for stats in entropy_stats]
    
    ax1.errorbar(cut_sizes, mean_entropies, yerr=entropy_errors, 
                marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    
    # Add p-value significance markers
    for i, (x, y, p) in enumerate(zip(cut_sizes, mean_entropies, p_values)):
        if not np.isnan(p):
            if p < 0.001:
                ax1.annotate('***', (x, y + entropy_errors[i] + 0.02), ha='center', fontsize=12)
            elif p < 0.01:
                ax1.annotate('**', (x, y + entropy_errors[i] + 0.02), ha='center', fontsize=12)
            elif p < 0.05:
                ax1.annotate('*', (x, y + entropy_errors[i] + 0.02), ha='center', fontsize=12)
    
    ax1.set_xlabel('Cut Size (Number of Qubits)')
    ax1.set_ylabel('Shannon Entropy')
    ax1.set_title('Area Law Entropy Scaling')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Area law fit
    if area_law_analysis['area_law_fit']:
        fit_result = area_law_analysis['area_law_fit']
        x_fit = np.linspace(1, max(cut_sizes), 100)
        y_fit = fit_result['A'] * np.log(x_fit) + fit_result['B']
        
        ax2.plot(x_fit, y_fit, 'r--', linewidth=2, label=f"Area Law Fit\nR² = {fit_result['r2']:.3f}")
        ax2.errorbar(cut_sizes, mean_entropies, yerr=entropy_errors, 
                    marker='o', capsize=5, capthick=2, linewidth=2, markersize=8, label='Data')
        ax2.set_xlabel('Cut Size')
        ax2.set_ylabel('Entropy')
        ax2.set_title('Area Law Fit Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: P-values
    ax3.bar(cut_sizes, p_values, alpha=0.7, color='skyblue')
    ax3.axhline(y=0.05, color='red', linestyle='--', label='p = 0.05')
    ax3.axhline(y=0.01, color='orange', linestyle='--', label='p = 0.01')
    ax3.axhline(y=0.001, color='green', linestyle='--', label='p = 0.001')
    ax3.set_xlabel('Cut Size')
    ax3.set_ylabel('P-value')
    ax3.set_title('Statistical Significance')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Effect sizes
    effect_sizes = [stats['effect_size'] for stats in entropy_stats]
    ax4.bar(cut_sizes, effect_sizes, alpha=0.7, color='lightgreen')
    ax4.axhline(y=0.2, color='orange', linestyle='--', label='Small effect')
    ax4.axhline(y=0.5, color='red', linestyle='--', label='Medium effect')
    ax4.axhline(y=0.8, color='purple', linestyle='--', label='Large effect')
    ax4.set_xlabel('Cut Size')
    ax4.set_ylabel("Cohen's d")
    ax4.set_title('Effect Sizes')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to: {save_path}")

def save_results(results, base_path):
    """Save results to files."""
    # Save JSON results
    results_file = os.path.join(base_path, 'results.json')
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_json = convert_numpy(results)
    
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # Save summary
    summary_file = os.path.join(base_path, 'summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("SIMPLE AREA LAW ENTROPY EXPERIMENT - HARDWARE VERSION\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("EXPERIMENTAL SETUP\n")
        f.write("-" * 20 + "\n")
        f.write(f"Device: {results['experiment_info']['device']}\n")
        f.write(f"Shots per run: {results['experiment_info']['shots']}\n")
        f.write(f"Number of qubits: {results['experiment_info']['num_qubits']}\n")
        f.write(f"Circuit depth: {results['experiment_info']['depth']}\n")
        f.write(f"Number of runs: {results['experiment_info']['num_runs']}\n")
        f.write(f"Timestamp: {results['experiment_info']['timestamp']}\n\n")
        
        f.write("STATISTICAL RESULTS\n")
        f.write("-" * 20 + "\n")
        for stats in results['entropy_statistics']:
            f.write(f"Cut size {stats['cut_size']}: S = {stats['mean']:.4f} ± {stats['std']:.4f}, p = {stats['p_value']:.6f}\n")
        
        f.write("\nAREA LAW ANALYSIS\n")
        f.write("-" * 20 + "\n")
        if results['area_law_analysis']['area_law_fit']:
            fit = results['area_law_analysis']['area_law_fit']
            f.write(f"Area Law Fit: S = {fit['A']:.3f} * log(A) + {fit['B']:.3f}\n")
            f.write(f"R² = {fit['r2']:.6f}\n")
            f.write(f"Chi² p-value = {fit['p_value']:.6f}\n")
        else:
            f.write("Area law fit failed\n")
        
        if results['area_law_analysis']['volume_law_fit']:
            fit = results['area_law_analysis']['volume_law_fit']
            f.write(f"Volume Law Fit: S = {fit['A']:.3f} * A + {fit['B']:.3f}\n")
            f.write(f"R² = {fit['r2']:.6f}\n")
    
    print(f"Results saved in: {base_path}")

def main():
    parser = argparse.ArgumentParser(description='Simple Area Law Entropy Experiment')
    parser.add_argument('--device', default='simulator', help='Device: simulator or IBM backend name')
    parser.add_argument('--shots', type=int, default=1000, help='Number of shots per run')
    parser.add_argument('--num_qubits', type=int, default=4, help='Number of qubits')
    parser.add_argument('--depth', type=int, default=2, help='Circuit depth')
    parser.add_argument('--num_runs', type=int, default=3, help='Number of experimental runs')
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_experiment(
        device=args.device,
        shots=args.shots,
        num_qubits=args.num_qubits,
        depth=args.depth,
        num_runs=args.num_runs
    )
    
    if results is None:
        print("Experiment failed!")
        return
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"simple_area_law_hardware_{timestamp}"
    output_dir = os.path.join('experiment_logs', experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    save_results(results, output_dir)
    
    # Create plots
    plot_file = os.path.join(output_dir, 'area_law_analysis.png')
    create_plots(results, plot_file)
    
    # Print summary
    print(f"\nExperiment completed successfully!")
    print(f"Results saved in: {output_dir}")
    
    print(f"\nKEY STATISTICAL RESULTS:")
    print("=" * 40)
    for stats in results['entropy_statistics']:
        print(f"Cut size {stats['cut_size']}: S = {stats['mean']:.4f} ± {stats['std']:.4f}, p = {stats['p_value']:.6f}")
    
    if results['area_law_analysis']['area_law_fit']:
        fit = results['area_law_analysis']['area_law_fit']
        print(f"\nArea Law Fit: S = {fit['A']:.3f} * log(A) + {fit['B']:.3f}")
        print(f"R² = {fit['r2']:.6f}")
        print(f"Chi² p-value = {fit['p_value']:.6f}")

if __name__ == "__main__":
    main() 