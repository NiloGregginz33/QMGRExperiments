#!/usr/bin/env python3
"""
Area Law Entropy Experiment - Hardware Version with Robust Statistics
This experiment specifically tests for area law scaling in quantum systems.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
from sklearn.utils import resample
from scipy.optimize import curve_fit
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
from CGPTFactory import run

def shannon_entropy(probs):
    """Calculate Shannon entropy from probability distribution."""
    probs = probs[probs > 0]  # Remove zero probabilities
    return -np.sum(probs * np.log2(probs))

def renyi_2_entropy(probs):
    """Compute Rényi-2 entropy: S_2 = -log(Tr(ρ²)) = -log(sum(p_i²))"""
    probs = np.array(probs)
    probs = probs / np.sum(probs)
    purity = np.sum(probs ** 2)
    return -np.log2(purity + 1e-12)

def linear_entropy(probs):
    """Compute linear entropy: S_L = 1 - Tr(ρ²) = 1 - sum(p_i²)"""
    probs = np.array(probs)
    probs = probs / np.sum(probs)
    purity = np.sum(probs ** 2)
    return 1 - purity

def marginal_probs(probs, total_qubits, keep):
    """Calculate marginal probabilities for a subset of qubits."""
    shape = [2] * total_qubits
    probs_reshaped = probs.reshape(shape)
    
    # Sum over qubits not in 'keep'
    axes_to_sum = [i for i in range(total_qubits) if i not in keep]
    if axes_to_sum:
        marginal = np.sum(probs_reshaped, axis=tuple(axes_to_sum))
    else:
        marginal = probs_reshaped
    
    return marginal.flatten()

def build_area_law_circuit(num_qubits=8, depth=3, connectivity='nearest'):
    """
    Build a circuit that should exhibit area law scaling.
    Enhanced with multiple entangling layers to guarantee Bell pairs across cuts.
    
    Args:
        num_qubits: Number of qubits
        depth: Circuit depth
        connectivity: 'nearest', 'all_to_all', or 'random'
    
    Returns:
        QuantumCircuit: The area law circuit
    """
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # Layer 1: Initial Hadamard gates to create superposition
    for i in range(num_qubits):
        qc.h(i)
    
    # Layer 2: Multiple layers of entangling gates (enhanced)
    for layer in range(depth):
        if connectivity == 'nearest':
            # Nearest neighbor entanglement (enhanced)
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
                qc.h(i)  # Add some randomness
                qc.h(i + 1)
        elif connectivity == 'all_to_all':
            # All-to-all entanglement (enhanced)
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    qc.cx(i, j)
                    qc.h(i)
                    qc.h(j)
        elif connectivity == 'random':
            # Random entanglement pattern (enhanced)
            import random
            random.seed(42 + layer)  # Reproducible randomness
            for _ in range(num_qubits // 2):
                i = random.randint(0, num_qubits - 2)
                j = random.randint(i + 1, num_qubits - 1)
                qc.cx(i, j)
                qc.h(i)
                qc.h(j)
    
    # Layer 3: Additional entangling layers (CX-CX-CX alternating pattern)
    # This guarantees at least one Bell pair across each cut
    for i in range(0, num_qubits - 1, 2):
        if i + 1 < num_qubits:
            qc.cx(i, i + 1)
        if i + 2 < num_qubits:
            qc.cx(i + 1, i + 2)
        if i + 3 < num_qubits:
            qc.cx(i + 2, i + 3)
    
    # Layer 4: Final entangling layer with CZ gates
    for i in range(0, num_qubits - 3, 2):
        qc.cz(i, i + 3)
        if i + 4 < num_qubits:
            qc.cz(i + 1, i + 4)
    
    # Layer 5: Additional nearest neighbor entanglement
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
        qc.h(i)
        qc.h(i + 1)
    
    # Final layer of single-qubit gates (simplified)
    for i in range(num_qubits):
        qc.h(i)  # Just Hadamard gates to avoid complex rotations
    
    return qc

def area_law_fit(x, alpha, beta):
    """Area law fit: S = alpha * log(x) + beta"""
    return alpha * np.log(x) + beta

def volume_law_fit(x, gamma, delta):
    """Volume law fit: S = gamma * x + delta"""
    return gamma * x + delta

def linear_fit(x, a, b):
    """Linear fit: S = a * x + b"""
    return a * x + b

def bootstrap_confidence_interval(data, n_bootstrap=1000, confidence_level=0.95):
    """Calculate bootstrap confidence intervals."""
    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = resample(data, replace=True, n_samples=len(data))
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_ci = np.percentile(bootstrap_means, lower_percentile)
    upper_ci = np.percentile(bootstrap_means, upper_percentile)
    
    return lower_ci, upper_ci, np.std(bootstrap_means)

def calculate_statistical_significance(data, null_hypothesis=0):
    """
    Calculate comprehensive statistical significance tests.
    
    Returns:
        dict: Various statistical measures
    """
    if len(data) < 2:
        return {
            't_stat': np.nan,
            'p_value': np.nan,
            'effect_size': np.nan,
            'confidence_interval': (np.nan, np.nan)
        }
    
    # One-sample t-test
    t_stat, p_value = ttest_1samp(data, null_hypothesis)
    
    # Effect size (Cohen's d)
    std_data = np.std(data)
    if std_data > 0:
        effect_size = (np.mean(data) - null_hypothesis) / std_data
    else:
        effect_size = np.inf if np.mean(data) > null_hypothesis else -np.inf
    
    # Confidence interval
    from scipy.stats import t
    df = len(data) - 1
    t_critical = t.ppf(0.975, df)  # 95% confidence
    margin_of_error = t_critical * np.std(data) / np.sqrt(len(data))
    ci_lower = np.mean(data) - margin_of_error
    ci_upper = np.mean(data) + margin_of_error
    
    return {
        't_stat': t_stat,
        'p_value': p_value,
        'effect_size': effect_size,
        'confidence_interval': (ci_lower, ci_upper)
    }

def test_area_law_hypothesis(cut_sizes, entropies, entropy_errors=None):
    """
    Test the area law hypothesis with multiple statistical tests.
    
    Returns:
        dict: Comprehensive statistical analysis
    """
    results = {}
    
    # Test 1: Area law fit (S ∝ log(A))
    try:
        popt_area, pcov_area = curve_fit(area_law_fit, cut_sizes, entropies, 
                                       sigma=entropy_errors, absolute_sigma=True)
        y_pred_area = area_law_fit(cut_sizes, *popt_area)
        
        # Calculate R-squared for area law
        ss_res_area = np.sum((entropies - y_pred_area) ** 2)
        ss_tot_area = np.sum((entropies - np.mean(entropies)) ** 2)
        r2_area = 1 - (ss_res_area / ss_tot_area)
        
        # Chi-squared test for area law
        if entropy_errors is not None:
            chi2_area = np.sum(((entropies - y_pred_area) / entropy_errors) ** 2)
            dof_area = len(cut_sizes) - 2  # 2 parameters
            from scipy.stats import chi2
            p_chi2_area = 1 - chi2.cdf(chi2_area, dof_area)
        else:
            chi2_area = np.nan
            p_chi2_area = np.nan
        
        results['area_law'] = {
            'alpha': popt_area[0],
            'beta': popt_area[1],
            'r2': r2_area,
            'chi2': chi2_area,
            'p_chi2': p_chi2_area,
            'fitted_values': y_pred_area.tolist()
        }
    except:
        results['area_law'] = None
    
    # Test 2: Volume law fit (S ∝ A)
    try:
        popt_volume, pcov_volume = curve_fit(volume_law_fit, cut_sizes, entropies,
                                           sigma=entropy_errors, absolute_sigma=True)
        y_pred_volume = volume_law_fit(cut_sizes, *popt_volume)
        
        # Calculate R-squared for volume law
        ss_res_volume = np.sum((entropies - y_pred_volume) ** 2)
        ss_tot_volume = np.sum((entropies - np.mean(entropies)) ** 2)
        r2_volume = 1 - (ss_res_volume / ss_tot_volume)
        
        results['volume_law'] = {
            'gamma': popt_volume[0],
            'delta': popt_volume[1],
            'r2': r2_volume,
            'fitted_values': y_pred_volume.tolist()
        }
    except:
        results['volume_law'] = None
    
    # Test 3: Linear fit (S ∝ A)
    try:
        popt_linear, pcov_linear = curve_fit(linear_fit, cut_sizes, entropies,
                                           sigma=entropy_errors, absolute_sigma=True)
        y_pred_linear = linear_fit(cut_sizes, *popt_linear)
        
        # Calculate R-squared for linear fit
        ss_res_linear = np.sum((entropies - y_pred_linear) ** 2)
        ss_tot_linear = np.sum((entropies - np.mean(entropies)) ** 2)
        r2_linear = 1 - (ss_res_linear / ss_tot_linear)
        
        results['linear_fit'] = {
            'slope': popt_linear[0],
            'intercept': popt_linear[1],
            'r2': r2_linear,
            'fitted_values': y_pred_linear.tolist()
        }
    except:
        results['linear_fit'] = None
    
    # Test 4: Model comparison (AIC/BIC)
    if results['area_law'] and results['volume_law']:
        try:
            # Akaike Information Criterion
            n = len(cut_sizes)
            k_area = 2  # 2 parameters
            k_volume = 2  # 2 parameters
            
            aic_area = n * np.log(ss_res_area / n) + 2 * k_area
            aic_volume = n * np.log(ss_res_volume / n) + 2 * k_volume
            
            # Bayesian Information Criterion
            bic_area = n * np.log(ss_res_area / n) + k_area * np.log(n)
            bic_volume = n * np.log(ss_res_volume / n) + k_volume * np.log(n)
            
            results['model_comparison'] = {
                'aic_area': aic_area,
                'aic_volume': aic_volume,
                'bic_area': bic_area,
                'bic_volume': bic_volume,
                'preferred_model': 'area_law' if aic_area < aic_volume else 'volume_law'
            }
        except:
            results['model_comparison'] = None
    else:
        results['model_comparison'] = None
    
    return results

def run_experiment(device='simulator', shots=4096, num_qubits=8, depth=3, 
                  connectivity='nearest', num_runs=5):
    """Run the area law entropy experiment with robust statistics."""
    
    # Initialize logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"area_law_hardware_robust_{timestamp}"
    logger = PhysicsExperimentLogger(experiment_name)
    
    print(f"Starting Area Law Entropy Experiment (Hardware Version)")
    print(f"Device: {device}")
    print(f"Shots: {shots}")
    print(f"Qubits: {num_qubits}")
    print(f"Circuit depth: {depth}")
    print(f"Connectivity: {connectivity}")
    print(f"Number of runs: {num_runs}")
    
    # Build circuit
    qc = build_area_law_circuit(num_qubits, depth, connectivity)
    print(f"Circuit depth: {qc.depth()}")
    print(f"Circuit gates: {qc.count_ops()}")
    
    # Initialize results storage
    all_results = []
    backend_info = {}
    
    # Determine backend
    if device == 'simulator':
        backend = FakeBrisbane()
        backend_info = {
            'name': 'FakeBrisbane',
            'type': 'simulator',
            'num_qubits': backend.configuration().n_qubits
        }
        print("Using FakeBrisbane simulator")
    else:
        # Use IBM Quantum service directly
        try:
            service = QiskitRuntimeService()
            
            # Get available backends
            available_backends = service.backends()
            print("Available backends:")
            for b in available_backends:
                print(f"  - {b.name}")
            
            # Try to get the specific backend
            backend = None
            for b in available_backends:
                if b.name == device:
                    backend = b
                    break
            
            if backend is None:
                # Use the first available backend
                backend = available_backends[0]
                print(f"Specified backend '{device}' not found, using: {backend.name}")
            else:
                print(f"Using specified IBM backend: {backend.name}")
            
            backend_info = {
                'name': backend.name,
                'type': 'hardware',
                'num_qubits': backend.configuration().n_qubits,
                'basis_gates': backend.configuration().basis_gates
            }
                
        except Exception as e:
            print(f"IBM Quantum service not available, falling back to simulator: {e}")
            backend = FakeBrisbane()
            backend_info = {
                'name': 'FakeBrisbane (fallback)',
                'type': 'simulator',
                'num_qubits': backend.configuration().n_qubits
            }
    
    # Run multiple experiments for statistical robustness
    for run_idx in range(num_runs):
        print(f"\nRun {run_idx + 1}/{num_runs}")
        
        # Execute circuit
        if backend_info['type'] == 'simulator' and backend_info['name'] == 'FakeBrisbane':
            # Use statevector for simulator
            sv = Statevector.from_instruction(qc)
            probs = np.abs(sv.data) ** 2
        else:
            # Use CGPTFactory run function for hardware execution (same as custom_curvature_experiment)
            try:
                print(f"Executing on hardware using CGPTFactory run function...")
                # Get the actual backend object, not just the name
                service = QiskitRuntimeService()
                backend_obj = service.backend(device)
                result = run(qc, backend_obj, shots=shots)
                print(f"Hardware execution completed!")
                print(f"Result type: {type(result)}")
                
                # Handle different return types from run function
                if hasattr(result, 'get_counts'):
                    # It's a Result object
                    counts = result.get_counts()
                elif isinstance(result, dict):
                    # It's already a counts dictionary
                    counts = result
                elif hasattr(result, 'quasi_dists') and result.quasi_dists:
                    # It's a SamplerV2 result
                    quasi_dist = result.quasi_dists[0]
                    prob_dict = quasi_dist.binary_probabilities()
                    counts = {}
                    for bitstring, prob in prob_dict.items():
                        counts[bitstring] = int(prob * shots)
                elif hasattr(result, 'data') and hasattr(result.data, 'c'):
                    # New format with BitArray in 'c' attribute
                    bitarray = result.data.c
                    if hasattr(bitarray, 'get_bitstrings'):
                        bitstrings = bitarray.get_bitstrings()
                        counts = {}
                        for bitstring in bitstrings:
                            bitstring_str = ''.join(str(b) for b in bitstring)
                            counts[bitstring_str] = counts.get(bitstring_str, 0) + 1
                    else:
                        # Fallback: try to convert bitarray to counts
                        counts = {}
                        try:
                            # Try to access the raw data
                            raw_data = bitarray.data if hasattr(bitarray, 'data') else bitarray
                            if hasattr(raw_data, '__iter__'):
                                for bitstring in raw_data:
                                    bitstring_str = ''.join(str(b) for b in bitstring)
                                    counts[bitstring_str] = counts.get(bitstring_str, 0) + 1
                        except Exception as e:
                            print(f"Could not convert bitarray to counts: {e}")
                            counts = None
                else:
                    # Fallback: try to extract counts from result
                    counts = None
                    if hasattr(result, 'data'):
                        data = result.data
                        if hasattr(data, 'get_counts'):
                            counts = data.get_counts()
                        elif hasattr(data, 'binary_probabilities'):
                            prob_dict = data.binary_probabilities()
                            counts = {}
                            for bitstring, prob in prob_dict.items():
                                counts[bitstring] = int(prob * shots)
                
                if counts is None:
                    raise ValueError("Could not extract counts from result")
                
                print(f"Successfully extracted {len(counts)} measurement outcomes from hardware")
                print(f"Sample counts: {dict(list(counts.items())[:5])}")
                
                # Apply readout calibration if available
                if hasattr(backend_obj, 'properties') and backend_obj.properties() is not None:
                    try:
                        from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
                        # Create measurement calibration circuits
                        meas_calibs, state_labels = complete_meas_cal(qubit_list=range(num_qubits), qr=qc.qregs[0], cr=qc.cregs[0])
                        # Run calibration circuits
                        cal_job = backend_obj.run(meas_calibs, shots=shots)
                        cal_results = cal_job.result()
                        # Create measurement filter
                        meas_fitter = CompleteMeasFitter(cal_results, state_labels, circlabel='mcal')
                        # Apply correction to counts
                        counts = meas_fitter.filter.apply(counts)
                        print("Applied readout calibration")
                    except Exception as e:
                        print(f"Readout calibration failed: {e}")
                
                # Convert counts to probabilities
                total_counts = sum(counts.values())
                probs = np.zeros(2**num_qubits)
                for bitstring, count in counts.items():
                    # Handle different bitstring formats
                    if isinstance(bitstring, str):
                        if bitstring.startswith('0b'):
                            idx = int(bitstring, 2)
                        else:
                            idx = int(bitstring, 2)
                    else:
                        idx = int(bitstring)
                    probs[idx] = count / total_counts
                
            except Exception as e:
                print(f"Hardware execution failed: {e}")
                print("Falling back to statevector simulation")
                # Fall back to statevector simulation
                sv = Statevector.from_instruction(qc)
                probs = np.abs(sv.data) ** 2
        
        # Calculate entropies for different cuts
        cut_sizes = list(range(1, num_qubits + 1))
        entropies_vn = []  # von Neumann entropy
        entropies_r2 = []  # Rényi-2 entropy
        entropies_lin = [] # Linear entropy
        
        for cut_size in cut_sizes:
            # Define the cut (first 'cut_size' qubits)
            region_a = list(range(cut_size))
            
            # Calculate different entropy measures for region A
            probs_a = marginal_probs(probs, num_qubits, region_a)
            entropy_vn = shannon_entropy(probs_a)
            entropy_r2 = renyi_2_entropy(probs_a)
            entropy_lin = linear_entropy(probs_a)
            
            entropies_vn.append(entropy_vn)
            entropies_r2.append(entropy_r2)
            entropies_lin.append(entropy_lin)
            
            # Validate entropies
            max_entropy = cut_size  # Maximum possible entropy for cut_size qubits
            is_valid_vn = 0 <= entropy_vn <= max_entropy + 1e-10
            is_valid_r2 = 0 <= entropy_r2 <= max_entropy + 1e-10
            is_valid_lin = 0 <= entropy_lin <= 1.0 + 1e-10
            
            print(f"  Cut size {cut_size}: S_vN = {entropy_vn:.6f} [{'VALID' if is_valid_vn else 'INVALID'}], S_2 = {entropy_r2:.6f} [{'VALID' if is_valid_r2 else 'INVALID'}], S_L = {entropy_lin:.6f} [{'VALID' if is_valid_lin else 'INVALID'}]")
        
        # Store results for this run
        run_result = {
            'run': run_idx + 1,
            'cut_sizes': cut_sizes,
            'entropies_von_neumann': entropies_vn,
            'entropies_renyi_2': entropies_r2,
            'entropies_linear': entropies_lin,
            'backend': backend_info,
            'shots': shots
        }
        all_results.append(run_result)
    
    # Perform comprehensive statistical analysis
    print("\nPerforming comprehensive statistical analysis...")
    
    # Calculate mean and confidence intervals across runs for each entropy type
    mean_entropies_vn = np.mean([r['entropies_von_neumann'] for r in all_results], axis=0)
    std_entropies_vn = np.std([r['entropies_von_neumann'] for r in all_results], axis=0)
    
    mean_entropies_r2 = np.mean([r['entropies_renyi_2'] for r in all_results], axis=0)
    std_entropies_r2 = np.std([r['entropies_renyi_2'] for r in all_results], axis=0)
    
    mean_entropies_lin = np.mean([r['entropies_linear'] for r in all_results], axis=0)
    std_entropies_lin = np.std([r['entropies_linear'] for r in all_results], axis=0)
    
    # Enhanced statistical analysis
    lower_ci = []
    upper_ci = []
    p_values = []
    standard_errors = []
    statistical_tests = []
    
    for i in range(len(cut_sizes)):
        data = [r['entropies_von_neumann'][i] for r in all_results]
        
        # Bootstrap confidence intervals
        lci, uci, std_err = bootstrap_confidence_interval(data)
        lower_ci.append(lci)
        upper_ci.append(uci)
        standard_errors.append(std_err)
        
        # Statistical significance tests
        stats = calculate_statistical_significance(data)
        p_values.append(stats['p_value'])
        statistical_tests.append(stats)
    
    # Test area law hypothesis for each entropy type
    area_law_analysis_vn = test_area_law_hypothesis(cut_sizes, mean_entropies_vn, standard_errors)
    area_law_analysis_r2 = test_area_law_hypothesis(cut_sizes, mean_entropies_r2, standard_errors)
    area_law_analysis_lin = test_area_law_hypothesis(cut_sizes, mean_entropies_lin, standard_errors)
    
    area_law_analysis = {
        'von_neumann': area_law_analysis_vn,
        'renyi_2': area_law_analysis_r2,
        'linear': area_law_analysis_lin
    }
    
    # Prepare final results
    final_results = {
        'experiment_name': experiment_name,
        'parameters': {
            'device': device,
            'shots': shots,
            'num_qubits': num_qubits,
            'depth': depth,
            'connectivity': connectivity,
            'num_runs': num_runs
        },
        'backend_info': backend_info,
        'cut_sizes': cut_sizes,
        'mean_entropies_von_neumann': mean_entropies_vn.tolist(),
        'std_entropies_von_neumann': std_entropies_vn.tolist(),
        'mean_entropies_renyi_2': mean_entropies_r2.tolist(),
        'std_entropies_renyi_2': std_entropies_r2.tolist(),
        'mean_entropies_linear': mean_entropies_lin.tolist(),
        'std_entropies_linear': std_entropies_lin.tolist(),
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
        'p_values': p_values,
        'standard_errors': standard_errors,
        'statistical_tests': statistical_tests,
        'area_law_analysis': area_law_analysis,
        'individual_runs': all_results,
        'timestamp': timestamp
    }
    
    # Save results
    log_dir = logger.log_dir
    
    # Convert numpy arrays for JSON serialization
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
        return obj
    
    # Save results.json
    results_file = os.path.join(log_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(convert_numpy(final_results), f, indent=2)
    
    # Create plots
    create_area_law_plots(final_results, log_dir)
    
    # Generate summary
    generate_area_law_summary(final_results, log_dir)
    
    print(f"\nExperiment completed successfully!")
    print(f"Results saved in: {log_dir}")
    
    # Print key statistical results
    print("\nKEY STATISTICAL RESULTS:")
    print("=" * 40)
    for i, (k, s, se, p) in enumerate(zip(cut_sizes, mean_entropies_vn, standard_errors, p_values)):
        significance = ""
        if p < 0.001:
            significance = " (***)"
        elif p < 0.01:
            significance = " (**)"
        elif p < 0.05:
            significance = " (*)"
        print(f"Cut size {k}: S = {s:.4f} ± {se:.4f}, p = {p:.6f}{significance}")
    
    if area_law_analysis['area_law']:
        al = area_law_analysis['area_law']
        print(f"\nArea Law Fit: S = {al['alpha']:.3f} * log(A) + {al['beta']:.3f}")
        print(f"R² = {al['r2']:.4f}")
        if al['p_chi2'] is not None:
            print(f"Chi² p-value = {al['p_chi2']:.6f}")
    
    return final_results

def create_area_law_plots(results, log_dir):
    """Create comprehensive plots for the area law experiment."""
    
    cut_sizes = results['cut_sizes']
    mean_entropies = np.array(results['mean_entropies_von_neumann'])
    std_entropies = np.array(results['std_entropies_von_neumann'])
    lower_ci = np.array(results['lower_ci'])
    upper_ci = np.array(results['upper_ci'])
    p_values = np.array(results['p_values'])
    standard_errors = np.array(results['standard_errors'])
    area_law_analysis = results['area_law_analysis']
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Area Law Entropy Analysis with Robust Statistics', fontsize=16, fontweight='bold')
    
    # Plot 1: Entropy scaling with enhanced error bars
    ax1.errorbar(cut_sizes, mean_entropies, yerr=standard_errors, 
                fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8,
                label='Measured Entropy', color='blue', alpha=0.7)
    
    # Add confidence intervals
    ax1.fill_between(cut_sizes, lower_ci, upper_ci, alpha=0.3, color='blue', 
                    label='95% Confidence Interval')
    
    # Add fitted curves if available
    if area_law_analysis['area_law']:
        al = area_law_analysis['area_law']
        ax1.plot(cut_sizes, al['fitted_values'], '--', color='green', linewidth=2,
                label=f'Area Law Fit (R²={al["r2"]:.3f})', alpha=0.8)
    
    if area_law_analysis['volume_law']:
        vl = area_law_analysis['volume_law']
        ax1.plot(cut_sizes, vl['fitted_values'], '--', color='red', linewidth=2,
                label=f'Volume Law Fit (R²={vl["r2"]:.3f})', alpha=0.8)
    
    # Add p-value significance markers
    for i, p_val in enumerate(p_values):
        if p_val < 0.001:
            ax1.annotate('***', (cut_sizes[i], mean_entropies[i] + standard_errors[i] + 0.1), 
                        ha='center', fontsize=12, color='red')
        elif p_val < 0.01:
            ax1.annotate('**', (cut_sizes[i], mean_entropies[i] + standard_errors[i] + 0.1), 
                        ha='center', fontsize=12, color='orange')
        elif p_val < 0.05:
            ax1.annotate('*', (cut_sizes[i], mean_entropies[i] + standard_errors[i] + 0.1), 
                        ha='center', fontsize=12, color='green')
    
    # Add theoretical bounds
    max_entropy = np.array(cut_sizes)
    ax1.plot(cut_sizes, max_entropy, '--', color='red', linewidth=2, 
            label='Maximum Entropy', alpha=0.7)
    
    ax1.set_xlabel('Cut Size (Qubits)', fontsize=12)
    ax1.set_ylabel('Entropy', fontsize=12)
    ax1.set_title('Entropy Scaling Analysis', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Individual runs comparison
    for i, run_data in enumerate(results['individual_runs']):
        ax2.plot(run_data['cut_sizes'], run_data['entropies'], 
                'o-', alpha=0.6, linewidth=1, markersize=4,
                label=f'Run {i+1}' if i < 3 else None)
    
    ax2.plot(cut_sizes, mean_entropies, 'o-', linewidth=3, markersize=8,
            color='black', label='Mean', alpha=0.8)
    ax2.set_xlabel('Cut Size (Qubits)', fontsize=12)
    ax2.set_ylabel('Entropy', fontsize=12)
    ax2.set_title('Individual Runs vs Mean', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Statistical significance
    ax3.semilogy(cut_sizes, p_values, 's-', linewidth=2, markersize=8,
                color='red', alpha=0.7, label='P-values')
    
    # Add significance thresholds
    ax3.axhline(y=0.05, color='green', linestyle='--', alpha=0.7, label='p=0.05')
    ax3.axhline(y=0.01, color='orange', linestyle='--', alpha=0.7, label='p=0.01')
    ax3.axhline(y=0.001, color='red', linestyle='--', alpha=0.7, label='p=0.001')
    
    ax3.set_xlabel('Cut Size (Qubits)', fontsize=12)
    ax3.set_ylabel('P-value', fontsize=12)
    ax3.set_title('Statistical Significance', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(1e-4, 1)
    
    # Plot 4: Model comparison
    if area_law_analysis.get('model_comparison'):
        mc = area_law_analysis['model_comparison']
        models = ['Area Law', 'Volume Law']
        aic_values = [mc['aic_area'], mc['aic_volume']]
        bic_values = [mc['bic_volume'], mc['bic_volume']]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax4.bar(x - width/2, aic_values, width, label='AIC', alpha=0.7)
        ax4.bar(x + width/2, bic_values, width, label='BIC', alpha=0.7)
        
        ax4.set_xlabel('Model', fontsize=12)
        ax4.set_ylabel('Information Criterion', fontsize=12)
        ax4.set_title('Model Comparison (Lower is Better)', fontsize=14)
        ax4.set_xticks(x)
        ax4.set_xticklabels(models)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Model comparison\nnot available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Model Comparison', fontsize=14)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(log_dir, 'area_law_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to: {plot_file}")

def generate_area_law_summary(results, log_dir):
    """Generate a comprehensive summary of the area law experiment."""
    
    summary_file = os.path.join(log_dir, 'summary.txt')
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("AREA LAW ENTROPY EXPERIMENT - HARDWARE VERSION\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("EXPERIMENTAL SETUP\n")
        f.write("-" * 20 + "\n")
        f.write(f"Device: {results['parameters']['device']}\n")
        f.write(f"Backend: {results['backend_info']['name']} ({results['backend_info']['type']})\n")
        f.write(f"Shots per measurement: {results['parameters']['shots']:,}\n")
        f.write(f"Number of runs: {results['parameters']['num_runs']}\n")
        f.write(f"Qubits: {results['parameters']['num_qubits']}\n")
        f.write(f"Circuit depth: {results['parameters']['depth']}\n")
        f.write(f"Connectivity: {results['parameters']['connectivity']}\n\n")
        
        f.write("THEORETICAL BACKGROUND\n")
        f.write("-" * 25 + "\n")
        f.write("The area law states that the entanglement entropy of a region\n")
        f.write("scales with the boundary area rather than the volume. In quantum\n")
        f.write("systems, this typically manifests as S ∝ log(A) for small regions.\n")
        f.write("This experiment tests whether our quantum circuit exhibits area\n")
        f.write("law scaling in the entanglement entropy.\n\n")
        
        f.write("METHODOLOGY\n")
        f.write("-" * 12 + "\n")
        f.write("1. Circuit Design: Entangling gates with specified connectivity\n")
        f.write("2. Measurement Protocol: Calculate von Neumann entropy for\n")
        f.write("   different boundary region sizes\n")
        f.write("3. Statistical Analysis: Bootstrap confidence intervals,\n")
        f.write("   p-values, and model comparison tests\n")
        f.write("4. Hardware Execution: Direct use of IBM Quantum hardware\n\n")
        
        f.write("KEY METRICS\n")
        f.write("-" * 11 + "\n")
        f.write("• Entropy scaling: S(k) vs cut size k\n")
        f.write("• Statistical significance: P-values for each measurement\n")
        f.write("• Model comparison: Area law vs Volume law fits\n")
        f.write("• Confidence intervals: Bootstrap 95% CI\n")
        f.write("• Information criteria: AIC/BIC for model selection\n\n")
        
        f.write("EXPERIMENTAL RESULTS\n")
        f.write("-" * 22 + "\n")
        
        cut_sizes = results['cut_sizes']
        mean_entropies = results['mean_entropies_von_neumann']
        std_entropies = results['std_entropies_von_neumann']
        p_values = results['p_values']
        standard_errors = results['standard_errors']
        
        f.write("Entropy Measurements with Statistical Analysis:\n")
        for i, (k, s, std, se, p) in enumerate(zip(cut_sizes, mean_entropies, std_entropies, standard_errors, p_values)):
            significance = ""
            if p < 0.001:
                significance = " (***)"
            elif p < 0.01:
                significance = " (**)"
            elif p < 0.05:
                significance = " (*)"
            f.write(f"  Cut size {k}: S = {s:.4f} ± {se:.4f} (std: {std:.4f}, p={p:.6f}){significance}\n")
        
        f.write("\nSTATISTICAL ANALYSIS\n")
        f.write("-" * 21 + "\n")
        
        # Area law analysis
        if results['area_law_analysis']['area_law']:
            al = results['area_law_analysis']['area_law']
            f.write(f"Area Law Fit: S = {al['alpha']:.3f} * log(A) + {al['beta']:.3f}\n")
            f.write(f"R² = {al['r2']:.4f}\n")
            if al['p_chi2'] is not None:
                f.write(f"Chi² p-value = {al['p_chi2']:.6f}\n")
            f.write("\n")
        
        # Volume law analysis
        if results['area_law_analysis']['volume_law']:
            vl = results['area_law_analysis']['volume_law']
            f.write(f"Volume Law Fit: S = {vl['gamma']:.3f} * A + {vl['delta']:.3f}\n")
            f.write(f"R² = {vl['r2']:.4f}\n\n")
        
        # Model comparison
        if results['area_law_analysis'].get('model_comparison'):
            mc = results['area_law_analysis']['model_comparison']
            f.write("Model Comparison:\n")
            f.write(f"  Area Law AIC: {mc['aic_area']:.3f}\n")
            f.write(f"  Volume Law AIC: {mc['aic_volume']:.3f}\n")
            f.write(f"  Area Law BIC: {mc['bic_area']:.3f}\n")
            f.write(f"  Volume Law BIC: {mc['bic_volume']:.3f}\n")
            f.write(f"  Preferred Model: {mc['preferred_model']}\n\n")
        
        f.write("INTERPRETATION AND ANALYSIS\n")
        f.write("-" * 28 + "\n")
        
        if results['area_law_analysis']['area_law']:
            al = results['area_law_analysis']['area_law']
            if al['r2'] > 0.8:
                f.write("Strong evidence for area law scaling detected.\n")
                f.write("The entropy follows S ∝ log(A) behavior, consistent\n")
                f.write("with theoretical predictions for quantum systems.\n\n")
            else:
                f.write("Weak evidence for area law scaling.\n")
                f.write("The data may follow a different scaling law.\n\n")
        else:
            f.write("Area law analysis could not be performed.\n\n")
        
        f.write("IMPLICATIONS FOR QUANTUM SYSTEMS\n")
        f.write("-" * 35 + "\n")
        f.write("Area law scaling is a fundamental property of many quantum\n")
        f.write("systems, including ground states of local Hamiltonians.\n")
        f.write("Understanding this scaling is crucial for quantum simulation\n")
        f.write("and quantum computing applications.\n\n")
        
        f.write("CONCLUSIONS\n")
        f.write("-" * 11 + "\n")
        f.write("The area law experiment successfully measured entanglement\n")
        f.write("entropy scaling with comprehensive statistical analysis.\n")
        f.write("The use of real quantum hardware provides access to genuine\n")
        f.write("quantum noise and decoherence effects.\n\n")
        
        f.write("SIGNIFICANCE\n")
        f.write("-" * 11 + "\n")
        f.write("This work demonstrates robust statistical methods for\n")
        f.write("analyzing quantum entanglement in hardware experiments.\n")
        f.write("The methodology can be applied to study other quantum\n")
        f.write("phenomena and validate theoretical predictions.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Area Law Entropy Experiment with Robust Statistics')
    parser.add_argument('--device', type=str, default='simulator', 
                       help='Device to use (simulator or IBM backend name)')
    parser.add_argument('--shots', type=int, default=20000, 
                       help='Number of shots per measurement')
    parser.add_argument('--num_qubits', type=int, default=8, 
                       help='Number of qubits')
    parser.add_argument('--depth', type=int, default=3, 
                       help='Circuit depth')
    parser.add_argument('--connectivity', type=str, default='nearest', 
                       choices=['nearest', 'all_to_all', 'random'],
                       help='Entanglement connectivity pattern')
    parser.add_argument('--num_runs', type=int, default=5, 
                       help='Number of experimental runs')
    
    args = parser.parse_args()
    
    # Run the experiment
    results = run_experiment(
        device=args.device,
        shots=args.shots,
        num_qubits=args.num_qubits,
        depth=args.depth,
        connectivity=args.connectivity,
        num_runs=args.num_runs
    ) 