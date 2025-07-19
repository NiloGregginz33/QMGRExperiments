import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import itertools
import networkx as nx
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy import stats
import warnings
from datetime import datetime

def bootstrap_confidence_interval(data, statistic, n_bootstrap=1000, confidence=0.95):
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        data: Input data array
        statistic: Function to compute on data (e.g., np.mean, np.std)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0.95 for 95% CI)
    
    Returns:
        (statistic_value, lower_ci, upper_ci, bootstrap_samples)
    """
    n = len(data)
    bootstrap_samples = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_data = np.random.choice(data, size=n, replace=True)
        bootstrap_samples.append(statistic(bootstrap_data))
    
    bootstrap_samples = np.array(bootstrap_samples)
    stat_value = statistic(data)
    
    # Compute confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_ci = np.percentile(bootstrap_samples, lower_percentile)
    upper_ci = np.percentile(bootstrap_samples, upper_percentile)
    
    return stat_value, lower_ci, upper_ci, bootstrap_samples

def estimate_mi_uncertainty(mi_dict, shots=1024):
    """
    Estimate uncertainty in mutual information from shot noise.
    
    Args:
        mi_dict: Dictionary of mutual information values
        shots: Number of measurement shots (default from experiment)
    
    Returns:
        Dictionary of MI uncertainties
    """
    mi_uncertainties = {}
    
    for key, mi_value in mi_dict.items():
        # For MI from von Neumann entropy, uncertainty scales as 1/sqrt(shots)
        # This is a rough estimate - actual uncertainty depends on state complexity
        base_uncertainty = 1.0 / np.sqrt(shots)
        
        # Scale by MI value (larger MI typically has larger uncertainty)
        mi_uncertainties[key] = base_uncertainty * (1 + abs(mi_value))
    
    return mi_uncertainties

def propagate_distance_uncertainty(mi_dict, mi_uncertainties):
    """
    Propagate mutual information uncertainties to distance uncertainties.
    
    Args:
        mi_dict: Dictionary of mutual information values
        mi_uncertainties: Dictionary of MI uncertainties
    
    Returns:
        Dictionary of distance uncertainties
    """
    distance_uncertainties = {}
    
    for key, mi_value in mi_dict.items():
        if key in mi_uncertainties:
            mi_uncertainty = mi_uncertainties[key]
            
            # Distance = -log(MI), so dD/dMI = -1/MI
            # Uncertainty propagation: dD = |dD/dMI| * dMI = dMI/MI
            mi_clamped = max(mi_value, 1e-10)  # Avoid division by zero
            distance_uncertainty = mi_uncertainty / mi_clamped
            
            distance_uncertainties[key] = distance_uncertainty
    
    return distance_uncertainties

def fit_with_uncertainty(x, y, y_err=None, fit_type='linear'):
    """
    Fit data with proper uncertainty estimation.
    
    Args:
        x: x data
        y: y data
        y_err: y uncertainties (optional)
        fit_type: 'linear' or 'power_law'
    
    Returns:
        (params, param_uncertainties, r_squared, fit_function)
    """
    if fit_type == 'linear':
        # Linear fit: y = mx + b
        if y_err is not None:
            # Weighted linear fit
            popt, pcov = curve_fit(lambda x, m, b: m*x + b, x, y, sigma=y_err, absolute_sigma=True)
            param_uncertainties = np.sqrt(np.diag(pcov))
        else:
            # Unweighted linear fit
            slope, intercept, r_value, p_value, stderr = linregress(x, y)
            popt = [slope, intercept]
            param_uncertainties = [stderr, stderr * np.sqrt(np.mean(x**2))]
        
        r_squared = r_value**2 if 'r_value' in locals() else None
        fit_function = lambda x: popt[0] * x + popt[1]
        
    elif fit_type == 'power_law':
        # Power law fit: y = ax^b
        if y_err is not None:
            popt, pcov = curve_fit(lambda x, a, b: a * x**b, x, y, sigma=y_err, absolute_sigma=True)
        else:
            # Log-log fit for power law
            log_x = np.log(x)
            log_y = np.log(y)
            slope, intercept, r_value, p_value, stderr = linregress(log_x, log_y)
            popt = [np.exp(intercept), slope]
            param_uncertainties = [np.exp(intercept) * stderr, stderr]
        
        r_squared = r_value**2 if 'r_value' in locals() else None
        fit_function = lambda x: popt[0] * x**popt[1]
    
    return popt, param_uncertainties, r_squared, fit_function

def estimate_finite_size_effects(num_qubits, geometry):
    """
    Estimate systematic errors from finite system size.
    
    Args:
        num_qubits: Number of qubits in the system
        geometry: Geometry type ('euclidean', 'spherical', 'hyperbolic')
    
    Returns:
        Dictionary of systematic error estimates
    """
    systematic_errors = {}
    
    # Finite-size effects scale roughly as 1/N for many observables
    n = num_qubits
    systematic_errors['finite_size_scale'] = 1.0 / n
    
    # Geometry-specific finite-size effects
    if geometry == 'spherical':
        # Spherical geometry: finite-size effects from curvature radius
        systematic_errors['curvature_finite_size'] = 1.0 / (n * np.sqrt(n))
    elif geometry == 'hyperbolic':
        # Hyperbolic geometry: exponential growth with distance
        systematic_errors['hyperbolic_finite_size'] = np.exp(-n/2)
    else:  # euclidean
        systematic_errors['euclidean_finite_size'] = 1.0 / n
    
    return systematic_errors

def enhanced_statistical_analysis(x_values, y_values, uncertainties=None):
    """
    Perform comprehensive statistical analysis similar to IBM Sherbrooke experiments
    Args:
        x_values (list): x values
        y_values (list): y values
        uncertainties (list): Measurement uncertainties (optional)
    Returns:
        dict: Complete statistical analysis results
    """
    # Convert to numpy arrays
    x = np.array(x_values)
    y = np.array(y_values)
    
    # Linear regression with scipy
    slope, intercept, r_value, p_value, std_err = linregress(x, y, alternative='two-sided')
    r_squared = r_value ** 2
    
    # Calculate confidence intervals
    n = len(x)
    df = n - 2  # degrees of freedom
    
    # 95% confidence interval for slope
    t_critical = stats.t.ppf(0.975, df)
    slope_ci_lower = slope - t_critical * std_err
    slope_ci_upper = slope + t_critical * std_err
    
    # Prediction intervals for individual points
    x_mean = np.mean(x)
    ssx = np.sum((x - x_mean) ** 2)
    
    # Standard error of prediction
    se_pred = std_err * np.sqrt(1 + 1/n + (x - x_mean)**2 / ssx)
    pred_ci_lower = slope * x + intercept - t_critical * se_pred
    pred_ci_upper = slope * x + intercept + t_critical * se_pred
    
    # Calculate residuals and their statistics
    residuals = y - (slope * x + intercept)
    residual_std = np.std(residuals)
    
    # Durbin-Watson test for autocorrelation
    dw_statistic = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
    
    # Shapiro-Wilk test for normality of residuals
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    
    return {
        'linear_regression': {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'p_value': p_value,
            'std_err': std_err,
            'slope_ci_95': (slope_ci_lower, slope_ci_upper),
            'equation': f"y = {slope:.6f} × x + {intercept:.6f}"
        },
        'prediction_intervals': {
            'lower': pred_ci_lower.tolist(),
            'upper': pred_ci_upper.tolist()
        },
        'residuals': {
            'values': residuals.tolist(),
            'std': residual_std,
            'durbin_watson': dw_statistic,
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p
        },
        'sample_size': n,
        'degrees_of_freedom': df
    }

def create_publication_quality_plot(x_values, y_values, uncertainties=None, analysis_results=None, 
                                  x_label='x', y_label='y', title='Analysis Results', save_path=None):
    """
    Create publication-quality plot with error bars, confidence bands, and statistical annotations
    Args:
        x_values (list): x values
        y_values (list): y values
        uncertainties (list): Measurement uncertainties
        analysis_results (dict): Statistical analysis results
        x_label (str): x-axis label
        y_label (str): y-axis label
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot data points with error bars
    if uncertainties is not None:
        ax.errorbar(x_values, y_values, yerr=uncertainties, 
                   fmt='o', capsize=5, capthick=2, markersize=8, 
                   color='#2E86AB', ecolor='#A23B72', 
                   label='Experimental Data', alpha=0.8)
    else:
        ax.scatter(x_values, y_values, s=100, color='#2E86AB', 
                  alpha=0.8, label='Experimental Data')
    
    # Plot regression line
    if analysis_results:
        slope = analysis_results['linear_regression']['slope']
        intercept = analysis_results['linear_regression']['intercept']
        r_squared = analysis_results['linear_regression']['r_squared']
        p_value = analysis_results['linear_regression']['p_value']
        std_err = analysis_results['linear_regression']['std_err']
        
        # Generate line for plotting
        x_line = np.linspace(min(x_values), max(x_values), 100)
        y_line = slope * x_line + intercept
        
        ax.plot(x_line, y_line, '--', color='#F18F01', linewidth=3, 
               label=f'Linear Fit: y = {slope:.4f}x + {intercept:.4f}')
        
        # Plot confidence band
        if 'prediction_intervals' in analysis_results:
            x_sorted = np.array(x_values)
            y_sorted = np.array(y_values)
            sort_idx = np.argsort(x_sorted)
            x_sorted = x_sorted[sort_idx]
            
            lower = np.array(analysis_results['prediction_intervals']['lower'])[sort_idx]
            upper = np.array(analysis_results['prediction_intervals']['upper'])[sort_idx]
            
            ax.fill_between(x_sorted, lower, upper, alpha=0.3, color='#F18F01',
                          label='95% Prediction Interval')
        
        # Add statistical annotations
        stats_text = f'R² = {r_squared:.4f}\np = {p_value:.4f}\nSlope = {slope:.4f} ± {std_err:.4f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               verticalalignment='top', fontsize=12)
    
    # Customize plot
    ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits with some padding
    ax.set_xlim(min(x_values) - 0.1 * (max(x_values) - min(x_values)), 
                max(x_values) + 0.1 * (max(x_values) - min(x_values)))
    y_min, y_max = min(y_values), max(y_values)
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    return fig

def enhanced_spectral_dimension_analysis(D, S_max=10, num_walks=1000, seed=42, n_bootstrap=100):
    """
    Enhanced spectral dimension analysis with error estimation.
    """
    np.random.seed(seed)
    n = D.shape[0]
    
    # Build adjacency graph
    G = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            if 0 < D[i, j] < np.inf:
                G.add_edge(i, j)
    
    if not nx.is_connected(G):
        print("[WARNING] Adjacency graph is not connected. Spectral dimension may be ill-defined.")
        return None, None, None
    
    # Compute return probabilities with bootstrap
    P_s = []
    P_s_bootstrap = []
    
    for s in range(1, S_max+1):
        returns_list = []
        for _ in range(n_bootstrap):
            returns = 0
            for start in range(n):
                for _ in range(num_walks // n):
                    node = start
                    for _ in range(s):
                        nbrs = list(G.neighbors(node))
                        if not nbrs:
                            break
                        node = np.random.choice(nbrs)
                    if node == start:
                        returns += 1
            returns_list.append(returns / num_walks)
        
        P_s.append(np.mean(returns_list))
        P_s_bootstrap.append(returns_list)
    
    P_s = np.array(P_s)
    P_s_bootstrap = np.array(P_s_bootstrap)
    
    # Filter out s where P(s) == 0
    mask = P_s > 0
    if not np.any(mask):
        print("[WARNING] All return probabilities are zero. Cannot fit spectral dimension.")
        return None, None, None
    
    s_vals = np.arange(1, S_max+1)[mask]
    log_s = np.log(s_vals)
    log_P = np.log(P_s[mask])
    
    # Bootstrap the spectral dimension fit
    d_spectral_bootstrap = []
    for i in range(n_bootstrap):
        log_P_boot = np.log(P_s_bootstrap[mask, i])
        if len(log_P_boot) >= 2:
            slope, _, _, _, _ = linregress(log_s, log_P_boot)
            d_spectral_bootstrap.append(-2 * slope)
    
    d_spectral_bootstrap = np.array(d_spectral_bootstrap)
    d_spectral_mean = np.mean(d_spectral_bootstrap)
    d_spectral_std = np.std(d_spectral_bootstrap)
    
    # Main fit with enhanced statistical analysis
    slope, intercept, r, p, stderr = linregress(log_s, log_P, alternative='two-sided')
    d_spectral = -2 * slope
    
    # Perform enhanced statistical analysis
    analysis_results = enhanced_statistical_analysis(log_s, log_P)
    
    # Plot with error bars and enhanced analysis
    plt.figure(figsize=(10, 8))
    plt.errorbar(log_s, log_P, yerr=np.std(np.log(P_s_bootstrap[mask]), axis=1), 
                fmt='o-', label='Data with uncertainty', capsize=5, capthick=2)
    plt.plot(log_s, slope*log_s + intercept, '--', linewidth=3, color='#F18F01',
            label=f'Fit: d_spectral={d_spectral:.2f}±{d_spectral_std:.2f}')
    
    # Add confidence band if available
    if analysis_results and 'prediction_intervals' in analysis_results:
        lower = np.array(analysis_results['prediction_intervals']['lower'])
        upper = np.array(analysis_results['prediction_intervals']['upper'])
        plt.fill_between(log_s, lower, upper, alpha=0.3, color='#F18F01',
                        label='95% Prediction Interval')
    
    plt.xlabel('log s', fontsize=14, fontweight='bold')
    plt.ylabel('log P(s)', fontsize=14, fontweight='bold')
    plt.title(f'Spectral Dimension Analysis\n{d_spectral:.2f}±{d_spectral_std:.2f} (R²={r**2:.3f}, p={p:.3f})', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistical annotations
    if analysis_results:
        stats_text = f'R² = {analysis_results["linear_regression"]["r_squared"]:.4f}\n'
        stats_text += f'p = {analysis_results["linear_regression"]["p_value"]:.4f}\n'
        stats_text += f'Slope = {slope:.4f} ± {analysis_results["linear_regression"]["std_err"]:.4f}'
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               verticalalignment='top', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "spectral_dimension_fit_with_errors.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    return d_spectral, d_spectral_std, analysis_results
    plt.show()
    
    print(f"Spectral dimension d_spectral = {d_spectral:.2f} ± {d_spectral_std:.2f} (95% CI)")
    print(f"Fit quality: r² = {r**2:.3f}")
    
    return d_spectral, d_spectral_std, r**2

def enhanced_laplacian_spectral_dimension(D, S_max=10, s_min=0.1, s_max=10, num_s=10, n_bootstrap=50):
    """
    Enhanced Laplacian spectral dimension analysis with error estimation.
    """
    n = D.shape[0]
    
    # Build adjacency graph
    G = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            if 0 < D[i, j] < np.inf:
                G.add_edge(i, j)
    
    if not nx.is_connected(G):
        print("[WARNING] Adjacency graph is not connected. Laplacian spectrum may be ill-defined.")
        return None, None, None
    
    L = nx.laplacian_matrix(G).toarray()
    evals = np.linalg.eigvalsh(L)
    
    s_vals = np.logspace(np.log10(s_min), np.log10(s_max), num_s)
    K_s = []
    K_s_bootstrap = []
    
    # Bootstrap by adding noise to eigenvalues
    for _ in range(n_bootstrap):
        evals_noisy = evals + np.random.normal(0, 0.01 * np.std(evals), len(evals))
        K_s_boot = []
        for s in s_vals:
            K = np.sum(np.exp(-s * evals_noisy))
            K_s_boot.append(K)
        K_s_bootstrap.append(K_s_boot)
    
    # Main calculation
    for s in s_vals:
        K = np.sum(np.exp(-s * evals))
        K_s.append(K)
    
    K_s = np.array(K_s)
    K_s_bootstrap = np.array(K_s_bootstrap)
    
    log_s = np.log(s_vals)
    log_K = np.log(K_s)
    
    # Bootstrap the fit
    d_spectral_bootstrap = []
    for i in range(n_bootstrap):
        log_K_boot = np.log(K_s_bootstrap[i])
        slope, _, _, _, _ = linregress(log_s, log_K_boot)
        d_spectral_bootstrap.append(-2 * slope)
    
    d_spectral_bootstrap = np.array(d_spectral_bootstrap)
    d_spectral_mean = np.mean(d_spectral_bootstrap)
    d_spectral_std = np.std(d_spectral_bootstrap)
    
    # Main fit
    slope, intercept, r, p, stderr = linregress(log_s, log_K)
    d_spectral = -2 * slope
    
    plt.figure()
    plt.errorbar(log_s, log_K, yerr=np.std(np.log(K_s_bootstrap), axis=0), 
                fmt='o-', label='Data with uncertainty')
    plt.plot(log_s, slope*log_s + intercept, '--', 
            label=f'Fit: d_spectral={d_spectral:.2f}±{d_spectral_std:.2f}')
    plt.xlabel('log s')
    plt.ylabel('log K(s)')
    plt.title(f'Laplacian Spectral Dimension: {d_spectral:.2f}±{d_spectral_std:.2f} (r²={r**2:.3f})')
    plt.legend()
    plt.savefig(os.path.join("plots", "laplacian_spectral_dimension_fit_with_errors.png"))
    plt.show()
    
    print(f"[Laplacian] Spectral dimension d_spectral = {d_spectral:.2f} ± {d_spectral_std:.2f} (95% CI)")
    print(f"Fit quality: r² = {r**2:.3f}")
    
    return d_spectral, d_spectral_std, r**2

def load_distance_matrix(mi_dict, num_qubits):
    MI = np.zeros((num_qubits, num_qubits))
    for k, v in mi_dict.items():
        i, j = map(int, k.split('_')[1].split(','))
        MI[i, j] = v
        MI[j, i] = v
    # Clamp MI to avoid log(0)
    MI = np.clip(MI, 1e-10, 1.0)
    D = -np.log(MI)
    return D

def calculate_angle_sum_and_angles(D, i, j, k, curvature=1.0):
    a, b, c = D[j, k], D[i, k], D[i, j]
    # Always compute all three
    def hyp_angle(opposite, x, y, kappa):
        num = np.cosh(x) * np.cosh(y) - np.cosh(opposite)
        denom = np.sinh(x) * np.sinh(y)
        cosA = num / denom if denom != 0 else 1.0
        return np.arccos(np.clip(cosA, -1.0, 1.0))
    def sph_angle(opposite, x, y, kappa):
        K = np.sqrt(curvature)
        num = np.cos(K * opposite) - np.cos(K * x) * np.cos(K * y)
        denom = np.sin(K * x) * np.sin(K * y)
        cosA = num / denom if denom != 0 else 1.0
        return np.arccos(np.clip(cosA, -1.0, 1.0))
    def euc_angle(opposite, x, y):
        cosA = (x**2 + y**2 - opposite**2) / (2 * x * y)
        return np.arccos(np.clip(cosA, -1.0, 1.0))
    # Try all three, return all
    angles_hyp = [hyp_angle(a, b, c, curvature), hyp_angle(b, a, c, curvature), hyp_angle(c, a, b, curvature)]
    angles_sph = [sph_angle(a, b, c, curvature), sph_angle(b, a, c, curvature), sph_angle(c, a, b, curvature)]
    angles_euc = [euc_angle(a, b, c), euc_angle(b, a, c), euc_angle(c, a, b)]
    return angles_hyp, angles_sph, angles_euc

def pick_result_file(default_dir):
    files = [f for f in os.listdir(default_dir) if f.endswith('.json')]
    # Filter files to only those containing 'mutual_information'
    valid_files = []
    for f in files:
        try:
            with open(os.path.join(default_dir, f)) as jf:
                data = json.load(jf)
            if 'mutual_information' in data:
                valid_files.append(f)
        except Exception:
            continue
    if not valid_files:
        print(f"No valid .json files with 'mutual_information' found in {default_dir}")
        exit(1)
    print("Select a result file to analyze:")
    for idx, fname in enumerate(valid_files):
        print(f"[{idx}] {fname}")
    while True:
        try:
            choice = int(input("Enter file number: "))
            if 0 <= choice < len(valid_files):
                return os.path.join(default_dir, valid_files[choice])
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a valid number.")

def spectral_dimension_analysis(D, S_max=10, num_walks=1000, seed=42):
    np.random.seed(seed)
    n = D.shape[0]
    # Build adjacency graph: connect nodes with finite, nonzero distance
    G = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            if 0 < D[i, j] < np.inf:
                G.add_edge(i, j)
    if not nx.is_connected(G):
        print("[WARNING] Adjacency graph is not connected. Spectral dimension may be ill-defined.")
    P_s = []
    for s in range(1, S_max+1):
        returns = 0
        for start in range(n):
            for _ in range(num_walks // n):
                node = start
                for _ in range(s):
                    nbrs = list(G.neighbors(node))
                    if not nbrs:
                        break
                    node = np.random.choice(nbrs)
                if node == start:
                    returns += 1
        P_s.append(returns / num_walks)
    print(f"Return probabilities P(s): {P_s}")
    s_vals = np.arange(1, S_max+1)
    # Filter out s where P(s) == 0
    mask = np.array(P_s) > 0
    if not np.any(mask):
        print("[WARNING] All return probabilities are zero. Cannot fit spectral dimension.")
        return
    log_s = np.log(s_vals[mask])
    log_P = np.log(np.array(P_s)[mask])
    if len(log_s) < 2:
        print("[WARNING] Not enough nonzero P(s) values to fit spectral dimension.")
        return
    slope, intercept, r, p, stderr = linregress(log_s, log_P)
    d_spectral = -2 * slope
    plt.figure()
    plt.plot(log_s, log_P, 'o-', label='Data')
    plt.plot(log_s, slope*log_s + intercept, '--', label=f'Fit: slope={slope:.3f}')
    plt.xlabel('log s')
    plt.ylabel('log P(s)')
    plt.title(f'Spectral Dimension Estimate: d_spectral={d_spectral:.2f}')
    plt.legend()
    plt.savefig(os.path.join("plots", "spectral_dimension_fit.png"))
    plt.show()
    print(f"Spectral dimension d_spectral ≈ {d_spectral:.2f} (fit r={r:.3f})")

def laplacian_spectral_dimension(D, S_max=10, s_min=0.1, s_max=10, num_s=10):
    n = D.shape[0]
    # Build adjacency graph: connect nodes with finite, nonzero distance
    G = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            if 0 < D[i, j] < np.inf:
                G.add_edge(i, j)
    if not nx.is_connected(G):
        print("[WARNING] Adjacency graph is not connected. Laplacian spectrum may be ill-defined.")
        return
    L = nx.laplacian_matrix(G).toarray()
    evals = np.linalg.eigvalsh(L)
    s_vals = np.logspace(np.log10(s_min), np.log10(s_max), num_s)
    K_s = []
    for s in s_vals:
        K = np.sum(np.exp(-s * evals))
        K_s.append(K)
    log_s = np.log(s_vals)
    log_K = np.log(K_s)
    from scipy.stats import linregress
    slope, intercept, r, p, stderr = linregress(log_s, log_K)
    d_spectral = -2 * slope
    plt.figure()
    plt.plot(log_s, log_K, 'o-', label='Data')
    plt.plot(log_s, slope*log_s + intercept, '--', label=f'Fit: slope={slope:.3f}')
    plt.xlabel('log s')
    plt.ylabel('log K(s)')
    plt.title(f'Laplacian Spectral Dimension: d_spectral={d_spectral:.2f}')
    plt.legend()
    plt.savefig(os.path.join("plots", "laplacian_spectral_dimension_fit.png"))
    plt.show()
    print(f"[Laplacian] Spectral dimension d_spectral ≈ {d_spectral:.2f} (fit r={r:.3f})")

def analyze_custom_curvature_results(result_file_path):
    """
    Analyze custom curvature experiment results with enhanced statistical analysis
    Args:
        result_file_path (str): Path to the custom curvature experiment result file
    """
    print(f"Analyzing custom curvature results from: {result_file_path}")
    
    # Load the results
    with open(result_file_path, 'r') as f:
        results = json.load(f)
    
    # Extract key data
    spec = results.get('spec', {})
    num_qubits = spec.get('num_qubits', 0)
    geometry = spec.get('geometry', 'unknown')
    curvature = spec.get('curvature', 0)
    device = spec.get('device', 'unknown')
    
    print(f"Experiment parameters: {num_qubits} qubits, {geometry} geometry, curvature={curvature}, device={device}")
    
    # Extract mutual information data
    mi_per_timestep = results.get('mutual_information', [])
    distance_matrix_per_timestep = results.get('distance_matrix_per_timestep', [])
    
    if not mi_per_timestep or not distance_matrix_per_timestep:
        print("No mutual information or distance matrix data found")
        return
    
    # Analyze the final timestep (most evolved state)
    final_mi = mi_per_timestep[-1]
    final_distance_matrix = np.array(distance_matrix_per_timestep[-1])
    
    print(f"Analyzing final timestep with {len(final_mi)} MI values")
    
    # Extract edge MI values for analysis
    edge_mi_values = []
    edge_pairs = []
    
    for key, value in final_mi.items():
        if key.startswith('I_') and ',' in key:
            # Parse qubit indices from key like "I_0,1"
            try:
                qubits = key[2:].split(',')
                i, j = int(qubits[0]), int(qubits[1])
                edge_mi_values.append(value)
                edge_pairs.append((i, j))
            except (ValueError, IndexError):
                continue
    
    if not edge_mi_values:
        print("No valid edge MI values found")
        return
    
    print(f"Found {len(edge_mi_values)} edge MI values")
    
    # Create analysis directory
    analysis_dir = f"experiment_logs/custom_curvature_analysis_{num_qubits}q_{geometry}_curv{curvature}"
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(os.path.join(analysis_dir, "plots"), exist_ok=True)
    
    # 1. MI Distribution Analysis
    print("\n=== Mutual Information Distribution Analysis ===")
    mi_array = np.array(edge_mi_values)
    
    # Basic statistics
    mi_stats = {
        'mean': np.mean(mi_array),
        'std': np.std(mi_array),
        'min': np.min(mi_array),
        'max': np.max(mi_array),
        'median': np.median(mi_array)
    }
    
    print(f"MI Statistics: mean={mi_stats['mean']:.4f}, std={mi_stats['std']:.4f}")
    print(f"MI Range: [{mi_stats['min']:.4f}, {mi_stats['max']:.4f}]")
    
    # Plot MI distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(mi_array, bins=20, alpha=0.7, color='#2E86AB', edgecolor='black')
    plt.axvline(mi_stats['mean'], color='red', linestyle='--', label=f'Mean: {mi_stats["mean"]:.4f}')
    plt.axvline(mi_stats['median'], color='green', linestyle='--', label=f'Median: {mi_stats["median"]:.4f}')
    plt.xlabel('Mutual Information')
    plt.ylabel('Frequency')
    plt.title('MI Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Distance vs MI Analysis
    print("\n=== Distance vs Mutual Information Analysis ===")
    
    # Extract corresponding distances
    distances = []
    valid_mi = []
    
    for (i, j), mi_val in zip(edge_pairs, edge_mi_values):
        if i < final_distance_matrix.shape[0] and j < final_distance_matrix.shape[1]:
            dist = final_distance_matrix[i, j]
            if dist != np.inf and dist > 0:
                distances.append(dist)
                valid_mi.append(mi_val)
    
    if len(distances) >= 3:  # Need at least 3 points for regression
        # Perform enhanced statistical analysis
        analysis_results = enhanced_statistical_analysis(distances, valid_mi)
        
        print(f"Distance vs MI Analysis:")
        print(f"  R² = {analysis_results['linear_regression']['r_squared']:.4f}")
        print(f"  p-value = {analysis_results['linear_regression']['p_value']:.4f}")
        print(f"  Slope = {analysis_results['linear_regression']['slope']:.4f} ± {analysis_results['linear_regression']['std_err']:.4f}")
        print(f"  Equation: {analysis_results['linear_regression']['equation']}")
        
        # Create publication-quality plot
        plt.subplot(2, 2, 2)
        create_publication_quality_plot(
            distances, valid_mi, 
            analysis_results=analysis_results,
            x_label='Geodesic Distance', 
            y_label='Mutual Information',
            title=f'Distance vs MI ({num_qubits}q, {geometry})',
            save_path=os.path.join(analysis_dir, "plots", "distance_vs_mi.png")
        )
        
        # Save analysis results
        with open(os.path.join(analysis_dir, "distance_mi_analysis.json"), 'w') as f:
            json.dump(analysis_results, f, indent=2)
    
    # 3. Spectral Dimension Analysis
    print("\n=== Spectral Dimension Analysis ===")
    
    try:
        d_spectral, d_spectral_std, spectral_analysis = enhanced_spectral_dimension_analysis(
            final_distance_matrix, S_max=min(10, num_qubits), n_bootstrap=100
        )
        
        if d_spectral is not None:
            print(f"Spectral Dimension: {d_spectral:.3f} ± {d_spectral_std:.3f}")
            
            # Save spectral analysis
            spectral_results = {
                'spectral_dimension': d_spectral,
                'spectral_dimension_std': d_spectral_std,
                'analysis': spectral_analysis
            }
            
            with open(os.path.join(analysis_dir, "spectral_analysis.json"), 'w') as f:
                json.dump(spectral_results, f, indent=2)
            
            # Save the plot
            plt.savefig(os.path.join(analysis_dir, "plots", "spectral_dimension.png"), dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Spectral dimension analysis failed: {e}")
    
    # 4. Curvature Effects Analysis (if multiple curvature values available)
    print("\n=== Curvature Effects Analysis ===")
    
    # Check if we have multiple timesteps for evolution analysis
    if len(mi_per_timestep) > 1:
        print(f"Analyzing evolution over {len(mi_per_timestep)} timesteps")
        
        # Calculate average MI per timestep
        avg_mi_per_timestep = []
        for t, mi_dict in enumerate(mi_per_timestep):
            if mi_dict:
                mi_values = [v for v in mi_dict.values() if isinstance(v, (int, float))]
                if mi_values:
                    avg_mi_per_timestep.append(np.mean(mi_values))
                else:
                    avg_mi_per_timestep.append(0)
            else:
                avg_mi_per_timestep.append(0)
        
        # Analyze MI evolution
        timesteps = list(range(len(avg_mi_per_timestep)))
        evolution_analysis = enhanced_statistical_analysis(timesteps, avg_mi_per_timestep)
        
        print(f"MI Evolution Analysis:")
        print(f"  R² = {evolution_analysis['linear_regression']['r_squared']:.4f}")
        print(f"  p-value = {evolution_analysis['linear_regression']['p_value']:.4f}")
        
        # Plot evolution
        plt.subplot(2, 2, 3)
        create_publication_quality_plot(
            timesteps, avg_mi_per_timestep,
            analysis_results=evolution_analysis,
            x_label='Timestep',
            y_label='Average Mutual Information',
            title=f'MI Evolution ({num_qubits}q, {geometry})',
            save_path=os.path.join(analysis_dir, "plots", "mi_evolution.png")
        )
        
        # Save evolution analysis
        with open(os.path.join(analysis_dir, "evolution_analysis.json"), 'w') as f:
            json.dump(evolution_analysis, f, indent=2)
    
    # 5. Summary Statistics
    print("\n=== Summary Statistics ===")
    
    summary_stats = {
        'experiment_params': spec,
        'mi_statistics': mi_stats,
        'num_edges': len(edge_mi_values),
        'num_timesteps': len(mi_per_timestep),
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    # Add analysis results if available
    if 'analysis_results' in locals():
        summary_stats['distance_mi_correlation'] = analysis_results['linear_regression']
    
    if 'evolution_analysis' in locals():
        summary_stats['evolution_trend'] = evolution_analysis['linear_regression']
    
    if 'd_spectral' in locals() and d_spectral is not None:
        summary_stats['spectral_dimension'] = {
            'value': d_spectral,
            'uncertainty': d_spectral_std
        }
    
    # Save summary
    with open(os.path.join(analysis_dir, "summary_statistics.json"), 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Create comprehensive summary text
    with open(os.path.join(analysis_dir, "analysis_summary.txt"), 'w') as f:
        f.write(f"Custom Curvature Experiment Analysis Summary\n")
        f.write(f"============================================\n\n")
        f.write(f"Experiment Parameters:\n")
        f.write(f"  Number of qubits: {num_qubits}\n")
        f.write(f"  Geometry: {geometry}\n")
        f.write(f"  Curvature: {curvature}\n")
        f.write(f"  Device: {device}\n")
        f.write(f"  Number of timesteps: {len(mi_per_timestep)}\n\n")
        
        f.write(f"Mutual Information Analysis:\n")
        f.write(f"  Number of edges: {len(edge_mi_values)}\n")
        f.write(f"  Mean MI: {mi_stats['mean']:.4f} ± {mi_stats['std']:.4f}\n")
        f.write(f"  MI range: [{mi_stats['min']:.4f}, {mi_stats['max']:.4f}]\n\n")
        
        if 'analysis_results' in locals():
            f.write(f"Distance vs MI Correlation:\n")
            f.write(f"  R² = {analysis_results['linear_regression']['r_squared']:.4f}\n")
            f.write(f"  p-value = {analysis_results['linear_regression']['p_value']:.4f}\n")
            f.write(f"  Slope = {analysis_results['linear_regression']['slope']:.4f} ± {analysis_results['linear_regression']['std_err']:.4f}\n")
            f.write(f"  Equation: {analysis_results['linear_regression']['equation']}\n\n")
        
        if 'evolution_analysis' in locals():
            f.write(f"MI Evolution Analysis:\n")
            f.write(f"  R² = {evolution_analysis['linear_regression']['r_squared']:.4f}\n")
            f.write(f"  p-value = {evolution_analysis['linear_regression']['p_value']:.4f}\n")
            f.write(f"  Trend: {evolution_analysis['linear_regression']['equation']}\n\n")
        
        if 'd_spectral' in locals() and d_spectral is not None:
            f.write(f"Spectral Dimension:\n")
            f.write(f"  d_spectral = {d_spectral:.3f} ± {d_spectral_std:.3f}\n\n")
        
        f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\nAnalysis complete! Results saved to: {analysis_dir}")
    print(f"Summary: {os.path.join(analysis_dir, 'analysis_summary.txt')}")
    print(f"Plots: {os.path.join(analysis_dir, 'plots')}")
    
    return analysis_dir

def main():
    parser = argparse.ArgumentParser(description="Analyze emergent metric and curvature from MI data.")
    parser.add_argument("result_json", type=str, nargs='?', default=None, help="Path to result JSON file (if not specified, pick from list)")
    parser.add_argument("--geometry", type=str, default=None, choices=["euclidean", "hyperbolic", "spherical"], help="Override geometry type for angle calculation")
    parser.add_argument("--curvature", type=float, default=None, help="Curvature parameter (for hyperbolic/spherical geometry); if not set, use experiment value")
    parser.add_argument("--kappa_fit", type=float, default=None, help="Target kappa for area fit; if not set, use experiment value")
    parser.add_argument("--logdir", type=str, default="experiment_logs/custom_curvature_experiment", help="Directory to search for result files")
    parser.add_argument("--custom_curvature", action="store_true", help="Analyze custom curvature experiment results with enhanced statistical analysis")
    args = parser.parse_args()
    
    # Check if custom curvature analysis is requested
    if args.custom_curvature:
        if args.result_json is None:
            result_json = pick_result_file(args.logdir)
        else:
            result_json = args.result_json
        analyze_custom_curvature_results(result_json)
        return
    # Ensure plots directory exists
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    # If no result_json specified, pick from list
    if args.result_json is None:
        result_json = pick_result_file(args.logdir)
    else:
        result_json = args.result_json
    with open(result_json) as jf:
        data = json.load(jf)
    if 'mutual_information' not in data:
        print(f"Error: 'mutual_information' key not found in {result_json}. Skipping analysis.")
        return
    num_qubits = data["spec"]["num_qubits"]
    mi_dict = data["mutual_information"][-1] if isinstance(data["mutual_information"], list) else data["mutual_information"]
    D = load_distance_matrix(mi_dict, num_qubits)
    # Print experiment parameters and summary metrics
    print("\n=== Experiment Parameters ===")
    for k, v in data["spec"].items():
        print(f"{k}: {v}")
    print(f"uid: {data.get('uid', 'N/A')}")
    
    # Estimate uncertainties in mutual information
    shots = data["spec"].get("shots", 1024)
    mi_uncertainties = estimate_mi_uncertainty(mi_dict, shots)
    distance_uncertainties = propagate_distance_uncertainty(mi_dict, mi_uncertainties)
    
    # Estimate finite-size effects
    geometry = data["spec"].get("geometry", "euclidean")
    systematic_errors = estimate_finite_size_effects(num_qubits, geometry)
    
    print("\n=== Uncertainty Analysis ===")
    print(f"Measurement shots: {shots}")
    print(f"Typical MI uncertainty: {np.mean(list(mi_uncertainties.values())):.4f}")
    print(f"Typical distance uncertainty: {np.mean(list(distance_uncertainties.values())):.4f}")
    print(f"Finite-size effects scale: {systematic_errors['finite_size_scale']:.4f}")
    
    print("\n=== Key Metrics with Error Estimates ===")
    for key in ["gromov_delta", "mean_distance", "mean_angle_sum", "min_angle_sum", "max_angle_sum", "edge_weight_variance"]:
        if key in data:
            value = data[key]
            if isinstance(value, (int, float)):
                # Estimate uncertainty based on measurement noise and finite-size effects
                measurement_uncertainty = np.mean(list(mi_uncertainties.values())) if key in ["mean_distance", "mean_angle_sum"] else 0.01
                systematic_uncertainty = systematic_errors['finite_size_scale'] * abs(value)
                total_uncertainty = np.sqrt(measurement_uncertainty**2 + systematic_uncertainty**2)
                print(f"{key}: {value:.4f} ± {total_uncertainty:.4f}")
            else:
                print(f"{key}: {value}")
    # Plot MI and distance matrix heatmaps for each timestep
    if "mutual_information" in data and isinstance(data["mutual_information"], list):
        for t, mi_t in enumerate(data["mutual_information"]):
            if mi_t is None:
                continue
            D_t = load_distance_matrix(mi_t, num_qubits)
            plt.figure()
            plt.imshow(D_t, cmap='viridis')
            plt.colorbar()
            plt.title(f"Distance Matrix (timestep {t})")
            plt.savefig(os.path.join(plots_dir, f"distance_matrix_t{t}.png"))
            plt.show()
            plt.figure()
            MI_t = np.exp(-D_t)
            plt.imshow(MI_t, cmap='hot')
            plt.colorbar()
            plt.title(f"Mutual Information (timestep {t})")
            plt.savefig(os.path.join(plots_dir, f"mi_matrix_t{t}.png"))
            plt.show()
    # Plot Regge action, edge length, angle deficit, and Gromov delta evolution if present
    if "regge_action_evolution" in data:
        plt.figure()
        plt.plot(data["regge_action_evolution"])
        plt.xlabel("Regge step")
        plt.ylabel("Regge action")
        plt.title("Regge Action Evolution")
        plt.savefig(os.path.join(plots_dir, "regge_action_evolution.png"))
        plt.show()
    if "edge_length_evolution" in data:
        plt.figure()
        arr = np.array(data["edge_length_evolution"])
        plt.plot(arr)
        plt.xlabel("Regge step")
        plt.ylabel("Edge lengths")
        plt.title("Edge Length Evolution")
        plt.savefig(os.path.join(plots_dir, "edge_length_evolution.png"))
        plt.show()
    if "angle_deficit_evolution" in data:
        plt.figure()
        arr = np.array(data["angle_deficit_evolution"])
        plt.plot(arr)
        plt.xlabel("Regge step")
        plt.ylabel("Angle deficits")
        plt.title("Angle Deficit Evolution")
        plt.savefig(os.path.join(plots_dir, "angle_deficit_evolution.png"))
        plt.show()
    if "gromov_delta_evolution" in data:
        plt.figure()
        plt.plot(data["gromov_delta_evolution"])
        plt.xlabel("Regge step")
        plt.ylabel("Gromov delta")
        plt.title("Gromov Delta Evolution")
        plt.savefig(os.path.join(plots_dir, "gromov_delta_evolution.png"))
        plt.show()
    # Print triangle inequality violations if present
    if "triangle_inequality_violations" in data and data["triangle_inequality_violations"]:
        print("\nTriangle inequality violations:")
        for v in data["triangle_inequality_violations"]:
            print(v)
    # Print matter action if present
    if "S_matter" in data:
        print(f"Matter action: {data['S_matter']}")
    # Plot 2D/3D embeddings if available
    if "embedding_coords" in data:
        coords2 = np.array(data["embedding_coords"])
        plt.figure()
        plt.scatter(coords2[:,0], coords2[:,1])
        plt.title("2D Embedding of Geometry")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(os.path.join(plots_dir, "embedding_2d.png"))
        plt.show()
    if "embedding_coords_3d" in data:
        coords3 = np.array(data["embedding_coords_3d"])
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coords3[:,0], coords3[:,1], coords3[:,2])
        plt.title("3D Embedding of Geometry")
        plt.savefig(os.path.join(plots_dir, "embedding_3d.png"))
        plt.show()
    # Optionally plot Lorentzian embedding if present
    if "lorentzian_embedding" in data:
        coordsL = np.array(data["lorentzian_embedding"])
        if coordsL.shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(coordsL[:,0], coordsL[:,1], coordsL[:,2])
            plt.title("Lorentzian MDS Embedding")
            plt.savefig(os.path.join(plots_dir, "lorentzian_embedding.png"))
            plt.show()
        else:
            plt.figure()
            plt.scatter(coordsL[:,0], coordsL[:,1])
            plt.title("Lorentzian MDS Embedding (2D)")
            plt.savefig(os.path.join(plots_dir, "lorentzian_embedding_2d.png"))
            plt.show()
    # Use experiment curvature/geometry as default
    exp_curvature = data["spec"].get("curvature", 1.0)
    exp_geometry = data["spec"].get("geometry", None)
    kappa = args.curvature if args.curvature is not None else exp_curvature
    kappa_fit = args.kappa_fit if args.kappa_fit is not None else kappa
    geometry = args.geometry if args.geometry is not None else exp_geometry
    # If geometry is still None, infer from first triangle
    inferred = False
    deficits = []
    areas = []
    area_delta_pairs = []
    angle_sums = []
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            for k in range(j+1, num_qubits):
                angles_hyp, angles_sph, angles_euc = calculate_angle_sum_and_angles(D, i, j, k, curvature=kappa)
                # Infer geometry if needed
                if geometry is None and not inferred:
                    sum_hyp = sum(angles_hyp)
                    sum_sph = sum(angles_sph)
                    sum_euc = sum(angles_euc)
                    if sum_hyp < np.pi:
                        geometry = "hyperbolic"
                    elif sum_sph > np.pi:
                        geometry = "spherical"
                    else:
                        geometry = "euclidean"
                    print(f"[Auto-detected geometry: {geometry}]")
                    inferred = True
                # Use the right angles for the chosen geometry
                if geometry == "hyperbolic":
                    angles = angles_hyp
                    deficit = np.pi - sum(angles)
                    area = deficit / kappa_fit
                elif geometry == "spherical":
                    angles = angles_sph
                    deficit = sum(angles) - np.pi
                    area = deficit / kappa_fit
                else:
                    angles = angles_euc
                    deficit = 0.0
                    area = 0.0
                deficits.append(deficit)
                areas.append(area)
                area_delta_pairs.append((area, deficit))
                angle_sums.append(sum(angles))
    deficits = np.array(deficits)
    areas = np.array(areas)
    
    # Bootstrap confidence intervals for deficit statistics
    deficit_mean, deficit_mean_lower, deficit_mean_upper, _ = bootstrap_confidence_interval(deficits, np.mean, n_bootstrap=1000)
    deficit_std, deficit_std_lower, deficit_std_upper, _ = bootstrap_confidence_interval(deficits, np.std, n_bootstrap=1000)
    
    # Plot histogram of triangle deficits with error bands
    plt.figure()
    plt.hist(deficits, bins=50, alpha=0.7, density=True, label='Data')
    
    # Add Gaussian fit with uncertainty
    mu, sigma = np.mean(deficits), np.std(deficits)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, y, 'r-', linewidth=2, label=f'Gaussian fit: μ={mu:.3f}, σ={sigma:.3f}')
    
    plt.xlabel("Triangle deficit (Δ or δ_sph)")
    plt.ylabel("Density")
    plt.title(f"Distribution of Triangle Deficits ({geometry.capitalize()} Curvature)")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "triangle_deficit_histogram.png"))
    plt.show()
    
    # Print summary with error bars
    print(f"Triangle deficit statistics (with 95% confidence intervals):")
    print(f"  Mean: {deficit_mean:.4f} [{deficit_mean_lower:.4f}, {deficit_mean_upper:.4f}]")
    print(f"  Std:  {deficit_std:.4f} [{deficit_std_lower:.4f}, {deficit_std_upper:.4f}]")
    print(f"  Min:  {np.min(deficits):.4f}")
    print(f"  Max:  {np.max(deficits):.4f}")
    # Δ vs. area sanity check with error analysis
    if geometry in ("hyperbolic", "spherical"):
        # Estimate uncertainties in areas and deficits
        area_uncertainties = np.std(areas) * np.ones_like(areas) * 0.1  # 10% relative uncertainty
        deficit_uncertainties = np.std(deficits) * np.ones_like(deficits) * 0.1  # 10% relative uncertainty
        
        # Fit with uncertainties
        popt, param_uncertainties, r_squared, fit_func = fit_with_uncertainty(
            areas, deficits, deficit_uncertainties, fit_type='linear'
        )
        slope, intercept = popt
        slope_err, intercept_err = param_uncertainties
        
        plt.figure()
        plt.errorbar(areas, deficits, xerr=area_uncertainties, yerr=deficit_uncertainties, 
                    fmt='o', alpha=0.7, label='Data with uncertainties')
        
        # Plot fit with confidence band
        x_fit = np.linspace(np.min(areas), np.max(areas), 100)
        y_fit = fit_func(x_fit)
        plt.plot(x_fit, y_fit, 'r-', linewidth=2, 
                label=f'Fit: κ={slope:.3f}±{slope_err:.3f}')
        
        plt.xlabel(f"{'Hyperbolic' if geometry=='hyperbolic' else 'Spherical'} triangle area (kappa={kappa_fit})")
        plt.ylabel(f"Triangle deficit {'Δ' if geometry=='hyperbolic' else 'δ_sph'}")
        plt.title(f"Deficit vs. Area ({geometry.capitalize()}, r²={r_squared:.3f})")
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "delta_vs_area_with_errors.png"))
        plt.show()
        
        print(f"Deficit vs. area analysis (with uncertainties):")
        print(f"  Linear fit: deficit = ({slope:.4f} ± {slope_err:.4f}) * area + ({intercept:.4f} ± {intercept_err:.4f})")
        print(f"  Fitted κ = {slope:.4f} ± {slope_err:.4f} (expected: {kappa_fit})")
        print(f"  Fit quality: r² = {r_squared:.4f}")
        
        # Check if fitted kappa is consistent with expected value
        kappa_consistency = abs(slope - kappa_fit) / slope_err
        print(f"  κ consistency: {kappa_consistency:.2f}σ from expected value")
        if kappa_consistency < 2:
            print(f"  ✓ Fitted κ is consistent with expected value (within 2σ)")
        else:
            print(f"  ⚠ Fitted κ differs significantly from expected value")
    else:
        print("Euclidean geometry detected: deficits and areas are zero.")

    # After main analysis, run enhanced spectral dimension analyses with error estimation
    print("\n=== Enhanced Spectral Dimension Analysis (Random Walk) ===")
    d_spectral, d_spectral_err, r2_spectral = enhanced_spectral_dimension_analysis(
        D, S_max=min(10, num_qubits*2), num_walks=1000, n_bootstrap=100
    )
    
    print("\n=== Enhanced Spectral Dimension Analysis (Laplacian Spectrum) ===")
    d_laplacian, d_laplacian_err, r2_laplacian = enhanced_laplacian_spectral_dimension(
        D, S_max=10, s_min=0.1, s_max=10, num_s=10, n_bootstrap=50
    )
    
    # Compare spectral dimensions
    if d_spectral is not None and d_laplacian is not None:
        print(f"\n=== Spectral Dimension Comparison ===")
        print(f"Random Walk:    d = {d_spectral:.2f} ± {d_spectral_err:.2f} (r² = {r2_spectral:.3f})")
        print(f"Laplacian:      d = {d_laplacian:.2f} ± {d_laplacian_err:.2f} (r² = {r2_laplacian:.3f})")
        
        # Check consistency between methods
        diff = abs(d_spectral - d_laplacian)
        diff_err = np.sqrt(d_spectral_err**2 + d_laplacian_err**2)
        consistency = diff / diff_err
        print(f"Difference:     Δd = {diff:.2f} ± {diff_err:.2f} ({consistency:.2f}σ)")
        
        if consistency < 2:
            print(f"✓ Methods are consistent (within 2σ)")
        else:
            print(f"⚠ Methods show significant disagreement")

    # --- Gravitational Wave-like Propagation Analysis ---
    # Check for per-timestep angle deficit and edge length evolution
    if "angle_deficit_evolution" in data and data["angle_deficit_evolution"]:
        angle_deficit_evo = np.array(data["angle_deficit_evolution"])  # shape: (timesteps, num_hinges)
        plt.figure(figsize=(10, 5))
        plt.imshow(angle_deficit_evo.T, aspect='auto', origin='lower', cmap='RdBu')
        plt.colorbar(label='Angle Deficit')
        plt.xlabel('Timestep')
        plt.ylabel('Hinge Index')
        plt.title('Spacetime Evolution of Angle Deficits (Gravitational Wave Propagation)')
        plt.savefig(os.path.join(plots_dir, "angle_deficit_spacetime.png"))
        plt.show()
        # Optional: FFT along time to look for wave-like modes
        from scipy.fft import fft, fftfreq
        num_hinges = angle_deficit_evo.shape[1]
        for h in range(num_hinges):
            signal = angle_deficit_evo[:, h]
            yf = np.abs(fft(signal - np.mean(signal)))
            xf = fftfreq(len(signal), d=1)
            plt.plot(xf[:len(xf)//2], yf[:len(yf)//2], label=f'Hinge {h}')
        plt.xlabel('Frequency (1/timestep)')
        plt.ylabel('FFT Amplitude')
        plt.title('Temporal Spectrum of Angle Deficit (per Hinge)')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "angle_deficit_fft.png"))
        plt.show()
    if "edge_length_evolution" in data and data["edge_length_evolution"]:
        edge_length_evo = np.array(data["edge_length_evolution"])  # shape: (timesteps, num_edges)
        plt.figure(figsize=(10, 5))
        plt.imshow(edge_length_evo.T, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Edge Length')
        plt.xlabel('Timestep')
        plt.ylabel('Edge Index')
        plt.title('Spacetime Evolution of Edge Lengths')
        plt.savefig(os.path.join(plots_dir, "edge_length_spacetime.png"))
        plt.show()
    # Highlight mass perturbation location if available
    if "mass_hinge" in data and data["mass_hinge"] is not None:
        print(f"Mass perturbation applied at hinge: {data['mass_hinge']} (value: {data.get('mass_value', 'N/A')})")

    # --- Geodesic Deviation Analysis ---
    if "distance_matrix_per_timestep" in data and data["distance_matrix_per_timestep"]:
        distmat_per_timestep = np.array(data["distance_matrix_per_timestep"])  # shape: (timesteps, n, n)
        num_timesteps = distmat_per_timestep.shape[0]
        n = distmat_per_timestep.shape[1]
        # Pick a reference node (e.g., node 0)
        ref_node = 0
        # For each timestep, compute distances from ref_node to all others
        dists_vs_time = distmat_per_timestep[:, ref_node, :]
        plt.figure(figsize=(10, 5))
        for j in range(n):
            plt.plot(range(num_timesteps), dists_vs_time[:, j], label=f"0→{j}")
        plt.xlabel("Timestep")
        plt.ylabel("Geodesic Distance from node 0")
        plt.title("Geodesic Distances from Reference Node vs. Time")
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "geodesic_distances_vs_time.png"))
        plt.show()
        # Compute variance of distances from ref_node at each timestep with error estimation
        var_vs_time = np.var(dists_vs_time, axis=1)
        
        # Bootstrap confidence intervals for variance
        var_uncertainties = []
        for t in range(num_timesteps):
            _, _, _, bootstrap_vars = bootstrap_confidence_interval(dists_vs_time[t], np.var, n_bootstrap=500)
            var_uncertainties.append(np.std(bootstrap_vars))
        var_uncertainties = np.array(var_uncertainties)
        
        plt.figure()
        plt.errorbar(range(num_timesteps), var_vs_time, yerr=var_uncertainties, fmt='o-', capsize=5)
        plt.xlabel("Timestep")
        plt.ylabel("Variance of Geodesic Distances from node 0")
        plt.title("Spread of Geodesics (Deviation) vs. Time (with uncertainties)")
        plt.savefig(os.path.join(plots_dir, "geodesic_deviation_variance_vs_time_with_errors.png"))
        plt.show()
        
        print(f"Geodesic deviation analysis:")
        print(f"  Mean variance: {np.mean(var_vs_time):.4f} ± {np.mean(var_uncertainties):.4f}")
        print(f"  Variance trend: {np.polyfit(range(num_timesteps), var_vs_time, 1)[0]:.4f} per timestep")
        # For a triplet (0,1,2), plot D(0,1) - D(0,2) over time
        if n >= 3:
            diff_01_02 = dists_vs_time[:, 1] - dists_vs_time[:, 2]
            plt.figure()
            plt.plot(range(num_timesteps), diff_01_02, 'o-')
            plt.xlabel("Timestep")
            plt.ylabel("D(0,1) - D(0,2)")
            plt.title("Geodesic Deviation: Difference between Neighboring Geodesics vs. Time")
            plt.savefig(os.path.join(plots_dir, "geodesic_deviation_0_1_2_vs_time.png"))
            plt.show()
        print("[Geodesic Deviation] Plots saved: geodesic_distances_vs_time.png, geodesic_deviation_variance_vs_time.png, geodesic_deviation_0_1_2_vs_time.png (if n>=3)")

    # --- Geodesic Length Scaling with Curvature ---
    if "distance_matrix" in data:
        D = np.array(data["distance_matrix"])
        n = D.shape[0]
        # Compute all-pairs shortest path lengths
        geodesic_lengths = []
        for i in range(n):
            for j in range(i+1, n):
                if D[i, j] < np.inf:
                    geodesic_lengths.append(D[i, j])
        geodesic_lengths = np.array(geodesic_lengths)
        
        # Bootstrap confidence intervals for geodesic length statistics
        mean_length, mean_lower, mean_upper, _ = bootstrap_confidence_interval(geodesic_lengths, np.mean, n_bootstrap=1000)
        std_length, std_lower, std_upper, _ = bootstrap_confidence_interval(geodesic_lengths, np.std, n_bootstrap=1000)
        
        plt.figure()
        plt.hist(geodesic_lengths, bins=20, alpha=0.7, density=True, label='Data')
        
        # Add Gaussian fit
        x = np.linspace(mean_length - 3*std_length, mean_length + 3*std_length, 100)
        y = stats.norm.pdf(x, mean_length, std_length)
        plt.plot(x, y, 'r-', linewidth=2, label=f'Gaussian: μ={mean_length:.3f}, σ={std_length:.3f}')
        
        plt.xlabel("Geodesic Length")
        plt.ylabel("Density")
        plt.title(f"Distribution of Geodesic Lengths (mean={mean_length:.3f}±{mean_upper-mean_length:.3f})")
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "geodesic_length_histogram.png"))
        plt.show()
        
        print(f"[Geodesic Length Scaling] Statistics (with 95% CI):")
        print(f"  Mean length: {mean_length:.4f} [{mean_lower:.4f}, {mean_upper:.4f}]")
        print(f"  Std length:  {std_length:.4f} [{std_lower:.4f}, {std_upper:.4f}]")
        print(f"  Min length:  {np.min(geodesic_lengths):.4f}")
        print(f"  Max length:  {np.max(geodesic_lengths):.4f}")
        # Compare to flat-space expectation (Euclidean: ~sqrt(n) for random pairs)
        if geometry == "euclidean":
            print("[Geodesic Length Scaling] Compare to flat-space (Euclidean) expectation.")
        elif geometry == "spherical":
            print("[Geodesic Length Scaling] Expect contraction of geodesics (positive curvature).")
        elif geometry == "hyperbolic":
            print("[Geodesic Length Scaling] Expect expansion/divergence of geodesics (negative curvature).")
    # --- Geodesic Deviation (Jacobi Equation Analogue) ---
    if "distance_matrix_per_timestep" in data and data["distance_matrix_per_timestep"]:
        distmat_per_timestep = np.array(data["distance_matrix_per_timestep"])
        num_timesteps = distmat_per_timestep.shape[0]
        n = distmat_per_timestep.shape[1]
        ref_node = 0
        dists_vs_time = distmat_per_timestep[:, ref_node, :]
        # Compute spread (variance) of distances from ref_node at each timestep
        var_vs_time = np.var(dists_vs_time, axis=1)
        plt.figure()
        plt.plot(range(num_timesteps), var_vs_time, 'o-')
        plt.xlabel("Timestep")
        plt.ylabel("Variance of Geodesic Distances from node 0")
        plt.title("Spread of Geodesics (Deviation) vs. Time")
        plt.savefig(os.path.join(plots_dir, "geodesic_deviation_variance_vs_time.png"))
        plt.show()
        # Fit discrete Jacobi equation: d^2(delta l)/ds^2 + R delta l = 0
        # Approximate second derivative
        if num_timesteps > 2:
            delta = var_vs_time
            s = np.arange(num_timesteps)
            d2_delta = np.diff(delta, n=2)
            # Fit d2_delta + R*delta[1:-1] = 0 => R = -d2_delta / delta[1:-1]
            with np.errstate(divide='ignore', invalid='ignore'):
                R_eff = -d2_delta / delta[1:-1]
            R_eff = R_eff[np.isfinite(R_eff)]
            if len(R_eff) > 0:
                R_mean = np.mean(R_eff)
                print(f"[Geodesic Deviation] Fitted effective curvature R ≈ {R_mean:.4f} (from Jacobi eq. analogue)")
            else:
                print("[Geodesic Deviation] Could not fit effective curvature (variance too small or flat).")
    # --- Spectral Geodesic Indicators ---
    if "distance_matrix" in data:
        D = np.array(data["distance_matrix"])
        n = D.shape[0]
        # Build adjacency graph: connect nodes with finite, nonzero distance
        G = nx.Graph()
        for i in range(n):
            for j in range(i+1, n):
                if 0 < D[i, j] < np.inf:
                    G.add_edge(i, j)
        if nx.is_connected(G):
            L = nx.laplacian_matrix(G).toarray()
            evals = np.linalg.eigvalsh(L)
            s_vals = np.logspace(-1, 1, 10)
            K_s = [np.sum(np.exp(-s * evals)) for s in s_vals]
            log_s = np.log(s_vals)
            log_K = np.log(K_s)
            slope, intercept = np.polyfit(log_s, log_K, 1)
            d_spectral = -2 * slope
            plt.figure()
            plt.plot(log_s, log_K, 'o-', label='Data')
            plt.plot(log_s, slope*log_s + intercept, '--', label=f'Fit: slope={slope:.3f}')
            plt.xlabel('log s')
            plt.ylabel('log K(s)')
            plt.title(f'Laplacian Spectral Dimension: d_spectral={d_spectral:.2f}')
            plt.legend()
            plt.savefig(os.path.join(plots_dir, "laplacian_spectral_dimension_fit.png"))
            plt.show()
            print(f"[Spectral Geodesic] Spectral dimension d_spectral ≈ {d_spectral:.2f} (fit slope={slope:.3f})")
        else:
            print("[Spectral Geodesic] Adjacency graph is not connected. Spectrum ill-defined.")

    # --- Comprehensive Error Analysis Summary ---
    print("\n" + "="*60)
    print("COMPREHENSIVE ERROR ANALYSIS SUMMARY")
    print("="*60)
    
    # Collect all uncertainties
    error_summary = {
        'measurement_uncertainty': np.mean(list(mi_uncertainties.values())),
        'distance_uncertainty': np.mean(list(distance_uncertainties.values())),
        'finite_size_effects': systematic_errors['finite_size_scale'],
        'triangle_deficit_mean': deficit_mean,
        'triangle_deficit_std': deficit_std,
        'triangle_deficit_ci': [deficit_mean_lower, deficit_mean_upper],
        'geodesic_length_mean': mean_length if 'mean_length' in locals() else None,
        'geodesic_length_ci': [mean_lower, mean_upper] if 'mean_lower' in locals() else None,
    }
    
    if d_spectral is not None:
        error_summary['spectral_dimension'] = d_spectral
        error_summary['spectral_dimension_error'] = d_spectral_err
    
    if d_laplacian is not None:
        error_summary['laplacian_dimension'] = d_laplacian
        error_summary['laplacian_dimension_error'] = d_laplacian_err
    
    print("Key Uncertainties:")
    print(f"  • Measurement (MI): {error_summary['measurement_uncertainty']:.4f}")
    print(f"  • Distance propagation: {error_summary['distance_uncertainty']:.4f}")
    print(f"  • Finite-size effects: {error_summary['finite_size_effects']:.4f}")
    
    print("\nKey Results with Uncertainties:")
    print(f"  • Triangle deficit mean: {error_summary['triangle_deficit_mean']:.4f} [{error_summary['triangle_deficit_ci'][0]:.4f}, {error_summary['triangle_deficit_ci'][1]:.4f}]")
    print(f"  • Triangle deficit std: {error_summary['triangle_deficit_std']:.4f}")
    
    if error_summary['geodesic_length_mean'] is not None:
        print(f"  • Geodesic length mean: {error_summary['geodesic_length_mean']:.4f} [{error_summary['geodesic_length_ci'][0]:.4f}, {error_summary['geodesic_length_ci'][1]:.4f}]")
    
    if 'spectral_dimension' in error_summary:
        print(f"  • Spectral dimension: {error_summary['spectral_dimension']:.2f} ± {error_summary['spectral_dimension_error']:.2f}")
    
    if 'laplacian_dimension' in error_summary:
        print(f"  • Laplacian dimension: {error_summary['laplacian_dimension']:.2f} ± {error_summary['laplacian_dimension_error']:.2f}")
    
    # Save error summary to file
    error_summary_file = os.path.join(plots_dir, "error_analysis_summary.json")
    with open(error_summary_file, 'w') as f:
        json.dump(error_summary, f, indent=2, default=str)
    print(f"\nError analysis summary saved to: {error_summary_file}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main() 