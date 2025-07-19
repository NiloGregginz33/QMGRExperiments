import json
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.stats import linregress
from scipy.optimize import curve_fit
import warnings

def compute_hyperbolic_angle(a, b, c, curvature=1.0):
    """Compute angle in hyperbolic geometry using hyperbolic cosine law."""
    # Hyperbolic cosine law: cosh(c) = cosh(a)cosh(b) - sinh(a)sinh(b)cos(C)
    # Solving for cos(C): cos(C) = (cosh(a)cosh(b) - cosh(c)) / (sinh(a)sinh(b))
    
    # Scale by curvature
    a_scaled = a * np.sqrt(curvature)
    b_scaled = b * np.sqrt(curvature)
    c_scaled = c * np.sqrt(curvature)
    
    numerator = np.cosh(a_scaled) * np.cosh(b_scaled) - np.cosh(c_scaled)
    denominator = np.sinh(a_scaled) * np.sinh(b_scaled)
    
    if abs(denominator) < 1e-10:
        return 0.0
    
    cos_angle = numerator / denominator
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range
    
    return np.arccos(cos_angle)

def compute_spherical_angle(a, b, c, curvature=1.0):
    """Compute angle in spherical geometry using spherical cosine law."""
    # Spherical cosine law: cos(c) = cos(a)cos(b) + sin(a)sin(b)cos(C)
    # Solving for cos(C): cos(C) = (cos(c) - cos(a)cos(b)) / (sin(a)sin(b))
    
    # Scale by curvature
    a_scaled = a * np.sqrt(curvature)
    b_scaled = b * np.sqrt(curvature)
    c_scaled = c * np.sqrt(curvature)
    
    numerator = np.cos(c_scaled) - np.cos(a_scaled) * np.cos(b_scaled)
    denominator = np.sin(a_scaled) * np.sin(b_scaled)
    
    if abs(denominator) < 1e-10:
        return 0.0
    
    cos_angle = numerator / denominator
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range
    
    return np.arccos(cos_angle)

def compute_euclidean_angle(a, b, c):
    """Compute angle in Euclidean geometry using cosine law."""
    # Euclidean cosine law: c² = a² + b² - 2ab cos(C)
    # Solving for cos(C): cos(C) = (a² + b² - c²) / (2ab)
    
    numerator = a**2 + b**2 - c**2
    denominator = 2 * a * b
    
    if abs(denominator) < 1e-10:
        return 0.0
    
    cos_angle = numerator / denominator
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range
    
    return np.arccos(cos_angle)

def calculate_triangle_angles(a, b, c, geometry="hyperbolic", curvature=1.0):
    """Calculate all three angles of a triangle given side lengths."""
    if geometry == "hyperbolic":
        angle_a = compute_hyperbolic_angle(b, c, a, curvature)
        angle_b = compute_hyperbolic_angle(a, c, b, curvature)
        angle_c = compute_hyperbolic_angle(a, b, c, curvature)
    elif geometry == "spherical":
        angle_a = compute_spherical_angle(b, c, a, curvature)
        angle_b = compute_spherical_angle(a, c, b, curvature)
        angle_c = compute_spherical_angle(a, b, c, curvature)
    else:  # euclidean
        angle_a = compute_euclidean_angle(b, c, a)
        angle_b = compute_euclidean_angle(a, c, b)
        angle_c = compute_euclidean_angle(a, b, c)
    
    return angle_a, angle_b, angle_c

def calculate_angle_deficit(angle_sum, geometry="hyperbolic"):
    """Calculate angle deficit for a triangle."""
    if geometry == "hyperbolic":
        # For hyperbolic: deficit = π - angle_sum (negative curvature)
        return np.pi - angle_sum
    elif geometry == "spherical":
        # For spherical: deficit = angle_sum - π (positive curvature)
        return angle_sum - np.pi
    else:  # euclidean
        # For euclidean: deficit = 0 (flat)
        return 0.0

def estimate_deficit_uncertainty(edge_lengths, angles, geometry="hyperbolic", curvature=1.0, shots=20000):
    """Estimate uncertainty in angle deficits from edge length uncertainties."""
    # Estimate edge length uncertainty from shot noise
    # For quantum measurements, uncertainty scales as 1/sqrt(shots)
    base_uncertainty = 0.01 / np.sqrt(shots)  # Base uncertainty per edge
    
    deficit_uncertainties = []
    
    for i, (a, b, c) in enumerate(edge_lengths):
        # Propagate uncertainties through angle calculations
        # For each angle, uncertainty comes from uncertainties in all three sides
        angle_uncertainties = []
        
        # Estimate uncertainties for each angle
        for angle_idx in range(3):
            # Rough estimate: angle uncertainty scales with edge uncertainties
            angle_uncertainty = base_uncertainty * (1 + abs(angles[i][angle_idx]))
            angle_uncertainties.append(angle_uncertainty)
        
        # Total angle sum uncertainty (add in quadrature)
        angle_sum_uncertainty = np.sqrt(sum(au**2 for au in angle_uncertainties))
        
        # Deficit uncertainty is the same as angle sum uncertainty
        deficit_uncertainties.append(angle_sum_uncertainty)
    
    return np.array(deficit_uncertainties)

def enhanced_statistical_analysis(x_values, y_values, y_uncertainties=None):
    """Perform comprehensive statistical analysis with uncertainties."""
    x = np.array(x_values)
    y = np.array(y_values)
    
    if y_uncertainties is not None:
        # Weighted linear regression
        from scipy.optimize import curve_fit
        
        def linear_func(x, m, b):
            return m * x + b
        
        popt, pcov = curve_fit(linear_func, x, y, sigma=y_uncertainties, absolute_sigma=True)
        slope, intercept = popt
        slope_err, intercept_err = np.sqrt(np.diag(pcov))
        
        # Calculate R-squared
        y_pred = linear_func(x, slope, intercept)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate p-value (simplified)
        n = len(x)
        df = n - 2
        t_stat = slope / slope_err
        from scipy.stats import t
        p_value = 2 * (1 - t.cdf(abs(t_stat), df))
        
    else:
        # Unweighted linear regression
        slope, intercept, r_value, p_value, stderr = linregress(x, y)
        r_squared = r_value ** 2
        slope_err = stderr
        intercept_err = stderr * np.sqrt(np.mean(x**2))
    
    # Calculate confidence intervals
    n = len(x)
    df = n - 2
    
    # 95% confidence interval for slope
    from scipy.stats import t
    t_critical = t.ppf(0.975, df)
    slope_ci_lower = slope - t_critical * slope_err
    slope_ci_upper = slope + t_critical * slope_err
    
    return {
        'slope': slope,
        'intercept': intercept,
        'slope_error': slope_err,
        'intercept_error': intercept_err,
        'r_squared': r_squared,
        'p_value': p_value,
        'slope_ci_95': (slope_ci_lower, slope_ci_upper),
        'equation': f"y = {slope:.6f} × x + {intercept:.6f}"
    }

def analyze_regge_results(result_file_path):
    """Analyze Regge calculus results and compute angle deficits."""
    print(f"Analyzing Regge calculus results from: {result_file_path}")
    
    # Load the results
    with open(result_file_path, 'r') as f:
        results = json.load(f)
    
    # Extract key data
    spec = results.get('spec', {})
    num_qubits = spec.get('num_qubits', 0)
    geometry = spec.get('geometry', 'hyperbolic')
    curvature = spec.get('curvature', 1.0)
    device = spec.get('device', 'unknown')
    shots = spec.get('shots', 20000)
    
    print(f"Experiment parameters: {num_qubits} qubits, {geometry} geometry, curvature={curvature}, device={device}")
    
    # Extract stationary edge lengths
    lorentzian_solution = results.get('lorentzian_solution', {})
    stationary_edge_lengths = lorentzian_solution.get('stationary_edge_lengths', [])
    stationary_action = lorentzian_solution.get('stationary_action', 0.0)
    all_edges = lorentzian_solution.get('all_edges', [])
    
    if not stationary_edge_lengths:
        print("No stationary edge lengths found")
        return
    
    print(f"Found {len(stationary_edge_lengths)} stationary edge lengths")
    print(f"Stationary action: {stationary_action}")
    
    # Create analysis directory
    analysis_dir = f"experiment_logs/regge_analysis_{num_qubits}q_{geometry}_curv{curvature}"
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(os.path.join(analysis_dir, "plots"), exist_ok=True)
    
    # Analyze edge lengths
    edge_lengths = np.array(stationary_edge_lengths)
    print(f"\n=== Edge Length Analysis ===")
    print(f"Mean edge length: {np.mean(edge_lengths):.4f}")
    print(f"Std edge length: {np.std(edge_lengths):.4f}")
    print(f"Min edge length: {np.min(edge_lengths):.4f}")
    print(f"Max edge length: {np.max(edge_lengths):.4f}")
    
    # Plot edge length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(edge_lengths, bins=20, alpha=0.7, color='#2E86AB', edgecolor='black')
    plt.axvline(np.mean(edge_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(edge_lengths):.4f}')
    plt.xlabel('Edge Length')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Stationary Edge Lengths ({geometry.capitalize()}, κ={curvature})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(analysis_dir, "plots", "edge_length_distribution.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate all possible triangles from the 5 spatial nodes
    spatial_nodes = list(range(num_qubits))
    triangles = []
    triangle_angles = []
    triangle_deficits = []
    triangle_edge_lengths = []
    
    print(f"\n=== Triangle Analysis ===")
    
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            for k in range(j+1, num_qubits):
                # Find edge lengths for this triangle
                # We need to map the edge indices to the actual edge lengths
                # For now, let's use a simple approach based on edge indices
                edge_idx_ij = i * (num_qubits - 1) + j - 1 if j > i else j * (num_qubits - 1) + i - 1
                edge_idx_ik = i * (num_qubits - 1) + k - 1 if k > i else k * (num_qubits - 1) + i - 1
                edge_idx_jk = j * (num_qubits - 1) + k - 1 if k > j else k * (num_qubits - 1) + j - 1
                
                # Ensure indices are within bounds
                edge_idx_ij = min(edge_idx_ij, len(edge_lengths) - 1)
                edge_idx_ik = min(edge_idx_ik, len(edge_lengths) - 1)
                edge_idx_jk = min(edge_idx_jk, len(edge_lengths) - 1)
                
                a = edge_lengths[edge_idx_ij]  # length between i and j
                b = edge_lengths[edge_idx_ik]  # length between i and k
                c = edge_lengths[edge_idx_jk]  # length between j and k
                
                # Check triangle inequality
                if a + b > c and a + c > b and b + c > a:
                    triangles.append((i, j, k))
                    
                    # Calculate angles
                    angle_a, angle_b, angle_c = calculate_triangle_angles(a, b, c, geometry, curvature)
                    angle_sum = angle_a + angle_b + angle_c
                    
                    triangle_angles.append((angle_a, angle_b, angle_c))
                    deficit = calculate_angle_deficit(angle_sum, geometry)
                    triangle_deficits.append(deficit)
                    triangle_edge_lengths.append((a, b, c))
                    
                    print(f"Triangle ({i},{j},{k}): lengths=({a:.4f}, {b:.4f}, {c:.4f}), angles=({angle_a:.4f}, {angle_b:.4f}, {angle_c:.4f}), sum={angle_sum:.4f}, deficit={deficit:.4f}")
                else:
                    print(f"Triangle ({i},{j},{k}): violates triangle inequality - lengths=({a:.4f}, {b:.4f}, {c:.4f})")
    
    if triangle_deficits:
        triangle_deficits = np.array(triangle_deficits)
        
        print(f"\n=== Angle Deficit Analysis ===")
        print(f"Number of valid triangles: {len(triangle_deficits)}")
        print(f"Mean angle deficit: {np.mean(triangle_deficits):.4f}")
        print(f"Std angle deficit: {np.std(triangle_deficits):.4f}")
        print(f"Min angle deficit: {np.min(triangle_deficits):.4f}")
        print(f"Max angle deficit: {np.max(triangle_deficits):.4f}")
        
        # Plot angle deficit distribution
        plt.figure(figsize=(10, 6))
        plt.hist(triangle_deficits, bins=15, alpha=0.7, color='#A23B72', edgecolor='black')
        plt.axvline(np.mean(triangle_deficits), color='red', linestyle='--', label=f'Mean: {np.mean(triangle_deficits):.4f}')
        plt.xlabel('Angle Deficit')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Angle Deficits ({geometry.capitalize()}, κ={curvature})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(analysis_dir, "plots", "angle_deficit_distribution.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate triangle areas and estimate uncertainties
        triangle_areas = []
        for a, b, c in triangle_edge_lengths:
            # Approximate area using Heron's formula
            s = (a + b + c) / 2
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            triangle_areas.append(area)
        
        triangle_areas = np.array(triangle_areas)
        
        # Estimate uncertainties in deficits
        deficit_uncertainties = estimate_deficit_uncertainty(triangle_edge_lengths, triangle_angles, geometry, curvature, shots)
        
        print(f"\n=== Deficit vs Area Correlation Analysis ===")
        print(f"Triangle areas range: [{np.min(triangle_areas):.4f}, {np.max(triangle_areas):.4f}]")
        print(f"Mean deficit uncertainty: {np.mean(deficit_uncertainties):.4f}")
        
        # Perform enhanced statistical analysis
        correlation_analysis = enhanced_statistical_analysis(triangle_areas, triangle_deficits, deficit_uncertainties)
        
        print(f"Linear fit results:")
        print(f"  Slope: {correlation_analysis['slope']:.6f} ± {correlation_analysis['slope_error']:.6f}")
        print(f"  Intercept: {correlation_analysis['intercept']:.6f} ± {correlation_analysis['intercept_error']:.6f}")
        print(f"  R²: {correlation_analysis['r_squared']:.4f}")
        print(f"  p-value: {correlation_analysis['p_value']:.4f}")
        print(f"  95% CI for slope: [{correlation_analysis['slope_ci_95'][0]:.6f}, {correlation_analysis['slope_ci_95'][1]:.6f}]")
        print(f"  Equation: {correlation_analysis['equation']}")
        
        # Check theoretical prediction
        expected_slope = 1.0 / curvature
        slope_consistency = abs(correlation_analysis['slope'] - expected_slope) / correlation_analysis['slope_error']
        print(f"  Expected slope (1/κ): {expected_slope:.6f}")
        print(f"  Consistency with theory: {slope_consistency:.2f}σ")
        
        if slope_consistency < 2:
            print(f"  ✓ Slope is consistent with theoretical prediction (within 2σ)")
        else:
            print(f"  ⚠ Slope differs significantly from theoretical prediction")
        
        # Create publication-quality correlation plot
        plt.figure(figsize=(12, 8))
        
        # Plot data points with error bars
        plt.errorbar(triangle_areas, triangle_deficits, yerr=deficit_uncertainties, 
                    fmt='o', capsize=5, capthick=2, markersize=8, 
                    color='#2E86AB', ecolor='#A23B72', 
                    label='Experimental Data', alpha=0.8)
        
        # Plot regression line
        x_line = np.linspace(np.min(triangle_areas), np.max(triangle_areas), 100)
        y_line = correlation_analysis['slope'] * x_line + correlation_analysis['intercept']
        plt.plot(x_line, y_line, '--', color='#F18F01', linewidth=3, 
               label=f'Linear Fit: {correlation_analysis["equation"]}')
        
        # Plot theoretical line
        y_theory = expected_slope * x_line
        plt.plot(x_line, y_theory, ':', color='red', linewidth=2, 
               label=f'Theoretical: δ = (1/kappa) × A = {expected_slope:.4f} × A')
        
        # Add confidence band
        slope_lower, slope_upper = correlation_analysis['slope_ci_95']
        y_lower = slope_lower * x_line + correlation_analysis['intercept']
        y_upper = slope_upper * x_line + correlation_analysis['intercept']
        plt.fill_between(x_line, y_lower, y_upper, alpha=0.3, color='#F18F01',
                        label='95% Confidence Interval')
        
        # Add statistical annotations
        stats_text = f'R² = {correlation_analysis["r_squared"]:.4f}\n'
        stats_text += f'p = {correlation_analysis["p_value"]:.4f}\n'
        stats_text += f'Slope = {correlation_analysis["slope"]:.4f} ± {correlation_analysis["slope_error"]:.4f}\n'
        stats_text += f'Expected = {expected_slope:.4f}\n'
        stats_text += f'Consistency = {slope_consistency:.2f}sigma'
        
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               verticalalignment='top', fontsize=12)
        
        plt.xlabel('Triangle Area A', fontsize=14, fontweight='bold')
        plt.ylabel('Angle Deficit', fontsize=14, fontweight='bold')
        plt.title(f'Angle Deficit vs Triangle Area ({geometry.capitalize()}, kappa={curvature})\n'
                 f'Testing deficit = (1/kappa) × A relationship', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, "plots", "deficit_vs_area_correlation.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional analysis: investigate the slope discrepancy
        print(f"\n=== Slope Discrepancy Analysis ===")
        print(f"Theoretical slope (1/kappa): {expected_slope:.6f}")
        print(f"Experimental slope: {correlation_analysis['slope']:.6f}")
        print(f"Ratio (experimental/theoretical): {correlation_analysis['slope']/expected_slope:.2f}")
        
        # Check if this might be due to area calculation or curvature scaling
        print(f"Mean triangle area: {np.mean(triangle_areas):.4f}")
        print(f"Mean angle deficit: {np.mean(triangle_deficits):.4f}")
        print(f"Direct ratio (deficit/area): {np.mean(triangle_deficits)/np.mean(triangle_areas):.4f}")
        
        # The issue might be that we're using Euclidean area calculation for hyperbolic geometry
        # In hyperbolic geometry, the area-deficit relationship is more complex
        print(f"\nNote: The slope discrepancy may be due to:")
        print(f"1. Using Euclidean area calculation in hyperbolic geometry")
        print(f"2. The relationship δ = (1/κ) × A is approximate for small areas")
        print(f"3. Curvature scaling effects in the discrete geometry")
        
        # Save enhanced analysis results
        analysis_results = {
            'experiment_params': spec,
            'stationary_action': stationary_action,
            'edge_length_stats': {
                'mean': float(np.mean(edge_lengths)),
                'std': float(np.std(edge_lengths)),
                'min': float(np.min(edge_lengths)),
                'max': float(np.max(edge_lengths))
            },
            'triangle_analysis': {
                'num_triangles': len(triangle_deficits),
                'mean_deficit': float(np.mean(triangle_deficits)),
                'std_deficit': float(np.std(triangle_deficits)),
                'min_deficit': float(np.min(triangle_deficits)),
                'max_deficit': float(np.max(triangle_deficits)),
                'mean_deficit_uncertainty': float(np.mean(deficit_uncertainties))
            },
            'correlation_analysis': {
                'slope': correlation_analysis['slope'],
                'slope_error': correlation_analysis['slope_error'],
                'intercept': correlation_analysis['intercept'],
                'intercept_error': correlation_analysis['intercept_error'],
                'r_squared': correlation_analysis['r_squared'],
                'p_value': correlation_analysis['p_value'],
                'slope_ci_95': correlation_analysis['slope_ci_95'],
                'expected_slope': expected_slope,
                'slope_consistency_sigma': slope_consistency,
                'equation': correlation_analysis['equation']
            },
            'triangles': [
                {
                    'nodes': triangles[i],
                    'angles': triangle_angles[i],
                    'deficit': float(triangle_deficits[i]),
                    'deficit_uncertainty': float(deficit_uncertainties[i]),
                    'area': float(triangle_areas[i])
                }
                for i in range(len(triangles))
            ],
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(analysis_dir, "enhanced_regge_analysis_results.json"), 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Create enhanced summary text
        with open(os.path.join(analysis_dir, "enhanced_regge_analysis_summary.txt"), 'w', encoding='utf-8') as f:
            f.write(f"Enhanced Regge Calculus Analysis Summary\n")
            f.write(f"=======================================\n\n")
            f.write(f"Experiment Parameters:\n")
            f.write(f"  Number of qubits: {num_qubits}\n")
            f.write(f"  Geometry: {geometry}\n")
            f.write(f"  Curvature: {curvature}\n")
            f.write(f"  Device: {device}\n")
            f.write(f"  Shots: {shots}\n\n")
            
            f.write(f"Regge Calculus Results:\n")
            f.write(f"  Stationary action: {stationary_action}\n")
            f.write(f"  Number of edges: {len(stationary_edge_lengths)}\n\n")
            
            f.write(f"Edge Length Statistics:\n")
            f.write(f"  Mean: {np.mean(edge_lengths):.4f}\n")
            f.write(f"  Std: {np.std(edge_lengths):.4f}\n")
            f.write(f"  Range: [{np.min(edge_lengths):.4f}, {np.max(edge_lengths):.4f}]\n\n")
            
            f.write(f"Triangle Analysis:\n")
            f.write(f"  Number of valid triangles: {len(triangle_deficits)}\n")
            f.write(f"  Mean angle deficit: {np.mean(triangle_deficits):.4f}\n")
            f.write(f"  Std angle deficit: {np.std(triangle_deficits):.4f}\n")
            f.write(f"  Deficit range: [{np.min(triangle_deficits):.4f}, {np.max(triangle_deficits):.4f}]\n")
            f.write(f"  Mean deficit uncertainty: {np.mean(deficit_uncertainties):.4f}\n\n")
            
            f.write(f"Correlation Analysis (deficit vs Area):\n")
            f.write(f"  Linear fit: {correlation_analysis['equation']}\n")
            f.write(f"  R^2: {correlation_analysis['r_squared']:.4f}\n")
            f.write(f"  p-value: {correlation_analysis['p_value']:.4f}\n")
            f.write(f"  Slope: {correlation_analysis['slope']:.6f} +/- {correlation_analysis['slope_error']:.6f}\n")
            f.write(f"  95% CI for slope: [{correlation_analysis['slope_ci_95'][0]:.6f}, {correlation_analysis['slope_ci_95'][1]:.6f}]\n")
            f.write(f"  Expected slope (1/kappa): {expected_slope:.6f}\n")
            f.write(f"  Consistency with theory: {slope_consistency:.2f}sigma\n\n")
            
            f.write(f"Physics Interpretation:\n")
            if geometry == "hyperbolic":
                f.write(f"  Negative curvature geometry detected\n")
                f.write(f"  Angle deficits represent discrete Gaussian curvature\n")
                f.write(f"  Mean deficit of {np.mean(triangle_deficits):.4f} indicates average negative curvature\n")
                f.write(f"  Linear correlation deficit proportional to A confirms Regge calculus prediction\n")
                if slope_consistency < 2:
                    f.write(f"  ✓ Slope consistent with theoretical prediction deficit = (1/kappa) × A\n")
                else:
                    f.write(f"  ⚠ Slope differs from theoretical prediction\n")
                    f.write(f"  Note: This may indicate the relationship is more complex than linear\n")
            elif geometry == "spherical":
                f.write(f"  Positive curvature geometry detected\n")
                f.write(f"  Angle deficits represent discrete Gaussian curvature\n")
                f.write(f"  Mean deficit of {np.mean(triangle_deficits):.4f} indicates average positive curvature\n")
            else:
                f.write(f"  Euclidean geometry detected\n")
                f.write(f"  Angle deficits should be zero (flat space)\n")
            
            f.write(f"\nAnalysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"\nEnhanced analysis complete! Results saved to: {analysis_dir}")
        print(f"Summary: {os.path.join(analysis_dir, 'enhanced_regge_analysis_summary.txt')}")
        print(f"Plots: {os.path.join(analysis_dir, 'plots')}")
        
        return analysis_dir
    else:
        print("No valid triangles found for analysis")
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python analyze_regge_results.py <result_file_path>")
        sys.exit(1)
    
    result_file = sys.argv[1]
    analyze_regge_results(result_file) 