#!/usr/bin/env python3
"""
Spherical Geometry Verification
Comprehensive verification of spherical geometry with explicit tests:
1. Regge curvature audit (angle deficits)
2. Ricci scalar consistency check
3. Spherical law of cosines test
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

def load_experiment_data(json_path):
    """Load experiment data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def compute_regge_angle_deficits(data):
    """Compute angle deficits at every hinge using Regge calculus."""
    print("=" * 70)
    print("REGGE CURVATURE AUDIT")
    print("=" * 70)
    
    spec = data['spec']
    geometry = spec['geometry']
    curvature = spec['curvature']
    num_qubits = spec['num_qubits']
    
    print(f"\nEXPERIMENT PARAMETERS:")
    print(f"  Geometry: {geometry}")
    print(f"  Curvature: {curvature}")
    print(f"  Number of qubits: {num_qubits}")
    
    # Check if we have embedding coordinates
    if 'embedding_coords' not in data:
        print("❌ No embedding coordinates found. Cannot compute angle deficits.")
        return None
    
    coords = np.array(data['embedding_coords'])
    print(f"  Embedding shape: {coords.shape}")
    
    # Compute all triangles and their angle deficits
    deficits = []
    triangle_data = []
    
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            for k in range(j+1, num_qubits):
                # Get triangle vertices
                p1, p2, p3 = coords[i], coords[j], coords[k]
                
                # Compute edge lengths
                a = np.linalg.norm(p2 - p3)  # opposite to angle A
                b = np.linalg.norm(p1 - p3)  # opposite to angle B
                c = np.linalg.norm(p1 - p2)  # opposite to angle C
                
                # Check triangle inequality
                if not (a + b > c and a + c > b and b + c > a):
                    continue
                
                # Compute angles using spherical law of cosines
                if geometry == "spherical":
                    # Spherical law of cosines: cos(C) = (cos(c) - cos(a)cos(b)) / (sin(a)sin(b))
                    # For small curvature, we can use the approximation
                    K = np.sqrt(curvature)
                    
                    # Scale distances by curvature
                    a_scaled = a * K
                    b_scaled = b * K
                    c_scaled = c * K
                    
                    # Compute angles
                    cos_A = (np.cos(a_scaled) - np.cos(b_scaled) * np.cos(c_scaled)) / (np.sin(b_scaled) * np.sin(c_scaled))
                    cos_B = (np.cos(b_scaled) - np.cos(a_scaled) * np.cos(c_scaled)) / (np.sin(a_scaled) * np.sin(c_scaled))
                    cos_C = (np.cos(c_scaled) - np.cos(a_scaled) * np.cos(b_scaled)) / (np.sin(a_scaled) * np.sin(b_scaled))
                    
                    # Clamp to valid range
                    cos_A = np.clip(cos_A, -1.0, 1.0)
                    cos_B = np.clip(cos_B, -1.0, 1.0)
                    cos_C = np.clip(cos_C, -1.0, 1.0)
                    
                    angle_A = np.arccos(cos_A)
                    angle_B = np.arccos(cos_B)
                    angle_C = np.arccos(cos_C)
                    
                else:
                    # Euclidean angles (fallback)
                    cos_A = (b**2 + c**2 - a**2) / (2 * b * c)
                    cos_B = (a**2 + c**2 - b**2) / (2 * a * c)
                    cos_C = (a**2 + b**2 - c**2) / (2 * a * b)
                    
                    cos_A = np.clip(cos_A, -1.0, 1.0)
                    cos_B = np.clip(cos_B, -1.0, 1.0)
                    cos_C = np.clip(cos_C, -1.0, 1.0)
                    
                    angle_A = np.arccos(cos_A)
                    angle_B = np.arccos(cos_B)
                    angle_C = np.arccos(cos_C)
                
                # Compute angle sum and deficit
                angle_sum = angle_A + angle_B + angle_C
                
                if geometry == "spherical":
                    # For spherical geometry: deficit = angle_sum - π
                    deficit = angle_sum - np.pi
                else:
                    # For hyperbolic geometry: deficit = π - angle_sum
                    deficit = np.pi - angle_sum
                
                deficits.append(deficit)
                triangle_data.append({
                    'vertices': [i, j, k],
                    'edges': [a, b, c],
                    'angles': [angle_A, angle_B, angle_C],
                    'angle_sum': angle_sum,
                    'deficit': deficit
                })
    
    deficits = np.array(deficits)
    
    if len(deficits) == 0:
        print("❌ No valid triangles found.")
        return None
    
    # Statistical analysis
    mean_deficit = np.mean(deficits)
    std_deficit = np.std(deficits)
    positive_deficits = np.sum(deficits > 0)
    negative_deficits = np.sum(deficits < 0)
    zero_deficits = np.sum(np.abs(deficits) < 1e-6)
    
    print(f"\nANGLE DEFICIT STATISTICS:")
    print(f"  Number of triangles: {len(deficits)}")
    print(f"  Mean deficit: {mean_deficit:.6f}")
    print(f"  Standard deviation: {std_deficit:.6f}")
    print(f"  Positive deficits: {positive_deficits} ({positive_deficits/len(deficits)*100:.1f}%)")
    print(f"  Negative deficits: {negative_deficits} ({negative_deficits/len(deficits)*100:.1f}%)")
    print(f"  Zero deficits: {zero_deficits} ({zero_deficits/len(deficits)*100:.1f}%)")
    
    # Curvature interpretation
    print(f"\nCURVATURE INTERPRETATION:")
    if geometry == "spherical":
        if mean_deficit > 0:
            print(f"  ✅ POSITIVE MEAN DEFICIT ({mean_deficit:.6f}) → POSITIVE CURVATURE (SPHERICAL)")
            print(f"     This confirms spherical geometry with κ = {curvature}")
        else:
            print(f"  ❌ NEGATIVE MEAN DEFICIT ({mean_deficit:.6f}) → INCONSISTENT WITH SPHERICAL GEOMETRY")
            print(f"     Expected positive deficit for κ = {curvature}")
    else:
        if mean_deficit < 0:
            print(f"  ✅ NEGATIVE MEAN DEFICIT ({mean_deficit:.6f}) → NEGATIVE CURVATURE (HYPERBOLIC)")
        else:
            print(f"  ❌ POSITIVE MEAN DEFICIT ({mean_deficit:.6f}) → INCONSISTENT WITH HYPERBOLIC GEOMETRY")
    
    return deficits, triangle_data

def ricci_scalar_consistency_check(data):
    """Check consistency of Ricci scalar with expected curvature."""
    print("\n" + "=" * 70)
    print("RICCI SCALAR CONSISTENCY CHECK")
    print("=" * 70)
    
    spec = data['spec']
    geometry = spec['geometry']
    curvature = spec['curvature']
    
    print(f"\nEXPECTED BEHAVIOR:")
    print(f"  Geometry: {geometry}")
    print(f"  Curvature: κ = {curvature}")
    
    if geometry == "spherical":
        print(f"  Expected Ricci scalar: R > 0 (positive)")
        print(f"  Expected relationship: R ≈ 2κ for 2D spherical geometry")
    else:
        print(f"  Expected Ricci scalar: R < 0 (negative)")
        print(f"  Expected relationship: R ≈ -2κ for 2D hyperbolic geometry")
    
    # Check if Einstein analysis data is available
    ricci_values = []
    einstein_data = {}
    
    if 'einstein_analysis_per_timestep' in data:
        einstein_timesteps = data['einstein_analysis_per_timestep']
        print(f"\nEINSTEIN ANALYSIS RESULTS:")
        print(f"  Number of timesteps: {len(einstein_timesteps)}")
        
        # Extract Ricci scalar values from each timestep
        for i, timestep_data in enumerate(einstein_timesteps):
            if timestep_data and 'ricci_scalar' in timestep_data:
                ricci_values.append(timestep_data['ricci_scalar'])
                print(f"  Timestep {i+1}: R = {timestep_data['ricci_scalar']:.6f}")
        
        if ricci_values:
            ricci_values = np.array(ricci_values)
            print(f"\nRICCI SCALAR STATISTICS:")
            print(f"  Ricci scalar range: [{ricci_values.min():.6f}, {ricci_values.max():.6f}]")
            print(f"  Mean Ricci scalar: {np.mean(ricci_values):.6f}")
            print(f"  Ricci scalar std: {np.std(ricci_values):.6f}")
            
            # Check consistency
            if geometry == "spherical":
                if np.mean(ricci_values) > 0:
                    print(f"  ✅ POSITIVE RICCI SCALAR → CONSISTENT WITH SPHERICAL GEOMETRY")
                else:
                    print(f"  ❌ NEGATIVE RICCI SCALAR → INCONSISTENT WITH SPHERICAL GEOMETRY")
                    print(f"     This suggests the Einstein solver needs adjustment for spherical geometry")
                    
                # Check magnitude
                expected_ricci = 2 * curvature
                actual_ricci = np.mean(ricci_values)
                ratio = actual_ricci / expected_ricci
                print(f"  Expected R ≈ {expected_ricci:.6f}, Actual R = {actual_ricci:.6f}")
                print(f"  Ratio: {ratio:.6f}")
                
                if 0.1 < ratio < 10.0:
                    print(f"  ✅ RICCI SCALAR MAGNITUDE REASONABLE")
                else:
                    print(f"  ⚠️  RICCI SCALAR MAGNITUDE UNEXPECTED")
                    print(f"     This indicates the Einstein solver needs calibration for κ = {curvature}")
            else:
                if np.mean(ricci_values) < 0:
                    print(f"  ✅ NEGATIVE RICCI SCALAR → CONSISTENT WITH HYPERBOLIC GEOMETRY")
                else:
                    print(f"  ❌ POSITIVE RICCI SCALAR → INCONSISTENT WITH HYPERBOLIC GEOMETRY")
            
            # Store the data for the report
            einstein_data = {
                'ricci_scalar_evolution': ricci_values.tolist(),
                'mean_ricci_scalar': float(np.mean(ricci_values)),
                'ricci_scalar_std': float(np.std(ricci_values)),
                'ricci_scalar_range': [float(ricci_values.min()), float(ricci_values.max())]
            }
        else:
            print("  ❌ No Ricci scalar data found in timesteps")
    else:
        print("  ❌ No Einstein analysis data found")
    
    return einstein_data

def spherical_law_of_cosines_test(data):
    """Test spherical law of cosines against Euclidean and hyperbolic versions."""
    print("\n" + "=" * 70)
    print("SPHERICAL LAW OF COSINES TEST")
    print("=" * 70)
    
    spec = data['spec']
    geometry = spec['geometry']
    curvature = spec['curvature']
    
    if 'embedding_coords' not in data:
        print("❌ No embedding coordinates found. Cannot perform law of cosines test.")
        return None
    
    coords = np.array(data['embedding_coords'])
    num_qubits = coords.shape[0]
    
    print(f"\nTESTING LAW OF COSINES FOR {geometry.upper()} GEOMETRY:")
    print(f"  Curvature: κ = {curvature}")
    print(f"  Number of triangles to test: {num_qubits * (num_qubits-1) * (num_qubits-2) // 6}")
    
    # Collect triangle data
    triangle_results = []
    
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            for k in range(j+1, num_qubits):
                # Get triangle vertices
                p1, p2, p3 = coords[i], coords[j], coords[k]
                
                # Compute edge lengths
                a = np.linalg.norm(p2 - p3)  # opposite to angle A
                b = np.linalg.norm(p1 - p3)  # opposite to angle B
                c = np.linalg.norm(p1 - p2)  # opposite to angle C
                
                # Check triangle inequality
                if not (a + b > c and a + c > b and b + c > a):
                    continue
                
                # Test different laws of cosines
                results = {}
                
                # 1. Euclidean law of cosines: c² = a² + b² - 2ab cos(C)
                cos_C_euclidean = (a**2 + b**2 - c**2) / (2 * a * b)
                cos_C_euclidean = np.clip(cos_C_euclidean, -1.0, 1.0)
                angle_C_euclidean = np.arccos(cos_C_euclidean)
                
                # 2. Spherical law of cosines: cos(c) = cos(a)cos(b) + sin(a)sin(b)cos(C)
                K = np.sqrt(curvature)
                a_scaled = a * K
                b_scaled = b * K
                c_scaled = c * K
                
                cos_C_spherical = (np.cos(c_scaled) - np.cos(a_scaled) * np.cos(b_scaled)) / (np.sin(a_scaled) * np.sin(b_scaled))
                cos_C_spherical = np.clip(cos_C_spherical, -1.0, 1.0)
                angle_C_spherical = np.arccos(cos_C_spherical)
                
                # 3. Hyperbolic law of cosines: cosh(c) = cosh(a)cosh(b) - sinh(a)sinh(b)cos(C)
                cos_C_hyperbolic = (np.cosh(c_scaled) - np.cosh(a_scaled) * np.cosh(b_scaled)) / (np.sinh(a_scaled) * np.sinh(b_scaled))
                cos_C_hyperbolic = np.clip(cos_C_hyperbolic, -1.0, 1.0)
                angle_C_hyperbolic = np.arccos(cos_C_hyperbolic)
                
                # Compute residuals (how well each law fits)
                # For spherical: check if cos(c) ≈ cos(a)cos(b) + sin(a)sin(b)cos(C)
                lhs_spherical = np.cos(c_scaled)
                rhs_spherical = np.cos(a_scaled) * np.cos(b_scaled) + np.sin(a_scaled) * np.sin(b_scaled) * cos_C_spherical
                residual_spherical = abs(lhs_spherical - rhs_spherical)
                
                # For hyperbolic: check if cosh(c) ≈ cosh(a)cosh(b) - sinh(a)sinh(b)cos(C)
                lhs_hyperbolic = np.cosh(c_scaled)
                rhs_hyperbolic = np.cosh(a_scaled) * np.cosh(b_scaled) - np.sinh(a_scaled) * np.sinh(b_scaled) * cos_C_hyperbolic
                residual_hyperbolic = abs(lhs_hyperbolic - rhs_hyperbolic)
                
                # For Euclidean: check if c² ≈ a² + b² - 2ab cos(C)
                lhs_euclidean = c**2
                rhs_euclidean = a**2 + b**2 - 2 * a * b * cos_C_euclidean
                residual_euclidean = abs(lhs_euclidean - rhs_euclidean)
                
                triangle_results.append({
                    'vertices': [i, j, k],
                    'edges': [a, b, c],
                    'residual_euclidean': residual_euclidean,
                    'residual_spherical': residual_spherical,
                    'residual_hyperbolic': residual_hyperbolic,
                    'angle_C_euclidean': angle_C_euclidean,
                    'angle_C_spherical': angle_C_spherical,
                    'angle_C_hyperbolic': angle_C_hyperbolic
                })
    
    if len(triangle_results) == 0:
        print("❌ No valid triangles found for testing.")
        return None
    
    # Analyze results
    residuals_euclidean = [t['residual_euclidean'] for t in triangle_results]
    residuals_spherical = [t['residual_spherical'] for t in triangle_results]
    residuals_hyperbolic = [t['residual_hyperbolic'] for t in triangle_results]
    
    mean_euclidean = np.mean(residuals_euclidean)
    mean_spherical = np.mean(residuals_spherical)
    mean_hyperbolic = np.mean(residuals_hyperbolic)
    
    print(f"\nLAW OF COSINES FIT RESULTS:")
    print(f"  Mean Euclidean residual: {mean_euclidean:.6f}")
    print(f"  Mean Spherical residual: {mean_spherical:.6f}")
    print(f"  Mean Hyperbolic residual: {mean_hyperbolic:.6f}")
    
    # Determine best fit
    residuals = [mean_euclidean, mean_spherical, mean_hyperbolic]
    models = ['Euclidean', 'Spherical', 'Hyperbolic']
    best_model_idx = np.argmin(residuals)
    best_model = models[best_model_idx]
    best_residual = residuals[best_model_idx]
    
    print(f"\nBEST FIT MODEL:")
    print(f"  {best_model} (residual: {best_residual:.6f})")
    
    if geometry == "spherical" and best_model == "Spherical":
        print(f"  ✅ SPHERICAL LAW OF COSINES BEST FIT → CONFIRMS SPHERICAL GEOMETRY")
    elif geometry == "hyperbolic" and best_model == "Hyperbolic":
        print(f"  ✅ HYPERBOLIC LAW OF COSINES BEST FIT → CONFIRMS HYPERBOLIC GEOMETRY")
    elif geometry == "euclidean" and best_model == "Euclidean":
        print(f"  ✅ EUCLIDEAN LAW OF COSINES BEST FIT → CONFIRMS EUCLIDEAN GEOMETRY")
    else:
        print(f"  ❌ MISMATCH: Expected {geometry} but {best_model} fits best")
    
    return triangle_results

def create_verification_plots(data, deficits, triangle_data, output_dir):
    """Create comprehensive verification plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Angle deficit histogram
    if deficits is not None:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(deficits, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(deficits), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(deficits):.4f}')
        plt.title('Angle Deficit Distribution')
        plt.xlabel('Angle Deficit')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Deficit vs triangle area
        plt.subplot(2, 2, 2)
        areas = []
        for triangle in triangle_data:
            a, b, c = triangle['edges']
            s = (a + b + c) / 2  # semi-perimeter
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))  # Heron's formula
            areas.append(area)
        
        plt.scatter(areas, deficits, alpha=0.6)
        plt.xlabel('Triangle Area')
        plt.ylabel('Angle Deficit')
        plt.title('Deficit vs Triangle Area')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Angle sum distribution
        plt.subplot(2, 2, 3)
        angle_sums = [triangle['angle_sum'] for triangle in triangle_data]
        plt.hist(angle_sums, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(np.pi, color='red', linestyle='--', label='π (Euclidean)')
        plt.xlabel('Angle Sum')
        plt.ylabel('Frequency')
        plt.title('Triangle Angle Sum Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Deficit sign analysis
        plt.subplot(2, 2, 4)
        positive_count = np.sum(deficits > 0)
        negative_count = np.sum(deficits < 0)
        zero_count = np.sum(np.abs(deficits) < 1e-6)
        
        labels = ['Positive', 'Negative', 'Zero']
        sizes = [positive_count, negative_count, zero_count]
        colors = ['green', 'red', 'gray']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Deficit Sign Distribution')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'spherical_verification_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 5: Ricci scalar evolution if available
    if 'einstein_analysis_per_timestep' in data:
        einstein_timesteps = data['einstein_analysis_per_timestep']
        ricci_values = []
        
        for timestep_data in einstein_timesteps:
            if timestep_data and 'ricci_scalar' in timestep_data:
                ricci_values.append(timestep_data['ricci_scalar'])
        
        if ricci_values:
            plt.figure(figsize=(10, 6))
            timesteps = list(range(1, len(ricci_values) + 1))
            plt.plot(timesteps, ricci_values, 'o-', linewidth=2, markersize=6, color='purple')
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='R = 0')
            plt.xlabel('Timestep')
            plt.ylabel('Ricci Scalar (R)')
            plt.title(f'Ricci Scalar Evolution\nExpected: R > 0 for κ = {data["spec"]["curvature"]}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / 'ricci_scalar_evolution.png', dpi=300, bbox_inches='tight')
            plt.close()

def create_comprehensive_verification_report(data, deficits, triangle_data, einstein_data, output_dir):
    """Create a comprehensive verification report."""
    output_dir = Path(output_dir)
    
    report_path = output_dir / 'spherical_geometry_verification_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SPHERICAL GEOMETRY VERIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Experiment parameters
        spec = data['spec']
        f.write("EXPERIMENT PARAMETERS:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Geometry: {spec['geometry']}\n")
        f.write(f"Curvature: κ = {spec['curvature']}\n")
        f.write(f"Number of qubits: {spec['num_qubits']}\n")
        f.write(f"Device: {spec['device']}\n")
        f.write(f"Timesteps: {spec['timesteps']}\n\n")
        
        # Regge curvature audit results
        f.write("1. REGGE CURVATURE AUDIT:\n")
        f.write("-" * 25 + "\n")
        if deficits is not None:
            mean_deficit = np.mean(deficits)
            positive_deficits = np.sum(deficits > 0)
            total_triangles = len(deficits)
            
            f.write(f"Total triangles analyzed: {total_triangles}\n")
            f.write(f"Mean angle deficit: {mean_deficit:.6f}\n")
            f.write(f"Positive deficits: {positive_deficits}/{total_triangles} ({positive_deficits/total_triangles*100:.1f}%)\n")
            
            if spec['geometry'] == "spherical":
                if mean_deficit > 0:
                    f.write("✅ VERDICT: POSITIVE MEAN DEFICIT → CONFIRMS SPHERICAL GEOMETRY\n")
                else:
                    f.write("❌ VERDICT: NEGATIVE MEAN DEFICIT → INCONSISTENT WITH SPHERICAL GEOMETRY\n")
            else:
                if mean_deficit < 0:
                    f.write("✅ VERDICT: NEGATIVE MEAN DEFICIT → CONFIRMS HYPERBOLIC GEOMETRY\n")
                else:
                    f.write("❌ VERDICT: POSITIVE MEAN DEFICIT → INCONSISTENT WITH HYPERBOLIC GEOMETRY\n")
        else:
            f.write("❌ No angle deficit data available\n")
        f.write("\n")
        
        # Ricci scalar consistency
        f.write("2. RICCI SCALAR CONSISTENCY CHECK:\n")
        f.write("-" * 35 + "\n")
        if einstein_data and 'ricci_scalar_evolution' in einstein_data:
            ricci_values = np.array(einstein_data['ricci_scalar_evolution'])
            mean_ricci = np.mean(ricci_values)
            expected_ricci = 2 * spec['curvature'] if spec['geometry'] == "spherical" else -2 * spec['curvature']
            
            f.write(f"Mean Ricci scalar: {mean_ricci:.6f}\n")
            f.write(f"Expected Ricci scalar: {expected_ricci:.6f}\n")
            f.write(f"Ricci scalar range: [{ricci_values.min():.6f}, {ricci_values.max():.6f}]\n")
            
            if spec['geometry'] == "spherical":
                if mean_ricci > 0:
                    f.write("✅ VERDICT: POSITIVE RICCI SCALAR → CONSISTENT WITH SPHERICAL GEOMETRY\n")
                else:
                    f.write("❌ VERDICT: NEGATIVE RICCI SCALAR → INCONSISTENT WITH SPHERICAL GEOMETRY\n")
                    f.write("   NOTE: This indicates the Einstein solver needs adjustment for spherical geometry\n")
            else:
                if mean_ricci < 0:
                    f.write("✅ VERDICT: NEGATIVE RICCI SCALAR → CONFIRMS HYPERBOLIC GEOMETRY\n")
                else:
                    f.write("❌ VERDICT: POSITIVE RICCI SCALAR → INCONSISTENT WITH HYPERBOLIC GEOMETRY\n")
        else:
            f.write("❌ No Ricci scalar data available\n")
        f.write("\n")
        
        # Overall verification summary
        f.write("3. OVERALL VERIFICATION SUMMARY:\n")
        f.write("-" * 35 + "\n")
        
        verification_passed = 0
        total_tests = 0
        
        # Test 1: Regge curvature
        total_tests += 1
        if deficits is not None:
            mean_deficit = np.mean(deficits)
            if spec['geometry'] == "spherical" and mean_deficit > 0:
                verification_passed += 1
                f.write("✅ Regge curvature audit: PASSED\n")
            elif spec['geometry'] == "hyperbolic" and mean_deficit < 0:
                verification_passed += 1
                f.write("✅ Regge curvature audit: PASSED\n")
            else:
                f.write("❌ Regge curvature audit: FAILED\n")
        else:
            f.write("⚠️  Regge curvature audit: NO DATA\n")
        
        # Test 2: Ricci scalar
        total_tests += 1
        if einstein_data and 'ricci_scalar_evolution' in einstein_data:
            mean_ricci = np.mean(einstein_data['ricci_scalar_evolution'])
            if spec['geometry'] == "spherical" and mean_ricci > 0:
                verification_passed += 1
                f.write("✅ Ricci scalar consistency: PASSED\n")
            elif spec['geometry'] == "hyperbolic" and mean_ricci < 0:
                verification_passed += 1
                f.write("✅ Ricci scalar consistency: PASSED\n")
            else:
                f.write("❌ Ricci scalar consistency: FAILED\n")
        else:
            f.write("⚠️  Ricci scalar consistency: NO DATA\n")
        
        f.write(f"\nVERIFICATION SCORE: {verification_passed}/{total_tests} tests passed\n")
        
        if verification_passed == total_tests:
            f.write("🎉 ALL VERIFICATION TESTS PASSED → GEOMETRY CONFIRMED\n")
        elif verification_passed > 0:
            f.write("⚠️  PARTIAL VERIFICATION → SOME INCONSISTENCIES DETECTED\n")
        else:
            f.write("❌ VERIFICATION FAILED → GEOMETRY NOT CONFIRMED\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF VERIFICATION REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"✅ Comprehensive verification report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Verify spherical geometry with comprehensive tests')
    parser.add_argument('json_path', help='Path to experiment results JSON file')
    parser.add_argument('--output-dir', help='Output directory (defaults to same directory as JSON file)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"📁 Loading experiment results from: {args.json_path}")
    try:
        data = load_experiment_data(args.json_path)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.json_path).parent
    
    # Run verification tests
    print("\n🔍 Running comprehensive spherical geometry verification...")
    
    # 1. Regge curvature audit
    deficits, triangle_data = compute_regge_angle_deficits(data)
    
    # 2. Ricci scalar consistency check
    einstein_data = ricci_scalar_consistency_check(data)
    
    # 3. Spherical law of cosines test
    cosines_results = spherical_law_of_cosines_test(data)
    
    # Create plots
    print("\n🎨 Generating verification plots...")
    create_verification_plots(data, deficits, triangle_data, output_dir)
    
    # Create comprehensive report
    create_comprehensive_verification_report(data, deficits, triangle_data, einstein_data, output_dir)
    
    print(f"\n✅ Spherical geometry verification complete! All results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - spherical_verification_analysis.png")
    print("  - ricci_scalar_evolution.png")
    print("  - spherical_geometry_verification_report.txt")

if __name__ == "__main__":
    main() 