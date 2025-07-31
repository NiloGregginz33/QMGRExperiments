#!/usr/bin/env python3
"""
Quick Experiment Analysis for "Undeniable" Evidence
Simplified analysis focusing on key issues and recommendations
"""

import numpy as np
import json
import os
import sys
from datetime import datetime

def analyze_experiment_file(file_path: str):
    """Quick analysis of experiment file."""
    print(f"🔍 Analyzing: {os.path.basename(file_path)}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    spec = data.get('spec', {})
    
    # Extract key parameters
    num_qubits = spec.get('num_qubits', 0)
    curvature = spec.get('curvature', 0.0)
    geometry = spec.get('geometry', 'unknown')
    timesteps = spec.get('timesteps', 0)
    device = spec.get('device', 'unknown')
    
    print(f"  📊 Parameters: {num_qubits} qubits, curvature={curvature}, {geometry} geometry, {timesteps} timesteps")
    
    # Analyze mutual information evolution
    mi_evolution = data.get('mutual_information_per_timestep', [])
    issues = []
    recommendations = []
    
    if mi_evolution:
        # Check if MI is static (all values identical)
        first_timestep = mi_evolution[0]
        is_static = all(
            all(first_timestep[key] == timestep[key] for key in first_timestep.keys())
            for timestep in mi_evolution
        )
        
        if is_static:
            issues.append("CRITICAL: Mutual information is static (no evolution)")
            recommendations.extend([
                "Increase entanglement strength and circuit depth",
                "Use more timesteps (8-12 minimum)",
                "Enable stronger charge injection"
            ])
    
    # Parameter analysis
    if num_qubits < 8:
        recommendations.append(f"Increase qubit count from {num_qubits} to 8-12")
    
    if curvature < 10.0:
        recommendations.append(f"Increase curvature from {curvature} to 10.0-20.0")
    
    if timesteps < 6:
        recommendations.append(f"Increase timesteps from {timesteps} to 8-12")
    
    if geometry == 'spherical':
        recommendations.append("Use hyperbolic geometry for stronger causal asymmetry")
    
    return {
        'spec': spec,
        'issues': issues,
        'recommendations': recommendations
    }

def generate_improved_command(spec):
    """Generate improved experiment command."""
    # Apply improvements
    improved_qubits = max(spec.get('num_qubits', 6), 8)
    improved_curvature = max(spec.get('curvature', 8.0), 15.0)
    improved_timesteps = max(spec.get('timesteps', 3), 8)
    improved_geometry = 'hyperbolic' if spec.get('geometry') == 'spherical' else spec.get('geometry', 'hyperbolic')
    
    cmd = f"""python run_experiment.py --args \\
  --num_qubits {improved_qubits} \\
  --curvature {improved_curvature} \\
  --timesteps {improved_timesteps} \\
  --geometry {improved_geometry} \\
  --charge_strength 5.0 \\
  --spin_strength 3.0 \\
  --entanglement_strength 8.0 \\
  --quantum_circuit_depth 20 \\
  --shots 8192 \\
  --strong_curvature --charge_injection --spin_injection --lorentzian --error_mitigation"""
    
    return cmd

def main():
    """Main analysis function."""
    if len(sys.argv) < 2:
        print("Usage: python quick_experiment_analysis.py <experiment_file>")
        return
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return
    
    # Analyze the experiment
    results = analyze_experiment_file(file_path)
    
    # Print results
    print("\n" + "="*60)
    print("🔍 QUICK EXPERIMENT ANALYSIS")
    print("="*60)
    
    print(f"\n📊 EXPERIMENT PARAMETERS:")
    spec = results['spec']
    print(f"  Qubits: {spec.get('num_qubits', 'N/A')}")
    print(f"  Curvature: {spec.get('curvature', 'N/A')}")
    print(f"  Geometry: {spec.get('geometry', 'N/A')}")
    print(f"  Timesteps: {spec.get('timesteps', 'N/A')}")
    print(f"  Device: {spec.get('device', 'N/A')}")
    
    print(f"\n🚨 IDENTIFIED ISSUES:")
    if results['issues']:
        for issue in results['issues']:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ No critical issues identified")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Generate improved command
    improved_cmd = generate_improved_command(spec)
    
    print(f"\n🚀 IMPROVED EXPERIMENT COMMAND:")
    print("  Use this command for 'undeniable' evidence:")
    print(improved_cmd)
    
    print(f"\n🎯 KEY IMPROVEMENTS:")
    improved_geometry = 'hyperbolic' if spec.get('geometry') == 'spherical' else spec.get('geometry', 'hyperbolic')
    print(f"  • Qubits: {spec.get('num_qubits', 6)} → {max(spec.get('num_qubits', 6), 8)}")
    print(f"  • Curvature: {spec.get('curvature', 8.0)} → {max(spec.get('curvature', 8.0), 15.0)}")
    print(f"  • Timesteps: {spec.get('timesteps', 3)} → {max(spec.get('timesteps', 3), 8)}")
    print(f"  • Geometry: {spec.get('geometry', 'spherical')} → {improved_geometry}")
    print(f"  • Charge strength: {spec.get('charge_strength', 2.5)} → 5.0")
    print(f"  • Entanglement strength: {spec.get('entanglement_strength', 0.35)} → 8.0")
    print(f"  • Circuit depth: {spec.get('quantum_circuit_depth', 12)} → 20")
    print(f"  • Shots: {spec.get('shots', 4096)} → 8192")

if __name__ == "__main__":
    main() 