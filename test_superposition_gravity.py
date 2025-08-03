#!/usr/bin/env python3
"""
Test script for the superposition gravity experiment.

This script demonstrates how to run the superposition of gravitational configurations
experiment using the custom_curvature_experiment.py file.

The experiment tests whether "mass" (curvature defect) can exist in a quantum superposition
in the emergent bulk, detectable via non-classical shadows in the mutual information pattern.
"""

import sys
import os
import subprocess
import json
import time

def run_superposition_gravity_experiment():
    """
    Run the superposition gravity experiment with different configurations.
    """
    print("🌌 SUPERPOSITION GRAVITY EXPERIMENT TEST")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        {
            "name": "Basic Superposition Test",
            "args": [
                "--superposition_gravity",
                "--num_qubits", "4",
                "--device", "simulator",
                "--shots", "1024",
                "--massive_bulk_mass_hinge", "2",
                "--massive_bulk_mass_value", "1.0",
                "--massless_bulk_mass_hinge", "None",
                "--massless_bulk_mass_value", "0.0",
                "--superposition_control_qubit", "0",
                "--superposition_phase", "0.0"
            ]
        },
        {
            "name": "Enhanced Interference Test",
            "args": [
                "--superposition_gravity",
                "--num_qubits", "6",
                "--device", "simulator", 
                "--shots", "2048",
                "--massive_bulk_mass_hinge", "1,2",
                "--massive_bulk_mass_value", "2.0",
                "--massless_bulk_mass_hinge", "None",
                "--massless_bulk_mass_value", "0.0",
                "--superposition_control_qubit", "0",
                "--superposition_phase", "0.5"
            ]
        },
        {
            "name": "Hardware Test (if available)",
            "args": [
                "--superposition_gravity",
                "--num_qubits", "4",
                "--device", "ibm_brisbane",
                "--shots", "1024",
                "--massive_bulk_mass_hinge", "2",
                "--massive_bulk_mass_value", "1.5",
                "--massless_bulk_mass_hinge", "None",
                "--massless_bulk_mass_value", "0.0",
                "--superposition_control_qubit", "0",
                "--superposition_phase", "0.0"
            ]
        }
    ]
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n🧪 Test {i}: {config['name']}")
        print("-" * 40)
        
        # Build command
        cmd = ["python", "src/experiments/custom_curvature_experiment.py"] + config["args"]
        
        print(f"Command: {' '.join(cmd)}")
        print(f"Running experiment...")
        
        try:
            # Run the experiment
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            end_time = time.time()
            
            if result.returncode == 0:
                print(f"✅ Experiment completed successfully!")
                print(f"   Runtime: {end_time - start_time:.2f} seconds")
                print(f"   Output: {result.stdout[-500:]}")  # Last 500 chars
                
                # Look for superposition results
                if "superposition_results.json" in result.stdout:
                    print(f"   📊 Superposition results saved")
                if "interference detected" in result.stdout.lower():
                    print(f"   🌟 INTERFERENCE DETECTED!")
                else:
                    print(f"   ⚠️  No significant interference detected")
                    
            else:
                print(f"❌ Experiment failed!")
                print(f"   Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Experiment timed out after 5 minutes")
        except Exception as e:
            print(f"❌ Error running experiment: {e}")
    
    print(f"\n🎯 SUPERPOSITION GRAVITY EXPERIMENT SUMMARY")
    print("=" * 60)
    print("The superposition gravity experiment tests:")
    print("1. Quantum superposition of massive and massless bulk configurations")
    print("2. Detection of non-classical shadows in mutual information patterns")
    print("3. Interference effects between gravitational configurations")
    print("4. Comparison with classical mixture of configurations")
    print("\nKey Features:")
    print("- Massive bulk: Localized curvature defect (mass hinge)")
    print("- Massless bulk: Flat or near-flat geometry")
    print("- Quantum superposition: Coherent superposition of both configurations")
    print("- Interference detection: Non-classical shadows in MI matrix")
    print("- Hardware compatibility: Works on both simulators and real quantum hardware")

def analyze_superposition_results():
    """
    Analyze existing superposition experiment results.
    """
    print(f"\n📊 ANALYZING EXISTING SUPERPOSITION RESULTS")
    print("=" * 60)
    
    # Look for superposition results in experiment logs
    experiment_logs_dir = "experiment_logs/custom_curvature_experiment"
    
    if not os.path.exists(experiment_logs_dir):
        print("No experiment logs found")
        return
    
    # Find recent experiments with superposition results
    import glob
    superposition_files = glob.glob(f"{experiment_logs_dir}/*/superposition_results.json")
    
    if not superposition_files:
        print("No superposition results found")
        return
    
    print(f"Found {len(superposition_files)} superposition result files:")
    
    for file_path in sorted(superposition_files, key=os.path.getmtime, reverse=True)[:3]:
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
            
            experiment_dir = os.path.dirname(file_path)
            experiment_name = os.path.basename(experiment_dir)
            
            print(f"\n📁 Experiment: {experiment_name}")
            print(f"   File: {file_path}")
            
            # Extract key results
            interference_analysis = results.get('interference_analysis', {})
            interference_detected = interference_analysis.get('interference_detected', False)
            max_interference = interference_analysis.get('interference_statistics', {}).get('max_interference', 0)
            num_significant_terms = interference_analysis.get('interference_statistics', {}).get('num_significant_terms', 0)
            
            print(f"   Interference detected: {'✅ YES' if interference_detected else '❌ NO'}")
            print(f"   Max interference: {max_interference:.6f}")
            print(f"   Significant terms: {num_significant_terms}")
            
            # Check for non-classical shadows
            non_classical_shadows = interference_analysis.get('non_classical_shadows', [])
            print(f"   Non-classical shadows: {len(non_classical_shadows)}")
            
            if non_classical_shadows:
                print(f"   🌟 QUANTUM EFFECTS DETECTED!")
                for shadow in non_classical_shadows[:3]:  # Show first 3
                    print(f"      Edge {shadow['edge']}: {shadow['interference_strength']:.6f}")
            
        except Exception as e:
            print(f"   Error reading {file_path}: {e}")

if __name__ == "__main__":
    print("🚀 SUPERPOSITION GRAVITY EXPERIMENT TEST SUITE")
    print("=" * 60)
    
    # Check if the main experiment file exists
    experiment_file = "src/experiments/custom_curvature_experiment.py"
    if not os.path.exists(experiment_file):
        print(f"❌ Experiment file not found: {experiment_file}")
        print("Please ensure you're running this from the project root directory")
        sys.exit(1)
    
    # Run the superposition experiments
    run_superposition_gravity_experiment()
    
    # Analyze existing results
    analyze_superposition_results()
    
    print(f"\n🎉 Superposition gravity experiment test completed!")
    print("Check the experiment_logs directory for detailed results.") 