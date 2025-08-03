#!/usr/bin/env python3
"""
Run comprehensive analysis on both experiment instances
"""

import os
import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.append('src')

def run_analysis_on_instance(instance_path):
    """Run comprehensive analysis on a single instance"""
    print(f"\n🔬 Analyzing: {instance_path}")
    print("=" * 60)
    
    # List of analysis scripts to run
    analysis_scripts = [
        "tools/comprehensive_quantum_geometry_analysis.py",
        "tools/emergent_spacetime_analysis.py", 
        "tools/ctc_analysis.py",
        "tools/causal_asymmetry_detection.py",
        "tools/page_curve_geometric_analysis.py"
    ]
    
    results = {}
    
    for script in analysis_scripts:
        if os.path.exists(script):
            print(f"\n📊 Running: {script}")
            try:
                # Run the analysis script
                result = subprocess.run([
                    sys.executable, script, instance_path
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print(f"✅ {script} completed successfully")
                    results[script] = "SUCCESS"
                else:
                    print(f"❌ {script} failed: {result.stderr}")
                    results[script] = f"FAILED: {result.stderr}"
                    
            except subprocess.TimeoutExpired:
                print(f"⏰ {script} timed out")
                results[script] = "TIMEOUT"
            except Exception as e:
                print(f"❌ {script} error: {e}")
                results[script] = f"ERROR: {e}"
        else:
            print(f"⚠️  Script not found: {script}")
            results[script] = "NOT_FOUND"
    
    return results

def main():
    """Main function to analyze both instances"""
    print("🚀 Starting Comprehensive Analysis of Both Experiment Instances")
    print("=" * 80)
    
    # Define the two instance paths
    instance1 = "experiment_logs/custom_curvature_experiment/instance_20250801_151819"
    instance2 = "experiment_logs/custom_curvature_experiment/instance_20250731_012542"
    
    # Check if instances exist
    if not os.path.exists(instance1):
        print(f"❌ Instance 1 not found: {instance1}")
        return
    
    if not os.path.exists(instance2):
        print(f"❌ Instance 2 not found: {instance2}")
        return
    
    # Run analysis on both instances
    print(f"\n📁 Instance 1: {instance1}")
    results1 = run_analysis_on_instance(instance1)
    
    print(f"\n📁 Instance 2: {instance2}")
    results2 = run_analysis_on_instance(instance2)
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\nInstance 1 ({instance1}):")
    for script, result in results1.items():
        status = "✅" if result == "SUCCESS" else "❌"
        print(f"  {status} {os.path.basename(script)}: {result}")
    
    print(f"\nInstance 2 ({instance2}):")
    for script, result in results2.items():
        status = "✅" if result == "SUCCESS" else "❌"
        print(f"  {status} {os.path.basename(script)}: {result}")
    
    print(f"\n🎉 Analysis complete! Results saved to respective instance directories.")

if __name__ == "__main__":
    main() 