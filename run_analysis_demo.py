#!/usr/bin/env python3
"""
Run analysis on both experiment instances and display results
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def run_analysis_script(script_path, instance_path):
    """Run an analysis script on an instance"""
    print(f"\n🔬 Running: {os.path.basename(script_path)}")
    print(f"📁 Target: {instance_path}")
    print("-" * 50)
    
    try:
        # Run the script
        result = subprocess.run([
            sys.executable, script_path, instance_path
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Analysis completed successfully")
            print("📊 Output:")
            print(result.stdout)
            return True
        else:
            print("❌ Analysis failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Analysis timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function"""
    print("🚀 COMPREHENSIVE EXPERIMENT ANALYSIS")
    print("=" * 60)
    
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
    
    # List of analysis scripts to try
    analysis_scripts = [
        "tools/comprehensive_quantum_geometry_analysis.py",
        "tools/ctc_analysis.py",
        "tools/causal_asymmetry_detection.py",
        "tools/page_curve_geometric_analysis.py"
    ]
    
    # Run analysis on both instances
    print(f"\n📁 INSTANCE 1: {instance1}")
    print("=" * 60)
    
    results1 = {}
    for script in analysis_scripts:
        if os.path.exists(script):
            success = run_analysis_script(script, instance1)
            results1[script] = success
        else:
            print(f"⚠️  Script not found: {script}")
            results1[script] = False
    
    print(f"\n📁 INSTANCE 2: {instance2}")
    print("=" * 60)
    
    results2 = {}
    for script in analysis_scripts:
        if os.path.exists(script):
            success = run_analysis_script(script, instance2)
            results2[script] = success
        else:
            print(f"⚠️  Script not found: {script}")
            results2[script] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\nInstance 1 ({instance1}):")
    for script, success in results1.items():
        status = "✅" if success else "❌"
        print(f"  {status} {os.path.basename(script)}")
    
    print(f"\nInstance 2 ({instance2}):")
    for script, success in results2.items():
        status = "✅" if success else "❌"
        print(f"  {status} {os.path.basename(script)}")
    
    print(f"\n🎉 Analysis complete! Results saved to respective instance directories.")

if __name__ == "__main__":
    main() 