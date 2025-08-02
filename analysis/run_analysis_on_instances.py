#!/usr/bin/env python3
"""
Run comprehensive analysis on both experiment instances using existing analysis scripts
"""

import os
import sys
import subprocess
from pathlib import Path

def run_analysis_script(script_path, target_file, output_dir=None):
    """Run an analysis script on a target file"""
    print(f"\n🔬 Running: {os.path.basename(script_path)}")
    print(f"📁 Target: {os.path.basename(target_file)}")
    print("-" * 50)
    
    try:
        # Run the script
        cmd = [sys.executable, script_path, target_file]
        if output_dir:
            cmd.extend(['--output_dir', output_dir])
            
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
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
    """Main function to analyze both instances"""
    print("🚀 COMPREHENSIVE EXPERIMENT ANALYSIS")
    print("=" * 60)
    
    # Define the two main result files
    instance1_file = "experiment_logs/custom_curvature_experiment/instance_20250801_151819/results_n8_geomS_curv1_ibm_brisbane_HSXEL9.json"
    instance2_file = "experiment_logs/custom_curvature_experiment/instance_20250731_012542/results_n12_geomS_curv2_ibm_brisbane_SNDD23.json"
    
    # Check if files exist
    if not os.path.exists(instance1_file):
        print(f"❌ Instance 1 file not found: {instance1_file}")
        return
    
    if not os.path.exists(instance2_file):
        print(f"❌ Instance 2 file not found: {instance2_file}")
        return
    
    # List of analysis scripts to run
    analysis_scripts = [
        "comprehensive_experiment_analysis.py",
        "quantum_emergent_spacetime_enhanced_analysis.py",
        "causal_asymmetry_analysis.py",
        "page_curve_comprehensive_analysis.py",
        "curvature_flow_analysis.py",
        "enhanced_curvature_analysis.py"
    ]
    
    # Run analysis on both instances
    print(f"\n📁 INSTANCE 1: {os.path.basename(instance1_file)}")
    print("=" * 60)
    
    results1 = {}
    for script in analysis_scripts:
        script_path = os.path.join("analysis", script)
        if os.path.exists(script_path):
            success = run_analysis_script(script_path, instance1_file)
            results1[script] = success
        else:
            print(f"⚠️  Script not found: {script}")
            results1[script] = False
    
    print(f"\n📁 INSTANCE 2: {os.path.basename(instance2_file)}")
    print("=" * 60)
    
    results2 = {}
    for script in analysis_scripts:
        script_path = os.path.join("analysis", script)
        if os.path.exists(script_path):
            success = run_analysis_script(script_path, instance2_file)
            results2[script] = success
        else:
            print(f"⚠️  Script not found: {script}")
            results2[script] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\nInstance 1 ({os.path.basename(instance1_file)}):")
    for script, success in results1.items():
        status = "✅" if success else "❌"
        print(f"  {status} {script}")
    
    print(f"\nInstance 2 ({os.path.basename(instance2_file)}):")
    for script, success in results2.items():
        status = "✅" if success else "❌"
        print(f"  {status} {script}")
    
    print(f"\n🎉 Analysis complete! Results saved to respective instance directories.")

if __name__ == "__main__":
    main() 