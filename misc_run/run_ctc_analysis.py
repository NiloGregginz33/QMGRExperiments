#!/usr/bin/env python3
"""
Run CTC analysis and capture output
"""

import subprocess
import sys

def run_analysis():
    filepath = "experiment_logs/custom_curvature_experiment/instance_20250801_151819/results_n8_geomS_curv1_ibm_brisbane_HSXEL9.json"
    
    print("🔍 RUNNING CTC ANALYSIS")
    print("=" * 50)
    print(f"Target: {filepath}")
    print("-" * 50)
    
    try:
        # Run the Deutsch CTC analysis
        result = subprocess.run([
            sys.executable, 
            "analysis/deutsch_ctc_analysis.py", 
            filepath
        ], capture_output=True, text=True, timeout=120)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        print(f"Return code: {result.returncode}")
        
    except subprocess.TimeoutExpired:
        print("Analysis timed out")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_analysis() 