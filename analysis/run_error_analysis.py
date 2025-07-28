#!/usr/bin/env python3
"""
Run Error Analysis on Custom Curvature Experiment
================================================

Simple script to run error analysis on any custom curvature experiment results file.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from custom_curvature_error_analysis import CustomCurvatureErrorAnalyzer

def main():
    """Run error analysis on a specified results file."""
    
    if len(sys.argv) < 2:
        print("Usage: python run_error_analysis.py <results_file_path>")
        print("Example: python run_error_analysis.py experiment_logs/custom_curvature_experiment/older_results/results_n7_geomH_curv5_ibm_brisbane_DUXTVJ.json")
        return
    
    results_file = sys.argv[1]
    
    if not Path(results_file).exists():
        print(f"Error: Results file not found: {results_file}")
        return
    
    print(f"Running error analysis on: {results_file}")
    
    # Run error analysis
    analyzer = CustomCurvatureErrorAnalyzer(results_file)
    analysis_data = analyzer.run_complete_analysis()
    
    print("\nError analysis completed successfully!")
    print(f"Results saved to: {analyzer.output_dir}")

if __name__ == "__main__":
    main() 