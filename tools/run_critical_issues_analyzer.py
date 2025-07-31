#!/usr/bin/env python3
"""
Runner script for Quantum Geometry Critical Issues Analyzer
"""

import sys
import os
from pathlib import Path

# Add the tools directory to the path
sys.path.append(str(Path(__file__).parent))

from quantum_geometry_critical_issues_analyzer import QuantumGeometryCriticalAnalyzer

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_critical_issues_analyzer.py <instance_directory>")
        print("Example: python run_critical_issues_analyzer.py experiment_logs/custom_curvature_experiment/instance_20250730_190246")
        sys.exit(1)
    
    instance_dir = sys.argv[1]
    
    if not os.path.exists(instance_dir):
        print(f"Error: Instance directory not found: {instance_dir}")
        sys.exit(1)
    
    try:
        analyzer = QuantumGeometryCriticalAnalyzer(instance_dir)
        
        print("Starting Quantum Geometry Critical Issues Analysis...")
        print(f"Analyzing instance: {instance_dir}")
        
        analyzer.analyze_continuum_limit()
        analyzer.analyze_classical_benchmarks()
        analyzer.analyze_decoherence_sensitivity()
        analyzer.analyze_causal_structure()
        analyzer.analyze_holographic_consistency()
        
        report = analyzer.generate_analysis_report()
        print(report)
        
        analyzer.create_visualization_plots()
        analyzer.save_analysis_results()
        
        print("Analysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 