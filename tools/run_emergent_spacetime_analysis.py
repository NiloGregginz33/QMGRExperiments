#!/usr/bin/env python3
"""
Run Emergent Spacetime Analysis

This script runs the comprehensive analysis framework on existing experimental data
to distinguish between emergent spacetime geometry and geometric quantum correlations.

Usage:
    python run_emergent_spacetime_analysis.py

Author: Quantum Gravity Research Team
Date: 2025
"""

import os
import glob
import sys
from pathlib import Path

# Add the tools directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from emergent_spacetime_analysis import EmergentSpacetimeAnalyzer

def find_experiment_data():
    """
    Find all available experiment data files for analysis.
    
    Returns:
        list: List of data file paths
    """
    # Look for custom curvature experiment data
    base_path = Path("experiment_logs/custom_curvature_experiment")
    
    if not base_path.exists():
        print(f"Error: Experiment data directory not found: {base_path}")
        return []
    
    # Find all result files
    result_files = []
    
    # Look for different system sizes
    system_sizes = [3, 4, 5, 7, 11]  # Common qubit counts in your experiments
    
    for instance_dir in base_path.glob("instance_*"):
        if not instance_dir.is_dir():
            continue
            
        # Find result files in this instance
        for result_file in instance_dir.glob("results_*.json"):
            try:
                # Quick check to see if it's a valid experiment file
                import json
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                if 'spec' in data and 'num_qubits' in data['spec']:
                    n_qubits = data['spec']['num_qubits']
                    device = data['spec'].get('device', 'unknown')
                    
                    print(f"Found experiment: {n_qubits} qubits, {device}")
                    result_files.append(str(result_file))
                    
            except Exception as e:
                print(f"Error reading {result_file}: {e}")
                continue
    
    return result_files

def select_data_for_analysis(data_files):
    """
    Select appropriate data files for comprehensive analysis.
    
    Args:
        data_files: List of all available data files
        
    Returns:
        list: Selected data files for analysis
    """
    if not data_files:
        return []
    
    # Group files by system size and device
    experiments = {}
    
    for file_path in data_files:
        try:
            import json
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            n_qubits = data['spec']['num_qubits']
            device = data['spec'].get('device', 'unknown')
            geometry = data['spec'].get('geometry', 'unknown')
            curvature = data['spec'].get('curvature', 0)
            
            key = (n_qubits, device, geometry, curvature)
            if key not in experiments:
                experiments[key] = []
            experiments[key].append(file_path)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Select representative experiments for analysis
    selected_files = []
    
    # Priority: different system sizes, both hardware and simulator
    priority_keys = [
        (3, 'simulator', 'spherical', 20),  # Small system, simulator
        (5, 'simulator', 'spherical', 20),  # Medium system, simulator
        (7, 'simulator', 'spherical', 20),  # Larger system, simulator
        (11, 'simulator', 'spherical', 20), # Largest system, simulator
        (11, 'ibm_brisbane', 'spherical', 20), # Hardware comparison
    ]
    
    for key in priority_keys:
        if key in experiments:
            # Take the first available file for this configuration
            selected_files.append(experiments[key][0])
            print(f"Selected: {key[0]} qubits, {key[1]}, {key[2]} geometry, curvature {key[3]}")
    
    # If we don't have enough priority files, add others
    if len(selected_files) < 3:
        for key, files in experiments.items():
            if files[0] not in selected_files:
                selected_files.append(files[0])
                print(f"Additional: {key[0]} qubits, {key[1]}, {key[2]} geometry, curvature {key[3]}")
                if len(selected_files) >= 5:  # Limit to 5 experiments
                    break
    
    return selected_files

def run_analysis():
    """
    Run the comprehensive emergent spacetime analysis.
    """
    print("EMERGENT SPACETIME ANALYSIS")
    print("=" * 50)
    print("Distinguishing between emergent spacetime geometry and geometric quantum correlations")
    print()
    
    # Find available data
    print("Searching for experiment data...")
    all_data_files = find_experiment_data()
    
    if not all_data_files:
        print("No experiment data found!")
        print("Please ensure you have run experiments and data is available in:")
        print("  experiment_logs/custom_curvature_experiment/")
        return
    
    print(f"Found {len(all_data_files)} experiment data files")
    
    # Select data for analysis
    print("\nSelecting data for analysis...")
    selected_files = select_data_for_analysis(all_data_files)
    
    if len(selected_files) < 2:
        print("Insufficient data for comprehensive analysis!")
        print("Need at least 2 experiments with different system sizes.")
        return
    
    print(f"\nSelected {len(selected_files)} experiments for analysis:")
    for i, file_path in enumerate(selected_files, 1):
        print(f"  {i}. {os.path.basename(file_path)}")
    
    # Run analysis
    print(f"\nRunning comprehensive analysis...")
    analyzer = EmergentSpacetimeAnalyzer()
    
    try:
        results = analyzer.run_comprehensive_analysis(selected_files)
        
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE")
        print("=" * 50)
        
        # Display key results
        final_assessment = results.get('final_assessment', {})
        
        print(f"\nFINAL CONCLUSION:")
        print(f"  {final_assessment.get('conclusion', 'Unknown')}")
        print(f"  Confidence: {final_assessment.get('confidence_level', 'Unknown')}")
        
        print(f"\nEVIDENCE SUMMARY:")
        print(f"  Emergent Geometry: {final_assessment.get('emergent_evidence', 0)}/{final_assessment.get('total_tests', 0)} tests")
        print(f"  Quantum Correlations: {final_assessment.get('quantum_evidence', 0)}/{final_assessment.get('total_tests', 0)} tests")
        
        print(f"\nDETAILED RESULTS:")
        for test_name, detail in final_assessment.get('assessment_details', {}).items():
            print(f"  {test_name}: {detail}")
        
        print(f"\nResults saved to meta_analysis/instance_*/ directory:")
        print(f"  - emergent_spacetime_analysis_*.json (detailed results)")
        print(f"  - emergent_spacetime_summary_*.txt (summary report)")
        print(f"  - experiments_analyzed_*.txt (list of experiments used)")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function."""
    run_analysis()

if __name__ == "__main__":
    main() 