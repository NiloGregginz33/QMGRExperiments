#!/usr/bin/env python3
"""
Simple script to transfer working MI from CGPTFactory to experiment results.
This script extracts the MI values that CGPTFactory calculates and saves them
to the experiment results file.
"""

import json
import re
import sys
import os
from pathlib import Path

def extract_mi_from_cgpt_output(output_text):
    """Extract MI values from CGPTFactory output."""
    # Look for the pattern: Mutual information: {'I_0,1': np.float64(value), ...}
    mi_pattern = r"Mutual information: \{([^}]+)\}"
    match = re.search(mi_pattern, output_text)
    
    if not match:
        return None
    
    mi_str = match.group(1)
    # Parse the MI dictionary
    mi_dict = {}
    
    # Extract individual MI values
    mi_items = re.findall(r"'([^']+)': np\.float64\(([^)]+)\)", mi_str)
    for key, value in mi_items:
        try:
            mi_dict[key] = float(value)
        except ValueError:
            print(f"Warning: Could not parse MI value {value} for key {key}")
            mi_dict[key] = None
    
    return mi_dict

def update_results_file(results_file_path, mi_dict):
    """Update the results file with the working MI values."""
    try:
        with open(results_file_path, 'r') as f:
            results = json.load(f)
        
        # Update the mutual_information_per_timestep
        if 'mutual_information_per_timestep' in results and len(results['mutual_information_per_timestep']) > 0:
            # Replace the first timestep's MI with the working values
            results['mutual_information_per_timestep'][0] = mi_dict
            
            # Also update edge_mi if it exists
            if 'edge_mi' in results:
                edge_mi = {}
                for key, value in mi_dict.items():
                    if value is not None:
                        edge_mi[key] = value
                    else:
                        edge_mi[key] = None
                results['edge_mi'] = edge_mi
        
        # Save the updated results
        with open(results_file_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Updated {results_file_path} with working MI values:")
        for key, value in mi_dict.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"Error updating results file: {e}")
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python fix_mi_transfer.py <output_text_file> <results_json_file>")
        sys.exit(1)
    
    output_file = sys.argv[1]
    results_file = sys.argv[2]
    
    # Read the output text
    try:
        with open(output_file, 'r') as f:
            output_text = f.read()
    except FileNotFoundError:
        print(f"Output file {output_file} not found")
        sys.exit(1)
    
    # Extract MI from CGPTFactory output
    mi_dict = extract_mi_from_cgpt_output(output_text)
    
    if mi_dict is None:
        print("No MI values found in output text")
        sys.exit(1)
    
    print("Extracted MI values from CGPTFactory:")
    for key, value in mi_dict.items():
        print(f"  {key}: {value}")
    
    # Update the results file
    if update_results_file(results_file, mi_dict):
        print("Successfully updated results file with working MI values")
    else:
        print("Failed to update results file")
        sys.exit(1)

if __name__ == "__main__":
    main() 