#!/usr/bin/env python3
"""
Search script to find instances in custom_curvature/older_results that have:
- lorentzian flag
- excite flag  
- 10+ qubits
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

def extract_qubit_count_from_filename(filename: str) -> Optional[int]:
    """Extract qubit count from filename patterns like 'n12_', 'n10_', etc."""
    match = re.search(r'n(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def extract_qubit_count_from_data(data: Dict[str, Any]) -> Optional[int]:
    """Extract qubit count from various possible locations in the data."""
    # Check common locations for qubit count
    possible_keys = ['num_qubits', 'n_qubits', 'qubits', 'n', 'size']
    
    for key in possible_keys:
        if key in data:
            value = data[key]
            if isinstance(value, (int, float)):
                return int(value)
    
    # Check if there's a circuit or state vector that indicates qubit count
    if 'circuit' in data:
        circuit = data['circuit']
        if isinstance(circuit, dict) and 'num_qubits' in circuit:
            return int(circuit['num_qubits'])
    
    if 'state_vector' in data:
        import numpy as np
        state_vector = data['state_vector']
        if isinstance(state_vector, (list, np.ndarray)):
            # Calculate qubits from state vector size: 2^n
            size = len(state_vector)
            import math
            n = math.log2(size)
            if n.is_integer():
                return int(n)
    
    return None

def check_flags_in_data(data: Dict[str, Any]) -> tuple[bool, bool]:
    """Check if data contains lorentzian and excite flags."""
    has_lorentzian = False
    has_excite = False
    
    def check_recursive(obj, path=""):
        nonlocal has_lorentzian, has_excite
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check key names
                if 'lorentzian' in key.lower():
                    has_lorentzian = True
                if 'excite' in key.lower():
                    has_excite = True
                
                # Check string values
                if isinstance(value, str):
                    if 'lorentzian' in value.lower():
                        has_lorentzian = True
                    if 'excite' in value.lower():
                        has_excite = True
                
                # Check boolean values
                if isinstance(value, bool):
                    if key.lower() == 'lorentzian' and value:
                        has_lorentzian = True
                    if key.lower() == 'excite' and value:
                        has_excite = True
                
                # Recursively check nested structures
                check_recursive(value, current_path)
        
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                current_path = f"{path}[{i}]"
                check_recursive(item, current_path)
    
    check_recursive(data)
    return has_lorentzian, has_excite

def search_instances(base_path: str = "experiment_logs/custom_curvature_experiment/older_results") -> List[Dict[str, Any]]:
    """Search for instances matching the criteria."""
    base_path = Path(base_path)
    matching_instances = []
    
    if not base_path.exists():
        print(f"❌ Base path does not exist: {base_path}")
        return matching_instances
    
    # Get all instance directories
    instance_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('instance_')]
    print(f"🔍 Found {len(instance_dirs)} instance directories to search")
    
    for instance_dir in instance_dirs:
        print(f"📁 Searching {instance_dir.name}...")
        
        # Find all JSON files in the instance
        json_files = list(instance_dir.glob("*.json"))
        
        instance_info = {
            'instance_path': str(instance_dir),
            'instance_name': instance_dir.name,
            'matching_files': [],
            'qubit_count': None,
            'has_lorentzian': False,
            'has_excite': False,
            'meets_criteria': False
        }
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract qubit count
                qubit_count = extract_qubit_count_from_data(data)
                if qubit_count is None:
                    qubit_count = extract_qubit_count_from_filename(json_file.name)
                
                # Check for flags
                has_lorentzian, has_excite = check_flags_in_data(data)
                
                file_info = {
                    'file_path': str(json_file),
                    'file_name': json_file.name,
                    'qubit_count': qubit_count,
                    'has_lorentzian': has_lorentzian,
                    'has_excite': has_excite,
                    'meets_criteria': (has_lorentzian and has_excite and qubit_count is not None and qubit_count >= 10)
                }
                
                instance_info['matching_files'].append(file_info)
                
                # Update instance-level flags
                if has_lorentzian:
                    instance_info['has_lorentzian'] = True
                if has_excite:
                    instance_info['has_excite'] = True
                if qubit_count is not None:
                    instance_info['qubit_count'] = qubit_count
                
            except Exception as e:
                print(f"⚠️  Error reading {json_file}: {e}")
                continue
        
        # Check if instance meets overall criteria
        instance_info['meets_criteria'] = (
            instance_info['has_lorentzian'] and 
            instance_info['has_excite'] and 
            instance_info['qubit_count'] is not None and 
            instance_info['qubit_count'] >= 10
        )
        
        if instance_info['meets_criteria']:
            matching_instances.append(instance_info)
            print(f"✅ Found matching instance: {instance_dir.name} (qubits: {instance_info['qubit_count']})")
        else:
            print(f"❌ Instance {instance_dir.name} does not meet criteria")
    
    return matching_instances

def generate_report(matching_instances: List[Dict[str, Any]], output_dir: str = "search_results") -> None:
    """Generate a comprehensive report of matching instances."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create detailed JSON report
    report_data = {
        'search_criteria': {
            'lorentzian_flag': True,
            'excite_flag': True,
            'min_qubits': 10
        },
        'total_matching_instances': len(matching_instances),
        'matching_instances': matching_instances,
        'summary': {
            'qubit_distribution': {},
            'flag_combinations': {}
        }
    }
    
    # Analyze qubit distribution
    qubit_counts = [inst['qubit_count'] for inst in matching_instances if inst['qubit_count'] is not None]
    for count in qubit_counts:
        report_data['summary']['qubit_distribution'][str(count)] = qubit_counts.count(count)
    
    # Save JSON report
    json_path = os.path.join(output_dir, "lorentzian_excite_search_results.json")
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Create human-readable summary
    summary_path = os.path.join(output_dir, "lorentzian_excite_search_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("LORENTZIAN + EXCITE FLAG SEARCH RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Search Criteria:\n")
        f.write(f"- Must have 'lorentzian' flag\n")
        f.write(f"- Must have 'excite' flag\n")
        f.write(f"- Must have 10+ qubits\n\n")
        
        f.write(f"Total Matching Instances: {len(matching_instances)}\n\n")
        
        if matching_instances:
            f.write("Matching Instances:\n")
            f.write("-" * 30 + "\n")
            
            for i, instance in enumerate(matching_instances, 1):
                f.write(f"{i}. {instance['instance_name']}\n")
                f.write(f"   Path: {instance['instance_path']}\n")
                f.write(f"   Qubits: {instance['qubit_count']}\n")
                f.write(f"   Files with flags:\n")
                
                for file_info in instance['matching_files']:
                    if file_info['meets_criteria']:
                        f.write(f"     - {file_info['file_name']}\n")
                
                f.write("\n")
            
            f.write("Qubit Distribution:\n")
            f.write("-" * 20 + "\n")
            for qubits, count in sorted(report_data['summary']['qubit_distribution'].items()):
                f.write(f"{qubits} qubits: {count} instances\n")
        else:
            f.write("No instances found matching the criteria.\n")
    
    print(f"\n📊 Report generated:")
    print(f"   JSON: {json_path}")
    print(f"   Summary: {summary_path}")

def main():
    """Main search function."""
    print("🔍 Searching for instances with lorentzian + excite flags and 10+ qubits...")
    print("=" * 70)
    
    matching_instances = search_instances()
    
    print(f"\n📋 Search Complete!")
    print(f"Found {len(matching_instances)} matching instances")
    
    if matching_instances:
        print("\nMatching Instances:")
        for i, instance in enumerate(matching_instances, 1):
            print(f"{i}. {instance['instance_name']} ({instance['qubit_count']} qubits)")
        
        # Generate detailed report
        generate_report(matching_instances)
    else:
        print("❌ No instances found matching the criteria.")

if __name__ == "__main__":
    main() 