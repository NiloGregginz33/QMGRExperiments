#!/usr/bin/env python3
"""
Analyze the 14 matching instances to identify Einstein solvers and Regge solvers.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

def load_matching_instances() -> List[Dict[str, Any]]:
    """Load the matching instances from the search results."""
    json_path = "search_results/lorentzian_excite_search_results.json"
    
    if not os.path.exists(json_path):
        print(f"❌ Search results not found: {json_path}")
        print("Please run the search script first.")
        return []
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data['matching_instances']

def check_solver_type(data: Dict[str, Any]) -> Dict[str, bool]:
    """Check if data contains Einstein or Regge solver indicators."""
    is_einstein = False
    is_regge = False
    
    # Check for solver fields in spec section
    if 'spec' in data and isinstance(data['spec'], dict):
        spec = data['spec']
        
        # Check for solve_regge field
        if 'solve_regge' in spec:
            is_regge = bool(spec['solve_regge'])
        
        # Check for einstein_solver field
        if 'einstein_solver' in spec:
            is_einstein = bool(spec['einstein_solver'])
    
    # Also check for any other solver-related fields recursively
    def check_recursive(obj, path=""):
        nonlocal is_einstein, is_regge
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check key names for solver indicators
                if 'einstein' in key.lower() and isinstance(value, bool):
                    is_einstein = is_einstein or value
                if 'regge' in key.lower() and isinstance(value, bool):
                    is_regge = is_regge or value
                
                # Check string values
                if isinstance(value, str):
                    if 'einstein' in value.lower():
                        is_einstein = True
                    if 'regge' in value.lower():
                        is_regge = True
                
                # Recursively check nested structures
                check_recursive(value, current_path)
        
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                current_path = f"{path}[{i}]"
                check_recursive(item, current_path)
    
    check_recursive(data)
    return {'einstein': is_einstein, 'regge': is_regge}

def analyze_instance_solver_types(instance_info: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze an instance to determine its solver types."""
    instance_path = instance_info['instance_path']
    matching_files = instance_info['matching_files']
    
    einstein_files = []
    regge_files = []
    other_files = []
    
    for file_info in matching_files:
        if not file_info['meets_criteria']:
            continue
            
        file_path = file_info['file_path']
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            solver_types = check_solver_type(data)
            
            file_analysis = {
                'file_name': file_info['file_name'],
                'file_path': file_path,
                'solver_types': solver_types,
                'is_einstein': solver_types['einstein'],
                'is_regge': solver_types['regge']
            }
            
            if solver_types['einstein']:
                einstein_files.append(file_analysis)
            elif solver_types['regge']:
                regge_files.append(file_analysis)
            else:
                other_files.append(file_analysis)
                
        except Exception as e:
            print(f"⚠️  Error reading {file_path}: {e}")
            continue
    
    return {
        'instance_name': instance_info['instance_name'],
        'instance_path': instance_path,
        'qubit_count': instance_info['qubit_count'],
        'einstein_files': einstein_files,
        'regge_files': regge_files,
        'other_files': other_files,
        'has_einstein': len(einstein_files) > 0,
        'has_regge': len(regge_files) > 0,
        'solver_summary': {
            'einstein_count': len(einstein_files),
            'regge_count': len(regge_files),
            'other_count': len(other_files)
        }
    }

def analyze_all_instances() -> Dict[str, Any]:
    """Analyze all matching instances for solver types."""
    matching_instances = load_matching_instances()
    
    if not matching_instances:
        return {}
    
    print(f"🔍 Analyzing {len(matching_instances)} matching instances for solver types...")
    
    analyzed_instances = []
    einstein_instances = []
    regge_instances = []
    both_instances = []
    other_instances = []
    
    for instance_info in matching_instances:
        print(f"📁 Analyzing {instance_info['instance_name']}...")
        
        analysis = analyze_instance_solver_types(instance_info)
        analyzed_instances.append(analysis)
        
        # Categorize instance
        if analysis['has_einstein'] and analysis['has_regge']:
            both_instances.append(analysis)
            print(f"  ✅ Both Einstein and Regge solvers")
        elif analysis['has_einstein']:
            einstein_instances.append(analysis)
            print(f"  🔬 Einstein solver only")
        elif analysis['has_regge']:
            regge_instances.append(analysis)
            print(f"  📐 Regge solver only")
        else:
            other_instances.append(analysis)
            print(f"  ❓ Other/Unknown solver type")
    
    return {
        'total_instances': len(analyzed_instances),
        'einstein_only': einstein_instances,
        'regge_only': regge_instances,
        'both_solvers': both_instances,
        'other_solvers': other_instances,
        'all_analyzed': analyzed_instances,
        'summary': {
            'einstein_count': len(einstein_instances),
            'regge_count': len(regge_instances),
            'both_count': len(both_instances),
            'other_count': len(other_instances)
        }
    }

def generate_solver_report(analysis_results: Dict[str, Any], output_dir: str = "search_results") -> None:
    """Generate a comprehensive report of solver types."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed JSON report
    json_path = os.path.join(output_dir, "solver_type_analysis.json")
    with open(json_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Create human-readable summary
    summary_path = os.path.join(output_dir, "solver_type_analysis_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("SOLVER TYPE ANALYSIS RESULTS\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Total Instances Analyzed: {analysis_results['total_instances']}\n\n")
        
        summary = analysis_results['summary']
        f.write("Solver Type Distribution:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Einstein solvers only: {summary['einstein_count']}\n")
        f.write(f"Regge solvers only: {summary['regge_count']}\n")
        f.write(f"Both Einstein & Regge: {summary['both_count']}\n")
        f.write(f"Other/Unknown solvers: {summary['other_count']}\n\n")
        
        # Einstein solvers
        if analysis_results['einstein_only']:
            f.write("EINSTEIN SOLVERS ONLY:\n")
            f.write("-" * 20 + "\n")
            for i, instance in enumerate(analysis_results['einstein_only'], 1):
                f.write(f"{i}. {instance['instance_name']} ({instance['qubit_count']} qubits)\n")
                f.write(f"   Path: {instance['instance_path']}\n")
                f.write(f"   Einstein files: {instance['solver_summary']['einstein_count']}\n")
                for file_info in instance['einstein_files']:
                    f.write(f"     - {file_info['file_name']}\n")
                f.write("\n")
        
        # Regge solvers
        if analysis_results['regge_only']:
            f.write("REGGE SOLVERS ONLY:\n")
            f.write("-" * 18 + "\n")
            for i, instance in enumerate(analysis_results['regge_only'], 1):
                f.write(f"{i}. {instance['instance_name']} ({instance['qubit_count']} qubits)\n")
                f.write(f"   Path: {instance['instance_path']}\n")
                f.write(f"   Regge files: {instance['solver_summary']['regge_count']}\n")
                for file_info in instance['regge_files']:
                    f.write(f"     - {file_info['file_name']}\n")
                f.write("\n")
        
        # Both solvers
        if analysis_results['both_solvers']:
            f.write("BOTH EINSTEIN & REGGE SOLVERS:\n")
            f.write("-" * 30 + "\n")
            for i, instance in enumerate(analysis_results['both_solvers'], 1):
                f.write(f"{i}. {instance['instance_name']} ({instance['qubit_count']} qubits)\n")
                f.write(f"   Path: {instance['instance_path']}\n")
                f.write(f"   Einstein files: {instance['solver_summary']['einstein_count']}\n")
                f.write(f"   Regge files: {instance['solver_summary']['regge_count']}\n")
                f.write("   Einstein files:\n")
                for file_info in instance['einstein_files']:
                    f.write(f"     - {file_info['file_name']}\n")
                f.write("   Regge files:\n")
                for file_info in instance['regge_files']:
                    f.write(f"     - {file_info['file_name']}\n")
                f.write("\n")
        
        # Other solvers
        if analysis_results['other_solvers']:
            f.write("OTHER/UNKNOWN SOLVER TYPES:\n")
            f.write("-" * 25 + "\n")
            for i, instance in enumerate(analysis_results['other_solvers'], 1):
                f.write(f"{i}. {instance['instance_name']} ({instance['qubit_count']} qubits)\n")
                f.write(f"   Path: {instance['instance_path']}\n")
                f.write(f"   Other files: {instance['solver_summary']['other_count']}\n")
                for file_info in instance['other_files']:
                    f.write(f"     - {file_info['file_name']}\n")
                f.write("\n")
    
    print(f"\n📊 Solver analysis report generated:")
    print(f"   JSON: {json_path}")
    print(f"   Summary: {summary_path}")

def main():
    """Main analysis function."""
    print("🔬 Analyzing solver types in matching instances...")
    print("=" * 60)
    
    analysis_results = analyze_all_instances()
    
    if not analysis_results:
        print("❌ No instances to analyze.")
        return
    
    summary = analysis_results['summary']
    
    print(f"\n📋 Analysis Complete!")
    print(f"Total instances: {analysis_results['total_instances']}")
    print(f"Einstein solvers only: {summary['einstein_count']}")
    print(f"Regge solvers only: {summary['regge_count']}")
    print(f"Both Einstein & Regge: {summary['both_count']}")
    print(f"Other/Unknown solvers: {summary['other_count']}")
    
    # Generate detailed report
    generate_solver_report(analysis_results)

if __name__ == "__main__":
    main() 