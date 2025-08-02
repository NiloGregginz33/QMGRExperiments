#!/usr/bin/env python3
"""
Analyze ALL instances in custom_curvature_experiment to find the most comprehensive ones.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import glob

def get_all_instances() -> List[str]:
    """Get all instance directories in custom_curvature_experiment including older_results."""
    base_path = "experiment_logs/custom_curvature_experiment"
    instances = []
    
    # Get instances from main directory
    if os.path.exists(base_path):
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path) and item.startswith("instance_"):
                instances.append(f"main/{item}")
    
    # Get instances from older_results directory
    older_results_path = os.path.join(base_path, "older_results")
    if os.path.exists(older_results_path):
        for item in os.listdir(older_results_path):
            item_path = os.path.join(older_results_path, item)
            if os.path.isdir(item_path) and item.startswith("instance_"):
                instances.append(f"older_results/{item}")
    
    return sorted(instances)

def analyze_instance_comprehensiveness(instance_path: str) -> Dict[str, Any]:
    """Analyze how comprehensive an instance is based on its files and data."""
    full_path = f"experiment_logs/custom_curvature_experiment/{instance_path}"
    
    if not os.path.exists(full_path):
        return {"instance": instance_path, "error": "Path not found"}
    
    # Count different types of files
    files = os.listdir(full_path)
    
    analysis = {
        "instance": instance_path,
        "total_files": len(files),
        "file_types": {},
        "has_results_json": False,
        "has_summary_txt": False,
        "has_analysis_files": False,
        "has_plots": False,
        "qubit_count": None,
        "geometry_type": None,
        "curvature": None,
        "device": None,
        "flags": {},
        "comprehensiveness_score": 0
    }
    
    # Analyze file types
    for file in files:
        if file.endswith('.json'):
            analysis["file_types"]["json"] = analysis["file_types"].get("json", 0) + 1
            if "results" in file.lower():
                analysis["has_results_json"] = True
        elif file.endswith('.txt'):
            analysis["file_types"]["txt"] = analysis["file_types"].get("txt", 0) + 1
            if "summary" in file.lower():
                analysis["has_summary_txt"] = True
        elif file.endswith('.png') or file.endswith('.jpg'):
            analysis["file_types"]["plots"] = analysis["file_types"].get("plots", 0) + 1
            analysis["has_plots"] = True
    
    # Check for analysis files
    analysis_files = [f for f in files if any(keyword in f.lower() for keyword in 
                   ["analysis", "validation", "comprehensive", "quantum_emergent", "final_validation"])]
    analysis["has_analysis_files"] = len(analysis_files) > 0
    analysis["analysis_files"] = analysis_files
    
    # Try to extract experiment parameters from results files
    results_files = [f for f in files if f.endswith('.json') and 'results' in f.lower()]
    if results_files:
        try:
            # Try the first results file
            results_path = os.path.join(full_path, results_files[0])
            with open(results_path, 'r') as f:
                data = json.load(f)
                
            # Extract parameters
            if 'spec' in data:
                spec = data['spec']
                analysis["qubit_count"] = spec.get('num_qubits')
                analysis["geometry_type"] = spec.get('geometry')
                analysis["curvature"] = spec.get('curvature')
                analysis["device"] = spec.get('device')
                
                # Extract flags
                for key, value in spec.items():
                    if isinstance(value, bool):
                        analysis["flags"][key] = value
                        
        except Exception as e:
            analysis["error"] = f"Could not parse results: {str(e)}"
    
    # Calculate comprehensiveness score
    score = 0
    score += analysis["total_files"] * 2  # More files = more comprehensive
    score += 10 if analysis["has_results_json"] else 0
    score += 10 if analysis["has_summary_txt"] else 0
    score += 15 if analysis["has_analysis_files"] else 0
    score += 10 if analysis["has_plots"] else 0
    score += len(analysis["analysis_files"]) * 5  # Each analysis file adds points
    score += analysis["qubit_count"] if analysis["qubit_count"] else 0  # Higher qubit count = more complex
    
    analysis["comprehensiveness_score"] = score
    
    return analysis

def main():
    """Main analysis function."""
    print("🔍 Analyzing ALL instances in custom_curvature_experiment (including older_results)...")
    
    instances = get_all_instances()
    print(f"Found {len(instances)} instances to analyze")
    
    all_analyses = []
    
    for i, instance in enumerate(instances):
        print(f"Analyzing {i+1}/{len(instances)}: {instance}...")
        analysis = analyze_instance_comprehensiveness(instance)
        all_analyses.append(analysis)
    
    # Sort by comprehensiveness score
    all_analyses.sort(key=lambda x: x.get("comprehensiveness_score", 0), reverse=True)
    
    # Save results
    os.makedirs("search_results", exist_ok=True)
    
    with open("search_results/comprehensive_instance_analysis.json", 'w') as f:
        json.dump({
            "total_instances": len(instances),
            "analyses": all_analyses
        }, f, indent=2)
    
    # Generate summary
    with open("search_results/comprehensive_instance_summary.txt", 'w') as f:
        f.write("COMPREHENSIVE INSTANCE ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total instances analyzed: {len(instances)}\n\n")
        
        f.write("TOP 15 MOST COMPREHENSIVE INSTANCES:\n")
        f.write("-" * 40 + "\n")
        
        for i, analysis in enumerate(all_analyses[:15]):
            f.write(f"{i+1}. {analysis['instance']}\n")
            f.write(f"   Score: {analysis['comprehensiveness_score']}\n")
            f.write(f"   Files: {analysis['total_files']} total\n")
            f.write(f"   Qubits: {analysis['qubit_count']}\n")
            f.write(f"   Geometry: {analysis['geometry_type']}\n")
            f.write(f"   Curvature: {analysis['curvature']}\n")
            f.write(f"   Device: {analysis['device']}\n")
            f.write(f"   Analysis files: {len(analysis.get('analysis_files', []))}\n")
            f.write(f"   Flags: {analysis['flags']}\n")
            f.write("\n")
        
        f.write("\nDETAILED BREAKDOWN:\n")
        f.write("-" * 20 + "\n")
        
        for analysis in all_analyses:
            f.write(f"\n{analysis['instance']}:\n")
            f.write(f"  Score: {analysis.get('comprehensiveness_score', 0)}\n")
            f.write(f"  File types: {analysis['file_types']}\n")
            f.write(f"  Analysis files: {analysis.get('analysis_files', [])}\n")
            if 'error' in analysis:
                f.write(f"  Error: {analysis['error']}\n")
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Results saved to:")
    print(f"   - search_results/comprehensive_instance_analysis.json")
    print(f"   - search_results/comprehensive_instance_summary.txt")
    
    # Show top 10
    print(f"\n🏆 TOP 10 MOST COMPREHENSIVE INSTANCES:")
    for i, analysis in enumerate(all_analyses[:10]):
        print(f"{i+1}. {analysis['instance']} (Score: {analysis['comprehensiveness_score']})")
        print(f"   Qubits: {analysis['qubit_count']}, Geometry: {analysis['geometry_type']}")
        print(f"   Analysis files: {len(analysis.get('analysis_files', []))}")

if __name__ == "__main__":
    main() 