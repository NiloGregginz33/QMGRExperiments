#!/usr/bin/env python3
"""
Quick analysis to find the most comprehensive instances in custom_curvature_experiment.
"""

import os
import json
from pathlib import Path

def get_all_instances():
    """Get all instance directories."""
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

def analyze_instance(instance_path):
    """Quick analysis of an instance."""
    full_path = f"experiment_logs/custom_curvature_experiment/{instance_path}"
    
    if not os.path.exists(full_path):
        return {"instance": instance_path, "score": 0, "files": 0}
    
    files = os.listdir(full_path)
    
    # Count analysis files
    analysis_files = [f for f in files if any(keyword in f.lower() for keyword in 
                   ["analysis", "validation", "comprehensive", "quantum_emergent", "final_validation"])]
    
    # Count plots
    plot_files = [f for f in files if f.endswith(('.png', '.jpg'))]
    
    # Count JSON files
    json_files = [f for f in files if f.endswith('.json')]
    
    # Count TXT files
    txt_files = [f for f in files if f.endswith('.txt')]
    
    # Calculate score
    score = len(files) * 2 + len(analysis_files) * 10 + len(plot_files) * 5
    
    return {
        "instance": instance_path,
        "score": score,
        "files": len(files),
        "analysis_files": len(analysis_files),
        "plot_files": len(plot_files),
        "json_files": len(json_files),
        "txt_files": len(txt_files)
    }

def main():
    print("🔍 Quick analysis of ALL instances...")
    
    instances = get_all_instances()
    print(f"Found {len(instances)} instances")
    
    results = []
    
    for i, instance in enumerate(instances):
        if i % 20 == 0:
            print(f"Progress: {i}/{len(instances)}")
        analysis = analyze_instance(instance)
        results.append(analysis)
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n🏆 TOP 15 MOST COMPREHENSIVE INSTANCES:")
    print("-" * 60)
    
    for i, result in enumerate(results[:15]):
        print(f"{i+1:2d}. {result['instance']}")
        print(f"     Score: {result['score']:4d} | Files: {result['files']:3d} | Analysis: {result['analysis_files']:2d} | Plots: {result['plot_files']:2d}")
        print()
    
    # Save top 20 to file
    os.makedirs("search_results", exist_ok=True)
    with open("search_results/top_comprehensive_instances.txt", 'w') as f:
        f.write("TOP 20 MOST COMPREHENSIVE INSTANCES\n")
        f.write("=" * 50 + "\n\n")
        
        for i, result in enumerate(results[:20]):
            f.write(f"{i+1:2d}. {result['instance']}\n")
            f.write(f"     Score: {result['score']:4d} | Files: {result['files']:3d} | Analysis: {result['analysis_files']:2d} | Plots: {result['plot_files']:2d}\n")
            f.write(f"     JSON: {result['json_files']:2d} | TXT: {result['txt_files']:2d}\n\n")
    
    print(f"📊 Results saved to: search_results/top_comprehensive_instances.txt")

if __name__ == "__main__":
    main() 