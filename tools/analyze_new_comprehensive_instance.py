#!/usr/bin/env python3
"""
Analyze the new comprehensive instance and compare with previous top instances.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any

def analyze_instance_comprehensiveness(instance_path: str) -> Dict[str, Any]:
    """Analyze how comprehensive an instance is."""
    full_path = f"experiment_logs/custom_curvature_experiment/{instance_path}"
    
    if not os.path.exists(full_path):
        return {"instance": instance_path, "error": "Path not found"}
    
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
        "comprehensiveness_score": 0,
        "file_sizes": {},
        "key_features": []
    }
    
    # Analyze file types
    for file in files:
        file_path = os.path.join(full_path, file)
        file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else 0
        
        if file.endswith('.json'):
            analysis["file_types"]["json"] = analysis["file_types"].get("json", 0) + 1
            analysis["file_sizes"][file] = file_size
            if "results" in file.lower():
                analysis["has_results_json"] = True
        elif file.endswith('.txt'):
            analysis["file_types"]["txt"] = analysis["file_types"].get("txt", 0) + 1
            analysis["file_sizes"][file] = file_size
            if "summary" in file.lower():
                analysis["has_summary_txt"] = True
        elif file.endswith('.png') or file.endswith('.jpg'):
            analysis["file_types"]["plots"] = analysis["file_types"].get("plots", 0) + 1
            analysis["has_plots"] = True
            analysis["file_sizes"][file] = file_size
    
    # Check for analysis files
    analysis_files = [f for f in files if any(keyword in f.lower() for keyword in 
                   ["analysis", "validation", "comprehensive", "quantum_emergent", "final_validation", "entropy_engineering"])]
    analysis["has_analysis_files"] = len(analysis_files) > 0
    analysis["analysis_files"] = analysis_files
    
    # Try to extract experiment parameters from results files
    results_files = [f for f in files if f.endswith('.json') and 'results' in f.lower()]
    if results_files:
        try:
            # Try the largest results file
            largest_file = max(results_files, key=lambda f: analysis["file_sizes"].get(f, 0))
            results_path = os.path.join(full_path, largest_file)
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
                        
                # Check for key features
                if spec.get('einstein_solver'):
                    analysis["key_features"].append("Einstein Solver")
                if spec.get('page_curve'):
                    analysis["key_features"].append("Page Curve")
                if spec.get('entropy_engineering'):
                    analysis["key_features"].append("Entropy Engineering")
                if spec.get('lorentzian'):
                    analysis["key_features"].append("Lorentzian")
                if spec.get('excite'):
                    analysis["key_features"].append("Excite")
                if spec.get('analyze_curvature'):
                    analysis["key_features"].append("Curvature Analysis")
                if spec.get('compute_entropies'):
                    analysis["key_features"].append("Entropy Computation")
                if spec.get('hyperbolic_triangulation'):
                    analysis["key_features"].append("Hyperbolic Triangulation")
                if spec.get('strong_curvature'):
                    analysis["key_features"].append("Strong Curvature")
                if spec.get('charge_injection'):
                    analysis["key_features"].append("Charge Injection")
                if spec.get('spin_injection'):
                    analysis["key_features"].append("Spin Injection")
                if spec.get('use_ryu_takayanagi_test'):
                    analysis["key_features"].append("Ryu-Takayanagi Test")
                if spec.get('compare_MI_vs_subsystem_entropy'):
                    analysis["key_features"].append("MI vs Subsystem Entropy")
                if spec.get('embed_boundary_entropy_in_geometry'):
                    analysis["key_features"].append("Boundary Entropy Embedding")
                if spec.get('detect_and_flag_causal_loops'):
                    analysis["key_features"].append("Causal Loop Detection")
                if spec.get('filter_noncausal_edges'):
                    analysis["key_features"].append("Non-causal Edge Filtering")
                if spec.get('benchmark_against_classical_geometry'):
                    analysis["key_features"].append("Classical Geometry Benchmark")
                if spec.get('verify_noise_robustness'):
                    analysis["key_features"].append("Noise Robustness Verification")
                        
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
    score += len(analysis["key_features"]) * 3  # Each key feature adds points
    
    # Bonus for large file sizes (more data)
    total_size = sum(analysis["file_sizes"].values())
    if total_size > 1000000:  # 1MB
        score += 20
    elif total_size > 500000:  # 500KB
        score += 10
    
    analysis["comprehensiveness_score"] = score
    analysis["total_size_bytes"] = total_size
    
    return analysis

def main():
    """Main analysis function."""
    print("🔍 Analyzing new comprehensive instance...")
    
    # Analyze the new instance
    new_instance = "instance_20250802_140555"
    new_analysis = analyze_instance_comprehensiveness(new_instance)
    
    print(f"\n📊 NEW INSTANCE ANALYSIS: {new_instance}")
    print("=" * 60)
    print(f"Comprehensiveness Score: {new_analysis['comprehensiveness_score']}")
    print(f"Total Files: {new_analysis['total_files']}")
    print(f"File Types: {new_analysis['file_types']}")
    print(f"Total Size: {new_analysis['total_size_bytes']:,} bytes")
    print(f"Qubits: {new_analysis['qubit_count']}")
    print(f"Geometry: {new_analysis['geometry_type']}")
    print(f"Curvature: {new_analysis['curvature']}")
    print(f"Device: {new_analysis['device']}")
    print(f"Key Features ({len(new_analysis['key_features'])}):")
    for feature in new_analysis['key_features']:
        print(f"  ✅ {feature}")
    print(f"Analysis Files: {new_analysis['analysis_files']}")
    
    # Compare with previous top instances
    print(f"\n🏆 COMPARISON WITH PREVIOUS TOP INSTANCES:")
    print("=" * 60)
    
    previous_top_instances = [
        "older_results/instance_20250730_190246",  # Score: 99
        "older_results/instance_20250801_075511",  # Score: 70
        "older_results/instance_20250801_081414",  # Score: 70
        "older_results/instance_20250731_012542",  # Score: 53
        "older_results/instance_20250731_190955",  # Score: 51
    ]
    
    comparisons = []
    for prev_instance in previous_top_instances:
        prev_analysis = analyze_instance_comprehensiveness(prev_instance)
        comparisons.append(prev_analysis)
    
    # Sort by score
    all_instances = [new_analysis] + comparisons
    all_instances.sort(key=lambda x: x.get("comprehensiveness_score", 0), reverse=True)
    
    print(f"\n📈 RANKING (by comprehensiveness score):")
    print("-" * 60)
    
    for i, instance in enumerate(all_instances):
        rank = i + 1
        score = instance.get("comprehensiveness_score", 0)
        name = instance["instance"]
        qubits = instance.get("qubit_count", "N/A")
        features = len(instance.get("key_features", []))
        
        marker = "🆕" if "20250802_140555" in name else "📊"
        print(f"{rank:2d}. {marker} {name}")
        print(f"     Score: {score:3d} | Qubits: {qubits:2} | Features: {features:2d}")
        print(f"     Files: {instance['total_files']:2d} | Size: {instance.get('total_size_bytes', 0):,} bytes")
        print()
    
    # Save detailed comparison
    os.makedirs("search_results", exist_ok=True)
    with open("search_results/new_instance_comprehensive_analysis.json", 'w') as f:
        json.dump({
            "new_instance": new_analysis,
            "comparisons": comparisons,
            "ranking": all_instances
        }, f, indent=2)
    
    print(f"✅ Analysis complete! Results saved to: search_results/new_instance_comprehensive_analysis.json")
    
    # Determine if this is the most comprehensive
    if all_instances[0]["instance"] == new_instance:
        print(f"\n🎉 CONGRATULATIONS! The new instance is the MOST COMPREHENSIVE!")
        print(f"   Previous best: {all_instances[1]['instance']} (Score: {all_instances[1]['comprehensiveness_score']})")
        print(f"   New best: {new_instance} (Score: {new_analysis['comprehensiveness_score']})")
    else:
        print(f"\n📊 The new instance ranks #{all_instances.index(new_analysis) + 1} out of {len(all_instances)}")
        print(f"   Most comprehensive: {all_instances[0]['instance']} (Score: {all_instances[0]['comprehensiveness_score']})")

if __name__ == "__main__":
    main() 