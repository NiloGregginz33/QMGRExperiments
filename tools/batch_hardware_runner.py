#!/usr/bin/env python3
"""
Batch Hardware Runner for Quantum Spacetime Experiments
=======================================================

This tool runs multiple quantum spacetime experiments in parallel with automatic
backend adaptation. It can test different parameters, backends, and configurations
to find optimal quantum spacetime signatures.

Features:
- Automatic backend detection and adaptation
- Parallel execution of multiple experiments
- Parameter sweeping across different configurations
- Real-time progress monitoring
- Automatic result analysis and comparison
- Hardware optimization for any IBM backend
"""

import os
import sys
import json
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def get_available_backends():
    """Get list of available IBM backends."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        backends = service.backends()
        return [backend.name for backend in backends]
    except Exception as e:
        print(f"[BATCH] Warning: Could not get backends: {e}")
        return ["ibm_brisbane", "ibm_kyoto", "ibm_osaka", "ibm_sherbrooke"]

def create_experiment_configs():
    """Create different experiment configurations to test."""
    configs = []
    
    # Base configurations
    base_configs = [
        {
            "name": "hardware_optimized_6q",
            "num_qubits": 6,
            "geometry": "spherical",
            "curvature": 8.0,
            "device": "ibm_brisbane",
            "shots": 4096,
            "timesteps": 4,
            "weight": 6.0,
            "gamma": 4.0,
            "init_angle": 1.0,
            "enhanced_entanglement": True,
            "page_curve": True,
            "compute_entropies": True
        },
        {
            "name": "hardware_optimized_7q",
            "num_qubits": 7,
            "geometry": "spherical", 
            "curvature": 10.0,
            "device": "ibm_brisbane",
            "shots": 4096,
            "timesteps": 6,
            "weight": 7.0,
            "gamma": 5.0,
            "init_angle": 1.0,
            "enhanced_entanglement": True,
            "page_curve": True,
            "compute_entropies": True
        },
        {
            "name": "hardware_optimized_8q",
            "num_qubits": 8,
            "geometry": "spherical",
            "curvature": 12.0,
            "device": "ibm_brisbane", 
            "shots": 8192,
            "timesteps": 8,
            "weight": 8.0,
            "gamma": 6.0,
            "init_angle": 1.0,
            "enhanced_entanglement": True,
            "page_curve": True,
            "compute_entropies": True
        }
    ]
    
    # Add different backend configurations
    backends = get_available_backends()
    for config in base_configs:
        for backend in backends[:3]:  # Limit to first 3 backends
            backend_config = config.copy()
            backend_config["name"] = f"{config['name']}_{backend}"
            backend_config["device"] = backend
            configs.append(backend_config)
    
    # Add parameter sweep configurations
    for weight in [4.0, 6.0, 8.0]:
        for curvature in [5.0, 8.0, 12.0]:
            sweep_config = {
                "name": f"param_sweep_w{weight}_k{curvature}",
                "num_qubits": 6,
                "geometry": "spherical",
                "curvature": curvature,
                "device": "ibm_brisbane",
                "shots": 4096,
                "timesteps": 4,
                "weight": weight,
                "gamma": weight * 0.7,
                "init_angle": 1.0,
                "enhanced_entanglement": True,
                "page_curve": True,
                "compute_entropies": True
            }
            configs.append(sweep_config)
    
    return configs

def build_command(config):
    """Build the command string for an experiment configuration."""
    cmd = [
        "python", "src/experiments/custom_curvature_experiment.py",
        "--num_qubits", str(config["num_qubits"]),
        "--geometry", config["geometry"],
        "--curvature", str(config["curvature"]),
        "--device", config["device"],
        "--shots", str(config["shots"]),
        "--timesteps", str(config["timesteps"]),
        "--weight", str(config["weight"]),
        "--gamma", str(config["gamma"]),
        "--init_angle", str(config["init_angle"])
    ]
    
    if config.get("enhanced_entanglement"):
        cmd.append("--enhanced_entanglement")
    if config.get("page_curve"):
        cmd.append("--page_curve")
    if config.get("compute_entropies"):
        cmd.append("--compute_entropies")
    
    return cmd

def run_experiment(config, experiment_id):
    """Run a single experiment with the given configuration."""
    print(f"[BATCH] 🚀 Starting experiment {experiment_id}: {config['name']}")
    print(f"[BATCH] 📊 Config: {config['num_qubits']}q, {config['geometry']}, k={config['curvature']}, {config['device']}")
    
    start_time = time.time()
    cmd = build_command(config)
    
    try:
        # Run the experiment
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        runtime = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[BATCH] ✅ Experiment {experiment_id} completed successfully in {runtime:.1f}s")
            return {
                "experiment_id": experiment_id,
                "config": config,
                "status": "success",
                "runtime": runtime,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        else:
            print(f"[BATCH] ❌ Experiment {experiment_id} failed after {runtime:.1f}s")
            return {
                "experiment_id": experiment_id,
                "config": config,
                "status": "failed",
                "runtime": runtime,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
    except subprocess.TimeoutExpired:
        print(f"[BATCH] ⏰ Experiment {experiment_id} timed out after 10 minutes")
        return {
            "experiment_id": experiment_id,
            "config": config,
            "status": "timeout",
            "runtime": 600,
            "stdout": "",
            "stderr": "Timeout after 10 minutes"
        }
    except Exception as e:
        print(f"[BATCH] 💥 Experiment {experiment_id} crashed: {e}")
        return {
            "experiment_id": experiment_id,
            "config": config,
            "status": "crashed",
            "runtime": time.time() - start_time,
            "stdout": "",
            "stderr": str(e)
        }

def analyze_batch_results(results):
    """Analyze the results of all batch experiments."""
    print("\n" + "="*80)
    print("BATCH EXPERIMENT ANALYSIS")
    print("="*80)
    
    # Statistics
    total_experiments = len(results)
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    timed_out = sum(1 for r in results if r["status"] == "timeout")
    crashed = sum(1 for r in results if r["status"] == "crashed")
    
    print(f"📊 Total experiments: {total_experiments}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏰ Timed out: {timed_out}")
    print(f"💥 Crashed: {crashed}")
    print(f"🎯 Success rate: {successful/total_experiments*100:.1f}%")
    
    # Runtime analysis
    successful_runtimes = [r["runtime"] for r in results if r["status"] == "success"]
    if successful_runtimes:
        print(f"⏱️  Average runtime: {np.mean(successful_runtimes):.1f}s")
        print(f"⏱️  Min runtime: {np.min(successful_runtimes):.1f}s")
        print(f"⏱️  Max runtime: {np.max(successful_runtimes):.1f}s")
    
    # Backend analysis
    backend_stats = {}
    for result in results:
        backend = result["config"]["device"]
        if backend not in backend_stats:
            backend_stats[backend] = {"total": 0, "success": 0}
        backend_stats[backend]["total"] += 1
        if result["status"] == "success":
            backend_stats[backend]["success"] += 1
    
    print("\n🔧 Backend Performance:")
    for backend, stats in backend_stats.items():
        success_rate = stats["success"] / stats["total"] * 100
        print(f"  {backend}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
    
    # Parameter analysis
    print("\n📈 Parameter Analysis:")
    weight_success = {}
    curvature_success = {}
    
    for result in results:
        weight = result["config"]["weight"]
        curvature = result["config"]["curvature"]
        
        if weight not in weight_success:
            weight_success[weight] = {"total": 0, "success": 0}
        if curvature not in curvature_success:
            curvature_success[curvature] = {"total": 0, "success": 0}
        
        weight_success[weight]["total"] += 1
        curvature_success[curvature]["total"] += 1
        
        if result["status"] == "success":
            weight_success[weight]["success"] += 1
            curvature_success[curvature]["success"] += 1
    
    print("  Weight success rates:")
    for weight, stats in sorted(weight_success.items()):
        success_rate = stats["success"] / stats["total"] * 100
        print(f"    weight={weight}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
    
    print("  Curvature success rates:")
    for curvature, stats in sorted(curvature_success.items()):
        success_rate = stats["success"] / stats["total"] * 100
        print(f"    curvature={curvature}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
    
    return {
        "total_experiments": total_experiments,
        "successful": successful,
        "failed": failed,
        "timed_out": timed_out,
        "crashed": crashed,
        "success_rate": successful/total_experiments*100,
        "backend_stats": backend_stats,
        "weight_success": weight_success,
        "curvature_success": curvature_success
    }

def save_batch_results(results, analysis, output_dir):
    """Save batch results and analysis to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_file = os.path.join(output_dir, f"batch_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save analysis
    analysis_file = os.path.join(output_dir, f"batch_analysis_{timestamp}.json")
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    # Save summary report
    summary_file = os.path.join(output_dir, f"batch_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write("BATCH QUANTUM SPACETIME EXPERIMENT SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total experiments: {analysis['total_experiments']}\n")
        f.write(f"Successful: {analysis['successful']}\n")
        f.write(f"Failed: {analysis['failed']}\n")
        f.write(f"Timed out: {analysis['timed_out']}\n")
        f.write(f"Crashed: {analysis['crashed']}\n")
        f.write(f"Success rate: {analysis['success_rate']:.1f}%\n\n")
        
        f.write("Backend Performance:\n")
        for backend, stats in analysis['backend_stats'].items():
            success_rate = stats["success"] / stats["total"] * 100
            f.write(f"  {backend}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)\n")
        
        f.write("\nBest performing configurations:\n")
        successful_experiments = [r for r in results if r["status"] == "success"]
        for exp in successful_experiments[:5]:  # Top 5
            f.write(f"  {exp['config']['name']}: {exp['config']['device']}, {exp['config']['num_qubits']}q, k={exp['config']['curvature']}\n")
    
    print(f"\n💾 Results saved to:")
    print(f"  📄 Detailed results: {results_file}")
    print(f"  📊 Analysis: {analysis_file}")
    print(f"  📝 Summary: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Batch runner for quantum spacetime experiments")
    parser.add_argument("--max_workers", type=int, default=3, help="Maximum parallel workers")
    parser.add_argument("--output_dir", default="experiment_logs/batch_experiments", help="Output directory")
    parser.add_argument("--configs_only", action="store_true", help="Only show available configurations")
    parser.add_argument("--backend", type=str, help="Specific backend to test")
    parser.add_argument("--num_qubits", type=int, help="Specific number of qubits to test")
    
    args = parser.parse_args()
    
    print("🚀 BATCH QUANTUM SPACETIME EXPERIMENT RUNNER")
    print("="*60)
    
    # Get available backends
    backends = get_available_backends()
    print(f"🔧 Available backends: {', '.join(backends)}")
    
    # Create experiment configurations
    configs = create_experiment_configs()
    
    # Filter configurations if specified
    if args.backend:
        configs = [c for c in configs if c["device"] == args.backend]
        print(f"🎯 Filtered to backend: {args.backend}")
    
    if args.num_qubits:
        configs = [c for c in configs if c["num_qubits"] == args.num_qubits]
        print(f"🎯 Filtered to {args.num_qubits} qubits")
    
    if args.configs_only:
        print(f"\n📋 Available configurations ({len(configs)}):")
        for i, config in enumerate(configs):
            print(f"  {i+1}. {config['name']}: {config['num_qubits']}q, {config['geometry']}, k={config['curvature']}, {config['device']}")
        return
    
    print(f"📋 Running {len(configs)} experiments with {args.max_workers} parallel workers")
    print(f"📁 Output directory: {args.output_dir}")
    
    # Run experiments in parallel
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all experiments
        future_to_config = {
            executor.submit(run_experiment, config, i+1): config 
            for i, config in enumerate(configs)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_config):
            result = future.result()
            results.append(result)
            
            # Progress update
            completed = len(results)
            total = len(configs)
            print(f"[BATCH] Progress: {completed}/{total} ({completed/total*100:.1f}%)")
    
    total_runtime = time.time() - start_time
    
    # Analyze results
    analysis = analyze_batch_results(results)
    
    # Save results
    save_batch_results(results, analysis, args.output_dir)
    
    print(f"\n🎉 Batch experiment completed in {total_runtime:.1f}s")
    print(f"📊 Overall success rate: {analysis['success_rate']:.1f}%")

if __name__ == "__main__":
    main() 