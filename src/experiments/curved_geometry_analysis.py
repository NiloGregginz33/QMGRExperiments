import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.AWSFactory import AdSGeometryAnalyzer6Q
import seaborn as sns
from scipy.stats import pearsonr
import json
from datetime import datetime

def run_curved_geometry_experiments():
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiment_logs/curved_geometry_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Initialize results storage
    results = {
        "parameters": [],
        "curvatures": [],
        "entropies": [],
        "distances": [],
        "mi_matrices": []
    }
    
    # Run experiments for different modes
    modes = ["flat", "curved"]
    for mode in modes:
        print(f"\nRunning experiments in {mode} mode...")
        from braket.devices import LocalSimulator
        analyzer = AdSGeometryAnalyzer6Q(n_qubits=6, timesteps=15, mode=mode, device=LocalSimulator())
        analyzer.run()
        
        # Store results
        results["parameters"].extend([mode] * len(analyzer.rt_data))
        results["entropies"].extend([s for _, s, _ in analyzer.rt_data])
        results["distances"].extend([d for _, _, d in analyzer.rt_data])
        
        # Calculate and store curvatures
        for coords3 in analyzer.coords_list_3d:
            curvatures = []
            for triplet in [(0,1,2), (1,2,3), (2,3,4), (3,4,5)]:
                curv = analyzer.estimate_local_curvature(coords3, triplet)
                curvatures.append(curv)
            results["curvatures"].append(np.mean(curvatures))
        
        # Store MI matrices
        results["mi_matrices"].extend(analyzer.mi_matrices)
        
        # Generate and save plots
        plt.figure(figsize=(10, 6))
        plt.plot(analyzer.timesteps, [s for _, s, _ in analyzer.rt_data], 
                label=f"Entropy ({mode})")
        plt.xlabel("Time (φ)")
        plt.ylabel("Entropy")
        plt.title(f"Entropy Evolution - {mode} Geometry")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{exp_dir}/entropy_{mode}.png")
        plt.close()
        
        # Plot RT correlation
        analyzer.fit_rt_plot()
        plt.savefig(f"{exp_dir}/rt_correlation_{mode}.png")
        plt.close()
    
    # Analyze correlations
    correlations = {
        "entropy_distance": pearsonr(results["entropies"], results["distances"]),
        "curvature_entropy": pearsonr(results["curvatures"], results["entropies"]),
        "curvature_distance": pearsonr(results["curvatures"], results["distances"])
    }
    
    # Save results
    with open(f"{exp_dir}/results.json", "w") as f:
        json.dump({
            "correlations": {
                k: {"correlation": float(v[0]), "p_value": float(v[1])} 
                for k, v in correlations.items()
            },
            "summary": {
                "modes": modes,
                "num_timesteps": 15,
                "num_qubits": 6
            }
        }, f, indent=2)
    
    # Generate correlation heatmap
    plt.figure(figsize=(8, 6))
    correlation_matrix = np.array([
        [1, correlations["entropy_distance"][0], correlations["curvature_entropy"][0]],
        [correlations["entropy_distance"][0], 1, correlations["curvature_distance"][0]],
        [correlations["curvature_entropy"][0], correlations["curvature_distance"][0], 1]
    ])
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap="coolwarm", 
                xticklabels=["Entropy", "Distance", "Curvature"],
                yticklabels=["Entropy", "Distance", "Curvature"])
    plt.title("Parameter Correlations")
    plt.tight_layout()
    plt.savefig(f"{exp_dir}/correlation_heatmap.png")
    plt.close()
    
    # Print summary
    print("\nExperiment Summary:")
    print(f"Results saved in: {exp_dir}")
    print("\nCorrelations:")
    for metric, (corr, p_val) in correlations.items():
        print(f"{metric}: {corr:.3f} (p-value: {p_val:.3e})")
    
    return exp_dir

if __name__ == "__main__":
    exp_dir = run_curved_geometry_experiments() 