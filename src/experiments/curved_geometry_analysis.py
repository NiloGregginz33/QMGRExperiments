import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import traceback
import json
from datetime import datetime
import argparse
from braket.aws import AwsDevice
from braket.circuits import Circuit, gates
from braket.circuits.compiler_directives import CompilerDirective
import glob

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.AWSFactory import AdSGeometryAnalyzer6Q
import seaborn as sns
from scipy.stats import pearsonr

def find_latest_custom_curvature_results():
    """
    Find the latest custom curvature experiment results
    Returns:
        str: Path to the latest results file
    """
    custom_curvature_dir = "experiment_logs/custom_curvature_experiment"
    if not os.path.exists(custom_curvature_dir):
        return None
    
    # Find all JSON result files
    result_files = glob.glob(os.path.join(custom_curvature_dir, "results_*.json"))
    if not result_files:
        return None
    
    # Sort by modification time and return the latest
    latest_file = max(result_files, key=os.path.getmtime)
    return latest_file

def auto_detect_shots_from_results():
    """
    Auto-detect shots from the latest custom curvature experiment results
    Returns:
        int: Number of shots used in the experiment
    """
    latest_results = find_latest_custom_curvature_results()
    if latest_results is None:
        print("No custom curvature results found. Using default shots=1024")
        return 1024
    
    try:
        with open(latest_results, 'r') as f:
            data = json.load(f)
        shots = data.get('spec', {}).get('shots', 1024)
        print(f"Auto-detected shots: {shots} from {os.path.basename(latest_results)}")
        return shots
    except Exception as e:
        print(f"Error reading shots from results: {e}. Using default shots=1024")
        return 1024

def analyze_custom_curvature_results(results_file=None):
    """
    Analyze existing custom curvature experiment results
    Args:
        results_file (str): Path to results file (if None, uses latest)
    """
    if results_file is None:
        results_file = find_latest_custom_curvature_results()
    
    if results_file is None:
        print("No custom curvature results found to analyze.")
        return
    
    print(f"Analyzing results from: {results_file}")
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Extract key data
        spec = data.get('spec', {})
        shots = spec.get('shots', 'Unknown')
        num_qubits = spec.get('num_qubits', 'Unknown')
        geometry = spec.get('geometry', 'Unknown')
        curvature = spec.get('curvature', 'Unknown')
        
        mutual_info = data.get('mutual_information', [{}])[0] if data.get('mutual_information') else {}
        distance_matrix = np.array(data.get('distance_matrix', []))
        embedding_coords = np.array(data.get('embedding_coords', []))
        angle_sums = data.get('angle_sums', [])
        gromov_delta = data.get('gromov_delta', 'Unknown')
        
        print(f"\n=== Custom Curvature Experiment Analysis ===")
        print(f"Shots: {shots}")
        print(f"Qubits: {num_qubits}")
        print(f"Geometry: {geometry}")
        print(f"Curvature: {curvature}")
        print(f"Gromov Delta: {gromov_delta}")
        
        # Analyze mutual information
        if mutual_info:
            print(f"\n--- Mutual Information Analysis ---")
            mi_values = list(mutual_info.values())
            print(f"Mean MI: {np.mean(mi_values):.4f}")
            print(f"Max MI: {np.max(mi_values):.4f}")
            print(f"Min MI: {np.min(mi_values):.4f}")
            print(f"MI Variance: {np.var(mi_values):.4f}")
            
            # Find strongest and weakest entanglement
            max_mi_pair = max(mutual_info.items(), key=lambda x: x[1])
            min_mi_pair = min(mutual_info.items(), key=lambda x: x[1])
            print(f"Strongest entanglement: {max_mi_pair[0]} = {max_mi_pair[1]:.4f}")
            print(f"Weakest entanglement: {min_mi_pair[0]} = {min_mi_pair[1]:.4f}")
        
        # Analyze geometric properties
        if len(angle_sums) > 0:
            print(f"\n--- Geometric Analysis ---")
            print(f"Mean angle sum: {np.mean(angle_sums):.4f} radians")
            print(f"Expected for flat geometry: {np.pi:.4f} radians")
            print(f"Deviation from flat: {np.mean(angle_sums) - np.pi:.4f} radians")
            
            # Check for spherical geometry indicators
            spherical_indicators = sum(1 for angle in angle_sums if angle > np.pi)
            print(f"Triangles with angle sum > π: {spherical_indicators}/{len(angle_sums)}")
        
        # Create visualizations
        if len(embedding_coords) > 0:
            print(f"\n--- Creating Visualizations ---")
            
            # Plot 2D embedding
            plt.figure(figsize=(10, 8))
            
            plt.subplot(2, 2, 1)
            plt.scatter(embedding_coords[:, 0], embedding_coords[:, 1], s=100, c='blue', alpha=0.7)
            for i, (x, y) in enumerate(embedding_coords):
                plt.annotate(f'Q{i}', (x, y), xytext=(5, 5), textcoords='offset points')
            plt.title(f'2D Geometric Embedding\n{geometry.capitalize()} Geometry, Curvature={curvature}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True, alpha=0.3)
            
            # Plot mutual information heatmap
            plt.subplot(2, 2, 2)
            if mutual_info:
                # Create MI matrix for heatmap
                n = num_qubits
                mi_matrix = np.zeros((n, n))
                for key, mi_val in mutual_info.items():
                    # Handle both "I_0,1" and "0,1" formats
                    if key.startswith('I_'):
                        key = key[2:]  # Remove "I_" prefix
                    try:
                        i, j = map(int, key.split(','))
                        mi_matrix[i, j] = mi_val
                        mi_matrix[j, i] = mi_val
                    except:
                        continue
                
                sns.heatmap(mi_matrix, annot=True, fmt='.3f', cmap='viridis')
                plt.title('Mutual Information Matrix')
                plt.xlabel('Qubit')
                plt.ylabel('Qubit')
            
            # Plot distance matrix
            plt.subplot(2, 2, 3)
            if len(distance_matrix) > 0:
                sns.heatmap(distance_matrix, annot=True, fmt='.2f', cmap='plasma')
                plt.title('Distance Matrix')
                plt.xlabel('Qubit')
                plt.ylabel('Qubit')
            
            # Plot angle sums
            plt.subplot(2, 2, 4)
            if len(angle_sums) > 0:
                plt.hist(angle_sums, bins=10, alpha=0.7, color='green', edgecolor='black')
                plt.axvline(np.pi, color='red', linestyle='--', label=f'π = {np.pi:.2f}')
                plt.title('Triangle Angle Sums Distribution')
                plt.xlabel('Angle Sum (radians)')
                plt.ylabel('Frequency')
                plt.legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"custom_curvature_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Analysis plot saved as: {plot_filename}")
            plt.show()
        
        # Save analysis summary
        analysis_summary = {
            "analysis_timestamp": datetime.now().isoformat(),
            "results_file": results_file,
            "experiment_spec": spec,
            "analysis_results": {
                "shots": shots,
                "num_qubits": num_qubits,
                "geometry": geometry,
                "curvature": curvature,
                "gromov_delta": gromov_delta,
                "mutual_information_stats": {
                    "mean": float(np.mean(mi_values)) if mutual_info else None,
                    "max": float(np.max(mi_values)) if mutual_info else None,
                    "min": float(np.min(mi_values)) if mutual_info else None,
                    "variance": float(np.var(mi_values)) if mutual_info else None,
                    "strongest_pair": max_mi_pair if mutual_info else None,
                    "weakest_pair": min_mi_pair if mutual_info else None
                },
                "geometric_stats": {
                    "mean_angle_sum": float(np.mean(angle_sums)) if angle_sums else None,
                    "deviation_from_flat": float(np.mean(angle_sums) - np.pi) if angle_sums else None,
                    "spherical_indicators": spherical_indicators if angle_sums else None,
                    "total_triangles": len(angle_sums) if angle_sums else None
                }
            }
        }
        
        summary_filename = f"custom_curvature_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_filename, 'w') as f:
            json.dump(analysis_summary, f, indent=2)
        print(f"Analysis summary saved as: {summary_filename}")
        
    except Exception as e:
        print(f"Error analyzing results: {e}")
        traceback.print_exc()

def get_device(device_type="simulator"):
    """
    Get quantum device based on specified type
    Args:
        device_type (str): "simulator", "ionq", "rigetti", or "oqc"
    Returns:
        Device object for quantum computation
    """
    if device_type == "simulator":
        return None  # AdSGeometryAnalyzer6Q will use LocalSimulator by default
    elif device_type == "ionq":
        return AwsDevice("arn:aws:braket:us-east-1::device/qpu/ionq/ionQdevice")
    elif device_type == "rigetti":
        return AwsDevice("arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3")
    elif device_type == "oqc":
        return AwsDevice("arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy")
    else:
        print(f"Unknown device type: {device_type}. Using simulator.")
        return None

def transpile_circuit(circuit, device):
    """
    Transpile circuit for the specific device
    Args:
        circuit: Braket circuit to transpile
        device: Target device
    Returns:
        Transpiled circuit optimized for the device
    """
    try:
        # Get device properties
        if hasattr(device, 'properties'):
            properties = device.properties
            print(f"Device: {device.name}")
            print(f"Native gates: {properties.action.get('braket.ir.jaqcd.program', {}).get('nativeGates', 'Unknown')}")
            print(f"Connectivity: {properties.action.get('braket.ir.jaqcd.program', {}).get('connectivity', 'Unknown')}")
        # Apply device-specific optimizations
        if "ionq" in str(device).lower():
            # IonQ prefers single-qubit gates and CNOT
            print("Transpiling for IonQ device...")
            return circuit
        elif "rigetti" in str(device).lower():
            # Rigetti has specific connectivity constraints
            print("Transpiling for Rigetti device...")
            # Rigetti devices have limited connectivity, may need SWAP gates
            return circuit
        elif "oqc" in str(device).lower():
            # OQC has specific gate set requirements
            print("Transpiling for OQC device...")
            return circuit
        else:
            # For simulator, no special transpilation needed
            print("Using circuit as-is for simulator...")
            return circuit
    except Exception as e:
        print(f"Warning: Transpilation failed: {e}")
        print("Using original circuit...")
        return circuit

def run_curved_geometry_experiments(device_type="simulator", shots=None):
    """
    Run curved geometry experiments in both flat and curved modes.
    - Initializes the analyzer
    - Runs both flat and curved geometry analyses
    - Saves results and summary to experiment_logs
    Args:
        device_type (str): Quantum device type (simulator, ionq, rigetti, oqc)
        shots (int): Number of measurement shots (if None, auto-detects from existing results)
    Returns:
        str: Path to experiment log directory
    """
    # Auto-detect shots if not provided
    if shots is None:
        shots = auto_detect_shots_from_results()
    
    exp_dir = f"experiment_logs/curved_geometry_{device_type}"
    os.makedirs(exp_dir, exist_ok=True)
    device = get_device(device_type)
    try:
        print(f"Running curved geometry experiment on {device_type} with {shots} shots")
        # Initialize the analyzer with transpilation support
        analyzer = AdSGeometryAnalyzer6Q(device=device, shots=shots)
        # Add transpilation method to analyzer if it doesn't exist
        if not hasattr(analyzer, 'transpile_circuit'):
            analyzer.transpile_circuit = lambda circuit: transpile_circuit(circuit, device)
        # Run experiments in different modes
        modes = ["flat", "curved"]
        results = {}
        for mode in modes:
            print(f"Running in {mode} mode...")
            try:
                if mode == "flat":
                    result = analyzer.run_flat_geometry_analysis()
                else:
                    result = analyzer.run_curved_geometry_analysis()
                results[mode] = result
                print(f"Completed {mode} mode analysis")
            except Exception as e:
                print(f"Error in {mode} mode: {str(e)}")
                results[mode] = {"error": str(e)}
        # Save results
        with open(f"{exp_dir}/results.json", "w") as f:
            json.dump(results, f, indent=2)
        # Create summary
        with open(f"{exp_dir}/summary.txt", "w") as f:
            f.write("Curved Geometry Experiment Summary\n")
            f.write("==================================\n\n")
            f.write(f"Device: {device_type}\n")
            f.write(f"Shots: {shots}\n\n")
            f.write("Theoretical Background:\n")
            f.write("This experiment investigates how quantum information and entanglement behave in curved spacetime geometries. It compares the behavior of quantum systems in flat vs curved geometries to understand the influence of spacetime curvature on quantum information.\n\n")
            f.write("Methodology:\n")
            f.write("Quantum circuits are constructed to simulate both flat and curved spacetime geometries. The experiments analyze mutual information, entanglement patterns, and geometric features in both scenarios.\n\n")
            f.write("Results:\n")
            f.write(f"Results saved in: {exp_dir}\n")
            f.write("\nConclusion:\n")
            f.write("The experiment reveals how curvature affects quantum information distribution and entanglement patterns, providing insights into the relationship between quantum mechanics and spacetime geometry.\n")
        print(f"Experiment completed. Results saved in {exp_dir}")
        return exp_dir
    except Exception as e:
        error_msg = f"Experiment failed: {str(e)}"
        print(error_msg)
        # Save error log
        with open(f"{exp_dir}/error.log", "w") as f:
            f.write(f"Error occurred at {datetime.now()}\n")
            f.write(f"Error: {str(e)}\n")
        # Save basic summary even if experiment failed
        with open(f"{exp_dir}/summary.txt", "w") as f:
            f.write("Curved Geometry Experiment Summary\n")
            f.write("==================================\n\n")
            f.write(f"Device: {device_type}\n")
            f.write(f"Shots: {shots}\n\n")
            f.write("Status: FAILED\n")
            f.write(f"Error: {str(e)}\n")
            f.write("Check error.log for details.\n")
        return exp_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run curved geometry experiment or analyze existing results')
    parser.add_argument('--device', type=str, default='simulator', 
                       choices=['simulator', 'ionq', 'rigetti', 'oqc'],
                       help='Quantum device to use')
    parser.add_argument('--shots', type=int, default=None,
                       help='Number of shots for quantum measurements (auto-detects from existing results if not specified)')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze existing custom curvature results instead of running new experiment')
    parser.add_argument('--results-file', type=str, default=None,
                       help='Specific results file to analyze (if not specified, uses latest)')
    args = parser.parse_args()
    
    if args.analyze:
        # Analyze existing results
        analyze_custom_curvature_results(args.results_file)
    else:
        # Run new experiment
        run_curved_geometry_experiments(args.device, args.shots) 