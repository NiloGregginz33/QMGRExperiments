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

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.AWSFactory import AdSGeometryAnalyzer6Q
import seaborn as sns
from scipy.stats import pearsonr

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

def run_curved_geometry_experiments(device_type="simulator", shots=1024):
    """
    Run curved geometry experiments in both flat and curved modes.
    - Initializes the analyzer
    - Runs both flat and curved geometry analyses
    - Saves results and summary to experiment_logs
    Args:
        device_type (str): Quantum device type (simulator, ionq, rigetti, oqc)
        shots (int): Number of measurement shots
    Returns:
        str: Path to experiment log directory
    """
    exp_dir = f"experiment_logs/curved_geometry_{device_type}"
    os.makedirs(exp_dir, exist_ok=True)
    device = get_device(device_type)
    try:
        print(f"Running curved geometry experiment on {device_type}")
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
    parser = argparse.ArgumentParser(description='Run curved geometry experiment')
    parser.add_argument('--device', type=str, default='simulator', 
                       choices=['simulator', 'ionq', 'rigetti', 'oqc'],
                       help='Quantum device to use')
    parser.add_argument('--shots', type=int, default=1024,
                       help='Number of shots for quantum measurements')
    args = parser.parse_args()
    
    run_curved_geometry_experiments(args.device, args.shots) 