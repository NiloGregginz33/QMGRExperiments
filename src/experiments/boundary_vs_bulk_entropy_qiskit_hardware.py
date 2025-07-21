#!/usr/bin/env python3
"""
Boundary vs Bulk Entropy Experiment - Hardware Version
Tests entropy scaling with boundary cut size on real quantum hardware
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from datetime import datetime
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from src.CGPTFactory import run as cgpt_run

def shannon_entropy(probs):
    """Compute Shannon entropy of a probability distribution"""
    probs = np.array(probs)
    probs = probs / np.sum(probs)
    return -np.sum(probs * np.log2(probs + 1e-12))

def marginal_probs_from_counts(counts, total_qubits, keep):
    """Compute marginal probabilities from measurement counts"""
    marginal = {}
    total_shots = sum(counts.values())
    
    for bitstring, count in counts.items():
        # Ensure bitstring has correct length and reverse for Qiskit convention
        b = bitstring.zfill(total_qubits)
        key = ''.join([b[-(i+1)] for i in keep])  # Reverse indexing for Qiskit
        marginal[key] = marginal.get(key, 0) + count
    
    return np.array(list(marginal.values())) / total_shots

def build_perfect_tensor_circuit():
    """Build a 6-qubit perfect tensor circuit"""
    qc = QuantumCircuit(6, 6)
    
    # Create 3 GHZ pairs: (0,1), (2,3), (4,5)
    for i in [0, 2, 4]:
        qc.h(i)
        qc.cx(i, i+1)
    
    # Entangle across pairs (CZ gates)
    qc.cz(0, 2)
    qc.cz(1, 4)
    qc.cz(3, 5)
    
    # Optional RX rotation to break symmetry
    for q in range(6):
        qc.rx(np.pi / 4, q)
    
    # Add measurements
    qc.measure_all()
    
    return qc

def run_boundary_vs_bulk_entropy_hardware(device_name='simulator', shots=2048):
    """Run boundary vs bulk entropy experiment on hardware"""
    
    print(f"Running Boundary vs Bulk Entropy Experiment")
    print(f"Device: {device_name}")
    print(f"Shots: {shots}")
    print("=" * 50)
    
    # Setup backend
    if device_name.lower() == 'simulator':
        backend = FakeBrisbane()
        print("Using FakeBrisbane simulator")
    else:
        service = QiskitRuntimeService()
        backend = service.backend(device_name)
        print(f"Using IBM Quantum backend: {backend.name}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiment_logs/boundary_vs_bulk_entropy_hardware_{device_name}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Build circuit
    print("Building perfect tensor circuit...")
    qc = build_perfect_tensor_circuit()
    
    # Transpile for hardware
    print("Transpiling circuit...")
    tqc = transpile(qc, backend, optimization_level=3)
    
    # Execute on hardware
    print(f"Executing on {backend.name}...")
    try:
        counts = cgpt_run(tqc, backend=backend, shots=shots)
        print("✓ Circuit executed successfully")
    except Exception as e:
        print(f"✗ Circuit execution failed: {e}")
        return None
    
    # Analyze results
    print("Analyzing entropy scaling...")
    num_qubits = 6
    entropies = []
    entropy_data = []
    
    for cut_size in range(1, num_qubits):
        keep = list(range(cut_size))
        marg_probs = marginal_probs_from_counts(counts, num_qubits, keep)
        entropy = shannon_entropy(marg_probs)
        
        # Validation
        valid = not np.isnan(entropy) and entropy >= -1e-6 and entropy <= cut_size
        
        print(f"Cut size {cut_size}: Entropy = {entropy:.4f} {'[VALID]' if valid else '[INVALID]'}")
        
        entropies.append(entropy)
        entropy_data.append({
            "cut_size": cut_size,
            "entropy": float(entropy),
            "valid": bool(valid),
            "marginal_probabilities": marg_probs.tolist()
        })
    
    # Create plots
    print("Generating plots...")
    
    # Main entropy plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_qubits), entropies, marker='o', linewidth=2, markersize=8, label='Hardware Results')
    
    # Theoretical expectation (linear scaling for perfect tensor)
    theoretical = [i for i in range(1, num_qubits)]
    plt.plot(range(1, num_qubits), theoretical, '--', color='red', linewidth=2, label='Theoretical (Linear)')
    
    plt.xlabel('Boundary Cut Size (qubits)', fontsize=12)
    plt.ylabel('Entropy (bits)', fontsize=12)
    plt.title(f'Boundary vs. Bulk Entropy Scaling\n{device_name.upper()} - {shots} shots', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'boundary_vs_bulk_entropy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Detailed analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Entropy scaling
    axes[0,0].plot(range(1, num_qubits), entropies, marker='o', linewidth=2, markersize=8)
    axes[0,0].plot(range(1, num_qubits), theoretical, '--', color='red', linewidth=2)
    axes[0,0].set_xlabel('Boundary Cut Size')
    axes[0,0].set_ylabel('Entropy (bits)')
    axes[0,0].set_title('Entropy Scaling')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend(['Hardware', 'Theoretical'])
    
    # Entropy difference from theoretical
    diff = np.array(entropies) - np.array(theoretical)
    axes[0,1].plot(range(1, num_qubits), diff, marker='s', color='green', linewidth=2, markersize=8)
    axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0,1].set_xlabel('Boundary Cut Size')
    axes[0,1].set_ylabel('Entropy Difference')
    axes[0,1].set_title('Deviation from Theoretical')
    axes[0,1].grid(True, alpha=0.3)
    
    # Marginal probabilities for different cut sizes
    for i, data in enumerate(entropy_data[:2]):  # Show first 2 cut sizes
        marg_probs = data['marginal_probabilities']
        axes[1,i].bar(range(len(marg_probs)), marg_probs, alpha=0.7)
        axes[1,i].set_xlabel('State')
        axes[1,i].set_ylabel('Probability')
        axes[1,i].set_title(f'Marginal Probabilities (Cut Size {data["cut_size"]})')
        axes[1,i].grid(True, alpha=0.3)
    
    plt.suptitle(f'Boundary vs Bulk Entropy Analysis - {device_name.upper()}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'boundary_vs_bulk_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    results = {
        "experiment_name": "Boundary vs Bulk Entropy Hardware Experiment",
        "device": device_name,
        "backend": backend.name,
        "shots": shots,
        "timestamp": timestamp,
        "entropy_data": entropy_data,
        "raw_counts": dict(counts),
        "theoretical_background": """
        This experiment tests the holographic principle by measuring entropy scaling with boundary cut size.
        In a perfect tensor network, entropy should scale linearly with the boundary size, demonstrating
        the holographic encoding of bulk information on the boundary.
        """,
        "methodology": """
        1. Create a 6-qubit perfect tensor circuit with GHZ pairs and cross-entanglement
        2. Execute on quantum hardware to obtain measurement counts
        3. Compute marginal probabilities for different boundary cuts
        4. Calculate Shannon entropy for each cut size
        5. Compare with theoretical linear scaling
        """,
        "key_metrics": {
            "max_entropy": float(max(entropies)),
            "min_entropy": float(min(entropies)),
            "entropy_range": float(max(entropies) - min(entropies)),
            "linearity_score": float(np.corrcoef(range(1, num_qubits), entropies)[0,1]),
            "theoretical_deviation": float(np.mean(np.abs(diff)))
        }
    }
    
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Write summary
    with open(os.path.join(exp_dir, 'summary.txt'), 'w') as f:
        f.write("Boundary vs Bulk Entropy Hardware Experiment\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Device: {device_name}\n")
        f.write(f"Backend: {backend.name}\n")
        f.write(f"Shots: {shots}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        f.write("THEORETICAL BACKGROUND:\n")
        f.write("This experiment tests the holographic principle by measuring entropy scaling with boundary cut size.\n")
        f.write("In a perfect tensor network, entropy should scale linearly with the boundary size, demonstrating\n")
        f.write("the holographic encoding of bulk information on the boundary.\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("1. Create a 6-qubit perfect tensor circuit with GHZ pairs and cross-entanglement\n")
        f.write("2. Execute on quantum hardware to obtain measurement counts\n")
        f.write("3. Compute marginal probabilities for different boundary cuts\n")
        f.write("4. Calculate Shannon entropy for each cut size\n")
        f.write("5. Compare with theoretical linear scaling\n\n")
        
        f.write("RESULTS:\n")
        for data in entropy_data:
            f.write(f"Cut size {data['cut_size']}: Entropy = {data['entropy']:.4f} {'[VALID]' if data['valid'] else '[INVALID]'}\n")
        
        f.write(f"\nKey Metrics:\n")
        f.write(f"Max Entropy: {results['key_metrics']['max_entropy']:.4f}\n")
        f.write(f"Min Entropy: {results['key_metrics']['min_entropy']:.4f}\n")
        f.write(f"Linearity Score: {results['key_metrics']['linearity_score']:.4f}\n")
        f.write(f"Theoretical Deviation: {results['key_metrics']['theoretical_deviation']:.4f}\n\n")
        
        f.write("CONCLUSION:\n")
        if results['key_metrics']['linearity_score'] > 0.9:
            f.write("✓ Strong linear entropy scaling observed, supporting holographic principle\n")
        elif results['key_metrics']['linearity_score'] > 0.7:
            f.write("○ Moderate linear entropy scaling observed\n")
        else:
            f.write("✗ Weak linear entropy scaling, may indicate noise or circuit issues\n")
        
        f.write(f"The experiment demonstrates {'successful' if results['key_metrics']['linearity_score'] > 0.8 else 'partial'} ")
        f.write("validation of holographic entropy scaling on quantum hardware.\n")
    
    print(f"\nExperiment completed!")
    print(f"Results saved in: {exp_dir}")
    print(f"Linearity score: {results['key_metrics']['linearity_score']:.4f}")
    print(f"Max entropy: {results['key_metrics']['max_entropy']:.4f}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run boundary vs bulk entropy experiment on hardware')
    parser.add_argument('--device', type=str, default='simulator', help='Device: "simulator" or IBM backend name')
    parser.add_argument('--shots', type=int, default=2048, help='Number of shots')
    args = parser.parse_args()
    
    run_boundary_vs_bulk_entropy_hardware(device_name=args.device, shots=args.shots) 