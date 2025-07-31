#!/usr/bin/env python3
"""
ZNE-Enhanced Quantum Experiment
===============================

This script runs quantum experiments with Zero Noise Extrapolation (ZNE)
using the CGPTFactory run function as the base.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix
import json
import argparse
from datetime import datetime

# Import CGPTFactory functions
from CGPTFactory import run, get_best_backend, service


def create_quantum_spacetime_circuit(num_qubits, entanglement_strength=3.0, circuit_depth=8):
    """Create a highly entangled quantum spacetime circuit."""
    qc = QuantumCircuit(num_qubits)
    
    # Initial Hadamard on all qubits for superposition
    qc.h(range(num_qubits))
    
    # Apply layers of entangling gates
    for _ in range(circuit_depth):
        # Apply RZZ, RYY, RXX gates for strong, varied entanglement
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                qc.rzz(entanglement_strength * np.pi / 4, i, j)
                qc.ryy(entanglement_strength * np.pi / 4, i, j)
                qc.rxx(entanglement_strength * np.pi / 4, i, j)
        
        # Add CNOTs for additional entanglement
        for i in range(0, num_qubits - 1, 2):
            qc.cx(i, i + 1)
        for i in range(1, num_qubits - 1, 2):
            qc.cx(i, i + 1)
        
        # Apply random single-qubit rotations
        for i in range(num_qubits):
            qc.rz(np.random.rand() * np.pi, i)
            qc.ry(np.random.rand() * np.pi, i)
    
    # Add measurement
    qc.measure_all()
    
    return qc


def run_with_zne(qc, backend=None, shots=2048, noise_factors=[1.0, 2.0, 3.0]):
    """
    Run circuit with Zero Noise Extrapolation using CGPTFactory run function.
    """
    if backend is None:
        backend = get_best_backend(service)
    
    print(f"[ZNE] Running with noise factors: {noise_factors}")
    
    # Store results for each noise factor
    noise_results = []
    
    for i, noise_factor in enumerate(noise_factors):
        print(f"[ZNE] Running with noise factor {noise_factor} ({i+1}/{len(noise_factors)})")
        
        # Create noise-scaled circuit
        scaled_circuit = create_noise_scaled_circuit(qc, noise_factor)
        
        # Run using CGPTFactory run function
        try:
            result = run(scaled_circuit, backend=backend, shots=shots)
            
            # Handle different result formats from CGPTFactory
            counts = {}
            if isinstance(result, dict) and 'counts' in result:
                counts = result['counts']
            elif isinstance(result, dict):
                # Try to extract counts from various possible formats
                for key in ['counts', 'distribution', 'quasi_dists']:
                    if key in result:
                        if key == 'quasi_dists':
                            # Convert quasi_dists to counts
                            quasi_dists = result[key]
                            for bitstring, probability in quasi_dists.items():
                                count = int(probability * shots)
                                if count > 0:
                                    counts[bitstring] = count
                        else:
                            counts = result[key]
                        break
            
            if counts:
                # Calculate success metric (total probability)
                total_shots = sum(counts.values())
                success_metric = total_shots / shots if total_shots > 0 else 0.0
                noise_results.append((noise_factor, success_metric, counts))
                
                print(f"[ZNE] Noise factor {noise_factor}: success metric = {success_metric:.4f}")
            else:
                print(f"[ZNE] No valid counts for noise factor {noise_factor}")
                noise_results.append((noise_factor, 0.0, {}))
                
        except Exception as e:
            print(f"[ZNE] Error with noise factor {noise_factor}: {e}")
            noise_results.append((noise_factor, 0.0, {}))
    
    # Extrapolate to zero noise
    if len(noise_results) >= 2:
        noise_factors_vals = [r[0] for r in noise_results]
        success_metrics = [r[1] for r in noise_results]
        
        # Linear extrapolation
        x = np.array(noise_factors_vals)
        y = np.array(success_metrics)
        coeffs = np.polyfit(x, y, 1)
        extrapolated_metric = max(0.0, coeffs[-1])  # intercept
        
        # Apply correction
        base_counts = noise_results[0][2]
        base_metric = noise_results[0][1]
        
        if extrapolated_metric > 0 and base_metric > 0:
            correction_factor = extrapolated_metric / base_metric
            corrected_counts = {}
            for bitstring, count in base_counts.items():
                corrected_count = int(count * correction_factor)
                if corrected_count > 0:
                    corrected_counts[bitstring] = corrected_count
            
            print(f"[ZNE] Extrapolated metric: {extrapolated_metric:.4f}, Correction: {correction_factor:.4f}")
            
            return {
                'counts': corrected_counts,
                'zne_info': {
                    'noise_factors': noise_factors_vals,
                    'success_metrics': success_metrics,
                    'extrapolated_metric': extrapolated_metric,
                    'correction_factor': correction_factor
                }
            }
    
    # Fallback
    print("[ZNE] Extrapolation failed, using base result")
    return {
        'counts': noise_results[0][2] if noise_results else {},
        'zne_info': {'error': 'Extrapolation failed'}
    }


def create_noise_scaled_circuit(qc, noise_factor):
    """Create a noise-scaled version of the circuit."""
    if noise_factor <= 1.0:
        return qc.copy()
    
    # Create a new circuit with the same number of qubits and classical bits
    new_qc = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    
    # Use the new Qiskit format for iterating over circuit data
    for instruction in qc.data:
        # Add the original instruction
        new_qc.append(instruction.operation, instruction.qubits, instruction.clbits)
        
        # Add identity gates for noise scaling
        if noise_factor > 1.5:
            for qubit in instruction.qubits:
                new_qc.id(qubit)
    
    return new_qc


def compute_mutual_information_matrix(counts, num_qubits):
    """Compute mutual information matrix from measurement counts."""
    if not counts:
        return {}
    
    # Convert counts to probability distribution
    total_shots = sum(counts.values())
    probabilities = {bitstring: count / total_shots for bitstring, count in counts.items()}
    
    # Compute mutual information for each pair of qubits
    mi_matrix = {}
    
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            # Compute marginal probabilities for qubits i and j
            p_i = {0: 0.0, 1: 0.0}
            p_j = {0: 0.0, 1: 0.0}
            p_ij = {(0, 0): 0.0, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.0}
            
            for bitstring, prob in probabilities.items():
                if len(bitstring) >= max(i, j) + 1:
                    bit_i = int(bitstring[-(i+1)])
                    bit_j = int(bitstring[-(j+1)])
                    
                    p_i[bit_i] += prob
                    p_j[bit_j] += prob
                    p_ij[(bit_i, bit_j)] += prob
            
            # Compute mutual information
            mi = 0.0
            for bi in [0, 1]:
                for bj in [0, 1]:
                    if p_ij[(bi, bj)] > 0 and p_i[bi] > 0 and p_j[bj] > 0:
                        mi += p_ij[(bi, bj)] * np.log2(p_ij[(bi, bj)] / (p_i[bi] * p_j[bj]))
            
            mi_matrix[f"I_{i},{j}"] = max(0.0, mi)
    
    return mi_matrix


def main():
    parser = argparse.ArgumentParser(description="Run ZNE-enhanced quantum experiment")
    parser.add_argument("--num_qubits", type=int, default=4, help="Number of qubits")
    parser.add_argument("--entanglement_strength", type=float, default=3.0, help="Entanglement strength")
    parser.add_argument("--circuit_depth", type=int, default=8, help="Circuit depth")
    parser.add_argument("--shots", type=int, default=2000, help="Number of shots")
    parser.add_argument("--noise_factors", type=float, nargs='+', default=[1.0, 2.0, 3.0], help="Noise factors for ZNE")
    parser.add_argument("--device", type=str, default="ibm_brisbane", help="Backend device")
    
    args = parser.parse_args()
    
    print(f"[ZNE EXPERIMENT] Starting ZNE-enhanced quantum experiment")
    print(f"[ZNE EXPERIMENT] Parameters: {args.num_qubits} qubits, strength={args.entanglement_strength}, depth={args.circuit_depth}")
    print(f"[ZNE EXPERIMENT] Device: {args.device}, Shots: {args.shots}")
    
    # Create quantum spacetime circuit
    qc = create_quantum_spacetime_circuit(args.num_qubits, args.entanglement_strength, args.circuit_depth)
    
    # Get backend
    backend = get_best_backend(service)
    print(f"[ZNE EXPERIMENT] Using backend: {backend.name}")
    
    # Run with ZNE
    zne_result = run_with_zne(qc, backend=backend, shots=args.shots, noise_factors=args.noise_factors)
    
    # Compute mutual information matrix
    mi_matrix = compute_mutual_information_matrix(zne_result['counts'], args.num_qubits)
    
    # Create results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "experiment_type": "zne_quantum_spacetime",
        "timestamp": timestamp,
        "parameters": {
            "num_qubits": args.num_qubits,
            "entanglement_strength": args.entanglement_strength,
            "circuit_depth": args.circuit_depth,
            "shots": args.shots,
            "noise_factors": args.noise_factors,
            "device": args.device
        },
        "results": {
            "counts": zne_result['counts'],
            "mutual_information_matrix": mi_matrix,
            "zne_info": zne_result['zne_info']
        }
    }
    
    # Save results
    output_dir = f"experiment_logs/zne_quantum_experiment/instance_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"results_n{args.num_qubits}_zne_{args.device}_{timestamp[:8]}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"[ZNE EXPERIMENT] Results saved to: {filepath}")
    print(f"[ZNE EXPERIMENT] ZNE correction factor: {zne_result['zne_info'].get('correction_factor', 'N/A')}")
    print(f"[ZNE EXPERIMENT] Extrapolated metric: {zne_result['zne_info'].get('extrapolated_metric', 'N/A')}")


if __name__ == "__main__":
    main() 