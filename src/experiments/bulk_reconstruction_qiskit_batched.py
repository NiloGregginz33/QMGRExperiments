#!/usr/bin/env python3
"""
Batched Bulk Reconstruction Qiskit Experiment
Runs all parameter combinations in a single hardware submission for efficiency
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeJakartaV2
from qiskit_ibm_runtime import QiskitRuntimeService
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) )
try:
    from src.CGPTFactory import run as cgpt_run
except ImportError:
    cgpt_run = None
from qiskit.quantum_info import partial_trace, entropy
from pathlib import Path
import argparse
import json
from datetime import datetime

def log_result(metrics, log_dir="experiment_logs/bulk_reconstruction_qiskit_batched"):
    os.makedirs(log_dir, exist_ok=True)
    idx = len([f for f in os.listdir(log_dir) if f.startswith('result_') and f.endswith('.json')]) + 1
    with open(os.path.join(log_dir, f"result_{idx}.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

class BatchedBulkReconstructionQiskit:
    def __init__(self, backend=None, shots=2048):
        self.num_qubits = 7  # 6 boundary + 1 bulk
        self.alpha_values = [1, 2, 3]
        self.backend = backend if backend is not None else FakeJakartaV2()
        self.shots = shots

    def shannon_entropy(self, probs):
        probs = np.array(probs)
        probs = probs / np.sum(probs)
        return -np.sum(probs * np.log2(probs + 1e-12))

    def renyi_entropy(self, probs, alpha):
        probs = np.array(probs)
        probs = probs / np.sum(probs)
        if alpha == 1:
            return self.shannon_entropy(probs)
        return 1/(1-alpha) * np.log2(np.sum(probs**alpha) + 1e-12)

    def marginal_fisher_info(self, p_bulk_current, p_bulk_prev, phi_current, phi_prev):
        delta_phi = phi_current - phi_prev
        if delta_phi == 0:
            return 0
        derivative = (np.array(p_bulk_current) - np.array(p_bulk_prev)) / delta_phi
        fisher = np.sum((derivative ** 2) / (np.array(p_bulk_current) + 1e-12))
        return fisher

    def build_happy_circuit(self, logical_state=0, phi=0):
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        # Create perfect tensor (6 qubits)
        for i in [0, 2, 4]:
            qc.h(i)
            qc.cx(i, i+1)
        # Entangle across pairs
        qc.cx(0, 2)
        qc.rz(np.pi, 2)
        qc.cx(0, 2)
        qc.cx(1, 4)
        qc.rz(np.pi, 4)
        qc.cx(1, 4)
        qc.cx(3, 5)
        qc.rz(np.pi, 5)
        qc.cx(3, 5)
        # Add bulk qubit (index 6) and encode logical state
        qc.h(6)
        if logical_state:
            qc.x(6)
        qc.cx(6, 2)
        qc.rz(phi, 6)
        # Add measurement to all qubits
        qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc

    def run_batched(self):
        """Run all parameter combinations in a single batch"""
        print("Building batched circuits...")
        
        # Generate all parameter combinations
        phi_values = np.linspace(0, 2*np.pi, 10)
        logical_states = [0, 1]
        
        # Create all circuits
        circuits = []
        circuit_params = []  # Track which parameters each circuit corresponds to
        
        for logical_state in logical_states:
            for phi_val in phi_values:
                qc = self.build_happy_circuit(logical_state, phi_val)
                circuits.append(qc)
                circuit_params.append((logical_state, phi_val))
        
        print(f"Created {len(circuits)} circuits for batch execution")
        print(f"Parameters: {len(logical_states)} logical states × {len(phi_values)} phi values")
        
        # Transpile all circuits
        print("Transpiling circuits...")
        transpiled_circuits = transpile(circuits, self.backend)
        
        # Execute all circuits in a single batch
        print(f"Submitting batch job to {self.backend.name}...")
        print(f"Total shots: {self.shots} × {len(circuits)} = {self.shots * len(circuits)}")
        
        if cgpt_run is not None:
            # Use CGPTFactory for batch execution
            all_counts = []
            for i, tqc in enumerate(transpiled_circuits):
                print(f"Running circuit {i+1}/{len(transpiled_circuits)}...")
                counts = cgpt_run(tqc, backend=self.backend, shots=self.shots)
                all_counts.append(counts)
        else:
            # Fallback: run circuits sequentially (not ideal but functional)
            all_counts = []
            for i, tqc in enumerate(transpiled_circuits):
                print(f"Running circuit {i+1}/{len(transpiled_circuits)}...")
                job = self.backend.run(tqc, shots=self.shots)
                result = job.result()
                counts = result.get_counts()
                all_counts.append(counts)
        
        print("Processing results...")
        
        # Process all results
        results = []
        for i, (counts, (logical_state, phi_val)) in enumerate(zip(all_counts, circuit_params)):
            # Convert counts to probabilities
            probs = np.zeros(2**self.num_qubits)
            for bitstring, count in counts.items():
                idx = int(bitstring.replace(' ', ''), 2)
                probs[idx] = count / self.shots
            
            # Reduced density matrix for bulk qubit (index 6)
            p_bulk0 = sum(probs[i] for i in range(len(probs)) if (i >> 6) & 1 == 0)
            p_bulk1 = sum(probs[i] for i in range(len(probs)) if (i >> 6) & 1 == 1)
            
            # Compute metrics
            shannon_entropy = self.shannon_entropy(probs)
            renyi_entropies = {f"S_{alpha}": self.renyi_entropy(probs, alpha) for alpha in self.alpha_values}
            vn_entropy = self.shannon_entropy([p_bulk0, p_bulk1])
            ent_spectrum = [p_bulk0, p_bulk1]
            
            # Fisher info (simplified - no previous state comparison in batch)
            fisher_info = 0  # Will be computed in post-processing
            
            metrics = {
                "logical_state": logical_state,
                "phi": phi_val,
                "shannon_entropy": shannon_entropy,
                "renyi_entropies": renyi_entropies,
                "von_neumann_entropy": vn_entropy,
                "entanglement_spectrum": ent_spectrum,
                "fisher_info": fisher_info,
                "circuit_index": i
            }
            
            print(f"Circuit {i+1}: Logical state {logical_state}, phi={phi_val:.2f}, Shannon entropy: {shannon_entropy:.4f}")
            results.append(metrics)
        
        # Post-process to compute Fisher information
        print("Computing Fisher information...")
        for logical_state in logical_states:
            state_results = [r for r in results if r['logical_state'] == logical_state]
            state_results.sort(key=lambda x: x['phi'])
            
            for i in range(1, len(state_results)):
                prev = state_results[i-1]
                curr = state_results[i]
                
                p_bulk_prev = prev['entanglement_spectrum']
                p_bulk_curr = curr['entanglement_spectrum']
                phi_prev = prev['phi']
                phi_curr = curr['phi']
                
                fisher_info = self.marginal_fisher_info(p_bulk_curr, p_bulk_prev, phi_curr, phi_prev)
                curr['fisher_info'] = fisher_info
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"experiment_logs/bulk_reconstruction_qiskit_batched_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        
        # Save all results
        with open(os.path.join(log_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Write summary
        with open(os.path.join(log_dir, "summary.txt"), 'w') as f:
            f.write(f"Batched Bulk Reconstruction Qiskit Experiment\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Backend: {self.backend.name}\n")
            f.write(f"Shots per circuit: {self.shots}\n")
            f.write(f"Total circuits: {len(circuits)}\n")
            f.write(f"Total shots: {self.shots * len(circuits)}\n")
            f.write(f"Num Qubits: {self.num_qubits}\n")
            f.write(f"Alpha values: {self.alpha_values}\n")
            f.write(f"Phi values: {list(phi_values)}\n")
            f.write(f"Logical states: {logical_states}\n\n")
            
            for state in logical_states:
                state_results = [r for r in results if r['logical_state'] == state]
                f.write(f"Logical state {state}:\n")
                for r in state_results:
                    f.write(f"  phi={r['phi']:.2f}, Shannon Entropy={r['shannon_entropy']:.4f}, Renyi S_2={r['renyi_entropies']['S_2']:.4f}, Fisher Info={r['fisher_info']:.4f}\n")
                f.write("\n")
        
        # Create plots
        self.plot_results(results, log_dir)
        
        print(f"Experiment completed! Results saved to {log_dir}")
        return results

    def plot_results(self, results, log_dir):
        """Create publication-quality plots"""
        # Shannon Entropy
        plt.figure(figsize=(10, 6))
        for state in [0, 1]:
            data = [r for r in results if r['logical_state'] == state]
            phis = [r['phi'] for r in data]
            entropies = [r['shannon_entropy'] for r in data]
            plt.plot(phis, entropies, marker='o', label=f'Logical {state}', linewidth=2, markersize=6)
        plt.xlabel('Parameter φ', fontsize=12)
        plt.ylabel('Shannon Entropy', fontsize=12)
        plt.title('Bulk Reconstruction: Entropy vs Parameter', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'shannon_entropy.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Renyi Entropies
        plt.figure(figsize=(10, 6))
        for state in [0, 1]:
            data = [r for r in results if r['logical_state'] == state]
            phis = [r['phi'] for r in data]
            renyis = [r['renyi_entropies']['S_2'] for r in data]
            plt.plot(phis, renyis, marker='s', label=f'Logical {state}', linewidth=2, markersize=6)
        plt.xlabel('Parameter φ', fontsize=12)
        plt.ylabel('Renyi Entropy (α=2)', fontsize=12)
        plt.title('Bulk Reconstruction: Renyi Entropies vs Parameter', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'renyi_entropy.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Fisher Information
        plt.figure(figsize=(10, 6))
        for state in [0, 1]:
            data = [r for r in results if r['logical_state'] == state]
            phis = [r['phi'] for r in data]
            fishers = [r['fisher_info'] for r in data]
            plt.plot(phis, fishers, marker='^', label=f'Logical {state}', linewidth=2, markersize=6)
        plt.xlabel('Parameter φ', fontsize=12)
        plt.ylabel('Fisher Information', fontsize=12)
        plt.title('Bulk Reconstruction: Fisher Information vs Parameter', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'fisher_info.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Combined plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Shannon entropy
        for state in [0, 1]:
            data = [r for r in results if r['logical_state'] == state]
            phis = [r['phi'] for r in data]
            entropies = [r['shannon_entropy'] for r in data]
            axes[0,0].plot(phis, entropies, marker='o', label=f'Logical {state}', linewidth=2)
        axes[0,0].set_xlabel('Parameter φ')
        axes[0,0].set_ylabel('Shannon Entropy')
        axes[0,0].set_title('Shannon Entropy')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Renyi entropy
        for state in [0, 1]:
            data = [r for r in results if r['logical_state'] == state]
            phis = [r['phi'] for r in data]
            renyis = [r['renyi_entropies']['S_2'] for r in data]
            axes[0,1].plot(phis, renyis, marker='s', label=f'Logical {state}', linewidth=2)
        axes[0,1].set_xlabel('Parameter φ')
        axes[0,1].set_ylabel('Renyi Entropy (α=2)')
        axes[0,1].set_title('Renyi Entropy')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Fisher information
        for state in [0, 1]:
            data = [r for r in results if r['logical_state'] == state]
            phis = [r['phi'] for r in data]
            fishers = [r['fisher_info'] for r in data]
            axes[1,0].plot(phis, fishers, marker='^', label=f'Logical {state}', linewidth=2)
        axes[1,0].set_xlabel('Parameter φ')
        axes[1,0].set_ylabel('Fisher Information')
        axes[1,0].set_title('Fisher Information')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Von Neumann entropy
        for state in [0, 1]:
            data = [r for r in results if r['logical_state'] == state]
            phis = [r['phi'] for r in data]
            vn_entropies = [r['von_neumann_entropy'] for r in data]
            axes[1,1].plot(phis, vn_entropies, marker='d', label=f'Logical {state}', linewidth=2)
        axes[1,1].set_xlabel('Parameter φ')
        axes[1,1].set_ylabel('Von Neumann Entropy')
        axes[1,1].set_title('Von Neumann Entropy')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.suptitle('Bulk Reconstruction: Complete Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'bulk_reconstruction_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

def run_batched_bulk_reconstruction_qiskit(backend=None, shots=2048):
    experiment = BatchedBulkReconstructionQiskit(backend=backend, shots=shots)
    results = experiment.run_batched()
    print("Batched Bulk Reconstruction Qiskit experiment completed.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the batched Qiskit bulk reconstruction experiment")
    parser.add_argument('--device', type=str, default='simulator', help='Device to use: "simulator" or IBMQ backend name')
    parser.add_argument('--shots', type=int, default=2048, help='Number of shots per circuit')
    args = parser.parse_args()

    if args.device.lower() == 'simulator':
        backend = FakeJakartaV2()
        print("Using FakeJakartaV2 simulator")
    else:
        service = QiskitRuntimeService()
        backend = service.backend(args.device)
        print(f"Using IBM Quantum backend: {backend.name}")
    
    run_batched_bulk_reconstruction_qiskit(backend=backend, shots=args.shots) 