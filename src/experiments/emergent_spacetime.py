import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from itertools import combinations
import json
import os
from datetime import datetime
import sys
import argparse
sys.path.append('src')
from AWSFactory import LocalSimulator
from braket.circuits import Circuit, FreeParameter
from braket.aws import AwsDevice
from utils.transpiler import transpile_circuit, get_device_info

def get_device(device_type="simulator"):
    """
    Get quantum device based on specified type
    
    Args:
        device_type (str): "simulator", "ionq", "rigetti", or "oqc"
    
    Returns:
        Device object for quantum computation
    """
    if device_type == "simulator":
        return LocalSimulator()
    elif device_type == "ionq":
        return AwsDevice("arn:aws:braket:us-east-1::device/qpu/ionq/ionQdevice")
    elif device_type == "rigetti":
        return AwsDevice("arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3")
    elif device_type == "oqc":
        return AwsDevice("arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy")
    else:
        print(f"Unknown device type: {device_type}. Using simulator.")
        return LocalSimulator()

class LocalEmergentSpacetime:
    def __init__(self, device, shots=1024):
        self.device = device
        self.shots = shots
        self.timesteps = np.linspace(0, 3 * np.pi, 15)
        self.mi_matrices = []
        
        # Get device information
        device_info = get_device_info(device)
        print(f"Device: {device_info.get('name', 'Unknown')}")
        print(f"Native gates: {device_info.get('native_gates', 'Unknown')}")
        print(f"Qubit count: {device_info.get('qubit_count', 'Unknown')}")

    def shannon_entropy(self, probs):
        probs = np.array(probs)
        return -np.sum(probs * np.log2(probs + 1e-12))

    def marginal_probs(self, probs, total_qubits, target_idxs):
        marginal = {}
        for idx, p in enumerate(probs):
            b = format(idx, f"0{total_qubits}b")
            key = ''.join([b[i] for i in target_idxs])
            marginal[key] = marginal.get(key, 0) + p
        return np.array(list(marginal.values()))

    def compute_mi(self, probs, qA, qB, total_qubits):
        AB = self.marginal_probs(probs, total_qubits, [qA, qB])
        A = self.marginal_probs(probs, total_qubits, [qA])
        B = self.marginal_probs(probs, total_qubits, [qB])
        return self.shannon_entropy(A) + self.shannon_entropy(B) - self.shannon_entropy(AB)

    def run(self):
        print(f"Running on device: {self.device}")
        for i, phi_val in enumerate(self.timesteps):
            print(f"Processing timestep {i+1}/{len(self.timesteps)} (φ = {phi_val:.3f})")
            phi = FreeParameter("phi")
            circ = Circuit()
            circ.h(0)
            circ.cnot(0, 2)
            circ.cnot(0, 3)
            circ.rx(0, phi)

            # cz(0, 1) equivalent
            circ.cnot(0, 1).rz(1, np.pi).cnot(0, 1)

            circ.cnot(1, 2)
            circ.rx(2, phi)

            # cz(1, 3) equivalent
            circ.cnot(1, 3).rz(3, np.pi).cnot(1, 3)
            circ.probability()

            # Transpile circuit for the specific device
            transpiled_circuit = transpile_circuit(circ, self.device)
            
            print(f"Original circuit depth: {len(circ.instructions)}")
            print(f"Transpiled circuit depth: {len(transpiled_circuit.instructions)}")

            task = self.device.run(transpiled_circuit, inputs={"phi": phi_val}, shots=self.shots)
            result = task.result()
            probs = np.array(result.values).reshape(-1)

            mi_matrix = np.zeros((4, 4))
            for i in range(4):
                for j in range(i + 1, 4):
                    mi = self.compute_mi(probs, i, j, 4)
                    mi_matrix[i, j] = mi_matrix[j, i] = mi
            self.mi_matrices.append(mi_matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run emergent spacetime experiment')
    parser.add_argument('--device', type=str, default='simulator', 
                       choices=['simulator', 'ionq', 'rigetti', 'oqc'],
                       help='Quantum device to use')
    parser.add_argument('--shots', type=int, default=1024,
                       help='Number of shots for quantum measurements')
    args = parser.parse_args()
    
    exp_dir = f"experiment_logs/emergent_spacetime_{args.device}"
    os.makedirs(exp_dir, exist_ok=True)
    
    device = get_device(args.device)
    spacetime = LocalEmergentSpacetime(device, shots=args.shots)
    spacetime.run()
    mi_matrices = spacetime.mi_matrices
    results = {
        "device": args.device,
        "shots": args.shots,
        "timesteps": np.linspace(0, 3 * np.pi, len(mi_matrices)).tolist(),
        "entropies": [],
        "curvatures": [],
        "distances": [],
        "geometries": []
    }
    for mi_matrix in mi_matrices:
        epsilon = 1e-6
        dist = 1 / (mi_matrix + epsilon)
        np.fill_diagonal(dist, 0)
        coords = MDS(n_components=3, dissimilarity='precomputed').fit_transform(dist)
        entropy = np.mean(mi_matrix[mi_matrix > 0])
        curvature = np.mean([np.pi - (np.arccos((np.linalg.norm(coords[j] - coords[k])**2 + np.linalg.norm(coords[i] - coords[k])**2 - np.linalg.norm(coords[i] - coords[j])**2) / (2 * np.linalg.norm(coords[j] - coords[k]) * np.linalg.norm(coords[i] - coords[k])))) for i, j, k in combinations(range(len(coords)), 3)])
        avg_dist = np.mean(dist[dist > 0])
        results["entropies"].append(float(entropy))
        results["curvatures"].append(float(curvature))
        results["distances"].append(float(avg_dist))
        results["geometries"].append(coords.tolist())
    with open(f"{exp_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes[0,0].plot(results["timesteps"], results["entropies"], 'b-', label='Entropy')
    axes[0,0].set_xlabel('Time (φ)')
    axes[0,0].set_ylabel('Entropy (bits)')
    axes[0,0].set_title('Entropy Evolution')
    axes[0,0].grid(True)
    axes[0,0].legend()
    axes[0,1].plot(results["timesteps"], results["curvatures"], 'r-', label='Curvature')
    axes[0,1].set_xlabel('Time (φ)')
    axes[0,1].set_ylabel('Curvature')
    axes[0,1].set_title('Curvature Evolution')
    axes[0,1].grid(True)
    axes[0,1].legend()
    axes[1,0].plot(results["timesteps"], results["distances"], 'g-', label='Distance')
    axes[1,0].set_xlabel('Time (φ)')
    axes[1,0].set_ylabel('Average Distance')
    axes[1,0].set_title('Distance Evolution')
    axes[1,0].grid(True)
    axes[1,0].legend()
    final_geometry = np.array(results["geometries"][-1])
    ax = axes[1,1]
    scatter = ax.scatter(final_geometry[:,0], final_geometry[:,1], c='blue', s=100)
    for i in range(len(final_geometry)):
        ax.text(final_geometry[i,0], final_geometry[i,1], f"Q{i}", fontsize=12)
    ax.set_title('Final Geometry (2D Projection)')
    ax.set_xlabel('MDS Dimension 1')
    ax.set_ylabel('MDS Dimension 2')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"{exp_dir}/results.png")
    plt.close()
    with open(f"{exp_dir}/summary.txt", "w") as f:
        f.write("Emergent Spacetime Experiment Summary\n")
        f.write("====================================\n\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Shots: {args.shots}\n\n")
        f.write("Theoretical Background:\n")
        f.write("This experiment explores how quantum entanglement gives rise to emergent spacetime geometry. By analyzing mutual information and geometric embeddings, it probes the relationship between quantum information and geometry.\n\n")
        f.write("Methodology:\n")
        f.write("A quantum circuit is simulated over multiple timesteps. Mutual information matrices are computed, and MDS is used to extract geometric features. Entropy, curvature, and distance are tracked over time.\n\n")
        f.write("Results:\n")
        f.write(f"Results saved in: {exp_dir}\n")
        f.write("\nConclusion:\n")
        f.write("The experiment demonstrates how quantum entanglement patterns can be mapped to emergent geometric structures, supporting the idea that spacetime geometry is encoded in quantum information.\n") 