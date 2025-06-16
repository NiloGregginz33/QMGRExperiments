import numpy as np
import matplotlib.pyplot as plt
from braket.circuits import Circuit, FreeParameter
from braket.devices import LocalSimulator
from sklearn.manifold import MDS
from scipy.stats import pearsonr
import seaborn as sns
from datetime import datetime
import os
import json
from itertools import combinations
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class ParameterGeometryMapper:
    def __init__(self, n_qubits=6, shots=1024):
        self.device = LocalSimulator()
        self.n_qubits = n_qubits
        self.shots = shots
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = f"experiment_logs/parameter_mapping_{self.timestamp}"
        os.makedirs(self.exp_dir, exist_ok=True)
        
    def shannon_entropy(self, probs):
        probs = np.array(probs)
        probs = probs / np.sum(probs)
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
    
    def estimate_curvature(self, coords, triplet):
        i, j, k = triplet
        a = np.linalg.norm(coords[j] - coords[k])
        b = np.linalg.norm(coords[i] - coords[k])
        c = np.linalg.norm(coords[i] - coords[j])
        
        def safe_acos(x):
            return np.arccos(np.clip(x, -1.0, 1.0))
        
        angle_i = safe_acos((b**2 + c**2 - a**2) / (2 * b * c))
        angle_j = safe_acos((a**2 + c**2 - b**2) / (2 * a * c))
        angle_k = safe_acos((a**2 + b**2 - c**2) / (2 * a * b))
        
        return (angle_i + angle_j + angle_k) - np.pi

    def run_phase_sweep_experiment(self, n_points=20):
        """Experiment 1: Sweep through phase parameters to map their effect on geometry"""
        print("\nRunning Phase Sweep Experiment...")
        results = {
            "phases": [],
            "entropies": [],
            "curvatures": [],
            "distances": []
        }
        
        phases = np.linspace(0, 2*np.pi, n_points)
        for phi in phases:
            circ = Circuit()
            circ.h(0)
            
            # Create entanglement structure
            for i in range(0, self.n_qubits-1, 2):
                circ.cnot(i, i+1)
            
            # Apply phase rotations
            for i in range(self.n_qubits):
                circ.rx(i, phi)
            
            # Add non-local couplings
            for i in range(0, self.n_qubits-2, 2):
                circ.cz(i, i+2)
            
            circ.probability()
            
            task = self.device.run(circ, shots=self.shots)
            result = task.result()
            probs = np.array(result.values).reshape(-1)
            
            # Compute MI matrix
            mi_matrix = np.zeros((self.n_qubits, self.n_qubits))
            for i in range(self.n_qubits):
                for j in range(i+1, self.n_qubits):
                    mi = self.compute_mi(probs, i, j, self.n_qubits)
                    mi_matrix[i,j] = mi_matrix[j,i] = mi
            
            # Compute geometry
            dist = np.exp(-mi_matrix)
            np.fill_diagonal(dist, 0)
            coords = MDS(n_components=3, dissimilarity='precomputed').fit_transform(dist)
            
            # Compute metrics
            entropy = self.shannon_entropy(probs)
            curvatures = []
            for triplet in combinations(range(self.n_qubits), 3):
                curv = self.estimate_curvature(coords, triplet)
                curvatures.append(curv)
            
            avg_curvature = np.mean(curvatures)
            avg_distance = np.mean(dist[dist > 0])
            
            results["phases"].append(phi)
            results["entropies"].append(entropy)
            results["curvatures"].append(avg_curvature)
            results["distances"].append(avg_distance)
            
            print(f"φ = {phi:.2f}, S = {entropy:.4f}, K = {avg_curvature:.4f}, d = {avg_distance:.4f}")
        
        # Save results
        with open(f"{self.exp_dir}/phase_sweep_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Plot results
        self.plot_phase_sweep_results(results)
        
        return results

    def run_coupling_strength_experiment(self, n_points=20):
        """Experiment 2: Vary coupling strengths to study their impact on geometry"""
        print("\nRunning Coupling Strength Experiment...")
        results = {
            "strengths": [],
            "entropies": [],
            "curvatures": [],
            "distances": []
        }
        
        strengths = np.linspace(0, 2*np.pi, n_points)
        for strength in strengths:
            circ = Circuit()
            circ.h(0)
            
            # Apply coupling gates with varying strength
            for i in range(0, self.n_qubits-1):
                circ.cnot(i, i+1)
                circ.rz(i, strength)
                circ.cz(i, (i+2) % self.n_qubits)
            
            circ.probability()
            
            task = self.device.run(circ, shots=self.shots)
            result = task.result()
            probs = np.array(result.values).reshape(-1)
            
            # Compute metrics (similar to phase sweep)
            mi_matrix = np.zeros((self.n_qubits, self.n_qubits))
            for i in range(self.n_qubits):
                for j in range(i+1, self.n_qubits):
                    mi = self.compute_mi(probs, i, j, self.n_qubits)
                    mi_matrix[i,j] = mi_matrix[j,i] = mi
            
            dist = np.exp(-mi_matrix)
            np.fill_diagonal(dist, 0)
            coords = MDS(n_components=3, dissimilarity='precomputed').fit_transform(dist)
            
            entropy = self.shannon_entropy(probs)
            curvatures = []
            for triplet in combinations(range(self.n_qubits), 3):
                curv = self.estimate_curvature(coords, triplet)
                curvatures.append(curv)
            
            avg_curvature = np.mean(curvatures)
            avg_distance = np.mean(dist[dist > 0])
            
            results["strengths"].append(strength)
            results["entropies"].append(entropy)
            results["curvatures"].append(avg_curvature)
            results["distances"].append(avg_distance)
            
            print(f"Strength = {strength:.2f}, S = {entropy:.4f}, K = {avg_curvature:.4f}, d = {avg_distance:.4f}")
        
        # Save results
        with open(f"{self.exp_dir}/coupling_strength_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Plot results
        self.plot_coupling_strength_results(results)
        
        return results

    def run_topology_experiment(self):
        """Experiment 3: Study different topological configurations"""
        print("\nRunning Topology Experiment...")
        topologies = {
            "linear": [(i, i+1) for i in range(self.n_qubits-1)],
            "ring": [(i, (i+1)%self.n_qubits) for i in range(self.n_qubits)],
            "star": [(0, i) for i in range(1, self.n_qubits)],
            "complete": [(i, j) for i in range(self.n_qubits) for j in range(i+1, self.n_qubits)]
        }
        
        results = {}
        for name, edges in topologies.items():
            print(f"\nTesting {name} topology...")
            circ = Circuit()
            circ.h(0)
            
            # Apply entangling gates based on topology
            for i, j in edges:
                circ.cnot(i, j)
                circ.cz(i, j)
            
            circ.probability()
            
            task = self.device.run(circ, shots=self.shots)
            result = task.result()
            probs = np.array(result.values).reshape(-1)
            
            # Compute metrics
            mi_matrix = np.zeros((self.n_qubits, self.n_qubits))
            for i in range(self.n_qubits):
                for j in range(i+1, self.n_qubits):
                    mi = self.compute_mi(probs, i, j, self.n_qubits)
                    mi_matrix[i,j] = mi_matrix[j,i] = mi
            
            dist = np.exp(-mi_matrix)
            np.fill_diagonal(dist, 0)
            coords = MDS(n_components=3, dissimilarity='precomputed').fit_transform(dist)
            
            entropy = self.shannon_entropy(probs)
            curvatures = []
            for triplet in combinations(range(self.n_qubits), 3):
                curv = self.estimate_curvature(coords, triplet)
                curvatures.append(curv)
            
            results[name] = {
                "entropy": entropy,
                "avg_curvature": np.mean(curvatures),
                "avg_distance": np.mean(dist[dist > 0]),
                "mi_matrix": mi_matrix.tolist()
            }
            
            print(f"Topology: {name}")
            print(f"Entropy: {entropy:.4f}")
            print(f"Average Curvature: {np.mean(curvatures):.4f}")
            print(f"Average Distance: {np.mean(dist[dist > 0]):.4f}")
        
        # Save results
        with open(f"{self.exp_dir}/topology_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Plot results
        self.plot_topology_results(results)
        
        return results

    def plot_phase_sweep_results(self, results):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot entropy and curvature
        ax1.plot(results["phases"], results["entropies"], 'b-', label='Entropy')
        ax1.set_xlabel('Phase (φ)')
        ax1.set_ylabel('Entropy')
        ax1.set_title('Entropy vs Phase')
        ax1.grid(True)
        
        ax2.plot(results["phases"], results["curvatures"], 'r-', label='Curvature')
        ax2.set_xlabel('Phase (φ)')
        ax2.set_ylabel('Average Curvature')
        ax2.set_title('Curvature vs Phase')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.exp_dir}/phase_sweep_plots.png")
        plt.close()

    def plot_coupling_strength_results(self, results):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot entropy and curvature
        ax1.plot(results["strengths"], results["entropies"], 'b-', label='Entropy')
        ax1.set_xlabel('Coupling Strength')
        ax1.set_ylabel('Entropy')
        ax1.set_title('Entropy vs Coupling Strength')
        ax1.grid(True)
        
        ax2.plot(results["strengths"], results["curvatures"], 'r-', label='Curvature')
        ax2.set_xlabel('Coupling Strength')
        ax2.set_ylabel('Average Curvature')
        ax2.set_title('Curvature vs Coupling Strength')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.exp_dir}/coupling_strength_plots.png")
        plt.close()

    def plot_topology_results(self, results):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot entropy and curvature for each topology
        topologies = list(results.keys())
        entropies = [results[t]["entropy"] for t in topologies]
        curvatures = [results[t]["avg_curvature"] for t in topologies]
        
        ax1.bar(topologies, entropies)
        ax1.set_xlabel('Topology')
        ax1.set_ylabel('Entropy')
        ax1.set_title('Entropy by Topology')
        ax1.grid(True)
        
        ax2.bar(topologies, curvatures)
        ax2.set_xlabel('Topology')
        ax2.set_ylabel('Average Curvature')
        ax2.set_title('Curvature by Topology')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.exp_dir}/topology_plots.png")
        plt.close()

    def analyze_theoretical_implications(self, results):
        """Analyze results in the context of holographic and string theory"""
        analysis = {
            "phase_sweep": {
                "holographic": [],
                "string_theory": []
            },
            "coupling_strength": {
                "holographic": [],
                "string_theory": []
            },
            "topology": {
                "holographic": [],
                "string_theory": []
            }
        }
        
        # Analyze phase sweep results
        phase_entropy_corr = pearsonr(results["phase_sweep"]["phases"], 
                                    results["phase_sweep"]["entropies"])[0]
        phase_curv_corr = pearsonr(results["phase_sweep"]["phases"], 
                                  results["phase_sweep"]["curvatures"])[0]
        
        if abs(phase_entropy_corr) > 0.7:
            analysis["phase_sweep"]["holographic"].append(
                "Strong correlation between phase and entropy suggests emergent time-like direction"
            )
        if abs(phase_curv_corr) > 0.7:
            analysis["phase_sweep"]["string_theory"].append(
                "Phase-dependent curvature indicates potential connection to string compactification"
            )
        
        # Analyze coupling strength results
        strength_entropy_corr = pearsonr(results["coupling_strength"]["strengths"],
                                       results["coupling_strength"]["entropies"])[0]
        strength_curv_corr = pearsonr(results["coupling_strength"]["strengths"],
                                    results["coupling_strength"]["curvatures"])[0]
        
        if abs(strength_entropy_corr) > 0.7:
            analysis["coupling_strength"]["holographic"].append(
                "Coupling strength affects entropy scaling, consistent with holographic principle"
            )
        if abs(strength_curv_corr) > 0.7:
            analysis["coupling_strength"]["string_theory"].append(
                "Coupling-dependent curvature suggests connection to string coupling constant"
            )
        
        # Analyze topology results
        topology_results = results["topology"]
        if topology_results["ring"]["avg_curvature"] > topology_results["linear"]["avg_curvature"]:
            analysis["topology"]["holographic"].append(
                "Ring topology shows higher curvature, consistent with AdS/CFT correspondence"
            )
        if topology_results["complete"]["entropy"] > topology_results["star"]["entropy"]:
            analysis["topology"]["string_theory"].append(
                "Complete graph shows maximal entanglement, suggesting connection to string field theory"
            )
        
        # Save analysis
        with open(f"{self.exp_dir}/theoretical_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
        
        return analysis

def run_all_experiments():
    mapper = ParameterGeometryMapper()
    
    # Run all experiments
    phase_results = mapper.run_phase_sweep_experiment()
    coupling_results = mapper.run_coupling_strength_experiment()
    topology_results = mapper.run_topology_experiment()
    
    # Combine results
    all_results = {
        "phase_sweep": phase_results,
        "coupling_strength": coupling_results,
        "topology": topology_results
    }
    
    # Analyze theoretical implications
    analysis = mapper.analyze_theoretical_implications(all_results)
    
    print("\nTheoretical Analysis Summary:")
    for experiment, implications in analysis.items():
        print(f"\n{experiment.upper()} EXPERIMENT:")
        print("Holographic Implications:")
        for imp in implications["holographic"]:
            print(f"- {imp}")
        print("String Theory Implications:")
        for imp in implications["string_theory"]:
            print(f"- {imp}")
    
    return all_results, analysis

if __name__ == "__main__":
    results, analysis = run_all_experiments() 