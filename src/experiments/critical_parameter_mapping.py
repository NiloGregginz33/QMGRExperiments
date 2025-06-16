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

class CriticalParameterMapper:
    def __init__(self, n_qubits=6, shots=1024):
        self.device = LocalSimulator()
        self.n_qubits = n_qubits
        self.shots = shots
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = f"experiment_logs/critical_mapping_{self.timestamp}"
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

    def run_critical_point_experiment(self, n_points=20):
        """Experiment 1: Fine-grained exploration around critical points"""
        print("\nRunning Critical Point Experiment...")
        results = {
            "phases": [],
            "strengths": [],
            "entropies": [],
            "curvatures": [],
            "distances": [],
            "mi_matrices": []
        }
        
        # Focus on regions where we saw interesting behavior
        phases = np.linspace(0, 0.5, n_points)  # Around φ = 0
        strengths = np.linspace(0.5, 1.0, n_points)  # Around g = 0.70
        
        for phi in phases:
            for strength in strengths:
                circ = Circuit()
                circ.h(0)
                
                # Create entanglement structure with phase
                for i in range(0, self.n_qubits-1, 2):
                    circ.cnot(i, i+1)
                    circ.rz(i, phi)
                
                # Apply coupling gates with varying strength
                for i in range(0, self.n_qubits-1):
                    circ.cz(i, (i+2) % self.n_qubits)
                    circ.rx(i, strength)
                
                # Add non-local interactions
                for i in range(0, self.n_qubits-3, 2):
                    circ.cz(i, (i+3) % self.n_qubits)
                
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
                
                avg_curvature = np.mean(curvatures)
                avg_distance = np.mean(dist[dist > 0])
                
                results["phases"].append(phi)
                results["strengths"].append(strength)
                results["entropies"].append(entropy)
                results["curvatures"].append(avg_curvature)
                results["distances"].append(avg_distance)
                results["mi_matrices"].append(mi_matrix.tolist())
                
                print(f"φ = {phi:.2f}, g = {strength:.2f}, S = {entropy:.4f}, K = {avg_curvature:.4f}, d = {avg_distance:.4f}")
        
        # Save results
        with open(f"{self.exp_dir}/critical_point_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Plot results
        self.plot_critical_point_results(results)
        
        return results

    def run_phase_transition_experiment(self, n_points=30):
        """Experiment 2: Study phase transitions in the parameter space"""
        print("\nRunning Phase Transition Experiment...")
        results = {
            "parameters": [],
            "entropies": [],
            "curvatures": [],
            "distances": [],
            "mi_matrices": []
        }
        
        # Create a path through parameter space that crosses potential phase boundaries
        t = np.linspace(0, 1, n_points)
        phases = 0.5 * np.sin(2*np.pi*t)  # Oscillating phase
        strengths = 0.5 + 0.5 * np.sin(4*np.pi*t)  # Oscillating strength
        
        for i in range(n_points):
            phi = phases[i]
            strength = strengths[i]
            
            circ = Circuit()
            circ.h(0)
            
            # Create entanglement structure
            for i in range(0, self.n_qubits-1, 2):
                circ.cnot(i, i+1)
                circ.rz(i, phi)
            
            # Apply coupling gates
            for i in range(0, self.n_qubits-1):
                circ.cz(i, (i+2) % self.n_qubits)
                circ.rx(i, strength)
            
            # Add non-local interactions
            for i in range(0, self.n_qubits-3, 2):
                circ.cz(i, (i+3) % self.n_qubits)
            
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
            
            avg_curvature = np.mean(curvatures)
            avg_distance = np.mean(dist[dist > 0])
            
            results["parameters"].append({"phase": phi, "strength": strength})
            results["entropies"].append(entropy)
            results["curvatures"].append(avg_curvature)
            results["distances"].append(avg_distance)
            results["mi_matrices"].append(mi_matrix.tolist())
            
            print(f"t = {i/n_points:.2f}, φ = {phi:.2f}, g = {strength:.2f}, S = {entropy:.4f}, K = {avg_curvature:.4f}, d = {avg_distance:.4f}")
        
        # Save results
        with open(f"{self.exp_dir}/phase_transition_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Plot results
        self.plot_phase_transition_results(results)
        
        return results

    def plot_critical_point_results(self, results):
        # Create 3D plots
        fig = plt.figure(figsize=(15, 5))
        
        # Entropy surface
        ax1 = fig.add_subplot(131, projection='3d')
        phases = np.array(results["phases"]).reshape(-1, int(np.sqrt(len(results["phases"]))))
        strengths = np.array(results["strengths"]).reshape(-1, int(np.sqrt(len(results["strengths"]))))
        entropies = np.array(results["entropies"]).reshape(phases.shape)
        ax1.plot_surface(phases, strengths, entropies, cmap='viridis')
        ax1.set_xlabel('Phase (φ)')
        ax1.set_ylabel('Coupling (g)')
        ax1.set_zlabel('Entropy')
        ax1.set_title('Entropy Surface')
        
        # Curvature surface
        ax2 = fig.add_subplot(132, projection='3d')
        curvatures = np.array(results["curvatures"]).reshape(phases.shape)
        ax2.plot_surface(phases, strengths, curvatures, cmap='plasma')
        ax2.set_xlabel('Phase (φ)')
        ax2.set_ylabel('Coupling (g)')
        ax2.set_zlabel('Curvature')
        ax2.set_title('Curvature Surface')
        
        # Distance surface
        ax3 = fig.add_subplot(133, projection='3d')
        distances = np.array(results["distances"]).reshape(phases.shape)
        ax3.plot_surface(phases, strengths, distances, cmap='magma')
        ax3.set_xlabel('Phase (φ)')
        ax3.set_ylabel('Coupling (g)')
        ax3.set_zlabel('Distance')
        ax3.set_title('Distance Surface')
        
        plt.tight_layout()
        plt.savefig(f"{self.exp_dir}/critical_point_surfaces.png")
        plt.close()

    def plot_phase_transition_results(self, results):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot entropy evolution
        ax1.plot(range(len(results["entropies"])), results["entropies"], 'b-')
        ax1.set_xlabel('Parameter Path')
        ax1.set_ylabel('Entropy')
        ax1.set_title('Entropy Evolution')
        ax1.grid(True)
        
        # Plot curvature evolution
        ax2.plot(range(len(results["curvatures"])), results["curvatures"], 'r-')
        ax2.set_xlabel('Parameter Path')
        ax2.set_ylabel('Curvature')
        ax2.set_title('Curvature Evolution')
        ax2.grid(True)
        
        # Plot distance evolution
        ax3.plot(range(len(results["distances"])), results["distances"], 'g-')
        ax3.set_xlabel('Parameter Path')
        ax3.set_ylabel('Distance')
        ax3.set_title('Distance Evolution')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.exp_dir}/phase_transition_evolution.png")
        plt.close()

    def analyze_theoretical_implications(self, results):
        """Analyze results in the context of holographic and string theory"""
        analysis = {
            "critical_points": {
                "holographic": [],
                "string_theory": []
            },
            "phase_transitions": {
                "holographic": [],
                "string_theory": []
            }
        }
        
        # Analyze critical point results
        critical_results = results["critical_points"]
        entropy_corr = pearsonr(critical_results["phases"], 
                              critical_results["entropies"])[0]
        strength_corr = pearsonr(critical_results["strengths"],
                               critical_results["entropies"])[0]
        
        if abs(entropy_corr) > 0.7:
            analysis["critical_points"]["holographic"].append(
                "Strong phase-entropy correlation near critical points suggests emergent time-like direction"
            )
        if abs(strength_corr) > 0.7:
            analysis["critical_points"]["string_theory"].append(
                "Coupling strength affects entropy scaling near critical points, consistent with string coupling"
            )
        
        # Analyze phase transition results
        transition_results = results["phase_transitions"]
        entropy_trend = np.polyfit(range(len(transition_results["entropies"])), 
                                 transition_results["entropies"], 1)[0]
        curvature_trend = np.polyfit(range(len(transition_results["curvatures"])),
                                   transition_results["curvatures"], 1)[0]
        
        if entropy_trend > 0:
            analysis["phase_transitions"]["holographic"].append(
                "Increasing entropy during phase transitions suggests emergent spacetime"
            )
        if curvature_trend != 0:
            analysis["phase_transitions"]["string_theory"].append(
                "Non-zero curvature trend during phase transitions indicates potential connection to string compactification"
            )
        
        # Save analysis
        with open(f"{self.exp_dir}/theoretical_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
        
        return analysis

def run_all_experiments():
    mapper = CriticalParameterMapper()
    
    # Run all experiments
    critical_point_results = mapper.run_critical_point_experiment()
    phase_transition_results = mapper.run_phase_transition_experiment()
    
    # Combine results
    all_results = {
        "critical_points": critical_point_results,
        "phase_transitions": phase_transition_results
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