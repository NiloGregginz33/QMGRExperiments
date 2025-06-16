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

class UnifiedGeometryMapper:
    def __init__(self, n_qubits=6, shots=1024):
        self.device = LocalSimulator()
        self.n_qubits = n_qubits
        self.shots = shots
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = f"experiment_logs/unified_mapping_{self.timestamp}"
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

    def run_unified_mapping_experiment(self, n_points=10):
        """Experiment 1: Unified exploration of parameter space"""
        print("\nRunning Unified Mapping Experiment...")
        results = {
            "phases": [],
            "strengths": [],
            "patterns": [],
            "entropies": [],
            "curvatures": [],
            "distances": [],
            "mi_matrices": []
        }
        
        # Define parameter ranges
        phases = np.linspace(0, 2*np.pi, n_points)
        strengths = np.linspace(0, 2*np.pi, n_points)
        
        # Define entanglement patterns
        patterns = [
            # Pattern 1: Nearest neighbor
            lambda circ: [circ.cnot(i, i+1) for i in range(self.n_qubits-1)],
            
            # Pattern 2: Next-to-nearest neighbor
            lambda circ: [circ.cnot(i, i+2) for i in range(self.n_qubits-2)],
            
            # Pattern 3: Star pattern
            lambda circ: [circ.cnot(0, i) for i in range(1, self.n_qubits)],
            
            # Pattern 4: Ring pattern
            lambda circ: [circ.cnot(i, (i+1)%self.n_qubits) for i in range(self.n_qubits)]
        ]
        
        for phi in phases:
            for strength in strengths:
                for pattern_idx, pattern in enumerate(patterns):
                    circ = Circuit()
                    circ.h(0)
                    
                    # Apply entanglement pattern
                    pattern(circ)
                    
                    # Apply phase rotations
                    for i in range(self.n_qubits):
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
                    
                    results["phases"].append(phi)
                    results["strengths"].append(strength)
                    results["patterns"].append(pattern_idx)
                    results["entropies"].append(entropy)
                    results["curvatures"].append(avg_curvature)
                    results["distances"].append(avg_distance)
                    results["mi_matrices"].append(mi_matrix.tolist())
                    
                    print(f"φ = {phi:.2f}, g = {strength:.2f}, P = {pattern_idx}, S = {entropy:.4f}, K = {avg_curvature:.4f}, d = {avg_distance:.4f}")
        
        # Save results
        with open(f"{self.exp_dir}/unified_mapping_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Plot results
        self.plot_unified_mapping_results(results)
        
        return results

    def run_emergent_geometry_experiment(self, n_steps=20):
        """Experiment 2: Study emergent geometry evolution"""
        print("\nRunning Emergent Geometry Experiment...")
        results = {
            "steps": [],
            "entropies": [],
            "curvatures": [],
            "distances": [],
            "mi_matrices": []
        }
        
        # Start with a simple pattern
        current_edges = [(i, i+1) for i in range(self.n_qubits-1)]
        phase = 0
        strength = 0
        
        for step in range(n_steps):
            circ = Circuit()
            circ.h(0)
            
            # Apply current entanglement pattern
            for i, j in current_edges:
                circ.cnot(i, j)
                circ.rz(i, phase)
            
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
            
            results["steps"].append(step)
            results["entropies"].append(entropy)
            results["curvatures"].append(avg_curvature)
            results["distances"].append(avg_distance)
            results["mi_matrices"].append(mi_matrix.tolist())
            
            print(f"Step {step}: S = {entropy:.4f}, K = {avg_curvature:.4f}, d = {avg_distance:.4f}")
            
            # Evolve parameters and pattern
            if step < n_steps - 1:
                # Update phase and strength
                phase = 2*np.pi*step/n_steps
                strength = np.pi*step/n_steps
                
                # Add new connections based on high mutual information
                new_edges = []
                for i in range(self.n_qubits):
                    for j in range(i+1, self.n_qubits):
                        if mi_matrix[i,j] > 0.5 and (i,j) not in current_edges:
                            new_edges.append((i,j))
                current_edges.extend(new_edges)
        
        # Save results
        with open(f"{self.exp_dir}/emergent_geometry_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Plot results
        self.plot_emergent_geometry_results(results)
        
        return results

    def plot_unified_mapping_results(self, results):
        # Create 3D plots for each pattern
        patterns = np.unique(results["patterns"])
        for pattern in patterns:
            fig = plt.figure(figsize=(15, 5))
            
            # Filter results for this pattern
            mask = np.array(results["patterns"]) == pattern
            phases = np.array(results["phases"])[mask].reshape(-1, int(np.sqrt(sum(mask))))
            strengths = np.array(results["strengths"])[mask].reshape(-1, int(np.sqrt(sum(mask))))
            entropies = np.array(results["entropies"])[mask].reshape(phases.shape)
            curvatures = np.array(results["curvatures"])[mask].reshape(phases.shape)
            distances = np.array(results["distances"])[mask].reshape(phases.shape)
            
            # Entropy surface
            ax1 = fig.add_subplot(131, projection='3d')
            ax1.plot_surface(phases, strengths, entropies, cmap='viridis')
            ax1.set_xlabel('Phase (φ)')
            ax1.set_ylabel('Coupling (g)')
            ax1.set_zlabel('Entropy')
            ax1.set_title(f'Entropy Surface (Pattern {pattern})')
            
            # Curvature surface
            ax2 = fig.add_subplot(132, projection='3d')
            ax2.plot_surface(phases, strengths, curvatures, cmap='plasma')
            ax2.set_xlabel('Phase (φ)')
            ax2.set_ylabel('Coupling (g)')
            ax2.set_zlabel('Curvature')
            ax2.set_title(f'Curvature Surface (Pattern {pattern})')
            
            # Distance surface
            ax3 = fig.add_subplot(133, projection='3d')
            ax3.plot_surface(phases, strengths, distances, cmap='magma')
            ax3.set_xlabel('Phase (φ)')
            ax3.set_ylabel('Coupling (g)')
            ax3.set_zlabel('Distance')
            ax3.set_title(f'Distance Surface (Pattern {pattern})')
            
            plt.tight_layout()
            plt.savefig(f"{self.exp_dir}/unified_mapping_pattern_{pattern}.png")
            plt.close()

    def plot_emergent_geometry_results(self, results):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot entropy evolution
        ax1.plot(results["steps"], results["entropies"], 'b-')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Entropy')
        ax1.set_title('Entropy Evolution')
        ax1.grid(True)
        
        # Plot curvature evolution
        ax2.plot(results["steps"], results["curvatures"], 'r-')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Curvature')
        ax2.set_title('Curvature Evolution')
        ax2.grid(True)
        
        # Plot distance evolution
        ax3.plot(results["steps"], results["distances"], 'g-')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Distance')
        ax3.set_title('Distance Evolution')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.exp_dir}/emergent_geometry_evolution.png")
        plt.close()

    def analyze_theoretical_implications(self, results):
        """Analyze results in the context of holographic and string theory"""
        analysis = {
            "unified_mapping": {
                "holographic": [],
                "string_theory": []
            },
            "emergent_geometry": {
                "holographic": [],
                "string_theory": []
            }
        }
        
        # Analyze unified mapping results
        unified_results = results["unified_mapping"]
        for pattern in np.unique(unified_results["patterns"]):
            mask = np.array(unified_results["patterns"]) == pattern
            entropy_corr = pearsonr(unified_results["phases"][mask], 
                                  unified_results["entropies"][mask])[0]
            strength_corr = pearsonr(unified_results["strengths"][mask],
                                   unified_results["entropies"][mask])[0]
            
            if abs(entropy_corr) > 0.7:
                analysis["unified_mapping"]["holographic"].append(
                    f"Pattern {pattern} shows strong phase-entropy correlation, suggesting emergent time-like direction"
                )
            if abs(strength_corr) > 0.7:
                analysis["unified_mapping"]["string_theory"].append(
                    f"Pattern {pattern} shows coupling-dependent entropy scaling, consistent with string coupling"
                )
        
        # Analyze emergent geometry results
        emergent_results = results["emergent_geometry"]
        entropy_trend = np.polyfit(emergent_results["steps"], 
                                 emergent_results["entropies"], 1)[0]
        curvature_trend = np.polyfit(emergent_results["steps"],
                                   emergent_results["curvatures"], 1)[0]
        
        if entropy_trend > 0:
            analysis["emergent_geometry"]["holographic"].append(
                "Increasing entropy during evolution suggests emergent spacetime"
            )
        if curvature_trend != 0:
            analysis["emergent_geometry"]["string_theory"].append(
                "Non-zero curvature trend indicates potential connection to string compactification"
            )
        
        # Save analysis
        with open(f"{self.exp_dir}/theoretical_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
        
        return analysis

def run_all_experiments():
    mapper = UnifiedGeometryMapper()
    
    # Run all experiments
    unified_results = mapper.run_unified_mapping_experiment()
    emergent_results = mapper.run_emergent_geometry_experiment()
    
    # Combine results
    all_results = {
        "unified_mapping": unified_results,
        "emergent_geometry": emergent_results
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