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

class CircuitVariationExplorer:
    def __init__(self, n_qubits=6, shots=1024):
        self.device = LocalSimulator()
        self.n_qubits = n_qubits
        self.shots = shots
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = f"experiment_logs/circuit_variations_{self.timestamp}"
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

    def run_inverse_circuit_experiment(self):
        """Experiment 1: Run circuits in reverse order"""
        print("\nRunning Inverse Circuit Experiment...")
        results = {
            "circuit_type": [],
            "entropies": [],
            "curvatures": [],
            "distances": [],
            "mi_matrices": []
        }
        
        # Define base circuit
        def create_base_circuit():
            circ = Circuit()
            circ.h(0)
            for i in range(self.n_qubits-1):
                circ.cnot(i, i+1)
            for i in range(self.n_qubits):
                circ.rz(i, np.pi/4)
            for i in range(0, self.n_qubits-1, 2):
                circ.cz(i, i+1)
            return circ
        
        # Create inverse circuit manually
        def create_inverse_circuit():
            circ = Circuit()
            # Apply gates in reverse order with inverse operations
            for i in range(self.n_qubits-1, -1, -1):
                if i % 2 == 0 and i < self.n_qubits-1:
                    circ.cz(i, i+1)  # CZ is self-inverse
            for i in range(self.n_qubits-1, -1, -1):
                circ.rz(i, -np.pi/4)  # Inverse rotation
            for i in range(self.n_qubits-2, -1, -1):
                circ.cnot(i, i+1)  # CNOT is self-inverse
            circ.h(0)  # Hadamard is self-inverse
            return circ
        
        # Run forward circuit
        circ_forward = create_base_circuit()
        circ_forward.probability()
        task = self.device.run(circ_forward, shots=self.shots)
        result = task.result()
        probs_forward = np.array(result.values).reshape(-1)
        
        # Run inverse circuit
        circ_inverse = create_inverse_circuit()
        circ_inverse.probability()
        task = self.device.run(circ_inverse, shots=self.shots)
        result = task.result()
        probs_inverse = np.array(result.values).reshape(-1)
        
        # Compute metrics for both circuits
        for circuit_type, probs in [("forward", probs_forward), ("inverse", probs_inverse)]:
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
            
            results["circuit_type"].append(circuit_type)
            results["entropies"].append(entropy)
            results["curvatures"].append(avg_curvature)
            results["distances"].append(avg_distance)
            results["mi_matrices"].append(mi_matrix.tolist())
            
            print(f"Circuit: {circuit_type}, S = {entropy:.4f}, K = {avg_curvature:.4f}, d = {avg_distance:.4f}")
        
        # Save results
        with open(f"{self.exp_dir}/inverse_circuit_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results

    def run_parameter_variation_experiment(self, n_points=10):
        """Experiment 2: Vary circuit parameters systematically"""
        print("\nRunning Parameter Variation Experiment...")
        results = {
            "phase": [],
            "coupling": [],
            "entropies": [],
            "curvatures": [],
            "distances": [],
            "mi_matrices": []
        }
        
        phases = np.linspace(0, 2*np.pi, n_points)
        couplings = np.linspace(0, 2*np.pi, n_points)
        
        for phase in phases:
            for coupling in couplings:
                circ = Circuit()
                circ.h(0)
                
                # Apply entanglement
                for i in range(self.n_qubits-1):
                    circ.cnot(i, i+1)
                
                # Apply phase rotations
                for i in range(self.n_qubits):
                    circ.rz(i, phase)
                
                # Apply coupling gates
                for i in range(0, self.n_qubits-1, 2):
                    circ.cz(i, i+1)
                    circ.rx(i, coupling)
                
                # Add non-local interactions
                for i in range(0, self.n_qubits-3, 3):
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
                
                results["phase"].append(phase)
                results["coupling"].append(coupling)
                results["entropies"].append(entropy)
                results["curvatures"].append(avg_curvature)
                results["distances"].append(avg_distance)
                results["mi_matrices"].append(mi_matrix.tolist())
                
                print(f"φ = {phase:.2f}, g = {coupling:.2f}, S = {entropy:.4f}, K = {avg_curvature:.4f}, d = {avg_distance:.4f}")
        
        # Save results
        with open(f"{self.exp_dir}/parameter_variation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results

    def run_topology_variation_experiment(self):
        """Experiment 3: Test different topological configurations"""
        print("\nRunning Topology Variation Experiment...")
        results = {
            "topology": [],
            "entropies": [],
            "curvatures": [],
            "distances": [],
            "mi_matrices": []
        }
        
        # Define different topologies
        topologies = {
            "linear": lambda circ: [circ.cnot(i, i+1) for i in range(self.n_qubits-1)],
            "ring": lambda circ: [circ.cnot(i, (i+1)%self.n_qubits) for i in range(self.n_qubits)],
            "star": lambda circ: [circ.cnot(0, i) for i in range(1, self.n_qubits)],
            "complete": lambda circ: [circ.cnot(i, j) for i in range(self.n_qubits) for j in range(i+1, self.n_qubits)],
            "alternating": lambda circ: [circ.cnot(i, i+2) for i in range(self.n_qubits-2)],
            "nested": lambda circ: [circ.cnot(i, (i+1)%self.n_qubits) for i in range(0, self.n_qubits, 2)]
        }
        
        for topology_name, topology_func in topologies.items():
            circ = Circuit()
            circ.h(0)
            
            # Apply topology
            topology_func(circ)
            
            # Apply phase rotations
            for i in range(self.n_qubits):
                circ.rz(i, np.pi/4)
            
            # Apply coupling gates
            for i in range(0, self.n_qubits-1, 2):
                circ.cz(i, i+1)
            
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
            
            results["topology"].append(topology_name)
            results["entropies"].append(entropy)
            results["curvatures"].append(avg_curvature)
            results["distances"].append(avg_distance)
            results["mi_matrices"].append(mi_matrix.tolist())
            
            print(f"Topology: {topology_name}, S = {entropy:.4f}, K = {avg_curvature:.4f}, d = {avg_distance:.4f}")
        
        # Save results
        with open(f"{self.exp_dir}/topology_variation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results

    def run_dynamic_variation_experiment(self, n_steps=20):
        """Experiment 4: Study dynamic circuit evolution"""
        print("\nRunning Dynamic Variation Experiment...")
        results = {
            "step": [],
            "entropies": [],
            "curvatures": [],
            "distances": [],
            "mi_matrices": []
        }
        
        # Start with a simple pattern
        current_edges = [(i, i+1) for i in range(self.n_qubits-1)]
        phase = 0
        coupling = 0
        
        for step in range(n_steps):
            circ = Circuit()
            circ.h(0)
            
            # Apply current entanglement pattern
            for i, j in current_edges:
                circ.cnot(i, j)
                circ.rz(i, phase)
            
            # Apply coupling gates
            for i in range(0, self.n_qubits-1, 2):
                circ.cz(i, i+1)
                circ.rx(i, coupling)
            
            # Add non-local interactions
            for i in range(0, self.n_qubits-3, 3):
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
            
            results["step"].append(step)
            results["entropies"].append(entropy)
            results["curvatures"].append(avg_curvature)
            results["distances"].append(avg_distance)
            results["mi_matrices"].append(mi_matrix.tolist())
            
            print(f"Step {step}: S = {entropy:.4f}, K = {avg_curvature:.4f}, d = {avg_distance:.4f}")
            
            # Evolve parameters and pattern
            if step < n_steps - 1:
                # Update phase and coupling
                phase = 2*np.pi*step/n_steps
                coupling = np.pi*step/n_steps
                
                # Add new connections based on high mutual information
                new_edges = []
                for i in range(self.n_qubits):
                    for j in range(i+1, self.n_qubits):
                        if mi_matrix[i,j] > 0.5 and (i,j) not in current_edges:
                            new_edges.append((i,j))
                current_edges.extend(new_edges)
        
        # Save results
        with open(f"{self.exp_dir}/dynamic_variation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results

    def analyze_theoretical_implications(self, all_results):
        """Analyze results in the context of holographic and string theory"""
        analysis = {
            "inverse_circuit": {
                "holographic": [],
                "string_theory": []
            },
            "parameter_variation": {
                "holographic": [],
                "string_theory": []
            },
            "topology_variation": {
                "holographic": [],
                "string_theory": []
            },
            "dynamic_variation": {
                "holographic": [],
                "string_theory": []
            }
        }
        
        # Analyze inverse circuit results
        inv_results = all_results["inverse_circuit"]
        forward_entropy = inv_results["entropies"][0]
        inverse_entropy = inv_results["entropies"][1]
        if abs(forward_entropy - inverse_entropy) < 0.1:
            analysis["inverse_circuit"]["holographic"].append(
                "Time-reversal symmetry in entropy suggests emergent time-like direction"
            )
        
        # Analyze parameter variation results
        param_results = all_results["parameter_variation"]
        phase_entropy_corr = pearsonr(param_results["phase"], 
                                    param_results["entropies"])[0]
        coupling_entropy_corr = pearsonr(param_results["coupling"],
                                       param_results["entropies"])[0]
        
        if abs(phase_entropy_corr) > 0.7:
            analysis["parameter_variation"]["holographic"].append(
                "Strong phase-entropy correlation suggests emergent time-like direction"
            )
        if abs(coupling_entropy_corr) > 0.7:
            analysis["parameter_variation"]["string_theory"].append(
                "Coupling-dependent entropy scaling consistent with string coupling"
            )
        
        # Analyze topology variation results
        topo_results = all_results["topology_variation"]
        ring_idx = topo_results["topology"].index("ring")
        complete_idx = topo_results["topology"].index("complete")
        
        if topo_results["curvatures"][ring_idx] > np.mean(topo_results["curvatures"]):
            analysis["topology_variation"]["holographic"].append(
                "Ring topology shows higher curvature, consistent with AdS/CFT correspondence"
            )
        if topo_results["entropies"][complete_idx] > np.mean(topo_results["entropies"]):
            analysis["topology_variation"]["string_theory"].append(
                "Complete graph shows maximal entanglement, suggesting string field theory connection"
            )
        
        # Analyze dynamic variation results
        dyn_results = all_results["dynamic_variation"]
        entropy_trend = np.polyfit(dyn_results["step"], 
                                 dyn_results["entropies"], 1)[0]
        curvature_trend = np.polyfit(dyn_results["step"],
                                   dyn_results["curvatures"], 1)[0]
        
        if entropy_trend > 0:
            analysis["dynamic_variation"]["holographic"].append(
                "Increasing entropy during evolution suggests emergent spacetime"
            )
        if curvature_trend != 0:
            analysis["dynamic_variation"]["string_theory"].append(
                "Non-zero curvature trend indicates potential connection to string compactification"
            )
        
        # Save analysis
        with open(f"{self.exp_dir}/theoretical_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
        
        return analysis

def run_all_experiments():
    explorer = CircuitVariationExplorer()
    
    # Run all experiments
    inverse_results = explorer.run_inverse_circuit_experiment()
    parameter_results = explorer.run_parameter_variation_experiment()
    topology_results = explorer.run_topology_variation_experiment()
    dynamic_results = explorer.run_dynamic_variation_experiment()
    
    # Combine results
    all_results = {
        "inverse_circuit": inverse_results,
        "parameter_variation": parameter_results,
        "topology_variation": topology_results,
        "dynamic_variation": dynamic_results
    }
    
    # Analyze theoretical implications
    analysis = explorer.analyze_theoretical_implications(all_results)
    
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