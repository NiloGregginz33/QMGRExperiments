import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from itertools import combinations
import json
import os
from datetime import datetime
import sys
sys.path.append('src')
from AWSFactory import LocalSimulator
from braket.circuits import Circuit, FreeParameter

class LocalEmergentSpacetime:
    def __init__(self, device):
        self.device = device
        self.timesteps = np.linspace(0, 3 * np.pi, 15)
        self.mi_matrices = []

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
        for phi_val in self.timesteps:
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

            task = self.device.run(circ, inputs={"phi": phi_val}, shots=1024)
            result = task.result()
            probs = np.array(result.values).reshape(-1)

            mi_matrix = np.zeros((4, 4))
            for i in range(4):
                for j in range(i + 1, 4):
                    mi = self.compute_mi(probs, i, j, 4)
                    mi_matrix[i, j] = mi_matrix[j, i] = mi
            self.mi_matrices.append(mi_matrix)

class EmergentSpacetimeExperiment:
    def __init__(self, n_qubits=4, shots=1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.device = LocalSimulator()
        self.exp_dir = f"experiments/emergent_spacetime_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.exp_dir, exist_ok=True)
        
    def run_experiment(self, n_steps=15):
        """Run the emergent spacetime experiment"""
        print("\nRunning Emergent Spacetime Experiment...")
        
        # Initialize experiment
        spacetime = LocalEmergentSpacetime(self.device)
        
        # Run the experiment
        spacetime.run()
        
        # Get results
        mi_matrices = spacetime.mi_matrices
        
        # Analyze results
        results = self.analyze_results(mi_matrices)
        
        # Save results
        self.save_results(results)
        
        # Plot results
        self.plot_results(results)
        
        return results
    
    def analyze_results(self, mi_matrices):
        """Analyze the mutual information matrices to extract geometric features"""
        results = {
            "timesteps": np.linspace(0, 3 * np.pi, len(mi_matrices)).tolist(),
            "entropies": [],
            "curvatures": [],
            "distances": [],
            "geometries": []
        }
        
        for mi_matrix in mi_matrices:
            # Convert MI to distances
            epsilon = 1e-6
            dist = 1 / (mi_matrix + epsilon)
            np.fill_diagonal(dist, 0)
            
            # Compute 3D embedding
            coords = MDS(n_components=3, dissimilarity='precomputed').fit_transform(dist)
            
            # Compute entropy
            entropy = self.compute_entropy(mi_matrix)
            
            # Compute curvature
            curvature = self.compute_curvature(coords)
            
            # Compute average distance
            avg_dist = np.mean(dist[dist > 0])
            
            # Store results
            results["entropies"].append(float(entropy))
            results["curvatures"].append(float(curvature))
            results["distances"].append(float(avg_dist))
            results["geometries"].append(coords.tolist())
        
        return results
    
    def compute_entropy(self, mi_matrix):
        """Compute von Neumann entropy from mutual information matrix"""
        # Use average mutual information as a proxy for entropy
        return np.mean(mi_matrix[mi_matrix > 0])
    
    def compute_curvature(self, coords):
        """Compute local curvature using angle defect method"""
        curvatures = []
        for triplet in combinations(range(len(coords)), 3):
            # Get triangle vertices
            a, b, c = coords[triplet[0]], coords[triplet[1]], coords[triplet[2]]
            
            # Compute edge lengths
            ab = np.linalg.norm(b - a)
            bc = np.linalg.norm(c - b)
            ca = np.linalg.norm(a - c)
            
            # Compute angles using law of cosines
            alpha = np.arccos((ab**2 + ca**2 - bc**2) / (2 * ab * ca))
            beta = np.arccos((ab**2 + bc**2 - ca**2) / (2 * ab * bc))
            gamma = np.arccos((bc**2 + ca**2 - ab**2) / (2 * bc * ca))
            
            # Compute angle defect
            defect = np.pi - (alpha + beta + gamma)
            curvatures.append(defect)
        
        return np.mean(curvatures)
    
    def save_results(self, results):
        """Save experiment results to file"""
        with open(f"{self.exp_dir}/results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    def plot_results(self, results):
        """Plot experiment results"""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot entropy vs time
        axes[0,0].plot(results["timesteps"], results["entropies"], 'b-', label='Entropy')
        axes[0,0].set_xlabel('Time (φ)')
        axes[0,0].set_ylabel('Entropy (bits)')
        axes[0,0].set_title('Entropy Evolution')
        axes[0,0].grid(True)
        axes[0,0].legend()
        
        # Plot curvature vs time
        axes[0,1].plot(results["timesteps"], results["curvatures"], 'r-', label='Curvature')
        axes[0,1].set_xlabel('Time (φ)')
        axes[0,1].set_ylabel('Curvature')
        axes[0,1].set_title('Curvature Evolution')
        axes[0,1].grid(True)
        axes[0,1].legend()
        
        # Plot distance vs time
        axes[1,0].plot(results["timesteps"], results["distances"], 'g-', label='Distance')
        axes[1,0].set_xlabel('Time (φ)')
        axes[1,0].set_ylabel('Average Distance')
        axes[1,0].set_title('Distance Evolution')
        axes[1,0].grid(True)
        axes[1,0].legend()
        
        # Plot final geometry
        final_geometry = np.array(results["geometries"][-1])
        ax = axes[1,1]
        scatter = ax.scatter(final_geometry[:,0], final_geometry[:,1], 
                           c=final_geometry[:,2], cmap='viridis')
        for i in range(len(final_geometry)):
            ax.text(final_geometry[i,0], final_geometry[i,1], f'Q{i}', 
                   fontsize=12, ha='center', va='center')
        ax.set_title('Final Emergent Geometry')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(scatter, ax=ax, label='Z')
        
        plt.tight_layout()
        plt.savefig(f"{self.exp_dir}/results.png")
        plt.close()
        
        # Plot 3D geometry evolution
        self.plot_geometry_evolution(results["geometries"], results["timesteps"])
    
    def plot_geometry_evolution(self, geometries, timesteps):
        """Plot the evolution of the emergent geometry in 3D"""
        fig = plt.figure(figsize=(15, 5))
        
        # Plot geometries at different time steps
        for i, (geometry, t) in enumerate(zip(geometries, timesteps)):
            if i % 3 == 0:  # Plot every third geometry to avoid overcrowding
                ax = fig.add_subplot(1, 5, i//3 + 1, projection='3d')
                coords = np.array(geometry)
                
                # Plot points
                scatter = ax.scatter(coords[:,0], coords[:,1], coords[:,2], 
                                   c=coords[:,2], cmap='viridis')
                
                # Add labels
                for j in range(len(coords)):
                    ax.text(coords[j,0], coords[j,1], coords[j,2], f'Q{j}', 
                           fontsize=10, ha='center', va='center')
                
                ax.set_title(f't = {t:.2f}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
        
        plt.tight_layout()
        plt.savefig(f"{self.exp_dir}/geometry_evolution.png")
        plt.close()

if __name__ == "__main__":
    # Run the experiment
    experiment = EmergentSpacetimeExperiment()
    results = experiment.run_experiment()
    
    # Print summary
    print("\nExperiment Summary:")
    print(f"Final Entropy: {results['entropies'][-1]:.4f}")
    print(f"Final Curvature: {results['curvatures'][-1]:.4f}")
    print(f"Final Average Distance: {results['distances'][-1]:.4f}")
    print(f"\nResults saved to: {experiment.exp_dir}") 