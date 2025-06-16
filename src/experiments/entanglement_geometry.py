import numpy as np
import matplotlib.pyplot as plt
from braket.circuits import Circuit, FreeParameter
from braket.devices import LocalSimulator
from utils.experiment_logger import PhysicsExperimentLogger
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

class EntanglementGeometryExperiment:
    def __init__(self, num_qubits=6, shots=1024):
        self.device = LocalSimulator()
        self.num_qubits = num_qubits
        self.shots = shots
        self.logger = PhysicsExperimentLogger("entanglement_geometry")
        
    def shannon_entropy(self, probs):
        probs = np.array(probs)
        probs = probs / np.sum(probs)
        return -np.sum(probs * np.log2(probs + 1e-12))
    
    def marginal_probs(self, probs, total_qubits, keep):
        marginal = {}
        for idx, p in enumerate(probs):
            b = format(idx, f"0{total_qubits}b")
            key = ''.join([b[i] for i in keep])
            marginal[key] = marginal.get(key, 0) + p
        return np.array(list(marginal.values()))
    
    def compute_mi(self, probs, qA, qB, total_qubits):
        AB = self.marginal_probs(probs, total_qubits, [qA, qB])
        A = self.marginal_probs(probs, total_qubits, [qA])
        B = self.marginal_probs(probs, total_qubits, [qB])
        return self.shannon_entropy(A) + self.shannon_entropy(B) - self.shannon_entropy(AB)
    
    def build_circuit(self, pattern="linear", strength=1.0):
        """Build circuit with different entanglement patterns.
        
        Args:
            pattern: "linear", "star", "ring", or "random"
            strength: Parameter controlling entanglement strength (0-1)
        """
        circ = Circuit()
        
        # Initialize all qubits in superposition
        for i in range(self.num_qubits):
            circ.h(i)
        
        if pattern == "linear":
            # Linear chain of entanglement
            for i in range(self.num_qubits - 1):
                circ.cnot(i, i+1)
                circ.rz(i+1, strength * np.pi/4)
                
        elif pattern == "star":
            # Star topology with central qubit
            center = 0
            for i in range(1, self.num_qubits):
                circ.cnot(center, i)
                circ.rz(i, strength * np.pi/4)
                
        elif pattern == "ring":
            # Ring topology
            for i in range(self.num_qubits):
                circ.cnot(i, (i+1) % self.num_qubits)
                circ.rz((i+1) % self.num_qubits, strength * np.pi/4)
                
        elif pattern == "random":
            # Random entanglement pattern
            np.random.seed(42)  # For reproducibility
            for _ in range(self.num_qubits * 2):
                i, j = np.random.choice(self.num_qubits, 2, replace=False)
                circ.cnot(i, j)
                circ.rz(j, strength * np.pi/4)
        
        circ.probability()
        return circ
    
    def compute_geometry(self, mi_matrix):
        """Compute emergent geometry from mutual information matrix."""
        epsilon = 1e-6
        dist = 1 / (mi_matrix + epsilon)
        np.fill_diagonal(dist, 0)
        
        # 2D embedding
        coords_2d = MDS(n_components=2, dissimilarity='precomputed').fit_transform(dist)
        
        # 3D embedding
        coords_3d = MDS(n_components=3, dissimilarity='precomputed').fit_transform(dist)
        
        return coords_2d, coords_3d
    
    def estimate_curvature(self, coords_3d):
        """Estimate local curvature using angle defect method."""
        curvatures = []
        for i in range(self.num_qubits):
            for j in range(i+1, self.num_qubits):
                for k in range(j+1, self.num_qubits):
                    # Compute angles of triangle
                    a = np.linalg.norm(coords_3d[j] - coords_3d[k])
                    b = np.linalg.norm(coords_3d[i] - coords_3d[k])
                    c = np.linalg.norm(coords_3d[i] - coords_3d[j])
                    
                    # Law of cosines
                    alpha = np.arccos((b**2 + c**2 - a**2) / (2*b*c))
                    beta = np.arccos((a**2 + c**2 - b**2) / (2*a*c))
                    gamma = np.arccos((a**2 + b**2 - c**2) / (2*a*b))
                    
                    # Angle defect
                    curvature = np.pi - (alpha + beta + gamma)
                    curvatures.append(curvature)
        
        return np.mean(curvatures)
    
    def run(self):
        patterns = ["linear", "star", "ring", "random"]
        strengths = np.linspace(0.2, 1.0, 5)
        
        results = []
        for pattern in patterns:
            for strength in strengths:
                # Build and run circuit
                circ = self.build_circuit(pattern, strength)
                task = self.device.run(circ, shots=self.shots)
                probs = np.array(task.result().values).reshape(-1)
                
                # Compute mutual information matrix
                mi_matrix = np.zeros((self.num_qubits, self.num_qubits))
                for i in range(self.num_qubits):
                    for j in range(i+1, self.num_qubits):
                        mi = self.compute_mi(probs, i, j, self.num_qubits)
                        mi_matrix[i,j] = mi_matrix[j,i] = mi
                
                # Compute geometry
                coords_2d, coords_3d = self.compute_geometry(mi_matrix)
                curvature = self.estimate_curvature(coords_3d)
                
                # Log results
                result = {
                    "pattern": pattern,
                    "strength": strength,
                    "mi_matrix": mi_matrix,
                    "coords_2d": coords_2d,
                    "coords_3d": coords_3d,
                    "curvature": curvature
                }
                results.append(result)
                self.logger.log_result(result)
                
                # Plot geometry
                plt.figure(figsize=(10, 5))
                
                # 2D plot
                plt.subplot(121)
                plt.scatter(coords_2d[:,0], coords_2d[:,1])
                for i in range(self.num_qubits):
                    plt.text(coords_2d[i,0], coords_2d[i,1], f"Q{i}")
                plt.title(f"2D Geometry ({pattern}, s={strength:.1f})")
                
                # MI heatmap
                plt.subplot(122)
                sns.heatmap(mi_matrix, annot=True, cmap='viridis')
                plt.title(f"MI Matrix ({pattern}, s={strength:.1f})")
                
                plt.tight_layout()
                plt.savefig(f'plots/geometry_{pattern}_s{strength:.1f}.png')
                plt.close()
                
                print(f"Pattern: {pattern}, Strength: {strength:.1f}")
                print(f"Average MI: {np.mean(mi_matrix):.4f}")
                print(f"Curvature: {curvature:.4f}")
                print("-" * 50)
        
        return results

if __name__ == "__main__":
    experiment = EntanglementGeometryExperiment()
    results = experiment.run()
    experiment.logger.finalize() 