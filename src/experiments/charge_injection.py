import numpy as np
import matplotlib.pyplot as plt
from braket.circuits import Circuit, FreeParameter
from braket.devices import LocalSimulator
from utils.experiment_logger import PhysicsExperimentLogger
from sklearn.manifold import MDS
import seaborn as sns

class ChargeInjectionExperiment:
    def __init__(self, num_qubits=5, shots=8192):  # 1 BH + 4 radiation qubits
        self.device = LocalSimulator()
        self.num_qubits = num_qubits
        self.shots = shots
        self.logger = PhysicsExperimentLogger("charge_injection")
        
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
    
    def build_circuit(self, injection_pattern="time_gaps", num_injections=10, gap_cycles=100):
        """Build circuit with different charge injection patterns.
        
        Args:
            injection_pattern: "alternating", "prolonged", "time_gaps", or "oscillating"
            num_injections: Number of charge injections (default: 10)
            gap_cycles: Number of idle cycles between injections (default: 100)
        """
        circ = Circuit()
        
        # Initialize black hole qubit in superposition
        circ.h(0)
        
        # Charge injection patterns
        if injection_pattern == "alternating":
            # Alternate between positive (X) and negative (Z) charge
            for t in range(num_injections):
                if t % 2 == 0:
                    # Positive charge injection
                    circ.x(0)
                else:
                    # Negative charge injection
                    circ.z(0)
                
                # Entangle with radiation qubits sequentially
                for i in range(1, self.num_qubits):
                    circ.h(0)  # Reset superposition
                    circ.cnot(0, i)
                    
        elif injection_pattern == "prolonged":
            # Prolonged charge injection with cycles
            cycle_length = num_injections // 2
            for t in range(num_injections):
                if (t // cycle_length) % 2 == 0:
                    # Positive charge cycle
                    circ.x(0)
                else:
                    # Negative charge cycle
                    circ.z(0)
                
                # Entangle with radiation qubits sequentially
                for i in range(1, self.num_qubits):
                    circ.h(0)  # Reset superposition
                    circ.cnot(0, i)
                    
        elif injection_pattern == "time_gaps":
            # Charge injection with time gaps
            for t in range(num_injections):
                if t % 2 == 0:
                    # Positive charge injection
                    circ.x(0)
                else:
                    # Negative charge injection
                    circ.z(0)
                
                # Entangle with radiation qubits sequentially
                for i in range(1, self.num_qubits):
                    circ.h(0)  # Reset superposition
                    circ.cnot(0, i)
                
                # Add time gaps
                for _ in range(gap_cycles):
                    circ.i(0)  # Identity gate for time gap
                    
        elif injection_pattern == "oscillating":
            # Oscillating charge injection
            for t in range(num_injections):
                phase = np.sin(2 * np.pi * t / num_injections) * np.pi
                if phase > 0:
                    circ.x(0)
                else:
                    circ.z(0)
                
                # Entangle with radiation qubits sequentially
                for i in range(1, self.num_qubits):
                    circ.h(0)  # Reset superposition
                    circ.cnot(0, i)
        
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
        patterns = ["alternating", "prolonged", "time_gaps", "oscillating"]
        num_injections = 10  # Match ex3.py
        gap_cycles = 100    # Match ex3.py
        
        results = []
        for pattern in patterns:
            # Build and run circuit
            circ = self.build_circuit(pattern, num_injections, gap_cycles)
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
                "num_injections": num_injections,
                "gap_cycles": gap_cycles,
                "mi_matrix": mi_matrix,
                "coords_2d": coords_2d,
                "coords_3d": coords_3d,
                "curvature": curvature
            }
            results.append(result)
            self.logger.log_result(result)
            
            # Plot geometry
            plt.figure(figsize=(12, 5))
            
            # 2D plot
            plt.subplot(121)
            plt.scatter(coords_2d[:,0], coords_2d[:,1])
            for i in range(self.num_qubits):
                plt.text(coords_2d[i,0], coords_2d[i,1], f"Q{i}")
            plt.title(f"2D Geometry ({pattern}, n={num_injections}, g={gap_cycles})")
            
            # MI heatmap
            plt.subplot(122)
            sns.heatmap(mi_matrix, annot=True, cmap='viridis')
            plt.title(f"MI Matrix ({pattern}, n={num_injections}, g={gap_cycles})")
            
            plt.tight_layout()
            plt.savefig(f'plots/charge_injection_{pattern}_n{num_injections}_g{gap_cycles}.png')
            plt.close()
            
            print(f"Pattern: {pattern}, Injections: {num_injections}, Gaps: {gap_cycles}")
            print(f"Average MI: {np.mean(mi_matrix):.4f}")
            print(f"Curvature: {curvature:.4f}")
            print("-" * 50)
        
        return results

if __name__ == "__main__":
    experiment = ChargeInjectionExperiment()
    results = experiment.run()
    experiment.logger.finalize() 