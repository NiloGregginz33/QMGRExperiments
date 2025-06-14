import numpy as np
import matplotlib.pyplot as plt
from braket.circuits import Circuit, FreeParameter
from braket.devices import LocalSimulator
from braket.aws import AwsDevice
from braket.aws import AwsQuantumTask
from braket.default_simulator import DefaultSimulator
import numpy as np
import matplotlib.pyplot as plt
from braket.circuits import Circuit, FreeParameter
from braket.devices import LocalSimulator
from braket.aws import AwsDevice
from braket.aws import AwsQuantumTask
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform

# Step 1: Choose a device
device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")   # swap with AwsDevice(arn) for cloud runs
# Step 2: Page curve config
timesteps = np.linspace(0, 3 * np.pi, 30)  # φ(t) from 0 to 3π
entropy_values = []

class AdSGeometryAnalyzer:
    def __init__(self, device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1"):
        self.device = AwsDevice(device_arn)
        self.timesteps = np.linspace(0, 3 * np.pi, 15)

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

    def run_experiment(self):
        entropy_vals = []
        rt_area_comparison = []

        for phi_val in self.timesteps:
            phi = FreeParameter("phi")
            circ = Circuit()
            circ.h(0)
            circ.cnot(0, 2)
            circ.cnot(0, 3)
            circ.rx(0, phi)
            circ.cz(0, 1)
            circ.cnot(1, 2)
            circ.rx(2, phi)
            circ.cz(1, 3)
            circ.probability()

            task = self.device.run(circ, inputs={"phi": phi_val}, shots=1024)
            result = task.result()
            probs = np.array(result.values).reshape(-1)

            # Compute MI matrix
            mi_matrix = np.zeros((4, 4))
            for i in range(4):
                for j in range(i + 1, 4):
                    mi = self.compute_mi(probs, i, j, 4)
                    mi_matrix[i, j] = mi_matrix[j, i] = mi

            # Compute radiation entropy
            rad_probs = self.marginal_probs(probs, 4, [2, 3])
            S_rad = self.shannon_entropy(rad_probs)
            entropy_vals.append(S_rad)

            # Emergent geometry
            epsilon = 1e-6
            dist = 1 / (mi_matrix + epsilon)
            np.fill_diagonal(dist, 0)
            coords = MDS(n_components=2, dissimilarity='precomputed', random_state=42).fit_transform(dist)

            # Estimate RT area: use geodesic between Q2-Q3
            d_Q2Q3 = np.linalg.norm(coords[2] - coords[3])
            rt_area_comparison.append((phi_val, S_rad, d_Q2Q3))

            # Plot geometry
            plt.figure(figsize=(5, 5))
            plt.scatter(coords[:, 0], coords[:, 1], c='blue')
            for i in range(4):
                plt.text(coords[i, 0], coords[i, 1], f"Q{i}", fontsize=12)
            plt.title(f"Emergent Geometry φ={phi_val:.2f}")
            plt.axis('equal')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return rt_area_comparison, entropy_vals

    def plot_rt_relation(self, rt_data):
        phi_vals, entropies, dists = zip(*rt_data)
        plt.figure(figsize=(6, 4))
        plt.plot(dists, entropies, 'o-', label='S_rad vs. d(Q2,Q3)')
        plt.xlabel("Distance(Q2, Q3) in Emergent Geometry")
        plt.ylabel("Radiation Entropy S(A)")
        plt.title("RT-style Correlation: Geometry vs. Entanglement")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
# Step 3: Entropy function
def shannon_entropy(probs):
    probs = np.array(probs)
    return -np.sum(probs * np.log2(probs + 1e-12))  # avoid log(0)

def page_curve_demo():
    # Step 4: Run experiment across φ(t)
    for phi_val in timesteps:
        phi = FreeParameter("phi")
        circ = Circuit()

        # Qubits: [0,1] = black hole core + mode | [2,3] = radiation
        circ.h(0)
        circ.cnot(0, 2)
        circ.cnot(0, 3)

        # Charge injection (simulates quantum evaporation step)
        circ.rx(0, phi)         # Inject info/energy into BH
        circ.cz(0, 1)           # Phase coupling to auxiliary mode
        circ.cnot(1, 2)         # Evolve radiation entanglement
        circ.rx(2, phi)
        circ.cz(1, 3)

        # Measurement of radiation subsystem
        circ.probability(target=[2, 3])

        # Execute circuit with φ=phi_val
        task = device.run(circ, inputs={"phi": phi_val}, shots=1024)
        result = task.result()

        # Get flat probability vector
        raw_probs = result.values
        probs = np.array(raw_probs).reshape(-1)

        # Compute entropy
        entropy = shannon_entropy(probs)
        entropy_values.append(entropy)

        # Optional debug
        print(f"φ = {phi_val:.2f} → Entropy = {entropy:.4f}")

    # Step 5: Plot Page curve
    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, entropy_values, marker='o', label="Radiation Entropy")
    plt.xlabel("Evaporation Phase φ(t)")
    plt.ylabel("Shannon Entropy")
    plt.title("Simulated Page Curve via Charge Injection (Braket)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def page_curve_mi_demo():
    use_local = False
    device = LocalSimulator() if use_local else AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")

    timesteps = np.linspace(0, 3 * np.pi, 15)
    entropy_vals = []
    mi_snapshots = []

    def shannon_entropy(probs):
        probs = np.array(probs)
        return -np.sum(probs * np.log2(probs + 1e-12))

    def marginal_probs(probs, total_qubits, target_idxs):
        marginal = {}
        for idx, p in enumerate(probs):
            b = format(idx, f"0{total_qubits}b")
            key = ''.join([b[i] for i in target_idxs])
            marginal[key] = marginal.get(key, 0) + p
        return np.array(list(marginal.values()))

    def compute_mi(probs, qA, qB, total_qubits):
        AB = marginal_probs(probs, total_qubits, [qA, qB])
        A = marginal_probs(probs, total_qubits, [qA])
        B = marginal_probs(probs, total_qubits, [qB])
        return shannon_entropy(A) + shannon_entropy(B) - shannon_entropy(AB)

    # --- Run experiment ---
    for phi_val in timesteps:
        phi = FreeParameter("phi")
        circ = Circuit()
        circ.h(0)
        circ.cnot(0, 2)
        circ.cnot(0, 3)
        circ.rx(0, phi)
        circ.cz(0, 1)
        circ.cnot(1, 2)
        circ.rx(2, phi)
        circ.cz(1, 3)
        circ.probability()

        task = device.run(circ, inputs={"phi": phi_val}, shots=1024)
        result = task.result()
        probs = np.array(result.values).reshape(-1)

        rad_probs = marginal_probs(probs, 4, [2, 3])
        entropy_vals.append(shannon_entropy(rad_probs))

        mi_matrix = np.zeros((4, 4))
        for i in range(4):
            for j in range(i + 1, 4):
                mi = compute_mi(probs, i, j, 4)
                mi_matrix[i, j] = mi_matrix[j, i] = mi
        mi_snapshots.append(mi_matrix)

        print(f"φ = {phi_val:.2f}, Entropy = {entropy_vals[-1]:.4f}")

    # --- Plot Page Curve ---
    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, entropy_vals, marker='o', label="Radiation Entropy")
    plt.xlabel("Evaporation Phase φ(t)")
    plt.ylabel("Entropy")
    plt.title("Simulated Page Curve (Braket)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot MI Matrices + Emergent Geometry ---
    for idx, (phi_val, mi_matrix) in enumerate(zip(timesteps, mi_snapshots)):
        # MI Heatmap
        plt.figure(figsize=(5, 4))
        plt.imshow(mi_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(f"Mutual Info Matrix at φ={phi_val:.2f}")
        plt.xlabel("Qubits")
        plt.ylabel("Qubits")
        plt.xticks(range(4))
        plt.yticks(range(4))
        plt.tight_layout()
        plt.show()

        # Geometry from MI → Dissimilarity
        epsilon = 1e-6
        dist = 1 / (mi_matrix + epsilon)
        np.fill_diagonal(dist, 0)

        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
        coords = mds.fit_transform(dist)

        # Spatial Geometry Plot
        plt.figure(figsize=(5, 5))
        plt.scatter(coords[:, 0], coords[:, 1], c='blue')
        for i in range(4):
            plt.text(coords[i, 0], coords[i, 1], f"Q{i}", fontsize=12)
        plt.title(f"Emergent Geometry at φ={phi_val:.2f}")
        plt.axis('equal')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

ads = AdSGeometryAnalyzer()
rt_data, entropy_vals = ads.run_experiment()
ads.plot_rt_relation(rt_data)
