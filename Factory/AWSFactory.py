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
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from itertools import combinations
import warnings
import seaborn as sns
import pandas as pd

from braket.circuits import Circuit, Instruction, QubitSet
import os

print("ENV CHECK:")
print("AWS_ACCESS_KEY_ID =", os.getenv("AWS_ACCESS_KEY_ID"))
print("AWS_SECRET_ACCESS_KEY =", os.getenv("AWS_SECRET_ACCESS_KEY"))
print("AWS_DEFAULT_REGION =", os.getenv("AWS_DEFAULT_REGION"))

def remap_circuit_to_contiguous_qubits(circuit: Circuit):
    """
    Remaps a Braket circuit to use contiguous qubit indices starting from 0.
    Returns the new circuit and a mapping from old to new qubit indices.
    """
    used_qubits = set()

    for instr in circuit.instructions:
        # Normalize targets
        if instr.target:
            for q in QubitSet(instr.target):
                used_qubits.add(int(q))
        # Normalize controls
        if instr.control:
            for q in QubitSet(instr.control):
                used_qubits.add(int(q))

    sorted_qubits = sorted(used_qubits)
    qubit_mapping = {old: new for new, old in enumerate(sorted_qubits)}

    new_circuit = Circuit()

    for instr in circuit.instructions:
        targets = [qubit_mapping[int(q)] for q in QubitSet(instr.target)] if instr.target else []
        controls = [qubit_mapping[int(q)] for q in QubitSet(instr.control)] if instr.control else []

        for ctrl in controls or [None]:
            new_instr = Instruction(
                operator=instr.operator,
                target=targets,
                control=ctrl
            )
            new_circuit.add_instruction(new_instr)

    return new_circuit, qubit_mapping

warnings.filterwarnings("ignore", category=UserWarning)
arn = "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1"
# Step 1: Choose a device
device = AwsDevice(arn)   # swap with AwsDevice(arn) for cloud runs
# Step 2: Page curve config
timesteps = np.linspace(0, 3 * np.pi, 30)  # φ(t) from 0 to 3π
entropy_values = []

##def log_func(x, a, b):
##    return a * np.log(x + 1e-6) + b
##
##def fit_rt_plot(rt_data):
##    phi_vals, entropies, dists = zip(*rt_data)
##    dists = np.array(dists)
##    entropies = np.array(entropies)
##
##    # Linear fit
##    lin_model = LinearRegression()
##    lin_model.fit(dists.reshape(-1, 1), entropies)
##    lin_pred = lin_model.predict(dists.reshape(-1, 1))
##
##    # Logarithmic fit
##    popt, _ = curve_fit(log_func, dists, entropies)
##    log_pred = log_func(dists, *popt)
##
##    # Plot both fits
##    plt.figure(figsize=(7, 5))
##    plt.scatter(dists, entropies, c='black', label="Data")
##    plt.plot(dists, lin_pred, label=f"Linear Fit: S = {lin_model.coef_[0]:.2f}·d + {lin_model.intercept_:.2f}", linestyle='--')
##    plt.plot(dists, log_pred, label=f"Log Fit: S = {popt[0]:.2f}·log(d) + {popt[1]:.2f}", linestyle='-.')
##    plt.xlabel("Distance(Q2, Q3) in Emergent Geometry")
##    plt.ylabel("Radiation Entropy S(Q2,Q3)")
##    plt.title("RT-style Correlation + Regression")
##    plt.grid(True)
##    plt.legend()
##    plt.tight_layout()
##    plt.show()

class ScaledEmergentSpacetime:
    def __init__(self, device, num_qubits=6, max_cut_size=3):
        self.device = device
        self.num_qubits = num_qubits
        self.max_cut_size = max_cut_size
        self.timesteps = np.linspace(0, 3 * np.pi, 15)
        self.entropy_area_data = []

    def shannon_entropy(self, probs):
        probs = np.array(probs)
        probs /= np.sum(probs)
        return -np.sum(probs * np.log2(probs + 1e-12))

    def marginal_probs(self, probs, total_qubits, keep):
        marginal = {}
        for idx, p in enumerate(probs):
            b = format(idx, f"0{total_qubits}b")
            key = ''.join([b[i] for i in keep])
            marginal[key] = marginal.get(key, 0) + p
        return np.array(list(marginal.values()))

    def build_scaled_circuit(self, phi):
        circ = Circuit()
        circ.h(0)
        # Spread entanglement and inject φ-dependent dynamics
        for i in range(1, self.num_qubits):
            circ.cnot(0, i)
            circ.rx(i, phi if i % 2 == 0 else -phi)
        circ.probability()
        return circ

    def run_entropy_area_experiment(self):
        phi = FreeParameter("phi")
        for phi_val in self.timesteps:
            circ = self.build_scaled_circuit(phi)
            task = self.device.run(circ, inputs={"phi": phi_val}, shots=2048)
            result = task.result()
            probs = np.array(result.values).reshape(-1)

            cut_sizes = list(range(1, self.max_cut_size + 1))
            for cut in cut_sizes:
                keep = list(range(1, 1 + cut))  # Skip qubit 0 (core BH)
                marg = self.marginal_probs(probs, self.num_qubits, keep)
                S = self.shannon_entropy(marg)
                self.entropy_area_data.append((cut, phi_val, S))

    def plot_entropy_vs_area(self):
        df = pd.DataFrame(self.entropy_area_data, columns=["cut_size", "phi", "entropy"])
        avg_entropy = df.groupby("cut_size")["entropy"].mean().reset_index()

        plt.figure(figsize=(7, 5))
        plt.plot(avg_entropy["cut_size"], avg_entropy["entropy"], marker='o')
        plt.xlabel("Radiation Cut Size (# Qubits)")
        plt.ylabel("Average Entropy (bits)")
        plt.title("Entropy vs Area Law (Scaled Circuit)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return df


def marginal(probs, keep, total=7):
    out = {}
    for idx, p in enumerate(probs):
        bit = format(idx, f"0{total}b")
        key = ''.join(bit[i] for i in keep)
        out[key] = out.get(key, 0) + p
    return np.array(list(out.values()))

def entropy(p):
    p = p+1e-12
    return -(p*np.log2(p)).sum()

def perfect_tensor():
    """Return a 6-qubit perfect-tensor sub-circuit (HaPPY building block)."""
    qc = QuantumCircuit(6)
    # One simple construction: 3 GHZ pairs → layer of CZs → random phase layer
    for i in range(0, 6, 2):
        qc.h(i)
        qc.cx(i, i+1)
    # Entangle across pairs
    for (a, b) in [(0,2), (1,4), (3,5)]:
        qc.cz(a, b)
    # Optional random single-qubit rotations to break trivial structure
    for q in range(6):
        qc.rx(np.pi/4, q)
    return qc

def happpy_7qubit():
    qc = perfect_tensor()      # outputs 6 legs
    qc.qubit = qc.qubits + [qc.qregs[0].add(1)]  # add 7th qubit as “logical center”
    # Route one leg into the logical center (index 6)
    qc.cx(2,6)
    qc.h(6)
    return qc

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

class AdSGeometryAnalyzer3D:
    def __init__(self, use_local=False):
        self.timesteps = np.linspace(0, 3 * np.pi, 15)
        self.device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")
        self.rt_data = []
        self.coords_list_2d = []
        self.coords_list_3d = []

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

    def run(self, mode="original"):
        for phi_val in self.timesteps:
            phi = FreeParameter("phi")
            circ = Circuit()
            circ.h(0)

            # Common initial entanglement
            circ.cnot(0, 2)
            circ.cnot(0, 3)

            # Charge injection: controlled vs. original
            if mode == "original":
                circ.rx(0, phi)
                circ.cz(0, 1)
                circ.cnot(1, 2)
                circ.rx(2, phi)
                circ.cz(1, 3)
            elif mode == "control":
                # Scrambled sequence: same gates, different order
                circ.rx(2, phi)
                circ.cnot(1, 2)
                circ.cz(0, 1)
                circ.cz(1, 3)
                circ.rx(0, phi)
            else:
                raise ValueError("Unknown mode. Use 'original' or 'control'.")

            circ.probability()

            task = self.device.run(circ, inputs={"phi": phi_val}, shots=1024)
            result = task.result()
            probs = np.array(result.values).reshape(-1)

            mi_matrix = np.zeros((4, 4))
            for i in range(4):
                for j in range(i + 1, 4):
                    mi = self.compute_mi(probs, i, j, 4)
                    mi_matrix[i, j] = mi_matrix[j, i] = mi

            rad_probs = self.marginal_probs(probs, 4, [2, 3])
            S_rad = self.shannon_entropy(rad_probs)

            epsilon = 1e-6
            dist = 1 / (mi_matrix + epsilon)
            np.fill_diagonal(dist, 0)

            coords2 = MDS(n_components=2, dissimilarity='precomputed').fit_transform(dist)
            coords3 = MDS(n_components=3, dissimilarity='precomputed').fit_transform(dist)

            d_Q2Q3 = np.linalg.norm(coords2[2] - coords2[3])

            self.rt_data.append((phi_val, S_rad, d_Q2Q3))
            self.coords_list_2d.append(coords2)
            self.coords_list_3d.append(coords3)

        triplets_to_probe = [(0,1,2), (1,2,3), (2,3,4)]  # Adjust based on circuit size

        curvatures = []
        for triplet in triplets_to_probe:
            curv = self.estimate_local_curvature(coords3, triplet)
            curvatures.append((triplet, curv))
            print(f"Triplet {triplet}: Curvature = {curv:.4f}")



    def fit_rt_plot(self):
        def log_func(x, a, b):
            return a * np.log(x + 1e-6) + b

        phi_vals, entropies, dists = zip(*self.rt_data)
        dists = np.array(dists)
        entropies = np.array(entropies)
        
        def log_func(x, a, b):
            return a * np.log(x + 1e-6) + b

        # Linear fit
        lin_model = LinearRegression()
        lin_model.fit(dists.reshape(-1, 1), entropies)
        lin_pred = lin_model.predict(dists.reshape(-1, 1))

        # Logarithmic fit
        popt, _ = curve_fit(log_func, dists, entropies)
        log_pred = log_func(dists, *popt)

        # Plot
        plt.figure(figsize=(7, 5))
        plt.scatter(dists, entropies, c='black', label="Data")
        plt.plot(dists, lin_pred, label=f"Linear Fit: S = {lin_model.coef_[0]:.2f}·d + {lin_model.intercept_:.2f}", linestyle='--')
        plt.plot(dists, log_pred, label=f"Log Fit: S = {popt[0]:.2f}·log(d) + {popt[1]:.2f}", linestyle='-.')
        plt.xlabel("Distance(Q2, Q3)")
        plt.ylabel("Entropy S(Q2,Q3)")
        plt.title("RT Correlation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def estimate_local_curvature(self, coords, triplet):
        """Estimates Gaussian curvature via angle defect for a triangle."""
        from numpy.linalg import norm
        import numpy as np

        i, j, k = triplet
        a = norm(coords[j] - coords[k])
        b = norm(coords[i] - coords[k])
        c = norm(coords[i] - coords[j])

        # Clamp for numerical stability
        def safe_acos(x):
            return np.arccos(np.clip(x, -1.0, 1.0))

        angle_i = safe_acos((b**2 + c**2 - a**2) / (2 * b * c))
        angle_j = safe_acos((a**2 + c**2 - b**2) / (2 * a * c))
        angle_k = safe_acos((a**2 + b**2 - c**2) / (2 * a * b))

        curvature = (angle_i + angle_j + angle_k) - np.pi
        return curvature

    def animate_geometry(self):
        fig = plt.figure(figsize=(12, 5))
        ax2d = fig.add_subplot(121)
        ax3d = fig.add_subplot(122, projection='3d')

        def update(frame):
            ax2d.clear()
            ax3d.clear()

            coords2 = self.coords_list_2d[frame]
            coords3 = self.coords_list_3d[frame]

            ax2d.scatter(coords2[:, 0], coords2[:, 1], c='blue')
            for i in range(4):
                ax2d.text(coords2[i, 0], coords2[i, 1], f"Q{i}", fontsize=12)
            ax2d.set_title(f"2D Geometry φ={self.timesteps[frame]:.2f}")
            ax2d.axis('equal')

            ax3d.scatter(coords3[:, 0], coords3[:, 1], coords3[:, 2], c='purple')
            for i in range(4):
                ax3d.text(coords3[i, 0], coords3[i, 1], coords3[i, 2], f"Q{i}", fontsize=12)
            ax3d.set_title(f"3D Geometry φ={self.timesteps[frame]:.2f}")
            ax3d.set_xlim(-2, 2)
            ax3d.set_ylim(-2, 2)
            ax3d.set_zlim(-2, 2)

        ani = animation.FuncAnimation(fig, update, frames=len(self.coords_list_2d), interval=1200, repeat=True)
        plt.tight_layout()
        plt.show()

class AdSGeometryAnalyzer6Q:
    def __init__(self, n_qubits=6, timesteps=15, mode="flat"):
        self.device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")
        self.n_qubits = n_qubits
        self.timesteps = np.linspace(0, 3 * np.pi, timesteps)
        self.rt_data = []
        self.coords_list_2d = []
        self.coords_list_3d = []
        self.mode = mode

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

    def estimate_local_curvature(self, coords, triplet):
        from numpy.linalg import norm

        i, j, k = triplet
        a = norm(coords[j] - coords[k])
        b = norm(coords[i] - coords[k])
        c = norm(coords[i] - coords[j])

        def safe_acos(x):
            return np.arccos(np.clip(x, -1.0, 1.0))

        angle_i = safe_acos((b**2 + c**2 - a**2) / (2 * b * c))
        angle_j = safe_acos((a**2 + c**2 - b**2) / (2 * a * c))
        angle_k = safe_acos((a**2 + b**2 - c**2) / (2 * a * b))

        return (angle_i + angle_j + angle_k) - np.pi

    def run(self):
        for phi_val in self.timesteps:
            phi = FreeParameter("phi")
            circ = Circuit()

            circ.h(0)
            circ.cnot(0, 2)
            circ.cnot(0, 3)

            if self.mode == "flat":
                circ.rx(0, phi)
                circ.cz(0, 1)
                circ.cnot(1, 2)
                circ.rx(2, phi)
                circ.cz(1, 3)
                circ.cnot(3, 4)
                circ.rx(4, phi)
                circ.cnot(4, 5)
            elif self.mode == "curved":
                # More interwoven, nonlocal couplings to mimic negative curvature
                circ.rx(0, phi)
                circ.rx(1, phi)
                circ.rx(2, phi)
                circ.cz(0, 3)
                circ.cz(1, 4)
                circ.cz(2, 5)
                circ.cnot(0, 5)
                circ.cnot(5, 3)
                circ.cz(3, 4)
                circ.cz(4, 1)
                circ.cnot(4, 2)

            circ.probability()
            task = self.device.run(circ, inputs={"phi": phi_val}, shots=1024)
            result = task.result()
            probs = np.array(result.values).reshape(-1)

            mi_matrix = np.zeros((self.n_qubits, self.n_qubits))
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    mi = self.compute_mi(probs, i, j, self.n_qubits)
                    mi_matrix[i, j] = mi_matrix[j, i] = mi

            rad_probs = self.marginal_probs(probs, self.n_qubits, [3, 4])
            S_rad = self.shannon_entropy(rad_probs)

            epsilon = 1e-6
            dist = np.exp(-mi_matrix)
            dist[dist > 1e4] = 1e4 
            np.fill_diagonal(dist, 0)

            coords2 = MDS(n_components=2, dissimilarity='precomputed').fit_transform(dist)
            coords3 = MDS(n_components=3, dissimilarity='precomputed').fit_transform(dist)

            d_Q34 = np.linalg.norm(coords2[3] - coords2[4])

            self.rt_data.append((phi_val, S_rad, d_Q34))
            self.coords_list_2d.append(coords2)
            self.coords_list_3d.append(coords3)

            print(f"φ = {phi_val:.2f}, S_rad = {S_rad:.4f}, d(Q3,Q4) = {d_Q34:.4f}")

        print("\nCurvature Estimates:")
        for triplet in combinations(range(self.n_qubits), 3):
            curv = self.estimate_local_curvature(coords3, triplet)
            print(f"Triplet {triplet}: Curvature = {curv:.4f}")

        plt.figure(figsize=(5, 4))
        sns.heatmap(mi_matrix, annot=True, cmap='viridis')
        plt.title(f'MI Matrix at φ={phi_val:.2f}')
        plt.show()

    def fit_rt_plot(self):
        def log_func(x, a, b):
            return a * np.log(x + 1e-6) + b

        phi_vals, entropies, dists = zip(*self.rt_data)
        dists = np.array(dists)
        entropies = np.array(entropies)

        lin_model = LinearRegression()
        lin_model.fit(dists.reshape(-1, 1), entropies)
        lin_pred = lin_model.predict(dists.reshape(-1, 1))

        popt, _ = curve_fit(log_func, dists, entropies)
        log_pred = log_func(dists, *popt)

        plt.figure(figsize=(7, 5))
        plt.scatter(dists, entropies, c='black', label="Data")
        plt.plot(dists, lin_pred, label=f"Linear: {lin_model.coef_[0]:.2f}·d + {lin_model.intercept_:.2f}", linestyle='--')
        plt.plot(dists, log_pred, label=f"Log: {popt[0]:.2f}·log(d) + {popt[1]:.2f}", linestyle='-.')
        plt.xlabel("Distance(Q3, Q4)")
        plt.ylabel("Entropy S(Q3,Q4)")
        plt.title("RT Correlation - 6 Qubits")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

def perfect_tensor_braket():
    """Return a 6-qubit perfect-tensor style circuit using Braket syntax."""
    circ = Circuit()
    
    # Create 3 GHZ pairs: (0,1), (2,3), (4,5)
    for i in [0, 2, 4]:
        circ.h(i)
        circ.cnot(i, i+1)

    def cz_decomposed(q0, q1):
        return Circuit().cnot(q0, q1).rz(q1, np.pi).cnot(q0, q1)

    circ += cz_decomposed(0, 2)
    circ += cz_decomposed(1, 4)
    circ += cz_decomposed(3, 5)

    # Optional RX rotation to break symmetry
    for q in range(6):
        circ.rx(q, np.pi / 4)

    return circ

def marginal_probs(probs, total_qubits, keep):
    marginal = {}
    for idx, p in enumerate(probs):
        b = format(idx, f"0{total_qubits}b")
        key = ''.join([b[i] for i in keep])
        marginal[key] = marginal.get(key, 0) + p
    return np.array(list(marginal.values()))

def entropy(p):
    p = np.array(p)
    p = p / np.sum(p)
    return -np.sum(p * np.log2(p + 1e-12))

def happy_7qubit_braket():
    """Build HaPPY-style 7-qubit circuit in Braket format."""
    circ = perfect_tensor_braket()
    
    # Add 7th qubit (index 6): logical center entangled to one leg (qubit 2)
    circ.cnot(2, 6)
    circ.h(6)

    return circ

class EmergentSpacetime:
    def __init__(self, device_arn):
        self.device = AwsDevice(device_arn)
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

    def construct_4d_embedding(self):
        # Stack MI matrices into a 3D tensor over time
        mi_tensor = np.stack(self.mi_matrices)

        # Invert MI to get distance tensor
        epsilon = 1e-6
        dist_tensor = 1 / (mi_tensor + epsilon)

        # Flatten time into a single dimension (coordinates: qubit pairs x time)
        num_time_steps, num_qubits, _ = dist_tensor.shape
        flat_distances = dist_tensor.reshape(num_time_steps * num_qubits, num_qubits)

        # Perform 4D embedding
        mds = MDS(n_components=4, dissimilarity='euclidean', random_state=42)
        coords4 = mds.fit_transform(flat_distances)
        return coords4.reshape(num_time_steps, num_qubits, 4)

    def estimate_curvature(self, coords4):
        # Approximate 4D curvature numerically by analyzing local volume distortion
        curvatures = []
        for t in range(1, len(coords4)-1):
            prev_coords = coords4[t-1]
            curr_coords = coords4[t]
            next_coords = coords4[t+1]

            prev_dists = np.linalg.norm(curr_coords - prev_coords, axis=1)
            next_dists = np.linalg.norm(curr_coords - next_coords, axis=1)

            local_curvature = np.mean(next_dists - prev_dists)
            curvatures.append(local_curvature)

        return np.array(curvatures)

    def analyze_causal_structure(self, curvatures):
        plt.figure(figsize=(8,5))
        plt.plot(self.timesteps[1:-1], curvatures, marker='o')
        plt.xlabel('Time (φ)')
        plt.ylabel('Approx. Curvature')
        plt.title('Emergent Spacetime Curvature Dynamics')
        plt.grid(True)
        plt.show()

def build_control_circuit(phi):
    circ = Circuit()
    circ.h(0)
    # Only half the entanglement you used before
    circ.cnot(0, 2)
    # Omit or shuffle the second entangling layer
    circ.rx(0, phi)
    circ.probability(target=[2,3])
    return circ


for phi_val in timesteps:
    phi = FreeParameter("phi")
    circ = build_control_circuit(phi)
    rm_circ, mapping = remap_circuit_to_contiguous_qubits(circ)
    task = device.run(rm_circ, inputs={"phi": phi_val}, shots=2048)
    probs = np.array(task.result().values).reshape(-1)
    # Compute entropy on qubits 2 & 3
    p = marginal_probs(probs, 4, [2,3])
    p = p / p.sum()
    entropies_control.append(-(p * np.log2(p + 1e-12)).sum())

# Plot to confirm the area‐law collapse
import matplotlib.pyplot as plt
plt.plot(range(1, len(entropies_control)+1), entropies_control, marker='o')
plt.title("Control: Entropy (flat) vs. φ")
plt.xlabel("Time step")
plt.ylabel("Entropy (bits)")
plt.show()

##scaled_sim = ScaledEmergentSpacetime(device=device, num_qubits=6, max_cut_size=3)
##scaled_sim.run_entropy_area_experiment()
##df_entropy_area = scaled_sim.plot_entropy_vs_area()

##bound = happy_7qubit_braket()
##task = device.run(bound, shots=4096)
##probs = np.array(task.result().values).reshape(-1)
##
##cut_sizes = [1, 2]  # use [2,3] if more qubits available
##entropies = []
##for k in cut_sizes:
##    keep = list(range(2, 2+k))  # qubits 2,3,... as “radiation”
##    marg = marginal_probs(probs, total_qubits=4, keep=keep)
##    entropies.append(entropy(marg))
##
##import matplotlib.pyplot as plt
##plt.plot(cut_sizes, entropies, 'o-')
##plt.xlabel('Radiation cut size (qubits)')
##plt.ylabel('Entropy (bits)')
##plt.title('Entropy vs. Area in Existing Circuit')
##plt.grid(True)
##plt.show()

### Original Injection
##spacetime_sim = EmergentSpacetime(arn)
##
### Run experiments
##spacetime_sim.run()
##
### Construct 4D embedding
##coords4 = spacetime_sim.construct_4d_embedding()
##
### Curvature analysis
##curvature = spacetime_sim.estimate_curvature(coords4)
##
### Causal structure analysis
##spacetime_sim.analyze_causal_structure(curvature)


