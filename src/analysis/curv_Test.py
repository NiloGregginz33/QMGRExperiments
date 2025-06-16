from braket.circuits import Circuit, FreeParameter
from braket.aws import AwsDevice
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from itertools import combinations
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class DynamicAdSGeometry:
    def __init__(self, n_qubits=6, steps=15):
        self.device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")
        self.n_qubits = n_qubits
        self.timesteps = np.linspace(0, 3 * np.pi, steps)
        self.rt_data = []
        self.coords_list_2d = []
        self.coords_list_3d = []
        self.alpha = 0.3  # spatial phase offset per qubit index

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

    def build_dynamic_circuit(self, phi_val, time_index, total_steps):
        phi = FreeParameter("phi")
        circ = Circuit()
        circ.h(0)

        circ.cnot(0, 2)
        circ.cnot(0, 3)

        if time_index >= int(total_steps * 0.25):
            circ.rx(0, phi)
            circ.cz(0, 1)

        if time_index >= int(total_steps * 0.50):
            circ.rx(2, phi)
            circ.cnot(1, 2)
            circ.cz(1, 3)

        if time_index >= int(total_steps * 0.75):
            circ.rx(4, phi)
            circ.cnot(3, 4)
            circ.cnot(4, 5)
            circ.cz(5, 0)

        # New: spatially warped phase injections
        for q in range(self.n_qubits):
            circ.rx(q, phi + self.alpha * q)

        for q in range(self.n_qubits):
            if not any(q in instr.target or q in instr.control for instr in circ.instructions):
                circ.i(q)

        circ.probability()
        return circ, {"phi": phi_val}

    def run(self):
        for idx, phi_val in enumerate(self.timesteps):
            circ, inputs = self.build_dynamic_circuit(phi_val, idx, len(self.timesteps))
            task = self.device.run(circ, inputs=inputs, shots=1024)
            result = task.result()
            probs = np.array(result.values).reshape(-1)

            mi_matrix = np.zeros((self.n_qubits, self.n_qubits))
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    mi = self.compute_mi(probs, i, j, self.n_qubits)
                    mi_matrix[i, j] = mi_matrix[j, i] = mi

            rad_probs = self.marginal_probs(probs, self.n_qubits, [3, 4])
            S_rad = self.shannon_entropy(rad_probs)

            dist = np.exp(-mi_matrix)
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
            curv = self.estimate_local_curvature(self.coords_list_3d[-1], triplet)
            print(f"Triplet {triplet}: Curvature = {curv:.4f}")

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
        plt.title("RT Correlation - Dynamic AdS Circuit (Warped)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

# Example Usage:
ads_warped = DynamicAdSGeometry()
ads_warped.run()
ads_warped.fit_rt_plot()
