import numpy as np
import matplotlib.pyplot as plt
from braket.circuits import Circuit, FreeParameter, Gate
from braket.devices import LocalSimulator
from sklearn.manifold import MDS
from itertools import combinations
import json
import os
from datetime import datetime
import pandas as pd

def remap_circuit_to_contiguous_qubits(circuit):
    """Remaps a circuit to use contiguous qubits if non-contiguous are used.
    This is necessary for some Braket local simulators.
    Returns the remapped circuit and a dictionary of old_qubit_idx -> new_qubit_idx.
    """
    used_qubits = sorted(list(circuit.qubits))
    if not used_qubits or (used_qubits[-1] - used_qubits[0] + 1) == len(used_qubits):
        return circuit, {q: q for q in used_qubits} # Already contiguous or empty

    # Create a mapping to contiguous qubits starting from 0
    mapping = {old_q: new_q for new_q, old_q in enumerate(used_qubits)}
    new_num_qubits = len(used_qubits)
    new_circuit = Circuit().set_ir_type(circuit.ir_type)

    # Iterate through instructions and rebuild the circuit with remapped qubits
    for instruction in circuit.instructions:
        # Remap target qubits
        new_target_qubits = [mapping[q] for q in instruction.target]

        # Remap control qubits if they exist
        new_control_qubits = [mapping[q] for q in instruction.control] if instruction.control else []

        # Create new instruction. Special handling for some gate types might be needed
        if isinstance(instruction.operator, Gate):
            new_circuit.add_instruction(instruction.operator(new_target_qubits, new_control_qubits))
        else:
            # For other types of instructions (e.g., measurements, channels), 
            # we might need to handle them based on their specific properties.
            # For simplicity, assuming most operations are simple gates for this context.
            # If it's a Probability instruction, it doesn't have target/control. Handle carefully.
            if instruction.operator.name == 'Probability':
                new_circuit.probability()
            else:
                # This path might need more specific handling based on the instruction type.
                # For this experiment, we assume simple gate operations.
                pass # Fallback for unhandled instruction types
    
    # Re-add probability instruction if it was present
    if hasattr(circuit, '_ir_type') and circuit.ir_type == 'OPENQASM' and circuit.instructions[-1].operator.name == 'Probability':
        new_circuit.probability() # Re-add if it was the last instruction

    return new_circuit, mapping


class LocalEmergentSpacetime:
    def __init__(self, n_qubits=4, shots=1024):
        self.device = LocalSimulator()
        self.n_qubits = n_qubits
        self.shots = shots
        self.timesteps = np.linspace(0, 3 * np.pi, 15)
        self.mi_matrices = []
        self.exp_dir = f"experiment_logs/charge_injection_spacetime_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.exp_dir, exist_ok=True)

    def shannon_entropy(self, probs):
        probs = np.array(probs)
        # Ensure probabilities sum to 1 to avoid log(0) issues
        probs = probs / np.sum(probs) if np.sum(probs) > 0 else probs
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

    def run_charge_injection_experiment(self):
        print("\nRunning Charge Injection Spacetime Experiment...")
        all_results = {
            "phi": [],
            "entropies": [],
            "curvatures": [],
            "distances": [],
            "mi_matrices": []
        }

        for phi_val in self.timesteps:
            circ = Circuit()
            circ.h(0)
            circ.cnot(0, 2)
            circ.cnot(0, 3)
            # circ.rx(0, phi_val) # Removed as it's part of the new sequence

            # --- Simulate Charge Injection with CNOT pairs ---
            # Add a CNOT gate between qubit 0 and 1
            # circ.cnot(0, 1) # Removed
            # Add another CNOT gate between qubit 2 and 3
            # circ.cnot(2, 3) # Removed
            # --- End Charge Injection ---

            # Replaced with the new charge injection sequence
            circ.rx(0, phi_val)         # Inject charge/information into BH core
            circ.cz(0, 1)           # Phase coupling to auxiliary mode
            circ.cnot(1, 2)         # Radiative coupling
            circ.rx(2, phi_val)         # Inject into radiation
            circ.cz(1, 3)

            # cz(0, 1) equivalent (original circuit element) # Redundant due to new sequence
            # circ.cnot(0, 1).rz(1, np.pi).cnot(0, 1) # Removed

            # circ.cnot(1, 2) # Redundant due to new sequence
            # circ.rx(2, phi_val) # Redundant due to new sequence

            # cz(1, 3) equivalent (original circuit element) # Redundant due to new sequence
            # circ.cnot(1, 3).rz(3, np.pi).cnot(1, 3) # Removed
            circ.probability()

            rm_circ, mapping = remap_circuit_to_contiguous_qubits(circ)
            task = self.device.run(rm_circ, shots=self.shots)
            result = task.result()
            probs = np.array(result.values).reshape(-1)

            mi_matrix = np.zeros((self.n_qubits, self.n_qubits))
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    mi = self.compute_mi(probs, i, j, self.n_qubits)
                    mi_matrix[i, j] = mi_matrix[j, i] = mi
            self.mi_matrices.append(mi_matrix)
            
            # Calculate average distance for current timestep
            # current_avg_dist = np.mean(mi_matrix[mi_matrix > 0]) # Removed

            all_results["phi"].append(float(phi_val))
            all_results["entropies"].append(float(self.shannon_entropy(probs)))
            # all_results["curvatures"].append(float(current_curvature)) # Removed
            # all_results["distances"].append(float(current_avg_dist)) # Removed
            all_results["mi_matrices"].append(mi_matrix.tolist())

            # print(f"φ={phi_val:.2f}, S={all_results['entropies'][-1]:.4f}, K={all_results['curvatures'][-1]:.4f}, d={all_results['distances'][-1]:.4f}") # Modified
            print(f"φ={phi_val:.2f}, S={all_results['entropies'][-1]:.4f}")

        # Compute emergent geometry and curvature after all MI matrices are collected
        epsilon = 1e-6
        dist_tensor = 1 / (np.array(self.mi_matrices) + epsilon)
        
        # Flatten time into a single dimension (coordinates: qubit pairs x time)
        num_time_steps, num_qubits, _ = dist_tensor.shape
        flat_distances = dist_tensor.reshape(num_time_steps * num_qubits, num_qubits)

        # Perform 4D embedding
        mds = MDS(n_components=3, dissimilarity='euclidean', random_state=42)
        coords4 = mds.fit_transform(flat_distances)
        coords4 = coords4.reshape(num_time_steps, num_qubits, 3)

        # Estimate curvature using the AWSFactory method (volume distortion)
        curvatures = []
        # The original AWSFactory.EmergentSpacetime.estimate_curvature was calculating a single array
        # Here we adapt it to calculate for each timestep
        for t in range(num_time_steps):
            # Ensure we have enough timesteps to calculate local curvature based on neighbors
            if t == 0 or t == num_time_steps - 1: # No previous or next state
                curvatures.append(0.0) # Assign 0 curvature for boundary conditions
            else:
                prev_coords = coords4[t-1]
                curr_coords = coords4[t]
                next_coords = coords4[t+1]

                prev_dists = np.linalg.norm(curr_coords - prev_coords, axis=1)
                next_dists = np.linalg.norm(curr_coords - next_coords, axis=1)

                local_curvature = np.mean(next_dists - prev_dists)
                curvatures.append(float(local_curvature))

        print(f"DEBUG: len(curvatures) = {len(curvatures)}") # Debug print
        all_results["curvatures"] = curvatures

        # Calculate average distances for each timestep
        distances = []
        for mi_matrix in self.mi_matrices:
            distances.append(float(np.mean(mi_matrix[mi_matrix > 0])))

        print(f"DEBUG: len(distances) = {len(distances)}") # Debug print
        all_results["distances"] = distances

        # Save and plot results
        with open(f"{self.exp_dir}/charge_injection_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        self._plot_results(all_results)

        # Theoretical analysis (simplified for this run)
        analysis = {
            "holographic_implications": [],
            "string_theory_implications": []
        }

        avg_curvature = np.mean(curvatures)
        if avg_curvature < 0:
            analysis["holographic_implications"].append("Average negative curvature suggests AdS-like spacetime.")
        elif avg_curvature > 0:
            analysis["holographic_implications"].append("Average positive curvature suggests de Sitter-like spacetime.")
        else:
            analysis["holographic_implications"].append("Average curvature near zero suggests flat spacetime.")

        entropy_curv_corr = np.corrcoef(all_results["entropies"], all_results["curvatures"])[0, 1]
        if abs(entropy_curv_corr) > 0.5:
            analysis["string_theory_implications"].append(
                f"Strong correlation (R={entropy_curv_corr:.2f}) between entropy and curvature suggests connection to string dynamics."
            )
        else:
            analysis["string_theory_implications"].append(
                f"Weak correlation (R={entropy_curv_corr:.2f}) between entropy and curvature."
            )

        with open(f"{self.exp_dir}/theoretical_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)

        print("\nTheoretical Analysis Summary:")
        for k, v in analysis.items():
            print(f"{k.replace('_', ' ').title()}:")
            for item in v:
                print(f"- {item}")

        return all_results

    def _plot_results(self, results):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot Entropy
        axes[0].plot(results["phi"], results["entropies"], 'b-o')
        axes[0].set_xlabel('Phase (φ)')
        axes[0].set_ylabel('Entropy')
        axes[0].set_title('Entropy Evolution')
        axes[0].grid(True)

        # Plot Curvature
        axes[1].plot(results["phi"], results["curvatures"], 'r-o')
        axes[1].set_xlabel('Phase (φ)')
        axes[1].set_ylabel('Curvature')
        axes[1].set_title('Curvature Evolution')
        axes[1].grid(True)

        # Plot Average Distance
        axes[2].plot(results["phi"], results["distances"], 'g-o')
        axes[2].set_xlabel('Phase (φ)')
        axes[2].set_ylabel('Average Distance')
        axes[2].set_title('Average Distance Evolution')
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.exp_dir}/summary_plots.png")
        plt.close()

if __name__ == "__main__":
    experiment = LocalEmergentSpacetime()
    experiment.run_charge_injection_experiment() 