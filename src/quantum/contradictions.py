import numpy as np
from braket.circuits import Circuit
from braket.devices import LocalSimulator
import matplotlib.pyplot as plt
from pathlib import Path

class ContradictionsExperiment:
    def __init__(self, device):
        self.device = device
        self.results = []

    def shannon_entropy(self, probs):
        """Calculate Shannon entropy of a probability distribution."""
        probs = np.array(probs)
        probs = probs / np.sum(probs)  # Normalize
        return -np.sum(probs * np.log2(probs + 1e-12))

    def marginal_probs(self, probs, total_qubits, keep):
        """Calculate marginal probabilities for specified qubits."""
        marginal = {}
        for idx, p in enumerate(probs):
            b = format(idx, f"0{total_qubits}b")
            key = ''.join([b[i] for i in keep])
            marginal[key] = marginal.get(key, 0) + p
        return np.array(list(marginal.values()))

    def compute_mi(self, probs, qA, qB, total_qubits):
        """Compute mutual information between two qubits."""
        AB = self.marginal_probs(probs, total_qubits, [qA, qB])
        A = self.marginal_probs(probs, total_qubits, [qA])
        B = self.marginal_probs(probs, total_qubits, [qB])
        return self.shannon_entropy(A) + self.shannon_entropy(B) - self.shannon_entropy(AB)

    def build_circuit(self, test_type):
        """Build the quantum circuit for the specified test."""
        circ = Circuit()
        if test_type == "disconnected":
            # Test 1: Disconnected Bulk
            circ.h(0)
            circ.h(1)
            circ.cnot(0, 2)
            circ.cnot(1, 3)
        elif test_type == "over_entangled":
            # Test 2: Over-entangled Boundary
            circ.h(0)
            circ.cnot(0, 1)
            circ.cnot(0, 2)
            circ.cnot(0, 3)
            circ.cnot(0, 4)
            circ.cnot(0, 5)
        elif test_type == "non_local":
            # Test 3: Non-local Bulk
            circ.h(0)
            circ.h(1)
            circ.cnot(0, 1)
            circ.cnot(0, 2)
            circ.cnot(1, 3)
            circ.cnot(2, 4)
            circ.cnot(3, 5)
        else:  # broken
            # Test 4: Broken Holographic Encoding
            circ.h(0)
            circ.cnot(0, 1)
            circ.cnot(0, 2)
            circ.cnot(1, 3)
        
        circ.probability()
        return circ

    def run(self):
        """Run the contradictions experiment."""
        test_types = ["disconnected", "over_entangled", "non_local", "broken"]
        test_names = ["Disconnected Bulk", "Over-entangled Boundary", 
                     "Non-local Bulk", "Broken Holographic Encoding"]
        
        for test_type, test_name in zip(test_types, test_names):
            print(f"\nTest: {test_name}")
            
            # Build and run circuit
            circ = self.build_circuit(test_type)
            task = self.device.run(circ, shots=1024)
            result = task.result()
            probs = np.array(result.values).reshape(-1)
            
            # Calculate metrics
            entropies = []
            mis = []
            
            # Individual qubit entropies
            print("\nIndividual Qubit Entropies:")
            for i in range(6):
                probs_i = self.marginal_probs(probs, 6, [i])
                entropy = self.shannon_entropy(probs_i)
                entropies.append(entropy)
                print(f"Qubit {i}: {entropy:.4f} bits")
            
            # Pairwise mutual information
            print("\nPairwise Mutual Information:")
            for i in range(6):
                for j in range(i + 1, 6):
                    mi = self.compute_mi(probs, i, j, 6)
                    mis.append(mi)
                    print(f"Qubits {i}-{j}: {mi:.4f} bits")
            
            # Store results
            self.results.append({
                "test_type": test_type,
                "test_name": test_name,
                "entropies": entropies,
                "mis": mis,
                "circuit_depth": len(circ.instructions),
                "success_rate": 1.0  # Local simulator is perfect
            })
        
        # Plot results
        self.plot_results()
        return self.results

    def plot_results(self):
        """Plot the experiment results."""
        plt.figure(figsize=(15, 10))
        
        # Plot individual qubit entropies
        plt.subplot(2, 2, 1)
        for i, result in enumerate(self.results):
            plt.bar(np.arange(6) + i*0.2, result["entropies"], 
                   width=0.2, label=result["test_name"])
        plt.title('Individual Qubit Entropies')
        plt.xlabel('Qubit')
        plt.ylabel('Entropy (bits)')
        plt.legend()
        
        # Plot average mutual information
        plt.subplot(2, 2, 2)
        avg_mis = [np.mean(r["mis"]) for r in self.results]
        plt.bar([r["test_name"] for r in self.results], avg_mis)
        plt.title('Average Mutual Information')
        plt.ylabel('MI (bits)')
        plt.xticks(rotation=45)
        
        # Plot entropy distribution
        plt.subplot(2, 2, 3)
        for result in self.results:
            plt.hist(result["entropies"], alpha=0.5, label=result["test_name"])
        plt.title('Entropy Distribution')
        plt.xlabel('Entropy (bits)')
        plt.ylabel('Count')
        plt.legend()
        
        # Plot mutual information distribution
        plt.subplot(2, 2, 4)
        for result in self.results:
            plt.hist(result["mis"], alpha=0.5, label=result["test_name"])
        plt.title('Mutual Information Distribution')
        plt.xlabel('MI (bits)')
        plt.ylabel('Count')
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_dir = Path("plots")
        plot_dir.mkdir(exist_ok=True)
        plt.savefig(plot_dir / "holographic_contradictions.png")
        plt.close()
        print("\nAnalysis complete. Results saved to holographic_contradictions.png") 