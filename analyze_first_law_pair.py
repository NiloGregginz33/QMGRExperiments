import json
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
import argparse

parser = argparse.ArgumentParser(description="Analyze entanglement first law from paired runs.")
parser.add_argument("unperturbed_json", type=str, help="Unperturbed result JSON")
parser.add_argument("perturbed_json", type=str, help="Perturbed result JSON")
parser.add_argument("--timestep", type=int, default=None, help="Timestep to analyze (default: last)")
parser.add_argument("--max_samples", type=int, default=256, help="Max bipartitions to sample")
args = parser.parse_args()

def load_mi(data, timestep, num_qubits):
    mi_list = data["mutual_information"]
    if timestep is None:
        timestep = len(mi_list) - 1
    mi = mi_list[timestep]
    MI = np.zeros((num_qubits, num_qubits))
    for k, v in mi.items():
        i, j = map(int, k.split('_')[1].split(','))
        MI[i, j] = v
        MI[j, i] = v
    return MI

def all_bipartitions(num_qubits, max_samples=256):
    all_A = []
    for r in range(1, num_qubits//2+1):
        all_A += list(itertools.combinations(range(num_qubits), r))
    if len(all_A) > max_samples:
        all_A = random.sample(all_A, max_samples)
    return all_A

def entropy_for_region(MI, A):
    # Approximate S_A as sum of MI for all pairs in A
    return sum(MI[i, j] for i in A for j in A if i < j)

def main():
    with open(args.unperturbed_json) as f:
        data0 = json.load(f)
    with open(args.perturbed_json) as f:
        data1 = json.load(f)
    num_qubits = data0["spec"]["num_qubits"]
    timestep = args.timestep if args.timestep is not None else len(data0["mutual_information"]) - 1
    MI0 = load_mi(data0, timestep, num_qubits)
    MI1 = load_mi(data1, timestep, num_qubits)
    biparts = all_bipartitions(num_qubits, args.max_samples)
    delta_S = []
    region_sizes = []
    for A in biparts:
        S0 = entropy_for_region(MI0, A)
        S1 = entropy_for_region(MI1, A)
        delta_S.append(S1 - S0)
        region_sizes.append(len(A))
    delta_S = np.array(delta_S)
    region_sizes = np.array(region_sizes)
    # Plot delta_S vs region size
    plt.figure()
    plt.scatter(region_sizes, delta_S, alpha=0.5)
    plt.xlabel("Region size |A|")
    plt.ylabel("Delta S_A (perturbed - unperturbed)")
    plt.title(f"First Law: Entropy Change by Region Size (timestep {timestep})")
    plt.savefig("delta_S_vs_region_size.png")
    plt.show()
    # Histogram
    plt.figure()
    plt.hist(delta_S, bins=30, alpha=0.7)
    plt.xlabel("Delta S_A")
    plt.ylabel("Count")
    plt.title("Distribution of Entropy Changes (First Law)")
    plt.savefig("delta_S_histogram.png")
    plt.show()
    # Print summary
    print(f"Delta S_A: mean={np.mean(delta_S):.4f}, std={np.std(delta_S):.4f}, max={np.max(delta_S):.4f}, min={np.min(delta_S):.4f}")

if __name__ == "__main__":
    main() 