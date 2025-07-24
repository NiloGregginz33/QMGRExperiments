import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the experiment results
result_file = "experiment_logs/custom_curvature_experiment/results_n7_geomH_curv3_ibm_brisbane_Y4CW0U.json"

with open(result_file, 'r') as f:
    data = json.load(f)

# Extract mutual information data (use the last timestep)
mi_data = data.get('mutual_information_per_timestep', [])
if mi_data:
    # Use the last timestep for final MI values
    final_mi = mi_data[-1]
    print(f"Found MI data with {len(final_mi)} entries")
    print(f"Sample MI keys: {list(final_mi.keys())[:5]}")
else:
    print("No mutual information data found")
    exit()

# Extract edge lengths from stationary solution
edge_lengths = data.get('stationary_solution', {}).get('regge_evolution_data', {}).get('regge_edge_lengths_per_timestep', [])
if edge_lengths:
    # Use the last timestep for final edge lengths
    final_edge_lengths = edge_lengths[-1]
    print(f"Found {len(final_edge_lengths)} edge lengths")
else:
    print("No edge length data found")
    exit()

# Create edge pairs for 7 qubits (21 edges)
n_qubits = 7
edge_pairs = []
for i in range(n_qubits):
    for j in range(i+1, n_qubits):
        edge_pairs.append((i, j))

print(f"Created {len(edge_pairs)} edge pairs")

# Extract MI values for each edge
mi_values = []
edge_length_values = []

for i, (q1, q2) in enumerate(edge_pairs):
    edge_key = f"{q1},{q2}"
    mi_key = f"I_{q1},{q2}"  # Add I_ prefix to match the actual keys
    if mi_key in final_mi:
        mi_val = final_mi[mi_key]
        edge_len = final_edge_lengths[i] if i < len(final_edge_lengths) else 0.001
        
        mi_values.append(mi_val)
        edge_length_values.append(edge_len)
        print(f"Edge {edge_key}: MI={mi_val:.3f}, Length={edge_len:.6f}")

print(f"Extracted {len(mi_values)} data points")

# Create scatter plot
plt.figure(figsize=(12, 8))

# Add note about MI values
if all(mi == 0.1 for mi in mi_values):
    plt.suptitle('MI vs Final Edge Length for 7-Qubit Hyperbolic Geometry (κ=3.0)\nNote: All MI values are 0.1 (quantum measurements may not have worked properly)', fontsize=14)
else:
    plt.title('MI vs Final Edge Length for 7-Qubit Hyperbolic Geometry (κ=3.0)', fontsize=16)

# Separate points on floor vs above floor
floor_threshold = 0.002  # Slightly above the 0.001 floor
floor_indices = [i for i, el in enumerate(edge_length_values) if el <= floor_threshold]
above_floor_indices = [i for i, el in enumerate(edge_length_values) if el > floor_threshold]

print(f"Floor indices: {floor_indices}")
print(f"Above floor indices: {above_floor_indices}")

# Plot points on floor
if floor_indices:
    floor_mi = [mi_values[i] for i in floor_indices]
    floor_edges = [edge_length_values[i] for i in floor_indices]
    plt.scatter(floor_edges, floor_mi, color='red', alpha=0.7, s=100, label=f'On floor (≤{floor_threshold})')

# Plot points above floor
if above_floor_indices:
    above_mi = [mi_values[i] for i in above_floor_indices]
    above_edges = [edge_length_values[i] for i in above_floor_indices]
    plt.scatter(above_edges, above_mi, color='blue', alpha=0.7, s=100, label=f'Above floor (>={floor_threshold})')

plt.xlabel('Final Edge Length (from Lorentzian solver)', fontsize=14)
plt.ylabel('Mutual Information (from quantum measurement)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Add statistics
total_edges = len(mi_values)
floor_count = len(floor_indices)
above_count = len(above_floor_indices)

if floor_count > 0:
    avg_mi_floor = np.mean([mi_values[i] for i in floor_indices])
    plt.figtext(0.02, 0.02, f'Floor edges: {floor_count}/{total_edges} (avg MI: {avg_mi_floor:.3f})', 
                fontsize=10, style='italic', color='red')

if above_count > 0:
    avg_mi_above = np.mean([mi_values[i] for i in above_floor_indices])
    plt.figtext(0.02, 0.05, f'Above floor: {above_count}/{total_edges} (avg MI: {avg_mi_above:.3f})', 
                fontsize=10, style='italic', color='blue')

# Add footnote about floor clustering
if floor_count > 0:
    plt.figtext(0.02, -0.05, 
                f'Note: {floor_count} edges are at the floor value (≤{floor_threshold}), suggesting the Lorentzian solver\n'
                f'is hitting the minimum edge length constraint. These edges show lower MI values on average.',
                fontsize=9, style='italic', wrap=True)

plt.tight_layout()
plt.savefig('mi_vs_edge_length_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Scatter plot saved as 'mi_vs_edge_length_scatter.png'")
print(f"Total edges: {total_edges}")
print(f"Edges on floor: {floor_count}")
print(f"Edges above floor: {above_count}")
if floor_count > 0:
    print(f"Average MI for floor edges: {avg_mi_floor:.3f}")
if above_count > 0:
    print(f"Average MI for above-floor edges: {avg_mi_above:.3f}") 