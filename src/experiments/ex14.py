# Is the green charge and the other charges have a limit on the info about them that passes through them as a whole
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, entropy, partial_trace
import numpy as np

# Define rotations for different color charges
def apply_color_charge(qc, color_charge):
    if color_charge == "red":
        qc.rx(np.pi / 2, 0)
    elif color_charge == "green":
        qc.ry(np.pi / 3, 0)
    elif color_charge == "blue":
        qc.rz(np.pi / 2, 0)

def calculate_entropy(state, subsystems):
    reduced_state = partial_trace(state, subsystems)
    return entropy(reduced_state)

def test_color_charge(color_charge):
    qc = QuantumCircuit(2)
    apply_color_charge(qc, color_charge)
    qc.h(0)
    qc.cx(0, 1)
    state = Statevector.from_instruction(qc)
    ent_black_hole = calculate_entropy(state, [1])
    ent_radiation = calculate_entropy(state, [0])
    return state, ent_black_hole, ent_radiation

def combined_color_test():
    qc = QuantumCircuit(3)
    apply_color_charge(qc, "red")
    apply_color_charge(qc, "green")
    apply_color_charge(qc, "blue")
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    state = Statevector.from_instruction(qc)
    ent_black_hole = calculate_entropy(state, [1, 2])
    ent_radiation = calculate_entropy(state, [0])
    return state, ent_black_hole, ent_radiation
# Test individual color charges
for color in ["red", "green", "blue"]:
    state, ent_bh, ent_rad = test_color_charge(color)
    print(f"Color Charge: {color}, Black Hole Entropy: {ent_bh:.4f}, Radiation Entropy: {ent_rad:.4f}")

# Test combined color charges
state_combined, ent_comb_bh, ent_comb_rad = combined_color_test()
print(f"Combined Color Charges, Black Hole Entropy: {ent_comb_bh:.4f}, Radiation Entropy: {ent_comb_rad:.4f}")
