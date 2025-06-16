from qiskit import QuantumCircuit, QuantumRegister, pulse, transpile, ClassicalRegister, assemble
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix, state_fidelity, random_unitary, Operator, mutual_information
from qiskit_aer import Aer, AerSimulator
from qiskit.quantum_info import entropy as qiskit_entropy
from qiskit.pulse import Play, DriveChannel, Gaussian, Schedule
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, amplitude_damping_error
from collections import Counter
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from scipy.spatial.distance import jensenshannon
from qiskit.result import Result, Counts  # Import for local simulation results
from qiskit.circuit.library import QFT, RZGate, MCXGate, ZGate, XGate, HGate
import random
from scipy.linalg import expm
import hashlib
from qiskit.circuit import Instruction
import time
from datetime import datetime
from scipy.stats import binom_test
import requests
from qutip import *
import sys
import psutil
import itertools
import csv
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import inspect
from collections import Counter
from qiskit_ibm_provider import IBMProvider
from scipy.optimize import minimize
import os

# Character to 5-bit binary encoding (ASCII-based subset for A-Z)
char_to_bit_map = {
    'A': ('0', '0', '0', '0', '0'),
    'B': ('0', '0', '0', '0', '1'),
    'C': ('0', '0', '0', '1', '0'),
    'D': ('0', '0', '0', '1', '1'),
    'E': ('0', '0', '1', '0', '0'),
    'F': ('0', '0', '1', '0', '1'),
    'G': ('0', '0', '1', '1', '0'),
    'H': ('0', '0', '1', '1', '1'),
    'I': ('0', '1', '0', '0', '0'),
    'J': ('0', '1', '0', '0', '1'),
    'K': ('0', '1', '0', '1', '0'),
    'L': ('0', '1', '0', '1', '1'),
    'M': ('0', '1', '1', '0', '0'),
    'N': ('0', '1', '1', '0', '1'),
    'O': ('0', '1', '1', '1', '0'),
    'P': ('0', '1', '1', '1', '1'),
    'Q': ('1', '0', '0', '0', '0'),
    'R': ('1', '0', '0', '0', '1'),
    'S': ('1', '0', '0', '1', '0'),
    'T': ('1', '0', '0', '1', '1'),
    'U': ('1', '0', '1', '0', '0'),
    'V': ('1', '0', '1', '0', '1'),
    'W': ('1', '0', '1', '1', '0'),
    'X': ('1', '0', '1', '1', '1'),
    'Y': ('1', '1', '0', '0', '0'),
    'Z': ('1', '1', '0', '0', '1'),
    '.': ('1', '1', '0', '1', '0'),
    ' ': ('1', '1', '0', '1', '1'),
    ':)': ('1', '1', '1', '0', '0'),
    '?': ('1', '1', '1', '0', '1'),
    '^^': ('1', '1', '1', '1', '1'),
}

# Reverse map for decoding
bit_to_char_map = {v: k for k, v in char_to_bit_map.items()}

def find_angles():
    target_entropy = 0.4
    angles = find_angles_from_entropy(target_entropy)

    if angles:
        θ_s, θ_d, φ = angles
        print(f"🧠 For target entropy {target_entropy:.3f}, use:")
        print(f"  θ_static  = {θ_s:.4f}")
        print(f"  θ_dynamic = {θ_d:.4f}")
        print(f"  φ         = {φ:.4f}")

def target_entropy(target = 0.75):
    # Desired entropy
    qc, measured = reverse_entropy_oracle(target, n_qubits=2)
    print("Final entropy:", measured)
    qc.draw('mpl')

def find_angles_from_entropy(S_target: float) -> tuple:
    C = np.log(2) * np.pi      # ≈ 2.17
    B = 247.2744               # Your tuned exponential coefficient

    # Assume θ_static = θ_dynamic = π/4 → sin²(θ) = 0.5
    angle_factor = 0.5 * 0.5
    base_strength = S_target / (C * angle_factor)

    if base_strength >= 1:
        raise ValueError("Target entropy too high for current assumptions")

    try:
        phi = np.sqrt(-np.log(1 - base_strength) / B)
        theta_static = np.pi / 4
        theta_dynamic = np.pi / 4
        return theta_static, theta_dynamic, phi
    except Exception as e:
        print("Failed to compute angles:", e)
        return None

def find_angles_for_target_entropy(target_entropy, A, B):
    from scipy.optimize import minimize
    import numpy as np

    def model(params):
        θs, θd, φ = params
        return A * np.sin(θs)**2 * np.sin(θd)**2 * (1 - np.exp(-B * φ**2))

    def loss(params):
        return (model(params) - target_entropy)**2

    initial_guess = [np.pi/4, np.pi/4, 1.0]
    bounds = [(0, np.pi), (0, np.pi), (0.01, 5.0)]
    
    result = minimize(loss, initial_guess, bounds=bounds)
    θs_opt, θd_opt, φ_opt = result.x
    return θs_opt, θd_opt, φ_opt

def reverse_entropy_oracle(target_entropy: float, n_qubits: int = 1, attempts: int = 100):
    """
    Try to generate a quantum circuit whose final state on Q0 has the given entropy.
    Returns the best-approximating circuit and its measured entropy.
    """
    
    backend = Aer.get_backend('statevector_simulator')
    best_qc = None
    best_entropy = -1
    min_diff = float('inf')

    for _ in range(attempts):
        qc = QuantumCircuit(n_qubits)
        
        # Apply random rotations
        for q in range(n_qubits):
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            lam = np.random.uniform(0, 2*np.pi)
            qc.u(theta, phi, lam, q)
        
        # Optional: Entangle with ancilla qubit (for multi-qubit)
        if n_qubits > 1:
            for i in range(n_qubits - 1):
                qc.cx(i, i+1)

        # Simulate
        sv = Statevector.from_instruction(qc)
        reduced = partial_trace(sv, list(range(1, n_qubits)))  # Trace out all except Q0
        ent = qiskit_entropy(reduced)

        diff = abs(ent - target_entropy)
        if diff < min_diff:
            min_diff = diff
            best_entropy = ent
            best_qc = qc
            if diff < 1e-3:
                break

    return best_qc, best_entropy

##def reverse_subsystem_entropy_oracle(target_entropy: float, 
##                                     n_total_qubits: int = 2, 
##                                     trace_out: list = [1], 
##                                     attempts: int = 100):
##    """
##    Try to generate a quantum circuit whose reduced state (after tracing out `trace_out`)
##    has entropy close to `target_entropy`.
##
##    Args:
##        target_entropy (float): Desired entropy of the subsystem.
##        n_total_qubits (int): Total number of qubits in the circuit.
##        trace_out (list): Qubit indices to trace out. Remaining qubits define the subsystem.
##        attempts (int): Number of random circuits to try.
##
##    Returns:
##        best_qc (QuantumCircuit): Circuit approximating the desired entropy.
##        best_entropy (float): Achieved entropy.
##        min_diff (float): Absolute difference from target.
##    """
##    backend = Aer.get_backend('statevector_simulator')
##    min_diff = float('inf')
##    best_entropy = None
##    best_qc = None
##
##    for _ in range(attempts):
##        qc = QuantumCircuit(n_total_qubits)
##
##        # Apply random U gates to all qubits
##        for q in range(n_total_qubits):
##            qc.u(np.random.uniform(0, 2*np.pi),
##                 np.random.uniform(0, 2*np.pi),
##                 np.random.uniform(0, 2*np.pi),
##                 q)
##
##        # Add random entangling CXs
##        for i in range(n_total_qubits - 1):
##            if np.random.rand() < 0.7:  # 70% chance to add entanglement
##                qc.cx(i, i+1)
##
##        # Get full statevector
##        sv = Statevector.from_instruction(qc)
##
##        # Trace out specified qubits
##        keep_qubits = [i for i in range(n_total_qubits) if i not in trace_out]
##        reduced = partial_trace(sv, trace_out)
##        ent = qiskit_entropy(reduced)
##
##        diff = abs(ent - target_entropy)
##        if diff < min_diff:
##            min_diff = diff
##            best_entropy = ent
##            best_qc = qc
##            if diff < 1e-3:
##                break
##
##    return best_qc, best_entropy, min_diff

def time_reversal_simulation(qc):
    """
    Modifies the circuit to simulate time reversal by inverting gates.
    Handles the fact that measurement operations cannot be inverted.
    """
    # Create a copy of the circuit without measurements
    qc_no_measure = qc.remove_final_measurements(inplace=False)

    # Invert the circuit
    reversed_qc = qc_no_measure.inverse()

    # Add measurement gates back to the inverted circuit
    reversed_qc.measure_all()

    return reversed_qc

def entropy_matching_oracle(target_entropies, x, y, z, attempts=100):
    """
    Attempts to generate a circuit whose qubits match target subsystem entropies.
    
    Args:
        target_entropies (List[float]): target entropy for each qubit [q0, q1, q2...]
        attempts (int): number of random trials to optimize.
    
    Returns:
        dict: with keys 'circuit', 'entropies', and 'counts'.
    """
    n_qubits = len(target_entropies)
    backend_sv = Aer.get_backend('statevector_simulator')
    backend_qasm = Aer.get_backend('qasm_simulator')
    
    best_qc = None
    best_entropies = None
    min_diff = float('inf')
    qc = None
    for _ in range(attempts):
        qc = QuantumCircuit(n_qubits)

        results = modify_and_run_quantum_experiment_multi_analysis(qc)
        print("Results: ", results)
        # Random single-qubit unitaries
        for q in range(n_qubits):
            base = np.pi  # or use 1.0 if you prefer
            epsilon = np.random.uniform(-0.1, 0.1)
            theta = base * (q + 1) / x + epsilon
            phi   = base * (q + 1) / y + epsilon
            lam   = base * (q + 1) / z + epsilon

            qc.u(theta, phi, lam, q)

        # Add some entanglement
        for i in range(n_qubits - 1):
            qc.cx(i, i+1)

        sv = Statevector.from_instruction(qc)
        current_entropies = []
        for i in range(n_qubits):
            reduced = partial_trace(sv, [q for q in range(n_qubits) if q != i])
            current_entropies.append(qiskit_entropy(reduced))

        diff = sum(abs(te - ce) for te, ce in zip(target_entropies, current_entropies))

        if diff < min_diff:
            min_diff = diff
            best_qc = qc
            best_entropies = current_entropies
            if diff < 1e-3:
                break

    print("Best match entropies:", [round(e, 6) for e in best_entropies])
    print(best_qc)

    # Simulate counts
    qc.measure_all()
    job = backend_sv.run(qc, shots=1024)
    counts = job.result().get_counts()
    
    print("Measurement counts:")
    for bitstring, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {bitstring}: {count}")

    return {
        "circuit": best_qc,
        "entropies": best_entropies,
        "counts": counts
    }

def shannon_entropy(probs):
    return -sum(p * np.log2(p) for p in probs if p > 0)

def get_subsystem_counts(counts, qubit_indices):
    """
    Collapse measurement results onto selected qubit indices (e.g., [0] for Q0).
    """
    collapsed = Counter()
    for bitstring, freq in counts.items():
        selected_bits = ''.join([bitstring[::-1][i] for i in qubit_indices])  # reverse for Qiskit's order
        collapsed[selected_bits] += freq
    return collapsed

def compute_subsystem_entropy(counts, qubit_indices):
    """
    Compute Shannon entropy of the subsystem defined by qubit_indices.
    """
    marginal_counts = get_subsystem_counts(counts, qubit_indices)
    total = sum(marginal_counts.values())
    probs = [count / total for count in marginal_counts.values()]
    return shannon_entropy(probs)

def amplify_external_state(target_state_str='111', num_branches=3, shots=8192, amplify_rounds=1):
    backend = Aer.get_backend('aer_simulator')

    # Define quantum registers
    qr_internal = QuantumRegister(num_branches, name="q")
    num_bits = len(target_state_str)
    qr_external = QuantumRegister(num_bits, 'qr_ext')

    cr = ClassicalRegister(num_branches, name="c")
    qc = QuantumCircuit(qr_internal, qr_external, cr)

    # Step 1: Initialize external register to superposition
    qc.h(qr_external)

    # Step 2: Entangle external and internal systems
    for i in range(num_branches):
        qc.cx(qr_external[i], qr_internal[i])

    # Step 3: Controlled charge injection from external to internal
    for i in range(num_branches):
        qc.crx(np.pi / 3, qr_external[i], qr_internal[i])

    # Step 4: Grover-style amplification of target external state
    for _ in range(amplify_rounds):
        # Oracle for the target state (e.g., '111')
        for i, bit in enumerate(reversed(target_state_str)):
            if bit == '0':
                qc.x(qr_external[i])
        qc.h(qr_external[-1])
        qc.mcx(qr_external[:-1], qr_external[-1])
        qc.h(qr_external[-1])
        for i, bit in enumerate(reversed(target_state_str)):
            if bit == '0':
                qc.x(qr_external[i])

        # Diffusion operator
        qc.h(qr_external)
        qc.x(qr_external)
        qc.h(qr_external[-1])
        qc.mcx(qr_external[:-1], qr_external[-1])
        qc.h(qr_external[-1])
        qc.x(qr_external)
        qc.h(qr_external)

    # Step 5: Measure internal system only
    qc.measure(qr_internal, cr)

    # Step 6: Simulate and get full statevector
    transpiled = transpile(qc, backend)
    result = backend.run(transpiled, shots=shots).result()
    counts = result.get_counts()
    print("\n📞 Internal Branch Selection:", counts)

    # Remove measurement gates for post-statevector analysis
    qc.data = [gate for gate in qc.data if gate[0].name != 'measure']
    full_state = Statevector.from_instruction(qc)
    full_density = DensityMatrix(full_state)

    # Trace out internal system to view amplified external state
    external_dm = partial_trace(full_density, [i for i in range(num_branches)])
    probs = np.real(np.diag(external_dm.data))

    print("\n🎯 External State Probabilities:")
    for i, p in enumerate(probs):
        b = format(i, f'0{num_branches}b')
        print(f"State |{b}>: {round(p * 100, 2)}%")

    return external_dm, probs, counts

charge_history = {'Positive': np.zeros(3), 'Neutral': np.zeros(3)}

def multiversal_telephone_v3(num_branches=3, backend=None, shots=8192):
    if backend is None:
        backend = AerSimulator(method="statevector")

    # Step 1: Generate Registers
    qr = QuantumRegister(num_branches, name="q")        # Internal state
    qr_ext = QuantumRegister(num_branches, name="ext")  # External target
    cr = ClassicalRegister(num_branches, name="c")
    qc = QuantumCircuit(qr, qr_ext, cr)

    # Step 2: Encode symbolic external openness (e.g., π/4 rotation as "partial openness")
    for i in range(num_branches):
        qc.ry(np.pi/4, qr_ext[i])  # Symbolic encoding of external state

    # Step 3: Entangle internal and external qubits
    for i in range(num_branches):
        qc.cx(qr_ext[i], qr[i])  # External state controls internal response

    # Step 4: Controlled charge injection (bias) based on external qubit state
    for i in range(num_branches):
        charge = (np.pi / 3)  # Could be dynamically defined
        qc.crx(charge, qr_ext[i], qr[i])  # Only inject if external state allows

    # Step 5: Entanglement among internal branches (optional bias coherence)
    for i in range(num_branches - 1):
        qc.cx(qr[i], qr[i+1])

    # Step 6: Measurement of internal system only
    qc.measure(qr, cr)

    # Step 7: Run circuit
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()
    selected_branch = max(counts, key=counts.get)

    print(f"\n📞 Selected Reality Branch: {selected_branch}")
    print("Measurement Counts:", counts)

    # Step 8: Remove measurements to analyze quantum state
    qc.data = [gate for gate in qc.data if gate[0].name != 'measure']
    full_state = Statevector.from_instruction(qc)
    full_density = DensityMatrix(full_state)

    # Step 9: Extract external state (partial trace over internal system)
    external_density_matrix = partial_trace(full_density, [i for i in range(num_branches)])

    print("\n🎯 External Reality Influence State:")
    print(external_density_matrix)

    return selected_branch, external_density_matrix, counts

def multiversal_reply_test(initial_message='HI', direct_reply_map=None, shots=4096):
    if direct_reply_map is None:
        direct_reply_map = {'HI': 'HELLO'}

    charge_histories = {'Rad1': np.zeros(3), 'Rad2': np.zeros(3)}
    decoded_messages = {'Rad1': '', 'Rad2': ''}

    # --- Rad1 sends message ---
    rad1_chars = list(initial_message)
    rad1_outputs = []
    print("\n=== RAD1 OUTPUT ===")
    for char in rad1_chars:
        target_bits = char_to_bit_map[char]
        charge_histories['Rad1'], counts = amplify_target_state(target_bits, charge_histories['Rad1'], shots=shots)
        dominant_state = max(counts, key=counts.get)
        decoded_char = bit_to_char_map.get(tuple(dominant_state[::-1]), '?')
        rad1_outputs.append(decoded_char)
        print(f"Character '{char}' Output: {counts}, Dominant: {dominant_state}, Decoded as: {decoded_char}")

    rad1_message_decoded = ''.join(rad1_outputs)
    charge_histories['Rad2'] = charge_histories['Rad1'].copy()

    # --- Rad2 replies according to the reply map ---
    reply_message = direct_reply_map.get(initial_message, '???')
    rad2_chars = list(reply_message)  # Full length reply now
    rad2_outputs = []
    print("\n=== RAD2 REPLY ===")
    for char in rad2_chars:
        target_bits = char_to_bit_map.get(char, ('0','0','0'))
        # Reset charge history for each character to prevent bleed-over
        charge_history_rad2 = np.zeros(3)
        charge_history_rad2, counts = amplify_target_state(target_bits, charge_history_rad2, shots=shots)
        dominant_state = max(counts, key=counts.get)
        decoded_char = bit_to_char_map.get(tuple(dominant_state[::-1]), '?')
        rad2_outputs.append(decoded_char)
        print(f"Character '{char}' Reply Output: {counts}, Dominant: {dominant_state}, Decoded as: {decoded_char}")

    rad2_message_decoded = ''.join(rad2_outputs)

    # --- Summary ---
    print("\n=== SUMMARY ===")
    print(f"Rad1 Sent: {initial_message} -> Decoded: {rad1_message_decoded}")
    print(f"Rad2 Replied: {reply_message} -> Decoded: {rad2_message_decoded}")

        # --- Manual decode and accuracy analysis ---
    print("\n=== MANUAL DECODING ANALYSIS ===")
    manual_decoder(rad1_message_decoded, initial_message)
    manual_decoder(rad2_message_decoded, reply_message)

    # --- Success check ---
    success = rad2_message_decoded == reply_message
    print(f"\n✅ Success: {success}")

    # --- Success check ---
    success = rad2_message_decoded == reply_message
    print(f"\n✅ Success: {success}")

    return {
        'Rad1_message_decoded': rad1_message_decoded,
        'Rad2_message_decoded': rad2_message_decoded,
        'Intended_reply': reply_message,
        'Success': success
    }

def send_entropy_message(message='HI', target_entropy=0.75, num_branches=3, shots=8192):
    print(f"📡 Sending message '{message}' with entropy targeting at {target_entropy}")
    
    # Use entropy targeting to bias the universe
    qc, final_entropy = reverse_entropy_oracle(target_entropy, n_qubits=num_branches)
    print(f"🌀 Entropy-biased circuit created with entropy ≈ {round(final_entropy, 4)}")
    
    # Apply amplification and symbolic message encoding
    char_states = []
    for char in message:
        bits = char_to_bit_map.get(char.upper(), ('0','0','0'))
        print(f"🔠 Encoding char '{char}': {bits}")
        _, _, counts = amplify_external_state(target_state_str=''.join(bits), shots=shots)
        char_states.append(counts)

    return qc, final_entropy, char_states

def receive_entropy_message(bits_per_char=5, shots=8192):
    backend = Aer.get_backend("qasm_simulator")
    qc, qr, cr = create_measurement_circuit(bits_per_char)
    job = backend.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Take the most common outcome as the received bitstring
    most_common = max(counts.items(), key=lambda x: x[1])[0]
    chars = []
    for i in range(0, len(most_common), bits_per_char):
        chunk = most_common[i:i+bits_per_char]
        char = bit_to_char_map.get(tuple(chunk), '?')
        chars.append(char)
    return ''.join(chars)

def create_measurement_circuit(n_qubits):
    qr = QuantumRegister(n_qubits)
    cr = ClassicalRegister(n_qubits)
    qc = QuantumCircuit(qr, cr)
    qc.h(qr)  # Use Hadamard to put all qubits in superposition
    qc.measure(qr, cr)
    return qc, qr, cr


def run_quantum_read():
    # Load IBM Runtime service (assumes you're already logged in)
    service = QiskitRuntimeService()
    backend = service.backend("ibm_brisbane")  # or any available backend

    # Create the 5-qubit circuit
    qc = QuantumCircuit(5)
    qc.h(0)
    for i in range(1, 5):
        qc.cx(0, i)
    qc.measure_all()

    # Transpile to the backend's native gate set
    transpiled_qc = transpile(qc, backend=backend)

    # Initialize the sampler and run it
    sampler = Sampler(backend)
    job = sampler.run([transpiled_qc])
    result = job.result()

    # Get the most likely result (quasi-distribution output)
    raw_counts = result[0].data
    print(raw_counts)
    counts = raw_counts.dict()
    most_likely = max(counts, key=counts.get)
    
    # Return as string of bits (ensure it's 5-bit)
    return format(int(most_likely), '05b')

def decode_character(bit_tuple):
    if bit_tuple in bit_to_char_map:
        return bit_to_char_map[bit_tuple]
    else:
        print(f"[DEBUG] Unrecognized bits: {bit_tuple}")
        return '?'

def test_multiversal_reply():
    reply = ""
    for _ in range(10):
        bits = run_quantum_read()
        char = decode_character(bits)
        reply += char
    print("Decoded multiversal reply:", reply)

def calculate_entropy_from_statevector(state, subsystem_qubits):
    """Compute von Neumann entropy for a subsystem."""
    reduced_state = partial_trace(state, subsystem_qubits)
    eigenvals = np.real(np.linalg.eigvalsh(reduced_state.data))
    eigenvals = eigenvals[eigenvals > 0]  # avoid log(0)
    return -np.sum(eigenvals * np.log2(eigenvals))

def evolve_circuit_with_history(base_circuit, iterations=10, q_subsystems=None):
    """Apply temporal evolution and track entropy for qubit subsystems."""
    entropy_log = []

    for i in range(iterations):
        qc = base_circuit.copy()
        
        # Add some evolving structure (rotation + entanglement)
        for q in range(qc.num_qubits - 1):
            qc.ry(np.pi/4 * np.random.rand(), q)
            qc.cz(q, q+1)

        qc.barrier()
        state = Statevector.from_instruction(qc)

        # Entropy per subsystem
        step_data = {"iteration": i}
        for q in q_subsystems:
            traced_out = list(set(range(qc.num_qubits)) - {q})
            entropy = calculate_entropy_from_statevector(state, traced_out)
            step_data[f"Q{q}"] = entropy
        entropy_log.append(step_data)

    return entropy_log

def reverse_subsystem_entropy_oracle(target_entropies, n_qubits=3, attempts=500, tolerance=0.01):
    """
    Try to generate a quantum circuit that approximates the target entropy for each qubit.
    
    Parameters:
        target_entropies (list): Target entropy per qubit, e.g., [0.9, 0.8, 1.0]
        n_qubits (int): Number of qubits (should match len(target_entropies))
        attempts (int): Number of random circuit samples to try
        tolerance (float): Acceptable entropy difference threshold
    
    Returns:
        best_qc: QuantumCircuit with closest match
        best_entropies: list of entropies for each qubit
    """
    backend = Aer.get_backend('statevector_simulator')
    best_qc = None
    best_score = float('inf')
    best_entropies = None

    for _ in range(attempts):
        qc = QuantumCircuit(n_qubits)

        # Apply random U gates and entangle adjacent pairs
        for q in range(n_qubits):
            theta, phi, lam = np.random.uniform(0, 2*np.pi, 3)
            qc.u(theta, phi, lam, q)

        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

        sv = Statevector.from_instruction(qc)

        entropies = []
        total_diff = 0
        for i in range(n_qubits):
            reduced = partial_trace(sv, [j for j in range(n_qubits) if j != i])
            ent = qiskit_entropy(reduced)
            entropies.append(ent)
            total_diff += abs(ent - target_entropies[i])

        if total_diff < best_score:
            best_score = total_diff
            best_qc = qc
            best_entropies = entropies
            if best_score < tolerance * n_qubits:
                break

    return best_qc, best_entropies

def fast_amplify_11(iterations=10, shots=2048, initial_phase=np.pi/4, scaling_factor=0.3):
    """
    Amplifies the probability of the `11` state using adaptive charge injection and phase shifts.

    Args:
        iterations (int): Number of feedback cycles.
        shots (int): Shots per experiment.
        initial_phase (float): Initial phase shift (radians).
        scaling_factor (float): Charge update strength.
    """
    n_qubits = 2
    backend = AerSimulator()

    # Track charge history dynamically
    charge_history = np.zeros(n_qubits)

    for iteration in range(iterations):
        qc = QuantumCircuit(n_qubits, n_qubits)

        # Step 1: Initialize in equal superposition
        qc.h(range(n_qubits))

        # Step 2: Apply dynamic phase shifts based on past success
        phase_shift = initial_phase + (scaling_factor * charge_history.sum())

        # Reinforce the |11> state with targeted interference
        qc.cp(phase_shift, 0, 1)  # Controlled phase gate
        qc.cz(0, 1)  # Phase kick to steer amplitude toward 11
        qc.rx(np.pi * charge_history[0] * scaling_factor, 0)
        qc.rx(np.pi * charge_history[1] * scaling_factor, 1)

        # Step 3: Grover-like Amplification
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(1)
        qc.cp(-phase_shift, 0, 1)  # Inverse phase shift
        qc.h(1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

        # Step 4: Measure and Update Charge History
        qc.measure(range(n_qubits), range(n_qubits))

        transpiled_qc = transpile(qc, backend)
        result = backend.run(transpiled_qc, shots=shots).result()
        counts = result.get_counts()

        # Update charge tracking
        for state, count in counts.items():
            if state == "11":
                charge_history += count / shots  # Reinforce `11`
            else:
                charge_history -= count / (shots * 2)  # Reduce non-`11` states

        # Print iteration results
        print(f"Iteration {iteration + 1}/{iterations}:")
        print(f"Measurement Results: {counts}")
        print(f"Charge History: {charge_history}")

    # Final plot
    plot_histogram(counts)
    plt.title(f"Final Measurement Results (Amplifying `11` Faster)")
    plt.show()

    return counts

def build_parametrized_circuit(params, n_qubits):
    """
    Build a parameterized circuit with given angles for single-qubit unitaries and entanglement.
    Params should be a flat list: [theta0, phi0, lam0, theta1, phi1, lam1, ...]
    """
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        theta = params[3 * i + 0]
        phi   = params[3 * i + 1]
        lam   = params[3 * i + 2]
        qc.u(theta, phi, lam, i)

    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    return qc

def set_entropy_level(target_entropy: float, num_qubits: int = 1, attempts: int = 100):
    """
    Generates a quantum circuit whose subsystem entropy matches the target.

    Args:
        target_entropy (float): Desired von Neumann entropy (e.g., 0.5, 0.9).
        num_qubits (int): Number of qubits in the circuit. Entropy measured on qubit 0.
        attempts (int): How many random circuits to try.

    Returns:
        qc (QuantumCircuit): Circuit closest to target entropy.
        measured_entropy (float): Achieved entropy.
    """
    backend = Aer.get_backend('statevector_simulator')
    best_qc = None
    best_entropy = -1
    min_diff = float('inf')

    for _ in range(attempts):
        qc = QuantumCircuit(num_qubits)

        # Apply random unitaries to each qubit
        for q in range(num_qubits):
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            lam = np.random.uniform(0, 2*np.pi)
            qc.u(theta, phi, lam, q)

        # Entangle qubits linearly
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)

        # Simulate and trace out all qubits except qubit 0
        sv = Statevector.from_instruction(qc)
        reduced = partial_trace(sv, list(range(1, num_qubits)))  # Keep only qubit 0
        ent = analyze_von_neumann_qiskit_entropy(reduced)
        ent = ent["von_neumann_entropy"]

        diff = abs(ent - target_entropy)
        if diff < min_diff:
            min_diff = diff
            best_entropy = ent
            best_qc = qc
            if diff < 1e-3:
                break

    print(f"🎯 Target entropy: {target_entropy}")
    print(f"✅ Achieved entropy: {round(best_entropy, 6)}")
    return best_qc, best_entropy

def analyze_von_neumann_qiskit_entropy(statevector):
    """
    Analyzes the Von Neumann entropy of a quantum state.

    Parameters:
        statevector (np.ndarray): The statevector of the quantum system.

    Returns:
        dict: Analysis results, including Von Neumann entropy.
    """
    if statevector is None:
        print("Statevector is None; cannot calculate Von Neumann entropy.")
        return None

    # Construct the density matrix
    density_matrix = np.outer(statevector, np.conj(statevector))

    # Calculate eigenvalues of the density matrix
    eigenvalues = np.linalg.eigvalsh(density_matrix)

    # Filter out zero eigenvalues to avoid log(0)
    eigenvalues = eigenvalues[eigenvalues > 0]

    # Calculate Von Neumann entropy: S = -Tr(ρ log(ρ))
    von_neumann_entropy = -np.sum(eigenvalues * np.log2(eigenvalues))

    print("Von Neumann Entropy:", von_neumann_entropy)
    return {"von_neumann_entropy": von_neumann_entropy}

def compute_entropies(qc, n_qubits):
    state = Statevector.from_instruction(qc)
    entropies = []
    for i in range(n_qubits):
        reduced = partial_trace(state, [j for j in range(n_qubits) if j != i])
        entropies.append(qiskit_entropy(reduced))
    return np.array(entropies)


def entropy_loss(params, target_entropies, n_qubits):
    qc = build_parametrized_circuit(params, n_qubits)
    current_entropies = compute_entropies(qc, n_qubits)
    return np.sum((target_entropies - current_entropies) ** 2)


def manipulate_entropy_distribution(target_entropies, attempts=1):
    n_qubits = len(target_entropies)
    best_loss = float('inf')
    best_qc = None
    best_entropies = None

    for _ in range(attempts):
        initial_params = np.random.uniform(0, 2 * np.pi, size=3 * n_qubits)
        result = minimize(
            entropy_loss,
            initial_params,
            args=(target_entropies, n_qubits),
            method='COBYLA',
            options={'maxiter': 300, 'disp': False}
        )

        qc = build_parametrized_circuit(result.x, n_qubits)
        final_entropies = compute_entropies(qc, n_qubits)
        loss = np.sum((target_entropies - final_entropies) ** 2)

        if loss < best_loss:
            best_loss = loss
            best_qc = qc
            best_entropies = final_entropies

        if best_loss < 1e-4:
            break

    return best_qc, best_entropies

def analyze_subsystems(best_entropies, x=1.0, y=1.0, z=1.0, target_qubit_indices=[0]):
    print(f"🎯 Running entropy oracle for targets: {best_entropies}")
    result = entropy_matching_oracle(best_entropies, x, y, z)

    full_counts = result["counts"]
    subsystem_counts = get_subsystem_counts(full_counts, target_qubit_indices)

    print(f"📈 Subsystem Counts for Qubits {target_qubit_indices}:")
    for state, count in subsystem_counts.items():
        print(f"  {state}: {count}")

    return subsystem_counts

def scan_and_match_entropy_oracle(target_entropies, 
                                  x_vals, y_vals, z_vals, 
                                  qubit_indices=[0], 
                                  threshold=0.05, 
                                  attempts=5):
    from temp_funcs import entropy_matching_oracle, get_subsystem_counts

    best_result = None
    best_diff = float('inf')
    best_params = (None, None, None)

    for x in x_vals:
        for y in y_vals:
            for z in z_vals:
                print(f"🔍 Testing x={x:.2f}, y={y:.2f}, z={z:.2f}...")
                result = entropy_matching_oracle(target_entropies, x, y, z, attempts=attempts)
                entropies = result['entropies']
                diff = sum(abs(te - ae) for te, ae in zip(target_entropies, entropies))

                if diff < best_diff:
                    best_result = result
                    best_diff = diff
                    best_params = (x, y, z)

                if diff < threshold:
                    print(f"\n✅ Early match found with total error {diff:.4f}")
                    break

    print("\n=== 🎯 BEST MATCH SUMMARY ===")
    x_best, y_best, z_best = best_params
    print(f"🔑 Best x, y, z: {x_best:.4f}, {y_best:.4f}, {z_best:.4f}")
    print(f"📊 Target Entropies: {target_entropies}")
    print(f"📈 Achieved Entropies: {[round(e, 6) for e in best_result['entropies']]}")
    
    # Subsystem Analysis
    counts = best_result['counts']
    sub_counts = get_subsystem_counts(counts, qubit_indices)
    print(f"\n🧩 Subsystem Counts for qubits {qubit_indices}:")
    for state, count in sorted(sub_counts.items()):
        print(f"  {state}: {count}")

    return {
        'x': x_best,
        'y': y_best,
        'z': z_best,
        'entropies': best_result['entropies'],
        'counts': counts,
        'subsystem_counts': sub_counts,
        'total_error': best_diff
    }

def run_entropy_chain(target_entropies, x_vals, y_vals, z_vals, 
                      qubit_indices=[0], 
                      num_iterations=5, 
                      threshold=0.05, 
                      attempts=50):
    """
    Runs a chain of entropy-matching experiments and returns measurement counts for each.
    Suitable for analyzing temporal correlations.
    """
    all_counts = []

    for i in range(num_iterations):
        print(f"\n🌀 Running iteration {i + 1}/{num_iterations}")
        result = scan_and_match_entropy_oracle(
            target_entropies, x_vals, y_vals, z_vals,
            qubit_indices=qubit_indices,
            threshold=threshold,
            attempts=attempts
        )
        all_counts.append(result["counts"])
        print(result["counts"])

    return all_counts

def analyze_temporal_correlation(results_list):
    """
    Analyzes temporal correlation between successive experiment iterations.

    Parameters:
        results_list (list[dict]): List of measurement counts from each iteration.

    Returns:
        list[float]: Jensen-Shannon divergence values between successive iterations.
    """
    divergences = []
    for i in range(len(results_list) - 1):
        counts_1 = results_list[i]
        counts_2 = results_list[i + 1]

        # Normalize counts to probabilities
        total_1 = sum(counts_1.values())
        total_2 = sum(counts_2.values())

        prob_1 = [counts_1.get(bitstring, 0) / total_1 for bitstring in set(counts_1.keys()).union(counts_2.keys())]
        prob_2 = [counts_2.get(bitstring, 0) / total_2 for bitstring in set(counts_1.keys()).union(counts_2.keys())]

        # Calculate Jensen-Shannon divergence
        divergence = jensenshannon(prob_1, prob_2, base=2)
        divergences.append(divergence)

    return divergences

def is_simulator(backend):
    """
    Determines if the given backend is a simulator.

    Parameters:
        backend: Qiskit backend object.

    Returns:
        bool: True if the backend is a simulator, False otherwise.
    """
    return backend.configuration().simulator

def modify_and_run_quantum_experiment_multi_analysis(qc, backend_name="ibm_brisbane", shots=8192, modify_circuit=None, analyze_results=None):
    """
    Modifies a quantum circuit and runs it to explore experimental directions with multiple analysis functions.

    Parameters:
        qc (QuantumCircuit): The base quantum circuit.
        backend (Backend): The IBM backend or simulator for execution.
        shots (int): Number of shots for the experiment.
        modify_circuit (function): A function to modify the circuit before execution.
        analyze_results (list of functions): List of analysis functions to process results.

    Returns:
        dict: A dictionary of results from all analyses.
        
    """


    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    
    if analyze_results is None:
        analyze_results = []

    # Apply circuit modification if provided
    if modify_circuit:
        qc = modify_circuit(qc)

    # Check if backend is a simulator
    use_simulator = is_simulator(backend)

    # Store results from all analyses
    results = {}

    if use_simulator:
        # Use Aer simulator for execution
        try:
            simulator_backend = Aer.get_backend('aer_simulator')
            transpiled_qc = transpile(qc, backend=simulator_backend)
            time_reversal_qc = time_reversal_simulation(transpiled_qc)
            job = simulator_backend.run(transpiled_qc, backend=simulator_backend, shots=shots)
            result = job.result()
            counts = result.get_counts()
            print("Counts (simulated): ", counts)
            return counts
        
        except Exception as e:
            print(f"Error executing on simulated backend: {e}")
            return None
        
        # Use hardware backend for Von Nuemman entropy only
        counts = run_and_extract_counts_quantum(qc, backend, shots)
        analyze_von_neumman_entropy(counts)


    else:
        # Use IBM Runtime Sampler for hardware execution
        try:
            transpiled_qc = transpile(qc, backend=backend)
            
            with Session(backend=backend) as session:
                sampler = Sampler()
                job = sampler.run([transpiled_qc], shots=shots)
                result = job.result()

            # Debug: Inspect the raw result object
            print("Raw Result Object:", result)

            # Access the first `SamplerPubResult`
            pub_result = result._pub_results[0]  # Safely access `SamplerPubResult`
            data_bin = pub_result.data  # Access `DataBin`
            bit_array = data_bin.meas  # Access the `BitArray`

            print("Bit array: ", bit_array)

            # Convert the BitArray to a list of bitstrings
            results = extract_counts_from_bitarray(bit_array)

            analyze_shannon_entropy(results)

        except Exception as e:
            print(e)

    return results

def add_causality_to_circuit(qc, previous_results, qubits):
    """
    Introduces causal feedback to a quantum circuit based on previous results.

    Parameters:
        qc (QuantumCircuit): The quantum circuit to modify.
        previous_results (dict): Counts from the previous experiment run.
        qubits (list[int]): Indices of qubits to modify causally.

    Returns:
        QuantumCircuit: The modified quantum circuit.
    """
    from qiskit.circuit import ClassicalRegister

    # Add a classical register for storing previous measurement results
    if not hasattr(qc, "clbits") or len(qc.clbits) < len(qubits):
        creg = ClassicalRegister(len(qubits), name="c")
        qc.add_register(creg)

    # Determine the most likely outcome from previous results
    if previous_results:
        max_outcome = max(previous_results, key=previous_results.get)
        feedback_bits = [int(bit) for bit in max_outcome]

        # Add gates conditionally based on feedback bits
        for idx, bit in enumerate(feedback_bits):
            if bit == 1:
                qc.x(qubits[idx])  # Apply an X gate conditionally
    else:
        # Initialize with no feedback for the first run
        pass

    return qc

if __name__ == "__main__":
    qc, ent = set_entropy_level(0.75, num_qubits=2)
    qc.draw('mpl')

##    target_ents = [0.1, 0.3, 0.5, 0.7]
##    x, y = manipulate_entropy_distribution(target_ents)
##    x.draw("mpl")
##    print("Best Entropies: ", y)
##    analyze_subsystems(y, target_qubit_indices=[0])
##    x_vals = np.linspace(1.0, 4.0, 5)
##    y_vals = np.linspace(1.0, 4.0, 5)
##    z_vals = np.linspace(1.0, 4.0, 5)
##
##    scan_and_match_entropy_oracle(y, x_vals, y_vals, z_vals, qubit_indices=[0])
##
##    x_range = np.linspace(3.0, 4.0, 3)
##    y_range = np.linspace(1.0, 2.0, 3)
##    z_range = np.linspace(2.0, 3.0, 3)
##
##    # Run the chain of experiments
##    results = run_entropy_chain(target_ents, x_range, y_range, z_range, num_iterations=5)
##    divergences = analyze_temporal_correlation(results)
##
##    # Output
##    for i, d in enumerate(divergences):
##        print(f"📉 JS Divergence Run {i} → Run {i+1}: {d:.6f}")

        
##    # Example usage
##    qc = QuantumCircuit(3)
##    qc.h(0)
##    qc.cx(0, 1)
##    qc.cx(1, 2)
##    entropy_log = evolve_circuit_with_history(qc, iterations=10, q_subsystems=[0, 1, 2])
##
##    # Print entropy evolution
##    for step in entropy_log:
##        print(step)

    
##    counts = {'000': 100, '001': 300, '110': 600}
##    entropy_q0 = compute_subsystem_entropy(counts, [0])  # entropy of qubit 0
##    entropy_q1 = compute_subsystem_entropy(counts, [1])  # entropy of qubit 1
##    entropy_q0q1 = compute_subsystem_entropy(counts, [0, 1])  # joint entropy
##    print("Q1: ", entropy_q0, " Q2: ", entropy_q1, " Q1Q2: ", entropy_q0q1)

##    for turn in range(10):
##        if turn % 2 == 0:
##            msg = input("🧠 Your message: ")
##            send_entropy_message(msg.upper(), 0.8, 5, 8192)
##        else:
##            print("📡 Waiting for multiversal reply...")
##            reply = receive_entropy_message()
##            print(f"🌌 Their message: {reply}")

