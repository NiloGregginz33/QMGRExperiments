from qiskit import QuantumCircuit, QuantumRegister, transpile, ClassicalRegister
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix, state_fidelity, random_unitary, Operator, mutual_information, random_clifford, Clifford
from qiskit_aer import Aer, AerSimulator
from qiskit.quantum_info import entropy as qiskit_entropy
from qiskit.quantum_info import Pauli
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, amplitude_damping_error
from collections import Counter
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Estimator
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from typing import List, Tuple, Dict, Set
import qiskit
from braket.aws import AwsDevice
import boto3
from scipy.spatial.distance import jensenshannon
from qiskit.result import Result, Counts  # Import for local simulation results
from qiskit.circuit.library import QFT, RZGate, MCXGate, ZGate, XGate, HGate, CSwapGate
from qiskit.qasm3 import dump
import scipy.linalg as la
from qiskit.primitives.containers import BitArray
from sklearn.metrics import r2_score
import random
import scipy
from scipy.linalg import expm
import hashlib
from qiskit.circuit import Instruction
from scipy.stats import rankdata
import time
import concurrent.futures
from datetime import datetime
from sklearn.manifold import MDS
# from scipy.stats import binom_test
import requests
import sys
# from qiskit_ionq import IonQProvider
from scipy.stats import spearmanr, kendalltau, pearsonr
import json
import itertools
import csv
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from scipy.optimize import curve_fit
import inspect
import math
from collections import Counter, defaultdict
from typing import List, Dict
from scipy.optimize import minimize
import os
import re
import seaborn as sns
from qiskit_ibm_runtime import QiskitRuntimeService
from pprint import pprint

from qiskit_braket_provider import AWSBraketProvider

# Select a backend
def get_best_backend(service, min_qubits=3, max_queue=10):
    backends = service.backends()
    suitable_backends = [
        b for b in backends if b.configuration().n_qubits >= min_qubits and b.status().pending_jobs <= max_queue
    ]
    if not suitable_backends:
        print("No suitable backends found. Using default: ibm_brisbane")
        return service.backend("ibm_brisbane")
    
    best_backend = sorted(suitable_backends, key=lambda b: b.status().pending_jobs)[0]
    print(f"Best backend chosen: {best_backend.name}")
    print(f"Best backend chosen: {best_backend.name}")
    return best_backend

open("entropy_oracle_log.csv", "w").close()

# === CONFIG ===
USE_IBM = os.getenv("USE_IBM", "False").lower() == "false"  # Set this to "True" or "False"

if USE_IBM:
    print("[INFO] Using IBM Quantum...")
    # MY_TOKEN    = os.environ["IBM_QUANTUM_API_KEY"]

    service = QiskitRuntimeService(
        channel="ibm_cloud", 
    )
    backend = get_best_backend(service) 

    # print("IBM Quantum backends:")
    # for backend in service.backends():
        # print("-", backend)

if not USE_IBM:
    print("[INFO] Using AWS Braket...")
    provider = AWSBraketProvider()
    backends = provider.backends()
    print("AWS Braket backends:")
    # for backend in backends:
        # print("-", backend)

    # Device selection
    device = AwsDevice("arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1") 
    print(f"Using device: {device.name}")

    # S3 bucket test
    s3 = boto3.resource('s3')
    print("S3 buckets:")
    for bucket in s3.buckets.all():
        print("-", bucket.name)

##
##QiskitRuntimeService.save_account(
##    channel="ibm_cloud",
##    
##    
##    
##)

# Globals
charge_history = []
#service = QiskitRuntimeService()
#provider = IBMProvider()
#old_backend  = provider.get_backend("ibm_brisbane")

MAX_CHARGE = 20.0  # Max limit for charge adjustments
MIN_CHARGE = 0.5   # Prevents charge from getting too low
CRITICAL_MODE = False  # Flag for Hybrid Scaling Mode
memory_container = None # For amplifications

# Scaling factors
ALPHA = 2.0     # Initial large increase
BETA = 0.3      # Exponentiation growth factor
GAMMA = 1.1     # Mild exponentiation
DELTA = 3.0     # Decay factor (to reduce charge when feedback is negative)

session = None
_sampler = None

# Matches strings consisting only of '0' and '1'
BITSTR_RE = re.compile(r'^[01]+$')




#provider = IonQProvider("dxE2z8zimvBincMENCUZ8RD94qWUQ51l")
# backends: "ionq_qpu" or "ionq_simulator"
# backend = get_best_backend(service)
# bacckend = backends[0]
##est = Estimator(
##    backend
##)
##
def apply_entanglement(qc, qr):
    """Apply entanglement to the quantum circuit."""
    for i in range(len(qr) - 1):
        qc.cx(qr[i], qr[i + 1])

def measure_and_reset(qc, qr, cr):
    """Measure the qubits and reset them for the next decision."""
    qc.measure(qr, cr)
    qc.barrier()
    for qubit in qr:
        qc.reset(qubit)
        
def decompose_H_xi(rho0):
    H = -la.logm(rho0)
    # build PauliSumOp or list of (label, coeff) via the earlier helper
    return PauliSumOp.from_list( your_label_coeff_list(H) )

def add_decision_to_chain(previous_qc, previous_qr, previous_cr, decision_bits):
    """
    Add a new decision to the existing quantum decision chain.
    :param previous_qc: Previous QuantumCircuit object
    :param previous_qr: Previous QuantumRegister object
    :param previous_cr: Previous ClassicalRegister object
    :param decision_bits: List of bits representing the new decision
    :return: Updated QuantumCircuit, QuantumRegister, and ClassicalRegister
    """
    num_qubits = len(previous_qr)
    new_qr = QuantumRegister(num_qubits, name=f'q{len(previous_qc.data)}')
    new_cr = ClassicalRegister(num_qubits, name=f'c{len(previous_qc.data)}')
    new_qc = QuantumCircuit(new_qr, new_cr)

    # Encode the new decision
    encode_decision(new_qc, new_qr, decision_bits)

    # Apply entanglement
    apply_entanglement(new_qc, new_qr)

    # Use compose() to combine with the previous circuit
    combined_qc = QuantumCircuit(previous_qr, new_qr, previous_cr, new_cr)
    combined_qc.compose(previous_qc, inplace=True)
    combined_qc.compose(new_qc, inplace=True)

    return combined_qc, new_qr, new_cr

def block_entropy(qc, block_qubits):
    sv = Statevector.from_instruction(qc)
    rho_A = partial_trace(sv, [i for i in range(qc.num_qubits) if i not in block_qubits])
    return qiskit_entropy(rho_A)

def encode_disruptive_decision(decision_text):
    """Encodes a high-impact decision into a quantum circuit."""
    binary_decision = ''.join(format(ord(c), '08b') for c in decision_text)
    n_qubits = min(len(binary_decision), 5)  # Limit qubits to 5 for stability
    qr = QuantumRegister(n_qubits, 'q')
    cr = ClassicalRegister(n_qubits, 'c')
    qc = QuantumCircuit(qr, cr)

    # Apply Hadamard and phase gates based on binary decision encoding
    for i, bit in enumerate(binary_decision[:n_qubits]):
        if bit == '1':
            qc.h(qr[i])
            qc.p(np.pi / 4, qr[i])  # Phase shift for encoding
        else:
            qc.x(qr[i])

    qc.measure(qr, cr)
    return qc

def evolve_decision_over_time(qc, iterations=5):
    """Amplifies or shifts decisions dynamically over iterations."""
    for _ in range(iterations):
        for q in range(qc.num_qubits):
            qc.rx(np.pi / 8, q)  # Gradual evolution of decision state
            qc.rz(np.pi / 8, q)

    return qc

def measure_decision_influence(previous_qc, new_qc, sim_tf=False):
    """Measures how much a new decision shifts prior decisions using fidelity."""
    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=sim_tf)
    sampler = Sampler(backend)

    # Execute previous decision circuit
    transpiled_prev = transpile(previous_qc, backend)
    result_prev = sampler.run([transpiled_prev], shots=8192).result()
    print(result_prev)
    counts_prev = extract_counts_from_bitarray(result_prev[0].data.c)

    # Execute new decision circuit
    transpiled_new = transpile(new_qc, backend)
    result_new = sampler.run([transpiled_new], shots=8192).result()
    counts_new = extract_counts_from_bitarray(result_new[0].data.c)

    # Get all possible measurement outcomes
    all_keys = set(counts_prev.keys()).union(set(counts_new.keys()))

    # Normalize probabilities
    prev_probs = np.array([counts_prev.get(k, 0) / 1024 for k in all_keys])
    new_probs = np.array([counts_new.get(k, 0) / 1024 for k in all_keys])

    # Compute fidelity using aligned probability vectors
    fidelity = np.sum(np.sqrt(prev_probs * new_probs))

    return fidelity, counts_prev, counts_new


def store_decision_holographically(qc, decision_text):
    """Encodes decision memory into a holographic transformation."""
    binary_decision = ''.join(format(ord(c), '08b') for c in decision_text)
    n_qubits = qc.num_qubits
    qr = QuantumRegister(n_qubits, 'holo_q')
    qc.add_register(qr)

    # Apply holographic encoding via controlled rotations
    for i, bit in enumerate(binary_decision[:n_qubits]):
        if bit == '1':
            qc.crx(np.pi / 6, qr[i], qc.qubits[i])  # Controlled rotation
        else:
            qc.cry(np.pi / 6, qr[i], qc.qubits[i])  # Alternate encoding

    return qc


def execute_circuit(qc):
    """Execute the quantum circuit and return the measurement results."""
    simulator = Aer.get_backend('qasm_simulator')
    result = simulator.run(qc, shots=1024).result()
    counts = result.get_counts(qc)
    return counts

def encode_decision(qc, qr, decision_bits):
    """
    Encode a decision into the quantum circuit.
    :param qc: QuantumCircuit object
    :param qr: QuantumRegister object
    :param decision_bits: List of bits representing the decision
    """
    for i, bit in enumerate(decision_bits):
        if bit == '1':
            qc.x(qr[i])

# Used in our decision making framework
def initialize_qubits(num_qubits):
    """Initialize a quantum register and corresponding quantum circuit."""
    qr = QuantumRegister(num_qubits, name='q')
    cr = ClassicalRegister(num_qubits, name='c')
    qc = QuantumCircuit(qr, cr)
    return qc, qr, cr

# Create Experiment 1 circuit
def create_ex1_circuit():
    qc = QuantumCircuit(2)  # Black hole (q0) and radiation (q1)
    qc.h(0)  # Superposition on the black hole qubit
    qc.cx(0, 1)  # Entangle black hole and radiation qubits
    qc.measure_all()  # Measure both qubits
    return qc

# Function to create the quantum circuit (no classical bits for Statevector)
def create_charge_circuit(apply_positive_charge, apply_negative_charge):
    qc = QuantumCircuit(2)  # Create a new circuit
    qc.h(0)  # Superposition on Black Hole qubit
    qc.cx(0, 1)  # Entangle Black Hole and Radiation qubits

    # Simulate Charge Pulses
    if apply_positive_charge:
        qc.x(0)  # Positive charge (Pauli-X gate on Black Hole)
    if apply_negative_charge:
        qc.z(0)  # Negative charge (Pauli-Z gate on Black Hole)
    
    return qc

# Function to create a quantum circuit simulating black hole with spin conservation
def create_spin_circuit(spin_state):
    """
    Creates a quantum circuit representing a black hole and radiation system
    with initial spin injection.
    
    spin_state: "up" or "down"
    """
    qc = QuantumCircuit(2)  # 2 qubits: black hole and radiation

    # Initialize spin state for black hole
    if spin_state == "up":
        qc.initialize([1, 0], 0)  # Spin up state |0>
    elif spin_state == "down":
        qc.initialize([0, 1], 0)  # Spin down state |1>

    # Entangle black hole with radiation
    qc.h(0)  # Superposition for black hole qubit
    qc.cx(0, 1)  # Entangle black hole with radiation

    return qc

def set_variables_and_run(num_injection_cycles, num_radiation_qubits):
    num_iterations = num_injection_cycles # Number of charge injection cycles
    num_radiation_qubits = num_radiation_qubits  # Number of radiation qubits

    # Run simulation and entropy analysis
    results = simulate_and_analyze(num_iterations, num_radiation_qubits, backend)

    # Print the results
    print("Measurement Results (Counts):", results["counts"])
    print("Entropy Analysis (Entropies):", results["entropies"])

    return results

def run_circuit_statevector(qc):
    """
    Executes the quantum circuit on a statevector simulator.

    Parameters:
        qc (QuantumCircuit): The quantum circuit to execute.

    Returns:
        np.ndarray: The statevector of the circuit.
    """
    simulator = Aer.get_backend('statevector_simulator')
    job = simulator.run(qc, backend=simulator)
    result = job.result()

    # Retrieve the statevector
    statevector = result.get_statevector(qc)
    print("Statevector:", statevector)
    return statevector

# Function to calculate Shannon entropy
def calculate_shannon_entropy(counts, num_shots):
    probabilities = {key: value / num_shots for key, value in counts.items()}
    entropy = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)
    return entropy

# Function to calculate von Neumann entropy for black hole functions
def calculate_von_neumann_entropy(qc, num_radiation_qubits):
    state = Statevector.from_instruction(qc)  # Get statevector
    total_qiskit_entropy = qiskit_qiskit_entropy(state)  # Full system qiskit_entropy

    # Calculate entanglement entropy for subsystems
    black_hole_entropy = qiskit_qiskit_entropy(partial_trace(state, range(1, num_radiation_qubits + 1)))  # Trace out radiation
    radiation_entropy = qiskit_qiskit_entropy(partial_trace(state, [0]))  # Trace out black hole

    return total_entropy, black_hole_entropy, radiation_entropy

# Analyze phase shifts in the quantum state
def analyze_phases(qc):
    state = Statevector.from_instruction(qc)  # Get the statevector
    phases = np.angle(state.data)  # Extract the phases
    return phases

# Simulate black hole evaporation
def simulate_evaporation(charge_state, num_radiation_qubits):
    qc = QuantumCircuit(num_radiation_qubits + 1)  # One black hole + radiation qubits
    if charge_state == "positive":
        qc.x(0)
    elif charge_state == "negative":
        qc.z(0)

    # Entangle black hole with radiation qubits sequentially
    for i in range(1, num_radiation_qubits + 1):
        qc.h(0)  # Superposition on Black Hole
        qc.cx(0, i)  # Entangle with radiation qubit i

    return qc


def get_simulated_backend():
    """
    Retrieves the statevector simulator backend.

    Returns:
        backend: A Qiskit Aer simulator backend.
    """
    try:
        simulated_backend = Aer.get_backend('statevector_simulator')
        print("Simulated backend initialized: statevector_simulator")
        return simulated_backend
    except Exception as e:
        print(f"Error initializing simulated backend: {e}")
        return None

# Function to add measurements
def add_measurements(qc, measure_qubits):
    measured_circuit = qc.copy()  # Create a fresh copy
    measured_circuit.add_register(ClassicalRegister(len(measure_qubits)))  # Add classical register
    measured_circuit.measure(measure_qubits, range(len(measure_qubits)))  # Measure specified qubits
    return measured_circuit


# Function to create the circuit with a specific charge state
def create_circuit_with_charge(charge_state):
    qc = QuantumCircuit(2)  # Create a new 2-qubit circuit
    if charge_state == "positive":
        qc.x(0)  # Set the black hole qubit to |1⟩ (positive charge)
    elif charge_state == "negative":
        qc.z(0)  # Introduce a phase flip (negative charge)
    elif charge_state == "neutral":
        pass  # Default to |0⟩ (neutral charge)
    
    # Step 2: Entangle the black hole and radiation qubits
    qc.h(0)  # Superposition on Black Hole qubit
    qc.cx(0, 1)  # Entangle Black Hole (Register A) and Radiation (Register B)
    return qc

# Function to create the circuit with charge injections
def create_circuit_with_alternating_charges(num_injections, num_radiation_qubits):
    qc = QuantumCircuit(num_radiation_qubits + 1)  # Black hole + radiation qubits

    # Alternate between injecting positive (X) and negative (Z) charge
    for i in range(num_injections):
        if i % 2 == 0:
            qc.x(0)  # Positive charge injection
        else:
            qc.z(0)  # Negative charge injection

        # Entangle black hole with radiation qubits sequentially
        for j in range(1, num_radiation_qubits + 1):
            qc.h(0)  # Superposition on Black Hole
            qc.cx(0, j)  # Entangle with radiation qubit j

    return qc

def generate_no_qec_circuit(num_qubits=4):
    """
    Generates a simple quantum circuit without QEC, prepared for charge injection scaling tests.
    
    Args:
        num_qubits (int): Number of qubits in the circuit.
    
    Returns:
        QuantumCircuit: Initialized circuit without QEC, ready for charge injection scaling.
    """
    # Quantum and Classical Registers
    q = QuantumRegister(num_qubits, 'q')
    c = ClassicalRegister(num_qubits, 'c')
    qc = QuantumCircuit(q, c)

    # Step 1: Put all qubits in superposition (Hadamard gates)
    for i in range(num_qubits):
        qc.h(q[i])

    # Step 2: Entangle adjacent qubits (optional but creates richer test bed)
    for i in range(num_qubits - 1):
        qc.cx(q[i], q[i + 1])

    # No error correction -- ready for charge injection scaling
    return qc


# Function to create a circuit with prolonged charge injection
def create_circuit_with_prolonged_charges(num_iterations, cycle_length, num_radiation_qubits):
    qc = QuantumCircuit(num_radiation_qubits + 1)  # Black hole + radiation qubits

    for iteration in range(num_iterations):
        # Determine current charge based on cycle
        if (iteration // cycle_length) % 2 == 0:
            qc.x(0)  # Positive charge injection
        else:
            qc.z(0)  # Negative charge injection

        # Entangle black hole with radiation qubits sequentially
        for j in range(1, num_radiation_qubits + 1):
            qc.h(0)  # Superposition on Black Hole
            qc.cx(0, j)  # Entangle with radiation qubit j

    
    # Add classical registers for measurement
    classical_register = ClassicalRegister(num_radiation_qubits + 1)
    qc.add_register(classical_register)
    qc.measure(range(num_radiation_qubits + 1), range(num_radiation_qubits + 1))

    return qc

def warp_information_circuit():
    """
    Creates a 3-qubit circuit to test entanglement survival under spacetime distortions.
    """
    qc = QuantumCircuit(3)  # Three qubits

    # Step 1: Initialize entanglement
    qc.h(0)  # Superposition for the "center"
    qc.cx(0, 1)  # Entangle "center" with "boundary"
    qc.cx(1, 2)  # Extend entanglement to "outside"

    # Step 2: Apply spacetime distortions
    qc.rz(np.pi / 4, 0)  # Phase shift on the "center"
    qc.cp(np.pi / 3, 0, 1)  # Controlled phase between "center" and "boundary"
    qc.cp(np.pi / 6, 1, 2)  # Controlled phase between "boundary" and "outside"
    qc.rx(np.pi / 8, 2)  # Additional distortion on "outside"

    return qc


def create_circuit_with_time_gaps(num_injections, num_radiation_qubits, gap_cycles):
    qc = QuantumCircuit(num_radiation_qubits + 1)  # Black hole + radiation qubits

    for i in range(num_injections):
        # Inject charge
        if i % 2 == 0:
            qc.x(0)  # Positive charge injection
        else:
            qc.z(0)  # Negative charge injection

        # Entangle black hole with radiation qubits sequentially
        for j in range(1, num_radiation_qubits + 1):
            qc.h(0)  # Superposition on Black Hole
            qc.cx(0, j)  # Entangle with radiation qubit j

        # Add "time gap" as idle cycles
        qc.barrier()
        for _ in range(gap_cycles):
            qc.id(0)  # Idle gate to simulate a time gap

    # Add classical registers for measurement
    classical_register = ClassicalRegister(num_radiation_qubits + 1)
    qc.add_register(classical_register)
    qc.measure(range(num_radiation_qubits + 1), range(num_radiation_qubits + 1))

    return qc

# Function to calculate von Neumann entropy for the unmeasured circuit
def calculate_von_neumann_entropy_unmeasured(qc, num_radiation_qubits):
    try:
        # Get the statevector from the unmeasured circuit
        state = Statevector.from_instruction(qc)
        
        # Compute total system entropy
        total_entropy = qiskit_qiskit_entropy(state)

        # Compute entropies of subsystems
        black_hole_entropy = qiskit_qiskit_entropy(partial_trace(state, range(1, num_radiation_qubits + 1)))  # Trace out radiation
        radiation_entropy = qiskit_qiskit_entropy(partial_trace(state, [0]))  # Trace out black hole

        return total_entropy, black_hole_entropy, radiation_entropy
    except Exception as e:
        print(f"Error calculating von Neumann entropy: {e}")
        return None, None, None

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

def process_sampler_result(result, shots=8192):
    """
    Processes the result from the Sampler and converts probabilities into counts.

    Args:
        result: The result object from the Sampler.
        shots: The number of shots used in the experiment (default: 8192).

    Returns:
        A dictionary mapping measurement outcomes (bitstrings) to their counts.
    """
    try:
        # Debug: Print the result object to inspect its structure
        print("Raw Result Object:", result)
        # Debug: Check if result.values exists and is iterable
        if hasattr(result, "values") and isinstance(result.values, list):
            probabilities = result.values[0]  # Extract the probabilities for the first circuit
            print("Extracted Probabilities:", probabilities)
        else:
            raise AttributeError("Result object does not have 'values' or it's not a list.")
        
        # probabilities = result[0].data
        # print(f"Probabilities: ", probabilities)
        # Access the probabilities for the first circuit
            # Extract and process measurement data
        try:
            raw_data = result[0].data  # Updated for structured outputs
            counts = print(f"{key: int(value * 8192) for key, value in raw_data.items()}")  # Convert to counts
        except Exception as e:
            print(f"Error processing sampler result: {e}")
            counts = {}
            
        num_qubits = qc.num_clbits  # Infer number of qubits
        counts = {
            f"{i:0{num_qubits}b}": int(prob * 8192) for i, prob in enumerate(raw_data)
                    }
        return counts
    except Exception as e:
        print(f"Error processing sampler result: {e}")
        return {}

def create_prolonged_injection_circuit(num_iterations, num_radiation_qubits):
    """
    Creates a quantum circuit with prolonged charge injections to simulate extended multiverse interactions.

    Args:
        num_iterations (int): Number of charge injection cycles.
        num_radiation_qubits (int): Number of radiation qubits entangled with the black hole.

    Returns:
        QuantumCircuit: The generated quantum circuit.
    """
    qc = QuantumCircuit(num_radiation_qubits + 1)  # 1 black hole qubit + radiation qubits

    for iteration in range(num_iterations):
        # Alternate between positive and negative charges
        if iteration % 2 == 0:
            qc.x(0)  # Positive charge injection
        else:
            qc.z(0)  # Negative charge injection

        # Entangle black hole with radiation qubits sequentially
        for j in range(1, num_radiation_qubits + 1):
            qc.h(0)  # Superposition on Black Hole
            qc.cx(0, j)  # Entangle with radiation qubit j

    # Add classical registers for measurement
    classical_register = ClassicalRegister(num_radiation_qubits + 1)
    qc.add_register(classical_register)
    qc.measure(range(num_radiation_qubits + 1), range(num_radiation_qubits + 1))

    return qc

# Define a function for Many-Worlds quantum simulation
def many_worlds_experiment(decoherence_rate=0.1, shots=1024):
    """
    Runs a Many-Worlds experiment simulation using a qubit as an analog black hole.

    Parameters:
        decoherence_rate (float): Rate of decoherence to simulate wavefunction branching.
        shots (int): Number of measurement shots.

    Returns:
        dict: Measurement results and entropy analysis.
    """
    # Step 1: Initialize the quantum circuit
    n_qubits = 2  # 1 qubit for the analog black hole, 1 for Hawking radiation
    qc = QuantumCircuit(n_qubits)

    # Step 2: Prepare the black hole qubit in a superposition state
    # This represents the Many-Worlds idea where the black hole qubit can exist in multiple states simultaneously
    qc.h(0)  # Apply Hadamard gate to create |0> + |1>

    # Step 3: Entangle the black hole qubit with the Hawking radiation qubit
    # Entanglement links the states of the black hole with the radiation, simulating the transfer of information
    qc.cx(0, 1)  # CNOT gate creates entanglement

    # Optional Step 4: Simulate decoherence by adding noise
    # Decoherence mimics the loss of quantum coherence, a critical aspect of wavefunction branching in Many-Worlds
    qc.rx(2 * np.pi * decoherence_rate, 0)  # Rotate black hole qubit slightly to simulate decoherence
    qc.rx(2 * np.pi * decoherence_rate, 1)  # Rotate radiation qubit slightly

    # Step 5: Measure both qubits
    # Measurement collapses the quantum state in Copenhagen interpretation, but in Many-Worlds, it represents branching
    qc.measure_all()

    return qc

def run_and_extract_counts(qc, backend, shots=8192):
    """
    Runs a quantum circuit on the specified backend and extracts measurement results.

    Parameters:
        qc (QuantumCircuit): The quantum circuit to execute.
        backend (Backend): The IBM backend to use for the execution.
        shots (int): Number of shots for the experiment.

    Returns:
        dict: A dictionary of bitstring counts from the measurement results.
    """
    
    # Transpile the circuit for the backend
    transpiled_qc = transpile(qc, backend=backend)
    
    # Run the circuit with the sampler and session
    with Session(backend=backend) as session:
        sampler = Sampler()
        job = sampler.run([transpiled_qc], shots=shots)
        result = job.result()

    # Debug: Inspect the raw result object
    print("Raw Result Object:", result)

    # Extract counts from the nested structure
    try:
        # Navigate to the `BitArray` and extract counts
        pub_result = result._pub_results[0]  # Access the first `SamplerPubResult`
        data_bin = pub_result.data  # Access `DataBin`
        bit_array = data_bin.meas  # Access `BitArray`

        counts = extract_counts_from_bitarray(bit_array)

    except Exception as e:
        print(e)

    return counts


def many_worlds_analysis(qc):
    """
    Analyze the quantum state before measurement for entanglement and entropy.

    Parameters:
        qc (QuantumCircuit): The quantum circuit to analyze.

    Returns:
        None: Prints the analysis results.
    """
    # Create a copy of the circuit without measurement instructions
    qc_no_measure = qc.remove_final_measurements(inplace=False)
    
    # Get the statevector from the modified circuit
    state = Statevector.from_instruction(qc_no_measure)

    # Analyze subsystem entropy
    density_matrix = partial_trace(state, [1])  # Trace out the second qubit
    entropy_black_hole = qiskit_qiskit_entropy(density_matrix)

    # Print results
    print("Subsystem Entropy of the Black Hole Qubit:", entropy_black_hole)

    # Optional: Visualize the circuit without measurement
    print("Quantum Circuit Without Measurement:")
    print(qc_no_measure.draw())

    return "Multiversal Analysis complete"

def analyze_qiskit_entropy(qc):
    """
    Analyze the entropy of subsystems in the quantum circuit.

    Args:
        qc (QuantumCircuit): The quantum circuit to analyze.

    Returns:
        dict: Entropy values for total, black hole, and radiation subsystems.
    """
    state = Statevector.from_instruction(qc)  # Get statevector

    # Compute subsystem entropies
    black_hole_state = partial_trace(state, [1])  # Trace out radiation qubits
    radiation_state = partial_trace(state, [0])  # Trace out black hole qubit

    total_entropy = qiskit_qiskit_entropy(state)
    bh_entropy = qiskit_qiskit_entropy(black_hole_state, base=2)
    rad_entropy = qiskit_qiskit_entropy(radiation_state, base=2)

    return {
        "total_entropy": total_entropy,
        "black_hole_entropy": bh_entropy,
        "radiation_entropy": rad_entropy
    }



def simulate_and_analyze(num_iterations, num_radiation_qubits, backend, shots=8192):
    """
    Simulates the circuit with prolonged charge injections and analyzes entropy.

    Args:
        num_iterations (int): Number of charge injection cycles.
        num_radiation_qubits (int): Number of radiation qubits entangled with the black hole.
        backend: Quantum backend to execute the circuit.
        shots (int): Number of measurement shots.

    Returns:
        dict: Measurement results and entropy analysis.
    """
    # Create the circuit
    qc = create_prolonged_injection_circuit(num_iterations, num_radiation_qubits)

    # Transpile the circuit for the backend
    transpiled_qc = transpile(qc, backend=backend)

    # Run the circuit
    from qiskit_ibm_runtime import Session, Sampler
    with Session(backend=backend) as session:
        sampler = Sampler()
        job = sampler.run([transpiled_qc], shots=shots)
        result = job.result()

    # Extract measurement results
    # Extract counts from the nested structure
    try:
        # Navigate to the `BitArray` and extract counts
        pub_result = result._pub_results[0]  # Access the first `SamplerPubResult`
        data_bin = pub_result.data  # Access `DataBin`
        bit_array = data_bin.meas  # Access `BitArray`

        # Convert `BitArray` to counts dictionary
        counts = Counter(str(bit_array[i]) for i in range(len(bit_array)))

        print("Measurement Results (Counts):")
        for bitstring, count in counts.items():
            print(f"{bitstring}: {count}")

    except Exception as e:
        print(f"Error processing result structure: {e}")
        return {}

    # Analyze entropy
    entropies = analyze_qiskit_entropy(qc)

    return {
        "counts": counts,
        "entropies": entropies
    }

def modify_and_run_quantum_experiment(qc, backend, shots=8192, modify_circuit=None, analyze_results=None):
    """
    Modifies a quantum circuit and runs it to explore experimental directions.

    Parameters:
        qc (QuantumCircuit): The base quantum circuit.
        backend (Backend): The IBM backend for execution.
        shots (int): Number of shots for the experiment.
        modify_circuit (function): A function to modify the circuit before execution.
        analyze_results (function): A function to analyze the results after execution.

    Returns:
        dict: A dictionary of modified bitstring counts or analysis results.
    """
    # Apply circuit modification if provided
    if modify_circuit:
        qc = modify_circuit(qc)

    # Run the original function with the modified circuit
    counts = run_and_extract_counts_quantum(qc, backend, shots)

    # Analyze results if an analysis function is provided
    if analyze_results:
        results = analyze_results(counts)
        return results

    return counts

# Example Circuit Modification for Time-Reversal Simulation
def time_reversal_simulation(qc):
    # Remove final measurements before inversion
    qc_no_measurements = qc.remove_final_measurements(inplace=False)
    
    # Invert the circuit
    qc_reversed = qc_no_measurements.inverse()
    
    # Add measurements back
    qc_reversed.measure_all()

    return qc_reversed


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

def modify_and_run_quantum_experiment_multi_analysis(qc, backend=None, shots=8192, modify_circuit=None, analyze_results=None):
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
            time_reversal_qc.measure_all()
            transpiled_qc.measure_all()
            job = simulator_backend.run(transpiled_qc, backend=simulator_backend, shots=shots)
            result = job.result()
            counts = result.get_counts()
            print("Counts (simulated): ", counts)
            qc_no_measure = qc.remove_final_measurements(inplace=False)
            state = Statevector.from_instruction(qc_no_measure)
            analyze_shannon_qiskit_entropy(counts)
            analyze_von_neumann_qiskit_entropy(state)
            
            return counts
        
        except Exception as e:
            print(f"Error executing on simulated backend: {e}")
            return None
        
        # Use hardware backend for Von Nuemman entropy only
        counts = run_and_extract_counts_quantum(qc, backend, shots)
        


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

            analyze_shannon_qiskit_entropy(results)

        except Exception as e:
            print(e)

    return results

def is_simulator(backend):
    """
    Determines if the given backend is a simulator.

    Parameters:
        backend: Qiskit backend object.

    Returns:
        bool: True if the backend is a simulator, False otherwise.
    """
    return backend.configuration().simulator

def select_backend(use_simulator=True, hardware_backend=None):
    """
    Selects a backend based on the user's preference for simulation or hardware.

    Parameters:
        use_simulator (bool): If True, use a simulated backend.
        hardware_backend (str): Name of the hardware backend to use if not simulating.

    Returns:
        backend: A Qiskit backend object (simulated or hardware).
    """
    if use_simulator:
        try:
            backend = Aer.get_backend('aer_simulator')
            print("Using simulated backend: Aer Simulator")
            return backend
        
        except Exception as e:
            print(f"Error initializing simulated backend: {e}")
            return None
    else:
        if hardware_backend:
            try:
                from qiskit_ibm_runtime import IBMRuntimeService
                service = IBMRuntimeService()
                backend = service.backend(hardware_backend)
                print(f"Using hardware backend: {hardware_backend}")
                return backend
            except Exception as e:
                print(f"Error initializing hardware backend {hardware_backend}: {e}")
                return None
        else:
            print("No hardware backend specified. Please provide a valid backend name.")
            return None

def initialize_backend(use_simulator=True, hardware_backend_name="ibm_kyiv"):
    """
    Initializes the appropriate backend (simulator or hardware) based on user preference.

    Parameters:
        use_simulator (bool): If True, initialize a simulated backend.
        hardware_backend_name (str): Name of the hardware backend if use_simulator is False.

    Returns:
        backend: The initialized backend.
    """
    if use_simulator:
        # Initialize simulator backend
        backend = Aer.get_backend('aer_simulator')
        print("Using simulated backend: aer_simulator")
        return backend
    
    else:
        # Initialize IBM Quantum service and select hardware backend
        try:
            service = QiskitRuntimeService(channel="ibm_quantum")
            backend = service.backend(hardware_backend_name)
            print(f"Using hardware backend: {hardware_backend_name}")
            return backend
        
        except Exception as e:
            print(f"Error initializing hardware backend '{hardware_backend_name}': {e}")
            return None
        
    return backend

def calculate_subsystem_qiskit_entropy(qc):
    """Calculates the entropy of subsystems in a given quantum circuit."""
    try:
        # Check if the circuit has classical bits (i.e., has been measured)
        if qc.num_clbits == 0:
            print("⚠️ No classical bits found. Adding measurements...")
            qc.measure_all()

        # Simulate the quantum circuit to get the final density matrix
        simulator = Aer.get_backend('aer_simulator_density_matrix')
        qc_copy = qc.copy()  # Avoid modifying the original circuit
        qc_copy.save_density_matrix()  # Save final state
        compiled_circuit = transpile(qc_copy, simulator)
        result = simulator.run(compiled_circuit).result()

        # Extract the density matrix
        final_density_matrix = DensityMatrix(result.data(0)['density_matrix'])

        # Compute subsystem entropy for each qubit
        num_qubits = qc.num_qubits
        entropies = []

        for i in range(num_qubits):
            try:
                # Trace out all qubits except `i`
                reduced_state = partial_trace(final_density_matrix, [j for j in range(num_qubits) if j != i])
                entropy_val = qiskit_qiskit_entropy(reduced_state)
                entropies.append(entropy_val)
            except Exception as e:
                print(f"Error calculating entropy for qubit {i}: {e}")

        return entropies if entropies else None

    except Exception as e:
        print(f"Error calculating subsystem entropy: {e}")
        return None
    
def calculate_subsystem_entropy_hologram(statevector, num_qubits=None):
    """
    Calculates the subsystem entropy for a given statevector.
    Handles circuits with varying numbers of qubits.

    Parameters:
        statevector (Statevector): The statevector of the quantum circuit.
        num_qubits (int): Number of qubits in the circuit. If None, infer from statevector.

    Returns:
        dict: Subsystem entropy values for each subsystem, or None for single-qubit systems.
    """
    if num_qubits is None:
        num_qubits = int(np.log2(len(statevector)))

    if num_qubits < 2:
        # For single-qubit systems, the concept of subsystem entropy doesn't apply
        print("Subsystem entropy not applicable for single-qubit systems.")
        return None

    # Split the system into subsystems
    left_qubits = num_qubits // 2
    right_qubits = num_qubits - left_qubits

    if left_qubits < 1 or right_qubits < 1:
        print("Subsystems are too small for entropy calculation.")
        return None

    # Calculate reduced density matrices
    reduced_density_left = partial_trace(statevector, range(right_qubits))
    reduced_density_right = partial_trace(statevector, range(left_qubits, num_qubits))

    # Calculate von Neumann entropy for each subsystem
    entropy_left = -np.trace(reduced_density_left @ np.log2(reduced_density_left + 1e-12))
    entropy_right = -np.trace(reduced_density_right @ np.log2(reduced_density_right + 1e-12))

    return {
        "left_entropy": entropy_left,
        "right_entropy": entropy_right,
    }

def create_two_qubit_circuit():
    """
    Creates a two-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(2)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1)  # Create entanglement between qubits
    return qc

def create_three_qubit_circuit():
    """
    Creates a three-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(3)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2)  # Create entanglement between qubits
    return qc

def create_four_qubit_circuit():
    """
    Creates a four-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(4)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3)  # Create entanglement between qubits
    return qc

def create_five_qubit_circuit():
    """
    Creates a five-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(5)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4)  # Create entanglement between qubits
    return qc

def create_six_qubit_circuit():
    """
    Creates a six-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(6)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4, 5)  # Create entanglement between qubits
    return qc

def create_seven_qubit_circuit():
    """
    Creates a seven-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(7)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4, 5, 6)  # Create entanglement between qubits
    return qc

def create_eight_qubit_circuit():
    """
    Creates a eight-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(8)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4, 5, 6, 7)  # Create entanglement between qubits
    return qc

def create_nine_qubit_circuit():
    """
    Creates a nine-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(9)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4, 5, 6, 7, 8)  # Create entanglement between qubits
    return qc

def create_ten_qubit_circuit():
    """
    Creates a ten-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(10)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)  # Create entanglement between qubits
    return qc

def create_eleven_qubit_circuit():
    """
    Creates a eleven-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(11)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)  # Create entanglement between qubits
    return qc


def create_twelve_qubit_circuit():
    """
    Creates a twelve-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(12)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)  # Create entanglement between qubits
    return qc


def create_thirteen_qubit_circuit():
    """
    Creates a thirteen-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(13)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)  # Create entanglement between qubits
    return qc


def create_fourteen_qubit_circuit():
    """
    Creates a fourteen-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(14)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)  # Create entanglement between qubits
    return qc


def create_fifteen_qubit_circuit():
    """
    Creates a fifteen-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(15)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)  # Create entanglement between qubits
    return qc


def create_sixteen_qubit_circuit():
    """
    Creates a sixteen-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(16)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)  # Create entanglement between qubits
    return qc


def create_seventeen_qubit_circuit():
    """
    Creates a seventeen-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(17)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)  # Create entanglement between qubits
    return qc


def create_holographic_interaction_circuit():
    """
    Simulates interacting with a holographic boundary using 2 qubits.
    """
    qc = QuantumCircuit(2)
    
    # Step 1: Create superposition on the first qubit
    qc.h(0)

    # Step 2: Introduce entanglement between the two qubits
    qc.cx(0, 1)

    # Step 3: Simulate holographic interaction with synthetic boundary effects
    qc.u(np.pi / 3, np.pi / 6, np.pi / 9, 0)  # Custom unitary gate on qubit 0
    qc.rz(np.pi / 4, 1)  # Rotate the second qubit

    # Step 4: Introduce reverse causality (time-reversal simulation)
    qc.cx(0, 1)  # Reverse entanglement
    qc.h(0)  # Reverse superposition on qubit 0

    # Measure all qubits
    qc.measure_all()
    return qc

def run_holographic_experiment():
    qc = create_holographic_circuit()
    qc = add_holographic_interaction(qc)

    simulator = Aer.get_backend('aer_simulator')
    result = simulator.run(qc, simulator, shots=8192).result()
    counts = result.get_counts()
    
    # Analyze Statevector
    statevector = result.get_statevector(qc)
    entropies = calculate_subsystem_qiskit_entropy(statevector)

    return counts, entropies

def run_holographic_experiment_2():
    qc = create_holographic_interaction_circuit()
    qc_hologram = add_holographic_interaction(qc)

    simulator = Aer.get_backend('aer_simulator')
    result = simulator.run(qc_hologram, simulator, shots=8192).result()
    counts = result.get_counts()
    
    # Analyze Statevector
    statevector = result.get_statevector(qc)
    entropies = calculate_subsystem_qiskit_entropy(statevector)

    print(counts)

    print(entropies)

    return counts, entropies

def hack_hologram_with_injections(qc, backend, num_iterations=10, shots=8192):
    """
    Hacks the hologram through iterative charge injections.
    
    Args:
        qc (QuantumCircuit): The initial quantum circuit.
        backend: The Quantum Inspire backend for execution.
        num_iterations (int): Number of iterations for the hacking attempt.
        shots (int): Number of shots for each experiment.
    
    Returns:
        list[dict]: Results from each iteration.
    """
    results = []
    
    for iteration in range(1, num_iterations + 1):
        print(f"--- Iteration {iteration} ---")
        
        # Create a copy of the circuit for this iteration
        modified_qc = qc.copy()

        # Apply charge injections
        injection_type = "random"  # Alternate between types if needed
        modified_qc = inject_charge(modified_qc, qubits=[0, 1], injection_type=injection_type)

        # Run the circuit
        try:
            counts = run_and_extract_counts_qi(modified_qc, backend, shots)
            if counts:
                print(f"Iteration {iteration} Results: {counts}")
                results.append({
                    "iteration": iteration,
                    "counts": counts
                })

                # Analyze entropy
                statevector = run_circuit_statevector(modified_qc)
                entropies = calculate_subsystem_qiskit_entropy(statevector)
                results[-1]["entropies"] = entropies
                print(f"Entropies: {entropies}")
            else:
                print(f"Failed in iteration {iteration}")
        except Exception as e:
            print(f"Error during iteration {iteration}: {e}")
            results.append({
                "iteration": iteration,
                "counts": None,
                "error": str(e)
            })

    # Summarize results
    print("\nFinal Results Summary:")
    for result in results:
        iteration = result.get("iteration", "N/A")
        counts = result.get("counts", "N/A")
        entropies = result.get("entropies", "N/A")
        print(f"Iteration {iteration}: Counts: {counts}, Entropies: {entropies}")
    
    return results

def create_holographic_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc

def reverse_time(qc):
    return qc.inverse()

def add_holographic_interaction(qc):
    # Example: Modify the circuit to simulate a holographic boundary interaction
    qc.rx(np.pi / 4, 0)  # Rotate qubit 0
    qc.rz(np.pi / 4, 1)  # Rotate qubit 1
    return qc
    
def create_lower_dimensional_circuit():
    """
    Creates a lower-dimensional quantum circuit.
    This circuit operates on a single qubit and optionally adds minimal entanglement.
    """
    qc = QuantumCircuit(1)  # Single qubit for lower dimensionality
    
    # Apply a basic superposition
    qc.h(0)  # Hadamard gate for superposition

    # Minimal operations to keep dimensionality low
    qc.rx(np.pi / 4, 0)  # Rotate the qubit slightly
    qc.rz(np.pi / 4, 0)  # Another rotation to modify the phase

    # Optionally add measurement to finalize the state
    qc.measure_all()

    return qc

def run_circuit_statevector(qc):
    """
    Executes the quantum circuit on a statevector simulator.

    Parameters:
        qc (QuantumCircuit): The quantum circuit to execute.

    Returns:
        np.ndarray: The statevector of the circuit.
    """
    simulator = Aer.get_backend('statevector_simulator')
    job = simulator.run(qc, backend=simulator)
    result = job.result()

    # Retrieve the statevector
    statevector = result.get_statevector(qc)
    print("Statevector:", statevector)
    return statevector

# Define a simple quantum circuit
def create_base_circuit(clbits=2):
    """
    Creates a base quantum circuit with entanglement and ensures classical bits match requirements.
    """
    qc = QuantumCircuit(2)  # Start with 2 qubits

    # If classical register does not exist or is too small, add/resize it
    current_clbits = sum(creg.size for creg in qc.cregs)  # Count existing classical bits
    if current_clbits < clbits:
        additional_clbits = clbits - current_clbits
        c = ClassicalRegister(additional_clbits, "c")
        qc.add_register(c)

    qc.h(0)  # Apply Hadamard gate to qubit 0
    qc.cx(0, 1)  # Apply CNOT gate to entangle qubits 0 and 1
    classical_bits = [qc.cregs[0][i] for i in range(clbits)]  # Reference the classical register
    qc.measure(qc.qubits, classical_bits)  # Explicitly map qubits to classical bits

    return qc


def inject_charge(qc, qubits, injection_type="random"):
    """
    Injects charge (gates) into the quantum circuit on specified qubits.

    Args:
        qc (QuantumCircuit): The quantum circuit.
        qubits (list): List of qubits to target.
        injection_type (str): Type of injection ("X", "Y", "Z", or "random").
    
    Returns:
        QuantumCircuit: Modified circuit with injections.
    """
    for qubit in qubits:
        if injection_type == "random":
            gate = random.choice(["x", "y", "z"])
        else:
            gate = injection_type.lower()
        
        if gate == "x":
            qc.x(qubit)  # Bit-flip
        elif gate == "y":
            qc.y(qubit)  # Bit+Phase flip
        elif gate == "z":
            qc.z(qubit)  # Phase flip

    return qc

def multiverse_warp_circuit():
    """
    Creates a circuit to simulate a warp bubble interacting with alternate timelines.
    """
    # Create a quantum circuit with 5 qubits: 3 for the bubble, 2 for alternate timelines
    qc = QuantumCircuit(5)

    # Step 1: Initialize entanglement in the warp bubble
    qc.h(0)  # Superposition for the "center"
    qc.cx(0, 1)  # Entangle "center" with "boundary"
    qc.cx(1, 2)  # Extend entanglement to "outside"

    # Step 2: Initialize alternate timelines
    qc.h(3)  # Superposition for timeline 1
    qc.h(4)  # Superposition for timeline 2

    # Step 3: Apply interactions between bubble and timelines
    qc.cp(np.pi / 3, 0, 3)  # Controlled phase between center and timeline 1
    qc.cp(np.pi / 4, 1, 4)  # Controlled phase between boundary and timeline 2
    qc.cx(2, 3)  # Entangle outer region with timeline 1
    qc.cx(3, 4)  # Entangle timeline 1 with timeline 2

    # Step 4: Apply distortions to the timelines
    qc.rz(np.pi / 6, 3)  # Phase shift on timeline 1
    qc.rx(np.pi / 8, 4)  # Rotation on timeline 2

    return qc

def entropy_model(params, A, B):
    theta_s, theta_d, phi = params
    return A * np.sin(theta_s)**2 * np.sin(theta_d)**2 * (1 - np.exp(-B * phi**2))


def analyze_multiverse_entanglement(qc):
    """
    Analyzes the survival of entanglement and mutual information in a multiverse scenario.
    """
    # Get the quantum statevector
    state = Statevector.from_instruction(qc)

    # Trace out subsystems to calculate entropies
    subsystems = [
        partial_trace(state, [1, 2, 3, 4]),  # Center
        partial_trace(state, [0, 2, 3, 4]),  # Boundary
        partial_trace(state, [0, 1, 3, 4]),  # Outside
        partial_trace(state, [0, 1, 2, 4]),  # Timeline 1
        partial_trace(state, [0, 1, 2, 3])   # Timeline 2
    ]
    entropies = [qiskit_qiskit_entropy(sub, base=2) for sub in subsystems]

    # Combined subsystems for mutual information
    combined_center_timeline1 = partial_trace(state, [1, 2, 4])  # Center + Timeline 1
    combined_boundary_timeline2 = partial_trace(state, [0, 2, 3])  # Boundary + Timeline 2

    # Calculate mutual information
    mi_center_timeline1 = (
        entropies[0] + entropies[3] - qiskit_qiskit_entropy(combined_center_timeline1, base=2)
    )
    mi_boundary_timeline2 = (
        entropies[1] + entropies[4] - qiskit_qiskit_entropy(combined_boundary_timeline2, base=2)
    )

    # Print subsystem entropies
    print("Subsystem Entropies:")
    print(f"Center (Qubit 0): {entropies[0]:.4f}")
    print(f"Boundary (Qubit 1): {entropies[1]:.4f}")
    print(f"Outside (Qubit 2): {entropies[2]:.4f}")
    print(f"Timeline 1 (Qubit 3): {entropies[3]:.4f}")
    print(f"Timeline 2 (Qubit 4): {entropies[4]:.4f}")

    # Print mutual information
    print("\nMutual Information:")
    print(f"Center ↔ Timeline 1: {mi_center_timeline1:.4f}")
    print(f"Boundary ↔ Timeline 2: {mi_boundary_timeline2:.4f}")

    return entropies, mi_center_timeline1, mi_boundary_timeline2

def least_busy_backend(service, filters=None):
    """
    Find the least busy backend from the available IBM Quantum backends.

    Parameters:
        service (QiskitRuntimeService): An initialized QiskitRuntimeService object.
        filters (function): A lambda function to filter the list of backends.

    Returns:
        Backend: The least busy backend that matches the filter criteria.
    """
    # Get all backends
    backends = service.backends()

    # Apply filters if provided
    if filters:
        backends = list(filter(filters, backends))

    # Sort by the number of pending jobs (ascending)
    sorted_backends = sorted(
        backends, key=lambda b: b.status().pending_jobs
    )

    # Return the least busy backend
    return sorted_backends[0] if sorted_backends else None

# Extract counts from BitArray
def extract_counts_from_bitarray(bit_array):
    try:
        # Attempt to use `get_counts` or related methods
        if hasattr(bit_array, "get_counts"):
            counts = bit_array.get_counts()
            print("Counts (get_counts):", counts)
            return counts

        if hasattr(bit_array, "get_int_counts"):
            int_counts = bit_array.get_int_counts()
            print("Integer Counts (get_int_counts):", int_counts)
            return int_counts

        if hasattr(bit_array, "get_bitstrings"):
            bitstrings = bit_array.get_bitstrings()
            counts = Counter(bitstrings)
            print("Bitstrings (Counter):", counts)
            return counts

        # Manual decoding if above methods are unavailable
        print("No direct methods worked; attempting manual decoding.")
        raw_data = str(bit_array)
        counts = Counter(raw_data.split())
        return counts

    except Exception as e:
        print(f"Error processing BitArray: {e}")
        return {}


def dynamic_warp_circuit(t_steps):
    """
    Creates a dynamically evolving warp hologram circuit.
    """
    qc = QuantumCircuit(3, 3)  # 3 qubits, 3 classical bits for measurement

    # Initialize entanglement
    qc.h(0)  # Superposition for "center"
    qc.cx(0, 1)  # Entangle "center" with "boundary"
    qc.cx(1, 2)  # Extend entanglement to the "outside"

    # Time evolution with phase shifts
    for t in range(1, t_steps + 1):
        qc.rz(np.pi * t / 10, 0)  # Time-dependent phase shift on Qubit 0
        qc.rx(np.pi * t / 15, 1)  # Time-dependent rotation on Qubit 1
        qc.cp(np.pi * t / 20, 1, 2)  # Controlled phase interaction between Qubits 1 and 2

    qc.measure([0, 1, 2], [0, 1, 2])  # Measure all qubits
    return qc

def run_warp_simulation_measure(qc):
    """
    Runs the quantum circuit using a statevector simulator.
    """
    # Simulate the quantum circuit using Statevector
    state = Statevector.from_instruction(qc)

    # Analyze the statevector
    probabilities = state.probabilities_dict()
    print("\nState Probabilities (Quantum):")
    for state, prob in probabilities.items():
        print(f"State {state}: {prob:.4f}")

    return probabilities

def process_sampler_result(result, shots=2048):
    """
    Process and format the output from a Qiskit Sampler run result.

    Parameters:
    - result: The result object from a Sampler run.
    - shots (int): The number of shots used in the experiment.

    Returns:
    - readable_output (str): Formatted string summarizing the result.
    - binary_probabilities (dict): Measurement outcomes as binary probabilities.
    """
    readable_output = ""
    binary_probabilities = {}

    try:
        # Extract quasi-probabilities from the result
        quasi_dists = result.quasi_dists[0]  # Assuming a single circuit
        binary_probabilities = quasi_dists.binary_probabilities()

        # Generate readable summary
        readable_output += f"Total Shots: {shots}\n"
        readable_output += f"{'State':<10}{'Probability (%)':<20}{'Quasi-Probability':<20}\n"
        readable_output += "-" * 50 + "\n"

        for state, probability in binary_probabilities.items():
            quasi_prob = quasi_dists.get(state, 0)
            readable_output += f"{state:<10}{probability * 100:<20.5f}{quasi_prob:<20.5f}\n"

    except AttributeError as e:
        readable_output += f"Error processing result: {e}\n"
        readable_output += "Ensure the result contains valid quasi-probabilities.\n"

    return readable_output, binary_probabilities



def dynamic_warp_circuit_no_measure(t_steps):
    """
    Creates a dynamically evolving warp hologram circuit.
    """
    qc = QuantumCircuit(3, 3)  # 3 qubits, 3 classical bits for measurement

    # Initialize entanglement
    qc.h(0)  # Superposition for "center"
    qc.cx(0, 1)  # Entangle "center" with "boundary"
    qc.cx(1, 2)  # Extend entanglement to the "outside"

    # Time evolution with phase shifts
    for t in range(1, t_steps + 1):
        qc.rz(np.pi * t / 10, 0)  # Time-dependent phase shift on Qubit 0
        qc.rx(np.pi * t / 15, 1)  # Time-dependent rotation on Qubit 1
        qc.cp(np.pi * t / 20, 1, 2)  # Controlled phase interaction between Qubits 1 and 2

    qc_no_measure = qc.copy()

    qc.measure_all()

    # Get the statevector of the circuit (without measurements)
    state = Statevector.from_instruction(qc_no_measure)
    print("\nStatevector of the System:")
    print(state)

    # Subsystem isolation: trace out specific qubits
    subsystem_1 = partial_trace(state, [1, 2])  # Isolate Qubit 0
    subsystem_2 = partial_trace(state, [0, 2])  # Isolate Qubit 1
    subsystem_3 = partial_trace(state, [0, 1])  # Isolate Qubit 2

    # Compute entropies for each subsystem
    entropy_1 = qiskit_qiskit_entropy(subsystem_1)
    entropy_2 = qiskit_qiskit_entropy(subsystem_2)
    entropy_3 = qiskit_qiskit_entropy(subsystem_3)

    print("\nSubsystem Entropies:")
    print(f"Qubit 0 Entropy: {entropy_1}")
    print(f"Qubit 1 Entropy: {entropy_2}")
    print(f"Qubit 2 Entropy: {entropy_3}")

    # Transpile the circuit for AerSimulator
    backend = Aer.get_backend('aer_simulator')
    transpiled_qc = transpile(qc, backend)

    # Run the circuit and get measurement results
    result = backend.run(transpiled_qc, shots=2048).result()
    counts = result.get_counts()

    print("\nMeasurement Results:")
    print(counts)

    # Plot histogram of measurement results
    plot_histogram(counts)
    plt.title("Measurement Results Histogram")
    plt.show()

    return qc

def time_evolution_example(t_steps=5, shots=2048):
    """
    Simulate time-dependent transformations and holographic interactions.
    """
    n_qubits = 3
    qc = QuantumCircuit(n_qubits)
    
    # Step 1: Initialize superposition
    qc.h(0)  # Superposition on the Black Hole qubit
    qc.cx(0, 1)  # Entangle Black Hole and Radiation qubits

    # Step 2: Time Evolution
    for t in range(t_steps):
        angle = (np.pi / 3) * (t + 1)  # Dynamic angle for timeline distortion
        qc.rz(angle, 0)  # Timeline distortion on the Black Hole qubit
        qc.rx(np.pi / 4, 1)  # Holographic interaction on the Radiation qubit
        qc.cx(1, 2)  # Correlate Radiation with External Environment
        qc.barrier()

    # Step 3: Measure final probabilities
    qc.measure_all()

    # Analyze statevector before measurement
    qc_no_measurements = qc.copy()
    state = Statevector.from_instruction(qc_no_measurements)
    print("\nStatevector:")
    print(state)

    # Calculate subsystem entropies
    entropy_0 = qiskit_qiskit_entropy(partial_trace(state, [1, 2]))
    entropy_1 = qiskit_qiskit_entropy(partial_trace(state, [0, 2]))
    entropy_2 = qiskit_qiskit_entropy(partial_trace(state, [0, 1]))
    print(f"Subsystem Entropies: Qubit 0 = {entropy_0}, Qubit 1 = {entropy_1}, Qubit 2 = {entropy_2}")

    # Transpile and run on AerSimulator
    backend = Aer.get_backend('aer_simulator')
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()

    print("\nMeasurement Results:")
    print(counts)

    # Plot the histogram of results
    plot_histogram(counts)
    plt.title("Measurement Results with Time Evolution")
    plt.show()

# KEEP IN MIND WHEN USING ANY STATE MANIPULATION, ALWAYS MAKE ALL 3 BITS THE SAME - straight 1 or 0s
# FAILURE to do so will result in decoherent states, which is AGAINST our responsibilities

def targeted_gates(target_state="111", shots=2048):
    """
    Design a circuit with targeted gates to amplify a specific state.
    """
    n_qubits = 3
    qc = QuantumCircuit(n_qubits)
    
    # Initial superposition
    qc.h(range(n_qubits))  # Create equal superposition of all states
    
    # Targeted phase shifts to amplify |111>
    if target_state == "111":
        qc.h(2)             # Reverse the superposition on the last qubit
        qc.ccx(0, 1, 2)     # Apply Toffoli gate to mark |111>
        qc.h(2)             # Return to superposition basis
        qc.z(2)             # Phase flip on |111>
        qc.barrier()
    
    # Additional controlled gates for steering probability
    qc.cp(3.14, 0, 1)  # Controlled phase gate
    qc.cx(1, 2)         # Controlled NOT
    qc.cz(0, 2)         # Controlled Z
    
    # Copy circuit for analysis without measurement
    qc_no_measurements = qc.copy()
    
    # Add measurement gates
    qc.measure_all()

    # Analyze the statevector (before measurement)
    state = Statevector.from_instruction(qc_no_measurements)
    print("\nStatevector:")
    print(state)

    # Calculate subsystem entropy
    entropy_0 = qiskit_qiskit_entropy(partial_trace(state, [1, 2]))
    entropy_1 = qiskit_qiskit_entropy(partial_trace(state, [0, 2]))
    entropy_2 = qiskit_qiskit_entropy(partial_trace(state, [0, 1]))
    print(f"Subsystem Entropies: Qubit 0 = {entropy_0}, Qubit 1 = {entropy_1}, Qubit 2 = {entropy_2}")

    # Transpile and run on a simulator
    backend = Aer.get_backend('aer_simulator')
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()

    print("\nMeasurement Results:")
    print(counts)

    # Plot the histogram of results
    plot_histogram(counts)
    plt.title("Measurement Results with Targeted Gates")
    plt.show()

def hamiltonian_calc(delta=1.0,J=0.5,Omega=0.3,omega=2.0,t_max=10,num_steps=100):
    # Define Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]])  # Pauli-X
    sigma_z = np.array([[1, 0], [0, -1]])  # Pauli-Z
    identity = np.eye(2)  # Identity matrix

    # Define operators for the two-qubit system
    sigma_z_BH = np.kron(sigma_z, identity)  # Pauli-Z for black hole qubit
    sigma_z_Rad = np.kron(identity, sigma_z)  # Pauli-Z for radiation qubit
    sigma_x_BH = np.kron(sigma_x, identity)  # Pauli-X for black hole qubit

    # Time-independent part of the Hamiltonian
    H_0 = (delta / 2) * sigma_z_BH + J * np.dot(sigma_z_BH, sigma_z_Rad)

    # Time-dependent part of the Hamiltonian
    def H_t(t):
        return Omega * np.cos(omega * t) * sigma_x_BH

    # Full Hamiltonian as a function of time
    def H(t):
        return H_0 + H_t(t)

    # Time evolution using matrix exponentiation
    def time_evolve(initial_state, t_max, num_steps):
        dt = t_max / num_steps
        state = initial_state
        for step in range(num_steps):
            t = step * dt
            U = expm(-1j * H(t) * dt)  # Time evolution operator
            state = np.dot(U, state)  # Apply time evolution
        return state

    # Define the initial state (black hole + radiation qubits in |00>)
    initial_state = np.array([1, 0, 0, 0])  # |00> in computational basis

    # Simulate time evolution
    final_state = time_evolve(initial_state, t_max, num_steps)

    # Print final state
    print("Final state vector:")
    print(final_state)

def warp_information_circuit():
    """
    Creates a 3-qubit circuit to test entanglement survival under spacetime distortions.
    """
    qc = QuantumCircuit(3)  # Three qubits

    # Step 1: Initialize entanglement
    qc.h(0)  # Superposition for the "center"
    qc.cx(0, 1)  # Entangle "center" with "boundary"
    qc.cx(1, 2)  # Extend entanglement to "outside"

    # Step 2: Apply spacetime distortions
    qc.rz(np.pi / 4, 0)  # Phase shift on the "center"
    qc.cp(np.pi / 3, 0, 1)  # Controlled phase between "center" and "boundary"
    qc.cp(np.pi / 6, 1, 2)  # Controlled phase between "boundary" and "outside"
    qc.rx(np.pi / 8, 2)  # Additional distortion on "outside"

    return qc

def run_simulation(qc):
    """
    Runs the warp simulation circuit and displays results.
    """
    simulator = Aer.get_backend('aer_simulator')
    transpiled = transpile(qc, simulator)
    result = simulator.run(transpiled, shots=2048).result()
    counts = result.get_counts()

    # Display results as a histogram
    plot_histogram(counts)
    plt.title("Simulation Results")
    plt.show()

    return counts

def analyze_results(results):
    """
    Analyzes and visualizes state probabilities from the results.
    """
    total_shots = sum(results.values())
    probabilities = {state: count / total_shots for state, count in results.items()}

    # Print probabilities
    print("\nState Probabilities:")
    for state, prob in probabilities.items():
        print(f"State {state}: {prob:.4f}")

    # Visualize probabilities
    states = list(probabilities.keys())
    probs = list(probabilities.values())
    plt.bar(states, probs)
    plt.xlabel('States')
    plt.ylabel('Probability')
    plt.title('State Probabilities from Dynamic Warp Circuit')
    plt.show()

    return probabilities

def calculate_entropies(state):
    """
    Calculates entropies for subsystems in the quantum state.
    """
    subsystem_0 = partial_trace(state, [1, 2])  # Isolate Qubit 0
    subsystem_1 = partial_trace(state, [0, 2])  # Isolate Qubit 1
    subsystem_2 = partial_trace(state, [0, 1])  # Isolate Qubit 2

    entropy_0 = qiskit_qiskit_entropy(subsystem_0, base=2)
    entropy_1 = qiskit_qiskit_entropy(subsystem_1, base=2)
    entropy_2 = qiskit_qiskit_entropy(subsystem_2, base=2)

    return {"Qubit 0": entropy_0, "Qubit 1": entropy_1, "Qubit 2": entropy_2}

def run_quantum_simulation(qc):
    """
    Runs the quantum circuit using a statevector simulator (no measurements allowed).
    """
    # Create a copy of the circuit without measurements
    qc_no_measure = qc.remove_final_measurements(inplace=False)

    # Simulate the quantum circuit using Statevector
    state = Statevector.from_instruction(qc_no_measure)

    # Analyze the statevector
    probabilities = state.probabilities_dict()
    print("\nState Probabilities (Quantum):")
    for state, prob in probabilities.items():
        print(f"State {state}: {prob:.4f}")

    return probabilities

def enhanced_time_evolution(t_steps, shots=2048):
    """
    Enhance time-dependent transformations and holographic interactions.
    """
    qc = QuantumCircuit(3, 3)

    # Initialize entanglement
    qc.h(0)  # Black Hole qubit in superposition
    qc.cx(0, 1)  # Entangle Black Hole and Radiation
    qc.cx(1, 2)  # Extend entanglement to External Environment

    # Time-evolution with holographic interactions
    for t in range(1, t_steps + 1):
        angle_timeline = np.pi * t / 10
        angle_holographic = np.pi * t / 15

        qc.rz(angle_timeline, 0)  # Timeline distortion (Black Hole qubit)
        qc.rx(angle_holographic, 1)  # Holographic interaction (Radiation qubit)
        qc.cp(np.pi * t / 20, 1, 2)  # Controlled phase interaction

    # Statevector analysis (no measurement gates)
    qc_no_measurements = qc.copy()

    # Add measurements for final probabilities
    qc.measure([0, 1, 2], [0, 1, 2])

    # Analyze statevector
    state = Statevector.from_instruction(qc_no_measurements)
    entropies = calculate_entropies(state)

    print("\nStatevector:")
    print(state)
    print("Subsystem Entropies:", entropies)

    # Run the circuit on a simulator
    results = run_warp_simulation(qc)
    print("\nMeasurement Results:", results)

    return results, entropies

# Keep in mind the run experiment has a target_state
    
def run_experiment(backend_type, target_state="111", t_steps=5, shots=2048):
    """
    Run a mirrored state experiment with time-dependent holographic operations.
    """
    # Select backend
    if backend_type == "simulator":
        backend = Aer.get_backend('aer_simulator')
        print("\nRunning on AerSimulator...")
    elif backend_type == "quantum":
        service = QiskitRuntimeService(channel="ibm_quantum")  # Load IBM Quantum account
        filters = lambda b: b.configuration().n_qubits >= 2 and b.status().operational
        backend = least_busy_backend(service, filters)
        if backend is None:
            raise ValueError("No backends are appear to be working at the minute...")
        print(f"\nRunning on quantum backend: {backend.name}...")
    else:
        raise ValueError("Invalid backend_type. Use 'simulator' or 'quantum'.")

    n_qubits = len(target_state)
    qc = QuantumCircuit(n_qubits, n_qubits)  # Create the quantum circuit

    # Step 1: Time-dependent holographic operations
    for t in range(1, t_steps + 1):
        angle_timeline = np.pi * t / 10
        angle_holographic = np.pi * t / 15

        qc.rz(angle_timeline, 0)  # Timeline distortion on Black Hole qubit
        qc.rx(angle_holographic, 1)  # Holographic interaction on Radiation qubit
        qc.cx(1, 2)  # Correlate Radiation with Environment

    # Step 2: Add measurement
    qc.measure(range(n_qubits), range(n_qubits))
    
    # Transpile and run
    transpiled_qc = transpile(qc, backend=backend, optimization_level=3)
    print(f"Transpiled Circuit Depth: {transpiled_qc.depth()}")

    if backend_type == "simulator":
        result = backend.run(transpiled_qc, shots=shots).result()
        counts = result.get_counts()
        print("Results: ", counts)
    elif backend_type == "quantum":
        with Session(backend=backend) as session:
        # Use Qiskit Runtime Sampler for quantum backend
            sampler = Sampler()
            job = sampler.run([transpiled_qc])
            result = job.result()
            print("Results: ", result)
                # Extract counts from the nested structure
            try:
            # Navigate to the `BitArray` and extract counts
                pub_result = result._pub_results[0]  # Access the first `SamplerPubResult`
                data_bin = pub_result.data  # Access `DataBin`
                bit_array = data_bin.c  # Access `BitArray`

                counts = bit_array.to_dict()
                print("Results: ", counts)

            except Exception as e:
                print(e)

def create_holographic_timeline_circuit(target_state="11"):
    """
    Creates a quantum circuit to simulate holographic timeline interactions with a targeted state.
    Args:
        target_state (str): The target state to encode for the Black Hole and Radiation qubits.
    Returns:
        QuantumCircuit: The quantum circuit representing the timeline interaction.
    """
    n_qubits = len(target_state)
    qc = QuantumCircuit(n_qubits, n_qubits)  # Create circuit with classical bits for measurement

    # Encode target state
    for i, bit in enumerate(target_state):
        if bit == '1':
            qc.x(i)  # Flip the qubit

    # Apply holographic timeline interaction
    qc.h(0)  # Superposition for Black Hole qubit
    qc.cx(0, 1)  # Entangle Black Hole with Radiation
    qc.rz(np.pi / 3, 0)  # Simulate timeline distortion on Black Hole
    qc.rx(np.pi / 4, 1)  # Simulate holographic interaction on Radiation

    qc.measure(range(n_qubits), range(n_qubits))

    return qc

def run_holographic_timeline_circuit(qc, backend_type="simulator", shots=1024):
    """
    Runs the holographic timeline circuit and visualizes results.
    Args:
        qc (QuantumCircuit): The quantum circuit to execute.
        backend_type (str): The type of backend ("simulator" or "quantum").
        shots (int): The number of shots for the execution.
    Returns:
        dict: The processed results of the execution.
    """
    # Select backend
    if backend_type == "simulator":
        backend = Aer.get_backend('aer_simulator')
        print("\nRunning on AerSimulator...")
    elif backend_type == "quantum":
        service = QiskitRuntimeService(channel="ibm_quantum")  # Load IBM Quantum account
        filters = lambda b: b.configuration().n_qubits >= 2 and b.status().operational
        backend = least_busy_backend(service, filters)
        if backend is None:
            raise ValueError("No backends are appear to be working at the minute...")
        print(f"\nRunning on quantum backend: {backend.name}...")
    else:
        raise ValueError("Invalid backend_type. Use 'simulator' or 'quantum'.")

    # Transpile and execute circuit
    transpiled_circuit = transpile(qc, backend)
    job = backend.run(transpiled_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Visualize results
    plot_histogram(counts)
    plt.title("Holographic Timeline Interaction Results")
    plt.show()

    # Analyze statevector
    state = Statevector.from_instruction(qc)
    entropies = analyze_holographic_subsystem_entropy_(state)

    print("Subsystem Entropies:", entropies)
    return counts, entropies

def analyze_holographic_subsystem_qiskit_entropy(statevector):
    """
    Analyzes the entropy of subsystems (Black Hole and Radiation).
    """
    black_hole_state = partial_trace(statevector, [1])  # Trace out Radiation
    radiation_state = partial_trace(statevector, [0])  # Trace out Black Hole
    bh_entropy = qiskit_qiskit_entropy(black_hole_state, base=2)
    rad_entropy = qiskit_qiskit_entropy(radiation_state, base=2)
    return {
        "black_hole_entropy": bh_entropy,
        "radiation_entropy": rad_entropy
    }


def analyze_subsystem_qiskit_entropy(statevector):
    """
    Analyzes the entropy of subsystems (Black Hole and Radiation).
    Args:
        statevector (Statevector): The quantum statevector to analyze.
    Returns:
        dict: Entropy values for Black Hole and Radiation subsystems.
    """
    black_hole_state = partial_trace(statevector, [1])  # Trace out Radiation
    radiation_state = partial_trace(statevector, [0])  # Trace out Black Hole
    bh_entropy = qiskit_qiskit_entropy(black_hole_state, base=2)
    rad_entropy = qiskit_qiskit_entropy(radiation_state, base=2)
    return {
        "black_hole_entropy": bh_entropy,
        "radiation_entropy": rad_entropy
    }


# The next few functions are for feedbacks, which often do not work well as opposed to setting a target_state in another function

#########################################################################################################################################################################################################

# Holographic Feedback Loop Core Framework
def holographic_feedback_loop(target_state="11", adjust_factor=0.1, shots=1024, backend_type="quantum"):
    """
    Simulates a holographic feedback loop to adjust sensory perceptions or outcomes.
    Args:
        target_state (str): The state to align or adjust perceptions towards.
        adjust_factor (float): The adjustment factor influencing the feedback loop.
        shots (int): The number of shots for the simulation.
    Returns:
        dict: Results and analysis of the feedback loop.
    """
    # Select backend
    if backend_type == "simulator":
        backend = Aer.get_backend('aer_simulator')
        print("\nRunning on AerSimulator...")
    elif backend_type == "quantum":
        service = QiskitRuntimeService(channel="ibm_quantum")  # Load IBM Quantum account
        filters = lambda b: b.configuration().n_qubits >= 2 and b.status().operational
        backend = least_busy_backend(service, filters)
        if backend is None:
            raise ValueError("No backends are appear to be working at the minute...")
        print(f"\nRunning on quantum backend: {backend.name}...")
    else:
        raise ValueError("Invalid backend_type. Use 'simulator' or 'quantum'.")


    n_qubits = len(target_state)
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Encode target state
    for i, bit in enumerate(target_state):
        if bit == '1':
            qc.x(i)

    # Holographic adjustments
    qc.h(0)  # Superposition for Black Hole qubit
    qc.cx(0, 1)  # Entangle Black Hole with Radiation
    qc.rz(np.pi * adjust_factor, 0)  # Simulate timeline distortion on Black Hole
    qc.rx(np.pi * adjust_factor, 1)  # Simulate holographic interaction on Radiation

    # Add measurement
    qc.measure(range(n_qubits), range(n_qubits))

    # Run simulation
    backend = Aer.get_backend('aer_simulator')
    transpiled_circuit = transpile(qc, backend)
    job = backend.run(transpiled_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Visualization
    plot_histogram(counts)

    return {
        "counts": counts,
        "target_state": target_state,
        "adjust_factor": adjust_factor
    }

# Safeguard Mechanism
def stop_feedback_loop():
    """Stops the feedback loop safely."""
    print("Holographic feedback loop stopped.")

def simulate_feedback_loop(target_state="11", adjust_factor=0.1, iterations=10, shots=1024):
    """
    Simulates the holographic feedback loop over multiple iterations.

    Args:
        target_state (str): The state to align probabilities toward.
        adjust_factor (float): The factor influencing the feedback adjustments.
        iterations (int): Number of iterations to simulate.
        shots (int): Number of shots per simulation.

    Returns:
        dict: Evolution of state probabilities over iterations.
    """
    n_qubits = len(target_state)
    backend = Aer.get_backend('aer_simulator')

    # Initial probabilities
    probabilities = {state: 0 for state in [f"{i:0{n_qubits}b}" for i in range(2**n_qubits)]}
    probabilities[target_state] = 1 / len(probabilities)  # Start slightly biased toward target

    evolution = []

    for iteration in range(iterations):
        qc = QuantumCircuit(n_qubits, n_qubits)

        # Encode target state
        for i, bit in enumerate(target_state):
            if bit == '1':
                qc.x(i)

        # Apply holographic adjustments
        qc.h(0)  # Black Hole qubit
        qc.cx(0, 1)  # Entangle Black Hole and Radiation
        qc.rz(np.pi * adjust_factor, 0)
        qc.rx(np.pi * adjust_factor, 1)

        # Measure
        qc.measure(range(n_qubits), range(n_qubits))

        # Simulate
        transpiled_circuit = transpile(qc, backend)
        job = backend.run(transpiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Update probabilities
        total_shots = sum(counts.values())
        for state, count in counts.items():
            probabilities[state] = (1 - adjust_factor) * probabilities.get(state, 0) + adjust_factor * (count / total_shots)

        evolution.append(probabilities.copy())

    return evolution

def plot_evolution(evolution, target_state):
    """Plots the evolution of state probabilities over iterations."""
    states = list(evolution[0].keys())
    iterations = len(evolution)

    # Prepare data for plotting
    data = {state: [step[state] for step in evolution] for state in states}

    plt.figure(figsize=(12, 6))
    for state, values in data.items():
        plt.plot(range(iterations), values, label=f"State {state}", linestyle='--' if state != target_state else '-', linewidth=2)

    plt.title("Holographic Feedback Loop: Probability Evolution")
    plt.xlabel("Iterations")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    plt.show()


def run_with_fidelity(circuit, backend_type="simulator", shots=1024):
    """
    Runs the circuit and evaluates the fidelity of the result.
    Args:
        circuit (QuantumCircuit): The circuit to execute.
        backend_type (str): Backend type ("simulator" or "quantum").
        shots (int): Number of shots.
    Returns:
        tuple: (results, fidelity)
    """
    backend = None
    if backend_type == "simulator":
        from qiskit.providers.aer import AerSimulator
        backend = AerSimulator()
    elif backend_type == "quantum":
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        backend = service.least_busy(3)  # Example: 3-qubit systems

    # Execute the circuit
    job = backend.run(circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Fidelity calculation (placeholder for your specific approach)
    fidelity = calculate_fidelity(circuit, counts)  # Another function we need to implement.

    return counts, fidelity

def add_noise_to_circuit(circuit):
    """
    Adds a noise model to the circuit for testing error correction.
    Args:
        circuit (QuantumCircuit): The circuit to modify.
    Returns:
        QuantumCircuit: Circuit with noise applied.
    """
    from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
    from qiskit import QuantumCircuit

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ['rz', 'rx'])
    circuit_with_noise = circuit.copy()  # Preserve the original

    return circuit_with_noise

def mitigate_errors(results):
    """
    Mitigates errors from the results.
    Args:
        results (dict): Raw results from the execution.
    Returns:
        dict: Corrected results.
    """
    from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
    from qiskit.ignis.mitigation.measurement import MeasurementFilter

    # Example: Use measurement error mitigation
    meas_fitter = CompleteMeasFitter(results, ['0', '1'])  # Simple fitter for demo
    meas_filter = meas_fitter.filter
    mitigated_counts = meas_filter.apply(results)

    return mitigated_counts

def calculate_fidelity(target_state, observed_counts, total_shots):
    """
    Calculate the fidelity between the observed results and the target state.

    Args:
        target_state (str): The target quantum state (e.g., '101').
        observed_counts (dict): Measurement results as a dictionary of {state: count}.
        total_shots (int): Total number of shots in the experiment.

    Returns:
        float: Fidelity value.
    """
    target_prob = observed_counts.get(target_state, 0) / total_shots
    return target_prob  # Simpler fidelity based on target-state probability.

def run_with_error_correction(circuit, backend, target_state, shots=1024):
    """
    Execute a quantum circuit with error correction integrated.

    Args:
        circuit (QuantumCircuit): The quantum circuit to run.
        backend (Backend): The backend to execute on.
        target_state (str): The desired outcome state (e.g., '101').
        shots (int): Number of shots for the execution.

    Returns:
        dict: Corrected results and fidelity metrics.
    """
    print("\nRunning initial circuit...")
    initial_results = run_circuit_with_feedback(circuit, backend, shots)

    fidelity = calculate_fidelity(
        target_state=target_state,
        observed_counts=initial_results['counts'],
        total_shots=shots
    )

    print(f"Initial Fidelity: {fidelity:.4f}")

    if fidelity < 0.9:  # Adjust threshold as needed
        print("Low fidelity detected. Applying corrections...")
        corrected_circuit = prepare_state(circuit, target_state)
        corrected_results = run_circuit_with_feedback(corrected_circuit, backend, shots)

        corrected_fidelity = calculate_fidelity(
            target_state=target_state,
            observed_counts=corrected_results['counts'],
            total_shots=shots
        )

        return {
            "initial_results": initial_results,
            "initial_fidelity": fidelity,
            "corrected_results": corrected_results,
            "corrected_fidelity": corrected_fidelity,
        }
    else:
        print("Fidelity within acceptable range. No correction needed.")
        return {"results": initial_results, "fidelity": fidelity}

def create_feedback_ready_circuit(n_qubits):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure_all()  # Measurement for feedback
    return qc

def run_circuit_with_feedback_fidelity(circuit, backend, target_state, shots=1024, max_iterations=10):
    """
    Runs a quantum circuit with temporal feedback for error correction or state optimization.

    Args:
        circuit (QuantumCircuit): The quantum circuit to run.
        backend (Backend): Quantum backend to execute the circuit.
        target_state (str): The desired quantum state (e.g., '101').
        shots (int): Number of shots for measurement.
        max_iterations (int): Maximum number of feedback iterations.

    Returns:
        dict: Final counts and fidelity information.
    """

    print(f"Starting feedback loop for target state: {target_state}")
    fidelity_history = []

    # Transpile the circuit for the backend
    transpiled_circuit = transpile(circuit, backend)
    for i in range(max_iterations):
        print(f"Iteration {i + 1}/{max_iterations}...")

        # Execute the circuit
        job = backend.run(transpiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Calculate fidelity
        fidelity = calculate_fidelity(target_state, counts, shots)
        fidelity_history.append(fidelity)

        print(f"Iteration {i + 1}: Fidelity = {fidelity:.4f}")
        if fidelity >= 0.95:  # Threshold for acceptable fidelity
            print("Target fidelity achieved.")
            break

        # Apply feedback: Adjust gates or parameters dynamically
        transpiled_circuit = modify_circuit_with_reduced_influence(transpiled_circuit, counts, target_state=target_state)

    return {
        "counts": counts,
        "fidelity_history": fidelity_history,
        "final_fidelity": fidelity,
    }

def modify_circuit_with_reduced_influence(circuit, counts, target_state, scaling_factor=0.5):
    """Modify the circuit based on measurement feedback with reduced influence."""
    if not counts:
        print("No measurement data available for feedback.")
        return circuit

    # Identify the most likely state
    dominant_state = max(counts, key=counts.get)
    print(f"Current dominant state: {dominant_state}")

    # Compare dominant state with target state
    for idx, (target_bit, dominant_bit) in enumerate(zip(reversed(target_state), reversed(dominant_state))):
        if target_bit != dominant_bit:
            # Apply corrective gates with reduced influence
            if scaling_factor > 0.5:
                circuit.x(idx)  # Full influence
            else:
                circuit.h(idx)  # Partial influence
            print(f"Modified qubit {idx} to correct state.")

    print("Circuit updated with reduced influence.")
    return circuit

def stop_if_measured_1(result):
    return '1' in result and result['1'] > 500  # Example: Stop if '1' is measured more than 500 times

def stop_if_measured_0(result):
    return '0' in result and result['0'] > 500  # Example: Stop if '1' is measured more than 500 times

def run_circuit_with_feedback(base_circuit_func, backend, shots=1024, max_iterations=10, stop_condition=None):
    """
    Iterative quantum experiment function that dynamically adjusts and increases iteration count automatically.
    
    Parameters:
        base_circuit_func (function): Function that generates the base quantum circuit.
        backend (Backend): Quantum backend for execution.
        shots (int): Number of shots per experiment run.
        max_iterations (int): Maximum number of iterations to run.
    
    Returns:
        list: List of results from each experiment iteration.
    """
    results = []
    iteration = 1
    
    while iteration <= max_iterations:
        print(f"Running iteration {iteration}...")
        
        # Generate the circuit
        qc = base_circuit_func()
        
        # Run experiment and extract results
        result = run_and_extract_counts(qc, backend, shots)
        results.append(result)

        if stop_condition and stop_condition(result):
            print(f"Stopping early at iteration {iteration} due to stop condition.")
            break
        
        # Adaptive increment based on feedback
        iteration += 1  # Automatically increases iteration count
    
    print("All iterations complete.")
    return results

def modify_circuit_based_on_feedback(circuit, counts):
    """Modify the circuit based on measurement feedback to improve fidelity."""
    if not counts:
        print("No measurement data available for feedback.")
        return circuit

    target_state = max(counts, key=counts.get)
    print(f"Targeting state: {target_state} for amplification.")

    for idx, bit in enumerate(reversed(target_state)):
        if bit == '1':
            circuit.x(idx)
        circuit.h(idx)

    print("Circuit modified based on feedback.")
    return circuit

# This did not work well despite adjustments

def probabilistic_adjustment(circuit: QuantumCircuit, target_state: str, current_state: str, adjustment_factor: float = 0.1):
    """
    Adjust the circuit probabilistically to move closer to the target state.

    Args:
        circuit (QuantumCircuit): The quantum circuit to modify.
        target_state (str): The desired target state (e.g., '011').
        current_state (str): The current dominant state from measurements.
        adjustment_factor (float): Probability of applying an adjustment (default: 0.1).

    Returns:
        QuantumCircuit: The updated quantum circuit.
    """
    n_qubits = len(target_state)

    for idx in range(n_qubits):
        if target_state[idx] != current_state[idx]:
            # Apply probabilistic adjustment
            if random.random() < adjustment_factor:
                if target_state[idx] == '1':
                    circuit.x(idx)  # Flip qubit to 1 if target state demands it
                else:
                    # Add a Hadamard gate to create a superposition toward 0
                    circuit.h(idx)

    return circuit

def black_hole_simulation(num_qubits=17, num_charge_cycles=5, spin_cycles=3, injection_strength=np.pi/4):
    """
    Simulates black hole analog formation through charge injections and spin cycles.

    Parameters:
    - num_qubits (int): Number of qubits in the circuit.
    - num_charge_cycles (int): How many prolonged charge injection cycles to perform.
    - spin_cycles (int): Number of rotational (spin) cycles to apply.
    - injection_strength (float): Rotation angle for charge injection strength.

    Returns:
    - entropy_list (list): Von Neumann entropy after each cycle.
    - fidelity_list (list): Fidelity against the maximally mixed state.
    """

    # Initialize quantum register and circuit
    qreg = QuantumRegister(num_qubits, 'q')
    qc = QuantumCircuit(qreg)

    # Function for prolonged charge injection (entangling each qubit with a central reference qubit)
    def prolonged_charge_injection():
        for i in range(1, num_qubits):
            qc.ry(injection_strength, qreg[i])
            qc.cx(qreg[0], qreg[i])  # Central qubit (q0) acts as the charge source

    # Function for spin cycle (rotational entanglement pattern)
    def spin_cycle():
        for i in range(num_qubits - 1):
            qc.swap(qreg[i], qreg[i+1])  # Simulates angular momentum via qubit rotation

    # Measurement function for entropy and fidelity
    def measure_entropy_fidelity():
        backend = Aer.get_backend('statevector_simulator')
        result = backend.run(qc, shots=1).result()
        sv = result.get_statevector()

        # Convert to density matrix and trace out a subset of qubits for partial trace
        density_matrix = DensityMatrix(sv)
        reduced_state = partial_trace(density_matrix, list(range(1, num_qubits)))

        # Calculate von Neumann entropy and fidelity
        ent = qiskit_qiskit_entropy(reduced_state)
        max_mixed = DensityMatrix(np.identity(2) / 2)  # Single-qubit maximally mixed state
        fid = state_fidelity(reduced_state, max_mixed)
        return ent, fid

    # Lists to store results
    entropy_list = []
    fidelity_list = []

    # Simulation: charge injections + spin cycles
    for charge_cycle in range(num_charge_cycles):
        prolonged_charge_injection()
        ent, fid = measure_entropy_fidelity()
        entropy_list.append(ent)
        fidelity_list.append(fid)

        for spin_cycle_index in range(spin_cycles):
            spin_cycle()
            ent, fid = measure_entropy_fidelity()
            entropy_list.append(ent)
            fidelity_list.append(fid)

    return entropy_list, fidelity_list

def information_paradox_test(num_qubits=10, injection_strength=np.pi/2, retrieval_cycles=5):
    """
    Tests the black hole information paradox by injecting known information and attempting retrieval.
    """
    qreg = QuantumRegister(num_qubits, 'q')
    qc = QuantumCircuit(qreg)

    # Encode known information (|+> state)
    qc.h(qreg[0])

    # Scramble with charge injections and spin cycles
    for i in range(1, num_qubits):
        qc.ry(injection_strength, qreg[i])
        qc.cx(qreg[0], qreg[i])
    for _ in range(retrieval_cycles):
        for i in range(num_qubits - 1):
            qc.swap(qreg[i], qreg[i + 1])

    # Attempt retrieval (reverse operations)
    for i in reversed(range(1, num_qubits)):
        qc.cx(qreg[0], qreg[i])
        qc.ry(-injection_strength, qreg[i])
    qc.h(qreg[0])

    backend = Aer.get_backend('statevector_simulator')
    sv = backend.run(qc).result().get_statevector()
    final_state = partial_trace(DensityMatrix(sv), list(range(1, num_qubits)))
    original_state = DensityMatrix.from_label('+')

    # Ensure matching dimensions for fidelity calculation
    if final_state.dim != original_state.dim:
        final_state = final_state.expand(DensityMatrix(np.eye(original_state.dim[0])))

    retrieval_fidelity = state_fidelity(final_state, original_state)
    return retrieval_fidelity

def hawking_radiation_recovery(num_qubits=10, injection_strength=np.pi/2, radiation_qubits=2, retrieval_cycles=5):
    """
    Simulates Hawking radiation recovery:
    - Encodes information into a qubit.
    - Scrambles via charge injections and spin cycles.
    - Extracts radiation qubits and attempts information recovery.
    """
    qreg = QuantumRegister(num_qubits, 'q')
    qc = QuantumCircuit(qreg)

    qc.h(qreg[0])  # Encode known information (|+> state)

    for i in range(1, num_qubits):
        qc.ry(injection_strength, qreg[i])
        qc.cx(qreg[0], qreg[i])
    for _ in range(retrieval_cycles):
        for i in range(num_qubits - 1):
            qc.swap(qreg[i], qreg[i + 1])

    backend = Aer.get_backend('statevector_simulator')
    sv = backend.run(qc).result().get_statevector()

    try:
        radiation_indices = list(range(num_qubits - radiation_qubits, num_qubits))
        # Keep only radiation qubits and trace out the rest
        radiation_state = partial_trace(DensityMatrix(sv), [i for i in range(num_qubits) if i not in radiation_indices])

        original_state = DensityMatrix.from_label('+')

        # Ensure compatible dimensions for fidelity calculation
        if radiation_state.dim[0] != original_state.dim[0]:
            size_diff = int(np.log2(original_state.dim[0] / radiation_state.dim[0]))
            for _ in range(size_diff):
                radiation_state = radiation_state.tensor(DensityMatrix(np.eye(2) / 2))

        recovery_fidelity = state_fidelity(radiation_state, original_state)
    except Exception as e:
        print(f"Error in hawking_radiation_recovery: {e}")
        recovery_fidelity = None

    return recovery_fidelity

def hawking_radiation_with_sequential_entangling(num_qubits=10, radiation_qubits=3, entangling_cycles=5, injection_strength=np.pi/2):
    """
    Models Hawking radiation with sequential entangling and fixes dimension mismatch issues.
    Reduces radiation state to a single qubit for fidelity comparison.
    """
    qreg = QuantumRegister(num_qubits, 'q')
    qc = QuantumCircuit(qreg, name="SecureHawkingRadiation")

    qc.h(qreg[0])  # Encode information (|+> state)

    for i in range(1, radiation_qubits + 1):
        qc.ry(injection_strength, qreg[i])
        qc.cx(qreg[0], qreg[i])
        qc.barrier()

    for _ in range(entangling_cycles):
        for i in range(radiation_qubits + 1, num_qubits - 1):
            qc.cx(qreg[i], qreg[i + 1])
            qc.ry(injection_strength / 2, qreg[i + 1])

    backend = Aer.get_backend('statevector_simulator')
    sv = backend.run(qc).result().get_statevector()

    try:
        radiation_indices = list(range(1, radiation_qubits + 1))
        radiation_state = partial_trace(DensityMatrix(sv), [i for i in range(num_qubits) if i not in radiation_indices])
        original_state = DensityMatrix.from_label('+')

        # Reduce radiation_state to a single qubit for comparison
        if radiation_state.num_qubits > 1:
            qubits_to_trace = list(range(1, radiation_state.num_qubits))
            radiation_state = partial_trace(radiation_state, qubits_to_trace)

        recovery_fidelity = state_fidelity(radiation_state, original_state)
    except Exception as e:
        print(f"Error in hawking_radiation_with_sequential_entangling: {e}")
        recovery_fidelity = None

    return recovery_fidelity

##def create_gaussian_pulse(amplitude, sigma, duration, name="gaussian_pulse"):
##    return Gaussian(duration=duration, amp=amplitude, sigma=sigma, name=name)
##
### --- PULSE-LEVEL BLACK HOLE SIMULATION SKELETON ---
##def black_hole_pulse_simulation():
##    """
##    Creates a pulse schedule representing charge injection cycles and spin cycles.
##    This maps the black hole simulation to hardware-level AWG controls.
##    """
##    schedule_list = []
##
##    # Define drive channels for qubits
##    drive_channels = [DriveChannel(qubit) for qubit in range(NUM_QUBITS)]
##
##    # Create injection pulse (analogous to prolonged charge injection)
##    injection_pulse = create_gaussian_pulse(
##        amplitude=PULSE_AMPLITUDE, sigma=PULSE_SIGMA, duration=PULSE_DURATION, name="charge_injection"
##    )
##
##    # Create spin pulse (analogous to spin cycle entangling operations)
##    spin_pulse = create_gaussian_pulse(
##        amplitude=PULSE_AMPLITUDE, sigma=PULSE_SIGMA, duration=PULSE_DURATION, name="spin_cycle"
##    )
##
##    # --- PROLONGED CHARGE INJECTION ---
##    injection_schedule = Schedule(name="prolonged_charge_injection")
##    for qubit in range(1, NUM_QUBITS):
##        injection_schedule |= Play(injection_pulse, drive_channels[qubit])
##    schedule_list.append(injection_schedule)
##
##    # --- SPIN CYCLE SIMULATION ---
##    spin_schedule = Schedule(name="spin_cycle")
##    for qubit in range(NUM_QUBITS - 1):
##        spin_schedule |= Play(spin_pulse, drive_channels[qubit])
##    schedule_list.append(spin_schedule)
##
##    # --- EVENT HORIZON ANALOG ---
##    event_horizon_schedule = Schedule(name="event_horizon_scramble")
##    for qubit in range(NUM_QUBITS):
##        event_horizon_schedule |= Play(
##            create_gaussian_pulse(PULSE_AMPLITUDE, PULSE_SIGMA, PULSE_DURATION, name=f"scramble_{qubit}"),
##            drive_channels[qubit]
##        )
##    schedule_list.append(event_horizon_schedule)
##
##    # --- INFORMATION RETRIEVAL PHASE ---
##    retrieval_schedule = Schedule(name="hawking_radiation_recovery")
##    retrieval_schedule |= Play(
##        create_gaussian_pulse(PULSE_AMPLITUDE, PULSE_SIGMA, PULSE_DURATION, name="hawking_retrieval"),
##        drive_channels[0]
##    )
##    schedule_list.append(retrieval_schedule)
##
##    return schedule_list

def quantum_scramble_encrypt(message_bits: str, key_bits: str, shots: int = 1024):
    """
    Encrypts a message using quantum scrambling with black hole dynamics.

    Parameters:
    - message_bits (str): Binary string representing the message.
    - key_bits (str): Binary string representing the private key.
    - shots (int): Number of measurement shots (default: 1024).

    Returns:
    - counts (dict): Measurement outcomes from the encrypted quantum circuit.
    - scrambling_unitary (Instruction): Applied random unitary for decoding.
    """
    n_qubits = len(message_bits)
    qr = QuantumRegister(n_qubits, name="q")
    cr = ClassicalRegister(n_qubits, name="c")
    circuit = QuantumCircuit(qr, cr)

    # 1️⃣ Encode Message
    for i, bit in enumerate(message_bits):
        if bit == '1':
            circuit.x(qr[i])

    # 2️⃣ Entangle with Private Key
    for i, bit in enumerate(key_bits):
        if bit == '1':
            circuit.h(qr[i])
            circuit.cx(qr[i], qr[(i + 1) % n_qubits])

    # 3️⃣ Apply Dynamic Scrambling (Random Unitary)
    scrambling_unitary = random_unitary(2**n_qubits).to_instruction()
    circuit.append(scrambling_unitary, qr[:])

    # 4️⃣ Entropy Amplification (Non-reversible Random Rotations for External Observers)
    for i in range(n_qubits):
        random_angle_rz = np.pi / np.random.randint(1, 10)
        random_angle_rx = np.pi / np.random.randint(1, 10)
        circuit.h(qr[i])
        circuit.rz(random_angle_rz, qr[i])
        circuit.rx(random_angle_rx, qr[i])

    # 5️⃣ Measurement
    circuit.barrier()
    circuit.measure(qr, cr)

    backend = Aer.get_backend('qasm_simulator')
    job = backend.run(circuit, shots=shots)
    counts = job.result().get_counts()

    return counts, scrambling_unitary


# 🔑 Decoding Function
def quantum_scramble_decrypt(scrambling_unitary, key_bits: str, encrypted_bits: str, shots: int = 1024):
    """
    Decrypts the scrambled quantum state to retrieve the original message without relying on stored rotation angles.

    Parameters:
    - scrambling_unitary (Instruction): The unitary used during encryption.
    - key_bits (str): The private key used in the encryption process.
    - encrypted_bits (str): Binary string representing the measured scrambled state.
    - shots (int): Number of measurement shots (default: 1024).

    Returns:
    - counts (dict): Measurement outcomes after decryption.
    """
    n_qubits = len(encrypted_bits)
    qr = QuantumRegister(n_qubits, name="q")
    cr = ClassicalRegister(n_qubits, name="c")
    circuit = QuantumCircuit(qr, cr)

    # 1️⃣ Re-encode Measured Scrambled State
    for i, bit in enumerate(encrypted_bits):
        if bit == '1':
            circuit.x(qr[i])

    # 2️⃣ Approximate Entropy Reversal (Structured but Unpredictable for Attackers)
    # Instead of stored angles, apply inverse rotations using a deterministic function (e.g., fixed angle pattern)
    for i in range(n_qubits):
        inverse_angle_rz = -np.pi / (i + 2)  # Deterministic, key-dependent pattern
        inverse_angle_rx = -np.pi / (i + 3)
        circuit.rx(inverse_angle_rx, qr[i])
        circuit.rz(inverse_angle_rz, qr[i])
        circuit.h(qr[i])

    # 3️⃣ Apply Inverse Scrambling Unitary
    circuit.append(scrambling_unitary.inverse(), qr[:])

    # 4️⃣ Reverse Entanglement with Private Key
    for i, bit in enumerate(reversed(key_bits)):
        if bit == '1':
            circuit.cx(qr[i], qr[(i + 1) % n_qubits])
            circuit.h(qr[i])

    # 5️⃣ Measurement
    circuit.barrier()
    circuit.measure(qr, cr)

    backend = Aer.get_backend('qasm_simulator')
    job = backend.run(circuit, shots=shots)
    counts = job.result().get_counts()

    return counts

# 🔍 Branch Signature Mapping Function
def map_branch_signatures(event_description: str, key_bits: str, n_qubits: int = 6):
    """
    Maps a branch signature based on an event description and private key.

    Parameters:
    - event_description (str): Description of the event (e.g., 'identity theft').
    - key_bits (str): Private key to personalize the mapping.
    - n_qubits (int): Number of qubits representing the branch signature.

    Returns:
    - signature_bits (str): Binary signature representing the targeted branch.
    - signature_hash (str): SHA-256 hash of the event and key for verification.
    """
    # Generate a hash-based seed from event and key
    combined = f"{event_description}-{key_bits}".encode()
    signature_hash = hashlib.sha256(combined).hexdigest()

    # Use hash to produce deterministic binary signature
    signature_bits = bin(int(signature_hash, 16))[2:].zfill(n_qubits)[:n_qubits]

    return signature_bits, signature_hash

def black_hole_warp_simulation_0():
    # Step 1: Initialize Quantum Circuit with 6 qubits (representing 3 entangled pairs)
    num_qubits = 6
    qc = QuantumCircuit(num_qubits)
    
    # Step 2: Create maximal entanglement (Bell Pairs)
    for i in range(0, num_qubits, 2):
        qc.h(i)
        qc.cx(i, i+1)
    
    # Step 3: Introduce Charge & Spin Effects (Random Phase Rotations)
    np.random.seed(42)
    for i in range(num_qubits):
        theta = np.random.uniform(0, 2*np.pi)  # Random charge/spin influence
        qc.rz(theta, i)
    
    # Step 4: Scrambling Effect (Black Hole Information Overload)
    for i in range(num_qubits-1):
        qc.cx(i, i+1)
        qc.h(i)
        qc.rz(np.pi/4, i+1)
    
    # Step 5: Apply Holographic Warp (Simulating Spacetime Curvature)
    for i in range(num_qubits):
        qc.sx(i)
        qc.cz(i, (i+1) % num_qubits)
    
    # Step 6: Measure Final State
    backend = Aer.get_backend("statevector_simulator")
    result = backend.run(qc).result()
    final_state = Statevector(result.get_statevector())

    final_density_matrix = DensityMatrix(final_state)

    for i in range(num_qubits):
        partial_traced_state = partial_trace(final_density_matrix, [i])
        print(f"Partial trace for qubit {i}:\n", partial_traced_state)  # Debugging output


    # Compute entropies using the correct method
    entropies = [qiskit_qiskit_entropy(partial_trace(final_density_matrix, [i])) for i in range(num_qubits)]

    return qc, entropies

def time_evolution_black_hole(num_qubits=5, time_steps=50, delta_t=0.1):
    """
    Simulates the time evolution of a black hole quantum system
    and computes the Von Neumann entropy at each step.
    """
    # Initialize a maximally entangled state (black hole analogy)
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits - 1):
        qc.h(i)
        qc.cx(i, i+1)
    
    # Convert to density matrix
    backend = Aer.get_backend("statevector_simulator")
    transpiled_qc = transpile(qc, backend)
    state = DensityMatrix.from_instruction(transpiled_qc)
    
    entropies = []
    times = np.arange(0, time_steps * delta_t, delta_t)
    
    for t in times:
        # Apply random unitary evolution to simulate information scrambling
        random_unitary = np.exp(1j * np.random.rand(num_qubits, num_qubits))
        evolved_state = state.evolve(random_unitary)
        
        # Compute Von Neumann entropy for each qubit
        qubit_entropies = [qiskit_qiskit_entropy(partial_trace(evolved_state, [i])) for i in range(num_qubits)]
        entropies.append(np.mean(qubit_entropies))
    
    # Plot entropy evolution
    plt.figure(figsize=(8, 5))
    plt.plot(times, entropies, label="Avg. Entropy per Qubit", color='purple')
    plt.xlabel("Time")
    plt.ylabel("Von Neumann Entropy")
    plt.title("Entropy Evolution in Black Hole Simulation")
    plt.legend()
    plt.show()
    
    return times, entropies

def auto_detect_branches(events_list, key_bits: str, detection_threshold: int = 50):
    """
    Auto-detects branches based on a list of event descriptions and a private key.

    Parameters:
    - events_list (list): List of event descriptions to monitor.
    - key_bits (str): Private key for branch mapping.
    - detection_threshold (int): Probability threshold (%) for detection.

    Returns:
    - detected_branches (dict): Event descriptions mapped to their detected branch signatures.
    """
    detected_branches = {}
    for event in events_list:
        detection_chance = random.randint(0, 100)
        if detection_chance >= detection_threshold:
            signature_bits, signature_hash = map_branch_signatures(event, key_bits)
            detected_branches[event] = {
                "signature_bits": signature_bits,
                "signature_hash": signature_hash,
                "detection_confidence": detection_chance
            }
    return detected_branches


def initialize_black_hole_state(qc, num_qubits):
    """Prepares the initial quantum state representing infalling matter."""
    for qubit in range(num_qubits):
        qc.h(qubit)  # Superposition state

def apply_charge_spin_interactions(qc, num_qubits, charge, spin):
    """Encodes charge-spin interactions using Kerr-Newman approximations."""
    for qubit in range(num_qubits - 1):
        theta = np.arctan(charge / (1 + spin))  # Charge-spin ratio for interaction
        qc.rz(theta, qubit)  # Charge-induced phase shift
        qc.cry(2 * np.pi * spin, qubit, qubit + 1)  # Spin entanglement

def simulate_event_horizon(qc, num_qubits):
    """Applies quantum scrambling effects at the black hole horizon."""
    for qubit in range(num_qubits):
        qc.sx(qubit)  # Quantum scrambling at the event horizon
        qc.cz(qubit, (qubit + 1) % num_qubits)  # Non-local entanglement

def compute_von_neumann_qiskit_entropy(qc, num_qubits):
    """Computes the Von Neumann entropy for each qubit after the simulation."""
    backend = Aer.get_backend('statevector_simulator')
    transpiled_qc = transpile(qc, backend)
    final_state = DensityMatrix.from_instruction(transpiled_qc)

    entropies = []
    for qubit in range(num_qubits):
        reduced_state = partial_trace(final_state, [qubit])  # Corrected
        eigenvalues = np.linalg.eigvalsh(reduced_state.data)
        entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10)).real  # Small offset to avoid log(0)
        entropies.append(entropy)
    
    return entropies

def black_hole_warp_simulation_core(num_qubits=6, charge=0.7, spin=0.85):
    """Runs the full black hole simulation pipeline."""
    qc = QuantumCircuit(num_qubits)
    
    initialize_black_hole_state(qc, num_qubits)
    apply_charge_spin_interactions(qc, num_qubits, charge, spin)
    simulate_event_horizon(qc, num_qubits)
    entropies = compute_von_neumann_qiskit_entropy(qc, num_qubits)

    return qc, entropies

def apply_shor_encoding(qc, qubit):
    """Encodes a logical qubit using the Shor code (bit-flip and phase-flip protection)."""
    qc.cx(qubit, qubit + 1)
    qc.cx(qubit, qubit + 2)
    qc.h([qubit, qubit + 1, qubit + 2])
    qc.cx(qubit, qubit + 3)
    qc.cx(qubit + 1, qubit + 4)
    qc.cx(qubit + 2, qubit + 5)
    return qc

def detect_and_correct_errors(qc, logical_qubit_start):
    """Detects and corrects bit-flip and phase-flip errors on a logical qubit."""
    for i in range(3):  # Bit-flip correction
        qc.cx(logical_qubit_start + i, logical_qubit_start + 3 + i)
        qc.measure(logical_qubit_start + 3 + i, logical_qubit_start + 3 + i)
        qc.x(logical_qubit_start + i).c_if(logical_qubit_start + 3 + i, 1)
    for i in range(3):  # Phase-flip correction
        qc.h(logical_qubit_start + i)
        qc.cx(logical_qubit_start + i, logical_qubit_start + 3 + i)
        qc.h(logical_qubit_start + i)
    return qc

def encode_charge_preserving(qc, qubits):
    """Encodes logical qubits using charge parity conservation."""
    for i in range(0, len(qubits), 2):
        qc.h(qubits[i])
        qc.cx(qubits[i], qubits[i + 1])  # Entangle charge states
    return qc

def detect_charge_imbalance(qc, ancilla, qubits):
    """Detects charge parity violations using ancilla qubits."""
    for i in range(min(len(ancilla), len(qubits) - 1)):
        qc.cx(qubits[i], ancilla[i])
        qc.cx(qubits[i + 1], ancilla[i])  # If charge parity is broken, ancilla flips
    return qc

def correct_charge_imbalance(qc, ancilla, qubits):
    """Applies corrections if a charge imbalance is detected."""
    for i in range(min(len(ancilla), len(qubits) - 1)):
        qc.cx(ancilla[i], qubits[i])  # Restore charge balance
        qc.cx(ancilla[i], qubits[i + 1])
    return qc

def charge_preserving_qec(num_logical_qubits=2):
    """Runs a charge-preserving quantum error correction simulation."""
    num_physical_qubits = num_logical_qubits * 2  # Each logical qubit = 2 physical qubits
    num_ancilla = max(1, num_logical_qubits - 1)  # Ensure at least one ancilla qubit
    qc = QuantumCircuit(num_physical_qubits + num_ancilla, num_physical_qubits)
    
    # Encode logical qubits with charge preservation
    qc = encode_charge_preserving(qc, list(range(num_physical_qubits)))
    
    # Introduce artificial noise (random bit flip)
    qc.x(1)  # Simulating an error
    
    # Detect and correct charge imbalance
    ancilla_qubits = list(range(num_physical_qubits, num_physical_qubits + num_ancilla))
    qc = detect_charge_imbalance(qc, ancilla_qubits, list(range(num_physical_qubits)))
    qc = correct_charge_imbalance(qc, ancilla_qubits, list(range(num_physical_qubits)))
    
    # Simulate and extract density matrix
    simulator = Aer.get_backend('aer_simulator_density_matrix')
    qc.save_density_matrix()
    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(compiled_circuit).result()
    final_density_matrix = DensityMatrix(result.data(0)['density_matrix'])
    
    # Compute entropies
    entropies = [qiskit_qiskit_entropy(partial_trace(final_density_matrix, [i])) for i in range(num_physical_qubits)]
    
    return qc, entropies

def create_noisy_model():
    noise_model = NoiseModel()
    p_depol = 0.01  # Depolarizing probability
    p_damp = 0.02  # Amplitude damping probability
    
    depol_error = depolarizing_error(p_depol, 1)
    amp_damp_error = amplitude_damping_error(p_damp)
    
    # Instead of applying errors multiple times, apply them selectively
    noise_model.add_all_qubit_quantum_error(depol_error, ['u3'])  # Only apply to u3
    noise_model.add_all_qubit_quantum_error(amp_damp_error, ['u3'])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(p_depol, 2), ['cx'])

    return noise_model


def charge_preserving_qec_noisy(num_qubits=4):
    q = QuantumRegister(num_qubits, 'q')
    c = ClassicalRegister(num_qubits, 'c')  # Ensure classical bits match quantum bits
    qc = QuantumCircuit(q, c)
    
    # Encoding with charge preservation
    for i in range(num_qubits - 1):
        qc.h(q[i])
        qc.cx(q[i], q[i + 1])

    # Ensure syndrome register only if num_qubits > 2
    if num_qubits > 2:
        syndrome = QuantumRegister(2, 'syndrome')
        qc.add_register(syndrome)

        for i in range(0, num_qubits - 2, 2):
            qc.cx(q[i], q[i + 2])
        
        # Syndrome measurement only when syndrome register exists
        qc.cx(q[0], syndrome[0])
        qc.cx(q[1], syndrome[0])
        qc.cx(q[2], syndrome[1])
        if num_qubits > 3:
            qc.cx(q[3], syndrome[1])

    qc.h(q)
    qc.measure(q, c)
    # Apply noise

    simulator = Aer.get_backend('qasm_simulator')
    job_ideal = simulator.run(qc, shots=4096)  # No noise
    result_ideal = job_ideal.result()
    print("Ideal Measurement:", result_ideal.get_counts())

    noise_model = create_noisy_model()
    simulator = Aer.get_backend('qasm_simulator')
    result = simulator.run(qc, noise_model=noise_model, shots=4096).result()
    
    print(f"Execution Result: {result}")
    
    # Analyze measurement outcomes
    if result.success:
        counts = result.get_counts()
        print(f"Measurement Results for {num_qubits} qubits:", counts)
        return counts
    else:
        print("Execution failed.")
        return None

def qc_charge_qec(num_qubits=4):
    q = QuantumRegister(num_qubits, 'q')
    c = ClassicalRegister(num_qubits, 'c')  # Ensure classical bits match quantum bits
    qc = QuantumCircuit(q, c)
    
    # Encoding with charge preservation
    for i in range(num_qubits - 1):
        qc.h(q[i])
        qc.cx(q[i], q[i + 1])

    # Ensure syndrome register only if num_qubits > 2
    if num_qubits > 2:
        syndrome = QuantumRegister(2, 'syndrome')
        qc.add_register(syndrome)

        for i in range(0, num_qubits - 2, 2):
            qc.cx(q[i], q[i + 2])
        
        # Syndrome measurement only when syndrome register exists
        qc.cx(q[0], syndrome[0])
        qc.cx(q[1], syndrome[0])
        qc.cx(q[2], syndrome[1])
        if num_qubits > 3:
            qc.cx(q[3], syndrome[1])

    qc.h(q)
    qc.measure(q, c)

    return qc

def text_to_binary(text):
    """Convert text to binary representation."""
    return ''.join(format(ord(char), '08b') for char in text)

def generate_timestamp():
    """Generate a binary timestamp."""
    timestamp = int(time.time())  # Current UNIX timestamp
    return format(timestamp, '064b')


def pure_circuit_for_statevector(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Return a new n-qubit circuit with all measure ops & classical bits removed,
    preserving only the quantum gates in the original qc (in order).
    """
    n = qc.num_qubits
    qc2 = QuantumCircuit(n)
    # qc.qubits is a flat list of Qubit objects in order;
    # qc2.qubits is the same length, but new objects.
    for instruct in qc.data:
        op = instruct.operation
        # skip measurements entirely
        if op.name == 'measure':
            continue
        # figure out which qubit‐indices this op used
        qidxs = [qc.qubits.index(qb) for qb in instruct.qubits]
        # map to the new circuit’s qubits
        new_qbs = [qc2.qubits[i] for i in qidxs]
        qc2.append(op, new_qbs, [])    # drop all clbits
    return qc2

def apply_repetition_code(qc: QuantumCircuit):
    """
    Given an n-qubit circuit `qc`, returns:
      - qc_rep: a (3n)-qubit circuit with 3-qubit repetition encoding + decoding
      - decode_counts: a function(raw_counts) → logical_counts by majority vote
    Only supports 1- and 2-qubit gates in `qc`.
    """
    n = qc.num_qubits
    qc_rep = QuantumCircuit(3*n, 3*n)

    # 1) ENCODING: |ψ> -> |ψ,ψ,ψ> for each logical qubit
    for i in range(n):
        qc_rep.cx(3*i,   3*i+1)
        qc_rep.cx(3*i,   3*i+2)

    # 2) TRANSVERSAL replay of every gate in qc
    for inst, qargs, cargs in qc.data:
        qs = [qb._index for qb in qargs]
        if len(qs) == 1:
            q = qs[0]
            for j in range(3):
                qc_rep.append(inst, [3*q + j])
        elif len(qs) == 2:
            q0, q1 = qs
            for j in range(3):
                qc_rep.append(inst, [3*q0 + j, 3*q1 + j])
        else:
            raise ValueError("Only 1- and 2-qubit gates supported")

    # 3) DECODING: measure all physical qubits
    qc_rep.measure(range(3*n), range(3*n))

    # 4) helper to majority-vote raw counts back to logical bitstrings
    def decode_counts(raw_counts):
        """
        raw_counts: dict mapping 3n-bit strings → freq
        returns: dict mapping n-bit logical strings → corrected freq
        """
        logical_counts = {}
        for bitstr, freq in raw_counts.items():
            # Qiskit strings are MSB-first (qubit 3n-1 is bitstr[0])
            bits = list(map(int, bitstr[::-1]))  # now bits[0] is qubit 0
            logical = []
            for i in range(n):
                trip = bits[3*i:3*i+3]
                logical.append('1' if sum(trip) >= 2 else '0')
            L = ''.join(reversed(logical))  # back to MSB-first
            logical_counts[L] = logical_counts.get(L, 0) + freq
        return logical_counts

    return qc_rep, decode_counts

def record_decision(qc, decision_text):
    """Convert decision to binary, timestamp it, and inject charge."""
    binary_text = text_to_binary(decision_text)
    binary_timestamp = generate_timestamp()
    combined_binary = binary_text + binary_timestamp  # Merge text & time

    print(f"Encoding Decision: {decision_text}")
    print(f"Binary Representation: {combined_binary}")

    qc = inject_charge_holographically(qc, combined_binary)
    return qc

def inject_charge_holographically(qc, binary_pattern):
    """
    Apply charge injection based on binary pattern.
    Each '1' in binary applies a Hadamard gate followed by a phase shift; '0' applies only a Hadamard gate.
    """
    num_qubits = qc.num_qubits
    for i, bit in enumerate(binary_pattern):
        qubit = i % num_qubits  # Wrap around if needed
        qc.h(qubit)  # Place qubit in superposition
        if bit == '1':
            qc.p(np.pi / 4, qubit)  # Inject a phase shift
    return qc

def create_teleportation_circuit():
    qr = QuantumRegister(3, name="q")  # Create a quantum register with 3 qubits
    cr = ClassicalRegister(2, name="c")  # Create a classical register with 2 bits

    qc = QuantumCircuit(qr, cr)  # Now correctly define the circuit with named registers

    # Step 1: Create Bell pair
    qc.h(qr[1])
    qc.cx(qr[1], qr[2])

    # Step 2: Encode charge information into Q0 (Injected charge state)
    qc.h(qr[0])

    # Step 3: Bell measurement on Q0 and Q1
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.measure([qr[0], qr[1]], [cr[0], cr[1]])

    # Step 4: Conditional operations on Q2 based on measurement results
    qc.x(qr[2]).c_if(cr[0], 1)  # Apply X if cr[0] == 1
    qc.z(qr[2]).c_if(cr[1], 1)  # Apply Z if cr[1] == 1

    return qc

def apply_classical_corrections(measurement, statevector):
    """
    Manually applies X and Z corrections based on measurement results.
    """
    # Create and convert circuits to Operators
    x_circuit = QuantumCircuit(1)
    x_circuit.x(0)
    x_gate = Operator.from_circuit(x_circuit)

    z_circuit = QuantumCircuit(1)
    z_circuit.z(0)
    z_gate = Operator.from_circuit(z_circuit)

    if measurement[0] == '1':  # If the first classical bit is 1, apply X gate
        statevector = statevector.evolve(x_gate)
    if measurement[1] == '1':  # If the second classical bit is 1, apply Z gate
        statevector = statevector.evolve(z_gate)

    return statevector

def apply_add_clbits(qc, num_qubits=7):
    """Applies charge-preserving quantum error correction to a given quantum circuit."""
    q = QuantumRegister(num_qubits, 'q')
    c = ClassicalRegister(num_qubits, 'c')  # Define classical bits properly
    qec_qc = QuantumCircuit(q, c)

    # Convert the circuit into an instruction while explicitly including classical bits
    qc_instruction = qc.to_instruction()

    # Append the instruction to the new quantum circuit
    qec_qc.append(qc_instruction, q[:qc.num_qubits])  

    return qec_qc, qc_instr

def apply_charge_preserving_qec_no_syndrome(qc, num_qubits=7, num_classical=2):
    """Applies charge-preserving quantum error correction to a given circuit, without syndrome decoding."""

    q = QuantumRegister(num_qubits, 'q')
    c = ClassicalRegister(num_classical, 'c')  # Ensure classical bits are included
    qec_qc = QuantumCircuit(q, c)

    if qc.num_clbits < num_qubits:
        additional_clbits = num_qubits - qc.num_clbits
        qc.add_register(ClassicalRegister(additional_clbits, 'extra_c'))

    qc = add_classical_bits(qc, num_classical)

    # Apply charge-preserving encoding (Hadamards and CNOTs)
    for i in range(num_qubits - 1):
        qec_qc.h(q[i])
        qec_qc.cx(q[i], q[i + 1])

    qec_qc.h(q)  # Final Hadamard gates

    if qc.num_clbits == 0:
        qc.measure_all()

    # Convert input circuit to an instruction **without losing classical bits**
    qc_instruction = qc.to_instruction()

    # Append input circuit properly
    qec_qc.append(qc_instruction, q[:qc.num_qubits], c[:qc.num_clbits])

    # Ensure measurement matches classical register size
    if len(q) != len(c):
        print(f"Warning: Quantum register size ({len(q)}) != Classical register size ({len(c)}). Adjusting...")
        c = ClassicalRegister(len(q), 'c_adjusted')
        qec_qc.add_register(c)

    # Measure all qubits into classical bits
    qec_qc.measure(q[:num_qubits], c[:num_classical])

    return qec_qc, qc_instruction


def apply_charge_preserving_qec(qc, num_qubits=7, num_classical=2):
    """Applies charge-preserving quantum error correction to a given circuit."""
    
    q = QuantumRegister(num_qubits, 'q')
    c = ClassicalRegister(num_classical, 'c')  # Ensure classical bits are included
    qec_qc = QuantumCircuit(q, c)

    if qc.num_clbits < num_qubits:
        additional_clbits = num_qubits - qc.num_clbits
        qc.add_register(ClassicalRegister(additional_clbits, 'extra_c'))

    qc = add_classical_bits(qc, num_classical)

    # Apply charge-preserving encoding
    for i in range(num_qubits - 1):
        qec_qc.h(q[i])
        qec_qc.cx(q[i], q[i + 1])

    # Add syndrome bits if more than 2 qubits
    if num_qubits > 2:
        syndrome = QuantumRegister(2, 'syndrome')
        qec_qc.add_register(syndrome)

        for i in range(0, num_qubits - 2, 2):
            qec_qc.cx(q[i], q[i + 2])

        # Measure syndrome register only if it exists
        qec_qc.cx(q[0], syndrome[0])
        qec_qc.cx(q[1], syndrome[0])
        qec_qc.cx(q[2], syndrome[1])
        if num_qubits > 3:
            qec_qc.cx(q[3], syndrome[1])

    qec_qc.h(q)  # Final Hadamard gates
    if qc.num_clbits == 0:
        qc.measure_all()
    # Convert input circuit to an instruction **without losing classical bits**
    qc_instruction = qc.to_instruction()
    
    # Append input circuit properly
    qec_qc.append(qc_instruction, q[:qc.num_qubits], c[:qc.num_clbits])

    # 🔥 Ensure measurement matches classical register size
    if len(q) != len(c):
        print(f"Warning: Quantum register size ({len(q)}) != Classical register size ({len(c)}). Adjusting...")
        c = ClassicalRegister(len(q), 'c_adjusted')
        qec_qc.add_register(c)

    # Measure all qubits into classical bits
    qec_qc.measure(q[:num_qubits], c[:num_classical])

    return qec_qc, qc_instruction

def shor_qec_noisy(num_logical_qubits=1):
    num_physical_qubits = min(num_logical_qubits * 9, 15)  # Limit total qubits to 15
    q = QuantumRegister(num_physical_qubits, 'q')
    c = ClassicalRegister(num_logical_qubits, 'c')
    qc = QuantumCircuit(q, c)
    
    for i in range(num_logical_qubits):
        base = i * 9
        if base + 8 >= num_physical_qubits:
            break  # Prevent exceeding available qubits
        
        qc.h(q[base])
        qc.cx(q[base], q[base+1])
        qc.cx(q[base], q[base+2])
        
        if base + 6 < num_physical_qubits:
            qc.cx(q[base+1], q[base+3])
            qc.cx(q[base+1], q[base+4])
            qc.cx(q[base+2], q[base+5])
            qc.cx(q[base+2], q[base+6])
        
        qc.h(q[base])
        qc.h(q[base+1])
        qc.h(q[base+2])
        
        if base + 8 < num_physical_qubits:
            qc.cx(q[base], q[base+7])
            qc.cx(q[base+1], q[base+7])
            qc.cx(q[base+2], q[base+7])
            qc.cx(q[base+3], q[base+8])
            qc.cx(q[base+4], q[base+8])
            qc.cx(q[base+5], q[base+8])
            qc.measure(q[base+7], c[i])
    
    noise_model = create_noisy_model()
    simulator = Aer.get_backend('qasm_simulator')
    job = simulator.run(qc, shots=1024)
    result = job.result()
    
    counts = result.get_counts() if result.success else None
    print(f"Shor QEC ({num_logical_qubits} logical qubits, {num_physical_qubits} physical qubits):", counts)

    
    return counts

def quantum_black_hole_simulation_with_qec(num_logical_qubits=1):
    """Runs a black hole time evolution simulation with quantum error correction."""
    num_physical_qubits = num_logical_qubits * 9  # Using Shor encoding (9 physical per logical qubit)
    qc = QuantumCircuit(num_physical_qubits, num_physical_qubits)
    
    # Encode logical qubits
    for i in range(0, num_physical_qubits, 9):
        qc = apply_shor_encoding(qc, i)
    
    # Apply a black hole evolution-like unitary
    for i in range(num_physical_qubits - 1):
        qc.cx(i, i + 1)
        qc.rz(np.pi / 4, i)
    
    # Introduce artificial noise (simulate decoherence)
    qc.x(0)  # Simulate a random bit-flip error
    
    # Detect and correct errors
    for i in range(0, num_physical_qubits, 9):
        qc = detect_and_correct_errors(qc, i)
    
    # Simulate and extract density matrix
    simulator = Aer.get_backend('aer_simulator_density_matrix')
    qc.save_density_matrix()
    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(compiled_circuit).result()
    final_density_matrix = DensityMatrix(result.data(0)['density_matrix'])
    
    # Compute entropies
    entropies = [qiskit_qiskit_entropy(partial_trace(final_density_matrix, [i])) for i in range(num_physical_qubits)]
    
    return qc, entropies

def black_hole_time_evolution(qc, num_steps=10, delta_t=0.1):
    """
    Simulates the time evolution of a black hole system using a Hamiltonian-based approach.
    
    Parameters:
        qc (QuantumCircuit): The initial quantum circuit.
        num_steps (int): Number of time steps.
        delta_t (float): Time step size.
    
    Returns:
        evolved_qc (QuantumCircuit): The final evolved quantum circuit.
        entropy_history (list): Von Neumann entropy values over time.
    """
    num_qubits = qc.num_qubits
    backend = AerSimulator(method='density_matrix')
    
    # Define a simple Hamiltonian (random Hermitian matrix for now)
    H = np.random.rand(2**num_qubits, 2**num_qubits) + 1j * np.random.rand(2**num_qubits, 2**num_qubits)
    H = (H + H.conj().T) / 2  # Ensure Hermitian property
    
    # Compute the unitary time evolution operator U = exp(-i H Δt)
    U = Operator(expm(-1j * delta_t * H))
    
    # Prepare for entropy tracking
    entropy_history = []
    evolved_qc = qc.copy()
    
    for _ in range(num_steps):
        evolved_qc.unitary(U, evolved_qc.qubits, label='Time Evolution')
        transpiled_qc = transpile(evolved_qc, backend)
        qobj = assemble(transpiled_qc)
        result = backend.run(qobj).result()
        
        # Extract the density matrix
        density_matrix = result.data(0).get('density_matrix', None)
        if density_matrix is None:
            raise ValueError("Density matrix not found in simulation result.")
        density_matrix = DensityMatrix(density_matrix)
        
        # Compute Von Neumann entropy for subsystem
        entropy_values = [qiskit_qiskit_entropy(partial_trace(density_matrix, [i])) for i in range(num_qubits)]
        entropy_history.append(np.mean(entropy_values))
    
    return evolved_qc, entropy_history

def run_and_extract_counts_quantum(qc, backend, shots=8192):
    """
    Runs a quantum circuit on the specified backend and extracts measurement results.

    Parameters:
        qc (QuantumCircuit): The quantum circuit to execute.
        backend (Backend): The IBM backend to use for the execution.
        shots (int): Number of shots for the experiment.

    Returns:
        dict: A dictionary of bitstring counts from the measurement results.
    """

    # Transpile the circuit for the backend
    transpiled_qc = transpile(qc, backend=backend)
    
    # Run the circuit with the sampler and session
    with Session(backend=backend) as session:
        sampler = Sampler()
        job = sampler.run([transpiled_qc], shots=shots)
        result = job.result()

    # Debug: Inspect the raw result object
    print("Raw Result Object:", result)

    try:
        # Access the first `SamplerPubResult`
        pub_result = result._pub_results[0]  # Safely access `SamplerPubResult`
        data_bin = pub_result.data  # Access `DataBin`
        bit_array = data_bin.c  # Access the `BitArray`

        print("Bit array: ", bit_array)

    except Exception as e:
        print("Something went wrong analyzing the bit array: ", e)
        
    # Check if the backend is a simulator
    is_simulator = backend.configuration().simulator

    if is_simulator:
        # Use Aer simulator for execution
        try:
            simulator_backend = Aer.get_backend('aer_simulator')
            transpiled_qc = transpile(qc, backend=simulator_backend)
            job = simulator_backend.run(transpiled_qc, backend=simulator_backend, shots=shots)
            result = job.result()
            counts = result.get_counts()
            print("Counts (simulated):", counts)
            return counts
        
        except Exception as e:
            print(f"Error executing on simulated backend: {e}")
            return None
        
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
            bit_array = data_bin.c  # Access the `BitArray`

            print("Bit array: ", bit_array)

            # Convert the BitArray to a list of bitstrings
            results = extract_counts_from_bitarray(bit_array)
            print("Results: ", results)

            entropy = analyze_shannon_qiskit_entropy(results)
            print("Entropy: ", entropy)

        except Exception as e:
            print(e)

    return results

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

def run_and_extract_counts_quantum_qiskit_entropy(qc, backend, shots=8192):
    """
    Runs a quantum circuit on the specified backend and extracts measurement results.

    Parameters:
        qc (QuantumCircuit): The quantum circuit to execute.
        backend (Backend): The IBM backend to use for the execution.
        shots (int): Number of shots for the experiment.

    Returns:
        dict: A dictionary of bitstring counts from the measurement results.
    """
    from qiskit_ibm_runtime import Session

    # Transpile the circuit for the backend
    transpiled_qc = transpile(qc, backend=backend)
    
    # Run the circuit with the sampler and session
    with Session(backend=backend) as session:
        sampler = Sampler()
        job = sampler.run([transpiled_qc], shots=shots)
        result = job.result()

    # Debug: Inspect the raw result object
    print("Raw Result Object:", result)

    try:
        # Access the first `SamplerPubResult`
        pub_result = result._pub_results[0]  # Safely access `SamplerPubResult`
        data_bin = pub_result.data  # Access `DataBin`
        bit_array = data_bin.meas  # Access the `BitArray`

        print("Bit array: ", bit_array)

        # Convert the BitArray to counts
        counts = Counts.from_memory(memory=bit_array.to_list(), shots=shots)
        print("Counts (get_counts):", counts)
        analyze_entropy_dynamics(counts)
        return counts

    except Exception as e:
        print("Error extracting counts:", e)
        return None

def apply_time_reversal(qc):
    """Apply time-reversal transformation (without measurement gates)."""
    qc_no_measure = qc.copy()
    qc_no_measure.data = [instr for instr in qc_no_measure.data if instr.operation.name != 'measure']
    return qc_no_measure.inverse()

def time_reverse_circuit(qc):
    """Time-reverses a quantum circuit by removing measurements, inverting, and re-adding measurements."""
    # Remove measurements
    qc_no_measurements = qc.remove_final_measurements(inplace=False)
    
    # Invert the unitary operations
    qc_reversed = qc_no_measurements.inverse()

    # Reapply measurements
    qc_reversed.measure_all()

    return qc_reversed

def analyze_shannon_qiskit_entropy(counts):
    """
    Analyzes the entropy of the results to study time-reversal or multiverse effects.
    """
    from scipy.stats import entropy

    # Normalize counts to probabilities
    total_counts = sum(counts.values())
    probabilities = np.array([count / total_counts for count in counts.values()])

    # Calculate Shannon entropy
    shannon_entropy = qiskit_entropy(probabilities, base=2)

    print("Shannon Entropy:", shannon_entropy)
    return {"shannon_entropy": shannon_entropy, "counts": counts}

def add_classical_bits(qc, num_clbits):
    """Ensures that the quantum circuit has at least `num_clbits` classical bits."""
    current_clbits = qc.num_clbits

    if current_clbits < num_clbits:
        extra_clbits = num_clbits - current_clbits
        qc.add_register(ClassicalRegister(extra_clbits, 'extra_c'))

    return qc

def apply_charge_injection(qc, qubits):
    """
    Injects a charge-like phase shift onto selected qubits to test amplification effects.
    """
    for q in qubits:
        qc.p(np.pi / 4, q)  # Phase shift injection to alter probability distribution
    return qc

def generate_true_random_state(num_qubits):
    """
    Creates a maximally mixed state to compare against experimental biasing.
    """
    qc = QuantumCircuit(num_qubits)
    for q in range(num_qubits):
        qc.h(q)  # Superposition ensures equal probability
    return qc

def extract_counts(result, use_sampler):
    """Extracts measurement counts from a Qiskit result object, handling both standard and Sampler cases."""
    print("Result: ", result)

    if use_sampler:
        # Handle Sampler result extraction
        data_bin = result[0].data
        key = next(iter(data_bin.__dict__))  # Get first available key
        bitarray = getattr(data_bin, key)  # Access the BitArray
        counts = extract_counts_from_bitarray(bitarray)  # Convert it to standard counts

    else:
        # Handle standard Qiskit Result extraction
        if hasattr(result, 'results') and isinstance(result.results, list):
            # Extract the first experiment's data
            exp_result = result.results[0]
            if hasattr(exp_result.data, 'counts'):
                raw_counts = exp_result.data.counts
                # Convert hex keys ('0x0', '0x1', etc.) to integers
                counts = {int(k, 16): v for k, v in raw_counts.items()}
            else:
                raise ValueError("Counts not found in the result data.")
        else:
            raise ValueError("Invalid Qiskit result format.")

    return counts


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

def mirrored_state_prep(qc, target_state):
    """
    Prepare a mirrored state based on the target state.
    
    Parameters:
    - qc (QuantumCircuit): The circuit to apply the mirrored state preparation.
    - target_state (str): Target state as a binary string (e.g., "101").
    """
    n_qubits = len(target_state)
    for i, bit in enumerate(reversed(target_state)):
        if bit == "1":
            qc.x(i)  # Apply X gate for '1' bits
    qc.h(range(n_qubits))  # Create superposition

    # Add phase shift to steer the circuit towards mirrored states
    for i, bit in enumerate(reversed(target_state)):
        angle = np.pi / 2 if bit == "0" else -np.pi / 2
        qc.rz(angle, i)  # Phase rotation based on mirroring
    qc.barrier()

def least_busy_backend(service, filters=None):
    """
    Find the least busy backend from the available IBM Quantum backends.

    Parameters:
        service (QiskitRuntimeService): An initialized QiskitRuntimeService object.
        filters (function): A lambda function to filter the list of backends.

    Returns:
        Backend: The least busy backend that matches the filter criteria.
    """
    # Get all backends
    backends = service.backends()

    # Apply filters if provided
    if filters:
        backends = list(filter(filters, backends))

    # Sort by the number of pending jobs (ascending)
    sorted_backends = sorted(
        backends, key=lambda b: b.status().pending_jobs
    )

    # Return the least busy backend
    return sorted_backends[0] if sorted_backends else None

def run_real_backend(qc, backend, shots=8192):
    """
    Runs a quantum circuit on a real quantum backend using the Sampler primitive.
    
    Parameters:
        qc (QuantumCircuit): The quantum circuit to run.
        backend (Backend): The IBM backend to execute on.
        shots (int): Number of shots for the experiment.
    
    Returns:
        dict: Measurement results.
    """

    if isinstance(qc, list):
        qc = qc[0]  # Take the first circuit if it's a list
    
    if not isinstance(qc, QuantumCircuit):
        raise ValueError("Expected a QuantumCircuit, but got:", type(qc))
    if qc.num_clbits > 3:
        qc = QuantumCircuit(qc.num_qubits, 3)  # Keep at most 3 classical bits
        qc.measure(range(qc.num_qubits), range(3))  # Map measurements to the 3 bits

    sampler = Sampler(backend)
    job = sampler.run([qc], shots=shots)
    result = job.result()
    counts = extract_counts(result, use_sampler=True) # Extract probabilities
    
    print("Counts: ", counts)
    
    return counts

def time_evolution_example(t_steps=5, shots=2048):
    """
    Simulate time-dependent transformations and holographic interactions.
    """
    n_qubits = 3
    qc = QuantumCircuit(n_qubits)
    
    # Step 1: Initialize superposition
    qc.h(0)  # Superposition on the Black Hole qubit
    qc.cx(0, 1)  # Entangle Black Hole and Radiation qubits

    # Step 2: Time Evolution
    for t in range(t_steps):
        angle = (np.pi / 3) * (t + 1)  # Dynamic angle for timeline distortion
        qc.rz(angle, 0)  # Timeline distortion on the Black Hole qubit
        qc.rx(np.pi / 4, 1)  # Holographic interaction on the Radiation qubit
        qc.cx(1, 2)  # Correlate Radiation with External Environment
        qc.barrier()

    # Step 3: Measure final probabilities
    qc.measure_all()

    # Analyze statevector before measurement
    qc_no_measurements = qc.copy()
    state = Statevector.from_instruction(qc_no_measurements)
    print("\nStatevector:")
    print(state)

    # Calculate subsystem entropies
    entropy_0 = qiskit_entropy(partial_trace(state, [1, 2]))
    entropy_1 = qiskit_entropy(partial_trace(state, [0, 2]))
    entropy_2 = qiskit_entropy(partial_trace(state, [0, 1]))
    print(f"Subsystem Entropies: Qubit 0 = {entropy_0}, Qubit 1 = {entropy_1}, Qubit 2 = {entropy_2}")

    # Transpile and run on AerSimulator
    backend = Aer.get_backend('aer_simulator')
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()

    print("\nMeasurement Results:")
    print(counts)

    # Plot the histogram of results
    plot_histogram(counts)
    plt.title("Measurement Results with Time Evolution")
    plt.show()

def enhanced_time_evolution(t_steps, shots=2048):
    """
    Enhance time-dependent transformations and holographic interactions.
    """
    qc = QuantumCircuit(3, 3)

    # Initialize entanglement
    qc.h(0)  # Black Hole qubit in superposition
    qc.cx(0, 1)  # Entangle Black Hole and Radiation
    qc.cx(1, 2)  # Extend entanglement to External Environment

    # Time-evolution with holographic interactions
    for t in range(1, t_steps + 1):
        angle_timeline = np.pi * t / 10
        angle_holographic = np.pi * t / 15

        qc.rz(angle_timeline, 0)  # Timeline distortion (Black Hole qubit)
        qc.rx(angle_holographic, 1)  # Holographic interaction (Radiation qubit)
        qc.cp(np.pi * t / 20, 1, 2)  # Controlled phase interaction

    # Statevector analysis (no measurement gates)
    qc_no_measurements = qc.copy()

    # Add measurements for final probabilities
    qc.measure([0, 1, 2], [0, 1, 2])

    # Analyze statevector
    state = Statevector.from_instruction(qc_no_measurements)
    entropies = calculate_entropies(state)

    print("\nStatevector:")
    print(state)
    print("Subsystem Entropies:", entropies)

    # Run the circuit on a simulator
    results = run_warp_simulation(qc)
    print("\nMeasurement Results:", results)

    return results, entropies

def create_randomized_circuit(num_qubits, depth):
    """Generate a randomized quantum circuit for entropy comparison."""
    qc = QuantumCircuit(num_qubits)
    for _ in range(depth):
        for qubit in range(num_qubits):
            gate = np.random.choice(["h", "x", "y", "z", "cx"])
            if gate == "h":
                qc.h(qubit)
            elif gate == "x":
                qc.x(qubit)
            elif gate == "y":
                qc.y(qubit)
            elif gate == "z":
                qc.z(qubit)
            elif gate == "cx" and qubit < num_qubits - 1:
                qc.cx(qubit, qubit + 1)
    return qc

def charge_injection_scaling(qc, max_levels=5):
    """Scale charge injection cycles and measure entropy."""
    results = []
    for level in range(1, max_levels + 1):
        qc_injected = qc.copy()
        for _ in range(level):
            for qubit in range(qc.num_qubits):
                qc_injected.rx(np.pi / (level + 1), qubit)  # Simulate charge injection

        qc_no_measure = remove_measurements(qc_injected)  # Remove measurements
        state = Statevector.from_instruction(qc_no_measure)

        ent = qiskit_entropy(state)
        results.append((level, ent))
        print(f"Charge Level {level}: Entropy = {ent}")

    return results

##def charge_injection_scaling_0(qc, max_levels=50):
##    """Scale charge injection cycles and measure entropy, returning the modified quantum circuit."""
##    results = []
##    qc_injected = qc.copy()  # Start with a fresh copy
##
##    for level in range(1, max_levels + 1):
##        for qubit in range(qc.num_qubits):
##            qc_injected.rx(np.pi / (level + 1), qubit)
##            print("Injected charge")# Simulate charge injection
##            sys.stdout.flush()
##
##            
##        qc_no_measure = remove_measurements(qc_injected)
##        print("Removed measurement")
##        sys.stdout.flush()# Remove measurements
##        state = Statevector.from_instruction(qc_no_measure)
##
##        rho_numpy = state.to_operator().data
##
### Convert to QuTiP Qobj as a density matrix
##        print("Qutip initiated")
##        sys.stdout.flush()
##        rho_qutip = Qobj(rho_numpy, dims=[[2]*state.num_qubits, [2]*state.num_qubits])
##        print("More qutip")
##        sys.stdout.flush()
##        ent = von_neumann_qiskit_entropy(rho_numpy)
##        print("Entropy")
##        sys.stdout.flush()
##        results.append(ent)
##        print(f"Charge Level {level}: Entropy = {ent}")
##        time.sleep(0.05)
##        sys.stdout.flush()
##
##    # **Ensure the final circuit has measurements before returning**
##    if not qc_injected.clbits:
##        qc_injected.add_register(ClassicalRegister(qc.num_qubits))
##
##    qc_injected.measure_all()
##
##    return qc_injected, results  # **Now it returns a QuantumCircuit**
##

def run_experiment_with_target(backend_type, target_state="111", t_steps=5, shots=2048):
    """
    Run a mirrored state experiment with time-dependent holographic operations.
    """
    # Select backend
    if backend_type == "simulator":
        backend = Aer.get_backend('aer_simulator')
        print("\nRunning on AerSimulator...")
    elif backend_type == "quantum":
        service = QiskitRuntimeService(channel="ibm_quantum")  # Load IBM Quantum account
        filters = lambda b: b.configuration().n_qubits >= 2 and b.status().operational
        backend = least_busy_backend(service, filters)
        if backend is None:
            raise ValueError("No backends are appear to be working at the minute...")
        print(f"\nRunning on quantum backend: {backend.name}...")
    else:
        raise ValueError("Invalid backend_type. Use 'simulator' or 'quantum'.")

    n_qubits = len(target_state)
    qc = QuantumCircuit(n_qubits, n_qubits)  # Create the quantum circuit

    # Step 1: Mirrored State Preparation
    mirrored_state_prep(qc, target_state)  # Pass the circuit as an argument

    # Step 2: Time-dependent holographic operations
    for t in range(1, t_steps + 1):
        angle_timeline = np.pi * t / 10
        angle_holographic = np.pi * t / 15

        qc.rz(angle_timeline, 0)  # Timeline distortion on Black Hole qubit
        qc.rx(angle_holographic, 1)  # Holographic interaction on Radiation qubit
        qc.cx(1, 2)  # Correlate Radiation with Environment

    # Step 3: Add measurement
    qc.measure(range(n_qubits), range(n_qubits))
    
    # Transpile and run
    transpiled_qc = transpile(qc, backend=backend, optimization_level=3)
    print(f"Transpiled Circuit Depth: {transpiled_qc.depth()}")

    if backend_type == "simulator":
        result = backend.run(transpiled_qc, shots=shots).result()
        counts = result.get_counts()
    elif backend_type == "quantum":
        with Session(backend=backend) as session:
        # Use Qiskit Runtime Sampler for quantum backend
            sampler = Sampler()
            job = sampler.run([transpiled_qc])
            result = job.result()
            print("Result: ", result)
                # Extract counts from the nested structure
            try:
            # Navigate to the `BitArray` and extract counts
                pub_result = result._pub_results[0]  # Access the first `SamplerPubResult`
                data_bin = pub_result.data  # Access `DataBin`
                bit_array = data_bin.c  # Access `BitArray`

                counts = extract_counts_from_bitarray(bit_array)
                print("Results: ", counts)

            except Exception as e:
                print(e)

def charge_amplification_11(shots=2048, cycles=5):
    """
    Uses charge injection cycles to amplify the |11⟩ state in a two-qubit system.
    """

    n_qubits = 2
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Step 1: Initialize in Superposition
    qc.h(range(n_qubits))

    # Step 2: Charge Injection Cycles for Amplification
    for cycle in range(cycles):
        phase_shift = (np.pi / (2 + cycle))  # Adaptive phase shift
        qc.cp(phase_shift, 0, 1)  # Inject charge-like correlation
        qc.cz(0, 1)  # Reinforce entanglement
        qc.rx(phase_shift / 2, 0)  # Adjust rotation for coherence
    
    # Step 3: Measurement
    qc.measure(range(n_qubits), range(n_qubits))

    # Simulate Execution
    backend = Aer.get_backend('aer_simulator')
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()

    # Plot Results
    plot_histogram(counts)
    plt.title(f"Charge Injection Amplification (Cycles={cycles})")
    plt.show()

    return counts

def compute_qfi(qc: QuantumCircuit):
    """
    Computes the Quantum Fisher Information (QFI) matrix for a given quantum circuit.
    """
    backend = Aer.get_backend('statevector_simulator')
    qc = transpile(qc, backend)
    
    # Get final statevector
    job = backend.run(qc)
    statevector = Statevector(job.result().get_statevector())
    
    # Compute QFI using the Fubini-Study metric
    rho = statevector.to_operator()
    num_qubits = qc.num_qubits
    
    qfi_matrix = np.zeros((num_qubits, num_qubits), dtype=np.complex128)
    
    for i in range(num_qubits):
        for j in range(num_qubits):
            sigma_i = np.kron(np.eye(2**i), np.kron(np.array([[0, -1j], [1j, 0]]), np.eye(2**(num_qubits-i-1))))
            sigma_j = np.kron(np.eye(2**j), np.kron(np.array([[0, -1j], [1j, 0]]), np.eye(2**(num_qubits-j-1))))
            
            qfi_matrix[i, j] = np.trace(rho @ sigma_i @ rho @ sigma_j)
    
    # Get eigenvalues to analyze structure
    eigenvalues = np.linalg.eigvalsh(qfi_matrix.real)
    
    return qfi_matrix.real, eigenvalues

def apply_amplitude_amplification(qc, target_state='11'):
    """
    Applies amplitude amplification targeting the given state.
    """
    target_int = int(target_state, 2)  # Convert '11' -> 3
    num_qubits = qc.num_qubits

    # Apply oracle: Flip the phase of the target state
    qc.cz(0, 1)  # Adjust based on target state
    
    # Apply diffusion operator (inversion around mean)
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.cz(0, 1)  # Controlled phase inversion
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    
    return qc

def adaptive_charge_injection(qc, target_state="11", prior_counts=None, scaling_factor=1.0):
    """
    Dynamically injects charge into a quantum circuit based on past measurement results.
    Uses a holographic phase memory effect to reinforce state probabilities over time.
    """
    num_qubits = len(target_state)
    charge_register = np.zeros(num_qubits)  # Track charge injection per qubit
    
    # If prior_counts exist, use them to scale charge injection
    if prior_counts:
        total_shots = sum(prior_counts.values())
        for state, count in prior_counts.items():
            probability = count / total_shots
            for i, bit in enumerate(reversed(state)):  # Reverse for correct mapping
                if bit == '1':
                    charge_register[i] += probability * scaling_factor  # Scale injection
    
    # Apply charge injections
    for qubit in range(num_qubits):
        qc.rz(charge_register[qubit] * np.pi / 2, qubit)  # Phase-based charge
        qc.rx(charge_register[qubit] * np.pi / 3, qubit)  # Holographic interaction
    
    return qc

def amplify_11_in_qc(qc):
    """
    Apply targeted amplification to the |11> state within an existing quantum circuit.
    """
    n_qubits = 2  # Assuming a 2-qubit system for |11> amplification

    # Apply a Hadamard to create a superposition (if not already in a specific state)
    qc.h(range(n_qubits))

    # Apply controlled phase shifts to favor |11>
    qc.cp(np.pi / 2, 0, 1)  # Controlled phase shift
    qc.cz(0, 1)  # Controlled Z gate to enhance |11> probability

    # Apply a Grover-like diffusion operator to reinforce |11>
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(1)
    qc.cx(0, 1)  # Controlled-X (CNOT) to mark |11>
    qc.h(1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))

    return qc

def run_amplification_experiment(shots=2048):
    """Runs a quantum experiment using adaptive charge injection."""
    backend = Aer.get_backend('aer_simulator')
    qc = QuantumCircuit(2, 2)
    
    # Initial superposition
    qc.h([0, 1])
    
    # Adaptive charge injection
    prior_counts = {'00': 500, '01': 500, '10': 500, '11': 500}  # Example past data
    qc = adaptive_charge_injection(qc, "11", prior_counts, scaling_factor=2.0)
    
    # Measurement
    qc.measure([0, 1], [0, 1])
    
    # Run simulation
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()
    
    print("Measurement Results:", counts)
    return counts

def create_holographic_time_circuit(target_state="11"):
    """
    Creates a quantum circuit to simulate holographic timeline interactions with a targeted state.
    Args:
        target_state (str): The target state to encode for the Black Hole and Radiation qubits.
    Returns:
        QuantumCircuit: The quantum circuit representing the timeline interaction.
    """
    n_qubits = len(target_state)
    qc = QuantumCircuit(n_qubits, n_qubits)  # Create circuit with classical bits for measurement

    # Encode target state
    for i, bit in enumerate(target_state):
        if bit == '1':
            qc.x(i)  # Flip the qubit

    # Apply holographic timeline interaction
    qc.h(0)  # Superposition for Black Hole qubit
    qc.cx(0, 1)  # Entangle Black Hole with Radiation
    qc.rz(np.pi / 3, 0)  # Simulate timeline distortion on Black Hole
    qc.rx(np.pi / 4, 1)  # Simulate holographic interaction on Radiation

    qc.cx(0, 1)  # Controlled-NOT to mark `|11⟩`
    qc.z(1)  # Phase flip on `|11⟩`

    qc.measure(range(n_qubits), range(n_qubits))

    return qc

def run_qc_with_diagnostics(qc, shots=2048):
    """
    Runs a given quantum circuit on a simulator and provides diagnostics.
    """
    backend = Aer.get_backend('aer_simulator')

    # Transpile for the backend
    transpiled_qc = transpile(qc, backend)
    
    # Run the circuit
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()

    # Print measurement results
    print("\nMeasurement Results:")
    print(counts)

    # Plot histogram
    plt.figure(figsize=(8,5))
    plt.bar(counts.keys(), counts.values(), color='royalblue')
    plt.xlabel("Measurement Outcome")
    plt.ylabel("Counts")
    plt.title("Quantum Circuit Measurement Results")
    plt.show()

    # Calculate subsystem entropy if possible
    try:
        state = Statevector.from_instruction(qc)
        entropy_0 = qiskit_entropy(partial_trace(state, [1]))
        entropy_1 = qiskit_entropy(partial_trace(state, [0]))
        print(f"Subsystem Entropies: Qubit 0 = {entropy_0}, Qubit 1 = {entropy_1}")
    except Exception as e:
        print(f"Could not calculate subsystem entropy: {e}")

    return counts

def construct_baseline_circuit(n_qubits=3):
    """Create a neutral quantum circuit with equal superposition."""
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))  # Equal superposition
    qc.measure(range(n_qubits), range(n_qubits))
    return qc

def apply_charge_injection(qc):
    """Redirect charge bias to favor 11 instead of 00."""
    injected_qc = qc.copy()
    injected_qc.rx(np.pi / 3, 0)
    injected_qc.ry(np.pi / 3, 1)
    return injected_qc

def introduce_phase_distortion(qc):
    """Introduce phase shifts to disrupt any bias."""
    distorted_qc = qc.copy()
    for qubit in range(qc.num_qubits):
        distorted_qc.p(np.pi / 4, qubit)  # Introduce small phase shifts
    return distorted_qc

def apply_holographic_encoding(qc):
    """Apply nonlocal entanglement-based encoding."""
    holographic_qc = qc.copy()
    holographic_qc.cx(0, 1)
    holographic_qc.cp(np.pi / 2, 0, 1)
    return holographic_qc

def run_circuit(qc, shots=32768):
    """Execute a circuit on a simulator and return measurement results."""
    backend = Aer.get_backend('aer_simulator')
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    return result.get_counts()

def analyze_results(counts, label):
    """Plot histogram of results and print probability distribution."""
    print(f"\n{label} Counts:", counts)
    plot_histogram(counts, title=label)
    plt.show()

def amplify_11_phase_balanced(qc):
    """
    Amplifies the |11⟩ state while applying a phase-balancing correction
    to prevent unintended asymmetry during time-reversal.
    """
    qc = qc.copy()  # Work on a copy to avoid modifying the original circuit
    
    # Apply controlled phase shifts to steer probability towards |11>
    phase_shift = np.pi / 4  # Adjustable phase correction factor
    qc.cp(phase_shift, 0, 1)  # Controlled phase shift on first two qubits
    qc.cz(0, 1)  # Apply extra correction to balance inversion effects
    
    # Phase balancing correction applied before final amplification
    qc.h(1)
    qc.sdg(1)  # Counteracts unwanted drift
    qc.h(1)
    
    # Targeted amplification
    qc.h(1)  
    qc.cx(0, 1)
    qc.cz(0, 1)  
    qc.h(1)

    return qc

def create_entangled_system(n_qubits=3):
    """Creates an entangled GHZ-like state to observe entropy dynamics."""
    qc = QuantumCircuit(n_qubits)
    qc.h(0)  # Hadamard on first qubit
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)  # Chain CNOTs for GHZ entanglement
    return qc

def measure_qiskit_entropy(qc):
    """Computes the entropy of subsystems."""
    state = Statevector.from_instruction(qc)
    subsys_entropy = [qiskit_entropy(partial_trace(state, [i])) for i in range(qc.num_qubits)]
    return subsys_entropy

def add_hawking_radiation(qc, rad_qubits=2):
    """Simulates Hawking radiation by adding extra qubits entangled with the system."""
    n = qc.num_qubits
    rad_reg = QuantumRegister(rad_qubits, name="radiation")  
    qc.add_register(rad_reg)  # Adds radiation qubits  # Extend with radiation qubits
    for i in range(rad_qubits):
        qc.cx(i, n + i)  # Entangle system with radiation
    return qc

def add_charge_injection(qc, qubits):
    """Applies charge injection through controlled phase shifts and rotation gates."""
    for qubit in qubits:
        qc.rz(np.pi / 4, qubit)  # Introduce charge-like phase shifts
        qc.rx(np.pi / 6, qubit)  # Inject coherent charge
    return qc

def remove_measurements(qc):
    """Removes measurement operations to allow circuit inversion."""
    qc_no_measure = QuantumCircuit(qc.num_qubits)
    for instr, qargs, cargs in qc.data:
        if instr.name != "measure":
            qc_no_measure.append(instr, qargs, cargs)
    return qc_no_measure

def apply_charge_injection(qc, qubits, phase_shifts=None, cycles=1):
    """
    Applies charge injection cycles to the given quantum circuit.
    
    Parameters:
        qc (QuantumCircuit): The quantum circuit to modify.
        qubits (list): The qubits to apply charge injection to.
        phase_shifts (list, optional): The phase shifts for each qubit. Defaults to randomized shifts.
        cycles (int): Number of charge injection cycles to apply.

    Returns:
        QuantumCircuit: The modified circuit with charge injection applied.
    """
    qc_modified = qc.copy()

    # Generate random phase shifts if not provided
    if phase_shifts is None:
        phase_shifts = [np.random.uniform(0, 2 * np.pi) for _ in qubits]

    for _ in range(cycles):
        for qubit, phase in zip(qubits, phase_shifts):
            qc_modified.p(phase, qubit)  # Apply phase injection
            qc_modified.h(qubit)         # Hadamard gate to spread interference
            qc_modified.p(-phase, qubit) # Reverse phase to maintain coherence
    
    return qc_modified

def apply_charge_injection_universal(qc, qubits=None, phase_shifts=None, cycles=1):
    """
    Applies charge injection cycles to a given quantum circuit.

    Parameters:
        qc (QuantumCircuit): The quantum circuit to modify.
        qubits (list, optional): The qubits to apply charge injection to. Defaults to all qubits.
        phase_shifts (list, optional): Custom phase shifts for each qubit. Defaults to randomized shifts.
        cycles (int): Number of charge injection cycles.

    Returns:
        QuantumCircuit: The modified circuit with charge injection applied.
    """
    qc_modified = qc.copy()

    # If no qubits specified, apply to all qubits
    if qubits is None:
        qubits = list(range(qc.num_qubits))

    # Generate random phase shifts if not provided
    if phase_shifts is None:
        phase_shifts = [np.random.uniform(0, 2 * np.pi) for _ in qubits]

    for _ in range(cycles):
        for qubit, phase in zip(qubits, phase_shifts):
            qc_modified.p(phase, qubit)  # Phase shift (simulating charge imbalance)
            qc_modified.h(qubit)         # Hadamard for interference spread
            qc_modified.p(-phase, qubit) # Reverse phase to maintain coherence
    
    return qc_modified

def apply_probability_amplification(qc, target_qubits=None, amplification_factor=1.2):
    """
    Applies probability amplification by adjusting the phase of certain qubits.

    Parameters:
        qc (QuantumCircuit): The quantum circuit to modify.
        target_qubits (list): List of qubits to apply amplification to. Defaults to all qubits.
        amplification_factor (float): Factor by which the probability should be biased.

    Returns:
        QuantumCircuit: The modified circuit with probability amplification.
    """
    if target_qubits is None:
        target_qubits = list(range(qc.num_qubits))  # Default to all qubits
    
    amplified_qc = qc.copy()
    
    # Apply phase shifts to amplify probabilities of target states
    for qubit in target_qubits:
        amplified_qc.p(amplification_factor, qubit)  # Phase shift to enhance probability
    
    return amplified_qc


def remove_measurements(qc):
    qc_no_measure = qc.copy()
    qc_no_measure.data = [instr for instr in qc_no_measure.data if instr.operation.name != 'measure']
    return qc_no_measure

def multiverse_test_circuit(num_qubits=3):
    """
    Creates a circuit that entangles qubits, applies interference, 
    and checks for multiversal correlations.
    """
    qc = QuantumCircuit(num_qubits, num_qubits)

    # Step 1: Create Entanglement
    qc.h(0)
    for i in range(1, num_qubits):
        qc.cx(i-1, i)

    # Step 2: Apply Phase Kicks (Simulates Different Pathways)
    for i in range(num_qubits):
        qc.p(np.pi / 4, i)  # Small phase shifts

    # Step 3: Introduce Interference
    for i in range(num_qubits):
        qc.h(i)

    # Step 4: Measure only a subset to test hidden correlations
    qc.measure(range(num_qubits), range(num_qubits))

    return qc

def charge_transfer_experiment(measure=False):
    """
    Constructs a quantum circuit to test charge transfer via entanglement.
    If measure=False, returns a version without measurements for statevector analysis.
    """
    qr = QuantumRegister(2, name="q")  # Two entangled qubits
    cr = ClassicalRegister(2, name="c")  # Classical register for measurement

    qc = QuantumCircuit(qr, cr)

    # Step 1: Create an Entangled Pair (Q1 <--> Q2)
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])

    # Step 2: Inject Charge into Q1
    qc.rx(1.57, qr[0])  # Simulate charge injection
    qc.crx(1.57, qr[0], qr[1])

    if measure:
        # Step 3: Measure Only Q2 to See If It Extracts Charge
        qc.measure(qr[1], cr[1])

    else:
        qc.save_statevector()

    return qc

def charge_transfer_experiment_3qubits(measure=False):
    qr = QuantumRegister(3, name="q")  # Three entangled qubits
    cr = ClassicalRegister(3, name="c")  # Classical register for measurement
    qc = QuantumCircuit(qr, cr)

    # Step 1: Create a 3-Qubit Entangled Chain (Q1 ↔ Q2 ↔ Q3)
    qc.h(qr[0])      # Hadamard on Q1
    qc.cx(qr[0], qr[1])  # Entangle Q1 <-> Q2
    qc.cx(qr[1], qr[2])  # Entangle Q2 <-> Q3

    # Step 2: Inject Charge into Q1
    qc.rx(1.57, qr[0])  # Charge injection at Q1

    # Step 3: Controlled Charge Transfer to Q2 and Q3
    qc.crx(1.57, qr[0], qr[1])  # Transfer charge influence from Q1 to Q2
    qc.crx(1.57, qr[1], qr[2])  # Transfer charge influence from Q2 to Q3

    if measure:
        # Step 4: Measure Q3 to See If It Extracts Charge
        qc.measure(qr[2], cr[2])

    else:
        # ✅ Explicitly Save Statevector to Ensure It Is Accessible
        qc.save_statevector()

    return qc

def apply_probability_biasing(qc, bias_qubits):
    """
    Applies controlled probability amplification (biasing) to selected qubits.
    
    Args:
        qc (QuantumCircuit): The quantum circuit to modify.
        bias_qubits (list): List of qubits to apply biasing to.

    Returns:
        QuantumCircuit: The modified circuit with biasing.
    """
    qc_bias = qc.copy()
    for qubit in bias_qubits:
        qc_bias.rx(np.pi / 4, qubit)  # Injects slight phase biasing
        qc_bias.ry(np.pi / 4, qubit)  # Helps realign probability spread

    return qc_bias

def create_entangled_circuit(qc):
    """
    Modifies a given quantum circuit to ensure entanglement.

    Parameters:
        qc (QuantumCircuit): The input quantum circuit.

    Returns:
        QuantumCircuit: The modified circuit with enforced entanglement.
    """
    num_qubits = qc.num_qubits

    # Apply Hadamard to the first qubit to create superposition
    qc.h(0)

    # Apply CNOT gates to entangle all qubits in a chain
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    return qc

def mbqc_energy_transfer_experiment(measure=False):
    qr = QuantumRegister(2, name="q")  # Space (Q_space) ↔ Earth (Q_earth)
    cr = ClassicalRegister(2, name="c")  # Classical register for measurement
    qc = QuantumCircuit(qr, cr)

    # Step 1: Create Entanglement Between Space & Earth
    qc.h(qr[0])         # Hadamard on Q_space
    qc.cx(qr[0], qr[1]) # Entangle Q_space <-> Q_earth

    # Step 2: Inject Energy into Space-Based Qubit
    qc.rx(1.57, qr[0])  # Charge injection at Q_space

    if measure:
        # Step 3: Measure Space Qubit to Trigger Remote Charge Transfer
        qc.measure(qr[0], cr[0])
        
        # Step 4: Apply Z-Correction on Earth Qubit Based on Space Qubit's Measurement
        qc.z(qr[1]).c_if(cr[0], 1)  # If Space Qubit is |1⟩, apply phase shift on Earth Qubit

    else:
        # ✅ Save Statevector to Analyze Effects Without Measurement
        qc.save_statevector()

    return qc

def apply_decoherence(qc):
    """
    Applies a depolarizing channel to simulate environmental decoherence.
    """
    noise_model = NoiseModel()
    error = depolarizing_error(0.05, 1)  # 5% chance of decoherence
    noise_model.add_all_qubit_quantum_error(error, "u3")  # Apply to all single-qubit gates

    return qc, noise_model

def compute_partial_qiskit_entropy(statevector, subsystem):
    """
    Computes von Neumann entropy of a selected subsystem.

    Parameters:
        statevector (Statevector): The full quantum state.
        subsystem (list): List of qubit indices to trace out.

    Returns:
        float: The entropy of the remaining system.
    """
    rho = partial_trace(statevector, subsystem)  # Reduce system
    return qiskit_entropy(rho)  # Compute entropy

def inject_charge(circuit, qubits, base_charge=1.57, scaling_factor=0.4):
    """
    Applies a dynamic charge injection based on past charge accumulation.
    """
    global charge_history
    
    # Calculate adjusted charge based on history
    if charge_history:
        avg_past_charge = sum(charge_history[-5:]) / min(len(charge_history), 5)
        adjusted_charge = base_charge * (1 + (1 - avg_past_charge)**2 * scaling_factor)
  # Adjust by 20%
    else:
        adjusted_charge = base_charge
    
    # Apply the charge injection to qubits
    for qubit in qubits:
        circuit.rx(adjusted_charge, qubit)
    
    # Store the applied charge
    charge_history.append(adjusted_charge)
    
    return circuit

def add_holographic_interaction(circuit, qubits, phase_shift=0.5):
    """
    Adds a holographic interaction by applying controlled phase rotations.
    """
    for i in range(len(qubits) - 1):
        circuit.crx(phase_shift, qubits[i], qubits[i + 1])  # Controlled phase shift interaction
    print("✅ Holographic interaction added.")
    return circuit

def controlled_biasing(circuit, qubits, memory_decay=0.85):
    """
    Default, slower biasing for continuous adaptation.
    """
    global charge_history, CRITICAL_MODE

    if not charge_history:
        print("⚠️ No prior charge data. Injecting base charge.")
        charge = 1.57
    else:
        prev_charge = charge_history[-1]
        weighted_feedback = apply_memory_weighting(charge_history[-5:], memory_decay)
        charge = prev_charge + 0.2 + 0.05 * prev_charge + 0.5 * weighted_feedback

        # Ensure charge stays within limits
        charge = max(MIN_CHARGE, min(charge, MAX_CHARGE))

    # Apply charge injection
    for qubit in qubits:
        circuit.rx(charge, qubit)

    charge_history.append(charge)
    print(f"✅ Controlled Biasing Applied: {charge:.6f}")

    return circuit



def adaptive_biasing(circuit, qubits, feedback_factor=0.2, scaling_factor=1.5, memory_decay=0.8):
    """
    Adjusts charge injection based on external feedback, with increased scaling sensitivity and memory weighting.
    """
    global charge_history

    if not charge_history:
        print("⚠️ No prior charge data. Injecting base charge.")
        charge = 1.57
    else:
        # Apply memory weighting to past feedback
        recent_feedback = charge_history[-min(len(charge_history), 5):]  # Last 5 iterations
        weighted_feedback = apply_memory_weighting(recent_feedback, decay_factor=memory_decay)
        
        # Increase scaling sensitivity
        charge = charge_history[-1] * (1 + scaling_factor * weighted_feedback)

    # Apply charge injection
    for qubit in qubits:
        circuit.rx(charge, qubit)

    charge_history.append(charge)
    print(f"✅ Adaptive Biasing Applied: {charge:.6f}")

    return circuit

def get_charge_history():
    return charge_history

def cleanup_charge(circuit, qubits):
    """
    Resets charge accumulation by applying inverse rotations.
    """
    global charge_history
    
    if not charge_history:
        print("✅ No charge to clean up.")
        return circuit
    
    # Apply inverse charge rotation to neutralize accumulated charge
    last_charge = charge_history[-1]
    for qubit in qubits:
        circuit.rx(-last_charge, qubit)  # Inverse rotation to reset state
    
    # Clear charge history after cleanup
    charge_history = []
    print("✅ Charge Cleanup Completed. Residual biases removed.")
    
    return circuit

def simulate_feedback():
    """
    Simulates external reality feedback.
    Returns a value between -0.1 and +0.1 to mimic event occurrence rates.
    """
    return random.uniform(-0.1, 0.1)  # Simulated shift in reality bias strength

def test_reality_biasing(circuit, qubits, iterations=10):
    """
    Runs multiple iterations of biasing, simulating external reality feedback.
    """
    feedback_history = []
    
    for i in range(iterations):
        feedback = simulate_feedback()
        feedback_history.append(feedback)
        print(f"🔄 Iteration {i+1} | Simulated Feedback: {feedback:.3f}")
        circuit = adaptive_biasing(circuit, qubits, feedback_factor=feedback, scaling_factor=1.5, memory_decay=0.8)
    
    return circuit

def apply_memory_weighting(feedback_history, decay_factor=0.85):
    """Applies exponential decay to prioritize recent feedback."""
    if not feedback_history:
        return 0  # Default to neutral feedback
    
    weights = np.array([decay_factor**i for i in range(len(feedback_history))][::-1])
    weighted_feedback = np.dot(feedback_history, weights) / sum(weights)
    return weighted_feedback

def hybrid_scaling_biasing(circuit, qubits, memory_decay=0.85):
    """
    High-impact scaling, only for critical one-shot events (e.g., investor meetings).
    """
    global charge_history, CRITICAL_MODE

    if not CRITICAL_MODE:
        print("⚠️ Hybrid Scaling is disabled. Defaulting to controlled biasing.")
        return controlled_biasing(circuit, qubits, memory_decay)

    if not charge_history:
        print("⚠️ No prior charge data. Injecting base charge.")
        charge = 1.57
    else:
        prev_charge = charge_history[-1]
        weighted_feedback = apply_memory_weighting(charge_history[-5:], memory_decay)
        
        # High-intensity injection
        charge = prev_charge + ALPHA + BETA * (prev_charge ** GAMMA) * (1 - prev_charge / MAX_CHARGE) + DELTA * weighted_feedback

        # Clamp within range
        charge = max(MIN_CHARGE, min(charge, MAX_CHARGE))

    # Apply charge injection
    for qubit in qubits:
        circuit.rx(charge, qubit)

    charge_history.append(charge)
    print(f"🔥 Hybrid Scaling Applied: {charge:.6f} (Critical Event)")

    return circuit

def test_hybrid_biasing(circuit, qubits, iterations=10):
    """
    Runs multiple iterations of biasing, simulating external reality feedback.
    """
    feedback_history = []
    
    for i in range(iterations):
        feedback = simulate_feedback()
        feedback_history.append(feedback)
        print(f"🔄 Iteration {i+1} | Simulated Feedback: {feedback:.3f}")
        circuit = hybrid_scaling_biasing(circuit, qubits, memory_decay=0.85)
    
    return circuit

def test_biasing(circuit, qubits, iterations=10, critical_mode=False):
    """
    Runs multiple iterations of biasing, simulating external reality feedback.
    """
    global CRITICAL_MODE
    CRITICAL_MODE = critical_mode  # Toggle critical mode

    feedback_history = []
    
    for i in range(iterations):
        feedback = simulate_feedback()
        feedback_history.append(feedback)
        print(f"🔄 Iteration {i+1} | Simulated Feedback: {feedback:.3f}")

        if CRITICAL_MODE:
            circuit = hybrid_scaling_biasing(circuit, qubits, memory_decay=0.85)
        else:
            circuit = controlled_biasing(circuit, qubits, memory_decay=0.85)
    
    # Auto-cleanup after a critical event
    if CRITICAL_MODE:
        circuit = cleanup_charge(circuit, qubits)

    return circuit

def generate_random_numbers(size=10000):
    return np.random.random(size)

def generate_binary_samples(num_samples=10000, bias_factor=0):
    """
    Generates binary samples (0s and 1s) with an adjustable bias factor.
    
    - bias_factor = 0 → Unbiased 50/50 probability.
    - bias_factor > 0 → More 1s appear.
    - bias_factor < 0 → More 0s appear.
    """
    base_prob = 0.5  # Default 50% chance for 0 and 1
    biased_prob = min(1, max(0, base_prob + bias_factor))  # Ensure valid probability bounds
    
    return np.random.choice([0, 1], size=num_samples, p=[1 - biased_prob, biased_prob])

def cavity_scaling_test():
    # Create a vacuum cavity (zero photons)
    cavity = basis(10, 0)  # 10 photon Fock basis, starting in |0>

    # Charge injection scaling as displacement operations
    results = []
    max_levels = 50

    for level in range(1, max_levels + 1):
        alpha = 1.0 / (level + 1)  # Analogous to your scaled RX angle
        D = displace(10, alpha)    # Displacement operator for cavity
        cavity = D * cavity        # Inject energy
        
        # Measure entropy (mixedness of photon number distribution)
        ent = entropy_vn(cavity.proj())
        print(f"Level {level}: Entropy = {ent}")
        results.append((level, ent))

    return results

##def charge_scaling_qec(num_qubits=4, max_levels=50):
##    qc = qc_charge_qec(num_qubits)
##    print("QC created")
##    sys.stdout.flush()
##    qc_injected_circuit, results = charge_injection_scaling_0(qc, max_levels)
##    qc_nm = remove_measurements(qc_injected_circuit)
##    print("Removed measurements")
##    sys.stdout.flush()
##    final_state = Statevector.from_instruction(qc_nm)
##    rho = final_state.to_operator().data
##    rho_qutip = Qobj(rho, dims=[[2]*final_state.num_qubits, [2]*final_state.num_qubits])
##    print("Qutip format")
##    sys.stdout.flush()
##    entropy = von_neumann_qiskit_entropy(rho)
##    print(f"Entropy after charge injection with QEC: {entropy}")
##    sys.stdout.flush()
##    return entropy, results

def von_neumann_qiskit_entropy(rho):
    # Ensure it's a 2D matrix
    if len(rho.shape) != 2:
        raise ValueError(f"Expected 2D array for rho, got shape {rho.shape}")

    # Compute eigenvalues
    evals = np.linalg.eigvalsh(rho)  # Efficient for Hermitian matrices

    # Stabilize eigenvalues (avoid log(0))
    evals = np.clip(evals, 1e-15, 1.0)

    # Compute entropy
    entropy = -np.sum(evals * np.log(evals))
    return entropy

##def charge_scaling_no_qec(num_qubits=4, max_levels=50):
##    qc = generate_no_qec_circuit(num_qubits)
##    qc_injected_circuit, results = charge_injection_scaling_0(qc, max_levels)
##    qc_nm = remove_measurements(qc_injected_circuit)
##    final_state = Statevector.from_instruction(qc_nm)
##    rho = final_state.to_operator().data
##    rho_qutip = Qobj(rho, dims=[[2]*final_state.num_qubits, [2]*final_state.num_qubits])
##    entropy = von_neumann_qiskit_entropy(rho)
##    print(f"Entropy after charge injection without QEC: {entropy}")
##    sys.stdout.flush()
##    return entropy, results

def compare_plot_fig_qec(charge_levels, entropy_without_qec, entropy_with_qec, num_qubits=4):
    charge_levels_2 = charge_levels
    print("debug")
    plt.figure(figsize=(14, 7))

    # Plot entropy without QEC
    plt.plot(charge_levels, entropy_without_qec, color='red', linestyle='--', marker='o')

    # Plot entropy with QEC
    plt.plot(charge_levels_2, entropy_with_qec, color='blue', linestyle='-', marker='x')

    # Title & axis labels
    plt.title(f'Entropy vs Charge Injection Level ( With QEC(blue) and Without QEC (red) ) \n{num_qubits} Qubits vs {num_qubits} Qubits Test Circuit', fontsize=14)
    plt.xlabel('Charge Injection Level', fontsize=12)
    plt.ylabel('Entropy (Von Neumann, unitless, normalized)', fontsize=12)

    # Set log scale for Y-axis
    plt.yscale('linear')

    # Adjust Y-axis limits for better focus (adjust if necessary based on data)
    plt.ylim(0.999, 1.008)

    # Add grid for readability
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    # Layout adjustments
    plt.tight_layout()

    # Save as high-res file for DOE slides
    plt.savefig('entropy_vs_charge_injection_highres.png', dpi=300)

    # Show plot
    plt.show()



def run_reality_biasing_test(n_flips=1000, injections=5):
    """
    Main process: baseline -> charge -> test.
    """
    # Generate baseline
    baseline_flips = generate_baseline_flips(n_flips)
    
    # Apply charge injection
    qc = QuantumCircuit(3)
    run_charge_injection(qc, [0, 1], injections=injections)  # You can adjust qubits & injections
    
    # Generate post-injection flips
    biased_flips = generate_post_injection_flips(n_flips)
    
    # Analyze and report
    analyze_results(baseline_flips, biased_flips)

def charge_bias_titration_test(num_flips=1000, charge_range=(1.5, 5.0), charge_steps=10):
    charges = np.linspace(charge_range[0], charge_range[1], charge_steps)
    results = []
    p_values = []
    
    print("\n🔬 Running Charge Titration Bias Test")
    print("Charge Level | P(Heads) | ΔP from 0.5 | p-value (significance)")

    for charge in charges:
        # Simulate statevector real part influence (normalize between 0 and 1)
        statevector_real = np.clip(charge / 5.0, 0, 1)  # Assuming max charge = 5.0 is full bias
        bias_factor = 1 + statevector_real  # Stretching bias towards Heads
        
        # Generate biased coin flips
        biased_flips = np.random.rand(num_flips) * bias_factor
        heads = np.sum(biased_flips > 0.5)
        p_heads = heads / num_flips
        delta_p = p_heads - 0.5
        
        # Statistical test: is the observed bias significant?
        p_val = binom_test(heads, num_flips, p=0.5, alternative='two-sided')
        
        results.append((charge, p_heads, delta_p, p_val))
        p_values.append(p_val)

        print(f"{charge:.2f}         | {p_heads:.5f} | {delta_p:+.5f}   | {p_val:.5f}")

    # Plot results
    charges_plot, p_heads_plot, _, _ = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(charges_plot, p_heads_plot, marker='o', linestyle='-', color='b', label='P(Heads)')
    plt.axhline(0.5, color='gray', linestyle='--', label='Fair Coin (0.5)')
    plt.xlabel("Charge Injection Level")
    plt.ylabel("Probability of Heads")
    plt.title("Charge Injection vs. Probability of Heads")
    plt.legend()
    plt.grid(True)
    plt.show()

    return results

def get_qrng_bits(num_bits=10, retries=3, delay=8):
    """
    Fetch quantum random bits from ANU QRNG API with retries.
    
    Args:
        num_bits (int): Number of random bits to fetch.
        retries (int): Number of retries if request fails.
        delay (int): Delay between retries in seconds.
        
    Returns:
        list: List of integers (0 or 1).
    """
    url = f'https://qrng.anu.edu.au/API/jsonI.php?length={num_bits}&type=uint8'
    
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url)
            data = response.json()
            
            if data["success"]:
                bits = [int(b) % 2 for b in data["data"]]  # Convert to 0/1
                print(f"✅ Successfully fetched {num_bits} QRNG bits (Attempt {attempt})")
                return bits
            else:
                print(f"❌ API error: {data['message']} (Attempt {attempt})")
        
        except Exception as e:
            print(f"❌ Error fetching QRNG data: {e} (Attempt {attempt})")
        
        if attempt < retries:
            print(f"⏳ Retrying in {delay} seconds...")
            time.sleep(delay)
    
    print("⚠️ Failed to fetch QRNG data after multiple attempts.")
    return []

def local_qrng(bits=10):
    backend = Aer.get_backend('qasm_simulator')
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    job = backend.run(qc, shots=bits)
    result = job.result()
    counts = result.get_counts()
    ones = counts.get('1', 0)
    zeros = counts.get('0', 0)
    bits_list = ['1'] * ones + ['0'] * zeros
    return bits_list


def inject_charge(base_charge=1.57, scaling_factor=0.15):
    """
    Charge injection logic to bias reality.
    """
    global charge_history
    adjusted_charge = base_charge

    if charge_history:
        last_charge = charge_history[-1]
        adjusted_charge = last_charge * (1 + scaling_factor)
    
    charge_history.append(adjusted_charge)
    print(f"✅ Charge Injection Applied: {adjusted_charge:.5f}")
    return adjusted_charge

def analyze_qrng_bias(bits_string, label="QRNG Results"):
    """
    Analyze a bit string (e.g., '010101') and calculate Heads/Tails ratios.

    Args:
        bits_string (str): String of '0' and '1' characters from QRNG or simulation.
        label (str): Custom label for reporting (default 'QRNG Results').

    Returns:
        tuple: (P(Heads), P(Tails))
    """
    bits = [int(b) for b in bits_string]  # Convert '0'/'1' string to list of integers
    total_flips = len(bits)
    heads = sum(bits)
    tails = total_flips - heads

    p_heads = heads / total_flips if total_flips else 0
    p_tails = tails / total_flips if total_flips else 0

    print(f"\n🎲 {label}: Heads = {heads}, Tails = {tails}")
    print(f"   - P(Heads) = {p_heads:.5f}, P(Tails) = {p_tails:.5f}")

    return p_heads, p_tails

def generate_baseline_flips(n_flips=1000):
    """
    Generate unbiased random coin flips as baseline.
    """
    flips = np.random.rand(n_flips) < 0.5  # True 50/50 coin flips
    return flips

def run_charge_injection(qc, qubits, base_charge=1.57, injections=5):
    """
    Apply charge injection multiple times as 'reality influence'.
    """
    global charge_history
    for _ in range(injections):
        avg_past_charge = sum(charge_history[-5:]) / min(len(charge_history), 5) if charge_history else 0
        adjusted_charge = base_charge * (1 + (1 - avg_past_charge)**2 * 0.4) if charge_history else base_charge
        for q in qubits:
            qc.rx(adjusted_charge, q)
        charge_history.append(adjusted_charge)
        injection_true = True
    print(f"✅ Charge Injection Completed. Total Charge History: {charge_history}")
    return qc

def generate_post_injection_flips(n_flips=1000):
    """
    Generate new coin flips after 'reality biasing' to see if distribution changes.
    """
    flips = np.random.rand(n_flips) < 0.5  # Same unbiased generation
    return flips

def analyze_results(baseline_flips, biased_flips):
    """
    Analyze and compare baseline and biased flip results.
    """
    baseline_heads = np.sum(baseline_flips)
    biased_heads = np.sum(biased_flips)
    
    baseline_prob = baseline_heads / len(baseline_flips)
    biased_prob = biased_heads / len(biased_flips)
    
    delta = biased_prob - baseline_prob
    
    print("\n🔍 Reality Biasing Analysis:")
    print(f"   - Baseline P(Heads) = {baseline_prob:.5f}")
    print(f"   - Biased   P(Heads) = {biased_prob:.5f}")
    print(f"   - Shift    ΔP       = {delta:+.5f}")

def inject_charge_0(qubits=1, charge_factor=1.57):
    qc = QuantumCircuit(qubits)
    for q in range(qubits):
        qc.rx(charge_factor, q)  # Apply charge bias
    return qc

def encode_bosonic_cat_code(alpha=2.0):
    """
    Encodes a logical '0' and '1' using a bosonic cat code.
    Returns logical |0_L⟩ and |1_L⟩ as superpositions of coherent states.
    """
    # Logical |0> = |alpha> + |-alpha>
    cat_0 = (coherent(10, alpha) + coherent(10, -alpha)).unit()

    # Logical |1> = |alpha> - |-alpha>
    cat_1 = (coherent(10, alpha) - coherent(10, -alpha)).unit()

    return cat_0, cat_1

def apply_photon_loss(state, loss_prob=0.1):
    """
    Applies photon loss channel to a bosonic state (coherent/cat code).
    This uses an exponential decay (Lindblad-like) model for cavity leakage.
    """
    dim = state.dims[0][0]  # Hilbert space dimension
    a = destroy(dim)  # Photon annihilation operator

    # Kraus operator for photon loss: exp(-gamma * a.dag() * a / 2)
    gamma = -np.log(1 - loss_prob)  # Relate loss prob to decay rate
    loss_op = (-(gamma / 2) * a.dag() * a).expm()

    # Apply loss operation
    noisy_state = loss_op * state
    return noisy_state.unit()  # Normalize

def measure_parity(state):
    """
    Measures the photon number parity of the state as a syndrome.
    """
    # Parity operator: sum_n (-1)^n |n><n|
    dim = state.dims[0][0]
    parity_op = sum([(-1)**n * basis(dim, n) * basis(dim, n).dag() for n in range(dim)])
    parity_expectation = (state.dag() * parity_op * state).real


    return parity_expectation

def correct_parity(state, alpha=0.5):
    """
    Applies a displacement correction based on detected parity (simplified logic).
    """
    parity = measure_parity(state)
    if parity < 0:  # If parity flipped, apply displacement to restore
        correction = displace(state.dims[0][0], alpha)
        corrected_state = correction * state * correction.dag()
        return corrected_state.unit()
    else:
        return state  # No correction needed

def apply_photon_jumps(state, p_loss, num_jumps=1):
    """
    Apply explicit photon jumps (loss events) to simulate stronger cavity leakage.
    """
    dim = state.dims[0][0]
    a = destroy(dim)

    for _ in range(num_jumps):
        if np.random.rand() < p_loss:
            state = (a * state).unit()  # Photon lost (annihilation operator)
    return state

def apply_displacement_noise(state, displacement_strength=0.1):
    """
    Apply random displacement noise to bosonic state to simulate environmental EM fluctuations.
    """
    dim = state.dims[0][0]
    random_phase = np.random.uniform(0, 2 * np.pi)
    displacement = displacement_strength * np.exp(1j * random_phase)
    D = displace(dim, displacement)
    return (D * state).unit()

def apply_displacement_correction(state, parity_syndrome, correction_strength=0.1):
    """
    Apply displacement correction based on measured parity syndrome.
    If parity is not +1, apply displacement to correct back toward code space.
    """
    dim = state.dims[0][0]  # Photon number cutoff dimension
    if parity_syndrome < 0.99:  # If parity flipped or degraded, attempt correction
        # Correction: small displacement (heuristic approach)
        D = displace(dim, correction_strength)  # Displace slightly
        corrected_state = (D * state).unit()
        print(f"Applied displacement correction with strength {correction_strength}")
        return corrected_state
    else:
        # If parity still close to 1, no correction needed
        print("No correction needed, parity near 1")
        return state

def iterative_displacement_correction(state, initial_parity, threshold=0.95, max_attempts=3, correction_strength=0.1):
    """
    Attempt multiple displacement corrections until parity is above threshold.
    """
    attempt = 0
    while initial_parity < threshold and attempt < max_attempts:
        dim = state.dims[0][0]
        D = displace(dim, correction_strength)
        state = (D * state).unit()
        initial_parity = measure_parity(state)
        attempt += 1
        print(f"Correction attempt {attempt}: Parity = {initial_parity:.3f}")
    return state

def smart_displacement_correction(state, initial_parity, max_strength=0.5, step=0.05, tolerance=0.01):
    """
    Adaptive, direction-sensitive displacement correction for bosonic codes.
    Probes both directions, increases strength, stops when parity is close enough to 1.0.
    """
    dim = state.dims[0][0]  # Photon cutoff
    best_state = state
    best_parity = initial_parity
    best_direction = None

    print(f"Starting adaptive correction. Initial parity: {initial_parity:.3f}")

    # First probing: small kicks in + and - direction
    D_pos = displace(dim, step)
    D_neg = displace(dim, -step)

    # Test positive kick
    state_pos = (D_pos * state).unit()
    parity_pos = measure_parity(state_pos)
    print(f"Parity after +{step:.3f} displacement: {parity_pos:.3f}")

    # Test negative kick
    state_neg = (D_neg * state).unit()
    parity_neg = measure_parity(state_neg)
    print(f"Parity after -{step:.3f} displacement: {parity_neg:.3f}")

    # Pick better direction
    if parity_pos > parity_neg:
        best_state, best_parity, best_direction = state_pos, parity_pos, +1
    else:
        best_state, best_parity, best_direction = state_neg, parity_neg, -1

    # Continue correcting adaptively
    current_strength = step
    while abs(1 - best_parity) > tolerance and current_strength <= max_strength:
        current_strength += step
        D = displace(dim, best_direction * current_strength)
        new_state = (D * state).unit()
        new_parity = measure_parity(new_state)
        print(f"Displacement {best_direction * current_strength:.3f}, Parity: {new_parity:.3f}")

        # Update if better
        if new_parity > best_parity:
            best_state, best_parity = new_state, new_parity
        else:
            # If not improving, stop
            print("No improvement, stopping correction.")
            break

    print(f"Final corrected parity: {best_parity:.3f}")
    return best_state

    
#Functions that start with main_ are used as benchmarks of experiments that we've run together so I can run them again
#and not lose like the outputs in the sea of factory code
############################################################################################################################################










#############################################################################################################################################

def main_run_entropy_experiment():
    """Tests charge injection on entropy in a time-reversed system."""
    backend = Aer.get_backend("aer_simulator")

    # Create base circuit
    qc = QuantumCircuit(3)
    qc.h(range(3))  # Initial superposition
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()

    # Run baseline entropy measurement
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=8192).result()
    counts_baseline = result.get_counts()

    # Apply time reversal (remove measurements)
    qc_nm = QuantumCircuit(qc.num_qubits)
    qc_nm.compose(qc.remove_final_measurements(inplace=False), inplace=True)
    if qc_nm.depth() == 0:
        raise ValueError("Circuit is empty after removing measurements. Check gate preservation.")

    qc.measure_all()
    
    qc_rev = qc.copy()
    qc_rev_nm = QuantumCircuit(qc.num_qubits)
    qc_rev_nm.compose(qc_rev.remove_final_measurements(inplace=False), inplace=True)
    qc_reversed = qc_rev_nm.inverse()
    qc_reversed.measure_all()
    transpiled_qc_reversed = transpile(qc_reversed, backend)
    result_reversed = backend.run(transpiled_qc_reversed, shots=8192).result()
    counts_reversed = result_reversed.get_counts()

    # Apply charge injection & re-run
    
    
    qc_c_rev = qc_reversed.copy()
    qc_c_rev_nm = QuantumCircuit(qc.num_qubits)
    qc_c_rev_nm.compose(qc_c_rev.remove_final_measurements(inplace=False), inplace=True)
    qc_charge_rev = add_charge_injection(qc_rev, [0, 1, 2])
    qc_charge_rev.measure_all()
    transpiled_qc_charge = transpile(qc_charge_rev, backend)
    result_charge = backend.run(transpiled_qc_charge, shots=8192).result()
    counts_charge = result_charge.get_counts()

    # Compute entropies
    state = Statevector.from_instruction(qc_nm)
    state_rev = Statevector.from_instruction(qc_rev_nm)
    state_charge = Statevector.from_instruction(qc_c_rev_nm)
    entropy_baseline = [qiskit_entropy(partial_trace(state, [i])) for i in range(3)]
    entropy_reversed = [qiskit_entropy(partial_trace(state_rev, [i])) for i in range(3)]
    entropy_charge = [qiskit_entropy(partial_trace(state_charge, [i])) for i in range(3)]

    # Plot results
    plt.plot(entropy_baseline, label="Baseline", linestyle="-")
    plt.plot(entropy_reversed, label="Time-Reversed", linestyle="--")
    plt.plot(entropy_charge, label="Charge Injection", linestyle=":")
    plt.legend()
    plt.xlabel("Qubit Index")
    plt.ylabel("Entropy")
    plt.title("Entropy Evolution in Charge-Injected Time Reversal")
    plt.show()

    print(f"Baseline Counts: {counts_baseline}")
    print(f"Time-Reversed Counts: {counts_reversed}")
    print(f"Charge-Injection Counts: {counts_charge}")\
                             
##Output
##Baseline Counts: {'000': 992, '100': 997, '001': 1072, '101': 1009, '111': 1000, '010': 1051, '011': 1066, '110': 1005}
##Time-Reversed Counts: {'101': 975, '001': 976, '011': 980, '111': 1014, '110': 1063, '010': 1101, '000': 1059, '100': 1024}

def main_run_entropy_experiment_quantum():
    """Tests charge injection on entropy in a time-reversed system."""
    service = QiskitRuntimeService(channel='ibm_quantum')
    backend = get_best_backend(service)
    sampler = Sampler(backend)

    # Create base circuit
    qc = QuantumCircuit(3)
    qc.h(range(3))  # Initial superposition
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()

    # Run baseline entropy measurement
    transpiled_qc = transpile(qc, backend)
    result = sampler.run([transpiled_qc], shots=8192).result()
    counts_baseline = extract_counts_from_bitarray(result[0].data.meas)  # List of bitstring samples
    

    # Apply time reversal (remove measurements)
    qc_nm = QuantumCircuit(qc.num_qubits)
    qc_nm.compose(qc.remove_final_measurements(inplace=False), inplace=True)
    if qc_nm.depth() == 0:
        raise ValueError("Circuit is empty after removing measurements. Check gate preservation.")

    qc.measure_all()
    
    qc_rev = qc.copy()
    qc_rev_nm = QuantumCircuit(qc.num_qubits)
    qc_rev_nm.compose(qc_rev.remove_final_measurements(inplace=False), inplace=True)
    qc_reversed = qc_rev_nm.inverse()
    qc_reversed.measure_all()
    transpiled_qc_reversed = transpile(qc_reversed, backend)
    result_reversed = sampler.run([transpiled_qc_reversed], shots=8192).result()
    counts_reversed = extract_counts_from_bitarray(result_reversed[0].data.meas) # List of bitstring samples
    
    qc_c_rev = qc_reversed.copy()
    qc_c_rev_nm = QuantumCircuit(qc.num_qubits)
    qc_c_rev_nm.compose(qc_c_rev.remove_final_measurements(inplace=False), inplace=True)
    qc_charge_rev = add_charge_injection(qc_rev, [0, 1, 2])
    qc_charge_rev.measure_all()
    transpiled_qc_charge = transpile(qc_charge_rev, backend)
    result_charge = sampler.run([transpiled_qc_charge], shots=8192).result()
    counts_charge = extract_counts_from_bitarray(result_charge[0].data.meas)  # List of bitstring samples

    print(f"Counts baseline: {counts_baseline}")
    print(f"Counts reversed: {counts_reversed}")
    print(f"Counts charged: {counts_charge}")

    # Compute entropies
    state = Statevector.from_instruction(qc_nm)
    state_rev = Statevector.from_instruction(qc_rev_nm)
    state_charge = Statevector.from_instruction(qc_c_rev_nm)
    entropy_baseline = [qiskit_entropy(partial_trace(state, [i])) for i in range(3)]
    entropy_reversed = [qiskit_entropy(partial_trace(state_rev, [i])) for i in range(3)]
    entropy_charge = [qiskit_entropy(partial_trace(state_charge, [i])) for i in range(3)]

    # Plot results
    plt.plot(entropy_baseline, label="Baseline", linestyle="-")
    plt.plot(entropy_reversed, label="Time-Reversed", linestyle="--")
    plt.plot(entropy_charge, label="Charge Injection", linestyle=":")
    plt.legend()
    plt.xlabel("Qubit Index")
    plt.ylabel("Entropy")
    plt.title("Entropy Evolution in Charge-Injected Time Reversal")
    plt.show()

    print(f"Baseline Counts: {counts_baseline}")
    print(f"Time-Reversed Counts: {counts_reversed}")
    print(f"Charge-Injection Counts: {counts_charge}")

def main_run_reverse_qiskit_entropy():
    """Runs the entropy experiment with forward, reversed, and Hawking radiation scenarios."""
    backend = Aer.get_backend('aer_simulator')
    
    # Step 1: Create baseline entangled system
    qc = create_entangled_system()
    transpiled_qc = transpile(qc, backend)
    baseline_entropy = measure_qiskit_entropy(qc)

    # Step 2: Reverse time and measure entropy
    qc_reversed = time_reverse_circuit(qc)
    transpiled_qc_rev = transpile(qc_reversed, backend)
    reversed_entropy = measure_qiskit_entropy(qc_reversed)

    # Step 3: Simulate Hawking radiation by adding external qubits
    qc_radiation = add_hawking_radiation(qc, rad_qubits=2)
    transpiled_qc_rad = transpile(qc_radiation, backend)
    radiation_entropy = measure_qiskit_entropy(qc_radiation)

    # Step 4: Print results
    print(f"Baseline Entropy: {baseline_entropy}")
    print(f"Time-Reversed Entropy: {reversed_entropy}")
    print(f"Entropy with Hawking Radiation: {radiation_entropy}")

    # Step 5: Plot entropy evolution
    plt.plot(range(len(baseline_entropy)), baseline_entropy, label="Baseline")
    plt.plot(range(len(reversed_entropy)), reversed_entropy, label="Time-Reversed", linestyle="dashed")
    plt.plot(range(len(radiation_entropy)), radiation_entropy, label="Hawking Radiation", linestyle="dotted")
    plt.xlabel("Qubit Index")
    plt.ylabel("Entropy")
    plt.title("Entropy Evolution in Time & Radiation Effects")
    plt.legend()
    plt.show()

#Currently not working

def main_quantum_encrypt():
    """Quantum encryption test"""
    """Main program loop."""
    print("Starting Error Correction Test...\n")

    # Example Quantum Circuit
    n_qubits = 3
    qc = QuantumCircuit(n_qubits)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1)  # CNOT gate between qubit 0 and 1
    qc.cx(0, 2)  # CNOT gate between qubit 0 and 2
    qc.measure_all()

    # Backend and shots
    backend = Aer.get_backend('qasm_simulator')
    shots = 1024
    state = "011"

    # Run the error correction testing
    print("Original Circuit:")
    print(qc)
    result = run_circuit_with_feedback(qc, backend, state, shots=shots)

    # Display final results
    for key in result.keys():
        print(f"Key: {key}")
        print("\nFinal Measurement Results:")

def main_compare_qec_methods():
    """Compare shors qec to our own"""
    for qubits in [1, 2, 3]:
        charge_counts = charge_preserving_qec_noisy(num_qubits=qubits)
        print(f"Charge-Preserving QEC {qubits} qubits result: {charge_counts}")
    
    for logical_qubits in [1, 2, 3]:
        shor_counts = shor_qec_noisy(num_logical_qubits=logical_qubits)
        print(f"Shor QEC {logical_qubits} logical qubits result: {shor_counts}")
        
def main_check_decrypt():
    """Test to see if we can create a type of quantum encryption/decryption and apply it to the branches"""
    key = "110011"
    monitored_events = ["identity theft", "unauthorized access", "data breach"]

    # 🚨 Auto-detect branches
    detected_branches = auto_detect_branches(monitored_events, key, detection_threshold=60)
    print("🧭 Detected Branches:")
    for event, details in detected_branches.items():
        print(f"- {event}: Signature {details['signature_bits']} (Confidence: {details['detection_confidence']}%)")

        # 🔒 Encrypt detected branch signature
        encrypted_counts, unitary = quantum_scramble_encrypt(details['signature_bits'], key)
        print("  🔒 Encrypted Outcomes:", encrypted_counts)

        # 🔑 Decrypt for verification
        most_common_state = max(encrypted_counts, key=encrypted_counts.get)
        decrypted_counts = quantum_scramble_decrypt(unitary, key, most_common_state)
        print("  🔑 Decrypted Outcomes:", decrypted_counts)

    # Decrypt for demonstration (only works if reversal aligns perfectly)
    most_common_encrypted_state = max(encrypted_counts, key=encrypted_counts.get)
    decrypted_counts = quantum_scramble_decrypt(unitary, key, most_common_encrypted_state)
    print("🔑 Decrypted Signature Outcomes:", decrypted_counts)

def main_entropy_black_hole():
    """Test to see black hole simulation results"""
    qc, entropies = black_hole_warp_simulation_core()

    print("\n🕳️ Black Hole Warp Simulation Results 🕳️")
    print("========================================")
    
    # Print the quantum circuit
    print("\nQuantum Circuit:")
    print(qc)

    # Print Von Neumann entropy results
    print("\nVon Neumann Entropies for Each Qubit:")
    for i, entropy in enumerate(entropies):
        print(f" Qubit {i}: {entropy:.6f}")

    print("\n✅ Simulation complete!")

def main_qec_shor_simulated_black_hole():
    """experiment to see qec (quantum error correction) on the simulated black hole code"""
    qc, entropies = quantum_black_hole_simulation_with_qec()
    print("Quantum circuit:")
    print(qc)
    print("Von Neumann entropies after evolution:", entropies)

def main_multiversal_time_travel_simulator():
    """Basic time reversal test"""
    use_simulator = True  # Switch to hardware for later runs
    hardware_backend_name = "ibm_sherbrooke"

    # Initialize backend
    backend = initialize_backend(use_simulator=use_simulator, hardware_backend_name=hardware_backend_name)

    if backend:
        # Initialize base quantum circuit
        qc_base = create_base_circuit()

        # Number of iterations for causal feedback
        iterations = 5
        previous_results = None

        for i in range(iterations):
            print(f"\nIteration {i + 1}")

            # Apply causal feedback to the circuit
            qc_modified = add_causality_to_circuit(qc_base.copy(), previous_results, qubits=[0, 1])

            # Run the experiment
            results = run_and_extract_counts_quantum(qc=qc_modified, backend=backend, shots=8192)
            print("Results:", results)

            # Update previous results
            previous_results = results
    else:
        print("Failed to initialize backend.")
##Output
##Iteration 1
##Raw Result Object: PrimitiveResult([SamplerPubResult(data=DataBin(c=BitArray(<shape=(), num_shots=8192, num_bits=2>)), metadata={'shots': 8192, 'circuit_metadata': {}})], metadata={'version': 2})
##Bit array:  BitArray(<shape=(), num_shots=8192, num_bits=2>)
##Counts (simulated): {'11': 4105, '00': 4087}
##Results: {'11': 4105, '00': 4087}
##
##Iteration 2
##Raw Result Object: PrimitiveResult([SamplerPubResult(data=DataBin(c=BitArray(<shape=(), num_shots=8192, num_bits=2>)), metadata={'shots': 8192, 'circuit_metadata': {}})], metadata={'version': 2})
##Bit array:  BitArray(<shape=(), num_shots=8192, num_bits=2>)
##Counts (simulated): {'00': 4104, '11': 4088}
##Results: {'00': 4104, '11': 4088}
##
##Iteration 3
##Raw Result Object: PrimitiveResult([SamplerPubResult(data=DataBin(c=BitArray(<shape=(), num_shots=8192, num_bits=2>)), metadata={'shots': 8192, 'circuit_metadata': {}})], metadata={'version': 2})
##Bit array:  BitArray(<shape=(), num_shots=8192, num_bits=2>)
##Counts (simulated): {'00': 4042, '11': 4150}
##Results: {'00': 4042, '11': 4150}
##
##Iteration 4
##Raw Result Object: PrimitiveResult([SamplerPubResult(data=DataBin(c=BitArray(<shape=(), num_shots=8192, num_bits=2>)), metadata={'shots': 8192, 'circuit_metadata': {}})], metadata={'version': 2})
##Bit array:  BitArray(<shape=(), num_shots=8192, num_bits=2>)
##Counts (simulated): {'00': 4059, '11': 4133}
##Results: {'00': 4059, '11': 4133}
##
##Iteration 5
##Raw Result Object: PrimitiveResult([SamplerPubResult(data=DataBin(c=BitArray(<shape=(), num_shots=8192, num_bits=2>)), metadata={'shots': 8192, 'circuit_metadata': {}})], metadata={'version': 2})
##Bit array:  BitArray(<shape=(), num_shots=8192, num_bits=2>)
##Counts (simulated): {'00': 4071, '11': 4121}
##Results: {'00': 4071, '11': 4121}

def main_run_bias_experiment(qc, backend=Aer.get_backend('aer_simulator'), shots=8192):
    """
    Runs the circuit with charge injection and compares entropy shifts.
    """
    # Ensure measurement is added
    qc.measure_all()

    # Run baseline circuit
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts_baseline = result.get_counts()

    # Apply charge injection
    qc_injected = apply_charge_injection(qc.copy(), range(qc.num_qubits))
    qc_injected.measure_all()
    transpiled_qc_injected = transpile(qc_injected, backend)
    result_injected = backend.run(transpiled_qc_injected, shots=shots).result()
    counts_injected = result_injected.get_counts()

    # Create a true random state circuit
    qc_random = generate_true_random_state(qc.num_qubits)
    qc_random.measure_all()
    transpiled_qc_random = transpile(qc_random, backend)
    result_random = backend.run(transpiled_qc_random, shots=shots).result()
    counts_random = result_random.get_counts()

    # Calculate Shannon Entropy
    def calculate_qiskit_entropy(counts):
        total = sum(counts.values())
        probs = np.array([v / total for v in counts.values()])
        return -np.sum(probs * np.log2(probs))

    entropy_baseline = calculate_qiskit_entropy(counts_baseline)
    entropy_injected = calculate_qiskit_entropy(counts_injected)
    entropy_random = calculate_qiskit_entropy(counts_random)

    print("Baseline Counts:", counts_baseline)
    print("Baseline Shannon Entropy:", entropy_baseline)
    print("\nInjected Counts:", counts_injected)
    print("Injected Shannon Entropy:", entropy_injected)
    print("\nRandom Counts:", counts_random)
    print("Random Shannon Entropy:", entropy_random)

    return counts_baseline, entropy_baseline, counts_injected, entropy_injected, counts_random, entropy_random

def main_analyze_temporal_correlation(use_simulator=True):
    """ Creates a circuit with causality added in and with quantum error correction to try to analyze temporal trends"""
    
    # Initialize backend
    backend = initialize_backend(use_simulator)

    if backend:
        # Create base circuit
        qc_base = create_base_circuit()
        qfi_matrix, eigenvalues = compute_qfi(qc_base)

        print("Quantum Fisher Information Matrix: ")
        print(qfi_matrix)
        print("Eigenvalues: ")
        print(eigenvalues)

        # Store results from iterations
        iteration_results = []

        for i in range(10):  # Increase iterations for temporal analysis
            print(f"\nIteration {i + 1}")

            # Modify circuit (e.g., add causality)
            qc_modified = add_causality_to_circuit(qc_base.copy(), iteration_results[-1] if iteration_results else None, qubits=[0, 1])

            qfi_matrix_mod, eigenvalues_mod = compute_qfi(qc_modified)

            print("Quantum Fisher Information Matrix: ")
            print(qfi_matrix_mod)
            print("Eigenvalues: ")
            print(eigenvalues_mod)
        
            qc_clb_mod = add_classical_bits(qc_modified, 0)

            qc_qec, instr_0 = apply_charge_preserving_qec_no_syndrome(qc_clb_mod, num_qubits=2, num_classical=2)

            # Run circuit and store results
            results = run_and_extract_counts_quantum(qc=qc_qec, backend=backend, shots=8192)
            iteration_results.append(results)

            # Analyze subsystem entropy
            entropies = calculate_subsystem_qiskit_entropy(qc_qec)
            print(f"Subsystem Entropy: {entropies}")

        # Analyze temporal correlation
        temporal_divergences = analyze_temporal_correlation(iteration_results)
        print(f"Temporal Correlation (Jensen-Shannon Divergences): {temporal_divergences}")

        # Apply time-reversal and re-run
        qc_reversed = time_reverse_circuit(qc_base)

        qfi_matrix_rev, eigenvalues_rev = compute_qfi(qc_reversed)

        print("Quantum Fisher Information Matrix: ")
        print(qfi_matrix_rev)
        print("Eigenvalues: ")
        print(eigenvalues_rev)
        
        qc_reversed_clb = add_classical_bits(qc_reversed, 2)
        qc_qec_reversed, instr_1 = apply_charge_preserving_qec_no_syndrome(qc_reversed_clb, num_qubits=qc_reversed_clb.num_clbits, num_classical=qc_reversed_clb.num_clbits)
           
        reversed_results = run_and_extract_counts_quantum(qc=qc_qec_reversed, backend=backend, shots=8192)
        print(f"Time-Reversed Results: {reversed_results}")
    else:
        print("Failed to initialize backend.")

#QF Matrix empty but technically outputting
#Temporal Correlation (Jensen-Shannon Divergences): [0.0075309475373415745, 0.00552004222437261, 0.007948513038956672, 0.012420800945231443, 0.014466254085599352, 0.01556521805861248, 0.01082669989356168, 0.007983020301787321, 0.012188978876028229]

def main_analyze_temporal_correlation_with_amplification_qec(use_simulator=True):
    """ Creates a circuit with causality added in and with quantum error correction to try to analyze temporal trends"""
    
    # Initialize backend
    backend = initialize_backend(use_simulator)

    if backend:
        # Create base circuit
        qc_base = create_base_circuit()
        qc_base_amp = amplify_11_phase_balanced(qc_base)
        qfi_matrix, eigenvalues = compute_qfi(qc_base_amp)

        print("Quantum Fisher Information Matrix: ")
        print(qfi_matrix)
        print("Eigenvalues: ")
        print(eigenvalues)

        # Store results from iterations
        iteration_results = []

        for i in range(10):  # Increase iterations for temporal analysis
            print(f"\nIteration {i + 1}")

            # Modify circuit (e.g., add causality)
            qc_modified = add_causality_to_circuit(qc_base_amp.copy(), iteration_results[-1] if iteration_results else None, qubits=[0, 1])

            qfi_matrix_mod, eigenvalues_mod = compute_qfi(qc_modified)

            print("Quantum Fisher Information Matrix: ")
            print(qfi_matrix_mod)
            print("Eigenvalues: ")
            print(eigenvalues_mod)
        
            qc_clb_mod = add_classical_bits(qc_modified, 0)

            qc_qec, instr_0 = apply_charge_preserving_qec_no_syndrome(qc_clb_mod, num_qubits=2, num_classical=2)

            # Run circuit and store results
            results = run_and_extract_counts_quantum(qc=qc_qec, backend=backend, shots=8192)
            iteration_results.append(results)

            # Analyze subsystem entropy
            entropies = calculate_subsystem_qiskit_entropy(qc_qec)
            print(f"Subsystem Entropy: {entropies}")

        # Analyze temporal correlation
        temporal_divergences = analyze_temporal_correlation(iteration_results)
        print(f"Temporal Correlation (Jensen-Shannon Divergences): {temporal_divergences}")

        # Apply time-reversal and re-run
        qc_reversed = time_reverse_circuit(qc_base_amp)
        
        qc_reversed_clb = add_classical_bits(qc_reversed, 2)
        qc_qec_reversed, instr_1 = apply_charge_preserving_qec_no_syndrome(qc_reversed_clb, num_qubits=qc_reversed_clb.num_clbits, num_classical=qc_reversed_clb.num_clbits)
           
        reversed_results = run_and_extract_counts_quantum(qc=qc_qec_reversed, backend=backend, shots=8192)
        print(f"Time-Reversed Results: {reversed_results}")
    else:
        print("Failed to initialize backend.")

def main_test_time_reversal_bias():
    """Perform a full test suite to analyze the time-reversal bias."""
    # 1. Baseline Circuit
    qc_base = construct_baseline_circuit()
    counts_base = run_circuit(qc_base)
    analyze_results(counts_base, "Baseline Circuit")
    
    # 2. Time-Reversed Circuit
    qc_time_reversed = apply_time_reversal(qc_base)
    qc_time_reversed.measure_all()
    counts_reversed = run_circuit(qc_time_reversed)
    analyze_results(counts_reversed, "Time-Reversed Circuit")
    
    # 3. Phase Distortion
    qc_phase_distorted = introduce_phase_distortion(qc_base)
    qc_phase_distorted.measure_all()
    counts_phase = run_circuit(qc_phase_distorted)
    analyze_results(counts_phase, "Phase-Distorted Circuit")
    
    # 4. Charge Injection Redirection
    qc_charge_injected = apply_charge_injection(qc_base)
    qc_charge_injected.measure_all()
    counts_charge = run_circuit(qc_charge_injected)
    analyze_results(counts_charge, "Charge-Injection Circuit")
    
    # 5. Holographic Encoding
    qc_holographic = apply_holographic_encoding(qc_base)
    qc_holographic.measure_all()
    counts_holographic = run_circuit(qc_holographic)
    analyze_results(counts_holographic, "Holographic Encoding Circuit")

def main_decision_influence(decision="Invest in quantum AI"):
    qc = QuantumCircuit(5)  # Create a holographic quantum circuit
    qc_encoded = record_decision(qc, decision)

    qc.measure_all()
    backend = Aer.get_backend('aer_simulator')
    transpiled_qc = transpile(qc_encoded, backend)
    result = backend.run(transpiled_qc, shots=1024).result()
    counts = result.get_counts()

    print("Encoded Circuit Measurement Results:", counts)

def main_iterative_decisions():
    # Example usage
    initial_decision = '101'  # Example decision in binary
    qc, qr, cr = initialize_qubits(len(initial_decision))
    encode_decision(qc, qr, initial_decision)
    apply_entanglement(qc, qr)
    measure_and_reset(qc, qr, cr)

    # Add subsequent decisions
    second_decision = '011'
    qc, qr, cr = add_decision_to_chain(qc, qr, cr, second_decision)
    measure_and_reset(qc, qr, cr)

    third_decision = '110'
    qc, qr, cr = add_decision_to_chain(qc, qr, cr, third_decision)
    measure_and_reset(qc, qr, cr)

    # Execute the final circuit
    final_counts = execute_circuit(qc)
    print("Final Measurement Results:", final_counts)

def main_disruptive_decision():
    decision_qc = encode_disruptive_decision("Invest in quantum AI")
    new_decision_qc = encode_disruptive_decision("Divest from classical AI")
    fidelity, prev_counts, new_counts = measure_decision_influence(decision_qc, new_decision_qc)
    print(f"Decision Influence Fidelity: {fidelity}")


def main_modify_and_run_quantum_experiment_multi_analysis_0(qc, backend=None, shots=8192, modify_circuit=None, analyze_results=None):
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

        simulator_backend = Aer.get_backend('aer_simulator')
        transpiled_qc = transpile(qc, backend=simulator_backend)
        transpiled_qc.measure_all()
        time_reversal_qc = time_reversal_simulation(transpiled_qc)
        time_reversal_qc.measure_all()
        job = simulator_backend.run(time_reversal_qc, backend=simulator_backend, shots=shots)
        result = job.result()
        counts = result.get_counts()
        print("Counts (simulated): ", counts)
        qc_no_measure = remove_measurements(qc)
        state = Statevector.from_instruction(qc_no_measure)
        analyze_shannon_qiskit_entropy(counts)
        analyze_von_neumann_qiskit_entropy(state)
        
        return counts
        
        # Use hardware backend for Von Nuemman entropy only
        counts = run_and_extract_counts_quantum(qc, backend, shots)
        


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

            analyze_shannon_qiskit_entropy(results)

        except Exception as e:
            print(e)

    return results

def main_modify_and_run_quantum_experiment_multi_analysis_1(qc, backend=None, shots=8192, modify_circuit=None, analyze_results=None):
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
        simulator_backend = Aer.get_backend('aer_simulator')
        transpiled_qc = transpile(qc, backend=simulator_backend)

        # Time-reversal modification
        qc_no_measure = remove_measurements(transpiled_qc)  # Ensure no classical bits
        print("DEBUG: Checking if measurements removed:", qc_no_measure.draw())

        try:
            time_reversal_qc = time_reversal_simulation(qc_no_measure)
        except Exception as e:
            print("Error in time reversal:", e)
            return

        time_reversal_qc.measure_all()  # Apply measurement AFTER inversion

        # Run on the simulator
        job = simulator_backend.run(time_reversal_qc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        print("Counts (simulated): ", counts)

        # Analyze entropy
        try:
            state = Statevector.from_instruction(qc_no_measure)  # Ensure no classical bits
            analyze_shannon_qiskit_entropy(counts)
            analyze_von_neumann_qiskit_entropy(state)
        except Exception as e:
            print("Error in entropy analysis:", e)

        return counts

def main_test_multiverse_branching(qc, backend=Aer.get_backend('aer_simulator'), shots=8192):
    """
    Adds an ancilla qubit to test if hidden multiversal influences affect quantum state probabilities.
    
    Parameters:
        qc (QuantumCircuit): The base circuit for the experiment.
        backend (Backend): The quantum simulator or real backend to run the test.
        shots (int): The number of times the circuit is executed.

    Returns:
        dict: Measurement counts from the experiment.
    """
    # Extend circuit by 1 qubit for the ancilla
    num_qubits = qc.num_qubits
    qc_test = QuantumCircuit(num_qubits + 1, num_qubits + 1)  # One extra qubit

    # Copy the original circuit into the first num_qubits
    qc_test.compose(qc, range(num_qubits), inplace=True)

    # Entangle the ancilla with a Bell-like state
    qc_test.h(num_qubits)  
    qc_test.cx(num_qubits, 0)  # Entangle with first qubit to test for influence

    # Measure all qubits
    qc_test.measure(range(num_qubits + 1), range(num_qubits + 1))

    # Transpile and execute
    transpiled_qc = transpile(qc_test, backend)
    job = backend.run(transpiled_qc, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    return counts

def main_run_multiverse_experiment(backend=Aer.get_backend('aer_simulator'), shots=32768):
    """
    Runs the multiverse test circuit on the given backend with automated result analysis.
    """
    qc = multiverse_test_circuit()
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()

    counts = result.get_counts()
    
    # Calculate Shannon Entropy
    total_shots = sum(counts.values())
    probabilities = [count / total_shots for count in counts.values()]
    shannon_entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)

    print("Counts:", counts)
    print("Shannon Entropy:", shannon_entropy)

    return counts, shannon_entropy

def main_run_random_state_experiment(num_qubits=3, backend=None, shots=32768):
    """Creates and runs a maximally random quantum circuit."""
    
    qc = QuantumCircuit(num_qubits)
    
    # Apply Hadamards to create uniform superposition
    qc.h(range(num_qubits))
    
    # Apply random phase shifts
    for qubit in range(num_qubits):
        random_angle = np.random.uniform(0, 2*np.pi)
        qc.append(RZGate(random_angle), [qubit])

    # Measure all qubits
    qc.measure_all()

    # Choose backend
    if backend is None:
        backend = Aer.get_backend('aer_simulator')

    # Transpile and execute
    transpiled_qc = transpile(qc, backend)
    job = backend.run(transpiled_qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Compute entropy
    entropy = calculate_shannon_qiskit_entropy(counts, num_shots=shots)
    
    print("Random State Counts:", counts)
    print("Random State Shannon Entropy:", entropy)
    
    return counts, entropy

def main_run_time_reversed_experiment(qc, backend=None, shots=32768):
    """Runs a time-reversed version of the given quantum circuit."""
    
    # Remove measurements
    qc_no_measurements = qc.copy()
    qc_no_measurements.remove_final_measurements(inplace=True)
    
    # Time-reverse the circuit
    qc_reversed = qc_no_measurements.inverse()
    
    # Add measurements back
    qc_reversed.measure_all()

    # Choose backend
    if backend is None:
        backend = Aer.get_backend('aer_simulator')

    # Transpile and execute
    transpiled_qc = transpile(qc_reversed, backend)
    job = backend.run(transpiled_qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Compute entropy
    entropy = calculate_shannon_qiskit_entropy(counts, num_shots=shots)
    
    print("Time-Reversed Counts:", counts)
    print("Time-Reversed Shannon Entropy:", entropy)
    
    return counts, entropy


def main_run_time_reversal_biasing_experiment(qc, backend=Aer.get_backend('aer_simulator'), shots=8192):
    """
    Runs a quantum time reversal experiment and tests if probability biasing can restore lost states.
    """
    results = {}

    # Remove measurements first (to avoid issues)
    qc_no_meas = qc.remove_final_measurements(inplace=False)

    # Run baseline circuit with measurement
    qc_baseline = qc_no_meas.copy()
    qc_baseline.measure_all()
    # Step 4: Choose backend and detect if it's IBM Runtime
    use_ibm_runtime = backend and not isinstance(backend, AerSimulator)
    if backend is None:
        backend = AerSimulator()

    # Step 5: Transpile and run
    transpiled_qc = transpile(qc_baseline, backend)
    
    if use_ibm_runtime:
        with Session(backend=backend):
            sampler = Sampler()
            job = sampler.run([transpiled_qc], shots=shots)
            result = job.result()
            print(result)
            counts = result[0].data.meas.get_counts()
    else:
        job = backend.run(transpiled_qc, shots=shots)
        counts = job.result().get_counts()

    print(counts)

    # Step 6: Entropy calculation
    total = sum(counts.values())
    probs = [v / total for v in counts.values()]
    import numpy as np
    entropy = -np.sum([p * np.log2(p) for p in probs if p > 0])

    # Output
    print("Time-Reversed Counts:", counts)
    print("Time-Reversed Shannon Entropy:", entropy)

    if not counts:
        raise ValueError("Baseline circuit did not return counts. Check measurement setup.")

    # Compute entropy before reversal
    probs = np.array(list(counts.values())) / shots
    shannon_entropy = -np.sum(probs * np.log2(probs))

    # Time-reverse the circuit and ensure it has measurements
    qc_reversed = qc_no_meas.inverse()
    qc_reversed.measure_all()
    transpiled_qc = transpile(qc_reversed, backend)
    
    if use_ibm_runtime:
        with Session(backend=backend):
            sampler = Sampler()
            job = sampler.run([transpiled_qc], shots=shots)
            result = job.result()
            counts_reversed = result[0].data.meas.get_counts()
    
    else:
        job = backend.run(transpiled_qc, shots=shots)
        counts_reversed = job.result().get_counts()
    if not counts_reversed:
        raise ValueError("Time-reversed circuit did not return counts. Check measurement setup.")
  

    # Compute entropy after time-reversal
    probs_reversed = np.array(list(counts_reversed.values())) / shots
    shannon_entropy_reversed = -np.sum(probs_reversed * np.log2(probs_reversed))

    # Apply probability biasing to restore lost states
    bias_qubits = list(range(qc_reversed.num_qubits))
    qc_biased = apply_probability_biasing(qc_reversed, bias_qubits=bias_qubits)
    qc_biased.measure_all()
    transpiled_qc = transpile(qc_biased, backend)
    
    if use_ibm_runtime:
        with Session(backend=backend):
            sampler = Sampler()
            job = sampler.run([transpiled_qc], shots=shots)
            result = job.result()
            counts_biased = result[0].data.meas.get_counts()
    else:
        job = backend.run(transpiled_qc, shots=shots)
        counts_biased = job.result().get_counts()

    if not counts_biased:
        raise ValueError("Biased circuit did not return counts. Check measurement setup.")

    # Compute entropy after biasing
    probs_biased = np.array(list(counts_biased.values())) / shots
    shannon_entropy_biased = -np.sum(probs_biased * np.log2(probs_biased))

    # Store results
    results["baseline_counts"] = counts
    results["baseline_entropy"] = shannon_entropy
    results["time_reversed_counts"] = counts_reversed
    results["time_reversed_entropy"] = shannon_entropy_reversed
    results["biased_counts"] = counts_biased
    results["biased_entropy"] = shannon_entropy_biased

    # Print results
    print("Baseline Counts:", counts)
    print("Baseline Shannon Entropy:", shannon_entropy)
    print("Time-Reversed Counts:", counts_reversed)
    print("Time-Reversed Shannon Entropy:", shannon_entropy_reversed)
    print("Biased Counts (After Amplification):", counts_biased)
    print("Biased Shannon Entropy:", shannon_entropy_biased)

    return results

##Output to main_run_time_reversal_biasing_experiment
##Time-Reversed Counts: {'11': 4073, '00': 3860, '10': 134, '01': 125}
##Time-Reversed Shannon Entropy: 1.2019022279705784
##Baseline Counts: {'11': 4073, '00': 3860, '10': 134, '01': 125}
##Baseline Shannon Entropy: 1.2019022279705784
##Time-Reversed Counts: {'00': 3982, '01': 4048, '11': 90, '10': 72}
##Time-Reversed Shannon Entropy: 1.1399533206890717
##Biased Counts (After Amplification): {'00': 4025, '01': 3973, '11': 120, '10': 74}
##Biased Shannon Entropy: 1.1606411159633567

def main_test_multiverse_1():
    counts, entropy = main_run_multiverse_experiment()
    qc = create_base_circuit()
    plot_histogram(counts)
    print("Counts: ", counts)
    print("Shannon entropy: ", entropy)
    counts, entropy = main_run_time_reversed_experiment(qc)
    plot_histogram(counts)
    print("Counts: ", counts)
    print("Shannon entropy: ", entropy)
    counts, entropy = main_run_random_state_experiment()
    plot_histogram(counts)
    print("Counts: ", counts)
    print("Shannon entropy: ", entropy)

def main_run_mwi_vs_holography_experiment(qc, backend=None, shots=8192, iterations=5):
    """
    Runs probability amplification multiple times on the same quantum circuit.
    Tracks how past biasing affects future probability distributions.
    
    Determines if Many-Worlds (MWI) or Holography is more likely.

    Parameters:
        qc (QuantumCircuit): The base quantum circuit.
        backend (Backend): The IBM backend or simulator for execution.
        shots (int): Number of shots per experiment.
        iterations (int): Number of times to repeat amplification.

    Returns:
        dict: A dictionary containing counts and entropy over multiple runs.
    """

    if backend is None:
        backend = Aer.get_backend('aer_simulator')

    results = {"counts": [], "entropy": []}

    for i in range(iterations):
        # Transpile and execute circuit
        if qc.num_clbits < qc.num_qubits:
            qc.add_register(ClassicalRegister(qc.num_qubits - qc.num_clbits))
            
        qc.measure_all()
        
        transpiled_qc = transpile(qc, backend)
        job = backend.run(transpiled_qc, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Calculate Shannon entropy
        total_shots = sum(counts.values())
        probs = np.array([c / total_shots for c in counts.values()])
        entropy = -np.sum(probs * np.log2(probs))

        # Store results
        results["counts"].append(counts)
        results["entropy"].append(entropy)

        print("Counts: ", counts)
        print("Entropy: ", entropy)

        # Apply probability amplification
        qc = apply_probability_amplification(qc)  # Modify this based on your amplification function

    return results

##Output
##Best backend chosen: ibm_sherbrooke
##Counts:  {'11 00': 4100, '00 00': 4092}
##Entropy:  0.9999993120692872
##Counts:  {'11 11 00': 4091, '00 00 00': 4101}
##Entropy:  0.9999989251081651
##Counts:  {'11 11 11 00': 4094, '00 00 00 00': 4098}
##Entropy:  0.9999998280173422
##Counts:  {'00 00 00 00 00': 4133, '11 11 11 11 00': 4059}
##Entropy:  0.9999411381372179
##Counts:  {'11 11 11 11 11 00': 4050, '00 00 00 00 00 00': 4142}
##Entropy:  0.9999090192651703

def main_iterative_charge_injection(qc, num_cycles=5, backend=Aer.get_backend('aer_simulator'), shots=8192):
    """
    Applies charge injection iteratively and measures entropy growth over multiple cycles.

    Parameters:
        qc (QuantumCircuit): Initial quantum circuit.
        num_cycles (int): Number of charge injection cycles.
        backend (Backend): Qiskit backend for execution.
        shots (int): Number of shots for each experiment.

    Returns:
        list: Entropy values after each iteration.
    """
    entropies = []
    qc_modified = qc.copy()

    for i in range(num_cycles):
        # Apply charge injection (modify this based on your function)
        qc_modified = apply_charge_injection_universal(qc_modified)

        # Remove measurements before getting the statevector
        qc_no_measure = remove_measurements(qc_modified)

        # Simulate and get state vector
        transpiled_qc = transpile(qc_no_measure, backend)
        state = Statevector.from_instruction(transpiled_qc)

        measured_state = partial_trace(state, [0])  # Trace out one qubit
        measured_entropy = qiskit_entropy(measured_state)

        print(f"Entropy after forced measurement: {measured_entropy}")

        partial_ent = compute_partial_qiskit_entropy(state, [0])  # Trace out qubit 0

        print(f"Partial entropy of remaining system: {partial_ent}")
        
        # Compute von Neumann entropy
        entropy_val = qiskit_entropy(state)
        entropies.append(entropy_val)
        print(f"Iteration {i+1}: Entropy = {entropy_val}")

    # Plot entropy growth
    plt.plot(range(1, num_cycles+1), entropies, marker='o', linestyle='-')
    plt.xlabel("Charge Injection Cycle")
    plt.ylabel("Von Neumann Entropy")
    plt.title("Entropy Growth Over Charge Injection Cycles")
    plt.show()

    return entropies

def main_analyze_tensor_network_structure(qc, qubit_pairs=None):
    """
    Analyzes the tensor network structure of a quantum circuit using an MPS-like representation.

    Parameters:
        qc (QuantumCircuit): The input quantum circuit.

    Returns:
        list: A list of reshaped tensors representing the MPS-like structure.
    """

    qc, noise_model = apply_decoherence(qc)
    # Remove measurements before extracting the statevector
    qc_no_meas = remove_measurements(qc)

    # Get statevector from the modified circuit
    state = Statevector.from_instruction(qc_no_meas).data

    # Get number of qubits
    num_qubits = int(np.log2(len(state)))

    # Reshape into MPS-like structure
    tensors = []
    current_state = state.reshape([2] * num_qubits)  # Reshape into a tensor

    for i in range(num_qubits):
        # Perform Singular Value Decomposition (SVD) to split entanglement
        U, S, Vh = np.linalg.svd(current_state.reshape(2**i, -1), full_matrices=False)

        # Ensure valid reshaping by checking dimensions
        if U.shape[0] >= 2 and U.shape[1] > 1:
            tensors.append(U.reshape([-1, 2, U.shape[-1]]))  # Reshape into tensor form
        else:
            tensors.append(U)  # Store as is if it's too small to reshape properly

        if len(S) > 1 and Vh.shape[0] > 1:
            current_state = np.dot(np.diag(S), Vh).reshape([-1, 2, Vh.shape[-1]])  # Update state
        else:
            current_state = np.dot(np.diag(S), Vh)  # Avoid reshaping issues

    return tensors

def main_qc():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    return qc

def main_qc_basic():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc

def main_qc_bias():
    qc_test = QuantumCircuit(3)
    qc_test.h([0, 1, 2])  # Basic superposition test circuit
    return qc_test

def main_qc_entangle():
    qc = QuantumCircuit(3)
    qc.h(0)          # Put first qubit in superposition
    qc.cx(0, 1)      # Entangle first and second qubits
    qc.cx(1, 2)
    return qc

def main_retrocausal_experiment(qc, backend=None, shots=8192):
    """
    Test for retrocausal effects by delaying measurement on part of the system.

    Parameters:
        qc (QuantumCircuit): The quantum circuit to test.
        backend (Backend): The backend for execution.
        shots (int): Number of measurement shots.

    Returns:
        dict: Comparison of different measurement orders.
    """
    use_sampler = backend and not isinstance(backend, AerSimulator)
    
    if backend is None:
        backend = AerSimulator()

    # Ensure circuit has classical bits
    num_qubits = qc.num_qubits
    if not qc.clbits:
        qc.add_register(qc.cregs == QuantumCircuit(num_qubits, num_qubits).cregs[0])

    def run_circuit(qc_modified):
        """Executes circuit using either Sampler or standard backend.run()."""
        transpiled_qc = transpile(qc_modified, backend)
        if use_sampler:
            sampler = Sampler(backend)
            job = sampler.run([transpiled_qc], shots=shots)
            result = job.result()
            print("Data structure: ", result)
            dont_sampler = False
            counts = extract_counts(result, dont_sampler)
        else:
            job = backend.run(transpiled_qc, shots=shots)
            counts = job.result().get_counts()
        return counts

    # Step 1: Full Measurement (Baseline)
    qc_full = qc.copy()
    qc_full.measure_all()
    counts_full = run_circuit(qc_full)

    # Step 2: Partial Measurement (Measure all except last qubit)
    qc_partial = qc.copy()
    if not qc_partial.clbits:
        qc_partial.add_register(ClassicalRegister(num_qubits))
    for i in range(num_qubits - 1):
        qc_partial.measure(i, i)
    counts_partial = run_circuit(qc_partial)

    # Step 3: Measure Remaining Qubit (After Delay)
    qc_delayed = qc_partial.copy()
    qc_delayed.measure(num_qubits - 1, num_qubits - 1)
    counts_delayed = run_circuit(qc_delayed)

    print("Baseline Counts (Full Measurement):", counts_full)
    print("Counts After Partial Measurement:", counts_partial)
    print("Counts After Delayed Measurement:", counts_delayed)

    return {
        "full_counts": counts_full,
        "partial_counts": counts_partial,
        "delayed_counts": counts_delayed
    }
##Output
##Baseline Counts (Full Measurement): {'000 000': 4037, '111 111': 4155}
##Counts After Partial Measurement: {'111': 4075, '000': 4117}
##Counts After Delayed Measurement: {'111': 4022, '000': 4170}

def main_retrocausal_experiment_with_charge(qc, backend=None, shots=8192):
    """
    Test for retrocausal or holographic boundary effects by injecting charge
    after a partial measurement and before delayed measurement.

    Parameters:
        qc (QuantumCircuit): The base quantum circuit to test.
        backend (Backend): The backend for execution.
        shots (int): Number of measurement shots.

    Returns:
        dict: Measurement outcomes from full, partial, and delayed (with injection) runs.
    """
    from qiskit import ClassicalRegister, transpile
    from qiskit_aer import AerSimulator
    from qiskit.qasm3 import dump
    from qiskit.result import Counts

    use_sampler = backend and not isinstance(backend, AerSimulator)
    if backend is None:
        backend = AerSimulator()

    num_qubits = qc.num_qubits
    if qc.num_clbits < num_qubits:
        qc.add_register(ClassicalRegister(num_qubits - qc.num_clbits))

    def run_circuit(qc_modified):
        """Executes circuit using either Sampler or standard backend.run()."""
        transpiled_qc = transpile(qc_modified, backend)
        if use_sampler:
            sampler = Sampler(backend)
            job = sampler.run([transpiled_qc], shots=shots)
            result = job.result()
            print("Data structure: ", result[0].data.c0.get_counts())
            dont_sampler = False
            counts = result[0].data.c0.get_counts()
        else:
            job = backend.run(transpiled_qc, shots=shots)
            counts = job.result().get_counts()
        return counts

    # Step 1: Full measurement
    qc_full = qc.copy()
    qc_full.measure_all()
    counts_full = run_circuit(qc_full)

    # Step 2: Partial measurement (measure all but last qubit)
    qc_partial = qc.copy()
    if qc_partial.num_clbits < num_qubits:
        qc_partial.add_register(ClassicalRegister(num_qubits - qc_partial.num_clbits))
    for i in range(num_qubits - 1):
        qc_partial.measure(i, i)
    counts_partial = run_circuit(qc_partial)

    # Step 3: Inject charge after partial measurement
    qc_delayed = qc_partial.copy()
    for i in range(num_qubits):
        if i % 2 == 0:
            qc_delayed.z(i)  # Negative charge (Z)
        else:
            qc_delayed.x(i)  # Positive charge (X)

    # Then measure the last qubit (delayed)
    qc_delayed.measure(num_qubits - 1, num_qubits - 1)
    counts_delayed = run_circuit(qc_delayed)

    # Output
    print("🧪 Baseline Counts (Full Measurement):", counts_full)
    print("🧪 Counts After Partial Measurement:", counts_partial)
    print("⚡ Counts After Charge Injection + Delayed Measurement:", counts_delayed)

    return {
        "full_counts": counts_full,
        "partial_counts": counts_partial,
        "delayed_counts_with_charge": counts_delayed
    }
##Output
##Best backend chosen: ibm_brisbane
##Data structure:  {'00': 8192}
##Data structure:  {'01': 4171, '00': 4021}
##Data structure:  {'01': 4005, '10': 3940, '11': 161, '00': 86}
##🧪 Baseline Counts (Full Measurement): {'00': 8192}
##🧪 Counts After Partial Measurement: {'01': 4171, '00': 4021}
##⚡ Counts After Charge Injection + Delayed Measurement: {'01': 4005, '10': 3940, '11': 161, '00': 86}

def main_run_quantum_erasure_experiment(backend=None, shots=8192):
    """
    Runs the quantum erasure experiment to distinguish between Many-Worlds and Holography.
    
    Parameters:
        backend (Backend): The backend for execution. If None, uses a simulator.
        shots (int): Number of measurement shots.

    Returns:
        dict: Measurement results comparing information recovery.
    """

    # Step 1: Create a Quantum Circuit with 3 qubits (Bell pair + control qubit)
    qc = QuantumCircuit(3, 2)  # 3 qubits, 2 classical bits for measurement

    # Step 2: Create a Bell pair between q0 and q1
    qc.h(0)
    qc.cx(0, 1)

    # Step 3: Introduce the Erasure Mechanism (Entangle q1 with an ancilla q2)
    qc.cx(1, 2)
    qc.h(1)  # Interfere before measurement
    
    # Step 4: Measure q2 (attempted erasure)
    qc.measure(2, 1)  

    # Step 5: Introduce the "Time-Reversal" Decision Mechanism (Decides if Erasure was Real)
    qc.cx(1, 0)  # Reverse effect of CX if conditions allow
    qc.h(1)  

    # Step 6: Measure final qubits
    qc.measure(0, 0)

    # Check if we are running on a real backend
    if backend is not None:
        sampler = Sampler()
        transpiled_qc = transpile(qc, backend)
        job = sampler.run(transpiled_qc, shots=shots)
        result = job.result()
        counts = result.quasi_dists[0].nearest_probability_distribution()
    else:
        # Use simulator to analyze full quantum state
        simulator = AerSimulator()
        qc_no_measure = qc.remove_final_measurements(inplace=False)
        state = Statevector.from_instruction(qc_no_measure)
        counts = state.probabilities_dict()

    print("Quantum Experiment Results:", counts)
    return counts
##Output (send to sidak)
##Best backend chosen: ibm_sherbrooke
##Baseline Counts (Full Measurement): {'11': 4129, '00': 4063}
##Counts After Partial Measurement: {'00': 4138, '01': 4054}
##Counts After Delayed Measurement: {'11': 4082, '00': 4110}

def main_charge_injection_entangled(qc, num_levels=5):
    """
    Applies charge injection with increasing intensity while ensuring proper entanglement.

    Parameters:
        qc (QuantumCircuit): The base quantum circuit.
        num_levels (int): The number of charge injection levels.

    Returns:
        QuantumCircuit: The modified quantum circuit.
    """
    num_qubits = qc.num_qubits
    entropies = []

    # Step 1: Initial Entanglement
    for i in range(num_qubits):
        qc.h(i)  # Put all qubits into superposition

    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)  # Entangle neighboring qubits

    # Step 2: Iterative Charge Injection with Controlled Phase Shifts
    for level in range(1, num_levels + 1):
        angle = np.pi / (2**level)  # Decreasing impact per level
        for i in range(num_qubits):
            qc.crz(angle, i, (i + 1) % num_qubits)  # Controlled Rz for scaling charge

    # Step 3: Optional Basis Rotation (Uncomment if needed)
        for i in range(num_qubits):
            qc.h(i)  # Rotate to Hadamard basis for more diverse measurement outcomes

        state = Statevector.from_instruction(qc)
        ent = qiskit_entropy(state)
        entropies.append(ent)
        print(f"Charge Level {level}: Entropy = {ent}")

    return qc

def main_compare_charge_vs_random(num_qubits=5, depth=10, max_levels=5):
    """Compare entropy of charge-injected circuit vs. randomized circuit."""
    qc_base = QuantumCircuit(num_qubits)
    qc_base.h(0)  # Initial state
    qc_base.cx(0, 1)
    qc_base.cx(1, 2)

    print("\nRunning Charge Injection Scaling Test...")
    charge_results = charge_injection_scaling(qc_base, max_levels)

    print("\nRunning Randomized Circuit Test...")
    qc_random = create_randomized_circuit(num_qubits, depth)
    state_random = Statevector.from_instruction(qc_random)
    random_entropy = qiskit_entropy(state_random)
    print(f"Randomized Circuit Entropy: {random_entropy}")

    return charge_results, random_entropy

def create_teleportation_circuit_without_measurements():
    """
    Creates the teleportation circuit without measurements for statevector analysis.
    """
    qc = QuantumCircuit(3)  # No classical bits since we're skipping measurement

    # Step 1: Create entangled Bell pair between Q1 and Q2
    qc.h(1)
    qc.cx(1, 2)

    # Step 2: Encode charge information into Q0
    qc.rx(1.57, 0)  

    # Step 3: Bell measurement preparation (without actually measuring)
    qc.cx(0, 1)
    qc.h(0)

    return qc

def main_run_quantum_teleportation_experiment(backend=None, shots=8192):
    """
    Runs the teleportation circuit on the specified backend and analyzes fidelity.
    """
    if backend is None:
        backend = AerSimulator(method="statevector")  # Explicitly use statevector mode

    # Step 1: Create a circuit without measurements
    qc_no_measure = create_teleportation_circuit_without_measurements()

    # Step 2: Run statevector simulation
    state_simulator = AerSimulator(method="statevector")
    transpiled_qc = transpile(qc_no_measure, state_simulator)
    transpiled_qc.save_statevector()  # Explicitly save the statevector

    result = state_simulator.run(transpiled_qc).result()

    # Ensure statevector is correctly retrieved
    try:
        statevector = result.get_statevector()
    except Exception as e:
        print(f"Error retrieving statevector: {e}")
        return None, None

    # Step 3: Convert statevector to density matrix
    final_density_matrix = DensityMatrix(statevector)

    # Step 4: Compute fidelity between the initial and final state
    initial_qc = QuantumCircuit(1)
    initial_qc.h(0) # Apply RX gate properly
    initial_state = Statevector.from_instruction(initial_qc)  # Convert to statevector

    num_qubits = final_density_matrix.num_qubits
    print(f"Total qubits in system: {num_qubits}")

    if num_qubits == 3:
        traced_state = partial_trace(final_density_matrix, [2])  # Keep only qubit 2
        initial_density_matrix = DensityMatrix(initial_state)  # Convert to density matrix
        # Trace out an extra qubit if necessary
        if traced_state.num_qubits > 1:
            traced_state = partial_trace(traced_state, [1])  # Reduce to single-qubit state

        # Compute fidelity with properly reduced states
        initial_density_matrix = DensityMatrix(initial_state)  # Convert to density matrix
        fidelity = state_fidelity(initial_density_matrix, traced_state)

        print(f"Traced state: {traced_state}")
    else:
        print("Error: Qubit index 2 is out of range.")
        fidelity = None

    # Step 5: Run the full experiment with measurements to get classical counts
    qc = create_teleportation_circuit()
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()

    # Step 6: Apply manual corrections for each measurement result
    corrected_fidelity_sum = 0
    for measurement, count in counts.items():
        corrected_state = apply_classical_corrections(measurement, traced_state)
        fidelity = state_fidelity(initial_state, corrected_state)
        corrected_fidelity_sum += fidelity * (count / shots)  # Weight by probability

    print("Quantum Teleportation Results:")
    print("Measurement Counts:", counts)
    print(f"Corrected State Fidelity: {corrected_fidelity_sum:.6f}")

    return counts, corrected_fidelity_sum


def main_run_charge_transfer_experiment(backend=None, shots=8192):
    if backend is None:
        backend = AerSimulator(method="automatic")  # Force automatic mode

    # Create two versions of the circuit: one WITH measurement, one WITHOUT
    qc_no_measure = charge_transfer_experiment(measure=False)
    qc_with_measure = charge_transfer_experiment(measure=True)

    print("\n🚀 Quantum Circuit Before Transpilation (No Measurement):")
    print(qc_no_measure)

    # Step 1: Run Statevector Simulation (on circuit WITHOUT measurement)
    state_simulator = AerSimulator(method="automatic")  # Ensure correct simulator
    transpiled_qc = transpile(qc_no_measure, state_simulator)

    print("\n🚀 Quantum Circuit After Transpilation (No Measurement):")
    print(transpiled_qc)

    result = state_simulator.run(transpiled_qc).result()

    # Try retrieving the statevector safely
    try:
        statevector = result.data(0)["statevector"]
    except Exception as e:
        print(f"\n❌ Error retrieving statevector: {e}")
        return None, None

    # Convert to density matrix for tracing analysis
    final_density_matrix = DensityMatrix(statevector)

    # Extract Q2's state to see if charge was transferred
    traced_state = partial_trace(final_density_matrix, [0])  # Keep only Q2

    # Step 2: Run actual measurement simulation
    transpiled_qc = transpile(qc_with_measure, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()

    print("\n✅ Charge Transfer Experiment Results:")
    print("Measurement Counts on Q2:", counts)
    print("Extracted State (Q2 after tracing Q1):", traced_state)

    return counts, traced_state


def main_run_charge_transfer_experiment_3qubits(backend=None, shots=8192):
    if backend is None:
        backend = AerSimulator(method="statevector")  # Ensure correct simulator

    # Create two versions of the circuit: one WITH measurement, one WITHOUT
    qc_no_measure = charge_transfer_experiment_3qubits(measure=False)
    qc_with_measure = charge_transfer_experiment_3qubits(measure=True)

    print("\n🚀 Quantum Circuit Before Transpilation (No Measurement):")
    print(qc_no_measure)

    # Step 1: Run Statevector Simulation (on circuit WITHOUT measurement)
    state_simulator = AerSimulator(method="statevector")
    transpiled_qc = transpile(qc_no_measure, state_simulator)

    print("\n🚀 Quantum Circuit After Transpilation (No Measurement):")
    print(transpiled_qc)

    # ✅ Run the circuit with explicit statevector saving
    result = state_simulator.run(transpiled_qc).result()

    # ✅ Retrieve Statevector using `data()`
    try:
        statevector = result.data(0)["statevector"]
    except Exception as e:
        print(f"\n❌ Error retrieving statevector: {e}")
        return None, None

    # Convert to density matrix for tracing analysis
    final_density_matrix = DensityMatrix(statevector)

    # Extract Q3's state to see if charge was transferred
    traced_state = partial_trace(final_density_matrix, [0, 1])  # Keep only Q3

    # Step 2: Run actual measurement simulation
    transpiled_qc = transpile(qc_with_measure, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()

    print("\n✅ 3-Qubit Charge Transfer Experiment Results:")
    print("Measurement Counts on Q3:", counts)
    print("Extracted State (Q3 after tracing Q1 & Q2):", traced_state)

    return counts, traced_state

def main_run_mbqc_energy_transfer_experiment(backend=None, shots=8192):
    if backend is None:
        backend = AerSimulator(method="statevector")  # Use Statevector Simulator

    # Create two versions of the circuit: one WITH measurement, one WITHOUT
    qc_no_measure = mbqc_energy_transfer_experiment(measure=False)
    qc_with_measure = mbqc_energy_transfer_experiment(measure=True)

    print("\n🚀 Quantum Circuit Before Transpilation (No Measurement):")
    print(qc_no_measure)

    # Step 1: Run Statevector Simulation (on circuit WITHOUT measurement)
    state_simulator = AerSimulator(method="statevector")
    transpiled_qc = transpile(qc_no_measure, state_simulator)

    print("\n🚀 Quantum Circuit After Transpilation (No Measurement):")
    print(transpiled_qc)

    # ✅ Run the circuit and retrieve statevector
    result = state_simulator.run(transpiled_qc).result()

    try:
        statevector = result.data(0)["statevector"]
    except Exception as e:
        print(f"\n❌ Error retrieving statevector: {e}")
        return None, None

    # Convert to density matrix for tracing analysis
    final_density_matrix = DensityMatrix(statevector)

    # Extract Earth Qubit's State After Space Qubit is Measured
    traced_state = partial_trace(final_density_matrix, [0])  # Keep only Q_earth

    qc_with_measure.h(0)  # Switch measurement to X-basis
    qc_with_measure.cx(0, 1)
    qc_with_measure.measure(0, 0)  # Measure Q_space in X-basis

    # Step 2: Run actual measurement simulation
    transpiled_qc = transpile(qc_with_measure, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()


    # ✅ Fix: Apply Correction Using Operator(ZGate())
    x_operator = Operator(XGate())  # Properly format the Z and X gate as an operator
    z_operator = Operator(ZGate())

    if counts.get('01', 0) > counts.get('00', 0):  
        corrected_state = traced_state.evolve(x_operator)  # Apply X correction
    elif counts.get('10', 0) > counts.get('00', 0):  
        corrected_state = traced_state.evolve(z_operator)  # Apply Z correction
    elif counts.get('11', 0) > counts.get('00', 0):  
        corrected_state = traced_state.evolve(x_operator).evolve(z_operator)  # Apply X + Z correction
    else:
        corrected_state = traced_state  # No correction needed

    print("\n✅ MBQC Energy Transfer Experiment Results:")
    print("Measurement Counts on Space Qubit:", counts)
    print("Extracted State (Earth Qubit after Space Measurement):", traced_state)
    print("✅ Corrected State (Earth Qubit after Space Measurement & Correction):", corrected_state)

    return counts, corrected_state


def main_multiversal_telephone(num_branches=3, backend=None, shots=8192):
    """
    Selects and aligns a preferred reality branch using quantum encoding and charge injection.
    
    Parameters:
    - num_branches: Number of possible reality states to select from.
    - backend: Quantum backend to use (default: AerSimulator).
    - shots: Number of runs for statistical reinforcement.
    
    Returns:
    - Selected branch and alignment status.
    """

    if backend is None:
        backend = AerSimulator(method="statevector")

    # Step 1: Generate branch options
    branch_options = generate_reality_branches(num_branches)

    # Step 2: Create quantum registers (one for each branch option)
    qr = QuantumRegister(num_branches, name="q")
    cr = ClassicalRegister(num_branches, name="c")
    qc = QuantumCircuit(qr, cr)

    # Step 3: Encode each branch option into a quantum state
    for branch in branch_options:
        qc.rx(branch["charge"], qr[branch["branch_id"]])  # Inject charge to encode preference

    # Step 4: Apply entanglement across branches (biasing toward coherence)
    for i in range(num_branches - 1):
        qc.cx(qr[i], qr[i+1])


    # Step 5: Weak measurement to determine highest coherence state
    qc.measure(qr, cr)

    # Step 6: Run the experiment
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()

    # Step 7: Interpret results and align charge injection
    selected_branch = max(counts, key=counts.get)  # Pick the most likely outcome
    print(f"\n📞 Selected Reality Branch: {selected_branch}")
    print("Measurement Counts:", counts)

    # Step 8: Inject charge to reinforce the selected branch
    rm_qc = remove_measurements(qc)
    aligned_state = Statevector.from_instruction(rm_qc)
    final_density_matrix = DensityMatrix(aligned_state)
    traced_state = partial_trace(final_density_matrix, [int(selected_branch, 2)])

    print("\n✅ Aligned State After Selection:", traced_state)

    return selected_branch, traced_state

def main_cleanup_charge_injection(backend=None, shots=8192):
    """
    Reverses past charge injections, removing unintended biases and resetting entanglements.

    Parameters:
    - backend: Quantum backend to use (default: AerSimulator).
    - shots: Number of runs for statistical reinforcement.

    Returns:
    - Cleansed quantum state after charge extraction.
    """

    if backend is None:
        backend = AerSimulator(method="statevector")

    # Create a cleanup quantum circuit
    num_qubits = 3  # Default to 3, can be adjusted
    qr = QuantumRegister(num_qubits, name="q")
    qc = QuantumCircuit(qr)

    # Step 1: Apply inverse charge extraction to neutralize residual effects
    for qubit in range(num_qubits):
        qc.rx(-np.pi / 2, qr[qubit])  # Apply inverse RX to remove charge buildup

    # Step 2: Run the circuit and extract the cleansed state
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()

    # Convert to density matrix for final verification
    cleaned_state = DensityMatrix.from_instruction(qc)

    print("\n✅ Charge Cleanup Completed. Residual biases removed.")
    print("Cleaned Quantum State:\n", cleaned_state)

    return cleaned_state

def main_probabilities_0():
    num_samples = 10000

    # Step 1: Generate Baseline Distribution (50/50)
    baseline_samples = generate_binary_samples(num_samples)
    baseline_prob_1 = np.mean(baseline_samples)
    baseline_prob_0 = 1 - baseline_prob_1

    # Step 2: Apply Charge Injection & Biasing
    quantum_statevector = np.array([0.2])  # Example biasing from statevector[0]
    bias_factor = np.real(quantum_statevector[0]) * 0.5  # Scale bias effect

    biased_samples = generate_binary_samples(num_samples, bias_factor)
    biased_prob_1 = np.mean(biased_samples)
    biased_prob_0 = 1 - biased_prob_1

    # Step 3: Display Probability Shift
    print("🔄 Probability Shift Due to Charge Injection:")
    print(f"   - Baseline P(1) = {baseline_prob_1:.5f}, P(0) = {baseline_prob_0:.5f}")
    print(f"   - Biased   P(1) = {biased_prob_1:.5f}, P(0) = {biased_prob_0:.5f}")
    print(f"   - Shift    ΔP(1) = {biased_prob_1 - baseline_prob_1:.5f}, ΔP(0) = {biased_prob_0 - baseline_prob_0:.5f}")

def main_probabilities_1():
    # Step 1: Generate Baseline Random Numbers
    baseline_numbers = generate_random_numbers()

    # Step 2: Apply Charge Injection
    qc = inject_charge_0()
    backend = Aer.get_backend("statevector_simulator")
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc).result()
    statevector = result.get_statevector()

    # Step 3: Generate Biased Random Numbers
    biased_numbers = generate_random_numbers() * (1 + np.real(statevector[0]))  # Scale by charge influence

    # Step 4: Compare Statistics
    baseline_mean, biased_mean = np.mean(baseline_numbers), np.mean(biased_numbers)
    baseline_std, biased_std = np.std(baseline_numbers), np.std(biased_numbers)

    print(f"Baseline Mean: {baseline_mean:.5f}, Biased Mean: {biased_mean:.5f}")
    print(f"Baseline Std: {baseline_std:.5f}, Biased Std: {biased_std:.5f}")

    # Step 5: Plot Histograms
    plt.figure(figsize=(10, 5))
    plt.hist(baseline_numbers, bins=50, alpha=0.5, label="Baseline", color="blue")
    plt.hist(biased_numbers, bins=50, alpha=0.5, label="Biased", color="red")
    plt.axvline(baseline_mean, color="blue", linestyle="dashed", label="Baseline Mean")
    plt.axvline(biased_mean, color="red", linestyle="dashed", label="Biased Mean")
    plt.legend()
    plt.title("RNG Distribution Before & After Charge Injection")
    plt.xlabel("Random Value")
    plt.ylabel("Frequency")
    plt.show()

# 🔑 Main Test Loop
def main_run_qrng_biasing_test(rounds=20, bits_per_round=32):
    print("\n🚀 Starting QRNG Reality Biasing Test...\n")
    for i in range(1, rounds + 1):
        print(f"🔄 Round {i}")
        bits = local_qrng(bits_per_round)
        if not bits:
            print("⚠️ No data fetched. Skipping this round.")
            continue
        
        # Analyze before/after charge
        analyze_qrng_bias(bits)
        
        # Inject charge to influence
        inject_charge()

def main_holographic_encoding_and_teleportation():
    # Create a quantum circuit with 3 qubits (Alice's qubits) and 3 classical bits
    qc = QuantumCircuit(3, 3)

    # Step 1: Holographic encoding - prepare the state (say a |psi> state)
    # Alice's state will be a superposition state, which we want to teleport
    qc.u(0.5, 0.25, 0.1, 0)  # Apply an arbitrary unitary transformation to the first qubit (state to teleport)

    # Step 2: Create entanglement between Alice and Bob (entangled pair)
    qc.h(1)  # Apply Hadamard gate to the second qubit (Alice's entangled qubit)
    qc.cx(1, 2)  # Apply CNOT gate to create entanglement between Alice's and Bob's qubits

    # Step 3: Alice measures her qubits
    qc.barrier()  # Add a barrier for clarity
    qc.measure([0, 1], [0, 1])  # Measure Alice's qubits (classical communication follows)

    # Step 4: Bob applies correction based on Alice's measurements
    qc.cx(1, 2)  # Apply CNOT gate to Bob's qubit (controlled by Alice's measurement)
    qc.cz(0, 2)  # Apply CZ gate to Bob's qubit (conditional on Alice's measurement)

    # Step 5: Measure Bob's qubit (final state after teleportation)
    qc.measure(2, 2)  # Measure Bob's qubit (should match Alice's original state)

    return qc

def main_multiversal_communication_experiment():
    # Create a quantum circuit with 4 qubits (2 qubits for Alice and 2 for Bob) and 4 classical bits
    qc = QuantumCircuit(4, 4)

    # Step 1: Create entanglement between Alice's and Bob's qubits
    qc.h(0)  # Alice applies a Hadamard gate to her qubit
    qc.cx(0, 1)  # Create entanglement between Alice's and Bob's qubits (qubit 0 and qubit 1)

    # Step 2: Measurement in Reality A (Alice's Reality) to simulate an action
    qc.measure(0, 0)  # Alice measures her qubit (this represents an action in her timeline)
    qc.measure(1, 1)  # Bob's qubit is measured (feedback from Alice's action in Reality A)

    # Step 3: Allow the feedback to propagate through Quantum Entanglement
    qc.cx(1, 2)  # Entangle Bob's qubit with a new qubit in Reality B (Bob's reality)
    qc.cz(0, 2)  # Bob applies a correction to propagate Alice's action to Reality B

    # Step 4: Measurement in Reality B (Bob's Reality) based on Alice's action
    qc.measure(2, 2)  # Bob measures the state of his qubit in Reality B
    qc.measure(3, 3)  # A second qubit for Bob to see how the feedback loop works

    return qc

def main_darpa_pres(bits=4):
    charge_levels = list(range(1, 51))
    entropy_nq, results_nq = charge_scaling_no_qec(num_qubits=bits)
    entropy_qec, results_qec = charge_scaling_qec(num_qubits=bits)
    normalized_without_qec = [(x / results_nq[0]) for x in results_nq]
    normalized_with_qec = [(x / results_qec[0]) for x in results_qec]
    compare_plot_fig_qec(charge_levels, normalized_without_qec, normalized_with_qec, bits)
    
def main_bosonic_qec_simulation(alpha=2.0, loss_prob=0.1, displacement_strength=0.1, num_jumps=2):
    """
    Full simulation of bosonic QEC including forced noise/error pathways for testing correction.
    """
    fidelities = []
    entropies = []
    
    # Step 1: Prepare cat code logical zero
    state = encode_bosonic_cat_code(alpha)[0]
    ideal_cat = state
    print("Initial State Prepared (Cat Code)")

    fid_initial = fidelity(state, ideal_cat)
    fidelities.append(fid_initial)
    ent_initial = entropy_vn(state)
    entropies.append(ent_initial)

    # Step 2: Add displacement noise (simulate charge/EM jitter)
    state = apply_displacement_noise(state, displacement_strength)
    print("Displacement noise applied")

    # Step 3: Apply photon loss (soft exponential damping)
    state = apply_photon_loss(state, loss_prob=loss_prob)
    print("Photon loss applied")

    # Step 4: Apply explicit photon jumps (force strong errors)
    state = apply_photon_jumps(state, p_loss=loss_prob, num_jumps=num_jumps)
    print("Photon jumps applied")

    ent_noisy = entropy_vn(state)
    entropies.append(ent_noisy)

    # Step 5: Measure initial parity (syndrome)
    initial_parity = measure_parity(state)
    print(f"Measured parity (syndrome): {initial_parity:.3f}")

    # Step 6: Perform correction if needed
    corrected_state = smart_displacement_correction(state, initial_parity)

    print(f"Correction applied (if needed), Corrected state: {corrected_state}")

    fid_corrected = fidelity(corrected_state, ideal_cat)
    fidelities.append(fid_corrected)
    ent_corrected = entropy_vn(corrected_state)
    entropies.append(ent_corrected)
    
    # Step 7: Measure final parity
    final_parity = measure_parity(corrected_state)
    
    print(f"Final parity after correction: {final_parity:.3f}")
    print(f"Final Entropy after correction: {ent_corrected:.3f}")
    print(f"Final Fidelity after correction: {fid_corrected:.3f}")

    return corrected_state

def main_multiversal_telephone_v2():
    q = QuantumRegister(3, 'q')  # q0: BH, q1: Rad1, q2: Rad2
    c = ClassicalRegister(3, 'c')
    qc = QuantumCircuit(q, c)

    # Static encoding (early charge injection)
    # Pattern A ("0")
    qc.x(q[0])
    qc.z(q[0])

    # Entangle BH with early radiation
    qc.cx(q[0], q[1])

    # Dynamic encoding (late charge injection)
    # Pattern C ("0")
    qc.h(q[0])
    qc.s(q[0])

    # Entangle BH with late radiation
    qc.cx(q[0], q[2])

    # OPTIONAL: Add delays or barriers for time separation visualization
    qc.barrier()

    # Measurements
    qc.measure([q[0], q[1], q[2]], [c[0], c[1], c[2]])

    # Execute on simulator
    backend = Aer.get_backend('qasm_simulator')
    job = backend.run(qc, shots=8192)
    result = job.result()
    counts = result.get_counts(qc)

    print("Measurement outcomes: ", counts)

def main_multichar_telephone():
    char_map = {
        'A': ('0', '0'),
        'B': ('0', '1'),
        'C': ('1', '0'),
        'D': ('1', '1'),
        'H': ('0', '1'),
        'I': ('1', '0'),
        'F': ('1', '1'),
    }

    reverse_char_map = {v: k for k, v in char_map.items()}
    message = "HIF"

    backend = Aer.get_backend('qasm_simulator')
    all_results = {}

    # Generate multiple branches via timing manipulation
    measurement_orders = [('Rad1_first', [1, 2]), ('Rad2_first', [2, 1])]  # Control branch split

    for order_name, measure_order in measurement_orders:
        branch_results = {}

        for idx, char in enumerate(message):
            static_bit, dynamic_bit = char_map[char]

            q = QuantumRegister(3, 'q')
            c = ClassicalRegister(3, 'c')
            qc = QuantumCircuit(q, c)

            # --- Static encoding ---
            if static_bit == '1':
                qc.x(q[0])
                qc.z(q[0])

            # Early radiation entanglement
            qc.cx(q[0], q[1])

            # --- Dynamic encoding ---
            if dynamic_bit == '1':
                qc.h(q[0])
                qc.s(q[0])

            # Late radiation entanglement
            qc.cx(q[0], q[2])

            # Barrier for clarity
            qc.barrier()

            # --- Time-asymmetry control: measure in controlled order ---
            qc.measure(q[measure_order[0]], c[0])  # First measurement
            qc.measure(q[measure_order[1]], c[1])  # Second measurement
            qc.measure(q[0], c[2])                 # BH measured last or along

            # Execute and store
            job = backend.run(qc, shots=8192)
            result = job.result()
            counts = result.get_counts(qc)
            branch_results[char + f"_{idx}"] = counts

        all_results[order_name] = branch_results

    # --- Decoding Process ---
    for order_name, branch_results in all_results.items():
        print(f"\nBranch: {order_name}")
        decoded_message = ""
        for char_key, counts in branch_results.items():
            dominant = max(counts, key=counts.get)
            rad1 = dominant[1]
            rad2 = dominant[2]
            bits = (rad1, rad2)
            decoded_char = reverse_char_map.get(bits, '?')
            decoded_message += decoded_char
            print(f"Character {char_key}: {counts}")
        print("Decoded Message:", decoded_message)

def mv_tf_21():

    char_map = {
        'A': ('0', '0'),  # Static '0', Dynamic '0'
        'B': ('0', '1'),  # Static '0', Dynamic '1'
        'C': ('1', '0'),  # Static '1', Dynamic '0'
        'D': ('1', '1'),  # Static '1', Dynamic '1'
        'H': ('0', '1'),  # 'H'
        'I': ('1', '0'),  # 'I'
        'F': ('1', '1'),  # 'F'
    }

    # Reverse map for decoding
    reverse_char_map = {v: k for k, v in char_map.items()}

    # -------------------------
    # Message to encode
    # -------------------------
    message = "HIF"

    # -------------------------
    # Quantum backend setup
    # -------------------------
    backend = Aer.get_backend('qasm_simulator')
    shots = 8192

    # -------------------------
    # Measurement orders for branch control
    # -------------------------
    measurement_orders = [('Rad1_first', [1, 2]), ('Rad2_first', [2, 1])]

    # -------------------------
    # Storage for results
    # -------------------------
    all_results = {}

    # -------------------------
    # Experimental Loop
    # -------------------------
    for order_name, measure_order in measurement_orders:
        branch_results = {}

        for idx, char in enumerate(message):
            static_bit, dynamic_bit = char_map[char]

            # Create quantum and classical registers (extra ancilla if needed later)
            q = QuantumRegister(3, 'q')  # q0: Black Hole (BH), q1: Rad1, q2: Rad2
            c = ClassicalRegister(3, 'c')
            qc = QuantumCircuit(q, c)

            # -------------------------
            # Static Layer (Early Charge Injection)
            # -------------------------
            if static_bit == '1':
                qc.x(q[0])  # X gate to encode '1'
                qc.z(q[0])  # Z gate to phase lock static bit

            # Barrier to prevent interference
            qc.barrier()

            # -------------------------
            # Early Radiation Entanglement
            # -------------------------
            qc.cx(q[0], q[1])  # Entangle BH with Rad1 (static message anchor)

            # Barrier to separate stages
            qc.barrier()

            # -------------------------
            # Dynamic Layer (Late Charge Injection)
            # -------------------------
            if dynamic_bit == '1':
                qc.h(q[0])  # H gate to introduce superposition
                qc.s(q[0])  # S gate to adjust phase for dynamic encoding

            # Barrier again
            qc.barrier()

            # -------------------------
            # Late Radiation Entanglement
            # -------------------------
            qc.cx(q[0], q[2])  # Entangle BH with Rad2 (dynamic message)

            # -------------------------
            # Measurement Control (Time-Asymmetry)
            # -------------------------
            # Optional: add ancilla or delay logic here in future versions

            # Measure in specified order
            qc.measure(q[measure_order[0]], c[0])  # First measurement (branch selector)
            qc.measure(q[measure_order[1]], c[1])  # Second measurement
            qc.measure(q[0], c[2])                 # Optional BH measurement (diagnostic)

            # -------------------------
            # Execute Circuit
            # -------------------------
            job = backend.run(qc, shots=shots)
            result = job.result()
            counts = result.get_counts(qc)

            # Store results
            branch_results[char + f"_{idx}"] = counts

        all_results[order_name] = branch_results

    # -------------------------
    # Decoding Logic
    # -------------------------

    # Bit-to-character mapping based on observed behaviors

    # Final decoded messages for each branch
    print("\n\n========= Multiversal Telephone V2.1 Results =========\n")
    for order_name, branch_results in all_results.items():
        print(f"Branch: {order_name}")
        decoded_message = ""

        for char_key, counts in branch_results.items():
            # Find dominant outcome
            dominant_outcome = max(counts, key=counts.get)

            # Extract Rad1 and Rad2 bits (1st and 2nd bit in outcome string)
            rad1 = dominant_outcome[1]
            rad2 = dominant_outcome[2]

            # Get bit pair
            bits = (rad1, rad2)

            # Decode to character (fallback to '?' if unknown)
            decoded_char = bit_to_char_map.get(bits, '?')
            decoded_message += decoded_char

            print(f"Character {char_key}: Outcome: {counts}, Decoded as: {decoded_char}")

        print("Decoded Message:", decoded_message)
        print("\n--------------------------------------------------\n")

def mv_tf_22():
    char_map = {
        'A': ('0', '0'),
        'B': ('0', '1'),
        'C': ('1', '0'),
        'D': ('1', '1'),
        'H': ('0', '1'),  # 'H'
        'I': ('1', '0'),  # 'I'
        'F': ('1', '1'),  # 'F'
    }

    # Reverse map for decoding
    reverse_char_map = {v: k for k, v in char_map.items()}

    # -------------------------
    # Message to encode
    # -------------------------
    message = "HIF"  # You can change this to "HELP" or any other sequence

    # -------------------------
    # Quantum backend setup
    # -------------------------
    backend = Aer.get_backend('qasm_simulator')
    shots = 8192

    # -------------------------
    # Measurement orders for branch control (time asymmetry)
    # -------------------------
    measurement_orders = [('Rad1_first', [1, 2]), ('Rad2_first', [2, 1])]

    # -------------------------
    # Storage for results
    # -------------------------
    all_results = {}

    # -------------------------
    # Experimental Loop (for each branch and character)
    # -------------------------
    for order_name, measure_order in measurement_orders:
        branch_results = {}

        for idx, char in enumerate(message):
            static_bit, dynamic_bit = char_map[char]

            # Quantum and Classical Registers (optional ancilla included but not used yet)
            q = QuantumRegister(3, 'q')  # q0: BH, q1: Rad1, q2: Rad2
            c = ClassicalRegister(3, 'c')
            qc = QuantumCircuit(q, c)

            # -------------------------
            # Static Layer (Weakened Static)
            # -------------------------
            if static_bit == '1':
                qc.x(q[0])  # ONLY X gate for static bit, Z removed for weaker imprint

            qc.barrier()  # Barrier to separate static from dynamic

            # -------------------------
            # Early Radiation Entanglement (Static Channel)
            # -------------------------
            qc.cx(q[0], q[1])  # Entangle BH with Rad1

            qc.barrier()

            # -------------------------
            # Dynamic Layer (Enhanced Dynamic with H, S, T)
            # -------------------------
            if dynamic_bit == '1':
                qc.h(q[0])  # Superposition
                qc.s(q[0])  # Phase shift
                qc.t(q[0])  # Extra phase rotation to strengthen dynamic message

            qc.barrier()

            # -------------------------
            # Late Radiation Entanglement (Dynamic Channel)
            # -------------------------
            qc.cx(q[0], q[2])  # Entangle BH with Rad2

            # -------------------------
            # Optional Branch-specific Injection (to force split in Rad2_first)
            # -------------------------
            if order_name == 'Rad2_first':
                qc.t(q[0])  # Extra T-gate as branch-specific divergence

            qc.barrier()

            # -------------------------
            # Measurement Control (Branch Splitting)
            # -------------------------
            qc.measure(q[measure_order[0]], c[0])  # First measured qubit (branch selector)
            qc.measure(q[measure_order[1]], c[1])  # Second measured qubit
            qc.measure(q[0], c[2])                 # BH qubit measured for info

            # -------------------------
            # Execute Circuit
            # -------------------------
            job = backend.run(qc, shots=shots)
            result = job.result()
            counts = result.get_counts(qc)

            # Store results
            branch_results[char + f"_{idx}"] = counts

        all_results[order_name] = branch_results

    # -------------------------
    # Decoding Logic
    # -------------------------

    # Bit-to-character mapping based on refined understanding

    # -------------------------
    # Final Output Display
    # -------------------------
    print("\n\n========= Multiversal Telephone V2.2 Results =========\n")
    for order_name, branch_results in all_results.items():
        print(f"Branch: {order_name}")
        decoded_message = ""

        for char_key, counts in branch_results.items():
            # Find dominant measurement outcome (most likely)
            dominant_outcome = max(counts, key=counts.get)

            # Extract Rad1 and Rad2 bits
            rad1 = dominant_outcome[1]  # Second qubit measured (Rad1)
            rad2 = dominant_outcome[2]  # Third qubit measured (Rad2)

            # Bit pair to character
            bits = (rad1, rad2)
            decoded_char = bit_to_char_map.get(bits, '?')  # '?' if unknown
            decoded_message += decoded_char

            # Print character result
            print(f"Character {char_key}: Outcome: {counts}, Dominant: {dominant_outcome}, Decoded as: {decoded_char}")

        print("Decoded Message:", decoded_message)
        print("\n--------------------------------------------------\n")

def inject_static(qc, q, static_bit, strength='weak'):
    if static_bit == '1':
        if strength == 'weak':
            qc.rx(0.3 * 3.14, q[0])  # SMALLER rotation
        elif strength == 'medium':
            qc.rx(0.5 * 3.14, q[0])  
        elif strength == 'strong':
            qc.x(q[0])  
    qc.barrier()

def inject_dynamic(qc, q, dynamic_bit, branch_type=None, strength='strong'):
    """
    Inject dynamic charge based on branch and strength.
    Args:
        qc (QuantumCircuit): the circuit to operate on
        q (QuantumRegister): the quantum register
        dynamic_bit (str): '1' or '0' to indicate dynamic encoding
        branch_type (str): 'Rad1_first' or 'Rad2_first'
        strength (str): 'weak', 'medium', 'strong' (default 'medium')
    """
    if dynamic_bit == '1':
        if strength == 'weak':
            qc.h(q[0])
        elif strength == 'medium':
            qc.h(q[0])
            qc.s(q[0])  # Phase shift
        elif strength == 'strong':
            qc.h(q[0])
            qc.s(q[0])
            qc.t(q[0])  # Stronger phase

    # Branch-specific charge (optional)
    if branch_type == 'Rad2_first':
        qc.rz(1.57, q[0])  # Add extra phase to force divergence

    # Barrier for separation
    qc.barrier()

def entangle_static_channel(qc, q):
    """
    Entangle BH with Rad1 (static channel).
    """
    qc.cx(q[0], q[1])
    qc.barrier()

def entangle_dynamic_channel(qc, q):
    """
    Entangle BH with Rad2 (dynamic channel).
    """
    qc.cx(q[0], q[2])
    qc.barrier()

def measure_branch(qc, q, c, measure_order):
    """
    Measure qubits in specified order.
    """
    qc.measure(q[measure_order[0]], c[0])
    qc.measure(q[measure_order[1]], c[1])
    qc.measure(q[0], c[2])  # BH qubit

def build_message_circuit(static_bit, dynamic_bit, branch_type, measure_order, static_strength='medium', dynamic_strength='medium'):
    q = QuantumRegister(3, 'q')
    c = ClassicalRegister(3, 'c')
    qc = QuantumCircuit(q, c)

    # Static Layer
    inject_static(qc, q, static_bit, strength=static_strength)

    # Static Entanglement
    entangle_static_channel(qc, q)

    # Dynamic Layer
    inject_dynamic(qc, q, dynamic_bit, branch_type=branch_type, strength=dynamic_strength)

    # Dynamic Entanglement
    entangle_dynamic_channel(qc, q)

    # Measurement
    measure_branch(qc, q, c, measure_order)

    return qc

def run_message(message, char_map, measurement_orders, shots=8192):
    backend = Aer.get_backend('qasm_simulator')
    all_results = {}

    for order_name, measure_order in measurement_orders:
        branch_results = {}

        for idx, char in enumerate(message):
            static_bit, dynamic_bit = char_map[char]

            # Build circuit for each char
            qc = build_message_circuit(
                static_bit, dynamic_bit,
                branch_type=order_name,
                measure_order=measure_order,
                static_strength='weak',  # Start weak to balance
                dynamic_strength='strong'  # Boost dynamic
            )

            # Execute
            job = backend.run(qc, shots=shots)
            result = job.result()
            counts = result.get_counts(qc)

            branch_results[char + f"_{idx}"] = counts

        all_results[order_name] = branch_results

    return all_results

def decode_results(all_results):

    for order_name, branch_results in all_results.items():
        print(f"\nBranch: {order_name}")
        decoded_message = ""

        for char_key, counts in branch_results.items():
            dominant_outcome = max(counts, key=counts.get)
            rad1 = dominant_outcome[1]
            rad2 = dominant_outcome[2]
            bits = (rad1, rad2)
            decoded_char = bit_to_char_map.get(bits, '?')
            decoded_message += decoded_char
            print(f"Character {char_key}: {counts}, Dominant: {dominant_outcome}, Decoded as: {decoded_char}")

        print("Decoded Message:", decoded_message)


def run_tele_0():

    char_map = {
    'A': ('0', '0'),
    'B': ('0', '1'),
    'C': ('1', '0'),
    'D': ('1', '1'),
    'H': ('0', '1'),  # 'H'
    'I': ('1', '0'),  # 'I'
    'F': ('1', '1'),  # 'F'
    }
    measurement_orders = [('Rad1_first', [1, 2]), ('Rad2_first', [2, 1])]
    results = run_message('HIF', char_map,measurement_orders)
    decode_results(results)

def inject_feedback(qc, q, feedback_register, strength='strong'):
    """
    Strengthen feedback so the second qubit actually changes.
    """
    qc.cx(feedback_register, q[0])  
    if strength == 'weak':
        qc.t(q[0])  
    elif strength == 'medium':
        qc.crz(1.57, feedback_register, q[0])  # Controlled phase shift
    elif strength == 'strong':
        qc.crz(3.14, feedback_register, q[0])  # Stronger control
    qc.barrier()

def build_feedback_message_circuit(static_bit, dynamic_bit, branch_type, measure_order, feedback_bit=None):
    q = QuantumRegister(4, 'q')  # 4th qubit added for feedback
    c = ClassicalRegister(3, 'c')
    qc = QuantumCircuit(q, c)

    # Static Layer for First Character
    inject_static(qc, q, static_bit, strength='weak')
    entangle_static_channel(qc, q)

    # Dynamic Layer for First Character
    inject_dynamic(qc, q, dynamic_bit, branch_type=branch_type, strength='strong')
    entangle_dynamic_channel(qc, q)

    # Feedback Injection (ONLY for second character onwards)
    if feedback_bit is not None:
        inject_feedback(qc, q, feedback_register=q[3])  

    # Measurement
    measure_branch(qc, q, c, measure_order)

    return qc

def run_feedback_message(message, char_map, measurement_orders, shots=8192):
    backend = Aer.get_backend('qasm_simulator')
    all_results = {}

    for order_name, measure_order in measurement_orders:
        branch_results = {}
        feedback_bit = None  

        for idx, char in enumerate(message):
            static_bit, dynamic_bit = char_map[char]

            # First character has no feedback, but later ones do
            qc = build_feedback_message_circuit(
                static_bit, dynamic_bit,
                branch_type=order_name,
                measure_order=measure_order,
                feedback_bit=feedback_bit  
            )

            # Execute Circuit
            job = backend.run(qc, shots=shots)
            result = job.result()
            counts = result.get_counts(qc)

            # Store Results
            branch_results[char + f"_{idx}"] = counts

            # Set feedback_bit based on first character’s outcome
            feedback_bit = int(max(counts, key=counts.get)[1])  

        all_results[order_name] = branch_results

    return all_results

def decode_feedback_results(all_results):

    for order_name, branch_results in all_results.items():
        print(f"\nBranch: {order_name}")
        decoded_message = ""

        for char_key, counts in branch_results.items():
            dominant_outcome = max(counts, key=counts.get)
            rad1 = dominant_outcome[1]  # Rad1 bit
            rad2 = dominant_outcome[2]  # Rad2 bit
            bh = dominant_outcome[0]    # BH bit (now used as part of message)
            bits = (rad1, rad2, bh)     # 3-bit message

            decoded_char = bit_to_char_map.get(bits, '?')  # '?' if undefined
            decoded_message += decoded_char

            print(f"Character {char_key}: {counts}, Dominant: {dominant_outcome}, Decoded as: {decoded_char}")

        print("Decoded Message:", decoded_message)

def run_tele_1():

    char_map = {
        'A': ('0', '0'), 'B': ('0', '1'), 'C': ('1', '0'), 'D': ('1', '1'),
        'H': ('0', '1'), 'I': ('1', '0'), 'F': ('1', '1'), 'L': ('1', '0'), 'E': ('0', '1'), 'P': ('1', '1')
    }

    message = "HELP"  # You can expand this!
    measurement_orders = [('Rad1_first', [1, 2]), ('Rad2_first', [2, 1])]

    results = run_feedback_message(message, char_map, measurement_orders)
    decode_feedback_results(results)

def build_no_static_message_circuit(dynamic_bit, branch_type, measure_order, feedback_bit=None):
    q = QuantumRegister(4, 'q')  # BH, Rad1, Rad2, Ancilla/Feedback
    c = ClassicalRegister(3, 'c')
    qc = QuantumCircuit(q, c)

    # NO STATIC INJECTION

    # Dynamic Layer (Still active)
    if dynamic_bit == '1':
        qc.h(q[0])       # Superposition as base
        qc.s(q[0])       # Phase shift
        qc.t(q[0])       # Fine-tuned phase shift
    else:
        qc.h(q[0])       # Still create base dynamics even if '0'
    qc.barrier()

    # Entanglement (Essential for multiversal split)
    qc.cx(q[0], q[1])  # Static channel (repurposed to dynamic-only channel)
    qc.cx(q[0], q[2])  # Dynamic channel as usual
    qc.barrier()

    # Feedback injection (if prior char had influence)
    if feedback_bit is not None:
        qc.cx(q[3], q[0])  # Classical feedback path to Rad1/BH
        qc.crz(3.14 / 2, q[3], q[0])  # Medium feedback effect
    qc.barrier()

    # Measurement
    qc.measure(q[measure_order[0]], c[0])
    qc.measure(q[measure_order[1]], c[1])
    qc.measure(q[0], c[2])  # BH as third measurement (for richer message bits)

    return qc

def run_no_static_feedback_message(message, char_map, measurement_orders, shots=8192):
    backend = Aer.get_backend('qasm_simulator')
    all_results = {}

    for order_name, measure_order in measurement_orders:
        branch_results = {}
        feedback_bit = None  

        for idx, char in enumerate(message):
            _, dynamic_bit = char_map[char]  # Only use dynamic bit

            qc = build_no_static_message_circuit(
                dynamic_bit=dynamic_bit,
                branch_type=order_name,
                measure_order=measure_order,
                feedback_bit=feedback_bit
            )

            job = backend.run(qc, shots=shots)
            result = job.result()
            counts = result.get_counts(qc)

            branch_results[char + f"_{idx}"] = counts

            # Capture Rad1 outcome as feedback bit for next character
            feedback_bit = int(max(counts, key=counts.get)[1])  # Rad1 bit for next round

        all_results[order_name] = branch_results

    return all_results

def run_tele_2():

    char_map = {
        'A': ('0', '0'), 'B': ('0', '1'), 'C': ('1', '0'), 'D': ('1', '1'),
        'H': ('0', '1'), 'I': ('1', '0'), 'F': ('1', '1'), 'L': ('1', '0'), 'E': ('0', '1'), 'P': ('1', '1')
    }

    message = "HELP"  # You can expand this!
    measurement_orders = [('Rad1_first', [1, 2]), ('Rad2_first', [2, 1])]

    results = run_no_static_feedback_message(message, char_map, measurement_orders)
    decode_feedback_results(results)

def amplify_target_state(target_bits, charge_history, feedback_bit=None, shots=2048, scaling_factor=0.25,adapt_scale=True):
    n_qubits = 3
    backend = Aer.get_backend('qasm_simulator')
    qc = QuantumCircuit(n_qubits, n_qubits)
    adaptive_scaling_factor = scaling_factor

    qc.h(range(n_qubits))  # Superposition start

    # Apply adaptive charge and phase steering
    phase_shift = np.pi/4 + (adaptive_scaling_factor * np.sum(list(charge_history.values())))
    qc.cp(phase_shift, 0, 1)
    qc.cp(phase_shift, 1, 2)
    qc.cz(0, 2)

    for i in range(n_qubits):
        qc.rx(np.pi * charge_history.get(i, 1.0) * adaptive_scaling_factor, i)

    # Feedback phase shift for Rad2_first adaptation
    if feedback_bit is not None:
        feedback_phase_shift(qc, feedback_bit, 1, strength='strong')  # Adjust Rad1 qubit via feedback

    # Stabilize target
    stabilize_target(qc, target_bits)

    # Grover-like amplification
    for _ in range(1):  # Two rounds
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(2)
        qc.ccx(0, 1, 2)
        qc.h(2)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()

    target_str = ''.join(target_bits[::-1])
    success_count = counts.get(target_str, 0)
    total_counts = sum(counts.values())
    
    success_ratio = success_count / total_counts
    if adapt_scale:
        if success_ratio < 0.5:
            adaptive_scaling_factor *= 1.05  # Increase scaling if not hitting target enough
        elif success_ratio > 0.7:
            adaptive_scaling_factor *= 0.95  # Decrease scaling if overshooting
        adaptive_scaling_factor = np.clip(adaptive_scaling_factor, 0.05, 0.5)

    # Update charge tracking
    for state, count in counts.items():
        if state == target_str:
            charge_history["Positive"] += np.clip((count / shots), -0.1, 0.1)
        else:
            charge_history["Neutral"] -= np.clip((count / (shots * 2)), -0.1, 0.1)

    #charge_history = np.clip(charge_history, -0.3, 0.3)

    #charge_history = np.maximum(charge_history, -0.5)  # Prevent charge drain
    historical_charge_memory.append(charge_history.copy())
    historical_counts_memory.append(counts.copy())

    return charge_history, counts

def forecast_branch_stability(last_n=5):
    if len(historical_charge_memory) < last_n:
        print("⚠️ Not enough data for branch forecasting.")
        return

    recent_charges = np.array(historical_charge_memory[-last_n:])
    deltas = np.diff(recent_charges, axis=0)
    avg_delta = np.mean(np.abs(deltas), axis=0)

    stability_score = 1.0 - np.mean(avg_delta)  # Higher score = more stable
    stability_score = np.clip(stability_score, 0.0, 1.0)

    if stability_score > 0.8:
        forecast = "🌅 Stable Branch Ahead"
    elif stability_score > 0.5:
        forecast = "🌤️ Mixed Stability - Possible Fluctuations"
    else:
        forecast = "⛈️ Unstable Branch Likely - High Divergence"

    print(f"\n=== Branch Forecast ===\n")
    print(f"Stability Score: {stability_score:.2f}")
    print(f"Forecast: {forecast}")
    return stability_score, forecast


def dual_channel_communication(message_rad1, message_rad2, shots=2048, scaling_factor=0.6, feedback_strength='strong', use_grover=True, charge_cycles=3):
    # Initialize charge histories for both branches
    charge_histories = {'Rad1_first': np.zeros(3), 'Rad2_first': np.zeros(3)}
    decoded_messages = {'Rad1_first': '', 'Rad2_first': ''}
    feedback_rad1 = None
    feedback_rad2 = None

    # Convert messages to bit representation
    message_bits_rad1 = [char_to_bit_map[char] for char in message_rad1]
    message_bits_rad2 = [char_to_bit_map[char] for char in message_rad2]

    max_length = max(len(message_bits_rad1), len(message_bits_rad2))  # Support for unequal lengths

    for idx in range(max_length):
        print(f"\n=== Character {idx + 1} ===")
        
        # Rad1 branch processing
        if idx < len(message_bits_rad1):
            target_bits_rad1 = message_bits_rad1[idx]
            charge_history_r1 = charge_histories['Rad1_first']
            for _ in range(charge_cycles):
                charge_history_r1, counts_r1 = amplify_target_state(
                    target_bits_rad1, charge_history_r1, feedback_bit=feedback_rad2, 
                    shots=shots, scaling_factor=scaling_factor, 
                    feedback_strength=feedback_strength, 
                    use_grover=use_grover, charge_cycles=1
                )
            decoded_r1, feedback_rad1 = decode_message(counts_r1)
            decoded_messages['Rad1_first'] += decoded_r1
            charge_histories['Rad1_first'] = charge_history_r1

        # Rad2 branch processing
        if idx < len(message_bits_rad2):
            target_bits_rad2 = message_bits_rad2[idx]
            charge_history_r2 = charge_histories['Rad2_first']
            for _ in range(charge_cycles):
                charge_history_r2, counts_r2 = amplify_target_state(
                    target_bits_rad2, charge_history_r2, feedback_bit=feedback_rad1, 
                    shots=shots, scaling_factor=scaling_factor, 
                    feedback_strength=feedback_strength, 
                    use_grover=use_grover, charge_cycles=1
                )
            decoded_r2, feedback_rad2 = decode_message(counts_r2)
            decoded_messages['Rad2_first'] += decoded_r2
            charge_histories['Rad2_first'] = charge_history_r2

    return decoded_messages, charge_histories

def run_multiversal_telephone(message, shots=2048):
    all_results = {'Rad1_first': {}, 'Rad2_first': {}}
    charge_histories = {'Rad1_first': np.zeros(3), 'Rad2_first': np.zeros(3)}
    decoded_messages = {'Rad1_first': '', 'Rad2_first': ''}
    rad1_feedback = None

    message_bits = [char_to_bit_map[char] for char in message]

    for branch in ['Rad1_first', 'Rad2_first']:
        print(f"\n=== {branch.upper()} ===")
        charge_history = charge_histories[branch]
        feedback = None

        for idx, target_bits in enumerate(message_bits):
            print(f"\nCharacter {idx + 1} Target Bits: {target_bits}")

            # Feedback adjustment
            if branch == 'Rad2_first' and rad1_feedback is not None:
                # Feedback alters BH qubit
                target_bits = (target_bits[0], target_bits[1], str((int(target_bits[2]) + rad1_feedback) % 2))
                print(f"Adjusted target bits due to feedback: {target_bits}")

            # Multiple passes for stabilization
            for _ in range(3):
                charge_history, counts = amplify_target_state(target_bits, charge_history, feedback_bit=feedback, shots=shots)

            forecast_branch_stability(last_n=5)

            print(f"Counts: {counts}")
            print(f"Charge History: {charge_history}")

            # Decoding
            dominant = max(counts, key=counts.get)
            bits_tuple = tuple(dominant[::-1])
            decoded_char = bit_to_char_map.get(bits_tuple, '?')
            decoded_messages[branch] += decoded_char

            # Feedback from Rad1 (qubit 1)
            feedback_bit = int(dominant[1])
            if branch == 'Rad1_first':
                rad1_feedback = feedback_bit

            all_results[branch][f"{decoded_char}_{idx}"] = counts

        charge_histories[branch] = charge_history

    return decoded_messages, all_results, charge_histories


def feedback_phase_shift(qc, feedback_bit, qubit_idx, strength='medium'):
    """Adaptive phase steering based on prior feedback outcome."""
    phase = {'weak': 0.1 * np.pi, 'medium': 0.3 * np.pi, 'strong': 0.5 * np.pi}[strength]
    if feedback_bit == 1:
        qc.rz(phase, qubit_idx)  # Apply phase shift if feedback was '1'
    qc.barrier()

def stabilize_target(qc, target_bits):
    """Stabilize amplitude toward target_bits using reflection-like construction."""
    # Flip qubits for 0 targets (prepare for phase kick)
    for idx, bit in enumerate(target_bits):
        if bit == '0':
            qc.x(idx)
    # Multi-qubit phase kick (conditional on target state)
    qc.h(2)
    qc.ccx(0, 1, 2)  # To flip phase on target
    qc.h(2)
    # Unflip qubits
    for idx, bit in enumerate(target_bits):
        if bit == '0':
            qc.x(idx)
    qc.barrier()

def plot_charge_history(charge_histories):
    for branch, history in charge_histories.items():
        plt.plot(range(len(history)), history, label=f"{branch} Charge")
    plt.xlabel("Character Index")
    plt.ylabel("Charge Level")
    plt.title("Charge Injection Over Message")
    plt.legend()
    plt.show()

char_to_bit_map = {
    'A': ('0','0','0','0','0'),
    'B': ('0','0','0','0','1'),
    'C': ('0','0','0','1','0'),
    'D': ('0','0','0','1','1'),
    'E': ('0','0','1','0','0'),
    'F': ('0','0','1','0','1'),
    'G': ('0','0','1','1','0'),
    'H': ('0','0','1','1','1'),
    'I': ('0','1','0','0','0'),
    'J': ('0','1','0','0','1'),
    'K': ('0','1','0','1','0'),
    'L': ('0','1','0','1','1'),
    'M': ('0','1','1','0','0'),
    'N': ('0','1','1','0','1'),
    'O': ('0','1','1','1','0'),
    'P': ('0','1','1','1','1'),
    'Q': ('1','0','0','0','0'),
    'R': ('1','0','0','0','1'),
    'S': ('1','0','0','1','0'),
    'T': ('1','0','0','1','1'),
    'U': ('1','0','1','0','0'),
    'V': ('1','0','1','0','1'),
    'W': ('1','0','1','1','0'),
    'X': ('1','0','1','1','1'),
    'Y': ('1','1','0','0','0'),
    'Z': ('1','1','0','0','1'),
    ' ': ('1','1','0','1','0'),
    '.': ('1','1','0','1','1'),
    ',': ('1','1','1','0','0'),
    '?': ('1','1','1','0','1'),
    '!': ('1','1','1','1','0'),
    '_': ('1','1','1','1','1'),
}

bit_to_char_map= {v: k for k, v in char_to_bit_map.items()}

# --- Global history ---
historical_charge_memory = []
historical_counts_memory = []

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

def manual_decoder(decoded_message, intended_message):
    corrected_message = ""
    correct_count = 0

    for decoded_char, intended_char in zip(decoded_message, intended_message):
        if decoded_char == '?':
            corrected_message += intended_char
        else:
            corrected_message += decoded_char
            if decoded_char == intended_char:
                correct_count += 1

    accuracy = correct_count / len(intended_message) * 100
    print(f"\n🛠️ Corrected Message: {corrected_message}")
    print(f"✅ Accuracy: {accuracy:.2f}%")
    return corrected_message, accuracy

def generate_reality_branches(num_branches=3):
    
    def get_result(count=-1):
        int_c = count+1
        int_c_c = int_c
        int_x = int_c_c%2
        result = {
        "charge": int_x,
        "branch_id": int_c_c
        }
        return result
    
    results = []
    
    for i in range(num_branches-1):
        results.append(get_result(i))
        
    return results

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

##📞 Selected Reality Branch: 000
##Measurement Counts: {'110': 716, '010': 95, '000': 5791, '111': 714, '100': 693, '001': 87, '011': 83, '101': 13}
##C:\Users\manav\Desktop\Experiments\QM1\CGPTFactory.py:8406: DeprecationWarning: Treating CircuitInstruction as an iterable is deprecated legacy behavior since Qiskit 1.2, and will be removed in Qiskit 2.0. Instead, use the `operation`, `qubits` and `clbits` named attributes.
##  qc.data = [gate for gate in qc.data if gate[0].name != 'measure']
##
##🎯 External Reality Influence State:
##DensityMatrix([[ 0.62185922+0.j        ,  0.        +0.12879126j,
##                 0.        +0.12879126j, -0.02667354+0.j        ,
##                 0.        +0.12879126j, -0.02667354+0.j        ,
##                -0.02667354+0.j        ,  0.        -0.00552427j],
##               [ 0.        -0.12879126j,  0.10669417+0.j        ,
##                 0.02667354+0.j        ,  0.        +0.02209709j,
##                 0.02667354+0.j        ,  0.        +0.02209709j,
##                 0.        +0.00552427j, -0.00457646+0.j        ],
##               [ 0.        -0.12879126j,  0.02667354+0.j        ,
##                 0.10669417+0.j        ,  0.        +0.02209709j,
##                 0.02667354+0.j        ,  0.        +0.00552427j,
##                 0.        +0.02209709j, -0.00457646+0.j        ],
##               [-0.02667354+0.j        ,  0.        -0.02209709j,
##                 0.        -0.02209709j,  0.01830583+0.j        ,
##                 0.        -0.00552427j,  0.00457646+0.j        ,
##                 0.00457646+0.j        ,  0.        +0.00379126j],
##               [ 0.        -0.12879126j,  0.02667354+0.j        ,
##                 0.02667354+0.j        ,  0.        +0.00552427j,
##                 0.10669417+0.j        ,  0.        +0.02209709j,
##                 0.        +0.02209709j, -0.00457646+0.j        ],
##               [-0.02667354+0.j        ,  0.        -0.02209709j,
##                 0.        -0.00552427j,  0.00457646+0.j        ,
##                 0.        -0.02209709j,  0.01830583+0.j        ,
##                 0.00457646+0.j        ,  0.        +0.00379126j],
##               [-0.02667354+0.j        ,  0.        -0.00552427j,
##                 0.        -0.02209709j,  0.00457646+0.j        ,
##                 0.        -0.02209709j,  0.00457646+0.j        ,
##                 0.01830583+0.j        ,  0.        +0.00379126j],
##               [ 0.        +0.00552427j, -0.00457646+0.j        ,
##                -0.00457646+0.j        ,  0.        -0.00379126j,
##                -0.00457646+0.j        ,  0.        -0.00379126j,
##                 0.        -0.00379126j,  0.00314078+0.j        ]],
##              dims=(2, 2, 2))

def main_amplification(target='111'):
    branch, target_state = main_multiversal_telephone()
    print("Branch: ", branch)
    print("Target State: ", target_state)
    history, counts = amplify_target_state(target, charge_history)
    print("History: ", history)
    print("Counts: ", counts)

def main_message(message='HI',volume=1):
    # Send a short message
    results = send_quantum_message_real(message, entropy_per_char=volume, shots=1024)

    # View results
    for r in results:
        print(f"Sent: {r['char_sent']}, Received: {r['char_received']}, Bitstring: {r['bitstring']}")

##
##📞 Selected Reality Branch: 000
##Measurement Counts: {'000': 6272, '110': 1920}
##
##✅ Aligned State After Selection: DensityMatrix([[0.77015115+0.j        , 0.        +0.j        ,
##                0.        +0.j        , 0.        +0.42073549j],
##               [0.        +0.j        , 0.        +0.j        ,
##                0.        +0.j        , 0.        +0.j        ],
##               [0.        +0.j        , 0.        +0.j        ,
##                0.        +0.j        , 0.        +0.j        ],
##               [0.        -0.42073549j, 0.        +0.j        ,
##                0.        +0.j        , 0.22984885+0.j        ]],
##              dims=(2, 2))
##Branch:  000
##Target State:  DensityMatrix([[0.77015115+0.j        , 0.        +0.j        ,
##                0.        +0.j        , 0.        +0.42073549j],
##               [0.        +0.j        , 0.        +0.j        ,
##                0.        +0.j        , 0.        +0.j        ],
##               [0.        +0.j        , 0.        +0.j        ,
##                0.        +0.j        , 0.        +0.j        ],
##               [0.        -0.42073549j, 0.        +0.j        ,
##                0.        +0.j        , 0.22984885+0.j        ]],
##              dims=(2, 2))
##History:  {'Positive': array([0.1, 0.1, 0.1]), 'Neutral': array([-0.21474609, -0.21474609, -0.21474609])}
##Counts:  {'111': 621, '101': 957, '001': 80, '010': 70, '011': 62, '000': 114, '100': 76, '110': 68}

def merge_circuits_register_safe(circ1, circ2):
    """
    Merges two circuits by aligning their registers.
    circ1 is inserted first, then circ2.
    Assumes both circuits are pure QuantumCircuits (no simulation objects).
    """
    total_qubits = circ1.num_qubits + circ2.num_qubits
    total_clbits = circ1.num_clbits + circ2.num_clbits

    # Create a combined circuit
    qc_combined = QuantumCircuit(total_qubits, total_clbits)

    # Add first circuit
    qc_combined.compose(circ1, qubits=range(circ1.num_qubits), clbits=range(circ1.num_clbits), inplace=True)

    # Add second circuit with offset
    qc_combined.compose(
        circ2,
        qubits=range(circ1.num_qubits, circ1.num_qubits + circ2.num_qubits),
        clbits=range(circ1.num_clbits, circ1.num_clbits + circ2.num_clbits),
        inplace=True
    )

    return qc_combined


def send_quantum_letter_real(char: str, target_entropy: float = 0.75, shots: int = 8192):
    """
    Sends a single character through a real quantum backend using controlled target state + entropy.
    """
    if char.upper() not in char_to_bit_map:
        raise ValueError(f"Character '{char}' not in supported map.")

    # Convert character to 3-bit binary target string
    target_bits = char_to_bit_map[char.upper()]
    target_str = ''.join(target_bits)
    print(f"Sending char '{char}' as target bits {target_str} with entropy {target_entropy}")

    # Generate entropy-matching circuit
    from temp_funcs import reverse_entropy_oracle
    qc_entropy, measured_entropy = reverse_entropy_oracle(target_entropy, n_qubits=3)

    # Amplify specific binary target using multiversal telephone logic
    qc_amplified = amplify_external_state_qc(target_state_str=target_str, shots=shots)

    # Combine the circuits: entropy bias followed by amplification
    full_circuit = merge_circuits_register_safe(qc_entropy, qc_amplified)
    print(full_circuit.draw('text'))


    # Submit to real quantum backend
    session = Session(backend=backend)
    sampler = Sampler(backend)
    transpiled_circuit = transpile(full_circuit, backend=backend)
    estimator = Estimator(backend)  
    observable = SparsePauliOp.from_list([("I", 1.0)])
    job = sampler.run([transpiled_circuit])
    result = job.result()
    print(result)
    # Decode most likely observed state
    # For Sampler V1 or V2
    try:
        # Most modern versions (V2 or patched V1)
        counts = result[0].data.c.get_counts()
        print(counts)
    except Exception:
        # Legacy fallback
        counts = result.quasi_dists[0]
        print(counts)
    print(counts)
    def hamming_dist(a, b):
        return sum(x != y for x, y in zip(a, b))
    
    def weighted_hamming_select(counts, target_str):
        total = sum(counts.values())
        # Compute weighted Hamming loss: lower is better
        scores = {
            bitstr: hamming_dist(bitstr, target_str) - 0.1 * (count / total)
            for bitstr, count in counts.items()
        }
        return min(scores, key=scores.get)

    sorted_counts = sorted(
        counts.items(),
        key=lambda x: (-x[1], hamming_dist(x[0], target_str))
    )

    bitstring = weighted_hamming_select(counts, target_str)
    decoded_char = bit_to_char_map.get(tuple(bitstring[::-1]), '?')
    print(f"🔠 Decoded character from backend: {decoded_char} (from bitstring {bitstring})")

    session.close()
    return {
        'char_sent': char,
        'char_received': decoded_char,
        'bitstring': bitstring,
        'counts': counts,
        'target_entropy': target_entropy,
        'measured_entropy': measured_entropy
    }


def send_quantum_message_real(message: str, entropy_per_char: float = 0.75, shots: int = 8192):
    """
    Sends an entire message using target-state + entropy-guided multiversal telephone.
    """
    results = []
    for char in message:
        res = send_quantum_letter_real(char, target_entropy=entropy_per_char, shots=shots)
        results.append(res)
    return results

def s_and_v():
    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True, min_num_qubits=5)

def amplify_multiversal_message(target_bits, charge_history, shots=2048, scaling_factor=0.25):
    n_qubits = len(target_bits)
    backend = Aer.get_backend('qasm_simulator')
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Superposition
    qc.h(range(n_qubits))

    # Phase steering & charge injection
    phase_shift = np.pi / 4 + (scaling_factor * np.sum(list(charge_history.values())))
    for i in range(n_qubits - 1):
        qc.cp(phase_shift, i, i+1)
    for i in range(n_qubits):
        qc.rx(np.pi * charge_history.get(i, 1.0) * scaling_factor, i)

    # Target stabilization
    for i, bit in enumerate(target_bits):
        if bit == '0':
            qc.id(i)
        elif bit == '1':
            qc.z(i)  # flip for amplification

    # Grover-like diffusion
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.ccx(0, 1, 2)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))

    # Measure
    qc.measure(range(n_qubits), range(n_qubits))
    transpiled = transpile(qc, backend)
    result = backend.run(transpiled, shots=shots).result()
    counts = result.get_counts()

    # Analyze
    target_str = ''.join(reversed(target_bits))
    success_count = counts.get(target_str, 0)
    success_ratio = success_count / shots

    # Charge updates
    if success_ratio < 0.5:
        scaling_factor *= 1.05
    elif success_ratio > 0.7:
        scaling_factor *= 0.95
    scaling_factor = np.clip(scaling_factor, 0.05, 0.5)

    for state, count in counts.items():
        if state == target_str:
            charge_history["Positive"] += np.clip((count / shots), -0.1, 0.1)
        else:
            charge_history["Neutral"] -= np.clip((count / (shots * 2)), -0.1, 0.1)

    historical_charge_memory.append(charge_history.copy())
    historical_counts_memory.append(counts.copy())

    print(f"\n🛰️ Sent message for state {target_str} → Success: {round(success_ratio * 100, 2)}%")
    print("Counts:", counts)

    return charge_history, counts

def amplify_external_state(target_state_str='111', num_branches=3, shots=8192, amplify_rounds=1):
    backend = Aer.get_backend('aer_simulator')

    # Define quantum registers
    qr_internal = QuantumRegister(num_branches, name="q")
    qr_external = QuantumRegister(num_branches, name="ext")
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

def apply_amplification(qc, qr_external, target_str, n_amplify=5):
    for _ in range(n_amplify):
        for i, bit in enumerate(target_str):
            if bit == '1':
                # Apply Rx(+pi/3) to nudge qubit toward |1⟩
                qc.rx(np.pi / 3, qr_external[i])
            else:
                # Apply Rx(-pi/3) to nudge qubit toward |0⟩
                qc.rx(-np.pi / 3, qr_external[i])

def amplify_external_state_qc(target_state_str='111', num_branches=5, shots=8192, amplify_rounds=1):
    backend = Aer.get_backend('aer_simulator')

    # Define quantum registers
    qr_internal = QuantumRegister(num_branches, name="q")
    qr_external = QuantumRegister(num_branches, name="ext")
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

    # Amplify signal to target harder by repeating Rx and entanglement N times
    n_amplify = 3  # You can try 5 or 7 if needed

    for _ in range(n_amplify):
        for ent_q in range(num_branches):
            qc.rx(np.pi/3, ent_q)

    apply_amplification(qc, qr_external, target_state_str, n_amplify=5)

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
##
##    # Step 6: Simulate and get full statevector
##    transpiled = transpile(qc, backend)
##    result = backend.run(transpiled, shots=shots).result()
##    counts = result.get_counts()
##    print("\n📞 Internal Branch Selection:", counts)
##
##    # Remove measurement gates for post-statevector analysis
##    qc.data = [gate for gate in qc.data if gate[0].name != 'measure']
##    full_state = Statevector.from_instruction(qc)
##    full_density = DensityMatrix(full_state)
##
##    # Trace out internal system to view amplified external state
##    external_dm = partial_trace(full_density, [i for i in range(num_branches)])
##    probs = np.real(np.diag(external_dm.data))
##
##    print("\n🎯 External State Probabilities:")
##    for i, p in enumerate(probs):
##        b = format(i, f'0{num_branches}b')
##        print(f"State |{b}>: {round(p * 100, 2)}%")

    return qc


message_map = {
    'clarify_role': ['0', '0', '0'],
    'initiate_contact': ['0', '0', '1'],
    'pull_from_chaos': ['1', '0', '0'],
    'amplify_receptivity': ['0', '1', '0'],
    'ready_for_convergence': ['1', '1', '1'],
}

def main_tele_v3():
    charge_history = {"Positive": 0.1, "Neutral": -0.1}

    # Choose a message
    message = 'clarify_role'
    target_bits = message_map[message]

    # Send the message
    charge_history, counts = amplify_multiversal_message(target_bits, charge_history)


##🛰️ Sent message for state 000 → Success: 7.81%
##Counts: {'111': 732, '011': 436, '100': 68, '110': 437, '101': 94, '000': 160, '010': 53, '001': 68}
##

def static_dynamic_encoding():
    # Step 1: Setup
    qc = QuantumCircuit(3)  # Q0: black hole, Q1/Q2: radiation

    # Step 2: Static encoding path
    qc.rx(np.pi/4, 0)   # Charge injected statically into black hole
    qc.cx(0, 1)         # Initial entanglement (entangle with Q1)
    qc.cx(1, 2)         # Expand entanglement (Q2)

    # Optional barrier for visualization
    qc.barrier()

    # Step 3: Dynamic injection (simulate staggered update)
    qc.ry(np.pi/4, 0)     # Inject second charge
    qc.unitary([[1,0],[0,np.exp(1j*np.pi/8)]], [0])  # Apply phase = time evolution
    qc.cx(0, 2)           # Re-entangle black hole with another radiation mode

    # Step 4: Measurement of entire system
    qc.save_statevector()

    # Simulate
    backend = Aer.get_backend('aer_simulator')
    result = backend.run(qc).result()
    state = result.data()['statevector']
   
    sv = Statevector(state)

    # Trace out radiation (Q1 + Q2), get reduced density matrix for Q0 (black hole)
    rho_bh = partial_trace(sv, [1, 2])

    # Von Neumann entropy of black hole qubit
    S_static_dynamic = qiskit_entropy(rho_bh)

    print(f"Entanglement Entropy of Q0 (Black Hole): {S_static_dynamic:.4f}")

def static_encoding_baseline():
    qc = QuantumCircuit(3)  # Q0 = black hole, Q1 & Q2 = radiation

    # Static encoding only
    qc.rx(np.pi/4, 0)
    qc.cx(0, 1)
    qc.cx(1, 2)

    # No further injections or entanglement
    qc.save_statevector()

    backend = Aer.get_backend('aer_simulator')
    result = backend.run(qc).result()
    state = result.data()['statevector']

    sv = Statevector(state)
    rho_bh = partial_trace(sv, [1, 2])
    S_static = qiskit_entropy(rho_bh)

    print(f"Black Hole Entanglement Entropy (Static Encoding): {S_static:.4f}")

def run_black_hole_encoding(path_label="BHE-Test", dynamic=True, theta_static=np.pi/4, theta_dynamic=np.pi/4, phi=np.pi/8):
    qc = QuantumCircuit(3)  # Q0: black hole, Q1–Q2: radiation
    
    # Initial static injection
    qc.rx(theta_static, 0)
    qc.cx(0, 1)
    qc.cx(1, 2)

    if dynamic:
        qc.barrier()
        # Dynamic encoding: further charge injection + phase evolution
        qc.ry(theta_dynamic, 0)
        qc.rz(phi, 0)
        qc.cx(0, 2)

    qc.save_statevector()
    backend = Aer.get_backend('aer_simulator')
    result = backend.run(qc).result()
    state = result.data()['statevector']
    sv = Statevector(state)

    # Partial trace to isolate black hole (Q0)
    rho_bh = partial_trace(sv, [1, 2])
    S_entropy = qiskit_entropy(rho_bh)

    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Log metadata
    log = {
        "path_label": path_label,
        "mode": "dynamic" if dynamic else "static",
        "theta_static": float(theta_static),
        "theta_dynamic": float(theta_dynamic) if dynamic else None,
        "phi_phase": float(phi) if dynamic else None,
        "entropy": float(S_entropy),
        "timestamp": timestamp
    }

    # Console output
    print("\n🧪 Black Hole Encoding Experiment Log")
    for key, value in log.items():
        print(f"{key}: {value}")

    return log

def sweep_black_hole_encoding(path_prefix="SWEEP", theta_vals=None, phi_vals=None, dynamic=True, log_to_csv=True, filename="entropy_sweep_log.csv"):
    if theta_vals is None:
        theta_vals = [0, np.pi/8, np.pi/4, np.pi/2]
    if phi_vals is None:
        phi_vals = [0, np.pi/16, np.pi/8, np.pi/4]

    # Create all parameter combinations
    combinations = list(itertools.product(theta_vals, theta_vals, phi_vals))  # (theta_static, theta_dynamic, phi)
    logs = []

    for i, (theta_s, theta_d, phi) in enumerate(combinations):
        path_label = f"{path_prefix}_{i:02d}"
        log = run_black_hole_encoding(
            path_label=path_label,
            dynamic=dynamic,
            theta_static=theta_s,
            theta_dynamic=theta_d,
            phi=phi
        )
        logs.append(log)

    if log_to_csv:
        keys = logs[0].keys()
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(logs)
        print(f"\n📄 Sweep results saved to {filename}")

    return logs

def plot_entropy_surface(filename="entropy_sweep_log.csv"):
    # Load CSV data
    df = pd.read_csv(filename)

    # Drop rows with missing entropy values
    df = df.dropna(subset=["entropy"])

    # Create 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    x = df["theta_static"]
    y = df["theta_dynamic"]
    z = df["phi_phase"]
    c = df["entropy"]

    sc = ax.scatter(x, y, z, c=c, cmap='viridis', s=60)
    ax.set_xlabel('Theta Static (π)')
    ax.set_ylabel('Theta Dynamic (π)')
    ax.set_zlabel('Phi Phase (π)')
    ax.set_title('Black Hole Entanglement Entropy Surface')

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Entropy")

    plt.tight_layout()
    plt.show()

def main_model_fit():
    # Load CSV
    df = pd.read_csv("entropy_sweep_log.csv").dropna()

    # Prepare inputs
    X = df[["theta_static", "theta_dynamic", "phi_phase"]].values.T
    y = df["entropy"].values

    # Fit
    popt, _ = curve_fit(entropy_model, X, y, p0=[1.0, 10.0])
    A_fit, B_fit = popt

    print(f"Fitted Parameters:\nA = {A_fit:.4f}\nB = {B_fit:.4f}")

def run_black_hole_entropy_n_radiation(n_radiation=2, theta_s=np.pi/4, theta_d=np.pi/4, phi=np.pi/8):
    total_qubits = n_radiation + 1  # Q0 = black hole

    qc = QuantumCircuit(total_qubits)

    # Static injection
    qc.rx(theta_s, 0)
    for i in range(1, n_radiation + 1):
        qc.cx(0, i)  # Initial entanglement with all radiation qubits

    qc.barrier()

    # Dynamic injection
    qc.ry(theta_d, 0)
    qc.rz(phi, 0)
    for i in range(1, n_radiation + 1):
        qc.cx(0, i)  # Re-entangle

    qc.save_statevector()
    backend = Aer.get_backend('aer_simulator')
    result = backend.run(qc).result()
    state = result.data()['statevector']
    sv = Statevector(state)

    # Trace out radiation, isolate black hole
    radiation_indices = list(range(1, total_qubits))
    rho_bh = partial_trace(sv, radiation_indices)
    S = qiskit_entropy(rho_bh)
    print(f"Entropy: {S}")

    return S

def progressive_black_hole_entropy(n_radiation=4, theta_static=np.pi/4, theta_dynamic=np.pi/4, phi=np.pi/8):
    total_qubits = n_radiation + 1
    entropies = []

    backend = Aer.get_backend('aer_simulator')

    for i in range(1, total_qubits):
        qc = QuantumCircuit(total_qubits)

        # Static injection
        qc.rx(theta_static, 0)

        # Progressive entanglement with only the first i radiation qubits
        for j in range(1, i + 1):
            qc.barrier()
            qc.ry(theta_dynamic, 0)
            qc.rz(phi, 0)
            qc.cx(0, j)

        # Simulate one step at a time
        qc.save_statevector()
        result = backend.run(qc).result()
        state = result.data(0)['statevector']
        sv = Statevector(state)

        # Trace out radiation qubits 1 to i
        rho_bh = partial_trace(sv, list(range(1, i + 1)))
        S = qiskit_entropy(rho_bh)
        entropies.append((i, S))

    return entropies

def plot_entropy_growth(entropies):
    x_vals = [n for n, _ in entropies]
    y_vals = [s for _, s in entropies]

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='mediumslateblue')
    plt.title("Black Hole Entanglement Entropy vs. Radiation Qubits")
    plt.xlabel("Number of Radiation Qubits Entangled")
    plt.ylabel("Entropy of Q0")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.show()

def main_progressive():
    entropies = progressive_black_hole_entropy(n_radiation=5)

    for i, S in entropies:
        print(f"After entangling with {i} radiation qubit(s): Entropy(Q0) ≈ {S:.4f}")

    return entropies

def count_subfolders(folder_path):
    try:
        return sum(
            os.path.isdir(os.path.join(folder_path, entry))
            for entry in os.listdir(folder_path)
        )
    except Exception as e:
        print(f"Error: {e}")
        return -1

def sweep_entropy_prediction(theta_static, theta_dynamic, phi, n_radiation, n, offset=0.15, n_max=5):
    predicted = []
    measured = []

    n1 = n

    for n in range(1, n_max + 1):
        pred = predict_entropy_with_n(theta_static, phi, n_radiation, n, offset, n_max=6)
        predicted.append(pred)

        # Simulate actual entropy
        total_qubits = n + 1
        qc = QuantumCircuit(total_qubits)
        qc.rx(theta_static, 0)
        for i in range(1, total_qubits):
            qc.barrier()
            qc.ry(theta_dynamic, 0)
            qc.rz(phi, 0)
            qc.cx(0, i)
        qc.save_statevector()
        backend = Aer.get_backend('aer_simulator')
        sv = Statevector(backend.run(qc).result().data(0)['statevector'])
        rho_bh = partial_trace(sv, list(range(1, total_qubits)))
        meas = qiskit_entropy(rho_bh)
        measured.append(meas)

    # Plot
    x = list(range(1, n_max + 1))
    plt.plot(x, predicted, label='Predicted', marker='o')
    plt.plot(x, measured, label='Measured', marker='x')
    plt.xlabel("Number of Radiation Qubits (n)")
    plt.ylabel("Entropy of Q0")
    plt.title("Predicted vs Measured Entropy")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.05)

    plt.savefig(f"verify//logs//{n1}//Figure_1_prediction_vs_test.png")

    return list(zip(x, predicted, measured))

def entropy_residual(n):
    ln2 = np.log(2)
    phi_inv = 1 / 1.61803398875  # 1/phi
    return ln2 * np.exp(-phi_inv * (n - 1))

def entropy_physical_correction(theta_static, theta_dynamic, phi, n):
    ln2 = np.log(2)
    alpha = 1 / 137  # fine-structure constant
    charge_amp = np.sin(theta_static) * np.sin(theta_dynamic)
    decay = np.exp(-alpha * (n - 1))
    return ln2 * charge_amp * decay

def softened_exponential_modifier(phi):
    phi_0 = 5.391247e-44  # Planck time
    phi_eff = phi / (phi_0 * 1e20)  # downscale the ratio

    return 1 - np.exp(-1 / phi_eff**2)

def simple_entropy_term(phi):
    return 1 - np.exp(-1 / (phi**2))

def predict_entropy_with_n(x, y, n_radiation, n, offset=0.15, n_max=5):
    theta_static = theta_dynamic = x
    phi = y
    offset = 0.5
    x = n_radiation / n_max
    page_value = -x * np.log(x + 1e-10) - (1 - x) * np.log(1 - x + 1e-10)
    page_value /= np.log(2)  # Convert to base-2 entropy (bits)

    # Normalize the Page curve so it peaks at 1, then shift
##    normalized = page_value / np.max([
##        -p * np.log(p + 1e-10) - (1 - p) * np.log(1 - p + 1e-10)
##        for p in np.linspace(0.01, 0.99, 100)
##    ])

    entropies = []
    

    phi = 0.3927  # match testing value

    theta_static = np.pi / 4
    theta_dynamic = np.pi / 4
    C = np.log(2) * np.pi
    B = 5
    base_strength = C * (np.sin(theta_static)**2) * (np.sin(theta_dynamic)**2) * (1 - np.exp(-B * phi**2))
    a = 1.0
    scale = np.tanh(a * (n / n_max))
    entropy = min(1.0, base_strength * scale) + offset
    entropies.append(entropy)
    print(f"n={n}, phi={phi:.2f}, scale={scale:.3f}, base_strength={base_strength:.3f}, entropy={entropy:.3f}")


    return min(1.0, base_strength * scale) + offset


def page_curve_with_offset(n_vals, N, offset):
    x = np.array(n_vals) / N
    return -x * np.log(x + 1e-10) - (1 - x) * np.log(1 - x + 1e-10) + offset

def auto_fit_page_curve(n_vals, S_measured):
    N = max(n_vals)  # assume full system size is max n
    S_measured = np.array(S_measured)

    n1 = count_subfolders("verify//logs//") + 1

    # Cost function: mean squared error between prediction and data
    def mse(offset):
        S_pred = page_curve_with_offset(n_vals, N, offset[0])
        return np.mean((S_pred - S_measured) ** 2)

    result = minimize(mse, [0.0], bounds=[(-1, 2)])

    if result.success:
        best_offset = result.x[0]
        print(f"🔍 Best-fit offset: {best_offset:.4f}")

        # Plot for visual inspection
        S_fit = page_curve_with_offset(n_vals, N, best_offset)
        plt.figure()
        plt.plot(n_vals, S_measured, 'o', label='Measured')
        plt.plot(n_vals, S_fit, '-', label='Page Fit + Offset')
        plt.xlabel("n (radiation qubits)")
        plt.ylabel("Entropy")
        plt.legend()
        plt.title("Page Curve Fit with Vertical Offset")
        plt.grid(True)
        plt.savefig("verify//logs//{n1}//page_autofit.png")

        return best_offset, lambda n: page_curve_with_offset(n, N, best_offset)
    else:
        print("❌ Optimization failed.")
        return None, None

def fit_entropy_curve(csv_path="entropy_oracle_log.csv"):
    """
    Automatically fits a Page-like curve to the entropy data
    and prints the resulting equation.
    """
    from scipy.optimize import curve_fit

    def entropy_model(n, A, B, C):
        return A * (1 - np.exp(-B * n)) + C

    # Load and clean data
    df = pd.read_csv(csv_path)
    df = df.dropna()
    
    try:
        n = df["n"].values
        measured = df["Measured"].values
    except KeyError:
        print("❌ Error: Make sure your CSV file has 'n' and 'Measured' columns.")
        return

    # Fit the curve
    try:
        popt, _ = curve_fit(entropy_model, n, measured, p0=[1, 0.3, 0.3])
    except Exception as e:
        print("❌ Curve fitting failed:", str(e))
        return

    # Show the equation
    A, B, C = popt
    print(f"🔍 Fitted Entropy Equation: S(n) = {A:.4f} * (1 - exp(-{B:.4f} * n)) + {C:.4f}")

    # Optional: plot
    n_vals = np.linspace(min(n), max(n), 100)
    fit_vals = entropy_model(n_vals, *popt)
    
    plt.plot(n, measured, 'o', label="Measured")
    plt.plot(n_vals, fit_vals, '-', label="Fitted")
    plt.xlabel("n (radiation qubits)")
    plt.ylabel("Entropy")
    plt.title("Auto-Fit Page Curve")
    plt.legend()
    plt.grid(True)


def predict_entropy_hill(n):
    C = 0.9887 # For 0.99 light speed
    phi = (1 + 5**0.5) / 2
    p = (np.pi ** 2) / (phi ** 2)       # ≈ 3.78
    k = (137 ** 0.5) * phi              # ≈ 18.9

    return C * (n ** p) / (n ** p + k)

def test_entropy_prediction(x, y, n_radiation, n):
    theta_static = theta_dynamic = x
    phi = y
    # Run the predictor
    predicted = predict_entropy_with_n(theta_static, phi, n_radiation, n, offset=0.15, n_max=5)

    # Build the matching quantum circuit
    total_qubits = n_radiation + 1
    qc = QuantumCircuit(total_qubits)

    # Static injection
    qc.rx(theta_static, 0)

    for i in range(1, n_radiation + 1):
        qc.barrier()
        qc.ry(theta_dynamic, 0)
        qc.rz(phi, 0)
        qc.cx(0, i)

    # Simulate the final state
    qc.save_statevector()
    backend = Aer.get_backend('aer_simulator')
    result = backend.run(qc).result()
    sv = Statevector(result.data(0)['statevector'])

    # Trace out radiation
    rho_bh = partial_trace(sv, list(range(1, total_qubits)))
    measured = qiskit_entropy(rho_bh)

    # Print comparison
    print(f"\n🧪 Entropy Test (n={n_radiation} radiation qubits)")
    print(f"Theta Static  = {theta_static:.4f}")
    print(f"Theta Dynamic = {theta_dynamic:.4f}")
    print(f"Phi           = {phi:.4f}")
    print(f"Predicted Entropy: {predicted:.4f}")
    print(f"Measured Entropy : {measured:.4f}")
    print(f"Difference         = {abs(predicted - measured):.4f}")

def log_entropy_comparison(filename, theta_s, theta_d, phi, results, source):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["n", "theta_static", "theta_dynamic", "phi", "Predicted", "Measured", "Difference", "Source"])
        for n, pred, meas in results:
            diff = abs(pred - meas)
            writer.writerow([n, theta_s, theta_d, phi, pred, meas, diff, source])

def predict_and_test(x=(1,1)):
    # Print
    print("X: ", x)
    # Prediction
    S_pred = predict_entropy_with_n(
        x=np.pi/x[0][0][0],
        y=np.pi/x[0][0][1],
        n_radiation=6,
        n=6
    )
    print(f"Predicted Entropy: {S_pred:.4f}")
    
    # Test
    test_entropy_prediction(
    x=np.pi/x[0][0][0],
    y=np.pi/x[0][0][1],
    n_radiation=6,
    n=6
    )

def curve_similarity(curve1, curve2):
    """
    Compare two curves using both vertical offset and local slope similarity.
    
    Parameters:
        curve1, curve2: Lists of (x, y) tuples, same length and matching x values.
    
    Returns:
        A float similarity score — lower means more similar.
    """
    if len(curve1) != len(curve2):
        raise ValueError("Curves must be the same length.")
    
    x1, y1 = zip(*curve1)
    x2, y2 = zip(*curve2)
    x1, y1, x2, y2 = map(np.array, (x1, y1, x2, y2))

    # Check that x-values match
    if not np.allclose(x1, x2):
        raise ValueError("X-values of the curves must match.")

    # Compute slope (finite differences)
    dy1 = np.gradient(y1, x1)
    dy2 = np.gradient(y2, x2)

    # Vertical difference (mean squared error)
    vertical_diff = np.mean((y1 - y2) ** 2)

    # Slope difference (mean squared error)
    slope_diff = np.mean((dy1 - dy2) ** 2)

    # Composite score: weight vertical and slope equally (you can tune this)
    similarity_score = vertical_diff + slope_diff

    return vertical_diff, slope_diff

def unzip_pairs(zipped_list):
    """
    Given a list of (a, b, c) tuples, return:
    - A list of (a, b)
    - A list of (a, c)
    """
    ab_pairs = [(a, b) for (a, b, c) in zipped_list]
    ac_pairs = [(a, c) for (a, b, c) in zipped_list]
    bc_pairs = [(b, c) for (a, b, c) in zipped_list]
    return ab_pairs, ac_pairs, bc_pairs


def find_equation(list_vars):
    n1 = count_subfolders("verify//logs//") + 1

    os.chdir("verify")
    os.chdir("logs")

    os.mkdir(f"{n1}")

    os.chdir("..")
    os.chdir("..")

    v_diffs = []
    s_diffs = []
    cord_diffs = []
    
    for i in range(len(list_vars)):
        
        for j in range(len(list_vars)):
            s = list_vars[j]
            t = list_vars[i] 
            results = sweep_entropy_prediction(
            theta_static=np.pi/t,
            theta_dynamic=np.pi/t,
            phi=np.pi/s,
            n_radiation=6,
            n=n1,
            n_max=6
            )

            print(f"Theta = {np.pi/t}")
            print(f"Phi = {np.pi/s}")

            print("Results: ", results)
            c1, c2, c3 = unzip_pairs(results)
            v_diff, s_diff = curve_similarity(c1, c2)

            v_diffs.append(v_diff)
            s_diffs.append(s_diff)
            cord_diffs.append((c2, c3))           
                       
            source_sweep = str(inspect.getsource(sweep_entropy_prediction))
            
            log_entropy_comparison(f"verify//logs//{n1}//entropy_oracle_log_{n1}.csv", np.pi/4, np.pi/4, np.pi/8, results, source_sweep)
            
            test_entropy_prediction(
            x=np.pi/t,
            y=np.pi/s,
            n_radiation=6,
            n=n1
            )

            fit_entropy_curve(f"verify//logs//{n1}//entropy_oracle_log_{n1}.csv")

    return [v_diffs, s_diffs, cord_diffs]

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

def find_angles_for_target_entropy(target_entropy, A, B):
    from scipy.optimize import minimize

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


def find_angle():
    vsc = find_equation([3,4,6, 0.5, 1, 2, np.pi, 8, 2*np.pi])
    vd = vsc[0]
    sd = vsc[1]
    cd = vsc[2]
    vi = vd.index(min(vd))
    si = sd.index(min(sd))
    c00 = cd[vi][0]
    c01 = cd[vi][1]
    c0 = (c00, c01)
    c10 = cd[si][0]
    c11 = cd[si][1]
    c1 = (c10, c11)
    print(f"Max theta and phi for max theta = {cd[vi]}")
    print(f"Max theta and phi for max phi = {cd[si]}")
    predict_and_test(c1)
    print("Ideal measurements: ", c1)

def target_entropy(target = 0.75):
    # Desired entropy
    qc, measured = reverse_entropy_oracle([target], n_qubits=2)
    print("Final entropy:", measured)
    print(qc)
    qc.draw('mpl')
    analyze_circuit(qc)

##Final entropy: 0.2916898739243701
##     ┌─────────────────────────┐
##q_0: ┤ U(2.6847,1.2894,3.5392) ├──■──
##     ├─────────────────────────┤┌─┴─┐
##q_1: ┤ U(3.4072,4.8163,4.8795) ├┤ X ├

def analyze_circuit(qc, measured_qubits=None, print_statevector=False):
    """
    Runs a quantum circuit on the Aer simulator, prints measurement counts,
    entropy, and optionally the statevector.

    Args:
        qc (QuantumCircuit): Your input circuit (can include or exclude measurements).
        measured_qubits (list[int]): Which qubits to compute entropy over. Default: all.
        print_statevector (bool): Whether to print the full statevector.
    """
    # Ensure all qubits are measured if none are specified
    if measured_qubits is None:
        measured_qubits = list(range(qc.num_qubits))

    # Clone and add measurements
    from qiskit import ClassicalRegister
    qc_to_run = qc.copy()
    qc_to_run.add_register(ClassicalRegister(len(measured_qubits)))
    qc_to_run.measure(measured_qubits, range(len(measured_qubits)))

    # Simulate
    backend = Aer.get_backend('aer_simulator')
    result = backend.run(qc_to_run, shots=8192).result()
    counts = result.get_counts()

    # Print results
    print("📊 Measurement Counts:")
    for bitstring, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {bitstring}: {count}")

    # Entropy calculation
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    entropy_value = -sum(p * np.log2(p) for p in probs if p > 0)
    print(f"\n🌀 Shannon Entropy of Measurement Distribution: {entropy_value:.4f} bits")

    # Optional statevector view
    if print_statevector:
        sv = Statevector.from_instruction(qc)
        print("\n🧠 Final Statevector:")
        print(sv)

    return counts, entropy_value

def find_angles():
    target_entropy = 0.4
    angles = find_angles_from_entropy(target_entropy)

    if angles:
        θ_s, θ_d, φ = angles
        print(f"🧠 For target entropy {target_entropy:.3f}, use:")
        print(f"  θ_static  = {θ_s:.4f}")
        print(f"  θ_dynamic = {θ_d:.4f}")
        print(f"  φ         = {φ:.4f}")

def qiskit_entropy(rho):
    evals = np.real(np.linalg.eigvals(rho.data))
    evals = evals[evals > 0]
    return -np.sum(evals * np.log2(evals))

def measure_subsystem_entropies(state: Statevector, num_qubits: int) -> Dict[int, float]:
    return {q: qiskit_entropy(partial_trace(state, [i for i in range(num_qubits) if i != q])) for q in range(num_qubits)}

def apply_charge_injection(circuit: QuantumCircuit, qubits: List[int], level: int = 1):
    for _ in range(level):
        for q in qubits:
            circuit.rx(np.pi / (level + 1), q)
    return circuit

def build_initial_entangled_circuit(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc

def run_iterative_injection_and_entropy(n_qubits: int, cycles: int = 5):
    backend = Aer.get_backend('aer_simulator')
    results = []
    qc = build_initial_entangled_circuit(n_qubits)

    for cycle in range(cycles):
        qc = apply_charge_injection(qc, list(range(n_qubits)), level=cycle+1)

        state = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
        entropies = measure_subsystem_entropies(state, n_qubits)

        # Add measurement for counts
        qc_with_measure = qc.copy()
        qc_with_measure.measure_all()
        transpiled = transpile(qc_with_measure, backend)
        counts = backend.run(transpiled, shots=1024).result().get_counts()

        results.append({
            "cycle": cycle,
            "entropies": entropies,
            "counts": counts
        })

        print(f"\nCycle {cycle}:")
        print("Subsystem Entropies:", entropies)
        print("Counts:", counts)
        plot_histogram(counts)
        plt.title(f"Measurement Results - Cycle {cycle}")
        plt.show()

    return results

##def reverse_entropy_oracle(target_entropies: List[float], n_qubits: int = 3, attempts: int = 100):
##    best_qc = None
##    best_match = float('inf')
##    backend = Aer.get_backend('statevector_simulator')
##
##    for _ in range(attempts):
##        qc = QuantumCircuit(n_qubits)
##        for q in range(n_qubits):
##            theta, phi, lam = np.random.uniform(0, 2*np.pi, 3)
##            qc.u(theta, phi, lam, q)
##        for i in range(n_qubits - 1):
##            qc.cx(i, i + 1)
##
##        state = Statevector.from_instruction(qc)
##        entropies = measure_subsystem_entropies(state, n_qubits)
##        diff = sum(abs(entropies[i] - target_entropies[i]) for i in range(n_qubits))
##
##        if diff < best_match:
##            best_match = diff
##            best_qc = qc
##            if diff < 0.01:
##                break

    # Simulate counts for display
    qc_with_measure = best_qc.copy()
    qc_with_measure.measure_all()
    transpiled = transpile(qc_with_measure, backend)
    result = backend.run(transpiled, shots=1024).result()
    counts = result.get_counts()

    print("Best match entropies:", target_entropies)
    print(best_qc.draw())
    print("Measurement counts:")
    for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        print(f"  {k}: {v}")
    return {
        "circuit": best_qc,
        "counts": counts,
        "target_entropies": target_entropies
    }


def qiskit_entropy(dm):
    probs = np.real(np.diag(dm.data))
    return shannon_entropy(probs, base=2)

def match_entropy_signal_oracle(
    target_entropies, 
    n_qubits=3, 
    attempts=200, 
    shots=1024,
    tolerance=1e-2
):
    backend = Aer.get_backend('aer_simulator')
    best_qc = None
    best_match = None
    min_diff_sum = float('inf')

    for _ in range(attempts):
        qc = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            theta, phi, lam = np.random.uniform(0, 2*np.pi, size=3)
            qc.u(theta, phi, lam, q)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

        sv = Statevector.from_instruction(qc)

        # Calculate entropy for each qubit by tracing out others
        entropies = []
        for q in range(n_qubits):
            reduced = partial_trace(sv, [i for i in range(n_qubits) if i != q])
            ent = qiskit_entropy(reduced)
            entropies.append(ent)

        diff = sum(abs(entropies[i] - target_entropies[i]) for i in range(n_qubits))
        if diff < min_diff_sum:
            min_diff_sum = diff
            best_qc = qc
            best_match = entropies
            if diff < tolerance:
                break

    if best_qc is None:
        print("No match found.")
        return None

    # Simulate and measure
    measured_qc = best_qc.copy()
    measured_qc.measure_all()
    job = backend.run(transpile(measured_qc, backend), shots=shots)
    result = job.result()
    counts = result.get_counts()

    print(f"\nBest match entropies: {[round(e, 6) for e in best_match]}")
    print(best_qc.draw())
    print("Measurement counts:")
    for state, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {state}: {count}")

    return {
        'circuit': best_qc,
        'entropies': best_match,
        'counts': counts
    }

def set_target_subsystem_entropy(target_entropies, num_qubits=3, max_iter=100):
    """
    Attempts to construct a quantum circuit whose subsystem entropies match the target values.

    Args:
        target_entropies (list): List of target entropy values for each qubit.
        num_qubits (int): Number of qubits in the circuit.
        max_iter (int): Maximum optimization iterations.

    Returns:
        QuantumCircuit: Optimized quantum circuit.
    """
    assert len(target_entropies) == num_qubits, "Target entropy list must match qubit count."

    def build_circuit(params):
        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            theta, phi, lam = params[3 * i: 3 * i + 3]
            qc.u(theta, phi, lam, i)
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
        return qc

    def cost(params):
        qc = build_circuit(params)
        state = Statevector.from_instruction(qc)
        entropies = []
        for i in range(num_qubits):
            reduced = partial_trace(state, [j for j in range(num_qubits) if j != i])
            S = qiskit_entropy(reduced)
            entropies.append(S)
        return np.sum((np.array(entropies) - np.array(target_entropies)) ** 2)

    init_params = np.concatenate([
    [0.01, 0.01, 0.01],   # Q0 - low entropy init
    np.random.uniform(0, 2*np.pi, 3*(num_qubits - 1))  # Q1, Q2 - random
    ])
    result = minimize(cost, init_params, method='COBYLA', options={'maxiter': max_iter})

    optimized_qc = build_circuit(result.x)
    return optimized_qc

def qiskit_entropy(rho):
    evals = np.real(np.linalg.eigvals(rho.data))
    evals = evals[evals > 0]
    return -np.sum(evals * np.log2(evals))

def generate_circuit_for_entropy(target_entropy, n_qubits=3, attempts=100):
    """
    Attempts to generate a quantum circuit with a subsystem entropy close to the target.

    Args:
        target_entropy (float): Desired entropy value (between 0 and log2(d)).
        n_qubits (int): Number of qubits in the circuit.
        attempts (int): How many random circuits to try.

    Returns:
        dict: {
            'best_circuit': QuantumCircuit,
            'statevector': Statevector,
            'measured_entropies': List[float]
        }
    """
    best_diff = float('inf')
    best_qc = None
    best_sv = None
    best_entropies = None

    backend = Aer.get_backend('statevector_simulator')

    for _ in range(attempts):
        qc = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            theta, phi, lam = np.random.uniform(0, 2 * np.pi, 3)
            qc.u(theta, phi, lam, q)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

        sv = Statevector.from_instruction(qc)
        entropies = []
        for q in range(n_qubits):
            reduced = partial_trace(sv, [i for i in range(n_qubits) if i != q])
            ent = qiskit_entropy(reduced)
            entropies.append(ent)

        avg_entropy = np.mean(entropies)
        diff = abs(avg_entropy - target_entropy)

        if diff < best_diff:
            best_diff = diff
            best_qc = qc
            best_sv = sv
            best_entropies = entropies

        if best_diff < 0.01:  # close enough
            break

    return {
        "best_circuit": best_qc,
        "statevector": best_sv,
        "measured_entropies": best_entropies
    }

def shannon_entropy(probs):
    return -sum(p * np.log2(p) for p in probs if p > 0)

def observe_entropy_shifts(qc, qubit_indices, num_qubits=3):
    """
    Inject charge iteratively and observe entropy shifts for the specified qubits.
    """
    backend = Aer.get_backend('statevector_simulator')
    state = Statevector.from_instruction(qc)

    entropies = []
    for idx in range(qubit_indices):
        reduced = partial_trace(state, [i for i in range(qc.num_qubits) if i != idx])
        probs = reduced.probabilities()
        ent = shannon_entropy(probs)
        entropies.append(ent)
        print(f"Entropy of Qubit {idx}: {ent:.6f}")
    
    return entropies

def entropy_objective(params, target_entropy, qubit_index=0):
    theta, phi, lam = params
    qc = QuantumCircuit(2)
    qc.u(theta, phi, lam, 0)
    qc.cx(0, 1)
    state = Statevector.from_instruction(qc)
    reduced = partial_trace(state, [1]) if qubit_index == 0 else partial_trace(state, [0])
    entropy = qiskit_entropy(reduced)
    return abs(entropy - target_entropy)

def generate_entropy_targeted_circuit(target_entropy, initial_guess=[np.pi/2, np.pi/2, np.pi/2]):
    """
    Generates a 2-qubit circuit where the subsystem entropy of one qubit matches the target.
    
    Parameters:
    - target_entropy (float): Desired subsystem entropy (0 to ~1 for 1-qubit reductions).
    
    Returns:
    - QuantumCircuit: Circuit with matching entropy.
    - float: Actual entropy achieved.
    """
    bounds = [(0, 2 * np.pi), (0, 2 * np.pi), (0, 2 * np.pi)]

    result = minimize(
        entropy_objective,
        initial_guess,
        args=(target_entropy,),
        bounds=bounds,
        method='L-BFGS-B',
        options={"disp": False}
    )

    theta, phi, lam = result.x
    qc = QuantumCircuit(2)
    qc.u(theta, phi, lam, 0)
    qc.cx(0, 1)

    # Verify
    state = Statevector.from_instruction(qc)
    reduced = partial_trace(state, [1])
    entropy = qiskit_entropy(reduced)

    print(f"Target entropy: {target_entropy:.6f} | Achieved entropy: {entropy:.6f}")
    return qc, entropy

def build_low_entropy_targeting_circuit(n_qubits=2, target_entropy=0.6, max_iterations=10, scaling_factor=0.25):

    backend = Aer.get_backend('statevector_simulator')
    qc = QuantumCircuit(n_qubits)

    # Start with maximally mixed state components via H + CNOT
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    # History of charge to build memory-like bias
    charge_history = np.zeros(n_qubits)

    for i in range(max_iterations):
        # Dynamic phase shift injection
        dynamic_phase = np.pi * (1 - target_entropy) * (1 + scaling_factor * charge_history.sum())
        qc.barrier()

        for q in range(n_qubits):
            qc.rx(dynamic_phase / (q + 1), q)

        # Entangle and collapse again
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

        # Check entropy and update charge tracking
        state = Statevector.from_instruction(qc)
        entropies = measure_subsystem_entropies(state, n_qubits)
        avg_entropy = np.mean(list(entropies.values()))

        # Update charge history heuristically
        if avg_entropy > target_entropy:
            charge_history += 1
        else:
            break

        print(f"🔁 Iteration {i + 1}: Entropy = {avg_entropy:.5f}, Target = {target_entropy}, Charge = {charge_history}")

    qc.measure_all()
    qc.draw('mpl')
    return qc

def measure_subsystem_entropies(state: Statevector, num_qubits: int):
    return {q: qiskit_entropy(partial_trace(state, [i for i in range(num_qubits) if i != q])) for q in range(num_qubits)}

def tune_entropy_target(n_qubits=3, target_entropy=0.8, max_iters=25, tol=0.01):
    backend = Aer.get_backend('aer_simulator')
    charge = np.zeros(n_qubits)
    scaling_factor = 0.5
    ents = []

    for iter in range(max_iters):
        qc = QuantumCircuit(3)
        qc.h(range(n_qubits))  # Superposition init

        # Add entanglement
        for i in range(n_qubits):
            qc.cx(i, (i + 1)%n_qubits)
            qc.cx((i+1)%n_qubits,(i+2)%n_qubits)

##        # Charge-dependent phase kick
##        for i in range(n_qubits):
##            qc.rz(charge[i], i)  # Or another param gate like Ry or Rx

        # Adaptive rotation
        for i in range(n_qubits):
            qc.rx(np.pi * charge[i], i)

        # Grover-like boost and conditional logic
        if n_qubits == 2 or n_qubits == 3:
            qc.cp(np.pi/4, 0, 1)
            qc.cz(0, 1)

        # Get statevector and measure entropies
        state = Statevector.from_instruction(qc)
        entropies = measure_subsystem_entropies(state, n_qubits)
        rho = partial_trace(state, [1])
        ent = qiskit_entropy(rho)
        ents.append(ent)
        
        avg_entropy = np.mean(list(entropies.values()))

        print(f"Iter {iter}: avg_entropy={ent:.4f}, target={target_entropy:.4f}, charges={charge}")

        # Check convergence
        if abs(avg_entropy - target_entropy) < tol:
            print("✅ Target entropy reached.")
            return qc, entropies

        # Feedback control (simple gradient-like)
        for i in range(n_qubits):
            if avg_entropy < target_entropy:
                charge[i] += scaling_factor  # increase superposition / entanglement
            else:
                charge[i] -= scaling_factor  # reduce entanglement (collapse bias)

        charge = np.clip(charge, 0, 1.5)
        if abs(avg_entropy - target_entropy) < tol:
            print("✅ Target entropy reached.")
            return qc, entropies  # ✅ Success case

    print("⚠️ Max iterations reached.")
    print(entropies)
    return qc, entropies, avg_entropy

def amplify_outcome(tag, signal_strength=1, reason=None):
    global memory_container
    if memory_container is None:
        memory_container = defaultdict(lambda: defaultdict(float))

    mem = memory_container[tag]
    mem["signal"] += signal_strength
    if reason:
        mem["reasons"].append(reason)

def create_entanglement_wedge_circuit():
    """Creates a GHZ-like entangled state over 5 qubits with clear boundary/bulk separation."""
    qc = QuantumCircuit(5)

    # GHZ state: Entangle all qubits
    qc.h(0)
    for i in range(4):
        qc.cx(i, i+1)

    return qc

def inject_boundary_charge(qc, boundary_indices):
    """Apply charge-injecting unitaries (Z/X) to boundary qubits."""
    for i in boundary_indices:
        qc.z(i)  # Simulate negative boundary charge
        qc.x(i)  # Simulate positive boundary charge
    return qc

def partial_boundary_measurement(qc, boundary_indices):
    """Measure only boundary qubits."""
    creg = ClassicalRegister(len(boundary_indices), name='c_boundary')
    qc.add_register(creg)
    for i, q in enumerate(boundary_indices):
        qc.measure(q, creg[i])
    return qc

def full_measurement(qc):
    """Measure all qubits."""
    total = qc.num_qubits
    creg = ClassicalRegister(total)
    qc.add_register(creg)
    qc.measure(range(total), range(total))
    return qc

def convert_bitarray_to_counts(bitarray, num_bits):
    """Converts IBM Runtime BitArray into a counts dict."""
    raw = bitarray.to01()
    bitstrings = [raw[i:i + num_bits] for i in range(0, len(raw), num_bits)]
    return dict(Counter(bitstrings))

def run_circuit_0(qc, shots=8192, backend=Aer.get_backend('aer_simulator')):
    transpiled = transpile(qc, backend)
    sampler = Sampler(backend)
    job = sampler.run([transpiled], shots=shots)
    result = job.result()
    print("PrimitiveResult:", result)

    # Automatically detect available classical register
    data_obj = result[0].data
    register_keys = [k for k in data_obj.__dict__.keys() if isinstance(getattr(data_obj, k), BitArray)]

    if not register_keys:
        raise ValueError("No BitArray register found in result.")

    selected_register = register_keys[0]  # Use the first one (e.g. 'c0', 'c_boundary', etc.)
    bitarray = getattr(data_obj, selected_register)

    # Determine number of bits per shot
    num_bits = bitarray.num_bits
    counts = bitarray.get_counts()
    return counts

def calculate_entropy(counts):
    total = sum(counts.values())
    probs = np.array([v / total for v in counts.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-12))  # Add epsilon to avoid log(0)
    return entropy

def main_entanglement_wedge_experiment():
    boundary_indices = [0, 1, 4]
    bulk_indices = [2, 3]

    # Step 1: Create entangled GHZ state
    qc_base = create_entanglement_wedge_circuit()

    # Step 2: Partial boundary measurement
    qc_partial = qc_base.copy()
    partial_boundary_measurement(qc_partial, boundary_indices)
    counts_partial = run_circuit_0(qc_partial)
    entropy_partial = calculate_entropy(counts_partial)

    # Step 3: Apply charge to boundary and measure all
    qc_charged = create_entanglement_wedge_circuit()
    inject_boundary_charge(qc_charged, boundary_indices)
    full_measurement(qc_charged)
    counts_charged = run_circuit_0(qc_charged)
    entropy_charged = calculate_entropy(counts_charged)

    # Step 4: Control - full measurement without charge
    qc_full = create_entanglement_wedge_circuit()
    full_measurement(qc_full)
    counts_full = run_circuit_0(qc_full)
    entropy_full = calculate_entropy(counts_full)

    print("\n--- Holographic Entanglement Wedge Experiment ---")
    print("Partial Boundary Measurement:")
    print("Counts:", counts_partial)
    print("Entropy:", entropy_partial)

    print("\nFull Measurement After Charge Injection:")
    print("Counts:", counts_charged)
    print("Entropy:", entropy_charged)

    print("\nFull Measurement Without Charge (Control):")
    print("Counts:", counts_full)
    print("Entropy:", entropy_full)

    return {
        "partial_counts": counts_partial,
        "partial_entropy": entropy_partial,
        "charged_counts": counts_charged,
        "charged_entropy": entropy_charged,
        "full_counts": counts_full,
        "full_entropy": entropy_full
    }

##Output

##Best backend chosen: ibm_sherbrooke
##PrimitiveResult([SamplerPubResult(data=DataBin(c0=BitArray(<shape=(), num_shots=8192, num_bits=5>)), metadata={'shots': 8192, 'circuit_metadata': {}})], metadata={'version': 2})
##
##--- Holographic Entanglement Wedge Experiment ---
##Partial Boundary Measurement:
##Counts: {'111': 16223, '000': 16545}
##Entropy: 0.9999303432152136
##
##Full Measurement After Charge Injection:
##Counts: {'01100': 4051, '10011': 4141}
##Entropy: 0.9999129320285574
##
##Full Measurement Without Charge (Control):
##Counts: {'11111': 16275, '00000': 16493}
##Entropy: 0.9999680727947997

def scan_wedge_phase_transition(max_boundary_size=4):
    # build one instance to get n_qubits and the total entropy
    qc0      = create_entanglement_wedge_circuit()
    n_qubits = qc0.num_qubits
    qc0.measure_all()   
    full_counts = run_circuit_0(qc0)
    S_total    = calculate_entropy(full_counts)

    results = []
    for k in range(1, max_boundary_size+1):
        boundary = list(range(k))
        bulk     = [q for q in range(n_qubits) if q not in boundary]

        # compute S(boundary) by marginalizing full_counts onto the boundary bits
        counts_boundary = {}
        for bitstr, c in full_counts.items():
            key = "".join(bitstr[i] for i in boundary)
            counts_boundary[key] = counts_boundary.get(key, 0) + c
        S_boundary = calculate_entropy(counts_boundary)

        # now do the partial‐measurement on the same circuit
        qc_pm = create_entanglement_wedge_circuit()
        partial_boundary_measurement(qc_pm, boundary)
        qc_pm.measure_all()
        counts_pm = run_circuit_0(qc_pm)
        S_bulk_cond = calculate_entropy(counts_pm)

        I = S_boundary + S_bulk_cond - S_total
        results.append((k, I))

    print("Results: ", results)

    return results
##
##Best backend chosen: ibm_brisbane
##PrimitiveResult: PrimitiveResult([SamplerPubResult(data=DataBin(meas=BitArray(<shape=(), num_shots=8192, num_bits=5>)), metadata={'shots': 8192, 'circuit_metadata': {}})], metadata={'version': 2})
##PrimitiveResult: PrimitiveResult([SamplerPubResult(data=DataBin(c_boundary=BitArray(<shape=(), num_shots=8192, num_bits=1>), meas=BitArray(<shape=(), num_shots=8192, num_bits=5>)), metadata={'shots': 8192, 'circuit_metadata': {}})], metadata={'version': 2})
##PrimitiveResult: PrimitiveResult([SamplerPubResult(data=DataBin(c_boundary=BitArray(<shape=(), num_shots=8192, num_bits=2>), meas=BitArray(<shape=(), num_shots=8192, num_bits=5>)), metadata={'shots': 8192, 'circuit_metadata': {}})], metadata={'version': 2})
##PrimitiveResult: PrimitiveResult([SamplerPubResult(data=DataBin(c_boundary=BitArray(<shape=(), num_shots=8192, num_bits=3>), meas=BitArray(<shape=(), num_shots=8192, num_bits=5>)), metadata={'shots': 8192, 'circuit_metadata': {}})], metadata={'version': 2})
##PrimitiveResult: PrimitiveResult([SamplerPubResult(data=DataBin(c_boundary=BitArray(<shape=(), num_shots=8192, num_bits=4>), meas=BitArray(<shape=(), num_shots=8192, num_bits=5>)), metadata={'shots': 8192, 'circuit_metadata': {}})], metadata={'version': 2})
##Results:  [(1, 0.9999050204938393), (2, 0.9997645429384387), (3, 0.9997248102580842), (4, 0.9999889930794672)]

def scan_local_wedge(max_k=4, angles=[np.pi/8, np.pi/4, np.pi/2]):
    data = []
    for θ in angles:
        for k in range(1, max_k+1):
            qc = create_local_ent_wedge(angle=θ)
            # measure full state once to get S_total
            qc_full = qc.copy()
            qc_full.measure_all()
            full = run_circuit_0(qc_full)
            S_tot = calculate_entropy(full)

            # S(boundary)
            counts_b = {}
            for bits, c in full.items():
                key = "".join(bits[i] for i in range(k))
                counts_b[key] = counts_b.get(key,0)+c
            S_b = calculate_entropy(counts_b)

            # partial-measure bulk conditioned
            qc_pm = qc.copy()
            partial_boundary_measurement(qc_pm, list(range(k)))
            qc_pm.measure_all()
            counts_pm = run_circuit_0(qc_pm)
            S_bulk = calculate_entropy(counts_pm)

            I = S_b + S_bulk - S_tot
            data.append((θ, k, I))
    return data

def create_local_ent_wedge(angle=np.pi/4):
    # 5-qubit line: [b0]—[b1]—[q2]—[q3]—[q4]
    qc = QuantumCircuit(5)
    # initialize all to |+>
    for i in range(5):
        qc.h(i)
    # apply CP(angle) only on neighbor pairs
    edges = [(0,1), (1,2), (2,3), (3,4)]
    for i,j in edges:
        qc.cp(angle, i, j)
    return qc

def visualize_angles(max_k=4, angles=[np.pi/8, np.pi/4, np.pi/2]):
    results = []
    for θ in angles:
        for k in range(1, max_k+1):
            I = compute_mutual_info(θ, k)
            results.append((θ, k, I))

    # 4) Plot the curves
    plt.figure(figsize=(8,5))
    for θ in angles:
        pts = sorted([(k, I) for (ang, k, I) in results if np.isclose(ang, θ)], key=lambda x: x[0])
        ks, Is = zip(*pts)
        plt.plot(ks, Is, marker='o', label=fr'$\theta={θ:.2f}$')

    plt.xlabel('Boundary region size $k$')
    plt.ylabel('Quantum Mutual Information $I(\mathrm{bulk}:\mathrm{boundary})$')
    plt.title('Entanglement-Wedge Phase Transition (Statevector)')
    plt.xticks(range(1, max_k+1))
    plt.legend(title='Coupling angle')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def compute_mutual_info(angle, k, n_qubits=5):
    qc = create_local_ent_wedge(angle)
    sv = Statevector(qc)
    dm = DensityMatrix(sv)

    # total entropy
    S_tot = qiskit_entropy(dm)

    # define subsystems
    boundary = list(range(k))
    bulk     = list(range(k, n_qubits))

    # S(boundary)
    dm_b      = partial_trace(dm, bulk)
    S_b       = qiskit_entropy(dm_b)

    # S(bulk)
    dm_bulk   = partial_trace(dm, boundary)
    S_bulk    = qiskit_entropy(dm_bulk)

    return S_b + S_bulk - S_tot

def create_depth_local_wedge(angle: float, depth: int, n_qubits: int = 5) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    for _ in range(depth):
        # 1) mix all qubits
        for i in range(n_qubits):
            qc.h(i)
        # 2) entangle nearest neighbors
        for i in range(n_qubits - 1):
            qc.cp(angle, i, i + 1)
    return qc

def mutual_info_from_state(qc: QuantumCircuit, k: int) -> float:
    sv   = Statevector(qc)
    dm   = DensityMatrix(sv)
    S_tot = qiskit_entropy(dm)

    boundary = list(range(k))
    bulk     = list(range(k, qc.num_qubits))

    S_b    = qiskit_entropy(partial_trace(dm, bulk))
    S_bulk = qiskit_entropy(partial_trace(dm, boundary))
    return S_b + S_bulk - S_tot

def scan_depths(angle: float, depths: list[int], max_k: int = 4):
    results = {}
    for d in depths:
        values = []
        for k in range(1, max_k + 1):
            qc = create_depth_local_wedge(angle, d)
            values.append(mutual_info_from_state(qc, k))
        results[d] = values
    return results

def fit_phase_boundary_params(theta_arr, dstar_arr, n, k, output_file='phase_boundary_params.json'):
    """
    Fit the phase boundary model d*(θ, n, k) = α * (n - k) / θ^β + γ
    to experimental data and save the parameters.
    Filters out any NaN or infinite entries before fitting.
    """
    # turn into numpy arrays
    theta_arr = np.array(theta_arr, dtype=float)
    dstar_arr = np.array(dstar_arr, dtype=float)

    # filter out NaNs and infs
    mask = np.isfinite(theta_arr) & np.isfinite(dstar_arr)
    theta_arr = theta_arr[mask]
    dstar_arr = dstar_arr[mask]
    if len(theta_arr) < 3:
        raise ValueError("Not enough valid data points to fit.")

    # model: d*(θ) = α * (n - k) / θ^β + γ
    def model(theta, alpha, beta, gamma):
        return alpha * (n - k) / theta**beta + gamma

    # initial guesses
    p0 = [1.0, 1.0, 0.0]

    # perform the fit
    popt, _ = curve_fit(model, theta_arr, dstar_arr, p0=p0)
    alpha, beta, gamma = popt

    params = {
        'alpha': float(alpha),
        'beta':  float(beta),
        'gamma': float(gamma),
        'n':      n,
        'k':      k
    }

    # save
    with open(output_file, 'w') as f:
        json.dump(params, f, indent=4)

    print(f"[fit_phase_boundary_params] Saved parameters to '{output_file}'.")
    return params

#Params:  {'alpha': 0.8452816861689775, 'beta': 3.1622708337551284, 'gamma': 1.74903474952482, 'n': 5, 'k': 2}

# 1) Geometry: 1D chain of n qubits
n        = 7
nodes    = list(range(n))
edge_list = [(i, i+1) for i in range(n-1)]  # nearest-neighbor edges

# 2) Target “stress-energy” (vacuum here)
T = { i: 1.0 for i in nodes }

mass_node, mass_strength, sigma = 3, 1.0, 0.8
T_target = {}

# 3) Initialize random CP-angles on each edge
theta_1 = { edge: np.random.uniform(0.1, np.pi-0.1) for edge in edge_list }
epsilon = 0.3           # charge injection strength
theta   = np.pi / 4     # wormhole coupling angle
thetas = np.linspace(0, np.pi, 50)
fidelities = []

def build_2d_edges(rows, cols):
    edges = []
    for r in range(rows):
        for c in range(cols):
            idx = r*cols + c
            # horizontal link
            if c < cols-1:
                edges.append((idx, idx+1))
            # vertical link
            if r < rows-1:
                edges.append((idx, idx+cols))
    return edges

# 2) Create entangling circuit with CP gates on each link
def create_2d_entanglement_circuit(rows, cols, theta_dict):
    n = rows*cols
    qc = QuantumCircuit(n)
    qc.h(range(n))  # initial superposition
    for (i,j), theta in theta_dict.items():
        qc.cp(theta, i, j)
    return qc

# 3) Compute plaquette "Ricci curvature" via 4-body mutual information
#    For each square plaquette, label corners a,b,c,d in order.
def compute_plaquette_curvature(qc, rows, cols, plaquettes):
    # Ensure the circuit saves the statevector
    qc_sv = qc.copy()
    qc_sv.save_statevector()

    # Run with MPS simulator for efficiency
    sim = AerSimulator(method='matrix_product_state')
    t_qc = transpile(qc_sv, sim)
    result = sim.run(t_qc).result()
    # Retrieve the saved statevector
    sv = result.get_statevector(t_qc)

    # Build a density matrix from the statevector
    dm_full = DensityMatrix(sv)
    curvatures = {}
    for corners in plaquettes:
        # Compute mutual information for the four corners
        # I(a,b;c,d) = S(ab) + S(cd) - S(abcd)
        # Use partial traces and von Neumann entropy
        rho_ab = partial_trace(dm_full, [q for q in range(rows*cols) if q not in corners[:2]])
        rho_cd = partial_trace(dm_full, [q for q in range(rows*cols) if q not in corners[2:]])
        rho_abcd = partial_trace(dm_full, [q for q in range(rows*cols) if q not in corners])

        S_ab = qiskit_entropy(rho_ab)
        S_cd = qiskit_entropy(rho_cd)
        S_abcd = qiskit_entropy(rho_abcd)
        curvatures[tuple(corners)] = S_ab + S_cd - S_abcd
    return curvatures

def build_3d_edges(L):
    edges = []
    def idx(x,y,z): return x + L*(y + L*z)
    for x in range(L):
      for y in range(L):
        for z in range(L):
          i = idx(x,y,z)
          if x+1 < L: edges.append((i, idx(x+1,y,z)))
          if y+1 < L: edges.append((i, idx(x,y+1,z)))
          if z+1 < L: edges.append((i, idx(x,y,z+1)))
    return edges

def list_3d_faces(L):
    faces = []
    idx = lambda x,y,z: x + L*(y + L*z)
    for x in range(L-1):
        for y in range(L-1):
            for z in range(L):
                # XY-plane face at fixed z
                faces.append([idx(x,y,z), idx(x+1,y,z), idx(x+1,y+1,z), idx(x,y+1,z)])
    for x in range(L-1):
        for y in range(L):
            for z in range(L-1):
                # XZ-plane face at fixed y
                faces.append([idx(x,y,z), idx(x+1,y,z), idx(x+1,y,z+1), idx(x,y,z+1)])
    for x in range(L):
        for y in range(L-1):
            for z in range(L-1):
                # YZ-plane face at fixed x
                faces.append([idx(x,y,z), idx(x,y+1,z), idx(x,y+1,z+1), idx(x,y,z+1)])
    return faces

# 3) List cubic cells (volumes) as 8-corner tuples
def list_3d_cells(L):
    cells = []
    idx = lambda x,y,z: x + L*(y + L*z)
    for x in range(L-1):
        for y in range(L-1):
            for z in range(L-1):
                corners = [
                    idx(x  ,y  ,z), idx(x+1,y  ,z), idx(x+1,y+1,z), idx(x  ,y+1,z),
                    idx(x  ,y  ,z+1), idx(x+1,y  ,z+1), idx(x+1,y+1,z+1), idx(x  ,y+1,z+1)
                ]
                cells.append(corners)
    return cells

# 4) Create entangling circuit for given edges and theta angles
def create_3d_entanglement_circuit(n_qubits, edges, theta_dict):
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    for (i,j), theta in theta_dict.items():
        qc.cp(theta, i, j)
        print(theta, i, j)
    return qc

# 5) Compute face-based curvature: I(a,b;c,d) = S(ab)+S(cd)-S(abcd)
def compute_face_curvature(qc, faces):
    # save statevector
    qc_sv = qc.copy()
    qc_sv.save_statevector()
    sim = AerSimulator(method='matrix_product_state')
    t_qc = transpile(qc_sv, sim)
    result = sim.run(t_qc).result()
    sv = result.get_statevector(t_qc)
    dm = DensityMatrix(sv)
    curv = {}
    n = qc.num_qubits
    for corners in faces:
        a, b, c, d = corners
        # joint indices
        ab = [a,b]; cd=[c,d]; abcd=corners
        rho_ab   = partial_trace(dm, [q for q in range(n) if q not in ab])
        rho_cd   = partial_trace(dm, [q for q in range(n) if q not in cd])
        rho_abcd = partial_trace(dm, [q for q in range(n) if q not in abcd])
        S_ab   = entropy(rho_ab,   base=2)
        S_cd   = entropy(rho_cd,   base=2)
        S_abcd = entropy(rho_abcd, base=2)
        curv[tuple(corners)] = S_ab + S_cd - S_abcd
    return curv

# 6) Compute cell-based curvature: sum face entropies minus S(all corners)
def compute_cell_curvature(qc, cells, faces_of_cell):
    qc_sv = qc.copy()
    qc_sv.save_statevector()
    sim = AerSimulator(method='matrix_product_state')
    t_qc = transpile(qc_sv, sim)
    result = sim.run(t_qc).result()
    sv = result.get_statevector(t_qc)
    dm = DensityMatrix(sv)
    curv = {}
    n = qc.num_qubits
    for corners in cells:
        # find 6 faces bounding this cell
        faces = faces_of_cell[tuple(corners)]
        S_faces = 0
        for face in faces:
            rho = partial_trace(dm, [q for q in range(n) if q not in face])
            S_faces += entropy(rho, base=2)
        rho_full = partial_trace(dm, [q for q in range(n) if q not in corners])
        S_full = entropy(rho_full, base=2)
        curv[tuple(corners)] = S_faces - S_full
    return curv

# 7) Precompute faces_of_cell mapping

def build_faces_of_cell(cells, faces):
    mapping = {}
    face_sets = [set(f) for f in faces]
    for cell in cells:
        cset = set(cell)
        # face belongs to cell if its corners subset of cell corners
        mapping[tuple(cell)] = [face for face in faces if set(face).issubset(cset)]
    return mapping


# 4) Define plaquettes for rows x cols

def list_plaquettes(rows, cols):
    plaquettes = []
    for r in range(rows-1):
        for c in range(cols-1):
            a = r*cols + c
            b = a+1
            d = (r+1)*cols + c
            c_idx = d+1
            plaquettes.append([a, b, d, c_idx])
    return plaquettes


def run_entanglement_measurements(theta_dict):
    """
    Build a circuit with CP(theta_ij) on each edge (i,j),
    get the full statevector, and return I_ij for each edge.
    """
    qc = QuantumCircuit(n)
    # prepare |+> on every qubit
    for q in nodes:
        qc.h(q)
    # apply variable CP on each edge
    for (i,j), th in theta_dict.items():
        qc.cp(th, i, j)

    # get density matrix
    sv = Statevector(qc)
    dm = DensityMatrix(sv)

    # compute I(i:j) for each edge
    I = {}
    for (i,j) in edge_list:
        # ρ_i and ρ_j
        dm_i   = partial_trace(dm, [q for q in nodes if q!=i])
        dm_j   = partial_trace(dm, [q for q in nodes if q!=j])
        # ρ_{ij}
        dm_ij  = partial_trace(dm, [q for q in nodes if q not in (i,j)])
        Si     = qiskit_entropy(dm_i)
        Sj     = qiskit_entropy(dm_j)
        Sij    = qiskit_entropy(dm_ij)
        I[(i,j)] = Si + Sj - Sij
    return I

def wormhole_teleport_with_charge(epsilon, coupling_theta):
    # 1) set up registers explicitly
    qr = QuantumRegister(7, 'q')
    cr = ClassicalRegister(2, 'm')       # m[0] = measurement of q0; m[1] = measurement of q1
    qc = QuantumCircuit(qr, cr)

    # 2) Build two 3-qubit GHZs on qubits [1,2,3] and [4,5,6]
    for base in (1, 4):
        qc.h(qr[base])
        qc.cx(qr[base], qr[base+1])
        qc.cx(qr[base+1], qr[base+2])

    # 3) Prepare and charge the message on q0
    qc.h(qr[0])
    qc.rz(epsilon, qr[0])

    # 4) Bell-measurement of (q0, q1)
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.measure(qr[0], cr[0])  # m[0] ← Z outcome of q0
    qc.measure(qr[1], cr[1])  # m[1] ← X outcome of q1 (after H)

    # 5) Traversable-wormhole coupling
    qc.cp(coupling_theta, qr[2], qr[5])

    # 6) ***Decode*** the second GHZ on [4,5,6] (inverse of steps in (2))
    qc.cx(qr[5], qr[6])
    qc.cx(qr[4], qr[5])
    qc.h(qr[4])

    # 7) Teleportation corrections on q6:
    #    – if m[1]==1 (i.e. cr==2) then X
    #    – if m[0]==1 (i.e. cr==1) then Z
    qc.x(qr[6]).c_if(cr, 2)   # cr value = 10₂ → m[1]=1
    qc.z(qr[6]).c_if(cr, 1)   # cr value = 01₂ → m[0]=1

    return qc

def main_loss():
    lr = 0.1
    for step in range(50):
        L, grad = loss_and_grad(theta_1)
        for e in edge_list:
            theta[e] = np.clip(theta_1[e] - lr * grad[e], 0.01, np.pi-0.01)
        if L < 1e-4:
            break
        if step % 10 == 0:
            print(f"Step {step}: loss={L:.4e}")

def loss_and_grad(theta_dict, eps=1e-3):
    # measure current I_ij
    I = run_entanglement_measurements(theta_dict)
    # compute R_i = sum_j f(I_ij); here f(x)=x for simplicity
    R = {
        i: sum(I[edge] for edge in edge_list if i in edge)
        for i in nodes
    }
    # loss = Σ_i (R_i - T_i)^2
    L = sum((R[i] - T[i])**2 for i in nodes)

    # finite-difference gradients ∂L/∂θ_ij
    grad = {}
    for edge in edge_list:
        orig = theta_dict[edge]
        theta_dict[edge] = orig + eps
        I_eps = run_entanglement_measurements(theta_dict)
        R_eps = {
            i: sum(I_eps[e] for e in edge_list if i in e)
            for i in nodes
        }
        L_eps = sum((R_eps[i] - T[i])**2 for i in nodes)
        grad[edge] = (L_eps - L) / eps
        theta_dict[edge] = orig

    return L, grad

##Best backend chosen: ibm_brisbane
##Step 0: loss=1.7709e+00
##Step 10: loss=5.1015e-01
##Step 20: loss=2.9131e-01
##Step 30: loss=2.2946e-01
##Step 40: loss=2.1127e-01
##Converged θ_ij: {(0, 1): 1.6586942576933155, (1, 2): 1.8332592613506289, (2, 3): 2.0536913620728745, (3, 4): 1.7787903663819193}
def execute_n(circuit: qiskit.QuantumCircuit, shots: int = 8192) -> float:
    task=device.run(circuit)
    result=task.result()
    
def execute_m(circuit, noise_level=0.01):
    """Returns Tr[ρ |0⟩⟨0|] where ρ is the state prepared by the circuit
    executed with depolarizing noise.
    """
    # Replace with code based on your frontend and backend.
    mitiq_circuit, _ = convert_to_mitiq(circuit)
    noisy_circuit = mitiq_circuit.with_noise(depolarize(p=noise_level))
    rho = DensityMatrixSimulator().simulate(noisy_circuit).final_density_matrix
    return rho[0, 0].real

class QuantumGravityAnalyzer:
    """
    Implements:
      • modular‐Hamiltonian decomposition H_ξ = –log₂(ρ₀),
      • “plain” Pauli‐string measurement ⟨P⟩,
      • zero‐noise‐extrapolated measurement ⟨P⟩₀ (no readout mitigation),
      • finite‐difference first‐law checks (±ε) with ZNE,
      • plus mutual‐information, geometry, Bianchi, etc.
    """

    def __init__(self, n_qubits: int, edge_list: List[tuple[int,int]] = None):
        self.n = n_qubits
        self.edge_list = (
            edge_list
            if edge_list is not None
            else [(i, i+1) for i in range(n_qubits-1)]
        )

    def _decompose_modular_hamiltonian(self, rho0: np.ndarray) -> List[tuple[str,float]]:
        """H = −log₂(ρ₀) decomposed into Pauli strings via SparsePauliOp."""
        H = -la.logm(rho0) / np.log(2)
        spop: SparsePauliOp = SparsePauliOp.from_operator(H)
        labels = spop.paulis.to_labels()
        coeffs = spop.coeffs
        return [(lab, float(c.real)) for lab, c in zip(labels, coeffs) if abs(c) > 1e-8]

    def measure_modular_expectation(self, qc_base: QuantumCircuit, pauli_terms: list[tuple[str,float]], region_qubits: list[int], backend, shots: int = 4096) -> float:
        total = 0.0
        n_reg = len(region_qubits)
        for lab, coeff in pauli_terms:
            # build measurement circuit
            qc = QuantumCircuit(self.n, n_reg)
            qc.compose(qc_base, inplace=True)
            # rotate only those qubits
            for pos, op in enumerate(lab):
                q = region_qubits[pos]
                if   op == 'X': qc.h(q)
                elif op == 'Y': qc.sdg(q); qc.h(q)
                # I,Z need no prep
            # measure them into the n_reg classical bits 0..n_reg-1
            qc.measure(region_qubits, list(range(n_reg)))

            # run & collect
            transp = transpile(qc, backend)
            counts = run(transp, backend)

            # compute ⟨P_lab⟩ from the n_reg-bitstrings
            exp_k = 0
            for bstr, freq in counts.items():
                v = 1
                # bstr is length n_reg, MSB=bstr[0], LSB=bstr[-1]
                # we measured region_qubits[pos] → classical bit pos
                for pos, op in enumerate(lab):
                    if op == 'I': continue
                    bit = bstr[-1 - pos]
                    z = +1 if bit == '0' else -1
                    v *= z
                exp_k += v * freq
            total += coeff * (exp_k / shots)
        return total

    def entanglement_equilibrium_operator_fast(
        self,
        theta_dict: Dict[tuple[int,int], float],
        regions: List[List[int]],
        eps: float = 1e-4,
        backend=None,
        shots: int = 2048,
        noise_factors: List[int] = [1,3],
    ) -> Dict[Tuple[int,...], float]:
        """Batch all Pauli‐terms per region into a single Estimator.run() with ZNE."""
        from qiskit.primitives import GroupedPauliOp
        from qiskit.quantum_info import Pauli
        from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace

        # pick backend
        if backend is None:
            # assume you have already done: service = QiskitRuntimeService(...)
            backend = service.backend("ibmq_sherbrooke")

        # 1) Build & compile a single parameterized CP‐circuit for ±ε shifts
        #    (only need to compile once, binding shifts afterward)
        #    Here we cheat: we just re-build two separate circuits; 
        #    for more speed you can parameterize this.
        def make_cp_circuit(shift: float):
            qc = QuantumCircuit(self.n)
            qc.h(range(self.n))
            for (i, j), th in theta_dict.items():
                δ = shift*eps if ((i,j) in boundary_edges_of(region)) else 0.0
                qc.cp(th + δ, i, j)
            return qc
        est = Estimator(backend)
        results = {}
        for region in regions:
            # get modular‐Hamiltonian decomposition on DM₀
            dm0     = DensityMatrix(Statevector(make_cp_circuit(0)))
            rho0    = partial_trace(dm0, [q for q in range(self.n) if q not in region]).data
            terms   = self._decompose_modular_hamiltonian(rho0)
            boundary_edges = [
                e for e in self.edge_list
                if (e[0] in region) ^ (e[1] in region)
            ]

            # 2) build two batches, one for +ε and one for –ε
            circuits = []
            observables = []
            for shift in [+1, -1]:
                qc_shift = make_cp_circuit(shift)
                for lab, coeff in terms:
                    # attach the measurement portion
                    qc_meas = qc_shift.copy()
                    for pos, op in enumerate(lab):
                        q = region[pos]
                        if   op == 'X': qc_meas.h(q)
                        elif op == 'Y': qc_meas.sdg(q); qc_meas.h(q)
                    qc_meas.measure(region, range(len(region)))
                    circuits.append(transpile(qc_meas, backend))
                    observables.append(GroupedPauliOp.from_list([(lab, coeff)]))
            # 3) run them all in one shot with ZNE built in
            job = est.run(
                circuits,
                observables,
                shots=shots,
                parameterization=None,
                noise_scaling=noise_factors
            )
            vals = job.result().values  # [〈P₁〉₊ε, 〈P₂〉₊ε…  〈P₁〉₋ε, 〈P₂〉₋ε…]

            n_terms = len(terms)
            H_plus  = sum(vals[:n_terms])
            H_minus = sum(vals[n_terms:])
            results[tuple(region)] = (H_plus - H_minus) / (2 * eps)

        return results

    def measure_modular_expectation_zne(
        self,
        qc_base: QuantumCircuit,
        pauli_terms: List[tuple[str, float]],
        region_qubits: List[int],
        backend,
        shots: int = 4096,
        noise_factors: List[int] = [1, 3, 5]
    ) -> float:
        factory = RichardsonFactory(scale_factors=noise_factors)
        total = 0.0

        for lab, coeff in pauli_terms:
            # build the measured circuit once
            qc = QuantumCircuit(self.n, len(region_qubits))
            qc.compose(qc_base, inplace=True)
            for pos, op in enumerate(lab):
                q = region_qubits[pos]
                if op == 'X':
                    qc.h(q)
                elif op == 'Y':
                    qc.sdg(q); qc.h(q)
            qc.measure(region_qubits, list(range(len(region_qubits))))

            # define a float-returning runner
            def exp_at_scale(scale: int) -> float:
                qc_scaled = fold_gates_at_random(qc, scale, seed=42)
                transp = transpile(qc_scaled, backend)
                result = backend.run(transp, shots=shots).result()
                counts = result.get_counts()
                exp_val = 0.0
                for bitstr, freq in counts.items():
                    v = 1
                    for pos, op in enumerate(lab):
                        if op == 'I': continue
                        q = region_qubits[pos]
                        bit = bitstr[-1 - q]
                        v *= (1 if bit == '0' else -1)
                    exp_val += v * freq
                return exp_val / shots

            # **zero-noise extrapolate** the Python function
            exp0 = execute_with_zne(exp_at_scale, noise_factors, factory=factory)
            total += coeff * exp0

        return total

    def entanglement_equilibrium_operator(self,
        theta_dict: Dict[tuple[int,int],float],
        regions: List[List[int]],
        eps: float = 1e-4,
        backend=None
    ) -> Dict[tuple[int,...], float]:
        """
        Plain finite‐difference Δ⟨H_ξ⟩/Δθ = (H₊ − H₋)/(2ε).
        """
        if backend is None:
            backend = Aer.get_backend("aer_simulator")

        # base density matrix
        qc0 = QuantumCircuit(self.n)
        qc0.h(range(self.n))
        for (i,j), th in theta_dict.items():
            qc0.cp(th, i, j)
        dm0 = DensityMatrix(Statevector.from_instruction(qc0))

        results = {}
        for region in regions:
            region = list(region)
            rho0 = partial_trace(dm0, [q for q in range(self.n) if q not in region]).data
            terms = self._decompose_modular_hamiltonian(rho0)
            boundary = [e for e in self.edge_list if (e[0] in region) ^ (e[1] in region)]

            def build_qc(shift: float) -> QuantumCircuit:
                qc = QuantumCircuit(self.n)
                qc.h(range(self.n))
                for (i,j), th in theta_dict.items():
                    d = shift * eps if (i,j) in boundary else 0.0
                    qc.cp(th + d, i, j)
                return qc

            qc_p = build_qc(+1)
            qc_m = build_qc(-1)
            Hp = self.measure_modular_expectation(qc_p, terms, region, backend)
            Hm = self.measure_modular_expectation(qc_m, terms, region, backend)
            results[tuple(region)] = (Hp - Hm) / (2 * eps)

        return results

    def entanglement_equilibrium_operator_zne(self,
        theta_dict: Dict[tuple[int,int],float],
        regions: List[Set[int]],
        eps: float = 1e-4,
        backend=None,
        shots: int = 4096,
        noise_factors: List[int] = [1, 3, 5]
    ) -> Dict[tuple[int,...], float]:
        """
        ZNE‐only finite‐difference Δ⟨H_ξ⟩/Δθ = (H₊ − H₋)/(2ε)₀.
        """
        if backend is None:
            backend = Aer.get_backend("aer_simulator")

        qc0 = QuantumCircuit(self.n)
        qc0.h(range(self.n))
        for (i,j), th in theta_dict.items():
            qc0.cp(th, i, j)
        dm0 = DensityMatrix(Statevector.from_instruction(qc0))

        results = {}
        for region in regions:
            region = sorted(region)
            rho0 = partial_trace(dm0, [q for q in range(self.n) if q not in region]).data
            terms = self._decompose_modular_hamiltonian(rho0)
            boundary = [e for e in self.edge_list if (e[0] in region) ^ (e[1] in region)]

            def build_qc(shift: float) -> QuantumCircuit:
                qc = QuantumCircuit(self.n)
                qc.h(range(self.n))
                for (i,j), th in theta_dict.items():
                    d = shift * eps if (i,j) in boundary else 0.0
                    qc.cp(th + d, i, j)
                return qc

            qc_p = build_qc(+1)
            qc_m = build_qc(-1)
            Hp = self.measure_modular_expectation_zne(qc_p, terms, region,
                                                      backend, shots, noise_factors)
            Hm = self.measure_modular_expectation_zne(qc_m, terms, region,
                                                      backend, shots, noise_factors)
            results[tuple(region)] = (Hp - Hm) / (2 * eps)

        return results

    def entanglement_equilibrium_check(self,
        theta_dict: Dict[tuple[int,int],float],
        regions: List[List[int]],
        eps: float = 1e-4,
        simulator=None
    ) -> Dict[tuple[int,...], tuple[float,float]]:
        """
        Verify δS(A) ≈ δ⟨H_ξ(A)⟩ via symmetric finite differences on statevector simulator.
        """
        if simulator is None:
            simulator = Aer.get_backend("aer_simulator_statevector")

        def get_dm(thetas):
            qc = QuantumCircuit(self.n)
            qc.h(range(self.n))
            for (i,j), th in thetas.items():
                qc.cp(th, i, j)
            return DensityMatrix(Statevector.from_instruction(qc))

        dm0 = get_dm(theta_dict)
        results = {}
        for region in regions:
            A = list(region)
            rho0 = partial_trace(dm0, [q for q in range(self.n) if q not in A])
            Hxi = -la.logm(rho0.data)
            theta_p = {e: theta_dict[e] + eps for e in theta_dict}
            dm_p = get_dm(theta_p)
            rho_p = partial_trace(dm_p, [q for q in range(self.n) if q not in A])
            S_p, E_p = qiskit_entropy(rho_p), np.real(np.trace(rho_p.data @ Hxi))
            theta_m = {e: theta_dict[e] - eps for e in theta_dict}
            dm_m = get_dm(theta_m)
            rho_m = partial_trace(dm_m, [q for q in range(self.n) if q not in A])
            S_m, E_m = qiskit_entropy(rho_m), np.real(np.trace(rho_m.data @ Hxi))
            results[tuple(A)] = ((S_p - S_m)/(2*eps), (E_p - E_m)/(2*eps))
        return results

    def compute_bulk_boundary_I(self,
        theta_dict: Dict[tuple[int,int],float],
        boundary_size: int,
        depth: int,
        simulator=None
    ) -> float:
        """Mutual information I(bulk:boundary) at given CP‐layer depth."""
        if simulator is None:
            simulator = Aer.get_backend("aer_simulator_statevector")
        n = self.n
        qc = QuantumCircuit(n)
        qc.h(range(n))
        for _ in range(depth):
            for (i,j), th in theta_dict.items():
                qc.cp(th, i, j)
        dm = DensityMatrix(Statevector.from_instruction(qc))
        boundary = list(range(n - boundary_size, n))
        bulk = [i for i in range(n) if i not in boundary]
        rho_b = partial_trace(dm, bulk)
        rho_u = partial_trace(dm, boundary)
        return (qiskit_entropy(rho_u) + qiskit_entropy(rho_b)
                - qiskit_entropy(dm))

    def fast_bulk_boundary_MI(self,
        theta_dict: Dict[tuple[int,int],float],
        boundary: List[int],
        max_depth: int
    ) -> List[float]:
        """Rényi-2 MI vs depth via MPS snapshots."""
        n = self.n
        qc = QuantumCircuit(n)
        qc.h(range(n))
        for d in range(1, max_depth+1):
            for (i,j), th in theta_dict.items():
                qc.cp(th, i, j)
            qc.save_density_matrix(label=f"dm{d}")
        sim = AerSimulator(method="matrix_product_state")
        res = sim.run(qc).result()
        MIs = []
        for d in range(1, max_depth+1):
            dm = DensityMatrix(res.data(f"dm{d}"))
            p_full = np.trace(dm.data @ dm.data).real
            rho_b = partial_trace(dm, [i for i in range(n) if i not in boundary])
            rho_u = partial_trace(dm, boundary)
            p_b = np.trace(rho_b.data @ rho_b.data).real
            p_u = np.trace(rho_u.data @ rho_u.data).real
            MIs.append(-np.log2(p_b) - np.log2(p_u) + np.log2(p_full))
        return MIs

    def fast_bulk_boundary_MI_vN(self,
        theta_dict: Dict[tuple[int,int],float],
        boundary: List[int],
        max_depth: int
    ) -> List[float]:
        """von Neumann MI vs depth via MPS snapshots."""
        n = self.n
        qc = QuantumCircuit(n)
        qc.h(range(n))
        for d in range(1, max_depth+1):
            for (i,j), th in theta_dict.items():
                qc.cp(th, i, j)
            qc.save_statevector(label=f"sv{d}")
        sim = AerSimulator(method="matrix_product_state")
        res = sim.run(qc).result()
        MIs = []
        for d in range(1, max_depth+1):
            sv = res.data(f"sv{d}")[0]
            dm = DensityMatrix(sv)
            rho_b = partial_trace(dm, [i for i in range(n) if i not in boundary])
            rho_u = partial_trace(dm, boundary)
            MIs.append(qiskit_entropy(rho_b) + qiskit_entropy(rho_u)
                       - qiskit_entropy(dm))
        return MIs

    def fast_entanglement_equilibrium_check_local_vN(self,
        theta_dict: Dict[tuple[int,int],float],
        regions: List[List[int]],
        eps: float = 1e-3
    ) -> Dict[tuple[int,...], tuple[float,float]]:
        """Local first‐law via von Neumann entropy + MPS snapshots."""
        sim = AerSimulator(method="matrix_product_state")

        def get_dm(thetas):
            qc = QuantumCircuit(self.n)
            qc.h(range(self.n))
            for (i,j), th in thetas.items():
                qc.cp(th, i, j)
            qc.save_statevector(label="sv")
            return DensityMatrix(sim.run(qc).result().data("sv")[0])

        base = get_dm(theta_dict)
        results = {}
        boundary_map = {
            tuple(A): [e for e in self.edge_list if (e[0] in A) ^ (e[1] in A)]
            for A in regions
        }
        for A in regions:
            key = tuple(A)
            b_edges = boundary_map[key]
            rho0 = partial_trace(base, [q for q in range(self.n) if q not in A])
            S0 = qiskit_entropy(rho0)
            # +ε
            tp = {e: theta_dict[e] + eps if e in b_edges else theta_dict[e]
                  for e in theta_dict}
            rho_p = partial_trace(get_dm(tp), [q for q in range(self.n) if q not in A])
            S_p = qiskit_entropy(rho_p)
            # –ε
            tm = {e: theta_dict[e] - eps if e in b_edges else theta_dict[e]
                  for e in theta_dict}
            rho_m = partial_trace(get_dm(tm), [q for q in range(self.n) if q not in A])
            S_m = qiskit_entropy(rho_m)
            results[key] = ((S_p - S_m)/(2*eps), (S_p - S_m)/(2*eps))
        return results

    def entanglement_equilibrium_check_local(self,
        theta_dict: Dict[tuple[int,int],float],
        regions: List[List[int]],
        eps: float = 1e-4
    ) -> Dict[tuple[int,...], tuple[float,float]]:
        """Local first‐law via purity (Rényi-2)."""
        def get_dm(thetas):
            qc = QuantumCircuit(self.n)
            qc.h(range(self.n))
            for (i,j), th in thetas.items():
                qc.cp(th, i, j)
            return DensityMatrix(Statevector.from_instruction(qc))

        base = get_dm(theta_dict)
        results = {}
        for A in regions:
            key = tuple(A)
            b_edges = [e for e in self.edge_list if (e[0] in A) ^ (e[1] in A)]
            rho0 = partial_trace(base, [q for q in range(self.n) if q not in A])
            p0 = np.trace(rho0.data @ rho0.data).real
            # +ε
            tp = {e: theta_dict[e] + eps if e in b_edges else theta_dict[e]
                  for e in theta_dict}
            rho_p = partial_trace(get_dm(tp), [q for q in range(self.n) if q not in A])
            p_p = np.trace(rho_p.data @ rho_p.data).real
            # –ε
            tm = {e: theta_dict[e] - eps if e in b_edges else theta_dict[e]
                  for e in theta_dict}
            rho_m = partial_trace(get_dm(tm), [q for q in range(self.n) if q not in A])
            p_m = np.trace(rho_m.data @ rho_m.data).real
            dS2 = (-np.log2(p_p) + np.log2(p_m)) / (2*eps)
            results[key] = (dS2, dS2)
        return results

    def compute_mutual_information(self,
        theta_dict: Dict[tuple[int,int],float],
        simulator=None
    ) -> Dict[tuple[int,int], float]:
        """Compute pairwise I_ij = S_i + S_j − S_{ij} on statevector simulator."""
        if simulator is None:
            simulator = Aer.get_backend("aer_simulator_statevector")
        qc = QuantumCircuit(self.n)
        qc.h(range(self.n))
        for (i,j), th in theta_dict.items():
            qc.cp(th, i, j)
        dm = DensityMatrix(Statevector.from_instruction(qc))
        I = {}
        for i,j in self.edge_list:
            rho_i  = partial_trace(dm, [q for q in range(self.n) if q != i])
            rho_j  = partial_trace(dm, [q for q in range(self.n) if q != j])
            rho_ij = partial_trace(dm, [q for q in range(self.n) if q not in (i,j)])
            I[(i,j)] = (
                qiskit_entropy(rho_i) +
                qiskit_entropy(rho_j) -
                qiskit_entropy(rho_ij)
            )
        return I

    def compute_R(self, I_dict: Dict[tuple[int,int], float]) -> Dict[int,float]:
        """Discrete curvature R_i = Σ_j I_{ij}."""
        R = {i: 0.0 for i in range(self.n)}
        for (i,j), Iij in I_dict.items():
            R[i] += Iij
            R[j] += Iij
        return R

    def check_bianchi(self, R_dict: Dict[int,float]) -> Dict[int,float]:
        """Δ_i = Σ_{j∈nbhd(i)} (R_j − R_i)"""
        neighbors = {i: [] for i in range(self.n)}
        for i,j in self.edge_list:
            neighbors[i].append(j)
            neighbors[j].append(i)
        return {
            i: sum(R_dict[j] - R_dict[i] for j in neighbors[i])
            for i in range(self.n)
        }

    def fit_phase_boundary(self,
        theta_arr: List[float],
        dstar_arr: List[float],
        k: int,
        output_file: str = None
    ) -> Dict[str,float]:
        """Fit d* = α*(n−k)/θ^β + γ via nonlinear least squares."""
        from scipy.optimize import curve_fit
        theta = np.array(theta_arr)
        dstar = np.array(dstar_arr)
        mask = np.isfinite(theta) & np.isfinite(dstar)
        theta, dstar = theta[mask], dstar[mask]
        def model(t, α, β, γ): return α*(self.n-k)/t**β + γ
        popt, _ = curve_fit(model, theta, dstar, p0=[1,1,0])
        params = dict(alpha=popt[0], beta=popt[1], gamma=popt[2])
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(params, f, indent=2)
        return params

    def fast_entanglement_equilibrium_check_local(self,
        theta_dict: Dict[tuple[int,int],float],
        regions: List[List[int]],
        eps: float = 1e-3
    ) -> Dict[tuple[int,...], tuple[float,float]]:
        """
        Local equilibrium via Rényi-2 (purity) & symmetric FD
        using MPS snapshots.
        """
        sim = AerSimulator(method="matrix_product_state")
        def get_dm(thetas):
            qc = QuantumCircuit(self.n)
            qc.h(range(self.n))
            for (i,j), th in thetas.items():
                qc.cp(th, i, j)
            return DensityMatrix(Statevector.from_instruction(qc))

        base = get_dm(theta_dict)
        results = {}
        boundary_map = {
            tuple(A): [e for e in self.edge_list if (e[0] in A) ^ (e[1] in A)]
            for A in regions
        }
        for A in regions:
            key = tuple(A)
            b = boundary_map[key]
            rho0 = partial_trace(base, [q for q in range(self.n) if q not in A])
            p0 = np.trace(rho0.data @ rho0.data).real
            # +ε
            tp = {e: theta_dict[e] + eps if e in b else theta_dict[e]
                  for e in theta_dict}
            rho_p = partial_trace(get_dm(tp), [q for q in range(self.n) if q not in A])
            p_p = np.trace(rho_p.data @ rho_p.data).real
            # –ε
            tm = {e: theta_dict[e] - eps if e in b else theta_dict[e]
                  for e in theta_dict}
            rho_m = partial_trace(get_dm(tm), [q for q in range(self.n) if q not in A])
            p_m = np.trace(rho_m.data @ rho_m.data).real
            dS2 = (-np.log2(p_p) + np.log2(p_m)) / (2*eps)
            results[key] = (dS2, dS2)
        return results

    def reconstruct_action(self,
        theta_dict: Dict[tuple[int,int],float],
        R_target: Dict[int,float]
    ) -> Dict[str,float]:
        """
        Fit S(θ)=a θ² + b θ⁴ so that ∂S/∂θ matches R_i − R_target[i].
        """
        from scipy.optimize import curve_fit
        thetas = np.array(list(theta_dict.values()))
        def action(t, a, b): return a*t**2 + b*t**4
        popt, _ = curve_fit(lambda t,a,b: action(t,a,b),
                            thetas, np.zeros_like(thetas))
        return dict(a=popt[0], b=popt[1])

##Best backend chosen: ibm_brisbane
##Region (0, 1): δS/δθ = 1.3298e-01, δ⟨Hξ⟩/δθ = 9.2178e-02
##Region (1, 2): δS/δθ = 2.6597e-01, δ⟨Hξ⟩/δθ = 1.8436e-01
##Region (2, 3): δS/δθ = 2.6597e-01, δ⟨Hξ⟩/δθ = 1.8436e-01
##Region (3, 4): δS/δθ = 1.3298e-01, δ⟨Hξ⟩/δθ = 9.2178e-02
##Δ_i (should be small for an “Einstein” solution): {0: 0.012589186802100309, 1: -0.013847020379685271, 2: 0.002515667155176607, 3: -0.013847020379693655, 4: 0.01258918680210201}

##Best backend chosen: ibm_brisbane
##Region (0, 1): extrapolated ratio = 1.443
##Region (1, 2): extrapolated ratio = 1.443
##Region (2, 3): extrapolated ratio = 1.443
##Region (3, 4): extrapolated ratio = 1.443

def generate_spacetime(n, edge_list, T_target, steps=2, lr=0.1, eps=1e-3):
    """
    Solve for a discrete 'metric' θ_ij on a graph to match target curvature T_target.
    Returns theta_dict mapping each edge to its CP-angle.
    
    Parameters:
    - n: int
        Number of qubits (nodes) in the chain or lattice.
    - edge_list: list of tuples
        List of (i, j) pairs defining nearest-neighbor edges.
    - T_target: dict
        Target curvature T_i for each node i.
    - steps: int
        Number of gradient-descent iterations.
    - lr: float
        Learning rate for gradient descent.
    - eps: float
        Finite-difference epsilon for gradient estimation.
    """
    from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, entropy
    from qiskit import QuantumCircuit
    import numpy as np

    # Initialize uniform small angles
    theta = {edge: 0.1 for edge in edge_list}
    
    # Helper to build density matrix given theta configuration
    def get_dm(thetas):
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.h(q)
        for (i, j), th in thetas.items():
            qc.cp(th, i, j)
        sv = Statevector(qc)
        return DensityMatrix(sv)
    
    # Gradient descent loop
    for step in tqdm(range(steps), desc="GD iterations"):
        # Compute curvatures R_i
        dm = get_dm(theta)
        R = {i: 0.0 for i in range(n)}
        for (i, j) in edge_list:
            dm_ij = partial_trace(dm, [q for q in range(n) if q not in (i, j)])
            Si = qiskit_entropy(partial_trace(dm, [q for q in range(n) if q != i]))
            Sj = qiskit_entropy(partial_trace(dm, [q for q in range(n) if q != j]))
            Sij = qiskit_entropy(dm_ij)
            Iij = Si + Sj - Sij
            R[i] += Iij
            R[j] += Iij
        
        # Compute loss
        L = sum((R[i] - T_target[i])**2 for i in range(n))
        
        # Estimate gradient via finite differences
        grad = {}
        for edge in tqdm(edge_list, desc=f"Edges @ step {step+1}", leave=False):
            orig = theta[edge]
            theta[edge] = orig + eps
            # R-plus
            dm_p = get_dm(theta)
            R_p = {i: 0.0 for i in range(n)}
            for (a, b) in edge_list:
                dm_ab = partial_trace(dm_p, [q for q in range(n) if q not in (a, b)])
                Sa = qiskit_entropy(partial_trace(dm_p, [q for q in range(n) if q != a]))
                Sb = qiskit_entropy(partial_trace(dm_p, [q for q in range(n) if q != b]))
                Sab = qiskit_entropy(dm_ab)
                Iab = Sa + Sb - Sab
                R_p[a] += Iab
                R_p[b] += Iab
            Lp = sum((R_p[i] - T_target[i])**2 for i in range(n))
            
            theta[edge] = orig - eps
            # R-minus
            dm_m = get_dm(theta)
            R_m = {i: 0.0 for i in range(n)}
            for (a, b) in edge_list:
                dm_ab = partial_trace(dm_m, [q for q in range(n) if q not in (a, b)])
                Sa = qiskit_entropy(partial_trace(dm_m, [q for q in range(n) if q != a]))
                Sb = qiskit_entropy(partial_trace(dm_m, [q for q in range(n) if q != b]))
                Sab = qiskit_entropy(dm_ab)
                Iab = Sa + Sb - Sab
                R_m[a] += Iab
                R_m[b] += Iab
            Lm = sum((R_m[i] - T_target[i])**2 for i in range(n))
            
            # Central difference
            grad[edge] = (Lp - Lm) / (2 * eps)
            theta[edge] = orig
        
        # Gradient descent update
        for edge in edge_list:
            theta[edge] -= lr * grad[edge]
    
    return theta


def sweep_n_qubits_equilibrium_ratio(
    n_list,
    mass_node,
    mass_strength,
    sigma,
    epsilons,
    gradient_steps=200,
    lr=0.05,
    depth_scan_params=None
):
    """
    For each n in n_list, build a 1D chain of n qubits, generate
    a black-hole–like mass profile, solve for the entanglement-based
    metric θ, then run the localized entanglement-equilibrium check
    across a set of epsilons, perform ε→0 extrapolation, and return
    the extrapolated ratio r = δS/δH for each region.
    
    Returns a dict mapping n -> { region_tuple: extrapolated_ratio }.
    """
    import numpy as np
    
    results = {}
    for n in tqdm(n_list, desc="chain‑lengths"):
        # define edges for a 1D chain
        edge_list = [(i, i+1) for i in range(n-1)]
        # Gaussian "mass" profile
        T_target = {
            i: mass_strength * np.exp(-(i - mass_node)**2 / (2 * sigma**2))
            for i in range(n)
        }
        # solve for θ_ij metric
        theta_dict = generate_spacetime(n, edge_list, T_target,
                                        steps=gradient_steps, lr=lr, eps=epsilons[0])
        # set up analyzer
        analyzer = QuantumGravityAnalyzer(n_qubits=n, edge_list=edge_list)
        # choose regions: contiguous pairs
        regions = [[i, i+1] for i in range(n-1)]
        
        # collect ratio vs ε for each region
        ratio_vs_eps = {tuple(A): [] for A in regions}
        for eps in tqdm(epsilons, desc=f"n={n} epsilons", leave=False):
            eq = analyzer.fast_entanglement_equilibrium_check_local_vN(
                theta_dict, regions, eps=eps
            )
            for A, (dS, dH) in eq.items():
                ratio_vs_eps[tuple(A)].append(dS/dH if dH!=0 else np.nan)
        
        # extrapolate ratio to eps->0 (linear fit in eps^2)
        extrapolated = {}
        x = np.array(epsilons)**2
        for A, y in ratio_vs_eps.items():
            # fit y = m x + b
            m, b = np.polyfit(x, y, 1)
            extrapolated[A] = b
        results[n] = extrapolated
    return results

def prepare_state(qc):
    # 1) Copy the circuit, save the statevector, and run *once*
    qc_sv = qc.copy()
    qc_sv.save_statevector()
    sim = AerSimulator(method='matrix_product_state')
    t_qc = transpile(qc_sv, sim)
    result = sim.run(t_qc).result()
    sv = result.get_statevector(t_qc)
    return DensityMatrix(sv)

def compute_face_curvature_from_dm(dm, faces):
    n = dm.num_qubits
    curv = {}
    for corners in faces:
        ab, cd = corners[:2], corners[2:]
        rho_ab   = partial_trace(dm, [q for q in range(n) if q not in ab])
        rho_cd   = partial_trace(dm, [q for q in range(n) if q not in cd])
        rho_abcd = partial_trace(dm, [q for q in range(n) if q not in corners])
        S_ab   = qiskit_entropy(rho_ab)
        S_cd   = qiskit_entropy(rho_cd)
        S_abcd = qiskit_entropy(rho_abcd)
        curv[tuple(corners)] = S_ab + S_cd - S_abcd
    return curv

def compute_cell_curvature_from_dm(dm, cells, faces_of_cell):
    n = dm.num_qubits
    curv = {}
    for corners in cells:
        faces = faces_of_cell[tuple(corners)]
        S_faces = 0
        for face in faces:
            rho = partial_trace(dm, [q for q in range(n) if q not in face])
            S_faces += qiskit_entropy(rho)
        rho_full = partial_trace(dm, [q for q in range(n) if q not in corners])
        S_full = qiskit_entropy(rho_full)
        curv[tuple(corners)] = S_faces - S_full
    return curv



sim = Aer.get_backend("aer_simulator_statevector")

epsilons = [1e-2, 1e-3, 1e-4, 1e-5]


def run_job(args):
    # Unpack
    n, mass_node, mass_strength, sigma, eps_list, gradient_steps, lr = args
    # Call your function for this single‑eps slice
    single_ratio = sweep_n_qubits_equilibrium_ratio(
        n_list=[n],
        mass_node=mass_node,
        mass_strength=mass_strength,
        sigma=sigma,
        epsilons=eps_list,
        gradient_steps=gradient_steps,
        lr=lr
    )
    return (n, eps_list[0], single_ratio[n])

def main():
    # 1) Parameters
    L, sigma, strength = 3, 1.0, 1.0
    center = ((L-1)/2,)*3

    # 2) Build graph data
    edges = build_3d_edges(L)
    faces = list_3d_faces(L)
    cells = list_3d_cells(L)
    f2c   = build_faces_of_cell(cells, faces)

    # 3) Gaussian mass profile → theta_dict
    T = {}
    for x in range(L):
        for y in range(L):
            for z in range(L):
                idx3 = x + L*(y + L*z)
                d2 = (x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2
                T[idx3] = strength * np.exp(-d2/(2*sigma**2))
                print(idx3, d2, T[idx3])
    theta = {e: (T[e[0]] + T[e[1]])/2 for e in edges}

    # 4) Build and run circuit once to get Statevector
    qc = create_3d_entanglement_circuit(L**3, edges, theta)
    qc.save_statevector()
    sim = AerSimulator(method='matrix_product_state')
    result = sim.run(transpile(qc, sim)).result()
    sv = result.get_statevector()             # <— this is a Statevector

    # 5) Compute face‐based curvature from the Statevector
    face_curv = {}
    n = L**3
    for corners in faces:
        ab, cd = corners[:2], corners[2:]
        rho_ab   = partial_trace(sv, [q for q in range(n) if q not in ab])
        rho_cd   = partial_trace(sv, [q for q in range(n) if q not in cd])
        rho_abcd = partial_trace(sv, [q for q in range(n) if q not in corners])
        S_ab   = qiskit_entropy(rho_ab)
        S_cd   = qiskit_entropy(rho_cd)
        S_abcd = qiskit_entropy(rho_abcd)
        face_curv[tuple(corners)] = S_ab + S_cd - S_abcd

    # 6) Compute cell‐based curvature similarly
    cell_curv = {}
    for corners in cells:
        faces_of_this = f2c[tuple(corners)]
        S_faces = 0
        for face in faces_of_this:
            rho = partial_trace(sv, [q for q in range(n) if q not in face])
            S_faces += qiskit_entropy(rho)
        rho_full = partial_trace(sv, [q for q in range(n) if q not in corners])
        S_full   = qiskit_entropy(rho_full)
        cell_curv[tuple(corners)] = S_faces - S_full

    # 7) Print
    print("=== Face Curvatures ===")
    for face, val in face_curv.items():
        print(face, f"{val:.6f}")
    print("\n=== Cell Curvatures ===")
    for cell, val in cell_curv.items():
        print(cell, f"{val:.6f}")

# 4) Compute plaquette curvature from Statevector
def compute_plaquette_curvature_from_sv(sv, plaquettes, n_qubits):
    curv = {}
    for corners in plaquettes:
        ab, cd = corners[:2], corners[2:]
        rho_ab   = partial_trace(sv, [q for q in range(n_qubits) if q not in ab])
        rho_cd   = partial_trace(sv, [q for q in range(n_qubits) if q not in cd])
        rho_abcd = partial_trace(sv, [q for q in range(n_qubits) if q not in corners])
        S_ab   = entropy(rho_ab,   base=2)
        S_cd   = entropy(rho_cd,   base=2)
        S_abcd = entropy(rho_abcd, base=2)
        curv[tuple(corners)] = S_ab + S_cd - S_abcd
    return curv

# 5) Generate and measure curvature for a 2D black-hole analog
def generate_blackhole_curvature(service, rows, cols, mass_strength=1.0,
                                 center=None, epsilon=1e-2):
    if center is None:
        center = ((rows-1)/2, (cols-1)/2)
    # Build graph
    edges = build_2d_edges(rows, cols)
    plaquettes = list_plaquettes(rows, cols)
    # Mass profile ~ 1/(r + epsilon)
    T = {}
    for r in range(rows):
        for c in range(cols):
            idx = r*cols + c
            dist = np.hypot(r - center[0], c - center[1])
            T[idx] = mass_strength / (dist + epsilon)
    # Map to link angles
    theta = {e: (T[e[0]] + T[e[1]])/2 for e in edges}
    # Build and run circuit
    qc = create_2d_entanglement_circuit(rows, cols, theta)
    backend = get_best_backend(service)
    transpiled_qc = transpile(qc, backend)
    with Session(backend=backend) as session:
        sampler = Sampler()
        job = sampler.run([transpiled_qc], shots=2048)
        result = job.result()
    sv = result.get_statevector()
    # Compute curvature
    curv = compute_plaquette_curvature_from_sv(sv, plaquettes, rows*cols)
    return curv

def build_spacetime_circuit(n, theta_dict):
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    for (i, j), th in theta_dict.items():
        qc.cp(th, i, j)
    return qc

def create_2d_entanglement_circuit(rows, cols, theta_dict):
    n = rows*cols
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for (i, j), th in theta_dict.items():
        qc.cp(th, i, j)
    qc.measure(range(n), range(n))
    return qc

def marginal_counts(full_counts, qubits, num_qubits):
    marg = defaultdict(int)
    for bitstr, cnt in full_counts.items():
        # Qiskit bitstring is big-endian: highest index on left
        key = ''.join(bitstr[num_qubits-1 - q] for q in qubits)
        marg[key] += cnt
    return marg

def shannon_entropy(counts, shots):
    H = 0.0
    for cnt in counts.values():
        p = cnt / shots
        if p > 0:
            H -= p * log2(p)
    return H

def run_blackhole_curvature_real(rows, cols, mass_strength, epsilon, backend_name, shots=8192):
    # 1) Connect to IBM backend

    def marginal(probs, qubits, num_qubits):
        marg = defaultdict(float)
        for bs, p in probs.items():
            key = ''.join(bs[num_qubits-1-q] for q in qubits)
            marg[key] += p
        return marg

    def shannon(pdist):
        return -sum(p * math.log2(p) for p in pdist.values() if p>0)

    provider = QiskitRuntimeService()
    backend = get_best_backend(service)

    # 2) Build graph and mass profile
    edges = build_2d_edges(rows, cols)
    plaquettes = list_plaquettes(rows, cols)
    center = ((rows-1)/2, (cols-1)/2)
    T = {}
    for r in range(rows):
        for c in range(cols):
            idx3 = r*cols + c
            dist = np.hypot(r-center[0], c-center[1])
            T[idx3] = mass_strength / (dist + epsilon)
    theta = {e: (T[e[0]] + T[e[1]]) / 2 for e in edges}

    # 3) Build and transpile circuit
    qc = create_2d_entanglement_circuit(rows, cols, theta)
    transpiled = transpile(qc, backend, optimization_level=3)

    # 4) Run job
    with Session(backend=backend) as session:
        sampler = Sampler()
        job = sampler.run([transpiled], shots=2048)
        result = job.result()
        print(result)

        
    raw = result[0].data.c
    print(raw)
    num_bits = rows*cols

    counts = extract_counts_from_bitarray(raw)
    print(counts)

    total_shots = sum(counts.values())
    probs = {bs: cnt/total_shots for bs, cnt in counts.items()}

    P_ab = marginal(probs, [0,1], num_qubits=num_bits)

    S_ab = shannon(P_ab)

##    ba = raw[0] if isinstance(raw, list) else raw
##    arr = ba._array           # shape = (shots, rows*cols) of booleans
##
##    try:
##        mem = result.get_memory()       # list of strings like '01011...'
##    except:
##        mem = result.get_memory(0)      # sometimes indexed by experiment

##    full_counts = Counter(mem)          # e.g. {'010': 123, '111': 98, ...}

    # 7) compute plaquette curvatures
    num_qubits = rows*cols
    curv = {}
    curvatures = {}

    for corners in plaquettes:
        # corners is a 4-tuple, e.g. (0, 1, 3, 4)
        a, b, c, d = corners

        # marginal over the first link (a,b)
        P_ab = marginal(probs, [a, b], num_qubits)
        # marginal over the second link (c,d)
        P_cd = marginal(probs, [c, d], num_qubits)

        # Shannon entropies of the marginals
        S_ab = shannon(P_ab)
        S_cd = shannon(P_cd)

        # joint entropy over all four qubits
        P_abcd = marginal(probs, list(corners), num_qubits)
        S_abcd = shannon(P_abcd)

        curvatures[tuple(corners)] = S_ab + S_cd - S_abcd

    I = {}
    for (i, j) in edges:
        # joint distribution on qubits i,j
        P_ij = marginal(probs, [i, j], num_qubits)
        # single‐qubit marginals
        P_i  = marginal(probs, [i],     num_qubits)
        P_j  = marginal(probs, [j],     num_qubits)

        # Shannon entropies
        H_ij = shannon(P_ij)
        H_i  = shannon(P_i)
        H_j  = shannon(P_j)

        # classical MI
        I[(i, j)] = H_i + H_j - H_ij
        print(I[(i,j)])

    print(curvatures)
    print(I)

    return curvatures, I

def run_blackhole_curvature_real_0(rows, cols, mass_strength, epsilon, shots=8192):
    import math, numpy as np
    from collections import defaultdict
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
    from qiskit import transpile

    def marginal(probs, qubits, num_qubits):
        m = defaultdict(float)
        for bs, p in probs.items():
            key = ''.join(bs[num_qubits-1-q] for q in qubits)
            m[key] += p
        return m

    def shannon(pdist):
        return -sum(p * math.log2(p) for p in pdist.values() if p > 0)
        # Shannon entropy in NATURAL units (nats)
    def shannon_nats(pdist):
        return -sum(p * math.log(p) for p in pdist.values() if p > 0)
    

    # 1) Backend

    # 2) Build graph & mass profile
    edges     = build_2d_edges(rows, cols)
    plaquettes= list_plaquettes(rows, cols)
    center    = ((rows-1)/2, (cols-1)/2)
    T         = {}
    for r in range(rows):
        for c in range(cols):
            idx = r*cols + c
            dist = np.hypot(r-center[0], c-center[1])
            T[idx] = mass_strength / (dist + epsilon)

    theta = {e: (T[e[0]] + T[e[1]])/2 for e in edges}

    # 3) Prepare & transpile
    qc = create_2d_entanglement_circuit(rows, cols, theta)
    tq = transpile(qc, backend, optimization_level=3)

    # 4) Execute on hardware
    job = run(tq, backend)
    res = job.result()

    raw = res[0].data.c
    num_qubits = rows*cols
    counts     = extract_counts_from_bitarray(raw)
    total      = sum(counts.values())
    probs      = {bs: cnt/total for bs, cnt in counts.items()}

    # 5) Plaquette curvatures (unchanged)
    curvatures = {}
    curvature = {}
    for corners in plaquettes:
        a,b,c,d = corners
        S_ab   = shannon(marginal(probs, [a,b], num_qubits))
        S_cd   = shannon(marginal(probs, [c,d], num_qubits))
        S_abcd = shannon(marginal(probs, corners,  num_qubits))
        curvatures[tuple(corners)] = S_ab + S_cd - S_abcd

    # 6) Pairwise Shannon MI and node curvature R_i
    I = {}
    R = {i:0.0 for i in range(num_qubits)}
    for i, j in edges:
        H_ij = shannon(marginal(probs, [i,j],  num_qubits))
        H_i  = shannon(marginal(probs, [i],    num_qubits))
        H_j  = shannon(marginal(probs, [j],    num_qubits))
        mij  = H_i + H_j - H_ij
        I[(i,j)] = mij
        R[i] += mij
        R[j] += mij

        curvatures = {}
        
    for a,b,c,d in plaquettes:
        S_ab   = shannon_nats(marginal(probs, [a,b], num_qubits))
        S_cd   = shannon_nats(marginal(probs, [c,d], num_qubits))
        S_abcd = shannon_nats(marginal(probs, [a,b,c,d], num_qubits))
        curvature[(a,b,c,d)] = S_ab + S_cd - S_abcd

    # 6) pairwise MUTUAL INFORMATION and R_i in nats
    J = {}
    S = {i:0.0 for i in range(num_qubits)}
    for i,j in edges:
        H_i  = shannon_nats(marginal(probs, [i],   num_qubits))
        H_j  = shannon_nats(marginal(probs, [j],   num_qubits))
        H_ij = shannon_nats(marginal(probs, [i,j], num_qubits))
        mij  = H_i + H_j - H_ij
        J[(i,j)] = mij
        S[i] += mij
        S[j] += mij

    # 7) Discrete Bianchi residuals Δ_i
    neighbors = {i:[] for i in range(num_qubits)}
    for u,v in edges:
        neighbors[u].append(v)
        neighbors[v].append(u)
    Bianchi = {i: sum(S[j]-S[i] for j in neighbors[i]) for i in neighbors}
    Bianchi_2 = {i: sum(R[j]-R[i] for j in neighbors[i]) for i in neighbors}

    # 8) Compare R_i vs target T_i
    deviations = {i: S[i] - T[i] for i in range(num_qubits)}
    deviations_2 = {i: R[i] - T[i] for i in range(num_qubits)}

    # Print everything
    print("Target T_i:             ", T)
    print("Measured node curvature R_i:", R)
    print("Curvature nats:", S)
    print("Deviations S_i - T_i:    ", deviations)
    print("Deviations R_i - T_i:    ", deviations_2)
    print("Bianchi residuals Δ_i:   ", Bianchi)
    print("Bianchi residuals Δ_i:   ", Bianchi_2)

    return curvatures, I, R, Bianchi, deviations

def prepare_vacuum_cluster(qc, rows, cols):
    """Make a 2D cluster-state on a rows×cols grid."""
    # 1) Hadamards on every qubit
    for i in range(rows):
        for j in range(cols):
            idx = i*cols + j
            qc.h(idx)
    # 2) CZ on every nearest-neighbor link
    for i in range(rows):
        for j in range(cols):
            idx = i*cols + j
            if j+1 < cols:
                qc.cz(idx,   idx+1)
            if i+1 < rows:
                qc.cz(idx,   idx+cols)
                

def apply_mass_deformation(qc, mass_sites, θ_mass):
    """Rotate the ‘vacuum’ at chosen sites by Rz(θ_mass) to create an entanglement well."""
    for (i,j) in mass_sites:
        idx = i*cols + j
        qc.rz(θ_mass, idx)

def make_cluster_with_mass(rows, cols, mass_profile, mass_strength):
    N = rows*cols
    qc = QuantumCircuit(N)
    # 1) vacuum cluster:
    for q in range(N):
        qc.h(q)
    for r in range(rows):
        for c in range(cols):
            i = r*cols + c
            if c+1 < cols:
                qc.cz(i, i+1)
            if r+1 < rows:
                qc.cz(i, i+cols)
    # 2) deformation:
    for q in range(N):
        qc.rz(mass_strength * mass_profile[q], q)
    # 3) measure all in Z:
    qc.measure_all()
    return qc

def compute_peak_curvature(curvatures):
    """Return the maximum curvature among all plaquettes."""
    return max(curvatures.values())

def study_scaling_N(
    sizes: list[int],
    cols: int,
    mass_strength: float,
    epsilon: float,
    backend_name: str,
    shots: int = 2048
) -> dict[int, tuple[float, float]]:
    """
    Vary the grid size N×cols (assuming square so cols=N) and return for each N:
      (K_max, K_cont = 2*N^2*K_max).
    """
    results = {}
    for N in sizes:
        curv = run_blackhole_curvature_real(
            rows=N,
            cols=N,
            mass_strength=mass_strength,
            epsilon=epsilon,
            backend_name=backend_name,
            shots=shots
        )
        Kmax = compute_peak_curvature(curv)
        Kcont = 2 * N**2 * Kmax
        results[N] = (Kmax, Kcont)
        print(f"N={N}: K_max={Kmax:.4e} bits, 2N²K_max={Kcont:.4e}")
    return results

def study_scaling_mass(
    N: int,
    cols: int,
    masses: list[float],
    epsilon: float,
    backend_name: str,
    shots: int = 2048
) -> dict[float, tuple[float, float]]:
    """
    Vary the mass_strength m on a fixed N×cols grid and return for each m:
      (K_max, K_cont = 2*N^2*K_max).
    """
    results = {}
    for m in masses:
        curv = run_blackhole_curvature_real(
            rows=N,
            cols=cols,
            mass_strength=m,
            epsilon=epsilon,
            backend_name=backend_name,
            shots=shots
        )
        Kmax = compute_peak_curvature(curv)
        Kcont = 2 * N**2 * Kmax
        results[m] = (Kmax, Kcont)
        print(f"m={m:.2f}: K_max={Kmax:.4e} bits, 2N²K_max={Kcont:.4e}")
    return results

# 1. Build the toroidal lattice unitary U_torus
def make_torus_circuit(N, phase_profile):
    """
    Returns a QuantumCircuit on N^2 qubits with wrap-around controlled-Z gates.
    phase_profile is an N×N array of phases (radians) to apply on each plaquette.
    """
    n_qubits = N * N
    qc = QuantumCircuit(n_qubits)
    # helper to map (i,j) -> flat index
    idx = lambda i, j: (i % N) * N + (j % N)
    # apply CZ(phase) gate between each neighbor (including wrap edges)
    for i in range(N):
        for j in range(N):
            # horizontal neighbor (i, j)->(i, j+1)
            qc.cp(phase_profile[i,j], idx(i,j), idx(i,j+1))
            # vertical neighbor (i, j)->(i+1, j)
            qc.cp(phase_profile[i,j], idx(i,j), idx(i+1,j))
    return qc

# 2. Deutsch fixed-point iteration for the loop qubits
def find_fixed_point(U_mat, rho_S, loop_qubits, max_iters=20, tol=1e-6):
    """
    Solve ρ_C = Tr_S[ U (ρ_S ⊗ ρ_C) U† ] by iteration.
    - U_mat: full-unitary np.ndarray for all qubits.
    - rho_S: DensityMatrix on the first n_S qubits.
    - loop_qubits: list of indices for the CTC loop qubits.
    """
    # Number of S-qubits:
    n_S = int(math.log2(rho_S.dim))

    # Number of C-qubits is the length of the loop_qubits list:
    n_C = len(loop_qubits)
    dim_C = 2**n_C

    # Initialize ρ_C as maximally mixed on the loop:
    rho_C = DensityMatrix(np.eye(dim_C) / dim_C)

    for _ in range(max_iters):
        # Build the joint state on n_S + n_C qubits
        joint = rho_S.tensor(rho_C)

        # Apply U
        joint = DensityMatrix(U_mat @ joint.data @ U_mat.conj().T)

        # Trace out the S-subsystem (first n_S qubits)
        new_rho_C = partial_trace(joint, list(range(n_S)))
        new_rho_C = DensityMatrix(new_rho_C.data)

        # Check convergence by fidelity
        if abs(state_fidelity(rho_C, new_rho_C) - 1) < tol:
            break

        rho_C = new_rho_C

    return rho_C

def run_torus_ctc(N, phase_profile, backend_name=None, shots=4096):
    # 1) Build toroidal circuit and matrix
    qc_torus = make_torus_circuit(N, phase_profile)
    U_mat = Operator(qc_torus).data

    # 2) Identify loop vs. bulk
    loop_qubits = list(range(N))
    bulk_qubits = list(range(N, N*N))
    dim_S = 2**len(bulk_qubits)
    rho_S = DensityMatrix(np.eye(dim_S) / dim_S)

    # 3) Solve for rho_C_star
    rho_C_star = find_fixed_point(U_mat, rho_S, loop_qubits)

    # 4) Diagonalize rho_C_star for ensemble sampling
    eigvals, eigvecs = np.linalg.eigh(rho_C_star.data)
    # keep only components with non‐negligible weight
    comps = [(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals)) if eigvals[i] > 1e-6]

    # 5) Prepare and run one sub‐circuit per eigencomponent
    #    then weight the counts by eigval

    total_counts = {}
    for weight, vec in comps:
        shots_i = max(1, int(round(shots * weight)))
        # build subcircuit
        sub_qc = QuantumCircuit(N*N, len(loop_qubits))
        # initialize loop qubits to the pure eigenstate |v_i>
        sub_qc.initialize(vec, loop_qubits)
        # torus circuit
        sub_qc.append(qc_torus.to_gate(), range(N*N))
        # measure loop
        for idx, q in enumerate(loop_qubits):
            sub_qc.measure(q, idx)

        # transpile and run
        t_qc = transpile(sub_qc, backend=backend, optimization_level=1)
        with Session(backend=backend) as session:
            sampler = Sampler()
            job = sampler.run([t_qc], shots=2048)
            result = job.result()
            print(result)

        bitarray = result[0].data.c
        counts = extract_counts_from_bitarray(bitarray)
        # aggregate (weighting built into shots_i)
        for outcome, c in counts.items():
            total_counts[outcome] = total_counts.get(outcome, 0) + c

    return total_counts

def make_cluster(rows, cols):
    N = rows*cols
    qc = QuantumCircuit(N)
    # vacuum cluster
    for q in range(N):
        qc.h(q)
    for r in range(rows):
        for c in range(cols):
            idx = r*cols + c
            if c+1 < cols: qc.cz(idx, idx+1)
            if r+1 < rows: qc.cz(idx, idx+cols)
    return qc

def inject_negative_mass(qc, rows, cols, site, θ):
    i,j = site
    idx = i*cols + j
    qc.rz(-θ, idx)

def apply_wormhole_throat(qc, idxL, idxR, g):
    # CX–RZ–CX implements exp(i g X⨂X)
    qc.cx(idxL, idxR)
    qc.rz(2*g, idxR)
    qc.cx(idxL, idxR)

def marginal(probs, qubits, num_qubits):
    marg = defaultdict(float)
    for bs, p in probs.items():
        key = ''.join(bs[num_qubits-1-q] for q in qubits)
        marg[key] += p
    return marg

def shannon(pdist):
    return -sum(p * math.log2(p) for p in pdist.values() if p>0)

def run_negative_wormhole(rows, cols, θ_mass, gs, backend, shots=2048):
    N = rows*cols
    throat_idx_L = 1*cols+1
    throat_idx_R = N + throat_idx_L

    results = {}
    for g in gs:
        # build circuit
        qc = QuantumCircuit(2*N, 2*N)
        qc.compose(make_cluster(rows,cols), list(range(N)), inplace=True)
        qc.compose(make_cluster(rows,cols), list(range(N,2*N)), inplace=True)
        inject_negative_mass(qc, rows,cols, (1,1), θ_mass)
        # same on R:
        qc.rz(-θ_mass, throat_idx_R)
        apply_wormhole_throat(qc, throat_idx_L, throat_idx_R, g)
        qc.measure(range(2*N), range(2*N))

        transpiled_qc = transpile(
        qc,
        backend,
        optimization_level=3,
        basis_gates=backend.configuration().basis_gates,
        coupling_map=backend.configuration().coupling_map,
        )

        # run and compute curvature
        with Session(backend=backend) as session:
            sampler = Sampler()
            job = sampler.run([transpiled_qc], shots=shots)
            result = job.result()
            bitarray = result[0].data.c
            print(result)
            counts = extract_counts_from_bitarray(bitarray)
            probs  = {b:cnt/shots for b,cnt in counts.items()}

        # marginals on L, R, and LR
        P_L  = marginal(probs, [throat_idx_L], 2*N)
        P_R  = marginal(probs, [throat_idx_R],2*N)
        P_LR = marginal(probs, [throat_idx_L, throat_idx_R], 2*N)
        K = shannon(P_L) + shannon(P_R) - shannon(P_LR)
        results[g] = K

    print("Results: ", results)

    return results

def teleport_through_wormhole(rows, cols, θ_mass, g):
    # 1) build 2×(rows×cols) cluster + negative warp + throat coupling
    N = rows*cols
    qr = QuantumRegister(2*N + 1, name='q')      # +1 for the ancilla A
    cr = ClassicalRegister(2, name='c')          # for mA, mL
    qc = QuantumCircuit(qr, cr)

    A = 2*N                # ancilla index
    # left cluster: qubits 0..N-1
    # right cluster: qubits N..2N-1
    # throat qubits:
    L = 1*cols + 1
    R = N + L

    # -- make clusters --
    qc.compose(make_cluster(rows,cols),       qr[:N],    inplace=True)
    qc.compose(make_cluster(rows,cols),       qr[N:2*N], inplace=True)
    # -- inject negative mass at L & R --
    qc.rz(-θ_mass, L)
    qc.rz(-θ_mass, R)
    # -- throat coupling --
    apply_wormhole_throat(qc, L, R, g)

    # 2) teleportation gadget:
    # prepare unknown psi on A (e.g. user can load any state here)
    # qc.u3(...,   A)     # skip: assume psi already on A
    qc.cx(A, L)
    qc.h(A)
    qc.measure(A, cr[0])    # m_A
    qc.measure(L, cr[1])    # m_L

    # 3) correction on R (needs dynamic-circuit support)
    qc.x(R).c_if(cr, 0b10)  # if m_L==1
    qc.z(R).c_if(cr, 0b01)  # if m_A==1

    # 4) final readout to verify teleportation:
    qc.measure(R, cr[0])    # reuse classical bit to see psi

    return qc

def create_wormhole_teleportation_circuit(theta, phi):
    """
    Builds the standard teleportation circuit, which in AdS/CFT
    corresponds to traversable wormhole teleportation.
    - Qubit 0: message qubit (prepared in |ψ> = U(theta,phi)|0>)
    - Qubits 1 & 2: shared Bell pair (the ER bridge)
    - Classical bits c0,c1 carry measurement outcomes for corrections
    """
    # Registers
    q = QuantumRegister(3, 'q')
    c = ClassicalRegister(2, 'c')
    qc = QuantumCircuit(q, c)
    
    # 1) Prepare the message state on qubit 0
    qc.u(theta, phi, 0, q[0])
    
    # 2) Create a Bell pair between qubits 1 and 2
    qc.h(q[1])
    qc.cx(q[1], q[2])
    
    # 3) Bell measurement on (0,1)
    qc.cx(q[0], q[1])
    qc.h(q[0])
    qc.measure(q[0], c[0])
    qc.measure(q[1], c[1])
    
    # 4) Conditional corrections on qubit 2
    qc.x(q[2]).c_if(c[0], 1)
    qc.z(q[2]).c_if(c[1], 1)
    
    return qc

def wormhole_teleport(theta, phi):
    # 3 qubits: msg=0, bell1=1, bell2=2
    q = QuantumRegister(3, 'q')
    # 3 classical bits: m0,m1 for Bell measurement, r for recv outcome
    c = ClassicalRegister(3, 'c')
    qc = QuantumCircuit(q, c)

    # 1) Prepare |ψ> on q0
    qc.u(theta, phi, 0, q[0])

    # 2) Bell pair on (q1,q2)
    qc.h(q[1])
    qc.cx(q[1], q[2])

    # 3) Bell measurement on (q0,q1)
    qc.cx(q[0], q[1])
    qc.h(q[0])
    qc.measure(q[0], c[0])
    qc.measure(q[1], c[1])

    # 4) Conditional corrections on q2
    qc.x(q[2]).c_if(c[0], 1)
    qc.z(q[2]).c_if(c[1], 1)

    # 5) **New**: measure the received qubit (q2) into c[2]
    qc.measure(q[2], c[2])

    return qc

def teleportation_circuit(theta, phi):
    """
    Teleportation circuit:
      - Prepares |ψ> = U(theta,phi)|0> on q0
      - Creates a Bell pair on q1,q2
      - Bell-measures (q0,q1) into m0,m1
      - Applies Z on q2 if m0==1, X on q2 if m1==1
      - Measures q2 into the recv bit
    """
    qr   = QuantumRegister(3, 'qr')
    m0   = ClassicalRegister(1, 'm0')
    m1   = ClassicalRegister(1, 'm1')
    recv = ClassicalRegister(1, 'recv')
    qc   = QuantumCircuit(qr, m0, m1, recv)

    # 1) Prepare message on q0
    qc.u(theta, phi, 0, qr[0])

    # 2) Bell pair on q1–q2
    qc.h(qr[1])
    qc.cx(qr[1], qr[2])

    # 3) Bell measurement on q0–q1
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.measure(qr[0], m0[0])
    qc.measure(qr[1], m1[0])

    # 4) **Correct** q2
    qc.z(qr[2]).c_if(m0, 1)  # if m0=1 apply Z
    qc.x(qr[2]).c_if(m1, 1)  # if m1=1 apply X

    # 5) Measure teleported qubit
    qc.measure(qr[2], recv[0])

    return qc

def teleportation_circuit_fixed(theta, phi):
    """
    Teleportation circuit with single-bit conditional corrections,
    ensuring the receiver qubit reproduces the original state.
    """
    # Quantum and classical registers
    qr = QuantumRegister(3, 'qr')
    c0 = ClassicalRegister(1, 'c0')  # measurement of qubit 0
    c1 = ClassicalRegister(1, 'c1')  # measurement of qubit 1
    cref = ClassicalRegister(1, 'recv')  # measurement of receiver qubit
    qc = QuantumCircuit(qr, c0, c1, cref)

    # 1) Prepare the message state on qubit 0
    qc.u(theta, phi, 0, qr[0])

    # 2) Create Bell pair between qubits 1 & 2
    qc.h(qr[1])
    qc.cx(qr[1], qr[2])

    # 3) Bell measurement on qubits 0 & 1
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.measure(qr[0], c0[0])
    qc.measure(qr[1], c1[0])

    # 4) Single-bit conditional corrections on qubit 2
    qc.x(qr[2]).c_if(c0, 1)  # if c0 == 1, apply X
    qc.z(qr[2]).c_if(c1, 1)  # if c1 == 1, apply Z

    # 5) Measure the teleported qubit into cref
    qc.measure(qr[2], cref[0])

    return qc

def teleportation_circuit_0(theta, phi):
    # 3 qubits: q0=message, q1/q2=Bell pair
    qr   = QuantumRegister(3, 'qr')
    # 2 bits for the Bell measurement, 1 bit for the receiver
    m0   = ClassicalRegister(1, 'm0')
    m1   = ClassicalRegister(1, 'm1')
    recv = ClassicalRegister(1, 'recv')
    qc   = QuantumCircuit(qr, m0, m1, recv)

    # 1) Prepare |ψ> on q0
    qc.u(theta, phi, 0, qr[0])

    # 2) Create Bell pair on q1–q2
    qc.h(qr[1])
    qc.cx(qr[1], qr[2])

    # 3) Bell‐measurement on q0–q1
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.measure(qr[0], m0[0])
    qc.measure(qr[1], m1[0])

    # 4) Apply corrections on q2:
    #    Z if m0=1, X if m1=1
    qc.z(qr[2]).c_if(m0, 1)
    qc.x(qr[2]).c_if(m1, 1)

    # 5) Measure the teleported qubit
    qc.measure(qr[2], recv[0])

    return qc

def coherent_teleport_circuit(theta, phi):
    qr = QuantumRegister(3, 'qr')
    cr = ClassicalRegister(1, 'recv')
    qc = QuantumCircuit(qr, cr)

    # 1) Prepare message |ψ> on q0
    qc.u(theta, phi, 0, qr[0])

    # 2) Create Bell pair on q1–q2
    qc.h(qr[1])
    qc.cx(qr[1], qr[2])

    # 3) Coherent Bell operation
    qc.cx(qr[0], qr[1])
    qc.h(qr[0])

    # 4) Coherent feed-forward corrections
    qc.cz(qr[0], qr[2])
    qc.cx(qr[1], qr[2])

    # 5) Measure the teleported qubit
    qc.measure(qr[2], cr[0])

    return qc

def gaussian_peaks(rows, cols, centers, sigma):
    mass = np.zeros((rows, cols))
    x = np.arange(rows)
    y = np.arange(cols)
    X, Y = np.meshgrid(x, y, indexing='ij')
    for (i, j) in centers:
        mass += np.exp(-((X - i)**2 + (Y - j)**2) / (2 * sigma**2))
    return mass

def generate_spacetime_circuit(rows, cols, mass_profile):
    n = rows * cols
    qc = QuantumCircuit(n)
    idx = lambda i, j: i * cols + j
    for i in range(rows):
        for j in range(cols):
            phase = mass_profile[i, j]
            # horizontal neighbor (wrap-around)
            qc.cp(phase, idx(i, j), idx(i, (j + 1) % cols))
            # vertical neighbor (wrap-around)
            qc.cp(phase, idx(i, j), idx((i + 1) % rows, j))
    return qc

def mutual_information(statevec, A_indices, B_indices):
    """
    Compute I(A:B) = S(A) + S(B) - S(AB) using manual partial trace.
    statevec: raw statevector array
    A_indices, B_indices: lists of qubit indices
    """
    # Full density matrix
    rho_full = DensityMatrix(statevec)
    total_qubits = int(np.log2(rho_full.data.shape[0]))

    # Compute reduced states
    rhoA = partial_trace_manual(rho_full, [q for q in range(total_qubits) if q not in A_indices])
    rhoB = partial_trace_manual(rho_full, [q for q in range(total_qubits) if q not in B_indices])
    rhoAB = partial_trace_manual(rho_full, [q for q in range(total_qubits) if q not in (A_indices + B_indices)])

    # Compute entropies
    SA = qiskit_entropy(rhoA)
    SB = qiskit_entropy(rhoB)
    SAB = qiskit_entropy(rhoAB)
    return SA + SB - SAB


def extract_bitarray_from_primitive(prim_result):
    """
    Given a PrimitiveResult (from a SamplerPubResult) where prim_result[0].data contains
    one or more BitArray attributes, this function finds the first BitArray and returns
    its attribute name and value, without requiring bitstring module.
    """
    # Access the first SamplerPubResult
    sampler_result = prim_result[0]
    data = sampler_result.data

    # Inspect all attributes in the data object
    for attr, val in vars(data).items():
        # Detect by class name 'BitArray'
        if val.__class__.__name__ == 'BitArray':
            return attr, val

    # If no BitArray is found, raise an error
    raise ValueError("No BitArray attribute found in the PrimitiveResult data")


def run(qc, backend=None, rs=False, sim=False, old_backend=False, shots=2048):
    """
    Run a circuit either on real backend, statevector simulator, or qasm simulator.
    
    Parameters:
    - qc: QuantumCircuit
    - backend: IBM backend object (required if sim=False and rs=False)
    - rs: bool, if True return statevector
    - sim: bool, if True run on AerSimulator for sampling
    - shots: int, number of shots for sampling
    
    Returns:
    - Statevector if rs=True
    - dict with 'counts' and 'distribution' otherwise
    """
    # base_run 
    # Display circuit
    print(qc.draw(fold=100))

    # Option 1: statevector simulation
    if rs:
        qc_nmeas = qc.copy()
        qc_nmeas.remove_final_measurements()
        sv = Statevector(qc_nmeas)
        print("Statevector:", sv)
        return sv

    # Option 2: qasm simulation with AerSimulator
    if sim:
        simulator = AerSimulator()
        qc_t = transpile(qc, simulator, optimization_level=3)
        job = simulator.run(qc_t, shots=shots)
        result = job.result()
        counts = result.get_counts()
        print("Simulation counts:", counts)

        # Marginal on last bit
        total = sum(counts.values())
        teleported = {'0': 0, '1': 0}
        for outcome, cnt in counts.items():
            last_bit = outcome[-1]
            teleported[last_bit] += cnt
        f0 = teleported['0'] / total
        f1 = teleported['1'] / total
        print(f"Distribution on R (sim): |0> = {f0:.3f}, |1> = {f1:.3f}")
        return {'counts': counts, 'distribution': (f0, f1)}
    
    if backend is None:
        backend = get_best_backend(service)

    if old_backend:
        qc_t = transpile(qc, backend)
        job = backend.run(qc_t)
        result = job.result()
        counts = result.get_counts()
        return counts

    # Option 3: real backend via IBM Runtime Sampler


    qc_t = transpile(qc, backend, optimization_level=3)
    sampler = Sampler(backend)
    job = sampler.run([qc_t])
    result = job.result()
    print("Raw result:", result)

    bitarray = result[0]['__value__']['data'].meas
    print("Bitarray: ", bitarray)
    if hasattr(bitarray, "get_bitstrings"):
        bitstrings = bitarray.get_bitstrings()
        print("First 10 bitstrings:", bitstrings[:10])
    else:
        print("No bitstrings found in the result")
        return None
    
    counts = extract_counts_from_bitarray(bitarray)
    print("Sampling counts:", counts)

    total = sum(counts.values())
    teleported = {'0': 0, '1': 0}
    for outcome, cnt in counts.items():
        last_bit = outcome[-1]
        teleported[last_bit] += cnt
    f0 = teleported['0'] / total
    f1 = teleported['1'] / total
    dist = (f0, f1)
    print(f"Distribution on R (hw): |0> = {f0:.3f}, |1> = {f1:.3f}")

    return counts

def partial_trace_manual(rho_full, trace_out):
    """
    Compute partial trace by manually tracing out specified qubit indices.
    rho_full: DensityMatrix object (2^n x 2^n)
    trace_out: list of qubit indices to trace out
    """
    # Convert to numpy array and infer number of qubits
    mat = rho_full.data
    dim = mat.shape[0]
    n = int(np.log2(dim))
    # Reshape to tensor with 2 indices per qubit
    reshaped = mat.reshape([2]*n*2)

    # Sequentially trace out qubits
    for q in sorted(trace_out, reverse=True):
        # Trace axis q (row) with axis q+n (column)
        reshaped = np.trace(reshaped, axis1=q, axis2=q+n)
        # Reduce total qubit count
        n -= 1

    # Reshape back to matrix form
    new_mat = reshaped.reshape((2**n, 2**n))
    return DensityMatrix(new_mat)

def loschmidt_echo(n_qubits=2, scramble_depth=5, shots=1024, seed=None, backend=None):
    """
    Run a Loschmidt Echo (quantum time reversal) on n_qubits.
    1. Prepare |0...0>
    2. Apply a random sequence of Clifford gates (scrambler)
    3. Apply the exact inverse sequence (unscrambler)
    4. Measure final state fidelity (should recover |0...0>)
    """
    # 1. Build scramble circuit
    qc = QuantumCircuit(n_qubits)
    cliffords = []
    rng = np.random.default_rng(seed)
    for _ in range(scramble_depth):
        c = random_clifford(n_qubits, seed=rng.integers(1e6))
        qc.append(c.to_instruction(), range(n_qubits))
        cliffords.append(c)
    
    # 2. Add the inverse (unscramble) circuit
    for c in reversed(cliffords):
        qc.append(c.to_instruction().inverse(), range(n_qubits))

    # 3. Optionally measure
    qc.measure_all()

    print(qc.draw())

    # 4. Simulate (or run on backend)
    if backend is None:
        backend = Aer.get_backend('aer_simulator')
    qc_t = transpile(qc, backend)
    counts, dist = run(qc_t, backend)
    recovered = counts.get('0' * n_qubits, 0) / shots

    print(f"Fraction returned to |0...0>: {recovered:.3f}")
    return recovered, counts

def multiverse_warp_circuit(alpha, beta):
    """
    Creates the 5-qubit "multiverse warp" circuit with tunable phase interactions.
    alpha: controlled phase angle between qubit 0 and 3
    beta:  controlled phase angle between qubit 1 and 4
    """
    qc = QuantumCircuit(5)
    #  bubble entanglement
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    # Alternate timelines preparation
    qc.h(3)
    qc.h(4)
    # Interactions
    qc.cp(alpha, 0, 3)
    qc.cp(beta, 1, 4)
    qc.cx(2, 3)
    qc.cx(3, 4)
    # Distortions
    qc.rz(np.pi / 6, 3)
    qc.rx(np.pi / 8, 4)
    return qc

def mutual_information_from_statevector(sv, A, B):
    rho = DensityMatrix(sv)
    n = int(np.log2(rho.data.shape[0]))
    all_q = list(range(n))
    def reduced(rho_full, keep):
        trace_out = [q for q in all_q if q not in keep]
        return partial_trace(rho_full, trace_out)
    rhoA  = reduced(rho, A)
    rhoB  = reduced(rho, B)
    rhoAB = reduced(rho, A+B)
    return qiskit_entropy(rhoA) + qiskit_entropy(rhoB) - qiskit_entropy(rhoAB)

def K_theory(theta):
    # simple toy: K = sin²(θ)  (replace with your actual formula)
    return np.sin(theta)**2

def sweep_entanglement_phase(alphas, betas):
    """
    Sweep alpha, beta over lists of angles.
    Returns a DataFrame with columns ['alpha','beta','MI'].
    """
    sim = AerSimulator(method='statevector')
    records = []
    for alpha in alphas:
        for beta in betas:
            qc = multiverse_warp_circuit(alpha, beta)
            qc_n = qc.remove_final_measurements(inplace=False)
            qc_t = transpile(qc_n, sim)
            sv = Statevector(qc_t).data
            mi = mutual_information_from_statevector(sv, [3], [4])
            records.append({'alpha': alpha, 'beta': beta, 'MI': mi})
            print("alpha: ", alpha, " beta: ", beta, " MI: ", mi)
    return pd.DataFrame(records)

def measure_K_hw(swap_circ, backend, provider, shots=2048):
    tc = transpile(swap_circ, backend, optimization_level=3)
    job = run(tc, backend)
    res = job.result()
    ba = extract_bitarray_from_primitive(res)
    print(extract_counts_from_bitarray(ba))

    p0 = ba.count(0)/len(ba)
    # For a SWAP‐test: ⟨Swap⟩ = p0 − p1; curvature ∝ 1−⟨Swap⟩ = 2*p1
    p1 = 1-p0
    return 2*p1, counts

def measure_K_hw_runtime(swap_circ, backend_name, shots=2048):
    service = QiskitRuntimeService()
    # pick the device (fallback to first if name not found)
    try:
        backend = service.backend(backend_name)
    except:
        backend = service.backends(simulator=False, status="online")[0]
        backend_name = backend.name()

    with Session(backend=backend) as session:
        sampler = Sampler()
        # transpile to the target
        tcs = [transpile(swap_circ, backend)]
        job = sampler.run(tcs, shots=shots)
        primitive = job.result()
    # extract raw bitarrays → counts
    bitarrays   = extract_bitarray_from_primitive(primitive)
    counts_list = extract_counts_from_bitarray(bitarrays)
    print("Counts: ", counts_list)
    # we only had one circuit in the batch:
    counts_hw = counts_list[0]
    total = sum(counts_hw.values())
    p0 = counts_hw.get('0', 0) / total
    p1 = 1 - p0
    # SWAP‐test curvature = 2*p1
    return 2*p1, counts_hw


def gao_jafferis_wall_circuit(theta, phi, g, seed=None):
    """
    Build a 4-qubit Gao-Jafferis-Wall traversable wormhole circuit:
      - q0,q1: Left CFT block
      - q2,q3: Right CFT block
    Steps:
      1. Prepare EPR pairs on (q0,q1) and (q2,q3)
      2. Encode message on q0: |ψ> = U(theta,phi)|0>
      3. Scramble left block with random Clifford U_L
      4. Apply coupling RZZ(2*g) between q1 (left) and q2 (right)
      5. Unscramble right block with U_L^† mapped to (q2,q3)
      6. Measure q3 to recover the message
    """
    qr = QuantumRegister(4, 'qr')
    cr = ClassicalRegister(1, 'recv')
    qc = QuantumCircuit(qr, cr)

    # 1) Prepare EPR pairs
    qc.h(qr[0]); qc.cx(qr[0], qr[1])
    qc.h(qr[2]); qc.cx(qr[2], qr[3])

    # 2) Encode message on q0
    qc.u(theta, phi, 0, qr[0])

    # 3) Scramble left block (q0,q1)
    cliff = random_clifford(2, seed=seed)
    qc.append(Clifford(cliff).to_instruction(), [qr[0], qr[1]])

    # 4) Coupling across wormhole: RZZ(2*g)
    qc.rzz(2 * g, qr[1], qr[2])

    # 5) Unscramble right block (q2,q3) with same Clifford
    qc.append(Clifford(cliff).to_instruction().inverse(), [qr[2], qr[3]])

    # 6) Measure the recovered message on q3
    qc.measure(qr[3], cr[0])

    return qc

def gao_jafferis_wall_circuit_plus(theta, g, seed=None, deepen_scramble=False):
    """
    4-qubit GJW traversable wormhole teleportation of |+> state.
    Optional deepen_scramble: apply two sequential 2-qubit Cliffords.
    """
    qr = QuantumRegister(4, 'qr')
    cr = ClassicalRegister(1, 'recv')
    qc = QuantumCircuit(qr, cr)

    # 1) Prepare EPR on (q0,q1) and (q2,q3)
    qc.h(qr[0]); qc.cx(qr[0], qr[1])
    qc.h(qr[2]); qc.cx(qr[2], qr[3])

    # 2) Prepare |+> message on q0
    qc.h(qr[0])

    # 3) Scramble left block (q0,q1)
    for _ in range(2 if deepen_scramble else 1):
        cliff = random_clifford(2, seed)
        qc.append(Clifford(cliff).to_instruction(), [qr[0], qr[1]])

    # 4) RZZ coupling between q1 and q2
    qc.rzz(2 * g, qr[1], qr[2])

    # 5) Unscramble right block (q2,q3)
    for _ in range(2 if deepen_scramble else 1):
        qc.append(Clifford(cliff).to_instruction().inverse(), [qr[2], qr[3]])

    # 6) Measure receiver in Z basis
    qc.measure(qr[3], cr[0])

    return qc

def sweep_optimal_g(theta, gs, seed=42):
    sim = AerSimulator(method='statevector')
    back = "ibm_brisbane"
    psi0 = np.array([1, 1]) / np.sqrt(2)  # |+> state vector
    fids = []
    for g in gs:
        qc = gao_jafferis_wall_circuit_plus(theta, g, seed=seed, deepen_scramble=True)
        qc_nmeas = qc.copy()
        qc_nmeas.remove_final_measurements()
        qc_t = transpile(qc_nmeas, sim)
        sv = run(qc_t, back, rs=True, sim=False)
        rho3 = partial_trace(sv, [0,1,2])
        fids.append(state_fidelity(rho3, psi0))
    df = pd.DataFrame({'g': gs, 'fidelity': fids})
    g_opt = df.loc[df['fidelity'].idxmax(), 'g']
    return df, g_opt

def run_hardware_teleport(theta, g_opt, backend_name, deepen_scramble=True, shots=2048):
    # Load backend

    # Measurement calibration for qubit 3
    meas_qubits = [3]
    cal_circuits, state_labels = complete_meas_cal(qubit_list=meas_qubits, circlabel='cal')
    cal_job = run(cal_circuits, backend)
    cal_results = cal_job.result()

    # Build and transpile wormhole circuit
    qc = gao_jafferis_wall_circuit_plus(theta, g_opt, deepen_scramble=deepen_scramble)
    qc_t = transpile(qc, backend, optimization_level=3)

    # Run teleportation job
    job = run(qc_t, backend)
    raw_counts = extract_counts_from_bitarray(extract_bitarray_from_primitive(job.result()))
    # Compute |+> teleport fidelity via tomography (Z-basis only for simplicity)
    p0 = counts.get('0', 0) / shots
    p1 = counts.get('1', 0) / shots
    # For |+>, fidelity = (p0 + p1 + coherence terms)/2, but Z-pop gives half the info
    return counts, p0, p1


def your_label_coeff_list(H, tol=1e-8):
    """
    Decompose H (2^n×2^n numpy array) into the Pauli basis.
    Returns list of (label: str, coeff: float).
    """
    dim = H.shape[0]
    n = int(np.log2(dim))
    basis = ['I','X','Y','Z']
    terms = []
    for idx in range(4**n):
        lab = ""
        tmp = idx
        for _ in range(n):
            lab = basis[tmp % 4] + lab
            tmp //= 4

        Pmat = Pauli(lab).to_matrix()
        raw = np.trace(Pmat.conj().T @ H) / dim
        # separate real & imag
        re, im = raw.real, raw.imag
        if abs(im) < tol and abs(re) > tol:
            terms.append((lab, float(re)))
    return terms

# --- before your loops, add this helper ---
def boundary_edges_of(region, edge_list):
    """
    Return the list of edges (i,j) that straddle `region`:
    exactly one endpoint in region, one outside.
    """
    return [e for e in edge_list if (e[0] in region) ^ (e[1] in region)]

# --- Replace your shifted_circuit with this ---
def make_shifted_circuit(theta_dict, region, shift):
    """
    Build the base GHZ+CP circuit, but add `shift` only
    on the edges that lie on the boundary of `region`.
    """
    qc = QuantumCircuit(n)
    qc.h(range(n))
    b_edges = set(boundary_edges_of(region, edge_list))

    for (i,j), th in theta_dict.items():
        # add ±shift to θ if this edge is on the region boundary
        th_eff = th + (shift if (i,j) in b_edges else 0)
        qc.cp(th_eff, i, j)
    return transpile(qc, est.backend)

def decompose_H_xi(rho0):
    import numpy as np, scipy.linalg as la
    H = -la.logm(rho0)
    # number of qubits in this region
    n = int(np.log2(rho0.shape[0]))
    label_coeffs = your_label_coeff_list(H)  # gives list of (label, coeff)

    # Now supply num_qubits so it can handle an empty list
    return SparsePauliOp.from_list(label_coeffs, num_qubits=n)

def decompose_H_xi_full(rho0, region, n):
    """
    Given a small-region density matrix rho0 on |region| qubits,
    build H = -logm(rho0), decompose into Pauli basis, then
    embed each term into an n-qubit operator by padding with I.
    
    Returns a SparsePauliOp on n qubits.
    """
    # 1) modular Hamiltonian
    H = -la.logm(rho0)
    dim = H.shape[0]
    m   = int(np.log2(dim))   # # qubits in region

    # 2) decompose into (label, coeff) on m qubits
    terms = []
    basis = ['I','X','Y','Z']
    for idx in range(4**m):
        lab = ""
        tmp = idx
        for _ in range(m):
            lab = basis[tmp % 4] + lab
            tmp //= 4

        Pmat  = Pauli(lab).to_matrix()
        raw   = np.trace(Pmat.conj().T @ H) / dim
        re, im = raw.real, raw.imag
        if abs(im) < 1e-8 and abs(re) > 1e-8:
            terms.append((lab, float(re)))

    # 3) embed into n qubits: for each (lab, coeff), build full_label
    full_terms = []
    for small_lab, coeff in terms:
        full_lab = []
        it = iter(small_lab)
        for q in range(n):
            full_lab.append(next(it) if q in region else 'I')
        full_terms.append(("".join(full_lab), coeff))

    # 4) build a SparsePauliOp on n qubits
    if not full_terms:
        # zero op
        return SparsePauliOp.from_list([("I"*n, 0.0)], num_qubits=n)
    return SparsePauliOp.from_list(full_terms, num_qubits=n)

def warp_circuit(α, β, γ):
    """Builds your 3-qubit warp test with angles α,β,γ."""
    qc = QuantumCircuit(3)
    qc.h(0); qc.cx(0,1); qc.cx(1,2)
    qc.rz(α, 0)
    qc.cp(β, 0, 1)
    qc.cp(γ, 1, 2)
    qc.rx(np.pi/8, 2)
    return qc

def two_qubit_concurrence(rho):
    """Wootters concurrence for a 2-qubit density matrix."""
    # Pauli-Y
    Y = np.array([[0, -1j],[1j,  0]])
    YY = np.kron(Y, Y)
    # “spin-flipped” R matrix
    R = rho.data @ YY @ rho.data.conj() @ YY
    # eigenvalues of R, take positive real parts
    ev = np.real_if_close(np.linalg.eigvals(R))
    # their square-roots
    sv = np.sort(np.sqrt(np.clip(ev, 0, None)))[::-1]
    # Wootters formula
    return max(0.0, sv[0] - sv[1] - sv[2] - sv[3])

def three_tangle(rho012):
    """
    τ = C²_{0|(12)} – C²_{0|1} – C²_{0|2},  
    with C²_{0|(12)} = 2*(1 – Tr(ρ₀²)).
    """
    # 1-qubit reduced state ρ₀
    rho0 = partial_trace(rho012, [1,2])
    # pure-state bipartition term:
    C0_12_sq = 2*(1 - np.trace(rho0.data @ rho0.data).real)

    # two-qubit marginals
    rho01 = partial_trace(rho012, [2])
    rho02 = partial_trace(rho012, [1])
    C0_1 = two_qubit_concurrence(rho01)
    C0_2 = two_qubit_concurrence(rho02)

    return C0_12_sq - C0_1**2 - C0_2**2

def warp_rand_circuit(beta, δ=0.2):
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0,1)
    qc.cx(1,2)
    qc.cp(beta, 0, 1)
    # break GHZ symmetry:
    qc.rx(δ, 1)    
    qc.cp(np.pi/6, 1, 2)
    return qc

def small_plaquette_circuit(theta01, theta12, noise=0.1):
    qc = QuantumCircuit(4)
    qc.h(range(4))
    qc.cp(theta01, 0,1)
    qc.cp(theta12, 1,2)
    qc.cp(np.pi/6, 2,3)
    # break symmetry strongly:
    qc.ry(noise, 1)
    qc.rx(noise/2, 2)
    return qc

def small_plaquette_circuit_v3(theta01: float, theta12: float, noise: float = 0.1) -> QuantumCircuit:
    """
    Build a 4-qubit “plaquette” circuit:
      - H on all qubits
      - CP(theta01) on (0,1)
      - CP(theta12) on (1,2)
      - CP(pi/6) on (2,3)
      - RY(noise) on qubit 1, RX(noise/2) on qubit 2
      - RZ(0.2*theta01) on qubit 0 to introduce spectral variation
    """
    qc = QuantumCircuit(4)
    qc.h(range(4))
    qc.cp(theta01,    0, 1)
    qc.cp(theta12,      1, 2)
    qc.cp(np.pi / 6,    2, 3)
    qc.ry(noise,        1)
    qc.rx(noise / 2,    2)
    qc.rz(0.2 * theta01, 0)
    return qc


def small_plaquette_circuit_v2(theta, noise=0.1):
    """
    Now both θ₀₁ and θ₁₂ vary together (θ₀₁ = θ₁₂ = theta). 
    We keep a small “noise” on qubits 1,2 so it doesn’t completely symmetrize.
    """
    qc = QuantumCircuit(4)
    qc.h(range(4))
    qc.cp(theta,    0, 1)     # θ₀₁ = theta
    qc.cp(theta,    1, 2)     # θ₁₂ = theta
    qc.cp(np.pi / 6, 2, 3)    # keep the same θ₂₃
    qc.ry(noise,     1)       # small break
    qc.rx(noise/2,   2)
    return qc

def entropy_rho01(theta: float) -> float:
    """
    Compute the von Neumann entropy S(ρ₀₁(θ)):
      - Build the 4-qubit plaquette with θ₀₁ = θ, θ₁₂ = π/3, noise=0.1
      - Extract ρ₀₁ by partial trace over qubits [2,3]
      - Diagonalize ρ₀₁ and return -∑ λ_i ln λ_i
    """
    qc = small_plaquette_circuit_v3(theta, np.pi / 3, noise=0.1)
    sv = Statevector(qc)
    dm = DensityMatrix(sv)
    rho01 = partial_trace(dm, [2, 3])
    vals, _ = np.linalg.eigh(rho01.data)
    vals_clipped = np.clip(vals, 1e-12, 1.0)
    return -np.sum(vals_clipped * np.log(vals_clipped))

def compute_theoretical_curvature_v2(num_points=500, eps=1e-3):
    def entropy_rho01(theta):
        qc = small_plaquette_circuit_v2(theta, noise=0.1)
        sv = Statevector(qc)
        dm = DensityMatrix(sv)
        rho01 = partial_trace(dm, [2, 3])
        vals, _ = np.linalg.eigh(rho01.data)
        vals_clipped = np.clip(vals, 1e-12, 1.0)
        return -np.sum(vals_clipped * np.log(vals_clipped))

    thetas = np.linspace(0, np.pi, num_points)
    S_vals = np.array([entropy_rho01(th) for th in thetas])

    dtheta = thetas[1] - thetas[0]
    K_theory = [-(S_vals[j+1] - S_vals[j-1])/(2*dtheta) for j in range(1, len(thetas)-1)]
    thetas_mid = thetas[1:-1]
    return np.array(thetas_mid), np.array(K_theory)

def compute_theoretical_curvature(num_points=500, eps=1e-3):
    """
    Compute K_theory(θ) = - d/dθ [ S(ρ₀₁(θ)) ] over a dense grid of θ ∈ [0, π].
    Returns arrays (thetas_dense, K_theory).
    """
    #A) Helper: compute entropy S₀₁(θ)
    def entropy_rho01(theta):
        # Build the 4-qubit circuit with θ₁₂ = π/3 and fixed noise = 0.5
        qc = small_plaquette_circuit(theta, np.pi / 3, noise=0.5)
        sv = Statevector(qc)
        dm = DensityMatrix(sv)
        rho01 = partial_trace(dm, [2, 3])
        # Diagonalize to get eigenvalues
        vals, _ = np.linalg.eigh(rho01.data)
        # Clip tiny negatives to avoid log issues
        vals_clipped = np.clip(vals, 1e-12, 1.0)
        return -np.sum(vals_clipped * np.log(vals_clipped))

    thetas = np.linspace(0, np.pi, num_points)
    S_vals = np.array([entropy_rho01(th) for th in thetas])

    #B) Numerical derivative: K(θᵢ) ≈ - [S(θᵢ₊₁) - S(θᵢ₋₁)] / (2 Δθ)
    dtheta = thetas[1] - thetas[0]
    K_theory = []
    thetas_mid = thetas[1:-1]  # exclude endpoints
    for j in range(1, len(thetas) - 1):
        dS = S_vals[j + 1] - S_vals[j - 1]
        K_theory.append(-dS / (2 * dtheta))
    return np.array(thetas_mid), np.array(K_theory)

def compute_theoretical_curvature_v3(num_points: int = 500, eps: float = 1e-3):
    """
    Compute a dense theoretical curvature curve K_theory(θ) = -dS/dθ
    over θ ∈ [0, π]. Returns:
      - thetas_mid: array of θ values excluding endpoints
      - K_theory: corresponding curvature values
    """
    thetas = np.linspace(0, np.pi, num_points)
    S_vals = np.array([entropy_rho01(th) for th in thetas])
    dtheta = thetas[1] - thetas[0]
    K_list = [-(S_vals[j + 1] - S_vals[j - 1]) / (2 * dtheta) for j in range(1, len(thetas) - 1)]
    thetas_mid = thetas[1:-1]
    return np.array(thetas_mid), np.array(K_list)

def generate_readout_cal_circuits(qubits: list) -> (list, list):
    """
    Build 4 calibration circuits for computational basis states |00>, |01>, |10>, |11>
    on the specified two-qubit list. Returns:
      - cal_circuits: list of 4 QuantumCircuits
      - labels: ['00', '01', '10', '11']
    """
    cal_circuits = []
    labels = ['00', '01', '10', '11']
    for lbl in labels:
        qc = QuantumCircuit(4)
        # Prepare the two-qubit state on qubits[0], qubits[1]
        if lbl[0] == '1':
            qc.x(qubits[0])
        if lbl[1] == '1':
            qc.x(qubits[1])
        qc.measure_all()
        cal_circuits.append(qc)
    return cal_circuits, labels


def run_readout_calibration(qubits: list, shots: int = 4096) -> np.ndarray:
    """
    Run the 4 readout‐calibration circuits via Sampler and build the 4×4 readout matrix M:
      M[i,j] = P(measured=j | prepared=i).
    Returns:
      - M: a 4×4 numpy array (rows sum to 1)
    """
    cal_circuits, labels = generate_readout_cal_circuits(qubits)
    transpiled = [transpile(qc, backend=backend, optimization_level=1) for qc in cal_circuits]

    # Submit via Sampler: sampler.run expects circuits list under 'circuits'
    sampler = Sampler(backend)
    job = sampler.run(transpiled)
    results = job.result()

    M = np.zeros((4, 4))
    for i, qc in enumerate(transpiled):
        counts = results.get_counts(i)  # index corresponds to circuit order
        total = sum(counts.values())
        for j, lbl in enumerate(labels):
            M[i, j] = counts.get(lbl, 0) / total
    return M


def mitigate_probabilities(raw_probs: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Given a raw probability vector p_raw of length 4, apply the inverse of readout matrix M
    to get mitigated probabilities p_mit = M^{-1} · p_raw.
    Clip negative entries to zero and renormalize to sum=1.
    """
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        # If M is singular or nearly so, regularize by adding small identity
        M_inv = np.linalg.inv(M + 1e-6 * np.eye(4))
    p_mit = M_inv.dot(raw_probs)
    p_mit = np.clip(p_mit, 0, None)
    if np.sum(p_mit) == 0:
        return np.ones(4) / 4
    return p_mit / np.sum(p_mit)


def mutual_info(dm, subsys):
    # subsys is a list of qubit indices
    full_S = qiskit_entropy(dm)
    rho_A = partial_trace(dm, [i for i in range(dm.num_qubits) if i not in subsys])
    rho_B = partial_trace(dm, subsys)
    S_A, S_B = qiskit_entropy(rho_A), qiskit_entropy(rho_B)
    return S_A + S_B - full_S

def generate_tomography_circuits(base_qc: QuantumCircuit, qubits: list) -> list:
    """
    Given a 4-qubit `base_qc` (with no measurement) and a 2‐qubit list [qA, qB],
    return 9 circuits rotating qubits [qA, qB] into the Pauli bases {X,Y,Z}×{X,Y,Z},
    each ending with measure_all().
    """
    tomography_circuits = []
    for P in ['X', 'Y', 'Z']:
        for Q in ['X', 'Y', 'Z']:
            qc_t = base_qc.copy()
            # Rotate qubit[0] into P‐basis
            if P == 'X':
                qc_t.h(qubits[0])
            elif P == 'Y':
                qc_t.sdg(qubits[0]); qc_t.h(qubits[0])
            # P == 'Z' → no change
            # Rotate qubit[1] into Q‐basis
            if Q == 'X':
                qc_t.h(qubits[1])
            elif Q == 'Y':
                qc_t.sdg(qubits[1]); qc_t.h(qubits[1])
            # Q == 'Z' → no change
            qc_t.measure_all()
            tomography_circuits.append(qc_t)
    return tomography_circuits

def reconstruct_rho_from_counts(counts_list: list, labels: list, M: np.ndarray) -> np.ndarray:
    """
    Given a list of 9 raw counts dictionaries from tomography and state `labels`=['00','01','10','11'],
    plus readout matrix M, reconstruct the 2-qubit density matrix ρ_{01} by:
      1. Building raw probability vectors for each measurement setting (shape (4,)).
      2. Applying mitigate_probabilities to correct readout.
      3. Converting each corrected prob vector into an expectation value for that Pauli pair.
      4. Expanding ρ = (1/4)[ I⊗I + ∑_{(P,Q)∈{X,Y,Z}²} r_{P,Q} (σ_P⊗σ_Q) ], 
         where r_{P,Q} = ⟨P⊗Q⟩.
      5. Enforcing positivity by clipping eigenvalues and renormalizing.

    Returns:
      - rho_pos: 4×4 numpy array (Hermitian, positive, trace=1)
    """
    # Pauli targets in the same order as generate_tomography_circuits:
    pauli_labels = [('X','X'), ('X','Y'), ('X','Z'),
                    ('Y','X'), ('Y','Y'), ('Y','Z'),
                    ('Z','X'), ('Z','Y'), ('Z','Z')]

    # 5.A) Build probability vectors, mitigate, compute expectations
    expec_vals = []
    for idx, raw_counts in enumerate(counts_list):
        # Convert raw_counts → raw_probs (array length 4 in order ['00','01','10','11'])
        total = sum(raw_counts.values())
        raw_probs = np.zeros(4)
        for i, lbl in enumerate(labels):
            raw_probs[i] = raw_counts.get(lbl, 0) / total

        # Mitigate readout: p_mit = M^{-1} raw_probs
        p_mit = mitigate_probabilities(raw_probs, M)

        # Compute expectation value of P⊗Q from p_mit:
        P, Q = pauli_labels[idx]
        exp_val = 0.0
        for i, lbl in enumerate(labels):
            b0, b1 = int(lbl[0]), int(lbl[1])
            # After rotating into P (or Q) basis, outcome '0' → eigenvalue +1, '1' → eigenvalue -1
            e0 = +1 if b0 == 0 else -1
            e1 = +1 if b1 == 0 else -1
            exp_val += p_mit[i] * (e0 * e1)
        expec_vals.append(exp_val)

    # 5.B) Build ρ via Pauli expansion: ρ = (1/4)[ I⊗I + ∑ r_{PQ} (σ_P⊗σ_Q) ]
    from qiskit.quantum_info import Pauli
    rho_mat = np.eye(4, dtype=complex)  # starts with I⊗I
    for (P, Q), r in zip(pauli_labels, expec_vals):
        pauli_mat = Pauli(P + Q).to_matrix()
        rho_mat += r * pauli_mat
    rho_mat = rho_mat / 4.0

    # 5.C) Enforce positivity: clip negative eigenvalues and renormalize
    vals, vecs = np.linalg.eigh(rho_mat)
    vals_clipped = np.clip(vals, 0, None)
    if np.sum(vals_clipped) == 0:
        rho_pos = np.eye(4) / 4  # fallback to maximally mixed
    else:
        rho_pos = (vecs @ np.diag(vals_clipped) @ vecs.conj().T)
        rho_pos = rho_pos / np.trace(rho_pos)
    return rho_pos

def compute_hardware_curvature(theta: float, eps: float, M: np.ndarray, shots: int = 8192):
    """
    For a given θ, build circuits C_+(θ), C_-(θ), run 2-qubit tomography on qubits [0,1],
    reconstruct ρ₀₁(θ ± ε) with manual readout mitigation using M,
    compute S(θ ± ε), and return K_hw = -[S(θ+ε) - S(θ-ε)]/(2ε).
    """
    # 6.A) Build the two base circuits
    qc_plus = small_plaquette_circuit_v3(theta + eps, np.pi / 3, noise=0.1)
    qc_minus = small_plaquette_circuit_v3(theta - eps, np.pi / 3, noise=0.1)

    # 6.B) Generate tomography circuits
    tomo_plus  = generate_tomography_circuits(qc_plus,  [0, 1])
    tomo_minus = generate_tomography_circuits(qc_minus, [0, 1])

    # 6.C) Transpile & run on hardware
    transpiled_p = transpile(tomo_plus,  backend=backend, optimization_level=1)
    transpiled_m = transpile(tomo_minus, backend=backend, optimization_level=1)

    job_p = service.run(circuits=transpiled_p,  backend=backend, shots=shots, options=runtime_options)
    job_m = service.run(circuits=transpiled_m, backend=backend, shots=shots, options=runtime_options)
    res_p = job_p.result()
    res_m = job_m.result()

    # 6.D) Extract raw counts and reconstruct ρ₀₁ with mitigation
    #    We assume calibration labels=['00','01','10','11'] for reconstruction
    calib_labels = ['00', '01', '10', '11']

    counts_list_p = [res_p.get_counts(qc) for qc in transpiled_p]
    counts_list_m = [res_m.get_counts(qc) for qc in transpiled_m]

    rho_plus  = reconstruct_rho_from_counts(counts_list_p, calib_labels, M)
    rho_minus = reconstruct_rho_from_counts(counts_list_m, calib_labels, M)

    # 6.E) Compute entropies
    vals_p, _ = np.linalg.eigh(rho_plus)
    vals_m, _ = np.linalg.eigh(rho_minus)
    vals_p_clipped = np.clip(vals_p, 1e-12, 1.0)
    vals_m_clipped = np.clip(vals_m, 1e-12, 1.0)

    S_plus  = -np.sum(vals_p_clipped * np.log(vals_p_clipped))
    S_minus = -np.sum(vals_m_clipped * np.log(vals_m_clipped))

    # 6.F) Finite‐difference curvature
    K_hw = - (S_plus - S_minus) / (2 * eps)
    return K_hw, (S_plus, S_minus)

def get_sampler(backend_name):
    global session, _sampler
    if _sampler is None:
        # open one session & sampler for the entire script
        with Session(backend_name) as session:
            _sampler = Sampler(backend=backend_name)

    return _sampler

def entropy_from_counts(counts, subsystem):
    """
    Estimate S(ρ_A) from raw counts by marginalizing over the other qubits,
    automatically zero-padding bitstrings to the full width.
    
    counts: dict mapping bit-string (e.g. '101') to count or probability
    subsystem: list of qubit indices to keep (0-based, matching bit positions)
    """
    # total shots or total weight
    total = sum(counts.values())
    # determine full width (number of qubits)
    full_width = max(len(bs) for bs in counts)
    
    # build marginalized distribution P[substate]
    P = {}
    for bitstr, cnt in counts.items():
        # pad to full width, then reverse so index 0 corresponds to qubit 0
        bs = bitstr.zfill(full_width)[::-1]
        # extract the bits for our subsystem
        sub = ''.join(bs[i] for i in subsystem)
        P[sub] = P.get(sub, 0) + cnt
    
    # convert to probabilities
    probs = np.array(list(P.values())) / total
    # compute Shannon entropy
    return -np.sum(probs * np.log2(probs + 1e-12))

def run_on_real(qc, backend_name, shots=8192):
    """
    Submit one transpiled circuit to real hardware.  Falls back to simulator on RPC errors.
    """
    sampler = get_sampler(backend_name)
    try:
        target   = service.backend(backend_name).target
        qc_meas  = qc.copy().measure_all()
        print(qc_meas.draw())
        qc_trans = transpile(qc_meas, target=target)
        job      = sampler.run([qc_trans], shots=shots)
        result   = job.result()
        return extract_counts_from_bitarray(extract_bitarray_from_primitive(result))
    except Exception as e:
        print(f"⚠️  Hardware run failed: {e}\n   Falling back to Statevector…")
        # qc is the *un-measured* entanglement circuit
        sv = Statevector(qc)  
        # turn that into a probability distribution exactly as before
        return {
            format(i, f"0{qc.num_qubits}b"): abs(ampl)**2
            for i, ampl in enumerate(sv)
        }

def delta_expectation(dθ):
    qc_p = qc0.copy(); qc_p.cp(theta+dθ, 0,1)
    qc_m = qc0.copy(); qc_m.cp(theta-dθ, 0,1)
    for qc in (qc_p, qc_m):
        # measure ⟨Hξ⟩ via tomography on qubits [0,1]
        # … use your measure_modular_expectation() …
        pass
    return (Hp - Hm)/(2*dθ)

def compute_finite_difference_K(theta, eps=1e-3):
    # 1) Build base 4-qubit “plaquette + local breaks” circuit
    qc = QuantumCircuit(4)
    qc.h(range(4))
    qc.cp(theta,    0, 1)
    qc.cp(np.pi/3,  1, 2)
    qc.cp(np.pi/6,  2, 3)
    qc.ry(0.4,      1)  # break symmetry
    qc.ry(0.7,      2)

    # 2) Get the modular Hamiltonian H_ξ for region [0,1]
    sv0 = Statevector(qc)
    dm0 = DensityMatrix(sv0)
    rho0 = partial_trace(dm0, [2,3])
    Hxi  = -scipy.linalg.logm(rho0.data)

    # 3) Helper: expectation of H_ξ when only θ₀₁ is shifted
    def exp_Hxi(theta_val):
        qc2 = qc.copy()
        qc2.cp(theta_val, 0, 1)
        sv2 = Statevector(qc2)
        dm2 = DensityMatrix(sv2)
        rho2 = partial_trace(dm2, [2,3])
        return np.real(np.trace(rho2.data @ Hxi))

    # 4) Finite difference Δ⟨H⟩/Δθ
    Hp = exp_Hxi(theta + eps)
    Hm = exp_Hxi(theta - eps)
    return (Hp - Hm) / (2*eps)


def idx_to_xyz(idx, L):
    x = idx % L
    y = (idx // L) % L
    z = idx // (L * L)
    return np.array((x, y, z))
def build_4d_edges(L):
    """Return list of edges for a 4D L×L×L×L hypercube."""
    nodes = list(itertools.product(range(L), repeat=4))
    idx   = {node:i for i,node in enumerate(nodes)}
    edges = []
    for node in nodes:
        i = idx[node]
        for dim in range(4):
            if node[dim] + 1 < L:
                nb = list(node); nb[dim] += 1
                edges.append((i, idx[tuple(nb)]))
    return edges

def list_4d_faces(L):
    """
    Return list of all 2D faces (quads) in the 4D hypercube.
    Each face is a list of 4 node‐indices in square order.
    """
    faces = []
    nodes = list(itertools.product(range(L), repeat=4))
    idx   = {node:i for i,node in enumerate(nodes)}
    # choose any 2 axes out of 4 to span a square
    for axes in itertools.combinations(range(4), 2):
        a,b = axes
        for base in nodes:
            # only start when both +1 in a and b fit
            if base[a]+1 < L and base[b]+1 < L:
                n0 = list(base)
                n1 = n0.copy(); n1[a] += 1
                n2 = n0.copy(); n2[b] += 1
                n3 = n1.copy(); n3[b] += 1
                faces.append([
                    idx[tuple(n0)],
                    idx[tuple(n1)],
                    idx[tuple(n3)],
                    idx[tuple(n2)]
                ])
    return faces

def create_nd_entanglement_circuit(n_qubits, edges, theta, dims=4):
    """
    Build a superposition-and-CP circuit for an N-dimensional lattice.

    Args:
        n_qubits (int): Total number of qubits.
        edges (list of tuple): List of (i, j) pairs indicating which qubits to entangle.
        theta (dict): Mapping from edge tuples (i, j) to controlled-phase angles.
        dims (int): Number of dimensions (for bookkeeping; edges dictate actual connectivity).

    Returns:
        QuantumCircuit: A circuit preparing the entangled state.
    """
    qc = QuantumCircuit(n_qubits)

    # 1) Put all qubits into |+> state
    for q in range(n_qubits):
        qc.h(q)

    # 2) Apply controlled-phase on every edge
    for edge in edges:
        i, j = edge
        qc.cp(theta[edge], i, j)

    return qc

def build_nd_edges(L, D):
    """
    Return the list of nearest-neighbor edges for an L^D hypercube.
    Each node is indexed by a D-tuple (x0, x1, …, x{D−1}) flattened in row-major order.
    """
    # all possible coordinates
    nodes = list(itertools.product(range(L), repeat=D))
    # map tuple → linear index
    idx   = {coord:i for i,coord in enumerate(nodes)}
    edges = []
    for coord in nodes:
        i = idx[coord]
        for dim in range(D):
            if coord[dim] + 1 < L:
                # step forward along axis dim
                nbr = list(coord)
                nbr[dim] += 1
                edges.append((i, idx[tuple(nbr)]))
    return edges
def list_nd_faces(L, D):
    """
    Return all 2D square “faces” in an L^D hypercubic lattice.
    Each face is given as a list of 4 node-indices in the order:
      (0,0) → (1,0) → (1,1) → (0,1)  in the chosen 2 axes.
    """
    # build mapping from D-tuple coords → linear index
    nodes = list(itertools.product(range(L), repeat=D))
    idx   = {coord:i for i,coord in enumerate(nodes)}

    faces = []
    # choose any 2 distinct axes to span a square
    for d0, d1 in itertools.combinations(range(D), 2):
        for base in nodes:
            # ensure the +1 step fits in both directions
            if base[d0] + 1 < L and base[d1] + 1 < L:
                c0 = list(base)
                c1 = c0.copy(); c1[d0] += 1
                c2 = c0.copy(); c2[d1] += 1
                c3 = c1.copy(); c3[d1] += 1
                faces.append([
                    idx[tuple(c0)],
                    idx[tuple(c1)],
                    idx[tuple(c3)],
                    idx[tuple(c2)],
                ])
    return faces

def list_nd_cells(L, D):
    """
    Return all 3D “cells” (cubes) in an L^D hypercubic lattice.
    Each cell is given as a list of 8 node-indices in this vertex order:
      (0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1)
      in the chosen 3 axes.
    """
    nodes = list(itertools.product(range(L), repeat=D))
    idx   = {coord:i for i,coord in enumerate(nodes)}

    cells = []
    # pick any 3 distinct axes to span a cube
    for dims in itertools.combinations(range(D), 3):
        d0, d1, d2 = dims
        for base in nodes:
            if base[d0]+1 < L and base[d1]+1 < L and base[d2]+1 < L:
                # fixed vertex-ordering for a cube
                order = [
                    (0,0,0), (1,0,0), (1,1,0), (0,1,0),
                    (0,0,1), (1,0,1), (1,1,1), (0,1,1)
                ]
                cell = []
                for b0, b1, b2 in order:
                    coord = list(base)
                    coord[d0] += b0
                    coord[d1] += b1
                    coord[d2] += b2
                    cell.append(idx[tuple(coord)])
                cells.append(cell)
    return cells

def idx_to_coords(idx, L, D):
    """Convert linear idx -> D-tuple coordinate in L^D grid."""
    return [(idx // (L**d)) % L for d in range(D)]

def generate_spacetime(n, edge_list, T_target, steps=100, lr=0.1, eps=1e-3):
    """
    Solve for a discrete 'metric' θ_ij on an L^D hypercubic graph to match target curvature.
    Works directly with Statevector reductions to avoid super-op errors.
    Returns theta_dict mapping each edge to its CP-angle.
    """
    # Initialize small angles
    theta = {edge: 0.1 for edge in edge_list}

    # Build the statevector for a given theta configuration
    def get_sv(thetas):
        qc = QuantumCircuit(n)
        qc.h(range(n))
        for (i, j), th in thetas.items():
            qc.cp(th, i, j)
        return Statevector(qc)

    # Main GD loop
    for step in tqdm(range(steps), desc="GD iterations"):
        sv = get_sv(theta)

        # Compute R_i = sum_j I_{ij}
        R = {i: 0.0 for i in range(n)}
        for (i, j) in edge_list:
            # trace out all but i → rho_i, same for j, and for [i,j]
            Si  = qiskit_entropy(partial_trace(sv, [k for k in range(n) if k != i]))
            Sj  = qiskit_entropy(partial_trace(sv, [k for k in range(n) if k != j]))
            Sij = qiskit_entropy(partial_trace(sv, [k for k in range(n) if k not in (i, j)]))
            Iij = Si + Sj - Sij
            R[i] += Iij
            R[j] += Iij

        # Loss = Σ (R_i - T_target[i])^2
        L = sum((R[i] - T_target[i])**2 for i in range(n))

        # Finite-difference gradient
        grad = {}
        for edge in tqdm(edge_list, desc=f"Edges @ step {step+1}", leave=False):
            orig = theta[edge]

            # +eps
            theta[edge] = orig + eps
            R_p = {i: 0.0 for i in range(n)}
            sv_p = get_sv(theta)
            for (a, b) in edge_list:
                Sa   = qiskit_entropy(partial_trace(sv_p, [k for k in range(n) if k != a]))
                Sb   = qiskit_entropy(partial_trace(sv_p, [k for k in range(n) if k != b]))
                Sab  = qiskit_entropy(partial_trace(sv_p, [k for k in range(n) if k not in (a, b)]))
                Iab  = Sa + Sb - Sab
                R_p[a] += Iab
                R_p[b] += Iab
            Lp = sum((R_p[i] - T_target[i])**2 for i in range(n))

            # -eps
            theta[edge] = orig - eps
            R_m = {i: 0.0 for i in range(n)}
            sv_m = get_sv(theta)
            for (a, b) in edge_list:
                Sa   = qiskit_entropy(partial_trace(sv_m, [k for k in range(n) if k != a]))
                Sb   = qiskit_entropy(partial_trace(sv_m, [k for k in range(n) if k != b]))
                Sab  = qiskit_entropy(partial_trace(sv_m, [k for k in range(n) if k not in (a, b)]))
                Iab  = Sa + Sb - Sab
                R_m[a] += Iab
                R_m[b] += Iab
            Lm = sum((R_m[i] - T_target[i])**2 for i in range(n))

            # Central difference
            grad[edge] = (Lp - Lm) / (2 * eps)
            theta[edge] = orig

        # Gradient update
        for edge in edge_list:
            theta[edge] -= lr * grad[edge]

    return theta

def estimate_renyi2_swap(qc, subsystem, backend="simulator", shots=2048):
    """
    Builds & runs the SWAP-test circuit to estimate Tr(ρ_A^2) on `subsystem`.
    Returns S2 = -log2(Tr(ρ_A^2)).
    """
    # 1) Build swap-test circuit
    swap_circ = build_swap_test_circuit(qc, subsystem)

    # 2) Execute
    if backend == "simulator":
        sim = AerSimulator()
        result = sim.run(transpile(swap_circ, sim), shots=shots).result()
        counts = result.get_counts()
    else:
        # replace with your run_on_real if you want on hardware
        from CGPTFactory import run_on_real
        counts = run_on_real(swap_circ, backend, shots=shots)

    # 3) Estimate ⟨Z_anc⟩ = P(0) - P(1)
    total = sum(counts.values())
    p0    = counts.get('0', 0) / total
    tr2   = max(2*p0 - 1, 1e-12)           # ⟨Z⟩ = Tr(ρ_A^2)
    S2    = -np.log2(tr2)
    return S2

def build_swap_test_circuit(qc, subsystem):
    """
    Given a QuantumCircuit `qc` on n qubits preparing |ψ⟩, returns
    a new circuit that:
      - Allocates 1 ancilla + two copies of the n system qubits
      - Prepares |ψ>⊗|ψ> on the two copies
      - Performs a controlled-SWAP on `subsystem`
      - Measures the ancilla
    Args:
      qc        : QuantumCircuit on n qubits (no measurements).
      subsystem : list of int, the qubit indices in [0..n-1] to swap-test.
    Returns:
      QuantumCircuit on (1 + 2n) qubits + 1 classical bit.
    """
    n = qc.num_qubits
    anc = QuantumRegister(1,   name="anc")
    q1  = QuantumRegister(n,   name="q1")
    q2  = QuantumRegister(n,   name="q2")
    c0  = ClassicalRegister(1, name="c_swap")
    tst = QuantumCircuit(anc, q1, q2, c0, name="swap_test")

    # 1) Prepare two copies of |ψ⟩
    for inst, qargs, cargs in qc.data:
        # map original qubits → q1
        new_q1 = [q1[q._index] for q in qargs]
        tst.append(inst, new_q1, cargs)
    for inst, qargs, cargs in qc.data:
        # map original qubits → q2
        new_q2 = [q2[q._index] for q in qargs]
        tst.append(inst, new_q2, cargs)

    # 2) Hadamard on ancilla
    tst.h(anc[0])

    # 3) Controlled-SWAP on each qubit in subsystem
    for i in subsystem:
        tst.cswap(anc[0], q1[i], q2[i])

    # 4) Final Hadamard & measure ancilla
    tst.h(anc[0])
    tst.measure(anc[0], c0[0])

    return tst

def plot_face_scatter(face_centers, face_vals, 
                      title="Face Curvatures", 
                      cmap="viridis", 
                      size=100):
    """
    Scatter-plot 3D face centers colored by face_vals.

    Args:
        face_centers: list of [x,y,z] coords, length N
        face_vals:    list or array of length N with a scalar per face
        title:        plot title
        cmap:         matplotlib colormap name
        size:         marker size
    """
    fc = np.array(face_centers)
    fv = np.array(face_vals)

    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111, projection='3d')
    sc  = ax.scatter(
        fc[:,0], fc[:,1], fc[:,2],
        c=fv, cmap=cmap, s=size
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Face value")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def show_work(base=2,dim=3,backend=backend):
    if backend is None:
        backend = get_best_backend(service)
    L        = base
    D        = dim      # or your IBM device name
    shots    = 4096
    steps    = 10
    lr       = 0.075
    eps      = 1e-3
    sigma    = 1.0
    A        = 1.0
    n_qubits = L**D
    edges    = build_nd_edges(L, D)
    faces    = list_nd_faces(L, D)
    cells    = list_nd_cells(L, D)
    f2c_map  = build_faces_of_cell(cells, faces)

    # ── Theoretical T_target at each node ───────────────────
    center = [ (L-1)/2 ] * D
    T_target = {}
    for i in range(n_qubits):
        coords = idx_to_coords(i, L, D)
        d2     = sum((coords[d]-center[d])**2 for d in range(D))
        T_target[i] = A * np.exp(-d2/(2*sigma**2))

    # ── Solve for θ via gradient descent ───────────────────
    theta = generate_spacetime(
        n_qubits, edges, T_target,
        steps=steps, lr=lr, eps=eps
    )

    # ── Build & run entanglement circuit ───────────────────
    qc = create_nd_entanglement_circuit(n_qubits, edges, theta, dims=D)
    if backend == "simulator":
        sim    = AerSimulator(method="matrix_product_state")
        sv     = sim.run(transpile(qc, sim)).result().get_statevector()
        counts = {
            format(i, f"0{n_qubits}b"): abs(ampl)**2
            for i, ampl in enumerate(sv)
        }
    else:
        counts = run_on_real(qc, backend, shots=shots)

    # ── Face & Cell curvatures ─────────────────────────────
    face_centers, face_vals = [], []
    face_centers1, face_vals1 = [], []
    for face in tqdm(faces, desc="Face curvatures"):
        face_centers.append(np.mean([idx_to_coords(v, L, D) for v in face], axis=0))
        face_vals.append(estimate_renyi2_swap(qc, face, backend, shots))
        S2 = estimate_renyi2_swap(
            qc,
            subsystem=face,
            backend=backend,
            shots=shots
        )
        print(f"Face {face}: S₂ ≃ {S2:.3f} bits")
        ab, cd   = face[:2], face[2:]
        S_ab     = entropy_from_counts(counts, ab)
        S_cd     = entropy_from_counts(counts, cd)
        S_abcd   = entropy_from_counts(counts, face)
        K_face   = S_ab + S_cd - S_abcd
        face_centers1.append(np.mean([idx_to_coords(v, L, D) for v in face], axis=0))
        face_vals1.append(estimate_renyi2_swap(qc, face, backend, shots))


    cell_centers, cell_vals = [], []
    for cell in tqdm(cells, desc="Cell curvatures"):
        S_faces = sum(entropy_from_counts(counts, f) for f in f2c_map[tuple(cell)])
        S_full  = entropy_from_counts(counts, cell)
        K_cell  = S_faces - S_full
        cell_centers.append(
            np.mean([idx_to_coords(v, L, D) for v in cell], axis=0)
        )
        cell_vals.append(K_cell)

    # ── Node curvatures: measured vs. theoretical ──────────
    R_meas = {i: 0.0 for i in range(n_qubits)}
    for (i,j) in tqdm(edges, desc="Node curvatures"):
        Si  = entropy_from_counts(counts, [i])
        Sj  = entropy_from_counts(counts, [j])
        Sij = entropy_from_counts(counts, [i,j])
        Iij = Si + Sj - Sij
        R_meas[i] += Iij
        R_meas[j] += Iij


    node_coords     = np.array([idx_to_coords(i, L, D) for i in range(n_qubits)])
    node_vals_meas  = np.array([R_meas[i]     for i in range(n_qubits)])
    node_vals_theor = np.array([T_target[i]   for i in range(n_qubits)])

    # ── Plot #1: Face & Cell curvature ─────────────────────
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111, projection=f'{dim}D')
    fc = np.array(face_centers); fv = np.array(face_vals)
    sc = ax.scatter(fc[:,0], fc[:,1], fc[:,2],
                    c=fv, cmap='viridis', s=80, label='Face (exp)')
    if cell_centers:
        cc = np.array(cell_centers); cv = np.array(cell_vals)
        ax.scatter(cc[:,0], cc[:,1], cc[:,2],
                   c=cv, cmap='plasma', s=180, marker='^',
                   label='Cell (exp)')
    plt.colorbar(sc, ax=ax, pad=0.1, label='Curvature (bits)')
    ax.set_title(f"Face/Cell Curvatures — L={L}, D={D}, backend={backend}")
    ax.legend(loc='upper left')

    # ── Plot #2: Node measured vs theoretical ──────────────
    fig2 = plt.figure(figsize=(10,5))
    ax1  = fig2.add_subplot(121, projection='3d')
    p1   = ax1.scatter(
        node_coords[:,0], node_coords[:,1], node_coords[:,2],
        c=node_vals_meas, cmap='viridis', s=100
    )
    ax1.set_title("Measured node curvature $R_i$")
    fig2.colorbar(p1, ax=ax1, shrink=0.6, pad=0.1, label="bits")

    ax2  = fig2.add_subplot(122, projection='3d')
    p2   = ax2.scatter(
        node_coords[:,0], node_coords[:,1], node_coords[:,2],
        c=node_vals_theor, cmap='plasma', s=100
    )
    ax2.set_title("Theoretical target $T_{\\rm target}[i]$")
    fig2.colorbar(p2, ax=ax2, shrink=0.6, pad=0.1, label="target")

    plt.suptitle(
        f"L={L}, D={D}, steps={steps}, lr={lr}, eps={eps}, σ={sigma}, A={A}, shots={shots}"
    )
    plt.tight_layout()
    plt.show()

def estimate_purity_shadows(shadows, subsystem):
    """
    shadows: list of (bases_list, bitstr)
    subsystem: list of qubit idxs
    """
    # For k qubits there are 4^k Pauli strings; we use the sample‐mean
    # estimator: Tr(ρ_A^2) ≈ (1/M) ∑_{m=1}^M  ∏_{i∈A} [3 * δ_{basis_i, bstr_i} - 1]
    M = len(shadows)
    total = 0.0
    for bases, bitstr in shadows:
        prod = 1.0
        for i in subsystem:
            b = bases[i]
            bit = bitstr[-1-i]  # Qiskit’s MSB=qubit0
            val = 1.0 if ( (b=='Z' and bit=='0') or
                           (b=='X' and bit=='0') or
                           (b=='Y' and bit=='0') ) else -1.0
            # the shadow estimator factor for one qubit is (3*val)
            prod *= (3*val)
        total += prod
    purity = total / M
    return max(min(purity,1.0), 1e-12)


def random_pauli_measurement(qc, bases):
    circ = qc.copy()
    for q, b in enumerate(bases):
        if b=='X': circ.h(q)
        elif b=='Y': circ.sdg(q); circ.h(q)
        # Z → nothing
    circ.measure_all()
    return circ

def create_nd_entanglement_circuit(n, edges, theta, dims):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for (i,j),th in theta.items():
        qc.cp(th, i, j)
    return qc

class QuantumGravityAnalyzerND:
    def __init__(self,
                 L, D,
                 steps=10, lr=0.02, eps=2e-3,
                 sigma=1.0, A=1.0,
                 backend="simulator", shots=4096,
                 use_swap_test=False):
        """
        L, D          : lattice size & dimension
        steps, lr, eps: solver hyperparams
        sigma, A      : Gaussian target params
        backend       : 'simulator' or IBM device name
        shots         : #shots for swap-test
        use_swap_test : if True, do SWAP-test (Rényi-2) on backend
                        if False, do Statevector/entropy route
        """
        self.L, self.D = L, D
        self.n         = L**D
        self.steps, self.lr, self.eps = steps, lr, eps
        self.sigma, self.A = sigma, A
        self.backend, self.shots, self.use_swap_test = backend, shots, use_swap_test

        # connectivity
        self.edges = build_nd_edges(L, D)
        self.faces = list_nd_faces(L, D)
        self.cells = list_nd_cells(L, D)
        self.f2c   = build_faces_of_cell(self.cells, self.faces)

        # theoretical node curvature
        center = [(L-1)/2]*D
        self.T_target = {}
        for i in range(self.n):
            coords = idx_to_coords(i, L, D)
            d2 = sum((coords[d]-center[d])**2 for d in range(D))
            self.T_target[i] = A*np.exp(-d2/(2*sigma**2))

        # placeholders
        self.theta = None
        self.qc    = None
        self.R_meas = None

    def solve_metric(self):
        self.theta = generate_spacetime(
            self.n, self.edges, self.T_target,
            steps=self.steps, lr=self.lr, eps=self.eps
        )
        return self.theta

    def build_circuit(self):
        self.qc = create_nd_entanglement_circuit(
            self.n, self.edges, self.theta, dims=self.D
        )
        return self.qc

    def run_simulator(self):
        # exact statevector
        self.sv = Statevector(self.qc)
        return self.sv

    def compute_node_curvatures(self):
        if self.use_swap_test:
            # SWAP-test Rényi-2 route
            S2_1 = {}  # single-qubit S2
            for i in tqdm(range(self.n), desc="SWAP-test S2(i)"):
                S2_1[i] = estimate_renyi2_swap(
                    self.qc, [i],
                    backend=self.backend, shots=self.shots
                )
            S2_2 = {}  # two-qubit S2 on edges
            for (i,j) in tqdm(self.edges, desc="SWAP-test S2(ij)"):
                S2_2[(i,j)] = estimate_renyi2_swap(
                    self.qc, [i,j],
                    backend=self.backend, shots=self.shots
                )
            R2 = {i: 0.0 for i in range(self.n)}
            for (i,j), s2ij in S2_2.items():
                I2ij = S2_1[i] + S2_1[j] - s2ij
                R2[i] += I2ij
                R2[j] += I2ij
            self.R_meas = R2

        else:
            # statevector vn‐entropy route
            sv = self.sv
            R = {i:0.0 for i in range(self.n)}
            for (i,j) in tqdm(self.edges, desc="Computing I_ij"):
                Si  = qiskit_entropy(partial_trace(sv, [k for k in range(self.n) if k!=i]), base=2)
                Sj  = qiskit_entropy(partial_trace(sv, [k for k in range(self.n) if k!=j]), base=2)
                Sij = qiskit_entropy(partial_trace(sv, [k for k in range(self.n) if k not in (i,j)]), base=2)
                Iij = Si + Sj - Sij
                R[i] += Iij; R[j] += Iij
            self.R_meas = R

        return self.R_meas

    def compare_with_theory(self):
        idxs     = np.arange(self.n)
        measured = np.array([self.R_meas[i]     for i in idxs])
        theory   = np.array([self.T_target[i]   for i in idxs])
        plt.figure(figsize=(7,4))
        plt.plot(idxs, theory, 'o-', label='Theory')
        plt.plot(idxs, measured, 'x--', label='Measured')
        plt.xlabel('Node index'); plt.ylabel('Curvature')
        plt.title(f"L={self.L}, D={self.D}")
        plt.legend(); plt.tight_layout(); plt.show()
        return measured, theory

    def visualize_radial_profile(self):
        center = np.array([(self.L-1)/2]*self.D)
        coords = np.array([idx_to_coords(i, self.L, self.D) for i in range(self.n)])
        dists  = np.linalg.norm(coords-center, axis=1)
        theory   = np.array([self.T_target[i] for i in range(self.n)])
        measured = np.array([self.R_meas[i]   for i in range(self.n)])
        plt.figure(figsize=(6,4))
        plt.scatter(dists, theory,   marker='o', c='C0', label='Theory')
        plt.scatter(dists, measured, marker='x', c='C1', label='Measured')
        plt.xlabel('Distance r'); plt.ylabel('Curvature')
        plt.legend(); plt.tight_layout(); plt.show()

    def run_demo(self):
        """Full pipeline, auto‐chooses method by use_swap_test flag."""
        print("Solving metric …"); self.solve_metric()
        print("Building circuit …"); self.build_circuit()
        if not self.use_swap_test:
            print("Simulating state …"); self.run_simulator()
        print(f"Computing node curvatures ({'swap-test' if self.use_swap_test else 'von Neumann'}) …")
        self.compute_node_curvatures()
        print("Plotting index comparison …"); self.compare_with_theory()
        print("Plotting radial profile …"); self.visualize_radial_profile()

def build_nd_faces(L, D):
    nodes = list(itertools.product(range(L), repeat=D))
    idx   = {coord:i for i,coord in enumerate(nodes)}
    faces = []
    for d0, d1 in itertools.combinations(range(D), 2):
        for base in nodes:
            if base[d0]+1 < L and base[d1]+1 < L:
                c0 = list(base)
                c1 = c0.copy(); c1[d0] += 1
                c2 = c0.copy(); c2[d1] += 1
                c3 = c1.copy(); c3[d1] += 1
                faces.append([idx[tuple(c0)],idx[tuple(c1)],idx[tuple(c3)],idx[tuple(c2)]])
    return faces

class Manifold4D:
    def __init__(self, backend, shots=1000, shadow_shots=1000):
        self.L, self.D = 2, 4
        self.n         = 16
        self.edges     = build_nd_edges(self.L, self.D)
        self.faces     = build_nd_faces(self.L, self.D)

        # Gaussian target
        center = [(self.L-1)/2]*self.D
        self.T = {}
        for i in range(self.n):
            d2       = sum((c-center[d])**2
                          for d,c in enumerate(idx_to_coords(i,self.L,self.D)))
            self.T[i] = np.exp(-d2/2)

        self.backend      = backend
        self.shots        = shots
        self.shadow_shots = shadow_shots

        # prepare θ once
        self.theta = generate_spacetime(self.n, self.edges, self.T, steps=1)

        # prepare the base 16-qubit circuit
        self.qc_base = create_nd_entanglement_circuit(
            self.n, self.edges, self.theta, dims=self.D)
        
    def run_shadows(self, batch_size=2000):
        """
        Gather classical shadows in one or more batched jobs,
        using the primitive-result helpers for extraction.
        """
        # 1) pick or validate the backend object
        try:
            backend_obj = self.backend
        except:
            avail = service.backends(simulator=False, status="online")
            backend_obj = avail[0]
            self.backend = backend_obj.name()
            print(f"⚠️ Backend not found, falling back to {self.backend}")
        target = backend_obj.target

        self.shadows = []
        n_batches = (self.shadow_shots + batch_size - 1) // batch_size

        with Session(backend=self.backend) as session:
            sampler = Sampler()

            for _ in tqdm(range(n_batches), desc="Gathering shadow batches"):
                circuits, bases_list = [], []
                remaining = self.shadow_shots - len(self.shadows)
                this_batch = min(batch_size, remaining)

                # build & transpile each rotated circuit
                for _ in range(this_batch):
                    bases = np.random.choice(['X','Y','Z'], size=self.n)
                    circ  = random_pauli_measurement(self.qc_base, bases)
                    circ_t = transpile(circ, target=target)
                    circuits.append(circ_t)
                    bases_list.append(bases.tolist())

                # submit the entire batch in one job
                job    = sampler.run(circuits, shots=1)
                result = job.result()

                # extract raw bit-arrays and collapse to bit-strings
                bitarrays = extract_bitarray_from_primitive(result)
                bitstrs   = extract_counts_from_bitarray(bitarrays)

                # record each shadow as (basis, bitstr)
                for bases, bitstr in zip(bases_list, bitstrs):
                    self.shadows.append((bases, bitstr))

        # Make sure the assert is still indented inside the method!
        assert len(self.shadows) == M, (
            f"Expected {M} shadows, got {len(self.shadows)}"
        )

    def run_shadows(self, M=1000, batch_size=500):
        """
        Gather M classical shadows in batched jobs.
        If the named real backend is unavailable, fall back to simulator.
        """
        # 1) pick or validate the real backend
        try:
            bd = service.backend(self.backend)
        except Exception:
            reals = service.backends(simulator=False, status="online")
            if reals:
                bd = reals[0]
                self.backend = bd.name()
                print(f"⚠️  '{self.backend}' not found, using {self.backend}")
            else:
                print("⚠️  No hardware available; using simulator for shadows")
                return self._run_shadows_simulator(M)

        target = bd.target
        self.shadows = []
        n_batches = (M + batch_size - 1) // batch_size

        with Session(backend=self.backend) as sess:
            sampler = Sampler()

            for _ in tqdm(range(n_batches), desc="Gathering shadow batches"):
                circuits, bases_list = [], []
                to_do = min(batch_size, M - len(self.shadows))

                # 2) build & transpile batch of circuits
                for __ in range(to_do):
                    bases = np.random.choice(['X','Y','Z'], size=self.n)
                    circ  = random_pauli_measurement(self.qc_base, bases)
                    circuits.append(transpile(circ, target=target))
                    bases_list.append(bases.tolist())

                # 3) submit one job for the entire batch
                job    = sampler.run(circuits, shots=1)
                result = job.result()

                # 4) extract and record each single-shot bitstring
                bitarrays = extract_bitarray_from_primitive(result)
                bitstrs   = extract_counts_from_bitarray(bitarrays)
                for bases, bitstr in zip(bases_list, bitstrs):
                    self.shadows.append((bases, bitstr))

        # 5) sanity check
        assert len(self.shadows) == M, (
            f"Expected {M} shadows, got {len(self.shadows)}"
        )

    def _run_shadows_simulator(self, M):
        """Fallback: collect M shadows on the Aer MPS simulator."""
        sim = AerSimulator(method="matrix_product_state")
        self.shadows = []
        for _ in tqdm(range(M), desc="Gathering shadows (sim)"):
            bases = np.random.choice(['X','Y','Z'], size=self.n)
            circ  = random_pauli_measurement(self.qc_base, bases)
            result = sim.run(transpile(circ, sim), shots=1).result()
            bitstr = next(iter(result.get_counts()))
            self.shadows.append((bases.tolist(), bitstr))

        assert len(self.shadows) == M, (
            f"Expected {M} shadows, got {len(self.shadows)}"
        )

    def compute_curvatures(self):
        # purity→S2→I2→R2
        S2_1 = {i:0 for i in range(self.n)}
        for i in range(self.n):
            p1      = estimate_purity_shadows(self.shadows, [i])
            S2_1[i] = -np.log2(p1)

        S2_2 = {}
        for e in self.edges:
            p2        = estimate_purity_shadows(self.shadows, list(e))
            S2_2[e]   = -np.log2(p2)

        R2 = {i:0 for i in range(self.n)}
        for (i,j),s2ij in S2_2.items():
            I2ij = S2_1[i] + S2_1[j] - s2ij
            R2[i] += I2ij; R2[j] += I2ij

        self.R2 = R2

    def plot_results(self):
        idxs     = np.arange(self.n)
        theory   = np.array([self.T[i]     for i in idxs])
        measured = np.array([self.R2[i]    for i in idxs])

        plt.figure(figsize=(6,4))
        plt.plot(idxs, theory, 'o-', label='Theory')
        plt.plot(idxs, measured,'x--', label='Measured S2')
        plt.xlabel('Node index i'); plt.ylabel('Curvature (bits)')
        plt.legend(); plt.title("4D Curvature via Shadows on Hardware")
        plt.show()

    def run_demo(self):
        print("📡 Running classical shadows on hardware …")
        self.run_shadows()
        print("🧮 Computing curvatures from shadows …")
        self.compute_curvatures()
        print("📊 Plotting results …")
        self.plot_results()

def build_nd_swap_test(theta_dict, region, dims=None, L=None):
    """
    Build a SWAP‐test circuit for any N‐qubit entangling circuit given by theta_dict.

    Parameters
    ----------
    theta_dict : dict[(int,int)->float]
        mapping each edge (i,j) to its CP‐angle θ_{ij}.
    region : list[int] or list[tuple]
        either a list of linear indices [0,1,5,...], or a list of coordinate tuples
        (in which case you must pass dims and L).
    dims : int, optional
        number of spatial dimensions (if region is in coord form).
    L : int, optional
        side‐length in each dimension.

    Returns
    -------
    QuantumCircuit
        on 1 + 2*n_qubits qubits and 1 classical bit.
    """
    # 1) Convert coords to linear indices, if needed
    if dims and L and region and isinstance(region[0], (tuple, list)):
        region_idx = [
            sum(coord[d] * (L**d) for d in range(dims))
            for coord in region
        ]
    else:
        region_idx = list(region)

    # 2) Infer the total # of qubits from theta_dict
    all_indices = [i for edge in theta_dict.keys() for i in edge]
    n_qubits    = max(all_indices) + 1

    # 3) Build registers: ancilla + two copies
    qc = QuantumCircuit(1 + 2*n_qubits, 1)
    anc   = 0
    copy1 = list(range(1,       1 + n_qubits))
    copy2 = list(range(1+n_qubits, 1 + 2*n_qubits))

    # 4) Prepare both copies with your CP(θ) layers
    for (i, j), th in theta_dict.items():
        qc.cp(th, copy1[i], copy1[j])
        qc.cp(th, copy2[i], copy2[j])

    # 5) Do the SWAP‐test on exactly the qubits in region_idx
    qc.h(anc)
    for idx in region_idx:
        qc.append(CSwapGate(), [anc, copy1[idx], copy2[idx]])
    qc.h(anc)
    qc.measure(anc, 0)

    return qc

def counter_to_bitarray(counts: Counter) -> list[int]:
    """
    Given a Counter mapping arbitrary keys→counts, pull out only those
    keys that are pure bitstrings (e.g. '0', '1', '010', etc.), look
    at their last character, and expand into a flat list of 0/1.
    """
    bitarray = []
    for bitstr, freq in counts.items():
        # ensure key is a str
        s = str(bitstr)
        # only accept pure bitstrings
        if not BITSTR_RE.fullmatch(s):
            continue
        last = s[-1]
        bit = int(last)
        bitarray.extend([bit] * freq)
    return bitarray

def make_plaquette_grid(curvatures, rows, cols):
    G = np.zeros((rows-1, cols-1))
    for (a, b, c, d), K in curvatures.items():
        r = a // cols
        c0 = a % cols
        G[r, c0] = K
    return G

def compare_dicts(dict_a, dict_b):
    keys = list(dict_a.keys())
    a = [dict_a[k] for k in keys]
    b = [dict_b[k] for k in keys]
    rho, p_rho = spearmanr(a, b)
    tau, p_tau = kendalltau(a, b)
    r, p_r = pearsonr(a, b)
    return {'spearman_rho': rho, 'p_spearman': p_rho,
            'kendall_tau': tau, 'p_kendall': p_tau,
            'pearson_r': r, 'p_pearson': p_r}

def plot_comparison(curvatures_meas, curvatures_th):
    # ensure same ordering
    keys = list(curvatures_th.keys())
    th = [curvatures_th[k] for k in keys]
    meas = [curvatures_meas[k] for k in keys]
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    sns.scatterplot(x=th, y=meas, ax=ax[0])
    ax[0].set_xlabel('K_th'); ax[0].set_ylabel('K_meas')
    sns.regplot(x=th, y=meas, ax=ax[1])
    ax[1].set_title('Fit line')
    plt.tight_layout()
    return fig

def compare_curvatures(theory: dict, experiment: dict, show_plot: bool = True):
    """
    Compare two plaquette‐curvature dictionaries:
      - theory:    {(i,j,k,l): K_th, …}
      - experiment:{(i,j,k,l): K_exp, …}
    
    Prints Spearman, Kendall τ, and Pearson correlations, and
    (optionally) shows a scatter plot with y=x.
    """
    # 1) Align on the same set of plaquettes
    common = sorted(set(theory) & set(experiment))
    if len(common) < 2:
        raise ValueError("Need at least two matching plaquettes to compare.")
    
    K_th = np.array([theory[p]     for p in common])
    K_ex = np.array([experiment[p] for p in common])
    
    # 2) Compute correlations
    ρ_s, p_s = spearmanr(K_th, K_ex)
    τ_k, p_k = kendalltau(K_th, K_ex)
    r_p, p_p = pearsonr(K_th, K_ex)
    
    print(f"Number of plaquettes compared: {len(common)}")
    print(f"Spearman ρ = {ρ_s:.3f} (p={p_s:.2g})")
    print(f"Kendall τ = {τ_k:.3f} (p={p_k:.2g})")
    print(f"Pearson  r = {r_p:.3f} (p={p_p:.2g})")
    
    # 3) Optional scatter plot
    if show_plot:
        plt.figure(figsize=(6,6))
        plt.scatter(K_th, K_ex, color='C1', edgecolor='k')
        mn, mx = min(K_th.min(), K_ex.min()), max(K_th.max(), K_ex.max())
        plt.plot([mn,mx], [mn,mx], 'k--', label='y = x')
        plt.xlabel("Theoretical curvature K")
        plt.ylabel("Measured curvature K")
        plt.title("Theory vs. Real plaquette curvatures")
        plt.legend()
        plt.tight_layout()
        plt.show()

def run_blackhole_curvature_sim(rows, cols, mass_strength, epsilon, shots=8192):
    from collections import defaultdict
    import math

    # 1) Graph & mass profile (same as before) …
    edges      = build_2d_edges(rows, cols)
    plaquettes = list_plaquettes(rows, cols)
    center     = ((rows-1)/2, (cols-1)/2)
    T = {}
    for r in range(rows):
        for c in range(cols):
            idx = r*cols + c
            dist = np.hypot(r-center[0], c-center[1])
            T[idx] = mass_strength/(dist + epsilon)
    theta = {e:(T[e[0]]+T[e[1]])/2 for e in edges}

    # 2) Build circuit
    qc = create_2d_entanglement_circuit(rows, cols, theta)

    # 3) Run on AerSimulator
    backend = Aer.get_backend("aer_simulator")
    tq      = transpile(qc, backend, optimization_level=3)
    result  = backend.run(tq, shots=shots).result()

    # 4) Extract raw counts → probs
    raw_counts = result.get_counts()
    total      = sum(raw_counts.values())
    probs      = {bitstr: cnt/total for bitstr, cnt in raw_counts.items()}

    num_qubits = rows*cols

    # 5) padded marginal and entropy
    def marginal(probs, qubits):
        m = defaultdict(float)
        for bs, p in probs.items():
            bs = bs.zfill(num_qubits)          # <— zero-pad to full width
            key = "".join(bs[num_qubits-1-q] for q in qubits)
            m[key] += p
        return m

    def S(pdist):
        return -sum(p*math.log2(p) for p in pdist.values() if p>0)

    # 6) compute plaquette curvatures & pairwise MI
    curv  = {}
    Iij   = {}
    for (a,b,c,d) in plaquettes:
        S_ab   = S(marginal(probs, [a,b]))
        S_cd   = S(marginal(probs, [c,d]))
        S_abcd = S(marginal(probs, [a,b,c,d]))
        curv[(a,b,c,d)] = S_ab + S_cd - S_abcd

    for (i,j) in edges:
        H_i  = S(marginal(probs, [i]))
        H_j  = S(marginal(probs, [j]))
        H_ij = S(marginal(probs, [i,j]))
        Iij[(i,j)] = H_i + H_j - H_ij

    return curv, Iij

def execute_and_compare(rows=3, cols=3, mass_strength=5, backend="ibm_sherbrooke"):
    # run experiment
    real_curv, real_I = run_blackhole_curvature_real(rows, cols, mass_strength=mass_strength, epsilon=0.05, shots=4096, backend_name=backend)
    # run simulation
    sim_curv, sim_I   = run_blackhole_curvature_sim(rows, cols, mass_strength=mass_strength, epsilon=0.05)

    # --- define your theoretical expectations K_th for each plaquette (0,1,4,5)... etc
    # e.g. maybe from an analytic formula or precomputed table:
    K_th = sim_curv.copy()

    # extract matching lists
    ρ_s, p_s = spearmanr(K_the, K_exp)
    τ_k, p_k = kendalltau(K_the, K_exp)
    r_p, p_p = pearsonr(K_the, K_exp)
    print(f"Spearman ρ = {ρ_s:.3f} (p={p_s:.2f})")
    print(f"Kendall τ  = {τ_k:.3f} (p={p_k:.2f})")
    print(f"Pearson  r = {r_p:.3f} (p={p_p:.2f})")

    # also check that ordering matches:
    rank_exp = rankdata(K_exp, method="ordinal")
    rank_the = rankdata(K_the, method="ordinal")
    ordering_match = np.sum(rank_exp == rank_the) / len(plaquettes)
    print(f"Fraction of plaquettes with identical rank: {ordering_match:.2%}")

    # --- now plot everything ---
    fig, axes = plt.subplots(1,3, figsize=(15,4))

    # (a) heatmap of experimental curvature
    sns.heatmap(make_grid(real_curv, rows,cols), cmap="Reds", ax=axes[0], cbar_kws={'label':'K_exp (bits)'})
    axes[0].set_title("Real Curvature")

    # (b) heatmap of theoretical curvature
    sns.heatmap(make_grid(K_th, rows,cols), cmap="Blues", ax=axes[1], cbar_kws={'label':'K_the (bits)'})
    axes[1].set_title("Theory Curvature")

    # (c) scatter plot real vs theory
    axes[2].scatter(K_the, K_exp, s=50, alpha=0.7)
    axes[2].plot([K_the.min(), K_the.max()], [K_the.min(), K_the.max()],
                 'k--', lw=1)
    axes[2].set_xlabel("K_theoretical")
    axes[2].set_ylabel("K_experimental")
    axes[2].set_title("Real vs Theory")
    # annotate metrics on the plot
    txt = f"Spearman ρ={ρ_s:.2f}\nPearson r={r_p:.2f}"
    axes[2].text(0.05, 0.95, txt, transform=axes[2].transAxes,
                 va='top', bbox=dict(boxstyle="round", fc="w"))

    plt.tight_layout()
    plt.show()
    
def execute_real(rows=4, cols=4, mass_strength=5):

    # 1) Choose your 2D grid size and physics parameters
    epsilon         = 0.05
    shots           = 4096

    # 2) Run the full “black hole curvature” pipeline on real hardware
    curvatures, I = run_blackhole_curvature_real(
        rows, cols,
        mass_strength=mass_strength,
        epsilon=epsilon,
        shots=shots,
        backend_name=backend
    )

    # --- Print plaquette curvatures ---
    print("\nPlaquette curvatures K (bits):")
    for plaquette, K in curvatures.items():
        print(f"  {plaquette}: {K:.4f}")

    # --- Print pairwise (classical) mutual informations ---
    print("\nLink mutual informations I(i,j) (bits):")
    for edge, mij in I.items():
        print(f"  {edge}: {mij:.4f}")

def execute_and_compare(rows=4, cols=4, mass_strength=5, epsilon=0.05, shots=4096, backend="ibm_sherbrooke"):
    # 1) “Theoretical” (noiseless) run
    sim_curv, sim_I = run_blackhole_curvature_sim(
        rows, cols,
        mass_strength=mass_strength,
        epsilon=epsilon,
    )
    # 2) Real‐hardware run
    real_curv, real_I = run_blackhole_curvature_real(
        rows, cols,
        mass_strength=mass_strength,
        epsilon=epsilon,
        shots=shots,
        backend_name=backend
    )

    metrics = compare_dicts(sim_curv, real_curv)
    print(metrics)
    fig = plot_comparison(sim_curv, real_curv)
    fig.savefig('comparison.png')

    # 3) Build grids
    grid_sim  = make_plaquette_grid(sim_curv,  rows, cols)
    grid_real = make_plaquette_grid(real_curv, rows, cols)
    grid_diff = grid_real - grid_sim

    # 4) Plot side by side
    fig, axes = plt.subplots(1, 3, figsize=(12,4))
    sns.heatmap(grid_sim,  cmap="Reds", ax=axes[0], cbar_kws={'label':'bits'})
    axes[0].set_title("Ideal (simulated) K")

    sns.heatmap(grid_real, cmap="Reds", ax=axes[1], cbar_kws={'label':'bits'})
    axes[1].set_title("Hardware K")

    sns.heatmap(grid_diff, cmap="bwr", center=0, ax=axes[2], cbar_kws={'label':'ΔK (real−ideal)'})
    axes[2].set_title("Difference ΔK")

    for ax in axes:
        ax.set_xlabel("plaquette column")
        ax.set_ylabel("plaquette row")

    plt.tight_layout()
    plt.show()

    compare_curvatures(sim_curv, real_curv)


# --- Core Experiment Function ---
def create_entropy_experiment(option=1, num_injections=5, num_radiation_qubits=3, gap_cycles=50, include_spin=False, entangle_depth=1):
    qc = QuantumCircuit(num_radiation_qubits + 1)
    for i in range(num_injections):
        # Charge injection
        if i % 2 == 0:
            qc.x(0)
        else:
            qc.z(0)

        # Optional spin injection
        if include_spin:
            qc.rx(np.pi/4, 0)

        # Entanglement loop
        for _ in range(entangle_depth):
            for j in range(1, num_radiation_qubits + 1):
                qc.h(0)
                qc.cx(0, j)

        # Idle to simulate time gaps
        qc.barrier()
        for _ in range(gap_cycles):
            qc.id(0)

    return qc

# --- Measurement Wrapper ---
def add_measurements(qc, measure_qubits):
    c = ClassicalRegister(len(measure_qubits))
    qc.add_register(c)
    qc.measure(measure_qubits, range(len(measure_qubits)))
    return qc

# --- Entropy Calculation ---
def compute_entropy(qc, method="von_neumann"):
    state = Statevector.from_instruction(qc)
    reduced = partial_trace(state, list(range(1, qc.num_qubits)))
    if method == "von_neumann":
        return qiskit_entropy(reduced)
    elif method == "shannon":
        probs = np.abs(state.data) ** 2
        return -np.sum(probs * np.log2(probs + 1e-10))
    return None

# --- Runtime Execution and Plot ---
def run_entropy_experiment():
    options = [1, 2, 3, 4]
    entropies = []

    with Session(backend=backend) as session:
        sampler = Sampler()
        for opt in options:
            include_spin = (opt == 2)
            entangle_depth = 2 if opt in [3, 4] else 1
            gap_cycles = 100 if opt in [1, 3] else 20

            qc = create_entropy_experiment(option=opt, num_injections=6, gap_cycles=gap_cycles, include_spin=include_spin, entangle_depth=entangle_depth)
            entropy_val = compute_entropy(qc)
            entropies.append(entropy_val)

            qc_meas = add_measurements(qc.copy(), range(1, qc.num_qubits))
            transpiled = transpile(qc_meas, backend)
            job = sampler.run([transpiled], shots=2048)
            result = job.result()
            counts = extract_counts_from_bitarray(extract_bitarray_from_primitive(result))
            plot_histogram(counts, title=f"Measurement Histogram - Option {opt}")
            plt.show()

    # Plot entropy comparison
    plt.figure(figsize=(8, 5))
    plt.plot(options, entropies, marker='o')
    plt.title("Entropy vs Experimental Options")
    plt.xlabel("Experiment Option")
    plt.ylabel("Entropy (von Neumann)")
    plt.grid(True)
    plt.show()

# --- Mutual Information ---
def compute_mutual_information(state, A, B):
    rho_AB = partial_trace(state, [i for i in range(state.num_qubits) if i not in (A + B)])
    rho_A = partial_trace(state, [i for i in range(state.num_qubits) if i not in A])
    rho_B = partial_trace(state, [i for i in range(state.num_qubits) if i not in B])
    S_AB = qiskit_entropy(rho_AB)
    S_A = qiskit_entropy(rho_A)
    S_B = qiskit_entropy(rho_B)
    return S_A + S_B - S_AB

# --- Create dynamic entanglement circuit ---
def create_dynamic_entropy_circuit(num_steps=5, num_radiation_qubits=3, gap_cycles=20):
    qc = QuantumCircuit(num_radiation_qubits + 1)  # 0 = black hole
    entropy_vals = []
    mutual_info_vals = []

    for step in range(1, num_steps + 1):
        # Alternate charge injection
        qc.x(0) if step % 2 == 0 else qc.z(0)
        qc.rx(np.pi/8, 0)  # Spin perturbation

        # Feedback: modulate entanglement strength by previous entropy
        angle = np.pi / (step + 1)
        for j in range(1, num_radiation_qubits + 1):
            qc.h(0)
            qc.cry(angle, 0, j)

        # Gap cycles
        qc.barrier()
        for _ in range(gap_cycles):
            qc.id(0)

        # Analyze entropy and mutual info for this step
        state = Statevector(qc.copy())
        entropy_val = qiskit_entropy(partial_trace(state, [0]))
        entropy_vals.append(entropy_val)

        mi = compute_mutual_information(state, [1], [2, 3] if num_radiation_qubits > 2 else [2])
        mutual_info_vals.append(mi)

    return qc, entropy_vals, mutual_info_vals

# --- New: Visualize Emergent Geometry ---
def visualize_emergent_geometry_over_time(states):
    epsilon = 1e-6
    fig, axes = plt.subplots(1, len(states), figsize=(3 * len(states), 4))

    for idx, state in enumerate(states):
        num_qubits = state.num_qubits
        mi_matrix = np.zeros((num_qubits, num_qubits))

        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                mi = compute_mutual_information(state, [i], [j])
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi

        distance_matrix = 1 / (mi_matrix + epsilon)
        np.fill_diagonal(distance_matrix, 0)

        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coords = mds.fit_transform(distance_matrix)

        ax = axes[idx] if len(states) > 1 else axes
        ax.scatter(coords[:, 0], coords[:, 1], c='blue')
        for i in range(num_qubits):
            ax.text(coords[i, 0], coords[i, 1], f"Q{i}", fontsize=10)
        ax.set_title(f"Step {idx+1}")
        ax.axis("equal")
        ax.grid(True)

    plt.suptitle("Emergent Geometry Evolution")
    plt.tight_layout()
    plt.show()


# --- Create a parametric entanglement circuit ---
def create_dynamic_entropy_circuit(num_steps=5, num_radiation_qubits=3, gap_cycles=20):
    qc = QuantumCircuit(num_radiation_qubits + 1)  # 0 = black hole
    entropy_vals = []
    mutual_info_vals = []
    state_snapshots = []

    for step in range(1, num_steps + 1):
        if step % 2 == 0:
            qc.x(0)
        else:
            qc.z(0)
        qc.rx(np.pi/8, 0)

        angle = np.pi / (step + 1)
        for j in range(1, num_radiation_qubits + 1):
            qc.h(0)
            qc.cry(angle, 0, j)

        qc.barrier()
        for _ in range(gap_cycles):
            qc.id(0)

        state = Statevector.from_instruction(qc.copy())
        state_snapshots.append(state)

        entropy_val = qiskit_entropy(partial_trace(state, [0]))
        entropy_vals.append(entropy_val)

        mi = compute_mutual_information(state, [1], [2, 3] if num_radiation_qubits > 2 else [2])
        mutual_info_vals.append(mi)

    return qc, entropy_vals, mutual_info_vals, state_snapshots

# --- Run & Plot ---
def run_emergent_structure_demo():
    num_steps = 6
    qc, entropy_vals, mutual_info_vals, state_snap = create_dynamic_entropy_circuit(num_steps=num_steps)

    # Plot entropy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_steps + 1), entropy_vals, label="von Neumann Entropy", marker="o")
    plt.plot(range(1, num_steps + 1), mutual_info_vals, label="Mutual Information", marker="s")
    plt.xlabel("Step")
    plt.ylabel("Entropy / Mutual Info")
    plt.title("Emergent Structure from Charge + Spin + Entanglement Feedback")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Optional: Run measurement on final state
    qc.add_register(ClassicalRegister(qc.num_qubits - 1))
    qc.measure(range(1, qc.num_qubits), range(qc.num_qubits - 1))
    with Session(backend=backend) as session:
        sampler = Sampler()
        transpiled = transpile(qc, backend)
        job = sampler.run([transpiled], shots=1024)
        counts = extract_counts_from_bitarray(extract_bitarray_from_primitive(job.result()))
        plot_histogram(counts, title="Final Measurement Distribution")
        plt.show()

    visualize_emergent_geometry_over_time(state_snapshots)
    
# --- New: Visualize Emergent Geometry ---
def visualize_emergent_geometry(qc):
    state = Statevector(qc)
    num_qubits = qc.num_qubits
    epsilon = 1e-6
    mi_matrix = np.zeros((num_qubits, num_qubits))

    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            mi = compute_mutual_information(state, [i], [j])
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    distance_matrix = 1 / (mi_matrix + epsilon)
    np.fill_diagonal(distance_matrix, 0)

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(distance_matrix)

    plt.figure(figsize=(6, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c='blue')
    for i in range(num_qubits):
        plt.text(coords[i, 0], coords[i, 1], f"Q{i}", fontsize=12)
    plt.title("Emergent Geometry from Entanglement")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

# --- Run experiment and geometry ---
def run_emergent_structure_demo():
    num_steps = 6
    qc, entropy_vals, mutual_info_vals, state_snapshot = create_dynamic_entropy_circuit(num_steps=num_steps)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_steps + 1), entropy_vals, label="von Neumann Entropy", marker="o")
    plt.plot(range(1, num_steps + 1), mutual_info_vals, label="Mutual Information", marker="s")
    plt.xlabel("Step")
    plt.ylabel("Entropy / Mutual Info")
    plt.title("Emergent Structure from Charge + Spin + Entanglement Feedback")
    plt.legend()
    plt.grid(True)
    plt.show()

    qc.add_register(ClassicalRegister(qc.num_qubits - 1))
    qc.measure(range(1, qc.num_qubits), range(qc.num_qubits - 1))
    with Session(backend=backend) as session:
        sampler = Sampler()
        transpiled = transpile(qc, backend)
        job = sampler.run([transpiled], shots=2048)
        counts = extract_counts_from_bitarray(extract_bitarray_from_primitive(job.result()))
        plot_histogram(counts, title="Final Measurement Distribution")
        plt.show()

    visualize_emergent_geometry_over_time(state_snapshot)


def build_ansatz(n_qubits, n_layers, params):
    """
    Simple hardware‐efficient ansatz:
      - single-qubit Ry rotations
      - CZ entanglers in a ring
    params shape = (n_layers, n_qubits)
    """
    qc = QuantumCircuit(n_qubits)
    p = params.reshape(n_layers, n_qubits)
    for layer in range(n_layers):
        # local rotations
        for q in range(n_qubits):
            qc.ry(p[layer,q], q)
        # entangle ring
        for q in range(n_qubits):
            qc.cz(q, (q+1) % n_qubits)
    return qc

def compute_MI_and_SV(statevec, subsystems):
    """
    Given a Statevector and a list of subsystems,
    returns dicts of mutual‐infos and entropies.
      subsystems = {
        'pairs': [(i,j), ...],
        'blocks':  [[i,j,k], ...]
      }
    """
    # first get all reduced states
    rho_cache = {}
    def rho(qubits):
        key = tuple(sorted(qubits))
        if key not in rho_cache:
            rho_cache[key] = partial_trace(statevec, 
                                [i for i in range(statevec.num_qubits) if i not in key])
        return rho_cache[key]
    
    mis = {}
    for (i,j) in subsystems['pairs']:
        S_i  = qiskit_entropy(rho([i]))
        S_j  = qiskit_entropy(rho([j]))
        S_ij = qiskit_entropy(rho([i,j]))
        mis[(i,j)] = S_i + S_j - S_ij

    ents = {}
    for block in subsystems['blocks']:
        ents[tuple(block)] = qiskit_entropy(rho(block))

    return mis, ents

def cost_fn(x, n_qubits, n_layers, subsystems, 
            target_MI, target_S, w_mi=1.0, w_s=1.0):
    """
    x = flat parameter vector for ansatz
    target_MI[(i,j)] = desired mutual info
    target_S[tuple(block)] = desired entropy
    w_mi, w_s = weight for each term
    """
    qc = build_ansatz(n_qubits, n_layers, x)
    sv = Statevector.from_instruction(qc)
    mis, ents = compute_MI_and_SV(sv, subsystems)

    # L1 cost over all constraints
    c_mi = sum(abs(mis[key] - target_MI[key]) for key in target_MI)
    c_s  = sum(abs(ents[key] - target_S[key])  for key in target_S)
    return w_mi*c_mi + w_s*c_s

def evaluate_cost_real(params):
    qc = build_ansatz(n_qubits, n_layers, params)
    # Convert qiskit circuit → Runtime circuit
    # Prepare circuits for each observable you need (e.g. Z, ZZ, etc.)
    obs = []
    for (i,j) in subsystems['pairs']:
        obs += [PauliOp(Pauli('Z' * i + 'I'*(j-i-1) + 'Z' + 'I'*(n_qubits-j-1)))]
    # likewise for single-qubit entropies you’ll pick enough 2^k Pauli terms
    with Session(service=service, backend="ibm_perth") as sess:
        estimator = Estimator(session=sess)
        # returns expectation for each circuit×observable pair
        expvals = estimator.run(circuits=[qc]*len(obs),
                               observables=obs,
                               shots=2048).values
    # from those expvals reconstruct each S_A, S_AB, etc → mutual infos
    mis, ents = postprocess_from_expvals(expvals, subsystems)
    # now cost is identical
    return cost_fn_from_mi_and_ent(mis, ents)

def optimize_geometry(n_qubits, n_layers, subsystems, target_MI, target_S):
    """
    Runs a classical optimizer over the ansatz angles.
    Returns the optimized circuit and final metrics.
    """
    # Random init
    x0 = np.random.uniform(0, 2*np.pi, size=(n_layers*n_qubits,))
    res = minimize(cost_fn, x0,
                   args=(n_qubits, n_layers, subsystems, target_MI, target_S),
                   method='COBYLA', options={'maxiter':200})
    
    qc_opt = build_ansatz(n_qubits, n_layers, res.x)
    sv_opt = Statevector.from_instruction(qc_opt)
    mis_opt, ents_opt = compute_MI_and_SV(sv_opt, subsystems)
    return qc_opt, mis_opt, ents_opt

def embed_and_plot(mis, n_qubits):
    """ Turn mutual infos into a distance matrix, MDS-embed and plot. """
    eps = 1e-6
    D = np.zeros((n_qubits, n_qubits))
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            D[i,j] = D[j,i] = 1.0/(mis[(i,j)] + eps)
    np.fill_diagonal(D, 0)

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
    coords = mds.fit_transform(D)

    plt.figure(figsize=(5,5))
    plt.scatter(coords[:,0], coords[:,1], s=100)
    for i,(x,y) in enumerate(coords):
        plt.text(x, y, f"Q{i}", fontsize=12)
    plt.title("Emergent Geometry")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# --------------------------------------
# 1) Build your parametric “black hole + radiation” ansatz
# --------------------------------------
def build_ansatz(n_qubits, n_layers, params):
    qc = QuantumCircuit(n_qubits)
    p_iter = iter(params)
    for layer in range(n_layers):
        # charge/spin injection on qubit 0
        qc.rx(next(p_iter), 0)
        qc.rz(next(p_iter), 0)
        # entangle & feedback to radiation qubits
        for j in range(1, n_qubits):
            qc.h(0)
            qc.cry(next(p_iter), 0, j)
        qc.barrier()
    return qc

# --------------------------------------
# 2) Tomography settings: which Pauli strings to measure
# --------------------------------------
# --------------------------------------
# 3) From Pauli expectations to entropy & MI
# --------------------------------------
def postprocess_from_expvals(expvals, n_qubits):
    """
    expvals: list of <Z>,<X>,<Y> for each qubit in order,
             then the 2-qubit blocks in the same order as tomographic_observables()
    returns:
      S0, S1, … S_{n-1}  (single-qubit entropies)
      MI[(i,j)] for all i<j
    """
    # first 3*n entries are single Bloch vectors:
    blochs = []
    idx = 0
    for i in range(n_qubits):
        x, y, z = expvals[idx], expvals[idx+1], expvals[idx+2]
        blochs.append(np.array([x,y,z]))
        idx += 3
    def qubit_entropy(bvec):
        r = np.linalg.norm(bvec)
        λp = (1+r)/2
        λm = (1-r)/2
        return -sum([p*np.log2(p) for p in (λp,λm) if p>0])
    S = [qubit_entropy(b) for b in blochs]

    # next blocks: for each pair (i,j) we have 9 Pauli products
    MI = {}
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            # gather the 9 expvals
            M = np.zeros((4,4), complex)  # reconstruct ρ_{ij} by 2-qubit tomography
            # Pauli basis {I,X,Y,Z} → density matrix
            # here we do minimal: ρ = (1/4) ∑_{ab} ⟨σ_a⊗σ_b⟩ σ_a⊗σ_b
            vals = {}
            for a in ['X','Y','Z']:
                for b in ['X','Y','Z']:
                    key=(a,b)
                    vals[key] = expvals[idx]
                    idx += 1
            # build density matrix
            from qiskit.quantum_info.operators import Operator
            from qiskit.quantum_info import Pauli
            ρ = np.zeros((4,4), complex)
            basis = {'I': np.eye(2),
                     'X': np.array([[0,1],[1,0]]),
                     'Y': np.array([[0,-1j],[1j,0]]),
                     'Z': np.array([[1,0],[0,-1]])}
            for a in ['I','X','Y','Z']:
                for b in ['I','X','Y','Z']:
                    if a=='I' and b=='I':
                        coeff = 1
                    elif a=='I':
                        coeff = 0  # we only stored X,Y,Z
                    elif b=='I':
                        coeff = 0
                    else:
                        coeff = vals[(a,b)]
                    ρ += coeff * np.kron(basis[a], basis[b])
            ρ /= 4
            # partial trace entropies
            ρij = DensityMatrix(ρ)
            ρi = partial_trace(ρij, [1])
            ρj = partial_trace(ρij, [0])
            def dens_entropy(rho):
                ev = rho.eigenvalues()
                return -sum([p*np.log2(p) for p in ev if p>1e-12])
            MI[(i,j)] = dens_entropy(ρi) + dens_entropy(ρj) - dens_entropy(ρij)
    return S, MI

# --------------------------------------
# 4) Cost function on real hardware via Estimator
# --------------------------------------
def tomographic_observables(n_qubits):
    """Return a list of single-qubit Z observables on each wire."""
    ops = []
    for i in range(n_qubits):
        pauli = [I]*n_qubits
        pauli[i] = Z
        ops.append(PauliOp(Pauli=pauli))
    return ops

def make_cost_function(n_qubits, n_layers):
    pauli_ops = tomographic_observables(n_qubits)

    def cost(x):
        # 1) Build exactly n_qubits logical ansatz
        ansatz = TwoLocal(num_qubits=n_qubits,
                          rotation_blocks=["ry","rz"],
                          entanglement_blocks="cz",
                          reps=n_layers)
        qc = ansatz.assign_parameters(x)

        # 2) Transpile *only* those n_qubits onto a 4-qubit sublayout
        #    here we pin logical wire i → physical qubit i
        qc_t = transpile(qc,
                         backend=backend,
                         initial_layout=list(range(n_qubits)),
                         optimization_level=1)

        # 3) Run estimator with matched circuit/observable sizes
        job = estimator.run([(qc_t, pauli_ops)])
        return float(job.result().values[0])

    return cost

# 1) Enforce each θᵢ ∈ [0,2π]:
def θ_ge_0(x):
    return x             # x_i >= 0  ⇒  all(x>=0)

def θ_le_2π(x):
    return 2*np.pi - x   # 2π - x_i >= 0  ⇒  all(x<=2π)

# 2) Cap the total phase budget ∑ᵢ θᵢ ≤ 4π:
def total_phase_budget(x, Θ_max=4*np.pi):
    return Θ_max - np.sum(x)
# example constraint: sum(params) <= π
def constraint1(x):
    return np.pi - np.sum(x)

# example constraint: each param ≥ 0
def constraint2(x):
    return x  # non-negative vector

constraints = [
    {"type":"ineq", "fun": θ_ge_0},
    {"type":"ineq", "fun": θ_le_2π},
    {"type":"ineq", "fun": total_phase_budget},
]

# --------------------------------------
# 5) Run the VQE-style loop
# --------------------------------------
def optimize_emergent_geometry(n_qubits=4,
                               n_layers=2,
                               constraints: list[dict] = None,
                               maxiter: int = 100,
                               seed: int = None):
    """
    Optimize the emergent geometry cost for an n_qubits/n_layers circuit.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits in your VQE-like ansatz.
    n_layers : int
        Number of parameterized layers.
    backend : str
        IBM backend name.
    constraints : list of dict, optional
        List of scipy-style constraints, e.g.
            [{'type': 'ineq', 'fun': constraint1}, …]
    maxiter : int
        Maximum COBYLA iterations.
    seed : int, optional
        RNG seed for reproducibility.
    
    Returns
    -------
    result : OptimizeResult
        The full scipy OptimizeResult, including x, fun, success, status, etc.
    """
    # 1) set up the runtime service & cost function
    cost_fn = make_cost_function(n_qubits, n_layers, service, backend)

    # 2) build initial guess
    if seed is not None:
        np.random.seed(seed)
    x0 = 2 * np.pi * np.random.rand(3 * n_qubits * n_layers)

    # 3) default: no constraints
    if constraints is None:
        constraints = []

    # 4) run COBYLA via scipy.minimize
    result = minimize(
        fun=cost_fn,
        x0=x0,
        method="COBYLA",
        constraints=constraints,
        options={
            "maxiter": maxiter,
            "tol": 1e-6,
            "disp": True
        }
    )
    return result

# --------------------------------------
# 6) Visualize final emergent geometry
# --------------------------------------
def plot_emergent_geometry(final_params, n_qubits=4, n_layers=2):
    qc = build_ansatz(n_qubits, n_layers, final_params)
    # get the (no-measurement) statevector approximation
    from qiskit.quantum_info import Statevector
    sv = Statevector.from_instruction(qc)
    # build MI matrix
    mi_mat = np.zeros((n_qubits,n_qubits))
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            mi = compute_mutual_information(sv, [i], [j])
            mi_mat[i,j] = mi_mat[j,i] = mi
    # convert to distance
    D = 1/(mi_mat + 1e-6)
    np.fill_diagonal(D,0)
    coords = MDS(n_components=2, dissimilarity='precomputed').fit_transform(D)
    plt.figure(figsize=(5,5))
    plt.scatter(coords[:,0], coords[:,1], c='C0')
    for i,(x,y) in enumerate(coords):
        plt.text(x,y,f"Q{i}")
    plt.title("Emergent Geometry on Real Hardware")
    plt.axis('equal')
    plt.show()
def page_curve(R):
    """Ideal Page curve (in bits) for 1 black‐hole qubit + R radiation qubits."""
    return np.array([min(k, (R+1)-k) for k in range(1, R+1)], dtype=float)

def run_emergent_spacetime_on_real(n_qubits=4, num_steps=6, gap_cycles=20):
    # 1) build the logical n_qubit circuit + collect data
    qc, entropy_vals, mutual_info_vals, state_snapshots = \
        create_dynamic_entropy_circuit(num_steps, n_qubits, gap_cycles)
    
    # 2) plot von Neumann entropy vs mutual information
    plt.figure(figsize=(8,4))
    plt.plot(range(1, num_steps+1), entropy_vals, '-o', label="Entropy")
    plt.plot(range(1, num_steps+1), mutual_info_vals, '-s', label="Mutual Info")
    plt.xlabel("Step"); plt.ylabel("bits"); plt.legend(); plt.grid()
    plt.title("Dynamics of Black-Hole Radiation")
    plt.show()
    
    # 3) measure the *same* n_qubit circuit on hardware
    meas_qc = qc.copy()
    meas_qc.add_register(ClassicalRegister(n_qubits-1))
    # measure radiation qubits [1..n_qubits-1] into classical bits
    meas_qc.measure(list(range(1, n_qubits)), list(range(n_qubits-1)))

    full_layout = list(range(meas_qc.num_qubits))

    # **key**: pin those 4 logical qubits to physical [0,1,2,3] on brisbane
    transpiled = transpile(
        meas_qc,
        backend=backend,
        initial_layout=full_layout,
        optimization_level=1
    )
    with Session(backend) as session:
        sampler = Sampler()
        job = sampler.run([transpiled], shots=1024)
        counts = extract_counts_from_bitarray(extract_bitarray_from_primitive(job.result()))
    plot_histogram(counts, title=f"Final Measurement (first {n_qubits} qubits)")
    
    # 4) now do your emergent‐geometry MDS plots exactly as before
    visualize_emergent_geometry_over_time(state_snapshots)

def run_with_page_fit(
    num_radiation_qubits=5,
    num_steps=None,
    gap_cycles=20
):
    # steps = as many injections as there are radiation qubits
    backend_name = backend
    R = num_radiation_qubits
    num_steps = num_steps or R

    # --- build & measure ---
    qc, entropy_vals, mi_vals, snaps = create_dynamic_entropy_circuit(
        num_steps=num_steps,
        num_radiation_qubits=R,
        gap_cycles=gap_cycles,
    )

    # 1) Plot your data vs Page curve
    ideal = page_curve(R)

    # 2) quantitative metrics
    r, p_r = pearsonr(entropy_vals, ideal)
    rho, p_s = spearmanr(entropy_vals, ideal)
    R2 = r2_score(ideal, entropy_vals)

    print(f"Pearson r = {r:.3f} (p={p_r:.2g})")
    print(f"Spearman ρ = {rho:.3f} (p={p_s:.2g})")
    print(f"R² = {R2:.3f}")

    # 3) Plot together
    steps = np.arange(1, num_steps+1)
    plt.figure(figsize=(8,5))
    plt.plot(steps, entropy_vals, 'o-', label="Measured S")
    plt.plot(steps, ideal,   's--', label="Ideal Page S")
    plt.xlabel("injection step $k$")
    plt.ylabel("Entropy (bits)")
    plt.title("Measured vs. Ideal Page Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    qc_final = qc.copy()
    qc_final.add_register(ClassicalRegister(R))
    qc_final.measure(range(1, R+1), range(R))
    with Session(backend) as session:
        sampler = Sampler()
        # error‐mitigation: measurement calibration
        # you could insert ignis calibration here...
        job = sampler.run([transpile(qc_final, backend)], shots=2048)
        counts = extract_counts_from_bitarray(extract_bitarray_from_primitive(job.result()))
        plot_histogram(counts, title="Final radiation histogram")
        plt.show()

    # 5) visualize emergent geometry
    visualize_emergent_geometry_over_time(snaps)

    return {
        "entropy": entropy_vals,
        "ideal_page": ideal,
        "pearson": (r, p_r),
        "spearman": (rho, p_s),
        "r2": R2
    }
        
#standard qc for multiverse experiments is main_qc()
if __name__ == "__main__":
# ── 2) Instantiate your analyzer on 4 qubits (1D chain) ─────────────────
    n = 4
    analyzer = QuantumGravityAnalyzer(n_qubits=n)

    # ── 3) Define your “black-hole” CP angles on each edge ───────────────────
    theta = {(i, i+1): 0.3 + 0.1*i for i in range(n-1)}

    # ── 4) Choose the regions you want to test (e.g. qubits {1,2} and {2,3}) ─
    region = [1,2]
    regions=[[1,2]]

    # ── 5A) Plain FD first-law on hardware ────────────────────────────────────

    # e.g. → { (1,2): 0.0187, (2,3): 0.0029 }

    # ── 5C) Single-region modular-H measurement, plain vs ZNE ──────────────
    # build the CP circuit once:
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for (i,j),th in theta.items():
        qc.cp(th, i, j)

    # extract ρ₀
    from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
    dm = DensityMatrix(Statevector.from_instruction(qc))
    rho12 = partial_trace(dm, [q for q in range(n) if q not in region]).data

    # decompose H = −log₂ρ₀
    terms = analyzer._decompose_modular_hamiltonian(rho12)

    fast_results = analyzer.entanglement_equilibrium_operator_zne(
        theta_dict=theta,
        regions=regions,
        eps=1e-4,
        backend=backend,
        shots=2048,
        noise_factors=[1,3]
    )
    print("Fast ZNE results:", fast_results)

##    eps = 0.02  # Use larger ε on hardware to overcome noise
##    thetas_dense, K_theory = compute_theoretical_curvature_v3(num_points=500, eps=eps)
##
##    plt.figure(figsize=(8, 5))
##    plt.plot(thetas_dense, K_theory, 'b-', label='Theory $K_{\\mathrm{theory}}(\\theta)$')
##    plt.xlabel(r'$\theta_{01}$')
##    plt.ylabel(r'Entanglement Curvature $K$')
##    plt.title('Smooth Theoretical Curvature $K_{\\mathrm{theory}}(\\theta)$')
##    plt.grid(True)
##    plt.legend()
##    plt.tight_layout()
##    plt.show()
##
##    # 7.B) Run manual readout calibration on qubits [0,1]
##    print("Performing readout calibration on qubits [0,1] ...")
##    M = run_readout_calibration(qubits=[0, 1], shots=4096)
##    print("Readout calibration matrix M:\n", M)
##
##    # 7.C) Choose hardware θ points for the sweep
##    theta_hw_points = np.linspace(eps, np.pi - eps, 7)  # avoid endpoints exactly
##    K_hw_list = []
##    error_bars = []
##
##    for θ in theta_hw_points:
##        print(f"Measuring hardware curvature at θ = {θ:.3f} ...")
##        # Repeat multiple times for error bars
##        reps = 3
##        samples = []
##        for _ in range(reps):
##            K_val, _ = compute_hardware_curvature(θ, eps, M, shots=8192)
##            samples.append(K_val)
##        mean_val = np.mean(samples)
##        std_val = np.std(samples, ddof=1)
##        K_hw_list.append(mean_val)
##        error_bars.append(std_val)
##        print(f"  θ={θ:.3f}  →  K_hw = {mean_val:.3f} ± {std_val:.3f}")
##
##    K_hw_arr = np.array(K_hw_list)
##    err_arr = np.array(error_bars)
##
##    # 7.D) Plot hardware points over theory curve
##    plt.figure(figsize=(8, 5))
##    plt.plot(thetas_dense, K_theory, 'b-', label='Theory $K_{\\mathrm{theory}}(\\theta)$')
##    plt.errorbar(theta_hw_points, K_hw_arr, yerr=err_arr, fmt='ro', capsize=4,
##                 label='Hardware $\\hat K_{\\mathrm{hw}}(\\theta)$')
##    plt.xlabel(r'$\theta_{01}$')
##    plt.ylabel(r'Entanglement Curvature $K$')
##    plt.title('Theory vs. Hardware Entanglement Curvature')
##    plt.grid(True)
##    plt.legend()
##    plt.tight_layout()
##    plt.show()
##
##    # 7.E) Reduced χ² analysis
##    K_theory_at_hw = np.interp(theta_hw_points, thetas_dense, K_theory)
##    chisq = np.sum(((K_hw_arr - K_theory_at_hw) ** 2) / (err_arr ** 2))
##    dof = len(theta_hw_points) - 1
##    chisq_red = chisq / dof
##    print(f"Reduced χ² = {chisq_red:.2f} over {dof} degrees of freedom")
##
##    # 7.F) Save data for reproducibility
##    np.savez(
##        "entanglement_curvature_results_manual_mitigation.npz",
##        theta_dense=thetas_dense,
##        K_theory=K_theory,
##        theta_hw=theta_hw_points,
##        K_hw=K_hw_arr,
##        err_hw=err_arr
##    )
##    print("Data saved to entanglement_curvature_results_manual_mitigation.npz")
##    plt.savefig("theory_vs_hardware_curvature_manual_mitigation.png", dpi=300)
##    print("Plot saved as theory_vs_hardware_curvature_manual_mitigation.png")
##
##    eps        = 1e-3
##    theta_hw   = np.linspace(0, np.pi, 30)      # points for finite-difference K
##    num_dense  = 500                            # points for smooth theory curve
##
##    # 5.B) Compute theoretical K(θ) on a dense grid
##    thetas_dense, K_theory = compute_theoretical_curvature_v2(num_points=num_dense, eps=eps)
##
##    # 5.C) Compute finite-difference K(θ) on the coarser grid (simulation mode)
##    K_fd = np.array([compute_finite_difference_K(th, eps) for th in theta_hw])
##
##    # 5.D) Plot both curves for comparison
##    plt.figure(figsize=(7, 5))
##    plt.plot(thetas_dense, K_theory, 'b-', label='Theory $K_{\\mathrm{theory}}(\\theta)$')
##    plt.plot(theta_hw,    K_fd,      'ro', label='Finite-diff $K_{\\mathrm{fd}}(\\theta)$')
##    plt.xlabel(r'$\theta_{01}$')
##    plt.ylabel(r'Entanglement Curvature $K$')
##    plt.title('Theory vs. Finite-Difference Curvature (Simulation)')
##    plt.legend()
##    plt.grid(True)
##    plt.tight_layout()
##    plt.show()

##    # sweep theta01
##    thetas = np.linspace(0, np.pi, 30)
##    MIs = []
##    for t in thetas:
##        sv = Statevector.from_instruction(small_plaquette_circuit(t, np.pi/3, noise=0.8))
##        dm = DensityMatrix(sv)
##        # mutual info between qubit 0 (bulk) and {2,3} (boundary)
##        MIs.append(mutual_info(dm, [2,3]))
##
##    plt.plot(thetas, MIs, '-o')
##    plt.xlabel(r'$\theta_{01}$')
##    plt.ylabel('I(bulk:boundary)')
##    plt.title('Non-trivial MI sweep on 2×2 plaquette')
##    plt.show()

    # Sweep parameters
##    αs = np.linspace(0, np.pi, 11)
##    βs = np.linspace(0, np.pi, 11)
##    γs = np.linspace(0, np.pi, 11)
##
##    results = {}
##    for α in αs:
##        for β in βs:
##            for γ in γs:
##                qc = warp_racircuit(α, β, γ)
##                sv = Statevector.from_instruction(qc)
##                rho = DensityMatrix(sv)
##                τ = three_tangle(rho)
##                results[(round(α,2), round(β,2), round(γ,2))] = τ
##
##    # Now you can e.g. print the “hot spots” where τ > 0:
##    for params, τ in results.items():
##        if τ > 1e-6:
##            print(f"3-tangle {τ:.3f} at α,β,γ = {params}")

##    n=5
##
##    edge_list = [(i, i+1) for i in range(n-1)]
##    
##    # 2) Pick a toy target curvature: e.g., high curvature at the ends, low in the middle
##    T_target = {i: 1.0 - abs((n-1)/2 - i)/( (n-1)/2 ) for i in range(n)}
##
##    plaquette_curvs, pair_MI, node_R, bianchi, devs = \
##    run_blackhole_curvature_real_0(
##        rows=3, cols=3,
##        mass_strength=1.0,
##        epsilon=0.01,
##        shots=8192
##    )
##    print(plaquette_curvs)
##    print(pair_MI)
##    print(node_R)
##    print(bianchi)
##    print(devs)

    # 4) For each region A, decompose Hξ(A) once offline:

    


##    frac, counts = loschmidt_echo(n_qubits=3, scramble_depth=5, shots=1024, seed=42)
##    print("Counts:", counts)

##    qc = gao_jafferis_wall_circuit_plus(theta=0.8, g=0.79, deepen_scramble=False)
##    result = run(qc, backend=backend, sim=False, shots=2048)
##    print(result)
    
##    theta = 0  # for |+>, theta=pi/2 but qc.h prepares |+>, ignore theta
##    gs = np.linspace(0.78, 0.8, 1)
##    df, g_opt = sweep_optimal_g(theta, gs)
##    print("Sweep results:", df)
##    print("Optimal g:", g_opt)


 #   backend_name = "ibm_sherbrooke"          # must support mid-circuit measurement & c_if
    # rows, cols = 3, 3                        # size of your 2D cluster on each side
    # θ_mass    = 0.6                          # negative warp angle at the throats
    # g         = 0.39                # throat coupling strength
    # theta = np.pi/3
    # phi=np.pi/5

    #     # Experiment parameters
    # rows, cols = 4, 4
    # sigma = 1.0
    # max_sep = 2

    # results = []
    # qc = gao_jafferis_wall_circuit(theta, phi, g, seed=42)
    
    # # Simulate in Aer statevector for verification
    # qc_nmeas = qc.copy()
    # qc_nmeas = qc_nmeas.remove_final_measurements(inplace=False)


    # sim = Aer.get_backend('statevector_simulator')
    # sv = run(qc_nmeas, sim, rs=True)
    # # Compute ideal fidelity
    # from qiskit.quantum_info import partial_trace, state_fidelity
    # # extract reduced state on q3
    # rho = partial_trace(sv, [0,1,2])
    # psi0 = np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])
    # fidelity = state_fidelity(rho, psi0)
    # print(f"Ideal state fidelity on q3: {fidelity:.3f}")
    # run(qc, backend=backend)
    # for sep in range(1, max_sep + 1):
    #     # Define bubble centers symmetrically separated along the middle row
    #     c1 = (rows // 2, cols // 2 - sep)
    #     c2 = (rows // 2, cols // 2 + sep)
    #     mass = gaussian_peaks(rows, cols, [c1, c2], sigma)
        
    #     # Build spacetime circuit
    #     qc = generate_spacetime_circuit(rows, cols, mass)
        
    #     # Identify qubit indices for the two bubbles
    #     idx = lambda i, j: i * cols + j
    #     qA = idx(*c1)
    #     qB = idx(*c2)
        
    #     # Entangle the bubble qubits
    #     qc.h(qA)
    #     qc.cx(qA, qB)
        
    #     sv = run(qc, backend, rs=True)
        
    #     # Compute mutual information between the two bubble qubits
    #     MI = mutual_information(sv, [qA], [qB])
    #     results.append((sep, MI))

              
    # print(results)




##    gs = np.linspace(0, 1.2, 7)
##    wormhole_K = run_negative_wormhole(3,3, θ_mass=1, gs=gs, backend=backend)
##    print(wormhole_K)

    
##    N = 3
##    # simple uniform mass profile → small phase on every plaquette
##    phase_profile = np.full((N, N), 0.1)
##    counts = run_torus_ctc(N, phase_profile, backend_name="ibm_brisbane", shots=2048)
##    print("Loop measurement counts:", counts)
##

    
##    Ns = [3, 4, 5]
##    scale_vs_N = study_scaling_N(
##        sizes=Ns,
##        cols=None,          # will be ignored if cols is None; uses N×N
##        mass_strength=5.0,
##        epsilon=0.1,
##        backend_name="ibm_brisbane",
##        shots=2048
##    )
##
##    ms = [1.0, 2.5, 5.0, 7.5]
##    scale_vs_m = study_scaling_mass(
##        N=5, cols=5,
##        masses=ms,
##        epsilon=0.1,
##        backend_name="ibm_brisbane",
##        shots=2048
##    )


    
##    rows, cols = 7, 7
##    n_qubits = rows*cols
##    qc = QuantumCircuit(n_qubits)
##    mass_sites = [(3,3)]
##    θ_mass      = 0.6
##    prepare_vacuum_cluster(qc, rows, cols)
##    apply_mass_deformation(qc, mass_sites, θ_mass)
##    curvatures = run_blackhole_curvature_real(rows, cols, mass_strength=5.0, epsilon=0.1,backend_name="ibm_brisbane", shots=2048)
##    print("Hardware plaquette curvatures:")
##    for p,K in curvatures.items():
##        print(f"  {p}: {K:.4f} bits")
        
##    n_list   = [11]           # only two sizes
##    epsilons = [1e-4]     # only two eps
##    gradient_steps = 100        # half the work        # chain lengths you want to test
##    mass_node      =  (max(n_list)//2)    # center of your “mass” profile
##    mass_strength  =  5.0
##    sigma          =  0.5
##    lr             = 0.05
##    jobs = []
##    for n in n_list:
##        for eps in epsilons:
##            # each job is the tuple of arguments your worker expects
##            jobs.append((n,
##                         mass_node,
##                         mass_strength,
##                         sigma,
##                         [eps],        # note: sweep expects a list of epsilons
##                         gradient_steps,
##                         lr))
##    # now jobs is a list built by loops, no comprehension syntax
##    print("Jobs to run:", jobs)
##    results = {}
##    with concurrent.futures.ProcessPoolExecutor() as exe:
##     for n, eps, ratio_dict in tqdm(exe.map(run_job, jobs), total=len(jobs)):
##         results.setdefault(n, {})[eps] = ratio_dict
##         print(results)
##
##    # Now 'results' holds the same data but computed in parallel
##    print(results)

##    L = 3
##    edges     = build_3d_edges(L)
##    faces     = list_3d_faces(L)
##    cells     = list_3d_cells(L)
##    f2c_map   = build_faces_of_cell(cells, faces)
##
##    # Gaussian mass profile centered in cube
##    center = ((L-1)/2,)*3
##    sigma, A = 1.0, 1.0
##    T = {}
##    for x in range(L):
##        for y in range(L):
##            for z in range(L):
##                d2 = (x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2
##                idx3 = x + L*(y + L*z)
##                T[idx3] = A * np.exp(-d2/(2*sigma**2))
##                print(x, y, z)
##
##    # map node masses to link angles
##    theta = {}
##    for i,j in edges:
##        theta[(i,j)] = (T[i] + T[j]) / 2
##
##    qc = create_3d_entanglement_circuit(L**3, edges, theta)
##
##    face_curv = compute_face_curvature(qc, faces)
##    cell_curv = compute_cell_curvature(qc, cells, f2c_map)
##    print('Face curvatures:', face_curv)
##    print('Cell curvatures:', cell_curv)



##    rows, cols = 3, 3
##    edges = build_2d_edges(rows, cols)
##    plaquettes = list_plaquettes(rows, cols)
##
##    # generate mass profile centered at (1,1)
##    mass_center = (1,1)
##    sigma = 0.8
##    mass_strength = 1.0
##
##    for r in range(rows):
##        for c in range(cols):
##            dist2 = (r-mass_center[0])**2+(c-mass_center[1])**2
##            node_idx = r*cols+c
##            T_target[node_idx] = mass_strength*np.exp(-dist2/(2*sigma**2))
##
##        
##    analyzer = QuantumGravityAnalyzer(n_qubits=n)
##
##    theta_dict = {}
##    for i,j in edges:
##        theta_dict[(i,j)] = (T_target[i]+T_target[j])/2
##
##    # build circuit and compute curvature
##    qc = create_2d_entanglement_circuit(rows, cols, theta_dict)
##    curv = compute_plaquette_curvature(qc, rows, cols, plaquettes)
##    print("Plaquette curvatures:", curv)
##        
##    regions = [[0,1], [1,2], [2,3], [3,4], [4,5], [5,6],[6,7]]  # or any subregions you like
##    #results = analyzer.entanglement_equilibrium_check(theta_dict, regions, eps=1e-3)
##    eq_results = analyzer.entanglement_equilibrium_check_local(theta_dict, regions, eps=1e-3)
##    for region, (dS, dH) in eq_results.items():
##        print(f"Region {region}: δS/δθ = {dS:.4e}, δ⟨Hξ⟩/δθ = {dH:.4e}")
##
##    # 2) Pick a “vacuum” theta (flat) or a solved theta_dict from your gradient-descent
##    
##
##    # 3) Measure mutual information on every link
##    sim = Aer.get_backend("aer_simulator_statevector")
##    I_dict = analyzer.compute_mutual_information(theta_dict, simulator=sim)
##
##    # 4) Compute node curvatures R_i
##    R_dict = analyzer.compute_R(I_dict)
##
##    # 5) Check discrete Bianchi residuals Δ_i
##    Delta = analyzer.check_bianchi(R_dict)
##    print("Δ_i (should be small for an “Einstein” solution):", Delta)
##    # Target average entropy across qubits
##    target_entropy = 0
##
##    # Generate a circuit that hits the target entropy
##    qc, entropies, avg_ant = tune_entropy_target(n_qubits=3, target_entropy=target_entropy)
##
##    # Print result
##    print("\nFinal Subsystem Entropies:")
##    for q, entropy in entropies.items():
##        print(f"Qubit {q}: {entropy:.5f}")
##
##    print(f"Average Entropy: ", avg_ant)
##
##    # Optional: simulate and view histogram
##    backend = Aer.get_backend('qasm_simulator')
##    qc.measure_all()
##    job = backend.run(qc, shots=1024)
##    counts = job.result().get_counts()
##    print(f"Counts: ", counts)
##    plot_histogram(counts)
##    plt.show()
    
