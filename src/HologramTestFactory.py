from qiskit import QuantumCircuit, QuantumRegister, pulse, transpile, ClassicalRegister, assemble
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix, state_fidelity, random_unitary, Operator, mutual_information
from qiskit_aer import Aer, AerSimulator
from qiskit.quantum_info import entropy as qiskit_entropy
from qiskit.pulse import Play, DriveChannel, Gaussian, Schedule
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, amplitude_damping_error
from collections import Counter
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session, Estimator
from qiskit.quantum_info import SparsePauliOp
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from scipy.spatial.distance import jensenshannon
from qiskit.result import Result, Counts  # Import for local simulation results
from qiskit.circuit.library import QFT, RZGate, MCXGate, ZGate, XGate, HGate
from qiskit.qasm3 import dump
from qiskit.primitives.containers import BitArray
import random
from scipy.linalg import expm
import hashlib
from qiskit.circuit import Instruction
import time
from datetime import datetime
from scipy.stats import binom_test
import requests
import sys
import psutil
import itertools
import csv
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import inspect
from collections import Counter, defaultdict
from typing import List, Dict
from scipy.optimize import minimize
import os

#Test functions until dotted line beginning with exN
############################################################################################################################################################################################
#EX1
def ex1(qc, backend=None, shots=8192):
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

##Output to main_retrocausal_experiment

##Baseline Counts (Full Measurement): {'000 000': 4037, '111 111': 4155}
##Counts After Partial Measurement: {'111': 4075, '000': 4117}
##Counts After Delayed Measurement: {'111': 4022, '000': 4170}

def ex2(qc, backend=None, shots=8192):
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


    use_sampler = backend and not isinstance(backend, AerSimulator)
    if backend is None:
        backend = AerSimulator()

    num_qubits = qc.num_qubits
    if qc.num_clbits < num_qubits:
        qc.add_register(ClassicalRegister(num_qubits - qc.num_clbits))

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

##Output to main_retrocausal_experiment_with_charge

##Best backend chosen: ibm_brisbane
##Data structure:  {'00': 8192}
##Data structure:  {'01': 4171, '00': 4021}
##Data structure:  {'01': 4005, '10': 3940, '11': 161, '00': 86}
##🧪 Baseline Counts (Full Measurement): {'00': 8192}
##🧪 Counts After Partial Measurement: {'01': 4171, '00': 4021}
##⚡ Counts After Charge Injection + Delayed Measurement: {'01': 4005, '10': 3940, '11': 161, '00': 86}

def ex3(backend=None, shots=8192):
    """
    Runs the quantum erasure experiment to distinguish between Many-Worlds and Holography.
    
    Parameters:
        backend (Backend): The backend for execution. If None, uses a simulator.
        shots (int): Number of measurement shots.

    Returns:
        dict: Measurement results comparing information recovery.
    """
    service = QiskitRuntimeService()
    backend = get_best_backend(service)

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
        transpiled_qc = transpile(qc, backend)
        sampler = Sampler(backend)
        job = sampler.run([transpiled_qc], shots=shots)
        result = job.result()
        counts = result.data.meas.get_counts()
    else:
        # Use simulator to analyze full quantum state
        simulator = AerSimulator()
        qc_no_measure = qc.remove_final_measurements(inplace=False)
        state = Statevector.from_instruction(qc_no_measure)
        counts = state.probabilities_dict()

    print("Quantum Experiment Results:", counts)
    return counts

##Output to main_run_quantum_erasure_experiment

##Best backend chosen: ibm_sherbrooke
##Baseline Counts (Full Measurement): {'11': 4129, '00': 4063}
##Counts After Partial Measurement: {'00': 4138, '01': 4054}
##Counts After Delayed Measurement: {'11': 4082, '00': 4110}

def ex4(qc, backend=None, shots=8192, iterations=5):
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

##Output to main_run_mwi_vs_holography_experiment

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

def ex5(qc, backend=Aer.get_backend('aer_simulator'), shots=8192):
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

def ex6():
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

##Output to main_entanglement_wedge_experiment
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


#########################################################################################################################################################

def run_circuit(qc, shots=32768):
    """Execute a circuit on a simulator and return measurement results."""
    backend = Aer.get_backend('aer_simulator')
    transpiled_qc = transpile(qc, backend)
    result = backend.run(transpiled_qc, shots=shots).result()
    return result.get_counts()

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

def run_circuit(qc_modified):
    """Executes circuit using either Sampler or standard backend.run()."""
    service = QiskitRuntimeService()
    backend = get_best_backend(service)
    use_sampler = backend and not isinstance(backend, AerSimulator)
    transpiled_qc = transpile(qc_modified, backend)
    if use_sampler:
        sampler = Sampler(backend)
        job = sampler.run([transpiled_qc], shots=1024)
        result = job.result()
        print("Data structure: ", result[0].data.c0.get_counts())
        dont_sampler = False
        counts = result[0].data.c0.get_counts()
    else:
        job = backend.run(transpiled_qc, shots=shots)
        counts = job.result().get_counts()
    return counts

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

def create_two_qubit_circuit():
    """
    Creates a two-qubit circuit for subsystem entropy testing.
    """
    qc = QuantumCircuit(2)
    qc.h(0)  # Apply Hadamard to the first qubit
    qc.cx(0, 1)  # Create entanglement between qubits
    return qc

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

################################################################################################################################################################

service = QiskitRuntimeService()

def get_best_backend(service, min_qubits=3, max_queue=10):
    backends = service.backends()
    suitable_backends = [
        b for b in backends if b.configuration().n_qubits >= min_qubits and b.status().pending_jobs <= max_queue
    ]
    if not suitable_backends:
        print("No suitable backends found. Using default: ibm_kyiv")
        return service.backend("ibm_kyiv")
    
    best_backend = sorted(suitable_backends, key=lambda b: b.status().pending_jobs)[0]
    print(f"Best backend chosen: {best_backend.name}")
    return best_backend

backend = get_best_backend(service)

################################################################################################################################################################

def run_daily_alignment_cycle():
    now = datetime.datetime.now()
    hour = now.hour
    date_str = now.strftime("%Y-%m-%d")
    log_file = "daily_alignment_log.csv"

    if hour < 14:  # Morning run
        print(f"\n☀️ Morning run detected ({hour}:00) — running entanglement wedge circuit...")

        results = main_entanglement_wedge_experiment()  # Your existing circuit

        # Format log entry
        log_entry = {
            "date": date_str,
            "mode": "morning",
            "entropy_partial": results["partial_entropy"],
            "entropy_charged": results["charged_entropy"],
            "entropy_full": results["full_entropy"],
            "symbol_partial": max(results["partial_counts"], key=results["partial_counts"].get),
            "symbol_charged": max(results["charged_counts"], key=results["charged_counts"].get),
            "symbol_full": max(results["full_counts"], key=results["full_counts"].get),
            "journal": "",
            "score": ""
        }

    else:  # Evening reflection
        print(f"\n🌙 Evening run detected ({hour}:00) — enter journal entry and score.")
        journal = input("📓 How did the day go? (Journal entry): ").strip()
        score = input("🧠 Score your day from 1 to 10: ").strip()

        log_entry = {
            "date": date_str,
            "mode": "evening",
            "entropy_partial": "",
            "entropy_charged": "",
            "entropy_full": "",
            "symbol_partial": "",
            "symbol_charged": "",
            "symbol_full": "",
            "journal": journal,
            "score": score
        }

    # Write to CSV
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=log_entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

    print(f"\n✅ Log entry saved for {date_str} ({log_entry['mode']}).")

if __name__ == "__main__":
    qc = create_two_qubit_circuit()
    ex4(qc)
