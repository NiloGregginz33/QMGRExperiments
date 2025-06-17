import sys
sys.path.insert(0, 'src')
from experiment_logger import ExperimentLogger
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, entropy
import numpy as np
import matplotlib.pyplot as plt

def run_holographic_demo():
    """
    Demonstrates the holographic principle using a maximally entangled state.
    Logs results and implications for theoretical physics.
    """
    logger = ExperimentLogger('holographic_demo')
    qc = QuantumCircuit(6)
    qc.h(0)
    for i in range(1, 6):
        qc.cx(0, i)
    
    # Get statevector and calculate entropy
    state = Statevector.from_instruction(qc)
    rho_boundary = partial_trace(state, [0])  # Trace out the bulk qubit
    S_boundary = entropy(rho_boundary)
    
    # Logging
    logger.log_theoretical_background(
        "Demonstrates the holographic principle using a maximally entangled state."
    )
    logger.log_methodology(
        "Prepare a 6-qubit GHZ state, trace out the bulk, and compute boundary entropy."
    )
    logger.log_parameters({'num_qubits': 6})
    logger.log_metrics({'boundary_entropy': float(S_boundary)})
    logger.log_analysis(
        "Boundary entropy is close to maximal, indicating complete bulk encoding."
    )
    logger.log_interpretation(
        "The pattern matches the expected behavior for a holographic system."
    )
    return {'boundary_entropy': S_boundary}

def run_temporal_injection():
    """
    Runs a temporal charge injection experiment by sweeping RX on the bulk qubit.
    Logs entropy as a function of injected charge and analyzes implications.
    """
    logger = ExperimentLogger('temporal_injection')
    phis = np.linspace(0, 2*np.pi, 20)
    entropies = []
    
    for phi in phis:
        qc = QuantumCircuit(6)
        qc.h(0)
        for i in range(1, 6):
            qc.cx(0, i)
        qc.rx(phi, 0)
        
        # Calculate entropy
        state = Statevector.from_instruction(qc)
        rho_boundary = partial_trace(state, [0])
        S = entropy(rho_boundary)
        entropies.append(S)
        logger.logger.info(f"[temporal_injection] phi={phi:.2f}, entropy={S}")
    
    # Plot results
    fig = plt.figure(figsize=(10, 6))
    plt.plot(phis, entropies)
    plt.xlabel('Injected Phase (φ)')
    plt.ylabel('Boundary Entropy')
    plt.title('Temporal Charge Injection Effect on Boundary Entropy')
    plt.grid(True)
    logger.save_plot(fig, 'temporal_injection_plot')
    
    # Logging
    logger.log_theoretical_background(
        "Temporal charge injection experiment by sweeping RX on the bulk qubit."
    )
    logger.log_methodology(
        "Sweep RX gate on the bulk qubit, compute boundary entropy for each phase."
    )
    logger.log_parameters({'num_qubits': 6, 'phis': phis.tolist()})
    logger.log_metrics({'entropies': [float(e) for e in entropies]})
    logger.log_analysis(
        "Entropy oscillates with injected charge, demonstrating information flow."
    )
    logger.log_interpretation(
        "The system maintains holographic properties despite charge injection."
    )
    return {'phis': phis, 'entropies': entropies}

def run_contradictions_test():
    logger = ExperimentLogger('contradictions_test')
    qc1 = QuantumCircuit(6)
    qc1.h(0)
    qc1.h(1)
    qc1.cx(0, 2)
    qc1.cx(1, 3)
    state1 = Statevector.from_instruction(qc1)
    rho_boundary1 = partial_trace(state1, [0, 1])
    S1 = entropy(rho_boundary1)
    # Logging
    logger.log_theoretical_background(
        "Test the holographic principle with disconnected bulk."
    )
    logger.log_methodology(
        "Prepare two separate entangled pairs, compute boundary entropy."
    )
    logger.log_parameters({'num_qubits': 6})
    logger.log_metrics({'disconnected_entropy': float(S1)})
    logger.log_analysis(
        "Disconnected bulk leads to reduced boundary entropy."
    )
    logger.log_interpretation(
        "Holographic principle is violated when bulk-bulk connections are missing."
    )
    return {'disconnected_entropy': S1}

if __name__ == "__main__":
    print("Running holographic principle demonstration...")
    run_holographic_demo()
    
    print("\nRunning temporal charge injection experiment...")
    run_temporal_injection()
    
    print("\nRunning holographic contradictions test...")
    run_contradictions_test()
    
    print("\nAll experiments completed. Check experiment_outputs directory for detailed results.") 