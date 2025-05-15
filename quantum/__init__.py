"""
Quantum computing package for advanced quantum simulations and experiments.
This package provides tools for quantum circuit manipulation, black hole simulations,
quantum communication, and error correction.
"""

from .circuits.basic_circuits import (
    create_base_circuit,
    create_entangled_system,
    create_teleportation_circuit,
    create_holographic_circuit,
    create_entanglement_wedge_circuit
)

from .analysis.entropy import (
    qiskit_entropy,
    shannon_entropy,
    measure_subsystem_entropies,
    calculate_von_neumann_entropy,
    analyze_subsystem_qiskit_entropy,
    compute_mutual_information
)

from .gravity.black_hole import (
    black_hole_simulation,
    information_paradox_test,
    hawking_radiation_recovery,
    create_black_hole_curvature
)

from .communication.quantum_telephone import (
    multiversal_telephone,
    send_quantum_message_real,
    dual_channel_communication,
    amplify_target_state
)

from .error_correction.qec import (
    charge_preserving_qec,
    shor_qec_noisy,
    apply_charge_preserving_qec,
    detect_and_correct_errors
)

from .utils.quantum_utils import (
    get_best_backend,
    is_simulator,
    select_backend,
    initialize_qubits,
    apply_entanglement,
    measure_and_reset,
    run_circuit_statevector,
    process_sampler_result
)

__version__ = '0.1.0'
__author__ = 'Quantum Research Team' 