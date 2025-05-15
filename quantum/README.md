# Quantum Computing Package

A comprehensive Python package for advanced quantum simulations and experiments, providing tools for quantum circuit manipulation, black hole simulations, quantum communication, and error correction.

## Features

- **Basic Quantum Circuits**: Create and manipulate fundamental quantum circuits
- **Entropy Analysis**: Calculate and analyze various types of quantum entropy
- **Black Hole Simulations**: Simulate quantum black holes and study their properties
- **Quantum Communication**: Implement quantum communication protocols
- **Error Correction**: Apply various quantum error correction techniques
- **Utility Functions**: Common quantum computing operations and backend management

## Installation

```bash
pip install -r requirements.txt
```

## Package Structure

```
quantum/
├── circuits/          # Basic quantum circuit operations
├── analysis/          # Entropy and measurement analysis
├── gravity/           # Black hole simulations
├── communication/     # Quantum communication protocols
├── error_correction/  # Error correction techniques
└── utils/            # Utility functions
```

## Quick Start

### Basic Circuit Creation

```python
from quantum import create_entangled_system, create_teleportation_circuit

# Create an entangled system with 3 qubits
entangled_circuit = create_entangled_system(n_qubits=3)

# Create a quantum teleportation circuit
teleport_circuit = create_teleportation_circuit()
```

### Entropy Analysis

```python
from quantum import calculate_von_neumann_entropy, analyze_subsystem_qiskit_entropy

# Calculate von Neumann entropy
entropy = calculate_von_neumann_entropy(circuit, num_radiation_qubits=2)

# Analyze subsystem entropies
subsystem_entropies = analyze_subsystem_qiskit_entropy(statevector)
```

### Black Hole Simulation

```python
from quantum import black_hole_simulation, information_paradox_test

# Simulate a quantum black hole
bh_circuit = black_hole_simulation(
    num_qubits=17,
    num_charge_cycles=5,
    spin_cycles=3
)

# Test the information paradox
paradox_circuit = information_paradox_test(
    num_qubits=10,
    injection_strength=np.pi/2
)
```

### Quantum Communication

```python
from quantum import multiversal_telephone, send_quantum_message_real

# Send a quantum message
message_circuit = multiversal_telephone("Hello Quantum World!")

# Send a message with controlled entropy
controlled_message = send_quantum_message_real(
    message="Quantum Message",
    entropy_per_char=0.75
)
```

### Error Correction

```python
from quantum import charge_preserving_qec, shor_qec_noisy

# Apply charge-preserving error correction
protected_circuit = charge_preserving_qec(num_logical_qubits=2)

# Implement Shor's error correction code
shor_circuit = shor_qec_noisy(num_logical_qubits=1)
```

## Advanced Usage

### Black Hole Curvature Simulation

```python
from quantum import create_black_hole_curvature

# Create a 5x5 lattice simulating black hole curvature
curvature_circuit = create_black_hole_curvature(
    rows=5,
    cols=5,
    mass_strength=1.0,
    center=(2, 2)
)
```

### Dual-Channel Quantum Communication

```python
from quantum import dual_channel_communication

# Send two messages through quantum channels
dual_circuit = dual_channel_communication(
    message_rad1="First Message",
    message_rad2="Second Message",
    scaling_factor=0.6
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{quantum_package,
  author = {Quantum Research Team},
  title = {Quantum Computing Package},
  year = {2024},
  version = {0.1.0}
}
``` 