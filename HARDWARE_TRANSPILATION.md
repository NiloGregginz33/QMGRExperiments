# Quantum Hardware Transpilation System

## Overview

This document describes the quantum circuit transpilation system that has been integrated into all experiments to ensure optimal performance on different quantum hardware devices.

## Features

### 1. Device-Specific Transpilation

All experiments now automatically transpile quantum circuits for the target hardware device before execution. This ensures:

- **Optimal gate decompositions** for each device's native gate set
- **Connectivity routing** for devices with limited qubit connectivity
- **Circuit depth optimization** to minimize decoherence effects
- **Error mitigation** through device-specific optimizations

### 2. Supported Hardware

The system supports the following quantum devices:

- **Local Simulator**: For testing and development
- **IonQ**: Ion trap quantum computer
- **Rigetti**: Superconducting quantum computer  
- **OQC**: Oxford Quantum Circuits device

### 3. Transpilation Process

For each experiment, the transpilation process:

1. **Extracts device properties** (native gates, connectivity, qubit count)
2. **Analyzes circuit requirements** (gate types, qubit interactions)
3. **Applies device-specific optimizations**:
   - IonQ: Optimizes for single-qubit gates and CNOT operations
   - Rigetti: Handles connectivity constraints with SWAP insertion
   - OQC: Adapts to specific gate set requirements
4. **Validates transpiled circuit** before execution

## Usage

### Running Experiments on Hardware

```bash
# Run on simulator (default)
python src/experiments/emergent_spacetime.py --device simulator --shots 1024

# Run on IonQ hardware
python src/experiments/emergent_spacetime.py --device ionq --shots 1000

# Run on Rigetti hardware
python src/experiments/emergent_spacetime.py --device rigetti --shots 1000

# Run on OQC hardware
python src/experiments/emergent_spacetime.py --device oqc --shots 1000
```

### Using the Hardware Experiment Runner

```bash
# Run all experiments on all devices
python run_hardware_experiments.py --device all --experiment all

# Run specific experiment on specific device
python run_hardware_experiments.py --device ionq --experiment emergent_spacetime

# Dry run to see what would be executed
python run_hardware_experiments.py --device all --experiment all --dry-run
```

## Implementation Details

### Transpiler Architecture

The transpilation system consists of:

1. **QuantumTranspiler Class** (`src/utils/transpiler.py`):
   - Main transpilation engine
   - Device property extraction
   - Device-specific optimization methods

2. **Device-Specific Methods**:
   - `_transpile_for_ionq()`: IonQ optimizations
   - `_transpile_for_rigetti()`: Rigetti connectivity handling
   - `_transpile_for_oqc()`: OQC gate set adaptation
   - `_apply_basic_optimizations()`: General optimizations

3. **Integration Points**:
   - All experiment scripts use `transpile_circuit()` function
   - Device information is logged for each run
   - Circuit depth comparisons are displayed

### Circuit Optimization Features

- **Gate Cancellation**: Removes redundant gate sequences
- **Depth Reduction**: Minimizes circuit depth for better fidelity
- **Connectivity Routing**: Handles limited qubit connectivity
- **Native Gate Decomposition**: Converts to device-specific gate sets

## Output and Logging

### Transpilation Logs

Each experiment run includes detailed transpilation information:

```
INFO:utils.transpiler:Transpiling circuit for device: IonQ Device
INFO:utils.transpiler:Original circuit depth: 12
INFO:utils.transpiler:Applying IonQ-specific optimizations...
INFO:utils.transpiler:Transpiled circuit depth: 10
```

### Device Information

Device properties are logged at the start of each experiment:

```
Device: IonQ Device
Native gates: ['H', 'X', 'Y', 'Z', 'S', 'T', 'Rx', 'Ry', 'Rz', 'CNOT', 'CZ', 'SWAP']
Qubit count: 11
```

### Results Organization

Results are organized by device type:

```
experiment_logs/
├── emergent_spacetime_simulator/
├── emergent_spacetime_ionq/
├── emergent_spacetime_rigetti/
└── emergent_spacetime_oqc/
```

## Error Handling

The transpilation system includes robust error handling:

- **Graceful degradation**: Falls back to original circuit if transpilation fails
- **Device property extraction**: Handles missing or incomplete device information
- **Circuit validation**: Ensures transpiled circuits are valid before execution
- **Detailed logging**: Provides comprehensive error information

## Performance Considerations

### Shot Counts

Different devices have different optimal shot counts:

- **Simulator**: 1024+ shots for high precision
- **IonQ**: 100-1000 shots (higher fidelity, slower execution)
- **Rigetti**: 100-1000 shots (moderate fidelity)
- **OQC**: 100-1000 shots (moderate fidelity)

### Execution Time

Hardware execution times vary significantly:

- **Simulator**: Seconds to minutes
- **Real hardware**: Minutes to hours (including queue time)

## Future Enhancements

Planned improvements to the transpilation system:

1. **Advanced Optimizations**:
   - Quantum error correction codes
   - Noise-aware transpilation
   - Dynamic circuit optimization

2. **Additional Devices**:
   - IBM Quantum devices
   - Google Quantum devices
   - Other cloud quantum platforms

3. **Enhanced Routing**:
   - Machine learning-based routing
   - Multi-objective optimization
   - Real-time connectivity adaptation

## Troubleshooting

### Common Issues

1. **Device Properties Not Found**:
   - Normal for simulators
   - Check AWS credentials for real hardware

2. **Transpilation Warnings**:
   - Usually indicates device-specific limitations
   - Circuit will still execute with basic optimizations

3. **Hardware Queue Times**:
   - Real hardware may have long queue times
   - Consider using simulators for development

### Getting Help

For issues with hardware transpilation:

1. Check the experiment logs in `experiment_logs/` directories
2. Verify AWS credentials and permissions
3. Ensure device availability and queue status
4. Review transpilation warnings and error messages

## Conclusion

The quantum hardware transpilation system ensures that all experiments can run optimally on different quantum devices while maintaining scientific rigor and reproducibility. The system automatically adapts circuits to device-specific constraints and provides detailed logging for analysis and debugging. 