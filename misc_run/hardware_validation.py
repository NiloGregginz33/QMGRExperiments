#!/usr/bin/env python3
"""
Hardware Validation Script
Ensures experiments run on real quantum hardware, not simulators.
"""

import sys
import os
import argparse
import json
import time
import numpy as np
from datetime import datetime

# Add Factory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Factory'))

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from CGPTFactory import run
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

def validate_hardware_backend(backend_name):
    """
    Validate that a backend is real hardware, not a simulator.
    
    Args:
        backend_name (str): Name of the backend to validate
        
    Returns:
        dict: Validation results
    """
    validation_results = {
        'backend_name': backend_name,
        'is_real_hardware': False,
        'backend_type': None,
        'error': None,
        'properties': {},
        'validation_tests': {}
    }
    
    try:
        # Connect to IBM Quantum
        service = QiskitRuntimeService()
        
        # Get backend
        backend = service.backend(backend_name)
        
        # Check if it's a simulator
        if hasattr(backend, 'name'):
            backend_name_lower = backend.name.lower()
            if any(sim_keyword in backend_name_lower for sim_keyword in ['fake', 'simulator', 'aer']):
                validation_results['error'] = f"Backend '{backend_name}' is a simulator, not real hardware"
                validation_results['backend_type'] = 'simulator'
                return validation_results
        
        # Get backend properties
        try:
            properties = backend.properties()
            validation_results['properties'] = {
                'qubits': len(properties.qubits) if hasattr(properties, 'qubits') else 'unknown',
                'gates': list(properties.gates.keys()) if hasattr(properties, 'gates') else [],
                'backend_version': getattr(backend, 'version', 'unknown')
            }
        except Exception as e:
            validation_results['properties']['error'] = str(e)
        
        # Check backend status
        try:
            status = backend.status()
            validation_results['validation_tests']['status_check'] = {
                'operational': status.operational,
                'pending_jobs': status.pending_jobs,
                'status_msg': status.status_msg
            }
        except Exception as e:
            validation_results['validation_tests']['status_check'] = {'error': str(e)}
        
        # Run a simple test circuit
        try:
            test_circuit = QuantumCircuit(2, 2)
            test_circuit.h(0)
            test_circuit.cx(0, 1)
            test_circuit.measure([0, 1], [0, 1])
            
            # Run on real hardware
            counts = run(test_circuit, backend=backend, shots=100)
            
            # Analyze results for noise signatures
            total_shots = sum(counts.values())
            bitstrings = list(counts.keys())
            
            # Check for noise patterns typical of real hardware
            noise_indicators = {
                'has_readout_errors': False,
                'has_decoherence': False,
                'has_crosstalk': False,
                'uniformity_score': 0.0
            }
            
            # Calculate uniformity (real hardware should be less uniform than simulator)
            expected_uniform = total_shots / len(bitstrings)
            actual_variance = np.var(list(counts.values()))
            uniformity_score = 1.0 - (actual_variance / (expected_uniform ** 2))
            noise_indicators['uniformity_score'] = uniformity_score
            
            # Check for readout errors (should see some '01' and '10' states)
            readout_states = [bs for bs in bitstrings if bs in ['01', '10']]
            if readout_states:
                readout_count = sum(counts[bs] for bs in readout_states)
                if readout_count > 0:
                    noise_indicators['has_readout_errors'] = True
            
            # Check for decoherence (should see some '11' states)
            if '11' in counts and counts['11'] > 0:
                noise_indicators['has_decoherence'] = True
            
            validation_results['validation_tests']['noise_test'] = {
                'counts': counts,
                'noise_indicators': noise_indicators,
                'total_shots': total_shots,
                'bitstring_variance': actual_variance
            }
            
            # Determine if this looks like real hardware
            is_real_hardware = (
                status.operational and
                uniformity_score < 0.95 and  # Real hardware is less uniform
                (noise_indicators['has_readout_errors'] or noise_indicators['has_decoherence'])
            )
            
            validation_results['is_real_hardware'] = is_real_hardware
            validation_results['backend_type'] = 'real_hardware' if is_real_hardware else 'simulator'
            
        except Exception as e:
            validation_results['validation_tests']['noise_test'] = {'error': str(e)}
            validation_results['error'] = f"Failed to run test circuit: {e}"
        
    except Exception as e:
        validation_results['error'] = f"Failed to validate backend: {e}"
    
    return validation_results

def run_hardware_experiment_with_validation(experiment_script, backend_name, **kwargs):
    """
    Run an experiment with hardware validation.
    
    Args:
        experiment_script (str): Path to experiment script
        backend_name (str): Backend name
        **kwargs: Additional arguments for the experiment
        
    Returns:
        dict: Experiment results with validation
    """
    print(f"🔍 Validating hardware backend: {backend_name}")
    
    # Validate the backend first
    validation = validate_hardware_backend(backend_name)
    
    if not validation['is_real_hardware']:
        print(f"❌ VALIDATION FAILED: {validation['error']}")
        print(f"   Backend type: {validation['backend_type']}")
        return {
            'success': False,
            'error': f"Backend validation failed: {validation['error']}",
            'validation': validation
        }
    
    print(f"✅ VALIDATION PASSED: {backend_name} is confirmed real hardware")
    print(f"   Noise indicators: {validation['validation_tests']['noise_test']['noise_indicators']}")
    
    # Run the experiment
    print(f"🚀 Running experiment on validated hardware...")
    
    # Construct command
    cmd_parts = ['python', experiment_script, '--device', backend_name]
    for key, value in kwargs.items():
        if value is not None:
            cmd_parts.extend([f'--{key}', str(value)])
    
    cmd = ' '.join(cmd_parts)
    print(f"Command: {cmd}")
    
    # Execute the experiment
    import subprocess
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print("✅ Experiment completed successfully")
            return {
                'success': True,
                'validation': validation,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            print(f"❌ Experiment failed with return code {result.returncode}")
            return {
                'success': False,
                'error': f"Experiment failed: {result.stderr}",
                'validation': validation,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': "Experiment timed out after 1 hour",
            'validation': validation
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Failed to run experiment: {e}",
            'validation': validation
        }

def list_available_backends():
    """List all available backends with their types."""
    try:
        service = QiskitRuntimeService()
        backends = service.backends()
        
        print("Available backends:")
        print("-" * 80)
        
        for backend in backends:
            backend_type = "simulator" if any(sim in backend.name.lower() for sim in ['fake', 'simulator', 'aer']) else "hardware"
            status = "operational" if backend.status().operational else "not operational"
            print(f"{backend.name:<30} | {backend_type:<10} | {status}")
            
    except Exception as e:
        print(f"Error listing backends: {e}")

def main():
    parser = argparse.ArgumentParser(description="Hardware validation and experiment runner")
    parser.add_argument('--validate', type=str, help='Validate a specific backend')
    parser.add_argument('--list', action='store_true', help='List all available backends')
    parser.add_argument('--run-experiment', type=str, help='Path to experiment script')
    parser.add_argument('--backend', type=str, help='Backend name for experiment')
    parser.add_argument('--num-qubits', type=int, default=7, help='Number of qubits')
    parser.add_argument('--shots', type=int, default=20000, help='Number of shots')
    parser.add_argument('--curvature', type=float, default=0.5, help='Curvature parameter')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_backends()
        return
    
    if args.validate:
        validation = validate_hardware_backend(args.validate)
        print(json.dumps(validation, indent=2, default=str))
        return
    
    if args.run_experiment and args.backend:
        result = run_hardware_experiment_with_validation(
            args.run_experiment,
            args.backend,
            num_qubits=args.num_qubits,
            shots=args.shots,
            curvature=args.curvature
        )
        print(json.dumps(result, indent=2, default=str))
        return
    
    print("Usage:")
    print("  python hardware_validation.py --list")
    print("  python hardware_validation.py --validate ibm_brisbane")
    print("  python hardware_validation.py --run-experiment src/experiments/custom_curvature_experiment.py --backend ibm_brisbane")

if __name__ == "__main__":
    main() 