#!/usr/bin/env python3
"""
Robust Hardware Runner
Ensures experiments run on real quantum hardware with comprehensive validation.
"""

import sys
import os
import argparse
import json
import time
import numpy as np
from datetime import datetime
import subprocess
import traceback

# Add Factory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from CGPTFactory import run
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

class HardwareValidator:
    """Comprehensive hardware validation and execution system."""
    
    def __init__(self):
        self.service = None
        self.validation_results = {}
        
    def connect_to_ibm(self):
        """Connect to IBM Quantum Runtime Service."""
        try:
            self.service = QiskitRuntimeService()
            print("✅ Connected to IBM Quantum Runtime Service")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to IBM Quantum: {e}")
            return False
    
    def validate_backend(self, backend_name):
        """
        Comprehensive backend validation.
        
        Args:
            backend_name (str): Name of the backend to validate
            
        Returns:
            dict: Validation results
        """
        validation = {
            'backend_name': backend_name,
            'is_real_hardware': False,
            'backend_type': None,
            'error': None,
            'properties': {},
            'validation_tests': {},
            'noise_signatures': {}
        }
        
        try:
            # Get backend
            backend = self.service.backend(backend_name)
            
            # Check if it's a simulator
            if hasattr(backend, 'name'):
                backend_name_lower = backend.name.lower()
                if any(sim_keyword in backend_name_lower for sim_keyword in ['fake', 'simulator', 'aer']):
                    validation['error'] = f"Backend '{backend_name}' is a simulator, not real hardware"
                    validation['backend_type'] = 'simulator'
                    return validation
            
            # Get backend properties
            try:
                properties = backend.properties()
                validation['properties'] = {
                    'qubits': len(properties.qubits) if hasattr(properties, 'qubits') else 'unknown',
                    'gates': list(properties.gates.keys()) if hasattr(properties, 'gates') else [],
                    'backend_version': getattr(backend, 'version', 'unknown')
                }
            except Exception as e:
                validation['properties']['error'] = str(e)
            
            # Check backend status
            try:
                status = backend.status()
                validation['validation_tests']['status_check'] = {
                    'operational': status.operational,
                    'pending_jobs': status.pending_jobs,
                    'status_msg': status.status_msg
                }
            except Exception as e:
                validation['validation_tests']['status_check'] = {'error': str(e)}
            
            # Run comprehensive noise test
            noise_test = self._run_noise_test(backend)
            validation['validation_tests']['noise_test'] = noise_test
            
            # Determine if this looks like real hardware
            is_real_hardware = (
                status.operational and
                noise_test['noise_indicators']['uniformity_score'] < 0.95 and
                (noise_test['noise_indicators']['has_readout_errors'] or 
                 noise_test['noise_indicators']['has_decoherence'] or
                 noise_test['noise_indicators']['has_crosstalk'])
            )
            
            validation['is_real_hardware'] = is_real_hardware
            validation['backend_type'] = 'real_hardware' if is_real_hardware else 'simulator'
            
        except Exception as e:
            validation['error'] = f"Failed to validate backend: {e}"
            traceback.print_exc()
        
        return validation
    
    def _run_noise_test(self, backend):
        """Run comprehensive noise signature test."""
        noise_test = {
            'counts': {},
            'noise_indicators': {
                'has_readout_errors': False,
                'has_decoherence': False,
                'has_crosstalk': False,
                'uniformity_score': 0.0,
                'bitstring_variance': 0.0,
                'total_shots': 0
            },
            'error': None
        }
        
        try:
            # Create test circuit with known quantum behavior
            test_circuit = QuantumCircuit(3, 3)
            test_circuit.h(0)  # Create superposition
            test_circuit.cx(0, 1)  # Create entanglement
            test_circuit.cx(1, 2)  # Create chain entanglement
            test_circuit.measure([0, 1, 2], [0, 1, 2])
            
            # Run on hardware
            counts = run(test_circuit, backend=backend, shots=1000)
            
            # Analyze results for noise signatures
            total_shots = sum(counts.values())
            bitstrings = list(counts.keys())
            
            # Calculate uniformity (real hardware should be less uniform than simulator)
            expected_uniform = total_shots / len(bitstrings)
            actual_variance = np.var(list(counts.values()))
            uniformity_score = 1.0 - (actual_variance / (expected_uniform ** 2))
            
            # Check for readout errors (should see some '001', '010', '100' states)
            readout_states = [bs for bs in bitstrings if bs in ['001', '010', '100']]
            if readout_states:
                readout_count = sum(counts[bs] for bs in readout_states)
                if readout_count > 0:
                    noise_test['noise_indicators']['has_readout_errors'] = True
            
            # Check for decoherence (should see some '111' states)
            if '111' in counts and counts['111'] > 0:
                noise_test['noise_indicators']['has_decoherence'] = True
            
            # Check for crosstalk (should see some '011', '101', '110' states)
            crosstalk_states = [bs for bs in bitstrings if bs in ['011', '101', '110']]
            if crosstalk_states:
                crosstalk_count = sum(counts[bs] for bs in crosstalk_states)
                if crosstalk_count > 0:
                    noise_test['noise_indicators']['has_crosstalk'] = True
            
            noise_test['counts'] = counts
            noise_test['noise_indicators']['uniformity_score'] = uniformity_score
            noise_test['noise_indicators']['bitstring_variance'] = actual_variance
            noise_test['noise_indicators']['total_shots'] = total_shots
            
        except Exception as e:
            noise_test['error'] = str(e)
            traceback.print_exc()
        
        return noise_test
    
    def run_experiment_with_validation(self, experiment_script, backend_name, **kwargs):
        """
        Run an experiment with comprehensive hardware validation.
        
        Args:
            experiment_script (str): Path to experiment script
            backend_name (str): Backend name
            **kwargs: Additional arguments for the experiment
            
        Returns:
            dict: Experiment results with validation
        """
        print(f"🔍 Validating hardware backend: {backend_name}")
        
        # Connect to IBM
        if not self.connect_to_ibm():
            return {
                'success': False,
                'error': "Failed to connect to IBM Quantum Runtime Service",
                'validation': None
            }
        
        # Validate the backend
        validation = self.validate_backend(backend_name)
        
        if not validation['is_real_hardware']:
            print(f"❌ VALIDATION FAILED: {validation['error']}")
            print(f"   Backend type: {validation['backend_type']}")
            print(f"   Noise indicators: {validation['validation_tests']['noise_test']['noise_indicators']}")
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
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=7200)
            
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
                'error': "Experiment timed out after 2 hours",
                'validation': validation
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to run experiment: {e}",
                'validation': validation
            }
    
    def list_available_backends(self):
        """List all available backends with their types and status."""
        if not self.connect_to_ibm():
            return
        
        try:
            backends = self.service.backends()
            
            print("Available backends:")
            print("-" * 100)
            print(f"{'Backend Name':<30} | {'Type':<10} | {'Status':<15} | {'Qubits':<8} | {'Pending Jobs':<15}")
            print("-" * 100)
            
            for backend in backends:
                backend_type = "simulator" if any(sim in backend.name.lower() for sim in ['fake', 'simulator', 'aer']) else "hardware"
                
                try:
                    status = backend.status()
                    operational = "operational" if status.operational else "not operational"
                    pending_jobs = str(status.pending_jobs)
                except:
                    operational = "unknown"
                    pending_jobs = "unknown"
                
                try:
                    properties = backend.properties()
                    num_qubits = str(len(properties.qubits)) if hasattr(properties, 'qubits') else "unknown"
                except:
                    num_qubits = "unknown"
                
                print(f"{backend.name:<30} | {backend_type:<10} | {operational:<15} | {num_qubits:<8} | {pending_jobs:<15}")
                
        except Exception as e:
            print(f"Error listing backends: {e}")

def main():
    parser = argparse.ArgumentParser(description="Robust hardware validation and experiment runner")
    parser.add_argument('--validate', type=str, help='Validate a specific backend')
    parser.add_argument('--list', action='store_true', help='List all available backends')
    parser.add_argument('--run-experiment', type=str, help='Path to experiment script')
    parser.add_argument('--backend', type=str, help='Backend name for experiment')
    parser.add_argument('--num-qubits', type=int, default=7, help='Number of qubits')
    parser.add_argument('--shots', type=int, default=20000, help='Number of shots')
    parser.add_argument('--curvature', type=float, default=0.5, help='Curvature parameter')
    parser.add_argument('--geometry', type=str, default='hyperbolic', help='Geometry type')
    parser.add_argument('--topology', type=str, default='triangulated', help='Topology type')
    parser.add_argument('--timesteps', type=int, default=3, help='Number of timesteps')
    
    args = parser.parse_args()
    
    validator = HardwareValidator()
    
    if args.list:
        validator.list_available_backends()
        return
    
    if args.validate:
        if not validator.connect_to_ibm():
            return
        validation = validator.validate_backend(args.validate)
        print(json.dumps(validation, indent=2, default=str))
        return
    
    if args.run_experiment and args.backend:
        result = validator.run_experiment_with_validation(
            args.run_experiment,
            args.backend,
            num_qubits=args.num_qubits,
            shots=args.shots,
            curvature=args.curvature,
            geometry=args.geometry,
            topology=args.topology,
            timesteps=args.timesteps
        )
        print(json.dumps(result, indent=2, default=str))
        return
    
    print("Usage:")
    print("  python robust_hardware_runner.py --list")
    print("  python robust_hardware_runner.py --validate ibm_brisbane")
    print("  python robust_hardware_runner.py --run-experiment src/experiments/custom_curvature_experiment.py --backend ibm_brisbane")

if __name__ == "__main__":
    main() 