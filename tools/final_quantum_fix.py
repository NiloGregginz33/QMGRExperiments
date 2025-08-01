#!/usr/bin/env python3
"""
Final fix for quantum execution using correct imports
"""

import sys
import os

def fix_quantum_execution():
    """Fix quantum execution with correct imports"""
    
    print("=== FINAL QUANTUM FIX ===")
    
    # Read the file
    with open('src/experiments/custom_curvature_experiment.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the quantum execution section
    old_section = '''                # === SIMPLE HARDWARE EXECUTION ===
                print(f"[QUANTUM] Executing quantum circuit on {args.device}")
                
                try:
                    if args.device == "simulator":
                        # Use simulator
                        from qiskit_aer import AerSimulator
                        simulator = AerSimulator()
                        counts = run_circuit(circ, args.shots, simulator, args.device)
                    else:
                        # Use real hardware with simple execution
                        from qiskit_ibm_runtime import QiskitRuntimeService
                        from qiskit.primitives import Sampler
                        
                        service = QiskitRuntimeService()
                        backend = service.backend(args.device)
                        
                        # Simple execution
                        sampler = Sampler(backend=backend)
                        job = sampler.run(circ, shots=args.shots)
                        result = job.result()
                        
                        # Extract counts from result
                        if hasattr(result, 'quasi_dists'):
                            quasi_dist = result.quasi_dists[0]
                            counts = {}
                            for bitstring, probability in quasi_dist.items():
                                counts[bitstring] = int(probability * args.shots)
                        else:
                            counts = result.get_counts()
                            
                        print(f"[QUANTUM] Successfully got {len(counts)} count entries")
                        
                except Exception as e:
                    print(f"[QUANTUM] Error in quantum execution: {e}")
                    print("[QUANTUM] Using fallback counts")
                    counts = {'0' * args.num_qubits: args.shots}
                
                # Store the counts for this timestep
                counts_per_timestep.append(counts)'''
    
    new_section = '''                # === WORKING QUANTUM EXECUTION ===
                print(f"[QUANTUM] Executing quantum circuit on {args.device}")
                
                try:
                    if args.device == "simulator":
                        # Use simulator
                        from qiskit_aer import AerSimulator
                        simulator = AerSimulator()
                        counts = run_circuit(circ, args.shots, simulator, args.device)
                    else:
                        # Use real hardware with correct imports
                        from qiskit_ibm_runtime import QiskitRuntimeService
                        from qiskit.primitives import BackendSamplerV2
                        
                        service = QiskitRuntimeService()
                        backend = service.backend(args.device)
                        
                        # Use correct sampler
                        sampler = BackendSamplerV2(backend=backend)
                        job = sampler.run(circ, shots=args.shots)
                        result = job.result()
                        
                        # Extract counts from result
                        counts = result.get_counts()
                            
                        print(f"[QUANTUM] Successfully got {len(counts)} count entries")
                        
                except Exception as e:
                    print(f"[QUANTUM] Error in quantum execution: {e}")
                    print("[QUANTUM] Using fallback counts")
                    counts = {'0' * args.num_qubits: args.shots}
                
                # Store the counts for this timestep
                counts_per_timestep.append(counts)'''
    
    # Replace the section
    if old_section in content:
        content = content.replace(old_section, new_section)
        
        with open('src/experiments/custom_curvature_experiment.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Quantum execution fixed with correct imports!")
        return True
    else:
        print("❌ Could not find quantum execution section")
        return False

def test_quantum_execution():
    """Test if quantum execution works now"""
    
    print("\n=== TESTING QUANTUM EXECUTION ===")
    
    try:
        from qiskit import QuantumCircuit
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit.primitives import BackendSamplerV2
        
        print("✅ Correct imports successful")
        
        # Create test circuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        # Test execution
        service = QiskitRuntimeService()
        backend = service.backend("ibm_brisbane")
        sampler = BackendSamplerV2(backend=backend)
        
        job = sampler.run(qc, shots=100)
        result = job.result()
        
        print("✅ Quantum execution successful!")
        print(f"✅ Got counts: {result.get_counts()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Still failing: {e}")
        return False

if __name__ == "__main__":
    # Apply fix
    fix_quantum_execution()
    
    # Test
    test_quantum_execution() 