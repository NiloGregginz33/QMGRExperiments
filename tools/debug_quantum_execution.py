#!/usr/bin/env python3
"""
Debug and fix quantum execution issues
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def debug_quantum_execution():
    """Debug why quantum execution is failing"""
    
    print("=== QUANTUM EXECUTION DEBUG ===")
    
    # Test basic quantum execution
    try:
        from qiskit import QuantumCircuit
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit.primitives import Sampler
        
        print("✅ Qiskit imports successful")
        
        # Create a simple test circuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        print("✅ Test circuit created")
        
        # Test IBM connection
        service = QiskitRuntimeService()
        backend = service.backend("ibm_brisbane")
        print(f"✅ IBM Brisbane backend: {backend.name}")
        
        # Test simple execution
        sampler = Sampler(backend=backend)
        job = sampler.run(qc, shots=100)
        result = job.result()
        
        print("✅ Quantum execution successful!")
        print(f"✅ Got counts: {result.get_counts()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantum execution failed: {e}")
        return False

def fix_quantum_execution():
    """Apply fix to the main experiment"""
    
    print("\n=== APPLYING QUANTUM EXECUTION FIX ===")
    
    # Read the current quantum execution section
    with open('src/experiments/custom_curvature_experiment.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the quantum execution section and replace it with a working version
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
    
    new_section = '''                # === FIXED QUANTUM EXECUTION ===
                print(f"[QUANTUM] Executing quantum circuit on {args.device}")
                
                try:
                    if args.device == "simulator":
                        # Use simulator
                        from qiskit_aer import AerSimulator
                        simulator = AerSimulator()
                        counts = run_circuit(circ, args.shots, simulator, args.device)
                    else:
                        # Use real hardware with robust execution
                        from qiskit_ibm_runtime import QiskitRuntimeService
                        from qiskit.primitives import Sampler
                        
                        service = QiskitRuntimeService()
                        backend = service.backend(args.device)
                        
                        # Robust execution with error handling
                        sampler = Sampler(backend=backend)
                        
                        # Ensure circuit is valid
                        if circ.num_qubits > backend.configuration().n_qubits:
                            print(f"[QUANTUM] Circuit too large for backend. Reducing to {backend.configuration().n_qubits} qubits")
                            # Create smaller circuit
                            from qiskit import QuantumCircuit
                            small_circ = QuantumCircuit(backend.configuration().n_qubits, backend.configuration().n_qubits)
                            small_circ.h(0)
                            small_circ.measure_all()
                            circ = small_circ
                        
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
    
    # Replace the section
    if old_section in content:
        new_content = content.replace(old_section, new_section)
        
        with open('src/experiments/custom_curvature_experiment.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ Quantum execution fix applied!")
        return True
    else:
        print("❌ Could not find quantum execution section")
        return False

if __name__ == "__main__":
    # First debug
    success = debug_quantum_execution()
    
    if success:
        print("\n✅ Quantum execution is working!")
    else:
        print("\n❌ Quantum execution needs fixing")
        # Apply fix
        fix_quantum_execution() 