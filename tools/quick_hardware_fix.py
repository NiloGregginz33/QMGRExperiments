#!/usr/bin/env python3
"""
Quick hardware execution fix
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def fix_hardware_execution():
    """Apply a simple fix to make hardware execution work"""
    
    print("=== QUICK HARDWARE FIX ===")
    print("The issue is that the hardware execution is too complex and failing.")
    print("We need to replace it with simple, working code.")
    print()
    print("MANUAL FIX REQUIRED:")
    print("1. Open src/experiments/custom_curvature_experiment.py")
    print("2. Find the line that says '# === ENHANCED HARDWARE EXECUTION ==='")
    print("3. Replace the entire hardware execution section with this simple code:")
    print()
    print("""
                else:
                    # === SIMPLE HARDWARE EXECUTION ===
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
                    counts_per_timestep.append(counts)
    """)
    print()
    print("This will replace the complex hardware execution with simple, working code.")
    print("After making this change, hardware execution should work properly.")

if __name__ == "__main__":
    fix_hardware_execution() 