#!/usr/bin/env python3
"""
Fix quantum execution using CGPTFactory run function
"""

import sys
import os

def fix_with_cgptfactory():
    """Replace quantum execution with CGPTFactory run function"""
    
    print("=== FIXING WITH CGPTFACTORY ===")
    
    # Read the file
    with open('src/experiments/custom_curvature_experiment.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace all run_circuit_with_mitigation calls with CGPTFactory run function
    old_calls = [
        "counts = run_circuit_with_mitigation(circ, args.shots, args.device, use_mitigation=False)",
        "counts = run_circuit_with_mitigation(circ_optimized, args.shots, args.device, use_mitigation=False)"
    ]
    
    new_call = '''                    # Use CGPTFactory run function
                    from CGPTFactory import run
                    
                    # Execute the circuit using CGPTFactory
                    if args.device == "simulator":
                        # Use FakeBrisbane for simulation
                        from qiskit_aer import FakeBrisbane
                        backend = FakeBrisbane()
                    else:
                        # Use real hardware
                        from qiskit_ibm_runtime import QiskitRuntimeService
                        service = QiskitRuntimeService()
                        backend = service.backend(args.device)
                    
                    # Run the circuit using CGPTFactory
                    result = run(circ, backend=backend, shots=args.shots)
                    
                    # Extract counts from result
                    if hasattr(result, 'get_counts'):
                        counts = result.get_counts()
                    elif hasattr(result, 'quasi_dists'):
                        quasi_dist = result.quasi_dists[0]
                        counts = {}
                        for bitstring, probability in quasi_dist.items():
                            counts[bitstring] = int(probability * args.shots)
                    else:
                        counts = result'''
    
    # Replace each call
    for old_call in old_calls:
        if old_call in content:
            content = content.replace(old_call, new_call)
            print(f"✅ Replaced: {old_call[:50]}...")
    
    # Write back the file
    with open('src/experiments/custom_curvature_experiment.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Quantum execution fixed with CGPTFactory!")
    return True

if __name__ == "__main__":
    fix_with_cgptfactory() 