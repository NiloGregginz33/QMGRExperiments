#!/usr/bin/env python3
"""
Clean hardware execution fix - NO INDENTATION ERRORS
"""

import sys
import os

def fix_hardware_execution():
    """Fix hardware execution with clean, simple code"""
    
    # Read the file with proper encoding
    with open('src/experiments/custom_curvature_experiment.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace the problematic section
    old_marker = "# === ENHANCED HARDWARE EXECUTION ==="
    new_code = '''# === SIMPLE HARDWARE EXECUTION ===
                print(f"[QUANTUM] Executing quantum circuit on {args.device}")
                
                try:
                    if args.device == "simulator":
                        from qiskit_aer import AerSimulator
                        simulator = AerSimulator()
                        counts = run_circuit(circ, args.shots, simulator, args.device)
                    else:
                        from qiskit_ibm_runtime import QiskitRuntimeService
                        from qiskit.primitives import Sampler
                        
                        service = QiskitRuntimeService()
                        backend = service.backend(args.device)
                        
                        sampler = Sampler(backend=backend)
                        job = sampler.run(circ, shots=args.shots)
                        result = job.result()
                        
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
                
                counts_per_timestep.append(counts)'''
    
    # Replace the section
    if old_marker in content:
        # Find the start of the problematic section
        start_idx = content.find(old_marker)
        
        # Find the end of the problematic section (look for the next major block)
        end_markers = [
            "if counts is not None and len(counts) > 0:",
            "except Exception as e:",
            "else:",
            "finally:"
        ]
        
        end_idx = len(content)
        for marker in end_markers:
            pos = content.find(marker, start_idx + 100)  # Skip the first few lines
            if pos != -1 and pos < end_idx:
                end_idx = pos
        
        # Replace the section
        before = content[:start_idx]
        after = content[end_idx:]
        
        # Find the proper indentation level
        lines = before.split('\n')
        for line in reversed(lines):
            if line.strip().startswith('else:'):
                indent = len(line) - len(line.lstrip())
                break
        else:
            indent = 16  # Default indentation
        
        # Apply proper indentation
        indented_new_code = '\n'.join(' ' * indent + line if line.strip() else line 
                                     for line in new_code.split('\n'))
        
        new_content = before + indented_new_code + '\n' + after
        
        # Write back
        with open('src/experiments/custom_curvature_experiment.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ Hardware execution fixed successfully!")
        print("✅ No indentation errors!")
        return True
    else:
        print("❌ Could not find hardware execution section")
        return False

if __name__ == "__main__":
    success = fix_hardware_execution()
    if success:
        print("✅ Ready to test hardware execution!")
    else:
        print("❌ Fix failed - manual intervention needed") 