#!/usr/bin/env python3
"""
Hardware Diagnostic Experiment
Identifies and fixes IBM hardware transpilation issues
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import logging
import numpy as np
import json
from datetime import datetime
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from Factory.CGPTFactory import run as cgpt_run, extract_counts_from_bitarray

def create_simple_test_circuit(n_qubits=3):
    """Create the simplest possible circuit that should work on hardware"""
    qc = QuantumCircuit(n_qubits)
    
    # Simple Hadamard on each qubit
    for i in range(n_qubits):
        qc.h(i)
    
    # Single CX gate between first two qubits
    if n_qubits > 1:
        qc.cx(0, 1)
    
    qc.measure_all()
    return qc

def create_medium_test_circuit(n_qubits=5):
    """Create a medium complexity circuit"""
    qc = QuantumCircuit(n_qubits)
    
    # Initial superposition
    for i in range(n_qubits):
        qc.h(i)
    
    # Simple entangling pattern
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    
    # Add some single-qubit rotations
    for i in range(n_qubits):
        qc.rz(0.1, i)
    
    qc.measure_all()
    return qc

def analyze_circuit(qc, backend_name="ibm_brisbane"):
    """Analyze circuit before and after transpilation"""
    print(f"\n=== CIRCUIT ANALYSIS ===")
    print(f"Original circuit depth: {qc.depth()}")
    print(f"Original circuit size: {qc.size()}")
    print(f"Original circuit gates: {qc.count_ops()}")
    
    # Transpile with different optimization levels
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    
    results = {}
    
    for opt_level in [0, 1, 2, 3]:
        print(f"\n--- Optimization Level {opt_level} ---")
        
        # Transpile
        tqc = transpile(qc, backend, optimization_level=opt_level)
        
        print(f"Transpiled depth: {tqc.depth()}")
        print(f"Transpiled size: {tqc.size()}")
        print(f"Transpiled gates: {tqc.count_ops()}")
        
        # Check if circuit is too complex
        if tqc.depth() > 100:
            print("⚠️  WARNING: Circuit depth > 100 - may cause issues")
        
        # Simulate the transpiled circuit
        try:
            statevector = Statevector.from_instruction(tqc)
            # Get measurement probabilities
            probs = statevector.probabilities_dict()
            print(f"Statevector probabilities (top 5): {dict(list(probs.items())[:5])}")
            
            # Check if all zeros
            zero_prob = probs.get('0' * tqc.num_qubits, 0)
            print(f"Probability of all zeros: {zero_prob:.4f}")
            
            results[f"opt_level_{opt_level}"] = {
                "depth": tqc.depth(),
                "size": tqc.size(),
                "gates": dict(tqc.count_ops()),
                "zero_probability": zero_prob,
                "has_entanglement": zero_prob < 0.9  # Simple check for entanglement
            }
            
        except Exception as e:
            print(f"❌ Error simulating transpiled circuit: {e}")
            results[f"opt_level_{opt_level}"] = {"error": str(e)}
    
    return results

def test_hardware_execution(qc, backend_name="ibm_brisbane", shots=1000):
    """Test actual hardware execution"""
    print(f"\n=== HARDWARE EXECUTION TEST ===")
    
    try:
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
        
        print(f"Backend: {backend.name}")
        print(f"Backend status: {backend.status()}")
        
        # Try different transpilation strategies
        strategies = [
            ("minimal", transpile(qc, backend, optimization_level=0)),
            ("balanced", transpile(qc, backend, optimization_level=1)),
            ("aggressive", transpile(qc, backend, optimization_level=2)),
            ("maximum", transpile(qc, backend, optimization_level=3))
        ]
        
        results = {}
        
        for strategy_name, tqc in strategies:
            print(f"\n--- Strategy: {strategy_name} ---")
            print(f"Circuit depth: {tqc.depth()}")
            print(f"Circuit size: {tqc.size()}")
            
            try:
                # Execute on hardware
                counts = cgpt_run(tqc, backend=backend, shots=shots)
                
                print(f"Raw counts: {dict(list(counts.items())[:5])}")
                
                # Analyze results
                total_shots = sum(counts.values())
                zero_count = counts.get('0' * tqc.num_qubits, 0)
                zero_ratio = zero_count / total_shots if total_shots > 0 else 0
                
                print(f"Total shots: {total_shots}")
                print(f"Zero count: {zero_count}")
                print(f"Zero ratio: {zero_ratio:.4f}")
                
                # Calculate entropy
                if total_shots > 0:
                    probs = {k: v/total_shots for k, v in counts.items()}
                    entropy = -sum(p * np.log2(p) for p in probs.values() if p > 0)
                    print(f"Shannon entropy: {entropy:.4f}")
                else:
                    entropy = 0
                    print("Shannon entropy: 0 (no valid measurements)")
                
                results[strategy_name] = {
                    "counts": dict(counts),
                    "total_shots": total_shots,
                    "zero_count": zero_count,
                    "zero_ratio": zero_ratio,
                    "entropy": entropy,
                    "success": True
                }
                
                if zero_ratio > 0.95:
                    print("❌ PROBLEM: >95% zeros - circuit collapsed to ground state")
                elif zero_ratio > 0.5:
                    print("⚠️  WARNING: >50% zeros - possible transpilation issues")
                else:
                    print("✅ SUCCESS: Reasonable measurement distribution")
                
            except Exception as e:
                print(f"❌ Hardware execution failed: {e}")
                results[strategy_name] = {
                    "error": str(e),
                    "success": False
                }
        
        return results
        
    except Exception as e:
        print(f"❌ Failed to connect to hardware: {e}")
        return {"error": str(e)}

def test_alternative_backends(qc, shots=1000):
    """Test different IBM backends"""
    print(f"\n=== ALTERNATIVE BACKEND TEST ===")
    
    try:
        service = QiskitRuntimeService()
        backends = service.backends()
        
        print(f"Available backends: {[b.name for b in backends]}")
        
        # Try different backends
        test_backends = ["ibm_brisbane", "ibm_kyoto", "ibm_osaka"]
        results = {}
        
        for backend_name in test_backends:
            if backend_name in [b.name for b in backends]:
                print(f"\n--- Testing {backend_name} ---")
                try:
                    backend = service.backend(backend_name)
                    print(f"Status: {backend.status()}")
                    
                    if backend.status().status == "active":
                        # Simple transpilation
                        tqc = transpile(qc, backend, optimization_level=1)
                        counts = cgpt_run(tqc, backend=backend, shots=shots)
                        
                        total_shots = sum(counts.values())
                        zero_count = counts.get('0' * tqc.num_qubits, 0)
                        zero_ratio = zero_count / total_shots if total_shots > 0 else 0
                        
                        print(f"Zero ratio: {zero_ratio:.4f}")
                        
                        results[backend_name] = {
                            "counts": dict(counts),
                            "zero_ratio": zero_ratio,
                            "success": True
                        }
                        
                        if zero_ratio < 0.5:
                            print(f"✅ {backend_name} works better!")
                            return backend_name, results[backend_name]
                    else:
                        print(f"Backend {backend_name} not active")
                        results[backend_name] = {"status": "inactive"}
                        
                except Exception as e:
                    print(f"Error with {backend_name}: {e}")
                    results[backend_name] = {"error": str(e)}
            else:
                print(f"Backend {backend_name} not available")
                results[backend_name] = {"status": "unavailable"}
        
        return None, results
        
    except Exception as e:
        print(f"❌ Failed to test alternative backends: {e}")
        return None, {"error": str(e)}

def create_fixed_circuit(n_qubits=5):
    """Create a circuit designed to work with IBM hardware limitations"""
    qc = QuantumCircuit(n_qubits)
    
    # Use only hardware-native gates
    # IBM Brisbane supports: h, x, cx, rz, sx
    
    # Simple initialization
    for i in range(n_qubits):
        qc.h(i)
    
    # Minimal entanglement - just nearest neighbors
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    
    # Add some rz rotations (these are native)
    for i in range(n_qubits):
        qc.rz(0.1, i)
    
    # Final Hadamard
    for i in range(n_qubits):
        qc.h(i)
    
    qc.measure_all()
    return qc

def main():
    parser = argparse.ArgumentParser(description="Diagnose IBM hardware issues")
    parser.add_argument("--n", type=int, default=5, help="Number of qubits")
    parser.add_argument("--shots", type=int, default=1000, help="Number of shots")
    parser.add_argument("--device", type=str, default="ibm_brisbane", help="Backend name")
    args = parser.parse_args()
    
    print("🔍 IBM HARDWARE DIAGNOSTIC")
    print("=" * 50)
    
    # Test 1: Simple circuit
    print("\n📋 TEST 1: Simple Circuit Analysis")
    simple_qc = create_simple_test_circuit(args.n)
    simple_analysis = analyze_circuit(simple_qc, args.device)
    
    # Test 2: Medium circuit
    print("\n📋 TEST 2: Medium Circuit Analysis")
    medium_qc = create_medium_test_circuit(args.n)
    medium_analysis = analyze_circuit(medium_qc, args.device)
    
    # Test 3: Hardware execution with simple circuit
    print("\n📋 TEST 3: Hardware Execution (Simple)")
    simple_hw_results = test_hardware_execution(simple_qc, args.device, args.shots)
    
    # Test 4: Hardware execution with medium circuit
    print("\n📋 TEST 4: Hardware Execution (Medium)")
    medium_hw_results = test_hardware_execution(medium_qc, args.device, args.shots)
    
    # Test 5: Alternative backends
    print("\n📋 TEST 5: Alternative Backends")
    best_backend, backend_results = test_alternative_backends(simple_qc, args.shots)
    
    # Test 6: Fixed circuit
    print("\n📋 TEST 6: Hardware-Optimized Circuit")
    fixed_qc = create_fixed_circuit(args.n)
    fixed_hw_results = test_hardware_execution(fixed_qc, args.device, args.shots)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "n_qubits": args.n,
        "shots": args.shots,
        "device": args.device,
        "simple_circuit_analysis": simple_analysis,
        "medium_circuit_analysis": medium_analysis,
        "simple_hardware_results": simple_hw_results,
        "medium_hardware_results": medium_hw_results,
        "backend_comparison": backend_results,
        "best_backend": best_backend,
        "fixed_circuit_results": fixed_hw_results
    }
    
    # Create results directory
    os.makedirs("experiment_logs", exist_ok=True)
    results_file = f"experiment_logs/hardware_diagnostic_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Summary
    print("\n📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    # Check if any strategy worked
    working_strategies = []
    for test_name, test_results in [("Simple", simple_hw_results), ("Medium", medium_hw_results), ("Fixed", fixed_hw_results)]:
        for strategy, result in test_results.items():
            if isinstance(result, dict) and result.get("success") and result.get("zero_ratio", 1) < 0.5:
                working_strategies.append(f"{test_name}-{strategy}")
    
    if working_strategies:
        print(f"✅ WORKING STRATEGIES: {working_strategies}")
    else:
        print("❌ NO WORKING STRATEGIES FOUND")
    
    if best_backend:
        print(f"✅ BEST BACKEND: {best_backend}")
    else:
        print("❌ NO BETTER BACKEND FOUND")
    
    print("\n🔧 RECOMMENDATIONS:")
    if working_strategies:
        print("1. Use one of the working strategies for future experiments")
        print("2. Consider using the best backend if different from current")
    else:
        print("1. Circuit transpilation is causing collapse to ground state")
        print("2. Try using only hardware-native gates (h, x, cx, rz, sx)")
        print("3. Reduce circuit depth and complexity")
        print("4. Consider using a different IBM backend")
        print("5. Check if there are hardware maintenance issues")

if __name__ == "__main__":
    main() 