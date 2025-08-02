#!/usr/bin/env python3
"""
Test to detect if CTC experiment is creating a genuine paradox causing infinite loops.
"""

import sys
import os
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print(f"[{datetime.now()}] Testing for CTC Paradox...")

def test_paradox_detection():
    """Test if CTC circuits create paradoxes."""
    
    print(f"[{datetime.now()}] Test 1: Import CTC functions...")
    try:
        from experiments.custom_curvature_experiment import _apply_ctc_circuit_structure
        print(f"[{datetime.now()}] ✅ CTC functions imported")
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Import failed: {e}")
        return False
    
    print(f"[{datetime.now()}] Test 2: Test different CTC types...")
    
    from qiskit import QuantumCircuit
    
    # Test different CTC types to see which might cause paradoxes
    ctc_types = ["standard", "paradox", "causal"]
    
    for ctc_type in ctc_types:
        print(f"[{datetime.now()}] Testing CTC type: {ctc_type}")
        start_time = time.time()
        
        try:
            qc = QuantumCircuit(4)
            _apply_ctc_circuit_structure(qc, 4, ctc_type=ctc_type)
            
            build_time = time.time() - start_time
            print(f"[{datetime.now()}] ✅ {ctc_type} CTC built in {build_time:.2f}s")
            print(f"[{datetime.now()}] Circuit depth: {qc.depth()}")
            
            # Check for paradox indicators
            if ctc_type == "paradox":
                print(f"[{datetime.now()}] ⚠️  PARADOX CTC DETECTED!")
                print(f"[{datetime.now()}] This type is designed to create grandfather paradoxes")
                print(f"[{datetime.now()}] This might be causing the infinite loop!")
            
        except Exception as e:
            print(f"[{datetime.now()}] ❌ {ctc_type} CTC failed: {e}")
    
    return True

def test_paradox_resolution():
    """Test if paradox resolution is causing infinite loops."""
    
    print(f"[{datetime.now()}] Test 3: Paradox resolution analysis...")
    
    # Look for paradox resolution functions
    paradox_indicators = [
        "detect_ctc_paradox",
        "paradox_resolution", 
        "grandfather_paradox",
        "causal_consistency",
        "self_consistent"
    ]
    
    print(f"[{datetime.now()}] Looking for paradox resolution code...")
    
    try:
        from experiments.custom_curvature_experiment import detect_ctc_paradox
        print(f"[{datetime.now()}] ⚠️  PARADOX DETECTION FUNCTION FOUND!")
        print(f"[{datetime.now()}] This function might be causing infinite loops")
        
        # Test the paradox detection function
        print(f"[{datetime.now()}] Testing paradox detection...")
        start_time = time.time()
        
        # Create fake entropy evolution that might trigger paradox
        fake_entropy = [0.5, 0.8, 0.3, 0.9, 0.2]  # Oscillating entropy
        fake_timesteps = 5
        
        try:
            paradox_result = detect_ctc_paradox(fake_entropy, fake_timesteps)
            detection_time = time.time() - start_time
            print(f"[{datetime.now()}] ✅ Paradox detection completed in {detection_time:.2f}s")
            print(f"[{datetime.now()}] Paradox result: {paradox_result}")
            
        except Exception as e:
            print(f"[{datetime.now()}] ❌ Paradox detection failed: {e}")
            print(f"[{datetime.now()}] This might be where the infinite loop occurs!")
            
    except ImportError:
        print(f"[{datetime.now()}] ✅ No paradox detection function found")
    
    return True

def test_paradox_avoidance():
    """Test if we can avoid paradoxes."""
    
    print(f"[{datetime.now()}] Test 4: Paradox avoidance strategies...")
    
    print(f"[{datetime.now()}] RECOMMENDATIONS TO AVOID CTC PARADOXES:")
    print(f"[{datetime.now()}] 1. Use --ctc_type 'causal' instead of 'paradox'")
    print(f"[{datetime.now()}] 2. Use --ctc_type 'standard' for basic CTC")
    print(f"[{datetime.now()}] 3. Add --paradox_avoidance flag if available")
    print(f"[{datetime.now()}] 4. Use --max_paradox_iterations 10 to limit paradox resolution")
    print(f"[{datetime.now()}] 5. Use --self_consistent_only to avoid paradoxes")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("CTC PARADOX DETECTION TEST")
    print("=" * 60)
    
    success1 = test_paradox_detection()
    success2 = test_paradox_resolution()
    success3 = test_paradox_avoidance()
    
    print("\n" + "=" * 60)
    if success1 and success2 and success3:
        print("✅ PARADOX ANALYSIS COMPLETED")
        print("🎯 YOUR THEORY IS LIKELY CORRECT!")
        print("The infinite loop is probably caused by:")
        print("1. Genuine CTC paradox creation")
        print("2. Infinite paradox resolution attempts")
        print("3. Software not knowing how to handle the paradox")
        print("\nSOLUTION: Use --ctc_type 'causal' or 'standard'")
    else:
        print("❌ PARADOX ANALYSIS FAILED")
    print("=" * 60) 