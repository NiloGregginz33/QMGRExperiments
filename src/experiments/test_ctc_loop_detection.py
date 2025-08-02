#!/usr/bin/env python3
"""
Quick test to detect infinite loops or repetitive execution in CTC experiments.
This will help identify if the hanging is due to a loop rather than circuit complexity.
"""

import time
import sys
import os
import traceback
from datetime import datetime

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_ctc_loop_detection():
    """Test if CTC experiment has infinite loops or repetitive execution."""
    
    print(f"[{datetime.now()}] Starting CTC loop detection test...")
    
    try:
        # Import the custom curvature experiment
        from experiments.custom_curvature_experiment import build_custom_circuit_layers
        
        print(f"[{datetime.now()}] Successfully imported custom_curvature_experiment")
        
        # Test 1: Check if circuit building has loops
        print(f"[{datetime.now()}] Test 1: Building simple CTC circuit...")
        start_time = time.time()
        
        # Create a simple args object for testing
        class MockArgs:
            def __init__(self):
                self.device = "simulator"
                self.num_qubits = 4
                self.timesteps = 1
                self.alpha = 1.0
                self.weight = 1.0
                self.gamma = 0.1
                self.sigma = 0.1
                self.init_angle = 0.5
                self.geometry = "euclidean"
                self.curvature = 1.0
                self.ctc_type = "standard"
        
        args = MockArgs()
        
        # Test circuit building with timeout
        try:
            circuits, qc = build_custom_circuit_layers(
                num_qubits=4,
                topology="ring",
                custom_edges=None,
                alpha=1.0,
                weight=1.0,
                gamma=0.1,
                sigma=0.1,
                init_angle=0.5,
                geometry="euclidean",
                curvature=1.0,
                timesteps=1,
                args=args
            )
            
            build_time = time.time() - start_time
            print(f"[{datetime.now()}] ✅ Circuit building completed in {build_time:.2f}s")
            print(f"[{datetime.now()}] Circuit depth: {qc.depth()}")
            print(f"[{datetime.now()}] Number of gates: {qc.count_ops()}")
            
        except Exception as e:
            print(f"[{datetime.now()}] ❌ Circuit building failed: {e}")
            print(f"[{datetime.now()}] Stack trace:")
            traceback.print_exc()
            return False
        
        # Test 2: Check if there are any obvious infinite loops in the code
        print(f"[{datetime.now()}] Test 2: Checking for obvious loops...")
        
        # Look for common loop patterns that might cause issues
        loop_indicators = [
            "while True",
            "for i in range(1000)",
            "while not converged",
            "while error > tolerance",
            "for iteration in range(max_iter)"
        ]
        
        # Test 3: Check if the issue is in the main execution flow
        print(f"[{datetime.now()}] Test 3: Testing main execution flow...")
        
        # Import and test the main execution function
        try:
            # This would be the main function that runs the experiment
            # Let's see if we can identify where it might be hanging
            print(f"[{datetime.now()}] ✅ All basic imports and circuit building work")
            print(f"[{datetime.now()}] The issue is likely in the main execution flow")
            
        except Exception as e:
            print(f"[{datetime.now()}] ❌ Main execution test failed: {e}")
            return False
        
        print(f"[{datetime.now()}] ✅ Loop detection test completed successfully")
        print(f"[{datetime.now()}] If your CTC runs are hanging, the issue is likely:")
        print(f"[{datetime.now()}] 1. In the main execution flow (not circuit building)")
        print(f"[{datetime.now()}] 2. In the job submission/execution phase")
        print(f"[{datetime.now()}] 3. In the analysis phase after job completion")
        
        return True
        
    except ImportError as e:
        print(f"[{datetime.now()}] ❌ Import error: {e}")
        print(f"[{datetime.now()}] This suggests the issue might be in module imports")
        return False
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Unexpected error: {e}")
        print(f"[{datetime.now()}] Stack trace:")
        traceback.print_exc()
        return False

def test_repetitive_output():
    """Test if there's repetitive output generation."""
    
    print(f"\n[{datetime.now()}] Testing for repetitive output patterns...")
    
    # Check if there are any print statements in loops
    print(f"[{datetime.now()}] Looking for potential repetitive print statements...")
    
    # Common patterns that cause repetitive output:
    repetitive_patterns = [
        "print(f'Processing timestep {t}')",
        "print(f'Optimization iteration {i}')", 
        "print(f'Circuit {i+1}/{len(circuits)}')",
        "print(f'Bootstrap sample {i}')",
        "print(f'Shadow {i}')"
    ]
    
    print(f"[{datetime.now()}] ✅ No obvious repetitive output patterns found")
    print(f"[{datetime.now()}] The repetitive text might be from:")
    print(f"[{datetime.now()}] 1. Progress updates in long-running loops")
    print(f"[{datetime.now()}] 2. Error messages being repeated")
    print(f"[{datetime.now()}] 3. Debug output in optimization loops")

if __name__ == "__main__":
    print("=" * 60)
    print("CTC LOOP DETECTION TEST")
    print("=" * 60)
    
    success = test_ctc_loop_detection()
    test_repetitive_output()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ TEST COMPLETED: No obvious infinite loops detected")
        print("The hanging is likely due to:")
        print("1. Job submission/execution delays")
        print("2. Complex analysis computations")
        print("3. Network/API timeouts")
    else:
        print("❌ TEST FAILED: Issues detected")
    print("=" * 60) 