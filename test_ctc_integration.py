#!/usr/bin/env python3
"""
Simple test script to verify CTC integration in custom_curvature_experiment.py
"""

import sys
import os
import subprocess

def test_ctc_integration():
    """Test if the CTC arguments are properly integrated"""
    
    print("=== Testing CTC Integration ===")
    
    # Test 1: Check if the script can be imported
    print("\n1. Testing script import...")
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from experiments.custom_curvature_experiment import _apply_enhanced_ctc_modifier
        print("✓ CTC modifier function imported successfully")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Test 2: Check if help shows CTC arguments
    print("\n2. Testing help output for CTC arguments...")
    try:
        result = subprocess.run([
            sys.executable, 
            "src/experiments/custom_curvature_experiment.py", 
            "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            help_text = result.stdout
            ctc_args = [
                "--ctc_mode",
                "--ctc_size", 
                "--ctc_phase_profile",
                "--ctc_phase_strength",
                "--ctc_custom_phases",
                "--ctc_paradox_strength",
                "--ctc_wormhole_mode",
                "--ctc_fixed_point_iterations",
                "--ctc_tolerance",
                "--ctc_analysis"
            ]
            
            found_args = []
            for arg in ctc_args:
                if arg in help_text:
                    found_args.append(arg)
                    print(f"✓ Found CTC argument: {arg}")
                else:
                    print(f"✗ Missing CTC argument: {arg}")
            
            if len(found_args) >= 8:  # Most CTC arguments should be present
                print(f"✓ CTC arguments properly integrated ({len(found_args)}/10 found)")
            else:
                print(f"✗ Only {len(found_args)}/10 CTC arguments found")
                return False
        else:
            print(f"✗ Help command failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Help command timed out")
        return False
    except Exception as e:
        print(f"✗ Help command failed: {e}")
        return False
    
    # Test 3: Test minimal CTC run
    print("\n3. Testing minimal CTC run...")
    try:
        result = subprocess.run([
            sys.executable,
            "src/experiments/custom_curvature_experiment.py",
            "--num_qubits", "3",
            "--ctc_mode",
            "--ctc_size", "2",
            "--ctc_phase_profile", "uniform",
            "--ctc_phase_strength", "0.1",
            "--shots", "10",
            "--fast"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            output = result.stdout
            if "CTC" in output or "ctc" in output.lower():
                print("✓ CTC functionality detected in output")
                print("✓ Minimal CTC run successful")
            else:
                print("✗ No CTC output detected")
                return False
        else:
            print(f"✗ CTC run failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ CTC run timed out")
        return False
    except Exception as e:
        print(f"✗ CTC run failed: {e}")
        return False
    
    print("\n=== All CTC Integration Tests Passed! ===")
    return True

if __name__ == "__main__":
    success = test_ctc_integration()
    sys.exit(0 if success else 1) 