#!/usr/bin/env python3
"""
Simple test script to verify CTC integration without importing the main file
"""

import sys
import os
import subprocess

def test_ctc_help():
    """Test if CTC arguments appear in help"""
    print("=== Testing CTC Help Arguments ===")
    
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
                "--ctc_phase_strength"
            ]
            
            found_args = []
            for arg in ctc_args:
                if arg in help_text:
                    found_args.append(arg)
                    print(f"✓ Found CTC argument: {arg}")
                else:
                    print(f"✗ Missing CTC argument: {arg}")
            
            if len(found_args) >= 3:
                print(f"✓ CTC arguments found: {len(found_args)}/4")
                return True
            else:
                print(f"✗ Only {len(found_args)}/4 CTC arguments found")
                return False
        else:
            print(f"✗ Help command failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Help command failed: {e}")
        return False

def test_minimal_ctc_run():
    """Test a minimal CTC run"""
    print("\n=== Testing Minimal CTC Run ===")
    
    try:
        result = subprocess.run([
            sys.executable,
            "src/experiments/custom_curvature_experiment.py",
            "--num_qubits", "3",
            "--ctc_mode",
            "--ctc_size", "2",
            "--shots", "10",
            "--fast"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            output = result.stdout
            if "CTC" in output or "ctc" in output.lower():
                print("✓ CTC functionality detected in output")
                return True
            else:
                print("✗ No CTC output detected")
                print("Output preview:", output[:200])
                return False
        else:
            print(f"✗ CTC run failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ CTC run failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing CTC Integration...")
    
    help_ok = test_ctc_help()
    if help_ok:
        run_ok = test_minimal_ctc_run()
        if run_ok:
            print("\n=== CTC Integration Test PASSED ===")
            sys.exit(0)
        else:
            print("\n=== CTC Run Test FAILED ===")
            sys.exit(1)
    else:
        print("\n=== CTC Help Test FAILED ===")
        sys.exit(1) 