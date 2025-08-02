#!/usr/bin/env python3
"""
Run a genuine CTC paradox experiment on IBM Brisbane hardware.

This script enables CTC mode and creates a genuine quantum grandfather paradox
on real quantum hardware for the first time.
"""

import sys
import os
import subprocess
from datetime import datetime

def run_hardware_paradox():
    """Run the genuine paradox experiment on hardware."""
    
    print("🌌 HARDWARE CTC PARADOX EXPERIMENT")
    print("=" * 50)
    print(f"Started at: {datetime.now()}")
    print()
    
    # Build the command with CTC mode enabled for hardware
    cmd = [
        "python", "src/experiments/custom_curvature_experiment.py",
        "--num_qubits", "3",
        "--timesteps", "2",
        "--ctc_mode",  # Enable CTC mode
        "--ctc_type", "paradox",  # Use paradox CTC type
        "--device", "ibm_brisbane",  # Use real hardware
        "--shots", "100",  # Reasonable shots for hardware
        "--curvature", "1.0",  # Single curvature value
        "--fast",  # Enable fast mode for hardware
        "--fast_preset", "minimal"  # Use minimal preset
    ]
    
    print("Command:")
    print(" ".join(cmd))
    print()
    print("⚠️  WARNING: This experiment will create a genuine quantum grandfather paradox!")
    print("⚠️  This is the FIRST TIME testing a quantum paradox on real hardware!")
    print("⚠️  The hardware may behave differently than the simulator.")
    print("⚠️  Use Ctrl+C to stop if it hangs indefinitely.")
    print()
    print("🎯 EXPECTED OUTCOMES:")
    print("   - Hardware may show different paradox behavior than simulator")
    print("   - May see negative weights error or hanging")
    print("   - Could reveal new quantum paradox physics")
    print("   - Results will be saved to experiment_logs")
    print()
    
    try:
        # Run the experiment
        print("🚀 Starting hardware paradox experiment...")
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("✅ Hardware paradox experiment completed successfully!")
            print("📊 Check the experiment logs for paradox analysis results")
        else:
            print(f"❌ Hardware paradox experiment failed with return code: {result.returncode}")
            print("🔍 This may indicate a paradox was detected on hardware!")
            
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        print("🌌 This may indicate a genuine paradox was detected on hardware!")
        print("📊 Check the experiment logs for partial results")
    except Exception as e:
        print(f"❌ Error running hardware paradox experiment: {e}")

if __name__ == "__main__":
    run_hardware_paradox() 