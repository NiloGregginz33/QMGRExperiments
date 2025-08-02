#!/usr/bin/env python3
"""
Run a genuine CTC paradox experiment using the custom curvature experiment.

This script enables CTC mode and creates a genuine quantum grandfather paradox.
"""

import sys
import os
import subprocess
from datetime import datetime

def run_genuine_paradox():
    """Run the genuine paradox experiment."""
    
    print("🌌 GENUINE CTC PARADOX EXPERIMENT")
    print("=" * 50)
    print(f"Started at: {datetime.now()}")
    print()
    
    # Build the command with CTC mode enabled
    cmd = [
        "python", "src/experiments/custom_curvature_experiment.py",
        "--num_qubits", "3",
        "--timesteps", "3",
        "--ctc_mode",  # Enable CTC mode
        "--ctc_type", "paradox",  # Use paradox CTC type
        "--device", "simulator",  # Use simulator for safety
        "--shots", "100",  # Reduced shots for faster execution
        "--curvature", "1.0",  # Single curvature value
        "--fast",  # Enable fast mode
        "--fast_preset", "minimal"  # Use minimal preset
    ]
    
    print("Command:")
    print(" ".join(cmd))
    print()
    print("⚠️  WARNING: This experiment may create a genuine quantum grandfather paradox!")
    print("⚠️  The system may hang if a paradox is detected.")
    print("⚠️  Use Ctrl+C to stop if it hangs indefinitely.")
    print()
    
    try:
        # Run the experiment
        print("🚀 Starting experiment...")
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("✅ Experiment completed successfully!")
        else:
            print(f"❌ Experiment failed with return code: {result.returncode}")
            
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        print("This may indicate a genuine paradox was detected!")
    except Exception as e:
        print(f"❌ Error running experiment: {e}")

if __name__ == "__main__":
    run_genuine_paradox() 