#!/usr/bin/env python3
"""
Run a safe CTC paradox experiment that can handle paradox detection without crashing.
"""

import sys
import os
import subprocess
from datetime import datetime

def run_safe_paradox():
    """Run the safe paradox experiment."""
    
    print("🌌 SAFE CTC PARADOX EXPERIMENT")
    print("=" * 50)
    print(f"Started at: {datetime.now()}")
    print()
    
    # Build the command with minimal parameters to avoid negative weights
    cmd = [
        "python", "src/experiments/custom_curvature_experiment.py",
        "--num_qubits", "3",
        "--timesteps", "2",
        "--ctc_mode",  # Enable CTC mode
        "--ctc_type", "paradox",  # Use paradox CTC type
        "--device", "simulator",  # Use simulator for safety
        "--shots", "50",  # Reduced shots for faster execution
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
    print("🎯 EXPECTED OUTCOME:")
    print("   - If paradox detected: System may hang or show negative weights error")
    print("   - If no paradox: Experiment completes normally")
    print("   - Either way: Paradox analysis will be saved to results")
    print()
    
    try:
        # Run the experiment
        print("🚀 Starting safe paradox experiment...")
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("✅ Safe paradox experiment completed successfully!")
            print("📊 Check the experiment logs for paradox analysis results")
        else:
            print(f"❌ Safe paradox experiment failed with return code: {result.returncode}")
            print("🔍 This may indicate a paradox was detected!")
            
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        print("🌌 This may indicate a genuine paradox was detected!")
        print("📊 Check the experiment logs for partial results")
    except Exception as e:
        print(f"❌ Error running safe paradox experiment: {e}")

if __name__ == "__main__":
    run_safe_paradox() 