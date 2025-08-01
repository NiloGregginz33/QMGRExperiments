#!/usr/bin/env python3
"""
Run control experiments to verify CTC paradox detection.

This script runs multiple experiments with different CTC types to:
1. Compare paradox vs standard vs causal CTC types
2. Increase statistical confidence with more shots
3. Verify the paradox detection is working correctly
"""

import sys
import os
import subprocess
from datetime import datetime

def run_control_experiments():
    """Run comprehensive control experiments."""
    
    print("🔬 CTC PARADOX CONTROL EXPERIMENTS")
    print("=" * 50)
    print(f"Started at: {datetime.now()}")
    print()
    
    experiments = [
        {
            "name": "Standard CTC (Control)",
            "ctc_type": "standard",
            "shots": "500",
            "description": "Control experiment with standard CTC type"
        },
        {
            "name": "Causal CTC (Control)", 
            "ctc_type": "causal",
            "shots": "500",
            "description": "Control experiment with causal CTC type"
        },
        {
            "name": "Paradox CTC (High Confidence)",
            "ctc_type": "paradox",
            "shots": "500",
            "description": "Paradox CTC with more shots for better statistics"
        }
    ]
    
    results = []
    
    for i, exp in enumerate(experiments, 1):
        print(f"🧪 Experiment {i}/3: {exp['name']}")
        print(f"   Type: {exp['ctc_type']}")
        print(f"   Shots: {exp['shots']}")
        print(f"   Description: {exp['description']}")
        print()
        
        # Build the command
        cmd = [
            "python", "src/experiments/custom_curvature_experiment.py",
            "--num_qubits", "3",
            "--timesteps", "2",
            "--ctc_mode",  # Enable CTC mode
            "--ctc_type", exp["ctc_type"],  # Use specific CTC type
            "--device", "simulator",  # Use simulator for faster testing
            "--shots", exp["shots"],  # Use more shots
            "--curvature", "1.0",  # Single curvature value
            "--fast",  # Enable fast mode
            "--fast_preset", "minimal"  # Use minimal preset
        ]
        
        print(f"Command: {' '.join(cmd)}")
        print()
        
        try:
            # Run the experiment
            print(f"🚀 Starting {exp['name']}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            exp_result = {
                "name": exp["name"],
                "ctc_type": exp["ctc_type"],
                "shots": exp["shots"],
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
            
            results.append(exp_result)
            
            if result.returncode == 0:
                print(f"✅ {exp['name']} completed successfully!")
                
                # Check for paradox detection in output
                if "paradox detection" in result.stdout.lower():
                    print(f"🔍 Paradox detection found in output")
                if "insufficient entropy data" in result.stdout.lower():
                    print(f"⚠️  Insufficient entropy data detected")
                if "nan" in result.stdout.lower():
                    print(f"⚠️  NaN values detected")
                    
            else:
                print(f"❌ {exp['name']} failed with return code: {result.returncode}")
                if "negative weights" in result.stderr.lower():
                    print(f"🔍 Negative weights error detected!")
                    
        except subprocess.TimeoutExpired:
            print(f"⏰ {exp['name']} timed out after 5 minutes")
            exp_result = {
                "name": exp["name"],
                "ctc_type": exp["ctc_type"],
                "shots": exp["shots"],
                "return_code": -1,
                "stdout": "",
                "stderr": "TIMEOUT",
                "success": False
            }
            results.append(exp_result)
        except Exception as e:
            print(f"❌ Error running {exp['name']}: {e}")
            exp_result = {
                "name": exp["name"],
                "ctc_type": exp["ctc_type"],
                "shots": exp["shots"],
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "success": False
            }
            results.append(exp_result)
        
        print("-" * 50)
        print()
    
    # Analyze results
    print("📊 EXPERIMENT ANALYSIS")
    print("=" * 50)
    
    for result in results:
        print(f"🧪 {result['name']}:")
        print(f"   Success: {'✅' if result['success'] else '❌'}")
        print(f"   Return Code: {result['return_code']}")
        
        # Check for key indicators
        stdout_lower = result['stdout'].lower()
        stderr_lower = result['stderr'].lower()
        
        if "paradox detection" in stdout_lower:
            print(f"   Paradox Detection: ✅ Found")
        if "insufficient entropy data" in stdout_lower:
            print(f"   Insufficient Data: ⚠️  Detected")
        if "nan" in stdout_lower:
            print(f"   NaN Values: ⚠️  Detected")
        if "negative weights" in stderr_lower:
            print(f"   Negative Weights: 🔍 Error detected!")
        if "timeout" in stderr_lower:
            print(f"   Timeout: ⏰ Experiment hung")
            
        print()
    
    # Summary
    print("🎯 SUMMARY:")
    print("=" * 50)
    
    paradox_results = [r for r in results if r['ctc_type'] == 'paradox']
    control_results = [r for r in results if r['ctc_type'] != 'paradox']
    
    print(f"Paradox CTC experiments: {len(paradox_results)}")
    print(f"Control CTC experiments: {len(control_results)}")
    
    paradox_success = [r for r in paradox_results if r['success']]
    control_success = [r for r in control_results if r['success']]
    
    print(f"Paradox success rate: {len(paradox_success)}/{len(paradox_results)}")
    print(f"Control success rate: {len(control_success)}/{len(control_results)}")
    
    # Check for differences
    paradox_errors = [r for r in paradox_results if not r['success']]
    control_errors = [r for r in control_results if not r['success']]
    
    if len(paradox_errors) > len(control_errors):
        print("🔍 Paradox CTC shows more errors than controls - possible paradox evidence!")
    elif len(paradox_errors) == len(control_errors):
        print("⚖️  Paradox CTC shows similar error rate to controls")
    else:
        print("❓ Paradox CTC shows fewer errors than controls - unexpected!")
    
    print()
    print("📁 Results saved to experiment_logs/")
    print("🔍 Check individual experiment folders for detailed analysis")

if __name__ == "__main__":
    run_control_experiments() 