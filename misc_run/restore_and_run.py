#!/usr/bin/env python3
"""
Restore file from git and run experiment
"""

import subprocess
import sys

def restore_and_run():
    """Restore the file from git and run the experiment"""
    
    print("=== Restoring file from git ===")
    
    try:
        # Restore the file from git
        result = subprocess.run(['git', 'checkout', 'HEAD', '--', 'src/experiments/custom_curvature_experiment.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ File restored from git")
        else:
            print(f"✗ Git restore failed: {result.stderr}")
            return
        
        # Run the experiment
        print("=== Running experiment ===")
        cmd = [
            sys.executable,
            'src/experiments/custom_curvature_experiment.py',
            '--num_qubits', '12',
            '--geometry', 'spherical',
            '--curvature', '2.0',
            '--charge_injection',
            '--charge_strength', '1.5',
            '--charge_location', 'center',
            '--compute_entropies',
            '--entropy_method', 'von_neumann',
            '--page_curve',
            '--page_curve_timesteps', '4',
            '--timesteps', '4',
            '--shots', '8192',
            '--device', 'ibm_brisbane',
            '--solve_regge',
            '--analyze_curvature'
        ]
        
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode == 0:
            print("✓ Experiment completed successfully")
        else:
            print(f"✗ Experiment failed with return code {result.returncode}")
            
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    restore_and_run() 