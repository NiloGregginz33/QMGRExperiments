#!/usr/bin/env python3
"""
Hardware Experiment Runner
==========================

This script demonstrates how to run quantum experiments on different hardware options:
- Local Simulator (default)
- IonQ (real quantum hardware)
- Rigetti (real quantum hardware) 
- OQC (real quantum hardware)

Usage:
    python run_hardware_experiments.py --device simulator --experiment emergent_spacetime
    python run_hardware_experiments.py --device ionq --experiment test --shots 1000
    python run_hardware_experiments.py --device all --experiment all
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime

def run_experiment(device, experiment, shots=1024):
    """Run a specific experiment on the specified device"""
    
    print(f"\n{'='*60}")
    print(f"Running {experiment} on {device}")
    print(f"{'='*60}")
    
    # Map experiment names to their script files
    experiment_scripts = {
        'emergent_spacetime': 'src/experiments/emergent_spacetime.py',
        'curved_geometry': 'src/experiments/curved_geometry_analysis.py',
        'test': 'src/experiments/test_experiment.py',
        'holographic': 'src/experiments/run_simple_experiments.py',
        'temporal': 'src/experiments/run_simple_experiments.py',
        'contradictions': 'src/experiments/run_simple_experiments.py'
    }
    
    if experiment not in experiment_scripts:
        print(f"Unknown experiment: {experiment}")
        return False
    
    script = experiment_scripts[experiment]
    
    # Build command
    if experiment in ['holographic', 'temporal', 'contradictions']:
        cmd = f"python {script} --device {device} --experiment {experiment} --shots {shots}"
    else:
        cmd = f"python {script} --device {device} --shots {shots}"
    
    print(f"Command: {cmd}")
    print(f"Starting at: {datetime.now()}")
    
    try:
        # Run the experiment
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        print(f"Completed at: {datetime.now()}")
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {experiment} on {device} completed successfully!")
            return True
        else:
            print(f"❌ {experiment} on {device} failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error running {experiment} on {device}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run quantum experiments on different hardware')
    parser.add_argument('--device', type=str, default='simulator',
                       choices=['simulator', 'ionq', 'rigetti', 'oqc', 'all'],
                       help='Quantum device to use')
    parser.add_argument('--experiment', type=str, default='test',
                       choices=['emergent_spacetime', 'curved_geometry', 'test', 
                               'holographic', 'temporal', 'contradictions', 'all'],
                       help='Which experiment to run')
    parser.add_argument('--shots', type=int, default=1024,
                       help='Number of shots for quantum measurements')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be run without executing')
    
    args = parser.parse_args()
    
    # Define devices and experiments
    devices = ['simulator', 'ionq', 'rigetti', 'oqc'] if args.device == 'all' else [args.device]
    experiments = ['emergent_spacetime', 'curved_geometry', 'test', 
                  'holographic', 'temporal', 'contradictions'] if args.experiment == 'all' else [args.experiment]
    
    print(f"Hardware Experiment Runner")
    print(f"=========================")
    print(f"Devices: {devices}")
    print(f"Experiments: {experiments}")
    print(f"Shots: {args.shots}")
    print(f"Dry run: {args.dry_run}")
    
    if args.dry_run:
        print("\nDRY RUN - Commands that would be executed:")
        for device in devices:
            for experiment in experiments:
                if experiment in ['holographic', 'temporal', 'contradictions']:
                    cmd = f"python src/experiments/run_simple_experiments.py --device {device} --experiment {experiment} --shots {args.shots}"
                else:
                    cmd = f"python src/experiments/{experiment}.py --device {device} --shots {args.shots}"
                print(f"  {cmd}")
        return
    
    # Track results
    results = {}
    total_experiments = len(devices) * len(experiments)
    successful_experiments = 0
    
    print(f"\nStarting {total_experiments} experiment(s)...")
    
    for device in devices:
        results[device] = {}
        for experiment in experiments:
            success = run_experiment(device, experiment, args.shots)
            results[device][experiment] = success
            if success:
                successful_experiments += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {successful_experiments}")
    print(f"Failed: {total_experiments - successful_experiments}")
    print(f"Success rate: {successful_experiments/total_experiments*100:.1f}%")
    
    print(f"\nDetailed Results:")
    for device in devices:
        print(f"\n{device.upper()}:")
        for experiment in experiments:
            status = "✅" if results[device][experiment] else "❌"
            print(f"  {status} {experiment}")
    
    print(f"\nResults saved in experiment_logs/ directories")
    print(f"Check individual experiment folders for detailed outputs")

if __name__ == "__main__":
    main() 