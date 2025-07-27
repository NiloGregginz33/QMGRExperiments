#!/usr/bin/env python3
"""
Analyze Experiment Results

This script provides a convenient way to analyze experiment results from the experiment_logs folder.
It can analyze the last run experiment or any specific experiment instance.

Usage:
    python analyze_experiment.py                                    # Analyze last experiment
    python analyze_experiment.py --experiment <name>               # Analyze last instance of specific experiment
    python analyze_experiment.py --path <full_path>                # Analyze specific instance
    python analyze_experiment.py --list                            # List available experiments
    python analyze_experiment.py --list --experiment <name>        # List instances of specific experiment
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import glob

# Add src to path for imports
sys.path.append('src')

EXPERIMENT_LOGS_DIR = 'experiment_logs'
ANALYSIS_DIR = 'analysis'

def get_available_experiments():
    """Get list of all available experiments in experiment_logs."""
    if not os.path.exists(EXPERIMENT_LOGS_DIR):
        return []
    
    experiments = []
    for item in os.listdir(EXPERIMENT_LOGS_DIR):
        item_path = os.path.join(EXPERIMENT_LOGS_DIR, item)
        if os.path.isdir(item_path):
            # Check if it has instance subdirectories
            instances = [d for d in os.listdir(item_path) if os.path.isdir(os.path.join(item_path, d)) and d.startswith('instance_')]
            if instances:
                experiments.append(item)
    
    return sorted(experiments)

def get_experiment_instances(experiment_name):
    """Get list of all instances for a specific experiment."""
    experiment_path = os.path.join(EXPERIMENT_LOGS_DIR, experiment_name)
    if not os.path.exists(experiment_path):
        return []
    
    instances = []
    for item in os.listdir(experiment_path):
        item_path = os.path.join(experiment_path, item)
        if os.path.isdir(item_path) and item.startswith('instance_'):
            # Get the timestamp from instance name
            try:
                timestamp = item.replace('instance_', '')
                dt = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
                instances.append((item, dt, item_path))
            except ValueError:
                continue
    
    # Sort by timestamp (newest first)
    instances.sort(key=lambda x: x[1], reverse=True)
    return instances

def get_last_experiment():
    """Get the most recent experiment instance across all experiments."""
    all_instances = []
    
    for experiment in get_available_experiments():
        instances = get_experiment_instances(experiment)
        for instance_name, timestamp, instance_path in instances:
            all_instances.append((experiment, instance_name, timestamp, instance_path))
    
    if not all_instances:
        return None, None, None, None
    
    # Sort by timestamp (newest first)
    all_instances.sort(key=lambda x: x[2], reverse=True)
    return all_instances[0]

def get_available_analysis_scripts():
    """Get list of available analysis scripts in the analysis folder."""
    if not os.path.exists(ANALYSIS_DIR):
        return []
    
    scripts = []
    for file in os.listdir(ANALYSIS_DIR):
        if file.endswith('.py') and not file.startswith('__'):
            scripts.append(file)
    
    return sorted(scripts)

def run_analysis_script(script_name, experiment_path, output_dir=None):
    """Run an analysis script on the experiment data."""
    script_path = os.path.join(ANALYSIS_DIR, script_name)
    
    if not os.path.exists(script_path):
        print(f"ERROR: Analysis script '{script_name}' not found in {ANALYSIS_DIR}/")
        return False
    
    # Find the results.json file in the experiment directory
    results_file = os.path.join(experiment_path, 'results.json')
    if not os.path.exists(results_file):
        # Look for any JSON file in the experiment directory
        json_files = glob.glob(os.path.join(experiment_path, '*.json'))
        if json_files:
            results_file = json_files[0]  # Use the first JSON file found
            print(f"Using JSON file: {results_file}")
        else:
            print(f"ERROR: No JSON files found in {experiment_path}")
            return False
    
    # Check if the script accepts --output_dir argument
    try:
        result = subprocess.run([sys.executable, script_path, '--help'], 
                              capture_output=True, text=True, timeout=10)
        accepts_output_dir = '--output-dir' in result.stdout or '--output_dir' in result.stdout
    except:
        accepts_output_dir = False
    
    # Construct the command - most analysis scripts expect JSON file as positional argument
    cmd = [sys.executable, script_path, results_file]
    
    if output_dir and accepts_output_dir:
        # Try both --output-dir and --output_dir formats
        if '--output-dir' in result.stdout:
            cmd.extend(['--output-dir', output_dir])
        else:
            cmd.extend(['--output_dir', output_dir])
    
    print(f"Running analysis: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"[SUCCESS] Analysis '{script_name}' completed successfully!")
            return True
        else:
            print(f"[ERROR] Analysis '{script_name}' failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Analysis '{script_name}' timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"[ERROR] Error running analysis '{script_name}': {e}")
        return False

def analyze_experiment(experiment_path, analysis_scripts=None, output_dir=None):
    """Analyze an experiment instance with specified analysis scripts."""
    print(f"\n{'='*80}")
    print(f"ANALYZING EXPERIMENT: {experiment_path}")
    print(f"{'='*80}")
    
    # Check if experiment path exists
    if not os.path.exists(experiment_path):
        print(f"ERROR: Experiment path does not exist: {experiment_path}")
        return False
    
    # Check if it contains results.json
    results_file = os.path.join(experiment_path, 'results.json')
    if not os.path.exists(results_file):
        print(f"WARNING: No results.json found in {experiment_path}")
        print("This might not be a valid experiment instance.")
    
    # Get available analysis scripts
    available_scripts = get_available_analysis_scripts()
    
    if not available_scripts:
        print(f"ERROR: No analysis scripts found in {ANALYSIS_DIR}/")
        return False
    
    # If no specific scripts provided, show available options
    if not analysis_scripts:
        print(f"\nAvailable analysis scripts:")
        for i, script in enumerate(available_scripts, 1):
            print(f"  {i}. {script}")
        
        print(f"\nTo run specific analysis scripts, use:")
        print(f"  python analyze_experiment.py --path {experiment_path} --analysis <script1> <script2> ...")
        print(f"  python analyze_experiment.py --path {experiment_path} --analysis all")
        
        # Ask user which scripts to run
        try:
            choice = input(f"\nEnter script numbers to run (comma-separated) or 'all': ").strip()
            
            if choice.lower() == 'all':
                analysis_scripts = available_scripts
            else:
                try:
                    indices = [int(x.strip()) - 1 for x in choice.split(',')]
                    analysis_scripts = [available_scripts[i] for i in indices if 0 <= i < len(available_scripts)]
                except (ValueError, IndexError):
                    print("Invalid selection. Running all scripts.")
                    analysis_scripts = available_scripts
        except KeyboardInterrupt:
            print("\nAnalysis cancelled.")
            return False
    
    # Run the selected analysis scripts
    success_count = 0
    total_count = len(analysis_scripts)
    
    for script in analysis_scripts:
        print(f"\n--- Running {script} ---")
        if run_analysis_script(script, experiment_path, output_dir):
            success_count += 1
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully completed: {success_count}/{total_count} analyses")
    print(f"Experiment: {experiment_path}")
    
    if success_count == total_count:
        print("[CHECK] All analyses completed successfully!")
        return True
    else:
        print("[WARNING] Some analyses failed. Check the output above for details.")
        return False

def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results from experiment_logs')
    parser.add_argument('--list', action='store_true', help='List available experiments and instances')
    parser.add_argument('--experiment', type=str, help='Experiment name to analyze (uses most recent instance)')
    parser.add_argument('--path', type=str, help='Full path to specific experiment instance')
    parser.add_argument('--analysis', nargs='+', help='Analysis scripts to run (or "all" for all scripts)')
    parser.add_argument('--output_dir', type=str, help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        if args.experiment:
            # List instances of specific experiment
            instances = get_experiment_instances(args.experiment)
            if instances:
                print(f"\nInstances of experiment '{args.experiment}':")
                print(f"{'Instance':<25} {'Timestamp':<20} {'Path'}")
                print("-" * 80)
                for instance_name, timestamp, instance_path in instances:
                    print(f"{instance_name:<25} {timestamp.strftime('%Y-%m-%d %H:%M:%S'):<20} {instance_path}")
            else:
                print(f"No instances found for experiment '{args.experiment}'")
        else:
            # List all experiments
            experiments = get_available_experiments()
            if experiments:
                print(f"\nAvailable experiments:")
                print(f"{'Experiment':<40} {'Latest Instance':<20} {'Instance Count'}")
                print("-" * 80)
                for experiment in experiments:
                    instances = get_experiment_instances(experiment)
                    if instances:
                        latest_instance, latest_timestamp, _ = instances[0]
                        print(f"{experiment:<40} {latest_timestamp.strftime('%Y-%m-%d %H:%M:%S'):<20} {len(instances)}")
                    else:
                        print(f"{experiment:<40} {'No instances':<20} 0")
            else:
                print("No experiments found in experiment_logs/")
        return
    
    # Determine which experiment to analyze
    experiment_path = None
    
    if args.path:
        # Use specific path
        experiment_path = args.path
    elif args.experiment:
        # Use most recent instance of specific experiment
        instances = get_experiment_instances(args.experiment)
        if instances:
            instance_name, timestamp, experiment_path = instances[0]
            print(f"Using most recent instance of '{args.experiment}': {experiment_path}")
        else:
            print(f"ERROR: No instances found for experiment '{args.experiment}'")
            return
    else:
        # Use most recent experiment overall
        experiment, instance_name, timestamp, experiment_path = get_last_experiment()
        if experiment_path:
            print(f"Using most recent experiment: {experiment} ({instance_name})")
            print(f"Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("ERROR: No experiments found in experiment_logs/")
            return
    
    # Run analysis
    analysis_scripts = args.analysis
    if analysis_scripts and 'all' in analysis_scripts:
        analysis_scripts = get_available_analysis_scripts()
    
    analyze_experiment(experiment_path, analysis_scripts, args.output_dir)

if __name__ == "__main__":
    main() 