#!/usr/bin/env python3
"""
List Generated Files Analysis Script

This script analyzes experiment results and lists all files that were generated
during the experiment execution. It provides a comprehensive overview of the
experiment output including file sizes, timestamps, and file types.

Usage:
    python list_generated_files.py <results.json>                    # Analyze specific experiment
    python list_generated_files.py --experiment <name>              # Analyze last instance of experiment
    python list_generated_files.py --path <experiment_path>         # Analyze specific experiment path
    python list_generated_files.py --list                           # List available experiments
"""

import os
import sys
import json
import argparse
import glob
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append('src')

EXPERIMENT_LOGS_DIR = 'experiment_logs'

def get_file_info(file_path):
    """Get detailed information about a file."""
    try:
        stat = os.stat(file_path)
        size = stat.st_size
        mtime = datetime.fromtimestamp(stat.st_mtime)
        
        # Simple human-readable size conversion
        def humanize_size(size_bytes):
            if size_bytes == 0:
                return "0 B"
            size_names = ["B", "KB", "MB", "GB", "TB"]
            i = 0
            while size_bytes >= 1024 and i < len(size_names) - 1:
                size_bytes /= 1024.0
                i += 1
            return f"{size_bytes:.1f} {size_names[i]}"
        
        return {
            'size': size,
            'size_human': humanize_size(size),
            'modified': mtime,
            'modified_str': mtime.strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        return {
            'size': 0,
            'size_human': '0 B',
            'modified': None,
            'modified_str': 'Unknown',
            'error': str(e)
        }

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
            try:
                # Extract timestamp from instance name
                timestamp_str = item.replace('instance_', '')
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                instances.append((item, timestamp, item_path))
            except ValueError:
                # If timestamp parsing fails, use file modification time
                try:
                    stat = os.stat(item_path)
                    timestamp = datetime.fromtimestamp(stat.st_mtime)
                    instances.append((item, timestamp, item_path))
                except:
                    continue
    
    # Sort by timestamp (newest first)
    instances.sort(key=lambda x: x[1], reverse=True)
    return instances

def get_last_experiment():
    """Get the most recent experiment and instance."""
    experiments = get_available_experiments()
    if not experiments:
        return None, None, None, None
    
    # Get the most recent experiment
    latest_experiment = None
    latest_instance = None
    latest_timestamp = None
    latest_path = None
    
    for experiment in experiments:
        instances = get_experiment_instances(experiment)
        if instances:
            instance_name, timestamp, instance_path = instances[0]
            if latest_timestamp is None or timestamp > latest_timestamp:
                latest_experiment = experiment
                latest_instance = instance_name
                latest_timestamp = timestamp
                latest_path = instance_path
    
    return latest_experiment, latest_instance, latest_timestamp, latest_path

def categorize_files(files):
    """Categorize files by type."""
    categories = {
        'Data Files': [],
        'Images': [],
        'Text Files': [],
        'Logs': [],
        'Other': []
    }
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.pdf'}
    text_extensions = {'.txt', '.md', '.log', '.csv', '.json', '.yaml', '.yml'}
    data_extensions = {'.json', '.csv', '.npy', '.npz', '.h5', '.hdf5'}
    
    for file_path in files:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in image_extensions:
            categories['Images'].append(file_path)
        elif ext in text_extensions:
            if ext in data_extensions:
                categories['Data Files'].append(file_path)
            else:
                categories['Text Files'].append(file_path)
        elif 'log' in os.path.basename(file_path).lower():
            categories['Logs'].append(file_path)
        else:
            categories['Other'].append(file_path)
    
    return categories

def list_generated_files(experiment_path):
    """List all files generated in an experiment directory."""
    print(f"\n{'='*80}")
    print(f"GENERATED FILES ANALYSIS")
    print(f"{'='*80}")
    print(f"Experiment Path: {experiment_path}")
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Check if experiment path exists
    if not os.path.exists(experiment_path):
        print(f"[ERROR] Experiment path does not exist: {experiment_path}")
        return False
    
    # Get all files in the experiment directory
    all_files = []
    total_size = 0
    
    for root, dirs, files in os.walk(experiment_path):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, experiment_path)
            all_files.append(rel_path)
            
            # Get file info
            file_info = get_file_info(file_path)
            total_size += file_info['size']
    
    if not all_files:
        print("[WARNING] No files found in experiment directory")
        return False
    
    # Categorize files
    categories = categorize_files(all_files)
    
    # Print summary
    print(f"\n📊 SUMMARY")
    print(f"Total Files: {len(all_files)}")
    # Calculate human-readable total size
    def humanize_size(size_bytes):
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.1f} {size_names[i]}"
    print(f"Total Size: {humanize_size(total_size)}")
    print(f"Directory: {experiment_path}")
    
    # Print file categories
    print(f"\n📁 FILE CATEGORIES")
    for category, files in categories.items():
        if files:
            category_size = sum(get_file_info(os.path.join(experiment_path, f))['size'] for f in files)
            # Calculate human-readable size for category
            def humanize_size(size_bytes):
                if size_bytes == 0:
                    return "0 B"
                size_names = ["B", "KB", "MB", "GB", "TB"]
                i = 0
                while size_bytes >= 1024 and i < len(size_names) - 1:
                    size_bytes /= 1024.0
                    i += 1
                return f"{size_bytes:.1f} {size_names[i]}"
            size_str = humanize_size(category_size)
            print(f"  {category}: {len(files)} files ({size_str})")
    
    # Print detailed file list by category
    print(f"\n📋 DETAILED FILE LISTING")
    
    for category, files in categories.items():
        if files:
            print(f"\n{category.upper()}:")
            print("-" * 60)
            
            # Sort files by size (largest first)
            files_with_info = []
            for file_path in files:
                full_path = os.path.join(experiment_path, file_path)
                file_info = get_file_info(full_path)
                files_with_info.append((file_path, file_info))
            
            files_with_info.sort(key=lambda x: x[1]['size'], reverse=True)
            
            for file_path, file_info in files_with_info:
                print(f"  {file_path:<50} {file_info['size_human']:<10} {file_info['modified_str']}")
    
    # Print largest files
    print(f"\n🔍 LARGEST FILES")
    print("-" * 60)
    
    files_with_info = []
    for file_path in all_files:
        full_path = os.path.join(experiment_path, file_path)
        file_info = get_file_info(full_path)
        files_with_info.append((file_path, file_info))
    
    files_with_info.sort(key=lambda x: x[1]['size'], reverse=True)
    
    for i, (file_path, file_info) in enumerate(files_with_info[:10], 1):
        print(f"  {i:2d}. {file_path:<45} {file_info['size_human']:<10} {file_info['modified_str']}")
    
    if len(files_with_info) > 10:
        print(f"  ... and {len(files_with_info) - 10} more files")
    
    # Print file types summary
    print(f"\n📊 FILE TYPE BREAKDOWN")
    print("-" * 60)
    
    extensions = {}
    for file_path in all_files:
        ext = os.path.splitext(file_path)[1].lower()
        if ext:
            extensions[ext] = extensions.get(ext, 0) + 1
        else:
            extensions['(no extension)'] = extensions.get('(no extension)', 0) + 1
    
    for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ext:<15} {count:>3} files")
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='List all files generated during an experiment')
    parser.add_argument('results_file', nargs='?', help='Path to results.json file')
    parser.add_argument('--list', action='store_true', help='List available experiments and instances')
    parser.add_argument('--experiment', type=str, help='Experiment name to analyze (uses most recent instance)')
    parser.add_argument('--path', type=str, help='Full path to specific experiment instance')
    
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
    elif args.results_file:
        # Extract experiment path from results file
        if os.path.exists(args.results_file):
            experiment_path = os.path.dirname(os.path.abspath(args.results_file))
        else:
            print(f"ERROR: Results file not found: {args.results_file}")
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
    list_generated_files(experiment_path)

if __name__ == "__main__":
    main() 