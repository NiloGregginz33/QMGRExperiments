import subprocess
import argparse
import sys
import os
import re

EXPERIMENTS_DIR = 'src/experiments'
EXCLUDE = {'__init__.py', 'experiment_logger.py', 'run_experiment.py', 'run_all_experiments.py', 'run_hardware_experiments.py'}
SIMPLE_EXPERIMENTS = ['holographic', 'temporal', 'contradictions']

# Dynamically find experiment scripts
experiment_files = [f for f in os.listdir(EXPERIMENTS_DIR)
                    if f.endswith('.py') and f not in EXCLUDE and not f.startswith('run_')]

# Add run_simple_experiments.py sub-experiments
experiments = []
for f in experiment_files:
    name = re.sub(r'_', ' ', f[:-3]).title()
    experiments.append((name, os.path.join(EXPERIMENTS_DIR, f), None))

# Add sub-experiments from run_simple_experiments.py
for subexp in SIMPLE_EXPERIMENTS:
    name = f"Simple: {subexp.title()}"
    script = os.path.join(EXPERIMENTS_DIR, 'run_simple_experiments.py')
    experiments.append((name, script, subexp))


def run_experiment(name, script, device, shots, subexp=None):
    print(f"\n{'='*60}")
    print(f"Running {name}")
    print(f"{'='*60}")
    if subexp:
        cmd = f"python {script} --device {device} --shots {shots} --experiment {subexp}"
    else:
        cmd = f"python {script} --device {device} --shots {shots}"
    print(f"Command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run a selected quantum experiment')
    parser.add_argument('--experiment', type=int, default=None,
                        help='Experiment number to run (see list)')
    parser.add_argument('--device', type=str, default='simulator',
                        help='Quantum device to use (simulator, ionq, rigetti, oqc, or any IBMQ backend name)')
    parser.add_argument('--shots', type=int, default=1024,
                        help='Number of shots for quantum measurements')
    args = parser.parse_args()

    print("Available experiments:")
    for idx, (name, _, subexp) in enumerate(experiments, 1):
        if subexp:
            print(f"  {idx}. {name} (sub-experiment)")
        else:
            print(f"  {idx}. {name}")

    if args.experiment is None:
        try:
            choice = int(input("Select an experiment by number: "))
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        choice = args.experiment

    if not (1 <= choice <= len(experiments)):
        print("Invalid choice.")
        sys.exit(1)

    name, script, subexp = experiments[choice - 1]
    print(f"\nRunning {name} on device: {args.device} with {args.shots} shots\n")
    success = run_experiment(name, script, args.device, args.shots, subexp)
    print(f"\n{'✅' if success else '❌'} {name} complete. Check experiment_logs/ for results.")

if __name__ == "__main__":
    main() 