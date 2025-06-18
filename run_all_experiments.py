import subprocess
import argparse
import sys

EXPERIMENTS = [
    ("Emergent Spacetime", "src/experiments/emergent_spacetime.py"),
    ("Curved Geometry", "src/experiments/curved_geometry_analysis.py"),
    ("Test Experiment", "src/experiments/test_experiment.py"),
    ("Holographic Demo", "src/experiments/run_simple_experiments.py --experiment holographic"),
    ("Temporal Injection", "src/experiments/run_simple_experiments.py --experiment temporal"),
    ("Contradictions Test", "src/experiments/run_simple_experiments.py --experiment contradictions"),
]

def run_experiment(name, script, device, shots):
    print(f"\n{'='*60}")
    print(f"Running {name}")
    print(f"{'='*60}")
    if 'run_simple_experiments.py' in script:
        cmd = f"python {script} --device {device} --shots {shots}"
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
    parser = argparse.ArgumentParser(description='Run all quantum experiments')
    parser.add_argument('--device', type=str, default='simulator',
                       choices=['simulator', 'ionq', 'rigetti', 'oqc'],
                       help='Quantum device to use')
    parser.add_argument('--shots', type=int, default=1024,
                       help='Number of shots for quantum measurements')
    args = parser.parse_args()

    print(f"Running all experiments on device: {args.device} with {args.shots} shots\n")
    results = {}
    for name, script in EXPERIMENTS:
        success = run_experiment(name, script, args.device, args.shots)
        results[name] = success
    print("\nSummary:")
    for name, success in results.items():
        print(f"  {'✅' if success else '❌'} {name}")
    print("\nAll experiments complete. Check experiment_logs/ for results.")

if __name__ == "__main__":
    main() 