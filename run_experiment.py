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

# First, add the custom curvature experiment if it exists
custom_curvature_file = 'custom_curvature_experiment.py'
if custom_curvature_file in experiment_files:
    name = re.sub(r'_', ' ', custom_curvature_file[:-3]).title()
    experiments.append((name, os.path.join(EXPERIMENTS_DIR, custom_curvature_file), None))
    # Remove it from the list so it doesn't get added again
    experiment_files.remove(custom_curvature_file)

# Add all other experiments in their original order
for f in experiment_files:
    name = re.sub(r'_', ' ', f[:-3]).title()
    experiments.append((name, os.path.join(EXPERIMENTS_DIR, f), None))

# Add sub-experiments from run_simple_experiments.py
for subexp in SIMPLE_EXPERIMENTS:
    name = f"Simple: {subexp.title()}"
    script = os.path.join(EXPERIMENTS_DIR, 'run_simple_experiments.py')
    experiments.append((name, script, subexp))


def show_experiment_help(script_path, subexp=None):
    """
    Show the help output for a selected experiment.
    
    Args:
        script_path: Path to the experiment script
        subexp: Sub-experiment name (for run_simple_experiments.py)
    """
    print(f"\n{'='*80}")
    print(f"HELP FOR EXPERIMENT: {os.path.basename(script_path)}")
    if subexp:
        print(f"SUB-EXPERIMENT: {subexp}")
    print(f"{'='*80}")
    
    # Construct the help command
    if subexp:
        cmd = f"python {script_path} --help"
    else:
        cmd = f"python {script_path} --help"
    
    try:
        # Set environment variables to handle Unicode properly
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'
        
        # Run the help command and capture output
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, 
                              timeout=30, env=env, encoding='utf-8', errors='replace')
        
        if result.returncode == 0:
            # Clean up the output to handle any remaining encoding issues
            help_output = result.stdout
            # Replace problematic Unicode characters with ASCII equivalents
            help_output = help_output.replace('∝', 'proportional to')
            help_output = help_output.replace('≈', 'approximately')
            help_output = help_output.replace('≠', 'not equal to')
            help_output = help_output.replace('≤', 'less than or equal to')
            help_output = help_output.replace('≥', 'greater than or equal to')
            help_output = help_output.replace('±', '+/-')
            help_output = help_output.replace('×', 'x')
            help_output = help_output.replace('÷', '/')
            help_output = help_output.replace('∞', 'infinity')
            help_output = help_output.replace('∑', 'sum')
            help_output = help_output.replace('∫', 'integral')
            help_output = help_output.replace('∂', 'partial')
            help_output = help_output.replace('∇', 'nabla')
            help_output = help_output.replace('Δ', 'delta')
            help_output = help_output.replace('θ', 'theta')
            help_output = help_output.replace('φ', 'phi')
            help_output = help_output.replace('ψ', 'psi')
            help_output = help_output.replace('α', 'alpha')
            help_output = help_output.replace('β', 'beta')
            help_output = help_output.replace('γ', 'gamma')
            help_output = help_output.replace('δ', 'delta')
            help_output = help_output.replace('ε', 'epsilon')
            help_output = help_output.replace('ζ', 'zeta')
            help_output = help_output.replace('η', 'eta')
            help_output = help_output.replace('ι', 'iota')
            help_output = help_output.replace('κ', 'kappa')
            help_output = help_output.replace('λ', 'lambda')
            help_output = help_output.replace('μ', 'mu')
            help_output = help_output.replace('ν', 'nu')
            help_output = help_output.replace('ξ', 'xi')
            help_output = help_output.replace('ο', 'omicron')
            help_output = help_output.replace('π', 'pi')
            help_output = help_output.replace('ρ', 'rho')
            help_output = help_output.replace('σ', 'sigma')
            help_output = help_output.replace('τ', 'tau')
            help_output = help_output.replace('υ', 'upsilon')
            help_output = help_output.replace('χ', 'chi')
            help_output = help_output.replace('ω', 'omega')
            
            print(help_output)
        else:
            print("Error getting help information:")
            print(result.stderr)
            print("\nThis experiment may not have proper argument parsing.")
            
    except subprocess.TimeoutExpired:
        print("Help command timed out. The experiment may be taking too long to start.")
    except Exception as e:
        print(f"Error running help command: {e}")
        print("This experiment may not have proper argument parsing.")
        print("You can still proceed with default parameters.")


def get_user_confirmation(prompt="Press Enter to continue or 'q' to quit: "):
    """
    Get user confirmation to continue or quit.
    
    Args:
        prompt: The prompt to show to the user
        
    Returns:
        bool: True if user wants to continue, False if they want to quit
    """
    while True:
        response = input(prompt).strip().lower()
        if response in ['', 'y', 'yes', 'continue']:
            return True
        elif response in ['q', 'quit', 'exit', 'no']:
            return False
        else:
            print("Please enter 'y' to continue or 'q' to quit.")


def run_experiment(name, script, device, shots, subexp=None, additional_cmd_args=None, auto_analyze=True, analysis_scripts=None):
    print(f"\n{'='*60}")
    print(f"Running {name}")
    print(f"{'='*60}")
    
    # Show help for the experiment first
    show_experiment_help(script, subexp)
    
    # If command line arguments were provided, skip interactive prompts
    if additional_cmd_args:
        print(f"\nUsing command line arguments: {' '.join(additional_cmd_args)}")
        print("Skipping interactive prompts due to command line arguments provided.")
        
        # Construct the base command
        if subexp:
            cmd = f"python {script} --device {device} --shots {shots} --experiment {subexp}"
        else:
            cmd = f"python {script} --device {device} --shots {shots}"

        # Add additional arguments directly to the command
        if additional_cmd_args:
            cmd += " " + " ".join(additional_cmd_args)

        print(f"\nExecuting command: {cmd}")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        success = result.returncode == 0
        
        # Auto-analyze if experiment was successful and auto_analyze is enabled
        if success and auto_analyze:
            print(f"\n{'='*60}")
            print(f"Experiment completed successfully! Running automatic analysis...")
            print(f"{'='*60}")
            
            # Determine the experiment name for analysis
            experiment_name = None
            if subexp:
                # For sub-experiments, use the sub-experiment name
                experiment_name = subexp
            else:
                # For main experiments, extract name from script path
                script_basename = os.path.basename(script)
                experiment_name = script_basename.replace('.py', '')
            
            # If no specific analysis scripts provided, run the list generated files script
            if not analysis_scripts:
                print("No specific analysis scripts provided. Running file listing analysis...")
                
                # Run the list generated files script
                list_cmd = ['python', 'tools/list_generated_files.py', '--experiment', experiment_name]
                print(f"Running file listing command: {' '.join(list_cmd)}")
                
                try:
                    list_result = subprocess.run(list_cmd, capture_output=True, text=True, timeout=300)
                    
                    if list_result.stdout:
                        print("FILE LISTING OUTPUT:")
                        print(list_result.stdout)
                    
                    if list_result.stderr:
                        print("FILE LISTING ERRORS:")
                        print(list_result.stderr)
                    
                    if list_result.returncode == 0:
                        print("✅ File listing analysis completed successfully!")
                    else:
                        print(f"⚠️ File listing analysis completed with warnings (return code: {list_result.returncode})")
                        
                except subprocess.TimeoutExpired:
                    print("⚠️ File listing analysis timed out after 5 minutes")
                except Exception as e:
                    print(f"⚠️ Error during file listing analysis: {e}")
            else:
                # Run specific analysis scripts
                analysis_cmd = ['python', 'tools/analyze_experiment.py', '--experiment', experiment_name]
                
                # Add specific analysis scripts if provided
                if analysis_scripts:
                    analysis_cmd.extend(['--analysis'] + analysis_scripts)
                
                print(f"Running analysis command: {' '.join(analysis_cmd)}")
                
                try:
                    analysis_result = subprocess.run(analysis_cmd, capture_output=True, text=True, timeout=600)
                    
                    if analysis_result.stdout:
                        print("ANALYSIS OUTPUT:")
                        print(analysis_result.stdout)
                    
                    if analysis_result.stderr:
                        print("ANALYSIS ERRORS:")
                        print(analysis_result.stderr)
                    
                    if analysis_result.returncode == 0:
                        print("✅ Automatic analysis completed successfully!")
                    else:
                        print(f"⚠️ Automatic analysis completed with warnings (return code: {analysis_result.returncode})")
                        
                except subprocess.TimeoutExpired:
                    print("⚠️ Automatic analysis timed out after 10 minutes")
                except Exception as e:
                    print(f"⚠️ Error during automatic analysis: {e}")
        
        return success
    
    # Interactive mode - ask user if they want to continue after seeing help
    if not get_user_confirmation("\nDo you want to continue with this experiment? (y/q): "):
        print("Experiment cancelled by user.")
        return False
    
    # Prompt for additional arguments based on the selected experiment
    additional_args = {}
    
    # Enhanced argument prompting based on experiment type
    if 'teleportation' in name.lower():
        try:
            additional_args['num_qubits'] = int(input("Enter the number of qubits: "))
        except ValueError:
            print("Invalid input for number of qubits. Using default value of 5.")
            additional_args['num_qubits'] = 5
    
    # Add common arguments that users might want to modify
    print("\nCommon experiment parameters:")
    print("(Press Enter to use defaults, or enter custom values)")
    
    # Ask for num_qubits if not already set
    if 'num_qubits' not in additional_args:
        try:
            num_qubits_input = input("Number of qubits (default: varies by experiment): ").strip()
            if num_qubits_input:
                additional_args['num_qubits'] = int(num_qubits_input)
        except ValueError:
            print("Invalid input. Using experiment default.")
    
    # Ask for shots if different from default
    try:
        shots_input = input(f"Number of shots (default: {shots}): ").strip()
        if shots_input:
            shots = int(shots_input)
    except ValueError:
        print("Invalid input. Using default shots value.")
    
    # Ask for device if different from default
    device_input = input(f"Device (default: {device}): ").strip()
    if device_input:
        device = device_input
    
    # Ask for geometry if it's a geometry experiment
    if any(geo in name.lower() for geo in ['geometry', 'curvature', 'spherical', 'hyperbolic']):
        try:
            geometry_input = input("Geometry type (euclidean/spherical/hyperbolic, default: varies): ").strip()
            if geometry_input:
                additional_args['geometry'] = geometry_input
        except ValueError:
            print("Invalid input. Using experiment default.")
    
    # Ask for curvature if it's a curvature experiment
    if 'curvature' in name.lower():
        try:
            curvature_input = input("Curvature value(s) (default: varies): ").strip()
            if curvature_input:
                # Handle multiple curvature values
                if ',' in curvature_input:
                    additional_args['curvature'] = [float(x.strip()) for x in curvature_input.split(',')]
                else:
                    additional_args['curvature'] = float(curvature_input)
        except ValueError:
            print("Invalid input. Using experiment default.")

    # Construct the command with additional arguments
    if subexp:
        cmd = f"python {script} --device {device} --shots {shots} --experiment {subexp}"
    else:
        cmd = f"python {script} --device {device} --shots {shots}"

    # Add additional arguments to the command
    for arg, value in additional_args.items():
        if isinstance(value, list):
            # Handle list arguments (like curvature sweep)
            cmd += f" --{arg} {' '.join(map(str, value))}"
        elif isinstance(value, bool):
            # Handle boolean flags
            if value:
                cmd += f" --{arg}"
        else:
            cmd += f" --{arg} {value}"

    print(f"\nFinal command: {cmd}")
    
    # Ask about automatic analysis
    auto_analyze = True
    analysis_scripts = None
    
    if not additional_cmd_args:  # Only ask in interactive mode
        auto_analyze_input = input("\nRun automatic analysis after completion? (y/n, default: y): ").strip().lower()
        auto_analyze = auto_analyze_input in ['', 'y', 'yes']
        
        if auto_analyze:
            analysis_choice = input("Run all analysis scripts or specific ones? (all/specific, default: all): ").strip().lower()
            if analysis_choice == 'specific':
                print("Available analysis scripts:")
                # Get available analysis scripts
                analysis_dir = 'analysis'
                if os.path.exists(analysis_dir):
                    available_scripts = [f for f in os.listdir(analysis_dir) if f.endswith('.py') and not f.startswith('__')]
                    for i, script in enumerate(available_scripts, 1):
                        print(f"  {i}. {script}")
                    
                    try:
                        script_choice = input("Enter script numbers to run (comma-separated): ").strip()
                        indices = [int(x.strip()) - 1 for x in script_choice.split(',')]
                        analysis_scripts = [available_scripts[i] for i in indices if 0 <= i < len(available_scripts)]
                    except (ValueError, IndexError):
                        print("Invalid selection. Running all scripts.")
    
    # Final confirmation before running
    if not get_user_confirmation("\nReady to run the experiment? (y/q): "):
        print("Experiment cancelled by user.")
        return False
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    success = result.returncode == 0
    
    # Auto-analyze if experiment was successful and auto_analyze is enabled
    if success and auto_analyze:
        print(f"\n{'='*60}")
        print(f"Experiment completed successfully! Running automatic analysis...")
        print(f"{'='*60}")
        
        # Determine the experiment name for analysis
        experiment_name = None
        if subexp:
            # For sub-experiments, use the sub-experiment name
            experiment_name = subexp
        else:
            # For main experiments, extract name from script path
            script_basename = os.path.basename(script)
            experiment_name = script_basename.replace('.py', '')
        
        # If no specific analysis scripts provided, run the list generated files script
        if not analysis_scripts:
            print("No specific analysis scripts provided. Running file listing analysis...")
            
            # Run the list generated files script
            list_cmd = ['python', 'tools/list_generated_files.py', '--experiment', experiment_name]
            print(f"Running file listing command: {' '.join(list_cmd)}")
            
            try:
                list_result = subprocess.run(list_cmd, capture_output=True, text=True, timeout=300)
                
                if list_result.stdout:
                    print("FILE LISTING OUTPUT:")
                    print(list_result.stdout)
                
                if list_result.stderr:
                    print("FILE LISTING ERRORS:")
                    print(list_result.stderr)
                
                if list_result.returncode == 0:
                    print("✅ File listing analysis completed successfully!")
                else:
                    print(f"⚠️ File listing analysis completed with warnings (return code: {list_result.returncode})")
                    
            except subprocess.TimeoutExpired:
                print("⚠️ File listing analysis timed out after 5 minutes")
            except Exception as e:
                print(f"⚠️ Error during file listing analysis: {e}")
        else:
            # Run specific analysis scripts
            analysis_cmd = ['python', 'tools/analyze_experiment.py', '--experiment', experiment_name]
            
            # Add specific analysis scripts if provided
            if analysis_scripts:
                analysis_cmd.extend(['--analysis'] + analysis_scripts)
            
            print(f"Running analysis command: {' '.join(analysis_cmd)}")
            
            try:
                analysis_result = subprocess.run(analysis_cmd, capture_output=True, text=True, timeout=600)
                
                if analysis_result.stdout:
                    print("ANALYSIS OUTPUT:")
                    print(analysis_result.stdout)
                
                if analysis_result.stderr:
                    print("ANALYSIS ERRORS:")
                    print(analysis_result.stderr)
                
                if analysis_result.returncode == 0:
                    print("✅ Automatic analysis completed successfully!")
                else:
                    print(f"⚠️ Automatic analysis completed with warnings (return code: {analysis_result.returncode})")
                    
            except subprocess.TimeoutExpired:
                print("⚠️ Automatic analysis timed out after 10 minutes")
            except Exception as e:
                print(f"⚠️ Error during automatic analysis: {e}")
    
    return success


def get_available_analysis_scripts():
    """Get list of available analysis scripts."""
    analysis_dir = 'analysis'
    tools_dir = 'tools'
    scripts = []
    
    # Check analysis directory
    if os.path.exists(analysis_dir):
        for file in os.listdir(analysis_dir):
            if file.endswith('.py') and not file.startswith('__'):
                scripts.append(os.path.join(analysis_dir, file))
    
    # Check tools directory for analysis scripts
    if os.path.exists(tools_dir):
        for file in os.listdir(tools_dir):
            if file.endswith('.py') and ('analyze' in file.lower() or 'list' in file.lower()):
                scripts.append(os.path.join(tools_dir, file))
    
    return sorted(scripts)

def main():
    parser = argparse.ArgumentParser(description='Run a selected quantum experiment')
    parser.add_argument('--experiment', type=int, default=None,
                        help='Experiment number to run (see list)')
    parser.add_argument('--device', type=str, default='simulator',
                        help='Quantum device to use (simulator, ionq, rigetti, oqc, or any IBMQ backend name)')
    parser.add_argument('--shots', type=int, default=1024,
                        help='Number of shots for quantum measurements')
    parser.add_argument('--args', nargs='*', default=[],
                        help='Additional arguments to pass to the experiment script (e.g., --args --num_qubits=5 --geometry=spherical)')
    parser.add_argument('--no-auto-analyze', action='store_true',
                        help='Disable automatic analysis after successful experiment completion')
    parser.add_argument('--analysis-scripts', nargs='*',
                        help='Specific analysis scripts to run automatically (e.g., --analysis-scripts analyze_curvature_results.py analyze_emergent_metric.py)')
    args, unknown = parser.parse_known_args()

    print("Available experiments:")
    for idx, (name, _, subexp) in enumerate(experiments, 1):
        if subexp:
            print(f"  {idx}. {name} (sub-experiment)")
        else:
            print(f"  {idx}. {name}")
    
    # Show available analysis scripts
    analysis_scripts = get_available_analysis_scripts()
    if analysis_scripts:
        print(f"\nAvailable analysis scripts:")
        for i, script in enumerate(analysis_scripts, 1):
            script_name = os.path.basename(script)
            print(f"  {i}. {script_name}")
        print(f"  {len(analysis_scripts) + 1}. list_generated_files.py (auto-default when no analysis specified)")

    if args.experiment is None:
        while True:
            try:
                user_input = input("\nSelect an experiment by number (or 'q' to quit): ").strip().lower()
                
                # Check for quit commands
                if user_input in ['q', 'quit', 'exit', '']:
                    print("Exiting...")
                    sys.exit(0)
                
                # Try to convert to integer
                choice = int(user_input)
                break
            except ValueError:
                print("Invalid input. Please enter a number or 'q' to quit.")
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)
            except Exception as e:
                print(f"Error: {e}")
                sys.exit(1)
    else:
        choice = args.experiment

    if not (1 <= choice <= len(experiments)):
        print("Invalid choice.")
        sys.exit(1)

    name, script, subexp = experiments[choice - 1]
    print(f"\nSelected: {name}")
    
    # Combine args.args with any unknown arguments
    additional_args = args.args + unknown
    
    # Determine auto-analyze settings
    auto_analyze = not args.no_auto_analyze
    analysis_scripts = args.analysis_scripts
    
    success = run_experiment(name, script, args.device, args.shots, subexp, additional_args, auto_analyze, analysis_scripts)
    print(f"\n{'✅' if success else '❌'} {name} complete. Check experiment_logs/ for results.")

if __name__ == "__main__":
    main() 