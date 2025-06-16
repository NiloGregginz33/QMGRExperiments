import os
import shutil

# Define the organization structure
organization = {
    'src/quantum': [
        'Qex*.py',  # Quantum experiments
        'hamiltonian_calc.py',
        'numpypatch.py',
    ],
    'src/analysis': [
        'curv_Test.py',
        'entropy_sweep_log.csv',
        'entropy_oracle_log.csv',
    ],
    'src/utils': [
        'experiment_logger.py',
        'setenv.py',
        'qi_auth.py',
        'ibmtool.py',
        'save_account.py',
        'save_credentials.py',
    ],
    'src/visualization': [
        'show_graph.py',
        'colortest.py',
    ],
    'src/factories': [
        '*Factory.py',  # All factory files
    ],
    'src/experiments': [
        'run_experiments.py',
        'run_all_awsfactory.py',
        'ex*.py',  # Experiment files
    ]
}

def ensure_dir(directory):
    """Ensure directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def move_files():
    """Move files to their appropriate directories."""
    for directory, patterns in organization.items():
        ensure_dir(directory)
        
        for pattern in patterns:
            # Handle wildcard patterns
            if '*' in pattern:
                import glob
                files = glob.glob(pattern)
            else:
                files = [pattern]
            
            for file in files:
                if os.path.exists(file):
                    try:
                        shutil.move(file, os.path.join(directory, os.path.basename(file)))
                        print(f"Moved {file} to {directory}")
                    except Exception as e:
                        print(f"Error moving {file}: {e}")

if __name__ == "__main__":
    move_files() 