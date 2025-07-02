from qiskit_ibm_runtime import QiskitRuntimeService
import os
import platform
import subprocess
import getpass
from pathlib import Path


QiskitRuntimeService.save_account(token=os.environ["IBM_QUANTUM_API_KEY"], overwrite=True)
def set_windows_env(var: str, val: str):
    # setx sets permanently for the current user
    subprocess.check_call(['setx', var, val], shell=True)

def set_unix_env(var: str, val: str):
    # detect your shell rc file
    shell = os.environ.get('SHELL', '')
    if 'zsh' in shell:
        rc = Path.home() / '.zshrc'
    else:
        rc = Path.home() / '.bashrc'
    line = f'\n# added by set_qiskit_env.py\nexport {var}="{val}"\n'
    with open(rc, 'a') as f:
        f.write(line)
    print(f"  ↳ appended to {rc}")

def main():
    print("\n💫 IBM Quantum Environment Setter\n")

    token = getpass.getpass("Enter your IBM Quantum API token: ").strip()
    if not token:
        print("✖ no token entered, aborting.")
        return

    default_url = "https://auth.quantum-computing.ibm.com/api"
    url = input(f"Enter IBM Quantum API URL [{default_url}]: ").strip() or default_url

    os_name = platform.system()
    print(f"\nDetected OS: {os_name}")

    # set both variables
    for var, val in (
        ("QISKIT_IBM_PROVIDER_TOKEN", token),
        ("QISKIT_IBM_PROVIDER_URL", url),
    ):
        print(f"Setting {var} → {val!r} ...")
        if os_name == "Windows":
            set_windows_env(var, val)
        else:
            set_unix_env(var, val)

    print("\n✅ Done! Please restart your terminal (or open a new one) for changes to take effect.\n")

#!/usr/bin/env python3
from qiskit_ibm_runtime import QiskitRuntimeService

# === FILL THESE IN ===
MY_TOKEN    = "Qfu3e8LAv3aqbOFynW4DgibgUEwHlaue3WnqlJyVKGq0"
MY_URL      = "https://us-east.quantum-computing.cloud.ibm.com"
MY_INSTANCE = "experiment"   # or the full CRN if required

# Persist to disk (~/.qiskit/qiskit-ibm.json):
QiskitRuntimeService.save_account(
    channel="ibm_cloud",
    token=MY_TOKEN,
    url=MY_URL,
    instance=MY_INSTANCE,
    overwrite=True
)

print("✅ Credentials saved! Now restart your shell/IDE.")


##
##QiskitRuntimeService.save_account(
##  token="f56367772bb8e1ccbbd9abcbed8d6a18a4633173cd6d04da31f6e709e866b4c7ab101bcc1154ca5587b97ce24240aa2f84bb990956be3a7f34b30c1fc4ea8765",
##  channel="ibm_quantum", # `channel` distinguishes between different account types.
##  instance="ibm-q/open/main",
##  overwrite=True # Only needed if you already have Cloud credentials.
##)

from qiskit_ibm_runtime import QiskitRuntimeService
##
##service = QiskitRuntimeService(
##    channel='ibm_quantum',
##    instance='ibm-q/open/main',
##    token='<IBM Quantum API key>'
##)

# Or save your credentials on disk.
# QiskitRuntimeService.save_account(channel='ibm_quantum', instance='ibm-q/open/main', token='<IBM Quantum API key>')
