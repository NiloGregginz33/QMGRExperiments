#!/usr/bin/env python3
import os
import platform
import subprocess

def set_env_var(key: str, val: str):
    system = platform.system()
    if system == "Windows":
        # setx writes to your user environment (needs a new shell to take effect)
        subprocess.run(["setx", key, val], check=True)
    else:
        # append to your shell rc file (~/.bashrc, ~/.zshrc, or ~/.profile)
        shell = os.environ.get("SHELL","")
        if "zsh" in shell:
            rc = os.path.expanduser("~/.zshrc")
        elif "bash" in shell:
            rc = os.path.expanduser("~/.bashrc")
        else:
            rc = os.path.expanduser("~/.profile")
        line = f'\nexport {key}="{val}"\n'
        with open(rc, "a") as f:
            f.write(line)
        # also set for this process
        os.environ[key] = val

if __name__ == "__main__":
    print("This will store your IBM Quantum API token & URL as environment variables.")
    token = input("→ Paste your API token: ").strip()
    url   = input("→ Paste your IBM Quantum URL (e.g. https://us-east.quantum-computing.cloud.ibm.com): ").strip()
    if not token or not url:
        print("Aborted: both token and URL are required.")
        exit(1)

    set_env_var("QISKIT_IBM_PROVIDER_TOKEN", token)
    set_env_var("QISKIT_IBM_PROVIDER_URL",   url)

    print("\n✅  Done.  Now restart your shell or IDE, and Qiskit will pick up:")
    print("    QISKIT_IBM_PROVIDER_TOKEN & QISKIT_IBM_PROVIDER_URL")
