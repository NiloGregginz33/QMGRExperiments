import subprocess

def generate_requirements_from_pip_list(output_file="requirements.txt"):
    try:
        # Run `pip list` and capture the output
        result = subprocess.run(
            ["pip", "list", "--format=freeze"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        # Check for errors
        if result.returncode != 0:
            print("Error running pip list:", result.stderr)
            return
        
        # Write the output to the specified file
        with open(output_file, "w") as f:
            f.write(result.stdout)
        
        print(f"Requirements saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
generate_requirements_from_pip_list()
