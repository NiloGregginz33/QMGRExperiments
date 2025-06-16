import os
import subprocess

def list_python_files():
    """List all Python files in the current directory."""
    return [f for f in os.listdir('.') if f.endswith('.py')]

def display_menu(files):
    """Display a menu of Python files to the user."""
    print("\nAvailable Python Files:")
    for idx, file in enumerate(files, start=1):
        print(f"{idx}. {file}")
    print("0. Exit")

def execute_file(file_name):
    """Execute a Python file and display its output."""
    try:
        print(f"\nExecuting {file_name}...\n")
        result = subprocess.run(['python', file_name], capture_output=True, text=True)
        print("Output:")
        print(result.stdout)
        print("\nErrors (if any):")
        print(result.stderr)
    except Exception as e:
        print(f"Error executing {file_name}: {e}")

def main():
    """Main program loop."""
    while True:
        python_files = list_python_files()
        if not python_files:
            print("No Python files found in the current directory.")
            break

        display_menu(python_files)

        try:
            choice = int(input("\nSelect a file to run by number (0 to exit): "))
            if choice == 0:
                print("Exiting program.")
                break
            elif 1 <= choice <= len(python_files):
                execute_file(python_files[choice - 1])
            else:
                print("Invalid choice. Please select a valid option.")
        except ValueError:
            print("Please enter a valid number.")

if __name__ == "__main__":
    main()
