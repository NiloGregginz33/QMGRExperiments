import os
from qiskit_ibm_runtime import QiskitRuntimeService

def initialize_ibm_backend():
    """Automatically initializes IBM Quantum backend using an API key from environment variables."""
    
    api_key = os.getenv("IBM_QUANTUM_API_KEY")  # Set this in your environment before running
    if not api_key:
        print("Error: IBM Quantum API key not found. Set it using 'export IBM_QUANTUM_API_KEY=your_api_key'")
        return None

    try:
        IBMProvider.save_account(api_key, overwrite=True)  # Save the API key
        provider = IBMProvider()  # Load the provider
        backend = provider.get_backend("ibmq_qasm_simulator")  # Change to any preferred backend
        print(f"Initialized backend: {backend.name}")
        return backend
    except Exception as e:
        print(f"Failed to initialize backend: {e}")
        return None

if __name__ == "__main__":
    api_key = input("Enter your IBM Quantum API key: ")
    os.environ["IBM_QUANTUM_API_KEY"] = api_key
    instance = input("Enter your instance CRN: ")
    os.environ["IBM_CRN"] = instance
    QiskitRuntimeService.save_account(
    channel="ibm_cloud",
    token=api_key,
    instance=instance,
    overwrite=True
    )
    backend = initialize_ibm_backend()

