import os
from qiskit_ibm_runtime import QiskitRuntimeService

if __name__ == "__main__":
    api_key = input("Enter your IBM Quantum API key: ")
    os.environ["IBM_QUANTUM_API_KEY"] = api_key
    instance = input("Enter your instance CRN: ")
    os.environ["IBM_CRN"] = instance
    
    try:
        QiskitRuntimeService.save_account(
            channel="ibm_cloud",
            token=api_key,
            instance=instance,
            overwrite=True
        )
        print("✅ IBM Quantum account configured successfully!")
        print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
        print(f"Instance: {instance}")
        print("\nYou can now run experiments with IBM Quantum hardware.")
        
    except Exception as e:
        print(f"❌ Error configuring IBM Quantum account: {e}")
        print("Please check your API key and instance CRN.")

