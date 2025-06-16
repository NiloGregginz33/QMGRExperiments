from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()

print("Available instances:")
print(service.backends())
print(service.hub, service.group, service.project)
