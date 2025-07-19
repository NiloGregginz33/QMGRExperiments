from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# Minimal test: 2-qubit Bell state
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")  # Change to any valid backend if needed

qc_t = transpile(qc, backend)

print(f"Submitting job to backend: {backend.name}")

sampler = Sampler(backend)
job = sampler.run([qc_t], shots=128)
result = job.result()
print("Raw result:", result)

# Try to extract bitstrings if possible
try:
    pub_result = result[0]
    bitstrings = pub_result.data.meas.get_bitstrings()
    print("First 10 bitstrings:", bitstrings[:10])
except Exception as e:
    print("Could not extract bitstrings:", e)
