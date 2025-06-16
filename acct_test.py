from qiskit_ibm_provider import IBMProvider
from qiskit import transpile

# same credentials, same CRN, same URL
provider = IBMProvider(
#    channel  = "ibm_cloud",
    token    = "Qfu3e8LAv3aqbOFynW4DgibgUEwHlaue3WnqlJyVKGq0"
#    url      = "https://quantum-computing.cloud.ibm.com",
#    instance = "crn:v1:bluemix:public:quantum-computing:us-east:a/7452d0294011465fa5f5e4ebb4ff75cc:164174a4-635c-40d2-aaa6-770d19977db2::"
)

backend = provider.get_backend("ibm_brisbane")

# build & run
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

tbell    = transpile(qc, backend, optimization_level=1)
job      = backend.run(tbell, shots=1024)
counts   = job.result().get_counts()
print(counts)
