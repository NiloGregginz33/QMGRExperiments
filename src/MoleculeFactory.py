from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit_algorithms import VQE
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Estimator
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

# Use PySCFDriver to define the molecule
driver = PySCFDriver(
    atom="O 0.0 0.0 0.0; H 0.0 0.76 0.58; H 0.0 -0.76 0.58",
    basis="sto-3g"
)

# Extract electronic structure problem
problem = driver.run()

hamiltonian = problem.hamiltonian

print(hamiltonian)

# Map the problem to a qubit Hamiltonian
mapper = JordanWignerMapper()
qubit_op = mapper.map(problem.second_q_ops()["ElectronicEnergy"])

# Define a variational form (ansatz)
ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz", reps=1)

# Use AerSimulator backend
simulator = Aer.get_backend("aer_simulator")

# VQE algorithm with the Estimator primitive
vqe = VQE(estimator=Estimator(), ansatz=ansatz)

# Solve for the ground state energy
result = vqe.compute_minimum_eigenvalue(qubit_op)
print("Ground state energy (H2):", result.eigenvalue.real)
