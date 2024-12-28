import numpy as np
from scipy.linalg import expm  # For matrix exponentiation

# Define parameters
delta = 1.0  # Energy splitting of the black hole qubit
J = 0.5      # Coupling strength between black hole and radiation qubits
Omega = 0.3  # Driving amplitude
omega = 2.0  # Driving frequency
t_max = 10   # Total simulation time
num_steps = 100  # Number of time steps

# Define Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])  # Pauli-X
sigma_z = np.array([[1, 0], [0, -1]])  # Pauli-Z
identity = np.eye(2)  # Identity matrix

# Define operators for the two-qubit system
sigma_z_BH = np.kron(sigma_z, identity)  # Pauli-Z for black hole qubit
sigma_z_Rad = np.kron(identity, sigma_z)  # Pauli-Z for radiation qubit
sigma_x_BH = np.kron(sigma_x, identity)  # Pauli-X for black hole qubit

# Time-independent part of the Hamiltonian
H_0 = (delta / 2) * sigma_z_BH + J * np.dot(sigma_z_BH, sigma_z_Rad)

# Time-dependent part of the Hamiltonian
def H_t(t):
    return Omega * np.cos(omega * t) * sigma_x_BH

# Full Hamiltonian as a function of time
def H(t):
    return H_0 + H_t(t)

# Time evolution using matrix exponentiation
def time_evolve(initial_state, t_max, num_steps):
    dt = t_max / num_steps
    state = initial_state
    for step in range(num_steps):
        t = step * dt
        U = expm(-1j * H(t) * dt)  # Time evolution operator
        state = np.dot(U, state)  # Apply time evolution
    return state

# Define the initial state (black hole + radiation qubits in |00>)
initial_state = np.array([1, 0, 0, 0])  # |00> in computational basis

# Simulate time evolution
final_state = time_evolve(initial_state, t_max, num_steps)

# Print final state
print("Final state vector:")
print(final_state)
