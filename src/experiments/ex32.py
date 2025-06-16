import numpy as np
import matplotlib.pyplot as plt

# Parameters for Lattice QCD simulation
lattice_size = 8  # Lattice grid size
beta = 5.5  # Inverse coupling constant (related to the strength of QCD interactions)
kappa = 0.16  # Hopping parameter (quark mass effect)
n_iterations = 1000  # Number of iterations for Monte Carlo updates

# Initialize the lattice gauge field
# Each link has SU(3) matrices representing color charge interactions
def initialize_lattice(size):
    return np.random.rand(size, size, size, size, 3, 3)  # Random SU(3) elements

def su3_project(matrix):
    """ Project matrix to SU(3) (simplified placeholder). """
    u, _, v = np.linalg.svd(matrix)
    return np.dot(u, v)

# Action for the lattice field
def plaquette_action(U):
    action = 0.0
    for x in range(lattice_size):
        for y in range(lattice_size):
            for z in range(lattice_size):
                for t in range(lattice_size):
                    # Simplified plaquette action term
                    U_mu_nu = U[x, y, z, t]  # Links in mu-nu plane (placeholder logic)
                    action += np.trace(U_mu_nu.T @ U_mu_nu).real
    return action

# Monte Carlo update step for lattice field
def monte_carlo_step(U, beta):
    for x in range(lattice_size):
        for y in range(lattice_size):
            for z in range(lattice_size):
                for t in range(lattice_size):
                    old_link = U[x, y, z, t]
                    random_update = np.random.randn(3, 3)
                    updated_link = su3_project(old_link + beta * random_update)

                    # Metropolis-Hastings condition
                    delta_action = plaquette_action(updated_link) - plaquette_action(old_link)
                    if delta_action < 0 or np.exp(-beta * delta_action) > np.random.rand():
                        U[x, y, z, t] = updated_link  # Accept new link
    return U

# Measure quark propagators (simplified example)
def measure_quark_propagator(U):
    """ Compute quark propagator (simplified placeholder). """
    propagator = np.sum(U, axis=(0, 1, 2, 3)) / (lattice_size**4)
    return propagator

# Main simulation loop
U = initialize_lattice(lattice_size)
action_history = []
propagators = []

for step in range(n_iterations):
    U = monte_carlo_step(U, beta)
    if step % 10 == 0:  # Measure every 10 steps
        action = plaquette_action(U)
        propagator = measure_quark_propagator(U)
        action_history.append(action)
        propagators.append(propagator)
        print(f"Step {step}, Action: {action}, Propagator: {propagator}")

# Plot action history
plt.plot(action_history)
plt.title("Action History")
plt.xlabel("Iteration")
plt.ylabel("Action")
plt.show()

# Analyze propagators
plt.plot([np.linalg.norm(p) for p in propagators])
plt.title("Quark Propagators Over Time")
plt.xlabel("Iteration")
plt.ylabel("Norm of Propagator")
plt.show()

# Additional analysis can focus on color charge contributions to propagators
