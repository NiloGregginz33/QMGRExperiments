import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
from collections import Counter
import numpy as np
import numpy as np
from qiskit.result import Counts

# Parameters for Lattice QCD simulation
lattice_size = 8  # Lattice grid size
beta = 5.5  # Inverse coupling constant (related to the strength of QCD interactions)
kappa = 0.16  # Hopping parameter (quark mass effect)
n_iterations = 200  # Number of iterations for Monte Carlo updates

# Initialize the lattice gauge field
# Each link has SU(3) matrices representing color charge interactions
def initialize_lattice(size):
    return np.random.rand(size, size, size, size, 3, 3)  # Random SU(3) elements

def su3_project(matrix):
    """ Project matrix to SU(3) (simplified placeholder). """
    u, _, v = np.linalg.svd(matrix)
    return np.dot(u, v)

# Compute Polyakov loops
def compute_polyakov_loop(U):
    """ Calculate the Polyakov loop to study deconfinement. """
    polyakov_loops = []
    for x in range(lattice_size):
        for y in range(lattice_size):
            for z in range(lattice_size):
                loop = np.trace(U[x, y, z, 0])
                for t in range(1, lattice_size):
                    loop = loop @ U[x, y, z, t]
                polyakov_loops.append(np.trace(loop).real)
    return np.mean(polyakov_loops)

def plaquette_action(U, x=None, y=None, z=None, t=None, new_link=None):
    """
    Compute the plaquette action for the lattice or a specific link.
    U: Lattice configuration
    x, y, z, t: Coordinates for a single plaquette (optional)
    new_link: Optional updated link for Metropolis-Hastings updates
    """
    action = 0.0
    directions = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]

    if x is None:  # Compute global action
        for x in range(lattice_size):
            for y in range(lattice_size):
                for z in range(lattice_size):
                    for t in range(lattice_size):
                        action += plaquette_action(U, x, y, z, t)
        return action

    # Local action for a specific plaquette
    for dx, dy, dz, dt in directions:
        neighbor_x = (x + dx) % lattice_size
        neighbor_y = (y + dy) % lattice_size
        neighbor_z = (z + dz) % lattice_size
        neighbor_t = (t + dt) % lattice_size
        U_mu = U[x, y, z, t] if new_link is None else new_link
        U_nu = U[neighbor_x, neighbor_y, neighbor_z, neighbor_t]
        action += np.trace(U_mu @ U_nu.T.conj()).real
    return action

def initialize_lattice(size):
    """Initialize a 4D lattice with SU(3) matrices at each link."""
    lattice = np.empty((size, size, size, size, 3, 3), dtype=np.complex128)
    for x in range(size):
        for y in range(size):
            for z in range(size):
                for t in range(size):
                    random_matrix = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
                    lattice[x, y, z, t] = su3_project(random_matrix)
    return lattice


# Monte Carlo update step for lattice field
def monte_carlo_step(U, beta):
    for x in range(lattice_size):
        for y in range(lattice_size):
            for z in range(lattice_size):
                for t in range(lattice_size):
                    old_link = U[x, y, z, t]
                    random_update = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
                    updated_link = su3_project(old_link + beta * random_update)

                    # Metropolis-Hastings condition
                    delta_action = (
                        plaquette_action(U, x, y, z, t, updated_link)
                        - plaquette_action(U, x, y, z, t, old_link)
                    )
                    if delta_action < 0 or np.exp(-beta * delta_action) > np.random.rand():
                        U[x, y, z, t] = updated_link  # Accept new link
    return U


# Measure quark propagators (simplified example)
def measure_quark_propagator(U):
    """ Compute quark propagator (simplified placeholder). """
    propagator = np.sum(U, axis=(0, 1, 2, 3)) / (lattice_size**4)
    return propagator

# Compute Wilson loops
def compute_wilson_loop(U):
    """ Calculate the Wilson loop to study confinement. """
    wilson_loops = []
    for x in range(lattice_size):
        for y in range(lattice_size):
            for z in range(lattice_size):
                for t in range(lattice_size):
                    loop = np.trace(U[x, y, z, t].T @ U[x, (y + 1) % lattice_size, z, t]).real
                    wilson_loops.append(loop)
    return np.mean(wilson_loops)

# Compute Polyakov loops
def compute_polyakov_loop(U):
    """ Calculate the Polyakov loop to study deconfinement. """
    polyakov_loops = []
    for x in range(lattice_size):
        for y in range(lattice_size):
            for z in range(lattice_size):
                loop = U[x, y, z, 0]  # Start with the first time slice
                for t in range(1, lattice_size):
                    loop = loop @ U[x, y, z, t]  # Multiply matrices along the time direction
                polyakov_loops.append(np.trace(loop).real)  # Take the trace at the end
    return np.mean(polyakov_loops)

# Compute color charge contributions
def compute_color_contributions(U):
    """ Analyze contributions of red, green, and blue color charges in the propagator. """
    red_contribution = 0
    green_contribution = 0
    blue_contribution = 0

    for x in range(lattice_size):
        for y in range(lattice_size):
            for z in range(lattice_size):
                for t in range(lattice_size):
                    link = U[x, y, z, t]
                    red_contribution += np.abs(link[0, 0])**2
                    green_contribution += np.abs(link[1, 1])**2
                    blue_contribution += np.abs(link[2, 2])**2

    total_contribution = red_contribution + green_contribution + blue_contribution
    red_fraction = red_contribution / total_contribution
    green_fraction = green_contribution / total_contribution
    blue_fraction = blue_contribution / total_contribution

    return red_fraction, green_fraction, blue_fraction

# Main simulation loop
U = initialize_lattice(lattice_size)
action_history = []
propagators = []
wilson_loop_history = []
polyakov_loop_history = []
color_contributions = []

for step in range(n_iterations):
    U = monte_carlo_step(U, beta)
    if step % 10 == 0:  # Measure every 10 steps
        action = plaquette_action(U)  # Global action
        propagator = measure_quark_propagator(U)
        wilson_loop = compute_wilson_loop(U)
        polyakov_loop = compute_polyakov_loop(U)
        red_frac, green_frac, blue_frac = compute_color_contributions(U)

        action_history.append(action)
        propagators.append(propagator)
        wilson_loop_history.append(wilson_loop)
        polyakov_loop_history.append(polyakov_loop)
        color_contributions.append((red_frac, green_frac, blue_frac))

        print(f"Step {step}, Action: {action}, Propagator: {propagator}, Wilson Loop: {wilson_loop}, Polyakov Loop: {polyakov_loop}, Color Contributions (R,G,B): ({red_frac:.3f}, {green_frac:.3f}, {blue_frac:.3f})")


        print(f"Step {step}, Action: {action}, Propagator: {propagator}, Wilson Loop: {wilson_loop}, Polyakov Loop: {polyakov_loop}")

        print(f"Step {step}, Action: {action}, Propagator: {propagator}")

# Plot color charge contributions
reds, greens, blues = zip(*color_contributions)
plt.plot(reds, label="Red Contribution")
plt.plot(greens, label="Green Contribution")
plt.plot(blues, label="Blue Contribution")
plt.title("Color Charge Contributions")
plt.xlabel("Iteration")
plt.ylabel("Fraction of Total Contribution")
plt.legend()
plt.show()

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
