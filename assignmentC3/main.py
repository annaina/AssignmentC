import numpy as np
import random
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2

def generate_locations(n, lat_range, lon_range):
    # generate n random (lat, lon) coordinates within given range
    return np.array([[random.uniform(*lat_range), random.uniform(*lon_range)] for _ in range(n)])


# define taxi stops and customer locations (real Oslo locations)
n_taxis = 10
n_customers = 10
oslo_lat_range = (59.8, 60.0)
oslo_lon_range = (10.6, 10.9)

# generate locations
taxis = generate_locations(n_taxis, oslo_lat_range, oslo_lon_range)
customers = generate_locations(n_customers, oslo_lat_range, oslo_lon_range)


# distance matrix w/ haversine
def haversine(coord1, coord2):
    """Calculate the great-circle distance between two points on the Earth."""
    R = 6371  # Earth's radius in km
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c  # Distance in km


# compute the distance matrix (taxis × customers)
dist_matrix = np.array([[haversine(t, c) for c in customers] for t in taxis])

# define binary pso
n_particles = 30
n_iterations = 120
w = 0.9
c1, c2 = 2.0, 2.0

convergence = []

# initialize BPSO Particles

def initialize_particle():
    # 10x10 matrix (one taxi per customer)
    identity = np.eye(n_taxis)
    np.random.shuffle(identity)  # Randomly permute rows to create diversity
    return identity


# initialize population
particles = np.array([initialize_particle() for _ in range(n_particles)])
velocities = np.random.uniform(-1, 1, (n_particles, n_taxis, n_customers))

# cost func
def calculate_cost(assignment):
    return np.sum(dist_matrix * assignment)


# evaluate initial fitness
fitness = np.array([calculate_cost(p) for p in particles])
p_best = particles.copy()
p_best_fitness = fitness.copy()
g_best = particles[np.argmin(fitness)]
g_best_fitness = np.min(fitness)

# run BPSO Optimization Loop (one to one)

for iteration in range(n_iterations):
    # update velocities
    r1, r2 = np.random.rand(n_particles, n_taxis, n_customers), np.random.rand(n_particles, n_taxis, n_customers)
    velocities = (
            w * velocities +
            c1 * r1 * (p_best - particles) +
            c2 * r2 * (g_best - particles)
    )

    # convert velocity to probability (sigmoid function)
    probabilities = 1 / (1 + np.exp(-velocities))

    # update particles (binary assignment)
    particles = (probabilities > np.random.rand(n_particles, n_taxis, n_customers)).astype(int)

    # enforce one-to-one constraint - code was a little buggy before this
    for i in range(n_particles):
        assigned_customers = set()
        assigned_taxis = set()

        # ensure each taxi is assigned exactly one customer
        for j in range(n_taxis):
            assigned_indices = np.where(particles[i, j, :] == 1)[0]

            if len(assigned_indices) != 1:
                particles[i, j, :] = 0  # reset if not assigned
                available_customers = list(set(range(n_customers)) - assigned_customers)
                choice = random.choice(available_customers)
                particles[i, j, choice] = 1
                assigned_customers.add(choice)

        # ensure each customer is assigned exactly one taxi
        for j in range(n_customers):
            assigned_indices = np.where(particles[i, :, j] == 1)[0]

            if len(assigned_indices) != 1:
                particles[i, :, j] = 0  # Reset
                available_taxis = list(set(range(n_taxis)) - assigned_taxis)
                choice = random.choice(available_taxis)
                particles[i, choice, j] = 1
                assigned_taxis.add(choice)

    # evaluate new fitness
    fitness = np.array([calculate_cost(p) for p in particles])

    # update personal best
    for i in range(n_particles):
        if fitness[i] < p_best_fitness[i]:
            p_best[i] = particles[i].copy()
            p_best_fitness[i] = fitness[i]

    # update global best
    if np.min(fitness) < g_best_fitness:
        g_best = particles[np.argmin(fitness)].copy()
        g_best_fitness = np.min(fitness)

    convergence.append(g_best_fitness)

    if iteration % 10 == 0:
        print(f"Iteration {iteration}: Best Cost = {g_best_fitness:.2f} km")

print(f"Optimal Assignment Found: {g_best_fitness:.2f} km")
print("Final Assignment Matrix:")
print(g_best)

# visualization
plt.figure(figsize=(10, 6))
plt.scatter(taxis[:, 1], taxis[:, 0], c='blue', marker='s', label="Taxis")
plt.scatter(customers[:, 1], customers[:, 0], c='red', marker='o', label="Customers")

for i in range(n_taxis):
    customer_idx = np.where(g_best[i] == 1)[0][0]
    plt.plot([taxis[i, 1], customers[customer_idx, 1]], [taxis[i, 0], customers[customer_idx, 0]], 'k--')

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Optimized Taxi Assignments in Oslo")
plt.legend()
plt.grid()
plt.show()

def plot_convergence_bpso(convergence, filename="convergence_bpso.png"):

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(convergence)), convergence, marker='o', linestyle='-', color='b', label="Best Cost (km)")

    plt.xlabel("Iteration")
    plt.ylabel("Best Cost (km)")
    plt.title("Convergence Plot of BPSO for Taxi Assignment")
    plt.grid(True)
    plt.legend()

    plt.savefig(filename)  # Save the figure
    print(f"Convergence plot saved as {filename}")
    plt.show()

plot_convergence_bpso(convergence)

