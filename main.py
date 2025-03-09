import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import random
from math import radians, sin, cos, sqrt, atan2


# loads city coordinates from a text input
def load_cities_from_text(data):
    cities = []
    city_names = []

    for line in data.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) == 3:
            name, x, y = parts
            city_names.append(name)
            cities.append([float(x), float(y)])
    return np.array(cities, dtype=np.float64), city_names


city_data = """
Oslo 59.9139 10.7522
Bergen 60.3913 5.3221
Trondheim 63.4305 10.3951
Stavanger 58.9690 5.7331
Tromsø 69.6496 18.9560
Kristiansand 58.1467 7.9956
Ålesund 62.4722 6.1549
Drammen 59.7439 10.2045
Bodø 67.2804 14.4049
Sandnes 58.8524 5.7352
Tønsberg 59.2675 10.4070
Haugesund 59.4138 5.2680
Molde 62.7375 7.1591
Lillehammer 61.1153 10.4662
Harstad 68.7983 16.5417
Hamar 60.7945 11.0670
Gjøvik 60.7957 10.6916
Arendal 58.4617 8.7722
Halden 59.1297 11.3871
Moss 59.4340 10.6576
Porsgrunn 59.1406 9.6561
Skien 59.2096 9.5513
Sarpsborg 59.2830 11.1096
Fredrikstad 59.2180 10.9298
Steinkjer 64.0149 11.4954
Narvik 68.4385 17.4279
Alta 69.9689 23.2716
Honningsvåg 70.9821 25.9704
Vadsø 70.0743 29.7487
Kirkenes 69.7271 30.0450
Svolvær 68.2340 14.5684
Leknes 68.1475 13.6114
Brønnøysund 65.4745 12.2119
Mo i Rana 66.3128 14.1425
Førde 61.4510 5.8566
Florø 61.5999 5.0320
Voss 60.6281 6.4244"""

cities, city_names = load_cities_from_text(city_data)
n_cities = len(cities)


# function to find the Euclidean distance matrix between all cities
def distance_matrix(cities):
    n = len(cities)
    dist_matrix = np.zeros((n, n), dtype=np.float64) # initialize empty distance matrix
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(cities[i] - cities[j]) # compute Euclidean distance
    return dist_matrix


dist_matrix = distance_matrix(cities)

# Parameters
n_ants = 20  # Number of ants in each iteration
n_iterations = 150  # Number of iterations
alpha = 1  # Influence of pheromone trails
beta = 2  # Influence of visibility (1/distance)
decay = 0.1  # Pheromone evaporation rate
Q = 100  # Pheromone deposit factor

# initialize pheromone matrix with ones
def init_pheromones(n):
    return np.ones((n, n))


pheromones = init_pheromones(n_cities)

# function to select the next city for an ant based on probabilities
def select_next_city(ant, visited, pheromones, dist_matrix, alpha, beta):
    current_city = ant[-1]
    probabilities = []

    # compute selection probabilities for !unvisited! cities
    for city in range(n_cities):
        if city not in visited:
            pheromone = pheromones[current_city, city] ** alpha # pheromone influence

            # visibility - inverse of distance (shorter distances are better)
            visibility = (1 / dist_matrix[current_city, city]) ** beta if dist_matrix[current_city, city] > 0 else 1e6
            probabilities.append((city, pheromone * visibility))

    # normalize probabilities
    total = sum(prob for _, prob in probabilities)
    if total == 0: # if all probabilities are zero, select a random unvisited city
        return random.choice([city for city in range(n_cities) if city not in visited])

    # select city based on probabilities
    probabilities = [(city, prob / total) for city, prob in probabilities]
    rand_val = random.uniform(0, 1)
    cumulative = 0

    for city, prob in probabilities:
        cumulative += prob
        # select city when cumulative probability reaches random value
        if rand_val <= cumulative:
            return city

    return probabilities[-1][0] # default to last city if no selection happens

# function implementing ACO algorithm
def main_op(n_ants, n_iterations, pheromones, dist_matrix, alpha, beta, decay, Q):
    best_tour = None
    best_length = float('inf')

    print("Initial pheromones matrix:")
    print(pheromones)  # Debug
    convergence = [] # store best solution at each iteration - used for plot

    for iteration in range(n_iterations):
        all_tours = []
        all_lengths = []

        for ant in range(n_ants): # loop through each ant
            start = random.randint(0, n_cities - 1) # select a random starting city
            ant_tour = [start]
            visited = set(ant_tour)

            # create a full tour (visiting each city once)
            while len(visited) < n_cities:
                next_city = select_next_city(ant_tour, visited, pheromones, dist_matrix, alpha, beta)
                ant_tour.append(next_city)
                visited.add(next_city)

            ant_tour.append(start) # complete the tour
            tour_length = sum(dist_matrix[ant_tour[i], ant_tour[i + 1]] for i in range(n_cities)) # calculate the total length of the tour

            all_tours.append(ant_tour)
            all_lengths.append(tour_length)

        # check if we found a new best solution
        min_index = np.argmin(all_lengths)
        if all_lengths[min_index] < best_length:
            best_length = all_lengths[min_index]
            best_tour = all_tours[min_index]

        convergence.append(best_length)
        # Debug print: Track progress
        if iteration % 10 == 0 or iteration == n_iterations - 1:  # Print every 10 iterations
            print(f"Iteration {iteration}: Best length so far = {best_length}")

        pheromones *= (1 - decay) # pheromone evaporation

        # update pheromones based on tours found in this iteration
        for tour, length in zip(all_tours, all_lengths):
            for i in range(n_cities):
                city_a, city_b = tour[i], tour[i + 1]
                pheromones[city_a, city_b] += Q / length # deposit pheromones inversely proportional to length
                pheromones[city_b, city_a] += Q / length # ensure bidirectional update

    print("Final pheromones matrix:")
    print(pheromones)  # Debug
    print(f"Final best tour length: {best_length}")


    return best_tour, best_length, convergence


best_tour, best_length, convergence = main_op(n_ants, n_iterations, pheromones, dist_matrix, alpha, beta, decay, Q)

# Haversine function
def haversine(coord1, coord2):
    R = 6371  # Earth's radius in km
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

tour_length_km = sum(haversine(cities[best_tour[i]], cities[best_tour[i + 1]]) for i in range(n_cities))
print(f"Best tour length (km): {tour_length_km:.2f} km")


# visualisation
def plot_tour(cities, tour, city_names):
    plt.figure(figsize=(10, 6))
    plt.scatter(cities[:, 0], cities[:, 1], c='red', marker='o', label="Cities")

    # plot city names
    for i, city in enumerate(cities):
        plt.text(city[0] + 0.1, city[1] + 0.1, city_names[i], fontsize=9, color='blue')

    # draw path
    for i in range(len(tour) - 1):
        city1, city2 = tour[i], tour[i + 1]
        plt.plot([cities[city1][0], cities[city2][0]], [cities[city1][1], cities[city2][1]], 'b-')

    plt.plot([cities[tour[-1]][0], cities[tour[0]][0]], [cities[tour[-1]][1], cities[tour[0]][1]], 'b-', label="Path")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Best Tour Found by ACO")
    plt.legend()
    plt.show()

# convergence visualisation
def plot_convergence(convergence, filename="convergence_plot.png"):

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(convergence)), convergence, marker='o', linestyle='-', color='b', label="Best Tour Length")

    plt.xlabel("Iteration")
    plt.ylabel("Best Tour Length (km)")
    plt.title("Convergence Plot of ACO for TSP")
    plt.grid(True)
    plt.legend()

    plt.savefig(filename)  # save the figure
    print(f"Convergence plot saved as {filename}")
    plt.show()

plot_tour(cities, best_tour, city_names)
plot_convergence(convergence)






