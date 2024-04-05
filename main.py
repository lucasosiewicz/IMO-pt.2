from tsplib95 import load
from math import sqrt
from random import choice

def load_problem(filename):
    prob = load(filename)
    nodes_x = [x[0] for _, x in prob.node_coords.items()]
    nodes_y = [y[1] for _, y in prob.node_coords.items()]

    return {'x': nodes_x, 'y': nodes_y}
 
 
def create_distance_matrix(prob):
    n = len(prob['x'])
    matrix_of_distances = []
    for i in range(n):
        row = []
        for j in range(n):
            value = sqrt((prob['x'][i] - prob['x'][j])**2 + (prob['y'][i] - prob['y'][j])**2)
            row.append(value)
        matrix_of_distances.append(row)
    
    return matrix_of_distances
 

def generate_random_solution(matrix):
    unique_nodes = [i for i in range(len(matrix))]
    result_left = 0
    result_right = 0
    solution_left = []
    solution_right = []
    left_or_right = False # True - left, False - right
    while unique_nodes:
        left_or_right = not left_or_right
        node = choice(unique_nodes)
        unique_nodes.remove(node)
        solution_left.append(node) if left_or_right else solution_right.append(node)

    return [solution_left, solution_right]


def count_result(solution, matrix):
    total_cost = 0
    for i in range(len(solution) - 1):
        total_cost += matrix[solution[i]][solution[i+1]]
    return total_cost


def delta_edge(matrix, solution, i, j):
    n = len(matrix)
    a, b = min(i,j), max (i,j)
    s, l = (a-1) % n, (b-1) % n # Świetnie się tu bawiłem
    return -matrix[solution[a]][solution[s]] - matrix[solution[b]][solution[l]] \
    + matrix[solution[a]][solution[b]]+matrix[solution[s]][solution[l]]

'''
def greedy_algorithm(matrix, solution):
    current_solution = solution.copy()
    current_cost = calculate_solution_cost(current_solution, matrix)
    improving = True

    while improving:
        improving = False

        #TODO Wewnątrz trasowa i miedzy trasowa zamiana wierzchołkow
        #Zamiana krawędzi i sprawdzanie poprawy
        new_solution = swap_edges(current_solution)
        new_cost = calculate_solution_cost(new_solution, matrix)
        if new_cost < current_cost:
            current_solution = new_solution
            current_cost = new_cost
            improving = True

    return current_solution
'''
def greedy_algorithm(matrix):
    n = len(matrix)
    solution = [0] 
    visited = [False] * n
    visited[0] = True

    for _ in range(n - 1):
        last_node = solution[-1]
        min_dist = float('inf')
        min_node = -1
        for neighbor in range(n):
            if not visited[neighbor] and matrix[last_node][neighbor] < min_dist:
                min_dist = matrix[last_node][neighbor]
                min_node = neighbor
        solution.append(min_node)
        visited[min_node] = True

    improving = True
    while improving:
        improving = False
        for i in range(n):
            for j in range(i + 2, n):
                delta = delta_edge(matrix, solution, i, j)
                if delta < 0:
                    continue
                if delta < min_dist:
                    min_dist = delta
                    min_i, min_j = i, j
                    improving = True

        if improving:
            solution[min_i + 1:min_j + 1] = solution[min_i + 1:min_j + 1][::-1]  # Reverse the subsequence

    return solution
            
def main():
    prob = load_problem('kroA100.tsp')
    matrix = create_distance_matrix(prob)
    random_solution = generate_random_solution(matrix)

if __name__ == '__main__':
    main()