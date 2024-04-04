from tsplib95 import load
from random import choice
from math import sqrt

#TODO:
# steepest, wierzchołki
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
    # count whole path's length
    result = 0
    for i in range(len(solution)-1):
        result += matrix[solution[i]][solution[i+1]]
    return result

def delta_for_vertices(matrix, solution, i, j):
    # object function
    return -matrix[solution[i-1]][solution[i]] - matrix[solution[i+1]][solution[i]] - matrix[solution[j-1]][solution[j]] - matrix[solution[j+1]][solution[j]] \
    + matrix[solution[i-1]][solution[j]] + matrix[solution[i+1]][solution[j]] + matrix[solution[j-1]][solution[i]] + matrix[solution[j+1]][solution[i]]


def steepest_vertex(solution, matrix):
    delta = count_result(solution, matrix)
    improving = True
    while improving:
        vertices = [None, None]
        for i in range(1, len(solution)-3):
            for j in range(i+2, len(solution)-1):
                if delta + delta_for_vertices(matrix, solution, i, j) < delta:
                    vertices = [solution.index(solution[i]), solution.index(solution[j])]
                    delta = delta + delta_for_vertices(matrix, solution, i, j)

        if vertices != [None, None]:
            solution[vertices[0]], solution[vertices[1]] = solution[vertices[1]], solution[vertices[0]]
            print(vertices)
        else:
            improving = False

    return solution
        

def main():
    prob = load_problem('kroA100.tsp')
    matrix = create_distance_matrix(prob)
    random_solution = generate_random_solution(matrix)
    print(f'Before: {count_result(random_solution[0], matrix)}')
    #print(steepest_vertex(random_solution[0], matrix))
    print(f'After: {count_result(steepest_vertex(random_solution[0], matrix), matrix)}')


if __name__ == '__main__':
    main()