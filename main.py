from tsplib95 import load
from random import choice
from math import sqrt, inf

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

def delta_for_inner_vertices(matrix, solution, i, j, l_or_r):
    # object function
    return -matrix[solution[l_or_r][i-1]][solution[l_or_r][i]] - matrix[solution[l_or_r][i+1]][solution[l_or_r][i]] \
           -matrix[solution[l_or_r][j-1]][solution[l_or_r][j]] - matrix[solution[l_or_r][j+1]][solution[l_or_r][j]] \
           +matrix[solution[l_or_r][i-1]][solution[l_or_r][j]] + matrix[solution[l_or_r][i+1]][solution[l_or_r][j]] \
           +matrix[solution[l_or_r][j-1]][solution[l_or_r][i]] + matrix[solution[l_or_r][j+1]][solution[l_or_r][i]]


def delta_for_outer_vertices(matrix, solution, i, j, l_or_r):
    # object function
    return -matrix[solution[l_or_r][i-1]][solution[l_or_r][i]] - matrix[solution[l_or_r][i+1]][solution[l_or_r][i]] \
           -matrix[solution[abs(l_or_r-1)][j-1]][solution[abs(l_or_r-1)][j]] - matrix[solution[abs(l_or_r-1)][j+1]][solution[abs(l_or_r-1)][j]] \
           +matrix[solution[l_or_r][i-1]][solution[abs(l_or_r-1)][j]] + matrix[solution[l_or_r][i+1]][solution[abs(l_or_r-1)][j]] \
           +matrix[solution[abs(l_or_r-1)][j-1]][solution[l_or_r][i]] + matrix[solution[abs(l_or_r-1)][j+1]][solution[l_or_r][i]]

def switch_inner_vertices(solution, left_or_right, vertices):
    pass


def switch_outer_vertices(solution, left_or_right, vertices):
    pass


def steepest_vertex(solution, matrix):
    improving = True
    left_or_right = 0 # 0 - left, 1 - right
    while improving:
        delta_inner = delta_outer = 0
        vertices_inner = vertices_outer = [None, None]

        # inner vertices
        for i in range(1, len(solution[left_or_right])-3):
            for j in range(i+2, len(solution[left_or_right])-1):
                delta = delta_for_inner_vertices(matrix, solution, i, j, left_or_right)
                if delta < 0:
                    vertices_inner = [solution[left_or_right].index(solution[left_or_right][i]), solution[left_or_right].index(solution[left_or_right][j])]
                    delta_inner = delta

        # outer vertices
        for i in range(1, len(solution[left_or_right])-1):
            for j in range(1, len(solution[abs(left_or_right-1)])-1):
                delta = delta_for_outer_vertices(matrix, solution, i, j, left_or_right)
                if delta < 0:
                    vertices_outer = [solution[left_or_right].index(solution[left_or_right][i]), solution[abs(left_or_right-1)].index(solution[abs(left_or_right-1)][j])]
                    delta_outer = delta
        
        if vertices_inner != [None, None] or vertices_outer != [None, None]:

            if delta_inner < delta_outer:
                solution[left_or_right][vertices_inner[0]], solution[left_or_right][vertices_inner[1]] = solution[left_or_right][vertices_inner[1]], solution[left_or_right][vertices_inner[0]]
            else:
                solution[left_or_right][vertices_outer[0]], solution[abs(left_or_right-1)][vertices_outer[1]] = solution[abs(left_or_right-1)][vertices_outer[1]], solution[left_or_right][vertices_outer[0]]
        else:
            improving = False

        left_or_right = abs(left_or_right-1)

    return solution
        

def main():
    prob = load_problem('kroA100.tsp')
    matrix = create_distance_matrix(prob)
    random_solution = generate_random_solution(matrix)
    print(f'Before left: {count_result(random_solution[0], matrix)}')
    print(f'Before right: {count_result(random_solution[1], matrix)}')
    #print(steepest_vertex(random_solution[0], matrix))
    solution = steepest_vertex(random_solution, matrix)
    print(f'After left: {count_result(solution[0], matrix)}')
    print(f'After left: {count_result(solution[1], matrix)}')


if __name__ == '__main__':
    main()
