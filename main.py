import matplotlib.pyplot as plt
from tsplib95 import load
from random import choice
from pathlib import Path
from math import sqrt
import numpy as np
import time


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

def delta_for_edges(matrix, solution, i, j, k, l, l_or_r):
    # object function
    return -matrix[solution[l_or_r][i]][solution[l_or_r][i+1]] - matrix[solution[1-l_or_r][k]][solution[1-l_or_r][k+1]] \
           - matrix[solution[l_or_r][j]][solution[l_or_r][j+1]] - matrix[solution[1-l_or_r][l]][solution[1-l_or_r][l+1]] \
           + matrix[solution[l_or_r][i]][solution[1-l_or_r][k]] + matrix[solution[1-l_or_r][k+1]][solution[l_or_r][i+1]] \
           + matrix[solution[l_or_r][j]][solution[1-l_or_r][l]] + matrix[solution[1-l_or_r][l+1]][solution[l_or_r][j+1]]

def switch_inner_vertices(solution, left_or_right, vertices):
    solution[left_or_right][vertices[0]], solution[left_or_right][vertices[1]] = solution[left_or_right][vertices[1]], solution[left_or_right][vertices[0]]


def switch_outer_vertices(solution, left_or_right, vertices):
    solution[left_or_right][vertices[0]], solution[abs(left_or_right-1)][vertices[1]] = solution[abs(left_or_right-1)][vertices[1]], solution[left_or_right][vertices[0]]

def edge_swap(solution, left_or_right, vertices):
    solution[left_or_right][vertices[0]], solution[left_or_right][vertices[2]] = solution[left_or_right][vertices[2]], solution[left_or_right][vertices[0]]
    solution[left_or_right][vertices[1]], solution[1 - left_or_right][vertices[3]] = solution[1 - left_or_right][vertices[3]], solution[left_or_right][vertices[1]]
    return solution

def greedy_edges(solution, matrix):
    improving = True
    while improving:
        min_delta = float('inf')
        vertices = [None, None, None, None]
        for i in range(len(solution[0])-1):
            for j in range(i+1, len(solution[0])-1):
                for k in range(len(solution[1])-1):
                    for l in range(k+1, len(solution[1])-1):
                        delta = delta_for_edges(matrix, solution, i, j, k, l, 0)
                        if delta < min_delta:
                            min_delta = delta
                            vertices = [i, j, k, l]
        if min_delta < 0:
            solution = edge_swap(solution, 0, vertices)
        else:
            improving = False
    return solution

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
                switch_inner_vertices(solution, left_or_right, vertices_inner)
            else:
                switch_outer_vertices(solution, left_or_right, vertices_outer)
        else:
            improving = False

        left_or_right = abs(left_or_right-1)

    return solution


def draw_and_save_paths(prob, solution, dir_name, filename):

    # define path and dir to save plots
    path = Path.cwd() / dir_name

    # if dir doesn't exists, create it
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    # create a figure
    plt.figure()
    plt.scatter(prob['x'], prob['y'])
    for sol, c in zip(solution, ['r', 'b']):
        for i in range(len(sol)-1):
            x = [prob['x'][sol[i]], prob['x'][sol[i+1]]]
            y = [prob['y'][sol[i]], prob['y'][sol[i+1]]]
            plt.plot(x, y, c=c, linewidth=0.7)
        last_x = [prob['x'][sol[0]], prob['x'][sol[-1]]]
        last_y = [prob['y'][sol[0]], prob['y'][sol[-1]]]
        plt.plot(last_x, last_y, c=c, linewidth=0.7)

    # remove axes and make figure smooth and tight
    plt.axis(False)
    plt.tight_layout()
    # save figure
    plt.savefig(f'{path}\{filename}.png')
    plt.close()


def random_walk(solution, matrix):
    improving = True
    type_of_neighborhood = [0,1,2]  # 0 - inner vertices, 1 - inner edges, 2 - outer vertices
    start = time.time()
    stop = time.time()
    while stop - start < 0.981:
        # random choice of movement and path
        movement = choice(type_of_neighborhood)
        left_or_right = choice([0,1])
        if movement == 0:       
            # inner vertices
            vertices = [choice(solution[left_or_right][1:-2]), choice(solution[left_or_right][1:-2])]
            i = solution[left_or_right].index(vertices[0])
            j = solution[left_or_right].index(vertices[1])

            if delta_for_inner_vertices(matrix, solution, i, j, left_or_right) < 0:
                switch_inner_vertices(solution, left_or_right, [i, j])
        elif movement == 1:
            # inner edges
            vertices = [choice(solution[left_or_right][1:-2]), choice(solution[left_or_right][1:-2]),
                        choice(solution[left_or_right][1:-2]), choice(solution[left_or_right][1:-2])]
            i, j, k, l = [solution[left_or_right].index(v) for v in vertices]

            if delta_for_edges(matrix, solution, i, j, k, l, left_or_right) < 0:
                solution = edge_swap(solution, left_or_right, [i, j, k, l])
        else:
            # outer vertices
            vertices = [choice(solution[left_or_right][1:-1]), choice(solution[abs(left_or_right-1)][1:-1])]
            i = solution[left_or_right].index(vertices[0])
            j = solution[abs(left_or_right-1)].index(vertices[1])

            if delta_for_outer_vertices(matrix, solution, i, j, left_or_right) < 0:
                switch_outer_vertices(solution, left_or_right, [i, j])
        stop = time.time()

    return solution

def second_regret(solution, matrix):
    improving = True
    type_of_neighborhood = [0, 1, 2]  # 0 - inner vertices, 1 - inner edges, 2 - outer vertices
    best_solution = solution
    best_delta = float('inf')
    second_best_solution = solution
    second_best_delta = float('inf')

    while improving:
        # Random choice of movement
        movement = choice(type_of_neighborhood)
        left_or_right = choice([0, 1])
        if movement == 0:
            # Inner vertices
            vertices = [choice(solution[left_or_right][1:-2]), choice(solution[left_or_right][1:-2])]
            i = solution[left_or_right].index(vertices[0])
            j = solution[left_or_right].index(vertices[1])

            delta = delta_for_inner_vertices(matrix, solution, i, j, left_or_right)
            if delta < 0:
                switch_inner_vertices(solution, left_or_right, [i, j])
            elif delta < best_delta:
                second_best_delta = best_delta
                second_best_solution = best_solution.copy()
                best_delta = delta
                best_solution = solution.copy()

        elif movement == 1:
            # Inner edges
            vertices = [choice(solution[left_or_right][1:-2]), choice(solution[left_or_right][1:-2]),
                        choice(solution[left_or_right][1:-2]), choice(solution[left_or_right][1:-2])]
            i, j, k, l = [solution[left_or_right].index(v) for v in vertices]

            delta = delta_for_edges(matrix, solution, i, j, k, l, left_or_right)
            if delta < 0:
                solution = edge_swap(solution, left_or_right, [i, j, k, l])
            elif delta < best_delta:
                second_best_delta = best_delta
                second_best_solution = best_solution.copy()
                best_delta = delta
                best_solution = solution.copy()

        else:
            # Outer vertices
            vertices = [choice(solution[left_or_right][1:-1]), choice(solution[abs(left_or_right - 1)][1:-1])]
            i = solution[left_or_right].index(vertices[0])
            j = solution[abs(left_or_right - 1)].index(vertices[1])

            delta = delta_for_outer_vertices(matrix, solution, i, j, left_or_right)
            if delta < 0:
                switch_outer_vertices(solution, left_or_right, [i, j])
            elif delta < best_delta:
                second_best_delta = best_delta
                second_best_solution = best_solution.copy()
                best_delta = delta
                best_solution = solution.copy()

        if best_delta >= 0:
            # Revert to second best solution if no improvement is achieved
            improving = False
            solution = second_best_solution.copy()

    return solution



def main():
    time_results = []
    path_results = []

    prob = load_problem('kroA100.tsp')
    matrix = create_distance_matrix(prob)
    random_solution = generate_random_solution(matrix)
    dir_name = 'steepest_vertices_random'
    for n in range(100):
        start = time.time()
        random_solution = generate_random_solution(matrix)
        solution = steepest_vertex(random_solution, matrix)
        stop = time.time()
        time_results.append(stop - start)
        path_results.append(count_result(solution[0], matrix) + count_result(solution[1], matrix))
        draw_and_save_paths(prob, random_solution, dir_name, f'{dir_name}_{n}')
    print(f'Mean time: {np.mean(time_results)}')
    print(f'Mean path length: {np.mean(path_results)}')
    print(f'Best path length: {np.min(path_results)}')
    print(f'Worst path length: {np.max(path_results)}')
    print(f'Best iteration: {np.argmax(path_results)}')

    time_results = []
    path_results = []
    dir_name = 'random_walk'
    for n in range(100):
        start = time.time()
        random_solution = generate_random_solution(matrix)
        solution = steepest_vertex(random_solution, matrix)
        stop = time.time()
        time_results.append(stop - start)
        path_results.append(count_result(solution[0], matrix) + count_result(solution[1], matrix))
        draw_and_save_paths(prob, random_solution, dir_name, f'{dir_name}_{n}')

if __name__ == '__main__':
    main()
