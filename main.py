from random import choice, randint
import matplotlib.pyplot as plt
from tsplib95 import load
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


def two_regret(matrix):
    unique_nodes = [i for i in range(len(matrix))]
    solution = []

    for _ in range(2):

        start = choice(unique_nodes)
        unique_nodes.remove(start)
        graph_path = [start] * 2
        total_length = 0
        
        while len(graph_path) < (len(matrix) // 2) + 1:
            regrets = []
            for vertex in unique_nodes:
                considered = sorted(((total_length - matrix[graph_path[i - 1]][graph_path[i]] +
                                    matrix[graph_path[i - 1]][vertex] + matrix[vertex][graph_path[i]],
                                    i) for i in range(1, len(graph_path))), key=lambda x: x[0])
                if len(considered) >= 2:
                    regret = considered[1][0] - considered[0][0]
                else:
                    regret = -considered[0][0]
                length, best_i = considered[0]
                regrets.append((regret, vertex, best_i, length))
            temp = max(regrets, key=lambda x: x[0])
            best_vertex, best_i, total_length = temp[1], temp[2], temp[3]
            graph_path = graph_path[:best_i] + [best_vertex] + graph_path[best_i:]
            unique_nodes.remove(best_vertex)
        solution.append(graph_path)

    return solution


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


def delta_for_inner_edges(matrix, solution, i, j, l_or_r):
    # object function
    return - matrix[solution[l_or_r][i-1]][solution[l_or_r][i]] \
           - matrix[solution[l_or_r][j]][solution[l_or_r][j+1]] \
           + matrix[solution[l_or_r][i-1]][solution[l_or_r][j]] \
           + matrix[solution[l_or_r][i]][solution[l_or_r][j+1]]


def switch_inner_vertices(solution, left_or_right, vertices):
    solution[left_or_right][vertices[0]], solution[left_or_right][vertices[1]] = solution[left_or_right][vertices[1]], solution[left_or_right][vertices[0]]


def switch_outer_vertices(solution, left_or_right, vertices):
    solution[left_or_right][vertices[0]], solution[abs(left_or_right-1)][vertices[1]] = solution[abs(left_or_right-1)][vertices[1]], solution[left_or_right][vertices[0]]


def switch_inner_edges(solution, left_or_right, vertices):
    solution[left_or_right][vertices[0]:vertices[1]+1] = [x for x in solution[left_or_right][vertices[0]:vertices[1]+1][::-1]]


def steepest(solution, matrix, delta_function, replace_function):
    improving = True
    left_or_right = 0 # 0 - left, 1 - right
    while improving:
        delta_inner = delta_outer = 0
        vertices_inner = vertices_outer = [None, None]

        # inner vertices
        for i in range(1, len(solution[left_or_right])-3):
            for j in range(i+2, len(solution[left_or_right])-1):
                delta = round(delta_function(matrix, solution, i, j, left_or_right), 2)
                if delta < delta_inner:
                    vertices_inner = [solution[left_or_right].index(solution[left_or_right][i]), solution[left_or_right].index(solution[left_or_right][j])]
                    delta_inner = delta

        # outer vertices
        for i in range(1, len(solution[left_or_right])-1):
            for j in range(1, len(solution[abs(left_or_right-1)])-1):
                delta = round(delta_for_outer_vertices(matrix, solution, i, j, left_or_right), 2)
                if delta < delta_outer:
                    vertices_outer = [solution[left_or_right].index(solution[left_or_right][i]), solution[abs(left_or_right-1)].index(solution[abs(left_or_right-1)][j])]
                    delta_outer = delta
        
        if vertices_inner != [None, None] or vertices_outer != [None, None]:
            if delta_inner < delta_outer:
                replace_function(solution, left_or_right, vertices_inner)
            else:
                switch_outer_vertices(solution, left_or_right, vertices_outer)
        else:
            improving = False

        left_or_right = abs(left_or_right-1)

    return solution


def greedy(solution, matrix, delta_function, replace_function):
    left_or_right = 0
    improving = True
    i = 1
    while improving:
        improving = False

        # randomizing
        slide = randint(1, len(solution[left_or_right])-2)
        solution[left_or_right] = solution[left_or_right][slide:] + solution[left_or_right][:slide]

        for i in range(1, len(solution[left_or_right])-3,):
            for j in range(i+2, len(solution[left_or_right])-1):
                delta = round(delta_function(matrix, solution, i, j, left_or_right), 2)
                if delta < 0:
                    vertices = [solution[left_or_right].index(solution[left_or_right][i]), solution[left_or_right].index(solution[left_or_right][j])]
                    replace_function(solution, left_or_right, vertices)
                    improving = True
                    break
            if improving:
                break
        if improving:
            continue

        for i in range(1, len(solution[left_or_right])-1):
            for j in range(1, len(solution[abs(left_or_right-1)])-1):
                    
                delta = round(delta_for_outer_vertices(matrix, solution, i, j, left_or_right), 2)
                if delta < 0:
                    vertices = [solution[left_or_right].index(solution[left_or_right][i]), solution[abs(left_or_right-1)].index(solution[abs(left_or_right-1)][j])]
                    switch_outer_vertices(solution, left_or_right, vertices)
                    improving = True
                    break
            if improving:
                break
        
        solution[left_or_right] = solution[left_or_right][:slide] + solution[left_or_right][slide:]
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
    type_of_neighborhood = [0,1,2]  # 0 - inner vertices, 1 - inner edges, 2 - outer vertices
    start = time.time()
    stop = start
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
            vertices = [choice(solution[left_or_right][1:-2]), choice(solution[left_or_right][1:-2])]
            i = solution[left_or_right].index(vertices[0])
            j = solution[solution[left_or_right].index(vertices[1]) + 1]

            if delta_for_inner_edges(matrix, solution, i, j, left_or_right) < 0:
                switch_inner_edges(solution, left_or_right, [i, j])
        #    pass
        else:
            # outer vertices
            vertices = [choice(solution[left_or_right][1:-1]), choice(solution[abs(left_or_right-1)][1:-1])]
            i = solution[left_or_right].index(vertices[0])
            j = solution[abs(left_or_right-1)].index(vertices[1])

            if delta_for_outer_vertices(matrix, solution, i, j, left_or_right) < 0:
                switch_outer_vertices(solution, left_or_right, [i, j])
        stop = time.time()

    return solution


def save_results_to_file(filepath, results):
    with open(filepath, 'w') as file:
        for k, v in results.items():
            file.write(f'{k}: {v}\n')


def main():
    files = ['kroA100.tsp', 'kroB100.tsp']

    for prob, prob_name in zip(files, ['kroA100', 'kroB100']):
        prob = load_problem(prob)
        matrix = create_distance_matrix(prob)

        for cycle, cycle_name in zip([generate_random_solution, two_regret], ['random', 'two_regret']):
                for alg, alg_name in zip([greedy, steepest], ['greedy', 'steepest']):
                    for neighborhood, neighborhood_name in zip([[delta_for_inner_vertices, switch_inner_vertices], [delta_for_inner_edges, switch_inner_edges]], ['vertices', 'edges']):
                        dir_name = f'{prob_name}_{cycle_name}_{alg_name}_{neighborhood_name}'
                        print(dir_name)
                        time_results = []
                        path_results = []
                        for n in range(100):
                            start = time.time()
                            solution = cycle(matrix)
                            solution = alg(solution, matrix, neighborhood[0], neighborhood[1])
                            stop = time.time()
                            time_results.append(stop - start)
                            path_results.append(count_result(solution[0], matrix) + count_result(solution[1], matrix))
                            draw_and_save_paths(prob, solution, dir_name, f'{n}')

                        results = {
                            'Mean time': np.mean(time_results),
                            'Mean path length': np.mean(path_results),
                            'Best path length': np.min(path_results),
                            'Worst path length': np.max(path_results),
                            'Best iteration': np.argmax(path_results)
                        }

                        save_results_to_file(f'{dir_name}\\results.txt', results)


if __name__ == '__main__':
    main()


