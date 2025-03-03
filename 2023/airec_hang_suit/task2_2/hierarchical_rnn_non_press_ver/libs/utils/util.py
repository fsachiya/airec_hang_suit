import math
import numpy as np
import itertools

def moving_average(data, size):
    b = np.ones(size)/size
    data_mean = np.convolve(data, b, mode="same")

    n_conv = math.ceil(size/2)

    # 補正部分
    data_mean[0] *= size/n_conv
    for i in range(1, n_conv):
        data_mean[i] *= size/(i+n_conv)
        data_mean[-i] *= size/(i + n_conv - (size % 2)) 
	# size%2は奇数偶数での違いに対応するため
    return data_mean

def calculate_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def total_distance(path, coordinates):
    distance = 0
    for i in range(len(path) - 1):
        distance += calculate_distance(coordinates[path[i]], coordinates[path[i + 1]])
    distance += calculate_distance(coordinates[path[-1]], coordinates[path[0]])
    return distance

def tsp_bruteforce(coordinates):
    n = len(coordinates)
    indices = list(range(n))
    shortest_path = None
    min_distance = float('inf')

    for path in itertools.permutations(indices):
        distance = total_distance(path, coordinates)
        if distance < min_distance:
            min_distance = distance
            shortest_path = path

    return shortest_path, min_distance