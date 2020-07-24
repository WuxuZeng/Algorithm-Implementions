from RoughSet.neighbourhood_rough_set import *
from neighborhood_rough_set_ct import dependency_neighborhood_neighborhood
import pandas as pd
import time


def generate_delta_neighborhood_test():
    data = pd.read_csv("ExampleData.csv", header=None)
    del data[4]
    result = generate_delta_neighborhood(np.array(data), [0, 1, 2, 3], 33, euclidean_distance)
    print("result:", result)
    # result: [[0], [1, 4], [2, 3, 6], [5, 7]]
    return


# for continuous data
def generate_distance_triangle_matrix_test():
    data = pd.read_csv("ExampleData.csv", header=None)
    del data[4]
    result = generate_distance_matrix(np.array(data), [0, 1, 2, 3], euclidean_distance)
    print("result:", result[0])
    k_nearest_index = heapq.nsmallest(3, range(len(result[0])), result[0].take)
    print(k_nearest_index)
    print(type(k_nearest_index))
    print(k_nearest_index[1:])
    print(k_nearest_index.pop(0))
    print(k_nearest_index.pop(0))
    print(k_nearest_index.pop(0))
    print(k_nearest_index.pop(0))
    return


def generate_k_nearest_neighborhood_test():
    data = pd.read_csv("ExampleData.csv", header=None)
    del data[4]
    result = generate_k_nearest_neighborhood(
        np.array(data), [0, 1, 2, 3], 2, euclidean_distance)
    print(result)
    return


def dependency_test():
    data = np.array(pd.read_csv("arcene_train.csv", delimiter=','))
    start = time.time()
    temp = start
    dependency_neighborhood(data, [i for i in range(data.shape[1] - 1)], [data.shape[1] - 1], 0.2)
    print("{}  ".format(time.time() - temp), "{}".format(time.time() - start))
    temp = time.time()
    dependency_neighborhood_neighborhood(data, [i for i in range(data.shape[1] - 1)], [data.shape[1] - 1], 0.2)
    print("{}  ".format(time.time() - temp), "{}".format(time.time() - start))


def main():
    # generate_delta_neighborhood_test()
    # generate_k_nearest_neighborhood_test()
    dependency_test()
    pass


if __name__ == '__main__':
    main()
