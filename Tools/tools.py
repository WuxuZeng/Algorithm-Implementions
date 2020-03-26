import math
import numpy as np
import pandas as pd


# for continuous data
def euclidean_distance(universe, x, y, features):
    """
    calculate the distance of two objects
    :param universe: the universe of objects(feature vector/sample/instance)
    :param x: feature vector, sample, instance
    :param y: feature vector, sample, instance
    :param features: list, a set of features' serial number
    :return: float, distance
    """
    total = 0
    for i in range(len(features)):
        total += (universe[x][features[i]] - universe[y][features[i]])**2
    return math.sqrt(total)


def euclidean_distance_test():
    data = pd.read_csv("ExampleData.csv", header=None)
    del data[4]
    result = euclidean_distance(np.array(data), 0, 1, [i for i in range(0, 4)])
    print("result:", result)
    return


# for continuous data
def standardized_euclidean_distance(universe, x, y, features):
    """
    calculate the standardized euclidean distance of two objects
    :param universe: the universe of objects(feature vector/sample/instance)
    :param x: feature vector, sample, instance
    :param y: feature vector, sample, instance
    :param features: list, a set of features' serial number
    :return: float, distance
    """
    total = 0
    standard_deviation = []
    for feature in features:
        standard_deviation.append(np.std(universe[:, feature], ddof=1))
    for i in range(len(features)):
        total += (((universe[x][features[i]] - universe[y][features[i]]) / standard_deviation[i]) ** 2)
    return math.sqrt(total)


def standardized_euclidean_distance_test():
    data = pd.read_csv("ExampleData.csv", header=None)
    del data[4]
    result = standardized_euclidean_distance(np.array(data), 0, 2, [0, 1])
    print("x1, x3:", result)
    result = standardized_euclidean_distance(np.array(data), 1, 2, [0, 1])
    print("x2, x3:", result)
    return


def main():
    # euclidean_distance_test()
    standardized_euclidean_distance_test()
    pass


if __name__ == '__main__':
    main()
