# delta neighborhood rough set
# process the numerical data(discrete&continuous data)


import heapq
import numpy as np
from RoughSet.traditional_rough_set import set_is_include, partition, is_contain

# for continuous data
from Tools.tools import euclidean_distance


def in_delta_neighborhood(universe, x, y, attributes, delta, distance, display=False):
    """
    if the sample y is the neighborhood of the sample x by the limitation of the radius, return True, else return False
    Applies to attributes whose attribute values are numerical data
    :param universe: the universe of objects(feature vector/sample/instance)
    :param x: int, index of object
    :param y: the same as above
    :param delta: the radius
    :param distance: the function to calculate the distance
    :param attributes: the feature(s)/attribute(s) of object
    :param display: default is Fault ,if is True, the distance will display
    :return: True/False
    """
    dis = distance(universe, x, y, attributes)
    if display:
        print(x, y, dis)
    if dis <= delta:
        return True
    else:
        return False


# for continuous data
def generate_distance_matrix(universe, attributes, distance=euclidean_distance, display_distance=False):
    """
    generate the distance triangle matrix
    :param universe: the universe of objects(feature vector/sample/instance)
    :param attributes: features' index
    :param distance: the method to calculate the distance
    :param display_distance: default is Fault ,if is True, the distance will display
    :return: the distance triangle matrix
    """
    matrix = np.triu(np.zeros(len(universe)**2).reshape(len(universe), len(universe)))
    for j in range(len(universe)):
        for k in range(j, len(universe)):
            matrix[j][k] = distance(universe, j, k, attributes)
    matrix += matrix.T - np.diag(matrix.diagonal())
    if display_distance:
        print(matrix)
    return matrix


def generate_delta_neighborhood(universe, attributes, delta, distance=euclidean_distance, sort=False):
    """
    generate the delta neighborhoods of the universe
    :param universe: the universe of objects(feature vector/sample/instance)
    :param attributes: features' index
    :param delta: radius
    :param distance: the function to calculate the distance
    :param sort: the result is sorted
    :return: list, each element is a list and represent the delta_neighborhood
    """
    distance_matrix = generate_distance_matrix(universe, attributes, distance)
    elementary_sets = []
    if sort:
        for i in range(len(universe)):
            element_set = []
            for j in range(len(universe)):
                if distance_matrix[i][j] < delta:
                    element_set.append(j)
            elementary_sets.append(element_set)
    else:
        for i in range(len(universe)):
            element_set = [i]
            for j in range(len(universe)):
                if j == i:
                    continue
                if distance_matrix[i][j] < delta:
                    element_set.append(j)
            elementary_sets.append(element_set)
    return elementary_sets


# for continuous data
def generate_k_nearest_neighborhood(universe, attributes, k, distance, display_distance=False):
    """
    generate the k nearest neighborhoods of the universe
    :param universe: the universe of objects(feature vector/sample/instance)
    :param attributes: features' index
    :param k: k
    :param distance: the method to calculate the distance
    :param display_distance: default is Fault ,if is True, the distance will display
    :return: list, the k_nearest_neighborhood(raw_universe/attributes)
    """
    distance = generate_distance_matrix(universe, attributes, distance, display_distance)
    universe_index = list(np.arange(universe.shape[0]))
    elementary_sets = []  # R-elementary sets
    j = 0
    while j < len(universe_index):
        k_nearest_index = \
            heapq.nsmallest(k + 1, range(len(distance[universe_index[j]])), distance[universe_index[j]].take)
        elementary_sets.append(k_nearest_index)
        j += 1
    return elementary_sets


def low_approximations_of_sample_subset_neighborhood(universe, sample_subset, feature_subset, delta):
    """
    get the feature_subset lower approximation of sample_subset
    :param universe: the universe of objects(feature vector/sample/instance)
    :param sample_subset: list, a set of objects' index
    :param feature_subset: features' index
    :param delta: radius
    :return: list, lower_approximations is composed by a set of objects' index
    """
    lower_approximations = []
    partition_1 = generate_delta_neighborhood(universe, feature_subset, delta)
    for x in partition_1:
        if set_is_include(x, [sample_subset]):
            lower_approximations.append(x[0])
    lower_approximations.sort()
    return lower_approximations


def lower_approximations_of_universe_neighborhood(universe, attributes, labels, delta):
    """
    get the features lower approximations of U/R
    :param universe: the universe of objects(feature vector/sample/instance)
    :param attributes: features' index
    :param labels: labels' index
    :param delta: radius
    :return: list, lower_approximations is composed by a set of objects' index
    """
    lower_approximations = []
    partition_1 = generate_delta_neighborhood(universe, attributes, delta)
    partition_2 = partition(universe, labels)
    for x in partition_1:
        if set_is_include(x, partition_2):
            lower_approximations.append(x[0])
    lower_approximations.sort()
    return lower_approximations


def upper_approximations_of_universe_neighborhood(universe):
    """
    get the features upper approximations of U/R
    :param universe: the universe of objects(feature vector/sample/instance)
    :return: list, upper_approximations is composed by a set of objects' index
    """
    upper_approximations = list(np.arange(len(universe)))
    return upper_approximations


def upper_approximations_of_sample_subset_neighborhood(universe, sample_subset, feature_subset, delta):
    """
    get the feature_subset upper approximations of sample_subset
    :param universe: the universe of objects(feature vector/sample/instance)
    :param sample_subset: list, a set of objects' index
    :param feature_subset: features' index
    :param delta: radius
    :return: list, upper_approximations is composed by a set of objects' index
    """
    upper_approximations = []
    partition_1 = generate_delta_neighborhood(universe, feature_subset, delta)
    for x in partition_1:
        if is_contain(x, sample_subset):
            upper_approximations.extend(x)
    upper_approximations.sort()
    return upper_approximations


def positive_region_of_sample_subset_neighborhood(universe, sample_subset, feature_subset, delta):
    """
    get the feature_subset positive_region of sample_subset
    :param universe: the universe of objects(feature vector/sample/instance)
    :param sample_subset: list, a set of objects' index
    :param feature_subset: features' index
    :param delta: radius
    :return: list, positive_region is composed by a set of objects' index
    """
    positive_region = low_approximations_of_sample_subset_neighborhood(
        universe, sample_subset, feature_subset, delta)
    return positive_region


def boundary_region_of_sample_subset_neighborhood(universe, sample_subset, feature_subset, delta):
    """
    get the feature_subset boundary_region of sample_subset
    :param universe: the universe of objects(feature vector/sample/instance)
    :param sample_subset: list, a set of objects' index
    :param feature_subset: features' index
    :param delta: radius
    :return: list, boundary_region is composed by a set of objects' index
    """
    upper_approximations = upper_approximations_of_sample_subset_neighborhood(
        universe, sample_subset, feature_subset, delta)
    lower_approximations = low_approximations_of_sample_subset_neighborhood(
        universe, sample_subset, feature_subset, delta)
    boundary_region = [i for i in upper_approximations if i not in lower_approximations]
    return boundary_region


def negative_region_of_sample_subset_neighborhood(universe, sample_subset, feature_subset, delta):
    """
    get the feature_subset negative_region of sample_subset
    :param universe: the universe of objects(feature vector/sample/instance)
    :param sample_subset: list, a set of objects' index
    :param feature_subset: features' index
    :param delta: radius
    :return: list, negative_region is composed by a set of objects' index
    """
    upper_approximations = upper_approximations_of_sample_subset_neighborhood(
        universe, sample_subset, feature_subset, delta)
    return [i for i in np.arange(len(universe)) if i not in upper_approximations]


def dependency_neighborhood(universe, attributes, labels, delta):
    """
    to calculate the dependency between attributes
    :param universe: the universe of objects(feature vector/sample/instance)
    :param attributes: list, a set of features' index
    :param labels: list, a set of features' index
    :param delta: radius
    :return: float number(0-->1, 1 represent that features_1 completely depends on features_2,
    All values of attributes from D are uniquely determined by the values of attributes from C.),
    the dependency of features_1 to features_2, POS_features_1(features_2)
    """
    partition_2 = generate_delta_neighborhood(universe, labels, delta, euclidean_distance)
    positive_region_size = 0
    for y in partition_2:
        positive_region_size += \
            len(positive_region_of_sample_subset_neighborhood(universe, y, attributes, delta))
    dependency_degree = positive_region_size/len(universe)
    return dependency_degree
