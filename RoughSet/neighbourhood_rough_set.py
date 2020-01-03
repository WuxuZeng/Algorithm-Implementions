# delta neighbourhood rough set
# process the numerical data(discrete&continuous data)


from RoughSet.traditional_rough_set import *
from tools import *


# for continuous data
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


def generate_delta_neighborhood(universe, attributes, delta, distance=euclidean_distance, display_distance=False):
    """
    generate the delta neighborhoods of the universe
    :param universe: the universe of objects(feature vector/sample/instance)
    :param attributes: features' index
    :param delta: radius
    :param distance: the function to calculate the distance
    :param display_distance: default is Fault ,if is True, the distance will display
    :return: list, each element is a list and represent the delta_neighbourhood
    """
    distance_matrix = generate_distance_matrix(universe, attributes, distance)
    elementary_sets = []
    for i in range(len(universe)):
        element_set = []
        for j in range(len(universe)):
            if distance_matrix[i][j] < delta:
                element_set.append(j)
        elementary_sets.append(element_set)
    return elementary_sets
