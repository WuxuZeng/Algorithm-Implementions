# delta neighbourhood rough set
# process the numerical data(discrete&continuous data)


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


def generate_delta_neighborhood(universe, attributes, delta, distance, display_distance=False):
    """
    generate the delta neighborhoods of the universe
    :param universe: the universe of objects(feature vector/sample/instance)
    :param attributes: features' index
    :param delta: radius
    :param distance: the function to calculate the distance
    :param display_distance: default is Fault ,if is True, the distance will display
    :return: list, each element is a list and represent the delta_neighbourhood
    """
    elementary_sets = []
    for i in range(len(universe)):
        flag = True
        for elementary_single_set in elementary_sets:
            if in_delta_neighborhood(
                    universe, i, elementary_single_set[0], attributes, delta, distance, display=display_distance):
                elementary_single_set.append(i)
                flag = False
                break
        if flag:
            elementary_sets.append([i])
    return elementary_sets


def generate_delta_neighborhood_by_distance_matrix(distance_matrix, delta):
    """
    generate the delta neighborhoods of the universe
    :param distance_matrix: the distance matrix
    :param delta: radius
    :return: list, each element is a list and represent the delta_neighbourhood
    """
    elementary_sets = []
    return elementary_sets
