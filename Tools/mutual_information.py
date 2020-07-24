"""
entropy
mutual information
conditional mutual information(partial mutual information)
"""
import math
import numpy as np
import pandas as pd
from RoughSet.traditional_rough_set import partition, part_partition


def conditional_mutual_information(universe, feature_a, feature_b, feature_c):
    """
    calculate the conditional mutual information between a and b when the c is given. : I(a;b|c)
    :param universe: the universe of objects(feature vector/sample/instance)
    :param feature_a: list, features' index
    :param feature_b: list, features' index
    :param feature_c:
    :return:
    """
    return conditional_entropy(universe, feature_a, feature_c) - \
           conditional_entropy(universe, feature_a, [i for i in feature_b or feature_c])


def mutual_information(universe, feature_a, feature_b):
    """
    calculate the mutual information between a and b : I(a;b)
    :param universe: the universe of objects(feature vector/sample/instance)
    :param feature_a: list, features' index
    :param feature_b: list, features' index
    :return:
    """
    return entropy(universe, feature_a) - conditional_entropy(universe, feature_a, feature_b)


def joint_entropy(universe, feature_a, feature_b):
    """
    calculate the joint entropy of a and b, H(a,b)
    :param universe: the universe of objects(feature vector/sample/instance)
    :param feature_b: list, features' index
    :param feature_a: list, features' index
    :return:
    """
    partitions_a = partition(universe, feature_a)
    partitions_b = partition(universe, feature_b)
    total = 0
    for a in partitions_a:
        for b in partitions_b:
            a_b = [i for i in a if i in b]
            probability = len(a_b) / universe.shape[0]
            if probability > 0:
                total += probability * math.log2(probability)
    return -1 * total


def conditional_entropy(universe, feature_b, feature_a, display=False):
    """
    to calculate the conditional entropy, H(B|A) or H(B|A1,A2...)
    :param universe: the universe of objects(feature vector/sample/instance)
    :param feature_b: list, features' index
    :param feature_a: list, features' index
    :param display: to display the probability
    :return:
    """
    partitions_a = partition(universe, feature_a)
    partitions_b = partition(universe, feature_b)
    total = 0
    for a in partitions_a:
        inner_total = 0
        # length = len(a)
        for b in partitions_b:
            length = len(b)
            a_b = [i for i in a if i in b]
            probability = len(a_b) / length
            if display:
                print(probability)
            if probability > 0:
                inner_total += probability * math.log2(probability)
        total += inner_total * length / universe.shape[0]
    return -1 * total


def part_entropy(universe, samples, feature):
    """
    to calculate the entropy of feature in part universe, H(a)
    :param universe: the universe of objects(feature vector/sample/instance)
    :param samples: list, the index of samples
    :param feature: list, features' index
    :return:
    """
    partitions = part_partition(universe, samples, feature)
    total = 0
    length = len(samples)
    for yi in partitions:
        probability = len(yi)/length
        total += probability * math.log2(probability)
    return -1 * total


def entropy(universe, feature):
    """
    to calculate the entropy of feature, H(a)
    :param universe: the universe of objects(feature vector/sample/instance)
    :param feature: list, features' index
    :return:
    """
    partitions = partition(universe, feature)
    total = 0
    for yi in partitions:
        probability = len(yi)/universe.shape[0]
        total += probability * math.log2(probability)
    return -1 * total


def main():
    # entropy
    data = pd.read_csv("watermelon2.csv", header=None)
    result = entropy(np.array(data), [6])
    print(result)
    # part entropy
    partitions_ = partition(np.array(data), [0])
    print(partitions_)
    for part in partitions_:
        result = part_entropy(np.array(data), part, [6])
        print(result)
    # conditional entropy
    for feature in range(6):
        print(feature, "#")
        conditional_entropy(np.array(data), [feature], [6], True)
    conditional_entropy(np.array(data), [4], [6])
    # conditional mutual information(no example to check)
    # conditional_mutual_information(np.array(data), [], [], [])
    return


# confirm the function of the above function
if __name__ == '__main__':
    main()
