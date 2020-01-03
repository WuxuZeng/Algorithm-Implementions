"""
traditional rough set
process the categorical data(nominal&ordinal) and discrete numerical data

partition
lower and upper approximations
positive, boundary and negative regions
"""


import numpy as np


def is_indiscernible(universe, x, y, attributes):
    """
    if the two feature vector is indiscernible, return True, else return False
    Applies to attributes whose attribute values are discrete data
    :param universe: the universe of objects(feature vector/sample/instance)
    :param x: int, index of object
    :param y: the same as above
    :param attributes: list, a set of features' index
    :return: True/False
    """
    flag = True
    for attribute in attributes:
        if universe[x][attribute] == universe[y][attribute]:
            pass
        else:
            flag = False
            break
    return flag


# 伪代码
# 输入：样本集，特征
# 输出：基本集
# 1 基本集置空集
# 2 for x in 样本集
# 	flag = True
# 	for y in 基本集(y为等价类)
# 		如果 x 与y[0]相比为不可分辨关系
# 			将x加入该等价类
# 			Flag = False
# 			break
# 	if flag
# 		为 x 创建等价类加入到基本集中
def partition(universe, attributes):
    """
    calculate the partition of universe on attributes
    :param universe: the universe of objects(feature vector/sample/instance)
    :param attributes: list, a set of features' index
    :return: list, each element is a list and represent the equivalence class
    """
    elementary_sets = []
    for i in range(len(universe)):
        flag = True
        for elementary_single_set in elementary_sets:
            if is_indiscernible(universe, i, elementary_single_set[0], attributes):
                elementary_single_set.append(i)
                flag = False
                break
        if flag:
            elementary_sets.append([i])
    return elementary_sets


# for discrete data
# 对给出下标的样本进行划分
def part_partition(universe, samples, attributes):
    """
    calculate the partition of part universe on attributes
    :param universe: the universe of objects(feature vector/sample/instance)
    :param samples: the index of part samples
    :param attributes: features' index
    :return: list, each element is a list and represent the equivalence class
    """
    elementary_sets = []
    for i in samples:
        flag = True
        for elementary_single_set in elementary_sets:
            if is_indiscernible(universe, i, elementary_single_set[0], attributes):
                elementary_single_set.append(i)
                flag = False
                break
        if flag:
            elementary_sets.append([i])
    return elementary_sets


# partition实现方式2
# 伪代码
# 输入：样本集，特征
# 输出：基本集
# 1 基本集置空集
# 2 for x in 样本集
#     为 x 创建等价类
#     for y in 样本集-x
#         如果 y 和 x为不可分辨关系
#             将y加入到x的等价类中
# 	    将y从样本集中移除
def partition2(raw_universe, attributes):
    """
    Method 2 to calculate the partition of raw_universe on attributes
    :param raw_universe: the universe of objects(feature vector/sample/instance)
    :param attributes: features' index
    :return: list, each element is a list and represent the equivalence class
    """
    universe = list(np.arange(raw_universe.shape[0]))
    elementary_sets = []  # R-elementary sets
    i = 0
    while i < len(universe):
        IND_IS_R = [universe[i]]  # Equivalence class
        j = i + 1
        while j < len(universe):
            if is_indiscernible(raw_universe, universe[i], universe[j], attributes):
                IND_IS_R.append(universe[j])
                universe.remove(universe[j])
                j -= 1
            j += 1
        universe.remove(universe[i])
        elementary_sets.append(IND_IS_R)
    return elementary_sets


# unused
def element_is_include(element, element_set):
    """
    judge if the element is included by(belong to) the mylist2
    :param element: a object's index
    :param element_set: list, a set of objects' index
    :return: True/False
    """
    flag = True
    try:
        element_set.index(element)
    except ValueError:
        flag = False
    return flag


def set_is_include(set1, set2):
    """
    judge if the set1 is included by(belong to) the set2
    :param set1: a set of objects' index
    :param set2: list, a set of objects' index
    :return: True/False
    """
    for element in set2:
        flag = True
        for x1 in set1:
            try:
                element.index(x1)
            except ValueError:
                flag = False
                break
        if flag:
            return True
        else:
            continue
    return False


def feature_subset_low_approximations_of_sample_subset(universe, sample_subset, feature_subset):
    """
    get the feature_subset lower approximations of sample_subset
    :param universe: the universe of objects(feature vector/sample/instance)
    :param sample_subset: list, a set of objects' index
    :param feature_subset: features' index
    :return: list, lower_approximations is composed by a set of objects' index
    """
    lower_approximations = []
    partition_1 = partition(universe, feature_subset)
    for x in partition_1:
        if set_is_include(x, [sample_subset]):
            lower_approximations.extend(x)
    lower_approximations.sort()
    return lower_approximations


def features_lower_approximations_of_universe(universe, attributes, labels):
    """
    get the features lower approximations of U/R
    :param universe: the universe of objects(feature vector/sample/instance)
    :param attributes: features' index
    :param labels: labels' index
    :return: list, lower_approximations is composed by a set of objects' index
    """
    lower_approximations = []
    partition_1 = partition(universe, attributes)
    partition_2 = partition(universe, labels)
    for x in partition_1:
        if set_is_include(x, partition_2):
            lower_approximations.extend(x)
    lower_approximations.sort()
    return lower_approximations


def is_contain(x, y):
    """
    judge that the intersection of x and y is not an empty set
    :param x: the set of objects' index
    :param y: the set of objects' index
    :return: True/False
    """
    intersection = [i for i in x if i in y]
    if len(intersection) > 0:
        return True
    else:
        return False


def features_upper_approximations_of_universe(universe):
    """
    get the features upper approximations of U/R
    :param universe: the universe of objects(feature vector/sample/instance)
    :return: list, upper_approximations is composed by a set of objects' index
    """
    upper_approximations = list(np.arange(len(universe)))
    upper_approximations.sort()
    return upper_approximations


def feature_subset_upper_approximations_of_sample_subset(universe, sample_subset, feature_subset):
    """
    get the feature_subset upper approximations of sample_subset
    :param universe: the universe of objects(feature vector/sample/instance)
    :param sample_subset: list, a set of objects' index
    :param feature_subset: features' index
    :return: list, upper_approximations is composed by a set of objects' index
    """
    upper_approximations = []
    partition_1 = partition(universe, feature_subset)
    for x in partition_1:
        if is_contain(x, sample_subset):
            upper_approximations.extend(x)
    upper_approximations.sort()
    return upper_approximations


def feature_subset_positive_region_of_sample_subset(universe, sample_subset, feature_subset):
    """
    get the feature_subset positive_region of sample_subset
    :param universe: the universe of objects(feature vector/sample/instance)
    :param sample_subset: list, a set of objects' index
    :param feature_subset: features' index
    :return: list, positive_region is composed by a set of objects' index
    """
    positive_region = feature_subset_low_approximations_of_sample_subset(universe, sample_subset, feature_subset)
    return positive_region


def feature_subset_boundary_region_of_sample_subset(universe, sample_subset, feature_subset):
    """
    get the feature_subset boundary_region of sample_subset
    :param universe: the universe of objects(feature vector/sample/instance)
    :param sample_subset: list, a set of objects' index
    :param feature_subset: features' index
    :return: list, boundary_region is composed by a set of objects' index
    """
    upper_approximations = feature_subset_upper_approximations_of_sample_subset(universe, sample_subset, feature_subset)
    lower_approximations = feature_subset_low_approximations_of_sample_subset(universe, sample_subset, feature_subset)
    boundary_region = [i for i in upper_approximations if i not in lower_approximations]
    return boundary_region


def feature_subset_negative_region_of_sample_subset(universe, sample_subset, feature_subset):
    """
    get the feature_subset negative_region of sample_subset
    :param universe: the universe of objects(feature vector/sample/instance)
    :param sample_subset: list, a set of objects' index
    :param feature_subset: features' index
    :return: list, negative_region is composed by a set of objects' index
    """
    upper_approximations = feature_subset_upper_approximations_of_sample_subset(universe, sample_subset, feature_subset)
    return [i for i in np.arange(len(universe)) if i not in upper_approximations]


def dependency(universe, attributes, labels):
    """
    to calculate the dependency between attributes
    :param universe: the universe of objects(feature vector/sample/instance)
    :param attributes: list, a set of features' index
    :param labels: list, a set of features' index
    :return: float number(0-->1, 1 represent that features_1 completely depends on features_2,
    All values of attributes from D are uniquely determined by the values of attributes from C.),
    the dependency of features_1 to features_2, POS_features_1(features_2)
    """
    partition_2 = partition(universe, labels)
    positive_region_size = 0
    for y in partition_2:
        positive_region_size += \
            len(feature_subset_positive_region_of_sample_subset(universe, y, attributes))
        # print(feature_subset_positive_region_of_sample_subset(universe, y, features_1))
    dependency_degree = positive_region_size/len(universe)
    return dependency_degree