"""
traditional rough set
process the categorical data(nominal&ordinal) and discrete numerical data

partition
lower and upper approximations
positive, boundary and negative regions
"""


import pandas as pd
import numpy as np
import time


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


def partition_test_by_mushroom():
    """
    to test the partition and partition by mushroom
    calculate their time usage
    :return: None

    result:
    The time used: 37.04471778869629 seconds
    The time used: 0.01600813865661621 seconds
    The time used: 29.257173776626587 seconds
    The time used: 0.30783629417419434 seconds
    """
    start_time = time.time()
    data = pd.read_csv("mushroom.csv", header=None)
    labels = np.array(data.pop(0))
    attributes = np.array(data)
    result = partition(attributes, np.arange(attributes.shape[1]))
    print(len(result))
    print('The time used: {} seconds'.format(time.time() - start_time))
    start_time = time.time()
    result = partition(labels, np.arange(1))
    print(len(result))
    print('The time used: {} seconds'.format(time.time() - start_time))

    start_time = time.time()
    data = pd.read_csv("mushroom.csv", header=None)
    labels = np.array(data.pop(0))
    attributes = np.array(data)
    result = partition2(attributes, np.arange(attributes.shape[1]))
    print(len(result))
    print('The time used: {} seconds'.format(time.time() - start_time))
    start_time = time.time()
    result = partition2(labels, np.arange(1))
    print(len(result))
    print('The time used: {} seconds'.format(time.time() - start_time))
    return None


def partition_test():
    """
    check partition and partition2 result to confirm it's correct
    print the partition result
    :return: None

    result:
    The time used: 37.04471778869629 seconds
    The time used: 0.01600813865661621 seconds
    The time used: 29.257173776626587 seconds
    The time used: 0.30783629417419434 seconds
    """
    start_time = time.time()
    data = pd.read_csv("mushroom_little.csv", header=None)
    labels = np.array(data.pop(0))
    attributes = np.array(data)
    result = partition(attributes, np.arange(attributes.shape[1]))
    print(len(result))
    print(result)
    print('The time used: {} seconds'.format(time.time() - start_time))
    start_time = time.time()
    result = partition(labels, np.arange(1))
    print(len(result))
    print(result)
    print('The time used: {} seconds'.format(time.time() - start_time))

    start_time = time.time()
    data = pd.read_csv("mushroom_little.csv", header=None)
    labels = np.array(data.pop(0))
    attributes = np.array(data)
    result = partition2(attributes, np.arange(attributes.shape[1]))
    print(len(result))
    print(result)
    print('The time used: {} seconds'.format(time.time() - start_time))
    start_time = time.time()
    result = partition2(labels, np.arange(1))
    print(len(result))
    print(result)
    print('The time used: {} seconds'.format(time.time() - start_time))
    return None


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


def set_is_include_test():
    result = set_is_include([1, 2, 3], [[0, 1, 2, 3]])
    print(result)
    result = set_is_include([0, 1, 2, 3], [[1, 2, 3]])
    print(result)
    return


def feature_subset_low_approximations_of_sample_subset(
        universe, sample_subset, feature_subset, partition_function=partition):
    """
    get the feature_subset lower approximations of sample_subset
    :param universe: the universe of objects(feature vector/sample/instance)
    :param sample_subset: list, a set of objects' index
    :param feature_subset: features' index
    :param partition_function: the function to divide the sample_subset
    :return: list, lower_approximations is composed by a set of objects' index
    """
    lower_approximations = []
    partition_1 = partition_function(universe, feature_subset)
    for x in partition_1:
        if set_is_include(x, [sample_subset]):
            lower_approximations.extend(x)
    lower_approximations.sort()
    return lower_approximations


def features_lower_approximations_of_universe(universe, attributes, labels, partition_function=partition):
    """
    get the features lower approximations of U/R
    :param universe: the universe of objects(feature vector/sample/instance)
    :param attributes: features' index
    :param labels: labels' index
    :param partition_function: the function to divide the universe
    :return: list, lower_approximations is composed by a set of objects' index
    """
    lower_approximations = []
    partition_1 = partition_function(universe, attributes)
    partition_2 = partition_function(universe, labels)
    for x in partition_1:
        if set_is_include(x, partition_2):
            lower_approximations.extend(x)
    lower_approximations.sort()
    return lower_approximations


def features_lower_approximations_of_universe_test():
    """
    test features_lower_approximations_of_universe
    :return: None
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    result = features_lower_approximations_of_universe(np.array(data), np.arange(4), np.arange(4, 5), partition)
    print("approximation result:", result)
    print("\t\t\t\t\t", "[0, 1, 3, 4, 6, 7]")
    result = partition(np.array(data), np.arange(4))
    print("partition by attributes:", result)
    result = partition2(np.array(data), np.arange(4))
    print("partition2 by attributes:", result)
    result = partition(np.array(data), np.arange(4, 5))
    print("partition by label:", result)
    result = partition(np.array(data), np.arange(4, 5))
    print("partition2 by label:", result)
    return None


def feature_subset_low_approximations_of_sample_subset_test():
    """
    test feature_subset_low_approximations_of_sample_subset
    :return: None
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    print(data.shape)
    del data[4]
    print(data.shape)
    # result = feature_subset_low_approximations_of_sample_subset(
    # np.array(data), [i for i in range(8)], np.arange(4), partition)
    result = feature_subset_low_approximations_of_sample_subset(
        np.array(data), [1, 2, 3], [i for i in range(4)], partition)
    print("approximation result:", result)
    result = partition(np.array(data), np.arange(4))
    print("partition by attributes:", result)
    return None


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


def features_upper_approximations_of_universe_test():
    """
    test upper_features_lower_approximations_of_universe
    :return: None
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    result = features_upper_approximations_of_universe(np.array(data))
    print("result:", result)
    print(len(result))
    return None


def feature_subset_upper_approximations_of_sample_subset(
        universe, sample_subset, feature_subset, partition_function=partition):
    """
    get the feature_subset upper approximations of sample_subset
    :param universe: the universe of objects(feature vector/sample/instance)
    :param sample_subset: list, a set of objects' index
    :param feature_subset: features' index
    :param partition_function: the function to divide the sample_subset
    :return: list, upper_approximations is composed by a set of objects' index
    """
    upper_approximations = []
    partition_1 = partition_function(universe, feature_subset)
    for x in partition_1:
        if is_contain(x, sample_subset):
            upper_approximations.extend(x)
    upper_approximations.sort()
    return upper_approximations


def feature_subset_positive_region_of_sample_subset(
        universe, sample_subset, feature_subset, partition_function=partition):
    """
    get the feature_subset positive_region of sample_subset
    :param universe: the universe of objects(feature vector/sample/instance)
    :param sample_subset: list, a set of objects' index
    :param feature_subset: features' index
    :param partition_function: the function to divide the universe
    :return: list, positive_region is composed by a set of objects' index
    """
    positive_region = feature_subset_low_approximations_of_sample_subset(
        universe, sample_subset, feature_subset, partition_function)
    return positive_region


def feature_subset_positive_region_of_sample_subset_test():
    """
    test feature_subset_positive_region_of_sample_subset
    :return: None
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    del data[4]
    result = feature_subset_low_approximations_of_sample_subset(np.array(data), [0, 1, 4, 6, 7], [0, 3], partition)
    print("result:", result)
    print(len(result))
    return None


def feature_subset_boundary_region_of_sample_subset(
        universe, sample_subset, feature_subset, partition_function=partition):
    """
    get the feature_subset boundary_region of sample_subset
    :param universe: the universe of objects(feature vector/sample/instance)
    :param sample_subset: list, a set of objects' index
    :param feature_subset: features' index
    :param partition_function: the function to divide the universe
    :return: list, boundary_region is composed by a set of objects' index
    """
    upper_approximations = feature_subset_upper_approximations_of_sample_subset(
        universe, sample_subset, feature_subset, partition_function)
    lower_approximations = feature_subset_low_approximations_of_sample_subset(
        universe, sample_subset, feature_subset, partition_function)
    boundary_region = [i for i in upper_approximations if i not in lower_approximations]
    return boundary_region


def feature_subset_boundary_region_of_sample_subset_test():
    """
    test feature_subset_boundary_region_of_sample_subset
    :return: None
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    del data[4]
    result = feature_subset_boundary_region_of_sample_subset(np.array(data), [0, 1, 4, 6, 7], [0, 3], partition)
    print("result:", result)
    print(len(result))
    return None


def feature_subset_negative_region_of_sample_subset(
        universe, sample_subset, feature_subset, partition_function=partition):
    """
    get the feature_subset negative_region of sample_subset
    :param universe: the universe of objects(feature vector/sample/instance)
    :param sample_subset: list, a set of objects' index
    :param feature_subset: features' index
    :param partition_function: the function to divide the universe
    :return: list, negative_region is composed by a set of objects' index
    """
    upper_approximations = feature_subset_upper_approximations_of_sample_subset(
        universe, sample_subset, feature_subset, partition_function)
    return [i for i in np.arange(len(universe)) if i not in upper_approximations]


def feature_subset_negative_region_of_sample_subset_test():
    """
    test feature_subset_negative_region_of_sample_subset
    :return:
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    del data[4]
    result = feature_subset_negative_region_of_sample_subset(
        np.array(data), [0, 1, 4, 6, 7], [0, 3], partition)
    print("result:", result)
    print(len(result))
    return None


def feature_subset_upper_approximations_of_sample_subset_test():
    """
    test feature_subset_upper_approximations_of_sample_subset
    :return: None
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    print(data.shape)
    del data[4]
    print(data.shape)
    # result = feature_subset_low_approximations_of_sample_subset(
    # np.array(data), [i for i in range(8)], np.arange(4), partition)
    result = feature_subset_upper_approximations_of_sample_subset(
        np.array(data), [1, 2, 3, 4], [i for i in range(4)], partition)
    print("result:", result)
    return None


def dependency(
        universe, attributes, labels, attribute_partition_function=partition, label_partition_function=partition):
    """
    to calculate the dependency between attributes
    :param universe: the universe of objects(feature vector/sample/instance)
    :param attributes: list, a set of features' index
    :param labels: list, a set of features' index
    :param attribute_partition_function: the function to divide the universe by the attributes
    :param label_partition_function: the function to divide the universe by the labels
    :return: float number(0-->1, 1 represent that features_1 completely depends on features_2,
    All values of attributes from D are uniquely determined by the values of attributes from C.),
    the dependency of features_1 to features_2, POS_features_1(features_2)
    """
    partition_2 = label_partition_function(universe, labels)
    positive_region_size = 0
    for y in partition_2:
        positive_region_size += \
            len(feature_subset_positive_region_of_sample_subset(universe, y, attributes, attribute_partition_function))
        # print(feature_subset_positive_region_of_sample_subset(universe, y, features_1, attribute_partition_function))
    dependency_degree = positive_region_size/len(universe)
    return dependency_degree


def dependency_test():
    """
    test dependency
    :return: None
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    # result = feature_subset_low_approximations_of_sample_subset(
    # np.array(data), [i for i in range(8)], np.arange(4), partition)
    result = dependency(np.array(data), [0, 3], [4], partition, partition)
    print("dependency:", result)
    return None


def main():
    # print("lower approximations:\n")
    # features_lower_approximations_of_universe_test()
    # feature_subset_low_approximations_of_sample_subset_test()
    # print("\n", "upper approximations:\n")
    # features_upper_approximations_of_universe_test()
    # feature_subset_upper_approximations_of_sample_subset_test()
    # feature_subset_positive_region_of_sample_subset_test()
    # feature_subset_boundary_region_of_sample_subset_test()
    # feature_subset_negative_region_of_sample_subset_test()
    # dependency_test()
    set_is_include_test()
    pass


# confirm the function of the above function
if __name__ == '__main__':
    main()
