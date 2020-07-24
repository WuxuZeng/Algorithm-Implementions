import time
import pandas as pd
from RoughSet.traditional_rough_set import *


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
    result = partition_old(attributes, np.arange(attributes.shape[1]), is_indiscernible_discrete)
    print(len(result))
    print('The time used: {} seconds'.format(time.time() - start_time))
    # start_time = time.time()
    # result = partition_(labels, np.arange(1), is_indiscernible_discrete)
    # print(len(result))
    # print('The time used: {} seconds'.format(time.time() - start_time))

    # start_time = time.time()
    # data = pd.read_csv("mushroom.csv", header=None)
    # labels = np.array(data.pop(0))
    # attributes = np.array(data)
    # result = partition2(attributes, np.arange(attributes.shape[1]))
    # print(len(result))
    # print('The time used: {} seconds'.format(time.time() - start_time))
    # start_time = time.time()
    # result = partition2(labels, np.arange(1))
    # print(len(result))
    # print('The time used: {} seconds'.format(time.time() - start_time))
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
    # start_time = time.time()
    # data = pd.read_csv("mushroom_little.csv", header=None)
    # labels = np.array(data.pop(0))
    # attributes = np.array(data)
    # result = partition(attributes, np.arange(attributes.shape[1]))
    # print(len(result))
    # print(result)
    # print('The time used: {} seconds'.format(time.time() - start_time))
    # start_time = time.time()
    # result = partition(labels, np.arange(1))
    # print(len(result))
    # print(result)
    # print('The time used: {} seconds'.format(time.time() - start_time))

    # start_time = time.time()
    # data = pd.read_csv("mushroom_little.csv", header=None)
    # labels = np.array(data.pop(0))
    # attributes = np.array(data)
    # result = partition2(attributes, np.arange(attributes.shape[1]))
    # print(len(result))
    # print(result)
    # print('The time used: {} seconds'.format(time.time() - start_time))
    # start_time = time.time()
    # result = partition2(labels, np.arange(1))
    # print(len(result))
    # print(result)
    # print('The time used: {} seconds'.format(time.time() - start_time))

    # check partition with two parameter which achieve by group by
    start_time = time.time()
    data = pd.read_csv("mushroom.csv", header=None)
    labels = np.array(data.pop(0))
    attributes = np.array(data)
    # result = partition(attributes, [i for i in range(attributes.shape[1])])
    result = partition(attributes, [i for i in range(4)])
    print(len(result))
    # print(result)
    print('The time used: {} seconds'.format(time.time() - start_time))
    start_time = time.time()
    # result = partition(labels, np.arange(1))
    # print(len(result))
    # print('The time used: {} seconds'.format(time.time() - start_time))
    return None


def set_is_include_test():
    result = set_is_include([1, 2, 3], [[0, 1, 2, 3]])
    print(result)
    result = set_is_include([0, 1, 2, 3], [[1, 2, 3]])
    print(result)
    return


def features_lower_approximations_of_universe_test():
    """
    test features_lower_approximations_of_universe
    :return: None
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    result = lower_approximations_of_universe(np.array(data), np.arange(4), np.arange(4, 5))
    print("approximation result:", result)
    print("\t\t\t\t\t", "[0, 1, 3, 4, 6, 7]")
    result = partition(np.array(data), np.arange(4))
    print("partition by attributes:", result)
    # result = partition2(np.array(data), np.arange(4))
    # print("partition2 by attributes:", result)
    result = partition(np.array(data), np.arange(4, 5))
    print("partition by label:", result)
    result = partition(np.array(data), np.arange(4, 5))
    print("partition2 by label:", result)
    return None


# def feature_subset_low_approximations_of_sample_subset_test():
#     """
#     test feature_subset_low_approximations_of_sample_subset
#     :return: None
#     """
#     data = pd.read_csv("approximation_data.csv", header=None)
#     print(data.shape)
#     del data[4]
#     print(data.shape)
#     # result = feature_subset_low_approximations_of_sample_subset(np.array(data), [i for i in range(8)], np.arange(4))
#     result = feature_subset_low_approximations_of_sample_subset(np.array(data), [1, 2, 3], [i for i in range(4)])
#     print("approximation result:", result)
#     result = partition(np.array(data), np.arange(4))
#     print("partition by attributes:", result)
#     return None


def features_upper_approximations_of_universe_test():
    """
    test upper_features_lower_approximations_of_universe
    :return: None
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    result = upper_approximations_of_universe(np.array(data))
    print("result:", result)
    print(len(result))
    return None


# def feature_subset_positive_region_of_sample_subset_test():
#     """
#     test feature_subset_positive_region_of_sample_subset
#     :return: None
#     """
#     data = pd.read_csv("approximation_data.csv", header=None)
#     del data[4]
#     result = feature_subset_low_approximations_of_sample_subset(np.array(data), [0, 1, 4, 6, 7], [0, 3])
#     print("result:", result)
#     print(len(result))
#     return None


def feature_subset_boundary_region_of_sample_subset_test():
    """
    test feature_subset_boundary_region_of_sample_subset
    :return: None
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    del data[4]
    result = boundary_region_of_sample_subset(np.array(data), [0, 1, 4, 6, 7], [0, 3])
    print("result:", result)
    print(len(result))
    return None


def feature_subset_negative_region_of_sample_subset_test():
    """
    test feature_subset_negative_region_of_sample_subset
    :return:
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    del data[4]
    result = negative_region_of_sample_subset(
        np.array(data), [0, 1, 4, 6, 7], [0, 3])
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
    # result = feature_subset_low_approximations_of_sample_subset(np.array(data), [i for i in range(8)], np.arange(4))
    result = upper_approximations_of_sample_subset(np.array(data), [1, 2, 3, 4], [i for i in range(4)])
    print("result:", result)
    return None


def dependency_test():
    """
    test dependency
    :return: None
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    # result = feature_subset_low_approximations_of_sample_subset(np.array(data), [i for i in range(8)], np.arange(4))
    result = dependency(np.array(data), [0, 3], [4])
    print("dependency:", result)
    return None


def main():
    partition_test_by_mushroom()
    partition_test()
    # print("lower approximations:\n")
    # features_lower_approximations_of_universe_test()
    # # feature_subset_low_approximations_of_sample_subset_test()
    # print("\n", "upper approximations:\n")
    # features_upper_approximations_of_universe_test()
    # feature_subset_upper_approximations_of_sample_subset_test()
    # # feature_subset_positive_region_of_sample_subset_test()
    # feature_subset_boundary_region_of_sample_subset_test()
    # feature_subset_negative_region_of_sample_subset_test()
    # dependency_test()
    # set_is_include_test()
    pass


if __name__ == '__main__':
    main()
