"""
Online streaming feature selection using adapted Neighborhood Rough Set
author: Peng Zhou, Xuegang Hu, Peipei Li, Xindong Wu
https://doi.org/10.1016/j.ins.2018.12.074

Conference
A New Online Feature Selection Method Using Neighborhood Rough Set
https://doi.org/10.1109/ICBK.2017.41
"""


import copy
import heapq
import random
from RoughSet.traditional_rough_set import *
from tools import *
from itertools import combinations


def generate_delta_neighborhood_test():
    data = pd.read_csv("ExampleData.csv", header=None)
    del data[4]
    result = generate_delta_neighborhood(np.array(data), [0, 1, 2, 3], 33, euclidean_distance)
    print("result:", result)
    # result: [[0], [1, 4], [2, 3, 6], [5, 7]]
    return


# for continuous data
def generate_distance_matrix(universe, attributes, distance, display_distance=False):
    """
    generate the distance triangle matrix
    :param universe: the universe of objects(feature vector/sample/instance)
    :param attributes: features' index
    :param distance: the method to calculate the distance
    :param display_distance: default is Fault ,if is True, the distance will display
    :return: the distance triangle matrix
    """
    matrix = np.triu(np.zeros(len(universe)**2).reshape(len(universe), len(universe)))
    for i in range(len(universe)):
        for j in range(i, len(universe)):
            matrix[i][j] = distance(universe, i, j, attributes)
    matrix += matrix.T - np.diag(matrix.diagonal())
    if display_distance:
        print(matrix)
    return matrix


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
    i = 0
    while i < len(universe_index):
        k_nearest_index = \
            heapq.nsmallest(k + 1, range(len(distance[universe_index[i]])), distance[universe_index[i]].take)
        elementary_sets.append(k_nearest_index)
        i += 1
    return elementary_sets


def generate_k_nearest_neighborhood_test():
    data = pd.read_csv("ExampleData.csv", header=None)
    del data[4]
    result = generate_k_nearest_neighborhood(
        np.array(data), [0, 1, 2, 3], 2, euclidean_distance)
    print(result)
    return


def generate_gap_neighborhood(
        universe, attributes, gap_weight=1., distance=standardized_euclidean_distance, display_distance=False):
    """
    generate the gap neighborhoods of the universe
    :param universe: the universe of objects(feature vector/sample/instance)
    :param attributes: features' index
    :param gap_weight: the weight of gap
    :param distance: the method to calculate the distance
    :param display_distance: default is Fault ,if is True, the distance will display
    :return: list, the k_nearest_neighborhood(raw_universe/attributes)
    """
    distance_matrix = generate_distance_matrix(universe, attributes, distance, display_distance)
    universe_index = list(np.arange(universe.shape[0]))
    elementary_sets = []  # R-elementary sets
    i = 0
    while i < len(universe_index):
        # get the sorted distance of index
        distance_sort = heapq.nsmallest(universe.shape[0], range(len(distance_matrix[universe_index[i]])),
                                        distance_matrix[universe_index[i]].take)

        # calculate the max, min, and mean of the distance
        distance_max = distance_matrix[universe_index[i]][distance_sort[-1]]
        distance_min = distance_matrix[universe_index[i]][distance_sort[1]]
        # the formulate in the paper is G_mean = (D_max-D_min)/(n-1)
        # the source code imply with Matlab by the author is G_mean = (D_max-D_min)/(n-2)
        distance_mean = (distance_max - distance_min) / (universe.shape[0] - 1)
        gap = distance_mean * gap_weight
        j = 2

        # find the gap
        while j < len(distance_sort):
            if distance_matrix[universe_index[i]][distance_sort[j]] - \
                    distance_matrix[universe_index[i]][distance_sort[j - 1]] >= gap:
                break
            j += 1

        # put the xi to the front
        index = distance_sort.index(i)
        if index != 0:
            distance_sort.remove(i)
            distance_sort.insert(index, distance_sort[0])
            distance_sort.remove(distance_sort[0])
            distance_sort.insert(0, i)

        elementary_sets.append(distance_sort[:j])
        i += 1

    return elementary_sets


def generate_gap_neighborhood_test():
    data = pd.read_csv("ExampleData.csv", header=None)
    del data[4]
    result = generate_gap_neighborhood(
        np.array(data), [0], 1.5, standardized_euclidean_distance)
    print(result)
    return


class OnlineFeatureSelectionAdapted3Max:
    def __init__(self, universe, conditional_features, decision_features, gap_weight, distance):
        self.universe = universe
        self.conditional_features = conditional_features
        self.decision_features = decision_features
        self.gap_weight = gap_weight
        self.distance = distance
        return

    def dep_adapted(self, attributes):
        s_card = 0
        gap_neighborhoods = \
            generate_gap_neighborhood(self.universe, attributes, gap_weight=self.gap_weight, distance=self.distance)
        partitions = partition(self.universe, self.decision_features)

        # 用正域去理解
        # result: [3]
        # for gap_neighborhood in gap_neighborhoods:
        #     if set_is_include(gap_neighborhood, partitions):
        #         s_card += 1
        # d_s = s_card / self.universe.shape[0]

        # for gap_neighborhood in gap_neighborhoods:
        #     if set_is_include(gap_neighborhood, partitions):
        #         s_card += 1
        # d_s = s_card / len(attributes)

        # 样本本身不包含进邻域,s_card计算的是相似度，同样本标签一致的对象为positive sample，f1相似度正确
        # result: [0, 3]
        # for gap_neighborhood in gap_neighborhoods:
        #     for single_partition in partitions:
        #         if element_is_include(gap_neighborhood[0], single_partition):
        #             s_card += \
        #                 (len([i for i in gap_neighborhood if i in single_partition])-1) / (len(gap_neighborhood)-1)
        # d_s = s_card / self.universe.shape[0]

        # for gap_neighborhood in gap_neighborhoods:
        #     for single_partition in partitions:
        #         if element_is_include(gap_neighborhood[0], single_partition):
        #             s_card += \
        #                 (len([i for i in gap_neighborhood if i in single_partition])-1) / (len(gap_neighborhood)-1)
        # d_s = s_card / len(attributes)

        # 样本本身包含进邻域,s_card计算的是相似度，同样本标签一致的对象为positive sample
        # 根据作者源代码，确定是这种形式
        # 邻域中同目标样本一致的标签的样本数（排除目标样本）除以邻域中样本总数（不排除目标样本）
        # result: [0, 3]
        for gap_neighborhood in gap_neighborhoods:
            for single_partition in partitions:
                if element_is_include(gap_neighborhood[0], single_partition):
                    s_card += \
                        (len([i for i in gap_neighborhood if i in single_partition]) - 1) / (len(gap_neighborhood))
        d_s = s_card / self.universe.shape[0]

        # for gap_neighborhood in gap_neighborhoods:
        #     for single_partition in partitions:
        #         if element_is_include(gap_neighborhood[0], single_partition):
        #             s_card += \
        #                 (len([i for i in gap_neighborhood if i in single_partition])) / (len(gap_neighborhood))
        # d_s = s_card / len(attributes)

        # 样本本身包含进邻域,s_card计算的是相似度，标签为1的对象为positive sample
        # result: [0, 2]
        # for gap_neighborhood in gap_neighborhoods:
        #     if element_is_include(gap_neighborhood[0], partitions[1]):
        #         s_card += (len([i for i in gap_neighborhood if i in partitions[1]]) - 1) / (len(gap_neighborhood) - 1)
        #     if element_is_include(gap_neighborhood[0], partitions[0]):
        #         s_card += (len([i for i in gap_neighborhood if i in partitions[1]])) / (len(gap_neighborhood) - 1)
        # d_s = s_card / self.universe.shape[0]

        # for gap_neighborhood in gap_neighborhoods:
        #     if element_is_include(gap_neighborhood[0], partitions[1]):
        #         s_card += (len([i for i in gap_neighborhood if i in partitions[1]]) - 1) / (len(gap_neighborhood) - 1)
        #     if element_is_include(gap_neighborhood[0], partitions[0]):
        #         s_card += (len([i for i in gap_neighborhood if i in partitions[1]])) / (len(gap_neighborhood) - 1)
        # d_s = s_card / len(attributes)

        # 样本本身不包含进邻域,s_card计算的是相似度，标签为1的对象为positive sample
        # result: [0, 3]
        # for gap_neighborhood in gap_neighborhoods:
        #     if element_is_include(gap_neighborhood[0], partitions[1]):
        #         s_card += (len([i for i in gap_neighborhood if i in partitions[1]])) / (len(gap_neighborhood))
        #     if element_is_include(gap_neighborhood[0], partitions[0]):
        #         s_card += (len([i for i in gap_neighborhood if i in partitions[0]])) / (len(gap_neighborhood))
        # d_s = s_card / self.universe.shape[0]

        # for gap_neighborhood in gap_neighborhoods:
        #     if element_is_include(gap_neighborhood[0], partitions[1]):
        #         s_card += (len([i for i in gap_neighborhood if i in partitions[1]])) / (len(gap_neighborhood))
        #     if element_is_include(gap_neighborhood[0], partitions[0]):
        #         s_card += (len([i for i in gap_neighborhood if i in partitions[0]])) / (len(gap_neighborhood))
        # d_s = s_card / len(attributes)

        # result: [0, 3]
        # for gap_neighborhood in gap_neighborhoods:
        #     if element_is_include(gap_neighborhood[0], partitions[1]):
        #         s_card += (len([i for i in gap_neighborhood if i in partitions[1]]) - 1) / (len(gap_neighborhood) - 1)
        #     if element_is_include(gap_neighborhood[0], partitions[0]):
        #         s_card += (len([i for i in gap_neighborhood if i in partitions[0]]) - 1) / (len(gap_neighborhood) - 1)
        # d_s = s_card / self.universe.shape[0]

        # for gap_neighborhood in gap_neighborhoods:
        #     if element_is_include(gap_neighborhood[0], partitions[1]):
        #         s_card += (len([i for i in gap_neighborhood if i in partitions[1]]) - 1) / (len(gap_neighborhood) - 1)
        #     if element_is_include(gap_neighborhood[0], partitions[0]):
        #         s_card += (len([i for i in gap_neighborhood if i in partitions[0]]) - 1) / (len(gap_neighborhood) - 1)
        # d_s = s_card / len(attributes)

        return d_s

    def get_new_feature(self):
        """
        to transfer a feature to the algorithm through yield
        :return: None
        """
        conditional_features = self.conditional_features
        for i in range(len(conditional_features)):
            yield conditional_features[i]
        return None

    def run(self):
        candidate_features = []
        candidate_dependency = 0
        mean_dependency_of_candidate = 0
        for feature in self.get_new_feature():
            feature_dependency = self.dep_adapted([feature])
            if feature_dependency < mean_dependency_of_candidate:
                continue
            temp_candidate_features = copy.deepcopy(candidate_features)
            temp_candidate_features.append(feature)
            temp_candidate_dependency = self.dep_adapted(temp_candidate_features)
            if temp_candidate_dependency > candidate_dependency:
                candidate_features = temp_candidate_features
                candidate_dependency = temp_candidate_dependency
                mean_dependency_of_candidate = \
                    ((mean_dependency_of_candidate * (len(candidate_features) - 1)) + feature_dependency) / \
                    len(candidate_features)
            elif temp_candidate_dependency == candidate_dependency:
                if candidate_dependency == 0 and len(candidate_features) == 0:
                    continue
                candidate_features.append(feature)
                random.shuffle(candidate_features)
                for test_feature in candidate_features:
                    temp_candidate_features = copy.deepcopy(candidate_features)
                    temp_candidate_features.remove(test_feature)
                    temp_candidate_features_dependency = self.dep_adapted(temp_candidate_features)
                    if (self.dep_adapted(candidate_features) - temp_candidate_features_dependency) == 0:
                        candidate_features = temp_candidate_features
                        test_feature_dependency = self.dep_adapted([test_feature])
                        if mean_dependency_of_candidate > 0:
                            mean_dependency_of_candidate = \
                                (mean_dependency_of_candidate*(len(candidate_features)+1) - test_feature_dependency) /\
                                len(candidate_features)
                    pass
            pass
        return candidate_features


def dep_adapted_test():
    data = pd.read_csv("ExampleData.csv", header=None)
    algorithm = OnlineFeatureSelectionAdapted3Max(np.array(data), [0, 1, 2, 3], [4], gap_weight=1.5,
                                                  distance=standardized_euclidean_distance)
    conditional_features = [0, 1, 2, 3]
    count = 0
    for i in range(1, len(conditional_features)):  # 子集
        for features in combinations(conditional_features, i):
            count += 1
            result = algorithm.dep_adapted(list(features))
            print(list(features), result)
        if count == 4:
            break
    result = algorithm.dep_adapted([0, 1])
    print([0, 1], result)
    result = algorithm.dep_adapted([0, 3])
    print([0, 3], result)
    result = algorithm.dep_adapted([0, 1, 3])
    print([0, 1, 3], result)
    return


def online_feature_selection_adapted3max_test():
    data = pd.read_csv("ExampleData.csv", header=None)
    algorithm = OnlineFeatureSelectionAdapted3Max(np.array(data), [0, 1, 2, 3], [4], gap_weight=1.5,
                                                  distance=standardized_euclidean_distance)
    result = algorithm.run()
    print("result:", result)
    return


def main():
    # generate_delta_neighborhood_test()
    # generate_distance_triangle_matrix_test()
    # generate_k_nearest_neighborhood_test()
    # generate_gap_neighborhood_test()
    dep_adapted_test()
    online_feature_selection_adapted3max_test()
    pass


if __name__ == '__main__':
    i = 0
    main()
