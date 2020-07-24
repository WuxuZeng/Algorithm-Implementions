"""
2020-06-27code review：
论文中4.2节求邻域的例子能够对照上，Dep-Adapted部分对照作者源代码修改
其中关于G_mean的求解，作者的代码(分母取n-2)与论文(分母取n-1)有冲突，这里取n-1，同论文和论文中例子保持一致
另：G_mean取n-2能够得出与论文4.4节例子的结果相一致。

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
import time

import numpy as np
import pandas as pd
from itertools import combinations
from RoughSet.neighbourhood_rough_set import generate_euclidean_distance_matrix_by_vector, generate_distance_matrix
from RoughSet.traditional_rough_set import partition
from Tools.tools import standardized_euclidean_distance, euclidean_distance


def generate_gap_neighborhood(universe, attributes, gap_weight=1., distance=standardized_euclidean_distance):
    """
    generate the gap neighborhoods of the universe
    :param universe: the universe of objects(feature vector/sample/instance)
    :param attributes: features' index
    :param gap_weight: the weight of gap
    :param distance: the method to calculate the distance
    :return: list, the k_nearest_neighborhood(raw_universe/attributes)
    """
    if distance == euclidean_distance:
        distance_matrix = generate_euclidean_distance_matrix_by_vector(universe, attributes)
    elif distance == standardized_euclidean_distance:
        distance_matrix = generate_euclidean_distance_matrix_by_vector(universe, attributes, standard=True)
    else:
        distance_matrix = generate_distance_matrix(universe, attributes, distance)
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
        # distance_mean = (distance_max - distance_min) / (universe.shape[0] - 2)
        gap = distance_mean * gap_weight
        j = 2

        # find the gap
        while j < len(distance_sort):
            if distance_matrix[universe_index[i]][distance_sort[j]] - \
                    distance_matrix[universe_index[i]][distance_sort[j - 1]] >= gap:
                break
            j += 1

        elementary_sets.append(distance_sort[:j])
        i += 1

    return elementary_sets


def generate_gap_neighborhood_test():
    data = pd.read_csv("ExampleData.csv", header=None)
    del data[4]
    result = generate_gap_neighborhood(
        np.array(data), [0, 1], 1.5, standardized_euclidean_distance)
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
        if len(attributes) == 0:
            return 0
        part_d_s = 0
        gap_neighborhoods = \
            generate_gap_neighborhood(self.universe, attributes, gap_weight=self.gap_weight, distance=self.distance)
        partitions = partition(self.universe, self.decision_features)

        # 样本本身包含进邻域,s_card计算的是相似度，同样本标签一致的对象为positive sample
        # 根据作者源代码，确定是这种形式
        # 邻域中同目标样本一致的标签的样本数（排除目标样本）除以邻域中样本总数（也排除目标样本）
        # result: [0, 3]
        for gap_neighborhood in gap_neighborhoods:
            for single_partition in partitions:
                if gap_neighborhood[0] in single_partition:
                    part_d_s += \
                        (len([j for j in gap_neighborhood if j in single_partition]) - 1) / (len(gap_neighborhood) - 1)
        d_s = part_d_s / self.universe.shape[0]
        return d_s

    def get_new_feature(self):
        """
        to transfer a feature to the algorithm through yield
        :return: None
        """
        conditional_features = self.conditional_features
        for j in range(len(conditional_features)):
            yield conditional_features[j]
        return None

    def run(self):
        start = time.time()
        redundant_time = 0
        redundant_count = 0
        temp = start
        candidate_features = []
        candidate_dependency = 0
        mean_dependency_of_candidate = 0
        count = 0
        for feature in self.get_new_feature():
            count += 1
            if count % 1000 == 0:
                print(count)
                print("{}  ".format(time.time() - temp), "{}".format(time.time() - start))
                temp = time.time()
            feature_dependency = self.dep_adapted([feature])
            if feature_dependency < mean_dependency_of_candidate:
                continue
            temp_candidate_features = copy.deepcopy(candidate_features)
            temp_candidate_features.append(feature)
            temp_candidate_dependency = self.dep_adapted(temp_candidate_features)
            if temp_candidate_dependency > candidate_dependency:
                candidate_features = temp_candidate_features
                candidate_dependency = temp_candidate_dependency
                # 先把其加进来，直接用len取长度个数就是总个数，所以下面这个操作是
                # 新平均 = (先前平均*先前个数 + 新特征依赖度)/总个数
                mean_dependency_of_candidate = \
                    ((mean_dependency_of_candidate * (len(candidate_features) - 1)) + feature_dependency) / \
                    len(candidate_features)
            elif temp_candidate_dependency == candidate_dependency:
                if candidate_dependency == 0 and len(candidate_features) == 0:
                    continue
                candidate_features.append(feature)
                random.shuffle(candidate_features)
                i = 0
                redundant_start = time.time()
                redundant_count += 1
                while i < len(candidate_features):
                    temp_candidate_features = copy.deepcopy(candidate_features)
                    temp_candidate_features.pop(i)
                    temp_candidate_features_dependency = self.dep_adapted(temp_candidate_features)
                    if (self.dep_adapted(candidate_features) - temp_candidate_features_dependency) == 0:
                        test_feature_dependency = self.dep_adapted([candidate_features[i]])
                        candidate_features.pop(i)
                        i -= 1
                        if mean_dependency_of_candidate > 0:
                            mean_dependency_of_candidate = \
                                (mean_dependency_of_candidate*(len(candidate_features)+1) - test_feature_dependency) /\
                                len(candidate_features)
                    i += 1
                redundant_time += time.time() - redundant_start
            pass
        print('The time used: {} seconds'.format(time.time() - start))
        print('redundant_time: {} seconds'.format(redundant_time))
        print(redundant_count)
        return candidate_features


def dep_adapted_test():
    data = pd.read_csv("ExampleData.csv", header=None)
    algorithm = OnlineFeatureSelectionAdapted3Max(np.array(data), [0, 1, 2, 3], [4], gap_weight=1.5,
                                                  distance=standardized_euclidean_distance)
    conditional_features = [0, 1, 2, 3]
    result = algorithm.dep_adapted([0, 1])
    print(str([0, 1]), result)
    count = 0
    for j in range(1, len(conditional_features)):  # subset
        for features in combinations(conditional_features, j):
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
    # generate_distance_triangle_matrix_test()
    # generate_k_nearest_neighborhood_test()
    # generate_gap_neighborhood_test()
    dep_adapted_test()
    online_feature_selection_adapted3max_test()
    pass


if __name__ == '__main__':
    main()
