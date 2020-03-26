"""
2020-03-16
result = algorithm.run(non_significant=no_significance)
论文上的例子与实验进行了验证，无误

两个测试结果与之前记录的数据不同
"""


import numpy as np
import pandas as pd
from OS_NRRSAR_SA import *


class NoiseResistantDependencyMeasureTest:
    @staticmethod
    def mean_positive_region_test():
        """
        test mean_positive_region
        :return: None
        """
        data = pd.read_csv("real_value_data.csv", header=None)
        result = NoiseResistantDependencyMeasure.mean_positive_region(
            np.array(data), [0, 1, 2, 3, 4], [i for i in range(5)])
        print("result:", result)
        # 2020-03-16
        # result: [3.0, 3.0, 3.0, 3.0, 3.0]
        return None

    @staticmethod
    def proximity_of_objects_in_boundary_region_to_mean_positive_region_based_distance_test():
        """
        test proximity_of_objects_in_boundary_region_from_mean_positive_region
        :return: None
        """
        data = np.array(pd.read_csv("approximation_data.csv", header=None))
        proximity = NoiseResistantDependencyMeasure. \
            proximity_of_objects_in_boundary_region_to_mean_positive_region_based_distance(
                data, [0, 1, 2], [3, 4], distance=NoiseResistantDependencyMeasure.variation_of_euclidean_distance)
        print("proximity:", proximity)
        # 2020-03-16
        # proximity: 0.2886751345948129
        # 0.5555555555555556
        # 0
        # 0.2222222222222222
        # 0.7777777777777778
        # 0.1
        # 0.07142857142857142
        # 0.75
        return None

    @staticmethod
    def related_information_of_subset_b_test():
        """
        test related_information_of_subset_b_test and impurity_rate
        :return: None
        """
        print(NoiseResistantDependencyMeasure.impurity_rate([0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 5, 25, 7, 96, 2]))
        print(NoiseResistantDependencyMeasure.related_information_of_subset_b(
            [0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 5, 25, 7, 96, 2]))
        print(NoiseResistantDependencyMeasure.impurity_rate(
            [0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 25, 7, 96, 2, 59, 85, 75]))
        print(NoiseResistantDependencyMeasure.related_information_of_subset_b(
            [0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 25, 7, 96, 2, 59, 85, 75]))
        return

    @staticmethod
    def proximity_of_boundary_region_to_positive_region_based_portion_test():
        """
        test proximity_of_boundary_region_to_positive_region_based_portion
        :return: None
        """
        data = pd.read_csv("approximation_data.csv", header=None)
        proximity = NoiseResistantDependencyMeasure.proximity_of_boundary_region_to_positive_region_based_portion(
            np.array(data), [0, 1, 2], [i for i in range(2)])
        print(proximity)  # 1/2 / 5
        proximity = NoiseResistantDependencyMeasure.proximity_of_boundary_region_to_positive_region_based_portion(
            np.array(data), [0, 1, 2], [i for i in range(3)])
        print(proximity)  # 1/2 / 7 = 1/14
        return None

    @staticmethod
    def noisy_dependency_of_feature_subset_a_on_feature_subset_b_test():
        """
        test noisy_dependency_of_feature_subset_a_on_feature_subset_b
        :return: None
        """
        data = pd.read_csv("example_of_noisy_data.csv", header=None)
        the_dependency = NoiseResistantDependencyMeasure.noisy_dependency_of_feature_subset_d_on_feature_subset_c(
            np.array(data), [0], [1])
        print(the_dependency)  # 0.75
        return None


def noise_resistant_assisted_quick_reduction_test():
    """
    test noise_resistant_assisted_quick_reduct
    :return: None
    """
    data = pd.read_csv("approximation_data.csv", header=None)
    algorithm = NoiseResistantAssistedQuickReduct()
    result = algorithm.run(np.array(data), [0, 1, 2, 3], [4])
    # reduction: [2, 0]
    # data = pd.read_csv("arcene_train.csv", header=None)
    # result = algorithm.run(np.array(data), [i for i in range(0, 10000)], [10000])\
    # reduction: [3355]
    print("reduction:", result)
    return None


def online_streaming_noise_resistant_assistant_aided_rough_set_attribute_reduction_using_significance_analysis_test():
    universe = np.array(pd.read_csv("example_data.csv", header=None))
    conditional_features = [0, 1, 2, 3]
    decision_features = [4]
    algorithm = OnlineStreamingNoiseResistantAidedRoughSetAttributeRecutionSignificanceAnalysis(
        universe, conditional_features, decision_features)
    no_significance = "max subset"
    result = algorithm.run(non_significant=no_significance)
    print(no_significance, end="__")
    print("approximation_data reduction:", result)
    # approximation_data reduction: [0, 1, 2]
    # The time used: 0.006995677947998047 seconds

    # 2020-03-16实验结果
    # max subset__approximation_data reduction: [0, 1, 2, 3]

    universe = np.array(pd.read_csv("example_data.csv", header=None))
    conditional_features = [i for i in range(0, 5)]
    decision_features = [5]
    algorithm = OnlineStreamingNoiseResistantAidedRoughSetAttributeRecutionSignificanceAnalysis(
        universe, conditional_features, decision_features)
    no_significance = "max subset"
    result = algorithm.run(non_significant=no_significance)
    print(no_significance, end="__")
    print("example_data reduction:", result)
    # The time used: 0.011988639831542969 seconds
    # example_data reduction: [2, 4]

    universe = np.array(pd.read_csv("example_data.csv", header=None))
    conditional_features = [i for i in range(0, 5)]
    decision_features = [5]
    algorithm = OnlineStreamingNoiseResistantAidedRoughSetAttributeRecutionSignificanceAnalysis(
        universe, conditional_features, decision_features)
    no_significance = "SEB1"
    result = algorithm.run(non_significant=no_significance)
    print(no_significance, end="__")
    print("example_data reduction:", result)
    # The time used: 0.009993553161621094 seconds
    # example_data reduction: [3, 4]

    universe = np.array(pd.read_csv("arcene_train.csv", header=None))
    conditional_features = [i for i in range(0, 10000)]
    decision_features = [10000]
    algorithm = OnlineStreamingNoiseResistantAidedRoughSetAttributeRecutionSignificanceAnalysis(
        universe, conditional_features, decision_features)
    no_significance = "max subset"
    result = algorithm.run(non_significant=no_significance)
    print(no_significance, end="__")
    print("arcene_train reduction:", result)
    # arcene_train reduction: [9997, 9998, 9999]
    # The time used: 374.79046154022217 secondsprint('The time used: {} seconds'.format(time.time() - start))

    # 2020-03-16实验结果
    # max subset__arcene_train reduction: [6579]
    return


def main():
    noise_resistant_assisted_quick_reduction_test()
    online_streaming_noise_resistant_assistant_aided_rough_set_attribute_reduction_using_significance_analysis_test()
    NoiseResistantDependencyMeasureTest.mean_positive_region_test()
    NoiseResistantDependencyMeasureTest. \
        proximity_of_objects_in_boundary_region_to_mean_positive_region_based_distance_test()
    NoiseResistantDependencyMeasureTest.related_information_of_subset_b_test()
    NoiseResistantDependencyMeasureTest.proximity_of_boundary_region_to_positive_region_based_portion_test()
    NoiseResistantDependencyMeasureTest.noisy_dependency_of_feature_subset_a_on_feature_subset_b_test()
    pass


if __name__ == '__main__':
    main()
