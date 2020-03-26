"""
Online streaming feature selection using rough sets
author: S.Eskandari M.M.Javidi
https://doi.org/10.1016/j.ijar.2015.11.006
"""
import time
import math
import copy
import random
from RoughSet.traditional_rough_set import positive_region_of_sample_subset, partition, \
    boundary_region_of_sample_subset, dependency


class NoiseResistantDependencyMeasure:
    def __init__(self):
        return

    @staticmethod
    def mean_positive_region(universe, sample_subset, feature_subset):
        """
        [important] Only applicable for continuous values!!!!!!
        Not applicable to discrete values.
        don't consider the condition that the positive region is empty

        get the feature_subset lower approximations of sample_subset
        :param universe: the universe of objects(feature vector/sample/instance)
        :param sample_subset: list, a set of features' serial number
        :param feature_subset: features' index
        :return: list(object), the mean of all object attributes values in positive region
        """
        positive_region = positive_region_of_sample_subset(universe, sample_subset, feature_subset)
        total = []
        mean = []
        for i in range(len(feature_subset)):
            total.append(0)
            mean.append(0)
        for x in positive_region:
            for i in range(len(feature_subset)):
                total[i] += (universe[x])[feature_subset[i]]
        for i in range(len(feature_subset)):
            mean[i] = total[i] / len(positive_region)
        return mean

    # for continuous data
    @staticmethod
    def proximity_of_objects_in_boundary_region_to_mean_positive_region_based_distance(
            universe, features_1, features_2, distance):
        """
        proximity_of_objects_in_boundary_region_from_mean_positive_region
        don't consider the condition that the positive region is empty
        :param universe: the universe of objects(feature vector/sample/instance)
        :param features_1: list, a set of features' serial number
        :param features_2: list, a set of features' serial number
        :param distance: the method to calculate the distance of objects
        :return: float
        """
        partition_2 = partition(universe, features_2)
        boundary = []
        positive = []
        for subset in partition_2:
            boundary.extend(boundary_region_of_sample_subset(universe, subset, features_1))
            boundary = list(set(boundary))
            positive.extend(positive_region_of_sample_subset(universe, subset, features_1))
        if len(boundary) == 0:
            return 1
        if len(positive) == 0:
            return 1 / len(boundary)
        mean = NoiseResistantDependencyMeasure.mean_positive_region(universe, positive, features_1)
        proximity_of_object_in_boundary_from_mean = 0
        for y in boundary:
            proximity_of_object_in_boundary_from_mean += distance(mean, universe[y], features_1)
        return 1 / proximity_of_object_in_boundary_from_mean

    @staticmethod
    def variation_of_euclidean_distance(x, y, features):
        """
        calculate the distance of two objects
        :param x: feature vector, sample, instance
        :param y: feature vector, sample, instance
        :param features: list, a set of features' serial number
        :return: float, distance
        """
        total = 0
        for i in range(len(features)):
            if x[i] == y[features[i]]:
                continue
            total += 1
        return math.sqrt(total)

    @staticmethod
    def impurity_rate(subset_a, subset_b):
        """
        the noise portion of subset_a to subset_b
        impurity rate of subset_a with respect to subset_b, the noise information
        :param subset_a: a set of objects' serial number
        :param subset_b: a set of objects' serial number
        :return: float, impurity rate of subset_a with respect to subset_b
        """
        difference = [i for i in subset_a if i not in subset_b]
        return len(difference) / len(subset_a)

    @staticmethod
    def related_information_of_subset_b(subset_a, subset_b):
        """
        related_information in subset_b from subset_a, the useful information
        generated by impurity_rate
        :param subset_a: a set of objects' serial number
        :param subset_b: a set of objects' serial number
        :return: float, related_information in subset_b
        """
        impurity = NoiseResistantDependencyMeasure.impurity_rate(subset_a, subset_b)
        if impurity > 0.5:
            return 0
        else:
            return 1 - impurity

    @staticmethod
    def proximity_of_boundary_region_to_positive_region_based_portion(universe, sample_subset, feature_subset):
        """
        a noise measure function
        to describe the information contain by the boundary of partition(universe, sample_subset)
        :param universe: the universe of objects(feature vector/sample/instance)
        :param sample_subset: list, a set of objects' serial number
        :param feature_subset: list, a set of features' serial number
        :return: float, the proximity
        """
        partition_1 = partition(universe, feature_subset)
        total = 0
        for elementary_set in partition_1:
            related_information = NoiseResistantDependencyMeasure.related_information_of_subset_b(
                elementary_set, sample_subset)
            if related_information != 1:
                total += related_information
        return total / (len(partition_1))

    @staticmethod
    def noisy_dependency_of_feature_subset_d_on_feature_subset_c(universe, feature_subset_c, feature_subset_d):
        """
        :param universe: the universe of objects(feature vector/sample/instance)
        :param feature_subset_c: list, a set of features' serial number
        :param feature_subset_d: list, a set of features' serial number
        :return: noisy dependency of feature subset a on feature subset b
        """
        partition_d = partition(universe, feature_subset_d)
        total_dependency = 0
        for p in partition_d:
            the_dependency = NoiseResistantDependencyMeasure. \
                proximity_of_boundary_region_to_positive_region_based_portion(universe, p, feature_subset_c)
            total_dependency += the_dependency
        return total_dependency

    # notice the data's type
    @staticmethod
    def proximity_combine_noisy_dependency(universe, feature_subset_c, feature_subset_d):
        """
        :param universe: the universe of objects(feature vector/sample/instance)
        :param feature_subset_c: list, a set of features' serial number
        :param feature_subset_d: list, a set of features' serial number
        :return: float, the combined measure value
        """
        proximity = \
            NoiseResistantDependencyMeasure. \
                proximity_of_objects_in_boundary_region_to_mean_positive_region_based_distance(
                    universe, feature_subset_c, feature_subset_d,
                    distance=NoiseResistantDependencyMeasure.variation_of_euclidean_distance)
        noisy_dependency = NoiseResistantDependencyMeasure.noisy_dependency_of_feature_subset_d_on_feature_subset_c(
            universe, feature_subset_c, feature_subset_d)
        return proximity + noisy_dependency

    @staticmethod
    def noise_resistant_evaluation_measure(universe, feature_subset_c, feature_subset_d):
        """
        :param universe: the universe of objects(feature vector/sample/instance)
        :param feature_subset_c: list, a set of features' serial number
        :param feature_subset_d: list, a set of features' serial number
        :return: float, the noise resistant evaluation measure value
        """
        combined_value = NoiseResistantDependencyMeasure.proximity_combine_noisy_dependency(
            universe, feature_subset_c, feature_subset_d)
        dependency_value = dependency(universe, feature_subset_c, feature_subset_d)
        return (combined_value + dependency_value) / 2


class NoiseResistantAssistedQuickReduct:
    def __init__(self):
        return

    @staticmethod
    def run(universe, raw_conditional_features, decision_features):
        """
        to get a reduction by Sequential Forward Selection method using dependency based on positive region
        :param universe: the universe of objects(feature vector/sample/instance)
        :param raw_conditional_features: list, a set of features' serial number
        :param decision_features: list, a set of features' serial number
        :return: candidate_features
        """
        destination_dependency = dependency(universe, raw_conditional_features, decision_features)
        conditional_features = copy.deepcopy(raw_conditional_features)
        candidate_features = []
        candidate_dependency = 0
        # the dependency is monotonic with positive,
        # because more conditional features can divide the universe into more partitions
        count = 0
        while destination_dependency != candidate_dependency:
            count += 1
            # print(count)
            noise_resistant_increase = 0
            test_features = copy.deepcopy(candidate_features)
            count1 = 0
            index = 0
            for i in range(len(conditional_features)):
                count1 += 1
                if count1 % 1000 == 0:
                    print(count1, end="-")
                test_features.append(conditional_features[i])
                result = NoiseResistantDependencyMeasure.noise_resistant_evaluation_measure(
                    universe, test_features, [decision_features]) - \
                         NoiseResistantDependencyMeasure.noise_resistant_evaluation_measure(
                             universe, candidate_features, [decision_features])
                test_features.remove(conditional_features[i])
                if result > noise_resistant_increase:
                    noise_resistant_increase = result
                    index = i
            feature = conditional_features[index]
            candidate_features.append(feature)
            conditional_features.remove(feature)
            candidate_dependency = dependency(universe, candidate_features, [decision_features])
        return candidate_features


class OnlineStreamingNoiseResistantAidedRoughSetAttributeRecutionSignificanceAnalysis:
    def __init__(self, universe, conditional_features, decision_features):
        self.universe = universe
        self.conditional_features = conditional_features
        self.decision_features = decision_features
        return

    @staticmethod
    def significance_of_feature_subset(universe, candidate_features, decision_features, features):
        """
        significance of features belongs to candidate_features + features
        :param universe: the universe of objects(feature vector/sample/instance)
        :param candidate_features: list, a set of features' serial number
        :param decision_features: list, a set of features' serial number
        :param features:
        :return: the significance
        """
        test_features = copy.deepcopy(candidate_features)
        for feature in features:
            test_features.append(feature)
        test_features_dependency = dependency(universe, test_features, decision_features)
        candidate_features_dependency = dependency(universe, candidate_features, decision_features)
        return (test_features_dependency - candidate_features_dependency) / test_features_dependency

    @staticmethod
    def subsets_recursive(items):
        # the power set of the empty set has one element, the empty set
        result = [[]]
        for x in items:
            result.extend([subset + [x] for subset in result])
        return result

    @staticmethod
    def heuristic_non_significant(universe, raw_candidate_features, decision_features, feature):
        """
        non_significance in candidate_features + feature due to the feature
        :param universe: the universe of objects(feature vector/sample/instance)
        :param raw_candidate_features:
        :param decision_features:
        :param feature:
        :return:
        """
        result = []
        max_size = 0
        candidate_features = copy.deepcopy(raw_candidate_features)
        test_candidate_features = copy.deepcopy(candidate_features)
        candidate_features.append(feature)
        subsets = OnlineStreamingNoiseResistantAidedRoughSetAttributeRecutionSignificanceAnalysis.subsets_recursive(
            test_candidate_features)
        subsets.remove([])
        for item in subsets:
            if OnlineStreamingNoiseResistantAidedRoughSetAttributeRecutionSignificanceAnalysis. \
                    significance_of_feature_subset(universe, [i for i in candidate_features if i not in item],
                                                   decision_features, item) == 0:
                if len(item) > max_size:
                    max_size = len(item)
                    result = item
        return result

    @staticmethod
    def sequential_backward_elimination_non_significant(universe, raw_candidate_features, decision_features,
                                                        feature):
        """
        non_significance in candidate_features + feature due to the feature
        :param universe: the universe of objects(feature vector/sample/instance)
        :param raw_candidate_features:
        :param decision_features:
        :param feature:
        :return:
        """
        result = []
        candidate_features = copy.deepcopy(raw_candidate_features)
        test_candidate_features = copy.deepcopy(candidate_features)
        candidate_features.append(feature)
        while len(test_candidate_features) != 0:
            g = test_candidate_features[random.randint(0, len(test_candidate_features) - 1)]
            if OnlineStreamingNoiseResistantAidedRoughSetAttributeRecutionSignificanceAnalysis. \
                    significance_of_feature_subset(universe, candidate_features, decision_features, [g]) == 0:
                result.append(g)
                candidate_features.remove(g)
            test_candidate_features.remove(g)
        return result

    def get_new_feature(self):
        """
        to transfer a feature to the algorithm through yield
        :return: None
        """
        conditional_features = self.conditional_features
        for i in range(len(conditional_features)):
            yield conditional_features[i]
        return None

    def run(self, non_significant="SEB3"):
        """
        :param non_significant: the non-significant implementation method, max subset, SEB1, SEB3, default is SEB3
        :return: the reduction
        """
        start = time.time()
        temp = start
        candidate_features = []
        count = 0
        for feature in self.get_new_feature():
            count += 1
            if count % 1000 == 0:
                print(count)
                print("{}  ".format(time.time() - temp), "{}".format(time.time() - start))
                temp = time.time()
            test_features = copy.deepcopy(candidate_features)
            test_features.append(feature)
            NoiseResistantDependencyMeasure.noise_resistant_evaluation_measure(
                self.universe, [feature], self.decision_features)
            non_significant_subset = []
            if dependency(self.universe, candidate_features, self.decision_features) != 1:
                if (dependency(self.universe, test_features, self.decision_features) -
                    dependency(self.universe, candidate_features, self.decision_features)) > 0 or \
                        NoiseResistantDependencyMeasure.noise_resistant_evaluation_measure(
                            self.universe, [feature], self.decision_features) > 0:
                    candidate_features.append(feature)
            else:
                if non_significant == "max subset":
                    non_significant_subset = self.heuristic_non_significant(self.universe, candidate_features,
                                                                            self.decision_features, feature)
                elif non_significant == "SEB1":
                    non_significant_subset = self.sequential_backward_elimination_non_significant(
                        self.universe, candidate_features, self.decision_features, feature)
                elif non_significant == "SEB3":
                    max_length = 0
                    for i in range(0, 3):
                        temp_non_significant_subset1 = self.sequential_backward_elimination_non_significant(
                            self.universe, candidate_features, self.decision_features, feature)
                        temp_length = len(temp_non_significant_subset1)
                        if temp_length > max_length:
                            max_length = temp_length
                            non_significant_subset = temp_non_significant_subset1
                else:
                    print("wrong parameter of " + non_significant)
                    return
                if len(non_significant_subset) > 1:
                    candidate_features.append(feature)
                    candidate_features = [i for i in candidate_features if i not in non_significant_subset]
                if len(non_significant_subset) == 1:
                    noise_resistant_measure_a = NoiseResistantDependencyMeasure.noise_resistant_evaluation_measure(
                        self.universe, [feature], self.decision_features)
                    noise_resistant_measure_b = NoiseResistantDependencyMeasure.noise_resistant_evaluation_measure(
                        self.universe, non_significant_subset, self.decision_features)
                    if noise_resistant_measure_a > noise_resistant_measure_b:
                        candidate_features.remove(non_significant_subset[0])
                        candidate_features.append(feature)
                    elif noise_resistant_measure_a < noise_resistant_measure_b:
                        candidate_features.append(feature)
                    else:
                        flag = random.randint(0, 1)
                        if flag == 1:
                            candidate_features.remove(non_significant_subset[0])
                            candidate_features.append(feature)
                        else:
                            pass
        print('The time used: {} seconds'.format(time.time() - start))
        return candidate_features