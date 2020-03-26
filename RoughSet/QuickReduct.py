import copy
from traditional_rough_set import dependency


class QuickReduction:
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
        current_dependency = 0
        # the dependency is monotonic with positive,
        # because more conditional features can divide the universe into more partitions
        while destination_dependency != candidate_dependency:
            index = 0
            for i in range(len(conditional_features)):
                candidate_features.append(conditional_features[i])
                result = dependency(universe, candidate_features, [decision_features])
                candidate_features.remove(conditional_features[i])
                if result > current_dependency:
                    current_dependency = result
                    index = i
            feature = conditional_features[index]
            candidate_features.append(feature)
            conditional_features.remove(feature)
            candidate_dependency = current_dependency
        return candidate_features
