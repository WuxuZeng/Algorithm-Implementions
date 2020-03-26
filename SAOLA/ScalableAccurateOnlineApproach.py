"""
Towards Scalable and Accurate Online Feature Selection for Big Data
author: Kui Yu1, Xindong Wu, Wei Ding, and Jian Pei
https://doi.org/10.1109/ICDM.2014.63
"""


# from mutual_information import *
from Tools.mutual_information import mutual_information


class OnlineFeatureSelectionAdapted3Max:
    def __init__(self, universe, conditional_features, decision_features, delta_1=0):
        self.universe = universe
        self.conditional_features = conditional_features
        self.decision_features = decision_features
        self.delta_1 = delta_1
        self.delta_2 = 0
        return

    def get_new_feature(self):
        """
        to transfer a feature to the algorithm through yield
        :return: None
        """
        conditional_features = self.conditional_features
        for i in range(len(conditional_features)):
            yield conditional_features[i]
        return None

    # unchecked
    def run(self):
        candidate_features = []
        candidate_features_mi = []
        for feature in self.get_new_feature():
            feature_mi = mutual_information(self.universe, feature, self.decision_features)
            if len(candidate_features) == 0:
                self.delta_2 = feature_mi
            if feature_mi < self.delta_2:
                self.delta_2 = feature_mi
            if feature_mi < self.delta_1:
                continue
            flag = True
            for i in range(len(candidate_features)):
                if (candidate_features_mi[i] > feature_mi) and \
                        (mutual_information(self.universe, candidate_features[i], feature) >= self.delta_2):
                    flag = False
                    break
                if (feature_mi > candidate_features_mi[i]) and \
                        (mutual_information(self.universe, candidate_features[i], feature) >= self.delta_2):
                    candidate_features.pop(i)
                    candidate_features_mi.pop(i)
                    i -= 1
                    break
            if flag:
                candidate_features.append(feature)
                candidate_features_mi.append(feature_mi)
        return candidate_features
