"""
Towards Scalable and Accurate Online Feature Selection for Big Data
author: Kui Yu1, Xindong Wu, Wei Ding, and Jian Pei
https://doi.org/10.1109/ICDM.2014.63
"""
import time
from Tools.mutual_information import mutual_information


class OnlineFeatureSelectionAdapted3Max:
    def __init__(self, universe, conditional_features, decision_features, delta_1=0):
        self.universe = universe
        self.conditional_features = conditional_features
        self.decision_features = decision_features
        self.delta_1 = delta_1
        # self.delta_2 = 0
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

    # checked
    def run(self):
        start = time.time()
        temp = start
        candidate_features = []
        candidate_features_mi = []
        count = 0
        for feature in self.get_new_feature():
            count += 1
            if count % 1000 == 0:
                print(count)
                print("{}  ".format(time.time() - temp), "{}".format(time.time() - start))
                temp = time.time()
            feature_mi = mutual_information(self.universe, [feature], self.decision_features)
            if feature_mi < self.delta_1:
                continue
            # if len(candidate_features) == 0:
            #     self.delta_2 = feature_mi
            # if feature_mi < self.delta_2:
            #     self.delta_2 = feature_mi
            flag = True
            i = 0
            while i < len(candidate_features):
                if (candidate_features_mi[i] > feature_mi) and \
                        (mutual_information(self.universe, [candidate_features[i]], [feature]) >= feature_mi):
                    flag = False
                    break
                if (feature_mi > candidate_features_mi[i]) and \
                        (mutual_information(self.universe, [candidate_features[i]], [feature]) >=
                         candidate_features_mi[i]):
                    candidate_features.pop(i)
                    candidate_features_mi.pop(i)
                    i -= 1
                i += 1
            if flag:
                candidate_features.append(feature)
                candidate_features_mi.append(feature_mi)
        print('The time used: {} seconds'.format(time.time() - start))
        return candidate_features
