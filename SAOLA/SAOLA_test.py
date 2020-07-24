import numpy as np
from SAOLA.ScalableAccurateOnlineApproach import OnlineFeatureSelectionAdapted3Max


data = np.loadtxt("approximation_data.csv", delimiter=",")
data = np.loadtxt("samples.csv", delimiter=",")
# print(data.shape)
# print(data)
# print(data[:, -1])
# label = data[:, -1]
# data = data[:, 0:-1]
# print(data.shape)


conditional_features = [i for i in range(data.shape[1] - 1)]
decision_features = [data.shape[1] - 1]
# print(conditional_features)
# print(decision_features)
algorithm = OnlineFeatureSelectionAdapted3Max(data, conditional_features, decision_features, delta_1=0)
result = algorithm.run()
print(result)
