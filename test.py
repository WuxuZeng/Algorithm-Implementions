import numpy as np
import pandas as pd
# sklearn.preprocessing  SimpleImputer
# SimpleImputer

# 不会默认读取缺失值
pd_data = pd.read_csv("Resources\\approximation_data.csv", header=None)
# print(pd_data)

# data = np.genfromtxt('Resources\\ExampleData.csv', skip_header=True, delimiter=',')
data = np.genfromtxt('Resources\\ExampleData.csv', delimiter=',')
# print(data)

data = np.array(data)
result = np.isnan(data)
# print(type(result))  # <class 'numpy.ndarray'>
# print(result)

# result = np.isinf(data)
# print(result)

nan_numbers = np.count_nonzero(data != data)
# print(nan_numbers)

print(data[0] != data[0])
print(data[1] != data[1])
a = data[0] != data[0]
b = data[1] != data[1]
c = [False, False, True, False, False]
print(a & b)
print(b & c)
nan_numbers0 = np.count_nonzero(data[0] != data[0])
nan_numbers1 = np.count_nonzero(data[1] != data[1])
# print(nan_numbers1)

flag = np.array([False, False, True, False, False])
data = np.transpose(data)
# print(data)
b = data[flag]
b = np.transpose(b)
# print(b)
data = np.transpose(data)
# print(data)
