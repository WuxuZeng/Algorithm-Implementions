import pandas as pd
import numpy as np
import time
from RoughSet.neighbourhood_rough_set import generate_distance_matrix_old, generate_distance_matrix_by_vector
from Tools.tools import euclidean_distance
from RoughSet.traditional_rough_set import partition, partition_by_equal_array


def test_vector_matrix_calculation(filename):
    data = pd.read_csv(filename, header=None)
    data = np.array(data)
    print(data.shape)
    attributes = [i for i in range(data.shape[1])]

    print("generate_distance_matrix_by_vector")
    start_time = time.time()
    # matrix1 = generate_distance_matrix_by_vector(data, attributes, distance=euclidean_distance)
    generate_distance_matrix_by_vector(data, attributes, distance=euclidean_distance)
    print('The time used: {} seconds'.format(time.time() - start_time))

    print("generate_distance_matrix_old")
    start_time = time.time()
    # matrix2 = generate_distance_matrix_old(data, attributes, distance=euclidean_distance)
    generate_distance_matrix_old(data, attributes, distance=euclidean_distance)
    print('The time used: {} seconds'.format(time.time() - start_time))


def test_len_shape(filename):
    data = pd.read_csv(filename, header=None)
    data = np.array(data)
    print(data.shape)

    start_time = time.time()
    for i in range(1000000):
        len(data)
        len(data.T)
    print('The time used: {} seconds'.format(time.time() - start_time))

    start_time = time.time()
    for i in range(1000000):
        x = data.shape[0]
        y = data.shape[1]
        print(x, y)
    print('The time used: {} seconds'.format(time.time() - start_time))


def classical_rough_set_partition_all():
    data = pd.read_csv("mushroom.csv", header=None)
    data = np.array(data)
    print(data.shape)
    attributes = [i for i in range(data.shape[1])]

    print("partition_by_array_equal")
    start_time = time.time()
    # matrix1 = partition_by_equal_array(data, attributes)
    partition_by_equal_array(data, attributes)
    print('The time used: {} seconds'.format(time.time() - start_time))

    print("partition")
    start_time = time.time()
    # matrix2 = partition(data, attributes)
    partition(data, attributes)
    print('The time used: {} seconds'.format(time.time() - start_time))
    print()
    pass


def main():
    # test_vector_matrix_calculation("arcene_train.csv")
    # 求样本之间的距离矩阵 向量运算快了115倍
    # (100, 10001)
    # generate_distance_matrix_by_vector
    # The time used: 0.5047097206115723 seconds
    # generate_distance_matrix
    # The time used: 58.32284116744995 seconds

    # test_vector_matrix_calculation("wdbc_for_test_data.csv")
    # (569, 30)
    # generate_distance_matrix_by_vector
    # The time used: 1.045215368270874 seconds
    # generate_distance_matrix_old
    # The time used: 6.350998163223267 seconds

    # test_vector_matrix_calculation("approximation_data.csv")
    # the result of two method is same

    # test_len_shape("arcene_train.csv")
    # shape稍快
    # (100, 10001)
    # The time used: 0.285184383392334 seconds
    # The time used: 0.20051932334899902 seconds
    # The time used: 0.3164510726928711 seconds
    # The time used: 0.18492364883422852 seconds

    classical_rough_set_partition_all()
    # partition_by_all()
    # The time used: 168.76050519943237 seconds
    # partition
    # The time used: 84.94578409194946 seconds

    # partition_by_array_equal
    # The time used: 451.3062219619751 seconds
    # partition
    # The time used: 37.42107963562012 seconds


if __name__ == '__main__':
    main()
