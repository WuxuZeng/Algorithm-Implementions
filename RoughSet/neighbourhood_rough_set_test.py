from RoughSet.neighbourhood_rough_set import generate_delta_neighborhood
from tools import *


def generate_delta_neighborhood_test():
    data = pd.read_csv("ExampleData.csv", header=None)
    del data[4]
    result = generate_delta_neighborhood(np.array(data), [0, 1, 2, 3], 33, euclidean_distance)
    print("result:", result)
    # result: [[0], [1, 4], [2, 3, 6], [5, 7]]
    return


def main():
    generate_delta_neighborhood_test()
    pass


if __name__ == '__main__':
    main()
