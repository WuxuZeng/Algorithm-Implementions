import pandas as pd
import numpy as np
from RoughSet.QuickReduct import QuickReduction


def quick_reduction_test():
    """
    test quick reduction
    :return: None
    """
    algorithm = QuickReduction()

    # data = pd.read_csv("approximation_data.csv", header=None)
    # result = algorithm.run(np.array(data), [0, 1, 2, 3], [4])
    # reduction: [0, 2]

    data = pd.read_csv("arcene_train.csv", header=None)
    result = algorithm.run(np.array(data), [i for i in range(0, 10000)], [10000])
    print("reduction:", result)
    # reduction: [3355]
    return None


def main():
    quick_reduction_test()
    pass


if __name__ == '__main__':
    main()
