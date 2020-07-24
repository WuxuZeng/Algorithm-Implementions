"""
complete tolerance neighborhood rough set
"""
import numpy as np
from traditional_rough_set import is_indiscernible_discrete, partition, set_is_include, upper_approximations_of_universe


def generate_complete_tolerance_neighborhood(
        universe, conditional_features, discrete_state, radius, distance_function):
    """
    :param universe:
    :param conditional_features: discrete value and continuous value
    :param discrete_state: list, discrete variable are true
    :param radius: the radius of the continuous value
    :param distance_function: the function to calculate the distance between two samples with continuous value
    :return:
    """
    # 直接操作universe更快还是取出对应的列操作更快？待验证
    # discrete_features = np.array(conditional_features)[np.array(discrete_state)]
    discrete_features = [i for i in conditional_features if discrete_state[i]]
    # continuous_features = np.array(conditional_features)[~(np.array(discrete_state))]
    continuous_features = [i for i in conditional_features if not discrete_state[i]]
    tmp_universe = np.transpose(np.transpose(universe)[conditional_features])
    systematic_complete_degree = \
        np.count_nonzero(tmp_universe != tmp_universe) / tmp_universe.shape[0] * tmp_universe.shape[1]
    elementary_sets = []  # neighborhood classes
    for i in range(universe.shape[0]):
        # print(i)
        IND_IS_R = [i]  # neighborhood class
        for j in range(universe.shape[0]):
            if i == j:
                continue
            i_nan_flag: list = (universe[i][np.array(conditional_features)] !=
                                universe[i][np.array(conditional_features)])
            j_nan_flag: list = (universe[j][np.array(conditional_features)] !=
                                universe[j][np.array(conditional_features)])
            intersection = [i for i in i_nan_flag if i in j_nan_flag]
            if len(intersection) == 0:
                continue
            if not len(intersection) / min(len(i_nan_flag), len(j_nan_flag)) >= systematic_complete_degree:
                continue
            tmp_features = discrete_features
            not_nan_features = [k for k in [tmp_features[i] for i in range(len(universe[i][tmp_features]))
                                            if (universe[i][tmp_features] == universe[i][tmp_features])[i]]
                                if k in [tmp_features[j] for j in range(len(universe[j][tmp_features]))
                                         if (universe[j][tmp_features] == universe[j][tmp_features])[j]]]
            if not is_indiscernible_discrete(universe, i, j, not_nan_features):
                continue
            tmp_features = continuous_features
            not_nan_features = [k for k in [tmp_features[i] for i in range(len(universe[i][tmp_features]))
                                            if (universe[i][tmp_features] == universe[i][tmp_features])[i]]
                                if k in [tmp_features[j] for j in range(len(universe[j][tmp_features]))
                                         if (universe[j][tmp_features] == universe[j][tmp_features])[j]]]
            if not distance_function(universe, i, j, not_nan_features) <= radius:
                continue
            IND_IS_R.append(j)
        elementary_sets.append(IND_IS_R)
    return elementary_sets


def lower_approximations_of_mix_universe_neighborhood(
        universe, conditional_features, decision_features, discrete_state, radius, distance_function):
    lower_approximations = []
    equivalence_classes = \
        generate_complete_tolerance_neighborhood(
            universe, conditional_features, discrete_state, radius, distance_function)
    decision_classes = partition(universe, decision_features)
    for x in equivalence_classes:
        if set_is_include(x, decision_classes):
            # print(x)
            lower_approximations.append(x[0])
            # lower_approximations.extend(x) # wrong
    lower_approximations.sort()
    return lower_approximations


def positive_region_of_universe_mix_neighborhood(
        universe, conditional_features, decision_features, discrete_state, radius, distance_function):
    positive_region = lower_approximations_of_mix_universe_neighborhood(
        universe, conditional_features, decision_features, discrete_state, radius, distance_function)
    return positive_region


def boundary_region_of_sample_subset_mix_neighborhood(
        universe, conditional_features, decision_features, discrete_state, radius, distance_function):
    upper_approximations = upper_approximations_of_universe(universe)
    lower_approximations = lower_approximations_of_mix_universe_neighborhood(
        universe, conditional_features, decision_features, discrete_state, radius, distance_function)
    boundary_region = [i for i in upper_approximations if i not in lower_approximations]
    return boundary_region


def dependency_mix_neighborhood(universe, conditional_features, decision_features,
                                discrete_state, radius, distance_function):
    if len(conditional_features) == 0:
        return 0
    positive_region_size = \
        len(positive_region_of_universe_mix_neighborhood(
            universe, conditional_features, decision_features, discrete_state, radius, distance_function))
    dependency_degree = positive_region_size/len(universe)
    return dependency_degree


from neighbourhood_rough_set import lower_approximations_of_universe_neighborhood


def dependency_neighborhood_neighborhood(universe, conditional_features, decision_features, radius):
    if len(conditional_features) == 0:
        return 0
    positive_region_size = \
        len(lower_approximations_of_universe_neighborhood(universe, conditional_features,
                                                          decision_features, radius))
    dependency_degree = positive_region_size/len(universe)
    return dependency_degree
