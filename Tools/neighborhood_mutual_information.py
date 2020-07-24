from math import log10


# 为了通用，这里的参数不设置为features，而是设置为邻域
# 因为不同的邻域需要的参数不同，后续可考虑不定参数改进
def neighborhood_uncertainty_of_sample(universe_num, neighborhood):
    return -log10(len(neighborhood) / universe_num)


def average_neighborhood_uncertainty_of_universe(universe_num, neighborhoods, kinds):
    add_sum = 0
    for neighborhood in neighborhoods:
        add_sum += (log10(len(neighborhood)/universe_num))/log10(kinds)
    return -add_sum/universe_num

