import numpy as np
import scipy
from sobol import sobol, scramble

def calculate_G(kappa, t, T):
    return (1/kappa) * (1 - np.exp(-kappa*(T-t)))


def generate_normal_random_numbers(number_paths, number_time_steps):
    np.random.seed(42)

    path1 = np.random.normal(size=(int(number_paths/2), number_time_steps))
    path2 = -path1
    return np.vstack((path1, path2))
    #return np.random.normal(size=(number_paths, number_time_steps))


def generate_sobol_numbers(number_paths, number_time_steps):

    unscrambled = sobol(m=int(np.log2(number_paths)), s= number_time_steps, scramble=True)
    #U = scramble(unscrambled).T  # generate set of M Sobol points
    res = unscrambled

    return scipy.stats.norm.ppf(res)  # inverts Normal cum. fn.



def get_random_number_generator(type):
    if type=="normal":
        return generate_normal_random_numbers
    elif type=="sobol":
        return generate_sobol_numbers
    else:
        raise Exception("Type not well defined")
