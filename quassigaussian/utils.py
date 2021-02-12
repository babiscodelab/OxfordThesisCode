import numpy as np
import scipy
from Sobol.sobol import sobol, scramble
from Sobol.brownian_bridge import bb


def calculate_G(kappa, t, T):
    return (1/kappa) * (1 - np.exp(-kappa*(T-t)))


def generate_normal_random_numbers(number_paths, number_time_steps, T, number_scrambles=16):
    np.random.seed(88)

    path1 = np.random.normal(size=(int(number_paths/2), number_time_steps))
    path2 = -path1
    res = np.vstack((path1, path2))
    res = res * np.sqrt(T/number_time_steps)
    return res

def generate_sobol_numbers(number_paths, number_time_steps, T, number_scrambles=32):


    unscrambled = sobol(m=int(np.log2(number_paths/number_scrambles)), s=number_time_steps, scramble=False)

    all_dW = []
    for m in range(1, number_scrambles + 1):
        U = scramble(unscrambled).T
        Z = scipy.stats.norm.ppf(U)
        dW = bb(Z, T)
        all_dW.append(dW)

    #U = scramble(unscrambled).T  # generate set of M Sobol points
    res = np.hstack(all_dW)

    return res.T



def get_random_number_generator(type):
    if type=="normal":
        return generate_normal_random_numbers
    elif type=="sobol":
        return generate_sobol_numbers
    else:
        raise Exception("Type not well defined")

def midpoint(f, a, b, n, *args, **kwargs):
    assert a == 0
    n = n - 1
    h = float(b - a) / n
    x = np.linspace(a + h / 2, b - h / 2, n)
    return np.append(np.array([0]), h * np.cumsum(f(x, *args, **kwargs)))