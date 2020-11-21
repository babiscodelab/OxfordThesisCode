import numpy as np

def calculate_G(kappa, t, T):
    return (1/kappa) * (1 - np.exp(-kappa*(T-t)))


def generate_random_numbers(number_paths, number_time_steps):
    np.random.seed(42)
    return np.random.normal(size=(number_paths, number_time_steps))


def extract_x0_result(res, x_grid, y_grid):

    x0_pos = np.where(x_grid == 0)[0]
    y0_pos = np.where(y_grid == 0)[0]

    return res[x0_pos, y0_pos][0]