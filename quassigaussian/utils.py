import numpy as np

def calculate_G(kappa, t, T):
    return (1/kappa) * (1 - np.exp(-kappa*(T-t)))


def generate_random_numbers(number_paths, number_time_steps):
    np.random.seed(42)
    return np.random.normal(size=(number_paths, number_time_steps))


