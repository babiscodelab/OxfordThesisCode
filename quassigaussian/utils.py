import numpy as np

def calculate_G(kappa, t, T):
    return (1/kappa) * (1 - np.exp(-kappa*(T-t)))
