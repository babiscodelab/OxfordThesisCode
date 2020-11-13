import numpy as np

def calculate_x_boundaries(y, kappa, maturity, eta_square):

    exp_x = y/kappa * (1-np.exp(-kappa*maturity))
    var_x = eta_square/(2*kappa) * (1-np.exp(-2*kappa*maturity))

    x_max = exp_x + 3*np.sqrt(var_x)
    x_min = exp_x - 3*np.sqrt(var_x)

    return x_min, x_max



if __name__ == "__main__":

    y = 0.0005
    kappa = 0.3
    maturity = 1
    eta_square = 0.01**2
    print(np.sqrt(eta_square*maturity)*5)
    print(calculate_x_boundaries(y, kappa, maturity, eta_square))