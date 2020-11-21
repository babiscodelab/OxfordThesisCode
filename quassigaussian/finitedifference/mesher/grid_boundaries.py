import numpy as np
from quassigaussian.volatility.local_volatility import LinearLocalVolatility
import scipy.integrate as integrate

def calculate_x_boundaries(y, kappa, maturity, volatility: LinearLocalVolatility):

    eta_square = volatility.calculate_vola(0, 0, 0)
    exp_x = y/kappa * (1-np.exp(-kappa*maturity))
    var_x = eta_square/(2*kappa) * (1-np.exp(-2*kappa*maturity))

    x_max = exp_x + 3*np.sqrt(var_x)
    x_min = exp_x - 3*np.sqrt(var_x)

    return x_min, x_max


def calculate_x_boundaries2(maturity, volatility: LinearLocalVolatility, alpha=5):

    x_max = alpha * volatility.calculate_vola(0, 0, 0) * np.sqrt(maturity)
    x_min = -alpha * volatility.calculate_vola(0, 0, 0) * np.sqrt(maturity)

    return x_min, x_max

def calculate_y_boundaries(maturity, kappa, volatility: LinearLocalVolatility, alpha=5):

    def exp_y_integral(u):
        eta_square = np.power(volatility.calculate_vola(u, 0, 0), 2)
        return eta_square * np.exp(-2*kappa*(maturity-u))

    def var_y_integral(u):
        return 4*np.power(volatility.d_vola_dx(u, 0, 0), 2) * np.power(volatility.calculate_vola(u, 0, 0), 2) * np.exp(-4*kappa*(maturity-u))

    exp_y = integrate.quad(exp_y_integral, 0, maturity)[0]
    var_y = integrate.quad(var_y_integral, 0, maturity)[0]

    # y cannot be negative
    y_min = 0
    y_max = exp_y + alpha * np.sqrt(var_y)

    return y_min, y_max


if __name__ == "__main__":

    y = 0.0005
    kappa = 0.3
    maturity = 1
    eta_square = 0.01**2
    print(np.sqrt(eta_square*maturity)*5)

    linear_local_vola = LinearLocalVolatility()
    print(calculate_x_boundaries(y, kappa, maturity, ))