import numpy as np
from quassigaussian.parameters.volatility.local_volatility import LinearLocalVolatility, LocalVolatility
import scipy.integrate as integrate

def calculate_x_boundaries(y, kappa, maturity, volatility: LinearLocalVolatility):

    eta_square = volatility.calculate_vola(0, 0, 0)
    exp_x = y/kappa * (1-np.exp(-kappa*maturity))
    var_x = eta_square/(2*kappa) * (1-np.exp(-2*kappa*maturity))

    x_max = exp_x + 3*np.sqrt(var_x)
    x_min = exp_x - 3*np.sqrt(var_x)

    return x_min, x_max


def calculate_x_moments(maturity, kappa, volatility: LocalVolatility):

    exp_value = np.power(volatility.calculate_vola(0,0,0), 2)/(2*np.power(kappa,2)) * (1-2*np.exp(-kappa*maturity) + np.exp(-2*kappa*maturity))
    variance = np.power(volatility.calculate_vola(0,0,0), 2)/(2*kappa) *(1-np.exp(-2*kappa*maturity))

    return exp_value, variance

def calculate_x_boundaries3(maturity, kappa, volatility: LocalVolatility, alpha=5):

    exp_value, variance = calculate_x_moments(maturity, kappa, volatility)
    xmax = exp_value + alpha*np.sqrt(variance)
    xmin = exp_value - alpha*np.sqrt(variance)

    return xmin, xmax

def calculate_y_bar(maturity, volatility: LocalVolatility, kappa):
    return np.power(volatility.calculate_vola(0,0,0), 2)/(2*kappa)*(1-np.exp(-2*kappa*maturity))


def calculate_x_boundaries2(maturity, volatility: LocalVolatility, alpha=5):

    x_max = alpha * volatility.calculate_vola(0, 0, 0) * np.sqrt(maturity)
    x_min = -alpha * volatility.calculate_vola(0, 0, 0) * np.sqrt(maturity)

    return x_min, x_max

def calculate_y_boundaries(maturity, kappa, volatility: LocalVolatility, alpha=5):

    def exp_y_integral(u):
        exp_x = np.power(volatility.calculate_vola(u,0,0), 2)/(2*np.power(kappa,2)) * (1-2*np.exp(-kappa*u) + np.exp(-2*kappa*u))
        return 2*np.exp(-2*kappa*(maturity-u)) * exp_x * volatility.calculate_vola(u, 0, 0) * volatility.d_vola_dx(u, 0, 0)

    def var_y_integral(u):
        variance_x = np.power(volatility.calculate_vola(u, 0, 0), 2) / (2 * kappa) * (1 - np.exp(-2 * kappa * u))
        return 4*np.power(volatility.d_vola_dx(u, 0, 0), 2) * np.power(volatility.calculate_vola(u, 0, 0), 2) * np.exp(-4*kappa*(maturity-u)) * variance_x

    exp_y = integrate.quad(exp_y_integral, 0, maturity)[0]
    var_y = integrate.quad(var_y_integral, 0, maturity)[0]

    # y cannot be negative
    y_min =  -alpha * np.sqrt(var_y)
    y_max =  +alpha * np.sqrt(var_y)

    return y_min, y_max


if __name__ == "__main__":

    y = 0.0005
    kappa = 0.3
    maturity = 1
    eta_square = 0.01**2
    print(np.sqrt(eta_square*maturity)*5)

    linear_local_vola = LinearLocalVolatility()
    print(calculate_x_boundaries(y, kappa, maturity, ))