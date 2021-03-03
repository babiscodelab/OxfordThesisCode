import numpy as np
import scipy.integrate as integrate
from quassigaussian.products.pricer import SwapPricer
from quassigaussian.parameters.volatility.local_volatility import LinearLocalVolatility
from scipy.interpolate.interpolate import interp1d
from quassigaussian.utils import midpoint

class DiscreteParameterAveraging():

    def __init__(self, integration_grid_size, swap_pricer: SwapPricer, sigma_r: LinearLocalVolatility, swap,
                 xy_calculator):
        self.integration_grid_size = integration_grid_size
        self.g_t = lambda t: np.exp(-swap_pricer.kappa * t)
        self.sigma_r = sigma_r
        self.swap_pricer = swap_pricer
        self.swap = swap
        self.time_grid = np.linspace(0, swap.T0, integration_grid_size)
        self.s0 = self.swap_pricer.price(self.swap, 0, 0, 0)
        self.xy_calculator = xy_calculator

    def calculate_average_param(self):
        xbar, ybar = self.xy_calculator.calculate_xy()
        lambda_s, b_s = self.calculate_lambda_beta(xbar, ybar)
        lambda_avg, b_avg = calculate_avg_lambda_beta(lambda_s, b_s, self.time_grid)
        return lambda_avg, b_avg

    def calculate_lambda_beta(self, x_bar, y_bar):
        lambda_r = self.sigma_r.lambda_t(self.time_grid)
        alpha_r = self.sigma_r.alpha_t(self.time_grid)
        b_r = self.sigma_r.b_t(self.time_grid)
        dsdx = self.swap_pricer.dsdx(self.swap, x_bar, y_bar, self.time_grid)
        d2sdx2 = self.swap_pricer.d2sdx2(self.swap, x_bar, y_bar, self.time_grid)

        lambda_s = lambda_r * (alpha_r + b_r * x_bar) * dsdx / (self.s0)
        b_s = self.s0 * b_r / ((alpha_r + b_r * x_bar) * dsdx) + self.s0 * d2sdx2 / np.power(dsdx, 2)

        return lambda_s, b_s





def calculate_avg_lambda_beta(lambda_s: np.array, b_s: np.array, time_grid):
    t0 = time_grid[0]
    tn = time_grid[-1]

    lambda_square = np.power(lambda_s, 2)
    lambda_square_func = interp1d(time_grid, lambda_square, kind='zero')

    dt = time_grid[1] - time_grid[0]
    lambda_integral = midpoint(lambda_square_func, t0, tn, len(time_grid))
    lambda_integral_product = lambda_integral * lambda_square

    denominator = integrate.romb(lambda_integral_product, dt)
    numerator = integrate.romb(b_s * lambda_integral_product, dt)

    lambda_avg = np.sqrt(lambda_integral[-1] / time_grid[-1])
    b_avg = numerator / denominator

    return lambda_avg, b_avg
