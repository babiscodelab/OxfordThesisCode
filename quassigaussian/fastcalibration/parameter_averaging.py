import scipy.integrate
import numpy as np
from quassigaussian.products.instruments import Swaption
from quassigaussian.products.pricer import Black76Pricer, find_implied_black_vola, SwapPricer, BondPricer
from scipy.interpolate import interp1d




def calculate_swaption_approx_price(swaption: Swaption, swap_pricer: SwapPricer, w_s, lambda_integral, b_s):


    lambda_s_bar = calculate_lambda_s_bar(lambda_integral, swaption.expiry)

    b_bar = calculate_b_s_bar(w_s, b_s, swaption.expiry)
    black76price = Black76Pricer(lambda_s_bar, b_bar, swap_pricer, swap_pricer.bond_pricer)

    swaption_value = black76price.black76_price(swaption)
    black_implied_vola = find_implied_black_vola(swaption_value, swaption, swap_pricer, swap_pricer.bond_pricer)
    return swaption_value, black_implied_vola



class ApproximatorDiscretizer():

    def __init__(self, swap, swap_pricer, number_points=100):
        self.swap = swap
        self.swap_pricer = swap_pricer
        self.T0 = swap.T0
        self.time_grid = np.arange(0, self.T0+self.T0/number_points, self.T0 / number_points)

    def calculate_discretized_expectations(self, xbar_formula: callable, ybar_formula: callable):

        time_grid = self.time_grid
        swap_0 = self.swap_pricer.price(self.swap, 0, 0, 0)
        x_bar_d = y_bar_d = np.zeros(len(time_grid))

        for i in range(1, len(time_grid)):
            print(i)
            t = time_grid[i]
            y_bar_d[i] = ybar_formula(t)
            x_bar_d[i] = xbar_formula(t, y_bar_d[i], self.swap, swap_0, x0_guess=x_bar_d[i-1])

        xbar = interp1d(time_grid, y_bar_d)
        ybar = interp1d(time_grid, x_bar_d)

        return xbar, ybar


    def calculate_discretized_lambda(self, lambda_square):

        time_grid = self.time_grid
        lambda_square_d = np.zeros(len(time_grid))
        lambda_square_integral_d = np.zeros(len(time_grid))

        for i in range(1, len(time_grid)):
            t = time_grid[i]
            lambda_square_d[i] = lambda_square(t)
            lambda_square_integral_d[i] = lambda_square_integral_d[i-1] + calculate_lambda_integral(lambda_square, time_grid[i-1], time_grid[i])

        lambda_square = interp1d(time_grid, lambda_square_d)
        lambda_square_integral = interp1d(time_grid, lambda_square_integral_d)

        return lambda_square, lambda_square_integral

    def calculate_discretized_w(self, lambda_square, lambda_integral):

        time_grid = self.time_grid
        w_s_d = np.zeros(len(time_grid))
        w_s = w_s_wrapper(self.T0, lambda_square, lambda_integral)

        for i in range(1, len(time_grid)):
            t = time_grid[i]
            w_s_d[i] = w_s(t)

        return interp1d(time_grid, w_s_d)

def calculate_lambda_integral(lambda_square: callable, t_from: float, t_to: float):
    return scipy.integrate.quad(lambda_square, t_from, t_to)[0]

def calculate_lambda_s_bar(lambda_square_integral: callable, T0: float):
    return np.sqrt(lambda_square_integral(T0)/ T0)


def w_s_wrapper(T0: float, lambda_square: callable, lambda_integral: callable):

    def ws_denominator_integral(u):
        return lambda_integral(u) * lambda_square(u)

    denominator = scipy.integrate.quad(ws_denominator_integral, 0, T0)[0]

    def w_s(t):
        numerator = lambda_square(t) * lambda_integral(t)
        return numerator/denominator

    return w_s


def calculate_b_s_bar(w_s: callable, b_s: callable, T0):

   def b_integral(t, w_s, b_s):
       return w_s(t) * b_s(t)

   b_bar = scipy.integrate.quad(b_integral, 0, T0, args=(w_s, b_s))[0]

   return b_bar