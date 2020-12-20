import numpy as np
from scipy.interpolate import interp1d
from quassigaussian.fastcalibration.parameter_averaging import calculate_lambda_integral, w_s_wrapper

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
