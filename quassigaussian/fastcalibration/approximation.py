import scipy.integrate as integrate
import math
from quassigaussian.products.pricer import SwapPricer, CapitalX
from quassigaussian.products.instruments import Swap
from scipy.optimize import fsolve
import numpy as np
from quassigaussian.volatility.local_volatility import LinearLocalVolatility
from scipy.interpolate.interpolate import interp1d

class PiterbargExpectationApproximator():

    def __init__(self, sigma_r: LinearLocalVolatility, swap_pricer: SwapPricer):
        self.g_t = lambda t: np.exp(-swap_pricer.kappa * t)
        self.sigma_r = sigma_r
        self.swap_pricer = swap_pricer
        self.capital_x = CapitalX(swap_pricer)



    def calculate_sigma_0(self, t):
        return self.sigma_r.calculate_vola(x=0, t=t)

    def ybar_formula(self, t):

        def integrand(s):
            return np.power(self.calculate_sigma_0(s) * 1 / self.g_t(s), 2)

        return np.power(self.g_t(t), 2) * integrate.quad(integrand, 0, t)[0]

    def _calculate_x0(self, t, swap, s0, y_bar, x0_guess=0):

        def _solve_for_x0(x_bar, swap, s0, t, y_bar):
            return self.swap_pricer.price(swap, x_bar, y_bar, t) - s0

        return fsolve(_solve_for_x0, x0=np.array([x0_guess]), args=(swap, s0, t, y_bar))[0]

    def xbar_formula(self, t, y_bar, swap, s0, x0_guess=0):
        # See Piterbarg p546. However, he forgot 0.5
        x0 = self._calculate_x0(t, swap, s0, y_bar, x0_guess)
        var_s = self._calculate_var_s(t, swap)
        x_bar = x0 + 0.5 * var_s * self.capital_x.d2xds2(swap, x0, y_bar, t)

        return x_bar

    def _calculate_var_s(self, t, swap):

        def integrand(s):
            return np.power(self.swap_pricer.dsdx(swap, 0, 0, s) * self.calculate_sigma_0(s), 2)

        return integrate.quad(integrand, 0, t)[0]

    def calculate_ksi(self, t, swap_value, swap):

        s0 = self.swap_pricer.price(swap, 0, 0, 0)
        y_bar = self.ybar_formula(t)
        x_bar = self.xbar_formula(t, y_bar, swap, s0)
        dsdx = self.swap_pricer.dsdx(swap, x_bar, y_bar, t)
        d2sdx2 = self.swap_pricer.d2sdx2(swap, x_bar, y_bar, t)
        swap_price = self.swap_pricer.price(swap, x_bar, y_bar, t)

        alpha = 0.5 * d2sdx2
        beta = dsdx - d2sdx2 * x_bar
        gamma = swap_price - dsdx * x_bar + 0.5 * d2sdx2 * math.pow(x_bar, 2) - swap_value

        sqrt_delta = math.sqrt(math.pow(beta, 2) - 4 * alpha * gamma)
        res1 = (-beta + sqrt_delta) / (2 * alpha)
        res2 = (-beta - sqrt_delta) / (2 * alpha)

        abs_dif1 = abs(res1 - x_bar)
        abs_dif2 = abs(res2 - x_bar)

        if abs_dif1 > abs_dif2:
            return res2
        else:
            return res1

    def x_bar_simple(self, t, swap_price, swap):
        swap_xy_0 = self.swap_pricer.price(swap, x=0, y=0, t=t)
        return (swap_price - swap_xy_0) / self.swap_pricer.dsdx(swap, x=0, y=0, t=t)


class DisplacedDiffusionParameterApproximator():

    def __init__(self, sigma_r: LinearLocalVolatility, swap_pricer: SwapPricer, swap, expectation_approximator: PiterbargExpectationApproximator):
        self.sigma_r = sigma_r
        self.swap_pricer = swap_pricer
        self.swap = swap
        self.swap_0 = self.swap_pricer.price(swap, 0, 0, 0)
        self.expectation_approximator = expectation_approximator

    def get_lambda_s_square_callable_decorator(self, x_bar, y_bar):
        lambda_s_callable = self.get_lambda_s_callable_decorator(x_bar, y_bar)

        def lambda_s_square_callable(t):
            return np.math.pow(lambda_s_callable(t), 2)

        return lambda_s_square_callable

    def get_lambda_s_callable_decorator(self, x_bar: callable, y_bar: callable):
        def get_lambda_s_callable(t):
            swap_dsdx = self.swap_pricer.dsdx(self.swap, x_bar(t), y_bar(t), t)
            return self.sigma_r.calculate_vola(t=t, x=x_bar(t)) * swap_dsdx / self.swap_0
        return get_lambda_s_callable

    def calculate_lambda_square(self, t):
        y_bar = self.expectation_approximator.ybar_formula(t)
        x_bar = self.expectation_approximator.xbar_formula(t, y_bar, self.swap, self.swap_0, x0_guess=0)
        swap_dsdx = self.swap_pricer.dsdx(self.swap, x_bar, y_bar, t)
        return np.power(self.sigma_r.calculate_vola(t=t, x=x_bar) * swap_dsdx / self.swap_0, 2)


    def get_bs_callable_decorator(self, x_bar: callable, y_bar: callable):

        swap_0 = self.swap_pricer.price(self.swap, 0, 0, 0)
        def get_bs_callable(t):
            swap_dsdx = self.swap_pricer.dsdx(self.swap, x_bar(t), y_bar(t), t)
            b_s = (swap_0 * self.sigma_r.b_t(t)) / ((self.sigma_r.alpha_t(t) + self.sigma_r.b_t(t) * x_bar(t)) * swap_dsdx) \
                  + swap_0 * self.swap_pricer.d2sdx2(self.swap, x_bar(t), y_bar(t), t) / (math.pow(swap_dsdx, 2))
            return b_s

        return get_bs_callable

    def approximate_parameters(self, t):

        swap_0 = self.swap_pricer.price(self.swap, 0, 0, 0)
        y_bar = self.expectation_approximator.ybar_formula(t)
        x_bar = self.expectation_approximator.xbar_formula(t, y_bar, self.swap, self.swap_0)

        swap_dsdx = self.swap_pricer.dsdx(self.swap, x_bar, y_bar, t)

        lambda_s = self.sigma_r.calculate_vola(t=t, x=x_bar) * swap_dsdx/swap_0
        b_s = (swap_0 * self.sigma_r.b_t(t))/((self.sigma_r.alpha_t(t) + self.sigma_r.b_t(t) * x_bar) *swap_dsdx)  \
        + swap_0*self.swap_pricer.d2sdx2(self.swap, x_bar, y_bar, t)/(math.pow(swap_dsdx, 2))

        return lambda_s, b_s