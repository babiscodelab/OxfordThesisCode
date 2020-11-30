import scipy.integrate as integrate
import math
from quassigaussian.products.pricer import SwapPricer, CapitalX
from scipy.optimize import fsolve
import numpy as np
from quassigaussian.volatility.local_volatility import LinearLocalVolatility

class PiterbargApproximator():

    def __init__(self, sigma_r: LinearLocalVolatility, swap_pricer: SwapPricer):
        self.g_t = lambda t: np.exp(-swap_pricer.kappa*t)
        self.sigma_r = sigma_r
        self.swap_pricer = swap_pricer
        self.capital_x = CapitalX(swap_pricer)

    def approximate_parameters(self, swap, t):

        swap_0 = self.swap_pricer.price(swap, 0, 0, 0)
        y_bar = self._calculate_ybar(t)
        x_bar = self._calculate_xbar(t, y_bar, swap, swap_0)

        swap_dsdx = self.swap_pricer.dsdx(swap, x_bar, y_bar, t)

        lambda_s = self.sigma_r.calculate_vola(t=t, x=x_bar) * swap_dsdx/swap_0
        b_s = (swap_0 * self.sigma_r.b_t(t))/((self.sigma_r.alpha_t(t) + self.sigma_r.b_t(t) * x_bar) *swap_dsdx)  \
        + swap_0*self.swap_pricer.d2sdx2(swap, x_bar, y_bar, t)/(math.pow(swap_dsdx, 2))

        return lambda_s, b_s

    def calculate_sigma_0(self, t):
        return self.sigma_r.calculate_vola(x=0, t=t)

    def _calculate_ybar(self, t):

        def integrand(s):
            return np.power(self.calculate_sigma_0(s) * 1/self.g_t(s), 2)

        return np.power(self.g_t(t), 2)*integrate.quad(integrand, 0, t)[0]


    def _calculate_x0(self, t, swap, s0, y_bar):

        def _solve_for_x0(x_bar, swap, s0, t, y_bar):
            return self.swap_pricer.price(swap, x_bar, y_bar, t) - s0

        return fsolve(_solve_for_x0, x0=np.array([0]), args=(swap, s0, t, y_bar))[0]


    def _calculate_xbar(self, t, y_bar, swap, s0):

        x0 = self._calculate_x0(t, swap, s0, y_bar)
        var_s = self._calculate_var_s(t, swap)
        x_bar = x0 + var_s*self.capital_x.d2xds2(swap, x0, y_bar, t)

        return x_bar

    def _calculate_var_s(self, t, swap):

        def integrand(s):
            return np.power(self.swap_pricer.dsdx(swap, 0, 0, s) * self.calculate_sigma_0(s), 2)

        return integrate.quad(integrand, 0, t)[0]


    def _calculate_ksi(self, t, s, swap, s0):

        y_bar = self._calculate_ybar(t)
        x_bar = self._calculate_xbar(t, y_bar, swap, s0)
        dsdx = self.swap_pricer.dsdx(swap, x_bar, y_bar, t)
        d2sdx2 = self.swap_pricer.d2sdx2(swap, x_bar, y_bar, t)
        swap_price = self.swap_pricer.price(swap, x_bar, y_bar, t)

        alpha = 0.5 * d2sdx2
        beta = dsdx - dsdx*x_bar
        gamma = swap_price - dsdx * x_bar + 0.5 * d2sdx2 * math.pow(x_bar, 2) - s

        sqrt_delta = math.sqrt(math.pow(beta, 2) - 4*alpha*gamma)
        res1 = (-beta + sqrt_delta)/(2*alpha)
        res2 = (-beta - sqrt_delta)/(2*alpha)

        abs_dif1 = abs(res1 - x_bar)
        abs_dif2 = abs(res2 - x_bar)

        if abs_dif1>abs_dif2:
            return res2
        else:
            return res1


    def x_bar_simple(self, t, swap_price, swap):
        swap_xy_0 = self.swap_pricer.price(swap, x=0, y=0, t=t)
        return (swap_price - swap_xy_0)/self.swap_pricer.dsdx(swap, x=0, y=0, t=t)