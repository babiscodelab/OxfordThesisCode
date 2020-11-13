import scipy.integrate as integrate
import math
from quassigaussian.instruments import SwapPricer, CapitalX
from scipy.optimize import fsolve
import numpy as np
from quassigaussian.volatility.local_volatility import LinearLocalVolatility

class PiterbargApproximator():

    def __init__(self, g_t, sigma_r: LinearLocalVolatility, swap_pricer: SwapPricer):
        self.g_t = g_t
        self.sigma_r = sigma_r
        self.swap_pricer = swap_pricer
        self.capital_x = CapitalX(swap_pricer)

    def approximate_parameters(self, swap, x, y, t):
        lamba_s = self.calculate_lamba_s(swap, x, y, t)
        b_s = self.calculate_b_s(swap, x, y, t)
        return lamba_s, b_s

    def calculate_lamba_s(self, swap, x, y, t):

        swap_0 = self.swap_pricer.price(swap, 0, 0, 0)
        y_bar = self._calculate_ybar(t)
        x_bar = self._calculate_xbar(t, swap_0, y_bar, swap)

        return self.sigma_r.calculate_vola(t, x_bar) * self.swap_pricer.dsdx(swap, x_bar, y_bar, t)/swap_0

    def calculate_b_s(self, swap, x, y, t):

        swap_0 = self.swap_pricer.price(swap, 0, 0, 0)
        y_bar = self._calculate_ybar(t)
        x_bar = self._calculate_xbar(t, swap_0, y_bar, swap)
        bs = (swap_0 * self.sigma_r.b_t[t])/((self.sigma_r.alpha_t[0] + self.sigma_r.b_t[t] * x_bar)
                                             *self.swap_pricer.dsdx(swap, x_bar, y_bar, t))  \
        + swap_0*math.pow(self.swap_pricer.d2sdx2(swap, x_bar, y_bar, t), 2)/(math.pow(self.swap_pricer.dsdx(swap, x_bar, y_bar, t), 2))

        return bs

    def calculate_sigma_0(self, t):
        return self.sigma_r.calculate_vola(x=0, t=t)

    def _calculate_ybar(self, t):

        def integrand(s):
            return math.pow(self.calculate_sigma_0(s) * 1/self.g_t(s), 2)

        return self.g_t(t)*integrate.quad(integrand, 0, 1)


    def _calculate_x0(self, t, s0, y_bar):

        def _solve_for_x0(x_bar, swap, t, y_bar, s0):
            self.swap_pricer.price(swap, t, x_bar, y_bar) - s0

        return fsolve(_solve_for_x0, x0=np.array([0]), args=(t, s0, y_bar))[0]


    def _calculate_xbar(self, t, s0, y_bar, swap):

        x0 = self._calculate_x0(t, s0, y_bar)
        var_s = self._calculate_var_s(t, swap)
        x_bar = x0 + var_s*self.capital_x.d2ds2(swap, x0, y_bar, t)

        return x_bar

    def _calculate_var_s(self, t, swap):
        def integrand(s):
            return math.pow(self.swap_pricer.dsdx(swap, 0, 0, s) * self.calculate_sigma_0(s), 2)
        return integrate.quad(integrand, 0, t)


    def _calculate_ksi(self, t, s, swap, s0):

        y_bar = self._calculate_ybar(t)
        x_bar = self._calculate_xbar(t, s0, y_bar, swap)
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