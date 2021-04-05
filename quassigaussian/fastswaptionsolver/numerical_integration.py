import numpy as np
from quassigaussian.products.pricer import SwapPricer, CapitalX
from quassigaussian.fastswaptionsolver.approximation import PiterbargExpectationApproximator
from quassigaussian.parameters.volatility.local_volatility import LinearLocalVolatility
from scipy.integrate import solve_ivp
from abc import ABC
import abc
from quassigaussian.utils import midpoint


class XYApproximators(ABC):

    @abc.abstractmethod
    def calculate_xy(self):
        pass

class PitergargDiscreteXY(XYApproximators):

    def __init__(self, integration_grid_size, swap_pricer: SwapPricer, sigma_r: LinearLocalVolatility, swap):
        self.integration_grid_size = integration_grid_size
        self.g_t = lambda t: np.exp(-swap_pricer.kappa * t)
        self.swap_pricer = swap_pricer
        self.capital_x = CapitalX(swap_pricer)
        self.swap = swap
        self.piterbarg_approx = PiterbargExpectationApproximator(sigma_r, swap_pricer)
        self.time_grid = np.linspace(0, swap.T0, integration_grid_size)


    def calculate_xy(self):

        t0 = self.time_grid[0]
        tn = self.time_grid[-1]
        s0 = self.swap_pricer.price(self.swap, 0, 0, 0)
        x0_guess = 0

        y_int = midpoint(self.piterbarg_approx.ybar_integrand, t0, tn, self.integration_grid_size)
        y_bar = y_int * np.power(self.g_t(self.time_grid), 2)

        var_s = midpoint(self.piterbarg_approx.var_s_integrand, t0, tn, self.integration_grid_size, swap=self.swap)

        x0 = []
        for idx, t in enumerate(self.time_grid):
            x0.append(self.piterbarg_approx.calculate_x0(t, self.swap, s0, y_bar[idx], x0_guess))
            x0_guess = x0[-1]

        xbar = x0 + 0.5 * var_s * self.capital_x.d2xds2(self.swap, x0, y_bar, self.time_grid)


        return xbar, y_bar


    def __str__(self):
        return "piterbarg"



class RungeKuttaApproxXY(XYApproximators):


    def __init__(self,  integration_grid_size, swap_pricer: SwapPricer, sigma_r: LinearLocalVolatility, swap):
        self.sigma_r = sigma_r
        self.swap_pricer = swap_pricer
        self.kappa = self.swap_pricer.kappa
        self.annuity = swap.annuity
        self.annuity_pricer = swap_pricer.annuity_pricer
        self.time_grid = np.linspace(0, swap.T0, integration_grid_size)


    def calculate_xy(self):
        sol = solve_ivp(self.rhs_system, t_span=[self.time_grid[0], self.time_grid[-1]], y0=np.array([0, 0]), t_eval=self.time_grid)
        return sol.y[0], sol.y[1]


    def rhs_system(self, t, xy):

        x = xy[0]
        y = xy[1]

        rhs_x = -self.kappa*x + y + \
        self.annuity_pricer.annuity_dx(t, x, y, self.kappa, self.annuity)*1/self.annuity_pricer.annuity_price(t, x, y, self.annuity)\
                *(np.square(self.sigma_r.calculate_vola(t, x)))
        rhs_y = np.square(self.sigma_r.calculate_vola(t, x)) - 2*self.kappa*y
        return np.array([rhs_x, rhs_y])

    def __str__(self):
        return "rungekutta"