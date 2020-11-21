from quassigaussian.curves.libor import LiborCurve
from quassigaussian.volatility.local_volatility import LinearLocalVolatility
import numpy as np

class CoefficientConstruction():

    def __init__(self, xmesh, ymesh, initial_curve: LiborCurve, kappa, local_volatility: LinearLocalVolatility):
        self.xmesh = xmesh
        self.ymesh = ymesh
        self.initial_curve = initial_curve
        self.kappa = kappa
        self.local_volatility = local_volatility

    def update_coefficients(self, t):
        self.mu_x = self.mu_x_calculate()
        self.eta_sq = self.eta_sq_calculate(t)
        self.mu_y = self.mu_y_calculate(self.eta_sq)
        self.r = self.r_calculate(t)

    def mu_x_calculate(self):
        return (self.ymesh - self.kappa * self.xmesh)

    def eta_sq_calculate(self, t):
        return self.local_volatility.calculate_vola(t, self.xmesh)

    def mu_y_calculate(self, eta_sq):
        return eta_sq - 2*self.kappa*self.ymesh

    def r_calculate(self, t):
        return 0.04 + self.xmesh[:, 0]



class DummyCoefficients():

    def __init__(self, xmesh, ymesh, initial_curve, kappa):
        self.xmesh = xmesh
        self.ymesh = ymesh
        self.initial_curve = initial_curve
        self.kappa = kappa

    def update_coefficients(self, t):
        self.mu_x = self.mu_x_calculate()
        self.eta_sq = self.eta_sq_calculate(t)
        self.mu_y = self.mu_y_calculate(self.eta_sq)
        self.r = self.r_calculate(t)

    def mu_x_calculate(self):
        return 1 * np.ones(self.xmesh.shape)
        # return (self.ymesh - self.kappa * self.xmesh)

    def eta_sq_calculate(self, t):
        return 0.001 * np.ones(self.xmesh.shape)

    def mu_y_calculate(self, eta_sq):
        return 2 * np.ones(self.xmesh.shape)
        # return eta_sq - 2*self.kappa*self.ymesh

    def r_calculate(self, t):
        return self.xmesh[:, 0] * 0 + 4
        # return self.xmesh[:, 0] + 0.02

