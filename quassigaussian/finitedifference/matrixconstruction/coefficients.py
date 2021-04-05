from quassigaussian.curves.libor import LiborCurve
from quassigaussian.parameters.volatility.local_volatility import LinearLocalVolatility
import numpy as np

class CoefficientConstruction():

    def __init__(self, xmesh, umesh, initial_curve: LiborCurve, kappa, local_volatility: LinearLocalVolatility):

        """
        Quasi Gaussian construct coefficients
        :param xmesh:
        :param umesh:
        :param initial_curve:
        :param kappa:
        :param local_volatility:
        """
        self.xmesh = xmesh
        self.umesh = umesh
        self.initial_curve = initial_curve
        self.kappa = kappa
        self.local_volatility = local_volatility

    def update_coefficients(self, t):
        self.mu_x = self.mu_x_calculate(t)
        self.eta_sq = self.eta_sq_calculate(t)
        self.mu_u = self.mu_u_calculate(self.eta_sq)
        self.r = self.r_calculate(t)

    def mu_x_calculate(self, t):
        y_bar = np.power(self.local_volatility.calculate_vola(t, 0, 0), 2) / (2 * self.kappa) * (1 - np.exp(-2 * self.kappa * t))
        return (self.umesh - self.kappa * self.xmesh + y_bar)

    def eta_sq_calculate(self, t):
        y_bar = np.power(self.local_volatility.calculate_vola(t, 0, 0), 2) / (2 * self.kappa) * (1 - np.exp(-2 * self.kappa * t))
        return np.square(self.local_volatility.calculate_vola(t, self.xmesh, self.umesh+y_bar))

    def mu_u_calculate(self, eta_sq):
        return eta_sq - 2 * self.kappa * self.umesh - np.square(self.local_volatility.calculate_vola(0, 0, 0))

    def r_calculate(self, t):
        return self.initial_curve.get_inst_forward(t) + self.xmesh[:, 0]
        #return 0.04 + self.xmesh[:, 0]*1



class DummyCoefficients():

    def __init__(self, xmesh, ymesh, initial_curve, kappa):
        """
        Dummy coefficients. Only for debugging and testing!!!!!!!!!!!
        :param xmesh:
        :param ymesh:
        :param initial_curve:
        :param kappa:
        """
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

