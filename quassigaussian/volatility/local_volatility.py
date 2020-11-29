from scipy.interpolate import interp1d
from abc import ABC, abstractmethod
import numpy as np
from quassigaussian.products.instruments import Swap
from quassigaussian.products.pricer import SwapPricer

class LocalVolatility(ABC):

    @abstractmethod
    def calculate_vola(self, t, x, y):
        pass

    @abstractmethod
    def d_vola_dx(selfs, t, x, y):
        pass

class LinearLocalVolatility():

    def __init__(self, lambda_t: callable, alpha_t: callable, b_t: callable):
        self.lambda_t = lambda_t
        self.alpha_t = alpha_t
        self.b_t = b_t

    def calculate_vola(self, t: float, x: float, y=0):
        return self.lambda_t(t) * (self.alpha_t(t) + x * self.b_t(t))

    def d_vola_dx(self, t, x, y=0):
        return self.lambda_t(t) * self.alpha_t(t)

    @classmethod
    def from_const(cls, maturity, lambda_const, alpha_const, b_const):

        x = np.arange(0, maturity+1)
        y = np.ones(maturity+1)

        lambda_t = interp1d(x, y*lambda_const, kind='previous')
        alpha_t = interp1d(x, y*alpha_const, kind='previous')
        b_t = interp1d(x, y*b_const, kind='previous')

        return cls(lambda_t, alpha_t, b_t)



class BlackVolatilityModel(LocalVolatility):

    def __init__(self, logvola: float, swap: Swap, swap_pricer: SwapPricer):
        self.logvola = logvola
        self.swap = swap
        self.swap_pricer = swap_pricer

    def calculate_vola(self, t, x, y):
        return self.logvola*self.swap_pricer.price(self.swap, x, y, t)/self.swap_pricer.dsdx(self.swap, x, y, t)

    def d_vola_dx(self, t, x, y):
        return self.logvola*(1-self.swap_pricer.d2sdx2(self.swap, x, y, t)/np.power(self.swap_pricer.dsdx(self.swap, x, y, t), 2))