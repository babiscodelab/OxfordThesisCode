import numpy as np
from scipy.interpolate import interp1d
from abc import ABC, abstractmethod
import pandas as pd
from scipy.misc import derivative

class Curve(ABC):

    @abstractmethod
    def get_discount(self, t):
        pass

    def get_inst_forward(self, t):
        pass


class LiborCurve(Curve):

    def __init__(self, time_grid, disc_grid):
        self.disc_grid = disc_grid
        self.time_grid = time_grid
        self.interp_rate_func = self.construct_libor_curve_func(disc_grid, time_grid)
        self.log_disc = interp1d(time_grid, np.log(disc_grid), fill_value="extrapolate")

    def get_discount(self, t):
        return np.exp(-self.interp_rate_func(t) * t)

    def construct_libor_curve_func(self, disc_grid, time_grid):
        rate = - np.log(disc_grid)/time_grid
        return interp1d(time_grid, rate, fill_value="extrapolate")

    def get_inst_forward(self, t):
        return -derivative(self.log_disc, x0=t)

    @classmethod
    def from_file(self, file_path, date):

        df = pd.read_csv(file_path)
        df = df.loc[df["CURVE_DATE"] == date]
        time_grid = df["CURVE_OFFSET"]/365
        disc_grid = df["CURVE_BID"]

        return LiborCurve(time_grid, disc_grid)

    @classmethod
    def from_constant_rate(cls, rate=0.04):

        time_grid = np.arange(1, 50)
        disc_grid = np.exp(-rate*time_grid)

        return LiborCurve(time_grid, disc_grid)