from quassigaussian.finitedifference.adi.douglas_rachford import DouglasRachfordAdi
from quassigaussian.finitedifference.mesher.linear_mesher import Mesher2d
from quassigaussian.finitedifference.mesher.grid_boundaries import calculate_x_boundaries2, calculate_y_boundaries
import numpy as np


class AdiRunner():

    def __init__(self, theta, kappa, initial_curve, local_volatility, tmin, tmax, t_grid_size, x_grid_size, y_grid_size):

        self.theta = theta

        x_min, x_max = calculate_x_boundaries2(tmax, local_volatility)
        y_min, y_max = calculate_y_boundaries(tmax, kappa, local_volatility)

        self.mesher = Mesher2d()
        self.mesher.create_mesher_2d(tmin, tmax, t_grid_size, x_min, x_max, x_grid_size, y_min, y_max, y_grid_size)

        self.douglas_rachford = DouglasRachfordAdi(theta, self.mesher, initial_curve, kappa, local_volatility)


    def run_adi(self, instrument, instrument_pricer):

        # value of the instrument at maturity
        v_maturity = instrument_pricer.maturity_price(instrument, self.mesher.xmesh, self.mesher.ymesh)

        price = self.douglas_rachford.solve(v_maturity)
        return price