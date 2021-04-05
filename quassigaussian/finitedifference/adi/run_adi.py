from quassigaussian.finitedifference.adi.douglas_rachford import DouglasRachfordAdi
from quassigaussian.finitedifference.mesher.linear_mesher import Mesher2d
from quassigaussian.finitedifference.mesher.grid_boundaries import calculate_x_boundaries2, calculate_u_boundaries, calculate_y_bar
import numpy as np


class AdiRunner():

    def __init__(self, theta, kappa, initial_curve, local_volatility, mesher: Mesher2d):

        self.theta = theta
        self.kappa = kappa
        self.mesher = mesher
        self.local_volatility = local_volatility
        self.douglas_rachford = DouglasRachfordAdi(theta, self.mesher, initial_curve, kappa, local_volatility)


    def run_adi(self, instrument, instrument_pricer):

        # adjustment because we use transformed to u mesher by removing the drift y_bar
        ymesher = self.mesher.umesh + calculate_y_bar(self.mesher.tmax, self.local_volatility, self.kappa)

        # value of the instrument at maturity
        v_maturity = instrument_pricer.maturity_price(instrument, self.mesher.xmesh, ymesher)

        price = self.douglas_rachford.solve(v_maturity)

        return price