import numpy as np
from quassigaussian.utils import generate_random_numbers
from quassigaussian.products.pricer import AnnuityPricer
from quassigaussian.products.instruments import Annuity
from quassigaussian.volatility.local_volatility import LinearLocalVolatility


class ResultSimulatorObj():

    def __init__(self, x, y, time_grid, number_samples, number_time_steps, kappa, local_volatility, annuity_measure=None):

        self.x = x
        self.y = y
        self.time_grid = time_grid
        self.number_samples = number_samples
        self.numer_time_steps = number_time_steps
        self.kappa = kappa
        self.local_volatility = local_volatility

        self.x_bar = x.mean(axis=0)
        self.y_bar = y.mean(axis=0)
        self.annuity_measure = annuity_measure



class ProcessSimulator():

    def __init__(self, number_samples, number_time_steps, dt, annuity_pricer: AnnuityPricer = None):

        self.number_samples = number_samples
        self.number_time_steps = number_time_steps
        self.dt = dt
        self.annuity_pricer = annuity_pricer
        self.time_grid = np.arange(0, self.number_time_steps+1) * self.dt


    def simulate_xy(self, kappa: float, local_volatility: LinearLocalVolatility, annuity_measure : Annuity = None):

        random_numbers = generate_random_numbers(self.number_samples, self.number_time_steps)
        x = np.zeros(shape=(self.number_samples, self.number_time_steps+1))
        y = np.zeros(shape=(self.number_samples, self.number_time_steps+1))

        for i in np.arange(0, self.number_samples):
            for j in np.arange(0, self.number_time_steps):

                t = self.dt * j
                eta = local_volatility.calculate_vola(t, x[i][j])
                mu_x = y[i][j] - kappa * x[i][j]

                if annuity_measure:
                    mu_x += 1/self.annuity_pricer.annuity_price(t, x[i][j], y[i][j], annuity_measure) * \
                            self.annuity_pricer.annuity_dx(t, x[i][j], y[i][j], kappa, annuity_measure) * eta

                x[i][j+1] = x[i][j] + mu_x * self.dt + eta * random_numbers[i][j]
                y[i][j+1] = y[i][j] + (np.power(eta, 2) - 2*kappa*y[i][j]) * self.dt

        return ResultSimulatorObj(x, y, self.time_grid, self.number_samples, self.number_time_steps, kappa, local_volatility)