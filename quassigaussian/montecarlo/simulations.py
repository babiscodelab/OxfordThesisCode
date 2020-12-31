import numpy as np
import pandas as pd
from quassigaussian.utils import get_random_number_generator
from quassigaussian.products.pricer import AnnuityPricer, BondPricer
from quassigaussian.products.instruments import Annuity, Bond
from quassigaussian.volatility.local_volatility import LinearLocalVolatility


class ResultSimulatorObj():

    def __init__(self, x, y, time_grid, number_samples, number_time_steps, kappa, local_volatility, measure, random_number_generator_type):

        self.x = x
        self.y = y
        self.time_grid = time_grid
        self.number_samples = number_samples
        self.number_time_steps = number_time_steps
        self.kappa = kappa
        self.local_volatility = local_volatility

        self.x_bar = x.mean(axis=0)
        self.y_bar = y.mean(axis=0)
        self.x_std = x.var(axis=0)
        self.y_std = y.std(axis=0)
        self.measure = measure
        self.random_number_generator_type = random_number_generator_type

        self.res = pd.DataFrame({'time grid': self.time_grid, "x bar mc": self.x_bar, "y bar mc": self.y_bar,
                                 "x std mc": self.x_std, "y std mc": self.y_std})


class ProcessSimulatorQMeasure():

    def __init__(self, number_samples, number_time_steps, dt, random_number_generator_type="normal"):

        self.number_samples = number_samples
        self.number_time_steps = number_time_steps
        self.dt = dt
        self.random_number_generator = get_random_number_generator(random_number_generator_type)
        self.random_number_generator_type = random_number_generator_type
        self.time_grid = np.arange(0, self.number_time_steps + 1) * self.dt
        self.measure = "Risk Neutral"

    def simulate_xy(self, kappa: float, local_volatility: LinearLocalVolatility) -> ResultSimulatorObj:

        random_numbers = self.random_number_generator(self.number_samples, self.number_time_steps)
        x = np.zeros(shape=(self.number_samples, self.number_time_steps + 1))
        y = np.zeros(shape=(self.number_samples, self.number_time_steps + 1))

        for i in np.arange(0, self.number_samples):
            print("Simulation: " + str(i))
            for j in np.arange(0, self.number_time_steps):

                t = self.dt * j
                eta = local_volatility.calculate_vola(t, x[i][j], y[i][j])
                #mu_x = y[i][j] - kappa * x[i][j]
                # x[i][j + 1] = x[i][j] + mu_x * self.dt + eta * random_numbers[i][j] * np.sqrt(self.dt)
                # y[i][j + 1] = y[i][j] + (np.power(eta, 2) - 2 * kappa * y[i][j]) * self.dt

                x[i][j + 1] = x[i][j] + self.get_drift_x(kappa, y[i][j], x[i][j], eta, t) * self.dt + eta * random_numbers[i][j] * np.sqrt(self.dt)
                y[i][j + 1] = y[i][j] + self.get_drift_y(eta, kappa, y[i][j]) * self.dt

        return ResultSimulatorObj(x, y, self.time_grid, self.number_samples, self.number_time_steps, kappa,
                                  local_volatility, self.measure, self.random_number_generator_type)

    def get_drift_x(self, kappa, y_prev, x_prev, eta, t):
        return y_prev - kappa * x_prev

    def get_drift_y(self, eta, kappa, y_prev):
        return np.power(eta, 2) - 2 * kappa * y_prev



class ProcessSimulatorAnnuity(ProcessSimulatorQMeasure):

    def __init__(self, number_samples, number_time_steps, dt, random_number_generator_type, annuity_measure, annuity_pricer: AnnuityPricer):
        super(ProcessSimulatorAnnuity, self).__init__(number_samples, number_time_steps, dt, random_number_generator_type)
        self.annuity_measure = annuity_measure
        self.annuity_pricer = annuity_pricer
        self.measure = self.annuity_measure

    def get_drift_x(self, kappa, y_prev, x_prev, eta, t):
        return super(ProcessSimulatorAnnuity, self).get_drift_x(kappa, y_prev, x_prev, eta, t) + 1 / \
                self.annuity_pricer.annuity_price(t, x_prev, y_prev, self.annuity_measure) * \
                self.annuity_pricer.annuity_dx(t, x_prev, y_prev, kappa, self.annuity_measure) * np.power(eta, 2)


class ProcessSimulatorTerminalMeasure(ProcessSimulatorQMeasure):
    def __init__(self, number_samples, number_time_steps, dt, random_number_generator_type="normal", bond=None, bond_pricer: BondPricer = None):
        super(ProcessSimulatorTerminalMeasure, self).__init__(number_samples, number_time_steps, dt, random_number_generator_type)
        self.bond_pricer = bond_pricer
        self.bond = bond
        self.measure = self.bond

    def get_drift_x(self, kappa, y_prev, x_prev, eta, t):
        return super(ProcessSimulatorTerminalMeasure, self).get_drift_x(kappa, y_prev, x_prev, eta, t) \
               + 1/self.bond_pricer.price(self.bond, x_prev, y_prev, t) * self.bond_pricer.dpdx(self.bond, x_prev, y_prev, t)