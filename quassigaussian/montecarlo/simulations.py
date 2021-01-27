import numpy as np
import pandas as pd
from quassigaussian.utils import get_random_number_generator
from quassigaussian.products.pricer import AnnuityPricer, BondPricer
from quassigaussian.products.instruments import Annuity, Bond
from quassigaussian.volatility.local_volatility import LinearLocalVolatility
import pickle
from quassigaussian.executor import Executor
from quassigaussian.utils import calculate_G

class ResultSimulatorObj():

    def __init__(self, x, y, time_grid, number_samples, number_time_steps, kappa, local_volatility: LinearLocalVolatility, measure, random_number_generator_type):

        self.x = x
        self.y = y
        self.time_grid = time_grid
        self.number_samples = number_samples
        self.number_time_steps = number_time_steps
        self.kappa = kappa
        self.local_volatility = local_volatility

        self.x_bar = x.mean(axis=0)
        self.y_bar = y.mean(axis=0)
        self.n_scrambles = 32

        self.x_std, self.x_error = self.calculate_std_error(x, self.n_scrambles)
        self.y_std, self.y_error = self.calculate_std_error(y, self.n_scrambles)

        self.measure = measure
        self.random_number_generator_type = random_number_generator_type

        self.meta_data = {'time grid': self.time_grid, "kappa": kappa, "lambda": float(local_volatility.lambda_t(0)),
                          'beta': float(local_volatility.b_t(0)), "alpha": float(local_volatility.alpha_t(0)),
                          "random_number_generator_type": random_number_generator_type}

        self.res = pd.DataFrame({'time grid': self.time_grid, "x bar mc": self.x_bar, "y bar mc": self.y_bar,
                                 "x std mc": self.x_std, "y std mc": self.y_std})


    @staticmethod
    def calculate_std_error(simulations, n_scrambles):

        number_samples = simulations.shape[0]
        paths_scramble = number_samples/n_scrambles
        x_tmp = []

        for i in range(0, n_scrambles):
            from_i = int(i*paths_scramble)
            to_i = int((i+1)*paths_scramble)
            x_tmp.append(np.mean(simulations[from_i:to_i], axis=0))

        std = np.array(x_tmp).std(axis=0)
        error = std/np.sqrt(n_scrambles)

        return std, 3*error


    def store_data(self, file):
        with open(file) as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


class ProcessSimulatorQMeasure():

    def __init__(self, number_samples, number_time_steps, dt, random_number_generator_type="normal", nr_processes=3, n_scrambles=32):

        self.number_samples = number_samples
        self.number_time_steps = number_time_steps
        self.dt = dt
        self.random_number_generator = get_random_number_generator(random_number_generator_type)
        self.random_number_generator_type = random_number_generator_type
        self.time_grid = np.arange(0, self.number_time_steps + 1) * self.dt
        self.measure = "Risk Neutral"
        self.nr_processes = nr_processes
        self.n_scrambles = n_scrambles



    def simulate_xy(self, kappa: float, local_volatility: LinearLocalVolatility, parallel_simulation=False) -> ResultSimulatorObj:

        random_numbers = self.random_number_generator(self.number_samples, self.number_time_steps, self.dt*self.number_time_steps, self.n_scrambles)
        chunksize = 100
        futures = []
        if parallel_simulation:
            executor = Executor(use_dask=False, nr_processes=self.nr_processes)
            random_numbers_chunk_generator = (random_numbers[i:i+chunksize, :] for i in range(0, self.number_samples, chunksize))
            j = 0
            for random_numbers_chunk in random_numbers_chunk_generator:
                futures.append(executor.submit(self.simulate_process, kappa, local_volatility, random_numbers_chunk, j))
                j = j+chunksize
            executor.await_futures(futures)

            all_x = []
            all_y = []
            for fut in futures:
                x, y = fut.result()
                all_x.append(x)
                all_y.append(y)
            all_x = np.concatenate(all_x)
            all_y = np.concatenate(all_y)
            return ResultSimulatorObj(all_x, all_y, self.time_grid, self.number_samples, self.number_time_steps, kappa,
                                      local_volatility, self.measure, self.random_number_generator_type)

        else:
            x, y = self.simulate_process(kappa, local_volatility, random_numbers)
            return ResultSimulatorObj(x, y, self.time_grid, self.number_samples, self.number_time_steps, kappa,
                                      local_volatility, self.measure, self.random_number_generator_type)

    def simulate_process(self, kappa: float, local_volatility: LinearLocalVolatility, random_numbers, chunk_sim=0):

        number_samples, number_time_steps = random_numbers.shape

        x = np.zeros(shape=(number_samples, number_time_steps + 1))
        y = np.zeros(shape=(number_samples, number_time_steps + 1))

        for i in np.arange(0, number_samples):
            print("Simulation: " + str(i+chunk_sim))
            for j in np.arange(0, number_time_steps):
                t = self.dt * j
                eta = local_volatility.calculate_vola(t, x[i][j], y[i][j])
                # mu_x = y[i][j] - kappa * x[i][j]
                # x[i][j + 1] = x[i][j] + mu_x * self.dt + eta * random_numbers[i][j] * np.sqrt(self.dt)
                # y[i][j + 1] = y[i][j] + (np.power(eta, 2) - 2 * kappa * y[i][j]) * self.dt

                x[i][j + 1] = x[i][j] + self.get_drift_x(kappa, y[i][j], x[i][j], eta, t) * self.dt + eta * \
                              random_numbers[i][j]
                y[i][j + 1] = y[i][j] + self.get_drift_y(eta, kappa, y[i][j]) * self.dt
        return x, y



    def get_drift_x(self, kappa, y_prev, x_prev, eta, t):
        return y_prev - kappa * x_prev

    def get_drift_y(self, eta, kappa, y_prev):
        return np.power(eta, 2) - 2 * kappa * y_prev



class ProcessSimulatorAnnuity(ProcessSimulatorQMeasure):

    def __init__(self, number_samples, number_time_steps, dt, random_number_generator_type, annuity_measure, annuity_pricer: AnnuityPricer, nr_processes=3, n_scrambles=32):
        super(ProcessSimulatorAnnuity, self).__init__(number_samples, number_time_steps, dt, random_number_generator_type, nr_processes, n_scrambles)
        self.annuity_measure = annuity_measure
        self.annuity_pricer = annuity_pricer
        self.measure = self.annuity_measure

    def get_drift_x(self, kappa, y_prev, x_prev, eta, t):
        return super(ProcessSimulatorAnnuity, self).get_drift_x(kappa, y_prev, x_prev, eta, t) + 1 / \
                self.annuity_pricer.annuity_price(t, x_prev, y_prev, self.annuity_measure) * \
                self.annuity_pricer.annuity_dx(t, x_prev, y_prev, kappa, self.annuity_measure) * np.power(eta, 2)


class ProcessSimulatorTerminalMeasure(ProcessSimulatorQMeasure):
    def __init__(self, number_samples, number_time_steps, dt, random_number_generator_type="normal",
                 bond:Bond =None, bond_pricer: BondPricer = None, nr_processes=3, n_scrambles=32):
        super(ProcessSimulatorTerminalMeasure, self).__init__(number_samples, number_time_steps, dt, random_number_generator_type, nr_processes, n_scrambles)
        self.bond_pricer = bond_pricer
        self.bond = bond
        self.measure = self.bond

    def get_drift_x(self, kappa, y_prev, x_prev, eta, t):
        return super(ProcessSimulatorTerminalMeasure, self).get_drift_x(kappa, y_prev, x_prev, eta, t) - \
               calculate_G(kappa, t, self.bond.maturity) * np.power(eta, 2)