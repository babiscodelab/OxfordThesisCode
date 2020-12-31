
from quassigaussian.montecarlo.simulations import ProcessSimulatorAnnuity

import pandas as pd
from quassigaussian.volatility.local_volatility import LinearLocalVolatility, BlackVolatilityModel
from quassigaussian.products.instruments import Bond, Swap, Swaption
from quassigaussian.products.pricer import BondPricer, SwapPricer, SwaptionPricer, find_implied_black_vola
import os
from qgtests.utis import get_mock_yield_curve_const
from report.directories import output_data_raw, date_timestamp
from report.utils import get_nonexistant_path
from quassigaussian.fastcalibration.approximation import PiterbargExpectationApproximator

output_data_raw_approx = os.path.join(output_data_raw, "approximation", "xy_approx")

def compare_approximated_values():

    output_path = os.path.join(output_data_raw_approx, date_timestamp)

    curve_rate = 0.06
    random_number_generator_type = "normal"
    kappa_grid = [0.03]
    lambda_grid = [0.05]
    alpha_grid = [0.5]
    beta_grid = [0.5]
    initial_curve = get_mock_yield_curve_const(rate=curve_rate)

    vola_grid_df = pd.DataFrame({"lambda": lambda_grid, "alpha": alpha_grid, "beta": beta_grid})

    output_path = get_nonexistant_path(output_path)

    number_samples = 100
    number_steps = 10

    T0 = 5
    TN = 10


    for kappa in kappa_grid:
        swap_pricer = SwapPricer(initial_curve, kappa)
        swap = Swap(T0, TN, 1/2)
        for index, vola_grid_row in vola_grid_df.iterrows():
            dt = T0/number_steps
            loca_vola = LinearLocalVolatility.from_const(swap.TN, vola_grid_row["lambda"], vola_grid_row["alpha"],
                                                         vola_grid_row["beta"])
            res_annuity_mc = simulate_xy(kappa, loca_vola, number_samples, number_steps, dt, random_number_generator_type,
                                         swap.annuity, swap_pricer.annuity_pricer)
            res_approx_xy = calculate_approximation_xy(res_annuity_mc.time_grid, swap, loca_vola, swap_pricer)
            approximation_comparison_df = pd.merge(res_approx_xy, res_annuity_mc.res, on="time grid")
            meta_data = pd.DataFrame({"kappa": [kappa], "swap_T0": [T0], "swap_TN": [TN], "lambda value": [vola_grid_row["lambda"]],
                         "alpha": [vola_grid_row['alpha']], "beta": [vola_grid_row["beta"]],
                         "number samples": [number_samples], "number steps": [number_steps],
                         "curve rate": [curve_rate], 'random number generator type': random_number_generator_type})
            output_file = os.path.join(output_path, "xy_approx.hdf")
            file_path = get_nonexistant_path(output_file)

            approximation_comparison_df.to_hdf(file_path, key="approximation_comparison", complevel=5)
            meta_data.to_hdf(file_path, key="meta_data", complevel=5)



def simulate_xy(kappa, local_volatility, number_samples, number_steps, dt, random_number_generator_type, annuity, annuity_pricer):
    x_simulator = ProcessSimulatorAnnuity(number_samples, number_steps, dt, random_number_generator_type, annuity, annuity_pricer)
    res_annuity = x_simulator.simulate_xy(kappa=kappa, local_volatility=local_volatility)
    return res_annuity



def calculate_approximation_xy(time_grid, swap, loca_vola, swap_pricer):

    s0 = swap_pricer.price(swap, 0, 0, 0)
    exp_approx = PiterbargExpectationApproximator(loca_vola, swap_pricer)
    x0_guess = 0
    y_bar = []
    x_bar = []
    for t in time_grid:
        y_bar.append(exp_approx.ybar_formula(t))
        x_bar.append(exp_approx.xbar_formula(t, y_bar[-1], swap, s0, x0_guess))
        x0_guess = x_bar

    res = pd.DataFrame({'time grid': time_grid, "x bar approx": x_bar, "y bar approx": y_bar})
    return res


compare_approximated_values()