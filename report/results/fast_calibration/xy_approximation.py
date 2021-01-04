from quassigaussian.montecarlo.simulations import ProcessSimulatorAnnuity

import pandas as pd
from quassigaussian.volatility.local_volatility import LinearLocalVolatility
from quassigaussian.products.instruments import Bond, Swap, Swaption
from quassigaussian.products.pricer import BondPricer, SwapPricer
import os
from qgtests.utis import get_mock_yield_curve_const
from report.directories import output_data_raw, date_timestamp
from report.utils import get_nonexistant_path
from quassigaussian.fastcalibration.approximation import PiterbargExpectationApproximator, RungeKuttaApproximator

output_data_raw_approx = os.path.join(output_data_raw, "approximation", "xy_approx")
NR_PROCESSES = 3

def compare_approximated_values():

    output_path = os.path.join(output_data_raw_approx, date_timestamp, "result")

    curve_rate = 0.06
    random_number_generator_type = "normal"
    kappa_grid = [0.03]
    initial_curve = get_mock_yield_curve_const(rate=curve_rate)

    vola_parameters = [(i, curve_rate, j) for i in [0.05, 0.1, 0.25, 0.45] for j in [0.05, 0.1, 0.3, 0.7]]
    vola_grid_df = pd.DataFrame(vola_parameters, columns=["lambda", "alpha", "beta"])

    output_path = get_nonexistant_path(output_path)

    number_samples = 1024
    number_steps = 128

    swap_ls = [(1, 6), (5, 10), (10, 20), (20, 30), (25, 30)]

    #vola_grid_df = vola_grid_df[-1:]

    #swap_ls = swap_ls[1:2]
    #vola_grid_df = vola_grid_df.iloc[0:1]

    for swap_T0_TN in swap_ls:
        print(swap_T0_TN)
        T0, TN = swap_T0_TN
        for kappa in kappa_grid:
            swap_pricer = SwapPricer(initial_curve, kappa)
            swap = Swap(T0, TN, 1/2)
            for index, vola_grid_row in vola_grid_df.iterrows():
                dt = T0/number_steps
                loca_vola = LinearLocalVolatility.from_const(swap.TN, vola_grid_row["lambda"], vola_grid_row["alpha"],
                                                             vola_grid_row["beta"])
                # loca_vola = LinearLocalVolatility.from_swap_const(swap, swap_pricer, vola_grid_row["lambda"], vola_grid_row["beta"])

                res_annuity_mc = simulate_xy(kappa, loca_vola, number_samples, number_steps, dt, random_number_generator_type,
                                             swap.annuity, swap_pricer.annuity_pricer)
                res_approx_xy = calculate_approximation_xy(res_annuity_mc.time_grid, swap, loca_vola, swap_pricer)
                runge_kutta_approx_df = runge_kutta_solution(res_annuity_mc.time_grid, swap, loca_vola, swap_pricer)

                approximation_comparison_df = pd.merge(res_approx_xy, res_annuity_mc.res, on="time grid")
                approximation_comparison_df = pd.merge(approximation_comparison_df, runge_kutta_approx_df, on="time grid")

                meta_data = pd.DataFrame({"kappa": [kappa], "swap_T0": [T0], "swap_TN": [TN],
                             "lambda value": [float(loca_vola.lambda_t(0))],
                             "alpha": [float(loca_vola.alpha_t(0))], "beta": [float(loca_vola.b_t(0))],
                             "number samples": [number_samples], "number steps": [number_steps],
                             "curve rate": [curve_rate], 'random number generator type': random_number_generator_type})
                output_file = os.path.join(output_path, "xy_approx.hdf")
                file_path = get_nonexistant_path(output_file)

                approximation_comparison_df.to_hdf(file_path, key="approximation_comparison", complevel=5)
                meta_data.to_hdf(file_path, key="meta_data", complevel=5)



def simulate_xy(kappa, local_volatility, number_samples, number_steps, dt, random_number_generator_type, annuity, annuity_pricer):
    x_simulator = ProcessSimulatorAnnuity(number_samples, number_steps, dt, random_number_generator_type, annuity, annuity_pricer, nr_processes=NR_PROCESSES)
    res_annuity = x_simulator.simulate_xy(kappa=kappa, local_volatility=local_volatility, parallel_simulation=True)
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


def runge_kutta_solution(time_grid, swap, loca_vola, swap_pricer):

    rk_approx = RungeKuttaApproximator(loca_vola, swap_pricer, swap.annuity, swap_pricer.annuity_pricer)
    res = rk_approx.approximate_x_y(time_grid)

    df = pd.DataFrame({"x_runge_kutta": res.y[0],
                  "y_runge_kutta": res.y[1],
                  "time grid": time_grid})

    return df


if __name__ == "__main__":
    compare_approximated_values()