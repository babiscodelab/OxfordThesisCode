from quassigaussian.montecarlo.simulations import ProcessSimulatorTerminalMeasure
from quassigaussian.montecarlo.monte_carlo_pricer import monte_carlo_pricer_terminal_measure
from quassigaussian.volatility.local_volatility import LinearLocalVolatility
from quassigaussian.products.instruments import Bond, Swap, Swaption, Annuity
from quassigaussian.montecarlo.control_variate import apply_control_variate, apply_control_variate_annuity
from quassigaussian.products.pricer import BondPricer, SwapPricer, SwaptionPricer, find_implied_black_vola
import pandas as pd
import os
import numpy as np
from qgtests.utis import get_mock_yield_curve_const
from report.directories import output_data_raw, date_timestamp
from report.utils import get_nonexistant_path

output_data_raw_monte_carlo = os.path.join(output_data_raw, "monte_carlo", "swaption")


def mc_swaption_report():

    output_path = os.path.join(output_data_raw_monte_carlo, date_timestamp)
    output_file = os.path.join(output_path, "swaption_price_mc.hdf")
    file_path = get_nonexistant_path(output_file)

    #random_number_generator_type = "sobol"
    random_number_generator_type = "normal"

    curve_rate = 0.06
    kappa_grid = [0.03]

    initial_curve = get_mock_yield_curve_const(rate=curve_rate)

    vola_parameters = [(i, curve_rate, j) for i in [0.05, 0.2, 0.4, 0.5] for j in [0.05, 0.1, 0.3, 0.7]]

    vola_parameters = [(i, curve_rate, j) for i in [0.4] for j in [0.3]]

    vola_grid_df = pd.DataFrame(vola_parameters, columns=["lambda", "alpha", "beta"])

    coupon_grid = [0, +0.0025, -0.0025, +0.005, -0.005, +0.01, -0.01, 0.015, -0.015, 0.02, -0.02, 0.025, -0.025]
    #vola_grid_df = vola_grid_df.iloc[[10]]

    number_paths = np.power(2, 15)
    number_time_steps = np.power(2, 11)
    swap_ls = [(1, 6), (5, 10), (10, 20), (20, 30), (25, 30)]
    swap_ls = [(5, 10),  (10, 20), (20, 30)]

    swap_ls = [(5, 10)]
    #swap_ls = [(1, 11)]

    for number_paths in [np.power(2,12), np.power(2, 13), np.power(2, 14), np.power(2, 15), np.power(2, 16), np.power(2, 17)]:
        for swap_exp_mat in swap_ls:
            print("swap: ", swap_exp_mat)
            expiry, maturity = swap_exp_mat
            for kappa in kappa_grid:
                swap_pricer = SwapPricer(initial_curve, kappa)
                swaption_pricer = SwaptionPricer(swap_pricer)
                swap = Swap(expiry, maturity, 0.5)
                atm_swap_price = swap_pricer.price(swap, 0, 0, 0)
                strike_grid = [atm_swap_price+coupon for coupon in coupon_grid]
                for index, vola_grid_row in vola_grid_df.iterrows():
                    loca_vola = LinearLocalVolatility.from_const(maturity, vola_grid_row["lambda"],
                                                                 vola_grid_row["alpha"], vola_grid_row["beta"])
                    bond_measure = swap.bond_T0
                    process_simulator = ProcessSimulatorTerminalMeasure(number_paths, number_time_steps,
                                                                 expiry / number_time_steps,
                                                                 random_number_generator_type, bond_measure,
                                                                        swap_pricer.bond_pricer, nr_processes=6,
                                                                        n_scrambles=64)

                    result_obj = process_simulator.simulate_xy(kappa, loca_vola, parallel_simulation=True)

                    for strike in strike_grid:
                        swaption = Swaption(expiry, strike, swap)

                        swaption_value_paths = monte_carlo_pricer_terminal_measure(result_obj, swaption, swaption_pricer)
                        last_mc_time = result_obj.time_grid[-1]
                        # swaption_value_paths_cv = apply_control_variate(last_mc_time, result_obj.x[:,-1], result_obj.y[:,-1],
                        #                                 swaption_value_paths, swap.bond_TN, swap_pricer.bond_pricer, swap_pricer.initial_curve)
                        swaption_value_paths_cv2 = apply_control_variate_annuity(last_mc_time, result_obj.x[:, -1],
                                                                        result_obj.y[:, -1], swaption_value_paths,
                                                                        swap.annuity, swap_pricer.annuity_pricer,
                                                                         swap_pricer.annuity_pricer.bond_pricer.initial_curve)

                        swaption_value_mean = swaption_value_paths.mean()
                        std, swaption_value_error = result_obj.calculate_std_error(swaption_value_paths, result_obj.n_scrambles)

                        # swaption_value_mean_cv = swaption_value_paths_cv.mean()
                        # std, swaption_value_error_cv = result_obj.calculate_std_error(swaption_value_paths_cv, result_obj.n_scrambles)

                        swaption_value_mean_cv = swaption_value_paths_cv2.mean()
                        std, swaption_value_error_cv = result_obj.calculate_std_error(swaption_value_paths_cv2,
                                                                                      result_obj.n_scrambles)

                        bond_pricer = swap_pricer.bond_pricer
                        output_data = {"number_paths": number_paths, "number_time_steps": number_time_steps,
                                       "random_number_generator_type": random_number_generator_type, "expiry": expiry,
                                       "maturity": maturity, "strike": strike, "atm strike": atm_swap_price,
                                       "moneyness": strike-atm_swap_price, "vola_lambda": vola_grid_row["lambda"],
                                       "vola_alpha": vola_grid_row["alpha"], "vola_beta": vola_grid_row["beta"],
                                       "curve_rate": curve_rate, "kappa": kappa, "swaption value": swaption_value_mean,
                                       "swaption value error": swaption_value_error,
                                       "swaption value cv": swaption_value_mean_cv,
                                       "swaption value error cv": swaption_value_error_cv,

                                       "implied_vola": find_implied_black_vola(swaption_value_mean, swaption,
                                                                               swap_pricer, bond_pricer),
                                       "implied_vola_max": find_implied_black_vola(swaption_value_mean+swaption_value_error,
                                                                                   swaption, swap_pricer, bond_pricer),
                                       "implied_vola_min": find_implied_black_vola(swaption_value_mean-swaption_value_error,
                                                                                   swaption, swap_pricer, bond_pricer),
                                       "implied_vola_cv": find_implied_black_vola(swaption_value_mean_cv, swaption,
                                                                               swap_pricer, bond_pricer),
                                       "implied_vola_cv_max": find_implied_black_vola(swaption_value_mean_cv + swaption_value_error_cv, swaption,
                                                                               swap_pricer, bond_pricer),
                                       "implied_vola_cv_min": find_implied_black_vola(swaption_value_mean_cv - swaption_value_error_cv, swaption,
                                                                               swap_pricer, bond_pricer)}

                        output_df_new = pd.DataFrame(output_data, index=[0])

                        try:
                            ouput_df_old = pd.read_hdf(file_path, key="output_data")
                        except:
                            ouput_df_old = pd.DataFrame()

                        output_df_new = pd.concat([ouput_df_old, output_df_new])
                        output_df_new.to_hdf(file_path, key="output_data", complevel=9)


if __name__ == "__main__":
    mc_swaption_report()