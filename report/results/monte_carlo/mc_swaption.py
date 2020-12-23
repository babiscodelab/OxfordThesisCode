from quassigaussian.montecarlo.simulations import ProcessSimulator
from quassigaussian.montecarlo.monte_carlo_pricer import monte_carlo_pricer
from quassigaussian.volatility.local_volatility import LinearLocalVolatility, BlackVolatilityModel
from quassigaussian.products.instruments import Bond, Swap, Swaption, Annuity
from quassigaussian.products.pricer import BondPricer, SwapPricer, SwaptionPricer, find_implied_black_vola, AnnuityPricer
import pandas as pd
import os
import numpy as np
from qgtests.utis import get_mock_yield_curve_const
from report.directories import output_data_raw, date_timestamp
from report.utils import get_nonexistant_path

output_data_raw_monte_carlo = os.path.join(output_data_raw, "monte_carlo", "swaption")


def mc_swaption_report():
    output_path = os.path.join(output_data_raw_monte_carlo, date_timestamp)

    expiry_grid = [5]
    maturity_grid = [10]

    curve_rate = 0.06
    kappa_grid = [0.03]

    lambda_grid = [0.05]
    alpha_grid = [0.5]
    beta_grid = [0.5]
    initial_curve = get_mock_yield_curve_const(rate=curve_rate)
    coupon_grid = [0]
    vola_grid_df = pd.DataFrame({"lambda": lambda_grid, "alpha": alpha_grid, "beta": beta_grid})


    number_paths = np.power(2, 14)
    number_time_steps = np.power(2, 8)

    for expiry in expiry_grid:
        for maturity in maturity_grid:
            for kappa in kappa_grid:
                swap_pricer = SwapPricer(initial_curve, kappa)
                swaption_pricer = SwaptionPricer(swap_pricer)
                annuity_pricer = AnnuityPricer(swap_pricer.bond_pricer)
                swap = Swap(expiry, maturity, 0.5)
                atm_swap_price = swap_pricer.price(swap, 0, 0, 0)
                strike_grid = [atm_swap_price+coupon for coupon in coupon_grid]
                for strike in strike_grid:
                    swaption = Swaption(expiry, strike, swap)
                    for index, vola_grid_row in vola_grid_df.iterrows():
                        loca_vola = LinearLocalVolatility.from_const(maturity, vola_grid_row["lambda"], vola_grid_row["alpha"], vola_grid_row["beta"])

                        process_simulator = ProcessSimulator(number_paths, number_time_steps, expiry/number_time_steps, annuity_pricer)
                        result_obj = process_simulator.simulate_xy(kappa, loca_vola, swaption.swap.annuity)
                        swaption_value_paths = monte_carlo_pricer(result_obj, swaption, swaption_pricer)

                        swaption_value_mean = swaption_value_paths.mean()
                        swaption_value_error = swaption_value_paths.std()/np.sqrt(number_paths)
                        implied_black_vola = find_implied_black_vola(swaption_value_mean, swaption, swap_pricer, swap_pricer.bond_pricer)

                        implied_black_vola_upper = find_implied_black_vola(swaption_value_mean + swaption_value_error, swaption, swap_pricer, swap_pricer.bond_pricer)
                        implied_black_vola_lower = find_implied_black_vola(swaption_value_mean - swaption_value_error, swaption, swap_pricer, swap_pricer.bond_pricer)

                        output_data = {
                                    "number_paths": number_paths, "number_time_steps": number_time_steps,
                                    "expiry": expiry, "maturity": maturity, "strike": strike,
                                     "atm strike": atm_swap_price, "moneyness": atm_swap_price - strike,
                                     "vola_lambda": vola_grid_row["lambda"], "vola_alpha": vola_grid_row["alpha"],
                                     "vola_beta": vola_grid_row["beta"], "curve_rate": curve_rate, "kappa": kappa,
                                     "swaption value": swaption_value_mean, "swaption value error": swaption_value_error,
                                     "implied black vola": implied_black_vola,
                                     "implied black vola upper": implied_black_vola_upper,
                                     "implied black vola lower": implied_black_vola_lower}

                        ouput_df = pd.DataFrame(output_data, index=[0])
                        output_file = os.path.join(output_path, "swaption_price_mc.hdf")
                        file_path = get_nonexistant_path(output_file)

                        ouput_df.to_hdf(file_path, key="output_data", complevel=5)

mc_swaption_report()