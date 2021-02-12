from quassigaussian.finitedifference.adi.run_adi import AdiRunner
from quassigaussian.volatility.local_volatility import LinearLocalVolatility, BlackVolatilityModel
from quassigaussian.products.instruments import Bond, Swap, Swaption
from quassigaussian.products.pricer import BondPricer, SwapPricer, SwaptionPricer, find_implied_black_vola
from quassigaussian.finitedifference.mesher.grid_boundaries import calculate_x_boundaries2, calculate_y_boundaries, calculate_x_boundaries3
from quassigaussian.finitedifference.mesher.linear_mesher import Mesher2d, extract_x0_result
import pandas as pd
import os
from qgtests.utis import get_mock_yield_curve_const
from report.directories import output_data_raw, date_timestamp
from report.utils import get_nonexistant_path

output_data_raw_finite_difference = os.path.join(output_data_raw, "finite_difference", "swaption")


def adi_swaption_report():

    output_path = os.path.join(output_data_raw_finite_difference, date_timestamp)

    curve_rate = 0.06
    kappa_grid = [0.03]
    theta = 1/2

    initial_curve = get_mock_yield_curve_const(rate=curve_rate)

    finite_difference_parameter = [(100, 150, 10), (400, 800, 60)]

    #finite_difference_parameter = [(150, 200, 40)]
    #finite_difference_parameter = [(400, 800, 60)]

    finite_difference_parameter = [(150, 300, 200)]


    finite_difference_grid_df = pd.DataFrame(finite_difference_parameter, columns=["t_grid_size", "x_grid_size", "y_grid_size"])
    output_path = get_nonexistant_path(output_path)
    vola_parameters = [(i, curve_rate, j) for i in [0.5] for j in [0.05]]
    vola_grid_df = pd.DataFrame(vola_parameters, columns=["lambda", "alpha", "beta"])

    #coupon_grid = [0, +0.0025, -0.0025, +0.005, -0.005, +0.01, -0.01, 0.015, -0.015, 0.02, -0.02, 0.025, -0.025]

    #swap_ls = [(1, 6), (5, 10), (10, 20), (20, 30), (25, 30)]

    swap_ls = [(20, 30)]
    coupon_grid = [0, +0.005, -0.005, +0.01, -0.01, 0.015, -0.015]

    #swap_ls = [(5, 10)]
    #finite_difference_grid_df = finite_difference_grid_df[:-1]
    #vola_grid_df = vola_grid_df.iloc[9:10]

    for swap_exp_mat in swap_ls:
        expiry, maturity = swap_exp_mat
        for kappa in kappa_grid:
            swap_pricer = SwapPricer(initial_curve, kappa)
            swaption_pricer = SwaptionPricer(swap_pricer)
            swap = Swap(expiry, maturity, 0.5)
            atm_swap_price = swap_pricer.price(swap, 0, 0, 0)
            strike_grid = [atm_swap_price+coupon for coupon in coupon_grid]
            for strike in strike_grid:
                swaption = Swaption(expiry, strike, swap)
                for index, vola_grid_row in vola_grid_df.iterrows():
                    loca_vola = LinearLocalVolatility.from_const(maturity, vola_grid_row["lambda"], vola_grid_row["alpha"], vola_grid_row["beta"])
                    for index, finite_difference_grid_row in finite_difference_grid_df.iterrows():

                        x_grid_size = finite_difference_grid_row["x_grid_size"]
                        y_grid_size = finite_difference_grid_row["y_grid_size"]
                        t_grid_size = finite_difference_grid_row["t_grid_size"]

                        t_min = 0
                        t_max = expiry

                        x_min, x_max = calculate_x_boundaries2(t_max, loca_vola, alpha=3)
                        x_min, x_max = calculate_x_boundaries3(expiry, kappa, loca_vola, alpha=3)
                        y_min, y_max = calculate_y_boundaries(t_max, kappa, loca_vola, alpha=3)

                        mesher = Mesher2d()
                        mesher.create_mesher_2d(t_min, t_max, t_grid_size, x_min, x_max, x_grid_size, y_min, y_max,
                                                y_grid_size)

                        adi_runner = AdiRunner(theta, kappa, initial_curve, loca_vola, mesher)

                        swaption_t0 = pd.DataFrame(adi_runner.run_adi(swaption, swaption_pricer))

                        output_file = os.path.join(output_path, "swaption_price_fd.hdf")
                        file_path = get_nonexistant_path(output_file)

                        swaption_t0_x0_y0 = extract_x0_result(swaption_t0.values, mesher.xgrid, mesher.ygrid)
                        implied_black_vola = find_implied_black_vola(swaption_t0_x0_y0, swaption, swap_pricer, swap_pricer.bond_pricer)

                        meta_data = {"expiry": expiry, "maturity": maturity, "strike": strike,
                                     "atm strike": atm_swap_price, "moneyness": strike - atm_swap_price,
                                     "x_grid_size": int(x_grid_size), "y_grid_size": int(y_grid_size),
                                      "t_grid_size": int(t_grid_size),
                                     "vola_lambda": vola_grid_row["lambda"], "vola_alpha": vola_grid_row["alpha"],
                                     "vola_beta": vola_grid_row["beta"], "curve_rate": curve_rate, "kappa": kappa,
                                     "swaption_value": swaption_t0_x0_y0, "implied_black_vola": implied_black_vola}

                        meta_data = pd.DataFrame(meta_data, index=[0])
                        swaption_t0.to_hdf(file_path, key="data", complevel=5)
                        meta_data.to_hdf(file_path, key="metadata", complevel=5)

                        pd.DataFrame(mesher.xmesh).to_hdf(file_path, key='xmesh', complevel=5)
                        pd.DataFrame(mesher.ymesh).to_hdf(file_path, key='ymesh', complevel=5)

                        pd.DataFrame(mesher.xgrid).to_hdf(file_path, key='xgrid', complevel=5)
                        pd.DataFrame(mesher.ygrid).to_hdf(file_path, key='ygrid', complevel=5)

if __name__ == "__main__":
    adi_swaption_report()