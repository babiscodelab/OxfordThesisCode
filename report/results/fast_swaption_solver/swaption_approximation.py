from quassigaussian.parameters.volatility.local_volatility import LinearLocalVolatility
from quassigaussian.products.instruments import Swap, Swaption
from quassigaussian.curves.libor import LiborCurve
import pandas as pd
import os
from report.directories import output_data_raw, date_timestamp
from quassigaussian.products.pricer import  SwapPricer
from report.utils import get_nonexistant_path
from quassigaussian.fastswaptionsolver.discrete_averaging import DiscreteParameterAveraging
from quassigaussian.fastswaptionsolver.numerical_integration import PitergargDiscreteXY, RungeKuttaApproxXY
from quassigaussian.fastswaptionsolver.parameter_averaging import lognormalimpliedvola


output_data_raw_approx = os.path.join(output_data_raw, "approximation", "piterbarg_swaption_approx")



def calculate_swaption_prices():

    output_path = os.path.join(output_data_raw_approx, date_timestamp, "result")
    output_file = os.path.join(output_path, "swaption_approximation.hdf")
    file_path = get_nonexistant_path(output_file)

    grid_size = 2**12 + 1
    swap_freq = 0.5
    curve_rate = 0.06

    initial_curve = LiborCurve.from_constant_rate(curve_rate)

    kappa_grid = [0.03]

    vola_parameters = [(i, curve_rate, j) for i in [0.6, 0.8] for j in [0.05, 0.2]]
    vola_grid_df = pd.DataFrame(vola_parameters, columns=["lambda", "alpha", "beta"])
    vola_grid_df = vola_grid_df.iloc[[0, 3]]

    #coupon_grid = [0, +0.0025, -0.0025, +0.005, -0.005, +0.01, -0.01, 0.015, -0.015, 0.02, -0.02, 0.025, -0.025]

    XYApproximator = PitergargDiscreteXY
    XYApproximator = RungeKuttaApproxXY

    swap_ls = [(20, 21)]
    coupon_grid = [0, +0.005, -0.005, +0.01, -0.01, 0.015, -0.015]
    #vola_grid_df = vola_grid_df.iloc[9:10]


    for swap_T0_TN in swap_ls:
        print(swap_T0_TN)
        T0, TN = swap_T0_TN
        for kappa in kappa_grid:
            swap_pricer = SwapPricer(initial_curve, kappa)
            swap = Swap(T0, TN, 1/2)
            for index, vola_grid_row in vola_grid_df.iterrows():
                sigma_r = LinearLocalVolatility.from_const(swap.TN, vola_grid_row["lambda"], vola_grid_row["alpha"],
                                                             vola_grid_row["beta"])

                swap = Swap(T0, TN, swap_freq)

                atm_swap_price = swap_pricer.price(swap, 0, 0, 0)
                strike_grid = [atm_swap_price + coupon for coupon in coupon_grid]
                #strike_grid = [0.01, 0.015, 0.02, 0.025, 0.03]

                xy_calculator = XYApproximator(grid_size, swap_pricer, sigma_r, swap)
                integration_approx = DiscreteParameterAveraging(grid_size, swap_pricer, sigma_r, swap, xy_calculator)
                lambda_avg, beta_avg = integration_approx.calculate_average_param()

                for strike in strike_grid:
                    swaption = Swaption(T0, strike, swap)
                    swaption_value, black_implied_vola = lognormalimpliedvola(swaption, swap_pricer, lambda_avg,
                                                                              beta_avg)

                    output_data = pd.DataFrame({'expiry': [T0], "maturity": [TN], "atm strike": atm_swap_price,
                                                "swaption_value":[swaption_value],
                                                "kappa": [kappa], "vola_lambda": [vola_grid_row["lambda"]],
                                                "vola_alpha": [vola_grid_row["alpha"]],
                                                "vola_beta": [vola_grid_row['beta']], "strike": [strike],
                                                'moneyness': [strike-atm_swap_price], "curve_rate": [curve_rate],
                                                "implied_black_vola": [black_implied_vola], 'integration_grid': [grid_size],
                                                 "xy_approximation": [str(xy_calculator)]})
                    try:
                        all_output_data = pd.read_hdf(file_path, key="data")
                    except:
                        all_output_data = pd.DataFrame()
                    all_output_data = pd.concat([all_output_data, output_data])
                    all_output_data.to_hdf(file_path, key="data", complevel=9)



if __name__ == '__main__':
    calculate_swaption_prices()