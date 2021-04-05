from quassigaussian.parameters.volatility.local_volatility import LinearLocalVolatility
from quassigaussian.products.instruments import Swap, Swaption
from quassigaussian.curves.libor import LiborCurve
import pandas as pd
import os
import numpy as np
from report.directories import output_data_raw, date_timestamp
from quassigaussian.products.pricer import  SwapPricer
from report.utils import get_nonexistant_path
from quassigaussian.fastswaptionsolver.discrete_averaging import DiscreteParameterAveraging
from quassigaussian.fastswaptionsolver.numerical_integration import PitergargDiscreteXY
from quassigaussian.fastswaptionsolver.parameter_averaging import lognormalimpliedvola
from qgtests.utis import get_mock_yield_curve_const
from scipy.optimize import minimize
from quassigaussian.fastswaptionsolver.numerical_integration import PitergargDiscreteXY, RungeKuttaApproxXY
from scipy.optimize import Bounds
import matplotlib.pyplot as plt

from quassigaussian.finitedifference.adi.run_adi import AdiRunner
from quassigaussian.parameters.volatility.local_volatility import LinearLocalVolatility
from quassigaussian.products.instruments import Swap, Swaption
from quassigaussian.products.pricer import SwapPricer, SwaptionPricer, find_implied_black_vola
from quassigaussian.finitedifference.mesher.grid_boundaries import calculate_x_boundaries2, calculate_u_boundaries, calculate_x_boundaries3
from quassigaussian.finitedifference.mesher.linear_mesher import Mesher2d, extract_x0_result
import pandas as pd
import os
from qgtests.utis import get_mock_yield_curve_const
from report.directories import output_data_raw, date_timestamp
from report.utils import get_nonexistant_path



grid_size = 2 ** 9 + 1

def parse_bbg_file(bbg_file):
    bbg_df = pd.read_excel(bbg_file, "Quotes")
    expiry = bbg_df["Expiry"].apply(lambda x: parse_expiry(x))
    bbg_df["Expiry"] = expiry
    new_col = [parse_column(x) for x in bbg_df.columns]

    bbg_df.columns = new_col
    bbg_df = bbg_df.melt(id_vars=["Expiry", "Tenor", "Rate"], var_name="moneyness", value_name="implied black vola")
    bbg_df["implied black vola"] = bbg_df["implied black vola"]/100

    return bbg_df



def parse_expiry(x):

    if "Mo" in x:
        months = x.split('Mo')[0]
        return 1/12 * int(months)
    if "Yr" in x:
        years = x.split("Yr")[0]
        return int(years)

def parse_column(x):

    if "bps" in x:
        return int(x.split("bps")[0])/10000
    elif x == "ATM":
        return 0
    else:
        return x

def calibrate_model(market_data, XYApproximator):

    expiry = market_data["Expiry"].iloc[0]
    tenor = market_data["Tenor"].iloc[0]
    curve_rate = market_data["Rate"].iloc[0]

    kappa = 0.03
    initial_curve = get_mock_yield_curve_const(rate=curve_rate)
    swap_pricer = SwapPricer(initial_curve, kappa)

    swap = Swap(expiry, expiry+tenor, frequency=0.5)

    atm_swap_price = swap_pricer.price(swap, 0, 0, 0)


    def minimize_func(x):
        print("run minimize")
        lambda_param = x[0]
        beta_param = x[1]
        sigma_r = LinearLocalVolatility.from_const(int(swap.TN), lambda_param, curve_rate,
                                                   beta_param)

        xy_calculator = XYApproximator(grid_size, swap_pricer, sigma_r, swap)
        integration_approx = DiscreteParameterAveraging(grid_size, swap_pricer, sigma_r, swap, xy_calculator)
        lambda_avg, beta_avg = integration_approx.calculate_average_param()

        error = 0

        for row_id, inp_data in market_data.iterrows():
            expiry = inp_data["Expiry"]
            coupon = inp_data["moneyness"] + atm_swap_price
            swaption = Swaption(expiry, coupon, swap)
            market_implied_vola = inp_data["implied black vola"]
            swaption_value, black_implied_vola = lognormalimpliedvola(swaption, swap_pricer, lambda_avg,
                                                                      beta_avg)
            error += (black_implied_vola-market_implied_vola)**2

        return error

    bounds = Bounds([0, -0.9], [0.9, 0.9])
    res = minimize(minimize_func, (0.4, 0.2), bounds=bounds)
    print(res.x)
    return res


def price_fd(swaption_pricer, loca_vola, swaption):

    expiry = swaption.expiry
    initial_curve = swaption_pricer.swap_pricer.initial_curve

    kappa = swaption_pricer.swap_pricer.kappa

    x_min, x_max = calculate_x_boundaries3(expiry, kappa, loca_vola, alpha=4)
    y_min, y_max = calculate_u_boundaries(expiry, kappa, loca_vola, alpha=4)

    mesher = Mesher2d()
    mesher.create_mesher_2d(0, expiry, 200, x_min, x_max, 400, y_min, y_max,
                            40)

    adi_runner = AdiRunner(1/2, kappa, initial_curve, loca_vola, mesher)

    swaption_t0 = pd.DataFrame(adi_runner.run_adi(swaption, swaption_pricer))

    swaption_t0_x0_y0 = extract_x0_result(swaption_t0.values, mesher.xgrid, mesher.ugrid)
    implied_black_vola = find_implied_black_vola(swaption_t0_x0_y0, swaption, swaption_pricer.swap_pricer, swaption_pricer.swap_pricer.bond_pricer)

    return implied_black_vola

def plot_model_vs_market_data(market_data, res, approximator):

    expiry = market_data["Expiry"].iloc[0]
    tenor = market_data["Tenor"].iloc[0]
    curve_rate = market_data["Rate"].iloc[0]

    kappa = 0.03
    initial_curve = get_mock_yield_curve_const(rate=curve_rate)
    swap_pricer = SwapPricer(initial_curve, kappa)
    swaption_pricer = SwaptionPricer(swap_pricer)
    swap = Swap(expiry, expiry+tenor, frequency=0.5)

    atm_swap_price = swap_pricer.price(swap, 0, 0, 0)

    sigma_r = LinearLocalVolatility.from_const(int(swap.TN),  res.x[0], curve_rate,
                                               res.x[1])

    xy_calculator = approximator(grid_size, swap_pricer, sigma_r, swap)
    aprox_type = str(xy_calculator)
    integration_approx = DiscreteParameterAveraging(grid_size, swap_pricer, sigma_r, swap, xy_calculator)
    lambda_avg, beta_avg = integration_approx.calculate_average_param()

    moneyness_ls = []
    market_implied_vola_ls = []
    black_implied_vola_ls = []
    swaption_value_fd_ls = []
    for row_id, inp_data in market_data.iterrows():
        expiry = inp_data["Expiry"]
        coupon = inp_data["moneyness"] + atm_swap_price
        if coupon<0:
            continue
        swaption = Swaption(expiry, coupon, swap)
        market_implied_vola_ls.append(inp_data["implied black vola"])

        swaption_value_fd = price_fd(swaption_pricer, sigma_r, swaption)

        swaption_value_fd_ls.append(swaption_value_fd)
        swaption_value, black_implied_vola = lognormalimpliedvola(swaption, swap_pricer, lambda_avg, beta_avg)
        black_implied_vola_ls.append(black_implied_vola)
        print(swaption_value_fd_ls)
        moneyness_ls.append(inp_data["moneyness"])

    fig = plt.figure()
    plt.plot(moneyness_ls, market_implied_vola_ls, "g-x", label="Market data")
    plt.plot(moneyness_ls, black_implied_vola_ls, "r-x", label="Approximate solution")
    plt.plot(moneyness_ls, swaption_value_fd_ls, "b-x", label="Finite Difference repricing")
    plt.legend()

    return fig


def select_market_data(market_data, expiry, tenor):

    market_data = market_data.loc[(market_data["Expiry"]==expiry) & (market_data["Tenor"]==tenor)]
    return market_data


def run(market_data_file, expiry, tenor, strike_calib, approximator):

    bbg_market_data = parse_bbg_file(market_data_file)
    bbg_market_data = select_market_data(bbg_market_data, expiry, tenor)

    bbg_market_data_calib = bbg_market_data.loc[bbg_market_data["moneyness"].isin(strike_calib)]

    res = calibrate_model(bbg_market_data_calib, approximator)

    print("Adw")
    fig = plot_model_vs_market_data(bbg_market_data, res, approximator)

    return fig

if __name__ == "__main__":

    XYApproximator = RungeKuttaApproxXY


    # bbg_file = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\data\market_data\2016_07_03_XY10Y_bbg_usd_irs_swaption.xlsx"
    # #XYApproximator = PitergargDiscreteXY
    # run(bbg_file, 10, 10, [-0.01, 0], XYApproximator)

    #bbg_file = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\data\market_data\2019_08_03_XY10Y_bbg_usd_irs_swaption.xlsx"
    #run(bbg_file, 10, 10, [-0.02, 0, 0.01], XYApproximator)

    #bbg_file = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\data\market_data\2017_03_16_XY1Y_bbg_nok_irs_swaption.xlsx"

    #run(bbg_file, 20, 1, [-0.01, 0, 0.01], XYApproximator)

    bbg_file = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\data\market_data\2014_03_24_XY1Y_bbg_nok_irs_swaption.xlsx"

    #run(bbg_file, 15, 1, [-0.01, 0, 0.01], XYApproximator)

    bbg_file = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\data\market_data\2016_08_29_XY1Y_bbg_nok_irs_swaption.xlsx"

    #run(bbg_file, 20, 1, [-0.01, -0.005, 0, 0.01, 0.02], XYApproximator)


    bbg_file = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\data\market_data\2020_06_11_XY10Y_bbg_usd_irs_swaption.xlsx"
    XYApproximator = RungeKuttaApproxXY

    fig_usd2 = run(bbg_file, 10, 10, [-0.01, -0.005, 0, 0.005, 0.01, 0.02], XYApproximator)


    #plt.show()
