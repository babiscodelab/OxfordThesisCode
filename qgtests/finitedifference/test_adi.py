from quassigaussian.finitedifference.adi.run_adi import AdiRunner
from quassigaussian.volatility.local_volatility import LinearLocalVolatility, BlackVolatilityModel
from quassigaussian.products.instruments import Bond, Swap, Swaption
from quassigaussian.products.pricer import BondPricer, SwapPricer, SwaptionPricer
from quassigaussian.utils import extract_x0_result
from quassigaussian.finitedifference.mesher.grid_boundaries import calculate_x_boundaries2, calculate_y_boundaries
from quassigaussian.finitedifference.mesher.linear_mesher import Mesher2d
import numpy as np
from scipy.interpolate import interp1d

from qgtests.utis import get_mock_yield_curve_from_file, get_mock_yield_curve_const

def test_adi_bond():

    maturity = 10

    linear_local_volatility = LinearLocalVolatility.from_const(30, 0.1, 0.1, 0.1)
    #linear_local_volatility = LinearLocalVolatility.from_const(30, 0.1, 0.1, 0)


    theta = 0.5
    kappa = 0.3
    initial_curve = get_mock_yield_curve_const(rate=0.06)

    t_min = 0
    t_max = maturity
    t_grid_size = 200
    x_grid_size = 201

    y_grid_size = 20

    x_min, x_max = calculate_x_boundaries2(t_max, linear_local_volatility, alpha=2.5)
    y_min, y_max = calculate_y_boundaries(t_max, kappa, linear_local_volatility, alpha=2.5)

    mesher = Mesher2d()
    mesher.create_mesher_2d(t_min, t_max, t_grid_size, x_min, x_max, x_grid_size, y_min, y_max, y_grid_size)

    adi_runner = AdiRunner(theta, kappa, initial_curve, linear_local_volatility, mesher)

    bond = Bond(maturity)
    bond_pricer = BondPricer(initial_curve, kappa)

    bond_t0 = adi_runner.run_adi(bond, bond_pricer)
    bond_xyt0 = extract_x0_result(bond_t0, adi_runner.mesher.xgrid, adi_runner.mesher.ygrid)

    actual = bond_pricer.price(bond, 0, 0, 0)

    np.testing.assert_approx_equal(bond_xyt0, actual)

#
# def test_adi_swap():
#     maturity = 10
#
#     linear_local_volatility = LinearLocalVolatility.from_const(30, 0.1, 0.1, 0.1)
#
#     theta = 0.5
#     kappa = 0.3
#     initial_curve = get_mock_yield_curve_const(rate=0.04)
#
#     t_min = 0
#     t_max = maturity
#     t_grid_size = 200
#     x_grid_size = 201
#
#     y_grid_size = 20
#
#     x_min, x_max = calculate_x_boundaries2(t_max, linear_local_volatility, alpha=2.5)
#     y_min, y_max = calculate_y_boundaries(t_max, kappa, linear_local_volatility, alpha=2.5)
#
#     mesher = Mesher2d()
#     mesher.create_mesher_2d(t_min, t_max, t_grid_size, x_min, x_max, x_grid_size, y_min, y_max, y_grid_size)
#
#     adi_runner = AdiRunner(theta, kappa, initial_curve, linear_local_volatility, mesher)
#
#     swap = Swap(10, 30, 0.5)
#
#     swap_pricer = SwapPricer(initial_curve, kappa)
#
#     expected_swap_price = swap_pricer.price(swap, 0, 0, 0)
#
#     actual_swap_price_t0 = adi_runner.run_adi(swap, swap_pricer)
#     actual_swap_price_xyt0 = extract_x0_result(actual_swap_price_t0, adi_runner.mesher.xgrid, adi_runner.mesher.ygrid)
#
#     np.testing.assert_approx_equal(expected_swap_price, actual_swap_price_xyt0)


def test_adi_bond2():

    swaption_expiry = 10
    swaption_maturity = 15
    freq = 0.5
    logvola = 0.2
    maturity = 10

    initial_curve = get_mock_yield_curve_const(rate=0.06)
    kappa = 0.5

    swap_pricer = SwapPricer(initial_curve, kappa)
    swap = Swap(swaption_expiry, swaption_maturity, freq)

    local_volatility = BlackVolatilityModel(logvola, swap, swap_pricer)

    t_min = 0
    t_max = swaption_expiry
    t_grid_size = 100
    x_grid_size = 401

    y_grid_size = 40

    x_min, x_max = calculate_x_boundaries2(t_max, local_volatility, alpha=2)
    y_min, y_max = calculate_y_boundaries(t_max, kappa, local_volatility, alpha=2)

    x_min, x_max = -0.05, +0.05

    mesher = Mesher2d()
    mesher.create_mesher_2d(t_min, t_max, t_grid_size, x_min, x_max, x_grid_size, y_min, y_max, y_grid_size)

    theta = 0.5
    adi_runner = AdiRunner(theta, kappa, initial_curve, local_volatility, mesher)

    bond = Bond(maturity)
    bond_pricer = BondPricer(initial_curve, kappa)

    bond_value0 = adi_runner.run_adi(bond, bond_pricer)
    print("pause")

def test_adi_swaption():

    swaption_expiry = 1
    swaption_maturity = 2
    freq = 0.5
    coupon = 0.062
    rate = 0.06

    initial_curve = get_mock_yield_curve_const(rate=rate)
    kappa = 0.3

    swap_pricer = SwapPricer(initial_curve, kappa)
    swap = Swap(swaption_expiry, swaption_maturity, freq)
    swaption_pricer = SwaptionPricer(swap_pricer)

    swaption = Swaption(swaption_expiry, coupon, swap)

    x = np.arange(0, 31)
    y = np.ones(31)*0.1

    lambda_t = interp1d(x, y, kind='previous')
    alpha_t = interp1d(x, y, kind='previous')
    b_t = interp1d(x, y*0, kind='previous')

    local_volatility = LinearLocalVolatility(lambda_t, alpha_t, b_t)

    #local_volatility = LinearLocalVolatility.from_const(30, 0.1, 0.1, 0.1)

    t_min = 0
    t_max = swaption_expiry
    t_grid_size = 100
    x_grid_size = 401

    y_grid_size = 20

    x_min, x_max = calculate_x_boundaries2(t_max, local_volatility, alpha=3)
    y_min, y_max = calculate_y_boundaries(t_max, kappa, local_volatility, alpha=3)

    mesher = Mesher2d()
    mesher.create_mesher_2d(t_min, t_max, t_grid_size, x_min, x_max, x_grid_size, y_min, y_max, y_grid_size)

    theta = 0.5
    adi_runner = AdiRunner(theta, kappa, initial_curve, local_volatility, mesher)

    swaption_value0 = adi_runner.run_adi(swaption, swaption_pricer)

    x0 = extract_x0_result(swaption_value0, mesher.xgrid, mesher.ygrid)
    print("Swaption value at 0: ", x0)