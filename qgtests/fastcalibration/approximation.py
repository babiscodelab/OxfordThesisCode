from quassigaussian.fastcalibration.approximation import PiterbargApproximator
from quassigaussian.volatility.local_volatility import LinearLocalVolatility
from quassigaussian.products.pricer import SwapPricer
from quassigaussian.products.instruments import Swap
from quassigaussian.curves.libor import LiborCurve
import numpy as np
from scipy.interpolate import interp1d


def test_piterbarg_y_bar():

    kappa = 0.001
    t = 15

    initial_curve = LiborCurve.from_constant_rate(0.06)
    swap_pricer = SwapPricer(initial_curve, kappa=kappa)

    sigma_r = LinearLocalVolatility.from_const(t, 0.1, 0.1, 0)
    piterbarg_approx = PiterbargApproximator(sigma_r, swap_pricer)

    y_bar_actual = piterbarg_approx._calculate_ybar(t)
    y_bar_expected = np.exp(-2*kappa*t)*(np.exp(2*kappa*t) - 1)/(2*kappa)*0.0001

    np.testing.assert_approx_equal(y_bar_actual, y_bar_expected)


def test_piterbarg_ksi():

    kappa = 0.001
    t = 15

    initial_curve = LiborCurve.from_constant_rate(0.06)
    swap_pricer = SwapPricer(initial_curve, kappa=kappa)

    sigma_r = LinearLocalVolatility.from_const(t, 0.5, 0.5, 0.2)

    piterbarg_approx = PiterbargApproximator(sigma_r, swap_pricer)

    swap = Swap(4, 5, 0.5)
    swap_value = swap_pricer.price(swap, 0.01717964, 0.0025, 0.04)
    ksi = piterbarg_approx.calculate_ksi(0.04, swap_value, swap)

    print("pause")

    #
    # swap = Swap(29, 30, frequency=0.5)
    #
    # swap_0 = piterbarg_approx.swap_pricer.price(swap, 0, 0, 0)
    #
    # x0 = piterbarg_approx._calculate_x0(t, swap, swap_0, y_bar_actual)
    #
    # x_bar = piterbarg_approx._calculate_xbar(t, y_bar_actual, swap, swap_0)
    #
    # swap_price = swap_pricer.price(swap, x=0, y=0, t=t)
    #
    # x_simple = piterbarg_approx.x_bar_simple(t, swap_price, swap)
    # print("pause")

test_piterbarg_y_bar()
