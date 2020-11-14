from quassigaussian.fastcalibration.approximation import PiterbargApproximator
from quassigaussian.volatility.local_volatility import LinearLocalVolatility
from quassigaussian.products.pricer import SwapPricer
from quassigaussian.products.instruments import Swap
from quassigaussian.curves.libor import LiborCurve
import numpy as np
from scipy.interpolate import interp1d


def test_piterbarg_y_bar():

    kappa = 0.2
    x = np.arange(0, 31)
    y = np.ones(31)
    t = 0.5


    lambda_t = interp1d(x, y, kind='previous')
    alpha_t = interp1d(x, y, kind='previous')
    b_t = interp1d(x, y, kind='previous')

    tmp_file = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\data\market_data\libor_curve\usd_libor\sofr_curve.csv"
    initial_curve = LiborCurve.from_file(tmp_file, "2013-05-20")

    swap_pricer = SwapPricer(initial_curve, kappa=kappa)

    sigma_r = LinearLocalVolatility(lambda_t, alpha_t, b_t)
    piterbarg_approx = PiterbargApproximator(sigma_r, swap_pricer)

    y_bar_actual = piterbarg_approx._calculate_ybar(t)
    y_bar_expected = np.exp(-2*kappa*t)*(np.exp(2*kappa*t) - 1)/(2*kappa)

    np.testing.assert_approx_equal(y_bar_actual, y_bar_expected)

    swap = Swap(0, 1, frequency=0.5)
    swap_pricer.price(swap, 0, 0, 0)
    swap_0 = piterbarg_approx.swap_pricer.price(swap, 0, 0, 0)

    x0 = piterbarg_approx._calculate_x0(t, swap, swap_0, y_bar_actual)

    print("pause")

