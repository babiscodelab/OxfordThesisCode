from quassigaussian.fastcalibration.parameter_averaging import calculate_lambda_s_bar, w_s_wrapper, calculate_b_s_bar
from scipy.interpolate import interp1d
import numpy as np


def test_lambda_s_bar():

    x = np.arange(0, 31)
    y = np.where(x >= 16, 1, 0)
    lambda_s = interp1d(x, y, kind='previous')
    lambda_bar = calculate_lambda_s_bar(lambda_s, 30)

    np.testing.assert_almost_equal(lambda_bar, np.sqrt(14/30))

def test_ws():

    t = 2
    T0 = 30

    x = np.arange(0, 31)
    y = np.ones(31)
    lambda_s = interp1d(x, y, kind='previous')

    w_s = w_s_wrapper(T0, lambda_s)
    w_s_bar = w_s(t, T0)
    np.testing.assert_almost_equal(w_s_bar, 2*t/np.power(T0, 2))

def test_bs_bar():

    x = np.arange(0, 31)
    y = np.ones(31)

    lambda_s = interp1d(x, y, kind='previous')
    b_s = interp1d(x, y, kind='previous')

    w_s = w_s_wrapper(lambda_s)
    b_bar = calculate_b_s_bar(w_s, b_s, 30)

    return b_bar

test_bs_bar()