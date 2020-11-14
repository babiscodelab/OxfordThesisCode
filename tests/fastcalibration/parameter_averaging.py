from quassigaussian.fastcalibration.parameter_averaging import lambda_s_bar, w_s_wrapper, b_s_bar
from scipy.interpolate import interp1d
import numpy as np


def test_lambda_s_bar():

    x = np.arange(0, 30)
    y = np.ones(30)
    lambda_s = interp1d(x, y, kind='previous')

    lambda_s_bar(lambda_s, 30)


def test_ws():

    x = np.arange(0, 30)
    y = np.ones(30)
    lambda_s = interp1d(x, y, kind='previous')

    w_s = w_s_wrapper(lambda_s)
    w_s(1, 29)


def test_bs_bar():

    x = np.arange(0, 30)
    y = np.ones(30)

    lambda_s = interp1d(x, y, kind='previous')
    b_s = interp1d(x, y, kind='previous')

    w_s = w_s_wrapper(lambda_s)
    b_bar = b_s_bar(w_s, b_s, 29)

    return b_bar

test_bs_bar()