from quassigaussian.fastcalibration.approximation import DisplacedDiffusionParameterApproximator, PiterbargExpectationApproximator
from quassigaussian.volatility.local_volatility import LinearLocalVolatility
from quassigaussian.products.pricer import SwapPricer, BondPricer
from quassigaussian.products.instruments import Swap, Swaption
from quassigaussian.curves.libor import LiborCurve
import numpy as np
from quassigaussian.fastcalibration.parameter_averaging import calculate_swaption_approx_price, ApproximatorDiscretizer


def test_piterbarg_y_bar():

    kappa = 0.001
    t = 15

    initial_curve = LiborCurve.from_constant_rate(0.06)
    swap_pricer = SwapPricer(initial_curve, kappa=kappa)

    sigma_r = LinearLocalVolatility.from_const(t, 0.1, 0.1, 0)
    piterbarg_approx = PiterbargExpectationApproximator(sigma_r, swap_pricer)

    y_bar_actual = piterbarg_approx.calculate_ybar(t)
    y_bar_expected = np.exp(-2*kappa*t)*(np.exp(2*kappa*t) - 1)/(2*kappa)*0.0001

    np.testing.assert_approx_equal(y_bar_actual, y_bar_expected)


def test_piterbarg_ksi():

    kappa = 0.001
    t = 15

    initial_curve = LiborCurve.from_constant_rate(0.06)
    swap_pricer = SwapPricer(initial_curve, kappa=kappa)

    sigma_r = LinearLocalVolatility.from_const(t, 0.5, 0.5, 0.2)

    piterbarg_approx = PiterbargExpectationApproximator(sigma_r, swap_pricer)

    swap = Swap(4, 5, 0.5)
    swap_value = swap_pricer.price(swap, 0.01717964, 0.0025, 0.04)
    ksi = piterbarg_approx.calculate_ksi(0.04, swap_value, swap)

    np.testing.assert_approx_equal(ksi, 0.01717964, significant=4)


def test_linear_local_volatility_approximation():

    kappa = 0.001
    t = 15

    initial_curve = LiborCurve.from_constant_rate(0.06)
    swap_pricer = SwapPricer(initial_curve, kappa=kappa)

    sigma_r = LinearLocalVolatility.from_const(t, 0.1, 0.1, 0)

    piterbarg_approx = PiterbargExpectationApproximator(sigma_r, swap_pricer)
    swap = Swap(4, 5, 0.5)

    displaced_diffusion = DisplacedDiffusionParameterApproximator(sigma_r, swap_pricer, swap, piterbarg_approx)

    lambda_s, b_s = displaced_diffusion.approximate_parameters(1)

    pass


def test_swaption_price():

    kappa = 0.3
    t = 15
    swaption_expiry = 4

    swap_start = swaption_expiry
    swap_maturity = 5
    swap_freq = 0.5

    initial_curve = LiborCurve.from_constant_rate(0.06)
    swap_pricer = SwapPricer(initial_curve, kappa=kappa)

    sigma_r = LinearLocalVolatility.from_const(t, 0.05, 0.5, 0)

    piterbarg_approx = PiterbargExpectationApproximator(sigma_r, swap_pricer)
    swap = Swap(swaption_expiry, swap_maturity, swap_freq)

    displaced_diffusion = DisplacedDiffusionParameterApproximator(sigma_r, swap_pricer, swap, piterbarg_approx)
    bond_pricer = BondPricer(initial_curve, kappa)
    coupon = swap_pricer.price(swap, 0, 0, 0)

    swaption = Swaption(swaption_expiry, coupon, swap)
    approx_discr = ApproximatorDiscretizer(swap, swap_pricer, number_points=100)
    x_bar, y_bar = approx_discr.calculate_discretized_expectations(piterbarg_approx.xbar_formula, piterbarg_approx.ybar_formula)

    # \lambda^2(s), & the integral_0^t lambda^2(s)
    lambda_square, lambda_square_integral = approx_discr.calculate_discretized_lambda(displaced_diffusion.get_lambda_s_square_callable_decorator(x_bar, y_bar))

    # b(s)
    b_s = displaced_diffusion.get_bs_callable_decorator(x_bar, y_bar)

    # w(s)
    w_s = approx_discr.calculate_discretized_w(lambda_square, lambda_square_integral)
    swaption_value, black_implied_vola = calculate_swaption_approx_price(swaption, swap_pricer, w_s, lambda_square_integral, b_s)

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
