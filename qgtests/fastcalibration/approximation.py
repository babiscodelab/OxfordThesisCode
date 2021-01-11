from quassigaussian.fastcalibration.approximation import DisplacedDiffusionParameterApproximator, PiterbargExpectationApproximator
from quassigaussian.fastcalibration.numerical_integration import PitergargDiscreteXY, RungeKuttaApproxXY
from quassigaussian.volatility.local_volatility import LinearLocalVolatility
from quassigaussian.products.pricer import SwapPricer, BondPricer
from quassigaussian.products.instruments import Swap, Swaption
from quassigaussian.curves.libor import LiborCurve
import numpy as np
from quassigaussian.fastcalibration.parameter_averaging import calculate_swaption_approx_price, w_s_wrapper, calculate_vola_skew, lognormalimpliedvola
from quassigaussian.fastcalibration.discrete_averaging import DiscreteParameterAveraging


def test_runge_kutta_approx():

    kappa = 0.3
    t = 10

    initial_curve = LiborCurve.from_constant_rate(0.06)
    swap_pricer = SwapPricer(initial_curve, kappa=kappa)

    sigma_r = LinearLocalVolatility.from_const(t, 0.4, 0.1, 0)
    swap = Swap(4, 5, 0.5)

    swap0 = swap_pricer.price(swap, 0, 0, 0)

    integration_grid_size = 2*10 + 1
    rk_approx = RungeKuttaApproxXY(integration_grid_size, swap_pricer, sigma_r, swap)
    res = rk_approx.calculate_xy()

    piterbarg_approx = PiterbargExpectationApproximator(sigma_r, swap_pricer)

    time_grid = res.t
    xbar = []
    ybar = []
    for t in time_grid:
        ybar.append(piterbarg_approx.ybar_formula(t))
        xbar.append(piterbarg_approx.xbar_formula(t, ybar[-1], swap, swap0, 0))

    print("As")



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

    pass


def test_swaption_price():

    kappa = 0.03

    swaption_expiry = 4
    swap_maturity = 5
    swap_freq = 0.5

    initial_curve = LiborCurve.from_constant_rate(0.06)
    swap_pricer = SwapPricer(initial_curve, kappa=kappa)

    sigma_r = LinearLocalVolatility.from_const(swap_maturity, 0.4, 0.1, 0)

    piterbarg_approx = PiterbargExpectationApproximator(sigma_r, swap_pricer)
    swap = Swap(swaption_expiry, swap_maturity, swap_freq)

    displaced_diffusion = DisplacedDiffusionParameterApproximator(sigma_r, swap_pricer, swap, piterbarg_approx)
    coupon = swap_pricer.price(swap, 0, 0, 0)

    swaption = Swaption(swaption_expiry, coupon, swap)

    b_s = displaced_diffusion.calculate_bs
    lambda_s_bar, b_bar = calculate_vola_skew(swaption.expiry, displaced_diffusion.calculate_lambda_square, b_s)

    swaption_value, black_implied_vola = lognormalimpliedvola(swaption, swap_pricer, lambda_s_bar, b_bar)

    print(lambda_s_bar, b_bar)
    print(swaption_value, black_implied_vola)





def test_approx():
    kappa = 0.03

    initial_curve = LiborCurve.from_constant_rate(0.06)
    swap_pricer = SwapPricer(initial_curve, kappa=kappa)

    sigma_r = LinearLocalVolatility.from_const(15, 0.4, 0.06, 0.2)
    swaption_expiry = 4
    swap = Swap(swaption_expiry, 5, 0.5)

    coupon = swap_pricer.price(swap, 0, 0, 0)

    swaption = Swaption(swaption_expiry, coupon, swap)
    xyapproximator = RungeKuttaApproxXY
    #xyapproximator = PitergargDiscreteXY

    for xyapproximator in [RungeKuttaApproxXY, PitergargDiscreteXY]:
        for k in [16]:
            grid_size = 2**k + 1
            #xy_calculator = PitergargDiscreteXY(grid_size, swap_pricer, sigma_r, swap)
            xy_calculator = xyapproximator(grid_size, swap_pricer, sigma_r, swap)
            integration_approx = DiscreteParameterAveraging(grid_size, swap_pricer, sigma_r, swap, xy_calculator)
            lambda_avg, beta_avg = integration_approx.calculate_average_param()
            swaption_value, black_implied_vola = lognormalimpliedvola(swaption, swap_pricer, lambda_avg, beta_avg)

            print(lambda_avg, beta_avg)
            print(swaption_value, black_implied_vola)
