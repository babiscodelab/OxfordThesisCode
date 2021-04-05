import scipy.integrate
import numpy as np
from quassigaussian.products.instruments import Swaption
from quassigaussian.products.pricer import Black76Pricer, find_implied_black_vola, SwapPricer



def lognormalimpliedvola(swaption: Swaption, swap_pricer: SwapPricer, lambda_s_bar, b_bar):

    black76price = Black76Pricer(lambda_s_bar, b_bar, swap_pricer, swap_pricer.bond_pricer)
    swaption_value = black76price.black76_price(swaption)
    black_implied_vola = find_implied_black_vola(swaption_value, swaption, swap_pricer, swap_pricer.bond_pricer)

    return swaption_value, black_implied_vola


def calculate_swaption_approx_price(swaption: Swaption, swap_pricer: SwapPricer, lambda_square, b_s):

    lambda_s_bar, b_bar = calculate_vola_skew(swaption.expiry, lambda_square, b_s)

    black76price = Black76Pricer(lambda_s_bar, b_bar, swap_pricer, swap_pricer.bond_pricer)

    swaption_value = black76price.black76_price(swaption)
    black_implied_vola = find_implied_black_vola(swaption_value, swaption, swap_pricer, swap_pricer.bond_pricer)

    return swaption_value, black_implied_vola

def calculate_vola_skew(expiry, lambda_square, b_s):
    lambda_integral = calculate_lambda_integral_callable(lambda_square)
    lambda_s_bar = calculate_lambda_s_bar(lambda_integral, expiry)

    w_s = w_s_wrapper(expiry, lambda_square, lambda_integral)
    b_bar = calculate_b_s_bar(w_s, b_s, expiry)
    return lambda_s_bar, b_bar

def calculate_lambda_integral_callable(lambda_square):
    def calculate_lambda_integral(t_to: float):
        return scipy.integrate.quad(lambda_square, 0, t_to, epsrel=0.000001)[0]
    return calculate_lambda_integral

def calculate_lambda_integral(lambda_square: callable, t_from: float, t_to: float):
    return scipy.integrate.quad(lambda_square, t_from, t_to, epsrel=0.000001)[0]

def calculate_lambda_s_bar(lambda_square_integral: callable, T0: float):
    return np.sqrt(lambda_square_integral(T0)/ T0)


def w_s_wrapper(T0: float, lambda_square: callable, lambda_integral: callable):

    def ws_denominator_integral(u):
        return lambda_integral(u) * lambda_square(u)

    denominator = scipy.integrate.quad(ws_denominator_integral, 0, T0)[0]

    def w_s(t):
        numerator = lambda_square(t) * lambda_integral(t)
        return numerator/denominator

    return w_s


def calculate_b_s_bar(w_s: callable, b_s: callable, T0):

   def b_integral(t, w_s, b_s):
       return w_s(t) * b_s(t)

   b_bar = scipy.integrate.quad(b_integral, 0, T0, args=(w_s, b_s))[0]

   return b_bar