
from quassigaussian.products.pricer import BondPricer, SwapPricer, SwaptionPricer
from quassigaussian.products.instruments import Bond, Swaption, Swap
from quassigaussian.curves.libor import LiborCurve
import numpy as np
import scipy.integrate as integrate

def test_bond_pricer():

    bond = Bond(1)
    tmp_file = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\data\market_data\libor_curve\usd_libor\sofr_curve.csv"
    initial_curve = LiborCurve.from_file(tmp_file, "2013-05-20")
    bond_pricer = BondPricer(initial_curve, 0.2)
    initial_curve.get_discount(0.3)
    bond_pricer.price(bond, x=0, y=0, t=0)
    print("pause")


def test_forward_rate_pricer_from_file():

    bond = Bond(10)
    tmp_file = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\data\market_data\libor_curve\usd_libor\sofr_curve.csv"
    initial_curve = LiborCurve.from_file(tmp_file, "2013-05-20")

    bond_pricer = BondPricer(initial_curve, 0.2)
    expected_price = bond_pricer.price(bond, 0, 0, 0)
    actual_price = bond_pricer.calculate_price_from_forward(bond)

    np.testing.assert_approx_equal(expected_price, actual_price)



def test_forward_rate_pricer_from_constant_curve():

    bond = Bond(10)
    initial_curve = LiborCurve.from_constant_rate(0.04)
    bond_pricer = BondPricer(initial_curve, 0.2)

    expected_price = bond_pricer.price(bond, 0, 0, 0)
    actual_price = bond_pricer.calculate_price_from_forward(bond)

    np.testing.assert_approx_equal(expected_price, actual_price)

def test_swap_pricer():
    swap = Swap(1, 20, 0.25)
    tmp_file = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\data\market_data\libor_curve\usd_libor\sofr_curve.csv"
    initial_curve = LiborCurve.from_file(tmp_file, "2013-05-20")

    swap_pricer = SwapPricer(initial_curve, kappa=0.2)
    price = swap_pricer.price(swap, 0, 0, 0)

    print("pas")


def test_swaption_pricer():

    swap = Swap(1, 10, 0.5)
    swaption = Swaption(expiry=1, coupon=0.001,swap=swap)

    tmp_file = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\data\market_data\libor_curve\usd_libor\sofr_curve.csv"
    initial_curve = LiborCurve.from_file(tmp_file, "2013-05-20")

    swap_pricer = SwapPricer(initial_curve, kappa=0.2)
    bond_pricer = BondPricer(initial_curve, 0.2)

    lambda_s = 1000000000
    b_s = 0.1

    swaption_pricer = SwaptionPricer(lambda_s, b_s, swap_pricer, bond_pricer)
    price = swaption_pricer.price(swaption)

    print("pause")
