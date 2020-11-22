import pytest
from quassigaussian.products.pricer import BondPricer, SwapPricer, SwaptionPricer
from quassigaussian.products.instruments import Bond, Swaption, Swap
from quassigaussian.curves.libor import LiborCurve
import numpy as np
import scipy.integrate as integrate

def test_bond_pricer():

    rate = 0.04
    maturity = 5
    bond = Bond(maturity)
    initial_curve = LiborCurve.from_constant_rate(rate)
    bond_pricer = BondPricer(initial_curve, 0.2)
    actual_price = bond_pricer.price(bond, x=0, y=0, t=0)
    expected_price = np.exp(-rate*maturity)

    np.testing.assert_approx_equal(actual_price, expected_price)

@pytest.mark.skip(reason="Need to check interpolation and curve construcion from market data methodologies")
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
    initial_curve = LiborCurve.from_constant_rate(0.04)

    swap_pricer = SwapPricer(initial_curve, kappa=0.2)
    price = swap_pricer.price(swap, 0, 0, 0)

    print("pas")


def test_black_swaption_pricer():
    # see John Hull ex 28.4, page 661

    kappa = 0.3
    swap = Swap(5, 8, 0.5)
    initial_curve = LiborCurve.from_constant_rate(0.06)

    bond_pricer = BondPricer(initial_curve, kappa=kappa)
    swap_pricer = SwapPricer(initial_curve, kappa=kappa)

    coupon = 0.062
    swaption = Swaption(expiry=5, coupon=coupon, swap=swap)


    swaption_pricer = SwaptionPricer(lambda_s=0.2, b_s=1, swap_pricer=swap_pricer, bond_pricer=bond_pricer)
    swaption_value = 100*swaption_pricer.black76_price(swaption)

    np.testing.assert_approx_equal(swaption_value, 2.07, significant=4)