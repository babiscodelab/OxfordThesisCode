
from quassigaussian.products.pricer import BondPricer
from quassigaussian.products.instruments import Bond
from quassigaussian.curves.libor import LiborCurve


def test_bond_pricer():

    bond = Bond(1)
    tmp_file = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\data\market_data\libor_curve\usd_libor\sofr_curve.csv"
    initial_curve = LiborCurve.from_file(tmp_file, "2013-05-20")
    bond_pricer = BondPricer(initial_curve, 0.2)
    initial_curve.get_discount(0.3)
    bond_pricer.price(bond, x=0, y=0, t=0)
    print("pause")


test_bond_pricer()