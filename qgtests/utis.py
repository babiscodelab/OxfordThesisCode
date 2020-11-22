
from quassigaussian.curves.libor import LiborCurve

def get_mock_yield_curve_from_file():
    tmp_file = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\data\market_data\libor_curve\usd_libor\sofr_curve.csv"
    initial_curve = LiborCurve.from_file(tmp_file, "2013-05-20")
    return initial_curve


def get_mock_yield_curve_const(rate=0.04):
    return LiborCurve.from_constant_rate(rate)