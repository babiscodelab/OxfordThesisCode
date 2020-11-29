from quassigaussian.montecarlo.simulations import ResultSimulatorObj
from quassigaussian.products.instruments import Swaption
from quassigaussian.products.pricer import SwaptionPricer


def monte_carlo_pricer(res: ResultSimulatorObj, swaption: Swaption, swaption_pricer: SwaptionPricer):

    x = res.x[:, -1]
    y = res.y[:, -1]

    annuity_texp = swaption_pricer.swap_pricer.annuity_pricer.annuity_price(swaption.expiry, x, y, swaption.swap.annuity)
    annuity_t0 = swaption_pricer.swap_pricer.annuity_pricer.annuity_price(0, x, y, swaption.swap.annuity)
    return swaption_pricer.maturity_price(swaption, x, y)/annuity_texp * annuity_t0