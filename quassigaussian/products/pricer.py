from quassigaussian.utils import calculate_G
import math
import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate
from quassigaussian.products.instruments import Bond, Swap, Swaption, Annuity
from quassigaussian.curves.libor import Curve

class BondPricer():

    def __init__(self, initial_curve: Curve, kappa: float):
        self.kappa = kappa
        self.initial_curve = initial_curve

    def maturity_price(self, bond, x, y):
        return np.ones(x.shape)

    def price(self, bond: Bond, x: float, y: float, t: float):
        G = calculate_G(self.kappa, t, bond.maturity)
        return self.initial_curve.get_discount(bond.maturity)/self.initial_curve.get_discount(t) * np.exp(-G*x - 0.5*math.pow(G, 2)*y)

    def dpdx(self, bond: Bond, x: float, y: float, t: float):
        return - self.price(bond, x, y, t) * calculate_G(self.kappa, t, bond.maturity)

    def d2pdx2(self, bond, x, y, t):
        return self.price(bond, x, y, t) * math.pow(calculate_G(self.kappa, t, bond.maturity), 2)


    def calculate_price_from_forward(self, bond: Bond):

        def forward_integral(u):
            return self.initial_curve.get_inst_forward(u)

        return np.exp(-integrate.quad(forward_integral, 0, bond.maturity)[0])


class SwapPricer():

    def __init__(self, initial_curve: Curve, kappa: float):
        self.kappa = kappa
        self.initial_curve = initial_curve
        self.bond_pricer = BondPricer(initial_curve, kappa)
        self.annuity_pricer = AnnuityPricer(self.bond_pricer)


    def maturity_price(self, swap, x_mesh, y_mesh):
        return self.price(swap, x_mesh, y_mesh, swap.bond_T0.maturity)


    def price(self, swap: Swap, x: float, y: float, t: float):
        return (self.bond_pricer.price(swap.bond_T0, x, y, t) -
                self.bond_pricer.price(swap.bond_TN, x, y, t)) / self.annuity_pricer.annuity_price(t, x, y, swap.annuity)


    def dsdx(self, swap: Swap, x: float, y: float, t: float):

        annuity_price = self.annuity_pricer.annuity_price(t, x, y, swap.annuity)

        res = -1/annuity_price * \
        (self.bond_pricer.price(swap.bond_T0, x, y, t) * calculate_G(self.kappa, t, swap.bond_T0.maturity) -
         self.bond_pricer.price(swap.bond_TN, x, y, t) * calculate_G(self.kappa, t, swap.bond_TN.maturity)) +\
        self.price(swap, x, y, t)/annuity_price * (self.annuity_pricer.annuity_times_g(t, x, y, self.kappa, swap.annuity))

        return res

    def dsdx_v2(self, swap: Swap, x: float, y: float, t: float):
        annuity = self.annuity_pricer.annuity_price(t, x, y, swap.annuity)

        res = (self.bond_pricer.dpdx(swap.bond_T0, x, y, t) - self.bond_pricer.dpdx(swap.bond_TN, x, y, t))/annuity + \
              - (self.bond_pricer.price(swap.bond_T0, x, y, t) - self.bond_pricer.price(swap.bond_TN, x, y, t))\
              *(1/math.pow(annuity, 2)) * self.annuity_pricer.annuity_dx(t, x, y, swap.frequency, swap.bond_list)

        return res

    def d2sdx2(self, swap: Swap, x: float, y: float, t: float):

        annuity = self.annuity_pricer.annuity_price(t, x, y, swap.annuity)
        dannuity = self.annuity_pricer.annuity_dx(t, x, y, self.kappa, swap.annuity)
        d2annuity = self.annuity_pricer.annuity_d2x(t, x, y, self.kappa, swap.annuity)

        pT0 = self.bond_pricer.price(swap.bond_T0, x, y, t)
        pTN = self.bond_pricer.price(swap.bond_TN, x, y, t)
        dpdtT0 = self.bond_pricer.dpdx(swap.bond_T0, x, y, t)
        dpdtTN = self.bond_pricer.dpdx(swap.bond_TN, x, y, t)

        d2pdt2T0 = self.bond_pricer.d2pdx2(swap.bond_T0, x, y, t)
        d2pdt2TN = self.bond_pricer.d2pdx2(swap.bond_TN, x, y, t)

        return (d2pdt2T0-d2pdt2TN)*1/annuity \
               - 2*(dpdtT0 - dpdtTN)*(1/np.power(annuity, 2)) * dannuity \
               + 2*(pT0 - pTN)*(1/np.power(annuity, 3)) * np.power(dannuity, 2) \
               - (pT0 - pTN)/(np.power(annuity, 2)) * d2annuity


class CapitalX():

    def __init__(self, swap_pricer: SwapPricer):
        self.swap_pricer = swap_pricer

    def dxds(self, swap: Swap, x: float, y: float, t: float):
        return 1/self.swap_pricer.dsdx(swap, x, y, t)

    def d2xds2(self, swap: Swap, x: float, y: float, t: float):

        dxds = self.dxds(swap, x, y, t)
        dsdx = self.swap_pricer.dsdx(swap, x, y, t)
        d2sdx2 = self.swap_pricer.d2sdx2(swap, x, y, t)

        return -d2sdx2*dxds/dsdx


class AnnuityPricer():

    def __init__(self, bond_pricer: BondPricer):
        self.bond_pricer = bond_pricer

    def annuity_times_g(self, t: float, x: float, y: float, kappa: float, annuity: Annuity):

        res = 0
        for bond in annuity.bond_list:
            res += annuity.freq * self.bond_pricer.price(bond, x, y, t) * calculate_G(kappa, t, bond.maturity)
        return res

    def annuity_price(self, t: float, x: float, y: float, annuity: Annuity):

        res = 0
        for bond in annuity.bond_list:
            res += annuity.freq*self.bond_pricer.price(bond, x, y, t)

        return res


    def annuity_dx(self, t: float, x: float, y: float,  kappa: float, annuity: Annuity):

        res = 0
        for bond in annuity.bond_list:
            res += - annuity.freq * self.bond_pricer.price(bond, x, y, t) * calculate_G(kappa, t, bond.maturity)

        return res

    def annuity_d2x(self, t: float, x: float, y: float, kappa: float, annuity: Annuity):

        res = 0
        for bond in annuity.bond_list:
            res += annuity.freq * self.bond_pricer.price(bond, x, y, t) * math.pow(calculate_G(kappa, t, bond.maturity), 2)

        return res



class SwaptionPricer():

    def __init__(self, lambda_s: float, b_s: float, swap_pricer: SwapPricer, bond_pricer: BondPricer):
        self.lambda_s = lambda_s
        self.b_s = b_s
        self.swap_pricer = swap_pricer
        self.bond_pricer = bond_pricer

    def maturity_price(self, swaption: Swaption, x_mesh, y_mesh):
        swap_price = self.swap_pricer.maturity_price(swaption.swap, x_mesh, y_mesh)
        return np.maximum(swap_price-swaption.coupon, 0)

    def price(self, swaption: Swaption):

        annuity_pricer = AnnuityPricer(self.bond_pricer)
        annuity_0 = annuity_pricer.annuity_price(0, 0, 0, swaption.swap.annuity)
        swap_0 = self.swap_pricer.price(swaption.swap, 0, 0, 0)

        dplus, dminus = self.d_plus_minus(swaption)

        res = (swap_0 + swap_0*(1 - self.b_s)/self.b_s) * norm.cdf(dplus) - \
              (swaption.coupon + swap_0*(1-self.b_s)/self.b_s) * norm.cdf(dminus)
        res *= annuity_0

        return res

    def d_plus_minus(self, swaption: Swaption):

        swap_0 = self.swap_pricer.price(swaption.swap, 0, 0, 0)
        d = np.log((swap_0 + swap_0*(1 - self.b_s)/self.b_s)/(swaption.coupon + swap_0*(1-self.b_s)/self.b_s))

        denominator = self.b_s * self.lambda_s * math.sqrt(swaption.expiry)

        plus = 0.5 * math.pow(self.b_s, 2) * math.pow(self.lambda_s, 2) * swaption.expiry

        dplus = (d + plus)/denominator
        dminus = (d - plus)/denominator

        return dplus, dminus