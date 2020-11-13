
from quassigaussian.utils import calculate_G
import math
import numpy as np
from scipy.stats import norm

class Bond():

    def __init__(self, maturity):
        self.maturity = maturity


class BondPricer():

    def __init__(self, initial_curve, kappa):
        self.kappa = kappa
        self.initial_curve = initial_curve

    def price(self, bond: Bond, x, y, t):
        G = calculate_G(self.kappa, t, bond.maturity)
        return self.initial_curve(bond.maturity)/self.initial_curve(t) * np.exp(-G*x - 0.5*math.pow(G, 2)*y)

    def dpdx(self, bond: Bond, x, y, t):
        return - self.price(bond, x, y, t) * calculate_G(self.kappa, t, bond.maturity)

    def d2pdx2(self, bond, x, y, t):
        return self.price(bond, x, y, t) * math.pow(calculate_G(self.kappa, t, bond.maturity), 2)

class Swap():

    def __init__(self, T0, TN, initial_curve, frequency):

        self.T0 = T0
        self.TN = TN
        self.start = T0 + frequency # T1
        self.initial_curve = initial_curve
        self.frequency = frequency
        self.number_payments = (self.T0 - self.start)/self.frequency
        self.payment_schedule = range(self.start, self.TN, self.frequency)
        self.bond_list = self.get_bond_list()

        self.bond_T0 = Bond(self.T0)
        self.bond_TN = Bond(self.TN)

        self.bond_list = self.get_bond_list()

    def get_bond_list(self):

        bond_list = []
        for payment_date in self.payment_schedule:
            bond_list.append(Bond(payment_date))

        return bond_list

class SwapPricer():

    def __init__(self, initial_curve, kappa):
        self.kappa = kappa
        self.initial_curve = initial_curve
        self.bond_pricer = BondPricer(initial_curve, kappa)
        self.annuity_pricer = AnnuityPricer()


    def price(self, swap: Swap, x, y, t):
        return (self.bond_pricer.price(swap.bond_T0, x, y, t) -
                self.bond_pricer.price(swap.bond_TN, x, y, t)) / self.annuity_pricer.annuity_price(t, x, y, swap.frequency, swap.bond_list)


    def dsdx(self, swap: Swap, x, y, t):

        annuity = self.annuity_pricer.annuity_price(t, x, y, swap.frequency, swap.bond_list)

        res = -1/annuity * \
        (self.bond_pricer.price(swap.bond_T0, x, y, t) * calculate_G(self.kappa, t, swap.bond_T0.maturity) -
         self.bond_pricer.price(swap.bond_TN, x, y, t) * calculate_G(self.kappa, t, swap.bond_TN.maturity)) +\
        self.price(swap, x, y, t)/annuity * (self.annuity_pricer.annuity_times_g(t, x, y, swap.frequency, self.kappa, swap.bond_list)) / annuity

        return res

    def dsdx_v2(self, swap: Swap, x, y, t):
        annuity = self.annuity_pricer.annuity_price(t, x, y, swap.frequency, swap.bond_list)

        res = (self.bond_pricer.dpdx(swap.bond_T0, x, y, t) - self.bond_pricer.dpdx(swap.bond_TN, x, y, t))/annuity + \
              - (self.bond_pricer.price(swap.bond_T0, x, y, t) - self.bond_pricer.price(swap.bond_TN, x, y, t))\
              *(1/math.pow(annuity, 2)) * self.annuity_pricer.annuity_dx(t, x, y, swap.frequency, swap.bond_list)

        return res

    def d2sdx2(self, swap: Swap, x, y, t):

        annuity = self.annuity_pricer.annuity_price(t, x, y, swap.frequency, swap.bond_list)
        dannuity = self.annuity_pricer.annuity_dx(t, x, y, swap.frequency, self.kappa, swap.bond_list)
        d2annuity = self.annuity_pricer.annuity_d2x(t, x, y, swap.frequency, self.kappa, swap.bond_list)

        pT0 = self.bond_pricer.price(swap.bond_T0, x, y, t)
        pTN = self.bond_pricer.price(swap.bond_TN, x, y, t)
        dpdtT0 = self.bond_pricer.dpdx(swap.bond_T0, x, y, t)
        dpdtTN = self.bond_pricer.dpdx(swap.bond_TN, x, y, t)

        d2pdt2T0 = self.bond_pricer.d2pdx2(swap.bond_T0, x, y, t)
        d2pdt2TN = self.bond_pricer.d2pdx2(swap.bond_TN, x, y, t)

        return (d2pdt2T0-d2pdt2TN)*1/annuity \
               - 2*(dpdtT0 - dpdtTN)*(1/math.pow(annuity, 2)) * dannuity \
               + 2*(pT0 - pTN)*(1/math.pow(annuity, 3)) * dannuity \
               - (pT0 - pTN)/(1/math.pow(annuity, 2)) * math.pow(d2annuity, 2)


class CapitalX():

    def __init__(self, swap_pricer: SwapPricer):
        self.swap_pricer = swap_pricer

    def dxds(self, swap, x, y, t):
        return 1/self.swap_pricer.dsdx(swap, x, y, t)

    def d2ds2(self, swap, x, y, t):

        dxds = self.dxds(swap, x, y, t)
        dsdx = self.swap_pricer.dsdx(swap, x, y, t)
        d2sdx2 = self.swap_pricer.d2sdx2(swap, x, y, t)

        return -d2sdx2*dxds/dsdx





class AnnuityPricer():


    def annuity_times_g(self, t, x, y, freq, kappa, bond_list):

        res = 0

        for bond in bond_list:
            res += freq * bond.price(x, y, t) * calculate_G(kappa, t, bond.maturity)
        return res

    def annuity_price(self, t, x, y, freq, bond_list):

        annuity = 0
        for bond in bond_list:
            annuity += freq*bond.price(x, y, t)

        return annuity


    def annuity_dx(self, t, x, y, freq, kappa, bond_list):
        return - self.annuity_times_g(t, x, y, freq, kappa, bond_list)

    def annuity_d2x(self, t, x, y, freq, kappa, bond_list):

        res = 0
        for bond in bond_list:
            res += freq * bond.price(x, y, t) * math.pow(calculate_G(kappa, t, bond.maturity), 2)

        return res



class Swaption():

    def __init__(self, expiry, maturity, coupon, swap: Swap):
        self.expiry = expiry
        self.maturity = maturity
        self.coupon = coupon
        self.swap = swap


class SwaptionPricer():

    def __init__(self, lambda_s, b_s, swap_pricer: SwapPricer):
        self.lambda_s = lambda_s
        self.b_s = b_s
        self.swap_pricer = swap_pricer


    def calculate_swaption(self, swaption: Swaption):

        annuity_pricer = AnnuityPricer()
        annuity_0 = annuity_pricer.annuity_price(0, 0, 0, swaption.swap.frequency, swaption.swap.bond_list)
        swap_0 = self.swap_pricer.price(swaption.swap, 0, 0, 0)

        dplus, dminus = self.d_plus_minus(swaption)
        #TODO check
        res = (swap_0 + swap_0*(1 + self.b_s)/self.b_s) * norm.cdf(dplus) - \
              (swaption.coupon + swap_0*(1-self.b_s)/self.b_s)*norm.cdf(dminus)
        res *= annuity_0

        return res

    def d_plus_minus(self, swaption: Swaption):

        swap_0 = self.swap_pricer.price(swaption.swap, 0, 0, 0)
        d = np.log((swap_0 + swap_0*(1 - self.b_s)/self.b_s)/(swaption.coupon + swap_0*(1-self.b_s)/self.b_s))

        numerator = self.b_s * self.lambda_s * math.sqrt(swaption.expiry)
        plus =  0.5 * math.pow(self.b_s, 2) * math.pow(self.lambda_s, 2)
        minus = -plus
        dplus = (d + plus)/numerator
        dminus = (d + minus)/numerator

        return dplus, dminus
