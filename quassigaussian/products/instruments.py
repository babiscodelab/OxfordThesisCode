from numpy import arange

class Bond():

    def __init__(self, maturity: float):
        self.maturity = maturity




class Swap():

    def __init__(self, T0: int, TN: int, frequency: float):

        self.T0 = T0
        self.TN = TN
        self.start = T0 + frequency # T1
        self.frequency = frequency

        self.bond_T0 = Bond(self.T0)
        self.bond_TN = Bond(self.TN)

        payment_schedule = arange(self.start, self.TN + 0.0000001, self.frequency)
        self.bond_list = self.get_bond_list(payment_schedule)
        self.annuity = Annuity(self.bond_list, self.frequency)

    def get_bond_list(self, payment_schedule):

        bond_list = []
        for payment_date in payment_schedule:
            bond_list.append(Bond(payment_date))

        return bond_list

class Swaption():

    def __init__(self, expiry: float, coupon: float, swap: Swap):
        self.expiry = expiry
        self.coupon = coupon
        self.swap = swap


class Annuity():

    def __init__(self, bond_list, freq):
        self.bond_list = bond_list
        self.freq = freq