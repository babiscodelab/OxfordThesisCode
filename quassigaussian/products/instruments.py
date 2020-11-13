
class Bond():

    def __init__(self, maturity: float):
        self.maturity = maturity




class Swap():

    def __init__(self, T0: int, TN: int, frequency: int):

        self.T0 = T0
        self.TN = TN
        self.start = T0 + frequency # T1
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

class Swaption():

    def __init__(self, expiry: float, maturity: float, coupon: float, swap: Swap):
        self.expiry = expiry
        self.maturity = maturity
        self.coupon = coupon
        self.swap = swap



