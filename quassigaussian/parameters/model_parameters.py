from quassigaussian.parameters.volatility.local_volatility import LinearLocalVolatility
import tensorflow as tf

class QgModelParameters():

    def __init__(self, kappa, volatility: LinearLocalVolatility, number_factors):

        self.kappa = kappa
        self.volatility = volatility
        self.number_factors = number_factors