
class LinearLocalVolatility():

    def __init__(self, lambda_t: callable, alpha_t: callable, b_t: callable):
        self.lamba_t = lambda_t
        self.alpha_t = alpha_t
        self.b_t = b_t

    def calculate_vola(self, t: float, x: float):
        return self.lamba_t(t) * (self.alpha_t(t) + x * self.b_t(t))