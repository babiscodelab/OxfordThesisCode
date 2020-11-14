import scipy.integrate

def lambda_s_bar(lambda_square: callable, T0):
        return scipy.integrate.quad(lambda_square, 0, T0)[0] / T0


def w_s_wrapper(lambda_square):

    def w_s(t, T0):

        numerator = lambda_square(t) * scipy.integrate.quad(lambda_square, 0, t)[0]

        def ws_integral(u):
            return scipy.integrate.quad(lambda_square, 0, u)[0] * lambda_square(u)

        denominator = scipy.integrate.quad(ws_integral, 0, T0)[0]

        return numerator/denominator

    return w_s


def b_s_bar(w_s: callable, b_s: callable, T0):

   def b_integral(t, w_s, b_s):
       return w_s(t, T0) * b_s(t)

   b_bar = scipy.integrate.quad(b_integral, 0, T0, args=(w_s, b_s, ))[0]

   return b_bar