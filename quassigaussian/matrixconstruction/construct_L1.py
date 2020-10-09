import numpy as np
import math
import scipy
from scipy.sparse import diags

class ConstructL1A():

    def __init__(self, theta, delta_t, delta_p, delta_m, msize):
        self.msize = len(delta_p)
        self.delta_p = delta_p
        self.delta_m = delta_m
        self.theta = theta
        self.delta_t = delta_t

    def construct_l1_operator(self, eta_sq, mu_x, r):
        ci = self.construct_ci(eta_sq, r)
        li = self.construct_li(mu_x, eta_sq)
        ui = self.construct_ui(mu_x, eta_sq)

        l1_lhs = self.construct_l1_lhs_operator(ci, li, ui, mu_x, eta_sq, r)
        l1_rhs = self.construct_l1_rhs_operator(ci, li, ui)

        return l1_lhs.toarray(), l1_rhs.toarray()


    def construct_l1_rhs_operator(self, ci, li, ui):
        # returns the full matrix.
        # the dimension will be (msize-2)x msize and the operator will be applied to the msize vector.
        sparse_m = scipy.sparse.diags([li[1:], ci[1:], ui[1:-1]], offsets=[0, 1, 2], shape=(self.msize-2, self.msize))
        return sparse_m

    def construct_l1_lhs_operator(self, ci, li, ui, mu_x, eta_sq, r):
        b1, b2, bI, bI_m1 = self.calculate_boundary_values(mu_x, eta_sq, r)
        li, ci, ui = self.add_boundary_values(ci, li, ui, b1, b2, bI, bI_m1)
        sparse_m = scipy.sparse.diags([li, ci, ui], offsets=[-1, 0, 1], shape=(self.msize-2, self.msize-2))

        return sparse_m

    def construct_ci(self, eta_sq, r):
        ci = (self.delta_p - self.delta_m)/(self.delta_p*self.delta_m) -eta_sq/(self.delta_p * self.delta_m) - 1/2 * r
        return ci

    def construct_li(self, mu_x, eta_sq):
        li = - self.delta_p/((self.delta_m + self.delta_p)*self.delta_m) * mu_x + eta_sq/((self.delta_p + self.delta_m)*self.delta_m)
        return li

    def construct_ui(self, mu_x, eta_sq):
        ui = self.delta_m/((self.delta_m + self.delta_p)*self.delta_p)*mu_x + eta_sq/((self.delta_p + self.delta_m)*self.delta_p)
        return ui

    def calculate_boundary_values(self, mu_x, eta_sq, r):

        k_0 = - mu_x/self.delta_p[0] + eta_sq/((self.delta_p[1] + self.delta_p[0])*self.delta_p[0]) - 1/2*r
        k_1 = mu_x/self.delta_p[0] - eta_sq/(self.delta_p[1]*self.delta_p[0])
        k_2 = eta_sq/(self.delta_p[0]*(self.delta_p[1]) + self.delta_p[0])

        k_IpI = mu_x/self.delta_p[-2] + eta_sq/(self.delta_p[-2]*(self.delta_p[-2] + self.delta_p[-3])) - 1/2*r
        k_I = (-mu_x/self.delta_p[-2] - eta_sq/(self.delta_p[-2]*self.delta_p[-3]))
        k_Im1 = - eta_sq/(self.delta_p[-3]*(self.delta_p[-2] + self.delta_p[-3]))

        b1 = k_1*self.theta*self.delta_t/(1-self.theta*self.delta_t*k_0)
        b2 = k_2*self.theta*self.delta_t/(1-self.theta*self.delta_t*k_0)

        bI = 1 - (self.theta*self.delta_t * k_I)/(1-self.theta*self.delta_t * k_IpI)
        bI_m1 = 1 - (self.theta*self.delta_t * k_Im1)/(1-self.theta*self.delta_t * k_IpI)

        return b1, b2, bI, bI_m1


    def add_boundary_values(self, ci, li, ui, b1, b2, bI, bI_m1):

        # take only the inner values
        ci = ci[1:-1].copy()
        li = li[1:-1].copy()
        ui = ui[1:-1].copy()

        ci[0] = ci[0] + b1*li[0]
        ui[0] = ui[0] + b2*li[0]

        ci[-1] = ci[-1] + bI*ui[-1]
        ci[-2] = ci[-2] + bI_m1*ui[-2]
        return li, ci, ui


    def construct_omega(self, x_sol, t):
        pass


if __name__ == '__main__':

    theta = 0.5
    r = 0.05
    delta_t = 0.1

    eta_sq = 0.2
    mu_x = 0.7
    from quassigaussian.mesher.linear_mesher import create_mesher_2d, calculate_delta

    xgrid, ygrid, xv, yv = create_mesher_2d(0, 100, 10, 0, 100, 5)
    delta_p, delta_m = calculate_delta(xgrid)

    construct_l1 = ConstructL1A(theta, delta_t, delta_p, delta_m, len(delta_p))
    l1_lhs, l1_rhs = construct_l1.construct_l1_operator(eta_sq, mu_x, r)
    print("pause")