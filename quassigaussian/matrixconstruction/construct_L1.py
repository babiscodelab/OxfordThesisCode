import numpy as np
import math
import scipy
from scipy.sparse import diags

class ConstructL1A():

    def __init__(self, theta, delta_t, delta_p, delta_m, msize):
        self.msize = msize
        self.delta_p = delta_p
        self.delta_m = delta_m
        self.theta = theta
        self.delta_t = delta_t

    def construct_l1_operator(self, eta_sq, mu_x, r):
        ci = self.construct_ci(eta_sq, r)
        li = self.construct_li(mu_x, eta_sq)
        ui = self.construct_ui(mu_x, eta_sq)



    def construct_l1_rhs_operator(self, eta_sq, mu_x, r):
        pass

    def construct_l1_lhs_operator(self, ci, li, ui, mu_x, eta_sq, r):
        b1, b2 = self.calculate_boundary_values(mu_x, eta_sq, r)
        l1, c1, ui = self.add_boundary_values(ci, li, ui, b1, b2)
        sparse_m = scipy.sparse.diags([li, ci, ui], offsets=[-1, 0, 1], shape=(self.msize, self.msize))

        return sparse_m

    def construct_ci(self, eta_sq, r):
        ci = (self.delta_p - self.delta_m)/(self.delta_p + self.delta_m) -eta_sq/(self.delta_p * self.delta_m) - 1/2 * r
        return ci

    def construct_li(self, mu_x, eta_sq):
        li = - self.delta_p/((self.delta_m + self.delta_p)*self.delta_m) * mu_x + eta_sq/((self.delta_p + self.delta_m)*self.delta_m)
        return li

    def construct_ui(self, mu_x, eta_sq):
        ui = self.delta_m/((self.delta_m + self.delta_p)*self.delta_p)*mu_x + eta_sq/((self.delta_p+self.delta_m)*self.delta_m)
        return ui

    def calculate_boundary_values(self, mu_x, eta_sq, r):

        k_0 = - mu_x/self.delta_p[0] + eta_sq/((self.delta_p[1] + self.delta_p[0])*self.delta_p[0]) - 1/2*r
        k_1 = mu_x/self.delta_p[0] - eta_sq/(self.delta_p[1]*self.delta_p[0])
        k_2 = eta_sq/(self.delta_p[0]*(self.delta_p[1]) + self.delta_p[0])



        b1 = k_1*self.theta*self.delta_t/(1-self.theta*self.delta_t*k_0)
        b2 = k_2*self.theta*self.delta_t/(1-self.theta*self.delta_t*k_0)

        return b1, b2


    def add_boundary_values(self, ci, li, ui, b1, b2, bI, bI_1):
        ci = ci.copy()
        li = li.copy()
        ui = ui.copy()

        ci[0] = ci[0] + b1*li[0]
        ui[0] = ui[0] + b2*li[0]

        ci[-1] = ci[-1] + bI*ui[-1]
        ci[-2] = ci[-2] + bI_1*ui[-2]
        return li, ci, ui