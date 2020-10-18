import numpy as np
import math
import scipy
from scipy.sparse import diags
from quassigaussian.mesher.linear_mesher import Mesher2d


class ConstructL1A():

    def __init__(self, theta, mesher: Mesher2d):

        self.theta = theta
        self.msize = mesher.xdim
        self.delta_t = mesher.delta_t
        self.delta_p = mesher.delta_px
        self.delta_m = mesher.delta_mx

    def update_data(self, eta_sq, mu_x, r):

        ci_full, li_full, ui_full = self.construct_diagonal_elements(eta_sq, mu_x, r)
        self.l_1 = li_full[1]
        self.u_I = ui_full[-2]


        self.k_0, self.k_1, self.k_2, self.k_IpI, self.k_I, self.k_Im1 = self.calcualate_boundary_coefficients(mu_x, eta_sq, r)
        self.b1, self.b2, self.bI, self.bI_m1 = self.calculate_boundary_values(self.k_0, self.k_1, self.k_2, self.k_IpI, self.k_I, self.k_Im1)

        self.l1_lhs = self.construct_l1_lhs_matrix(ci_full, li_full, ui_full, self.b1, self.b2, self.bI, self.bI_m1)
        self.l1_rhs = self.construct_l1_rhs_matrix(eta_sq, mu_x, r)


    def construct_l1_rhs_matrix(self, eta_sq, mu_x, r):

        ci, li, ui = self.construct_diagonal_elements(eta_sq, mu_x, r)
        # returns the full matrix.
        # the dimension will be (msize-2) x msize and the operator will be applied to the msize vector.
        sparse_m = scipy.sparse.diags([li[1:], ci[1:], ui[1:-1]], offsets=[0, 1, 2], shape=(self.msize-2, self.msize))
        return sparse_m


    def construct_l1_lhs_matrix(self, ci, li, ui, b1, b2, bI, bI_m1):
        li, ci, ui = self.add_boundary_values(ci, li, ui, b1, b2, bI, bI_m1)
        li = - self.theta * self.delta_t * li
        ci = 1 - self.theta * self.delta_t * ci
        ui = - self.theta * self.delta_t * ui
        l1_lhs = scipy.sparse.diags([li[1:], ci, ui[:-1]], offsets=[-1, 0, 1], shape=(self.msize - 2, self.msize - 2))
        return l1_lhs

    def construct_diagonal_elements(self, eta_sq, mu_x, r):

        ci = self.construct_ci(eta_sq, r)
        li = self.construct_li(mu_x, eta_sq)
        ui = self.construct_ui(mu_x, eta_sq)

        return ci, li, ui


    def post_boundary_update(self, v_sol, v_sol_prev, l2_v):
        v_sol[0] = self.b1 * v_sol[1] + self.b2 * v_sol[2] + self.f_lower_calc(v_sol_prev, l2_v)
        v_sol[-1] = self.bI * v_sol[-2] + self.bI_m1 * v_sol[-3] + self.f_upper_calc(v_sol_prev, l2_v)

    def calculate_omega_rhs(self, v_sol, l2_v):

        omega_v = np.zeros(self.msize-2)
        omega_v[0] = self.f_lower_calc(v_sol, l2_v) * self.l_1
        omega_v[-1] = self.f_upper_calc(v_sol, l2_v) * self.u_I

        return omega_v

    def f_upper_calc(self, v_sol, l2_v):
        f_upper = (1 - self.theta)*self.delta_t*(v_sol[-1] * self.k_IpI + v_sol[-2] * self.k_I + v_sol[-3] * self.k_Im1) \
                      + self.delta_t*l2_v[-1] + v_sol[-1]
        f_upper *= 1/(1-self.theta * self.delta_t * self.k_IpI)
        return f_upper

    def f_lower_calc(self, v_sol, l2_v):
        f_lower = ((1 - self.theta) * self.delta_t * (v_sol[0] * self.k_0 + v_sol[1] * self.k_1 + v_sol[2] * self.k_2)) \
                  + v_sol[0] + self.delta_t * l2_v[0]
        f_lower *= 1 / (1 - self.theta * self.delta_t * self.k_0)
        return f_lower

    def calculate_v_lower(self, v_sol, l2_v):
        v_lower = self.b1 * v_sol[1] + self.b2*v_sol[2] + self.f_lower_calc(v_sol, l2_v)
        return v_lower

    def calculate_v_upper(self, v_sol, l2_v):
        v_upper = self.bI * v_sol[-2] + self.bI_m1 * v_sol[-3] * self.f_upper_calc(v_sol, l2_v)
        return v_upper

    def construct_ci(self, eta_sq, r):
        ci = (self.delta_p - self.delta_m)/(self.delta_p*self.delta_m) - eta_sq/(self.delta_p * self.delta_m) - 1/2 * r
        return ci

    def construct_li(self, mu_x, eta_sq):
        li = - self.delta_p/((self.delta_m + self.delta_p)*self.delta_m) * mu_x + eta_sq/((self.delta_p + self.delta_m)*self.delta_m)
        return li

    def construct_ui(self, mu_x, eta_sq):
        ui = self.delta_m/((self.delta_m + self.delta_p)*self.delta_p)*mu_x + eta_sq/((self.delta_p + self.delta_m)*self.delta_p)
        return ui

    def calcualate_boundary_coefficients(self, mu_x, eta_sq, r):

        k_0 = - mu_x[0] / self.delta_p[0] + eta_sq[0] / ((self.delta_p[1] + self.delta_p[0]) * self.delta_p[0]) - 1 / 2 * r[0]
        k_1 = mu_x[0] / self.delta_p[0] - eta_sq[0] / (self.delta_p[1] * self.delta_p[0])
        k_2 = eta_sq[0] / (self.delta_p[0] * (self.delta_p[1] + self.delta_p[0]))

        k_IpI = mu_x[-1] / self.delta_p[-2] + eta_sq[-1] / (self.delta_p[-2] * (self.delta_p[-2] + self.delta_p[-3])) - 1 / 2 * r[-1]
        k_I = -mu_x[-1] / self.delta_p[-2] - eta_sq[-1] / (self.delta_p[-2] * self.delta_p[-3])
        k_Im1 = eta_sq[-1] / (self.delta_p[-3] * (self.delta_p[-2] + self.delta_p[-3]))

        return k_0, k_1, k_2, k_IpI, k_I, k_Im1

    def calculate_boundary_values(self, k_0, k_1, k_2, k_IpI, k_I, k_Im1):

        b1 = k_1*self.theta*self.delta_t/(1-self.theta*self.delta_t*k_0)
        b2 = k_2*self.theta*self.delta_t/(1-self.theta*self.delta_t*k_0)

        bI = (self.theta*self.delta_t * k_I)/(1-self.theta*self.delta_t * k_IpI)
        bI_m1 = (self.theta*self.delta_t * k_Im1)/(1-self.theta*self.delta_t * k_IpI)

        return b1, b2, bI, bI_m1


    def add_boundary_values(self, ci, li, ui, b1, b2, bI, bI_m1):

        # take only the inner values
        ci = ci[1:-1].copy()
        li = li[1:-1].copy()
        ui = ui[1:-1].copy()

        ci[0] = ci[0] + b1*li[0]
        ui[0] = ui[0] + b2*li[0]

        ci[-1] = ci[-1] + bI*ui[-1]
        li[-1] = li[-1] + bI_m1*ui[-1]
        return li, ci, ui

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