import numpy as np
import math
import scipy
from scipy.sparse import diags
from quassigaussian.mesher.linear_mesher import Mesher2d

ident = lambda x: np.where(x > 0, 1, 0)


class ConstructL2():

    def __init__(self, theta, mesher: Mesher2d):

        self.y_size_full = len(mesher.ygrid)
        self.delta_p = mesher.delta_py
        self.delta_m = mesher.delta_my

        self.delta_t = mesher.delta_t
        self.theta = theta

    def constructL2_rhs(self, r, mu_y):

        ident_muy = ident(mu_y)
        tmp = (1-ident_muy)/self.delta_m
        c_j = mu_y * (tmp - ident_muy/self.delta_p) -1/2 * r
        l_j = -mu_y * tmp
        u_j = mu_y * ident_muy/self.delta_p

        sparse_m = scipy.sparse.diags([l_j[1:], c_j[1:], u_j[1:-1]], offsets=[0, 1, 2], shape=(self.y_size_full-2, self.y_size_full))
        return sparse_m


    def post_boundary_update(self, v_new):
        v_new[0] = 2 * v_new[1] - v_new[2]
        v_new[-1] = 2 * v_new[-2] - v_new[-3]
        pass

    def constructL2(self, r, mu_y):

        ident_muy = ident(mu_y)
        tmp = (1-ident_muy)/self.delta_m
        c_j = mu_y * (tmp - ident_muy/self.delta_p) -1/2 * r
        l_j = -mu_y * tmp
        u_j = mu_y * ident_muy/self.delta_p

        c_j = c_j[1:-1]
        l_j = l_j[1:-1]
        u_j = u_j[1:-1]

        c_j[0] = c_j[0] + 2*l_j[0]
        u_j[0] = u_j[0] - l_j[0]

        c_j[-1] = c_j[-1] + 2*u_j[-1]
        l_j[-1] = l_j[-1] - u_j[-1]

        c_j = 1 - self.theta * self.delta_t * c_j
        l_j = - self.theta * self.delta_t * l_j
        u_j = - self.theta * self.delta_t * u_j

        l2_lhs = scipy.sparse.diags([l_j[1:], c_j, u_j[:-1]], offsets=[-1, 0, 1], shape=(self.y_size_full-2, self.y_size_full-2))
        return l2_lhs



