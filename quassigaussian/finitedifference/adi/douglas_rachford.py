import numpy as np
from quassigaussian.finitedifference.matrixconstruction.construct_L1 import  ConstructL1A
from quassigaussian.finitedifference.matrixconstruction.construct_L2 import  ConstructL2
from quassigaussian.finitedifference.matrixconstruction.coefficients import CoefficientConstruction
from quassigaussian.finitedifference.mesher.linear_mesher import Mesher2d
from quassigaussian.linearalgebra.solve_banded import solve_banded_array
from quassigaussian.curves.libor import LiborCurve
from quassigaussian.parameters.volatility.local_volatility import LinearLocalVolatility

class DouglasRachfordAdi():

    def __init__(self, theta, mesher: Mesher2d, initial_curve: LiborCurve, kappa, local_volatility: LinearLocalVolatility):
        """
        Solve the Quasi-Gaussian ADI
        :param theta:
        :param mesher:
        :param initial_curve:
        :param kappa:
        :param local_volatility:
        """

        self.mesher = mesher
        self.theta = theta
        self.coefficient_constr = CoefficientConstruction(mesher.xmesh, mesher.umesh, initial_curve, kappa, local_volatility)


    def solve(self, v_init):

        self.constructL1 = ConstructL1A(self.theta, self.mesher)
        self.constructL2 = ConstructL2(self.theta, self.mesher)

        time_grid_reversed = self.mesher.tgrid[::-1]

        v_new = v_init.copy()

        for time_k in time_grid_reversed[1:]:
            # Iterate backwards
            print(time_k)
            v_old = v_new.copy()
            # the coefficients are valued at t_theta
            self.coefficient_constr.update_coefficients(time_k+self.mesher.delta_t*(1-self.theta))
            v_new = self.solve_eq(v_old, self.coefficient_constr.mu_x, self.coefficient_constr.mu_u,
                                  self.coefficient_constr.eta_sq, self.coefficient_constr.r)

        return v_new

    def solve_eq(self, vold, mu_x, mu_u, eta_sq, r):


        l1_vold = self.apply_L1(vold, eta_sq, mu_x, r)
        l2_vold = self.apply_L2(vold, r, mu_u)

        u_temp = np.zeros(shape=(self.mesher.xdim, self.mesher.udim))
        v_new = np.zeros(shape=(self.mesher.xdim, self.mesher.udim))

        for j in range(len(self.mesher.ugrid)):
            # Apply scheme in the direction of x
            self.constructL1.update_data(eta_sq[:, j], mu_x[:, j], r)
            rhs = vold[1:-1, j] + (1-self.theta) * self.mesher.delta_t * l1_vold[:, j] + self.mesher.delta_t * l2_vold[1:-1, j]
            lhs_matrix = self.constructL1.l1_lhs
            # pass th lhs omega to the right so you can solve for V
            lhs_omega = self.constructL1.calculate_omega_rhs(vold[:, j], l2_vold[:, j])
            rhs = lhs_omega*self.theta * self.mesher.delta_t + rhs
            u_temp[1:-1, j] = solve_banded_array(lhs_matrix, rhs)
            self.constructL1.post_boundary_update(u_temp[:, j], vold[:, j], l2_vold[:, j])


        for i in range(len(self.mesher.xgrid)):
            # Apply scheme in the direction of u
            rhs = u_temp[i, 1:-1] - self.theta * self.mesher.delta_t * l2_vold[i, 1:-1]
            l2_lhs = self.constructL2.constructL2(r[i], mu_u[i, :])
            v_new[i, 1:-1] = solve_banded_array(l2_lhs, rhs)
            self.constructL2.post_boundary_update(v_new[i, :])

        # result of one time step
        return v_new

    def apply_L1(self, v_old, eta_sq, mu_x, r):

        # Apply in direction x RHS
        u_outp = np.zeros(shape=(self.mesher.xdim-2, self.mesher.udim))
        for j in range(len(self.mesher.ugrid)):
            l1_rhs = self.constructL1.construct_l1_rhs_matrix(eta_sq[:,j], mu_x[:,j], r)
            u_outp[:, j] = l1_rhs*v_old[:, j]

        return u_outp


    def apply_L2(self, v_old, r, mu_u):

        # Direction u (can use y instead as well), RHS
        v_new = np.zeros(shape=(self.mesher.xdim, self.mesher.udim))
        for i in range(len(self.mesher.xgrid)):
            l2 = self.constructL2.constructL2_rhs(r[i], mu_u[i, :])
            v_new[i, 1:-1] = l2*v_old[i, :]
            v_new[i, 0] = 2*v_new[i, 1] - v_new[i, 2]
            v_new[i, -1] = 2*v_new[i, -2] - v_new[i, -3]

        return v_new


