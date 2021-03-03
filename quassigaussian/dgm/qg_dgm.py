from quassigaussian.finitedifference.mesher.grid_boundaries import calculate_x_boundaries3, calculate_y_boundaries, calculate_x_moments
from quassigaussian.parameters.model_parameters import QgModelParameters
from quassigaussian.parameters.volatility.local_volatility import LinearLocalVolatilityTf
from quassigaussian.dgm.dgm_parameters import DgmParameters
import quassigaussian.dgm.DGM
import tensorflow as tf
import numpy as np
from quassigaussian.products.instruments import Swaption, Swap
from quassigaussian.products.pricer import SwaptionPricer, SwapPricer
# %% Parameters
from quassigaussian.dgm.DGM import DGMNet
from report.directories import output_dgm_model, date_timestamp
import os
from qgtests.utis import get_mock_yield_curve_const
from report.utils import get_nonexistant_path

import pickle


class QuasiGaussianDGM():

    def __init__(self, model_parameters: QgModelParameters, dgm_parameters: DgmParameters,
                 swaption: Swaption, swaption_pricer: SwaptionPricer, output_path):

        self.dgm_parameters = dgm_parameters

        self.model_parameters = model_parameters
        self.volatility_func = self.model_parameters.volatility
        self.kappa = self.model_parameters.kappa
        self.qg_n_factors = self.model_parameters.number_factors

        self.swaption = swaption
        self.swaption_pricer = swaption_pricer
        self.output_path = output_path

        self.xmin, self.xmax = calculate_x_boundaries3(swaption_expiry, self.kappa, self.volatility_func, alpha=2.5)

        #self.xmin, self.xmax = -0.2, 0.2

        self.ymin, self.ymax = calculate_y_boundaries(swaption_expiry, self.kappa, self.volatility_func, alpha=2.5)


    def sampler2(self, nSim_interior, nSim_terminal):
        t_interior = tf.random.uniform(minval=0, maxval=self.swaption.expiry, shape=[nSim_interior, 1])

        x_mean, x_std = calculate_x_moments(t_interior, self.kappa, self.volatility_func)
        x_interior = tf.random.normal(mean=x_mean, stddev=x_std, shape=[nSim_interior, 1])

        t_terminal = self.swaption.expiry * tf.ones((nSim_terminal, 1))
        x_mean, x_std = calculate_x_moments(t_terminal, self.kappa, self.volatility_func)
        x_terminal = tf.random.normal(mean=x_mean, stddev=x_std, shape=[nSim_terminal, 1])

        y_interior = tf.random.uniform(minval=self.ymin, maxval=self.ymax, shape=[nSim_interior, self.qg_n_factors])
        y_terminal = tf.random.uniform(minval=self.ymin, maxval=self.ymax, shape=[nSim_terminal, self.qg_n_factors])
        return t_interior, x_interior, y_interior, t_terminal, x_terminal, y_terminal

    # %% Sampling function - randomly sample time-space pairs

    def sampler(self, x_low, x_high, y_low, y_high, nSim_interior, nSim_terminal):
        ''' Sample time-space points from the function's domain; points are sampled
            uniformly on the interior of the domain, at the initial/terminal time points
            and along the spatial boundary at different time points.

        Args:
            nSim_interior: number of space points in the interior of the function's domain to sample
            nSim_terminal: number of space points at terminal time to sample (terminal condition)
        '''

        # Sampler #1: domain interior
        t_interior = tf.random.uniform(minval=0, maxval=self.swaption.expiry, shape=[nSim_interior, 1])
        x_interior = tf.random.uniform(minval=x_low, maxval=x_high, shape=[nSim_interior, self.qg_n_factors])
        y_interior = tf.random.uniform(minval=y_low, maxval=y_high, shape=[nSim_interior, self.qg_n_factors])

        #y_interior = (y_interior + tf.transpose(y_interior, perm=(0, 2, 1)))/2
        #y_interior = tf.reshape(y_interior, shape=(nSim_interior, int(self.qg_n_factors * (self.qg_n_factors + 1) / 2)))

        # Sampler #2: initial/terminal condition
        t_terminal = self.swaption.expiry * tf.ones((nSim_terminal, 1))
        x_terminal = tf.random.uniform(minval=x_low, maxval=x_high, shape=[nSim_terminal, self.qg_n_factors])
        y_terminal = tf.random.uniform(minval=y_low, maxval=y_high, shape=[nSim_terminal, self.qg_n_factors])
        #y_terminal = (y_terminal + tf.transpose(y_terminal, perm=(0, 2, 1)))/2
        #y_terminal = tf.reshape(y_terminal, shape=(nSim_interior, int(self.qg_n_factors * (self.qg_n_factors + 1) / 2)))

        return t_interior, x_interior, y_interior, t_terminal, x_terminal, y_terminal

    # %% Loss function for Quasi-Gaussian

    def loss(self, model, t_interior, x_interior, y_interior, t_terminal, x_terminal, y_terminal):
        ''' Compute total loss for training.

        Args:
            model:      DGM model object
            t_interior: sampled time points in the interior of the function's domain
            S_interior: sampled space points in the interior of the function's domain
            t_terminal: sampled time points at terminal point (vector of terminal times)
            S_terminal: sampled space points at terminal time
        '''

        # Loss term #1: PDE
        # compute function value and derivatives at current sampled points
        with tf.GradientTape(persistent=True) as tape:
            input = (t_interior, x_interior, y_interior)
            V = model(input)
            V_x = tape.gradient(V, x_interior)

        V_y = tape.gradient(V, y_interior)
        V_t = tape.gradient(V, t_interior)
        V_xx = tape.gradient(V_x, x_interior)

        vola = self.volatility_func.calculate_vola(t_interior, x_interior, y_interior)
        volatr = tf.transpose(vola, perm=[0, 1])
        short_rate = 0.06 + tf.reduce_sum(x_interior, axis=[1])
        short_rate = tf.expand_dims(short_rate, 1)
        # loss_inter = V_t + tf.matmul(V_x, (tf.matmul(y_interior, tf.ones(shape=(1000, 1, 1))) - self.kappa*x_interior)) \
        #           + 1/2 * tf.linalg.trace(V_xx * vola*volatr) \
        #           + tf.linalg.trace((volatr * vola - self.kappa*y_interior - y_interior*self.kappa)*V_y) \
        #           - short_rate

        loss_inter = V_t + V_x * (y_interior - self.kappa*x_interior) + 1.0/2 * V_xx*vola*volatr + \
                            (volatr * vola - 2*self.kappa*y_interior)*V_y - short_rate*V

        loss_inter = tf.reduce_mean(tf.square(loss_inter))
        target_payoff = 1

        fitted_payoff = model((t_terminal, x_terminal, y_terminal))
        loss_terminal = 100*tf.reduce_mean(tf.square(target_payoff-fitted_payoff))

        L = loss_inter + loss_terminal

        return L, loss_inter, loss_terminal


    def train_network(self):

        nodes_per_layer = self.dgm_parameters.nodes_per_layer
        num_layers = self.dgm_parameters.num_layers
        pde_dim = int(self.qg_n_factors*(self.qg_n_factors + 3)/2)

        model = DGMNet(nodes_per_layer, num_layers, pde_dim)

        learning_rate = 0.0001
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        current_learning_rate = get_learning_rate(1)
        for epoch in range(50000):

            new_learning_rate = get_learning_rate(epoch)
            if new_learning_rate != current_learning_rate:
                print("Update learning rate")
                optimizer.lr.assign(new_learning_rate)
                current_learning_rate = new_learning_rate

            for i in range(1):

                t_interior, x_interior, y_interior, t_terminal, x_terminal, y_terminal = self.sampler2(2000, 400)

                # t_interior, x_interior, y_interior, t_terminal, x_terminal, y_terminal = self.sampler(self.xmin, self.xmax,
                #                                                                                       self.ymin, self.ymax, 1000, 200)
                t_interior = tf.Variable(t_interior)
                t_terminal = tf.Variable(t_terminal)
                x_interior = tf.Variable(x_interior)
                y_interior = tf.Variable(y_interior)
                x_terminal = tf.Variable(x_terminal)
                y_terminal = tf.Variable(y_terminal)

                with tf.GradientTape() as tape:
                    loss_value, l_int, l_bound = self.loss(model, t_interior, x_interior, y_interior, t_terminal, x_terminal, y_terminal)
                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                print(loss_value, l_int, l_bound)

        if (epoch % 1000==0):
            save_model(model, self.output_path, epoch)


            print("new sample")
        pass
        save_model(model, self.output_path, None)

        return model


def get_learning_rate(epoch):

    epoch = epoch*25

    if epoch < 25000:
        return np.power(10., -4)
    elif epoch < 50000:
        return np.power(10., -5)*5
    elif epoch < 100000:
        return np.power(10., -5)
    elif epoch < 150000:
        return np.power(10., -6)*5
    elif epoch < 200000:
        return np.power(10., -6)
    elif epoch < 225000:
        return np.power(10., -7)*5
    else:
        return np.power(10., -7)


def save_model(model, file_path, epoch=None):
    file_name = "dgm_epoch_{}".format(epoch)
    file_path = os.path.join(file_path, file_name)
    model.save(file_path)

def save_meta_data(qg_model: QuasiGaussianDGM, file_path):
    object = qg_model
    os.mkdir(file_path)
    file_path_save = os.path.join(file_path, "meta_data.obj")
    filehandler = open(file_path_save, 'wb')
    pickle.dump(object, filehandler)

if __name__ == "__main__":

    output_path = os.path.join(output_dgm_model, date_timestamp, "dgm")
    output_path = get_nonexistant_path(output_path)

    dgm_param = DgmParameters(num_layers=3)

    number_factors = 1
    local_volatility = LinearLocalVolatilityTf.from_const(number_factors, 15, 0.4, 0.06, 0.2)
    kappa = 0.3
    qg_model_parameters = QgModelParameters(kappa, local_volatility, number_factors)

    swaption_expiry = 10
    swap = Swap(swaption_expiry, 2, 0.5)

    initial_curve = get_mock_yield_curve_const(rate=0.06)

    swap_pricer = SwapPricer(initial_curve, kappa)
    swaption_pricer = SwaptionPricer(swap_pricer)


    coupon = swap_pricer.price(swap, 0, 0, 0)
    swaption = Swaption(swaption_expiry, coupon, swap)


    qg_dgm = QuasiGaussianDGM(qg_model_parameters, dgm_param, swaption, swaption_pricer, output_path)
    save_meta_data(qg_dgm, output_path)

    model = qg_dgm.train_network()

    # open session
