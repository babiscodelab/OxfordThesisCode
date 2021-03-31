from report.directories import output_dgm_model
import tensorflow.keras as keras
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from quassigaussian.products.pricer import BondPricer
from quassigaussian.products.instruments import Bond
from quassigaussian.dgm.qg_dgm import QuasiGaussianDGM
import numpy as np

def plot_bond(dgm_file_path, epoch):

    dgm_epoch_path = os.path.join(dgm_file_path, "dgm_epoch_" + str(epoch))
    model = keras.models.load_model(dgm_epoch_path)


    meta_obj_path = os.path.join(dgm_file_path, "meta_data.obj")

    with open(meta_obj_path, "rb") as f:
        meta_obj = pickle.load(f)

    t_max = meta_obj.swaption.swap.T0
    n_samples = 100
    time_lin_space = tf.linspace(0., t_max, n_samples)
    time_lin_space = tf.expand_dims(time_lin_space, axis=1)
    x = tf.constant(0., float, (n_samples, 1))
    y = tf.constant(0., float, (n_samples, 1))

    bond_pricer = meta_obj.swaption_pricer.swap_pricer.bond_pricer

    bond = Bond(t_max)
    bond_value_t = bond_pricer.price(bond, x, y, time_lin_space).reshape(n_samples)
    bond_value_dgm = model((time_lin_space, x, y))[:, 0]

    fig, ax1 = plt.subplots()
    ax1.plot(time_lin_space, bond_value_t, "r-+", label="Bond exact")
    ax1.plot(time_lin_space, bond_value_dgm, "b-+", label="Bond DGM")
    ax1.set_title("Bond value, Exact vs Dgm")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Bond Value")
    plt.legend()

    ax2 = ax1.twinx()
    ax2.plot(time_lin_space, 10000*(bond_value_dgm.numpy()-bond_value_t)/bond_value_t, "k--", label="Error")
    ax2.set_ylabel("Error (bps)")
    plt.legend()


    fig, ax1 = plt.subplots()

    x_lin_space = tf.linspace(-0.2, 0.2, n_samples)
    x_lin_space = tf.expand_dims(x_lin_space, axis=1)
    t = tf.constant(0., float, (n_samples, 1))
    y = tf.constant(0., float, (n_samples, 1))

    bond_value_x = bond_pricer.price(bond, x_lin_space, y, t).reshape(n_samples)
    bond_value_dgm = model((t, x_lin_space, y))[:, 0]

    ax1.plot(x_lin_space, bond_value_x, "r-+", label="Bond exact")
    ax1.plot(x_lin_space, bond_value_dgm.numpy(), "b-+", label="Bond DGM")
    ax1.set_title("Bond value, Exact vs Dgm")
    ax1.set_xlabel("x")
    ax1.set_ylabel("Bond Value")
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(x_lin_space, 10000*(bond_value_dgm.numpy()-bond_value_x)/bond_value_x, "k--", label="Error")
    ax2.set_ylabel("Error (bps)")
    ax2.legend()




if __name__ == "__main__":

    #file_path = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\report\data\output_data\dgm_model\2021_02_27\dgm-3"

    # Good for 1Y
    #file_path = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\report\data\output_data\dgm_model\2021_02_26\dgm-11"
    # #
    # #
    # file_path = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\report\data\output_data\dgm_model\2021_02_28\dgm-33"
    # #
    # #file_path = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\report\data\output_data\dgm_model\2021_03_01\dgm-4"
    # file_path = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\report\data\output_data\dgm_model\2021_03_01\dgm-8"
    #
    # # can use it for the 10year
    # file_path = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\report\data\output_data\dgm_model\2021_03_01\dgm-11"
    # # also good
    # file_path = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\report\data\output_data\dgm_model\2021_03_01\dgm-12"
    # # 10000
    # #file_path = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\report\data\output_data\dgm_model\2021_03_03\dgm-4"
    # file_path = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\report\data\output_data\dgm_model\2021_03_03\dgm-5"
    #
    # # test 14000
    # # not bad
    #file_path = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\report\data\output_data\dgm_model\2021_03_04\dgm-1"
    # #
    # # #plot_bond(file_path, "5000")
    # # None
    # # The bellow is good
    # Good for the 10 Y
    file_path = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\report\data\output_data\dgm_model\2021_03_04\dgm-4"
    #
    # # not
    # file_path = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\report\data\output_data\dgm_model\2021_03_09\dgm-4"
    #
    plot_bond(file_path, "None")

    plt.show()