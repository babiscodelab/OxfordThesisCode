import matplotlib.pyplot as plt
import matplotlib
from quassigaussian.montecarlo.simulations import ResultSimulatorObj
matplotlib.rcParams['text.usetex'] = True

class QuasiGaussianPlotter():

    def __init__(self, save_path=None, fig_format="png"):
        self.save_path = save_path
        self.fig_format = fig_format

    def save_fig(self, fig_list):

        for fig in fig_list:
            fig.savefig()

class ApproximatorPlotter(QuasiGaussianPlotter):


    def plot_xy_approximation(self, res_simu: ResultSimulatorObj):


        if res_simu.annuity_measure:
            measure_str = " under annuity measure."
        else:
            measure_str = " under risk neutral measure."


        xlabel = "Time (years)"

        plt.figure()
        plt.plot(res_simu.time_grid, res_simu.x.T)
        plt.title("Simulations of X" + measure_str)
        plt.xlabel(xlabel)
        plt.ylabel("X(t)")

        plt.figure()
        plt.plot(res_simu.time_grid, res_simu.y.T)
        plt.title("Simulations of Y" + measure_str)
        plt.xlabel(xlabel)
        plt.ylabel("Y(t)")

        plt.figure()
        plt.plot(res_simu.time_grid, res_simu.x_bar)
        plt.title("Expected value of X" + measure_str)
        plt.xlabel(xlabel)
        plt.ylabel(r"$E[X(t)]$")


        plt.figure()
        plt.plot(res_simu.time_grid, res_simu.y_bar)
        plt.title("Expected value of Y" + measure_str)
        plt.xlabel(xlabel)
        plt.ylabel(r"$E[Y(t)]$")
