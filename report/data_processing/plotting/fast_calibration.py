import pandas as pd
import os
import numpy as np
from report.directories import output_data_raw, output_plots_approx_solution, output_tables_approx_solution
import matplotlib.pyplot as plt
from report.config import outp_file_format
from report.utils import get_nonexistant_path, savefig_metadata
import matplotlib.pyplot as plt

def process_all(file_directory):
    processed_data = []
    for file in os.listdir(file_directory):
        file = os.path.join(file_directory, file)
        meta_data = pd.read_hdf(file, key="meta_data").loc[0]
        output_data = pd.read_hdf(file, key="approximation_comparison")
        plot_x(output_data, meta_data)
        plot_y(output_data, meta_data)

def plot_x(output_data, meta_data):

    fig, ax1 = plt.subplots()
    time_grid = output_data["time grid"]

    ax1.plot(time_grid, output_data["x bar approx"], label=r"Approximate solution", color='r')
    ax1.plot(time_grid, output_data["x bar mc"], label="Monte Carlo Simulation", color='b')

    upper = output_data["x bar mc"] + 3 * output_data["x std mc"]/np.sqrt(meta_data["number samples"])
    lower = output_data["x bar mc"] - 3 * output_data["x std mc"] / np.sqrt(meta_data["number samples"])

    ax1.fill_between(time_grid, lower, upper, label="3 std conf level", alpha=0.28)

    ax1.set_xlabel("time (years)")
    ax1.set_ylabel(r"$\mathbb{E}^A[x(t)]$")
    title = r"{}Y{}Y Swap: Comparison of approximate $\mathbb{{E}}^A[x(t)]$ with Monte-Carlo solution".format(meta_data["swap_T0"],
                                                                                                            meta_data["swap_TN"]-meta_data["swap_T0"])
    ax1.set_title(title)
    ax1.legend()

    file_name = r"x_approx_{}Y{}Y_swap_kappa_{}_lambda_{}_alpha_{}_beta_{}".format(meta_data["swap_T0"], meta_data["swap_TN"]-meta_data["swap_T0"],
                                                            meta_data["kappa"], meta_data["lambda value"],
                                                            meta_data["alpha"], meta_data["beta"])
    file_name += "." + outp_file_format
    output_file = os.path.join(output_plots_approx_solution, file_name)

    file_path = get_nonexistant_path(output_file)
    savefig_metadata(file_path, outp_file_format, fig, meta_data)

def plot_y(output_data, meta_data):

    fig, ax1 = plt.subplots()
    time_grid = output_data["time grid"]
    ax1.plot(time_grid, output_data["y bar approx"], label="Approximate Solution", color='r')
    ax1.plot(time_grid, output_data["y bar mc"], label="Monte Carlo Simulation", color='b')

    upper = output_data["y bar mc"] + 3 * output_data["y std mc"]/np.sqrt(meta_data["number samples"])
    lower = output_data["y bar mc"] - 3 * output_data["y std mc"]/np.sqrt(meta_data["number samples"])

    ax1.fill_between(time_grid, lower, upper, label="3 std conf level", alpha=0.28)

    ax1.set_xlabel("time (years)")
    ax1.set_ylabel(r"$\mathbb{E}^A[y(t)]$")

    title = "Comparison of approximate $\mathbb{E}^A[y(t)]$ with Monte-Carlo solution"
    ax1.set_title(title)
    ax1.legend()



def table_result(output_data, meta_data):

    output_table = pd.DataFrame()
    output_table["y"] = output_data["y mc"] - output_data["y"]
    output_table["x"] = output_data["x mc"] - output_data["x"]


    pass

if __name__ == "__main__":
    input_dir = os.path.join(output_data_raw, "approximation", "xy_approx", "2020_12_31")
    process_all(input_dir)
    #plt.show()
