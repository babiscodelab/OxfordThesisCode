import pandas as pd
import os
import numpy as np
from report.directories import output_data_raw, output_tables_fd
from quassigaussian.products.instruments import Bond
from quassigaussian.products.pricer import BondPricer
from quassigaussian.curves.libor import LiborCurve
from quassigaussian.finitedifference.mesher.linear_mesher import extract_x0_result
import matplotlib.pyplot as plt
from report.config import outp_file_format
from report.directories import output_plots_fd
from report.utils import savefig_metadata


def process_all(file_directory):

    processed_data = []

    for file in os.listdir(file_directory):
        file = os.path.join(file_directory, file)
        fd_bond, analytics_bond, output_fd, bond_value, meta_data, xgrid, ygrid = process_bond_fd(file)
        error = (output_fd - bond_value)/bond_value
        error_average = np.mean(error.values)
        error_median = np.median(error.values)
        tmp = pd.DataFrame({"t grid": int(meta_data["t_grid_size"]), "x grid": int(meta_data["x_grid_size"]),
                           "y grid": int(meta_data["y_grid_size"]), "fd price": [fd_bond], "exact price": [analytics_bond],
                            "average error (bps)": [error_average*10000], "median error (bps)": [error_median*10000]})
        processed_data.append(tmp)

        x0_pos = np.where(xgrid == 0)[0][0]
        y0_pos = np.where(ygrid == 0)[0][0]

        xgrid_plot = (xgrid[0]>=-0.025) & (xgrid[0]<=0.025)
        plot_bond_price_for_different(xgrid[xgrid_plot], output_fd.loc[xgrid_plot, y0_pos], bond_value.loc[xgrid_plot, y0_pos], meta_data, "x")
        plot_bond_price_for_different(ygrid, output_fd.values[x0_pos, :], bond_value.values[x0_pos, :], meta_data, "u")

    processed_data = pd.concat(processed_data)
    processed_data.sort_values(by="t grid", inplace=True, ascending=True)

    output_tables_fd_file = os.path.join(output_tables_fd, "5Y_exact_bond_value_vs_fd.csv")
    processed_data = dataframe_format(processed_data)
    processed_data.to_csv(output_tables_fd_file, index=False)
    plt.show()
    return processed_data


def dataframe_format(df):
    return df.round({"fd price": 5, "exact price": 5, "average error (bps)": 2, "median error (bps)": 2})


def process_bond_fd(input_file):

    output_fd = pd.read_hdf(input_file, key="data")

    meta_data = pd.read_hdf(input_file, key="metadata").iloc[0]
    xmesh = pd.read_hdf(input_file, key="xmesh")
    ymesh = pd.read_hdf(input_file, key="ymesh")

    xgrid = pd.read_hdf(input_file, key="xgrid")
    ygrid = pd.read_hdf(input_file, key="ygrid")

    bond = Bond(meta_data["maturity"])
    curve = LiborCurve.from_constant_rate(meta_data["curve_rate"])
    bond_pricer = BondPricer(curve, meta_data["kappa"])
    bond_value = bond_pricer.price(bond, xmesh, ymesh, 0)

    #relative_diff = (output_fd - bond_value)/bond_value
    analytics_bond = extract_x0_result(bond_value.values, np.array(xgrid[0].values), ygrid[0].values)
    fd_bond = extract_x0_result(output_fd.values, np.array(xgrid[0].values), ygrid[0].values)

    return fd_bond, analytics_bond, output_fd, bond_value, meta_data, xgrid, ygrid



def plot_bond_price_for_different(grid, output_fd, bond_value, meta_data, grid_dir="x"):

    fig, ax1 = plt.subplots()
    ax1.plot(grid, bond_value, "b--", label="Exact formula")
    ax1.plot(grid, output_fd, "rx", label="Finite difference", markersize=2.5)

    error = (output_fd-bond_value)/bond_value * 10000
    ax1.set_xlabel(grid_dir + " value")
    ax1.set_ylabel("{}Y bond value".format(int(meta_data["maturity"])))

    ax2 = ax1.twinx()
    ax2.plot(grid, error, "kd", label="Error", markersize=3)

    ax2.set_ylabel("Error in bond value (bps)")

    if (grid_dir=="x"):
        lgnd = ax1.legend(loc="upper center")
    else:
        lgnd = ax1.legend(loc="lower center")

    lgnd.legendHandles[0]._legmarker.set_markersize(6)
    lgnd.legendHandles[1]._legmarker.set_markersize(6)


    ax2.legend(loc="upper right")
    title = "Exact bond formula vs Finite Difference solution with grid size t, x, u: {:d}, {:d}, {:d}".format(int(meta_data["t_grid_size"]),
                                                            int(meta_data["x_grid_size"]), int(meta_data["y_grid_size"]))
    ax1.set_title(title)
    file_name = "{}Y_exact_bond_value_vs_fd_{}_{}_{}_grid_{}".format(meta_data["maturity"], meta_data["t_grid_size"],
                                                            meta_data["x_grid_size"], meta_data["y_grid_size"], grid_dir) + "." + outp_file_format
    fig.savefig(os.path.join(output_plots_fd, file_name), format=outp_file_format)
    savefig_metadata(os.path.join(output_plots_fd, file_name), outp_file_format, fig, meta_data)
    return fig


#input_file = os.path.join(output_data_raw, "finite_difference", "2020_12_13", "bond_price_fd.hdf")
input_dir = os.path.join(output_data_raw, "finite_difference", "bond", "2021_02_13-13")
process_all(input_dir)