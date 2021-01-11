from report.data_processing.plotting.finite_difference_swaption import process_all as process_fd
from report.data_processing.plotting.monte_carlo_swaption import read_results as process_mc
from report.utils import savefig_metadata, get_nonexistant_path
import pandas as pd
import os
import matplotlib.ticker as mtick

import matplotlib.pyplot as plt
from report.directories import output_data_raw, output_plots_swaption, date_timestamp
from report.config import outp_file_format

def load_swaption_data(path_fast_approx, path_mc, path_fd):

    join_on = ["expiry", "maturity", "strike", "moneyness", "vola_lambda",
                            "vola_alpha", "vola_beta", "curve_rate", "kappa", "atm strike"]
    fast_approx = pd.read_hdf(path_fast_approx)
    mc_results = process_mc(path_mc)
    fd_results = process_fd(path_fd)
    all_results = pd.merge(mc_results, mc_results, on=join_on, suffixes=("_mc", "_fd"))

    all_results = pd.merge(fast_approx, fd_results, on=join_on)
    return all_results

def plot_implied_vola(results, outp_path):

    meta_colums = ["number_paths", "number_time_steps", "random_number_generator_type", "expiry", "maturity", "kappa",
                   "x_grid_size", "y_grid_size", "t_grid_size", "curve_rate",  "vola_lambda", "vola_alpha", "vola_beta"]
    meta_data = dict(results.loc[0, meta_colums])
    results["moneyness"] = results["moneyness"]*100
    fig1, ax1 = plt.subplots()
    results = results.sort_values(by="strike")
    ax1.plot(results["moneyness"], results["implied_black_vola"], label="Finite Difference", color="r")
    ax1.plot(results["moneyness"], results["implied_vola_cv"], label="Monte Carlo", color="b")
    ax1.fill_between(results["moneyness"], results["implied_vola_cv_min"], results["implied_vola_cv_max"], label="3 std conf level", alpha=0.28)

    ax1.set_xlabel(r"Moneyness (Strike-ATM)\%")
    ax1.set_ylabel("Implied Volatility")
    ax1.set_title(r"{}Y{}Y Swaption Implied Volatility".format(meta_data["expiry"],
                                                       meta_data["maturity"]-meta_data["expiry"]))
    ax1.legend(loc="upper center")
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter())

    fig2, ax2 = plt.subplots()
    diff = (results["implied_black_vola"] - results["implied_vola_cv"])*10000
    ax2.plot(results["moneyness"], diff, label="Implied Volatility Difference", color="r")
    ax2.set_title(r"{}Y{}Y Finite Difference-Monte Carlo Implied Volatility difference".format(meta_data["expiry"],
                                                       meta_data["maturity"]-meta_data["expiry"]))
    ax2.set_xlabel(r"Moneyness (Strike-ATM)\%")
    ax2.set_ylabel("Difference (bps)")
    ax2.legend()
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter())


    file_name = r"{}Y{}Y_swaption_kappa_{}_lambda_{:.3f}_alpha_{:.3f}_beta_{:.3f}".format(
                                                        meta_data["expiry"], meta_data["maturity"]-meta_data["expiry"],
                                                        meta_data["kappa"], meta_data["vola_lambda"],
                                                        meta_data["vola_alpha"], meta_data["vola_beta"]) + "." + outp_file_format
    file_name_diff = "diff_" + file_name

    file_implied_vola = os.path.join(outp_path, file_name)
    file_diff = os.path.join(outp_path, file_name_diff)

    savefig_metadata(file_implied_vola, outp_file_format, fig1, meta_data)
    savefig_metadata(file_diff, outp_file_format, fig2, meta_data)


    return

def plot_all(all_results, output_path):

    for swap, group in all_results.groupby(["expiry", "maturity"]):
        plot_implied_vola(group, output_path)
        pass


fd_file = os.path.join(output_data_raw, "finite_difference", "swaption", "2021_01_08")
mc_file = os.path.join(output_data_raw, "monte_carlo", "swaption", "2021_01_07")
approx_file = os.path.join(output_data_raw, "approximation", "piterbarg_swaption_approx", "2021_01_11", "result", 'swaption_approximation-2.hdf')

output_swaption = os.path.join(output_plots_swaption)

all_results = load_swaption_data(approx_file, mc_file, fd_file)

output_path = os.path.join(output_swaption, date_timestamp)
output_path = get_nonexistant_path(output_path)

try:
    os.mkdir(output_path)
except:
    pass
plot_all(all_results, output_path)