
from report.data_processing.plotting.finite_difference_swaption import process_all as process_fd
from report.utils import savefig_metadata, get_nonexistant_path, read_results_walk, read_results
import pandas as pd
import os
import matplotlib.ticker as mtick
from report.utils import open_with_excel
import matplotlib.pyplot as plt
from report.directories import output_data_raw, output_plots_swaption, date_timestamp, output_plots_approx_solution
from report.config import outp_file_format

join_c = ["expiry", "maturity", "atm strike", "moneyness", "vola_lambda", "vola_alpha", "kappa", "curve_rate",
          "vola_beta", "strike"]

def plot_approximated_implied_vola(df):

    for gr, df_gr in df.groupby(["expiry", "maturity", "vola_lambda", "vola_beta"]):
        df_gr.sort_values("moneyness", inplace=True)
        fig = plt.figure()
        plt.plot(df_gr["moneyness"], df_gr["implied_black_vola_pit"], "r-x", label="Approximate solution (Method A)")
        plt.plot(df_gr["moneyness"], df_gr["implied_black_vola_rg"], "g-x", label="ODE system approx (Method B)")

        plt.plot(df_gr["moneyness"], df_gr["implied_black_vola"], "b-x", label="Finite Difference")
        plt.legend()
        plt.xlabel(r"Strike - ATM \%")
        plt.ylabel("Implied Black volatility")
        title_str = "{}Y{}Y swaption implied volatility".format(gr[0], gr[1]-gr[0])
        plt.title(title_str)

        fname_output = "{}Y{}Y_lambda_{}_beta_{}.pdf".format(*gr)
        outp_file = os.path.join(output_plots_approx_solution, "swaption", fname_output)
        fig.savefig(outp_file)
        #plt.savefig("tmp_test.pdf")
        #break
        #plt.savefig()

if __name__ == "__main__":

    approximated_file_path = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\report\data\output_data\raw\approximation\piterbarg_swaption_approx\2021_02_15\result"
    #approximated_file_path = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\report\data\output_data\raw\finite_difference\swaption\2021_02_14-12"
    df_approx = read_results(approximated_file_path)
    df_approx["implied_black_vola"] = df_approx["implied_black_vola"].apply(lambda row: row[0])

    fd_file_path = r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\report\data\output_data\raw\finite_difference\swaption\2021_02_14-12"
    df_finite_difference = read_results_walk(fd_file_path, key="metadata")

    df_finite_difference2 = read_results(
        r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\report\data\output_data\raw\finite_difference\swaption\2021_02_14-13",
        key="metadata").sort_values(["maturity", "moneyness", "vola_lambda", "vola_beta", "x_grid_size"])
    df_finite_difference = pd.concat([df_finite_difference, df_finite_difference2])


    df_finite_difference = read_results(
        r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\report\data\output_data\raw\finite_difference\swaption\2021_02_15",
        key="metadata").sort_values(["maturity", "moneyness", "vola_lambda", "vola_beta", "x_grid_size"])

    df_finite_difference = read_results(
        r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\report\data\output_data\raw\finite_difference\swaption\2021_02_15-1",
        key="metadata").sort_values(["maturity", "moneyness", "vola_lambda", "vola_beta", "x_grid_size"])

    df_finite_difference = read_results(
        r"C:\Users\d80084\Google Drive\01oxford\7 Thesis\code\quasigaussian\report\data\output_data\raw\finite_difference\swaption\2021_02_15-3",
        key="metadata").sort_values(["maturity", "moneyness", "vola_lambda", "vola_beta", "x_grid_size"])

    df_finite_difference = df_finite_difference.loc[df_finite_difference["x_grid_size"]==200]
    df_approx.reset_index(drop=True, inplace=True)

    rg_class = "rungekutta"
    rg_df = df_approx.loc[df_approx["xy_approximation"]==rg_class]
    non_rg_df = df_approx.loc[df_approx["xy_approximation"]!=rg_class]

    approx_df = pd.merge(rg_df, non_rg_df, on=join_c, suffixes=("_rg", "_pit"))
    all_df = pd.merge(approx_df, df_finite_difference, on=join_c, suffixes=("_approx", "_fd"))
    all_df = all_df.loc[all_df["x_grid_size"]==200]
    plot_approximated_implied_vola(all_df)
    plt.show()