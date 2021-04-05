from quassigaussian.products.pricer import BondPricer, SwapPricer, SwaptionPricer, find_implied_black_vola
import pandas as pd
import os
import numpy as np
from qgtests.utis import get_mock_yield_curve_const
from report.directories import output_data_raw, date_timestamp, output_tables_mc, output_plots_swaption
from report.utils import get_nonexistant_path
from report.utils import open_with_excel
import pandas as pd
import os
import numpy as np
from report.directories import output_data_raw, output_tables_fd

import matplotlib.pyplot as plt

def plot_smile_beta(df):
    for gr, df_gr in df.groupby(["vola_lambda"]):
        plt.figure()
        str_gr = ""
        count = 0
        map_str = {0: "r-x", 1: "b-x"}
        for gr2, df_gr2 in df_gr.groupby(["vola_beta"]):
            df_gr2 = df_gr2.sort_values("moneyness")
            plt.plot(df_gr2["moneyness"], df_gr2["implied_black_vola"], map_str[count], label=r"$\beta_r(t)$=" + str(gr2))
            plt.title("Volatility smile for different skew parameters")
            plt.xlabel(r"Strike-ATM %")
            plt.ylabel("Implied Black volatility")
            plt.legend()
            str_gr = str_gr + "_" + str(gr2)
            count += 1

        plt.savefig(os.path.join(output_plots_swaption, "swaption_beta" + str(str_gr) + "_lambda_" + str(gr) + ".pdf"))

def plot_smile_lambda(df):
    for gr, df_gr in df.groupby(["vola_beta"]):
        plt.figure()
        str_gr = ""
        count = 0
        map_str = {0: "r-x", 1: "b-x"}
        for gr2, df_gr2 in df_gr.groupby(["vola_lambda"]):
            df_gr2 = df_gr2.sort_values("moneyness")
            plt.plot(df_gr2["moneyness"], df_gr2["implied_black_vola"], map_str[count], label=r"$\lambda_r(t)$=" + str(gr2))
            plt.title("Volatility smile for different vola parameters")
            plt.xlabel(r"Strike-ATM %")
            plt.ylabel("Implied Black volatility")
            plt.legend()
            str_gr = str_gr + "_" + str(gr2)
            count += 1

        plt.savefig(os.path.join(output_plots_swaption, "swaption_lambda" + str(str_gr) + "_beta_" + str(gr) + ".pdf"))



def read_results(file_directory, key=None):

    all_df = []
    for file in os.listdir(file_directory):
        df = pd.read_hdf(os.path.join(file_directory, file), key=key)
        all_df.append(df)
    return pd.concat(all_df)

fd_swaption_dir = os.path.join(output_data_raw, "finite_difference", "swaption", "2021_01_08")
fd_swaption = read_results(fd_swaption_dir, key="metadata")


fd_swaption = fd_swaption.loc[(fd_swaption["expiry"]==5) & (fd_swaption["vola_beta"].isin([0.1])) &
                              (fd_swaption["vola_lambda"].isin([0.1, 0.25]))]
plot_smile_lambda(fd_swaption)

plt.show()



