
from quassigaussian.products.pricer import BondPricer, SwapPricer, SwaptionPricer, find_implied_black_vola
import pandas as pd
import os
import numpy as np
from qgtests.utis import get_mock_yield_curve_const
from report.directories import output_data_raw, date_timestamp, output_tables_mc
from report.utils import get_nonexistant_path
from report.utils import open_with_excel

joint_c = ["expiry", "maturity", "moneyness", "vola_lambda", "vola_beta", "strike", "atm strike", "curve_rate", "kappa", "vola_alpha"]
to_keep = ["expiry", "maturity", "moneyness", "swaption value FD",  "swaption value MC cv", "swaption value MC",
            "swaption value error MC cv", "swaption value error MC",
            "implied_vola MC cv", "implied_vola MC", "implied_vola FD"]

to_keep1 = ["expiry", "maturity", "moneyness", "vola_lambda", "vola_beta", "strike",
            "swaption value FD",  "swaption value MC cv", "swaption value MC",
            "swaption value error MC cv", "swaption value error MC",
            "implied_vola MC cv", "implied_vola MC", "implied_vola FD"
            ]
def format_mc_swaption_table(df_mc, df_finite_dif):

    df_mc["moneyness"] = df_mc["moneyness"].round(decimals=5)
    df_finite_dif["moneyness"] = df_finite_dif["moneyness"].round(decimals=5)
    df_finite_dif["strike"] = df_finite_dif["strike"].round(decimals=5)
    df_mc["strike"] = df_mc["strike"].round(decimals=5)
    df_mc["atm strike"] = df_mc["atm strike"].round(decimals=5)

    df_finite_dif["atm strike"] = df_finite_dif["atm strike"].round(decimals=5)

    df = pd.merge(df_mc, df_finite_dif, on=joint_c)



    df = df[to_keep1]
    swaption_value_c = ["swaption value FD", "swaption value MC", "swaption value error MC",
                        "swaption value MC cv", "swaption value error MC cv"]

    df[swaption_value_c] = df[swaption_value_c]*100



    df_tmp = df.loc[(df["moneyness"].isin([0, 0.015, -0.0150])) & (df["vola_lambda"].isin([0.4])) &
                    (df["vola_beta"].isin([0.3]))][to_keep].reset_index(drop=True).sort_values(["expiry", "maturity", "moneyness"])

    float_to_str3 = swaption_value_c
    float_to_str4 = ["implied_vola MC", "implied_vola MC cv", "implied_vola FD"]

    df_tmp[float_to_str3] = df_tmp[float_to_str3].applymap(lambda x: '{0:.3f}'.format(x))
    df_tmp[float_to_str4] = df_tmp[float_to_str4].applymap(lambda x: '{0:.4f}'.format(x))

    df_tmp["Swaption"] = df_tmp["expiry"].astype(str) + "Y" + (df_tmp["maturity"]-df_tmp["expiry"]).astype(str) + "Y"
    df_tmp["FD value"] = df_tmp["swaption value FD"].astype(str)
    df_tmp["MC cv value"] = df_tmp["swaption value MC cv"] + " (" + df_tmp["swaption value error MC cv"] +  ")"
    df_tmp["MC value"] = df_tmp["swaption value MC"] + " (" + df_tmp["swaption value error MC"] +  ")"

    map_vola = {"implied_vola MC cv": "MC cv impl vol", "implied_vola MC": "MC impl vol", "implied_vola FD": "FD impl vol",
                "moneyness": "Strike-ATM"
                }

    df_tmp.rename(map_vola, axis=1, inplace=True)
    return_c = ["Swaption", "Strike-ATM", "FD value", "MC cv value", "MC value", "FD impl vol", "MC cv impl vol", "MC impl vol"]

    return df_tmp[return_c]

def read_results(file_directory, key=None):

    all_df = []
    for file in os.listdir(file_directory):
        df = pd.read_hdf(os.path.join(file_directory, file), key=key)
        all_df.append(df)
    return pd.concat(all_df)



def process_fd_dataframe(df):
    columns_map = {"swaption_value": "swaption value FD", "implied_black_vola": "implied_vola FD"}
    return df.rename(columns_map, axis=1)

def process_mc_dataframe(df):
    columns_map = {"swaption value": "swaption value MC", "swaption value error": "swaption value error MC",
                   "swaption value cv": "swaption value MC cv", "swaption value error cv": "swaption value error MC cv",
                   "implied_vola": "implied_vola MC", "implied_vola_cv": "implied_vola MC cv"}

    df["moneyness"] = df["strike"] - df["atm strike"]
    return df.rename(columns_map, axis=1)

mc_swaption_dir = os.path.join(output_data_raw, "monte_carlo", "swaption", "2021_01_03")

#mc_swaption_dir = os.path.join(output_data_raw, "monte_carlo", "swaption", "2021_01_27-compare-error")
mc_swaption = read_results(mc_swaption_dir)

fd_swaption_dir = os.path.join(output_data_raw, "finite_difference", "swaption", "2021_01_26")
fd_swaption = read_results(fd_swaption_dir, key="metadata")

fd_swaption = process_fd_dataframe(fd_swaption)
mc_swaption = process_mc_dataframe(mc_swaption)

swaption_cv_table = format_mc_swaption_table(mc_swaption, fd_swaption)
#swaption_cv_table.to_csv(os.path.join(output_tables_mc, "control_variate.csv"), index=False)
print("paus")

