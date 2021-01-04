
from quassigaussian.products.pricer import BondPricer, SwapPricer, SwaptionPricer, find_implied_black_vola
import pandas as pd
import os
import numpy as np
from qgtests.utis import get_mock_yield_curve_const
from report.directories import output_data_raw, date_timestamp
from report.utils import get_nonexistant_path


def read_results(file_directory):

    all_df = []
    for file in os.listdir(file_directory):
        df = pd.read_hdf(os.path.join(file_directory, file))



        all_df.append(df)
    return pd.concat(all_df)


def calculate_implied_vola(swaption_value):
    pass

file_dir = os.path.join(output_data_raw, "monte_carlo", "swaption", "2021_01_03")
all_df = read_results(file_dir)

print("paus")

