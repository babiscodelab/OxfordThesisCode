import os
import pandas as pd
from report.directories import output_data_raw

def read_results(file_directory):

    all_df = []
    for file in os.listdir(file_directory):
        df = pd.read_hdf(os.path.join(file_directory, file))
        all_df.append(df)
    return pd.concat(all_df)

file_dir = os.path.join(output_data_raw, "monte_carlo", "swaption", "2020_12_23")
all_df = read_results(file_dir)

print("paus")

