import pandas as pd
import os
from report.directories import output_data_raw


def process_all(file_directory):

    output_data_ls = []
    for file in os.listdir(file_directory):
        file = os.path.join(file_directory, file)
        output_data_ls.append(process_swaption_fd(file))
    output_data = pd.concat(output_data_ls)
    output_data = output_data.sort_values(by=["strike", "t_grid_size"])
    return output_data

def dataframe_format(df):
    return df.round({"fd price": 5, "exact price": 5, "average error (bps)": 2, "median error (bps)": 2})


def process_swaption_fd(input_file):

    output_fd = pd.read_hdf(input_file, key="data")

    meta_data = pd.read_hdf(input_file, key="metadata")
    return meta_data


input_dir = os.path.join(output_data_raw, "finite_difference", "swaption", "2020_12_21")

process_all(input_dir)