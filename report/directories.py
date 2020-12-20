import os
from datetime import datetime

data_path = os.path.join(os.path.realpath(__file__), "..", "data")
output_data = os.path.join(data_path, "output_data")

output_data_raw = os.path.join(output_data, "raw")
output_plots = os.path.join(output_data, "plots")
output_tables = os.path.join(output_data, "tables")

output_tables_fd = os.path.join(output_tables, "finite_difference")
output_plots_fd = os.path.join(output_plots, "finite_difference")

report_path = os.path.join(os.path.realpath(__file__), "..")


date_timestamp = datetime.today().strftime('%Y_%m_%d')  # class variable so that it gets set only one time
