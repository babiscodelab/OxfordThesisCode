from quassigaussian.finitedifference.adi.run_adi import AdiRunner
from quassigaussian.parameters.volatility.local_volatility import LinearLocalVolatility
from quassigaussian.products.instruments import Bond
from quassigaussian.products.pricer import BondPricer
from quassigaussian.finitedifference.mesher.grid_boundaries import calculate_x_boundaries2, calculate_y_boundaries, calculate_x_boundaries3
from quassigaussian.finitedifference.mesher.linear_mesher import Mesher2d
import pandas as pd
import os
from qgtests.utis import get_mock_yield_curve_const
from report.directories import output_data_raw, date_timestamp
from report.utils import get_nonexistant_path

output_data_raw_finite_difference = os.path.join(output_data_raw, "finite_difference", "bond")



def adi_bond_report():

    output_path = os.path.join(output_data_raw_finite_difference, date_timestamp)

    curve_rate = 0.01
    maturity_grid = [30]
    kappa_grid = [0.03]
    theta = 1/2

    initial_curve = get_mock_yield_curve_const(rate=curve_rate)

    vola_parameters = [(i, curve_rate, j) for i in [0.05, 0.1, 0.2, 0.4] for j in [0.1, 0.3, 0.5, 0.7, 0.9]]
    vola_grid_df = pd.DataFrame(vola_parameters, columns=["lambda", "alpha", "beta"])

    finite_difference_parameter = [(100, 150, 20), (300, 400, 80)]

    #finite_difference_parameter = [(100, 150, 20)]

    finite_difference_grid_df = pd.DataFrame(finite_difference_parameter, columns=["t_grid_size", "x_grid_size", "y_grid_size"])

    output_path = get_nonexistant_path(output_path)

    vola_grid_df = vola_grid_df.loc[(vola_grid_df["lambda"]==0.4) & (vola_grid_df["beta"]==0.35)]

    for maturity in maturity_grid:
        for kappa in kappa_grid:
            bond = Bond(maturity)
            bond_pricer = BondPricer(initial_curve, kappa)
            for index, vola_grid_row in vola_grid_df.iterrows():
                loca_vola = LinearLocalVolatility.from_const(maturity, vola_grid_row["lambda"], vola_grid_row["alpha"], vola_grid_row["beta"])
                for index, finite_difference_grid_row in finite_difference_grid_df.iterrows():

                    x_grid_size = finite_difference_grid_row["x_grid_size"]
                    y_grid_size = finite_difference_grid_row["y_grid_size"]
                    t_grid_size = finite_difference_grid_row["t_grid_size"]

                    t_min = 0
                    t_max = maturity

                    x_min, x_max = calculate_x_boundaries2(t_max, loca_vola, alpha=3)
                    x_min, x_max = calculate_x_boundaries3(t_max, kappa, loca_vola, alpha=3)
                    y_min, y_max = calculate_y_boundaries(t_max, kappa, loca_vola, alpha=4)

                    mesher = Mesher2d()
                    mesher.create_mesher_2d(t_min, t_max, t_grid_size, x_min, x_max, x_grid_size, y_min, y_max,
                                            y_grid_size)

                    adi_runner = AdiRunner(theta, kappa, initial_curve, loca_vola, mesher)

                    bond_t0 = pd.DataFrame(adi_runner.run_adi(bond, bond_pricer))

                    output_file = os.path.join(output_path, "bond_price_fd.hdf")
                    file_path = get_nonexistant_path(output_file)

                    meta_data = {"x_grid_size": int(x_grid_size), "y_grid_size": int(y_grid_size), "maturity": maturity,
                                "t_grid_size": int(t_grid_size), "vola_lambda": vola_grid_row["lambda"],"vola_alpha": vola_grid_row["alpha"],
                                 "vola_beta": vola_grid_row["beta"], "curve_rate": curve_rate, "kappa": kappa}

                    meta_data = pd.DataFrame(meta_data, index=[0])
                    bond_t0.to_hdf(file_path, key="data", complevel=5)
                    meta_data.to_hdf(file_path, key="metadata", complevel=5)

                    pd.DataFrame(mesher.xmesh).to_hdf(file_path, key='xmesh', complevel=5)
                    pd.DataFrame(mesher.ymesh).to_hdf(file_path, key='ymesh', complevel=5)

                    pd.DataFrame(mesher.xgrid).to_hdf(file_path, key='xgrid', complevel=5)
                    pd.DataFrame(mesher.ygrid).to_hdf(file_path, key='ygrid', complevel=5)

if __name__ == "__main__":
    adi_bond_report()