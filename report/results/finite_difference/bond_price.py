from quassigaussian.finitedifference.adi.run_adi import AdiRunner
from quassigaussian.volatility.local_volatility import LinearLocalVolatility, BlackVolatilityModel
from quassigaussian.products.instruments import Bond, Swap, Swaption
from quassigaussian.products.pricer import BondPricer, SwapPricer, SwaptionPricer, find_implied_black_vola
from quassigaussian.finitedifference.mesher.grid_boundaries import calculate_x_boundaries2, calculate_y_boundaries
from quassigaussian.finitedifference.mesher.linear_mesher import Mesher2d
import pandas as pd
import os
from qgtests.utis import get_mock_yield_curve_const
from report.directories import output_data_raw, date_timestamp
from report.utils import get_nonexistant_path

output_data_raw_finite_difference = os.path.join(output_data_raw, "finite_difference", "bond")



def adi_bond_report():

    output_path = os.path.join(output_data_raw_finite_difference, date_timestamp)

    curve_rate = 0.02
    maturity_grid = [5]
    kappa_grid = [0.03]
    theta = 1/2

    lambda_grid = [0.05]
    alpha_grid = [0.5]
    beta_grid = [0.5]
    initial_curve = get_mock_yield_curve_const(rate=curve_rate)

    vola_grid_df = pd.DataFrame({"lambda": lambda_grid, "alpha": alpha_grid, "beta": beta_grid})

    t_grid_size_grid = [6, 12, 18, 48, 64]
    x_grid_size_grid = [50, 100, 200, 400, 800]
    y_grid_size_grid = [10, 20, 40, 60, 80]

    finite_difference_grid_df = pd.DataFrame({"t_grid_size": t_grid_size_grid, "y_grid_size": y_grid_size_grid,
                                           "x_grid_size": x_grid_size_grid})
    output_path = get_nonexistant_path(output_path)

    for maturity in maturity_grid:
        for kappa in kappa_grid:
            bond = Bond(maturity)
            bond_pricer = BondPricer(initial_curve, kappa)
            for index, vola_grid_row in vola_grid_df.iterrows():
                loca_vola = LinearLocalVolatility.from_const(maturity, vola_grid_row["lambda"], vola_grid_row["alpha"], vola_grid_row["beta"])
                for index, finite_difference_grid_row in finite_difference_grid_df.iterrows():

                    x_grid_size = finite_difference_grid_row["x_grid_size"]
                    y_grid_size = finite_difference_grid_row["y_grid_size"]
                    t_grid_size = finite_difference_grid_row["t_grid_size"]*maturity

                    t_min = 0
                    t_max = maturity

                    x_min, x_max = calculate_x_boundaries2(t_max, loca_vola, alpha=2.5)
                    y_min, y_max = calculate_y_boundaries(t_max, kappa, loca_vola, alpha=2.5)

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