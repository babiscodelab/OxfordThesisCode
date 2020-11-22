from quassigaussian.finitedifference.adi.run_adi import AdiRunner
from quassigaussian.volatility.local_volatility import LinearLocalVolatility
from quassigaussian.products.instruments import Bond
from quassigaussian.products.pricer import BondPricer
from quassigaussian.utils import extract_x0_result
from quassigaussian.finitedifference.mesher.grid_boundaries import calculate_x_boundaries2, calculate_y_boundaries
from quassigaussian.finitedifference.mesher.linear_mesher import Mesher2d
import numpy as np

from qgtests.utis import get_mock_yield_curve_from_file, get_mock_yield_curve_const

def test_adi():

    maturity = 10

    linear_local_volatility = LinearLocalVolatility.from_const(30, 0.1, 0.1, 0.1)
    linear_local_volatility = LinearLocalVolatility.from_const(30, 0.1, 0.1, 0)

    theta = 0.5
    kappa = 0.3
    initial_curve = get_mock_yield_curve_const(rate=0.04)

    t_min = 0
    t_max = maturity
    t_grid_size = 200
    x_grid_size = 201

    y_grid_size = 20

    x_min, x_max = calculate_x_boundaries2(t_max, linear_local_volatility, alpha=2.5)
    y_min, y_max = calculate_y_boundaries(t_max, kappa, linear_local_volatility, alpha=2.5)

    mesher = Mesher2d()
    mesher.create_mesher_2d(t_min, t_max, t_grid_size, x_min, x_max, x_grid_size, y_min, y_max, y_grid_size)

    adi_runner = AdiRunner(theta, kappa, initial_curve, linear_local_volatility, mesher)

    bond = Bond(maturity)
    bond_pricer = BondPricer(initial_curve, kappa)

    bond_t0 = adi_runner.run_adi(bond, bond_pricer)
    bond_xyt0 = extract_x0_result(bond_t0, adi_runner.mesher.xgrid, adi_runner.mesher.ygrid)

    actual = bond_pricer.price(bond, 0, 0, 0)

    np.testing.assert_approx_equal(bond_xyt0, actual)

#test_adi()