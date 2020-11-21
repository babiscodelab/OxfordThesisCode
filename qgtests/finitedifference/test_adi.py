from quassigaussian.finitedifference.adi.run_adi import AdiRunner
from quassigaussian.volatility.local_volatility import LinearLocalVolatility
from quassigaussian.products.instruments import Bond
from quassigaussian.products.pricer import BondPricer
from quassigaussian.utils import extract_x0_result

from qgtests.utis import get_mock_yield_curve

def test_adi():

    maturity = 10

    linear_local_volatility = LinearLocalVolatility.from_const(30, 0.1, 0.1, 0.1)
    linear_local_volatility = LinearLocalVolatility.from_const(30, 0.1, 0.1, 0)

    theta = 0.5
    kappa = 0.3
    initial_curve = get_mock_yield_curve()

    t_min = 0
    t_max = maturity
    t_grid_size = 100
    x_grid_size = 400
    y_grid_size = 20

    adi_runner = AdiRunner(theta, kappa, initial_curve, linear_local_volatility, t_min, t_max, t_grid_size, x_grid_size, y_grid_size)

    bond = Bond(maturity)
    bond_pricer = BondPricer(initial_curve, kappa)

    bond_t0 = adi_runner.run_adi(bond, bond_pricer)
    bond_xyt0 = extract_x0_result(bond_t0, adi_runner.mesher.xgrid, adi_runner.mesher.ygrid)

    actual = bond_pricer.price(bond, 0, 0, 0)

    return bond_xyt0

test_adi()