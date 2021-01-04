import numpy as np
from qgtests.utis import get_mock_yield_curve_const
# Simulate paths and calculate bond price
from quassigaussian.montecarlo.simulations import ProcessSimulator
from quassigaussian.volatility.local_volatility import LinearLocalVolatility, BlackVolatilityModel
from quassigaussian.products.pricer import AnnuityPricer, BondPricer, SwapPricer, SwaptionPricer
from quassigaussian.products.instruments import Annuity, Swap, Swaption
from scipy.interpolate import interp1d
from quassigaussian.montecarlo.monte_carlo_pricer import monte_carlo_pricer_annuity

def test_mc_rn_measure():

    kappa = 0.3
    rate = 0.04
    initial_curve = get_mock_yield_curve_const(rate=rate)

    linear_local = LinearLocalVolatility.from_const(30, 0.1, 0.1, 0.1)

    number_samples = 400
    number_steps = 100

    t_horizon = 5
    dt = t_horizon / number_steps
    x_simulator = ProcessSimulator(number_samples, number_steps, dt)
    local_volatility = linear_local

    res = x_simulator.simulate_xy(kappa=kappa, local_volatility=local_volatility)

    interest_rate = res.x + initial_curve.get_inst_forward(res.time_grid)

    bond_mc = np.exp(-np.trapz(interest_rate, res.time_grid))

    avg_bond = np.mean(bond_mc)
    std_mc = np.std(bond_mc)/np.sqrt(number_samples)

    upper_bound = avg_bond + 3*std_mc
    lower_bound = avg_bond - 3*std_mc

    assert np.exp(-rate*t_horizon)>lower_bound and np.exp(-rate*t_horizon)<upper_bound

def test_mc_annuity_measure():

    kappa = 0.001
    rate = 0.06
    initial_curve = get_mock_yield_curve_const(rate=rate)

    linear_local = LinearLocalVolatility.from_const(30, 0.1, 0.1, 0.1)

    number_samples = 200
    number_steps = 50


    local_volatility = linear_local

    swap_T0 = 4
    swap_TN = 5
    frequency = 0.5

    t_horizon = swap_T0
    dt = t_horizon / number_steps

    bond_pricer = BondPricer(initial_curve, kappa)
    annuity_pricer = AnnuityPricer(bond_pricer)
    swap = Swap(swap_T0, swap_TN, frequency)
    annuity = Annuity(swap.bond_list, frequency)

    x_simulator = ProcessSimulator(number_samples, number_steps, dt, annuity_pricer=annuity_pricer)

    res = x_simulator.simulate_xy(kappa=kappa, local_volatility=local_volatility, annuity_measure=annuity)

    mc_price = annuity_pricer.annuity_price(0, 0, 0, annuity) * 1/annuity_pricer.annuity_price(t_horizon, res.x[:, -1], res.y[:, -1], annuity)

    avg_bond = np.mean(mc_price)
    std_mc = np.std(mc_price)/np.sqrt(number_samples)
    actual_price = np.exp(-rate*t_horizon)

    assert actual_price > avg_bond -3 * std_mc and actual_price < avg_bond + 3 * std_mc

def test_mc_swaption():

    swap_T0 = 1
    swap_TN = 2
    coupon = 0.062
    frequency = 0.5

    swap = Swap(swap_T0, swap_TN, frequency)

    kappa = 0.3
    rate = 0.06
    initial_curve = get_mock_yield_curve_const(rate=rate)

    x = np.arange(0, 31)
    y = np.ones(31)*0.1

    lambda_t = interp1d(x, y, kind='previous')
    alpha_t = interp1d(x, y, kind='previous')
    b_t = interp1d(x, y*0, kind='previous')

    linear_local = LinearLocalVolatility(lambda_t, alpha_t, b_t)

    #linear_vola = local_volatility
    #linear_local = LinearLocalVolatility.from_const(30, 0.1, 0.1, 0.1)

    number_samples = 100
    number_steps = 10
    swap_pricer = SwapPricer(initial_curve, kappa=kappa)

    t_horizon = swap_T0
    dt = t_horizon / number_steps
    x_simulator = ProcessSimulator(number_samples, number_steps, dt)
    local_volatility = linear_local

    res = x_simulator.simulate_xy(kappa=kappa, local_volatility=local_volatility)

    interest_rate = res.x + initial_curve.get_inst_forward(res.time_grid)

    swaption = Swaption(swap_T0, coupon, swap)
    swaption_pricer = SwaptionPricer(swap_pricer)

    swaption_fair = swaption_pricer.maturity_price(swaption, res.x[:, -1], res.y[:, -1])
    swaption_mc = np.exp(-np.trapz(interest_rate, res.time_grid)) * swaption_fair

    exp_swaption_mc = np.mean(swaption_mc)
    upper_value = exp_swaption_mc + 3*np.std(swaption_mc)/np.sqrt(number_samples)
    lower_value = exp_swaption_mc - 3*np.std(swaption_mc)/np.sqrt(number_samples)

    print("Swaption mean: {:.4f}, lower: {:.4f}, upper: {:.4f}".format(exp_swaption_mc, lower_value, upper_value))


def test_mc_swaption_annuity():
    swap_T0 = 1
    swap_TN = 2
    coupon = 0.062
    frequency = 0.5

    swap = Swap(swap_T0, swap_TN, frequency)
    random_number_generator_type = "sobol"
    random_number_generator_type = "normal"

    kappa = 0.3
    rate = 0.06
    initial_curve = get_mock_yield_curve_const(rate=rate)

    x = np.arange(0, 31)
    y = np.ones(31) * 0.1

    lambda_t = interp1d(x, y, kind='previous')
    alpha_t = interp1d(x, y, kind='previous')
    b_t = interp1d(x, y * 0, kind='previous')

    linear_local = LinearLocalVolatility(lambda_t, alpha_t, b_t)

    # linear_vola = local_volatility
    # linear_local = LinearLocalVolatility.from_const(30, 0.1, 0.1, 0.1)

    number_samples = np.power(2, 12)
    number_steps = np.power(2, 7)
    swap_pricer = SwapPricer(initial_curve, kappa=kappa)

    t_horizon = swap_T0
    dt = t_horizon / number_steps
    annuity_pricer = AnnuityPricer(swap_pricer.bond_pricer)
    x_simulator = ProcessSimulator(number_samples, number_steps, dt, random_number_generator_type, annuity_pricer)

    local_volatility = linear_local
    annuity = Annuity(swap.bond_list, frequency)

    res = x_simulator.simulate_xy(kappa=kappa, local_volatility=local_volatility, annuity_measure=annuity)
    swaption_pricer = SwaptionPricer(swap_pricer)
    swaption = Swaption(swap_T0, coupon, swap)

    res = monte_carlo_pricer_annuity(res, swaption, swaption_pricer)

    exp_swaption_mc = np.mean(res)
    upper_value = exp_swaption_mc + 3*np.std(res)/np.sqrt(number_samples)
    lower_value = exp_swaption_mc - 3*np.std(res)/np.sqrt(number_samples)

    fd_price = 0.002313
    print("Swaption mean: {:.6f}, lower: {:.6f}, upper: {:.6f}".format(exp_swaption_mc, lower_value, upper_value))
