import numpy as np

def apply_control_variate(last_time, x, y, mc_simulation_product, bond, bond_pricer, initial_curve):

    mc_simulation_bond = initial_curve.get_discount(last_time)*bond_pricer.price(bond, x, y, last_time)

    beta = calculate_beta(mc_simulation_bond, mc_simulation_product)
    bond_value = bond_pricer.price(bond, 0, 0, 0)

    res = mc_simulation_product - beta*(mc_simulation_bond - bond_value)

    return res


def apply_control_variate_annuity(last_time, x, y, mc_simulation_product, annuity, annuity_pricer, initial_curve):

    mc_simulation_bond = initial_curve.get_discount(last_time)*annuity_pricer.annuity_price(last_time, x, y, annuity)

    beta = calculate_beta(mc_simulation_bond, mc_simulation_product)
    bond_value = annuity_pricer.annuity_price(0, 0, 0, annuity)

    res = mc_simulation_product - beta*(mc_simulation_bond - bond_value)

    return res

def calculate_beta(mc_simulation_bond, mc_simulation_product):

    sigma_x = np.var(mc_simulation_bond)
    correlation = np.cov(mc_simulation_product, mc_simulation_bond)
    beta = correlation/sigma_x

    return beta[0, 1]