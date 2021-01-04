import numpy as np

def apply_control_variate(x, y, mc_simulation_product, bond, bond_pricer):

    mc_simulation_bond = bond_pricer.price(bond, x, y, 0)

    beta = calculate_beta(mc_simulation_bond, mc_simulation_product)
    bond_value = bond_pricer.price(bond, 0, 0, 0)

    res = mc_simulation_product - beta*(mc_simulation_bond - bond_value)

    return res


def calculate_beta(mc_simulation_bond, mc_simulation_product):

    sigma_x = np.var(mc_simulation_bond)
    correlation = np.cov(mc_simulation_product, mc_simulation_bond)
    beta = correlation/sigma_x

    return beta[0, 1]