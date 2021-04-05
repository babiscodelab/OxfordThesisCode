import numpy as np
from scipy.linalg import solve_banded

def solve_banded_array(sparse_diag, v):
    """
    solves ax = b using Thomas Algorithm
    :param a:
    :param b:
    :return:
    """

    Ab = np.zeros((3, sparse_diag.shape[0]))
    Ab[0, 1:] = sparse_diag.diagonal(1)
    Ab[1, :] = sparse_diag.diagonal(0)
    Ab[2, :-1] = sparse_diag.diagonal(-1)

    return solve_banded((1, 1), Ab, v)
