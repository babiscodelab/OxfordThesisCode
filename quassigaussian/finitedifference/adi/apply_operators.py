


def apply_L2(l2_matrix, v_full, xpos):

    """
    :param l2_matrix: expected dimension: JxJ
    :param v_full: expected dimension: I+2 x J+2
    :param xpos:
    :return:
    """

    return l2_matrix * v_full[xpos:, 1:-1].T


def apply_L1(l1_matrix, v_full, ypos):
    """

    :param l1_matrix: expected dimension: IxI
    :param v_full:
    :param ypos:
    :return:
    """


    return l1_matrix * v_full[:, ypos]