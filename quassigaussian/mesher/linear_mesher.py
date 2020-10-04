import numpy as np


def create_mesher_2d(xmin, xmax, xsize, ymin, ymax, ysize):
    """
    Calculate the grid.
    xgrid[0], ygrid[0]: lower boundary
    xgrid[xmax], ygrid[ymax]: upper boundary
    size: Number of grid points including the two boundaries.
    """
    xgrid = np.linspace(xmin, xmax, xsize)
    ygrid = np.linspace(ymin, ymax, ysize)

    return xgrid, ygrid



def calculate_delta(gridv):
    gridv_length = len(gridv)
    delta_m = np.zeros(shape=(gridv_length,1))
    delta_p = np.zeros(shape=(gridv_length,1))

    for i in range(gridv_length):
        if i==0:
            delta_m[i] = np.NaN
        else:
            delta_m[i] = gridv[i] - gridv[i - 1]
        if i==gridv_length-1:
            delta_p[i] = np.NaN
        else:
            delta_p[i] = gridv[i+1] - gridv[i]

    return delta_p, delta_m


def calculate_pass():
    pass


if __name__ == "__main__":

    tmp = np.array([0, 2, 3, 7, 11, 22, 44, 45])
    calculate_delta(tmp)