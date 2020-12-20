import numpy as np


class Mesher2d():

    def __init__(self):
        pass

    def create_mesher_2d(self, tmin, tmax, tsize, xmin, xmax, xsize, ymin, ymax, ysize):
        """
        Calculate the grid.
        xgrid[0], ygrid[0]: lower boundary
        xgrid[xmax], ygrid[ymax]: upper boundary
        size: Number of grid points including the two boundaries.
        """

        self.tmin = tmin
        self.tmax = tmax
        self.tsize = tsize

        self.tgrid = np.linspace(tmin, tmax, tsize)
        self.delta_t = self.tgrid[1] - self.tgrid[0]

        xgrid1 = np.linspace(xmin, 0, int(xsize/2))
        xgrid2 = np.linspace(0, xmax, int(xsize/2))
        xgrid1 = np.delete(xgrid1, -1)
        self.xgrid = np.concatenate([xgrid1, xgrid2])


        self.xdim = len(self.xgrid)

        self.ygrid = np.linspace(ymin, ymax, ysize)


        self.xmesh, self.ymesh = np.meshgrid(self.xgrid, self.ygrid, indexing='ij')

        self.delta_px, self.delta_mx = calculate_delta(self.xgrid)
        self.delta_py, self.delta_my = calculate_delta(self.ygrid)
        self.ydim = len(self.ygrid)


def calculate_delta(gridv):
    gridv_length = len(gridv)
    delta_m = np.zeros(gridv_length)
    delta_p = np.zeros(gridv_length)

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


def extract_x0_result(res, x_grid, y_grid):

    x0_pos = np.where(x_grid == 0)[0][0]
    y0_pos = np.where(y_grid == 0)[0][0]

    return res.values[x0_pos, y0_pos]


if __name__ == "__main__":

    tmp = np.array([0, 2, 3, 7, 11, 22, 44, 45])
    calculate_delta(tmp)


