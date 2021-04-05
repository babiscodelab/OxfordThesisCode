import numpy as np


class Mesher2d():

    def __init__(self):
        pass

    def create_mesher_2d(self, tmin, tmax, tsize, xmin, xmax, xsize, umin, umax, usize):
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

        step_x = (xmax - xmin)/xsize
        size_m = int(np.abs(xmin)/(np.abs(xmin) + np.abs(xmax)) * xsize)
        size_p = xsize - size_m

        xgrid1 = np.linspace(-step_x*size_m, 0, size_m+1)
        xgrid2 = np.linspace(0, step_x*size_p, size_p+1)
        xgrid1 = np.delete(xgrid1, -1)
        self.xgrid = np.concatenate([xgrid1, xgrid2])

        ugrid1 = np.linspace(umin, 0, int(usize / 2))
        ugrid2 = np.linspace(0, umax, int(usize / 2))
        ugrid1 = np.delete(ugrid1, -1)
        self.ugrid = np.concatenate([ugrid1, ugrid2])

        # self.xgrid = np.linspace(xmin, xmax, xsize-1)
        # self.xgrid = np.append(self.xgrid, [0])
        # self.xgrid.sort(axis=0)

        self.xdim = len(self.xgrid)

        #self.ygrid = np.linspace(ymin, ymax, ysize)


        self.xmesh, self.umesh = np.meshgrid(self.xgrid, self.ugrid, indexing='ij')

        self.delta_px, self.delta_mx = calculate_delta(self.xgrid)
        self.delta_pu, self.delta_mu = calculate_delta(self.ugrid)
        self.udim = len(self.ugrid)


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


def extract_x0_result(res: np.array, x_grid: np.array, y_grid: np.array):

    x0_pos = np.where(x_grid == 0)[0][0]
    y0_pos = np.where(y_grid == 0)[0][0]

    return res[x0_pos, y0_pos]


if __name__ == "__main__":

    tmp = np.array([0, 2, 3, 7, 11, 22, 44, 45])
    calculate_delta(tmp)


