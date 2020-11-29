from quassigaussian.adi.douglas_rachford import DouglasRachfordAdi
from quassigaussian.mesher.linear_mesher import Mesher2d
import numpy as np

theta = 0.5
mesher = Mesher2d()
mesher.create_mesher_2d(0, 1, 200, -0.05, 0.05, 201, 0, 0.001, 20)
initial_curve = mesher.tgrid*0.2
kappa = 0.3

douglas_rachford = DouglasRachfordAdi(theta, mesher, initial_curve, kappa)

v_0 = np.ones(shape=(mesher.xmesh.shape))

douglas_rachford.solve(v_0)