from quassigaussian.adi.douglas_rachford import DouglasRachfordAdi
from quassigaussian.mesher.linear_mesher import Mesher2d
import numpy as np

theta = 0
mesher = Mesher2d()
mesher.create_mesher_2d(0, 10, 100, 0, 0.99, 100, 0, 0.9, 10)
initial_curve = mesher.tgrid*0.2
kappa = 0.2

douglas_rachford = DouglasRachfordAdi(theta, mesher, initial_curve, kappa)

v_0 = np.ones(shape=(mesher.xmesh.shape))

douglas_rachford.solve(v_0)