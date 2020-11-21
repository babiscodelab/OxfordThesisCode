from quassigaussian.finitedifference.adi.douglas_rachford import DouglasRachfordAdi
from quassigaussian.finitedifference.mesher.linear_mesher import Mesher2d
import numpy as np

theta = 0.5
mesher = Mesher2d()
mesher.create_mesher_2d(0, 30, 100, -0.04, 0.04, 150, 0, 0.002, 10)
initial_curve = mesher.tgrid*0.2
kappa = 0.3

douglas_rachford = DouglasRachfordAdi(theta, mesher, initial_curve, kappa)

v_0 = np.ones(shape=(mesher.xmesh.shape))*100

douglas_rachford.solve(v_0)