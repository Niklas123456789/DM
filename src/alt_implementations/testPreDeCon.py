# 2.2 Density-based Projected-Clustering (PreDeCon) - Each point is assigned to one subspace cluster or noise.
#Source: https://blog.xa0.de/post/PreDeCon%20---%20Density-based-Projected-Subspace-Clustering/

import numpy as np
#from .preDeCon import PreDeCon
from preDeCon import *

Test_1 = np.array([
    [1, 6],#Point p1
    [2, 6],#Point p2
    [3, 6],#Point p3
    [4, 6],#Point p4
    [5, 6],#Point p5
    [6, 6],#Point p6
    [7, 9],#Point p7
    [7, 8],#Point p8
    [7, 7],#Point p9
    [7, 6],#Point p10
    [7, 5],#Point p11
    [7, 4] #Point p12
])


#### MAIN ####
minpts = 3
epsilon = 1
lam = 1 #lamda
delta = 0.25
weightPunishment = 100 #K
print("TEST")

preDeCon = PreDeCon(minimumPoints=minpts, epsilon=epsilon, lam=lam, delta=delta, weightPunishment=weightPunishment)
print(preDeCon.calculate_if_point_is_core_point(Test_1, Test_1[2]))

#print('Is point p3 a core point after applying PreDeCon? ', PreDeCon.calculate_if_point_is_core_point(Test_1, Test_1[2]))
#print('Is point p6 a core point after applying PreDeCon?', calculate_if_point_is_core_point(Test_1, Test_1[5]))
