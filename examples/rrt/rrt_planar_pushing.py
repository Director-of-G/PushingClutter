# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np

import pickle
import sys
sys.path.append('../../')
from rrt_pack.rrt.planar_rrt import PlanarRRT
from rrt_pack.search_space.planar_search_space import PlanarSearchSpace
from rrt_pack.utilities.plotting import Plot

X_dimensions = np.array([(0, 0.5), (0, 0.5), (-np.pi, np.pi)])
Obstacles = np.array([(0.05, 0.0, 0.5, 0.05), 
                      (0.0, 0.0, 0.05, 0.5),
                      (0.05, 0.45, 0.5, 0.5),
                      (0.45, 0.05, 0.5, 0.45),
                      (0.15, 0.15, 0.45, 0.35)])
x_init = (0.38, 0.10, np.pi / 2)  # starting location
x_goal = (0.38, 0.40, -np.pi / 2)  # goal location
slider_geom = [0.07, 0.12]

Q = np.arange(0.1, 1.1, 0.1)  # length of tree edges
r = 1  # length of smallest edge to check for intersection with obstacles
max_samples = 1024  # max number of samples to take before timing out
prc = 0.1  # probability of checking for a connection to goal

# create search space
X = PlanarSearchSpace(X_dimensions, Obstacles)
X.create_slider_geometry(geom=slider_geom)

# create rrt_search
rrt = PlanarRRT(X, Q, x_init, x_goal, max_samples, r, prc)
path = rrt.rrt_search()

dic = {'obstacles': Obstacles,
       'path': np.array(path)}

file = open(r'../../output/data/rrt_planar_pushing.pkl', 'wb')
pickle.dump(dic, file)
