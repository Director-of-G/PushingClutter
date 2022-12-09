# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np

import pickle
import sys
sys.path.append('../../')
from rrt_pack.rrt.planar_rrt import PlanarRRT
from rrt_pack.rrt.planar_rrt_star import PlanarRRTStar
from rrt_pack.search_space.planar_search_space import PlanarSearchSpace
from rrt_pack.utilities.planar_plotting import PlanarPlot


save_file = True
plot_figure = True


X_dimensions = np.array([(0, 0.5), (0, 0.5), (-np.pi, np.pi)])

# Letter 'C' with tunnel width=0.1
# Obstacles = np.array([(0.05, 0.0, 0.5, 0.05), 
#                       (0.0, 0.0, 0.05, 0.5),
#                       (0.05, 0.45, 0.5, 0.5),
#                       (0.45, 0.05, 0.5, 0.45),
#                       (0.15, 0.15, 0.45, 0.35)])

# Letter 'C' with tunnel width=0.08
# Obstacles = np.array([(0.07, 0.0, 0.5, 0.06), 
#                       (0.0, 0.0, 0.07, 0.5),
#                       (0.07, 0.44, 0.5, 0.5),
#                       (0.45, 0.06, 0.5, 0.44),
#                       (0.15, 0.14, 0.45, 0.36)])

# Letter 'F' with tunnel width=0.1
Obstacles = np.array([(0.0, 0.0, 0.5, 0.025), 
                      (0.0, 0.025, 0.025, 0.5),
                      (0.125, 0.025, 0.5, 0.175),
                      (0.025, 0.475, 0.5, 0.5),
                      (0.475, 0.175, 0.5, 0.475),
                      (0.125, 0.275, 0.475, 0.375)])

# Letter 'C'
# x_init = (0.38, 0.10, np.pi / 2)  # starting location
# x_goal = (0.38, 0.40, -np.pi / 2)  # goal location

# Letter 'F' | transition 1
# x_init = (0.405, 0.225, np.pi / 2)  # starting location
# x_goal = (0.405, 0.425, np.pi / 2)  # goal location

# Letter 'F' | transition 2
x_init = (0.075, 0.095, 0.)  # starting location
x_goal = (0.405, 0.425, np.pi / 2)  # goal location


slider_geom = [0.07, 0.12]

Q = np.arange(0.1, 1.1, 0.1)  # length of tree edges
r = 0.001  # length of smallest edge to check for intersection with obstacles
max_samples = 8192  # max number of samples to take before timing out
prc = 0.05  # probability of checking for a connection to goal

rewire_count = 32  # optional, number of nearby branches to rewire

# create search space
miu = 0.2
ab_ratio = 1 / 726.136
X = PlanarSearchSpace(X_dimensions, Obstacles)
X.create_slider_geometry(geom=slider_geom)
X.create_slider_dynamics(ratio=ab_ratio, miu=miu)

# create rrt_search
# rrt = PlanarRRT(X, Q, x_init, x_goal, max_samples, r, prc)
# path = rrt.rrt_search()

rrt = PlanarRRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)
path = rrt.rrt_star()


base_info = {'geom': slider_geom,
             'ab_ratio': ab_ratio,
             'miu': miu,
             'x_dims': X_dimensions}
dic = {'base_info': base_info,
       'obstacles': Obstacles,
       'path': np.array(path)}

if save_file:
       file = open(r'../../output/data/rrt_planar_pushing.pkl', 'wb')
       pickle.dump(dic, file)
       
if plot_figure:
       # plot rrt tree
       plot = PlanarPlot("rrt_planar_pushing", X)
       plot.plot_tree(X, rrt.trees)
       if path is not None:
              plot.plot_path(X, path)
       plot.plot_start(X, x_init)
       plot.plot_goal(X, x_goal)
       # plot configuration space
       plot.plot_cylindrical_surface(height=1.0,
                                     radius=1.0)
       plot.plot_cylindrical_surface(height=1.0,
                                     radius=2.0)
       plot.plot_pointcloud(plot_col=False, plot_free=True)
       plot.draw(auto_open=True)
       
       import pdb; pdb.set_trace()
       
       # plot 2d visualization
       plot.load_result_file(data=dic)
       plot.plot_rrt_result()
       

