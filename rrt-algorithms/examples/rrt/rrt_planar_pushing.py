# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np

import argparse
from enum import Enum
import pickle
import sys
sys.path.append('../../')
from rrt_pack.rrt.planar_rrt import PlanarRRT
from rrt_pack.rrt.planar_rrt_star import PlanarRRTStar
from rrt_pack.rrt.planar_rrt_connect import PlanarRRTConnect
from rrt_pack.search_space.planar_search_space import PlanarSearchSpace
from rrt_pack.utilities.planar_plotting import PlanarPlot


class Method(Enum):
       rrt = 'rrt'
       rrt_star = 'rrt_star'
       rrt_connect = 'rrt_connect'

       def __str__(self):
              return self.value


parser = argparse.ArgumentParser()
parser.add_argument('-sf', '--save_file', action='store_true', default=True, help='save the planning result to pickle file')
parser.add_argument('-pf', '--plot_figure', action='store_true', default=True, help='plot fdata in figures')
parser.add_argument('-m', '--method', type=Method, choices=list(Method), help='planning method', required=True)
args = parser.parse_args()


save_file = args.save_file
plot_figure = args.plot_figure


X_dimensions = np.array([(-0.1, 0.6), (-0.1, 0.6), (-np.pi, np.pi)])

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
# Obstacles = np.array([(0.0, 0.0, 0.5, 0.025), 
#                       (0.0, 0.025, 0.025, 0.5),
#                       (0.125, 0.025, 0.5, 0.175),
#                       (0.025, 0.475, 0.5, 0.5),
#                       (0.475, 0.175, 0.5, 0.475),
#                       (0.125, 0.275, 0.475, 0.375)])

# Letter 'C'
# x_init = (0.38, 0.10, np.pi / 2)  # starting location
# x_goal = (0.38, 0.40, -np.pi / 2)  # goal location

# Letter 'F' | transition 1
# x_init = (0.405, 0.225, np.pi / 2)  # starting location
# x_goal = (0.405, 0.425, np.pi / 2)  # goal location

# Letter 'F' | transition 2
# x_init = (0.075, 0.095, 0.)  # starting location
# x_goal = (0.405, 0.425, np.pi / 2)  # goal location

# object retrieval from clutter
O_file = '../../rrt_pack/search_space/data/debug_obs3.npy'
O_index = 0
Obstacles = np.load(O_file)
x_init = tuple(Obstacles[O_index, :])  # starting location
x_goal = (0.35, 0.48, 0.)  # goal location


slider_geom = [0.07, 0.12]

Q = np.arange(0.1, 1.1, 0.1)  # length of tree edges
r = 0.001  # length of smallest edge to check for intersection with obstacles
max_samples = 1024  # max number of samples to take before timing out
prc = 0.1  # probability of checking for a connection to goal
pri = 0.2 # probability of generating new vertex from current vertex

rewire_count = 32  # optional, number of nearby branches to rewire

# create search space
miu = 0.2
ab_ratio = 1 / 726.136

# Solid letter obstacle case
# X = PlanarSearchSpace(X_dimensions, Obstacles)

# Object retrieval case
X = PlanarSearchSpace(X_dimensions, O=None, O_file=O_file, O_index=O_index)
Obstacles = np.delete(Obstacles, O_index, axis=0)

X.create_slider_geometry(geom=slider_geom)
X.create_slider_dynamics(ratio=ab_ratio, miu=miu)

# check goal feasibility
if not X.obstacle_free(x_goal):
       raise Exception('Planar RRT: Goal pose infeasible!')

# create rrt_search
if args.method == Method.rrt:
       rrt = PlanarRRT(X, Q, x_init, x_goal, max_samples, r, prc, pri)
       path = rrt.rrt_search()

elif args.method == Method.rrt_star:
       rrt = PlanarRRTStar(X, Q, x_init, x_goal, max_samples, r, prc, pri, rewire_count)
       path = rrt.rrt_star()

elif args.method == Method.rrt_connect:
       rrt = PlanarRRTConnect(X, Q, x_init, x_goal, max_samples, r, prc)
       path = rrt.rrt_connect()


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
       plot.plot_pointcloud(plot_col=False, plot_free=False)
       plot.draw(auto_open=True)
       
       import pdb; pdb.set_trace()
       
       np.save('../../output/data/rrt_planar_pushing_samples.npy', np.array(list(rrt.trees[0].E.keys())))
       np.save('../../output/data/rrt_planar_pushing_obstacles.npy', Obstacles)
       
       # plot 2d visualization
       plot.load_result_file(data=dic)
       plot.plot_rrt_result(scatter=True)
       
       import pdb; pdb.set_trace()

