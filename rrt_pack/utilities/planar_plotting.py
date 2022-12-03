# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

from matplotlib import pyplot as plt
import numpy as np
import pickle

import sys
sys.path.append('../../')
from rrt_pack.search_space.planar_search_space import PlanarSearchSpace
from rrt_pack.utilities.geometry import rotation_matrix

colors = ['darkblue', 'teal']


class PlanarPlot(object):
    def __init__(self, filename, X):
        """
        Create a plot
        :param filename: filename
        """
        self.filename = "../../output/data/" + filename + ".pkl"
        self.obstacles = None
        self.path = None
        self.tree = None
        self.X = X  # search space (Cfree)
        self.layout = {'title': 'Plot',
                       'showlegend': False
                       }

        # self.fig = {'data': self.data,
        #             'layout': self.layout}
        
    def load_result_file(self):
        file = open(self.filename, 'rb')
        result = pickle.load(file)
        self.obstacles = result['obstacles']
        self.path = np.array(result['path'])

    def plot_obstacles(self, ax, Obstacles):
        for i in range(len(Obstacles)):
            x1, y1, x2, y2 = Obstacles[i]
            width, height = x2 - x1, y2 - y1
            rect = plt.Rectangle((x1, y1), width, height, color='grey')
            ax.add_patch(rect)
        
        return ax
    
    def plot_slider(self, state):
        ptr_set = np.expand_dims(state[:2], axis=1) + rotation_matrix(state[2]) @ self.X.slider_relcoords
        ptr_set = np.concatenate((ptr_set, np.expand_dims(ptr_set[:, 0], 1)), axis=1).T
        arrow = rotation_matrix(state[2]) @ np.array([1., 0.])
        arrow = 0.035 * arrow / np.linalg.norm(arrow, ord=2)
        
        return ptr_set, arrow
    
    def plot_contact(self):
        """
        Get contact locations and force directions
        """
        dirs, pts = np.zeros((2, 0)), np.zeros((2, 0))
        for i in range(len(self.path) - 1):
            _, force_dir, contact_pt = self.X.flatness_free(self.path[i], self.path[i + 1])
            # check the feasible force direction that push the slider forward
            forward_dir = np.linalg.inv(rotation_matrix(self.path[i, 2])) @ (self.path[i + 1, :2] - self.path[i, :2])
            idx = np.argmax(force_dir.T @ forward_dir)
            dirs = np.concatenate((dirs, np.expand_dims(force_dir[:, idx], axis=1)), axis=1)
            pts = np.concatenate((pts, np.expand_dims(contact_pt[:, idx], axis=1)), axis=1)
            
        return dirs, pts
    
    def plot_rrt_result(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.plot_obstacles(ax, self.obstacles)
        
        dirs, pts = self.plot_contact()
        
        all_ptrs, all_arrows = np.zeros((0, 5, 2)), np.zeros((0, 2))
        ptr_set, arrow = self.plot_slider(self.path[0, :])
        all_ptrs = np.concatenate((all_ptrs, np.expand_dims(ptr_set, 0)), axis=0)
        all_arrows = np.concatenate((all_arrows, np.expand_dims(arrow, 0)), axis=0)
        for node in self.path[1:-1]:
            ptr_set, arrow = self.plot_slider(node)
            all_ptrs = np.concatenate((all_ptrs, np.expand_dims(ptr_set, 0)), axis=0)
            all_arrows = np.concatenate((all_arrows, np.expand_dims(arrow, 0)), axis=0)
        ptr_set, arrow = self.plot_slider(self.path[-1, :])
        all_ptrs = np.concatenate((all_ptrs, np.expand_dims(ptr_set, 0)), axis=0)
        all_arrows = np.concatenate((all_arrows, np.expand_dims(arrow, 0)), axis=0)
        
        for i in range(self.path.shape[0]):
            plt.arrow(self.path[i, 0], self.path[i, 1], all_arrows[i, 0], all_arrows[i, 1])
            plt.plot(all_ptrs[i, :, 0], all_ptrs[i, :, 1], color='red')
            
        # plot contact point and force directions
        for i in range(self.path.shape[0]):
            # plot contact point
            contact_pt = self.path[i, :2] + rotation_matrix(self.path[2]) @ pts[:, i]
            plt.scatter(contact_pt[0], contact_pt[1])
            
            # plot contact force axis
            arrow = 0.035 * rotation_matrix(self.path[2]) @ dir[:, i]
            plt.arrow(contact_pt[0], contact_pt[1], arrow[0], arrow[1])
        
        plt.xlim([0.0, 0.5])
        plt.ylim([0.0, 0.5])
        plt.gca().set_aspect('equal')
        plt.show()
        
        
if __name__ == '__main__':
    X_dimensions = np.array([(0, 0.5), (0, 0.5), (-np.pi, np.pi)])
    Obstacles = np.array([(0.05, 0.0, 0.5, 0.05), 
                        (0.0, 0.0, 0.05, 0.5),
                        (0.05, 0.45, 0.5, 0.5),
                        (0.45, 0.05, 0.5, 0.45),
                        (0.15, 0.15, 0.45, 0.35)])
    slider_geom = [0.07, 0.12]
    # [(0.38, 0.1, 1.5707963267948966), (0.38, 0.4, -1.5707963267948966)]
    
    X = PlanarSearchSpace(X_dimensions, Obstacles)
    X.create_slider_geometry(geom=slider_geom)
    plot = PlanarPlot(filename='rrt_planar_pushing',
                      X=X)
    plot.load_result_file()
    plot.plot_rrt_result()
