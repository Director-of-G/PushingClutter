# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import plotly as py
from plotly import graph_objs as go
from shapely.plotting import plot_polygon

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import pickle

import sys
sys.path.append('../../')
from rrt_pack.search_space.planar_search_space import PlanarSearchSpace
from rrt_pack.utilities.geometry import rotation_matrix, angle_limit, planar_dist_between_points, es_points_along_arc

colors = ['darkblue', 'teal']

def plot_obstacles(ax, Obstacles):
    for i in range(len(Obstacles)):
        x1, y1, x2, y2 = Obstacles[i]
        width, height = x2 - x1, y2 - y1
        rect = plt.Rectangle((x1, y1), width, height, color='grey')
        ax.add_patch(rect)

class PlanarPlot(object):
    def __init__(self, filename=None, X=None):
        """
        Create a plot
        :param filename: filename
        """
        if filename is not None:
            self.filename = "../../output/data/" + filename + ".pkl"
            self.htmlname = "../../output/visualizations/" + filename + ".html"
            self.obscloud_filename = "../../output/data/X_collision_F_letter.npy"  # sampled point cloud that represent X_collision
            self.freecloud_filename = "../../output/data/X_free_F_letter.npy"  # sampled point cloud that represent X_free
        self.obstacles = None
        self.path = None
        self.tree = None
        self.X = X  # search space (Cfree)
        self.X_dimensions = self.X.X_dimensions
        self.r = 0.001  # resolution when plotting tree branches
        
        # for plotly
        self.data = []
        self.layout = {'title': 'Plot',
                       'showlegend': False,
                       'xaxis': {'autorange': False, 'range': [-2.5, 2.5]},
                       'yaxis': {'autorange': False, 'range': [-2.5, 2.5]},
                    #    'zaxis': {'autorange': False, 'range': [-2.5, 2.5]}
                       }

        self.fig = {'data': self.data,
                    'layout': self.layout}
        
    def load_result_file(self, data=None):
        if data is not None:
            result = data
        else:
            file = open(self.filename, 'rb')
            result = pickle.load(file)
        self.obstacles = result['obstacles']
        self.path = np.array(result['path'])

    def plot_obstacles(self, ax, Obstacles, scatter=False):
        for i in range(len(Obstacles)):
            if not scatter:
                x1, y1, x2, y2 = Obstacles[i]
                width, height = x2 - x1, y2 - y1
                rect = plt.Rectangle((x1, y1), width, height, color='grey')
                ax.add_patch(rect)
            else:
                coord = Obstacles[i]
                ptr_set, arrow = self.plot_slider(coord)
                ax.arrow(coord[0], coord[1], arrow[0], arrow[1])
                ax.plot(ptr_set[:, 0], ptr_set[:, 1], color='red')
        
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
            forward_dir = rotation_matrix(self.path[i, 2]).T @ (self.path[i + 1, :2] - self.path[i, :2])
            try:
                idx = np.argmax(force_dir.T @ forward_dir)
            except:
                import pdb; pdb.set_trace()
            dirs = np.concatenate((dirs, np.expand_dims(force_dir[:, idx], axis=1)), axis=1)
            pts = np.concatenate((pts, np.expand_dims(contact_pt[:, idx], axis=1)), axis=1)
            
        return dirs, pts
    
    def plot_convex_hull(self, ax):
        """
        Plot the convex hull of obstacles
        :param ax: the canvas
        """
        plot_polygon(self.X.obs.obs_conv_hull, ax=ax, add_points=False, color='lightgreen', alpha=0.3)
    
    def plot_debug(self, samples, delay_show=False):
        """
        Plot sampled poses.
        :param samples: array of (x, y, theta)
        :param delay_show: if true, won't call plt.show() and return plt.ax
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # for i in range(len(self.obstacles)):
        #     coord = self.obstacles[i]
        #     ptr_set, arrow = self.plot_slider(coord)
        #     ax.arrow(coord[0], coord[1], arrow[0], arrow[1])
        #     ax.plot(ptr_set[:, 0], ptr_set[:, 1], color='red')
        for i in range(len(samples)):
            coord = samples[i]
            ptr_set, arrow = self.plot_slider(coord)
            ax.arrow(coord[0], coord[1], arrow[0], arrow[1])
            ax.plot(ptr_set[:, 0], ptr_set[:, 1], color='blue')
        plt.xlim(self.X_dimensions[0])
        plt.ylim(self.X_dimensions[1])
        plt.gca().set_aspect('equal')
        if not delay_show:
            plt.show()
        else:
            return ax
    
    def plot_rrt_result(self, scatter=False):
        """
        Plot obstacles, sliders (key frame locations, orientations)
        :param scatter: if True, obstacles are described by (xmin, ymin, xmax, ymax);
                        if False, obstacles are described by (x, y, theta)
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.plot_obstacles(ax, self.obstacles, scatter)
        
        if scatter:
            self.plot_convex_hull(ax)
        
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
        for i in range(self.path.shape[0] - 1):
            # plot contact point
            # import pdb; pdb.set_trace()
            contact_pt = self.path[i, :2] + rotation_matrix(self.path[i, 2]) @ pts[:, i]
            plt.scatter(contact_pt[0], contact_pt[1], color='deepskyblue')
            
            # plot contact force axis
            arrow = 0.035 * rotation_matrix(self.path[i, 2]) @ dirs[:, i]
            plt.arrow(contact_pt[0], contact_pt[1], arrow[0], arrow[1], color='deepskyblue')
        
            # plot arcs of rotation
            for i in range(len(self.path)-1):
                slider1, slider2 = self.path[i, :], self.path[i+1, :]
                revol = self.X.pose2steer(slider1, slider2)
                pt_set1, pt_set2 = self.X.point_pairs(slider1, slider2, revol)
                for j in range(pt_set1.shape[1]):
                    points = es_points_along_arc(pt_set1[:, j], pt_set2[:, j], 0.01, revol)
                    arc_points = []
                    while True:
                        try:
                            arc_points.append(list(next(points)))
                        except:
                            break
                    arc_points = np.array(arc_points)
                    plt.plot(arc_points[:, 0], arc_points[:, 1], color='deepskyblue')

        plt.xlim([0.0, 0.5])
        plt.ylim([0.0, 0.5])
        plt.gca().set_aspect('equal')
        plt.show()
        
    def plot_tree(self, X, trees):
        """
        Plot tree
        :param X: Search Space
        :param trees: list of trees
        """
        try:
            assert X.dimensions == 3  # planar pushing is represented in 3D
        except:
            print("Planar pushing should have 3 dimensions")
        self.plot_tree_3d(trees)
        
    def trans_cart2toroi(self, cart_coords):
        """
        Translate Cartesian coordinates to Toroidal coordinates
        :param cart_coords: cartesian coordinates (x, y, theta)
        """
        x, y, theta = cart_coords
        x_lb, x_ub = self.X_dimensions[0]
        y_lb, y_ub = self.X_dimensions[1]
        
        radius = 1. * (1. + (x - x_lb) / (x_ub - x_lb))
        height = -0.5 + (y - y_lb) / (y_ub - y_lb)
        
        toroi_coords = (radius * np.cos(theta), radius * np.sin(theta), height)
        
        return toroi_coords
    
    def interpolate_pose(self, start, end, p):
        """
        Get the linear interpolated pose between start and end
        :param start: start pose
        :param end: end pose
        :param p: coefficient (result=start*(1-p)+end*p)
        """
        x_interp = start[0] * (1 - p) + end[0] * p
        y_interp = start[1] * (1 - p) + end[1] * p
        theta_diff = angle_limit(end[2] - start[2])
        theta_interp = angle_limit(start[2] + theta_diff * p)
        return (x_interp, y_interp, theta_interp)
    
    def distance_between_poses(self, start, end):
        """
        Calculate the distance between start and end
        :param start: start pose
        :param end: end pose
        """
        weights = [1, 1, np.linalg.norm(self.X.geom, ord=2)/2]
        return planar_dist_between_points(start, end, weights)
            
    def plot_tree_3d(self, trees):
        """
        Plot 3D trees
        :param trees: trees to plot
        """
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        for i, tree in enumerate(trees):
            for start, end in tree.E.items():
                if end is not None:
                    distance = self.distance_between_poses(start, end)
                    N = max(10, int(distance / self.r))
                    p = np.linspace(0, 1, N)
                    scatter_pts = []
                    for k in range(len(p)):
                        pt = self.interpolate_pose(start, end, p[k])
                        pt = self.trans_cart2toroi(pt)
                        scatter_pts.append(pt)
                    scatter_pts = np.array(scatter_pts)
                    
                    # trace = go.Scatter3d(
                    #     x=scatter_pts[:, 0].tolist(),
                    #     y=scatter_pts[:, 1].tolist(),
                    #     z=scatter_pts[:, 2].tolist(),
                    #     line=dict(
                    #         color=colors[i]
                    #     ),
                    #     mode="lines"
                    # )
                    # self.data.append(trace)
                    self.ax.plot(scatter_pts[:, 0].tolist(),
                                 scatter_pts[:, 1].tolist(),
                                 scatter_pts[:, 2].tolist(),
                                 linewidth=1,
                                 color=colors[i])
                    
    def plot_path(self, X, path):
        """
        Plot path through Search Space
        :param X: Search Space
        :param path: path through space given as a sequence of points
        """
        try:
            assert X.dimensions == 3  # planar pushing is represented in 3D
        except:
            print("Planar pushing should have 3 dimensions")
        path_pts = []
        for i in range(len(path) - 1):
            distance = self.distance_between_poses(path[i], path[i + 1])
            N = max(10, int(distance / self.r))
            p = np.linspace(0, 1, N)
            for k in range(len(p)):
                pt = self.interpolate_pose(path[i], path[i + 1], p[k])
                pt = self.trans_cart2toroi(pt)
                path_pts.append(pt)
        path_pts = np.array(path_pts)
        
        # trace = go.Scatter3d(
        #     x=path_pts[:, 0].tolist(),
        #     y=path_pts[:, 1].tolist(),
        #     z=path_pts[:, 2].tolist(),
        #     line=dict(
        #         color="red",
        #         width=4
        #     ),
        #     mode="lines"
        # )
        # self.data.append(trace)
        
        self.ax.plot(path_pts[:, 0].tolist(),
                     path_pts[:, 1].tolist(),
                     path_pts[:, 2].tolist(),
                     linewidth=1,
                     color='red')
        
    def plot_start(self, X, x_init):
        """
        Plot starting point
        :param X: Search Space
        :param x_init: starting location
        """
        try:
            assert X.dimensions == 3  # planar pushing is represented in 3D
        except:
            print("Planar pushing should have 3 dimensions")
        pt = self.trans_cart2toroi(x_init)
        
        # trace = go.Scatter3d(
        #     x=[pt[0]],
        #     y=[pt[1]],
        #     z=[pt[2]],
        #     line=dict(
        #         color="orange",
        #         width=10
        #     ),
        #     mode="markers"
        # )
        # self.data.append(trace)
        
        self.ax.scatter(pt[0], pt[1], pt[2], marker='o', s=10, color='orange')
        
    def plot_goal(self, X, x_goal):
        """
        Plot goal point
        :param X: Search Space
        :param x_goal: goal location
        """
        try:
            assert X.dimensions == 3  # planar pushing is represented in 3D
        except:
            print("Planar pushing should have 3 dimensions")
        pt = self.trans_cart2toroi(x_goal)
        
        # trace = go.Scatter3d(
        #     x=[pt[0]],
        #     y=[pt[1]],
        #     z=[pt[2]],
        #     line=dict(
        #         color="green",
        #         width=10
        #     ),
        #     mode="markers"
        # )
        # self.data.append(trace)
        
        self.ax.scatter(pt[0], pt[1], pt[2], marker='o', s=10, color='orange')
        
    def plot_cylindrical_surface(self, height, radius):
        theta = np.linspace(0, 2 * np.pi, 256).reshape(1, 256)
        z = np.arange(-0.5 * height, 0.5 * height, 0.01 * height).reshape(100, 1)
        Z = np.repeat(z, 256).reshape(100, 256)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        self.ax.plot_surface(x, y, Z, color='grey', alpha=0.5)
        
    def plot_pointcloud(self, plot_col=False, plot_free=False):
        """
        Plot sampled points in X_free(green) and X_collision(red) respectively
        """
        if plot_col:
            obs_cloud = np.load(self.obscloud_filename)
            xcol_pts = []  # samples points in X_collision
            for i in range(len(obs_cloud)):
                pt = self.trans_cart2toroi(obs_cloud[i, :])
                xcol_pts.append(pt)
            xcol_pts = np.array(xcol_pts)
            self.ax.scatter(xcol_pts[:, 0],
                            xcol_pts[:, 1],
                            xcol_pts[:, 2],
                            s=0.5,
                            color='red',
                            alpha=0.5)
        
        if plot_free:
            free_cloud = np.load(self.freecloud_filename)
            xfree_pts = []  # samples points in X_collision
            for i in range(len(free_cloud)):
                pt = self.trans_cart2toroi(free_cloud[i, :])
                xfree_pts.append(pt)
            xfree_pts = np.array(xfree_pts)
            self.ax.scatter(xfree_pts[:, 0],
                            xfree_pts[:, 1],
                            xfree_pts[:, 2],
                            s=0.5,
                            color='green',
                            alpha=0.5)
        
    def draw(self, auto_open=True):
        """
        Render the plot to a file
        """
        fig = self.fig
        ax  =self.ax
        self.ax.set_xlim([-2.5, 2.5])
        self.ax.set_ylim([-2.5, 2.5])
        self.ax.set_zlim([-1.5, 1.5])
        plt.show()
        # py.offline.plot(self.fig, filename=self.htmlname, auto_open=auto_open)
        
        
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
    X.create_slider_dynamics(ratio = 1 / 726.136, miu=0.2)
    
    from rrt_pack.utilities.sampling import WeightedSampler
    sampler = WeightedSampler()
    samples = []
    for i in range(200):
        sample = sampler.sample(X_dimensions[:, 0], X_dimensions[:, 1])
        samples.append(sample.tolist())
    plot = PlanarPlot(filename='rrt_planar_pushing',
                      X=X)
    plot.plot_debug(samples)
    
    # plot 2d visualization
    # plot.load_result_file()
    # plot.plot_rrt_result()
    
    # test plot the sampled X_free
    # plot.fig = plt.figure()
    # plot.ax = Axes3D(plot.fig)
    # plot.plot_cylindrical_surface(1, 1)
    # plot.plot_cylindrical_surface(1, 2)
    # plot.plot_pointcloud(plot_col=False, plot_free=True)
    # plot.draw(auto_open=True)
    
    # samples = np.load('../../output/data/rrt_planar_pushing_samples.npy')
    # obstacles = np.load('../../output/data/rrt_planar_pushing_obstacles.npy')
    # plot.obstacles = obstacles
    # plot.plot_debug(samples)
