# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import numpy as np

from shapely import intersection, affinity
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon, LineString
from shapely.ops import nearest_points

from rrt_pack.utilities.geometry import gen_polygon


class PlanarIndex(object):
    """
    PlanarIndex is a partial extension of RTree.
    PlanarIndex provides polygon intersection, point containment detection based on Polygon.
    """
    def __init__(self, dimension_lengths, O=None, shape=None, Obj=None, Obj_shape=None) -> None:
        """
        Initialize Planar Indexing
        :param dimension_lengths: range of each dimension
        :param O: list of tuple(x, y, theta), the location of obstacles
        :param shape: list of (xl, yl), the geometry of obstacles
        :param Obj: tuple(x, y, theta), the tuple of object
        :param Obj_shape: (xl, yl), the geometry of object
        """
        self.X_dimensions = dimension_lengths
        if not (O is None and shape is None) and (len(O) != len(shape)):
            raise Exception("The size of obstacle should be compatible with their shape")
        self.obs_num = 0
        self.obs = []
        self.obj = gen_polygon(Obj, Obj_shape)
        self.obj_theta0 = Obj[2]  # the initial theta value
        self.obj_shape = Obj_shape  # default geometry
        
        # initialize all polygons
        if O is not None and shape is not None:
            self.append(O, shape)
            self.obs_conv_hull = self.calc_convex_hull()
            
        # initialize the arena
        xmin, xmax = self.X_dimensions[0]
        ymin, ymax = self.X_dimensions[1]
        self.arena = Polygon([(xmax, ymax), (xmin, ymax), (xmin, ymin), (xmax, ymin), (xmax, ymax)])
            
    def append(self, O, shape):
        """
        Add new polygons
        :param O: list of tuple(x, y, theta), the location of obstacles
        :param shape: list of (xl, yl), the geometry of obstacles
        """
        for oi, shapei in zip(O, shape):
            poly = gen_polygon(oi, shapei)
            self.obs.append(poly)
        self.obs_num += len(O)
            
    def count(self, x, shape=None):
        """
        Return the number of polygons in intersection with point x
        :param x: point (x, y) or polygon(x, y, theta)
        :param shape: slider geometry (xl, yl, rl)
        :return: number of polygons in collision with x
        """
        # basic shape
        if type(x) in [Point, LineString, Polygon]:
            return sum([not (poly.intersection(x).is_empty) for poly in self.obs])
        else:
            # point input
            if len(x) == 2:
                point = Point(x)
                return sum([not (poly.intersection(point).is_empty) for poly in self.obs])
            # polygon input
            elif len(x) == 3:
                coord, geom = x, shape
                polygon = gen_polygon(coord, geom)
                return sum([not (poly.intersection(polygon).is_empty) for poly in self.obs])
            
    def collision_free(self, pose1, pose2, shape):
        """
        Return true is movement (translation only or small angle rotation) between pose1 and pose2 are collision free
        Return false otherwise
        :param pose1: pose1
        :param pose2: pose2
        :param shape: slider geometry (xl, yl, rl)
        :return: true of false
        """
        poly1, poly2 = gen_polygon(pose1, shape), gen_polygon(pose2, shape)
        return self.count(MultiPolygon([poly1, poly2]).convex_hull) == 0
        
    def calc_convex_hull(self):
        """
        Calculate the convex hull of all polygons
        :return: Polygon
        """
        points = []
        for poly in self.obs:
            # extract coordinates
            vertex = poly.exterior.coords.xy
            points += [(vertex[0][i], vertex[1][i]) for i in range(len(vertex[0]))]
        
        return MultiPoint(points).convex_hull
    
    def nearest_dist(self, x):
        """
        Calculate the maximum obstacle-free distance from x to all obstacles.
        :param x: the query point
        :return: the largest obstacle-free distance
        """
        point = Point(x)
        multi_obs = MultiPolygon(self.obs)
        return point.distance(multi_obs)

        
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from rrt_pack.utilities.geometry import random_place_rect, rotation_matrix
    geom=[0.07, 0.12]
    xlim, ylim = 0.5, 0.5
    
    # debug_obs = random_place_rect(bound=[0.0, 0.0, xlim, ylim],
    #                               geom=geom,
    #                               num=9,
    #                               max_iter=5000,
    #                               obs=PlanarIndex())
    X_dimensions = np.array([(-0.1, 0.6), (-0.1, 0.6), (-np.pi, np.pi)])
    debug_obs = np.load('./data/debug_obs6.npy')
    debug_shape = [[0.07, 0.12] for i in range(len(debug_obs))]
    
    debug_index = PlanarIndex(X_dimensions, debug_obs, debug_shape, debug_obs[2, :], debug_shape[2])
    conv_hull = debug_index.calc_convex_hull()
    import pdb; pdb.set_trace()
    
    slider_relcoords = np.array([[ geom[0], geom[1]],  # quad I
                                 [-geom[0], geom[1]],  # quad II
                                 [-geom[0],-geom[1]],  # quad III
                                 [ geom[0],-geom[1]]]) # quad IV
    slider_relcoords = slider_relcoords.transpose(1, 0)/2
    for i, state in enumerate(debug_obs):
        ptr_set = np.expand_dims(state[:2], axis=1) + rotation_matrix(state[2]) @ slider_relcoords
        ptr_set = np.concatenate((ptr_set, np.expand_dims(ptr_set[:, 0], 1)), axis=1).T
        arrow = rotation_matrix(state[2]) @ np.array([1., 0.])
        arrow = 0.035 * arrow / np.linalg.norm(arrow, ord=2)
        
        plt.arrow(state[0], state[1], arrow[0], arrow[1])
        plt.plot(ptr_set[:, 0], ptr_set[:, 1], color='red')
    plt.gca().set_aspect('equal')
    plt.xlim([0.0, xlim])
    plt.ylim([0.0, ylim])
    plt.show()
    # np.save('./data/debug_obs.npy', debug_obs)
    
    # generate path
    # xa = np.array([0.189, 0.195])
    # xb = np.array([0.484, 0.329])
    # xm = 0.5 * (xa + xb)
    # dx = (xm - xa)[0]
    # dy = (xm - xa)[1]
    # dtheta = 2 * np.arctan(dy/dx)
    # r = 0.5*np.sqrt(dx**2 + dy**2) / (np.cos(0.5*np.pi-dtheta/2))
    # data_pts = []
    # for theta in np.linspace(0., dtheta, 100):
    #     data_pts.append((xa+np.array([0., r]) + np.array([r*np.cos(theta-0.5*np.pi), r*np.sin(theta-0.5*np.pi)])).tolist() + [-0.5*np.pi+theta])
    # for theta in np.linspace(0., dtheta, 100)[::-1]:
    #     data_pts.append((xb-np.array([0., r]) + np.array([r*np.cos(theta+0.5*np.pi), r*np.sin(theta+0.5*np.pi)])).tolist() + [-0.5*np.pi+theta])
    # np.save('./data/central_path.npy', data_pts)
    
    # data_pts = np.load('./data/central_path.npy')
    # plt.plot(data_pts[:, 0], data_pts[:, 1])
    # plt.xlim([0, 0.5])
    # plt.ylim([0, 0.5])
    # plt.gca().set_aspect('equal')
    # plt.show()
        