# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import numpy as np

from shapely import intersection, affinity
from shapely.geometry import Point, MultiPoint, Polygon
from shapely.ops import nearest_points

from rrt_pack.utilities.geometry import gen_polygon


"""
PlanarIndex is a partial extension of RTree.
PlanarIndex provides polygon intersection, point containment detection based on Polygon.
"""
class PlanarIndex(object):
    def __init__(self, O=None, shape=None) -> None:
        """
        Initialize Planar Indexing
        :param O: list of tuple(x, y, theta), the location of obstacles
        :param shape: list of (xl, yl), the geometry of obstacles
        """
        if not (O is None and shape is None) and (len(O) != len(shape)):
            raise Exception("The size of obstacle should be compatible with their shape")
        self.obs_num = len(O) if O is not None else 0
        self.obs = []
        
        # initialize all polygons
        if O is not None and shape is not None:
            self.append(O, shape)
            self.obs_conv_hull = self.calc_convex_hull()
            
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
        :return: number of polygons
        """
        # point input
        if len(x) == 2:
            point = Point(x)
            return sum([not (poly.intersection(point).is_empty) for poly in self.obs])
        # polygon input
        elif len(x) == 3:
            coord, geom = x, shape
            polygon = gen_polygon(coord, geom)
            return sum([not (poly.intersection(polygon).is_empty) for poly in self.obs])
        
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

        
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from rrt_pack.utilities.geometry import random_place_rect, rotation_matrix
    geom=[0.07, 0.12]
    xlim, ylim = 0.5, 0.5
    
    debug_obs = random_place_rect(bound=[0.0, 0.0, xlim, ylim],
                                  geom=geom,
                                  num=9,
                                  max_iter=5000,
                                  obs=PlanarIndex())
    
    # debug_obs = np.load('./data/debug_obs3.npy')
    debug_shape = [[0.07, 0.12] for i in range(len(debug_obs))]
    
    debug_index = PlanarIndex(debug_obs, debug_shape)
    conv_hull = debug_index.calc_convex_hull()
    # import pdb; pdb.set_trace()
    
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
    np.save('./data/debug_obs.npy', debug_obs)
    
        