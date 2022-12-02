# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import numpy as np
from rtree import index

import sys
sys.path.append('../../')
from rrt_pack.utilities.math import angle_clip
from rrt_pack.utilities.geometry import es_points_along_line, es_points_along_arc, Revolute, rotation_matrix
from rrt_pack.utilities.obstacle_generation import obstacle_generator


class PlanarSearchSpace(object):
    def __init__(self, dimension_lengths, O=None):
        """
        Initialize Search Space
        :param dimension_lengths: range of each dimension
        :param O: list of obstacles
        """
        # slider geometry
        self.geom = None
        self.slider_relcoords = None  # relative coords of vertices in slider frame
        # sanity check
        if len(dimension_lengths) < 2:
            raise Exception("Must have at least 2 dimensions")
        self.dimensions = len(dimension_lengths)  # number of dimensions
        # sanity checks
        if any(len(i) != 2 for i in dimension_lengths):
            raise Exception("Dimensions can only have a start and end")
        if any(i[0] >= i[1] for i in dimension_lengths):
            raise Exception("Dimension start must be less than dimension end")
        self.dimension_lengths = dimension_lengths  # length of each dimension
        p = index.Property()
        p.dimension = int(len(O[0]) / 2)
        if O is None:
            self.obs = index.Index(interleaved=True, properties=p)
        else:
            # r-tree representation of obstacles
            # sanity check
            if any(len(o) / 2 != p.dimension for o in O):
                raise Exception("Obstacle has incorrect dimension definition")
            if any(o[i] >= o[int(i + len(o) / 2)] for o in O for i in range(int(len(o) / 2))):
                raise Exception("Obstacle start must be less than obstacle end")
            self.obs = index.Index(obstacle_generator(O), interleaved=True, properties=p)
            
    def create_slider_geometry(self, geom):
        self.geom = geom  # slider's geometric boundary
        self.slider_relcoords = np.array([[ geom[0], geom[1]],  # quad I
                                          [-geom[0], geom[1]],  # quad II
                                          [-geom[0],-geom[1]],  # quad III
                                          [ geom[0],-geom[1]]]) # quadIV
        self.slider_relcoords = self.slider_relcoords.transpose(1, 0)/2
        
    def obstacle_free_base(self, x):
        return self.obs.count(x) == 0

    def obstacle_free(self, x):
        """
        Check if a location resides inside of an obstacle
        :param x: location to check
        :return: True if not inside an obstacle, False otherwise
        """
        ptr_set = np.expand_dims(x[:2], axis=1) + rotation_matrix(x[2]) @ self.slider_relcoords
        for i in range(ptr_set.shape[1]):
            points = es_points_along_line(ptr_set[:, i % ptr_set.shape[1]], ptr_set[:, (i+1) % ptr_set.shape[1]], 0.01)  # set r to default value 0.01
            coll_free = all(map(self.obstacle_free_base, points))
            if coll_free == False:
                return False
        return True

    def sample_free(self):
        """
        Sample a location within X_free
        :return: random location within X_free
        """
        while True:  # sample until not inside of an obstacle
            x = self.sample()
            if self.obstacle_free(x):
                return x

    def collision_free(self, start, end, r):
        """
        Check if a line segment intersects an obstacle
        :param start: starting point of line
        :param end: ending point of line
        :param r: resolution of points to sample along edge when checking for collisions
        :return: True if line segment does not intersect an obstacle, False otherwise
        """
        revol = self.pose2steer(start, end)
        pt_set1, pt_set2 = self.point_pairs(self, start, end)
        
        # check arc collision
        for i in range(pt_set1.shape[1]):
            if revol.finite == True:
                points = es_points_along_arc(pt_set1[:, i], pt_set2[:, i], r, revol)
            else:
                points = es_points_along_line(pt_set1[:, i], pt_set2[:, i], r)
            coll_free = all(map(self.obstacle_free_base, points))
            if coll_free == False:
                return False
                
        return True

    def pose2steer(self, start, end):
        """
        Convert the change between pose to revolution w.r.t. fixed point
        :param start: current pose
        :param end: goal pose
        :return: Revolute (center: (x, y), angle: theta)
        """
        p_start, p_end = np.zeros((2, 2)), np.zeros((2, 2))
        p_start[0, :], p_end[0, :] = start[:2], end[:2]
        p_start[1, :] = start[:2] + np.array([np.cos(start[2]), np.sin(start[2])])
        p_end[1, :] = end[:2] + np.array([np.cos(end[2]), np.sin(end[2])])
        # A = 2 * (p_start - p_end)
        # b = np.sum(np.power(p_end, 2) - np.power(p_start, 2), axis=1)
        A = np.array([[p_start[1, 1] - p_start[0, 1], p_start[0, 0] - p_start[1, 0]],
                      [p_end[1, 1] - p_end[0, 1], p_end[0, 0] - p_end[1, 0]]])
        b = np.array([[np.linalg.det(p_start)],
                      [np.linalg.det(p_end)]])

        if np.linalg.matrix_rank(A) < 2:
            revol = Revolute(finite=False, x=0, y=0, theta=0)
        else:
            center = np.linalg.inv(A) @ b
            theta = angle_clip(end[2] - start[2])
            revol = Revolute(finite=False, x=center[0], y=center[1], theta=theta)

        return revol
    
    def point_pairs(self, start, end):
        """
        Return the corresponding point pairs on start pose and end pose
        :param start: current pose
        :param end: goal pose
        :return pt_set1: points on start pose
        :return pt_set2: points on end pose
        """
        pt_set1 = np.expand_dims(start[:2], 1) + rotation_matrix(start[2]) @ self.slider_relcoords
        pt_set2 = np.expand_dims(end[:2], 1) + rotation_matrix(end[2]) @ self.slider_relcoords
        
        return pt_set1, pt_set2

    def sample(self):
        """
        Return a random location within X
        :return: random location within X (not necessarily X_free)
        """
        x = np.random.uniform(self.dimension_lengths[:, 0], self.dimension_lengths[:, 1])
        return tuple(x)


if __name__ == '__main__':
    from rrt_pack.rrt.planar_rrt import PlanarRRT
    X_dimensions = np.array([(0, 0.5), (0, 0.5), (-np.pi, np.pi)])
    Obstacles = np.array([(0.05, 0.0, 0.5, 0.05), 
                          (0.0, 0.0, 0.05, 0.5),
                          (0.05, 0.45, 0.5, 0.5),
                          (0.45, 0.05, 0.5, 0.45),
                          (0.15, 0.15, 0.45, 0.35)])
    X = PlanarSearchSpace(X_dimensions, Obstacles)
    X.create_slider_geometry(geom=[0.07, 0.12])
    
    from matplotlib import pyplot as plt
    
    # check obstacle settings
    """
    list = []
    dims = np.array([(0, 0.5), (0, 0.5)])
    for i in range(10000):
        sample = np.random.uniform(dims[:, 0], dims[:, 1])
        if X.obs.count(sample) == 0:
            list.append(sample.tolist())
    list = np.array(list)
    plt.scatter(list[:, 0], list[:, 1])
    plt.show()
    """
    
    # check sample free
    """
    while True:
        boundary = np.array([[0.05, 0.05], [0.45, 0.05], [0.45, 0.15], [0.15, 0.15], \
                            [0.15, 0.35], [0.45, 0.35], [0.45, 0.45], [0.05, 0.45], \
                            [0.05, 0.05]])
        slider = X.sample_free()
        ptr_set = np.expand_dims(slider[:2], axis=1) + rotation_matrix(slider[2]) @ X.slider_relcoords
        ptr_set = np.concatenate((ptr_set, np.expand_dims(ptr_set[:, 0], 1)), axis=1).T
        arrow = rotation_matrix(slider[2]) @ np.array([1., 0.])
        arrow = 0.035 * arrow / np.linalg.norm(arrow, ord=2)
        plt.arrow(slider[0], slider[1], arrow[0], arrow[1])
        plt.plot(boundary[:, 0], boundary[:, 1])
        plt.plot(ptr_set[:, 0], ptr_set[:, 1])
        plt.gca().set_aspect('equal')
        plt.xlim(0, 0.5)
        plt.ylim(0, 0.5)
        plt.show()
    """
    
    # check pose2steer
    while True:
        boundary = np.array([[0.05, 0.05], [0.45, 0.05], [0.45, 0.15], [0.15, 0.15], \
                            [0.15, 0.35], [0.45, 0.35], [0.45, 0.45], [0.05, 0.45], \
                            [0.05, 0.05]])
        plt.plot(boundary[:, 0], boundary[:, 1])
        
        slider1 = X.sample_free()
        ptr_set1 = np.expand_dims(slider1[:2], axis=1) + rotation_matrix(slider1[2]) @ X.slider_relcoords
        ptr_set1 = np.concatenate((ptr_set1, np.expand_dims(ptr_set1[:, 0], 1)), axis=1).T
        arrow1 = rotation_matrix(slider1[2]) @ np.array([1., 0.])
        arrow1 = 0.035 * arrow1 / np.linalg.norm(arrow1, ord=2)
        plt.arrow(slider1[0], slider1[1], arrow1[0], arrow1[1])
        plt.plot(ptr_set1[:, 0], ptr_set1[:, 1], color='red')
        
        slider2 = X.sample_free()
        ptr_set2 = np.expand_dims(slider2[:2], axis=1) + rotation_matrix(slider2[2]) @ X.slider_relcoords
        ptr_set2 = np.concatenate((ptr_set2, np.expand_dims(ptr_set2[:, 0], 1)), axis=1).T
        arrow2 = rotation_matrix(slider2[2]) @ np.array([1., 0.])
        arrow2 = 0.035 * arrow2 / np.linalg.norm(arrow2, ord=2)
        plt.arrow(slider2[0], slider2[1], arrow2[0], arrow2[1])
        plt.plot(ptr_set2[:, 0], ptr_set2[:, 1], color='orange')
        
        revol = X.pose2steer(slider1, slider2)
        plt.scatter(revol.x, revol.y)
        
        plt.gca().set_aspect('equal')
        plt.show()
