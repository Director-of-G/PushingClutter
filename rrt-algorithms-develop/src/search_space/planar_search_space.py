# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import numpy as np
from rtree import index

from src.utilities.math import angle_clip
from src.utilities.geometry import es_points_along_line, Revolute
from src.utilities.obstacle_generation import obstacle_generator


class PlanarSearchSpace(object):
    def __init__(self, dimension_lengths, O=None):
        """
        Initialize Search Space
        :param dimension_lengths: range of each dimension
        :param O: list of obstacles
        """
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
        p.dimension = self.dimensions
        if O is None:
            self.obs = index.Index(interleaved=True, properties=p)
        else:
            # r-tree representation of obstacles
            # sanity check
            if any(len(o) / 2 != len(dimension_lengths) for o in O):
                raise Exception("Obstacle has incorrect dimension definition")
            if any(o[i] >= o[int(i + len(o) / 2)] for o in O for i in range(int(len(o) / 2))):
                raise Exception("Obstacle start must be less than obstacle end")
            self.obs = index.Index(obstacle_generator(O), interleaved=True, properties=p)

    def obstacle_free(self, x):
        """
        Check if a location resides inside of an obstacle
        :param x: location to check
        :return: True if not inside an obstacle, False otherwise
        """
        return self.obs.count(x) == 0

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
        points = es_points_along_line(start, end, r)
        coll_free = all(map(self.obstacle_free, points))
        return coll_free

    def pose2steer(self, start, end):
        """
        Convert the change between pose to revolution w.r.t. fixed point
        :param start: current pose
        :param end: goal pose
        :return: Revolute (center: (x, y), angle: theta)
        """
        p_start, p_end = np.zeros((2, 2)), np.zeros((2, 2))
        p_start[0, :], p_end[0, :] = start[:2], end[:2]
        p_start[1, :] = p_start[0, :] + np.array([np.cos(start[2]), np.sin(start[2])])
        p_end[1, :] = p_end[0, :] + np.array([np.cos(end[2]), np.sin(end[2])])
        A = 2 * (p_start - p_end)
        b = np.sum(np.power(p_end) - np.power(p_start), axis=1)

        if np.linalg.matrix_rank(A) < 2:
            revol = Revolute(finite=False, x=0, y=0, theta=0)
        else:
            center = np.linalg.inv(A) @ b
            theta = angle_clip(end[2] - start[2])
            revol = Revolute(finite=False, x=center[0], y=center[1], theta=theta)

        return revol

    def sample(self):
        """
        Return a random location within X
        :return: random location within X (not necessarily X_free)
        """
        x = np.random.uniform(self.dimension_lengths[:, 0], self.dimension_lengths[:, 1])
        return tuple(x)
