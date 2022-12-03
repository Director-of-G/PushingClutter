# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

from itertools import tee

import numpy as np

from rrt_pack.utilities.math import angle_clip


def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
    
    
def dist_between_points_arc(a, b, revol):
    """
    Return the Euclidean distance between two points on an arc
    :param a: first point
    :param b: second point
    :return: Euclidean distance between a and b
    """
    center = np.array([revol.x, revol.y])
    radius = np.linalg.norm(a - center, ord=2)
    distance = np.abs(radius * revol.theta)
    return distance


def dist_between_points(a, b):
    """
    Return the Euclidean distance between two points
    :param a: first point
    :param b: second point
    :return: Euclidean distance between a and b
    """
    distance = np.linalg.norm(np.array(b) - np.array(a))
    return distance


def pairwise(iterable):
    """
    Pairwise iteration over iterable
    :param iterable: iterable
    :return: s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def es_points_along_arc(start, end, r, revol):
    """
    Equally-spaced points along an arc defined by start, end, with resolution r
    :param start: starting point
    :param end: ending point
    :param r: maximum distance between points
    :return: yields points along line from start to end, separated by distance r
    """
    d = dist_between_points_arc(start, end, revol)
    n_points = max(int(np.ceil(d / r)), 10)
    center = np.array([revol.x, revol.y])
    if n_points > 1:
        for i in range(n_points):
            next_point = center + rotation_matrix(i / n_points * revol.theta) @ (start - center)
            if type(next_point) is not tuple:
                next_point = tuple(next_point.squeeze())
            yield next_point


def es_points_along_line(start, end, r):
    """
    Equally-spaced points along a line defined by start, end, with resolution r
    :param start: starting point
    :param end: ending point
    :param r: maximum distance between points
    :return: yields points along line from start to end, separated by distance r
    """
    d = dist_between_points(start, end)
    n_points = int(np.ceil(d / r))
    if n_points > 1:
        step = d / (n_points - 1)
        for i in range(n_points):
            next_point = steer(start, end, i * step)
            yield next_point


def steer(start, goal, d):
    """
    Return a point in the direction of the goal, that is distance away from start
    :param start: start location
    :param goal: goal location
    :param d: distance away from start
    :return: point in the direction of the goal, distance away from start
    """
    start, end = np.array(start), np.array(goal)
    v = end - start
    u = v / (np.sqrt(np.sum(v ** 2)))
    steered_point = start + u * d
    return tuple(steered_point)


def sweep(start, goal, revol, q):
    """
    Return the farthest point towards goal from start
    :param start: start location
    :param goal: goal location
    :param q: discrete sweeping distance away from start
    :return: point in the direction of the goal, distance away from start
    """
    if type(start) is np.ndarray and len(start.shape) == 2:
        start = start.squeeze()
    pts = np.zeros((len(start), 0))
    center = np.array([revol.x, revol.y])
    for i in range(len(q)):
        next_point = center + rotation_matrix(q[i] * revol.theta) @ (start[:2] - center)
        next_yaw = angle_clip(start[2] + q[i] * revol.theta)
        next_point = np.concatenate((next_point, [next_yaw]), axis=0)
        pts = np.concatenate((pts, np.expand_dims(next_point, 1)), axis=1)
        
    return pts


class Revolute(object):
    def __init__(self, finite, x, y, theta) -> None:
        self.finite = finite  # False=trans(lation) only or True=revol(ution)
        self.x = x
        self.y = y
        self.theta = theta
