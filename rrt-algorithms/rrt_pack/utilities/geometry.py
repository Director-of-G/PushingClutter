# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

from itertools import tee

import numpy as np
from shapely import Polygon
from shapely import affinity, contains

from rrt_pack.utilities.math import angle_clip, angle_limit


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


def planar_dist_between_points(a, b, w):
    """
    Return the Euclidean distance between two points
    :param a: first point
    :param b: second point
    :param w: weights for each dimension
    :return: Weighted-sum distance between a and b
    """
    if type(a) is not np.ndarray:
        a = np.array(a)
    if type(b) is not np.ndarray:
        b = np.array(b)
    diff = a - b
    if diff[2] > np.pi:
        diff[2] = 2 * np.pi - diff[2]
    elif diff[2] < -np.pi:
        diff[2] = 2 * np.pi + diff[2]
    distance = np.sqrt(w[0] * diff[0] ** 2 + w[1] * diff[1] ** 2 + w[2] * diff[2] ** 2)
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
        next_yaw = angle_limit(start[2] + q[i] * revol.theta)
        next_point = np.concatenate((next_point, [next_yaw]), axis=0)
        pts = np.concatenate((pts, np.expand_dims(next_point, 1)), axis=1)
        
    return pts


def gen_polygon(coord, geom):
    """
    Return the shapely.Polygon object of (x, y, theta)
    :param coord: (x, y, theta) coordinates
    :param beta: (xl, yl) geometry
    :return: Polygon
    """
    x, y, theta = coord
    xl, yl = geom
    poly = Polygon([(0.5*xl, 0.5*yl), (-0.5*xl, 0.5*yl), (-0.5*xl, -0.5*yl), (0.5*xl, -0.5*yl), (0.5*xl, 0.5*yl)])
    poly = affinity.rotate(poly, theta, origin='center', use_radians=True)
    poly = affinity.translate(poly, x, y)
    
    return poly


def centering_polygon(poly:Polygon):
    """
    Move the polygon's center to origin
    :param poly: the input polygon
    :return: xoff, yoff
    """
    xoff = poly.centroid.xy[0][0]
    yoff = poly.centroid.xy[1][0]
    return affinity.translate(poly, -xoff, -yoff), xoff, yoff


def random_place_rect(bound, geom, num, max_iter, obs):
    """
    Generate rectangles that do not overlap.
    :param bound: search space (xmin, ymin, xmax, ymax)
    :param geom: rectangle geometry (xl, yl)
    :param num: expected rectangle num
    :param max_iter: max iteration
    :return: list of rectangles (x, y, theta) 
    """
    thmin, thmax = -np.pi, np.pi
    xmin, ymin, xmax, ymax = bound
    xl, yl = xmax - xmin, ymax - ymin
    bound_poly = gen_polygon(coord=(xl/2, yl/2, 0.),
                             geom=(xl, yl))
    
    # debug
    debug_obs = []
    
    # create planar indexing
    n_iters = 0
    # obs = PlanarIndex()
    while n_iters < max_iter and obs.obs_num < num:
        print('--- iteration {0}, {1} samples valid ---'.format(n_iters, obs.obs_num))
        # random sample
        # import pdb; pdb.set_trace()
        samp = np.random.uniform([xmin, ymin, thmin], [xmax, ymax, thmax])
        samp_poly = gen_polygon(samp, geom)
        if contains(bound_poly, samp_poly) and obs.count(samp, geom) == 0:
            debug_obs.append(samp.tolist())
            obs.append([samp.tolist()], [geom])
        n_iters = n_iters + 1
    
    print('Finished! Total iterations:{0}.'.format(n_iters))
    return debug_obs
    

class Revolute(object):
    def __init__(self, finite, x, y, theta) -> None:
        self.finite = finite  # False=trans(lation) only or True=revol(ution)
        self.x = x
        self.y = y
        self.theta = theta

