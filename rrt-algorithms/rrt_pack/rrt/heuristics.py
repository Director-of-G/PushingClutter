# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

from rrt_pack.utilities.geometry import dist_between_points, planar_dist_between_points


def cost_to_go(a: tuple, b: tuple) -> float:
    """
    :param a: current location
    :param b: next location
    :return: estimated segment_cost-to-go from a to b
    """
    return dist_between_points(a, b)


def planar_cost_to_go(a: tuple, b: tuple, w:list) -> float:
    """
    :param a: current location
    :param b: next location
    :param w: weights for each dimension
    :return: estimated segment_cost-to-go from a to b
    """
    return planar_dist_between_points(a, b, w)


def path_cost(E, a, b):
    """
    Cost of the unique path from x_init to x
    :param E: edges, in form of E[child] = parent
    :param a: initial location
    :param b: goal location
    :return: segment_cost of unique path from x_init to x
    """
    cost = 0
    while not b == a:
        p = E[b]
        cost += dist_between_points(b, p)
        b = p

    return cost


def planar_path_cost(E, a, b, w):
    """
    Cost of the unique path from x_init to x in planar pushing
    :param E: edges, in form of E[child] = parent
    :param a: initial pose
    :param b: goal pose
    :param w: weights for each dimension
    :return: segment_cost of unique path from x_init to x
    """
    cost = 0
    while not b == a:
        p = E[b]
        cost += planar_dist_between_points(b, p, w)
        b = p

    return cost


def segment_cost(a, b):
    """
    Cost function of the line between x_near and x_new
    :param a: start of line
    :param b: end of line
    :return: segment_cost function between a and b
    """
    return dist_between_points(a, b)


def planar_segment_cost(a, b, w):
    """
    Cost function of the line between x_near and x_new in planar pushing
    :param a: start of line
    :param b: end of line
    :param w: weights for each dimension
    :return: segment_cost function between a and b
    """
    return planar_dist_between_points(a, b, w)
