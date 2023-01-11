# Author: Yongpeng Jiang
# Date: 07/01/2023
#  -------------------------------------------------------------------
# Description:
#  This script implements a brute-force global planner. The planner
#  search for key poses and a coarse plan to push the goal object
#  out of clutter. The plan is only feasible in geometric level. We
#  make the following assumptions:
#    - All objects are convex.
#    - The clutter is defined as convex hull of all obstacles.
#  -------------------------------------------------------------------

#  import libraries
#  -------------------------------------------------------------------
import numpy as np
from interval3 import Interval, IntervalSet
from shapely.geometry import MultiPoint, Polygon
from shapely import contains, intersects
from shapely import intersection
from typing import List


def get_openset_length(intervals):
    """
    Get the lengths of opensets that is within the range of given intervals.
    :param intervals: (N, 2) contains N intervals of [l, u]
    """
    interv_set = IntervalSet([Interval(rang[0], rang[1]) for rang in intervals])
    open_set = IntervalSet()
    end_pts = np.sort(intervals.reshape(-1))
    for i in range(len(end_pts)-1):
        if not ((end_pts[i] + end_pts[i+1])/2 in interv_set):
            open_set.add(Interval(end_pts[i], end_pts[i+1]))
    
    open_set = [(set.upper_bound - set.lower_bound) for set in open_set]

    return open_set

def extract_vertex_coords(polygons: List[Polygon], index=0, extract_all=False):
    """
    Get the coordinates of the polygon vertex.
    :param polygons: list of Polygon
    :param index: the index of Polygon
    :param extract_all: if true, return the coordinates of all polygon vertex
    :return: the coordinates of dimension (N, 4, 2)
             N - the number of polygons
             4 - take the rectangle as an example
             2 - each coordinate has two dimensions
    """
    poly_coords = np.zeros((0, 4, 2))
    for i, poly in enumerate([polygons[j] for j in index] if not extract_all else polygons):
        xy = poly.exterior.coords.xy
        poly_coords[i, ...] = np.c_[xy[0].tolist(), xy[1].tolist()]
    
    return poly_coords

def get_proj_lutable(polygons: List[Polygon], r):
    """
    The projection of any convex set on a 1-d line is a range [l, u].
    The function computes the projected range for each polygon on all
    directions uniformly distributed on the unit circle, with the angle
    resolution r.
    :param polygons: list of Polygon
    :param r: spatial resolution of projection directions (rad)
    :return: a look-up table of dimension (N, M, 2)
             N - the number of polygons
             M - the number of projection directions
             2 - the lower and upper bound of an interval
             the look-up table is in np.ndarray form
    """
    proj_dirs = np.arange(0, 2*np.pi, r)
    proj_vecs = np.c_[np.cos(proj_dirs), np.sin(proj_dirs)]
    
    poly_coords = extract_vertex_coords(polygons, extract_all=True)
    proj_coords = np.einsum('kij, lj -> kli', poly_coords, proj_vecs)
    
    proj_interv = np.concatenate((np.min(proj_coords, axis=-1, keepdims=True), np.max(proj_coords, axis=-1, keepdims=True)), axis=-1)
    
    return proj_interv

def check_feasible_group_poly(polygons: List[Polygon], index_o:List[int], index_t:int, r, lut=None):
    """
    Brute-force search of the largest object width to be pushed from the
    group of polygons.
    :param polygons: list of Polygon
    :param index_o: the index of obstacles
    :param index_t: the index of target object
    :param r: spatial resolution of projection directions (rad)
    :param lut: the look-up table of project intervals
    :return: to be decided
    """
    if intersection(MultiPoint(extract_vertex_coords(polygons, index_o)[-1, 2]).convex_hull, polygons[index_t]).area > \
        0.5 * polygons[index_t].area:
        return True, None
    
    proj_dirs = np.arange(0, 2*np.pi, r)
    feasible_plan = np.zeros((0, 2))  # (yaw of push, max object width)
    width_tmin = np.min(lut[index_t, :, 1] - lut[index_t, :, 1])
    
    for i in range(lut.shape[1]):
        # relative obstacles: the upper bound of its range is larger than that of the target object
        range_i = lut[:, i, :]
        rel_obs_idx = np.where(range_i[:, 1] > range_i[index_t, 1])[0]
        
        # find the orthogonal direction to i
        j = np.argmin(np.abs(proj_dirs - (proj_dirs[i] + 0.5*np.pi) % (2*np.pi)))
        range_j = lut[:, j, :]
        
        gap_length = get_openset_length(range_j[rel_obs_idx, :])
        
        if len(gap_length) > 0 and np.max(gap_length) > 1.1*width_tmin:
            feasible_plan = np.concatenate((feasible_plan, np.array([[proj_dirs[i], np.max(gap_length)]])), axis=0)
