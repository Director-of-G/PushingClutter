import numpy as np
from rrt_pack.utilities.math import get_arg_k_min

class PlanarIndex(object):
    def __init__(self, weights) -> None:
        self.weights = weights
        self.data = np.zeros((len(weights), 0))
        
    def insert(self, pt):
        if type(pt) is not np.ndarray:
            pt = np.array(pt)
        if len(pt.shape) == 1:
            pt = np.expand_dims(pt, 1)
        self.data = np.concatenate((self.data, pt), axis=1)
            
    def dist(self, pt):
        if type(pt) is not np.ndarray:
            pt = np.array(pt)
        if len(pt.shape) == 1:
            pt = np.expand_dims(pt, 1)
        diff = self.data - pt
        diff[2, diff[2] > np.pi] = 2 * np.pi - diff[2, diff[2] > np.pi]
        diff[2, diff[2] < -np.pi] = 2 * np.pi + diff[2, diff[2] < -np.pi]
        weighted_diff = np.diag(np.sqrt(self.weights)) @ diff
        dist = np.linalg.norm(weighted_diff, ord=2, axis=0)
        
        return dist
    
    def nearest(self, pt, num_results):
        idx = get_arg_k_min(self.dist(pt), num_results)
        pts = self.data[:, idx.tolist()]
        for i in range(pts.shape[1]):
            yield tuple(pts[:, i])
            
    def count(self, pt):
        return np.sum(self.dist(pt) < 1e-8)


class PlanarTree(object):
    def __init__(self, X):
        """
        Tree representation
        :param X: Search Space
        """
        weights = [1, 1, np.linalg.norm(X.geom, ord=2)/2]
        self.weights = weights
        self.V = PlanarIndex(weights)  # vertices in an rtree
        self.V_count = 0
        self.E = {}  # edges in form E[child] = parent
