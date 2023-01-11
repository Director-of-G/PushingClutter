# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import random
import numpy as np
from rrt_pack.utilities.math import angle_diff
from rrt_pack.utilities.geometry import planar_dist_between_points

class WeightedSampler(object):
    def __init__(self) -> None:
        self.central_path = np.load('../../rrt_pack/search_space/data/central_path6_2.npy')
        self.gauss_weight = np.array([1.0, 1.0, 0.0695*100])
        # self.gauss_weight = np.array([1.0, 1.0, 0.])
        self.sigma = 0.15
    def sample(self, lower, upper):
        while True:
            u = np.random.uniform(lower, upper)
            dist = [planar_dist_between_points(u, self.central_path[i], self.gauss_weight) for i in range(len(self.central_path))]
            dist = np.min(dist)
            prob = np.exp(-(dist/self.sigma)**2)
            
            rand = random.uniform(0, 1)
            if rand < prob:
                return u

if __name__ == '__main__':
    pass
