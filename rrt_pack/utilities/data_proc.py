import numpy as np
import pickle

import sys
sys.path.append('../../')
from rrt_pack.search_space.planar_search_space import PlanarSearchSpace


class KinoPlanData(object):
    """
    Kinematic planning result with RRT
    """
    def __init__(self, filename) -> None:
        self.filename = "../../output/data/" + filename + ".pkl"
        self.data = None
        
    def data_parser(self):
        file = open(self.filename, 'rb')
        result = pickle.load(file)
        self.base_info = result['base_info']
        self.obstacles = result['obstacles']
        self.path = result['path']
        if type(self.path[0]) is not np.ndarray:
            self.path = np.array(self.path)
            
        self.data = {'path': self.path,
                     'contact': None,
                     'force': None,
                     'obstacle': self.obstacles,
                     }
            
    def flat_input_decider(self):
        """
        Compute the differential flatness input (contact point and force direction)
        """
        X = PlanarSearchSpace(dimension_lengths=self.base_info['x_dims'],
                              O=self.obstacles)
        dirs, pts = np.zeros((0, 2)), np.zeros((0, 2))
        for i in range(len(self.path) - 1):
            pose_now, pose_next = self.path[i], self.path[i + 1]
            _, dir, pt = X.flatness_free(start=pose_now,
                                         end=pose_next)
            # keep the direction that pushes the slider forward
            idx = np.argmax(dir.T @ (pose_next[:2] - pose_now[:2]))
            dirs = np.concatenate((dirs, np.expand_dims(dir[:, idx], axis=0)), axis=0)
            pts = np.concatenate((pts, np.expand_dims(pt[:, idx], axis=0)), axis=0)
        
        self.data['contact'] = pts
        self.data['force'] = dir
        
    def data_packer(self):
        self.data_parser()
        self.flat_input_decider()
        return self.data
