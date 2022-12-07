import numpy as np
import pickle

from rrt_pack.utilities.geometry import dist_between_points_arc, rotation_matrix

from rrt_pack.search_space.planar_search_space import PlanarSearchSpace


class KinoPlanData(object):
    """
    Kinematic planning result with RRT
    """
    def __init__(self, filename) -> None:
        # self.filename = "../../output/data/" + filename + ".pkl"
        self.filename = "D:/learning/PlanarPushing/rrt-algorithms/output/data/" + filename + ".pkl"
        self.data = None
        
    def data_parser(self):
        """
        Get the COG trajectory, obstacle information from pickle file
        """
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
                     'dubins': None,
                     'path_len': None
                     }
            
    def flat_input_decider(self):
        """
        Compute the differential flatness input
        - contact point
        - force direction
        - path length
        - dubins curve parameters (yaw, center, radius)
        """
        X = PlanarSearchSpace(dimension_lengths=self.base_info['x_dims'],
                              O=self.obstacles)
        X.create_slider_geometry(self.base_info['geom'])
        X.create_slider_dynamics(self.base_info['ab_ratio'],
                                 self.base_info['miu'])
        dirs, pts = np.zeros((0, 2)), np.zeros((0, 2))
        dubins = np.zeros((0, 5))
        path_len = np.zeros((0, 1))
        for i in range(len(self.path) - 1):
            pose_now, pose_next = self.path[i], self.path[i + 1]
            _, dir, pt = X.flatness_free(start=pose_now,
                                         end=pose_next)
            # keep the direction that pushes the slider forward
            rot_mat = rotation_matrix(pose_now[2])
            idx = np.argmax(dir.T @ rot_mat.T @ (pose_next[:2] - pose_now[:2]))
            dirs = np.concatenate((dirs, np.expand_dims(dir[:, idx], axis=0)), axis=0)
            pts = np.concatenate((pts, np.expand_dims(pt[:, idx], axis=0)), axis=0)
            # dubins parameters
            revol = X.pose2steer(pose_now, pose_next)
            lenth = dist_between_points_arc(pose_now[:2], pose_next[:2], revol)
            # azimuth angle relative to dubins rotation center
            theta0 = np.arctan2(pose_now[1] - revol.y, pose_now[0] - revol.x)
            dtheta = revol.theta
            center_x = revol.x
            center_y = revol.y
            radious = np.linalg.norm(pose_now[:2] - np.array([revol.x, revol.y]), ord=2)
            path_len = np.concatenate((path_len, np.array([[lenth]])), axis=0)
            dubins = np.concatenate((dubins, np.array([[theta0, dtheta, center_x, center_y, radious]])), axis=0)
        
        self.data['contact'] = pts
        self.data['force'] = dir
        self.data['path_len'] = path_len
        self.data['dubins'] = dubins
        
    def data_packer(self):
        """
        Return the data in a dictionary
        """
        self.data_parser()
        self.flat_input_decider()
        return self.data
    
    
if __name__ == '__main__':
    plan_data = KinoPlanData(filename="rrt_planar_pushing_test")
    data = plan_data.data_packer()
    import pdb; pdb.set_trace()
