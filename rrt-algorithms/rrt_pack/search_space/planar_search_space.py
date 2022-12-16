# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import numpy as np
from rtree import index

from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

from rrt_pack.utilities.math import angle_limit, angle_diff
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
        self.ab_ratio = None  # the exact value of (a/b) in differential flatness
        self.miu = None
        self.slider_relcoords = None  # relative coords of vertices in slider frame
        self.X_dimensions = dimension_lengths
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
        
    def create_slider_dynamics(self, ratio, miu):
        self.ab_ratio = ratio
        self.miu = miu
        
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
            
    def sample_collision(self):
        """
        Sample a location outside X_free
        :return: random location outside X_free
        """
        while True:
            x = self.sample()
            if not self.obstacle_free(x):
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
        pt_set1, pt_set2 = self.point_pairs(start, end, revol)
        
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
    
    def flatness_free(self, start, end):
        """
        Check if a Revolute satisfies differential flatness constraints
        If true, compute the contact point and force direction
        :param start: starting pose
        :param end: ending pose
        :return: True if Revolute is feasible, and False if Revolute is unfeasible
        """
        revol = self.pose2steer(start, end)
        Xrev = np.array([revol.x, revol.y])
        Xrev = np.linalg.inv(rotation_matrix(start[2])) @ (Xrev - start[:2])
        Xc, Yc = Xrev[0] + np.sign(Xrev[0]) * 1e-5, Xrev[1] + np.sign(Xrev[1]) * 1e-5  # ROC coordinates
        Kc = Yc / Xc
        
        # forwarding direction
        forward_dir = rotation_matrix(start[2]).T @ (np.array(end[:2]) - np.array(start[:2]))
        
        # check feasible contact point on all 4 faces
        x_lim, y_lim = 0.5 * self.geom[0], 0.5 * self.geom[1]
        force_dirs, contact_pts = [], []
        
        # +X face
        y0 = (-self.ab_ratio - x_lim * Xc) / Yc
        if (-y_lim <= y0 <= y_lim) and (Kc <= -1 / self.miu or Kc >= 1 / self.miu):
            contact_pts.append([x_lim, y0])
            if Yc >= 0:
                force_dirs.append([-Yc, Xc])
            else:
                force_dirs.append([Yc, -Xc])
        # -X face
        y0 = (-self.ab_ratio - (-x_lim) * Xc) / Yc
        if (-y_lim <= y0 <= y_lim) and (Kc <= -1 / self.miu or Kc >= 1 / self.miu):
            contact_pts.append([-x_lim, y0])
            if Yc >= 0:
                force_dirs.append([Yc, -Xc])
            else:
                force_dirs.append([-Yc, Xc])
        # +Y face
        x0 = (-self.ab_ratio - y_lim * Yc) / Xc
        if (-x_lim <= x0 <= x_lim) and (-self.miu <= Kc <= self.miu):
            contact_pts.append([x0, y_lim])
            if Xc >= 0:
                force_dirs.append([Yc, -Xc])
            else:
                force_dirs.append([-Yc, Xc])
        # -Y face
        x0 = (-self.ab_ratio - (-y_lim) * Yc) / Xc
        if (-x_lim <= x0 <= x_lim) and (-self.miu <= Kc <= self.miu):
            contact_pts.append([x0, -y_lim])
            if Xc >= 0:
                force_dirs.append([-Yc, Xc])
            else:
                force_dirs.append([Yc, -Xc])
        
        # the pusher's contact force and the slider's forwarding direction keeps acute angle
        idx = np.where((np.array(force_dirs).reshape(-1, 2) @ forward_dir) > 0)[0]
            
        if len(idx) > 0:
            contact_pts = np.array(contact_pts).T
            force_dirs = np.array(force_dirs).T
            force_dirs = force_dirs / np.linalg.norm(force_dirs, ord=2, axis=0)  # normalization
            return True, force_dirs[:, idx], contact_pts[:, idx]
        else:
            return False, None, None

    def pose2steer(self, start, end):
        """
        Convert the change between pose to revolution w.r.t. fixed point
        :param start: current pose
        :param end: goal pose
        :return: Revolute (center: (x, y), angle: theta)
        """
        p_start, p_end = np.zeros((2, 2)), np.zeros((2, 2))
        p_start[0, :], p_end[0, :] = start[:2], end[:2]
        p_start[1, :] = start[:2] + np.array([np.cos(start[2]), np.sin(start[2])]) * self.geom[0] * 0.5
        p_end[1, :] = end[:2] + np.array([np.cos(end[2]), np.sin(end[2])]) * self.geom[0] * 0.5
        # p_center = np.concatenate((np.expand_dims(p_start[0, :], 0), np.expand_dims(p_end[0, :], 0)), axis=0)
        # p_bound = np.concatenate((np.expand_dims(p_start[1, :], 0), np.expand_dims(p_end[1, :], 0)), axis=0)
        A = 2 * (p_end - p_start)
        b = np.sum(np.power(p_end, 2) - np.power(p_start, 2), axis=1)
        # A = np.array([[p_end[0, 1] - p_start[0, 1], p_start[0, 0] - p_end[0, 0]],
        #               [p_end[1, 1] - p_start[1, 1], p_start[1, 0] - p_end[1, 0]]])
        # b = np.array([[p_start[0, 0] * p_end[0, 1] - p_end[0, 0] * p_start[0, 1]],
        #               [p_start[1, 0] * p_end[1, 1] - p_end[1, 0] * p_start[1, 1]]])

        if np.linalg.matrix_rank(A) < 2:
            revol = Revolute(finite=False, x=0, y=0, theta=0)
        else:
            center = np.linalg.inv(A) @ b
            # theta = angle_limit(end[2] - start[2])
            theta = angle_diff(start[2], end[2])
            revol = Revolute(finite=True, x=center[0], y=center[1], theta=theta)

        return revol
    
    def point_pairs(self, start, end, revol=None):
        """
        Return the corresponding point pairs on start pose and end pose
        :param start: current pose
        :param end: goal pose
        :return pt_set1: points on start pose
        :return pt_set2: points on end pose
        """
        pt_set1 = np.expand_dims(start[:2], 1) + rotation_matrix(start[2]) @ self.slider_relcoords
        pt_set2 = np.expand_dims(end[:2], 1) + rotation_matrix(end[2]) @ self.slider_relcoords
        # if we know the revolution (COR, angle) between 'start' and 'end',
        # we return an extra pair of points, which is the nearest point between
        # COR and the 'start', 'end' poses
        if revol is not None:
            start_pts = np.concatenate((pt_set1, np.expand_dims(pt_set1[:, 0], axis=1)), axis=1).T
            end_pts = np.concatenate((pt_set2, np.expand_dims(pt_set2[:, 0], axis=1)), axis=1).T
            cor_pt = Point(revol.x, revol.y)
            start_poly = Polygon(start_pts)
            end_poly = Polygon(end_pts)

            nearest_pt1 = nearest_points(start_poly, cor_pt)
            nearest_pt2 = nearest_points(end_poly, cor_pt)

            pt_set1 = np.concatenate((pt_set1, np.expand_dims([nearest_pt1[0].x, nearest_pt1[1].y], axis=1)), axis=1)
            pt_set2 = np.concatenate((pt_set2, np.expand_dims([nearest_pt2[0].x, nearest_pt2[1].y], axis=1)), axis=1)
        
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
    # Letter 'C' with tunnel width=0.1
    # Obstacles = np.array([(0.05, 0.0, 0.5, 0.05), 
    #                       (0.0, 0.0, 0.05, 0.5),
    #                       (0.05, 0.45, 0.5, 0.5),
    #                       (0.45, 0.05, 0.5, 0.45),
    #                       (0.15, 0.15, 0.45, 0.35)])

    # Letter 'C' with tunnel width=0.08
    # Obstacles = np.array([(0.07, 0.0, 0.5, 0.06), 
    #                       (0.0, 0.0, 0.07, 0.5),
    #                       (0.07, 0.44, 0.5, 0.5),
    #                       (0.45, 0.06, 0.5, 0.44),
    #                       (0.15, 0.14, 0.45, 0.36)])

    # Letter 'F' with tunnel width=0.1
    Obstacles = np.array([(0.0, 0.0, 0.5, 0.025), 
                        (0.0, 0.025, 0.025, 0.5),
                        (0.125, 0.025, 0.5, 0.175),
                        (0.025, 0.475, 0.5, 0.5),
                        (0.475, 0.175, 0.5, 0.475),
                        (0.125, 0.275, 0.475, 0.375)])
    X = PlanarSearchSpace(X_dimensions, Obstacles)
    X.create_slider_geometry(geom=[0.07, 0.12])
    X.create_slider_dynamics(ratio = 1 / 726.136, miu=0.2)
    
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
    """
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
    """

    # check collision free
    """
    while True:            
        slider1 = X.sample_free()
        ptr_set1 = np.expand_dims(slider1[:2], axis=1) + rotation_matrix(slider1[2]) @ X.slider_relcoords
        ptr_set1 = np.concatenate((ptr_set1, np.expand_dims(ptr_set1[:, 0], 1)), axis=1).T
        
        slider2 = X.sample_free()
        ptr_set2 = np.expand_dims(slider2[:2], axis=1) + rotation_matrix(slider2[2]) @ X.slider_relcoords
        ptr_set2 = np.concatenate((ptr_set2, np.expand_dims(ptr_set2[:, 0], 1)), axis=1).T

        revol = X.pose2steer(slider1, slider2)
        pt_set1, pt_set2 = X.point_pairs(slider1, slider2, revol)
        
        # check differential flatness constraints
        flat_feas, force_dirs, contact_pts = X.flatness_free(slider1, slider2)
        if X.collision_free(slider1, slider2, 0.01) and flat_feas:
            try:
                print(pt_set1.shape)

                # plot Cfree boundary
                boundary = np.array([[0.05, 0.05], [0.45, 0.05], [0.45, 0.15], [0.15, 0.15], \
                                    [0.15, 0.35], [0.45, 0.35], [0.45, 0.45], [0.05, 0.45], \
                                    [0.05, 0.05]])
                plt.plot(boundary[:, 0], boundary[:, 1])

                # plot orientation arrow 1
                arrow1 = rotation_matrix(slider1[2]) @ np.array([1., 0.])
                arrow1 = 0.035 * arrow1 / np.linalg.norm(arrow1, ord=2)
                plt.arrow(slider1[0], slider1[1], arrow1[0], arrow1[1])
                plt.plot(ptr_set1[:, 0], ptr_set1[:, 1], color='red')

                # plot orientation arrow 2
                arrow2 = rotation_matrix(slider2[2]) @ np.array([1., 0.])
                arrow2 = 0.035 * arrow2 / np.linalg.norm(arrow2, ord=2)
                plt.arrow(slider2[0], slider2[1], arrow2[0], arrow2[1])
                plt.plot(ptr_set2[:, 0], ptr_set2[:, 1], color='orange')

                for i in range(pt_set1.shape[1]):
                    if revol.finite == True:
                        points = es_points_along_arc(pt_set1[:, i], pt_set2[:, i], 0.01, revol)
                        arc_points = []
                        while True:
                            try:
                                arc_points.append(list(next(points)))
                            except:
                                break
                        arc_points = np.array(arc_points)
                        plt.scatter(arc_points[:, 0], arc_points[:, 1], color='deepskyblue')
                
                # plot contact point
                contact_pts = np.expand_dims(slider1[:2], axis=1) + rotation_matrix(slider1[2]) @ contact_pts
                plt.scatter(contact_pts[0, :], contact_pts[1, :])
                
                # plot contact force axis
                for i in range(force_dirs.shape[1]):
                    arrow = 0.035 * rotation_matrix(slider1[2]) @ force_dirs[:, i]
                    plt.arrow(contact_pts[0, i], contact_pts[1, i], arrow[0], arrow[1])
                
                plt.gca().set_aspect('equal')
                plt.show()
            except:
                import pdb; pdb.set_trace()
    """
    
    # sample poses in X_collision
    sample_num = 8000
    collision_pts = []
    for i in range(sample_num):
        sample = X.sample_collision()
        collision_pts.append(sample)
    np.save('../../output/data/X_collision_F_letter.npy', collision_pts)
    # sample poses in X_free
    sample_num = 2000
    free_pts = []
    for i in range(sample_num):
        sample = X.sample_free()
        free_pts.append(sample)
    np.save('../../output/data/X_free_F_letter.npy', free_pts)
