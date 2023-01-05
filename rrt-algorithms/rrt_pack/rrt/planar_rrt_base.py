import random

import numpy as np
import random
from shapely import contains

from rrt_pack.rrt.planar_tree import PlanarTree
from rrt_pack.utilities.geometry import steer, sweep, gen_polygon, rotation_matrix, Revolute


class PlanarRRTBase(object):
    def __init__(self, X, Q, x_init, x_goal, max_samples, r, prc=0.01, pri=0.0):
        """
        Template RRT planner
        :param X: Search Space
        :param Q: list of lengths of edges added to tree
        :param x_init: tuple, initial location
        :param x_goal: tuple, goal location
        :param max_samples: max number of samples to take
        :param r: resolution of points to sample along edge when checking for collisions
        :param prc: probability of checking whether there is a solution
        :param pri: probability of informed sampling (steer from current tree node)
        """
        self.X = X
        self.samples_taken = 0
        self.max_samples = max_samples
        self.Q = Q
        self.r = r
        self.prc = prc
        self.pri = pri
        self.x_init = x_init
        self.x_goal = x_goal
        self.trees = []  # list of all trees
        self.add_tree()  # add initial tree

    def add_tree(self):
        """
        Create an empty tree and add to trees
        """
        self.trees.append(PlanarTree(self.X))

    def add_vertex(self, tree, v):
        """
        Add vertex to corresponding tree
        :param tree: int, tree to which to add vertex
        :param v: tuple, vertex to add
        """
        self.trees[tree].V.insert(v)  # insert a point
        self.trees[tree].V_count += 1  # increment number of vertices in tree
        self.samples_taken += 1  # increment number of samples taken

    def add_edge(self, tree, child, parent):
        """
        Add edge to corresponding tree
        :param tree: int, tree to which to add vertex
        :param child: tuple, child vertex
        :param parent: tuple, parent vertex
        """
        self.trees[tree].E[child] = parent

    def nearby(self, tree, x, n):
        """
        Return nearby vertices
        :param tree: int, tree being searched
        :param x: tuple, vertex around which searching
        :param n: int, max number of neighbors to return
        :return: list of nearby vertices
        """
        return self.trees[tree].V.nearest(x, num_results=n)

    def get_nearest(self, tree, x):
        """
        Return vertex nearest to x
        :param tree: int, tree being searched
        :param x: tuple, vertex around which searching
        :return: tuple, nearest vertex to x
        """
        return next(self.nearby(tree, x, 1))
    
    def new_from_current(self, tree, q):
        """
        Return a new steered vertex from random current vertex
        :param tree: int, tree being searched
        :param q: length of edge when steering
        """
        v = random.sample(self.trees[tree].E.keys(), k=1)[0]
        
        # random a push face, and random a push point
        face_list = ['+x', '-x', '+y', '-y']
        f = random.choice(face_list)
        if f == '+x':
            local_xc, local_yc = np.random.uniform(-0.5*self.X.geom[1], 0.5*self.X.geom[1]), -0.5*self.X.geom[0]
        elif f == '-x':
            local_xc, local_yc = np.random.uniform(-0.5*self.X.geom[1], 0.5*self.X.geom[1]), -0.5*self.X.geom[0]
        elif f == '+y':
            local_xc, local_yc = np.random.uniform(-0.5*self.X.geom[0], 0.5*self.X.geom[0]), -0.5*self.X.geom[1]
        elif f == '-y':
            local_xc, local_yc = np.random.uniform(-0.5*self.X.geom[0], 0.5*self.X.geom[0]), -0.5*self.X.geom[1]
        
        # random a push angle
        pt_gamma = np.random.uniform(-np.arctan2(self.X.miu, 1.), np.arctan2(self.X.miu, 1.))
        slope = np.tan(0.5*np.pi+pt_gamma)
        # center of rotation in local coordinates
        local_xr = -self.X.ab_ratio / (-local_yc / slope + local_xc)
        local_yr = slope * local_xc
        
        # convert to global frame
        if f == '+x':
            yaw = v[2] + 0.5*np.pi
        elif f == '-x':
            yaw = v[2] - 0.5*np.pi
        elif f == '+y':
            yaw = v[2] + np.pi
        elif f == '-y':
            yaw = v[2]
        global_xr, global_yr = v[:2] + rotation_matrix(yaw) @ np.array([local_xr, local_yr])
        
        # compute revolution
        revol = Revolute(finite=True, x=global_xr, y=global_yr, theta=0.5*np.pi)
        x_new = sweep(v, None, revol, q)
        feas_index = -1
        for i in range(x_new.shape[1]):
            if self.trees[tree].V.count(x_new[:, i]) == 0 and \
                self.X.obstacle_free(x_new[:, i]) and \
                self.X.collision_free(v, x_new[:, i], self.r):
                feas_index = i
            else:
                break
        
        if feas_index > -1:
            print('Generating new node from current ones by steering!')
            return True, tuple(x_new[:, feas_index]), v
        else:
            return False, None, None

    def new_and_near(self, tree, q):
        """
        Return a new steered vertex and the vertex in tree that is nearest
        :param tree: int, tree being searched
        :param q: length of edge when steering
        :return: vertex, new steered vertex, vertex, nearest vertex in tree to new vertex
        """
        # choose new vertex from current vertex
        if self.pri and random.random() < self.pri:
            success, x_new, x_nearest = self.new_from_current(tree, q)
            if success:
                return x_new, x_nearest
        
        # else random choose a new vertex
        x_rand = self.X.sample_free()
        x_nearest = self.get_nearest(tree, x_rand)
        revol = self.X.pose2steer(x_nearest, x_rand)
        x_new = sweep(x_nearest, x_rand, revol, q)
        
        # check if new point is in X_free and not already in V
        feasibility = False
        feas_index = None
        for i in range(x_new.shape[1]-1, -1, -1):
            if self.trees[tree].V.count(x_new[:, i]) == 0 and self.X.obstacle_free(x_new[:, i]):
                feasibility = True
                feas_index = i
                break
        
        if not feasibility:
            return None, None
        
        self.samples_taken += 1
        return tuple(x_new[:, feas_index]), x_nearest

    def connect_to_point(self, tree, x_a, x_b, x_b_on_tree=False):
        """
        Connect vertex x_a in tree to vertex x_b
        :param tree: int, tree to which to add edge
        :param x_a: tuple, vertex
        :param x_b: tuple, vertex
        :param x_b_on_tree: bool, for rrt_connect, set x_b_on_tree=True, because x_a is x_rand and x_b is x_nearest
        :return: bool, True if able to add edge, False if prohibited by an obstacle
        """
        if x_b_on_tree is False:
            if self.trees[tree].V.count(x_b) == 0 and self.X.collision_free(x_a, x_b, self.r):
                # if tree == 0:
                flat_feas, _, _ = self.X.flatness_free(x_a, x_b)  # check feasibility based on differential flatness
                # else:  # tree == 1
                #     flat_feas, _, _ = self.X.flatness_free(x_b, x_a)  # for the tree rooted at goal, the branch grows from x_new to x_near
                if flat_feas is not True:
                    return False
                self.add_vertex(tree, x_b)
                self.add_edge(tree, x_b, x_a)
                if self.X.goal_mode == 'alterable':
                    new_poly = gen_polygon(x_b, geom=[0.07, 0.12])
                    if not contains(self.X.obs.obs_conv_hull, new_poly):
                        self.x_goal = x_b
                        self.X.goal_mode = 'invariant'
                return True
            return False
        else:
            if self.trees[tree].V.count(x_a) == 0 and self.X.collision_free(x_a, x_b, self.r):
                # if tree == 0:
                flat_feas, _, _ = self.X.flatness_free(x_a, x_b)  # check feasibility based on differential flatness
                # else:  # tree == 1
                #     flat_feas, _, _ = self.X.flatness_free(x_b, x_a)  # for the tree rooted at goal, the branch grows from x_new to x_near
                if flat_feas is not True:
                    return False
                self.add_vertex(tree, x_a)
                self.add_edge(tree, x_a, x_b)
                if self.X.goal_mode == 'alterable':
                    new_poly = gen_polygon(x_a, geom=[0.07, 0.12])
                    if not contains(self.X.obs.obs_conv_hull, new_poly):
                        self.x_goal = x_a
                        self.X.goal_mode = 'invariant'
                return True
            return False

    def can_connect_to_goal(self, tree):
        """
        Check if the goal can be connected to the graph
        :param tree: rtree of all Vertices
        :return: True if can be added, False otherwise
        """
        x_nearest = self.get_nearest(tree, self.x_goal)
        if self.x_goal in self.trees[tree].E and ((x_nearest == self.x_goal) or (x_nearest in self.trees[tree].E[self.x_goal])):
            # tree is already connected to goal using nearest vertex
            return True
        if self.X.collision_free(x_nearest, self.x_goal, self.r) and self.X.flatness_free(x_nearest, self.x_goal)[0]:  # check if obstacle-free
            return True
        return False

    def get_path(self):
        """
        Return path through tree from start to goal
        :return: path if possible, None otherwise
        """
        if self.can_connect_to_goal(0):
            print("Can connect to goal")
            self.connect_to_goal(0)
            return self.reconstruct_path(0, self.x_init, self.x_goal)
        print("Could not connect to goal")
        return None

    def connect_to_goal(self, tree):
        """
        Connect x_goal to graph
        (does not check if this should be possible, for that use: can_connect_to_goal)
        :param tree: rtree of all Vertices
        """
        x_nearest = self.get_nearest(tree, self.x_goal)
        if x_nearest != self.x_goal:
            self.trees[tree].E[self.x_goal] = x_nearest

    def reconstruct_path(self, tree, x_init, x_goal):
        """
        Reconstruct path from start to goal
        :param tree: int, tree in which to find path
        :param x_init: tuple, starting vertex
        :param x_goal: tuple, ending vertex
        :return: sequence of vertices from start to goal
        """
        path = [x_goal]
        current = x_goal
        if x_init == x_goal:
            return path
        while not self.trees[tree].E[current] == x_init:
            path.append(self.trees[tree].E[current])
            current = self.trees[tree].E[current]
        path.append(x_init)
        path.reverse()
        return path

    def check_solution(self):
        # probabilistically check if solution found
        if self.prc and random.random() < self.prc:
            print("Checking if can connect to goal at", str(self.samples_taken), "samples")
            path = self.get_path()
            if path is not None:
                return True, path
        # check if can connect to goal after generating max_samples
        if self.samples_taken >= self.max_samples:
            return True, self.get_path()
        return False, None

    def bound_point(self, point):
        # if point is out-of-bounds, set to bound
        point = np.maximum(point, self.X.dimension_lengths[:, 0])
        point = np.minimum(point, self.X.dimension_lengths[:, 1])
        return tuple(point)
