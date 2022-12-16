import enum

import numpy as np

from rrt_pack.rrt.planar_rrt_base import PlanarRRTBase
from rrt_pack.utilities.geometry import steer, sweep, planar_dist_between_points


class Status(enum.Enum):
    FAILED = 1
    TRAPPED = 2
    ADVANCED = 3
    REACHED = 4


class PlanarRRTConnect(PlanarRRTBase):
    def __init__(self, X, Q, x_init, x_goal, max_samples, r, prc=0.01):
        """
        Template RRTConnect planner
        :param X: Search Space
        :param Q: list of lengths of edges added to tree
        :param x_init: tuple, initial location
        :param x_goal: tuple, goal location
        :param max_samples: max number of samples to take
        :param r: resolution of points to sample along edge when checking for collisions
        :param prc: probability of checking whether there is a solution
        """
        super().__init__(X, Q, x_init, x_goal, max_samples, r, prc)
        self.swapped = False
        print('Using method {0}!'.format('PlanarRRTConnect'))

    def swap_trees(self):
        """
        Swap trees only
        """
        # swap trees
        self.trees[0], self.trees[1] = self.trees[1], self.trees[0]
        self.swapped = not self.swapped

    def unswap(self):
        """
        Check if trees have been swapped and unswap
        :return: swapping actually occurs
        """
        if self.swapped:
            self.swap_trees()
            return True
        else:
            return False

    def extend(self, tree, x_rand):
        x_nearest = self.get_nearest(tree, x_rand)
        if (self.swapped == False and tree == 0) or (self.swapped == True and tree == 1):
            revol = self.X.pose2steer(x_nearest, x_rand)
            x_new = sweep(x_nearest, x_rand, revol, self.Q)
            for i in range(x_new.shape[1]-1, -1, -1):
                if self.trees[tree].V.count(x_new[:, i]) > 0 or self.X.obstacle_free(x_new[:, i]) is False:
                    continue
                x_start, x_end = x_nearest, tuple(x_new[:, i])
                if self.connect_to_point(tree, x_start, x_end):
                    # print('Successfully connect to the tree rooted at x_init')
                    if planar_dist_between_points(x_nearest, tuple(x_new[:, i]), self.trees[tree].weights) < 1e-2:
                        return tuple(x_new[:, i]), Status.REACHED
                    return tuple(x_new[:, i]), Status.ADVANCED
            return tuple(x_new[:, i]), Status.TRAPPED
        else:
            revol = self.X.pose2steer(x_rand, x_nearest)
            x_new = sweep(x_rand, x_nearest, revol, self.Q)
            for i in range(x_new.shape[1]):
                if self.trees[tree].V.count(x_new[:, i]) > 0 or self.X.obstacle_free(x_new[:, i]) is False:
                    continue
                x_start, x_end = tuple(x_new[:, i]), x_nearest
                if self.connect_to_point(tree, x_start, x_end, x_b_on_tree=True):
                    # print('Successfully connect to the tree rooted at x_goal')
                    if planar_dist_between_points(x_nearest, tuple(x_new[:, i]), self.trees[tree].weights) < 1e-2:
                        return tuple(x_new[:, i]), Status.REACHED
                    return tuple(x_new[:, i]), Status.ADVANCED
            return tuple(x_new[:, i]), Status.TRAPPED

    def connect(self, tree, x):
        S = Status.ADVANCED
        while S == Status.ADVANCED:
            x_new, S = self.extend(tree, x)
        return x_new, S

    def rrt_connect(self):
        """
        RRTConnect
        :return: set of Vertices; Edges in form: vertex: [neighbor_1, neighbor_2, ...]
        """
        self.add_vertex(0, self.x_init)
        self.add_edge(0, self.x_init, None)
        self.add_tree()
        self.add_vertex(1, self.x_goal)
        self.add_edge(1, self.x_goal, None)
        while self.samples_taken < self.max_samples:
            if self.samples_taken % 200 == 0:
                print('Total samples: {0}/{1}'.format(self.samples_taken, self.max_samples))
            x_rand = self.X.sample_free()
            x_new, status = self.extend(0, x_rand)
            if status != Status.TRAPPED:
                print('Connecting to tree!')
                x_new, connect_status = self.connect(1, x_new)
                if connect_status == Status.REACHED:
                    swapped = self.unswap()
                    nearest0, nearest1 = self.get_nearest(0, x_new), self.get_nearest(1, x_new)
                    if self.X.collision_free(nearest0, nearest1, self.r) and self.X.flatness_free(nearest0, nearest1)[0]:
                        first_part = self.reconstruct_path(0, self.x_init, nearest0)
                        second_part = self.reconstruct_path(1, self.x_goal, nearest1)
                        second_part.reverse()
                        return first_part + second_part
                    else:  # restore to before unswap()
                        if swapped:
                            self.swap_trees()
            self.swap_trees()
            # print('Swapping trees!')
            self.samples_taken += 1
