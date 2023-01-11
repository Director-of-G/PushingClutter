# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import cvxpy as cp
from functools import reduce
from itertools import product
import networkx as nx
import numpy as np
# import pyomo.environ as pe
# from pao.pyomo import *
import portion as P
from shapely.geometry import LineString, Polygon, MultiLineString, MultiPolygon
from shapely import affinity, intersection

import rrt_pack
from rrt_pack.search_space.planar_search_space import PlanarSearchSpace
from rrt_pack.utilities.geometry import planar_dist_between_points, centering_polygon

class MinkowskiSum(object):
    """
    MinkowskiSum generates collision free space CFree for each rotation layer
    All objects and obstacles are polygons.
    :param X: Search Space
    :param r: the resolution in collision checking
    :param w: the weight for measuring planar distance
    :param n_slice: the number of slices
    :param n_line: the number of sweeping lines per slice
    """
    def __init__(self, X:PlanarSearchSpace, r, w, n_slice, n_line) -> None:
        self.X = X
        self.G = nx.Graph()  # the Graph Structure
        self.G_index = {}  # the Graph vertex index by name
        self.theta = [0.0]  # list of discretized theta
        
        self.r = r
        self.w = w
        self.n_slice = n_slice
        self.n_line = n_line
        
    def minkowski_operation_polygons(self, poly1:Polygon, poly2:Polygon, sum=True):
        """
        Calculate the Minkowski sum or difference of two polygons, denoted by polygonB and polygonA
        The Minkowski sum here follows the common definition, but the Minkowski difference here do not
        The Minkowski difference is defined with polygon containment problem, which refers to finding
        the centroids of polygonB such that polygonB is completely within polygonA
        :param poly1, poly2: two polygons
        :return: mink_result (sum or difference), Polygon
        """
        # minkowski difference
        if sum == False:
            poly2 = affinity.rotate(poly2, np.pi, origin='center', use_radians=True)
        # convert input polygons to coordinates
        vertex1, vertex2 = self.get_polygon_coords(poly1, normalize=False, rearrange=True), \
                           self.get_polygon_coords(poly2, normalize=False, rearrange=True)
        vertex_result = []
        # initialize pointers
        ptr1, ptr2 = 0, 0
        while True:
            vertex_result.append((vertex1[ptr1%len(vertex1), :]+vertex2[ptr2%len(vertex2), :]).tolist())
            # increase the corresponding pointer
            # edge_polar1 = np.arctan2(vertex1[(ptr1+1)%len(vertex1), 1]-vertex1[ptr1%len(vertex1), 1], \
            #                          vertex1[(ptr1+1)%len(vertex1), 0]-vertex1[ptr1%len(vertex1), 0])
            # edge_polar2 = np.arctan2(vertex2[(ptr2+1)%len(vertex2), 1]-vertex2[ptr2%len(vertex2), 1], \
            #                          vertex2[(ptr2+1)%len(vertex2), 0]-vertex2[ptr2%len(vertex2), 0])
            next_edge1 = vertex1[(ptr1+1)%len(vertex1), :] - vertex1[ptr1%len(vertex1), :]
            next_edge2 = vertex2[(ptr2+1)%len(vertex2), :] - vertex2[ptr2%len(vertex2), :]
            cross = np.cross(next_edge1, next_edge2)
            # if edge_polar1 <= edge_polar2:
            #     ptr1 += 1
            # if edge_polar2 <= edge_polar1:
            #     ptr2 += 1
            if cross > 0:
                ptr1 += 1
            elif cross < 0:
                ptr2 += 1
            else:
                ptr1 += 1; ptr2 += 1
            if (ptr1 >= len(vertex1)) or (ptr2 >= len(vertex2)):
                break
        mink_result = Polygon(vertex_result)
        
        return mink_result
        
    def get_polygon_coords(self, poly:Polygon, normalize=False, rearrange=False):
        """
        Return the (x, y) coordinates of poly's vertex.
        :param poly: the Polygon object
        :param normalize: if true, move the centroid of poly to the origin
        :param rearrange: if true, roll the vertex utill the vertex with minimum y
                          (and minimum x if more than one vertex has minimum y) is
                          the first vertex
        :return: vertex, the np.ndarray of shape(N, 2)
                 each row is a pair of coordinate (x, y)
                 there are N pairs of coordinates in total
                 the coordinates form a closed linestring
        """
        vertex = poly.boundary.coords.xy
        vertex = np.c_[vertex[0].tolist(), vertex[1].tolist()]
        if normalize:
            centroid = poly.centroid.coords.xy
            vertex -= np.array([centroid[0][0], centroid[1][0]])
        if rearrange:
            ymin = np.min(vertex[:, 1])
            argmin = np.where(vertex[:, 1] == ymin)[0]
            if len(argmin) > 1:
                xmin = np.min(vertex[argmin, 0])
                argmin = np.where((vertex[:, 0] == xmin)&(vertex[:, 1] == ymin))[0]
            vertex = np.roll(vertex[:-1, :], -argmin, axis=0)
            vertex = np.append(vertex, np.expand_dims(vertex[0, :], axis=0), axis=0)
        
        return vertex
        
    def parse_polygons(self, theta):
        """
        Convert shapely.Polygons into Ax≤b form
        :param theta: the additional rotation angle (anti-clockwise) of the object
        :return: poly_mat, the dict of {index of Polygon: {'A': matrix A, 'b': matrix b}}
        """
        poly_mat = {}
        
        # rotate the object
        obj = affinity.rotate(self.X.obs.obj, theta, origin='center', use_radians=True)
        arena = self.X.obs.arena
        
        N_all = self.X.obs.obs_num + 2
        # convert all
        for idx, poly in enumerate(self.X.obs.obs + [obj, arena]):
            vertex = self.get_polygon_coords(poly, normalize=True)
            N = len(vertex)-1
            matA, vecb = np.zeros((N, 2)), np.zeros((N, 1))
            for i in range(N):
                x1, y1 = vertex[i]
                x2, y2 = vertex[i+1]
                a1, a2 = y2 - y1, x1 - x2
                b = x1 * y2 - x2 * y1
                if b >= 0:
                    matA[i, :] = [a1, a2]
                    vecb[i, :] = b
                else:
                    matA[i, :] = [-a1, -a2]
                    vecb[i, :] = -b
            if idx < N_all - 2:
                poly_mat[idx] = {'A': matA, 'b': vecb}
            elif idx == N_all - 2:
                poly_mat['obj'] = {'A': matA, 'b': vecb}  # index ['obj'] represents the object
            elif idx == N_all - 1:
                poly_mat['arena'] = {'A': matA, 'b': vecb}  # index ['arena'] represents the arena
        
        return poly_mat
    
    def get_minkowski_proj_interval(self, A1, A2, b1, b2, y0, sum=True):
        """
        Build a LP to solve the following problem:
            Compute the projection of P1⊕P2 onto a horizontal line y=y0, where P1 and P2 are two 2d polygons
        ----------
        P1 and P2 are represented as
            A1x ≤ b1    A2x ≤ b2
        The LP is defined as
            min  cx                 max  cx
            s.t. Ax ≤ b  (1)        s.t. Ax ≤ b  (2)
                 px = q                  px = q
        where
            A=[A1,0;0,A2]
            b=[b1;b2]
            c=[1,0,1,0]
            p=[0,1,0,1]
            q=y0
        Change c and p as
            c=[1,0,-1,0]
            p=[0,1,0,-1]
        we generalize the problem to compute the projection of P1⊕(-P2)
        The optimal values of (1) and (2) are respectively l and u.
        Then the projection, which is an interval, could be derived as [l,u].
        ----------
        :param A1, A2, b1, b2: problem definition matrices
        :param y0: the horizontal line
        :param sum: if true, do minkowski sum P1⊕P2; if false, do minkowski difference P1⊕(-P2)
        :return: interv, the Portion closed range [l, u]
        """
        A = np.block([[A1, np.zeros((A1.shape[0], 2))], [np.zeros((A2.shape[0], 2)), A2]])
        b = np.block([b1.reshape(-1,), b2.reshape(-1,)])
        # # Minkowski sum
        if sum:
            c = np.array([1., 0., 1., 0.])
            p = np.array([0., 1., 0., 1.])
        # # Minkowski difference
        else:
            c = np.array([1., 0., -1., 0.])
            p = np.array([0., 1., 0., -1.])
        q = y0
        
        x = cp.Variable(4)
        min_prob = cp.Problem(cp.Minimize(c.T @ x),
                            [A @ x <= b, p.T @ x == q])
        max_prob = cp.Problem(cp.Maximize(c.T @ x),
                            [A @ x <= b, p.T @ x == q])
        
        min_prob.solve()
        max_prob.solve()
        min_state, max_state = min_prob.status, max_prob.status
        
        if min_state == 'optimal' and max_state == 'optimal':
            interv = P.closed(min_prob.value, max_prob.value)
        elif min_state != 'optimal' and max_state != 'optimal':
            interv = P.empty()
        else:
            raise Exception('Unexpected status, check the problem definition!')
        # # Minkowski difference (the min-max bilevel optimization do not work)
        # else:
        #     M = pe.ConcreteModel()
        #     M.x1 = pe.Var(bounds=(None,None))
        #     M.y1 = pe.Var(bounds=(None,None))
        #     M.x2 = pe.Var(bounds=(None,None))
        #     M.y2 = pe.Var(bounds=(None,None))
            
        #     M.o = pe.Objective(expr=M.x1+M.x2, sense=pe.minimize)
        #     M.L = SubModel(fixed=[M.x1,M.y1])
        #     M.L.o = pe.Objective(expr=M.x1+M.x2, sense=pe.maximize)
            
        #     M.L.c1 = pe.Constraint(expr= A1[0,0]*M.x1+A1[0,1]*M.y1 <= b1[0,0])
        #     M.L.c2 = pe.Constraint(expr= A1[1,0]*M.x1+A1[1,1]*M.y1 <= b1[1,0])
        #     M.L.c3 = pe.Constraint(expr= A1[2,0]*M.x1+A1[2,1]*M.y1 <= b1[2,0])
        #     M.L.c4 = pe.Constraint(expr= A1[3,0]*M.x1+A1[3,1]*M.y1 <= b1[3,0])
        #     M.L.c5 = pe.Constraint(expr= A2[0,0]*M.x2+A2[0,1]*M.y2 <= b2[0,0])
        #     M.L.c6 = pe.Constraint(expr= A2[1,0]*M.x2+A2[1,1]*M.y2 <= b2[1,0])
        #     M.L.c7 = pe.Constraint(expr= A2[2,0]*M.x2+A2[2,1]*M.y2 <= b2[2,0])
        #     M.L.c8 = pe.Constraint(expr= A2[3,0]*M.x2+A2[3,1]*M.y2 <= b2[3,0])
        #     M.L.c9 = pe.Constraint(expr= M.y1+M.y2 == y0)
            
        #     M.c1 = pe.Constraint(expr= A1[0,0]*M.x1+A1[0,1]*M.y1 <= b1[0,0])
        #     M.c2 = pe.Constraint(expr= A1[1,0]*M.x1+A1[1,1]*M.y1 <= b1[1,0])
        #     M.c3 = pe.Constraint(expr= A1[2,0]*M.x1+A1[2,1]*M.y1 <= b1[2,0])
        #     M.c4 = pe.Constraint(expr= A1[3,0]*M.x1+A1[3,1]*M.y1 <= b1[3,0])
        #     M.c5 = pe.Constraint(expr= A2[0,0]*M.x2+A2[0,1]*M.y2 <= b2[0,0])
        #     M.c6 = pe.Constraint(expr= A2[1,0]*M.x2+A2[1,1]*M.y2 <= b2[1,0])
        #     M.c7 = pe.Constraint(expr= A2[2,0]*M.x2+A2[2,1]*M.y2 <= b2[2,0])
        #     M.c8 = pe.Constraint(expr= A2[3,0]*M.x2+A2[3,1]*M.y2 <= b2[3,0])
        #     M.c9 = pe.Constraint(expr= M.y1+M.y2 == y0)
            
        #     with Solver('pao.pyomo.FA') as solver:
        #         results = solver.solve(M)
            
        return interv
    
    def get_rel_coordinate(self, y_abs):
        """
        Convert the absolute y coordinate to relative y coordinate to all obstacle's and the arena's centers
        :param y_abs: the absolute y coordinate
        :return: rel_all, the dict of {index of Polygon: {'x': x_rel, 'y': y_rel}
                 - x_rel is the x translation for all obstacles and arena
                 - y_rel is the y coordinate w.r.t. all obstacles and arena
        """
        rel_all = {}
        arena = self.X.obs.arena
        
        N_all = self.X.obs.obs_num + 1  # all obstacles and the arena
        for idx, poly in enumerate(self.X.obs.obs + [arena]):
            centroid = poly.centroid.coords.xy
            if idx < N_all - 1:
                rel_all[idx] = {'x': centroid[0][0], 'y': y_abs - centroid[1][0]}
            elif idx == N_all - 1:
                rel_all['arena'] = {'x': centroid[0][0], 'y': y_abs - centroid[1][0]}
        
        return rel_all
    
    def shift_range(self, x, range):
        """
        Shift the range (l, u) -> (l+x, u+x)
        :param x: shift value
        :return: (l+x, u+x)
        """
        if range.empty:
            return range
        if range.left == P.CLOSED and range.right == P.CLOSED:
            return P.closed(range.lower+x, range.upper+x)
        elif range.left == P.CLOSED and range.right == P.OPEN:
            return P.closedopen(range.lower+x, range.upper+x)
        elif range.left == P.OPEN and range.right == P.CLOSED:
            return P.openclosed(range.lower+x, range.upper+x)
        else:
            return P.open(range.lower+x, range.upper+x)
    
    # def minkowski_operations(self, theta):
    #     """
    #     Do minkowski operations and get linear inequality representations of all polygons
    #     :param theta: the additional rotation angle (anti-clockwise) of the object
    #     :return: dict of {index of Polygon: {'A': matrix A, 'b': matrix b}}
    #              the order of polygons: all obstacles, the object, the arena
    #              - obstacles are indexed by 0, 1, ..., obs_num-1
    #              - the object is indexed by 'obs'
    #              - the arena is indexed by 'arena'
    #     """
    #     poly_mat = self.parse_polygons(theta)
        
    #     return poly_mat
    
    # def sweep_line_process(self, poly_mat, n_line):
    #     """
    #     Do sweep line operations to get the C_free (approximation)
    #     :param poly_mat: dict of {index of Polygon: {'A': matrix A, 'b': matrix b}}
    #     :param n_line: the number of sweep lines
    #     :return: C_free, the dict of {y_coord: [range1, range2, ..., rangek]}
    #              ranges are in form (l, u)
    #     """
    #     C_free = {}
        
    #     # slice the y dimension
    #     ymin, ymax = self.X.X_dimensions[1]
    #     yslice = np.linspace(ymin, ymax, n_line+2)[1:-1]  # remove the top and bottom lines
        
    #     # get polygons
    #     poly_obj, poly_arena = poly_mat['obj'], poly_mat['arena']
    #     A_obj, b_obj = poly_obj['A'], poly_obj['b']
        
    #     for i in range(n_line):
    #         y0 = yslice[i]
    #         rel_all = self.get_rel_coordinate(y0)
    #         obs_range_list = []
    #         # P_arena ⊕ P_obs
    #         for j in range(self.X.obs.obs_num):
    #             A_obs, b_obs = poly_mat[j]['A'], poly_mat[j]['b']
    #             y_rel_obs = rel_all[j]['y']
    #             mink_range = self.get_minkowski_proj_interval(A_obs, A_obj, b_obs, b_obj, y_rel_obs, sum=True)
    #             obs_range_list.append(self.shift_range(rel_all[j]['x'], mink_range))
    #         A_arena, b_arena = poly_arena['A'], poly_arena['b']
    #         y_rel_arena = rel_all['arena']['y']
    #         # P_arena ⊕ (-P_obj)
    #         arena_range = self.get_minkowski_proj_interval(A_arena, A_obj, b_arena, b_obj, y_rel_arena, sum=False)
    #         arena_range = self.shift_range(rel_all['arena']['x'], arena_range)
    #         # arena_range - U(obs_range)
    #         C_free[y0] = arena_range - reduce(lambda x, y: x | y, obs_range_list)
        
    #     return C_free
    
    def convert_mlinestring_to_ranges(self, mlinestr):
        """
        Convert the list of LineString to union of ranges [l0,u0] | [l1,u1] | ... | [lk,uk]
        :param linestr: the input list of LineString
        :return: union_ranges, [l0,u0] | [l1,u1] | ... | [lk,uk]
        """
        ranges = []
        for line in mlinestr:
            if line.is_empty:
                ranges.append(P.empty())
            else:
                xy = line.coords.xy
                ranges.append(P.closed(xy[0][0], xy[0][1]))
        
        return reduce(lambda x, y: x | y, ranges)
    
    def minkowski_operations(self, theta):
        """
        Do minkowski operations and get minkowski polygons for all objects (and the arena)
        :param theta: the additional rotation angle (anti-clockwise) of the object
        :return: mink_polys, MultiPolygon
        """
        mink_polys = []
        obj, _, _ = centering_polygon(affinity.rotate(self.X.obs.obj, theta, origin='center', use_radians=True))
        for i in range(self.X.obs.obs_num):
            obsi, obsi_xoff, obsi_yoff = centering_polygon(self.X.obs.obs[i])
            mink_sum = self.minkowski_operation_polygons(obsi, obj, sum=False)
            mink_sum = affinity.translate(mink_sum, obsi_xoff, obsi_yoff)
            mink_polys.append(mink_sum)
            
        return MultiPolygon(mink_polys)
    
    def sweep_line_process(self, mink_polys:MultiPolygon, n_line):
        """
        Do sweep line operations to get the C_free (approximation)
        :param mink_polys: the minkowski sum of object and all the obstacles
        :param n_line: the number of sweep lines
        :return: C_free, the dict of {y_coord: [range1, range2, ..., rangek]}
                 ranges are in form (l, u)
        """
        C_free = {}
    
        # slice the y dimension
        xmin, xmax = self.X.X_dimensions[0]
        ymin, ymax = self.X.X_dimensions[1]
        yslice = np.linspace(ymin, ymax, n_line+2)[1:-1]  # remove the top and bottom lines
        
        for i in range(n_line):
            y0 = yslice[i]
            sweep_line = LineString([(xmin, y0), (xmax, y0)])
            line_coll = [intersection(sweep_line, poly) for poly in list(mink_polys.geoms)]
            line_free = P.open(xmin, xmax) - self.convert_mlinestring_to_ranges(line_coll)
            C_free[y0] = line_free
        
        return C_free
    
    def get_range_middle_point(self, range):
        """
        Get the middle point of a range
        :param range: (l, u)
        :return: middle point
        """
        return (range.lower + range.upper) / 2.
    
    def get_range_nearest_point(self, target, range):
        """
        Get the point in range that is the nearest to target
        :param target: the target point
        :param range: (l, u)
        :return: nearest point
        """
        if target <= range.lower:
            return range.lower
        elif target >= range.upper:
            return range.upper
        else:
            return target
        
    def add_node_to_graph(self, x, y, theta, line_index, line_num_dict):
        """
        Add new node to graph
        :param x: x position
        :param x: y position
        :param theta: theta value
        :param line_index: the index of sweep line on the current C layer
        :param line_num_dict: the dict of number of vertex already on the sweep lines
        """
        # update the line_num_dict
        if line_index not in line_num_dict:
            line_num_dict[line_index] = 1
        else:
            line_num_dict[line_index] += 1
        # get the indexes and node name
        theta_index = self.theta.index(theta)
        node_index = line_num_dict[line_index]-1
        node_name = 'theta{0}_line{1}_node{2}'.format(theta_index, line_index, node_index)
        # add the node to Graph index
        if theta_index not in self.G_index:
            self.G_index[theta_index] = {}
        if line_index not in self.G_index[theta_index]:
            self.G_index[theta_index][line_index] = []
        self.G_index[theta_index][line_index].append(node_index)
        self.G.add_node(node_name, loc=(x, y, theta+self.X.obs.obj_theta0))
    
    def generate_and_add_vertex(self, C_free, theta):
        """
        Generate basic and enhancement vertex in C_free, and add the vertex to Graph
        :param C_free: the dict {y_coord: [range1, range2, ..., rangek]} obtained from sweep line process
        :param theta: the additional rotation angle (anti-clockwise) of the object
        """
        N = len(C_free)  # n_line
        line_vertex_num_dict = {}  # store the number vertex per sweep line
        for idx, (yi, range_free) in enumerate(C_free.items()):
            if idx < (N - 1):
                # get the y coord and C_free range of the next line
                yi_next = list(C_free.keys())[idx+1]
                range_free_next = C_free[yi_next]
            else:
                # the top line has no next line
                yi_next = None
                range_free_next = P.empty()
            # traverse all atomic range on line idx
            for range_i in range_free:
                # add the basic vertex (middle point)
                middle_i = self.get_range_middle_point(range_i)
                self.add_node_to_graph(x=middle_i, y=yi, theta=theta, line_index=idx, line_num_dict=line_vertex_num_dict)
                # add more vertex
                for range_i_next in range_free_next:
                    if not range_i.overlaps(range_i_next):
                        continue
                    middle_i_next = self.get_range_middle_point(range_i_next)
                    overlap = range_i.intersection(range_i_next)
                    if not middle_i in overlap:
                        new_point_i = self.get_range_nearest_point(middle_i, overlap)
                        self.add_node_to_graph(x=new_point_i, y=yi, theta=theta, line_index=idx, line_num_dict=line_vertex_num_dict)
                    if not middle_i_next in overlap:
                        new_point_i_next = self.get_range_nearest_point(middle_i_next, overlap)
                        self.add_node_to_graph(x=new_point_i_next, y=yi_next, theta=theta, line_index=idx+1, line_num_dict=line_vertex_num_dict)
    
    def get_sorted_subgraph(self, theta):
        """
        Get subgraph and sort the lines by line index
        :param theta: the additional rotation angle (anti-clockwise) of the object
        :return: subgraph
        """
        theta_index = self.theta.index(theta)
        subgraph = sorted(self.G_index[theta_index].items(), key=lambda d: d[0], reverse=False)
        
        return subgraph
    
    def get_vertex_on_line(self, theta, subgraph, line_index):
        """
        Get vertex on line
        :param theta: the additional rotation angle (anti-clockwise) of the object
        :param subgraph: the subgraph
        :param line_index: the line index
        :return: vertex, the list of tuple [(node_name, node_position)]
        """
        theta_index = self.theta.index(theta)
        vertex = subgraph[line_index][1]
        node_name = ['theta{0}_line{1}_node{2}'.format(theta_index, line_index, node_index) for node_index in vertex]
        vertex = sorted(dict(zip(node_name, [self.G.nodes[name]['loc'] for name in node_name])).items(), key=lambda d: d[1][0], reverse=False)
        
        return vertex
    
    def add_edge_to_graph(self, node1, node2):
        """
        Add new edge to graph
        :param node1: node1 name
        :param node2: node2 name
        :return: if true, the edge is added successfully; if false, the edge is not added due to collision
        """
        pos1, pos2 = self.G.nodes[node1]['loc'], self.G.nodes[node2]['loc']
        # # too slow
        # if self.X.collision_free(pos1, pos2, r=self.r):
        # # much faster
        if self.X.obs.translation_collision_free(pos1, pos2, self.X.obs.obj_shape):
        # if True:
            dist = planar_dist_between_points(pos1, pos2, self.w)
            self.G.add_edge(node1, node2, length=dist)
            return True
        else:
            return False
    
    def generate_and_add_edge(self, theta):
        """
        Connect vertex in the same C layer
        :param theta: the additional rotation angle (anti-clockwise) of the object
        """
        subgraph = self.get_sorted_subgraph(theta)
        for i in range(len(subgraph)):
            # connect vertex on the same line
            node_i = self.get_vertex_on_line(theta, subgraph, i)
            for j in range(len(node_i)):
                if j < len(node_i) - 1:
                    self.add_edge_to_graph(node_i[j][0], node_i[j+1][0])
            # connect vertex on adjacent lines
            if i < len(subgraph) - 1:
                node_i_next = self.get_vertex_on_line(theta, subgraph, i+1)
                for (node_i_j, node_i_next_j) in product(node_i, node_i_next):
                    self.add_edge_to_graph(node_i_j[0], node_i_next_j[0])
    
    def decompose_single_C_layer(self, theta):
        """
        Do cell decomposition on a single C-layer
        :param theta: the additional rotation angle (anti-clockwise) of the object
        :return: C_free, the dict {y_coord: [range1, range2, ..., rangek]} obtained from sweep line process
        """
        # # COMPUTATION THROUGH LINEAR PROGRAMMING
        # poly_mat = self.minkowski_operations(theta)
        # C_free = self.sweep_line_process(poly_mat, n_line)
        
        # # COMPUTATION THROUGH DIRECT MINKOWSKI OPERATION
        mink_polys = self.minkowski_operations(theta)
        C_free = self.sweep_line_process(mink_polys, self.n_line)
        
        self.generate_and_add_vertex(C_free, theta)
        self.generate_and_add_edge(theta)
        
        return C_free


if __name__ == '__main__':
    X_dimensions = np.array([(-0.1, 0.6), (-0.1, 0.6), (-np.pi, np.pi)])
    O_file = '../search_space/data/debug_obs8.npy'
    O_index = 2
    debug_obs = np.load(O_file)
    debug_shape = [[0.07, 0.12] for i in range(len(debug_obs))]
    X = PlanarSearchSpace(X_dimensions, None, O_file, O_index)
    X.create_slider_geometry(geom=[0.07, 0.12])
    X.create_slider_dynamics(ratio = 1 / 726.136, miu=0.2)
    
    # test parse polygons
    MS = MinkowskiSum(X, r=0.001, w=[1.0, 1.0, 0.0695*100], n_slice=60, n_line=35)
    # poly_mat = MS.parse_polygons(theta=0.0)
    
    # test Minkowski sum
    """
    A1 = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    b1 = np.array([[0.035], [0.06], [0.035], [0.06]])
    A2 = np.array([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]])
    b2 = np.array([[0.01], [0.01], [0.01], [0.01]])
    
    mksum0 = MS.get_minkowski_proj_interval(A1, A2, b1, b2, y0=0., sum=True)
    mksum1 = MS.get_minkowski_proj_interval(A1, A2, b1, b2, y0=0.06, sum=True)
    mksum2 = MS.get_minkowski_proj_interval(A1, A2, b1, b2, y0=0.065, sum=True)
    mksum3 = MS.get_minkowski_proj_interval(A1, A2, b1, b2, y0=0.075, sum=True)
    """
    # triangle = Polygon([(-4,2),(-3,1),(-2,2),(-4,2)])
    # rectangle = Polygon([(2,3),(2,1),(4,1),(4,3),(2,3)])
    # mink_sum = MS.minkowski_operation_polygons(rectangle, triangle, sum=False)
    # import pdb; pdb.set_trace()
    
    # poly_mat = MS.minkowski_operations(theta=0.0)
    # mink_polys = MS.minkowski_operations(theta=0.0)
    # import pdb; pdb.set_trace()
    
    # test sweep line process
    # C_free = MS.sweep_line_process(poly_mat, n_line=35)
    # C_free = MS.sweep_line_process(mink_polys, n_line=35)
    # import pdb; pdb.set_trace()
    
    # # test generate vertex
    # MS.generate_and_add_vertex(C_free, theta=0.0)
    # import pdb; pdb.set_trace()
    
    # # test generate edge
    # MS.generate_and_add_edge(theta=0.0)
    # import pdb; pdb.set_trace()
    
    # # test constructing one slice
    C_free = MS.decompose_single_C_layer(theta=0.0)
    
    # plot debug
    from matplotlib import pyplot as plt
    O_plot = np.delete(debug_obs, O_index, axis=0)
    plot = rrt_pack.pl_plot.PlanarPlot(filename=None, X=X)
    ax = plot.plot_debug(O_plot, delay_show=True)
    # plot C_free
    for y, free_range in C_free.items():
        for range in free_range:
            if range.empty: continue
            xl, xu = range.lower, range.upper
            ax.plot([xl, xu], [y, y], linestyle='-', c='seagreen')
    # plot vertex
    for theta_index in MS.G_index:
        for line_index in MS.G_index[theta_index]:
            for node_index in MS.G_index[theta_index][line_index]:
                node_name = 'theta{0}_line{1}_node{2}'.format(theta_index, line_index, node_index)
                node_loc = MS.G.nodes[node_name]['loc']
                plt.scatter(node_loc[0], node_loc[1], s=8, color='red')
    # plot edges
    for node1, node2 in list(MS.G.edges()):
        node_loc1, node_loc2 = MS.G.nodes[node1]['loc'], MS.G.nodes[node2]['loc']
        ax.plot([node_loc1[0], node_loc2[0]], [node_loc1[1], node_loc2[1]], linestyle='-', linewidth=2, c='seagreen')
    
    plt.show()
