import random

import kdtree
import networkx as nx
import numpy as np
import random
from shapely import contains, intersection
from shapely.geometry import Point, LineString

from rrt_pack.search_space.planar_search_space import PlanarSearchSpace


class PlanarRRGBase(object):
    def __init__(self, X:PlanarSearchSpace, x_init, rmin, vmin, nmax, p_rej, knn, npath):
        """
        Template RRG planner
        :param X: Search Space
        :param x_init: the init node position
        :param rmin: minimum radius
        :param vmin: minimum volume
        :param nmax: maximum number of nodes
        :param p_rej: the probability of rejecting a point that fall out of the convex hull of objects
        :param knn: max number of nearest neighbors to be searched
        :param npath: the number of paths to be returned
        """
        self.X = X
        self.G = nx.Graph()
        self.tree = kdtree.create(dimensions=len(x_init))
        self.node_idx = {}
        
        self.x_init = x_init
        self.rmin = rmin
        self.vmin = vmin
        self.nmax = nmax
        self.p_rej = p_rej
        self.npath = npath
        self.knn = knn
        
    def radius_search(self, x):
        """
        Return the safety radius of x
        :param x: tuple(x, y) which represents the location
        :return: the safety radius
        """
        return self.X.radius_search(x)
    
    def nearest_neighbor(self, x):
        """
        Return the nearest neighbor of x in G
        :param x: tuple(x, y) which represents the location
        :return: the index and distance of nearest neighbor
        """
        node_nn = self.tree.search_nn(x)
        if node_nn is None:
            return None, 0.
        loc_nn = node_nn[0].data
        dist_nn = np.sqrt(node_nn[1])
        return self.node_idx[loc_nn], dist_nn
    
    def overlap_volume(self, n1, n2, d):
        """
        Return the intersection area of safety spheres of n1 and n2
        :param n1, n2: two node indexed in G
        :param d: the distance between n1 and n2
        :return: the overlap volume
        """
        node1, node2 = self.G.nodes[n1], self.G.nodes[n2]
        r1, r2 = node1['r'], node2['r']
        
        if d >= r1 + r2:
            return 0.
        
        r1, r2 = min(r1, r2), max(r1, r2)
        
        if r2 - r1 >= d:
            return np.pi * r1 ** 2
        
        ang1 = np.arccos((r1 ** 2 + d ** 2 - r2 ** 2) / (2 * r1 * d))
        ang2 = np.arccos((r2 ** 2 + d ** 2 - r1 ** 2) / (2 * r2 * d))
        
        return ang1 * r1 ** 2 + ang2 * r2 ** 2 - r1 * d * np.sin(ang1)
        
    def add_to_graph(self, x_new, x_nearest):
        """
        Add new node x_new to graph
        :param x_new, x_nearest: x_new, x_nearest positions
        :return: if x_new was added successfully
        """
        if type(x_new) is not tuple:
            x_new = tuple(x_new)
        # add node to RRG
        x_new_idx = self.node_idx[x_new] = self.G.number_of_nodes()
        r = self.radius_search(x_new)
        if r >= self.rmin:
            self.G.add_node(node_for_adding=x_new_idx, c=x_new, r=r)
            # add node to kdtree
            self.tree.add(x_new)
            if not self.tree.is_balanced:
                self.tree = self.tree.rebalance()
            # connect node to nn in RRG
            if x_nearest != None:
                x_nearest_idx = self.node_idx[x_nearest]
                self.G.add_edge(x_nearest_idx, x_new_idx, length=np.linalg.norm(np.array(x_new) - np.array(x_nearest), ord=2))
            return True
        return False
    
    def connect_two_nodes(self, c1, c2, d):
        """
        Connect two nodes c1 and c2 in G
        :param c1, c2: two node positions in G
        :param d: the distance between c1 and c2
        """
        n1, n2 = self.node_idx[c1], self.node_idx[c2]
        self.G.add_edge(n1, n2, length=d)
            
    def rand_and_near(self):
        """
        Get a random collision free position and its nearest neighbor
        :return: x_rand, x_nearest
        """
        while True:
            x_rand = self.X.sample_free_2d()
            x_near_idx, _ = self.nearest_neighbor(x_rand)
            x_nearest = self.G.nodes[x_near_idx]['c']
            if self.X.collision_free_2d(x_rand, x_nearest):
                return x_rand, x_nearest
            
    def intersection(self, x_rand, x_nearest):
        """
        Return the intersection of ray(x_nearest, x_rand) and the safety sphere of x_nearest
        :param x_rand: x_rand position
        :param x_nearest: x_nearest position
        :return x_new: x_new (intersection point) position
        """
        r = self.G.nodes[self.node_idx[x_nearest]]['r']
        x_rand, x_nearest = np.array(x_rand), np.array(x_nearest)
        d = np.linalg.norm(x_nearest - x_rand, ord=2)
        return tuple(x_nearest + (r / d) * (x_rand - x_nearest))
    
    def connect_to_neighbors(self, x_new):
        """
        Connect x_new to nearest neighbors is the overlap volume is sufficiently large
        :param x_new: x_new position
        """
        x_new_idx = self.node_idx[x_new]
        k_nearest = self.tree.search_knn(x_new, k=self.knn)
        for x_near_node in k_nearest:
            x_near, d_near = x_near_node[0].data, np.sqrt(x_near_node[1])
            x_near_idx = self.node_idx[x_near]
            v_overlap = self.overlap_volume(x_new_idx, x_near_idx, d_near)
            if v_overlap >= self.vmin:
                self.connect_two_nodes(x_new, x_near, d_near)
                
    def reject_new_sample(self, x_new):
        """
        Return true is x_new is rejected, return false otherwise.
        :param x_new: x_new position
        """
        if not contains(self.X.obs.obs_conv_hull, Point(x_new)):
            if np.random.uniform(0, 1) < self.p_rej:
                return True
        # refuse samples that fall out of the workspace
        if not self.X.check_in_range_2d(x_new):
            return True
        return False
                
    def compute_shortest_paths(self):
        """
        Compute the single source shortest path on G
        :return: distance, path
        """
        return nx.single_source_dijkstra(self.G, 0, weight='length')
    
    def get_target_nodes(self):
        """
        Get all the nodes outside the convex hull of obstacles
        :return: list of node indexes
        """
        x_target = []
        for i in range(self.G.number_of_nodes()):
            c = self.G.nodes[i]['c']
            if not contains(self.X.obs.obs_conv_hull, Point(c)):
                x_target.append(i)
        return x_target
    
    def rrg_build(self):
        """
        Build the RRG
        :return: false if the problem is infeasible (x_init not collision free)
                 true if the problem is feasible (x_init collision free)
        """
        success = self.add_to_graph(self.x_init, None)
        if not success:
            return False
        
        while True:
            x_rand, x_nearest = self.rand_and_near()
            x_new = self.intersection(x_rand, x_nearest)
            
            if self.reject_new_sample(x_new):
                continue
            
            if (self.add_to_graph(x_new, x_nearest)):
                self.connect_to_neighbors(x_new)
                if self.G.number_of_nodes() >= self.nmax:
                    return True
    
    def rrg_path(self):
        """
        Get the shortest path from source point
        :return: paths, the dictionary of {node_idx:(distance, path)}
        """
        dist_shortest, path_shortest = self.compute_shortest_paths()
        x_target_idx = self.get_target_nodes()
        
        for idx in dist_shortest.keys():
            if (idx not in x_target_idx) or (idx == 0):
                dist_shortest[idx] = np.inf

        dist_shortest = sorted(dist_shortest.items(), key=lambda d:d[1])
        paths = {}
        for i in range(min(len(dist_shortest), self.npath)):
            node_idx = dist_shortest[i][0]
            paths[node_idx] = (dist_shortest[i][1], path_shortest[node_idx])
            
        return paths
            

if __name__ == '__main__':
    X_dimensions = np.array([(-0.1, 0.6), (-0.1, 0.6), (-np.pi, np.pi)])
    O_file = '../search_space/data/debug_obs8.npy'
    O_index = 2
    N_downsample = 200  # the number of downsampled route points
    Obstacles = np.load(O_file)
    rmin = 0.07 / 2
    vmin = 0.5 * 0.12 * 0.07
    
    x_init = tuple(Obstacles[O_index, 0:2])  # starting location
    X = PlanarSearchSpace(X_dimensions, O=None, O_file=O_file, O_index=O_index)
    X.create_slider_geometry(geom=[0.07, 0.12])
    rrg = PlanarRRGBase(X=X, x_init=x_init, rmin=rmin, vmin=vmin, nmax=200, p_rej=0.9, knn=3, npath=10)
    
    rrg.rrg_build()
    
    paths = rrg.rrg_path()
    
    from rrt_pack.utilities.planar_plotting import PlanarPlot
    from matplotlib import pyplot as plt
    
    # plot obstacles
    plot = PlanarPlot(filename='rrt_planar_pushing', X=X)
    Obstacles = np.delete(Obstacles, obj=O_index, axis=0)
    ax = plot.plot_debug(Obstacles, delay_show=True)
    
    # plot RRG
    for node_key in rrg.G.nodes():
        node = rrg.G.nodes[node_key]
        c, r = node['c'], node['r']
        ax.scatter(c[0], c[1], alpha=0.2)
        ax.add_artist(plt.Circle(c, r, fill=False, alpha=0.2))
    
    # plot shortest paths on RRG
    route_coords = []
    for x_target_idx, (dist, route_node_list) in paths.items():
        x_nodes = []
        for node_idx in route_node_list:
            x_nodes.append(rrg.G.nodes[node_idx]['c'])
        x_nodes = np.array(x_nodes)
        # plt.plot(x_nodes[:, 0], x_nodes[:, 1], marker='o', markersize=2, linestyle='-')
        
        # get the interpolated positions on path
        path_coords = []
        route_line = LineString(x_nodes)
        for l in np.linspace(0, route_line.length, N_downsample):
            xy = route_line.interpolate(l).coords.xy
            path_coords.append([xy[0][0], xy[1][0]])
        path_coords = np.array(path_coords)
        path_coords = np.c_[path_coords, np.zeros(N_downsample,)]
        # if x_target_idx == list(paths.keys())[0]:
        #     np.save('../search_space/data/central_path3.npy', route_coords) 
        plt.plot(path_coords[:, 0], path_coords[:, 1])
        route_coords += path_coords.tolist()
    np.save('../search_space/data/central_path3.npy', np.array(route_coords)) 
        
    plt.show()
    
    
