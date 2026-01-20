from space_groups import SpaceGroup, PlaneGroup
from space_groups.utils import sympy_to_numpy
import numpy as np
import networkx as nx
from scipy.linalg import norm
import scipy.sparse as sp
import math

class SpaceGraph:
    def __init__(self, group, embedding_dim, points=None, decimals=5, k=2):
        ''' Construct graph on the reciprocal lattice for a given space group.
        Each subgraph corresponds to an eigenfunction in the symmetry-adapted 
        Fourier series.

        Args:
            group (int): Space group number
            embedding_dim (int): Minimum number of points to include
            points (list): Optional pre-computed points to use
            decimals (int): Number of decimals to round for floating point robustness
            k (float): Factor for logarithmic spacing between shells (default: 1.5)
        '''
        self.group = SpaceGroup(group)
        self.embedding_dim = embedding_dim
        self.decimals = decimals
        self.points = points
        self.k = k

        operations = sympy_to_numpy(self.group.operations)
        self.basis = sympy_to_numpy(self.group.basic_basis)
        self.basis_inv = np.linalg.inv(self.basis)
        self.basis_inv_T = np.linalg.inv(self.basis).T

        # Helper for rounding/canonicalizing points
        def canonical_point(pt):
            arr = np.asarray(pt, dtype=float)
            return tuple(np.round(arr, decimals=self.decimals))

        self.canonical_point = canonical_point

        # Change of basis for group operations
        A_list = [self.basis @ op[:3, :3] @ self.basis_inv for op in operations]
        t_list = [self.basis @ op[:3, 3] for op in operations]
        self.transforms = list(zip(A_list, t_list))
        self.graph = nx.Graph()
        self.frustrated_nodes = set()  # Nodes that are part of frustrated components
        self._construct_graph()
        
    def _construct_graph(self):
        """Construct the graph of points within a radius that gives us enough points."""
        print("Constructing graph for group ", self.group.number)
        if self.points is None:
            # Start with a small radius and increase until we have enough points
            radius = math.ceil((1.5 * self.embedding_dim / (4/3 * math.pi)) ** (1/3))
            points_set = set()
            
            # Get all integer points within radius (in the original basis)
            for i in range(-radius, radius+1):
                for j in range(-radius, radius+1):
                    for k in range(-radius, radius+1):
                        pt = np.array([i, j, k], dtype=float)
                        pt_new = self.basis_inv_T @ pt
                        if np.linalg.norm(pt_new) <= radius and not (i == 0 and j == 0 and k == 0):
                            points_set.add(tuple(pt_new))  # store as tuple, but NOT rounded yet
            # Convert to sorted list for consistency
            self.points = sorted(list(points_set))   # Note self.points are full-precision tuples

        # Add nodes using canonical (rounded) version
        self.graph.add_nodes_from([self.canonical_point(p) for p in self.points])

        # Add edges based on the group operations
        for point in self.points:
            pt = np.array(point).reshape(1, -1)
            src = self.canonical_point(point)
            for A, t in self.transforms:
                transformed = (pt @ A.T)[0]
                transformed_point = self.canonical_point(transformed)
                weight = np.exp(2j * np.pi * (pt @ A.T @ t)[0])
                self.graph.add_edge(src, transformed_point, weight=weight)
                self.graph.nodes[transformed_point]['weight'] = weight

                # Check if the transformed point forms frustrated self-loop
                if transformed_point == src and not np.isclose(weight.real, 1):
                    self.frustrated_nodes.add(src)
        self.graph.remove_nodes_from(self.frustrated_nodes)

    def get_adjacency_matrix(self):
        """Get the adjacency matrix of the graph, handling complex weights."""
        nodelist = self.points
        n = len(nodelist)
        node_to_index = {tuple(node): i for i, node in enumerate(nodelist)}
        
        row, col, data = [], [], []
        for u, v, attr in self.graph.edges(data=True):
            u_key = tuple(np.round(u, decimals=self.decimals))
            v_key = tuple(np.round(v, decimals=self.decimals))
            if u_key in node_to_index and v_key in node_to_index:
                i, j = node_to_index[u_key], node_to_index[v_key]
                weight = attr.get('weight', 1+0j)  # Ensure weight is complex
                if i != j:  # Only add off-diagonal elements twice
                    row.extend([i, j])
                    col.extend([j, i])
                    data.extend([weight, weight.conjugate()])
                else:  # For diagonal elements, add only once
                    row.append(i)
                    col.append(i)
                    data.append(1+0j)  
        
        row = np.array(row, dtype=int)
        col = np.array(col, dtype=int)
        data = np.array(data, dtype=complex)
        return sp.csr_array((data, (row, col)), shape=(n, n))
    
    def get_nodelist(self):
        return list(self.graph.nodes())
    
class WallpaperGraph:
    def __init__(self, group, embedding_dim, points=None, decimals=5):
        ''' Construct graph on the reciprocal lattice for a given wallpaper group.
        Each subgraph corresponds to an eigenfunction in the symmetry-adapted 
        Fourier series.

        Args:
            group (int): Wallpaper group number
            embedding_dim (int): Minimum number of points to include
            points (list): Optional pre-computed points to use
            decimals (int): Number of decimals to round for floating point robustness
        '''
        self.group = PlaneGroup(group)
        self.embedding_dim = embedding_dim
        self.decimals = decimals
        self.points = points

        operations = sympy_to_numpy(self.group.operations)
        self.basis = sympy_to_numpy(self.group.basic_basis)
        self.basis_inv = np.linalg.inv(self.basis)
        self.basis_inv_T = np.linalg.inv(self.basis).T

        # Helper for rounding/canonicalizing points
        def canonical_point(pt):
            arr = np.asarray(pt, dtype=float)
            return tuple(np.round(arr, decimals=self.decimals))

        self.canonical_point = canonical_point

        # Change of basis for group operations
        A_list = [self.basis @ op[:2, :2] @ self.basis_inv for op in operations]
        t_list = [self.basis @ op[:2, 2] for op in operations]
        self.transforms = list(zip(A_list, t_list))
        self.graph = nx.Graph()
        self.frustrated_nodes = set()  # Nodes that are part of frustrated components
        self._construct_graph()
        
    def _construct_graph(self):
        """Construct the graph of points within a radius that gives us enough points."""
        if self.points is None:
            # Start with a small radius and increase until we have enough points
            radius = math.ceil((1.5 * self.embedding_dim / math.pi) ** (1/2)) + 1
            points_set = set()
            
            # Get all integer points within radius (in the original [0,1],[1,0] basis)
            for i in range(-radius, radius+1):
                for j in range(-radius, radius+1):
                    if norm([i, j]) <= radius and not (i == 0 and j == 0):
                        # Transform to new basis: w' = B^{-T} w
                        pt = np.array([i, j], dtype=float)
                        pt_new = self.basis_inv_T @ pt
                        points_set.add(tuple(pt_new))  # store as tuple, but NOT rounded yet
            # Convert to sorted list for consistency
            self.points = sorted(list(points_set))   # Note self.points are full-precision tuples

        # Add nodes using canonical (rounded) version
        self.graph.add_nodes_from([self.canonical_point(p) for p in self.points])

        # Add edges based on the group operations
        for point in self.points:
            pt = np.array(point).reshape(1, -1)
            src = self.canonical_point(point)
            for A, t in self.transforms:
                transformed = (pt @ A.T)[0]
                transformed_point = self.canonical_point(transformed)
                weight = np.exp(2j * np.pi * (pt @ A.T @ t)[0])
                self.graph.add_edge(src, transformed_point, weight=weight)
                self.graph.nodes[transformed_point]['weight'] = weight

                # Check if the transformed point forms frustrated self-loop
                if transformed_point == src and not np.isclose(weight.real, 1):
                    self.frustrated_nodes.add(src)
        self.graph.remove_nodes_from(self.frustrated_nodes)

    def get_adjacency_matrix(self):
        """Get the adjacency matrix of the graph, handling complex weights."""
        nodelist = self.points
        n = len(nodelist)
        node_to_index = {tuple(node): i for i, node in enumerate(nodelist)}
        
        row, col, data = [], [], []
        for u, v, attr in self.graph.edges(data=True):
            u_key = tuple(np.round(u, decimals=self.decimals))
            v_key = tuple(np.round(v, decimals=self.decimals))
            if u_key in node_to_index and v_key in node_to_index:
                i, j = node_to_index[u_key], node_to_index[v_key]
                weight = attr.get('weight', 1+0j)  # Ensure weight is complex
                if i != j:  # Only add off-diagonal elements twice
                    row.extend([i, j])
                    col.extend([j, i])
                    data.extend([weight, weight.conjugate()])
                else:  # For diagonal elements, add only once
                    row.append(i)
                    col.append(i)
                    data.append(1+0j)  
        
        row = np.array(row, dtype=int)
        col = np.array(col, dtype=int)
        data = np.array(data, dtype=complex)
        return sp.csr_array((data, (row, col)), shape=(n, n))
    
    def get_nodelist(self):
        return list(self.graph.nodes())