#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Class to generate a network to simulate random walking"""

import numpy as np
from scipy.sparse.csgraph import shortest_path


class Network:

    def __init__(self, dim=(10, 10), dilution=0.0, seed=None):
        """
        Initialize the network and diameter.
        Args:
            dim (tuple): Dimensions of the lattice (e.g., (10, 10) for a 2D lattice).
            dilution (float): Probability of dropping edges (0.0 means no dilution).
            seed (int, optional): Random seed for reproducibility.
        """
        self.dim = dim
        self.dilution = dilution
        self.seed = seed
        self.coordinates = self._generate_coordinates()
        self.adj = self.nd_lattice()

        if np.prod(self.dim) > 1000:
            self.dia = self.get_diameter_node_via_bfs()
            print("using bfs to assign some  distinct initial and target nodes not necessarly diameter")

        else:
            self.dia = self.get_diameter_node()
            
        print(f"Initialized network with dimensions {self.dim}")

    def _generate_coordinates(self):
        """Generate coordinates for all nodes in the lattice."""
        ranges = [range(d) for d in self.dim]
        return np.array(np.meshgrid(*ranges, indexing="ij")).T.reshape(-1, len(self.dim))

    def get_node_index(self, coordinates):
        """Convert node coordinates to index."""
        offsets = np.cumprod((1,) + self.dim[:-1])
        return np.dot(coordinates, offsets)

    def get_node_coordinates(self, index):
        """Convert node index to coordinates."""
        return self.coordinates[index]

    def nd_lattice(self):
        """
        Generate adjacency matrix for an N-dimensional lattice with optional dilution.
        
        Returns:
            A: Symmetric adjacency matrix (NxN) as a 2D numpy array.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
    
        T = np.prod(self.dim)  # Total number of nodes
        A = np.zeros((T, T), dtype=int)
    
        for node_idx, coord in enumerate(self.coordinates):
            for axis in range(len(self.dim)):
                # Only traverse in the positive direction to avoid double processing
                neighbor = coord.copy()
                neighbor[axis] += 1
                
                # Check bounds
                if neighbor[axis] < self.dim[axis]:  # Positive direction within bounds
                    neighbor_idx = self.get_node_index(neighbor)
                    
                    # Apply dilution: keep edge with probability (1 - dilution)
                    if np.random.rand() > self.dilution:
                        A[node_idx, neighbor_idx] = 1
                        A[neighbor_idx, node_idx] = 1  # Ensure symmetry
    
        return A



    def get_diameter_node(self):
        """
        Finds the diameter and coordinates in the nD lattice of the given network.
        
        Returns:
            dia: Diameter of the network.
            start_coord: Coordinates of the starting node.
            end_coord: Coordinates of the ending node.
        """
        # Calculate shortest paths using Floyd-Warshall
        dist_matrix = shortest_path(self.adj, method="FW", directed=False, unweighted=True)

        # Handle disconnected components
        dist_matrix[~np.isfinite(dist_matrix)] = -1

        # Find the maximum finite distance (diameter) and its endpoints
        diameter = -1
        start_idx, end_idx = 0, 0

        n = len(self.adj)
        for i in range(n):
            for j in range(i + 1, n):  # Only look at upper triangle
                if dist_matrix[i, j] > diameter:
                    diameter = dist_matrix[i, j]
                    start_idx, end_idx = i, j

        # Get the coordinates for the diameter endpoints
        start_coord = self.coordinates[start_idx]
        end_coord = self.coordinates[end_idx]

        print(f"Diameter: {diameter}")
        return int(diameter), start_coord, end_coord

    
    def bfs_farthest_node(self, adj, start_node):
    
        """
        Perform BFS to find the farthest node from the given start node.
        Args:
            adj (np.ndarray): Adjacency matrix of the graph.
            start_node (int): Index of the starting node.
        Returns:
            farthest_node (int): Index of the farthest node from the start.
            distances (np.ndarray): Array of shortest distances from start_node to all other nodes.
        """
        n = adj.shape[0]
        distances = -1 * np.ones(n, dtype=int)  # Initialize distances as -1 (unvisited)
        distances[start_node] = 0
        queue = deque([start_node])
        
        while queue:
            current = queue.popleft()
            for neighbor, connected in enumerate(adj[current]):
                if connected and distances[neighbor] == -1:  # If connected and not visited
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        
        farthest_node = np.argmax(distances)  # Node with the maximum distance
        return farthest_node, distances
    
    def get_diameter_node_via_bfs(self):
        """
        Finds the diameter and coordinates in the nD lattice of the given network.
        Args:
            None
        Returns:
            dia (tuple): Tuple containing coordinates of starting and ending nodes of the diameter.
        """
        total_nodes = np.prod(self.dim)
        adj = self.adj
        
        # Step 1: Start from an arbitrary node (e.g., node 0)
        start_node = 0
        farthest_node_1, _ = self.bfs_farthest_node(adj, start_node)
        
        # Step 2: Find the farthest node from `farthest_node_1`
        farthest_node_2, distances = self.bfs_farthest_node(adj, farthest_node_1)
        diameter_length = distances[farthest_node_2]  # Diameter of the graph
        
        # Step 3: Convert linear indices to N-dimensional coordinates
        offsets = np.cumprod((1,) + self.dim[:-1])
        coord_1 = np.unravel_index(farthest_node_1, self.dim)
        coord_2 = np.unravel_index(farthest_node_2, self.dim)

        print(f"Diameter: {diameter_length}")
        return diameter_length, np.array(coord_1), np.array(coord_2)

    def adj_gen(self, adj, i, n):
    
        ## base case
        if i == n:
            return
    
        ## main
    
        ### creating a copy 
        adj[i] = np.zeros_like(adj[0])
    
        for j in range(len(adj[0])):
    
            ### curr node cordinate vector
            curr_node = self.get_node_coordinates(j)
            
            ### finding all neighbours
            neighbours = np.where(adj[i-1][j, :] == 1)[0]
    
            for k in neighbours:
    
                ### each connected neighbour cordinates
                conn_node = self.get_node_coordinates(k)

                ### normalization step for direction
                norm = np.linalg.norm((conn_node - curr_node))
                
    
                ### step node computation along certain axis

                step_node =  curr_node + (i+1) * ((conn_node - curr_node) / norm).astype(int)
                step_idx = self.get_node_index(step_node)
                
    
                ### check for bounds and check connection b/w conn_node and step node in adj[0]
                if (step_node >= 0).all() and (step_node < self.dim).all():

                    if adj[0][k][step_idx]:
                    
                        ### update curr adj 
                        adj[i][j][step_idx] = 1
                    
        self.adj_gen(adj, i+1, n)
