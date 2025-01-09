#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Class to generate a network to simulate random walking"""

class network:
    
    """

    """

    def __init__(self, dim = (10, 10), dilution = 0.):
        """
        Initialize the network and diameter.
        Args:
            dim (dimensions): Tuple specifying the size of the lattice in each dimension (d1, d2, ..., dN).
            
        Returns:
            None
        """
        self.dim = dim 
        self.dilution = dilution
        self.adj = self.nd_lattice_adjacency()
        self.dia = self.get_diameter_node()
        print(f"Initialize network of {self.dim} dimensions")


    def nd_lattice_adjacency(self):
        """
        Generate the adjacency matrix for an N-dimensional lattice.
        Args:
            None    
        Returns:
            A: Adjacency matrix of the lattice as a 2D numpy array.
        """
        T = np.prod(self.dim)  # Total number of nodes
        A = np.zeros((T, T), dtype=int)
        offsets = np.cumprod((1,) + self.dim[:-1])  # Multipliers for index calculation
    
        # Generate all coordinates in the lattice
        ranges = [range(d) for d in self.dim]
        coordinates = np.array(np.meshgrid(*ranges, indexing="ij")).T.reshape(-1, len(self.dim))
    
        # For each node, find its neighbors
        for node_idx, coord in enumerate(coordinates):
            for axis in range(len(self.dim)):
                for direction in [-1, 1]:
                    neighbor = coord.copy()
                    neighbor[axis] += direction
                    # Check bounds
                    if 0 <= neighbor[axis] < self.dim[axis]:
                        neighbor_idx = np.dot(neighbor, offsets)
                        A[node_idx, neighbor_idx] = np.random.binomial(1, 1 - self.dilution)
    
        return A

    """
    use only if the high nD network
    """
    # def bfs_farthest_node(self, adj, start_node):
    #     """
    #     Perform BFS to find the farthest node from the given start node.
    #     Args:
    #         adj (np.ndarray): Adjacency matrix of the graph.
    #         start_node (int): Index of the starting node.
    #     Returns:
    #         farthest_node (int): Index of the farthest node from the start.
    #         distances (np.ndarray): Array of shortest distances from start_node to all other nodes.
    #     """
    #     n = adj.shape[0]
    #     distances = -1 * np.ones(n, dtype=int)  # Initialize distances as -1 (unvisited)
    #     distances[start_node] = 0
    #     queue = deque([start_node])
        
    #     while queue:
    #         current = queue.popleft()
    #         for neighbor, connected in enumerate(adj[current]):
    #             if connected and distances[neighbor] == -1:  # If connected and not visited
    #                 distances[neighbor] = distances[current] + 1
    #                 queue.append(neighbor)
        
    #     farthest_node = np.argmax(distances)  # Node with the maximum distance
    #     return farthest_node, distances

    # def get_diameter_node(self):
    #     """
    #     Finds the diameter and coordinates in the nD lattice of the given network.
    #     Args:
    #         None
    #     Returns:
    #         dia (tuple): Tuple containing coordinates of starting and ending nodes of the diameter.
    #     """
    #     total_nodes = np.prod(self.dim)
    #     adj = self.adj
        
    #     # Step 1: Start from an arbitrary node (e.g., node 0)
    #     start_node = 0
    #     farthest_node_1, _ = self.bfs_farthest_node(adj, start_node)
        
    #     # Step 2: Find the farthest node from `farthest_node_1`
    #     farthest_node_2, distances = self.bfs_farthest_node(adj, farthest_node_1)
    #     diameter_length = distances[farthest_node_2]  # Diameter of the graph
        
    #     # Step 3: Convert linear indices to N-dimensional coordinates
    #     offsets = np.cumprod((1,) + self.dim[:-1])
    #     coord_1 = np.unravel_index(farthest_node_1, self.dim)
    #     coord_2 = np.unravel_index(farthest_node_2, self.dim)
        
    #     print(f"Graph Diameter: {diameter_length}")
    #     return (coord_1, coord_2)


    def get_diameter_node(self):
        """
        Finds the diameter and coordinates in the nD lattice of the given network.
        
        Returns:
            dia: Diameter of the network.
            start_coord: Coordinates of the starting node.
            end_coord: Coordinates of the ending node.
        """
        import numpy as np
        from scipy.sparse.csgraph import shortest_path
    
        # Reconstruct coordinates using the same logic as nd_lattice_adjacency
        ranges = [range(d) for d in self.dim]
        coordinates = np.array(np.meshgrid(*ranges, indexing="ij")).T.reshape(-1, len(self.dim))
    
        # Calculate shortest paths using Floyd-Warshall
        dist_matrix = shortest_path(
            self.adj,
            method="FW",
            directed=False,
            unweighted=True
        )
        
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
        start_coord = coordinates[start_idx]
        end_coord = coordinates[end_idx]
    
        print(f"Diameter: {diameter}")
        return int(diameter), start_coord, end_coord
