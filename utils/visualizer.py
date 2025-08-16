import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


class Visualize:

    def __init__(self, dim, adj_path, data_path, fixed_points, iteration=None):

        self.dim = dim
        self.dim_len = len(self.dim)
        self.adj = np.load(adj_path)

        self.iteration = iteration
        
        self.lattice_opacity = 0.5
        self.lattice_width = 0.4
        self.lattice_color = 'w'
        
        self.lattice_bg = 'k'

        self.grid_opacity = 0.2
        self.grid_width = 0.2
        self.grid_color = 'w'

        ## temporary point details
        self.point_opacity = 1
        self.point_width = 20
        self.point_color = 'skyblue'

        self.data, self.tensors_shape = self._load_data(data_path)
        self.tensors_cumsum = np.cumsum(list(self.tensors_shape.values()))

        ## handling user iteration input for visualization
        
        if self.iteration is not None:
            self.data_tensor =  self.data[self.iteration]
        else:

            all_tensors = [self.data[it] for it in sorted(self.data.keys())]
            self.data_tensor = np.concatenate(all_tensors, axis=1)
            self.iteration = 0

        self.init_coordinates = fixed_points #start and target
        self.fig, self.ax = self._lattice_gen()

        ## functionality attributes

        self.hold = False

    
    def animate(self):

        max_steps = self.data_tensor.shape[1]
        
        def update(frame):
            if self.dim_len == 2:
                simulation_ended = self.update_2d(frame)
                if simulation_ended:
                    return []
            elif self.dim_len == 3:
                simulation_ended = self.update_3d(frame)
                if simulation_ended:
                    return []
            
            return [self.current_scatter] if self.current_scatter else []
        
        anim = FuncAnimation(
            self.fig, 
            update, 
            frames=max_steps, 
            interval=350,  # 200ms between frames
            blit=True,
            repeat=False
        )

        return anim

    def update_2d(self, walker_step):
        
        #### Get current data
        curr_position_tensor = self.data_tensor[:, walker_step, :]

        if self.hold:

            time.sleep(1.0)

            self.hold = False


        if walker_step == self.tensors_cumsum[self.iteration] - 1:
 
            self.ax.set_title(f'Simulation Complete - Iteration: {self.iteration}', color='red', fontsize=7)
            self.iteration += 1
            self.hold = True

            if not np.any(np.isnan(curr_position_tensor)):
    
                coords = curr_position_tensor[:, 0], curr_position_tensor[:, 1]
                    
                if hasattr(self, 'current_scatter') and self.current_scatter is not None:
                    self.current_scatter.remove()
        
                # Add new scatter plot
                self.current_scatter = self.ax.scatter(
                    coords[0], coords[1],
                    s=self.point_width,
                    c=self.walker_colors,
                    alpha=self.point_opacity,
                    zorder=10
                )
                
                # Update title/info
                self.ax.set_title(f'Step {walker_step}', color='white', fontsize=7)
            
            return True


        #### nan in tensor denote the sim has ended
        if not np.any(np.isnan(curr_position_tensor)):
    
            coords = curr_position_tensor[:, 0], curr_position_tensor[:, 1]
                
            if hasattr(self, 'current_scatter') and self.current_scatter is not None:
                self.current_scatter.remove()
    
            # Add new scatter plot
            self.current_scatter = self.ax.scatter(
                coords[0], coords[1],
                s=self.point_width,
                c=self.walker_colors,
                alpha=self.point_opacity,
                zorder=10
            )
            
            # Update title/info
            self.ax.set_title(f'Step {walker_step}', color='white', fontsize=7)
            
            return False  # Continue simulation

            
    def update_3d(self, walker_step):
        
        curr_position_tensor = self.data_tensor[:, walker_step, :]
        
        if self.hold:
            time.sleep(1.0)
            self.hold = False

        if walker_step == self.tensors_cumsum[self.iteration] - 1:
            
            self.ax.set_title(f'Simulation Complete - Iteration: {self.iteration}', color='red', fontsize=7)
            self.iteration += 1
            self.hold = True

            if not np.any(np.isnan(curr_position_tensor)):
                coords = curr_position_tensor[:, 0], curr_position_tensor[:, 1], curr_position_tensor[:, 2]
    
                if hasattr(self, 'current_scatter') and self.current_scatter is not None:
                    self.current_scatter.remove()
            
                self.current_scatter = self.ax.scatter(
                    coords[0], coords[1], coords[2],
                    s=self.point_width,
                    c=self.walker_colors,
                    alpha=self.point_opacity,
                    depthshade=False,
                    zorder=10
                )
                    
                self.ax.set_title(f'Step {walker_step}', color='white', fontsize=7)

            
            return True
        
        if not np.any(np.isnan(curr_position_tensor)):
            coords = curr_position_tensor[:, 0], curr_position_tensor[:, 1], curr_position_tensor[:, 2]

            if hasattr(self, 'current_scatter') and self.current_scatter is not None:
                self.current_scatter.remove()
        
            self.current_scatter = self.ax.scatter(
                coords[0], coords[1], coords[2],
                s=self.point_width,
                c=self.walker_colors,
                alpha=self.point_opacity,
                depthshade=False,
                zorder=10
            )
                
            self.ax.set_title(f'Step {walker_step}', color='white', fontsize=7)
            return False

        
        



    # ## old data processing
    # def _load_data(self, data_path):
        
    #     df = pd.read_csv(data_path)

    #     coords = [c for c in ['x0', 'x1', 'x2'] if c in df.columns]
    #     iterations = np.sort(df['Iteration'].unique())
    #     tensors = {}
        
    #     for it in iterations:
    #         df_it = df[df['Iteration'] == it]
    #         ids = np.sort(df_it['Walker_ID'].unique())
    #         steps = np.sort(df_it['Step'].unique())
            
    #         id_map = {v: i for i, v in enumerate(ids)}
    #         step_map = {v: i for i, v in enumerate(steps)}
            
    #         tensor = np.full((len(ids), len(steps), len(coords)), np.nan)
            
    #         for _, row in df_it.iterrows():
    #             i = id_map[row['Walker_ID']]
    #             s = step_map[row['Step']]
    #             tensor[i, s, :] = [row[c] for c in coords]
            
    #         tensors[it] = tensor

    #     return tensors

    def _load_data(self, data_path):  ## faster and simple
        df = pd.read_csv(data_path)

        ### unique walker types 
        unique_types = df['Walker_Type'].unique()
        
        ### map from type to color
        colormap = plt.cm.get_cmap('Set1', len(unique_types))
        type_to_color_map = {agent_type: colormap(i) for i, agent_type in enumerate(unique_types)}
    
        ### Walker ID  -> assigned color
 
        id_to_type_map = df.drop_duplicates('Walker_ID').set_index('Walker_ID')['Walker_Type']
        
        ##### Now, map that type to a color for each ID
        id_to_color_map = id_to_type_map.map(type_to_color_map)
    
        #### sorted list of walker IDs. This order matches the tensor rows.
        self.sorted_walker_ids = sorted(df['Walker_ID'].unique())
    
        ##### Create the final, ordered list of colors.
        self.walker_colors = [id_to_color_map[id] for id in self.sorted_walker_ids]

        coords = [c for c in ['x0', 'x1', 'x2'] if c in df.columns]
            
        tensors = {}
        tensors_shape = {}
        
        for it, df_it in df.groupby('Iteration'):
            
            # Pivot the data for this specific iteration.
            pivoted = df_it.pivot_table(
                index='Walker_ID',
                columns='Step',
                values=coords
            )
            
            
            ## swap the levels steps <-> coordinates
            pivoted = pivoted.swaplevel(0, 1, axis=1)
            
            # simple sort steps together
            pivoted.sort_index(axis=1, inplace=True)
        
            num_walkers = df_it['Walker_ID'].nunique()
            num_steps = df_it['Step'].nunique()
            num_coords = len(coords)
            
            tensor = pivoted.values.reshape(num_walkers, num_steps, num_coords)
            
            tensors[it] = tensor
            tensors_shape[it] = tensor.shape[1]

        return tensors, tensors_shape

        
    def _lattice_gen(self):

        start_point = self.init_coordinates[0]
        target_point = self.init_coordinates[1]

        if self.dim_len == 2:
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=300)
            fig.set_facecolor(self.lattice_bg)
            ax.set_facecolor(self.lattice_bg)
            
            ## outer box based on x-ylim
            ax.set_xlim(-0.5, self.dim[0] - 0.5)
            ax.set_ylim(-0.5, self.dim[1] - 0.5)
            
            ax.grid(color=self.grid_color, linewidth=self.grid_width, alpha=self.grid_opacity, zorder=0)

            ## generate lattice
            width_dim, height_dim = self.dim
            total_nodes = width_dim * height_dim

            # Preallocate edge segments
            segments = []
            
            # Check horizontal edges
            for y in range(height_dim):
                for x in range(width_dim - 1):
                    idx1 = y * width_dim + x
                    idx2 = idx1 + 1
                    if self.adj[idx1, idx2] != 0:
                        segments.append([(x, y), (x + 1, y)])
            
            # Check vertical edges
            for y in range(height_dim - 1):
                for x in range(width_dim):
                    idx1 = y * width_dim + x
                    idx2 = idx1 + width_dim
                    if self.adj[idx1, idx2] != 0:
                        segments.append([(x, y), (x, y + 1)])
    
            # Render all edges in one operation
            lc = LineCollection(
                segments,
                linewidths=self.lattice_width,
                colors=self.lattice_color,
                alpha=self.lattice_opacity
            )
            ax.add_collection(lc)
            ax.autoscale_view()

            ax.scatter(
                x= start_point[0] ,
                y= start_point[1] ,
                s=self.point_width,          
                c='teal',    
                marker='*',  
                alpha=0.8,
                label='Start'
            )
            
            # Plot the TARGET point
            ax.scatter(
                x= target_point[0],
                y= target_point[1] ,
                s=self.point_width,
                c='silver',
                marker='8',  
                alpha=0.8,
                label='Target'
            )

            return fig, ax

        elif self.dim_len == 3:

            L, M, N = self.dim
            total_nodes = L * M * N
            
            fig = plt.figure(figsize=(10, 10), dpi=300)
            ax = plt.axes(projection='3d')
            
            fig.set_facecolor(self.lattice_bg)
            ax.set_facecolor(self.lattice_bg)
            
            ax.set_xlim(0, self.dim[0] - 1)
            ax.set_ylim(0, self.dim[1] - 1)
            ax.set_zlim(0, self.dim[2] - 1)
            
            ax.grid(False)
            ax.set_axis_off()
            
            
            # Add reference grid (complete wireframe)
            ref_segments = []
            for x in range(L):
                for y in range(M):
                    for z in range(N):
                        # X-direction edges
                        if x < L-1:
                            ref_segments.append([(x, y, z), (x+1, y, z)])
                        # Y-direction edges
                        if y < M-1:
                            ref_segments.append([(x, y, z), (x, y+1, z)])
                        # Z-direction edges
                        if z < N-1:
                            ref_segments.append([(x, y, z), (x, y, z+1)])
            
            ref_lc = Line3DCollection(
                ref_segments,
                linewidths=getattr(self, 'grid_width', 0.2),
                colors=getattr(self, 'grid_color', 'w'),
                alpha=getattr(self, 'grid_opacity', 0.1)
            )
            ax.add_collection3d(ref_lc)
            
            # Generate edge segments based on adj
            segments = []
            for x in range(L):
                for y in range(M):
                    for z in range(N):
                        idx = x+ y*N + z*(M*N) 
                        
                        # Right neighbor (x-direction)
                        if x < L-1 and self.adj[idx, idx + 1]:
                            segments.append([(x, y, z), (x+1, y, z)])
                        
                        # Up neighbor (y-direction)
                        if y < M-1 and self.adj[idx, idx + N]:
                            segments.append([(x, y, z), (x, y+1, z)])
                        
                        # Above neighbor (z-direction)
                        if z < N-1 and self.adj[idx, idx + M*N]:
                            segments.append([(x, y, z), (x, y, z+1)])
            
            # Draw edges

            lc = Line3DCollection(
                segments, 
                linewidths=self.lattice_width, 
                colors=self.lattice_color,
                alpha=self.lattice_opacity
            )
            
            ax.add_collection3d(lc)
            ax.scatter(start_point[0], start_point[1], start_point[2],
                s=self.point_width,          
                c='teal',    
                marker='*',  
                alpha=0.8,
                label='Start'
            )
            
            # Plot the TARGET point
            ax.scatter(target_point[0], target_point[1], target_point[2],
                s=self.point_width,
                c='silver',
                marker='8',  
                alpha=0.8,
                label='Target'
            )



            return fig, ax

                


if __name__ == "__main__":

    start = Visualize((7, 7, 7), "../results/adj_20250811_200645.npy", data_path="../results/positions_20250811_200645.csv")

    
    anim = start.animate()
    plt.show()
