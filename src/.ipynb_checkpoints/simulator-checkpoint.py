#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import csv
import os
import json
import secrets
from datetime import datetime

from .network import Network
from .walkers import Walkers
from utils.visualizer import Visualize

import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Simulation(Network):
    def __init__(self, dim, dilution, total_agents, seed=None, truely_random=False, pars=None):
        """
        Initialize the simulation class.
        Args:
            dim (tuple): Dimensions of the grid.
            dilution (float): Dilution factor for the grid.
            total_agents (int): Total number of agent types.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            truely_random (bool, optional): Use true randomness or not. Defaults to False.
            pars (dict, optional): Predefined parameters for agents. Defaults to None.
        """
        super().__init__(dim, dilution, seed)
        self.total_agents = total_agents
        self.agents = {}
        self.truely_random = truely_random

        if pars is not None:
            self.len_arr = self._load_pars(pars)
        else:
            self.len_arr = self._get_input()
            
        #> Generate required adjacency matrices for different step lengths
        self.adj_n = {0: self.adj}  #>> Initialize adjacency matrix for 1-step traversal
        max_step_length = np.max(self.len_arr)
        super().adj_gen(self.adj_n, 1, max_step_length)
        
        #> Generate diameter using the appropriate adjacency matrix
        if np.prod(self.dim) > 1000:
            self.dia = super().get_diameter_node_via_bfs(self.adj_n[max_step_length - 1])
            print("Using BFS to assign distinct initial and target nodes (not necessarily diameter)")
        else:
            self.dia = super().get_diameter_node(self.adj_n[max_step_length - 1])
        
        print(f"distance: start -> target : {self.dia}")

        self.init_agents()

        

    def _get_input(self):
        """Get user input for initializing agents."""
        step_l_arr = np.zeros(self.total_agents, dtype=int)
        self.agent_configs = []
        
        for i in range(self.total_agents):
            name = input("Agent name: ")
            n = int(input(f"Number of {name}: "))
            step = int(input(f"Number of steps for {name}: "))
            step_length = int(input(f"Step length for {name}: "))
            
            step_l_arr[i] = step_length
            self.agent_configs.append({
                'name': name,
                'n': n,
                'step': step,
                'step_length': step_length
            })
        
        return step_l_arr
    
    def _load_pars(self, pars):
        """loading_initializations"""
        step_l_arr = np.zeros(len(pars), dtype=int)
        self.agent_configs = []
        
        for i, (name, config) in enumerate(pars.items()):
            step_l_arr[i] = config["step_length"]
            self.agent_configs.append({
                'name': name,
                'n': config["n"],
                'step': config["step"],
                'step_length': config["step_length"]
            })
        
        return step_l_arr
    
    def init_agents(self):
        """Initialize agents from stored configurations."""
        start_id = 0
        
        for config in self.agent_configs:
            name = config['name']
            n = config['n']
            step = config['step']
            step_length = config['step_length']
            
            self.agents[name] = Walkers(
                self.dia, self.dim, n=n, step=step, 
                step_len=step_length, start_id=start_id
            )
            start_id += n
            
        print(f"Simulation initialized with {len(self.agent_configs)} agents.")

    def simulate(self, iterations=1, output_dir="results", log_paths=False, verbose=True):
        """
        Your original simple approach - just with better logging
        """
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(output_dir, f"results_{self.timestamp}.csv")
        self.positions_file = os.path.join(output_dir, f"positions_{self.timestamp}.csv") if log_paths else None
        self.meta_file = os.path.join(output_dir, f"meta_{self.timestamp}.json")
        self.adj_file = os.path.join(output_dir, f"adj_{self.timestamp}.npy")
        
        # Save metadata
        self._save_metadata(self.meta_file, iterations)

        # save adjacency mat
        np.save(self.adj_file, self.adj)
        
        # Open files
        summary_csv = open(summary_file, 'w', newline='')
        summary_writer = csv.writer(summary_csv)
        summary_writer.writerow(["Iteration", "Winner_Type", "Winner_ID", "Steps", "Distance"])
        
        positions_csv = None
        positions_writer = None
        walker_id = 0
        if log_paths:
            positions_csv = open(self.positions_file, 'w', newline='')
            positions_writer = csv.writer(positions_csv)
            nd = len(self.dim)
            positions_writer.writerow(["Iteration", "Walker_Type", "Walker_ID", "Step"] + [f"x{i}" for i in range(nd)])
        
        try:
            for iteration in range(iterations):
                if verbose:
                    print(f"Iteration {iteration + 1}/{iterations}")
                
                winner_found = False
                
                # Your original simulation loop - simple and effective
                while not winner_found:
                    for walker_name, walker in self.agents.items():
                        for i in range(walker.n):
                            
                            # Check for convergence first
                            if (walker.walker[i, 0] == walker.end).all():
                                if verbose:
                                    print(f"  Winner: {walker_name}[{i}] reached target after {walker.walker[i, 1]} steps, distance {walker.walker[i, 2]}")
                                
                                # Save winner data
                                summary_writer.writerow([iteration, walker_name, i, walker.walker[i, 1], walker.walker[i, 2]])
                                
                                winner_found = True
                                break
                            
                            # Move walker
                            old_pos = walker.walker[i, 0].copy()
                            walker.walker[i, 0] = self.sample_walker(walker.walker[i, 0], walker.step, walker.step_len)
                            walker.walker[i, 1] += walker.step
                            walker.walker[i, 2] += walker.step * walker.step_len
                            
                            # Log position if needed
                            if log_paths:
                                row = [iteration, walker_name, walker.walker[i, 3], walker.walker[i, 1]] + list(walker.walker[i, 0])
                                positions_writer.writerow(row)
                        
                        if winner_found:
                            break
                
                # Reset all walkers for next iteration
                for walker in self.agents.values():
                    walker.reset()
        
        finally:
            summary_csv.close()
            if positions_csv:
                positions_csv.close()
        
        if verbose:
            print(f"Results saved to {output_dir}")

    def _save_metadata(self, meta_file, iterations):
        """Save simulation metadata"""
        print(int(self.dia[0]), list(self.dia[1]), list(self.dia[2]))
        meta = {
            "timestamp": datetime.now().isoformat(),
            "dimensions": list(self.dim),
            "dilution": float(self.dilution),
            "iterations": iterations,
            "diameter" : self.dia[0],
            "start": self.dia[1].tolist(),
            "target" : self.dia[2].tolist(),
            
            "agents": {
                name: {
                    "count": w.n,
                    "steps_per_move": int(w.step),
                    "step_length": int(w.step_len)
                } for name, w in self.agents.items()
            }
            
        }
        
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)

    def sample_walker(self, curr, n_steps, step_l):
        """Your original walker movement - works perfectly"""
        
        for i in range(n_steps):
            # Get adj index
            idx = super().get_node_index(curr)
            
            # Find all degrees of freedom at idx
            freedom = np.where(self.adj_n[step_l - 1][idx] == 1)[0]
            
            if len(freedom) == 0:
                print(f"No valid moves for position {curr} at step length {step_l}.")
                return curr
            
            # Sample the freedom randomly
            if not self.truely_random:
                choice = np.random.choice(freedom)
            else:
                choice = secrets.choice(freedom)
            
            new_loc = super().get_node_coordinates(choice)
            curr = new_loc
        
        return new_loc

    def visualize_sim(self, iteration=None, save=False, Type='gif'):

        viz = Visualize(self.dim, self.adj_file, self.positions_file, self.dia[1:], iteration=iteration)

        anim = viz.animate()

        if save:

            if Type != 'gif':
                anim.save(f'{self.timestamp}.mp4')
            else:
                writer = animation.PillowWriter(fps=120,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
                anim.save(f'{self.timestamp}.gif', writer=writer)

        

        plt.show()