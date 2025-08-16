#!/usr/bin/env python
# -*- coding: utf-8 -*

import os
import csv
import time
import json
from datetime import datetime
import numpy as np
import secrets

import matplotlib.pyplot as plt

from .network import network
from .walkers import walkers

class simulation(network):

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
            self.len_arr = self.init_pars(pars)
        else:
            self.len_arr = self.init_agents()

        self.adj_n = {0: self.adj}  # Initialize adjacency matrix for 1-step traversal
        super().adj_gen(self.adj_n, 1, np.max(self.len_arr))

    def init_agents(self):
        """
        Initialize agents via manual input.

        Returns:
            np.ndarray: Array of step lengths for all agents.
        """
        step_l_arr = np.zeros(self.total_agents, dtype=int)
        for i in range(self.total_agents):
            name = input("Agent name: ")
            n = int(input(f"Number of {name}: "))
            step = int(input(f"Number of steps for {name}: "))
            step_length = int(input(f"Step length for {name}: "))
            step_l_arr[i] = step_length
            self.agents[name] = walkers(self.dia, self.dim, n=n, step=step, step_l=step_length)
        print(f"Simulation initialized with {self.total_agents} agents.")
        print("Step length array:", step_l_arr)
        return step_l_arr

    def init_pars(self, pars):
        """
        Initialize agents using predefined parameters.
    
        Args:
            pars (dict): Dictionary of parameters. Format
    
        Returns:
            np.ndarray: Array of step lengths for all agents.
        """
        step_l_arr = np.zeros(len(pars), dtype=int)
        for i, (name, config) in enumerate(pars.items()):
            n = config["n"]
            step = config["step"]
            step_length = config["step_length"]
            step_l_arr[i] = step_length
            self.agents[name] = walkers(self.dia, self.dim, n, step, step_length)
        print(f"Simulation initialized with {len(pars)} agents from predefined parameters.")
        print("Step length array:", step_l_arr)
        return step_l_arr
    
    
    def simulate(self,
             iterations=1,
             output_dir="results",
             stop_on_first_reach=False,
             verbose=True,
             log_paths=False,
             realtime_update_fn=None,
             seed=None):
        """
        iterations: number of independent simulation runs
        output_dir: directory for csv files
        stop_on_first_reach: if True, stop iteration as soon as any walker reaches target
        log_paths: if True, log every walker step into the paths CSV
        realtime_update_fn: callable called as realtime_update_fn(iteration, walker_type, walker_id, position, step_count)
        seed: base seed for reproducibility (per-iteration seed will be derived)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        dims_str = "x".join(map(str, self.dim))
        base = f"sim_{now}_dim{dims_str}_dil{self.dilution}_iters{iterations}"
        
        paths_fn = os.path.join(output_dir, f"{base}_paths.csv")
        conv_fn = os.path.join(output_dir, f"{base}_convergence.csv")
        summary_fn = os.path.join(output_dir, f"{base}_summary.csv")
        
        # headers
        paths_header = ["timestamp", "iteration", "walker_type", "walker_id",
                        "step_count", "distance", "coords", "status"]
        conv_header = ["timestamp", "iteration", "walker_type", "walker_id",
                       "steps_taken", "distance_travelled", "convergence_time", "target_position"]
        summary_header = ["timestamp", "iteration", "total_walkers", "converged_walkers",
                          "avg_steps", "avg_distance", "min_steps", "max_steps", "percent_converged"]
        
        with open(paths_fn, "w", newline="") as paths_file, \
             open(conv_fn, "w", newline="") as conv_file, \
             open(summary_fn, "w", newline="") as summary_file:
        
            paths_writer = csv.writer(paths_file)
            conv_writer = csv.writer(conv_file)
            summary_writer = csv.writer(summary_file)
        
            if log_paths:
                paths_writer.writerow(paths_header)
            conv_writer.writerow(conv_header)
            summary_writer.writerow(summary_header)
        
            # total walkers count
            total_walkers = sum(w.n for w in self.agents.values())
        
            for it in range(iterations):
                iter_seed = None
                if seed is not None:
                    iter_seed = int(seed) + it
                    np.random.seed(iter_seed)
                    # secrets has no seed API; we keep secrets for true randomness if requested
        
                # reinitialize walkers
                for w in self.agents.values():
                    w.reinit()
        
                converged_records = []  # store (walker_type, id, steps, distance, time, target)
                per_iter_steps = []
                per_iter_dist = []
        
                # track which walkers have converged
                converged = {name: set() for name in self.agents.keys()}
        
                start_wall = time.time()
                # run until all walkers converged or stop_on_first_reach triggered
                while True:
                    # check stop condition
                    all_converged = all(len(converged[name]) == self.agents[name].n for name in self.agents)
                    if all_converged:
                        break
        
                    for walker_name, walker in self.agents.items():
                        for i in range(walker.n):
                            if i in converged[walker_name]:
                                continue
        
                            # take n-step samples for this walker
                            prev = walker.walker[i, 0].copy()
                            walker.walker[i, 0] = self.sample_walker(prev, walker.step, walker.step_len)
                            walker.walker[i, 1] += walker.step
                            walker.walker[i, 2] += walker.step * walker.step_len
        
                            step_count = int(walker.walker[i, 1])
                            distance = float(walker.walker[i, 2])
                            coords = tuple(map(int, walker.walker[i, 0].tolist()))
        
                            ts = datetime.now().isoformat()
        
                            # path logging
                            if log_paths:
                                status = "converged" if (walker.walker[i, 0] == walker.end).all() else "moving"
                                paths_writer.writerow([ts, it, walker_name, i, step_count, distance, json.dumps(coords), status])
        
                            # realtime hook
                            if realtime_update_fn is not None:
                                try:
                                    realtime_update_fn(it, walker_name, i, coords, step_count)
                                except Exception:
                                    pass
        
                            # convergence check
                            if (walker.walker[i, 0] == walker.end).all():
                                conv_time = time.time() - start_wall
                                target_pos = tuple(map(int, walker.end.tolist()))
                                conv_writer.writerow([ts, it, walker_name, i, step_count, distance, conv_time, json.dumps(target_pos)])
                                if verbose:
                                    print(f"[Iter {it}] {walker_name}-{i} converged in {step_count} steps, {distance} distance")
                                converged[walker_name].add(i)
                                converged_records.append((walker_name, i, step_count, distance, conv_time))
                                per_iter_steps.append(step_count)
                                per_iter_dist.append(distance)
        
                                if stop_on_first_reach:
                                    # mark remaining as not converged and break everything to end iteration
                                    for wname, w in self.agents.items():
                                        if wname != walker_name:
                                            # no need to update their states; we will not count them as converged
                                            pass
                                    break
                        else:
                            # inner for didn't break
                            continue
                        # inner for broke (stop_on_first_reach triggered)
                        break
        
                    if stop_on_first_reach and any(len(s) > 0 for s in converged.values()):
                        break
        
                # iteration finished -> compute summary
                conv_count = sum(len(s) for s in converged.values())
                avg_steps = float(np.mean(per_iter_steps)) if per_iter_steps else 0.0
                avg_dist = float(np.mean(per_iter_dist)) if per_iter_dist else 0.0
                min_steps = int(np.min(per_iter_steps)) if per_iter_steps else 0
                max_steps = int(np.max(per_iter_steps)) if per_iter_steps else 0
                percent_conv = float(conv_count) / float(total_walkers) * 100.0
        
                summary_writer.writerow([datetime.now().isoformat(), it, total_walkers,
                                         conv_count, avg_steps, avg_dist, min_steps, max_steps, percent_conv])
        
                if verbose:
                    print(f"Iteration {it} summary: converged {conv_count}/{total_walkers} ({percent_conv:.1f}%), avg steps {avg_steps:.2f}")
        
                # ensure file buffers flushed so realtime viewers can read
                paths_file.flush()
                conv_file.flush()
                summary_file.flush()
        
        return {"paths": paths_fn, "convergence": conv_fn, "summary": summary_fn}





    def sample_walker(self, curr, n_steps, step_l):

        for i in range(n_steps):

            ### get adj index
            idx = super().get_node_index(curr)

            ### find all degrees of freedom at idx
            freedom = np.where(self.adj_n[step_l - 1][idx] == 1)[0]

            if len(freedom) == 0:
                print(f"No valid moves for position {curr} at step length {step_l}.")
                return curr

            ### sample the freedom randomly

            if not self.truely_random:

                choice = np.random.choice(freedom)

            else:

                choice = secrets.choice(freedom)

            new_loc = super().get_node_coordinates(choice)

            curr = new_loc

        return new_loc

    def realtime_visualize(self, interval=0.05):
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Walker Simulation (Realtime)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
    
        # Set limits assuming known area; adjust if dynamic bounds are needed
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
    
        colors = plt.cm.tab10(np.arange(len(self.agents)))  # unique colors per agent type
        paths = {}
        scatters = {}
        end_markers = {}
    
        # init plot for each agent type
        for (idx, (walker_name, walker)) in enumerate(self.agents.items()):
            start_pos = walker.walker[:, 0]  # shape (n, 2)
            scatters[walker_name] = ax.scatter(start_pos[:, 0], start_pos[:, 1], color=colors[idx], label=walker_name)
            paths[walker_name] = [ax.plot([], [], color=colors[idx], alpha=0.5)[0] for _ in range(walker.n)]
            end_markers[walker_name] = ax.scatter([walker.end[0]], [walker.end[1]], color='red', marker='X')
    
        ax.legend()
    
        # main update loop
        while True:
            for (idx, (walker_name, walker)) in enumerate(self.agents.items()):
                positions = walker.walker[:, 0]
                scatters[walker_name].set_offsets(positions)
    
                for i in range(walker.n):
                    if hasattr(walker, "path_log"):
                        path = np.array(walker.path_log[i])
                        if path.shape[0] > 0:
                            paths[walker_name][i].set_data(path[:, 0], path[:, 1])
    
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(interval)
