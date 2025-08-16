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
                 stop_on_target=False,
                 verbose=True,
                 log_paths=True,
                 realtime_update_fn=None,
                 seed=None):
        """
        iterations: number of independent simulation runs
        output_dir: directory for csv files
        stop_on_target: if True, stop iteration as soon as any walker reaches target
        log_paths: if True, save full trajectories to positions.csv
        realtime_update_fn: callable(iteration, walker_type, walker_id, coords, step)
        seed: base seed for reproducibility
        """
        os.makedirs(output_dir, exist_ok=True)
    
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        dims_str = "x".join(map(str, self.dim))
        base = f"walkersim_{now}_dim{dims_str}_dil{self.dilution}_iters{iterations}"
    
        positions_fn = os.path.join(output_dir, f"{base}_positions.csv")
        summary_fn = os.path.join(output_dir, f"{base}_summary.csv")
        meta_fn = os.path.join(output_dir, f"{base}_meta.json")
    
        nd = len(self.dim)
        coord_cols = [f"x{i}" for i in range(nd)]
    
        # headers
        pos_header = ["iteration", "walker_type", "walker_id", "step"] + coord_cols
        summary_header = ["iteration", "total_walkers", "converged_walkers",
                          "avg_steps", "avg_distance", "min_steps", "max_steps", "percent_converged",
                          "mean_position", "var_position"]
    
        # write meta
        meta = {
            "timestamp": now,
            "dim": list(self.dim),
            "dilution": float(self.dilution),
            "iterations": iterations,
            "total_agents_types": len(self.agents),
            "agents": {name: {"n": w.n, "step": int(w.step), "step_length": int(w.step_len)} for name, w in self.agents.items()},
            "stop_on_target": bool(stop_on_target),
            "log_paths": bool(log_paths),
            "seed": None if seed is None else int(seed)
        }
        with open(meta_fn, "w") as mf:
            json.dump(meta, mf, indent=2)
    
        # open files
        with open(positions_fn, "w", newline="") as pos_file, \
             open(summary_fn, "w", newline="") as sum_file:
    
            pos_writer = csv.writer(pos_file)
            sum_writer = csv.writer(sum_file)
    
            if log_paths:
                pos_writer.writerow(pos_header)
            sum_writer.writerow(summary_header)
    
            total_walkers = sum(w.n for w in self.agents.values())
    
            for it in range(iterations):
                if seed is not None:
                    np.random.seed(int(seed) + it)
    
                # reset walkers
                for w in self.agents.values():
                    w.reinit()
                    # ensure each walker has an optional per-walker path buffer if visualization later wants it
                    if log_paths:
                        if not hasattr(w, "path_log"):
                            w.path_log = [[] for _ in range(w.n)]
                        else:
                            for i in range(w.n):
                                w.path_log[i] = []
    
                converged = {name: set() for name in self.agents.keys()}
                per_iter_steps = []
                per_iter_dist = []
    
                start_wall = time.time()
    
                while True:
                    all_conv = all(len(converged[name]) == self.agents[name].n for name in self.agents)
                    if all_conv:
                        break
    
                    stop_now = False
    
                    for walker_name, walker in self.agents.items():
                        for i in range(walker.n):
                            if i in converged[walker_name]:
                                continue
    
                            prev = walker.walker[i, 0].copy()
                            walker.walker[i, 0] = self.sample_walker(prev, walker.step, walker.step_len)
                            walker.walker[i, 1] += walker.step
                            walker.walker[i, 2] += walker.step * walker.step_len
    
                            step_count = int(walker.walker[i, 1])
                            distance = float(walker.walker[i, 2])
                            coords_arr = np.asarray(walker.walker[i, 0], dtype=int)
                            coords = tuple(int(x) for x in coords_arr)
    
                            # store in per-walker buffer for optional in-memory replay
                            if log_paths:
                                walker.path_log[i].append(coords)
    
                                # write row: iteration, walker_type, walker_id, step, x0, x1, ...
                                row = [it, walker_name, i, step_count] + list(coords)
                                pos_writer.writerow(row)
    
                            # realtime hook
                            if realtime_update_fn is not None:
                                try:
                                    realtime_update_fn(it, walker_name, i, coords, step_count)
                                except Exception:
                                    pass
    
                            if (coords_arr == walker.end).all():
                                converged[walker_name].add(i)
                                per_iter_steps.append(step_count)
                                per_iter_dist.append(distance)
                                if verbose:
                                    print(f"[Iter {it}] {walker_name}-{i} converged in {step_count} steps, {distance} distance")
                                if stop_on_target:
                                    stop_now = True
                                    break
                        if stop_now:
                            break
    
                    if stop_on_target and any(len(s) > 0 for s in converged.values()):
                        break
    
                # compute summary
                conv_count = sum(len(s) for s in converged.values())
                avg_steps = float(np.mean(per_iter_steps)) if per_iter_steps else 0.0
                avg_dist = float(np.mean(per_iter_dist)) if per_iter_dist else 0.0
                min_steps = int(np.min(per_iter_steps)) if per_iter_steps else 0
                max_steps = int(np.max(per_iter_steps)) if per_iter_steps else 0
                percent_conv = float(conv_count) / float(total_walkers) * 100.0
    
                # mean/var of final positions across all walkers (use their last logged pos or current)
                final_positions = []
                for wname, w in self.agents.items():
                    for i in range(w.n):
                        if log_paths and len(w.path_log[i]) > 0:
                            final_positions.append(np.array(w.path_log[i][-1], dtype=float))
                        else:
                            final_positions.append(np.asarray(w.walker[i, 0], dtype=float))
                if final_positions:
                    mean_pos = np.mean(final_positions, axis=0).tolist()
                    var_pos = np.var(final_positions, axis=0).tolist()
                else:
                    mean_pos = []
                    var_pos = []
    
                sum_writer.writerow([it, total_walkers, conv_count, avg_steps, avg_dist, min_steps, max_steps, percent_conv,
                                     json.dumps(mean_pos), json.dumps(var_pos)])
    
                if verbose:
                    print(f"Iteration {it} summary: {conv_count}/{total_walkers} converged ({percent_conv:.1f}%), avg steps {avg_steps:.2f}")
    
                pos_file.flush()
                sum_file.flush()
    
        return {"positions": positions_fn, "summary": summary_fn, "meta": meta_fn}





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
