#!/usr/bin/env python
# -*- coding: utf-8 -*

import numpy as np
import csv
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


    def simulate(self, iterations=0, output_file='simulation_results.csv'):

        ### save all the iterations data in csv
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Walker Name", "Walker ID", "Steps", "Distance", "Iteration"])

            # clear any random seed S.T its totally random
            for iteration in range(iterations):
                flag = False

                # simulate till it reaches end
                while 1:
                    for walker_name, walker in self.agents.items():

                        # loop over all n agents of each agent type
                        for i in range(walker.n):

                            # check for convergence
                            if (walker.walker[i, 0] == walker.end).all():
                                print(f"{i+1} / {walker.n} in all {walker_name} reached destination after {walker.walker[i, 1]} steps and {walker.walker[i, 2]} distance in  {iteration}_th iteation")
                                flag = True
                                break

                            # sample the i_th walker step and updated current location
                            walker.walker[i, 0] = self.sample_walker(walker.walker[i, 0], walker.step, walker.step_len)
                            walker.walker[i, 1] += walker.step
                            walker.walker[i, 2] += walker.step * walker.step_len

                    if flag:
                        for walkers in self.agents.values():
                            walkers.reinit()
                        break




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
