#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.network import Network
from src.walkers import Walkers 
from src.simulator import Simulation

def main():
    pars = {
    "turtle": {"n": 5, "step": 1, "step_length": 1},
    "rabbit": {"n": 1, "step": 1, "step_length": 2}
    }

    # Initialize the simulation
    sim = Simulation(dim=(4, 4, 4), dilution=0, total_agents=2, seed=42, truely_random=False, pars=pars)
    
    ## simulate 100 iterations of random walk with same parameters
    sim.simulate(100, log_paths=True)

    sim.visualize_sim(save=True, Type='gif')


if __name__ == "__main__":
    main()