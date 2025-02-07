#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.network import network
from src.walkers import walkers 
from src.simulator import simulation

def main():
    pars = {
    "turtle": {"n": 5, "step": 1, "step_length": 1},
    "rabbit": {"n": 1, "step": 1, "step_length": 2}
    }

    # Initialize the simulation
    sim = simulation(dim=(7, 7), dilution=0.01, total_agents=2, seed=42, truely_random=False, pars=pars)
    
    ## simulate 100 iterations of random walk with same parameters
    sim.simulate(100)


if __name__ == "__main__":
    main()
