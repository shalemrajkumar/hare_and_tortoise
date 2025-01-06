#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from . import network

class walkers(network):
    """
    This class define and track walkers along the simulation
    
    """

    def __init__(self, network_parameters, dimensions, n = 1, step = 1, step_len = 1):
        self.n = n
        self.step = step
        self.step_len = step_len
        self.diameter = network_parameters[0]
        self.start = network_parameters[1]
        self.end = network_parameters[2]
        self.dimensions = dimensions

        ### 0 -> current location, 1 -> steps, 2 -> distance
        self.walker = walker = np.zeros((self.n, 3), dtype=object)
        self.reinit()
        

    def reinit (self):
        
        self.walker = np.zeros((self.n, 3), dtype=object)
        for i in range(self.n):
            self.walker[i, 0] = self.start.copy()
