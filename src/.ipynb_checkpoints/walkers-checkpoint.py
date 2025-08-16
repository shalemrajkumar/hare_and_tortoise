#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .network import Network

class Walkers(Network):
    """
    Simple walker class - just tracks position, steps, and distance
    """
    def __init__(self, dia, dimensions, n=1, step=1, step_len=1, start_id=0):
        self.n = n
        self.step = step
        self.step_len = step_len
        self.start = dia[1]
        self.end = dia[2]
        
        # Only need position, steps, distance
        self.walker = np.zeros((self.n, 4), dtype=object)
        
        #initializing the walkers
        for i in range(self.n):
            self.walker[i, 0] = self.start.copy()  # position
            self.walker[i, 1] = 0                  # steps taken
            self.walker[i, 2] = 0                  # distance traveled
            self.walker[i, 3] = start_id
            start_id += 1

    def reset(self):
        """Reset all walkers to start position"""
        for i in range(self.n):
            self.walker[i, 0] = self.start.copy()  # position
            self.walker[i, 1] = 0                  # steps taken
            self.walker[i, 2] = 0                  # distance traveled