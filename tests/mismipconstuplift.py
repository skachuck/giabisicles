"""
Author: Samuel B. Kachuck
Date: August 8, 2018

Computing and (offline) coupling for giapy and BISICLES PIG example.
"""

import os
import numpy as np

class testsurfaceflux(object):
    def __init__(self): 
        pass 
    def __call__(self, x, y, t, thck, topg, *args, **kwargs):
	return 0.1 * (1-x/800000.)

gia2_surface_flux_findiff = testsurfaceflux() 
