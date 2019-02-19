"""
Author: Samuel B. Kachuck
Date: August 8, 2018

Computing and (offline) coupling for giapy and BISICLES PIG example.
"""

import os
import numpy as np

RUNNAME = 'best2-bueler-long'
host = os.uname()[1]

if host == 'csrwks2018-0087':
    DRCTRY = '/data/piggia/'+RUNNAME+'/'
    DRIVER = 'Linux.64.mpic++.gfortran.OPT.MPI.ex'
if 'edison' in host:
    DRCTRY = '/scratch2/scratchdirs/skachuck/piggia/'+RUNNAME+'/'
    DRIVER = 'Linux.64.CC.ftn.OPT.MPI.GNU.ex'
TMAX = 150
DT = 0.03125

PBASENAME = 'plot.pigv5.1km.l1l2.2lev.{:06d}.2d.hdf5'
GBASENAME = 'giaarr-findiff.pigv5.1km.l1l2.2lev.{:06d}.2d.npy'

NX, NY = 128,192
X, Y =  np.linspace(1000, 255000, NX), np.linspace(1000, 383000, NY)

#ekwargs = {'u'   :  1e18,
#           'D':  1e23}

# 'Best 2' from Barletta et al., 2018
ekwargs = {'u2'  :  4.e18,
           'u1'  :  2.e19,
           'h'   :  200000.,
           'D':  13e23}

from giascript import *

#u0load = np.load('/home/skachuck/work/giabisicles/pig2dgia/best2_u0_final.np.npy')
u0load=None
gia2_surface_flux_findiff = BuelerTopgFlux(X, Y, DRCTRY,
                                    PBASENAME, GBASENAME, TMAX, DT, ekwargs,
                                    driver=DRIVER, U0=u0load, read='amrread',
                                    skip=1, fac=2, include_elastic=False)
