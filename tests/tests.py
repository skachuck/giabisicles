"""
Author: Samuel B. Kachuck
Date: August 8, 2018

Computing and (offline) coupling for giapy and BISICLES PIG example.
"""

import os
import numpy as np

RUNNAME = 'best2'
host = os.uname()[1]

if host == 'csrwks2018-0087':
    DRCTRY = '/data/giabisiclestests/'+RUNNAME+'/'
    DRIVER = 'Linux.64.mpic++.gfortran.OPT.MPI.ex'
if 'edison' in host:
    DRCTRY = '/scratch2/scratchdirs/skachuck/giabisiclestests/'+RUNNAME+'/'
    DRIVER = 'Linux.64.CC.ftn.OPT.MPI.GNU.ex'
TMAX = 75
DT = 0.03125

PBASENAME = 'plot.pigv5.1km.l1l2.2lev.{:06d}.2d.hdf5'
GBASENAME = 'giaarr-findiff.pigv5.1km.l1l2.2lev.{:06d}.2d.npy'

NX, NY = 128,16
X, Y =  np.linspace(1000, 255000, NX), np.linspace(1000, 383000, NY)

#ekwargs = {'u'   :  1e-3,
#           'fr23':  1.}

# 'Best 2' from Barletta et al., 2018
ekwargs = {'u1'  :  4e-3,
           'u2'  :  2e-2,
           'h'   :  200.,
           'fr23':  20.}

from giascript import *

gia2_surface_flux_findiff = gia2_surface_flux_fixeddt_object(X, Y, DRCTRY,
                                    PBASENAME, GBASENAME, TMAX, DT, ekwargs,
                                    DRIVER, read='amrread')

def prescribed_1D_loadchange():
    pass

def prescribed_MISMIP3D_change():
    pass

def prescribed_topographic_change():
    pass
