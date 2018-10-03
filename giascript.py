"""
Author: Samuel B. Kachuck
Date: August 8, 2018

Computing and (offline) coupling for giapy and BISICLES PIG example.
"""

import os
import numpy as np
from scipy.interpolate import RectBivariateSpline

from amrfile import io as amrio

from giapy.giaflat import compute_2d_uplift_stage, calc_earth

RUNNAME = '1en3-1-coup'
DRCTRY = '/data/piggia/'+RUNNAME+'/float2/'
PBASENAME = 'plot.pigv5.1km-'+RUNNAME+'.l1l2.2lev.{:06d}.2d.hdf5'
GBASENAME 'giaarr-findiff.pigv5.1km-'+RUNNAME+'.l1l2.2lev.{:06d}.2d.npy'
X, Y =  np.linspace(1000, 255000, 128), np.linspace(1000, 383000, 192)
SHAPE = (192,128)

X, Y = np.linspace(1152500, 2175500, 1024), np.linspace(2048500,3327500,1280)
SHAPE = (1280, 1024)

ekwargs = {'u'   :  1e-3,
           'fr23':  1}


def extract_field(amrID, field='thickness', level=0, order=0, returnxy=False):
    """ """
    lo,hi = amrio.queryDomainCorners(amrID, level)
    x,y,thk = amrio.readBox2D(amrID, level, lo, hi, field, order)
    if returnxy:
        return x,y,thk
    else:
        return thk

class PIG_2D_BISICLES_Ice(object):
    def __init__(self, drctry, basename):
        self.x = X
        self.y = Y
        self.shape = SHAPE
        self.drctry = drctry#'/data/piggia/raw/'
        self.basename = basename#'plot.pigv5.1km.l1l2.2lev.'
        #self.fnames = 
        
        self.readargs=[]
        self.readkwargs={}

        self.update_flist()

    def update_flist(self):
        # List of all output files, following basename
        fnames = [i for i in os.listdir(self.drctry) if self.basename in i]
        # Extract the timestep associated, for sorting
        inds = np.array([int(i.split('.')[-3]) for i in fnames])

        # Sort the fnames by timestep
        self.fnames = [fnames[i] for i in np.argsort(inds)]
        # Only take every 10 stages for first 1000 steps.
        sl1 = np.s_[0:-1000:10]
        sl2 = np.s_[-1000::1]
        self.fnames = self.fnames[sl1] + self.fnames[sl2]

    def get_ice_times(self):
        times = []
        for f in self.names:
            try:
                amrID = amrio.load(self.drctry+f)
            except:
                print('Unable to load {}'.format(self.drctry+f))
                raise
            times.append(amrio.queryTime(amrID))
            amrio.free(amrID)
        return np.array(times)

    def pairIter(self):
        """Iterate pariwise over the ice stages, keeping the newest one saved
        in memory to cut on read time.
        """
        icefname0 = self.fnames[0]
        t0, ice0, bas0 = self.read_icestage(self.drctry+icefname0, *self.readargs,
                                    **self.readkwargs)

        for next_ice_load_path in self.fnames[1:]:
            while not os.path.exists(self.drctry+next_ice_load_path):
                time.sleep(1)
            if os.path.isfile(self.drctry+next_ice_load_path):
                t1, ice1, bas1 = self.read_icestage(self.drctry+next_ice_load_path, *self.readargs,
                                        **self.readkwargs)  
                yield ice0, bas0, t0, ice1, bas1, t1
            else:
                raise ValueError("%s isn't a file!" % next_ice_load_path)

            # Save current step for next step
            ice0, t0 = ice1, t1

    def read_icestage(self, fname):
        # Load the BISICLES PIG ice thickness 
        amrID = amrio.load(fname)
        thk = extract_field(amrID, 'thickness')
        bas = extract_field(amrID, 'Z_base')
        t = amrio.queryTime(amrID)
        amrio.free(amrID)

        return t, thk, bas

def get_latest_plot_file(drctry, basename):
    fnames = [i for i in os.listdir(drctry) if basename in i]
    inds = [int(i.split('.')[-3]) for i in fnames]
    return max(inds)

def get_time_from_plot_file(fname):
    amrID = amrio.load(fname)
    t = amrio.queryTime(amrID)
    amrio.free(amrID)
    return t

class gia2_surface_flux_object(object):
    def __init__(self, rate=False):
        self.xg = X
        self.yg = Y
        self.drctry = DRCTRY
        pbasename = PBASENAME
        gbasename = GBASENAME
        self.gfname = self.drctry+gbasename
        self.pfname = self.drctry+pbasename

        self.rate = rate

        self.t = 0
        self.update_interp(0)

    def __call__(self, x, y, t, thck, topg, *args, **kwargs):

        if not t == self.t:
            self.update_interp(t)
    
        # Interpolate to x,y
        uplrate = float(self.uplinterp.ev(x,y))
    
        # Return
        return uplrate

    def update_interp(self,t): 
        if t == 0: 
            if not os.path.exists(self.gfname.format(0)):
                zeroupl = np.zeros(SHAPE)
                np.save(self.gfname.format(0), zeroupl)
                self.uplinterp = RectBivariateSpline(self.xg, self.yg, zeroupl.T)
                return
            else:
                self.uplinterp = RectBivariateSpline(self.xg, self.yg,
                                np.load(self.gfname.format(0)).T)
                return

        tstep = get_latest_plot_file(self.drctry, 'plot')

        # Check if uplrate at t has already been computed
        if os.path.exists(self.gfname.format(tstep)):
           # If so, load the array
            uplarr1 = np.load(self.gfname.format(tstep))
        else:
           # If not, compute it, make and save the array
            ice = PIG_2D_BISICLES_Ice(drctry=self.drctry, basename='plot')
            #assert ice.times[-1] == t, 'Time {} inconsistent with {}'.format(t, ice.times[-1])
            uplarr1 = compute_2d_uplift_stage(t, ice, 2, 2, self.rate,
                                                    **ekwargs)
            np.save(self.gfname.format(tstep), uplarr1)
        # If using finite difference, load previous stage and difference.
        if not self.rate:
            # Load previous step
            uplarr0 = np.load(self.gfname.format(tstep-1))
            t0 = get_time_from_plot_file(self.pfname.format(tstep-1))
            uplrate = ((uplarr1-uplarr0)/(t-t0))
        else:
            uplrate = uplarr1
    
        # Make the interpolation object
        self.uplinterp = RectBivariateSpline(self.xg, self.yg, uplrate.T)
        self.t = t

gia2_surface_flux_findiff = gia2_surface_flux_object()
