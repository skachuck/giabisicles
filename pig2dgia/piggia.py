"""
Author: Samuel B. Kachuck
Date: August 8, 2018

Computing and (offline) coupling for giapy and BISICLES PIG example.
"""

import os, time
import numpy as np
#from scipy.interpolate import RectBivariateSpline

from amrfile import io as amrio

from giapy.giaflat import compute_2d_uplift_stage, calc_earth

RUNNAME = 'best2'
DRCTRY = '/data/piggia/'+RUNNAME+'/'
TMAX = 75
DT = 0.03125

#ekwargs = {'u'   :  1e0,
#           'fr23':  1.}

# 'Best 2' from Barletta et al., 2018
ekwargs = {'u1'  :  4e-3,
           'u2'  :  2e-2,
           'h'   :  200.,
           'fr23':  20.}

TAUS, ELUP, ALPHA = calc_earth(nx=128, ny=192, dx=2, dy=2, **ekwargs)


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
        self.x = np.linspace(1000, 255000, 128)
        self.y = np.linspace(1000, 383000, 192)
        self.shape = (192, 128)
        self.drctry = drctry#'/data/piggia/raw/'
        self.basename = basename#'plot.pigv5.1km.l1l2.2lev.'
        #self.fnames = 
        
        self.readargs=[]
        self.readkwargs={}

        self.update_flist()

    def __getitem__(self, key):
        return self.read_icestage(self.fnames[key])

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
        for f in self.fnames:
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

#class gia2_surface_flux_object(object):
#    def __init__(self, rate=False):
#        self.xg = np.linspace(1000, 255000, 128)
#        self.yg = np.linspace(1000, 383000, 192) 
#        self.drctry = DRCTRY
#        pbasename ='plot.pigv5.1km.l1l2.2lev.{:06d}.2d.hdf5'
#        gbasename = 'giaarr-findiff.pigv5.1km.l1l2.2lev.{:06d}.2d.npy'
#        self.gfname = self.drctry+gbasename
#        self.pfname = self.drctry+pbasename
#
#        self.rate = rate
#
#        self.t = 0
#        self.update_interp(0)
#
#    def __call__(self, x, y, t, thck, topg, *args, **kwargs):
#
#        if not t == self.t:
#            self.update_interp(t)
#    
#        # Interpolate to x,y
#        uplrate = float(self.uplinterp.ev(x,y))
#    
#        # Return
#        return uplrate
#
#    def update_interp(self,t): 
#        if t == 0: 
#            if not os.path.exists(self.gfname.format(0)):
#                zeroupl = np.zeros((192, 128))
#                np.save(self.gfname.format(0), zeroupl)
#                self.uplinterp = RectBivariateSpline(self.xg, self.yg, zeroupl.T)
#                return
#            else:
#                self.uplinterp = RectBivariateSpline(self.xg, self.yg,
#                                np.load(self.gfname.format(0)).T)
#                return
#
#        tstep = get_latest_plot_file(self.drctry, 'plot')
#
#        # Check if uplrate at t has already been computed
#        if os.path.exists(self.gfname.format(tstep)):
#           # If so, load the array
#            uplarr1 = np.load(self.gfname.format(tstep))
#        else:
#           # If not, compute it, make and save the array
#            ice = PIG_2D_BISICLES_Ice(drctry=self.drctry, basename='plot')
#            #assert ice.times[-1] == t, 'Time {} inconsistent with {}'.format(t, ice.times[-1])
#            uplarr1 = compute_2d_uplift_stage(t, ice, 2, 2, self.rate,
#                                                    **ekwargs)
#            np.save(self.gfname.format(tstep), uplarr1)
#        # If using finite difference, load previous stage and difference.
#        if not self.rate:
#            # Load previous step
#            uplarr0 = np.load(self.gfname.format(tstep-1))
#            t0 = get_time_from_plot_file(self.pfname.format(tstep-1))
#            uplrate = ((uplarr1-uplarr0)/(t-t0))
#        else:
#            uplrate = uplarr1
#    
#        # Make the interpolation object
#        #self.uplinterp = RectBivariateSpline(self.xg, self.yg, uplrate.T)
#        self.t = t

class gia2_surface_flux_fixeddt_object(object):
    def __init__(self, rate=False):
        self.xg = np.linspace(1000, 255000, 128)
        self.yg = np.linspace(1000, 383000, 192) 
        self.drctry = DRCTRY
        pbasename ='plot.pigv5.1km.l1l2.2lev.{:06d}.2d.hdf5'
        gbasename = 'giaarr-findiff.pigv5.1km.l1l2.2lev.{:06d}.2d.npy'
        self.gfname = self.drctry+gbasename
        self.pfname = self.drctry+pbasename

        self.rate = rate

        self.t = 0
        self.uplift = np.zeros((int(TMAX/DT), 192, 128))
        self.ts = np.linspace(0, TMAX, int(TMAX/DT+1))
        self.update_interp(0)

    def __call__(self, x, y, t, thck, topg, *args, **kwargs):

        if not t == self.t:
            self.update_interp(t)
    
        xind = int((x - 1000)/2000)
        yind = int((y - 1000)/2000)

        # Interpolate to x,y
        #uplrate = float(self.uplinterp.ev(x,y))
        uplrate = float(self.uplinterp[yind, xind])
    
        # Return
        return uplrate

    def update_interp(self,t): 
        if t == 0: 
            if not os.path.exists(self.gfname.format(0)):


                #self.uplinterp = RectBivariateSpline(self.xg, self.yg,
                #                                        self.uplift[0].T)
                self.uplinterp = self.uplift[0]
                return
            else:
                #self.uplinterp = RectBivariateSpline(self.xg, self.yg,
                #                np.load(self.gfname.format(0)).T)
                self.uplinterp = np.load(self.gfname.format(0))
                return

        tstep = int(t/DT)

        # Check if uplrate at t has already been computed
        if os.path.exists(self.gfname.format(tstep)):
           # If so, load the array
            self.uplift = np.load(self.gfname.format(tstep))
        else:
            
            amrID = amrio.load(self.pfname.format(tstep))
            thk1 = extract_field(amrID, 'thickness')
            bas1 = extract_field(amrID, 'Z_base')
            amrio.free(amrID)

            amrID = amrio.load(self.pfname.format(tstep - 1))
            thk0 = extract_field(amrID, 'thickness')
            bas0 = extract_field(amrID, 'Z_base')
            amrio.free(amrID)

            dLoad = (thickness_above_floating(thk1, bas1) - 
                        thickness_above_floating(thk0, bas0))

            self.propagate_2d_adjustment(t, dLoad, tstep)

        uplrate = (self.uplift[tstep] - self.uplift[tstep-1])/DT

        #self.uplinterp = RectBivariateSpline(self.xg, self.yg, uplrate.T)
        self.uplinterp = uplrate
        self.t = t

        if tstep % 10 == 0:
            np.save(self.gfname.format(tstep), self.uplift)

    def propagate_2d_adjustment(self, t, dLoad, tstep):
        """
        Propagate the effect of dLoad at t to future times in self.times.
        """
        
        # A factor for padding the FFT space
        padfac = 1
        nx, ny = 128, 192
        # A dload array, for padding and multiplying over more dims.
        dload = np.zeros((1, ny*padfac, nx*padfac))
        padsl = np.s_[..., 0:ny, 0:nx]
        dload[padsl] = dLoad

        # Fourier transform the laod.
        dload_f = np.fft.fft2(dload)
        # Compute the duration of dLoad from t to future times, with some
        # hackery for array slicing.
        durs = (self.ts[1:,None,None] if tstep == 0 
                else self.ts[1:-tstep,None,None])
        # Construct the array of unit responses.
        unit_resp = -(1 - np.exp(durs/TAUS*ALPHA/1000))
        # Propagate response to current and future times and transform back.
        self.uplift[tstep:] += np.real(0.3*np.fft.ifft2(unit_resp/ALPHA*dload_f))
        # Add elastic uplift off lithosphere.
        self.uplift[tstep:] += np.real(0.3*np.fft.ifft2((1-1./ALPHA)*ELUP*dload_f[0])) 

def thickness_above_floating(thk, bas, beta=0.9):
    """Compute the (water equivalent) thickness above floating.

    thk - ice thickness
    bas - topography
    beta - ratio of ice density to water density (defauly=0.9)
    """
    #   Segment over ocean, checks for flotation    Over land
    taf = (beta*thk+bas)*(beta*thk>-bas)*(bas<0) + beta*thk*(bas>0)
    return taf

def calc_earth(nx,ny,dx,dy,return_freq=False,**kwargs):
    """Compute the decay constants, elastic uplift, and lithosphere filter.

    Parameters
    ----------
    nx, ny : int
        Shape of flat earth grid.
    dx, dy : float
        Grid spacing.
    
    Returns
    -------

    """ 
    freqx = np.fft.fftfreq(nx, dx)
    freqy = np.fft.fftfreq(ny, dx)
    freq = 2*pi*np.sqrt(freqx[None,:]**2 + freqy[:,None]**2)

    u = kwargs.get('u', 1e0)
    u1 = kwargs.get('u1', None)
    u2 = kwargs.get('u2', None)
    h = kwargs.get('h', None)
    g = kwargs.get('g', 9.8)
    rho = kwargs.get('rho', 3313)
    mu = kwargs.get('mu', 26.6)
    fr23 = kwargs.get('fr23', 1.)

    # Error catching. For two viscous layers, u1, u2, and u3 must be set.
    assert (u1 is not None) == (u2 is not None) == (h is not None), 'Two layer model must have u1 and u2 set.'

    if u1 is not None:
        # Cathles (1975) III-21
        c = np.cosh(freq*h)
        s = np.sinh(freq*h)
        ur = u2/u1
        ui = 1./ur
        r = 2*c*s*ur + (1-ur**2)*(freq*h)**2 + ((ur*s)**2+c**2)
        r = r/((ur+ui)*s*c + freq*h*(ur-ui) + (s**2+c**2))

        u = u1

    else:
        r = 1


    # taus is in kyr
    taus = -2*u*np.abs(freq/g/rho * 1e8/np.pi)*r

    # elup is in m
    elup = -rho*g/2/mu/freq*1e-6
    elup[0,0] = 0

    # alpha is dimensionless
    alpha = 1 + freq**4*fr23/g/rho*1e11

    if return_freq:
        return freq, taus, elup, alpha
    else:
        return taus, elup, alpha
gia2_surface_flux_findiff = gia2_surface_flux_fixeddt_object()
