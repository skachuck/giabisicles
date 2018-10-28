"""
Author: Samuel B. Kachuck
Date: August 8, 2018

Computing and (offline) coupling for giapy and BISICLES PIG example.
"""

import os, subprocess
import numpy as np

from amrfile import io as amrio
#import h5py
#import warnings

def extract_field(amrID, field='thickness', level=0, order=0, returnxy=False):
    """ """
    lo,hi = amrio.queryDomainCorners(amrID, level)
    x,y,thk = amrio.readBox2D(amrID, level, lo, hi, field, order)
    if returnxy:
        return x,y,thk
    else:
        return thk

def get_latest_plot_file(drctry, basename):
    fnames = [i for i in os.listdir(drctry) if basename in i]
    inds = [int(i.split('.')[-3]) for i in fnames]
    return max(inds)

def get_time_from_plot_file(fname):
    amrID = amrio.load(fname)
    t = amrio.queryTime(amrID)
    amrio.free(amrID)
    return t

class gia2_surface_flux_fixeddt_object(object):
    def __init__(self, xg, yg, drctry, pbasename, gbasename, tmax, dt, ekwargs,
                    driver, rate=False, read='amrread'):
        self.xg = xg
        self.yg = yg
        self.nx, self.ny = len(xg), len(yg)
        self.dx = xg[1]-xg[0]
        self.dy = yg[1]-yg[0]

        self.taus, self.elup, self.alpha = calc_earth(nx=self.nx, ny=self.ny,
                        dx=self.dx, dy=self.dy, **ekwargs)

        self.dt = dt

        self.drctry = drctry
        self.gfname = self.drctry+gbasename
        self.pfname = self.drctry+pbasename

        self.DRIVER = driver

        self.rate = rate
        if read == 'amrread':
            self.read = self.amrread
        elif read == 'flatten_and_read':
            self.read = self.flatten_and_read
        else:
            raise(ValueError, 'read type not understood')

        # Initilize fields
        self.t = 0
        self.uplift = np.zeros((int(tmax/dt), self.ny,self.nx))
        self.ts = np.linspace(0, tmax, int(tmax/dt+1))
        self.update_interp(0)

    def __call__(self, x, y, t, thck, topg, *args, **kwargs):

        if not t == self.t:
            self.update_interp(t)
    
        xind = int((x - self.xg[0])/self.dx)
        yind = int((y - self.yg[0])/self.dy)

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
                #self.uplinterp = np.load(self.gfname.format(0))
                self.uplinterp = self.uplift[0]
                return

        tstep = int(t/self.dt)

        # Check if uplrate at t has already been computed
        if os.path.exists(self.gfname.format(tstep)):
           # If so, load the array
            self.uplift = np.load(self.gfname.format(tstep))
        else:
            thk1, bas1 = self.read(self.pfname.format(tstep))
            thk0, bas0 = self.read(self.pfname.format(tstep-1))

            dLoad = (thickness_above_floating(thk1, bas1) - 
                        thickness_above_floating(thk0, bas0))

            self.propagate_2d_adjustment(t, dLoad, tstep)

        uplrate = (self.uplift[tstep] - self.uplift[tstep-1])/self.dt

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
        nx, ny = self.nx, self.ny
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
        unit_resp = -(1 - np.exp(durs/self.taus*self.alpha/1000))
        # Propagate response to current and future times and transform back.
        self.uplift[tstep:] += np.real(0.3*np.fft.ifft2(unit_resp/self.alpha*dload_f))
        # Add elastic uplift off lithosphere.
        self.uplift[tstep:]+=np.real(0.3*np.fft.ifft2((1-1./self.alpha)*self.elup*dload_f[0]))

    def flatten_and_read(self, fname):
        oname = self.drctry+'tempfile{:d}.hdf5'.format(np.random.randint(255))
        warnings.warn('Beginning of function with oname {}'.format(oname), Warning)
        cmd = '$BISICLES_HOME/BISICLES/code/filetools/flatten2d.'+self.DRIVER
        subprocess.call(' '.join([cmd,fname,oname,'0']), shell=True)
        warnings.warn('File {} flattened'.format(oname), Warning)
        f = h5py.File(oname, 'r')
        warnings.warn('Flatenned file {} read'.format(oname), Warning)
        # Load the file, reshape it, and trim the artificial edges
        foo = f['level_0/data:datatype=0'].value.reshape((-1,self.ny+2,self.nx+2))[:,1:-1,1:-1]
        warnings.warn('Array read from {}'.format(oname), Warning)
        f.close()
        warnings.warn('File {} closed'.format(oname), Warning)
        thk, bas = foo[0], foo[6]
        warnings.warn('Arrays separated from {}'.format(oname), Warning)

        subprocess.call('rm '+oname, shell=True)
        warnings.warn('Flattened file {} deleted'.format(oname), Warning)
        return thk, bas

    def amrread(self, fname): 
        amrID = amrio.load(fname)
        thk = extract_field(amrID, 'thickness')
        bas = extract_field(amrID, 'Z_base')
        amrio.free(amrID)
        return thk, bas


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
    freq = 2*np.pi*np.sqrt(freqx[None,:]**2 + freqy[:,None]**2)

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



