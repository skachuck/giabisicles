"""
Author: Samuel B. Kachuck
Date: August 8, 2018

Computing and (offline - read in and out) coupling for giapy and BISICLES.
"""

import os, subprocess
import numpy as np

from amrfile import io as amrio
#import h5py
#import warnings
_SECSPERYEAR = 31536000

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

class TopgFluxBase(object):
    def __init__(self, xg, yg, drctry, pbasename, gbasename, tmax, dt, ekwargs,
                    driver, rate=False, read='amrread', skip=1, fac=1):
        self.xg = xg
        self.yg = yg 
        self.nx, self.ny = len(xg), len(yg)
        nx, ny = self.nx, self.ny
        self.dx = xg[1]-xg[0]                           # m
        self.dy = yg[1]-yg[0]                           # m
        self.fac = fac
        kx = np.fft.fftfreq(nx*fac, self.dx)            # m^-1
        ky = np.fft.fftfreq(ny*fac, self.dy)            # m^-1
        self.k = 2*np.pi*np.sqrt(kx[None,:]**2 + ky[:,None]**2) 
 
        # Earth rheology components
        self.u =     ekwargs.get('u', 1e0)*1e21         # Pa s
        self.u1 =    ekwargs.get('u1', None)            # Pa s
        self.u2 =    ekwargs.get('u2', None)            # Pa s
        self.h =     ekwargs.get('h', None)             # km
        self.g =     ekwargs.get('g', 9.8)              # m / s
        self.rho_r = ekwargs.get('rho', 3313)           # kg m^-3
        self.mu =    ekwargs.get('mu', 26.6)*1e9        # Pa
        self.fr23 =  ekwargs.get('fr23', 1.)*1e23       # N m
        
        warn = 'Two layer model must have u1 and u2 set.'
        assert (self.u1 is not None) == (self.u2 is not None) == (self.h is not None), warn

        self.t = 0
        self.tmax = tmax
        self.dt = dt                                    # yrs
        self.skip = skip

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

        self.initialize()

    def __call__(self, x, y, t, thck, topg, *args, **kwargs):

        if (not t == self.t) and (t/self.dt % self.skip == 0):
            self._update_Udot(t)
    
        xind = int((x - self.xg[0])/self.dx)
        yind = int((y - self.yg[0])/self.dy)

        uplrate = float(self.Udot[yind, xind])
    
        return uplrate

    def initialize(self):
        NotImplemented

    def fft2andpad(self, arr):
        shape = (self.ny*self.fac, self.nx*self.fac)
        return np.fft.fft2(arr, shape)

    def ifft2andcrop(self, arr):
        return np.real(np.fft.ifft2(arr))[..., :self.ny, :self.nx]

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

class BuelerTopgFlux(TopgFluxBase):
    """
    As far as I can tell, this method requires interleaving the timesteps of the
    ice-flow load and the gia. This works, however, especially well with the
    surfaceFlux interface, as we may assume that the velocities remain relatively
    unchanged, so we can assume that we compute the loads at each half-timestep
    and use a finite difference approximation of the uplift at integer time-steps
    to compute the displacement velocity.
    
    To compute the load at each half-timestep requires keeping in memory the load
    above flotation at the start of the simulation to get the load (relative to
    the start).
    """
    def initialize(self):
        if self.u1 is not None:
            # Bueler, et al. (2007) (15)
            c = np.cosh(self.k*self.h)
            s = np.sinh(self.k*self.h)
            ur = u2/u1
            ui = 1./ur
            r = 2*c*s*ur + (1-ur**2)*(self.k*self.h)**2 + ((ur*s)**2+c**2)
            r = r/((ur+ui)*s*c + self.k*self.h*(ur-ui) + (s**2+c**2))
            self.u = u1
        else:
            r = 1

        alpha = 2*self.u*r*self.k                         # N m^-3 s
        self.beta = self.rho_r*self.g+self.fr23*self.k**4    # N m^-3 
        self.gamma = (alpha + 0.5*self.dt*_SECSPERYEAR*self.beta)**(-1) # m/s/Pa 
        self.gamma[0,0] = 0.

        nx, ny = self.nx, self.ny

        # Initialize fields in memory
        # We save the initial thickness to find the excess load in future.
        self.taf0hat = np.zeros((ny*self.fac,nx*self.fac), dtype=np.complex128)
        self.dLhat = np.zeros((ny*self.fac,nx*self.fac), dtype=np.complex128)
        self.Uhatn = np.zeros((ny*self.fac,nx*self.fac), dtype=np.complex128)
        self.Udot = np.zeros((ny,nx))


    def _update_Udot(self,t):
        if not np.any(self.taf0hat):
            thk0, bas0 = self.read(self.pfname.format(0))
            self.taf0hat = self.fft2andpad(thickness_above_floating(thk0,bas0))
 
        n = int(t/self.dt)
        thkn, basn = self.read(self.pfname.format(n))
        dLhat = (self.fft2andpad(thickness_above_floating(thkn,basn)) -
                                                    self.taf0hat)*1000*self.g
        # Bueler, et al. 2007 eq 11
        Uhatdot = -self.gamma*(dLhat + self.beta*self.Uhatn)*_SECSPERYEAR
        self.Uhatn += Uhatdot*self.dt
        self.Udot = self.ifft2andcrop(Uhatdot)
        self.t = t


class gia2_surface_flux_fixeddt_object(object):
    def __init__(self, xg, yg, drctry, pbasename, gbasename, tmax, dt, ekwargs,
                    driver, rate=False, read='amrread', skip=1):
        self.xg = xg
        self.yg = yg 
        self.nx, self.ny = len(xg), len(yg)
        nx, ny = self.nx, self.ny
        self.dx = xg[1]-xg[0]                           # m
        self.dy = yg[1]-yg[0]                           # m
        self.fac = fac
        kx = np.fft.fftfreq(nx*fac, self.dx)            # m^-1
        ky = np.fft.fftfreq(ny*fac, self.dy)            # m^-1
        k = 2*np.pi*np.sqrt(kx[None,:]**2 + ky[:,None]**2)

        self.taus, self.elup, self.alpha = calc_earth(nx=self.nx, ny=self.ny,
                        dx=self.dx, dy=self.dy, **ekwargs)
 
        # Earth rheology components
        self.u =     ekwargs.get('u', 1e0)*1e21         # Pa s
        self.u1 =    ekwargs.get('u1', None)            # Pa s
        self.u2 =    ekwargs.get('u2', None)            # Pa s
        self.h =     ekwargs.get('h', None)             # km
        self.g =     ekwargs.get('g', 9.8)              # m / s
        self.rho_r = ekwargs.get('rho', 3313)           # kg m^-3
        self.mu =    ekwargs.get('mu', 26.6)*1e6        # Pa
        self.fr23 =  ekwargs.get('fr23', 1.)*1e23       # N m
        self.taus = -2*self.u*k/self.g/self.rho/_SECSPERYEAR 

        self.dt = dt
        self.skip = skip

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
        self.uplift = np.zeros((int(tmax/dt/self.skip), self.ny,self.nx))
        self.ts = np.linspace(0, tmax, int(tmax/dt/self.skip+1))
        self.update_interp(0)

    def __call__(self, x, y, t, thck, topg, *args, **kwargs):

        if (not t == self.t) and (t/self.dt % self.skip == 0):
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
            thk0, bas0 = self.read(self.pfname.format(tstep-self.skip))

            dLoad = (thickness_above_floating(thk1, bas1) - 
                        thickness_above_floating(thk0, bas0))

            self.propagate_2d_adjustment(t, dLoad, tstep)

        if self.rate:
            uplrate = self.uplift[tstep/self.skip]
        else:
            uplrate = (self.uplift[tstep/self.skip]-self.uplift[tstep/self.skip-1])/(self.dt*self.skip)

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
                else self.ts[1:-tstep/self.skip,None,None])
        # Construct the array of unit responses.
        if not self.rate:
            unit_resp = -(1 - np.exp(durs/self.taus*self.alpha/1000))
        else: 
            unit_resp = self.alpha/self.taus/1000*np.exp(durs/self.taus*self.alpha/1000)
            unit_resp[:,0,0] = 0.
        # Propagate response to current and future times and transform back.
        self.uplift[tstep/self.skip:] += np.real(0.3*np.fft.ifft2(unit_resp/self.alpha*dload_f))
        # Add elastic uplift off lithosphere.
        #self.uplift[tstep/self.skip:]+=np.real(0.3*np.fft.ifft2((1-1./self.alpha)*self.elup*dload_f[0]))

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
    freqy = np.fft.fftfreq(ny, dy)
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



