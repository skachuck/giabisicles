"""
Author: Samuel B. Kachuck
Date: August 8, 2018

Computing and (offline - read in and out) coupling for giapy and BISICLES.

Classes
-------
TopgFluxBase
BuelerTopgFlux
CathlesTopgFlux
TO BE DELETED: gia2_surface_flux_fixeddt_object

thickness_above_floating(thk, bas, beta=0.9)
extract_field(amrID, field='thickness', level=0, order=0, returnxy=False)
get_latest_plot_file(drctry, basename)
get_time_from_plot_file(fname)
TO BE DELETED: calc_earth
"""

import os, subprocess
import numpy as np
import pickle
try:
    from amrfile import io as amrio
except:
    pass
from scipy.interpolate import RectBivariateSpline
#import h5py
#import warnings
_SECSPERYEAR = 31536000.

class TopgFluxBase(object):
    """
    Base class for a Python Topg Flux implementation.

    Provides methods for initialization, computation of earth model-based decay
    times, and the calling interface. It does not provide methods for
    implementing the GIA, which need to be provided via, at least, initialize
    (for method-specific initialization) and _update_Udot, which performs the
    GIA.



    """
    def __init__(self, xg, yg, drctry, pbasename, gbasename, tmax, dt, ekwargs,
                    U0=None, driver=None, rate=False, fixeddt=True,
                    read='amrread', skip=1, fac=1, nwrite=10):
        # Grid and FFT properties
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
        self.u =     ekwargs.get('u', 1e21)             # Pa s
        self.u1 =    ekwargs.get('u1', None)            # Pa s
        self.u2 =    ekwargs.get('u2', None)            # Pa s
        self.h =     ekwargs.get('h', None)             # m
        self.g =     ekwargs.get('g', 9.8)              # m / s
        self.rho_r = ekwargs.get('rho', 3313)           # kg m^-3
        self.mu =    ekwargs.get('mu', 26.6)*1e9        # Pa
        self.D =     ekwargs.get('D', 1e23)             # N m
        
        warn = 'Two layer model must have u1 and u2 set.'
        assert (self.u1 is not None) == (self.u2 is not None) == (self.h is not None), warn
        if self.u1 is not None:
            # Cathles (1975) III-21
            # Bueler, et al. (2007) (15)
            c = np.cosh(self.k*self.h)
            s = np.sinh(self.k*self.h)
            ur = self.u2/self.u1
            ui = 1./ur
            r = 2*c*s*ur + (1-ur**2)*(self.k*self.h)**2 + ((ur*s)**2+c**2)
            r = r/((ur+ui)*s*c + self.k*self.h*(ur-ui) + (s**2+c**2))
            # Some hackery to correct overflow for large wavenumbers
            if ur < 1:
                r[r<ur] = ur
            else:
                r[r>ur] = ur
            r[np.isnan(r)] = ur
            self.u = self.u1
        else:
            r = 1
        self.r = r
        self.alpha_l = 1 + self.k**4*self.D/self.g/self.rho_r
        self.taus = 2*self.u*self.k/self.g/self.rho_r/self.alpha_l # s
#        self.taus = self.taus/_SECSPERYEAR                       # yrs

        # Time and coupling properties
        self.t = 0
        self.tmax = tmax
        self.dt = dt                                    # yrs
        self.fixeddt = dt
        self.skip = skip

        # Miscellanious properties
        self.drctry = drctry
        self.gfname = self.drctry+gbasename
        self.pfname = self.drctry+pbasename
        self.U0 = U0
        self.DRIVER = driver
        self.rate = rate
        self.nwrite = nwrite
        if read == 'amrread':
            self.read = self.amrread
        elif read == 'flatten_and_read':
            self.read = self.flatten_and_read
        else:
            raise(ValueError, 'read type not understood')

        self.initialize()

    def needToUpdate(self, t):
        """Check if updated velocities needed.
        
        Will return true if stored time is different from current time and
        appropriate number of steps skipped (self.skip, for fixeddt) or the
        coupling timestep has passed (self.dt, for not fixeddt)."""
        needtoupdate = False
        if not t == self.t:
            if self.fixeddt:
                needtoupdate = (t/self.dt % self.skip == 0)
            elif t >= self.t + self.dt:
                needtoupdate = True
        return needtoupdate

    def __call__(self, x, y, t, thck, topg, *args, **kwargs):

        if self.needToUpdate(t):
            self._update_Udot(t)
    
        xind = int((x - self.xg[0])/self.dx)
        yind = int((y - self.yg[0])/self.dy)

        #uplrate = float(self.Udot[yind, xind])
        uplrate = float(self.interper.ev(x,y))

        return uplrate

    def initialize(self):
        raise NotImplementedError

    def _update_Udot(self):
        raise NotImplementedError

    def fft2andpad(self, arr):
        shape = (self.ny*self.fac, self.nx*self.fac)
        return np.fft.fft2(arr, shape)

    def ifft2andcrop(self, arr):
        return np.real(np.fft.ifft2(arr))[..., :self.ny, :self.nx]

    def flatten_and_read(self, fname):
        """Reads an AMRfile in fname and flattens it to the coarsest resolution.

        Note: Does not require the amrfile.amrio python module or
                libamrfile.so library to be compiled.
        """
        # Create a temporary file for the flattened file.
        oname = self.drctry+'tempfile{:d}.hdf5'.format(np.random.randint(255))
        cmd = '$BISICLES_HOME/BISICLES/code/filetools/flatten2d.'+self.DRIVER
        subprocess.call(' '.join([cmd,fname,oname,'0']), shell=True)

        # Load the flattened file, reshape it, and trim the artificial edges.
        f = h5py.File(oname, 'r')
        foo = f['level_0/data:datatype=0'].value.reshape((-1,self.ny+2,self.nx+2))[:,1:-1,1:-1]
        f.close()
        
        # Extract the arrays and delete the temporary file.
        thk, bas = foo[0], foo[6]
        subprocess.call('rm '+oname, shell=True)

        return thk, bas

    def amrread(self, fname):
        """Reads the lowest 

        Note: Requries the amrfile python module and the libamrfile.so library
        to be compiled.
        """
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
        self.beta = self.rho_r*self.g+self.D*self.k**4    # Pa / m 
        self.gamma = (self.beta*(self.taus + 0.5*self.dt*_SECSPERYEAR))**(-1) # m/yr/Pa 
        self.gamma[0,0] = 0.

        nx, ny = self.nx, self.ny

        # Initialize fields in memory
        # We save the initial thickness to find the excess load in future.
        self.taf0hat = np.zeros((ny*self.fac,nx*self.fac), dtype=np.complex128)
        self.dLhat = np.zeros((ny*self.fac,nx*self.fac), dtype=np.complex128)
        self.Uhatn = np.zeros((ny*self.fac,nx*self.fac), dtype=np.complex128)
        if self.U0 is not None:
            self.Uhatn = self.fft2andpad(self.U0)
        self.Udot = np.zeros((ny,nx))
        self.interper = RectBivariateSpline(self.xg, self.yg, self.Udot.T)


    def _update_Udot(self,t):
        if not np.any(self.taf0hat) and self.U0 is None:
            thk0, bas0 = self.read(self.pfname.format(0))
            self.taf0hat = self.fft2andpad(thickness_above_floating(thk0,bas0))
            self.bas0hat = self.fft2andpad(bas0)
 
        if self.fixeddt:
            n = int(t/self.dt)
        else:
            n = get_latest_plot_file(self.drctry, 'plot')

        thkn, basn = self.read(self.pfname.format(n))
        dLhat = (self.fft2andpad(thickness_above_floating(thkn,basn)) -
                                                    self.taf0hat)*1000*self.g

        # Bueler, et al. 2007 eq 11
        Uhatdot = -self.gamma*(dLhat + self.beta*self.Uhatn)*_SECSPERYEAR    # m / yr
        self.Uhatn += Uhatdot*self.dt
        self.Udot = self.ifft2andcrop(Uhatdot)
        self.interper = RectBivariateSpline(self.xg, self.yg, self.Udot.T)
        self.t = t

        if n % self.nwrite == 0:
            pickle.dump(self.Uhatn, open(self.gfname.format(n), 'w'))

class CathlesTopgFlux(TopgFluxBase):
    def initialize(self):
        nsteps = int(tmax/dt/self.skip)
        nx, ny = self.nx, self.ny

        self.uplift = np.zeros((nsteps, ny*self.fac,nx*self.fac))
        self.ts = np.linspace(0, tmax, nsteps+1)
        self._update_Udot(0)
        self.Udot = np.zeros((ny,nx))

    def _update_Udot(self,t): 
        if t == 0: 
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
            self.Udot = self.uplift[tstep/self.skip]
        else:
            self.Udot = (self.uplift[tstep/self.skip]-self.uplift[tstep/self.skip-1])/(self.dt*self.skip)

        self.t = t

        if tstep % 10 == 0:
            np.save(self.gfname.format(tstep), self.uplift)

    def propagate_2d_adjustment(self, t, dLoad, tstep):
        """
        Propagate the effect of dLoad at t to future times in self.times.
        """
        
        # Fourier transform the laod.
        dload_f = self.fft2andpad(dLoad)
        # Compute the duration of dLoad from t to future times, with some
        # hackery for array slicing.
        durs = (self.ts[1:,None,None] if tstep == 0 
                else self.ts[1:-tstep/self.skip,None,None])
        # Construct the array of unit responses.
        if not self.rate:
            unit_resp = -(1 - np.exp(-durs/self.taus))
        else: 
            unit_resp = -1./self.taus*np.exp(-durs/self.taus)
            unit_resp[:,0,0] = 0.
        # Propagate response to current and future times and transform back.
        self.uplift[tstep/self.skip:] += np.real(0.3*self.ifft2andcrop(unit_resp/self.alpha_l*dload_f))

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
        self.D =  ekwargs.get('D', 1.)                  # N m
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

def ocean_load(thk, bas):
    wl = -bas*(bas<0)

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
    D = kwargs.get('D')/1e23

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



