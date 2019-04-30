"""
Author: Samuel B. Kachuck
Date: August 8, 2018

Computing and (offline - read in and out) coupling for giapy and BISICLES.

Classes
-------
TopgFluxBase: An abstract class for BISICLES interface and common tasks
BuelerTopgFlux: The Bueler 2007 method, in velocity form, with elasticity
CathlesTopgFlux: 

Methods
-------
thickness_above_floating(thk, bas, beta=0.9)

BISICLES HELPER FUNCTIONS:
extract_field(amrID, field='thickness', level=0, order=0, returnxy=False)
get_latest_plot_file(drctry, basename)
get_time_from_plot_file(fname)
"""

from __future__ import division
import os, subprocess, sys
import numpy as np
import pickle
try:
    # BISICLES AMR tools for reading and writing.
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
    
    Parameters
    ----------
    xg, yg  :   1D ndarrays representing x- and y-coordinates (in m) of domain.
    drctry  :   The directory of the BISICLES run plot files
    pbasename:  
    tmax    :   The maximum time of simulation (used to load plot files)
    dt      :   The time step of the coupling.
    ekwargs :   a dict of earth rheology parameters. 
        Accepted keys:  u (viscosity, single vinscoust halfspace)
                        u1, u2, h (h thick layer with u2, over u1 halfspace)
                        g (gravity, defualt 9.8)
                        rho_r (density of mantle rock, default 3313)
                        rho_i (density of ice, default 910)
                        lam, mu (first and second lame parameters of crust)
                        D (flexural rigidity of lithosphere)
    U0      :   2D ndarray with size (len(xg), len(yg)) of disequilbrium uplift
        at simulation start (default None, begins in equilbrium)
    driver  :   BISICLES driver, needed for flattening AMR data
    rate    :   Return uplift velocity (deprecated)
    fixeddt :   Whether dt is fixed in simulation (default: True)
    read    :   BISICLES read function, either 'amrread' from amrfile library
        or 'flatten_and_read' to use filetools library (required driver set)
    skip    :   Fixed dt time steps to skip between velocity updates (default 1)
    fac     :   Pad the FFT of load and response by this many times (default 1,
        suggested 2).
    nwrite  :   Write the uplift field every nwrite steps.
    include_elastic :   Include elastic effects.

    """
    def __init__(self, xg, yg, drctry, pbasename, gbasename, tmax, dt, ekwargs,
                    U0=None, taf0=None, driver=None, rate=False, fixeddt=True,
                    read='amrread', skip=1, fac=2, nwrite=10,
                    include_elastic=False):
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
        self.rho_r = ekwargs.get('rho_r', 3313)         # kg m^-3
        self.rho_i = ekwargs.get('rho_i', 910)          # kg m^-3
        self.rho_w = ekwargs.get('rho_w', 1028)         # kg m^-3
        self.mu =    ekwargs.get('mu', 26.6)*1e9        # Pa
        self.lam =   ekwargs.get('lam', 34.2666)*1e9    # Pa
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
        # Lithospher filter
        self.alpha_l = 1 + self.k**4*self.D/self.g/self.rho_r
        # Relaxation time
        self.taus = 2*self.u*self.k/self.g/self.rho_r/self.alpha_l*self.r  # s
        # Elastic halfspace response, Cathles (1975) III-46
        self.ue = -1/(2*self.k)*(1/self.mu + 1/(self.mu+self.lam))  # m/Pa
        self.ue[0,0] = 0
        # with litosphere filter.
        self.ue *= (1-self.alpha_l**(-1))                           # m/Pa

        # Time and coupling properties
        self.t = 0
        self.tmax = tmax
        self.dt = dt                                    # yrs
        self.fixeddt = fixeddt
        self.skip = skip
        self.include_elastic = include_elastic

        # Miscellanious properties
        self.drctry = drctry
        self.gfname = self.drctry+gbasename
        self.pfname = self.drctry+pbasename
        self.U0 = U0
        self.taf0 = taf0
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
        # Restart values
        if self.U0 is not None:      # Set disequilibrium uplift at t=0
            self.Uhatn = self.fft2andpad(self.U0)
        else:
            self.Uhatn = np.zeros((ny*self.fac,nx*self.fac), dtype=np.complex128)    
        if self.taf0 is not None:    # Set initial (compensated) load at t=0
            self.taf0hat = self.fft2andpad(self.taf0)
        else:
            self.taf0hat = np.zeros((ny*self.fac,nx*self.fac), dtype=np.complex128)

        self.dLhat = np.zeros((ny*self.fac,nx*self.fac), dtype=np.complex128)    
        self.Udot = np.zeros((ny,nx))
        self.interper = RectBivariateSpline(self.xg, self.yg, self.Udot.T)

        self.uedotold = 0.
        self.dLhatold = 0.


    def _update_Udot(self,t):
        if not np.any(self.taf0hat) and self.U0 is None:
            thk0, bas0 = self.read(self.pfname.format(0))
            self.taf0hat = self.fft2andpad(thickness_above_floating(thk0,bas0,
                                                        self.rho_i/self.rho_w))
            self.bas0hat = self.fft2andpad(bas0)
 
        if self.fixeddt:
            n = int(t/self.dt)
        else:
            n = get_latest_plot_file(self.drctry, 'plot')
 
        print(self.fixeddt, n)

        thkn, basn = self.read(self.pfname.format(n))
        dLhat = (self.fft2andpad(thickness_above_floating(thkn,basn,self.rho_i/self.rho_w)) -
                                                    self.taf0hat)*self.rho_i*self.g

        self._Udot_from_dLhat(dLhat)

        self.dLhatold = dLhat

        self.interper = RectBivariateSpline(self.xg, self.yg, self.Udot.T)
        self.t = t

        if n % self.nwrite == 0:
            pickle.dump(self.Uhatn, open(self.gfname.format(n), 'w'))

    
    def _Udot_from_dLhat(self, dLhat):
        """Update the velocity field from the stress field dLhat (Pa).

        NOTE: dLhat should be the stress associated with the thickness above
        flotation of the ice, TAF. dLhat = FFT(TAF*den_ice*g)
        """
        # Bueler, et al. 2007 eq 11
        Uhatdot = -self.gamma*(dLhat + self.beta*self.Uhatn)*_SECSPERYEAR    # m / yr
        # Update uplift field prior to including elastic effect, so that fluid
        # equilibrium is corrct.
        self.Uhatn += Uhatdot*self.dt
        # Now include the elastic effect if requested.
        if self.include_elastic:
            uedot = self.ue*(dLhat - self.dLhatold)/self.dt 
            Uhatdot += uedot
            self.uedotold = uedot
        self.Udot = self.ifft2andcrop(Uhatdot)
        
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

def thickness_above_floating(thk, bas, beta=0.9):
    """Compute the (water equivalent) thickness above floating.

    thk - ice thickness
    bas - topography
    beta - ratio of ice density to water density (defauly=0.9)
    """
    #   Segment over ocean, checks for flotation    Over land
    taf = (beta*thk+bas)*(beta*thk>-bas)*(bas<0) + beta*thk*(bas>0)
    return taf

# TO BE IMPLEMENTED
#def ocean_load(thk, bas):
#    wl = -bas*(bas<0)

# BISICLES HELPER FUNCTIONS
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

if __name__ == '__main__':
    TMAX = 10
    if len(sys.argv) > 1:
        test = sys.argv[1]
        Nx, Ny = int(sys.argv[2]), int(sys.argv[3])
        xi, yj = np.meshgrid(np.arange(Nx), np.arange(Ny))
        
        # Generate the heaviside load (applied at t=0)
        load = np.zeros((Ny, Nx))
        if test == 'periodic':
            fx, fy = float(sys.argv[4]), float(sys.argv[5])
            load = np.cos(xi*fx*2*np.pi/Nx)*np.cos(yj*fy*2*np.pi/Ny)
        elif test in ['square', 'restart']:
            load[(xi/Nx > 0.333)*(xi/Nx < 0.666)*(yj/Ny > 0.333) *(yj/Ny<0.666)] = 1.
        else:
            raise ValueError('test style not understood')
        # Make the flux object
        buelerflux = BuelerTopgFlux(np.linspace(0,128000,Nx), np.linspace(0,128000,Ny), './', 'blah', 'blah', TMAX, 1., {},fac=1)       
        if test in ['periodic', 'square']:
            for i in range(TMAX):
                # The load is heaviside, so it doesn't change step to step, but the uplift field keeps updating.
                buelerflux._Udot_from_dLhat(buelerflux.fft2andpad(load))
                np.savetxt("{0}test_t{1:d}.txt".format(test,i), buelerflux.ifft2andcrop(buelerflux.Uhatn))
        elif test in ['restart']:
            buelerflux._Udot_from_dLhat(buelerflux.fft2andpad(load))
            buelerrestart = buelerflux = BuelerTopgFlux(np.linspace(0,128000,Nx), np.linspace(0,128000,Ny), 
                                                                './', 'blah',
                                                                'blah', TMAX,
                                                                1., {},fac=1,
                                                                U0=buelerflux.ifft2andcrop(buelerflux.Uhatn),
                                                                taf0=load)
            buelerflux._Udot_from_dLhat(buelerflux.fft2andpad(load))
            buelerrestart._Udot_from_dLhat(buelerflux.fft2andpad(load))
            if np.allclose(buelerflux.Uhatn, buelerrestart.Uhatn):
                print('Restart test success')
            else:
                print('Restart test fail')

