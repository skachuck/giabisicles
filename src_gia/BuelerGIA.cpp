/* c++ implementation of Bueler et al, 2007 GIA model
 * Author: Samuel B. Kachuck
 * Date: Jan 19, 2019
 *
 * Status: still very much in progress.
 */

#include <iostream>
#include <cmath>
#include <fstream>
#include <fftw3.h>

#include "SurfaceFlux.H"
#include "BisiclesF_F.H"
#include "BuelerGIA.H"
#include "HotspotFlux.H"
#include "AmrIce.H"


#include "NamespaceHeader.H"

/* TODO
 * Store initial thickness above flotation
 * Nondimensionalize the GIA step to limit mults.
 * Pad FFT
 * Control for time interval for update
 * Include two layer models
 * Include N-layer models
 * Include initialization
 */

/* Ideal ParmParse structure:
 * topographyFlux.type = buelerGIA
 * topographyFlux.nlayers = 1
 * topographyFlux.flex = 1e23
 * topographyFlux.visc = 1e18
 * topographyFlux.dt = 1
 * topographyFlux.pad = 1
 *
 * topographyFlux.layers = 2
 * topographyFlux.visc = 4e18 2e19
 * topographyFlux.thk  = 200
 * topographyFlux.dt = 1
 *
 * topographyFlux.init = /path/to/init.hdf5
 */


/* Information we need from BISICLES run
 * Domain properties: Num of x, y points; size of domain;
 * Initial ice thickness and topography OR inital thickness above flotation
 * Every time step: ice thickness, topography OR thickness above flotation
 *                  Update check info (time, magnitude of change, etc.)
 */


class BuelerGIAFlux : public SurfaceFlux
{
public:
  BuelerGIAFlux( );
  virtual ~BuelerGIAFlux();
 

  /// factory method
  /** return a pointerto a new SurfaceFlux object
   */
  virtual SurfaceFlux* new_surfaceFlux();

  virtual void surfaceThicknessFlux(LevelData<FArrayBox>& a_flux,
  			    const AmrIceBase& a_amrIce, 
  			    int a_level, Real a_dt);
  // Set 1-layer viscosity
  void setViscosity( Real& a_visc );
  // Set 2-layer viscosity
  //void setViscosity( RealVect a_viscvec, Real& a_thk );
  // Set N-layer viscosity (N>2)
  //void setViscosity( RealVect a_viscvec, RealVect a_thkvec ); 
  void setFlexural( Real& a_flex );
  void setTimestep( Real a_dt );
  void setDomain( int a_Nx, int a_Ny, Real a_Lx, Real a_Ly );
  void precomputeGIAstep();
protected:
  const Real PI2=2.*3.14159267;
  const Real SECSPERYEAR=31536000.;

  Real m_flex, m_visc, m_thk;
  RealVect m_viscvec, m_thkvec;
  Real m_dt;
  // physical constants
  Real m_iceDensity, m_gravity, m_mantleDensity;
  // Domain constants
  int m_Nx, m_Ny;
  Real m_Lx, m_Ly;
  bool m_isDomainSet;
  Real m_updatedTime;
  // Constants for GIA step (computed once, stored) (size Nx x Ny)
  LevelData<FArrayBox> m_beta, m_gamma;
  // Other quantities required during computation (size Nx x Ny)
  LevelData<FArrayBox> m_tafhat0;         // Initial thickness above flotation 
                                          // (assumes initially isostatic equilbrium).
  LevelData<FArrayBox> m_taf, m_tafhat;   // Thickness above flotation and FFT'd.
  LevelData<FArrayBox> m_udot, m_udothat; // Surface velocity and FFT'd.
  LevelData<FArrayBox> m_uhat;            // FFT'd surface displacements.
  // FFTW transformations.
  fftw_plan fftfor_load, fftinv_udot, fftinv_u;

protected:
  // Check if a velocity field update is necessary.
  bool updateCheck( Real t );
  void updateUdot( RealVect Hab );
  void fftpadfor();
  void fftinv();
  
}

/// factory method
/** return a pointerto a new SurfaceFlux object
 */
SurfaceFlux* 
BuelerGIAFluxFlux::new_surfaceFlux()
{
  BuelerGIAFlux* newPtr = new BuelerGIAFlux;

  // NEEDS TO BE IMPLEMENTED

  return static_cast<SurfaceFlux*>(newPtr);
}

BuelerTopgFlux::~BuelerTopgFlux() {
  fftw_destroy_plan(fftfor_load);
  fftw_destroy_plan(fftinv_udot);
  fftw_destroy_plan(fftinv_u);
}

void
BuelerGIAFlux::setDomain( int a_Nx, int a_Ny, Real a_Lx, Real a_Ly ) {
  m_Nx = a_Nx;
  m_Ny = a_Ny;
  m_Lx = a_Lx;
  m_Ly = a_Ly;
  m_isDomainSet = true;


void 
BuelerGIAFlux::setViscosity( Real& a_visc ) {
  m_visc = a_visc;
}
void 
BuelerGIAFlux::setViscosity( RealVect a_viscvec, Real& a_thk ) {
  m_viscvec = a_viscvec;
  m_thk = a_thk;
}

void 
BuelerGIAFlux::setViscosity( RealVect a_viscvec, RealVect a_thkvec ) {
}

void
BuelerGIAFlux::setFlexural( Real a_flex ){
  m_flex = a_flex;
}

void
BuelerGIAFlux::setTimestep( Real a_dt ){
  m_dt = a_dt;
}

void 
BuelerGIAFlux::surfaceThicknessFlux(LevelData<FArrayBox>& a_flux,
                                  const AmrIceBase& a_amrIce, 
                                  int a_level, Real a_dt)
{
  // Get time and check if need to update
  Real time = a_amrIce.time();
  bool needToUpdate = updateCheck(time);

  if ( needToUpdate ) {  
    // If need to update, do so. 
    updateUdot(a_amrIce);
    m_updatedTime = time; 
  }
  RealVect dx = a_amrIce.dx(a_level);
  FillFromReference(a_flux,
                    *m_udot,
                    dx, m_inputFluxDx,
                    m_verbose);

  // Using FillFromReference instead of below.
  //// Fill in flux
  //DataIterator dit = a_flux.dataIterator();
  //for (dit.begin(); dit.ok(); ++dit)
  //  {
  //    FArrayBox& thisFlux = a_flux[dit];
  //    thisFlux.setVal(0.0);
  //    BoxIterator bit(thisFlux.box());
  //    // compute distance from spot center
  //    for (bit.begin(); bit.ok(); ++bit)
  //      {
  //        IntVect iv = bit();
  //        thisFlux(iv,0) = m_udot(iv,0);
  //      } // end loop over cells in this box
  //  } // end loop over boxes
}

void
BuelerGIAFlux::precomputeGIAstep() {
  // We use real-to-real (discrete hartley) transformations.
  // Note: FFTW in column-major order by swapping order of Nx, Ny. 
  // Note: Inverse fft needs to be normalized by N.
  fftfor_load = fftw_plan_r2r_2d(Ny, Nx, &dL[0][0], &dLhat[0][0], 
                                  FFTW_DHT, FFTW_DHT, FFTW_ESTIMATE);
  fftinv_udot = fftw_plan_r2r_2d(Ny, Nx, &Uhatdot[0][0], &Udot[0][0], 
                                  FFTW_DHT, FFTW_DHT, FFTW_ESTIMATE);
  fftinv_u = fftw_plan_r2r_2d(Ny, Nx, &Uhatn[0][0], &Un[0][0], 
                                            FFTW_DHT, FFTW_DHT, FFTW_ESTIMATE);

  IntVect loVect = IntVect::Zero;
  IntVect hiVect(Nx-1, Ny-1);
  Box domBox(loVect, hiVect);
  Vector<Box> thisVectBox(1);
  thisVectBox[0] = domBox;
  Vector<int> procAssign(1,0);
  DisjointBoxLayout dbl(thisVectBox, procAssign);

  // Resize the arrays
  m_beta.resize(dbl,1);
  m_gamma.resize(dbl,1);
  m_tafhat0.resize(dbl,1);        
  m_taf.resize(dbl,1); 
  m_tafhat.resize(dbl,1); 
  m_udot.resize(dbl,1); 
  m_udothat.resize(dbl,1); 
  m_uhat.resize(dbl,1); 

  Real kx, ky, kij, tau, alpha_l;

  for (int i=0;i<Nx;i++){
    for (int j=0;j<Ny;j++){
      kx = PI2*min(i,m_Nx-i)/m_Nx*(m_Nx-1)/m_Lx;
      ky = PI2*min(j,m_Ny-j)/m_Ny*(m_Nx-1)/m_Ly;
      kij = sqrt(pow(kx,2),pow(ky,2));
      // The lithosphere filter.
      alpha_l = 1. + pow(kij,4)*m_flex/m_mantleDensity/m_gravity;
      // The explonential viscous relaxation time constant.
      tau = 2*m_visc*kij/m_gravity/m_mantleDensity/alpha_l;
      // The Bueler et al., 2007 fields.  
      m_beta[i][j] = m_mantleDensity*m_gravity + m_flex*pow(kij,4);
      m_gamma[i][j] = pow((m_beta[i][j]*(tau[i][j] + 0.5*m_dt*SECSPERYEAR)),-1);
    }
  }
  m_gamma[0][0] = 0.;
}

bool BuelerTopgFlux::updateCheck(Real time) {
  return time != m_updatedTime;
}

// The main physics are here. Right now set up to receive the loading stress
// (in Pa), but probably should accept height above flotation, from BISICLES.
void 
BuelerTopgFlux::updateUdot( const AmrIceBase& a_amrIce ) {
  //extract height above flotation for each level,
  // flatten it to a single level and compute response.
  int n = amrIce.finestLevel() + 1;
  Vector<LevelData<FArrayBox>* > data(n);
  Vector<RealVect> amrDx(n);
   
  for (int lev=0; lev<n; lev++) {
    data[lev] = const_cast<LevelData<FArrayBox>* >(&(a_amrIce.geometry(lev)->getThicknessOverFlotation()));
    amrDx[lev] = amrIce.dx(lev);
  }

  RealVect m_destDx = amrIce.dx(0)

  flattenCellData(m_taf, m_destDx,data,amrDx,m_verbose); 

  // FFT forward transform the load
  fftpadfor();
  // Update transformed velocity and uplift fields using Bueler, et al. 2007, eq 11.
  for (int i=0;i<Nx;i++) {
    for (int j=0;j<Ny;j++) {
      m_udothat[i][j] = -m_gamma[i][j]*((m_tafhat[i][j] - m_tafhat0)*m_iceDensity*m_gravity + 
                                        m_beta[i][j]*m_uhat[i][j])*SECSPERYEAR;
      m_uhat[i][j] += m_udothat[i][j]*m_dt;
    }
  }

  // Transform velocity to space-domain for flux readout.
  fftinv();
};

// Forward transform the load
// TODO Implement padding
void 
BuelerTopgFlux::fftpadfor () {
  fftw_execute(fftfor_load);
}

// Inverse transform velocities and uplifts and normalize
// TODO Implement padding
void 
BuelerTopgFlux::fftinv () {
  fftw_execute(fftinv_udot);
  fftw_execute(fftinv_u);
  for (int i=0;i<m_Nx;i++) {
    for (int j=0;j<m_Ny;j++) {
      // Normalization for inverse transform, see
      m_udot[i][j] /= pow((m_Nx*m_Ny),1);
      m_u[i][j] /= pow((m_Nx*m_Ny),1);
    }
  }
}

// A square load in the center of an Nx x Ny domain, for testing.
MatDoub squareLoad(int Nx, int Ny) {
  MatDoub l(Nx, Ny);
  for (int i=0;i<Nx;i++) {
    for (int j=0;j<Ny;j++) {
      float ni = float(i)/float(Nx);
      float nj = float(j)/float(Ny);
      if ( ni > 0.333 && ni < 0.666 && nj > 0.333 && nj < 0.666 ) { 
        l[i][j] = 1.;
      }
      else {
        l[i][j] = 0.;
      }
    }
  }
  return  l;
}

// Periodic loads in x and y.
MatDoub cosineLoad(int Nx, int Ny, float fx, float fy) {
  MatDoub l(Nx, Ny);
  for (int i=0;i<Nx;i++) {
    for (int j=0;j<Ny;j++) {
      float ni = float(i)/float(Nx);
      float nj = float(j)/float(Ny);
      l[i][j] = cos(PI2*fx*ni)*cos(PI2*fy*nj);
    }
  }
  return  l;
}

int main(int argc, char** argv) {
  if (argc > 1) {
    float TMAX = 10;
    string test = argv[1];
    const int Nx = stoi(argv[2]);
    const int Ny = stoi(argv[3]); 
    float fx, fy;
    if (test == "periodic" && argc > 3) {
      fx = stof(argv[4]);
      fy = stof(argv[5]);
    }
      
    MatDoub l;
    if (test == "square"){ 
      l=squareLoad(Nx,Ny); 
    }
    else if (test == "periodic"){
      l=cosineLoad(Nx,Ny,fx,fy);
    }
  
    BuelerTopgFlux flux(Nx, Ny, 128000., 128000., 1., 1);
    int t = 0;
    while (t<TMAX) {   
      flux.updateUdot(l);
      string fname = test+"test_t"+std::to_string(t)+".txt";
      savemat(flux.Un, fname);
      t++;
    }
  
    // learning fftw
    //MatDoub lhat(Nx*2,Ny*2);
    //fftw_plan fftfor = fftw_plan_r2r_2d(Nx, Ny, &flux.dL[0][0], &flux.dLhat[0][0], FFTW_DHT, FFTW_DHT, FFTW_ESTIMATE);
    //fftw_plan fftinv = fftw_plan_r2r_2d(Nx, Ny, &flux.dLhat[0][0], &flux.dL[0][0], FFTW_DHT, FFTW_DHT, FFTW_ESTIMATE);
    //fftw_execute(fftfor);
    //fftw_execute(fftfor);
    //fftw_execute(fftinv);
    //savemat(flux.k, "test.txt");
  }

  return 0;
}

#include "NamespaceFooter.H"
