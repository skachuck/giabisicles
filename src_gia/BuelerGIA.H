#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef _BUELERGIA_H_
#define _BUELERGIA_H_
#include "SurfaceFlux.H"
#include "RefCountedPtr.H"
#include <fftw3.h>

#include "NamespaceHeader.H"

class BuelerGIAFlux : public SurfaceFlux
{
public:
  BuelerGIAFlux();
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
  // Set the flexural rigidity (N m) of the lithosphere.
  void setFlexural( Real& a_flex );
  // Set the timestep of the GIA iteration
  void setTimestep( Real& a_dt );
  // Set domain size characteristics
  void setDomain( int a_Nx, int a_Ny, Real a_Lx, Real a_Ly );
  // Compute and store values for GIA steps 
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
  RefCountedPtr<LevelData<FArrayBox>> m_beta, m_gamma;
  // Other quantities required during computation (size Nx x Ny)
  RefCountedPtr<LevelData<FArrayBox>> m_tafhat0;         // Initial thickness above flotation 
                                          // (assumes initially isostatic equilbrium).
  RefCountedPtr<LevelData<FArrayBox>> m_taf, m_tafhat;   // Thickness above flotation and FFT'd.
  RefCountedPtr<LevelData<FArrayBox>> m_udot, m_udothat; // Surface velocity and FFT'd.
  RefCountedPtr<LevelData<FArrayBox>> m_uhat;            // FFT'd surface displacements.
  // FFTW transformations.
  fftw_plan fftfor_load, fftinv_udot, fftinv_u;

protected:
  // Check if a velocity field update is necessary.
  bool updateCheck( Real t );
  void updateUdot( const AmrIceBase& a_amrIce );
  void fftpadfor();
  void fftinv();
  
};

#include "NamespaceFooter.H"
#endif
