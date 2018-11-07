#ifdef CH_LANG_CC
/*
*      _______              __
*     / ___/ /  ___  __ _  / /  ___
*    / /__/ _ \/ _ \/  V \/ _ \/ _ \
*    \___/_//_/\___/_/_/_/_.__/\___/
*    Please refer to Copyright.txt, in Chombo's root directory.
*/
#endif

// Virtual base class encapsulating surface fluxes for ice-sheet model

#ifndef _PYTHONGIAFLUX_H_
#define _PYTHONGIAFLUX_H_

#include "SurfaceFlux.H"
#include "NamespaceHeader.H"

/// implementation of SurfaceFlux for GIA thickness flux computed via python
/**
 */
class pythonGIAflux : public SurfaceFlux
{
public:
  pythonGIAflux();

  virtual ~pythonGIAflux();

  virtual SurfaceFlux* new_surfaceFlux();


  virtual void surfaceThicknessFlux(LevelData<FArrayBox>& a_flux,
				    const AmrIceBase& a_amrIce, 
				    int a_level, Real a_dt);

private:
  // Disallowed for all the usual reasons
  void operator=(const pythonGIAflux& a_input)
  {
    MayDay::Error("invalid operator");
  }

  // Disallowed for all the usual reasons
  pythonGIAflux(const pythonGIAflux& a_input)
  {
    MayDay::Error("invalid operator");
  }
};




#include "NamespaceFooter.H"

#endif
