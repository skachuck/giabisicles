#ifdef CH_LANG_CC
/*
*      _______              __
*     / ___/ /  ___  __ _  / /  ___
*    / /__/ _ \/ _ \/  V \/ _ \/ _ \
*    \___/_//_/\___/_/_/_/_.__/\___/
*    Please refer to Copyright.txt, in Chombo's root directory.
*/
#endif

#include "pythonGIAflux.H"
#include "NamespaceHeader.H"

/// implementation of SurfaceFlux for GIA thickness flux computed via python
/**
 */


pythonGIAflux::pythonGIAflux()
{
}


pythonGIAflux::~pythonGIAflux()
{
}

SurfaceFlux*
pythonGIAflux::new_surfaceFlux()
{
}

void
pythonGIAflux::surfaceThicknessFlux(LevelData<FArrayBox>& a_flux,
				    const AmrIceBase& a_amrIce, 
				    int a_level, Real a_dt)
{
}

#include "NamespaceFooter.H"

