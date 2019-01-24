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


/// full constructor
pythonGIAflux::pythonGIAflux(const std::string& a_pyModuleName , 
			     const std::string& a_pyFuncGIAName,
			     std::map<std::string,Real>& a_kwarg)
{
#ifdef HAVE_PYTHON
  m_kwarg = a_kwarg;
  PythonInterface::InitializePythonModule(&m_pModule,  a_pyModuleName);
  PythonInterface::InitializePythonFunction(&m_pFuncGIA, m_pModule,  a_pyFuncGIAName);
#endif
}

pythonGIAflux::~pythonGIAflux()
{
  
#ifdef HAVE_PYTHON  
  delete m_pModule;
  delete m_pFuncGIA;
#endif
  
}

SurfaceFlux*
pythonGIAflux::new_surfaceFlux()
{
  MayDay::Error("pythonGIAflux::new_surfaceFlux not defined yet");
  pythonGIAflux* ptr = new pythonGIAflux;
  //pythonGIAflux* ptr = new pythonGIAflux(m_pModule, m_pFuncGIA, m_kwarg);
  return static_cast<SurfaceFlux*>( ptr);  
}

void
pythonGIAflux::surfaceThicknessFlux(LevelData<FArrayBox>& a_flux,
				    const AmrIceBase& a_amrIce, 
				    int a_level, Real a_dt)
{
}

#include "NamespaceFooter.H"

