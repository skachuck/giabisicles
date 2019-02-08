#ifdef CH_LANG_CC
/*
*      _______              __
*     / ___/ /  ___  __ _  / /  ___
*    / /__/ _ \/ _ \/  V \/ _ \/ _ \
*    \___/_//_/\___/_/_/_/_.__/\___/
*    Please refer to Copyright.txt, in Chombo's root directory.
*/
#endif

#include "SurfaceFlux.H"
#include "ComplexSurfaceFlux.H"
#include "LevelDataSurfaceFlux.H"
#include "GroundingLineLocalizedFlux.H"
#include "HotspotFlux.H"
#include <map>
#ifdef HAVE_PYTHON
#include "PythonInterface.H"
#endif
#include "IceConstants.H"
#include "FineInterp.H"
#include "CoarseAverage.H"
#include "computeNorm.H"
#include "BisiclesF_F.H"
#include "ParmParse.H"
#include "AmrIceBase.H"
#include "FortranInterfaceIBC.H"
#include "FillFromReference.H"

#include "NamespaceHeader.H"

  /// factory method
  /** return a pointerto a new SurfaceFlux object
   */

SurfaceFlux* 
zeroFlux::new_surfaceFlux()
{
  zeroFlux* newPtr = new zeroFlux;
  return static_cast<SurfaceFlux*>(newPtr);
}

  /// define source term for thickness evolution and place it in flux
  /** dt is included in case one needs integrals or averages over a
      timestep
  */
void
zeroFlux::surfaceThicknessFlux(LevelData<FArrayBox>& a_flux,
			       const AmrIceBase& a_amrIce, 
			       int a_level, Real a_dt)
{
  DataIterator dit = a_flux.dataIterator();
  for (dit.begin(); dit.ok(); ++dit)
    {
      a_flux[dit].setVal(0.0);
    }
}


constantFlux::constantFlux() : m_isValSet(false)
{
}

SurfaceFlux* 
constantFlux::new_surfaceFlux()
{
  constantFlux* newPtr = new constantFlux;
  newPtr->m_fluxVal = m_fluxVal;
  newPtr->m_isValSet = m_isValSet;
  return static_cast<SurfaceFlux*>(newPtr);
}

  /// define source term for thickness evolution and place it in flux
  /** dt is included in case one needs integrals or averages over a
      timestep
  */
void
constantFlux::surfaceThicknessFlux(LevelData<FArrayBox>& a_flux,
				   const AmrIceBase& a_amrIce, 
				   int a_level, Real a_dt)
{
  CH_assert(m_isValSet);
  DataIterator dit = a_flux.dataIterator();
  for (dit.begin(); dit.ok(); ++dit)
    {
      a_flux[dit].setVal(m_fluxVal);
    }
}


///
void
constantFlux::setFluxVal(const Real& a_fluxVal) 
{
  m_fluxVal = a_fluxVal; 
  // input value is in meters/year divide by secondsperyear 
  // to get flux in meters/second
  // slc: switch to flux in m/a
  //m_fluxVal/= secondsperyear;
  
  m_isValSet = true;
}


// --------------------------------------------------------------
// fortran interface surface flux
// --------------------------------------------------------------

/// class which takes an input fortran array 
/** averages or interpolates as necessary to fill the flux
 */

/// constructor
fortranInterfaceFlux::fortranInterfaceFlux()
  : m_isValSet(false)
{
}

/// factory method
/** return a pointer to a new SurfaceFlux object
 */
SurfaceFlux* 
fortranInterfaceFlux::new_surfaceFlux()
{
  if (m_verbose)
    {
      pout() << "in fortranInterfaceFlux::new_surfaceFlux" << endl;
    }

  fortranInterfaceFlux* newPtr = new fortranInterfaceFlux;

  newPtr->m_grids = m_grids;
  newPtr->m_gridsSet = m_gridsSet;
    // keep these as aliases, if they're actually defined
  if (!m_inputFlux.box().isEmpty())
    {
      newPtr->m_inputFlux.define(m_inputFlux.box(), 
                                 m_inputFlux.nComp(),
                                 m_inputFlux.dataPtr());
    }

  if (!m_ccInputFlux.box().isEmpty())
    {
      newPtr->m_ccInputFlux.define(m_ccInputFlux.box(), 
                                   m_ccInputFlux.nComp(),
                                   m_ccInputFlux.dataPtr());
    }      
  
  newPtr->m_inputFluxLDF = m_inputFluxLDF;
  
  newPtr->m_fluxGhost = m_fluxGhost;
  newPtr->m_inputFluxDx = m_inputFluxDx;
  newPtr->m_grids = m_grids;
  newPtr->m_gridsSet = m_gridsSet;
  
  newPtr->m_verbose = m_verbose;

  newPtr->m_isValSet = m_isValSet;
  return static_cast<SurfaceFlux*>(newPtr);  
}

/// define source term for thickness evolution and place it in flux
/** dt is included in case one needs integrals or averages over a
    timestep. flux should be defined in meters/second in the current 
    implementation. 
*/
void 
fortranInterfaceFlux::surfaceThicknessFlux(LevelData<FArrayBox>& a_flux,
					   const AmrIceBase& a_amrIce, 
					   int a_level, Real a_dt)
{
  CH_assert(m_isValSet);

  // this looks a lot like the code in FortranInterfaceIBC

  DisjointBoxLayout levelGrids = m_grids;
  RealVect dx = a_amrIce.dx(a_level);

  FillFromReference(a_flux,
                    *m_inputFluxLDF,
                    dx, m_inputFluxDx,
                    m_verbose);
#if 0
  // refinement ratio for flux
  Real refRatio = m_inputFluxDx[0]/dx[0];
 
  Real tolerance = 1.0e-6;

  if (refRatio > 1 + tolerance)
    {
      // importFlux coarser than what we want, have to interpolate
      //int nRef = (int)(refRatio + tolerance);

    }
  else if (refRatio < 1 + tolerance)
    {
      // importFlux finer than what we want, have to average
      //int nRef = (int)(1.0/refRatio + tolerance);
    }
  else
    {
      // same size, just copy  
      m_inputFluxLDF->copyTo(a_flux);
    }
  
#endif

}

/// set fortran array-valued surface flux
void
fortranInterfaceFlux::setFluxVal(Real* a_data_ptr,
                                 const int* a_dimInfo,
                                 const int* a_boxlo, const int* a_boxhi, 
                                 const Real* a_dew, const Real* a_dns,
                                 const IntVect& a_offset,
                                 const IntVect& a_nGhost,
                                 const ProblemDomain& a_domain,
                                 const bool a_nodal)

{

  m_fluxGhost = a_nGhost;
  m_nodalFlux = a_nodal;
  m_domain = a_domain;

  // dimInfo is (SPACEDIM, nz, nx, ny)

  // assumption is that data_ptr is indexed using fortran 
  // ordering from (1:dimInfo[1])1,dimInfo[2])
  // we want to use c ordering
  //cout << "a_dimonfo" << a_dimInfo[0] << a_dimInfo[1] << endl;  

  if (m_verbose)
    {
      pout() << "In FortranInterfaceIBC::setFlux:" << endl;
      pout() << " -- entering setFAB..." << endl;
    }
  
  FortranInterfaceIBC::setFAB(a_data_ptr, a_dimInfo,a_boxlo, a_boxhi,
                              a_dew,a_dns,a_offset,a_nGhost,
                              m_inputFlux, m_ccInputFlux, a_nodal);

  if (m_verbose)
    {
      pout() << "... done" << endl;
    }

  // if we haven't already set the grids, do it now
  if (!gridsSet())
    {
      if (m_verbose) 
        {
          pout() << " -- entering setGrids" << endl;
        }
      Box gridBox(m_ccInputFlux.box());
      gridBox.grow(-a_nGhost);
      FortranInterfaceIBC::setGrids(m_grids, gridBox, m_domain, m_verbose);
      m_gridsSet = true;
      if (m_verbose)
        {
          pout() << " -- out of setGrids" << endl;
        }
    }
  


  m_inputFluxDx = RealVect(D_DECL(*a_dew, *a_dns, 1));

  // now define LevelData and copy from FAB->LevelData 
  // (at some point will likely change this to be an  aliased 
  // constructor for the LevelData, but this should be fine for now....
  
  // if nodal, we'd like at least one ghost cell for the LDF
  // (since we'll eventually have to average back to nodes)
  IntVect LDFghost = m_fluxGhost;
  if (a_nodal && (LDFghost[0] == 0))
    {
      LDFghost += IntVect::Unit;
    }
      
  RefCountedPtr<LevelData<FArrayBox> > localLDFPtr(new LevelData<FArrayBox>(m_grids, 1, LDFghost) );

  m_inputFluxLDF = localLDFPtr;
  // fundamental assumption that there is no more than one box/ processor 
  // don't do anything if there is no data on this processor
  DataIterator dit = m_grids.dataIterator();
  for (dit.begin(); dit.ok(); ++dit)
    {
      Box copyBox = (*m_inputFluxLDF)[dit].box();
      copyBox &= m_inputFlux.box();
      (*m_inputFluxLDF)[dit].copy(m_inputFlux, copyBox);
      
    } // end DataIterator loop

  m_isValSet = true;
}

// constructor
ProductSurfaceFlux::ProductSurfaceFlux  (SurfaceFlux* a_flux1, SurfaceFlux* a_flux2)
{
  m_flux1 = a_flux1;
  m_flux2 = a_flux2;
}


/// destructor
ProductSurfaceFlux::~ProductSurfaceFlux()
{
  // I think we should be deleting m_flux1 and m_flux2 here
}

/// factory method
/** return a pointer to a new SurfaceFlux object
 */
SurfaceFlux* 
ProductSurfaceFlux::new_surfaceFlux()
{
  SurfaceFlux* f1 = m_flux1->new_surfaceFlux();
  SurfaceFlux* f2 = m_flux2->new_surfaceFlux();
  return static_cast<SurfaceFlux*>(new ProductSurfaceFlux(f1,f2));
}

/// define source term for thickness evolution and place it in flux
/** dt is included in case one needs integrals or averages over a
    timestep. flux should be defined in meters/second in the current 
    implementation. 
*/
void 
ProductSurfaceFlux::surfaceThicknessFlux(LevelData<FArrayBox>& a_flux,
					 const AmrIceBase& a_amrIce, 
					 int a_level, Real a_dt)
{
  LevelData<FArrayBox> f2(a_flux.getBoxes(), a_flux.nComp(), a_flux.ghostVect());
  // compute flux1, put in a_flux, compute flux2 in f2, then multiply
  m_flux1->surfaceThicknessFlux(a_flux, a_amrIce, a_level, a_dt);
  m_flux2->surfaceThicknessFlux(f2, a_amrIce, a_level, a_dt);
  DataIterator dit = a_flux.dataIterator();
  for (dit.begin(); dit.ok(); ++dit)
    {
      a_flux[dit].mult(f2[dit]);
    }
}




/// constructor
MaskedFlux::MaskedFlux(SurfaceFlux* a_groundedIceFlux, SurfaceFlux* a_floatingIceFlux,
		       SurfaceFlux* a_openSeaFlux, SurfaceFlux* a_openLandFlux)
  :m_groundedIceFlux(a_groundedIceFlux),m_floatingIceFlux(a_floatingIceFlux),
   m_openSeaFlux(a_openSeaFlux),m_openLandFlux(a_openLandFlux)
{
  CH_assert(a_groundedIceFlux);
  CH_assert(a_floatingIceFlux);
  CH_assert(a_openSeaFlux);
  CH_assert(a_openLandFlux);
}
/// factory method
/** return a pointer to a new SurfaceFlux object
 */
SurfaceFlux* MaskedFlux::new_surfaceFlux()
{
  SurfaceFlux* f = m_floatingIceFlux->new_surfaceFlux();
  SurfaceFlux* g = m_groundedIceFlux->new_surfaceFlux();
  SurfaceFlux* s = m_openSeaFlux->new_surfaceFlux();
  SurfaceFlux* l = m_openLandFlux->new_surfaceFlux();
  return static_cast<SurfaceFlux*>(new MaskedFlux(g,f,s,l));
}

void MaskedFlux::surfaceThicknessFlux(LevelData<FArrayBox>& a_flux,
				      const AmrIceBase& a_amrIce, 
				      int a_level, Real a_dt)
{

  //somewhat ineffcient, because we compute all fluxes everywhere. 
  //At some point, come back and only compute (say) grounded ice flux
  //in boxes where at least some of the ice is grounded.

  //first, grounded ice values
  m_groundedIceFlux->surfaceThicknessFlux(a_flux,a_amrIce,a_level,a_dt);

  //floating,open sea,open land ice values
  std::map<int,SurfaceFlux*> mask_flux;
  mask_flux[FLOATINGMASKVAL] = m_floatingIceFlux;
  mask_flux[OPENSEAMASKVAL] =  m_openSeaFlux ;
  mask_flux[OPENLANDMASKVAL] = m_openLandFlux;
  LevelData<FArrayBox> tmpFlux;
  tmpFlux.define(a_flux);
  for (std::map<int,SurfaceFlux*>::iterator i = mask_flux.begin(); i != mask_flux.end(); ++i)
    {
      i->second->surfaceThicknessFlux(tmpFlux, a_amrIce,a_level,a_dt);
      for (DataIterator dit(a_flux.dataIterator()); dit.ok(); ++dit)
	{
	  const BaseFab<int>& mask =  a_amrIce.geometry(a_level)->getFloatingMask()[dit];
      
	  Box box = mask.box();
	  box &= a_flux[dit].box();

	  int m = i->first;
	  FORT_MASKEDREPLACE(CHF_FRA1(a_flux[dit],0),
			     CHF_CONST_FRA1(tmpFlux[dit],0),
			     CHF_CONST_FIA1(mask,0),
			     CHF_CONST_INT(m),
			     CHF_BOX(box));
	}
    }

}

SurfaceFlux* AxbyFlux::new_surfaceFlux()
{
  return static_cast<SurfaceFlux*>(new AxbyFlux(m_a, m_x,m_b, m_y) );
}

AxbyFlux::AxbyFlux(const Real& a_a, SurfaceFlux* a_x, 
		   const Real& a_b, SurfaceFlux* a_y)
{

  m_a = a_a;
  m_b = a_b;
  
  CH_assert(a_x != NULL);
  CH_assert(a_y != NULL);
  m_x = a_x->new_surfaceFlux();
  m_y = a_y->new_surfaceFlux();
  CH_assert(m_x != NULL);
  CH_assert(m_y != NULL);

}

AxbyFlux::~AxbyFlux()
{
  if (m_x != NULL)
    {
      delete m_x; m_x = NULL;
    }
  if (m_y != NULL)
    {
      delete m_y; m_y = NULL;
    }
}

void AxbyFlux::surfaceThicknessFlux(LevelData<FArrayBox>& a_flux,
					  const AmrIceBase& a_amrIce, 
					  int a_level, Real a_dt)
{

  LevelData<FArrayBox> y_flux(a_flux.disjointBoxLayout(),1,a_flux.ghostVect());
  m_x->surfaceThicknessFlux(a_flux, a_amrIce, a_level,a_dt );
  m_y->surfaceThicknessFlux(y_flux, a_amrIce, a_level,a_dt );
  for (DataIterator dit(a_flux.disjointBoxLayout()); dit.ok(); ++dit)
    {
      a_flux[dit].axby(a_flux[dit],y_flux[dit],m_a,m_b);
    }
  

}


/// factory method
/** return a pointer to a new SurfaceFlux object
 */
SurfaceFlux* CompositeFlux::new_surfaceFlux()
{
  return static_cast<SurfaceFlux*>(new CompositeFlux(m_fluxes));
}

CompositeFlux::CompositeFlux(const Vector<SurfaceFlux*>& a_fluxes)
{
  m_fluxes.resize(a_fluxes.size());
  for (int i =0; i < a_fluxes.size(); i++)
    {
      CH_assert(a_fluxes[i] != NULL);
      m_fluxes[i] =  a_fluxes[i]->new_surfaceFlux();
    }
}

CompositeFlux::~CompositeFlux()
{
  for (int i =0; i < m_fluxes.size(); i++)
    {
      if (m_fluxes[i] != NULL)
	{
	  delete m_fluxes[i];
	  m_fluxes[i] = NULL;
	}
    }
}

void CompositeFlux::surfaceThicknessFlux(LevelData<FArrayBox>& a_flux,
					  const AmrIceBase& a_amrIce, 
					  int a_level, Real a_dt)
{
  m_fluxes[0]->surfaceThicknessFlux(a_flux, a_amrIce, a_level,a_dt );

  // this is hardly effcient... but it is convenient
  LevelData<FArrayBox> tmpFlux(a_flux.disjointBoxLayout(),1,a_flux.ghostVect());
  for (int i = 1; i <  m_fluxes.size(); i++)
    {
      m_fluxes[i]->surfaceThicknessFlux(tmpFlux, a_amrIce, a_level,a_dt );
      for (DataIterator dit(a_flux.disjointBoxLayout()); dit.ok(); ++dit)
	{
	  a_flux[dit] += tmpFlux[dit];
	}
    }

}



SurfaceFlux* BoxBoundedFlux::new_surfaceFlux()
{
  return static_cast<SurfaceFlux*>( new BoxBoundedFlux(m_lo, m_hi,m_startTime,m_endTime,m_fluxPtr));
}


void BoxBoundedFlux::surfaceThicknessFlux(LevelData<FArrayBox>& a_flux,
					  const AmrIceBase& a_amrIce, 
					  int a_level, Real a_dt)
{

  Real time = a_amrIce.time();
  

  for (DataIterator dit(a_flux.disjointBoxLayout()); dit.ok(); ++dit)
    {
      a_flux[dit].setVal(0.0);
    }

  if (time >= m_startTime && time < m_endTime)
    {
      // this is hardly efficient... but it is convenient
      LevelData<FArrayBox> tmpFlux(a_flux.disjointBoxLayout(),1,a_flux.ghostVect());
      m_fluxPtr->surfaceThicknessFlux(tmpFlux, a_amrIce, a_level,a_dt);
      const RealVect& dx = a_amrIce.dx(a_level);
      
      IntVect ilo,ihi;
      for (int dir =0; dir < SpaceDim; dir++)
	{
	  ilo[dir] = int(m_lo[dir]/dx[dir] - 0.5);
	  ihi[dir] = int(m_hi[dir]/dx[dir] - 0.5);
	}
      
    
      for (DataIterator dit(a_flux.disjointBoxLayout()); dit.ok(); ++dit)
	{
	  const Box& b = a_flux[dit].box();
	  if (b.intersects(Box(ilo,ihi)))
	    { 
	      Box sub(max(b.smallEnd(),ilo), min(b.bigEnd(),ihi));
	      a_flux[dit].plus(tmpFlux[dit],sub,0,0,1);
	    }
	}
    }

}

PiecewiseLinearFlux::PiecewiseLinearFlux(const Vector<Real>& a_abscissae, 
					 const Vector<Real>& a_ordinates, 
					 Real a_minWaterDepth)
  :m_abscissae(a_abscissae),m_ordinates(a_ordinates),
   m_minWaterDepth(a_minWaterDepth)
{
  CH_assert(m_abscissae.size() == m_ordinates.size());
}


SurfaceFlux* PiecewiseLinearFlux::new_surfaceFlux()
{
  return static_cast<SurfaceFlux*>(new PiecewiseLinearFlux(m_abscissae,m_ordinates,m_minWaterDepth));
}

void PiecewiseLinearFlux::surfaceThicknessFlux(LevelData<FArrayBox>& a_flux,
					       const AmrIceBase& a_amrIce, 
					       int a_level, Real a_dt)
{
  Vector<Real> dx(m_abscissae.size());
  Vector<Real> db(m_abscissae.size());

  const LevelData<FArrayBox>& levelH = a_amrIce.geometry(a_level)->getH();
  const LevelData<FArrayBox>& levelS = a_amrIce.geometry(a_level)->getSurfaceHeight();
  const LevelData<FArrayBox>& levelR = a_amrIce.geometry(a_level)->getTopography();
  for (DataIterator dit(a_flux.dataIterator()); dit.ok(); ++dit)
    {

      FORT_PWLFILL(CHF_FRA1(a_flux[dit],0),
		   CHF_CONST_FRA1(levelH[dit],0),
		   CHF_CONST_VR(m_abscissae),
		   CHF_CONST_VR(m_ordinates),
		   CHF_VR(dx),CHF_VR(db),
		   CHF_BOX(a_flux[dit].box()));
       
      if (m_minWaterDepth > 0.0)
	{
	  
	  FArrayBox D(a_flux[dit].box(),1);
	  FORT_WATERDEPTH(CHF_FRA1(D,0),
			  CHF_CONST_FRA1(levelH[dit],0),
			  CHF_CONST_FRA1(levelS[dit],0),
			  CHF_CONST_FRA1(levelR[dit],0),
			  CHF_BOX(a_flux[dit].box()));
  
	  
	  FORT_ZEROIFLESS(CHF_FRA1(a_flux[dit],0),
			  CHF_CONST_FRA1(D,0),
			  CHF_CONST_REAL(m_minWaterDepth),
			  CHF_BOX(a_flux[dit].box()));

	}

    }
}





/// factory method
/** return a pointer to a new SurfaceFlux object
 */
SurfaceFlux* NormalizedFlux::new_surfaceFlux()
{
  return static_cast<SurfaceFlux*>(new NormalizedFlux(m_direction, m_amplitude));
}

NormalizedFlux::NormalizedFlux(SurfaceFlux* a_direction, const Real& a_amplitude)
{
  m_direction = a_direction->new_surfaceFlux();
  m_amplitude = a_amplitude;
}

NormalizedFlux::~NormalizedFlux()
{
  if (m_direction != NULL)
    {
      delete m_direction; m_direction = NULL;
    }
}

void NormalizedFlux::surfaceThicknessFlux(LevelData<FArrayBox>& a_flux,
					  const AmrIceBase& a_amrIce, 
					  int a_level, Real a_dt)
{
  // Need to compute the norm over *all* the levels, which will
  // mean that under typical circumstances the flux and its norm is computed n_level times.
  // alternatives would involve risky assumptions or interface redesign

  Vector<LevelData<FArrayBox>*>  flux;
  for (int lev = 0; lev <= a_amrIce.finestLevel(); lev++)
    {
      flux.push_back(new LevelData<FArrayBox>(a_amrIce.grids(lev), a_flux.nComp(), a_flux.ghostVect()));
      m_direction->surfaceThicknessFlux(*flux[lev], a_amrIce, lev, a_dt );
    }
  Real norm = computeNorm(flux, a_amrIce.refRatios(), a_amrIce.dx(0)[0],  Interval(0,0), 1);
  for (int lev = 0; lev <= a_amrIce.finestLevel(); lev++)
    {
      if (flux[lev] != NULL)
	{
	  delete flux[lev]; flux[lev] = NULL;
	}
    }			  
  m_direction->surfaceThicknessFlux(a_flux, a_amrIce, a_level, a_dt );
  Real factor = m_amplitude/norm;
  for (DataIterator dit (a_flux.disjointBoxLayout()); dit.ok(); ++dit)
    {
      a_flux[dit] *= factor;
    }
}



SurfaceFlux* SurfaceFlux::parse(const char* a_prefix)
{
  
  SurfaceFlux* ptr = NULL;
  std::string type = "";
  
  ParmParse pp(a_prefix);
  pp.query("type",type);
  
  if (type == "zeroFlux")
    {
      ptr = new zeroFlux;
    }
  else if (type == "constantFlux")
    {
      constantFlux* constFluxPtr = new constantFlux;
      Real fluxVal;
      pp.get("flux_value", fluxVal);
      constFluxPtr->setFluxVal(fluxVal);
      ptr = static_cast<SurfaceFlux*>(constFluxPtr);
    }
  else if (type == "hotspotFlux")
    {
      HotspotFlux* hotspotFluxPtr = new HotspotFlux;
      Real fluxVal;
      pp.get("flux_value", fluxVal);
      hotspotFluxPtr->setFluxVal(fluxVal);
      Vector<Real> vect(SpaceDim,0.0);

      pp.getarr("radius",vect,0,SpaceDim);
      RealVect radius(D_DECL(vect[0], vect[1],vect[2]));      

      pp.getarr("center",vect,0,SpaceDim);
      RealVect center(D_DECL(vect[0], vect[1],vect[2]));

      hotspotFluxPtr->setSpotLoc(radius, center);
      
      Real startTime = -1.2345e300;
      Real stopTime = 1.2345e300;
      pp.query("start_time", startTime);
      pp.query("stop_time", stopTime);
      hotspotFluxPtr->setSpotTimes(startTime, stopTime);
      
      ptr = static_cast<SurfaceFlux*>(hotspotFluxPtr);
    }
  else if (type == "LevelData")
    {
      std::string fileFormat;
      pp.get("fileFormat",fileFormat);
      int n;
      pp.get("n",n);
      int offset = 0;
      pp.query("offset",offset);

      Real startTime = 0.0, timeStep = 1.0;
      pp.query("startTime", startTime);
      pp.query("timeStep", timeStep);
      std::string name = "flux";
      pp.query("name", name);
      bool linearInterp = true;
      pp.query("linearInterp", linearInterp);

      RefCountedPtr<std::map<Real,std::string> > tf
	(new std::map<Real,std::string>);
      
      for (int i =0; i < n; i++)
	{
	  char* file = new char[fileFormat.length()+32];
	  sprintf(file, fileFormat.c_str(),i + offset);
	  tf->insert(make_pair(startTime + Real(i)*timeStep, file));
	  delete file;
	}
      
      LevelDataSurfaceFlux* ldptr = new LevelDataSurfaceFlux(tf,name,linearInterp);
      ptr = static_cast<SurfaceFlux*>(ldptr);
    }
  else if (type == "fortran")
    {
      // don't have the context here to actually set values, but
      // we can at least allocate the object here and return it 
      fortranInterfaceFlux* fifptr = new fortranInterfaceFlux;
      ptr = static_cast<SurfaceFlux*>(fifptr);
    }
  else if (type == "piecewiseLinearFlux")
    {
      int n = 1;  
      pp.query("n",n);
      Vector<Real> vabs(n,0.0);
      Vector<Real> vord(n,0.0);
      pp.getarr("abscissae",vabs,0,n);
      pp.getarr("ordinates",vord,0,n);
      
      Real dmin = -1.0;
      pp.query("minWaterDepth",dmin);
      
      PiecewiseLinearFlux* pptr = new PiecewiseLinearFlux(vabs,vord,dmin);
      ptr = static_cast<SurfaceFlux*>(pptr);
    }
  else if (type == "productFlux")
    {
      std::string flux1Prefix(a_prefix);
      flux1Prefix += ".flux1";
      SurfaceFlux* flux1Ptr = parse(flux1Prefix.c_str());
      if (flux1Ptr == NULL)
	{
	  MayDay::Error("undefined flux1 in productFlux");
	}

      std::string flux2Prefix(a_prefix);
      flux2Prefix += ".flux2";
      SurfaceFlux* flux2Ptr = parse(flux2Prefix.c_str());
      if (flux2Ptr == NULL)
	{
	  MayDay::Error("undefined flux2 in productFlux");
	}
 

      ptr = static_cast<SurfaceFlux*>
	(new ProductSurfaceFlux(flux1Ptr->new_surfaceFlux(),
				flux2Ptr->new_surfaceFlux()));

      
      delete flux1Ptr;
      delete flux2Ptr;
    }
  else if (type == "maskedFlux")
    {
      std::string groundedPrefix(a_prefix);
      groundedPrefix += ".grounded";
      SurfaceFlux* groundedPtr = parse(groundedPrefix.c_str());
      if (groundedPtr == NULL)
	{
	  groundedPtr = new zeroFlux;
	}
 
      std::string floatingPrefix(a_prefix);
      floatingPrefix += ".floating";
      SurfaceFlux* floatingPtr = parse(floatingPrefix.c_str());
      if (floatingPtr == NULL)
	{
	  floatingPtr = new zeroFlux;
	}

      std::string openLandPrefix(a_prefix);
      openLandPrefix += ".openLand";
      SurfaceFlux* openLandPtr = parse(openLandPrefix.c_str());
      if (openLandPtr == NULL)
	{
	  openLandPtr = groundedPtr->new_surfaceFlux();
	}

      
      std::string openSeaPrefix(a_prefix);
      openSeaPrefix += ".openSea";
      SurfaceFlux* openSeaPtr = parse(openSeaPrefix.c_str());
      if (openSeaPtr == NULL)
	{
	  openSeaPtr = floatingPtr->new_surfaceFlux();
	}

      ptr = static_cast<SurfaceFlux*>
	(new MaskedFlux(groundedPtr->new_surfaceFlux(),
			floatingPtr->new_surfaceFlux(),
			openSeaPtr->new_surfaceFlux(),
			openLandPtr->new_surfaceFlux()));
      
      delete groundedPtr;
      delete floatingPtr;
      delete openSeaPtr;
      delete openLandPtr;
    }
  else if (type == "boxBoundedFlux")
    {
      Vector<Real> tmp(SpaceDim); 
      pp.getarr("lo",tmp,0,SpaceDim);
      RealVect lo (D_DECL(tmp[0], tmp[1],tmp[2]));
      pp.getarr("hi",tmp,0,SpaceDim);
      RealVect hi (D_DECL(tmp[0], tmp[1],tmp[2]));

      Vector<Real> time(2);
      time[0] = -1.2345678e+300;
      time[1] = 1.2345678e+300;
      pp.queryarr("time",time,0,2);
           
      std::string prefix(a_prefix);
      prefix += ".flux";
      SurfaceFlux* fluxPtr = parse(prefix.c_str());
      CH_assert(fluxPtr != NULL);
      BoxBoundedFlux bbf(lo,hi,time[0],time[1],fluxPtr);
      ptr = static_cast<SurfaceFlux*>(bbf.new_surfaceFlux());

    }
  else if (type == "axbyFlux")
   {
     Real a; 
     pp.get("a",a);
     
     std::string xpre(a_prefix);
     xpre += ".x";
     SurfaceFlux* x = parse(xpre.c_str());
     
     Real b; 
     pp.get("b",b);
     
     std::string ypre(a_prefix);
     ypre += ".y";
     SurfaceFlux* y = parse(ypre.c_str());
    
     AxbyFlux axbyFlux(a,x,b,y);
     ptr = static_cast<SurfaceFlux*>(axbyFlux.new_surfaceFlux());

   }
  else if (type == "compositeFlux")
   {
     
     
     int nElements;
     pp.get("nElements",nElements);
     
     std::string elementPrefix(a_prefix);
     elementPrefix += ".element";

     Vector<SurfaceFlux*> elements(nElements);
     for (int i = 0; i < nElements; i++)
       {
	 std::string prefix(elementPrefix);
	 char s[32];
	 sprintf(s,"%i",i);
	 prefix += s;
	 ParmParse pe(prefix.c_str());
	 elements[i] = parse(prefix.c_str());
	 CH_assert(elements[i] != NULL);
	 
       }
     CompositeFlux compositeFlux(elements);
     ptr = static_cast<SurfaceFlux*>(compositeFlux.new_surfaceFlux());
   
   }
  else if (type == "normalizedFlux")
   {
     
     Real amplitude;
     pp.get("amplitude",amplitude);
     std::string prefix(a_prefix);
     prefix += ".direction";
     SurfaceFlux* direction = parse(prefix.c_str());
     NormalizedFlux flux(direction, amplitude);
     ptr = static_cast<SurfaceFlux*>(flux.new_surfaceFlux());
   
   }

  else if (type == "groundingLineLocalizedFlux")
    {
      Real powerOfThickness = 0.0;
      pp.query("powerOfThickness",powerOfThickness);

      std::string glPrefix(a_prefix);
      glPrefix += ".groundingLine";
      SurfaceFlux* glPtr = parse(glPrefix.c_str());
      if (glPtr == NULL)
	{
	  glPtr = new zeroFlux;
	}
       
      std::string ambientPrefix(a_prefix);
      ambientPrefix += ".ambient";
      SurfaceFlux* ambientPtr = parse(ambientPrefix.c_str());
      if (ambientPtr == NULL)
	{
	  ambientPtr = new zeroFlux;
	}
      
      ptr = static_cast<SurfaceFlux*>
	(new GroundingLineLocalizedFlux(glPtr->new_surfaceFlux(),
					ambientPtr->new_surfaceFlux(),
					powerOfThickness ));
	 
      delete glPtr;
      delete ambientPtr;

    }
  else if (type == "buelerGIA") {

    BuelerGIAFlux* buelerGIAFluxPtr = new BuelerGIAFlux;

    int nlayers;
    pp.get("nlayers", nlayers);
    // Interpret the number of layers and type the properties accordingly.
    if ( nlayres == 1) {
      Real visc;
      pp.get("visc", visc);
    }
    else if ( nlayers == 2 ) {
      Vector<Real> visc(2);
      pp.getarr("visc",visc,0,nlayers);
      Real thk;
      pp.get("thk",thk);
    }
    else if ( nlayers>2 ) {
      Vector<Real> visc(nlayers); 
      pp.getarr("visc",visc,0,nlayers);
      Vector<Real> thk(nlayers-1);
      pp.getarr("thk",thk,0,nlayers-1);
    }
    else { 
      MayDay::Error("Bueler flux nlayers not understood.");
    }
    Real flex;
    pp.get("flex", flex);

    Real dt;
    pp.get("dt", dt);

    ParmParse ppCon("constants");
    Real m_iceDensity, m_gravity, m_mantleDensity;
    ppCon.query("ice_density",m_iceDensity);
    ppCon.query("gravity",m_gravity);
    ppCon.query("mantle_density",m_mantleDensity);

    ParmParse ppAmr("amr");
    Vector<int> ancells(3);
    ppAmr.getarr("num_cells",ancells, 0, ancells.size());
    int Nx, Ny;
    Nx = ancells[0];
    Ny = ancells[1];

    ParmParse ppMain("main");
    Vector<Real> domsize(3);
    ppMain.getarr("domain_size", domsize, 0, 3);
    Real Lx, Ly;
    Lx = domsize[0];
    Ly = domsize[1];

    BuelerGIA::BuelerGIAFlux buelerFlux();
    buelerFlux->setDomain(Nx, Ny, Lx, Ly);
    buelerFlux->setViscosity(visc);
    buelerFlux->setFlexural(flex);
    buelerFlux->setTimeStep(dt);
    buelerFlux->precomputeGIAstep();
    ptr = static_cast<SurfaceFlux*>(buelerFlux.new_surfaceFlux());

  }
#ifdef HAVE_PYTHON
  else if (type == "pythonFlux") {
    
    std::string module;
    pp.get("module",module);
    std::string function;
    pp.get("function",function);

    int nkwargs = 0;
    pp.query("n_kwargs",nkwargs);
    Vector<std::string> kwargName(nkwargs);
    if (nkwargs > 0)
      {
	pp.queryarr("kwargs",kwargName,0,nkwargs);
      }
    std::map<std::string, Real> kwarg;
    for (int i = 0; i < nkwargs; i++)
      {
	kwarg[kwargName[i]] = 0.0;
      }
    PythonInterface::PythonSurfaceFlux pythonFlux(module, function, kwarg);
    ptr = static_cast<SurfaceFlux*>(pythonFlux.new_surfaceFlux());

  }
#endif
  else if (type == "")
    {
      ptr = NULL; // return a NULL and leave it up to the caller to care
    }
  else
    {
      // a type was specified but it made no sense...
      pout() << "unknown flux type " << type << std::endl;
      MayDay::Error("unknown flux type");
    }
  return ptr;
  
}




#ifdef HAVE_PYTHON
#include "signal.h"


#endif
#include "NamespaceFooter.H"
