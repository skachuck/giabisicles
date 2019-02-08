#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <cmath>

using std::ifstream; 
using std::ios;

using std::cout;
using std::cin;
using std::cerr;
using std::endl;
using std::string;
#include "BISICLES_VERSION.H"
#include "Box.H"
#include "Vector.H"
#include "DisjointBoxLayout.H"
#include "ParmParse.H"
#include "LayoutIterator.H"
#include "BoxIterator.H"
#include "parstream.H"
#include "CoarseAverage.H"
#include "CoarseAverageFace.H"
#include "FineInterp.H"
#include "AMRIO.H"
#include "BRMeshRefine.H"
#include "LoadBalance.H"
#include "MayDay.H"
#include "AmrIce.H"
#include "computeNorm.H" 
#include "PatchGodunov.H"
#include "AdvectPhysics.H"
#include "PiecewiseLinearFillPatch.H"
#include "FineInterp.H"
#include "CoarseAverage.H"
#include "CellToEdge.H"
#include "EdgeToCell.H"
#include "DerivativesF_F.H"
#include "DivergenceF_F.H"
#include "computeSum.H"
#include "CONSTANTS.H"
#include "IceConstants.H"
#include "ExtrapBCF_F.H"
#include "amrIceF_F.H"
#include "BisiclesF_F.H"
#include "IceThermodynamics.H"
#include "PicardSolver.H"
#include "JFNKSolver.H"
#include "InverseVerticallyIntegratedVelocitySolver.H"
#include "PetscIceSolver.H"
#include "RelaxSolver.H"
#ifdef CH_USE_FAS
#include "FASIceSolverI.H"
#endif
#include "KnownVelocitySolver.H"
#include "VCAMRPoissonOp2.H"
#include "AMRPoissonOpF_F.H"
#include "CH_HDF5.H"
#include "IceUtility.H"
#include "LevelMappedDerivatives.H"
#ifdef HAVE_PYTHON
#include "PythonInterface.H"
#endif

#include "NamespaceHeader.H"

// small parameter defining when times are equal
#define TIME_EPS 1.0e-12

int AmrIce::s_verbosity = 1;

#if 0
void zeroBCValue(Real* pos,
                 int* dir,
                 Side::LoHiSide* side,
                 Real* a_values)
{
  a_values[0]=0.0;
}


void iceNeumannBC(FArrayBox& a_state,
                  const Box& a_valid,
                  const ProblemDomain& a_domain,
                  Real a_dx,
                  bool a_homogeneous)
{
  if(!a_domain.domainBox().contains(a_state.box()))
    {
      Box valid = a_valid;
      for(int dir=0; dir<CH_SPACEDIM; ++dir)
        {
          // don't do anything if periodic
          if (!a_domain.isPeriodic(dir))
            {
              Box ghostBoxLo = adjCellBox(valid, dir, Side::Lo, 1);
              Box ghostBoxHi = adjCellBox(valid, dir, Side::Hi, 1);
              if(!a_domain.domainBox().contains(ghostBoxLo))
                {
                  //Real bcVal = 0.0;
                  NeumBC(a_state,
                         valid,
                         a_dx,
                         a_homogeneous,
                         zeroBCValue,
                         dir,
                         Side::Lo);
                }

              if(!a_domain.domainBox().contains(ghostBoxHi))
                {
                  
                  NeumBC(a_state,
                         valid,
                         a_dx,
                         a_homogeneous,
                         zeroBCValue,
                         dir,
                         Side::Hi);
                }

            } // end if is not periodic in ith direction
        }
    }
}


void iceDirichletBC(FArrayBox& a_state,
                    const Box& a_valid,
                    const ProblemDomain& a_domain,
                    Real a_dx,
                    bool a_homogeneous)
{
  if(!a_domain.domainBox().contains(a_state.box()))
    {
      Box valid = a_valid;
      for(int dir=0; dir<CH_SPACEDIM; ++dir)
        {
          // don't do anything if periodic
          if (!a_domain.isPeriodic(dir))
            {
              Box ghostBoxLo = adjCellBox(valid, dir, Side::Lo, 1);
              Box ghostBoxHi = adjCellBox(valid, dir, Side::Hi, 1);
              if(!a_domain.domainBox().contains(ghostBoxLo))
                {
                  //Real bcVal = 0.0;
                  DiriBC(a_state,
                         valid,
                         a_dx,
                         a_homogeneous,
                         zeroBCValue,
                         dir,
                         Side::Lo);
                }

              if(!a_domain.domainBox().contains(ghostBoxHi))
                {
                  //Real bcVal = 0.0;
                  DiriBC(a_state,
                         valid,
                         a_dx,
                         a_homogeneous,
                         zeroBCValue,
                         dir,
                         Side::Hi);
                }

            } // end if is not periodic in ith direction
        }
    }
}

#endif

// compute crevasse depths
void
AmrIce::computeCrevasseDepths(LevelData<FArrayBox>& a_surfaceCrevasse,
			      LevelData<FArrayBox>& a_basalCrevasse,
			      int a_level) const
{
  const LevelSigmaCS& levelCoords = *geometry(a_level);
  //const LevelData<FArrayBox>& thk = levelCoords.getH();
  const LevelData<FArrayBox>& vt  = *viscousTensor(a_level);
  const Real& rhoi = levelCoords.iceDensity();
  const Real& rhoo = levelCoords.waterDensity();
  const Real& grav = levelCoords.gravity();
  for (DataIterator dit (levelCoords.grids()); dit.ok(); ++dit)
    {
      const FArrayBox& thck = levelCoords.getH()[dit];
      const FArrayBox& VT = vt[dit];
      const FArrayBox& Hab = levelCoords.getThicknessOverFlotation()[dit];
      const Box& b = levelCoords.grids()[dit];
      Box remnantBox = b; remnantBox.grow(1);
      FArrayBox remnant(remnantBox,1);
      
      for (BoxIterator bit(remnantBox); bit.ok(); ++bit)
	{
	  const IntVect& iv = bit();
	  const Real& sxx = VT(iv,0);
	  const Real& syy = VT(iv,3);
	  const Real& sxy = 0.5 *(VT(iv,2) + VT(iv,1));

	  //vertically integrated first principal stress
	  Real s1 = 
	    0.5 * (sxx + syy) +
	    std::sqrt ( std::pow ( 0.5*(sxx - syy), 2) + std::pow(sxy,2));

	  //vertically averaged first principal stress
	  s1 *= thck(iv) / (1.0e-6 + thck(iv)*thck(iv));

	  //crevasse depths
	  //only allow crevasses for extensional flow
	  if (s1 > 1e-6)
	    {

	      a_surfaceCrevasse[dit](iv) = 
		std::max(0.0,(s1 / (grav*rhoi) + 
			      1000.0/rhoi * m_waterDepth));
	      a_basalCrevasse[dit](iv) =
		std::max(0.0,((rhoi/(rhoo-rhoi)) * ((s1/(grav*rhoi))
						    - Hab(iv))));

	    }
	  else
	    {
	      //no crevasses if s1 <= 0.0
	      a_surfaceCrevasse[dit](iv) = 0.0;
	      a_basalCrevasse[dit](iv) = 0.0;
	    }
	}
    }
}
      
/// fill flattened Fortran array of data with ice thickness
void
AmrIce::getIceThickness(Real* a_data_ptr, int* a_dim_info, 
			Real* a_dew, Real* a_dns) const
{
  // dimInfo is (SPACEDIM, nz, nx, ny)

  // assumption is that data_ptr is indexed using fortran 
  // ordering from (1:dimInfo[1])1,dimInfo[2])
  // we want to use c ordering
  IntVect hiVect(D_DECL(a_dim_info[2]-1,a_dim_info[3]-1, a_dim_info[1]-1));
  Box fabBox(IntVect::Zero, hiVect);

  FArrayBox exportHfab(fabBox, 1, a_data_ptr); 

  // now pack this into a LevelData  -- this will need to be expanded in MPI
  Vector<Box> exportBoxes(1,fabBox);
  Vector<int> procAssign(1,0);
  // ignore periodicity, since we don't have ghost cells
  DisjointBoxLayout exportGrids(exportBoxes, procAssign);
  LevelData<FArrayBox> exportLDF(exportGrids, 1);

  // this isn't correct in 3d...
  CH_assert(SpaceDim != 3);
  RealVect exportDx = RealVect(D_DECL(*a_dew, *a_dns, 1));

  // assume that dx = dy, at least for now
  CH_assert (exportDx[0] == exportDx[1]);

  // start at level 0, then work our way up to finest level, 
  // over-writing as we go. An optimzation would be to check 
  // to see if finer levels cover the entire domain...
  for (int lev=0; lev<= m_finest_level; lev++)
    {
      const LevelSigmaCS& levelCS = *(m_vect_coordSys[lev]);
      const DisjointBoxLayout& levelGrids = m_amrGrids[lev];
      const LevelData<FArrayBox>& levelThickness = levelCS.getH();

      // refinement ratio
      Real refRatio = exportDx[0]/m_amrDx[lev];

      Real tolerance = 1.0e-6;

      if (refRatio > 1.0 + tolerance)
        {
          // current level finer than export level -- average solution
          // onto output
          int nRef = (int)(refRatio + tolerance);

          CoarseAverage averager(levelGrids,exportGrids,
                                 1, nRef);
          averager.averageToCoarse(exportLDF, levelThickness);
        }
      else if (refRatio < 1.0-tolerance)
        {
          // current level coarser than export level -- interpolate solution
          int nRef = (int)(1.0/refRatio + tolerance);
          
          // FineInterp needs a problem domain
          ProblemDomain exportDomain(m_amrDomains[lev]);
          exportDomain.refine(nRef);
                                     

          FineInterp interpolator(exportGrids, 1, nRef, exportDomain);
          interpolator.interpToFine(exportLDF, levelThickness);
        }
      else
        {
          // same resolution -- just do a copyTo
          levelThickness.copyTo(exportLDF);
        }

    } // end loop over levels

  // now copy to input fab
  DataIterator exportDit = exportLDF.dataIterator();
  for (exportDit.begin(); exportDit.ok(); ++exportDit)
    {
      // in parallel, this should only be on proc 0
      exportHfab.copy(exportLDF[exportDit]);
    }
  
}

bool 
AmrIce::isDefined() const
{
  return m_is_defined;
}


AmrIce::AmrIce() : m_velSolver(NULL),
                   m_constitutiveRelation(NULL),
		   m_rateFactor(NULL),
		   m_basalFrictionRelation(NULL),
		   m_basalRateFactor(NULL),
                   m_thicknessPhysPtr(NULL),
                   m_thicknessIBCPtr(NULL), 
                   m_surfaceFluxPtr(NULL),
		   m_basalFluxPtr(NULL),
		   m_surfaceHeatBoundaryDataPtr(NULL),
		   m_basalHeatBoundaryDataPtr(NULL),
		   m_topographyFluxPtr(NULL),
                   m_basalFrictionPtr(NULL)
		   
{
  setDefaults();
}

void
AmrIce::setDefaults()
{
  m_sigmaSet = false;
  // set some bogus values as defaults 
  m_is_defined = false;
  m_max_level = -1;
  m_finest_level = -1;
  m_finest_timestep_level = -1;
  m_tag_cap = 100;
  m_block_factor = -1;
  m_fill_ratio = -1;
  m_do_restart = false;
  m_restart_step = -1;
  //  m_constitutiveRelation = NULL;
  m_solverType = JFNK;
  // at the moment, 1 is the only one which works
  m_temporalAccuracy = 1;
  m_num_thickness_ghost = 4;
  // default is -1, which means use the solver's own defaults
  m_maxSolverIterations = -1;
  
  m_velocity_solver_tolerance = 1e-10;

  //by default, solve the full velocity problem on every timestep
  m_velocity_solve_interval = 1;
  //m_velSolver = NULL;
  m_domainSize = -1*RealVect::Unit;

  //m_beta_type = constantBeta;
  m_betaVal = 1000.0;
  m_betaEps = 0.0;
  m_basalSlope = RealVect::Zero;
  
  m_interpolate_zb = true;
  m_regrid_thickness_interpolation_method = 0;

  // set the rest of these to reasonable defaults
  m_nesting_radius = 1;
#ifdef CH_USE_PETSC
  m_nesting_radius = 3;
#endif
  m_tagOnGradVel = false;
  m_tagging_val = 0.1;
  m_tagOnLapVel = false;
  m_tagOnGroundedLapVel = false;
  m_laplacian_tagging_val = 1.0;
  m_laplacian_tagging_max_basal_friction_coef = 1.2345678e+300;
  m_tagOnEpsSqr = false;  
  m_epsSqr_tagVal =0.1;
  m_tagOnVelRHS = false;
  m_velRHS_tagVal = 1.0;
  m_tagOndivHgradVel = false;
  m_divHGradVel_tagVal = 1.0;
  m_tagGroundingLine  = false;
  m_tagVelDx = false;
  m_velDx_tagVal = 0.0;
  m_velDx_tagVal_finestLevelGrounded = -1;
  m_velDx_tagVal_finestLevelFloating = -1;
  m_tagMargin  = false;
  m_margin_tagVal_finestLevel = -1;
  m_tagAllIce  = false;
  m_tagEntireDomain = false;
  m_groundingLineTaggingMinVel = 200.0;
  m_groundingLineTaggingMaxBasalFrictionCoef = 1.2345678e+300;
  m_tag_thin_cavity = false;
  m_tag_thin_cavity_thickness = TINY_THICKNESS;
#ifdef HAVE_PYTHON
  m_tagPython = false;
  m_tagPythonModule = NULL;
  m_tagPythonFunction = NULL;
#endif

  m_tags_grow = 1;
  m_tags_grow_dir = IntVect::Zero;
  m_cfl = 0.25;
  m_max_dt_grow = 1.5;
  m_dt = 1.0e20;
  m_stable_dt = m_dt;
  m_max_box_size = 64;
  m_max_base_grid_size = -1;
  m_isothermal = true;
  m_waterDepth = 0.0;
  m_surfaceBoundaryHeatDataDirichlett = true;
  m_surfaceBoundaryHeatDataTemperature = true;
  m_iceDensity = 910.0; 
  m_seaWaterDensity = 1028.0;
  m_mantleDensity = 3313.0;
  m_gravity = 9.81;

  m_report_total_flux = false;
  m_report_calving = false;
  m_report_grounded_ice = false;
  m_report_area = false;
  m_report_discharge = false;
  m_report_time_interval = 0.01;
  m_eliminate_remote_ice = false;
  m_eliminate_remote_ice_max_iter = 10;
  m_eliminate_remote_ice_tol = 1.0;
  m_eliminate_remote_ice_after_regrid = false;
  setIsThckRecorded(false);

  m_plot_prefix = "plot";
  m_plot_interval = 10000000;
  m_plot_time_interval = 1.0e+12;
  m_reduced_plot = false;
  m_write_presolve_plotfiles = false;
  m_write_solver_rhs = false;
  m_write_dHDt = true;
  m_write_fluxVel = true;
  m_write_viscousTensor = false;
  m_write_baseVel = true;
  m_write_solver_rhs = false;
  m_write_internal_energy = false;
  m_write_map_file = false;
  m_write_thickness_sources = false;
  m_write_layer_velocities = false;
  m_write_mask = false;

  m_check_prefix = "chk";
  m_check_interval = -1;
  m_check_overwrite = true;
  m_check_exit = false;

  m_diffusionTreatment = NONE;
  m_additionalDiffusivity = 0.0;
  m_additionalVelocity = false;
  m_timeStepTicks = false;
  m_fixed_dt  = 0.0;

  m_limitVelRHS = false;
  m_gradLimitRadius = 10;
  
  m_limitFluxSpeed = -1.0;

  m_velocitySolveInitialResidualNorm = -1.0;
  m_doInitialVelSolve = true; 
  m_doInitialVelGuess = false; 
  m_initialGuessType = SlidingLaw;
  m_initialGuessConstMu = 1.0e+7; // a number that seems plausible, only needed
                                  // if m_initialGuessType = ConstMu
  m_initialGuessSolverType = JFNK; 
  m_initialGuessConstVel = RealVect::Zero; // only needed if m_initialGuessType = ConstMu *and* the basal traction relation is nonlinear
  m_reset_floating_friction_to_zero = true; // set basal friction to zero where ice is floating
 
  m_basalLengthScale = 0.0; // don't mess about with the basal friction / rhs by default
 
  m_wallDrag = true; //compute additional drag due to contact with rocky walls 
  m_wallDragExtra = 0.0; // assume wall drag proportional to basal drag;

  m_groundingLineCorrection = true;
  m_groundingLineSubdivision = 0;
  m_evolve_thickness = true;
  m_evolve_velocity = true;
  m_evolve_topography_fix_surface = false;
  m_grounded_ice_stable = false;
  m_floating_ice_stable = false;
  m_floating_ice_basal_flux_is_dhdt = false;
  m_grounded_ice_basal_flux_is_dhdt = false;
  m_frac_sources = false;

  m_groundingLineProximityScale = 1.0e+4;
  m_groundingLineProximityCalcType = 0 ; // default to the old (odd) behaviour
  //cache validity flags
  m_A_valid = false;
  m_groundingLineProximity_valid = false;
  m_viscousTensor_valid = false;

  zeroFlux* cfptr = new zeroFlux;
  m_basalFluxPtr = cfptr;
  
  m_calvingModelPtr = NULL;//new NoCalvingModel; 

  
  m_offsetTime = 0.0;

}

AmrIce::~AmrIce()
{
  if (s_verbosity > 4)
    {
      pout() << "AmrIce::~AmrIce()" << endl;
    }

  // clean up memory
  for (int lev=0; lev<m_velocity.size(); lev++)
    {
      if (m_velocity[lev] != NULL)
        {
          delete m_velocity[lev];
          m_velocity[lev] = NULL;
        }
    }
 
  // clean up ice fraction
  for (int lev=0; lev<m_iceFrac.size(); lev++)
    {
      if (m_iceFrac[lev] != NULL)
        {
          delete m_iceFrac[lev];
          m_iceFrac[lev] = NULL;
        }
    }
 


  // clean up velocityRHS storage if appropriate
  for (int lev=0; lev<m_velRHS.size(); lev++)
    {
      if (m_velRHS[lev] != NULL)
        {
          delete m_velRHS[lev];
          m_velRHS[lev] = NULL;
        }
    }
 
  // clean up basal C storage if appropriate
  for (int lev=0; lev < m_velBasalC.size(); lev++)
    {
      if (m_velBasalC[lev] != NULL)
        {
          delete m_velBasalC[lev];
          m_velBasalC[lev] = NULL;
        }
    }

  // clean up cell centered mu coefficient storage if appropriate
  for (int lev=0; lev < m_cellMuCoef.size(); lev++)
    {
      if (m_cellMuCoef[lev] != NULL)
        {
          delete m_cellMuCoef[lev];
          m_cellMuCoef[lev] = NULL;
        }
    }

  // clean up face advection velocity storage if appropriate
  for (int lev=0; lev < m_faceVelAdvection.size(); lev++)
    {
      if (m_faceVelAdvection[lev] != NULL)
        {
          delete m_faceVelAdvection[lev];
          m_faceVelAdvection[lev] = NULL;
        }
    }
  
  // clean up face total velocity storage if appropriate
  for (int lev=0; lev < m_faceVelTotal.size(); lev++)
    {
      if (m_faceVelTotal[lev] != NULL)
        {
          delete m_faceVelTotal[lev];
          m_faceVelTotal[lev] = NULL;
        }
    }

  for (int lev=0; lev < m_diffusivity.size(); lev++)
    {
      if (m_diffusivity[lev] != NULL)
	{
	  delete m_diffusivity[lev];
	  m_diffusivity[lev] = NULL;
	}
    }

  // clean up surface thickness storage if appropriate
  for (int lev=0; lev < m_surfaceThicknessSource.size(); lev++)
    {
      if (m_surfaceThicknessSource[lev] != NULL)
	{
	  delete m_surfaceThicknessSource[lev];
	  m_surfaceThicknessSource[lev] = NULL;
	}
      if (m_basalThicknessSource[lev] != NULL)
	{
	  delete m_basalThicknessSource[lev];
	  m_basalThicknessSource[lev] = NULL;
	}

      if (m_calvedIceThickness[lev] != NULL)
	{
	  delete m_calvedIceThickness[lev];
	  m_calvedIceThickness[lev] = NULL;
	}
      if (m_removedIceThickness[lev] != NULL)
	{
	  delete m_removedIceThickness[lev];
	  m_removedIceThickness[lev] = NULL;
	}
      if (m_addedIceThickness[lev] != NULL)
	{
	  delete m_addedIceThickness[lev];
	  m_addedIceThickness[lev] = NULL;
	}
      if (m_melangeThickness[lev] != NULL)
	{
	  delete m_melangeThickness[lev];
	  m_melangeThickness[lev] = NULL;
	}
      if (m_recordThickness[lev] != NULL)
	{
	  delete m_recordThickness[lev];
	  m_recordThickness[lev] = NULL;
	}
    }
  
  for (int lev=0; lev < m_divThicknessFlux.size(); lev++)
    {
      if (m_divThicknessFlux[lev] != NULL)
	{
	  delete m_divThicknessFlux[lev];
	  m_divThicknessFlux[lev] = NULL;
	}
    }

#if BISICLES_Z == BISICLES_LAYERED
  for (int lev=0; lev < m_layerSFaceXYVel.size(); lev++)
    {
      if (m_layerSFaceXYVel[lev] != NULL)
        {
          delete m_layerSFaceXYVel[lev];
          m_layerSFaceXYVel[lev] = NULL;
        }
    }

  for (int lev=0; lev < m_layerXYFaceXYVel.size(); lev++)
    {
      if (m_layerXYFaceXYVel[lev] != NULL)
        {
          delete m_layerXYFaceXYVel[lev];
          m_layerXYFaceXYVel[lev] = NULL;
        }
    }

#endif
  

  // clean up internalEnergy  storage if appropriate
  for (int lev=0; lev < m_internalEnergy.size(); lev++)
    {
      if (m_internalEnergy[lev] != NULL)
        {
          delete m_internalEnergy[lev];
          m_internalEnergy[lev] = NULL;
        }
    }
  for (int lev=0; lev < m_A.size(); lev++)
    {
      if (m_A[lev] != NULL)
        {
          delete m_A[lev];
          m_A[lev] = NULL;
        }
    }
#if BISICLES_Z == BISICLES_LAYERED

  for (int lev=0; lev < m_sA.size(); lev++)
    {
      if (m_sA[lev] != NULL)
        {
          delete m_sA[lev];
          m_sA[lev] = NULL;
        }
    }
 
  for (int lev=0; lev < m_sInternalEnergy.size(); lev++)
    {
      if (m_sInternalEnergy[lev] != NULL)
	{
	  delete m_sInternalEnergy[lev];
	  m_sInternalEnergy[lev] = NULL;
	}
    }
  for (int lev=0; lev < m_sHeatFlux.size(); lev++)
    {
      if (m_sHeatFlux[lev] != NULL)
	{
	  delete m_sHeatFlux[lev];
	  m_sHeatFlux[lev] = NULL;
	}
    }
  for (int lev=0; lev < m_bInternalEnergy.size(); lev++)
    {
      if (m_bInternalEnergy[lev] != NULL)
        {
          delete m_bInternalEnergy[lev];
          m_bInternalEnergy[lev] = NULL;
        }
    }

  for (int lev=0; lev < m_bHeatFlux.size(); lev++)
    {
      if (m_bHeatFlux[lev] != NULL)
	{
	  delete m_bHeatFlux[lev];
	  m_bHeatFlux[lev] = NULL;
	}
    }
  for (int lev=0; lev < m_bA.size(); lev++)
    {
      if (m_bA[lev] != NULL)
	{
	  delete m_bA[lev];
	  m_bA[lev] = NULL;
	}
    }
#endif

  for (int lev = 0; lev < m_old_thickness.size(); lev++) 
    {
      if (m_old_thickness[lev] != NULL) 
        {
          delete m_old_thickness[lev];
          m_old_thickness[lev] = NULL;
        }
    }

  for (int lev = 0; lev < m_groundingLineProximity.size(); lev++) 
    {
      if (m_groundingLineProximity[lev] != NULL) 
        {
          delete m_groundingLineProximity[lev];
          m_groundingLineProximity[lev] = NULL;
        }
    }

  for (int lev = 0; lev < m_dragCoef.size(); lev++) 
    {
      if (m_dragCoef[lev] != NULL) 
        {
          delete m_dragCoef[lev];
          m_dragCoef[lev] = NULL;
        }
    }
  
  for (int lev = 0; lev < m_viscosityCoefCell.size(); lev++) 
    {
      if (m_viscosityCoefCell[lev] != NULL) 
        {
          delete m_viscosityCoefCell[lev];
          m_viscosityCoefCell[lev] = NULL;
        }
    }

  for (int lev = 0; lev < m_viscousTensorCell.size(); lev++) 
    {
      if (m_viscousTensorCell[lev] != NULL) 
        {
          delete m_viscousTensorCell[lev];
          m_viscousTensorCell[lev] = NULL;
        }
    }
  for (int lev = 0; lev < m_viscousTensorFace.size(); lev++) 
    {
      if (m_viscousTensorFace[lev] != NULL) 
        {
          delete m_viscousTensorFace[lev];
          m_viscousTensorFace[lev] = NULL;
        }
    }

  for (int lev = 0; lev < m_deltaTopography.size(); lev++) 
    {
      if (m_deltaTopography[lev] != NULL) 
        {
          delete m_deltaTopography[lev];
          m_deltaTopography[lev] = NULL;
        }
    }

  if (m_velSolver != NULL)
    {
      // code crashes here. comment out until I figure out the problem...
      delete m_velSolver;
      m_velSolver = NULL;
    }

  if (m_constitutiveRelation != NULL)
    {
      delete m_constitutiveRelation;
      m_constitutiveRelation = NULL;
    }

  if (m_rateFactor != NULL)
    {
      delete m_rateFactor;
      m_rateFactor = NULL;
    }

  if (m_basalFrictionRelation != NULL)
    {
      delete m_basalFrictionRelation;
      m_basalFrictionRelation = NULL;
    }
  
  if (m_basalRateFactor != NULL)
    {
      delete m_basalRateFactor;
      m_basalRateFactor = NULL;
    }

  for (int lev=0; lev<m_thicknessPatchGodVect.size(); lev++)
    {
      if (m_thicknessPatchGodVect[lev] != NULL)
	{
	  delete m_thicknessPatchGodVect[lev];
	  m_thicknessPatchGodVect[lev] = NULL;
	}
    }


  if (m_thicknessPhysPtr != NULL)
    {
      delete m_thicknessPhysPtr;
      m_thicknessPhysPtr = NULL;
    }

  if (m_thicknessIBCPtr != NULL)
    {
      delete m_thicknessIBCPtr;
      m_thicknessIBCPtr = NULL;
    }


  if (m_internalEnergyIBCPtr != NULL)
    {
      delete m_internalEnergyIBCPtr;
      m_internalEnergyIBCPtr = NULL;
    }

  if (m_muCoefficientPtr != NULL)
    {
      delete m_muCoefficientPtr;
      m_muCoefficientPtr = NULL;
    }

  if (m_surfaceFluxPtr != NULL)
    {
      delete m_surfaceFluxPtr;
      m_surfaceFluxPtr = NULL;
    }
  if (m_basalFluxPtr != NULL)
    {
      delete m_basalFluxPtr;
      m_basalFluxPtr = NULL;
    }
  if (m_calvingModelPtr != NULL)
    {
      delete m_calvingModelPtr;
      m_calvingModelPtr = NULL;
    }
  if (m_basalFrictionPtr != NULL)
    {
      delete m_basalFrictionPtr;
      m_basalFrictionPtr = NULL;
    }

#ifdef HAVE_PYTHON
  if (m_tagPythonFunction != NULL)
    {
      Py_DECREF(m_tagPythonFunction);
    }
  if (m_tagPythonModule != NULL)
    {
      Py_DECREF(m_tagPythonModule);
    }
#endif

  // that should be it!

}


void
AmrIce::initialize()
{

  if (s_verbosity > 3) 
    {
      pout() << "AmrIce::initialize" << endl;
    }

  // set time to be 0 -- do this now in case it needs to be changed later
  m_time = 0.0;
  m_cur_step = 0;

  // first, read in info from parmParse file
  ParmParse ppCon("constants");
  ppCon.query("ice_density",m_iceDensity);
  ppCon.query("sea_water_density",m_seaWaterDensity);
  ppCon.query("gravity",m_gravity);
  ParmParse ppAmr("amr");
  Vector<int> ancells(3); 
  // allows for domains with lower indices which are not positive
  Vector<int> domLoIndex(SpaceDim,0); 
  // slc : SpaceDim == 2 implies poor-mans multidim mode, in which we still
  // care about the number of vertical layers. 
  Vector<Real> domsize(SpaceDim);

  // assumption is that domains are not periodic
  bool is_periodic[SpaceDim];
  for (int dir=0; dir<SpaceDim; dir++)
    is_periodic[dir] = false;
  Vector<int> is_periodic_int(SpaceDim, 0);

  ppAmr.get("maxLevel", m_max_level);
  ppAmr.query("tagCap",m_tag_cap);
  
  ppAmr.getarr("num_cells", ancells, 0, ancells.size());
  
  // this one doesn't have a vertical dimension
  ppAmr.queryarr("domainLoIndex", domLoIndex, 0, SpaceDim);




# if 0
  // this is now handled in main and passed in using the
  // setDomainSize function
  if (ppAmr.contains("domain_size"))
    {
      ppAmr.getarr("domain_size", domsize, 0, SpaceDim);
      m_domainSize = RealVect(D_DECL(domsize[0], domsize[1], domsize[2]));
    }
  
#endif
  
 

  ppAmr.getarr("is_periodic", is_periodic_int, 0, SpaceDim);
  for (int dir=0; dir<SpaceDim; dir++) 
    {
      is_periodic[dir] = (is_periodic_int[dir] == 1);
    }

  ppAmr.query("cfl", m_cfl);

  m_initial_cfl = m_cfl;
  ppAmr.query("initial_cfl", m_initial_cfl);

  ppAmr.query("max_dt_grow_factor", m_max_dt_grow);

  ppAmr.query("time_step_ticks", m_timeStepTicks);
  // max_dt_grow must be at least two in this case - or it will never grow
  if (m_timeStepTicks)
    m_max_dt_grow = std::max(m_max_dt_grow, two);

  ppAmr.query("fixed_dt", m_fixed_dt);
  ppAmr.query("offsetTime", m_offsetTime);


  ppAmr.query("isothermal",m_isothermal);

  ppAmr.query("plot_interval", m_plot_interval);
  ppAmr.query("plot_time_interval", m_plot_time_interval);
  ppAmr.query("plot_prefix", m_plot_prefix);
  ppAmr.query("reduced_plot", m_reduced_plot);
  ppAmr.query("write_map_file", m_write_map_file);

  ppAmr.query("write_preSolve_plotfiles", m_write_presolve_plotfiles);

  ppAmr.query("write_flux_velocities", m_write_fluxVel);
  ppAmr.query("write_viscous_tensor", m_write_viscousTensor);
  ppAmr.query("write_base_velocities", m_write_baseVel);
  ppAmr.query("write_internal_energy", m_write_internal_energy);
  ppAmr.query("write_thickness_sources", m_write_thickness_sources);
  ppAmr.query("write_layer_velocities", m_write_layer_velocities);

  ppAmr.query("evolve_thickness", m_evolve_thickness);
  ppAmr.query("evolve_topography_fix_surface", m_evolve_topography_fix_surface);
  ppAmr.query("evolve_velocity", m_evolve_velocity);
  ppAmr.query("velocity_solve_interval", m_velocity_solve_interval);

  ppAmr.query("grounded_ice_stable", m_grounded_ice_stable);
  ppAmr.query("floating_ice_stable", m_floating_ice_stable);
  ppAmr.query("floating_ice_basal_flux_is_dhdt", m_floating_ice_basal_flux_is_dhdt);
  ppAmr.query("grounded_ice_basal_flux_is_dhdt",m_grounded_ice_basal_flux_is_dhdt);
  ppAmr.query("mask_sources", m_frac_sources);

  ppAmr.query("grounding_line_proximity_scale",m_groundingLineProximityScale);
  ppAmr.query("grounding_line_proximity_calc_type", m_groundingLineProximityCalcType);

  ppAmr.query("check_interval", m_check_interval);
  ppAmr.query("check_prefix", m_check_prefix);
  ppAmr.query("check_overwrite", m_check_overwrite);
  ppAmr.query("check_exit", m_check_exit);

  bool tempBool = m_write_dHDt;
  ppAmr.query("write_dHDt", tempBool);
  m_write_dHDt = (tempBool == 1);
  ppAmr.query("write_mask", m_write_mask);
  ppAmr.query("write_solver_rhs", m_write_solver_rhs);

  if (m_max_level > 0)
    {
      //m_refinement_ratios.resize(m_max_level+1,-1);
      ppAmr.getarr("ref_ratio", m_refinement_ratios, 0, m_max_level);
    }
  else
    {
      m_refinement_ratios.resize(1);
      m_refinement_ratios[0] = -1;
    }

  ppAmr.query("verbosity", s_verbosity);

  ppAmr.get("regrid_interval", m_regrid_interval);
  m_n_regrids = 0;
  
  ppAmr.query("interpolate_zb", m_interpolate_zb);

  ppAmr.query("regrid_thickness_interpolation_method", m_regrid_thickness_interpolation_method);

  ppAmr.get("blockFactor", m_block_factor);

  // int n_tag_subset_boxes = 0;
  // m_tag_subset.define();
  // ppAmr.query("n_tag_subset_boxes",n_tag_subset_boxes);
  // if ( n_tag_subset_boxes > 0)
  //   {
      
  //     Vector<int> corners(2*SpaceDim*n_tag_subset_boxes);
  //     ppAmr.getarr("tag_subset_boxes", corners, 0, corners.size());
  //     int j = 0;
  //     for (int i =0; i < n_tag_subset_boxes; i++)
  // 	{
  // 	  IntVect small;
  // 	  for (int dir = 0; dir < SpaceDim; ++dir)
  // 	    {
  // 	      small[dir] = corners[j++];
  // 	    }
  // 	  IntVect big;
  // 	  for (int dir = 0; dir < SpaceDim; ++dir)
  // 	    {
  // 	      big[dir] = corners[j++];
  // 	    }
  // 	  m_tag_subset |= Box(small,big);
  // 	}
  //   }

 
 


  ppAmr.get("fill_ratio", m_fill_ratio);

  ppAmr.query("nestingRadius", m_nesting_radius);

#ifdef CH_USE_PETSC
  // petsc solvers require nesting radius >= 3
  if (m_nesting_radius < 3)
    {
      MayDay::Warning("PETSC solvers require nesting radius >= 3 -- resetting to 3");
      m_nesting_radius = 3;
    }
#endif

  bool isThereATaggingCriterion = false;
  ppAmr.query("tag_on_grad_velocity", m_tagOnGradVel);
  isThereATaggingCriterion |= m_tagOnGradVel;

  ppAmr.query("tagging_val", m_tagging_val);

  ppAmr.query("tag_on_laplacian_velocity", m_tagOnLapVel);
  isThereATaggingCriterion |= m_tagOnLapVel;

  ppAmr.query("tag_on_grounded_laplacian_velocity", m_tagOnGroundedLapVel);
  isThereATaggingCriterion |= m_tagOnGroundedLapVel;

  // if we set either of these to be true, require that we also provide the threshold
  if (m_tagOnLapVel | m_tagOnGroundedLapVel)
    {
      ppAmr.get("lap_vel_tagging_val", m_laplacian_tagging_val);
      ppAmr.query("lap_vel_tagging_max_basal_friction_coef", m_laplacian_tagging_max_basal_friction_coef);
    }


  ppAmr.query("tag_on_strain_rate_invariant",m_tagOnEpsSqr);
  isThereATaggingCriterion |= m_tagOnEpsSqr;
  // if we set this to be true, require that we also provide the threshold
  if (m_tagOnEpsSqr)
    {
      ppAmr.get("strain_rate_invariant_tagging_val", m_epsSqr_tagVal);
    }


  ppAmr.query("tag_on_velocity_rhs",m_tagOnVelRHS);
  isThereATaggingCriterion |= m_tagOnVelRHS;

  // if we set this to be true, require that we also provide the threshold
  if (m_tagOnVelRHS)
    {
      ppAmr.get("velocity_rhs_tagging_val", m_velRHS_tagVal);
    }
  
  ppAmr.query("tag_grounding_line", m_tagGroundingLine);
  isThereATaggingCriterion |= m_tagGroundingLine;
  // if we set this to be true, require that we also provide the threshold
  if (m_tagGroundingLine)
    {
      ppAmr.get("grounding_line_tagging_min_vel",m_groundingLineTaggingMinVel);
      ppAmr.query("grounding_line_tagging_max_basal_friction_coef", m_groundingLineTaggingMaxBasalFrictionCoef);
    }
  
  
  ppAmr.query("tag_vel_dx", m_tagVelDx);
  isThereATaggingCriterion |= m_tagVelDx;
  // if we set this to be true, require that we also provide the threshold
  if (m_tagVelDx)
    {
      ppAmr.get("vel_dx_tagging_val",m_velDx_tagVal);
      ppAmr.query("vel_dx_finest_level_grounded",m_velDx_tagVal_finestLevelGrounded);
      m_velDx_tagVal_finestLevelFloating = m_velDx_tagVal_finestLevelGrounded;
      ppAmr.query("vel_dx_finest_level_floating",m_velDx_tagVal_finestLevelFloating);
    }

  ppAmr.query("tag_thin_cavity", m_tag_thin_cavity);
  ppAmr.query("tag_thin_cavity_thickness", m_tag_thin_cavity_thickness);

#ifdef HAVE_PYTHON
  ppAmr.query("tag_python", m_tagPython);
  isThereATaggingCriterion |= m_tagPython;
  if (m_tagPython)
    {
      std::string s;
      ppAmr.get("tag_python_module", s);
      PythonInterface::InitializePythonModule
	(&m_tagPythonModule,  s);
      ppAmr.get("tag_python_function",s);
      PythonInterface::InitializePythonFunction
	(&m_tagPythonFunction, m_tagPythonModule , s);
    }
#endif
  
  ppAmr.query("tag_ice_margin", m_tagMargin);
  isThereATaggingCriterion |= m_tagMargin;
  // if we set this to be true, require finest level to tag
  if (m_tagMargin)
    {
      m_margin_tagVal_finestLevel = m_max_level+1;
      ppAmr.query("margin_finest_level",m_margin_tagVal_finestLevel);
    }

  ppAmr.query("tag_all_ice", m_tagAllIce);
  isThereATaggingCriterion |= m_tagAllIce;


  ppAmr.query("tag_entire_domain", m_tagEntireDomain);
  isThereATaggingCriterion |= m_tagEntireDomain;


  ppAmr.query("tag_on_div_H_grad_vel",m_tagOndivHgradVel);
  isThereATaggingCriterion |= m_tagOndivHgradVel;

  // if we set this to be true, require that we also provide the threshold
  if (m_tagOndivHgradVel)
    {
      ppAmr.get("div_H_grad_vel_tagging_val", m_divHGradVel_tagVal);
    }


  // here is a good place to set default to grad(vel)
  //if ((!m_tagOnGradVel) && (!m_tagOnLapVel) && (!m_tagOnEpsSqr))
  if (!isThereATaggingCriterion)
    {
      m_tagOnGradVel = true;
    }
  
  ppAmr.query("tags_grow", m_tags_grow);
  {
    Vector<int> tgd(SpaceDim,0);
    ppAmr.queryarr("tags_grow_dir", tgd, 0, tgd.size());
    for (int dir =0; dir < SpaceDim; dir++)
      {
	m_tags_grow_dir[dir] = tgd[dir];
      } 
  }
  ppAmr.query("max_box_size", m_max_box_size);

  if (ppAmr.contains("max_base_grid_size") )
    {
      ppAmr.get("max_base_grid_size", m_max_base_grid_size);
    }
  else 
    {
      m_max_base_grid_size = m_max_box_size;
    }

  ppAmr.query("report_sum_grounded_ice",   m_report_grounded_ice);

  ppAmr.query("report_ice_area",   m_report_area);

  ppAmr.query("report_total_flux", m_report_total_flux);

  ppAmr.query("report_calving", m_report_calving);

  ppAmr.query("report_discharge", m_report_discharge);

  ppAmr.query("report_time_interval", m_report_time_interval);
  
  ppAmr.query("eliminate_remote_ice", m_eliminate_remote_ice);
  ppAmr.query("eliminate_remote_ice_max_iter", m_eliminate_remote_ice_max_iter);
  ppAmr.query("eliminate_remote_ice_tol", m_eliminate_remote_ice_tol);
  ppAmr.query("eliminate_remote_ice_after_regrid", m_eliminate_remote_ice_after_regrid);

  // get temporal accuracy
  ppAmr.query("temporal_accuracy", m_temporalAccuracy);

  // number of ghost cells depends on what scheme we're using
  if (m_temporalAccuracy < 3)
    {
      m_num_thickness_ghost = 4;
    }
  else 
    {
      m_num_thickness_ghost = 1;      
    }

  // get solver type
  ppAmr.query("velocity_solver_type", m_solverType);

  ppAmr.query("max_solver_iterations",m_maxSolverIterations);

  ppAmr.query("velocity_solver_tolerance", m_velocity_solver_tolerance);

  ppAmr.query("limit_velocity_rhs", m_limitVelRHS);
  ppAmr.query("limit_rhs_radius", m_gradLimitRadius);

  ppAmr.query("limit_flux_speed",m_limitFluxSpeed);

  ppAmr.query("do_initial_velocity_solve", m_doInitialVelSolve);
  ppAmr.query("do_initial_velocity_guess", m_doInitialVelGuess);
  ppAmr.query("initial_velocity_guess_type", m_initialGuessType);
  ppAmr.query("initial_velocity_guess_const_mu", m_initialGuessConstMu);
  ppAmr.query("initial_velocity_guess_solver_type", m_initialGuessSolverType);

  {
    Vector<Real> t(SpaceDim,0.0);
    ppAmr.queryarr("initial_velocity_guess_const_vel", t, 0, SpaceDim);
    m_initialGuessConstVel = RealVect(D_DECL(t[0], t[1], t[2]));
  }

  ppAmr.query("additional_velocity",m_additionalVelocity);


  //thickness diffusion options
  std::string diffusionTreatment = "none";
  ppAmr.query("diffusion_treatment", diffusionTreatment);
  if (diffusionTreatment == "implicit")
    {
      m_diffusionTreatment = IMPLICIT;
    }
  else if (diffusionTreatment == "explicit")
    m_diffusionTreatment = EXPLICIT;
  ppAmr.query("additional_diffusivity",m_additionalDiffusivity);


  //option to advance thickness/internalEnergy only on coarser levels
  ppAmr.query("finest_timestep_level",m_finest_timestep_level);

  ppAmr.query("reset_floating_friction", m_reset_floating_friction_to_zero);
  ppAmr.query("basal_length_scale", m_basalLengthScale);
 
  ppAmr.query("wallDrag",m_wallDrag);
  ppAmr.query("wallDragExtra",m_wallDragExtra);

  ppAmr.query("grounding_line_correction",m_groundingLineCorrection);
  ppAmr.query("grounding_line_subdivision", m_groundingLineSubdivision);

  //calving model options
  m_calvingModelPtr = CalvingModel::parseCalvingModel("CalvingModel");
  if (m_calvingModelPtr == NULL)
    {
      MayDay::Warning("trying to parse old style amr.calving_model_type");

      std::string calvingModelType = "NoCalvingModel";
      ppAmr.query("calving_model_type",calvingModelType);
      if (calvingModelType == "NoCalvingModel")
	{
	  m_calvingModelPtr = new NoCalvingModel;
	}
      else if (calvingModelType == "DeglaciationCalvingModelA")
	{
	  ParmParse ppc("DeglaciationCalvingModelA");
	  Real minThickness = 0.0;
	  ppc.get("min_thickness", minThickness );
	  Real calvingThickness = 0.0;
	  ppc.get("calving_thickness", calvingThickness );
	  Real calvingDepth = 0.0;
	  ppc.get("calving_depth", calvingDepth );
	  Real startTime = -1.2345678e+300;
	  ppc.query("start_time",  startTime);
	  Real endTime = 1.2345678e+300;
	  ppc.query("end_time",  endTime);
	  DeglaciationCalvingModelA* ptr = new DeglaciationCalvingModelA
	    (calvingThickness,  calvingDepth, minThickness, startTime, endTime);
	  m_calvingModelPtr = ptr;
	  
	}
      else if (calvingModelType == "DomainEdgeCalvingModel")
	{
	  ParmParse ppc("DomainEdgeCalvingModel");
	  Vector<int> frontLo(2,false); 
	  ppc.getarr("front_lo",frontLo,0,frontLo.size());
	  Vector<int> frontHi(2,false);
	  ppc.getarr("front_hi",frontHi,0,frontHi.size());
	  bool preserveSea = false;
	  ppc.query("preserveSea",preserveSea);
	  bool preserveLand = false;
	  ppc.query("preserveLand",preserveLand);
	  DomainEdgeCalvingModel* ptr = new DomainEdgeCalvingModel
	    (frontLo, frontHi,preserveSea,preserveLand);
	  m_calvingModelPtr = ptr;
	}
      else
	{
	  MayDay::Error("Unknown calving model");
	}
    }

  // now set up problem domains
  {
    IntVect loVect = IntVect(D_DECL(domLoIndex[0], domLoIndex[1], domLoIndex[3]));
    IntVect hiVect(D_DECL(domLoIndex[0]+ancells[0]-1, 
                          domLoIndex[1]+ancells[1]-1, 
                          domLoIndex[2]+ancells[2]-1));
#if BISICLES_Z == BISICLES_LAYERED
    {
      int nLayers = ancells[2];
      Vector<Real> faceSigma(nLayers+1);
      Real dsigma = 1.0 / Real(nLayers);
      for (unsigned int l = 0; l < faceSigma.size(); ++l)
	faceSigma[l] = dsigma * (Real(l));
      {
	ParmParse ppGeo("geometry");
	ppGeo.queryarr("sigma",faceSigma,0,faceSigma.size());
      }
      ppAmr.queryarr("sigma",faceSigma,0,faceSigma.size());
      setLayers(faceSigma);
    }
#endif
    ProblemDomain baseDomain(loVect, hiVect);
    // now set periodicity
    for (int dir=0; dir<SpaceDim; dir++) 
      baseDomain.setPeriodic(dir, is_periodic[dir]);

    // now set up vector of domains
    m_amrDomains.resize(m_max_level+1);
    m_amrDx.resize(m_max_level+1);

    m_amrDomains[0] = baseDomain;
    m_amrDx[0] = m_domainSize[0]/baseDomain.domainBox().size(0);

    for (int lev=1; lev<= m_max_level; lev++)
      {
        m_amrDomains[lev] = refine(m_amrDomains[lev-1],
                                   m_refinement_ratios[lev-1]);
        m_amrDx[lev] = m_amrDx[lev-1]/m_refinement_ratios[lev-1];
      }
  } // leaving problem domain setup scope
  
  std::string tagSubsetBoxesFile = "";
  m_vectTagSubset.resize(m_max_level);
  
  ppAmr.query("tagSubsetBoxesFile",tagSubsetBoxesFile);
  
  if (tagSubsetBoxesFile != "")
    {
      if (procID() == uniqueProc(SerialTask::compute))
  	{
	 
  	  ifstream is(tagSubsetBoxesFile.c_str(), ios::in);
  	  int lineno = 1;
  	  if (is.fail())
  	    {
  	      pout() << "Can't open " << tagSubsetBoxesFile << std::endl;
  	      MayDay::Error("Cannot open refine boxes file");
  	    }

  	  for (int lev = 0; lev < m_max_level; lev++)
  	    {
              // allowable tokens to identify levels in tag subset file
  	      const char level[6] = "level";
              const char domain[7] = "domain";
  	      char s[6];
  	      is >> s;
  	      if (std::string(level) == std::string(s))
  		{
                  int inlev;
                  is >> inlev;
                  if (inlev != lev)
                    {
                      pout() << "expected ' " << lev << "' at line " << lineno << std::endl;
                      MayDay::Error("bad input file");
                    }
                } 
              else if (std::string(domain) == std::string(s))
                {
                  // basic idea here is that we read in domain box
                  // (domains must be ordered from coarse->fine)
                  // until we get to a domain box which matches ours.
                  // This lets us make a single list of subset regions
                  // which we can use for any coarsening/refining of the domain
                  const Box& levelDomainBox = m_amrDomains[lev].domainBox();
                  bool stillLooking = true;
                  while (stillLooking)
                    {
                      Box domainBox;
                      is >> domainBox;
                      if (domainBox == levelDomainBox)
                        {
                          pout() << "Found a domain matching level " << lev << endl;
                          stillLooking = false;
                        }
                      else // move on until we find our level
                        {
                          // read in info for the level we're ignoring
                          //advance to next line
                          while (is.get() != '\n');
			  lineno++;
                          int nboxes;
                          is >> nboxes;
                          if (nboxes > 0)
                            {
                              for (int i = 0; i < nboxes; ++i)
                                {
                                  Box box;
                                  is >> box;
				  while (is.get() != '\n');
				  lineno++;
                                }
                            } 
                          is >> s;
                          if (std::string(domain) != std::string(s))
                            {
                              pout() << "expected '" << domain
                                     << "' at line " << lineno << ", got " 
                                     << s << std::endl;
                              MayDay::Error("bad input file");
                            }                            
                        }
                    }
                }
              else
                {
  		  pout() << "expected '" << level << "' or '" << domain
                         << "' at line " << lineno << ", got " 
                         << s << std::endl;
  		  MayDay::Error("bad input file");
  		}
              //advance to next line
              while (is.get() != '\n');
	      lineno++;
              int nboxes;
              is >> nboxes;
              if (nboxes > 0)
                {
                  for (int i = 0; i < nboxes; ++i)
                    {
                      Box box;
                      is >> box;
		      while (is.get() != '\n');
		      lineno++;
                      m_vectTagSubset[lev] |= box;
                      pout() << " level " << lev << " refine box : " << box << std::endl;
                    }
                }
              //advance to next line
              while (is.get() != '\n');
	      lineno++;
              
              if (lev > 0)
		{
		  //add lower level's subset to this subset
		  IntVectSet crseSet (m_vectTagSubset[lev-1]);
		  if (!crseSet.isEmpty())
		    {
		      crseSet.refine(m_refinement_ratios[lev-1]);
		      // crseSet.nestingRegion(m_block_factor,m_amrDomains[lev]);
		      if (m_vectTagSubset[lev].isEmpty())
			{
			  m_vectTagSubset[lev] = crseSet;
			} 
		      else
			{
			  m_vectTagSubset[lev] &= crseSet;
			} 
		    }
		 
		}
	      
  	    } // end loop over levels

	} // end if serial compute
      for (int lev = 0; lev < m_max_level; lev++)
	broadcast(m_vectTagSubset[lev], uniqueProc(SerialTask::compute));
    }
  /// PatchGodunov used for thickness advection
  if (m_temporalAccuracy < 3)
    {
      // get PatchGodunov options -- first set reasonable defaults.
      // can over-ride from ParmParse
      int normalPredOrder = 2;
      bool useFourthOrderSlopes = true;
      bool usePrimLimiting = true;
      bool useCharLimiting = false;
      bool useFlattening = false;
      bool useArtificialViscosity = false;
      Real artificialViscosity = 0.0;
      
      // define AdvectionPhysics ptr
      // (does this need to be a special Ice-advection pointer due
      // to H factors?      
      
      m_thicknessPhysPtr = new AdvectPhysics;
      m_thicknessPhysPtr->setPhysIBC(m_thicknessIBCPtr);
    
      

      m_thicknessPatchGodVect.resize(m_max_level+1, NULL);

      for (int lev=0; lev<=m_max_level; lev++)
        {


          m_thicknessPatchGodVect[lev] = new PatchGodunov;
          m_thicknessPatchGodVect[lev]->define(m_amrDomains[lev],
                                               m_amrDx[lev],
                                               m_thicknessPhysPtr,
                                               normalPredOrder,
                                               useFourthOrderSlopes,
                                               usePrimLimiting,
                                               useCharLimiting,
                                               useFlattening,
                                               useArtificialViscosity,
                                               artificialViscosity);
        }
      

      //m_internalEnergyIBCPtr = new IceInternalEnergyIBC;
     

    } // end if temporal accuracy < 3
  
 

  // check to see if we're using predefined grids
  bool usePredefinedGrids = false;
  std::string gridFile;
  if (ppAmr.contains("gridsFile"))
    {
      usePredefinedGrids = true;
      ppAmr.get("gridsFile",gridFile);
    }

  
  ParmParse geomPP("geometry");
  if (geomPP.contains("basalSlope") )
    {
      Vector<Real> basalSlope(SpaceDim, 0.0);
      geomPP.getarr("basalSlope", basalSlope, 0, SpaceDim);
      D_TERM(
             m_basalSlope[0] = basalSlope[0];,
             m_basalSlope[1] = basalSlope[1];,
             m_basalSlope[2] = basalSlope[2];);
    }


  // check to see if we're restarting from a checkpoint file
  if (!ppAmr.contains("restart_file"))
    {
      // if we're not restarting
      
      // now set up data holders
      m_old_thickness.resize(m_max_level+1, NULL);
      m_velocity.resize(m_max_level+1, NULL);
      m_iceFrac.resize(m_max_level+1, NULL);
      m_faceVelAdvection.resize(m_max_level+1, NULL);
      m_faceVelTotal.resize(m_max_level+1, NULL);
      m_diffusivity.resize(m_max_level+1);
      m_velBasalC.resize(m_max_level+1,NULL);
      m_cellMuCoef.resize(m_max_level+1,NULL);
      m_velRHS.resize(m_max_level+1, NULL);
      m_surfaceThicknessSource.resize(m_max_level+1, NULL);
      m_basalThicknessSource.resize(m_max_level+1, NULL);
      m_calvedIceThickness.resize(m_max_level+1, NULL);
      m_removedIceThickness.resize(m_max_level+1, NULL);
      m_addedIceThickness.resize(m_max_level+1, NULL);
      m_melangeThickness.resize(m_max_level+1, NULL);
      m_recordThickness.resize(m_max_level+1, NULL);
      m_divThicknessFlux.resize(m_max_level+1, NULL);
      m_internalEnergy.resize(m_max_level+1, NULL);
      m_deltaTopography.resize(m_max_level+1, NULL);
#if BISICLES_Z == BISICLES_LAYERED
      m_layerXYFaceXYVel.resize(m_max_level+1, NULL);
      m_layerSFaceXYVel.resize(m_max_level+1, NULL);
      m_sInternalEnergy.resize(m_max_level+1, NULL);
      m_bInternalEnergy.resize(m_max_level+1, NULL);
      m_sHeatFlux.resize(m_max_level+1, NULL);
      m_bHeatFlux.resize(m_max_level+1, NULL);
#endif
      // allocate storage for m_old_thickness,  m_velocity, etc
      for (int lev=0; lev<m_velocity.size(); lev++)
        {
          m_old_thickness[lev] = new LevelData<FArrayBox>;
          m_velocity[lev] = new LevelData<FArrayBox>;
          m_iceFrac[lev] = new LevelData<FArrayBox>;
	  m_faceVelAdvection[lev] = new LevelData<FluxBox>;
	  m_faceVelTotal[lev] = new LevelData<FluxBox>;
	  m_velBasalC[lev] = new LevelData<FArrayBox>;
	  m_cellMuCoef[lev] = new LevelData<FArrayBox>;
	  m_velRHS[lev] = new LevelData<FArrayBox>;
	  m_diffusivity[lev] = new LevelData<FluxBox>;
	  m_internalEnergy[lev] = new LevelData<FArrayBox>;
	  m_deltaTopography[lev] = new LevelData<FArrayBox>;
#if BISICLES_Z == BISICLES_LAYERED
	  m_sInternalEnergy[lev] = new LevelData<FArrayBox>;
	  m_bInternalEnergy[lev] = new LevelData<FArrayBox>;
	  m_sHeatFlux[lev] = new LevelData<FArrayBox>;
	  m_bHeatFlux[lev] = new LevelData<FArrayBox>;
	  m_layerXYFaceXYVel[lev] = new LevelData<FluxBox>;
	  m_layerSFaceXYVel[lev] = new LevelData<FArrayBox>;
#endif

        }

      int finest_level = -1;
      if (usePredefinedGrids)
        {
          setupFixedGrids(gridFile);
        } 
      else
        {
          // now create  grids
          initGrids(finest_level);
        }
      
      // last thing to do is to set this to true from here on out...
      m_doInitialVelSolve = true;

      // that should be it
    }
  else
    {
      // we're restarting from a checkpoint file
      string restart_file;
      ppAmr.get("restart_file", restart_file);
      m_do_restart = true;
#ifdef CH_USE_HDF5
      restart(restart_file);
      // once we've set up everything, this lets us over-ride the
      // time and step number in the restart checkpoint file with
      // one specified in the inputs      
      
      
      if (ppAmr.contains("restart_time") )
        {
	  bool set_time = true;
	  ppAmr.query("restart_set_time",set_time); // set amr.restart_set_time = false to prevent time reset
	  if (set_time){
	    Real restart_time;
	    ppAmr.get("restart_time", restart_time);
	    m_time = restart_time;
	  }
        }
      
      if (ppAmr.contains("restart_step") )
        {
          int restart_step;
          ppAmr.get("restart_step", restart_step);
          m_cur_step = restart_step;
          m_restart_step = restart_step;
        }
#endif // hdf5
    }


  // set up counter of number of cells
  m_num_cells.resize(m_max_level+1, 0);
  for (int lev=0; lev<=m_finest_level; lev++)
    {
      const DisjointBoxLayout& levelGrids = m_amrGrids[lev];
      LayoutIterator lit = levelGrids.layoutIterator();
      for (lit.begin(); lit.ok(); ++lit)
        {
          const Box& thisBox = levelGrids.get(lit());
          m_num_cells[lev] += thisBox.numPts();
        }
    }


  // finally, set up covered_level flags
  m_covered_level.resize(m_max_level+1, 0);

  // note that finest level can't be covered.
  for (int lev=m_finest_level-1; lev>=0; lev--)
    {

      // if the next finer level is covered, then this one is too.
      if (m_covered_level[lev+1] == 1)
        {
          m_covered_level[lev] = 1;
        }
      else
        {
          // see if the grids finer than this level completely cover it
          IntVectSet fineUncovered(m_amrDomains[lev+1].domainBox());
          const DisjointBoxLayout& fineGrids = m_amrGrids[lev+1];

          LayoutIterator lit = fineGrids.layoutIterator();
          for (lit.begin(); lit.ok(); ++lit)
            {
              const Box& thisBox = fineGrids.get(lit());
              fineUncovered.minus_box(thisBox);
            }

          if (fineUncovered.isEmpty()) 
            {
              m_covered_level[lev] = 1;
            }
        }
    } // end loop over levels to determine covered levels

  m_initialSumIce = computeTotalIce();
  m_lastSumIce = m_initialSumIce;
  if (m_report_grounded_ice)
    {
      m_initialSumGroundedIce = computeTotalGroundedIce();
      m_lastSumGroundedIce = m_initialSumGroundedIce;
      m_initialVolumeAboveFlotation = computeVolumeAboveFlotation();
      m_lastVolumeAboveFlotation = m_initialVolumeAboveFlotation; 
    }
  if (m_report_calving)
    {
      m_lastSumCalvedIce = computeSum(m_melangeThickness, m_refinement_ratios,m_amrDx[0],
				      Interval(0,0), 0);
    }


}  
  
/// set BC for thickness advection
void
AmrIce::setThicknessBC( IceThicknessIBC* a_thicknessIBC)
{
  m_thicknessIBCPtr = a_thicknessIBC->new_thicknessIBC(); 
}

/// set BC for internalEnergy advection
void 
AmrIce::setInternalEnergyBC( IceInternalEnergyIBC* a_internalEnergyIBC)
{
  m_internalEnergyIBCPtr = a_internalEnergyIBC->new_internalEnergyIBC();
}

void 
AmrIce::defineSolver()
{
  if (m_solverType == Picard)
    {
      MayDay::Error("PicardSolver is deprecated (for now)");
      
      // for now, at least, just delete any existing solvers
      // and rebuild them from scratch
      if (m_velSolver != NULL)
        {
          delete m_velSolver;
          m_velSolver = NULL;
        }

      m_velSolver = new PicardSolver;
      
      RealVect dxCrse = m_amrDx[0]*RealVect::Unit;

      int numLevels = m_finest_level +1;

      // make sure that the IBC has the correct grid hierarchy info
      m_thicknessIBCPtr->setGridHierarchy(m_vect_coordSys, m_amrDomains);

      m_velSolver->define(m_amrDomains[0],
                          m_constitutiveRelation,
			  m_basalFrictionRelation,
                          m_amrGrids,
                          m_refinement_ratios,
                          dxCrse,
                          m_thicknessIBCPtr,
                          numLevels);
      m_velSolver->setVerbosity(s_verbosity);

      m_velSolver->setTolerance(m_velocity_solver_tolerance);

      if (m_maxSolverIterations > 0)
        {
          m_velSolver->setMaxIterations(m_maxSolverIterations);
        }
    }
  else  if (m_solverType == JFNK)
    {
      // for now, at least, just delete any existing solvers
      // and rebuild them from scratch
     
      JFNKSolver* jfnkSolver;

      RealVect dxCrse = m_amrDx[0]*RealVect::Unit;
      int numLevels = m_finest_level +1;
      
      // make sure that the IBC has the correct grid hierarchy info
      m_thicknessIBCPtr->setGridHierarchy(m_vect_coordSys, m_amrDomains);

      if (m_velSolver != NULL)
	{
	  // assume that any extant solver is also a JFNKSolver
	  jfnkSolver = dynamic_cast<JFNKSolver*>(m_velSolver);
	  CH_assert(jfnkSolver != NULL);
	}
      else {
	jfnkSolver = new JFNKSolver();
      }
      
      jfnkSolver->define(m_amrDomains[0],
			 m_constitutiveRelation,
			 m_basalFrictionRelation,
			 m_amrGrids,
			 m_refinement_ratios,
			 dxCrse,
			 m_thicknessIBCPtr,
			 numLevels);

      m_velSolver = jfnkSolver;

    }
  else  if (m_solverType == KnownVelocity)
    {
      if (m_velSolver != NULL)
        {
          delete m_velSolver;
          m_velSolver = NULL;
        }
      m_velSolver = new KnownVelocitySolver;
      RealVect dxCrse = m_amrDx[0]*RealVect::Unit;
      int numLevels = m_finest_level +1;
      m_velSolver->define(m_amrDomains[0],
                          m_constitutiveRelation,
			  m_basalFrictionRelation,
                          m_amrGrids,
                          m_refinement_ratios,
                          dxCrse,
                          m_thicknessIBCPtr,
                          numLevels);
    }
#ifdef CH_USE_PETSC
  else if (m_solverType == PetscNLSolver)
    {
      // for now, at least, just delete any existing solvers
      // and rebuild them from scratch
      if (m_velSolver != NULL)
        {
          delete m_velSolver;
          m_velSolver = NULL;
        }

      m_velSolver = new PetscIceSolver;
      
      RealVect dxCrse = m_amrDx[0]*RealVect::Unit;

      int numLevels = m_finest_level +1;

      // make sure that the IBC has the correct grid hierarchy info
      m_thicknessIBCPtr->setGridHierarchy(m_vect_coordSys, m_amrDomains);

      m_velSolver->define(m_amrDomains[0],
                          m_constitutiveRelation,
			  m_basalFrictionRelation,
                          m_amrGrids,
                          m_refinement_ratios,
                          dxCrse,
                          m_thicknessIBCPtr,
                          numLevels);
      m_velSolver->setVerbosity(s_verbosity);

      m_velSolver->setTolerance(m_velocity_solver_tolerance);

      if (m_maxSolverIterations > 0)
        {
          m_velSolver->setMaxIterations(m_maxSolverIterations);
        }
    }
#endif
#ifdef CH_USE_FAS
  else if (m_solverType == FASMGAMR)
    {
      // for now, at least, just delete any existing solvers
      // and rebuild them from scratch
      if (m_velSolver != NULL)
        {
          delete m_velSolver;
          m_velSolver = NULL;
        }
      
      FASIceSolver *solver = new FASIceSolver;
      m_velSolver = solver;
      
      solver->setParameters( "FASSolver" );

      RealVect dxCrse = m_amrDx[0]*RealVect::Unit;

      int numLevels = m_finest_level + 1;

      // make sure that the IBC has the correct grid hierarchy info
      m_thicknessIBCPtr->setGridHierarchy( m_vect_coordSys, m_amrDomains );

      solver->define( m_amrDomains[0],
		      m_constitutiveRelation,
		      m_basalFrictionRelation,
		      m_amrGrids,
		      m_refinement_ratios,
		      dxCrse,
		      m_thicknessIBCPtr,
		      numLevels );

      solver->setTolerance( m_velocity_solver_tolerance );

      if (m_maxSolverIterations > 0)
        {
          solver->setMaxIterations( m_maxSolverIterations );
        }
    }
#endif
#ifdef HAVE_PYTHON
  else if (m_solverType == Python)
    {
      // for now, at least, just delete any existing solvers
      // and rebuild them from scratch
      if (m_velSolver != NULL)
        {
          delete m_velSolver;
          m_velSolver = NULL;
        }
      m_velSolver = new PythonInterface::PythonVelocitySolver;
      m_velSolver->define( m_amrDomains[0],
			   m_constitutiveRelation,
			   m_basalFrictionRelation,
			   m_amrGrids,
			   m_refinement_ratios,
			   m_amrDx[0]*RealVect::Unit,
			   m_thicknessIBCPtr,
			   m_finest_level + 1 );
    }
#endif
  else if (m_solverType == InverseVerticallyIntegrated)
    {
      if (m_velSolver != NULL)
        {
	  //not sure if we are OK with rebuilding solvers?
          delete m_velSolver;
          m_velSolver = NULL;
        }
      InverseVerticallyIntegratedVelocitySolver* ptr 
	= new InverseVerticallyIntegratedVelocitySolver;
      
      ptr->define( *this, 
		   m_amrDomains[0],
		   m_constitutiveRelation,
		   m_basalFrictionRelation,
		   m_amrGrids,
		   m_refinement_ratios,
		   m_amrDx[0]*RealVect::Unit,
		   m_thicknessIBCPtr,
		   m_finest_level + 1 );
      m_velSolver = ptr;
      
    }
  else
    {
      MayDay::Error("unsupported velocity solver type");
    }
 
}

//inline 
//Real remainder(Real a, Real b)
//{
//  Real p = a/b; int i(p);
//  return std::min( p - i, p - 1 - i);
//}

void AmrIce::setToZero(Vector<LevelData<FArrayBox>*>& a_data)
{
  for (int lev=0; lev < std::min(int(a_data.size()),finestLevel()+1); lev++)
    {
      LevelData<FArrayBox>& data = *a_data[lev];
      for (DataIterator dit(m_amrGrids[lev]); dit.ok(); ++dit)
	{
	  data[dit].setVal(0.0);
	}
    }
}


void
AmrIce::run(Real a_max_time, int a_max_step)
{

  if (s_verbosity > 3) 
    {
      pout() << "AmrIce::run -- max_time= " << a_max_time 
             << ", max_step = " << a_max_step << endl;
    }

  Real dt;
  // only call computeInitialDt if we're not doing restart
  if (!m_do_restart)
    {
      dt = computeInitialDt();
    } 
  else
    {
      dt = computeDt();
    }

  
  // advance solution until done
  if ( !(m_plot_time_interval > TIME_EPS) || m_plot_time_interval > a_max_time) m_plot_time_interval = a_max_time;
  if ( !(m_report_time_interval > TIME_EPS) || m_report_time_interval > a_max_time) m_report_time_interval = a_max_time;
  while ( a_max_time > m_time && (m_cur_step < a_max_step))
    {
      Real next_plot_time = m_plot_time_interval * (1.0 + Real(int((m_time/m_plot_time_interval))));
      if ( !(next_plot_time > m_time))
	{
	  //trap case where machine precision results in (effectively)
          // m_plot_time_interval * (1.0 + Real(int((m_time/m_plot_time_interval)))) == m_time
	  next_plot_time += m_plot_time_interval;
	}

      next_plot_time = std::min(next_plot_time, a_max_time); 

      m_next_report_time = m_time;
      m_next_report_time = std::min(m_next_report_time, a_max_time); 
 
      while ( (next_plot_time > m_time) && (m_cur_step < a_max_step)
	      && (dt > TIME_EPS))
	{
	  
	  // dump plotfile before regridding
	  if ( (m_cur_step%m_plot_interval == 0) && m_plot_interval > 0)
	    {
#ifdef CH_USE_HDF5
	      writePlotFile();
#endif
	    }
	  
	  setToZero(m_calvedIceThickness); 
	  setToZero(m_removedIceThickness);
	  setToZero(m_addedIceThickness);

	  if ((m_cur_step != 0) && (m_cur_step%m_regrid_interval ==0))
	    {
	      regrid();
	    }
	  
	  if (m_cur_step != 0)
	    {
	      // compute dt after regridding in case number of levels has changed
	      dt = computeDt();           
	    }
	  
	  //Real trueDt = dt; //we will need to restore dt if we change it below
	  if (next_plot_time - m_time + TIME_EPS < dt) 
	    dt =  std::max(2 * TIME_EPS, next_plot_time - m_time);
	  
	  if ((m_cur_step%m_check_interval == 0) && (m_check_interval > 0)
	      && (m_cur_step != m_restart_step))
	    {
#ifdef CH_USE_HDF5
	      writeCheckpointFile();
#endif
	      if (m_cur_step > 0 && m_check_exit)
		{
		  if (s_verbosity > 2)
		    {
		      pout() << "AmrIce::exit on checkpoint" << endl;
		      return;
		    }
		}


	    }
	  
	  
	  timeStep(dt);
	  //m_dt = trueDt; 
	  // restores the correct timestep in cases where it was chosen just to reach a plot file
	  
	} // end of plot_time_interval
#ifdef CH_USE_HDF5
      if (m_plot_interval >= 0)
	writePlotFile();
#endif
    } // end timestepping loop

  // dump out final plotfile, if appropriate
#ifdef CH_USE_HDF5

  if (m_plot_interval >= 0)
    {
      writePlotFile();
    }
  
  // dump out final checkpoint file, if appropriate
  if (m_check_interval >= 0)
    {
      writeCheckpointFile();
    }
#endif    
  
  if (s_verbosity > 2)
    {
      pout() << "AmrIce::run finished" << endl;
    }
}


void
AmrIce::timeStep(Real a_dt)
{

  if (s_verbosity >=2) 
    {
      pout() << "Timestep " << m_cur_step 
             << " Advancing solution from time " 
             << m_time << " ( " << time() << ")" " with dt = " << a_dt << endl;
    }

  m_dt = a_dt;

  // first copy thickness into old thickness   
  for (int lev=0; lev <= m_finest_level ; lev++)
    {
      
      LevelData<FArrayBox>& oldThickness = *m_old_thickness[lev];
      LevelData<FArrayBox>& currentThickness = m_vect_coordSys[lev]->getH();

      // this way we avoid communication and maintain ghost cells...
      DataIterator dit = oldThickness.dataIterator();
      for (dit.begin(); dit.ok(); ++dit)
        {
          oldThickness[dit].copy(currentThickness[dit],0, 0, 1);
        }

    }
        
  // assumption here is that we've already computed the current velocity 
  // field, most likely at initialization or at the end of the last timestep...
  // so, we don't need to recompute the velocity at the start.

  // use PatchGodunov hyperbolic solver
  
#if 0 // this doesn't appear to be used anywhere anymore
  // need a grown velocity field
  IntVect grownVelGhost(2*IntVect::Unit);
  Vector<LevelData<FArrayBox>* > grownVel(m_finest_level+1, NULL);
#endif

  // holder for half-time face velocity
  Vector<LevelData<FluxBox>* > H_half(m_finest_level+1,NULL);
  // thickness fluxes 
  Vector<LevelData<FluxBox>* > vectFluxes(m_finest_level+1, NULL);
  
  
  // allocate storage
  for (int lev = finestTimestepLevel() ; lev>=0 ; lev--)
    {
      
      const DisjointBoxLayout& levelGrids = m_amrGrids[lev];

      IntVect ghostVect = IntVect::Unit;      
      H_half[lev] = new LevelData<FluxBox>(m_amrGrids[lev], 1, 
                                           ghostVect);

      // if we're doing AMR, we'll need to average these fluxes
      // down to coarser levels. As things stand now, 
      // CoarseAverageFace requires that the coarse LevelData<FluxBox>
      // have a ghost cell. 
      vectFluxes[lev] = new LevelData<FluxBox>(m_amrGrids[lev],1, ghostVect);

      LevelData<FArrayBox>& levelOldThickness = *m_old_thickness[lev];
      
      
      
      // ensure that ghost cells for thickness  are filled in
      if (lev > 0)
        {          
          int nGhost = levelOldThickness.ghostVect()[0];
          PiecewiseLinearFillPatch thicknessFiller(levelGrids, 
                                                   m_amrGrids[lev-1],
                                                   1, 
                                                   m_amrDomains[lev-1],
                                                   m_refinement_ratios[lev-1],
                                                   nGhost);
          
          // since we're not subcycling, don't need to interpolate in time
          Real time_interp_coeff = 0.0;
          thicknessFiller.fillInterp(levelOldThickness,
                                     *m_old_thickness[lev-1],
                                     *m_old_thickness[lev-1],
                                     time_interp_coeff,
                                     0, 0, 1);
          
          
          
        }
      // these are probably unnecessary...
      levelOldThickness.exchange();
      
      
      // do we need to also do a coarseAverage for the vel here?
    }
    // compute face-centered thickness (H) at t + dt/2
  computeH_half(H_half, a_dt);
  
  //  compute face- and layer- centered E*H and H  at t + dt/2 (E is internal energy)
  Vector<LevelData<FluxBox>* > layerEH_half(m_finest_level+1,NULL);
  Vector<LevelData<FluxBox>* > layerH_half(m_finest_level+1,NULL);
  if (!m_isothermal)
    computeInternalEnergyHalf(layerEH_half, layerH_half, m_layerXYFaceXYVel, a_dt, m_time);
  
  // Having found H_half we can define temporary LevelSigmaCS at t + dt / 2
  // We want this for the metric terms that appear in the internal energy advection, 
  // and also when m_temporalAccuracy == 2 to compute a new velocity field 
  Vector<RefCountedPtr<LevelSigmaCS> > vectCoords_half (m_finest_level+1);
  
  if (m_temporalAccuracy == 1)
    {
      // just use the old time LevelSigmaCS
      for (int lev=0; lev<= m_finest_level; lev++)
        vectCoords_half[lev] = m_vect_coordSys[lev];
    }
  else if (m_temporalAccuracy == 2)
    {
      for (int lev=0; lev<= finestTimestepLevel(); lev++)
        {
          IntVect sigmaCSGhost = m_vect_coordSys[lev]->ghostVect();
          RealVect dx = m_amrDx[lev]*RealVect::Unit;
          vectCoords_half[lev] = RefCountedPtr<LevelSigmaCS> 
            (new LevelSigmaCS(m_amrGrids[lev], dx, sigmaCSGhost));
          LevelSigmaCS& levelCoords_half = *vectCoords_half[lev];
          LevelSigmaCS& levelCoords = *m_vect_coordSys[lev];
	  
          ///todo : Here, assume that the base height doesn't change during the
          ///timestep, which is not strictly true. Instead, we should perform 
          ///an isostasy calculation at this point.
          levelCoords_half.setTopography(levelCoords.getTopography());
          levelCoords_half.setFaceSigma(levelCoords.getFaceSigma());
          levelCoords_half.setIceDensity(levelCoords.iceDensity());
          levelCoords_half.setGravity(levelCoords.gravity());
          levelCoords_half.setWaterDensity(levelCoords.waterDensity());
          //now set the thickness from H_half
          LevelData<FluxBox>& levelH = *H_half[lev];
          LevelData<FluxBox>& levelFaceH = levelCoords_half.getFaceH();
          for (DataIterator dit( m_amrGrids[lev]); dit.ok(); ++dit)
            {
              FluxBox& faceH = levelFaceH[dit];
              faceH.copy(levelH[dit], levelH[dit].box());
            }
          {
            LevelSigmaCS* crseCoords = (lev > 0)?&(*vectCoords_half[lev-1]):NULL;
            int refRatio = (lev > 0)?m_refinement_ratios[lev-1]:-1;
            levelCoords_half.recomputeGeometryFace(crseCoords, refRatio);
          }
        }
    }
  
  // do velocity solve for half-time velocity field
  if (m_temporalAccuracy == 2)
    {
      
      // first, reset H in coordSys using H_half 
      // (slc :: calculation was already done above and we will need the old time
      // also, so change solveVelocityField so we can just swap LevelSigmaCSPointers)
      MayDay::Error("m_temporalAccuracy ==  doesn't work yet");
      for (int lev=0; lev<= m_finest_level; lev++)
        {
          LevelData<FluxBox>& levelH = *H_half[lev];
          LevelSigmaCS& levelCoords = *m_vect_coordSys[lev];
          LevelData<FluxBox>& levelFaceH = levelCoords.getFaceH();
          DataIterator dit = levelH.dataIterator();
          for (dit.begin(); dit.ok(); ++dit)
            {
              FluxBox& faceH = levelFaceH[dit];
              faceH.copy(levelH[dit], levelH[dit].box());
            }
          {
            LevelSigmaCS* crseCoords = (lev > 0)?&(*m_vect_coordSys[lev-1]):NULL;
            int refRatio = (lev > 0)?m_refinement_ratios[lev-1]:-1;
            levelCoords.recomputeGeometryFace(crseCoords, refRatio);
          }
          
        } // end loop over levels
      
          // compute new ice velocity field
      if (m_evolve_velocity )
	{
	  if (s_verbosity > 3) 
	    {
	      pout() << "AmrIce::timeStep solveVelocityField() [m_temporalAccuracy == 2]" << endl;
	    }
	  solveVelocityField();
	}
      // average cell-centered velocity field to faces just like before
      
    }

  // compute thickness fluxes
  computeThicknessFluxes(vectFluxes, H_half, m_faceVelAdvection);
 
  if (m_report_discharge && (m_next_report_time - m_time) < (a_dt + TIME_EPS))
    {
      computeDischarge(vectFluxes);
    }

  // update ice fraction through advection
  advectIceFrac(m_iceFrac, m_faceVelAdvection, a_dt);

  // make a copy of m_vect_coordSys before it is overwritten
  Vector<RefCountedPtr<LevelSigmaCS> > vectCoords_old (m_finest_level+1);
  for (int lev=0; lev<= m_finest_level; lev++)
    {
      IntVect sigmaCSGhost = m_vect_coordSys[lev]->ghostVect();
      RealVect dx = m_amrDx[lev]*RealVect::Unit;
      vectCoords_old[lev] = RefCountedPtr<LevelSigmaCS> 
        (new LevelSigmaCS(m_amrGrids[lev], dx, sigmaCSGhost));
      LevelSigmaCS& levelCoords_old = *vectCoords_old[lev];
      const LevelSigmaCS& levelCoords = *m_vect_coordSys[lev];
      
      
      levelCoords_old.setIceDensity(levelCoords.iceDensity());
      levelCoords_old.setWaterDensity(levelCoords.waterDensity());
      levelCoords_old.setGravity(levelCoords.gravity());
      // todo replace the copies below with a deepCopy of levelCoords
      for (DataIterator dit( m_amrGrids[lev]); dit.ok(); ++dit)
        {
          FArrayBox& oldH = levelCoords_old.getH()[dit];
          const FArrayBox& H = levelCoords.getH()[dit];
          oldH.copy(H);
        }
      levelCoords_old.setTopography(levelCoords.getTopography());
      {
        LevelSigmaCS* crseCoords = (lev > 0)?&(*vectCoords_old[lev-1]):NULL;
        int refRatio = (lev > 0)?m_refinement_ratios[lev-1]:-1;
        levelCoords_old.recomputeGeometry( crseCoords, refRatio);
      }
#if BISICLES_Z == BISICLES_LAYERED
      levelCoords_old.setFaceSigma(levelCoords.getFaceSigma());
#endif
    }

  // compute div(F) and update geometry
  updateGeometry(m_vect_coordSys, vectCoords_old, vectFluxes, a_dt);

  // update internal energy
  if (!m_isothermal)
    updateInternalEnergy(layerEH_half, layerH_half, m_layerXYFaceXYVel,
			 m_layerSFaceXYVel,  a_dt, m_time,
			 m_vect_coordSys, vectCoords_old, 
			 m_surfaceThicknessSource, m_basalThicknessSource);

  
  notifyObservers(Observer::PostGeometryUpdate);
  
  // clean up temporary storage
  for (int lev=0; lev<=m_finest_level; lev++)
    {
          
      if (H_half[lev] != NULL)
        {
          delete H_half[lev];
          H_half[lev] = NULL;
        }
      
      if (layerEH_half[lev] != NULL)
        {
          delete layerEH_half[lev];
          layerEH_half[lev] = NULL;
        }
      if (layerH_half[lev] != NULL)
        {
          delete layerH_half[lev];
          layerH_half[lev] = NULL;
        }

      if (vectFluxes[lev] != NULL)
        {
          delete vectFluxes[lev];
          vectFluxes[lev] = NULL;
        }      
    }
  
  if (m_temporalAccuracy > 2)
    {
      MayDay::Error("AmrIce::timestep -- un-defined temporal accuracy");
    }

  //update time (velocity is to be computed at the step end)
  m_time += a_dt;
  m_cur_step += 1;
  // compute new ice velocity field
  if (m_evolve_velocity )
    {
      if (s_verbosity > 3) 
	{
	  pout() << "AmrIce::timeStep solveVelocityField() (step end) " << endl;
	}
      solveVelocityField();
    }
  
  // write diagnostic info, like sum of ice
  if ((m_next_report_time - m_time) < (a_dt + TIME_EPS) && !(m_time < m_next_report_time))
    {

      endTimestepDiagnostics();

      Real old_report_time=m_next_report_time;
      m_next_report_time = m_report_time_interval * (1.0 + Real(int((m_time/m_report_time_interval))));
      if (!(m_next_report_time > old_report_time))
	{ 
	  m_next_report_time += m_report_time_interval;
	}

      pout() << "  Next report time will be " 
	     << m_next_report_time << endl;
    }

  if (s_verbosity > 0) 
    {
      pout () << "AmrIce::timestep " << m_cur_step
              << " --     end time = " 
	      << setiosflags(ios::fixed) << setprecision(6) << setw(12)
              << m_time  << " ( " << time() << " )"
        //<< " (" << m_time/secondsperyear << " yr)"
              << ", dt = " 
        //<< setiosflags(ios::fixed) << setprecision(6) << setw(12)
              << a_dt
        //<< " ( " << a_dt/secondsperyear << " yr )"
	      << resetiosflags(ios::fixed)
              << endl;
    }

  
  int totalCellsAdvanced = 0;
  for (int lev=0; lev<m_num_cells.size(); lev++) 
    {
      totalCellsAdvanced += m_num_cells[lev];
    }
     
  if (s_verbosity > 0) 
    {
      pout() << "Time = " << m_time  
             << " cells advanced = " 
             << totalCellsAdvanced << endl;

      for (int lev=0; lev<m_num_cells.size(); lev++) 
        {
          pout () << "Time = " << m_time 
                  << "  level " << lev << " cells advanced = " 
                  << m_num_cells[lev] << endl;
        }
    }
}


// compute half-time face-centered thickness using unsplit PPM
void
AmrIce::computeH_half(Vector<LevelData<FluxBox>* >& a_H_half, Real a_dt)
{
  for (int lev=0; lev<= finestTimestepLevel();  lev++)
    {
      
      // get AdvectPhysics object from PatchGodunov object
      PatchGodunov* patchGod = m_thicknessPatchGodVect[lev];
      AdvectPhysics* advectPhysPtr = dynamic_cast<AdvectPhysics*>(patchGod->getGodunovPhysicsPtr());
      if (advectPhysPtr == NULL)
        {
          MayDay::Error("AmrIce::timestep -- unable to upcast GodunovPhysics to AdvectPhysics");
        }
      
      patchGod->setCurrentTime(m_time);
      
      // loop over grids on this level and compute H-Half
      const DisjointBoxLayout& levelGrids = m_amrGrids[lev];
      LevelData<FluxBox>& levelFaceVel = *m_faceVelAdvection[lev];
      LevelData<FArrayBox>& levelOldThickness = *m_old_thickness[lev];
      LevelData<FluxBox>& levelHhalf = *a_H_half[lev];
      
      LevelData<FArrayBox>& levelSTS = *m_surfaceThicknessSource[lev];
      LevelData<FArrayBox>& levelBTS = *m_basalThicknessSource[lev];
      CH_assert(m_surfaceFluxPtr != NULL);
      
      // set surface thickness source
      m_surfaceFluxPtr->surfaceThicknessFlux(levelSTS, *this, lev, a_dt);
      
      // set basal thickness source
      m_basalFluxPtr->surfaceThicknessFlux(levelBTS, *this, lev, a_dt);
      
      LevelData<FArrayBox> levelCCVel(levelGrids, SpaceDim, IntVect::Unit);
      EdgeToCell( levelFaceVel, levelCCVel);
      
      DataIterator dit = levelGrids.dataIterator();
      for (dit.begin(); dit.ok(); ++dit)
        {
          patchGod->setCurrentBox(levelGrids[dit]);
          advectPhysPtr->setVelocities(&(levelCCVel[dit]), 
                                       &(levelFaceVel[dit]));
          
          FArrayBox advectiveSource(levelSTS[dit].box(),1);
          advectiveSource.copy(levelSTS[dit]);
          advectiveSource.plus(levelBTS[dit]);
	  
          // add a diffusive source term div(D grad H)) to  advectiveSource
          if (m_diffusionTreatment == IMPLICIT)
            {
              for (int dir=0; dir<SpaceDim; dir++)
                {
                  Box faceBox = levelGrids[dit].surroundingNodes(dir);
                  FArrayBox flux(faceBox,1);
                  FORT_FACEDERIV(CHF_FRA1(flux,0),
                                 CHF_CONST_FRA1(levelOldThickness[dit],0),
                                 CHF_BOX(faceBox),
                                 CHF_CONST_REAL(dx(lev)[dir]),
                                 CHF_INT(dir),
                                 CHF_INT(dir));
                  CH_assert(flux.norm(0) < HUGE_NORM);
                  flux *= (*m_diffusivity[lev])[dit][dir];
                  CH_assert(flux.norm(0) < HUGE_NORM);
                  FORT_DIVERGENCE(CHF_CONST_FRA(flux),
                                  CHF_FRA(advectiveSource),
                                  CHF_BOX(levelGrids[dit]),
                                  CHF_CONST_REAL(dx(lev)[dir]),
                                  CHF_INT(dir));
                  
                }
            }
          
	  
          
          patchGod->computeWHalf(levelHhalf[dit],
                                 levelOldThickness[dit],
                                 advectiveSource,
                                 a_dt,
                                 levelGrids[dit]);
          
          
        } //end loop over grids
      
      
    } // end loop over levels for computing Whalf
  
  // coarse average new H-Half to covered regions
  for (int lev= finestTimestepLevel(); lev>0; lev--)
    {
      CoarseAverageFace faceAverager(m_amrGrids[lev],
                                     1, m_refinement_ratios[lev-1]);
      faceAverager.averageToCoarse(*a_H_half[lev-1], *a_H_half[lev]);
    }
  

}


void 
AmrIce::computeThicknessFluxes(Vector<LevelData<FluxBox>* >& a_vectFluxes,
                               const Vector<LevelData<FluxBox>* >& a_H_half,
                               const Vector<LevelData<FluxBox>* >& a_faceVelAdvection)
{
  for (int lev=0; lev<=finestTimestepLevel(); lev++)
    {
      LevelData<FluxBox>& levelFaceVel = *a_faceVelAdvection[lev];
      LevelData<FluxBox>& levelFaceH = *a_H_half[lev];
      const DisjointBoxLayout& levelGrids = m_amrGrids[lev];
      DataIterator dit = levelGrids.dataIterator();
      for (dit.begin(); dit.ok(); ++dit)
        {
          FluxBox& faceVel = levelFaceVel[dit];
          FluxBox& faceH = levelFaceH[dit];
          FluxBox& flux = (*a_vectFluxes[lev])[dit];
	  
          const Box& gridBox = levelGrids[dit];
          
          for (int dir=0; dir<SpaceDim; dir++)
            {
              Box faceBox(gridBox);
              faceBox.surroundingNodes(dir);
              flux[dir].copy(faceH[dir], faceBox);
              flux[dir].mult(faceVel[dir], faceBox, 0, 0, 1);
            }
        }
    } // end loop over levels
  
  // average fine fluxes down to coarse levels
  for (int lev=finestTimestepLevel(); lev>0; lev--)
    {
      CoarseAverageFace faceAverager(m_amrGrids[lev],
                                     1, m_refinement_ratios[lev-1]);
      faceAverager.averageToCoarse(*a_vectFluxes[lev-1], *a_vectFluxes[lev]);
    }
  
}

// update  ice thickness *and* bedrock elevation
void
AmrIce::updateGeometry(Vector<RefCountedPtr<LevelSigmaCS> >& a_vect_coordSys_new, 
		       Vector<RefCountedPtr<LevelSigmaCS> >& a_vect_coordSys_old, 
		       const Vector<LevelData<FluxBox>* >& a_vectFluxes, 
		       Real a_dt)
{


  for (int lev=0; lev <= finestTimestepLevel() ; lev++)
    {
      DisjointBoxLayout& levelGrids = m_amrGrids[lev];
      LevelData<FluxBox>& levelFlux = *a_vectFluxes[lev];
      LevelSigmaCS& levelCoords = *(a_vect_coordSys_new[lev]);
      LevelData<FArrayBox>& levelNewH = levelCoords.getH();
      LevelData<FArrayBox>& levelOldH = (*a_vect_coordSys_old[lev]).getH();
      LevelData<FArrayBox>& levelDivThckFlux = *m_divThicknessFlux[lev];
      const RealVect& dx = levelCoords.dx();              
      
      DataIterator dit = levelGrids.dataIterator();          
      
      for (dit.begin(); dit.ok(); ++dit)
        {
          const Box& gridBox = levelGrids[dit];
          FArrayBox& newH = levelNewH[dit];
          FArrayBox& oldH = levelOldH[dit];//(*m_old_thickness[lev])[dit];
          FluxBox& thisFlux = levelFlux[dit];
          newH.setVal(0.0);
          
          // loop over directions and increment with div(F)
          for (int dir=0; dir<SpaceDim; dir++)
            {
              // use the divergence from 
              // Chombo/example/fourthOrderMappedGrids/util/DivergenceF.ChF
              FORT_DIVERGENCE(CHF_CONST_FRA(thisFlux[dir]),
                              CHF_FRA(newH),
                              CHF_BOX(gridBox),
                              CHF_CONST_REAL(dx[dir]),
                              CHF_INT(dir));
              
              
            }
          
	  levelDivThckFlux[dit].copy(newH);
          // add in thickness source
          // if there are still diffusive fluxes to deal
          // with, the source term will be included then
          if (m_evolve_thickness)
            {
              if (m_floating_ice_stable || m_floating_ice_basal_flux_is_dhdt)
                {
                  //keep floating ice stable if required
                  const BaseFab<int>& mask = levelCoords.getFloatingMask()[dit];
                  for (BoxIterator bit(gridBox); bit.ok(); ++bit)
                    {
                      const IntVect& iv = bit();
                      if (mask(iv) == FLOATINGMASKVAL)
                        {
			  (*m_surfaceThicknessSource[lev])[dit](iv) = 0.0;
			  if (!m_floating_ice_basal_flux_is_dhdt)
			    {
			      (*m_basalThicknessSource[lev])[dit](iv) = 0.0;
			    }
			  (*m_basalThicknessSource[lev])[dit](iv) += newH(iv);
                        }
                    }
                }
              
              if (m_grounded_ice_stable || m_grounded_ice_basal_flux_is_dhdt)
                {
                  //keep grounded ice stable if required
                  const BaseFab<int>& mask = levelCoords.getFloatingMask()[dit];
                  for (BoxIterator bit(gridBox); bit.ok(); ++bit)
                    {
                      const IntVect& iv = bit();
                      if (mask(iv) == GROUNDEDMASKVAL)
                        {
                          (*m_surfaceThicknessSource[lev])[dit](iv) = 0.0;
			  if (!m_grounded_ice_basal_flux_is_dhdt)
			    {
			      (*m_basalThicknessSource[lev])[dit](iv) = 0.0;
			    }
			  (*m_basalThicknessSource[lev])[dit](iv) += newH(iv);
                        }
                    }
                }
            }
          else 
            {
	      (*m_surfaceThicknessSource[lev])[dit].copy(newH);
	      (*m_basalThicknessSource[lev])[dit].setVal(0.0);
            }

	  if (m_diffusionTreatment != IMPLICIT)
            {
              if (m_frac_sources)
                {
                  // scale surface fluxes by mask values
                  const FArrayBox& thisFrac = (*m_iceFrac[lev])[dit];
                  FArrayBox sources(gridBox,1);
                  sources.setVal(0.0);
                  sources.plus((*m_surfaceThicknessSource[lev])[dit], gridBox,
                               0, 0, 1);
                  sources.plus((*m_basalThicknessSource[lev])[dit], gridBox, 
                               0, 0, 1);
                  
                  sources.mult(thisFrac, gridBox, 0, 0, 1);
                  newH.minus(sources, gridBox, 0, 0, 1);
                  
                }
              else
                {

                  // just add in sources directly
                  newH.minus((*m_surfaceThicknessSource[lev])[dit], gridBox,0,0,1);
                  newH.minus((*m_basalThicknessSource[lev])[dit], gridBox,0,0,1);
                }
            }
          
	          
	  newH *= -1*a_dt;
          newH.plus(oldH, 0, 0, 1);


	  for (BoxIterator bit(gridBox); bit.ok(); ++bit)
	    {
	      const IntVect& iv = bit();

	      Real H=newH(iv);
	      // Remove negative thickness by limiting low bmb and/or smb
	      // Calculate the effective basal and surface mass fluxes
	      if (H < 0.0)
		{
		  Real excessSource=0.0;
		  if (a_dt > 0.0)
		    {
		      excessSource=H/a_dt;
		    }
		  Real bts = (*m_basalThicknessSource[lev])[dit](iv);
		  Real sts = (*m_surfaceThicknessSource[lev])[dit](iv);
		  Real oldBTS=bts;
		  if (bts < 0.0)
		    {
		      bts = std::min(bts-excessSource,0.0);
		    }
		  sts = sts+oldBTS - excessSource - bts;
		  (*m_basalThicknessSource[lev])[dit](iv) = bts;
		  (*m_surfaceThicknessSource[lev])[dit](iv) = sts;
		  
                  newH(iv)=0.0;
		}
	    }

          
        } // end loop over grids 
    } // end loop over levels
  
  
  
  //include any diffusive fluxes
  if (m_evolve_thickness && m_diffusionTreatment == IMPLICIT)
    {
      if (m_grounded_ice_stable || m_floating_ice_stable)
	{
	  CH_assert( !(m_grounded_ice_stable || m_floating_ice_stable));
	  MayDay::Error("implicit diffusion not implemented with grounded_ice_stable or floating_ice_stable ");
	}
      //MayDay::Error("m_diffusionTreatment == IMPLICIT no yet implemented");
      //implicit thickness correction
      if (m_frac_sources)
        {
          MayDay::Error("scaling sources by ice fraction values not implemented yet");
        }
      implicitThicknessCorrection(a_dt, m_surfaceThicknessSource,  m_basalThicknessSource);
    }

  //update the topography (fixed surface case)
  if (m_evolve_topography_fix_surface)
    {
      // update the bedrock so that the surface remains constant on grounded ice
      for (int lev=0; lev <= finestTimestepLevel() ; lev++)
	{
	  for (DataIterator dit(m_amrGrids[lev]);dit.ok();++dit)
	    {
	      FArrayBox& newH = a_vect_coordSys_new[lev]->getH()[dit];
	      FArrayBox& oldH = a_vect_coordSys_old[lev]->getH()[dit];
	      FArrayBox& topg = a_vect_coordSys_new[lev]->getTopography()[dit];
	      
	      const BaseFab<int>& mask = a_vect_coordSys_old[lev]->getFloatingMask()[dit];
	      FORT_EVOLVEGROUNDEDBED(CHF_FRA1(newH,0), CHF_FRA1(oldH,0), 
				     CHF_FRA1(topg,0), CHF_CONST_FIA1(mask,0), 
				     CHF_BOX(topg.box()));
	      FArrayBox& deltaTopg = (*m_deltaTopography[lev])[dit];
	      deltaTopg += topg;
	      deltaTopg -= a_vect_coordSys_old[lev]->getTopography()[dit];	      
	    }
	}
    }


  //update the topography (gia)
  if (m_topographyFluxPtr != NULL)
    {
      for (int lev=0; lev <= finestTimestepLevel() ; lev++)
	{
	  LevelSigmaCS& levelCoords = *(a_vect_coordSys_new[lev]);
	  DisjointBoxLayout& levelGrids = m_amrGrids[lev];
	  LevelData<FArrayBox>& levelTopg = levelCoords.getTopography();
	  LevelData<FArrayBox>& levelDeltaTopg = *m_deltaTopography[lev];
	  LevelData<FArrayBox> levelSrc(levelGrids,1,IntVect::Zero);
	  m_topographyFluxPtr->surfaceThicknessFlux(levelSrc, *this, lev, a_dt);
	  for (DataIterator dit(levelGrids);dit.ok();++dit)
	    {
	      FArrayBox& src = levelSrc[dit];
	      src *= a_dt;
	      levelTopg[dit] += src;
	      levelDeltaTopg[dit] += src;
	    }
	}
    }
  
  


  // average down thickness and topography to coarser levels and fill in ghost cells
  // before calling recomputeGeometry. 
  int Hghost = 2;
  Vector<LevelData<FArrayBox>* > vectH(m_finest_level+1, NULL);
  Vector<LevelData<FArrayBox>* > vectB(m_finest_level+1, NULL);
  for (int lev=0; lev<vectH.size(); lev++)
    {
      IntVect HghostVect = Hghost*IntVect::Unit;
      LevelSigmaCS& levelCoords = *(a_vect_coordSys_new[lev]);
      vectH[lev] = &levelCoords.getH();
      vectB[lev] = &levelCoords.getTopography();
    }
  
  //average from the finest level down
  for (int lev =  finestTimestepLevel() ; lev > 0 ; lev--)
    {
      CoarseAverage averager(m_amrGrids[lev],
                             1, m_refinement_ratios[lev-1]);
      averager.averageToCoarse(*vectH[lev-1],
                               *vectH[lev]);
      averager.averageToCoarse(*vectB[lev-1],
                               *vectB[lev]);
    }

  // now pass back over and do PiecewiseLinearFillPatch
  for (int lev=1; lev<vectH.size(); lev++)
    {
      
      PiecewiseLinearFillPatch filler(m_amrGrids[lev],
                                      m_amrGrids[lev-1],
                                      1, 
                                      m_amrDomains[lev-1],
                                      m_refinement_ratios[lev-1],
                                      Hghost);
      
      Real interp_coef = 0;
      filler.fillInterp(*vectH[lev],
                        *vectH[lev-1],
                        *vectH[lev-1],
                        interp_coef,
                        0, 0, 1);
      filler.fillInterp(*vectB[lev],
                        *vectB[lev-1],
                        *vectB[lev-1],
                        interp_coef,
                        0, 0, 1);
    }
  
  
  //re-fill ghost cells ouside the domain
  for (int lev=0; lev <= finestTimestepLevel()  ; ++lev)
    {
      RealVect levelDx = m_amrDx[lev]*RealVect::Unit;
      m_thicknessIBCPtr->setGeometryBCs(*a_vect_coordSys_new[lev],
                                        m_amrDomains[lev],levelDx, m_time, m_dt);
    }
  
  //allow calving model to modify geometry and velocity
  applyCalvingCriterion(CalvingModel::PostThicknessAdvection);
  
  
  //dont allow thickness to be negative
  for (int lev=0; lev<= m_finest_level; lev++)
    {
      DisjointBoxLayout& levelGrids = m_amrGrids[lev];
      LevelSigmaCS& levelCoords = *(a_vect_coordSys_new[lev]);
      LevelData<FArrayBox>& levelH = levelCoords.getH();
      DataIterator dit = levelGrids.dataIterator();          
      
      for (DataIterator dit(levelGrids); dit.ok(); ++dit)
        {
          Real lim = 0.0;
          FORT_MAXFAB1(CHF_FRA(levelH[dit]), 
                       CHF_CONST_REAL(lim), 
                       CHF_BOX(levelH[dit].box()));
        }
    }
  
  //average from the finest level down
  for (int lev =  finestTimestepLevel() ; lev > 0 ; lev--)
    {
      CoarseAverage averager(m_amrGrids[lev],
                             1, m_refinement_ratios[lev-1]);
      averager.averageToCoarse(*vectH[lev-1],
                               *vectH[lev]);      
    }
  
  for (int lev=1; lev<vectH.size(); lev++)
    {      
      PiecewiseLinearFillPatch filler(m_amrGrids[lev],
                                      m_amrGrids[lev-1],
                                      1, 
                                      m_amrDomains[lev-1],
                                      m_refinement_ratios[lev-1],
                                      Hghost);
      
      Real interp_coef = 0;
      filler.fillInterp(*vectH[lev],
                        *vectH[lev-1],
                        *vectH[lev-1],
                        interp_coef,
                        0, 0, 1);
    }
  
  //interpolate levelSigmaCS to any levels above finestTimestepLevel()
  for (int lev = finestTimestepLevel()+1 ; lev<= m_finest_level; lev++)
    {
      m_vect_coordSys[lev]->interpFromCoarse(*m_vect_coordSys[lev-1],
                                             m_refinement_ratios[lev-1],
                                             false , true);
    }
  
  
  // recompute thickness-derived data in SigmaCS
  for (int lev=0; lev<= m_finest_level; lev++)
    {
      LevelSigmaCS& levelCoords = *(m_vect_coordSys[lev]);
      LevelSigmaCS* crseCoords = (lev > 0)?&(*m_vect_coordSys[lev-1]):NULL;
      int refRatio = (lev > 0)?m_refinement_ratios[lev-1]:-1;
      levelCoords.recomputeGeometry(crseCoords, refRatio);            
    }
  
}


// do regridding
void
AmrIce::regrid()
{

  CH_TIME("AmrIce::regrid");

  if (s_verbosity > 3) 
    { 
      pout() << "AmrIce::regrid" << endl;
    }
  
  //first part of a conservation of volume check
  Real volumeBefore = computeTotalIce();

  // only do any of this if the max level > 0
  if (m_max_level > 0) 
    {

      m_n_regrids++;

      // in this code, lbase is always 0
      int lbase =0;
      
      // first generate tags
      Vector<IntVectSet> tagVect(m_max_level);
      tagCells(tagVect);
      
      {
	// now generate new boxes
	int top_level = min(m_finest_level, m_max_level-1);
	Vector<Vector<Box> > old_grids(m_finest_level+1);
	Vector<Vector<Box> > new_grids;
	
	// this is clunky, but i don't know of a better way to turn 
	// a DisjointBoxLayout into a Vector<Box>
	for (int lev=0; lev<= m_finest_level; lev++) 
	  {
	    const DisjointBoxLayout& levelDBL = m_amrGrids[lev];
	    old_grids[lev].resize(levelDBL.size());
	    LayoutIterator lit = levelDBL.layoutIterator();
	    int boxIndex = 0;
	    for (lit.begin(); lit.ok(); ++lit, ++boxIndex) 
	      {
		old_grids[lev][boxIndex] = levelDBL[lit()];
	      }
	  }
	
	int new_finest_level;
	
	BRMeshRefine meshrefine(m_amrDomains[0], m_refinement_ratios,
				m_fill_ratio, m_block_factor, 
				m_nesting_radius, m_max_box_size);
	
	new_finest_level = meshrefine.regrid(new_grids, tagVect, 
					     lbase, top_level, 
					     old_grids);

	//test to see if grids have changed
	bool gridsSame = true;
	for (int lev=lbase+1; lev<= new_finest_level; ++lev)
	  {
	    int numGridsNew = new_grids[lev].size();
	    Vector<int> procIDs(numGridsNew);
	    LoadBalance(procIDs, new_grids[lev]);
	    const DisjointBoxLayout newDBL(new_grids[lev], procIDs,
					   m_amrDomains[lev]);
	    const DisjointBoxLayout oldDBL = m_amrGrids[lev];
	    gridsSame &= oldDBL.sameBoxes(newDBL);
	  }
	if (gridsSame)
	  {
	    if (s_verbosity > 3) 
	      { 
		pout() << "AmrIce::regrid -- grids unchanged" << endl;
	      }
	    //return;
	  }

#ifdef REGRID_EH
	// We need to regrid the internal energy, but the conserved quantity is H*E
	// Set E <- E*H now (and E <- E/H later)
	for (int lev=0; lev <= m_finest_level ; ++lev)
	  {
	    for (DataIterator dit(m_amrGrids[lev]); dit.ok(); ++dit)
	      {
		FArrayBox& E = (*m_internalEnergy[lev])[dit];
		FArrayBox H(E.box(),1);
		H.copy((*m_old_thickness[lev])[dit]);
		H += 1.0e-10;
		for (int comp  = 0; comp < m_internalEnergy[0]->nComp(); comp++)
		  {
		    E.mult( H,0,comp,1);
		  }
	      }
	  }
#endif

	// now loop through levels and redefine if necessary
	for (int lev=lbase+1; lev<= new_finest_level; ++lev)
	  {
	    int numGridsNew = new_grids[lev].size();
	    Vector<int> procIDs(numGridsNew);
	    LoadBalance(procIDs, new_grids[lev]);
	      
	    const DisjointBoxLayout newDBL(new_grids[lev], procIDs,
					   m_amrDomains[lev]);
	      
	    const DisjointBoxLayout oldDBL = m_amrGrids[lev];
	      
	    m_amrGrids[lev] = newDBL;
	      
	    // build new storage
	    LevelData<FArrayBox>* old_old_thicknessDataPtr = m_old_thickness[lev];
	    LevelData<FArrayBox>* old_velDataPtr = m_velocity[lev];
	    LevelData<FArrayBox>* old_tempDataPtr = m_internalEnergy[lev];
	    LevelData<FArrayBox>* old_accumCalvDataPtr = m_melangeThickness[lev];
	    LevelData<FArrayBox>* old_calvDataPtr = m_calvedIceThickness[lev];
	    LevelData<FArrayBox>* old_removedDataPtr = m_removedIceThickness[lev];
	    LevelData<FArrayBox>* old_addedDataPtr = m_addedIceThickness[lev];

	    LevelData<FArrayBox>* old_deltaTopographyDataPtr = m_deltaTopography[lev];
            LevelData<FArrayBox>* old_iceFracDataPtr = m_iceFrac[lev];
	     
	    LevelData<FArrayBox>* new_old_thicknessDataPtr = 
	      new LevelData<FArrayBox>(newDBL, 1, m_old_thickness[0]->ghostVect());
	      
	    LevelData<FArrayBox>* new_velDataPtr = 
	      new LevelData<FArrayBox>(newDBL, SpaceDim, m_velocity[0]->ghostVect());

	    LevelData<FArrayBox>* new_tempDataPtr = 
	      new LevelData<FArrayBox>(newDBL, m_internalEnergy[0]->nComp(), 
				       m_internalEnergy[0]->ghostVect());
	    //since the internalEnergy data has changed
	    m_A_valid = false;

            LevelData<FArrayBox>* new_iceFracDataPtr = 
              new LevelData<FArrayBox>(newDBL, 1, m_iceFrac[0]->ghostVect());
            
	    LevelData<FArrayBox>* new_accumCalvDataPtr = 
	      new LevelData<FArrayBox>(newDBL, m_melangeThickness[0]->nComp(), 
				       m_melangeThickness[0]->ghostVect());
	    LevelData<FArrayBox>* new_calvDataPtr = 
	      new LevelData<FArrayBox>(newDBL, m_calvedIceThickness[0]->nComp(), 
				       m_calvedIceThickness[0]->ghostVect());
	    LevelData<FArrayBox>* new_removedDataPtr = 
	      new LevelData<FArrayBox>(newDBL, m_removedIceThickness[0]->nComp(), 
				       m_removedIceThickness[0]->ghostVect());
	    LevelData<FArrayBox>* new_addedDataPtr = 
	      new LevelData<FArrayBox>(newDBL, m_addedIceThickness[0]->nComp(), 
				       m_addedIceThickness[0]->ghostVect());
	      
	    LevelData<FArrayBox>* new_deltaTopographyDataPtr = 
	      new LevelData<FArrayBox>(newDBL, m_deltaTopography[0]->nComp(), 
				       m_deltaTopography[0]->ghostVect());

	      
#if BISICLES_Z == BISICLES_LAYERED
	    LevelData<FArrayBox>* old_sTempDataPtr = m_sInternalEnergy[lev];
	    LevelData<FArrayBox>* old_bTempDataPtr = m_bInternalEnergy[lev];
	    LevelData<FArrayBox>* new_sTempDataPtr = 
	      new LevelData<FArrayBox>(newDBL, m_sInternalEnergy[0]->nComp(),
				       m_sInternalEnergy[0]->ghostVect());
	    LevelData<FArrayBox>* new_bTempDataPtr = 
	      new LevelData<FArrayBox>(newDBL, m_bInternalEnergy[0]->nComp(),
				       m_bInternalEnergy[0]->ghostVect());
	     
#endif	      
	      
	    {
	      // first we need to regrid m_deltaTopography, it will be needed to 
	      // regrid the bedrock topography & hence LevelSigmaCS
	      // may eventually want to do post-regrid smoothing on this
	      FineInterp interpolator(newDBL,m_deltaTopography[0]->nComp(),
				      m_refinement_ratios[lev-1],
				      m_amrDomains[lev]);
	      interpolator.interpToFine(*new_deltaTopographyDataPtr, *m_deltaTopography[lev-1]);

		
	      PiecewiseLinearFillPatch ghostFiller
		(m_amrGrids[lev],
		 m_amrGrids[lev-1],
		 m_deltaTopography[lev-1]->nComp(),
		 m_amrDomains[lev-1],
		 m_refinement_ratios[lev-1],
		 m_deltaTopography[lev-1]->ghostVect()[0]);

	      ghostFiller.fillInterp(*new_deltaTopographyDataPtr,*m_deltaTopography[lev-1],
				     *m_deltaTopography[lev-1],1.0,0,0,
				     m_deltaTopography[lev-1]->nComp());

	      if (old_deltaTopographyDataPtr != NULL && oldDBL.isClosed())
		{
		  old_deltaTopographyDataPtr->copyTo(*new_deltaTopographyDataPtr);
		}
	      delete old_deltaTopographyDataPtr;
		
	    }



	    // also need to handle LevelSigmaCS 

	    // assume level 0 has correct ghosting
	    IntVect sigmaCSGhost = m_vect_coordSys[0]->ghostVect();
	    {
	      RealVect dx = m_amrDx[lev]*RealVect::Unit;
	      RefCountedPtr<LevelSigmaCS > oldCoordSys = m_vect_coordSys[lev];
		
	      RefCountedPtr<LevelSigmaCS > auxCoordSys = (lev > 0)?m_vect_coordSys[lev-1]:oldCoordSys;

	      m_vect_coordSys[lev] = RefCountedPtr<LevelSigmaCS >
		(new LevelSigmaCS(newDBL, dx, sigmaCSGhost));
	      m_vect_coordSys[lev]->setIceDensity(auxCoordSys->iceDensity());
	      m_vect_coordSys[lev]->setWaterDensity(auxCoordSys->waterDensity());
	      m_vect_coordSys[lev]->setGravity(auxCoordSys->gravity());
	      m_vect_coordSys[lev]->setBackgroundSlope(auxCoordSys->getBackgroundSlope());
#if BISICLES_Z == BISICLES_LAYERED
	      m_vect_coordSys[lev]->setFaceSigma(auxCoordSys->getFaceSigma());
#endif		
	      LevelSigmaCS* crsePtr = &(*m_vect_coordSys[lev-1]);
	      int refRatio = m_refinement_ratios[lev-1];

	      bool interpolate_zb = (m_interpolate_zb ||
				     !m_thicknessIBCPtr->regridIceGeometry
				     (*m_vect_coordSys[lev],dx,  m_domainSize, 
				      m_time,  crsePtr,refRatio ) );
		
	      if (!interpolate_zb)
		{
		  // need to re-apply accumulated bedrock (GIA). Could be optional?
		  for (DataIterator dit(newDBL); dit.ok(); ++dit)
		    {
		      m_vect_coordSys[lev]->getTopography()[dit] += (*new_deltaTopographyDataPtr)[dit];
		    }
		}

		{
		  //interpolate thickness & (maybe) topography
		  bool interpolateThickness(true);
		  bool preserveMask(true);
		  bool interpolateTopographyGhost(true); 
		  bool interpolateThicknessGhost(true); 
		  bool preserveMaskGhost(true);
		  m_vect_coordSys[lev]->interpFromCoarse(*m_vect_coordSys[lev-1],
							 m_refinement_ratios[lev-1],
							 interpolate_zb,
							 interpolateThickness, 
							 preserveMask,
							 interpolateTopographyGhost, 
							 interpolateThicknessGhost, 
							 preserveMaskGhost, 
							 m_regrid_thickness_interpolation_method);
		}


	      LevelData<FArrayBox>& thisLevelH = m_vect_coordSys[lev]->getH();
	      LevelData<FArrayBox>& thisLevelB = m_vect_coordSys[lev]->getTopography();
		
	      // overwrite interpolated fields in valid regiopns with such valid old data as there is
	      if (oldDBL.isClosed()){	  
		const LevelData<FArrayBox>& oldLevelH = oldCoordSys->getH();
		oldLevelH.copyTo(thisLevelH);
		const LevelData<FArrayBox>& oldLevelB = oldCoordSys->getTopography();
		oldLevelB.copyTo(thisLevelB);
	      }

	      //Defer to m_thicknessIBCPtr for boundary values - 
	      //interpolation won't cut the mustard because it only fills
	      //ghost cells overlying the valid regions.
	      RealVect levelDx = m_amrDx[lev]*RealVect::Unit;
	      m_thicknessIBCPtr->setGeometryBCs(*m_vect_coordSys[lev],
						m_amrDomains[lev],levelDx, m_time, m_dt);



	      // exchange is necessary to fill periodic ghost cells
	      // which aren't filled by the copyTo from oldLevelH
	      thisLevelH.exchange();
	      m_vect_coordSys[lev]->exchangeTopography();

	      {
		LevelSigmaCS* crseCoords = (lev > 0)?&(*m_vect_coordSys[lev-1]):NULL;
		int refRatio = (lev > 0)?m_refinement_ratios[lev-1]:-1;
		m_vect_coordSys[lev]->recomputeGeometry(crseCoords,refRatio);
	      }
	    }
		
	    // first fill with interpolated data from coarser level
	      
	    {
	      // may eventually want to do post-regrid smoothing on this!
	      FineInterp interpolator(newDBL, 1, 
				      m_refinement_ratios[lev-1],
				      m_amrDomains[lev]);
	    
	      interpolator.interpToFine(*new_old_thicknessDataPtr, *m_old_thickness[lev-1]);
	
	      // now copy old-grid data into new holder
	      if (old_old_thicknessDataPtr != NULL) 
		{
		  if ( oldDBL.isClosed())
		    {
		      old_old_thicknessDataPtr->copyTo(*new_old_thicknessDataPtr);
		    }
		  
		}

		interpolator.interpToFine(*new_iceFracDataPtr, *m_iceFrac[lev-1]);
	
		// now copy old-grid data into new holder
		if (old_iceFracDataPtr != NULL) 
		  {
		    if ( oldDBL.isClosed())
		      {
			old_iceFracDataPtr->copyTo(*new_iceFracDataPtr);
		      }
		    // can now delete old data 
		    delete old_iceFracDataPtr;
		  }

		
	    }
	      
	    {
	      // may eventually want to do post-regrid smoothing on this!
	      FineInterp interpolator(newDBL, SpaceDim, 
				      m_refinement_ratios[lev-1],
				      m_amrDomains[lev]);
		
	      interpolator.interpToFine(*new_velDataPtr, *m_velocity[lev-1]);

		
		
	      // now copy old-grid data into new holder
	      if (old_velDataPtr != NULL)
		{
		  if (oldDBL.isClosed()) 
		    {
		      old_velDataPtr->copyTo(*new_velDataPtr);
		    }
		  // can now delete old data 
		  delete old_velDataPtr;
		}

	      //handle ghost cells on the coarse-fine interface
	      QuadCFInterp qcfi(m_amrGrids[lev], &m_amrGrids[lev-1],
				m_amrDx[lev], m_refinement_ratios[lev-1],
				2, m_amrDomains[lev]);
	      qcfi.coarseFineInterp(*new_velDataPtr, *m_velocity[lev-1]);
		
	      //boundary ghost cells
	      m_thicknessIBCPtr->velocityGhostBC
		(*new_velDataPtr, *m_vect_coordSys[lev],m_amrDomains[lev],m_time);
		

	    }

	    {
	      // may eventually want to do post-regrid smoothing on this
	    
	      FineInterp interpolator(newDBL,m_internalEnergy[0]->nComp(),
				      m_refinement_ratios[lev-1],
				      m_amrDomains[lev]);
	      interpolator.interpToFine(*new_tempDataPtr, *m_internalEnergy[lev-1]);

		
	      PiecewiseLinearFillPatch ghostFiller
		(m_amrGrids[lev],
		 m_amrGrids[lev-1],
		 m_internalEnergy[lev-1]->nComp(),
		 m_amrDomains[lev-1],
		 m_refinement_ratios[lev-1],
		 m_internalEnergy[lev-1]->ghostVect()[0]);

	      ghostFiller.fillInterp(*new_tempDataPtr,*m_internalEnergy[lev-1],
				     *m_internalEnergy[lev-1],1.0,0,0,
				     m_internalEnergy[lev-1]->nComp());

	      if (old_tempDataPtr != NULL && oldDBL.isClosed())
		{
		  old_tempDataPtr->copyTo(*new_tempDataPtr);
		}
	      delete old_tempDataPtr;	
	    }
	      
	    {
	      // may eventually want to do post-regrid smoothing on this
	      FineInterp interpolator(newDBL,m_melangeThickness[0]->nComp(),
				      m_refinement_ratios[lev-1],
				      m_amrDomains[lev]);
	      interpolator.interpToFine(*new_accumCalvDataPtr, *m_melangeThickness[lev-1]);
		
	      PiecewiseLinearFillPatch ghostFiller
		(m_amrGrids[lev],
		 m_amrGrids[lev-1],
		 m_melangeThickness[lev-1]->nComp(),
		 m_amrDomains[lev-1],
		 m_refinement_ratios[lev-1],
		 m_melangeThickness[lev-1]->ghostVect()[0]);

	      ghostFiller.fillInterp(*new_accumCalvDataPtr,*m_melangeThickness[lev-1],
				     *m_melangeThickness[lev-1],1.0,0,0,
				     m_melangeThickness[lev-1]->nComp());

	      if (old_accumCalvDataPtr != NULL && oldDBL.isClosed())
		{
		  old_accumCalvDataPtr->copyTo(*new_accumCalvDataPtr);
		}
	      delete old_accumCalvDataPtr;
		
	    }
	    {
	      FineInterp interpolator(newDBL,m_calvedIceThickness[0]->nComp(),
				      m_refinement_ratios[lev-1],
				      m_amrDomains[lev]);
	      interpolator.interpToFine(*new_calvDataPtr, *m_calvedIceThickness[lev-1]);
		
	      PiecewiseLinearFillPatch ghostFiller
		(m_amrGrids[lev],
		 m_amrGrids[lev-1],
		 m_calvedIceThickness[lev-1]->nComp(),
		 m_amrDomains[lev-1],
		 m_refinement_ratios[lev-1],
		 m_calvedIceThickness[lev-1]->ghostVect()[0]);

	      ghostFiller.fillInterp(*new_calvDataPtr,*m_calvedIceThickness[lev-1],
				     *m_calvedIceThickness[lev-1],1.0,0,0,
				     m_calvedIceThickness[lev-1]->nComp());

	      if (old_calvDataPtr != NULL && oldDBL.isClosed())
		{
		  old_calvDataPtr->copyTo(*new_calvDataPtr);
		}
	      delete old_calvDataPtr;
		
	    }
	    {
	      FineInterp interpolator(newDBL,m_removedIceThickness[0]->nComp(),
				      m_refinement_ratios[lev-1],
				      m_amrDomains[lev]);
	      interpolator.interpToFine(*new_removedDataPtr, *m_removedIceThickness[lev-1]);
		
	      PiecewiseLinearFillPatch ghostFiller
		(m_amrGrids[lev],
		 m_amrGrids[lev-1],
		 m_removedIceThickness[lev-1]->nComp(),
		 m_amrDomains[lev-1],
		 m_refinement_ratios[lev-1],
		 m_removedIceThickness[lev-1]->ghostVect()[0]);

	      ghostFiller.fillInterp(*new_removedDataPtr,*m_removedIceThickness[lev-1],
				     *m_removedIceThickness[lev-1],1.0,0,0,
				     m_removedIceThickness[lev-1]->nComp());

	      if (old_removedDataPtr != NULL && oldDBL.isClosed())
		{
		  old_removedDataPtr->copyTo(*new_removedDataPtr);
		}
	      delete old_removedDataPtr;
		
	    }
	    {
	      FineInterp interpolator(newDBL,m_addedIceThickness[0]->nComp(),
				      m_refinement_ratios[lev-1],
				      m_amrDomains[lev]);
	      interpolator.interpToFine(*new_addedDataPtr, *m_addedIceThickness[lev-1]);
		
	      PiecewiseLinearFillPatch ghostFiller
		(m_amrGrids[lev],
		 m_amrGrids[lev-1],
		 m_addedIceThickness[lev-1]->nComp(),
		 m_amrDomains[lev-1],
		 m_refinement_ratios[lev-1],
		 m_addedIceThickness[lev-1]->ghostVect()[0]);

	      ghostFiller.fillInterp(*new_addedDataPtr,*m_addedIceThickness[lev-1],
				     *m_addedIceThickness[lev-1],1.0,0,0,
				     m_addedIceThickness[lev-1]->nComp());

	      if (old_addedDataPtr != NULL && oldDBL.isClosed())
		{
		  old_addedDataPtr->copyTo(*new_addedDataPtr);
		}
	      delete old_addedDataPtr;
		
	    }

 

#if BISICLES_Z == BISICLES_LAYERED
	    {
	      // may eventually want to do post-regrid smoothing on this
	      FineInterp interpolator(newDBL,m_sInternalEnergy[0]->nComp(),
				      m_refinement_ratios[lev-1],
				      m_amrDomains[lev]);

	      PiecewiseLinearFillPatch ghostFiller
		(m_amrGrids[lev],
		 m_amrGrids[lev-1],
		 m_sInternalEnergy[lev-1]->nComp(),
		 m_amrDomains[lev-1],
		 m_refinement_ratios[lev-1],
		 m_sInternalEnergy[lev-1]->ghostVect()[0]);

	

	      interpolator.interpToFine(*new_sTempDataPtr, *m_sInternalEnergy[lev-1]);
		
	      ghostFiller.fillInterp(*new_sTempDataPtr,*m_sInternalEnergy[lev-1],
				     *m_sInternalEnergy[lev-1],1.0,0,0,
				     m_sInternalEnergy[lev-1]->nComp());


	      if (old_sTempDataPtr != NULL && oldDBL.isClosed())
		{
		  old_sTempDataPtr->copyTo(*new_sTempDataPtr);
		}
	      delete old_sTempDataPtr;
		
	      interpolator.interpToFine(*new_bTempDataPtr, *m_bInternalEnergy[lev-1]);

	      ghostFiller.fillInterp(*new_bTempDataPtr,*m_bInternalEnergy[lev-1],
				     *m_bInternalEnergy[lev-1],1.0,0,0,
				     m_bInternalEnergy[lev-1]->nComp());
		
	      if (old_bTempDataPtr != NULL && oldDBL.isClosed())
		{
		  old_bTempDataPtr->copyTo(*new_bTempDataPtr);
		}
	      delete old_bTempDataPtr;

	      new_tempDataPtr->exchange();
	      new_sTempDataPtr->exchange();
	      new_bTempDataPtr->exchange();
	      //set boundary for non-periodic cases values
	      m_internalEnergyIBCPtr->setIceInternalEnergyBC
		(*new_tempDataPtr,*new_sTempDataPtr,*new_bTempDataPtr,
		 *m_vect_coordSys[lev] );

	    }
#endif
	    // can now delete old data 
	    delete old_old_thicknessDataPtr;

	    // now copy new holders into multilevel arrays
	    m_old_thickness[lev] = new_old_thicknessDataPtr;
	    m_velocity[lev] = new_velDataPtr;
	    m_internalEnergy[lev] = new_tempDataPtr;
	    m_melangeThickness[lev] = new_accumCalvDataPtr;
	    m_calvedIceThickness[lev] = new_calvDataPtr;
	    m_removedIceThickness[lev] = new_removedDataPtr;
	    m_addedIceThickness[lev] = new_addedDataPtr;
	    m_deltaTopography[lev] = new_deltaTopographyDataPtr;
            m_iceFrac[lev] = new_iceFracDataPtr;      
#if BISICLES_Z == BISICLES_LAYERED
	    m_sInternalEnergy[lev] = new_sTempDataPtr;
	    m_bInternalEnergy[lev] = new_bTempDataPtr;
#endif


	    if (m_velBasalC[lev] != NULL)
	      {
		delete m_velBasalC[lev];
	      }
	    m_velBasalC[lev] = new LevelData<FArrayBox>(newDBL, 1, IntVect::Unit);
	      
	    if (m_cellMuCoef[lev] != NULL)
	      {
		delete m_cellMuCoef[lev];
	      }
	    m_cellMuCoef[lev] = new LevelData<FArrayBox>(newDBL, 1, IntVect::Unit);
	     
	    if (m_velRHS[lev] != NULL)
	      {
		delete m_velRHS[lev];
	      }
	    m_velRHS[lev] = new LevelData<FArrayBox>(newDBL, SpaceDim, 
						     IntVect::Zero);

	    if (m_faceVelAdvection[lev] != NULL)
	      {
		delete m_faceVelAdvection[lev];
	      }
	    m_faceVelAdvection[lev] = new LevelData<FluxBox>(newDBL, 1, IntVect::Unit);

	    if (m_faceVelTotal[lev] != NULL)
	      {
		delete m_faceVelTotal[lev];
	      }
	    m_faceVelTotal[lev] = new LevelData<FluxBox>(newDBL, 1, IntVect::Unit);


	    if (m_diffusivity[lev] != NULL)
	      {
		delete m_diffusivity[lev];
	      }
	    m_diffusivity[lev] = new LevelData<FluxBox>(newDBL, 1, IntVect::Unit);


	    if (m_surfaceThicknessSource[lev] != NULL)
	      {
		delete m_surfaceThicknessSource[lev];
	      }
	    m_surfaceThicknessSource[lev] = 
	      new LevelData<FArrayBox>(newDBL,   1, IntVect::Unit) ;
	      
	    if (m_basalThicknessSource[lev] != NULL)
	      {
		delete m_basalThicknessSource[lev];
	      }
	    m_basalThicknessSource[lev] = 
	      new LevelData<FArrayBox>(newDBL,   1, IntVect::Unit) ;

	    if (m_recordThickness[lev] != NULL)
	      {
		delete m_recordThickness[lev];
	      }
	    m_recordThickness[lev] = 
	      new LevelData<FArrayBox>(newDBL,   1, IntVect::Unit) ;

	    if (m_divThicknessFlux[lev] != NULL)
	      {
		delete m_divThicknessFlux[lev];
	      }
	    m_divThicknessFlux[lev] = 
	      new LevelData<FArrayBox>(newDBL,   1, IntVect::Zero) ;


	    if (m_bHeatFlux[lev] != NULL)
	      {
		delete m_bHeatFlux[lev];
	      }
	    m_bHeatFlux[lev] = 
	      new LevelData<FArrayBox>(newDBL,   1, IntVect::Unit);

	    if (m_sHeatFlux[lev] != NULL)
	      {
		delete m_sHeatFlux[lev];
	      }
	    m_sHeatFlux[lev] = 
	      new LevelData<FArrayBox>(newDBL,   1, IntVect::Unit);



#if BISICLES_Z == BISICLES_LAYERED
	    if (m_layerXYFaceXYVel[lev] != NULL)
	      {
		delete m_layerXYFaceXYVel[lev];
	      }

	    m_layerXYFaceXYVel[lev] = new LevelData<FluxBox>
	      (newDBL, m_nLayers, IntVect::Unit);
	      
	    if (m_layerSFaceXYVel[lev] != NULL)
	      {
		delete m_layerSFaceXYVel[lev];
	      }
	      
	    m_layerSFaceXYVel[lev] = new LevelData<FArrayBox>
	      (newDBL, SpaceDim*(m_nLayers + 1), IntVect::Unit);
#endif		
	      
	  } // end loop over currently defined levels

	  
	// now ensure that any remaining levels are null pointers
	// (in case of de-refinement)
	for (int lev=new_finest_level+1; lev<m_old_thickness.size(); lev++)
	  {
	    if (m_old_thickness[lev] != NULL) 
	      {
		delete m_old_thickness[lev];
		m_old_thickness[lev] = NULL;
	      }


	    if (m_velocity[lev] != NULL) 
	      {
		delete m_velocity[lev];
		m_velocity[lev] = NULL;
	      }
	      
	    if (m_internalEnergy[lev] != NULL) 
	      {
		delete m_internalEnergy[lev];
		m_internalEnergy[lev] = NULL;
	      }

	      if (m_iceFrac[lev] != NULL) 
		{
		  delete m_iceFrac[lev];
		  m_iceFrac[lev] = NULL;
		}

#if BISICLES_Z == BISICLES_LAYERED
	    if (m_sInternalEnergy[lev] != NULL) 
	      {
		delete m_sInternalEnergy[lev];
		m_sInternalEnergy[lev] = NULL;
	      }
	    if (m_bInternalEnergy[lev] != NULL) 
	      {
		delete m_bInternalEnergy[lev];
		m_bInternalEnergy[lev] = NULL;
	      }
#endif	      	      
  
	    if (m_velRHS[lev] != NULL)
	      {
		delete m_velRHS[lev];
		m_velRHS[lev] = NULL;
	      }
	      
	    if (m_velBasalC[lev] != NULL)
	      {
		delete m_velBasalC[lev];
		m_velBasalC[lev] = NULL;
	      }

	  
	    DisjointBoxLayout emptyDBL;
	    m_amrGrids[lev] = emptyDBL;
	  }
      
	m_finest_level = new_finest_level;



	// set up counter of number of cells
	for (int lev=0; lev<=m_max_level; lev++)
	  {
	    m_num_cells[lev] = 0;
	    if (lev <= m_finest_level) 
	      {
		const DisjointBoxLayout& levelGrids = m_amrGrids[lev];
		LayoutIterator lit = levelGrids.layoutIterator();
		for (lit.begin(); lit.ok(); ++lit)
		  {
		    const Box& thisBox = levelGrids.get(lit());
		    m_num_cells[lev] += thisBox.numPts();
		  }
	      } 
	  }
      
      
	// finally, set up covered_level flags
	m_covered_level.resize(m_max_level+1, 0);
	// note that finest level can't be covered.
	for (int lev=m_finest_level-1; lev>=0; lev--)
	  {
          
	    // if the next finer level is covered, then this one is too.
	    if (m_covered_level[lev+1] == 1)
	      {
		m_covered_level[lev] = 1;
	      }
	    else
	      {
		// see if the grids finer than this level completely cover it
		IntVectSet fineUncovered(m_amrDomains[lev+1].domainBox());
		const DisjointBoxLayout& fineGrids = m_amrGrids[lev+1];
              
		LayoutIterator lit = fineGrids.layoutIterator();
		for (lit.begin(); lit.ok(); ++lit)
		  {
		    const Box& thisBox = fineGrids.get(lit());
		    fineUncovered.minus_box(thisBox);
		  }
              
		if (fineUncovered.isEmpty()) 
		  {
		    m_covered_level[lev] = 1;
		  }
	      }
	  } // end loop over levels to determine covered levels

	// this is a good time to check for remote ice
	if ((m_eliminate_remote_ice_after_regrid) 
	    && !(m_eliminate_remote_ice))
	  eliminateRemoteIce();
#ifdef REGRID_EH 	
	// Since we set E <- E*H earlier, set E <- E/H later now
	for (int lev=0; lev<= new_finest_level; ++lev)
	  {
	    for (DataIterator dit(m_amrGrids[lev]); dit.ok(); ++dit)
	      {
		FArrayBox& E = (*m_internalEnergy[lev])[dit];
		FArrayBox H(E.box(),1);
		H.copy((*m_old_thickness[lev])[dit]);
		H += 1.0e-10;
		for (int comp  = 0; comp < m_internalEnergy[0]->nComp(); comp++)
		  {
		    E.divide( H,0,comp,1);
		  }
	      }
	  }
#endif


	applyCalvingCriterion(CalvingModel::PostRegrid);

	if (m_evolve_velocity)
	  {
	    //velocity solver needs to be re-defined
	    defineSolver();
	    //solve velocity field, but use the previous initial residual norm in place of this one
	    //and force a solve even if other conditions (e.g the timestep interval condition) are not met
	    solveVelocityField(true, m_velocitySolveInitialResidualNorm);
	  }
	else
	  {
	    CH_assert(m_evolve_velocity);
	    MayDay::Error("AmrIce::regrid() not implemented for !m_evolve_velocity");
	  }
         
	  
      } // end if tags changed
    } // end if max level > 0 in the first place
  
  Real volumeAfter = computeTotalIce();
  Real volumeDifference = volumeAfter - volumeBefore;
  if (s_verbosity > 3) 
    { 
      
      pout() << "AmrIce::regrid: volume on input,output,difference =  " 
	     << volumeBefore << "," << volumeAfter << "," << volumeDifference << " m^3" << endl;
    }


  m_groundingLineProximity_valid = false;
  m_viscousTensor_valid = false;
}
      
                              
void 
AmrIce::tagCells(Vector<IntVectSet>& a_tags)
{
  
  if (s_verbosity > 3) 
    { 
      pout() << "AmrIce::tagCells" << endl;
    }

  
  int top_level = a_tags.size();
  top_level = min(m_tag_cap,min(top_level-1, m_finest_level));
  // loop over levels
  for (int lev=0; lev<=top_level; lev++)
    {
      IntVectSet& levelTags = a_tags[lev];
      tagCellsLevel(levelTags, lev);
      IntVectSet& tagSubset = m_vectTagSubset[lev];
      if ( tagSubset.numPts() > 0)
	{
	  levelTags &= tagSubset;
	}
    }

  //throw away any coarse level tags outside m_tag_subset
  // if (s_verbosity > 3) 
  //   { 
  //     pout() << "AmrIce::tagCells, subset II" << endl;
  //   }
  // if (m_tag_subset.numPts() > 0)
  //   {
  //     IntVectSet tag_subset = m_tag_subset;
  //     a_tags[0] &= tag_subset;
  //     for (int lev = 1; lev <= top_level; lev++)
  // 	{
  // 	  tag_subset.refine(m_refinement_ratios[lev-1]);
  // 	  a_tags[lev] &= tag_subset;
  // 	}

  //   }

}

void
AmrIce::tagCellsLevel(IntVectSet& a_tags, int a_level)
{

  if (s_verbosity > 4) 
    { 
      pout() << "AmrIce::tagCellsLevel " << a_level << endl;
    }


  // base tags on undivided gradient of velocity
  // first stab -- don't do BC's; just do one-sided
  // stencils at box edges (hopefully good enough), 
  // since doing BC's properly is somewhat expensive.

  DataIterator dit = m_velocity[a_level]->dataIterator();
  
  LevelData<FArrayBox>& levelVel = *m_velocity[a_level];

  const DisjointBoxLayout& levelGrids = m_amrGrids[a_level];

  const LevelSigmaCS& levelCS = *m_vect_coordSys[a_level];

  // need to ensure that ghost cells are set properly
  levelVel.exchange(levelVel.interval());

  const LevelData<FluxBox>& levelFaceH = levelCS.getFaceH();

  LevelData<FArrayBox>& levelC = *m_velBasalC[a_level];

  IntVectSet local_tags;
  if (m_tagOnGradVel)
    {
      for (dit.begin(); dit.ok(); ++dit)
        {
          // note that we only need one component here
          // because the fortran subroutine stores the max(abs(grad)) 
          // over all components into the 0th position
          FArrayBox gradVel(levelGrids[dit()], 1);
          
          for (int dir=0; dir<SpaceDim; dir++)
            {
              const Box b = levelGrids[dit()];
              const Box bcenter = b & grow ( m_amrDomains[a_level], 
                                             -BASISV(dir) );
              const Box blo = b & adjCellLo( bcenter, dir );
              const Box bhi = b & adjCellHi( bcenter, dir );
              const int haslo = ! blo.isEmpty();
              const int hashi = ! bhi.isEmpty();
              FORT_UNDIVIDEDGRAD ( CHF_FRA1(gradVel,0),
                                   CHF_CONST_FRA(levelVel[dit()]),
                                   CHF_BOX(bcenter),
                                   CHF_BOX(blo),
                                   CHF_BOX(bhi),
                                   CHF_CONST_INT(dir),
                                   CHF_CONST_INT(haslo),
                                   CHF_CONST_INT(hashi));
              
              
              // now tag cells based on values
              BoxIterator bit(levelGrids[dit()]);
              for (bit.begin(); bit.ok(); ++bit)
                {
                  const IntVect& iv = bit();
                  if (abs(gradVel(iv,0)) > m_tagging_val) 
                    local_tags |= iv;
                } // end loop over cells
            } // end loop over directions
        } // end loop over grids
    } // end if tag on grad vel


  // tag on laplacian(velocity)     
  if (m_tagOnLapVel | m_tagOnGroundedLapVel)
    {
      for (dit.begin(); dit.ok(); ++dit)
        {
          FArrayBox lapVel(levelGrids[dit()], SpaceDim);
	  const BaseFab<int>& mask = levelCS.getFloatingMask()[dit];
          lapVel.setVal(0.0);
          Real alpha = 0;
          Real beta = 1.0;
              
          // use undivided laplacian (set dx = 1)
          Real bogusDx = 1.0;
	  Box lapBox = levelVel[dit].box();
	  lapBox.grow(-2);
          lapBox &=  levelGrids[dit];
          // assumes that ghost cells boundary conditions are properly set
          FORT_OPERATORLAP(CHF_FRA(lapVel),
                           CHF_FRA(levelVel[dit]),
                           CHF_BOX(lapBox),
                           CHF_CONST_REAL(bogusDx),
                           CHF_CONST_REAL(alpha),
                           CHF_CONST_REAL(beta));
                            
          // now tag cells based on values
          BoxIterator bit(lapBox);
	  
          for (bit.begin(); bit.ok(); ++bit)
            {
              const IntVect& iv = bit();
	      for (int comp=0; comp<lapVel.nComp(); comp++)
		{
		  if ( (m_tagOnGroundedLapVel && mask(iv) == GROUNDEDMASKVAL) | m_tagOnLapVel )
		    {
		      if ( (abs(lapVel(iv,comp)) > m_laplacian_tagging_val) 
			   &&  (levelC[dit](iv) < m_laplacian_tagging_max_basal_friction_coef)) 
			local_tags |= iv;
		    }
		}
	      
            } // end loop over cells
        } // end loop over grids
    } // end if tag on laplacian(vel)
    

  // sometimes, it is easier to note where the grounding line is
  // and refine to the maximum level there
  if (m_tagGroundingLine)
    {
     
      for (dit.begin(); dit.ok(); ++dit)
	{
	  Box sbox = levelGrids[dit()];
	  //sbox.grow(-1);
	  const BaseFab<int>& mask = levelCS.getFloatingMask()[dit];
	  for (BoxIterator bit(sbox) ; bit.ok(); ++bit)
	    {
	      const IntVect& iv = bit();
	      for (int dir = 0; dir < SpaceDim; ++dir)
		{
		  int  tdir = (dir + 1)%SpaceDim;
		  const IntVect& ivm = iv - BASISV(dir);
		  const IntVect& ivp = iv + BASISV(dir);
		 
		  if (mask(iv) == GROUNDEDMASKVAL &&  levelC[dit](iv) <  m_groundingLineTaggingMaxBasalFrictionCoef)
		    {
		      if (mask(ivm) == FLOATINGMASKVAL || mask(ivm) == OPENSEAMASKVAL )
			{
			  if (std::abs(levelVel[dit](iv,dir)) > m_groundingLineTaggingMinVel
			      || std::abs(levelVel[dit](ivm,dir)) > m_groundingLineTaggingMinVel
			      || std::abs(levelVel[dit](iv,tdir)) > m_groundingLineTaggingMinVel
			      || std::abs(levelVel[dit](ivm,tdir)) > m_groundingLineTaggingMinVel)
			    {
			      local_tags |= iv;  
			      local_tags |= ivm;
			    }   
			}
			 
		      if ( mask(ivp) == FLOATINGMASKVAL || mask(ivp) == OPENSEAMASKVAL)
			{
			  if (std::abs(levelVel[dit](iv,dir)) > m_groundingLineTaggingMinVel
			      || std::abs(levelVel[dit](ivp,dir)) > m_groundingLineTaggingMinVel
			      || std::abs(levelVel[dit](iv,tdir)) > m_groundingLineTaggingMinVel
			      || std::abs(levelVel[dit](ivp,tdir)) > m_groundingLineTaggingMinVel)
			    {
			      local_tags |= iv;  
			      local_tags |= ivp;
			    } 
			}
		    }
		}
	    }
	}
    }
  
  // tag on |vel| * dx > m_velDx_tagVal. This style of tagging is used in the AMRControl.cpp
  // (but with the observed field), so can be used to construct similar meshes
  if (m_tagVelDx)
    {
      for (dit.begin(); dit.ok(); ++dit)
        {
	  const Box& box = levelGrids[dit()];
	  const BaseFab<int>& mask = levelCS.getFloatingMask()[dit];
	  const FArrayBox& vel = levelVel[dit];
	  for (BoxIterator bit(box) ; bit.ok(); ++bit)
	    {
	      const IntVect& iv = bit();
	      if ((mask(iv) == GROUNDEDMASKVAL && a_level < m_velDx_tagVal_finestLevelGrounded) ||
		  (mask(iv) == FLOATINGMASKVAL && a_level < m_velDx_tagVal_finestLevelFloating) )
		{
		  Real v = 0.0;
		  for (int dir = 0; dir < SpaceDim; ++dir)
		    {
		      v += std::pow(vel(iv,dir),2);
		    }
		  if (sqrt(v)*dx(a_level)[0] > m_velDx_tagVal)
		    {
		      local_tags |= iv;
		    }
		}
	    }
	}
    }

  // tag on div(H grad (vel)) 
  if (m_tagOndivHgradVel)
    {
      for (dit.begin(); dit.ok(); ++dit)
        {      
	  Box box = levelGrids[dit];
	  box.grow(-1);
	  const FluxBox& faceH = levelFaceH[dit];
	  const FArrayBox& vel = levelVel[dit];
	  BoxIterator bit(levelGrids[dit()]);
	  for (bit.begin(); bit.ok(); ++bit)
	    {
	      const IntVect& iv = bit();
	      for (int comp=0; comp < vel.nComp() ; comp++)
		{
		  Real t = 0.0;
		  for (int dir=0; dir < SpaceDim ; dir++)
		    {		  
		      IntVect ivp = iv + BASISV(dir);
		      IntVect ivm = iv - BASISV(dir);
		  
		      t += faceH[dir](iv) * (vel(iv,comp)-vel(ivm,comp))
			- faceH[dir](ivp) * (vel(ivp,comp)-vel(iv,comp));
		      
		    }
	
		  if (abs(t) > m_divHGradVel_tagVal)
		    {
		   
		      local_tags |= iv;
		    }
		}
	    }// end loop over cells
        } // end loop over grids
    } // end if tag on div(H grad (vel)) 
  
  if (m_tagOnEpsSqr)
    {
      IntVect tagsGhost = IntVect::Zero;
      LevelData<FArrayBox> epsSqr(levelGrids, 1, tagsGhost);
      LevelData<FArrayBox>* crseVelPtr = NULL;
      int nRefCrse = -1;
      if (a_level > 0)
        {
          crseVelPtr = m_velocity[a_level-1];
          nRefCrse = m_refinement_ratios[a_level-1];
        }

      m_constitutiveRelation->computeStrainRateInvariant(epsSqr,
                                                         levelVel,
                                                         crseVelPtr, 
                                                         nRefCrse,
                                                         levelCS,
                                                         tagsGhost);

      
      for (dit.begin(); dit.ok(); ++dit)
        {                    
          // now tag cells based on values
          // want undivided gradient
          epsSqr[dit] *= m_amrDx[a_level] * m_amrDx[a_level] ;
          Real levelTagVal = m_epsSqr_tagVal;
          BoxIterator bit(levelGrids[dit()]);
          for (bit.begin(); bit.ok(); ++bit)
            {
              const IntVect& iv = bit();
              if (abs(epsSqr[dit](iv,0)) > levelTagVal) 
                local_tags |= iv;
            } // end loop over cells
        } // end loop over grids
    } // end if tagging on strain rate invariant


  if (m_tagOnVelRHS)
    {
      for (dit.begin(); dit.ok(); ++dit)
        {                          
          const FArrayBox& thisVelRHS = (*m_velRHS[a_level])[dit];

          // now tag cells based on values
          // want RHS*dx (undivided gradient)
          Real levelTagVal = m_velRHS_tagVal/m_amrDx[a_level];
          BoxIterator bit(levelGrids[dit()]);
          for (int comp=0; comp<thisVelRHS.nComp(); comp++)
            {
              for (bit.begin(); bit.ok(); ++bit)
                {
                  const IntVect& iv = bit();
                  if (abs(thisVelRHS(iv,comp)) > levelTagVal) 
                    local_tags |= iv;
                } // end loop over cells
            } // end loop over components
        } // end loop over grids
    } // end if tagging on velRHS
  
  // tag cells  with thin cavities
  if (m_tag_thin_cavity)
    {
      for (dit.begin(); dit.ok(); ++dit)
	{
	  const BaseFab<int>& mask = levelCS.getFloatingMask()[dit];
	  Box gridBox = levelGrids[dit];
	  const FArrayBox& H  = levelCS.getH()[dit];
	  for (BoxIterator bit(gridBox); bit.ok(); ++bit)
	    {
	      const IntVect& iv = bit();
	      if (mask(iv) == FLOATINGMASKVAL && 
		  H(iv) <  m_tag_thin_cavity_thickness)
		{
		  local_tags |= iv;
		}
	    }
	}
    }

  // tag cells where thickness goes to zero
  if (m_tagMargin)
    {
      const LevelData<FArrayBox>& levelH = levelCS.getH();
      for (dit.begin(); dit.ok(); ++dit)
        {
          Box gridBox = levelGrids[dit];
          const FArrayBox& H = levelH[dit];

          for (BoxIterator bit(gridBox); bit.ok(); ++bit)
            {
              const IntVect& iv = bit();
	      if ( a_level < m_margin_tagVal_finestLevel )
		{
		  // neglect diagonals for now...
		  for (int dir=0; dir<SpaceDim; dir++)
		    {
		      IntVect ivm = iv - BASISV(dir);
		      IntVect ivp = iv + BASISV(dir);
		      if ( (H(iv,0) > 0) && (H(ivm,0) < TINY_THICKNESS) )
			{
			  local_tags |= iv;
			  local_tags |= ivm;
			} // end if low-side margin
		      if ( (H(iv,0) > 0) && (H(ivp,0) < TINY_THICKNESS) )
			{
			  local_tags |= iv;
			  local_tags |= ivp;
			} // end high-side margin
		    } // end loop over directions
		}
	    } // end loop over cells
	} // end loop over boxes
    } // end if tagging on ice margins

  // tag anywhere there's ice
  if (m_tagAllIce)
    {
      const LevelData<FArrayBox>& levelH = levelCS.getH();
      for (dit.begin(); dit.ok(); ++dit)
        {
          Box gridBox = levelGrids[dit];
          const FArrayBox& H = levelH[dit];
          BoxIterator bit(gridBox);
          for (bit.begin(); bit.ok(); ++bit)
            {
              const IntVect& iv = bit();
              if (H(iv,0) > 0.0)
                {
                  local_tags |= iv;
                }
            } // end bit loop
        } // end loop over boxes
    } // end if tag all ice

  


  // tag anywhere and everywhere
  if (m_tagEntireDomain)
    {
      // this is super-simple...
      Box domainBox = m_amrDomains[a_level].domainBox();
      local_tags |= domainBox;
          
    } // end if tag entire domain



#ifdef HAVE_PYTHON
  if (m_tagPython)
    {
      //tag via a python function f(x,y,dx,H,R) (more args to come)
      // 
      Vector<Real> args(SpaceDim + 3);
      Vector<Real> rval;
      for (dit.begin(); dit.ok(); ++dit)
        {
	  for (BoxIterator bit(levelGrids[dit]); bit.ok(); ++bit)
	    {
	      const IntVect& iv = bit();
	      int i = 0;
	      for (int dir=0; dir < SpaceDim; dir++)
		{
		  args[i++] = (Real(iv[dir]) + 0.5)*m_amrDx[a_level];
		}
	      args[i++]  = m_amrDx[a_level];
	      args[i++] = levelCS.getH()[dit](iv,0);
	      args[i++] = levelCS.getTopography()[dit](iv,0);				    
	      PythonInterface::PythonEval(m_tagPythonFunction, rval,  args);
	      if (rval[0] > 0.0)
		local_tags |= iv;
	    } // end bit loop
	} // end loop over boxes
    } // end if tag via python
#endif

  // now buffer tags
  
  local_tags.grow(m_tags_grow);
  for (int dir = 0; dir < SpaceDim; dir++)
    {
      if (m_tags_grow_dir[dir] > m_tags_grow)
	local_tags.grow(dir, std::max(0,m_tags_grow_dir[dir]-m_tags_grow));
    }
  local_tags &= m_amrDomains[a_level];

  a_tags = local_tags;

}

void
AmrIce::tagCellsInit(Vector<IntVectSet>& a_tags)
{

  if (s_verbosity > 3) 
    { 
      pout() << "AmrIce::tagCellsInit" << endl;
    }


  tagCells(a_tags);
  m_vectTags = a_tags;
  
}


void
AmrIce::initGrids(int a_finest_level)
{

  if (s_verbosity > 3) 
    { 
      pout() << "AmrIce::initGrids" << endl;
    }


  m_finest_level = 0;
  // first create base level
  Vector<Box> baseBoxes;
  domainSplit(m_amrDomains[0], baseBoxes, m_max_base_grid_size, 
              m_block_factor);

  Vector<int> procAssign(baseBoxes.size());
  LoadBalance(procAssign,baseBoxes);
  
  DisjointBoxLayout baseGrids(baseBoxes, procAssign, m_amrDomains[0]);

  if (s_verbosity > 3) 
    {
      long long numCells0 = baseGrids.numCells();
      pout() << "Level 0: " << numCells0 << " cells: " << baseGrids << endl;
    }

  m_amrGrids.resize(m_max_level+1);
  m_amrGrids[0] = baseGrids;

  levelSetup(0,baseGrids);

  LevelData<FArrayBox>& baseLevelVel = *m_velocity[0];
  DataIterator baseDit = baseGrids.dataIterator();
  for (baseDit.begin(); baseDit.ok(); ++baseDit)
    {
      // initial guess at base-level velocity is zero
      baseLevelVel[baseDit].setVal(0.0);
    }

  // define solver before calling initData
  defineSolver();

  // initialize base level data
  initData(m_vect_coordSys,
           m_velocity);

  bool moreLevels = (m_max_level > 0);
  int baseLevel = 0;
  
  BRMeshRefine meshrefine;
  if (moreLevels)
    {
      meshrefine.define(m_amrDomains[0], m_refinement_ratios,
                        m_fill_ratio, m_block_factor, 
                        m_nesting_radius, m_max_box_size);
    }
  
  Vector<IntVectSet> tagVect(m_max_level);
  
  Vector<Vector<Box> > oldBoxes(1);
  Vector<Vector<Box> > newBoxes;
  oldBoxes[0] = baseBoxes;
  newBoxes = oldBoxes;
  int new_finest_level = 0;

  while (moreLevels)
    {
      // default is moreLevels = false
      // (only repeat loop in the case where a new level is generated
      // which is still coarser than maxLevel)
      moreLevels = false;
      tagCellsInit(tagVect);
      
      // two possibilities -- need to generate grids
      // level-by-level, or we are refining all the
      // way up for the initial time.  check to 
      // see which it is by seeing if the finest-level
      // tags are empty
      if (tagVect[m_max_level-1].isEmpty())
        {
          int top_level = m_finest_level;
          int old_top_level = top_level;
          new_finest_level = meshrefine.regrid(newBoxes,
                                               tagVect, baseLevel,
                                               top_level,
                                               oldBoxes);

          if (new_finest_level > top_level) top_level++;
          oldBoxes = newBoxes;

          // now see if we need another pass through grid generation
          if ((top_level < m_max_level) && (top_level > old_top_level) && (new_finest_level <= m_tag_cap))
            {
              moreLevels = true;
            }
          
        }
      else 
        {
          
          // for now, define old_grids as just domains
          oldBoxes.resize(m_max_level+1);
          for (int lev=1; lev<=m_max_level; lev++) 
            {
              oldBoxes[lev].push_back(m_amrDomains[lev].domainBox());
            }
          
          int top_level = m_max_level -1;
          new_finest_level = meshrefine.regrid(newBoxes,
                                               tagVect, baseLevel,
                                               top_level,
                                               oldBoxes);
        }
      
  
      // now loop through levels and define
      for (int lev=baseLevel+1; lev<= new_finest_level; ++lev)
        {
          int numGridsNew = newBoxes[lev].size();
          Vector<int> procIDs(numGridsNew);
          LoadBalance(procIDs, newBoxes[lev]);
          const DisjointBoxLayout newDBL(newBoxes[lev], procIDs,
                                         m_amrDomains[lev]);
          m_amrGrids[lev] = newDBL;

          if (s_verbosity > 2)
            {
              long long levelNumCells = newDBL.numCells();          
              pout() << "   Level " << lev << ": " 
                     << levelNumCells << " cells: " 
                     << m_amrGrids[lev] << endl;
            }
              

          levelSetup(lev,m_amrGrids[lev]);
	  m_A_valid = false;
	  m_groundingLineProximity_valid = false;
	  m_viscousTensor_valid = false;

        } // end loop over levels

      m_finest_level = new_finest_level;
      
      // finally, initialize data on final hierarchy
      // only do this if we've created new levels
      if (m_finest_level > 0) 
        {
          defineSolver();

          initData(m_vect_coordSys,
                   m_velocity);
        }
    } // end while more levels to do

  


}


void
AmrIce::setupFixedGrids(const std::string& a_gridFile)
{
  Vector<Vector<Box> > gridvect;
  
  if (procID() == uniqueProc(SerialTask::compute))
    {
      gridvect.push_back(Vector<Box>(1,m_amrDomains[0].domainBox()));
    
      // read in predefined grids
      ifstream is(a_gridFile.c_str(), ios::in);
      
      if (is.fail())
        {
          MayDay::Error("Cannot open grids file");
        }

      // format of file:
      //   number of levels, then for each level (starting with level 1):
      //   number of grids on level, list of boxes
      int inNumLevels;
      is >> inNumLevels;

      CH_assert (inNumLevels <= m_max_level+1);

      if (s_verbosity > 3)
        {
          pout() << "numLevels = " << inNumLevels << endl;
        }

      while (is.get() != '\n');

      gridvect.resize(inNumLevels);

      // check to see if coarsest level needs to be broken up
      domainSplit(m_amrDomains[0],gridvect[0], m_max_base_grid_size, 
                  m_block_factor);

      if (s_verbosity >= 3)
        {
          pout() << "level 0: ";
          for (int n=0; n < gridvect[0].size(); n++)
            {
              pout() << gridvect[0][n] << endl;
            }
        }

      // now loop over levels, starting with level 1
      int numGrids = 0;
      for (int lev=1; lev<inNumLevels; lev++) 
        {
          is >> numGrids;

          if (s_verbosity >= 3)
            {
              pout() << "level " << lev << " numGrids = " 
                     << numGrids <<  endl;
              pout() << "Grids: ";
            }

          while (is.get() != '\n');

          gridvect[lev].resize(numGrids);

          for (int i=0; i<numGrids; i++)
            {
              Box bx;
              is >> bx;

              while (is.get() != '\n');

              // quick check on box size
              Box bxRef(bx);

              if (bxRef.longside() > m_max_box_size)
                {
                  pout() << "Grid " << bx << " too large" << endl;
                  MayDay::Error();
                }

              if (s_verbosity >= 3) 
                {
                  pout() << bx << endl;
                }

              gridvect[lev][i] = bx;
            } // end loop over boxes on this level
        } // end loop over levels
    } // end if serial proc

  // broadcast results
  broadcast(gridvect, uniqueProc(SerialTask::compute));

  // now create disjointBoxLayouts and allocate grids

  m_amrGrids.resize(m_max_level+1);
  IntVect sigmaCSGhost = m_num_thickness_ghost*IntVect::Unit;
  m_vect_coordSys.resize(m_max_level+1);
  
  // probably eventually want to do this differently
  RealVect dx = m_amrDx[0]*RealVect::Unit;

  for (int lev=0; lev<gridvect.size(); lev++)
    {
      int numGridsLev = gridvect[lev].size();
      Vector<int> procIDs(numGridsLev);
      LoadBalance(procIDs, gridvect[lev]);
      const DisjointBoxLayout newDBL(gridvect[lev],
                                     procIDs, 
                                     m_amrDomains[lev]);

      m_amrGrids[lev] = newDBL;

      // build storage for this level

      levelSetup(lev, m_amrGrids[lev]);
      if (lev < gridvect.size()-1)
        {
          dx /= m_refinement_ratios[lev];
        }
    }
  
  // finally set finest level and initialize data on hierarchy
  m_finest_level = gridvect.size() -1;

  // define solver before calling initData
  defineSolver();
  
  initData(m_vect_coordSys, m_velocity);

}
    

void
AmrIce::levelSetup(int a_level, const DisjointBoxLayout& a_grids)
{
  IntVect ghostVect = IntVect::Unit;
  // 4 ghost cells needed for advection. Could later go back and
  // make this a temporary if the additional storage becomes an issue...
  IntVect thicknessGhostVect = m_num_thickness_ghost*IntVect::Unit;
  m_old_thickness[a_level]->define(a_grids, 1,
                                   thicknessGhostVect);

  if (a_level == 0 || m_velocity[a_level] == NULL)
    {
      m_velocity[a_level] = new LevelData<FArrayBox>(a_grids, SpaceDim,
                                                     ghostVect);
    }
  else
    {
      // do velocity a bit differently in order to use previously 
      // computed velocity field as an initial guess
      {
        LevelData<FArrayBox>* newVelPtr = new LevelData<FArrayBox>(a_grids,
                                                                   SpaceDim,
                                                                   ghostVect);
        
        // first do interp from coarser level
        FineInterp velInterp(a_grids, SpaceDim, 
                             m_refinement_ratios[a_level-1],
                             m_amrDomains[a_level]);
        
        velInterp.interpToFine(*newVelPtr, *m_velocity[a_level-1]);
        
        // can only copy from existing level if we're not on the
        // newly created level
        //if (a_level != new_finest_level)
        if (m_velocity[a_level]->isDefined())
          {
            m_velocity[a_level]->copyTo(*newVelPtr);
          }
        
        // finally, do an exchange (this may wind up being unnecessary)
        newVelPtr->exchange();
        
        delete (m_velocity[a_level]);
        m_velocity[a_level] = newVelPtr;
      }
    } // end interpolate/copy new velocity

  levelAllocate(&m_faceVelAdvection[a_level] ,a_grids,1,IntVect::Unit);
  levelAllocate(&m_faceVelTotal[a_level],a_grids,1,IntVect::Unit);
  levelAllocate(&m_diffusivity[a_level],a_grids, 1, IntVect::Zero);
  levelAllocate(&m_iceFrac[a_level],a_grids, 1, IntVect::Unit);

#if BISICLES_Z == BISICLES_LAYERED
  levelAllocate(&m_layerXYFaceXYVel[a_level] ,a_grids, m_nLayers, IntVect::Unit);
  levelAllocate(&m_layerSFaceXYVel[a_level], a_grids, SpaceDim*(m_nLayers + 1), IntVect::Unit);
#endif

  levelAllocate(&m_velBasalC[a_level],a_grids, 1, ghostVect);
  levelAllocate(&m_cellMuCoef[a_level],a_grids, 1, ghostVect);
  levelAllocate(&m_velRHS[a_level],a_grids, SpaceDim,  IntVect::Zero);
  levelAllocate(&m_surfaceThicknessSource[a_level], a_grids,   1, IntVect::Unit) ;
  levelAllocate(&m_basalThicknessSource[a_level], a_grids,  1, IntVect::Unit) ;
  levelAllocate(&m_divThicknessFlux[a_level],a_grids,   1, IntVect::Zero) ;
  levelAllocate(&m_calvedIceThickness[a_level],a_grids, 1, IntVect::Unit);
  levelAllocate(&m_removedIceThickness[a_level],a_grids, 1, IntVect::Unit);
  levelAllocate(&m_addedIceThickness[a_level],a_grids, 1, IntVect::Unit);
  levelAllocate(&m_melangeThickness[a_level],a_grids, 1, IntVect::Unit);
  levelAllocate(&m_recordThickness[a_level],a_grids, 1, IntVect::Unit);
  levelAllocate(&m_deltaTopography[a_level],a_grids, 1, IntVect::Zero);
  // probably eventually want to do this differently
  RealVect dx = m_amrDx[a_level]*RealVect::Unit;

  //IntVect sigmaCSGhost = IntVect::Unit;
  IntVect sigmaCSGhost = thicknessGhostVect;
  m_vect_coordSys.resize(m_max_level+1);
  m_vect_coordSys[a_level] = RefCountedPtr<LevelSigmaCS >(new LevelSigmaCS(a_grids, 
									   dx,
									   sigmaCSGhost));
  m_vect_coordSys[a_level]->setIceDensity(m_iceDensity);
  m_vect_coordSys[a_level]->setWaterDensity(m_seaWaterDensity);
  m_vect_coordSys[a_level]->setGravity(m_gravity);

#if BISICLES_Z == BISICLES_LAYERED
  //in poor-man's multidim mode, use one FArrayBox component per layer
  //to hold the 3D internalEnergy field
  levelAllocate(&m_internalEnergy[a_level],a_grids, m_nLayers,thicknessGhostVect);
  levelAllocate(&m_sInternalEnergy[a_level], a_grids, 1, thicknessGhostVect);
  levelAllocate(&m_bInternalEnergy[a_level], a_grids, 1, thicknessGhostVect);
  levelAllocate(&m_sHeatFlux[a_level], a_grids, 1, thicknessGhostVect);
  levelAllocate(&m_bHeatFlux[a_level], a_grids, 1, thicknessGhostVect);
  m_vect_coordSys[a_level]->setFaceSigma(getFaceSigma());

#elif BISICLES_Z == BISICLES_FULLZ
  levelAllocate(&m_internalEnergy[a_level],a_grids, 1, thicknessGhostVect);
#endif




}

void
AmrIce::initData(Vector<RefCountedPtr<LevelSigmaCS> >& a_vectCoordSys,
                 Vector<LevelData<FArrayBox>* >& a_velocity)
{

  if (s_verbosity > 3) 
    { 
      pout() << "AmrIce::initData" << endl;
    }

  m_groundingLineProximity_valid = false;
  m_A_valid = false;

  for (int lev=0; lev<=m_finest_level; lev++)
    {
      RealVect levelDx = m_amrDx[lev]*RealVect::Unit;
      m_thicknessIBCPtr->define(m_amrDomains[lev],levelDx[0]);
      LevelSigmaCS* crsePtr = (lev > 0)?&(*m_vect_coordSys[lev-1]):NULL;
      int refRatio = (lev > 0)?m_refinement_ratios[lev-1]:0;
      
      m_thicknessIBCPtr->initializeIceGeometry(*a_vectCoordSys[lev],
					       levelDx,
					       m_domainSize,
					       m_time,
					       crsePtr,
					       refRatio);
      

	


      a_vectCoordSys[lev]->recomputeGeometry(crsePtr, refRatio);



      const LevelData<FArrayBox>& levelThickness = m_vect_coordSys[lev]->getH();
      setIceFrac(levelThickness, lev);
      a_vectCoordSys[lev]->recomputeGeometry(crsePtr, refRatio);

      // initialize oldH to be the current value
      LevelData<FArrayBox>& currentH = a_vectCoordSys[lev]->getH();
      currentH.copyTo(*m_old_thickness[lev]);

#if BISICLES_Z == BISICLES_LAYERED
      m_internalEnergyIBCPtr->initializeIceInternalEnergy
	(*m_internalEnergy[lev], *m_sInternalEnergy[lev], *m_bInternalEnergy[lev], *this, lev, 0.0);

#elif BISICLES_Z == BISICLES_FULLZ
      m_internalEnergyIBCPtr->initializeIceInternalEnergy(*m_temperature[lev],*this, lev, 0.0);
#endif
    }

  setToZero(m_melangeThickness);

  // this is a good time to check for remote ice
  // (don't bother if we're doing it as a matter of course, since we'd
  // wind up doing it 2x)
  if ((m_eliminate_remote_ice_after_regrid) && !(m_eliminate_remote_ice))
    eliminateRemoteIce();
  
  setToZero(m_deltaTopography);

  applyCalvingCriterion(CalvingModel::Initialization);

  // now call velocity solver to initialize velocity field, force a solve no matter what the time step
  solveVelocityField(true);

  // may be necessary to average down here
  for (int lev=m_finest_level; lev>0; lev--)
    {
      CoarseAverage avgDown(m_amrGrids[lev],
                            SpaceDim, m_refinement_ratios[lev-1]);
      avgDown.averageToCoarse(*m_velocity[lev-1], *m_velocity[lev]);
    }


  //#define writePlotsImmediately
#ifdef  writePlotsImmediately
  if (m_plot_interval >= 0)
    {
#ifdef CH_USE_HDF5
      writePlotFile();
#endif
    }
#endif


}

/// solve for velocity field
void
AmrIce::solveVelocityField(bool a_forceSolve, Real a_convergenceMetric)
{

  CH_TIME("AmrIce::solveVelocityField");

  notifyObservers(Observer::PreVelocitySolve);

  if (m_eliminate_remote_ice)
    eliminateRemoteIce();

  //ensure A is up to date
#if BISICLES_Z == BISICLES_LAYERED
  if (!m_A_valid)
    {
      computeA(m_A,m_sA,m_bA,m_internalEnergy,m_sInternalEnergy,m_bInternalEnergy,
	       m_vect_coordSys);
      m_A_valid = true;
    }
#else
  MayDay::Error("AmrIce::SolveVelocityField full z calculation of A not done"); 
#endif
  //certainly the viscous tensr field will need re-computing
  m_viscousTensor_valid = false;

  // define basal friction
  Vector<LevelData<FArrayBox>* > vectC(m_finest_level+1, NULL);
  Vector<LevelData<FArrayBox>* > vectC0(m_finest_level+1, NULL);
  Vector<LevelData<FArrayBox>* > vectRhs(m_finest_level+1, NULL);
  for (int lev=0; lev<=m_finest_level; lev++)
    {
      vectRhs[lev] = m_velRHS[lev];
      vectC[lev] = m_velBasalC[lev];
      vectC0[lev] = new LevelData<FArrayBox>; vectC0[lev]->define(*vectC[lev]);
    }

  //
  setMuCoefficient(m_cellMuCoef);

  // set basal friction coeffs C,C0. C = 0 for floating ice. C0 != 0 at walls
  setBasalFriction(vectC, vectC0);

  // right hand side of the stress-balance
  defineVelRHS(vectRhs);




  
  // write out sumRhs if appropriate
  if (s_verbosity > 3)
    {
      Real sumRhs = computeSum(vectRhs,
                               m_refinement_ratios,
                               m_amrDx[0],
                               Interval(0,0),
                               0);

      pout() << "Sum(rhs) for velocity solve = " << sumRhs << endl;

    }
  
  // put this in place to catch runs where plotfile writing is
  // going to hang _before_ I waste a few hours waiting for the 
  // velocity solve
  //#define writeTestPlots
#ifdef  writeTestPlots
  if (m_plot_interval >= 0 && m_cur_step == 0)
    {
      writePlotFile();
    }
#endif

  if (m_doInitialVelSolve) 
    {      
      if (m_finest_level == 0 && m_doInitialVelGuess)
        {
          // only really want or need to do this once
          m_doInitialVelGuess = false;
	  if (m_initialGuessType == SlidingLaw)
	    {
	      pout() << "computing an initial guess via a sliding law u = rhs/C "  << endl;
	      // compute initial guess as rhs/beta
	      LevelData<FArrayBox>& vel = *m_velocity[0];
	      LevelData<FArrayBox>& C = *m_velBasalC[0];
	      LevelData<FArrayBox>& rhs = *m_velRHS[0];
	      const DisjointBoxLayout& levelGrids = m_amrGrids[0];
	      
	      DataIterator dit = vel.dataIterator();
	      for (dit.begin(); dit.ok(); ++dit)
		{
		  FORT_VELINITIALGUESS(CHF_FRA(vel[dit]),
				       CHF_FRA(rhs[dit]),
				       CHF_FRA1(C[dit],0),
				       CHF_BOX(levelGrids[dit]));
		}
	    }
	  else if (m_initialGuessType == ConstMu)
	    {
	      if (s_verbosity > 3) 
		{
		  pout() << "computing an initial guess by solving the velocity equations "
			 <<" with constant mu = " 
			 << m_initialGuessConstMu   
			 << " and constant initial velocity = " << m_initialGuessConstVel
			 << endl;
		}

	      // compute initial guess by solving a linear problem with a
	      // modest constant viscosity
	      constMuRelation* newPtr = new constMuRelation;
	      newPtr->setConstVal(m_initialGuessConstMu);
	      for (int lev=0; lev < m_finest_level + 1; lev++)
		{
		  for (DataIterator dit(m_amrGrids[lev]);dit.ok();++dit)
		    {
		      for (int dir = 0; dir < SpaceDim; dir++)
			{
			  (*m_velocity[lev])[dit].setVal(m_initialGuessConstVel[dir],dir);
			}
		    }

		}

              // do this by saving the exisiting velSolver and 
              // constitutiveRelation, re-calling defineSolver, then 
              // doing solve.
              IceVelocitySolver* velSolverSave = m_velSolver;
              ConstitutiveRelation* constRelSave = m_constitutiveRelation;
              int solverTypeSave = m_solverType;

              // new values prior to calling defineSolver
             
              m_constitutiveRelation = static_cast<ConstitutiveRelation*>(newPtr);

	      Real finalNorm = 0.0, initialNorm = 0.0, convergenceMetric = -1.0;
	      //Vector<LevelData<FArrayBox>* > muCoef(m_finest_level + 1,NULL);
	      int rc = -1;
	      if (m_initialGuessSolverType == JFNK)
		{
		  //JFNK can be instructed to assume a linear solve
		  m_solverType = JFNK;
                  // (DFM 2/4/14) this is not a memory leak -- velSolver is 
                  // saved in velSolverSave and will be swapped back after 
                  // the initial guess solve
		  m_velSolver = NULL;
		  defineSolver();
		  JFNKSolver* jfnkSolver = dynamic_cast<JFNKSolver*>(m_velSolver);
		  CH_assert(jfnkSolver != NULL);
		  const bool linear = true;
		  rc = jfnkSolver->solve( m_velocity, 
					  m_calvedIceThickness, m_addedIceThickness, m_removedIceThickness,
					  initialNorm,finalNorm,convergenceMetric,
					  linear , m_velRHS, m_velBasalC, vectC0, m_A, m_cellMuCoef,
					  m_vect_coordSys, m_time, 0, m_finest_level);
		}
	      else if (m_initialGuessSolverType == Picard)
		{
		  ParmParse pp("picardSolver");
		  Real tol = 1.e-4; int nits = 1;
		  // since the constant-viscosity solve is a linear solve,
		  // Picard is the best option.
		  m_solverType = Picard;
		  m_velSolver = NULL;
		  defineSolver();

		  pp.query("linearsolver_tolerance", tol );
		  pp.query("max_picard_iterations", nits );
		  m_velSolver->setTolerance(nits);		  
		  m_velSolver->setMaxIterations(tol);

		  rc = m_velSolver->solve(m_velocity, 
					  m_calvedIceThickness, m_addedIceThickness, m_removedIceThickness,
					  initialNorm,finalNorm,convergenceMetric,
					  m_velRHS, m_velBasalC, vectC0, m_A, m_cellMuCoef,
					  m_vect_coordSys, m_time, 0, m_finest_level);
		}
	      else
		{
		  MayDay::Error("unknown initial guess solver type");
		}


	      if (rc != 0)
		{
		  MayDay::Warning("constant mu solve failed");
		}

              // now put everything back the way it was...
	      delete m_constitutiveRelation;
              delete m_velSolver;
              m_velSolver = velSolverSave;
              m_constitutiveRelation = constRelSave;
              m_solverType = solverTypeSave;

#if 0	      
	      //put the solver back how it was
	      m_velSolver->define(m_amrDomains[0],
				  m_constitutiveRelation,
				  m_basalFrictionRelation,
				  m_amrGrids,
				  m_refinement_ratios,
				  dxCrse,
				  m_thicknessIBCPtr,
				  numLevels);
#endif
      
	    }
	  else if (m_initialGuessType == Function)
	    {
	      ParmParse pp("amr");
	      std::string functionType = "constant";
	      pp.query("initial_velocity_function_type", functionType );
	      if (functionType == "flowline")
		{
		  Real dx; 
		  std::string file, set;
		  pp.get("initial_velocity_function_flowline_dx", dx);
		  pp.get("initial_velocity_function_flowline_file", file);
		  pp.get("initial_velocity_function_flowline_set", set);
		  ExtrudedPieceWiseLinearFlowline f(file,set,dx);
		  for (int lev = 0; lev <  m_finest_level + 1; lev++)
		    {
		      LevelData<FArrayBox>& levelVel = *m_velocity[lev];
		      for (DataIterator dit(levelVel.disjointBoxLayout());
			   dit.ok(); ++dit)
			{
			  FArrayBox& vel = levelVel[dit];
			  const Box& box = vel.box(); 
			  for (BoxIterator bit(box); bit.ok(); ++bit)
			    {
			      const IntVect& iv = bit();
			      RealVect x = RealVect(iv) * m_amrDx[lev] 
				+ 0.5 * m_amrDx[lev];
			      vel(iv,0) = f(x);
			    }
			}
		    }
		}
	      
	    }
	  else
	    {
	      MayDay::Error("AmrIce::SolveVelocityField unknown initial guess type");
	    }
	}
#ifdef CH_USE_HDF5
      if (m_write_presolve_plotfiles)
        {
          string save_prefix = m_plot_prefix;
          m_plot_prefix.append("preSolve.");
	  bool t_write_fluxVel = m_write_fluxVel;
	  m_write_fluxVel = false; // turning this off in preSolve files for now
          writePlotFile();
          m_write_fluxVel = t_write_fluxVel;
	  m_plot_prefix = save_prefix;
        }
#endif

      int solverRetVal; 
    
      //set u = 0 in ice free cells
      for (int lev=0; lev <= m_finest_level ; ++lev)
	{
	  const DisjointBoxLayout& levelGrids = m_amrGrids[lev];
	  LevelSigmaCS& levelCS = *m_vect_coordSys[lev];
	  for (DataIterator dit(levelGrids); dit.ok(); ++dit)
	    {
	      const BaseFab<int>& mask = levelCS.getFloatingMask()[dit];
	      FArrayBox& vel = (*m_velocity[lev])[dit];
	      for (BoxIterator bit(levelGrids[dit]); bit.ok(); ++bit)
		{
		  const IntVect& iv = bit();
		  if (mask(iv) == OPENSEAMASKVAL || 
		      mask(iv) == OPENLANDMASKVAL )
		    {
		      vel(iv,0) = 0.0; vel(iv,1) = 0.0;
		    } 
		}
	    }
	}

	if (a_forceSolve || ((m_cur_step+1)%m_velocity_solve_interval == 0))
	  {

	  // Need to record ice lost through eliminate fast/remote ice in the velocity solve
	    for (int lev=0; lev<= m_finest_level; lev++)
	      {
		const LevelData<FArrayBox>& thck = m_vect_coordSys[lev]->getH();
		resetRecordThickness(thck, lev);
	      }

	    solverRetVal = m_velSolver->solve(m_velocity, 
					      m_calvedIceThickness, 
					      m_addedIceThickness,
					      m_removedIceThickness,
					      m_velocitySolveInitialResidualNorm, 
					      m_velocitySolveFinalResidualNorm,
					      a_convergenceMetric,
					      m_velRHS, m_velBasalC, vectC0,
					      m_A, m_cellMuCoef,
					      m_vect_coordSys,
					      m_time,
					      0, m_finest_level);

	    if (solverRetVal != 0)
	      {
		pout() << " solver return value = "
		       << solverRetVal << std::endl;
		MayDay::Warning("solver return value != 0"); 
	      }
	    for (int lev = 0; lev <= m_finest_level; lev++)
	      {
		m_thicknessIBCPtr->velocityGhostBC
		  (*m_velocity[lev],*m_vect_coordSys[lev],
		   m_amrDomains[lev], m_time);
	      }
	    
	    //special case for inverse problems : read back C and muCoef
	    InverseIceVelocitySolver* invPtr = dynamic_cast<InverseIceVelocitySolver*>(m_velSolver);
	    if (invPtr)
	      {
		if (m_basalFrictionPtr)
		  delete m_basalFrictionPtr; 
		m_basalFrictionPtr = invPtr->basalFriction();
		
		if (m_muCoefficientPtr)
		  delete m_muCoefficientPtr;
		m_muCoefficientPtr = invPtr->muCoefficient();
	      } // end special case for inverse problems

	    //update calved ice thickness 
	    for (int lev=0; lev<= m_finest_level; lev++)
	      {
		LevelData<FArrayBox>& thck = m_vect_coordSys[lev]->getH();
		updateAccumulatedCalvedIce(thck, lev);
	      }
	    
	  } // end if (a_forceSolve || ((m_cur_step+1)%m_velocity_solve_interval == 0))

    } // end if (m_doInitialSolve) 

  //allow calving model to modify geometry 
  applyCalvingCriterion(CalvingModel::PostVelocitySolve);

  //calculate the face centred (flux) velocity and diffusion coefficients
  computeFaceVelocity(m_faceVelAdvection,m_faceVelTotal,m_diffusivity,m_layerXYFaceXYVel, m_layerSFaceXYVel);

  for (int lev=0; lev<=m_finest_level; lev++)
    {
      if (vectC0[lev] != NULL)
	{
	  delete vectC0[lev]; vectC0[lev] = NULL;
	}
    }

  /// This is probably the most useful notification, as a velocity 
  /// solve is carried out at the end of every major stage
  notifyObservers(Observer::PostVelocitySolve);


#if 0  
  // debugging test -- redefine velocity as a constant field
  for (int lev=0; lev<m_velocity.size(); lev++)
    {
      DataIterator dit = m_velocity[lev]->dataIterator();
      for (dit.begin(); dit.ok(); ++dit)
        {
          (*m_velocity[lev])[dit].setVal(1.0);
        }
    }
#endif

}

	  

    


void AmrIce::defineVelRHS(Vector<LevelData<FArrayBox>* >& a_vectRhs)
{

  Vector<RealVect> dx;
  Vector<LevelData<FArrayBox>*> rhs;
  for (int lev=0; lev<=m_finest_level; lev++)
    {
      dx.push_back(m_vect_coordSys[lev]->dx());
      rhs.push_back(a_vectRhs[lev]);
    }

  IceUtility::defineRHS(rhs, m_vect_coordSys,  m_amrGrids, dx);

  //\todo : move this into IceUtility::defineRHS
  for (int lev=0; lev<=m_finest_level; lev++)
    {
      // finally, modify RHS in problem-dependent ways,
      m_thicknessIBCPtr->modifyVelocityRHS(*a_vectRhs[lev],  *m_vect_coordSys[lev],
                                           m_amrDomains[lev],m_time, m_dt);
    }

}


/// set mu coefficient (phi) prior to velocity solve
void
AmrIce::setMuCoefficient(Vector<LevelData<FArrayBox>* >& a_cellMuCoef)
{
  CH_assert(m_muCoefficientPtr != NULL);
  for (int lev=0; lev<=m_finest_level; lev++)
    {
      m_muCoefficientPtr->setMuCoefficient(*a_cellMuCoef[lev],
					   *m_vect_coordSys[lev],
                                           m_time,
                                           m_dt);
      if (lev > 0)
	{
	  PiecewiseLinearFillPatch ghostFiller
	    (m_amrGrids[lev],m_amrGrids[lev-1],1,m_amrDomains[lev-1],
	     m_refinement_ratios[lev-1],1);
	  
	  ghostFiller.fillInterp(*a_cellMuCoef[lev],
				 *a_cellMuCoef[lev-1],
				 *a_cellMuCoef[lev-1],
				 1.0,0,0,1);
	}
      a_cellMuCoef[lev]->exchange();
    }
}


/// set basal friction coefficients C,C0 prior to velocity solve
void
AmrIce::setBasalFriction(Vector<LevelData<FArrayBox>* >& a_vectC,Vector<LevelData<FArrayBox>* >& a_vectC0)
{

  // first, compute C and C0 as though there was no floating ice
  CH_assert(m_basalFrictionPtr != NULL);
  for (int lev=0; lev<=m_finest_level; lev++)
    {
      m_basalFrictionPtr->setBasalFriction(*a_vectC[lev], *m_vect_coordSys[lev],
                                           this->time(),m_dt); 
      if (m_basalRateFactor != NULL)
	{
	  //basal temperature dependence
	  LevelData<FArrayBox>& C = *a_vectC[lev];
	  Vector<Real> bSigma(1,1.0);
	  LevelData<FArrayBox> A(C.disjointBoxLayout(),1,C.ghostVect());
	  IceUtility::computeA(A, bSigma,*m_vect_coordSys[lev],  
			       m_basalRateFactor, *m_bInternalEnergy[lev]);
	  for (DataIterator dit = C.dataIterator(); dit.ok(); ++dit)
	    {
	      C[dit] /= A[dit];
	    }
	}
     
      a_vectC[lev]->exchange();
    }

  // compute C0 (wall drag) before setting C = 0 in floating regions
  for (int lev=0; lev<=m_finest_level; lev++)
    {
      LevelSigmaCS& levelCS = *m_vect_coordSys[lev];
      const DisjointBoxLayout& grids = m_amrGrids[lev];
      for (DataIterator dit(grids); dit.ok(); ++dit)
        {
	  FArrayBox& thisC0 = (*a_vectC0[lev])[dit];
	  const FArrayBox& thisC = (*a_vectC[lev])[dit];
	  thisC0.setVal(0.0);
	  if (m_wallDrag)
	    {
	      IceUtility::addWallDrag(thisC0, levelCS.getFloatingMask()[dit], 
				      levelCS.getSurfaceHeight()[dit], levelCS.getH()[dit], 
				      levelCS.getTopography()[dit], thisC, m_wallDragExtra,
				      RealVect::Unit*m_amrDx[lev], grids[dit]);
	    }
	}
    }

  if ( m_reset_floating_friction_to_zero )
    {
      //set C = 0 in floating region, possibly employing a thickness-above-flotation interpolation 
      for (int lev=0; lev<=m_finest_level; lev++)
	{
	  IceUtility::setFloatingBasalFriction(*a_vectC[lev], *m_vect_coordSys[lev] ,
					       m_amrGrids[lev], m_groundingLineSubdivision);
	}
    }

}




/// given the current cell centred velocity field, compute a face centred velocity field
void 
AmrIce::computeFaceVelocity(Vector<LevelData<FluxBox>* >& a_faceVelAdvection, 
			    Vector<LevelData<FluxBox>* >& a_faceVelTotal,
			    Vector<LevelData<FluxBox>* >& a_diffusivity,
			    Vector<LevelData<FluxBox>* >& a_layerXYFaceXYVel,
			    Vector<LevelData<FArrayBox>* >& a_layerSFaceXYVel) 
{
  CH_assert(m_constitutiveRelation != NULL);

  LevelData<FArrayBox>* cellDiffusivity = NULL;

  for (int lev = 0; lev <= m_finest_level; lev++)
    {
      LevelData<FArrayBox>* crseVelPtr = (lev > 0)?m_velocity[lev-1]:NULL;
      int nRefCrse = (lev > 0)?m_refinement_ratios[lev-1]:1;

      

      LevelData<FArrayBox>* crseCellDiffusivityPtr = 
	(lev > 0)?cellDiffusivity:NULL;

      cellDiffusivity = new LevelData<FArrayBox>(m_amrGrids[lev],1,IntVect::Unit);
      
      CH_assert(cellDiffusivity != NULL);
      CH_assert(a_faceVelAdvection[lev] != NULL);
      CH_assert(a_faceVelTotal[lev] != NULL);
      CH_assert(a_diffusivity[lev] != NULL);
      CH_assert(a_layerXYFaceXYVel[lev] != NULL);
      CH_assert(a_layerSFaceXYVel[lev] != NULL);
      CH_assert(m_velocity[lev] != NULL);
      CH_assert(m_vect_coordSys[lev] != NULL);
      CH_assert(m_A[lev] != NULL);
      CH_assert(m_sA[lev] != NULL);
      CH_assert(m_bA[lev] != NULL);
      
      IceUtility::computeFaceVelocity
       	(*a_faceVelAdvection[lev], *a_faceVelTotal[lev], *a_diffusivity[lev],
	 *cellDiffusivity,*a_layerXYFaceXYVel[lev], *a_layerSFaceXYVel[lev],
	 *m_velocity[lev],*m_vect_coordSys[lev], m_thicknessIBCPtr, 
	 *m_A[lev], *m_sA[lev], *m_bA[lev], 
	 crseVelPtr,crseCellDiffusivityPtr, nRefCrse, 
	 m_constitutiveRelation, m_additionalVelocity);

      if (crseCellDiffusivityPtr != NULL)
	delete crseCellDiffusivityPtr;

    }

  if (cellDiffusivity != NULL)
    delete cellDiffusivity;
}


/// compute div(vel*H) at a given time
void
AmrIce::computeDivThicknessFlux(Vector<LevelData<FArrayBox>* >& a_divFlux,
                                Vector<LevelData<FluxBox>* >& a_flux,
                                Vector<LevelData<FArrayBox>* >& a_thickness,
                                Real a_time, Real a_dt)
{

  //Vector<LevelData<FluxBox>* > faceVel(m_finest_level+1, NULL);
  
  for (int lev=0; lev<= m_finest_level; lev++)
    {
      const DisjointBoxLayout& levelGrids = m_amrGrids[lev];
      // construct face-centered velocity field
      LevelData<FluxBox> faceVel(levelGrids, 1, IntVect::Unit);
                                 
      if (lev > 0)
        {
          int nVelGhost = m_velocity[lev]->ghostVect()[0];
          
          PiecewiseLinearFillPatch velFiller(levelGrids, 
                                             m_amrGrids[lev-1],
                                             m_velocity[0]->nComp(), 
                                             m_amrDomains[lev-1],
                                             m_refinement_ratios[lev-1],
                                             nVelGhost);
          
          // since we're not subcycling, don't need to interpolate in time
          Real time_interp_coeff = 0.0;
          velFiller.fillInterp(*m_velocity[lev],
                               *m_velocity[lev-1],
                               *m_velocity[lev-1],
                               time_interp_coeff,
                               0, 0, m_velocity[0]->nComp());
          
        } // end if lev > 0
      m_velocity[lev]->exchange();

      // average velocities to faces
      CellToEdge(*m_velocity[lev],faceVel);
      faceVel.exchange();
      
      // flux = faceVel*faceH      
      LevelData<FluxBox>& levelFlux = *a_flux[lev];
      LevelSigmaCS& levelCoords = *m_vect_coordSys[lev];
      LevelData<FluxBox>& faceH = levelCoords.getFaceH();

      DataIterator dit = levelGrids.dataIterator();
      for (dit.begin(); dit.ok(); ++dit)
        {
          FluxBox& thisflux = levelFlux[dit];
          FluxBox& thisVel = faceVel[dit];
          FluxBox& thisH = faceH[dit];

          for (int dir=0; dir<SpaceDim; dir++)
            {
              thisflux[dir].copy(thisVel[dir]);
              thisflux[dir].mult(thisH[dir]);
            }
        }

      // average fluxes to coarser levels, if needed
      if (lev>0)
        {
          CoarseAverageFace faceAverager(m_amrGrids[lev],
                                         1, m_refinement_ratios[lev-1]);
          faceAverager.averageToCoarse(*a_flux[lev-1], *a_flux[lev]);
          
        }
      
    } // end loop over levels

  // now compute div(flux)
  
  // compute div(F) and add source term
  for (int lev=0; lev<= m_finest_level; lev++)
    {
      DisjointBoxLayout& levelGrids = m_amrGrids[lev];
      LevelData<FluxBox>& levelFlux = *a_flux[lev];
      LevelData<FArrayBox>& levelDiv = *a_divFlux[lev];
      LevelSigmaCS& levelCoords = *(m_vect_coordSys[lev]);

      LevelData<FArrayBox>& surfaceThicknessSource = *m_surfaceThicknessSource[lev];
      m_surfaceFluxPtr->surfaceThicknessFlux(surfaceThicknessSource, *this, lev, a_dt);
      
      LevelData<FArrayBox>& basalThicknessSource = *m_basalThicknessSource[lev];
      m_basalFluxPtr->surfaceThicknessFlux(basalThicknessSource, *this, lev, a_dt);

      const RealVect& dx = levelCoords.dx();          

      DataIterator dit = levelGrids.dataIterator();
      
      for (dit.begin(); dit.ok(); ++dit)
        {
          const Box& gridBox = levelGrids[dit];
          FArrayBox& thisDiv = levelDiv[dit];
          
          FluxBox& thisFlux = levelFlux[dit];
          thisDiv.setVal(0.0);
          
          // loop over directions and increment with div(F)
          for (int dir=0; dir<SpaceDim; dir++)
            {
              // use the divergence from 
              // Chombo/example/fourthOrderMappedGrids/util/DivergenceF.ChF
              FORT_DIVERGENCE(CHF_CONST_FRA(thisFlux[dir]),
                              CHF_FRA(thisDiv),
                              CHF_BOX(gridBox),
                              CHF_CONST_REAL(dx[dir]),
                              CHF_INT(dir));
            }

          // add in thickness source here
          thisDiv.minus(surfaceThicknessSource[dit], gridBox,0,0,1);
	  thisDiv.minus(basalThicknessSource[dit], gridBox,0,0,1);
          //thisDiv *= -1*a_dt;
        } // end loop over grids
    } // end loop over levels
  
}

// increment phi := phi + dt*dphi
void
AmrIce::incrementWithDivFlux(Vector<LevelData<FArrayBox>* >& a_phi,
                             const Vector<LevelData<FArrayBox>* >& a_dphi,
                             Real a_dt)
{
  for (int lev=0; lev<a_phi.size(); lev++)
    {
      LevelData<FArrayBox>& levelPhi = *a_phi[lev];
      const LevelData<FArrayBox>& level_dPhi = *a_dphi[lev];

      DataIterator dit = levelPhi.dataIterator();
      for (dit.begin(); dit.ok(); ++dit)
        {
          levelPhi[dit].plus(level_dPhi[dit], a_dt);
        }
    }
}
 

// increment coordSys with new thickness
void
AmrIce::updateCoordSysWithNewThickness(const Vector<LevelData<FArrayBox>* >& a_thickness)
{
  CH_assert(a_thickness.size() >= m_finest_level);
  
  for (int lev=0; lev<= m_finest_level; lev++)
    {
      const LevelData<FArrayBox>& levelH = *a_thickness[lev];
      LevelSigmaCS& levelCS = *m_vect_coordSys[lev];
      LevelData<FArrayBox>& levelCS_H = levelCS.getH();
      DataIterator dit = levelH.dataIterator();
      for (dit.begin(); dit.ok(); ++dit)
        {
          FArrayBox& thisH = levelCS_H[dit];
          thisH.copy(levelH[dit]);
        }
      {
	LevelSigmaCS* crseCoords = (lev > 0)?&(*m_vect_coordSys[lev-1]):NULL;
	int refRatio = (lev > 0)?m_refinement_ratios[lev-1]:-1;
	levelCS.recomputeGeometry(crseCoords, refRatio);
      }
    } // end loop over levels      
}

void
AmrIce::setIceFrac(const LevelData<FArrayBox>& a_thickness, int a_level)
{
  // initialize fraction to 1 if H>0, 0 o/w...
  DataIterator dit = m_iceFrac[a_level]->dataIterator();
  for (dit.begin(); dit.ok(); ++dit)
    {
      FArrayBox& thisFrac = (*m_iceFrac[a_level])[dit];
      thisFrac.setVal(0.0);
      const FArrayBox& thisH = a_thickness[dit];
      BoxIterator bit(thisFrac.box());
      for (bit.begin(); bit.ok(); ++bit)
        {
          IntVect iv = bit();
          if (thisH(iv,0) > 0) thisFrac(iv,0) = 1.0;
        }
    }
}

void
AmrIce::updateIceFrac(LevelData<FArrayBox>& a_thickness, int a_level)
{
  // set ice fraction to 0 if no ice in cell...

  // "zero" thickness value
  Real ice_eps = 1.0e-6;
  DataIterator dit = m_iceFrac[a_level]->dataIterator();
  for (dit.begin(); dit.ok(); ++dit)
    {
      FArrayBox& thisFrac = (*m_iceFrac[a_level])[dit];
      FArrayBox& thisH = a_thickness[dit];
      BoxIterator bit(thisFrac.box());
      for (bit.begin(); bit.ok(); ++bit)
        {
          IntVect iv = bit();
          if (thisH(iv,0) < ice_eps) 
            {
              thisFrac(iv,0) = 0.0;
              thisH(iv,0) = 0.0;
            }          
        }
    }
}



/// update real-valued ice fraction through advection from neighboring cells
void
AmrIce::advectIceFrac(Vector<LevelData<FArrayBox>* >& a_iceFrac,
                      const Vector<LevelData<FluxBox>* >& a_faceVelAdvection,
                      Real a_dt)
{
  // for now, set fill threshold to be (1-cfl) on the theory 
  // that we want to declare a cell full before it actually over-fills
  Real fillThreshold = (1.0 - m_cfl);
  
  for (int lev=0; lev<= m_finest_level; lev++)
    {
      LevelData<FArrayBox>& levelFrac = *a_iceFrac[lev];
      const LevelData<FluxBox>& levelFaceVel = *a_faceVelAdvection[lev];
      const DisjointBoxLayout& fracGrids = levelFrac.getBoxes();
      Real levelDx = m_amrDx[lev];

      DataIterator dit = levelFrac.dataIterator();
      for (dit.begin(); dit.ok(); ++dit)
        {
          // only update valid cells
          const Box& gridBox = fracGrids[dit];
          FArrayBox& thisFrac = levelFrac[dit];
          const FluxBox& thisFaceVel = levelFaceVel[dit];          
          for (int dir=0; dir<SpaceDim; dir++)
            {
              FORT_ADVECTFRAC(CHF_FRA1(thisFrac,0),
                              CHF_CONST_FRA1(thisFaceVel[dir],0),
                              CHF_REAL(levelDx),
                              CHF_REAL(a_dt),
                              CHF_REAL(fillThreshold),
                              CHF_BOX(gridBox),
                              CHF_INT(dir));
            } // end loop over directions
        } // end loop over boxes
    } // end loop over levels

}

void
AmrIce::resetRecordThickness(const LevelData<FArrayBox>& a_thickness, int a_level)
{
  if (a_level == 0)
    {
      CH_assert(!getIsThckRecorded());
    }

  DataIterator dit = m_recordThickness[a_level]->dataIterator();
  for (dit.begin(); dit.ok(); ++dit)
    {
      (*m_recordThickness[a_level])[dit].copy(a_thickness[dit]);
    }

  if (a_level == m_finest_level)
    {
      setIsThckRecorded(true);
    }
}

void 
AmrIce:: setIsThckRecorded(const bool a_thicknessIsRecorded)
{
  m_thicknessIsRecorded = a_thicknessIsRecorded;
}

void
AmrIce::updateAccumulatedCalvedIce(LevelData<FArrayBox>& a_thickness, int a_level)
{

  if (a_level == 0)
    {
      CH_assert(getIsThckRecorded());
    }
  
  LevelData<FArrayBox>& accumCalv = *m_melangeThickness[a_level];
  LevelData<FArrayBox>& prevThck = *m_recordThickness[a_level];
  for (DataIterator dit(m_amrGrids[a_level]); dit.ok(); ++dit)
    {
      accumCalv[dit] += prevThck[dit];
      accumCalv[dit] -= a_thickness[dit];
    }

  if (a_level == m_finest_level)
    {
      setIsThckRecorded(false);
    }

}

// compute timestep
Real 
AmrIce::computeDt()
{
  if (s_verbosity > 3) 
    { 
      pout() << "AmrIce::computeDt" << endl;
    }

  if (m_fixed_dt > TINY_NORM)
    return m_fixed_dt;

  Real dt = 1.0e50;
  for (int lev=0; lev<= finestTimestepLevel(); lev++)
    {

      Real dtLev = dt;
      const DisjointBoxLayout& levelGrids = m_amrGrids[lev];
      const LevelData<FluxBox>& levelVel = *m_faceVelAdvection[lev]; 
      DataIterator levelDit = levelVel.dataIterator();
      for (levelDit.reset(); levelDit.ok(); ++levelDit)
	{
	  for (int dir = 0; dir < SpaceDim; dir++)
	    {
	      int p = 0;
	      Box faceBox = levelGrids[levelDit];
	      faceBox.surroundingNodes(dir);
	      Real maxVel = 1.0 + levelVel[levelDit][dir].norm(faceBox,p, 0, 1);
	      CH_assert(maxVel < HUGE_VEL);
	      Real localDt = m_amrDx[lev]/maxVel;
	      dtLev = min(dtLev, localDt);
	    }
	}
      
      if (m_diffusionTreatment == EXPLICIT){
	MayDay::Error("diffusion_treatment == explicit not supported now : use none");
      }
      dt = min(dt, dtLev);
    }

#ifdef CH_MPI
  Real tmp = 1.;
  int result = MPI_Allreduce(&dt, &tmp, 1, MPI_CH_REAL,
			     MPI_MIN, Chombo_MPI::comm);
  if (result != MPI_SUCCESS)
    {
      MayDay::Error("communication error on norm");
    }
  dt = tmp;
#endif

  if (m_cur_step == 0)
    {
      dt *= m_initial_cfl;
    } 
  else 
    {
      dt *= m_cfl;
    }

  // also check to see if max grow rate applies
  // (m_dt > 0 test screens out initial time, when we set m_dt to a negative 
  // number by default)
  // Use the value stored in m_stable_dt in case dt was altered to hit a plot interval
  // m_max_dt_grow < 0 implies that we don't enforce this.
  if ((m_max_dt_grow > 0) && (dt > m_max_dt_grow*m_stable_dt) && (m_stable_dt > 0) )
    dt = m_max_dt_grow*m_stable_dt;
  
  if (m_timeStepTicks){
    // reduce time step to integer power of two
    dt = std::pow(2.0, std::floor(std::log(dt)/std::log(two)));
    
  }
  
  m_stable_dt = dt;
  if (s_verbosity > 3) 
    { 
      pout() << "AmrIce::computeDt dt = " << dt << endl;
    }
  CH_assert(dt > TIME_EPS);
  return dt;// min(dt,2.0);

}

Real 
AmrIce::computeInitialDt()
{

  if (s_verbosity > 3) 
    { 
      pout() << "AmrIce::computeInitialDt" << endl;
    }


  // for now, just call computeDt;
  Real dt = computeDt();
  return dt;
}



//determine the grouding line proximity 
/**

   Solves the elliptic problem 
   a * phi - b* grad^2 phi = 0;
   with natural boundary conditions.

   for grounded ice, a = 10^5 and b = 1
   for floating ice, s = 0 and b = 1
*/
void AmrIce::updateGroundingLineProximity() const
{

  CH_TIME("AmrIce::updateGroundingLineProximity");

  if (m_groundingLineProximity_valid)
    return;

  if (m_groundingLineProximity.size() < m_finest_level + 1)
    {
      m_groundingLineProximity.resize(m_finest_level + 1, NULL);
    }

  if (s_verbosity > 0)
    {
      pout() << "AmrIce::updateGroundingLineProximity() max level = " << m_finest_level << " " << endl; 
    }

  //Natural boundary conditions
  BCHolder bc(ConstDiriNeumBC(IntVect::Zero, RealVect::Zero,
  			      IntVect::Zero, RealVect::Zero));

  //BCHolder bc(ConstDiriNeumBC(IntVect(0,0), RealVect(-1.0,-1.0),
  //			      IntVect(0,0), RealVect(1.0,1.0)));

  Vector<RefCountedPtr<LevelData<FArrayBox> > > a(m_finest_level + 1);
  Vector<RefCountedPtr<LevelData<FluxBox> > > b(m_finest_level + 1);
  Vector<LevelData<FArrayBox>* > rhs(m_finest_level+ 1,NULL);
  Vector<DisjointBoxLayout> grids(finestTimestepLevel() + 1);
  Vector<ProblemDomain> domains(finestTimestepLevel() + 1);
  Vector<RealVect> dx(finestTimestepLevel() + 1);

  for (int lev=0; lev <= m_finest_level; ++lev)
    {
      dx[lev] = m_amrDx[lev]*RealVect::Unit;
      domains[lev] = m_amrDomains[lev];

      const LevelSigmaCS& levelCS = *m_vect_coordSys[lev];
      const LevelData<BaseFab<int> >& levelMask = levelCS.getFloatingMask();
      const DisjointBoxLayout& levelGrids = m_amrGrids[lev];
      a[lev] = RefCountedPtr<LevelData<FArrayBox> >
 	(new LevelData<FArrayBox>(levelGrids, 1, IntVect::Zero));
      b[lev] = RefCountedPtr<LevelData<FluxBox> >
 	(new LevelData<FluxBox>(levelGrids, 1, IntVect::Zero));
      rhs[lev] = new LevelData<FArrayBox>(levelGrids, 1, IntVect::Zero);
      
      grids[lev] = levelGrids;

     

      if (m_groundingLineProximity[lev] != NULL)
	{
	  delete m_groundingLineProximity[lev];
	  m_groundingLineProximity[lev] = NULL;
	}
      m_groundingLineProximity[lev] =  new LevelData<FArrayBox>(levelGrids, 1, IntVect::Unit);
      
      LevelData<FArrayBox>& levelPhi = *m_groundingLineProximity[lev];

      const Real& crseDx = m_amrDx[0];
      Real crseDxSq = crseDx*crseDx;

      for (DataIterator dit(levelGrids); dit.ok(); ++dit)
 	{
	  FluxBox& B = (*b[lev])[dit];
	  
	  for (int dir = 0; dir < SpaceDim; dir++)
	    {
	      B[dir].setVal(crseDxSq);
	    }

 	  FArrayBox& r =  (*rhs[lev])[dit];
 	  r.setVal(0.0);
	  FArrayBox& A =  (*a[lev])[dit];
	  A.setVal(0.0);
	  FArrayBox& phi = levelPhi[dit];
	  phi.setVal(0.0);

 	  const BaseFab<int>& mask = levelMask[dit];
 	  const Box& gridBox = levelGrids[dit];
	  //	  const FArrayBox& u = (*m_velocity[lev])[dit];

	  Real AcoefF = crseDx / m_groundingLineProximityScale;
	  Real AcoefG = 1.0 ;
	  if (m_groundingLineProximityCalcType > 0)
	    {
	      AcoefF = crseDx / m_groundingLineProximityScale;
	      AcoefF *= AcoefF;
	      
	    }

 	  for (BoxIterator bit(gridBox);bit.ok();++bit)
 	    {
 	      const IntVect& iv = bit();
 	      if (mask(iv) == GROUNDEDMASKVAL )
 		{
 		  A(iv) = AcoefG;
		  r(iv) = AcoefG;
		  
 		} 
	      else
		{
		  
		  A(iv) = AcoefF;
		  r(iv) = 0.0;
		}
	      
 	    }
	  phi.copy(r);
 	}

      rhs[lev]->exchange();
      levelPhi.exchange();
      m_groundingLineProximity[lev]->exchange();
      a[lev]->exchange();
      b[lev]->exchange();
    }


  VCAMRPoissonOp2Factory* poissonOpFactory = new VCAMRPoissonOp2Factory;
  poissonOpFactory->define(domains[0], grids , m_refinement_ratios,
 			   m_amrDx[0], bc, 1.0, a,  1.0 , b);
  RefCountedPtr< AMRLevelOpFactory<LevelData<FArrayBox> > > 
    opFactoryPtr(poissonOpFactory);

  MultilevelLinearOp<FArrayBox> poissonOp;
  poissonOp.define(grids, m_refinement_ratios, domains, dx, opFactoryPtr, 0);
    
  RelaxSolver<Vector<LevelData<FArrayBox>* > >* relaxSolver
    = new RelaxSolver<Vector<LevelData<FArrayBox>* > >();

  relaxSolver->define(&poissonOp,false);
  relaxSolver->m_verbosity = s_verbosity;
  relaxSolver->m_normType = 0;
  relaxSolver->m_eps = 1.0e-8;
  relaxSolver->m_imax = 12;
  relaxSolver->m_hang = 0.05;
  relaxSolver->solve(m_groundingLineProximity,rhs);

  delete(relaxSolver);

#ifdef DUMP_PROXIMITY
  std::string file("proximity.2d.hdf5");
  Real dt = 0.0; 
  Real time = 0.0;
  Vector<std::string> names(1,"proximity");
  WriteAMRHierarchyHDF5(file ,grids, m_groundingLineProximity ,names, m_amrDomains[0].domainBox(),
  			m_amrDx[0], dt, m_time, m_refinement_ratios, m_groundingLineProximity.size());
#endif
  
  for (int lev=0; lev <= m_finest_level ; ++lev)
    {
      if (rhs[lev] != NULL)
 	{
 	  delete rhs[lev];
	  rhs[lev] = NULL;
 	}
    }

  m_groundingLineProximity_valid = true;
}

//access the viscous tensor (cell-centered)
const LevelData<FArrayBox>* AmrIce::viscousTensor(int a_level) const
{
  updateViscousTensor();
  if (!(m_viscousTensorCell.size() > a_level))
    {
      std::string msg("AmrIce::viscousTensor !(m_viscousTensorCell.size() > a_level))");
      pout() << msg << endl;
      CH_assert((m_viscousTensorCell.size() > a_level));
      MayDay::Error(msg.c_str());
    }

  LevelData<FArrayBox>* ptr = m_viscousTensorCell[a_level];
  if (ptr == NULL)
    {
      std::string msg("AmrIce::viscousTensor m_viscousTensorCell[a_level] == NULL ");
      pout() << msg << endl;
      CH_assert(ptr != NULL);
      MayDay::Error(msg.c_str());
    }

  return ptr;

}

//access the viscous tensor (cell-centered)
const LevelData<FArrayBox>* AmrIce::viscosityCoefficient(int a_level) const
{
  updateViscousTensor();
  if (!(m_viscosityCoefCell.size() > a_level))
    {
      std::string msg("AmrIce::viscosityCoef !(m_viscosityCoefCell.size() > a_level))");
      pout() << msg << endl;
      CH_assert((m_viscosityCoefCell.size() > a_level));
      MayDay::Error(msg.c_str());
    }

  LevelData<FArrayBox>* ptr = m_viscosityCoefCell[a_level];
  if (ptr == NULL)
    {
      std::string msg("AmrIce::viscosityCoef m_viscosityCoefCell[a_level] == NULL ");
      pout() << msg << endl;
      CH_assert(ptr != NULL);
      MayDay::Error(msg.c_str());
    }

  return ptr;

}

const LevelData<FArrayBox>* AmrIce::surfaceThicknessSource(int a_level) const
{
  if (!(m_surfaceThicknessSource.size() > a_level))
    {
      std::string msg("AmrIce::surfaceThicknessSource !(m_surfaceThicknessSource.size() > a_level))");
      pout() << msg << endl;
      CH_assert((m_surfaceThicknessSource.size() > a_level));
      MayDay::Error(msg.c_str());
    }

  LevelData<FArrayBox>* ptr = m_surfaceThicknessSource[a_level];
  if (ptr == NULL)
    {
      std::string msg("AmrIce::surfaceThicknessSource m_surfaceThicknessSource[a_level] == NULL ");
      pout() << msg << endl;
      CH_assert(ptr != NULL);
      MayDay::Error(msg.c_str());
    }

  return ptr;
}

const LevelData<FArrayBox>* AmrIce::basalThicknessSource(int a_level) const
{
  if (!(m_basalThicknessSource.size() > a_level))
    {
      std::string msg("AmrIce::basalThicknessSource !(m_basalThicknessSource.size() > a_level))");
      pout() << msg << endl;
      CH_assert((m_basalThicknessSource.size() > a_level));
      MayDay::Error(msg.c_str());
    }

  LevelData<FArrayBox>* ptr = m_basalThicknessSource[a_level];
  if (ptr == NULL)
    {
      std::string msg("AmrIce::basalThicknessSource m_basalThicknessSource[a_level] == NULL ");
      pout() << msg << endl;
      CH_assert(ptr != NULL);
      MayDay::Error(msg.c_str());
    }

  return ptr;
}


//access the drag coefficient (cell-centered)
const LevelData<FArrayBox>* AmrIce::dragCoefficient(int a_level) const
{
  updateViscousTensor();
  if (!(m_dragCoef.size() > a_level))
    {
      std::string msg("AmrIce::dragCoef !(m_dragCoef.size() > a_level))");
      pout() << msg << endl;
      CH_assert((m_dragCoef.size() > a_level));
      MayDay::Error(msg.c_str());
    }

  LevelData<FArrayBox>* ptr = m_dragCoef[a_level];
  if (ptr == NULL)
    {
      std::string msg("AmrIce::dragCoef m_dragCoefCell[a_level] == NULL ");
      pout() << msg << endl;
      CH_assert(ptr != NULL);
      MayDay::Error(msg.c_str());
    }

  return ptr;
}



//update the viscous tensor components
void AmrIce::updateViscousTensor() const
{
  CH_TIME("AmrIce::updateViscousTensor");

  if (m_viscousTensor_valid)
    return;
  
  if (m_viscousTensorCell.size() < m_finest_level + 1)
    {
      m_viscousTensorCell.resize(m_finest_level + 1, NULL);
    }
  if (m_viscosityCoefCell.size() < m_finest_level + 1)
    {
      m_viscosityCoefCell.resize(m_finest_level + 1, NULL);
    }
  if (m_dragCoef.size() < m_finest_level + 1)
    {
      m_dragCoef.resize(m_finest_level + 1, NULL);
    }

  if (m_viscousTensorFace.size() < m_finest_level + 1)
    {
      m_viscousTensorFace.resize(m_finest_level + 1, NULL);
    }

 
  Vector<LevelData<FluxBox>*> faceA(m_finest_level + 1,NULL);
  Vector<RefCountedPtr<LevelData<FluxBox> > > viscosityCoef;
  Vector<RefCountedPtr<LevelData<FArrayBox> > > dragCoef;
  Vector<LevelData<FArrayBox>* > C0(m_finest_level + 1,  NULL);

  Vector<RealVect> vdx(m_finest_level + 1);
  for (int lev =0; lev <= m_finest_level; lev++)
    {
      faceA[lev] = new LevelData<FluxBox>(m_amrGrids[lev],m_A[lev]->nComp(),IntVect::Unit);
      CellToEdge(*m_A[lev],*faceA[lev]);

      if (m_viscousTensorFace[lev] != NULL)
	{
	  delete m_viscousTensorFace[lev];m_viscousTensorFace[lev]=NULL;
	}
      m_viscousTensorFace[lev] = new LevelData<FluxBox>(m_amrGrids[lev],SpaceDim,IntVect::Unit);

      if (m_viscousTensorCell[lev] != NULL)
	{
	  delete m_viscousTensorCell[lev];m_viscousTensorCell[lev]=NULL;
	}
      m_viscousTensorCell[lev] = new LevelData<FArrayBox>(m_amrGrids[lev],SpaceDim*SpaceDim,IntVect::Unit);
      
      if (m_dragCoef[lev] != NULL)
	{
	  delete m_dragCoef[lev]; m_dragCoef[lev] = NULL;
	}
      m_dragCoef[lev] = new LevelData<FArrayBox>(m_amrGrids[lev],SpaceDim,IntVect::Zero);
      
      if (m_viscosityCoefCell[lev] != NULL)
	{
	  delete m_viscosityCoefCell[lev]; m_viscosityCoefCell[lev] = NULL;
	}
      m_viscosityCoefCell[lev] = new LevelData<FArrayBox>(m_amrGrids[lev],SpaceDim,IntVect::Zero);

      if (C0[lev] != NULL)
	{
	  delete C0[lev];C0[lev] = NULL;
	}
      C0[lev] = new LevelData<FArrayBox>(m_amrGrids[lev],1,m_velBasalC[0]->ghostVect());
      DataIterator dit = m_amrGrids[lev].dataIterator();
      for (dit.begin(); dit.ok(); ++dit)
        {
          (*C0[lev])[dit].setVal(0.0);
        }
      vdx[lev] = RealVect::Unit*m_amrDx[lev];
    }

  //these parameters don't matter because we don't solve anything here. 
  Real vtopSafety = 1.0;
  int vtopRelaxMinIter = 4;
  Real vtopRelaxTol = 1.0;
  Real muMin = 0.0; 
  Real muMax = 1.23456789e+300;

  int numLevels = m_finest_level + 1;
  IceNonlinearViscousTensor state(m_amrGrids, m_refinement_ratios, m_amrDomains, vdx, m_vect_coordSys, 
				  m_velocity, m_velBasalC, C0, numLevels-1, 
				  *m_constitutiveRelation,  *m_basalFrictionRelation, *m_thicknessIBCPtr,  
				  m_A, faceA, m_time, vtopSafety, vtopRelaxMinIter, vtopRelaxTol, 
				  muMin, muMax);
  state.setState(m_velocity);
  viscosityCoef = state.mu();
  dragCoef = state.alpha();
  state.computeViscousTensorFace(m_viscousTensorFace);
  
  for (int lev =0; lev < numLevels; lev++)
    {
      
      //If a cell is adjacent to a calving front, we  set the (vertically integrated)
      //viscous tensor components at the intervening face to zero. That works well enough for the velocity solves,
      //but causes pain here because the cell average (in EdgeToCell) will end up half the value at the other face.
      for (DataIterator dit(m_amrGrids[lev]); dit.ok(); ++dit)
      	{

      	  const FArrayBox& thck = m_vect_coordSys[lev]->getH()[dit];
      	  //const FArrayBox& dsdx = m_vect_coordSys[lev]->getGradSurface()[dit];
	  const FArrayBox& usrf = m_vect_coordSys[lev]->getSurfaceHeight()[dit];
      	  const BaseFab<int>& mask = m_vect_coordSys[lev]->getFloatingMask()[dit];
      	  const Real& rhoi = m_vect_coordSys[lev]->iceDensity();
      	  //const Real& rhoo = m_vect_coordSys[lev]->waterDensity();
      	  const Real& gravity = m_vect_coordSys[lev]->gravity();
      	  //const Real rgr = rhoi * gravity * (1.0-rhoi/rhoo);
      	  //const RealVect& dx = m_vect_coordSys[lev]->dx();

      	  for (int dir = 0; dir < SpaceDim; dir++)
      	    {
      	      FArrayBox& facevt = (*m_viscousTensorFace[lev])[dit][dir];
      	      Real factor = rhoi * gravity;
      	      FORT_SETFRONTFACEVT(CHF_FRA1(facevt,dir),
      				  CHF_CONST_FRA1(thck,0),
      				  CHF_CONST_FRA1(usrf,0),
      				  CHF_CONST_FIA1(mask,0),
      				  CHF_CONST_INT(dir),
      				  CHF_CONST_REAL(factor),
      				  CHF_BOX(m_amrGrids[lev][dit]));
      	    }
      	}


      EdgeToCell(*m_viscousTensorFace[lev],*m_viscousTensorCell[lev]);
      if (lev > 0)
	{
	  PiecewiseLinearFillPatch ghostFiller
	    (m_amrGrids[lev],
	     m_amrGrids[lev-1],
	     m_viscousTensorCell[lev-1]->nComp(),
	     m_amrDomains[lev-1],
	     m_refinement_ratios[lev-1],
	     m_viscousTensorCell[lev-1]->ghostVect()[0]);
	  
	  ghostFiller.fillInterp(*m_viscousTensorCell[lev], 
				 *m_viscousTensorCell[lev-1], 
				 *m_viscousTensorCell[lev-1],1.0,0,0,
				 m_viscousTensorCell[lev-1]->nComp());

	}
      m_viscousTensorCell[lev]->exchange();

      EdgeToCell(*viscosityCoef[lev],*m_viscosityCoefCell[lev]);

      for (DataIterator dit(m_amrGrids[lev]); dit.ok(); ++dit)
      	{
	  const BaseFab<int>& mask = m_vect_coordSys[lev]->getFloatingMask()[dit];
	  FArrayBox& cellvt = (*m_viscousTensorCell[lev])[dit];
	  const Real z = 0.0;
	  for (int comp = 0; comp < SpaceDim * SpaceDim; comp++)
	    {
	      FORT_SETICEFREEVAL(CHF_FRA1(cellvt,comp), 
				 CHF_CONST_FIA1(mask,0),
				 CHF_CONST_REAL(z),
				 CHF_BOX(m_amrGrids[lev][dit]));
	    }
	}

      dragCoef[lev]->copyTo(Interval(0,0),*m_dragCoef[lev],Interval(0,0));

      if (faceA[lev] != NULL)
	{
	  delete faceA[lev]; faceA[lev] = NULL;
	}
      if (C0[lev] != NULL)
	{
	  delete C0[lev]; C0[lev] = NULL;
	}
    }

  m_viscousTensor_valid = true;

}

//access the grounding line proximity
const LevelData<FArrayBox>* AmrIce::groundingLineProximity(int a_level) const
{

  updateGroundingLineProximity();
  
  if (!(m_groundingLineProximity.size() > a_level))
    {
      std::string msg("AmrIce::groundingLineProximity !(m_groundingLineProximity.size() > a_level)");
      pout() << msg << endl;
      CH_assert((m_groundingLineProximity.size() > a_level));
      MayDay::Error(msg.c_str());
    }


  LevelData<FArrayBox>* ptr = m_groundingLineProximity[a_level];
  if (ptr == NULL)
    {
      std::string msg("AmrIce::groundingLineProximity m_groundingLineProximity[a_level] == NULL)");
      pout() << msg << endl;
      CH_assert(ptr != NULL);
      MayDay::Error(msg.c_str());
    }

  return ptr;
}


void AmrIce::applyCalvingCriterion(CalvingModel::Stage a_stage)
{

  //need to copy the thickness to keep track of the calved ice
  for (int lev=0; lev<= m_finest_level; lev++)
    {
      const LevelData<FArrayBox>& thck = m_vect_coordSys[lev]->getH();
      resetRecordThickness(thck, lev);
    }

  //allow calving model to modify geometry 
  for (int lev=0; lev<= m_finest_level; lev++)
    {
      LevelData<FArrayBox>& thck = m_vect_coordSys[lev]->getH();
      LevelData<FArrayBox>& frac = *m_iceFrac[lev];
      LevelData<FArrayBox>& calvedIce = *m_calvedIceThickness[lev];
      LevelData<FArrayBox>& addedIce = *m_addedIceThickness[lev];
      LevelData<FArrayBox>& removedIce = *m_removedIceThickness[lev];
      m_calvingModelPtr->applyCriterion(thck, calvedIce, addedIce, removedIce, frac, *this, lev, a_stage);	  
 
   }
    
  //update calved ice thickness and fraction
  for (int lev=0; lev<= m_finest_level; lev++)
    {
      LevelData<FArrayBox>& thck = m_vect_coordSys[lev]->getH();
      updateAccumulatedCalvedIce(thck, lev);
      updateIceFrac(thck, lev);
    }

  // usually a good time to eliminate remote ice
  if (m_eliminate_remote_ice) eliminateRemoteIce();
  
}


///Identify regions of floating ice that are remote
///from grounded ice and eliminate them.
void AmrIce::eliminateRemoteIce()
{
  //need to copy the thickness to keep track of the calved ice
  for (int lev=0; lev<= m_finest_level; lev++)
    {
      const LevelData<FArrayBox>& thck = m_vect_coordSys[lev]->getH();
      resetRecordThickness(thck, lev);
    }

  IceUtility::eliminateRemoteIce(m_vect_coordSys, m_velocity, 
				 m_calvedIceThickness, m_addedIceThickness,
				 m_removedIceThickness,
				 m_amrGrids, m_amrDomains, 
				 m_refinement_ratios, m_amrDx[0], 
				 m_finest_level, m_eliminate_remote_ice_max_iter,
				 m_eliminate_remote_ice_tol,s_verbosity);

  //update calved ice thickness and fraction
  for (int lev=0; lev<= m_finest_level; lev++)
    {
      LevelData<FArrayBox>& thck = m_vect_coordSys[lev]->getH();
      updateAccumulatedCalvedIce(thck, lev);
      updateIceFrac(thck, lev);
    }

}




void 
AmrIce::implicitThicknessCorrection(Real a_dt,
				    const Vector<LevelData<FArrayBox>* >& a_sts,
				    const Vector<LevelData<FArrayBox>* >& a_bts
				    )
{

  CH_TIME("AmrIce::implicitThicknessCorrection");
  if (s_verbosity > 3)
    {
      pout() << "AmrIce::implicitThicknessCorrection" << std::endl;
    }

  if  (m_temporalAccuracy == 1)
    {  
      //implicit Euler : solve (I - dt P) H = H_pred + dt * S
      
      //slc: at the moment, I'm setting eveything up every time-step,
      //pretending that diffusion is constant in time, and using the multi-grid
      //solver only. All these things are to be improved 

      //Natural boundary conditions - OK for now, but ought to get 
      //moved into subclasses of IceThicknessIBC
      BCHolder bc(ConstDiriNeumBC(IntVect::Zero, RealVect::Zero,
      				  IntVect::Zero, RealVect::Zero));

      Vector<RefCountedPtr<LevelData<FArrayBox> > > I(finestTimestepLevel() + 1);
      Vector<RefCountedPtr<LevelData<FluxBox> > > D(finestTimestepLevel() + 1);
      Vector<LevelData<FArrayBox>* > H(finestTimestepLevel() + 1);
      Vector<LevelData<FArrayBox>* > rhs(finestTimestepLevel()+ 1);
      Vector<DisjointBoxLayout> grids(finestTimestepLevel() + 1);

      for (int lev=0; lev <= finestTimestepLevel(); ++lev)
	{
	  LevelSigmaCS& levelCoords = *m_vect_coordSys[lev];
	  const DisjointBoxLayout& levelGrids = m_amrGrids[lev];

	  I[lev] = RefCountedPtr<LevelData<FArrayBox> >
	    (new LevelData<FArrayBox>(levelGrids, 1, IntVect::Unit));

	  H[lev] = new LevelData<FArrayBox>(levelGrids, 1, IntVect::Unit);

	  D[lev] = RefCountedPtr<LevelData<FluxBox> >(m_diffusivity[lev]);
	  D[lev].neverDelete();

	  rhs[lev] = new LevelData<FArrayBox>(levelGrids, 1, IntVect::Unit);

	  grids[lev] = levelGrids;

	  const LevelData<FArrayBox>& levelSTS = *a_sts[lev];
	  const LevelData<FArrayBox>& levelBTS = *a_bts[lev];
	  

	  for (DataIterator dit(levelGrids); dit.ok(); ++dit)
	    {
	      
	      (*I[lev])[dit].setVal(one);
	      (*H[lev])[dit].copy(levelCoords.getH()[dit] , 0 , 0, 1);
	      (*rhs[lev])[dit].copy( (*H[lev])[dit] , 0 , 0, 1);
	      (*rhs[lev])[dit].plus(levelSTS[dit],a_dt);
	      (*rhs[lev])[dit].plus(levelBTS[dit],a_dt); 
	      (*D[lev])[dit][0].plus(m_additionalDiffusivity);
	      (*D[lev])[dit][1].plus(m_additionalDiffusivity);
	      
	    }
	  rhs[lev]->exchange();
	  H[lev]->exchange();
	  m_diffusivity[lev]->exchange();
	  I[lev]->exchange();
	}

      VCAMRPoissonOp2Factory poissonOpFactory;//= new VCAMRPoissonOp2Factory;
      poissonOpFactory.define(m_amrDomains[0], grids , m_refinement_ratios,
			      m_amrDx[0], bc, 1.0, I,  a_dt, D);
    
      //Plain MG
      BiCGStabSolver<LevelData<FArrayBox> > bottomSolver;
      AMRMultiGrid<LevelData<FArrayBox> > mgSolver;
      mgSolver.define(m_amrDomains[0], poissonOpFactory , &bottomSolver, finestTimestepLevel()+1);
      //parse these
      mgSolver.m_eps = 1.0e-10;
      mgSolver.m_normThresh = 1.0e-10;
    
      int numMGSmooth = 4;
      mgSolver.m_pre = numMGSmooth;
      mgSolver.m_post = numMGSmooth;
      mgSolver.m_bottom = numMGSmooth;
      
      mgSolver.solve(H, rhs, finestTimestepLevel(), 0,  false);
   
      for (int lev=0; lev <= finestTimestepLevel()  ; ++lev)
	{
	  const DisjointBoxLayout& levelGrids = m_amrGrids[lev];
	  LevelSigmaCS& levelCoords = *m_vect_coordSys[lev];
          LevelData<FArrayBox>& levelCoord_H = levelCoords.getH();
	  
	  for (DataIterator dit(levelGrids); dit.ok(); ++dit)
	    {
	      CH_assert( (*H[lev])[dit].norm(0,0,1) < HUGE_THICKNESS);
	      levelCoord_H[dit].copy( (*H[lev])[dit], 0, 0, 1);

	      //put sensible values into the corners.
	      FArrayBox &thisH = levelCoord_H[dit];
	      Box sbox = thisH.box();
	      sbox.grow(-levelCoord_H.ghostVect()[0]);
	      FORT_EXTRAPCORNER2D(CHF_FRA(thisH),
      				  CHF_BOX(sbox));

	    }

	  if (rhs[lev] != NULL)
	    {
	      delete rhs[lev];
	      rhs[lev] = NULL;
	    }
	  if (H[lev] != NULL)
	    {
	      delete H[lev];
	      H[lev] = NULL;
	    }
	}
    }
  else 
    {    
      MayDay::Error("AmrIce::implicitThicknessCorrection, invalid temporal accuracy");
    }


  

}



#ifdef CH_USE_HDF5

void AmrIce::writeMetaDataHDF5(HDF5Handle& a_handle) const
{

 
      //Additional data (BISICLES specific)
      HDF5HeaderData headerData = m_headerData;
      headerData.m_int["max_level"] = m_max_level;
      headerData.m_int["finest_level"] = m_finest_level;
      headerData.m_int["current_step"] = m_cur_step; 
      headerData.m_real["time"] = time();
      headerData.m_real["dt"] = m_dt;
      headerData.m_string["svn_version"] = SVN_REV;
      headerData.m_string["svn_repository"] = SVN_REP;
      headerData.m_string["svn_url"] = SVN_URL;
      headerData.m_int["bisicles_version_major"] = BISICLES_VERSION_MAJOR;
      headerData.m_int["bisicles_version_minor"] = BISICLES_VERSION_MINOR;
      headerData.m_int["bisicles_patch_number"] = BISICLES_PATCH_NUMBER;
      headerData.m_int["chombo_version_major"] = CHOMBO_VERSION_MAJOR;
      headerData.m_int["chombo_version_minor"] = CHOMBO_VERSION_MINOR;
#ifdef CHOMBO_TRUNK
      headerData.m_int["chombo_patch_number"] = -1;
#else
      headerData.m_int["chombo_patch_number"] = CHOMBO_PATCH_NUMBER;
#endif
      //m_headerData.writeToFile(a_handle);
      headerData.writeToFile(a_handle);
    
  
}


void AmrIce::writeAMRHierarchyHDF5(const string& filename,
				   const Vector<DisjointBoxLayout>& a_grids,
				   const Vector<LevelData<FArrayBox>* > & a_data,
				   const Vector<string>& a_name,
				   const Box& a_domain,
				   const Real& a_dx,
				   const Real& a_dt,
				   const Real& a_time,
				   const Vector<int>& a_ratio,
				   const int& a_numLevels) const
{
 
  HDF5Handle handle(filename.c_str(), HDF5Handle::CREATE);
     
  //Chombo AMR data (VisIt compatible)
  WriteAMRHierarchyHDF5(handle, a_grids, a_data, a_name, 
			a_domain, a_dx, a_dt, a_time, a_ratio, 
			a_numLevels);

  
  writeMetaDataHDF5(handle);
 
  handle.close();
  
  

}


/// write hdf5 plotfile to the standard location
void 
AmrIce::writePlotFile() 
{
  if (s_verbosity > 3) 
    { 
      pout() << "AmrIce::writePlotFile" << endl;
    }
  
  // plot comps: thickness + horizontal velocity + z_bottom + z_surface + z_base 
  int numPlotComps = 4 + SpaceDim;

  if (m_reduced_plot)
    {
      // plot comps: thickness + horizontal velocity + zb + zs
      numPlotComps = 3 + SpaceDim;
    }

  // may need a zvel for visit to do "3d" streamlines correctly
  bool writeZvel = !m_reduced_plot;
  if (writeZvel) numPlotComps+=1;

  if (m_write_fluxVel)
    {
      numPlotComps += SpaceDim;
      if (writeZvel) numPlotComps+=1;
    }
  
 
  if (m_write_baseVel)
    {
      numPlotComps += SpaceDim;
      if (writeZvel) numPlotComps+=1;
    }
  // write both integer and real-valued masks
  if (m_write_mask) numPlotComps += 2;
  if (m_write_dHDt) numPlotComps += 1;
  if (m_write_solver_rhs) 
    {
      numPlotComps += SpaceDim ;
      // include basal_friction and C0 iff !m_reduced_plot 
      if (!m_reduced_plot)
        {
          numPlotComps += 2;
        }
    }
  
  

  if (m_write_internal_energy) 
    numPlotComps += m_internalEnergy[0]->nComp();
#if BISICLES_Z == BISICLES_LAYERED
  if (m_write_internal_energy)
    numPlotComps += 4;// surface and basal internalEnergys and heat fluxes

  //layer velocities
  if (m_write_layer_velocities)
    {
      numPlotComps += SpaceDim * (m_nLayers+1);
      if (writeZvel) numPlotComps += (m_nLayers+1);
    }

  if (m_write_viscousTensor)
    {
      // effective drag and viscosity coefficients
      numPlotComps += 2;
      if (!m_reduced_plot)
	{
	  numPlotComps += SpaceDim * SpaceDim; // viscous tensor components                        
	}
    }
  
  if (m_write_thickness_sources)
    {
      numPlotComps += 2;  // surface and basal sources
      if (!m_reduced_plot)
	{
	  numPlotComps += 5; // divThicknessFlux, calving flux and accumulated calving
	}
    }


#endif
  // generate data names

  string thicknessName("thickness");
  string xVelName("xVel");
  string yVelName("yVel");
  string zVelName("zVel");
  string zsName("Z_surface");
  string zbName("Z_base");
  string zbottomName("Z_bottom");
  string dthicknessName("dThickness/dt");
  string betaName("basal_friction");
  string solverRhsxName("xRhs");
  string solverRhsyName("yRhs");
  string C0Name("C0");
  string maskName("mask");
  string fracName("iceFrac");
  string xfVelName("xfVel");
  string yfVelName("yfVel");
  string zfVelName("zfVel");
  string xbVelName("xbVel");
  string ybVelName("ybVel");
  string zbVelName("zbVel");

  string internalEnergyName("internalEnergy");
  string heatFluxName("heatflux");
#if BISICLES_Z == BISICLES_LAYERED
  string xlayerVelName("xlayerVel");
  string ylayerVelName("ylayerVel");
  string zlayerVelName("zlayerVel");
#endif

  string xxVTname("xxViscousTensor");
  string xyVTname("xyViscousTensor");
  string xzVTname("xzViscousTensor");
  string yxVTname("yxViscousTensor");
  string yyVTname("yyViscousTensor");
  string yzVTname("yzViscousTensor");
  string zxVTname("zxViscousTensor");
  string zyVTname("zyViscousTensor");
  string zzVTname("zzViscousTensor");
  string viscosityCoefName("viscosityCoef");
  //string yViscosityCoefName("yViscosityCoef");
  //string zViscosityCoefName("zViscosityCoef");
  string dragCoefName("dragCoef");

  string basalThicknessSourceName("basalThicknessSource");
  string surfaceThicknessSourceName("surfaceThicknessSource");
  string divergenceThicknessFluxName("divergenceThicknessFlux");
  string activeBasalThicknessSourceName("activeBasalThicknessSource");
  string activeSurfaceThicknessSourceName("activeSurfaceThicknessSource");
  string calvedIceThicknessName("calvingFlux");
  string melangeThicknessName("melangeThickness");
  string calvedThicknessSourceName("calvedThicknessSource");

  Vector<string> vectName(numPlotComps);
  //int dThicknessComp;

  vectName[0] = thicknessName;
  vectName[1] = xVelName;
  if (SpaceDim > 1)
    vectName[2] = yVelName;
  int comp = SpaceDim+1;
  if (writeZvel) 
    {
      vectName[comp] = zVelName;
      comp++;
    }

  vectName[comp] = zsName;
  comp++;

  if (!m_reduced_plot)
    {
      vectName[comp] = zbottomName;
      comp++;
    }

  vectName[comp] = zbName;
  comp++;

  if (m_write_solver_rhs)
    {
      if (!m_reduced_plot)
	{
	  vectName[comp] = betaName;
	  comp++;
	  
	  vectName[comp] = C0Name;
	  comp++;
	}

      if (SpaceDim == 1)
        {
          vectName[comp] = solverRhsxName;
          comp++;
        }
      else if (SpaceDim == 2)
        {
          vectName[comp] = solverRhsxName;
          comp++;
          vectName[comp] = solverRhsyName;
          comp++;
        }
      else
        {
          MayDay::Error("writeSolverRHS undefined for this dimensionality");
        }
    }

  if (m_write_dHDt)
    {
      vectName[comp] = dthicknessName;      
      comp++;
    } 

  if (m_write_mask)
    {
      vectName[comp] = maskName;      
      comp++;
      vectName[comp] = fracName;
      comp++;
    } 


  if (m_write_fluxVel)
    {
      vectName[comp] = xfVelName;
      comp++;
      if (SpaceDim > 1)
        {
          vectName[comp] = yfVelName;
          comp++;
        }
      
      if (writeZvel) 
        {
          vectName[comp] = zfVelName;
          comp++;
        }
    }

 

  if (m_write_baseVel)
    {
      vectName[comp] = xbVelName;
      comp++;
      if (SpaceDim > 1)
        {
          vectName[comp] = ybVelName;
          comp++;
        }
      
      if (writeZvel) 
        {
          vectName[comp] = zbVelName;
          comp++;
        }
    }

  if (m_write_internal_energy)
    {
#if BISICLES_Z == BISICLES_LAYERED
      vectName[comp] = internalEnergyName + string("Surface");
      comp++;
#endif    
      for (int l = 0; l < m_internalEnergy[0]->nComp(); ++l)
	{
	  char idx[8]; sprintf(idx, "%04d", l);
	  vectName[comp] = internalEnergyName + string(idx);
	  comp++;
	}
    
#if BISICLES_Z == BISICLES_LAYERED
      vectName[comp] = internalEnergyName + string("Base");
      comp++;
      vectName[comp] = heatFluxName + string("Surface");
      comp++;
      vectName[comp] = heatFluxName + string("Base");
      comp++;
#endif 
    }

#if BISICLES_Z == BISICLES_LAYERED
  if (m_write_layer_velocities){
    for (int l = 0; l < m_nLayers + 1; ++l)
      {
	char idx[5]; sprintf(idx, "%04d", l);
	vectName[comp] = xlayerVelName + string(idx);
	comp++;
	vectName[comp] = ylayerVelName + string(idx);
	comp++;
	if (writeZvel) 
	  {
	    vectName[comp] = zlayerVelName + string(idx);
	    comp++;
	  }
      }
  }
#endif

  if (m_write_viscousTensor)
    {
      vectName[comp] = dragCoefName; comp++;
      vectName[comp] = viscosityCoefName; comp++;
      if (!m_reduced_plot)
	{
	  vectName[comp] = xxVTname;comp++;
	  if (SpaceDim > 1)
	    {
	      vectName[comp] = yxVTname;comp++;
	      if (SpaceDim > 2)
		{
		  vectName[comp] = zzVTname;comp++;
		}
	      vectName[comp] = xyVTname;comp++;
	      vectName[comp] = yyVTname;comp++;
	      
	      if (SpaceDim > 2)
		{
		  vectName[comp] = zyVTname;comp++;
		  vectName[comp] = xzVTname;comp++;
		  vectName[comp] = yzVTname;comp++;
		  vectName[comp] = zzVTname;comp++;
		}
	    }
	}
    }


  if (m_write_thickness_sources)
    {
      vectName[comp] = basalThicknessSourceName; comp++;
      vectName[comp] = surfaceThicknessSourceName; comp++;
      
      if (!m_reduced_plot)
	{
	  vectName[comp] = divergenceThicknessFluxName; comp++;	
	  vectName[comp] = activeBasalThicknessSourceName; comp++;
	  vectName[comp] = activeSurfaceThicknessSourceName; comp++;
	  vectName[comp] = calvedIceThicknessName; comp++;
	  vectName[comp] = melangeThicknessName; comp++;
	}
    }


  if (m_write_thickness_sources )
    {
      //update the surface thickness sources 
      for (int lev = 0; lev <= m_finest_level ; lev++)
	{
	  m_surfaceFluxPtr->surfaceThicknessFlux
	    (*m_surfaceThicknessSource[lev], *this, lev, m_dt);
	  m_basalFluxPtr->surfaceThicknessFlux
	    (*m_basalThicknessSource[lev], *this, lev, m_dt);
	}

    }
  // allow observers to add variables to the plot file
  for (int i = 0; i < m_observers.size(); i++)
    m_observers[i]->addPlotVars(vectName);
  numPlotComps = vectName.size();


  Box domain = m_amrDomains[0].domainBox();
  int numLevels = m_finest_level +1;
  // compute plot data
  Vector<LevelData<FArrayBox>* > plotData(m_velocity.size(), NULL);

  // temp storage for C0
  Vector<LevelData<FArrayBox>* > vectC0(m_velocity.size(), NULL);

  // ghost vect makes things simpler
  IntVect ghostVect(IntVect::Unit);
  
  for (int lev=0; lev<numLevels; lev++)
    {
      // first allocate storage
      plotData[lev] = new LevelData<FArrayBox>(m_amrGrids[lev],
                                               numPlotComps,
                                               ghostVect);

      vectC0[lev] = new LevelData<FArrayBox>(m_amrGrids[lev],
                                             1,
                                             m_velBasalC[0]->ghostVect());
      DataIterator dit = m_amrGrids[lev].dataIterator();
      for (dit.begin(); dit.ok(); ++dit)
        {
          (*vectC0[lev])[dit].setVal(0.0);
        }
    }

  if (m_write_solver_rhs)
    {
      setBasalFriction(m_velBasalC, vectC0);
      defineVelRHS(m_velRHS);
      
    }


  for (int lev=0; lev<numLevels; lev++)
    {
      // now copy new-time solution into plotData
      Interval thicknessComps(0,0);
      Interval velocityComps(1,SpaceDim);

      LevelData<FArrayBox>& plotDataLev = *plotData[lev];

      const LevelSigmaCS& levelCS = (*m_vect_coordSys[lev]);
      const LevelData<FArrayBox>& levelH = levelCS.getH();
      const LevelData<FArrayBox>& levelZbase = levelCS.getTopography();
      LevelData<FArrayBox> levelZsurf(m_amrGrids[lev], 1, ghostVect);
      levelCS.getSurfaceHeight(levelZsurf);

      LevelData<FArrayBox> levelSurfaceCrevasseDepth (m_amrGrids[lev], 1, ghostVect);
      LevelData<FArrayBox> levelBasalCrevasseDepth (m_amrGrids[lev], 1, ghostVect);
      LevelData<FArrayBox> levelSTS (m_amrGrids[lev], 1, ghostVect);
      LevelData<FArrayBox> levelBTS (m_amrGrids[lev], 1, ghostVect);
      
      if (m_write_viscousTensor)
	{
	  computeCrevasseDepths(levelSurfaceCrevasseDepth,levelBasalCrevasseDepth,lev);
	}

      if (m_write_thickness_sources)
	{
	  m_surfaceFluxPtr->surfaceThicknessFlux(levelSTS, *this, lev, m_dt);
	  m_basalFluxPtr->surfaceThicknessFlux(levelBTS, *this, lev, m_dt);      
	}

      DataIterator dit = m_amrGrids[lev].dataIterator();
      for (dit.begin(); dit.ok(); ++dit)
        {
          const Box& gridBox = m_amrGrids[lev][dit];
          FArrayBox& thisPlotData = plotDataLev[dit];
          comp = 0;
          const FArrayBox& thisH = levelH[dit];
          
          thisPlotData.copy(thisH, 0, comp, 1);

          comp++;
          const FArrayBox& thisVel = (*m_velocity[lev])[dit];
          thisPlotData.copy(thisVel, 0, comp, SpaceDim);
          
          comp += SpaceDim;
	 
          if (writeZvel) 
            {
              // use zVel = zero for the moment
              Real zVel = 0.0;
              thisPlotData.setVal(zVel, comp);
              ++comp;
            }

          const FArrayBox& zBase = levelZbase[dit];
          
          // account for background slope of base 
          FArrayBox backgroundBase(thisPlotData.box(), 1);
          BoxIterator bit(thisPlotData.box());
	  const RealVect& basalSlope = levelCS.getBackgroundSlope();
          for (bit.begin(); bit.ok(); ++bit)
            {
              IntVect iv = bit();
              RealVect loc(iv);
              loc += 0.5*RealVect::Unit;
              loc *= m_amrDx[lev];

              backgroundBase(iv,0) = D_TERM(loc[0]*basalSlope[0],
                                            +loc[1]*basalSlope[1],
                                            +loc[2]*basalSlope[2]);
            }
          // zsurface
          FArrayBox& zSurf = levelZsurf[dit];
          thisPlotData.copy(zSurf, 0, comp, 1);
          thisPlotData.plus(backgroundBase, 0, comp, 1);
          ++comp;

	  if (!m_reduced_plot)
	    {
	      // zbottom (bottom of ice
	      thisPlotData.copy(zSurf, 0, comp, 1);
	      thisPlotData.minus(thisH, 0, comp, 1);
	      thisPlotData.plus(backgroundBase, 0, comp, 1);
	      ++comp;
	    }

          // zbase 
          thisPlotData.copy(zBase, 0, comp, 1);
          thisPlotData.plus(backgroundBase, 0, comp, 1);
          ++comp;

          if (m_write_solver_rhs)
            {
	      if (!m_reduced_plot)
		{
		  thisPlotData.copy((*m_velBasalC[lev])[dit],0,comp,1);
		  comp++;
		  thisPlotData.copy((*vectC0[lev])[dit],0,comp,1);
		  comp++;
		}
	      thisPlotData.copy((*m_velRHS[lev])[dit],0,comp,SpaceDim);
              comp += SpaceDim;
            }

          // now copy for dthickness/dt 
          if (m_write_dHDt)
            {
              const FArrayBox& thisOldH = (*m_old_thickness[lev])[dit];
              thisPlotData.copy(thisH, 0, comp, 1);
              thisPlotData.minus(thisOldH, 0, comp, 1);
              if (m_dt > 0)
                {
                  thisPlotData.divide(m_dt, comp, 1);

                }              
              ++comp;

	    } // end if we are computing dHDt
      
	  if (m_write_mask)
            {
	      const BaseFab<int>& mask = levelCS.getFloatingMask()[dit];
	      FArrayBox tmp(mask.box(),1);
	      for (BoxIterator bit(mask.box());bit.ok();++bit)
		{
		  tmp(bit()) = Real( mask(bit()) ) ;
		}
	      thisPlotData.copy(tmp,0,comp,1);
	      comp++;
              // now copy real-valued ice fraction
              const FArrayBox& iceFracFab = (*m_iceFrac[lev])[dit];
              thisPlotData.copy(iceFracFab,0,comp,1);
	      comp++;

	    }

	  // const FArrayBox& thisSurfaceVel = (*m_velocity[lev])[dit];
          // thisPlotData.copy(thisSurfaceVel, 0, comp, 2);
          
          // comp += 2;

	  

          if (m_write_fluxVel)
            {
              for (int dir = 0; dir < SpaceDim; ++dir)
                {
                  
                  const FArrayBox& thisVel = (*m_faceVelTotal[lev])[dit][dir];
                  for (BoxIterator bit(gridBox); bit.ok(); ++bit)
                    {
                      const IntVect& iv = bit();
                      const IntVect ivp = iv + BASISV(dir);
                      thisPlotData(iv,comp) = half*(thisVel(iv) + thisVel(ivp));
                    }
                  comp++;
                }            

              if (writeZvel) 
                {
                  // use zVel = zero for the moment
                  Real zVel = 0.0;
                  thisPlotData.setVal(zVel, comp);
                  ++comp;
                }
            }



          if (m_write_baseVel)
            {
              const FArrayBox& thisBaseVel = (*m_velocity[lev])[dit];
              thisPlotData.copy(thisBaseVel, 0, comp, SpaceDim);
              
              comp += SpaceDim;
              
              if (writeZvel) 
                {
                  // use zVel = zero for the moment
                  Real zVel = 0.0;
                  thisPlotData.setVal(zVel, comp);
                  ++comp;
                }	  
            }
	  if (m_write_internal_energy)
	    {
#if BISICLES_Z == BISICLES_LAYERED
	      {
		const FArrayBox& thisTemp = (*m_sInternalEnergy[lev])[dit];
		thisPlotData.copy(thisTemp, 0, comp, thisTemp.nComp());
		comp++;
	      }
#endif
	      {
		const FArrayBox& thisTemp = (*m_internalEnergy[lev])[dit];
		thisPlotData.copy(thisTemp, 0, comp, thisTemp.nComp());
		comp += thisTemp.nComp();
	      }
#if BISICLES_Z == BISICLES_LAYERED
	      {
		const FArrayBox& thisTemp = (*m_bInternalEnergy[lev])[dit];
		thisPlotData.copy(thisTemp, 0, comp, thisTemp.nComp());
		comp++;
		thisPlotData.copy((*m_sHeatFlux[lev])[dit], 0, comp, 1);
		comp++;
		thisPlotData.copy((*m_bHeatFlux[lev])[dit], 0, comp, 1);
		comp++;
	      }
#endif
	    }
#if BISICLES_Z == BISICLES_LAYERED
	  if (m_write_layer_velocities)
	    {
	      const FArrayBox& thisVel = (*m_layerSFaceXYVel[lev])[dit];
	     
	      for (int j = 0; j < m_nLayers + 1; ++j)
		{
		  thisPlotData.copy(thisVel, j*SpaceDim, comp, SpaceDim);
		  
		  comp+= SpaceDim;
		  // end loop over components
		  if (writeZvel) 
		    {
		      // use zVel = zero for the moment
		      Real zVel = 0.0;
		      thisPlotData.setVal(zVel, comp);
		      ++comp;
		    } 
	      
		} // end loop over layers
	    }
#endif
	  if (m_write_viscousTensor)
	    {
	      thisPlotData.copy( (*dragCoefficient(lev))[dit],0,comp);
	      comp++;
	      thisPlotData.copy( (*viscosityCoefficient(lev))[dit],0,comp);
	      comp++;
	      if (!m_reduced_plot)
		{
		  thisPlotData.copy( (*viscousTensor(lev))[dit],0,comp, SpaceDim*SpaceDim);
		  comp += SpaceDim * SpaceDim;
		}
	    }
	 
	  if (m_write_thickness_sources)
	    {
	      thisPlotData.copy(levelBTS[dit], 0, comp, 1);
              if (m_frac_sources)
                {
                  thisPlotData.mult( (*m_iceFrac[lev])[dit],0,comp,1);
                }

	      comp++;
	      thisPlotData.copy(levelSTS[dit], 0, comp, 1);
              if (m_frac_sources)
                {
                  // scale by ice fraction
                  thisPlotData.mult( (*m_iceFrac[lev])[dit],0,comp,1);
                }
	      comp++;

	      if (!m_reduced_plot)
		{
		  thisPlotData.copy((*m_divThicknessFlux[lev])[dit], 0, comp, 1);
		  comp++;

		  thisPlotData.copy((*m_basalThicknessSource[lev])[dit], 0, comp, 1);
		  if (m_frac_sources)
		    {
		      thisPlotData.mult( (*m_iceFrac[lev])[dit],0,comp,1);
		    }
		  comp++;
		  
	 
		  thisPlotData.copy((*m_surfaceThicknessSource[lev])[dit], 0, comp, 1);
		  if (m_frac_sources)
		    {
		      // scale by ice fraction
		      thisPlotData.mult( (*m_iceFrac[lev])[dit],0,comp,1);
		    }
		  comp++;
	      
		  thisPlotData.copy((*m_calvedIceThickness[lev])[dit], 0, comp, 1);
		  if (m_dt > 0)
		    {
		      thisPlotData.divide(m_dt, comp, 1);
		    }              
		  comp++;
		  
		  thisPlotData.copy((*m_melangeThickness[lev])[dit], 0, comp, 1);
		  comp++;
      
		}
	    }
		
	} // end loop over boxes on this level

      
      //allow observers to write data. 
      for (int i = 0; i < m_observers.size(); i++)
	{
	  Vector<std::string> vars;
	  m_observers[i]->addPlotVars(vars);
	  if (vars.size() > 0)
	    {
	      Interval interval(comp, comp + vars.size() - 1);
	      LevelData<FArrayBox> obsPlotData;
	      aliasLevelData( obsPlotData, plotData[lev], interval);
	      m_observers[i]->writePlotData(obsPlotData, lev);
	    }
	}


      // this is just so that visit surface plots look right
      // fill coarse-fine ghost-cell values with interpolated data
      if (lev > 0)
        {
          PiecewiseLinearFillPatch interpolator(m_amrGrids[lev],
                                                m_amrGrids[lev-1],
                                                numPlotComps,
                                                m_amrDomains[lev-1],
                                                m_refinement_ratios[lev-1],
                                                ghostVect[0]);
          
          // no interpolation in time
          Real time_interp_coeff = 0.0;
          interpolator.fillInterp(*plotData[lev],
                                  *plotData[lev-1],
                                  *plotData[lev-1],
                                  time_interp_coeff,
                                  0, 0,  numPlotComps);
        }
      // just in case...
      plotData[lev]->exchange();
    } // end loop over levels for computing plot data
  
  // generate plotfile name
  std::string fs("%s%06d.");
  char* iter_str = new char[m_plot_prefix.size() + fs.size() + 16];
  sprintf(iter_str, fs.c_str(), m_plot_prefix.c_str(), m_cur_step );
  string filename(iter_str);

  delete[] iter_str;
 
  // need to pull out SigmaCS pointers:
  Vector<const LevelSigmaCS* > vectCS(m_vect_coordSys.size(), NULL);
  for (int lev=0; lev<numLevels; lev++)
    {
      vectCS[lev] = dynamic_cast<const LevelSigmaCS* >(&(*m_vect_coordSys[lev]));
    }
  if (m_write_map_file)
    {
      WriteSigmaMappedAMRHierarchyHDF5(filename, m_amrGrids, plotData, vectName, 
                                       vectCS, domain, m_dt, m_time,
                                       m_refinement_ratios,
                                       numLevels);
    }
  else
    {
      if (SpaceDim == 1)
        {
          filename.append("1d.hdf5");
        } 
      else if (SpaceDim == 2)
        {
          filename.append("2d.hdf5");
        } 
      else if (SpaceDim == 3)
        {
          filename.append("3d.hdf5");
        }

      this->writeAMRHierarchyHDF5(filename, m_amrGrids, plotData, vectName, 
				  domain, m_amrDx[0], m_dt, time(), m_refinement_ratios, 
				  numLevels);

    }

  // need to delete plotData
  for (int lev=0; lev<numLevels; lev++)
    {
      if (plotData[lev] != NULL)
        {
          delete plotData[lev];
          plotData[lev] = NULL;
        }      

      if (vectC0[lev] != NULL)
        {
          delete vectC0[lev];
          vectC0[lev] = NULL;
        } 

      // if (faceA[lev] != NULL)
      // 	{
      // 	  delete faceA[lev];
      // 	  faceA[lev] = NULL;
      // 	}
    
      // if (viscousTensor[lev] != NULL)
      // 	{
      // 	  delete viscousTensor[lev];
      // 	  viscousTensor[lev] = NULL;
      // 	}
    }
}

/// write checkpoint file out for later restarting
void 
AmrIce::writeCheckpointFile() 
{
  if (s_verbosity > 3) 
    { 
      pout() << "AmrIce::writeCheckpointfile" << endl;
    }

  CH_TIME("AmrIce::writeCheckpointFile");

  // generate checkpointfile name
  char* iter_str;
  if (m_check_overwrite)
    {
      // overwrite the same checkpoint file, rather than re-writing them
      std::string fs("%s.%dd.hdf5");
      iter_str = new char[m_check_prefix.size() + fs.size() + 16];
      sprintf(iter_str, "%s.%dd.hdf5", m_check_prefix.c_str(), SpaceDim);
      
    }
  else 
    {
      // or hang on to them, if you are a bit sentimental. It's better than keeping
      // every core dump you generate.
      std::string fs("%s%06d.%dd.hdf5");
      iter_str = new char[m_check_prefix.size() + fs.size() + 16];
      sprintf(iter_str, "%s%06d.%dd.hdf5", m_check_prefix.c_str(), m_cur_step, SpaceDim);
     
    }

  CH_assert(iter_str != NULL);

  if (s_verbosity > 3) 
    {
      pout() << "checkpoint file name = " << iter_str << endl;
    }

  writeCheckpointFile(std::string(iter_str));
  delete[] iter_str;
}

/// write checkpoint file out for later restarting
void 
AmrIce::writeCheckpointFile(const string& a_file) 
{

  if (s_verbosity > 3) 
    { 
      pout() << "AmrIce::writeCheckpointfile" << endl;
      pout() << "checkpoint file name = " << a_file << endl;
    }

#ifdef CH_USE_HDF5

  string thicknessName("thickness");
  Vector<string> vectName(1);
  for (int comp=0; comp<1; comp++)
    {
      char idx[5]; sprintf(idx, "%d", comp);
      vectName[comp] = thicknessName+string(idx);
    } 
  Box domain = m_amrDomains[0].domainBox();
  //int numLevels = m_finest_level +1;      

  
  HDF5Handle handle(a_file.c_str(), HDF5Handle::CREATE);

  // write amr data -- only dump out things which are essential
  // to restarting the computation (i.e. max_level, finest_level, 
  // time, refinement ratios, etc.).  Other paramters (regrid 
  // intervals, block-factor, etc can be changed by the inputs
  // file of the new run.
  // At the moment, the maximum level is not allowed to change,
  // although in principle, there is no real reason why it couldn't
  // 
  HDF5HeaderData&  header = m_headerData;
  header.m_int["max_level"] = m_max_level;
  header.m_int["finest_level"] = m_finest_level;
  header.m_int["current_step"] = m_cur_step;
  header.m_real["time"] = m_time;
  header.m_real["dt"] = m_dt;
  header.m_int["num_comps"] = 2 +  m_velocity[0]->nComp() 
    + m_internalEnergy[0]->nComp() + m_deltaTopography[0]->nComp()
    + m_melangeThickness[0]->nComp();
#if BISICLES_Z == BISICLES_LAYERED



  header.m_int["num_comps"] +=2; // surface and base internalEnergys
#endif
  // at the moment, save cfl, but it can be changed by the inputs
  // file if desired.
  header.m_real["cfl"] = m_cfl;

  // periodicity info
  D_TERM(
         if (m_amrDomains[0].isPeriodic(0))
	   header.m_int["is_periodic_0"] = 1;
         else
	   header.m_int["is_periodic_0"] = 0; ,

         if (m_amrDomains[0].isPeriodic(1))
	   header.m_int["is_periodic_1"] = 1;
         else
	   header.m_int["is_periodic_1"] = 0; ,

         if (m_amrDomains[0].isPeriodic(2))
	   header.m_int["is_periodic_2"] = 1;
         else
	   header.m_int["is_periodic_2"] = 0; 
         );
         

  // set up component names
  char compStr[30];
  //string thicknessName("thickness");
  string compName;
  int nComp = 0;
  for (int comp=0; comp < 1; comp++)
    {
      // first generate component name
      char idx[5]; sprintf(idx, "%04d", comp);
      compName = thicknessName + string(idx);
      sprintf(compStr, "component_%04d", comp);
      header.m_string[compStr] = compName;
     
    }
  nComp++;

  string baseHeightName("bedHeight");
  for (int comp=0; comp < 1; comp++)
    {
      // first generate component name
      char idx[5]; sprintf(idx, "%04d", comp);
      compName = baseHeightName + string(idx);
      sprintf(compStr, "component_%04d", comp + nComp);
      header.m_string[compStr] = compName;
      
    }
  nComp++;

  string baseDeltaName("deltaBedHeight");
  for (int comp=0; comp < 1; comp++)
    {
      // first generate component name
      char idx[5]; sprintf(idx, "%04d", comp);
      compName = baseDeltaName + string(idx);
      sprintf(compStr, "component_%04d", comp + nComp);
      header.m_string[compStr] = compName;
      
    }
  nComp++;

  string calvedIceThckName("melangeThck");
  for (int comp=0; comp < 1; comp++)
    {
      // first generate component name
      char idx[5]; sprintf(idx, "%04d", comp);
      compName = calvedIceThckName + string(idx);
      sprintf(compStr, "component_%04d", comp + nComp);
      header.m_string[compStr] = compName;
      
    }
  nComp++;

  string iceFracName("iceFrac");
  for (int comp=0; comp < 1; comp++)
    {
      // first generate component name
      char idx[5]; sprintf(idx, "%04d", comp);
      compName = iceFracName + string(idx);
      sprintf(compStr, "component_%04d", comp + nComp);
      header.m_string[compStr] = compName;
      
    }
  nComp++;

  string velocityName("velocity");
  for (int comp=0; comp < m_velocity[0]->nComp() ; comp++) 
    {
      // first generate component name
      char idx[5]; sprintf(idx, "%04d", comp);
      compName = velocityName + string(idx);
      sprintf(compStr, "component_%04d", comp + nComp);
      header.m_string[compStr] = compName;
    }
  nComp += m_velocity[0]->nComp() ;

  string basalFrictionName("basalFriction");
  for (int comp=0; comp < 1; comp++)
    {
      char idx[5]; sprintf(idx, "%04d", comp);
      compName = basalFrictionName + string(idx);
      sprintf(compStr, "component_%04d", comp + nComp);
      header.m_string[compStr] = compName;
    }
  nComp++;

  string muCoefName("muCoef");
  for (int comp=0; comp < 1; comp++)
    {
      char idx[5]; sprintf(idx, "%04d", comp);
      compName = muCoefName + string(idx);
      sprintf(compStr, "component_%04d", comp + nComp);
      header.m_string[compStr] = compName;
    }
  nComp++;


  string internalEnergyName("internalEnergy");
  for (int comp=0; comp < m_internalEnergy[0]->nComp() ; comp++) 
    {
      char idx[5]; sprintf(idx, "%04d", comp);
      compName = internalEnergyName + string(idx);
      sprintf(compStr, "component_%04d", comp + nComp);
      header.m_string[compStr] = compName;
    }
  
  nComp += m_internalEnergy[0]->nComp() ;

#if BISICLES_Z == BISICLES_LAYERED
  {
    sprintf(compStr, "component_%04d", nComp);
    compName = "sInternalEnergy";
    header.m_string[compStr] = compName;
    nComp += 1;
    sprintf(compStr, "component_%04d", nComp);
    compName = "bInternalEnergy";
    header.m_string[compStr] = compName;
    nComp += 1;
    //layer data
    const Vector<Real>& sigma = getFaceSigma();
    string s("sigma");
    for (int l =0; l < sigma.size(); ++l)
      {
	char idx[5]; sprintf(idx, "%04d", l);
	header.m_real[s + string(idx)] = sigma[l];  
      }
  }
#endif

  //allow observers to add checkpoint variables
  for (int i = 0; i < m_observers.size(); i++)
    {
      Vector<std::string> vars;
      m_observers[i]->addCheckVars(vars);
      for (int j = 0; j < vars.size(); j++)
	{
	  nComp++;
	  sprintf(compStr, "component_%04d", nComp);
	  header.m_string[compStr] = vars[j].c_str();
	}
    }

  header.writeToFile(handle);

  // now loop over levels and write out each level's data
  // note that we loop over all allowed levels, even if they
  // are not defined at the moment.
  for (int lev=0; lev<= m_max_level; lev++)
    {
      // set up the level string
      char levelStr[20];
      sprintf(levelStr, "%d", lev);
      const std::string label = std::string("level_") + levelStr;
      
      handle.setGroup(label);
      
      // set up the header info
      HDF5HeaderData levelHeader;
      if (lev < m_max_level)
        {
          levelHeader.m_int["ref_ratio"] = m_refinement_ratios[lev];
        }
      levelHeader.m_real["dx"] = m_amrDx[lev];
      levelHeader.m_box["prob_domain"] = m_amrDomains[lev].domainBox();
      
      levelHeader.writeToFile(handle);
      
      // now write the data for this level
      // only try to write data if level is defined.
      if (lev <= m_finest_level)
        {
          write(handle, m_amrGrids[lev]);

	  const IntVect ghost = IntVect::Unit*2;
          const LevelSigmaCS& levelCS = *m_vect_coordSys[lev];

	  write(handle, levelCS.getH() , "thicknessData", levelCS.getH().ghostVect());

	  write(handle, levelCS.getTopography() , "bedHeightData",
		levelCS.getTopography().ghostVect()  );

	  write(handle, *m_deltaTopography[lev] , "deltaBedHeightData",
		m_deltaTopography[lev]->ghostVect()  );

	  write(handle, *m_melangeThickness[lev] , "melangeThckData",
		m_melangeThickness[lev]->ghostVect()  );

	  write(handle, *m_iceFrac[lev] , "iceFracData",
		m_iceFrac[lev]->ghostVect()  );

	  write(handle, *m_velocity[lev], "velocityData", 
		m_velocity[lev]->ghostVect());

	  write(handle, *m_velBasalC[lev], "basalFrictionData", 
		m_velBasalC[lev]->ghostVect());

	  write(handle, *m_cellMuCoef[lev], "muCoefData", 
		m_cellMuCoef[lev]->ghostVect());
	  
	  write(handle, *m_internalEnergy[lev], "internalEnergyData", 
		m_internalEnergy[lev]->ghostVect());

#if BISICLES_Z == BISICLES_LAYERED
	  write(handle, *m_sInternalEnergy[lev], "sInternalEnergyData", 
		m_sInternalEnergy[lev]->ghostVect());
	  write(handle, *m_bInternalEnergy[lev], "bInternalEnergyData", 
		m_bInternalEnergy[lev]->ghostVect());
#endif
	  //allow observers to write to the checkpoint
	  for (int i = 0; i < m_observers.size(); i++)
	    {
	      m_observers[i]->writeCheckData(handle, lev);
	    }
        }
    }// end loop over levels
  
  handle.close();
#endif
}


/// read checkpoint file for restart 
void 
AmrIce::readCheckpointFile(HDF5Handle& a_handle)
{

  if (s_verbosity > 3) 
    { 
      pout() << "AmrIce::readCheckpointFile" << endl;
    }

#ifndef CH_USE_HDF5
  MayDay::Error("code must be compiled with HDF5 to read checkpoint files");
#endif

#ifdef CH_USE_HDF5

 
  HDF5HeaderData& header = m_headerData;
  header.readFromFile(a_handle);

  //check for various components. Maybe rethink this when HDF5::SetGroup
  //is fixed...
  bool containsDeltaBedHeight(false);
  bool containsAccumCalvedIceThck(false);
  bool containsInternalEnergy(false);
  bool containsTemperature(false);
  bool containsIceFrac(false);
  bool containsBasalFriction(false);
  bool containsMuCoef(false);

  map<std::string, std::string>::const_iterator i;
  for (i = header.m_string.begin(); i!= header.m_string.end(); ++i)
    {
      if (i->second == "deltaBedHeight0000")
	{
	  containsDeltaBedHeight = true;
	}
      if (i->second == "melangeThck0000")
	{
	  containsAccumCalvedIceThck = true;
	}
      if (i->second == "temperature0000")
	{
	  containsTemperature = true;
	}
      if (i->second == "internalEnergy0000")
	{
	  containsInternalEnergy = true;
	}
      if (i->second == "iceFrac0000")
	{
	  containsIceFrac = true;
	}
      if (i->second == "basalFriction0000")
	{
	  containsBasalFriction = true;
	}
      if (i->second == "muCoef0000")
	{
	  containsMuCoef = true;
	}
    }

  if (s_verbosity >= 3)
    {
      pout() << "hdf5 header data: " << endl;
      pout() << header << endl;
    }

  // read max level
  if (header.m_int.find("max_level") == header.m_int.end())
    {
      MayDay::Error("checkpoint file does not contain max_level");
    }
  // we can change max level upon restart
  int max_level_check = header.m_int["max_level"];
  if (max_level_check != m_max_level)
    {
      if (s_verbosity > 0)
        {
          pout() << "Restart file has a different max level than inputs file"
                 << endl;
          pout() << "     max level from inputs file = " 
                 << m_max_level << endl;
          pout() << "     max level in checkpoint file = " 
                 << max_level_check << endl;                 
          pout() << "Using max level from inputs file" << endl;
        }
    }
  // read finest level
  if (header.m_int.find("finest_level") == header.m_int.end())
    {
      MayDay::Error("checkpoint file does not contain finest_level");
    }

  m_finest_level = header.m_int["finest_level"];
  if (m_finest_level > m_max_level)
    {
      MayDay::Error("finest level in restart file > max allowable level!");
    }

  // read current step
  if (header.m_int.find("current_step") == header.m_int.end())
    {
      MayDay::Error("checkpoint file does not contain current_step");
    }

  m_cur_step = header.m_int["current_step"];
  m_restart_step = m_cur_step;

  // read time
  if (header.m_real.find("time") == header.m_real.end())
    {
      MayDay::Error("checkpoint file does not contain time");
    }

  m_time = header.m_real["time"];
  m_dt = header.m_real["dt"];

  // read num comps
  if (header.m_int.find("num_comps") == header.m_int.end())
    {
      MayDay::Error("checkpoint file does not contain num_comps");
    }
  
  //int numComps = header.m_int["num_comps"];

  // read cfl
  if (header.m_real.find("cfl") == header.m_real.end())
    {
      MayDay::Error("checkpoint file does not contain cfl");
    }

  Real check_cfl = header.m_real["cfl"];
  ParmParse ppCheck("amr");

  if (ppCheck.contains("cfl"))
    { 
      // check for consistency and warn if different
      if (check_cfl != m_cfl)
	{
	  if (s_verbosity > 0)
	    {
	      pout() << "CFL in checkpoint file different from inputs file" 
		     << endl;
	      pout() << "     cfl in inputs file = " << m_cfl << endl;
	      pout() << "     cfl in checkpoint file = " << check_cfl 
		     << endl;
	      pout() << "Using cfl from inputs file" << endl;                
	    }
	}  // end if cfl numbers differ
    } // end if cfl present in inputs file
  else
    {
      m_cfl = check_cfl;
    }          

  // read periodicity info
  // Get the periodicity info -- this is more complicated than it really
  // needs to be in order to preserve backward compatibility 
  // bool isPeriodic[SpaceDim];
  // D_TERM(if (!(header.m_int.find("is_periodic_0") == header.m_int.end()))
  //          isPeriodic[0] =  (header.m_int["is_periodic_0"] == 1);
  //        else
  //          isPeriodic[0] = false; ,

  //        if (!(header.m_int.find("is_periodic_1") == header.m_int.end()))
  //          isPeriodic[1] =  (header.m_int["is_periodic_1"] == 1);
  //        else
  //          isPeriodic[1] = false; ,

  //        if (!(header.m_int.find("is_periodic_2") == header.m_int.end()))
  //          isPeriodic[2] =  (header.m_int["is_periodic_2"] == 1);
  //        else
  //          isPeriodic[2] = false;);

#if BISICLES_Z == BISICLES_LAYERED
  //retrieve sigma data
  Vector<Real> sigma;
  int l = 0;
  string s("sigma");
  bool found = false;
  do {
    char idx[6]; sprintf(idx, "%04d", l);
    string ss = s + string(idx);
    map<std::string, Real>::const_iterator it = header.m_real.find(ss);
    found = (it != header.m_real.end());
    if (found)
      sigma.push_back(it->second);
    ++l;
  } while (found);
  m_nLayers = sigma.size() - 1;
  CH_assert(m_nLayers > 0 && sigma[0] < TINY_NORM && abs(sigma[m_nLayers] - 1.0) < TINY_NORM);
#endif

  // now resize stuff 
  m_amrDomains.resize(m_max_level+1);
  m_amrGrids.resize(m_max_level+1);
  m_amrDx.resize(m_max_level+1);
  m_old_thickness.resize(m_max_level+1, NULL);
  m_iceFrac.resize(m_max_level+1, NULL);
  m_velocity.resize(m_max_level+1, NULL);
  m_diffusivity.resize(m_max_level+1);
  m_vect_coordSys.resize(m_max_level+1);
  m_velRHS.resize(m_max_level+1);
  m_surfaceThicknessSource.resize(m_max_level+1,NULL);
  m_basalThicknessSource.resize(m_max_level+1,NULL);
  m_calvedIceThickness.resize(m_max_level+1, NULL);
  m_removedIceThickness.resize(m_max_level+1, NULL);
  m_addedIceThickness.resize(m_max_level+1, NULL);
  m_melangeThickness.resize(m_max_level+1, NULL);
  m_recordThickness.resize(m_max_level+1, NULL);
  m_deltaTopography.resize(m_max_level+1, NULL);
  m_divThicknessFlux.resize(m_max_level+1,NULL);
  m_velBasalC.resize(m_max_level+1,NULL);
  m_cellMuCoef.resize(m_max_level+1,NULL);
  m_faceVelAdvection.resize(m_max_level+1,NULL);
  m_faceVelTotal.resize(m_max_level+1,NULL);
  m_internalEnergy.resize(m_max_level+1,NULL);
#if BISICLES_Z == BISICLES_LAYERED
  m_sInternalEnergy.resize(m_max_level+1,NULL);
  m_bInternalEnergy.resize(m_max_level+1,NULL);
  m_sHeatFlux.resize(m_max_level+1,NULL);
  m_bHeatFlux.resize(m_max_level+1,NULL);
  m_layerSFaceXYVel.resize(m_max_level+1,NULL);
  m_layerXYFaceXYVel.resize(m_max_level+1,NULL);
#endif
  IntVect sigmaCSGhost = m_num_thickness_ghost*IntVect::Unit;
	 

  // now read in level-by-level data
  for (int lev=0; lev<= m_max_level; lev++)
    {
      // set up the level string
      char levelStr[20];
      sprintf(levelStr, "%d", lev);
      const std::string label = std::string("level_") + levelStr;
      
      a_handle.setGroup(label);

      // read header info
      HDF5HeaderData levheader;
      levheader.readFromFile(a_handle);
      
      if (s_verbosity >= 3)
        {
          pout() << "level " << lev << " header data" << endl;
          pout() << levheader << endl;
        }

      // Get the refinement ratio
      if (lev < m_max_level)
        {
          int checkRefRatio;
          if (levheader.m_int.find("ref_ratio") == levheader.m_int.end())
            {
              MayDay::Error("checkpoint file does not contain ref_ratio");
            }
          checkRefRatio = levheader.m_int["ref_ratio"];

          // check for consistency
          if (checkRefRatio != m_refinement_ratios[lev])
            {
	      
	      MayDay::Error("inputs file and checkpoint file ref ratios inconsistent");
            }
        }
      
      // read dx
      if (levheader.m_real.find("dx") == levheader.m_real.end())
        {
          MayDay::Error("checkpoint file does not contain dx");
        }
      
      if ( Abs(m_amrDx[lev] - levheader.m_real["dx"]) > TINY_NORM )
	{
	  MayDay::Error("restart file dx != input file dx");
	}
      
      // read problem domain box
      if (levheader.m_box.find("prob_domain") == levheader.m_box.end())
        {
          MayDay::Error("checkpoint file does not contain prob_domain");
        }
      Box domainBox = levheader.m_box["prob_domain"];

      if (m_amrDomains[lev].domainBox() != domainBox)
	{ 
	  MayDay::Error("restart file domain != input file domain");
	}

      // the rest is only applicable if this level is defined
      if (lev <= m_finest_level)
        {
          // read grids          
          Vector<Box> grids;
          const int grid_status = read(a_handle, grids);
          if (grid_status != 0) 
            {
              MayDay::Error("checkpoint file does not contain a Vector<Box>");
            }
          // do load balancing
          int numGrids = grids.size();
          Vector<int> procIDs(numGrids);
          LoadBalance(procIDs, grids);
          DisjointBoxLayout levelDBL(grids, procIDs, m_amrDomains[lev]);
          m_amrGrids[lev] = levelDBL;

          // allocate this level's storage
	  // 4 ghost cells needed for advection.
          m_old_thickness[lev] = new LevelData<FArrayBox>
	    (levelDBL, 1, m_num_thickness_ghost*IntVect::Unit);
#if BISICLES_Z == BISICLES_LAYERED
	  m_internalEnergy[lev] =  new LevelData<FArrayBox>
	    (levelDBL, m_nLayers, m_num_thickness_ghost*IntVect::Unit);
	  m_sInternalEnergy[lev] =  new LevelData<FArrayBox>
	    (levelDBL, 1, m_num_thickness_ghost*IntVect::Unit);
	  m_bInternalEnergy[lev] =  new LevelData<FArrayBox>
	    (levelDBL, 1, m_num_thickness_ghost*IntVect::Unit);
	  m_sHeatFlux[lev] =  new LevelData<FArrayBox>
	    (levelDBL, 1, m_num_thickness_ghost*IntVect::Unit);
	  m_bHeatFlux[lev] =  new LevelData<FArrayBox>
	    (levelDBL, 1, m_num_thickness_ghost*IntVect::Unit);
#elif BISICLES_Z == BISICLES_FULLZ
	  m_internalEnergy[lev] =  new LevelData<FArrayBox>
	    (levelDBL, 1, m_num_thickness_ghost*IntVect::Unit);
#endif
	  // other quantities need only one;
	  IntVect ghostVect(IntVect::Unit);
          m_velocity[lev] = new LevelData<FArrayBox>(levelDBL, SpaceDim, 
                                                     ghostVect);

          m_iceFrac[lev] = new LevelData<FArrayBox>(levelDBL, 1, ghostVect);

	  m_faceVelAdvection[lev] = new LevelData<FluxBox>(m_amrGrids[lev], 1, IntVect::Unit);
	  m_faceVelTotal[lev] = new LevelData<FluxBox>(m_amrGrids[lev], 1, IntVect::Unit);
#if BISICLES_Z == BISICLES_LAYERED
	  m_layerXYFaceXYVel[lev] = new LevelData<FluxBox>
	    (m_amrGrids[lev], m_nLayers, IntVect::Unit);
	  m_layerSFaceXYVel[lev] = new LevelData<FArrayBox>
	    (m_amrGrids[lev], SpaceDim*(m_nLayers + 1), IntVect::Unit);
#endif

	  m_velBasalC[lev] = new LevelData<FArrayBox>(levelDBL, 1, ghostVect);
	  m_cellMuCoef[lev] = new LevelData<FArrayBox>(levelDBL, 1, ghostVect);
	  m_velRHS[lev] = new LevelData<FArrayBox>(levelDBL, SpaceDim, IntVect::Zero);
	  m_surfaceThicknessSource[lev] =  new LevelData<FArrayBox>(levelDBL,   1, IntVect::Unit) ;
	  m_basalThicknessSource[lev] = new LevelData<FArrayBox>(levelDBL,   1, IntVect::Unit) ;
	  m_calvedIceThickness[lev] =  new LevelData<FArrayBox>(levelDBL,   1, IntVect::Unit) ;
	  m_removedIceThickness[lev] =  new LevelData<FArrayBox>(levelDBL,   1, IntVect::Unit) ;
	  m_addedIceThickness[lev] =  new LevelData<FArrayBox>(levelDBL,   1, IntVect::Unit) ;
	  m_melangeThickness[lev] =  new LevelData<FArrayBox>(levelDBL,   1, IntVect::Unit) ;
	  m_recordThickness[lev] =  new LevelData<FArrayBox>(levelDBL,   1, IntVect::Unit) ;
	  m_deltaTopography[lev] =  new LevelData<FArrayBox>(levelDBL,   1, IntVect::Zero) ;
	  m_divThicknessFlux[lev] =  new LevelData<FArrayBox>(levelDBL,   1, IntVect::Zero) ;
	  m_diffusivity[lev] = new LevelData<FluxBox>(levelDBL, 1, IntVect::Zero);

          // read this level's data
          LevelData<FArrayBox>& old_thickness = *m_old_thickness[lev];  
	  LevelData<FArrayBox> tmpThickness;
	  tmpThickness.define(old_thickness);

          int dataStatus = read<FArrayBox>(a_handle, tmpThickness, "thicknessData", levelDBL);
	  for (DataIterator dit(levelDBL);dit.ok();++dit)
	    {
	      old_thickness[dit].copy(tmpThickness[dit]);
	    }

          if (dataStatus != 0)
            {
              MayDay::Error("checkpoint file does not contain thickness data");
            }

	  LevelData<FArrayBox> bedHeight;
	  bedHeight.define(old_thickness);
	  dataStatus = read<FArrayBox>(a_handle, bedHeight, "bedHeightData", levelDBL);

	  if (dataStatus != 0)
            {
              MayDay::Error("checkpoint file does not contain bed height data");
            }
	  
	  LevelData<FArrayBox> deltaBedHeight;
	  deltaBedHeight.define(old_thickness);
	  
	  if (containsDeltaBedHeight)
	    {
	      dataStatus = read<FArrayBox>(a_handle,deltaBedHeight,"deltaBedHeightData",levelDBL);
	      if (dataStatus != 0)
		{
		  MayDay::Error("checkpoint file does not contain delta bed height data");
		}
	    }
	  else
	    {
	      for (DataIterator dit = deltaBedHeight.disjointBoxLayout(); dit.ok();++dit)
		{
		  deltaBedHeight[dit].setVal(0.0);
		}
	    }

	  LevelData<FArrayBox> melangeThck;
	  melangeThck.define(old_thickness);
	  
	  if (containsAccumCalvedIceThck)
	    {
	      dataStatus = read<FArrayBox>(a_handle,melangeThck,"melangeThckData",levelDBL);
	      if (dataStatus != 0)
		{
		  MayDay::Error("checkpoint file does not contain accumulated calved ice thickess data");
		}
	    }
	  else
	    {
	      for (DataIterator dit = melangeThck.disjointBoxLayout(); dit.ok();++dit)
		{
		  melangeThck[dit].setVal(0.0);
		}
	    }

	  //having read thickness and base data, we can define
          //the co-ordinate system 
	  RealVect dx = m_amrDx[lev]*RealVect::Unit;
          m_vect_coordSys[lev] = RefCountedPtr<LevelSigmaCS >
            (new LevelSigmaCS(m_amrGrids[lev], dx, sigmaCSGhost));
	  m_vect_coordSys[lev]->setIceDensity(m_iceDensity);
	  m_vect_coordSys[lev]->setWaterDensity(m_seaWaterDensity);
	  m_vect_coordSys[lev]->setGravity(m_gravity);
#if BISICLES_Z == BISICLES_LAYERED
	  m_vect_coordSys[lev]->setFaceSigma(sigma);
#endif
          LevelSigmaCS& levelCS = *m_vect_coordSys[lev];
          LevelData<FArrayBox>& levelH = levelCS.getH();

          DataIterator dit = levelH.dataIterator();
          for (dit.begin(); dit.ok(); ++dit)
            {
              levelH[dit].copy((*m_old_thickness[lev])[dit]);
            }
          levelCS.setTopography(bedHeight);
	  if (deltaBedHeight.isDefined())
	    {
	      for (dit.begin(); dit.ok(); ++dit)
		{
		  (*m_deltaTopography[lev])[dit].copy(deltaBedHeight[dit]);
		} 
	    }
	  if (melangeThck.isDefined())
	    {
	      for (dit.begin(); dit.ok(); ++dit)
		{
		  (*m_melangeThickness[lev])[dit].copy(melangeThck[dit]);
		} 
	    }
	  {
	    LevelSigmaCS* crseCoords = (lev > 0)?&(*m_vect_coordSys[lev-1]):NULL;
	    int refRatio = (lev > 0)?m_refinement_ratios[lev-1]:-1;
	    levelCS.recomputeGeometry(crseCoords, refRatio);
	  }
          LevelData<FArrayBox>& velData = *m_velocity[lev];
	  dataStatus = read<FArrayBox>(a_handle,
                                       velData,
                                       "velocityData",
				       levelDBL);
	  m_velocitySolveInitialResidualNorm = 1.0e+6; //\todo fix this
	  // dit.reset();
	  // for (dit.begin(); dit.ok(); ++dit)
	  //   {
	  //     (*m_velocity[lev])[dit].setVal(0.0);
	  //   }
	  
	  //this check doesn't work, because HDF5::SetGroup attempts
	  //to create a group if it doesn't exist. And since the file has been
	  // opened readonly, the previous call generates an exception
 			       
          if (dataStatus != 0)
            {
	      MayDay::Error("checkpoint file does not contain velocity data");
	      
            }


	  if (containsIceFrac)
	    {
	      LevelData<FArrayBox>& iceFracData = *m_iceFrac[lev];
	      dataStatus = read<FArrayBox>(a_handle,
					   iceFracData,
					   "iceFracData",
				       levelDBL);
          
	      /// note that although this check appears to work, it makes a mess of a_handle and the next lot of data are not read...
	      if (dataStatus != 0)
		{
		  MayDay::Warning("checkpoint file does not contain ice fraction data -- initializing based on current ice thicknesses"); 
		  const LevelData<FArrayBox>& levelThickness = m_vect_coordSys[lev]->getH();
		  setIceFrac(levelThickness, lev);
		} // end if no ice fraction in data
	      else
		{
		  // ensure that ice fraction is set to zero where there's no ice
		  updateIceFrac(m_vect_coordSys[lev]->getH(), lev);
		}
	    } 
	  else
	    {
	      MayDay::Warning("checkpoint file does not contain ice fraction data -- initializing based on current ice thicknesses"); 
	      const LevelData<FArrayBox>& levelThickness = m_vect_coordSys[lev]->getH();
	      setIceFrac(levelThickness, lev);
	    }

	  if (containsBasalFriction)
	    {
	      dataStatus = read<FArrayBox>(a_handle, *m_velBasalC[lev],
					   "basalFrictionData", levelDBL);
	    }
	  else
	    {
	      MayDay::Warning("checkpoint file does not basal friction coefficient data"); 
	    }

	  if (containsMuCoef)
	    {
	      dataStatus = read<FArrayBox>(a_handle, *m_cellMuCoef[lev],
					   "muCoefData", levelDBL);
	    }
	  else
	    {
	      MayDay::Warning("checkpoint file does not mu coefficient data"); 
	    }

	  {
	    // read internal energy , or read temperature and convert to internal energy
 	    std::string dataName, sDataName, bDataName;

	    if (containsInternalEnergy)
	      {
		dataName = "internalEnergyData";
		sDataName = "sInternalEnergyData";
		bDataName = "bInternalEnergyData";
	      }
	    else if (containsTemperature)
	      {
		dataName = "temperatureData";
		sDataName = "sTemperatureData";
		bDataName = "bTemperatureData";	
	      }
	    else
	      {
		MayDay::Error("checkpoint file does not contain internal energy or temperature data"); 
	      }
	    

	    LevelData<FArrayBox>& internalEnergyData = *m_internalEnergy[lev];
	    dataStatus = read<FArrayBox>(a_handle, internalEnergyData, dataName,levelDBL);
	    if (dataStatus != 0)
	      {
		MayDay::Error("checkpoint file does not contain internal energy data"); 
	      }

#if BISICLES_Z == BISICLES_LAYERED	  
	    LevelData<FArrayBox>& sInternalEnergyData = *m_sInternalEnergy[lev];
	    dataStatus = read<FArrayBox>(a_handle,  sInternalEnergyData, sDataName,levelDBL);	  
	    if (dataStatus != 0)
	      {
		MayDay::Error("checkpoint file does not contain surface internal energy data"); 
	      }
	    
	    LevelData<FArrayBox>& bInternalEnergyData = *m_bInternalEnergy[lev];
	    dataStatus = read<FArrayBox>(a_handle, bInternalEnergyData, bDataName, levelDBL);	  
	    if (dataStatus != 0)
	      {
		MayDay::Error("checkpoint file does not contain basal internal energy data"); 
	      }
#endif
	  
	    if (containsTemperature)
	      {
		// need to covert tempearture to internal energy
		for (DataIterator dit(levelDBL);dit.ok();++dit)
		  {
		    FArrayBox& E = (*m_internalEnergy[lev])[dit];
		    FArrayBox T(E.box(),E.nComp()); T.copy(E);
		    IceThermodynamics::composeInternalEnergy(E,T,E.box() );
#if BISICLES_Z == BISICLES_LAYERED
		    FArrayBox& sE = (*m_sInternalEnergy[lev])[dit];
		    FArrayBox sT(sE.box(),sE.nComp()); sT.copy(sE);
		    IceThermodynamics::composeInternalEnergy(sE,sT,sE.box(),false );
		    // it is possible for sT to be meaningless, so avoid test 
		    FArrayBox& bE = (*m_bInternalEnergy[lev])[dit];
		    FArrayBox bT(bE.box(),bE.nComp()); bT.copy(bE);
		    IceThermodynamics::composeInternalEnergy(bE,bT,bE.box(),false); 
		    // it is possible for bT to be meaningless, so avoid test 
#endif	    
		  }
	      }
	    CH_assert(m_internalEnergy[lev]->nComp() == sigma.size() -1);
	  }

	  //allow observers to read from the checkpoint
	  for (int i = 0; i < m_observers.size(); i++)
	    {
	      m_observers[i]->readCheckData(a_handle, header,  lev, levelDBL);
	    }


	} // end if this level is defined
    } // end loop over levels                                    
          
  // do we need to close the handle?
 
  //this is just to make sure the diffusivity is computed
  //(so I should improve that)
  defineSolver();
  m_doInitialVelSolve = false; // since we have just read the velocity field
  m_doInitialVelGuess = false; // ditto

  
  if (dynamic_cast<InverseIceVelocitySolver*>(m_velSolver))
    {
      //special case for inverse problems : in most cases the basal friction and
      //mu coeffcient data read from the checkpoint file will be overwritten
      //in solveVelocityField(). Here we want them to be input data 
      if (containsBasalFriction)
	{	
	  if (m_basalFrictionPtr) delete m_basalFrictionPtr; 
	  m_basalFrictionPtr = InverseIceVelocitySolver::basalFriction(m_velBasalC, refRatios(), dx(0));
	}
      
      if (containsMuCoef)
	{ 	
	  if (m_muCoefficientPtr) delete m_muCoefficientPtr;
	  m_muCoefficientPtr = InverseIceVelocitySolver::muCoefficient(m_cellMuCoef, refRatios(), dx(0));
	}
    }

  if (s_verbosity > 3) 
    {
      pout() << "AmrIce::readCheckPointFile solveVelocityField() " << endl;
    }
  solveVelocityField();
  m_doInitialVelSolve = true;

#endif
  
}

/// set up for restart
void 
AmrIce::restart(const string& a_restart_file)
{
  if (s_verbosity > 3) 
    { 
      pout() << "AmrIce::restart" << endl;
    }

  HDF5Handle handle(a_restart_file, HDF5Handle::OPEN_RDONLY);
  // first read in data from checkpoint file
  readCheckpointFile(handle);
  handle.close();
  // don't think I need to do anything else, do I?


}
#endif


void AmrIce::helmholtzSolve
(Vector<LevelData<FArrayBox>* >& a_phi,
 const Vector<LevelData<FArrayBox>* >& a_rhs,
 Real a_alpha, Real a_beta) const
{

  // AMRPoissonOp supports only one component of phi
  // if m_finest_level > 0 (its LevelFluxRegisters are
  // defined with only one component, so we will do
  // one component at a time. should try to avoid some
  // of the rhs copies...
  for (int icomp = 0; icomp < a_phi[0]->nComp(); ++icomp)
    {
      
      //make a copy of a_phi with one ghost cell
      Vector<LevelData<FArrayBox>* > phi(m_finest_level + 1, NULL);
      Vector<LevelData<FArrayBox>* > rhs(m_finest_level + 1, NULL);
      Vector<DisjointBoxLayout> grids(m_finest_level + 1);
      for (int lev=0; lev < m_finest_level + 1; ++lev)
	{
	  grids[lev] = m_amrGrids[lev];

	  const LevelData<FArrayBox>& levelPhi = *a_phi[lev]; 
	  phi[lev] = new LevelData<FArrayBox>(m_amrGrids[lev], 
					      1, IntVect::Unit);
	  levelPhi.copyTo(Interval(icomp,icomp),*phi[lev], Interval(0,0));
	  phi[lev]->exchange();
	  

	  const LevelData<FArrayBox>& levelRhs = *a_rhs[lev];
	  rhs[lev] = new LevelData<FArrayBox>(m_amrGrids[lev], 
					      1, IntVect::Zero);
	  levelRhs.copyTo(Interval(icomp,icomp),*rhs[lev], Interval(0,0));
	  rhs[lev]->exchange();
      
	}


      //Natural boundary conditions
      BCHolder bc(ConstDiriNeumBC(IntVect::Zero, RealVect::Zero,
				  IntVect::Zero, RealVect::Zero));
      
      
      AMRPoissonOpFactory opf;
      opf.define(m_amrDomains[0],  grids , m_refinement_ratios,
		 m_amrDx[0], bc, a_alpha, -a_beta );
      
      AMRMultiGrid<LevelData<FArrayBox> > mgSolver;
      BiCGStabSolver<LevelData<FArrayBox> > bottomSolver;
      mgSolver.define(m_amrDomains[0], opf, &bottomSolver, m_finest_level+1);
      mgSolver.m_eps = TINY_NORM;
      mgSolver.m_normThresh = TINY_NORM;
      mgSolver.m_iterMax = 8;
      int numMGSmooth = 4;
      mgSolver.m_pre = numMGSmooth;
      mgSolver.m_post = numMGSmooth;
      mgSolver.m_bottom = numMGSmooth;
      mgSolver.m_verbosity = s_verbosity - 1;
      
      mgSolver.solve(phi, rhs, m_finest_level, 0,  false);
      
      for (int lev=0; lev < m_finest_level + 1; ++lev)
	{
	  LevelData<FArrayBox>& levelPhi = *a_phi[lev];
	  phi[lev]->copyTo(Interval(0,0), levelPhi, Interval(icomp, icomp));
	  
	  if (phi[lev] != NULL){
	    delete phi[lev];
	    phi[lev] = NULL;
	  }
	}
    }
}


void AmrIce::helmholtzSolve
(Vector<LevelData<FArrayBox>* >& a_phi, Real a_alpha, Real a_beta) const
{
  
  Vector<LevelData<FArrayBox>* > rhs(m_finest_level + 1, NULL);
 
  for (int lev=0; lev < m_finest_level + 1; ++lev)
    {
      const LevelData<FArrayBox>& levelPhi = *a_phi[lev]; 
      rhs[lev] = new LevelData<FArrayBox>
	(m_amrGrids[lev], levelPhi.nComp(), IntVect::Zero);
      levelPhi.copyTo(*rhs[lev]);
    }
 
  helmholtzSolve(a_phi, rhs, a_alpha, a_beta);

  for (int lev=0; lev < m_finest_level + 1; ++lev)
    {
      if (rhs[lev] != NULL){
	delete rhs[lev];
	rhs[lev] = NULL;
      }
    }

}


#if BISICLES_Z == BISICLES_LAYERED

/// update the flow law coefficient A
void AmrIce::computeA(Vector<LevelData<FArrayBox>* >& a_A, 
		      Vector<LevelData<FArrayBox>* >& a_sA,
		      Vector<LevelData<FArrayBox>* >& a_bA,
		      const Vector<LevelData<FArrayBox>* >& a_internalEnergy, 
		      const Vector<LevelData<FArrayBox>* >& a_sInternalEnergy,
		      const Vector<LevelData<FArrayBox>* >& a_bInternalEnergy,
		      const Vector<RefCountedPtr<LevelSigmaCS> >& a_coordSys) const
		      
{
  if (s_verbosity > 0)
    {
      pout() <<  "AmrIce::computeA" <<  std::endl;
    }

  //for now, throw a_A etc away and recompute
  for (int lev = 0; lev < a_A.size(); ++lev)
    {
      if (a_A[lev] != NULL)
	{
	  delete a_A[lev]; a_A[lev] = NULL;
	}
      
      if (a_sA[lev] != NULL)
	{
	  delete a_sA[lev]; a_sA[lev] = NULL;
	}
      if (a_bA[lev] != NULL)
	{
	  delete a_bA[lev]; a_bA[lev] = NULL;
	}
      
    }
  a_A.resize(m_finest_level+1,NULL);
  a_sA.resize(m_finest_level+1,NULL);
  a_bA.resize(m_finest_level+1,NULL);
	
  for (int lev = 0; lev <= m_finest_level; ++lev)
    {
      const LevelSigmaCS& levelCoords = *a_coordSys[lev];
      
      const Vector<Real>& sigma = levelCoords.getSigma();
      a_A[lev] = new LevelData<FArrayBox>(m_amrGrids[lev], m_nLayers, IntVect::Unit);
      IceUtility::computeA(*a_A[lev], sigma, levelCoords,  m_rateFactor, *a_internalEnergy[lev] );
      
      Vector<Real> sSigma(1,0.0);
      a_sA[lev] = new LevelData<FArrayBox>(m_amrGrids[lev],1, IntVect::Unit);
      IceUtility::computeA(*a_sA[lev], sSigma, levelCoords,  
			   m_rateFactor, *a_sInternalEnergy[lev]);
      Vector<Real> bSigma(1,1.0);
      a_bA[lev] = new LevelData<FArrayBox>(m_amrGrids[lev],1, IntVect::Unit);
      IceUtility::computeA(*a_bA[lev], bSigma, levelCoords,  
			   m_rateFactor, *a_bInternalEnergy[lev]);
   
    }//end loop over AMR levels

  if (s_verbosity > 0)
    {
      Real Amin = computeMin(a_A,  m_refinement_ratios, Interval(0,a_A[0]->nComp()-1));
      Real Amax = computeMax(a_A,  m_refinement_ratios, Interval(0,a_A[0]->nComp()-1));
      pout() << Amin << " <= A(x,y,sigma) <= " << Amax << std::endl;

    }
}


//compute the face- and layer- centered internal energy (a_layerEH_half)
//and thickness (a_layerH_half) at time a_time + 1/2 * a_dt
void AmrIce::computeInternalEnergyHalf(Vector<LevelData<FluxBox>* >& a_layerEH_half,
				       Vector<LevelData<FluxBox>* >& a_layerH_half,
				       const Vector<LevelData<FluxBox>* >& a_layerXYFaceXYVel, 
				       const Real a_dt, const Real a_time)
{

  //delete and re-create storage for a_layerEH_half and a_layerH_half.
  for (int lev = 0 ; lev <= m_finest_level; lev++)
    {
      
      if (a_layerEH_half[lev] != NULL)
	delete(a_layerEH_half[lev]);

      a_layerEH_half[lev] = new LevelData<FluxBox>(m_amrGrids[lev], 
						   m_internalEnergy[lev]->nComp(), 
						   IntVect::Unit);
      if (a_layerH_half[lev] != NULL)
	delete(a_layerH_half[lev]);
      
      a_layerH_half[lev] = new LevelData<FluxBox>(m_amrGrids[lev], 
						  m_internalEnergy[lev]->nComp(), 
						  IntVect::Unit);
    }


  //assume the ghost regions of m_internalEnergy are not correct
  for (int lev = 0 ; lev <= m_finest_level; lev++)
    {
      if (lev > 0)
	{
	  PiecewiseLinearFillPatch pwl(m_amrGrids[lev],
				       m_amrGrids[lev-1],
				       m_internalEnergy[lev]->nComp(),
				       m_amrDomains[lev-1],
				       m_refinement_ratios[lev-1],
				       m_internalEnergy[lev]->ghostVect()[0]);
	  pwl.fillInterp(*m_internalEnergy[lev],*m_internalEnergy[lev-1],
			 *m_internalEnergy[lev-1],1.0,0,0,m_internalEnergy[lev]->nComp());
	}
      m_internalEnergy[lev]->exchange();
    }
   
  for (int lev = 0 ; lev <= m_finest_level; lev++)
    {
      //in the 2D case (with poor man's multidim) this
      //is a little pained using AdvectPhysics, but for the time being
      //we need to construct a single component thisLayerEH_Half for each layer, 
      //given a internalEnergy and horizontal velocity and then copy it into 
      //the multicomponent EH_half[lev]
     
      // PatchGodunov object for layer thickness/energy advection
      PatchGodunov patchGodunov;
      {
	int normalPredOrder = 1;
	bool useFourthOrderSlopes = false;
	bool usePrimLimiting = false;
	bool useCharLimiting = false;
	bool useFlattening = false;
	bool useArtificialViscosity = false;
	Real artificialViscosity = 0.0;
	AdvectPhysics advectPhys;
	advectPhys.setPhysIBC(m_internalEnergyIBCPtr);

	patchGodunov.define(m_amrDomains[lev], m_amrDx[lev],
			    &advectPhys, normalPredOrder,
			    useFourthOrderSlopes,usePrimLimiting,
			    useCharLimiting,useFlattening,
			    useArtificialViscosity,artificialViscosity);
	patchGodunov.setCurrentTime(m_time);
      }
      
      AdvectPhysics* advectPhysPtr = dynamic_cast<AdvectPhysics*>(patchGodunov.getGodunovPhysicsPtr());
      if (advectPhysPtr == NULL)
	{
	  MayDay::Error("AmrIce::computeInternalEnergyHalf -- unable to upcast GodunovPhysics to AdvectPhysics");
	}

      const LevelData<FArrayBox>& levelInternalEnergy = *m_internalEnergy[lev]; 
      const LevelData<FArrayBox>& levelOldThickness = *m_old_thickness[lev]; 
      const LevelData<FluxBox>& levelLayerXYFaceXYVel = *a_layerXYFaceXYVel[lev]; 
      const DisjointBoxLayout& levelGrids = m_amrGrids[lev];

      for (int layer = 0; layer < m_nLayers; ++layer)
	{
	  for (DataIterator dit(levelGrids); dit.ok(); ++dit)
	    {
	      const Box& box = levelInternalEnergy[dit].box(); // grid box plus ghost cells
	      
	      FluxBox layerXYFaceXYVel(box,1);
	      layerXYFaceXYVel.setVal(0.0);
	      for (int dir = 0; dir < SpaceDim; ++dir){
		layerXYFaceXYVel[dir].copy(levelLayerXYFaceXYVel[dit][dir],layer,0,1);
		Box faceBox = levelGrids[dit].surroundingNodes(dir);
		CH_assert(layerXYFaceXYVel[dir].norm(faceBox,0) < HUGE_NORM);
	      }

	      FArrayBox layerCellXYVel(box,SpaceDim);
	      EdgeToCell(layerXYFaceXYVel,layerCellXYVel);

	      //\todo compute bulk heat sources
	      FArrayBox heatSource(levelGrids[dit], 1);
	      heatSource.setVal(0.0);

	      patchGodunov.setCurrentBox(levelGrids[dit]);
	      advectPhysPtr->setVelocities(&layerCellXYVel,&layerXYFaceXYVel);

	      FArrayBox WGdnv(box,1);

	      //HE at half time and cell faces
	      WGdnv.copy(levelInternalEnergy[dit],layer,0,1);
	      WGdnv *= levelOldThickness[dit];
	      Box grownBox = levelGrids[dit];
	      grownBox.grow(1);
	      FluxBox HEhalf(grownBox,1);
	      patchGodunov.computeWHalf(HEhalf,
					WGdnv,
					heatSource,
					a_dt,
					levelGrids[dit]);
	      for (int dir = 0; dir < SpaceDim; ++dir)
		{
		  Box faceBox(levelGrids[dit]);
		  faceBox.surroundingNodes(dir);
		  CH_assert(HEhalf[dir].norm(faceBox,0) < HUGE_NORM);
		  (*a_layerEH_half[lev])[dit][dir].copy(HEhalf[dir],0,layer,1);
		}
	      
	      //H at half time and cell faces
	      WGdnv.copy(levelOldThickness[dit]);
	      FluxBox Hhalf(grownBox,1);
	      //\todo compute layer thickness sources
	      FArrayBox HSource(levelGrids[dit], 1);
	      HSource.setVal(0.0);
	      patchGodunov.computeWHalf(Hhalf,
					WGdnv,
					HSource,
					a_dt,
					levelGrids[dit]);
	      for (int dir = 0; dir < SpaceDim; ++dir)
		{
		  Box faceBox(levelGrids[dit]);
		  faceBox.surroundingNodes(dir);
		  CH_assert(Hhalf[dir].norm(faceBox,0) < HUGE_NORM);
		  (*a_layerH_half[lev])[dit][dir].copy(Hhalf[dir],0,layer,1);
		}
	      
	    }
	  
	}
    }
      
  // coarse average new EH-Half to covered regions
  for (int lev=m_finest_level; lev>0; lev--)
    {
      CoarseAverageFace faceAverager(m_amrGrids[lev],a_layerEH_half[lev]->nComp(), m_refinement_ratios[lev-1]);
      faceAverager.averageToCoarse(*a_layerEH_half[lev-1], *a_layerEH_half[lev]);
      faceAverager.averageToCoarse(*a_layerH_half[lev-1], *a_layerH_half[lev]);
    }

}

void AmrIce::updateInternalEnergy(Vector<LevelData<FluxBox>* >& a_layerEH_half, 
				  Vector<LevelData<FluxBox>* >& a_layerH_half,
				  const Vector<LevelData<FluxBox>* >& a_layerXYFaceXYVel,
				  const Vector<LevelData<FArrayBox>* >& a_layerSFaceXYVel,
				  const Real a_dt, const Real a_time,
				  Vector<RefCountedPtr<LevelSigmaCS> >& a_coordSysNew,
				  Vector<RefCountedPtr<LevelSigmaCS> >& a_coordSysOld,
				  const Vector<LevelData<FArrayBox>*>& a_surfaceThicknessSource,
				  const Vector<LevelData<FArrayBox>*>& a_basalThicknessSource)
{

  //update the internalEnergy fields, 2D case
  Vector<LevelData<FluxBox>* > vectLayerFluxes(m_finest_level+1, NULL);
  Vector<LevelData<FluxBox>* > vectLayerThicknessFluxes(m_finest_level+1, NULL);
  Vector<LevelData<FArrayBox>* > vectUSigma(m_finest_level+1, NULL);
  Vector<LevelData<FArrayBox>* > vectDivUHxy(m_finest_level+1, NULL);

  for (int lev=0; lev<=m_finest_level; lev++)
    {
      LevelData<FluxBox>& levelXYFaceXYVel = *a_layerXYFaceXYVel[lev];
      LevelData<FluxBox>& levelFaceEH = *a_layerEH_half[lev];
      LevelData<FluxBox>& levelFaceH = *a_layerH_half[lev];
      IntVect ghostVect = IntVect::Unit;//CoarseAverageFace requires a ghost cell

      vectUSigma[lev] = new LevelData<FArrayBox>
	(m_amrGrids[lev], m_nLayers + 1 , IntVect::Zero);
       
      vectDivUHxy[lev] = new LevelData<FArrayBox>
	(m_amrGrids[lev], m_nLayers + 1 , ghostVect);
     
      vectLayerFluxes[lev] = new LevelData<FluxBox>
	(m_amrGrids[lev], levelXYFaceXYVel.nComp() , ghostVect);

      vectLayerThicknessFluxes[lev] = new LevelData<FluxBox>
	(m_amrGrids[lev], levelXYFaceXYVel.nComp() , ghostVect);

      const DisjointBoxLayout& levelGrids = m_amrGrids[lev];
      for (DataIterator dit(levelGrids); dit.ok(); ++dit)
	{
	  FluxBox& faceVel = levelXYFaceXYVel[dit];
	  FluxBox& faceEH = levelFaceEH[dit];
	  FluxBox& faceH = levelFaceH[dit];
	  FluxBox& flux = (*vectLayerFluxes[lev])[dit];
	  FluxBox& thicknessFlux = (*vectLayerThicknessFluxes[lev])[dit];

	  const Box& gridBox = levelGrids[dit];
	  for (int dir=0; dir<SpaceDim; dir++)
	    {
	      Box faceBox(gridBox);
	      faceBox.surroundingNodes(dir);
	      flux[dir].copy(faceEH[dir], faceBox);
	      flux[dir].mult(faceVel[dir], faceBox, 0, 0, faceVel[dir].nComp());

	      

	      thicknessFlux[dir].copy(faceH[dir],faceBox);
	      thicknessFlux[dir].mult(faceVel[dir], faceBox, 0, 0, faceVel[dir].nComp());
	        
	      // CH_assert(flux[dir].norm(faceBox,0,0,flux[dir].nComp()) < HUGE_NORM);
	      // CH_assert(thicknessFlux[dir].norm(faceBox,0,0,thicknessFlux[dir].nComp()) < HUGE_NORM);
		
	    }
	}
    }
  // average fine fluxes down to coarse levels
  for (int lev=m_finest_level; lev>0; lev--)
    {
      CoarseAverageFace faceAverager(m_amrGrids[lev],
				     vectLayerFluxes[lev]->nComp(), m_refinement_ratios[lev-1]);
      faceAverager.averageToCoarse(*vectLayerFluxes[lev-1], *vectLayerFluxes[lev]);
      faceAverager.averageToCoarse(*vectLayerThicknessFluxes[lev-1], *vectLayerThicknessFluxes[lev]);
    }
 

     
  //vertical and cross-layer velocity components (u[z] and u^sigma)
  for (int lev=0; lev <= m_finest_level; lev++)
    {
      DisjointBoxLayout& levelGrids = m_amrGrids[lev];
     
      LevelSigmaCS& levelCoordsNew = *(a_coordSysNew[lev]);
      LevelSigmaCS& levelCoordsOld = *(a_coordSysOld[lev]);
      const Vector<Real>& dSigma = levelCoordsNew.getDSigma();

      LevelData<FArrayBox> levelGradHNew(m_amrGrids[lev], SpaceDim, IntVect::Zero);
      computeCCDerivatives(levelGradHNew, levelCoordsNew.getH(), levelCoordsNew,
			   Interval(0,0),Interval(0,SpaceDim-1));
      LevelData<FArrayBox> levelGradHOld(m_amrGrids[lev], SpaceDim, IntVect::Zero);
      computeCCDerivatives(levelGradHOld, levelCoordsOld.getH(), levelCoordsOld,
			   Interval(0,0),Interval(0,SpaceDim-1));


      for (DataIterator dit(levelGrids); dit.ok(); ++dit)
	{
	  const Box& box = levelGrids[dit];
	   
	  // this copy perhaps indicates layer should run faster than
	  // dir in sFaceXYVel, but for now ...
	  const FArrayBox& sFaceXYVel = (*a_layerSFaceXYVel[lev])[dit];
	  FArrayBox uX(box, m_nLayers+1);
	  FArrayBox uY(box, m_nLayers+1);
	  
	  for (int l = 0; l < m_nLayers+1; l++)
	    {
	      uX.copy(sFaceXYVel, l*SpaceDim, l);
	      uY.copy(sFaceXYVel, l*SpaceDim + 1, l);
	    }

	  FArrayBox& oldH = levelCoordsOld.getH()[dit];
	  FArrayBox& newH = levelCoordsNew.getH()[dit];
	  //cell centered thickness at t + dt/2
	  FArrayBox Hhalf(box, 1);
	  Hhalf.copy(oldH);
	  Hhalf.plus(newH);
	  Hhalf *= 0.5;

	  //cell centered grad(thickness) at t + dt/2
	  FArrayBox gradH(box, SpaceDim);
	  gradH.copy(levelGradHNew[dit]);
	  gradH.plus(levelGradHOld[dit]);
	  gradH*=0.5;
	  //cell centered grad(surface) at t + dt/2
	  FArrayBox gradS(box, SpaceDim);
	  gradS.copy(levelCoordsOld.getGradSurface()[dit]);
	  gradS.plus(levelCoordsNew.getGradSurface()[dit]);
	  gradS*=0.5;
	  //horizontal contribution to div(Hu) at cell centres, 
	  // viz d(Hu_x)/dx' + d(Hu_y)/dy'
	  FArrayBox divUHxy(box, m_nLayers);
	  {
	    divUHxy.setVal(0.0);
	    
	    const RealVect& dx = levelCoordsNew.dx(); 
	    for (int dir =0; dir < SpaceDim; dir++)
	      {
		const FArrayBox& uH = (*vectLayerThicknessFluxes[lev])[dit][dir];
		FORT_DIVERGENCE(CHF_CONST_FRA(uH),
				CHF_FRA(divUHxy),
				CHF_BOX(box),
				CHF_CONST_REAL(dx[dir]),
				CHF_INT(dir));
	      }
	    
	  }

	  //dH / dt
	  FArrayBox dHdt(box,1);  
	  dHdt.copy(newH);
	  dHdt.plus(oldH,-1.0,0,0,1);
	  dHdt *= 1.0/a_dt;
	    
	  //calculation of dS/dt assumes surface elevation is up to date
	  //in LevelSigmaCS
	  FArrayBox dSdt(box,1); 
	  dSdt.copy(levelCoordsNew.getSurfaceHeight()[dit]);
	  dSdt -= levelCoordsOld.getSurfaceHeight()[dit];
	  dSdt *= 1.0/a_dt;

	  //surface and basal thickness source
	  const FArrayBox& bts = (*a_basalThicknessSource[lev])[dit];
	  const FArrayBox& sts = (*a_surfaceThicknessSource[lev])[dit];
	  // z-component of velocity at layer faces
	  FArrayBox uZ(box,m_nLayers + 1); 
	  // sigma-componnet of velocity at layer faces
	  FArrayBox& uSigma = (*vectUSigma[lev])[dit]; 
	  // z-component of velocity at surface 
	  FArrayBox uZs(box, 1);
	  //divUHxy.setVal(0.0);
	  int nLayers = m_nLayers;
	  uSigma.setVal(0.0);

	  FORT_COMPUTEZVEL(CHF_FRA(uZ),
			   CHF_FRA1(uZs,0),
			   CHF_FRA(uSigma),
			   CHF_CONST_FRA(uX),
			   CHF_CONST_FRA(uY),
			   CHF_CONST_FRA(divUHxy),
			   CHF_CONST_VR(levelCoordsNew.getFaceSigma()),
			   CHF_CONST_VR(levelCoordsNew.getSigma()),
			   CHF_CONST_VR(dSigma),
			   CHF_CONST_FRA1(Hhalf,0),
			   CHF_CONST_FRA1(gradS,0), 
			   CHF_CONST_FRA1(gradH,0),
			   CHF_CONST_FRA1(gradS,1), 
			   CHF_CONST_FRA1(gradH,1),
			   CHF_CONST_FRA1(dSdt,0), 
			   CHF_CONST_FRA1(dHdt,0),
			   CHF_CONST_FRA1(sts,0),
			   CHF_CONST_FRA1(bts,0),
			   CHF_CONST_INT(nLayers),
			   CHF_BOX(box));

	  CH_assert(uSigma.norm(0) < HUGE_NORM);
	} //end compute vertical velocity loop over boxes

      vectUSigma[lev]->exchange();

    }//end compute vertical velocity loop over levels


  // compute rhs =a_dt *(H*dissipation - div(u H T)) and update solution
  for (int lev=0; lev <= m_finest_level; lev++)
    {
      DisjointBoxLayout& levelGrids = m_amrGrids[lev];
      LevelData<FluxBox>& levelFlux = *vectLayerFluxes[lev];
      LevelSigmaCS& levelCoordsOld = *(a_coordSysOld[lev]);
      LevelSigmaCS& levelCoordsNew = *(a_coordSysNew[lev]);
      const Vector<Real>& dSigma = levelCoordsNew.getDSigma();
      //caculate dissipation due to internal stresses
      LevelData<FArrayBox> dissipation(levelGrids,m_nLayers,IntVect::Zero);
      {
	LevelData<FArrayBox>* crseVelPtr = NULL;
	int nRefCrse = -1;
	if (lev > 0)
	  {
	    crseVelPtr = m_velocity[lev-1];
	    nRefCrse = m_refinement_ratios[lev-1];
	  }

	m_velocity[lev]->exchange();

	m_constitutiveRelation->computeDissipation
	  (dissipation,*m_velocity[lev],  crseVelPtr,
	   nRefCrse, *m_A[lev],
	   levelCoordsOld , m_amrDomains[lev], IntVect::Zero);
      }
      
      LevelData<FArrayBox>& surfaceHeatFlux = *m_sHeatFlux[lev];
      if (surfaceHeatBoundaryDirichlett())
	{
	  surfaceHeatBoundaryData().evaluate(*m_sInternalEnergy[lev], *this, lev, a_dt);
	  if (surfaceHeatBoundaryTemperature())
	    {
	      //convert surface temperature to internal energy
	      for (DataIterator dit(levelGrids); dit.ok(); ++dit)
		{
		  FArrayBox& E = (*m_sInternalEnergy[lev])[dit];
		  FArrayBox T(E.box(),1);
		  T.copy(E);
		  FArrayBox W(E.box(),1);
		  W.setVal(0.0);
		  IceThermodynamics::composeInternalEnergy(E, T, W, E.box());
		}
	    }
	}
      else
	{
	  m_surfaceHeatBoundaryDataPtr->evaluate(surfaceHeatFlux, *this, lev, a_dt);
	}
      
      LevelData<FArrayBox>& basalHeatFlux = *m_bHeatFlux[lev];
      basalHeatBoundaryData().evaluate(basalHeatFlux, *this, lev, a_dt);

      for (DataIterator dit(levelGrids); dit.ok(); ++dit)
	{
	  const Box& box = levelGrids[dit];
	  const FArrayBox& oldH = levelCoordsOld.getH()[dit];
	  const FArrayBox& newH = levelCoordsNew.getH()[dit];
	  FArrayBox& E = (*m_internalEnergy[lev])[dit];
	  FArrayBox& sT = (*m_sInternalEnergy[lev])[dit];	
	  FArrayBox& bT = (*m_bInternalEnergy[lev])[dit];
	  

	  // first, do the ordinary fluxes : if we just had
	  // horizontal advection and grad(H) = grad(S) = 0., 
	  // this would be the lot
	       
	  FArrayBox rhs(box, m_nLayers);
	  rhs.setVal(0.0);
	  for (int dir=0; dir<SpaceDim; dir++)
	    {
	      Real dx = levelCoordsOld.dx()[dir];              
	   
	      FORT_DIVERGENCE(CHF_CONST_FRA(levelFlux[dit][dir]),
			      CHF_FRA(rhs),
			      CHF_BOX(box),
			      CHF_CONST_REAL(dx),
			      CHF_INT(dir));
	     

	    }
	  for (int layer = 0; layer < dissipation.nComp(); ++layer)
	    {
	      dissipation[dit].mult(newH,0,layer,1);
	      
	    } 
	  dissipation[dit] /= levelCoordsOld.iceDensity();
	  rhs -= dissipation[dit]; 
	  rhs *= -a_dt;

	  //compute heat flux across base due to basal dissipation
	  FArrayBox basalDissipation(rhs.box(),1);
	  m_basalFrictionRelation->computeDissipation
	    (basalDissipation , (*m_velocity[lev])[dit] , (*m_velBasalC[lev])[dit],
	     levelCoordsOld , dit ,rhs.box());
	  

	  //add to user set (e.g geothermal) heat flux
	  basalHeatFlux[dit] += basalDissipation;

	  //zero heat flux outside grounded ice
	  for (BoxIterator bit(rhs.box());bit.ok();++bit)
	    {
	      const IntVect& iv = bit();
	      if (levelCoordsOld.getFloatingMask()[dit](iv) != GROUNDEDMASKVAL)
		{
		  basalHeatFlux[dit](iv) = 0.0;
		}
	    }
	  
	  //basalHeatFlux[dit] /= levelCoordsNew.iceDensity()); // scale conversion
	  FArrayBox scaledBasalHeatFlux(basalHeatFlux[dit].box(),basalHeatFlux[dit].nComp());
	  scaledBasalHeatFlux.copy(basalHeatFlux[dit]);
	  scaledBasalHeatFlux /=  levelCoordsNew.iceDensity();

	  //surfaceHeatFlux[dit] /= (levelCoordsNew.iceDensity()); // scale conversion
	  FArrayBox scaledSurfaceHeatFlux(surfaceHeatFlux[dit].box(),surfaceHeatFlux[dit].nComp());
	  scaledSurfaceHeatFlux.copy(surfaceHeatFlux[dit]);
	  scaledSurfaceHeatFlux /= levelCoordsNew.iceDensity();

	  //solve H(t+dt)E(t+dt) + vertical transport terms = H(t)E(t) - rhs(t+/dt)
	  //with either a Dirichlett or flux boundary condition at the upper surface and a flux condition at base
	 
	  Real halftime = time() + 0.5*a_dt;
	  int nLayers = m_nLayers;
	  const Real& rhoi = levelCoordsNew.iceDensity();
	  const Real& rhoo = levelCoordsNew.waterDensity();
	  const Real& gravity = levelCoordsNew.gravity();
	 
	  int surfaceTempDirichlett = surfaceHeatBoundaryDirichlett()?1:0;
	  
	  FORT_UPDATEINTERNALENERGY
	    (CHF_FRA(E), 
	     CHF_FRA1(sT,0), 
	     CHF_FRA1(bT,0),
	     CHF_CONST_FRA1(scaledSurfaceHeatFlux,0),
	     CHF_CONST_FRA1(scaledBasalHeatFlux,0),
	     CHF_CONST_FIA1(levelCoordsOld.getFloatingMask()[dit],0),
	     CHF_CONST_FIA1(levelCoordsNew.getFloatingMask()[dit],0),
	     CHF_CONST_FRA(rhs),
	     CHF_CONST_FRA1(oldH,0),
	     CHF_CONST_FRA1(newH,0),
	     CHF_CONST_FRA((*vectUSigma[lev])[dit]),
	     CHF_CONST_VR(levelCoordsOld.getFaceSigma()),
	     CHF_CONST_VR(dSigma),
	     CHF_CONST_REAL(halftime), 
	     CHF_CONST_REAL(a_dt),
	     CHF_CONST_REAL(rhoi),
	     CHF_CONST_REAL(rhoo),
	     CHF_CONST_REAL(gravity),
	     CHF_CONST_INT(nLayers),
	     CHF_CONST_INT(surfaceTempDirichlett),
	     CHF_BOX(box));

	  scaledBasalHeatFlux *= (levelCoordsNew.iceDensity());
	  basalHeatFlux[dit].copy(scaledBasalHeatFlux);
	  scaledSurfaceHeatFlux *= (levelCoordsNew.iceDensity());
	  surfaceHeatFlux[dit].copy(scaledSurfaceHeatFlux);	    
	} // end update internal energy loop over grids
    } // end update internal energy loop over levels

  //coarse average from finer levels & exchange
  for (int lev = m_finest_level; lev >= 0 ; --lev)
    {
      if (lev > 0)
	{
	  CoarseAverage avN(m_amrGrids[lev],
			    m_amrGrids[lev-1],
			    m_internalEnergy[lev]->nComp(),
			    m_refinement_ratios[lev-1], 
			    IntVect::Zero);
	  
	  
	  
	  avN.averageToCoarse(*m_internalEnergy[lev-1], *m_internalEnergy[lev]);
	
	  
	  CoarseAverage avOne(m_amrGrids[lev],m_amrGrids[lev-1],
			      1,m_refinement_ratios[lev-1], IntVect::Zero);
	  
	  avOne.averageToCoarse(*m_sInternalEnergy[lev-1], *m_sInternalEnergy[lev]);
	  avOne.averageToCoarse(*m_bInternalEnergy[lev-1], *m_bInternalEnergy[lev]);
	  avOne.averageToCoarse(*m_sHeatFlux[lev-1], *m_sHeatFlux[lev]);
	  avOne.averageToCoarse(*m_bHeatFlux[lev-1], *m_bHeatFlux[lev]);
	}
      
      m_internalEnergy[lev]->exchange();
      m_sInternalEnergy[lev]->exchange();
      m_bInternalEnergy[lev]->exchange();
      m_sHeatFlux[lev]->exchange();
      m_bHeatFlux[lev]->exchange();
    }
  
  for (int lev = 0; lev < vectLayerFluxes.size(); ++lev)
    {
      if (vectUSigma[lev] != NULL)
	{
	  delete vectUSigma[lev]; vectUSigma[lev] = NULL;
	}

      if (vectDivUHxy[lev] != NULL)
	{
	  delete  vectDivUHxy[lev]; vectDivUHxy[lev] = NULL;
	}

      if (vectLayerFluxes[lev] != NULL)
	{
	  delete vectLayerFluxes[lev];vectLayerFluxes[lev] = NULL;
	}

      if (vectLayerThicknessFluxes[lev] != NULL)
	{
	  delete vectLayerThicknessFluxes[lev];vectLayerThicknessFluxes[lev] = NULL;
	}

    }

  //finally, A is no longer valid 
  m_A_valid = false;
  //#endif

}
#endif

// DIAGNOSTICS
// Diagnostic routine -- compute discharge and calving flux
// Calving flux defined as flux of ice from the ice sheet directly into the ocean. 
void 
AmrIce::computeDischarge(const Vector<LevelData<FluxBox>* >& a_vectFluxes)
{

  Real sumDischarge = 0.0;
  Real sumGroundedDischarge = 0.0;
  Real sumDischargeToOcean = 0.0;

  Vector<LevelData<FArrayBox>* > vectDischarge ( m_finest_level+1, NULL);
  Vector<LevelData<FArrayBox>* > vectGroundedDischarge ( m_finest_level+1, NULL);
  Vector<LevelData<FArrayBox>* > vectDischargeToOcean ( m_finest_level+1, NULL);

  for (int lev=0; lev<=m_finest_level; lev++)
    {
      vectDischarge[lev] = new LevelData<FArrayBox>(m_amrGrids[lev],1,
							    IntVect::Zero);
      LevelData<FArrayBox>& levelDischarge = *vectDischarge[lev];
      vectGroundedDischarge[lev] = new LevelData<FArrayBox>(m_amrGrids[lev],1,
							    IntVect::Zero);
      LevelData<FArrayBox>& levelGroundedDischarge = *vectGroundedDischarge[lev];
      vectDischargeToOcean[lev] = new LevelData<FArrayBox>(m_amrGrids[lev],1,
							    IntVect::Zero);
      LevelData<FArrayBox>& levelDischargeToOcean = *vectDischargeToOcean[lev];

      const LevelData<FArrayBox>& levelThickness =  m_vect_coordSys[lev]->getH();
      const LevelData<BaseFab<int> >& levelMask = m_vect_coordSys[lev]->getFloatingMask();

      DataIterator dit=levelDischarge.dataIterator();
      for (dit.begin(); dit.ok(); ++dit)
	{
	  const FluxBox& vflux = (*a_vectFluxes[lev])[dit];
	  const BaseFab<int>& mask = levelMask[dit];
	  const FArrayBox& thk = levelThickness[dit];

	  FArrayBox& discharge = levelDischarge[dit];
	  FArrayBox& groundedDischarge = levelGroundedDischarge[dit];
	  FArrayBox& dischargeToOcean = levelDischargeToOcean[dit];
	  discharge.setVal(0.0);
	  groundedDischarge.setVal(0.0);
	  dischargeToOcean.setVal(0.0);

	  for (int dir=0; dir<SpaceDim; dir++)
	    {

	      const FArrayBox& flux = vflux[dir];
	      BoxIterator bit(discharge.box());
	      for (bit.begin(); bit.ok(); ++bit)
		{
		  IntVect iv = bit();
		  Real smallThk = 10.0;
		  if ((thk(iv) < smallThk) || (mask(iv) != GROUNDEDMASKVAL))
		    {
		      if (thk(iv + BASISV(dir)) > smallThk && (mask(iv + BASISV(dir)) == GROUNDEDMASKVAL) ) 
			{
			  groundedDischarge(iv) += -flux(iv + BASISV(dir)) / m_amrDx[lev];
			}
		      if (thk(iv - BASISV(dir)) > smallThk && (mask(iv - BASISV(dir)) == GROUNDEDMASKVAL) )
			{
			  groundedDischarge(iv) += flux(iv) / m_amrDx[lev];
			}

		    }		  
		  if (thk(iv) < tiny_thickness) 
		    {
		      if (thk(iv + BASISV(dir)) > tiny_thickness)
			{
			  discharge(iv) += -flux(iv + BASISV(dir)) / m_amrDx[lev];
			}
		      if (thk(iv - BASISV(dir)) > tiny_thickness)
			{
			  discharge(iv) += flux(iv) / m_amrDx[lev];
			}

		    }
		  if ((thk(iv) < tiny_thickness) && (mask(iv) == OPENSEAMASKVAL)) 
		    {
		      if (thk(iv + BASISV(dir)) > tiny_thickness)
			{
			  dischargeToOcean(iv) += -flux(iv + BASISV(dir)) / m_amrDx[lev];
			}
		      if (thk(iv - BASISV(dir)) > tiny_thickness)
			{
			  dischargeToOcean(iv) += flux(iv) / m_amrDx[lev];
			}

		    }

		}
	    } // end direction 
	}

    } // end loop over levels
  
  // now compute sum
    sumDischarge = computeSum(vectDischarge, m_refinement_ratios,
  				m_amrDx[0], Interval(0,0), 0);
    sumGroundedDischarge = computeSum(vectGroundedDischarge, m_refinement_ratios,
  				m_amrDx[0], Interval(0,0), 0);
    sumDischargeToOcean = computeSum(vectDischargeToOcean, m_refinement_ratios,
  				m_amrDx[0], Interval(0,0), 0);

  if (s_verbosity > 0) 
    {
      pout() << "Step " << m_cur_step << ", time = " << m_time << " ( " << time() << " ) "
	     << ": DischargeFromIceEdge = " << sumDischarge << " m3/y " << endl;

      pout() << "Step " << m_cur_step << ", time = " << m_time << " ( " << time() << " ) "
	     << ": DischargeFromGroundedIce = " << sumGroundedDischarge << " m3/y " << endl;
      pout() << "Step " << m_cur_step << ", time = " << m_time << " ( " << time() << " ) "
	     << ": DischargeToOcean = " << sumDischargeToOcean << " m3/y " << endl;


    }  

  // clean up temp storage
  for (int lev=0; lev<vectDischarge.size(); lev++)
    {
      if (vectDischarge[lev] != NULL)
	{
	  delete vectDischarge[lev];
	  vectDischarge[lev] = NULL;
	}
    }
  for (int lev=0; lev<vectGroundedDischarge.size(); lev++)
    {
      if (vectGroundedDischarge[lev] != NULL)
	{
	  delete vectGroundedDischarge[lev];
	  vectGroundedDischarge[lev] = NULL;
	}
    }
  for (int lev=0; lev<vectDischargeToOcean.size(); lev++)
    {
      if (vectDischargeToOcean[lev] != NULL)
	{
	  delete vectDischargeToOcean[lev];
	  vectDischargeToOcean[lev] = NULL;
	}
    }

}

/// diagnostic function -- integrates thickness over domain
Real
AmrIce::computeTotalIce() const
{
  Vector<LevelData<FArrayBox>* > thickness(m_finest_level+1, NULL);
  for (int lev=0; lev<=m_finest_level; lev++)
    {
      const LevelSigmaCS& levelCoords = *m_vect_coordSys[lev];
      // need a const_cast to make things all line up right
      // (but still essentially const)
      thickness[lev] = const_cast<LevelData<FArrayBox>* >(&levelCoords.getH());
    }

  Interval thicknessInt(0,0);
  Real totalIce = computeSum(thickness, m_refinement_ratios,
                             m_amrDx[0], thicknessInt, 0);


  return totalIce;

}

Real
AmrIce::computeVolumeAboveFlotation() const
{

  //Compute the total thickness above flotation
  Vector<LevelData<FArrayBox>* > thk(m_finest_level+1, NULL);
  for (int lev=0; lev <= m_finest_level ; lev++)
    {
      const LevelSigmaCS& levelCoords = *m_vect_coordSys[lev];
      // need a const_cast to make things all line up right
      // (but still essentially const)
      thk[lev] = const_cast<LevelData<FArrayBox>*>(&levelCoords.getThicknessOverFlotation());
    }
  Real VAF = computeSum(thk, m_refinement_ratios,m_amrDx[0], Interval(0,0), 0);
  return VAF;
}
Real AmrIce::computeTotalGroundedIce() const
{
  
  Real totalGroundedIce = 0;

  Vector<LevelData<FArrayBox>* > vectGroundedThickness(m_finest_level+1, NULL);

  for (int lev=0; lev<=m_finest_level; lev++)
    {
      const LevelData<FArrayBox>& levelThickness = m_vect_coordSys[lev]->getH();
      // temporary with only ungrounded ice
      vectGroundedThickness[lev] = new LevelData<FArrayBox>(m_amrGrids[lev],1,
							    IntVect::Zero);

      LevelData<FArrayBox>& levelGroundedThickness = *vectGroundedThickness[lev];
      // now copy thickness to       
      levelThickness.copyTo(levelGroundedThickness);

      const LevelData<BaseFab<int> >& levelMask = m_vect_coordSys[lev]->getFloatingMask();
      // now loop through and set to zero where we don't have grounded ice.
      // do this the slow way, for now
      DataIterator dit=levelGroundedThickness.dataIterator();
      for (dit.begin(); dit.ok(); ++dit)
	{
	  const BaseFab<int>& thisMask = levelMask[dit];
	  FArrayBox& thisThick = levelGroundedThickness[dit];
	  BoxIterator bit(thisThick.box());
	  for (bit.begin(); bit.ok(); ++bit)
	    {
	      IntVect iv = bit();
	      if (thisMask(iv,0) != GROUNDEDMASKVAL)
		{
		  thisThick(iv,0) = 0.0;
		}
	    }
	}
    

    }

  // now compute sum
  Interval thicknessInt(0,0);
  totalGroundedIce = computeSum(vectGroundedThickness, m_refinement_ratios,
				m_amrDx[0], thicknessInt, 0);

  
  // clean up temp storage
  for (int lev=0; lev<vectGroundedThickness.size(); lev++)
    {
      if (vectGroundedThickness[lev] != NULL)
	{
	  delete vectGroundedThickness[lev];
	  vectGroundedThickness[lev] = NULL;
	}
    }

  return totalGroundedIce;

}

Real AmrIce::computeGroundedArea() const
{
  
  Real groundedArea = 0.0;

  Vector<LevelData<FArrayBox>* > vectGroundedIce(m_finest_level+1, NULL);

  for (int lev=0; lev<=m_finest_level; lev++)
    {
      vectGroundedIce[lev] = new LevelData<FArrayBox>(m_amrGrids[lev],1,
							    IntVect::Zero);

      LevelData<FArrayBox>& levelGroundedIce = *vectGroundedIce[lev];

      const LevelData<BaseFab<int> >& levelMask = m_vect_coordSys[lev]->getFloatingMask();
      // now loop through and set to one where we have grounded ice
      DataIterator dit=levelGroundedIce.dataIterator();
      for (dit.begin(); dit.ok(); ++dit)
	{
	  const BaseFab<int>& thisMask = levelMask[dit];
	  FArrayBox& thisIce = levelGroundedIce[dit];
	  thisIce.setVal(0.0);
	  BoxIterator bit(thisIce.box());
	  for (bit.begin(); bit.ok(); ++bit)
	    {
	      IntVect iv = bit();
	      if (thisMask(iv,0) == GROUNDEDMASKVAL)
		{
		  thisIce(iv,0) = 1.0;
		}
	    }
	}
    

    }


  // now compute sum
  groundedArea = computeSum(vectGroundedIce, m_refinement_ratios,
				m_amrDx[0], Interval(0,0), 0);

  // clean up temp storage
  for (int lev=0; lev<vectGroundedIce.size(); lev++)
    {
      if (vectGroundedIce[lev] != NULL)
	{
	  delete vectGroundedIce[lev];
	  vectGroundedIce[lev] = NULL;
	}
    }

  return groundedArea;

}

Real AmrIce::computeFloatingArea() const
{
  
  Real floatingArea = 0.0;

  Vector<LevelData<FArrayBox>* > vectFloatingIce(m_finest_level+1, NULL);

  for (int lev=0; lev<=m_finest_level; lev++)
    {
      vectFloatingIce[lev] = new LevelData<FArrayBox>(m_amrGrids[lev],1,
							    IntVect::Zero);

      LevelData<FArrayBox>& levelFloatingIce = *vectFloatingIce[lev];

      const LevelData<BaseFab<int> >& levelMask = m_vect_coordSys[lev]->getFloatingMask();
      // now loop through and set to one where we have floating ice
      DataIterator dit=levelFloatingIce.dataIterator();
      for (dit.begin(); dit.ok(); ++dit)
	{
	  const BaseFab<int>& thisMask = levelMask[dit];
	  FArrayBox& thisIce = levelFloatingIce[dit];
	  thisIce.setVal(0.0);
	  BoxIterator bit(thisIce.box());
	  for (bit.begin(); bit.ok(); ++bit)
	    {
	      IntVect iv = bit();
	      if (thisMask(iv,0) == FLOATINGMASKVAL)
		{
		  thisIce(iv,0) = 1.0;
		}
	    }
	}
    

    }

  // now compute sum
  floatingArea = computeSum(vectFloatingIce, m_refinement_ratios,
				m_amrDx[0], Interval(0,0), 0);

  
  // clean up temp storage
  for (int lev=0; lev<vectFloatingIce.size(); lev++)
    {
      if (vectFloatingIce[lev] != NULL)
	{
	  delete vectFloatingIce[lev];
	  vectFloatingIce[lev] = NULL;
	}
    }

  return floatingArea;

}

Real 
AmrIce::computeFluxOverIce(const Vector<LevelData<FArrayBox>* > a_flux)
{

  //compute sum of a flux component over ice
  //construct fluxOverIce
  Vector<LevelData<FArrayBox>* > fluxOverIce ( m_finest_level+1, NULL);
  for (int lev = 0; lev <= m_finest_level ; lev++)
    {
      fluxOverIce[lev] = new
	LevelData<FArrayBox>(m_amrGrids[lev],1, IntVect::Zero);
      const LevelData<FArrayBox>& thk = m_vect_coordSys[lev]->getH();
      //const LevelData<FArrayBox>* flux = a_flux[lev];
       
      for (DataIterator dit(m_amrGrids[lev]); dit.ok(); ++dit)
	{
	  const Box& box =  m_amrGrids[lev][dit];
	  const FArrayBox& source = (*a_flux[lev])[dit];
	  const FArrayBox& dit_thck = thk[dit];
	  FArrayBox& dit_fluxOverIce = (*fluxOverIce[lev])[dit];
	     
	  for (BoxIterator bit(box); bit.ok(); ++bit)
	    {
	      const IntVect& iv = bit();
	      // set fluxOverIce to source if thck > 0
	      if (dit_thck(iv) < 1e-10)
		{
		  dit_fluxOverIce(iv) = 0.0;
		}
	      else
		{
		  dit_fluxOverIce(iv) = source(iv);
		}
	    }
	    
	}
    }
  // compute sum
  Real tot_per_year = computeSum(fluxOverIce, m_refinement_ratios,m_amrDx[0],
				Interval(0,0), 0);

  //free storage
  for (int lev = 0; lev < m_finest_level ; lev++)
    {
      if (fluxOverIce[lev] != NULL)
	{
	  delete fluxOverIce[lev]; fluxOverIce[lev] = NULL;


	}
    }

  return tot_per_year;
}

Real 
AmrIce::computeTotalFlux(const Vector<LevelData<FArrayBox>* > a_flux)
{

  //compute sum of a flux for whole domain  
  Real tot_per_year = computeSum(a_flux, m_refinement_ratios,m_amrDx[0],
				Interval(0,0), 0);
  return tot_per_year;
}

void 
AmrIce::endTimestepDiagnostics()
{

      Real sumIce = computeTotalIce();
      if (s_verbosity > 0) 
	{
	  Real diffSum = sumIce - m_lastSumIce;
	  Real totalDiffSum = sumIce - m_initialSumIce;
  
	  Real sumGroundedIce = 0.0, diffSumGrounded = 0.0, totalDiffGrounded = 0.0;
	  Real VAF=0.0, diffVAF = 0.0, totalDiffVAF = 0.0;
	  Real groundedArea = 0.0, floatingArea = 0.0;
	  Real sumBasalFluxOverIce = 0.0, sumBasalFlux = 0.0;
	  Real sumCalvedIce = 0.0, sumRemovedIce = 0.0, sumAddedIce = 0.0;
	  Real sumAccumCalvedIce = 0.0;
	  Real diffAccumCalvedIce;
	  Real totalLostIce = 0.0;
	  //Real totalLostOverIce = 0.0;
	  Real sumSurfaceFluxOverIce = 0.0, sumSurfaceFlux = 0.0;
	  Real sumDivThckFluxOverIce = 0.0, sumDivThckFlux = 0.0;
	  //Real sumCalvedOverIce = 0.0;
	  Real sumRemovedOverIce = 0.0, sumAddedOverIce = 0.0;
	  if (m_report_grounded_ice)
	    {
	      sumGroundedIce = computeTotalGroundedIce();
	      diffSumGrounded = sumGroundedIce - m_lastSumGroundedIce;
	      totalDiffGrounded = sumGroundedIce - m_initialSumGroundedIce;      
	      m_lastSumGroundedIce = sumGroundedIce;
      
	      VAF = computeVolumeAboveFlotation();
	      diffVAF = VAF -  m_lastVolumeAboveFlotation;
	      totalDiffVAF = VAF - m_initialVolumeAboveFlotation;
	      m_lastVolumeAboveFlotation = VAF;
	    }
	  
	  if (m_report_area)
	    {
	      groundedArea = computeGroundedArea();
	      floatingArea = computeFloatingArea();
	    }

	  if (m_report_total_flux)

	    {
	      sumBasalFluxOverIce = computeFluxOverIce(m_basalThicknessSource);
	      sumSurfaceFluxOverIce = computeFluxOverIce(m_surfaceThicknessSource);
	      sumDivThckFluxOverIce = computeFluxOverIce(m_divThicknessFlux);
	      sumBasalFlux = computeTotalFlux(m_basalThicknessSource);
	      sumSurfaceFlux = computeTotalFlux(m_surfaceThicknessSource);
	      sumDivThckFlux = computeTotalFlux(m_divThicknessFlux);
	    }
	  
	  if (m_report_calving)

	    {
	      sumAccumCalvedIce = computeSum(m_melangeThickness, m_refinement_ratios,m_amrDx[0],
					Interval(0,0), 0);
	      diffAccumCalvedIce=sumAccumCalvedIce-m_lastSumCalvedIce;
	      sumCalvedIce = computeSum(m_calvedIceThickness, m_refinement_ratios,m_amrDx[0],
					Interval(0,0), 0);
	      sumRemovedIce = computeSum(m_removedIceThickness, m_refinement_ratios,m_amrDx[0],
					 Interval(0,0), 0);
	      sumAddedIce = computeSum(m_addedIceThickness, m_refinement_ratios,m_amrDx[0],
					 Interval(0,0), 0);
	      //sumCalvedOverIce = computeFluxOverIce(m_calvedIceThickness);
	      sumRemovedOverIce = computeFluxOverIce(m_removedIceThickness);
	      sumAddedOverIce = computeFluxOverIce(m_addedIceThickness);
	      totalLostIce = sumCalvedIce+sumRemovedIce+sumAddedIce;
	      //totalLostOverIce = sumCalvedOverIce+sumRemovedOverIce+sumAddedOverIce;

	      m_lastSumCalvedIce = sumAccumCalvedIce;

	    }


	  pout() << "Step " << m_cur_step << ", time = " << m_time << " ( " << time() << " ) " 
		 << ": sum(ice) = " << sumIce 
		 << " ( " << diffSum
		 << " " << totalDiffSum
		 << " )" << endl;
      
	  if (m_report_grounded_ice)
	    {
	      pout() << "Step " << m_cur_step << ", time = " << m_time << " ( " << time() << " ) "
		     << ": sum(grounded ice) = " << sumGroundedIce 
		     << " ( " << diffSumGrounded
		     << " " << totalDiffGrounded
		     << " )" << endl;

	      pout() << "Step " << m_cur_step << ", time = " << m_time << " ( " << time() << " ) "
		     << ": VolumeAboveFlotation = " << VAF
		     << " ( " << diffVAF
		     << " " << totalDiffVAF
		     << " )" << endl;
	    } 
	  if (m_report_area)
	    {
	      pout() << "Step " << m_cur_step << ", time = " << m_time << " ( " << time() << " ) "
		     << ": GroundedArea = " << groundedArea << " m2 " << endl;

	      pout() << "Step " << m_cur_step << ", time = " << m_time << " ( " << time() << " ) "
		     << ": FloatingArea = " << floatingArea << " m2 " << endl;

	    } 

	  if (m_report_total_flux)
	    {
	      if (m_dt > 0)
		{
		  pout() << "Step " << m_cur_step << ", time = " << m_time << " ( " << time() << " ) "
			 << ": BasalFlux = " << sumBasalFluxOverIce << " m3/yr " 
			 << " ( " << sumBasalFlux 
			 << "  " << sumBasalFlux-sumBasalFluxOverIce
			 << " )"
			 << endl;

		  pout() << "Step " << m_cur_step << ", time = " << m_time << " ( " << time() << " ) "
			 << ": SurfaceFlux = " << sumSurfaceFluxOverIce << " m3/yr  " 
			 << " ( " << sumSurfaceFlux 
			 << "  " << sumSurfaceFlux-sumSurfaceFluxOverIce 
			 << " )"
			 << endl;

		  pout() << "Step " << m_cur_step << ", time = " << m_time << " ( " << time() << " ) "
			 << ": DivergenceThicknessFlux = " << sumDivThckFluxOverIce << " m3/yr " 
			 << " ( " << sumDivThckFlux 
			 << "  " << sumDivThckFlux-sumDivThckFluxOverIce
			 << " )"
			 << endl;
		}
	    }



	  if (m_report_calving)
	    {
	      if (m_dt > 0)
		{
		  pout() << "Step " << m_cur_step << ", time = " << m_time << " ( " << time() << " ) "
			 << ": AccumCalvedIce = " << sumAccumCalvedIce << " m3 " 
			 << " ( " << diffAccumCalvedIce << "  " << diffAccumCalvedIce - totalLostIce << " ) " << endl;
		  pout() << "Step " << m_cur_step << ", time = " << m_time << " ( " << time() << " ) "
			 << ": CalvedIce = " << sumCalvedIce << " m3 " << " RemovedIce = " << sumRemovedIce << " m3 " << " AddedIce = " << sumAddedIce << " m3 Sum " << totalLostIce << " m3 " << endl;
		}
	    }

	  if (m_report_calving && m_report_total_flux)
	    {
	      if (m_dt > 0)
		{
		  Real cflux=sumCalvedIce/m_dt;
		  Real adjflux=(sumRemovedIce+sumAddedIce)/m_dt;
		  Real calvingerr=sumSurfaceFlux+sumBasalFlux-(cflux+diffSum/m_dt+adjflux);
		  pout() << "Step " << m_cur_step << ", time = " << m_time << " ( " << time() << " ) "
			 << ": Domain error = " << calvingerr << " m3/yr"
			 << " ( dV/dt = " << diffSum/m_dt 
			 << " calving flux = " << cflux
			 << " SMB = " << sumSurfaceFlux
			 << " BMB = " << sumBasalFlux
			 << " adjustment flux to maintain front = " << adjflux
			 << " )"  << endl;
	      
		  adjflux=(sumRemovedOverIce+sumAddedOverIce)/m_dt;
		  Real err=sumSurfaceFluxOverIce+sumBasalFluxOverIce-(sumDivThckFluxOverIce+diffSum/m_dt+adjflux);
		  pout() << "Step " << m_cur_step << ", time = " << m_time << " ( " << time() << " ) "
			 << ": Ice sheet error = " << err << " m3/yr"
			 << " ( dV/dt = " << diffSum/m_dt 
			 << " flux = " << sumDivThckFluxOverIce
			 << " smb = " << sumSurfaceFluxOverIce
			 << " bmb = " << sumBasalFluxOverIce
			 << " adjustment flux to maintain front = " << adjflux
			 << " )" << endl;
		}
	    }
	}

      m_lastSumIce = sumIce;

}

#include "NamespaceFooter.H"
