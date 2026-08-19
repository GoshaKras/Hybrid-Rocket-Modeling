import time as time_module
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as scp
import rocketcea as rcea
from CoolProp.CoolProp import PropsSI
from rocketcea.cea_obj import CEA_Obj
import CoolProp.CoolProp as CP
import sys
import inputs_io
from rocketcea.cea_obj import CEA_Obj, add_new_fuel, add_new_oxidizer, add_new_propellant
from scipy.ndimage import distance_transform_edt
from skimage.draw import polygon
from skimage import measure
import trimesh
from scipy import ndimage
import attempyforfgreg as regression_3d

# =============================================================================
# REGRESSION IMPLEMENTATIONS IN THIS FILE
#
# 1. LEGACY 2-D PIXEL METHOD: methods named ``*_legacy`` retain the old
#    fuel_grain_regression2 workflow for reference only.
# 2. ACTIVE 3-D METHOD: methods named ``*_3d`` use attempyforfgreg.py and are
#    what motormodel_copy2.py uses to generate its pixel-method coefficients.
# =============================================================================





class FuelGrain():
    def __init__(self,FuelGrainDiameter,fuelGrain_AreaReal,FuelGrainLength,start_chamber_pressure_pa,start_chamber_temperature_k,HowmuchInj_Help,Is_fuelGrain_Helix,helixrundiameter,Is_fuelgrain_Transient,Is_start_mass_gas_input,StartMass_Gas,Is_FuelGrain_GoshaStar,OneArchlengthestimate,preccandpostvolume,start_gas_gpermole,RealTime,Is_FuelGrain_PixelMethod,helixloopdiameter,plot_length_unit='mm',plot_area_unit=None,effectivelengthconstant=None):
        
        self.time=RealTime
        self.diameter_grain=FuelGrainDiameter
        self.area_grain=fuelGrain_AreaReal
        self.fuelgrain_raduis=self.diameter_grain/2
        self.insurfacearea=None
        self.regression_M_persec=0
        self.regression_MM_persec=None
        self.regression2=None
        self.reg3=None
        self.maybe_A=None
        self.initial_surface_area=None
        self.intail_surface_area=None
        self.surface_area_coeffs=None
        self.BlowingCo=0
        self.use_blowingco=0
        self.prandtl_number=None
        self.viscosity_forweirdreg=None
        self.totalrgession=0
        self.totalregression_MM=0
        self.oxflux=None
        self.mdotfuel=0
        self.sum_mdot=0
        self.totalmdot=None
        self.OF=None
        self.chamberpressure_PA=start_chamber_pressure_pa
        self.chamberpressure_PSI=self.chamberpressure_PA/6894.76
        self.cstar=None
        self.chamber_temp=None
        self.MW_chamber=None
        self.throatpressure_PA=None
        self.throatpressure_PSI=None
        self.injectorregression_help=HowmuchInj_Help
        self.regression_M_persec_withratio=None
        self.regression_MM_persec_withratio=None
        self.flowrateRatio_Use=1
        self.effectivelengthconstant=effectivelengthconstant
        if Is_fuelGrain_Helix==True:  #fix this
            self.chamberDensity=None
            self.streamvelocity=None
            self.viscosity=None
            self.reynoldsnumber=None
            self.cfstraight=None
            self.helixloopdiameter=helixrundiameter
            self.helixlength_meters=None
            self.HelixP_meters=None
            self.RC=None
            self.helixNominalDiameter=None
            self.CFhelix=None
            self.CFratio=1
            self.S_forhelix=0.00 
            self.S_forhelix2=0.00
            self.Reg_Helix_Ratio=None
            self.CorrectionRC=None
            self.RCattemp3=None
            self.RCDIff=None
            self.chamber_enthalpy=None
            self.BlowingRatio=None
            self.addtoarea_bcHelix=0
            self.lengthinbetween_2=None
            self.startinglength_inbetween_segment=None
            self.nottouchd_byoverlap=None
            self.triangehelicallength=None
            self.p_RC=None
            self.p_correctionfactor=None
            self.ratio_forCF=None
            self.Blowingp2=None
            self.helix_fc=None
            self.hydrolicdiamter=self.diameter_grain
            self.blowing_helix=None
            self.blowing_normal=None
            self.newblowingratio=None
            self.helix_wallshear=None
            self.ratiofor_pRC=None
            self.ratio2=None
            self.cf_for_blowing=None


        if Is_fuelgrain_Transient==True: 
            self.newchamberpressure_PA=start_chamber_pressure_pa
            self.newcstar= None
            self.mdotgen=None
            self.mdotnozzle=1
            self.mdotgain=None
            self.volumeflowrate=None
            self.volumeflowrate2=None
            self.oldvolumegrain=self.area_grain*FuelGrainLength+preccandpostvolume
            self.massGas=((self.oldvolumegrain*self.newchamberpressure_PA)/(8.314*start_chamber_temperature_k))*start_gas_gpermole/1000
            if Is_start_mass_gas_input==True:
                self.massGas=StartMass_Gas
            self.deltaPressure_PA=None
            self.newchamberpressure_psi=None
            self.deltaPressure_PA=None
            self.SpecR=None
            self.throatcrit_ratio=None
            self.realthroatratio=None
            self.Chocked_status=False

            
            
        if Is_FuelGrain_GoshaStar==True:#change this later
            #fix this goofy ah calling
            self.complexcircum=None
            self.complexarea=None
            self.eQraduis=None
            self.eQarea=None
            self.bigcircleraduis=None
            self.smallcircleraduis=None
            self.circleamount=None
            self.eQPerimeter=None
            self.arearatio=None
            self.circumratio=None
            self.onearchlengthestimate=OneArchlengthestimate
            self.arclength_rweclose=None
            self.Newarclengthesitmate=None
            self.complexSHapetotalReg=0
            self.oldraduis=None
            self.oldarea=None
            self.arearatio2=None
            self.addtoarea_bcHelix=0
            self.ratioprecent=0
            self.oldpermeter=None
            self.Newstararea=0
            self.newstarpermeter=0
            self.circleareachange=0
            self.circle_circum_change=0
            self.anotherarea_ratio=0
            self.anotherarea_ratio2=0
            self.areaDiffprecent=0
            self.mod_circcheck=0
            self.testArea=0
            self.testCircum=0
            self.modratio=0
            self.anothermdot=0
            self.insurfacearea2=0
            self.mdotdiff=0
            self.circumdiff=0
            self.areadiff=0
            self.newblowingratio=None
            self.hydrolicdiamter=self.diameter_grain
            
        
            self.flowrateRatio_Use=1
            self.surfaceareaadd=0
        
        # Initialize coefficients for all cases (required for pixelmethodgeo)
        self.area_coeffs = None  # Will be filled by run_regression_analysis()
        self.perimeter_coeffs = None  # Will be filled by run_regression_analysis()
        
        if Is_FuelGrain_PixelMethod==True:
            self.p_1archlengthestimate="something"
            # Polynomial coefficients for pixel method geometry (from regression analysis)
            self.max_inscribed_coeffs = None  # Will be filled by run_regression_analysis()
            self.min_enclosing_coeffs = None  # Will be filled by run_regression_analysis()
            self.circle_overlap_coeffs = None  # Will be filled by run_regression_analysis()
            self.largest_arm_coeffs = None  # Will be filled by run_regression_analysis()
            self._coefficient_warnings_printed = False  # Flag to print coefficient warnings only once
            self.pixelAREA=None
            self.P_perimeter=None
            self.P_surfacearea=None
            self.p_mdotfuel=None
            self.Howcircle=None
            self.AreaDiff_p=None
            self.oldpixelcircum=None
            self.deltacircum=None
            self.P_min_enclosing_diameter=None
            self.P_max_inscribed_diameter=None
            self.diameter_minmax_diff=None
            self.P_largest_arm_contact=None
            self.P_largest_arm_percentage=None
            self.P_circle_overlap=None
            self.arclengthdiffernce=None
            self.ratio_touchedByhelix=None
            self.p_helixloopdiameter=helixloopdiameter
            self.plot_length_unit = plot_length_unit
            self.plot_area_unit = plot_area_unit
            
            
        
            
            




        #stuff for csv
        self._init_attrs = list(self.__dict__.keys())
        self.outputcsv = pd.DataFrame(columns=self._init_attrs)
        self._output_rows = []
    def stuffnoprint (self,Is_fuelGrain_PixelMethod,outerdiameter_inches,totalComplexArea,Is_fuelGrain_Helix,revPitch,PitchFor_Helix,Usepixel_calcs_SAV):
        self.dontcare=0
        self.gamma_chamber=None
        self.gamma_throat=None
        self.gamma_exit=None
        self.enthapy_vaporpyrolsis_fuel=51.88 #kj/kg
        self.Is_pixel=Is_fuelGrain_PixelMethod
        self.Is_helix=Is_fuelGrain_Helix
        self.Usepixel_calcs_SAV=Usepixel_calcs_SAV
        self.OuterDiameter_inches=outerdiameter_inches
        self.Areabasedon_OuterDiameter_mm=( ( (outerdiameter_inches*25.4)/2 )**2 )*3.14
        
        
        self.oldarea_p=totalComplexArea
        self.fuelregcon=0
        self.OGHD=self.diameter_grain
        if Is_fuelGrain_Helix==True:
            self.helixlength_meters=revPitch*np.sqrt(((self.helixloopdiameter*3.14)**2)+(PitchFor_Helix)**2) 
            self.HelixP_meters=(self.helixlength_meters/revPitch)
            self.RC=(self.helixloopdiameter/2)*(1+(self.HelixP_meters/(3.14*self.helixloopdiameter))**2)

    



    def should_I_Print(self,Is_fuelGrain_Helix,Is_fuelgrain_Transient,Is_FuelGrain_GoshaStar,FuelGrainLength,start_chamber_pressure_pa,start_chamber_temperature_k,start_gas_gpermole,Is_start_mass_gas_input,StartMass_Gas,OneArchlengthestimate,helixrundiameter):
        if Is_fuelGrain_Helix==False: 
            self.chamberDensity=None
            self.streamvelocity=None
            self.viscosity=None
            self.reynoldsnumber=None
            self.cfstraight=None
            self.helixloopdiameter=helixrundiameter
            self.helixlength_meters=None
            self.HelixP_meters=None
            self.RC=None
            self.helixNominalDiameter=None
            self.CFhelix=None
            self.CFratio=1
            self.S_forhelix=0.00 
            self.S_forhelix2=0.00
            self.Reg_Helix_Ratio=None
            self.CorrectionRC=None
            self.RCattemp3=None
            self.RCDIff=None
            self.chamber_enthalpy=None
            self.BlowingRatio=None
            
        if Is_fuelgrain_Transient==False: 
            self.newchamberpressure_PA=start_chamber_pressure_pa
            self.newcstar= None
            self.mdotgen=None
            self.mdotnozzle=1
            self.mdotgain=None
            self.volumeflowrate=None
            self.volumeflowrate2=None
            self.oldvolumegrain=self.area_grain*FuelGrainLength
            self.massGas=((self.oldvolumegrain*self.newchamberpressure_PA)/(8.314*start_chamber_temperature_k))*start_gas_gpermole/1000
            if Is_start_mass_gas_input==True:
                self.massGas=StartMass_Gas
            self.deltaPressure_PA=None
            self.SpecR=None
            self.throatcrit_ratio=None
            self.realthroatratio=None
            self.Chocked_status=False

            
            
        if Is_FuelGrain_GoshaStar==False:#change this later
            #fix this goofy ah calling
            self.complexcircum=None
            self.complexarea=None
            self.eQraduis=None
            self.eQarea=None
            self.bigcircleraduis=None
            self.smallcircleraduis=None
            self.circleamount=None
            self.eQPerimeter=None
            self.arearatio=None
            self.circumratio=None
            self.onearchlengthestimate=OneArchlengthestimate
            self.arclength_rweclose=None
            self.Newarclengthesitmate=None
            self.complexSHapetotalReg=0
            self.oldraduis=None
            self.oldarea=None
            self.arearatio2=None
    
            self.flowrateRatio_Use=1
            self.surfaceareaadd=0
        
           
    def starshape(self,CenterCircleFuelGrain_Raduis,OutsideSmallCircle_raduis,AmountofSmallCircles,whatRaduisWouldNeedtobeOfNormalCircleToMatch,OneArchlengthestimate,TotalComplexArea,ComplexPerimeter,Fuelgrainlength):
        self.complexcircum= ComplexPerimeter
        self.complexarea= TotalComplexArea
        self.eQraduis= whatRaduisWouldNeedtobeOfNormalCircleToMatch
        self.eqradOG=self.eQraduis
        self.eQarea=self.eQraduis*self.eQraduis*3.14
        self.bigcircleraduis= CenterCircleFuelGrain_Raduis
        self.smallcircleraduis= OutsideSmallCircle_raduis
        self.circleamount= AmountofSmallCircles
        self.eQPerimeter= self.eQraduis*6.28
        self.circumratio=1
        self.arearatio=1
        self.Newarclengthesitmate=OneArchlengthestimate
        self.oldraduis=whatRaduisWouldNeedtobeOfNormalCircleToMatch
        self.oldarea=self.oldraduis*self.oldraduis*3.14
        self.oldpermeter=whatRaduisWouldNeedtobeOfNormalCircleToMatch*6.28
        self.Newstararea=TotalComplexArea
        self.newstarpermeter=ComplexPerimeter
        self.arearatio2=None
        self.newraduis=self.oldraduis
        self.newarea=self.oldarea
        self.testRad=self.oldraduis
        self.insurfacearea2=self.oldpermeter*Fuelgrainlength*self.effectivelengthconstant
        self.testArea=TotalComplexArea
        self.testCircum=ComplexPerimeter
        
    
   



        

        
        



    def oxflux_func(self,RealMassFlowrate,RealTime):
        self.time=RealTime
        self.oxflux=RealMassFlowrate/self.area_grain
    def regression_func (self,Status,Regression_Aco,Regression_Nco,regfluxstuff,Timestep):
        if Status=="Vapour":
            self.injectorregression_help=1
        self.regression_MM_persec=(Regression_Aco*(self.oxflux/regfluxstuff)**Regression_Nco)*self.injectorregression_help
        
        self.regression_MM_persec_withratio=self.regression_MM_persec*self.CFratio
        self.regression_M_persec=self.regression_MM_persec/1000
        self.regression_M_persec_withratio=self.regression_MM_persec_withratio/1000 
        self.totalrgession=self.totalrgession+(self.regression_M_persec_withratio*Timestep*2)
        self.totalregression_MM=self.totalrgession*1000/2 #for pixel method
    def simpleCircleGeo(self,FuelGrainLength,Timestep,Is_FuelGrain_GoshaStar,Is_fuelGrain_Helix):
        self.portperimeter= self.diameter_grain*3.14
        
        self.insurfacearea=self.portperimeter*self.effectivelengthconstant
        self.diameter_grain=self.diameter_grain+(self.regression_M_persec*Timestep*2)
        self.fuelgrain_raduis=self.diameter_grain/2
        self.area_grain=(self.fuelgrain_raduis*self.fuelgrain_raduis*3.14)
        self.volume_grain=self.area_grain*FuelGrainLength
        if Is_FuelGrain_GoshaStar==True:
            self.area_grain=self.eQarea
            self.insurfacearea=self.complexcircum*FuelGrainLength*self.effectivelengthconstant
        if self.Is_pixel==True:
            self.area_grain=self.pixelAREA
            self.insurfacearea=self.P_perimeter*FuelGrainLength*self.effectivelengthconstant
        if Is_fuelGrain_Helix==True:
            self.surfaceareaadd=0
            self.addtoarea_bcHelix=0
                         
        self.volume_grain=self.area_grain*FuelGrainLength
    def _store_initial_surface_area(self, surface_area_m2=None):
        """Store the initial inner surface area in both spelling variants for downstream use."""
        if surface_area_m2 is None:
            if self.surface_area_coeffs is not None:
                surface_area_mm2 = float(np.polyval(self.surface_area_coeffs, 0.0))
                surface_area_m2 = max(surface_area_mm2, 0.0) / (1000.0 * 1000.0)
            elif self.P_surfacearea is not None:
                surface_area_m2 = float(self.P_surfacearea)
            else:
                surface_area_m2 = None

        if surface_area_m2 is not None:
            self.initial_surface_area = float(surface_area_m2)
            self.intail_surface_area = float(surface_area_m2)
        print(f"Initial inner surface area stored: {self.initial_surface_area:.6f} m²")
        return self.initial_surface_area

    # =========================================================================
    # PIXEL-METHOD RUNTIME GEOMETRY
    # Uses coefficients loaded by run_regression_analysis(), which currently
    # delegates to the active 3-D attempyforfgreg implementation below.
    # =========================================================================
    def pixelmethodgeo(self,Is_fuelGrain_Helix,FuelGrainLength,Fuel_Density,Timestep,amountofsmallcircles,effectivelengthconstant,Override_effectivelengthconstant):
        """
        Calculate pixel method geometry using polynomial curve fits from regression analysis.
        Uses 2nd-degree polynomial fits for area and perimeter extracted from the OBJ geometry.
        """
        # Get current regression distance in mm
        regression_distance_mm = self.totalregression_MM
        
        # Use the polynomial coefficients extracted from regression analysis
        # If coefficients haven't been set yet, use placeholder values (warnings printed only once)
        if self.area_coeffs is None:
            if not self._coefficient_warnings_printed:
                print("WARNING: Polynomial coefficients not yet calculated. Run run_regression_analysis() first.")
                print("  Using placeholder values for now. Results will be inaccurate until proper coefficients are loaded.")
                self._coefficient_warnings_printed = True
            area_coeffs = np.array([-0.0485, 3.5892, 3142.75])
        else:
            area_coeffs = self.area_coeffs
            
        if self.perimeter_coeffs is None:
            perimeter_coeffs = np.array([0.0012, -2.8456, 188.94])
        else:
            perimeter_coeffs = self.perimeter_coeffs
        
        if self.max_inscribed_coeffs is None:
            max_inscribed_coeffs = np.array([0.0, 0.0, 50.0])
        else:
            max_inscribed_coeffs = self.max_inscribed_coeffs
            
        if self.min_enclosing_coeffs is None:
            min_enclosing_coeffs = np.array([0.0, 0.0, 100.0])
        else:
            min_enclosing_coeffs = self.min_enclosing_coeffs
        
        if self.circle_overlap_coeffs is None:
            circle_overlap_coeffs = np.array([0.0, 0.0, 0.0, 100.0])
        else:
            circle_overlap_coeffs = self.circle_overlap_coeffs
        
        if self.largest_arm_coeffs is None:
            largest_arm_coeffs = np.array([0.0, 0.0, 0.0, 50.0])
        else:
            largest_arm_coeffs = self.largest_arm_coeffs
        
        # Calculate values using polynomial fits
        x = regression_distance_mm
        
        # Area calculation (mm²) from polynomial fit - uses np.polyval for variable degree
        self.pixelAREA = np.polyval(area_coeffs, x)
        self.pixelAREA = max(0, self.pixelAREA)  # Ensure non-negative
        self.pixelAREA=self.pixelAREA/(1000*1000)  # Subtract from initial area based on OD
      
        if self.time>Timestep:
            self.oldarea2=self.P_eqraduis*self.P_eqraduis*3.14
            self.P_eqraduis+=self.regression_M_persec*Timestep
            self.oldarea_p=self.P_eqraduis*self.P_eqraduis*3.14

            self.AreaDiff_p=(self.pixelAREA-self.oldarea2)/(self.oldarea_p-self.oldarea2)
        self.P_eqraduis = np.sqrt(self.pixelAREA / 3.14)  
        self.P_eqPerimeter = self.P_eqraduis * 6.28
        
        # Perimeter calculation (mm) from polynomial fit - uses np.polyval for variable degree
        self.P_perimeter = np.polyval(perimeter_coeffs, x)
        self.P_perimeter = max(0, self.P_perimeter)
        self.P_perimeter=self.P_perimeter/1000  # Ensure non-negative
        if self.time==Timestep and self.Is_helix==True:
            
            self.effectivelengthconstant=self.initial_surface_area/(self.P_perimeter*FuelGrainLength)
            self.P_surfacearea=self.initial_surface_area
        else:
            self.P_surfacearea=self.P_perimeter*FuelGrainLength
        if Override_effectivelengthconstant==True:
            self.effectivelengthconstant=effectivelengthconstant
        self.insurfacearea=self.P_surfacearea
        if self.initial_surface_area is None:
            self._store_initial_surface_area(self.P_surfacearea)
        
        # Max inscribed circle diameter calculation (mm) from polynomial fit - uses np.polyval for variable degree
        self.P_max_inscribed_diameter = np.polyval(max_inscribed_coeffs, x)
        self.P_max_inscribed_diameter = max(0, self.P_max_inscribed_diameter)
        self.P_max_inscribed_diameter = self.P_max_inscribed_diameter / 1000  # Convert to meters
        
        # Min enclosing circle diameter calculation (mm) from polynomial fit - uses np.polyval for variable degree
        self.P_min_enclosing_diameter = np.polyval(min_enclosing_coeffs, x)
        self.P_min_enclosing_diameter = max(0, self.P_min_enclosing_diameter)
        self.P_min_enclosing_diameter = self.P_min_enclosing_diameter / 1000  # Convert to meters
        
        # Circle boundary overlap calculation (mm) from polynomial fit - uses np.polyval for variable degree
        self.P_circle_overlap = np.polyval(circle_overlap_coeffs, x)
        self.P_circle_overlap = max(0, self.P_circle_overlap)
        self.P_circle_overlap = self.P_circle_overlap / 1000  # Convert to meters
        self.P_circle_overlap=self.P_circle_overlap*5/3
        
        # Largest arm contact arc length calculation (mm) from polynomial fit - uses np.polyval for variable degree
        self.P_largest_arm_contact = np.polyval(largest_arm_coeffs, x)
        self.P_largest_arm_contact = max(0, self.P_largest_arm_contact)
        self.P_largest_arm_contact = self.P_largest_arm_contact / 1000  # Convert to meters
        
        # Calculate largest arm as percentage of total circle perimeter
        if self.P_circle_overlap > 0:
            self.P_largest_arm_percentage = (self.P_largest_arm_contact *amountofsmallcircles/ self.P_perimeter) * 100
        else:
            self.P_largest_arm_percentage = 0
        if self.P_largest_arm_percentage<5:
            self.P_largest_arm_contact=0
        
        self.Howcircle=1/(self.P_eqPerimeter/self.P_perimeter)
        if self.time>=Timestep:
            self.p_mdotfuel=((self.pixelAREA-self.oldarea)*Fuel_Density*FuelGrainLength*self.effectivelengthconstant)/Timestep
        self.oldsurfacearea=self.P_surfacearea
        self.oldarea=self.pixelAREA
        self.oldpixelcircum=self.P_perimeter
        self.diameter_minmax_diff=self.P_min_enclosing_diameter - self.P_max_inscribed_diameter
        

    
        
        # Calculate hydraulic diameter
        if self.P_perimeter > 0:
            self.hydrolicdiamter = (4 * self.pixelAREA) / self.P_perimeter
        else:
            self.hydrolicdiamter = 0
        
        # Calculate inner surface area (for mass flow calculations)
        #self.insurfacearea = self.P_perimeter * FuelGrainLength
        #self.insurfacearea = self.P_surfacearea
        
       # if Is_fuelGrain_Helix==True:
           # self.insurfacearea = self.insurfacearea + self.surfaceareaadd

    # =========================================================================
    # LEGACY 2-D PIXEL REGRESSION (fuel_grain_regression2.py)
    # Retained for comparison only. It is not called by the motor model.
    # =========================================================================
    def _run_regression_analysis_legacy(self, obj_file_path, regression_rate=3.0, time_seconds=30, cross_section_axis=2, show_plots=True, plot_length_unit=None, plot_area_unit=None):
        """
        Run full fuel grain regression analysis with all graphs and curve fits.
        This will display area, perimeter, and inscribed circle diameter graphs.
        
        Args:
            obj_file_path: Path to the OBJ file
            regression_rate: Regression rate in mm/sec
            time_seconds: Time to simulate in seconds
            cross_section_axis: Which axis for cross-section (0=X, 1=Y, 2=Z)
        """
        # Optionally disable interactive plotting so graphs don't pop up
        _plt_was_interactive = plt.isinteractive()
        _prev_backend = mpl.get_backend()
        if not show_plots:
            try:
                plt.switch_backend('Agg')
            except Exception:
                pass
            plt.ioff()

        # Create simulator instance
        simulator = FuelGrainRegressionSimulator(
            obj_file_path,
            outer_diameter_inches=self.OuterDiameter_inches,
            resolution=500,
            cross_section_axis=cross_section_axis,
            plot_length_unit=self.plot_length_unit if plot_length_unit is None else plot_length_unit,
            plot_area_unit=self.plot_area_unit if plot_area_unit is None else plot_area_unit
        )
        
        print("\n" + "="*60)
        print("Fuel Grain Regression Analysis")
        print("="*60)
        print(f"Configuration:")
        print(f"  OBJ File: {obj_file_path}")
        print(f"  Regression Rate: {regression_rate} mm/sec")
        print(f"  Simulation Time: {time_seconds} sec")
        print(f"  Total Regression: {regression_rate * time_seconds} mm")
        print("="*60 + "\n")
        
        # Run geometry extraction
        section_points_2d = simulator.read_obj_geometry()
        X, Y = simulator.create_initial_grid(section_points_2d)
        
        # Debug info
        print(f"\nGrid Statistics:")
        print(f"  Grid shape: {simulator.grid.shape}")
        print(f"  Grid min: {simulator.grid.min()}, max: {simulator.grid.max()}")
        print(f"  Non-zero cells: {np.sum(simulator.grid)}")
        print(f"  Grid fill percentage: {100 * np.sum(simulator.grid) / simulator.grid.size:.2f}%")
        
        # Calculate and display pixel size
        radius_with_margin = simulator.od_radius * 1.2
        grid_width = 2 * radius_with_margin
        mm_per_pixel = grid_width / simulator.resolution
        print(f"  Pixel size: {mm_per_pixel:.4f} mm/pixel")
        
        # Display raw OBJ cross-section (optional)
        if show_plots:
            print("\nGenerating visualization: OBJ Cross-Section Geometry...")
            simulator.plot_dxf_geometry()
            # Display interactive regression slider
            print("Generating interactive regression visualization...")
            print("Use the slider to see how regression changes over time\n")
            simulator.plot_cross_section_interactive(regression_rate, time_seconds)
        
        # Display area vs regression graph and capture coefficients
        if show_plots:
            print("Generating area vs regression graph...")
            self.area_coeffs = simulator.plot_area_vs_regression(max_regression_distance=simulator.id_radius)
        else:
            # compute coefficients without showing plots
            self.area_coeffs = simulator.plot_area_vs_regression(max_regression_distance=simulator.id_radius)
        
        # Display perimeter vs regression graph and capture coefficients
        print("Generating perimeter vs regression graph...")
        self.perimeter_coeffs = simulator.plot_perimeter_vs_regression(max_regression_distance=simulator.id_radius)

        # Display surface area vs regression graph and capture coefficients
        print("Generating surface area vs regression graph...")
        self.surface_area_coeffs = simulator.plot_surface_area_vs_regression(max_regression_distance=simulator.id_radius)
        self._store_initial_surface_area()
        
        # Display max inscribed circle vs regression graph and capture coefficients
        print("Generating max inscribed circle diameter vs regression graph...")
        self.max_inscribed_coeffs = simulator.plot_max_inscribed_circle_vs_regression(max_regression_distance=simulator.id_radius)
        
        # Display min enclosing circle vs regression graph and capture coefficients
        print("Generating min enclosing circle diameter vs regression graph...")
        self.min_enclosing_coeffs = simulator.plot_min_enclosing_circle_vs_regression(max_regression_distance=simulator.id_radius)
        
        # Display circle boundary overlap vs regression graph and capture coefficients
        print("Generating circle boundary overlap vs regression graph...")
        self.circle_overlap_coeffs = simulator.plot_circle_boundary_overlap_vs_regression(max_regression_distance=simulator.id_radius)
        
        # Display largest arm overlap vs regression graph and capture coefficients
        print("Generating largest arm contact length vs regression graph...")
        self.largest_arm_coeffs = simulator.plot_largest_arm_overlap_vs_regression(max_regression_distance=simulator.id_radius)
        
        print("\nRegression analysis complete!")
        print(f"Polynomial coefficients stored for pixel method geometry calculations.")
        # restore interactive plotting state and backend
        if not show_plots:
            try:
                plt.switch_backend(_prev_backend)
            except Exception:
                pass
            if _plt_was_interactive:
                plt.ion()

    # =========================================================================
    # ACTIVE 3-D PIXEL REGRESSION (attempyforfgreg.py)
    # This is the source of area/perimeter/surface/contact fits for the motor.
    # =========================================================================
    def run_regression_analysis_3d(self, obj_file_path, regression_rate=3.0, time_seconds=30,
                                cross_section_axis=2, show_plots=True, plot_length_unit=None,
                                plot_area_unit=None, in_plane_resolution=500,
                                length_resolution=200, unit_scale_mm=None,
                                graph_points=101, snapshot_regression_mm=None,
                                snapshot_path=None, show_plane_preview=False,
                                open_interactive_viewer=False):
        """Build pixel-method geometry fits from the accurate 3-D voxel regression model.

        This replaces the 2-D ``FuelGrainRegressionSimulator`` dependency for
        port area, burnable perimeter, burning surface area, and inscribed
        diameter. Coefficients use regression distance in millimetres.
        """
        obj_path = regression_3d.Path(obj_file_path)
        if not obj_path.is_file():
            raise FileNotFoundError(f"OBJ file not found: {obj_path}")
        if cross_section_axis not in (0, 1, 2):
            raise ValueError("cross_section_axis must be 0 (X), 1 (Y), or 2 (Z)")

        if in_plane_resolution < 20 or length_resolution < 2 or graph_points < 3:
            raise ValueError("in_plane_resolution >=20, length_resolution >=2, and graph_points >=3 are required")
        scale_to_mm = regression_3d.infer_unit_scale_to_mm(obj_path, unit_scale_mm)
        mesh = regression_3d.load_mesh(obj_path, scale_to_mm)
        if show_plots and show_plane_preview:
            cross_section_axis = regression_3d.preview_high_resolution_plane(
                mesh, cross_section_axis, in_plane_resolution
            )
        axis_name = ("X", "Y", "Z")[cross_section_axis]
        print(f"3-D regression: plots {'will show' if show_plots else 'will not show'}.")
        print(f"3-D regression plane: {axis_name}; raster: {in_plane_resolution} x "
              f"{in_plane_resolution}; length slices: {length_resolution}.")
        with regression_3d.activity_indicator("Rasterizing high-resolution cross-sections...", len(mesh.faces)):
            solid, pitch = regression_3d.make_solid_voxels(
                mesh, cross_section_axis, in_plane_resolution, length_resolution
            )
        initial_port = regression_3d.central_port_mask(solid, cross_section_axis)
        if snapshot_regression_mm is not None:
            regression_3d.export_regressed_fuel_snapshot(
                solid, initial_port, pitch, float(snapshot_regression_mm),
                regression_3d.Path(snapshot_path or "regression_outputs/regressed_fuel_snapshot.obj"),
            )
        max_regression_mm = regression_rate * time_seconds
        distances = np.linspace(0.0, max_regression_mm, graph_points)
        plane_rows = regression_3d.picked_plane_regression(
            solid, initial_port, pitch, cross_section_axis, distances
        )
        surface_rows = regression_3d.regression_table(
            solid, initial_port, pitch, cross_section_axis, distances,
            mesh_volume=float("nan"), mesh_area=float(mesh.area),
            initial_bore_area=regression_3d.mesh_bore_surface_area(mesh, cross_section_axis),
        )
        x = distances

        def fit_metric(rows, key):
            values = np.asarray([row[key] for row in rows], dtype=float)
            degree = min(5, len(x) - 1)
            return np.polyfit(x, values, degree)

        self.area_coeffs = fit_metric(plane_rows, "port_area_mm2")
        self.perimeter_coeffs = fit_metric(plane_rows, "port_perimeter_mm")
        self.max_inscribed_coeffs = fit_metric(plane_rows, "max_inscribed_diameter_mm")
        self.min_enclosing_coeffs = fit_metric(plane_rows, "min_enclosing_diameter_mm")
        self.circle_overlap_coeffs = fit_metric(plane_rows, "circle_boundary_overlap_mm")
        self.largest_arm_coeffs = fit_metric(plane_rows, "largest_arm_contact_mm")
        self.surface_area_coeffs = fit_metric(surface_rows, "burning_surface_area_mm2")
        self._store_initial_surface_area(float(surface_rows[0]["burning_surface_area_mm2"]) / 1_000_000.0)

        print("3-D regression geometry loaded into FuelGrain:")
        print(f"  plane: {('X', 'Y', 'Z')[cross_section_axis]}; grid: {tuple(solid.shape)}")
        print(f"  fitted range: 0 to {max_regression_mm:.3f} mm")
        if show_plots:
            display_unit, display_scale = regression_3d.display_unit_scale(
                self.plot_length_unit if plot_length_unit is None else plot_length_unit
            )
            if open_interactive_viewer:
                regression_3d.show_interactive_viewer(
                    solid, initial_port, pitch, distances, surface_rows, cross_section_axis,
                    regression_rate, display_unit, display_scale,
                )
            regression_3d.save_picked_plane_analysis(
                plane_rows, cross_section_axis, None, display_unit, display_scale,
                regression_3d.Path("regression_outputs") / "fuelgrain_3d_plane_data.csv",
                regression_3d.Path("regression_outputs") / "fuelgrain_3d_curve_fits.csv", show=True,
            )

    def run_regression_analysis(self, *args, **kwargs):
        """Compatibility entry point: run the active 3-D regression workflow."""
        return self.run_regression_analysis_3d(*args, **kwargs)

    def masses_of_stuff(self, Fuel_Density, OxtankMath,Timestep, FuelGrainLength):
        
        #self.mdotfuel=self.insurfacearea2*self.regression_M_persec*Fuel_Density#*self.flowrateRatio_Use
        if self.Is_pixel==True :
            # If it's the first time step, calculate the mass flow rate
            if self.time==Timestep:
                self.mdotfuel=(self.P_perimeter+self.oldpixelcircum)*Fuel_Density*FuelGrainLength*self.regression_M_persec_withratio*0.5
            else:
                self.mdotfuel=self.p_mdotfuel
            #self.oldpixelcircum=self.P_perimeter
        else:
            self.mdotfuel=self.insurfacearea*Fuel_Density*self.regression_M_persec_withratio

        
        self.totalmdot=self.mdotfuel+OxtankMath.RealMassFlowRate
    def Ox_Fuel_Ratio(self,RealMassFlowRate):
        self.OF= RealMassFlowRate/self.mdotfuel
    def cstar_func(self,CEAforRocket):
        self.cstar=CEAforRocket.get_Cstar(Pc=self.chamberpressure_PSI, MR=self.OF) * 0.3048 #m/sec
        self.chamber_temp=(CEAforRocket.get_Temperatures(Pc=self.chamberpressure_PSI, MR=self.OF, eps=1, frozen=0)[0])*0.555556
       
    def chamberpressure_func(self,NozzleMath,NozzleCD_ForChamberPressure,Ncomb_ForChamberPressure,throatarea_fromNozzle,Is_fuelgrain_Transient):
        self.chamberpressure_PA=(self.totalmdot*self.cstar*Ncomb_ForChamberPressure)/(throatarea_fromNozzle*NozzleCD_ForChamberPressure)
        if Is_fuelgrain_Transient==True:
            self.chamberpressure_PA=self.newchamberpressure_PA
        self.chamberpressure_PSI=self.chamberpressure_PA/6894.76
        
    
    def gammas(self,CEAforRocket,expansionratio):
        self.MW_exit, self.gamma_exit = CEAforRocket.get_exit_MolWt_gamma(Pc=self.chamberpressure_PSI, MR=self.OF)
        self.MW_throat, self.gamma_throat = CEAforRocket.get_Throat_MolWt_gamma(Pc=self.chamberpressure_PSI, MR=self.OF, eps=expansionratio)
        self.MW_chamber, self.gamma_chamber = CEAforRocket.get_Chamber_MolWt_gamma(Pc=self.chamberpressure_PSI, MR=self.OF, eps=expansionratio)

    def throatPressure(self,CEAforRocket):
        self.throatpressure_PA=self.chamberpressure_PA/CEAforRocket.get_Throat_PcOvPe(Pc=self.chamberpressure_PSI, MR=self.OF)
        self.throatpressure_PSI=self.throatpressure_PA/6894.76
    
    def densitys(self,CEAforRocket):
        self.chamberDensity= CEAforRocket.get_Chamber_Density(Pc=self.chamberpressure_PSI, MR=self.OF)*16.01846 #convert from lb/ft^3 to kg/m^3

    @staticmethod
    def calculate_fc(epsilon, r0, radius):
        """Return F_c from the Reynolds-free implicit friction relation."""
        if min(epsilon, r0, radius) <= 0:
            raise ValueError("epsilon, r0, and radius must be positive")

        radius_ratio_sqrt = np.sqrt(r0 / radius)

        def residual(fc):
            log_argument = 0.104 * epsilon / fc * radius_ratio_sqrt
            return 1 / np.sqrt(fc) + 0.923 * np.log(log_argument)

        candidates = np.logspace(-12, 0, 400)
        for lower, upper in zip(candidates[:-1], candidates[1:]):
            lower_residual = residual(lower)
            upper_residual = residual(upper)
            if lower_residual == 0:
                return lower
            if lower_residual * upper_residual < 0:
                return scp.optimize.brentq(residual, lower, upper)

        raise ValueError("The Reynolds-free F_c equation has no positive solution between 1e-12 and 1")

    def helixmath(self,CEAforRocket,Timestep,Is_FuelGrain_GoshaStar,Is_fuelGrain_Helix,revPitch,PitchFor_Helix,AmountofSmallCircles,Is_FuelGrain_PixelMethod,FuelGrainDiameter,expansionratio,OusideSmallCircle_raduis,helical_archsegment,thoart_area,fuelgrainlength,mw_ox,Fuel_Density,oxden,sandgrainroughness):
        self.fac_CR=(self.hydrolicdiamter*self.hydrolicdiamter*3.14/4)/thoart_area
        self.chambermach=CEAforRocket.get_Chamber_MachNumber(Pc=self.chamberpressure_PSI, MR=self.OF, fac_CR=self.fac_CR)
        # Try to get chamber sonic velocity via available API; fall back to a reasonable default
        #try:
        self.chambersonicvelocity = CEAforRocket.get_SonicVelocities(Pc=self.chamberpressure_PSI, MR=self.OF, eps=expansionratio)[0]
        #except Exception:
            #self.chambersonicvelocity = 340.0
        self.streamvelocity=self.chambersonicvelocity*self.chambermach
        #self.streamvelocity=np.sqrt(2*((self.chamberpressure_PA-self.throatpressure_PA)/self.chamberDensity))
        self.chamberstuff=CEAforRocket.get_Chamber_Transport(Pc=self.chamberpressure_PSI, MR=self.OF, eps=expansionratio,frozen=0)
        self.viscosity= self.chamberstuff[1] * 0.0001# kg/m/s
        if Is_FuelGrain_GoshaStar==True:
            self.hydrolicdiamter=self.testArea*4/self.testCircum 
        elif self.Is_pixel==False:
            self.hydrolicdiamter=self.diameter_grain
        self.reynoldsnumber=(self.streamvelocity*self.hydrolicdiamter*self.chamberDensity)/self.viscosity
        if self.Is_pixel==True:
            self.P_reynoldsnumber=(self.streamvelocity*self.hydrolicdiamter*self.chamberDensity)/self.viscosity
        self.cfstraight=0.074/(self.reynoldsnumber**0.2) 
        #self.cfstraight=0.0776*(((np.log10(self.reynoldsnumber)-1.88)**-2))+(60*self.reynoldsnumber**-1)
        self.helixloopdiameter+= self.regression_M_persec*Timestep*2
        if self.Is_pixel==True:
            self.helixNominalarea=((self.pixelAREA-((self.P_max_inscribed_diameter/2)**2*3.14))/(AmountofSmallCircles*0.5))
            self.p_helixNomminaldiameter=np.sqrt(self.helixNominalarea/3.14)*2
            self.p_helixloopdiameter=self.P_max_inscribed_diameter+(self.P_min_enclosing_diameter -self.P_max_inscribed_diameter)*0.5
            self.ratio_touchedByhelix=self.P_largest_arm_contact*AmountofSmallCircles/self.P_perimeter
            self.nottouchd_byoverlap=(self.P_max_inscribed_diameter*3.14/AmountofSmallCircles)-self.P_largest_arm_contact
            if self.time==Timestep:
                self.startinglength_inbetween_segment=helical_archsegment*(self.P_largest_arm_contact/(self.P_max_inscribed_diameter*3.14/AmountofSmallCircles))
           
            self.startinglength_inbetween_segment-=self.regression_M_persec_withratio*Timestep*2
           # self.lengthinbetween_2=(self.P_largest_arm_contact/(self.P_max_inscribed_diameter*3.14/AmountofSmallCircles))*helical_archsegment
            
            #self.triangehelicallength=np.sqrt((self.lengthinbetween_2)**2+( (self.P_min_enclosing_diameter-self.P_max_inscribed_diameter) *0.5)**2)
            self.ratio_forCF=(self.P_perimeter-AmountofSmallCircles*self.P_largest_arm_contact)/self.P_perimeter
            if self.time==Timestep:
                self.starttrianglevalue=self.triangehelicallength
        if Is_FuelGrain_GoshaStar==True:
            self.helixloopdiameter=self.bigcircleraduis*2+OusideSmallCircle_raduis
            self.helixNominalarea_star=((self.testArea-((self.bigcircleraduis)**2*3.14))/(AmountofSmallCircles*0.5))
            self.helixNominalDiameter=np.sqrt(self.helixNominalarea_star/3.14)*2
        else:
            self.helixloopdiameter=+ self.regression_M_persec_withratio*Timestep*2
            self.helixNominalDiameter=+ self.regression_M_persec_withratio*Timestep*2 # work this out later
        if self.Is_pixel==True:
                    self.helixloopdiameter=self.p_helixloopdiameter
        self.helixlength_meters=revPitch*np.sqrt(((self.helixloopdiameter*3.14)**2)+(PitchFor_Helix)**2)
        self.HelixP_meters=(fuelgrainlength/revPitch)
        


        
        self.RC=(self.helixloopdiameter/2)*(1+(self.HelixP_meters/(3.14*self.helixloopdiameter))**2)
        
        self.S_forhelix=self.totalrgession/2
        if self.Is_pixel==True:
            self.p_RC=(self.p_helixloopdiameter/2)*(1+(self.HelixP_meters/(3.14*self.p_helixloopdiameter))**2)
            #self.ratiofor_pRC=np.sqrt(1+((self.p_RC/(self.triangehelicallength*6.28))**2))
           # self.ratiofor_pRC=np.sqrt(1+3.14*((self.starttrianglevalue-self.triangehelicallength)/self.starttrianglevalue)**2)
            self.p_correctionfactor=(np.sqrt(1+6.28*(self.totalrgession/2*AmountofSmallCircles/self.p_RC)**2))
            #self.p_correctionfactor=(np.sqrt(1+(self.triangehelicallength*6.28/(self.p_RC))**2)) #fake
            self.p_correctionRC=self.p_correctionfactor*self.p_RC
        self.S_forhelix2=self.S_forhelix*AmountofSmallCircles
        self.Reg_Helix_Ratio=self.S_forhelix2/self.RC
        self.CorrectionRC=(np.sqrt(1+6.28*self.Reg_Helix_Ratio*self.Reg_Helix_Ratio))*self.RC
        self.RCattemp3=(self.helixloopdiameter/2)*(1+((self.HelixP_meters+self.S_forhelix)/(3.14*self.helixloopdiameter))**2)
        self.RCDIff=self.RC/self.CorrectionRC
        self.chamber_enthalpy=CEAforRocket.get_Chamber_H(Pc=self.chamberpressure_PSI, MR=self.OF)*2.326 #convert from btu/lb to kj/kg
        #if Is_2phase_flow_injector_model==True:
            #self.chamber_enthalpy=OxtankMath.downstream_enthalpy
        self.BlowingRatio=1+((self.p_helixloopdiameter/self.p_helixNomminaldiameter)/(self.chamber_enthalpy/self.enthapy_vaporpyrolsis_fuel))

        # Reynolds-free F_c calculation for the helix.
        # The first port radius is retained as r0; the hydraulic radius is R.
        port_radius = self.hydrolicdiamter / 2
        if not hasattr(self, "_helix_fc_reference_radius"):
            self._helix_fc_reference_radius = port_radius
        relative_roughness = sandgrainroughness / port_radius
        
        self.helix_fc = self.calculate_fc(
                epsilon=relative_roughness,
                r0=self._helix_fc_reference_radius,
                radius=self.CorrectionRC,
            )
      
        
               
        # Match TRY.PY: f_D = F_c for wall shear, while c_f for the normal
        # blowing equation is a separately supplied coefficient.
        self.cf_for_blowing = self.cfstraight
        surface_mass_flux = self.mdotfuel / self.insurfacearea
        self.helix_wallshear = self.helix_fc * (
            self.chamberDensity * self.streamvelocity ** 2
        ) / 8
        self.blowing_helix = (
            surface_mass_flux * self.streamvelocity / self.helix_wallshear
        )
        self.blowing_normal = surface_mass_flux / (
            self.chamberDensity * self.streamvelocity * self.cf_for_blowing / 2
        )
        self.newblowingratio=(self.blowing_normal/self.blowing_helix)**0.77
        self.ratio2=(self.helix_fc*0.25/self.cfstraight)**0.77
        self.CFhelix=self.cfstraight+0.0075*(np.sqrt(self.p_helixNomminaldiameter/(2*self.p_correctionRC)))
        if self.Is_pixel==True:
            self.p_NEWarclengthesitmate="WHOKNOWS"

        #if self.Newarclengthesitmate>0:
            #self.CFratio=(self.CFhelix*self.BlowingRatio/self.cfstraight)**((self.Newarclengthesitmate/self.onearchlengthestimate)**(1/5)) #figure out better way to do this
        #else:
            #self.CFratio=1
        #self.CFratio=((self.CFhelix*self.BlowingRatio/self.cfstraight)-1)*self.ratiofor_pRC+1
        self.CFratio=(((self.CFhelix*self.newblowingratio/self.cfstraight)-1)*self.ratio_forCF*self.ratio_forCF)+1
        if Is_fuelGrain_Helix==False:
            self.CFratio=1
    def Whole_thing_helix(self,CEAforRocket,expansionratio,thoart_area,Is_FuelGrain_GoshaStar,timestep,revPitch,PitchFor_Helix):
        self.fac_CR=(self.hydrolicdiamter*self.hydrolicdiamter*3.14/4)/thoart_area
        self.chambermach=CEAforRocket.get_Chamber_MachNumber(Pc=self.chamberpressure_PSI, MR=self.OF, fac_CR=self.fac_CR)
        self.chambersonicvelocity = CEAforRocket.get_SonicVelocities(Pc=self.chamberpressure_PSI, MR=self.OF, eps=expansionratio)[0]
        self.streamvelocity=self.chambersonicvelocity*self.chambermach
        self.chamberstuff=CEAforRocket.get_Chamber_Transport(Pc=self.chamberpressure_PSI, MR=self.OF, eps=expansionratio,frozen=0)
        self.viscosity= self.chamberstuff[1] * 0.0001# kg/m/s
        
        if self.Is_pixel==True:
            self.P_reynoldsnumber=(self.streamvelocity*self.hydrolicdiamter*self.chamberDensity)/self.viscosity
            self.reynoldsnumber=self.P_reynoldsnumber
        else:
            self.hydrolicdiamter=self.diameter_grain
            if self.time==timestep:
                self.OGHD=self.hydrolicdiamter
            self.reynoldsnumber=(self.streamvelocity*self.hydrolicdiamter*self.chamberDensity)/self.viscosity

        self.cfstraight=0.074/(self.reynoldsnumber**0.2)
        if self.Is_pixel==True:
            self.helixNomminaldiameter=self.hydrolicdiamter
        else:
            self.helixNomminaldiameter=+ self.regression_M_persec_withratio*timestep*2
        self.S_forhelix=self.totalrgession/2
        self.CorrectionRC=(np.sqrt(1+1.57*(((self.hydrolicdiamter-self.OGHD)/self.RC)**2)))*self.RC
        self.CFhelix=self.cfstraight+0.0075*(np.sqrt(self.helixNomminaldiameter/(2*self.CorrectionRC)))
        self.CFratio=self.CFhelix/self.cfstraight
        


        
            


        
        


        
        



    def complex_area_func(self,FuelGrainLength,Timestep,Is_fuelGrain_Helix,Is_FuelGrain_GoshaStar,AmountofSmallCircles,Fuel_Density): 
        self.eQraduis+=self.regression_M_persec*Timestep
        self.eQarea=self.eQraduis*self.eQraduis*3.14
        self.eQPerimeter=self.eQraduis*6.28
        self.complexSHapetotalReg=2*(self.eQraduis-self.eqradOG)
        self.smallcircleraduis+=self.regression_M_persec_withratio*Timestep
        self.bigcircleraduis+=self.regression_M_persec*Timestep
        self.complexarea= (self.smallcircleraduis*self.smallcircleraduis*3.14*0.5*self.circleamount)+(self.bigcircleraduis*self.bigcircleraduis*3.14)
        self.complexcircum=(self.bigcircleraduis*6.28)+(1.14*self.smallcircleraduis*self.circleamount)
        self.eqrad2=np.sqrt(self.complexarea/3.14)
        self.circumratio=1/(self.eqrad2*6.28/self.complexcircum)
        #self.circumratio=self.complexcircum/self.eQPerimeter
        self.arearatio=self.complexarea/self.eQarea
        
        self.flowrateRatio=(self.complexarea-self.oldarea)/(self.eQarea-self.oldarea)
        self.Newarclengthesitmate=((self.bigcircleraduis*6.28)-(self.smallcircleraduis*2*self.circleamount))/AmountofSmallCircles
        if self.Newarclengthesitmate > 0 :
            self.flowrateRatio_Use=((self.flowrateRatio)**(np.sqrt(self.Newarclengthesitmate/self.onearchlengthestimate)))
            if self.Is_pixel==True:
                self.arclengthdiffernce=(abs(1-(self.Newarclengthesitmate/self.P_largest_arm_contact)))+1
        else:
            self.flowrateRatio_Use=1
        self.ratioprecent=abs(1-self.flowrateRatio/self.flowrateRatio_Use)*100
        
        self.circleareachange=self.eQarea-self.oldarea
        self.complexarea_change=self.complexarea-self.oldarea
        self.anotherarea_ratio=((self.Newarclengthesitmate/self.onearchlengthestimate)**1)
        self.anotherarea_ratio2=((self.Newarclengthesitmate/self.onearchlengthestimate)*1)
        #self.testRad+=self.regression_M_persec_withratio*Timestep
        if self.anotherarea_ratio>0:
            self.Newstararea=self.Newstararea+(self.complexarea_change*self.anotherarea_ratio   ) + (self.circleareachange*(1-self.anotherarea_ratio2))
        else :
            self.Newstararea=self.Newstararea+self.circleareachange
        self.areaDiffprecent=abs(1-(self.Newstararea/self.pixelAREA))*100
        self.circle_circum_change=self.complexcircum-self.oldpermeter
        
        self.newstarpermeter=self.newstarpermeter+self.circle_circum_change*self.anotherarea_ratio+self.circle_circum_change*(1-self.anotherarea_ratio)
        self.mod_circcheck=(((self.circumratio-1)*self.anotherarea_ratio)+1) #rix this to go to 25%
        self.mod_areacheck=((self.arearatio-1)*self.anotherarea_ratio)+1
        self.testRad+=self.regression_M_persec_withratio*Timestep*self.mod_circcheck*1.065
        self.testArea=self.testRad*self.testRad*3.14#*self.mod_circcheck
        self.testCircum=self.testRad*6.28*self.mod_circcheck*1.0 # play with this a bit later
        self.anothermdot= (self.testCircum*Fuel_Density*FuelGrainLength*self.regression_M_persec_withratio) 
        self.modratio=self.Howcircle/self.mod_circcheck
        
        
        self.insurfacearea2=self.testCircum*FuelGrainLength*self.effectivelengthconstant
        self.anothermdot= self.insurfacearea2*Fuel_Density*self.regression_M_persec_withratio
        self.mdotdiff=abs(1-(self.anothermdot/self.mdotfuel))*100
        self.circumdiff=abs(1-(self.testCircum/self.P_perimeter))*100
        self.areadiff=abs(1-(self.testArea/self.pixelAREA))*100\
        


        self.eQraduis=np.sqrt(self.complexarea/3.14)
        self.areachange=self.complexarea- self.oldarea
        self.oldarea=self.complexarea
        self.oldpermeter=self.complexcircum
        self.arclength_rweclose=self.onearchlengthestimate-self.totalrgession
        if self.arclength_rweclose>0:
            self.ratiohelp=self.arclength_rweclose/self.onearchlengthestimate
            
        if Is_fuelGrain_Helix==True:
            self.surfaceareaadd=0
        else:
            self.surfaceareaadd=0


    def transeint_chamber_model(self,NozzleCD_ForChamberPressure,Ncomb_ForChamberPressure,NozzleMath,Fuel_Density,Timestep,Is_fuelgrain_Transient,ambientpressure_pa):
        if Is_fuelgrain_Transient==True:
            
            self.SpecR=8314/self.MW_chamber
            self.throatcrit_ratio=(2/(self.gamma_throat+1))**((self.gamma_throat+1)/(2*(self.gamma_throat-1)))
            self.realthroatratio=ambientpressure_pa/self.newchamberpressure_PA
            if self.realthroatratio>self.throatcrit_ratio:
                self.Chocked_status=False
            else:
                self.Chocked_status=True
            self.newcstar= self.cstar*Ncomb_ForChamberPressure
            self.mdotgen=self.totalmdot
            self.mdotnozzle= NozzleCD_ForChamberPressure* self.newchamberpressure_PA*NozzleMath.throatarea/ self.newcstar
            self.mdotgain=self.mdotgen-self.mdotnozzle
            self.volumeflowrate=((self.mdotfuel)/Fuel_Density)
            self.volumeflowrate2=self.volume_grain-self.oldvolumegrain
            self.oldvolumegrain=self.volume_grain
            self.massGas=self.massGas+(self.mdotgain*Timestep)
            self.deltaPressure_PA=self.newchamberpressure_PA*((self.mdotgain/self.massGas)-(self.volumeflowrate/self.volume_grain))
            self.newchamberpressure_PA=self.newchamberpressure_PA+self.deltaPressure_PA*Timestep
            self.newchamberpressure_psi=self.newchamberpressure_PA/6894.76
            self.sum_mdot=self.sum_mdot+(self.mdotnozzle*Timestep)
    
    def complexregression(self,CEAforRocket,FuelDensity,FuelGrainLength):
        self.chamber_Cp=CEAforRocket.get_Chamber_Cp(Pc=self.chamberpressure_PSI, MR=self.OF)*4.184 #convert from btu/lb-R to kJ/kg-K
        self.walltemp_kelvin=725 #estimate
        self.temp_kelvin=[(t/1.8) for t in CEAforRocket.get_Temperatures(Pc=self.chamberpressure_PSI, MR=self.OF)]
        self.chambertemp_kelvin=self.temp_kelvin[1]
        self.layerchangecon=0.8 # estimate how much less 
        self.deltaEnthapy=self.chamber_Cp*(self.chambertemp_kelvin*self.layerchangecon-self.walltemp_kelvin)
        self.heatof_gassification=350*1000 #kJ/kg estimate
        self.BlowingCo=self.deltaEnthapy/self.heatof_gassification
        self.use_blowingco=self.BlowingCo**0.23
        self.chamber_stuff=CEAforRocket.get_Chamber_Transport(Pc=self.chamberpressure_PSI, MR=self.OF, eps=1, frozen=0)
        self.prandtl_number=self.chamber_stuff[3]
        self.prandtl_number=1.0 # fudge this to see if it helps with the regression rate
        self.viscosity_forweirdreg= self.chamber_stuff[1] * 0.0001# kg/m/s
        self.fuelgrain_length=FuelGrainLength
        a=1
        n=0.8
        m=1-n
        p1=a/n
        p2=0.0376/((self.prandtl_number**(2/3))*FuelDensity)
        p3=(self.enthapy_vaporpyrolsis_fuel/self.fuelgrain_length)**(m)
        p4=self.oxflux*m
        p5=(m*4*a/n)/(n*(2-m))
        p6=0.0376/((self.prandtl_number**(2/3)))
        p7=(self.viscosity_forweirdreg**m)/(self.hydrolicdiamter*self.fuelgrain_length**n)
        p8=n/m
        p9=(p1*p2*self.use_blowingco*p3)
        p10=(p4+self.use_blowingco*p5*p6*p7)**p8
        self.regression2=p9*p10
        self.maybe_A=(a/n)*(0.0376/((self.prandtl_number**(2/3))*FuelDensity))*self.use_blowingco*self.viscosity_forweirdreg**m
        self.streamvelocity=np.sqrt(2*((self.chamberpressure_PA-self.throatpressure_PA)/self.chamberDensity))
        self.reynoldsnumber_forReg=(self.streamvelocity*self.hydrolicdiamter*self.chamberDensity)/self.viscosity_forweirdreg
        self.cfstraight_forreg=0.0776*(((np.log10(self.reynoldsnumber_forReg)-1.88)**-2))+(60*self.reynoldsnumber_forReg**-1)
        self.reg3=((0.635*self.oxflux)/(self.prandtl_number**(2/3)*FuelDensity))*self.use_blowingco*self.cfstraight_forreg





        
    
        
        

        

    




    
    







    


    # =========================================================================
    # LEGACY 2-D GEOMETRY LOADER (fuel_grain_regression2.py)
    # =========================================================================
    def _load_obj_file_geometry_legacy(self, obj_file_path, outer_diameter_inches=5.0, resolution=500, cross_section_axis=2, cross_section_pos=None):
        """
        Load an OBJ file and extract starting area, inner port area, and perimeter.
        Uses FuelGrainRegressionSimulator from fuel_grain_regression.py
        
        Args:
            obj_file_path: Path to the OBJ file containing the 3D fuel grain geometry
            outer_diameter_inches: Outer diameter of the fuel grain in inches
            resolution: Grid resolution for the simulation (pixels)
            cross_section_axis: Which axis to take cross-section perpendicular to (0=X, 1=Y, 2=Z)
            cross_section_pos: Position along axis for cross-section (None = center)
            
        Returns:
            Dictionary with extracted properties:
            {
                'starting_area_mm2': Fuel grain cross-section area (mm²),
                'inner_perimeter_mm': Inner port perimeter (mm),
                'outer_perimeter_mm': Outer perimeter (mm),
                'inner_port_diameter_mm': Characteristic inner port diameter (mm),
                'outer_diameter_mm': Outer diameter (mm),
                'center_x_mm': Center X coordinate (mm),
                'center_y_mm': Center Y coordinate (mm)
            }
        """
        # Create simulator instance
        simulator = FuelGrainRegressionSimulator(
            obj_file_path, 
            outer_diameter_inches=outer_diameter_inches,
            resolution=resolution, 
            cross_section_axis=cross_section_axis, 
            cross_section_pos=cross_section_pos
        )
        
        # Load geometry
        section_points_2d = simulator.read_obj_geometry()
        X, Y = simulator.create_initial_grid(section_points_2d)
        
        # Calculate area and perimeter
        pixel_width = abs(X[0, 1] - X[0, 0])
        pixel_height = abs(Y[1, 0] - Y[0, 0])
        pixel_area = pixel_width * pixel_height
        
        # Area of fuel grain (in mm²)
        fuel_area = np.sum(simulator.grid) * pixel_area
        
        # Find the inner hole and calculate perimeters
        empty_space = 1 - simulator.grid
        labeled, num_features = ndimage.label(empty_space)
        border_label = labeled[0, 0]
        
        # Calculate outer fuel grain perimeter
        outer_perimeter = measure.perimeter(simulator.grid) * pixel_width
        
        # Calculate inner port perimeter
        inner_perimeter = 0
        for label in range(1, num_features + 1):
            if label != border_label:
                inner_hole_mask = (labeled == label)
                # Calculate inner hole perimeter (boundary of hole)
                perimeter_pixels = measure.perimeter(inner_hole_mask)
                inner_perimeter = perimeter_pixels * pixel_width
                break
        
        # Inner port diameter (using characteristic radius)
        inner_port_diameter = 2 * simulator.id_radius
        
        results = {
            'starting_area_mm2': fuel_area,
            'inner_perimeter_mm': inner_perimeter,
            'outer_perimeter_mm': outer_perimeter,
            'inner_port_diameter_mm': inner_port_diameter,
            'outer_diameter_mm': 2 * simulator.od_radius,
            'center_x_mm': simulator.center[0],
            'center_y_mm': simulator.center[1]
        }
        
        print(f"\nExtracted Geometry Properties:")
        print(f"  Starting Area: {fuel_area:.2f} mm²")
        print(f"  Inner Port Perimeter: {inner_perimeter:.2f} mm")
        print(f"  Outer Perimeter: {outer_perimeter:.2f} mm")
        print(f"  Inner Port Diameter: {inner_port_diameter:.4f} mm")
        print(f"  Outer Diameter: {2 * simulator.od_radius:.4f} mm")
        
        return results

    # =========================================================================
    # ACTIVE 3-D GEOMETRY LOADER (attempyforfgreg.py)
    # =========================================================================
    def load_obj_file_geometry(self, obj_file_path, outer_diameter_inches=5.0, resolution=500,
                               cross_section_axis=2, cross_section_pos=None):
        """Return initial selected-plane geometry from the 3-D regression rasterizer."""
        if cross_section_pos is not None:
            raise NotImplementedError("The 3-D loader currently measures the centre slice only")
        obj_path = regression_3d.Path(obj_file_path)
        mesh = regression_3d.load_mesh(obj_path, regression_3d.infer_unit_scale_to_mm(obj_path, None))
        solid, pitch = regression_3d.make_solid_voxels(mesh, cross_section_axis, resolution, 200)
        port = regression_3d.central_port_mask(solid, cross_section_axis)
        index = solid.shape[cross_section_axis] // 2
        port_area, inner_perimeter = regression_3d.slice_metrics(port, solid, cross_section_axis, index, pitch)
        plane_axes = [number for number in range(3) if number != cross_section_axis]
        fuel_area = float(np.count_nonzero(np.take(solid, index, axis=cross_section_axis)) *
                          np.prod(pitch[plane_axes]))
        outer_perimeter = regression_3d.slice_metrics(~np.take(solid, index, axis=cross_section_axis)[None, ...],
                                                       np.take(solid, index, axis=cross_section_axis)[None, ...],
                                                       0, 0, np.array([1.0, *pitch[plane_axes]]))[1]
        results = {
            "starting_area_mm2": fuel_area,
            "inner_perimeter_mm": inner_perimeter,
            "outer_perimeter_mm": outer_perimeter,
            "inner_port_diameter_mm": 4.0 * port_area / inner_perimeter if inner_perimeter else 0.0,
            "outer_diameter_mm": float(max(mesh.extents[plane_axes])),
            "center_x_mm": float(np.mean(mesh.bounds[:, plane_axes[0]])),
            "center_y_mm": float(np.mean(mesh.bounds[:, plane_axes[1]])),
        }
        print(f"3-D initial geometry: port area {port_area:.2f} mm^2; inner perimeter {inner_perimeter:.2f} mm")
        return results

    def add_values(self):
        row = {}
        for attr in self._init_attrs:  
            val = getattr(self, attr)
            # Convert numpy arrays and lists to JSON strings for proper CSV serialization
            if isinstance(val, np.ndarray):
                row[attr] = json.dumps(val.tolist())
            elif isinstance(val, list):
                row[attr] = json.dumps(val)
            else:
                row[attr] = val
        self._output_rows.append(row)
    
    def finalize_output(self):
        """Convert buffered rows to DataFrame. Call this before accessing outputcsv."""
        if self._output_rows:
            self.outputcsv = pd.DataFrame(self._output_rows)
            self._output_rows = []
    
    # ========== PIXEL-BASED FUEL GRAIN REGRESSION METHODS ==========
    
    #finish gosha start
    #correct helix math
    # do the with or withou ratio for regression
    # look into x^m for regression
    # explore if temp ncom realtionship
