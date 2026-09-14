import time as time_module
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as scp
import rocketcea as rcea
from rocketcea import py_cea
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
import surfacearea


@dataclass
class RegressionSlot:
    area: float
    distance: float
    totalregression: float
    peremeter: float
    Diameter: float
    prev_regression: float
    totalflux: float = 0.0
    fuelflux: float =0.0
    OF: float = 0.0
    mdotfuel: float = 0.0
    Nominal_port_D: float = 0.0
    Helix_loop_d: float = 0.0
    surfacearea: float = 0.0


class CEALookupTable:
    """Persistent bilinear lookup table for complex-regression CEA states."""

    def __init__(self, cea, filename, pressure_range_psi, of_range, pressure_points, of_points):
        self.filename=Path(filename)
        self.minimum_requested_pressure=pressure_range_psi[0]
        # RocketCEA requires a positive pressure.  Keep the public lookup range
        # at 0..Pmax, but evaluate the zero-pressure endpoint at 0.1 psi.
        minimum_cea_pressure=max(pressure_range_psi[0], 0.1)
        self.pressures=np.linspace(minimum_cea_pressure, pressure_range_psi[1], pressure_points)
        self.of_values=np.linspace(*of_range, of_points)
        self.signature=f"{cea.oxName}/{cea.fuelName}"
        if not self._load():
            self._build(cea)

    def _load(self):
        if not self.filename.exists():
            return False
        try:
            with np.load(self.filename, allow_pickle=False) as data:
                if (str(data["signature"]) != self.signature
                        or not np.array_equal(data["pressures"], self.pressures)
                        or not np.array_equal(data["of_values"], self.of_values)):
                    return False
                self.temperature=data["temperature"]
                self.molecular_weight=data["molecular_weight"]
                self.prandtl=data["prandtl"]
                self.cp=data["cp"]
            return True
        except (OSError, KeyError, ValueError):
            return False

    def _build(self, cea):
        shape=(len(self.pressures), len(self.of_values))
        self.temperature=np.empty(shape)
        self.molecular_weight=np.empty(shape)
        self.prandtl=np.empty(shape)
        self.cp=np.empty(shape)
        print(f"Building CEA lookup table ({shape[0]} x {shape[1]}) at {self.filename}...")
        for pressure_index, pressure in enumerate(self.pressures):
            for of_index, of_value in enumerate(self.of_values):
                cp, _, _, prandtl=cea.get_Chamber_Transport(
                    Pc=pressure, MR=of_value, eps=1, frozen=0
                )
                chamber_index=cea.i_chm
                molecular_weight=py_cea.prtout.wm[chamber_index]
                try:
                    molecular_weight=1.0/py_cea.prtout.totn[chamber_index]
                except Exception:
                    pass
                self.temperature[pressure_index, of_index]=py_cea.prtout.ttt[chamber_index]*1.8*0.555556
                self.molecular_weight[pressure_index, of_index]=molecular_weight*0.45359237
                self.prandtl[pressure_index, of_index]=prandtl
                self.cp[pressure_index, of_index]=cp
        np.savez_compressed(
            self.filename,
            signature=np.asarray(self.signature), pressures=self.pressures,
            of_values=self.of_values, temperature=self.temperature,
            molecular_weight=self.molecular_weight, prandtl=self.prandtl, cp=self.cp,
        )
        print(f"Saved CEA lookup table to {self.filename}.")

    def state(self, pressure, of_value):
        """Return (T, molecular weight kg/mol, Prandtl, Cp), or None outside the table."""
        if not (self.minimum_requested_pressure <= pressure <= self.pressures[-1]
                and self.of_values[0] <= of_value <= self.of_values[-1]):
            return None

        pressure=max(pressure, self.pressures[0])

        def interpolate(values):
            pressure_index=np.clip(np.searchsorted(self.pressures, pressure)-1, 0, len(self.pressures)-2)
            of_index=np.clip(np.searchsorted(self.of_values, of_value)-1, 0, len(self.of_values)-2)
            p0, p1=self.pressures[pressure_index], self.pressures[pressure_index+1]
            o0, o1=self.of_values[of_index], self.of_values[of_index+1]
            pressure_fraction=(pressure-p0)/(p1-p0)
            of_fraction=(of_value-o0)/(o1-o0)
            return ((1-pressure_fraction)*(1-of_fraction)*values[pressure_index, of_index]
                    + pressure_fraction*(1-of_fraction)*values[pressure_index+1, of_index]
                    + (1-pressure_fraction)*of_fraction*values[pressure_index, of_index+1]
                    + pressure_fraction*of_fraction*values[pressure_index+1, of_index+1])

        return tuple(interpolate(values) for values in (
            self.temperature, self.molecular_weight, self.prandtl, self.cp,
        ))

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
        self.fuel_grain_length_m=float(FuelGrainLength)
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
        self.port_volume_coeffs=None
        # Retained only for optional diagnostics; runtime geometry is evaluated
        # from the selected polynomial equations.
        self._pixel_geometry_samples=None
        self._previous_pixel_port_volume_m3=None
        self.BlowingCo=0
        self.use_blowingco=0
        self.prandtl_number=None
        self.viscosity_forweirdreg=None
        self.totalrgession=0
        self.totalregression_MM=0
        self.oxflux=None
        self.mdotfuel=0
        self.sum_modtfuel=0
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
            self.extra_volume=preccandpostvolume
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
        
        self.insurfacearea=self.portperimeter*self.effectivelengthconstant*FuelGrainLength
        self.diameter_grain=self.diameter_grain+(self.regression_M_persec*Timestep*2)
        self.fuelgrain_raduis=self.diameter_grain/2
        self.area_grain=(self.fuelgrain_raduis*self.fuelgrain_raduis*3.14)
        self.volume_grain=self.area_grain*FuelGrainLength
        if Is_FuelGrain_GoshaStar==True:
            self.area_grain=self.eQarea
            self.insurfacearea=self.complexcircum*FuelGrainLength*self.effectivelengthconstant
        if self.Is_pixel==True:
            self.area_grain=self.pixelAREA
            if getattr(self, "pixel_geometry_mode", "slice") == "slice":
                self.insurfacearea=self.P_perimeter*self.pixel_geometry_length_m
        if Is_fuelGrain_Helix==True:
            self.surfaceareaadd=0
            self.addtoarea_bcHelix=0
                         
        if not (self.Is_pixel and getattr(self, "pixel_geometry_mode", "slice") == "three_d"):
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
    def pixelmethodgeo(self,Is_fuelGrain_Helix,FuelGrainLength,Fuel_Density,Timestep,amountofsmallcircles,effectivelengthconstant,Override_effectivelengthconstant,pixel_geometry_mode="slice",slice_length_m=None):
        """
        Calculate pixel method geometry using polynomial curve fits from regression analysis.
        Uses 2nd-degree polynomial fits for area and perimeter extracted from the OBJ geometry.
        """
        if pixel_geometry_mode not in ("slice", "three_d"):
            raise ValueError("pixel_geometry_mode must be 'slice' or 'three_d'")
        self.pixel_geometry_mode=pixel_geometry_mode
        self.pixel_geometry_length_m=(
            FuelGrainLength if slice_length_m is None else float(slice_length_m)
        )
        if self.pixel_geometry_length_m <= 0:
            raise ValueError("slice_length_m must be positive")

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
        # Evaluate the selected area equation.  The 3-D analysis chooses the
        # lowest suitable degree (2--5), rather than always forcing degree 5.
        area_mm2=float(np.polyval(area_coeffs, x))
        self.pixelAREA=max(0.0, area_mm2)
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
        
        self.effectivelengthconstant=effectivelengthconstant
        if pixel_geometry_mode == "three_d" and self.surface_area_coeffs is not None and self.port_volume_coeffs is not None:
            surface_area_mm2=float(np.polyval(self.surface_area_coeffs, x))
            port_volume_mm3=float(np.polyval(self.port_volume_coeffs, x))
            self.P_surfacearea=max(0.0, surface_area_mm2) / 1_000_000.0
            self.volume_grain=max(0.0, port_volume_mm3) / 1_000_000_000.0
        else:
            self.volume_grain=self.pixelAREA*self.pixel_geometry_length_m
        self.insurfacearea=self.P_surfacearea
        if self.initial_surface_area is None:
            self._store_initial_surface_area(self.P_surfacearea)
        self.volume_grain=self.volume_grain*self.effectivelengthconstant
        self.insurfacearea=self.insurfacearea*self.effectivelengthconstant
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
        # Fuel generation comes from the derivative of the active geometry
        # equation, evaluated at the current regression distance.  The fitted
        # independent variable is mm, so convert mm^2/s or mm^3/s to SI.
        regression_rate_mm_s=self.regression_MM_persec_withratio
        if regression_rate_mm_s is not None:
            if pixel_geometry_mode == "slice":
                darea_dr_mm2_per_mm=float(np.polyval(np.polyder(area_coeffs), x))
                darea_dt_m2_per_s=darea_dr_mm2_per_mm*regression_rate_mm_s/1_000_000.0
                self.p_mdotfuel=darea_dt_m2_per_s*self.pixel_geometry_length_m*Fuel_Density
            elif self.port_volume_coeffs is not None:
                dvolume_dr_mm3_per_mm=float(np.polyval(np.polyder(self.port_volume_coeffs), x))
                dvolume_dt_m3_per_s=dvolume_dr_mm3_per_mm*regression_rate_mm_s/1_000_000_000.0
                self.p_mdotfuel=dvolume_dt_m3_per_s*Fuel_Density
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
        with regression_3d.activity_indicator("Identifying the central port..."):
            initial_port = regression_3d.central_port_mask(solid, cross_section_axis)
        if snapshot_regression_mm is not None:
            regression_3d.export_regressed_fuel_snapshot(
                solid, initial_port, pitch, float(snapshot_regression_mm),
                regression_3d.Path(snapshot_path or "regression_outputs/regressed_fuel_snapshot.obj"),
            )
        max_regression_mm = regression_rate * time_seconds
        distances = np.linspace(0.0, max_regression_mm, graph_points)
        with regression_3d.activity_indicator("Measuring the initial OBJ bore surface..."):
            initial_bore_area=regression_3d.mesh_bore_surface_area(mesh, cross_section_axis)
        surface_rows = regression_3d.regression_table(
            solid, initial_port, pitch, cross_section_axis, distances,
            mesh_volume=float("nan"), mesh_area=float(mesh.area),
            initial_bore_area=initial_bore_area,
            use_triangulated_bore_surface_area=(
                regression_3d.USE_TRIANGULATED_BORE_SURFACE_AREA
            ),
        )
        plane_rows = regression_3d.picked_plane_regression(
            solid, initial_port, pitch, cross_section_axis, distances,
            surface_area_rows=surface_rows,
            surface_area_ratio_grain_length_mm=self.fuel_grain_length_m*1000.0,
        )
        x = distances

        # Match the selected-plane curve-fit logic: fit only the intact-fuel
        # points.  Including the burn-through point and its later flat values
        # forces a higher-degree polynomial and can make the port-area equation
        # bend downward near zero regression.
        plane_burnthrough_indices=np.flatnonzero(np.asarray(
            [row["burnthrough"] for row in plane_rows], dtype=bool
        ))
        if len(plane_burnthrough_indices):
            plane_fit_rows=plane_rows[:int(plane_burnthrough_indices[0])]
        else:
            plane_fit_rows=plane_rows

        # Whole-grain port volume becomes flat after all fuel is gone.  Those
        # post-exhaustion samples are useful for plots, but must not be part of
        # the operating geometry equation: they distort its derivative and can
        # produce negative dV/dr at the start of the burn.
        whole_grain_exhaustion_indices=np.flatnonzero(np.asarray(
            [row["remaining_fuel_volume_mm3"] <= 0.0 for row in surface_rows], dtype=bool
        ))
        if len(whole_grain_exhaustion_indices) and int(whole_grain_exhaustion_indices[0]) >= 3:
            whole_grain_fit_rows=surface_rows[:int(whole_grain_exhaustion_indices[0])]
        else:
            whole_grain_fit_rows=surface_rows

        selected_fit_degrees={}

        def fit_metric(rows, key):
            fit_x=np.asarray([row["regression_mm"] for row in rows], dtype=float)
            values = np.asarray([row[key] for row in rows], dtype=float)
            polynomial, degree, _ = regression_3d.polynomial_fit(fit_x, values)
            selected_fit_degrees[key]=degree
            return polynomial.c

        self.area_coeffs = fit_metric(plane_fit_rows, "port_area_mm2")
        self.perimeter_coeffs = fit_metric(plane_fit_rows, "port_perimeter_mm")
        self.max_inscribed_coeffs = fit_metric(plane_fit_rows, "max_inscribed_diameter_mm")
        self.min_enclosing_coeffs = fit_metric(plane_fit_rows, "min_enclosing_diameter_mm")
        self.circle_overlap_coeffs = fit_metric(plane_fit_rows, "circle_boundary_overlap_mm")
        self.largest_arm_coeffs = fit_metric(plane_fit_rows, "largest_arm_contact_mm")
        self.surface_area_coeffs = fit_metric(whole_grain_fit_rows, "burning_surface_area_mm2")
        self.port_volume_coeffs = fit_metric(whole_grain_fit_rows, "port_volume_mm3")
        self._previous_pixel_port_volume_m3=None
        self._store_initial_surface_area(float(surface_rows[0]["burning_surface_area_mm2"]) / 1_000_000.0)

        print("3-D regression geometry loaded into FuelGrain:")
        print(f"  plane: {('X', 'Y', 'Z')[cross_section_axis]}; grid: {tuple(solid.shape)}")
        print(f"  fitted range: 0 to {max_regression_mm:.3f} mm")
        if len(whole_grain_exhaustion_indices):
            print("  whole-grain geometry fit ends before fuel exhaustion at "
                  f"{surface_rows[int(whole_grain_exhaustion_indices[0])]['regression_mm']:.3f} mm")
        print("  selected polynomial degrees: " + ", ".join(
            f"{key}={degree}" for key, degree in selected_fit_degrees.items()
        ))
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
        # Fuel mass generation is the bore surface area swept by the radial
        # regression rate: mdot_fuel = A_bore * r_dot * rho_fuel.  Use this
        # consistently for pixel and analytic grain geometries.
        self.mdotfuel=(
            self.insurfacearea
            * self.regression_M_persec_withratio
            * Fuel_Density
        )
        self.sum_modtfuel=self.sum_modtfuel+(self.mdotfuel*Timestep)
        

        
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
        self.streamvelocity=self.chambersonicvelocity*self.chambermach*0.3048
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
            self.helixloopdiameter+= self.regression_M_persec_withratio*Timestep*2
            self.helixNominalDiameter+= self.regression_M_persec_withratio*Timestep*2 # work this out later
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
        self.streamvelocity=self.chambersonicvelocity*self.chambermach*0.3048
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
            if self.time>Timestep:
                self.delatTemp=(self.chamber_temp-self.oldchambertemp)/Timestep
                self.deltaSpecR=(self.SpecR-self.oldSpecR)/Timestep
            else:
                self.delatTemp=0
                self.deltaSpecR=0

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
            self.chambervolume=self.volume_grain+self.extra_volume
            self.massGas=self.massGas+(self.mdotgain*Timestep)
            self.deltaPressure_PA=self.newchamberpressure_PA*((self.mdotgain/self.massGas)-(self.volumeflowrate/self.chambervolume))#+(self.delatTemp/self.chamber_temp)+(self.deltaSpecR/self.SpecR))
            self.newchamberpressure_PA=self.newchamberpressure_PA+self.deltaPressure_PA*Timestep
            self.newchamberpressure_psi=self.newchamberpressure_PA/6894.76
            self.sum_mdot=self.sum_mdot+(self.mdotnozzle*Timestep)
            self.oldSpecR=self.SpecR
            self.oldchambertemp=self.chamber_temp
    
    def complexregression(self, CEAforRocket, FuelDensity, FuelGrainLength,
                          amount_of_slots, timestep, OFstartguess, mdotox,
                          Do_complex_regression=False,
                          Print_complex_regression=False,
                          Use_cea_lookuptable=False,
                          cea_lookup_filename="cea_complex_regression.npz",
                          cea_lookup_pressure_range_psi=(0.0, 1500.0),
                          cea_lookup_of_range=(0.0, 30.0),
                          cea_lookup_pressure_points=31,
                          cea_lookup_of_points=61,
                          usecomplexregression=False,
                          use_pixel_geometry=None):
        """Near-verbatim port of hardregresstionattemp.py from line 214.

        Variable names and the original calculation order are intentionally
        retained so they can be mapped to this class manually.
        """
        if not Do_complex_regression:
            return []

        amount_of_sections=amount_of_slots-1
        # ``totalregression`` is the radial change from the initial local
        # diameter, so both quantities must start at zero deformation.
        startreg=0.0
        lengthinFG=FuelGrainLength
        portD=self.diameter_grain
        # The pixel method supplies selected-plane area/perimeter equations as
        # functions of radial regression.  Complex regression has independent
        # axial slots, so evaluate that same cross-section for each slot's own
        # local regression distance.
        # None preserves automatic selection for existing callers. An explicit
        # False selects the built-in circular formulas even in pixel mode.
        use_pixel_slot_geometry=(
            self.Is_pixel if use_pixel_geometry is None else use_pixel_geometry
        )
        if use_pixel_slot_geometry and (
            self.area_coeffs is None or self.perimeter_coeffs is None
        ):
            raise RuntimeError(
                "Pixel complex regression requires pixel geometry coefficients. "
                "Run run_regression_analysis() before complexregression()."
            )

        def pixel_slot_geometry(regression_m):
            regression_mm=regression_m*1000.0
            area_m2=max(0.0, float(np.polyval(self.area_coeffs, regression_mm))) / 1_000_000.0
            perimeter_m=max(0.0, float(np.polyval(self.perimeter_coeffs, regression_mm))) / 1000.0
            # The complex-regression transport correlations need a flow
            # diameter, not the diameter of a circle with the same area.
            # For a non-circular pixel port, use hydraulic diameter, D_h=4A/P.
            hydraulic_diameter_m=(4.0*area_m2/perimeter_m) if perimeter_m>0 else 0.0
            return area_m2, perimeter_m, hydraulic_diameter_m

        if use_pixel_slot_geometry:
            _, _, portD=pixel_slot_geometry(startreg)
        fuelden=FuelDensity
        massflowox=mdotox
        chamber_pressure_psi=self.chamberpressure_PSI
        Nozzle_expansion_ratio=1
        moleweightN20=44.013
        sigma_hardspherediameter=5
        ref_temp_H=298
        Hg_frompaper=1812
        koX=1
        activation_energy=203*1000
        eachsectionlength=lengthinFG/amount_of_sections
        lookup_table=None
        if Use_cea_lookuptable:
            lookup_key=(
                str(cea_lookup_filename), tuple(cea_lookup_pressure_range_psi),
                tuple(cea_lookup_of_range), cea_lookup_pressure_points,
                cea_lookup_of_points, CEAforRocket.oxName, CEAforRocket.fuelName,
            )
            if getattr(self, "_cea_lookup_key", None) != lookup_key:
                self._cea_lookup_table=CEALookupTable(
                    CEAforRocket, cea_lookup_filename, cea_lookup_pressure_range_psi,
                    cea_lookup_of_range, cea_lookup_pressure_points, cea_lookup_of_points,
                )
                self._cea_lookup_key=lookup_key
            lookup_table=self._cea_lookup_table
            if getattr(self, "_cea_lookup_announced_key", None) != lookup_key:
                print(
                    "Using CEA lookup table: "
                    f"{cea_lookup_filename} "
                    f"(Pc {cea_lookup_pressure_range_psi[0]:g}–{cea_lookup_pressure_range_psi[1]:g} psi, "
                    f"O/F {cea_lookup_of_range[0]:g}–{cea_lookup_of_range[1]:g})."
                )
                self._cea_lookup_announced_key=lookup_key
        # CEA calls dominate runtime.  Cache exact state queries for this
        # timestep; no pressure/O/F rounding is used, so this does not alter
        # any regression inputs or equations.
        cea_state_cache={}

        def get_local_cea_state(pressure, OF, expansion_ratio=1):
            key=(pressure, OF, expansion_ratio)
            if key not in cea_state_cache:
                if lookup_table is not None:
                    lookup_state=lookup_table.state(pressure, OF)
                    if lookup_state is not None:
                        cea_state_cache[key]=lookup_state
                        return cea_state_cache[key]
                # get_Chamber_Transport performs one CEA solve.  Its result
                # leaves the exact chamber temperature and molecular-weight
                # fields in RocketCEA's common Fortran output block, so do
                # not make a second get_IvacCstrTc_ChmMwGam solve for them.
                cp, _, _, prandtl_number=CEAforRocket.get_Chamber_Transport(
                    Pc=pressure, MR=OF, eps=expansion_ratio, frozen=0
                )
                chamber_index=CEAforRocket.i_chm
                temperature_rankine=py_cea.prtout.ttt[chamber_index]*1.8
                molecular_weight=py_cea.prtout.wm[chamber_index]
                try:
                    molecular_weight=1.0/py_cea.prtout.totn[chamber_index]
                except Exception:
                    pass
                cea_state_cache[key]=(
                    temperature_rankine*0.555556,
                    molecular_weight*0.45359237,
                    prandtl_number,
                    cp,
                )
            return cea_state_cache[key]

        def make_reg_slots(amount_of_slots=amount_of_slots, grain_length=lengthinFG, port_diameter=portD, start_regression=startreg):
            slots=[]
            for x in range(amount_of_slots):
                if amount_of_slots <= 1:
                    length = 0
                else:
                    length=x*(grain_length/amount_of_sections)
                if use_pixel_slot_geometry:
                    area, peremeter, Diameter=pixel_slot_geometry(start_regression)
                else:
                    Diameter=port_diameter
                    area=Diameter*Diameter*np.pi/4
                    peremeter=Diameter*np.pi
                slot=RegressionSlot(area=area,distance=length,totalregression=start_regression,peremeter=peremeter,Diameter=Diameter,prev_regression=start_regression)
                slot.initial_diameter=Diameter
                slots.append(slot)
            return slots

        def update_slot_geometry(slot):
            """Update a slot after its local regression increment."""
            if use_pixel_slot_geometry:
                slot.area, slot.peremeter, slot.Diameter=pixel_slot_geometry(slot.totalregression)
            else:
                slot.Diameter += 2*slot.prev_regression*timestep
                slot.area = slot.Diameter*slot.Diameter*np.pi/4
                slot.peremeter = slot.Diameter*np.pi

        def get_constant_values(pressure,stoicOF):
            if lookup_table is None:
                temp_of_flame=(CEAforRocket.get_Temperatures(Pc=pressure, MR=stoicOF, eps=1, frozen=0)[0])*0.555556
                temp_of_surface=(CEAforRocket.get_Temperatures(Pc=chamber_pressure_psi, MR=0, eps=1, frozen=0)[0])*0.555556
                specific_heat_surface=(CEAforRocket.get_Chamber_Cp(Pc=pressure, MR=0, eps=1)*4.184)
                specific_heat_flame=(CEAforRocket.get_Chamber_Cp(Pc=pressure, MR=stoicOF, eps=1)*4.184)
            else:
                flame_state=get_local_cea_state(pressure,stoicOF)
                surface_state=get_local_cea_state(chamber_pressure_psi,0.0)
                temp_of_flame=flame_state[0]
                temp_of_surface=surface_state[0]
                specific_heat_surface=surface_state[3]*4.184
                specific_heat_flame=flame_state[3]*4.184
            deltaH=specific_heat_flame*(temp_of_flame-ref_temp_H)-specific_heat_surface*(temp_of_surface-ref_temp_H)
            enthapy_ratio=(deltaH/Hg_frompaper)
            Phi_C=(1.22*stoicOF*enthapy_ratio)/(koX+(stoicOF+koX)*enthapy_ratio)
            k_constant=(-0.005*(activation_energy/(8.314*temp_of_surface)))-0.08
            return [deltaH,temp_of_surface,temp_of_flame,enthapy_ratio,Phi_C,k_constant]

        def get_temps(pressure,OF,list_of_constants):
            temp_cea_local=get_local_cea_state(pressure,OF)[0]
            bulktemp_local=(temp_cea_local+list_of_constants[1]+list_of_constants[2])/3
            tempratio_local=(list_of_constants[1]/bulktemp_local)
            return [temp_cea_local,bulktemp_local,tempratio_local]

        def get_dynamic_viscosity(pressure,OF,bulktemp):
            moleweightmixture=get_local_cea_state(pressure,OF)[1]
            meanmoleweight=(moleweightN20+moleweightmixture)/2
            return 26.69*(np.sqrt(meanmoleweight*bulktemp/(sigma_hardspherediameter**2)))*10**-7

        def get_Acon_for_regression_slot(Local_OF,Chamberpressure,Diameter,x_inlength,List_of_constants,viscosity,tempratio,Nozzle_expansion_ratio=Nozzle_expansion_ratio):
            prandtl_number=get_local_cea_state(Chamberpressure,Local_OF,Nozzle_expansion_ratio)[2]
            Lcon_local=0.29+(0.0019*(x_inlength/Diameter))
            return ((0.022*(prandtl_number**-0.6)*(List_of_constants[4]**0.77)*(List_of_constants[3]**0.23))/((viscosity**List_of_constants[5])*(tempratio**Lcon_local)))
        def get_blowingandCF(flux,hydroD,viscosity,Chamberpressure,Local_OF,Nozzle_expansion_ratio,mdotfuel,surfacearea):
            reynoldsnumber=(flux*hydroD)/viscosity
            skinFrictionstaright=0.074/(reynoldsnumber**0.2)
            surfacemassflux=mdotfuel/surfacearea
            streamvelocity=get_local_cea_state(Chamberpressure,Local_OF,Nozzle_expansion_ratio)[3]
            chamberdensity=get_local_cea_state(Chamberpressure,Local_OF,Nozzle_expansion_ratio)[0]
            blowingnormal=surfacemassflux/(chamberdensity*streamvelocity*0.5*skinFrictionstaright)
            return blowingnormal,skinFrictionstaright
        #def get_helix_blowing_and_cf(self):
        

        def update_reg_slot_1(list,OFstartguess,chamber_pressure_psi,list_of_constants,startDiameter,current_time=0.0,verbose=True):
            last_regression=0.0
            end_regression=0.0
            lastmdotfuel=0.0
            for x in list:
                iterate=True
                iteration_count=0
                oxflux=massflowox/x.area
                while iterate==True:
                    if x.distance==0:
                        fuelflux=oxflux/OFstartguess
                        x.totalflux=oxflux+fuelflux
                        temp_values=get_temps(chamber_pressure_psi,OFstartguess,list_of_constants)
                        viscosity=get_dynamic_viscosity(chamber_pressure_psi,OFstartguess,temp_values[1])
                        A_con=get_Acon_for_regression_slot(OFstartguess,chamber_pressure_psi,startDiameter,x.distance,list_of_constants,viscosity,temp_values[2])
                        localregression=(A_con/fuelden)*(1+2*(((x.totalflux*startDiameter)/viscosity)**-0.22)*(2.74**((x.distance*-0.4)/lengthinFG)))*((oxflux**(list_of_constants[5]+1))*(startDiameter**list_of_constants[5]))
                        end_regression=localregression
                        last_regression=localregression
                        mdotfuel=0.0
                        iterate=False
                    else:
                        mdotfuel=(fuelden*((last_regression + end_regression)/2)*x.peremeter*eachsectionlength)+lastmdotfuel
                        OF_local=massflowox/mdotfuel
                        if OF_local>OFstartguess:
                            OF_local=OFstartguess
                            mdotfuel=massflowox/OFstartguess
                        fuelflux=mdotfuel/x.area
                        x.totalflux=oxflux+fuelflux
                        temp_values=get_temps(chamber_pressure_psi,OF_local,list_of_constants)
                        viscosity=get_dynamic_viscosity(chamber_pressure_psi,OF_local,temp_values[1])
                        A_con=get_Acon_for_regression_slot(OF_local,chamber_pressure_psi,startDiameter,x.distance,list_of_constants,viscosity,temp_values[2])
                        localregression=(A_con/fuelden)*(1+2*(((x.totalflux*startDiameter)/viscosity)**-0.22)*(2.74**((x.distance*-0.4)/lengthinFG)))*((oxflux**(list_of_constants[5]+1))*(startDiameter**list_of_constants[5]))
                        #blowingnormal,skinFrictionstaright=get_blowingandCF(fuelflux,x.Diameter,viscosity,chamber_pressure_psi,OF_local,Nozzle_expansion_ratio,mdotfuel,x.surfacearea)
                        amount_reg_difference=abs(1-(localregression/last_regression))
                        last_regression=localregression
                        iteration_count+=1
                        if amount_reg_difference<0.0001:
                            iterate=False
                x.prev_regression = localregression
                # Match hardregresstionattemp.py: retain the running radial
                # regression total rather than deriving it from diameter.
                x.totalregression += localregression*timestep
                update_slot_geometry(x)
                x.OF = OFstartguess if x.distance==0 else OF_local
                x.totalflux=oxflux+fuelflux
                x.mdotfuel=mdotfuel
                x.fuelflux=fuelflux
                end_regression=localregression
                lastmdotfuel=mdotfuel
            return list

        def update_reg_slot_after_regression(list,OFstartguess,chamber_pressure_psi,list_of_constants,startDiameter,current_time=0.0,verbose=True):
            """Second-and-later-step updater from hardregresstionattemp.py."""
            last_regression=0.0
            end_regression=0.0
            lastmdotfuel=0.0
            mdotfuel=0.0
            localregression=0.0
            fuelflux=0.0
            OF_local=OFstartguess
            for x in list:
                iterate=True
                iteration_count=0
                oxflux=massflowox/x.area
                if x.distance==0:
                    fuelflux=oxflux/OFstartguess
                    x.totalflux=oxflux+fuelflux
                    temp_values=get_temps(chamber_pressure_psi,OF_local,list_of_constants)
                    viscosity=get_dynamic_viscosity(chamber_pressure_psi,OF_local,temp_values[1])
                    A_con=get_Acon_for_regression_slot(OF_local,chamber_pressure_psi,startDiameter,x.distance,list_of_constants,viscosity,temp_values[2])
                    localregression=(A_con/fuelden)*(1+2*(((x.totalflux*startDiameter)/viscosity)**-0.22)*(2.74**((x.distance*-0.4)/lengthinFG)))*((oxflux**(list_of_constants[5]+1))*(startDiameter**list_of_constants[5]))
                    mdotfuel=0.0
                    lastmdotfuel=0.0
                else:
                    # Match the standalone fallback and cap behavior.
                    prevstep_cumulative=getattr(x, 'mdot_from_fit', None)
                    if prevstep_cumulative is not None:
                        mdotfuel=prevstep_cumulative
                    else:
                        mdotfuel=massflowox/OFstartguess
                    if mdotfuel <= 0:
                        mdotfuel=1e-12
                    OF_local=massflowox/mdotfuel
                    if OF_local>OFstartguess:
                        OF_local=OFstartguess
                        mdotfuel=massflowox/OF_local
                    fuelflux=mdotfuel/x.area
                    x.totalflux=oxflux+fuelflux
                    temp_values=get_temps(chamber_pressure_psi,OF_local,list_of_constants)
                    viscosity=get_dynamic_viscosity(chamber_pressure_psi,OF_local,temp_values[1])
                    A_con=get_Acon_for_regression_slot(OF_local,chamber_pressure_psi,startDiameter,x.distance,list_of_constants,viscosity,temp_values[2])
                    localregression=(A_con/fuelden)*(1+2*(((x.totalflux*startDiameter)/viscosity)**-0.22)*(2.74**((x.distance*-0.4)/lengthinFG)))*((oxflux**(list_of_constants[5]+1))*(startDiameter**list_of_constants[5]))
                x.prev_regression = localregression
                if verbose:
                    print('Regression Rate for slot at distance', x.distance, 'm:', localregression*1000, 'mm/s',current_time, 's')
                    print('OF for slot at distance', x.distance, 'm:', OF_local)
                    print('mdotfuel for slot at distance', x.distance, 'm:', mdotfuel)
                    print('fuelflux for slot at distance', x.distance, 'm:', fuelflux)
                    print(iteration_count, 'iterations to converge for slot at distance', x.distance, 'm')
                x.mdotfuel = x.peremeter*localregression*fuelden*eachsectionlength + lastmdotfuel
                lastmdotfuel = x.mdotfuel
                x.OF = OF_local
                x.fuelflux =fuelflux
                x.totalflux=oxflux+ x.fuelflux

                x.totalregression += localregression * timestep
                update_slot_geometry(x)

                end_regression = localregression
                lastmdotfuel = mdotfuel
            return list

        def update_mdot_from_previous_profile(slots, target_r2=0.98, max_degree=6):
            """Reproduce hardregresstionattemp.py's polynomial-fit mdot pass.

            The standalone model fits ``previous regression * perimeter`` as
            a function of axial distance, analytically integrates that fitted
            polynomial from the inlet to each station, then multiplies by
            density.  Keep the same fit order selection and normalization
            here so ``mdot_from_fit`` has the same meaning in both models.
            """
            if not slots:
                return

            ordered_slots=sorted(slots, key=lambda slot: slot.distance)
            x=np.asarray([slot.distance for slot in ordered_slots], dtype=float)
            y=np.asarray(
                [slot.prev_regression*slot.peremeter for slot in ordered_slots],
                dtype=float,
            )
            valid_mask=np.isfinite(x) & np.isfinite(y)
            x=x[valid_mask]
            y=y[valid_mask]
            if len(x)<2:
                for slot in slots:
                    slot.mdot_from_fit=0.0
                return

            x_mean=np.mean(x)
            x_scale=np.max(np.abs(x-x_mean))
            if x_scale==0:
                x_scale=1.0
            x_norm=(x-x_mean)/x_scale

            best_r2=-np.inf
            best_coeffs=None
            max_d=min(max_degree, len(x)-1)
            for degree in range(1, max_d+1):
                coeffs_norm=np.polyfit(x_norm, y, degree)
                fitted=np.poly1d(coeffs_norm)(x_norm)
                ss_res=np.sum((y-fitted)**2)
                ss_tot=np.sum((y-np.mean(y))**2)
                r2=1-(ss_res/ss_tot) if ss_tot!=0 else 1.0
                if r2>best_r2:
                    best_r2=r2
                    best_coeffs=coeffs_norm
                if r2>=target_r2:
                    break

            # This is the same un-normalization and analytical integration
            # used by plot_mdot_vs_distance() in hardregresstionattemp.py.
            u_poly=np.poly1d([1.0/x_scale, -x_mean/x_scale])
            poly_norm=np.poly1d(best_coeffs)
            poly_x=np.poly1d([0.0])
            for index, coefficient in enumerate(poly_norm.coeffs):
                degree=len(poly_norm.coeffs)-index-1
                poly_x=poly_x+coefficient*(u_poly**degree)
            integral_poly_x=poly_x.integ()
            inlet_distance=0.0

            for slot in slots:
                volume_flow=float(integral_poly_x(slot.distance)-integral_poly_x(inlet_distance))
                slot.mdot_from_fit=fuelden*volume_flow

        # Stoichiometric O/F is fixed for a given CEA fuel/oxidizer object.
        if not hasattr(self, "_complex_stoich_of"):
            self._complex_stoich_of=CEAforRocket.getMRforER(ERphi=1.0)
        StoicOF=self._complex_stoich_of
        list_of_constants=get_constant_values(chamber_pressure_psi,StoicOF)
        if not hasattr(self, "regression_slots"):
            self.regression_slots=make_reg_slots()
            self.complex_regression_history=[]
            self.complex_total_fuel_mass_kg=0.0
            self.complex_uses_pixel_geometry=use_pixel_slot_geometry
            if use_pixel_slot_geometry:
                print("Complex regression: using pixel area/perimeter equations for every slot.")
            else:
                print("Complex regression: using built-in circular area/perimeter equations for every slot.")
            self.regression_slots=update_reg_slot_1(self.regression_slots,OFstartguess,chamber_pressure_psi,list_of_constants,portD,current_time=self.time,verbose=False)
        else:
            update_mdot_from_previous_profile(self.regression_slots)
            self.regression_slots=update_reg_slot_after_regression(self.regression_slots,OFstartguess,chamber_pressure_psi,list_of_constants,portD,current_time=self.time,verbose=False)

        mdot_fullgrain=self.regression_slots[-1].mdotfuel if self.regression_slots else 0.0
        self.complex_total_fuel_mass_kg+=mdot_fullgrain*timestep
        # Every slot is evenly spaced along the grain, so this arithmetic mean
        # is the whole-grain, length-averaged radial regression rate for the
        # current timestep.
        average_regression_m_per_s=(
            float(np.mean([slot.prev_regression for slot in self.regression_slots]))
            if self.regression_slots else 0.0
        )

        # When complex regression is enabled, these are the values consumed by
        # the motor model below (O/F, C*, chamber pressure, and thrust).  Do
        # not let the independent single-port correlation overwrite them.
        mean_port_area=(
            float(np.mean([slot.area for slot in self.regression_slots]))
            if self.regression_slots else 0.0
        )
        mean_port_perimeter=(
            float(np.mean([slot.peremeter for slot in self.regression_slots]))
            if self.regression_slots else 0.0
        )
        mean_total_regression=(
            float(np.mean([slot.totalregression for slot in self.regression_slots]))
            if self.regression_slots else 0.0
        )
        if usecomplexregression==True:
            self.complex_mdot_fullgrain=mdot_fullgrain
            self.complex_average_regression_m_per_s=average_regression_m_per_s
            self.mdotfuel=mdot_fullgrain
            self.totalmdot=mdot_fullgrain+massflowox
            self.sum_modtfuel+=mdot_fullgrain*timestep
            self.OF=massflowox/mdot_fullgrain if mdot_fullgrain>0 else np.inf
            self.regression_M_persec=average_regression_m_per_s
            self.regression_MM_persec=average_regression_m_per_s*1000.0
            self.regression_M_persec_withratio=average_regression_m_per_s
            self.regression_MM_persec_withratio=average_regression_m_per_s*1000.0
            self.totalrgession=2.0*mean_total_regression
            self.totalregression_MM=mean_total_regression*1000.0
            self.area_grain=mean_port_area
            self.portperimeter=mean_port_perimeter
            # ``slot.Diameter`` is the pixel-port hydraulic diameter (4A/P),
            # so preserve that definition in the motor-level summary state.
            self.diameter_grain=(
                float(np.mean([slot.Diameter for slot in self.regression_slots]))
                if self.regression_slots else 0.0
            )
            self.fuelgrain_raduis=self.diameter_grain/2.0
            self.hydrolicdiamter=self.diameter_grain
            self.oxflux=massflowox/mean_port_area if mean_port_area>0 else 0.0
            self.insurfacearea=mean_port_perimeter*FuelGrainLength*self.effectivelengthconstant
            self.volume_grain=mean_port_area*FuelGrainLength

        if Print_complex_regression:
            for index, slot in enumerate(self.regression_slots):
                self.complex_regression_history.append({
                    "time_s": self.time,
                    "slot_index": index,
                    "distance_m": slot.distance,
                    "area_m2": slot.area,
                    "diameter_m": slot.Diameter,
                    "perimeter_m": slot.peremeter,
                    "regression_m_per_s": slot.prev_regression,
                    "average_regression_m_per_s": average_regression_m_per_s,
                    "total_regression_m": slot.totalregression,
                    "total_flux_kg_m2_s": slot.totalflux,
                    "fuel_flux_kg_m2_s": slot.fuelflux,
                    "OF": slot.OF,
                    "mdotfuel_kg_s": slot.mdotfuel,
                    "mdot_from_fit": getattr(slot, "mdot_from_fit", 0.0),
                    "mdot_fullgrain": mdot_fullgrain,
                    "total_fuel_mass_kg": self.complex_total_fuel_mass_kg,
                })
        return self.regression_slots

    

    def export_complex_regression_csv(self, filename="complex_regression.csv"):
        """Write complex-regression slots to their own CSV, never output_thrust.csv."""
        history=getattr(self, "complex_regression_history", [])
        if history:
            pd.DataFrame(history).to_csv(filename, index=False)

    def iterate_regression(self, Timestep, Fuel_Density, FuelGrainLength):
        """Backward-compatible placeholder for the former empty method."""
        return getattr(self, "regression_slots", [])
    
        
        

        

    




    
    







    


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
