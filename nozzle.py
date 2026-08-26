import time as time_module
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


class Nozzle():
    def __init__(self,NozzleExpansion,NozzleThroatRad,NozzleExitRad,NozzleThoatArea,NozzleExitArea):
        self.thrust=None
        self.ExpansionRatio=NozzleExpansion
        self.sonicVelocity_Exit=None
        self.exitvelocity=None
        self.momentumthrust=None
        self.pressurethrust=None
        self.rawthrust=None
        self.thrustreal=None
        self.exitpressure_PA=None
        self.seperation_pressure_pa=None
        self.schmucker_pressure_ratio=None
        self.exitpressure_PSI=None
        self.throat_pressureflux=None
        self.exit_pressureflux=None
        self.throatraduis=NozzleThroatRad
        self.exitraduis=NozzleExitRad
        self.throatarea=NozzleThoatArea
        self.exitarea=  NozzleExitArea
        self.throatabltionrate=None
        self.exitabltionrate=None
        self.ISP=None
        self.Is_flow_seperation=False
        self.phase_separation_target_exitpressure_PA=None
        self.real_exit_expansion_ratio=None
        self.real_exit_area=None
        self.real_exit_radius=None
        self.totalimpulse=0
        


        self._init_attrs = list(self.__dict__.keys())
        self.outputcsv = pd.DataFrame(columns=self._init_attrs)
        self._output_rows = []
    def stuffnotprint(self,max_at_what_pressure_psi,IntailGoalOf_OF_ratio,NozzleThroatRad,NozzleExitRad,NozzleExpansion,CEAforRocket):
        self.sonicVelocity_Chamber=None
        self.sonicVelocity_Throat=None
        self.pressure_exit_chamber_ratio=None
        self.startthroatflux=((max_at_what_pressure_psi*6894.76)/CEAforRocket.get_Throat_PcOvPe(Pc=max_at_what_pressure_psi, MR=IntailGoalOf_OF_ratio))/(NozzleThroatRad*NozzleThroatRad*3.14)
        self.startexitflux=((max_at_what_pressure_psi*6894.76)/CEAforRocket.get_PcOvPe(Pc=max_at_what_pressure_psi, MR=IntailGoalOf_OF_ratio, eps=NozzleExpansion))/(NozzleExitRad*NozzleExitRad*3.14)
       
        
    def CEA_Values(self,chamberpressure_PSI,OF_ratio,CEAforRocket):
        self.Exitmach=CEAforRocket.get_MachNumber(Pc=chamberpressure_PSI, MR=OF_ratio, eps=self.ExpansionRatio)
        self.sonicVelocity_Chamber,self.sonicVelocity_Throat,self.sonicVelocity_Exit=CEAforRocket.get_SonicVelocities(Pc=chamberpressure_PSI, MR=OF_ratio, eps=self.ExpansionRatio)
        self.sonicVelocity_Chamber*=0.3048
        self.sonicVelocity_Exit*=0.3048
        self.sonicVelocity_Throat*=0.3048
        self.exitvelocity=self.Exitmach*self.sonicVelocity_Exit
        self.pressure_exit_chamber_ratio=CEAforRocket.get_PcOvPe(Pc=chamberpressure_PSI, MR=OF_ratio, eps=self.ExpansionRatio)

    def easynozzleabltion(self,ThoartAblationRate,ExitAblationRate,Timestep,Throatpressure_PA):
        
        self.throat_pressureflux=Throatpressure_PA/self.throatarea
        self.exit_pressureflux=self.exitpressure_PA/self.exitarea
        self.throatabltionrate=ThoartAblationRate*(self.throat_pressureflux/self.startthroatflux)
        self.exitabltionrate=ExitAblationRate*(self.exit_pressureflux/self.startexitflux)
        self.throatraduis+= self.throatabltionrate*Timestep
        self.exitraduis+= self.exitabltionrate*Timestep
        self.throatarea=self.throatraduis*self.throatraduis*3.14
        self.exitarea=self.exitraduis*self.exitraduis*3.14

    def expansionratio(self):
        if self.Is_flow_seperation:
            self.ExpansionRatio=self.real_exit_expansion_ratio
        else:
            self.ExpansionRatio=self.exitarea/self.throatarea
        


    def Pressures(self,chamberpressure_PA):
        self.exitpressure_PA=chamberpressure_PA/self.pressure_exit_chamber_ratio
        self.exitpressure_PSI=self.exitpressure_PA/6894.76
    def flow_sepration(self, chamber_pressure_pa, ambient_pressure_pa, expansion_ratio,
                       chamberpressure_psi=None, of_ratio=None, CEAforRocket=None):
        if any(value is None for value in (chamberpressure_psi, of_ratio, CEAforRocket)):
            raise ValueError("Flow-separation calculation requires chamber pressure, O/F, and CEA inputs")

        # Schmucker separation criterion:
        # P_sep / P_a = (1.88 * M_sep - 1) ** 0.64.
        # Evaluate the condition at the *physical* nozzle exit. Do not use the
        # prior timestep's effective separated area here; doing so feeds the
        # separation result back into its own trigger and causes chatter.
        physical_expansion_ratio = self.exitarea / self.throatarea
        physical_exit_mach = CEAforRocket.get_MachNumber(
            Pc=chamberpressure_psi, MR=of_ratio, eps=physical_expansion_ratio
        )
        physical_pc_over_pe = CEAforRocket.get_PcOvPe(
            Pc=chamberpressure_psi, MR=of_ratio, eps=physical_expansion_ratio
        )
        physical_exit_pressure_pa = chamber_pressure_pa / physical_pc_over_pe
        schmucker_pressure_ratio = (1.88 * physical_exit_mach - 1) ** (-0.64)
        separation_pressure_pa = ambient_pressure_pa * schmucker_pressure_ratio
        self.schmucker_pressure_ratio = schmucker_pressure_ratio
        self.seperation_pressure_pa = separation_pressure_pa
        # A small recovery margin prevents numerical switching when the
        # physical exit pressure is essentially on the separation boundary.
        recovery_pressure_pa = separation_pressure_pa * 1.02
        has_flow_separation = (
            physical_exit_pressure_pa <= separation_pressure_pa
            if not self.Is_flow_seperation
            else physical_exit_pressure_pa <= recovery_pressure_pa
        )

        if has_flow_separation:
            self.Is_flow_seperation = True
            self.calculate_phase_separation_exit_area(
                chamberpressure_psi, of_ratio, CEAforRocket, ambient_pressure_pa
            )
            self.real_exit_expansion_ratio = self.real_exit_area / self.throatarea
            self.ExpansionRatio = self.real_exit_expansion_ratio
        else:
            self.Is_flow_seperation = False
            self.phase_separation_target_exitpressure_PA = None
            self.real_exit_expansion_ratio = None
            self.real_exit_area = None
            self.real_exit_radius = None

    def calculate_phase_separation_exit_area(self, chamberpressure_psi, of_ratio,
                                             CEAforRocket, ambient_pressure_pa):
        """Calculate the separation area from the coupled Schmucker criterion.

        The area is reported even when the installed nozzle is not separated;
        ``Is_flow_seperation`` indicates whether the current nozzle exit
        pressure is already below the separation threshold.
        """
        if min(chamberpressure_psi, of_ratio, ambient_pressure_pa,
               self.throatarea) <= 0:
            raise ValueError("Phase-separation exit-area calculation requires positive inputs")

        chamberpressure_pa = chamberpressure_psi * 6894.76
        def pressure_residual(expansion_ratio):
            pc_over_pe = CEAforRocket.get_PcOvPe(
                Pc=chamberpressure_psi, MR=of_ratio, eps=expansion_ratio
            )
            exit_pressure_pa = chamberpressure_pa / pc_over_pe
            separation_mach = CEAforRocket.get_MachNumber(
                Pc=chamberpressure_psi, MR=of_ratio, eps=expansion_ratio
            )
            separation_pressure_pa = ambient_pressure_pa * (
                1.88 * separation_mach - 1
            ) ** (-0.64)
            return exit_pressure_pa - separation_pressure_pa

        # CEA requires eps > 1 for an expanded nozzle. If the residual is
        # already non-positive at the throat, the Schmucker criterion places
        # separation at (or upstream of) the throat; use the throat as the
        # effective exit instead of attempting to find a nonexistent root.
        lower_eps = 1.000001
        upper_eps = 2.0
        lower_residual = pressure_residual(lower_eps)
        if lower_residual <= 0:
            self.real_exit_expansion_ratio = lower_eps
        else:
            upper_residual = pressure_residual(upper_eps)
            while upper_residual > 0 and upper_eps < 1_000_000:
                upper_eps *= 2
                upper_residual = pressure_residual(upper_eps)
            if upper_residual > 0:
                raise ValueError(
                    "Could not bracket an expansion ratio for the Schmucker separation condition"
                )
            self.real_exit_expansion_ratio = scp.optimize.brentq(
                pressure_residual, lower_eps, upper_eps
            )
        self.real_exit_area = self.real_exit_expansion_ratio * self.throatarea
        self.real_exit_radius = np.sqrt(self.real_exit_area / np.pi)
        separation_mach = CEAforRocket.get_MachNumber(
            Pc=chamberpressure_psi, MR=of_ratio,
            eps=self.real_exit_expansion_ratio
        )
        self.phase_separation_target_exitpressure_PA = ambient_pressure_pa * (
            1.88 * separation_mach - 1
        ) ** (-0.64)
       

    def Thrust(self,FuelGrainMath,Is_sim_flight,OutPressure_PA,PressureOutsidePa,NozzleExitArea,RawSillyMotorEffceincy,Ncomb_ForChamberPressure,RealTime,Timestep,mdotnozzle):
        self.momentumthrust=FuelGrainMath.mdotnozzle*self.exitvelocity
        if Is_sim_flight==True:
            self.outsidePressure_PA=OutPressure_PA
        else:
            self.outsidePressure_PA=PressureOutsidePa
        self.pressurethrust=(self.exitpressure_PA-self.outsidePressure_PA)*self.exitarea
        self.rawthrust=self.momentumthrust+self.pressurethrust
        self.thrustreal=self.rawthrust*RawSillyMotorEffceincy*Ncomb_ForChamberPressure
        if self.thrustreal<0 and RealTime<(Timestep*10):
            self.thrustreal=0
        self.ISP=self.thrustreal/(mdotnozzle*9.81)
        self.totalimpulse+=self.thrustreal*Timestep
    

    def add_values(self):
        row = {}
        for attr in self._init_attrs:  
            val = getattr(self, attr)
            row[attr] = val
        self._output_rows.append(row)
    
    def finalize_output(self):
        """Convert buffered rows to DataFrame. Call this before accessing outputcsv."""
        if self._output_rows:
            self.outputcsv = pd.DataFrame(self._output_rows)
            self._output_rows = []
