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
        self.throatraduis+= self.throatabltionrate*Timestep*2
        self.exitraduis+= self.exitabltionrate*Timestep*2
        self.throatarea=self.throatraduis*self.throatraduis*3.14
        self.exitarea=self.exitraduis*self.exitraduis*3.14

    def expansionratio(self):
        if self.flow_sepration==True:
            self.ExpansionRatio=self.real_exit_expansion_ratio
        else:
            self.ExpansionRatio=self.exitarea/self.throatarea
        


    def Pressures(self,chamberpressure_PA):
        self.exitpressure_PA=chamberpressure_PA/self.pressure_exit_chamber_ratio
        self.exitpressure_PSI=self.exitpressure_PA/6894.76
    def flow_sepration(self, chamber_pressure_pa, ambient_pressure_pa, expansion_ratio,
                       chamberpressure_psi=None, of_ratio=None, CEAforRocket=None):
        # Schmucker separation criterion:
        # P_sep / P_a = (1.88 * M_sep - 1) ** 0.64.
        # At the onset of separation the separation plane is the nozzle exit,
        # so the installed exit Mach number defines the onset threshold.
        schmucker_pressure_ratio = (1.88 * self.Exitmach - 1) ** 0.64
        separation_pressure_pa = ambient_pressure_pa * schmucker_pressure_ratio
        self.Is_flow_seperation = self.exitpressure_PA <= separation_pressure_pa
        if self.Is_flow_seperation:
            if any(value is None for value in (chamberpressure_psi, of_ratio, CEAforRocket)):
                raise ValueError("Flow-separation area calculation requires chamber pressure, O/F, and CEA inputs")
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
            ) ** 0.64
            return exit_pressure_pa - separation_pressure_pa

        # CEA requires eps > 1 for an expanded nozzle. Increase the upper
        # bracket until its exit pressure falls below the requested target.
        lower_eps = 1.000001
        upper_eps = 2.0
        while pressure_residual(upper_eps) > 0 and upper_eps < 1_000_000:
            upper_eps *= 2
        if pressure_residual(upper_eps) > 0:
            raise ValueError("Could not bracket an expansion ratio for the phase-separation pressure target")

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
        ) ** 0.64
       

    def Thrust(self,FuelGrainMath,Is_sim_flight,OutPressure_PA,PressureOutsidePa,NozzleExitArea,RawSillyMotorEffceincy,Ncomb_ForChamberPressure,RealTime,Timestep,mdotnozzle):
        self.momentumthrust=FuelGrainMath.mdotnozzle*self.exitvelocity
        if Is_sim_flight==True:
            self.outsidePressure_PA=OutPressure_PA
        else:
            self.outsidePressure_PA=PressureOutsidePa
        self.pressurethrust=(self.exitpressure_PA-self.outsidePressure_PA)*NozzleExitArea
        self.rawthrust=self.momentumthrust+self.pressurethrust
        self.thrustreal=self.rawthrust*RawSillyMotorEffceincy*Ncomb_ForChamberPressure
        if self.thrustreal<0 and RealTime<(Timestep*10):
            self.thrustreal=0
        self.ISP=self.thrustreal/(mdotnozzle*9.81)
    

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
