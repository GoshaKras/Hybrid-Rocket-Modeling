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
        self.ExpansionRatio=self.exitarea/self.throatarea


    def Pressures(self,chamberpressure_PA):
        self.exitpressure_PA=chamberpressure_PA/self.pressure_exit_chamber_ratio
        self.exitpressure_PSI=self.exitpressure_PA/6894.76

       

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