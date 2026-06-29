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

class Flight:
    def __init__(self,wetmass,drymass,rocketareaM,startingoutsidetempC,PressureOutsidePa,TempchangePerMeter,realativehumidity,dragcoIfnotcdchart,
                 df_cdinput,drouge_cd,drouge_area,infaltionint,infaltiontime,main_cd,main_area,maindelpyalt,Timestep,RealTime):
        
        self.status = 'burn'
        
        self.thrust = 0.0
        self.drag = 0.0
        self.dragcoreal = 0.0
        self.mach = 0.0
        self.mass= wetmass
        self.altitude = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0
        self.pressure_pa = PressureOutsidePa
        self.temp_k = startingoutsidetempC + 273.15
        self.speed_of_sound = 0.0
        

        
        # prepare output container consistent with other model classes
        self._init_attrs = list(self.__dict__.keys())
        self.outputcsv = pd.DataFrame(columns=self._init_attrs)
        self._output_rows = []

    def stuffnotprint(self,rocketareaM):
        self.rocketarea = rocketareaM

        self.drougeinflationcd = 0.0
        self.drougeinflionarea = 0.0
        self.maininflationcd = 0.0
        self.maininflationarea = 0.0


    def update_thrust(self,thrustreal):
        """Read current thrust from `NozzleMath` and store on `self.thrust`."""
        try:
            self.thrust = thrustreal
        except Exception:
            self.thrust = 0.0

    def update_drag(self,df_cdinput,dragcoIfnotcdchart,rocketareaM,drouge_cd,drouge_area,infaltionint,infaltiontime,main_cd,main_area):
     
        if self.status == 'drouge':
            if self.drougeinflationcd <= drouge_cd:
                self.drougeinflationcd += (drouge_cd / infaltionint)
                self.drougeinflionarea += (drouge_area / infaltiontime)
            self.dragcoreal = -self.drougeinflationcd
            self.rocketarea = self.drougeinflionarea
        elif self.status == 'main':
            if self.maininflationcd <= main_cd:
                self.maininflationcd += (main_cd / infaltionint)
                self.maininflationarea += (main_area / infaltionint)
            self.dragcoreal = -self.maininflationcd
            self.rocketarea = self.maininflationarea
        else:
            # Lookup CD by Mach in provided table
            try:
                differencescd = (df_cdinput['cd'] - self.mach).abs()
                idx = differencescd.idxmin()
                closest_row = df_cdinput.loc[idx]
                self.rocketarea = rocketareaM
                if self.status == 'burn':
                    self.dragcoreal = float(closest_row['pon'])
                else:
                    self.dragcoreal = float(closest_row['poff'])
            except Exception:
                # fallback
                self.dragcoreal = dragcoIfnotcdchart
                self.rocketarea = rocketareaM
            self.drag = 0.5 * self.air_density * self.velocity ** 2 * self.dragcoreal * self.rocketarea
    # --- Flight-step helpers (translate the original rocket.py formulas into methods) ---
    def mass_update(self, dt,mdot_nozzle,drymass):
       
        if self.status == 'burn':
            mdot = mdot_nozzle
            

            # consume fuel mass
            self.mass -= mdot * dt
        if self.status != 'burn':
                self.mass = drymass
        # update weight
        self.weight = self.mass * 9.81

    def atmosphere_update(self, startingoutsidetempC,PressureOutsidePa,TempchangePerMeter,realativehumidity):
        """Compute atmospheric properties at current altitude."""
        self.temp_k = (startingoutsidetempC - (self.altitude * TempchangePerMeter)) + 273.15
        # barometric-like falloff (kept from original code)
        self.pressure_pa = PressureOutsidePa * np.exp((-0.284 * self.altitude) / (8.314 * self.temp_k))
        self.relative_humidity = realativehumidity - (self.altitude * 0.004)
        p1 = 6.1078 * 10 ** ((7.5 * (self.temp_k - 273.15)) / (self.temp_k - 35.85))
        pv = p1 * self.relative_humidity
        pad = self.pressure_pa - pv
        self.air_density = (pad / (278.058 * self.temp_k)) + (pv / (461.495 * self.temp_k))
        self.speed_of_sound = np.sqrt(1.4 * 286 * self.temp_k)
        # avoid div by zero
        self.mach = 0.0 if self.speed_of_sound == 0 else (self.velocity / self.speed_of_sound)

    def dynamics_update(self, dt):
        """Compute acceleration, integrate velocity and altitude."""
        # net force = thrust - weight - drag
        netforce = self.thrust - self.weight - self.drag
        # protect against zero mass
        acc = netforce / max(self.mass, 1e-6)
        self.acceleration = acc
        self.velocity += acc * dt
        if self.status == 'burn' and self.velocity < 0:
            self.velocity = 0.0  
        self.altitude += (self.velocity * dt) + (0.5 * acc * dt * dt)

    def status_update(self, RealTime, Timestep, maindelpyalt):
        """Update flight status string from current kinematics."""
        if self.thrust > 0 or RealTime < Timestep * 10:
            self.status = 'burn'
        elif self.velocity >= 0:
            self.status = 'coast'
        elif self.altitude >= maindelpyalt:
            self.status = 'drouge'
        else:
            self.status = 'main'

    
    def add_values(self):
        """Append current flight state to `self.outputcsv` similar to other classes."""
        row = {}
        for attr in self._init_attrs:
            row[attr] = getattr(self, attr, None)
        self._output_rows.append(row)
    
    def finalize_output(self):
        """Convert buffered rows to DataFrame. Call this before accessing outputcsv."""
        if self._output_rows:
            self.outputcsv = pd.DataFrame(self._output_rows)
            self._output_rows = []