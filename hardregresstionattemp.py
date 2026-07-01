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
from oxtank2 import Oxtank
from fuelgrain import FuelGrain
from nozzle import Nozzle
from flight import Flight
import cProfile
import pstats
import io
import cea
import easy_cea as eCea

ABS_CARD = """
    fuel ABS  C  17.0300 H  18.9000 N   1.0000 wt%=100.00
    h,cal=21584.60803059273  t(k)=298
    rho=1.05
    """
try:
        add_new_fuel('ABS', ABS_CARD)
        print("Added ABS fuel")
except:
        print("ABS fuel already exists or failed to add")
massflowox=1.5
portD_in=4#in
portD=0.038
#regA=0.198
regA=0.127
#regN=0.325
regN=0.65
portArea=portD*portD*np.pi/4
Oxflux=massflowox/portArea
regression=regA*(Oxflux/10)**regN
#print('Regression Rate:', regression, 'mm/s')

WhatFuel='HTPB'
WhatOxidizer='N2O'
cea_2=CEA_Obj(oxName=WhatOxidizer, fuelName=WhatFuel)
thoart_area=0.0002006448
chamber_pressure_psi=300
mixture_ratio=5
chamber_transport=cea_2.get_Chamber_Transport(Pc=chamber_pressure_psi, MR=mixture_ratio, eps=1, frozen=0)
prandtl_number=chamber_transport[3]

#print(f'Prandtl number for {WhatFuel}/{WhatOxidizer} at Pc={chamber_pressure_psi} psi and O/F={mixture_ratio}: {prandtl_number}')

 #cea 2 stuff

pr=eCea.get_cea_output("prandtl", pressure_psi=300, o_f=5, should_print=False)

temp_of_flame=eCea.get_cea_output("chamber_temperature", pressure_psi=chamber_pressure_psi, o_f=mixture_ratio, should_print=False)
temp_of_surface=eCea.get_cea_output("chamber_temperature", pressure_psi=chamber_pressure_psi, o_f=0.00, should_print=False)
temp_old_cea=(cea_2.get_Temperatures(Pc=chamber_pressure_psi, MR=0, eps=1, frozen=0)[0])*0.555556
tcomb=cea_2.get_Tcomb(Pc=chamber_pressure_psi, MR=mixture_ratio)

#print(temp_of_flame, temp_of_surface, temp_old_cea,tcomb)
bulktemp=(temp_of_flame+temp_of_surface)/2
k1=0.2
x_lengthinFG=0.08 #m
D_portDiameter=0.038 #m
l1=0.29+(0.0019*(x_lengthinFG/D_portDiameter))
chamberDensity1=(cea_2.get_Chamber_Density(Pc=chamber_pressure_psi, MR=mixture_ratio, eps=1, ))*16.0185
chamberDensity2=eCea.get_cea_output("chamber_density", pressure_psi=chamber_pressure_psi, o_f=mixture_ratio, should_print=False)
chamberviscosity1=(cea_2.get_Chamber_Transport(Pc=chamber_pressure_psi, MR=mixture_ratio, eps=1, frozen=0)[1])*0.0001
chamber_sonic_velocity1=(cea_2.get_Chamber_SonicVel(Pc=chamber_pressure_psi, MR=mixture_ratio))
fac_CR=(portD*portD*3.14/4)/thoart_area
chamber_mach=cea_2.get_Chamber_MachNumber(Pc=chamber_pressure_psi, MR=mixture_ratio, fac_CR=fac_CR)
chamber_velocity=chamber_sonic_velocity1*chamber_mach*0.3048
reynolds_number=(chamberDensity1*chamber_velocity*D_portDiameter)/chamberviscosity1
print(chamberviscosity1, chamber_velocity, reynolds_number)

