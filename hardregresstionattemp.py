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
print('Regression Rate:', regression, 'mm/s')

