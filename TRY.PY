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

H=1
S=945
P=2758000
Q=0.5
D=45.91952720046596
p2=1552511.7673627536
qcheck=PropsSI('Q','D',D,'P',p2,'N2O')
print(qcheck)