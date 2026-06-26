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
Q=1
D=45.91952720046596
p2=1552511.7673627536
pressureup=800*6894.76

Entropyup=PropsSI('S','P',pressureup,'Q',Q,'N2O')
enthapyup=PropsSI('H','P',pressureup,'Q',Q,'N2O')

downstream_pressures=np.linspace(50*6894.76,pressureup,200)
enthalpyDiffs=[]
downstream_density=PropsSI('D','P',downstream_pressures,'S',Entropyup,'N2O')
upstream_density=PropsSI('D','P',pressureup,'Q',0,'N2O')
for pressuredown in downstream_pressures:
	Ethapydown=PropsSI('H','P',pressuredown,'S',Entropyup,'N2O')
	enthalpyDiffs.append(enthapyup-Ethapydown)

enthalpyDiffs=np.array(enthalpyDiffs)

print(Entropyup, enthapyup)
mdotspi=0.44*np.sqrt(2*upstream_density*(pressureup-downstream_pressures))
mdothem=0.9*np.sqrt(2*downstream_density*(enthalpyDiffs*1000))

plt.figure(figsize=(8,5))
plt.plot(downstream_pressures/6894.76, mdothem, label='mdothem', linewidth=2)
plt.plot(downstream_pressures/6894.76, mdotspi, label='mdotspi', linewidth=2)
plt.xlim(800, 50)
plt.xlabel('Downstream pressure (psi)')
plt.ylabel('Mass flow rate proxy')
plt.title('mdothem and mdotspi vs downstream pressure\n(800 to 50 psi, upstream state held constant)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(8,5))
plt.plot(downstream_pressures/6894.76, (enthalpyDiffs**1/2)*downstream_density, linewidth=2)
plt.xlim(800, 50)
plt.xlabel('Downstream pressure (psi), shown from 800 to 50')
plt.ylabel('Density x sqrt(Enthalpy difference)')
plt.title('Scaled enthalpy difference vs downstream pressure\n(800 to 50 psi, upstream state held constant)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()