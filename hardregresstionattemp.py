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
massflowox=0.1
portD_in=4#in
portD=3/100#cm
#regA=0.198
regA=0.304
#regN=0.325
regN=0.527
portArea=portD*portD*np.pi/4
Oxflux=(massflowox/portArea)/10#g/s/cm2
Oxflux=100
print('Oxflux:', Oxflux, 'g/s/cm2')
regression=regA*(Oxflux/10)**regN
regression_meters=regression/1000
print('Regression Rate:', regression, 'mm/s')
fuelden=930#kg/m3
WhatFuel='HTPB'
WhatOxidizer='GOX'
cea_2=CEA_Obj(oxName=WhatOxidizer, fuelName=WhatFuel)
thoart_area=0.0002006448
chamber_pressure_psi=150
mixture_ratio=2.7
combustion_efficiency=1
chamber_transport=cea_2.get_Chamber_Transport(Pc=chamber_pressure_psi, MR=mixture_ratio, eps=1, frozen=0)
prandtl_number=chamber_transport[3]
MwOxi=44.013
sigma_hardspherediameter=5
lengthinFG=portD*13.3 #m
#print(f'Prandtl number for {WhatFuel}/{WhatOxidizer} at Pc={chamber_pressure_psi} psi and O/F={mixture_ratio}: {prandtl_number}')

 #cea 2 stuff

pr=eCea.get_cea_output("prandtl", pressure_psi=chamber_pressure_psi, o_f=mixture_ratio, should_print=False)

temp_of_flame=eCea.get_cea_output("chamber_temperature", pressure_psi=chamber_pressure_psi, o_f=mixture_ratio, should_print=False)
temp_of_surface=eCea.get_cea_output("chamber_temperature", pressure_psi=chamber_pressure_psi, o_f=0.00, should_print=False)
temp_old_cea=(cea_2.get_Temperatures(Pc=chamber_pressure_psi, MR=2.7, eps=1, frozen=0)[0])*0.555556
tcomb=cea_2.get_Tcomb(Pc=chamber_pressure_psi, MR=mixture_ratio)

#print(temp_of_flame, temp_of_surface, temp_old_cea,tcomb)
bulktemp=(temp_of_flame+temp_of_surface)/2
k1=0.2
x_lengthinFG=0.08 #m
D_portDiameter=portD#m
l1=0.29+(0.0019*(x_lengthinFG/D_portDiameter))
chamberDensity1=(cea_2.get_Chamber_Density(Pc=chamber_pressure_psi, MR=mixture_ratio, eps=1, ))*16.0185
chamberDensity2=eCea.get_cea_output("chamber_density", pressure_psi=chamber_pressure_psi, o_f=mixture_ratio, should_print=False)
chamberviscosity1=(cea_2.get_Chamber_Transport(Pc=chamber_pressure_psi, MR=mixture_ratio, eps=1, frozen=0)[1])*0.0001
chamberviscosity2=cea_2.get_Chamber_Transport(Pc=chamber_pressure_psi, MR=0, eps=1, frozen=0)[1]*0.0001
#meanviscosity=(chamberviscosity1+chamberviscosity2)/2
molecularweight=(cea_2.get_Chamber_MolWt_gamma(Pc=chamber_pressure_psi, MR=mixture_ratio, eps=1, )[0])*0.45359237
meanmolecularweight=(MwOxi+molecularweight)/2
viscosity_calced=26.69*(np.sqrt(meanmolecularweight*bulktemp/(sigma_hardspherediameter**2)))*10**-7
print(viscosity_calced,meanmolecularweight,molecularweight)

chamber_sonic_velocity1=(cea_2.get_Chamber_SonicVel(Pc=chamber_pressure_psi, MR=mixture_ratio))
fac_CR=(portD*portD*3.14/4)/thoart_area
chamber_mach=cea_2.get_Chamber_MachNumber(Pc=chamber_pressure_psi, MR=mixture_ratio, fac_CR=fac_CR)
chamber_velocity=chamber_sonic_velocity1*chamber_mach*0.3048
reynolds_number=(chamberDensity1*chamber_velocity*portD)/viscosity_calced
print(chamberviscosity1, chamber_velocity, reynolds_number)
#cfstraight=0.074/(reynolds_number**0.2) 
#cfstraight=0.0776*(((np.log10(reynolds_number)-1.88)**-2))+(60*reynolds_number**-1)
cfstraight=0.06*(reynolds_number**-0.2)
blowing_marxman_boundrylayer=(fuelden*regression_meters)/(chamberDensity1*chamber_velocity*cfstraight*0.5)
print('Blowing Ratio:', blowing_marxman_boundrylayer,'reynolds number:',reynolds_number)
spec_blowing=blowing_marxman_boundrylayer*1.22
enthapy_surface=cea_2.get_Chamber_H(Pc=chamber_pressure_psi, MR=0, eps=1, )*2.326 #j/g
enthalpy_flame=cea_2.get_Chamber_H(Pc=chamber_pressure_psi, MR=mixture_ratio, eps=1, )*2.326 #j/g

deltaH=enthalpy_flame-enthapy_surface
start_deltaH_paper=25296 #j/g
deltaH=start_deltaH_paper
Hg_frompaper=1812 #j/g
print('Delta H:', deltaH, 'J/g',start_deltaH_paper,Hg_frompaper)
h_ratio=(deltaH/Hg_frompaper)
koX=1#oxidizer mass fraction at boundry layer
Confor_phiC=(1.22*mixture_ratio*h_ratio)
Con2for_phiC=koX+(mixture_ratio+koX)*h_ratio
Phi_C=Confor_phiC/Con2for_phiC
print('Phi_C:', Phi_C)
tempratio=(temp_of_surface/bulktemp)
activation_energy=203*1000#kj/mole
k_calced=(-0.005*(activation_energy/(8.314*temp_of_surface)))-0.08
n=1+k_calced
print('k:', k_calced)
A_con=((0.022*(prandtl_number**-0.6)*(Phi_C**0.77)*(h_ratio**0.23))/((viscosity_calced**k_calced)*(tempratio**l1)))
c1_constant=2
con1_reg=((Oxflux*portD)/viscosity_calced)**(-0.22)
con2_reg=((portD/lengthinFG)+4*A_con*(k_calced+1)*((Oxflux*portD)**(k_calced))*(lengthinFG/portD))
longpartreg=(1+2.5*c1_constant*con1_reg*con2_reg)
shortpartreg=(Oxflux**(k_calced+1))*(portD**k_calced)
regression_totaltry=(A_con/fuelden)*longpartreg*shortpartreg
print('Regression Rate:', regression_totaltry*1000, 'mm/s')

regression_at_thetop=((A_con/fuelden)*(1+2.5*con1_reg)*shortpartreg)
print('Regression Rate at the top:', regression_at_thetop*1000, 'mm/s')
print("deltaH =", deltaH)
print("h_ratio =", h_ratio)
print("l =", l1)

###

paper_deltaH=25296 #j/g
paper_Hg=1812 #j/g
paper_OF=2.5
paperHratio=(paper_deltaH/paper_Hg)
paperPhi_C=(1.22*paper_OF*paperHratio)/(1+(paper_OF+1)*paperHratio)
print('Paper Phi_C:', paperPhi_C)
#prob not porblem
#phi_c
paper_um_visc=1.37*10**-4
paper_temp_surface=935
paper_tempchamber=3701
k_paper=-0.2
prandtl_number_paper=0.7
A_paper=((0.022*(prandtl_number_paper**-0.6)*(paperPhi_C**0.77)*(paperHratio**0.23))/((paper_um_visc**k_paper)*((paper_temp_surface/paper_tempchamber)**0.29)))
print('Paper A:', A_paper,A_con)
print(paper_deltaH, deltaH, paper_Hg, Hg_frompaper, paperHratio, h_ratio, paperPhi_C, Phi_C)
print(viscosity_calced,paper_um_visc,chamberviscosity1,prandtl_number,prandtl_number_paper,temp_of_surface,paper_temp_surface,temp_of_flame,paper_tempchamber)
specific_heat_surface=cea_2.get_Chamber_Cp(Pc=chamber_pressure_psi, MR=0, eps=1, )*4.184 #j/gK
specific_heat_flame=cea_2.get_Chamber_Cp(Pc=chamber_pressure_psi, MR=mixture_ratio, eps=1, )*4.184 #j/gK
specific_heat_surface_ecea=eCea.get_cea_output("specific_heat", pressure_psi=chamber_pressure_psi, o_f=0.0, should_print=False)
specific_heat_flame_ecea=eCea.get_cea_output("specific_heat", pressure_psi=150, o_f=2.7, should_print=False)
flame_tempecea=eCea.get_cea_output("chamber_temperature", pressure_psi=150, o_f=2.7, should_print=False)
surface_tempecea=eCea.get_cea_output("chamber_temperature", pressure_psi=chamber_pressure_psi, o_f=0.0, should_print=False)
#print('Flame Temperature (eCea):', flame_tempecea, 'K')
#print('Flame Temperature:', temp_old_cea, 'K')
#print('Surface Temperature (eCea):', surface_tempecea, 'K')
#print('Specific Heat Surface:', specific_heat_surface, 'J/gK')
#print('Specific Heat Flame:', specific_heat_flame, 'J/gK')
#print('Specific Heat Surface (eCea):', specific_heat_surface_ecea, 'J/gK')
#print('Specific Heat Flame (eCea):', specific_heat_flame_ecea, 'J/gK')
#fix weird delta enthapy
Paper1=(Oxflux*portD/paper_um_visc)**-0.22
Paper2=((Oxflux)**(k_calced+1))
Paper3=(portD)**k_calced
reg_paper_exact=(A_paper/fuelden)*(1+2*Paper1)*Paper2*Paper3
print(reg_paper_exact)
print(f'Oxflux: {Oxflux}, portD: {portD}, paper_um_visc: {paper_um_visc}, k_paper: {k_paper}, A_paper: {A_paper}, fuelden: {fuelden}',)
print(f'Paper_A: {A_paper}, prandtl_number_paper: {prandtl_number_paper}, paperPhi_C: {paperPhi_C}, paperHratio: {paperHratio}, paper_um_visc: {paper_um_visc}, paper_temp_surface: {paper_temp_surface}, paper_tempchamber: {paper_tempchamber}')