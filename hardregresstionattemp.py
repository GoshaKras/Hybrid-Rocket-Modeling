import time as time_module
from dataclasses import dataclass
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as scp
import rocketcea as rcea

from rocketcea.cea_obj import CEA_Obj

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
ref_temp_H=298 #K
#print(f'Prandtl number for {WhatFuel}/{WhatOxidizer} at Pc={chamber_pressure_psi} psi and O/F={mixture_ratio}: {prandtl_number}')
Nozzle_expansion_ratio=4
 #cea 2 stuff

pr=eCea.get_cea_output("prandtl", pressure_psi=chamber_pressure_psi, o_f=mixture_ratio, should_print=False)

temp_of_flame=eCea.get_cea_output("chamber_temperature", pressure_psi=chamber_pressure_psi, o_f=mixture_ratio, should_print=False)
temp_of_surface=eCea.get_cea_output("chamber_temperature", pressure_psi=chamber_pressure_psi, o_f=0.00, should_print=False)
temp_old_cea=(cea_2.get_Temperatures(Pc=chamber_pressure_psi, MR=18, eps=1, frozen=0)[0])*0.555556
tcomb=cea_2.get_Tcomb(Pc=chamber_pressure_psi, MR=mixture_ratio)
print(temp_old_cea)
#print(temp_of_flame, temp_of_surface, temp_old_cea,tcomb)
bulktemp=(temp_of_flame+temp_of_surface)/2
k1=0.2
x_lengthinFG=lengthinFG #m
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
enthapy_surface=(cea_2.get_Chamber_H(Pc=chamber_pressure_psi, MR=0, eps=Nozzle_expansion_ratio))*2.326 #j/g
enthalpy_flame=(cea_2.get_Chamber_H(Pc=chamber_pressure_psi, MR=mixture_ratio, eps=Nozzle_expansion_ratio))*2.326 #j/g
ecea_enthalpy_surface=eCea.get_cea_output("H", pressure_psi=chamber_pressure_psi, o_f=0.0, should_print=False)
ecea_enthalpy_flame=eCea.get_cea_output("H", pressure_psi=chamber_pressure_psi, o_f=mixture_ratio, should_print=False)

print('Enthalpy of surface:', enthapy_surface, 'J/g')
print('Enthalpy of flame:', enthalpy_flame, 'J/g')
print('Enthalpy of surface (eCea):', ecea_enthalpy_surface, 'J/g')
print('Enthalpy of flame (eCea):', ecea_enthalpy_flame, 'J/g')
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
x_lengthinFG=(lengthinFG) #m
epart= 2.71828**(-0.4*(x_lengthinFG/portD))
regression_at_thetop=((A_con/fuelden)*(1+c1_constant*con1_reg*epart)*shortpartreg)
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
print('Flame Temperature (eCea):', flame_tempecea, 'K')
print('Flame Temperature:', temp_old_cea, 'K')
print('Surface Temperature (eCea):', surface_tempecea, 'K')
print('Specific Heat Surface:', specific_heat_surface, 'J/gK')
print('Specific Heat Flame:', specific_heat_flame, 'J/gK')
print('Specific Heat Surface (eCea):', specific_heat_surface_ecea, 'J/gK')
print('Specific Heat Flame (eCea):', specific_heat_flame_ecea, 'J/gK')
#fix weird delta enthapy
tempp_old_cea_surface=cea_2.get_Temperatures(Pc=chamber_pressure_psi, MR=0, eps=1, frozen=0)[0]*0.555556
deltaH_calced=specific_heat_flame*(temp_old_cea-ref_temp_H)-specific_heat_surface*(tempp_old_cea_surface-ref_temp_H)
print('Delta H (calculated):', deltaH_calced, 'J/g')
Paper1=(Oxflux*portD/paper_um_visc)**-0.22
Paper2=((Oxflux)**(k_calced+1))
Paper3=(portD)**k_calced
reg_paper_exact=(A_paper/fuelden)*(1+2*Paper1)*Paper2*Paper3
print(reg_paper_exact)
print(f'Oxflux: {Oxflux}, portD: {portD}, paper_um_visc: {paper_um_visc}, k_paper: {k_paper}, A_paper: {A_paper}, fuelden: {fuelden}',)
print(f'Paper_A: {A_paper}, prandtl_number_paper: {prandtl_number_paper}, paperPhi_C: {paperPhi_C}, paperHratio: {paperHratio}, paper_um_visc: {paper_um_visc}, paper_temp_surface: {paper_temp_surface}, paper_tempchamber: {paper_tempchamber}')
















#harder regression attemp
amount_of_slots=7
amount_of_sections=amount_of_slots-1


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
        
timestep=0.01      
startreg=regression_totaltry*timestep  
OFstartguess=25
def make_reg_slots(amount_of_slots=amount_of_slots, grain_length=lengthinFG, port_diameter=portD, start_regression=startreg):
        slots=[]
        for x in range(amount_of_slots):
                if amount_of_slots <= 1:
                        length = 0
                else:
                        length=x*(grain_length/amount_of_sections)
                Diameter=port_diameter
                area=Diameter*Diameter*np.pi/4
                peremeter=Diameter*np.pi
                slots.append(RegressionSlot(area=area,distance=length,totalregression=start_regression,peremeter=peremeter,Diameter=Diameter,prev_regression=start_regression))
        return slots
#A_con=((0.022*(prandtl_number**-0.6)*(Phi_C**0.77)*(h_ratio**0.23))/((viscosity_calced**k_calced)*(tempratio**l1)))
eachsectionlength=lengthinFG/amount_of_sections
moleweightN20=44.013
moleweightO2=31.998
def get_constant_values(pressure,stoicOF):
        temp_of_flame=(cea_2.get_Temperatures(Pc=pressure, MR=stoicOF, eps=1, frozen=0)[0])*0.555556
        temp_of_surface=(cea_2.get_Temperatures(Pc=chamber_pressure_psi, MR=0, eps=1, frozen=0)[0])*0.555556
        specific_heat_surface=(cea_2.get_Chamber_Cp(Pc=pressure, MR=0, eps=1, )*4.184) #j/gK
        specific_heat_flame=(cea_2.get_Chamber_Cp(Pc=pressure, MR=stoicOF, eps=1, )*4.184) #j/gK
        deltaH=specific_heat_flame*(temp_of_flame-ref_temp_H)-specific_heat_surface*(temp_of_surface-ref_temp_H)
        #Bulktemp=(temp_of_flame+temp_of_surface)/2
        #tempratio=(temp_of_surface/Bulktemp)
        enthapy_ratio=(deltaH/Hg_frompaper)
        Phi_C=(1.22*stoicOF*enthapy_ratio)/(koX+(stoicOF+koX)*enthapy_ratio)
        k_constant=(-0.005*(activation_energy/(8.314*temp_of_surface)))-0.08
        list_of_constants=[deltaH,temp_of_surface,temp_of_flame,enthapy_ratio,Phi_C,k_constant]
        return list_of_constants
def get_temps(pressure,OF,list_of_constants):
        temp_cea_local=(cea_2.get_Temperatures(Pc=pressure, MR=OF, eps=1, frozen=0)[0])*0.555556
        bulktemp_local=(temp_cea_local+list_of_constants[1]+list_of_constants[2])/3
        tempratio_local=(list_of_constants[1]/bulktemp_local)
        temp_values=[temp_cea_local,bulktemp_local,tempratio_local]
        return temp_values

        


def get_dynamic_viscosity(pressure,OF,bulktemp):
        moleweightmixture=(cea_2.get_Chamber_MolWt_gamma(Pc=pressure, MR=OF, eps=1, )[0])*0.45359237
        meanmoleweight=(moleweightN20+moleweightmixture)/2
        
        viscosity=26.69*(np.sqrt(meanmoleweight*bulktemp/(sigma_hardspherediameter**2)))*10**-7
        return viscosity
#A_con=((0.022*(prandtl_number**-0.6)*(Phi_C**0.77)*(h_ratio**0.23))/((viscosity_calced**k_calced)*(tempratio**l1)))
def get_Acon_for_regression_slot(Local_OF,Chamberpressure,Diameter,x_inlength,List_of_constants,viscosity,tempratio,Nozzle_expansion_ratio=Nozzle_expansion_ratio):
        prandtl_number=cea_2.get_Chamber_Transport(Pc=Chamberpressure, MR=Local_OF, eps=Nozzle_expansion_ratio, frozen=0)[3]
        Lcon_local=0.29+(0.0019*(x_inlength/Diameter))
        A_con_Local=((0.022*(prandtl_number**-0.6)*(List_of_constants[4]**0.77)*(List_of_constants[3]**0.23))/((viscosity**List_of_constants[5])*(tempratio**Lcon_local)))
        #print('prandtl_number:', prandtl_number, 'Lcon_local:', Lcon_local, 'phi_C:', List_of_constants[4], 'h_ratio:', List_of_constants[3], 'viscosity:', viscosity, 'tempratio:', tempratio, 'A_con_Local:', A_con_Local)
        return A_con_Local



def get_stoichiometric_of_for_species(fuel_name=WhatFuel, oxidizer_name=WhatOxidizer, chamber_pressure_psi=chamber_pressure_psi):
        """Return the stoichiometric O/F ratio for a fuel/oxidizer pair using CEA.

        CEA defines stoichiometric mixture ratio at equivalence ratio ERphi = 1.0.
        The helper uses the same RocketCEA interface as the rest of this file and
        falls back to the current script-level fuel and oxidizer names by default.
        """
        cea_for_species = CEA_Obj(oxName=oxidizer_name, fuelName=fuel_name)
        stoich_of = cea_for_species.getMRforER(ERphi=1.0)
        return stoich_of

timestep=0.01
time=0

def update_reg_slot_1(list,OFstartguess,chamber_pressure_psi,list_of_constants,startDiameter,current_time=0.0,verbose=True):
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
                
                while iterate==True:
                        if x.distance==0:
                                fuelflux=oxflux/OFstartguess
                                x.totalflux=oxflux+fuelflux
                                temp_values=get_temps(pressure=chamber_pressure_psi,OF=OF_local,list_of_constants=list_of_constants)
                                viscosity=get_dynamic_viscosity(chamber_pressure_psi,OF_local,temp_values[1])
                                A_con=get_Acon_for_regression_slot(Local_OF=OF_local,Chamberpressure=chamber_pressure_psi,Diameter=startDiameter,x_inlength=x.distance,List_of_constants=list_of_constants,viscosity=viscosity,tempratio=temp_values[2],Nozzle_expansion_ratio=Nozzle_expansion_ratio)
                                localregression=(A_con/fuelden)*(1+2*(((x.totalflux*startDiameter)/viscosity)**-0.22)*(2.74**((x.distance*-0.4)/lengthinFG)))*((oxflux**(list_of_constants[5]+1))*(startDiameter**list_of_constants[5]))
                                end_regression=localregression
                                last_regression=localregression
                                totalmdot=0
                                lastmdotfuel=0
                                mdotfuel=0.0
                                iterate=False
                        else: 
                                mdotfuel=(fuelden*((last_regression + end_regression)/2)*x.peremeter* eachsectionlength)+lastmdotfuel
                                OF_local=massflowox/mdotfuel
                                if OF_local>OFstartguess:
                                        OF_local=OFstartguess
                                        mdotfuel=massflowox/OFstartguess
                                fuelflux=mdotfuel/x.area
                                x.totalflux=oxflux+fuelflux
                                
                                temp_values=get_temps(pressure=chamber_pressure_psi,OF=OF_local,list_of_constants=list_of_constants)
                                viscosity=get_dynamic_viscosity(chamber_pressure_psi,OF_local,temp_values[1])
                                A_con=get_Acon_for_regression_slot(Local_OF=OF_local,Chamberpressure=chamber_pressure_psi,Diameter=startDiameter,x_inlength=x.distance,List_of_constants=list_of_constants,viscosity=viscosity,tempratio=temp_values[2],Nozzle_expansion_ratio=Nozzle_expansion_ratio)
                                localregression=(A_con/fuelden)*(1+2*(((x.totalflux*startDiameter)/viscosity)**-0.22)*(2.74**((x.distance*-0.4)/lengthinFG)))*((oxflux**(list_of_constants[5]+1))*(startDiameter**list_of_constants[5]))
                                amount_reg_difference=abs(1-(localregression/last_regression))
                                last_regression=localregression
                                iteration_count+=1
                                if amount_reg_difference<0.0001:
                                        iterate=False
                x.prev_regression = localregression
                if verbose:
                        print('Regression Rate for slot at distance', x.distance, 'm:', localregression*1000, 'mm/s',current_time, 's')
                        print('OF for slot at distance', x.distance, 'm:', OF_local)
                        print('mdotfuel for slot at distance', x.distance, 'm:', mdotfuel)
                        print('fuelflux for slot at distance', x.distance, 'm:', fuelflux)
                        print(iteration_count, 'iterations to converge for slot at distance', x.distance, 'm')
                x.totalregression += localregression * timestep
                x.Diameter += 2 * localregression * timestep
                x.area = x.Diameter * x.Diameter * np.pi / 4
                x.peremeter = x.Diameter * np.pi
                x.OF = OF_local
                x.totalflux=oxflux+fuelflux
                x.mdotfuel=mdotfuel
                x.fuelflux=fuelflux
                end_regression=localregression
                lastmdotfuel=mdotfuel
                
        return list
def update_reg_slot_after_regression(list,OFstartguess,chamber_pressure_psi,list_of_constants,startDiameter,current_time=0.0,verbose=True):
        last_regression=0.0
        end_regression=0.0
        lastmdotfuel=0.0
        mdotfuel=0.0
        localregression=0.0
        fuelflux=0.0
        OF_local=OFstartguess
        previous_mdot_poly = load_mdot_fit_polynomial()
        previous_mdot_integral = np.polyint(previous_mdot_poly) if previous_mdot_poly is not None else None
        for x in list:
                iterate=True
                iteration_count=0
                oxflux=massflowox/x.area
                
                if x.distance==0:
                                fuelflux=oxflux/OFstartguess
                                x.totalflux=oxflux+fuelflux
                                temp_values=get_temps(pressure=chamber_pressure_psi,OF=OF_local,list_of_constants=list_of_constants)
                                viscosity=get_dynamic_viscosity(chamber_pressure_psi,OF_local,temp_values[1])
                                A_con=get_Acon_for_regression_slot(Local_OF=OF_local,Chamberpressure=chamber_pressure_psi,Diameter=startDiameter,x_inlength=x.distance,List_of_constants=list_of_constants,viscosity=viscosity,tempratio=temp_values[2],Nozzle_expansion_ratio=Nozzle_expansion_ratio)
                                localregression=(A_con/fuelden)*(1+2*(((x.totalflux*startDiameter)/viscosity)**-0.22)*(2.74**((x.distance*-0.4)/lengthinFG)))*((oxflux**(list_of_constants[5]+1))*(startDiameter**list_of_constants[5]))
                                mdotfuel=0.0
                        
                else:
                                # Prefer the previous-step integral mdot (`mdot_from_fit`) computed from 0->distance
                                prevstep_cumulative = getattr(x, 'mdot_from_fit', None)
                                if prevstep_cumulative is not None:
                                        mdotfuel = prevstep_cumulative
                                else:
                                        # Fallback: if no integral is available, use a safe default (massflowox/OFstartguess)
                                        mdotfuel = massflowox / OFstartguess
                                if mdotfuel <= 0:
                                        mdotfuel = 1e-12
                                OF_local = massflowox / mdotfuel
                                if OF_local > OFstartguess:
                                        OF_local = OFstartguess
                                        mdotfuel = massflowox / OF_local
                                fuelflux = mdotfuel / x.area
                                x.totalflux = oxflux + fuelflux
                                
                                
                                temp_values=get_temps(pressure=chamber_pressure_psi,OF=OF_local,list_of_constants=list_of_constants)
                                viscosity=get_dynamic_viscosity(chamber_pressure_psi,OF_local,temp_values[1])
                                A_con=get_Acon_for_regression_slot(Local_OF=OF_local,Chamberpressure=chamber_pressure_psi,Diameter=startDiameter,x_inlength=x.distance,List_of_constants=list_of_constants,viscosity=viscosity,tempratio=temp_values[2],Nozzle_expansion_ratio=Nozzle_expansion_ratio)
                                localregression=(A_con/fuelden)*(1+2*(((x.totalflux*startDiameter)/viscosity)**-0.22)*(2.74**((x.distance*-0.4)/lengthinFG)))*((oxflux**(list_of_constants[5]+1))*(startDiameter**list_of_constants[5]))
                                
                x.prev_regression = localregression
                if verbose:
                        print('Regression Rate for slot at distance', x.distance, 'm:', localregression*1000, 'mm/s',current_time, 's')
                        print('OF for slot at distance', x.distance, 'm:', OF_local)
                        print('mdotfuel for slot at distance', x.distance, 'm:', mdotfuel)
                        print('fuelflux for slot at distance', x.distance, 'm:', fuelflux)
                        print(iteration_count, 'iterations to converge for slot at distance', x.distance, 'm')
                x.totalregression += localregression * timestep
                x.Diameter += 2 * localregression * timestep
                x.area = x.Diameter * x.Diameter * np.pi / 4
                x.peremeter = x.Diameter * np.pi
                x.OF = OF_local
                x.totalflux=oxflux+fuelflux
                x.mdotfuel = mdotfuel
                x.fuelflux = fuelflux
                end_regression = localregression
                lastmdotfuel = mdotfuel
                
        return list

def export_reg_slots_to_csv(slots, filename="reg_slots_snapshot.csv"):
        rows = []
        for index, slot in enumerate(slots):
                rows.append({
                        "slot_index": index,
                        "distance_m": slot.distance,
                        "area_m2": slot.area,
                        "diameter_m": slot.Diameter,
                        "perimeter_m": slot.peremeter,
                        "prev_regression_m_per_s": slot.prev_regression,
                        "totalregression_m": slot.totalregression,
                        "fuelflux": slot.fuelflux,
                        "totalflux": slot.totalflux,
                        "OF": slot.OF,
                })
        pd.DataFrame(rows).to_csv(filename, index=False)
        print(f"Exported {len(rows)} slots to {filename}")


def export_reg_slots_history_to_csv(slot_history, filename="reg_slots_history.csv"):
        pd.DataFrame(slot_history).to_csv(filename, index=False)
        print(f"Exported {len(slot_history)} slot rows to {filename}")


def load_mdot_fit_polynomial(filename="mdot_fit_coefficients.csv"):
        if not os.path.exists(filename):
                return None

        coeffs = pd.read_csv(filename)
        if coeffs.empty:
                return None

        coeffs = coeffs.sort_values("power", ascending=False)
        coefficient_values = coeffs["coefficient"].to_numpy(dtype=float)
        return np.poly1d(coefficient_values)


def plot_mdot_vs_distance(slot_history, time_value=None, target_r2=0.98, max_degree=6, filename="regression_perimeter_vs_distance_fit.png", verbose=True, save_plot=True):
        history = pd.DataFrame(slot_history)
        if history.empty:
                print("No slot history available for plotting")
                return None

        if time_value is None:
                time_value = history["time_s"].max()

        subset = history[history["time_s"] == time_value].copy()
        subset = subset.sort_values("distance_m")

        x_data = subset["distance_m"].to_numpy(dtype=float)
        # Compute regression * perimeter
        y_data = (subset["prev_regression_m_per_s"].to_numpy(dtype=float) * subset["perimeter_m"].to_numpy(dtype=float))

        valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
        x = x_data[valid_mask]
        y = y_data[valid_mask]

        if len(x) < 2:
                if verbose:
                        print("Not enough points to fit regression*perimeter vs distance")
                return None

        x_mean = np.mean(x)
        x_scale = np.max(np.abs(x - x_mean))
        if x_scale == 0:
                x_scale = 1.0
        x_norm = (x - x_mean) / x_scale

        best_degree = 1
        best_r2 = -np.inf
        best_coeffs = None

        max_d = min(max_degree, len(x) - 1)
        for degree in range(1, max_d + 1):
                coeffs_norm = np.polyfit(x_norm, y, degree)
                poly_norm = np.poly1d(coeffs_norm)
                fitted = poly_norm(x_norm)
                ss_res = np.sum((y - fitted) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
                if r2 > best_r2:
                        best_r2 = r2
                        best_degree = degree
                        best_coeffs = coeffs_norm
                if r2 >= target_r2:
                        break

        poly_norm = np.poly1d(best_coeffs)
        x_fit = np.linspace(np.min(x), np.max(x), 300)
        x_fit_norm = (x_fit - x_mean) / x_scale
        y_fit = poly_norm(x_fit_norm)

        # Compute cumulative integral (area under the fitted curve) using trapezoidal rule
        try:
                from scipy import integrate as _integrate
                if hasattr(_integrate, 'cumtrapz') and hasattr(_integrate, 'trapz'):
                        y_fit_integral = _integrate.cumtrapz(y_fit, x_fit, initial=0.0)
                        total_integral = float(_integrate.trapz(y_fit, x_fit))
                else:
                        raise Exception('scipy.integrate.cumtrapz/trapz not available')
        except Exception:
                # Manual trapezoidal cumulative integration without relying on scipy or np.trapz
                diffs = np.diff(x_fit)
                trapezoids = (y_fit[:-1] + y_fit[1:]) / 2.0 * diffs
                y_fit_integral = np.concatenate(([0.0], np.cumsum(trapezoids)))
                total_integral = float(y_fit_integral[-1])

        # Do not save intermediate CSVs; user requested only plots and reg_slots_history.csv
        fit_csv = None

        # Build polynomial in original x (un-normalize the fit) and its analytical integral
        try:
                alpha = 1.0 / x_scale
                beta = -x_mean / x_scale
                u_poly = np.poly1d([alpha, beta])
                poly_norm_p = np.poly1d(best_coeffs)
                poly_x = np.poly1d([0.0])
                for i, coeff in enumerate(poly_norm_p.coeffs):
                        degree = len(poly_norm_p.coeffs) - i - 1
                        poly_x = poly_x + coeff * (u_poly ** degree)
                integral_poly_x = poly_x.integ()  # poly1d object for integral (constant term = 0)
                # Create human-readable equation strings
                def poly_to_eq(p: np.poly1d, var='x'):
                        terms = []
                        coeffs = p.coeffs
                        deg = len(coeffs) - 1
                        for j, c in enumerate(coeffs):
                                power = deg - j
                                c_str = f"{c:.6g}"
                                if power == 0:
                                        terms.append(f"{c_str}")
                                elif power == 1:
                                        terms.append(f"{c_str}*{var}")
                                else:
                                        terms.append(f"{c_str}*{var}**{power}")
                        return ' + '.join(terms) if terms else '0'
                poly_eq = poly_to_eq(poly_x, var='x')
                integral_eq = poly_to_eq(integral_poly_x, var='x') + ' + C'
                eq_txt_file = filename.replace('.png', '_fit_equation.txt')
                try:
                        with open(eq_txt_file, 'w') as f:
                                f.write(f"Fitted polynomial (regression*perimeter) in x:\n")
                                f.write(poly_eq + '\n\n')
                                f.write("Analytical integral (indefinite):\n")
                                f.write(integral_eq + '\n\n')
                                f.write(f"Total definite integral over fit range: {total_integral}\n")
                except Exception:
                        eq_txt_file = None
        except Exception:
                poly_x = None
                integral_poly_x = None
                poly_eq = None
                integral_eq = None
                eq_txt_file = None

        # Compute mdot_from_fit (fuel mass flow) for each measured slot distance using analytical integral
        mdot_per_slot = None
        try:
                if integral_poly_x is not None:
                        x_min = 0.0
                        distances = subset['distance_m'].to_numpy(dtype=float)
                        mdots = []
                        for d in distances:
                                vol = float(integral_poly_x(d) - integral_poly_x(x_min))
                                mdot_val = float(fuelden * vol)
                                mdots.append(mdot_val)
                        mdot_per_slot = mdots
        except Exception:
                mdot_per_slot = None

        if save_plot:
                plt.figure(figsize=(8, 5))
                plt.scatter(x, y, color="black", label="regression*perimeter data")
                plt.plot(x_fit, y_fit, color="red", linewidth=2, label=f"poly fit deg {best_degree}, R²={best_r2:.4f}")
                # plot cumulative integral on secondary axis
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                ax2.plot(x_fit, y_fit_integral, color="blue", linestyle='--', linewidth=1.5, label='cumulative integral')
                ax1.set_xlabel("Distance (m)")
                ax1.set_ylabel("regression * perimeter")
                ax2.set_ylabel("cumulative integral (reg*perimeter · m)")
                plt.title(f"regression*perimeter vs distance at t={time_value:.3f} s")
                ax1.grid(True, alpha=0.3)
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
                plt.tight_layout()
                plt.savefig(filename, dpi=200)
                plt.close()

        if verbose:
                print(f"Saved regression*perimeter plot to {filename}")
                print(f"Best polynomial degree: {best_degree}")
                print(f"Best R^2: {best_r2:.6f}")
                print(f"Total integral (trapz over fit): {total_integral}")
                if best_r2 < target_r2:
                        print(f"WARNING: fit did not reach target R^2 >= {target_r2:.2f}")

        return {
                "degree": best_degree,
                "r2": best_r2,
                "coefficients_normalized": best_coeffs,
                "time_s": time_value,
                "filename": filename,
                "integral_total": total_integral,
                "mdot_per_slot": mdot_per_slot,
                "slot_distances": subset['distance_m'].to_numpy(dtype=float),
        }


StoicOF=get_stoichiometric_of_for_species(fuel_name=WhatFuel, oxidizer_name=WhatOxidizer, chamber_pressure_psi=chamber_pressure_psi)
print('Stoichiometric O/F for', WhatFuel, 'and', WhatOxidizer, 'at', chamber_pressure_psi, 'psi:', StoicOF)
                   


# Remove previous plot files from earlier runs
for pattern in ("mdot_vs_distance*.png", "regression_vs_distance*.png", "regression_vs_distance_*", "mdot_vs_distance_*"):
        for old_plot in glob.glob(os.path.join(os.getcwd(), pattern)):
                try:
                        os.remove(old_plot)
                except Exception:
                        pass

#calc
Local_OF=OFstartguess
startDiameter=portD
endtime=5
StoicOF=get_stoichiometric_of_for_species(fuel_name=WhatFuel, oxidizer_name=WhatOxidizer, chamber_pressure_psi=chamber_pressure_psi)
list_of_constants=get_constant_values(pressure=chamber_pressure_psi,stoicOF=StoicOF)
reg_slots=make_reg_slots(amount_of_slots=amount_of_slots, grain_length=lengthinFG, port_diameter=startDiameter, start_regression=startreg)
slot_history = []
step_index = 0
total_mdot_fuel = 0.0  # cumulative fuel mass consumed (kg)
while time <= endtime + 1e-12:
        is_first_step = step_index == 0
        is_last_step = time >= endtime - 1e-12
        should_print_details = is_first_step or is_last_step

        if should_print_details or (step_index % 20 == 0):
                print(f"Step {step_index + 1} at t={time:.2f}s")

        # Compute mdot_from_fit for current time based on a curve fit from the previous timestep
        prev_time = time - timestep
        if prev_time >= -1e-12:
                fit_prev = plot_mdot_vs_distance(slot_history, time_value=prev_time, verbose=False, save_plot=False)
                if fit_prev and fit_prev.get('mdot_per_slot') is not None:
                        mdot_list = fit_prev['mdot_per_slot']
                        # update the in-memory slot object so the upcoming update uses mdot_from_fit
                        for i, mdot_val in enumerate(mdot_list):
                                if i < len(reg_slots):
                                        reg_slots[i].mdot_from_fit = mdot_val

        if is_first_step:
                reg_slots = update_reg_slot_1(list=reg_slots,OFstartguess=Local_OF,chamber_pressure_psi=chamber_pressure_psi,list_of_constants=list_of_constants,startDiameter=startDiameter,current_time=time,verbose=should_print_details)
        else:
                reg_slots = update_reg_slot_after_regression(list=reg_slots,OFstartguess=Local_OF,chamber_pressure_psi=chamber_pressure_psi,list_of_constants=list_of_constants,startDiameter=startDiameter,current_time=time,verbose=should_print_details)
        mdot_fuel_last_slot = reg_slots[-1].mdotfuel if reg_slots else 0.0
        # accumulate mass: mass += mdot (kg/s) * dt (s)
        total_mdot_fuel += mdot_fuel_last_slot * timestep
        # mdot for full fuelgrain is defined as the mdotfuel at the last slot
        mdot_fullgrain = None
        if len(reg_slots) > 0:
                mdot_fullgrain = getattr(reg_slots[-1], 'mdotfuel', None)

        for index, slot in enumerate(reg_slots):
                slot_history.append({
                        "time_s": time,
                        "slot_index": index,
                        "distance_m": slot.distance,
                        "area_m2": slot.area,
                        "diameter_m": slot.Diameter,
                        "perimeter_m": slot.peremeter,
                        "prev_regression_m_per_s": slot.prev_regression,
                        "totalregression_m": slot.totalregression,
                        "totalflux": slot.totalflux,
                        "fuelflux": slot.fuelflux,
                        "OF": slot.OF,
                        "mdotfuel": slot.mdotfuel,
                        "mdot_from_fit": getattr(slot, 'mdot_from_fit', None),
                        "mdot_fullgrain": mdot_fullgrain,
                        "total_fuel_mass_kg": total_mdot_fuel,
                })
        

        # Save regression*perimeter plot only for first and last timestep
        if is_first_step or is_last_step:
                plot_filename = f"regression_vs_distance_t{step_index:04d}.png"
                plot_mdot_vs_distance(slot_history, time_value=time, verbose=False, save_plot=True, filename=plot_filename)
        time += timestep
        step_index += 1
print(f"Completed regression simulation at t={endtime:.2f}s with timestep {timestep:.4f}s.")
print(f"Total fuel mass consumed: {total_mdot_fuel:.6f} kg")
export_reg_slots_history_to_csv(slot_history)

# Write a small summary CSV with final mdot and first/last diameters
try:
        final_mdot = None
        first_diameter = None
        last_diameter = None
        if len(reg_slots) > 0:
                final_mdot = getattr(reg_slots[-1], 'mdotfuel', None)
                first_diameter = getattr(reg_slots[0], 'Diameter', None)
                last_diameter = getattr(reg_slots[-1], 'Diameter', None)
        summary_rows = [{
                'final_time_s': endtime,
                'final_mdot_fullgrain': final_mdot,
                'first_slice_diameter_m': first_diameter,
                'last_slice_diameter_m': last_diameter,
                'total_fuel_mass_kg': total_mdot_fuel,
        }]
        pd.DataFrame(summary_rows).to_csv('reg_slots_summary.csv', index=False)
        print(f"Wrote summary to reg_slots_summary.csv: final_mdot={final_mdot}, first_diameter={first_diameter}, last_diameter={last_diameter}")
except Exception as e:
        print("Failed to write summary CSV:", e)





