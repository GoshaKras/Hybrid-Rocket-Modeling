#Imports
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
#Custom imports

# Usage:
# python motormodel.py export_inputs inputs_sample.csv
# python motormodel.py import_inputs inputs_sample.csv

if True:#cea
    #CEA OUTPUT
    SORBITOL_CARD = """
    fuel Sorbitol  C   6.0000 H  14.0000 O   6.0000 wt%=100.00
    h,cal=-323542.06500956026   t(k)=298.15
    rho=1.2
    """

    PMMA_CARD = """
    oxid fuel  C   5.0000 H   8.0000 O   2.0000 wt%=100.00
    h,cal=-44091.78   t(k)=298.15
    rho=1.19
    """

    PBAN_CARD = """
    oxid fuel  C  54.0000 H  84.0000 N   8.0000 O   4.0000 wt%=100.00
    h,cal=1564246.03    t(k)=298.15
    rho=1.05
    """

    PARAFFIN_CARD = """
    fuel Paraffin  C  25.0000 H  52.0000 wt%=100.00
    h,cal=-148647.23t(k)=298.15
    rho=0.9
    """

    ABS_CARD = """
    fuel ABS  C  17.0300 H  18.9000 N   1.0000 wt%=100.00
    h,cal=21584.60803059273  t(k)=298.15
    rho=1.05
    """

    # Register custom fuels
    try:
        add_new_fuel('Sorbitol', SORBITOL_CARD)
        print("Added Sorbitol fuel")
    except:
        print("Sorbitol fuel already exists or failed to add")

    try:
        add_new_fuel('PMMA', PMMA_CARD)
        print("Added PMMA fuel")
    except:
        print("PMMA fuel already exists or failed to add")

    try:
        add_new_fuel('PBAN', PBAN_CARD)
        print("Added PBAN fuel")
    except:
        print("PBAN fuel already exists or failed to add")

    try:
        add_new_fuel('Paraffin', PARAFFIN_CARD)
        print("Added Paraffin fuel")
    except:
        print("Paraffin fuel already exists or failed to add")

    try:
        add_new_fuel('ABS', ABS_CARD)
        print("Added ABS fuel")
    except:
        print("ABS fuel already exists or failed to add")


if len(sys.argv) >= 3:
    if sys.argv[1] == 'export_inputs':
        inputs_io.export_inputs(sys.argv[2], globals())
        sys.exit(0)
    elif sys.argv[1] == 'import_inputs':
        inputs_io.import_inputs(sys.argv[2], globals())
        print(f"Imported inputs from {sys.argv[2]}")
        # Continue running with imported inputs

# Start timing
script_start_time = time_module.time()

#shit for output csv
Output="output_thrust.csv"
#input files
inputcdcsv="cd.csv"
df_cdinput=pd.read_csv(inputcdcsv)
#General Inputs
ShowPlots = False  # default: controls pixel-method plotting; may be overridden by vertical inputs

USE_VERTICAL_INPUTS = True # Set to True to load from vertical CSV, False to use hardcoded values
VERTICAL_INPUTS_FILE = 'inputs_vertical_sample.csv'
Exportinputs=False  # Set to True to export inputs to vertical CSV


if USE_VERTICAL_INPUTS:
    import inputs_io
    inputs_io.import_inputs_vertical(VERTICAL_INPUTS_FILE, globals())
    CEAforRocket = CEA_Obj(oxName=WhatOxidizer, fuelName=WhatFuel)
    print(f"Loaded inputs from {VERTICAL_INPUTS_FILE}")
else:
    Want_ThrustCurve=True
    Timestep=0.01
    outsidePressure_PA=101325
    UploadFromExel=False
    if UploadFromExel==False:

        # Ox tank inputs
        Oxtanktemp= 295 #kelvin
        MetalOxtanktemp=295 #kelvin
        calculate_volume_YN=True
        volumeinput= 0.06 #m^3
        oxtanklength=2.895 
        oxtankOD=0.168275 #m
        oxtankID=0.161 #m
        oxtankspecheat= 896 #J/kgK
        oxtankmass=20 #kg
        oxtankLstart=oxtankID # lstar just the diameter
        Ullage=0.1 #%
        OxamountwhenfizzstartsConstant=0.05
        Fizzstart2=1
        FizzCurveConstant=1
        NitrousGamma=1.27
        vapour_temp_damping=1 # 0-1: higher = slower cooling (0.95 = lose 5% per iteration, 0.9 = lose 10%)
        #Fuel Grain inputs
        Is_FuelGrain_just_a_circle=False
        FuelGrainRadius=0.0381 #meters
        Regression_Aco= 0.127
        Regression_Nco= 0.65
        Fuel_Density=920 #kg/m^3
        FuelGrainLength=0.8 #m
        IntailGoalOf_OF_ratio=6.3
        InitailGoalof_Chamberpressure_Psi= 400 #psi
        OuterGrainRaduis=0.06985 #m #idk if used
        OuterDiameter_inches=10.5 #inches # for pixel method
        helixloopdiameter=0.0375 #m
        RawSillyMotorEffceincy=0.8235
        Ncomb_ForChamberPressure=0.85
        NozzleCD_ForChamberPressure=0.97
        start_chamber_pressure_pa=101325
        start_chamber_temperature_k=300 
        regfluxstuff= 10 #based of units
        preccandpostvolume=0.00616
        start_gas_gpermole= 28.97 #g/mole
        #Keep CenterCircleFuelGrain_Raduis at zero if yu want this fully ignored
        CenterCircleFuelGrain_Raduis=0.0375 #m
        AmountofSmallCircles=5
        OutsideSmallCircle_raduis= 0.012 #m
        revPitch=1.05 # how many times it revolves in fuel grain
        helixrundiameter=0.0375
        WhatFuel='HTPB'  # your options: 'Sorbitol', 'PMMA', 'PBAN', 'Paraffin', 'ABS', 'HTPB'
        WhatOxidizer='N2O'
        TimeWhenBoostSops=4.75 #seconds
        Is_fuelGrain_Helix=True
        Is_FuelGrain_GoshaStar=False
        Is_FuelGrain_PixelMethod=True
        Is_fuelgrain_Transient=True
        Is_start_mass_gas_input=False
        Is_vapourPhase_constant_temp=False
        Is_sim_flight=True
        Is_2phase_flow_injector_model=True
        Is_easy_nozzle_regression=True
        Is_fizz_when_equal=False #if false set value
        Is_print_pixelstuff=False
        Is_fullshape_on_helix=False
        # Toggle to show or hide pixel-method plots during regression analysis
        ShowPlots = False
        
        
        #Injector Inputs
        InjectorArea=0.00010716494696 #m^2
        DischargeCo_SPI=0.65
        DischargeCo_HEM=0.9
        PrefferedMassFlowrate_SPI=3.9 #kg/sec
        HowmuchInj_Help=1.51
        HydrolicDiameter=9.53/1000 #m
        #Vent inputs....Probably wont be used for a while
        VentDiameter=0.001 #m
        VentCD=0.9
        #Nozzle Inputs
        NozzleThoatArea=0.002361911 #m
        NozzleExitArea= 0.010510502 #m
        TakeExpansionAsInput=True
        NozzleExpansionAs_Input=4
       
        #Simple Nozzle AblationRate
        ThoartAblationRate=0 #m/sec
        ExitAblationRate=(0.183/1000) #m/sec
        max_at_what_pressure_psi= 400 #psi
        max_at_what_OF_ratio= 6.3

        # flight inputs-
        wetmass=84.0 #kg
        drymass=33.75 #kg
        burntimecutoff=20 #sec
        dragcofrominputcsv=True
        dragcoIfnotcdchart=0.7
        drouge_area=0.67 #meters
        drouge_cd=0.97 #meters
        main_area=1 #meters
        main_cd=1.6
        infaltiontime=2 #sec
        rocketdiameterIn=6.625 #in
        startingoutsidetempC=23.0 #C
        TempchangePerMeter=0.0065 #C/m
        pressureOutsideIn= 29.97 #in#
        realativehumidity=27.44 #%
        maindelpyalt=500.0 #meters
        
        #general stuff
        Timestep=0.01
        outsidePressure_PA=101325

        max_at_what_pressure_psi= 400 #psi
        max_at_what_OF_ratio= 6.3
        regfluxstuff= 10
        Is_fizz_when_equal=False
        preccandpostvolume=0.00616
        Fizzstart2=1


#stuff that is calculated from inputs




if True:
    oxtanksurfaceareainput=(oxtankID*3.14*oxtanklength)+(2*3.14*((oxtankID/2)**2)) #m^2
    Calced_OxtankVolume=3.14*oxtankID*oxtankID*0.25*oxtanklength
    if calculate_volume_YN==True:
        OxtankVolume=volumeinput
    else:
        OxtankVolume=Calced_OxtankVolume
    FuelGrainDiameter= FuelGrainRadius*2
    InitailGoalof_Chamberpressure_Pa= InitailGoalof_Chamberpressure_Psi*6894.76 #pa
    if Is_FuelGrain_GoshaStar==False:
            CenterCircleFuelGrain_Raduis=FuelGrainRadius
    Pitchlength=FuelGrainLength/revPitch
    CEAforRocket=CEA_Obj(oxName=WhatOxidizer, fuelName=WhatFuel)
   
    if TakeExpansionAsInput == True:
            NozzleExpansion= NozzleExpansionAs_Input
            NozzleExitArea=NozzleThoatArea*NozzleExpansionAs_Input
    else:
            NozzleExpansion=NozzleExitArea/NozzleThoatArea
    NozzleThroatRad=np.sqrt(NozzleThoatArea/3.14)
    NozzleExitRad=np.sqrt(NozzleExitArea/3.14)
    infaltionint= infaltiontime/Timestep
    rocketdiameterM= rocketdiameterIn*0.0254 #M
    rocketareaM= rocketdiameterM*rocketdiameterM*0.25*3.14
    PressureOutsidePa= pressureOutsideIn*3386.39 #PA
    helixloopdiameter=helixrundiameter
    
    Use_U=True
    isventopen=True
    #stuff to add to csv
    #max_at_what_pressure_psi= 400 #psi
    #max_at_what_OF_ratio= 6.3
    #regfluxstuff= 10
    #Is_fizz_when_equal=False
    #preccandpostvolume=0.00616
    #Fizzstart2=1
    

    


    

        
#class shit-
class Inputy_Output_Value:
    def __init__(self,label,value,unit,precision):
        self.name= label
        self.value= value
        self.unit= unit
        self.precision= precision
    def __str__(self):
        if isinstance(self.value, float):
            return f"{self.name}: {self.value:.{self.precision}f} {self.unit}"
        else:
            return f"{self.name}: {self.value} {self.unit}"
class OutputGroup:
    def __init__(self, name):
        self.name = name
        self.outputs = []

    def add_output(self, label, value, unit, precision):
        self.outputs.append(Inputy_Output_Value(label, value, unit, precision))

    def print_outputs(self):
        print(f"--- {self.name} Outputs ---")
        for output in self.outputs:
            if isinstance(output.value, float):
                print(f"{output.name:<30} {output.value:.{output.precision}f} {output.unit}")
            else:
                print(f"{output.name:<30} {output.value} {output.unit}")
    def get_output_by_label(self, label):
        for output in self.outputs:
            if output.name == label:
                return output
        raise KeyError(f"Output label '{label}' not found in {self.name}")
class RocketOutputs:
    def __init__(self):
        self.cea = OutputGroup("CEA Properties")
        self.nozzle = OutputGroup("Nozzle")
        self.oxtank = OutputGroup("Oxtank")
        self.fuelgrain = OutputGroup("Fuelgrain")
        # You can add more groups here, e.g. self.oxtank, self.fuelgrain

    def print_all_outputs(self):
        self.cea.print_outputs()
        self.nozzle.print_outputs()
        self.oxtank.print_outputs()
        self.fuelgrain.print_outputs()
InputyOutputs=RocketOutputs()
#Outputs from inputs and like shit that could help inputs... INPUTY OUTPUTS !!!
if True:
    #ox tank outputs
    Calced_OxtankVolume=3.14*oxtankID*oxtankID*0.25*oxtanklength #m^3
    VolumeOf_Liquid=OxtankVolume*(1-Ullage) #m^3
    VolumeOf_Gas=OxtankVolume*Ullage #m^3
    StartingDen_Gas= PropsSI('D', 'T', Oxtanktemp, 'Q', 1, 'NitrousOxide') #kg/m^3
    StartingDen_Liquid= PropsSI('D', 'T', Oxtanktemp, 'Q', 0, 'NitrousOxide') #kg/M^3
    StartVapour_Pressure=PropsSI('P', 'T', Oxtanktemp, 'Q', 1, 'NitrousOxide') #PA
    StartMass_Liquid= VolumeOf_Liquid*StartingDen_Liquid #kg
    StartMass_Gas= VolumeOf_Gas*StartingDen_Gas #kg
    StartMass_TotalOx= StartMass_Gas+StartMass_Liquid #kg
    StartVapourPressure_PSI= StartVapour_Pressure/6894.76 #PSI
def print_OxTankStarting_Outputs():
    print(f"🔹 Liquid Volume:         {VolumeOf_Liquid:.4f} m³")
    print(f"🔹 Gas Volume:            {VolumeOf_Gas:.4f} m³")
    print(f"🔹 Total Tank Volume:     {OxtankVolume:.4f} m³")
    print(f"🔹 Liquid Density:        {StartingDen_Liquid:.2f} kg/m³")
    print(f"🔹 Vapour Density:        {StartingDen_Gas:.2f} kg/m³")
    print(f"🔹 Vapour Pressure:       {StartVapour_Pressure:.0f} Pa")
    print(f"🔹 Vapour Pressure:       {StartVapourPressure_PSI:.2f} PSI")
    print(f"🔹 Liquid Mass:           {StartMass_Liquid:.3f} kg")
    print(f"🔹 Gas Mass:              {StartMass_Gas:.3f} kg")
    print(f"🔹 Total Oxidizer Mass:   {StartMass_TotalOx:.3f} kg")
    print(f"🔹 Ullage:                {Ullage:.2f} %")
ox_outputs = {
        "Liquid Volume":         (VolumeOf_Liquid, "m³",4),
        "Gas Volume":            (VolumeOf_Gas, "m³",4),
        "Total Tank Volume":     (OxtankVolume, "m³",4),
        "Liquid Density":        (StartingDen_Liquid, "kg/m³",4),
        "Vapour Density":        (StartingDen_Gas, "kg/m³",4),
        "Vapour Pressure":       (StartVapour_Pressure, "Pa",4),
        "Vapour Pressure (PSI)": (StartVapourPressure_PSI, "PSI",4),
        "Liquid Mass":           (StartMass_Liquid, "kg",4),
        "Gas Mass":              (StartMass_Gas, "kg",4),
        "Total Oxidizer Mass":   (StartMass_TotalOx, "kg",4),
        "Ullage":                (Ullage * 100, "%",4)
    }
for label, (value, unit, prec) in ox_outputs.items():
    InputyOutputs.oxtank.add_output(label, value, unit, prec)

def print_OxTankStarting_Outputs_For_loop():
    for label, (value, unit, lastround) in ox_outputs.items():
        if isinstance(value, float):
            print(f"🔹 {label:<24} {value:,.{lastround}f} {unit}")
        else:
            print(f"🔹 {label:<24} {value} {unit}")
# Fuel Grain Outputs
if True:
    fuelGrainArea_Circle=FuelGrainRadius*FuelGrainRadius*3.14
    FuelGrainCircle_Permeter=FuelGrainDiameter*3.14
    ComplexBigCircle_Area= CenterCircleFuelGrain_Raduis*CenterCircleFuelGrain_Raduis*3.14
    ComplexSmallCircle_SingleArea=OutsideSmallCircle_raduis*OutsideSmallCircle_raduis*3.14
    ComplexSmallCircle_TotalArea_whatisadded=ComplexSmallCircle_SingleArea*0.5*AmountofSmallCircles
    TotalComplexArea=ComplexSmallCircle_TotalArea_whatisadded+ComplexBigCircle_Area
    whatRaduisWouldNeedtobeOfNormalCircleToMatch= np.sqrt((TotalComplexArea/3.14))
    ComplexPerimeter= (CenterCircleFuelGrain_Raduis*6.28)+(1.14*OutsideSmallCircle_raduis*AmountofSmallCircles)
    Star_VS_Normal_Perimeter_Ratio=ComplexPerimeter/FuelGrainCircle_Permeter
    PitchFor_Helix=FuelGrainLength/revPitch
    HelixLength= revPitch*np.sqrt(((helixrundiameter*3.14)**2)+(PitchFor_Helix)**2)
    MassNeededForFG_BasedOf_OF=StartMass_TotalOx/IntailGoalOf_OF_ratio
    WantedFuelGrainFlowRate=PrefferedMassFlowrate_SPI/IntailGoalOf_OF_ratio
    WantedTotalFlowrate=WantedFuelGrainFlowRate+PrefferedMassFlowrate_SPI
    OneArchlengthestimate=((CenterCircleFuelGrain_Raduis*6.28)-(OutsideSmallCircle_raduis*2*AmountofSmallCircles))/(AmountofSmallCircles)
    helicalarchsegmant=FuelGrainLength/(AmountofSmallCircles*revPitch)

    
    if Is_FuelGrain_PixelMethod==True:
        #fuelGrain_AreaReal=1
        fuelGrain_AreaReal=fuelGrainArea_Circle #fix later
    elif Is_FuelGrain_GoshaStar==True:
        fuelGrain_AreaReal=TotalComplexArea
    else:
        fuelGrain_AreaReal=fuelGrainArea_Circle



def Print_Out_FuelGrain_Outputs(): # fix
    print(f"🔹 Circular Fuel Grain Port Area:             {fuelGrainArea_Circle:.4f} m²")
    print(f"🔹 Circular Fuel Grain Port Perimeter:        {FuelGrainCircle_Permeter:.4f} m")
    print(f"🔹 Complex Core - Center Circle Area:         {ComplexBigCircle_Area:.4f} m²")
    print(f"🔹 Complex Core - Small Circle Total Area:    {ComplexSmallCircle_TotalArea_whatisadded:.4f} m²")
    print(f"🔹 Total Complex Core Area:                   {TotalComplexArea:.4f} m²")
    print(f"🔹 Equivalent Radius of Complex Core:         {whatRaduisWouldNeedtobeOfNormalCircleToMatch:.4f} m")
    print(f"🔹 Complex Core Total Perimeter:              {ComplexPerimeter:.4f} m")
    print(f"🔹 Star/Normal Perimeter Ratio:               {Star_VS_Normal_Perimeter_Ratio:.3f}")
    print(f"🔹 Pitch per Helix Revolution:                {PitchFor_Helix:.4f} m")
    print(f"🔹 Total Helix Length:                        {HelixLength:.4f} m")
    print(f"🔹 Estimated Fuel Grain Mass (from O/F):      {MassNeededForFG_BasedOf_OF:.4f} kg")
    print(f"🔹 Estimated Fuel flowrate (from O/F):      {WantedFuelGrainFlowRate:.4f} kg/sec")
    print(f"🔹 Estimated Total flowrate (from O/F):      {WantedTotalFlowrate:.4f} kg/sec")
    print(f"🔹 Real fuel grain area:      {fuelGrain_AreaReal:.4f} m^2")
FuelGrain_Outputs = {
        "Circular Fuel Grain Port Area": (fuelGrainArea_Circle, "m²", 4),
        "Circular Fuel Grain Port Perimeter": (FuelGrainCircle_Permeter, "m", 4),
        "Complex Core - Center Circle Area": (ComplexBigCircle_Area, "m²", 4),
        "Complex Core - Small Circle Total Area": (ComplexSmallCircle_TotalArea_whatisadded, "m²", 4),
        "Total Complex Core Area": (TotalComplexArea, "m²", 4),
        "Equivalent Radius of Complex Core": (whatRaduisWouldNeedtobeOfNormalCircleToMatch, "m", 4),
        "Complex Core Total Perimeter": (ComplexPerimeter, "m", 4),
        "Star/Normal Perimeter Ratio": (Star_VS_Normal_Perimeter_Ratio, "", 3),
        "Pitch per Helix Revolution": (PitchFor_Helix, "m", 4),
        "Total Helix Length": (HelixLength, "m", 4),
        "Estimated Fuel Grain Mass (from O/F)": (MassNeededForFG_BasedOf_OF, "kg", 4),
        "Estimated Fuel flowrate (from O/F)": (WantedFuelGrainFlowRate, "kg/sec", 4),
        "Estimated Total flowrate (from O/F)": (WantedTotalFlowrate, "kg/sec", 4),
        "Real fuelgrain Area": (fuelGrain_AreaReal,"m^2",4),
        "One Arch Length Estimate": (OneArchlengthestimate,"m",4)
        
        
    }
for label, (value, unit, prec) in FuelGrain_Outputs.items():
    InputyOutputs.fuelgrain.add_output(label, value, unit, prec)

def print_FuelGrain_Outputs_Forloop():
    for label, (value, unit, precision) in FuelGrain_Outputs.items():
        print(f"🔹 {label:<45} {value:.{precision}f} {unit}")

# Chamber outputs We assume wanted psi or 425
if True:
    Cstar = CEAforRocket.get_Cstar(Pc=InitailGoalof_Chamberpressure_Psi, MR=IntailGoalOf_OF_ratio) * 0.3048

    MW_exit, gamma_exit = CEAforRocket.get_exit_MolWt_gamma(Pc=InitailGoalof_Chamberpressure_Psi, MR=IntailGoalOf_OF_ratio)
    MW_throat, gamma_throat = CEAforRocket.get_Throat_MolWt_gamma(Pc=InitailGoalof_Chamberpressure_Psi, MR=IntailGoalOf_OF_ratio, eps=NozzleExpansion)
    MW_chamber, gamma_chamber = CEAforRocket.get_Chamber_MolWt_gamma(Pc=InitailGoalof_Chamberpressure_Psi, MR=IntailGoalOf_OF_ratio, eps=NozzleExpansion)

    Temp_chamber = CEAforRocket.get_Tcomb(Pc=InitailGoalof_Chamberpressure_Psi, MR=IntailGoalOf_OF_ratio)
    Temp_chamber,Temp_throat,Temp_exit = CEAforRocket.get_Temperatures(Pc=InitailGoalof_Chamberpressure_Psi, MR=IntailGoalOf_OF_ratio, eps=NozzleExpansion)
    Pressure_exit_to_chamber_ratio = CEAforRocket.get_PcOvPe(Pc=InitailGoalof_Chamberpressure_Psi, MR=IntailGoalOf_OF_ratio, eps=NozzleExpansion)
    Mach_exit = CEAforRocket.get_MachNumber(Pc=InitailGoalof_Chamberpressure_Psi, MR=IntailGoalOf_OF_ratio, eps=NozzleExpansion)
    Sonic_Velocity_chamber,Sonic_Velocity_throat,Sonic_Velocity_exit = CEAforRocket.get_SonicVelocities(Pc=InitailGoalof_Chamberpressure_Psi, MR=IntailGoalOf_OF_ratio, eps=NozzleExpansion)
    Temp_exit/=1.8
    Temp_chamber/=1.8
    Sonic_Velocity_exit*=0.3048
    Sonic_Velocity_chamber*=0.3048
    Sonic_Velocity_throat*=0.3048
    Exit_Velocity=Mach_exit*Sonic_Velocity_exit
    Pressure_exit=InitailGoalof_Chamberpressure_Psi/Pressure_exit_to_chamber_ratio




def Print_Certain_Cea_Values():
    print(f"🔹 C*: {Cstar:.2f} m/s")
    print(f"🔹 Gamma (chamber): {gamma_chamber:.4f}")
    print(f"🔹 Gamma (throat): {gamma_throat:.4f}")
    print(f"🔹 Gamma (exit): {gamma_exit:.4f}")
    print(f"🔹 Chamber Temp: {Temp_chamber:.2f} K")
    print(f"🔹 Exit Temp: {Temp_exit:.2f} K")  # Use index [1] for exit temp
    print(f"🔹 Exit Pressure: {Pressure_exit:.2f} psi")
    print(f"🔹 Exit Mach: {Mach_exit:.3f}")
    print(f"🔹 Sonic Exit Velocity: {Sonic_Velocity_exit:.2f} m/s")
    print(f"🔹 Exit Velocity: {Exit_Velocity:.2f} m/s")
def Print_full_CEA_FORROCKET():
     s = CEAforRocket.get_full_cea_output( Pc=InitailGoalof_Chamberpressure_Psi, MR=IntailGoalOf_OF_ratio, eps=NozzleExpansion, short_output=1, pc_units='psia')
     print(s) 
Cea_data = {
    "Cstar":                      (Cstar,                  "m/s",   2),
    "Gamma (chamber)":         (gamma_chamber,          "",      4),
    "Gamma (throat)":          (gamma_throat,           "",      4),
    "Gamma (exit)":            (gamma_exit,             "",      4),
    "Chamber Temp":            (Temp_chamber,           "K",     2),
    "Exit Temp":               (Temp_exit,              "K",     2),
    "Exit Pressure":           (Pressure_exit,          "psi",   2),
    "Exit Mach":               (Mach_exit,              "",      3),
    "Sonic Exit Velocity":     (Sonic_Velocity_exit,    "m/s",   2),
    "Exit Velocity":           (Exit_Velocity,          "m/s",   2),
}

for label, (value, unit, prec) in Cea_data.items() :
    InputyOutputs.cea.add_output(label, value, unit, prec)
def Print_Certain_Cea_Values_Forloop():
  

    print("🔹 CEA PROPERTIES")
    for label, (value, unit, precision) in Cea_data.items:
        print(f"{label:<25} {value:>{7 + precision}.{precision}f} {unit}")
# Nozzle Outputs
if True:
    ChamberPressurePA=(WantedTotalFlowrate*Cstar*Ncomb_ForChamberPressure)/(NozzleThoatArea*NozzleCD_ForChamberPressure) #pa
    AreaThroatNeeded_ForChamberPressure=(WantedTotalFlowrate*Cstar*Ncomb_ForChamberPressure)/(InitailGoalof_Chamberpressure_Pa*NozzleCD_ForChamberPressure)
    ChamberPressurePSI=ChamberPressurePA/6894.76
    

def Print_Nozzle_Outputs():
    print(f"🔹 Actual chamber Pressure: {ChamberPressurePA:.2f} PA")
    print(f"🔹 Actual chamber Pressure: {ChamberPressurePSI:.2f} Psi")
    print(f"🔹 AreaThroatNeeded: {AreaThroatNeeded_ForChamberPressure} M^2")
    print(f"🔹 Nozzle Expansion Ratio: {NozzleExpansion:.2f} ")
    print(f"🔹 nozzle Exit Area: {NozzleExitArea} M^2")

    

Nozzle_Outputs = {
        "Actual Chamber Pressure (Pa)":          (ChamberPressurePA, "Pa", 2),
        "Actual Chamber Pressure (Psi)":         (ChamberPressurePSI, "Psi", 2),
        "Area Throat Needed":                     (AreaThroatNeeded_ForChamberPressure, "m²", 4),
        "Nozzle Expansion Ratio":                 (NozzleExpansion, "", 2),
        "Nozzle Exit Area":                       (NozzleExitArea, "m²", 4),
        "Nozzle throat raduis": (NozzleThroatRad,"m",4),
        "Nozzle exit raduis": (NozzleExitRad,"m",4)
    }
for label, (value, unit, prec) in Nozzle_Outputs.items():
    InputyOutputs.nozzle.add_output(label, value, unit, prec)

def Print_Nozzle_Outputs_Forloop():
    print("🔹 Nozzle Outputs")
    for label, (value, unit, precision) in Nozzle_Outputs.items():
        print(f"{label:<30} {value:>{7 + precision}.{precision}f} {unit}")

# Injector outputs 

if True:
    PressureDiff_PA= StartVapour_Pressure-ChamberPressurePA
    PressureDiff_Psi =PressureDiff_PA/6894.76
    AreaNeededForMass_Injector= (PrefferedMassFlowrate_SPI/(DischargeCo_SPI*np.sqrt(2*StartingDen_Liquid*PressureDiff_PA)))
    MassFlowRate_SPI= (DischargeCo_SPI*InjectorArea*np.sqrt(2*StartingDen_Liquid*PressureDiff_PA))
    InjectirDiameter_m=(np.sqrt(InjectorArea/3.14))*2

#If you want stuff printed do it here
print(PressureDiff_PA,PressureDiff_Psi,AreaNeededForMass_Injector,MassFlowRate_SPI,InjectirDiameter_m)
print(InjectorArea)

#Inputs for math...
NitrousQuality=0
RealTime=0.0
def Time():
    global RealTime
    RealTime+=Timestep


#Math computation




        
if Exportinputs==True:       
    inputs_io.export_inputs_vertical('inputs_vertical_sample.csv', globals())


    

OxtankMath=Oxtank(RealTime,Oxtanktemp,NitrousQuality,StartMass_Gas,StartMass_Liquid,StartMass_TotalOx,Timestep,MetalOxtanktemp,volumetank=OxtankVolume)
OxtankMath.StuffNoPrint(OxtankVolume, Timestep, oxtankLstart,hydroD=HydrolicDiameter,isventopen=isventopen)


FuelGrainMath=FuelGrain(FuelGrainDiameter,fuelGrain_AreaReal,FuelGrainLength,start_chamber_pressure_pa,start_chamber_temperature_k,HowmuchInj_Help,Is_fuelGrain_Helix,helixrundiameter,Is_fuelgrain_Transient,Is_start_mass_gas_input,StartMass_Gas,Is_FuelGrain_GoshaStar,OneArchlengthestimate,preccandpostvolume,start_gas_gpermole,RealTime,Is_FuelGrain_PixelMethod,helixloopdiameter=helixloopdiameter)
FuelGrainMath.stuffnoprint(Is_FuelGrain_PixelMethod,outerdiameter_inches=OuterDiameter_inches,totalComplexArea=TotalComplexArea,Is_fuelGrain_Helix=Is_fuelGrain_Helix,revPitch=revPitch,PitchFor_Helix=PitchFor_Helix)
FuelGrainMath.should_I_Print(Is_fuelGrain_Helix,Is_fuelgrain_Transient,Is_FuelGrain_GoshaStar,FuelGrainLength,start_chamber_pressure_pa,start_chamber_temperature_k,start_gas_gpermole,Is_start_mass_gas_input,StartMass_Gas,OneArchlengthestimate,helixrundiameter)

NozzleMath=Nozzle(NozzleExpansion,NozzleThroatRad,NozzleExitRad,NozzleThoatArea,NozzleExitArea)
NozzleMath.stuffnotprint(max_at_what_pressure_psi,IntailGoalOf_OF_ratio,NozzleThroatRad,NozzleExitRad,NozzleExpansion,CEAforRocket)
NozzleMath.thrustreal=0
if Is_FuelGrain_GoshaStar==True:
    FuelGrainMath.starshape(CenterCircleFuelGrain_Raduis,OutsideSmallCircle_raduis,AmountofSmallCircles,whatRaduisWouldNeedtobeOfNormalCircleToMatch,OneArchlengthestimate,TotalComplexArea,ComplexPerimeter,Fuelgrainlength=FuelGrainLength)
if Is_sim_flight==True: #fix later
    flighmath=Flight(wetmass,drymass,rocketareaM,startingoutsidetempC,PressureOutsidePa,TempchangePerMeter,realativehumidity,dragcoIfnotcdchart,df_cdinput,drouge_cd,drouge_area,infaltionint,infaltiontime,main_cd,main_area,maindelpyalt,Timestep,RealTime)
    flighmath.stuffnotprint(rocketareaM)
    if Is_FuelGrain_PixelMethod==True:
        # Run full regression analysis with all graphs (controlled by ShowPlots)
        FuelGrainMath.run_regression_analysis(r'C:\Users\gosha\Desktop\MotorModelP2\cads\goddard.obj', cross_section_axis=2, show_plots=ShowPlots)



# create a Flight instance to track simple 1D flight state alongside motor model







# Write header to CSV before loop starts
with open(Output, 'w') as f:
    f.write('')  # Clear the file
iteration_count = 0
csvlist=[]
batch_size = 50 # Write to CSV every 50 iterations

# Set up profiling
profiler = cProfile.Profile()
profiler.enable()

try:
    while OxtankMath.totaloxmass>0.02 and NozzleMath.thrustreal>=0:
        Time()
        OxtankMath.CoolOxtankProp(RealTime)
        #OxtankMath.twophaseflow_genstuff()
        OxtankMath.Massflowrate(Timestep, Is_2phase_flow_injector_model, DischargeCo_SPI, InjectorArea, chamberpressure_PA=FuelGrainMath.chamberpressure_PA)
        
        
        
        if Use_U==False:
            OxtankMath.cheatcheack()
            OxtankMath.Massesofshit()
            OxtankMath.FindVapour(ventcd=VentCD, ventarea=3.14*(VentDiameter/2)**2,ambientpressure_pa=PressureOutsidePa)
            OxtankMath.Tempofnitrous()
            OxtankMath.Derivatives()
        else:
            OxtankMath.Internal_enegry_change(ventcd=VentCD, ventarea=3.14*(VentDiameter/2)**2,ambientpressure_pa=PressureOutsidePa)
        OxtankMath.Status(StartMass_Liquid,OxamountwhenfizzstartsConstant,Is_fizz_when_equal,Fizzstart2)
        OxtankMath.Fizz(FizzCurveConstant)
        OxtankMath.NitrousQuality_func()
        OxtankMath.intailvapour()
        OxtankMath.twophaseflow_genstuff(FuelGrainMath.chamberpressure_PA,DischargeCo_SPI,InjectorArea,DischargeCo_HEM)
        if Is_vapourPhase_constant_temp==True and OxtankMath.status=="Vapour":
            OxtankMath.Vapourphase_constant_temp()
        else:
            OxtankMath.VapourPhase(thrust=NozzleMath.thrustreal)
        #OxtankMath.heattransferoxtank(oxtankID,oxtanksurfaceareainput,oxtankmass,oxtankspecheat,Timestep)
        OxtankMath.Calc_CD(chamberpressure_PA=FuelGrainMath.chamberpressure_PA)
        if Is_FuelGrain_PixelMethod==True:
            FuelGrainMath.pixelmethodgeo(Is_fuelGrain_Helix=Is_fuelGrain_Helix,FuelGrainLength=FuelGrainLength,Fuel_Density=Fuel_Density,Timestep=Timestep,amountofsmallcircles=AmountofSmallCircles)
        FuelGrainMath.oxflux_func(OxtankMath.RealMassFlowRate,RealTime)
        FuelGrainMath.regression_func(OxtankMath.status,Regression_Aco,Regression_Nco,regfluxstuff,Timestep)
        
        FuelGrainMath.simpleCircleGeo(FuelGrainLength,Timestep,Is_FuelGrain_GoshaStar,Is_fuelGrain_Helix)
        FuelGrainMath.masses_of_stuff(Fuel_Density, OxtankMath,Timestep,FuelGrainLength=FuelGrainLength)
        FuelGrainMath.Ox_Fuel_Ratio(OxtankMath.RealMassFlowRate)
        FuelGrainMath.cstar_func(CEAforRocket)
        FuelGrainMath.chamberpressure_func(NozzleMath,NozzleCD_ForChamberPressure,Ncomb_ForChamberPressure,NozzleMath.throatarea,Is_fuelgrain_Transient)
        FuelGrainMath.gammas(CEAforRocket,expansionratio=NozzleMath.ExpansionRatio)
        FuelGrainMath.throatPressure(CEAforRocket)
        FuelGrainMath.densitys(CEAforRocket)
        #FuelGrainMath.complexregression(CEAforRocket,Fuel_Density,FuelGrainLength)
        if Is_fuelGrain_Helix==True:
            FuelGrainMath.helixmath(CEAforRocket,Timestep,Is_FuelGrain_GoshaStar,Is_fuelGrain_Helix,revPitch,PitchFor_Helix,AmountofSmallCircles,Is_FuelGrain_PixelMethod,FuelGrainDiameter,expansionratio=NozzleMath.ExpansionRatio,OusideSmallCircle_raduis=OutsideSmallCircle_raduis,helical_archsegment=helicalarchsegmant,thoart_area=NozzleMath.throatarea)
        if Is_FuelGrain_GoshaStar==True:
            FuelGrainMath.complex_area_func(FuelGrainLength,Timestep,Is_fuelGrain_Helix,Is_FuelGrain_GoshaStar,AmountofSmallCircles,Fuel_Density=Fuel_Density)
        if Is_fuelgrain_Transient==True:
            FuelGrainMath.transeint_chamber_model(NozzleCD_ForChamberPressure,Ncomb_ForChamberPressure,NozzleMath,Fuel_Density,Timestep,Is_fuelgrain_Transient)
        NozzleMath.CEA_Values(FuelGrainMath.chamberpressure_PSI,FuelGrainMath.OF,CEAforRocket)
        NozzleMath.expansionratio()
        NozzleMath.Pressures(FuelGrainMath.chamberpressure_PA)
        if Is_easy_nozzle_regression==True:
            NozzleMath.easynozzleabltion(ThoartAblationRate,ExitAblationRate,Timestep,FuelGrainMath.throatpressure_PA)
        NozzleMath.Thrust(FuelGrainMath,Is_sim_flight,flighmath.pressure_pa if Is_sim_flight else PressureOutsidePa,PressureOutsidePa,NozzleExitArea,RawSillyMotorEffceincy,Ncomb_ForChamberPressure,RealTime,Timestep,mdotnozzle=FuelGrainMath.mdotnozzle)
        if Is_sim_flight==True:
            flighmath.update_thrust(thrustreal=NozzleMath.thrustreal)
            flighmath.atmosphere_update(startingoutsidetempC,PressureOutsidePa,TempchangePerMeter,realativehumidity)
            flighmath.mass_update(Timestep,FuelGrainMath.mdotnozzle,drymass)
            flighmath.dynamics_update(Timestep)
            flighmath.update_drag(df_cdinput,dragcoIfnotcdchart,rocketareaM,drouge_cd,drouge_area,infaltionint,infaltiontime,main_cd,main_area)
            flighmath.status_update(RealTime, Timestep, maindelpyalt)

            

            flighmath.add_values()
        OxtankMath.add_values()
        FuelGrainMath.add_values()
        NozzleMath.add_values()
        
        iteration_count += 1
        
        # Print progress every 50 iterations (for feedback without writing CSV yet)
        if iteration_count % batch_size == 0:
            print(f"Current time: {RealTime:.2f} s, Total iterations: {iteration_count}")
except Exception as e:
    print(f"Error in main loop: {e}")
    import traceback
    traceback.print_exc()

# Finalize output from all classes (convert all buffered rows to DataFrames)
print("Finalizing output data...")
OxtankMath.finalize_output()
FuelGrainMath.finalize_output()
NozzleMath.finalize_output()
if Is_sim_flight:
    flighmath.finalize_output()

if Is_sim_flight:
    combined_csv = pd.concat([OxtankMath.outputcsv.reset_index(drop=True),
                            FuelGrainMath.outputcsv.reset_index(drop=True),
                            NozzleMath.outputcsv.reset_index(drop=True),
                            flighmath.outputcsv.reset_index(drop=True)], axis=1)
else:
    combined_csv = pd.concat([OxtankMath.outputcsv.reset_index(drop=True),
                            FuelGrainMath.outputcsv.reset_index(drop=True),
                            NozzleMath.outputcsv.reset_index(drop=True)], axis=1)
    
combined_csv.to_csv(Output, index=False)

# Stop profiling and print results
profiler.disable()
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
ps.print_stats(30)  # Show top 30 slowest functions
print("\n" + "="*80)
print("PROFILING RESULTS - TOP 30 SLOWEST FUNCTIONS")
print("="*80)
print(s.getvalue())
print("="*80)
print(f"Total iterations: {iteration_count}")
print("="*80 + "\n")

df_outputrockets=pd.read_csv(Output)
def make_a_graph (x,y):
    
    plt.figure(f"{y} vs {x}", figsize=(6,4))
    plt.plot(df_outputrockets[x], df_outputrockets[y], label=y)
    plt.xlabel(f"{x}")
    plt.ylabel(f"{y}")
    plt.title(f"{y} vs {x}")
    plt.legend()
    plt.grid(True)

make_a_graph('time','thrustreal')

# Create ox pressure vs time graph
make_a_graph('time','realgaspressure_PA')

# Use non-blocking show so script continues and prints timing info (only if plots enabled)
if ShowPlots:
    plt.ion()  # Turn on interactive mode
    plt.show(block=False)

print("done 2")

# Calculate total impulse and average thrust by phase
thrust_col = None
time_col = None
status_col = None

# Check for thrust column (try different possible names)
for col in ['thrustreal', 'thrust', 'Thrust']:
    if col in df_outputrockets.columns:
        thrust_col = col
        break

# Check for time column
for col in ['time', 'Time', 'RealTime']:
    if col in df_outputrockets.columns:
        time_col = col
        break

# Check for status column
for col in ['status', 'Status']:
    if col in df_outputrockets.columns:
        status_col = col
        break

if thrust_col and time_col:
    time_data = df_outputrockets[time_col].values
    thrust_data = df_outputrockets[thrust_col].values
    
    # Check if data is empty
    if len(time_data) == 0 or len(thrust_data) == 0:
        print(f"\nWarning: No data in CSV. Cannot calculate motor performance.")
        print(f"Columns found: {list(df_outputrockets.columns)}")
    else:
        # Calculate total impulse using trapezoidal integration
        total_impulse = np.trapezoid(thrust_data, time_data)
        total_time = time_data[-1] - time_data[0]
        avg_thrust_overall = total_impulse / total_time if total_time > 0 else 0
        
        print(f"\nMotor Performance Summary:")
        print(f"  Total Impulse: {total_impulse:.1f} N·s")
        print(f"  Overall Average Thrust: {avg_thrust_overall:.1f} N")
        print(f"  Total Burn Time: {total_time:.2f} s")
        
        # Analyze by phase if status column exists
        if status_col:
            print(f"\nThrust by Phase:")
            phases = ['Liquid', 'fizz', 'Vapour']
            for phase in phases:
                phase_mask = df_outputrockets[status_col] == phase
                if phase_mask.any():
                    phase_time = time_data[phase_mask]
                    phase_thrust = thrust_data[phase_mask]
                    
                    if len(phase_time) > 0:
                        phase_duration = phase_time[-1] - phase_time[0]
                        phase_impulse = np.trapezoid(phase_thrust, phase_time) if len(phase_time) > 1 else 0
                        phase_avg_thrust = phase_impulse / phase_duration if phase_duration > 0 else 0
                        
                        print(f"  {phase}:")
                        print(f"    Duration: {phase_duration:.2f} s")
                        print(f"    Average Thrust: {phase_avg_thrust:.1f} N")
                        print(f"    Impulse: {phase_impulse:.1f} N·s")
        
        # Motor designation (based on impulse: NFPA designation)
        # Standard designations: A (1.26-5.1), B (5.1-10.16), C (10.16-20.32), etc.
        if total_impulse < 5.1:
            designation = 'A'
        elif total_impulse < 10.16:
            designation = 'B'
        elif total_impulse < 20.32:
            designation = 'C'
        elif total_impulse < 40.64:
            designation = 'D'
        elif total_impulse < 81.28:
            designation = 'E'
        elif total_impulse < 162.56:
            designation = 'F'
        elif total_impulse < 325.12:
            designation = 'G'
        elif total_impulse < 650.24:
            designation = 'H';
        elif total_impulse < 1300.48:
            designation = 'I'
        elif total_impulse < 2600.96:
            designation = 'J'
        elif total_impulse < 5201.92:
            designation = 'K'
        elif total_impulse < 10403.84:
            designation = 'L'
        elif total_impulse < 20807.68:
            designation = 'M'
        elif total_impulse < 41615.36:
            designation = 'N'
        elif total_impulse < 83230.72:
            designation = 'O'
        elif total_impulse < 166461.44:
            designation = 'P'
        elif total_impulse < 332922.88:
            designation = 'Q'
        else:
            designation = 'R'
        
        print(f"\nMotor Designation (NFPA): {designation}{avg_thrust_overall:.0f}-{total_time:.0f}")
else:
    print(f"\nWarning: Could not find thrust/time columns in CSV")
    print(f"Available columns: {list(df_outputrockets.columns)}")

# Print elapsed time
elapsed_time = time_module.time() - script_start_time
print(f"\n{'='*50}")
print(f"Total runtime: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print(f"{'='*50}")
if ShowPlots:
    plt.ioff()  # Turn off interactive mode
    plt.show(block=True)  # Final blocking show to keep graph open until closed

# Auto-export inputs to vertical CSV

#fix the way you call what to print
#gui
#get from exel
#make exel
#add loading
