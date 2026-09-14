from CoolProp.CoolProp import PropsSI
import numpy as np
import matplotlib.pyplot as plt



def Is_choked(P1, P2, T, fluid):
    specheatratio = (
        PropsSI('Cpmass', 'T', T, 'P', P1, fluid)
        / PropsSI('Cvmass', 'T', T, 'P', P1, fluid)
    )
   
    critPratio=(2/(specheatratio+1))**(specheatratio/(specheatratio-1))
    return critPratio, specheatratio, P2/P1 < critPratio

def PSI_PA(PSI):
    return PSI * 6894.76

def mdot(P1, P2, T, fluid,A,Cd):
    if P2 >= P1:
        return 0.0, 0.0

    density = PropsSI('Dmass', 'T', T, 'P', P1, fluid)
    specenthalpy = PropsSI('Hmass', 'T', T, 'P', P1, fluid)
    critPratio,specheatratio,C_status = Is_choked(P1, P2, T, fluid)
    if C_status==True:
        critDensity = density * (2/(specheatratio+1))**(1/(specheatratio-1))
        critPressure=critPratio*P1
        mdot = Cd * A * np.sqrt(specheatratio * critPressure * critDensity)
        
    else:
        correction = comp_correction(P1, P2, specheatratio)
        mdot = Cd * A * correction * np.sqrt(2 * density * (P1 - P2))
    Hdot=mdot*specenthalpy
    return mdot, Hdot

def comp_correction(P1, P2, gamma):
    
    pressure_ratio = P2 / P1
    if P1 <= 0 or not 0 < pressure_ratio <= 1:
        raise ValueError("Require P1 > 0 and 0 < P2 <= P1.")
    if gamma <= 1:
        raise ValueError("gamma must be greater than 1.")

    return np.sqrt(
        pressure_ratio ** (2 / gamma)
        * gamma / (gamma - 1)
        * (1 - pressure_ratio ** ((gamma - 1) / gamma))
        / (1 - pressure_ratio)
    ) if pressure_ratio < 1 else 1.0
def find_volume(q,h,misc):
    volume= (((0.25*0.25*q*3.14/4)+(0.5*0.5*h*3.14/4))*0.000016387)+misc
    return volume
volume=find_volume(10, 0, 68*0.000016387)
print(f"Volume: {volume:.6f} m³,volume in imperial units: {volume/0.000016387:.6f} in³")
def sim_filling(volume, goalpressure, timestep, fluid, upPressure, area_orf, print_status):
    time=0
    next_status_time = 1.0
    volume=volume
    pressure=101325
    tempcont=298.15
    temp=298.15
    mass = PropsSI('Dmass', 'T', temp, 'P', pressure, fluid) * volume
    intenergy=PropsSI('Umass', 'T', temp, 'P', pressure, fluid) * mass
    while pressure < PSI_PA(goalpressure):
        time += timestep
        mass_flow, enthalpy_flow = mdot(upPressure, pressure, tempcont, fluid, area_orf, 0.6)
        mass += mass_flow * timestep
        intenergy += enthalpy_flow * timestep
        specinternalenergy= intenergy / mass
        density= mass / volume
        temp = PropsSI('T', 'Dmass', density, 'Umass', specinternalenergy, fluid)
        pressure = PropsSI('P', 'T', temp, 'Dmass', density, fluid)

        if print_status and time >= next_status_time:
            print(
                f"Time: {time:.2f} s, Pressure: {pressure / 6894.76:.2f} psi, "
                f"Temperature: {temp:.2f} K, Mass: {mass:.6f} kg"
            )
            next_status_time += 1.0

    pressure_psi = pressure / 6894.76
    if print_status:
        print(
            f"Final: Time: {time:.2f} s, Pressure: {pressure_psi:.2f} psi, "
            f"Temperature: {temp:.2f} K, Mass: {mass:.6f} kg"
        )
    return time
def area(diameter):
    diameter= diameter*0.0254
    return 3.14159 * (diameter/2)**2
area_orf=area(0.01)
#sim_filling(volume, 860, 0.01, 'H2', PSI_PA(1000), area_orf=area_orf, print_status=True)

print_mdot=True
print_fill=True
if print_mdot==True:
    x_values=np.geomspace(0.0025, 0.04, 35)
    y_values= []
    for simulation_number, diameter in enumerate(x_values, start=1):
            print(f"Simulation {simulation_number}/{len(x_values)}")
            mdot_calc,hdot_calc=mdot(PSI_PA(730), 101325, 298.15, 'O2', area(diameter), 0.6)
        
            y_values.append(mdot_calc)
    y_values = np.array(y_values)
    plt.figure(figsize=(8, 5))
    plt.semilogx(x_values, y_values, marker='o', markersize=3)
    plt.xlabel('Orifice diameter (in)')
    plt.ylabel('mdot_vent (kg/s)')
    plt.title('Orifice Diameter vs. mdotvent')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if print_fill==True:
    x_values=np.geomspace(0.0025, 0.03, 35)
    y_values = []
    for simulation_number, diameter in enumerate(x_values, start=1):
        print(f"Simulation {simulation_number}/{len(x_values)}")
        fill_time = sim_filling(
            volume, 730, 0.01, 'O2', PSI_PA(1000),
            print_status=False, area_orf=area(diameter)
        )
        y_values.append(fill_time)
    y_values = np.array(y_values)

    plt.figure(figsize=(8, 5))
    plt.semilogx(x_values, y_values, marker='o', markersize=3)
    plt.xlabel('Orifice diameter (in)')
    plt.ylabel('Fill time to 730 psi (s)')
    plt.title('Orifice Diameter vs. Fill Time')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()





