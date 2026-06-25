import pandas as pd
import numpy as np
import sys
import re

def get_input_vars_from_motormodel():
    """Dynamically extract input variable names from motormodel_copy2.py (lines 113-180) in order of appearance."""
    try:
        with open('motormodel_copy2.py', 'r') as f:
            lines = f.readlines()
        
        # Extract lines 113-180 (the main input section)
        code_section = ''.join(lines[112:180])  # 0-indexed, so line 113 = index 112
        
        # Find all variable assignments (pattern: variable_name = value)
        # Matches: word = (anything) but excludes comments and multi-word statements
        pattern = r'^(\w+)\s*='
        variables = []  # Use list to preserve order
        seen = set()  # Track seen variables to avoid duplicates
        
        for line in code_section.split('\n'):
            # Skip comments and special lines
            if line.strip().startswith('#') or 'if ' in line or 'for ' in line or 'class ' in line:
                continue
            match = re.match(pattern, line.strip())
            if match:
                var_name = match.group(1)
                # Exclude common non-input variables
                if var_name not in ['df_cdinput', 'OUTPUT'] and var_name not in seen:
                    variables.append(var_name)
                    seen.add(var_name)
        
        return variables  # Return in order of appearance
    except Exception as e:
        print(f"Warning: Could not auto-detect variables: {e}")
        return []

# Try to auto-detect, fall back to manual list if it fails
input_vars = get_input_vars_from_motormodel()
if not input_vars:
    # Fallback manual list - only INPUT variables, not calculated ones
    input_vars = [
        'Want_ThrustCurve','Timestep','outsidePressure_PA','UploadFromExel',
        'Oxtanktemp','MetalOxtanktemp','calculate_volume_YN','volumeinput','oxtanklength','oxtankOD','oxtankID','oxtankspecheat','oxtankmass','WhatOxidizer','oxtankLstart','Ullage','OxamountwhenfizzstartsConstant','Fizzstart2','FizzCurveConstant','NitrousGamma','vapour_temp_damping',
        'Is_FuelGrain_just_a_circle','FuelGrainRadius','Regression_Aco','Regression_Nco','Fuel_Density','FuelGrainLength','IntailGoalOf_OF_ratio','InitailGoalof_Chamberpressure_Psi','OuterGrainRaduis','OuterDiameter_inches','RawSillyMotorEffceincy','Ncomb_ForChamberPressure','NozzleCD_ForChamberPressure','start_chamber_pressure_pa','start_chamber_temperature_k','regfluxstuff','preccandpostvolume','start_gas_gpermole',
        'Is_FuelGrain_GoshaStar','CenterCircleFuelGrain_Raduis','AmountofSmallCircles','OutsideSmallCircle_raduis','revPitch','helixrundiameter','WhatFuel','TimeWhenBoostSops',
        'Is_fuelGrain_Helix','Is_FuelGrain_PixelMethod','Is_fuelgrain_Transient','Is_start_mass_gas_input','Is_vapourPhase_constant_temp','Is_sim_flight','Is_2phase_flow_injector_model','Is_easy_nozzle_regression','Is_fizz_when_equal','Is_print_pixelstuff',
        'InjectorArea','DischargeCo_SPI','DischargeCo_HEM','PrefferedMassFlowrate_SPI','HowmuchInj_Help','HydrolicDiameter','VentDiameter','VentCD',
        'NozzleThoatArea','NozzleExitArea','TakeExpansionAsInput','NozzleExpansionAs_Input','ThoartAblationRate','ExitAblationRate','max_at_what_pressure_psi','max_at_what_OF_ratio',
        'wetmass','drymass','burntimecutoff','dragcofrominputcsv','dragcoIfnotcdchart','drouge_area','drouge_cd','main_area','main_cd','infaltiontime','rocketdiameterIn','startingoutsidetempC','TempchangePerMeter','pressureOutsideIn','realativehumidity','maindelpyalt',
        'Exportinputs'
    ]

def export_inputs(filename, namespace):
    """Export input variables to a CSV file, only including variables that exist in the namespace."""
    # Only export variables that actually exist in the namespace (are defined)
    existing_vars = [var for var in input_vars if var in namespace]
    data = {var: namespace.get(var, None) for var in existing_vars}
    df = pd.DataFrame([data])
    df.to_csv(filename, index=False)
    print(f"Exported {len(existing_vars)} inputs to {filename}")

def import_inputs(filename, namespace):
    """Import input variables from a CSV file and update the namespace."""
    df = pd.read_csv(filename)
    for var in input_vars:
        if var in df.columns:
            val = df[var].iloc[0]
            # Try to convert to float/int/bool if possible
            if isinstance(val, str):
                if val.lower() == 'true':
                    val = True
                elif val.lower() == 'false':
                    val = False
            try:
                val = float(val)
                if val.is_integer():
                    val = int(val)
            except Exception:
                pass
            namespace[var] = val
    print(f"Imported inputs from {filename}")

def import_inputs_vertical(filename, namespace):
    """Import input variables from a vertical CSV (name, value, unit) and update the namespace."""
    df = pd.read_csv(filename)
    for _, row in df.iterrows():
        var = row['name']
        val = row['value']
        # Try to convert to float/int/bool if possible
        if isinstance(val, str):
            if val.lower() == 'true':
                val = True
            elif val.lower() == 'false':
                val = False
        try:
            val = float(val)
            if isinstance(val, float) and val.is_integer():
                val = int(val)
        except Exception:
            pass
        namespace[var] = val
    print(f"Imported vertical inputs from {filename}")

def export_inputs_vertical(filename, namespace):
    """Export input variables to a vertical CSV file (name, value, unit)."""
    # Map of variable names to their units
    units_map = {
        'Want_ThrustCurve': 'bool', 'Timestep': 's', 'outsidePressure_PA': 'Pa', 'UploadFromExel': 'bool', 'Exportinputs': 'bool',
        'Oxtanktemp': 'K', 'MetalOxtanktemp': 'K', 'calculate_volume_YN': 'bool', 'volumeinput': 'm^3', 'oxtanklength': 'm',
        'oxtankOD': 'm', 'oxtankID': 'm', 'oxtankspecheat': 'J/kgK', 'oxtankmass': 'kg',
        'oxtanksurfaceareainput': 'm^2', 'WhatOxidizer': 'str', 'oxtankLstart': 'm',
        'Calced_OxtankVolume': 'm^3', 'OxtankVolume': 'm^3', 'Ullage': 'percent',
        'OxamountwhenfizzstartsConstant': 'unitless', 'Fizzstart2': 'unitless', 'FizzCurveConstant': 'unitless', 'NitrousGamma': 'unitless', 'vapour_temp_damping': 'unitless',
        'Is_FuelGrain_just_a_circle': 'bool', 'FuelGrainRadius': 'm', 'FuelGrainDiameter': 'm', 'Regression_Aco': 'unitless',
        'Regression_Nco': 'unitless', 'Fuel_Density': 'kg/m^3', 'FuelGrainLength': 'm',
        'IntailGoalOf_OF_ratio': 'unitless', 'InitailGoalof_Chamberpressure_Psi': 'psi',
        'InitailGoalof_Chamberpressure_Pa': 'Pa', 'OuterGrainRaduis': 'm', 'OuterDiameter_inches': 'inches',
        'RawSillyMotorEffceincy': 'unitless', 'Ncomb_ForChamberPressure': 'unitless',
        'NozzleCD_ForChamberPressure': 'unitless', 'start_chamber_pressure_pa': 'Pa',
        'start_chamber_temperature_k': 'K', 'regfluxstuff': 'unitless', 'preccandpostvolume': 'm^3', 'start_gas_gpermole': 'g/mol',
        'Is_FuelGrain_GoshaStar': 'bool',
        'CenterCircleFuelGrain_Raduis': 'm', 'AmountofSmallCircles': 'unitless',
        'OutsideSmallCircle_raduis': 'm', 'revPitch': 'unitless', 'Pitchlength': 'm',
        'helixrundiameter': 'm', 'WhatFuel': 'str', 'TimeWhenBoostSops': 's',
        'Is_fuelGrain_Helix': 'bool', 'Is_FuelGrain_PixelMethod': 'bool', 'Is_fuelgrain_Transient': 'bool',
        'Is_start_mass_gas_input': 'bool', 'Is_vapourPhase_constant_temp': 'bool', 'Is_sim_flight': 'bool',
        'Is_2phase_flow_injector_model': 'bool', 'Is_easy_nozzle_regression': 'bool', 'Is_fizz_when_equal': 'bool', 'Is_print_pixelstuff': 'bool',
        'InjectorArea': 'm^2', 'DischargeCo_SPI': 'unitless', 'DischargeCo_HEM': 'unitless',
        'PrefferedMassFlowrate_SPI': 'kg/s', 'HowmuchInj_Help': 'unitless', 'HydrolicDiameter': 'm',
        'VentDiameter': 'm', 'VentCD': 'unitless', 'NozzleThoatArea': 'm^2',
        'NozzleExitArea': 'm^2', 'TakeExpansionAsInput': 'bool', 'NozzleExpansionAs_Input': 'unitless',
        'NozzleExpansion': 'unitless', 'NozzleThroatRad': 'm', 'NozzleExitRad': 'm',
        'ThoartAblationRate': 'm/s', 'ExitAblationRate': 'm/s', 'max_at_what_pressure_psi': 'psi', 'max_at_what_OF_ratio': 'unitless',
        'wetmass': 'kg', 'drymass': 'kg', 'burntimecutoff': 's', 'dragcofrominputcsv': 'bool', 'dragcoIfnotcdchart': 'unitless',
        'drouge_area': 'm^2', 'drouge_cd': 'unitless', 'main_area': 'm^2',
        'main_cd': 'unitless', 'infaltiontime': 's', 'infaltionint': 'unitless',
        'rocketdiameterIn': 'in', 'rocketdiameterM': 'm', 'rocketareaM': 'm^2',
        'startingoutsidetempC': 'C', 'TempchangePerMeter': 'C/m',
        'pressureOutsideIn': 'inHg', 'PressureOutsidePa': 'Pa',
        'realativehumidity': 'percent', 'maindelpyalt': 'm',
    }
    
    rows = []
    for var in input_vars:
        val = namespace.get(var, None)
        unit = units_map.get(var, 'unitless')
        rows.append({'name': var, 'value': val, 'unit': unit})
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Exported vertical inputs to {filename}")

# Example usage in motormodel_copy2.py:
# import inputs_io
# inputs_io.export_inputs('inputs_export.csv', globals())
# inputs_io.import_inputs('inputs_export.csv', globals())
# inputs_io.export_inputs_vertical('inputs_vertical_sample.csv', globals())
