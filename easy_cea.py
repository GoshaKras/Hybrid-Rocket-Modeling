from __future__ import annotations

import argparse
import os

os.environ.setdefault("CEA_USE_SITE_PACKAGES", "1")

import prop_maps
from CEA_Wrap import Fuel, Oxidizer, RocketProblem

DEFAULT_PRESSURE_PSI = 400
DEFAULT_O_F = 6

CEA_OUTPUT_ALIASES = {
    "cstar": "cstar",
    "prandtl": "pran",
    "pran": "pran",
    "mach": "mach",
    "gamma": "gamma",
    "pressure": "p",
    "chamber_pressure": "c_p",
    "throat_pressure": "t_p",
    "exit_pressure": "p",
    "temperature": "t",
    "chamber_temperature": "c_t",
    "throat_temperature": "t_t",
    "exit_temperature": "t_t",
    "sonic_velocity": "son",
    "molecular_weight": "mw",
    "viscosity": "visc",
    "heat_capacity": "cp",
    "chamber_density": "c_rho",
}

CEA_OUTPUT_LABELS = {
    "cstar": "cstar",
    "pran": "Prandtl number",
    "mach": "Mach number",
    "gamma": "Gamma",
    "p": "Pressure",
    "c_p": "Chamber pressure",
    "t_p": "Throat pressure",
    "t": "Temperature",
    "c_t": "Chamber temperature",
    "t_t": "Throat temperature",
    "son": "Sonic velocity",
    "mw": "Molecular weight",
    "visc": "Viscosity",
    "cp": "Heat capacity",
}


def build_materials() -> list:
    if "HTPB" not in prop_maps.PROP_MAP:
        raise KeyError("HTPB is missing from prop_maps.PROP_MAP")
    if "N2O" not in prop_maps.PROP_MAP:
        raise KeyError("N2O is missing from prop_maps.PROP_MAP")

    fuel = Fuel("HTPB")
    oxidizer = Oxidizer("N2O")
    return [fuel, oxidizer]


def get_cea_output(
    property_name: str,
    pressure_psi: float = DEFAULT_PRESSURE_PSI,
    o_f: float = DEFAULT_O_F,
    should_print: bool = False,
):
    property_key = CEA_OUTPUT_ALIASES.get(property_name.lower(), property_name)
    materials = build_materials()
    problem = RocketProblem(pressure=pressure_psi, o_f=o_f, materials=materials)
    data = problem.run()
    if not hasattr(data, property_key):
        raise KeyError(f"CEA output '{property_name}' is not available")
    value = getattr(data, property_key)

    if should_print:
        label = CEA_OUTPUT_LABELS.get(property_key, property_name)
        print(f"{label}: {value}")

    return value


def get_cstar_and_prandtl(pressure_psi: float = DEFAULT_PRESSURE_PSI, o_f: float = DEFAULT_O_F) -> tuple[float, float]:
    cstar_value = get_cea_output("cstar", pressure_psi, o_f)
    prandtl_value = get_cea_output("prandtl", pressure_psi, o_f)
    return cstar_value, prandtl_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Print HTPB/N2O CEA outputs using CEA_Wrap.")
    parser.add_argument("--pressure", type=float, default=DEFAULT_PRESSURE_PSI, help="Chamber pressure in psi")
    parser.add_argument("--of", type=float, default=DEFAULT_O_F, help="Oxidizer-to-fuel ratio")
    parser.add_argument(
        "--output",
        choices=tuple(CEA_OUTPUT_ALIASES.keys()) + ("both",),
        default="both",
        help="Which value to print",
    )
    parser.add_argument("--print", dest="should_print", action="store_true", help="Print the selected output")
    args = parser.parse_args()

    if args.output == "both":
        cstar_m_s, prandtl_number = get_cstar_and_prandtl(args.pressure, args.of)
        if args.should_print:
            print("HTPB / N2O CEA result")
            print(f"Pressure: {args.pressure:.2f} psi")
            print(f"O/F: {args.of:.3f}")
            print(f"cstar: {cstar_m_s:.2f} m/s")
            print(f"Prandtl number: {prandtl_number:.4f}")
    else:
        output_value = get_cea_output(args.output, args.pressure, args.of, should_print=args.should_print)
        if args.should_print:
            print("HTPB / N2O CEA result")
            print(f"Pressure: {args.pressure:.2f} psi")
            print(f"O/F: {args.of:.3f}")
            if args.output == "cstar":
                print(f"cstar: {output_value:.2f} m/s")
            else:
                print(f"{args.output}: {output_value}")


#if __name__ == "__main__":
    #main()
pr=get_cea_output("prandtl", pressure_psi=300, o_f=5, should_print=False)
print(pr)