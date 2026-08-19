"""Solve Eq. 43 for the chamber friction factor, f_c."""

import numpy as np
from scipy.optimize import brentq


def calculate_fc(epsilon, r0, R):
    """Return ``F_c`` from the supplied Reynolds-free implicit relation.

    1 / sqrt(F_c) = -0.923 * ln(
        0.104 * epsilon/F_c * sqrt(r0/R)
    )

    ``r0`` and ``R`` must use the same length unit.
    """
    if min(epsilon, r0, R) <= 0:
        raise ValueError("epsilon, r0, and R must all be positive")

    radius_ratio_sqrt = np.sqrt(r0 / R)

    def equation_residual(fc):
        logarithm_argument = 0.104 * epsilon / fc * radius_ratio_sqrt
        return 1 / np.sqrt(fc) + 0.923 * np.log(logarithm_argument)

    # Eq. 43 is implicit in f_c. Find a positive bracket, then solve it.
    candidates = np.logspace(-12, 0, 1000)
    residuals = [equation_residual(fc) for fc in candidates]
    for lower, upper, f_lower, f_upper in zip(
        candidates[:-1], candidates[1:], residuals[:-1], residuals[1:]
    ):
        if f_lower == 0:
            return lower
        if f_lower * f_upper < 0:
            return brentq(equation_residual, lower, upper)

    raise ValueError("The Reynolds-free equation has no positive F_c solution between 1e-12 and 1")


def calculate_tau_w(f_darcy, gas_density, velocity):
    """Return tau_w = (f_D / 8) * rho * U^2."""
    if min(f_darcy, gas_density, velocity) <= 0:
        raise ValueError("f_darcy, gas_density, and velocity must be positive")
    # tau_w = (f_D / 8) * rho * U^2
    return (f_darcy / 8) * gas_density * velocity ** 2


def calculate_blowing_from_shear(surface_mass_flux, edge_velocity, tau_w):
    """Return beta_bl = ((rho*v)_s * u_e) / tau_w."""
    if min(surface_mass_flux, edge_velocity, tau_w) <= 0:
        raise ValueError("surface_mass_flux, edge_velocity, and tau_w must be positive")
    # beta_bl = ((rho*v)_s * u_e) / tau_w
    return surface_mass_flux * edge_velocity / tau_w


def calculate_blowing_from_cf(surface_mass_flux, gas_density, edge_velocity, cf):
    """Return beta_bl = (rho*v)_s / (rho_e * u_e * c_f / 2)."""
    if min(surface_mass_flux, gas_density, edge_velocity, cf) <= 0:
        raise ValueError("surface_mass_flux, gas_density, edge_velocity, and cf must be positive")
    return surface_mass_flux / (gas_density * edge_velocity * cf / 2)


# Starter values for the current hybrid-motor geometry.  r0 and R are the
# initial port radius. Epsilon is the relative roughness term used by the
# supplied equation.

R0_M = 0.05490841665503017/2
R_M = 0.0864976836337236/2 
EPSILON = (100.0*10**-6)/R0_M
#EPSILON = 
# Representative first-iteration chamber-flow values.  ``SURFACE_MASS_FLUX``
# is (rho*v)_s: fuel mass flow divided by total burning surface area.
GAS_DENSITY_KG_M3 = 2.3135844874357803
EDGE_VELOCITY_M_S = 1354.8140684211392
SURFACE_MASS_FLUX_KG_M2_S = 1.3


if __name__ == "__main__":
    fc = calculate_fc(EPSILON, R0_M, R_M)
    f_darcy = fc  # Per your instruction: f_D = f_c.
    cf_for_blowing =  0.007144848566594666 # Replace this with a separate supplied c_f if needed.
    tau_w = calculate_tau_w(f_darcy, GAS_DENSITY_KG_M3, EDGE_VELOCITY_M_S)
    blowing_from_shear = calculate_blowing_from_shear(
        SURFACE_MASS_FLUX_KG_M2_S,
        EDGE_VELOCITY_M_S,
        tau_w,
    )
    blowing_from_cf = calculate_blowing_from_cf(
        SURFACE_MASS_FLUX_KG_M2_S,
        GAS_DENSITY_KG_M3,
        EDGE_VELOCITY_M_S,
        cf_for_blowing,
    )
    print("Reynolds-free F_c inputs:")
    print(f"  epsilon = {EPSILON:g}")
    print(f"  r0      = {R0_M:.4f} m")
    print(f"  R       = {R_M:.4f} m")
    print(f"  F_c     = {fc:.8g}")
    print("\nWall-shear and blowing inputs:")
    print(f"  rho_e   = {GAS_DENSITY_KG_M3:.4g} kg/m^3")
    print(f"  u_e     = {EDGE_VELOCITY_M_S:.4g} m/s")
    print(f"  (rho*v)_s = {SURFACE_MASS_FLUX_KG_M2_S:.4g} kg/(m^2 s)")
    print(f"  f_D     = {f_darcy:.8g}  (set equal to f_c)")
    print(f"  c_f     = {cf_for_blowing:.8g}")
    print(f"  tau_w   = {tau_w:.8g} Pa")
    print("\nBlowing-number comparison:")
    print(f"  beta from ((rho*v)_s*u_e)/tau_w = {blowing_from_shear:.8g}")
    print(f"  beta from (rho*v)_s/(rho_e*u_e*f_c/2) = {blowing_from_cf:.8g}")
    print(f"  absolute difference = {abs(blowing_from_shear - blowing_from_cf):.3e}")
blowingratio=(blowing_from_cf/blowing_from_shear)**0.77
print(f"blowing ratio = {blowingratio:.8g}")
blowingratio2=(4*cf_for_blowing/(4*fc))**0.77
print(f"blowing ratio 2 = {blowingratio2:.8g}")