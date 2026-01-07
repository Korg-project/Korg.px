"""
Physical constants used throughout Korg.

All constants are in CGS units unless otherwise specified.
"""

# Enable 64-bit precision in JAX for accurate spectral synthesis.
# This must be done before any JAX operations are performed.
import jax
jax.config.update("jax_enable_x64", True)

# Maximum atomic number supported (H=1 through U=92)
MAX_ATOMIC_NUMBER = 92

# Boltzmann constant
kboltz_cgs = 1.380649e-16  # erg/K

# Planck constant
hplanck_cgs = 6.62607015e-27  # erg*s

# Speed of light
c_cgs = 2.99792458e10  # cm/s

# Electron mass
electron_mass_cgs = 9.1093897e-28  # g

# Electron charge (electrostatic units)
electron_charge_cgs = 4.80320425e-10  # statcoulomb or cm^3/2 * g^1/2 / s

# Atomic mass unit
amu_cgs = 1.6605402e-24  # g

# Bohr radius (2018 CODATA recommended value)
bohr_radius_cgs = 5.29177210903e-9  # cm

# Solar mass
solar_mass_cgs = 1.9884e33  # g

# Gravitational constant
G_cgs = 6.67430e-8

# Electron volt to ergs conversion
eV_to_cgs = 1.602e-12  # ergs per eV

# Constants in eV units
kboltz_eV = 8.617333262145e-5  # eV/K
hplanck_eV = 4.135667696e-15  # eV*s
RydbergH_eV = 13.598287264  # eV
Rydberg_eV = 13.605693122994  # eV (2018 CODATA via wikipedia)
