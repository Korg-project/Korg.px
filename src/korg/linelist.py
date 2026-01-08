"""
Line list data structures and parsing functions.

This module provides the Line dataclass and functions for parsing various linelist formats
including VALD, Kurucz, MOOG, Turbospectrum, and ExoMol.
"""

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Union, Optional
import math

from .species import Species
from .constants import (
    c_cgs, electron_charge_cgs, electron_mass_cgs, hplanck_eV,
    kboltz_cgs, kboltz_eV, bohr_radius_cgs, RydbergH_eV, Rydberg_eV
)
from .data_loader import ionization_energies


@dataclass(frozen=True)
class Line:
    """
    Represents an individual spectral line.

    Attributes:
        wl: Wavelength in cm (converted from Å if input >= 1)
        log_gf: Log base 10 of oscillator strength times statistical weight
        species: Species object for this line
        E_lower: Lower energy level in eV (excitation potential)
        gamma_rad: Radiative damping parameter in rad/s (FWHM)
        gamma_stark: Stark broadening parameter in rad/s (FWHM) at 10,000 K
        vdW: Tuple (γ_vdW or σ, -1 or α) for van der Waals broadening
             - If second element is -1: first element is γ_vdW in rad/s at 10,000 K
             - Otherwise: (σ, α) are ABO theory parameters
    """
    wl: float
    log_gf: float
    species: Species
    E_lower: float
    gamma_rad: float
    gamma_stark: float
    vdW: Tuple[float, float]

    def __repr__(self):
        wl_angstrom = self.wl * 1e8
        return (f"{self.species} {wl_angstrom:.6f} Å "
                f"(log gf = {self.log_gf:.2f}, χ = {self.E_lower:.2f} eV)")


def approximate_radiative_gamma(wl: float, log_gf: float) -> float:
    """
    Approximate radiative broadening parameter.

    Args:
        wl: Wavelength in cm
        log_gf: Log of oscillator strength times statistical weight

    Returns:
        Radiative damping parameter in rad/s (FWHM)
    """
    return (8 * jnp.pi**2 * electron_charge_cgs**2 /
            (electron_mass_cgs * c_cgs * wl**2) * 10**log_gf)


def approximate_gammas(
    wl: float,
    species: Species,
    E_lower: float,
    ionization_energies_dict: dict = None
) -> Tuple[float, float]:
    """
    Approximate Stark and van der Waals broadening parameters using simplified
    Unsöld (1955) approximation for vdW and Cowley (1971) approximation for Stark,
    evaluated at 10,000 K.

    Args:
        wl: Wavelength in cm
        species: Species object
        E_lower: Lower energy level in eV
        ionization_energies_dict: Dict of ionization energies (defaults to global)

    Returns:
        Tuple of (γ_stark, log10(γ_vdW)) in rad/s, per-perturber quantities
        For autoionizing lines (E_upper > χ), returns 0.0 for γ_vdW.
        These are FWHM, not HWHM, of the Lorentzian component.
    """
    if ionization_energies_dict is None:
        ionization_energies_dict = ionization_energies

    Z = species.charge + 1  # Z is ionization stage, not atomic number

    # Molecules and highly ionized species
    if species.formula.is_molecule() or Z > 3:
        return 0.0, 0.0

    # Get ionization energy
    # Find the first non-zero atom (formula.atoms is zero-padded)
    non_zero_atoms = [int(a) for a in species.formula.atoms if a != 0]
    if len(non_zero_atoms) == 0:
        raise ValueError(f"Species {species} has no atoms in formula")
    atom_number = non_zero_atoms[0]
    # ionization_energies_dict maps atomic number to array [χ₁, χ₂, χ₃]
    # Z is ionization stage (1 for neutral, 2 for singly ionized, etc.)
    # Python is 0-indexed, so use Z-1
    chi = ionization_energies_dict[atom_number][Z - 1]

    # Calculate upper energy level
    E_upper = E_lower + (hplanck_eV * c_cgs / wl)

    # Effective quantum number for upper level
    nstar4_upper = (Z**2 * RydbergH_eV / (chi - E_upper))**2

    # Stark broadening from Cowley (1971)
    if Z == 1:
        # Equation 5 evaluated at T=10,000 K
        gamma_stark = 2.25910152e-7 * nstar4_upper
    else:
        # Equation 6 evaluated at T=10,000 K
        gamma_stark = 5.42184365e-7 * nstar4_upper / (Z + 1)**2

    # van der Waals broadening
    # Change in <r²> between lower and upper levels
    Delta_rbar2 = (5/2) * Rydberg_eV**2 * Z**2 * (
        1 / (chi - E_upper)**2 - 1 / (chi - E_lower)**2
    )

    # Check for autoionizing line
    if chi < E_upper:
        log_gamma_vdW = 0.0  # Will be interpreted as γ, not log γ
    else:
        # From Rutten's course notes / Gray (2005) eqs 11.29 and 11.30
        log_gamma_vdW = (6.33 + 0.4 * jnp.log10(Delta_rbar2) +
                         0.3 * jnp.log10(10_000) + jnp.log10(kboltz_cgs))

    return gamma_stark, log_gamma_vdW


def create_line(
    wl: float,
    log_gf: float,
    species: Union[Species, str],
    E_lower: float,
    gamma_rad: Optional[float] = None,
    gamma_stark: Optional[float] = None,
    vdW: Optional[Union[float, Tuple[float, float]]] = None,
    ionization_energies_dict: dict = None
) -> Line:
    """
    Create a Line object with automatic approximation of missing broadening parameters.

    Args:
        wl: Wavelength (assumed cm if < 1, otherwise Å)
        log_gf: Log of oscillator strength times statistical weight
        species: Species object or string
        E_lower: Lower energy level in eV
        gamma_rad: Radiative damping (rad/s), approximated if None
        gamma_stark: Stark broadening (rad/s) at 10,000 K, approximated if None
        vdW: van der Waals broadening parameter, approximated if None
             Can be:
             - Negative: interpreted as log10(γ_vdW)
             - 0: no vdW broadening
             - 0 < vdW < 20: fudge factor for Unsöld approximation
             - >= 20: packed ABO parameters
             - Tuple: (σ, α) ABO parameters
        ionization_energies_dict: Dict of ionization energies (defaults to global)

    Returns:
        Line object with all broadening parameters filled
    """
    if ionization_energies_dict is None:
        ionization_energies_dict = ionization_energies

    # Convert species if needed
    if isinstance(species, str):
        species = Species(species)

    # Convert wavelength to cm if in Angstroms
    if wl >= 1:
        wl = wl * 1e-8

    # Approximate missing broadening parameters
    # Note: Julia treats both 0 and 1 as flags to approximate (missing/placeholder values)
    need_stark = (gamma_stark is None or np.isnan(gamma_stark) or
                  gamma_stark == 0.0 or gamma_stark == 1.0)
    need_vdW = (vdW is None or (not isinstance(vdW, tuple) and np.isnan(vdW)))

    if need_stark or need_vdW:
        gamma_stark_approx, vdW_approx = approximate_gammas(
            wl, species, E_lower, ionization_energies_dict
        )
        if need_stark:
            gamma_stark = gamma_stark_approx
        if need_vdW:
            vdW = vdW_approx

    # Approximate radiative damping if missing
    # Note: Julia treats both 0 and 1 as flags to approximate (missing/placeholder values)
    if gamma_rad is None or np.isnan(gamma_rad) or gamma_rad == 0.0 or gamma_rad == 1.0:
        gamma_rad = approximate_radiative_gamma(wl, log_gf)

    # Process vdW parameter into (γ or σ, -1 or α) tuple
    if not isinstance(vdW, tuple):
        if vdW < 0:
            # Negative: it's log(γ_vdW)
            vdW = (10**vdW, -1.0)
        elif vdW == 0:
            # Exactly zero: no vdW broadening
            vdW = (0.0, -1.0)
        elif 0 < vdW < 20:
            # Fudge factor for Unsöld approximation
            _, log_gamma_vdW_base = approximate_gammas(
                wl, species, E_lower, ionization_energies_dict
            )
            vdW = (vdW * 10**log_gamma_vdW_base, -1.0)
        else:
            # Packed ABO parameters: unpack them
            # Format: σ/(a₀²) in integer part, α in fractional part
            sigma_over_a0_squared = np.floor(vdW)
            alpha = vdW - sigma_over_a0_squared
            vdW = (sigma_over_a0_squared * bohr_radius_cgs**2, alpha)

    # Convert JAX arrays to Python floats for storage
    return Line(
        wl=float(wl),
        log_gf=float(log_gf),
        species=species,
        E_lower=float(E_lower),
        gamma_rad=float(gamma_rad),
        gamma_stark=float(gamma_stark),
        vdW=(float(vdW[0]), float(vdW[1]))
    )


def air_to_vacuum(wl_air: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert air wavelength to vacuum wavelength using the Edlén (1966) formula.

    Args:
        wl_air: Wavelength in air (Ångströms)

    Returns:
        Wavelength in vacuum (Ångströms)
    """
    # Edlén (1966) formula, standard conversion
    # This is the IAU standard: https://www.iau.org/publications/proceedings_rules/units/
    sigma2 = (1e4 / wl_air) ** 2  # (μm⁻¹)²
    n = 1 + 0.00008336624212083 + 0.02408926 / (130.1065 - sigma2) + 0.0001599740 / (38.92568 - sigma2)
    return wl_air * n


def vacuum_to_air(wl_vac: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert vacuum wavelength to air wavelength.

    This uses an iterative approach to invert the air_to_vacuum formula.

    Args:
        wl_vac: Wavelength in vacuum (Ångströms)

    Returns:
        Wavelength in air (Ångströms)
    """
    # Start with vacuum wavelength as initial guess
    wl_air = wl_vac
    # Iterate to converge
    for _ in range(5):
        wl_air = wl_vac / (air_to_vacuum(wl_air) / wl_air)
    return wl_air


def read_vald_linelist(filename: str) -> list:
    """
    Read a VALD linelist file.

    Args:
        filename: Path to VALD linelist file

    Returns:
        List of Line objects

    Notes:
        This is a simplified parser that handles the standard VALD "extract stellar"
        format. It may not handle all VALD variants.
    """
    import re

    lines = []

    with open(filename, 'r') as f:
        # Skip header lines until we find the data section
        for line in f:
            if line.startswith("'"):
                # This is a data line
                # VALD format (extract stellar):
                # 'Element', lambda_air, log(gf), E_low, J_low, E_upp, J_upp, lower_lande, upper_lande, mean_lande,
                # Rad, Stark, Waals, Reference

                # Parse the line
                # Format: 'Spec Ion', WL_vac(A), Excit(eV), Vmic, log gf*, Rad., Stark, Waals, Lande, depth, Reference
                parts = line.strip().split(',')
                if len(parts) < 11:
                    continue

                try:
                    # Extract species name (in quotes)
                    species_str = parts[0].strip("' ")
                    # Parse species name (e.g., "Fe 1" -> "Fe_I", "Ca 2" -> "Ca_II")
                    species_match = re.match(r'(\w+)\s+(\d+)', species_str)
                    if not species_match:
                        continue
                    element = species_match.group(1)
                    ion_stage = int(species_match.group(2))
                    roman_numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
                    species = Species(f"{element}_{roman_numerals[ion_stage-1]}")

                    # Wavelength in vacuum (Angstroms) - already in vacuum!
                    wl_vac = float(parts[1].strip())

                    # Lower level energy (eV)
                    E_lower = float(parts[2].strip())

                    # Vmic is in parts[3], skip it

                    # log(gf)
                    log_gf = float(parts[4].strip())

                    # Broadening parameters
                    # Radiative damping (log scale in VALD)
                    rad_str = parts[5].strip()
                    gamma_rad = 10**float(rad_str) if rad_str else None

                    # Stark damping (log scale in VALD)
                    stark_str = parts[6].strip()
                    gamma_stark = 10**float(stark_str) if stark_str else None

                    # van der Waals damping (log scale in VALD or direct value)
                    vdw_str = parts[7].strip()
                    vdW = float(vdw_str) if vdw_str else None

                    # Create line
                    line_obj = create_line(wl_vac, log_gf, species, E_lower,
                                          gamma_rad, gamma_stark, vdW)
                    lines.append(line_obj)

                except (ValueError, IndexError) as e:
                    # Skip malformed lines
                    continue

    return lines


def get_VALD_solar_linelist() -> list:
    """
    Get a VALD "extract stellar" linelist produced at solar parameters.

    This linelist was downloaded with the "threshold" value set to 0.01.
    It is intended to be used for quick tests only.

    Returns:
        List of Line objects

    Notes:
        If you use this in a paper, please cite VALD appropriately:
        https://www.astro.uu.se/valdwiki/Acknowledgement
    """
    import os
    from .data_loader import _DATA_DIR

    filename = os.path.join(_DATA_DIR, "linelists",
                           "vald_extract_stellar_solar_threshold001.vald")
    return read_vald_linelist(filename)


def get_GALAH_DR3_linelist() -> list:
    """
    Get the GALAH DR3 linelist.

    The GALAH DR 3 linelist (also used for DR 4) ranges from roughly
    4,675 Å to 7,930 Å. This linelist is based on, but distinct from
    Heiter 2021 (https://ui.adsabs.harvard.edu/abs/2021A%26A...645A.106H/).

    Returns:
        List of Line objects

    References:
        Buder et al. 2021: https://ui.adsabs.harvard.edu/abs/2021MNRAS.506..150B

    Notes:
        Hydrogen lines are filtered out from this linelist.
    """
    import os
    import h5py
    from .data_loader import _DATA_DIR

    filename = os.path.join(_DATA_DIR, "linelists", "GALAH_DR3",
                           "galah_dr3_linelist.h5")

    lines = []

    with h5py.File(filename, 'r') as f:
        # Read data arrays
        wls = f['wl'][:]  # Wavelengths in Angstroms
        log_gfs = f['log_gf'][:]
        E_los = f['E_lo'][:]  # Lower energy levels in eV

        # Read species data
        # Formula is stored as array of atomic numbers (up to 3 atoms)
        formulas = f['formula'][:]  # Shape: (n_lines, 3)
        ionizations = f['ionization'][:]  # Ionization stage (1 for neutral, 2 for singly ionized, etc.)

        # Read broadening parameters
        gamma_rads = f['gamma_rad'][:]  # log10 values or special markers
        gamma_starks = f['gamma_stark'][:]
        vdWs = f['vdW'][:]

        # Helper to convert special GALAH values
        def convert_or_none(val):
            """Convert GALAH special values to None or actual value."""
            if np.isnan(val) or val == -999 or val == 0:
                return None
            # GALAH stores log10 values, convert to linear
            return 10**val

        def vdw_or_none(val):
            """Convert GALAH vdW values."""
            if np.isnan(val) or val == -999:
                return None
            return val

        # Parse each line
        for i in range(len(wls)):
            # Parse species from formula and ionization
            atoms = formulas[i]
            ion = ionizations[i]

            # Get non-zero atoms
            non_zero = [int(a) for a in atoms if a != 0]

            if len(non_zero) == 0:
                continue

            # Convert to Species
            # For atoms: use atomic number directly
            # For molecules: would need to handle multiple atoms
            if len(non_zero) == 1:
                # Atomic species
                from .atomic_data import atomic_symbols
                # Offset index by -1 for 0-based indexing
                element = atomic_symbols[non_zero[0] - 1]
                roman_numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
                species = Species(f"{element}_{roman_numerals[ion-1]}")
            else:
                # Molecular species - would need more complex handling
                # For now, skip molecules
                continue

            # Filter out hydrogen lines
            if species.charge == 0 and non_zero[0] == 1:  # H I
                continue

            # Create line
            wl = wls[i]
            log_gf = log_gfs[i]
            E_lower = E_los[i]
            gamma_rad = convert_or_none(gamma_rads[i])
            gamma_stark = convert_or_none(gamma_starks[i])
            vdW = vdw_or_none(vdWs[i])

            line = create_line(wl, log_gf, species, E_lower,
                             gamma_rad, gamma_stark, vdW)
            lines.append(line)

    return lines
