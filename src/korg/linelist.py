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
# Single implementation of the Birch & Downs (1994) air/vacuum conversion, as
# in Korg.jl's utils.jl. Re-exported here (and from ``korg``) so that
# ``korg.linelist.air_to_vacuum`` keeps working.
from .utils import air_to_vacuum, vacuum_to_air  # noqa: F401


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

    # Normalize list → tuple for vdW (JSON deserializes tuples as lists)
    if isinstance(vdW, list):
        vdW = tuple(vdW)

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
        if vdW == 0.0:
            # Explicit zero: no vdW broadening
            vdW = (0.0, -1.0)
        elif vdW < 0:
            # Negative: it's log(γ_vdW)
            vdW = (10**vdW, -1.0)
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
                    # 0.0 in VALD means "no data" — treat as None so create_line uses Unsöld
                    vdw_str = parts[7].strip()
                    _vdw_raw = float(vdw_str) if vdw_str else None
                    vdW = None if (_vdw_raw == 0.0) else _vdw_raw

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
            # For molecules: construct from multiple atoms
            from .atomic_data import atomic_symbols

            if len(non_zero) == 1:
                # Atomic species
                # Offset index by -1 for 0-based indexing
                element = atomic_symbols[non_zero[0] - 1]
                roman_numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
                species = Species(f"{element}_{roman_numerals[ion-1]}")
            else:
                # Molecular species - construct formula from atomic numbers
                # Julia sorts the atoms: Formula(sort(atoms[1:2])) or Formula(sort(atoms))
                # Build formula string like "OH", "CN", "TiO", etc.
                formula_parts = []
                for atom_num in sorted(non_zero):
                    element = atomic_symbols[atom_num - 1]
                    formula_parts.append(element)
                formula_str = ''.join(formula_parts)
                # Molecules have charge from ionization stage
                species = Species(formula_str, charge=ion-1)

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


def approximate_line_strength(line: Line, T: float) -> float:
    """
    Approximate the line strength (log10(gfλ) - θχ) of a line at temperature T.

    Used to quickly filter large linelists (especially molecular lines from ExoMol).

    Args:
        line: Line object
        T: Temperature in K

    Returns:
        Approximate log-line strength in arbitrary units
    """
    import math
    return line.log_gf + math.log10(line.wl) - math.log10(math.e) * line.E_lower / (kboltz_eV * T)


# NIST isotopic abundances: maps atomic number -> {mass_number -> abundance}
isotopic_abundances = {
    1: {1: 1.0, 2: 1e-10},
    2: {3: 1.34e-6, 4: 0.99999866},
    3: {6: 0.0759, 7: 0.9241},
    4: {9: 1.0},
    5: {10: 0.199, 11: 0.801},
    6: {12: 0.9893, 13: 0.0107},
    7: {14: 0.99636, 15: 0.00364},
    8: {16: 0.99757, 17: 0.00038, 18: 0.00205},
    9: {19: 1.0},
    10: {20: 0.9048, 21: 0.0027, 22: 0.0925},
    11: {23: 1.0},
    12: {24: 0.7899, 25: 0.1, 26: 0.1101},
    13: {27: 1.0},
    14: {28: 0.92223, 29: 0.04685, 30: 0.03092},
    15: {31: 1.0},
    16: {32: 0.9499, 33: 0.0075, 34: 0.0425, 36: 0.0001},
    17: {35: 0.7576, 37: 0.2424},
    18: {36: 0.003336, 38: 0.000629, 40: 0.996035},
    19: {39: 0.932581, 40: 0.000117, 41: 0.067302},
    20: {40: 0.96941, 42: 0.00647, 43: 0.00135, 44: 0.02086, 46: 4.0e-5, 48: 0.00187},
    21: {45: 1.0},
    22: {46: 0.0825, 47: 0.0744, 48: 0.7372, 49: 0.0541, 50: 0.0518},
    23: {50: 0.0025, 51: 0.9975},
    24: {50: 0.04345, 52: 0.83789, 53: 0.09501, 54: 0.02365},
    25: {55: 1.0},
    26: {54: 0.05845, 56: 0.91754, 57: 0.02119, 58: 0.00282},
    27: {59: 1.0},
    28: {58: 0.68077, 60: 0.26223, 61: 0.011399, 62: 0.036346, 64: 0.009255},
    29: {63: 0.6915, 65: 0.3085},
    30: {64: 0.4917, 66: 0.2773, 67: 0.0404, 68: 0.1845, 70: 0.0061},
}


def _moog_species_code_to_species(code_str: str):
    """
    Convert a MOOG species code string to a Species object.

    MOOG codes: integer part = atomic number (or concatenated Z for molecules),
    first decimal digit = charge. E.g. "26.0" = Fe I, "26.1" = Fe II.
    """
    from .atomic_data import atomic_symbols
    dot_idx = code_str.index('.')
    charge = int(code_str[dot_idx + 1])
    Z = int(code_str[:dot_idx])

    if Z <= 99:
        element = atomic_symbols[Z - 1]
        return Species(element, charge=charge)
    else:
        # Molecular: parse pairs of 2-digit atomic numbers from right of Z string
        z_str = code_str[:dot_idx]
        atoms = []
        s = z_str
        while len(s) >= 2:
            z = int(s[-2:])
            if z > 0:
                atoms.append(z)
            s = s[:-2]
        if s:
            z = int(s)
            if z > 0:
                atoms.append(z)
        atoms.sort()
        from .species import Formula
        formula = Formula(atoms)
        return Species(formula, charge=charge)


def parse_moog_linelist(f, iso_abundances=None, vacuum_wavelengths: bool = True) -> list:
    """
    Parse a MOOG-format linelist.

    Column order: wavelength(Å)  species_code  excitation_potential(eV)  log_gf

    Args:
        f: File path or file-like object
        iso_abundances: Isotopic abundances dict {Z: {mass: abundance}}.
            Defaults to NIST values from `isotopic_abundances`.
        vacuum_wavelengths: If True, wavelengths are vacuum. If False, convert air->vacuum.

    Returns:
        List of Line objects sorted by wavelength
    """
    import math
    if iso_abundances is None:
        iso_abundances = isotopic_abundances

    if isinstance(f, str):
        with open(f, 'r') as fp:
            content_lines = fp.readlines()
    else:
        content_lines = f.readlines()

    result = []
    for raw in content_lines[1:]:  # skip header line
        raw = raw.strip()
        if not raw or raw.startswith('#'):
            continue
        toks = raw.split()
        if len(toks) < 4:
            continue
        try:
            wl_angstrom = float(toks[0])
            if not vacuum_wavelengths:
                wl_angstrom = air_to_vacuum(wl_angstrom)

            code_str = toks[1]
            dot_idx = code_str.index('.')
            spec = _moog_species_code_to_species(code_str)

            # Isotope correction from digits after first decimal digit
            iso_str = code_str[dot_idx + 2:]
            delta_loggf = 0.0
            if iso_str and not all(c == '0' for c in iso_str):
                try:
                    atoms = [int(a) for a in spec.formula.atoms if a != 0]
                    natoms = len(atoms)
                    if natoms > 0 and len(iso_str) % natoms == 0:
                        digits_per = len(iso_str) // natoms
                        for j, Z in enumerate(atoms):
                            iso_start = j * digits_per
                            m_num = int(iso_str[iso_start:iso_start + digits_per])
                            if Z in iso_abundances and m_num in iso_abundances[Z]:
                                delta_loggf += math.log10(iso_abundances[Z][m_num])
                except (ValueError, AttributeError):
                    pass

            E_lower = float(toks[2])
            log_gf = float(toks[3]) + delta_loggf

            line_obj = create_line(wl_angstrom, log_gf, spec, E_lower)
            result.append(line_obj)
        except (ValueError, IndexError, KeyError):
            continue

    return sorted(result, key=lambda l: l.wl)


def parse_turbospectrum_linelist(fn: str, iso_abundances=None,
                                  vacuum: bool = False) -> list:
    """
    Parse a TurboSpectrum-format linelist.

    Args:
        fn: File path
        iso_abundances: Isotopic abundances dict. Defaults to NIST values.
        vacuum: If True, wavelengths are already in vacuum. If False (default), convert.

    Returns:
        List of Line objects sorted by wavelength
    """
    import math, re
    if iso_abundances is None:
        iso_abundances = isotopic_abundances

    with open(fn, 'r') as fp:
        content_lines = fp.readlines()

    # Find species header lines (pairs of lines starting with "'")
    species_headers = []
    for i in range(len(content_lines) - 1):
        if content_lines[i].startswith("'") and content_lines[i + 1].startswith("'"):
            species_headers.append(i)

    all_lines = []
    for h_idx, header_line_idx in enumerate(species_headers):
        first_line_idx = header_line_idx
        last_line_idx = (species_headers[h_idx + 1] - 1
                         if h_idx < len(species_headers) - 1
                         else len(content_lines) - 1)

        species_line = content_lines[first_line_idx]
        m = re.match(r"'\s*(?P<formula>\d+)\.(?P<isostring>\d+)\s+'\s+(?P<ion>\d+)\s+(?P<n_lines>\d+)",
                     species_line)
        if m is None:
            continue

        from .atomic_data import atomic_symbols
        Z = int(m.group('formula'))
        charge = int(m.group('ion')) - 1
        isostring = m.group('isostring')

        if Z <= 99:
            spec = Species(atomic_symbols[Z - 1], charge=charge)
        else:
            spec = _moog_species_code_to_species(f"{Z}.{charge}{isostring}")

        # Isotopic correction
        atoms = [int(a) for a in spec.formula.atoms if a != 0]
        delta_loggf = 0.0
        if isostring and len(isostring) >= 3 * len(atoms):
            for j, atom_Z in enumerate(atoms):
                m_start = j * 3
                m_num = int(isostring[m_start:m_start + 3])
                if m_num == 0:
                    continue
                if atom_Z in iso_abundances and m_num in iso_abundances[atom_Z]:
                    delta_loggf += math.log10(iso_abundances[atom_Z][m_num])

        for raw in content_lines[first_line_idx + 2:last_line_idx + 1]:
            raw = raw.strip()
            if not raw or raw.startswith("'"):
                break
            toks = raw.split()
            if len(toks) < 6:
                continue
            try:
                wl_angstrom = float(toks[0])
                wl_vac = wl_angstrom if vacuum else air_to_vacuum(wl_angstrom)
                E_lower = float(toks[1])
                log_gf = float(toks[2]) + delta_loggf
                vdW_val = float(toks[3])
                gamma_rad_val = float(toks[5])
                if gamma_rad_val in (0.0, 1.0):
                    gamma_rad_val = None

                gamma_stark_val = None
                if len(toks) > 6:
                    try:
                        gs = float(toks[6])
                        gamma_stark_val = gs if gs not in (0.0, 1.0) else None
                    except ValueError:
                        pass

                line_obj = create_line(
                    wl_vac, log_gf, spec, E_lower,
                    gamma_rad=gamma_rad_val,
                    gamma_stark=gamma_stark_val,
                    vdW=vdW_val
                )
                all_lines.append(line_obj)
            except (ValueError, IndexError):
                continue

    return sorted(all_lines, key=lambda l: l.wl)


def save_linelist(path: str, linelist: list) -> None:
    """
    Save a linelist to an HDF5 file readable by read_korg_linelist.

    Args:
        path: Output file path (should end in .h5)
        linelist: List of Line objects
    """
    import h5py

    with h5py.File(path, 'w') as f:
        f.attrs['version'] = '2024-12-18'

        f.create_dataset('wl', data=np.array([l.wl for l in linelist]))
        f['wl'].attrs['description'] = 'Wavelength in cm'

        f.create_dataset('log_gf', data=np.array([l.log_gf for l in linelist]))
        f['log_gf'].attrs['description'] = 'Log of oscillator strength times statistical weight'

        max_atoms = max((len([a for a in l.species.formula.atoms if a != 0])
                         for l in linelist), default=1)
        formula_arr = np.zeros((max_atoms, len(linelist)), dtype=np.uint8)
        for i, l in enumerate(linelist):
            atoms = [int(a) for a in l.species.formula.atoms if a != 0]
            for j, a in enumerate(atoms[:max_atoms]):
                formula_arr[j, i] = a
        f.create_dataset('formula', data=formula_arr)
        f['formula'].attrs['description'] = 'Array of atomic numbers (rows) per line (cols)'

        f.create_dataset('charge',
                         data=np.array([l.species.charge for l in linelist], dtype=np.int32))
        f['charge'].attrs['description'] = 'Ionization state (0=neutral, 1=singly ionized, etc)'

        f.create_dataset('E_lower', data=np.array([l.E_lower for l in linelist]))
        f['E_lower'].attrs['description'] = 'Lower energy level in eV'

        f.create_dataset('gamma_rad', data=np.array([l.gamma_rad for l in linelist]))
        f['gamma_rad'].attrs['description'] = 'Radiative damping parameter in rad/s'

        f.create_dataset('gamma_stark', data=np.array([l.gamma_stark for l in linelist]))
        f['gamma_stark'].attrs['description'] = 'Stark broadening parameter'

        f.create_dataset('vdW_1', data=np.array([l.vdW[0] for l in linelist]))
        f['vdW_1'].attrs['description'] = 'First van der Waals broadening parameter'

        f.create_dataset('vdW_2', data=np.array([l.vdW[1] for l in linelist]))
        f['vdW_2'].attrs['description'] = 'Second van der Waals broadening parameter'


def read_korg_linelist(path: str) -> list:
    """
    Read a Korg-format HDF5 linelist saved by save_linelist.

    Args:
        path: Path to HDF5 linelist file

    Returns:
        List of Line objects
    """
    import h5py
    from .species import Formula

    with h5py.File(path, 'r') as f:
        formula_arr = f['formula'][:]
        charges = f['charge'][:]
        wls = f['wl'][:]
        log_gfs = f['log_gf'][:]
        E_lowers = f['E_lower'][:]
        gamma_rads = f['gamma_rad'][:]
        gamma_starks = f['gamma_stark'][:]
        vdW_1s = f['vdW_1'][:]
        vdW_2s = f['vdW_2'][:]

    result = []
    for i in range(len(wls)):
        atoms = [int(a) for a in formula_arr[:, i] if a != 0]
        if not atoms:
            continue
        formula = Formula(atoms)
        spec = Species(formula, charge=int(charges[i]))
        line = Line(
            wl=float(wls[i]),
            log_gf=float(log_gfs[i]),
            species=spec,
            E_lower=float(E_lowers[i]),
            gamma_rad=float(gamma_rads[i]),
            gamma_stark=float(gamma_starks[i]),
            vdW=(float(vdW_1s[i]), float(vdW_2s[i]))
        )
        result.append(line)

    return result


def read_linelist(filename: str, format: str = None,
                  iso_abundances=None) -> list:
    """
    Read a linelist file in various formats.

    Args:
        filename: Path to linelist file
        format: One of "vald", "moog", "moog_air", "turbospectrum",
                "turbospectrum_vac", "korg". Defaults to "korg" if filename
                ends in .h5, else "vald".
        iso_abundances: Isotopic abundances dict for MOOG/TurboSpectrum formats.

    Returns:
        List of Line objects sorted by wavelength
    """
    if format is None:
        format = 'korg' if filename.endswith('.h5') else 'vald'

    if format == 'korg':
        return read_korg_linelist(filename)
    elif format == 'vald':
        return read_vald_linelist(filename)
    elif format == 'moog':
        return parse_moog_linelist(filename, iso_abundances, vacuum_wavelengths=True)
    elif format == 'moog_air':
        return parse_moog_linelist(filename, iso_abundances, vacuum_wavelengths=False)
    elif format == 'turbospectrum':
        return parse_turbospectrum_linelist(filename, iso_abundances, vacuum=False)
    elif format == 'turbospectrum_vac':
        return parse_turbospectrum_linelist(filename, iso_abundances, vacuum=True)
    else:
        raise ValueError(f"Unknown linelist format: {format!r}. "
                         "Use one of: vald, moog, moog_air, turbospectrum, "
                         "turbospectrum_vac, korg")


def get_APOGEE_DR17_linelist(include_water: bool = True) -> list:
    """
    Get the APOGEE DR17 linelist (15,000-17,000 Å).

    Args:
        include_water: Whether to include POKAZATEL water lines. Default True.

    Returns:
        List of Line objects sorted by wavelength
    """
    import os, h5py

    from .data_loader import _DATA_DIR
    py_dir = os.path.join(_DATA_DIR, 'linelists', 'APOGEE_DR17')
    julia_dir = os.path.expanduser(
        '~/.julia/packages/Korg/Rt7Dk/data/linelists/APOGEE_DR17'
    )
    data_dir = py_dir if os.path.isdir(py_dir) else julia_dir

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"APOGEE DR17 linelist data not found. Expected at: {py_dir}"
        )

    atoms = parse_turbospectrum_linelist(
        os.path.join(data_dir, 'turbospec.20180901t20.atoms_no_ba'),
        vacuum=False
    )
    mols = parse_turbospectrum_linelist(
        os.path.join(data_dir, 'turbospec.20180901t20.molec'),
        vacuum=False
    )
    all_lines = atoms + mols

    if include_water:
        water_file = os.path.join(data_dir, 'pokazatel_water_lines.h5')
        if os.path.exists(water_file):
            with h5py.File(water_file, 'r') as f:
                w_wls = f['wl'][:]
                w_loggfs = f['log_gf'][:]
                w_E_lowers = f['E_lower'][:]
                w_gamma_rads = f['gamma_rad'][:]
            water_spec = Species('H2O')
            for i in range(len(w_wls)):
                all_lines.append(Line(
                    wl=float(w_wls[i]),
                    log_gf=float(w_loggfs[i]),
                    species=water_spec,
                    E_lower=float(w_E_lowers[i]),
                    gamma_rad=float(w_gamma_rads[i]),
                    gamma_stark=0.0,
                    vdW=(0.0, -1.0)
                ))

    return sorted(all_lines, key=lambda l: l.wl)


def get_GES_linelist(include_molecules: bool = True) -> list:
    """
    Get the Gaia-ESO survey linelist from Heiter et al. 2021.

    Contains > 15 million lines. Requires Korg.jl to be installed in Julia
    (to download the artifact on first use).

    Args:
        include_molecules: Whether to include molecular lines. Default True.

    Returns:
        List of Line objects sorted by wavelength
    """
    import os, h5py, glob

    artifacts_pattern = os.path.expanduser(
        '~/.julia/artifacts/*/Heiter_et_al_2021_*/Heiter_et_al_2021.h5'
    )
    candidates = glob.glob(artifacts_pattern)
    if not candidates:
        raise FileNotFoundError(
            "GES linelist not found. Install Korg.jl in Julia and run "
            "Korg.get_GES_linelist() once to download the artifact."
        )
    path = candidates[0]

    with h5py.File(path, 'r') as f:
        species_strs = [s.decode() if isinstance(s, bytes) else s
                        for s in f['species'][:]]
        all_species = [Species(s) for s in species_strs]

        keep = np.ones(len(all_species), dtype=bool)
        if not include_molecules:
            keep = np.array([not s.formula.is_molecule() for s in all_species])

        wls_air = f['wl'][:][keep]
        log_gfs_arr = f['log_gf'][:][keep]
        E_lowers_arr = f['E_lower'][:][keep]
        gamma_rads_raw = f['gamma_rad'][:][keep]
        gamma_starks_raw = f['gamma_stark'][:][keep]
        vdWs_raw = f['vdW'][:][keep]

    species_kept = [s for s, k in zip(all_species, keep) if k]

    def _ten_or_none(val):
        return None if (np.isnan(val) or val == 0.0) else 10.0 ** val

    def _vdw_or_none(val):
        return None if np.isnan(val) else val

    wls_vac = air_to_vacuum(wls_air * 1e8)  # convert cm->Å then air->vac
    result = []
    for i in range(len(wls_vac)):
        line = create_line(
            wls_vac[i], float(log_gfs_arr[i]), species_kept[i], float(E_lowers_arr[i]),
            gamma_rad=_ten_or_none(gamma_rads_raw[i]),
            gamma_stark=_ten_or_none(gamma_starks_raw[i]),
            vdW=_vdw_or_none(vdWs_raw[i])
        )
        result.append(line)

    # Filter bad CH lines (see Korg.jl issue #356)
    ch_spec = Species('CH')
    result = [l for l in result if not (l.species == ch_spec and l.log_gf > -1.9)]

    return result


def load_ExoMol_linelist(spec, states_file: str, transitions_file: str,
                         lower_wavelength: float, upper_wavelength: float,
                         isotopes=None,
                         line_strength_cutoff: float = -15.0,
                         T_line_strength: float = 3500.0,
                         verbose: bool = True) -> list:
    """
    Load a molecular linelist from ExoMol data files.

    Reads the ExoMol states and transitions files, computes log(gf) values,
    applies isotopic corrections, and returns lines within the wavelength range.

    Args:
        spec: Species string (e.g. 'MgH') or Species object
        states_file: Path to the ExoMol .states file (space-delimited: id, E_wavenumber, g, ...)
        transitions_file: Path to the ExoMol .trans file (space-delimited: id_upper, id_lower, A, ...)
        lower_wavelength: Lower wavelength bound in Å
        upper_wavelength: Upper wavelength bound in Å
        isotopes: List of (Z, isotope_index) tuples. If None, uses the most abundant isotope.
        line_strength_cutoff: log10 strength cutoff (default -15; weaker lines are dropped)
        T_line_strength: Temperature for strength calculation (default 3500 K)
        verbose: Print progress messages (default True)

    Returns:
        List of Line objects sorted by wavelength (ascending)
    """
    from .isotopic_data import isotopic_abundances, isotopic_nuclear_spin_degeneracies

    if isinstance(spec, str):
        spec = Species(spec)

    if verbose:
        print(f"Loading ExoMol linelist from {states_file} and {transitions_file}.")

    # Read transitions file (columns: id_upper, id_lower, A)
    trans_ids_upper = []
    trans_ids_lower = []
    trans_A = []
    with open(transitions_file) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            trans_ids_upper.append(int(parts[0]))
            trans_ids_lower.append(int(parts[1]))
            trans_A.append(float(parts[2]))

    # Read states file (columns: id, E_wavenumber, g, ...)
    state_id_to_E = {}
    state_id_to_g = {}
    with open(states_file) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            sid = int(parts[0])
            state_id_to_E[sid] = float(parts[1])
            state_id_to_g[sid] = int(parts[2])

    # Compute log(gf) for each transition using Gray (4th ed), eq 11.12
    # gf = A * (electron_mass_cgs * c_cgs) / (8π² * electron_charge_cgs²) * g_upper / ν²
    prefactor = (electron_mass_cgs * c_cgs) / (8 * math.pi**2 * electron_charge_cgs**2)

    # Compute isotopic correction
    if isotopes is None:
        # Use most abundant isotope for each atom (keys are mass numbers)
        atoms = list(spec.formula.get_atoms())
        isotopes = [(int(Z), max(isotopic_abundances[int(Z)],
                                 key=lambda iso: isotopic_abundances[int(Z)][iso]))
                    for Z in atoms]
        if verbose:
            print("Assuming the most abundant isotope for all atoms.")

    try:
        iso_correction = math.log10(
            math.prod(isotopic_abundances[Z][iso] for Z, iso in isotopes)
        )
        iso_correction -= math.log10(
            math.prod(isotopic_nuclear_spin_degeneracies[Z][iso] for Z, iso in isotopes)
        )
    except (KeyError, IndexError, ValueError):
        iso_correction = 0.0

    lines = []
    lower_cm = lower_wavelength * 1e-8
    upper_cm = upper_wavelength * 1e-8

    for i_u, i_l, A in zip(trans_ids_upper, trans_ids_lower, trans_A):
        E_upper = state_id_to_E.get(i_u)
        E_lower = state_id_to_E.get(i_l)
        g_upper = state_id_to_g.get(i_u)
        g_lower = state_id_to_g.get(i_l)

        if E_upper is None or E_lower is None or g_upper is None or g_lower is None:
            continue

        wavenumber = E_upper - E_lower
        if wavenumber <= 0:
            continue

        wavelength_cm = 1.0 / wavenumber  # cm (wavenumber in cm⁻¹)
        if not (lower_cm <= wavelength_cm <= upper_cm):
            continue

        f = A * prefactor * g_upper / (g_lower * wavenumber**2)
        if f <= 0:
            continue

        log_gf = math.log10(g_lower * f) + iso_correction
        E_lower_eV = hplanck_eV * E_lower * c_cgs

        lines.append(create_line(wavelength_cm * 1e8, log_gf, spec, E_lower_eV))

    # Sort by wavelength descending (ExoMol convention: high→low energy = low→high wl)
    # then reverse to get ascending wavelength
    lines.sort(key=lambda l: l.wl)

    if not lines:
        return lines

    # Remove weak lines using approximate_line_strength
    filtered = [l for l in lines
                if approximate_line_strength(l, T_line_strength) > line_strength_cutoff]
    if verbose:
        n_removed = len(lines) - len(filtered)
        pct = 100 * n_removed // len(lines) if lines else 0
        print(f"Removed {n_removed} lines with strength below {line_strength_cutoff} "
              f"at T={T_line_strength} K out of {len(lines)} total ({pct}%).")

    return filtered
