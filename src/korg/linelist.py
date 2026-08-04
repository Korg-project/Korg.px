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
# NIST isotopic abundances, atomic number -> {mass number -> abundance}. This
# module used to carry a hand-transcribed copy that stopped at Z = 30, which
# silently disabled isotopic log gf scaling for everything heavier — Ba, the
# element Kurucz linelists split into the most HFS/isotope components, being the
# obvious casualty. korg.isotopic_data is the machine-generated transcription of
# Korg.jl's isotopic_data.jl, so there is one table and it is that one.
from .isotopic_data import isotopic_abundances  # noqa: F401


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


def _parse_or_zero(dtype, s: str):
    """
    Parse ``s``, treating a blank or whitespace-only field as zero.

    Kurucz linelists come out of Fortran ``FORMAT`` statements, where an all-blank
    numeric field reads back as zero rather than raising. Korg.jl's ``parse_or_zero``
    (linelist.jl) does the same.
    """
    s = s.strip()
    return dtype(s) if s else dtype(0)


def _first_nonempty_line(path: str):
    """Return the first line of ``path`` that is not entirely whitespace, or ''."""
    with open(path, 'r') as fp:
        for raw in fp:
            if raw.strip():
                return raw
    return ''


# Kurucz "gfall" records are fixed-width, 160 characters wide, and the fields are
# addressed by column rather than by whitespace splitting (many of them run into
# their neighbours). These are Korg.jl's slices from parse_kurucz_linelist in
# linelist.jl, translated from Julia's inclusive 1-based ranges to Python's
# half-open 0-based ones.
_KURUCZ_COLUMNS = {
    'wl': slice(0, 11),           # Julia  1:11   wavelength, nm
    'log_gf': slice(11, 18),      # Julia 12:18
    'species': slice(18, 24),     # Julia 19:24   MOOG-style code, e.g. " 26.00"
    'E_level_1': slice(24, 36),   # Julia 25:36   cm^-1
    'E_level_2': slice(52, 64),   # Julia 53:64   cm^-1
    'log_gamma_rad': slice(80, 86),    # Julia 81:86
    'log_gamma_stark': slice(86, 92),  # Julia 87:92
    'vdW': slice(92, 98),         # Julia 93:98
    'isotope_1': slice(106, 109),      # Julia 107:109
    'hyperfine_log_gf': slice(109, 115),  # Julia 110:115
    'isotope_2': slice(115, 118),      # Julia 116:118
    'isotope_log_gf': slice(118, 124),  # Julia 119:124
}


def parse_kurucz_linelist(f, isotopic_abundances=None, vacuum: bool = False,
                          verbose: bool = False) -> list:
    """
    Parse an *atomic* Kurucz-format linelist (http://kurucz.harvard.edu/linelists.html).

    Args:
        f: File path or file-like object.
        isotopic_abundances: Isotopic abundances dict {Z: {mass number: abundance}}.
            ``None`` (the default, as in Korg.jl) means "trust the log gf isotope
            adjustment Kurucz already wrote into the file" — see below.
        vacuum: If True the wavelengths are already vacuum. If False (the default)
            they are air and get converted.
        verbose: Print a note whenever a line's isotope is absent from
            ``isotopic_abundances`` and Kurucz's own adjustment is used instead.

    Returns:
        List of Line objects, in file order (read_linelist sorts).

    Notes:
        Kurucz encodes the isotopic scaling of log gf twice: once as his own
        additive correction (columns 119-124) and once as the bare isotope number
        (columns 107-109 or 116-118). Passing ``isotopic_abundances=None`` takes
        the former verbatim; passing a table recomputes the correction from it,
        which is more precise because Kurucz's column carries few digits. Isotopes
        missing from the table fall back on Kurucz's number.
    """
    if isinstance(f, str):
        with open(f, 'r') as fp:
            content_lines = fp.readlines()
    else:
        content_lines = f.readlines()

    result = []
    for raw in content_lines:
        row = raw.rstrip('\n').rstrip('\r')
        if not row.strip():
            continue

        # Some distributions of gfall drop one leading column of the wavelength
        # field, shifting every subsequent field left by one. Restore it.
        if len(row) == 159:
            row = ' ' + row
        # Others have had the trailing (non-numeric) columns stripped by an
        # editor. The Fortran FORMAT guarantees 160 characters, so pad back out
        # and let _parse_or_zero read the now-blank fields as zero.
        if len(row) < 160:
            row = row.ljust(160)

        # Kurucz gives the wavenumbers of "level 1" and "level 2" without saying
        # which is the lower one — that is set by parity — so take the smaller.
        # The values are negative when Kurucz predicted rather than measured them.
        E_levels = [abs(float(row[_KURUCZ_COLUMNS[k]])) * c_cgs * hplanck_eV
                    for k in ('E_level_1', 'E_level_2')]

        species = Species(row[_KURUCZ_COLUMNS['species']])

        # log gf, plus the hyperfine-structure splitting of this component
        log_gf = (float(row[_KURUCZ_COLUMNS['log_gf']])
                  + _parse_or_zero(float, row[_KURUCZ_COLUMNS['hyperfine_log_gf']]))

        kurucz_iso_adjust = _parse_or_zero(float, row[_KURUCZ_COLUMNS['isotope_log_gf']])
        if isotopic_abundances is None:
            log_gf += kurucz_iso_adjust
        else:
            # The isotope number lives in one of two columns depending on which
            # of the two splitting mechanisms produced the line.
            iso_number = _parse_or_zero(int, row[_KURUCZ_COLUMNS['isotope_1']])
            if iso_number == 0:
                iso_number = _parse_or_zero(int, row[_KURUCZ_COLUMNS['isotope_2']])
            if iso_number != 0:  # no isotope number means no adjustment at all
                Z = species.get_atom()
                if iso_number not in isotopic_abundances[Z]:
                    if verbose:
                        print(f"Isotope {iso_number} not in isoabunds for {species}. "
                              f"Using Kurucz's value of {kurucz_iso_adjust}.")
                else:
                    log_gf += math.log10(isotopic_abundances[Z][iso_number])

        wl_cm = float(row[_KURUCZ_COLUMNS['wl']]) * 1e-7  # nm -> cm
        if not vacuum:
            wl_cm = air_to_vacuum(wl_cm)

        # Columns 81-98 are log10(γ_rad), log10(γ_Stark) and the vdW parameter,
        # with an exact zero standing for "no data" in all three (Korg.jl's
        # tentotheOrMissing/idOrMissing). create_line then fills the gaps.
        log_gamma_rad = _parse_or_zero(float, row[_KURUCZ_COLUMNS['log_gamma_rad']])
        log_gamma_stark = _parse_or_zero(float, row[_KURUCZ_COLUMNS['log_gamma_stark']])
        vdW = _parse_or_zero(float, row[_KURUCZ_COLUMNS['vdW']])

        result.append(create_line(
            wl_cm, log_gf, species, min(E_levels),
            gamma_rad=None if log_gamma_rad == 0 else 10.0 ** log_gamma_rad,
            gamma_stark=None if log_gamma_stark == 0 else 10.0 ** log_gamma_stark,
            vdW=None if vdW == 0 else vdW,
        ))

    return result


def parse_kurucz_molecular_linelist(f, isotopic_abundances=isotopic_abundances,
                                    vacuum: bool = False) -> list:
    """
    Parse a *molecular* Kurucz-format linelist.

    Not implemented, exactly as in Korg.jl v1.2.1: its parse_kurucz_molecular_linelist
    throws before reaching the (still present but dead) parsing code, because the
    energy levels it reads are not reliably the ones Korg needs.
    """
    raise ValueError("Kurucz linelists are not yet supported for molecules. Please open an "
                     "issue at https://github.com/ajwheeler/Korg.jl/issues if this is a "
                     "problem for you.")


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


def parse_moog_linelist(f, isotopic_abundances=isotopic_abundances,
                        vacuum_wavelengths: bool = True) -> list:
    """
    Parse a MOOG-format linelist.

    Column order: wavelength(Å)  species_code  excitation_potential(eV)  log_gf

    Args:
        f: File path or file-like object
        isotopic_abundances: Isotopic abundances dict {Z: {mass: abundance}}.
            Defaults to the NIST table, as in Korg.jl.
        vacuum_wavelengths: If True, wavelengths are vacuum. If False, convert air->vacuum.

    Returns:
        List of Line objects sorted by wavelength
    """
    import math

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
                            if Z in isotopic_abundances and m_num in isotopic_abundances[Z]:
                                delta_loggf += math.log10(isotopic_abundances[Z][m_num])
                except (ValueError, AttributeError):
                    pass

            E_lower = float(toks[2])
            log_gf = float(toks[3]) + delta_loggf

            line_obj = create_line(wl_angstrom, log_gf, spec, E_lower)
            result.append(line_obj)
        except (ValueError, IndexError, KeyError):
            continue

    return sorted(result, key=lambda l: l.wl)


def parse_turbospectrum_linelist(fn: str, isotopic_abundances=isotopic_abundances,
                                 vacuum: bool = False) -> list:
    """
    Parse a TurboSpectrum-format linelist.

    Args:
        fn: File path
        isotopic_abundances: Isotopic abundances dict. Defaults to the NIST table,
            as in Korg.jl.
        vacuum: If True, wavelengths are already in vacuum. If False (default), convert.

    Returns:
        List of Line objects sorted by wavelength
    """
    import math, re

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
                if atom_Z in isotopic_abundances and m_num in isotopic_abundances[atom_Z]:
                    delta_loggf += math.log10(isotopic_abundances[atom_Z][m_num])

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

                # Column 7 holds log10(γ_Stark).  Korg.jl feeds it through
                # ``tentotheOrMissing`` (linelist.jl,
                # parse_turbospectrum_linelist_transition), i.e. exactly 0 means
                # "no data" and every other value is a base-10 logarithm.  If the
                # column is not numeric it is the orbital angular momentum letter
                # for the upper level, which also means "no data".
                gamma_stark_val = None
                if len(toks) > 6:
                    try:
                        gs = float(toks[6])
                    except ValueError:
                        pass
                    else:
                        gamma_stark_val = None if gs == 0.0 else 10.0 ** gs

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

        # Korg.jl writes ``reduce(hcat, [l.species.formula.atoms ...])``: a
        # (MAX_ATOMS_PER_MOLECULE, n_lines) Julia matrix, which HDF5 stores — and
        # h5py therefore sees — with shape (n_lines, MAX_ATOMS_PER_MOLECULE).
        # Formula.atoms is zero-padded at the *front*, so each row is
        # right-aligned. Matching this layout exactly is what makes the file
        # readable by Korg.jl's read_korg_linelist (and vice versa).
        from .species import MAX_ATOMS_PER_MOLECULE
        formula_arr = np.zeros((len(linelist), MAX_ATOMS_PER_MOLECULE), dtype=np.uint8)
        for i, l in enumerate(linelist):
            formula_arr[i, :] = [int(a) for a in l.species.formula.atoms]
        f.create_dataset('formula', data=formula_arr)
        f['formula'].attrs['description'] = 'Array of atomic numbers representing molecular formula'

        f.create_dataset('species',
                         data=np.array([str(l.species) for l in linelist], dtype=h5py.special_dtype(vlen=str)))
        f['species'].attrs['description'] = 'String representation of atomic/molecular species'

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
        # Row i holds the (front-zero-padded) atomic numbers of line i; see
        # save_linelist for why this matches Korg.jl's on-disk layout.
        atoms = [int(a) for a in formula_arr[i] if a != 0]
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
                  isotopic_abundances=isotopic_abundances) -> list:
    """
    Read a linelist file in various formats.

    Args:
        filename: Path to linelist file
        format: One of "vald", "kurucz", "kurucz_vac", "moog", "moog_air",
                "turbospectrum", "turbospectrum_vac", "korg". Defaults to "korg"
                if filename ends in .h5, else "vald".
        isotopic_abundances: Isotopic abundances dict {Z: {mass: abundance}} used
            to scale log gf, for the formats that carry isotope information.
            Defaults to the NIST table. Pass ``None`` to use the isotopic
            adjustments Kurucz embedded in a Kurucz-format linelist instead.

    Returns:
        List of Line objects sorted by wavelength

    Notes:
        "kurucz" is for air wavelengths, "kurucz_vac" for vacuum. Korg does not
        guess: Kurucz publishes vacuum wavelengths below 2000 Å and air above it,
        and it is the caller's job to say which file this is.
    """
    if format is None:
        format = 'korg' if filename.endswith('.h5') else 'vald'

    if format == 'korg':
        return read_korg_linelist(filename)
    elif format == 'vald':
        return read_vald_linelist(filename)
    elif format in ('kurucz', 'kurucz_vac'):
        # Atomic and molecular Kurucz records share no columns, and the only
        # thing telling them apart is the record width: molecular records are
        # ~74 characters, atomic ones 159-160. Korg.jl thresholds at 100.
        if len(_first_nonempty_line(filename)) > 100:
            lines = parse_kurucz_linelist(filename, isotopic_abundances,
                                          vacuum=format.endswith('_vac'))
        else:
            lines = parse_kurucz_molecular_linelist(filename, isotopic_abundances,
                                                    vacuum=format.endswith('_vac'))
        # Korg.jl applies this filter in read_linelist to every format. Korg.px's
        # other parsers predate it and their reference data was generated without
        # it, so it lives here rather than in the parser: gfall covers the whole
        # periodic table in every ionization stage, and Korg models neither
        # triply-ionized species nor H lines through the linelist machinery
        # (hydrogen gets its own Stark-broadened treatment).
        H_I = Species('H I')
        return sorted((l for l in lines
                       if 0 <= l.species.charge <= 2 and l.species != H_I),
                      key=lambda l: l.wl)
    elif format == 'moog':
        return parse_moog_linelist(filename, isotopic_abundances, vacuum_wavelengths=True)
    elif format == 'moog_air':
        return parse_moog_linelist(filename, isotopic_abundances, vacuum_wavelengths=False)
    elif format == 'turbospectrum':
        return parse_turbospectrum_linelist(filename, isotopic_abundances, vacuum=False)
    elif format == 'turbospectrum_vac':
        return parse_turbospectrum_linelist(filename, isotopic_abundances, vacuum=True)
    else:
        raise ValueError(f"Unknown linelist format: {format!r}. "
                         "Use one of: vald, kurucz, kurucz_vac, moog, moog_air, "
                         "turbospectrum, turbospectrum_vac, korg")


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
