"""
VALD linelist parser for Korg.

Parses VALD (Vienna Atomic Line Database) linelists in both short and long format,
supporting both "extract all" and "extract stellar" variants.
"""

import numpy as np
import pandas as pd
import re
from typing import List, Dict, Optional
import io

from .linelist import Line, approximate_radiative_gamma
from .species import Species
from .atomic_data import atomic_numbers, atomic_symbols
from .isotopic_data import isotopic_abundances
from .wavelengths import air_to_vacuum


def ten_to_the_or_missing(x: float) -> Optional[float]:
    """Convert log value to linear, or return None if zero."""
    return None if x == 0 else 10**x


def id_or_missing(x: float) -> Optional[float]:
    """Return value or None if zero."""
    return None if x == 0 else x


def parse_vald_linelist(file_content: str,
                         isotopic_abund: Dict = None) -> List[Line]:
    """
    Parse a VALD linelist from string content.

    Supports both short and long format, extract all and extract stellar variants.
    Handles isotopic abundance scaling and unit conversions.

    Args:
        file_content: Complete VALD linelist file content as string
        isotopic_abund: Dictionary of isotopic abundances (default: uses Korg values)

    Returns:
        List of Line objects

    Raises:
        ValueError: If linelist format cannot be determined
    """
    if isotopic_abund is None:
        isotopic_abund = isotopic_abundances

    # Split into lines and remove comments and empty lines
    lines = [line for line in file_content.split('\n')
             if len(line) > 0 and line[0] != '#']

    if len(lines) == 0:
        raise ValueError("Empty linelist")

    # Ignore truncation warning
    if lines[0].startswith(" WARNING: Output was truncated to 100000 lines"):
        lines = lines[1:]

    # Replace single quotes with double quotes
    lines = [line.replace("'", '"') for line in lines]

    # Detect format: "extract all" or "extract stellar"
    extract_all = not bool(re.match(r'^\s+\d', lines[0]))
    firstline = 2 if extract_all else 3
    header = lines[firstline - 1]  # Header is one line before data starts

    # Check if isotope scaling is needed
    scale_isotopes = any(line.startswith("* oscillator strengths were NOT scaled ")
                          for line in lines)
    if not scale_isotopes and not any(line.startswith("* oscillator strengths were scaled ")
                                       for line in lines):
        raise ValueError("Can't parse linelist. Can't detect whether log(gf)s are scaled by isotopic abundance.")

    # Detect short vs long format
    # Long format has second line after header starting with space or quote+space
    short_format = not bool(re.match(r'^"? ', lines[firstline + 1]))

    # Extract body lines (every 1st line for short format, every 4th for long)
    step = 1 if short_format else 4
    body = lines[firstline::step]

    # Find where the actual data ends (before footer)
    end_idx = None
    for i, line in enumerate(body):
        if len(line) == 0 or line[0] != '"' or (len(line) > 1 and not line[1].isupper()):
            end_idx = i
            break
    if end_idx is not None:
        body = body[:end_idx]

    # Define CSV headers based on format
    if short_format and extract_all:
        csv_header = ["species", "wl", "E_low", "loggf", "gamma_rad", "gamma_stark",
                       "gamma_vdW", "lande", "reference"]
    elif short_format:  # extract stellar
        csv_header = ["species", "wl", "E_low", "Vmic", "loggf", "gamma_rad",
                       "gamma_stark", "gamma_vdW", "lande", "depth", "reference"]
    else:  # long format (extract all or extract stellar)
        csv_header = ["species", "wl", "loggf", "E_low", "J_lo", "E_up", "J_up",
                       "lower_lande", "upper_lande", "mean_lande", "gamma_rad",
                       "gamma_stark", "gamma_vdW"]

    # Parse CSV body
    csv_data = '\n'.join(body)
    df = pd.read_csv(io.StringIO(csv_data), names=csv_header,
                      skipinitialspace=True, on_bad_lines='skip')

    # Convert E_low to eV if necessary
    if "cm" in header:
        from .constants import c_cgs, hplanck_eV
        E_low = df['E_low'].values * c_cgs * hplanck_eV
    elif "eV" in header:
        E_low = df['E_low'].values
    else:
        raise ValueError(f"Can't parse linelist. Can't determine energy units: {header}")

    # Convert wavelengths to vacuum if necessary
    if "air" in header:
        wl = 1e-8 * np.array([air_to_vacuum(w) for w in df['wl'].values])
    elif "vac" in header:
        wl = 1e-8 * df['wl'].values
    else:
        raise ValueError(f"Can't parse linelist. Can't determine vac/air wls: {header}")

    # Calculate isotopic abundance corrections
    if scale_isotopes:
        # Get references (from different location depending on format)
        if not short_format:
            # References are on separate lines (every 4th line starting from firstline+3)
            refs = lines[firstline + 3::4][:len(df)]
        else:
            # References are in the last column
            refs = df['reference'].values

        # Parse isotope information from references
        delta_log_gf = []
        for ref in refs:
            # Find things that look like (16)O or (64)Ni in reference string
            regexp = r'\((?P<isotope>\d{1,3})\)(?P<elem>[A-Z][a-z]?)'
            matches = re.finditer(regexp, str(ref))

            log_probs = []
            for match in matches:
                isotope_num = int(match.group('isotope'))
                element = match.group('elem')
                if element in atomic_numbers:
                    Z = atomic_numbers[element]
                    if Z in isotopic_abund and isotope_num in isotopic_abund[Z]:
                        log_probs.append(np.log10(isotopic_abund[Z][isotope_num]))

            delta_log_gf.append(sum(log_probs) if log_probs else 0.0)

        delta_log_gf = np.array(delta_log_gf)
    else:
        delta_log_gf = np.zeros(len(df))

    # Approximate radiative broadening when missing
    gamma_rad = []
    for i in range(len(df)):
        if df['gamma_rad'].iloc[i] == 0:
            gamma_rad.append(approximate_radiative_gamma(wl[i], df['loggf'].iloc[i]))
        else:
            gamma_rad.append(10**df['gamma_rad'].iloc[i])

    # Create Line objects
    result_lines = []
    for i in range(len(df)):
        try:
            spec = Species(df['species'].iloc[i].strip('"'))
            line = Line(
                wl=wl[i],
                log_gf=df['loggf'].iloc[i] + delta_log_gf[i],
                species=spec,
                E_lower=E_low[i],
                gamma_rad=gamma_rad[i],
                gamma_stark=ten_to_the_or_missing(df['gamma_stark'].iloc[i]),
                vdW=id_or_missing(df['gamma_vdW'].iloc[i])
            )
            result_lines.append(line)
        except Exception as e:
            # Skip lines that can't be parsed
            print(f"Warning: Skipping line {i}: {e}")
            continue

    return result_lines


def read_vald_linelist(filename: str, isotopic_abund: Dict = None) -> List[Line]:
    """
    Read and parse a VALD linelist file.

    Args:
        filename: Path to VALD linelist file
        isotopic_abund: Dictionary of isotopic abundances (optional)

    Returns:
        List of Line objects sorted by wavelength
    """
    with open(filename, 'r') as f:
        content = f.read()

    lines = parse_vald_linelist(content, isotopic_abund)

    # Filter out highly ionized species and hydrogen lines
    lines = [line for line in lines
             if 0 <= line.species.charge <= 2 and line.species != Species("H I")]

    # Sort by wavelength
    lines.sort(key=lambda l: l.wl)

    return lines
