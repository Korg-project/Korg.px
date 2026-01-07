"""
Metal bound-free absorption.

Uses precomputed tables from TOPBase and NORAD for various metals.

Elements covered:
- TOPBase: Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, S, Ar, Ca
- NORAD: Fe

For each element, tables are available for neutral and singly ionized species,
assuming LTE distribution of energy levels.

Valid ranges:
- Temperature: 100 K < T < 100,000 K (log10 T from 2.0 to 5.0)
- Wavelength: 500 Å < λ < 30,000 Å

Reference: Korg.jl ContinuumAbsorption/absorption_metals_bf.jl
"""

import os
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
from typing import Dict, Optional


# Module-level cache for cross-section interpolators
_metal_bf_cross_sections: Optional[Dict[str, RegularGridInterpolator]] = None


def _get_data_path() -> str:
    """Get path to the data directory."""
    # Path relative to the Korg.jl package root
    # This assumes the package structure: Korg.jl/python_src/korg/continuum_absorption/
    this_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up 3 levels to reach Korg.jl root, then into data
    korg_root = os.path.dirname(os.path.dirname(os.path.dirname(this_dir)))
    return os.path.join(korg_root, 'data', 'bf_cross-sections', 'bf_cross-sections.h5')


def _load_metal_bf_cross_sections() -> Dict[str, RegularGridInterpolator]:
    """
    Load metal bound-free cross-sections from HDF5 file.

    Returns
    -------
    dict
        Dictionary mapping species name to interpolator.
        Interpolator takes (frequency, log10(T)) and returns log10(cross-section).
    """
    global _metal_bf_cross_sections

    if _metal_bf_cross_sections is not None:
        return _metal_bf_cross_sections

    data_path = _get_data_path()
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Metal bf cross-section data file not found at: {data_path}"
        )

    cross_sections = {}

    with h5py.File(data_path, 'r') as f:
        # Read grid parameters
        logT_min = f['logT_min'][()]
        logT_max = f['logT_max'][()]
        logT_step = f['logT_step'][()]
        nu_min = f['nu_min'][()]
        nu_max = f['nu_max'][()]
        nu_step = f['nu_step'][()]

        # Create grids
        T_grid = np.arange(logT_min, logT_max + logT_step/2, logT_step)
        nu_grid = np.arange(nu_min, nu_max + nu_step/2, nu_step)

        # Load cross-sections for each species
        for name in f['cross-sections'].keys():
            sigma = f['cross-sections'][name][:]

            # Create interpolator with flat extrapolation
            # Data is stored as (nT, nfreq) in HDF5
            # Need to transpose for (nu, T) ordering expected by our interpolator
            interp = RegularGridInterpolator(
                (T_grid, nu_grid),
                sigma,
                method='linear',
                bounds_error=False,
                fill_value=None  # Use nearest for out-of-bounds (flat extrapolation)
            )
            cross_sections[name] = interp

    _metal_bf_cross_sections = cross_sections
    return cross_sections


def metal_bf_absorption(
    nus: np.ndarray,
    T: float,
    number_densities: Dict[str, float],
    out_alpha: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute metal bound-free absorption coefficients.

    Uses precomputed tables from TOPBase and NORAD for various metals.

    Parameters
    ----------
    nus : array
        Frequencies in Hz
    T : float
        Temperature in K
    number_densities : dict
        Dictionary mapping species names (e.g., "Fe I", "C I") to number
        densities in cm^-3
    out_alpha : array, optional
        Output array to add results to (in-place operation).
        If None, a new array is created.

    Returns
    -------
    array
        Absorption coefficients in cm^-1

    Notes
    -----
    Cross sections were computed for 100 K < T < 100,000 K and frequencies
    corresponding to 500 Å < λ < 30,000 Å. Outside of either range, flat
    extrapolation is used (i.e. the extreme value is returned).

    H I, He I, and H II are skipped even if present in number_densities,
    as these are handled with other approximations.
    """
    cross_sections = _load_metal_bf_cross_sections()

    if out_alpha is None:
        out_alpha = np.zeros(len(nus), dtype=np.float64)

    log_T = np.log10(T)

    # Species to skip (handled elsewhere)
    skip_species = {'H I', 'He I', 'H II'}

    for species_name, interp in cross_sections.items():
        if species_name in skip_species:
            continue
        if species_name not in number_densities:
            continue

        n_density = number_densities[species_name]
        if n_density <= 0:
            continue

        # Evaluate interpolator at all frequencies
        # Grid is (T, nu), so points should be (log_T, nu)
        points = np.column_stack([np.full(len(nus), log_T), nus])
        log_sigma = interp(points)

        # Apply mask for finite values (avoid NaN from log(0))
        mask = np.isfinite(log_sigma)
        if np.any(mask):
            # Convert from log10(sigma) in 10^-18 cm^2 to cm^2 and multiply by density
            # sigma is in units of 10^-18 cm^2 in the table
            out_alpha[mask] += np.exp(np.log(n_density) + log_sigma[mask]) * 1e-18

    return out_alpha


def get_available_species() -> list:
    """
    Get list of species for which metal bf cross-sections are available.

    Returns
    -------
    list
        List of species names (e.g., ['Al I', 'C I', 'Ca I', ...])
    """
    cross_sections = _load_metal_bf_cross_sections()
    return list(cross_sections.keys())
