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
import jax.numpy as jnp
import numpy as np
import h5py
from typing import Dict, Optional


# Module-level cache for cross-section data (JAX arrays)
_metal_bf_data: Optional[Dict] = None


def _get_data_path() -> str:
    """Path to the metal bound-free cross-section table, inside the package.

    This used to walk three directories up from here -- to the *repository*
    root -- and read a second, byte-identical 28 MB copy of the table kept at
    ``data/``. That was a leftover from a ``python_src/korg/`` layout the
    comment still described and the tree no longer had. Only a source checkout
    has a repository root: the wheel ships ``src/korg/data`` and nothing else,
    so an installed copy would have raised here. Resolve against the package,
    the way ``data_loader._DATA_DIR`` already does.
    """
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(package_dir, 'data', 'bf_cross-sections',
                        'bf_cross-sections.h5')


def _bilinear_interp_2d(table, x_grid, y_grid, x, y):
    """
    JAX-compatible bilinear interpolation on a regular 2D grid.

    Parameters
    ----------
    table : jnp.ndarray
        2D table of values with shape (len(x_grid), len(y_grid))
    x_grid : jnp.ndarray
        1D array of x coordinates (monotonically increasing)
    y_grid : jnp.ndarray
        1D array of y coordinates (monotonically increasing)
    x : float or array
        x coordinate(s) to interpolate at
    y : float or array
        y coordinate(s) to interpolate at

    Returns
    -------
    float or array
        Interpolated value(s)
    """
    # Convert to JAX arrays
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    # Find indices using searchsorted
    i_x = jnp.searchsorted(x_grid, x, side='right') - 1
    i_y = jnp.searchsorted(y_grid, y, side='right') - 1

    # Clip to valid range
    i_x = jnp.clip(i_x, 0, len(x_grid) - 2)
    i_y = jnp.clip(i_y, 0, len(y_grid) - 2)

    # Get grid points
    x0 = x_grid[i_x]
    x1 = x_grid[i_x + 1]
    y0 = y_grid[i_y]
    y1 = y_grid[i_y + 1]

    # Compute fractional positions
    t_x = jnp.where(x1 != x0, (x - x0) / (x1 - x0), 0.0)
    t_y = jnp.where(y1 != y0, (y - y0) / (y1 - y0), 0.0)

    # Get corner values from table (table is [x, y])
    v00 = table[i_x, i_y]
    v01 = table[i_x, i_y + 1]
    v10 = table[i_x + 1, i_y]
    v11 = table[i_x + 1, i_y + 1]

    # Bilinear interpolation
    result = ((1.0 - t_x) * (1.0 - t_y) * v00 +
              (1.0 - t_x) * t_y * v01 +
              t_x * (1.0 - t_y) * v10 +
              t_x * t_y * v11)

    return result


def _load_metal_bf_data() -> Dict:
    """
    Load metal bound-free cross-sections from HDF5 file into JAX arrays.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'logT_grid': JAX array of log10(T) values
        - 'nu_grid': JAX array of frequency values
        - 'species': Dict mapping species name to 2D JAX array of log(sigma) values
    """
    global _metal_bf_data

    if _metal_bf_data is not None:
        return _metal_bf_data

    data_path = _get_data_path()
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Metal bf cross-section data file not found at: {data_path}"
        )

    with h5py.File(data_path, 'r') as f:
        # Read grid parameters
        logT_min = float(f['logT_min'][()])
        logT_max = float(f['logT_max'][()])
        logT_step = float(f['logT_step'][()])
        nu_min = float(f['nu_min'][()])
        nu_max = float(f['nu_max'][()])
        nu_step = float(f['nu_step'][()])

        # Create grids as numpy arrays (converted to JAX inside metal_bf_absorption)
        logT_grid = np.arange(logT_min, logT_max + logT_step/2, logT_step)
        nu_grid = np.arange(nu_min, nu_max + nu_step/2, nu_step)

        # Load cross-sections for each species into numpy arrays
        species_data = {}
        for name in f['cross-sections'].keys():
            sigma = f['cross-sections'][name][:]
            # Data is stored as (nT, nfreq) in HDF5
            species_data[name] = np.array(sigma, dtype=np.float64)

    _metal_bf_data = {
        'logT_grid': logT_grid,
        'nu_grid': nu_grid,
        'species': species_data
    }

    return _metal_bf_data


def metal_bf_absorption(
    nus,
    T: float,
    number_densities: Dict[str, float],
    out_alpha=None
):
    """
    Compute metal bound-free absorption coefficients (JAX-compatible).

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
    extrapolation is used (i.e. the extreme value is returned via clipping).

    H I, He I, and H II are skipped even if present in number_densities,
    as these are handled with other approximations.

    This function is JAX-compatible and can be used with jax.jit.
    """
    # Convert to JAX arrays
    nus = jnp.asarray(nus)

    # Load data (cached at module level as numpy arrays; convert to JAX here)
    data = _load_metal_bf_data()
    logT_grid = jnp.asarray(data['logT_grid'])
    nu_grid = jnp.asarray(data['nu_grid'])
    species_data = data['species']

    if out_alpha is None:
        out_alpha = jnp.zeros(len(nus), dtype=jnp.float64)
    else:
        out_alpha = jnp.asarray(out_alpha)

    log_T = jnp.log10(T)

    # Species to skip (handled elsewhere)
    skip_species = {'H I', 'He I', 'H II'}

    for species_name, sigma_table_np in species_data.items():
        if species_name in skip_species:
            continue
        if species_name not in number_densities:
            continue

        n_density = number_densities[species_name]
        sigma_table = jnp.asarray(sigma_table_np)

        # Interpolate cross-sections for this species
        # Table is indexed as (logT, nu)
        # We have a constant logT and varying nus
        log_sigma = _bilinear_interp_2d(sigma_table, logT_grid, nu_grid, log_T, nus)

        # Apply mask for finite values (avoid NaN from log(0))
        # In the table, log(0) is represented as -inf
        mask = jnp.isfinite(log_sigma)

        # The tables store the *natural* log of sigma in Megabarns, so exponentiate
        # directly (Korg.jl: `exp.(log(n) .+ log_σ) * 1e-18`). Scaling log_sigma by
        # ln(10) here treated the table as log10 and was wrong by orders of magnitude.
        # sigma[cm^2] = exp(log_sigma) * 1e-18
        # Only add where cross-section is finite and density is positive
        contribution = jnp.where(
            mask,
            jnp.exp(jnp.log(jnp.maximum(n_density, 1e-300)) + log_sigma) * 1e-18,
            0.0
        )
        out_alpha = out_alpha + contribution

    return out_alpha


def get_available_species() -> list:
    """
    Get list of species for which metal bf cross-sections are available.

    Returns
    -------
    list
        List of species names (e.g., ['Al I', 'C I', 'Ca I', ...])
    """
    data = _load_metal_bf_data()
    return list(data['species'].keys())
