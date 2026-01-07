"""
Hydrogenic bound-free and free-free absorption.

This module provides free-free (thermal bremsstrahlung) absorption coefficients
for hydrogenic species (atoms/ions with a single electron). The core functionality
uses thermally-averaged Gaunt factors from van Hoof et al. (2014).

References:
    van Hoof et al. (2014): https://ui.adsabs.harvard.edu/abs/2014MNRAS.444..420V
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import RegularGridInterpolator
from ..constants import hplanck_cgs, hplanck_eV, kboltz_cgs, kboltz_eV, c_cgs, Rydberg_eV


# Module-level cache for interpolator
_gauntff_interpolator = None
_gauntff_table_data = None

# JAX-compatible cache
_jax_gauntff_table = None
_jax_log10_u = None
_jax_log10_γ2 = None


def _load_gauntff_table(fname=None):
    """
    Load thermally-averaged free-free Gaunt factor table from van Hoof et al. (2014).

    This loads the non-relativistic free-free data published by van Hoof et al. (2014).
    The table contains Gaunt factors as a function of log₁₀(γ²) and log₁₀(u).

    Parameters
    ----------
    fname : str, optional
        Path to the data file. If None, uses the default location in Korg's data directory.

    Returns
    -------
    table : np.ndarray
        2D array of Gaunt factors with shape (num_u, num_γ2)
    log10_γ2 : np.ndarray
        Grid values for log₁₀(γ²)
    log10_u : np.ndarray
        Grid values for log₁₀(u)

    Notes
    -----
    The table format:
    - Magic number: 20140210
    - Grid: 81 (γ²) × 146 (u) points
    - Start: log₁₀(γ²) = -6, log₁₀(u) = -16
    - Step: 0.2 dex for both axes
    - Valid ranges: T ∈ [100, 1e6] K, λ ∈ [100 Å, 100 μm]

    The non-relativistic approach is valid up to ~100 MK electron temperatures,
    which is more than adequate for stellar atmospheres.
    """
    if fname is None:
        # Find the data directory relative to this file (inside the package)
        module_dir = os.path.dirname(__file__)
        fname = os.path.join(module_dir, '..', 'data', 'vanHoof2014-nr-gauntff.dat')

    def parse_header_line(line, dtype=float):
        """Parse a header line, removing comments."""
        # Find comment marker
        comment_idx = line.find('#')
        if comment_idx >= 0:
            line = line[:comment_idx]
        values = line.split()
        return [dtype(v) for v in values]

    with open(fname, 'r') as f:
        # Skip initial comments
        for line in f:
            if not line.strip().startswith('#'):
                break

        # Parse header
        magic_number = parse_header_line(line, int)[0]
        assert magic_number == 20140210, f"Invalid magic number: {magic_number}"

        line = f.readline()
        num_γ2, num_u = parse_header_line(line, int)

        line = f.readline()
        log10_γ2_start = parse_header_line(line, float)[0]

        line = f.readline()
        log10_u_start = parse_header_line(line, float)[0]

        line = f.readline()
        step_size = parse_header_line(line, float)[0]

        # Create coordinate arrays
        log10_γ2 = np.arange(num_γ2) * step_size + log10_γ2_start
        log10_u = np.arange(num_u) * step_size + log10_u_start

        # Skip comments until we hit data
        for line in f:
            if not line.strip().startswith('#'):
                break

        # Read data table - first row is already in `line`
        table = np.zeros((num_u, num_γ2))
        table[0, :] = [float(x) for x in line.split()]

        for i in range(1, num_u):
            line = f.readline()
            table[i, :] = [float(x) for x in line.split()]

    return table, log10_γ2, log10_u


def _initialize_interpolator():
    """
    Initialize the Gaunt factor interpolator.

    This is called once when the module is first used. The interpolator is cached
    to avoid re-loading the data file on subsequent calls.

    Returns
    -------
    interpolator : RegularGridInterpolator
        2D interpolator for Gaunt factors
    """
    global _gauntff_interpolator, _gauntff_table_data

    if _gauntff_interpolator is not None:
        return _gauntff_interpolator

    # Load the table
    table, log10_γ2, log10_u = _load_gauntff_table()

    # Trim the table to reduce memory usage
    # We only need ranges relevant for stellar atmospheres:
    # T ∈ [100, 1e6] K, λ ∈ [1e-6, 1e-2] cm (100 Å to 100 μm), Z ∈ [1, 2]
    T_extrema = [100.0, 1e6]  # K
    λ_extrema = [1.0e-6, 1.0e-2]  # cm
    Z_extrema = [1, 2]

    def calc_log10_γ2(Z, T):
        """Calculate log₁₀(γ²) = log₁₀(Rydberg * Z² / (k * T))"""
        return np.log10(Rydberg_eV * Z**2 / (kboltz_eV * T))

    def calc_log10_u(λ, T):
        """Calculate log₁₀(u) = log₁₀(h * c / (λ * k * T))"""
        return np.log10(hplanck_cgs * c_cgs / (λ * kboltz_cgs * T))

    # Find bounds for trimming
    γ2_vals = [calc_log10_γ2(Z, T) for Z in Z_extrema for T in T_extrema]
    u_vals = [calc_log10_u(λ, T) for λ in λ_extrema for T in T_extrema]

    γ2_min, γ2_max = min(γ2_vals), max(γ2_vals)
    u_min, u_max = min(u_vals), max(u_vals)

    # Find indices that bracket the required range
    γ2_lb = np.searchsorted(log10_γ2, γ2_min, side='right') - 1
    γ2_ub = np.searchsorted(log10_γ2, γ2_max, side='left') + 1
    u_lb = np.searchsorted(log10_u, u_min, side='right') - 1
    u_ub = np.searchsorted(log10_u, u_max, side='left') + 1

    # Ensure we stay within array bounds
    γ2_lb = max(0, γ2_lb)
    γ2_ub = min(len(log10_γ2), γ2_ub)
    u_lb = max(0, u_lb)
    u_ub = min(len(log10_u), u_ub)

    # Extract trimmed table
    trimmed_table = table[u_lb:u_ub, γ2_lb:γ2_ub].copy()
    trimmed_log10_u = log10_u[u_lb:u_ub]
    trimmed_log10_γ2 = log10_γ2[γ2_lb:γ2_ub]

    # Create interpolator
    # Note: RegularGridInterpolator expects points in (u, γ2) order
    _gauntff_interpolator = RegularGridInterpolator(
        (trimmed_log10_u, trimmed_log10_γ2),
        trimmed_table,
        method='linear',
        bounds_error=True
    )

    # Store bounds for validation
    _gauntff_table_data = {
        'u_bounds': (trimmed_log10_u[0], trimmed_log10_u[-1]),
        'γ2_bounds': (trimmed_log10_γ2[0], trimmed_log10_γ2[-1])
    }

    return _gauntff_interpolator


def _initialize_jax_tables():
    """
    Initialize JAX-compatible arrays for Gaunt factor interpolation.

    This loads the trimmed table into JAX arrays for use with JIT-compiled functions.
    """
    global _jax_gauntff_table, _jax_log10_u, _jax_log10_γ2

    if _jax_gauntff_table is not None:
        return _jax_gauntff_table, _jax_log10_u, _jax_log10_γ2

    # Load the table (reuse existing loader)
    table, log10_γ2, log10_u = _load_gauntff_table()

    # Trim the table (same logic as _initialize_interpolator)
    T_extrema = [100.0, 1e6]
    λ_extrema = [1.0e-6, 1.0e-2]
    Z_extrema = [1, 2]

    def calc_log10_γ2(Z, T):
        return np.log10(Rydberg_eV * Z**2 / (kboltz_eV * T))

    def calc_log10_u(λ, T):
        return np.log10(hplanck_cgs * c_cgs / (λ * kboltz_cgs * T))

    γ2_vals = [calc_log10_γ2(Z, T) for Z in Z_extrema for T in T_extrema]
    u_vals = [calc_log10_u(λ, T) for λ in λ_extrema for T in T_extrema]

    γ2_min, γ2_max = min(γ2_vals), max(γ2_vals)
    u_min, u_max = min(u_vals), max(u_vals)

    γ2_lb = max(0, np.searchsorted(log10_γ2, γ2_min, side='right') - 1)
    γ2_ub = min(len(log10_γ2), np.searchsorted(log10_γ2, γ2_max, side='left') + 1)
    u_lb = max(0, np.searchsorted(log10_u, u_min, side='right') - 1)
    u_ub = min(len(log10_u), np.searchsorted(log10_u, u_max, side='left') + 1)

    # Convert to JAX arrays
    _jax_gauntff_table = jnp.array(table[u_lb:u_ub, γ2_lb:γ2_ub])
    _jax_log10_u = jnp.array(log10_u[u_lb:u_ub])
    _jax_log10_γ2 = jnp.array(log10_γ2[γ2_lb:γ2_ub])

    return _jax_gauntff_table, _jax_log10_u, _jax_log10_γ2


def _bilinear_interp(table, x_grid, y_grid, x, y):
    """
    JAX-compatible bilinear interpolation on a regular grid.

    Parameters
    ----------
    table : jnp.ndarray
        2D table of values with shape (len(x_grid), len(y_grid))
    x_grid : jnp.ndarray
        1D array of x coordinates (must be uniformly spaced)
    y_grid : jnp.ndarray
        1D array of y coordinates (must be uniformly spaced)
    x : float
        x coordinate to interpolate at
    y : float
        y coordinate to interpolate at

    Returns
    -------
    float
        Interpolated value
    """
    # Get grid spacing (assumes uniform)
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    # Find indices
    x_idx_f = (x - x_grid[0]) / dx
    y_idx_f = (y - y_grid[0]) / dy

    # Clamp to valid range
    x_idx_f = jnp.clip(x_idx_f, 0, len(x_grid) - 1.001)
    y_idx_f = jnp.clip(y_idx_f, 0, len(y_grid) - 1.001)

    x_idx = jnp.floor(x_idx_f).astype(jnp.int32)
    y_idx = jnp.floor(y_idx_f).astype(jnp.int32)

    # Ensure we don't go out of bounds
    x_idx = jnp.minimum(x_idx, len(x_grid) - 2)
    y_idx = jnp.minimum(y_idx, len(y_grid) - 2)

    # Fractional parts
    x_frac = x_idx_f - x_idx
    y_frac = y_idx_f - y_idx

    # Bilinear interpolation
    v00 = table[x_idx, y_idx]
    v01 = table[x_idx, y_idx + 1]
    v10 = table[x_idx + 1, y_idx]
    v11 = table[x_idx + 1, y_idx + 1]

    return (v00 * (1 - x_frac) * (1 - y_frac) +
            v01 * (1 - x_frac) * y_frac +
            v10 * x_frac * (1 - y_frac) +
            v11 * x_frac * y_frac)


def gaunt_ff_vanHoof_jax(log_u, log_γ2, table, log10_u_grid, log10_γ2_grid):
    """
    JIT-compatible version of gaunt_ff_vanHoof.

    Parameters
    ----------
    log_u : float
        log₁₀(u) where u = h*ν/(k*T)
    log_γ2 : float
        log₁₀(γ²) where γ² = Rydberg*Z²/(k*T)
    table : jnp.ndarray
        Gaunt factor table
    log10_u_grid : jnp.ndarray
        Grid of log₁₀(u) values
    log10_γ2_grid : jnp.ndarray
        Grid of log₁₀(γ²) values

    Returns
    -------
    float
        Thermally-averaged free-free Gaunt factor
    """
    return _bilinear_interp(table, log10_u_grid, log10_γ2_grid, log_u, log_γ2)


def hydrogenic_ff_absorption_jax(ν, T, Z, ni, ne, table, log10_u_grid, log10_γ2_grid):
    """
    JIT-compatible free-free absorption coefficient.

    Parameters
    ----------
    ν : float
        Frequency in Hz
    T : float
        Temperature in K
    Z : int
        Charge of the ion
    ni : float
        Ion number density in cm⁻³
    ne : float
        Electron number density in cm⁻³
    table : jnp.ndarray
        Gaunt factor table
    log10_u_grid : jnp.ndarray
        Grid of log₁₀(u) values
    log10_γ2_grid : jnp.ndarray
        Grid of log₁₀(γ²) values

    Returns
    -------
    float
        Linear absorption coefficient in cm⁻¹
    """
    inv_T = 1.0 / T
    Z2 = Z * Z

    # Compute log₁₀(u) = log₁₀(hν/kT)
    hν_div_kT = (hplanck_eV / kboltz_eV) * ν * inv_T
    log_u = jnp.log10(hν_div_kT)

    # Compute log₁₀(γ²) = log₁₀(Rydberg*Z²/kT)
    log_γ2 = jnp.log10((Rydberg_eV / kboltz_eV) * Z2 * inv_T)

    # Get Gaunt factor
    gaunt_ff = gaunt_ff_vanHoof_jax(log_u, log_γ2, table, log10_u_grid, log10_γ2_grid)

    # Compute absorption coefficient
    F_ν = 3.6919e8 * gaunt_ff * Z2 * jnp.sqrt(inv_T) / (ν * ν * ν)
    α = ni * ne * F_ν * (1 - jnp.exp(-hν_div_kT))

    return α


def gaunt_ff_vanHoof(log_u, log_γ2):
    """
    Compute thermally-averaged free-free Gaunt factor.

    Interpolates the table from van Hoof et al. (2014) to compute the
    non-relativistic, thermally-averaged free-free Gaunt factor.

    Parameters
    ----------
    log_u : float or array_like
        log₁₀(u) where u = h*ν/(k*T)
        This is the ratio of photon energy to thermal energy.
    log_γ2 : float or array_like
        log₁₀(γ²) where γ² = Rydberg*Z²/(k*T)
        This characterizes the ionization energy relative to thermal energy.

    Returns
    -------
    g_ff : float or array_like
        Thermally-averaged free-free Gaunt factor

    Notes
    -----
    The non-relativistic approach is accurate up to ~100 MK (more than adequate
    for stellar atmospheres). Relativistic effects introduce ~0.75% error at 100 MK
    for Z=1 (smaller for Z>1).

    This function uses linear interpolation. Van Hoof et al. (2014) note that
    third-order Lagrange interpolation achieves relative precision better than
    1.5e-4 everywhere, but linear interpolation is sufficient for most applications.

    References
    ----------
    van Hoof et al. (2014): https://ui.adsabs.harvard.edu/abs/2014MNRAS.444..420V
    """
    # Initialize interpolator if needed
    interpolator = _initialize_interpolator()

    # Handle scalar and array inputs
    log_u = np.atleast_1d(log_u)
    log_γ2 = np.atleast_1d(log_γ2)

    # Create points array for interpolation
    # RegularGridInterpolator expects shape (n_points, n_dims)
    if log_u.shape == log_γ2.shape:
        # Matched arrays
        points = np.column_stack([log_u, log_γ2])
    elif log_u.size == 1:
        # Broadcast scalar log_u
        points = np.column_stack([np.full_like(log_γ2, log_u[0]), log_γ2])
    elif log_γ2.size == 1:
        # Broadcast scalar log_γ2
        points = np.column_stack([log_u, np.full_like(log_u, log_γ2[0])])
    else:
        raise ValueError("log_u and log_γ2 must have compatible shapes")

    # Interpolate
    result = interpolator(points)

    # Return scalar if input was scalar
    if result.size == 1:
        return float(result[0])
    return result


def hydrogenic_ff_absorption(ν, T, Z, ni, ne):
    """
    Compute free-free (thermal bremsstrahlung) linear absorption coefficient.

    Calculates the free-free absorption coefficient for a hydrogenic species
    (an ion with a single bound electron) using the thermally-averaged Gaunt
    factors from van Hoof et al. (2014).

    Parameters
    ----------
    ν : float or array_like
        Frequency in Hz
    T : float
        Temperature in K
    Z : int
        Charge of the ion. For example:
        - Z=1 for H II (ionized hydrogen)
        - Z=2 for He III (doubly ionized helium)
    ni : float
        Number density of the ion in cm⁻³
    ne : float
        Number density of free electrons in cm⁻³

    Returns
    -------
    α : float or array_like
        Linear absorption coefficient in cm⁻¹
        (corrected for stimulated emission)

    Notes
    -----
    The naming convention for free-free absorption is counter-intuitive. A free-free
    interaction is named as though the species interacting with the free electron
    had one more bound electron. In practice, this means:
    - For "H I free-free", use ni = n(H II) with Z=1
    - For "He II free-free", use ni = n(He III) with Z=2

    The formula (from Rybicki & Lightman 2004, equation 5.18b) is:

        α = coef * Z² * ne * ni * (1 - exp(-hν/kT)) * g_ff / (√T * ν³)

    where coef ≈ 3.6919e8 and g_ff is the free-free Gaunt factor from van Hoof et al.

    This corrects an omission in Kurucz (1970) equation 5.8, which left out the
    dependence on density (see notes in the Julia implementation).

    References
    ----------
    van Hoof et al. (2014): https://ui.adsabs.harvard.edu/abs/2014MNRAS.444..420V
    Rybicki & Lightman (2004): Radiative Processes in Astrophysics
    Kurucz (1970): SAO Special Report 309
    """
    # Convert to numpy arrays for vectorization
    ν = np.atleast_1d(ν)

    # Compute dimensionless parameters
    inv_T = 1.0 / T
    Z2 = Z * Z

    # Compute log₁₀(u) = log₁₀(hν/kT)
    hν_div_kT = (hplanck_eV / kboltz_eV) * ν * inv_T
    log_u = np.log10(hν_div_kT)

    # Compute log₁₀(γ²) = log₁₀(Rydberg*Z²/kT)
    log_γ2 = np.log10((Rydberg_eV / kboltz_eV) * Z2 * inv_T)

    # Get Gaunt factor
    gaunt_ff = gaunt_ff_vanHoof(log_u, log_γ2)

    # Compute F_ν = coef * Z² * g_ff / (√T * ν³)
    # where coef ≈ 3.6919e8
    F_ν = 3.6919e8 * gaunt_ff * Z2 * np.sqrt(inv_T) / (ν * ν * ν)

    # Compute absorption coefficient with stimulated emission correction
    α = ni * ne * F_ν * (1 - np.exp(-hν_div_kT))

    # Return scalar if input was scalar
    if α.size == 1:
        return float(α)
    return α
