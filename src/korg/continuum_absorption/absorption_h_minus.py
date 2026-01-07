"""
H⁻ (H minus) bound-free and free-free absorption.

This module computes continuum opacity from H⁻ ions, which are the dominant
source of opacity in cool stellar atmospheres (Teff ~ 4000-7000 K) in the
visible and near-infrared wavelength ranges.

References:
    H⁻ bf: McLaughlin (2017) - https://ui.adsabs.harvard.edu/abs/2017JPhB...50k4001M
    H⁻ ff: Bell & Berrington (1987) - https://doi.org/10.1088/0022-3700/20/4/019
"""

import os
import numpy as np
import jax.numpy as jnp
from scipy.interpolate import RegularGridInterpolator
import h5py

from ..constants import hplanck_cgs, hplanck_eV, kboltz_cgs, kboltz_eV, c_cgs


# H⁻ ionization energy from McLaughlin+ 2017
H_MINUS_ION_ENERGY_EV = 0.754204  # eV

# Module-level cache for interpolators
_Hminus_bf_interpolator = None
_Hminus_bf_min_nu = None
_Hminus_ff_interpolator = None


def _load_Hminus_bf_data(fname=None):
    """
    Load H⁻ bound-free cross-section data from McLaughlin (2017).

    Parameters
    ----------
    fname : str, optional
        Path to the HDF5 data file. If None, uses default location.

    Returns
    -------
    nu : np.ndarray
        Frequency grid in Hz
    sigma : np.ndarray
        Cross-section in cm²

    Notes
    -----
    McLaughlin+ 2017 provides high-precision photodetachment cross-sections
    for H⁻ computed using R-matrix methods. The tabulated data covers
    photon energies from the ionization threshold (0.754204 eV) up to higher
    energies.

    For energies below the minimum tabulated value, cross-sections scale as
    σ ∝ (E - E₀)^1.5 where E₀ is the ionization energy.
    """
    if fname is None:
        # Find data directory (inside the package)
        module_dir = os.path.dirname(__file__)
        fname = os.path.join(module_dir, '..', 'data', 'McLaughlin2017Hminusbf.h5')

    with h5py.File(fname, 'r') as f:
        nu = f['nu'][:]
        sigma = f['sigma'][:]

    return nu, sigma


def _initialize_Hminus_bf_interpolator():
    """Initialize H⁻ bound-free cross-section interpolator."""
    global _Hminus_bf_interpolator, _Hminus_bf_min_nu

    if _Hminus_bf_interpolator is not None:
        return _Hminus_bf_interpolator, _Hminus_bf_min_nu

    nu, sigma = _load_Hminus_bf_data()

    # Create linear interpolator
    _Hminus_bf_interpolator = RegularGridInterpolator(
        (nu,),
        sigma,
        method='linear',
        bounds_error=True
    )
    _Hminus_bf_min_nu = nu[0]

    return _Hminus_bf_interpolator, _Hminus_bf_min_nu


def _ndens_Hminus(nH_I_div_partition, ne, T, ion_energy=H_MINUS_ION_ENERGY_EV):
    """
    Compute number density of H⁻ using Saha equation.

    This implements equation 5.10 of Kurucz (1970). The Saha equation is
    applied where the "ground state" is H⁻ and the "first ionization state"
    is H I. The partition function of H⁻ is 1 at all temperatures.

    Parameters
    ----------
    nH_I_div_partition : float
        Number density of H I divided by its partition function (cm⁻³)
    ne : float
        Electron number density (cm⁻³)
    T : float
        Temperature (K)
    ion_energy : float, optional
        H⁻ ionization energy in eV (default: 0.754204)

    Returns
    -------
    n_Hminus : float
        Number density of H⁻ in cm⁻³

    Notes
    -----
    The formula is:
        n(H⁻) = 0.25 × n(H I, gs) × ne × coef × β^1.5 × exp(E_ion × β)

    where:
        - n(H I, gs) = 2 × n(H I) / U(T) is the ground state H I density
          (Boltzmann factor is 1, degeneracy is 2)
        - coef = (h²/(2πm))^1.5 ≈ 3.31283018e-22 cm³·eV^1.5
        - β = 1/(k_B T) in eV^-1
    """
    if T < 1000:
        raise ValueError(f"Temperature {T} K is unexpectedly low for H⁻ calculation")

    # Ground state H I number density: Boltzmann factor = 1, degeneracy = 2
    nHI_groundstate = 2 * nH_I_div_partition

    # Coefficient: (h²/(2πm_e))^1.5
    coef = 3.31283018e-22  # cm³·eV^1.5

    # Inverse temperature in eV
    beta = 1.0 / (kboltz_eV * T)

    return 0.25 * nHI_groundstate * ne * coef * beta**1.5 * np.exp(ion_energy * beta)


def _Hminus_bf_cross_section(nu):
    """
    Get H⁻ bound-free cross-section at given frequency.

    Parameters
    ----------
    nu : float or array_like
        Frequency in Hz

    Returns
    -------
    sigma : float or array_like
        Cross-section in cm² (excludes stimulated emission)

    Notes
    -----
    Uses McLaughlin+ 2017 data for tabulated range.
    Below minimum tabulated frequency, uses scaling: σ ∝ (ν - ν_ion)^1.5
    Below ionization threshold, returns 0.
    """
    # Initialize interpolator if needed
    interpolator, min_nu = _initialize_Hminus_bf_interpolator()

    # Ionization frequency
    nu_ion = H_MINUS_ION_ENERGY_EV / hplanck_eV

    # Handle scalar and array inputs
    nu = np.atleast_1d(nu)
    sigma = np.zeros_like(nu, dtype=float)

    # Below ionization threshold: σ = 0
    above_threshold = nu > nu_ion

    if not np.any(above_threshold):
        return sigma if sigma.size > 1 else 0.0

    # Below minimum tabulated frequency: use power-law scaling
    # McLaughlin+ 2017 notes that for E < 0.7678 eV, σ = 460.8*(E - E₀)^1.5 Mb
    # We use the same scaling in terms of frequency
    below_min = above_threshold & (nu < min_nu)
    if np.any(below_min):
        # Get scaling coefficient from first tabulated point
        sigma_min = interpolator((min_nu,))[0]
        coef = sigma_min / (min_nu - nu_ion)**1.5
        sigma[below_min] = coef * (nu[below_min] - nu_ion)**1.5

    # In tabulated range: interpolate
    in_range = nu >= min_nu
    if np.any(in_range):
        sigma[in_range] = interpolator(nu[in_range])

    # Return scalar if input was scalar
    if sigma.size == 1:
        return float(sigma[0])
    return sigma


def Hminus_bf(nu, T, nH_I_div_partition, ne):
    """
    Compute H⁻ bound-free linear absorption coefficient.

    The absorption coefficient is:
        α_ν = σ_bf(H⁻) × n(H⁻) × (1 - exp(-hν/kT))

    Parameters
    ----------
    nu : float or array_like
        Frequency in Hz (must be sorted if array)
    T : float
        Temperature in K
    nH_I_div_partition : float
        Total number density of H I divided by its partition function (cm⁻³)
    ne : float
        Electron number density (cm⁻³)

    Returns
    -------
    alpha : float or array_like
        Linear absorption coefficient in cm⁻¹

    Notes
    -----
    This uses cross-sections from McLaughlin (2017), which are accurate
    for stellar atmosphere applications. H⁻ bound-free absorption is the
    dominant opacity source in the visible for cool stars (4000-7000 K).

    The function assumes n(H⁻) ≪ n(H I) + n(H II), so H⁻ number density
    is computed on-the-fly rather than from molecular equilibrium.

    Valid range:
        - Frequency: > ionization threshold (ν > 1.82e14 Hz, λ < 1.644 μm)
        - Temperature: > 1000 K (practical lower limit)

    References
    ----------
    McLaughlin (2017): https://ui.adsabs.harvard.edu/abs/2017JPhB...50k4001M
    """
    # Get cross-section (in cm²)
    cross_section = _Hminus_bf_cross_section(nu)

    # Compute H⁻ number density
    n_Hminus = _ndens_Hminus(nH_I_div_partition, ne, T)

    # Stimulated emission correction
    nu = np.atleast_1d(nu)
    stimulated = 1 - np.exp(-hplanck_cgs * nu / (kboltz_cgs * T))

    # Absorption coefficient
    alpha = n_Hminus * cross_section * stimulated

    # Return scalar if input was scalar
    if alpha.size == 1:
        return float(alpha[0])
    return alpha


# JAX-compatible data for Hminus_ff (loaded at module level)
# Table from Bell & Berrington (1987)
# https://doi.org/10.1088/0022-3700/20/4/019

# Temperature parameter: θ = 5040/T
_HMINUS_FF_THETA_GRID = jnp.array([0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.8, 3.6])

# Wavelength grid in Å
_HMINUS_FF_LAMBDA_GRID = jnp.array([
    1823, 2278, 2604, 3038, 3645, 4557, 5063, 5696, 6510, 7595, 9113,
    10126, 11392, 13019, 15189, 18227, 22784, 30378, 45567, 91134,
    113918, 151890
])

# Absorption coefficient table (λ rows × θ columns)
# Units: 1e-26 cm^4/dyn (the 1e-26 factor is built into the table)
_HMINUS_FF_TABLE = jnp.array([
    [0.0178, 0.0222, 0.0308, 0.0402, 0.0498, 0.0596, 0.0695, 0.0795, 0.0896, 0.131, 0.172],
    [0.0228, 0.0280, 0.0388, 0.0499, 0.0614, 0.0732, 0.0851, 0.0972, 0.110, 0.160, 0.211],
    [0.0277, 0.0342, 0.0476, 0.0615, 0.0760, 0.0908, 0.105, 0.121, 0.136, 0.199, 0.262],
    [0.0364, 0.0447, 0.0616, 0.0789, 0.0966, 0.114, 0.132, 0.150, 0.169, 0.243, 0.318],
    [0.0520, 0.0633, 0.0859, 0.108, 0.131, 0.154, 0.178, 0.201, 0.225, 0.321, 0.418],
    [0.0791, 0.0959, 0.129, 0.161, 0.194, 0.227, 0.260, 0.293, 0.327, 0.463, 0.602],
    [0.0965, 0.117, 0.157, 0.195, 0.234, 0.272, 0.311, 0.351, 0.390, 0.549, 0.711],
    [0.121, 0.146, 0.195, 0.241, 0.288, 0.334, 0.381, 0.428, 0.475, 0.667, 0.861],
    [0.154, 0.188, 0.249, 0.309, 0.367, 0.424, 0.482, 0.539, 0.597, 0.830, 1.07],
    [0.208, 0.250, 0.332, 0.409, 0.484, 0.557, 0.630, 0.702, 0.774, 1.06, 1.36],
    [0.293, 0.354, 0.468, 0.576, 0.677, 0.777, 0.874, 0.969, 1.06, 1.45, 1.83],
    [0.358, 0.432, 0.572, 0.702, 0.825, 0.943, 1.06, 1.17, 1.28, 1.73, 2.17],
    [0.448, 0.539, 0.711, 0.871, 1.02, 1.16, 1.29, 1.43, 1.57, 2.09, 2.60],
    [0.579, 0.699, 0.924, 1.13, 1.33, 1.51, 1.69, 1.86, 2.02, 2.67, 3.31],
    [0.781, 0.940, 1.24, 1.52, 1.78, 2.02, 2.26, 2.48, 2.69, 3.52, 4.31],
    [1.11, 1.34, 1.77, 2.17, 2.53, 2.87, 3.20, 3.51, 3.80, 4.92, 5.97],
    [1.73, 2.08, 2.74, 3.37, 3.90, 4.50, 5.01, 5.50, 5.95, 7.59, 9.06],
    [3.04, 3.65, 4.80, 5.86, 6.86, 7.79, 8.67, 9.50, 10.3, 13.2, 15.6],
    [6.79, 8.16, 10.7, 13.1, 15.3, 17.4, 19.4, 21.2, 23.0, 29.5, 35.0],
    [27.0, 32.4, 42.6, 51.9, 60.7, 68.9, 76.8, 84.2, 91.4, 117.0, 140.0],
    [42.3, 50.6, 66.4, 80.8, 94.5, 107.0, 120.0, 131.0, 142.0, 183.0, 219.0],
    [75.1, 90.0, 118.0, 144.0, 168.0, 191.0, 212.0, 234.0, 253.0, 325.0, 388.0]
])


def _bilinear_interp_jax(table, x_grid, y_grid, x, y):
    """
    JAX-compatible bilinear interpolation on a regular grid.

    Parameters
    ----------
    table : jnp.ndarray
        2D table of values with shape (len(x_grid), len(y_grid))
    x_grid : jnp.ndarray
        1D array of x coordinates
    y_grid : jnp.ndarray
        1D array of y coordinates
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


def Hminus_ff(nu, T, nH_I_div_partition, ne):
    """
    Compute H⁻ free-free linear absorption coefficient.

    The naming scheme for free-free absorption is counter-intuitive. This
    actually refers to the reaction: photon + e⁻ + H I → e⁻ + H I.

    Parameters
    ----------
    nu : float or array_like
        Frequency in Hz (must be sorted if array)
    T : float
        Temperature in K
    nH_I_div_partition : float
        Total number density of H I divided by its partition function (cm⁻³)
    ne : float
        Electron number density (cm⁻³)

    Returns
    -------
    alpha : float or array_like
        Linear absorption coefficient in cm⁻¹

    Notes
    -----
    This is based on Table 1 in Bell & Berrington (1987), which tabulates
    the H⁻ absorption coefficient K (including stimulated emission correction).
    K has units of cm^4/dyn and must be multiplied by:
        - Electron pressure: P_e = n_e × k_B × T
        - Ground-state H I number density: n(H I, n=1) ≈ 2 × n(H I) / U(T)

    The stipulation that hydrogen should be ground-state only is based on
    Section 2 in Bell & Berrington (1987) or Section 5.3 from Kurucz (1970).

    Valid ranges:
        - Wavelength: 1823-151890 Å (1.97e15 - 1.64e13 Hz)
        - Temperature: 1400-10080 K (θ = 5040/T ∈ [0.5, 3.6])

    H⁻ free-free is the dominant opacity source at λ > 15000 Å in cool stars.

    References
    ----------
    Bell & Berrington (1987): https://doi.org/10.1088/0022-3700/20/4/019
    """
    # Convert frequency to wavelength in Å
    lambda_angstrom = c_cgs * 1e8 / nu

    # Temperature parameter
    theta = 5040.0 / T

    # Clip theta to valid range [0.5, 3.6] (corresponding to T ∈ [1400, 10080] K)
    # At T > 10080 K (theta < 0.5), H⁻ is essentially destroyed, so extrapolating
    # with the edge value is physically reasonable
    theta = jnp.clip(theta, 0.5, 3.6)

    # Interpolate K (in units of 1e-26 cm^4/dyn, factor built into table)
    K = 1e-26 * _bilinear_interp_jax(_HMINUS_FF_TABLE, _HMINUS_FF_LAMBDA_GRID,
                                      _HMINUS_FF_THETA_GRID, lambda_angstrom, theta)

    # Electron pressure in dyn/cm²
    P_e = ne * kboltz_cgs * T

    # Ground-state H I number density
    # Boltzmann factor is 1, degeneracy is 2 for ground state
    # For temperatures where this approximation is valid, less than 0.23%
    # of H I atoms are not in the ground state
    nHI_groundstate = 2 * nH_I_div_partition

    # Absorption coefficient
    alpha = K * P_e * nHI_groundstate

    return alpha
    """
    JAX-compatible bilinear interpolation on a regular grid.

    Parameters
    ----------
    table : jnp.ndarray
        2D table of values with shape (len(x_grid), len(y_grid))
    x_grid : jnp.ndarray
        1D array of x coordinates
    y_grid : jnp.ndarray
        1D array of y coordinates
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


# JAX arrays for Hminus_ff (converted from numpy)
_JAX_THETA_GRID = jnp.array([0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.8, 3.6])

_JAX_LAMBDA_GRID = jnp.array([
    1823, 2278, 2604, 3038, 3645, 4557, 5063, 5696, 6510, 7595, 9113,
    10126, 11392, 13019, 15189, 18227, 22784, 30378, 45567, 91134,
    113918, 151890
])

_JAX_FF_TABLE = jnp.array([
    [0.0178, 0.0222, 0.0308, 0.0402, 0.0498, 0.0596, 0.0695, 0.0795, 0.0896, 0.131, 0.172],
    [0.0228, 0.0280, 0.0388, 0.0499, 0.0614, 0.0732, 0.0851, 0.0972, 0.110, 0.160, 0.211],
    [0.0277, 0.0342, 0.0476, 0.0615, 0.0760, 0.0908, 0.105, 0.121, 0.136, 0.199, 0.262],
    [0.0364, 0.0447, 0.0616, 0.0789, 0.0966, 0.114, 0.132, 0.150, 0.169, 0.243, 0.318],
    [0.0520, 0.0633, 0.0859, 0.108, 0.131, 0.154, 0.178, 0.201, 0.225, 0.321, 0.418],
    [0.0791, 0.0959, 0.129, 0.161, 0.194, 0.227, 0.260, 0.293, 0.327, 0.463, 0.602],
    [0.0965, 0.117, 0.157, 0.195, 0.234, 0.272, 0.311, 0.351, 0.390, 0.549, 0.711],
    [0.121, 0.146, 0.195, 0.241, 0.288, 0.334, 0.381, 0.428, 0.475, 0.667, 0.861],
    [0.154, 0.188, 0.249, 0.309, 0.367, 0.424, 0.482, 0.539, 0.597, 0.830, 1.07],
    [0.208, 0.250, 0.332, 0.409, 0.484, 0.557, 0.630, 0.702, 0.774, 1.06, 1.36],
    [0.293, 0.354, 0.468, 0.576, 0.677, 0.777, 0.874, 0.969, 1.06, 1.45, 1.83],
    [0.358, 0.432, 0.572, 0.702, 0.825, 0.943, 1.06, 1.17, 1.28, 1.73, 2.17],
    [0.448, 0.539, 0.711, 0.871, 1.02, 1.16, 1.29, 1.43, 1.57, 2.09, 2.60],
    [0.579, 0.699, 0.924, 1.13, 1.33, 1.51, 1.69, 1.86, 2.02, 2.67, 3.31],
    [0.781, 0.940, 1.24, 1.52, 1.78, 2.02, 2.26, 2.48, 2.69, 3.52, 4.31],
    [1.11, 1.34, 1.77, 2.17, 2.53, 2.87, 3.20, 3.51, 3.80, 4.92, 5.97],
    [1.73, 2.08, 2.74, 3.37, 3.90, 4.50, 5.01, 5.50, 5.95, 7.59, 9.06],
    [3.04, 3.65, 4.80, 5.86, 6.86, 7.79, 8.67, 9.50, 10.3, 13.2, 15.6],
    [6.79, 8.16, 10.7, 13.1, 15.3, 17.4, 19.4, 21.2, 23.0, 29.5, 35.0],
    [27.0, 32.4, 42.6, 51.9, 60.7, 68.9, 76.8, 84.2, 91.4, 117.0, 140.0],
    [42.3, 50.6, 66.4, 80.8, 94.5, 107.0, 120.0, 131.0, 142.0, 183.0, 219.0],
    [75.1, 90.0, 118.0, 144.0, 168.0, 191.0, 212.0, 234.0, 253.0, 325.0, 388.0]
])


def Hminus_ff_jax(nu, T, nH_I_div_partition, ne):
    """
    JIT-compatible H⁻ free-free absorption using JAX.

    Parameters
    ----------
    nu : float or array_like
        Frequency in Hz
    T : float
        Temperature in K
    nH_I_div_partition : float
        Total number density of H I divided by its partition function (cm⁻³)
    ne : float
        Electron number density (cm⁻³)

    Returns
    -------
    alpha : float or array_like
        Linear absorption coefficient in cm⁻¹

    Notes
    -----
    This is the JAX-compatible version of Hminus_ff using bilinear interpolation.
    The original Hminus_ff uses scipy's RegularGridInterpolator which is not
    JIT-compatible.

    See Hminus_ff for more details on the physics and references.
    """
    # Convert frequency to wavelength in Å
    lambda_angstrom = c_cgs * 1e8 / nu

    # Temperature parameter
    theta = 5040.0 / T

    # Clip theta to valid range [0.5, 3.6]
    theta = jnp.clip(theta, 0.5, 3.6)

    # Interpolate K (in units of 1e-26 cm^4/dyn, factor built into table)
    K = 1e-26 * _bilinear_interp_jax(_JAX_FF_TABLE, _JAX_LAMBDA_GRID, _JAX_THETA_GRID,
                                      lambda_angstrom, theta)

    # Electron pressure in dyn/cm²
    P_e = ne * kboltz_cgs * T

    # Ground-state H I number density
    nHI_groundstate = 2 * nH_I_div_partition

    # Absorption coefficient
    alpha = K * P_e * nHI_groundstate

    return alpha
