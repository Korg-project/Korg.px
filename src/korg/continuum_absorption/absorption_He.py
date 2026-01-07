"""
Helium continuum absorption.

According to Gray (2005), the bound-free contributions from He⁻ are usually assumed to be
negligible because it only has one bound level with an ionization energy 19 eV. Supposedly
the population of that level is too small to be worth considering.

Currently implements:
- He⁻ free-free absorption

Missing:
- He I free-free and bound-free contributions

Reference: Korg.jl ContinuumAbsorption/absorption_He.jl
"""

import numpy as np
import jax.numpy as jnp
from scipy.interpolate import RegularGridInterpolator

from ..constants import c_cgs, kboltz_cgs, kboltz_eV
from .bounds_checking import (
    Interval, closed_interval, lambda_to_nu_bound, bounds_checked_absorption
)


def ndens_state_He_I(n: int, nsdens_div_partition: float, T: float) -> float:
    """
    Compute the number density of atoms in different He I states.

    Taken from section 5.5 of Kurucz (1970).

    Parameters
    ----------
    n : int
        Principal quantum number (1, 2, 3, or 4)
    nsdens_div_partition : float
        Total number density of He I divided by its partition function
    T : float
        Temperature in K

    Returns
    -------
    float
        Number density in state n (cm^-3)
    """
    if n == 1:
        g_n, energy_level = 1.0, 0.0
    elif n == 2:
        g_n, energy_level = 3.0, 19.819
    elif n == 3:
        g_n, energy_level = 1.0, 20.615
    elif n == 4:
        g_n, energy_level = 9.0, 20.964
    else:
        raise ValueError(f"Unknown excited state properties for He I: n={n}")

    return nsdens_div_partition * g_n * jnp.exp(-energy_level / (kboltz_eV * T))


# OCR'd from John (1994) https://ui.adsabs.harvard.edu/abs/1994MNRAS.269..871J
# JAX arrays for bilinear interpolation
_THETA_FF_ABSORPTION = jnp.array([0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.8, 3.6])

_LAMBDA_FF_ABSORPTION = jnp.array([
    5063.0, 5695.0, 6509.0, 7594.0, 9113.0, 11391.0, 15188.0,
    18225.0, 22782.0, 30376.0, 36451.0, 45564.0, 60751.0, 91127.0, 113900.0, 151878.0
])

_FF_ABSORPTION_DATA = jnp.array([
    [0.033, 0.036, 0.043, 0.049, 0.055, 0.061, 0.066, 0.072, 0.078, 0.100, 0.121],
    [0.041, 0.045, 0.053, 0.061, 0.067, 0.074, 0.081, 0.087, 0.094, 0.120, 0.145],
    [0.053, 0.059, 0.069, 0.077, 0.086, 0.094, 0.102, 0.109, 0.117, 0.148, 0.178],
    [0.072, 0.079, 0.092, 0.103, 0.114, 0.124, 0.133, 0.143, 0.152, 0.190, 0.227],
    [0.102, 0.113, 0.131, 0.147, 0.160, 0.173, 0.186, 0.198, 0.210, 0.258, 0.305],
    [0.159, 0.176, 0.204, 0.227, 0.247, 0.266, 0.283, 0.300, 0.316, 0.380, 0.444],
    [0.282, 0.311, 0.360, 0.400, 0.435, 0.466, 0.495, 0.522, 0.547, 0.643, 0.737],
    [0.405, 0.447, 0.518, 0.576, 0.625, 0.670, 0.710, 0.747, 0.782, 0.910, 1.030],
    [0.632, 0.698, 0.808, 0.899, 0.977, 1.045, 1.108, 1.165, 1.218, 1.405, 1.574],
    [1.121, 1.239, 1.435, 1.597, 1.737, 1.860, 1.971, 2.073, 2.167, 2.490, 2.765],
    [1.614, 1.783, 2.065, 2.299, 2.502, 2.681, 2.842, 2.990, 3.126, 3.592, 3.979],
    [2.520, 2.784, 3.226, 3.593, 3.910, 4.193, 4.448, 4.681, 4.897, 5.632, 6.234],
    [4.479, 4.947, 5.733, 6.387, 6.955, 7.460, 7.918, 8.338, 8.728, 10.059, 11.147],
    [10.074, 11.128, 12.897, 14.372, 15.653, 16.798, 17.838, 18.795, 19.685, 22.747, 25.268],
    [15.739, 17.386, 20.151, 22.456, 24.461, 26.252, 27.882, 29.384, 30.782, 35.606, 39.598],
    [27.979, 30.907, 35.822, 39.921, 43.488, 46.678, 49.583, 52.262, 54.757, 63.395, 70.580]
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


def _Heminus_ff(nu: float, T: float, nHe_I_div_partition: float, ne: float) -> float:
    """
    Internal He⁻ free-free absorption calculation (JAX-compatible).

    Parameters
    ----------
    nu : float
        Frequency in Hz
    T : float
        Temperature in K
    nHe_I_div_partition : float
        Number density of He I divided by partition function
    ne : float
        Number density of free electrons (cm^-3)

    Returns
    -------
    float
        Absorption coefficient (cm^-1)
    """
    # Convert to wavelength in Å
    lam = c_cgs * 1.0e8 / nu  # Å
    theta = 5040.0 / T

    # Clip to valid ranges
    theta = jnp.clip(theta, 0.5, 3.6)

    # K includes contribution from stimulated emission
    K = 1e-26 * _bilinear_interp_jax(
        _FF_ABSORPTION_DATA, _LAMBDA_FF_ABSORPTION, _THETA_FF_ABSORPTION,
        lam, theta
    )  # [cm^4/dyn]

    # Partial pressure contributed by electrons
    Pe = ne * kboltz_cgs * T

    # Ground state number density of He I
    nHe_I_gs = ndens_state_He_I(1, nHe_I_div_partition, T)

    return K * nHe_I_gs * Pe


# Wavelength bounds from the interpolation table (in cm)
_LAMBDA_MIN_CM = 5.063e-5   # 5063 Å
_LAMBDA_MAX_CM = 1.518780e-3  # 15187.8 Å

# Temperature bounds: θ = 5040/T, θ ∈ [0.5, 3.6] => T ∈ [1400, 10080]
_TEMP_MIN = 1400.0
_TEMP_MAX = 10080.0


def Heminus_ff(nu, T: float, nHe_I_div_partition: float, ne: float):
    """
    Compute the He⁻ free-free opacity κ (JAX-compatible).

    The naming scheme for free-free absorption is counter-intuitive. This actually
    refers to the reaction: photon + e⁻ + He I -> e⁻ + He I.

    Parameters
    ----------
    nu : float or array
        Frequency in Hz
    T : float
        Temperature in K
    nHe_I_div_partition : float
        The total number density of He I divided by its partition function
    ne : float
        The number density of free electrons (cm^-3)

    Returns
    -------
    float or array
        Absorption coefficient (cm^-1)

    Notes
    -----
    This uses the tabulated values from
    John (1994) https://ui.adsabs.harvard.edu/abs/1994MNRAS.269..871J/abstract

    The quantity K is the same used by Bell and Berrington (1987).

    According to John (1994), improved calculations are unlikely to alter the
    tabulated data for λ > 10000 Å "by more than about 2%." The errors introduced
    by the approximations for 5063 Å ≤ λ ≤ 10000 Å "are expected to be well below 10%."

    Valid ranges:
    - Wavelength: 5063 Å to 151878 Å
    - Temperature: 1400 K to 10080 K (θ = 5040/T ∈ [0.5, 3.6])

    For JIT compatibility, bounds checking is done via clipping rather than raising errors.
    """
    return _Heminus_ff(nu, T, nHe_I_div_partition, ne)
