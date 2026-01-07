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

    return nsdens_div_partition * g_n * np.exp(-energy_level / (kboltz_eV * T))


# OCR'd from John (1994) https://ui.adsabs.harvard.edu/abs/1994MNRAS.269..871J
_THETA_FF_ABSORPTION = np.array([0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.8, 3.6])

_LAMBDA_FF_ABSORPTION = 1e4 * np.array([
    0.5063, 0.5695, 0.6509, 0.7594, 0.9113, 1.1391, 1.5188,
    1.8225, 2.2782, 3.0376, 3.6451, 4.5564, 6.0751, 9.1127, 11.390, 15.1878
])

_FF_ABSORPTION_DATA = np.array([
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

# Create interpolator (lambda, theta) -> K value
_Heminus_ff_absorption_interp = RegularGridInterpolator(
    (_LAMBDA_FF_ABSORPTION, _THETA_FF_ABSORPTION),
    _FF_ABSORPTION_DATA,
    method='linear',
    bounds_error=True
)


def _Heminus_ff(nu: float, T: float, nHe_I_div_partition: float, ne: float) -> float:
    """
    Internal He⁻ free-free absorption calculation.

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

    # K includes contribution from stimulated emission
    K = 1e-26 * _Heminus_ff_absorption_interp([[lam, theta]])[0]  # [cm^4/dyn]

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


def Heminus_ff(
    nus: np.ndarray,
    T: float,
    nHe_I_div_partition: float,
    ne: float,
    error_oobounds: bool = False,
    out_alpha: np.ndarray = None
) -> np.ndarray:
    """
    Compute the He⁻ free-free opacity κ.

    The naming scheme for free-free absorption is counter-intuitive. This actually
    refers to the reaction: photon + e⁻ + He I -> e⁻ + He I.

    Parameters
    ----------
    nus : array
        Sorted frequency vector in Hz
    T : float
        Temperature in K
    nHe_I_div_partition : float
        The total number density of He I divided by its partition function
    ne : float
        The number density of free electrons (cm^-3)
    error_oobounds : bool, optional
        If True, raise error for out-of-bounds values.
        If False, return 0 for out-of-bounds (default).
    out_alpha : array, optional
        Output array to add results to (in-place operation)

    Returns
    -------
    array
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
    - Wavelength: 5063 Å to 15187.8 Å
    - Temperature: 1400 K to 10080 K
    """
    # Create bounds-checked wrapper
    nu_bound = lambda_to_nu_bound(closed_interval(_LAMBDA_MIN_CM, _LAMBDA_MAX_CM))
    temp_bound = closed_interval(_TEMP_MIN, _TEMP_MAX)

    wrapped = bounds_checked_absorption(
        _Heminus_ff,
        nu_bound=nu_bound,
        temp_bound=temp_bound
    )

    return wrapped(nus, T, nHe_I_div_partition, ne,
                   error_oobounds=error_oobounds, out_alpha=out_alpha)
