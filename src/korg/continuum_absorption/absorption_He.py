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


# The John (1994) table and the bilinear interpolator that read it used to live here,
# behind a private ``_Heminus_ff``.  The interpolator assumed a uniformly spaced grid
# and neither axis of that table is uniform, so the result was wrong by 1.96x at
# 8000 AA and 64.1x at 20000 AA against Korg.jl.  ``Heminus_ff`` below forwards to
# korg.continuum, which uses searchsorted.  The defective kernel is deleted rather
# than kept as documentation: nothing on the synthesis path called it, and leaving a
# wrong private function in the tree is a liability, not a record.


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

    .. warning::
       This used to interpolate a local copy of the John (1994) table with a
       bilinear interpolator that assumed a uniformly spaced grid, which neither
       axis of that table is.  The error grew with wavelength — 1.96× at 8000 Å,
       5.3× at 10000 Å and 64.1× at 20000 Å against Korg.jl at T = 6000 K.  It
       now forwards to :func:`korg.continuum.Heminus_ff`, which locates the cell
       with ``searchsorted`` and reproduces Korg.jl to a few ULP, and which
       applies Korg.jl's bounds (exactly 0 outside 5063–151878 Å and
       1400–10080 K) instead of clipping θ.
    """
    from ..continuum import Heminus_ff as _Heminus_ff_correct
    return _Heminus_ff_correct(nu, T, nHe_I_div_partition, ne)
