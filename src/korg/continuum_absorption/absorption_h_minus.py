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

from ..constants import (hplanck_cgs, hplanck_eV, kboltz_cgs, kboltz_eV, c_cgs,
                         electron_mass_cgs)


# H⁻ ionization energy from McLaughlin+ 2017
H_MINUS_ION_ENERGY_EV = 0.754204  # eV

# Module-level cache for Hminus_bf data (JAX arrays)
_HMINUS_BF_NU = None
_HMINUS_BF_SIGMA = None
_HMINUS_BF_MIN_NU = None


def _initialize_Hminus_bf_jax_data():
    """Initialize H⁻ bound-free data as JAX arrays."""
    global _HMINUS_BF_NU, _HMINUS_BF_SIGMA, _HMINUS_BF_MIN_NU

    if _HMINUS_BF_NU is not None:
        return

    try:
        nu, sigma = _load_Hminus_bf_data()
        _HMINUS_BF_NU = jnp.array(nu)
        _HMINUS_BF_SIGMA = jnp.array(sigma)
        _HMINUS_BF_MIN_NU = float(nu[0])
    except FileNotFoundError:
        # Data file not found - will be handled at runtime
        pass


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
        - coef = (h²·k_eV / 2π·m_e·k_cgs)^1.5, derived from the physical constants
        - β = 1/(k_B T) in eV^-1

    Warning: For JIT compatibility, temperature validation is removed.
    Users should ensure T > 1000 K for physical results.
    """
    # Ground state H I number density: Boltzmann factor = 1, degeneracy = 2
    nHI_groundstate = 2 * nH_I_div_partition

    # Coefficient (h²·k_eV / 2π·m_e·k_cgs)^1.5, in cm³·eV^1.5. Korg.jl v1.1 carried this
    # as the literal 3.31283018e-22, which is high by 9.2e-7 relative; v1.2 dropped the
    # literal and evaluates the equivalent 1/translational_U(m_e, T) instead. Derive it
    # from the constants so the two agree exactly rather than to ~1e-6.
    coef = (hplanck_cgs ** 2 * kboltz_eV
            / (2 * jnp.pi * electron_mass_cgs * kboltz_cgs)) ** 1.5

    # Inverse temperature in eV
    beta = 1.0 / (kboltz_eV * T)

    return 0.25 * nHI_groundstate * ne * coef * beta**1.5 * jnp.exp(ion_energy * beta)


def _Hminus_bf_cross_section(nu):
    """
    Get H⁻ bound-free cross-section at given frequency (JAX-compatible).

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

    This version is JAX-compatible using jnp.interp and jnp.where.
    """
    # Initialize data if needed
    _initialize_Hminus_bf_jax_data()

    if _HMINUS_BF_NU is None:
        raise FileNotFoundError("H⁻ bound-free data file not found")

    # Ionization frequency
    nu_ion = H_MINUS_ION_ENERGY_EV / hplanck_eV

    # Below ionization threshold: σ = 0
    # Between nu_ion and min_nu: power-law extrapolation
    # Above min_nu: linear interpolation

    # Get scaling coefficient from first tabulated point
    sigma_min = _HMINUS_BF_SIGMA[0]
    coef = sigma_min / (_HMINUS_BF_MIN_NU - nu_ion)**1.5

    # Calculate sigma for all three regions using jnp.where
    # Region 1: nu <= nu_ion → sigma = 0
    # Region 2: nu_ion < nu < min_nu → power-law: coef * (nu - nu_ion)^1.5
    # Region 3: nu >= min_nu → interpolate from table

    # Strictly positive stand-in below the detachment threshold: a negative base to
    # the power 1.5 is NaN, and the jnp.where below masks the value but not the
    # cotangent.  See korg.continuum.Hminus_bf_cross_section for the full note.
    sigma_powerlaw = coef * jnp.where(nu > nu_ion, nu - nu_ion, 1.0)**1.5
    sigma_interp = jnp.interp(nu, _HMINUS_BF_NU, _HMINUS_BF_SIGMA)

    # Use jnp.where for conditional logic (JAX-compatible)
    sigma = jnp.where(
        nu <= nu_ion,
        0.0,
        jnp.where(
            nu < _HMINUS_BF_MIN_NU,
            sigma_powerlaw,
            sigma_interp
        )
    )

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
    stimulated = 1 - jnp.exp(-hplanck_cgs * nu / (kboltz_cgs * T))

    # Absorption coefficient
    alpha = n_Hminus * cross_section * stimulated

    return alpha


# The Bell & Berrington table and the bilinear interpolator that read it used to live
# here.  The interpolator derived its cell index from (x - x_grid[0]) / (x_grid[1] -
# x_grid[0]), which is only correct on a uniformly spaced grid, and neither axis of
# this table is uniform -- lambda runs 1823, 2278, 2604, 3038 ... and theta runs 0.5,
# 0.6, 0.8, 1.0 ....  The error grew with wavelength, reaching 68x at 20000 AA against
# Korg.jl.  Both entry points below now forward to korg.continuum, which locates the
# cell with searchsorted; the defective copies are deleted rather than kept for
# reference, since nothing called them and a wrong private function is reachable by
# the next person who goes looking for one.


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

    .. warning::
       This used to interpolate a local copy of the Bell & Berrington table with a
       bilinear interpolator that assumed a uniformly spaced grid, which neither
       axis of that table is.  The error grew with wavelength — 1.23× at 3000 Å,
       3.76× at 8000 Å and 68.4× at 20000 Å against Korg.jl at T = 6000 K.  It
       now forwards to :func:`korg.continuum.Hminus_ff`, which locates the cell
       with ``searchsorted`` and reproduces Korg.jl to a few ULP.  That also
       brings the Korg.jl bounds behaviour (exactly 0 outside 1823–151890 Å and
       1400–10080 K) instead of the previous θ clipping.
    """
    from ..continuum import Hminus_ff as _Hminus_ff_correct
    return _Hminus_ff_correct(nu, T, nH_I_div_partition, ne)


# ``Hminus_ff_jax`` below is an exact alias of ``Hminus_ff``.  It used to carry its own
# byte-identical copy of the Bell & Berrington grids plus a second copy of the bilinear
# interpolator stranded after ``Hminus_ff``'s ``return``, referencing names not in its
# scope — it would have raised ``NameError`` had anything reached it.  The two entry
# points now share one implementation.


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
    Deprecated alias for :func:`Hminus_ff`, kept for backwards compatibility.
    ``Hminus_ff`` is already written in JAX and is JIT-compatible; this used to be
    a byte-for-byte copy of it operating on a duplicate set of Bell & Berrington
    grids.  It now simply forwards.

    See :func:`Hminus_ff` for the physics and references, and
    :func:`korg.continuum.Hminus_ff` for the bounds-checked version that matches
    Korg.jl's public API.
    """
    return Hminus_ff(nu, T, nH_I_div_partition, ne)
