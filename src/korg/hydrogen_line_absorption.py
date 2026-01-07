"""
Hydrogen line absorption with Stark broadening.

This module implements hydrogen line absorption including:
- Brackett series Stark profiles
- Holtsmark broadening for quasistatic charged particles
- MHD occupation probabilities
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional, Dict, List
from jax.scipy.signal import convolve

from .constants import (
    c_cgs, kboltz_cgs, kboltz_eV, hplanck_cgs, hplanck_eV,
    RydbergH_eV, bohr_radius_cgs, amu_cgs, electron_charge_cgs,
    eV_to_cgs
)
from .line_absorption import sigma_line, doppler_width, scaled_vdW
from .atomic_data import atomic_masses
from .hydrogen_stark_data import hline_stark_profiles


def _interp_linear_2d_jax(x: float, y: float,
                          x_grid: jnp.ndarray, y_grid: jnp.ndarray,
                          values: jnp.ndarray) -> float:
    """
    JAX-compatible 2D linear interpolation.

    Args:
        x, y: Query points
        x_grid: 1D array of x grid points (sorted)
        y_grid: 1D array of y grid points (sorted)
        values: 2D array of shape (len(x_grid), len(y_grid))

    Returns:
        Interpolated value at (x, y)
    """
    # Find indices
    ix = jnp.searchsorted(x_grid, x) - 1
    iy = jnp.searchsorted(y_grid, y) - 1

    # Clip to valid range
    ix = jnp.clip(ix, 0, len(x_grid) - 2)
    iy = jnp.clip(iy, 0, len(y_grid) - 2)

    # Get surrounding grid points
    x0, x1 = x_grid[ix], x_grid[ix + 1]
    y0, y1 = y_grid[iy], y_grid[iy + 1]

    # Compute weights
    wx = (x - x0) / (x1 - x0)
    wy = (y - y0) / (y1 - y0)

    # Bilinear interpolation
    v00 = values[ix, iy]
    v01 = values[ix, iy + 1]
    v10 = values[ix + 1, iy]
    v11 = values[ix + 1, iy + 1]

    result = ((1 - wx) * (1 - wy) * v00 +
              (1 - wx) * wy * v01 +
              wx * (1 - wy) * v10 +
              wx * wy * v11)

    return result


def _interp_linear_3d_jax(x: float, y: float, z: float,
                          x_grid: jnp.ndarray, y_grid: jnp.ndarray, z_grid: jnp.ndarray,
                          values: jnp.ndarray) -> float:
    """
    JAX-compatible 3D linear interpolation.

    Args:
        x, y, z: Query points
        x_grid: 1D array of x grid points (sorted)
        y_grid: 1D array of y grid points (sorted)
        z_grid: 1D array of z grid points (sorted)
        values: 3D array of shape (len(x_grid), len(y_grid), len(z_grid))

    Returns:
        Interpolated value at (x, y, z)
    """
    # Find indices
    ix = jnp.searchsorted(x_grid, x) - 1
    iy = jnp.searchsorted(y_grid, y) - 1
    iz = jnp.searchsorted(z_grid, z) - 1

    # Clip to valid range
    ix = jnp.clip(ix, 0, len(x_grid) - 2)
    iy = jnp.clip(iy, 0, len(y_grid) - 2)
    iz = jnp.clip(iz, 0, len(z_grid) - 2)

    # Get surrounding grid points
    x0, x1 = x_grid[ix], x_grid[ix + 1]
    y0, y1 = y_grid[iy], y_grid[iy + 1]
    z0, z1 = z_grid[iz], z_grid[iz + 1]

    # Compute weights
    wx = (x - x0) / (x1 - x0)
    wy = (y - y0) / (y1 - y0)
    wz = (z - z0) / (z1 - z0)

    # Trilinear interpolation
    v000 = values[ix, iy, iz]
    v001 = values[ix, iy, iz + 1]
    v010 = values[ix, iy + 1, iz]
    v011 = values[ix, iy + 1, iz + 1]
    v100 = values[ix + 1, iy, iz]
    v101 = values[ix + 1, iy, iz + 1]
    v110 = values[ix + 1, iy + 1, iz]
    v111 = values[ix + 1, iy + 1, iz + 1]

    result = ((1 - wx) * (1 - wy) * (1 - wz) * v000 +
              (1 - wx) * (1 - wy) * wz * v001 +
              (1 - wx) * wy * (1 - wz) * v010 +
              (1 - wx) * wy * wz * v011 +
              wx * (1 - wy) * (1 - wz) * v100 +
              wx * (1 - wy) * wz * v101 +
              wx * wy * (1 - wz) * v110 +
              wx * wy * wz * v111)

    return result


def normal_pdf(delta: float, sigma: float) -> float:
    """
    Compute the Gaussian probability density function.

    Args:
        delta: Distance from mean
        sigma: Standard deviation

    Returns:
        PDF value at delta
    """
    return jnp.exp(-0.5 * delta**2 / sigma**2) / jnp.sqrt(2 * jnp.pi) / sigma


def autodiffable_conv(f: jnp.ndarray, g: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the convolution of two vectors.

    This is a simple wrapper around jax.scipy.signal.convolve that uses 'full' mode
    (equivalent to Julia's DSP.conv).

    Args:
        f: First array
        g: Second array

    Returns:
        Convolution of f and g
    """
    return convolve(f, g, mode='full')


def exponential_integral_1(x: float) -> float:
    """
    Compute the first exponential integral, E1(x).

    This is a rough approximation lifted from Kurucz's VCSE1F.
    Used in brackett_line_stark_profiles.

    JAX-compatible version using jnp.where instead of if/elif/else.

    Args:
        x: Input value

    Returns:
        E1(x) approximation
    """
    # Compute all branches
    branch_neg = 0.0
    branch_tiny = -jnp.log(jnp.maximum(x, 1e-10)) - 0.577215 + x  # Avoid log(0)
    branch_small = (-jnp.log(x) - 0.57721566 +
                    x * (0.99999193 + x * (-0.24991055 +
                         x * (0.05519968 + x * (-0.00976004 + x * 0.00107857)))))
    branch_mid = ((x * (x + 2.334733) + 0.25062) /
                  (x * (x + 3.330657) + 1.681534) / x * jnp.exp(-x))
    branch_large = 0.0

    # Use jnp.where to select the correct branch
    result = jnp.where(x < 0, branch_neg,
             jnp.where(x <= 0.01, branch_tiny,
             jnp.where(x <= 1.0, branch_small,
             jnp.where(x <= 30.0, branch_mid, branch_large))))

    return result


def brackett_oscillator_strength(n: int, m: int) -> float:
    """
    Oscillator strength of the transition from the 4th to the mth energy level of hydrogen.

    Adapted from HLINOP.f by Peterson and Kurucz.
    Comparison to the values in Goldwire 1968 indicates that this is
    accurate to 10^-4 for the Brackett series.

    Args:
        n: Lower level quantum number (4 for Brackett)
        m: Upper level quantum number (> n)

    Returns:
        Oscillator strength

    Note:
        JAX-compatible version - no assert statement.
        Behavior is undefined if n >= m.
    """
    GINF = 0.2027 / n**0.71
    GCA = 0.124 / n
    FKN = 1.9603 * n
    WTC = 0.45 - 2.4 / n**3 * (n - 1)
    FK = FKN * (m / ((m - n) * (m + n)))**3
    XMN12 = (m - n)**1.2
    WT = (XMN12 - 1) / (XMN12 + WTC)

    return FK * (1 - WT * GINF - (0.222 + GCA / m) * (1 - WT))


# Griem 1960 Knm constants for Brackett lines
_GREIM_KMN_TABLE = jnp.array([
    [0.0001716, 0.0090190, 0.1001000, 0.5820000],
    [0.0005235, 0.0177200, 0.1710000, 0.8660000],
    [0.0008912, 0.0250700, 0.2230000, 1.0200000]
])


def greim_1960_Knm(n: int, m: int) -> float:
    """
    Knm constants as defined by Griem 1960 for the long range Holtsmark profile.

    This function includes only the values for Brackett lines.
    K_nm = C_nm 2π c / λ² where C_nm F = Δω and F is the ion field.
    See Griem 1960 EQs 7 and 12. This works out to K_nm = λ/F.

    JAX-compatible version using jnp.where.

    Args:
        n: Lower level quantum number
        m: Upper level quantum number

    Returns:
        Knm constant
    """
    # Table lookup value (for m-n <= 3 and n <= 4)
    # Julia is 1-indexed, Python is 0-indexed
    table_value = _GREIM_KMN_TABLE[jnp.minimum(m - n - 1, 2), jnp.minimum(n - 1, 3)]

    # Analytical formula (Griem 1960 equation 33)
    # 1 / (1 + 0.13/(m-n)) is probably a Kurucz addition.
    analytical_value = 5.5e-5 * n**4 * m**4 / (m**2 - n**2) / (1 + 0.13 / (m - n))

    # Use table if (m - n <= 3) and (n <= 4), otherwise use formula
    use_table = (m - n <= 3) & (n <= 4)
    return jnp.where(use_table, table_value, analytical_value)


# Holtsmark profile constants
_HOLTSMARK_PROB7 = jnp.array([
    [0.005, 0.128, 0.260, 0.389, 0.504],
    [0.004, 0.109, 0.220, 0.318, 0.389],
    [-0.007, 0.079, 0.162, 0.222, 0.244],
    [-0.018, 0.041, 0.089, 0.106, 0.080],
    [-0.026, -0.003, 0.003, -0.023, -0.086],
    [-0.025, -0.048, -0.087, -0.148, -0.234],
    [-0.008, -0.085, -0.165, -0.251, -0.343],
    [0.018, -0.111, -0.223, -0.321, -0.407],
    [0.032, -0.130, -0.255, -0.354, -0.431],
    [0.014, -0.148, -0.269, -0.359, -0.427],
    [-0.005, -0.140, -0.243, -0.323, -0.386],
    [0.005, -0.095, -0.178, -0.248, -0.307],
    [-0.002, -0.068, -0.129, -0.187, -0.241],
    [-0.007, -0.049, -0.094, -0.139, -0.186],
    [-0.010, -0.036, -0.067, -0.103, -0.143]
])

_HOLTSMARK_C7 = jnp.array([511.318, 1.532, 4.044, 19.266, 41.812])
_HOLTSMARK_D7 = jnp.array([-6.070, -4.528, -8.759, -14.984, -23.956])
_HOLTSMARK_PP = jnp.array([0.0, 0.2, 0.4, 0.6, 0.8])
_HOLTSMARK_BETA_KNOTS = jnp.array([
    1.0, 1.259, 1.585, 1.995, 2.512, 3.162, 3.981,
    5.012, 6.310, 7.943, 10.0, 12.59, 15.85, 19.95, 25.12
])


def holtsmark_profile(beta: float, P: float) -> float:
    """
    Calculates the Holtsmark profile for broadening of hydrogen lines by quasistatic charged particles.

    Adapted from SOFBET in HLINOP by Peterson and Kurucz.
    Draws heavily from Griem 1960.

    JAX-compatible version using jnp.where instead of if/else.

    Args:
        beta: Scaled frequency detuning
        P: Shielding parameter

    Returns:
        Holtsmark profile value
    """
    # Very large β result
    large_beta_result = (1.5 / jnp.sqrt(beta) + 27 / beta**2) / beta**2

    # Determine relevant Debye range
    # Julia: IM = min(Int(floor((5 * P) + 1)), 4) with 1-indexed arrays
    # Python needs 0-indexed conversion
    IM = jnp.clip(jnp.floor(5 * P + 1).astype(jnp.int32) - 1, 0, 3)  # 0-indexed, clipped to [0,3]
    IP = IM + 1
    WTPP = 5 * (P - _HOLTSMARK_PP[IM])
    WTPM = 1 - WTPP

    # === Branch: beta <= 25.12 ===
    # Indices into β_boundaries which bound the value of β
    mask = beta <= _HOLTSMARK_BETA_KNOTS
    # Use jnp.where to avoid if: JP = max(1, argmax(mask)) if any(mask) else len-1
    JP = jnp.where(jnp.any(mask),
                   jnp.maximum(1, jnp.argmax(mask)),
                   len(_HOLTSMARK_BETA_KNOTS) - 1)
    JM = JP - 1

    # Linear interpolation into PROB7 wrt β_knots
    WTBP = ((beta - _HOLTSMARK_BETA_KNOTS[JM]) /
            (_HOLTSMARK_BETA_KNOTS[JP] - _HOLTSMARK_BETA_KNOTS[JM]))
    WTBM = 1 - WTBP
    CBP = _HOLTSMARK_PROB7[JP, IP] * WTPP + _HOLTSMARK_PROB7[JP, IM] * WTPM
    CBM = _HOLTSMARK_PROB7[JM, IP] * WTPP + _HOLTSMARK_PROB7[JM, IM] * WTPM
    CORR_small = 1 + CBP * WTBP + CBM * WTBM

    # Get approximate profile for the inner part
    WT = jnp.clip(0.5 * (10 - beta), 0, 1)

    # PR1: beta <= 10 ? value : 0.0
    PR1 = jnp.where(beta <= 10,
                    8 / (83 + (2 + 0.95 * beta**2) * beta),
                    0.0)

    # PR2: beta >= 8 ? value : 0.0
    PR2 = jnp.where(beta >= 8,
                    (1.5 / jnp.sqrt(beta) + 27 / beta**2) / beta**2,
                    0.0)

    small_beta_result = (PR1 * WT + PR2 * (1 - WT)) * CORR_small

    # === Branch: 25.12 < beta <= 500 (medium) ===
    # Asymptotic part for medium β's
    CC = _HOLTSMARK_C7[IP] * WTPP + _HOLTSMARK_C7[IM] * WTPM
    DD = _HOLTSMARK_D7[IP] * WTPP + _HOLTSMARK_D7[IM] * WTPM
    CORR_medium = 1 + DD / (CC + beta * jnp.sqrt(beta))
    medium_beta_result = (1.5 / jnp.sqrt(beta) + 27 / beta**2) / beta**2 * CORR_medium

    # Select correct branch using nested jnp.where
    # if beta > 500: large_beta_result
    # elif beta <= 25.12: small_beta_result
    # else: medium_beta_result
    result = jnp.where(beta > 500, large_beta_result,
             jnp.where(beta <= 25.12, small_beta_result, medium_beta_result))

    return result


def hummer_mihalas_w(T: float, n_eff: float, nH: float, nHe: float, ne: float,
                     use_hubeny_generalization: bool = False) -> float:
    """
    Calculate the correction to the occupation fraction of a hydrogen energy level.

    Uses the occupation probability formalism from Hummer and Mihalas 1988,
    optionally with the generalization by Hubeny+ 1994.

    The expression for w is in equation 4.71 of H&M. K, the QM correction,
    is defined in equation 4.24.

    JAX-compatible version using jnp.where instead of if/else.

    Args:
        T: Temperature in K
        n_eff: Effective principal quantum number
        nH: Neutral hydrogen number density in cm^-3
        nHe: Neutral helium number density in cm^-3
        ne: Electron number density in cm^-3
        use_hubeny_generalization: Use Hubeny+ 1994 generalization (default: False)

    Returns:
        Occupation probability correction factor w
    """
    # Contribution from neutral species (neutral H and He)
    # This is sqrt<r^2> assuming l=0
    r_level = jnp.sqrt(5 / 2 * n_eff**4 + 1 / 2 * n_eff**2) * bohr_radius_cgs
    neutral_term = (nH * (r_level + jnp.sqrt(3) * bohr_radius_cgs)**3 +
                    nHe * (r_level + 1.02 * bohr_radius_cgs)**3)

    # Contributions from ions (assumed to be all singly ionized, so n_ion = n_e)
    # K is a QM correction defined in H&M '88 equation 4.24
    # Use jnp.where instead of if n_eff > 3
    K_large = (16 / 3 * (n_eff / (n_eff + 1))**2 *
               ((n_eff + 7 / 6) / (n_eff**2 + n_eff + 1 / 2)))
    K = jnp.where(n_eff > 3, K_large, 1.0)

    chi = RydbergH_eV / n_eff**2 * eV_to_cgs  # binding energy
    e = electron_charge_cgs

    # Use jnp.where for use_hubeny_generalization branch
    # Hubeny generalization
    # Inner condition: (ne > 10) and (T > 10)
    A = 0.09 * jnp.exp(0.16667 * jnp.log(ne)) / jnp.sqrt(T)
    X = jnp.exp(3.15 * jnp.log(1 + A))
    BETAC = 8.3e14 * jnp.exp(-0.66667 * jnp.log(ne)) * K / n_eff**4
    F = 0.1402 * X * BETAC**3 / (1 + 0.1285 * X * BETAC * jnp.sqrt(BETAC))
    hubeny_charged_term = jnp.log(F / (1 + F)) / (-4 * jnp.pi / 3)

    # Select between hubeny formula or 0.0 based on (ne > 10) and (T > 10)
    hubeny_charged_term = jnp.where((ne > 10) & (T > 10), hubeny_charged_term, 0.0)

    # Standard H&M charged term
    hm_charged_term = 16 * ((e**2) / (chi * jnp.sqrt(K)))**3 * ne

    # Select between Hubeny and H&M based on use_hubeny_generalization
    charged_term = jnp.where(use_hubeny_generalization, hubeny_charged_term, hm_charged_term)

    return jnp.exp(-4 * jnp.pi / 3 * (neutral_term + charged_term))


def brackett_line_stark_profiles(m: int, wavelengths: np.ndarray, wavelength_center: float,
                                  T: float, ne: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stark-broadened line profile specialized to Brackett series.

    Translated and heavily adapted from HLINOP.f by Barklem, who adapted it from
    Peterson and Kurucz. Mostly follows Griem 1960 and Griem 1967.

    Ions and distant electrons have E fields which can be treated quasi-statically,
    leading to Holtsmark broadening.

    JAX-compatible version using vectorized operations and jnp.where.

    Args:
        m: Upper level of the transition
        wavelengths: Wavelengths at which to calculate profile [cm]
        wavelength_center: Line center wavelength [cm]
        T: Temperature [K]
        ne: Electron number density [cm^-3]

    Returns:
        Tuple of (impact_electron_profile, total_quasistatic_profile)
    """
    n = 4  # Brackett lines only
    nus = c_cgs / wavelengths
    nu_0 = c_cgs / wavelength_center

    ne_1_6 = ne**(1 / 6)
    F0 = 1.25e-9 * ne**(2 / 3)  # the Holtsmark field
    # Note: Julia code has 10_0004 which appears to be a typo for 10_000, but we match it exactly
    GCON1 = 0.2 + 0.09 * jnp.sqrt(T / 100004) / (1 + ne / 1.e13)
    GCON2 = 0.2 / (1 + ne / 1.e15)

    Knm = greim_1960_Knm(n, m)

    # Use jnp.where for conditional assignment
    Y1WHT = jnp.where(m - n <= 3, 1e14, 1e13)
    WTY1 = 1 / (1 + ne / Y1WHT)
    Y1B = 2 / (1 + 0.012 / T * jnp.sqrt(ne / T))
    C1CON = Knm / wavelength_center * (m**2 - n**2)**2 / (n**2 * m**2) * 1e-8
    Y1NUM = 320  # specialized to n=4
    Y1SCAL = Y1NUM * ((T / 10_000)**0.3 / ne_1_6) * WTY1 + Y1B * (1 - WTY1)
    C1 = F0 * 78940 / T * C1CON * Y1SCAL

    C2 = F0**2 / (5.96e-23 * ne) * (Knm / wavelength_center)**2 * 1e-16

    # Griem 1960 eqn 23. This is the argument of the Holtsmark profile.
    betas = jnp.abs(wavelengths - wavelength_center) / F0 / Knm * 1e8

    # y1 and y2 from Griem 1967 EQ 5, 6, 7
    y1 = C1 * betas
    y2 = C2 * betas**2

    G1 = 6.77 * jnp.sqrt(C1)

    # Calculate impact electron profile (called F in Kurucz) - vectorized with jnp.where
    # Width of the electron impact profile (called GAM in Kurucz)
    # Branch 1: (y2 <= 1e-4) and (y1 <= 1e-5)
    width_simple = G1 * jnp.maximum(0, 0.2114 + jnp.log(jnp.sqrt(C2) / C1)) * (1 - GCON1 - GCON2)

    # Branch 2: else
    GAM = (G1 *
           (0.5 * jnp.exp(-jnp.minimum(80, y1)) + jax.vmap(exponential_integral_1)(y1) -
            0.5 * jax.vmap(exponential_integral_1)(y2)) *
           (1 - GCON1 / (1 + (90 * y1)**3) - GCON2 / (1 + 2000 * y1)))
    width_complex = jnp.where(GAM <= 1e-20, 0.0, GAM)

    # Select width based on condition
    width = jnp.where((y2 <= 1e-4) & (y1 <= 1e-5), width_simple, width_complex)

    # Calculate impact profile using jnp.where
    impact_electron_profile = jnp.where(
        width > 0,
        width / (jnp.pi * (width**2 + betas**2)),  # Lorentz density
        0.0
    )

    # Quasistatic ion contribution - vectorized
    shielding_parameter = ne_1_6 * 0.08989 / jnp.sqrt(T)  # Called PP in Kurucz
    quasistatic_ion_contribution = jax.vmap(lambda beta: holtsmark_profile(beta, shielding_parameter))(betas)

    # Quasistatic electron contribution
    # Fit to (sqrt(π) - 2*gamma(3/2, y1))/sqrt(π) from HLINOP/Kurucz
    # Second term in eqn 8 of Griem 1967
    ps = (0.9 * y1)**2
    quasistatic_e_contrib = (ps + 0.03 * jnp.sqrt(y1)) / (ps + 1.0)
    # Fix potential NaNs from 0/0
    quasistatic_e_contrib = jnp.where(jnp.isnan(quasistatic_e_contrib), 0.0, quasistatic_e_contrib)

    total_quasistatic_profile = quasistatic_ion_contribution * (1 + quasistatic_e_contrib)

    dβ_dλ = 1e8 / (Knm * F0)

    # Apply corrections to both profiles
    # sqrt(λ/λ₀) corrects the long range part to Δν^(5/2) asymptote
    # (see Stehle and Hutcheon 1999, A&AS 140, 93)
    impact_electron_profile = impact_electron_profile * jnp.sqrt(wavelengths / wavelength_center)
    total_quasistatic_profile = total_quasistatic_profile * jnp.sqrt(wavelengths / wavelength_center)

    # The red wing is multiplied by the Boltzmann factor to roughly account
    # for quantum effects (Stehle 1994, A&AS 104, 509 eqn 7)
    # Assume absorption case
    # Apply Boltzmann factor to red wing (where ν < ν₀) using masking
    red_wing_mask = nus < nu_0
    boltzmann_factor = jnp.exp((hplanck_cgs * (nus - nu_0)) / kboltz_cgs / T)
    boltzmann_factor = jnp.where(red_wing_mask, boltzmann_factor, 1.0)

    impact_electron_profile = impact_electron_profile * boltzmann_factor
    total_quasistatic_profile = total_quasistatic_profile * boltzmann_factor

    # Apply scaling by dβ/dλ
    impact_electron_profile = impact_electron_profile * dβ_dλ
    total_quasistatic_profile = total_quasistatic_profile * dβ_dλ

    return impact_electron_profile, total_quasistatic_profile


def bracket_line_interpolator(m: int, λ0: float, T: float, ne: float, xi: float,
                                λmin: float = 0.0, λmax: float = np.inf,
                                n_wavelength_points: int = 201, window_size: int = 5,
                                include_doppler_threshold: float = 0.25):
    """
    Numerically convolve Brackett line Stark profile components with Doppler profile.

    This routine numerically convolves the two components of the Brackett line Stark profile
    (quasistatic/Holtsmark and impact) and the Doppler profile, if necessary. It returns a tuple
    containing the interpolator function and the distance from the line center at which it is defined.

    Args:
        m: Principle quantum number of the upper level
        λ0: Line center [cm]
        T: Temperature [K]
        ne: Electron number density [cm^-3]
        xi: Microturbulence [cm/s]
        λmin: Minimum wavelength at which profile should be computed [cm] (default: 0.0)
        λmax: Maximum wavelength at which profile should be computed [cm] (default: inf)
        n_wavelength_points: Number of wavelengths at which to sample profiles (default: 201)
        window_size: Size of wavelength range over which profiles should be calculated,
                     in units of the characteristic profile width (default: 5)
        include_doppler_threshold: Threshold for including Doppler broadening (default: 0.25)

    Returns:
        Tuple of (interpolator_function, window)
        - interpolator_function: callable that takes wavelength and returns profile value
        - window: distance from line center at which profile is defined [cm]
    """
    n = 4  # Brackett lines only

    # Get Stark width
    F0 = 1.25e-9 * ne**(2 / 3)  # the Holtsmark field
    Knm = greim_1960_Knm(n, m)
    stark_width = 1.6678e-18 * Knm * F0 * c_cgs

    # Get Doppler width
    # atomic_masses[0] is hydrogen mass (0-indexed in Python, 1-indexed in Julia)
    H_mass = atomic_masses[0]
    σdop = doppler_width(λ0, T, H_mass, xi)

    # Set wavelengths for calculations and convolutions
    window = window_size * max(σdop, stark_width)
    λstart = max(λmin, λ0 - window)
    λend = min(λ0 + window, λmax)

    # Handle edge cases
    if λstart > λmax or λend < λmin or λstart == λend:
        # Return a noop interpolator and null window
        def noop_interp(x):
            return 0.0
        return noop_interp, 0.0

    wls = np.linspace(λstart, λend, n_wavelength_points)
    start_ind = (n_wavelength_points - 1) // 2  # Used to get indices corresponding to original wls

    # Compute Stark profiles
    ϕ_impact, ϕ_quasistatic = brackett_line_stark_profiles(m, wls, λ0, T, ne)

    # Possibly include Doppler by convolving it with the quasistatic profile
    if σdop / stark_width > include_doppler_threshold:
        ϕ_dop = np.array([normal_pdf(wl - λ0, σdop) for wl in wls])
        step_size = wls[1] - wls[0]
        ϕ_quasistatic = autodiffable_conv(ϕ_quasistatic, ϕ_dop) * step_size
        ϕ_quasistatic = ϕ_quasistatic[start_ind:start_ind + n_wavelength_points]

    # Convolve impact and quasistatic profiles
    step_size = wls[1] - wls[0]
    ϕ_conv = autodiffable_conv(ϕ_impact, ϕ_quasistatic) * step_size
    ϕ_conv = ϕ_conv[start_ind:start_ind + n_wavelength_points]

    # Create linear interpolator using JAX instead of scipy
    # JAX's jnp.interp requires sorted x values (which wls is) and handles out-of-bounds with constant extrapolation
    # We'll create a wrapper function that mimics scipy's interp1d behavior
    def jax_linear_interp(x):
        """JAX-compatible linear interpolator."""
        # jnp.interp(x, xp, fp) - x: query points, xp: data x, fp: data y
        # For out-of-bounds, jnp.interp extrapolates with edge values, but we want 0.0
        # So we'll use jnp.where to clip to bounds
        result = jnp.interp(x, wls, ϕ_conv)
        # Set to 0.0 if out of bounds
        in_bounds = (x >= wls[0]) & (x <= wls[-1])
        return jnp.where(in_bounds, result, 0.0)

    return jax_linear_interp, window


def hydrogen_line_absorption(wavelengths: np.ndarray, T: float, ne: float,
                               nH_I: float, nHe_I: float, UH_I: float, xi: float,
                               window_size: float, use_MHD: bool = True,
                               stark_profiles: dict = None) -> np.ndarray:
    """
    Calculate the opacity coefficient from hydrogen lines.

    Uses profiles from Stehlé & Hutcheon (1999) which include Stark and Doppler broadening.
    For the Brackett series (n=4), specialized profiles are computed using brackett_line_stark_profiles.

    Args:
        wavelengths: Wavelengths at which to calculate absorption [cm]
        T: Temperature [K]
        ne: Electron number density [cm^-3]
        nH_I: Neutral hydrogen number density [cm^-3]
        nHe_I: Neutral helium number density [cm^-3]
        UH_I: Neutral hydrogen partition function
        xi: Microturbulent velocity [cm/s]
        window_size: Maximum distance in cm from each hydrogen line center at which
                     to calculate absorption
        use_MHD: Whether to use Mihalas-Daeppen-Hummer formalism for occupation
                 probabilities (default: True)
        stark_profiles: Dictionary of Stark profiles (default: uses loaded profiles)

    Returns:
        Absorption coefficient array [cm^-1] at each wavelength
    """
    if stark_profiles is None:
        stark_profiles = hline_stark_profiles

    if len(stark_profiles) == 0:
        # No Stark profiles available, return zeros
        return np.zeros_like(wavelengths)

    # Initialize absorption array
    alphas = np.zeros(len(wavelengths), dtype=np.float64)

    # Convert wavelengths to frequencies
    nus = c_cgs / wavelengths
    dnu_dlambda = c_cgs / wavelengths**2

    # Get hydrogen mass
    H_mass = atomic_masses[0]

    # Find maximum upper level in stark profiles
    n_max = max(line.upper for line in stark_profiles.values())

    # Precalculate occupation probabilities if using MHD
    if use_MHD:
        ws = np.array([hummer_mihalas_w(T, n, nH_I, nHe_I, ne) for n in range(1, n_max + 1)])
    else:
        ws = np.ones(n_max)

    beta = 1 / (kboltz_eV * T)

    # Holtsmark field for the interpolated Stark profiles
    F0 = 1.25e-9 * ne**(2 / 3)

    # Process Stehlé+ 1999 Stark-broadened profiles
    for transition, line in stark_profiles.items():
        # Check if this temperature and density are within interpolation bounds
        if not (line.temps.min() < T < line.temps.max() and
                line.electron_number_densities.min() < ne < line.electron_number_densities.max()):
            continue  # Skip transitions outside valid T, ne range

        # Get line center wavelength from interpolator
        λ0 = line.lambda0([T, ne])[0]

        # Calculate energy levels and occupation factors
        Elo = RydbergH_eV * (1 - 1 / line.lower**2)
        Eup = RydbergH_eV * (1 - 1 / line.upper**2)

        # Factor of w because transition can't happen if upper level doesn't exist
        levels_factor = ws[line.upper - 1] * (np.exp(-beta * Elo) - np.exp(-beta * Eup)) / UH_I
        amplitude = 10.0**line.log_gf * nH_I * sigma_line(λ0) * levels_factor

        # Find wavelength range for this line
        lb = np.searchsorted(wavelengths, λ0 - window_size, side='left')
        ub = np.searchsorted(wavelengths, λ0 + window_size, side='right')
        if lb >= ub:
            continue

        # Calculate Stark-broadened profile
        nu0 = c_cgs / λ0
        scaled_delta_nu = np.abs(nus[lb:ub] - nu0) / F0
        # Avoid log(0) by using a small epsilon
        scaled_delta_nu = np.maximum(scaled_delta_nu, np.finfo(float).tiny)

        # Interpolate profile (returns log of profile)
        log_profile_vals = np.array([line.profile([T, ne, np.log(sdn)])[0]
                                      for sdn in scaled_delta_nu])
        dIdnu = np.exp(log_profile_vals)

        # Add contribution to absorption
        alphas[lb:ub] += dIdnu * dnu_dlambda[lb:ub] * amplitude

    # Now process the Brackett series (n=4)
    n = 4
    E_low = RydbergH_eV * (1 - 1 / n**2)
    for m in range(5, n_max + 1):
        E = RydbergH_eV * (1 / n**2 - 1 / m**2)
        λ0 = hplanck_eV * c_cgs / E  # cm
        levels_factor = ws[m - 1] * np.exp(-beta * E_low) * (1 - np.exp(-beta * E)) / UH_I
        gf = 2 * n**2 * brackett_oscillator_strength(n, m)
        amplitude = gf * nH_I * sigma_line(λ0) * levels_factor

        # Get Stark profile interpolator
        stark_profile_itp, stark_window = bracket_line_interpolator(
            m, λ0, T, ne, xi, wavelengths[0], wavelengths[-1]
        )

        # Find wavelength range for this line
        lb = np.searchsorted(wavelengths, λ0 - stark_window, side='left')
        ub = np.searchsorted(wavelengths, λ0 + stark_window, side='right')

        if lb < ub:
            # Evaluate interpolated profile
            profile_vals = stark_profile_itp(wavelengths[lb:ub])
            alphas[lb:ub] += profile_vals * amplitude

    return alphas
