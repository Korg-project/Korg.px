"""
Continuum absorption functions.

Functions for computing free-free and bound-free absorption
in stellar atmospheres.
"""

# Enable 64-bit precision in JAX (required for astronomical calculations)
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from .constants import (hplanck_eV, kboltz_eV, Rydberg_eV, RydbergH_eV,
                        hplanck_cgs, kboltz_cgs, c_cgs,
                        electron_charge_cgs, electron_mass_cgs)
from .data_loader import (gauntff_table, gauntff_log10_gamma2, gauntff_log10_u,
                          Hminus_bf_frequencies, Hminus_bf_cross_sections,
                          HI_bf_cross_sections,
                          get_metal_bf_cross_sections)
from .statmech import hummer_mihalas_w
from . import stancil1994
from . import peach1970


# H⁻ ionization energy in eV (from McLaughlin+ 2017)
_H_MINUS_ION_ENERGY = 0.754204

# Convert data to JAX arrays at module load time
_gauntff_table_jax = jnp.array(gauntff_table)
_gauntff_log10_u_jax = jnp.array(gauntff_log10_u)
_gauntff_log10_gamma2_jax = jnp.array(gauntff_log10_gamma2)

# H⁻ bf data
_Hminus_bf_nu_jax = jnp.array(Hminus_bf_frequencies)
_Hminus_bf_sigma_jax = jnp.array(Hminus_bf_cross_sections)
_Hminus_min_nu = float(jnp.min(_Hminus_bf_nu_jax))
_Hminus_ion_nu = _H_MINUS_ION_ENERGY / hplanck_eV

# H I bf data - convert to JAX arrays for each level
# Each entry is (n, energies_eV, cross_sections_Mb)
_HI_bf_data_jax = []
for n, energies, sigmas in HI_bf_cross_sections:
    _HI_bf_data_jax.append((
        n,
        jnp.array(energies, dtype=jnp.float64),
        jnp.array(sigmas, dtype=jnp.float64)
    ))

# Bell & Berrington 1987 H⁻ ff table
_Hminus_ff_lambda_angstrom = jnp.array([
    1823, 2278, 2604, 3038, 3645, 4557, 5063, 5696, 6510, 7595, 9113,
    10126, 11392, 13019, 15189, 18227, 22784, 30378, 45567, 91134,
    113918, 151890], dtype=jnp.float64)
_Hminus_ff_theta = jnp.array([
    0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.8, 3.6], dtype=jnp.float64)
_Hminus_ff_table = jnp.array([
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
], dtype=jnp.float64)


def gaunt_ff_vanHoof(log_u, log_gamma2):
    """
    Thermally averaged, non-relativistic free-free gaunt factor.

    Interpolates the table from van Hoof et al. (2014) using bilinear interpolation.

    Parameters
    ----------
    log_u : float or array
        log₁₀(u) where u = h*ν/(k*Tₑ)
    log_gamma2 : float or array
        log₁₀(γ²) where γ² = Rydberg*Z²/(k*Tₑ)

    Returns
    -------
    float or array
        Free-free gaunt factor

    Notes
    -----
    van Hoof et al. (2014) computed this table with a non-relativistic
    approach, valid up to electron temperatures of ~100 MK (more than
    adequate for stellar atmospheres).

    This implementation uses JAX for automatic differentiation and GPU compatibility.

    Reference
    ---------
    van Hoof et al. 2014: https://ui.adsabs.harvard.edu/abs/2014MNRAS.444..420V
    """
    # Convert inputs to JAX arrays
    log_u = jnp.asarray(log_u)
    log_gamma2 = jnp.asarray(log_gamma2)

    # Find indices in the grid (searchsorted returns the index where value would be inserted)
    # We want i such that grid[i] <= value < grid[i+1]
    i_u = jnp.searchsorted(_gauntff_log10_u_jax, log_u, side='right') - 1
    i_gamma2 = jnp.searchsorted(_gauntff_log10_gamma2_jax, log_gamma2, side='right') - 1

    # Clip indices to valid range [0, len-2] to handle extrapolation/boundary
    i_u = jnp.clip(i_u, 0, len(_gauntff_log10_u_jax) - 2)
    i_gamma2 = jnp.clip(i_gamma2, 0, len(_gauntff_log10_gamma2_jax) - 2)

    # Get the grid points that bracket our query point
    u0 = _gauntff_log10_u_jax[i_u]
    u1 = _gauntff_log10_u_jax[i_u + 1]
    g0 = _gauntff_log10_gamma2_jax[i_gamma2]
    g1 = _gauntff_log10_gamma2_jax[i_gamma2 + 1]

    # Compute fractional positions within the cell
    # Handle division by zero if grid spacing is zero (shouldn't happen)
    t_u = jnp.where(u1 != u0, (log_u - u0) / (u1 - u0), 0.0)
    t_gamma2 = jnp.where(g1 != g0, (log_gamma2 - g0) / (g1 - g0), 0.0)

    # Get the 4 corner values
    # Table shape is (n_u, n_gamma2), indexed as table[i_u, i_gamma2]
    f00 = _gauntff_table_jax[i_u, i_gamma2]
    f01 = _gauntff_table_jax[i_u, i_gamma2 + 1]
    f10 = _gauntff_table_jax[i_u + 1, i_gamma2]
    f11 = _gauntff_table_jax[i_u + 1, i_gamma2 + 1]

    # Bilinear interpolation
    # f(u, gamma2) = (1-t_u)*(1-t_g)*f00 + (1-t_u)*t_g*f01 + t_u*(1-t_g)*f10 + t_u*t_g*f11
    result = ((1.0 - t_u) * (1.0 - t_gamma2) * f00 +
              (1.0 - t_u) * t_gamma2 * f01 +
              t_u * (1.0 - t_gamma2) * f10 +
              t_u * t_gamma2 * f11)

    return result


def hydrogenic_ff_absorption(nu, T, Z, ni, ne):
    """
    Free-free linear absorption coefficient for a hydrogenic species.

    Parameters
    ----------
    nu : float or array
        Frequency in Hz
    T : float
        Temperature in K
    Z : int
        Charge of the ion (e.g., 1 for H II, 2 for He III)
    ni : float
        Number density of the ion species in cm⁻³
    ne : float
        Number density of free electrons in cm⁻³

    Returns
    -------
    float or array
        Linear absorption coefficient α in cm⁻¹

    Notes
    -----
    The naming convention for free-free absorption is counter-intuitive:
    a free-free interaction is named as though the ion had one more bound
    electron. In practice, this means:
    - For H I free-free: ni should be n(H II)
    - For He II free-free: ni should be n(He III)
    - For Li III free-free: ni should be n(Li IV)

    The absorption coefficient (corrected for stimulated emission) is:
        α = coef * Z² * ne * ni * (1 - exp(-hν/(kT))) * g_ff / (√T * ν³)

    where g_ff is the free-free gaunt factor and coef ≈ 3.6919e8.

    This follows equation 5.18b of Rybicki & Lightman (2004) and
    equation 5.8 of Kurucz (1970).

    This implementation uses JAX for automatic differentiation and GPU compatibility.

    Reference
    ---------
    van Hoof et al. 2014 for gaunt factors
    """
    # Convert to JAX arrays
    nu = jnp.asarray(nu)

    inv_T = 1.0 / T
    Z2 = Z * Z

    # Compute dimensionless parameters
    h_nu_div_kT = (hplanck_eV / kboltz_eV) * nu * inv_T
    log_u = jnp.log10(h_nu_div_kT)
    log_gamma2 = jnp.log10((Rydberg_eV / kboltz_eV) * Z2 * inv_T)

    # Get gaunt factor via interpolation (now JAX-compatible)
    gaunt_ff = gaunt_ff_vanHoof(log_u, log_gamma2)

    # Frequency-dependent factor
    F_nu = 3.6919e8 * gaunt_ff * Z2 * jnp.sqrt(inv_T) / (nu * nu * nu)

    # Full absorption coefficient with stimulated emission correction
    return ni * ne * F_nu * (1.0 - jnp.exp(-h_nu_div_kT))


def ndens_Hminus(nH_I_div_partition, ne, T, ion_energy=_H_MINUS_ION_ENERGY):
    """
    Compute the number density of H⁻ using the Saha equation.

    This implements equation 5.10 of Kurucz (1970). The Saha equation is
    applied where the "ground state" is H⁻ and the "first ionization state"
    is H I. The partition function of H⁻ is 1 at all temperatures.

    Parameters
    ----------
    nH_I_div_partition : float
        Number density of H I divided by its partition function (cm⁻³)
    ne : float
        Number density of free electrons (cm⁻³)
    T : float
        Temperature in K
    ion_energy : float, optional
        H⁻ ionization energy in eV (default: 0.754204)

    Returns
    -------
    float
        Number density of H⁻ in cm⁻³

    Notes
    -----
    The Boltzmann factor is unity for the ground state, and the degeneracy
    g = 2 for H I ground state.

    This function assumes T >= 1000 K (typical stellar atmosphere range).
    For very low temperatures, the exponential can overflow.
    """
    # Convert to JAX arrays
    nH_I_div_partition = jnp.asarray(nH_I_div_partition)
    ne = jnp.asarray(ne)
    T = jnp.asarray(T)

    # n(H I, n=1) = 2 * n(H I) / U(T) (degeneracy = 2, Boltzmann factor = 1)
    nHI_groundstate = 2.0 * nH_I_div_partition

    # Coefficient: (h²/(2πm))^1.5 where m is electron mass
    coef = 3.31283018e-22  # cm³·eV^1.5

    # β = 1/(k·T) in eV units
    beta = 1.0 / (kboltz_eV * T)

    # Saha equation: n(H⁻) = 0.25 * n(H I, n=1) * ne * coef * β^1.5 * exp(χ * β)
    return 0.25 * nHI_groundstate * ne * coef * jnp.power(beta, 1.5) * jnp.exp(ion_energy * beta)


def Hminus_bf_cross_section(nu):
    """
    H⁻ bound-free photoionization cross-section.

    Interpolates the cross-sections from McLaughlin+ 2017 with power-law
    extrapolation below the tabulated range.

    Parameters
    ----------
    nu : float or array
        Frequency in Hz

    Returns
    -------
    float or array
        Cross-section in cm²

    Notes
    -----
    - Below ionization threshold (ν ≤ ν_ion): returns 0
    - Between threshold and table minimum: σ ∝ (ν - ν_ion)^1.5
    - Above table minimum: linear interpolation of McLaughlin data

    Reference
    ---------
    McLaughlin+ 2017: https://ui.adsabs.harvard.edu/abs/2017JPhB...50k4001M
    """
    # Convert to JAX array
    nu = jnp.asarray(nu)

    # Check if below ionization threshold
    below_threshold = nu <= _Hminus_ion_nu

    # Check if in extrapolation region (above threshold but below table)
    in_extrapolation = (nu > _Hminus_ion_nu) & (nu < _Hminus_min_nu)

    # For interpolation region, find indices
    i = jnp.searchsorted(_Hminus_bf_nu_jax, nu, side='right') - 1
    i = jnp.clip(i, 0, len(_Hminus_bf_nu_jax) - 2)

    # Get bracketing points
    nu0 = _Hminus_bf_nu_jax[i]
    nu1 = _Hminus_bf_nu_jax[i + 1]
    sigma0 = _Hminus_bf_sigma_jax[i]
    sigma1 = _Hminus_bf_sigma_jax[i + 1]

    # Linear interpolation
    t = jnp.where(nu1 != nu0, (nu - nu0) / (nu1 - nu0), 0.0)
    sigma_interp = sigma0 + t * (sigma1 - sigma0)

    # Power-law extrapolation: σ ∝ (ν - ν_ion)^1.5
    # Coefficient from matching at table minimum
    sigma_at_min = _Hminus_bf_sigma_jax[0]
    coef = sigma_at_min / jnp.power(_Hminus_min_nu - _Hminus_ion_nu, 1.5)
    sigma_extrap = coef * jnp.power(nu - _Hminus_ion_nu, 1.5)

    # Select appropriate value based on frequency range
    result = jnp.where(below_threshold, 0.0,
                      jnp.where(in_extrapolation, sigma_extrap, sigma_interp))

    return result


def Hminus_bf(nu, T, nH_I_div_partition, ne):
    """
    H⁻ bound-free linear absorption coefficient.

    Computes α_ν = σ_bf(H⁻) * n(H⁻) * (1 - exp(-hν/kT))

    Parameters
    ----------
    nu : float or array
        Frequency in Hz
    T : float
        Temperature in K
    nH_I_div_partition : float
        Number density of H I divided by its partition function (cm⁻³)
    ne : float
        Number density of free electrons (cm⁻³)

    Returns
    -------
    float or array
        Linear absorption coefficient α in cm⁻¹

    Notes
    -----
    This uses cross-sections from McLaughlin+ 2017.

    The number density of H⁻ is computed on the fly using the Saha equation,
    assuming n(H⁻) << n(H I) + n(H II).

    Reference
    ---------
    McLaughlin+ 2017: https://ui.adsabs.harvard.edu/abs/2017JPhB...50k4001M
    """
    # Convert to JAX arrays
    nu = jnp.asarray(nu)

    # Get cross-section in cm²
    cross_section = Hminus_bf_cross_section(nu)

    # Stimulated emission correction
    stimulated_emission = 1.0 - jnp.exp(-hplanck_cgs * nu / (kboltz_cgs * T))

    # H⁻ number density from Saha equation
    n_Hminus = ndens_Hminus(nH_I_div_partition, ne, T)

    # α = σ * n * (1 - exp(-hν/kT))
    return n_Hminus * cross_section * stimulated_emission


def Hminus_ff(nu, T, nH_I_div_partition, ne):
    """
    H⁻ free-free linear absorption coefficient.

    The naming scheme for free-free absorption is counter-intuitive. This
    actually refers to the reaction: photon + e⁻ + H I -> e⁻ + H I.

    Parameters
    ----------
    nu : float or array
        Frequency in Hz
    T : float
        Temperature in K
    nH_I_div_partition : float
        Number density of H I divided by its partition function (cm⁻³)
    ne : float
        Number density of free electrons (cm⁻³)

    Returns
    -------
    float or array
        Linear absorption coefficient α in cm⁻¹

    Notes
    -----
    Based on Table 1 from Bell & Berrington (1987), which tabulates "the H⁻
    absorption coefficient" K (including stimulated emission correction) in
    units of cm⁴/dyn.

    K must be multiplied by:
    - Electron partial pressure: P_e = ne * k * T
    - Ground-state H I number density: n(H I, n=1) = 2 * n(H I) / U(T)

    The table has wavelength range 1823-151890 Å and temperature (θ = 5040/T)
    range 0.5-3.6, corresponding to T = 1400-10080 K.

    Reference
    ---------
    Bell & Berrington 1987: https://doi.org/10.1088/0022-3700/20/4/019
    """
    # Convert frequency to wavelength in Angstroms
    nu = jnp.asarray(nu)
    lambda_angstrom = c_cgs * 1e8 / nu

    # Convert temperature to θ = 5040/T
    theta = 5040.0 / T

    # Bilinear interpolation of Bell & Berrington table
    # Find indices in lambda and theta grids
    i_lambda = jnp.searchsorted(_Hminus_ff_lambda_angstrom, lambda_angstrom, side='right') - 1
    i_theta = jnp.searchsorted(_Hminus_ff_theta, theta, side='right') - 1

    # Clip to valid range
    i_lambda = jnp.clip(i_lambda, 0, len(_Hminus_ff_lambda_angstrom) - 2)
    i_theta = jnp.clip(i_theta, 0, len(_Hminus_ff_theta) - 2)

    # Get grid points
    lam0 = _Hminus_ff_lambda_angstrom[i_lambda]
    lam1 = _Hminus_ff_lambda_angstrom[i_lambda + 1]
    th0 = _Hminus_ff_theta[i_theta]
    th1 = _Hminus_ff_theta[i_theta + 1]

    # Compute fractional positions
    t_lambda = jnp.where(lam1 != lam0, (lambda_angstrom - lam0) / (lam1 - lam0), 0.0)
    t_theta = jnp.where(th1 != th0, (theta - th0) / (th1 - th0), 0.0)

    # Get corner values from table
    # Table shape is (n_lambda, n_theta)
    K00 = _Hminus_ff_table[i_lambda, i_theta]
    K01 = _Hminus_ff_table[i_lambda, i_theta + 1]
    K10 = _Hminus_ff_table[i_lambda + 1, i_theta]
    K11 = _Hminus_ff_table[i_lambda + 1, i_theta + 1]

    # Bilinear interpolation
    K_interp = ((1.0 - t_lambda) * (1.0 - t_theta) * K00 +
                (1.0 - t_lambda) * t_theta * K01 +
                t_lambda * (1.0 - t_theta) * K10 +
                t_lambda * t_theta * K11)

    # K is in units of 10^-26 cm⁴/dyn (factor built into table)
    K = 1e-26 * K_interp

    # Electron partial pressure in dyn/cm²
    P_e = ne * kboltz_cgs * T

    # Ground-state H I number density
    # n(H I, n=1) = g_n=1 * n(H I) / U(T) * exp(-E_n=1/(kT))
    #             = 2 * n(H I) / U(T) * exp(0)
    #             = 2 * (n(H I) / U(T))
    nHI_groundstate = 2.0 * nH_I_div_partition

    # α = K * P_e * n(H I, n=1)
    return K * P_e * nHI_groundstate


def simple_hydrogen_bf_cross_section(n, nu):
    """
    Simple analytic approximation for H I bf cross-section.

    This implements equation 5.5 from Kurucz (1970) with the correction
    noted in the Korg documentation: Z² in the numerator should be Z⁴.
    This was confirmed by comparison with the Opacity Project and
    Rybicki & Lightman equation 10.54.

    Parameters
    ----------
    n : int
        Principal quantum number (should be >= 7 for accuracy)
    nu : float or array
        Frequency in Hz

    Returns
    -------
    float or array
        Cross-section in Megabarns (1 Mb = 10⁻¹⁸ cm²)

    Notes
    -----
    This function is used to extrapolate cross-sections for high-n levels
    where tabulated data is unavailable or unnecessary (since these levels
    are often pressure-dissolved).

    For n < 7, tabulated cross-sections from Nahar 2021 should be used
    instead for better accuracy.

    The formula is:
    σ = (64π⁴e¹⁰mₑ)/(c h⁶ 3√3) × Z⁴/n⁵ × 1/ν³

    For hydrogen, Z=1, and the constant ≈ 2.815×10²⁹.
    """
    # Convert to JAX arrays
    nu = jnp.asarray(nu)

    # For hydrogen, Z = 1
    inv_n = 1.0 / n
    inv_n2 = inv_n * inv_n
    inv_n5 = inv_n2 * inv_n2 * inv_n

    # Ionization threshold for this level
    ionization_energy = RydbergH_eV * inv_n2

    # Check if photon has enough energy to ionize
    photon_energy = hplanck_eV * nu
    has_enough_energy = photon_energy >= ionization_energy

    # Constant: 64π⁴e¹⁰mₑ/(c h⁶ 3√3) in units that give Mb
    bf_sigma_const = 2.815e29

    # Cross-section formula: const × 1/n⁵ × 1/ν³ × 1e18 (convert to Mb)
    inv_nu3 = 1.0 / (nu * nu * nu)
    sigma = bf_sigma_const * inv_n5 * inv_nu3 * 1e18

    # Return 0 if photon doesn't have enough energy
    return jnp.where(has_enough_energy, sigma, 0.0)


def _interpolate_HI_bf_cross_section(photon_energy_eV, n):
    """
    Interpolate H I bf cross-section for a given level from Nahar 2021 data.

    Parameters
    ----------
    photon_energy_eV : float or array
        Photon energy in eV
    n : int
        Principal quantum number (1-6)

    Returns
    -------
    float or array
        Cross-section in Megabarns
    """
    # Get data for this level (n is 1-indexed in the data)
    _, energies, sigmas = _HI_bf_data_jax[n - 1]

    # Convert to JAX array
    photon_energy_eV = jnp.asarray(photon_energy_eV)

    # Linear interpolation
    i = jnp.searchsorted(energies, photon_energy_eV, side='right') - 1
    i = jnp.clip(i, 0, len(energies) - 2)

    E0 = energies[i]
    E1 = energies[i + 1]
    sigma0 = sigmas[i]
    sigma1 = sigmas[i + 1]

    t = jnp.where(E1 != E0, (photon_energy_eV - E0) / (E1 - E0), 0.0)
    sigma_interp = sigma0 + t * (sigma1 - sigma0)

    # Extrapolate linearly for energies outside the table
    # (This matches Julia's Line() extrapolation)
    return sigma_interp


def H_I_bf(nu, T, nH_I, nHe_I, ne, invU_H, n_max_MHD=6,
           use_hubeny_generalization=False):
    """
    H I bound-free linear absorption coefficient with MHD level dissolution.

    Computes the bound-free absorption coefficient for neutral hydrogen using
    tabulated cross-sections from Nahar 2021 for n=1-6 and analytic formulas
    for higher levels, modified by the Mihalas-Hummer-Daeppen occupation
    probability formalism to account for pressure-induced level dissolution.

    Parameters
    ----------
    nu : float or array
        Frequency in Hz (should be sorted)
    T : float
        Temperature in K
    nH_I : float
        Total number density of H I in cm⁻³
    nHe_I : float
        Total number density of He I in cm⁻³
    ne : float
        Electron number density in cm⁻³
    invU_H : float
        Inverse of H I partition function (excluding MHD contributions)
    n_max_MHD : int, optional
        Maximum level to use MHD+tabulated cross-sections (default: 6)
    use_hubeny_generalization : bool, optional
        Use Hubeny 1994 generalization of MHD (default: False)

    Returns
    -------
    float or array
        Linear absorption coefficient α in cm⁻¹

    Notes
    -----
    This function uses the MHD formalism even though the partition function
    doesn't include MHD contributions. This means series limit jumps
    (e.g., Balmer jump) are "rounded off" because photons with less than
    the classical ionization energy can still ionize if the upper level is
    dissolved into the continuum.

    For n=1, MHD level dissolution is NOT applied (use_MHD_for_Lyman=False)
    to avoid inflated cross-sections in the visible.

    Cross-sections are from Nahar 2021:
    https://ui.adsabs.harvard.edu/abs/2021Atoms...9...73N

    Reference
    ---------
    Mihalas, Hummer, & Daeppen formalism for occupation probabilities
    """
    # Convert to JAX arrays and handle scalar input
    nu_input = jnp.asarray(nu)
    is_scalar = nu_input.ndim == 0
    nu = jnp.atleast_1d(nu_input)

    # H I ionization energy from ground state
    chi_ion = RydbergH_eV  # 13.598 eV

    # Initialize total cross-section (in Megabarns)
    total_cross_section = jnp.zeros_like(nu)

    # Loop over levels n=1 to n_max_MHD
    for n in range(1, n_max_MHD + 1):
        # Occupation probability for lower level (level n)
        w_lower = hummer_mihalas_w(
            T, float(n), nH_I, nHe_I, ne,
            use_hubeny_generalization=use_hubeny_generalization
        )

        # Boltzmann occupation (degeneracy already in Nahar cross-sections)
        boltzmann_factor = jnp.exp(-chi_ion * (1.0 - 1.0 / (n * n)) / (kboltz_eV * T))
        occupation_prob = w_lower * boltzmann_factor

        # Ionization threshold frequency for this level
        nu_break = chi_ion / (n * n * hplanck_eV)

        # Interpolate cross-sections from tabulated data
        # For nu < nu_break: extrapolate as σ ∝ ν⁻³
        # For nu >= nu_break: use tabulated values
        photon_energies = hplanck_eV * nu
        sigma_at_break = _interpolate_HI_bf_cross_section(chi_ion / (n * n), n)
        scaling_factor = sigma_at_break * nu_break**3

        # Power-law extrapolation below threshold
        sigma_extrap = scaling_factor / (nu**3)

        # Tabulated values above threshold
        sigma_table = _interpolate_HI_bf_cross_section(photon_energies, n)

        # Choose based on frequency
        cross_section = jnp.where(nu < nu_break, sigma_extrap, sigma_table)

        # Level dissolution fraction
        # For n=1, don't use MHD (use_MHD_for_Lyman=False by default)
        if n == 1:
            # No level dissolution for Lyman series
            dissolved_fraction = jnp.where(nu >= nu_break, 1.0, 0.0)
        else:
            # For nu >= nu_break: fully dissolved
            # For nu < nu_break: compute dissolution using MHD
            def compute_dissolution(nu_val):
                # Effective quantum number for absorbed photon
                n_eff = 1.0 / jnp.sqrt(1.0 / (n * n) - hplanck_eV * nu_val / chi_ion)

                # Occupation probability for upper level
                w_upper = hummer_mihalas_w(
                    T, n_eff, nH_I, nHe_I, ne,
                    use_hubeny_generalization=use_hubeny_generalization
                )

                # Fraction of upper levels that are dissolved
                return 1.0 - w_upper / w_lower

            # Vectorize over frequency array
            dissolved_below = jax.vmap(compute_dissolution)(nu)
            dissolved_fraction = jnp.where(nu >= nu_break, 1.0, dissolved_below)

        # Add contribution from this level
        total_cross_section += occupation_prob * cross_section * dissolved_fraction

    # Add contributions from high-n levels (n > n_max_MHD)
    for n in range(n_max_MHD + 1, 41):
        w_lower = hummer_mihalas_w(
            T, float(n), nH_I, nHe_I, ne,
            use_hubeny_generalization=use_hubeny_generalization
        )

        # Stop if occupation probability is negligible
        if float(w_lower) < 1e-5:
            break

        # Degeneracy for level n is 2n²
        degeneracy = 2 * n * n
        boltzmann_factor = jnp.exp(-chi_ion * (1.0 - 1.0 / (n * n)) / (kboltz_eV * T))
        occupation_prob = degeneracy * w_lower * boltzmann_factor

        # Use simple analytic formula
        sigma_simple = simple_hydrogen_bf_cross_section(n, nu)

        total_cross_section += occupation_prob * sigma_simple

    # Convert from Megabarns to cm² and apply stimulated emission
    stimulated_emission = 1.0 - jnp.exp(-hplanck_eV * nu / (kboltz_eV * T))

    # α = n(H I) / U(H I) × σ_total × (1 - exp(-hν/kT)) × 10⁻¹⁸
    result = nH_I * invU_H * total_cross_section * stimulated_emission * 1e-18

    # Return scalar if input was scalar
    return result[0] if is_scalar else result


def H2plus_bf_and_ff(nu, T, nH_I, nH_II):
    """
    Combined H₂⁺ bound-free and free-free linear absorption coefficient.

    Computes the total absorption from H₂⁺ using the tables from Stancil 1994.
    Note that this refers to interactions between free protons and neutral
    hydrogen (not electrons and doubly ionized H₂).

    Parameters
    ----------
    nu : float or array
        Frequency in Hz
    T : float
        Temperature in K
    nH_I : float
        Number density of H I in cm⁻³
    nH_II : float
        Number density of H II (protons) in cm⁻³

    Returns
    -------
    float or array
        Linear absorption coefficient α in cm⁻¹

    Notes
    -----
    This function computes n(H₂⁺) on the fly from n(H I) and n(H II) using
    the equilibrium constant K from Stancil 1994. The cross-sections have
    units of cm⁵ because they must be multiplied by n(H I) and n(H II).

    Valid ranges:
    - Temperature: 3150-25200 K
    - Wavelength: 700-200000 Å (bf), 500-200000 Å (ff)

    The absorption coefficient is:
    α = (σ_bf/K + σ_ff) × n(H I) × n(H II) × (1 - exp(-hν/kT))

    where K = n(H I) × n(H II) / n(H₂⁺).

    Reference
    ---------
    Stancil 1994: https://ui.adsabs.harvard.edu/abs/1994ApJ...430..360S/abstract
    """
    # Convert frequency to wavelength in Angstroms
    nu = jnp.asarray(nu)
    λ_angstrom = c_cgs * 1e8 / nu

    # Get equilibrium constant and cross-sections
    K = stancil1994.K_H2plus(T)
    σ_bf = stancil1994.σ_H2plus_bf(λ_angstrom, T)
    σ_ff = stancil1994.σ_H2plus_ff(λ_angstrom, T)

    # Stimulated emission correction
    beta_eV = 1.0 / (kboltz_eV * T)
    stimulated_emission = 1.0 - jnp.exp(-hplanck_eV * nu * beta_eV)

    # Combined absorption coefficient
    # (σ_bf/K + σ_ff) has units of cm²
    # Multiply by n(H I) × n(H II) to get cm⁻¹
    return (σ_bf / K + σ_ff) * nH_I * nH_II * stimulated_emission


# He I state energies and degeneracies
# From section 5.5 of Kurucz (1970)
_HE_I_STATE_DATA = {
    1: (1.0, 0.0),      # (degeneracy, energy_eV)
    2: (3.0, 19.819),
    3: (1.0, 20.615),
    4: (9.0, 20.964)
}


def ndens_state_He_I(n, nsdens_div_partition, T):
    """
    Compute the number density of He I in a specific state.

    Parameters
    ----------
    n : int
        Principal quantum number (1-4 supported)
    nsdens_div_partition : float
        Total number density of He I divided by its partition function (cm⁻³)
    T : float
        Temperature in K

    Returns
    -------
    float
        Number density of He I in state n (cm⁻³)

    Notes
    -----
    Uses energy levels and degeneracies from Kurucz (1970) section 5.5.
    Only states n=1-4 are currently supported.
    """
    if n not in _HE_I_STATE_DATA:
        raise ValueError(f"Unknown excited state properties for He I with n={n}")

    g_n, energy_level = _HE_I_STATE_DATA[n]

    # Boltzmann distribution: n_state = (total / U) × g × exp(-E/(kT))
    return nsdens_div_partition * g_n * jnp.exp(-energy_level / (kboltz_eV * T))


# John (1994) He⁻ ff table
# OCR'd from Table 1 of John (1994) https://ui.adsabs.harvard.edu/abs/1994MNRAS.269..871J
_Heminus_ff_lambda_angstrom = jnp.array([
    5063.0, 5695.0, 6509.0, 7594.0, 9113.0, 11391.0, 15188.0, 18225.0,
    22782.0, 30376.0, 36451.0, 45564.0, 60751.0, 91127.0, 113900.0, 151878.0
], dtype=jnp.float64)

_Heminus_ff_theta = jnp.array([
    0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.8, 3.6
], dtype=jnp.float64)

# K values in units of 10^-26 cm⁴/dyn (includes stimulated emission)
# Table is [n_lambda × n_theta]
_Heminus_ff_table = jnp.array([
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
], dtype=jnp.float64)


def Heminus_ff(nu, T, nHe_I_div_partition, ne):
    """
    He⁻ free-free linear absorption coefficient.

    The naming scheme for free-free absorption is counter-intuitive. This
    actually refers to the reaction: photon + e⁻ + He I -> e⁻ + He I.

    Parameters
    ----------
    nu : float or array
        Frequency in Hz
    T : float
        Temperature in K
    nHe_I_div_partition : float
        Number density of He I divided by its partition function (cm⁻³)
    ne : float
        Number density of free electrons (cm⁻³)

    Returns
    -------
    float or array
        Linear absorption coefficient α in cm⁻¹

    Notes
    -----
    Based on Table 1 from John (1994), which tabulates the He⁻ absorption
    coefficient K (including stimulated emission correction) in units of
    cm⁴/dyn.

    K must be multiplied by:
    - Electron partial pressure: P_e = ne * k * T
    - Ground-state He I number density: n(He I, n=1)

    Valid wavelength range: 5063-151878 Å
    Valid temperature range (θ = 5040/T): 0.5-3.6, corresponding to T = 1400-10080 K

    According to John (1994), improved calculations are unlikely to alter
    the tabulated data for λ > 10000 Å by more than about 2%. The errors
    for 5063 Å ≤ λ ≤ 10000 Å are expected to be well below 10%.

    Reference
    ---------
    John 1994: https://ui.adsabs.harvard.edu/abs/1994MNRAS.269..871J
    """
    # Convert frequency to wavelength in Angstroms
    nu = jnp.asarray(nu)
    lambda_angstrom = c_cgs * 1e8 / nu

    # Convert temperature to θ = 5040/T
    theta = 5040.0 / T

    # Bilinear interpolation of John (1994) table
    # Find indices in lambda and theta grids
    i_lambda = jnp.searchsorted(_Heminus_ff_lambda_angstrom, lambda_angstrom, side='right') - 1
    i_theta = jnp.searchsorted(_Heminus_ff_theta, theta, side='right') - 1

    # Clip to valid range
    i_lambda = jnp.clip(i_lambda, 0, len(_Heminus_ff_lambda_angstrom) - 2)
    i_theta = jnp.clip(i_theta, 0, len(_Heminus_ff_theta) - 2)

    # Get grid points
    lam0 = _Heminus_ff_lambda_angstrom[i_lambda]
    lam1 = _Heminus_ff_lambda_angstrom[i_lambda + 1]
    th0 = _Heminus_ff_theta[i_theta]
    th1 = _Heminus_ff_theta[i_theta + 1]

    # Compute fractional positions
    t_lambda = jnp.where(lam1 != lam0, (lambda_angstrom - lam0) / (lam1 - lam0), 0.0)
    t_theta = jnp.where(th1 != th0, (theta - th0) / (th1 - th0), 0.0)

    # Get corner values from table
    # Table shape is (n_lambda, n_theta)
    K00 = _Heminus_ff_table[i_lambda, i_theta]
    K01 = _Heminus_ff_table[i_lambda, i_theta + 1]
    K10 = _Heminus_ff_table[i_lambda + 1, i_theta]
    K11 = _Heminus_ff_table[i_lambda + 1, i_theta + 1]

    # Bilinear interpolation
    K_interp = ((1.0 - t_lambda) * (1.0 - t_theta) * K00 +
                (1.0 - t_lambda) * t_theta * K01 +
                t_lambda * (1.0 - t_theta) * K10 +
                t_lambda * t_theta * K11)

    # K is in units of 10^-26 cm⁴/dyn (factor built into table)
    K = 1e-26 * K_interp

    # Electron partial pressure in dyn/cm²
    P_e = ne * kboltz_cgs * T

    # Ground-state He I number density
    # n(He I, n=1) = g_n=1 * n(He I) / U(T) * exp(-E_n=1/(kT))
    #              = 1.0 * n(He I) / U(T) * exp(0)
    #              = n(He I) / U(T)
    nHeI_groundstate = ndens_state_He_I(1, nHe_I_div_partition, T)

    # α = K * P_e * n(He I, n=1)
    return K * P_e * nHeI_groundstate


def electron_scattering(ne):
    """
    Linear absorption coefficient from Thomson scattering by free electrons.

    This has no wavelength dependence and assumes isotropic scattering.

    Parameters
    ----------
    ne : float
        Number density of free electrons (cm⁻³)

    Returns
    -------
    float
        Linear absorption coefficient α in cm⁻¹

    Notes
    -----
    The Thomson scattering cross-section is:
    σ_T = (8π/3) × (e²/(m_e c²))²

    where e²/(m_e c²) is the classical electron radius (r_e ≈ 2.818×10⁻¹³ cm).

    The linear absorption coefficient is:
    α = σ_T × n_e

    Reference
    ---------
    Gray (2005) "The Observation and Analysis of Stellar Photospheres", p. 160
    """
    # Classical electron radius: r_e = e²/(m_e c²)
    r_e = electron_charge_cgs**2 / (electron_mass_cgs * c_cgs**2)

    # Thomson scattering cross-section: σ_T = (8π/3) × r_e²
    sigma_thomson = 8.0 * jnp.pi / 3.0 * r_e**2

    # Linear absorption coefficient
    return sigma_thomson * ne


def rayleigh(nu, nH_I, nHe_I, nH2):
    """
    Linear absorption coefficient from Rayleigh scattering.

    Computes Rayleigh scattering by neutral H, He, and H₂ using formulations
    from Colgan+ 2016 (H and He) and Dalgarno & Williams 1962 (H₂).

    Parameters
    ----------
    nu : float or array
        Frequency in Hz (must be ≤ c/1300Å ≈ 2.31×10¹⁵ Hz)
    nH_I : float
        Number density of H I (cm⁻³)
    nHe_I : float
        Number density of He I (cm⁻³)
    nH2 : float
        Number density of H₂ (cm⁻³)

    Returns
    -------
    float or array
        Linear absorption coefficient α in cm⁻¹

    Notes
    -----
    Valid redward of 1300 Å (Lyman alpha for H, breakdown wavelength for H₂).

    The H and He formulations are from Colgan+ 2016 equations 6 and 7:
    - σ_H/σ_T = 20.24 × E² + 239.2 × E³ + 2256 × E⁴
    - σ_He/σ_T = 1.913 × E² + 4.52 × E³ + 7.90 × E⁴

    where E = (hν / 2Ryd) and σ_T is the Thomson cross-section.

    The H₂ formulation is from Dalgarno & Williams 1962 equation 3:
    α_H₂ = (8.14×10⁻¹³ / λ⁴ + 1.28×10⁻⁶ / λ⁶ + 1.61 / λ⁸) × n(H₂)

    where λ is in Angstroms.

    References
    ----------
    Colgan+ 2016: https://ui.adsabs.harvard.edu/abs/2016ApJ...817..116C
    Dalgarno & Williams 1962: https://ui.adsabs.harvard.edu/abs/1962ApJ...136..690D
    """
    nu = jnp.asarray(nu)

    # Thomson scattering cross-section
    sigma_th = 6.65246e-25  # cm²

    # Photon energy over 2 Rydberg: E = hν / (2 Ryd)
    E_2Ryd = hplanck_eV * nu / (2.0 * Rydberg_eV)
    E_2Ryd_2 = E_2Ryd**2
    E_2Ryd_4 = E_2Ryd_2**2
    E_2Ryd_6 = E_2Ryd_2 * E_2Ryd_4
    E_2Ryd_8 = E_2Ryd_4**2

    # Colgan+ 2016 equation 6: H I Rayleigh cross-section
    sigma_H_over_sigma_th = 20.24 * E_2Ryd_4 + 239.2 * E_2Ryd_6 + 2256.0 * E_2Ryd_8

    # Colgan+ 2016 equation 7: He I Rayleigh cross-section
    sigma_He_over_sigma_th = 1.913 * E_2Ryd_4 + 4.52 * E_2Ryd_6 + 7.90 * E_2Ryd_8

    # Combined H and He contributions
    alpha_H_He = (nH_I * sigma_H_over_sigma_th + nHe_I * sigma_He_over_sigma_th) * sigma_th

    # H₂ contribution (Dalgarno & Williams 1962 equation 3)
    # Wavelength in Angstroms
    lambda_angstrom = c_cgs * 1e8 / nu
    inv_lambda_2 = 1.0 / (lambda_angstrom**2)
    inv_lambda_4 = inv_lambda_2**2
    inv_lambda_6 = inv_lambda_2 * inv_lambda_4
    inv_lambda_8 = inv_lambda_4**2

    alpha_H2 = (8.14e-13 * inv_lambda_4 + 1.28e-6 * inv_lambda_6 + 1.61 * inv_lambda_8) * nH2

    return alpha_H_He + alpha_H2


def _parse_species_charge(species_name):
    """
    Parse the charge (number of electrons removed) from a species name.

    Parameters
    ----------
    species_name : str
        Species name like 'H_I', 'He_II', 'C_III', etc.

    Returns
    -------
    int
        Charge (0 for neutral, 1 for singly ionized, etc.)

    Examples
    --------
    >>> _parse_species_charge('H_I')
    0
    >>> _parse_species_charge('He_II')
    1
    >>> _parse_species_charge('C_III')
    2
    """
    # Roman numeral mapping
    roman_to_charge = {
        'I': 0,
        'II': 1,
        'III': 2,
        'IV': 3,
        'V': 4,
        'VI': 5,
        'VII': 6,
        'VIII': 7,
        'IX': 8,
        'X': 9
    }

    # Split on underscore and get the ionization stage
    parts = species_name.split('_')
    if len(parts) == 2:
        roman = parts[1]
        return roman_to_charge.get(roman, 0)
    return 0


def positive_ion_ff_absorption(nu, T, number_densities, ne):
    """
    Free-free absorption from all positively charged ions.

    Computes the total free-free absorption coefficient from all positive ions,
    using Peach 1970 departure coefficients for He II, C II, Si II, and Mg II,
    and the hydrogenic approximation for all other species.

    Parameters
    ----------
    nu : float or array
        Frequency in Hz
    T : float
        Temperature in K
    number_densities : dict
        Dictionary mapping species names (e.g., 'H_II', 'He_II') to number
        densities in cm⁻³. Only positively charged species are included.
    ne : float
        Electron number density in cm⁻³

    Returns
    -------
    float or array
        Linear absorption coefficient α in cm⁻¹

    Notes
    -----
    The free-free interaction is named as though the ion had one more bound
    electron. For example:
    - H I ff uses n(H II) - the interaction is e⁻ + H II
    - C I ff uses n(C II) - the interaction is e⁻ + C II

    For species with Peach 1970 departure coefficients:
    α = α_hydrogenic × (1 + D(T, σ))

    where σ = hν/(Ryd × Z²) and D is the departure coefficient.

    For all other positive ions, uses the uncorrected hydrogenic approximation,
    accumulating contributions by charge (Z=1 and Z=2).

    References
    ----------
    Peach+ 1970: https://ui.adsabs.harvard.edu/abs/1970MmRAS..73....1P
    """
    nu = jnp.asarray(nu)

    # Initialize total absorption
    alpha_total = jnp.zeros_like(nu)

    # Accumulate number densities by charge for species without departure coefficients
    ndens_Z1 = 0.0  # Singly charged ions (e.g., H II, C II without correction)
    ndens_Z2 = 0.0  # Doubly charged ions (e.g., He III)

    for species_name, ndens in number_densities.items():
        # Parse charge from species name
        charge = _parse_species_charge(species_name)

        # Skip neutral species
        if charge <= 0:
            continue

        # Check if we have a departure coefficient for this species
        if species_name in peach1970.DEPARTURE_COEFFICIENTS:
            # Apply Peach 1970 correction
            D_func = peach1970.DEPARTURE_COEFFICIENTS[species_name]

            # Compute σ = hν/(Ryd × Z²) in dimensionless units
            sigma = hplanck_eV * nu / (Rydberg_eV * charge**2)

            # Get departure coefficient
            D = D_func(T, sigma)

            # Hydrogenic absorption with correction
            alpha_hydrogenic = hydrogenic_ff_absorption(nu, T, charge, ndens, ne)
            alpha_total += alpha_hydrogenic * (1.0 + D)

        else:
            # Use uncorrected hydrogenic approximation
            # Accumulate by charge for efficiency
            if charge == 1:
                ndens_Z1 += ndens
            elif charge == 2:
                ndens_Z2 += ndens
            else:
                # For higher charges, would need to handle separately
                # For now, skip (not supported in original Julia code either)
                pass

    # Add contributions from accumulated species
    if ndens_Z1 > 0:
        alpha_total += hydrogenic_ff_absorption(nu, T, 1, ndens_Z1, ne)
    if ndens_Z2 > 0:
        alpha_total += hydrogenic_ff_absorption(nu, T, 2, ndens_Z2, ne)

    return alpha_total


def metal_bf_absorption(nu, T, number_densities):
    """
    Metal bound-free photoionization absorption from TOPBase and NORAD.

    Computes the total bound-free absorption coefficient from metal species
    using precomputed cross-section tables from TOPBase (Li-Ca) and NORAD (Fe).

    Parameters
    ----------
    nu : float or array
        Frequency in Hz
    T : float
        Temperature in K
    number_densities : dict
        Dictionary mapping species names (e.g., 'Fe_I', 'Ca_I') to number
        densities in cm⁻³.

    Returns
    -------
    float or array
        Linear absorption coefficient α in cm⁻¹

    Notes
    -----
    Cross-sections are precomputed for:
    - Temperature: 100 K < T < 100,000 K
    - Wavelength: 500 Å < λ < 30,000 Å

    Outside these ranges, flat extrapolation is used (extreme values are returned).

    Species included: Li I/II, Be I/II, B I/II, C I/II, N I/II, O I/II, F I/II,
    Ne I/II, Na I/II, Mg I/II, Al I/II, Si I/II, S I/II, Ar I/II, Ca I/II, Fe I/II.

    The tables assume an LTE distribution of energy levels.

    H I, He I, and H II are handled separately with more sophisticated methods
    and are skipped here.

    References
    ----------
    TOPBase: http://cdsweb.u-strasbg.fr/topbase/topbase.html
    NORAD: https://www.astronomy.ohio-state.edu/nahar.1/nahar_radiativeatomicdata/
    """
    nu = jnp.asarray(nu)

    # Lazy-load metal bf cross-section data
    bf_data = get_metal_bf_cross_sections()

    nu_grid = bf_data['nu_grid']
    logT_grid = bf_data['logT_grid']
    species_data = bf_data['species']

    # Initialize total absorption
    alpha_total = jnp.zeros_like(nu)

    # Temperature for interpolation
    logT = jnp.log10(T)

    for species_name, ndens in number_densities.items():
        # Skip if no data for this species
        if species_name not in species_data:
            continue

        # Get cross-section table for this species
        # Shape: (n_logT, n_nu), values are ln(σ in Mb)
        log_sigma_table = species_data[species_name]

        # Bilinear interpolation in (nu, logT) space
        # Find indices for frequency
        i_nu = jnp.searchsorted(nu_grid, nu, side='right') - 1
        i_nu = jnp.clip(i_nu, 0, len(nu_grid) - 2)

        # Find indices for log temperature
        i_logT = jnp.searchsorted(logT_grid, logT, side='right') - 1
        i_logT = jnp.clip(i_logT, 0, len(logT_grid) - 2)

        # Get grid points
        nu0 = nu_grid[i_nu]
        nu1 = nu_grid[i_nu + 1]
        logT0 = logT_grid[i_logT]
        logT1 = logT_grid[i_logT + 1]

        # Compute fractional positions
        t_nu = jnp.where(nu1 != nu0, (nu - nu0) / (nu1 - nu0), 0.0)
        t_logT = jnp.where(logT1 != logT0, (logT - logT0) / (logT1 - logT0), 0.0)

        # Get corner values from table (table is [logT, nu])
        log_s00 = log_sigma_table[i_logT, i_nu]
        log_s01 = log_sigma_table[i_logT, i_nu + 1]
        log_s10 = log_sigma_table[i_logT + 1, i_nu]
        log_s11 = log_sigma_table[i_logT + 1, i_nu + 1]

        # Bilinear interpolation in log space
        log_sigma_interp = ((1.0 - t_logT) * (1.0 - t_nu) * log_s00 +
                            (1.0 - t_logT) * t_nu * log_s01 +
                            t_logT * (1.0 - t_nu) * log_s10 +
                            t_logT * t_nu * log_s11)

        # Check for finite values (log_sigma = -inf when sigma = 0)
        is_finite = jnp.isfinite(log_sigma_interp)

        # Convert from ln(σ in Mb) to α in cm⁻¹
        # σ[cm²] = σ[Mb] × 10^-18
        # α = n × σ
        # To avoid NaN in derivatives when log_sigma = -inf:
        # α = exp(ln(n) + ln(σ in Mb) - ln(10^18))
        # α = exp(ln(n) + log_sigma - 18 × ln(10))
        log_alpha = jnp.log(ndens) + log_sigma_interp - 18.0 * jnp.log(10.0)

        # Only add contribution where cross-section is finite
        alpha_total += jnp.where(is_finite, jnp.exp(log_alpha), 0.0)

    return alpha_total


def total_continuum_absorption(nu, T, ne, number_densities, partition_funcs):
    """
    Total continuum linear absorption coefficient at given frequencies.

    Combines all continuum opacity sources including H, He, metal bound-free/free-free
    and scattering contributions.

    Parameters
    ----------
    nu : float or array
        Frequencies in Hz (should be sorted for efficiency)
    T : float
        Temperature in K
    ne : float
        Electron number density in cm⁻³
    number_densities : dict
        Dictionary mapping species names (e.g., 'H_I', 'He_I', 'Fe_I') to number
        densities in cm⁻³. Should include at least 'H_I', 'H_II', 'He_I', 'H2'.
    partition_funcs : dict
        Dictionary mapping species names to partition function values at this
        temperature.

    Returns
    -------
    float or array
        Linear absorption coefficient α in cm⁻¹

    Notes
    -----
    This function combines absorption from:
    - H I bound-free (with MHD level dissolution)
    - H⁻ bound-free and free-free
    - H₂⁺ bound-free and free-free
    - He⁻ free-free
    - Positive ion free-free (H II, He II, metals)
    - Metal bound-free (TOPBase/NORAD)
    - Electron scattering (Thomson)
    - Rayleigh scattering (H I, He I, H₂)

    The function is JAX-compatible and can be used with jax.jit.

    References
    ----------
    Korg.jl ContinuumAbsorption/ContinuumAbsorption.jl
    """
    # Convert to JAX arrays
    nu = jnp.asarray(nu)

    # Initialize total absorption
    alpha = jnp.zeros_like(nu)

    # Get commonly used number densities
    nH_I = number_densities.get('H_I', 0.0)
    nH_II = number_densities.get('H_II', 0.0)
    nHe_I = number_densities.get('He_I', 0.0)
    nH2 = number_densities.get('H2', 0.0)

    # Get partition function values by calling with log(T)
    # Partition functions are CubicSpline objects that take log(T) as input
    log_T = jnp.log(T)
    U_H_I = partition_funcs.get('H_I', lambda x: 1.0)(log_T)
    U_He_I = partition_funcs.get('He_I', lambda x: 1.0)(log_T)

    # Compute number density divided by partition function
    nH_I_div_U = nH_I / U_H_I
    nHe_I_div_U = nHe_I / U_He_I

    # Hydrogen continuum absorption
    # Note: inclusion of He I density is NOT a typo - it's used for MHD level dissolution
    alpha += H_I_bf(nu, T, nH_I, nHe_I, ne, 1.0 / U_H_I)
    alpha += Hminus_bf(nu, T, nH_I_div_U, ne)
    alpha += Hminus_ff(nu, T, nH_I_div_U, ne)
    alpha += H2plus_bf_and_ff(nu, T, nH_I, nH_II)

    # Helium continuum absorption
    alpha += Heminus_ff(nu, T, nHe_I_div_U, ne)

    # Free-free absorption from positive ions (H II, He II, metals)
    alpha += positive_ion_ff_absorption(nu, T, number_densities, ne)

    # Bound-free absorption by metals from TOPBase and NORAD
    alpha += metal_bf_absorption(nu, T, number_densities)

    # Scattering
    alpha += electron_scattering(ne)
    alpha += rayleigh(nu, nH_I, nHe_I, nH2)

    return alpha
