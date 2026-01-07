"""
Line absorption calculation for atomic lines.

This module computes opacity from atomic spectral lines using Voigt profiles,
including proper treatment of Doppler and pressure broadening.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Callable, Tuple

from .linelist import Line
from .species import Species
from .constants import (
    c_cgs, electron_charge_cgs, electron_mass_cgs, hplanck_eV,
    kboltz_eV, kboltz_cgs, amu_cgs, bohr_radius_cgs
)
from .line_profiles import voigt_hjerting
from .atomic_data import atomic_masses
from scipy.special import gamma as gamma_function


def inverse_gaussian_density(rho: float, sigma: float) -> float:
    """
    Calculate the inverse of a (0-centered) Gaussian PDF with standard deviation sigma.

    Returns the value of x for which rho = exp(-0.5 x²/σ²) / √(2π),
    which is given by σ √[-2 log(√(2π)σρ)].

    Returns 0 when rho is larger than any value taken on by the PDF.

    Args:
        rho: Probability density value
        sigma: Standard deviation

    Returns:
        Distance from center where PDF equals rho, or 0 if rho is too large
    """
    max_density = 1.0 / (jnp.sqrt(2 * jnp.pi) * sigma)

    if rho > max_density:
        return 0.0
    else:
        return sigma * jnp.sqrt(-2 * jnp.log(jnp.sqrt(2 * jnp.pi) * sigma * rho))


def inverse_lorentz_density(rho: float, gamma: float) -> float:
    """
    Calculate the inverse of a (0-centered) Lorentz PDF with width gamma.

    Returns the value of x for which rho = 1 / (π γ (1 + x²/γ²)),
    which is given by √[γ/(πρ) - γ²].

    Returns 0 when rho is larger than any value taken on by the PDF.

    Args:
        rho: Probability density value
        gamma: Lorentz width (HWHM)

    Returns:
        Distance from center where PDF equals rho, or 0 if rho is too large
    """
    max_density = 1.0 / (jnp.pi * gamma)

    if rho > max_density:
        return 0.0
    else:
        return jnp.sqrt(gamma / (jnp.pi * rho) - gamma**2)


def sigma_line(wavelength: float) -> float:
    """
    Calculate the cross-section (divided by gf) at wavelength for a transition.

    This is the cross-section for which the product of the degeneracy and
    oscillator strength is 10^log_gf.

    Args:
        wavelength: Wavelength in cm

    Returns:
        Cross-section in cm²
    """
    # The factor of |dλ/dν| = λ²/c is because we are working in wavelength
    # rather than frequency
    return (jnp.pi * electron_charge_cgs**2 /
            (electron_mass_cgs * c_cgs)) * (wavelength**2 / c_cgs)


def doppler_width(wavelength: float, temperature: float, mass: float,
                  xi: float) -> float:
    """
    Calculate the standard deviation of the Doppler-broadening profile.

    Note: This is σ, not σ√2 as the "Doppler width" is often defined.

    Args:
        wavelength: Line center wavelength in cm
        temperature: Temperature in K
        mass: Atomic/molecular mass in g
        xi: Microturbulent velocity in cm/s

    Returns:
        Doppler width σ in cm
    """
    return wavelength * jnp.sqrt(kboltz_cgs * temperature / mass +
                                  (xi**2) / 2) / c_cgs


def scaled_stark(gamma_stark: float, temperature: float, T0: float = 10_000) -> float:
    """
    Scale the Stark broadening gamma according to its temperature dependence.

    Args:
        gamma_stark: Stark broadening parameter at T0 (rad/s)
        temperature: Temperature in K
        T0: Reference temperature (default: 10,000 K)

    Returns:
        Scaled Stark broadening parameter in rad/s
    """
    return gamma_stark * (temperature / T0)**(1/6)


def scaled_vdW(vdW: Tuple[float, float], mass: float,
               temperature: float) -> float:
    """
    Scale the van der Waals broadening gamma according to temperature dependence.

    Uses either simple scaling or ABO theory depending on the format of vdW.

    Args:
        vdW: Either (γ_vdW at 10,000 K, -1) or (σ, α) for ABO theory
        mass: Species mass in g
        temperature: Temperature in K

    Returns:
        Scaled van der Waals broadening parameter in rad/s
    """
    if vdW[1] == -1:
        # Simple scaling: γ_vdW ∝ T^0.3
        return vdW[0] * (temperature / 10_000)**0.3
    else:
        # ABO theory
        v0 = 1e6  # σ is given at 10,000 m/s = 10^6 cm/s
        sigma = vdW[0]
        alpha = vdW[1]

        # Inverse reduced mass: 1/μ = 1/m_H + 1/m_species
        inv_mu = 1 / (1.008 * amu_cgs) + 1 / mass

        # Mean relative velocity
        vbar = jnp.sqrt(8 * kboltz_cgs * temperature / jnp.pi * inv_mu)

        # ABO formula (n.b. gamma here is the gamma function, not broadening)
        return (2 * (4 / jnp.pi)**(alpha / 2) *
                gamma_function((4 - alpha) / 2) *
                v0 * sigma * (vbar / v0)**(1 - alpha))


def line_profile(wavelength_center: float, sigma: float, gamma: float,
                amplitude: float, wavelength: float) -> float:
    """
    Calculate a Voigt profile centered on wavelength_center.

    Args:
        wavelength_center: Line center in cm
        sigma: Doppler width (NOT √2 σ) in cm
        gamma: Lorentz HWHM in cm
        amplitude: Total integrated absorption coefficient
        wavelength: Wavelength at which to evaluate in cm

    Returns:
        Absorption coefficient in cm⁻¹
    """
    inv_sigma_sqrt2 = 1 / (sigma * jnp.sqrt(2))
    scaling = inv_sigma_sqrt2 / jnp.sqrt(jnp.pi) * amplitude

    # Call voigt_hjerting with proper arguments
    return (voigt_hjerting(gamma * inv_sigma_sqrt2,
                          jnp.abs(wavelength - wavelength_center) * inv_sigma_sqrt2) *
            scaling)


def line_absorption(
    linelist: List[Line],
    wavelengths: np.ndarray,
    temperatures: np.ndarray,
    electron_densities: np.ndarray,
    number_densities: Dict[Species, np.ndarray],
    partition_functions: Dict[Species, Callable],
    xi: float,
    continuum_opacity: Callable[[float], np.ndarray],
    cutoff_threshold: float = 3e-4
) -> np.ndarray:
    """
    Calculate the opacity coefficient from all lines in linelist.

    This is the main function for computing line absorption. It handles:
    - Voigt profile calculation for each line
    - Temperature-dependent broadening (Doppler, Stark, van der Waals)
    - Line cutoff based on continuum opacity
    - Boltzmann populations

    Args:
        linelist: List of Line objects to include
        wavelengths: Wavelength grid in cm (1D array)
        temperatures: Temperature at each layer in K (1D array)
        electron_densities: Electron number density in cm⁻³ at each layer
        number_densities: Dict mapping Species to number densities (cm⁻³) at each layer
        partition_functions: Dict mapping Species to partition function callables
        xi: Microturbulent velocity in cm/s
        continuum_opacity: Callable that takes wavelength (cm) and returns continuum
                          opacity at each layer (returns 1D array)
        cutoff_threshold: Lines contribute opacity only where significant
                         (default: 3e-4 of continuum)

    Returns:
        Absorption coefficient array of shape (n_layers, n_wavelengths) in cm⁻¹
    """
    if len(linelist) == 0:
        return jnp.zeros((len(temperatures), len(wavelengths)))

    # Precompute beta = 1/(kT) for Boltzmann factors
    beta = 1 / (kboltz_eV * temperatures)

    # Precompute number density / partition function for each species
    n_div_U = {}
    unique_species = list(set([line.species for line in linelist]))

    for spec in unique_species:
        if spec not in number_densities:
            raise ValueError(f"Species {spec} in linelist but not in number_densities")

        # Partition functions take log(T)
        log_temps = jnp.log(temperatures)
        U_values = jnp.array([partition_functions[spec](lt) for lt in log_temps])
        n_div_U[spec] = number_densities[spec] / U_values

    # Check for H I in linelist (should use hydrogen_line_absorption instead)
    if Species("H_I") in unique_species:
        raise ValueError("Atomic hydrogen should not be in the linelist. "
                        "Use hydrogen_line_absorption for H lines.")

    # Initialize absorption coefficient array
    alpha = jnp.zeros((len(temperatures), len(wavelengths)))

    # Process each line
    for line in linelist:
        mass = line.species.get_mass()

        # Doppler broadening width (σ, NOT √2 σ)
        sigma_vals = jnp.array([doppler_width(line.wl, T, mass, xi)
                                for T in temperatures])

        # Sum up damping parameters (these are FWHM in angular frequency)
        Gamma = jnp.full_like(temperatures, line.gamma_rad)

        # Add Stark and van der Waals for atoms (not molecules)
        if not line.species.formula.is_molecule():
            # Stark broadening
            Gamma = Gamma + electron_densities * jnp.array([
                scaled_stark(line.gamma_stark, T) for T in temperatures
            ])

            # van der Waals broadening (need H I density)
            if Species("H_I") in number_densities:
                Gamma = Gamma + number_densities[Species("H_I")] * jnp.array([
                    scaled_vdW(line.vdW, mass, T) for T in temperatures
                ])

        # Convert to Lorentz broadening parameter in wavelength units
        # Factor of λ²/c is |dλ/dν|, 1/(2π) for angular vs cyclical frequency,
        # and 1/2 for FWHM vs HWHM
        gamma_vals = Gamma * line.wl**2 / (c_cgs * 4 * jnp.pi)

        # Calculate energy levels and Boltzmann factor
        E_upper = line.E_lower + c_cgs * hplanck_eV / line.wl
        levels_factor = jnp.exp(-beta * line.E_lower) - jnp.exp(-beta * E_upper)

        # Total wavelength-integrated absorption coefficient
        amplitude = (10.0**line.log_gf * sigma_line(line.wl) *
                    levels_factor * n_div_U[line.species])

        # Determine line window based on cutoff threshold
        # Get continuum opacity at line center
        alpha_continuum_at_line = continuum_opacity(line.wl)

        # Critical density for cutoff
        rho_crit = alpha_continuum_at_line * cutoff_threshold / amplitude

        # Doppler wing extent
        inverse_densities_doppler = jnp.array([
            inverse_gaussian_density(rc, s)
            for rc, s in zip(rho_crit, sigma_vals)
        ])
        doppler_line_window = jnp.max(inverse_densities_doppler)

        # Lorentz wing extent
        inverse_densities_lorentz = jnp.array([
            inverse_lorentz_density(rc, g)
            for rc, g in zip(rho_crit, gamma_vals)
        ])
        lorentz_line_window = jnp.max(inverse_densities_lorentz)

        # Combined window (Pythagorean sum)
        window_size = jnp.sqrt(lorentz_line_window**2 + doppler_line_window**2)

        # Find wavelength indices in window
        lb = jnp.searchsorted(wavelengths, line.wl - window_size)
        ub = jnp.searchsorted(wavelengths, line.wl + window_size, side='right')

        # Skip if window is empty
        if lb >= ub:
            continue

        # Calculate line profile for all layers and wavelengths in window
        wl_window = wavelengths[lb:ub]

        # Vectorized calculation over layers and wavelengths
        for i_layer in range(len(temperatures)):
            profiles = jnp.array([
                line_profile(line.wl, sigma_vals[i_layer], gamma_vals[i_layer],
                           amplitude[i_layer], wl)
                for wl in wl_window
            ])
            alpha = alpha.at[i_layer, lb:ub].add(profiles)

    return alpha
