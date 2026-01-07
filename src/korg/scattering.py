"""
Scattering functions for continuum absorption.

Implements electron scattering and Rayleigh scattering.
"""

import jax.numpy as jnp
from .constants import (c_cgs, hplanck_eV, electron_charge_cgs,
                        electron_mass_cgs, Rydberg_eV)


def electron_scattering(n_e):
    """
    Thomson scattering linear absorption coefficient.

    Compute the linear absorption coefficient from scattering off free electrons.
    This has no wavelength dependence and assumes isotropic scattering.

    Parameters
    ----------
    n_e : float
        Number density of free electrons in cm⁻³.

    Returns
    -------
    float
        Linear absorption coefficient α in cm⁻¹.

    Notes
    -----
    The formula is: α = (8π/3) * σ_T * n_e
    where σ_T = (e²/(m_e c²))² is the Thomson cross section.
    """
    # Thomson cross section formula: (e²/(m_e c²))²
    sigma_T = (electron_charge_cgs**2 / (electron_mass_cgs * c_cgs**2))**2
    return 8 * jnp.pi / 3 * sigma_T * n_e


def rayleigh(frequencies, n_H_I, n_He_I, n_H2):
    """
    Rayleigh scattering absorption coefficient.

    Absorption coefficient from Rayleigh scattering by neutral H, He, and H₂.
    Formulations for H and He from Colgan+ 2016.
    Formulation for H₂ from Dalgarno & Williams 1962.

    Parameters
    ----------
    frequencies : array
        Frequencies in Hz. Must correspond to wavelengths >= 1300 Å.
    n_H_I : float
        Number density of neutral hydrogen in cm⁻³.
    n_He_I : float
        Number density of neutral helium in cm⁻³.
    n_H2 : float
        Number density of molecular hydrogen in cm⁻³.

    Returns
    -------
    array
        Linear absorption coefficient α in cm⁻¹.

    Notes
    -----
    The Dalgarno & Williams H₂ formula is applicable redward of 1300 Å.
    The H formula from Lee 2005/Colgan 2016 is applicable redward of Lyman α.

    References
    ----------
    - Colgan+ 2016: https://ui.adsabs.harvard.edu/abs/2016ApJ...817..116C
    - Dalgarno & Williams 1962: https://ui.adsabs.harvard.edu/abs/1962ApJ...136..690D
    - Lee 2005: Applicable redward of Lyman α
    """
    # Thomson scattering cross section
    sigma_th = 6.65246e-25  # cm²

    # (ℏω / 2E_H)² in Colgan+ 2016
    # The photon energy over 2 Rydbergs
    E_2Ryd_2 = (hplanck_eV * frequencies / (2 * Rydberg_eV))**2
    E_2Ryd_4 = E_2Ryd_2**2
    E_2Ryd_6 = E_2Ryd_2 * E_2Ryd_4
    E_2Ryd_8 = E_2Ryd_4**2

    # Colgan+ 2016 equation 6 (H)
    sigma_H_over_sigma_th = 20.24 * E_2Ryd_4 + 239.2 * E_2Ryd_6 + 2256 * E_2Ryd_8

    # Colgan+ 2016 equation 7 (He)
    sigma_He_over_sigma_th = 1.913 * E_2Ryd_4 + 4.52 * E_2Ryd_6 + 7.90 * E_2Ryd_8

    # Combined H and He absorption
    alpha_H_He = (n_H_I * sigma_H_over_sigma_th + n_He_I * sigma_He_over_sigma_th) * sigma_th

    # Dalgarno & Williams 1962 equation 3
    # Assumes wavelength in Ångstroms
    inv_lambda_angstrom_2 = (frequencies / (1e8 * c_cgs))**2
    inv_lambda_4 = inv_lambda_angstrom_2**2
    inv_lambda_6 = inv_lambda_angstrom_2 * inv_lambda_4
    inv_lambda_8 = inv_lambda_4**2

    alpha_H2 = (8.14e-13 * inv_lambda_4 + 1.28e-6 * inv_lambda_6 + 1.61 * inv_lambda_8) * n_H2

    return alpha_H_He + alpha_H2
