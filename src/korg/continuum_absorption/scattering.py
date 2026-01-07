"""
Scattering opacity sources for continuum absorption.

Includes Rayleigh scattering (H, He, H₂) and electron (Thomson) scattering.
"""

import jax.numpy as jnp
import numpy as np
from typing import Union

# Import constants
from ..constants import (
    c_cgs, hplanck_eV, Rydberg_eV,
    electron_charge_cgs, electron_mass_cgs
)


def rayleigh(nus: Union[np.ndarray, jnp.ndarray],
             nH_I: float, nHe_I: float, nH2: float) -> np.ndarray:
    """
    Absorption coefficient from Rayleigh scattering by neutral H, He, and H₂.

    Formulations for H and He are via Colgan+ 2016 (ApJ 817, 116).
    Formulation for H₂ from Dalgarno & Williams 1962 (ApJ 136, 690).

    The Dalgarno & Williams H₂ formula is applicable redward of 1300 Å.
    Since Rayleigh scattering breaks down when the particle size to
    wavelength ratio gets large, we require that all frequencies passed
    correspond to 1300 Å or greater.

    The formulation for H is adapted from Lee 2005, which states that it is
    applicable redward of Lyman alpha. See Colgan 2016 for details on He.

    Parameters
    ----------
    nus : array
        Frequencies in Hz (sorted)
    nH_I : float
        Number density of H I [cm⁻³]
    nHe_I : float
        Number density of He I [cm⁻³]
    nH2 : float
        Number density of H₂ [cm⁻³]

    Returns
    -------
    alpha : array
        Linear absorption coefficient [cm⁻¹]
    """
    # Ensure minimum frequency (corresponding to 1300 Å)
    assert c_cgs / jnp.max(nus) > 1.3e-5, "Frequencies must correspond to λ ≥ 1300 Å"

    # Thomson scattering cross section [cm²]
    sigma_th = 6.65246e-25

    # (ħω / 2E_H)^2 in Colgan+ 2016. The photon energy over 2 Ryd
    E_2Ryd_2 = (hplanck_eV * nus / (2 * Rydberg_eV))**2
    E_2Ryd_4 = E_2Ryd_2**2
    E_2Ryd_6 = E_2Ryd_2 * E_2Ryd_4
    E_2Ryd_8 = E_2Ryd_4**2

    # Colgan+ 2016 equation 6 (H)
    sigma_H_over_sigma_th = 20.24 * E_2Ryd_4 + 239.2 * E_2Ryd_6 + 2256 * E_2Ryd_8

    # Colgan+ 2016 equation 7 (He)
    sigma_He_over_sigma_th = 1.913 * E_2Ryd_4 + 4.52 * E_2Ryd_6 + 7.90 * E_2Ryd_8

    alpha_H_He = (nH_I * sigma_H_over_sigma_th + nHe_I * sigma_He_over_sigma_th) * sigma_th

    # Dalgarno & Williams 1962 equation 3 (assumes λ in Å)
    inv_lambda_2 = (nus / (1e8 * c_cgs))**2
    inv_lambda_4 = inv_lambda_2**2
    inv_lambda_6 = inv_lambda_2 * inv_lambda_4
    inv_lambda_8 = inv_lambda_4**2

    alpha_H2 = (8.14e-13 * inv_lambda_4 + 1.28e-6 * inv_lambda_6 + 1.61 * inv_lambda_8) * nH2

    return alpha_H_He + alpha_H2


def electron_scattering(ne: float) -> float:
    """
    Compute the linear absorption coefficient from electron scattering.

    This is Thomson scattering off free electrons. It has no wavelength
    dependence and assumes isotropic scattering. (See Gray p 160.)

    Parameters
    ----------
    ne : float
        Number density of free electrons [cm⁻³]

    Returns
    -------
    alpha : float
        Linear absorption coefficient [cm⁻¹]
    """
    # 8π/3 * (e²/(m*c²))² * ne
    # where e²/(m*c²) is the classical electron radius
    return (8 * jnp.pi / 3) * (electron_charge_cgs**2 / (electron_mass_cgs * c_cgs**2))**2 * ne
