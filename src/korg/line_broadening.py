"""
Line broadening functions.

Functions for computing temperature-dependent broadening parameters.
"""

import jax.numpy as jnp
from jax.scipy.special import gamma as jax_gamma
from .constants import kboltz_cgs, c_cgs, electron_charge_cgs, electron_mass_cgs, amu_cgs


def sigma_line(wavelength):
    """
    Line cross-section divided by gf.

    The cross-section at wavelength λ in cm of a transition for which the
    product of the degeneracy and oscillator strength is 10^log_gf.

    Parameters
    ----------
    wavelength : float
        Wavelength in cm.

    Returns
    -------
    float
        Cross-section in cm².

    Notes
    -----
    The factor of λ²/c accounts for working in wavelength rather than frequency.
    """
    e = electron_charge_cgs
    m_e = electron_mass_cgs
    c = c_cgs

    # The factor of |dλ/dν| = λ²/c is because we are working in wavelength
    # rather than frequency
    return (jnp.pi * e**2 / m_e / c) * (wavelength**2 / c)


def scaled_stark(gamma_stark, T, T_0=10_000):
    """
    Stark broadening parameter scaled for temperature.

    Parameters
    ----------
    gamma_stark : float
        Stark broadening parameter at reference temperature.
    T : float
        Temperature in K.
    T_0 : float, optional
        Reference temperature in K (default: 10,000 K).

    Returns
    -------
    float
        Temperature-scaled Stark broadening parameter.

    Notes
    -----
    Uses temperature scaling: γ(T) = γ(T₀) * (T/T₀)^(1/6)
    """
    return gamma_stark * (T / T_0)**(1 / 6)


def scaled_vdW(vdW, m, T):
    """
    van der Waals broadening parameter scaled for temperature.

    The vdW broadening gamma scaled according to its temperature dependence,
    using either simple scaling or ABO (Anstee, Barklem, O'Mara).

    Parameters
    ----------
    vdW : tuple of float
        Either (γ_vdW, -1) for simple scaling, or (σ, α) for ABO parameters.
    m : float
        Species mass in grams (ignored for simple scaling).
    T : float
        Temperature in K.

    Returns
    -------
    float
        Temperature-scaled vdW broadening parameter.

    Notes
    -----
    For simple scaling (α = -1): γ(T) = γ(10000K) * (T/10000)^0.3

    For ABO (α != -1): Uses Anstee & O'Mara (1995) formulation.
    See Paul Barklem's notes: https://github.com/barklem/public-data/tree/master/broadening-howto

    References
    ----------
    - Anstee & O'Mara (1995)
    - Barklem's broadening notes
    """
    sigma, alpha = vdW

    if alpha == -1:
        # Simple scaling
        return sigma * (T / 10_000)**0.3
    else:
        # ABO formulation
        v_0 = 1e6  # σ is given at 10,000 m/s = 10^6 cm/s

        # Inverse reduced mass (H is perturber with mass 1.008 amu)
        inv_mu = 1 / (1.008 * amu_cgs) + 1 / m

        # Mean relative velocity
        vbar = jnp.sqrt(8 * kboltz_cgs * T / jnp.pi * inv_mu)

        # Note: "gamma" here is the gamma function, not a broadening parameter
        gamma_func = jax_gamma((4 - alpha) / 2)

        return 2 * (4 / jnp.pi)**(alpha / 2) * gamma_func * v_0 * sigma * (vbar / v_0)**(1 - alpha)
