"""
Line profile and broadening functions.

Functions for computing line profiles, including Voigt profiles,
Doppler broadening, and related utilities.
"""

import jax
import jax.numpy as jnp
from .constants import (kboltz_cgs, c_cgs)

@jax.jit
def doppler_width(wavelength, T, m, xi):
    """
    Standard deviation of the Doppler broadening profile.

    Note: This returns σ, not σ√2 as often defined as "Doppler width" in texts.

    Parameters
    ----------
    wavelength : float
        Central wavelength (λ₀) in cm.
    T : float
        Temperature in K.
    m : float
        Particle mass in grams.
    xi : float
        Microturbulent velocity in cm/s.

    Returns
    -------
    float
        Doppler width σ in cm.
    """
    return wavelength * jnp.sqrt(kboltz_cgs * T / m + xi**2 / 2) / c_cgs

@jax.jit
def inverse_gaussian_density(rho, sigma):
    """
    Inverse of a zero-centered Gaussian PDF.

    Calculate the value of x for which ρ = exp(-0.5 x²/σ²) / √(2π)σ,
    which is given by σ √[-2 log(√(2π)σρ)]. Returns 0 when ρ is larger
    than any value taken on by the PDF.

    Parameters
    ----------
    rho : float
        Density value.
    sigma : float
        Standard deviation.

    Returns
    -------
    float
        The x value corresponding to the density ρ, or 0 if ρ > max(PDF).
    """
    max_density = 1 / (jnp.sqrt(2 * jnp.pi) * sigma)
    return jnp.where(
        rho > max_density,
        0.0,
        sigma * jnp.sqrt(-2 * jnp.log(jnp.sqrt(2 * jnp.pi) * sigma * rho))
    )

@jax.jit
def inverse_lorentz_density(rho, gamma):
    """
    Inverse of a zero-centered Lorentz PDF.

    Calculate the value of x for which ρ = 1 / (π γ (1 + x²/γ²)),
    which is given by √[γ/(πρ) - γ²]. Returns 0 when ρ is larger than
    any value taken on by the PDF.

    Parameters
    ----------
    rho : float
        Density value.
    gamma : float
        Lorentz width parameter.

    Returns
    -------
    float
        The x value corresponding to the density ρ, or 0 if ρ > max(PDF).
    """
    max_density = 1 / (jnp.pi * gamma)
    return jnp.where(
        rho > max_density,
        0.0,
        jnp.sqrt(gamma / (jnp.pi * rho) - gamma**2)
    )

@jax.jit
def exponential_integral_1(x):
    """
    First exponential integral, E1(x).

    This is a rough approximation from Kurucz's VCSE1F.
    Used in Brackett line profile calculations.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float
        E1(x) value.
    """
    # Piecewise approximation
    def small_x(x):
        # x <= 0.01
        return -jnp.log(x) - 0.577215 + x

    def medium_x(x):
        # 0.01 < x <= 1.0
        return (-jnp.log(x) - 0.57721566 +
                x * (0.99999193 + x * (-0.24991055 + x * (0.05519968 +
                x * (-0.00976004 + x * 0.00107857)))))

    def large_x(x):
        # 1.0 < x <= 30.0
        return ((x * (x + 2.334733) + 0.25062) /
                (x * (x + 3.330657) + 1.681534) / x * jnp.exp(-x))

    # Use jnp.where for conditional evaluation
    result = jnp.where(
        x < 0,
        0.0,
        jnp.where(
            x <= 0.01,
            small_x(x),
            jnp.where(
                x <= 1.0,
                medium_x(x),
                jnp.where(
                    x <= 30.0,
                    large_x(x),
                    0.0
                )
            )
        )
    )
    return result

@jax.jit
def harris_series(v):
    """
    Harris series for Voigt function computation.

    Helper function for voigt_hjerting. Assumes v < 5.

    Parameters
    ----------
    v : float
        Input parameter (must be < 5).

    Returns
    -------
    tuple of float
        (H₀, H₁, H₂) values.
    """
    v2 = v * v
    H0 = jnp.exp(-v2)

    # Piecewise definition of H1
    def h1_small(v):
        # v < 1.3
        return (-1.12470432 + (-0.15516677 + (3.288675912 +
                (-2.34357915 + 0.42139162 * v) * v) * v) * v)

    def h1_medium(v):
        # 1.3 <= v < 2.4
        return (-4.48480194 + (9.39456063 + (-6.61487486 +
                (1.98919585 - 0.22041650 * v) * v) * v) * v)

    def h1_large(v):
        # 2.4 <= v < 5
        return ((0.554153432 + (0.278711796 + (-0.1883256872 +
                (0.042991293 - 0.003278278 * v) * v) * v) * v) /
                (v2 - 3 / 2))

    H1 = jnp.where(
        v < 1.3,
        h1_small(v),
        jnp.where(
            v < 2.4,
            h1_medium(v),
            h1_large(v)
        )
    )

    H2 = (1 - 2 * v2) * H0

    return H0, H1, H2

@jax.jit
def voigt_hjerting(alpha, v):
    """
    The Hjerting function H(α, v).

    Sometimes called the Voigt-Hjerting function. Defined as:
    H(α, v) = ∫_{-∞}^{∞} exp(-y²) / ((v-y)² + α²) dy

    Approximation from Hunger 1965.

    Parameters
    ----------
    alpha : float
        Lorentz parameter.
    v : float
        Detuning parameter.

    Returns
    -------
    float
        H(α, v) value.

    Notes
    -----
    If x = λ-λ₀, Δλ_D = σ√2 is the Doppler width, and Δλ_L = 4πγ is the Lorentz width:

    voigt(x|Δλ_D, Δλ_L) = H(Δλ_L/(4πΔλ_D), x/Δλ_D) / (Δλ_D√π)
                        = H(γ/(σ√2), x/(σ√2)) / (σ√(2π))
    """
    v2 = v * v

    # Case 1: α <= 0.2 and v >= 5
    def case1(alpha, v, v2):
        # Safe division - if v2 is 0, this shouldn't be called anyway
        invv2 = jnp.where(v2 == 0, 1e10, 1 / jnp.maximum(v2, 1e-20))
        return (alpha / jnp.sqrt(jnp.pi) * invv2) * (1 + 1.5 * invv2 + 3.75 * invv2**2)

    # Case 2: α <= 0.2 and v < 5
    def case2(alpha, v):
        H0, H1, H2 = harris_series(v)
        return H0 + (H1 + H2 * alpha) * alpha

    # Case 3: α <= 1.4 and α + v < 3.2
    def case3(alpha, v, v2):
        H0, H1, H2 = harris_series(v)
        M0 = H0
        M1 = H1 + 2 / jnp.sqrt(jnp.pi) * M0
        M2 = H2 - M0 + 2 / jnp.sqrt(jnp.pi) * M1
        M3 = 2 / (3 * jnp.sqrt(jnp.pi)) * (1 - H2) - (2 / 3) * v2 * M1 + (2 / jnp.sqrt(jnp.pi)) * M2
        M4 = 2 / 3 * v2 * v2 * M0 - 2 / (3 * jnp.sqrt(jnp.pi)) * M1 + 2 / jnp.sqrt(jnp.pi) * M3
        psi = 0.979895023 + (-0.962846325 + (0.532770573 - 0.122727278 * alpha) * alpha) * alpha
        return psi * (M0 + (M1 + (M2 + (M3 + M4 * alpha) * alpha) * alpha) * alpha)

    # Case 4: α > 1.4 or (α > 0.2 and α + v > 3.2)
    def case4(alpha, v, v2):
        r2 = v2 / (alpha * alpha)
        alpha_invu = 1 / jnp.sqrt(2) / ((r2 + 1) * alpha)
        alpha2_invu2 = alpha_invu * alpha_invu
        return (jnp.sqrt(2 / jnp.pi) * alpha_invu *
                (1 + (3 * r2 - 1 + ((r2 - 2) * 15 * r2 + 2) * alpha2_invu2) * alpha2_invu2))

    # Apply conditions
    result = jnp.where(
        (alpha <= 0.2) & (v >= 5),
        case1(alpha, v, v2),
        jnp.where(
            alpha <= 0.2,
            case2(alpha, v),
            jnp.where(
                (alpha <= 1.4) & (alpha + v < 3.2),
                case3(alpha, v, v2),
                case4(alpha, v, v2)
            )
        )
    )

    return result

@jax.jit
def line_profile(wavelength_0, sigma, gamma, amplitude, wavelength):
    """
    Voigt profile for a spectral line.

    Parameters
    ----------
    wavelength_0 : float
        Line center wavelength in cm.
    sigma : float
        Doppler width (σ, NOT √2σ) in cm.
    gamma : float
        Lorentz HWHM in cm.
    amplitude : float
        Line amplitude (integrated absorption coefficient).
    wavelength : float
        Wavelength at which to evaluate the profile in cm.

    Returns
    -------
    float
        Absorption coefficient in cm⁻¹.
    """
    inv_sigma_sqrt2 = 1 / (sigma * jnp.sqrt(2))
    scaling = inv_sigma_sqrt2 / jnp.sqrt(jnp.pi) * amplitude
    return voigt_hjerting(gamma * inv_sigma_sqrt2,
                         jnp.abs(wavelength - wavelength_0) * inv_sigma_sqrt2) * scaling
