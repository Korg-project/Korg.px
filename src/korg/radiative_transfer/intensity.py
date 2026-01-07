"""
Intensity calculations for radiative transfer.

Implements various schemes for computing emergent intensity from optical depth
and source function profiles. The main schemes are:

1. Linear interpolation (fast, accurate)
2. Bezier interpolation (more accurate, slower)
3. Exponential integral optimization (fastest for flux-only)

Reference: Korg.jl RadiativeTransfer module
"""

import jax.numpy as jnp
from jax import jit
from .expint import exponential_integral_2


def compute_I_linear_flux_only(tau, S):
    """
    Compute emergent intensity using linear interpolation in τ-S space.

    This is the "linear_flux_only" scheme from Korg.jl - fast and accurate
    for flux calculations (angle-integrated intensity).

    Assumes source function varies linearly between layers:
    S(τ) = S[i] + (S[i+1] - S[i]) / Δτ * (τ - τ[i])

    The emergent intensity is:
    I = ∫₀^∞ S(τ) exp(-τ) dτ

    Parameters
    ----------
    tau : array, shape (n_layers,)
        Optical depth at each layer
        Must be monotonically increasing from surface (tau[0]=0) to depth
    S : array, shape (n_layers,)
        Source function at each layer [erg cm⁻² s⁻¹ sr⁻¹ Hz⁻¹]
        Typically Planck function B_ν(T)

    Returns
    -------
    I : float
        Emergent intensity [erg cm⁻² s⁻¹ sr⁻¹ Hz⁻¹]

    Notes
    -----
    The integration uses the analytic solution for linear S(τ):
    ∫ exp(-τ) * (a + b*τ) dτ = -exp(-τ) * (a + b*(1+τ))

    This is exact for piecewise linear S(τ) and very fast.
    """
    # Vectorized implementation
    tau_i = tau[:-1]  # Current layer
    tau_ip1 = tau[1:]  # Next layer
    S_i = S[:-1]
    S_ip1 = S[1:]

    # Linear interpolation slope
    delta_tau = tau_ip1 - tau_i
    m = (S_ip1 - S_i) / delta_tau

    # Exponentials
    exp_tau_i = jnp.exp(-tau_i)
    exp_tau_ip1 = jnp.exp(-tau_ip1)

    # Coefficients for linear fit
    a = S_i - m * tau_i
    b = m

    # Analytic integral for each layer
    contrib = (-exp_tau_ip1 * (a + b * (1.0 + tau_ip1)) +
               exp_tau_i * (a + b * (1.0 + tau_i)))

    # Sum contributions
    I = jnp.sum(contrib)

    return I


def expint_transfer_integral_core(tau, m, b):
    """
    Core antiderivative for flux calculation with linear source function.

    This computes the exact solution to ∫ (mτ + b) E₂(τ) dτ.

    Parameters
    ----------
    tau : float or array
        Optical depth
    m : float or array
        Slope of linear source function S(τ) = mτ + b
    b : float or array
        Intercept of linear source function

    Returns
    -------
    float or array
        Antiderivative value

    Notes
    -----
    From Julia's RadiativeTransfer.jl:
    1/6 * (τ * E₂(τ) * (3b + 2m*τ) - exp(-τ) * (3b + 2m*(τ+1)))
    """
    E2 = exponential_integral_2(tau)
    return (1.0 / 6.0) * (tau * E2 * (3.0 * b + 2.0 * m * tau) -
                          jnp.exp(-tau) * (3.0 * b + 2.0 * m * (tau + 1.0)))


def compute_F_flux_only_expint(tau, S):
    """
    Compute emergent flux using exponential integrals.

    This is an optimized version for computing flux (angle-integrated
    intensity) using exponential integral functions. Faster than explicit
    angle integration for plane-parallel atmospheres.

    The flux is:
    F = 2π ∫₀¹ I(μ) μ dμ

    where μ = cos(θ). For plane-parallel atmospheres with linear S(τ),
    this uses the expint_transfer_integral_core formula from Julia.

    Parameters
    ----------
    tau : array, shape (n_layers,)
        Optical depth at each layer (increasing into atmosphere)
    S : array, shape (n_layers,)
        Source function at each layer [erg cm⁻² s⁻¹ sr⁻¹ Hz⁻¹]

    Returns
    -------
    F : float
        Emergent flux [erg cm⁻² s⁻¹ Hz⁻¹]

    Notes
    -----
    For piecewise linear S(τ) = m*τ + b in each layer:
    F = Σᵢ (expint_core(τᵢ₊₁, m, b) - expint_core(τᵢ, m, b))

    This matches Julia's compute_F_flux_only_expint exactly.
    Note: The factor of 2π is applied by radiative_transfer()
    when integrating over angles (F = 2π * ∫ I μ dμ).
    """
    # Vectorized implementation
    tau_i = tau[:-1]
    tau_ip1 = tau[1:]
    S_i = S[:-1]
    S_ip1 = S[1:]

    # Linear interpolation: S(τ) = m*τ + b
    delta_tau = tau_ip1 - tau_i
    m = (S_ip1 - S_i) / delta_tau
    b = S_i - m * tau_i

    # Compute flux contributions using antiderivative
    # F = Σᵢ [F(τᵢ₊₁) - F(τᵢ)] where F is the antiderivative
    core_i = expint_transfer_integral_core(tau_i, m, b)
    core_ip1 = expint_transfer_integral_core(tau_ip1, m, b)

    contrib = core_ip1 - core_i

    # Sum contributions
    F = jnp.sum(contrib)

    return F


@jit
def compute_I_linear(tau, S, mu):
    """
    Compute emergent intensity at specific angle μ using linear interpolation.

    This computes intensity along a ray at angle θ (where μ = cos θ) from
    the vertical. More general than flux-only version.

    Parameters
    ----------
    tau : array, shape (n_layers,)
        Optical depth at each layer (for μ=1, vertical)
    S : array, shape (n_layers,)
        Source function at each layer
    mu : float
        Cosine of angle from vertical (0 < μ ≤ 1)
        μ = 1: vertical (disk center)
        μ → 0: horizontal (limb)

    Returns
    -------
    I : float
        Emergent intensity at angle μ

    Notes
    -----
    The slant optical depth τ_μ = τ / μ, so the emergent intensity is:
    I(μ) = ∫₀^∞ S(τ) exp(-τ/μ) dτ/μ

    For linear S(τ), this has an analytic solution.
    """
    n_layers = len(tau)
    I = 0.0

    # Scale optical depth by μ for slant path
    tau_mu = tau / mu

    for i in range(n_layers - 1):
        tau_i = tau_mu[i]
        tau_ip1 = tau_mu[i + 1]
        S_i = S[i]
        S_ip1 = S[i + 1]

        delta_tau = tau_ip1 - tau_i

        if delta_tau <= 0:
            continue

        m = (S_ip1 - S_i) / delta_tau

        exp_tau_i = jnp.exp(-tau_i)
        exp_tau_ip1 = jnp.exp(-tau_ip1)

        a = S_i - m * tau_i
        b = m

        contrib = (-exp_tau_ip1 * (a + b * (1.0 + tau_ip1)) +
                   exp_tau_i * (a + b * (1.0 + tau_i)))

        I += contrib

    return I


@jit
def compute_flux_from_intensities(intensities, mu_points, mu_weights):
    """
    Compute flux from intensities at discrete angles.

    Performs Gaussian quadrature integration:
    F = 2π ∫₀¹ I(μ) μ dμ ≈ 2π Σᵢ wᵢ I(μᵢ) μᵢ

    Parameters
    ----------
    intensities : array, shape (n_mu,)
        Intensity at each angle point
    mu_points : array, shape (n_mu,)
        Quadrature points (μ = cos θ values)
    mu_weights : array, shape (n_mu,)
        Quadrature weights

    Returns
    -------
    F : float
        Emergent flux
    """
    # Flux = 2π ∫ I(μ) μ dμ
    F = 2.0 * jnp.pi * jnp.sum(mu_weights * intensities * mu_points)
    return F
