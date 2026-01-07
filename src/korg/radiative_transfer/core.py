"""
Core radiative transfer solver.

Main orchestration function that combines optical depth calculation,
intensity integration, and flux computation.

Reference: Korg.jl RadiativeTransfer module
"""

import jax.numpy as jnp
from jax import jit
import numpy as np

from .optical_depth import compute_tau_anchored, compute_tau_direct
from .intensity import (compute_I_linear_flux_only, compute_F_flux_only_expint,
                         compute_I_linear, compute_flux_from_intensities)


def generate_mu_grid(n_mu=5):
    """
    Generate Gaussian quadrature points and weights for angle integration.

    Returns μ = cos(θ) points and weights for integrating over solid angle.

    Parameters
    ----------
    n_mu : int, optional
        Number of quadrature points (default: 5)
        More points = better accuracy but slower

    Returns
    -------
    mu_points : array, shape (n_mu,)
        Quadrature points in [0, 1]
    mu_weights : array, shape (n_mu,)
        Quadrature weights (sum to 1)

    Notes
    -----
    Uses Gauss-Legendre quadrature on [0, 1] interval.
    For n_mu=5, typical error in flux is < 0.1%.
    """
    # Get Gauss-Legendre quadrature on [-1, 1]
    # Then transform to [0, 1]
    from numpy.polynomial.legendre import leggauss

    points, weights = leggauss(n_mu)

    # Transform from [-1, 1] to [0, 1]
    mu_points = 0.5 * (points + 1.0)
    mu_weights = 0.5 * weights  # Jacobian factor

    return mu_points, mu_weights


def radiative_transfer_single_wavelength(
    alpha,
    S,
    spatial_coord,
    log_tau_ref,
    alpha_ref=None,
    spherical=False,
    intensity_scheme="linear_flux_only",
    use_expint_flux=True,
    n_mu=5
):
    """
    Solve radiative transfer equation at a single wavelength.

    Computes emergent flux and optionally intensity profile from absorption
    coefficient and source function throughout the atmosphere.

    Parameters
    ----------
    alpha : array, shape (n_layers,)
        Total absorption coefficient [cm⁻¹] at each atmospheric layer
    S : array, shape (n_layers,)
        Source function [erg cm⁻² s⁻¹ sr⁻¹ Hz⁻¹] at each layer
        For LTE: S = B_ν(T), the Planck function
    spatial_coord : array, shape (n_layers,)
        Spatial coordinate [cm] at each layer
        For planar: height above photosphere
        For spherical: radius from stellar center
    log_tau_ref : array, shape (n_layers,)
        log₁₀(optical depth) at reference wavelength (typically 5000 Å)
        Used for anchored optical depth calculation
    alpha_ref : array, shape (n_layers,), optional
        Absorption coefficient at reference wavelength [cm⁻¹]
        Required for anchored optical depth scheme.
        If None, uses tau_ref directly (assumes alpha ≈ alpha_ref)
    spherical : bool, optional
        If True, use spherical geometry
        If False, use plane-parallel geometry (default)
    intensity_scheme : str, optional
        Method for computing intensity:
        - "linear_flux_only": Fast, flux only (default)
        - "linear": Linear interpolation with angle integration
        - "bezier": Bezier interpolation (not yet implemented)
    use_expint_flux : bool, optional
        If True and intensity_scheme="linear_flux_only", use exponential
        integral optimization for flux (default: True)
    n_mu : int, optional
        Number of angle points for quadrature (default: 5)
        Only used if intensity_scheme requires angle integration

    Returns
    -------
    flux : float
        Emergent flux [erg cm⁻² s⁻¹ Hz⁻¹]
    intensity : array or None
        Emergent intensity at each μ point (if intensity_scheme computes it)
        Otherwise None

    Notes
    -----
    The radiative transfer equation in the τ coordinate is:
    dI/dτ = I - S

    With boundary condition I(τ=0) = 0 (no incoming radiation), the
    formal solution is:
    I = ∫₀^∞ S(τ') exp(-(τ'-τ)) dτ'

    At the surface (τ=0), this gives emergent intensity:
    I(0, μ) = ∫₀^∞ S(τ) exp(-τ/μ) dτ/μ

    The emergent flux is the angle integral:
    F = 2π ∫₀¹ I(0, μ) μ dμ
    """
    # Step 1: Compute optical depth using anchored scheme
    # tau(λ) = tau_ref * α(λ) / α_ref
    tau_ref = 10.0 ** log_tau_ref
    if alpha_ref is not None:
        # Anchored scheme: scale tau_ref by opacity ratio
        tau = tau_ref * alpha / alpha_ref
    else:
        # No alpha_ref provided: use tau_ref directly
        # This is correct for the reference wavelength (5000 Å)
        tau = tau_ref

    # Step 2: Compute emergent flux based on intensity scheme
    if intensity_scheme == "linear_flux_only":
        if use_expint_flux:
            # Fastest: use exponential integral formula
            # compute_F_flux_only_expint returns the value without the 2π factor
            # Julia multiplies by 2π at the end (line 136 in RadiativeTransfer.jl)
            flux = 2.0 * jnp.pi * compute_F_flux_only_expint(tau, S)
        else:
            # Fast: compute intensity along vertical ray, then approximate flux
            I_vertical = compute_I_linear_flux_only(tau, S)
            # For plane-parallel, F ≈ π * I_vertical (Eddington approximation)
            flux = jnp.pi * I_vertical

        intensity = None

    elif intensity_scheme == "linear":
        # Compute intensity at multiple angles, then integrate for flux
        mu_points, mu_weights = generate_mu_grid(n_mu)

        # Compute intensity at each angle
        intensities = jnp.array([compute_I_linear(tau, S, mu) for mu in mu_points])

        # Integrate to get flux
        flux = compute_flux_from_intensities(intensities, mu_points, mu_weights)
        intensity = intensities

    elif intensity_scheme == "bezier":
        raise NotImplementedError("Bezier intensity scheme not yet implemented")

    else:
        raise ValueError(f"Unknown intensity_scheme: {intensity_scheme}")

    return flux, intensity


def radiative_transfer(
    alpha_grid,
    S_grid,
    spatial_coord,
    log_tau_ref,
    alpha_ref=None,
    spherical=False,
    intensity_scheme="linear_flux_only",
    use_expint_flux=True,
    n_mu=5
):
    """
    Solve radiative transfer at multiple wavelengths.

    Parameters
    ----------
    alpha_grid : array, shape (n_wavelengths, n_layers)
        Absorption coefficient at each wavelength and layer
    S_grid : array, shape (n_wavelengths, n_layers)
        Source function at each wavelength and layer
    spatial_coord : array, shape (n_layers,)
        Spatial coordinates of layers
    log_tau_ref : array, shape (n_layers,)
        Reference optical depth (log scale)
    alpha_ref : array, shape (n_layers,), optional
        Absorption coefficient at reference wavelength [cm⁻¹]
        Required for anchored optical depth scheme.
        If None, uses tau_ref directly (assumes alpha ≈ alpha_ref)
    spherical : bool, optional
        Spherical geometry flag (default: False)
    intensity_scheme : str, optional
        Intensity calculation method (default: "linear_flux_only")
    use_expint_flux : bool, optional
        Use exponential integral flux optimization (default: True)
    n_mu : int, optional
        Number of angle quadrature points (default: 5)

    Returns
    -------
    fluxes : array, shape (n_wavelengths,)
        Emergent flux at each wavelength
    intensities : array or None
        Intensity profiles if computed, otherwise None
        Shape: (n_wavelengths, n_mu) if available

    Examples
    --------
    >>> # Setup atmosphere and opacity
    >>> n_layers = 56
    >>> n_wavelengths = 1000
    >>> alpha = np.random.rand(n_wavelengths, n_layers) * 1e-10
    >>> T = np.linspace(8000, 4000, n_layers)
    >>> from korg.continuum_absorption.planck import planck_function
    >>> nu = 3e10 / (5000e-8)  # Frequency at 5000 Å
    >>> S = np.array([planck_function(nu, T_i) for T_i in T])
    >>> S_grid = np.tile(S, (n_wavelengths, 1))
    >>> spatial_coord = np.linspace(1e10, 0, n_layers)
    >>> log_tau_ref = np.linspace(-4, 2, n_layers)
    >>> fluxes, _ = radiative_transfer(alpha, S_grid, spatial_coord, log_tau_ref)
    """
    n_wavelengths = alpha_grid.shape[0]

    fluxes = []
    intensities_list = [] if intensity_scheme != "linear_flux_only" else None

    # Process each wavelength
    for i in range(n_wavelengths):
        flux, intensity = radiative_transfer_single_wavelength(
            alpha_grid[i],
            S_grid[i],
            spatial_coord,
            log_tau_ref,
            alpha_ref=alpha_ref,
            spherical=spherical,
            intensity_scheme=intensity_scheme,
            use_expint_flux=use_expint_flux,
            n_mu=n_mu
        )

        fluxes.append(flux)
        if intensities_list is not None and intensity is not None:
            intensities_list.append(intensity)

    fluxes = jnp.array(fluxes)

    if intensities_list:
        intensities = jnp.array(intensities_list)
    else:
        intensities = None

    return fluxes, intensities


@jit
def radiative_transfer_single_wavelength_jit(alpha, S, tau_ref, alpha_ref):
    """
    JIT-compatible single wavelength radiative transfer.

    Uses exponential integral flux method for maximum performance.

    Parameters
    ----------
    alpha : array, shape (n_layers,)
        Absorption coefficient [cm⁻¹]
    S : array, shape (n_layers,)
        Source function [erg cm⁻² s⁻¹ sr⁻¹ Hz⁻¹]
    tau_ref : array, shape (n_layers,)
        Reference optical depth (NOT log)
    alpha_ref : array, shape (n_layers,)
        Reference absorption coefficient [cm⁻¹]

    Returns
    -------
    flux : float
        Emergent flux
    """
    # Compute optical depth using anchored scheme
    tau = tau_ref * alpha / jnp.clip(alpha_ref, 1e-30, jnp.inf)

    # Use exponential integral flux method
    flux = 2.0 * jnp.pi * compute_F_flux_only_expint(tau, S)

    return flux


@jit
def radiative_transfer_jit(
    alpha_grid,
    S_grid,
    spatial_coord,
    log_tau_ref,
    alpha_ref
):
    """
    Fully JIT-compatible radiative transfer for multiple wavelengths.

    Uses exponential integral flux method and vmap for parallelism.

    Parameters
    ----------
    alpha_grid : array, shape (n_wavelengths, n_layers)
        Absorption coefficient at each wavelength and layer
    S_grid : array, shape (n_wavelengths, n_layers)
        Source function at each wavelength and layer
    spatial_coord : array, shape (n_layers,)
        Spatial coordinates (not used in current implementation)
    log_tau_ref : array, shape (n_layers,)
        Reference optical depth (log scale)
    alpha_ref : array, shape (n_layers,)
        Reference absorption coefficient

    Returns
    -------
    fluxes : array, shape (n_wavelengths,)
        Emergent flux at each wavelength
    intensities : None
        Placeholder for API compatibility
    """
    tau_ref = 10.0 ** log_tau_ref

    # Vectorize over wavelengths using vmap
    def solve_one_wavelength(alpha_wl, S_wl):
        return radiative_transfer_single_wavelength_jit(alpha_wl, S_wl, tau_ref, alpha_ref)

    fluxes = jax.vmap(solve_one_wavelength)(alpha_grid, S_grid)

    return fluxes, None


# Make jax available for vmap
import jax
