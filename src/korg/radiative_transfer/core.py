"""
Core radiative transfer solver.

Main orchestration function that combines optical depth calculation,
intensity integration, and flux computation.

Reference: Korg.jl RadiativeTransfer module
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit

from .optical_depth import compute_tau_anchored, compute_tau_direct, compute_tau_bezier
from .intensity import (compute_I_linear_flux_only, compute_F_flux_only_expint,
                         compute_I_linear, compute_I_bezier, compute_flux_from_intensities)
from .spherical import spherical_ray_flux


def leggauss(n):
    """
    Gauss-Legendre quadrature nodes and weights.

    Parameters
    ----------
    n : int
        Number of points

    Returns
    -------
    nodes : array, shape (n_mu, )
    weights : array, shape (n_mu, )
    """
    # Companion matrix for Legendre polynomials
    i = jnp.arange(1, n)
    beta = i / jnp.sqrt(4 * i**2 - 1)

    # Symmetric tridiagonal matrix
    T = jnp.diag(beta, -1) + jnp.diag(beta, 1)

    # Eigenvalues are nodes, eigenvectors give weights
    nodes, V = jnp.linalg.eigh(T)
    weights = 2 * V[0, :]**2

    return nodes, weights


def generate_mu_grid(n_mu=5):
    """
    Generate Gaussian quadrature points and weights for angle integration.

    Returns μ = cos(θ) points and weights for integrating over solid angle.

    Parameters
    ----------
    n_mu : int or array_like, optional
        Number of quadrature points (default: 5).
        More points = better accuracy but slower.
        An array of μ values may be given instead, in which case those values
        are used directly and the integral is done with the trapezoid rule --
        this mirrors Korg.jl's ``generate_mu_grid(μ_values)`` method.

    Returns
    mu_points : array, shape (n_mu,)
        Quadrature points in [0, 1]
    mu_weights : array, shape (n_mu,)
        Quadrature weights (sum to 1)

    Notes
    -----
    Uses Gauss-Legendre quadrature on [0, 1] interval.
    For n_mu=5, typical error in flux is < 0.1%.
    """
    if np.ndim(n_mu) > 0:
        # Explicit μ values: trapezoid weights, as in Korg.jl.
        mu_points = jnp.asarray(n_mu)
        if mu_points.size == 1:
            return mu_points, jnp.ones_like(mu_points)
        delta = jnp.diff(mu_points)
        mu_weights = 0.5 * jnp.concatenate(
            [delta[:1], delta[:-1] + delta[1:], delta[-1:]]
        )
        return mu_points, mu_weights

    # Get Gauss-Legendre quadrature on [-1, 1]
    # Then transform to [0, 1]
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
    tau_scheme="anchored",
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
    tau_scheme : str, optional
        Method for computing optical depth:
        - "anchored": Scale tau_ref by opacity ratio (default, requires alpha_ref)
        - "bezier": Direct integration with Bezier interpolation (more stable)
    intensity_scheme : str, optional
        Method for computing intensity:
        - "linear_flux_only": Fast, flux only (default)
        - "linear": Linear interpolation with angle integration
        - "bezier": Bezier interpolation (more stable at large optical depths)
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

    In spherical geometry that last step is not a μ-quadrature over a slab:
    each μ selects a ray with its own impact parameter and its own set of
    intersected shells, and the flux is assembled from those rays.  See
    :mod:`korg.radiative_transfer.spherical`.
    """
    if spherical:
        fluxes, intensities = radiative_transfer_spherical(
            jnp.atleast_2d(alpha), jnp.atleast_2d(S), spatial_coord, log_tau_ref,
            alpha_ref, n_mu=n_mu, tau_scheme=tau_scheme,
            intensity_scheme=intensity_scheme,
        )
        return fluxes[0], intensities[0]

    # Step 1: Compute optical depth
    if tau_scheme == "anchored":
        # Anchored scheme: integrate dτ/d(log τ_ref) = α(λ) / α_ref * τ_ref
        if alpha_ref is None:
            raise ValueError("anchored tau scheme requires alpha_ref")
        tau = compute_tau_anchored(alpha, spatial_coord, log_tau_ref, alpha_ref, spherical=spherical)
    elif tau_scheme == "bezier":
        # Bezier scheme: direct integration of opacity along path
        # This ensures monotonic optical depth even with opacity inversions
        tau = compute_tau_bezier(alpha, spatial_coord)
    else:
        raise ValueError(f"Unknown tau_scheme: {tau_scheme}")

    # Step 2: Compute emergent flux based on intensity scheme
    if intensity_scheme == "linear_flux_only":
        if use_expint_flux:
            # Fastest: use exponential integral formula
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
        # Compute intensity at multiple angles using Bezier interpolation
        # More numerically stable than linear at large optical depths
        mu_points, mu_weights = generate_mu_grid(n_mu)

        # Compute intensity at each angle
        # Scale optical depth by μ for slant path (as in compute_I_linear)
        intensities = jnp.array([compute_I_bezier(tau / mu, S) for mu in mu_points])

        # Integrate to get flux
        flux = compute_flux_from_intensities(intensities, mu_points, mu_weights)
        intensity = intensities

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
    tau_scheme="anchored",
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
    tau_scheme : str, optional
        Optical depth calculation method (default: "anchored")
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
    if spherical:
        # Every ray is different in spherical geometry, so the whole
        # calculation is vectorised over (wavelength, μ) at once rather than
        # looped wavelength by wavelength.
        return radiative_transfer_spherical(
            alpha_grid, S_grid, spatial_coord, log_tau_ref, alpha_ref,
            n_mu=n_mu, tau_scheme=tau_scheme, intensity_scheme=intensity_scheme,
        )

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
            tau_scheme=tau_scheme,
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


def radiative_transfer_spherical(
    alpha_grid,
    S_grid,
    radii,
    log_tau_ref,
    alpha_ref,
    n_mu=5,
    mu_grid=None,
    mu_weights=None,
    tau_scheme="anchored",
    intensity_scheme="linear_flux_only",
    R_photosphere=None,
):
    """
    Solve radiative transfer in spherical (shell) geometry.

    Port of the spherical branch of Korg.jl's ``radiative_transfer``.  Each
    surface μ defines a ray with impact parameter ``b = r[0] sqrt(1 - μ²)``;
    the optical depth is integrated along the ray's own path-length
    coordinate, tangent rays (those that do not reach the innermost shell)
    are seeded by an inward ray, and the emergent flux is assembled from the
    rays.

    Parameters
    ----------
    alpha_grid : array, shape (n_wavelengths, n_layers)
        Absorption coefficient [cm⁻¹].
    S_grid : array, shape (n_wavelengths, n_layers)
        Source function.
    radii : array, shape (n_layers,)
        Radius from the stellar centre [cm], outermost layer first (the same
        ordering Korg.jl uses, with ``tau_ref`` increasing).
    log_tau_ref : array, shape (n_layers,)
        log₁₀ of the reference optical depth.
    alpha_ref : array, shape (n_layers,)
        Absorption coefficient at the reference wavelength [cm⁻¹].
    n_mu : int or array_like, optional
        Number of Gauss-Legendre μ points, or explicit μ values
        (default: 5).  Ignored if *mu_grid* is given.
    mu_grid, mu_weights : array, optional
        Explicit surface μ grid and quadrature weights.
    tau_scheme : {'anchored', 'bezier'}, optional
        Optical depth scheme (default: 'anchored').
    intensity_scheme : {'linear_flux_only', 'linear', 'bezier'}, optional
        Intensity scheme (default: 'linear_flux_only').  Korg.jl defaults to
        ``'linear'`` for shell atmospheres; the two agree to round-off.
    R_photosphere : float, optional
        Photospheric radius [cm].  If given, the flux is rescaled from the
        outermost radius to the photospheric radius by ``(r[0]/R)²``, which
        is Korg.jl's ``photosphere_correction``.  Default: no rescaling, i.e.
        the flux at ``radii[0]``, matching Korg.jl's low-level
        ``radiative_transfer(α, S, radii, μ, true)``.

    Returns
    -------
    fluxes : array, shape (n_wavelengths,)
        Emergent astrophysical flux.
    intensities : array, shape (n_wavelengths, n_mu)
        Emergent intensity along each ray.

    Examples
    --------
    >>> fluxes, intensities = radiative_transfer_spherical(
    ...     alpha_grid, S_grid, radii, log_tau_ref, alpha_ref, n_mu=20)
    """
    if mu_grid is None:
        mu_grid, mu_weights = generate_mu_grid(n_mu)
    elif mu_weights is None:
        mu_grid, mu_weights = generate_mu_grid(mu_grid)

    fluxes, intensities = spherical_ray_flux(
        alpha_grid, S_grid, radii, log_tau_ref, alpha_ref, mu_grid, mu_weights,
        tau_scheme=tau_scheme, intensity_scheme=intensity_scheme,
    )

    if R_photosphere is not None:
        fluxes = fluxes * (jnp.asarray(radii)[0] / R_photosphere) ** 2

    return fluxes, intensities


def _compute_tau_anchored_planar(alpha, log_tau_ref, alpha_ref):
    """
    Trapezoidal anchored optical depth (planar geometry, JIT-compatible).

    Matches Julia's compute_tau_anchored!: integrates d(tau)/d(ln tau_ref) = alpha/alpha_ref * tau_ref
    using the trapezoidal rule in log tau_ref space.

    tau[i] = sum_{j<i} 0.5*(f[j]+f[j+1]) * (ln tau_ref[j+1] - ln tau_ref[j])
    where f[j] = alpha[j] * tau_ref[j] / alpha_ref[j]
    """
    tau_ref = 10.0 ** log_tau_ref
    integrand = alpha * tau_ref / jnp.clip(alpha_ref, 1e-30, jnp.inf)
    delta_log_tau = jnp.diff(log_tau_ref) * jnp.log(10.0)  # convert log10 to natural log steps
    integrand_avg = 0.5 * (integrand[:-1] + integrand[1:])
    dtau = integrand_avg * delta_log_tau
    return jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dtau)])


# Precomputed 20-point GL nodes/weights on [0,1] — matches Julia's mu_values=20 default.
_mu_gl_nodes, _mu_gl_weights = generate_mu_grid(20)
_mu_gl_nodes = jnp.array(_mu_gl_nodes)
_mu_gl_weights = jnp.array(_mu_gl_weights)


def _compute_F_gl(tau, S):
    """Flux via 20-pt Gauss-Legendre quadrature over mu (matches Julia linear_flux_only default).

    For piecewise-linear S(tau), the exact intensity at angle mu contributes:
      I_i(mu) = (S_i + m_i*mu) exp(-tau_i/mu) - (S_{i+1} + m_i*mu) exp(-tau_{i+1}/mu)
    where m_i = (S_{i+1} - S_i) / (tau_{i+1} - tau_i).
    F = 2pi * sum_j w_j * I(mu_j) * mu_j  (caller multiplies by 2pi).
    """
    tau_i = tau[:-1]
    tau_ip1 = tau[1:]
    S_i = S[:-1]
    S_ip1 = S[1:]
    delta_tau = tau_ip1 - tau_i
    delta_tau_safe = jnp.where(delta_tau > 0, delta_tau, 1.0)
    m = (S_ip1 - S_i) / delta_tau_safe

    def I_at_mu(mu):
        exp_i = jnp.exp(-tau_i / mu)
        exp_ip1 = jnp.exp(-tau_ip1 / mu)
        return jnp.sum((S_i + m * mu) * exp_i - (S_ip1 + m * mu) * exp_ip1)

    I_mu = jax.vmap(I_at_mu)(_mu_gl_nodes)
    return jnp.sum(_mu_gl_weights * I_mu * _mu_gl_nodes)


@jit
def radiative_transfer_single_wavelength_jit(alpha, S, log_tau_ref, alpha_ref):
    """
    JIT-compatible single wavelength radiative transfer.

    Uses 20-point Gauss-Legendre quadrature over mu, matching Julia's
    default linear_flux_only + mu_values=20 scheme.

    Parameters
    ----------
    alpha : array, shape (n_layers,)
        Absorption coefficient [cm⁻¹]
    S : array, shape (n_layers,)
        Source function [erg cm⁻² s⁻¹ sr⁻¹ Hz⁻¹]
    log_tau_ref : array, shape (n_layers,)
        log10(reference optical depth) at each layer
    alpha_ref : array, shape (n_layers,)
        Reference absorption coefficient [cm⁻¹]

    Returns
    -------
    flux : float
        Emergent flux
    """
    tau = _compute_tau_anchored_planar(alpha, log_tau_ref, alpha_ref)
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
        Reference optical depth (log10 scale)
    alpha_ref : array, shape (n_layers,)
        Reference absorption coefficient

    Returns
    -------
    fluxes : array, shape (n_wavelengths,)
        Emergent flux at each wavelength
    intensities : None
        Placeholder for API compatibility
    """
    # Vectorize over wavelengths using vmap
    def solve_one_wavelength(alpha_wl, S_wl):
        return radiative_transfer_single_wavelength_jit(alpha_wl, S_wl, log_tau_ref, alpha_ref)

    fluxes = jax.vmap(solve_one_wavelength)(alpha_grid, S_grid)

    return fluxes, None
