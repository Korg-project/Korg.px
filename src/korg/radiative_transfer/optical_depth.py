"""
Optical depth calculations for radiative transfer.

Implements the "anchored" optical depth scheme, which computes optical depth
relative to a reference wavelength. This improves numerical stability and
accuracy compared to direct integration.

Reference: Korg.jl RadiativeTransfer module
"""

import jax
import jax.numpy as jnp
from jax import jit
from .intensity import fritsch_butland_C


def compute_tau_anchored(alpha, spatial_coord, log_tau_ref, alpha_ref, spherical=False,
                         dsdz=None, mask=None):
    """
    Compute optical depth using the anchored scheme.

    The anchored scheme computes τ(λ) by integrating:
    dτ/d(log τ_ref) = α(λ) / α(λ_ref) * τ_ref * ds/dz

    This is more stable than direct spatial integration, especially in
    spherical geometry or with steep opacity gradients.

    Parameters
    ----------
    alpha : array, shape (n_layers,)
        Total absorption coefficient at each layer [cm⁻¹]
    spatial_coord : array, shape (n_layers,)
        Spatial coordinate (height or radius) at each layer [cm]
        For planar: height above photosphere
        For spherical: radius from center
    log_tau_ref : array, shape (n_layers,)
        log₁₀(optical depth) at reference wavelength
    alpha_ref : array, shape (n_layers,)
        Absorption coefficient at reference wavelength [cm⁻¹]
        Required for computing the opacity ratio
    spherical : bool, optional
        Geometry flag.  The geometry enters this function *only* through
        *dsdz* and *mask*, which a caller obtains from
        :func:`~korg.radiative_transfer.rays.calculate_rays`; passing
        ``spherical=True`` without them is an error rather than a silent
        plane-parallel result.  Default: False (planar geometry).
    dsdz : array, shape (n_layers,), optional
        ``ds/dz`` along the ray: Korg.jl's ``dsdz``, i.e. ``1/μ`` for a
        plane-parallel ray and ``r/s`` for a spherical one.  ``None`` (the
        default) means the vertical plane-parallel ray, ``ds/dz = 1``.
    mask : array of bool, shape (n_layers,), optional
        Layers the ray intersects.  Segments with a masked endpoint
        contribute nothing.  ``None`` (the default) means every layer.

    Returns
    -------
    tau : array, shape (n_layers,)
        Optical depth at each layer, anchored to reference wavelength

    Raises
    ------
    ValueError
        If ``spherical=True`` but no *dsdz* is supplied.

    Notes
    -----
    The integration is performed using trapezoidal rule:
    τ[i+1] = τ[i] + 0.5 * (integrand[i+1] + integrand[i]) * Δ(log τ_ref)

    where integrand = α(λ) / α(λ_ref) * τ_ref * ds/dz

    In spherical geometry the path-length coordinate along a ray is *not* the
    vertical coordinate divided by μ, so ``ds/dz = r/s`` varies from layer to
    layer and must be supplied per ray.  The high-level entry point
    :func:`~korg.radiative_transfer.core.radiative_transfer` does this for
    you; see :mod:`korg.radiative_transfer.spherical`.

    The first layer (typically top of atmosphere) has τ = 0 by definition.

    Reference: Korg.jl compute_tau_anchored! function
    """
    if spherical and dsdz is None:
        raise ValueError(
            "spherical=True requires dsdz (and normally mask) from calculate_rays; "
            "use korg.radiative_transfer.radiative_transfer(..., spherical=True) "
            "for the full spherical solver"
        )

    # Convert log reference optical depth to linear
    tau_ref = 10.0 ** log_tau_ref

    # Compute integrand = α(λ) / α(λ_ref) * τ_ref
    # Handle division by zero by using a small epsilon
    alpha_ref_safe = jnp.where(jnp.abs(alpha_ref) > 1e-30, alpha_ref, 1e-30)
    integrand = alpha / alpha_ref_safe * tau_ref
    if dsdz is not None:
        integrand = integrand * dsdz

    # Trapezoidal integration over log(τ_ref)
    # tau[0] = 0 (top of atmosphere)
    # tau[i] = tau[i-1] + 0.5 * (integrand[i] + integrand[i-1]) * Δ(log τ_ref)

    # Compute increments using vectorized operations
    delta_log_tau = jnp.diff(log_tau_ref)  # log_tau_ref[i] - log_tau_ref[i-1] for i=1..n-1
    integrand_avg = 0.5 * (integrand[:-1] + integrand[1:])  # Average of adjacent layers
    # d(tau_ref) = tau_ref * ln(10) * d(log10_tau_ref), so multiply by ln(10)
    dtau = integrand_avg * delta_log_tau * jnp.log(10.0)

    if mask is not None:
        dtau = jnp.where(mask[:-1] & mask[1:], dtau, 0.0)

    # Cumulative sum to get tau at each layer
    # tau[0] = 0, tau[i] = sum(dtau[0:i]) for i >= 1
    tau = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dtau)])

    return tau


def compute_tau_direct(alpha, spatial_coord, spherical=False, mu=1.0):
    """
    Compute optical depth by direct spatial integration.

    This is the traditional approach: τ = ∫ α ds along the ray path.
    Less stable than anchored scheme but simpler to understand.

    Parameters
    ----------
    alpha : array, shape (n_layers,)
        Total absorption coefficient at each layer [cm⁻¹]
    spatial_coord : array, shape (n_layers,)
        Spatial coordinate at each layer [cm]
        Increasing outward (from deep to surface)
    spherical : bool, optional
        If True, use spherical geometry
        Default: False (planar geometry)
    mu : float, optional
        Cosine of angle from vertical (μ = cos θ)
        Only used in planar geometry: ds = dz / μ
        Default: 1.0 (vertical ray)

    Returns
    -------
    tau : array, shape (n_layers,)
        Optical depth at each layer
        tau[0] = total optical depth (bottom layer)
        tau[-1] = 0 (top layer, surface)

    Notes
    -----
    Integration proceeds from top (surface) downward, so tau decreases
    with increasing layer index (opposite of depth into atmosphere).
    """
    n_layers = len(alpha)

    if spherical:
        # Spherical: need to compute path length through each shell
        # For now, use simplified radial approximation
        # Full implementation would use impact parameter geometry
        ds = -jnp.diff(spatial_coord)  # Negative because coord increases outward
        path_lengths = ds  # Simplified: assumes radial ray
    else:
        # Planar: ds = dz / μ
        dz = -jnp.diff(spatial_coord)  # Negative because coord increases outward
        path_lengths = dz / mu

    # Compute α * ds at each interval
    # Use average of adjacent layers
    alpha_avg = 0.5 * (alpha[:-1] + alpha[1:])
    dtau = alpha_avg * path_lengths

    # Cumulative sum from surface (last element) to depth (first element)
    # tau[-1] = 0 (surface)
    # tau[i] = tau[i+1] + dtau[i] (going deeper)
    tau = jnp.concatenate([jnp.cumsum(dtau[::-1])[::-1], jnp.array([0.0])])

    return tau


@jit
def compute_tau_bezier(alpha, spatial_coord):
    """
    Compute optical depth using Bezier interpolation of opacity.

    This method ensures monotonically increasing optical depth by integrating
    opacity along the spatial path using Bezier control points. More stable
    than the anchored scheme when opacity has inversions.

    Parameters
    ----------
    alpha : array, shape (n_layers,)
        Total absorption coefficient at each layer [cm⁻¹]
    spatial_coord : array, shape (n_layers,)
        Spatial coordinate at each layer [cm]
        For planar: height/depth coordinate
        Should decrease from top to bottom (increasing into atmosphere)

    Returns
    -------
    tau : array, shape (n_layers,)
        Optical depth at each layer, monotonically increasing

    Notes
    -----
    The optical depth is computed by integrating:
    τ[i] = τ[i-1] + (s[i-1] - s[i]) / 3 * (α[i] + α[i-1] + C[i-1])

    where C are Bezier control points for the opacity profile.

    The factor of 1/3 comes from the Simpson's rule-like integration with
    Bezier control points.

    Reference: Korg.jl compute_tau_bezier! function
    """
    n_layers = len(alpha)

    # Start with small non-zero optical depth at surface
    tau = jnp.zeros(n_layers)
    tau = tau.at[0].set(1e-5)

    # Compute Bezier control points for opacity along spatial coordinate
    C = fritsch_butland_C(spatial_coord, alpha)

    # Clamp control points for numerical stability
    # (prevents extreme values that could cause overflow)
    C = jnp.clip(C, 0.5 * jnp.min(alpha), 2.0 * jnp.max(alpha))

    # Integrate optical depth using Bezier scheme
    # τ[i] = τ[i-1] + Δs / 3 * (α[i-1] + α[i] + C[i-1])
    def body_fn(i, tau_val):
        # Spatial step: use absolute value to ensure positive dtau
        # The direction doesn't matter since we're integrating opacity over path length
        ds = jnp.abs(spatial_coord[i] - spatial_coord[i-1])

        # Bezier-weighted average of opacity
        alpha_avg = (alpha[i-1] + alpha[i] + C[i-1]) / 3.0

        # Increment in optical depth
        dtau = ds * alpha_avg

        # Update tau
        return tau_val.at[i].set(tau_val[i-1] + dtau)

    # Loop through layers to build up optical depth
    tau = jax.lax.fori_loop(1, n_layers, body_fn, tau)

    return tau
