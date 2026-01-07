"""
Optical depth calculations for radiative transfer.

Implements the "anchored" optical depth scheme, which computes optical depth
relative to a reference wavelength. This improves numerical stability and
accuracy compared to direct integration.

Reference: Korg.jl RadiativeTransfer module
"""

import jax.numpy as jnp
from jax import jit


def compute_tau_anchored(alpha, spatial_coord, log_tau_ref, spherical=False):
    """
    Compute optical depth using the anchored scheme.

    The anchored scheme computes τ(λ) by integrating:
    dτ/d(log τ_ref) = α(λ) / α(λ_ref) * τ_ref

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
    spherical : bool, optional
        If True, use spherical geometry correction factor
        Default: False (planar geometry)

    Returns
    -------
    tau : array, shape (n_layers,)
        Optical depth at each layer, anchored to reference wavelength

    Notes
    -----
    The integration is performed using trapezoidal rule:
    τ[i+1] = τ[i] + 0.5 * (integrand[i+1] + integrand[i]) * Δ(log τ_ref)

    For spherical geometry, the integrand includes a factor accounting for
    the changing ray path length through shells.

    The first layer (typically top of atmosphere) has τ = 0 by definition.
    """
    n_layers = len(alpha)
    tau = jnp.zeros(n_layers)

    # Convert log reference optical depth to linear
    tau_ref = 10.0 ** log_tau_ref

    if spherical:
        # For spherical geometry, integrand includes geometric factor
        # integrand = α(λ) / α(λ_ref) * τ_ref * (1 + r/H_scale)
        # This accounts for curved ray paths through shells
        # For now, use simplified version (full version needs scale height)
        integrand = alpha * tau_ref
    else:
        # Planar geometry: simple ratio
        integrand = alpha * tau_ref

    # Trapezoidal integration over log(τ_ref)
    # tau[0] = 0 (top of atmosphere)
    # tau[i] = tau[i-1] + 0.5 * (integrand[i] + integrand[i-1]) * Δ(log τ_ref)

    # Compute increments using vectorized operations
    delta_log_tau = jnp.diff(log_tau_ref)  # log_tau_ref[i] - log_tau_ref[i-1] for i=1..n-1
    integrand_avg = 0.5 * (integrand[:-1] + integrand[1:])  # Average of adjacent layers
    dtau = integrand_avg * delta_log_tau  # Increment in tau

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
