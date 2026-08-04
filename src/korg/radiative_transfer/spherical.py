"""
Radiative transfer along the rays of a spherical (shell) atmosphere.

Port of the spherical branch of Korg.jl's ``radiative_transfer`` and
``radiative_transfer_core`` (``src/RadiativeTransfer/RadiativeTransfer.jl``).

Structure
---------
In plane-parallel geometry the optical depth along a slanted ray is simply the
vertical optical depth divided by μ, so a single τ column serves every angle.
That is *not* true in spherical geometry: each surface μ defines a ray with its
own impact parameter ``b``, its own path-length coordinate
``s = sqrt(r**2 - b**2)`` and its own set of intersected layers.  The optical
depth must therefore be integrated separately along every ray, with

    dτ = α * (τ_ref / α_ref) * (ds/dr) * d(ln τ_ref)

which is the anchored scheme with Korg.jl's ``integrand_factor``
(``τ_ref / α_ref * dsdz``); ``ds/dr = r/s`` replaces the plane-parallel
``1/μ``.

Rays are of two kinds:

* **core rays** (``b <= r[-1]``) pass through the whole atmosphere.  They are
  integrated outwards from the innermost layer with the usual ``I = 0``
  boundary condition at depth.
* **tangent rays** (``b > r[-1]``) turn around at ``r = b``.  Korg.jl handles
  these by first integrating an *inward* ray from the top of the atmosphere
  down to the tangent point, then using the resulting intensity to seed the
  outward leg.  That is the ``include_inward_rays`` machinery in Korg.jl:
  inward rays are generated exactly for those μ whose ray does not reach the
  innermost layer.

The emergent astrophysical flux is assembled from the rays rather than from a
μ-quadrature over a plane-parallel slab::

    F = 2π Σ_j w_j μ_j I_surface(μ_j)

which is the same expression, but ``I_surface(μ_j)`` is the intensity emerging
along ray *j* and μ_j is its *surface* direction cosine.  (Writing the flux as
an integral over impact parameter, ``F = 2π ∫ I b db / r[0]**2``, and
substituting ``b = r[0] sqrt(1 - μ**2)`` recovers ``2π ∫ I μ dμ``.)

jit and shapes
--------------
The number of layers a ray intersects is data dependent, so everything here
operates on fixed-size ``(n_layers,)`` arrays plus a boolean mask, exactly as
:mod:`korg.radiative_transfer.rays` produces them.  There is no Python control
flow on traced values, so the whole module is ``jax.jit``-traceable and
``jax.grad``-differentiable.

Reference: Korg.jl RadiativeTransfer module.
"""

import jax
import jax.numpy as jnp

from .intensity import fritsch_butland_C
from .rays import calculate_rays

# Optical-depth seed of the Bezier tau scheme, matching Korg.jl's
# ``compute_tau_bezier!``, which cannot start from exactly zero.
_BEZIER_TAU_SEED = 1e-5


def _segment_mask(mask):
    """
    Segments (layer pairs) that lie entirely inside the ray.

    Parameters
    ----------
    mask : array of bool, shape (n_layers,)
        Layers the ray intersects.

    Returns
    -------
    array of bool, shape (n_layers - 1,)
        True where both endpoints of the segment are intersected.
    """
    return mask[:-1] & mask[1:]


def ray_tau_anchored(alpha, log_tau_ref, alpha_ref, dsdz, mask):
    """
    Anchored optical depth along one ray of a spherical atmosphere.

    Integrates ``dτ/d(ln τ_ref) = α/α_ref * τ_ref * ds/dr`` with the
    trapezoid rule, which is Korg.jl's ``compute_tau_anchored!`` with
    ``integrand_factor = τ_ref ./ α_ref .* dsdz``.

    Parameters
    ----------
    alpha : array, shape (n_layers,)
        Absorption coefficient at the wavelength of interest [cm⁻¹].
    log_tau_ref : array, shape (n_layers,)
        log₁₀ of the reference optical depth at each layer.
    alpha_ref : array, shape (n_layers,)
        Absorption coefficient at the reference wavelength [cm⁻¹].
    dsdz : array, shape (n_layers,)
        ``ds/dr`` along the ray, from :func:`~korg.radiative_transfer.rays.calculate_rays`.
    mask : array of bool, shape (n_layers,)
        Layers the ray intersects.

    Returns
    -------
    tau : array, shape (n_layers,)
        Optical depth along the ray, zero at the first layer and constant
        beyond the deepest intersected layer.
    """
    tau_ref = 10.0 ** log_tau_ref
    alpha_ref_safe = jnp.where(jnp.abs(alpha_ref) > 1e-30, alpha_ref, 1e-30)
    integrand = alpha / alpha_ref_safe * tau_ref * dsdz

    integrand_avg = 0.5 * (integrand[:-1] + integrand[1:])
    dtau = integrand_avg * jnp.diff(log_tau_ref) * jnp.log(10.0)
    # Segments outside the ray contribute nothing.  The masked-out values are
    # finite (``dsdz`` is floored in ``calculate_rays``), so the exact zero
    # here kills their cotangents too.
    dtau = jnp.where(_segment_mask(mask), dtau, 0.0)

    return jnp.concatenate([jnp.zeros((1,), dtau.dtype), jnp.cumsum(dtau)])


def ray_tau_bezier(alpha, path, mask):
    """
    Bezier optical depth along one ray of a spherical atmosphere.

    Korg.jl's ``compute_tau_bezier!`` applied to the ray's path-length
    coordinate.

    Parameters
    ----------
    alpha : array, shape (n_layers,)
        Absorption coefficient [cm⁻¹].
    path : array, shape (n_layers,)
        Distance along the ray, strictly decreasing with layer index.
    mask : array of bool, shape (n_layers,)
        Layers the ray intersects.

    Returns
    -------
    tau : array, shape (n_layers,)
        Optical depth along the ray, seeded at 1e-5 as in Korg.jl.

    Notes
    -----
    Korg.jl builds the Fritsch-Butland control points from the *truncated*
    ray, whereas the padded representation used here builds them from the full
    (padded) arrays.  The control point of the deepest intersected segment
    therefore differs slightly from Korg.jl's.  This scheme is not used by
    synthesis, is described as "not recommended" by Korg.jl itself, and is not
    covered by the Julia reference fixtures; the anchored scheme is the
    validated one.
    """
    C = fritsch_butland_C(path, alpha)
    C = jnp.clip(C, 0.5 * jnp.min(alpha), 2.0 * jnp.max(alpha))

    ds = path[:-1] - path[1:]
    dtau = ds / 3.0 * (alpha[1:] + alpha[:-1] + C)
    dtau = jnp.where(_segment_mask(mask), dtau, 0.0)

    seed = jnp.full((1,), _BEZIER_TAU_SEED, dtau.dtype)
    return jnp.concatenate([seed, seed[0] + jnp.cumsum(dtau)])


def _linear_slopes(tau, S):
    """
    Per-segment source-function slope ``m = ΔS/Δτ`` with a safe denominator.

    Korg.jl bumps a numerically zero ``Δτ`` to one (``Δτ += (Δτ == 0)``);
    padded segments here have ``Δτ`` exactly zero for the same reason.  The
    substitute is a constant, so the dead branch contributes a zero cotangent
    rather than the ``0 * inf`` that a bare ``jnp.where`` around the quotient
    would produce.

    Parameters
    ----------
    tau : array, shape (n_layers,)
        Optical depth along the ray.
    S : array, shape (n_layers,)
        Source function.

    Returns
    -------
    delta : array, shape (n_layers - 1,)
        Optical depth increments.
    m : array, shape (n_layers - 1,)
        Source function slopes.
    """
    delta = tau[1:] - tau[:-1]
    delta_safe = jnp.where(delta > 0.0, delta, 1.0)
    return delta, (S[1:] - S[:-1]) / delta_safe


def ray_emergent_linear_flux_only(tau, S, mask):
    """
    Emergent intensity along one ray, ``I_scheme = "linear_flux_only"``.

    Reproduces Korg.jl's ``compute_I_linear_flux_only`` for the outward leg,
    plus the inward leg that seeds the intensity at a tangent ray's turning
    point.

    Parameters
    ----------
    tau : array, shape (n_layers,)
        Optical depth along the outward leg of the ray.
    S : array, shape (n_layers,)
        Source function at each layer.
    mask : array of bool, shape (n_layers,)
        Layers the ray intersects.

    Returns
    -------
    I : scalar
        Intensity emerging at the top of the atmosphere along the ray.

    Notes
    -----
    Korg.jl computes the inward leg on the reversed layer range and then sets
    ``I_out = I_in * exp(-τ_total)``.  Because the two legs of a tangent ray
    are mirror images, the inward optical depth satisfies
    ``τ_in[j] = τ_total - τ[L-1-j]``, so the seed can be written directly in
    terms of the outward ``τ``::

        seed = Σ_k [ -exp(-(2T - τ_k)) (S_k - m_k) + exp(-(2T - τ_{k+1})) (S_{k+1} - m_k) ]

    with ``T = τ_total``.  Every exponent is ``>= T >= 0``, so nothing
    overflows -- unlike the algebraically identical ``exp(-2T) * exp(+τ)``
    form.

    For a ray that reaches the innermost layer there is no inward leg (Korg.jl
    generates inward rays only where ``length(path) < length(spatial_coord)``),
    so the seed is switched off exactly.
    """
    seg = _segment_mask(mask)
    delta, m = _linear_slopes(tau, S)

    exp_lo = jnp.exp(-tau[:-1])
    exp_hi = jnp.exp(-tau[1:])
    emission = jnp.sum(
        jnp.where(seg, -exp_hi * (S[1:] + m) + exp_lo * (S[:-1] + m), 0.0)
    )

    total_tau = tau[-1]
    seed_lo = jnp.exp(-(2.0 * total_tau - tau[:-1]))
    seed_hi = jnp.exp(-(2.0 * total_tau - tau[1:]))
    seed = jnp.sum(
        jnp.where(seg, -seed_lo * (S[:-1] - m) + seed_hi * (S[1:] - m), 0.0)
    )
    # ``mask[-1]`` is True exactly when the ray reaches the innermost layer.
    seed = jnp.where(mask[-1], 0.0, seed)

    return emission + seed


def ray_emergent_linear(tau, S, mask):
    """
    Emergent intensity along one ray, ``I_scheme = "linear"``.

    Korg.jl's ``compute_I_linear!`` recursion, run inward first (to seed the
    tangent point) and then outward.

    Parameters
    ----------
    tau : array, shape (n_layers,)
        Optical depth along the outward leg of the ray.
    S : array, shape (n_layers,)
        Source function at each layer.
    mask : array of bool, shape (n_layers,)
        Layers the ray intersects.

    Returns
    -------
    I : scalar
        Intensity emerging at the top of the atmosphere along the ray.

    Notes
    -----
    ``compute_I_linear!`` marches from the far end of the ray towards the
    observer::

        I[k] = (I[k+1] - S[k] - m (δ + 1)) exp(-δ) + m + S[k]

    On the inward leg the ray is traversed in the opposite direction, which in
    the atmosphere's own layer ordering turns the recursion around: the deeper
    layer is the one being solved for, ``S[k]`` and ``S[k+1]`` swap, and the
    slope changes sign.  That is the forward scan below; its final value is
    the intensity at the tangent point, which becomes the boundary condition
    of the outward scan.
    """
    seg = _segment_mask(mask)
    delta, m = _linear_slopes(tau, S)
    exp_neg_delta = jnp.exp(-delta)

    # Inward leg: J[k+1] from J[k], starting from zero incident intensity at
    # the top of the atmosphere and freezing once the ray has turned around.
    def inward_step(J, x):
        S_hi, m_k, delta_k, exp_k, ok = x
        J_new = (J - S_hi + m_k * (delta_k + 1.0)) * exp_k - m_k + S_hi
        return jnp.where(ok, J_new, J), None

    I_tangent, _ = jax.lax.scan(
        inward_step, jnp.zeros((), tau.dtype),
        (S[1:], m, delta, exp_neg_delta, seg),
    )
    # No inward leg for a ray that passes through the innermost layer.
    I_bottom = jnp.where(mask[-1], 0.0, I_tangent)

    # Outward leg: standard backward recursion seeded with I_bottom.
    def outward_step(I, x):
        S_lo, m_k, delta_k, exp_k, ok = x
        I_new = (I - S_lo - m_k * (delta_k + 1.0)) * exp_k + m_k + S_lo
        return jnp.where(ok, I_new, I_bottom), None

    I_surface, _ = jax.lax.scan(
        outward_step, I_bottom,
        (S[:-1], m, delta, exp_neg_delta, seg),
        reverse=True,
    )
    return I_surface


def ray_emergent_bezier(tau, S, mask):
    """
    Emergent intensity along one ray, ``I_scheme = "bezier"``.

    Parameters
    ----------
    tau : array, shape (n_layers,)
        Optical depth along the outward leg of the ray.
    S : array, shape (n_layers,)
        Source function at each layer.
    mask : array of bool, shape (n_layers,)
        Layers the ray intersects.

    Returns
    -------
    I : scalar
        Intensity emerging at the top of the atmosphere along the ray.

    Notes
    -----
    As for :func:`ray_tau_bezier`, the Fritsch-Butland control points are
    built from the padded arrays rather than from the truncated ray, so the
    deepest intersected segment differs slightly from Korg.jl.  The padded τ
    increments are replaced by a strictly positive filler purely so the
    control-point construction has non-degenerate spacing; they contribute
    nothing to the intensity, which is masked segment by segment.
    """
    seg = _segment_mask(mask)

    delta_raw = tau[1:] - tau[:-1]
    # Strictly positive filler for the padded region: the mean intersected
    # increment, falling back to unity when the ray sees at most one layer.
    n_seg = jnp.sum(seg)
    filler = jnp.where(n_seg > 0, jnp.sum(jnp.where(seg, delta_raw, 0.0))
                       / jnp.where(n_seg > 0, n_seg, 1.0), 1.0)
    filler = jnp.where(filler > 0.0, filler, 1.0)
    delta = jnp.where(seg, delta_raw, filler)
    tau_padded = jnp.concatenate([tau[:1], tau[0] + jnp.cumsum(delta)])

    C = fritsch_butland_C(tau_padded, S)

    exp_neg_delta = jnp.exp(-delta)
    delta_sq = delta * delta
    delta_sq_safe = jnp.where(delta_sq > 1e-300, delta_sq, 1.0)
    a = (2.0 + delta_sq - 2.0 * delta - 2.0 * exp_neg_delta) / delta_sq_safe
    b = (2.0 - (2.0 + 2.0 * delta + delta_sq) * exp_neg_delta) / delta_sq_safe
    g = (2.0 * delta - 4.0 + (2.0 * delta + 4.0) * exp_neg_delta) / delta_sq_safe

    def inward_step(J, x):
        S_lo, S_hi, a_k, b_k, g_k, C_k, exp_k, ok = x
        J_new = J * exp_k + a_k * S_hi + b_k * S_lo + g_k * C_k
        return jnp.where(ok, J_new, J), None

    xs = (S[:-1], S[1:], a, b, g, C, exp_neg_delta, seg)
    I_tangent, _ = jax.lax.scan(inward_step, jnp.zeros((), tau.dtype), xs)
    I_bottom = jnp.where(mask[-1], 0.0, I_tangent)

    def outward_step(I, x):
        S_lo, S_hi, a_k, b_k, g_k, C_k, exp_k, ok = x
        I_new = I * exp_k + a_k * S_lo + b_k * S_hi + g_k * C_k
        return jnp.where(ok, I_new, I_bottom), None

    I_surface, _ = jax.lax.scan(outward_step, I_bottom, xs, reverse=True)
    # Korg.jl applies exp(-tau[1]) at the end, which matters because the
    # Bezier tau scheme seeds tau[0] at 1e-5 rather than zero.
    return I_surface * jnp.exp(-tau[0])


_TAU_SOLVERS = ("anchored", "bezier")
_I_SOLVERS = ("linear_flux_only", "linear", "bezier")


def spherical_ray_flux(alpha_grid, S_grid, radii, log_tau_ref, alpha_ref,
                       mu_grid, mu_weights,
                       tau_scheme="anchored", intensity_scheme="linear_flux_only",
                       spherical=True):
    """
    Emergent flux of a spherical atmosphere, assembled from rays.

    Parameters
    ----------
    alpha_grid : array, shape (n_wavelengths, n_layers)
        Absorption coefficient [cm⁻¹].
    S_grid : array, shape (n_wavelengths, n_layers)
        Source function.
    radii : array, shape (n_layers,)
        Radius of each layer [cm], outermost first.
    log_tau_ref : array, shape (n_layers,)
        log₁₀ of the reference optical depth.
    alpha_ref : array, shape (n_layers,)
        Absorption coefficient at the reference wavelength [cm⁻¹].
    mu_grid : array, shape (n_mu,)
        Surface direction cosines.
    mu_weights : array, shape (n_mu,)
        Quadrature weights matching *mu_grid*.
    tau_scheme : {'anchored', 'bezier'}, optional
        Optical depth scheme (default: 'anchored').
    intensity_scheme : {'linear_flux_only', 'linear', 'bezier'}, optional
        Intensity scheme (default: 'linear_flux_only').
    spherical : bool, optional
        Ray geometry (default: True).  Setting it to False runs the identical
        solver over plane-parallel rays (``ds/dz = 1/μ``, every layer
        intersected), which is what a geometrically thin shell must converge
        to and is used to check that limit with the geometry as the only
        difference.

    Returns
    -------
    fluxes : array, shape (n_wavelengths,)
        Emergent astrophysical flux at the outermost radius.
    intensities : array, shape (n_wavelengths, n_mu)
        Emergent intensity along each ray.
    """
    if tau_scheme not in _TAU_SOLVERS:
        raise ValueError(f"tau_scheme must be one of {_TAU_SOLVERS}, got {tau_scheme!r}")
    if intensity_scheme not in _I_SOLVERS:
        raise ValueError(
            f"intensity_scheme must be one of {_I_SOLVERS}, got {intensity_scheme!r}")

    alpha_grid = jnp.asarray(alpha_grid)
    S_grid = jnp.asarray(S_grid)
    radii = jnp.asarray(radii)
    log_tau_ref = jnp.asarray(log_tau_ref)
    mu_grid = jnp.asarray(mu_grid)
    mu_weights = jnp.asarray(mu_weights)
    if alpha_ref is None:
        if tau_scheme == "anchored":
            raise ValueError("the anchored tau scheme requires alpha_ref")
        alpha_ref = jnp.ones_like(log_tau_ref)
    alpha_ref = jnp.asarray(alpha_ref)

    path, dsdz, mask = calculate_rays(mu_grid, radii, spherical=spherical)

    if intensity_scheme == "linear_flux_only":
        emergent = ray_emergent_linear_flux_only
    elif intensity_scheme == "linear":
        emergent = ray_emergent_linear
    else:
        emergent = ray_emergent_bezier

    def one_ray_one_wavelength(alpha, S, path_mu, dsdz_mu, mask_mu):
        if tau_scheme == "anchored":
            tau = ray_tau_anchored(alpha, log_tau_ref, alpha_ref, dsdz_mu, mask_mu)
        else:
            tau = ray_tau_bezier(alpha, path_mu, mask_mu)
        return emergent(tau, S, mask_mu)

    # (n_wavelengths, n_mu)
    over_rays = jax.vmap(one_ray_one_wavelength, in_axes=(None, None, 0, 0, 0))
    over_wavelengths = jax.vmap(over_rays, in_axes=(0, 0, None, None, None))
    intensities = over_wavelengths(alpha_grid, S_grid, path, dsdz, mask)

    fluxes = 2.0 * jnp.pi * jnp.sum(intensities * (mu_weights * mu_grid), axis=1)
    return fluxes, intensities
