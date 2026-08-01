"""
Ray geometry for radiative transfer.

Port of Korg.jl's ``calculate_rays`` (``src/RadiativeTransfer/RadiativeTransfer.jl``).

For a plane-parallel atmosphere a ray that leaves the surface with direction
cosine μ traverses the vertical coordinate ``z`` with path length ``s = z / μ``
and ``ds/dz = 1/μ``: every ray sees every layer.

For a spherical atmosphere the geometry is genuinely different.  A ray leaving
the outermost shell at direction cosine ``μ_surface`` has impact parameter::

    b = r[0] * sqrt(1 - μ_surface**2)

and, at a shell of radius ``r``, the distance measured along the ray from the
point of closest approach to the centre is::

    s = sqrt(r**2 - b**2),      ds/dr = r / s

Rays with ``b > r[-1]`` never reach the innermost shell: they turn around at a
tangent point where ``r = b`` and ``s = 0``.  Such a ray intersects only the
layers with ``r >= b``, so the *number of layers a ray sees is data dependent*.

Because ``jax.jit`` requires static shapes, this port returns a fixed-size
padded representation: ``path``, ``dsdz`` and ``mask`` are all
``(n_mu, n_layers)`` and ``mask`` marks the layers the ray actually intersects.
``mask`` is a prefix mask (``r`` is ordered outermost-first, so ``r >= b``
selects a leading run), which is exactly Korg.jl's ``1:lowest_layer_index``.

Gradient safety
---------------
``s = sqrt(r**2 - b**2)`` has an infinite derivative where the radicand
vanishes, which is precisely what happens at a ray's tangent point and, by
construction, at every padded (non-intersected) layer where the radicand is
negative.  ``jnp.where(cond, safe, sqrt(radicand))`` would mask the NaN
*value* but not its cotangent.  The radicand is therefore floored at a
strictly positive value *before* the square root, which makes both the value
and the derivative finite everywhere, and the padded entries are additionally
multiplied by an exact zero downstream.

Reference: Korg.jl RadiativeTransfer module, ``calculate_rays``.
"""

import jax.numpy as jnp

# The squared path length ``r**2 - b**2`` is floored at ``(_S_FLOOR_FRAC *
# r[0])**2``.  With the default the smallest representable path length is
# 1e-12 of the stellar radius -- around a centimetre for a red giant, i.e.
# nine orders of magnitude below the thinnest MARCS layer -- so no layer a ray
# genuinely intersects is ever affected, while ``ds/dr`` stays below 1e12 and
# its derivative below 1e24 instead of being infinite.
_S_FLOOR_FRAC = 1e-12


def calculate_rays(mu_surface_grid, spatial_coord, spherical=True):
    """
    Ray path length and its derivative wrt the atmospheric spatial coordinate.

    Parameters
    ----------
    mu_surface_grid : array, shape (n_mu,)
        Direction cosines at the surface of the star, in (0, 1].
    spatial_coord : array, shape (n_layers,)
        Physical distance coordinate of each layer [cm].  Radius from the
        stellar centre when *spherical*, height above the photosphere
        otherwise.  Ordered outermost/highest first, as Korg.jl orders model
        atmosphere layers (i.e. monotonically decreasing, with
        ``tau_ref`` increasing).
    spherical : bool, optional
        Whether to use spherical ray geometry (default: True).

    Returns
    -------
    path : array, shape (n_mu, n_layers)
        Distance ``s`` along each ray at each layer.  For spherical geometry
        this is ``sqrt(r**2 - b**2)`` on layers the ray intersects; on padded
        layers it continues smoothly and negatively so that the array stays
        strictly decreasing (needed by the Bezier schemes) but the values are
        masked out of every physical sum.
    dsdz : array, shape (n_mu, n_layers)
        ``ds/d(spatial_coord)``: ``r / s`` for spherical geometry, ``1 / μ``
        for plane-parallel geometry.
    mask : array of bool, shape (n_mu, n_layers)
        True where the ray intersects the layer.  Plane-parallel rays
        intersect every layer, so the mask is all True.

    Notes
    -----
    Korg.jl selects the deepest intersected layer with::

        lowest_layer_index = argmin(abs.(r .- b))
        if r[lowest_layer_index] < b; lowest_layer_index -= 1; end

    falling back to the innermost layer when ``b < r[end]``.  Since ``r`` is
    monotonically decreasing, both branches reduce to "the last index with
    ``r >= b``", which is what ``mask`` encodes.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> r = jnp.array([1.10, 1.05, 1.00])
    >>> path, dsdz, mask = calculate_rays(jnp.array([1.0, 0.5]), r)
    >>> mask[0]  # a radial ray reaches the centre-most shell
    Array([ True,  True,  True], dtype=bool)
    >>> bool(mask[1, -1])  # b = 1.10*sqrt(0.75) = 0.9526 < 1.0
    True
    """
    mu = jnp.asarray(mu_surface_grid)
    coord = jnp.asarray(spatial_coord)

    if not spherical:
        # Plane-parallel: s = z / μ, ds/dz = 1/μ, and every ray sees every layer.
        inv_mu = 1.0 / mu
        path = coord[None, :] * inv_mu[:, None]
        dsdz = jnp.broadcast_to(inv_mu[:, None], path.shape)
        mask = jnp.ones(path.shape, dtype=bool)
        return path, dsdz, mask

    r_surface = coord[0]

    # b = r[0] * sqrt(1 - μ**2).  The radicand is floored strictly positive so
    # that the derivative wrt μ stays finite at μ = 1 (b = 0, a radial ray).
    # sqrt(0) is fine as a *value* but its derivative is infinite; substitute a
    # strictly positive number in the dead branch so the cotangent is finite.
    one_minus_mu2 = 1.0 - mu * mu
    tiny = jnp.finfo(jnp.asarray(mu).dtype).tiny
    one_minus_mu2_safe = jnp.where(one_minus_mu2 > tiny, one_minus_mu2, tiny)
    b = r_surface * jnp.sqrt(one_minus_mu2_safe)

    mask = coord[None, :] >= b[:, None]

    radicand = coord[None, :] ** 2 - b[:, None] ** 2
    floor = (_S_FLOOR_FRAC * r_surface) ** 2
    # Double-``where``: the magnitude is floored *before* the square root so
    # neither the value nor the derivative can be non-finite, and the padded
    # branch is a constant so it contributes an exactly zero cotangent.
    magnitude = jnp.abs(radicand)
    magnitude_safe = jnp.where(magnitude > floor, magnitude, floor)
    # Sign continues the path monotonically below the tangent point: for layers
    # the ray misses (r < b) the "path" is negative.  This keeps ``path``
    # strictly decreasing, which the Bezier schemes need, without changing any
    # value the ray actually uses.
    sign = jnp.where(mask, 1.0, -1.0)
    path = sign * jnp.sqrt(magnitude_safe)
    dsdz = coord[None, :] / path

    return path, dsdz, mask
