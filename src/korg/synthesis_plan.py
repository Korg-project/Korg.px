"""
Building a traceable synthesis: the static half, and the closure that carries it.

``synthesize`` has to be one XLA program to be placed on a GPU, ``vmap``ped over
stellar parameters, or differentiated end to end.  Nothing about the *physics*
prevented that — the numerical kernels were already traceable — but a handful of
quantities that decide an array **shape** or a **selection** were being derived
from traced values:

===================================================  ==============================
site                                                 decides
===================================================  ==============================
``float(wavelengths_cm[0])`` (x3)                    which H transitions are in
                                                     range; whether Brackett is;
                                                     the reference-wavelength pixel
``np.arange(wl_min, wl_max, cntm_step)``             the coarse continuum grid
                                                     *length*
``np.asarray(amp)`` -> ``int(in_bucket.sum())``      line-window bucket membership
``np.searchsorted`` + Python slice assignment        the scatter target
``~np.isnan(...)`` in ``interpolate_marcs``          the number of layers
===================================================  ==============================

Every one is a function of the wavelength grid, the linelist and the atmosphere
grid — all fixed before any parameter is varied.  :func:`prepare_synthesis`
computes them once on the host and captures them in a closure.

A closure rather than an object is the point.  Anything a closure captures is
already a compile-time constant to JAX, so there is no pytree to register, no
``aux_data`` to split shapes into, and no ``static_argnames`` at the call site;
retracing happens exactly when a new plan is built, which is exactly when a
shape can change.

What is deliberately *not* frozen matters as much.  The window half-width in
centimetres and the window start pixel are recomputed inside the kernel from the
traced amplitudes; only ``W``, the pixel capacity of each bucket, is fixed, and
that is a memory-allocation choice rather than physics.

This does *not* make the window differentiable, and an earlier version of this
note wrongly claimed it did.  The window reaches the output only through
``mask = |delta| <= win`` and through ``jnp.searchsorted``, a step function and
an integer index, so its derivative is structurally zero -- as it is in any
hard-cutoff scheme, Korg.jl's included.  What recomputing it buys is that the
window *tracks* the parameters under ``vmap`` and away from the plan's reference
point, instead of being frozen at plan-time conditions.  That is a correctness
property, not a gradient one.

The one approximation: if a traced window grows past its bucket's ``W``, the
profile truncates early.  The synthesizer returns ``max_needed_px`` so that can
be asserted outside ``jit`` rather than being silent, and ``window_safety``
buys headroom.
"""

from typing import Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

RYDBERG_CM = 1.0973731568539e5
H_LINE_WINDOW_CM = 150.0 * 1e-8
LAMBDA_REF_CM = 5e-5      # 5000 A, the MARCS reference wavelength
N_MARCS_LAYERS = 56


def _normalise_geometry(geometry):
    """Accept Korg's vocabulary and settle on one spelling.

    ``None`` means "decide from log g at call time, as Korg.jl does" --- one
    plan for dwarfs and giants alike, at the cost of compiling both branches and
    of being non-differentiable across log g = 3.5, which is a discontinuous
    change of model rather than a smooth transition.
    """
    if geometry is None:
        return None
    g = str(geometry).lower().replace("_", "-")
    if g in ("planar", "plane-parallel", "pp"):
        return "plane-parallel"
    if g == "spherical":
        return "spherical"
    raise ValueError(
        f"geometry must be None, 'spherical' or 'plane-parallel', got {geometry!r}")


def _stark_transitions_in_range(wl_min_cm: float, wl_max_cm: float) -> Tuple:
    """Which Stehle Stark transitions can touch this wavelength window."""
    from .hydrogen_line_absorption import hline_stark_profiles
    keys = []
    for k, v in hline_stark_profiles.items():
        lam = 1.0 / (RYDBERG_CM * (1.0 / v.lower ** 2 - 1.0 / v.upper ** 2))
        if wl_min_cm - H_LINE_WINDOW_CM <= lam <= wl_max_cm + H_LINE_WINDOW_CM:
            keys.append(k)
    return tuple(keys)


def _brackett_in_range(wl_min_cm: float, wl_max_cm: float) -> bool:
    return any(
        wl_min_cm - H_LINE_WINDOW_CM
        <= 1.0 / (RYDBERG_CM * (1.0 / 16.0 - 1.0 / m ** 2))
        <= wl_max_cm + H_LINE_WINDOW_CM
        for m in range(5, 31)
    )


def _widen(W: int, safety: float, n_wl: int) -> int:
    """Grow a bucket's pixel capacity to the next power of two past ``safety*W``."""
    target = min(int(np.ceil(W * safety)), n_wl)
    w = 1
    while w < target:
        w *= 2
    return min(w, n_wl)


def _A_X_to_absolute_traced(A_X):
    """Number fractions from A(X), in JAX.

    ``abundances.A_X_to_absolute`` uses ``np.sum``, which cannot consume a
    tracer. This is the same three lines with ``jnp``.
    """
    rel_to_H = 10.0 ** (A_X - 12.0)
    return rel_to_H / jnp.sum(rel_to_H)


def _format_A_X_traced(m_H, alpha_m, C_m, solar_A_X, alpha_mask, c_mask, metal_mask):
    """A(X) from ``[M/H]``, ``[alpha/M]`` and ``[C/M]``, in JAX.

    The element masks are host constants (which Z is an alpha element, which is
    carbon, which is a metal), so only the three offsets are traced. That is what
    lets ``d/d[M/H]`` pick up the composition change as well as the atmosphere
    change — the two paths that ``[M/H]`` drives.
    """
    return solar_A_X + metal_mask * m_H + alpha_mask * alpha_m + c_mask * C_m


class Synthesizer:
    """A compiled-shape synthesis: call it with stellar parameters, get a spectrum.

    Returned by :func:`prepare_synthesis`. Valid only for the wavelength grid,
    linelist, geometry and layer count it was built with — a different grid means
    building another one, and a shape mismatch raises rather than broadcasting.

    Examples
    --------
    >>> synth = prepare_synthesis(wavelengths_cm, linelist, data)
    >>> flux, cntm = synth(5777.0, 4.44, 0.0)
    >>> dflux_dTeff = jax.grad(lambda t: synth(t, 4.44, 0.0)[0].sum())(5777.0)
    >>> dflux_dA    = jax.grad(lambda a: synth(5777.0, 4.44, abundances=a)[0].sum())(A)
    """

    def __init__(self, *, wavelengths_cm, cntm_wl_cm, linelist_data, data,
                 bucket_line_idx, bucket_widths, stark_keys, brackett_in_range,
                 ref_pixel, geometry, wl_spacing, n_layers, marcs_arrays,
                 solar_A_X, alpha_mask, c_mask, metal_mask,
                 n_mu=20):
        self.wavelengths_cm = wavelengths_cm
        self.cntm_wl_cm = cntm_wl_cm
        self.linelist_data = linelist_data
        self.data = data
        self.bucket_line_idx = bucket_line_idx
        self.bucket_widths = bucket_widths
        self.stark_keys = stark_keys
        self.brackett_in_range = brackett_in_range
        self.ref_pixel = ref_pixel
        self.geometry = geometry
        self.wl_spacing = wl_spacing
        self.n_layers = n_layers
        self.n_wl = int(np.shape(wavelengths_cm)[0])
        self.n_lines = int(np.shape(linelist_data.wl)[0])
        self._marcs = marcs_arrays
        self._solar_A_X = solar_A_X
        self._alpha_mask = alpha_mask
        self._c_mask = c_mask
        self._metal_mask = metal_mask
        self.n_mu = n_mu

    # -- the traced entry points ------------------------------------------------

    def from_atmosphere(self, T, n_total, ne, z, log_tau_ref, abundances,
                        vmic_cm_s=1e5, R_photosphere=None, logg=None):
        """Synthesize from atmosphere arrays. Traced; differentiable in all six.

        Use this when the atmosphere does not come from the MARCS grid — a model
        read from a file, or one produced by something else entirely.
        """
        from .traced_synthesis import _synthesize_traced
        return _synthesize_traced(self, T, n_total, ne, z, log_tau_ref,
                                  abundances, vmic_cm_s,
                                  R_photosphere=R_photosphere, logg=logg)

    def __call__(self, Teff, logg, m_H=0.0, alpha_m=0.0, C_m=0.0,
                 abundances=None, vmic_cm_s=1e5):
        """Synthesize from stellar parameters. Traced; differentiable in all of them.

        The MARCS interpolation runs *inside* the traced region, so
        ``jax.grad(..., argnums=0)`` is d(flux)/d(Teff) through interpolation,
        chemical equilibrium, opacity and radiative transfer in one pass.

        ``abundances`` defaults to the composition implied by ``(m_H, alpha_m,
        C_m)``, so differentiating with respect to ``m_H`` picks up both the
        atmospheric structure change and the composition change. Passing it
        explicitly overrides that and leaves ``m_H`` acting on structure alone.
        """
        from .traced_synthesis import _interpolate_marcs_traced, _synthesize_traced

        if abundances is None:
            A_X = _format_A_X_traced(m_H, alpha_m, C_m, self._solar_A_X,
                                     self._alpha_mask, self._c_mask,
                                     self._metal_mask)
            abundances = _A_X_to_absolute_traced(A_X)

        T, n_total, ne, z, log_tau_ref, R_phot = _interpolate_marcs_traced(
            self, Teff, logg, m_H, alpha_m, C_m)
        return _synthesize_traced(self, T, n_total, ne, z, log_tau_ref,
                                  abundances, vmic_cm_s,
                                  R_photosphere=R_phot, logg=logg)

    def __repr__(self):
        return (f"Synthesizer(n_wl={self.n_wl}, n_layers={self.n_layers}, "
                f"n_lines={self.n_lines}, buckets={self.bucket_widths}, "
                f"geometry={self.geometry!r})")


def prepare_synthesis(
    wavelengths_cm,
    linelist,
    data=None,
    *,
    geometry: Optional[str] = None,
    cntm_step_cm: float = 1e-8,
    window_safety: float = 2.0,
    reference: Tuple[float, float, float] = (5777.0, 4.44, 0.0),
    n_layers: int = N_MARCS_LAYERS,
    n_mu: int = 20,
) -> Synthesizer:
    """Fix every shape a synthesis needs, and return a callable that does the rest.

    Parameters
    ----------
    wavelengths_cm : (n_wl,) array
        Synthesis grid. Concrete — this is host code.
    linelist : list of Line, or LinelistData
        The lines to synthesize.
    data : SynthesisData, optional
        Loaded from disk if omitted.
    geometry : {'planar', 'spherical'}
        Radiative transfer geometry. Baked in, so the two compile separately and
        neither pays for the other's branches.
    cntm_step_cm : float
        Spacing of the coarse grid the continuum is evaluated on.
    window_safety : float
        Headroom on each line's pixel window. The windows are *measured* at the
        reference parameters below rather than guessed, so this is only for how
        far the plan must stay valid: windows grow with abundance and with
        pressure broadening, and 2.0 covers roughly a dex. Check
        ``max_needed_px`` from a call to be sure.
    reference : (Teff, logg, [M/H])
        Where the window sizes are measured. Pick something near the middle of
        the range you intend to explore.
    n_layers : int
        Layers in the atmosphere. 56 for the MARCS grid, which is the count at
        every grid point checked.

    Returns
    -------
    Synthesizer
    """
    geometry = _normalise_geometry(geometry)

    from .synthesis import load_synthesis_data, preprocess_linelist, precompute_atmosphere
    from .marcs_interpolation import _get_marcs_jit_data
    from .abundances import format_A_X, A_X_to_absolute, DEFAULT_ALPHA_ELEMENTS
    from .atomic_data import MAX_ATOMIC_NUMBER

    wl_np = np.asarray(wavelengths_cm, dtype=np.float64)
    n_wl = int(wl_np.shape[0])
    if n_wl < 2:
        raise ValueError("need at least two wavelength points")
    wl_spacing = float(np.median(np.diff(wl_np)))
    wl_min_cm, wl_max_cm = float(wl_np[0]), float(wl_np[-1])

    if data is None:
        data = load_synthesis_data()

    linelist_data = (linelist if hasattr(linelist, "wl")
                     else preprocess_linelist(linelist, data.chem_eq_data, wl_np))

    cntm_wl = np.arange(wl_min_cm - cntm_step_cm,
                        wl_max_cm + 2 * cntm_step_cm, cntm_step_cm)

    # Measure the line windows at the reference parameters rather than guessing
    # them from log_gf. precompute_atmosphere already does exactly this work on
    # the host, and it is the code the NumPy path was validated against, so the
    # bucket partition here is the same one that path would have chosen.
    Teff_r, logg_r, m_H_r = reference
    A_X_r = format_A_X(m_H_r)
    ab_r = jnp.asarray(A_X_to_absolute(A_X_r))
    from .marcs_interpolation import interpolate_marcs
    atm_r = interpolate_marcs(Teff_r, logg_r, m_H_r)
    T_r = jnp.asarray([l.temperature for l in atm_r.layers])
    nt_r = jnp.asarray([l.number_density for l in atm_r.layers])
    ne_r = jnp.asarray([l.electron_number_density for l in atm_r.layers])
    z_r = jnp.asarray([l.z for l in atm_r.layers])
    lt_r = jnp.log(jnp.asarray([l.tau_ref for l in atm_r.layers]))

    pre_r = precompute_atmosphere(jnp.asarray(wl_np), T_r, nt_r, ne_r, z_r, lt_r,
                                  ab_r, 1e5, data, linelist_data)
    bgs = pre_r.bucket_geometry or []
    bucket_line_idx = tuple(np.asarray(bg.amp_idx, dtype=np.int32) for bg in bgs)
    bucket_widths = tuple(_widen(int(bg.W), window_safety, n_wl) for bg in bgs)

    ref_pixel = None
    if wl_min_cm <= LAMBDA_REF_CM <= wl_max_cm:
        ref_pixel = int(np.argmin(np.abs(wl_np - LAMBDA_REF_CM)))

    # Element masks for the traced [M/H] / [alpha/M] / [C/M] offsets.
    Z = np.arange(1, MAX_ATOMIC_NUMBER + 1)
    metal_mask = jnp.asarray((Z >= 3).astype(np.float64))
    alpha_mask = jnp.asarray(np.isin(Z, DEFAULT_ALPHA_ELEMENTS).astype(np.float64))
    c_mask = jnp.asarray((Z == 6).astype(np.float64))

    return Synthesizer(
        wavelengths_cm=jnp.asarray(wl_np),
        cntm_wl_cm=jnp.asarray(cntm_wl),
        linelist_data=linelist_data,
        data=data,
        bucket_line_idx=bucket_line_idx,
        bucket_widths=bucket_widths,
        stark_keys=_stark_transitions_in_range(wl_min_cm, wl_max_cm),
        brackett_in_range=_brackett_in_range(wl_min_cm, wl_max_cm),
        ref_pixel=ref_pixel,
        geometry=geometry,
        wl_spacing=wl_spacing,
        n_layers=int(n_layers),
        marcs_arrays=_get_marcs_jit_data(),
        solar_A_X=jnp.asarray(format_A_X(0.0)),
        alpha_mask=alpha_mask,
        c_mask=c_mask,
        metal_mask=metal_mask,
        n_mu=n_mu,
    )


def synthesize(atmosphere, linelist, wavelengths_angstrom, A_X, *,
               vmic=1.0, geometry=None, return_cntm=True, **plan_kwargs):
    """Korg.jl-compatible one-shot synthesis.

    Matches Korg.jl's ``synthesize(atm, linelist, A_X, wavelengths)`` ordering and
    semantics so ported scripts read the same. It builds a
    :class:`Synthesizer` and calls it once.

    **Prefer** :func:`prepare_synthesis` whenever you synthesize more than once.
    The plan is the expensive half --- filtering the linelist, sizing the line
    windows, selecting the hydrogen transitions --- and this function throws it
    away after a single call. Two syntheses through ``prepare_synthesis`` cost
    roughly what one costs here; a thousand cost barely more.

    Parameters
    ----------
    atmosphere : PlanarAtmosphere or ShellAtmosphere
        Model atmosphere. Its layers supply T, n, n_e, z and tau_ref.
    linelist : list of Line
    wavelengths_angstrom : (n_wl,) array
        Synthesis grid in Angstroms.
    A_X : (92,) array
        Abundances as A(X) = log10(N_X/N_H) + 12.
    vmic : float
        Microturbulence in km/s (Korg's unit), converted to cm/s internally.
    geometry : None, 'spherical' or 'plane-parallel'
        ``None`` follows the atmosphere type: a ShellAtmosphere is spherical.
    return_cntm : bool
        Return ``(flux, continuum)`` if True, else just ``flux``.

    Returns
    -------
    flux, continuum : (n_wl,) arrays, or flux alone if ``return_cntm`` is False.
    """
    from .abundances import A_X_to_absolute

    wl_cm = np.asarray(wavelengths_angstrom, dtype=np.float64) * 1e-8
    layers = atmosphere.layers
    n_layers = len(layers)

    if geometry is None:
        # Follow the atmosphere the caller actually handed us rather than
        # guessing from log g -- they have already made the choice by building a
        # ShellAtmosphere or a PlanarAtmosphere.
        geometry = "spherical" if hasattr(atmosphere, "R_photosphere") else "plane-parallel"

    synth = prepare_synthesis(wl_cm, linelist, geometry=geometry,
                              n_layers=n_layers, **plan_kwargs)

    T = jnp.asarray([l.temperature for l in layers])
    n_total = jnp.asarray([l.number_density for l in layers])
    ne = jnp.asarray([l.electron_number_density for l in layers])
    z = jnp.asarray([l.z for l in layers])
    log_tau = jnp.log(jnp.asarray([l.tau_ref for l in layers]))
    abundances = jnp.asarray(A_X_to_absolute(np.asarray(A_X)))

    R = getattr(atmosphere, "R_photosphere", None)
    flux, cntm = synth.from_atmosphere(T, n_total, ne, z, log_tau, abundances,
                                       vmic_cm_s=vmic * 1e5, R_photosphere=R)
    return (flux, cntm) if return_cntm else flux
