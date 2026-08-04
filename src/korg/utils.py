"""
Utility functions for Korg.

Simple mathematical and physics utility functions including:
- Air <-> vacuum wavelength conversion (the single implementation; re-exported
  by ``korg.wavelengths`` and ``korg.linelist``)
- LSF (Line Spread Function) convolution
- Rotational broadening
- Normal PDF
- Translational partition function

Reference: Korg.jl utils.jl
"""

import warnings
from bisect import bisect_left, bisect_right
from typing import Callable, Union

import jax
import jax.numpy as jnp
import numpy as np

from .constants import c_cgs, hplanck_cgs, kboltz_cgs

# NOTE: ``Wavelengths`` is imported lazily inside the functions that need it.
# ``korg.wavelengths`` imports ``air_to_vacuum``/``vacuum_to_air`` from this
# module, so a module-level import here would be circular.


# =============================================================================
# Air <-> vacuum conversion
#
# This is the *only* implementation in Korg.px.  ``korg.wavelengths`` and
# ``korg.linelist`` re-export these names so that every historical import path
# keeps working.
# =============================================================================

def _is_jax(x) -> bool:
    """Whether `x` is a JAX array or tracer (and so must stay in JAX-land)."""
    return isinstance(x, (jax.core.Tracer, jax.Array))


def _match_input_type(result, reference):
    """
    Demote `result` to NumPy/Python when `reference` was not a JAX value.

    Keeps the historical return types (``float`` for scalar input, ``ndarray``
    for array input) while letting gradients flow when the caller is tracing.
    """
    if _is_jax(reference):
        return result
    if np.ndim(reference) == 0:
        return float(result)
    return np.asarray(result)


def _cgs_scale(wavelength, cgs):
    """
    Factor converting `wavelength` to Å.

    Mirrors Korg.jl's ``cgs=λ<1`` default: λ is assumed to be in Å if it is
    ⩾ 1 and in cm otherwise.  The auto-detected version is expressed with
    ``jnp.where`` so that it stays traceable and element-wise, exactly like
    Korg.jl's broadcasted call.
    """
    if cgs is None:
        return jnp.where(jnp.asarray(wavelength) < 1, 1e8, 1.0)
    return 1e8 if cgs else 1.0


def air_to_vacuum(wavelength, cgs=None):
    """
    Convert wavelength from air to vacuum.

    Formula from Birch and Downs (1994) via the VALD website. Valid for
    wavelengths > 2000 Å.

    Parameters
    ----------
    wavelength : float or array
        Wavelength in air. Assumed to be in Å if ⩾ 1, in cm otherwise.
    cgs : bool, optional
        If True, `wavelength` is in cm; if False, in Å. The default (None)
        auto-detects element-wise, matching Korg.jl's ``cgs=λ<1``.

    Returns
    -------
    float or array
        Wavelength in vacuum, in the same units as the input.

    See Also
    --------
    vacuum_to_air : The inverse conversion.
    """
    scale = _cgs_scale(wavelength, cgs)
    lam = wavelength * scale  # Å
    s = 1e4 / lam
    n = (1 + 0.00008336624212083
         + 0.02408926869968 / (130.1065924522 - s**2)
         + 0.0001599740894897 / (38.92568793293 - s**2))
    return _match_input_type(lam * n / scale, wavelength)


def vacuum_to_air(wavelength, cgs=None):
    """
    Convert wavelength from vacuum to air.

    Formula from Birch and Downs (1994) via the VALD website. Valid for
    wavelengths > 2000 Å.

    Parameters
    ----------
    wavelength : float or array
        Wavelength in vacuum. Assumed to be in Å if ⩾ 1, in cm otherwise.
    cgs : bool, optional
        If True, `wavelength` is in cm; if False, in Å. The default (None)
        auto-detects element-wise, matching Korg.jl's ``cgs=λ<1``.

    Returns
    -------
    float or array
        Wavelength in air, in the same units as the input.

    See Also
    --------
    air_to_vacuum : The inverse conversion.
    """
    scale = _cgs_scale(wavelength, cgs)
    lam = wavelength * scale  # Å
    s = 1e4 / lam
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
    return _match_input_type(lam / n / scale, wavelength)


def normal_pdf(delta, sigma):
    """
    Probability density function of a normal distribution.

    Parameters
    ----------
    delta : float or array
        Deviation from mean (x - μ).
    sigma : float
        Standard deviation.

    Returns
    -------
    float or array
        PDF value(s).
    """
    return jnp.exp(-0.5 * delta**2 / sigma**2) / jnp.sqrt(2 * jnp.pi) / sigma


def translational_U(m, T):
    """
    Translational contribution to the partition function.

    Used in the Saha equation for ionization equilibrium. This represents
    the partition function contribution from the free movement of a particle.

    Parameters
    ----------
    m : float
        Particle mass in grams.
    T : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Translational partition function contribution.

    Notes
    -----
    The formula is: (2πmkT/h²)^(3/2)
    """
    return (2 * jnp.pi * m * kboltz_cgs * T / hplanck_cgs**2)**1.5


# =============================================================================
# Interval utilities (for bounds checking in continuum absorption)
# =============================================================================

def _nextfloat_skipsubnorm(v):
    """
    Return the next floating point value after v, skipping subnormals.

    Matches Julia's behavior for interval boundary handling.
    """
    floatmin = np.finfo(np.float64).tiny  # smallest positive normalized float
    if -floatmin <= v < 0:
        return 0.0
    elif 0 <= v < floatmin:
        return floatmin
    else:
        return np.nextafter(v, np.inf)


def _prevfloat_skipsubnorm(v):
    """
    Return the previous floating point value before v, skipping subnormals.

    Matches Julia's behavior for interval boundary handling.
    """
    floatmin = np.finfo(np.float64).tiny  # smallest positive normalized float
    if -floatmin < v <= 0:
        return -floatmin
    elif 0 < v <= floatmin:
        return 0.0
    else:
        return np.nextafter(v, -np.inf)


class Interval:
    """
    Represents an interval with configurable exclusive/inclusive bounds.

    By default, both bounds are exclusive (open interval).
    Use closed_interval() for inclusive bounds.

    Parameters
    ----------
    lower : float
        Lower bound of the interval.
    upper : float
        Upper bound of the interval.
    exclusive_lower : bool, optional
        If True (default), the lower bound is exclusive.
    exclusive_upper : bool, optional
        If True (default), the upper bound is exclusive.

    Examples
    --------
    >>> interval = Interval(3.0, 10.0)  # exclusive: (3, 10)
    >>> contained(5.0, interval)
    True
    >>> contained(3.0, interval)
    False
    """

    def __init__(self, lower, upper, exclusive_lower=True, exclusive_upper=True):
        if not lower < upper:
            raise ValueError("the upper bound must exceed the lower bound")

        lower, upper = float(lower), float(upper)

        # Adjust bounds for inclusive intervals (matching Julia exactly)
        if exclusive_lower or np.isinf(lower):
            self.lower = lower
        else:
            self.lower = _prevfloat_skipsubnorm(lower)

        if exclusive_upper or np.isinf(upper):
            self.upper = upper
        else:
            self.upper = _nextfloat_skipsubnorm(upper)


def closed_interval(lo, up):
    """
    Create an interval where both bounds are inclusive.

    Parameters
    ----------
    lo : float
        Lower bound (inclusive).
    up : float
        Upper bound (inclusive).

    Returns
    -------
    Interval
        An interval [lo, up] (closed on both ends).

    Examples
    --------
    >>> interval = closed_interval(3.0, 10.0)  # inclusive: [3, 10]
    >>> contained(3.0, interval)
    True
    >>> contained(10.0, interval)
    True
    """
    return Interval(lo, up, exclusive_lower=False, exclusive_upper=False)


def contained(value, interval):
    """
    Check whether a value is contained within an interval.

    Parameters
    ----------
    value : float
        The value to check.
    interval : Interval
        The interval to check against.

    Returns
    -------
    bool
        True if the value is within the interval bounds.

    Examples
    --------
    >>> contained(5.0, Interval(1.0, 10.0))
    True
    >>> contained(0.5, Interval(1.0, 10.0))
    False
    """
    return interval.lower < value < interval.upper


def contained_slice(vals, interval):
    """
    Get the slice indices for values contained in an interval.

    Returns a tuple (start, end) denoting the indices of elements in `vals`
    (assumed to be sorted in increasing order) that are contained by the
    interval. When no entries are contained, returns an empty range.

    Parameters
    ----------
    vals : array-like
        Sorted array of values.
    interval : Interval
        The interval to check against.

    Returns
    -------
    tuple
        (start_index, end_index) for slicing. Use vals[start:end] to get
        the contained values.

    Examples
    --------
    >>> vals = [1.0, 2.0, 5.0, 8.0, 12.0]
    >>> interval = Interval(3.0, 10.0)
    >>> start, end = contained_slice(vals, interval)
    >>> vals[start:end]
    [5.0, 8.0]
    """
    # Use bisect to find indices - matches Julia's searchsortedfirst/searchsortedlast
    start = bisect_right(vals, interval.lower)  # First index > lower bound
    end = bisect_left(vals, interval.upper)  # First index >= upper bound
    return start, end


# =============================================================================
# Post-processing: LSF convolution and rotational broadening
#
# These are written in JAX so that ``jax.grad`` flows both through the flux
# vector *and* through the shape parameters (``R``, ``vsini``) that
# ``korg.fit`` optimises.
#
# Korg.jl builds a variable-length convolution window per output pixel with
# ``searchsortedfirst``/``searchsortedlast``.  Array shapes cannot depend on a
# traced value, so instead we gather a *fixed* number of neighbours per output
# pixel (a static worst-case half-width computed from the wavelength grid and
# the concrete parameter value) and zero the kernel outside the true window
# with a mask.  The kernel is renormalised over the masked weights, so the
# result is numerically identical to the variable-length version, while the
# weights themselves -- and therefore the derivative with respect to R/vsini --
# stay differentiable.
# =============================================================================

# Converts an LSF FWHM, Δλ = λ0/R, to a Gaussian standard deviation.
_FWHM_TO_SIGMA_DENOM = 2 * np.sqrt(2 * np.log(2))

# Cap on the number of elements in a single (n_pixels, window) intermediate.
# Output pixels are processed in blocks no larger than this, so peak memory is
# bounded by the block size (~8 MB per float64 intermediate) rather than by
# n_pixels * window -- which would be hundreds of MB for a million-pixel
# synthesis convolved at R ~ 20000.
_MAX_WINDOW_ELEMENTS = 1 << 20


def _static_np(value, what="value"):
    """
    Concrete float64 NumPy view of `value`, for computing static window sizes.

    Convolution window *shapes* cannot depend on a traced value, so the window
    half-width is derived from the concrete value of R (or vsini). Under eager
    ``jax.grad`` the primal is concrete and this succeeds; under ``jax.jit``
    with the parameter as a traced argument it cannot, and we say so loudly
    rather than silently truncating the kernel.
    """
    try:
        return np.asarray(value, dtype=np.float64)
    except Exception:
        pass
    try:
        return np.asarray(jax.lax.stop_gradient(value), dtype=np.float64)
    except Exception as exc:
        raise TypeError(
            f"{what} must have a concrete value: the convolution window size "
            f"is a static quantity. Got {type(value).__name__}. Under jax.jit, "
            f"pass {what} via static_argnums (or close over it); eager "
            "jax.grad/jax.jacfwd work as-is."
        ) from exc


def _output_like(result):
    """Return `result` as a NumPy array unless it is still being traced."""
    if isinstance(result, jax.core.Tracer):
        return result
    return np.asarray(result)


def _row_blocks(n_rows: int, width: int):
    """Yield (start, stop) row blocks keeping intermediates under the cap."""
    per_block = max(1, _MAX_WINDOW_ELEMENTS // max(width, 1))
    for start in range(0, n_rows, per_block):
        yield start, min(start + per_block, n_rows)


def _window_indices(centers: np.ndarray, half_width: int, n: int):
    """
    Fixed-size gather indices around `centers`.

    Parameters
    ----------
    centers : ndarray of int
        Index of the grid point each window is centred on.
    half_width : int
        Number of neighbours gathered on each side.
    n : int
        Length of the grid being gathered from.

    Returns
    -------
    offsets : ndarray of int, shape (2 * half_width + 1,)
    indices : ndarray of int, shape (len(centers), 2 * half_width + 1)
        Clipped into ``[0, n - 1]`` so the gather is always in bounds.
    valid : ndarray of bool, same shape as `indices`
        False where the unclipped index fell off the end of the grid.
    """
    offsets = np.arange(-half_width, half_width + 1)
    # int32 keeps the (n_out, 2 * half_width + 1) index block half the size of
    # the default int64; wavelength grids never approach 2**31 points.
    raw = (centers[:, None] + offsets[None, :]).astype(np.int32)
    valid = (raw >= 0) & (raw < n)
    return offsets, np.clip(raw, 0, n - 1), valid


def _resolve_R(R: Union[float, Callable], lambda0):
    """
    Resolve R to a value based on its type.

    Parameters
    ----------
    R : float or callable
        Resolving power. If callable, it's called with λ in Å.
    lambda0 : float or array
        Wavelength(s) in cm.

    Returns
    -------
    float or array
        Resolved R value(s).
    """
    if callable(R):
        return R(lambda0 * 1e8)  # R is a function of λ in Å
    return R


def _resolve_R_values(R: Union[float, Callable], wl_cm: np.ndarray):
    """
    Resolving power at every wavelength in `wl_cm` (cm).

    A callable R is tried vectorised first (so that JAX-friendly functions stay
    traceable) and falls back to an element-wise call for scalar-only
    functions, matching Korg.jl's per-pixel ``_resolve_R``.
    """
    if not callable(R):
        return R
    try:
        values = _resolve_R(R, wl_cm)
        if np.ndim(values) != 0 and np.shape(values) != wl_cm.shape:
            raise ValueError("callable R did not return one value per wavelength")
        return values
    except Exception:
        return jnp.asarray([_resolve_R(R, float(lam)) for lam in wl_cm])


def _lsf_half_width(synth_wl: np.ndarray, centers: np.ndarray,
                    lambda0: np.ndarray, R_values, window_size: float):
    """
    Static worst-case window half-width, and the count of degenerate rows.

    The half-width is derived from the *concrete* value of R (see
    :func:`_static_np`): array shapes cannot depend on a traced quantity, but
    the mask applied inside the window is computed from the traced value, so
    the result is exact and stays differentiable.

    A row is "degenerate" when ``lb > ub``, i.e. no grid point at all lies
    inside the kernel window -- the LSF is narrower than the grid spacing.
    """
    half_static = window_size * (lambda0 / _static_np(R_values, "R")
                                 / _FWHM_TO_SIGMA_DENOM)
    lb = np.searchsorted(synth_wl, lambda0 - half_static, side="left")
    ub = np.searchsorted(synth_wl, lambda0 + half_static, side="right") - 1
    half_width = int(max(
        np.max(centers - lb, initial=0), np.max(ub - centers, initial=0), 0
    )) + 1
    return min(half_width, max(len(synth_wl) - 1, 1)), int(np.sum(lb > ub))


def _lsf_weight_blocks(synth_wl: np.ndarray, centers: np.ndarray,
                       lambda0: np.ndarray, R, window_size: float,
                       fallback_to_nearest: bool):
    """
    Yield fixed-size, differentiable Gaussian LSF weights, block by block.

    Parameters
    ----------
    synth_wl : ndarray
        Synthesis wavelength grid in cm (a static NumPy array).
    centers : ndarray of int
        Index into `synth_wl` that each window is centred on. For
        :func:`compute_LSF_matrix` this is the nearest synthesis pixel, which
        guarantees the nearest-neighbour fallback below is reachable.
    lambda0 : ndarray
        Output wavelengths in cm (static).
    R : float, array or callable
        Resolving power.
    window_size : float
        Kernel extent in units of sigma.
    fallback_to_nearest : bool
        If True, rows whose window contains no synthesis pixel fall back to
        nearest-neighbour interpolation instead of producing an all-zero row.
        This is the physically sensible answer: an LSF finer than the synthesis
        sampling cannot be resolved, and the alternative is a silently
        all-zero row.

    Yields
    ------
    indices : ndarray of int32, shape (rows, 2 * half_width + 1)
        Gather indices into the synthesis grid for this block of output rows.
    weights : array, same shape
        Normalised kernel weights; every row sums to one.

    Notes
    -----
    Output rows are emitted in blocks so that peak memory is bounded by
    ``_MAX_WINDOW_ELEMENTS`` rather than by ``n_out * window``.
    """
    n_synth = len(synth_wl)
    R_values = _resolve_R_values(R, lambda0)
    sigma = lambda0 / R_values / _FWHM_TO_SIGMA_DENOM
    half = window_size * sigma
    lo_all, hi_all = lambda0 - half, lambda0 + half

    half_width, _ = _lsf_half_width(synth_wl, centers, lambda0, R_values,
                                    window_size)

    for start, stop in _row_blocks(len(centers), 2 * half_width + 1):
        offsets, indices, valid = _window_indices(
            centers[start:stop], half_width, n_synth
        )
        lam_window = synth_wl[indices]
        lam0 = lambda0[start:stop, None]
        # Exactly reproduces searchsortedfirst/searchsortedlast on the grid.
        in_window = (valid & (lam_window >= lo_all[start:stop, None])
                     & (lam_window <= hi_all[start:stop, None]))

        # Zero the deviation outside the window *before* the Gaussian, so the
        # dead branch is finite: exp(-huge) underflows to 0 while its cotangent
        # factor delta**2 / sigma**3 blows up, and 0 * inf is NaN.
        delta = jnp.where(in_window, lam_window - lam0, 0.0)
        phi = jnp.where(in_window, normal_pdf(delta, sigma[start:stop, None]), 0.0)

        total = jnp.sum(phi, axis=1, keepdims=True)
        if fallback_to_nearest:
            safe = jnp.where(total > 0, total, 1.0)
            onehot = np.broadcast_to((offsets == 0), indices.shape)
            weights = jnp.where(total > 0, phi / safe, onehot.astype(phi.dtype))
        else:
            weights = phi / total
        yield indices, weights


def apply_LSF(flux, wls, R: Union[float, Callable], window_size: float = 4):
    """
    Apply a Gaussian line spread function to a spectrum.

    Convolves the spectrum with flux vector `flux` and wavelengths `wls`
    with a Gaussian LSF of resolving power R (R = λ/Δλ, where Δλ is FWHM).

    Parameters
    ----------
    flux : array
        The flux vector to convolve.
    wls : tuple, list of tuples, array, or Wavelengths
        Wavelengths in any format accepted by Wavelengths class.
    R : float or callable
        The resolving power R = λ/Δλ. Can be a constant or a function
        of wavelength (in Å).
    window_size : float, optional
        How far to extend the convolution kernel in units of sigma
        (not HWHM). Default: 4.

    Returns
    -------
    array
        Convolved flux vector.

    Notes
    -----
    - For multiple spectra on the same wavelength grid, compute_LSF_matrix
      is faster: it builds the kernel once and reduces each convolution to a
      matrix multiply.
    - apply_LSF will have weird behavior if your wavelength grid is not
      locally linearly-spaced. Run on a fine grid, then downsample.
    - For best results, extend your wavelength range a couple Δλ outside
      the region you will compare to data.
    - Differentiable: ``jax.grad`` works with respect to `flux` and to `R`.

    Examples
    --------
    >>> flux_convolved = apply_LSF(flux, (5000, 5500, 0.01), R=50000)
    """
    from .wavelengths import Wavelengths

    if not callable(R) and np.all(np.isinf(_static_np(R, "R"))):
        return _output_like(jnp.asarray(flux))

    wls = Wavelengths(wls)
    synth_wl = np.asarray(wls.all_wls)
    centers = np.arange(len(synth_wl))
    flux = jnp.asarray(flux)

    blocks = [
        jnp.sum(weights * flux[indices], axis=1)
        for indices, weights in _lsf_weight_blocks(
            synth_wl, centers, synth_wl, R, window_size,
            fallback_to_nearest=False,
        )
    ]
    return _output_like(blocks[0] if len(blocks) == 1
                        else jnp.concatenate(blocks, axis=0))


def compute_LSF_matrix(synth_wls, obs_wls, R: Union[float, Callable],
                       window_size: float = 4, verbose: bool = True):
    """
    Compute a matrix to apply an LSF to synthesis spectra.

    Given synthesis wavelengths `synth_wls` and observation wavelengths `obs_wls`,
    compute a matrix `LSF` such that `LSF @ flux` convolves the synthetic spectrum
    with a Gaussian LSF of resolving power R.

    This is more efficient than apply_LSF when you need to convolve many spectra
    on the same wavelength grid.

    Parameters
    ----------
    synth_wls : tuple, list of tuples, array, or Wavelengths
        Synthesis wavelengths in any format accepted by Wavelengths class.
    obs_wls : array
        Observation wavelengths. If values >= 1, assumed to be in Angstroms
        and will be converted to cm.
    R : float or callable
        The resolving power R = λ/Δλ. Can be a constant or a function
        of wavelength (in Å).
    window_size : float, optional
        How far to extend the convolution kernel in units of sigma
        (not HWHM). Default: 4.
    verbose : bool, optional
        Whether to emit warnings. Default: True.

    Returns
    -------
    array
        LSF matrix with shape (n_obs, n_synth). Apply with: convolved = LSF @ flux

    Notes
    -----
    The returned matrix is sparse in Julia but dense in Python/JAX for JIT
    compatibility. For best results, synthesis wavelengths should extend a
    couple Δλ outside the observation range.

    Every row is a normalised weighting, including when the LSF is narrower
    than the synthesis grid spacing. In that regime an observed wavelength can
    fall between two synthesis pixels with *no* synthesis pixel inside the
    kernel window; the row then falls back to nearest-neighbour interpolation
    (and, with ``verbose=True``, warns) rather than silently becoming zero.

    Examples
    --------
    >>> lsf_matrix = compute_LSF_matrix((5000, 5100, 0.01), obs_wls, R=50000)
    >>> convolved_flux = lsf_matrix @ flux
    """
    from .wavelengths import Wavelengths

    obs_wls = np.asarray(obs_wls, dtype=np.float64)
    if obs_wls[0] >= 1:
        obs_wls = obs_wls / 1e8  # Å to cm

    synth_wls = Wavelengths(synth_wls)
    synth_wl = np.asarray(synth_wls.all_wls)

    if verbose:
        synth_first, synth_last = synth_wl[0], synth_wl[-1]
        obs_first, obs_last = obs_wls[0], obs_wls[-1]
        margin = 0.01  # cm (~1000 Å)
        if not ((synth_first - margin) <= obs_first <= obs_last <= (synth_last + margin)):
            warnings.warn(
                f"Synthesis wavelengths ({synth_first*1e8:.1f} Å—{synth_last*1e8:.1f} Å) "
                f"are not superset of observation wavelengths "
                f"({obs_first*1e8:.1f} Å—{obs_last*1e8:.1f} Å) in LSF matrix."
            )

    # Centre each window on the nearest synthesis pixel so that the
    # nearest-neighbour fallback is always inside the gathered window.
    right = np.searchsorted(synth_wl, obs_wls, side="left")
    left = np.clip(right - 1, 0, len(synth_wl) - 1)
    right = np.clip(right, 0, len(synth_wl) - 1)
    centers = np.where(
        np.abs(synth_wl[right] - obs_wls) < np.abs(obs_wls - synth_wl[left]),
        right, left,
    )

    # Resolve a callable R once: it is needed both for the warning below and
    # for the weights, and it may be an expensive per-pixel Python call.
    R = _resolve_R_values(R, obs_wls)

    if verbose:
        _, n_degenerate = _lsf_half_width(synth_wl, centers, obs_wls, R,
                                          window_size)
        if n_degenerate:
            warnings.warn(
                f"The LSF is narrower than the synthesis grid spacing for "
                f"{n_degenerate} of {len(obs_wls)} observed wavelengths: no "
                "synthesis pixel falls inside the kernel window, so those rows "
                "fall back to nearest-neighbour interpolation. The synthesis "
                "grid is too coarse for the requested R."
            )

    LSF = jnp.zeros((len(obs_wls), len(synth_wl)), dtype=jnp.result_type(float))
    row = 0
    for indices, weights in _lsf_weight_blocks(
        synth_wl, centers, obs_wls, R, window_size, fallback_to_nearest=True
    ):
        rows = np.repeat(np.arange(row, row + len(indices), dtype=np.int32),
                         indices.shape[1])
        LSF = LSF.at[rows, indices.ravel()].add(weights.ravel())
        row += len(indices)
    return _output_like(LSF)


def _rotation_kernel_integral(c1, c2, c3, detuning, delta_lambda_rot):
    """
    Indefinite integral of the rotation kernel.

    Parameters
    ----------
    c1, c2 : float
        Limb-darkening constants, 2(1 - ε) and πε/2.
    c3 : float or array
        π(1 - ε/3) * delta_lambda_rot (the denominator).
    detuning : array
        Wavelength detuning from line centre, in cm.
    delta_lambda_rot : array
        Rotational broadening half-width, in cm.

    Returns
    -------
    array
        Integral value.

    Notes
    -----
    At ``|detuning| = delta_lambda_rot`` the closed form is exactly
    ``sign(detuning) / 2``, but ``sqrt(1 - x**2)`` and ``arcsin(x)`` both have
    infinite derivatives there and the two infinities cancel only analytically.
    The special case is therefore taken with a double-``where``: the argument is
    forced to zero in the dead branch so no NaN can reach the cotangent.
    """
    at_edge = jnp.abs(detuning) >= delta_lambda_rot
    d = jnp.where(at_edge, 0.0, detuning)
    ratio = d / delta_lambda_rot
    interior = (0.5 * c1 * d * jnp.sqrt(1 - ratio**2)
                + 0.5 * c1 * delta_lambda_rot * jnp.arcsin(ratio)
                + c2 * (d - d**3 / (3 * delta_lambda_rot**2))) / c3
    return jnp.where(at_edge, jnp.sign(detuning) * 0.5, interior)


def _apply_rotation_core(flux, wl_range: tuple, vsini, epsilon: float = 0.6):
    """
    Core rotation broadening implementation for a single wavelength range.

    Parameters
    ----------
    flux : array
        Flux vector for this range.
    wl_range : tuple
        (start_cm, stop_cm, n_points) wavelength range specification.
    vsini : float
        Projected rotational velocity in km/s.
    epsilon : float, optional
        Linear limb-darkening coefficient. Default: 0.6.

    Returns
    -------
    array
        Rotationally broadened flux.
    """
    vsini_static = float(_static_np(vsini, "vsini"))
    if vsini_static == 0:
        return jnp.asarray(flux)

    start_cm, stop_cm, n_points = wl_range
    wl = np.linspace(start_cm, stop_cm, n_points)
    step_cm = (stop_cm - start_cm) / (n_points - 1) if n_points > 1 else 0.0

    delta_rot = wl * (vsini * 1e5) / c_cgs          # traced, shape (n,)
    delta_rot_static = wl * (vsini_static * 1e5) / c_cgs

    # Static worst-case window from the concrete vsini.
    centers = np.arange(n_points)
    lb = np.searchsorted(wl, wl - delta_rot_static, side="left")
    ub = np.searchsorted(wl, wl + delta_rot_static, side="right") - 1
    half_width = int(max(
        np.max(centers - lb, initial=0), np.max(ub - centers, initial=0), 0
    )) + 1
    half_width = min(half_width, max(n_points - 1, 1))

    c1 = 2 * (1 - epsilon)
    c2 = np.pi * epsilon / 2
    c3_base = np.pi * (1 - epsilon / 3)

    lo_all, hi_all = wl - delta_rot, wl + delta_rot
    flux = jnp.asarray(flux)
    out = []
    for start, stop in _row_blocks(n_points, 2 * half_width + 1):
        offsets, indices, valid = _window_indices(
            centers[start:stop], half_width, n_points
        )
        lam_window = wl[indices]
        lo = lo_all[start:stop, None]
        hi = hi_all[start:stop, None]
        in_window = valid & (lam_window >= lo) & (lam_window <= hi)

        # The window is a contiguous run, so its first/last members are where
        # membership turns on and off. Those two cells are the ones whose outer
        # kernel boundary is ±Δλrot rather than a half-step (Korg.jl's
        # ``detunings = [-Δλrot; ...; Δλrot]``).
        pad = jnp.zeros((in_window.shape[0], 1), dtype=bool)
        is_first = in_window & ~jnp.concatenate([pad, in_window[:, :-1]], axis=1)
        is_last = in_window & ~jnp.concatenate([in_window[:, 1:], pad], axis=1)

        drot = delta_rot[start:stop, None]
        d_lo = jnp.where(is_first, -drot, (offsets - 0.5) * step_cm)
        d_hi = jnp.where(is_last, drot, (offsets + 0.5) * step_cm)

        c3 = c3_base * drot
        k_lo = _rotation_kernel_integral(c1, c2, c3, d_lo, drot)
        k_hi = _rotation_kernel_integral(c1, c2, c3, d_hi, drot)

        weights = jnp.where(in_window, k_hi - k_lo, 0.0)
        out.append(jnp.sum(weights * flux[indices], axis=1))

    return out[0] if len(out) == 1 else jnp.concatenate(out, axis=0)


def apply_rotation(flux, wls, vsini, epsilon: float = 0.6):
    """
    Apply rotational broadening to a spectrum.

    Given a spectrum `flux` sampled at wavelengths `wls` for a non-rotating
    star, compute the spectrum that would emerge given projected rotational
    velocity `vsini` and linear limb-darkening coefficient `epsilon`.

    The limb-darkening law used is: I(μ) = I(1) * (1 - ε + ε*μ)
    See Gray equation 18.14.

    Parameters
    ----------
    flux : array
        The flux vector to broaden.
    wls : tuple, list of tuples, array, or Wavelengths
        Wavelengths in any format accepted by Wavelengths class.
    vsini : float
        Projected rotational velocity in km/s.
    epsilon : float, optional
        Linear limb-darkening coefficient. Default: 0.6.

    Returns
    -------
    array
        Rotationally broadened flux vector.

    Notes
    -----
    ``vsini = 0`` returns the input unchanged. Otherwise the result is
    differentiable: ``jax.grad`` works with respect to `flux` and to `vsini`.

    Examples
    --------
    >>> flux_rotated = apply_rotation(flux, (5000, 5500, 0.01), vsini=10.0)
    """
    from .wavelengths import Wavelengths

    wls = Wavelengths(wls)
    flux = jnp.asarray(flux)

    pieces = [
        _apply_rotation_core(flux[lower:upper], wl_range, vsini, epsilon)
        for wl_range, (lower, upper) in zip(wls.wl_ranges, wls.subspectrum_indices())
    ]
    result = pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces, axis=0)
    return _output_like(result)
