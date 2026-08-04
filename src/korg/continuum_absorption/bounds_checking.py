"""
Bounds checking utilities for continuum absorption functions.

Korg.jl wraps four continuum sources — ``Hminus_bf``, ``Hminus_ff``,
``H2plus_bf_and_ff`` and ``Heminus_ff`` — in ``bounds_checked_absorption``, which
returns *exactly zero* for frequencies or temperatures outside the range over
which the underlying table is valid (see ``ContinuumAbsorption/bounds_checking.jl``
in Korg v1.2.1).  This module provides the Python/JAX equivalent.

Two flavours are provided:

``bounds_checked_absorption``
    A direct transliteration of Julia's wrapper: a NumPy loop over the in-bounds
    frequencies, with an ``error_oobounds`` flag that raises instead of
    truncating.  Used for testing and for scalar-kernel use; it is *not*
    JIT-traceable because the output slice depends on the values.

``in_bounds_mask`` / ``clamp_to_bound``
    JAX-traceable primitives used by :mod:`korg.continuum`.  ``in_bounds_mask``
    gives the boolean mask and ``clamp_to_bound`` folds the out-of-bounds inputs
    onto a *strictly positive* in-bounds value so that the masked-away branch
    can never manufacture a NaN cotangent (``jnp.where`` masks a NaN value but
    not its gradient).

The interval algebra itself lives in :mod:`korg.utils`, which is a faithful port
of Korg's ``Interval``/``closed_interval``/``contained`` including the one-ULP
nudge that makes ``closed_interval`` inclusive.  This module re-exports those
names rather than carrying a second, subtly different copy.
"""

import numpy as np
import jax.numpy as jnp
from typing import Callable, Optional

from ..utils import (Interval, closed_interval, contained,  # noqa: F401 (re-exported)
                     _nextfloat_skipsubnorm, _prevfloat_skipsubnorm)

__all__ = [
    "Interval", "closed_interval", "contained", "contained_slice",
    "lambda_to_nu_bound", "bounds_checked_absorption",
    "in_bounds_mask", "clamp_to_bound",
    "HMINUS_BF_NU_BOUND", "HMINUS_BF_TEMP_BOUND",
    "HMINUS_FF_NU_BOUND", "HMINUS_FF_TEMP_BOUND",
    "H2PLUS_NU_BOUND", "H2PLUS_TEMP_BOUND",
    "HEMINUS_FF_NU_BOUND", "HEMINUS_FF_TEMP_BOUND",
]


def contained_slice(values, interval: Interval) -> slice:
    """
    Indices of ``values`` (sorted ascending) that are contained by ``interval``.

    This is Julia's ``contained_slice``:
    ``searchsortedfirst(vals, lower):searchsortedlast(vals, upper)``, expressed
    as a half-open Python ``slice``.  When nothing is contained the result is an
    empty slice.

    Parameters
    ----------
    values : array
        Values sorted in increasing order.
    interval : Interval
        Interval to check against.

    Returns
    -------
    slice
        ``slice(start, stop)`` such that ``values[start:stop]`` are in bounds.
    """
    values = np.asarray(values)
    if values.size == 0:
        return slice(0, 0)

    # searchsortedfirst == first index with value >= lower == np.searchsorted(..., 'left')
    start = int(np.searchsorted(values, interval.lower, side="left"))
    # searchsortedlast == last index with value <= upper; +1 for a half-open slice
    stop = int(np.searchsorted(values, interval.upper, side="right"))
    if stop < start:
        stop = start
    return slice(start, stop)


def _convert_lambda_endpoint(lambda_endpoint: float, lambda_lower_bound: bool) -> float:
    """
    Convert a wavelength endpoint (cm) to the frequency endpoint (Hz).

    Port of Korg's ``_convert_λ_endpoint``.  The floating-point nudging matters:
    ``Interval`` is half-open in Julia, so the frequency endpoint has to be moved
    to the first float for which the round-trip ``c/ν`` lands on the correct side
    of the wavelength endpoint.
    """
    from ..constants import c_cgs

    if lambda_lower_bound:
        def inbound(v):
            return np.nextafter(v, -np.inf)

        def oobound(v):
            return np.nextafter(v, np.inf)

        def relation(a, b):
            return a > b
    else:
        def inbound(v):
            return np.nextafter(v, np.inf)

        def oobound(v):
            return np.nextafter(v, -np.inf)

        def relation(a, b):
            return a < b

    nu_endpoint = np.inf if lambda_endpoint == 0 else c_cgs / lambda_endpoint

    if np.isfinite(nu_endpoint) and nu_endpoint != 0:
        # Case 1: the neighbouring ν that should be in-bounds maps to an
        # out-of-bounds λ.  Nudge toward the in-bounds side until it does not.
        while not relation(c_cgs / inbound(nu_endpoint), lambda_endpoint):
            nu_endpoint = inbound(nu_endpoint)
        # Case 2: ν_endpoint itself maps to an in-bounds λ.  Nudge outward.
        while relation(c_cgs / nu_endpoint, lambda_endpoint):
            nu_endpoint = oobound(nu_endpoint)

    return float(nu_endpoint)


def lambda_to_nu_bound(lambda_interval: Interval) -> Interval:
    """
    Convert a wavelength interval (cm) to the equivalent frequency interval (Hz).

    Port of Korg's ``λ_to_ν_bound``.  Frequency runs the other way, so the
    interval endpoints swap.

    Parameters
    ----------
    lambda_interval : Interval
        Wavelength interval in cm.

    Returns
    -------
    Interval
        Frequency interval in Hz.
    """
    return Interval(_convert_lambda_endpoint(lambda_interval.upper, False),
                    _convert_lambda_endpoint(lambda_interval.lower, True))


# ---------------------------------------------------------------------------
# Canonical bounds, transcribed from Korg v1.2.1
#   absorption_H.jl:234 (Hminus_bf), :326 (Hminus_ff), :378 (H2plus_bf_and_ff)
#   absorption_He.jl:93 (Heminus_ff)
# ---------------------------------------------------------------------------

#: ν ≤ 2.417989242625068e19 Hz; temperature unrestricted (but must be > 0).
HMINUS_BF_NU_BOUND = closed_interval(0.0, 2.417989242625068e19)
HMINUS_BF_TEMP_BOUND = Interval(0, np.inf)

#: Bell & Berrington (1987) table: 1823 Å – 151890 Å, θ = 5040/T ∈ [0.5, 3.6].
HMINUS_FF_NU_BOUND = lambda_to_nu_bound(closed_interval(1823e-8, 151890e-8))
HMINUS_FF_TEMP_BOUND = closed_interval(1400, 10080)

#: Stancil (1994) tables: 700 Å – 200000 Å, 3150 K – 25200 K.
H2PLUS_NU_BOUND = lambda_to_nu_bound(closed_interval(7e-6, 2e-3))
H2PLUS_TEMP_BOUND = closed_interval(3150, 25200)

#: John (1994) table: 5063 Å – 151878 Å, θ = 5040/T ∈ [0.5, 3.6].
HEMINUS_FF_NU_BOUND = lambda_to_nu_bound(closed_interval(5.063e-5, 1.518780e-03))
HEMINUS_FF_TEMP_BOUND = closed_interval(1400, 10080)


# ---------------------------------------------------------------------------
# JAX-traceable helpers
# ---------------------------------------------------------------------------

def in_bounds_mask(nu, T, nu_bound: Interval = None, temp_bound: Interval = None):
    """
    Boolean mask reproducing ``bounds_checked_absorption``'s in-bounds selection.

    Unlike :func:`contained` this uses ``&`` rather than a chained comparison, so
    it works on arrays and on JAX tracers.

    .. note::
       The frequency and temperature comparisons are deliberately *not*
       symmetric, because Korg's wrapper is not either.  It selects frequencies
       with ``contained_slice``, i.e.
       ``searchsortedfirst(ν, lower):searchsortedlast(ν, upper)``, which is
       **inclusive** of both endpoints; but it tests temperature with
       ``contained``, which is **exclusive**.  (For a ``closed_interval`` the
       temperature endpoints are still admitted, because ``closed_interval``
       nudges them outward by one ULP.)  Reproducing this matters: for the
       Bell & Berrington table, ν = c/(151890·1e-8) lands exactly on
       ``nu_bound.lower``, so an exclusive test would wrongly zero it.

    Parameters
    ----------
    nu : float or array
        Frequency in Hz.
    T : float or array
        Temperature in K.
    nu_bound, temp_bound : Interval, optional
        Bounds to apply.  ``None`` means "unrestricted in that variable".

    Returns
    -------
    array of bool
        True where the source should contribute.
    """
    mask = jnp.asarray(True)
    if nu_bound is not None:
        nu = jnp.asarray(nu)
        # inclusive: mirrors contained_slice / searchsortedfirst..searchsortedlast
        mask = mask & (nu >= nu_bound.lower) & (nu <= nu_bound.upper)
    if temp_bound is not None:
        T = jnp.asarray(T)
        # exclusive: mirrors contained()
        mask = mask & (T > temp_bound.lower) & (T < temp_bound.upper)
    return mask


def clamp_to_bound(x, interval: Interval, floor: float):
    """
    Fold ``x`` onto a strictly positive value inside ``interval``.

    In-bounds inputs are returned unchanged, so this never perturbs a value that
    actually matters.  Out-of-bounds inputs are replaced by a finite, strictly
    positive stand-in whose only job is to keep the arithmetic (and therefore the
    reverse-mode cotangent) finite before :func:`in_bounds_mask` discards it.
    Clamping to *zero* would not do: ``sqrt(0)`` and ``1/0`` both have infinite
    derivatives, and ``0 * inf`` is NaN.

    Parameters
    ----------
    x : float or array
        Value to clamp.
    interval : Interval
        Interval to clamp into.
    floor : float
        Strictly positive stand-in used when ``interval`` reaches down to 0 or
        below.  Must be > 0.  An unbounded *upper* end is left unbounded — there
        is nothing unsafe about a large input, and substituting a finite
        sentinel would silently alter in-bounds values.

    Returns
    -------
    array
        ``x`` clamped into ``[lo, hi]`` with ``lo > 0``.
    """
    if not floor > 0:
        raise ValueError(f"floor must be strictly positive, got {floor!r}")
    lo = interval.lower if interval.lower > 0 else floor
    hi = interval.upper if np.isfinite(interval.upper) else np.inf
    if hi < lo:
        raise ValueError(f"floor {floor!r} exceeds the interval upper bound {hi!r}")
    return jnp.clip(jnp.asarray(x), lo, hi)


# ---------------------------------------------------------------------------
# NumPy wrapper mirroring Julia's bounds_checked_absorption
# ---------------------------------------------------------------------------

def bounds_checked_absorption(
    func: Callable,
    nu_bound: Interval = None,
    temp_bound: Interval = None,
) -> Callable:
    """
    Wrap a scalar absorption kernel with Julia's bounds-checking behaviour.

    Parameters
    ----------
    func : callable
        ``func(nu: float, T: float, *args) -> float`` with ``nu`` in Hz and
        ``T`` in K.
    nu_bound : Interval, optional
        Frequency range (Hz) over which ``func`` is valid.  Defaults to (0, ∞).
    temp_bound : Interval, optional
        Temperature range (K) over which ``func`` is valid.  Defaults to (0, ∞).

    Returns
    -------
    callable
        ``wrapped(nus, T, *args, error_oobounds=False, out_alpha=None)``.

    Notes
    -----
    This is deliberately *not* JIT-traceable: like Julia's version, it computes
    ``func`` only on the in-bounds slice, whose extent depends on the values of
    ``nus``.  :mod:`korg.continuum` uses :func:`in_bounds_mask` instead.
    """
    if nu_bound is None:
        nu_bound = Interval(0, np.inf)
    if temp_bound is None:
        temp_bound = Interval(0, np.inf)

    def wrapped_func(
        nus: np.ndarray,
        T: float,
        *args,
        error_oobounds: bool = False,
        out_alpha: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Absorption coefficients, zero (or an error) outside the valid range.

        Parameters
        ----------
        nus : array
            Sorted frequencies in Hz.
        T : float
            Temperature in K.
        *args
            Extra arguments forwarded to the wrapped kernel.
        error_oobounds : bool, optional
            If True, raise ``ValueError`` when any input is out of bounds.
            If False (default), those entries are left at zero.
        out_alpha : array, optional
            Array to accumulate into, in place.

        Returns
        -------
        array
            Absorption coefficients in cm⁻¹.
        """
        nus = np.asarray(nus)
        if nus.ndim == 0:
            raise ValueError("nus must be a 1-D array of frequencies, not a scalar")

        if len(nus) > 1:
            if nus[0] <= nus[-1]:
                assert np.all(np.diff(nus) >= 0), "nus must be sorted"
            else:
                raise ValueError(
                    "nus must be sorted in increasing order; Korg's contained_slice "
                    "assumes an ascending frequency grid"
                )

        alpha_type = np.result_type(nus.dtype, np.float64)

        if out_alpha is None:
            out_alpha = np.zeros(len(nus), dtype=alpha_type)
        else:
            assert len(out_alpha) == len(nus), "out_alpha must have same length as nus"
            assert out_alpha.dtype == alpha_type, (
                f"out_alpha dtype mismatch: {out_alpha.dtype} vs {alpha_type}")

        # Indices that can be updated: empty if T itself is out of bounds.
        T_ok = contained(T, temp_bound)
        idx = contained_slice(nus, nu_bound) if T_ok else slice(0, 0)

        if (idx.start != 0 or idx.stop != len(nus)) and error_oobounds:
            if not T_ok:
                raise ValueError(
                    f"{func.__name__}: invalid temperature. "
                    f"T={T} should lie between {temp_bound.lower} and {temp_bound.upper}"
                )
            # Julia reports ν[last(idx)+1] when the first element is in bounds,
            # otherwise the very first (out-of-bounds) frequency.
            bad_nu = nus[idx.stop] if idx.start == 0 else nus[0]
            raise ValueError(
                f"{func.__name__}: invalid frequency. "
                f"nu={bad_nu} Hz should lie between {nu_bound.lower} and "
                f"{nu_bound.upper} Hz"
            )

        for i in range(idx.start, idx.stop):
            out_alpha[i] += func(nus[i], T, *args)

        return out_alpha

    wrapped_func.nu_bound = nu_bound
    wrapped_func.temp_bound = temp_bound
    return wrapped_func
