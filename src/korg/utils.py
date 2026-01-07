"""
Utility functions for Korg.

Simple mathematical and physics utility functions including:
- LSF (Line Spread Function) convolution
- Rotational broadening
- Normal PDF
- Translational partition function
"""

import jax.numpy as jnp
import numpy as np
from typing import Union, Callable
from bisect import bisect_left, bisect_right

from .constants import kboltz_cgs, hplanck_cgs, c_cgs
from .wavelengths import Wavelengths


def air_to_vacuum(wavelength_air: float) -> float:
    """
    Convert wavelength from air to vacuum.

    Formula from Birch and Downs (1994) via the VALD website.
    Valid for wavelengths > 2000 Å.

    Parameters
    ----------
    wavelength_air : float
        Wavelength in air, in Angstroms.

    Returns
    -------
    float
        Wavelength in vacuum, in Angstroms.

    See Also
    --------
    vacuum_to_air : The inverse conversion.
    """
    s = 1e4 / wavelength_air
    n = (1 + 0.00008336624212083
         + 0.02408926869968 / (130.1065924522 - s**2)
         + 0.0001599740894897 / (38.92568793293 - s**2))
    return wavelength_air * n


def vacuum_to_air(wavelength_vacuum: float) -> float:
    """
    Convert wavelength from vacuum to air.

    Formula from Birch and Downs (1994) via the VALD website.
    Valid for wavelengths > 2000 Å.

    Parameters
    ----------
    wavelength_vacuum : float
        Wavelength in vacuum, in Angstroms.

    Returns
    -------
    float
        Wavelength in air, in Angstroms.

    See Also
    --------
    air_to_vacuum : The inverse conversion.
    """
    s = 1e4 / wavelength_vacuum
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
    return wavelength_vacuum / n


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


def _resolve_R(R: Union[float, Callable], lambda0: float) -> float:
    """
    Resolve R to a value based on its type.

    Parameters
    ----------
    R : float or callable
        Resolving power. If callable, it's called with λ in Å.
    lambda0 : float
        Wavelength in cm.

    Returns
    -------
    float
        Resolved R value.
    """
    if callable(R):
        return R(lambda0 * 1e8)  # R is a function of λ in Å
    return R


def _lsf_bounds_and_kernel(synth_wls: Wavelengths, lambda0: float,
                           R: Union[float, Callable], window_size: float):
    """
    Core LSF calculation shared by all variants.

    Parameters
    ----------
    synth_wls : Wavelengths
        Synthesis wavelengths.
    lambda0 : float
        Central wavelength in cm.
    R : float or callable
        Resolving power.
    window_size : float
        How far to extend the kernel in units of sigma.

    Returns
    -------
    tuple
        (lower_bound_index, upper_bound_index, normalized_kernel)
    """
    R_val = _resolve_R(R, lambda0)
    # Convert Δλ = λ0/R (FWHM) to sigma
    sigma = lambda0 / R_val / (2 * np.sqrt(2 * np.log(2)))

    # Calculate bounds and kernel
    lb = synth_wls.searchsortedfirst(lambda0 - window_size * sigma)
    ub = synth_wls.searchsortedlast(lambda0 + window_size * sigma)

    # Compute kernel using numpy for efficiency
    wl_slice = synth_wls.all_wls[lb:ub + 1]  # +1 because Python slicing is exclusive
    phi = np.exp(-0.5 * (wl_slice - lambda0)**2 / sigma**2) / np.sqrt(2 * np.pi) / sigma
    normalized_phi = phi / np.sum(phi)

    return lb, ub, normalized_phi


def apply_LSF(flux: np.ndarray, wls, R: Union[float, Callable],
              window_size: float = 4) -> np.ndarray:
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
    - This is a naive, slow implementation. For multiple spectra on the
      same wavelength grid, use compute_LSF_matrix instead.
    - apply_LSF will have weird behavior if your wavelength grid is not
      locally linearly-spaced. Run on a fine grid, then downsample.
    - For best results, extend your wavelength range a couple Δλ outside
      the region you will compare to data.

    Examples
    --------
    >>> flux_convolved = apply_LSF(flux, (5000, 5500, 0.01), R=50000)
    """
    if R == np.inf:
        return flux.copy()

    wls = Wavelengths(wls)
    conv_flux = np.zeros(len(flux), dtype=flux.dtype)

    for i in range(len(wls)):
        lambda0 = wls[i]
        lb, ub, normalized_phi = _lsf_bounds_and_kernel(wls, lambda0, R, window_size)
        conv_flux[i] = np.sum(flux[lb:ub + 1] * normalized_phi)

    return conv_flux


def _rotation_kernel_integral(c1: float, c2: float, c3: float,
                               detuning: float, delta_lambda_rot: float) -> float:
    """
    Indefinite integral of the rotation kernel.

    Parameters
    ----------
    c1, c2, c3 : float
        Precomputed constants.
    detuning : float
        Wavelength detuning.
    delta_lambda_rot : float
        Rotational broadening width.

    Returns
    -------
    float
        Integral value.
    """
    if abs(detuning) == delta_lambda_rot:
        return np.sign(detuning) * 0.5  # nan-safe for autodiff
    ratio = detuning / delta_lambda_rot
    return (0.5 * c1 * detuning * np.sqrt(1 - ratio**2)
            + 0.5 * c1 * delta_lambda_rot * np.arcsin(ratio)
            + c2 * (detuning - detuning**3 / (3 * delta_lambda_rot**2))) / c3


def _apply_rotation_core(flux: np.ndarray, wl_range: tuple,
                         vsini: float, epsilon: float = 0.6) -> np.ndarray:
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
    if vsini == 0:
        return flux.copy()

    start_cm, stop_cm, n_points = wl_range
    step_cm = (stop_cm - start_cm) / (n_points - 1) if n_points > 1 else 0

    # Create wavelength array
    wls = np.linspace(start_cm, stop_cm, n_points)

    # Convert vsini to cm/s
    vsini_cgs = vsini * 1e5

    new_flux = np.zeros(len(flux), dtype=np.float64)

    # Precompute constants
    c1 = 2 * (1 - epsilon)
    c2 = np.pi * epsilon / 2
    c3_base = np.pi * (1 - epsilon / 3)

    for i in range(len(flux)):
        delta_lambda_rot = wls[i] * vsini_cgs / c_cgs

        # Find bounds
        lb = bisect_left(wls, wls[i] - delta_lambda_rot)
        ub_search = bisect_right(wls, wls[i] + delta_lambda_rot)
        ub = ub_search - 1 if ub_search > 0 else 0

        flux_window = flux[lb:ub + 1]

        # Build detuning array: [-Δλrot; half-steps; Δλrot]
        # This matches Julia: detunings = [-Δλrot; (lb-i+1/2:ub-i-1/2) * step(wls); Δλrot]
        half_steps = np.arange(lb - i + 0.5, ub - i + 0.5) * step_cm
        detunings = np.concatenate([[-delta_lambda_rot], half_steps, [delta_lambda_rot]])

        c3 = c3_base * delta_lambda_rot

        # Compute kernel integrals
        ks = np.array([_rotation_kernel_integral(c1, c2, c3, d, delta_lambda_rot)
                       for d in detunings])

        # Apply kernel (difference of integrals)
        new_flux[i] = np.sum(ks[1:] * flux_window) - np.sum(ks[:-1] * flux_window)

    return new_flux


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
    The returned matrix is sparse in Julia but dense in Python/JAX for JIT compatibility.
    For best results, synthesis wavelengths should extend a couple Δλ outside
    the observation range.

    Examples
    --------
    >>> lsf_matrix = compute_LSF_matrix((5000, 5100, 0.01), obs_wls, R=50000)
    >>> convolved_flux = lsf_matrix @ flux
    """
    # Convert obs_wls to cm if in Angstroms
    obs_wls = np.asarray(obs_wls)
    if obs_wls[0] >= 1:
        obs_wls = obs_wls / 1e8  # Å to cm

    synth_wls = Wavelengths(synth_wls)

    # Warn if synthesis wavelengths don't cover observation range
    if verbose:
        synth_first = synth_wls.all_wls[0]
        synth_last = synth_wls.all_wls[-1]
        obs_first = obs_wls[0]
        obs_last = obs_wls[-1]
        margin = 0.01  # cm (~1000 Å)

        if not ((synth_first - margin) <= obs_first <= obs_last <= (synth_last + margin)):
            import warnings
            warnings.warn(
                f"Synthesis wavelengths ({synth_first*1e8:.1f} Å—{synth_last*1e8:.1f} Å) "
                f"are not superset of observation wavelengths "
                f"({obs_first*1e8:.1f} Å—{obs_last*1e8:.1f} Å) in LSF matrix."
            )

    # Build the LSF matrix
    n_synth = len(synth_wls)
    n_obs = len(obs_wls)
    LSF = np.zeros((n_synth, n_obs), dtype=np.float64)

    for i in range(n_obs):
        lambda0 = obs_wls[i]
        lb, ub, normalized_phi = _lsf_bounds_and_kernel(synth_wls, lambda0, R, window_size)
        LSF[lb:ub + 1, i] += normalized_phi

    # Transpose to get (n_obs, n_synth) for left multiplication
    return LSF.T


def apply_rotation(flux: np.ndarray, wls, vsini: float,
                   epsilon: float = 0.6) -> np.ndarray:
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

    Examples
    --------
    >>> flux_rotated = apply_rotation(flux, (5000, 5500, 0.01), vsini=10.0)
    """
    wls = Wavelengths(wls)
    new_flux = np.zeros(len(flux), dtype=np.float64)

    lower_index = 0
    upper_index = wls.wl_ranges[0][2]  # n_points for first range

    new_flux[lower_index:upper_index] = _apply_rotation_core(
        flux[lower_index:upper_index],
        wls.wl_ranges[0],
        vsini, epsilon
    )

    for i in range(1, len(wls.wl_ranges)):
        lower_index = upper_index
        upper_index = lower_index + wls.wl_ranges[i][2]  # n_points for this range
        new_flux[lower_index:upper_index] = _apply_rotation_core(
            flux[lower_index:upper_index],
            wls.wl_ranges[i],
            vsini, epsilon
        )

    return new_flux
