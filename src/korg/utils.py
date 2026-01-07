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
