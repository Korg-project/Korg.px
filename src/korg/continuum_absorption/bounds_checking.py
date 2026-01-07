"""
Bounds checking utilities for continuum absorption functions.

Provides interval checking and wrapper functions to validate frequency
and temperature bounds for opacity sources.
"""

import numpy as np
import jax.numpy as jnp
from typing import Union, Callable, Optional, Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    """
    Represents an interval [lower, upper].
    
    Parameters
    ----------
    lower : float
        Lower bound (inclusive)
    upper : float
        Upper bound (inclusive)
    """
    lower: float
    upper: float
    
    def __repr__(self):
        return f"Interval({self.lower}, {self.upper})"


def closed_interval(lower: float, upper: float) -> Interval:
    """
    Create a closed interval [lower, upper].
    
    Parameters
    ----------
    lower : float
        Lower bound
    upper : float
        Upper bound
        
    Returns
    -------
    Interval
        Closed interval
    """
    return Interval(lower, upper)


def contained(value: Union[float, np.ndarray], interval: Interval) -> Union[bool, np.ndarray]:
    """
    Check if value(s) are contained in the interval.
    
    Parameters
    ----------
    value : float or array
        Value(s) to check
    interval : Interval
        Interval to check against
        
    Returns
    -------
    bool or array of bool
        True if value is in [interval.lower, interval.upper]
    """
    return (value >= interval.lower) & (value <= interval.upper)


def contained_slice(values: np.ndarray, interval: Interval) -> slice:
    """
    Find the slice of values that are contained in the interval.
    
    Assumes values are sorted (either ascending or descending).
    
    Parameters
    ----------
    values : array
        Sorted array of values
    interval : Interval
        Interval to check against
        
    Returns
    -------
    slice
        Slice object representing the range of indices where values are in interval
    """
    # Check if values are sorted ascending or descending
    if len(values) == 0:
        return slice(0, 0)
    
    if len(values) == 1:
        return slice(0, 1) if contained(values[0], interval) else slice(0, 0)
    
    ascending = values[0] < values[-1]
    
    if ascending:
        # Find first index >= lower bound
        start = np.searchsorted(values, interval.lower, side='left')
        # Find first index > upper bound
        end = np.searchsorted(values, interval.upper, side='right')
    else:
        # Values are descending
        # Find first index <= upper bound (from the left, which has higher values)
        start = np.searchsorted(-values, -interval.upper, side='left')
        # Find first index < lower bound
        end = np.searchsorted(-values, -interval.lower, side='right')
    
    return slice(int(start), int(end))


def lambda_to_nu_bound(lambda_interval: Interval) -> Interval:
    """
    Convert a wavelength interval to a frequency interval.
    
    Parameters
    ----------
    lambda_interval : Interval
        Wavelength interval in cm
        
    Returns
    -------
    Interval
        Frequency interval in Hz (note: bounds are reversed)
    """
    from ..constants import c_cgs
    
    # Frequency is inversely related to wavelength
    # λ_min corresponds to ν_max and vice versa
    nu_max = c_cgs / lambda_interval.lower
    nu_min = c_cgs / lambda_interval.upper
    
    return Interval(nu_min, nu_max)


def bounds_checked_absorption(
    func: Callable,
    nu_bound: Interval = Interval(0, np.inf),
    temp_bound: Interval = Interval(0, np.inf)
) -> Callable:
    """
    Create a wrapped absorption function with bounds checking.
    
    Parameters
    ----------
    func : callable
        Function with signature func(nu: float, T: float, *args) -> float
        where nu is frequency in Hz and T is temperature in K
    nu_bound : Interval
        Valid frequency range in Hz
    temp_bound : Interval
        Valid temperature range in K
        
    Returns
    -------
    callable
        Wrapped function with signature:
        wrapped_func(nus: array, T: float, *args, 
                     error_oobounds: bool = False,
                     out_alpha: Optional[array] = None) -> array
    """
    
    def wrapped_func(
        nus: np.ndarray,
        T: float,
        *args,
        error_oobounds: bool = False,
        out_alpha: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Wrapped absorption function with bounds checking.
        
        Parameters
        ----------
        nus : array
            Sorted frequencies in Hz
        T : float
            Temperature in K
        *args
            Additional arguments for func
        error_oobounds : bool, optional
            If True, raise error for out-of-bounds values.
            If False, return 0 for out-of-bounds (default).
        out_alpha : array, optional
            Output array to add results to (in-place operation)
            
        Returns
        -------
        array
            Absorption coefficients in cm^-1
        """
        # Verify nus is sorted
        if len(nus) > 1:
            ascending = nus[0] < nus[-1]
            if ascending:
                assert np.all(np.diff(nus) >= 0), "nus must be sorted"
            else:
                assert np.all(np.diff(nus) <= 0), "nus must be sorted"
        
        # Determine output type
        alpha_type = np.result_type(nus.dtype, np.float64)
        
        # Initialize output array
        if out_alpha is None:
            out_alpha = np.zeros(len(nus), dtype=alpha_type)
        else:
            assert len(out_alpha) == len(nus), "out_alpha must have same length as nus"
            assert out_alpha.dtype == alpha_type, f"out_alpha dtype mismatch: {out_alpha.dtype} vs {alpha_type}"
        
        # Find indices where we can compute (in bounds)
        if not contained(T, temp_bound):
            idx = slice(0, 0)  # Empty slice - T out of bounds
        else:
            idx = contained_slice(nus, nu_bound)
        
        # Handle out-of-bounds
        if (idx.start != 0 or idx.stop != len(nus)) and error_oobounds:
            if not contained(T, temp_bound):
                raise ValueError(
                    f"{func.__name__}: invalid temperature. "
                    f"T={T} should be in [{temp_bound.lower}, {temp_bound.upper}]"
                )
            else:
                # Find the bad frequency
                if idx.start > 0:
                    bad_nu = nus[idx.start - 1]
                else:
                    bad_nu = nus[idx.stop]
                raise ValueError(
                    f"{func.__name__}: invalid frequency. "
                    f"nu={bad_nu} Hz should be in [{nu_bound.lower}, {nu_bound.upper}] Hz"
                )
        
        # Compute absorption for in-bounds values
        if idx.start < idx.stop:
            # Call the underlying function for each frequency
            for i in range(idx.start, idx.stop):
                out_alpha[i] += func(nus[i], T, *args)
        
        return out_alpha
    
    return wrapped_func
