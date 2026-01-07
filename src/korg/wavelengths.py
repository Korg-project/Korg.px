"""
Wavelength handling utilities.

Provides:
- Wavelengths class for managing wavelength grids
- Conversion between air and vacuum wavelengths using the
  Birch and Downs (1994) formula via the VALD website.

Reference: Korg.jl wavelengths.jl
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Iterator
from bisect import bisect_left, bisect_right

from .constants import c_cgs


def air_to_vacuum(wavelength, cgs=None):
    """
    Convert wavelength from air to vacuum.

    Parameters
    ----------
    wavelength : float or array
        Wavelength value(s). Assumed to be in Å if >= 1, in cm otherwise.
    cgs : bool, optional
        If True, treat wavelength as in cm. If False, treat as Å.
        If None (default), auto-detect based on wavelength >= 1.

    Returns
    -------
    float or array
        Wavelength in vacuum (same units as input).

    Notes
    -----
    Formula from Birch and Downs (1994) via the VALD website.
    """
    wavelength = np.asarray(wavelength)
    scalar_input = wavelength.ndim == 0
    wavelength = np.atleast_1d(wavelength)

    # Auto-detect units if not specified
    if cgs is None:
        cgs = np.all(wavelength < 1)

    lam = wavelength.copy()
    if cgs:
        lam = lam * 1e8  # cm to Å

    # Calculate refractive index
    s = 1e4 / lam
    n = (1 + 0.00008336624212083 +
         0.02408926869968 / (130.1065924522 - s**2) +
         0.0001599740894897 / (38.92568793293 - s**2))

    # Convert back to original units if needed
    result = lam * n
    if cgs:
        result = result * 1e-8  # Å to cm

    if scalar_input:
        return float(result[0])
    return result


def vacuum_to_air(wavelength, cgs=None):
    """
    Convert wavelength from vacuum to air.

    Parameters
    ----------
    wavelength : float or array
        Wavelength value(s). Assumed to be in Å if >= 1, in cm otherwise.
    cgs : bool, optional
        If True, treat wavelength as in cm. If False, treat as Å.
        If None (default), auto-detect based on wavelength >= 1.

    Returns
    -------
    float or array
        Wavelength in air (same units as input).

    Notes
    -----
    Formula from Birch and Downs (1994) via the VALD website.
    """
    wavelength = np.asarray(wavelength)
    scalar_input = wavelength.ndim == 0
    wavelength = np.atleast_1d(wavelength)

    # Auto-detect units if not specified
    if cgs is None:
        cgs = np.all(wavelength < 1)

    lam = wavelength.copy()
    if cgs:
        lam = lam * 1e8  # cm to Å

    # Calculate refractive index
    s = 1e4 / lam
    n = (1 + 0.0000834254 +
         0.02406147 / (130 - s**2) +
         0.00015998 / (38.9 - s**2))

    # Convert back to original units if needed
    result = lam / n
    if cgs:
        result = result * 1e-8  # Å to cm

    if scalar_input:
        return float(result[0])
    return result


class Wavelengths:
    """
    Represents wavelengths for spectral synthesis.

    This class manages (possibly non-contiguous) wavelength ranges, providing
    precomputed arrays of wavelengths and frequencies for efficient synthesis.

    Wavelengths can be specified in either Å or cm. Values >= 1 are assumed
    to be in Å and values < 1 are assumed to be in cm.

    Parameters
    ----------
    wl_spec : tuple, list of tuples, array, or Wavelengths
        Wavelength specification. Can be:
        - tuple: (start, stop) or (start, stop, step) in Å or cm
        - list of tuples: multiple wavelength ranges
        - array: explicit wavelength values (must be linearly spaced)
        - Wavelengths: copy constructor
    air_wavelengths : bool, optional
        If True, input wavelengths are air wavelengths to be converted to
        vacuum. Default: False
    wavelength_conversion_warn_threshold : float, optional
        Maximum allowed error in air-to-vacuum conversion (in Å).
        Default: 1e-4

    Attributes
    ----------
    wl_ranges : list of tuples
        List of (start, stop, n_points) for each wavelength range (in cm)
    all_wls : ndarray
        All wavelengths concatenated (in cm)
    all_freqs : ndarray
        All frequencies (in Hz), sorted in increasing order

    Examples
    --------
    >>> # Single wavelength range
    >>> wls = Wavelengths((5000, 5500))  # 5000-5500 Å, 0.01 Å step
    >>> len(wls)
    50001

    >>> # With explicit step
    >>> wls = Wavelengths((5000, 5500, 0.1))  # 0.1 Å step

    >>> # Multiple ranges
    >>> wls = Wavelengths([(5000, 5100), (6500, 6600)])

    >>> # From explicit wavelength array
    >>> wls = Wavelengths(np.linspace(5000, 5500, 1000))

    Notes
    -----
    Internally, all wavelengths are stored in cm. The `all_wls` attribute
    provides access to wavelengths in cm, while indexing with `[]` also
    returns cm values.

    This class matches Korg.jl's Wavelengths struct behavior.
    """

    def __init__(
        self,
        wl_spec: Union[Tuple, List[Tuple], np.ndarray, 'Wavelengths'],
        air_wavelengths: bool = False,
        wavelength_conversion_warn_threshold: float = 1e-4
    ):
        if isinstance(wl_spec, Wavelengths):
            # Copy constructor
            if air_wavelengths:
                # Re-process with air conversion
                self._init_from_ranges(
                    wl_spec._raw_ranges,
                    air_wavelengths=True,
                    wavelength_conversion_warn_threshold=wavelength_conversion_warn_threshold
                )
            else:
                self.wl_ranges = wl_spec.wl_ranges.copy()
                self.all_wls = wl_spec.all_wls.copy()
                self.all_freqs = wl_spec.all_freqs.copy()
                self._raw_ranges = wl_spec._raw_ranges.copy()
            return

        # Convert various input formats to list of (start, stop, step) tuples
        if isinstance(wl_spec, tuple):
            # Single tuple: (start, stop) or (start, stop, step)
            wl_spec = [wl_spec]

        if isinstance(wl_spec, np.ndarray) or (isinstance(wl_spec, list) and
                                                len(wl_spec) > 0 and
                                                not isinstance(wl_spec[0], tuple)):
            # Array of wavelength values - convert to range specification
            wl_spec = np.asarray(wl_spec)
            if len(wl_spec) == 0:
                raise ValueError("wavelengths must be non-empty")
            elif len(wl_spec) == 1:
                # Single wavelength - create a trivial range
                wl_spec = [(wl_spec[0], wl_spec[0], 1.0)]
            else:
                # Check if linearly spaced
                diffs = np.diff(wl_spec)
                min_step, max_step = diffs.min(), diffs.max()
                tolerance = 1e-6
                if max_step - min_step > tolerance:
                    raise ValueError(f"wavelengths are not linearly spaced to within {tolerance}.")
                # Create range specification
                step = (wl_spec[-1] - wl_spec[0]) / (len(wl_spec) - 1)
                wl_spec = [(wl_spec[0], wl_spec[-1], step)]

        # Now wl_spec should be a list of tuples
        self._init_from_ranges(
            wl_spec,
            air_wavelengths=air_wavelengths,
            wavelength_conversion_warn_threshold=wavelength_conversion_warn_threshold
        )

    def _init_from_ranges(
        self,
        tuples: List[Tuple],
        air_wavelengths: bool,
        wavelength_conversion_warn_threshold: float
    ):
        """Initialize from list of (start, stop) or (start, stop, step) tuples."""
        self._raw_ranges = tuples.copy()

        ranges_data = []  # Will store (start_cm, stop_cm, n_points)
        all_wls_list = []

        for tup in tuples:
            if len(tup) == 2:
                start, stop = tup
                # Default step: 0.01 Å if in Å, 1e-10 cm if in cm
                if start >= 1:
                    step = 0.01  # Å
                else:
                    step = 1e-10  # cm (0.01 Å)
            elif len(tup) == 3:
                start, stop, step = tup
            else:
                raise ValueError(f"Each wavelength range must be specified as "
                               f"(start, stop) or (start, stop, step). Got {tup}")

            # Convert to cm if in Å
            if start >= 1:
                start_cm = start * 1e-8
                stop_cm = stop * 1e-8
                step_cm = step * 1e-8
            else:
                start_cm = start
                stop_cm = stop
                step_cm = step

            # Create wavelength array for this range
            n_points = int(round((stop_cm - start_cm) / step_cm)) + 1
            wls = np.linspace(start_cm, stop_cm, n_points)

            # Handle air-to-vacuum conversion
            if air_wavelengths:
                vac_start = air_to_vacuum(start_cm, cgs=True)
                vac_stop = air_to_vacuum(stop_cm, cgs=True)
                vac_wls_linear = np.linspace(vac_start, vac_stop, n_points)

                # Check conversion accuracy
                vac_wls_exact = air_to_vacuum(wls, cgs=True)
                max_diff = np.max(np.abs(vac_wls_linear - vac_wls_exact)) * 1e8  # Convert to Å

                if max_diff > wavelength_conversion_warn_threshold:
                    raise ValueError(
                        f"A linear air wavelength range can't be approximated exactly "
                        f"with a linear vacuum wavelength range. This solution differs "
                        f"by up to {max_diff:.2e} Å. Adjust wavelength_conversion_warn_threshold "
                        f"if you want to suppress this error."
                    )

                wls = vac_wls_linear
                start_cm = vac_start
                stop_cm = vac_stop

            ranges_data.append((start_cm, stop_cm, n_points))
            all_wls_list.append(wls)

        # Concatenate all wavelengths
        self.all_wls = np.concatenate(all_wls_list)

        # Check that wavelengths are sorted
        if not np.all(np.diff(self.all_wls) > 0):
            raise ValueError("wl_ranges must be sorted and non-overlapping")

        # Store range metadata
        self.wl_ranges = ranges_data

        # Precompute frequencies (sorted in increasing order, i.e. reversed from wavelengths)
        self.all_freqs = c_cgs / self.all_wls[::-1]

    def __len__(self) -> int:
        """Return total number of wavelength points."""
        return len(self.all_wls)

    def __getitem__(self, idx) -> Union[float, np.ndarray]:
        """Get wavelength(s) by index (returns values in cm)."""
        return self.all_wls[idx]

    def __iter__(self) -> Iterator[float]:
        """Iterate over wavelengths (in cm)."""
        return iter(self.all_wls)

    def __repr__(self) -> str:
        ranges_str = ", ".join(
            f"{int(round(start*1e8))}-{int(round(stop*1e8))} Å"
            for start, stop, _ in self.wl_ranges
        )
        return f"Wavelengths({ranges_str})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Wavelengths):
            return False
        return self.wl_ranges == other.wl_ranges

    def eachwindow(self) -> Iterator[Tuple[float, float]]:
        """
        Iterate over wavelength windows.

        Yields
        ------
        (lambda_low, lambda_high) : tuple
            Start and end wavelength of each range (in cm).
        """
        for start, stop, _ in self.wl_ranges:
            yield (start, stop)

    def eachfreq(self) -> np.ndarray:
        """
        Get frequencies corresponding to wavelengths.

        Returns
        -------
        freqs : ndarray
            Frequencies in Hz, sorted in increasing order
            (reverse order of wavelengths).
        """
        return self.all_freqs

    def subspectrum_indices(self) -> List[Tuple[int, int]]:
        """
        Get index ranges for each wavelength window.

        Returns
        -------
        indices : list of tuples
            List of (start_idx, end_idx) for each wavelength range.
            Can be used to slice the full spectrum.

        Examples
        --------
        >>> wls = Wavelengths([(5000, 5100), (6500, 6600)])
        >>> indices = wls.subspectrum_indices()
        >>> # Get flux for first wavelength range
        >>> flux_range1 = flux[indices[0][0]:indices[0][1]]
        """
        indices = []
        start_idx = 0
        for _, _, n_points in self.wl_ranges:
            indices.append((start_idx, start_idx + n_points))
            start_idx += n_points
        return indices

    def searchsortedfirst(self, lam: float) -> int:
        """
        Find index of first element >= lam.

        Parameters
        ----------
        lam : float
            Wavelength to search for. If >= 1, assumed to be in Å.

        Returns
        -------
        idx : int
            Index of first element >= lam, or len(self) if none found.
        """
        # Convert Å to cm if needed
        if lam >= 1:
            lam = lam * 1e-8

        return bisect_left(self.all_wls, lam)

    def searchsortedlast(self, lam: float) -> int:
        """
        Find index of last element <= lam.

        Parameters
        ----------
        lam : float
            Wavelength to search for. If >= 1, assumed to be in Å.

        Returns
        -------
        idx : int
            Index of last element <= lam, or -1 if none found.
        """
        # Convert Å to cm if needed
        if lam >= 1:
            lam = lam * 1e-8

        idx = bisect_right(self.all_wls, lam)
        return idx - 1

    @property
    def wavelengths_angstrom(self) -> np.ndarray:
        """Get all wavelengths in Ångströms."""
        return self.all_wls * 1e8

    @property
    def wavelengths_cm(self) -> np.ndarray:
        """Get all wavelengths in cm (same as all_wls)."""
        return self.all_wls
