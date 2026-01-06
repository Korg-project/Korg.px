"""
Stellar abundance utilities and solar abundance data.

Provides functions for working with elemental abundances in stellar
spectroscopy format: A(X) = log10(N_X/N_H) + 12

Reference: Korg.jl abundances.jl
"""

import numpy as np
from typing import Dict, Optional, Union, List

from .atomic_data import atomic_numbers, atomic_symbols, MAX_ATOMIC_NUMBER


# Alpha elements: O, Ne, Mg, Si, S, Ar, Ca, Ti (Z = 8, 10, 12, 14, 16, 18, 20, 22)
DEFAULT_ALPHA_ELEMENTS = [8, 10, 12, 14, 16, 18, 20, 22]


# Solar abundances from various sources
# Format: A(X) = log10(N_X/N_H) + 12 for elements Z=1 to Z=92
# Missing elements are set to -5.0

ASPLUND_2009_SOLAR_ABUNDANCES = np.array([
    12.00, 10.93, 1.05, 1.38, 2.70, 8.43, 7.83, 8.69, 4.56, 7.93,
    6.24, 7.60, 6.45, 7.51, 5.41, 7.12, 5.50, 6.40, 5.03, 6.34,
    3.15, 4.95, 3.93, 5.64, 5.43, 7.50, 4.99, 6.22, 4.19, 4.56,
    3.04, 3.65, 2.30, 3.34, 2.54, 3.25, 2.52, 2.87, 2.21, 2.58,
    1.46, 1.88, -5.00, 1.75, 0.91, 1.57, 0.94, 1.71, 0.80, 2.04,
    1.01, 2.18, 1.55, 2.24, 1.08, 2.18, 1.10, 1.58, 0.72, 1.42,
    -5.00, 0.96, 0.52, 1.07, 0.30, 1.10, 0.48, 0.92, 0.10, 0.84,
    0.10, 0.85, -0.12, 0.85, 0.26, 1.40, 1.38, 1.62, 0.92, 1.17,
    0.90, 1.75, 0.65, -5.00, -5.00, -5.00, -5.00, -5.00, -5.00, 0.02,
    -5.00, -0.54
])

ASPLUND_2020_SOLAR_ABUNDANCES = np.array([
    12.00, 10.91, 0.96, 1.38, 2.70, 8.46, 7.83, 8.69, 4.40, 8.06,
    6.22, 7.55, 6.43, 7.51, 5.41, 7.12, 5.31, 6.38, 5.07, 6.30,
    3.14, 4.97, 3.90, 5.62, 5.42, 7.46, 4.94, 6.20, 4.18, 4.56,
    3.02, 3.62, 2.30, 3.34, 2.54, 3.12, 2.32, 2.83, 2.21, 2.59,
    1.47, 1.88, -5.00, 1.75, 0.78, 1.57, 0.96, 1.71, 0.80, 2.02,
    1.01, 2.18, 1.55, 2.22, 1.08, 2.27, 1.11, 1.58, 0.75, 1.42,
    -5.00, 0.95, 0.52, 1.08, 0.31, 1.10, 0.48, 0.93, 0.11, 0.85,
    0.10, 0.85, -0.15, 0.79, 0.26, 1.35, 1.32, 1.61, 0.91, 1.17,
    0.92, 1.95, 0.65, -5.00, -5.00, -5.00, -5.00, -5.00, -5.00, 0.03,
    -5.00, -0.54
])

GREVESSE_2007_SOLAR_ABUNDANCES = np.array([
    12.00, 10.93, 1.05, 1.38, 2.70, 8.39, 7.78, 8.66, 4.56, 7.84,
    6.17, 7.53, 6.37, 7.51, 5.36, 7.14, 5.50, 6.18, 5.08, 6.31,
    3.17, 4.90, 4.00, 5.64, 5.39, 7.45, 4.92, 6.23, 4.21, 4.60,
    2.88, 3.58, 2.29, 3.33, 2.56, 3.25, 2.60, 2.92, 2.21, 2.58,
    1.42, 1.92, -5.00, 1.84, 1.12, 1.66, 0.94, 1.77, 1.60, 2.00,
    1.00, 2.19, 1.51, 2.24, 1.07, 2.17, 1.13, 1.70, 0.58, 1.45,
    -5.00, 1.00, 0.52, 1.11, 0.28, 1.14, 0.51, 0.93, 0.00, 1.08,
    0.06, 0.88, -0.17, 1.11, 0.23, 1.25, 1.38, 1.64, 1.01, 1.13,
    0.90, 2.00, 0.65, -5.00, -5.00, -5.00, -5.00, -5.00, -5.00, 0.06,
    -5.00, -0.52
])

BERGEMANN_2025_SOLAR_ABUNDANCES = np.array([
    12.0, 10.922, 1.04, 1.21, 2.7, 8.51, 7.94, 8.76, 4.4, 8.15,
    6.29, 7.58, 6.43, 7.56, 5.44, 7.16, 5.43, 6.5, 5.09, 6.35,
    3.13, 4.97, 3.89, 5.74, 5.52, 7.51, 4.95, 6.24, 4.24, 4.55,
    3.02, 3.62, 2.34, 3.41, 2.65, 3.31, 2.35, 2.93, 2.3, 2.68,
    1.47, 1.88, -5.0, 1.75, 0.78, 1.57, 0.96, 1.77, 0.8, 2.02,
    1.08, 2.23, 1.76, 2.3, 1.12, 2.27, 1.1, 1.58, 0.75, 1.42,
    -5.0, 0.95, 0.57, 1.08, 0.31, 1.1, 0.48, 0.93, 0.11, 0.85,
    0.1, 0.86, -0.11, 0.79, 0.3, 1.36, 1.42, 1.64, 0.91, 1.14,
    0.95, 1.95, 0.7, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, 0.09,
    -5.0, -0.5
])

# Default solar abundances (matches Julia Korg default)
DEFAULT_SOLAR_ABUNDANCES = BERGEMANN_2025_SOLAR_ABUNDANCES


def format_A_X(
    default_metals_H: float = 0.0,
    default_alpha_H: Optional[float] = None,
    abundances: Optional[Dict[Union[int, str], float]] = None,
    solar_relative: bool = True,
    solar_abundances: Optional[np.ndarray] = None,
    alpha_elements: Optional[List[int]] = None
) -> np.ndarray:
    """
    Returns a 92-element vector containing abundances in A(X) format.

    A(X) = log10(N_X/N_H) + 12 for elements from hydrogen to uranium.

    This function exactly matches Korg.jl's format_A_X behavior.

    Parameters
    ----------
    default_metals_H : float, optional
        [metals/H] is the log10 solar-relative abundance of elements heavier
        than He. It is overridden by default_alpha_H and abundances on a
        per-element basis. Default: 0.0
    default_alpha_H : float, optional
        [alpha/H] is the log10 solar-relative abundance of the alpha elements.
        It is overridden by abundances on a per-element basis.
        Default: same as default_metals_H
    abundances : dict, optional
        A dict mapping atomic numbers (int) or symbols (str) to [X/H] abundances.
        Set solar_relative=False to use A(X) abundances instead.
        These override default_metals_H. This is the only way to specify a
        non-solar abundance of He.
    solar_relative : bool, optional
        When True, interpret abundances as being in [X/H] (log10 solar-relative)
        format. When False, interpret them as A(X) abundances.
        Default: True
    solar_abundances : array, optional
        The set of solar abundances to use, as a vector indexed by atomic number
        (0-indexed, so index 0 = H, index 25 = Fe).
        Default: BERGEMANN_2025_SOLAR_ABUNDANCES
    alpha_elements : list, optional
        List of atomic numbers of the alpha elements.
        Default: [8, 10, 12, 14, 16, 18, 20, 22] (O, Ne, Mg, Si, S, Ar, Ca, Ti)

    Returns
    -------
    A_X : array, shape (92,)
        Absolute abundances A(X) for elements Z=1 to 92

    Examples
    --------
    >>> # Solar abundances
    >>> A_X = format_A_X()
    >>> A_X[0]  # A(H) = 12.0 by definition
    12.0

    >>> # Metal-poor star
    >>> A_X = format_A_X(default_metals_H=-1.0)

    >>> # Alpha-enhanced metal-poor star
    >>> A_X = format_A_X(default_metals_H=-1.0, default_alpha_H=-0.6)

    >>> # Custom Fe abundance
    >>> A_X = format_A_X(abundances={26: -0.5})  # [Fe/H] = -0.5
    >>> A_X = format_A_X(abundances={'Fe': -0.5})  # Same thing

    >>> # Absolute A(X) for Fe instead of [Fe/H]
    >>> A_X = format_A_X(abundances={'Fe': 7.0}, solar_relative=False)
    """
    if solar_abundances is None:
        solar_abundances = DEFAULT_SOLAR_ABUNDANCES

    if alpha_elements is None:
        alpha_elements = DEFAULT_ALPHA_ELEMENTS

    if default_alpha_H is None:
        default_alpha_H = default_metals_H

    if abundances is None:
        abundances = {}

    # Convert keys to atomic numbers and validate
    clean_abundances = {}
    for el, abund in abundances.items():
        if isinstance(el, str):
            if el not in atomic_numbers:
                raise ValueError(f"{el} isn't a valid atomic symbol.")
            Z = atomic_numbers[el]
            if Z in abundances:
                raise ValueError(f"The abundance of {el} was specified by both "
                               f"atomic number and atomic symbol.")
            clean_abundances[Z] = abund
        elif isinstance(el, int):
            if not (1 <= el <= MAX_ATOMIC_NUMBER):
                raise ValueError(f"Z = {el} is not a supported atomic number.")
            clean_abundances[el] = abund
        else:
            raise ValueError(f"{el} isn't a valid element. Keys of the abundances "
                           f"dict should be strings or integers.")

    # Check that H abundance is not set incorrectly
    correct_H_abund = 0.0 if solar_relative else 12.0
    if 1 in clean_abundances and clean_abundances[1] != correct_H_abund:
        silly_abundance = "[H/H]" if solar_relative else "A(H)"
        silly_value = 0 if solar_relative else 12
        raise ValueError(f"{silly_abundance} set, but {silly_abundance} = {silly_value} by "
                        f"definition. Adjust metallicity and abundances to implicitly "
                        f"set the amount of H")

    # Build A(X) vector
    A_X = np.zeros(MAX_ATOMIC_NUMBER)

    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        idx = Z - 1  # 0-indexed

        if Z == 1:  # Hydrogen
            A_X[idx] = 12.0
        elif Z in clean_abundances:  # Explicitly set
            if solar_relative:
                A_X[idx] = clean_abundances[Z] + solar_abundances[idx]
            else:
                A_X[idx] = clean_abundances[Z]
        elif Z in alpha_elements:  # Alpha element
            A_X[idx] = solar_abundances[idx] + default_alpha_H
        else:  # Other element - use solar value adjusted for metallicity
            # Only adjust for metals (Z >= 3), not H or He
            delta = default_metals_H if Z >= 3 else 0.0
            A_X[idx] = solar_abundances[idx] + delta

    return A_X


def _get_multi_X_H(A_X: np.ndarray, Zs: List[int],
                   solar_abundances: np.ndarray) -> float:
    """
    Given a vector of abundances A_X, get [I+J+K/H], where Zs = [I,J,K] is a
    vector of atomic numbers. This is used to calculate [alpha/H] and [metals/H].

    This exactly matches Korg.jl's _get_multi_X_H function.
    """
    # There is no logsumexp in the julia stdlib, but it would make this more stable.
    # The lack of "12"s here is not a mistake. They all cancel.
    A_mX = np.log10(sum(10**A_X[Z-1] for Z in Zs))
    A_mX_solar = np.log10(sum(10**solar_abundances[Z-1] for Z in Zs))
    return A_mX - A_mX_solar


def get_metals_H(
    A_X: np.ndarray,
    solar_abundances: Optional[np.ndarray] = None,
    ignore_alpha: bool = True,
    alpha_elements: Optional[List[int]] = None
) -> float:
    """
    Calculate [metals/H] given a vector A_X of absolute abundances.

    This exactly matches Korg.jl's get_metals_H function.

    Parameters
    ----------
    A_X : array, shape (92,)
        Absolute abundances A(X) = log10(N_X/N_H) + 12
    solar_abundances : array, optional
        The set of solar abundances to use.
        Default: BERGEMANN_2025_SOLAR_ABUNDANCES
    ignore_alpha : bool, optional
        Whether to ignore the alpha elements when calculating [metals/H].
        If True, [metals/H] is calculated using all elements heavier than He
        except carbon and the alpha elements. If False, all metals are used.
        Default: True
    alpha_elements : list, optional
        List of atomic numbers of the alpha elements.
        Default: [8, 10, 12, 14, 16, 18, 20, 22]

    Returns
    -------
    M_H : float
        [metals/H] metallicity
    """
    if solar_abundances is None:
        solar_abundances = DEFAULT_SOLAR_ABUNDANCES

    if alpha_elements is None:
        alpha_elements = DEFAULT_ALPHA_ELEMENTS

    if ignore_alpha:
        # Exclude alpha elements and carbon (Z=6)
        els = [Z for Z in range(3, MAX_ATOMIC_NUMBER + 1)
               if Z not in alpha_elements and Z != 6]
    else:
        els = list(range(3, MAX_ATOMIC_NUMBER + 1))

    return _get_multi_X_H(A_X, els, solar_abundances)


def get_alpha_H(
    A_X: np.ndarray,
    solar_abundances: Optional[np.ndarray] = None,
    alpha_elements: Optional[List[int]] = None
) -> float:
    """
    Calculate [alpha/H] given a vector A_X of absolute abundances.

    Here, the alpha elements are defined to be O, Ne, Mg, Si, S, Ar, Ca, Ti.

    This exactly matches Korg.jl's get_alpha_H function.

    Parameters
    ----------
    A_X : array, shape (92,)
        Absolute abundances A(X) = log10(N_X/N_H) + 12
    solar_abundances : array, optional
        The set of solar abundances to use.
        Default: BERGEMANN_2025_SOLAR_ABUNDANCES
    alpha_elements : list, optional
        List of atomic numbers of the alpha elements.
        Default: [8, 10, 12, 14, 16, 18, 20, 22]

    Returns
    -------
    alpha_H : float
        [alpha/H] alpha enhancement
    """
    if solar_abundances is None:
        solar_abundances = DEFAULT_SOLAR_ABUNDANCES

    if alpha_elements is None:
        alpha_elements = DEFAULT_ALPHA_ELEMENTS

    return _get_multi_X_H(A_X, alpha_elements, solar_abundances)


def A_X_to_absolute(A_X: np.ndarray) -> np.ndarray:
    """
    Convert A(X) abundances to absolute number fractions.

    Parameters
    ----------
    A_X : array, shape (92,)
        Abundances in A(X) = log10(N_X/N_H) + 12 format

    Returns
    -------
    abundances : array, shape (92,)
        Absolute number fractions (N_X / N_total)
        Normalized so sum(abundances) = 1.0

    Examples
    --------
    >>> A_X = format_A_X()
    >>> abs_abund = A_X_to_absolute(A_X)
    >>> abs_abund.sum()
    1.0
    """
    # A(X) = log10(N_X/N_H) + 12
    # So: N_X/N_H = 10^(A(X) - 12)
    rel_to_H = 10.0 ** (A_X - 12.0)

    # N_total = N_H + sum(N_X) = N_H * (1 + sum(N_X/N_H))
    # So: N_X/N_total = (N_X/N_H) / sum(N_X/N_H)
    sum_rel = np.sum(rel_to_H)

    abs_abundances = rel_to_H / sum_rel

    return abs_abundances


def get_solar_abundances(source: str = "bergemann_2025") -> np.ndarray:
    """
    Get solar abundances from a specified source.

    Parameters
    ----------
    source : str, optional
        Source of solar abundances. Options:
        - "bergemann_2025" (default)
        - "asplund_2020"
        - "asplund_2009"
        - "grevesse_2007"

    Returns
    -------
    A_X : array, shape (92,)
        Solar abundances A(X) for elements Z=1 to 92

    Examples
    --------
    >>> A_X = get_solar_abundances()
    >>> A_X[0]  # A(H) = 12.0 by definition
    12.0
    >>> A_X[25]  # A(Fe) ~ 7.5
    7.51
    """
    sources = {
        "bergemann_2025": BERGEMANN_2025_SOLAR_ABUNDANCES,
        "asplund_2020": ASPLUND_2020_SOLAR_ABUNDANCES,
        "asplund_2009": ASPLUND_2009_SOLAR_ABUNDANCES,
        "grevesse_2007": GREVESSE_2007_SOLAR_ABUNDANCES,
    }
    source = source.lower()
    if source not in sources:
        raise ValueError(f"Unknown source: {source}. Options: {list(sources.keys())}")
    return sources[source].copy()
