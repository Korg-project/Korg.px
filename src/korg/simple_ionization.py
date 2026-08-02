"""
Simple ionization equilibrium solver using Saha equation.

Computes number densities of neutral and ionized species for all elements
in an atmosphere, given temperature, total number density, and abundances.

This is a simplified implementation that assumes electron density is known
(e.g., from a pre-computed model atmosphere). For a fully self-consistent
solution, use the iterative solvers in statmech.py.
"""

import numpy as np
from typing import Dict

from .species import Species, Formula
from .statmech import saha_ion_weights


def compute_ionization_states(
    temperatures: np.ndarray,
    electron_densities: np.ndarray,
    total_number_densities: np.ndarray,
    abundances_A_X: np.ndarray,
    atomic_symbols: list,
    ionization_energies: dict,
    partition_functions: dict,
) -> Dict[Species, np.ndarray]:
    """
    Compute number densities of all ionization states using Saha equation.

    Given an atmosphere with known electron densities (e.g., from a MARCS model),
    compute the number densities of neutral, singly ionized, and doubly ionized
    species for all elements.

    Parameters
    ----------
    temperatures : array, shape (n_layers,)
        Temperature at each layer in K
    electron_densities : array, shape (n_layers,)
        Electron number density at each layer in cm⁻³
        (Must be provided from atmosphere model)
    total_number_densities : array, shape (n_layers,)
        Total particle number density at each layer in cm⁻³
    abundances_A_X : array, shape (n_elements,)
        Elemental abundances in A(X) = log10(N_X/N_H) + 12 format, indexed by
        Z - 1 (index 0 is H, index 1 is He, ...), i.e. exactly what
        :func:`korg.abundances.format_A_X` returns.
    atomic_symbols : list
        List of atomic symbols indexed by Z - 1, i.e.
        ``korg.atomic_data.atomic_symbols`` (``['H', 'He', 'Li', ...]``).
    ionization_energies : dict
        Dictionary mapping atomic number to [χ_I, χ_II, χ_III] in eV
    partition_functions : dict
        Dictionary mapping Species to partition function callables

    Returns
    -------
    number_densities : dict
        Dictionary mapping Species to number density arrays (n_layers,)
        Includes neutral (charge=0), singly ionized (charge=1), and
        doubly ionized (charge=2) for each element.

    Notes
    -----
    This function uses the Saha equation:

        n(X^{i+1}) / n(X^i) = (2/n_e) * (U_{i+1}/U_i) * U_trans * exp(-χ_i/(kT))

    where:
        - U_i is the partition function of ionization state i
        - U_trans = (2πm_e kT/h²)^1.5 is the translational partition function
        - χ_i is the ionization energy from state i to i+1

    The total elemental abundance is partitioned as:
        n_X = n(X^0) + n(X^+) + n(X^{2+})
        n_X = n(X^0) * (1 + w_II + w_III)

    where w_II and w_III are the Saha ionization weights.

    Examples
    --------
    >>> from korg.data_loader import ionization_energies, load_atomic_partition_functions
    >>> from korg.atomic_data import atomic_symbols
    >>> partition_funcs = load_atomic_partition_functions()
    >>> number_densities = compute_ionization_states(
    ...     temperatures, electron_densities, total_densities,
    ...     abundances_A_X, atomic_symbols,
    ...     ionization_energies, partition_funcs
    ... )
    >>> # Access Na I density:
    >>> na_I_density = number_densities[Species('Na', 0)]
    """
    n_layers = len(temperatures)
    abundances_fractional = 10**(abundances_A_X - 12)

    number_densities = {}

    # Process each element. ``atomic_symbols`` and ``abundances_A_X`` are both
    # indexed by Z - 1, so the atomic number is i + 1.
    for i, symbol in enumerate(atomic_symbols):
        if i >= len(abundances_fractional):
            # Skip indices beyond the supplied abundance array
            continue

        atom_number = i + 1  # Atomic number (1=H, 2=He, etc.)

        # Total elemental abundance at each layer
        n_element_total = total_number_densities * abundances_fractional[i]

        # Arrays for ionization states
        n_I = np.zeros(n_layers)    # Neutral
        n_II = np.zeros(n_layers)   # Singly ionized
        n_III = np.zeros(n_layers)  # Doubly ionized

        # Compute ionization equilibrium at each layer
        for i_layer in range(n_layers):
            T = temperatures[i_layer]
            ne = electron_densities[i_layer]

            try:
                # Compute Saha ionization weights
                wII, wIII = saha_ion_weights(
                    T, ne, atom_number,
                    ionization_energies,
                    partition_functions
                )

                # Partition total abundance among ionization states
                # n_total = n_I + n_II + n_III = n_I * (1 + wII + wIII)
                # Therefore: n_I = n_total / (1 + wII + wIII)
                denominator = 1.0 + wII + wIII
                n_I[i_layer] = n_element_total[i_layer] / denominator
                n_II[i_layer] = n_I[i_layer] * wII
                n_III[i_layer] = n_I[i_layer] * wIII

            except (KeyError, IndexError):
                # If ionization energies or partition functions not available,
                # assume all neutral
                n_I[i_layer] = n_element_total[i_layer]
                n_II[i_layer] = 0.0
                n_III[i_layer] = 0.0

        # Store in dictionary
        number_densities[Species(symbol, 0)] = n_I
        number_densities[Species(symbol, 1)] = n_II
        number_densities[Species(symbol, 2)] = n_III

    return number_densities


def check_electron_density_consistency(
    number_densities: Dict[Species, np.ndarray],
    electron_densities: np.ndarray,
    tolerance: float = 0.1
) -> tuple:
    """
    Check if computed ionization states are consistent with provided electron density.

    In a self-consistent solution, the electron density should equal the sum of
    electrons from all ionized species:

        n_e = Σ_X [n(X^+) + 2*n(X^{2+})]

    This function computes the electron density implied by the ionization states
    and compares it to the provided electron density. Large discrepancies indicate
    that the provided electron density is not self-consistent with the Saha
    equation solution.

    Parameters
    ----------
    number_densities : dict
        Dictionary mapping Species to number density arrays
    electron_densities : array
        Provided electron densities (e.g., from atmosphere model)
    tolerance : float, optional
        Fractional tolerance for consistency check (default: 0.1 = 10%)

    Returns
    -------
    is_consistent : bool
        True if electron densities match within tolerance
    implied_ne : array
        Electron density implied by ionization states
    fractional_error : array
        Fractional error (implied - provided) / provided

    Examples
    --------
    >>> is_ok, ne_implied, error = check_electron_density_consistency(
    ...     number_densities, electron_densities
    ... )
    >>> if not is_ok:
    ...     print(f"Warning: electron density inconsistent by {error.max()*100:.1f}%")
    """
    # Compute electron density from ionization states
    # n_e = Σ_X [n(X^+) + 2*n(X^{2+})]
    implied_ne = np.zeros_like(electron_densities)

    for species, n_densities in number_densities.items():
        charge = species.charge
        implied_ne += charge * n_densities

    # Compare to provided electron density
    with np.errstate(divide='ignore', invalid='ignore'):
        fractional_error = (implied_ne - electron_densities) / electron_densities
        fractional_error = np.where(np.isfinite(fractional_error), fractional_error, 0)

    max_error = np.max(np.abs(fractional_error))
    is_consistent = max_error < tolerance

    return is_consistent, implied_ne, fractional_error
