"""
Statistical mechanics functions.

Functions for computing occupation probabilities, partition functions,
and related quantities in stellar atmospheres.
"""

import functools
from typing import NamedTuple

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from .constants import (kboltz_eV, kboltz_cgs, hplanck_cgs, bohr_radius_cgs,
                        RydbergH_eV, eV_to_cgs, electron_charge_cgs, electron_mass_cgs)


class ChemicalEquilibriumData(NamedTuple):
    """
    Pre-computed data for JIT-compatible chemical equilibrium calculations.

    All arrays are pre-computed on a temperature grid to allow fast interpolation
    inside JIT-compiled functions.
    """
    # Temperature grid for interpolation
    log_T_grid: jnp.ndarray  # shape (n_temps,)

    # Ionization energies: shape (92, 3) for [χ₁, χ₂, χ₃]
    ionization_energies: jnp.ndarray

    # Partition function values on T grid: shape (92, 3, n_temps)
    # For each element Z, ionization states 0,1,2
    partition_func_values: jnp.ndarray
    # Original CubicSpline knots for atomic partition functions (padded to 201 with inf/zeros)
    pf_orig_t: jnp.ndarray   # shape (92, 3, 201) — knot positions (log T)
    pf_orig_u: jnp.ndarray   # shape (92, 3, 201) — knot values
    pf_orig_h: jnp.ndarray   # shape (92, 3, 201) — h[i] = t[i] - t[i-1]
    pf_orig_z: jnp.ndarray   # shape (92, 3, 201) — second derivatives
    pf_orig_n: jnp.ndarray   # shape (92, 3) int32 — number of valid knots
    # h array for uniform-grid cubic spline (used for molecular partition funcs)
    log_T_h: jnp.ndarray     # shape (n_temps,): [0, Δlog_T, ...]

    # Molecular data
    n_molecules: int
    mol_atoms_array: jnp.ndarray  # shape (n_molecules, 6), padded with -1
    mol_charges: jnp.ndarray  # shape (n_molecules,)
    mol_n_atoms: jnp.ndarray  # shape (n_molecules,)
    mol_log_K_values: jnp.ndarray  # shape (n_molecules, n_temps) - log K on T grid
    mol_partition_func_values: jnp.ndarray  # shape (n_molecules, n_temps) - U(T) for each molecule
    mol_partition_func_z: jnp.ndarray  # cubic spline z for mol partition funcs: (n_molecules, n_temps)
    mol_atom_consume: jnp.ndarray  # shape (n_molecules, 92) - atoms of each element per molecule


def hummer_mihalas_w(T, n_eff, nH, nHe, ne, use_hubeny_generalization=False):
    """
    Calculate the correction, w, to the occupation fraction of a hydrogen energy level.

    Uses the occupation probability formalism from Hummer and Mihalas 1988,
    optionally with the generalization by Hubeny+ 1994.

    Parameters
    ----------
    T : float
        Temperature in K.
    n_eff : float
        Effective principal quantum number.
    nH : float
        Number density of neutral hydrogen in cm⁻³.
    nHe : float
        Number density of neutral helium in cm⁻³.
    ne : float
        Number density of electrons in cm⁻³.
    use_hubeny_generalization : bool, optional
        Use Hubeny+ 1994 generalization (default: False).

    Returns
    -------
    float
        Occupation probability correction factor w.

    Notes
    -----
    The expression for w is in equation 4.71 of Hummer & Mihalas 1988.
    K, the QM correction, is defined in equation 4.24.

    This is based partially on Paul Barklem and Kjell Eriksson's WCALC
    fortran routine, which is used by Turbospectrum and SME.

    References
    ----------
    - Hummer & Mihalas 1988
    - Hubeny+ 1994 (optional generalization)
    - Barklem & Eriksson's HBOP routine
    """
    # Contribution from neutral species (H and He in ground state)
    # This is sqrt<r^2> assuming l=0
    r_level = jnp.sqrt(5.0 / 2.0 * n_eff**4 + 1.0 / 2.0 * n_eff**2) * bohr_radius_cgs
    neutral_term = (nH * (r_level + jnp.sqrt(3.0) * bohr_radius_cgs)**3 +
                    nHe * (r_level + 1.02 * bohr_radius_cgs)**3)

    # Contributions from ions (assumed to be all singly ionized, so n_ion = n_e)
    # K is a QM correction defined in H&M '88 equation 4.24
    K = jnp.where(
        n_eff > 3,
        # WCALC drops the final factor, which is within 1% of unity for all n
        16.0 / 3.0 * (n_eff / (n_eff + 1.0))**2 *
        ((n_eff + 7.0 / 6.0) / (n_eff**2 + n_eff + 1.0 / 2.0)),
        1.0
    )

    χ = RydbergH_eV / n_eff**2 * eV_to_cgs  # binding energy
    e = electron_charge_cgs

    if use_hubeny_generalization:
        # Straight port from HBOP - not default
        def hubeny_term(ne, T):
            A = 0.09 * jnp.exp(0.16667 * jnp.log(ne)) / jnp.sqrt(T)
            X = jnp.exp(3.15 * jnp.log(1.0 + A))
            BETAC = 8.3e14 * jnp.exp(-0.66667 * jnp.log(ne)) * K / n_eff**4
            F = 0.1402 * X * BETAC**3 / (1.0 + 0.1285 * X * BETAC * jnp.sqrt(BETAC))
            return jnp.log(F / (1.0 + F)) / (-4.0 * jnp.pi / 3.0)

        charged_term = jnp.where(
            (ne > 10) & (T > 10),
            hubeny_term(ne, T),
            0.0
        )
    else:
        charged_term = 16.0 * ((e**2) / (χ * jnp.sqrt(K)))**3 * ne

    return jnp.exp(-4.0 * jnp.pi / 3.0 * (neutral_term + charged_term))


def hummer_mihalas_U_H(T, nH, nHe, ne, use_hubeny_generalization=False):
    """
    Calculate the partition function of neutral hydrogen using occupation probability formalism.

    WARNING: This is experimental and not used by Korg for spectral synthesis.

    Uses the occupation probability formalism from Hummer and Mihalas 1988.

    Parameters
    ----------
    T : float
        Temperature in K.
    nH : float
        Number density of neutral hydrogen in cm⁻³.
    nHe : float
        Number density of neutral helium in cm⁻³.
    ne : float
        Number density of electrons in cm⁻³.
    use_hubeny_generalization : bool, optional
        Use Hubeny+ 1994 generalization (default: False).

    Returns
    -------
    float
        Partition function for neutral hydrogen.

    Notes
    -----
    Energy levels and degeneracies are from NIST.

    See Also
    --------
    hummer_mihalas_w : Occupation probability correction function
    """
    # Hydrogen energy levels from NIST (in eV)
    hydrogen_energy_levels = jnp.array([
        0.0, 10.19880615024, 10.19881052514816, 10.19885151459, 12.0874936591,
        12.0874949611, 12.0875070783, 12.0875071004, 12.0875115582, 12.74853244632,
        12.74853299663, 12.7485381084, 12.74853811674, 12.74853999753, 12.748539998,
        12.7485409403, 13.054498182, 13.054498464, 13.054501074, 13.054501086,
        13.054502042, 13.054502046336, 13.054502526, 13.054502529303, 13.054502819633,
        13.22070146198, 13.22070162532, 13.22070313941, 13.22070314214, 13.220703699081,
        13.22070369934, 13.220703978574, 13.220703979103, 13.220704146258, 13.220704146589,
        13.220704258272, 13.320916647, 13.32091675, 13.320917703, 13.320917704,
        13.320918056, 13.38596007869, 13.38596014765, 13.38596078636, 13.38596078751,
        13.385961022639, 13.4305536, 13.430553648, 13.430554096, 13.430554098,
        13.430554262, 13.462451058, 13.462451094, 13.46245141908, 13.462451421,
        13.46245154007, 13.486051554, 13.486051581, 13.486051825, 13.486051827,
        13.486051916, 13.504001658, 13.504001678, 13.50400186581, 13.504001867,
        13.50400193582
    ])

    hydrogen_energy_level_degeneracies = jnp.array([
        2, 2, 2, 4, 2, 2, 4, 4, 6, 2, 2, 4, 4, 6, 6, 8, 2, 2, 4, 4, 6, 6, 8, 8, 10,
        2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 2, 2, 4, 4, 6, 2, 2, 4, 4, 6, 2, 2, 4, 4,
        6, 2, 2, 4, 4, 6, 2, 2, 4, 4, 6, 2, 2, 4, 4, 6
    ], dtype=jnp.int32)

    hydrogen_energy_level_n = jnp.array([
        1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9,
        9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12
    ], dtype=jnp.int32)

    # For each level, calculate the correction w and add the term to U
    # The expression for w comes from Hummer and Mihalas 1988 equation 4.71
    def level_contribution(E, g, n):
        n_eff = jnp.sqrt(RydbergH_eV / (RydbergH_eV - E))  # times Z, which is 1 for H
        w = hummer_mihalas_w(T, n_eff, nH, nHe, ne,
                             use_hubeny_generalization=use_hubeny_generalization)
        return w * g * jnp.exp(-E / (kboltz_eV * T))

    # Sum contributions from all levels
    U = jnp.sum(jnp.array([
        level_contribution(E, g, n)
        for E, g, n in zip(hydrogen_energy_levels,
                          hydrogen_energy_level_degeneracies,
                          hydrogen_energy_level_n)
    ]))

    return U


def translational_U(m, T):
    """
    Translational partition function contribution for a free particle.

    Used in the Saha equation to account for the translational motion of
    free electrons.

    Parameters
    ----------
    m : float
        Particle mass in grams
    T : float
        Temperature in K

    Returns
    -------
    float
        Translational partition function: (2π m k T / h²)^1.5

    Notes
    -----
    This is the quantum-mechanical partition function for a free particle
    in a unit volume, arising from the de Broglie wavelength.

    Reference
    ---------
    Kurucz 1970, section 5.2
    """
    return (2.0 * jnp.pi * m * kboltz_cgs * T / (hplanck_cgs**2))**1.5


def saha_ion_weights(T, ne, atom, ionization_energies, partition_funcs):
    """
    Calculate ionization ratios using the Saha equation.

    Returns the ratios of singly ionized to neutral and doubly ionized to
    neutral atoms for a given element.

    Parameters
    ----------
    T : float
        Temperature in K
    ne : float
        Electron number density in cm⁻³
    atom : int
        Atomic number (1 for H, 2 for He, etc.)
    ionization_energies : dict
        Dictionary mapping atomic numbers to [χ₁, χ₂, χ₃] in eV
    partition_funcs : dict
        Dictionary mapping Species to partition function callables
        (functions of ln(T))

    Returns
    -------
    tuple
        (wII, wIII) where:
        - wII = n(X II) / n(X I)  (ratio of singly ionized to neutral)
        - wIII = n(X III) / n(X I)  (ratio of doubly ionized to neutral)

    Notes
    -----
    The Saha equation for the first ionization is:

        n(X II) / n(X I) = (2/ne) × (U_II/U_I) × U_trans × exp(-χ_I/(kT))

    where U_trans = (2π m_e k T / h²)^1.5 is the translational partition
    function for the free electron.

    For hydrogen, wIII = 0 since it cannot be doubly ionized.

    Reference
    ---------
    Kurucz 1970, equation 5.10
    Gray 2005, "The Observation and Analysis of Stellar Photospheres", Ch. 8
    """
    from .species import Species, Formula

    χI, χII, χIII = ionization_energies[atom]

    # Get partition functions for neutral and ionized states
    formula = Formula(atom)
    UI = partition_funcs[Species(formula, 0)](jnp.log(T))
    UII = partition_funcs[Species(formula, 1)](jnp.log(T))

    # Translational partition function for free electron
    transU = translational_U(electron_mass_cgs, T)

    # Saha equation for first ionization (clip ne to prevent division by zero)
    ne_clipped = jnp.clip(ne, 1e-12, jnp.inf)
    wII = 2.0 / ne_clipped * (UII / UI) * transU * jnp.exp(-χI / (kboltz_eV * T))

    # Second ionization (if applicable)
    if atom == 1:  # Hydrogen cannot be doubly ionized
        wIII = 0.0
    else:
        UIII = partition_funcs[Species(formula, 2)](jnp.log(T))
        wIII = wII * 2.0 / ne_clipped * (UIII / UII) * transU * jnp.exp(-χII / (kboltz_eV * T))

    return wII, wIII


def get_log_nK(molecule, T, log_equilibrium_constants):
    """
    Convert equilibrium constant from partial pressure to number density form.

    Equilibrium constants for molecules are typically tabulated in terms of
    partial pressures. This function converts them to number density form
    for use in chemical equilibrium calculations.

    Parameters
    ----------
    molecule : Species
        The molecular species
    T : float
        Temperature in K
    log_equilibrium_constants : dict
        Dictionary mapping Species to log₁₀(K) functions in partial pressure form

    Returns
    -------
    float
        log₁₀(K) in number density form, where K = Π n(atoms) / n(molecule)

    Notes
    -----
    The conversion accounts for the ideal gas law relationship between
    partial pressure and number density:

        p = n × k × T

    For a reaction A + B ↔ AB:

        K_p = p(A) × p(B) / p(AB)
        K_n = n(A) × n(B) / n(AB)

    These are related by:

        log₁₀(K_n) = log₁₀(K_p) - (n_atoms - 1) × log₁₀(kT)

    where n_atoms is the number of atoms in the molecule (2 for diatomics,
    3 for triatomics, etc.).

    Reference
    ---------
    Tsuji 1973, A&A 23, 411
    """
    # Get log_K_p from the equilibrium constant function
    log_K_p = log_equilibrium_constants[molecule](jnp.log(T))

    # Number of atoms in the molecule
    n_atoms = molecule.n_atoms()

    # Convert from partial pressure to number density form
    log_nK = log_K_p - (n_atoms - 1) * jnp.log10(kboltz_cgs * T)

    return log_nK


# Maximum atomic number to consider
MAX_ATOMIC_NUMBER = 92


def precompute_equilibrium_data(T, n_total, absolute_abundances,
                                 ionization_energies, partition_funcs,
                                 log_equilibrium_constants):
    """
    Pre-compute all data needed for chemical equilibrium as pure arrays.

    This extracts all data from Python objects (dicts, partition funcs) into
    JAX-compatible arrays that can be used in JIT-compiled functions.

    Returns
    -------
    tuple
        (abund_array, wII_ne_array, wIII_ne2_array,
         log_nKs, mol_atoms_array, mol_charges, mol_n_atoms)
    """
    from .species import Species, Formula

    # Convert abundances to array if it's a dict
    if isinstance(absolute_abundances, dict):
        abund_array = jnp.zeros(MAX_ATOMIC_NUMBER)
        for Z, abund in absolute_abundances.items():
            abund_array = abund_array.at[Z-1].set(abund)
    else:
        abund_array = jnp.asarray(absolute_abundances)

    # Precompute Saha ion weights (with ne=1, will scale by actual ne later)
    wII_ne_array = jnp.zeros(MAX_ATOMIC_NUMBER)
    wIII_ne2_array = jnp.zeros(MAX_ATOMIC_NUMBER)

    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        wII, wIII = saha_ion_weights(T, 1.0, Z, ionization_energies, partition_funcs)
        wII_ne_array = wII_ne_array.at[Z-1].set(wII)
        wIII_ne2_array = wIII_ne2_array.at[Z-1].set(wIII)

    # Get list of molecules and filter out those with invalid equilibrium constants
    molecules_all = list(log_equilibrium_constants.keys())

    # Precompute log equilibrium constants and molecule data as arrays
    log_nKs_list = []
    mol_atoms_list = []  # Each molecule's atom indices (Z-1), padded to length 6
    mol_charges_list = []
    mol_n_atoms_list = []

    for mol in molecules_all:
        log_nK = get_log_nK(mol, T, log_equilibrium_constants)
        if jnp.isfinite(log_nK):
            log_nKs_list.append(float(log_nK))
            mol_charges_list.append(mol.charge)

            atoms = mol.get_atoms()
            mol_n_atoms_list.append(len(atoms))
            # Pad to 6 atoms, use -1 as sentinel for unused slots
            padded = list(atoms - 1) + [-1] * (6 - len(atoms))
            mol_atoms_list.append(padded)

    if len(log_nKs_list) > 0:
        log_nKs = jnp.array(log_nKs_list)
        mol_atoms_array = jnp.array(mol_atoms_list, dtype=jnp.int32)
        mol_charges = jnp.array(mol_charges_list, dtype=jnp.int32)
        mol_n_atoms = jnp.array(mol_n_atoms_list, dtype=jnp.int32)
    else:
        log_nKs = jnp.array([])
        mol_atoms_array = jnp.zeros((0, 6), dtype=jnp.int32)
        mol_charges = jnp.array([], dtype=jnp.int32)
        mol_n_atoms = jnp.array([], dtype=jnp.int32)

    return (abund_array, wII_ne_array, wIII_ne2_array,
            log_nKs, mol_atoms_array, mol_charges, mol_n_atoms)


def _compute_residuals_core(x, n_total, abund_array, wII_ne_array, wIII_ne2_array,
                            log_nKs, mol_atoms_array, mol_charges, mol_n_atoms):
    """
    Core residuals computation using pure JAX arrays.

    This is the JIT-compilable inner function.
    """
    # Extract electron density (scaled for numerical stability)
    ne = jnp.clip(jnp.abs(x[-1]) * n_total * 1e-5, 1e-12, jnp.inf)

    # Extract neutral fractions and ensure positive
    neutral_fractions = jnp.abs(x[:MAX_ATOMIC_NUMBER])

    # Total atom number densities (excluding electrons)
    atom_number_densities = abund_array * (n_total - ne)

    # Neutral atomic number densities
    neutral_number_densities = atom_number_densities * neutral_fractions

    # Vectorized Saha weights scaled by ne
    wII = wII_ne_array / jnp.clip(ne, 1e-12, jnp.inf)
    wIII = wIII_ne2_array / jnp.clip(ne * ne, 1e-24, jnp.inf)

    # Element conservation residuals (vectorized)
    F_elements = atom_number_densities - (1.0 + wII + wIII) * neutral_number_densities

    # Electron conservation: sum of contributions from ions
    F_electron = jnp.sum((wII + 2.0 * wIII) * neutral_number_densities) - ne

    # Combine into full residual vector
    F = jnp.concatenate([F_elements, jnp.array([F_electron])])

    # Add molecular contributions using scan to avoid Python loops
    log_neutral_densities = jnp.log10(jnp.clip(neutral_number_densities, 1e-99, jnp.inf))

    def process_molecule(F, mol_data):
        mol_idx, log_nK, atoms, charge, n_atoms = mol_data

        # For neutral molecules: sum log densities of constituent atoms
        # For ionized molecules: first atom is ionized

        def neutral_mol_contribution(F):
            # Sum log densities for each atom in the molecule
            log_sum = jnp.sum(jnp.where(
                jnp.arange(6) < n_atoms,
                log_neutral_densities[atoms],
                0.0
            ))
            n_mol = 10.0 ** jnp.clip(log_sum - log_nK, -300, 300)

            # Subtract from each element's conservation
            # Use a scatter-add approach
            updates = jnp.where(jnp.arange(6) < n_atoms, -n_mol, 0.0)
            F_new = F.at[atoms[0]].add(jnp.where(n_atoms > 0, updates[0], 0.0))
            F_new = F_new.at[atoms[1]].add(jnp.where(n_atoms > 1, updates[1], 0.0))
            F_new = F_new.at[atoms[2]].add(jnp.where(n_atoms > 2, updates[2], 0.0))
            F_new = F_new.at[atoms[3]].add(jnp.where(n_atoms > 3, updates[3], 0.0))
            F_new = F_new.at[atoms[4]].add(jnp.where(n_atoms > 4, updates[4], 0.0))
            F_new = F_new.at[atoms[5]].add(jnp.where(n_atoms > 5, updates[5], 0.0))
            return F_new

        def ionized_mol_contribution(F):
            # Singly ionized diatomic: first atom ionized, second neutral
            idx1, idx2 = atoms[0], atoms[1]
            wII_atom = wII[idx1]
            n1_II_log = log_neutral_densities[idx1] + jnp.log10(jnp.clip(wII_atom, 1e-99, jnp.inf))
            n2_I_log = log_neutral_densities[idx2]

            n_mol = 10.0 ** jnp.clip(n1_II_log + n2_I_log - log_nK, -300, 300)

            F_new = F.at[idx1].add(-n_mol)
            F_new = F_new.at[idx2].add(-n_mol)
            F_new = F_new.at[-1].add(n_mol)  # electron contribution
            return F_new

        F = jax.lax.cond(charge == 0, neutral_mol_contribution, ionized_mol_contribution, F)
        return F, None

    # Process all molecules
    n_molecules = log_nKs.shape[0]
    if n_molecules > 0:
        mol_indices = jnp.arange(n_molecules)
        mol_data = (mol_indices, log_nKs, mol_atoms_array, mol_charges, mol_n_atoms)
        F, _ = jax.lax.scan(process_molecule, F,
                            (mol_indices, log_nKs, mol_atoms_array, mol_charges, mol_n_atoms))

    # Normalize residuals (avoid division by zero for elements with zero abundance)
    F = F.at[:MAX_ATOMIC_NUMBER].set(
        jnp.where(atom_number_densities > 0,
                 F[:MAX_ATOMIC_NUMBER] / jnp.clip(atom_number_densities, 1e-99, jnp.inf),
                 0.0))
    F = F.at[-1].set(F[-1] / jnp.clip(ne * 1e-5, 1e-12, jnp.inf))

    return F


def setup_chemical_equilibrium_residuals(T, n_total, absolute_abundances,
                                        ionization_energies, partition_funcs,
                                        log_equilibrium_constants):
    """
    Set up the residual function for chemical equilibrium.

    This creates a closure that computes the residuals for the system of
    nonlinear equations that defines chemical equilibrium.

    Parameters
    ----------
    T : float
        Temperature in K
    n_total : float
        Total number density in cm⁻³
    absolute_abundances : dict or array
        Absolute abundances N(X)/N_total for each element (indexed by Z)
    ionization_energies : dict
        Dictionary mapping atomic numbers to [χ₁, χ₂, χ₃] in eV
    partition_funcs : dict
        Dictionary mapping Species to partition function callables
    log_equilibrium_constants : dict
        Dictionary mapping molecular Species to log₁₀(K) functions

    Returns
    -------
    callable
        Function residuals(x) that computes residuals given state vector x
    """
    # Pre-compute all data as arrays
    (abund_array, wII_ne_array, wIII_ne2_array,
     log_nKs, mol_atoms_array, mol_charges, mol_n_atoms) = precompute_equilibrium_data(
        T, n_total, absolute_abundances, ionization_energies,
        partition_funcs, log_equilibrium_constants
    )

    # Create a JIT-compiled residuals function
    @jax.jit
    def residuals(x):
        return _compute_residuals_core(
            x, n_total, abund_array, wII_ne_array, wIII_ne2_array,
            log_nKs, mol_atoms_array, mol_charges, mol_n_atoms
        )

    return residuals


def newton_solve_jax(residuals_func, x0, ftol=1e-8, max_iter=1000):
    """
    Solve nonlinear system using Newton's method with JAX autodiff.

    This mimics Julia's NLsolve with method=:newton and autodiff=:forward.
    Uses jax.lax.while_loop for JIT compatibility.

    Parameters
    ----------
    residuals_func : callable
        Function that computes residuals F(x) = 0
    x0 : jax array
        Initial guess
    ftol : float
        Tolerance on ||F(x)||
    max_iter : int
        Maximum number of iterations

    Returns
    -------
    tuple
        (solution, converged, residual_norm, iterations)
    """
    def cond_fun(state):
        x, iteration, converged, residual_norm = state
        return (iteration < max_iter) & (~converged) & jnp.all(jnp.isfinite(x))

    def body_fun(state):
        x, iteration, converged, residual_norm = state

        # Compute residuals
        F = residuals_func(x)
        residual_norm = jnp.max(jnp.abs(F))

        # Check convergence (infinity norm, matching NLsolve's f_converged)
        converged = residual_norm < ftol

        # Compute Jacobian using forward-mode autodiff
        J = jax.jacfwd(residuals_func)(x)

        # Tiny regularization only to handle exact singularity from zero-abundance
        # elements; 1e-12 is far below ftol=1e-8 so it does not bias the solution.
        n = J.shape[0]
        J_reg = J + 1e-12 * jnp.eye(n)

        # Solve J * dx = -F for the Newton step (LU, matching Julia's A \ b)
        dx = jnp.linalg.solve(J_reg, -F)

        # Replace NaN/Inf steps with zero
        dx = jnp.where(jnp.isfinite(dx), dx, 0.0)

        # Only update if not converged
        x_new = jnp.where(converged, x, x + dx)

        return (x_new, iteration + 1, converged, residual_norm)

    # Initial state
    F0 = residuals_func(x0)
    residual_norm0 = jnp.max(jnp.abs(F0))
    converged0 = residual_norm0 < ftol
    init_state = (x0, 0, converged0, residual_norm0)

    # Run the loop
    x_final, iterations, converged, residual_norm = jax.lax.while_loop(
        cond_fun, body_fun, init_state
    )

    return x_final, converged, residual_norm, iterations


def precompute_chemical_equilibrium_data(ionization_energies, partition_funcs,
                                          log_equilibrium_constants,
                                          T_min=1000.0, T_max=50000.0, n_temps=500):
    """
    Pre-compute all data needed for JIT-compatible chemical equilibrium.

    This function evaluates partition functions and equilibrium constants
    on a temperature grid, allowing interpolation inside JIT-compiled code.

    Parameters
    ----------
    ionization_energies : dict
        Dictionary mapping atomic numbers to [χ₁, χ₂, χ₃] in eV
    partition_funcs : dict
        Dictionary mapping Species to partition function callables
    log_equilibrium_constants : dict
        Dictionary mapping molecular Species to log₁₀(K) functions
    T_min, T_max : float
        Temperature range in K
    n_temps : int
        Number of temperature grid points

    Returns
    -------
    ChemicalEquilibriumData
        Pre-computed data structure for use with chemical_equilibrium_jit
    """
    from .species import Species, Formula

    # Create temperature grid (log-spaced for better interpolation)
    log_T_grid = jnp.linspace(jnp.log(T_min), jnp.log(T_max), n_temps)
    T_grid = jnp.exp(log_T_grid)

    # Build ionization energies array
    ion_energies = np.zeros((MAX_ATOMIC_NUMBER, 3))
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        if Z in ionization_energies:
            ion_energies[Z-1] = ionization_energies[Z]

    # Build partition function values on T grid using numpy (no JAX compilation).
    # Shape: (92, 3, n_temps) for elements 1-92, charge states 0,1,2
    from .cubic_splines import cubic_spline as _make_cubic_spline
    MAX_ATOM_KNOTS = 201
    pf_values = np.zeros((MAX_ATOMIC_NUMBER, 3, n_temps))
    # Original CubicSpline knot arrays, padded to MAX_ATOM_KNOTS
    pf_orig_t = np.full((MAX_ATOMIC_NUMBER, 3, MAX_ATOM_KNOTS), np.inf)
    pf_orig_u = np.zeros((MAX_ATOMIC_NUMBER, 3, MAX_ATOM_KNOTS))
    pf_orig_h = np.zeros((MAX_ATOMIC_NUMBER, 3, MAX_ATOM_KNOTS))
    pf_orig_z = np.zeros((MAX_ATOMIC_NUMBER, 3, MAX_ATOM_KNOTS))
    pf_orig_n = np.ones((MAX_ATOMIC_NUMBER, 3), dtype=np.int32)  # default 1 (safe dummy)
    log_T_np = np.asarray(log_T_grid)
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        formula = Formula(Z)
        for charge in range(3):
            species = Species(formula, charge)
            if species in partition_funcs:
                pf_func = partition_funcs[species]
                interp = getattr(pf_func, '_interpolator', pf_func)
                # Use numpy_eval for the tabulated values (Picard iteration / jnp.interp)
                if hasattr(pf_func, 'numpy_eval'):
                    pf_values[Z-1, charge, :] = pf_func.numpy_eval(log_T_np)
                else:
                    pf_values[Z-1, charge, :] = np.asarray(pf_func(log_T_np))
                # Store original CubicSpline knots for exact evaluation in line absorption
                if hasattr(interp, 't'):
                    n = len(interp.t)
                    pf_orig_t[Z-1, charge, :n] = np.asarray(interp.t)
                    pf_orig_u[Z-1, charge, :n] = np.asarray(interp.u)
                    pf_orig_h[Z-1, charge, :n] = np.asarray(interp.h)
                    pf_orig_z[Z-1, charge, :n] = np.asarray(interp.z)
                    pf_orig_n[Z-1, charge]      = n
                else:
                    # Fallback: fit spline through tabulated values
                    cs = _make_cubic_spline(log_T_np, pf_values[Z-1, charge, :], extrapolate=True)
                    pf_orig_t[Z-1, charge, :n_temps] = log_T_np
                    pf_orig_u[Z-1, charge, :n_temps] = np.asarray(cs.u)
                    pf_orig_h[Z-1, charge, :n_temps] = np.asarray(cs.h)
                    pf_orig_z[Z-1, charge, :n_temps] = np.asarray(cs.z)
                    pf_orig_n[Z-1, charge]            = n_temps

    # Process molecules
    molecules_all = list(log_equilibrium_constants.keys())
    mol_atoms_list = []
    mol_charges_list = []
    mol_n_atoms_list = []
    mol_log_K_list = []
    mol_pf_list = []

    for mol in molecules_all:
        mol_charges_list.append(mol.charge)
        atoms = mol.get_atoms()
        mol_n_atoms_list.append(len(atoms))
        padded = list(atoms - 1) + [-1] * (6 - len(atoms))
        mol_atoms_list.append(padded)

        # Evaluate log K on T grid using numpy eval to avoid per-molecule JAX recompilation
        log_K_func = log_equilibrium_constants[mol]
        if hasattr(log_K_func, 'numpy_eval'):
            mol_log_K_list.append(log_K_func.numpy_eval(log_T_np))
        else:
            # Polyatomic closure: evaluates atomic PF funcs inside; call with numpy array
            # so JAX operations only compile once per unique knot shape
            mol_log_K_list.append(np.asarray(log_K_func(log_T_np)))

        # Molecular partition function (needed for line opacity calculation)
        pf_func = partition_funcs.get(mol, None)
        if pf_func is not None:
            if hasattr(pf_func, 'numpy_eval'):
                mol_pf_list.append(pf_func.numpy_eval(log_T_np))
            else:
                mol_pf_list.append(np.asarray(pf_func(log_T_np)))
        else:
            mol_pf_list.append(np.ones(n_temps))  # fallback: U=1

    n_molecules = len(molecules_all)

    # Precompute h array for cubic spline eval: [0, Δlog_T, ...]
    log_T_h = np.concatenate([[0.0], np.diff(log_T_np)])

    if n_molecules > 0:
        mol_atoms_array = jnp.array(mol_atoms_list, dtype=jnp.int32)
        mol_charges = jnp.array(mol_charges_list, dtype=jnp.int32)
        mol_n_atoms = jnp.array(mol_n_atoms_list, dtype=jnp.int32)
        mol_log_K_values = jnp.array(mol_log_K_list)
        mol_pf_np = np.array(mol_pf_list)
        mol_partition_func_values = jnp.array(mol_pf_np)
        mol_pf_z_np = np.zeros_like(mol_pf_np)
        for i in range(n_molecules):
            cs = _make_cubic_spline(log_T_np, mol_pf_np[i], extrapolate=True)
            mol_pf_z_np[i] = np.asarray(cs.z)
        mol_partition_func_z = jnp.array(mol_pf_z_np)
        # mol_atom_consume[i, Z-1] = # atoms of element Z in molecule i
        M_consume = np.zeros((n_molecules, MAX_ATOMIC_NUMBER), dtype=np.float64)
        for i, mol in enumerate(molecules_all):
            for Z in mol.get_atoms():
                M_consume[i, int(Z) - 1] += 1.0
        mol_atom_consume = jnp.array(M_consume)
    else:
        mol_atoms_array = jnp.zeros((0, 6), dtype=jnp.int32)
        mol_charges = jnp.array([], dtype=jnp.int32)
        mol_n_atoms = jnp.array([], dtype=jnp.int32)
        mol_log_K_values = jnp.zeros((0, n_temps))
        mol_partition_func_values = jnp.zeros((0, n_temps))
        mol_partition_func_z = jnp.zeros((0, n_temps))
        mol_atom_consume = jnp.zeros((0, MAX_ATOMIC_NUMBER))

    return ChemicalEquilibriumData(
        log_T_grid=log_T_grid,
        ionization_energies=jnp.array(ion_energies),
        partition_func_values=jnp.array(pf_values),
        pf_orig_t=jnp.array(pf_orig_t),
        pf_orig_u=jnp.array(pf_orig_u),
        pf_orig_h=jnp.array(pf_orig_h),
        pf_orig_z=jnp.array(pf_orig_z),
        pf_orig_n=jnp.array(pf_orig_n, dtype=jnp.int32),
        log_T_h=jnp.array(log_T_h),
        n_molecules=n_molecules,
        mol_atoms_array=mol_atoms_array,
        mol_charges=mol_charges,
        mol_n_atoms=mol_n_atoms,
        mol_log_K_values=mol_log_K_values,
        mol_partition_func_values=mol_partition_func_values,
        mol_partition_func_z=mol_partition_func_z,
        mol_atom_consume=mol_atom_consume
    )


def _interp_partition_func(log_T, Z, charge, data):
    """Interpolate partition function value at given log(T)."""
    return jnp.interp(log_T, data.log_T_grid, data.partition_func_values[Z, charge])


@jax.jit
def _compute_saha_weights_jit(T, ne, data):
    """
    Compute Saha ionization weights for all elements (JIT-compatible).

    Returns wII and wIII arrays where:
    - wII[Z-1] = n(Z II) / n(Z I)
    - wIII[Z-1] = n(Z III) / n(Z I)
    """
    log_T = jnp.log(T)
    transU = translational_U(electron_mass_cgs, T)
    ne_clipped = jnp.clip(ne, 1e-12, jnp.inf)

    def compute_weights(Z_minus_1):
        Z = Z_minus_1  # 0-indexed
        χI = data.ionization_energies[Z, 0]
        χII = data.ionization_energies[Z, 1]

        UI = jnp.interp(log_T, data.log_T_grid, data.partition_func_values[Z, 0])
        UII = jnp.interp(log_T, data.log_T_grid, data.partition_func_values[Z, 1])
        UIII = jnp.interp(log_T, data.log_T_grid, data.partition_func_values[Z, 2])

        # Saha equation for first ionization
        wII = 2.0 / ne_clipped * (UII / jnp.clip(UI, 1e-99, jnp.inf)) * transU * jnp.exp(-χI / (kboltz_eV * T))

        # Second ionization
        wIII = wII * 2.0 / ne_clipped * (UIII / jnp.clip(UII, 1e-99, jnp.inf)) * transU * jnp.exp(-χII / (kboltz_eV * T))

        # Handle hydrogen (Z=1, cannot be doubly ionized)
        wIII = jnp.where(Z_minus_1 == 0, 0.0, wIII)

        return wII, wIII

    Z_indices = jnp.arange(MAX_ATOMIC_NUMBER)
    wII_array, wIII_array = jax.vmap(compute_weights)(Z_indices)

    return wII_array, wIII_array


def _get_log_nK_jit(mol_idx, log_T, data):
    """Get log equilibrium constant in number density form (JIT-compatible)."""
    # Interpolate log K from precomputed values
    log_K_p = jnp.interp(log_T, data.log_T_grid, data.mol_log_K_values[mol_idx])

    # Number of atoms
    n_atoms = data.mol_n_atoms[mol_idx]

    # Convert from partial pressure to number density form
    T = jnp.exp(log_T)
    log_nK = log_K_p - (n_atoms - 1) * jnp.log10(kboltz_cgs * T)

    return log_nK


def _compute_residuals_jit(x, n_total, abund_array, data):
    """
    Compute chemical equilibrium residuals (fully JIT-compatible).
    """
    log_T = jnp.log(jnp.clip(jnp.abs(x[0]) if x.shape[0] > MAX_ATOMIC_NUMBER + 1 else 5777.0, 100, 1e6))

    # Extract electron density (scaled for numerical stability)
    ne = jnp.clip(jnp.abs(x[-1]) * n_total * 1e-5, 1e-12, jnp.inf)

    # Extract neutral fractions
    neutral_fractions = jnp.abs(x[:MAX_ATOMIC_NUMBER])

    # Total atom number densities
    atom_number_densities = abund_array * (n_total - ne)

    # Neutral atomic number densities
    neutral_number_densities = atom_number_densities * neutral_fractions

    # Compute Saha weights
    T = jnp.exp(log_T)
    wII, wIII = _compute_saha_weights_jit(T, ne, data)

    # Element conservation residuals (vectorized)
    F_elements = atom_number_densities - (1.0 + wII + wIII) * neutral_number_densities

    # Electron conservation
    F_electron = jnp.sum((wII + 2.0 * wIII) * neutral_number_densities) - ne

    F = jnp.concatenate([F_elements, jnp.array([F_electron])])

    # Molecular contributions using scan
    log_neutral_densities = jnp.log10(jnp.clip(neutral_number_densities, 1e-99, jnp.inf))

    def process_molecule(F, mol_idx):
        atoms = data.mol_atoms_array[mol_idx]
        charge = data.mol_charges[mol_idx]
        n_atoms = data.mol_n_atoms[mol_idx]
        log_nK = _get_log_nK_jit(mol_idx, log_T, data)

        # Skip if log_nK is not finite
        valid = jnp.isfinite(log_nK)

        def neutral_contrib(F):
            log_sum = jnp.sum(jnp.where(
                jnp.arange(6) < n_atoms,
                log_neutral_densities[atoms],
                0.0
            ))
            n_mol = 10.0 ** jnp.clip(log_sum - log_nK, -300, 300)

            updates = jnp.where(jnp.arange(6) < n_atoms, -n_mol, 0.0)
            F_new = F.at[atoms[0]].add(jnp.where(n_atoms > 0, updates[0], 0.0))
            F_new = F_new.at[atoms[1]].add(jnp.where(n_atoms > 1, updates[1], 0.0))
            F_new = F_new.at[atoms[2]].add(jnp.where(n_atoms > 2, updates[2], 0.0))
            F_new = F_new.at[atoms[3]].add(jnp.where(n_atoms > 3, updates[3], 0.0))
            F_new = F_new.at[atoms[4]].add(jnp.where(n_atoms > 4, updates[4], 0.0))
            F_new = F_new.at[atoms[5]].add(jnp.where(n_atoms > 5, updates[5], 0.0))
            return F_new

        def ionized_contrib(F):
            idx1, idx2 = atoms[0], atoms[1]
            wII_atom = wII[idx1]
            n1_II_log = log_neutral_densities[idx1] + jnp.log10(jnp.clip(wII_atom, 1e-99, jnp.inf))
            n2_I_log = log_neutral_densities[idx2]
            n_mol = 10.0 ** jnp.clip(n1_II_log + n2_I_log - log_nK, -300, 300)

            F_new = F.at[idx1].add(-n_mol)
            F_new = F_new.at[idx2].add(-n_mol)
            F_new = F_new.at[-1].add(n_mol)
            return F_new

        F_updated = jax.lax.cond(
            valid,
            lambda F: jax.lax.cond(charge == 0, neutral_contrib, ionized_contrib, F),
            lambda F: F,
            F
        )
        return F_updated, None

    # Process molecules - use array shape (which is known at trace time)
    # since data.mol_charges has shape (n_molecules,)
    n_mols = data.mol_charges.shape[0]
    if n_mols > 0:
        F, _ = jax.lax.scan(process_molecule, F, jnp.arange(n_mols))

    # Normalize residuals
    F = F.at[:MAX_ATOMIC_NUMBER].set(
        jnp.where(atom_number_densities > 0,
                 F[:MAX_ATOMIC_NUMBER] / jnp.clip(atom_number_densities, 1e-99, jnp.inf),
                 0.0))
    F = F.at[-1].set(F[-1] / jnp.clip(ne * 1e-5, 1e-12, jnp.inf))

    return F


@jax.jit
def chemical_equilibrium_jit(T, n_total, ne_model, absolute_abundances, data):
    """
    Solve for chemical equilibrium (fully JIT-compatible version).

    Uses a Picard (fixed-point) iteration on the electron density instead of
    Newton's method, avoiding the need for jacfwd inside while_loop which is
    expensive to compile for high-dimensional systems.

    Parameters
    ----------
    T : float
        Temperature in K
    n_total : float
        Total number density in cm⁻³
    ne_model : float
        Model atmosphere electron number density (initial guess)
    absolute_abundances : jax array
        Absolute abundances N(X)/N_total, shape (92,), indexed 0=H, 1=He, ...
    data : ChemicalEquilibriumData
        Pre-computed data from precompute_chemical_equilibrium_data()

    Returns
    -------
    tuple
        (ne, neutral_fractions) where:
        - ne: Calculated electron number density in cm⁻³
        - neutral_fractions: Array of neutral fractions for each element
    """
    # Precompute Saha weights at ne=1 (scale as wII(ne) = wII_ne1/ne)
    wII_ne1, wIII_ne1 = _compute_saha_weights_jit(T, 1.0, data)

    def electron_sum_from_ne(ne):
        """Compute total electron count via Saha equation given ne."""
        wII = wII_ne1 / jnp.clip(ne, 1e-12, jnp.inf)
        wIII = wIII_ne1 / jnp.clip(ne * ne, 1e-24, jnp.inf)
        atom_densities = absolute_abundances * (n_total - ne)
        neutral_fracs = 1.0 / (1.0 + wII + wIII)
        neutral_densities = atom_densities * neutral_fracs
        return jnp.sum((wII + 2.0 * wIII) * neutral_densities)

    def body(state):
        ne, i = state
        ne_new = electron_sum_from_ne(ne)
        ne_new = jnp.clip(ne_new, 1.0, n_total)
        # Geometric-mean damping in log space (0.3/0.7 split) prevents oscillation
        log_ne_new = 0.3 * jnp.log(jnp.clip(ne, 1e-99, jnp.inf)) + 0.7 * jnp.log(jnp.clip(ne_new, 1e-99, jnp.inf))
        return jnp.exp(log_ne_new), i + 1

    def cond(state):
        _, i = state
        return i < 300

    ne_sol, _ = jax.lax.while_loop(
        cond, body, (jnp.asarray(ne_model, dtype=jnp.float64), jnp.int32(0))
    )

    wII_sol = wII_ne1 / jnp.clip(ne_sol, 1e-12, jnp.inf)
    wIII_sol = wIII_ne1 / jnp.clip(ne_sol * ne_sol, 1e-24, jnp.inf)
    neutral_fractions = 1.0 / (1.0 + wII_sol + wIII_sol)

    return ne_sol, neutral_fractions


@jax.jit
def _compute_mol_densities_jit(T, n_total, ne, absolute_abundances, neutral_fractions, data):
    """
    Compute molecular number densities given solved atomic state (JIT-compatible).

    Called after chemical_equilibrium_jit as a post-processing step.
    Returns array of shape (n_molecules,).
    """
    atom_densities = absolute_abundances * (n_total - ne)
    neutral_densities = atom_densities * neutral_fractions
    log_neutral_dens = jnp.log10(jnp.clip(neutral_densities, 1e-99, jnp.inf))

    log_T = jnp.log(T)
    # Need wII for ionized molecules (charge=1 diatomics)
    wII_ne1, _ = _compute_saha_weights_jit(T, 1.0, data)
    wII = wII_ne1 / jnp.clip(ne, 1e-12, jnp.inf)

    def compute_one_mol(mol_idx):
        atoms = data.mol_atoms_array[mol_idx]   # (6,), padded with -1
        n_atoms = data.mol_n_atoms[mol_idx]
        charge = data.mol_charges[mol_idx]
        log_nK = _get_log_nK_jit(mol_idx, log_T, data)
        valid = jnp.isfinite(log_nK)

        safe_atoms = jnp.where(atoms >= 0, atoms, 0)  # clamp -1 pads to 0

        def neutral_mol(_):
            log_sum = jnp.sum(jnp.where(
                jnp.arange(6) < n_atoms,
                log_neutral_dens[safe_atoms],
                0.0
            ))
            return 10.0 ** jnp.clip(log_sum - log_nK, -300, 300)

        def ionized_mol(_):
            idx1, idx2 = safe_atoms[0], safe_atoms[1]
            n1_II_log = log_neutral_dens[idx1] + jnp.log10(jnp.clip(wII[idx1], 1e-99, jnp.inf))
            n2_I_log = log_neutral_dens[idx2]
            return 10.0 ** jnp.clip(n1_II_log + n2_I_log - log_nK, -300, 300)

        result = jax.lax.cond(
            valid,
            lambda _: jax.lax.cond(charge == 0, neutral_mol, ionized_mol, None),
            lambda _: 0.0,
            None
        )
        return result

    n_mols = data.mol_charges.shape[0]
    if n_mols == 0:
        return jnp.zeros(0)
    return jax.vmap(compute_one_mol)(jnp.arange(n_mols))


def chemical_equilibrium_fast(T, n_total, ne_model, absolute_abundances, data, mol_species):
    """
    Fast chemical equilibrium using Picard iteration + molecular post-processing.

    Replaces the Newton-based chemical_equilibrium() with a ~10,000x faster
    approach: Picard iteration for electron density (JIT-compiled), then
    molecular densities computed as post-processing (no feedback on ne, since
    stellar photosphere molecules are nearly all neutral).

    Parameters
    ----------
    T : float
        Temperature in K
    n_total : float
        Total number density in cm⁻³
    ne_model : float
        Initial electron density guess in cm⁻³
    absolute_abundances : array, shape (92,)
        Absolute abundances N(X)/N_total
    data : ChemicalEquilibriumData
        Pre-computed data from precompute_chemical_equilibrium_data()
    mol_species : list of Species
        Ordered list of molecular species matching data.mol_* arrays

    Returns
    -------
    tuple
        (ne, number_densities) — same format as chemical_equilibrium()
    """
    from .species import Species, Formula

    abs_abund_jax = jnp.asarray(absolute_abundances, dtype=jnp.float64)

    # Picard iteration: fast JIT-compiled electron density solve
    ne_sol, neutral_fracs = chemical_equilibrium_jit(
        T, n_total, ne_model, abs_abund_jax, data
    )
    ne = float(ne_sol)

    # Molecular post-processing (JIT-compiled vmap over all molecules)
    mol_dens_arr = _compute_mol_densities_jit(
        T, n_total, ne_sol, abs_abund_jax, neutral_fracs, data
    )

    # Compute raw numpy arrays (used by continuum batch without Species dict overhead)
    atom_densities = float(n_total - ne) * np.asarray(abs_abund_jax)
    neutral_fracs_np = np.asarray(neutral_fracs)
    neutral_dens = atom_densities * neutral_fracs_np  # shape (92,)

    wII_arr, wIII_arr = _compute_saha_weights_jit(
        jnp.asarray(T, dtype=jnp.float64),
        jnp.asarray(ne, dtype=jnp.float64),
        data
    )
    wII_np = np.asarray(wII_arr)
    wIII_np = np.asarray(wIII_arr)
    ionized_dens = wII_np * neutral_dens    # shape (92,)
    doubly_ionized_dens = wIII_np * neutral_dens  # shape (92,)
    mol_dens_np = np.asarray(mol_dens_arr)

    # Build number_densities dict (needed for line absorption)
    number_densities = {}
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        formula = Formula(int(Z))
        number_densities[Species(formula, 0)] = float(neutral_dens[Z-1])
        number_densities[Species(formula, 1)] = float(ionized_dens[Z-1])
        number_densities[Species(formula, 2)] = float(doubly_ionized_dens[Z-1])
    for i, mol in enumerate(mol_species):
        number_densities[mol] = float(mol_dens_np[i])

    raw_arrays = {
        'neutral_dens': neutral_dens,
        'ionized_dens': ionized_dens,
        'doubly_ionized_dens': doubly_ionized_dens,
        'mol_dens': mol_dens_np,
    }
    return ne, number_densities, raw_arrays


# Vmapped batch version — processes all atmosphere layers in one XLA call
_chemical_equilibrium_batch_jit = jax.jit(
    jax.vmap(chemical_equilibrium_jit, in_axes=(0, 0, 0, None, None))
)
_compute_mol_densities_batch_jit = jax.jit(
    jax.vmap(_compute_mol_densities_jit, in_axes=(0, 0, 0, None, 0, None))
)
_compute_saha_weights_batch_jit = jax.jit(
    jax.vmap(_compute_saha_weights_jit, in_axes=(0, 0, None))
)


def chemical_equilibrium_all_layers(T_arr, n_total_arr, ne_model_arr,
                                     absolute_abundances, data, mol_species):
    """
    Process all atmosphere layers at once using vmapped JAX.

    Much faster than calling chemical_equilibrium_fast 56 times because:
    - Single XLA dispatch instead of 56
    - XLA can vectorize operations across layers
    - Avoids 56× dict-building overhead

    Returns
    -------
    tuple: (electron_densities, number_densities, raw_arrays_list)
        - electron_densities: (n_layers,) numpy array
        - number_densities: dict mapping Species → (n_layers,) arrays
        - raw_arrays_list: list of per-layer raw array dicts
    """
    from .species import Species, Formula
    abs_abund_jax = jnp.asarray(absolute_abundances, dtype=jnp.float64)
    T_jax = jnp.asarray(T_arr, dtype=jnp.float64)
    n_total_jax = jnp.asarray(n_total_arr, dtype=jnp.float64)
    ne_model_jax = jnp.asarray(ne_model_arr, dtype=jnp.float64)

    # Batch Picard iteration — all layers at once
    ne_sol_all, neutral_fracs_all = _chemical_equilibrium_batch_jit(
        T_jax, n_total_jax, ne_model_jax, abs_abund_jax, data
    )

    # Batch molecular densities (first pass — uses Picard neutral fracs which ignore mol. depletion)
    mol_dens_all = _compute_mol_densities_batch_jit(
        T_jax, n_total_jax, ne_sol_all, abs_abund_jax, neutral_fracs_all, data
    )

    # Batch Saha weights for ionized/doubly ionized densities
    wII_all, wIII_all = _compute_saha_weights_batch_jit(T_jax, ne_sol_all, data)

    # Convert to numpy (single sync point for all layers)
    ne_np = np.asarray(ne_sol_all)               # (n_layers,)
    nf_np = np.asarray(neutral_fracs_all)         # (n_layers, 92)
    wII_np = np.asarray(wII_all)                  # (n_layers, 92)
    wIII_np = np.asarray(wIII_all)                # (n_layers, 92)
    mol_np = np.asarray(mol_dens_all)             # (n_layers, n_mols)

    n_layers = len(ne_np)
    atom_dens = (n_total_arr[:, None] - ne_np[:, None]) * absolute_abundances[None, :]
    neutral_dens_picard = atom_dens * nf_np       # (n_layers, 92) — before molecular correction

    # Iterative molecular depletion correction:
    # The Picard iteration for ne doesn't account for atoms bound in molecules,
    # so neutral_dens_picard overestimates free atomic densities (e.g. n_CI includes CO/C2/CN).
    # We correct by subtracting molecular atom consumption and recomputing mol_dens (3 passes).
    if data.n_molecules > 0:
        mol_atoms_np = np.asarray(data.mol_atoms_array)  # (n_mols, 6), Z-1 indexed, -1 = pad
        mol_n_atoms_np = np.asarray(data.mol_n_atoms)    # (n_mols,)
        # M_consume[i, Z-1] = number of Z-atoms in molecule i
        M_consume = np.zeros((data.n_molecules, 92), dtype=np.float64)
        for i in range(data.n_molecules):
            for k in range(int(mol_n_atoms_np[i])):
                Z_idx = int(mol_atoms_np[i, k])
                if 0 <= Z_idx < 92:
                    M_consume[i, Z_idx] += 1.0

        for _ in range(5):
            # mol_atom_correction[layer, Z-1] = total atoms of element Z in all molecules
            mol_atom_correction = mol_np @ M_consume   # (n_layers, 92)
            neutral_dens_corrected = np.maximum(neutral_dens_picard - mol_atom_correction, 1e-99)
            nf_corrected = neutral_dens_corrected / np.maximum(atom_dens, 1e-99)
            mol_dens_all = _compute_mol_densities_batch_jit(
                T_jax, n_total_jax, ne_sol_all, abs_abund_jax,
                jnp.asarray(nf_corrected), data
            )
            mol_np = np.asarray(mol_dens_all)
    else:
        neutral_dens_corrected = neutral_dens_picard

    neutral_dens_all = neutral_dens_corrected     # (n_layers, 92)
    ionized_dens_all = wII_np * neutral_dens_all
    doubly_dens_all = wIII_np * neutral_dens_all

    # Build number_densities dict — only once, not per layer
    number_densities = {}
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        f = Formula(int(Z))
        number_densities[Species(f, 0)] = neutral_dens_all[:, Z-1]
        number_densities[Species(f, 1)] = ionized_dens_all[:, Z-1]
        number_densities[Species(f, 2)] = doubly_dens_all[:, Z-1]
    for i, mol in enumerate(mol_species):
        number_densities[mol] = mol_np[:, i]

    # Build raw_arrays_list for continuum batch
    raw_arrays_list = [
        {
            'neutral_dens': neutral_dens_all[i],
            'ionized_dens': ionized_dens_all[i],
            'doubly_ionized_dens': doubly_dens_all[i],
            'mol_dens': mol_np[i],
        }
        for i in range(n_layers)
    ]

    return ne_np, number_densities, raw_arrays_list


def chemical_equilibrium(T, n_total, ne_model, absolute_abundances,
                        ionization_energies, partition_funcs,
                        log_equilibrium_constants,
                        electron_density_warn_threshold=0.1,
                        electron_density_warn_min_value=1e-4):
    """
    Solve for chemical equilibrium number densities.

    Iteratively solves the system of nonlinear equations that defines
    chemical equilibrium, accounting for ionization (Saha equation) and
    molecular dissociation (equilibrium constants).

    Parameters
    ----------
    T : float
        Temperature in K
    n_total : float
        Total number density in cm⁻³
    ne_model : float
        Model atmosphere electron number density in cm⁻³ (used as initial guess)
    absolute_abundances : dict or array
        Absolute abundances N(X)/N_total for each element
    ionization_energies : dict
        Dictionary mapping atomic numbers to [χ₁, χ₂, χ₃] in eV
    partition_funcs : dict
        Dictionary mapping Species to partition function callables
    log_equilibrium_constants : dict
        Dictionary mapping molecular Species to log₁₀(K) functions
    electron_density_warn_threshold : float, optional
        Warn if calculated ne differs from model by this fraction (default: 0.1)
    electron_density_warn_min_value : float, optional
        Minimum ne for warnings (default: 1e-4)

    Returns
    -------
    tuple
        (ne, number_densities) where:
        - ne: Calculated electron number density in cm⁻³
        - number_densities: Dict mapping Species to number densities

    Notes
    -----
    This function:
    1. Computes an initial guess by neglecting molecules
    2. Solves the nonlinear system using Newton's method with JAX autodiff
    3. Computes number densities for all species from the solution

    The system of equations enforces:
    - Conservation of each element (atoms + ions + molecules)
    - Electron conservation

    Reference
    ---------
    Kurucz 1970, sections 5.1-5.3
    Gray 2005, "The Observation and Analysis of Stellar Photospheres", Ch. 8
    """
    from .species import Species, Formula

    # Convert abundances to array if needed
    if isinstance(absolute_abundances, dict):
        abund_array = jnp.zeros(MAX_ATOMIC_NUMBER)
        for Z, abund in absolute_abundances.items():
            abund_array = abund_array.at[Z-1].set(abund)
    else:
        abund_array = jnp.asarray(absolute_abundances)

    # Compute initial guess by neglecting molecules
    neutral_fraction_guess = []
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        wII, wIII = saha_ion_weights(T, ne_model, Z, ionization_energies,
                                     partition_funcs)
        neutral_frac = 1.0 / (1.0 + wII + wIII)
        neutral_fraction_guess.append(float(neutral_frac))

    # Initial state vector: [neutral_fractions, ne_scaled]
    x0 = jnp.array(neutral_fraction_guess + [ne_model / (n_total * 1e-5)])

    # Set up residual function
    residuals_func = setup_chemical_equilibrium_residuals(
        T, n_total, abund_array, ionization_energies,
        partition_funcs, log_equilibrium_constants
    )


    # Solve using JAX-based Newton's method (like Julia's NLsolve with
    # method=:newton, ftol=1e-8, iterations=1000; convergence is via ∞-norm).
    ftol = 1e-8
    try:
        x_solution, converged, residual_norm, iterations = newton_solve_jax(
            residuals_func, x0, ftol=ftol, max_iter=1000
        )

        if not converged:
            # Try again with very small ne guess (like Julia does)
            x0_retry = x0.at[-1].set(1e-5)
            x_solution, converged, residual_norm, iterations = newton_solve_jax(
                residuals_func, x0_retry, ftol=ftol, max_iter=1000
            )

        if not converged:
            raise RuntimeError(
                f"Chemical equilibrium solver failed to converge after {iterations} iterations. "
                f"Final residual norm: {residual_norm:.3e}"
            )

    except Exception as e:
        raise RuntimeError(f"Chemical equilibrium solver failed: {str(e)}")

    # Extract solution (convert from JAX to Python floats for output)
    ne = float(jnp.abs(x_solution[-1]) * n_total * 1e-5)
    neutral_fractions = jnp.abs(x_solution[:MAX_ATOMIC_NUMBER])

    # Warn if electron density differs significantly from model
    if (ne / n_total > electron_density_warn_min_value and
        abs((ne - ne_model) / ne_model) > electron_density_warn_threshold):
        import warnings
        warnings.warn(
            f"Electron number density differs from model atmosphere by "
            f"{abs((ne - ne_model) / ne_model):.1%}. "
            f"(calculated ne = {ne:.3e}, model ne = {ne_model:.3e})"
        )

    # Build number densities dictionary
    number_densities = {}

    # Neutral atomic species
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        formula = Formula(int(Z))
        n_neutral = float((n_total - ne) * abund_array[Z-1] * neutral_fractions[Z-1])
        number_densities[Species(formula, 0)] = n_neutral

    # Ionized atomic species
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        formula = Formula(int(Z))
        wII, wIII = saha_ion_weights(T, ne, Z, ionization_energies, partition_funcs)

        n_neutral = number_densities[Species(formula, 0)]
        number_densities[Species(formula, 1)] = float(wII * n_neutral)
        number_densities[Species(formula, 2)] = float(wIII * n_neutral)

    # Molecular species
    log_neutral_densities = {Z: float(jnp.log10(number_densities[Species(Formula(int(Z)), 0)] + 1e-99))
                             for Z in range(1, MAX_ATOMIC_NUMBER + 1)}

    for mol in log_equilibrium_constants.keys():
        log_nK = get_log_nK(mol, T, log_equilibrium_constants)

        if mol.charge == 0:  # Neutral molecule
            Zs = mol.get_atoms()
            log_sum = sum(log_neutral_densities[int(Z)] for Z in Zs)
            number_densities[mol] = float(10.0 ** (log_sum - log_nK))

        else:  # Singly ionized diatomic
            Z1, Z2 = mol.get_atoms()
            n1_II = number_densities[Species(Formula(int(Z1)), 1)]
            n2_I = number_densities[Species(Formula(int(Z2)), 0)]
            number_densities[mol] = float(n1_II * n2_I / (10.0 ** log_nK))

    return ne, number_densities
