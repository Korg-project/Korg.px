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

    # Molecular data
    n_molecules: int
    mol_atoms_array: jnp.ndarray  # shape (n_molecules, 6), padded with -1
    mol_charges: jnp.ndarray  # shape (n_molecules,)
    mol_n_atoms: jnp.ndarray  # shape (n_molecules,)
    mol_log_K_values: jnp.ndarray  # shape (n_molecules, n_temps) - log K on T grid


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
        residual_norm = jnp.linalg.norm(F)

        # Check convergence
        converged = residual_norm < ftol

        # Compute Jacobian using forward-mode autodiff
        J = jax.jacfwd(residuals_func)(x)

        # Solve J * dx = -F for the Newton step
        # Use lstsq for robustness against singular Jacobians
        dx = jnp.linalg.lstsq(J, -F, rcond=None)[0]

        # Only update if not converged
        x_new = jnp.where(converged, x, x + dx)

        return (x_new, iteration + 1, converged, residual_norm)

    # Initial state
    F0 = residuals_func(x0)
    residual_norm0 = jnp.linalg.norm(F0)
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

    # Build partition function values on T grid
    # Shape: (92, 3, n_temps) for elements 1-92, charge states 0,1,2
    pf_values = np.zeros((MAX_ATOMIC_NUMBER, 3, n_temps))
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        formula = Formula(Z)
        for charge in range(3):
            species = Species(formula, charge)
            if species in partition_funcs:
                pf_func = partition_funcs[species]
                for i, log_T in enumerate(log_T_grid):
                    pf_values[Z-1, charge, i] = pf_func(float(log_T))

    # Process molecules
    molecules_all = list(log_equilibrium_constants.keys())
    mol_atoms_list = []
    mol_charges_list = []
    mol_n_atoms_list = []
    mol_log_K_list = []

    for mol in molecules_all:
        mol_charges_list.append(mol.charge)
        atoms = mol.get_atoms()
        mol_n_atoms_list.append(len(atoms))
        padded = list(atoms - 1) + [-1] * (6 - len(atoms))
        mol_atoms_list.append(padded)

        # Evaluate log K on T grid
        log_K_func = log_equilibrium_constants[mol]
        log_Ks = np.array([log_K_func(float(log_T)) for log_T in log_T_grid])
        mol_log_K_list.append(log_Ks)

    n_molecules = len(molecules_all)
    if n_molecules > 0:
        mol_atoms_array = jnp.array(mol_atoms_list, dtype=jnp.int32)
        mol_charges = jnp.array(mol_charges_list, dtype=jnp.int32)
        mol_n_atoms = jnp.array(mol_n_atoms_list, dtype=jnp.int32)
        mol_log_K_values = jnp.array(mol_log_K_list)
    else:
        mol_atoms_array = jnp.zeros((0, 6), dtype=jnp.int32)
        mol_charges = jnp.array([], dtype=jnp.int32)
        mol_n_atoms = jnp.array([], dtype=jnp.int32)
        mol_log_K_values = jnp.zeros((0, n_temps))

    return ChemicalEquilibriumData(
        log_T_grid=log_T_grid,
        ionization_energies=jnp.array(ion_energies),
        partition_func_values=jnp.array(pf_values),
        n_molecules=n_molecules,
        mol_atoms_array=mol_atoms_array,
        mol_charges=mol_charges,
        mol_n_atoms=mol_n_atoms,
        mol_log_K_values=mol_log_K_values
    )


def _interp_partition_func(log_T, Z, charge, data):
    """Interpolate partition function value at given log(T)."""
    return jnp.interp(log_T, data.log_T_grid, data.partition_func_values[Z, charge])


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

    This version uses pre-computed data for partition functions and
    equilibrium constants, allowing it to be called from within
    other JIT-compiled functions.

    Parameters
    ----------
    T : float
        Temperature in K
    n_total : float
        Total number density in cm⁻³
    ne_model : float
        Model atmosphere electron number density (initial guess)
    absolute_abundances : jax array
        Absolute abundances N(X)/N_total, shape (92,)
    data : ChemicalEquilibriumData
        Pre-computed data from precompute_chemical_equilibrium_data()

    Returns
    -------
    tuple
        (ne, neutral_fractions) where:
        - ne: Calculated electron number density in cm⁻³
        - neutral_fractions: Array of neutral fractions for each element
    """
    log_T = jnp.log(T)

    # Compute initial guess using Saha equation
    wII_init, wIII_init = _compute_saha_weights_jit(T, ne_model, data)
    neutral_fraction_guess = 1.0 / (1.0 + wII_init + wIII_init)

    # Initial state vector
    x0 = jnp.concatenate([neutral_fraction_guess, jnp.array([ne_model / (n_total * 1e-5)])])

    # Create residuals function with fixed parameters
    def residuals(x):
        # We need to pass T through the state vector for JIT compatibility
        # But for now, T is fixed from the outer scope (captured in closure)
        ne = jnp.clip(jnp.abs(x[-1]) * n_total * 1e-5, 1e-12, jnp.inf)
        neutral_fractions = jnp.abs(x[:MAX_ATOMIC_NUMBER])
        atom_number_densities = absolute_abundances * (n_total - ne)
        neutral_number_densities = atom_number_densities * neutral_fractions

        wII, wIII = _compute_saha_weights_jit(T, ne, data)

        F_elements = atom_number_densities - (1.0 + wII + wIII) * neutral_number_densities
        F_electron = jnp.sum((wII + 2.0 * wIII) * neutral_number_densities) - ne
        F = jnp.concatenate([F_elements, jnp.array([F_electron])])

        # Molecular contributions
        log_neutral_densities = jnp.log10(jnp.clip(neutral_number_densities, 1e-99, jnp.inf))

        def process_molecule(F, mol_idx):
            atoms = data.mol_atoms_array[mol_idx]
            charge = data.mol_charges[mol_idx]
            n_atoms = data.mol_n_atoms[mol_idx]
            log_nK = _get_log_nK_jit(mol_idx, log_T, data)

            valid = jnp.isfinite(log_nK)

            def neutral_contrib(F):
                log_sum = jnp.sum(jnp.where(jnp.arange(6) < n_atoms, log_neutral_densities[atoms], 0.0))
                n_mol = 10.0 ** jnp.clip(log_sum - log_nK, -300, 300)
                updates = jnp.where(jnp.arange(6) < n_atoms, -n_mol, 0.0)
                F_new = F
                for k in range(6):
                    F_new = F_new.at[atoms[k]].add(jnp.where(n_atoms > k, updates[k], 0.0))
                return F_new

            def ionized_contrib(F):
                idx1, idx2 = atoms[0], atoms[1]
                n1_II_log = log_neutral_densities[idx1] + jnp.log10(jnp.clip(wII[idx1], 1e-99, jnp.inf))
                n2_I_log = log_neutral_densities[idx2]
                n_mol = 10.0 ** jnp.clip(n1_II_log + n2_I_log - log_nK, -300, 300)
                return F.at[idx1].add(-n_mol).at[idx2].add(-n_mol).at[-1].add(n_mol)

            F_updated = jax.lax.cond(valid,
                lambda F: jax.lax.cond(charge == 0, neutral_contrib, ionized_contrib, F),
                lambda F: F, F)
            return F_updated, None

        # Process molecules - use array shape (known at trace time)
        n_mols = data.mol_charges.shape[0]
        if n_mols > 0:
            F, _ = jax.lax.scan(process_molecule, F, jnp.arange(n_mols))

        # Normalize
        F = F.at[:MAX_ATOMIC_NUMBER].set(
            jnp.where(atom_number_densities > 0,
                     F[:MAX_ATOMIC_NUMBER] / jnp.clip(atom_number_densities, 1e-99, jnp.inf), 0.0))
        F = F.at[-1].set(F[-1] / jnp.clip(ne * 1e-5, 1e-12, jnp.inf))
        return F

    # Solve using Newton's method
    x_solution, converged, residual_norm, iterations = newton_solve_jax(residuals, x0)

    # Extract solution
    ne = jnp.abs(x_solution[-1]) * n_total * 1e-5
    neutral_fractions = jnp.abs(x_solution[:MAX_ATOMIC_NUMBER])

    return ne, neutral_fractions


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


    # Solve using JAX-based Newton's method (like Julia's NLsolve)
    try:
        x_solution, converged, residual_norm, iterations = newton_solve_jax(
            residuals_func, x0, ftol=1e-8, max_iter=1000
        )

        if not converged:
            # Try again with very small ne guess (like Julia does)
            x0_retry = x0.at[-1].set(1e-5)
            x_solution, converged, residual_norm, iterations = newton_solve_jax(
                residuals_func, x0_retry, ftol=1e-8, max_iter=1000
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
