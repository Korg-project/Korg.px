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
    mol_log_K_z: jnp.ndarray       # shape (n_molecules, n_temps) - cubic spline z for log K
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
        # Straight port from HBOP - not default.
        #
        # ``jnp.where`` masks the *value* of the discarded branch but not its
        # cotangent, so anything non-finite computed in here leaks a NaN into the
        # gradient even when the guard selects 0.0.  The reachable hazards are
        # ``log(ne)`` at ne <= 0, ``1/sqrt(T)`` at T <= 0, ``BETAC**3`` overflowing
        # to inf for absurdly small ne (giving inf/inf == NaN), and ``log(F)`` at
        # F == 0 when ``BETAC**3`` underflows for absurdly large ne or n_eff.
        # Feeding the block *strictly positive* stand-ins whenever its result is
        # discarded removes all of them at once — clamping to zero would not, since
        # log(0) and 1/sqrt(0) are still infinite.  Where the result *is* used the
        # stand-ins are the real arguments, so no selected value changes.
        # This is the same fix already applied to the sibling implementation in
        # hydrogen_line_absorption.hummer_mihalas_w.
        hubeny_live = (ne > 10) & (T > 10)
        ne_h = jnp.where(hubeny_live, ne, 1e14)
        T_h = jnp.where(hubeny_live, T, 1e4)
        n_eff_h = jnp.where(hubeny_live, n_eff, 1.0)
        K_h = jnp.where(hubeny_live, K, 1.0)

        A = 0.09 * jnp.exp(0.16667 * jnp.log(ne_h)) / jnp.sqrt(T_h)
        X = jnp.exp(3.15 * jnp.log(1.0 + A))
        BETAC = 8.3e14 * jnp.exp(-0.66667 * jnp.log(ne_h)) * K_h / n_eff_h**4
        F = 0.1402 * X * BETAC**3 / (1.0 + 0.1285 * X * BETAC * jnp.sqrt(BETAC))
        hubeny_term = jnp.log(F / (1.0 + F)) / (-4.0 * jnp.pi / 3.0)

        charged_term = jnp.where(hubeny_live, hubeny_term, 0.0)
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


def Hminus_nK(T):
    """
    Equilibrium coefficient for H⁻ formation: n(H⁻) = nK(T) · n(H I) · nₑ.

    The reaction is H I + e⁻ → H⁻. The partition function of H⁻ is 1 (it has only a
    singlet ground state) and the statistical weight of the free electron is 2. Unlike
    the molecular equilibrium constants this is written in terms of number densities
    rather than partial pressures.

    Parameters
    ----------
    T : float
        Temperature in K

    Returns
    -------
    float
        nK in cm³

    Notes
    -----
    Introduced in Korg.jl v1.2, which promoted H⁻ to a species carried by
    ``reference_chemical_equilibrium`` (and participating in charge balance) rather than one
    derived inside the H⁻ bound-free opacity.
    """
    # Electron affinity used by the McLaughlin+ 2017 H⁻ ff cross sections
    chi_ea = 0.754204  # eV
    # inverse of translational_U for the electron, times U(H⁻)/U(H I) = 1/2,
    # times exp(χ_ea / kT)
    return jnp.exp(chi_ea / (kboltz_eV * T)) / (4.0 * translational_U(electron_mass_cgs, T))


# Maximum atomic number to consider
MAX_ATOMIC_NUMBER = 92


def _pow10(x):
    """10**x with the exponent clamped to keep the residuals finite.

    Julia lets these overflow to Inf and treats that as a failed Newton step. Clamping
    at ±300 keeps values inside float64 range while staying far enough from any
    physically meaningful density that a clamped residual is still enormous, so the
    convergence test rejects it just the same.
    """
    return 10.0 ** jnp.clip(x, -300.0, 300.0)


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
        Pre-computed data structure for use with picard_chemical_equilibrium_guess
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

    # Some (element, charge) combinations have no partition function at all — H III is
    # the only one, since hydrogen cannot be doubly ionized. A one-knot placeholder makes
    # the spline evaluation degenerate and return NaN. The value is masked downstream, but
    # a masked NaN still poisons reverse-mode gradients, so give those entries a real
    # two-knot table that evaluates to a constant 1 with zero derivative.
    for Z_idx in range(MAX_ATOMIC_NUMBER):
        for charge in range(3):
            if pf_orig_n[Z_idx, charge] >= 2:
                continue
            pf_orig_t[Z_idx, charge, :2] = [log_T_np[0], log_T_np[-1]]
            pf_orig_u[Z_idx, charge, :2] = [1.0, 1.0]
            pf_orig_h[Z_idx, charge, :2] = [0.0, log_T_np[-1] - log_T_np[0]]
            pf_orig_z[Z_idx, charge, :2] = [0.0, 0.0]
            pf_orig_n[Z_idx, charge] = 2

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
        mol_log_K_np = np.array(mol_log_K_list)
        mol_log_K_values = jnp.array(mol_log_K_np)
        mol_log_K_z_np = np.zeros_like(mol_log_K_np)
        for i in range(n_molecules):
            cs = _make_cubic_spline(log_T_np, mol_log_K_np[i], extrapolate=True)
            mol_log_K_z_np[i] = np.asarray(cs.z)
        mol_log_K_z = jnp.array(mol_log_K_z_np)
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
        mol_log_K_z = jnp.zeros((0, n_temps))
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
        mol_log_K_z=mol_log_K_z,
        mol_partition_func_values=mol_partition_func_values,
        mol_partition_func_z=mol_partition_func_z,
        mol_atom_consume=mol_atom_consume
    )


def _interp_partition_func(log_T, Z, charge, data):
    """Interpolate the partition function of element ``Z`` at ``log_T``.

    ``Z`` is the atomic number (1-based), so the row index is ``Z - 1``; every
    other consumer of ``partition_func_values`` uses that convention (see
    ``precompute_chemical_equilibrium_data``, which fills
    ``pf_values[Z-1, charge, :]``, and ``_compute_saha_weights_jit``, whose
    outputs are documented as ``wII[Z-1]``).  This function indexed ``[Z, charge]``
    and therefore returned the partition function of element ``Z + 1`` — U(Co I)
    when asked for U(Fe I).  It has no callers anywhere in the package, which is
    why nothing caught it.
    """
    return jnp.interp(log_T, data.log_T_grid, data.partition_func_values[Z - 1, charge])


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

        UI   = _eval_atomic_pf_jit(log_T, data.pf_orig_t[Z, 0], data.pf_orig_u[Z, 0],
                                    data.pf_orig_h[Z, 0], data.pf_orig_z[Z, 0], data.pf_orig_n[Z, 0])
        UII  = _eval_atomic_pf_jit(log_T, data.pf_orig_t[Z, 1], data.pf_orig_u[Z, 1],
                                    data.pf_orig_h[Z, 1], data.pf_orig_z[Z, 1], data.pf_orig_n[Z, 1])
        UIII = _eval_atomic_pf_jit(log_T, data.pf_orig_t[Z, 2], data.pf_orig_u[Z, 2],
                                    data.pf_orig_h[Z, 2], data.pf_orig_z[Z, 2], data.pf_orig_n[Z, 2])

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


def _eval_atomic_pf_jit(log_T, t_arr, u_arr, h_arr, z_arr, n_knots):
    """Evaluate atomic partition function via cubic spline (JIT-compatible).

    Matches Julia's CubicSpline evaluation exactly. Uses original non-uniform knots.
    """
    t_max = t_arr[n_knots - 1]
    log_T_c = jnp.clip(log_T, t_arr[0], t_max)
    i = jnp.clip(jnp.searchsorted(t_arr, log_T_c, side='right') - 1, 0, n_knots - 2)
    ti  = t_arr[i];   ti1 = t_arr[i + 1]
    ui  = u_arr[i];   ui1 = u_arr[i + 1]
    zi  = z_arr[i];   zi1 = z_arr[i + 1]
    hi1 = h_arr[i + 1]
    return (zi  * (ti1 - log_T_c)**3 / (6.0 * hi1)
            + zi1 * (log_T_c - ti )**3 / (6.0 * hi1)
            + (ui1 / hi1 - zi1 * hi1 / 6.0) * (log_T_c - ti )
            + (ui  / hi1 - zi  * hi1 / 6.0) * (ti1 - log_T_c))


def _cubic_spline_eval_logK(log_T, data, mol_idx):
    """Cubic spline interpolation for log K using precomputed z (second derivatives)."""
    t_grid = data.log_T_grid
    h_grid = data.log_T_h
    u_vals = data.mol_log_K_values[mol_idx]
    z_vals = data.mol_log_K_z[mol_idx]
    log_T_c = jnp.clip(log_T, t_grid[0], t_grid[-1])
    i = jnp.clip(jnp.searchsorted(t_grid, log_T_c, side='right') - 1, 0, t_grid.shape[0] - 2)
    ti  = t_grid[i];   ti1 = t_grid[i + 1]
    ui  = u_vals[i];   ui1 = u_vals[i + 1]
    zi  = z_vals[i];   zi1 = z_vals[i + 1]
    hi1 = h_grid[i + 1]
    return (zi  * (ti1 - log_T_c)**3 / (6.0 * hi1)
            + zi1 * (log_T_c - ti )**3 / (6.0 * hi1)
            + (ui1 / hi1 - zi1 * hi1 / 6.0) * (log_T_c - ti )
            + (ui  / hi1 - zi  * hi1 / 6.0) * (ti1 - log_T_c))


def _mol_log_nK_all_jit(T, data):
    """log₁₀ number-density equilibrium constants for every molecule at once.

    Same cubic spline as :func:`_cubic_spline_eval_logK`, evaluated for all molecules in
    one shot so the chemical equilibrium residual and Jacobian stay scan-free.
    """
    log_T = jnp.log(T)
    t_grid = data.log_T_grid
    log_T_c = jnp.clip(log_T, t_grid[0], t_grid[-1])
    i = jnp.clip(jnp.searchsorted(t_grid, log_T_c, side='right') - 1, 0, t_grid.shape[0] - 2)
    ti, ti1 = t_grid[i], t_grid[i + 1]
    hi1 = data.log_T_h[i + 1]
    u_i, u_i1 = data.mol_log_K_values[:, i], data.mol_log_K_values[:, i + 1]
    z_i, z_i1 = data.mol_log_K_z[:, i], data.mol_log_K_z[:, i + 1]
    log_K_p = (z_i * (ti1 - log_T_c) ** 3 / (6.0 * hi1)
               + z_i1 * (log_T_c - ti) ** 3 / (6.0 * hi1)
               + (u_i1 / hi1 - z_i1 * hi1 / 6.0) * (log_T_c - ti)
               + (u_i / hi1 - z_i * hi1 / 6.0) * (ti1 - log_T_c))
    return log_K_p - (data.mol_n_atoms - 1.0) * jnp.log10(kboltz_cgs * T)


def _get_log_nK_jit(mol_idx, log_T, data):
    """Get log equilibrium constant in number density form (JIT-compatible)."""
    # Cubic spline interpolation for log K (matches Julia's CubicSpline exactly)
    log_K_p = _cubic_spline_eval_logK(log_T, data, mol_idx)

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
def picard_chemical_equilibrium_guess(T, n_total, ne_model, absolute_abundances, data):
    """
    Cheap Picard estimate of chemical equilibrium, used to seed the real solve.

    This is an *initialiser*, not the answer. It runs a Picard (fixed-point) iteration
    on the electron density, which is cheap to compile and converges from a poor
    starting point, but it neglects the depletion of atoms into molecules. Its output
    is fed to :func:`_chem_eq_newton_layer_jit`, which solves the full Korg.jl v1.2
    system to convergence.

    Do not use this on its own for physics. It is not the solver, and it is
    reverse-mode-differentiable only through :func:`_chem_eq_newton_layer_jit`'s
    implicit-function-theorem rule, which treats this guess as having no tangent.

    See Also
    --------
    _chem_eq_newton_layer_jit : the solver this seeds
    reference_chemical_equilibrium : slow, exact oracle used to verify both

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

    Called after picard_chemical_equilibrium_guess as a post-processing step.
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

    Replaces the Newton-based reference_chemical_equilibrium() with a ~10,000x faster
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
        (ne, number_densities) — same format as reference_chemical_equilibrium()
    """
    from .species import Species, Formula

    abs_abund_jax = jnp.asarray(absolute_abundances, dtype=jnp.float64)

    # Picard iteration: fast JIT-compiled electron density solve
    ne_sol, neutral_fracs = picard_chemical_equilibrium_guess(
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
_picard_chemical_equilibrium_guess_batch = jax.jit(
    jax.vmap(picard_chemical_equilibrium_guess, in_axes=(0, 0, 0, None, None))
)
_compute_mol_densities_batch_jit = jax.jit(
    jax.vmap(_compute_mol_densities_jit, in_axes=(0, 0, 0, None, 0, None))
)
_compute_saha_weights_batch_jit = jax.jit(
    jax.vmap(_compute_saha_weights_jit, in_axes=(0, 0, None))
)


# ── Log-space chemical equilibrium for the batched/JIT path (Korg.jl v1.2) ────
#
# The system, unknowns and scaling are identical to statmech.reference_chemical_equilibrium;
# what differs is that the Jacobian is written out by hand rather than obtained from
# jax.jacfwd, and the continuation schedule is fixed rather than adaptive so that the
# whole solve is one jittable, vmappable, reverse-mode-differentiable kernel.
#
# Unknowns  y = [log10 n(X I) for Z = 1..92, log10 nₑ].
# Writing L = ln 10 and using
#     n_I[i]    = 10^(y_i)
#     n_II[i]   = 10^(y_i + a_i - y_e)          a_i = log10 wII  at nₑ = 1
#     n_III[i]  = 10^(y_i + b_i - 2 y_e)        b_i = log10 wIII at nₑ = 1
#     n_mol[m]  = 10^(Σ_i C[m,i] y_i - logK[m] + logξ + s_m (a_f(m) - y_e))
#     n_H⁻      = 10^(logK_H⁻ + y_0 + y_e + logξ)
#     n_nuclei  = nₜ - nₑ + Σ_m (k_m - 1) n_mol[m]
# with C[m,i] the number of atoms of element i in molecule m, k_m the total atom count,
# s_m = 1 for the singly ionized diatomics and 0 otherwise, and f(m) the charged atom,
# the unscaled residuals are
#     G_i = Σ_m C[m,i] n_mol[m] + δ_i0 n_H⁻ + n_I[i] + n_II[i] + n_III[i] - A_i n_nuclei
#     G_e = Σ_m q_m n_mol[m] - n_H⁻ + Σ_i (n_II[i] + 2 n_III[i]) - nₑ
# and F_i = G_i / (A_i nₜ), F_e = G_e / nₜ.
#
# Every unknown appears only through 10^(linear combination), so each derivative is L
# times the term itself with an integer coefficient. Because the residual scaling is a
# constant, the Jacobian of F is just the Jacobian of G divided by the same constants —
# no extra normalisation-derivative term is needed.

_LN10 = 2.302585092994046

# Continuation schedule for the batched solve. Under vmap a data-dependent number of
# annealing steps would have to be materialised for every layer anyway, so we walk a
# fixed schedule for all layers: dense near logξ = 0, where molecules actually bite.
_XI_SCHEDULE = (-50.0, -12.0, -8.0, -5.0, -3.5, -2.5, -1.75, -1.25,
                -0.9, -0.6, -0.4, -0.25, -0.12, 0.0)
_XI_INNER_ITERS = 8
# Newton steps at logξ = 0 after the schedule, to polish to full convergence.
_XI_FINAL_ITERS = 24


def _chem_eq_log_terms(y, T, n_total, abundances, data, log_xi):
    """Species densities and the nucleus budget at state ``y``.

    Shared by the residual and Jacobian so the exponentials are formed once.
    """
    log_ne = y[MAX_ATOMIC_NUMBER]
    log_n_I = y[:MAX_ATOMIC_NUMBER]

    wII_ne1, wIII_ne1 = _compute_saha_weights_jit(T, 1.0, data)
    # Hydrogen has no doubly ionized state, so wIII[0] is exactly zero and log10 of it is
    # -Inf with a 1/0 = Inf derivative. The 1e-320 floor this used to carry never took
    # effect: it is subnormal, and XLA flushes subnormals to zero, so the maximum was
    # max(0, 0). The -Inf is harmless in the forward pass (_pow10 clips it back to a
    # 1e-300 density) but its derivative met a zero tangent, and 0 x Inf is NaN --- which
    # is why forward mode through the solver returned NaN in row 0 (hydrogen's nucleus
    # balance) and row 92 (charge balance), the only two rows n_III[0] enters.
    # The inner `where` keeps a strictly positive value out of log10 so no infinite
    # derivative is formed at all; the outer one restores the -Inf, so the primal is
    # bitwise what it was.
    def _safe_log10(w):
        positive = w > 0.0
        return jnp.where(positive, jnp.log10(jnp.where(positive, w, 1.0)), -jnp.inf)

    log_wII = _safe_log10(wII_ne1)
    log_wIII = _safe_log10(wIII_ne1)

    n_I = _pow10(log_n_I)
    n_II = _pow10(log_n_I + log_wII - log_ne)
    n_III = _pow10(log_n_I + log_wIII - 2.0 * log_ne)
    ne = _pow10(log_ne)

    # Molecules. mol_atom_consume is the C matrix: atoms of each element per molecule.
    C = data.mol_atom_consume
    log_nK = _mol_log_nK_all_jit(T, data)
    is_charged = (data.mol_charges != 0).astype(jnp.float64)
    first_atom = jnp.maximum(data.mol_atoms_array[:, 0], 0)

    log_nK_finite = jnp.isfinite(log_nK)
    # Substitute a finite exponent *before* _pow10 for the molecules we are about to
    # discard.  _pow10 clips its argument, but jnp.clip propagates NaN, so 10**NaN is
    # NaN and the outer jnp.where would mask that value while still pushing a NaN
    # cotangent back through the selected branch.  0.0 is a safe exponent here (it is
    # an *exponent*, not a density: 10**0 == 1), so no log/sqrt singularity is created.
    log_nK_safe = jnp.where(log_nK_finite, log_nK, 0.0)
    log_n_mol = (C @ log_n_I - log_nK_safe + log_xi
                 + is_charged * (log_wII[first_atom] - log_ne))
    # Molecules whose equilibrium constant is undefined at this T contribute nothing.
    n_mol = jnp.where(log_nK_finite, _pow10(log_n_mol), 0.0)

    n_Hminus = _pow10(jnp.log10(Hminus_nK(T)) + log_n_I[0] + log_ne + log_xi)

    n_nuclei = n_total - ne + jnp.sum((data.mol_n_atoms - 1.0) * n_mol)

    return (n_I, n_II, n_III, ne, n_mol, n_Hminus, n_nuclei,
            C, is_charged, log_wII)


def _chem_eq_log_residuals(y, T, n_total, abundances, data, log_xi=0.0):
    """Scaled residuals of the v1.2 chemical equilibrium system in log₁₀ space."""
    (n_I, n_II, n_III, ne, n_mol, n_Hminus, n_nuclei,
     C, is_charged, _) = _chem_eq_log_terms(y, T, n_total, abundances, data, log_xi)

    G_atom = C.T @ n_mol + n_I + n_II + n_III - abundances * n_nuclei
    G_atom = G_atom.at[0].add(n_Hminus)

    G_e = (jnp.sum(data.mol_charges * n_mol) - n_Hminus
           + jnp.sum(n_II + 2.0 * n_III) - ne)

    F_atom = G_atom / (abundances * n_total)
    F_e = G_e / n_total
    return jnp.concatenate([F_atom, jnp.reshape(F_e, (1,))])


def _chem_eq_log_jacobian(y, T, n_total, abundances, data, log_xi=0.0):
    """
    Analytic 93x93 Jacobian of :func:`_chem_eq_log_residuals`.

    Fully vectorised over molecules — no scan or cond — so it composes with an outer
    vmap over atmosphere layers and stays reverse-mode differentiable.
    """
    N = MAX_ATOMIC_NUMBER
    (n_I, n_II, n_III, ne, n_mol, n_Hminus, _,
     C, is_charged, _) = _chem_eq_log_terms(y, T, n_total, abundances, data, log_xi)

    extra_nuclei = data.mol_n_atoms - 1.0     # (n_mol,) nuclei locked up per molecule
    q = data.mol_charges                       # (n_mol,)

    # ∂G_i/∂y_j -----------------------------------------------------------------
    #   molecules:  Σ_m C[m,i] n_mol[m] C[m,j]
    #   H⁻:         δ_i0 δ_j0 n_H⁻
    #   atoms:      δ_ij (n_I + n_II + n_III)_i
    #   nuclei:     -A_i Σ_m (k_m - 1) n_mol[m] C[m,j]
    mol_block = (C * n_mol[:, None]).T @ C                     # (N, N)
    nuclei_row = (extra_nuclei * n_mol) @ C                    # (N,)
    J_aa = mol_block - jnp.outer(abundances, nuclei_row)
    J_aa = J_aa + jnp.diag(n_I + n_II + n_III)
    J_aa = J_aa.at[0, 0].add(n_Hminus)

    # ∂G_i/∂y_e -----------------------------------------------------------------
    #   molecules lose a factor of nₑ only through the ionized constituent (s_m)
    mol_charged = n_mol * is_charged
    d_nuclei_d_ye = -ne - jnp.sum(extra_nuclei * mol_charged)
    J_ae = (-(C.T @ mol_charged)
            - n_II - 2.0 * n_III
            - abundances * d_nuclei_d_ye)
    J_ae = J_ae.at[0].add(n_Hminus)

    # ∂G_e/∂y_j -----------------------------------------------------------------
    J_ea = (q * n_mol) @ C + n_II + 2.0 * n_III
    J_ea = J_ea.at[0].add(-n_Hminus)

    # ∂G_e/∂y_e -----------------------------------------------------------------
    J_ee = (-jnp.sum(q * mol_charged) - n_Hminus
            - jnp.sum(n_II + 4.0 * n_III) - ne)

    # Assemble, apply the ln(10) from d(10^u)/du, then the constant residual scaling.
    J = jnp.zeros((N + 1, N + 1))
    J = J.at[:N, :N].set(J_aa)
    J = J.at[:N, N].set(J_ae)
    J = J.at[N, :N].set(J_ea)
    J = J.at[N, N].set(J_ee)
    J = J * _LN10

    scale = jnp.concatenate([abundances * n_total,
                             jnp.reshape(jnp.asarray(n_total, dtype=J.dtype), (1,))])
    return J / scale[:, None]


@jax.custom_jvp
def _chem_eq_solve_log(T, n_total, abundances, data, y0):
    """
    Run the annealed, step-clipped Newton solve and return the solution in log₁₀ space.

    The iteration itself is not differentiated. At the solution F(y*, θ) = 0, so the
    implicit function theorem gives dy*/dθ = -J⁻¹ ∂F/∂θ exactly, using the analytic
    Jacobian we already form each step. That is both cheaper and better conditioned than
    unrolling the annealing schedule, and it avoids propagating cotangents through the
    guarded steps below, whose ``where``s are there to survive divergence in the forward
    pass and would otherwise poison the gradient with NaN.
    """
    def newton_step(y, log_xi):
        F = _chem_eq_log_residuals(y, T, n_total, abundances, data, log_xi)
        J = _chem_eq_log_jacobian(y, T, n_total, abundances, data, log_xi)
        step = jnp.linalg.solve(J, -F)
        step = jnp.where(jnp.isfinite(step), step, 0.0)
        # Clip so no component moves more than one decade, as Korg's clipped_newton does.
        smax = jnp.max(jnp.abs(step))
        alpha = jnp.where(smax > 1.0, 1.0 / smax, 1.0)
        y_new = y + alpha * step
        # A diverged step leaves y unchanged rather than poisoning the rest of the walk.
        return jnp.where(jnp.all(jnp.isfinite(y_new)), y_new, y)

    y = y0
    for log_xi in _XI_SCHEDULE:
        y = jax.lax.fori_loop(0, _XI_INNER_ITERS,
                              lambda _, yy, lx=log_xi: newton_step(yy, lx), y)
    return jax.lax.fori_loop(0, _XI_FINAL_ITERS, lambda _, yy: newton_step(yy, 0.0), y)


@_chem_eq_solve_log.defjvp
def _chem_eq_solve_log_jvp(primals, tangents):
    T, n_total, abundances, data, y0 = primals
    dT, dn_total, dabundances = tangents[0], tangents[1], tangents[2]

    y = _chem_eq_solve_log(T, n_total, abundances, data, y0)

    # ∂F/∂θ · dθ holding the solution fixed, for θ = (T, n_total, abundances). Two inputs
    # carry no tangent by construction: the initial guess does not appear in
    # F(y*, θ) = 0 and so cannot move the solution, and `data` is a static table of
    # precomputed partition functions and equilibrium constants (it is closed over here
    # rather than passed through jvp, which also keeps its integer fields out of the
    # tangent space).
    _, dF = jax.jvp(
        lambda T_, n_, ab_: _chem_eq_log_residuals(y, T_, n_, ab_, data, 0.0),
        (T, n_total, abundances),
        (dT, dn_total, dabundances),
    )
    J = _chem_eq_log_jacobian(y, T, n_total, abundances, data, 0.0)
    return y, -jnp.linalg.solve(J, dF)


def _chem_eq_newton_layer_jit(T, n_total, ne_guess, nf_guess, abundances, data):
    """
    Single-layer chemical equilibrium solve, log-space, jittable and differentiable.

    Implements the Korg.jl v1.2 algorithm: a step-clipped Newton iteration (no component
    of the state may move more than one decade per step) driven by the analytic Jacobian
    above, wrapped in a continuation that switches molecules and H⁻ off and anneals them
    back on. Unlike Korg's adaptive bisection the schedule is fixed, so the solve is a
    single traceable kernel that vmaps over layers, and derivatives come from the
    implicit function theorem rather than from unrolling the iteration.

    Parameters
    ----------
    T, n_total : float
        Layer temperature (K) and total number density (cm⁻³)
    ne_guess : float
        Initial electron density, e.g. from the Picard pass
    nf_guess : array, shape (92,)
        Initial neutral fractions
    abundances : array, shape (92,)
        Absolute abundances N(X)/N_total
    data : ChemicalEquilibriumData

    Returns
    -------
    tuple
        ``(ne, neutral_fractions)`` where the neutral fractions are defined against
        ``abundances * (n_total - ne)``, so multiplying gives the solved neutral
        densities exactly.
    """
    ne0 = jnp.maximum(ne_guess, 1.0)
    n_neutral0 = jnp.maximum(abundances * (n_total - ne0) * jnp.clip(nf_guess, 1e-30, 1.0),
                             1e-300)
    y0 = jnp.concatenate([jnp.log10(n_neutral0),
                          jnp.reshape(jnp.log10(ne0), (1,))])

    y = _chem_eq_solve_log(T, n_total, abundances, data, jax.lax.stop_gradient(y0))

    ne_sol = _pow10(y[MAX_ATOMIC_NUMBER])
    n_neutral = _pow10(y[:MAX_ATOMIC_NUMBER])
    nf_sol = n_neutral / jnp.maximum(abundances * (n_total - ne_sol), 1e-300)
    return ne_sol, nf_sol


# Batch (all layers) Newton solver -- vmapped; JIT-compiled on first call.
# With the analytical Jacobian, compile time is significantly reduced vs jacfwd.
_chem_eq_newton_batch_jit = jax.jit(
    jax.vmap(_chem_eq_newton_layer_jit, in_axes=(0, 0, 0, 0, None, None))
)


@jax.jit
def _chem_eq_newton_scan_jit(T_layers, n_total_layers, ne_init, nf_init, abundances, data):
    """
    Newton solver over all layers using a temperature-sorted sequential scan.

    Adjacent atmosphere layers have similar T and composition, so the converged
    solution from layer k provides an excellent initial guess for layer k+1.
    This reduces Newton iterations from ~5–8 per layer to ~2–3.

    The layers are sorted by temperature (ascending) before the scan so that
    each step is a small extrapolation.  Results are unshuffled back to the
    original layer order before returning.

    Parameters
    ----------
    T_layers : jax array, shape (n_layers,)
        Temperature in K for each layer.
    n_total_layers : jax array, shape (n_layers,)
        Total number density (cm⁻³) for each layer.
    ne_init : jax array, shape (n_layers,)
        Initial electron density guess from Picard pre-solver.
    nf_init : jax array, shape (n_layers, 92)
        Initial neutral-fraction guess from Picard pre-solver.
    abundances : jax array, shape (92,)
        Absolute abundances N(X)/N_total (same for all layers).
    data : ChemicalEquilibriumData
        Pre-computed data from precompute_chemical_equilibrium_data().

    Returns
    -------
    ne_all : jax array, shape (n_layers,)
    nf_all : jax array, shape (n_layers, 92)
    """
    n_layers = T_layers.shape[0]

    # Sort layers by temperature (ascending) so adjacent steps are small.
    sort_idx   = jnp.argsort(T_layers)          # (n_layers,) int
    unsort_idx = jnp.argsort(sort_idx)           # inverse permutation

    T_sorted       = T_layers[sort_idx]
    n_total_sorted = n_total_layers[sort_idx]
    ne_sorted      = ne_init[sort_idx]
    nf_sorted      = nf_init[sort_idx]           # (n_layers, 92)

    # Seed the scan carry with the first layer's Picard guess.
    # Subsequent layers inherit the previous converged (ne, nf).
    def scan_body(carry, xs):
        ne_prev, nf_prev = carry        # previous layer's converged solution
        T_k, n_k, ne_k, nf_k = xs      # current layer's data + Picard guess

        # Use the previous converged state as warm-start; fall back to
        # the Picard guess for the very first layer (carry == Picard guess[0]).
        ne_sol, nf_sol = _chem_eq_newton_layer_jit(T_k, n_k, ne_prev, nf_prev, abundances, data)
        return (ne_sol, nf_sol), (ne_sol, nf_sol)

    # Initial carry: Picard guess for the first (coldest) layer
    init_carry = (ne_sorted[0], nf_sorted[0])
    xs = (T_sorted, n_total_sorted, ne_sorted, nf_sorted)

    _, (ne_out, nf_out) = jax.lax.scan(scan_body, init_carry, xs)

    # Restore original layer order
    ne_all = ne_out[unsort_idx]
    nf_all = nf_out[unsort_idx]
    return ne_all, nf_all


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
    ne_sol_all, neutral_fracs_all = _picard_chemical_equilibrium_guess_batch(
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

        # Joint outer loop: correct ne and molecular densities together.
        # The Picard ne is computed ignoring molecular atom depletion; this loop
        # iterates (a) the molecular correction and (b) a Picard update for ne using
        # the molecular-corrected total densities, matching Julia's Newton solution.
        #
        # Correct formula: n_Z_neutral = (n_Z_total - n_mol_Z) * nf_Z
        # where nf_Z = 1/(1+wII+wIII) and n_Z_total = abs_abund[Z] * (n_total - ne).
        # (Wrong: neutral_picard - mol_correction; that neglects the factor nf_Z.)
        ne_work = ne_np.copy()
        wII_work = wII_np.copy()
        wIII_work = wIII_np.copy()

        for _outer in range(5):
            atom_dens_w = (n_total_arr[:, None] - ne_work[:, None]) * absolute_abundances[None, :]
            nf_work = 1.0 / (1 + wII_work + wIII_work)

            # Inner loop: converge molecular densities given current ne/nf
            for _inner in range(5):
                mol_atom_correction = mol_np @ M_consume       # (n_layers, 92)
                total_corr = np.maximum(atom_dens_w - mol_atom_correction, 1e-99)
                neutral_dens_corrected = total_corr * nf_work  # correct formula
                nf_corrected = neutral_dens_corrected / np.maximum(atom_dens_w, 1e-99)
                mol_dens_all = _compute_mol_densities_batch_jit(
                    T_jax, n_total_jax, jnp.asarray(ne_work), abs_abund_jax,
                    jnp.asarray(nf_corrected), data
                )
                mol_np = np.asarray(mol_dens_all)

            # Picard update for ne using molecular-corrected total densities
            ioniz_fac = (wII_work + 2*wIII_work) / np.maximum(1 + wII_work + wIII_work, 1e-300)
            ne_new_arr = np.sum(ioniz_fac * total_corr, axis=1)
            ne_new_arr = np.clip(ne_new_arr, 1.0, np.max(n_total_arr))
            log_ne = 0.3 * np.log(np.clip(ne_work, 1e-99, None)) + 0.7 * np.log(np.clip(ne_new_arr, 1e-99, None))
            ne_work = np.exp(log_ne)

            wII_new_all, wIII_new_all = _compute_saha_weights_batch_jit(
                T_jax, jnp.asarray(ne_work), data
            )
            wII_work = np.asarray(wII_new_all)
            wIII_work = np.asarray(wIII_new_all)

        ne_np = ne_work
        wII_np = wII_work
        wIII_np = wIII_work
    else:
        atom_dens_w = (n_total_arr[:, None] - ne_np[:, None]) * absolute_abundances[None, :]
        nf_work = 1.0 / (1 + wII_np + wIII_np)
        neutral_dens_corrected = atom_dens_w * nf_work

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
