"""
Reference chemical equilibrium solver — the oracle the package is checked against.

This is a second, independent implementation of the Korg.jl v1.2 chemical equilibrium
system. It deliberately lives in the test suite rather than in ``korg``: it is not the
solver anything in the package uses, and keeping it here makes that impossible to
confuse. ``korg.statmech`` provides the production solver
(``chemical_equilibrium_all_layers`` / ``_chem_eq_newton_layer_jit``), which is jittable,
vmappable and differentiable.

The two differ in exactly two ways, and those differences are the point:

* **Coefficients.** This solver evaluates partition functions and equilibrium constants
  directly from their cubic splines. The production solver interpolates them from a
  precomputed temperature grid, which is what makes it fast and traceable, and which
  costs it about nine digits.
* **Continuation.** This solver mirrors Korg's adaptive bisection on the molecular
  continuation parameter, branching on Python control flow. The production solver walks a
  fixed schedule so the whole solve is one traceable kernel.

Consequently this implementation is the more accurate of the two — it reproduces Korg.jl
v1.2.1 to 8e-15 — and is neither jittable nor differentiable. Having both, and asserting
they agree, is what makes a silent error in either one visible: an independent
implementation is worth more as an oracle than as a second thing to maintain.

Used by tests/test_chemical_equilibrium_jit.py and tests/test_julia_reference.py.
"""

import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# Import korg first so JAX x64 mode is enabled before anything numeric happens.
import korg  # noqa: F401
from korg.constants import kboltz_cgs
from korg.statmech import (
    MAX_ATOMIC_NUMBER,
    Hminus_nK,
    _pow10,
    get_log_nK,
    saha_ion_weights,
)


class EquilibriumArrays(NamedTuple):
    """Temperature-dependent arrays defining the chemical equilibrium system.

    All log quantities are base 10. The Saha weights are stored with their factors of
    nₑ⁻¹ and nₑ⁻² divided out, so they depend only on temperature.
    """
    abund: jnp.ndarray          # (92,) absolute abundances N(X)/N_total
    log_wII_ne: jnp.ndarray     # (92,) log10 of n(X II)/n(X I) at nₑ = 1
    log_wIII_ne2: jnp.ndarray   # (92,) log10 of n(X III)/n(X I) at nₑ = 1
    log_nKs: jnp.ndarray        # (n_mol,) log10 number-density equilibrium constants
    log_nK_Hminus: jnp.ndarray  # scalar, log10 of the H⁻ formation coefficient
    mol_counts: jnp.ndarray     # (n_mol, 92) atoms of each element per molecule
    mol_charges: jnp.ndarray    # (n_mol,)
    mol_n_atoms: jnp.ndarray    # (n_mol,) total atoms per molecule
    mol_first_atom: jnp.ndarray # (n_mol,) index of the charged atom in ionized diatomics


def precompute_equilibrium_data(T, n_total, absolute_abundances,
                                 ionization_energies, partition_funcs,
                                 log_equilibrium_constants):
    """
    Pre-compute all data needed for chemical equilibrium as pure arrays.

    This extracts all data from Python objects (dicts, partition funcs) into
    JAX-compatible arrays that can be used in JIT-compiled functions.

    Returns
    -------
    EquilibriumArrays
        Named tuple of arrays describing the atomic and molecular system at this
        temperature. ``mol_counts`` is the (n_molecules, 92) matrix giving how many
        atoms of each element every molecule contains, so element conservation and the
        molecular densities are both plain matrix products.
    """
    from korg.species import Species, Formula

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

    n_mol = len(log_nKs_list)
    if n_mol > 0:
        log_nKs = jnp.array(log_nKs_list)
        mol_atoms_array = jnp.array(mol_atoms_list, dtype=jnp.int32)
        mol_charges = jnp.array(mol_charges_list, dtype=jnp.float64)
        mol_n_atoms = jnp.array(mol_n_atoms_list, dtype=jnp.float64)
        # mol_counts[i, Z-1] = number of atoms of element Z in molecule i (with
        # multiplicity, so H2O contributes 2 to hydrogen).
        counts = np.zeros((n_mol, MAX_ATOMIC_NUMBER))
        for i, padded in enumerate(mol_atoms_list):
            for Z_idx in padded:
                if Z_idx >= 0:
                    counts[i, Z_idx] += 1.0
        mol_counts = jnp.asarray(counts)
        # For singly ionized diatomics the charged component is the first (lowest-Z) atom.
        mol_first_atom = mol_atoms_array[:, 0]
    else:
        log_nKs = jnp.zeros(0)
        mol_atoms_array = jnp.zeros((0, 6), dtype=jnp.int32)
        mol_charges = jnp.zeros(0)
        mol_n_atoms = jnp.zeros(0)
        mol_counts = jnp.zeros((0, MAX_ATOMIC_NUMBER))
        mol_first_atom = jnp.zeros(0, dtype=jnp.int32)

    return EquilibriumArrays(
        abund=abund_array,
        log_wII_ne=jnp.log10(wII_ne_array),
        log_wIII_ne2=jnp.log10(wIII_ne2_array),
        log_nKs=log_nKs,
        log_nK_Hminus=jnp.log10(Hminus_nK(T)),
        mol_counts=mol_counts,
        mol_charges=mol_charges,
        mol_n_atoms=mol_n_atoms,
        mol_first_atom=mol_first_atom,
    )


def _compute_residuals_core(x, n_total, arrays, log_xi):
    """
    Residuals of the chemical equilibrium system, in log₁₀ number density space.

    This is the Korg.jl v1.2 formulation. The free parameters are log₁₀ n(X I) for each
    element and log₁₀ nₑ, and there is one equation per element plus a charge balance
    equation:

        n(O) = n(CO) + n(OH) + n(O I) + n(O II) + n(O III) + ...
        0    = -nₑ - n(H⁻) + n(H II) + 2 n(H III) + ...

    Two things distinguish this from the v1.1 system it replaces. Molecules lock up more
    nuclei than they contribute particles, so the nucleus budget is
    ``nₜ - nₑ + Σ (n_atoms - 1) n_mol`` rather than simply ``nₜ - nₑ``; and H⁻ is carried
    explicitly, consuming a hydrogen nucleus and a unit of negative charge.

    Parameters
    ----------
    x : array, shape (93,)
        [log₁₀ n(X I) for Z = 1..92, log₁₀ nₑ]
    n_total : float
        Total number density in cm⁻³
    arrays : EquilibriumArrays
        Precomputed temperature-dependent coefficients
    log_xi : float
        Continuation parameter. Molecular and H⁻ densities are scaled by 10^log_xi, so
        log_xi = 0 is the physical system and log_xi → -∞ suppresses them entirely,
        leaving pure atomic Saha. Used to anneal into hard (cool, dense) regimes.

    Returns
    -------
    array, shape (93,)
        Residuals, scaled so that convergence tolerances are dimensionless
    """
    log_ne = x[-1]
    ne = _pow10(log_ne)
    log_n_neutral = x[:MAX_ATOMIC_NUMBER]

    # ---- molecules -------------------------------------------------------------
    # log n_mol = Σ log n(constituent) - log K_n, where every constituent is neutral
    # except the first atom of a singly ionized diatomic.
    log_n_mol = arrays.mol_counts @ log_n_neutral - arrays.log_nKs + log_xi
    if arrays.mol_charges.shape[0] > 0:
        is_charged = arrays.mol_charges != 0
        first = arrays.mol_first_atom
        # swap the neutral first atom for its ion: + log wII_ne[Z1] - log nₑ
        log_n_mol = log_n_mol + jnp.where(
            is_charged,
            arrays.log_wII_ne[first] - log_ne,
            0.0,
        )
    n_mol = _pow10(log_n_mol)

    # Element conservation picks up one term per constituent atom (with multiplicity),
    # which is exactly the transpose of the count matrix.
    F_elements = arrays.mol_counts.T @ n_mol

    # Molecules hold nuclei that are not free particles.
    n_nuclei = n_total - ne + jnp.sum((arrays.mol_n_atoms - 1.0) * n_mol)

    F_charge = jnp.sum(arrays.mol_charges * n_mol)

    # ---- H⁻ --------------------------------------------------------------------
    # n(H⁻) = nK(T) n(H I) nₑ: consumes an H nucleus, carries negative charge.
    n_Hminus = _pow10(arrays.log_nK_Hminus + log_n_neutral[0] + log_ne + log_xi)
    F_elements = F_elements.at[0].add(n_Hminus)
    F_charge = F_charge - n_Hminus

    # ---- atoms and their ions ---------------------------------------------------
    n_I = _pow10(log_n_neutral)
    n_II = _pow10(log_n_neutral + arrays.log_wII_ne - log_ne)
    n_III = _pow10(log_n_neutral + arrays.log_wIII_ne2 - 2.0 * log_ne)

    F_elements = F_elements + n_I + n_II + n_III - arrays.abund * n_nuclei
    F_charge = F_charge + jnp.sum(n_II + 2.0 * n_III) - ne

    # ---- scaling ----------------------------------------------------------------
    F_elements = F_elements / (arrays.abund * n_total)
    F_charge = F_charge / n_total

    return jnp.concatenate([F_elements, jnp.reshape(F_charge, (1,))])


def setup_chemical_equilibrium_residuals(T, n_total, absolute_abundances,
                                        ionization_energies, partition_funcs,
                                        log_equilibrium_constants, log_xi=0.0):
    """
    Build the residual function for chemical equilibrium at fixed T and nₜ.

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
    log_xi : float, optional
        Continuation parameter suppressing molecules and H⁻ (default: 0, the physical
        system). See :func:`_compute_residuals_core`.

    Returns
    -------
    callable
        ``residuals(x)`` for the state vector ``x = [log₁₀ n(X I)..., log₁₀ nₑ]``
    """
    arrays = precompute_equilibrium_data(
        T, n_total, absolute_abundances, ionization_energies,
        partition_funcs, log_equilibrium_constants
    )

    @jax.jit
    def residuals(x):
        return _compute_residuals_core(x, n_total, arrays, log_xi)

    return residuals


def clipped_newton(residuals_func, x0, tol=1e-8, max_iter=50, max_step=1.0):
    """
    Newton's method with step clipping, matching Korg.jl v1.2's ``clipped_newton``.

    Because the system is solved in log₁₀ space, limiting the largest component of each
    step to ``max_step`` decades keeps the iteration from leaping into regions where the
    molecular terms overflow.

    Parameters
    ----------
    residuals_func : callable
        Function computing F(x); a solution satisfies F(x) = 0
    x0 : array
        Initial guess
    tol : float, optional
        Convergence tolerance on ‖F‖_∞ (default: 1e-8)
    max_iter : int, optional
        Maximum iterations (default: 50)
    max_step : float, optional
        Largest permitted change in any component, in decades (default: 1.0)

    Returns
    -------
    tuple
        (x, converged, residual_inf_norm)
    """
    jac_func = jax.jit(jax.jacfwd(residuals_func))

    x = jnp.asarray(x0)
    inf_norm = jnp.inf
    for _ in range(max_iter):
        F = residuals_func(x)
        inf_norm = jnp.max(jnp.abs(F))
        if not jnp.isfinite(inf_norm):
            return x, False, inf_norm
        if inf_norm < tol:
            return x, True, inf_norm

        J = jac_func(x)
        if not jnp.all(jnp.isfinite(J)):
            return x, False, inf_norm

        step = jnp.linalg.solve(J, -F)
        if not jnp.all(jnp.isfinite(step)):
            return x, False, inf_norm

        # Clip so the largest component moves by at most max_step decades.
        smax = jnp.max(jnp.abs(step))
        alpha = jnp.where(smax > max_step, max_step / smax, 1.0)
        x = x + alpha * step

    return x, False, inf_norm


class ChemicalEquilibriumError(Exception):
    """Raised when the chemical equilibrium solver fails to converge."""


def _solve_chemical_equilibrium(T, n_total, absolute_abundances, neutral_fraction_guess,
                                ne_guess, ionization_energies, partition_funcs,
                                log_equilibrium_constants,
                                ftol=1e-8, minimum_annealing_dlog_xi=0.0625):
    """
    Solve the equilibrium system, with continuation on molecules and H⁻ if needed.

    Follows Korg.jl v1.2. A direct solve of the full system (log_xi = 0) converges in a
    handful of iterations for warm or low-density layers. Where it fails — cool, dense
    regimes (T ≲ 3000 K, nₜ ≳ 1e15) in which an atoms-only initial guess wildly
    over-predicts molecular densities — we instead switch molecules off entirely and
    anneal them back on, bisecting the continuation step whenever Newton stalls so that
    only as many sub-steps as the regime demands are taken.

    Returns
    -------
    array, shape (93,)
        [log₁₀ n(X I) for Z = 1..92, log₁₀ nₑ]
    """
    def solve_at(log_xi, x_start, max_iter):
        residuals = setup_chemical_equilibrium_residuals(
            T, n_total, absolute_abundances, ionization_energies,
            partition_funcs, log_equilibrium_constants, log_xi
        )
        return clipped_newton(residuals, x_start, tol=ftol,
                              max_iter=max_iter, max_step=1.0)

    n_neutral_guess = (n_total - ne_guess) * absolute_abundances * neutral_fraction_guess
    x0 = jnp.concatenate([
        jnp.log10(jnp.asarray(n_neutral_guess)),
        jnp.reshape(jnp.log10(jnp.asarray(ne_guess)), (1,)),
    ])

    # Try the full system first.
    x, converged, last_inf = solve_at(0.0, x0, 50)

    if not converged:
        # Find a value of log_xi small enough that the system is essentially atomic.
        x_anchor = None
        for log_xi_try in (-5.0, -20.0, -50.0):
            x_anchor, anchor_converged, last_inf = solve_at(log_xi_try, x0, 100)
            if anchor_converged:
                log_xi_anchor = log_xi_try
                break
        else:
            raise ChemicalEquilibriumError(
                f"chemical equilibrium unconverged at anchor (inf-norm={last_inf:.3e})"
            )

        # Anneal log_xi up to 0, halving the step after a failure and doubling it
        # (capped at the initial value) after a success.
        x = x_anchor
        dlog_xi_init = 2.0
        dlog_xi = dlog_xi_init
        log_xi = log_xi_anchor
        while log_xi < -1e-12:
            step = min(dlog_xi, -log_xi)
            x_try, conv, last_inf = solve_at(log_xi + step, x, 100)
            if not conv:
                step /= 2
                while not conv and step >= minimum_annealing_dlog_xi:
                    x_try, conv, last_inf = solve_at(log_xi + step, x, 100)
                    if not conv:
                        step /= 2
                if not conv:
                    raise ChemicalEquilibriumError(
                        f"chemical equilibrium unconverged at log_xi={log_xi + 2 * step:.4f} "
                        f"(inf-norm={last_inf:.3e})"
                    )
                dlog_xi = step
            else:
                dlog_xi = min(dlog_xi_init, 2 * step)
            x = x_try
            log_xi += step

    if not jnp.all(jnp.isfinite(x)):
        raise ChemicalEquilibriumError("chemical equilibrium solution contains non-finite values")

    return x


def reference_chemical_equilibrium(T, n_total, ne_model, absolute_abundances,
                        ionization_energies, partition_funcs,
                        log_equilibrium_constants,
                        electron_density_warn_threshold=0.1,
                        electron_density_warn_min_value=1e-4):
    """
    Reference (oracle) chemical equilibrium solver — not the synthesis path.

    Solves the Korg.jl v1.2 system exactly: partition functions and equilibrium
    constants are evaluated directly from their splines rather than interpolated from a
    temperature grid, and the molecular continuation uses Korg's adaptive bisection.
    That makes it the most accurate implementation here — it reproduces Korg.jl v1.2.1
    to 8e-15 — and also the slowest, and it is neither jittable nor differentiable,
    because the adaptive schedule branches on Python control flow and the result is
    returned as Python floats.

    Its job is to be the thing everything else is checked against. Synthesis uses
    :func:`chemical_equilibrium_all_layers` / :func:`_chem_eq_newton_layer_jit`, which
    solve the same system from precomputed tables on a fixed continuation schedule, and
    agree with this function to 1.6e-9.

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
    2. Solves the nonlinear system in log₁₀ space with a step-clipped Newton method,
       annealing molecules and H⁻ back on if the direct solve fails
    3. Computes number densities for all species from the solution

    The system of equations enforces:
    - Conservation of each element (atoms + ions + molecules + H⁻)
    - Charge balance, with nₑ as a free parameter

    ``number_densities`` includes H⁻, which Korg.jl v1.2 promoted to a species carried
    by chemical equilibrium.

    Reference
    ---------
    Kurucz 1970, sections 5.1-5.3
    Gray 2005, "The Observation and Analysis of Stellar Photospheres", Ch. 8
    """
    from korg.species import Species, Formula

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

    # Solve the system. The unknowns are log₁₀ n(X I) for each element plus log₁₀ nₑ.
    x_solution = _solve_chemical_equilibrium(
        T, n_total, abund_array, jnp.asarray(neutral_fraction_guess), ne_model,
        ionization_energies, partition_funcs, log_equilibrium_constants
    )

    ne = float(10.0 ** x_solution[-1])
    n_neutral_solved = 10.0 ** x_solution[:MAX_ATOMIC_NUMBER]

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

    # Neutral atomic species are what the solver returns directly
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        formula = Formula(int(Z))
        number_densities[Species(formula, 0)] = float(n_neutral_solved[Z-1])

    # Ionized atomic species
    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        formula = Formula(int(Z))
        wII, wIII = saha_ion_weights(T, ne, Z, ionization_energies, partition_funcs)

        n_neutral = number_densities[Species(formula, 0)]
        number_densities[Species(formula, 1)] = float(wII * n_neutral)
        number_densities[Species(formula, 2)] = float(wIII * n_neutral)

    # H⁻, from its formation equilibrium with H I and free electrons
    number_densities[Species(Formula(1), -1)] = float(
        Hminus_nK(T) * number_densities[Species(Formula(1), 0)] * ne
    )

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
