"""
Tests for the batched/JIT chemical equilibrium solver.

The batched path solves the same Korg.jl v1.2 system as
``tests.reference_chemical_equilibrium`` but with a hand-written analytic Jacobian, a fixed
continuation schedule, and derivatives supplied by the implicit function theorem, so
that the whole solve is one jittable, vmappable, differentiable kernel.

These tests pin the three things that can silently break:

  * the hand-derived Jacobian, against ``jax.jacfwd`` of the residual
  * the solution, against the (Julia-verified) reference implementation
  * jittability and reverse-mode differentiability
"""

import numpy as np
import pytest

# Import korg FIRST to enable JAX x64 mode
import korg  # noqa: F401

import jax
import jax.numpy as jnp


@pytest.fixture(scope="module")
def setup():
    from korg.statmech import precompute_chemical_equilibrium_data
    from korg.data_loader import (
        load_ionization_energies, load_atomic_partition_functions,
        default_log_equilibrium_constants,
    )
    from korg.abundances import format_A_X, A_X_to_absolute

    try:
        ionization_energies = load_ionization_energies()
        partition_funcs = load_atomic_partition_functions()
    except FileNotFoundError as e:
        pytest.skip(f"Data files not found: {e}")

    data = precompute_chemical_equilibrium_data(
        ionization_energies, partition_funcs, default_log_equilibrium_constants,
        T_min=2000.0, T_max=20000.0, n_temps=200,
    )
    abundances_np = np.asarray(A_X_to_absolute(format_A_X()))
    return {
        "data": data,
        "abundances": jnp.asarray(abundances_np),
        "abundances_np": abundances_np,
        "ionization_energies": ionization_energies,
        "partition_funcs": partition_funcs,
        "log_equilibrium_constants": default_log_equilibrium_constants,
    }


# (T, n_total, ne_model) spanning cool molecular layers through hot ionized ones.
REGIMES = [
    (2500.0, 1e17, 1e10),
    (3000.0, 1e16, 1e11),
    (3500.0, 3e16, 3e11),
    (4500.0, 1e16, 1e12),
    (5778.0, 1e17, 1e14),
    (8000.0, 1e15, 1e13),
    (12000.0, 1e14, 5e13),
]


class TestAnalyticJacobian:
    """The Jacobian is written out by hand, so it needs its own guard."""

    @pytest.mark.parametrize("T,n_total,log_xi", [
        (5000.0, 1e17, 0.0),
        (2500.0, 1e17, 0.0),
        (2500.0, 1e17, -3.0),    # part-way through the molecular continuation
        (8000.0, 1e15, 0.0),
        (4000.0, 1e16, -1.25),
    ])
    def test_matches_autodiff(self, setup, T, n_total, log_xi):
        from korg.statmech import _chem_eq_log_residuals, _chem_eq_log_jacobian

        ab = setup["abundances"]
        data = setup["data"]
        ne = 1e13 if T > 4000 else 1e10

        # Perturb off the initial guess so the test does not sit at a special point.
        rng = np.random.default_rng(0)
        y = jnp.concatenate([
            jnp.log10(ab * (n_total - ne) * 0.9),
            jnp.array([np.log10(ne)]),
        ]) + jnp.asarray(rng.normal(0.0, 0.15, 93))

        J_analytic = _chem_eq_log_jacobian(y, T, n_total, ab, data, log_xi)
        J_autodiff = jax.jacfwd(
            lambda yy: _chem_eq_log_residuals(yy, T, n_total, ab, data, log_xi)
        )(y)

        scale = jnp.maximum(jnp.abs(J_autodiff), 1e-6 * jnp.max(jnp.abs(J_autodiff)))
        max_rel = float(jnp.max(jnp.abs(J_analytic - J_autodiff) / scale))
        assert max_rel < 1e-10, f"analytic Jacobian differs from autodiff by {max_rel:.3e}"


class TestBatchedSolution:
    """The batched solver must agree with the Julia-verified reference solver."""

    @pytest.fixture(scope="class")
    def solved(self, setup):
        from korg.statmech import _chem_eq_newton_batch_jit, _picard_chemical_equilibrium_guess_batch

        T = jnp.array([r[0] for r in REGIMES])
        n_total = jnp.array([r[1] for r in REGIMES])
        ne_model = jnp.array([r[2] for r in REGIMES])
        ne_init, nf_init = _picard_chemical_equilibrium_guess_batch(
            T, n_total, ne_model, setup["abundances"], setup["data"]
        )
        ne, nf = _chem_eq_newton_batch_jit(
            T, n_total, ne_init, nf_init, setup["abundances"], setup["data"]
        )
        return np.asarray(ne), np.asarray(nf)

    @pytest.mark.parametrize("idx", range(len(REGIMES)))
    def test_matches_reference_solver(self, setup, solved, idx):
        from tests.reference_chemical_equilibrium import reference_chemical_equilibrium
        from korg.species import Species

        ne_arr, nf_arr = solved
        T, n_total, ne_model = REGIMES[idx]
        ab_np = setup["abundances_np"]

        ne_ref, nd_ref = reference_chemical_equilibrium(
            T, n_total, ne_model, ab_np, setup["ionization_energies"],
            setup["partition_funcs"], setup["log_equilibrium_constants"],
        )

        ne = float(ne_arr[idx])
        assert np.isclose(ne, ne_ref, rtol=1e-6), (
            f"T={T}: batched ne={ne:.6e} vs reference {ne_ref:.6e}"
        )

        # Neutral fractions are defined against abundances * (n_total - ne), so this
        # reconstructs the solved neutral densities exactly.
        densities = nf_arr[idx] * ab_np * (n_total - ne)
        for name, Z in [("H I", 1), ("C I", 6), ("Fe I", 26)]:
            ref = nd_ref[Species.from_string(name)]
            assert np.isclose(densities[Z - 1], ref, rtol=1e-6), (
                f"T={T}: {name} {densities[Z - 1]:.6e} vs reference {ref:.6e}"
            )

    def test_ionization_fraction_increases_with_temperature(self, solved):
        """REGIMES is ordered by temperature but also varies n_total, so the monotonic
        quantity is the electron fraction, not the electron density."""
        ne_arr, _ = solved
        n_total = np.array([r[1] for r in REGIMES])
        fraction = ne_arr / n_total
        assert np.all(np.diff(fraction) > 0), f"ne/n_total not monotonic: {fraction}"


class _LayerSolver:
    """The single-layer solve and its two gradients, compiled once for the class.

    Everything that varies between cases -- T, the total density, the Picard initial
    guess -- is an *argument* rather than a value closed over. That matters for the
    clock: a fresh closure carrying fresh constants hashes to a fresh HLO module, so
    building one lambda per case made every case pay the full reverse-mode compile of
    the Newton solver (~80 s each, four times over). As arguments they are one
    executable that all the cases share.
    """

    def __init__(self, abundances, data):
        from korg.statmech import (_chem_eq_newton_layer_jit,
                                   _picard_chemical_equilibrium_guess_batch)

        self._guess_batch = _picard_chemical_equilibrium_guess_batch
        self._abundances = abundances
        self._data = data

        def solve(T, n_total, ne_init, nf_init):
            return _chem_eq_newton_layer_jit(T, n_total, ne_init, nf_init,
                                             abundances, data)[0]

        self.ne = jax.jit(solve)
        self.d_ne_dT = jax.jit(jax.grad(solve, argnums=0))
        self.d_ne_dn = jax.jit(jax.grad(solve, argnums=1))

    def guess(self, T0, n_total, ne_model):
        """The Picard starting point for one layer, as ``(ne_init, nf_init)``."""
        ne_init, nf_init = self._guess_batch(
            jnp.array([T0]), jnp.array([n_total]), jnp.array([ne_model]),
            self._abundances, self._data,
        )
        return ne_init[0], nf_init[0]


class TestJitAndDifferentiability:
    """Jittability and reverse-mode gradients are requirements, not nice-to-haves."""

    @pytest.fixture(scope="class")
    def layer_solver(self, setup):
        return _LayerSolver(setup["abundances"], setup["data"])

    @pytest.mark.parametrize("T0,n_total,ne_model", [
        (5778.0, 1e17, 1e14),
        (3000.0, 1e16, 1e11),
        (2500.0, 1e17, 1e10),
    ])
    def test_reverse_mode_grad_matches_finite_difference(self, layer_solver,
                                                         T0, n_total, ne_model):
        ne_init, nf_init = layer_solver.guess(T0, n_total, ne_model)
        # jnp.float64 throughout: a Python float is weakly typed and would trace a
        # second time, which is the recompile this class exists to avoid.
        rest = (jnp.float64(n_total), ne_init, nf_init)

        grad = float(layer_solver.d_ne_dT(jnp.float64(T0), *rest))
        fd = (float(layer_solver.ne(jnp.float64(T0 * 1.00005), *rest))
              - float(layer_solver.ne(jnp.float64(T0 * 0.99995), *rest))) / (0.0001 * T0)
        assert np.isfinite(grad), "d(ne)/dT is not finite"
        assert np.isclose(grad, fd, rtol=1e-5), f"grad {grad:.6e} vs finite diff {fd:.6e}"

    def test_jit_compiles(self, layer_solver):
        ne_init, nf_init = layer_solver.guess(5778.0, 1e17, 1e14)
        ne = layer_solver.ne(jnp.float64(5778.0), jnp.float64(1e17), ne_init, nf_init)
        assert np.isfinite(float(ne))

    def test_grad_wrt_total_density(self, layer_solver):
        T0, n_total, ne_model = 5778.0, 1e17, 1e14
        ne_init, nf_init = layer_solver.guess(T0, n_total, ne_model)

        def ne_at(n):
            return float(layer_solver.ne(jnp.float64(T0), jnp.float64(n),
                                         ne_init, nf_init))

        grad = float(layer_solver.d_ne_dn(jnp.float64(T0), jnp.float64(n_total),
                                          ne_init, nf_init))
        fd = (ne_at(n_total * 1.00005) - ne_at(n_total * 0.99995)) / (0.0001 * n_total)
        assert np.isclose(grad, fd, rtol=1e-5)

    def test_saha_weights_grad_is_finite(self, setup):
        """Hydrogen has no third ionization state; its placeholder table must not
        produce a NaN gradient (a masked NaN still poisons reverse mode)."""
        from korg.statmech import _compute_saha_weights_jit

        grad = jax.grad(
            lambda t: jnp.sum(_compute_saha_weights_jit(t, 1.0, setup["data"])[1])
        )(jnp.float64(5778.0))
        assert np.isfinite(grad), "d(wIII)/dT is NaN"

        _, wIII = _compute_saha_weights_jit(jnp.float64(5778.0), 1.0, setup["data"])
        assert float(wIII[0]) == 0.0, "hydrogen cannot be doubly ionized"


class TestEquilibriumConstantTabulation:
    """The precomputed tables must equal the functions they tabulate."""

    def test_vectorised_matches_scalar(self, setup):
        """Polyatomic log K closures are called on the whole temperature grid at once
        when the tables are built. A reduction that collapsed the grid instead of
        summing over constituent atoms would silently corrupt every polyatomic."""
        from korg.statmech import get_log_nK

        K = setup["log_equilibrium_constants"]
        molecules = list(K.keys())
        log_T_grid = np.log(np.array([2500.0, 3500.0, 5778.0, 9000.0]))

        for mol in molecules:
            func = K[mol]
            if hasattr(func, "numpy_eval"):
                continue  # tabulated diatomics, not the closure path
            vectorised = np.asarray(func(log_T_grid))
            assert vectorised.shape == log_T_grid.shape, (
                f"{mol}: vectorised call returned shape {vectorised.shape}, "
                f"expected {log_T_grid.shape}"
            )
            scalar = np.array([float(func(np.array(lt))) for lt in log_T_grid])
            np.testing.assert_allclose(vectorised, scalar, rtol=1e-12)

    @pytest.mark.parametrize("T", [2500.0, 3500.0, 5778.0])
    def test_tabulated_log_nK_matches_direct(self, setup, T):
        from korg.statmech import _mol_log_nK_all_jit, get_log_nK

        K = setup["log_equilibrium_constants"]
        molecules = list(K.keys())
        tabulated = np.asarray(_mol_log_nK_all_jit(T, setup["data"]))
        direct = np.array([float(get_log_nK(m, T, K)) for m in molecules])

        # The JIT path interpolates a cubic spline through a temperature grid, so a
        # small interpolation error is expected; a broken tabulation is not.
        max_diff = float(np.max(np.abs(tabulated - direct)))
        assert max_diff < 1e-3, f"tabulated log_nK differs from direct by {max_diff:.3e}"
