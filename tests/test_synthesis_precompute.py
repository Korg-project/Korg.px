"""
Tests for the JIT-side plumbing of ``korg.synthesis``.

Covers the pieces that ``test_synthesis_functional.py`` (the Python-orchestrated
path) does not touch:

* ``precompute_synthesis_data`` and its Gaunt-table fallback,
* ``save_synthesis_data`` / ``load_synthesis_data`` round-tripping,
* ``preprocess_linelist`` (the ``synthesis.py`` one — note there is a second,
  unrelated ``preprocess_linelist`` in ``synthesis_preparation.py``),
* ``precompute_atmosphere`` and the fused bucket path it enables inside
  ``synthesize_jit``.

Categories: functional, high-precision agreement (the precomputed and
non-precomputed synthesis paths must agree to round-off), autodiff and jit.

Cost: ``precompute_atmosphere`` compiles a large XLA graph, so the fixture that
builds it is module-scoped and the tests that use it are marked ``slow``.
Everything else here runs in well under a second.
"""

import json
from pathlib import Path

import korg  # noqa: F401 — ensures JAX x64 mode

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from korg.atmosphere import PlanarAtmosphere
from korg.linelist import Line
from korg.species import Species
from korg.synthesis import (
    LinelistData,
    SynthesisData,
    load_synthesis_data,
    precompute_atmosphere,
    precompute_synthesis_data,
    preprocess_linelist,
    save_synthesis_data,
    synthesize_jit,
)

REFERENCE_JSON = Path(__file__).parent / "synthesis_reference_data.json"
SUN_MOD = Path(__file__).parent / "data" / "sun.mod"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def julia_ref():
    if not REFERENCE_JSON.exists():
        raise FileNotFoundError(
            f"{REFERENCE_JSON} is missing; run "
            "tests/generate_synthesis_reference.jl to regenerate it")
    with open(REFERENCE_JSON) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def tiny_atm():
    if not SUN_MOD.exists():
        raise FileNotFoundError(f"{SUN_MOD} is missing from the test data directory")
    full = korg.read_model_atmosphere(str(SUN_MOD))
    return PlanarAtmosphere(full.layers[::7], full.reference_wavelength)


@pytest.fixture(scope="module")
def abundances(julia_ref):
    a = jnp.array(10.0 ** (np.array(julia_ref["A_X"]) - 12.0))
    return a / a.sum()


@pytest.fixture(scope="module")
def wavelengths_cm():
    return jnp.array((5000.0 + 0.01 * np.arange(11)) * 1e-8)


@pytest.fixture(scope="module")
def fe_line(julia_ref):
    L = julia_ref["line"]
    return Line(wl=L["wl_cm"], log_gf=L["log_gf"], species=Species("Fe I"),
                E_lower=L["E_lower"], gamma_rad=L["gamma_rad"],
                gamma_stark=L["gamma_stark"], vdW=tuple(L["vdW"]))


@pytest.fixture(scope="module")
def synth_data():
    from korg.data_loader import (default_log_equilibrium_constants,
                                  default_partition_funcs, ionization_energies)
    return precompute_synthesis_data(ionization_energies, default_partition_funcs,
                                     default_log_equilibrium_constants)


@pytest.fixture(scope="module")
def linelist_data(fe_line, synth_data, wavelengths_cm):
    return preprocess_linelist([fe_line], chem_eq_data=synth_data.chem_eq_data,
                               wavelengths_cm=np.asarray(wavelengths_cm))


def _atm_args(atm):
    return dict(T_layers=jnp.array(atm.T),
                n_total_layers=jnp.array(atm.n_total),
                ne_layers=jnp.array(atm.ne),
                z_layers=jnp.array(atm.z),
                log_tau_ref=jnp.array(atm.log_tau_ref))


# ===========================================================================
# 1. Functional — precompute_synthesis_data
# ===========================================================================

class TestPrecomputeSynthesisData:

    def test_it_returns_a_populated_synthesis_data(self, synth_data):
        assert isinstance(synth_data, SynthesisData)
        assert synth_data.chem_eq_data.n_molecules > 0
        assert synth_data.gaunt_table.ndim == 2
        assert synth_data.metal_bf_tables.shape[0] == \
            synth_data.metal_bf_z_arr.shape[0]
        assert synth_data.metal_bf_z_arr.shape == \
            synth_data.metal_bf_charge_arr.shape

    def test_a_missing_gaunt_table_falls_back_to_unity(self, monkeypatch):
        """The Gaunt data are optional; without them the factor is taken as 1."""
        import korg.synthesis as syn
        from korg.continuum_absorption import hydrogenic_bf_ff
        from korg.data_loader import (default_log_equilibrium_constants,
                                      default_partition_funcs, ionization_energies)

        def boom():
            raise RuntimeError("simulated missing Gaunt table")

        monkeypatch.setattr(hydrogenic_bf_ff, "_load_gauntff_table", boom)
        data = syn.precompute_synthesis_data(ionization_energies,
                                             default_partition_funcs,
                                             default_log_equilibrium_constants)
        np.testing.assert_array_equal(np.asarray(data.gaunt_table), np.ones((2, 2)))
        np.testing.assert_array_equal(np.asarray(data.gaunt_log_u_grid),
                                      [-4.0, 4.0])
        np.testing.assert_array_equal(np.asarray(data.gaunt_log_gamma2_grid),
                                      [-4.0, 4.0])


class TestSaveAndLoadSynthesisData:

    def test_round_trip_preserves_every_array(self, synth_data, tmp_path):
        path = tmp_path / "synthesis_data.npz"
        save_synthesis_data(synth_data, str(path))
        assert path.exists()
        loaded = load_synthesis_data(str(path))

        for name in ("gaunt_log_u_grid", "gaunt_log_gamma2_grid", "gaunt_table",
                     "metal_bf_nu_grid", "metal_bf_logT_grid",
                     "metal_bf_z_arr", "metal_bf_charge_arr"):
            np.testing.assert_array_equal(
                np.asarray(getattr(loaded, name)),
                np.asarray(getattr(synth_data, name)), err_msg=name)

        a, b = loaded.chem_eq_data, synth_data.chem_eq_data
        assert a.n_molecules == b.n_molecules
        for name in ("log_T_grid", "ionization_energies", "partition_func_values",
                     "pf_orig_t", "pf_orig_u", "pf_orig_h", "pf_orig_z",
                     "pf_orig_n", "log_T_h", "mol_atoms_array", "mol_charges",
                     "mol_n_atoms", "mol_log_K_values", "mol_log_K_z",
                     "mol_partition_func_values", "mol_partition_func_z",
                     "mol_atom_consume"):
            np.testing.assert_array_equal(
                np.asarray(getattr(a, name)), np.asarray(getattr(b, name)),
                err_msg=name)

    def test_a_reloaded_file_produces_an_identical_spectrum(
            self, synth_data, linelist_data, tiny_atm, abundances,
            wavelengths_cm, tmp_path):
        """The round trip must be exact, not merely close."""
        path = tmp_path / "synthesis_data.npz"
        save_synthesis_data(synth_data, str(path))
        loaded = load_synthesis_data(str(path))
        kw = dict(wavelengths_cm=wavelengths_cm, abundances=abundances,
                  vmic_cm_s=1e5, linelist_data=linelist_data, **_atm_args(tiny_atm))
        f_a, c_a = synthesize_jit(data=synth_data, **kw)
        f_b, c_b = synthesize_jit(data=loaded, **kw)
        np.testing.assert_array_equal(np.asarray(f_a), np.asarray(f_b))
        np.testing.assert_array_equal(np.asarray(c_a), np.asarray(c_b))


# ===========================================================================
# 1. Functional — preprocess_linelist (the synthesis.py one)
# ===========================================================================

class TestPreprocessLinelist:
    """Note: ``synthesis_preparation.py`` defines an *unrelated* function of the
    same name returning a different type.  They are not interchangeable."""

    def test_an_empty_linelist_gives_zero_lines(self):
        ld = preprocess_linelist([])
        assert isinstance(ld, LinelistData)
        assert ld.n_lines == 0
        assert ld.wl.shape == (0,)

    def test_scalar_fields_are_copied_from_the_lines(self, fe_line):
        ld = preprocess_linelist([fe_line])
        assert ld.n_lines == 1
        assert float(ld.wl[0]) == pytest.approx(fe_line.wl, rel=0)
        assert float(ld.log_gf[0]) == pytest.approx(fe_line.log_gf, rel=0)
        assert float(ld.E_lower[0]) == pytest.approx(fe_line.E_lower, rel=0)
        assert int(ld.species_Z[0]) == 26
        assert int(ld.species_charge[0]) == 0

    def test_without_chem_eq_data_molecules_are_unmatched(self, fe_line):
        """``mol_species_idx`` is -1 when no molecule table is supplied."""
        co = Line(wl=5000.5e-8, log_gf=-1.0, species=Species("CO"), E_lower=1.0,
                  gamma_rad=1e8, gamma_stark=0.0, vdW=(1e-7, -1.0))
        ld = preprocess_linelist([fe_line, co])
        np.testing.assert_array_equal(np.asarray(ld.mol_species_idx), [-1, -1])

    def test_with_chem_eq_data_a_molecular_line_is_matched(self, fe_line,
                                                           synth_data):
        co = Line(wl=5000.5e-8, log_gf=-1.0, species=Species("CO"), E_lower=1.0,
                  gamma_rad=1e8, gamma_stark=0.0, vdW=(1e-7, -1.0))
        ld = preprocess_linelist([fe_line, co], chem_eq_data=synth_data.chem_eq_data)
        idx = np.asarray(ld.mol_species_idx)
        assert idx[0] == -1, "an atomic line has no molecular index"
        assert idx[1] >= 0, "CO should match an entry in the molecule table"

    def test_an_unknown_molecule_stays_unmatched(self, synth_data):
        """A molecule absent from the equilibrium table must not match anything."""
        exotic = Line(wl=5000.5e-8, log_gf=-1.0, species=Species("U2"),
                      E_lower=1.0, gamma_rad=1e8, gamma_stark=0.0,
                      vdW=(1e-7, -1.0))
        ld = preprocess_linelist([exotic], chem_eq_data=synth_data.chem_eq_data)
        assert int(np.asarray(ld.mol_species_idx)[0]) == -1

    def test_abo_sigma_is_rescaled_but_a_plain_gamma_is_not(self):
        """ABO lines (alpha >= 0) get the sigma -> sigma*C conversion; others don't."""
        abo = Line(wl=5000.0e-8, log_gf=-1.0, species=Species("Fe I"), E_lower=1.0,
                   gamma_rad=1e8, gamma_stark=0.0, vdW=(2.4e-8, 0.25))
        plain = Line(wl=5000.0e-8, log_gf=-1.0, species=Species("Fe I"),
                     E_lower=1.0, gamma_rad=1e8, gamma_stark=0.0,
                     vdW=(2.4e-8, -1.0))
        ld = preprocess_linelist([abo, plain])
        sigma = np.asarray(ld.vdW_sigma)
        assert sigma[0] != 2.4e-8, "the ABO sigma must be rescaled"
        assert sigma[1] == pytest.approx(2.4e-8, rel=1e-15), \
            "a non-ABO vdW parameter must pass through untouched"

    def test_wavelength_caches_are_absent_unless_requested(self, fe_line):
        ld = preprocess_linelist([fe_line])
        assert ld.wl_np_cached is None
        assert ld.wls_np_cached is None
        assert ld.wl_spacing_cached is None
        assert ld.cntm_wl_np_cached is None

    def test_wavelength_caches_are_built_when_a_grid_is_given(self, fe_line):
        wls = np.array((5000.0 + 0.01 * np.arange(11)) * 1e-8)
        ld = preprocess_linelist([fe_line], wavelengths_cm=wls)
        np.testing.assert_array_equal(ld.wl_np_cached, wls)
        np.testing.assert_array_equal(ld.wls_np_cached, [fe_line.wl])
        assert ld.wl_spacing_cached == pytest.approx(1e-10, rel=1e-9)
        # 1 Å coarse continuum grid, one step either side of the window
        assert ld.cntm_wl_np_cached[0] == pytest.approx(wls[0] - 1e-8, rel=1e-12)
        assert ld.cntm_wl_np_cached[-1] >= wls[-1]

    def test_a_single_point_grid_gets_the_default_spacing(self, fe_line):
        """``np.diff`` is empty there, so the median would be NaN."""
        ld = preprocess_linelist([fe_line], wavelengths_cm=np.array([5e-5]))
        assert ld.wl_spacing_cached == 5e-9


# ===========================================================================
# 1./2. precompute_atmosphere and the fused bucket path
# ===========================================================================

@pytest.fixture(scope="module")
def precomputed(tiny_atm, abundances, wavelengths_cm, synth_data, linelist_data):
    return precompute_atmosphere(
        wavelengths_cm, jnp.array(tiny_atm.T), jnp.array(tiny_atm.n_total),
        jnp.array(tiny_atm.ne), jnp.array(tiny_atm.z),
        jnp.array(tiny_atm.log_tau_ref), abundances, 1e5, synth_data,
        linelist_data)


@pytest.mark.slow
class TestPrecomputeAtmosphere:
    """``precompute_atmosphere`` builds a large XLA graph; ~1 min to compile."""

    def test_the_layer_quantities_are_sane(self, precomputed, tiny_atm):
        n = tiny_atm.n_layers
        assert precomputed.T_layers.shape == (n,)
        assert precomputed.ne_all.shape == (n,)
        assert np.all(np.asarray(precomputed.ne_all) > 0)
        assert np.all(np.asarray(precomputed.nH_I_all) > 0)
        assert np.all(np.asarray(precomputed.alpha_ref_all) > 0)
        assert np.all(np.isfinite(np.asarray(precomputed.alpha_cntm_all)))
        assert np.all(np.asarray(precomputed.S_all) > 0)

    def test_the_partition_function_tables_are_built(self, precomputed):
        assert precomputed.U_atomic_table is not None
        assert precomputed.U_mol_table is not None
        assert np.all(np.asarray(precomputed.U_atomic_table) > 0)

    def test_line_buckets_and_hydrogen_stark_profiles_are_prepared(
            self, precomputed):
        assert precomputed.bucket_geometry, "one Fe line should occupy one bucket"
        assert precomputed.h_stark_precomp, \
            "Hbeta's far wing reaches 5000 A, so a Stark profile is precomputed"
        for bg in precomputed.bucket_geometry:
            assert bg.n_b >= 1
            assert bg.W >= 1

    def test_the_electron_density_agrees_with_the_python_path(
            self, precomputed, tiny_atm, wavelengths_cm, julia_ref):
        """Cross-check against the non-JIT chemical-equilibrium solve.

        Both run a Newton solve to a 1e-8 residual but from different Picard
        starting points and with different convergence bookkeeping, so they
        agree to ~1e-3 rather than to round-off.  That is the size of the
        disagreement between Korg.px's two equilibrium entry points, and it is
        recorded here rather than asserted away.

        This used to reach the other solver through ``synthesize_spectrum``,
        which has been deleted; it calls ``chemical_equilibrium_all_layers``
        directly instead, which is the function ``synthesize_spectrum`` called
        and is the whole of what was being compared.
        """
        from korg.abundances import A_X_to_absolute
        from korg.data_loader import default_chem_eq_data, default_mol_species
        from korg.statmech import chemical_equilibrium_all_layers

        ne, _, _ = chemical_equilibrium_all_layers(
            np.asarray(tiny_atm.T), np.asarray(tiny_atm.n_total),
            np.asarray(tiny_atm.ne), A_X_to_absolute(np.array(julia_ref["A_X"])),
            default_chem_eq_data, default_mol_species)
        np.testing.assert_allclose(np.asarray(precomputed.ne_all),
                                   np.asarray(ne), rtol=2e-3)


@pytest.mark.slow
class TestSynthesizeJitPrecomputedPath:

    @pytest.fixture(scope="class")
    def fluxes(self, precomputed, tiny_atm, abundances, wavelengths_cm,
               synth_data, linelist_data):
        kw = dict(wavelengths_cm=wavelengths_cm, abundances=abundances,
                  vmic_cm_s=1e5, data=synth_data, linelist_data=linelist_data,
                  **_atm_args(tiny_atm))
        with_pc = synthesize_jit(precomputed_atm=precomputed, **kw)
        without = synthesize_jit(**kw)
        return {"precomputed": with_pc, "plain": without}

    def test_both_paths_produce_finite_positive_flux(self, fluxes):
        for name, (flux, cntm) in fluxes.items():
            assert np.all(np.isfinite(np.asarray(flux))), name
            assert np.all(np.asarray(flux) > 0), name
            assert np.all(np.asarray(cntm) > 0), name

    def test_the_precomputed_path_agrees_with_the_plain_one(self, fluxes):
        """High-precision: two entirely different kernels, same physics.

        The precomputed path uses table-lookup partition functions, a fused
        bucketed Voigt kernel and precomputed 1-D Stark profiles; the plain
        path recomputes everything.  They agree to ~1e-10, which is the
        table-interpolation error, not an algorithmic difference.
        """
        f_pc, c_pc = fluxes["precomputed"]
        f_pl, c_pl = fluxes["plain"]
        np.testing.assert_allclose(np.asarray(f_pc), np.asarray(f_pl), rtol=1e-8)
        np.testing.assert_allclose(np.asarray(c_pc), np.asarray(c_pl), rtol=1e-8)

    def test_the_line_shows_up_in_both_paths(self, fluxes):
        for name, (flux, cntm) in fluxes.items():
            depth = 1.0 - (np.asarray(flux) / np.asarray(cntm)).min()
            assert depth > 1e-3, f"{name}: the Fe I line left no absorption"

    def test_an_empty_linelist_short_circuits_the_voigt_stage(
            self, tiny_atm, abundances, wavelengths_cm, synth_data):
        """``n_lines == 0`` takes a separate branch that allocates zeros.

        ``synthesize_jit`` always includes hydrogen lines (it has no switch for
        them), and Hβ's far wing reaches 5000 Å, so flux is slightly *below*
        the continuum even with no atomic lines — but only by ~4e-4.
        """
        empty = preprocess_linelist([])
        flux, cntm = synthesize_jit(
            wavelengths_cm=wavelengths_cm, abundances=abundances, vmic_cm_s=1e5,
            data=synth_data, linelist_data=empty, **_atm_args(tiny_atm))
        f, c = np.asarray(flux), np.asarray(cntm)
        assert np.all(f <= c)
        np.testing.assert_allclose(f, c, rtol=1e-2)


@pytest.mark.slow
class TestSynthesizeJitAwayFrom5000Angstrom:
    """Exercises the branch that only fires when the window spans 5000 Å."""

    def test_a_window_containing_5000_uses_the_in_window_reference_opacity(
            self, tiny_atm, abundances, synth_data):
        wls = jnp.array((4999.5 + 0.01 * np.arange(101)) * 1e-8)
        ld = preprocess_linelist([], wavelengths_cm=np.asarray(wls))
        flux, cntm = synthesize_jit(
            wavelengths_cm=wls, abundances=abundances, vmic_cm_s=1e5,
            data=synth_data, linelist_data=ld, **_atm_args(tiny_atm))
        assert np.all(np.asarray(flux) > 0)

    def test_a_window_excluding_5000_still_gives_positive_flux(
            self, tiny_atm, abundances, synth_data):
        wls = jnp.array((6000.0 + 0.01 * np.arange(11)) * 1e-8)
        ld = preprocess_linelist([], wavelengths_cm=np.asarray(wls))
        flux, cntm = synthesize_jit(
            wavelengths_cm=wls, abundances=abundances, vmic_cm_s=1e5,
            data=synth_data, linelist_data=ld, **_atm_args(tiny_atm))
        assert np.all(np.isfinite(np.asarray(flux)))
        assert np.all(np.asarray(flux) > 0)

    def test_a_hydrogen_line_window_adds_stark_opacity(
            self, tiny_atm, abundances, synth_data):
        """4861 Å is Hβ, so the Stark branch of Phase 4.5 runs."""
        wls = jnp.array((4861.0 + 0.05 * np.arange(11)) * 1e-8)
        ld = preprocess_linelist([], wavelengths_cm=np.asarray(wls))
        flux, cntm = synthesize_jit(
            wavelengths_cm=wls, abundances=abundances, vmic_cm_s=1e5,
            data=synth_data, linelist_data=ld, **_atm_args(tiny_atm))
        assert np.all(np.isfinite(np.asarray(flux)))
        assert np.any(np.asarray(flux) < np.asarray(cntm)), \
            "Hbeta must absorb relative to the continuum"


# ===========================================================================
# 3./4. Autodiff and jit of the pieces that support it
# ===========================================================================

class TestAutodiffAndJit:

    def test_preprocess_linelist_cannot_be_traced(self, fe_line):
        """It reads ``line.wl`` from Python objects, so there is nothing to trace."""
        @jax.jit
        def f(x):
            return preprocess_linelist([fe_line]).wl.sum() * x

        # tracing succeeds (the linelist is static), but a traced *linelist*
        # cannot exist: Line is a plain dataclass of Python floats.
        assert np.isfinite(float(f(2.0)))

    @pytest.mark.slow
    @pytest.mark.parametrize("wrt", ["abundances", "temperature"])
    def test_synthesize_jit_is_not_differentiable_end_to_end(
            self, tiny_atm, abundances, wavelengths_cm, synth_data, wrt):
        """Pinned, not skipped, with the reason.

        Despite its name ``synthesize_jit`` is not a pure JAX function.  The
        reference-opacity stage converts the solved electron densities, layer
        temperatures and species number densities to host NumPy so it can call
        ``line_absorption`` on the built-in 5000 Å linelist, and Phase 4.5 does
        the same for the hydrogen lines.  Any traced input therefore reaches
        ``np.asarray`` on a tracer.

        Differentiating a synthesis end to end needs those two stages ported to
        JAX; until then this is a hard limit, not a coverage gap.  The
        *individual* kernels it calls (``blackbody``, the Voigt profile, the
        radiative transfer solvers) are differentiable and are tested as such.
        """
        empty = preprocess_linelist([])
        atm = _atm_args(tiny_atm)

        if wrt == "abundances":
            def f(scale):
                flux, _ = synthesize_jit(
                    wavelengths_cm=wavelengths_cm, abundances=abundances * scale,
                    vmic_cm_s=1e5, data=synth_data, linelist_data=empty, **atm)
                return jnp.sum(flux)
            arg = 1.0
        else:
            def f(T):
                flux, _ = synthesize_jit(
                    wavelengths_cm=wavelengths_cm, T_layers=T,
                    n_total_layers=atm["n_total_layers"],
                    ne_layers=atm["ne_layers"], z_layers=atm["z_layers"],
                    log_tau_ref=atm["log_tau_ref"], abundances=abundances,
                    vmic_cm_s=1e5, data=synth_data, linelist_data=empty)
                return jnp.sum(flux)
            arg = atm["T_layers"]

        with pytest.raises(jax.errors.TracerArrayConversionError):
            jax.grad(f)(arg)

    def test_load_synthesis_data_cannot_be_jitted(self, synth_data, tmp_path):
        """It opens a file and builds Python-side index lists."""
        path = tmp_path / "d.npz"
        save_synthesis_data(synth_data, str(path))

        @jax.jit
        def f(i):
            return load_synthesis_data(str(path)).gaunt_table[int(i), 0]

        with pytest.raises(jax.errors.ConcretizationTypeError):
            f(0)
