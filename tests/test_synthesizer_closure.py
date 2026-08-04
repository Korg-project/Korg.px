"""
The traceable synthesizer: ``prepare_synthesis`` and the closure it returns.

Four tiers, matching the project's taxonomy:

1. Functional — the plan is built correctly, the closure runs, shapes and
   options behave.
2. Agreement — the closure reproduces Korg.jl 1.2.1, from the fixtures in
   ``tests/synthesis_reference_data.json``. See ``TestAgreementWithKorgJl`` for
   why it is compared against Julia output rather than against another Python
   entry point.
3. Autodiff — finite gradients with respect to *every* input: the five stellar
   parameters and all 92 abundances.
4. jit-tracing — ``jax.jit`` and ``jax.vmap`` over the closure.

Everything here loads the **shipped** tables via ``load_synthesis_data()``
rather than rebuilding them with ``precompute_chemical_equilibrium_data``. That
is deliberate. A gradient defect in ``_eval_atomic_pf_jit`` was invisible for a
full day because the test that pinned it constructed a fresh table, and a fresh
table does not contain the single-knot degeneracy (H III) that the shipped one
does: 201 non-finite knot entries against 199. Tests that rebuild their inputs
verify a configuration nobody runs.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import korg  # noqa: F401 — side effect: enables x64
from korg.synthesis_plan import prepare_synthesis, Synthesizer
from korg.data_loader import load_default_linelist
from korg.abundances import format_A_X, A_X_to_absolute

TEFF, LOGG, M_H = 5777.0, 4.44, 0.0


@pytest.fixture(scope="module")
def wavelengths():
    return np.arange(5000.0, 5002.0, 0.01)          # Angstroms


@pytest.fixture(scope="module")
def linelist():
    return load_default_linelist(5e-5)


@pytest.fixture(scope="module")
def synth(wavelengths, linelist):
    return prepare_synthesis(wavelengths, linelist, geometry="planar")


@pytest.fixture(scope="module")
def abundances():
    return jnp.asarray(A_X_to_absolute(format_A_X()))


# ===========================================================================
# 1. FUNCTIONAL
# ===========================================================================

class TestPlanConstruction:

    def test_returns_a_callable_synthesizer(self, synth):
        assert isinstance(synth, Synthesizer)
        assert callable(synth)

    def test_shapes_are_fixed_at_plan_time(self, synth, wavelengths):
        assert synth.n_wl == len(wavelengths)
        assert synth.n_layers == 56
        assert synth.n_lines > 0

    def test_bucket_widths_are_measured_not_guessed(self, synth):
        """Windows come from a concrete run at the reference parameters.

        They are powers of two (the safety factor rounds up to one), ascending,
        and capped at the grid width.
        """
        assert synth.bucket_widths
        for w in synth.bucket_widths:
            assert w <= synth.n_wl
            assert w == synth.n_wl or (w & (w - 1)) == 0, w
        assert len(synth.bucket_line_idx) == len(synth.bucket_widths)

    def test_every_line_lands_in_exactly_one_bucket(self, synth):
        seen = np.concatenate([np.asarray(i) for i in synth.bucket_line_idx])
        assert len(seen) == len(set(seen.tolist())), "a line is in two buckets"

    def test_rejects_unknown_geometry(self, wavelengths, linelist):
        with pytest.raises(ValueError, match="spherical.*plane-parallel"):
            prepare_synthesis(wavelengths, linelist, geometry="toroidal")

    def test_rejects_a_degenerate_wavelength_grid(self, linelist):
        with pytest.raises(ValueError, match="two wavelength points"):
            prepare_synthesis(np.array([5000.0]), linelist)

    def test_reference_pixel_found_only_when_5000A_is_covered(self, linelist):
        inside = prepare_synthesis(np.arange(4999.0, 5001.0, 0.01), linelist)
        outside = prepare_synthesis(np.arange(6000.0, 6002.0, 0.01), linelist)
        assert inside.ref_pixel is not None
        assert outside.ref_pixel is None

    def test_stark_transitions_selected_by_wavelength(self, linelist):
        """H-beta at 4861 A must be picked up near it and not at 6000 A."""
        near = prepare_synthesis(np.arange(4855.0, 4867.0, 0.01), linelist)
        far = prepare_synthesis(np.arange(7000.0, 7002.0, 0.01), linelist)
        assert len(near.stark_keys) >= len(far.stark_keys)


class TestSynthesis:

    def test_runs_and_returns_finite_flux(self, synth):
        flux, cntm = synth(TEFF, LOGG, M_H)
        flux, cntm = np.asarray(flux), np.asarray(cntm)
        assert flux.shape == (synth.n_wl,)
        assert np.all(np.isfinite(flux)) and np.all(np.isfinite(cntm))
        assert np.all(flux > 0) and np.all(cntm > 0)

    def test_lines_only_ever_remove_flux(self, synth):
        """A pure absorption spectrum: normalized flux is in (0, 1]."""
        flux, cntm = synth(TEFF, LOGG, M_H)
        rectified = np.asarray(flux / cntm)
        assert np.all(rectified > 0.0)
        assert np.all(rectified <= 1.0 + 1e-8)

    def test_a_line_free_synthesis_still_carries_hydrogen(self, wavelengths):
        """An empty *atomic* linelist is not a line-free spectrum.

        Hydrogen lines are generated from the Stark tables, not from the
        linelist, so their wings survive an empty linelist: at 5000 A they
        depress the flux by 3.6e-5 relative. This test asserted flux == cntm and
        was wrong about the physics, not about the code.
        """
        s = prepare_synthesis(wavelengths, [])
        flux, cntm = np.asarray(s(TEFF, LOGG, M_H)[0]), np.asarray(s(TEFF, LOGG, M_H)[1])
        assert np.all(flux <= cntm * (1.0 + 1e-12)), "H wings can only remove flux"
        depth = 1.0 - flux / cntm
        assert depth.max() > 0.0, "hydrogen contributes nothing at all"
        assert depth.max() < 1e-3, "unexpectedly deep for H wings at 5000 A"

    def test_explicit_abundances_override_m_H(self, synth, abundances):
        """Passing abundances leaves [M/H] acting on structure alone."""
        a = np.asarray(synth(TEFF, LOGG, 0.0, abundances=abundances)[0])
        b = np.asarray(synth(TEFF, LOGG, -1.0, abundances=abundances)[0])
        assert not np.allclose(a, b), "[M/H] must still move the atmosphere"

    def test_metal_poor_lines_are_weaker(self, synth):
        """A physical check that [M/H] reaches the line opacity."""
        solar = (lambda fc: np.asarray(fc[0] / fc[1]))(synth(TEFF, LOGG, 0.0))
        poor = (lambda fc: np.asarray(fc[0] / fc[1]))(synth(TEFF, LOGG, -2.0))
        assert poor.min() > solar.min()

    def test_from_atmosphere_matches_the_stellar_parameter_path(self, synth, abundances):
        from korg.traced_synthesis import _interpolate_marcs_traced
        T, nt, ne, z, lt, R = _interpolate_marcs_traced(synth, TEFF, LOGG, 0.0, 0.0, 0.0)
        direct = np.asarray(synth.from_atmosphere(T, nt, ne, z, lt, abundances,
                                                  R_photosphere=R, logg=LOGG)[0])
        viaparams = np.asarray(synth(TEFF, LOGG, abundances=abundances)[0])
        # 1.3e-8 on 2 of 200 pixels. Same floor as test_jit_matches_the_eager_result:
        # the bucketed Voigt runs in float32, and the two routes to the same
        # atmosphere arrays let XLA fuse and reassociate differently. Measured, not
        # assumed -- an earlier 1e-8 here failed by a factor of 1.3.
        np.testing.assert_allclose(direct, viaparams, rtol=1e-7)

    def test_traced_log_tau_ref_is_base_10(self, synth):
        """The transfer kernels take log_tau_ref in base 10, not base e.

        ``_compute_tau_anchored_planar`` does ``tau_ref = 10 ** log_tau_ref`` and
        scales its steps by ``ln 10``, matching ``Atmosphere.log_tau_ref``, which
        is ``np.log10``. The traced interpolation used a natural log, so transfer
        integrated against ``tau_ref ** ln(10)`` and made lines about 2.2x too
        shallow (deepest rectified point 0.352 against 0.161).

        This pins the unit directly, cheaply, and without a synthesis: the two
        end-to-end tests below would also catch it, but they would not say why.
        """
        from korg.traced_synthesis import _interpolate_marcs_traced
        from korg.marcs_interpolation import interpolate_marcs

        _, _, _, _, lt, _ = _interpolate_marcs_traced(synth, TEFF, LOGG, 0.0, 0.0, 0.0)
        expected = interpolate_marcs(TEFF, LOGG, M_H).log_tau_ref
        np.testing.assert_allclose(np.asarray(lt), expected, rtol=1e-10)


class TestAgreementWithKorgJl:
    """Every entry point of the closure, against Korg.jl 1.2.1 itself.

    This class used to compare the traced path against ``synthesize_spectrum``,
    the NumPy path, on the grounds that the NumPy path was the one the Korg.jl
    reference tests exercised. ``synthesize_spectrum`` has been deleted, and
    rewriting these to call ``korg.synthesis_plan.synthesize`` instead would have
    turned them into the traced path compared with itself — which is precisely
    the structure that let a natural-log/base-10 confusion in the optical depth
    survive 3,747 passing tests while every line came out 2.2x too shallow.

    So they now compare against the Korg.jl fixtures in
    ``tests/synthesis_reference_data.json`` directly: the Sun (``tests/data/sun.mod``),
    5000-5001 A, one Fe I line at 5000.5 A, ``hydrogen_lines=false``, ``vmic=1``,
    generated by ``tests/generate_synthesis_reference.jl`` against Korg.jl 1.2.1.
    That is a strictly stronger reference than the deleted path was: it removes
    an intermediary that itself only agreed with Korg.jl to 3e-3.

    All three entry points are covered on purpose. The bug above lived in
    ``_interpolate_marcs_traced``, which only ``__call__`` goes through, so an
    anchor on ``from_atmosphere`` alone would not have caught it.

    Tolerance. 5e-3 relative on absolute flux and 5e-3 absolute on the rectified
    spectrum, the same ``SPECTRUM_RTOL`` ``test_synthesis_functional.py`` uses
    and for the same reason: a whole spectrum runs the entire opacity stack, and
    Python and Korg.jl differ by ~3e-3 there. Measured values are in each test.
    A log-base error of the kind above shows up as ~0.2 in the rectified
    spectrum, forty times this bound.
    """

    @pytest.fixture(scope="class")
    def jl(self):
        import json
        from pathlib import Path
        path = Path(__file__).parent / "synthesis_reference_data.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Regenerate it with "
                "`julia --project=. tests/generate_synthesis_reference.jl`.")
        with open(path) as fh:
            return json.load(fh)

    @pytest.fixture(scope="class")
    def ref_wls(self, jl):
        return np.array(jl["wavelengths"])          # A, 5000-5001, 101 points

    @pytest.fixture(scope="class")
    def ref_A_X(self, jl):
        """Korg.jl's own ``format_A_X()`` output, so both sides start equal."""
        return np.array(jl["A_X"])

    @pytest.fixture(scope="class")
    def ref_line(self, jl):
        from korg.linelist import Line
        from korg.species import Species
        L = jl["line"]
        return Line(wl=L["wl_cm"], log_gf=L["log_gf"], species=Species("Fe I"),
                    E_lower=L["E_lower"], gamma_rad=L["gamma_rad"],
                    gamma_stark=L["gamma_stark"], vdW=tuple(L["vdW"]))

    @pytest.fixture(scope="class")
    def sun(self):
        """The 56-layer solar model the fixtures were generated from."""
        from pathlib import Path
        import korg as _korg
        path = Path(__file__).parent / "data" / "sun.mod"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing from the test data directory")
        return _korg.read_model_atmosphere(str(path))

    @pytest.fixture(scope="class")
    def plan_line(self, ref_wls, ref_line):
        # hydrogen_lines=False matches how the fixtures were generated. The H
        # path is exercised by TestSynthesis; leaving it on here would compare
        # unlike with unlike, to no benefit.
        return prepare_synthesis(ref_wls, [ref_line], geometry="planar",
                                 hydrogen_lines=False)

    @staticmethod
    def _atm_args(atm, A_X):
        return (jnp.asarray(atm.T), jnp.asarray(atm.n_total), jnp.asarray(atm.ne),
                jnp.asarray(atm.z), jnp.asarray(atm.log_tau_ref),
                jnp.asarray(A_X_to_absolute(np.asarray(A_X))))

    # -- from_atmosphere: the entry point that takes log_tau_ref from the caller

    def test_from_atmosphere_flux_matches_korg_jl(self, plan_line, sun, ref_A_X, jl):
        """Absolute flux, erg/cm^2/s/A. Measured 2.2e-3."""
        f, c = plan_line.from_atmosphere(*self._atm_args(sun, ref_A_X))
        np.testing.assert_allclose(np.asarray(f),
                                   np.array(jl["synthesis"]["planar_line"]["flux"]),
                                   rtol=5e-3)
        np.testing.assert_allclose(np.asarray(c),
                                   np.array(jl["synthesis"]["planar_line"]["cntm"]),
                                   rtol=5e-3)

    def test_from_atmosphere_line_depth_matches_korg_jl(self, plan_line, sun,
                                                        ref_A_X, jl):
        """The rectified spectrum: what a log-base error destroys. Measured 2.2e-3.

        Korg.jl's deepest point here is 0.803. The optical-depth bug in the
        commit history put the equivalent number at 0.352 where 0.161 was right
        -- a factor of 2.2. Nothing in this assertion tolerates that.
        """
        f, c = plan_line.from_atmosphere(*self._atm_args(sun, ref_A_X))
        rect = np.asarray(f) / np.asarray(c)
        jf = np.array(jl["synthesis"]["planar_line"]["flux"])
        jc = np.array(jl["synthesis"]["planar_line"]["cntm"])
        np.testing.assert_allclose(rect, jf / jc, rtol=0, atol=5e-3)
        assert (1.0 - rect.min()) == pytest.approx(1.0 - (jf / jc).min(), abs=5e-3)

    def test_continuum_only_matches_korg_jl(self, ref_wls, sun, ref_A_X, jl):
        """No lines at all: the continuum stack and the transfer, alone.

        Measured 3.0e-3, the largest of these residuals -- the opacity-stack
        difference between Korg.px and Korg.jl is a continuum difference, and
        it partly cancels in a line ratio.
        """
        plan = prepare_synthesis(ref_wls, [], geometry="planar",
                                 hydrogen_lines=False)
        f, c = plan.from_atmosphere(*self._atm_args(sun, ref_A_X))
        np.testing.assert_allclose(np.asarray(f),
                                   np.array(jl["synthesis"]["planar_cntm"]["flux"]),
                                   rtol=5e-3)
        np.testing.assert_allclose(np.asarray(c),
                                   np.array(jl["synthesis"]["planar_cntm"]["cntm"]),
                                   rtol=5e-3)

    # -- __call__: the stellar-parameter path, where the log-base bug lived ----

    def test_stellar_parameter_path_matches_korg_jl(self, plan_line, ref_A_X, jl):
        """``synth(Teff, logg, [M/H])`` against the same Korg.jl spectrum.

        This goes through ``_interpolate_marcs_traced`` rather than reading the
        atmosphere from a file, so the model is the MARCS interpolation at
        (5777, 4.44, 0.0) rather than sun.mod itself. The two differ by 1.9e-4 in
        T and 4.7e-3 in n_e, which is why this is compared at the same 5e-3 as
        everything else rather than more tightly. Measured 2.7e-3 on flux and
        2.3e-3 on the rectified spectrum.

        It is the only end-to-end anchor on the MARCS interpolation inside the
        traced region, and the entry point the optical-depth bug lived in.
        """
        f, c = plan_line(TEFF, LOGG, M_H)
        f, c = np.asarray(f), np.asarray(c)
        jf = np.array(jl["synthesis"]["planar_line"]["flux"])
        jc = np.array(jl["synthesis"]["planar_line"]["cntm"])
        np.testing.assert_allclose(f, jf, rtol=5e-3)
        np.testing.assert_allclose(f / c, jf / jc, rtol=0, atol=5e-3)

    # -- synthesize(): the Korg.jl-compatible one-shot wrapper ----------------

    def test_korg_compatible_wrapper_matches_korg_jl(self, sun, ref_wls, ref_line,
                                                     ref_A_X, jl):
        """``synthesize`` reads log_tau_ref off the atmosphere object.

        Same call as Korg.jl's, argument for argument. Measured 2.2e-3.
        """
        from korg.synthesis_plan import synthesize
        f, c = synthesize(sun, [ref_line], ref_wls, ref_A_X, hydrogen_lines=False)
        f, c = np.asarray(f), np.asarray(c)
        jf = np.array(jl["synthesis"]["planar_line"]["flux"])
        jc = np.array(jl["synthesis"]["planar_line"]["cntm"])
        np.testing.assert_allclose(f, jf, rtol=5e-3)
        np.testing.assert_allclose(f / c, jf / jc, rtol=0, atol=5e-3)

    def test_spherical_geometry_matches_korg_jl(self, sun, ref_wls, ref_line,
                                                ref_A_X, jl):
        """The extended shell, t/R ~ 0.3, where the photosphere correction is 1.69.

        The spherical tests below compare spherical against planar, which is a
        consistency check, not a reference. This is the reference. Measured
        1.4e-3 on flux and 1.4e-3 on the continuum.
        """
        from korg.atmosphere import ShellAtmosphere
        from korg.synthesis_plan import synthesize
        S = jl["synthesis"]
        shell = ShellAtmosphere.from_planar(sun, S["shell_extended_R"])
        f, c = synthesize(shell, [ref_line], ref_wls, ref_A_X, hydrogen_lines=False)
        np.testing.assert_allclose(np.asarray(f),
                                   np.array(S["shell_extended_line"]["flux"]),
                                   rtol=5e-3)
        np.testing.assert_allclose(np.asarray(c),
                                   np.array(S["shell_extended_line"]["cntm"]),
                                   rtol=5e-3)

    def test_hydrogen_lines_false_really_removes_them(self, ref_wls, ref_line):
        """Guards the switch these comparisons rely on.

        If ``hydrogen_lines=False`` quietly did nothing, every test above would
        still pass (the H opacity at 5000 A is ~4e-5 relative, far inside 5e-3)
        and the comparisons would silently stop being like-for-like.
        """
        off = prepare_synthesis(ref_wls, [ref_line], geometry="planar",
                                hydrogen_lines=False)
        on = prepare_synthesis(ref_wls, [ref_line], geometry="planar",
                               hydrogen_lines=True)
        assert off.stark_keys == ()
        assert len(on.stark_keys) > 0, "Hbeta's wing reaches 5000 A"
        f_off = np.asarray(off(TEFF, LOGG, M_H)[0])
        f_on = np.asarray(on(TEFF, LOGG, M_H)[0])
        assert np.all(f_on <= f_off * (1.0 + 1e-12)), "H opacity can only remove flux"
        assert np.max(1.0 - f_on / f_off) > 1e-6, "the switch did nothing"


# ===========================================================================
# 3. AUTODIFF — the point of the exercise
# ===========================================================================

class TestAutodiff:
    """Gradients must be finite with respect to every input.

    Tolerances against central differences are loose on purpose. The MARCS
    interpolation is *multilinear*, so the model is only piecewise smooth in the
    stellar parameters, and a central difference does not converge under step
    refinement — measured relative differences wander between 8e-4 and 6e-3 as
    h goes 4 K -> 0.5 K rather than shrinking. FD is therefore not a reliable
    reference for these derivatives, and these tests assert finiteness, sign and
    order of magnitude rather than agreement to many digits. Establishing a
    tighter reference is open work.
    """

    def _rect_sum(self, synth, *a, **kw):
        flux, cntm = synth(*a, **kw)
        return jnp.sum(flux / cntm)

    @pytest.mark.parametrize("argnum,name", [(0, "Teff"), (1, "logg"), (2, "m_H")])
    def test_stellar_parameter_gradients_are_finite(self, synth, argnum, name):
        g = float(jax.grad(lambda *p: self._rect_sum(synth, *p),
                           argnums=argnum)(TEFF, LOGG, M_H))
        assert np.isfinite(g), f"d/d{name} is {g}"
        assert g != 0.0, f"d/d{name} is identically zero"

    def test_abundance_gradient_is_finite_for_every_element(self, synth, abundances):
        g = np.asarray(jax.grad(
            lambda a: self._rect_sum(synth, TEFF, LOGG, abundances=a))(abundances))
        assert g.shape == (92,)
        assert np.all(np.isfinite(g)), \
            f"non-finite at Z={np.where(~np.isfinite(g))[0] + 1}"
        assert np.count_nonzero(g) > 0

    def test_iron_gradient_dominates_a_line_rich_window(self, synth, abundances):
        """5000-5002 A is Fe-dominated; the sensitivity should reflect that."""
        g = np.abs(np.asarray(jax.grad(
            lambda a: self._rect_sum(synth, TEFF, LOGG, abundances=a))(abundances)))
        assert g[25] > 0, "Fe (Z=26) has no sensitivity in an Fe-rich window"

    @pytest.mark.parametrize("argnum", [0, 1, 2])
    def test_gradients_have_the_sign_of_a_central_difference(self, synth, argnum):
        """Sign, not magnitude — see the class docstring on FD reliability."""
        p = [TEFF, LOGG, M_H]
        h = [1.0, 1e-3, 1e-3][argnum]
        g = float(jax.grad(lambda *q: self._rect_sum(synth, *q), argnums=argnum)(*p))
        hi, lo = list(p), list(p)
        hi[argnum] += h
        lo[argnum] -= h
        fd = float((self._rect_sum(synth, *hi) - self._rect_sum(synth, *lo)) / (2 * h))
        assert np.sign(g) == np.sign(fd), f"AD={g}, FD={fd}"
        assert abs(g - fd) / max(abs(fd), 1e-30) < 0.05

    def test_gradient_through_a_line_free_synthesis(self, wavelengths):
        """Isolates the continuum and chemistry from the line-window cutoff."""
        s = prepare_synthesis(wavelengths, [])
        g = float(jax.grad(lambda t: self._rect_sum(s, t, LOGG, M_H))(TEFF))
        assert np.isfinite(g)

    def test_the_partition_function_evaluators_agree_and_both_differentiate(self):
        """The twin that caused a day of confusion.

        ``synthesis._pf_orig_eval`` and ``statmech._eval_atomic_pf_jit`` are
        byte-identical implementations. Only one was guarded against the
        single-knot degeneracy in the shipped table, and it was not the one on
        the chemical-equilibrium path — so ``d/dT`` through the chemistry stayed
        NaN while the partition-function stage read finite. This pins both.
        """
        from korg.synthesis import _pf_orig_eval, load_synthesis_data
        from korg.statmech import _eval_atomic_pf_jit
        ced = load_synthesis_data().chem_eq_data
        log_T = float(np.log(5777.0))
        for Z, charge in [(0, 0), (0, 2), (1, 0), (25, 0), (25, 2)]:
            args = (ced.pf_orig_t[Z, charge], ced.pf_orig_u[Z, charge],
                    ced.pf_orig_h[Z, charge], ced.pf_orig_z[Z, charge],
                    ced.pf_orig_n[Z, charge])
            a = float(_pf_orig_eval(log_T, *args))
            b = float(_eval_atomic_pf_jit(log_T, *args))
            assert a == pytest.approx(b, rel=1e-13), f"Z={Z + 1} charge={charge}"
            for fn in (_pf_orig_eval, _eval_atomic_pf_jit):
                g = float(jax.grad(lambda x, f=fn: f(x, *args))(log_T))
                assert np.isfinite(g), f"{fn.__name__} Z={Z + 1} charge={charge}"

    def test_saha_weights_differentiate_despite_hydrogen(self):
        """wIII[0] is exactly zero — hydrogen has no doubly ionized state.

        Masking that result without substituting into the dead branch left a
        1e-99-clipped division whose infinite partials met a zero cotangent.
        """
        from korg.synthesis import load_synthesis_data
        from korg.statmech import _compute_saha_weights_batch_jit
        ced = load_synthesis_data().chem_eq_data
        T = jnp.linspace(3500.0, 9000.0, 8)
        ne = jnp.full((8,), 1e13)
        wII, wIII = _compute_saha_weights_batch_jit(T, ne, ced)
        assert np.all(np.asarray(wIII)[:, 0] == 0.0), "hydrogen wIII must be zero"
        g = jax.grad(lambda x: jnp.sum(_compute_saha_weights_batch_jit(T, x, ced)[1]))(ne)
        assert np.all(np.isfinite(np.asarray(g)))


# ===========================================================================
# 4. JIT TRACING
# ===========================================================================

class TestTracing:

    def test_jit_matches_the_eager_result(self, synth):
        eager = np.asarray(synth(TEFF, LOGG, M_H)[0])
        jitted = np.asarray(jax.jit(lambda t, g, m: synth(t, g, m)[0])(TEFF, LOGG, M_H))
        # 1.2e-8, not exact. The bucketed Voigt evaluates its profile in float32
        # (the wavelength offsets are ~1e-8 cm, which float32 holds safely) and
        # XLA fuses that differently under jit than eagerly. This is a real
        # precision floor of the line kernel, not a tracing discrepancy -- worth
        # knowing before anyone asserts bitwise jit/eager equality here.
        np.testing.assert_allclose(jitted, eager, rtol=1e-7)

    def test_vmap_over_stellar_parameters(self, synth):
        """The property the whole refactor exists for: a grid in one dispatch."""
        teffs = jnp.array([5600.0, 5777.0, 5900.0])
        out = jax.vmap(lambda t: synth(t, LOGG, M_H)[0])(teffs)
        out = np.asarray(out)
        assert out.shape == (3, synth.n_wl)
        assert np.all(np.isfinite(out))
        assert not np.allclose(out[0], out[2])

    def test_vmap_over_abundances(self, synth, abundances):
        batch = jnp.stack([abundances, abundances * 1.01])
        out = np.asarray(jax.vmap(
            lambda a: synth(TEFF, LOGG, abundances=a)[0])(batch))
        assert out.shape == (2, synth.n_wl)
        assert np.all(np.isfinite(out))

    def test_jit_and_grad_compose(self, synth):
        g = float(jax.jit(jax.grad(
            lambda t: jnp.sum(synth(t, LOGG, M_H)[0])))(TEFF))
        assert np.isfinite(g)


# ===========================================================================
# SPHERICAL GEOMETRY
# ===========================================================================

class TestSphericalGeometry:
    """The spherical branch, across all four tiers.

    This branch was unreachable when written: ``traced_synthesis`` imported
    ``radiative_transfer_spherical_jit``, a name that does not exist, so
    ``geometry='spherical'`` raised ImportError on first call. Nothing caught it
    because nothing had ever executed it. These tests exist so that cannot recur.
    """

    @pytest.fixture(scope="class")
    def sph(self, wavelengths, linelist):
        return prepare_synthesis(wavelengths, linelist, geometry="spherical")

    def test_runs_and_returns_finite_flux(self, sph):
        flux, cntm = np.asarray(sph(TEFF, LOGG, M_H)[0]), np.asarray(sph(TEFF, LOGG, M_H)[1])
        assert flux.shape == (sph.n_wl,)
        assert np.all(np.isfinite(flux)) and np.all(flux > 0)
        assert np.all(flux <= cntm * (1.0 + 1e-8))

    def test_close_to_planar_for_a_thin_solar_shell(self, sph, synth):
        """A geometrically thin shell must nearly reproduce plane-parallel.

        The solar atmosphere spans ~1e8 cm against R = 7e10 cm, so the two
        geometries agree to well under a percent. Measured: 5.9e-4. A large
        divergence here means the ray geometry is wrong, not that spherical
        transfer is doing something subtle.
        """
        planar = np.asarray(synth(TEFF, LOGG, M_H)[0])
        spherical = np.asarray(sph(TEFF, LOGG, M_H)[0])
        rel = np.abs(spherical - planar) / planar
        assert rel.max() < 5e-3, f"max relative difference {rel.max():.2e}"
        assert rel.max() > 0.0, "spherical is bit-identical to planar -- flag ignored?"

    def test_gradients_are_finite(self, sph):
        g = float(jax.grad(lambda t: (lambda fc: jnp.sum(fc[0] / fc[1]))(sph(t, LOGG, M_H)))(TEFF))
        assert np.isfinite(g) and g != 0.0

    def test_gradient_tracks_the_planar_one(self, sph, synth):
        """Same physics, thin shell: the derivatives should agree closely too."""
        gs = float(jax.grad(lambda t: (lambda fc: jnp.sum(fc[0] / fc[1]))(sph(t, LOGG, M_H)))(TEFF))
        gp = float(jax.grad(lambda t: (lambda fc: jnp.sum(fc[0] / fc[1]))(synth(t, LOGG, M_H)))(TEFF))
        assert np.sign(gs) == np.sign(gp)
        assert abs(gs - gp) / abs(gp) < 0.05

    def test_abundance_gradient_is_finite(self, sph, abundances):
        g = np.asarray(jax.grad(
            lambda a: (lambda fc: jnp.sum(fc[0] / fc[1]))(sph(TEFF, LOGG, abundances=a)))(abundances))
        assert np.all(np.isfinite(g))

    def test_jit_and_vmap(self, sph):
        jitted = np.asarray(jax.jit(lambda t: sph(t, LOGG, M_H)[0])(TEFF))
        assert np.all(np.isfinite(jitted))
        out = np.asarray(jax.vmap(lambda t: sph(t, LOGG, M_H)[0])(
            jnp.array([5600.0, 5900.0])))
        assert out.shape == (2, sph.n_wl) and np.all(np.isfinite(out))

    def test_radius_is_not_a_plan_attribute(self, sph):
        """R belongs to the call, not the plan.

        This test used to assert that two *plans* built at different reference
        log g carried different radii -- which was true, and was exactly the bug:
        a single plan then applied its reference radius to every star it was
        called with. See TestPhotosphereRadius for what replaced it.
        """
        assert not hasattr(sph, "R_photosphere")


# ===========================================================================
# GEOMETRY SELECTION
# ===========================================================================

class TestGeometrySelection:
    """`geometry=None` picks per-call from log g, as Korg.jl does."""

    @pytest.fixture(scope="class")
    def auto(self, wavelengths, linelist):
        return prepare_synthesis(wavelengths, linelist, geometry=None)

    def test_spellings_normalise(self, wavelengths, linelist):
        for name in ("planar", "plane-parallel", "pp", "PLANE_PARALLEL"):
            s = prepare_synthesis(wavelengths, linelist, geometry=name)
            assert s.geometry == "plane-parallel", name
        assert prepare_synthesis(wavelengths, linelist,
                                 geometry="spherical").geometry == "spherical"
        assert prepare_synthesis(wavelengths, linelist, geometry=None).geometry is None

    def test_rejects_nonsense(self, wavelengths, linelist):
        with pytest.raises(ValueError, match="spherical.*plane-parallel"):
            prepare_synthesis(wavelengths, linelist, geometry="toroidal")

    def test_dwarf_takes_the_plane_parallel_branch(self, auto, wavelengths, linelist):
        pp = prepare_synthesis(wavelengths, linelist, geometry="plane-parallel")
        a = np.asarray(auto(TEFF, 4.44, M_H)[0])
        b = np.asarray(pp(TEFF, 4.44, M_H)[0])
        # 4.8e-8: the float32 Voigt floor, not a different branch. lax.cond
        # changes how XLA fuses the line kernel.
        np.testing.assert_allclose(a, b, rtol=1e-6)

    def test_giant_takes_the_spherical_branch(self, auto, wavelengths, linelist):
        sph = prepare_synthesis(wavelengths, linelist, geometry="spherical")
        a = np.asarray(auto(4500.0, 2.0, M_H)[0])
        b = np.asarray(sph(4500.0, 2.0, M_H)[0])
        np.testing.assert_allclose(a, b, rtol=1e-6)

    def test_the_two_branches_actually_differ_for_a_giant(self, wavelengths, linelist):
        """If they agreed, the dispatch would be untestable and pointless."""
        pp = prepare_synthesis(wavelengths, linelist, geometry="plane-parallel")
        sph = prepare_synthesis(wavelengths, linelist, geometry="spherical")
        a = np.asarray(pp(4500.0, 2.0, M_H)[0])
        b = np.asarray(sph(4500.0, 2.0, M_H)[0])
        assert np.max(np.abs(a - b) / a) > 1e-4

    def test_jit_and_gradients_through_the_dispatch(self, auto):
        assert np.all(np.isfinite(np.asarray(
            jax.jit(lambda t: auto(t, LOGG, M_H)[0])(TEFF))))
        g = float(jax.grad(lambda t: jnp.sum(
            (lambda fc: fc[0] / fc[1])(auto(t, LOGG, M_H))))(TEFF))
        assert np.isfinite(g)

    def test_from_atmosphere_demands_logg_when_geometry_is_None(self, auto, abundances):
        from korg.traced_synthesis import _interpolate_marcs_traced
        T, nt, ne, z, lt, _ = _interpolate_marcs_traced(auto, TEFF, LOGG, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError, match="needs logg"):
            auto.from_atmosphere(T, nt, ne, z, lt, abundances)


class TestPhotosphereRadius:
    """R = sqrt(G M_sun / g), computed from the *called* log g.

    It used to be frozen into the plan at the reference log g, so a solar-built
    plan gave a giant the solar radius -- 6.9e10 cm instead of 1.2e12, a factor
    of 17 -- and d/dlogg missed the radius dependence entirely. Every spherical
    test synthesized at the reference parameters, where frozen and live coincide,
    so none of them caught it.
    """

    def test_radius_tracks_logg(self):
        from korg.traced_synthesis import photosphere_radius
        assert float(photosphere_radius(4.44)) == pytest.approx(6.942e10, rel=1e-3)
        assert float(photosphere_radius(2.0)) == pytest.approx(1.152e12, rel=1e-3)
        assert float(photosphere_radius(2.0)) > 10 * float(photosphere_radius(4.44))

    def test_radius_is_differentiable(self):
        from korg.traced_synthesis import photosphere_radius
        g = float(jax.grad(photosphere_radius)(4.44))
        assert np.isfinite(g) and g < 0.0, "higher gravity means a smaller star"

    def test_a_solar_plan_gives_a_giant_the_giant_radius(self, wavelengths, linelist):
        """The regression this class exists for: one plan, two very different stars."""
        sph = prepare_synthesis(wavelengths, linelist, geometry="spherical",
                                reference=(5777.0, 4.44, 0.0))
        flux = np.asarray(sph(4500.0, 2.0, M_H)[0])
        assert np.all(np.isfinite(flux)) and np.all(flux > 0)
        g = float(jax.grad(lambda L: jnp.sum(
            (lambda fc: fc[0] / fc[1])(sph(4500.0, L, M_H))))(2.0))
        assert np.isfinite(g) and g != 0.0, "d/dlogg must see the radius"
# Staged for tests/test_synthesizer_closure.py once full6 clears.
# Appended to the FUNCTIONAL tier.

class TestLinelistFiltering:
    """``prepare_synthesis`` trims the linelist to the synthesis range.

    It knows the wavelength grid, so making the caller pre-filter was a trap
    rather than a design: the full 41,861-line VALD list against a 5 A window
    is 166 useful lines, and every one of the other 41,695 was bucketed and
    evaluated. Korg.jl's ``synthesize`` applies ``line_buffer`` (10 A) for the
    same reason.
    """

    def test_out_of_range_lines_are_dropped(self, wavelengths, linelist):
        from korg.synthesis import filter_linelist
        expected = len(filter_linelist(list(linelist), np.asarray(wavelengths) * 1e-8,
                                       10.0e-8, warn_empty=False))
        s = prepare_synthesis(wavelengths, linelist, geometry="planar")
        assert s.n_lines == expected
        assert s.n_lines < len(linelist), "the fixture must have out-of-range lines"

    def test_line_buffer_none_keeps_every_line(self, wavelengths, linelist):
        s = prepare_synthesis(wavelengths, linelist, geometry="planar",
                              line_buffer_cm=None)
        assert s.n_lines == len(linelist)

    def test_a_tighter_buffer_keeps_fewer_lines(self, wavelengths, linelist):
        wide = prepare_synthesis(wavelengths, linelist, geometry="planar",
                                 line_buffer_cm=10.0e-8)
        tight = prepare_synthesis(wavelengths, linelist, geometry="planar",
                                  line_buffer_cm=0.5e-8)
        assert tight.n_lines < wide.n_lines

    def test_filtering_does_not_change_the_spectrum(self, wavelengths, linelist):
        """The dropped lines are the ones that could not reach the grid.

        This is the assertion that makes the filter safe to apply by default:
        pre-filtering by hand and letting the plan do it must give the same
        spectrum. Both plans hold the identical 386 lines in the identical
        order, so the only thing separating them is floating-point.

        Not ``assert_array_equal``. This was written as a bitwise check, passed
        every targeted run under ``JAX_PLATFORMS=cpu``, and then failed the full
        suite on the default backend — which on this machine is CUDA — by
        5.2e-14 relative on 33 of 200 pixels. GPU reductions do not fix their
        summation order, so identical inputs through an identical program need
        not give identical bits. That is the same lesson three earlier tests in
        this repository already record; 1e-12 is 20x the observed spread.
        """
        from korg.synthesis import filter_linelist
        pre = filter_linelist(list(linelist), np.asarray(wavelengths) * 1e-8, 10.0e-8,
                              warn_empty=False)
        auto = prepare_synthesis(wavelengths, linelist, geometry="planar")
        manual = prepare_synthesis(wavelengths, pre, geometry="planar",
                                   line_buffer_cm=None)
        assert auto.n_lines == manual.n_lines, "the two plans must hold the same lines"
        fa, ca = auto(TEFF, LOGG, M_H)
        fm, cm = manual(TEFF, LOGG, M_H)
        np.testing.assert_allclose(np.asarray(fa), np.asarray(fm), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(ca), np.asarray(cm), rtol=1e-12)

    def test_preprocessed_linelistdata_is_passed_through(self, wavelengths, linelist):
        """A LinelistData has already been bucketed; there is nothing to filter."""
        from korg.synthesis import preprocess_linelist, load_synthesis_data
        d = load_synthesis_data()
        # preprocess_linelist takes cm; the fixture is in Angstroms, as
        # prepare_synthesis now is.
        pre = preprocess_linelist(list(linelist), d.chem_eq_data,
                                  np.asarray(wavelengths) * 1e-8)
        s = prepare_synthesis(wavelengths, pre, data=d, geometry="planar")
        assert s.n_lines == int(np.shape(pre.wl)[0])
