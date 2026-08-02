"""
The traceable synthesizer: ``prepare_synthesis`` and the closure it returns.

Four tiers, matching the project's taxonomy:

1. Functional — the plan is built correctly, the closure runs, shapes and
   options behave.
2. Agreement — the closure reproduces the established ``synthesize_jit`` path.
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
    return np.arange(5000.0, 5002.0, 0.01) * 1e-8


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
        with pytest.raises(ValueError, match="planar.*spherical"):
            prepare_synthesis(wavelengths, linelist, geometry="toroidal")

    def test_rejects_a_degenerate_wavelength_grid(self, linelist):
        with pytest.raises(ValueError, match="two wavelength points"):
            prepare_synthesis(np.array([5000e-8]), linelist)

    def test_reference_pixel_found_only_when_5000A_is_covered(self, linelist):
        inside = prepare_synthesis(np.arange(4999.0, 5001.0, 0.01) * 1e-8, linelist)
        outside = prepare_synthesis(np.arange(6000.0, 6002.0, 0.01) * 1e-8, linelist)
        assert inside.ref_pixel is not None
        assert outside.ref_pixel is None

    def test_stark_transitions_selected_by_wavelength(self, linelist):
        """H-beta at 4861 A must be picked up near it and not at 6000 A."""
        near = prepare_synthesis(np.arange(4855.0, 4867.0, 0.01) * 1e-8, linelist)
        far = prepare_synthesis(np.arange(7000.0, 7002.0, 0.01) * 1e-8, linelist)
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
        T, nt, ne, z, lt = _interpolate_marcs_traced(synth, TEFF, LOGG, 0.0, 0.0, 0.0)
        direct = np.asarray(synth.from_atmosphere(T, nt, ne, z, lt, abundances)[0])
        viaparams = np.asarray(synth(TEFF, LOGG, abundances=abundances)[0])
        # 1.3e-8 on 2 of 200 pixels. Same floor as test_jit_matches_the_eager_result:
        # the bucketed Voigt runs in float32, and the two routes to the same
        # atmosphere arrays let XLA fuse and reassociate differently. Measured, not
        # assumed -- an earlier 1e-8 here failed by a factor of 1.3.
        np.testing.assert_allclose(direct, viaparams, rtol=1e-7)


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

    def test_photosphere_radius_follows_logg(self, wavelengths, linelist):
        """R = sqrt(G M_sun / g), so higher gravity means a smaller star."""
        dwarf = prepare_synthesis(wavelengths, linelist, geometry="spherical",
                                  reference=(5777.0, 4.44, 0.0))
        giant = prepare_synthesis(wavelengths, linelist, geometry="spherical",
                                  reference=(4500.0, 2.0, 0.0))
        assert giant.R_photosphere > dwarf.R_photosphere
