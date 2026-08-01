"""
Autodiff and jit-tracing tests for ``korg.fit``.

``fit.py`` is split down the middle by what JAX can see (see the module
docstring there):

* everything after ``synthesize`` -- continuum rectification, rotational
  broadening, the LSF matrix, the linear continuum adjustment, the tan
  unscaling -- is pure JAX and is differentiated here, with every gradient
  checked against central finite differences and asserted finite and non-zero;
* ``synthesize`` and ``interpolate_marcs`` are not traceable.  The tests at the
  bottom of this file *pin* that with ``pytest.raises`` so the limitation is a
  checked fact rather than a claim in a comment, and so that the day
  ``synthesize_jit`` replaces ``synthesize_spectrum`` the failure points at the
  code that should then change.
"""

import types

import korg  # noqa: F401 — enables JAX x64

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import korg.fit as fit
from korg.fit import (
    _POSTPROCESSING_PARAMS, _linear_continuum_adjustment,
    _linear_continuum_adjustment_jax, _make_autodiff_chi2, _postprocess_flux,
    _tan_scale, _tan_unscale, _tan_unscale_jax, _unscale_param_jax,
)
from tests.fit_test_support import central_difference, gaussian_spectrum, patch_synthesis

#: relative tolerance for autodiff-vs-finite-difference agreement.  Central
#: differences on a smooth double-precision function are good to ~1e-10 in the
#: best case and ~1e-6 when the second derivative is large; 1e-5 is the honest
#: bound across all the parameters tested here.
FD_RTOL = 1e-5

WLS = np.linspace(4996.0, 5014.0, 361)
OBS = np.linspace(4999.0, 5011.0, 90)


@pytest.fixture(scope="module")
def spectrum():
    flux, cntm = gaussian_spectrum(WLS)
    LSF = np.asarray(fit.compute_LSF_matrix(WLS, OBS, 25000.0, verbose=False))
    return flux, cntm, LSF


@pytest.fixture(scope="module")
def observed(spectrum):
    flux, cntm, LSF = spectrum
    truth = np.asarray(_postprocess_flux(flux, cntm, WLS, LSF, 0.01, 0.0005, 7.0, 0.6))
    return truth, np.full_like(truth, 0.003)


# ---------------------------------------------------------------------------
# Scaling helpers
# ---------------------------------------------------------------------------

class TestTanUnscaleJax:

    @pytest.mark.parametrize("p", [-50.0, -3.0, -0.5, 0.0, 0.25, 4.0, 80.0])
    def test_matches_the_numpy_version(self, p):
        for lo, hi in [(2800.0, 8000.0), (-5.0, 1.0), (0.0, 1.0)]:
            assert float(_tan_unscale_jax(p, lo, hi)) == pytest.approx(
                _tan_unscale(p, lo, hi), rel=1e-15)

    @pytest.mark.parametrize("p", [-3.0, 0.0, 1.7])
    def test_gradient_matches_the_analytic_derivative(self, p):
        lo, hi = 2800.0, 8000.0
        g = jax.grad(lambda x: _tan_unscale_jax(x, lo, hi))(p)
        assert float(g) == pytest.approx((hi - lo) / (np.pi * (1 + p ** 2)), rel=1e-12)
        assert np.isfinite(g) and g != 0

    @pytest.mark.parametrize("name", sorted(_POSTPROCESSING_PARAMS))
    def test_unscale_param_jax_is_differentiable(self, name):
        g = jax.grad(lambda x: _unscale_param_jax(name, x))(0.4)
        assert np.isfinite(g) and float(g) != 0.0

    def test_unscale_param_jax_matches_the_numpy_path(self):
        from korg.fit import _unscale_params
        for name in ["Teff", "logg", "M_H", "epsilon", "vmic", "vsini", "Fe"]:
            got = float(_unscale_param_jax(name, 0.3))
            assert got == pytest.approx(_unscale_params({name: 0.3})[name], rel=1e-14)

    def test_unscale_param_jax_rejects_unknown_names(self):
        with pytest.raises(ValueError, match="Unknown parameter"):
            _unscale_param_jax("nonsense", 0.0)

    def test_vmic_gradient_is_finite_at_the_bottom_of_the_range(self):
        """
        The *forward* map for vmic/vsini is ``tan_scale(sqrt(p))`` and sqrt has an
        infinite derivative at 0.  The inverse squares instead, so nothing here
        is singular -- but check it, because a sqrt in the differentiated
        direction is exactly the shape of bug this project keeps finding.
        """
        s = _tan_scale(np.sqrt(1e-12), 0.0, np.sqrt(250.0))
        g = jax.grad(lambda x: _unscale_param_jax("vmic", x))(s)
        assert np.isfinite(g)
        assert float(_unscale_param_jax("vmic", s)) == pytest.approx(1e-12, rel=1e-6)


# ---------------------------------------------------------------------------
# _postprocess_flux
# ---------------------------------------------------------------------------

class TestPostprocessFluxGradients:
    """``jax.grad`` w.r.t. each post-processing parameter, vs central differences."""

    @pytest.mark.parametrize("index,name,value", [
        (0, "cntm_offset", 0.02),
        (1, "cntm_slope", 0.0004),
        (2, "vsini", 6.0),
        (3, "epsilon", 0.55),
    ])
    def test_gradient_matches_finite_differences(self, spectrum, observed, index,
                                                 name, value):
        flux, cntm, LSF = spectrum
        obs, err = observed
        theta0 = np.array([0.01, 0.0003, 5.0, 0.5])
        theta0[index] = value

        def chi2(theta):
            model = _postprocess_flux(flux, cntm, WLS, LSF, theta[0], theta[1],
                                      theta[2], theta[3])
            return jnp.sum(((model - obs) / err) ** 2)

        g = np.asarray(jax.grad(chi2)(jnp.asarray(theta0)))
        assert np.all(np.isfinite(g)), f"non-finite gradient: {g}"
        assert g[index] != 0.0, f"{name} gradient is exactly zero"

        step = {0: 1e-7, 1: 1e-9, 2: 1e-6, 3: 1e-6}[index]
        fd = central_difference(lambda t: float(chi2(jnp.asarray(t))), theta0, index,
                                h=step)
        assert g[index] == pytest.approx(fd, rel=FD_RTOL), (
            f"{name}: autodiff {g[index]!r} vs finite difference {fd!r}")

    def test_all_four_gradients_are_finite_and_non_zero_at_once(self, spectrum,
                                                               observed):
        flux, cntm, LSF = spectrum
        obs, err = observed

        def chi2(theta):
            model = _postprocess_flux(flux, cntm, WLS, LSF, theta[0], theta[1],
                                      theta[2], theta[3])
            return jnp.sum(((model - obs) / err) ** 2)

        g = np.asarray(jax.grad(chi2)(jnp.array([0.015, 0.0004, 6.0, 0.55])))
        assert np.all(np.isfinite(g))
        assert np.all(g != 0.0)

    def test_gradient_wrt_flux_is_finite(self, spectrum, observed):
        """The raw spectrum is a constant of the fit, but must stay differentiable."""
        flux, cntm, LSF = spectrum
        obs, err = observed

        def chi2(f):
            model = _postprocess_flux(f, cntm, WLS, LSF, 0.01, 0.0003, 5.0, 0.6)
            return jnp.sum(((model - obs) / err) ** 2)

        g = np.asarray(jax.grad(chi2)(jnp.asarray(flux)))
        assert np.all(np.isfinite(g))
        assert np.any(g != 0.0)

    @pytest.mark.parametrize("epsilon", [0.0, 0.3, 1.0])
    def test_no_nan_cotangent_at_the_limb_darkening_extremes(self, spectrum, observed,
                                                             epsilon):
        """
        ε = 0 (no limb darkening) and ε = 1 (full) are the endpoints of the
        allowed range and are where a masked-but-still-differentiated sqrt would
        show up as a NaN cotangent while the *value* stayed finite.  Check both
        the value and the gradient.
        """
        flux, cntm, LSF = spectrum
        obs, err = observed

        def chi2(theta):
            model = _postprocess_flux(flux, cntm, WLS, LSF, 0.0, 0.0, theta[0],
                                      theta[1])
            return jnp.sum(((model - obs) / err) ** 2)

        theta = jnp.array([8.0, epsilon])
        value = float(chi2(theta))
        g = np.asarray(jax.grad(chi2)(theta))
        assert np.isfinite(value)
        assert np.all(np.isfinite(g)), (
            f"NaN/Inf cotangent at epsilon={epsilon}: {g} (value was finite: {value})")

    def test_no_nan_cotangent_at_tiny_vsini(self, spectrum, observed):
        """vsini just above the ``vsini > 0`` short-circuit."""
        flux, cntm, LSF = spectrum
        obs, err = observed

        def chi2(v):
            model = _postprocess_flux(flux, cntm, WLS, LSF, 0.0, 0.0, v, 0.6)
            return jnp.sum(((model - obs) / err) ** 2)

        for v in (1e-6, 1e-3):
            g = jax.grad(chi2)(v)
            assert np.isfinite(g), f"non-finite cotangent at vsini={v}"

    def test_vsini_zero_gives_a_zero_not_nan_cotangent(self, spectrum, observed):
        """
        At exactly vsini = 0 the rotation kernel is skipped, so d(chi2)/d(vsini)
        is structurally zero.  It must be zero, not NaN.
        """
        flux, cntm, LSF = spectrum
        obs, err = observed

        def chi2(v):
            model = _postprocess_flux(flux, cntm, WLS, LSF, 0.0, 0.0, v, 0.6)
            return jnp.sum(((model - obs) / err) ** 2)

        g = jax.grad(chi2)(0.0)
        assert np.isfinite(g)
        assert float(g) == 0.0


# ---------------------------------------------------------------------------
# _linear_continuum_adjustment_jax
# ---------------------------------------------------------------------------

class TestContinuumAdjustmentGradients:

    OBS_WLS = np.linspace(5000.0, 5010.0, 101)

    def _fixtures(self):
        obs_flux = 1.0 - 0.4 * np.exp(-((self.OBS_WLS - 5005.0) ** 2) / 0.09)
        obs_err = np.full(101, 0.01)
        model = 0.97 * obs_flux + 0.004 * (self.OBS_WLS - 5005.0)
        return obs_flux, obs_err, model

    def test_matches_the_in_place_numpy_version(self):
        obs_flux, obs_err, model = self._fixtures()
        got = np.asarray(_linear_continuum_adjustment_jax(
            self.OBS_WLS, None, model.copy(), obs_flux, obs_err))
        expected = model.copy()
        _linear_continuum_adjustment(self.OBS_WLS, None, expected, obs_flux, obs_err)
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_gradient_wrt_model_flux_matches_finite_differences(self):
        obs_flux, obs_err, model = self._fixtures()

        def chi2(m):
            adjusted = _linear_continuum_adjustment_jax(self.OBS_WLS, None, m,
                                                        obs_flux, obs_err)
            return jnp.sum(((adjusted - obs_flux) / obs_err) ** 2)

        g = np.asarray(jax.grad(chi2)(jnp.asarray(model)))
        assert np.all(np.isfinite(g))
        for i in (0, 50, 100):
            fd = central_difference(lambda m: float(chi2(jnp.asarray(m))), model, i,
                                    h=1e-7)
            assert g[i] == pytest.approx(fd, rel=FD_RTOL, abs=1e-6)

    def test_windows_branch_is_differentiable(self):
        obs_flux, obs_err, model = self._fixtures()
        windows = [(5001.0, 5004.0), (5006.0, 5009.0)]

        def chi2(m):
            adjusted = _linear_continuum_adjustment_jax(self.OBS_WLS, windows, m,
                                                        obs_flux, obs_err)
            return jnp.sum(((adjusted - obs_flux) / obs_err) ** 2)

        g = np.asarray(jax.grad(chi2)(jnp.asarray(model)))
        assert np.all(np.isfinite(g))
        assert np.any(g != 0.0)

    def test_empty_window_is_skipped_without_breaking_the_trace(self):
        obs_flux, obs_err, model = self._fixtures()
        out = _linear_continuum_adjustment_jax(self.OBS_WLS, [(6000.0, 6001.0)],
                                               model, obs_flux, obs_err)
        np.testing.assert_array_equal(np.asarray(out), model)


# ---------------------------------------------------------------------------
# The full fit objective
# ---------------------------------------------------------------------------

class TestAutodiffChi2:

    def _chi2(self, spectrum, observed, names, adjust_continuum=False):
        flux, cntm, LSF = spectrum
        obs, err = observed
        fixed = {"Teff": 5777.0, "logg": 4.44, "M_H": 0.0, "vmic": 1.0,
                 "vsini": 0.0, "epsilon": 0.6, "cntm_offset": 0.0, "cntm_slope": 0.0}
        fixed = {k: v for k, v in fixed.items() if k not in names}
        return _make_autodiff_chi2((flux, cntm, WLS), LSF, names, fixed, OBS, obs,
                                   err, None, adjust_continuum)

    NAMES = ["vsini", "epsilon", "cntm_offset", "cntm_slope"]
    P0 = np.array([_tan_scale(np.sqrt(6.0), 0.0, np.sqrt(250.0)),
                   _tan_scale(0.55, 0.0, 1.0),
                   _tan_scale(0.015, -0.5, 0.5),
                   _tan_scale(0.0004, -0.1, 0.1)])

    def test_gradient_is_finite_and_non_zero(self, spectrum, observed):
        chi2 = self._chi2(spectrum, observed, self.NAMES)
        g = np.asarray(jax.grad(chi2)(jnp.asarray(self.P0)))
        assert np.all(np.isfinite(g))
        assert np.all(g != 0.0)

    @pytest.mark.parametrize("index", range(4))
    def test_gradient_matches_finite_differences(self, spectrum, observed, index):
        chi2 = self._chi2(spectrum, observed, self.NAMES)
        g = np.asarray(jax.grad(chi2)(jnp.asarray(self.P0)))
        fd = central_difference(lambda p: float(chi2(jnp.asarray(p))), self.P0, index,
                                h=1e-6)
        assert g[index] == pytest.approx(fd, rel=FD_RTOL), (
            f"{self.NAMES[index]}: autodiff {g[index]!r} vs central difference {fd!r}")

    def test_gradient_with_continuum_adjustment(self, spectrum, observed):
        chi2 = self._chi2(spectrum, observed, self.NAMES, adjust_continuum=True)
        g = np.asarray(jax.grad(chi2)(jnp.asarray(self.P0)))
        assert np.all(np.isfinite(g))
        for index in range(4):
            fd = central_difference(lambda p: float(chi2(jnp.asarray(p))), self.P0,
                                    index, h=1e-6)
            assert g[index] == pytest.approx(fd, rel=1e-4, abs=1e-6)

    def test_value_and_grad_agree_with_the_value_alone(self, spectrum, observed):
        chi2 = self._chi2(spectrum, observed, self.NAMES)
        value, _ = jax.value_and_grad(chi2)(jnp.asarray(self.P0))
        assert float(value) == pytest.approx(float(chi2(jnp.asarray(self.P0))), rel=0)

    def test_hessian_is_finite(self, spectrum, observed):
        """Second derivatives too -- BFGS builds an approximation to this."""
        chi2 = self._chi2(spectrum, observed, self.NAMES)
        H = np.asarray(jax.hessian(chi2)(jnp.asarray(self.P0)))
        assert H.shape == (4, 4)
        assert np.all(np.isfinite(H))


# ---------------------------------------------------------------------------
# fit_spectrum: autodiff path vs numerical path
# ---------------------------------------------------------------------------

class TestFitSpectrumAutodiffPath:

    FIXED = {"Teff": 5777.0, "logg": 4.44, "M_H": 0.0, "vmic": 1.0,
             "epsilon": 0.6, "cntm_slope": 0.0}
    TRUTH = {"vsini": 9.0, "cntm_offset": 0.025}

    @pytest.fixture
    def problem(self, monkeypatch, spectrum):
        flux, cntm, LSF = spectrum
        from tests.fit_test_support import FakeSynthesisResult

        counter = {"n": 0}

        def synth(atm, linelist, wls, A_X, **kw):
            counter["n"] += 1
            return FakeSynthesisResult(WLS, flux, cntm)

        monkeypatch.setattr(fit, "synthesize", synth)
        monkeypatch.setattr(fit, "interpolate_marcs",
                            lambda *a, **k: types.SimpleNamespace(Teff=0.0, logg=0.0))
        truth = np.asarray(fit._synthetic_spectrum(WLS, [], LSF,
                                                   {**self.FIXED, **self.TRUTH}, {}))
        counter["n"] = 0
        return LSF, truth, np.full_like(truth, 0.002), counter

    def _fit(self, problem, force_numerical, monkeypatch, **kw):
        LSF, truth, err, counter = problem
        if force_numerical:
            monkeypatch.setattr(fit, "_POSTPROCESSING_PARAMS", frozenset())
        counter["n"] = 0
        res = fit.fit_spectrum(OBS, truth, err, [],
                               initial_guesses={"vsini": 5.0, "cntm_offset": 0.0},
                               fixed_params=self.FIXED, LSF_matrix=LSF,
                               synthesis_wls=WLS, precision=1e-7, **kw)
        return res, counter["n"]

    def test_autodiff_needs_exactly_one_synthesis_for_the_whole_fit(self, problem,
                                                                    monkeypatch):
        res, n = self._fit(problem, False, monkeypatch)
        # one for the fit itself, one for the returned best-fit spectrum
        assert n == 2, f"expected 2 syntheses, got {n}"
        assert len(res["trace"]) > 2

    def test_numerical_path_needs_one_synthesis_per_evaluation(self, problem,
                                                               monkeypatch):
        res, n = self._fit(problem, True, monkeypatch)
        assert n == len(res["trace"]) + 1
        assert n > 20

    def test_both_paths_find_the_same_parameters(self, problem, monkeypatch):
        res_ad, _ = self._fit(problem, False, monkeypatch)
        res_nm, _ = self._fit(problem, True, monkeypatch)
        for name, truth in self.TRUTH.items():
            a = res_ad["best_fit_params"][name]
            b = res_nm["best_fit_params"][name]
            assert a == pytest.approx(truth, rel=1e-4), name
            assert a == pytest.approx(b, rel=2e-4), (
                f"{name}: autodiff {a!r} vs numerical {b!r}")

    def test_autodiff_path_records_a_trace(self, problem, monkeypatch):
        res, _ = self._fit(problem, False, monkeypatch)
        assert all(set(t) == {"vsini", "cntm_offset", "chi2"} for t in res["trace"])
        assert res["trace"][-1]["chi2"] <= res["trace"][0]["chi2"]

    def test_autodiff_path_honours_the_time_limit(self, problem, monkeypatch):
        LSF, truth, err, _ = problem
        res = fit.fit_spectrum(OBS, truth, err, [],
                               initial_guesses={"vsini": 5.0},
                               fixed_params={**self.FIXED, "cntm_offset": 0.0},
                               LSF_matrix=LSF, synthesis_wls=WLS, precision=1e-12,
                               time_limit=0.0)
        assert res["solver_result"].success is False

    def test_postprocess_forces_the_numerical_path(self, problem, monkeypatch):
        """``postprocess`` mutates a NumPy buffer in place; JAX arrays are immutable."""
        LSF, truth, err, counter = problem
        counter["n"] = 0
        res = fit.fit_spectrum(OBS, truth, err, [], initial_guesses={"vsini": 5.0},
                               fixed_params={**self.FIXED, "cntm_offset": 0.0},
                               LSF_matrix=LSF, synthesis_wls=WLS, precision=1e-3,
                               postprocess=lambda f, of, oe: None)
        assert counter["n"] == len(res["trace"]) + 1

    def test_falls_back_when_the_single_synthesis_fails(self, problem, monkeypatch):
        LSF, truth, err, _ = problem
        monkeypatch.setattr(fit, "interpolate_marcs",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope")))
        with pytest.warns(UserWarning, match="Falling back to numerical gradients"):
            res = fit.fit_spectrum(OBS, truth, err, [],
                                   initial_guesses={"vsini": 5.0},
                                   fixed_params={**self.FIXED, "cntm_offset": 0.0},
                                   LSF_matrix=LSF, synthesis_wls=WLS, precision=1e-2)
        assert np.isfinite(res["solver_result"].fun)

    def test_autodiff_path_with_continuum_adjustment(self, problem, monkeypatch):
        LSF, truth, err, counter = problem
        counter["n"] = 0
        res = fit.fit_spectrum(OBS, truth * 1.02, err, [],
                               initial_guesses={"vsini": 5.0},
                               fixed_params={**self.FIXED, "cntm_offset": 0.0},
                               LSF_matrix=LSF, synthesis_wls=WLS, precision=1e-5,
                               adjust_continuum=True)
        assert counter["n"] == 2
        assert res["best_fit_params"]["vsini"] == pytest.approx(9.0, abs=0.2)


# ---------------------------------------------------------------------------
# jit tracing: what works, and what provably cannot
# ---------------------------------------------------------------------------

class TestJitTracing:

    def test_tan_unscale_jax_jits(self):
        f = jax.jit(lambda p: _tan_unscale_jax(p, 2800.0, 8000.0))
        assert float(f(0.3)) == pytest.approx(_tan_unscale(0.3, 2800.0, 8000.0),
                                              rel=1e-14)

    def test_unscale_param_jax_jits(self):
        import functools
        f = jax.jit(functools.partial(_unscale_param_jax, "vsini"))
        assert float(f(0.3)) == pytest.approx(
            fit._unscale_params({"vsini": 0.3})["vsini"], rel=1e-14)

    def test_postprocess_flux_jits_over_the_spectrum(self, spectrum):
        """
        Flux, continuum, cntm_offset and cntm_slope may all be traced: none of
        them changes an array shape.
        """
        flux, cntm, LSF = spectrum

        @jax.jit
        def f(raw_flux, raw_cntm, offset, slope):
            return _postprocess_flux(raw_flux, raw_cntm, WLS, LSF, offset, slope,
                                     0.0, 0.6)

        got = np.asarray(f(jnp.asarray(flux), jnp.asarray(cntm), 0.02, 0.0004))
        expected = np.asarray(_postprocess_flux(flux, cntm, WLS, LSF, 0.02, 0.0004,
                                                0.0, 0.6))
        np.testing.assert_allclose(got, expected, rtol=1e-13)

    def test_postprocess_flux_jits_with_a_static_vsini(self, spectrum):
        """With vsini closed over as a constant the rotation kernel is static."""
        flux, cntm, LSF = spectrum

        @jax.jit
        def f(raw_flux):
            return _postprocess_flux(raw_flux, cntm, WLS, LSF, 0.0, 0.0, 7.0, 0.6)

        got = np.asarray(f(jnp.asarray(flux)))
        expected = np.asarray(_postprocess_flux(flux, cntm, WLS, LSF, 0.0, 0.0, 7.0,
                                                0.6))
        np.testing.assert_allclose(got, expected, rtol=1e-13)

    def test_traced_vsini_under_jit_is_rejected_loudly(self, spectrum):
        """
        vsini sets the *width* of the rotation kernel, and an array shape cannot
        depend on a traced value.  ``apply_rotation`` therefore needs a concrete
        vsini: eager ``jax.grad`` supplies one (the JVP primal), ``jax.jit`` with
        vsini as a traced argument does not.  The failure must be an explicit
        error, not a silently truncated kernel -- pin that here.  Passing vsini
        via ``static_argnums`` (the test above) is the supported route.
        """
        flux, cntm, LSF = spectrum

        @jax.jit
        def f(raw_flux, vsini):
            return _postprocess_flux(raw_flux, cntm, WLS, LSF, 0.0, 0.0, vsini, 0.6)

        with pytest.raises((TypeError, jax.errors.JAXTypeError)):
            f(jnp.asarray(flux), 7.0)

    def test_jit_with_vsini_as_a_static_argument_works(self, spectrum):
        import functools
        flux, cntm, LSF = spectrum

        @functools.partial(jax.jit, static_argnums=(1,))
        def f(raw_flux, vsini):
            return _postprocess_flux(raw_flux, cntm, WLS, LSF, 0.0, 0.0, vsini, 0.6)

        got = np.asarray(f(jnp.asarray(flux), 7.0))
        expected = np.asarray(_postprocess_flux(flux, cntm, WLS, LSF, 0.0, 0.0, 7.0,
                                                0.6))
        np.testing.assert_allclose(got, expected, rtol=1e-13)

    def test_linear_continuum_adjustment_jax_jits(self):
        obs_wls = np.linspace(5000.0, 5010.0, 101)
        obs_flux = 1.0 - 0.4 * np.exp(-((obs_wls - 5005.0) ** 2) / 0.09)
        obs_err = np.full(101, 0.01)
        model = 0.97 * obs_flux

        f = jax.jit(lambda m: _linear_continuum_adjustment_jax(obs_wls, None, m,
                                                               obs_flux, obs_err))
        expected = model.copy()
        _linear_continuum_adjustment(obs_wls, None, expected, obs_flux, obs_err)
        np.testing.assert_allclose(np.asarray(f(jnp.asarray(model))), expected,
                                   rtol=1e-12)

    def test_autodiff_chi2_jits_when_vsini_is_not_free(self, spectrum, observed):
        """
        With only cntm_offset/cntm_slope free, nothing traced reaches a shape, so
        the whole objective compiles.
        """
        flux, cntm, LSF = spectrum
        obs, err = observed
        names = ["cntm_offset", "cntm_slope"]
        fixed = {"vsini": 0.0, "epsilon": 0.6}
        chi2 = _make_autodiff_chi2((flux, cntm, WLS), LSF, names, fixed, OBS, obs,
                                   err, None, False)
        p = jnp.array([0.1, -0.2])
        assert float(jax.jit(chi2)(p)) == pytest.approx(float(chi2(p)), rel=1e-13)
        g = np.asarray(jax.jit(jax.grad(chi2))(p))
        assert np.all(np.isfinite(g)) and np.all(g != 0.0)

    def test_autodiff_chi2_cannot_be_jitted_when_vsini_is_free(self, spectrum,
                                                               observed):
        """
        Same reason as ``test_traced_vsini_under_jit_is_rejected_loudly``: a free
        vsini under ``jax.jit`` would have to set a kernel width from a traced
        value.  Eager ``jax.grad`` -- which is what ``fit_spectrum`` uses -- is
        unaffected, and is exercised by ``TestAutodiffChi2`` above.
        """
        flux, cntm, LSF = spectrum
        obs, err = observed
        chi2 = _make_autodiff_chi2((flux, cntm, WLS), LSF, ["vsini"],
                                   {"epsilon": 0.6}, OBS, obs, err, None, False)
        with pytest.raises((TypeError, jax.errors.JAXTypeError)):
            jax.jit(chi2)(jnp.array([0.3]))


class TestSynthesisIsNotTraceable:
    """
    Pin the reason ``fit.py`` cannot differentiate the atmospheric parameters.

    These are not aspirational xfails: they assert the *current* behaviour so
    that the module docstring's claim is checked by the suite.  When
    ``synthesize_jit`` replaces ``synthesize_spectrum`` these tests will fail,
    which is the signal to switch ``fit_spectrum`` over to a full ``jax.grad``.
    """

    @pytest.fixture(scope="class")
    def solar(self):
        from pathlib import Path
        atm_file = Path(__file__).parent / "data" / "sun.mod"
        if not atm_file.exists():
            raise AssertionError(f"Solar atmosphere fixture missing: {atm_file}")
        return korg.read_model_atmosphere(str(atm_file)), korg.format_A_X()

    def test_synthesize_cannot_be_differentiated_wrt_vmic(self, solar):
        atm, A_X = solar
        wls = np.linspace(4999.5, 5000.5, 12)

        def objective(vmic):
            sol = korg.synthesize(atm, [], wls, A_X, vmic=vmic, hydrogen_lines=False,
                                  verbose=False)
            return jnp.sum(jnp.asarray(sol.flux) / jnp.asarray(sol.continuum))

        with pytest.raises(jax.errors.TracerArrayConversionError):
            jax.grad(objective)(1.0)

    def test_synthesize_cannot_be_differentiated_wrt_abundances(self, solar):
        atm, A_X = solar
        wls = np.linspace(4999.5, 5000.5, 12)

        def objective(abundances):
            sol = korg.synthesize(atm, [], wls, abundances, hydrogen_lines=False,
                                  verbose=False)
            return jnp.sum(jnp.asarray(sol.flux) / jnp.asarray(sol.continuum))

        with pytest.raises(jax.errors.TracerArrayConversionError):
            jax.grad(objective)(jnp.asarray(A_X))

    def test_interpolate_marcs_cannot_be_traced(self):
        from korg.marcs_interpolation import interpolate_marcs

        def objective(Teff):
            atm = interpolate_marcs(Teff, 4.44, 0.0)
            return jnp.sum(jnp.asarray(atm.T))

        with pytest.raises(jax.errors.ConcretizationTypeError):
            jax.grad(objective)(5777.0)

    def test_the_postprocessing_parameters_are_exactly_the_differentiable_ones(self):
        """Documented split: everything else goes through synthesize/marcs."""
        assert _POSTPROCESSING_PARAMS == {"vsini", "epsilon", "cntm_offset",
                                          "cntm_slope"}
