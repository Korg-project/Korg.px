"""
High-precision agreement between ``korg.fit`` and Korg.jl's ``Korg.Fit``.

Reference values come from ``tests/fit_reference_data.json``, produced by
``tests/generate_fit_reference.jl`` against Korg.jl v1.2.1.

What can and cannot be compared
-------------------------------
Most of ``fit.py`` is pure algorithm and is compared here at rtol 1e-12 or
better.  Two things are *not*:

* ``fit_spectrum`` and ``ews_to_stellar_parameters_direct`` minimise with
  different optimisers -- scipy's BFGS with a Wolfe line search here, Optim.jl's
  BFGS with a ``BackTracking`` line search there, and Korg.jl differentiates the
  objective with ForwardDiff where this port cannot (see the ``fit.py`` module
  docstring).  Their iterates differ from the first step, so the fitted
  parameters are only comparable to the optimiser's own convergence tolerance.
  No such comparison is asserted; instead the *objective* those optimisers
  minimise is compared piece by piece.
* The end-to-end value of ``calculate_EWs`` depends on the whole synthesis
  port.  It is checked at a loose, separately-stated tolerance in
  ``TestCalculateEWsEndToEnd``.  The *EW extraction algorithm itself* -- window
  merging, blend-boundary location, trapezoid integration -- is isolated in
  ``TestCalculateEWsAlgorithm`` by feeding the Python code the exact absorption
  array Julia integrated, and is asserted at rtol 1e-10.
"""

import json
from pathlib import Path

import korg  # noqa: F401 — enables JAX x64

import numpy as np
import pytest

from korg.fit import (
    _get_slope, _get_slope_uncertainty, _linear_continuum_adjustment,
    _linear_continuum_adjustment_jax, _merge_windows, _scale_params, _tan_scale,
    _tan_unscale, _unscale_params, calculate_EWs, ews_to_abundances_approx,
)
from tests.fit_test_support import FakeSynthesisResult

REFERENCE_FILE = Path(__file__).parent / "fit_reference_data.json"
ATM_FILE = Path(__file__).parent / "data" / "sun.mod"


def _load_reference():
    """Load the Julia fixture, failing loudly rather than skipping.

    A skipped test is not a passing test: if the fixture is missing the suite
    must say so, not quietly drop the only Korg.jl comparison in this module.
    """
    if not REFERENCE_FILE.exists():
        raise AssertionError(
            f"Julia reference data not found at {REFERENCE_FILE}. "
            "Regenerate with: julia --project=. tests/generate_fit_reference.jl"
        )
    with open(REFERENCE_FILE) as fh:
        return json.load(fh)


REFERENCE = _load_reference()


def test_reference_is_korg_1_2_1():
    """The fixture must come from the Korg.jl version this port targets."""
    assert REFERENCE["korg_version"] == "1.2.1"


# ---------------------------------------------------------------------------
# Parameter scaling
# ---------------------------------------------------------------------------

class TestTanScaling:
    """``_tan_scale``/``_tan_unscale`` vs Korg.jl's ``tan_scale``/``tan_unscale``."""

    def test_tan_scale(self):
        cases = REFERENCE["tan_scale_cases"]
        assert len(cases) == 64
        for case in cases:
            got = _tan_scale(case["p"], case["lower"], case["upper"])
            assert got == pytest.approx(case["scaled"], rel=1e-14, abs=1e-300), (
                f"{case['name']}: p={case['p']} -> {got}, Julia {case['scaled']}"
            )

    def test_tan_unscale(self):
        for case in REFERENCE["tan_unscale_cases"]:
            got = _tan_unscale(case["scaled"], case["lower"], case["upper"])
            assert got == pytest.approx(case["unscaled"], rel=1e-14)

    def test_sqrt_scaled_vmic_and_vsini(self):
        """vmic/vsini are scaled as tan_scale(sqrt(p), 0, sqrt(250))."""
        for case in REFERENCE["sqrt_scale_cases"]:
            scaled = _scale_params({"vmic": case["p"]})["vmic"]
            assert scaled == pytest.approx(case["scaled"], rel=1e-14)
            back = _unscale_params({"vsini": case["scaled"]})["vsini"]
            assert back == pytest.approx(case["unscaled_roundtrip"], rel=1e-13)

    def test_scale_dict_matches_julia(self):
        """The whole ``scale(params)`` dict, element by element."""
        julia_in = REFERENCE["scale_dict_input"]
        julia_out = REFERENCE["scale_dict_output"]
        got = _scale_params(julia_in)
        assert set(got) == set(julia_out)
        for name, value in julia_out.items():
            assert got[name] == pytest.approx(value, rel=1e-14), name

    def test_unscale_dict_matches_julia(self):
        julia_out = REFERENCE["unscale_dict_output"]
        got = _unscale_params(_scale_params(REFERENCE["scale_dict_input"]))
        for name, value in julia_out.items():
            assert got[name] == pytest.approx(value, rel=1e-13), name


# ---------------------------------------------------------------------------
# Window merging
# ---------------------------------------------------------------------------

class TestMergeWindows:
    """``_merge_windows`` vs Korg.jl's ``Korg.merge_bounds``."""

    def test_merged_bounds_match(self):
        cases = REFERENCE["merge_bounds_cases"]
        assert len(cases) == 21
        for case in cases:
            raw = [(lo, hi) for lo, hi in case["raw"]]
            got = _merge_windows(raw, case["buffer"])
            expected = [(lo, hi) for lo, hi in case["merged"]]
            assert len(got) == len(expected), case
            for (a_lo, a_hi), (b_lo, b_hi) in zip(got, expected):
                assert a_lo == pytest.approx(b_lo, rel=1e-14, abs=1e-12)
                assert a_hi == pytest.approx(b_hi, rel=1e-14, abs=1e-12)

    def test_number_of_windows_matches_group_count(self):
        """One merged window per index group in Julia's output."""
        for case in REFERENCE["merge_bounds_cases"]:
            raw = [(lo, hi) for lo, hi in case["raw"]]
            assert len(_merge_windows(raw, case["buffer"])) == len(case["indices"])


# ---------------------------------------------------------------------------
# Excitation/ionization balance helpers
# ---------------------------------------------------------------------------

class TestSlopeHelpers:
    """``_get_slope``/``_get_slope_uncertainty`` vs Korg.jl's versions."""

    def test_get_slope(self):
        for case in REFERENCE["get_slope_cases"][:3]:
            got = _get_slope(case["xs"], case["ys"])
            assert got == pytest.approx(case["slope"], rel=1e-13, abs=1e-14)

    def test_get_slope_uncertainty(self):
        for case in REFERENCE["get_slope_cases"][:3]:
            got = _get_slope_uncertainty(case["xs"])
            assert got == pytest.approx(case["slope_uncertainty"], rel=1e-13)

    def test_nearly_degenerate_xs(self):
        """
        The last fixture case has xs spread over 1e-7, so
        ``sum(x²) - sum(x)²/n`` cancels to ~1e-15 and only a handful of
        significant digits survive in *either* language.  Agreement is asserted
        at the precision that cancellation leaves, not tighter.
        """
        case = REFERENCE["get_slope_cases"][3]
        assert _get_slope(case["xs"], case["ys"]) == pytest.approx(case["slope"], rel=1e-6)
        assert _get_slope_uncertainty(case["xs"]) == pytest.approx(
            case["slope_uncertainty"], rel=1e-6)


# ---------------------------------------------------------------------------
# Linear continuum adjustment
# ---------------------------------------------------------------------------

class TestLinearContinuumAdjustment:
    """``_linear_continuum_adjustment`` vs Korg.jl's ``linear_continuum_adjustment!``."""

    @staticmethod
    def _unpack(case):
        windows = None if case["windows"] is None else [
            (lo, hi) for lo, hi in case["windows"]]
        return (np.array(case["obs_wls"]), windows,
                np.array(case["model_flux_in"]), np.array(case["obs_flux"]),
                np.array(case["obs_err"]), np.array(case["model_flux_out"]))

    #: Both languages solve the *normal equations* of a fit in the basis
    #: (model, model·λ) with λ ≈ 5000 Å, so XᵀWX has condition number ~λ² ≈
    #: 2.5e7 and a double-precision solve retains ~11 digits.  numpy and Julia
    #: reach that limit by slightly different BLAS routes.  The observed
    #: disagreement is 1.2e-11 relative; this is the accuracy the algorithm has,
    #: not a slack tolerance.
    RTOL = 1e-10

    def test_numpy_version(self):
        cases = REFERENCE["linear_continuum_adjustment_cases"]
        assert len(cases) == 3
        for case in cases:
            wls, windows, model, flux, err, expected = self._unpack(case)
            _linear_continuum_adjustment(wls, windows, model, flux, err)
            np.testing.assert_allclose(model, expected, rtol=self.RTOL, atol=0,
                                       err_msg=case["label"])

    def test_jax_version_matches_julia(self):
        """The JAX twin used inside the differentiated objective agrees too."""
        for case in REFERENCE["linear_continuum_adjustment_cases"]:
            wls, windows, model, flux, err, expected = self._unpack(case)
            got = _linear_continuum_adjustment_jax(wls, windows, model, flux, err)
            np.testing.assert_allclose(np.asarray(got), expected, rtol=self.RTOL, atol=0,
                                       err_msg=case["label"])

    def test_jax_and_numpy_versions_agree(self):
        """
        The two implementations must not drift apart from each other.  Same
        conditioning caveat as above: XLA and numpy take different routes
        through the same ill-conditioned normal equations, and agree to ~2e-12.
        """
        for case in REFERENCE["linear_continuum_adjustment_cases"]:
            wls, windows, model, flux, err, _ = self._unpack(case)
            jax_out = np.asarray(
                _linear_continuum_adjustment_jax(wls, windows, model.copy(), flux, err))
            _linear_continuum_adjustment(wls, windows, model, flux, err)
            np.testing.assert_allclose(jax_out, model, rtol=self.RTOL, atol=0,
                                       err_msg=case["label"])

    def test_windows_none_equals_full_range(self):
        """Julia's ``isnothing(windows)`` branch spans the whole spectrum."""
        by_label = {c["label"]: c for c in REFERENCE["linear_continuum_adjustment_cases"]}
        np.testing.assert_allclose(by_label["nothing"]["model_flux_out"],
                                   by_label["full"]["model_flux_out"], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# calculate_EWs — algorithm, isolated from the synthesis port
# ---------------------------------------------------------------------------

def _lines_for(case):
    from korg.linelist import create_line
    return [create_line(wl, log_gf, species, E_lower)
            for wl, log_gf, species, E_lower
            in zip(case["line_wl_angstrom"], case["line_log_gf"],
                   case["line_species"], case["line_E_lower"])]


class TestCalculateEWsAlgorithm:
    """
    ``calculate_EWs``'s extraction algorithm vs Korg.jl's, at rtol 1e-10.

    ``synthesize`` is replaced by a stub returning the *exact* wavelength grid
    and absorption depths Korg.jl integrated (dumped by the generator), so any
    disagreement is in the boundary finding or the trapezoid rule, not in the
    synthesis port.
    """

    @pytest.fixture(params=range(4), ids=lambda i: REFERENCE["ew_isolated"][i]["label"])
    def case(self, request):
        return REFERENCE["ew_isolated"][request.param]

    def test_python_builds_the_same_windows(self, case, monkeypatch):
        """The Python grid must have the same per-window point counts as Julia's."""
        import korg.fit as fit

        captured = {}

        def capture(atm, linelist, wls, A_X, **kw):
            captured["wls"] = np.asarray(wls, dtype=float)
            grid = np.array(case["wl_grid"])
            depth = np.array(case["depth"])
            return FakeSynthesisResult(grid, 1.0 - depth, np.ones_like(grid))

        monkeypatch.setattr(fit, "synthesize", capture)
        calculate_EWs(None, _lines_for(case), np.full(92, 7.5),
                      ew_window_size=case["ew_window_size"],
                      wl_step=case["wl_step"], blend_warn_threshold=np.inf)
        assert len(captured["wls"]) == sum(case["window_lengths"])
        np.testing.assert_allclose(captured["wls"], case["wl_grid"], rtol=1e-12)

    def test_EWs_match_julia(self, case, monkeypatch):
        import korg.fit as fit

        grid = np.array(case["wl_grid"])
        depth = np.array(case["depth"])
        monkeypatch.setattr(
            fit, "synthesize",
            lambda *a, **k: FakeSynthesisResult(grid, 1.0 - depth, np.ones_like(grid)))

        got = calculate_EWs(None, _lines_for(case), np.full(92, 7.5),
                            ew_window_size=case["ew_window_size"],
                            wl_step=case["wl_step"], blend_warn_threshold=np.inf)
        np.testing.assert_allclose(got, case["EWs"], rtol=1e-10, atol=0,
                                   err_msg=case["label"])

    def test_blend_boundary_partitions_the_window(self, case, monkeypatch):
        """
        Julia splits each merged window at the point of least absorption, so the
        EWs of the lines in one window sum to the integral over the whole window.
        """
        import korg.fit as fit

        grid = np.array(case["wl_grid"])
        depth = np.array(case["depth"])
        monkeypatch.setattr(
            fit, "synthesize",
            lambda *a, **k: FakeSynthesisResult(grid, 1.0 - depth, np.ones_like(grid)))
        got = calculate_EWs(None, _lines_for(case), np.full(92, 7.5),
                            ew_window_size=case["ew_window_size"],
                            wl_step=case["wl_step"], blend_warn_threshold=np.inf)

        start = 0
        for length, indices in zip(case["window_lengths"], case["lines_per_window"]):
            sl = slice(start, start + length)
            whole = np.trapezoid(1.0 - (1.0 - depth[sl]), grid[sl]) * 1e3
            subset = sum(got[i - 1] for i in indices)   # Julia indices are 1-based
            assert subset == pytest.approx(whole, rel=1e-10)
            start += length


class TestEwsToAbundancesApproxArithmetic:
    """``ews_to_abundances_approx``'s formula vs Korg.jl's, given fixed EWs."""

    def test_matches_julia(self, monkeypatch):
        import korg.fit as fit
        from korg.linelist import create_line

        case = REFERENCE["ews_to_abundances_approx_case"]
        synth = np.array(case["synth_EWs"])
        lines = [create_line(5000.0 + 10 * i, -1.0, "Fe I", 1.0)
                 for i in range(len(synth))]
        A_X = np.zeros(92)
        A_X[25] = case["A0"][0]

        monkeypatch.setattr(fit, "calculate_EWs", lambda *a, **k: synth)
        got = ews_to_abundances_approx(None, lines, A_X, np.array(case["measured_EWs"]))
        np.testing.assert_allclose(got, case["expected"], rtol=1e-14, atol=0)


# ---------------------------------------------------------------------------
# calculate_EWs — end to end (depends on the whole synthesis port)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestCalculateEWsEndToEnd:
    """
    End-to-end ``calculate_EWs`` against Korg.jl on the committed solar model.

    SLOW: runs real syntheses (~20 s).  Unlike the isolated test above this
    exercises the whole synthesis port, so it is *not* a 1e-10 comparison: the
    tolerance below reflects the accuracy of the ported opacity and radiative
    transfer, not of ``fit.py``.  It is here to catch a gross regression, and is
    deliberately stated as a percentage.
    """

    #: measured agreement is 1.28% at worst across the four fixture cases; this
    #: is the accuracy of the ported opacities and radiative transfer, not of
    #: the EW algorithm (which agrees at 1e-10, see TestCalculateEWsAlgorithm).
    EW_RTOL = 0.02

    @pytest.fixture(scope="class")
    def solar(self):
        if not ATM_FILE.exists():
            raise AssertionError(f"Solar atmosphere fixture missing: {ATM_FILE}")
        return korg.read_model_atmosphere(str(ATM_FILE)), korg.format_A_X()

    @pytest.mark.parametrize("case_index", [0, 1])
    def test_ews_within_a_few_percent(self, solar, case_index):
        atm, A_X = solar
        case = REFERENCE["ew_isolated"][case_index]
        got = calculate_EWs(atm, _lines_for(case), A_X,
                            ew_window_size=case["ew_window_size"],
                            wl_step=case["wl_step"], blend_warn_threshold=np.inf)
        expected = np.array(case["EWs"])
        rel = np.abs(got - expected) / expected
        assert np.all(rel < self.EW_RTOL), (
            f"{case['label']}: Python {got}, Julia {expected}, rel {rel}")
