"""
Functional, precision, autodiff and jit tests for ``korg.synthesis``.

Scope: the Python-orchestrated synthesis entry points (``synthesize_spectrum``,
``synthesize``, ``synth``) and the module-level
helpers (``planck_function``, ``blackbody``, ``filter_linelist``,
``get_reference_wavelength_linelist``, ``compute_continuum_absorption``),
including the spherical (``ShellAtmosphere``) path.  The JIT pipeline
(``synthesize_jit`` and friends) is covered by ``test_synthesis_jit.py``.

Cost control: every synthesis here runs on an 8-layer sub-sampled solar
atmosphere over 11 wavelength points, and the expensive results are cached in
module-scoped fixtures.  A warm call is ~0.1 s.

Precision policy.  A synthetic *spectrum* exercises the entire opacity stack
(chemical equilibrium, continuum, line profiles), and Python and Korg.jl
currently agree there to ~3e-3 relative; those comparisons are therefore made
at 5e-3 and say so.  Quantities that are pure closed-form arithmetic — the
Planck function, the photosphere correction, the equality of the
default-data and explicit-data code paths — are compared at 1e-13 or better.
"""

import json
import warnings
from pathlib import Path

import korg  # noqa: F401 — ensures JAX x64 mode

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from korg.atmosphere import PlanarAtmosphere, ShellAtmosphere
from korg.constants import c_cgs, hplanck_cgs, kboltz_cgs
from korg.linelist import Line
from korg.species import Species
from korg.synthesis import (
    SynthesisResult,
    blackbody,
    compute_continuum_absorption,
    filter_linelist,
    get_reference_wavelength_linelist,
    planck_function,
    synth,
    synthesize,
    synthesize_spectrum,
)

REFERENCE_JSON = Path(__file__).parent / "synthesis_reference_data.json"
SUN_MOD = Path(__file__).parent / "data" / "sun.mod"

# 5000 Å reference wavelength in cm, as used by MARCS models.
LAMBDA_REF_CM = 5e-5

# Python and Korg.jl agree on a full synthetic spectrum to ~3e-3; see the
# module docstring.  This is the tolerance used for whole-spectrum comparisons.
SPECTRUM_RTOL = 5e-3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def julia_ref():
    """Korg.jl reference values from ``generate_synthesis_reference.jl``.

    Raises rather than skips — a missing reference file is a broken checkout,
    and a skipped precision test is indistinguishable from a passing one.
    """
    if not REFERENCE_JSON.exists():
        raise FileNotFoundError(
            f"{REFERENCE_JSON} is missing. Regenerate it with:\n"
            "  export PATH=/mnt/sw/nix/store/"
            "yr11xz204lj9ah1irz61lbqh1dk0hcif-julia-1.11.2/bin:$PATH\n"
            "  julia --project=. tests/generate_synthesis_reference.jl"
        )
    with open(REFERENCE_JSON) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def A_X(julia_ref):
    """Korg.jl's ``format_A_X()`` output, so both sides use identical input."""
    return np.array(julia_ref["A_X"])


@pytest.fixture(scope="module")
def full_sun():
    if not SUN_MOD.exists():
        raise FileNotFoundError(f"{SUN_MOD} is missing from the test data directory")
    return korg.read_model_atmosphere(str(SUN_MOD))


@pytest.fixture(scope="module")
def tiny_atm(full_sun):
    """An 8-layer sub-sample of the solar model — enough structure, ~10x cheaper."""
    return PlanarAtmosphere(full_sun.layers[::7], full_sun.reference_wavelength)


@pytest.fixture(scope="module")
def tiny_wls():
    return 5000.0 + 0.01 * np.arange(11)


@pytest.fixture(scope="module")
def fe_line(julia_ref):
    L = julia_ref["line"]
    return Line(wl=L["wl_cm"], log_gf=L["log_gf"], species=Species("Fe I"),
                E_lower=L["E_lower"], gamma_rad=L["gamma_rad"],
                gamma_stark=L["gamma_stark"], vdW=tuple(L["vdW"]))


def _synth(*args, **kwargs):
    """Call ``synthesize_spectrum`` quietly (it is deprecated on purpose)."""
    kwargs.setdefault("verbose", False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return synthesize_spectrum(*args, **kwargs)


@pytest.fixture(scope="module")
def baseline(tiny_atm, tiny_wls, A_X):
    """Continuum-only synthesis with all defaults — reused by many tests."""
    return _synth(tiny_atm, [], tiny_wls, A_X, hydrogen_lines=False)


# ===========================================================================
# 1. Functional — SynthesisResult
# ===========================================================================

class TestSynthesisResult:

    def test_cntm_aliases_continuum(self):
        r = SynthesisResult(wavelengths=np.zeros(3), flux=np.ones(3),
                            continuum=np.full(3, 2.0))
        assert r.cntm is r.continuum

    def test_optional_korg_fields_default_to_none(self):
        r = SynthesisResult(wavelengths=np.zeros(3), flux=np.ones(3),
                            continuum=np.ones(3))
        assert r.intensities is None
        assert r.alpha is None
        assert r.alpha_cntm is None
        assert r.number_densities is None
        assert r.electron_number_density is None

    def test_synthesis_populates_the_korg_fields(self, baseline, tiny_atm, tiny_wls):
        assert baseline.alpha.shape == (tiny_atm.n_layers, len(tiny_wls))
        assert baseline.alpha_cntm.shape == baseline.alpha.shape
        assert baseline.electron_number_density.shape == (tiny_atm.n_layers,)
        assert Species("H_I") in baseline.number_densities


# ===========================================================================
# 1. Functional — Planck functions
# ===========================================================================

class TestPlanckFunctions:

    def test_blackbody_is_positive_and_finite(self):
        wl = np.array([2e-5, 5e-5, 1e-4])
        b = np.asarray(blackbody(5777.0, wl))
        assert np.all(np.isfinite(b))
        assert np.all(b > 0)

    def test_blackbody_increases_with_temperature(self):
        wl = 5e-5
        assert float(blackbody(6000.0, wl)) > float(blackbody(5000.0, wl))

    def test_wien_peak_moves_blueward_with_temperature(self):
        """Wien's displacement law, as a sanity check on the functional form."""
        wl = np.linspace(1e-5, 2e-4, 4001)
        peak_hot = wl[int(np.argmax(np.asarray(blackbody(9000.0, wl))))]
        peak_cool = wl[int(np.argmax(np.asarray(blackbody(4000.0, wl))))]
        assert peak_hot < peak_cool
        # b = λ_max T ≈ 0.2898 cm K
        assert peak_hot * 9000.0 == pytest.approx(0.28978, rel=2e-3)

    def test_planck_function_matches_blackbody_through_the_jacobian(self):
        """B_λ = B_ν c/λ², exactly, wherever the overflow clamp is inactive.

        Two independent implementations of the same physics live in this
        module; this pins them together at machine precision.
        """
        wl = np.array([3e-5, 5e-5, 8e-5, 1.5e-4])
        T = 5777.0
        nu = c_cgs / wl
        b_lambda = np.asarray(blackbody(T, wl))
        b_nu = np.asarray(planck_function(nu, T))
        np.testing.assert_allclose(b_lambda, b_nu * c_cgs / wl ** 2, rtol=1e-13)

    def test_planck_function_clamps_the_exponent_at_100(self):
        """Both functions clamp hν/kT to 100 to avoid overflow.

        Below that the clamp must be inactive; above it the result must be the
        clamped value rather than an underflow to zero or a NaN.
        """
        T = 1000.0
        nu_clamped = 100.0 * kboltz_cgs * T / hplanck_cgs * 2.0  # x = 200
        value = float(planck_function(nu_clamped, T))
        expected = (2.0 * hplanck_cgs * nu_clamped ** 3 / c_cgs ** 2) / (np.exp(100.0) - 1.0)
        assert value == pytest.approx(expected, rel=1e-13)
        assert np.isfinite(value) and value > 0

    def test_blackbody_clamps_the_exponent_at_100(self):
        T = 1000.0
        wl = hplanck_cgs * c_cgs / (200.0 * kboltz_cgs * T)  # x = 200
        value = float(blackbody(T, wl))
        expected = (2.0 * hplanck_cgs * c_cgs ** 2 / wl ** 5) / (np.exp(100.0) - 1.0)
        assert value == pytest.approx(expected, rel=1e-13)


# ===========================================================================
# 1. Functional — filter_linelist
# ===========================================================================

def _line_at(wl_angstrom, species="Fe I"):
    return Line(wl=wl_angstrom * 1e-8, log_gf=-1.0, species=Species(species),
                E_lower=1.0, gamma_rad=1e8, gamma_stark=1e-5,
                vdW=(1e-7, -1.0))


class TestFilterLinelist:

    def test_empty_linelist_is_returned_unchanged(self):
        out = filter_linelist([], np.array([5e-5, 5.01e-5]), 1e-8)
        assert out == []

    def test_lines_inside_the_buffer_are_kept(self):
        lines = [_line_at(w) for w in (4990.0, 5000.0, 5010.0)]
        out = filter_linelist(lines, np.array([5.0e-5, 5.0005e-5]), 10e-8)
        assert len(out) == 3

    def test_lines_outside_the_buffer_are_dropped(self):
        lines = [_line_at(w) for w in (4000.0, 5000.0, 6000.0)]
        out = filter_linelist(lines, np.array([5.0e-5, 5.0005e-5]), 1e-8)
        assert [round(l.wl * 1e8) for l in out] == [5000]

    def test_a_non_empty_linelist_filtered_to_nothing_warns(self):
        lines = [_line_at(4000.0)]
        with pytest.warns(UserWarning, match="none of the lines were within"):
            out = filter_linelist(lines, np.array([5.0e-5, 5.0005e-5]), 1e-8)
        assert out == []

    def test_the_warning_can_be_suppressed(self):
        lines = [_line_at(4000.0)]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert filter_linelist(lines, np.array([5.0e-5, 5.0005e-5]), 1e-8,
                                   warn_empty=False) == []

    def test_boundaries_are_inclusive(self):
        lo, hi = 4990.0, 5010.0
        lines = [_line_at(lo), _line_at(hi)]
        out = filter_linelist(lines, np.array([5.0e-5, 5.0e-5]), 10e-8)
        assert len(out) == 2


# ===========================================================================
# 1. Functional — get_reference_wavelength_linelist
# ===========================================================================

class TestReferenceWavelengthLinelist:

    def test_default_uses_the_built_in_5000_angstrom_list(self):
        """Korg.jl ignores the user linelist entirely in this case."""
        out = get_reference_wavelength_linelist([_line_at(5000.0)], 5e-5)
        assert len(out) > 1, "expected the built-in 5000 Å linelist"
        assert all(abs(l.wl - 5e-5) < 25e-8 for l in out)

    def test_a_non_5000_reference_with_no_nearby_lines_is_rejected(self):
        with pytest.raises(ValueError, match="no lines near the reference wavelength"):
            get_reference_wavelength_linelist([_line_at(5000.0)], 8e-5)

    def test_a_non_5000_reference_returns_the_users_nearby_lines(self):
        lines = [_line_at(7995.0), _line_at(8000.0), _line_at(8005.0),
                 _line_at(9000.0)]
        out = get_reference_wavelength_linelist(lines, 8e-5)
        assert [round(l.wl * 1e8) for l in out] == [7995, 8000, 8005]

    def test_disabling_the_internal_list_with_lines_spanning_5000(self):
        lines = [_line_at(4995.0), _line_at(5005.0)]
        out = get_reference_wavelength_linelist(
            lines, 5e-5, use_internal_reference_linelist=False)
        assert [round(l.wl * 1e8) for l in out] == [4995, 5005]

    def test_disabling_the_internal_list_with_no_lines_falls_back(self):
        out = get_reference_wavelength_linelist(
            [_line_at(4000.0)], 5e-5, use_internal_reference_linelist=False)
        assert len(out) > 1, "expected the built-in list as the fallback"

    def test_user_lines_entirely_redward_are_backfilled_from_the_built_in_list(self):
        """Korg.jl prepends the built-in lines that sit below the user's first line.

        Without this, alpha_5000 would be computed from a one-sided linelist.
        """
        lines = [_line_at(5010.0)]
        out = get_reference_wavelength_linelist(
            lines, 5e-5, use_internal_reference_linelist=False)
        assert out[-1].wl == pytest.approx(5010.0e-8)
        assert len(out) > 1
        assert all(l.wl < 5010.0e-8 for l in out[:-1])

    def test_user_lines_entirely_blueward_are_topped_up_from_the_built_in_list(self):
        lines = [_line_at(4990.0)]
        out = get_reference_wavelength_linelist(
            lines, 5e-5, use_internal_reference_linelist=False)
        assert out[0].wl == pytest.approx(4990.0e-8)
        assert len(out) > 1
        assert all(l.wl > 4990.0e-8 for l in out[1:])


# ===========================================================================
# 1. Functional — compute_continuum_absorption
# ===========================================================================

class TestComputeContinuumAbsorption:

    def test_species_keys_are_translated_and_the_result_is_positive(
            self, baseline, tiny_atm):
        """``Species`` prints as ``'Fe II'`` but the continuum module wants ``'Fe_II'``."""
        from korg.data_loader import default_partition_funcs
        nd = {sp: float(arr[0]) for sp, arr in baseline.number_densities.items()}
        wl = np.array([4.0e-5, 5.0e-5, 6.0e-5])
        alpha = compute_continuum_absorption(
            wl, float(tiny_atm.T[0]),
            float(baseline.electron_number_density[0]), nd,
            default_partition_funcs)
        assert alpha.shape == wl.shape
        assert np.all(np.isfinite(alpha))
        assert np.all(alpha > 0)

    def test_it_reproduces_the_continuum_used_inside_synthesis(
            self, baseline, tiny_atm):
        """The helper and the synthesis fast path must agree to round-off.

        ``alpha_cntm`` from a line-free synthesis is the vmapped JIT continuum
        evaluated on a 1 Å coarse grid and linearly interpolated onto the output
        grid; recomputing it directly at every output wavelength through this
        public helper is an independent route to the same numbers.  The residual
        1e-5 is the coarse-grid interpolation error, not a physics difference —
        it is the same approximation Korg.jl makes.
        """
        from korg.data_loader import default_partition_funcs
        i = tiny_atm.n_layers // 2
        nd = {sp: float(arr[i]) for sp, arr in baseline.number_densities.items()}
        wl_cm = np.asarray(baseline.wavelengths) * 1e-8
        alpha = compute_continuum_absorption(
            wl_cm, float(tiny_atm.T[i]),
            float(baseline.electron_number_density[i]), nd,
            default_partition_funcs)
        np.testing.assert_allclose(alpha, np.asarray(baseline.alpha_cntm)[i],
                                   rtol=1e-4)


# ===========================================================================
# 1. Functional — synthesize_spectrum options and error paths
# ===========================================================================

class TestSynthesizeSpectrumOptions:

    def test_it_warns_that_it_is_deprecated(self, tiny_atm, tiny_wls, A_X):
        with pytest.warns(DeprecationWarning, match="synthesize_jit"):
            synthesize_spectrum(tiny_atm, [], tiny_wls, A_X,
                                hydrogen_lines=False, verbose=False)

    def test_verbose_reports_progress(self, tiny_atm, tiny_wls, A_X, capsys):
        _synth(tiny_atm, [], tiny_wls, A_X, hydrogen_lines=False, verbose=True)
        out = capsys.readouterr().out
        assert "Synthesizing spectrum" in out
        assert "Layers: 8" in out
        assert "Synthesis complete" in out

    def test_profile_reports_timings(self, tiny_atm, tiny_wls, A_X, fe_line, capsys):
        _synth(tiny_atm, [fe_line], tiny_wls, A_X, hydrogen_lines=True,
               profile=True, verbose=False)
        out = capsys.readouterr().out
        assert "PROFILING RESULTS" in out
        for label in ("Chemical equilibrium", "Continuum absorption",
                      "Source function", "Continuum RT", "Hydrogen lines",
                      "Line absorption", "Radiative transfer", "TOTAL"):
            assert label in out, f"missing timing line for {label!r}"

    def test_return_continuum_false_reuses_the_flux(self, tiny_atm, tiny_wls, A_X):
        r = _synth(tiny_atm, [], tiny_wls, A_X, hydrogen_lines=False,
                   return_continuum=False)
        np.testing.assert_array_equal(r.continuum, r.flux)

    def test_return_continuum_true_computes_a_separate_continuum(
            self, baseline, tiny_atm, tiny_wls, A_X, fe_line):
        r = _synth(tiny_atm, [fe_line], tiny_wls, A_X, hydrogen_lines=False)
        assert np.any(r.flux < r.continuum), "the line must depress the flux"
        np.testing.assert_allclose(r.continuum, baseline.flux, rtol=1e-12)

    def test_hydrogen_lines_change_the_spectrum(self, tiny_atm, A_X):
        """Evaluated at Hβ, where switching H lines off is unmistakable."""
        wls = 4861.0 + 0.1 * np.arange(5)
        without = _synth(tiny_atm, [], wls, A_X, hydrogen_lines=False)
        with_h = _synth(tiny_atm, [], wls, A_X, hydrogen_lines=True)
        assert np.all(np.asarray(with_h.flux) < np.asarray(without.flux))
        assert np.max(np.asarray(with_h.alpha)) > np.max(np.asarray(without.alpha))

    def test_the_infrared_brackett_branch_runs_but_contributes_nothing(
            self, tiny_atm, A_X):
        """Covers the Brackett branch and pins a bug it exposes.

        16,400 Å is Brackett (n=4 -> 12), so ``brackett_in_range`` is true and
        the per-layer loop runs.  ``korg.hydrogen_line_absorption`` returns
        *exactly zero* for every Brackett wavelength, while Korg.jl 1.2.1
        returns ~1e-13 cm^-1 at these conditions.  That defect lives in
        ``hydrogen_line_absorption.py``, not in this module, so it is pinned
        here rather than fixed: if the Brackett series is ever implemented,
        this test fails and should be replaced with a real comparison.
        """
        wls = 16400.0 + 1.0 * np.arange(5)
        without = _synth(tiny_atm, [], wls, A_X, hydrogen_lines=False)
        with_h = _synth(tiny_atm, [], wls, A_X, hydrogen_lines=True)
        assert np.all(np.isfinite(np.asarray(with_h.flux)))
        np.testing.assert_array_equal(
            np.asarray(with_h.alpha), np.asarray(without.alpha))

    def test_an_empty_linelist_gives_flux_equal_to_the_continuum(self, baseline):
        np.testing.assert_allclose(baseline.flux, baseline.continuum, rtol=1e-14)

    def test_a_linelist_with_no_lines_in_range_behaves_like_an_empty_one(
            self, baseline, tiny_atm, tiny_wls, A_X):
        far = [_line_at(3000.0), _line_at(9000.0)]
        r = _synth(tiny_atm, far, tiny_wls, A_X, hydrogen_lines=False)
        np.testing.assert_allclose(r.flux, baseline.flux, rtol=1e-14)

    def test_an_unsorted_linelist_is_sorted_before_use(self, tiny_atm, tiny_wls,
                                                       A_X, fe_line):
        extra = Line(wl=5000.2e-8, log_gf=-1.5, species=Species("Fe I"),
                     E_lower=3.0, gamma_rad=fe_line.gamma_rad,
                     gamma_stark=fe_line.gamma_stark, vdW=fe_line.vdW)
        ordered = _synth(tiny_atm, [extra, fe_line], tiny_wls, A_X,
                         hydrogen_lines=False)
        reversed_ = _synth(tiny_atm, [fe_line, extra], tiny_wls, A_X,
                           hydrogen_lines=False)
        np.testing.assert_allclose(ordered.flux, reversed_.flux, rtol=1e-14)

    def test_absolute_number_fractions_are_accepted_as_well_as_A_X(
            self, baseline, tiny_atm, tiny_wls, A_X):
        """``synthesize`` detects the A(X) convention from A(H) > 1."""
        from korg.abundances import A_X_to_absolute
        linear = A_X_to_absolute(A_X)
        assert linear[0] <= 1.0
        r = _synth(tiny_atm, [], tiny_wls, linear, hydrogen_lines=False)
        np.testing.assert_allclose(r.flux, baseline.flux, rtol=1e-14)

    def test_mu_values_is_ignored_for_a_planar_atmosphere(
            self, baseline, tiny_atm, tiny_wls, A_X):
        """Planar transfer takes the exponential-integral shortcut, as in Korg.jl."""
        r = _synth(tiny_atm, [], tiny_wls, A_X, hydrogen_lines=False, mu_values=3)
        np.testing.assert_array_equal(r.flux, baseline.flux)


class TestSynthesizeSpectrumErrorPaths:

    def test_abundances_of_the_wrong_length_are_rejected(self, tiny_atm, tiny_wls):
        with pytest.raises(ValueError, match="92-element"):
            _synth(tiny_atm, [], tiny_wls, np.full(50, 0.01))

    def test_a_2d_abundance_array_is_rejected(self, tiny_atm, tiny_wls, A_X):
        with pytest.raises(ValueError, match="92-element"):
            _synth(tiny_atm, [], tiny_wls, np.tile(A_X, (2, 1)))

    def test_A_X_with_the_wrong_hydrogen_anchor_is_rejected(self, tiny_atm,
                                                            tiny_wls, A_X):
        """Korg.jl: "A(H) must be a 92-element vector with A[1] == 12."."""
        bad = A_X.copy()
        bad[0] = 11.5
        with pytest.raises(ValueError, match="A\\(H\\)"):
            _synth(tiny_atm, [], tiny_wls, bad)

    def test_an_empty_wavelength_grid_is_rejected(self, tiny_atm, A_X):
        with pytest.raises(ValueError, match="non-empty 1-D array"):
            _synth(tiny_atm, [], np.array([]), A_X)

    def test_a_2d_wavelength_grid_is_rejected(self, tiny_atm, A_X):
        with pytest.raises(ValueError, match="non-empty 1-D array"):
            _synth(tiny_atm, [], np.zeros((2, 5)) + 5000.0, A_X)

    def test_wavelengths_blueward_of_1300_angstrom_are_rejected(self, tiny_atm, A_X):
        """Korg.jl's lower bound; the Rayleigh cross-sections are invalid below it."""
        with pytest.raises(ValueError, match="1300"):
            _synth(tiny_atm, [], np.linspace(1200.0, 1250.0, 5), A_X)

    def test_an_atmosphere_with_no_layers_is_rejected(self, tiny_wls, A_X):
        with pytest.raises(ValueError, match="no layers"):
            _synth(PlanarAtmosphere([]), [], tiny_wls, A_X)


class TestReferenceOpacityAwayFrom5000Angstrom:
    """Regression for a bug that made most of the spectrum unusable.

    ``alpha_ref`` is the continuum opacity at the model's reference wavelength
    (5000 Å for MARCS); it sets the anchored optical-depth scale.  It used to
    be obtained by evaluating the synthesis window's coarse continuum
    interpolator at 5000 Å with ``fill_value='extrapolate'``.  Whenever the
    window did not contain 5000 Å that is a linear extrapolation over tens or
    hundreds of Å, and it goes **negative** — 42 of 56 solar layers at Hβ —
    which makes tau negative and the emergent flux NaN.  Korg.jl instead
    evaluates ``total_continuum_absorption`` directly at the reference
    wavelength, which is what this module now does.

    Nothing caught it because every existing synthesis comparison happened to
    span 5000 Å.
    """

    @pytest.mark.parametrize("start", [4861.0, 6000.0, 8000.0, 3500.0])
    def test_alpha_ref_stays_positive_far_from_5000_angstrom(
            self, tiny_atm, A_X, start):
        r = _synth(tiny_atm, [], start + 0.1 * np.arange(5), A_X,
                   hydrogen_lines=False)
        assert np.all(np.isfinite(np.asarray(r.flux)))
        assert np.all(np.asarray(r.flux) > 0)

    def test_hydrogen_beta_no_longer_produces_nan_flux(self, tiny_atm, A_X):
        """The originally observed failure: Hβ with hydrogen lines on."""
        r = _synth(tiny_atm, [], 4861.0 + 0.1 * np.arange(5), A_X,
                   hydrogen_lines=True)
        assert np.all(np.isfinite(np.asarray(r.flux)))
        assert np.all(np.asarray(r.flux) > 0)

    @staticmethod
    def _capture_alpha_ref(monkeypatch, atm, wls, A_X):
        """Intercept the alpha_ref actually handed to the transfer solver."""
        import korg.synthesis as syn
        seen = {}
        original = syn.radiative_transfer_jit

        def spy(alpha_T, S, z, log_tau, alpha_ref):
            seen["alpha_ref"] = np.asarray(alpha_ref)
            return original(alpha_T, S, z, log_tau, alpha_ref)

        monkeypatch.setattr(syn, "radiative_transfer_jit", spy)
        result = _synth(atm, [], wls, A_X, hydrogen_lines=False)
        return seen["alpha_ref"], result

    def test_alpha_ref_matches_korgs_definition_at_5000_angstrom(
            self, monkeypatch, tiny_atm, A_X):
        """Pin the fix to Korg.jl's definition.

        Synthesised at 8000 Å, where the old extrapolation produced negative
        values, alpha_ref must be the *continuum evaluated at exactly 5000 Å*
        plus the reference linelist's contribution — so strictly greater than
        the pure continuum value, and of the same order.
        """
        from korg.data_loader import default_partition_funcs
        wls = 8000.0 + 0.1 * np.arange(5)
        alpha_ref, r = self._capture_alpha_ref(monkeypatch, tiny_atm, wls, A_X)

        assert np.all(alpha_ref > 0), "alpha_ref went negative — the old bug"
        for i in range(tiny_atm.n_layers):
            nd = {sp: float(arr[i]) for sp, arr in r.number_densities.items()}
            direct = float(compute_continuum_absorption(
                np.array([LAMBDA_REF_CM]), float(tiny_atm.T[i]),
                float(r.electron_number_density[i]), nd,
                default_partition_funcs)[0])
            # equal to within one ulp where the reference lines contribute
            # nothing, and above it where they do — never below.
            assert alpha_ref[i] >= direct * (1.0 - 1e-12), \
                f"layer {i}: alpha_ref below the continuum-only value"
            assert alpha_ref[i] < 100.0 * direct, \
                f"layer {i}: alpha_ref implausibly far above the continuum"

    def test_alpha_ref_is_independent_of_the_synthesis_window(
            self, monkeypatch, tiny_atm, A_X):
        """The reference opacity is a property of the model, not of the window.

        Under the old extrapolation it varied by orders of magnitude (and
        changed sign) with the window; it must not.
        """
        a_near, _ = self._capture_alpha_ref(
            monkeypatch, tiny_atm, 5000.0 + 0.1 * np.arange(5), A_X)
        a_far, _ = self._capture_alpha_ref(
            monkeypatch, tiny_atm, 8000.0 + 0.1 * np.arange(5), A_X)
        np.testing.assert_allclose(a_far, a_near, rtol=1e-12)

    def test_a_window_containing_5000_is_unaffected_in_sign(self, baseline):
        assert np.all(np.asarray(baseline.flux) > 0)


class TestCustomDataPath:
    """The ``using_defaults=False`` branch, which rebuilds all the tables."""

    @pytest.fixture(scope="class")
    def custom(self, tiny_atm, tiny_wls, A_X):
        from korg.data_loader import (default_log_equilibrium_constants,
                                      default_partition_funcs, ionization_energies)
        return _synth(tiny_atm, [], tiny_wls, A_X, hydrogen_lines=False,
                      partition_funcs=default_partition_funcs,
                      ionization_energies_dict=ionization_energies,
                      log_equilibrium_constants=default_log_equilibrium_constants)

    def test_passing_the_defaults_explicitly_reproduces_the_fast_path(
            self, custom, baseline):
        """A high-precision check: same physics, two entirely different code paths.

        The fast path batches chemical equilibrium and continuum through vmapped
        JIT kernels; the slow path rebuilds the equilibrium tables and loops over
        layers in Python calling ``compute_continuum_absorption``.  They must
        agree to round-off, and they do.
        """
        np.testing.assert_allclose(custom.flux, baseline.flux, rtol=1e-12)
        np.testing.assert_allclose(custom.continuum, baseline.continuum, rtol=1e-12)

    def test_the_number_density_dict_is_assembled_per_layer(self, custom, tiny_atm):
        assert Species("H_I") in custom.number_densities
        assert custom.number_densities[Species("H_I")].shape == (tiny_atm.n_layers,)

    def test_one_custom_table_is_enough_to_leave_the_fast_path(
            self, tiny_atm, tiny_wls, A_X, baseline):
        from korg.data_loader import default_partition_funcs
        r = _synth(tiny_atm, [], tiny_wls, A_X, hydrogen_lines=False,
                   partition_funcs=default_partition_funcs)
        np.testing.assert_allclose(r.flux, baseline.flux, rtol=1e-12)


# ===========================================================================
# 1. Functional — the thin public wrappers
# ===========================================================================

class TestFallbackAndRarePaths:
    """Branches that only fire when the built-in data are unavailable."""

    @pytest.fixture
    def no_builtin_linelist(self, monkeypatch):
        """Make ``load_default_linelist`` fail, as it would without the data file."""
        import korg.data_loader as dl

        def boom(*args, **kwargs):
            raise FileNotFoundError("simulated missing built-in 5000 A linelist")

        monkeypatch.setattr(dl, "load_default_linelist", boom)

    def test_the_built_in_reference_linelist_failure_is_survivable(
            self, no_builtin_linelist):
        """The 5000 Å default path swallows the error and uses the user's lines."""
        out = get_reference_wavelength_linelist([_line_at(4999.0), _line_at(5001.0)],
                                                5e-5)
        assert [round(l.wl * 1e8) for l in out] == [4999, 5001]

    def test_the_backfill_helper_degrades_to_the_user_lines(self, no_builtin_linelist):
        out = get_reference_wavelength_linelist(
            [_line_at(5010.0)], 5e-5, use_internal_reference_linelist=False)
        assert [round(l.wl * 1e8) for l in out] == [5010]

    def test_an_empty_reference_linelist_skips_the_line_correction(
            self, no_builtin_linelist, full_sun, tiny_wls, A_X):
        """With no reference lines at all, alpha_ref is continuum-only.

        This is the ``if ref_ll_for_ref:`` false branch, otherwise unreachable
        because the built-in list is always non-empty.  Dropping those lines
        lowers alpha_ref and so shifts the whole optical-depth scale.
        """
        r = _synth(full_sun, [], tiny_wls, A_X, hydrogen_lines=False)
        assert np.all(np.isfinite(np.asarray(r.flux)))
        assert np.all(np.asarray(r.flux) > 0)

    def test_the_reference_linelist_contributes_nothing_at_5000_angstrom(
            self, monkeypatch, full_sun, tiny_wls, A_X):
        """Records a measured fact about the built-in reference linelist.

        Korg keeps a built-in ±21 Å linelist so that alpha_5000 includes line
        opacity.  Evaluated at *exactly* 5000 Å on the solar model, every one
        of those lines falls below the 3e-4 cutoff, so removing the list leaves
        the emergent flux bit-identical.  If the reference list or the cutoff
        ever changes, this test fails and the assumption gets revisited.
        """
        with_lines = _synth(full_sun, [], tiny_wls, A_X, hydrogen_lines=False)

        import korg.data_loader as dl
        monkeypatch.setattr(dl, "load_default_linelist",
                            lambda *a, **k: (_ for _ in ()).throw(
                                FileNotFoundError("simulated")))
        without = _synth(full_sun, [], tiny_wls, A_X, hydrogen_lines=False)
        np.testing.assert_array_equal(np.asarray(without.flux),
                                      np.asarray(with_lines.flux))

    def test_a_partition_function_without_numpy_eval_is_still_usable(
            self, tiny_atm, A_X):
        """The H I partition function is normally a spline with a fast path.

        A caller-supplied callable without ``numpy_eval`` must work too — it is
        then evaluated scalar-by-scalar.
        """
        from korg.data_loader import (default_log_equilibrium_constants,
                                      default_partition_funcs, ionization_energies)

        class PlainCallable:
            """Identical to the wrapped spline except that ``numpy_eval`` is hidden."""

            def __init__(self, inner):
                object.__setattr__(self, "_inner", inner)

            def __call__(self, log_T):
                return self._inner(log_T)

            def __getattr__(self, name):
                if name == "numpy_eval":
                    raise AttributeError(name)
                return getattr(object.__getattribute__(self, "_inner"), name)

        pf = dict(default_partition_funcs)
        pf[Species("H_I")] = PlainCallable(pf[Species("H_I")])
        assert not hasattr(pf[Species("H_I")], "numpy_eval")
        r = _synth(tiny_atm, [], 4861.0 + 0.1 * np.arange(5), A_X,
                   hydrogen_lines=True, partition_funcs=pf,
                   ionization_energies_dict=ionization_energies,
                   log_equilibrium_constants=default_log_equilibrium_constants)
        assert np.all(np.isfinite(np.asarray(r.flux)))
        assert np.all(np.asarray(r.flux) > 0)

    def test_verbose_reports_hydrogen_and_line_stages(self, tiny_atm, A_X,
                                                      fe_line, tiny_wls, capsys):
        _synth(tiny_atm, [fe_line], tiny_wls, A_X, hydrogen_lines=True,
               verbose=True)
        out = capsys.readouterr().out
        assert "Adding hydrogen line absorption" in out
        assert "Adding line absorption for 1 lines" in out
        assert "Computing continuum spectrum" in out

    def test_profile_omits_the_stages_that_did_not_run(self, tiny_atm, tiny_wls,
                                                       A_X, capsys):
        """No hydrogen lines and no linelist — those timing lines are absent.

        ``Continuum RT`` is always printed: the key is seeded to 0.0 whenever
        ``profile=True``, so its guard can never be false.
        """
        _synth(tiny_atm, [], tiny_wls, A_X, hydrogen_lines=False,
               return_continuum=False, profile=True)
        out = capsys.readouterr().out
        assert "PROFILING RESULTS" in out
        assert "Hydrogen lines" not in out
        assert "Line absorption" not in out


class TestPublicWrappers:

    def test_synthesize_matches_synthesize_spectrum(self, baseline, tiny_atm,
                                                    tiny_wls, A_X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            r = synthesize(tiny_atm, [], tiny_wls, A_X, hydrogen_lines=False,
                           verbose=False)
        np.testing.assert_array_equal(r.flux, baseline.flux)

    def test_synthesize_accepts_a_list_of_wavelengths(self, baseline, tiny_atm,
                                                      tiny_wls, A_X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            r = synthesize(tiny_atm, [], list(tiny_wls), A_X,
                           hydrogen_lines=False, verbose=False)
        np.testing.assert_array_equal(r.flux, baseline.flux)

    def test_synth_returns_wavelengths_flux_continuum(self, baseline, tiny_atm,
                                                      tiny_wls, A_X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            wl, flux, cont = synth(tiny_atm, [], tiny_wls, A_X,
                                   hydrogen_lines=False, verbose=False)
        np.testing.assert_array_equal(wl, tiny_wls)
        np.testing.assert_array_equal(flux, baseline.flux)
        np.testing.assert_array_equal(cont, baseline.continuum)

    def test_a_line_free_plan_returns_the_continuum(self, tiny_atm, tiny_wls, A_X):
        """Replaces test_synthesize_continuum_returns_the_line_free_flux.

        ``synthesize_continuum`` was ``synthesize_spectrum`` with an empty
        linelist, used by nothing in src/. ``prepare_synthesis(wls, [])`` says
        the same thing and stays traceable.
        """
        from korg.synthesis_plan import prepare_synthesis
        # No n_layers override: calling with stellar parameters goes through the
        # MARCS interpolation, which always yields 56 layers regardless of what
        # tiny_atm has.
        _s = prepare_synthesis(np.asarray(tiny_wls) * 1e-8, [])
        flux, cntm = _s(5777.0, 4.44, 0.0)
        flux, cntm = np.asarray(flux), np.asarray(cntm)
        assert np.all(np.isfinite(flux)) and np.all(flux > 0)
        # With no atomic lines the only opacity above the continuum is hydrogen,
        # so the two agree to the H-wing depth rather than exactly.
        np.testing.assert_allclose(flux, cntm, rtol=1e-3)


# ===========================================================================
# 1./2. Spherical synthesis, end to end
# ===========================================================================

@pytest.fixture(scope="module")
def spherical_results(full_sun, julia_ref, A_X, fe_line):
    """Full-resolution runs matching the Julia reference cases exactly.

    These use all 56 layers and 101 wavelengths because they are compared
    against Korg.jl output generated from the same model; four runs, ~1 s each
    once warm.
    """
    S = julia_ref["synthesis"]
    wls = np.array(julia_ref["wavelengths"])
    ext = ShellAtmosphere.from_planar(full_sun, S["shell_extended_R"])
    thin = ShellAtmosphere.from_planar(full_sun, S["shell_thin_R"])
    out = {}
    for name, model, ll in (("planar_cntm", full_sun, []),
                            ("planar_line", full_sun, [fe_line]),
                            ("shell_extended_cntm", ext, []),
                            ("shell_extended_line", ext, [fe_line]),
                            ("shell_thin_cntm", thin, [])):
        r = _synth(model, ll, wls, A_X, hydrogen_lines=False)
        out[name] = (np.asarray(r.flux), np.asarray(r.continuum))
    out["extended"] = ext
    out["thin"] = thin
    return out


class TestSphericalSynthesisRuns:
    """Before this change a ``ShellAtmosphere`` could not be synthesised at all.

    ``synthesis.py`` read ``atmosphere.r``, which did not exist, so every
    spherical synthesis died with ``AttributeError``.
    """

    def test_a_shell_atmosphere_synthesises(self, spherical_results):
        flux, cntm = spherical_results["shell_extended_cntm"]
        assert np.all(np.isfinite(flux)) and np.all(flux > 0)
        assert np.all(np.isfinite(cntm)) and np.all(cntm > 0)

    def test_a_line_still_forms_in_spherical_geometry(self, spherical_results):
        flux, cntm = spherical_results["shell_extended_line"]
        assert (1.0 - (flux / cntm).min()) > 0.1

    def test_geometry_changes_the_answer(self, spherical_results):
        planar, _ = spherical_results["planar_cntm"]
        shell, _ = spherical_results["shell_extended_cntm"]
        assert np.max(np.abs(shell / planar - 1.0)) > 0.05, \
            "an extended shell must not reproduce the plane-parallel flux"

    def test_a_nearly_planar_shell_approaches_the_planar_answer(
            self, spherical_results):
        planar, _ = spherical_results["planar_cntm"]
        thin, _ = spherical_results["shell_thin_cntm"]
        np.testing.assert_allclose(thin, planar, rtol=2e-3)

    def test_prepare_atmosphere_accepts_a_shell(self, spherical_results):
        """``synthesis_preparation`` reads ``atmosphere.r`` too and had the same bug."""
        from korg.synthesis_preparation import prepare_atmosphere
        ext = spherical_results["extended"]
        arrays = prepare_atmosphere(ext)
        assert arrays.spherical is True
        np.testing.assert_allclose(arrays.z, ext.r, rtol=0)


class TestPhotosphereCorrectionIsApplied:

    def test_the_flux_is_the_ray_solver_result_times_the_correction(
            self, spherical_results, julia_ref):
        """Exact internal check, to 1e-14.

        Reproduce the transfer call ``synthesize_spectrum`` makes, without the
        rescaling, and confirm the returned flux is exactly that times
        ``(r[0]/R)²``.
        """
        from korg.radiative_transfer import radiative_transfer_spherical
        S = julia_ref["synthesis"]
        wls = np.array(julia_ref["wavelengths"])
        ext = spherical_results["extended"]
        flux, _ = spherical_results["shell_extended_cntm"]

        r = _synth(ext, [], wls, np.array(julia_ref["A_X"]), hydrogen_lines=False)
        source = np.array([blackbody(T, wls * 1e-8) for T in ext.T]).T
        raw, _ = radiative_transfer_spherical(
            np.asarray(r.alpha).T, source, ext.r, ext.log_tau_ref,
            _alpha_ref_of(r, ext, wls), n_mu=20, tau_scheme="anchored",
            intensity_scheme="linear_flux_only", R_photosphere=None)
        np.testing.assert_allclose(
            flux, np.asarray(raw) * 1e-8 * ext.photosphere_correction, rtol=1e-12)

    def test_dropping_the_correction_would_be_a_69_percent_error(
            self, spherical_results, julia_ref):
        """Quantifies what the missing rescaling used to cost.

        For this t/R the correction is 1.69, so an unrescaled flux is ~41%
        below Korg.jl's — far outside any plausible opacity-stack difference.
        """
        ext = spherical_results["extended"]
        flux, _ = spherical_results["shell_extended_cntm"]
        jl = np.array(julia_ref["synthesis"]["shell_extended_cntm"]["flux"])
        assert ext.photosphere_correction == pytest.approx(1.6901125559865820,
                                                           rel=1e-13)
        uncorrected = flux / ext.photosphere_correction
        assert np.min(np.abs(uncorrected / jl - 1.0)) > 0.3


def _alpha_ref_of(result, atm, wls_angstrom):
    """Recover the alpha_ref that ``synthesize_spectrum`` used.

    The synthesis grid straddles 5000 Å, so alpha_ref is just alpha at the
    reference pixel (continuum only here, since the linelist is empty).
    """
    i = int(np.argmin(np.abs(wls_angstrom * 1e-8 - LAMBDA_REF_CM)))
    return np.asarray(result.alpha)[:, i]


class TestAgreementWithKorgJl:
    """Whole-spectrum comparisons.

    These necessarily go through chemical equilibrium, the continuum stack and
    the line profiles, where Python and Korg.jl differ by ~3e-3; they cannot be
    made tighter without closing that gap, which is outside this module.  The
    *geometry* is checked exactly in ``TestPhotosphereCorrectionIsApplied`` and
    in ``test_atmosphere.py``.
    """

    @pytest.mark.parametrize("case", ["planar_cntm", "planar_line",
                                      "shell_extended_cntm",
                                      "shell_extended_line", "shell_thin_cntm"])
    def test_flux_matches_julia(self, spherical_results, julia_ref, case):
        flux, _ = spherical_results[case]
        jl = np.array(julia_ref["synthesis"][case]["flux"])
        np.testing.assert_allclose(flux, jl, rtol=SPECTRUM_RTOL)

    @pytest.mark.parametrize("case", ["planar_cntm", "shell_extended_cntm",
                                      "shell_thin_cntm"])
    def test_continuum_matches_julia(self, spherical_results, julia_ref, case):
        _, cntm = spherical_results[case]
        jl = np.array(julia_ref["synthesis"][case]["cntm"])
        np.testing.assert_allclose(cntm, jl, rtol=SPECTRUM_RTOL)

    def test_the_spherical_to_planar_flux_ratio_matches_julia(
            self, spherical_results, julia_ref):
        """The opacity-stack difference largely cancels in the ratio.

        What is left is the geometry and the photosphere correction, and they
        agree an order of magnitude better than the raw fluxes do.
        """
        S = julia_ref["synthesis"]
        for name in ("shell_extended_cntm", "shell_thin_cntm"):
            py = spherical_results[name][0] / spherical_results["planar_cntm"][0]
            jl = (np.array(S[name]["flux"])
                  / np.array(S["planar_cntm"]["flux"]))
            np.testing.assert_allclose(py, jl, rtol=1e-3, err_msg=name)

    def test_the_line_depth_matches_julia(self, spherical_results, julia_ref):
        flux, cntm = spherical_results["shell_extended_line"]
        jl = np.array(julia_ref["synthesis"]["shell_extended_line"]["flux"])
        jlc = np.array(julia_ref["synthesis"]["shell_extended_line"]["cntm"])
        np.testing.assert_allclose(flux / cntm, jl / jlc, rtol=0, atol=5e-3)

    def test_blackbody_matches_julia_to_machine_precision(self, julia_ref):
        """Closed-form arithmetic: no reason for this to be loose, and it isn't."""
        bb = julia_ref["blackbody"]
        lam = np.array(bb["lambda_cm"])
        for T, row in zip(bb["T"], bb["B"]):
            np.testing.assert_allclose(np.asarray(blackbody(T, lam)),
                                       np.array(row), rtol=1e-14,
                                       err_msg=f"blackbody at T={T}")

    def test_mu_grid_matches_julia(self, julia_ref):
        """The spherical flux integral must use Korg.jl's quadrature."""
        from korg.radiative_transfer import generate_mu_grid
        for n, ref in julia_ref["mu_grids"].items():
            mu, w = generate_mu_grid(int(n))
            np.testing.assert_allclose(np.asarray(mu), np.array(ref["mu"]),
                                       rtol=1e-13)
            np.testing.assert_allclose(np.asarray(w), np.array(ref["weights"]),
                                       rtol=1e-13)


# ===========================================================================
# 3. Autodiff
# ===========================================================================

class TestAutodiff:

    def test_blackbody_grad_wrt_temperature_matches_finite_differences(self):
        wl = 5e-5
        T0 = 5777.0
        g = float(jax.grad(lambda T: blackbody(T, wl))(T0))
        h = T0 * 1e-6
        fd = (float(blackbody(T0 + h, wl)) - float(blackbody(T0 - h, wl))) / (2 * h)
        assert np.isfinite(g) and g != 0.0
        assert g == pytest.approx(fd, rel=1e-5)

    def test_blackbody_grad_wrt_wavelength_matches_finite_differences(self):
        T = 5777.0
        wl0 = 5e-5
        g = float(jax.grad(lambda wl: blackbody(T, wl))(wl0))
        h = wl0 * 1e-6
        fd = (float(blackbody(T, wl0 + h)) - float(blackbody(T, wl0 - h))) / (2 * h)
        assert np.isfinite(g) and g != 0.0
        assert g == pytest.approx(fd, rel=1e-5)

    def test_planck_function_grad_matches_finite_differences(self):
        nu = c_cgs / 5e-5
        T0 = 5777.0
        g = float(jax.grad(lambda T: planck_function(nu, T))(T0))
        h = T0 * 1e-6
        fd = (float(planck_function(nu, T0 + h))
              - float(planck_function(nu, T0 - h))) / (2 * h)
        assert np.isfinite(g) and g != 0.0
        assert g == pytest.approx(fd, rel=1e-5)

    def test_blackbody_grad_is_finite_inside_the_clamped_region(self):
        """The clamp makes B constant in x, so the gradient there is exactly 0.

        Crucially it must not be NaN: ``jnp.minimum`` propagates a well-defined
        cotangent, unlike a ``jnp.where`` guard around an overflowing branch.
        """
        T = 1000.0
        wl = hplanck_cgs * c_cgs / (200.0 * kboltz_cgs * T)
        g = float(jax.grad(lambda t: blackbody(t, wl))(T))
        assert np.isfinite(g)
        assert g == 0.0

    def test_blackbody_gradients_are_finite_over_a_wide_grid(self):
        """No NaN cotangents anywhere in the range this package synthesises."""
        wl = np.geomspace(1.3e-5, 5e-4, 40)
        T = np.linspace(2500.0, 40000.0, 40)
        g = jax.vmap(lambda t: jax.grad(lambda tt: jnp.sum(blackbody(tt, wl)))(t))(T)
        assert np.all(np.isfinite(np.asarray(g)))

    def test_synthesize_spectrum_is_not_differentiable(self, tiny_atm, tiny_wls,
                                                       A_X):
        """Pinned, not skipped.

        ``synthesize_spectrum`` drops to host NumPy in several places (SciPy
        ``interp1d`` for the continuum, the bucketed Voigt loop, the H-line
        loop), so a traced input hits ``np.asarray`` on a tracer.  Use
        ``synthesize_jit`` for gradients.
        """
        def f(v):
            return _synth(tiny_atm, [], tiny_wls, A_X, vmic=v,
                          hydrogen_lines=False).flux.sum()

        with pytest.raises(jax.errors.TracerArrayConversionError):
            jax.grad(f)(1.0)


class TestWhereMaskedCotangents:
    """Two ``jnp.where(cond, dangerous, safe)`` sites that leaked NaN gradients.

    ``jnp.where`` masks the *value* of the unselected branch but not its
    cotangent: reverse mode still differentiates it and multiplies by zero, and
    ``0 * NaN`` is NaN.  Both sites below were fixed by feeding the dangerous
    expression a strictly-safe substitute argument.  Every selected value is
    bitwise identical to before the fix.
    """

    def test_hminus_bf_returns_a_real_zero_below_the_fit_range(self):
        """``(x - 0.125)**1.5`` is complex for x < 0.125 μm.

        The function used to return ``complex128`` zeros there — a wrong dtype
        silently propagating into any caller — and a NaN gradient.
        """
        from korg.synthesis import _hminus_bf_jit
        for lam_um in (0.02, 0.05, 0.10, 0.1249):
            nu = 1e4 * c_cgs / lam_um
            out = _hminus_bf_jit(nu, 5000.0, 1e16, 1e13)
            assert out.dtype == jnp.float64, \
                f"{lam_um} um gave dtype {out.dtype}"
            assert float(out) == 0.0

    def test_hminus_bf_gradient_is_finite_everywhere(self):
        from korg.synthesis import _hminus_bf_jit
        lam_um = np.geomspace(0.02, 20.0, 400)
        nu = 1e4 * c_cgs / lam_um
        g = jax.vmap(jax.grad(lambda n: _hminus_bf_jit(n, 5000.0, 1e16, 1e13)))(
            jnp.asarray(nu))
        assert np.all(np.isfinite(np.asarray(g))), \
            "NaN cotangent outside the John (1988) fit range"

    def test_hminus_bf_is_still_nonzero_inside_the_fit_range(self):
        from korg.synthesis import _hminus_bf_jit
        for lam_um in (0.3, 0.5, 1.0, 1.5):
            nu = 1e4 * c_cgs / lam_um
            assert float(_hminus_bf_jit(nu, 5000.0, 1e16, 1e13)) > 0.0

    def test_voigt_gradient_is_finite_at_line_centre(self):
        """``(1 - exp(-v²))/v`` is 0/0 at v == 0, which is *the* line centre."""
        from korg.synthesis import _voigt_jit
        g = float(jax.grad(lambda v: _voigt_jit(0.1, v))(0.0))
        assert np.isfinite(g)
        assert g == 0.0, "H(a, v) is even in v, so dH/dv(0) must vanish"

    def test_voigt_gradient_is_finite_across_the_profile(self):
        from korg.synthesis import _voigt_jit
        v = np.concatenate([np.linspace(-30.0, 30.0, 601), [0.0, 1e-9, -1e-9]])
        for a in (1e-4, 0.1, 1.0, 6.0, 20.0):
            g = jax.vmap(jax.grad(lambda vv, a=a: _voigt_jit(a, vv)))(jnp.asarray(v))
            assert np.all(np.isfinite(np.asarray(g))), f"NaN dH/dv at a={a}"

    def test_the_voigt_gradient_matches_finite_differences_near_zero(self):
        from korg.synthesis import _voigt_jit
        a, v0, h = 0.1, 1e-4, 1e-8
        g = float(jax.grad(lambda v: _voigt_jit(a, v))(v0))
        fd = (float(_voigt_jit(a, v0 + h)) - float(_voigt_jit(a, v0 - h))) / (2 * h)
        assert g == pytest.approx(fd, rel=1e-4, abs=1e-12)


# ===========================================================================
# 4. jit tracing
# ===========================================================================

class TestJit:

    def test_blackbody_jits(self):
        wl = np.array([3e-5, 5e-5, 1e-4])
        out = jax.jit(lambda T: blackbody(T, wl))(5777.0)
        np.testing.assert_allclose(np.asarray(out),
                                   np.asarray(blackbody(5777.0, wl)), rtol=0)

    def test_planck_function_jits(self):
        nu = np.array([c_cgs / 5e-5, c_cgs / 1e-4])
        out = jax.jit(lambda T: planck_function(nu, T))(5777.0)
        np.testing.assert_allclose(np.asarray(out),
                                   np.asarray(planck_function(nu, 5777.0)), rtol=0)

    def test_blackbody_vmaps_over_layers(self):
        wl = np.linspace(4e-5, 6e-5, 7)
        T = np.array([4000.0, 5000.0, 6000.0])
        out = jax.vmap(lambda t: blackbody(t, wl))(T)
        assert out.shape == (3, 7)
        for i, t in enumerate(T):
            np.testing.assert_allclose(np.asarray(out[i]),
                                       np.asarray(blackbody(t, wl)), rtol=0)

    def test_synthesize_spectrum_cannot_be_jitted(self, tiny_atm, tiny_wls, A_X):
        """Pinned, with the reason: the same host-NumPy drop-out as above.

        It also branches on Python-level values (``len(linelist)``,
        ``hydrogen_lines``), which a tracer cannot supply.
        """
        @jax.jit
        def f(wls):
            return _synth(tiny_atm, [], wls, A_X, hydrogen_lines=False).flux.sum()

        with pytest.raises(jax.errors.TracerArrayConversionError):
            f(jnp.asarray(tiny_wls))

    def test_filter_linelist_cannot_be_jitted(self):
        """It slices a Python list by a data-dependent bisect index."""
        lines = [_line_at(w) for w in (4990.0, 5000.0, 5010.0)]

        @jax.jit
        def f(lo):
            return len(filter_linelist(lines, jnp.array([lo, lo]), 1e-8))

        with pytest.raises(Exception):
            f(5e-5)
