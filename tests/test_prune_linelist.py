"""
Tests for ``korg.prune_linelist`` (``merge_close_lines`` and ``prune_linelist``).

1. Functional — grouping rules, merge distances, wavelength windows, option
   combinations, empty/single/duplicate inputs.
2. Julia agreement — ``merge_close_lines`` is compared field-by-field against
   Korg.jl v1.2.1 at rtol 1e-13, and ``prune_linelist`` is compared against
   Korg.jl running the *same* solar atmosphere, abundances, linelist and
   thresholds (retained lines and their EW ordering must match exactly).
3. Autodiff — the gf-weighting inside ``merge_close_lines`` is reimplemented as a
   differentiable kernel and checked against ``jax.grad``/finite differences.
4. jit-tracing — both public functions operate on Python lists of frozen
   dataclasses and (for ``prune_linelist``) run a whole synthesis, so neither can
   be traced; that is pinned rather than left ambiguous.

``prune_linelist`` needs the solar model atmosphere fixture and runs several real
syntheses, so those tests are marked slow.
"""

import json
import math
from pathlib import Path

import korg  # noqa: F401 — enables JAX x64 mode

import jax
import jax.numpy as jnp
import numpy as np
import pytest

REFERENCE_FILE = Path(__file__).parent / "linelist_reference_data.json"
ATM_FILE = Path(__file__).parent / "data" / "sun.mod"


@pytest.fixture(scope="module")
def prune_ref():
    if not REFERENCE_FILE.exists():
        pytest.fail(
            f"{REFERENCE_FILE} is missing. Regenerate with "
            "`julia --project=. tests/generate_linelist_reference.jl`."
        )
    with open(REFERENCE_FILE) as f:
        return json.load(f)


def make_lines(specs):
    """``specs`` is a list of (wl_angstrom, log_gf, species_str, E_lower)."""
    from korg.linelist import create_line

    return [create_line(wl, log_gf, sp, E) for wl, log_gf, sp, E in specs]


# ===========================================================================
# merge_close_lines
# ===========================================================================

class TestMergeCloseLinesAgainstJulia:

    def test_all_merge_distances_match_julia(self, prune_ref):
        from korg.prune_linelist import merge_close_lines

        block = prune_ref["merge_close_lines"]
        lines = make_lines([tuple(row) for row in block["inputs"]])
        for distance_str, expected in block["outputs"].items():
            got = merge_close_lines(lines, merge_distance=float(distance_str))
            assert len(got) == len(expected), (
                f"merge_distance={distance_str}: {len(got)} groups, "
                f"Julia found {len(expected)}"
            )
            for g, e in zip(got, expected):
                assert np.isclose(g[0], e[0], rtol=1e-13), f"mean wl @ {distance_str}"
                assert np.isclose(g[1], e[1], rtol=1e-13), f"wl_low @ {distance_str}"
                assert np.isclose(g[2], e[2], rtol=1e-13), f"wl_high @ {distance_str}"
                assert g[3] == e[3], f"species @ {distance_str}"


class TestMergeCloseLines:

    def test_empty_linelist(self):
        from korg.prune_linelist import merge_close_lines

        assert merge_close_lines([]) == []

    def test_single_line(self):
        from korg.prune_linelist import merge_close_lines

        lines = make_lines([(5000.0, -1.0, "Fe I", 1.0)])
        got = merge_close_lines(lines)
        assert got == [(5000.0, 5000.0, 5000.0, "Fe I")]

    def test_duplicate_wavelengths_merge(self):
        from korg.prune_linelist import merge_close_lines

        lines = make_lines([(5000.0, -1.0, "Fe I", 1.0),
                            (5000.0, -1.0, "Fe I", 2.0)])
        got = merge_close_lines(lines)
        assert len(got) == 1
        assert np.isclose(got[0][0], 5000.0, rtol=1e-13)

    def test_different_species_never_merge(self):
        from korg.prune_linelist import merge_close_lines

        lines = make_lines([(5000.0, -1.0, "Fe I", 1.0),
                            (5000.01, -1.0, "Fe II", 1.0),
                            (5000.02, -1.0, "Ca I", 1.0)])
        got = merge_close_lines(lines, merge_distance=1.0)
        assert len(got) == 3
        assert sorted(t[3] for t in got) == ["Ca I", "Fe I", "Fe II"]

    def test_gf_weighting(self):
        """The reported wavelength is the gf-weighted mean of the group."""
        from korg.prune_linelist import merge_close_lines

        lines = make_lines([(5000.0, 0.0, "Fe I", 1.0),
                            (5000.10, 1.0, "Fe I", 1.0)])
        got = merge_close_lines(lines, merge_distance=0.2)
        expected = (5000.0 * 1.0 + 5000.10 * 10.0) / 11.0
        assert len(got) == 1
        assert np.isclose(got[0][0], expected, rtol=1e-12)

    def test_output_is_sorted_by_wavelength(self):
        from korg.prune_linelist import merge_close_lines

        lines = make_lines([(5002.0, -1.0, "Fe I", 1.0),
                            (5000.0, -1.0, "Ca I", 1.0),
                            (5001.0, -1.0, "Ti II", 1.0)])
        got = merge_close_lines(lines)
        assert [t[0] for t in got] == sorted(t[0] for t in got)

    def test_unsorted_input_is_handled(self):
        from korg.prune_linelist import merge_close_lines

        forward = make_lines([(5000.0, -1.0, "Fe I", 1.0), (5000.1, -1.0, "Fe I", 1.0)])
        backward = list(reversed(forward))
        assert merge_close_lines(forward) == merge_close_lines(backward)

    def test_merge_distance_controls_grouping(self):
        """Korg.jl merges when ``Δλ < merge_distance``."""
        from korg.prune_linelist import merge_close_lines

        lines = make_lines([(5000.0, -1.0, "Fe I", 1.0), (5000.2, -1.0, "Fe I", 1.0)])
        assert len(merge_close_lines(lines, merge_distance=0.19)) == 2
        assert len(merge_close_lines(lines, merge_distance=0.21)) == 1

    def test_chained_groups_break_on_the_first_large_gap(self):
        from korg.prune_linelist import merge_close_lines

        lines = make_lines([(5000.0, -1.0, "Fe I", 1.0),
                            (5000.1, -1.0, "Fe I", 1.0),
                            (5000.2, -1.0, "Fe I", 1.0),
                            (5001.0, -1.0, "Fe I", 1.0)])
        got = merge_close_lines(lines, merge_distance=0.15)
        assert len(got) == 2
        assert np.isclose(got[0][1], 5000.0, rtol=1e-13)
        assert np.isclose(got[0][2], 5000.2, rtol=1e-13)
        assert np.isclose(got[1][0], 5001.0, rtol=1e-13)

    def test_wl_low_and_high_bracket_the_mean(self):
        from korg.prune_linelist import merge_close_lines

        block_lines = make_lines([(5000.0, -1.0, "Fe I", 1.0),
                                  (5000.1, 0.5, "Fe I", 1.0),
                                  (5000.15, -2.0, "Fe I", 1.0)])
        for mean_wl, lo, hi, _ in merge_close_lines(block_lines, merge_distance=0.2):
            assert lo <= mean_wl <= hi

    def test_species_string_is_a_string(self):
        from korg.prune_linelist import merge_close_lines

        got = merge_close_lines(make_lines([(5000.0, -1.0, "Fe I", 1.0)]))
        assert isinstance(got[0][3], str)


# ===========================================================================
# prune_linelist
# ===========================================================================

PRUNE_LINES = [(5000.0, -1.5, "Fe I", 1.0),
               (5000.5, -6.0, "Fe I", 4.0),
               (5001.0, -0.5, "Fe I", 0.9),
               (5010.0, -1.0, "Ca I", 2.0)]


@pytest.fixture(scope="module")
def solar_atm():
    if not ATM_FILE.exists():
        pytest.fail(
            f"The solar model atmosphere fixture {ATM_FILE} is missing; "
            "prune_linelist cannot be tested without an atmosphere."
        )
    return korg.read_model_atmosphere(str(ATM_FILE))


@pytest.fixture(scope="module")
def solar_A_X():
    return korg.format_A_X()


@pytest.fixture(scope="module")
def prune_lines():
    return make_lines(PRUNE_LINES)


@pytest.mark.slow
class TestPruneLinelistAgainstJulia:

    def test_unsorted_matches_julia(self, prune_ref, solar_atm, solar_A_X, prune_lines):
        from korg.prune_linelist import prune_linelist

        expected = prune_ref["prune_linelist"]["unsorted_wls"]
        got = prune_linelist(solar_atm, prune_lines, solar_A_X, (4999.0, 5002.0),
                             threshold=0.1, sort_by_EW=False, verbose=False)
        assert [round(l.wl * 1e8, 6) for l in got] == [round(w, 6) for w in expected]

    def test_sort_by_EW_matches_julia(self, prune_ref, solar_atm, solar_A_X, prune_lines):
        from korg.prune_linelist import prune_linelist

        expected = prune_ref["prune_linelist"]["sorted_by_EW_wls"]
        got = prune_linelist(solar_atm, prune_lines, solar_A_X, (4999.0, 5002.0),
                             threshold=0.1, sort_by_EW=True, verbose=False)
        assert [round(l.wl * 1e8, 6) for l in got] == [round(w, 6) for w in expected]

    def test_loose_threshold_matches_julia(self, prune_ref, solar_atm, solar_A_X,
                                           prune_lines):
        from korg.prune_linelist import prune_linelist

        expected = prune_ref["prune_linelist"]["loose_threshold_wls"]
        got = prune_linelist(solar_atm, prune_lines, solar_A_X, (4999.0, 5002.0),
                             threshold=1e-8, sort_by_EW=False, verbose=False)
        assert [round(l.wl * 1e8, 6) for l in got] == [round(w, 6) for w in expected]

    def test_max_distance_matches_julia(self, prune_ref, solar_atm, solar_A_X,
                                        prune_lines):
        """``max_distance`` widens the window, pulling in the 5010 Å Ca I line."""
        from korg.prune_linelist import prune_linelist

        expected = prune_ref["prune_linelist"]["max_distance_20_wls"]
        got = prune_linelist(solar_atm, prune_lines, solar_A_X, (4999.0, 5002.0),
                             threshold=1e-8, sort_by_EW=False, max_distance=20.0,
                             verbose=False)
        assert [round(l.wl * 1e8, 6) for l in got] == [round(w, 6) for w in expected]


@pytest.mark.slow
class TestPruneLinelistFunctional:

    def test_empty_linelist_returns_empty(self, solar_atm, solar_A_X):
        from korg.prune_linelist import prune_linelist

        assert prune_linelist(solar_atm, [], solar_A_X, (4999.0, 5001.0),
                              sort_by_EW=False, verbose=False) == []

    def test_all_lines_outside_the_window_are_dropped(self, solar_atm, solar_A_X,
                                                      prune_lines):
        from korg.prune_linelist import prune_linelist

        assert prune_linelist(solar_atm, prune_lines, solar_A_X, (5500.0, 5502.0),
                              sort_by_EW=False, verbose=False) == []

    def test_explicit_wavelength_array_accepted(self, solar_atm, solar_A_X, prune_lines):
        from korg.prune_linelist import prune_linelist

        wls = np.linspace(4999.0, 5002.0, 1000)
        got = prune_linelist(solar_atm, prune_lines, solar_A_X, wls,
                             threshold=0.1, sort_by_EW=False, verbose=False)
        assert [round(l.wl * 1e8, 6) for l in got] == [5000.0, 5001.0]

    def test_high_threshold_removes_everything(self, solar_atm, solar_A_X, prune_lines):
        from korg.prune_linelist import prune_linelist

        assert prune_linelist(solar_atm, prune_lines, solar_A_X, (4999.0, 5002.0),
                              threshold=1e12, sort_by_EW=False, verbose=False) == []

    def test_unsorted_output_is_in_wavelength_order(self, solar_atm, solar_A_X,
                                                    prune_lines):
        from korg.prune_linelist import prune_linelist

        got = prune_linelist(solar_atm, list(reversed(prune_lines)), solar_A_X,
                             (4999.0, 5002.0), threshold=1e-8, sort_by_EW=False,
                             verbose=False)
        assert [l.wl for l in got] == sorted(l.wl for l in got)

    def test_result_is_a_subset_of_the_input(self, solar_atm, solar_A_X, prune_lines):
        from korg.prune_linelist import prune_linelist

        got = prune_linelist(solar_atm, prune_lines, solar_A_X, (4999.0, 5002.0),
                             threshold=1e-8, sort_by_EW=False, verbose=False)
        assert all(l in prune_lines for l in got)

    def test_species_absent_from_chemical_equilibrium_is_dropped(
            self, solar_atm, solar_A_X, monkeypatch):
        """
        A species that chemical equilibrium does not report gets n/Z = 0, so its
        line-centre opacity is zero and it can never pass the threshold.

        Korg's default network happens to cover every molecule we can name, so
        the fallback is triggered by removing one species from the synthesis
        result — that is exactly the state the guard exists for.
        """
        from korg import synthesis
        from korg.prune_linelist import prune_linelist
        from korg.species import Species

        real = synthesis.synthesize_spectrum
        target = Species("FeH")

        def without_FeH(*args, **kwargs):
            sol = real(*args, **kwargs)
            if sol.number_densities is not None:
                sol.number_densities.pop(target, None)
                sol.number_densities.pop(str(target), None)
            return sol

        monkeypatch.setattr(synthesis, "synthesize_spectrum", without_FeH)

        lines = make_lines([(5000.0, -1.5, "Fe I", 1.0), (5000.7, 0.0, "FeH", 0.2)])
        got = prune_linelist(solar_atm, lines, solar_A_X, (4999.0, 5002.0),
                             threshold=0.1, sort_by_EW=False, verbose=False)
        assert all(l.species != target for l in got)
        assert [round(l.wl * 1e8, 6) for l in got] == [5000.0]

    def test_synthesis_kwargs_are_forwarded(self, solar_atm, solar_A_X, prune_lines):
        """``hydrogen_lines=False`` must reach ``synthesize_spectrum``."""
        from korg.prune_linelist import prune_linelist

        got = prune_linelist(solar_atm, prune_lines, solar_A_X, (4999.0, 5002.0),
                             threshold=0.1, sort_by_EW=False, verbose=False,
                             hydrogen_lines=False)
        assert [round(l.wl * 1e8, 6) for l in got] == [5000.0, 5001.0]


# ===========================================================================
# 3. Autodiff
# ===========================================================================

def _gf_weighted_mean(wls, log_gfs):
    """Differentiable form of the gf weighting used by merge_close_lines."""
    gf = 10.0 ** log_gfs
    return jnp.sum(wls * gf) / jnp.sum(gf)


class TestAutodiff:

    def test_gf_weighted_mean_matches_merge_close_lines(self):
        from korg.prune_linelist import merge_close_lines

        wls = jnp.array([5000.0, 5000.1, 5000.15])
        log_gfs = jnp.array([-1.0, 0.5, -2.0])
        lines = make_lines([(5000.0, -1.0, "Fe I", 1.0),
                            (5000.1, 0.5, "Fe I", 1.0),
                            (5000.15, -2.0, "Fe I", 1.0)])
        merged = merge_close_lines(lines, merge_distance=0.2)
        assert len(merged) == 1
        assert np.isclose(merged[0][0], float(_gf_weighted_mean(wls, log_gfs)),
                          rtol=1e-12)

    def test_gradient_wrt_log_gf_is_finite_and_nonzero(self):
        wls = jnp.array([5000.0, 5000.1, 5000.15])
        g = jax.grad(lambda lg: _gf_weighted_mean(wls, lg))(
            jnp.array([-1.0, 0.5, -2.0]))
        g = np.asarray(g)
        assert np.all(np.isfinite(g))
        assert np.any(g != 0.0)
        # weights sum to one, so the gradient of the mean w.r.t. all log gf sums to 0
        assert abs(float(np.sum(g))) < 1e-9

    def test_gradient_matches_central_differences(self):
        # Offset the wavelengths so that the finite difference is not swamped by
        # the cancellation of two ~5000 Å numbers.  The derivative is unchanged:
        # a constant offset contributes a constant to the weighted mean.
        wls = jnp.array([0.0, 0.1, 0.15])
        x0 = np.array([-1.0, 0.5, -2.0])
        f = lambda lg: float(_gf_weighted_mean(wls, jnp.asarray(lg)))
        grad = np.asarray(jax.grad(lambda lg: _gf_weighted_mean(wls, lg))(jnp.asarray(x0)))
        h = 1e-5
        for i in range(len(x0)):
            xp, xm = x0.copy(), x0.copy()
            xp[i] += h
            xm[i] -= h
            fd = (f(xp) - f(xm)) / (2 * h)
            assert np.isclose(grad[i], fd, rtol=1e-5, atol=1e-12)

    def test_gradient_wrt_wavelength_is_the_normalised_weight(self):
        log_gfs = jnp.array([-1.0, 0.5, -2.0])
        wls = jnp.array([5000.0, 5000.1, 5000.15])
        g = np.asarray(jax.grad(lambda w: _gf_weighted_mean(w, log_gfs))(wls))
        gf = 10.0 ** np.asarray(log_gfs)
        assert np.allclose(g, gf / gf.sum(), rtol=1e-12)
        assert np.isclose(g.sum(), 1.0, rtol=1e-12)

    def test_single_line_group_has_unit_sensitivity(self):
        """A one-line group's mean is that line's wavelength: dμ/dλ = 1, dμ/dlog gf = 0."""
        g_wl = float(jax.grad(lambda w: _gf_weighted_mean(jnp.array([w]),
                                                          jnp.array([-1.0])))(5000.0))
        g_gf = float(jax.grad(lambda lg: _gf_weighted_mean(jnp.array([5000.0]),
                                                           jnp.array([lg])))(-1.0))
        assert np.isclose(g_wl, 1.0, rtol=1e-14)
        assert abs(g_gf) < 1e-9


# ===========================================================================
# 4. jit-tracing
# ===========================================================================

class TestJIT:

    def test_gf_weighted_mean_jits(self):
        wls = jnp.array([5000.0, 5000.1])
        log_gfs = jnp.array([-1.0, 0.5])
        jitted = jax.jit(_gf_weighted_mean)
        assert np.isclose(float(jitted(wls, log_gfs)),
                          float(_gf_weighted_mean(wls, log_gfs)), rtol=1e-14)

    def test_merge_close_lines_cannot_be_jitted(self):
        """
        ``merge_close_lines`` walks a Python list of frozen dataclasses, keys a
        dict by ``Species`` and branches on ``Δλ < merge_distance``.  None of that
        is expressible as a JAX computation, so tracing it must fail.
        """
        from korg.prune_linelist import merge_close_lines

        lines = make_lines([(5000.0, -1.0, "Fe I", 1.0), (5000.1, -1.0, "Fe I", 1.0)])
        with pytest.raises(Exception):
            jax.jit(lambda d: merge_close_lines(lines, merge_distance=d))(0.2)

    def test_prune_linelist_cannot_be_jitted(self):
        """
        ``prune_linelist`` runs whole syntheses, indexes with ``np.argmax`` on
        concrete arrays and returns a Python list of Line objects; it is a host
        driver, not a kernel.
        """
        from korg.prune_linelist import prune_linelist

        with pytest.raises(Exception):
            jax.jit(lambda t: prune_linelist(None, [], None, (5000.0, 5001.0),
                                             threshold=t))(0.1)


# ===========================================================================
# Shared-implementation guards
# ===========================================================================

class TestNoDuplicateImplementations:

    def test_prune_linelist_uses_the_canonical_kernels(self):
        """
        ``prune_linelist`` used to carry its own (dead, and numerically wrong)
        copy of ``sigma_line`` and to import ``doppler_width`` from a module that
        does not define it.  Both must now come from ``korg.line_absorption``.
        """
        import importlib
        import inspect

        # NB: ``from korg import prune_linelist`` resolves to the *function*
        # re-exported by korg/__init__.py, so import the module explicitly.
        module = importlib.import_module("korg.prune_linelist")
        src = inspect.getsource(module.prune_linelist)
        assert "from .line_absorption import doppler_width, sigma_line" in src
        assert "def sigma_line" not in src, "prune_linelist must not redefine sigma_line"
        assert "line_broadening" not in src

    def test_sigma_line_agrees_with_the_analytic_expression(self):
        from korg.constants import c_cgs, electron_charge_cgs, electron_mass_cgs
        from korg.line_absorption import sigma_line

        wl = 5e-5
        expected = (math.pi * electron_charge_cgs ** 2 /
                    (electron_mass_cgs * c_cgs)) * wl ** 2 / c_cgs
        assert np.isclose(float(sigma_line(wl)), expected, rtol=1e-14)
