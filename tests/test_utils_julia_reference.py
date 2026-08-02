"""
High-precision agreement with Korg.jl 1.2.1 for the utility modules.

Covers ``korg.species``, ``korg.wavelengths``, ``korg.abundances`` and
``korg.cubic_splines`` against ``tests/utils_reference_data.json``, which is
produced by ``tests/generate_utils_reference.jl``.

Everything checked here is deterministic on both sides -- integer atom vectors,
parsed charges, spline coefficients and abundance vectors are the same
arithmetic in the same order -- so the tolerances are exact equality or 1e-13
relative, never looser.  Where Python and Korg.jl genuinely disagree the
divergence is called out by name in the relevant test rather than papered over
with a wide tolerance.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from korg.abundances import (
    ASPLUND_2009_SOLAR_ABUNDANCES,
    ASPLUND_2020_SOLAR_ABUNDANCES,
    BERGEMANN_2025_SOLAR_ABUNDANCES,
    DEFAULT_ALPHA_ELEMENTS,
    DEFAULT_SOLAR_ABUNDANCES,
    GREVESSE_2007_SOLAR_ABUNDANCES,
    format_A_X,
    get_alpha_H,
    get_metals_H,
)
from korg.cubic_splines import cubic_spline
from korg.species import Formula, Species, all_atomic_species
from korg.wavelengths import Wavelengths

REFERENCE_FILE = Path(__file__).parent / "utils_reference_data.json"


@pytest.fixture(scope="module")
def ref():
    """Korg.jl reference values.

    Deliberately raises rather than skipping: a silently skipped reference
    comparison is indistinguishable from a passing one in the summary line, and
    the file is checked in, so its absence is a broken checkout rather than an
    optional extra.
    """
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"{REFERENCE_FILE} is missing. Regenerate it with\n"
            f"    julia --project=. tests/generate_utils_reference.jl"
        )
    with open(REFERENCE_FILE) as f:
        return json.load(f)


def _atoms(obj):
    """Atom vector of a Species or Formula as a list of Python ints."""
    formula = obj.formula if isinstance(obj, Species) else obj
    return [int(a) for a in formula.atoms]


# ---------------------------------------------------------------------------
# Species
# ---------------------------------------------------------------------------


class TestSpeciesParsingReference:
    """Every species code form Korg.jl accepts must parse identically."""

    def test_reference_covers_every_documented_form(self, ref):
        """Guard against the reference silently shrinking."""
        codes = set(ref["species_parsing"]["outputs"])
        # the forms listed in Korg.jl's own Species docstring
        for documented in ("H I", "H 1", "H     1", "H_1", "H.I", "H 2", "H2",
                           "H", "01.00", "02.01", "02.1000"):
            assert documented in codes, f"{documented!r} dropped from the reference"
        assert len(codes) >= 50

    def test_all_species_codes_parse_identically(self, ref):
        """String form, charge, atom vector, n_atoms and ismolecule all match."""
        for code, expected in ref["species_parsing"]["outputs"].items():
            species = Species(code)
            assert str(species) == expected["string"], code
            assert species.charge == expected["charge"], code
            assert _atoms(species) == expected["atoms"], code
            assert species.is_molecule() is bool(expected["is_molecule"]), code
            assert species.n_atoms() == expected["n_atoms"], code

    def test_species_masses_match_exactly(self, ref):
        """``get_mass`` sums the same table entries in the same order."""
        for code, expected in ref["species_parsing"]["outputs"].items():
            assert float(Species(code).get_mass()) == expected["mass"], code

    @pytest.mark.parametrize("code", ["02.1000", "H 10", "Fe 20"])
    def test_trailing_zeros_are_stripped_like_julia(self, ref, code):
        """``strip(code, ['0', ' '])`` strips *trailing* zeros too.

        This is the whole reason ``02.1000`` is He II and not "He with charge
        1000".  A port that only strips leading zeros parses these three codes
        wrongly and nothing else in the suite notices, because ordinary VALD
        and MOOG linelists never emit a trailing-zero charge tag.
        """
        expected = ref["species_parsing"]["outputs"][code]
        assert str(Species(code)) == expected["string"]
        assert Species(code).charge == expected["charge"]

    @pytest.mark.parametrize("code", ["H-", "OH+", "CH+", "H2+", "CN-"])
    def test_charge_suffix_forms(self, ref, code):
        """"X-" is charge -1 and "X+" is charge +1, via the " 0"/" 2" rewrite."""
        expected = ref["species_parsing"]["outputs"][code]
        species = Species(code)
        assert species.charge == expected["charge"]
        assert str(species) == expected["string"]

    def test_anion_round_trips_through_its_own_string(self, ref):
        """``str(Species("H-")) == "H-"`` must itself re-parse to H-."""
        assert str(Species(str(Species("H-")))) == "H-"
        assert Species(str(Species("H-"))).charge == -1

    def test_species_from_float_matches(self, ref):
        """MOOG codes arrive as floats; ``str(float)`` must parse the same."""
        for code, expected in ref["species_from_float"].items():
            species = Species(float(code))
            assert str(species) == expected["string"], code
            assert species.charge == expected["charge"], code
            assert _atoms(species) == expected["atoms"], code

    def test_malformed_codes_are_rejected(self, ref):
        """Codes Korg.jl throws on must raise here too."""
        for code, julia_threw in ref["species_errors"].items():
            assert julia_threw, f"reference says Julia accepted {code!r}"
            with pytest.raises((ValueError, KeyError)):
                Species(code)

    def test_all_atomic_species_matches(self, ref):
        """Same 275 atomic species (Korg.jl iterates charge-major, we do not)."""
        got = sorted(str(s) for s in all_atomic_species())
        assert len(got) == ref["all_atomic_species"]["count"]
        assert got == sorted(ref["all_atomic_species"]["strings"])


class TestFormulaParsingReference:
    """Formula parsing: symbols, molecules and MOOG numeric codes."""

    def test_all_formula_codes_parse_identically(self, ref):
        for code, expected in ref["formula_parsing"]["outputs"].items():
            formula = Formula(code)
            assert str(formula) == expected["string"], code
            assert _atoms(formula) == expected["atoms"], code
            assert formula.n_atoms() == expected["n_atoms"], code
            assert formula.is_molecule() is bool(expected["is_molecule"]), code

    def test_formula_masses_match_exactly(self, ref):
        for code, expected in ref["formula_parsing"]["outputs"].items():
            assert float(Formula(code).get_mass()) == expected["mass"], code

    @pytest.mark.parametrize("code", ["0608", "0801", "0106", "26", "01",
                                      "060606", "080808", "812", "10608",
                                      "0106080808"])
    def test_numeric_codes(self, ref, code):
        """MOOG-style numeric codes.

        ``10608`` has an odd digit count, which exercises the leading-zero pad
        in the >4-digit branch; ``0106080808`` is the 5-nucleus case.
        """
        expected = ref["formula_parsing"]["outputs"][code]
        assert str(Formula(code)) == expected["string"]
        assert _atoms(Formula(code)) == expected["atoms"]


# ---------------------------------------------------------------------------
# Wavelengths
# ---------------------------------------------------------------------------


# (reference key, constructor argument, kwargs)
_WAVELENGTH_CASES = [
    ("5000_5001", (5000, 5001), {}),
    ("5000_5500_1", (5000, 5500, 1.0), {}),
    ("4000_4010_0.1", (4000, 4010, 0.1), {}),
    ("multi", [(5000, 5010, 1.0), (6000, 6010, 1.0)], {}),
    ("multi3", [(4000, 4002, 0.5), (5000, 5002, 0.5), (6000, 6002, 0.5)], {}),
    ("from_vector", np.linspace(5000.0, 5010.0, 11), {}),
    ("single_value", np.array([5000.0]), {}),
]


class TestWavelengthsReference:
    """``Wavelengths`` grids must be the same cm values Korg.jl produces."""

    @pytest.mark.parametrize("key,spec,kwargs", _WAVELENGTH_CASES,
                             ids=[c[0] for c in _WAVELENGTH_CASES])
    def test_grid_matches_exactly(self, ref, key, spec, kwargs):
        """Length, endpoints and sampled interior points, bit for bit."""
        expected = ref["wavelengths"][key]
        wls = Wavelengths(spec, **kwargs)

        assert len(wls) == expected["length"]
        assert float(wls[0]) == expected["first_cm"]
        assert float(wls[-1]) == expected["last_cm"]

        # Julia's sample indices are 1-based.
        for idx, value in zip(expected["sample_idx"], expected["sample_cm"]):
            assert float(wls[idx - 1]) == pytest.approx(value, rel=1e-15), idx

    @pytest.mark.parametrize("key,spec,kwargs", _WAVELENGTH_CASES,
                             ids=[c[0] for c in _WAVELENGTH_CASES])
    def test_range_metadata_matches(self, ref, key, spec, kwargs):
        """One entry per window, with matching lengths and endpoints."""
        expected = ref["wavelengths"][key]
        wls = Wavelengths(spec, **kwargs)

        assert len(wls.wl_ranges) == expected["n_ranges"]
        assert [n for _, _, n in wls.wl_ranges] == expected["range_lengths"]
        for (start, stop, _), ref_start, ref_stop in zip(
            wls.wl_ranges, expected["range_starts"], expected["range_stops"]
        ):
            assert float(start) == pytest.approx(ref_start, rel=1e-15)
            assert float(stop) == pytest.approx(ref_stop, rel=1e-15)

    @pytest.mark.parametrize("key,spec,kwargs", _WAVELENGTH_CASES,
                             ids=[c[0] for c in _WAVELENGTH_CASES])
    def test_frequencies_match(self, ref, key, spec, kwargs):
        """``all_freqs`` is c/λ reversed, and must agree with Korg.jl."""
        expected = ref["wavelengths"][key]
        wls = Wavelengths(spec, **kwargs)
        assert float(wls.all_freqs[0]) == pytest.approx(expected["first_freq"],
                                                        rel=1e-15)
        assert float(wls.all_freqs[-1]) == pytest.approx(expected["last_freq"],
                                                         rel=1e-15)
        assert np.all(np.diff(wls.all_freqs) > 0)

    def test_air_wavelength_ranges_match_julia(self, ref):
        """Air->vacuum conversion agrees with Korg.jl's ``wl_ranges``.

        The comparison is against ``wl_ranges`` rather than ``all_wls``
        because Korg.jl 1.2.1 builds ``all_wls`` *before* the air->vacuum
        conversion and never rebuilds it, so on the Julia side only
        ``wl_ranges`` carries vacuum values (see the note in
        ``generate_utils_reference.jl``). Korg.px converts consistently, and
        its converted endpoints reproduce Korg.jl's vacuum range exactly.
        """
        expected = ref["wavelengths"]["air"]
        wls = Wavelengths((5000, 5010, 1.0), air_wavelengths=True)

        assert len(wls) == expected["length"]
        assert float(wls.wl_ranges[0][0]) == pytest.approx(
            expected["range_starts"][0], rel=1e-15)
        assert float(wls.wl_ranges[0][1]) == pytest.approx(
            expected["range_stops"][0], rel=1e-15)
        # ...and the grid really is the vacuum one, not the air one.
        assert float(wls[0]) == pytest.approx(expected["range_starts"][0],
                                              rel=1e-15)

    def test_subspectrum_indices_match(self, ref):
        """Julia's 1-based inclusive ranges vs our 0-based half-open pairs."""
        wls = Wavelengths([(5000, 5010, 1.0), (6000, 6010, 1.0)])
        expected = ref["wavelengths_subspectrum"]["first_last"]
        got = wls.subspectrum_indices()
        assert len(got) == len(expected)
        for (lo, hi), (jl_first, jl_last) in zip(got, expected):
            assert lo == jl_first - 1
            assert hi == jl_last

    def test_eachwindow_matches(self, ref):
        wls = Wavelengths([(5000, 5010, 1.0), (6000, 6010, 1.0)])
        expected = ref["wavelengths_eachwindow"]["windows"]
        got = list(wls.eachwindow())
        assert len(got) == len(expected)
        for (lo, hi), (jl_lo, jl_hi) in zip(got, expected):
            assert float(lo) == pytest.approx(jl_lo, rel=1e-15)
            assert float(hi) == pytest.approx(jl_hi, rel=1e-15)

    def test_non_dividing_step_is_a_known_divergence(self, ref):
        """A step that does not divide the interval is handled differently.

        Korg.jl builds ``range(start; stop, step)``, which *truncates*: it keeps
        the requested step and drops the final partial point, so
        ``(5000, 5500, 0.03)`` gives 16667 points ending at 5499.98 A.  Korg.px
        rounds the point count and then uses ``linspace(start, stop, n)``, which
        keeps both endpoints and silently adjusts the step, giving 16668 points
        ending at 5500 A.

        This test documents the divergence rather than asserting agreement.
        It is not fixed here because the grid size feeds ``synthesize``,
        ``fit`` and the LSF helpers, which are owned elsewhere.  If the
        construction is ever changed to match Korg.jl, this test should be
        replaced by a normal agreement check against the reference.
        """
        expected = ref["wavelengths"]["5000_5500_0.03"]
        assert expected["length"] == 16667
        assert expected["last_cm"] == pytest.approx(5499.98e-8, rel=1e-12)

        wls = Wavelengths((5000, 5500, 0.03))
        assert len(wls) == expected["length"] + 1
        assert float(wls[-1]) == pytest.approx(5500e-8, rel=1e-15)
        # The endpoints are right; it is the interior spacing that differs.
        assert float(wls[0]) == expected["first_cm"]
        actual_step = (wls[-1] - wls[0]) / (len(wls) - 1)
        assert actual_step == pytest.approx(0.03e-8, rel=1e-4)
        assert actual_step != 0.03e-8


# ---------------------------------------------------------------------------
# Abundances
# ---------------------------------------------------------------------------


_ABUNDANCE_CASES = {
    "solar": {},
    "mh_-1": dict(default_metals_H=-1.0),
    "mh_-2.5_alpha_-2.0": dict(default_metals_H=-2.5, default_alpha_H=-2.0),
    "mh_+0.3": dict(default_metals_H=0.3),
    "fe_-0.5": dict(abundances={"Fe": -0.5}),
    "fe_Z_-0.5": dict(abundances={26: -0.5}),
    "He_absolute": dict(abundances={"He": 10.5}, solar_relative=False),
    "mixed": dict(default_metals_H=-1.0, default_alpha_H=-0.6,
                  abundances={"C": 0.3, 22: 0.1}),
    "asplund09": dict(default_metals_H=-0.5,
                      solar_abundances=ASPLUND_2009_SOLAR_ABUNDANCES),
    "asplund20": dict(default_metals_H=-0.5,
                      solar_abundances=ASPLUND_2020_SOLAR_ABUNDANCES),
    "grevesse07": dict(default_metals_H=-0.5,
                       solar_abundances=GREVESSE_2007_SOLAR_ABUNDANCES),
    "custom_alpha_elements": dict(default_metals_H=-1.0, default_alpha_H=-0.4,
                                  alpha_elements=[8, 12, 14]),
}


class TestSolarAbundanceTablesReference:
    """The four solar abundance tables are data copied verbatim from Korg.jl.

    They are compared with ``==``, matching the standard the constants tests
    were tightened to: a transcription typo in the last decimal place is a
    different table, not a rounding difference.
    """

    @pytest.mark.parametrize("name,table", [
        ("bergemann_2025", BERGEMANN_2025_SOLAR_ABUNDANCES),
        ("asplund_2020", ASPLUND_2020_SOLAR_ABUNDANCES),
        ("asplund_2009", ASPLUND_2009_SOLAR_ABUNDANCES),
        ("grevesse_2007", GREVESSE_2007_SOLAR_ABUNDANCES),
    ])
    def test_table_is_bitwise_identical(self, ref, name, table):
        expected = np.array(ref["solar_abundance_sets"][name])
        assert table.shape == (92,)
        assert np.array_equal(table, expected), (
            f"{name} differs at indices "
            f"{np.flatnonzero(table != expected).tolist()}"
        )

    def test_default_table_is_bergemann_2025(self, ref):
        assert ref["solar_abundance_sets"]["default_is"] == "bergemann_2025"
        assert np.array_equal(DEFAULT_SOLAR_ABUNDANCES,
                              BERGEMANN_2025_SOLAR_ABUNDANCES)

    def test_default_alpha_elements_match(self, ref):
        assert list(DEFAULT_ALPHA_ELEMENTS) == ref["default_alpha_elements"]


class TestFormatAXReference:
    """``format_A_X`` must reproduce Korg.jl's 92-vector exactly."""

    @pytest.mark.parametrize("key", list(_ABUNDANCE_CASES))
    def test_format_A_X_is_bitwise_identical(self, ref, key):
        expected = np.array(ref["format_A_X"][key])
        got = format_A_X(**_ABUNDANCE_CASES[key])
        assert got.shape == (92,)
        assert np.array_equal(got, expected), (
            f"{key} differs at Z = "
            f"{(np.flatnonzero(got != expected) + 1).tolist()}"
        )

    def test_metals_and_alpha_overrides_are_distinguishable(self, ref):
        """A regression guard: the alpha case must not equal the metals case."""
        metals_only = np.array(ref["format_A_X"]["mh_-1"])
        with_alpha = np.array(ref["format_A_X"]["mh_-2.5_alpha_-2.0"])
        assert not np.array_equal(metals_only, with_alpha)


class TestMetalsAndAlphaHReference:
    """``get_metals_H`` / ``get_alpha_H`` against Korg.jl, at 1e-13 relative."""

    @pytest.mark.parametrize("key", list(_ABUNDANCE_CASES))
    def test_metals_H_ignore_alpha(self, ref, key):
        A_X = np.array(ref["format_A_X"][key])
        expected = ref["metals_alpha_H"][key]["metals_H_ignore_alpha"]
        assert float(get_metals_H(A_X)) == pytest.approx(expected,
                                                         rel=1e-13, abs=1e-15)

    @pytest.mark.parametrize("key", list(_ABUNDANCE_CASES))
    def test_metals_H_all_elements(self, ref, key):
        A_X = np.array(ref["format_A_X"][key])
        expected = ref["metals_alpha_H"][key]["metals_H_all"]
        assert float(get_metals_H(A_X, ignore_alpha=False)) == pytest.approx(
            expected, rel=1e-13, abs=1e-15)

    @pytest.mark.parametrize("key", list(_ABUNDANCE_CASES))
    def test_alpha_H(self, ref, key):
        A_X = np.array(ref["format_A_X"][key])
        expected = ref["metals_alpha_H"][key]["alpha_H"]
        assert float(get_alpha_H(A_X)) == pytest.approx(expected,
                                                        rel=1e-13, abs=1e-15)

    def test_carbon_is_not_excluded_from_metals_H(self, ref):
        """``ignore_alpha=True`` drops only the alpha elements, not carbon.

        Korg.jl 1.2.1's implementation is
        ``[Z for Z in 3:MAX_ATOMIC_NUMBER if !(Z in alpha_elements)]``.  Its
        docstring claims carbon is dropped too; the docstring is wrong, and a
        port that follows it agrees with Korg.jl for every uniformly-scaled
        abundance pattern (where carbon cancels) and disagrees as soon as
        carbon is set individually.  The ``mixed`` case sets [C/H] = +0.3 and
        so is the case that separates the two.
        """
        A_X = np.array(ref["format_A_X"]["mixed"])
        expected = ref["metals_alpha_H"]["mixed"]["metals_H_ignore_alpha"]
        assert float(get_metals_H(A_X)) == pytest.approx(expected, rel=1e-13)

        no_carbon = [Z for Z in range(3, 93)
                     if Z not in DEFAULT_ALPHA_ELEMENTS and Z != 6]
        from korg.abundances import _get_multi_X_H
        carbon_excluded = float(
            _get_multi_X_H(A_X, no_carbon, BERGEMANN_2025_SOLAR_ABUNDANCES))
        assert abs(carbon_excluded - expected) > 0.1, (
            "the mixed case must actually distinguish the two definitions"
        )


# ---------------------------------------------------------------------------
# Cubic splines
# ---------------------------------------------------------------------------


class TestCubicSplineReference:
    """Spline coefficients and evaluations against Korg.jl's CubicSplines."""

    @pytest.mark.parametrize("case", ["quadratic", "wiggly", "nonuniform",
                                      "exp_like", "extrap"])
    def test_second_derivative_coefficients_match(self, ref, case):
        """``z`` comes from a tridiagonal solve; it must match to 1e-13."""
        c = ref["cubic_splines"][case]
        spline = cubic_spline(np.array(c["t"]), np.array(c["u"]),
                              extrapolate=c["extrapolate"])
        np.testing.assert_allclose(np.asarray(spline.z), np.array(c["z"]),
                                   rtol=1e-13, atol=1e-15)

    @pytest.mark.parametrize("case", ["quadratic", "wiggly", "nonuniform",
                                      "exp_like", "extrap"])
    def test_knot_spacings_match_exactly(self, ref, case):
        c = ref["cubic_splines"][case]
        spline = cubic_spline(np.array(c["t"]), np.array(c["u"]),
                              extrapolate=c["extrapolate"])
        assert np.array_equal(np.asarray(spline.h), np.array(c["h"]))

    @pytest.mark.parametrize("case", ["quadratic", "wiggly", "nonuniform",
                                      "exp_like", "extrap"])
    def test_evaluations_match(self, ref, case):
        c = ref["cubic_splines"][case]
        spline = cubic_spline(np.array(c["t"]), np.array(c["u"]),
                              extrapolate=c["extrapolate"])
        got = np.array([float(spline(float(x))) for x in c["eval_x"]])
        np.testing.assert_allclose(got, np.array(c["eval_y"]),
                                   rtol=1e-13, atol=1e-14)

    @pytest.mark.parametrize("case", ["quadratic", "wiggly", "nonuniform",
                                      "exp_like"])
    def test_numpy_eval_agrees_with_jax_eval(self, ref, case):
        """``numpy_eval`` is a second implementation; it must not drift."""
        c = ref["cubic_splines"][case]
        spline = cubic_spline(np.array(c["t"]), np.array(c["u"]),
                              extrapolate=c["extrapolate"])
        got = np.asarray(spline.numpy_eval(np.array(c["eval_x"])))
        np.testing.assert_allclose(got, np.array(c["eval_y"]),
                                   rtol=1e-13, atol=1e-14)

    def test_flat_extrapolation_matches_julia(self, ref):
        """Outside the knots Korg.jl returns u[1]/u[end] exactly."""
        c = ref["cubic_splines"]["extrap"]
        spline = cubic_spline(np.array(c["t"]), np.array(c["u"]),
                              extrapolate=True)
        for x, y in zip(c["eval_x"], c["eval_y"]):
            if x < c["t"][0]:
                assert float(spline(float(x))) == pytest.approx(c["u"][0],
                                                                rel=1e-13)
            elif x > c["t"][-1]:
                assert float(spline(float(x))) == pytest.approx(c["u"][-1],
                                                                rel=1e-13)
            assert float(spline(float(x))) == pytest.approx(y, rel=1e-13)

    def test_cumulative_integral_matches(self, ref):
        """``cumulative_integral`` against Korg.jl's ``cumulative_integral!``."""
        c = ref["cubic_spline_cumulative_integral"]
        spline = cubic_spline(np.array(c["t"]), np.array(c["u"]))
        got = np.asarray(spline.cumulative_integral(c["t1"], c["t2"]))
        expected = np.array(c["out"])
        assert got.shape == expected.shape
        np.testing.assert_allclose(got, expected, rtol=1e-13, atol=1e-14)
