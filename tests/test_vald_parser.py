"""
Tests for ``korg.vald_parser`` — the full VALD parser (short/long format,
"extract all"/"extract stellar", isotopic scaling).

Reference data comes from ``tests/linelist_reference_data.json``, produced by
``tests/generate_linelist_reference.jl`` running Korg.jl v1.2.1's
``parse_vald_linelist`` over the *same* input files (copied verbatim from
Korg.jl's test suite into ``tests/data/linelists``).

Categories covered here:

1. Functional — every format combination, isotopic scaling on/off, footer
   detection, malformed input, empty input.
2. Julia agreement — all seven line fields compared at rtol 1e-13.
3. Autodiff — the only float-in/float-out helpers in this module
   (``ten_to_the_or_missing`` / ``id_or_missing``) are differentiated.
4. jit-tracing — the parser reads files and branches on Python strings, so it
   provably cannot be traced; that is pinned rather than left ambiguous.
"""

import json
from pathlib import Path

import korg  # noqa: F401 — enables JAX x64 mode

import jax
import numpy as np
import pytest

DATA_DIR = Path(__file__).parent / "data" / "linelists"
REFERENCE_FILE = Path(__file__).parent / "linelist_reference_data.json"

from tests.test_linelist_io import _require, assert_lines_match  # noqa: E402


@pytest.fixture(scope="module")
def vald_ref():
    if not REFERENCE_FILE.exists():
        pytest.fail(
            f"{REFERENCE_FILE} is missing. Regenerate with "
            "`julia --project=. tests/generate_linelist_reference.jl`."
        )
    with open(REFERENCE_FILE) as f:
        return json.load(f)["vald"]


def _parse(name):
    from korg.vald_parser import parse_vald_linelist

    with open(name) as f:
        return parse_vald_linelist(f.read())


# ===========================================================================
# 2. Julia agreement over every stored VALD file
# ===========================================================================

VALD_KEYS = [
    "short_extract_stellar",
    "short_extract_all",
    "long_extract_stellar",
    "long_extract_all_air_wavenumber",
    "long_extract_all_noquotes",
    "linelist_vald",
    "vald_5000_5005",
    "iso_short_all_unscaled",
    "iso_short_stellar_unscaled",
    "iso_long_all_unscaled",
    "iso_long_stellar_unscaled",
    "iso_scaled",
]


class TestVALDAgainstJulia:

    @pytest.mark.parametrize("key", VALD_KEYS)
    def test_matches_julia(self, vald_ref, key):
        block = vald_ref[key]
        lines = _parse(_require(DATA_DIR / block["file"]))
        assert_lines_match(lines, block["lines"], context=f"vald {key}")

    def test_long_extract_stellar_is_not_silently_empty(self, vald_ref):
        """
        Regression test: the surplus trailing "depth" column used to make pandas
        promote the species column to an index, shifting every field by one and
        making the whole file parse to zero lines.
        """
        lines = _parse(_require(DATA_DIR / "long-extract-stellar.vald"))
        assert len(lines) == len(vald_ref["long_extract_stellar"]["lines"]) == 2

    def test_vdW_is_decoded_into_a_tuple(self):
        """
        Regression test: this parser used to build ``Line`` directly, leaving the
        raw VALD Waals column (e.g. -7.54) in the ``vdW`` field instead of
        decoding it into ``(10**-7.54, -1.0)``.
        """
        lines = _parse(_require(DATA_DIR / "short-extract-stellar.vald"))
        for l in lines:
            assert isinstance(l.vdW, tuple) and len(l.vdW) == 2
            assert l.vdW[0] >= 0.0
            assert l.vdW[1] == -1.0 or 0.0 <= l.vdW[1] <= 1.0
        assert np.isclose(lines[0].vdW[0], 10.0 ** -7.54, rtol=1e-13)

    def test_missing_stark_is_approximated_not_none(self):
        """A zero Stark column means "no data" and must be filled in by Unsöld."""
        from korg.linelist import approximate_gammas
        from korg.species import Species

        lines = _parse(_require(DATA_DIR / "vald_5000_5005_placeholder.vald")) \
            if (DATA_DIR / "vald_5000_5005_placeholder.vald").exists() \
            else _parse(_require(DATA_DIR / "5000-5005.vald"))
        zeros = [l for l in lines if l.gamma_stark is not None]
        assert len(zeros) == len(lines), "no line may carry gamma_stark=None"
        for l in lines:
            assert np.isfinite(l.gamma_stark)
        # the La II line in this file has 0.000 in every damping column
        la = [l for l in lines if str(l.species) == "La II"][0]
        expected, _ = approximate_gammas(la.wl, Species("La II"), la.E_lower)
        assert np.isclose(la.gamma_stark, float(expected), rtol=1e-13)


# ===========================================================================
# 1. Functional: isotopic scaling
# ===========================================================================

class TestIsotopicScaling:

    def test_unscaled_lists_match_the_prescaled_one(self):
        """
        Korg.jl's own test: the four "unscaled" files, once Korg applies the
        isotopic corrections from the reference strings, must reproduce the
        pre-scaled file (log gf to 1%, everything else exactly).
        """
        scaled = _parse(_require(DATA_DIR / "isotopic_scaling" / "scaled.vald"))
        for name in ["short-all-unscaled.vald", "long-all-unscaled.vald",
                     "short-stellar-unscaled.vald", "long-stellar-unscaled.vald"]:
            other = _parse(_require(DATA_DIR / "isotopic_scaling" / name))
            assert len(other) == len(scaled), name
            for a, b in zip(scaled, other):
                assert a.species == b.species, name
                assert np.isclose(a.wl, b.wl, rtol=1e-14), name
                assert np.isclose(a.E_lower, b.E_lower, rtol=1e-14, atol=1e-15), name
                assert np.isclose(a.gamma_rad, b.gamma_rad, rtol=1e-14), name
                assert np.isclose(a.gamma_stark, b.gamma_stark, rtol=1e-14,
                                  atol=1e-300), name
                assert np.isclose(a.vdW[0], b.vdW[0], rtol=1e-14, atol=1e-300), name
                assert np.isclose(a.log_gf, b.log_gf, rtol=1e-2), name

    def test_custom_isotopic_abundances_change_log_gf(self):
        from korg.isotopic_data import isotopic_abundances
        from korg.vald_parser import parse_vald_linelist

        text = (DATA_DIR / "isotopic_scaling" / "long-all-unscaled.vald").read_text()
        default = parse_vald_linelist(text)
        custom = {Z: dict(d) for Z, d in isotopic_abundances.items()}
        for Z in custom:
            for iso in custom[Z]:
                custom[Z][iso] = 1.0  # no isotopic penalty at all
        modified = parse_vald_linelist(text, isotopic_abund=custom)
        assert any(a.log_gf != b.log_gf for a, b in zip(default, modified))
        for a, b in zip(default, modified):
            assert b.log_gf >= a.log_gf - 1e-12

    def test_unknown_element_in_reference_is_ignored(self):
        """An isotope tag for an element we have no data for contributes nothing."""
        from korg.vald_parser import parse_vald_linelist

        text = (
            " 5000.00000, 5001.00000, 1, 1, 1.0 Wavelength region\n"
            "                                                  Damping parameters\n"
            "Spec Ion       WL_vac(A)  Excit(eV) Vmic log gf* Rad.   Stark   Waals\n"
            "'Fe 1', 5000.0000, 3.1124, 1.0, -0.760, 7.970,-5.260, -7.540, 0.780,"
            " 0.018, '(56)Zz (56)Fe'\n"
            "* oscillator strengths were NOT scaled by the solar isotopic ratios.\n"
        )
        lines = parse_vald_linelist(text)
        assert len(lines) == 1
        from korg.isotopic_data import isotopic_abundances
        assert np.isclose(lines[0].log_gf,
                          -0.760 + np.log10(isotopic_abundances[26][56]), rtol=1e-12)


# ===========================================================================
# 1. Functional: format detection and error paths
# ===========================================================================

class TestFormatDetectionAndErrors:

    HEADER_STELLAR = (
        " 5000.00000, 5001.00000, 1, 1, 1.0 Wavelength region\n"
        "                                                  Damping parameters\n"
        "Spec Ion       WL_vac(A)  Excit(eV) Vmic log gf* Rad.   Stark   Waals\n"
    )
    ROW = ("'Fe 1', 5000.0000, 3.1124, 1.0, -0.760, 7.970,-5.260, -7.540, 0.780,"
           " 0.018, 'ref'\n")
    SCALED = "* oscillator strengths were scaled by the solar isotopic ratios.\n"

    def test_empty_linelist_raises(self):
        from korg.vald_parser import parse_vald_linelist

        with pytest.raises(ValueError, match="Empty linelist"):
            parse_vald_linelist("")

    def test_only_comments_raises(self):
        from korg.vald_parser import parse_vald_linelist

        with pytest.raises(ValueError, match="Empty linelist"):
            parse_vald_linelist("# just a comment\n#and another\n")

    def test_missing_scaling_statement_raises(self):
        from korg.vald_parser import parse_vald_linelist

        with pytest.raises(ValueError, match="whether log\\(gf\\)s are scaled"):
            parse_vald_linelist(self.HEADER_STELLAR + self.ROW)

    def test_unknown_energy_units_raise(self):
        from korg.vald_parser import parse_vald_linelist

        header = self.HEADER_STELLAR.replace("Excit(eV)", "Excit(???)")
        with pytest.raises(ValueError, match="determine energy units"):
            parse_vald_linelist(header + self.ROW + self.SCALED)

    def test_unknown_wavelength_medium_raises(self):
        from korg.vald_parser import parse_vald_linelist

        header = self.HEADER_STELLAR.replace("WL_vac(A)", "WL_???(A)")
        with pytest.raises(ValueError, match="determine vac/air wls"):
            parse_vald_linelist(header + self.ROW + self.SCALED)

    def test_truncation_warning_line_is_ignored(self, vald_ref):
        """The ``WARNING: Output was truncated`` banner is stripped, as in Korg.jl."""
        block = vald_ref["long_extract_all_noquotes"]
        text = (DATA_DIR / block["file"]).read_text()
        assert text.startswith(" WARNING: Output was truncated to 100000 lines")
        lines = _parse(_require(DATA_DIR / block["file"]))
        assert_lines_match(lines, block["lines"], context="truncation banner")

    def test_wavenumber_energies_converted_to_eV(self, vald_ref):
        """``E_low(cm^-1)`` columns are converted with h·c."""
        from korg.constants import c_cgs, hplanck_eV

        lines = _parse(_require(DATA_DIR / "long-extract-all-air-wavenumber.vald"))
        assert np.isclose(lines[0].E_lower, 25102.875 * c_cgs * hplanck_eV, rtol=1e-13)

    def test_air_wavelengths_converted_to_vacuum(self):
        from korg.utils import air_to_vacuum

        lines = _parse(_require(DATA_DIR / "long-extract-all-air-wavenumber.vald"))
        assert np.isclose(lines[0].wl, float(air_to_vacuum(4998.72899)) * 1e-8, rtol=1e-13)

    def test_comment_lines_are_dropped(self):
        """``short-extract-all.vald`` has a commented-out duplicate data row."""
        lines = _parse(_require(DATA_DIR / "short-extract-all.vald"))
        assert len(lines) == 2  # not 3

    def test_gamma_rad_zero_is_approximated(self):
        from korg.linelist import approximate_radiative_gamma
        from korg.vald_parser import parse_vald_linelist

        row = ("'Fe 1', 5000.0000, 3.1124, 1.0, -0.760, 0.000, -5.260, -7.540,"
               " 0.780, 0.018, 'ref'\n")
        lines = parse_vald_linelist(self.HEADER_STELLAR + row + self.SCALED)
        expected = float(approximate_radiative_gamma(lines[0].wl, -0.760))
        assert np.isclose(lines[0].gamma_rad, expected, rtol=1e-13)

    def test_unparseable_species_is_skipped_with_a_warning(self, capsys):
        from korg.vald_parser import parse_vald_linelist

        bad = ("'Zz 1', 5000.0000, 3.1124, 1.0, -0.760, 7.970,-5.260, -7.540,"
               " 0.780, 0.018, 'ref'\n")
        lines = parse_vald_linelist(self.HEADER_STELLAR + bad + self.ROW + self.SCALED)
        assert len(lines) == 1
        assert "Skipping line" in capsys.readouterr().out

    def test_file_without_a_footer(self):
        """
        When the data runs to the end of the file the footer scan finds nothing
        and every body line must be kept.
        """
        from korg.vald_parser import parse_vald_linelist

        text = (
            "* oscillator strengths were scaled by the solar isotopic ratios.\n"
            "Elm Ion       WL_vac(A) Excit(eV) log gf*   Rad.  Stark    Waals factor"
            "   References\n"
            "'Fe 1', 5000.0000, 3.1124, -0.760, 7.970,-5.260, -7.540, 0.780,'ref'\n"
            "'Fe 1', 5001.0000, 3.1124, -0.860, 7.970,-5.260, -7.540, 0.780,'ref'\n"
        )
        lines = parse_vald_linelist(text)
        assert len(lines) == 2
        assert np.isclose(lines[1].log_gf, -0.860, rtol=1e-13)

    def test_missing_trailing_columns_are_padded(self):
        """
        A row with fewer fields than the header must not shift the remaining
        columns; the absent ones become NaN.
        """
        from korg.vald_parser import parse_vald_linelist

        text = (
            "* oscillator strengths were scaled by the solar isotopic ratios.\n"
            "Elm Ion       WL_vac(A) Excit(eV) log gf*   Rad.  Stark    Waals factor"
            "   References\n"
            "'Fe 1', 5000.0000, 3.1124, -0.760, 7.970,-5.260, -7.540, 0.780\n"
            "'Fe 1', 5001.0000, 3.1124, -0.860, 7.970,-5.260, -7.540, 0.780\n"
        )
        lines = parse_vald_linelist(text)
        assert len(lines) == 2
        assert np.isclose(lines[0].log_gf, -0.760, rtol=1e-13)
        assert np.isclose(lines[0].vdW[0], 10.0 ** -7.540, rtol=1e-13)

    def test_unknown_mass_number_in_reference_is_ignored(self):
        """
        ``(99)Fe`` names a real element but a mass number we have no abundance
        for, so it contributes nothing to Δlog gf.
        """
        from korg.vald_parser import parse_vald_linelist

        text = (
            self.HEADER_STELLAR
            + "'Fe 1', 5000.0000, 3.1124, 1.0, -0.760, 7.970,-5.260, -7.540, 0.780,"
              " 0.018, '(99)Fe'\n"
            + "* oscillator strengths were NOT scaled by the solar isotopic ratios.\n"
        )
        lines = parse_vald_linelist(text)
        assert len(lines) == 1
        assert np.isclose(lines[0].log_gf, -0.760, rtol=1e-13)

    def test_helper_functions(self):
        from korg.vald_parser import id_or_missing, ten_to_the_or_missing

        assert ten_to_the_or_missing(0) is None
        assert ten_to_the_or_missing(-5.0) == 10 ** -5.0
        assert id_or_missing(0) is None
        assert id_or_missing(-7.5) == -7.5


# ===========================================================================
# 1. Functional: read_vald_linelist (filtering + sorting)
# ===========================================================================

class TestReadVALDLinelist:

    def test_sorted_and_filtered(self):
        from korg.species import Species
        from korg.vald_parser import read_vald_linelist

        lines = read_vald_linelist(_require(DATA_DIR / "5000-5005.vald"))
        assert [l.wl for l in lines] == sorted(l.wl for l in lines)
        assert all(0 <= l.species.charge <= 2 for l in lines)
        assert all(l.species != Species("H I") for l in lines)

    def test_hydrogen_and_highly_ionized_removed(self, tmp_path):
        from korg.vald_parser import read_vald_linelist

        text = (
            " 5000.00000, 5001.00000, 3, 3, 1.0 Wavelength region\n"
            "                                                  Damping parameters\n"
            "Spec Ion       WL_vac(A)  Excit(eV) Vmic log gf* Rad.   Stark   Waals\n"
            "'H 1', 5000.0000, 3.1124, 1.0, -0.760, 7.970,-5.260, -7.540, 0.780,"
            " 0.018, 'ref'\n"
            "'Fe 4', 5001.0000, 3.1124, 1.0, -0.760, 7.970,-5.260, -7.540, 0.780,"
            " 0.018, 'ref'\n"
            "'Fe 1', 5002.0000, 3.1124, 1.0, -0.760, 7.970,-5.260, -7.540, 0.780,"
            " 0.018, 'ref'\n"
            "* oscillator strengths were scaled by the solar isotopic ratios.\n"
        )
        p = tmp_path / "f.vald"
        p.write_text(text)
        lines = read_vald_linelist(str(p))
        assert len(lines) == 1
        assert str(lines[0].species) == "Fe I"

    def test_custom_isotopic_abundances_forwarded(self):
        from korg.isotopic_data import isotopic_abundances
        from korg.vald_parser import read_vald_linelist

        path = _require(DATA_DIR / "isotopic_scaling" / "long-all-unscaled.vald")
        default = read_vald_linelist(path)
        custom = {Z: {iso: 1.0 for iso in d} for Z, d in isotopic_abundances.items()}
        modified = read_vald_linelist(path, isotopic_abund=custom)
        assert len(default) == len(modified)
        assert any(a.log_gf != b.log_gf for a, b in zip(default, modified))


# ===========================================================================
# 3. Autodiff
# ===========================================================================

class TestAutodiff:

    def test_ten_to_the_or_missing_gradient(self):
        from korg.vald_parser import ten_to_the_or_missing

        f = lambda x: ten_to_the_or_missing(x)
        g = float(jax.grad(f)(-5.0))
        assert np.isfinite(g) and g > 0.0
        h = 1e-6
        fd = (float(f(-5.0 + h)) - float(f(-5.0 - h))) / (2 * h)
        assert np.isclose(g, fd, rtol=1e-5)

    def test_id_or_missing_gradient_is_unity(self):
        from korg.vald_parser import id_or_missing

        assert float(jax.grad(lambda x: id_or_missing(x))(-7.5)) == 1.0

    def test_radiative_gamma_used_by_the_parser_is_differentiable(self):
        """The parser's only numeric kernel is approximate_radiative_gamma."""
        from korg.vald_parser import approximate_radiative_gamma

        g = float(jax.grad(lambda wl: approximate_radiative_gamma(wl, -0.76))(5e-5))
        assert np.isfinite(g) and g < 0.0


# ===========================================================================
# 4. jit-tracing
# ===========================================================================

class TestJIT:

    def test_ten_to_the_or_missing_jits(self):
        from korg.vald_parser import ten_to_the_or_missing

        # NB: the `x == 0` branch is a Python bool on a traced value, so only the
        # non-zero path can be compiled — jit therefore needs a concrete value.
        with pytest.raises(jax.errors.TracerBoolConversionError):
            jax.jit(ten_to_the_or_missing)(-5.0)

    def test_parse_vald_linelist_cannot_be_jitted(self):
        """
        ``parse_vald_linelist`` splits strings, dispatches on regex matches and
        builds pandas DataFrames.  None of that is a JAX operation, so tracing it
        is meaningless; pin the failure so the intent is unambiguous.
        """
        from korg.vald_parser import parse_vald_linelist

        text = (DATA_DIR / "short-extract-stellar.vald").read_text()
        with pytest.raises(Exception):
            jax.jit(lambda x: parse_vald_linelist(text))(1.0)

    def test_read_vald_linelist_cannot_be_jitted(self):
        """File I/O is not traceable."""
        from korg.vald_parser import read_vald_linelist

        path = _require(DATA_DIR / "5000-5005.vald")
        with pytest.raises(Exception):
            jax.jit(lambda x: read_vald_linelist(path))(1.0)
