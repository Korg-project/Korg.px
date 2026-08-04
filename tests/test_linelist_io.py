"""
Tests for the linelist I/O layer (``korg.linelist``).

Four categories, as required by this project:

1. **Functional** — round-trips, malformed input, option combinations and edge
   cases for every reader/writer in ``korg.linelist``.
2. **High-precision agreement with Korg.jl** — every parser is compared against
   ``tests/linelist_reference_data.json``, which is produced by
   ``tests/generate_linelist_reference.jl`` running Korg.jl v1.2.1 over the very
   same input files (they were copied verbatim out of Korg.jl's own test suite
   into ``tests/data/linelists``).  Tolerances are 1e-13 relative or tighter.
3. **Autodiff** — ``jax.grad`` of the float-in/float-out helpers is finite,
   non-zero and agrees with central finite differences.
4. **jit-tracing** — what can be traced is asserted to compile; what provably
   cannot (Python control flow on traced values, file I/O) is pinned with
   ``pytest.raises`` and an explanation.
"""

import json
import math
import os
from pathlib import Path

import korg  # noqa: F401 — enables JAX x64 mode before anything else

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

DATA_DIR = Path(__file__).parent / "data" / "linelists"
REFERENCE_FILE = Path(__file__).parent / "linelist_reference_data.json"


@pytest.fixture(scope="module")
def ref():
    """Julia reference data for the linelist I/O cluster."""
    if not REFERENCE_FILE.exists():
        pytest.fail(
            f"{REFERENCE_FILE} is missing. Regenerate it with:\n"
            '  export PATH="/mnt/sw/nix/store/'
            'yr11xz204lj9ah1irz61lbqh1dk0hcif-julia-1.11.2/bin:$PATH"\n'
            "  julia --project=. tests/generate_linelist_reference.jl"
        )
    with open(REFERENCE_FILE) as f:
        return json.load(f)


def _require(path: Path) -> str:
    """Return ``path`` as a string, failing loudly (never skipping) if absent."""
    if not path.exists():
        pytest.fail(
            f"Test input {path} is missing. It is copied from Korg.jl v1.2.1's "
            f"test/data/linelists; restore it from "
            f"~/.julia/packages/Korg/XvH45/test/data/linelists."
        )
    return str(path)


def assert_lines_match(py_lines, julia_lines, rtol=1e-13, context=""):
    """Assert a Python linelist matches the Julia reference line-for-line."""
    assert len(py_lines) == len(julia_lines), (
        f"{context}: parsed {len(py_lines)} lines, Julia parsed {len(julia_lines)}"
    )
    for i, (p, j) in enumerate(zip(py_lines, julia_lines)):
        where = f"{context} line {i} ({j['species']} @ {j['wl'] * 1e8:.4f} Å)"
        assert str(p.species) == j["species"], (
            f"{where}: species {str(p.species)!r} != {j['species']!r}"
        )
        assert p.species.charge == j["species_charge"], f"{where}: charge"
        assert np.isclose(p.wl, j["wl"], rtol=rtol, atol=0.0), f"{where}: wl"
        assert np.isclose(p.log_gf, j["log_gf"], rtol=rtol, atol=1e-15), f"{where}: log_gf"
        assert np.isclose(p.E_lower, j["E_lower"], rtol=rtol, atol=1e-15), f"{where}: E_lower"
        assert np.isclose(p.gamma_rad, j["gamma_rad"], rtol=rtol, atol=0.0), f"{where}: gamma_rad"
        assert np.isclose(p.gamma_stark, j["gamma_stark"], rtol=rtol, atol=1e-300), (
            f"{where}: gamma_stark {p.gamma_stark} != {j['gamma_stark']}"
        )
        assert np.isclose(p.vdW[0], j["vdW"][0], rtol=rtol, atol=1e-300), (
            f"{where}: vdW[0] {p.vdW[0]} != {j['vdW'][0]}"
        )
        assert np.isclose(p.vdW[1], j["vdW"][1], rtol=rtol, atol=0.0), f"{where}: vdW[1]"


# ===========================================================================
# 1. Functional + 2. Julia agreement: MOOG
# ===========================================================================

class TestMOOGParser:
    """``parse_moog_linelist`` — MOOG format, vacuum and air."""

    MOOG = DATA_DIR / "s5eqw_short.moog"

    def test_vacuum_matches_julia(self, ref):
        from korg.linelist import parse_moog_linelist

        lines = parse_moog_linelist(_require(self.MOOG), vacuum_wavelengths=True)
        assert_lines_match(lines, ref["moog"]["vacuum"], context="moog vacuum")

    def test_air_matches_julia(self, ref):
        from korg.linelist import parse_moog_linelist

        lines = parse_moog_linelist(_require(self.MOOG), vacuum_wavelengths=False)
        assert_lines_match(lines, ref["moog"]["air"], context="moog air")

    def test_air_is_air_to_vacuum_of_vacuum(self):
        """Reading the same file as air must equal air_to_vacuum of the vacuum read."""
        from korg.linelist import parse_moog_linelist
        from korg.utils import air_to_vacuum

        vac = parse_moog_linelist(_require(self.MOOG), vacuum_wavelengths=True)
        air = parse_moog_linelist(_require(self.MOOG), vacuum_wavelengths=False)
        for lv, la in zip(vac, air):
            assert np.isclose(float(air_to_vacuum(lv.wl)), la.wl, rtol=1e-14)

    def test_sorted_by_wavelength(self):
        from korg.linelist import parse_moog_linelist

        lines = parse_moog_linelist(_require(self.MOOG))
        assert [l.wl for l in lines] == sorted(l.wl for l in lines)

    def test_isotopic_scaling_applied(self):
        """The MgH / C2 / Mn I rows carry isotope codes that shift log_gf."""
        from korg.linelist import isotopic_abundances, parse_moog_linelist

        lines = parse_moog_linelist(_require(self.MOOG))
        # 5100.2490 112.00124 -> MgH with 1-H and 24-Mg
        mgh = [l for l in lines if str(l.species) in ("HMg", "MgH")][0]
        assert np.isclose(mgh.log_gf,
                          0.52 + math.log10(isotopic_abundances[12][24]), rtol=1e-13)
        # 5117.5820 606.01213 -> C2 with 12-C and 13-C
        c2 = [l for l in lines if str(l.species) == "C2"][0]
        assert np.isclose(c2.log_gf,
                          -0.082 + math.log10(isotopic_abundances[6][12])
                          + math.log10(isotopic_abundances[6][13]), rtol=1e-13)
        # 5117.8980 25.0055 -> Mn I with 55-Mn
        mn = [l for l in lines if str(l.species) == "Mn I"][0]
        assert np.isclose(mn.log_gf,
                          -3.363 + math.log10(isotopic_abundances[25][55]), rtol=1e-13)

    def test_custom_isotopic_abundances(self):
        """Passing custom abundances must change the isotope-corrected log_gf."""
        from korg.linelist import isotopic_abundances, parse_moog_linelist

        custom = {Z: dict(d) for Z, d in isotopic_abundances.items()}
        custom[25][55] = 0.5  # pretend only half of Mn is Mn-55
        lines = parse_moog_linelist(_require(self.MOOG), isotopic_abundances=custom)
        mn = [l for l in lines if str(l.species) == "Mn I"][0]
        assert np.isclose(mn.log_gf, -3.363 + math.log10(0.5), rtol=1e-13)

    def test_unknown_isotope_leaves_log_gf_unchanged(self, tmp_path):
        """An isotope that is not in the table contributes 0 to Δlog_gf."""
        from korg.linelist import parse_moog_linelist

        p = tmp_path / "iso.moog"
        p.write_text("header line\n 5000.000  26.0999  1.000  -1.000\n")
        lines = parse_moog_linelist(str(p))
        assert len(lines) == 1
        assert lines[0].log_gf == -1.0  # Fe-999 is unknown -> unchanged

    def test_file_object_accepted(self):
        """A file-like object is accepted as well as a path."""
        from korg.linelist import parse_moog_linelist

        with open(_require(self.MOOG)) as fh:
            lines = parse_moog_linelist(fh)
        assert len(lines) == 6

    def test_blank_comment_and_short_rows_skipped(self, tmp_path):
        from korg.linelist import parse_moog_linelist

        p = tmp_path / "messy.moog"
        p.write_text(
            "header\n"
            "\n"
            "# a comment\n"
            "   \n"
            " 5000.000  26.0  1.0\n"          # too few columns
            " not_a_number 26.0 1.0 -1.0\n"   # unparseable
            " 5000.000  26.0  1.0  -1.0\n"    # the only good row
        )
        lines = parse_moog_linelist(str(p))
        assert len(lines) == 1
        assert np.isclose(lines[0].wl, 5e-5, rtol=1e-14)

    def test_empty_linelist(self, tmp_path):
        """A file with only a header yields an empty list, not an error."""
        from korg.linelist import parse_moog_linelist

        p = tmp_path / "empty.moog"
        p.write_text("just a header\n")
        assert parse_moog_linelist(str(p)) == []

    def test_single_line(self, tmp_path):
        from korg.linelist import parse_moog_linelist

        p = tmp_path / "one.moog"
        p.write_text("header\n 5000.000  26.0  1.0  -1.0\n")
        lines = parse_moog_linelist(str(p))
        assert len(lines) == 1
        assert str(lines[0].species) == "Fe I"

    def test_duplicate_wavelengths_are_kept(self, tmp_path):
        from korg.linelist import parse_moog_linelist

        p = tmp_path / "dup.moog"
        p.write_text("header\n 5000.000  26.0  1.0  -1.0\n 5000.000  26.0  1.0  -2.0\n")
        lines = parse_moog_linelist(str(p))
        assert len(lines) == 2
        assert lines[0].wl == lines[1].wl

    def test_isotope_digits_not_divisible_by_atom_count_are_ignored(self, tmp_path):
        """3 isotope digits for a 2-atom molecule cannot be split, so Δlog gf = 0."""
        from korg.linelist import parse_moog_linelist

        p = tmp_path / "odd.moog"
        p.write_text("header\n 5000.000  106.0123  1.0  -1.0\n")
        lines = parse_moog_linelist(str(p))
        assert len(lines) == 1
        assert lines[0].log_gf == -1.0

    def test_non_numeric_isotope_digits_are_ignored(self, tmp_path):
        """A non-numeric isotope field must not abort the parse."""
        from korg.linelist import parse_moog_linelist

        p = tmp_path / "badiso.moog"
        p.write_text("header\n 5000.000  26.0ab  1.0  -1.0\n")
        lines = parse_moog_linelist(str(p))
        assert len(lines) == 1
        assert lines[0].log_gf == -1.0
        assert str(lines[0].species) == "Fe I"


class TestMOOGSpeciesCodes:
    """``_moog_species_code_to_species`` — atomic and molecular MOOG codes."""

    @pytest.mark.parametrize("code,expected_charge,expected_atoms", [
        ("26.0", 0, [26]),
        ("26.1", 1, [26]),
        ("22.10000", 1, [22]),
        ("106.0", 0, [1, 6]),      # CH
        ("606.0", 0, [6, 6]),      # C2
        ("112.0", 0, [1, 12]),     # MgH
        ("10108.0", 0, [1, 1, 8]),  # H2O
        # leading-zero and zero-padded variants exercise every branch of the
        # right-to-left 2-digit unpacking loop
        ("0608.0", 0, [6, 8]),      # even length, no odd digit left over
        ("10008.0", 0, [1, 8]),     # an all-zero 2-digit pair is dropped
        ("00112.0", 0, [1, 12]),    # odd length whose leftover digit is "0"
    ])
    def test_codes(self, code, expected_charge, expected_atoms):
        from korg.linelist import _moog_species_code_to_species

        spec = _moog_species_code_to_species(code)
        assert spec.charge == expected_charge
        assert [int(a) for a in spec.formula.atoms if a != 0] == expected_atoms


# ===========================================================================
# Kurucz
# ===========================================================================

KURUCZ_DIR = DATA_DIR / "kurucz"

# (reference key, filename) for every Kurucz file the Julia generator covers.
KURUCZ_FILES = [
    ("head", "gfallvac08oct17.head.dat"),
    ("head_missing_col", "gfallvac08oct17-missing-col.head.dat"),
    ("short_lines", "gfallvac08oct17-short-lines.stub.dat"),
    ("ba", "gfallvac08oct17_ba"),
    ("filtered_species", "gfallvac08oct17-filtered-species.dat"),
]


class TestKuruczParser:
    """``parse_kurucz_linelist`` and the ``kurucz``/``kurucz_vac`` dispatch."""

    @pytest.mark.parametrize("key,fname", KURUCZ_FILES)
    def test_air_matches_julia(self, ref, key, fname):
        from korg.linelist import isotopic_abundances, parse_kurucz_linelist

        lines = parse_kurucz_linelist(_require(KURUCZ_DIR / fname), isotopic_abundances)
        assert_lines_match(lines, ref["kurucz"][key]["air"], context=f"kurucz air {key}")

    @pytest.mark.parametrize("key,fname", KURUCZ_FILES)
    def test_vac_matches_julia(self, ref, key, fname):
        from korg.linelist import isotopic_abundances, parse_kurucz_linelist

        lines = parse_kurucz_linelist(_require(KURUCZ_DIR / fname), isotopic_abundances,
                                      vacuum=True)
        assert_lines_match(lines, ref["kurucz"][key]["vac"], context=f"kurucz vac {key}")

    @pytest.mark.parametrize("key,fname", KURUCZ_FILES)
    def test_kurucz_embedded_isotopic_adjustment_matches_julia(self, ref, key, fname):
        """``isotopic_abundances=None`` means "use Kurucz's own log gf adjustment"."""
        from korg.linelist import parse_kurucz_linelist

        lines = parse_kurucz_linelist(_require(KURUCZ_DIR / fname), None)
        assert_lines_match(lines, ref["kurucz"][key]["kurucz_iso"],
                           context=f"kurucz embedded iso {key}")

    @pytest.mark.parametrize("key,fname", KURUCZ_FILES)
    @pytest.mark.parametrize("fmt,refkey", [("kurucz", "read_linelist"),
                                            ("kurucz_vac", "read_linelist_vac")])
    def test_read_linelist_matches_julia(self, ref, key, fname, fmt, refkey):
        from korg.linelist import read_linelist

        lines = read_linelist(_require(KURUCZ_DIR / fname), format=fmt)
        assert_lines_match(lines, ref["kurucz"][key][refkey], context=f"{fmt} {key}")

    @pytest.mark.parametrize("key,fname", KURUCZ_FILES)
    def test_read_linelist_with_embedded_isotopes_matches_julia(self, ref, key, fname):
        from korg.linelist import read_linelist

        lines = read_linelist(_require(KURUCZ_DIR / fname), format="kurucz",
                              isotopic_abundances=None)
        assert_lines_match(lines, ref["kurucz"][key]["read_linelist_kurucz_iso"],
                           context=f"kurucz embedded iso via read_linelist {key}")

    def test_column_spec_on_a_single_record(self):
        """
        Pin the fixed-width column spec against hand-decoded values.

        The first record of gfallvac08oct17 is Be II 7232.0699 nm (air; gfall
        wavelengths are nanometres, so this is a 7.23 µm infrared line),
        log gf -0.826, levels 140020.580 and 141403.310 cm^-1, and
        log10 of γ_rad/γ_Stark/γ_vdW = 7.93/-2.41/-6.91.  These are the numbers
        Korg.jl's own test suite asserts (test/linelist.jl, "kurucz linelist
        parsing"), so a shifted slice cannot pass by accident.
        """
        from korg.constants import c_cgs, hplanck_eV
        from korg.linelist import isotopic_abundances, parse_kurucz_linelist
        from korg.species import Species
        from korg.utils import air_to_vacuum

        line = parse_kurucz_linelist(_require(KURUCZ_DIR / "gfallvac08oct17.head.dat"),
                                     isotopic_abundances)[0]
        assert line.species == Species("Be II")
        assert line.log_gf == -0.826
        assert np.isclose(line.wl, air_to_vacuum(7232.0699e-7), rtol=1e-15)
        assert np.isclose(line.wl, 0.0007234041763337705, rtol=1e-13)  # Korg.jl's value
        assert np.isclose(line.E_lower, 140020.580 * c_cgs * hplanck_eV, rtol=1e-15)
        assert np.isclose(line.E_lower, 17.360339371573698, rtol=1e-13)  # Korg.jl's value
        assert np.isclose(line.gamma_rad, 10.0 ** 7.93, rtol=1e-14)
        assert np.isclose(line.gamma_stark, 10.0 ** -2.41, rtol=1e-14)
        assert np.isclose(line.vdW[0], 10.0 ** -6.91, rtol=1e-14)
        assert line.vdW[1] == -1.0

    def test_missing_column_variant_parses_identically(self):
        """
        A 159-character record is the 160-character one with the leading column
        of the wavelength field lost.  Restoring it must reproduce the same lines.
        """
        from korg.linelist import read_linelist

        full = read_linelist(_require(KURUCZ_DIR / "gfallvac08oct17.head.dat"),
                             format="kurucz")
        short = read_linelist(_require(KURUCZ_DIR / "gfallvac08oct17-missing-col.head.dat"),
                              format="kurucz")
        assert full == short

    def test_records_with_stripped_trailing_columns_parse(self):
        """
        gfallvac08oct17-short-lines.stub.dat holds 160-, 106- and 98-character
        records of the *same* transition; padding back to 160 must make them
        identical apart from the isotope columns that were truncated away.
        """
        from korg.linelist import isotopic_abundances, parse_kurucz_linelist
        from korg.species import Species

        lines = parse_kurucz_linelist(
            _require(KURUCZ_DIR / "gfallvac08oct17-short-lines.stub.dat"),
            isotopic_abundances)
        assert len(lines) == 5
        assert all(l.species == Species("Be II") for l in lines[:4])
        assert all(l == lines[0] for l in lines[1:4])

    def test_blank_lines_are_skipped(self):
        """The missing-col file opens with blank lines; they contribute nothing."""
        from korg.linelist import isotopic_abundances, parse_kurucz_linelist

        path = _require(KURUCZ_DIR / "gfallvac08oct17-missing-col.head.dat")
        with open(path) as f:
            assert sum(1 for row in f if not row.strip()) >= 2
        assert len(parse_kurucz_linelist(path, isotopic_abundances)) == 20

    def test_read_linelist_filters_high_ionization_and_hydrogen(self):
        """
        Korg.jl's read_linelist keeps only 0 <= charge <= 2 and drops H I, which
        gfall (unlike the other formats' test files) actually exercises.
        """
        from korg.linelist import isotopic_abundances, parse_kurucz_linelist, read_linelist
        from korg.species import Species

        path = _require(KURUCZ_DIR / "gfallvac08oct17-filtered-species.dat")
        assert [str(l.species) for l in parse_kurucz_linelist(path, isotopic_abundances)] == \
            ["Be II", "S IV", "Cu V", "H I"]
        kept = read_linelist(path, format="kurucz")
        assert [l.species for l in kept] == [Species("Be II")]

    def test_sorted_by_wavelength(self):
        from korg.linelist import read_linelist

        for _, fname in KURUCZ_FILES:
            lines = read_linelist(_require(KURUCZ_DIR / fname), format="kurucz")
            assert lines == sorted(lines, key=lambda l: l.wl), fname

    def test_air_and_vac_differ_by_the_air_to_vacuum_conversion(self):
        from korg.linelist import read_linelist
        from korg.utils import air_to_vacuum

        path = _require(KURUCZ_DIR / "gfallvac08oct17.head.dat")
        air = read_linelist(path, format="kurucz")
        vac = read_linelist(path, format="kurucz_vac")
        assert all(np.isclose(a.wl, air_to_vacuum(v.wl), rtol=1e-15)
                   for a, v in zip(air, vac))
        assert all(a.wl > v.wl for a, v in zip(air, vac))

    def test_isotopic_scaling_recovers_the_unsplit_log_gf(self):
        """
        Korg.jl issue #463: the HFS/isotope components of the 6143.4 Å Ba II line
        must sum back to the log gf of the unsplit line (-0.03).  Kurucz's own
        embedded adjustments only get within ~0.02 because his column is short of
        digits, which is the whole reason the NIST table is the default.
        """
        from korg.linelist import isotopic_abundances, read_linelist

        path = _require(KURUCZ_DIR / "gfallvac08oct17_ba")
        nist = read_linelist(path, format="kurucz")
        embedded = read_linelist(path, format="kurucz", isotopic_abundances=None)
        assert math.isclose(math.log10(sum(10 ** l.log_gf for l in nist)), -0.03, abs_tol=0.01)
        assert math.isclose(math.log10(sum(10 ** l.log_gf for l in embedded)), -0.01,
                            abs_tol=0.01)

    def test_custom_isotopic_abundances_change_log_gf(self):
        from korg.linelist import isotopic_abundances, read_linelist

        path = _require(KURUCZ_DIR / "gfallvac08oct17_ba")
        custom = {Z: dict(d) for Z, d in isotopic_abundances.items()}
        custom[56][137] /= 2
        default = read_linelist(path, format="kurucz")
        halved = read_linelist(path, format="kurucz", isotopic_abundances=custom)
        # every component tagged with 137-Ba drops by log10(2); the rest are equal
        shifts = {round(d.log_gf - h.log_gf, 12) for d, h in zip(default, halved)}
        assert shifts == {0.0, round(math.log10(2), 12)}

    def test_unknown_isotope_falls_back_on_kuruczs_value(self, tmp_path, capsys):
        """
        An isotope Korg has no abundance for keeps Korg's log gf untouched (it does
        *not* silently pick up Kurucz's adjustment), and says so under verbose.
        """
        from korg.linelist import isotopic_abundances, parse_kurucz_linelist

        row = list(open(_require(KURUCZ_DIR / "gfallvac08oct17_ba")).readline().rstrip("\n"))
        row[106:109] = "199"  # no such Ba isotope
        p = tmp_path / "bad_isotope.dat"
        p.write_text("".join(row) + "\n")

        line = parse_kurucz_linelist(str(p), isotopic_abundances, verbose=True)[0]
        assert np.isclose(line.log_gf, -0.030 + -1.234, rtol=1e-13)  # log gf + HFS only
        assert "Isotope 199 not in isoabunds" in capsys.readouterr().out

    def test_molecular_linelist_is_rejected(self):
        """
        Korg.jl v1.2.1 throws for molecular Kurucz lists rather than parsing them;
        so does Korg.px, and the dispatcher must route the file there by width.
        """
        from korg.linelist import parse_kurucz_molecular_linelist, read_linelist

        path = _require(KURUCZ_DIR / "kurucz_cn.txt")
        with pytest.raises(ValueError, match="not yet supported for molecules"):
            read_linelist(path, format="kurucz")
        with pytest.raises(ValueError, match="not yet supported for molecules"):
            parse_kurucz_molecular_linelist(path)

    def test_file_object_accepted(self):
        """A file-like object works as well as a path, as for the MOOG parser."""
        from korg.linelist import isotopic_abundances, parse_kurucz_linelist

        path = _require(KURUCZ_DIR / "gfallvac08oct17.head.dat")
        with open(path) as f:
            from_object = parse_kurucz_linelist(f, isotopic_abundances)
        assert from_object == parse_kurucz_linelist(path, isotopic_abundances)

    def test_empty_file_yields_no_lines(self, tmp_path):
        from korg.linelist import isotopic_abundances, parse_kurucz_linelist

        p = tmp_path / "empty.dat"
        p.write_text("\n   \n\n")
        assert parse_kurucz_linelist(str(p), isotopic_abundances) == []


# ===========================================================================
# TurboSpectrum
# ===========================================================================

class TestTurbospectrumParser:
    """``parse_turbospectrum_linelist``."""

    TS = DATA_DIR / "Turbospectrum" / "goodlist"

    def test_air_matches_julia(self, ref):
        from korg.linelist import parse_turbospectrum_linelist

        lines = parse_turbospectrum_linelist(_require(self.TS), vacuum=False)
        assert_lines_match(lines, ref["turbospectrum"]["air"], context="turbospectrum air")

    def test_vac_matches_julia(self, ref):
        from korg.linelist import parse_turbospectrum_linelist

        lines = parse_turbospectrum_linelist(_require(self.TS), vacuum=True)
        assert_lines_match(lines, ref["turbospectrum"]["vac"], context="turbospectrum vac")

    def test_gamma_stark_is_ten_to_the_column(self, ref):
        """
        Regression test for the ``10**`` that used to be missing.

        Column 7 of a TurboSpectrum transition is log10(γ_Stark); Korg.jl runs it
        through ``tentotheOrMissing``.  The file has -1.75 in that column, so the
        stored value must be 10^-1.75, not -1.75 (a negative γ is unphysical).
        """
        from korg.linelist import parse_turbospectrum_linelist

        lines = parse_turbospectrum_linelist(_require(self.TS), vacuum=False)
        for l in lines:
            assert l.gamma_stark > 0.0, "gamma_stark must be positive"
            assert np.isclose(l.gamma_stark, 10.0 ** -1.75, rtol=1e-14)

    def test_air_vac_wavelength_relation(self):
        """Reading air vs vac differs exactly by air_to_vacuum, as in Korg.jl."""
        from korg.linelist import parse_turbospectrum_linelist
        from korg.utils import air_to_vacuum

        air = parse_turbospectrum_linelist(_require(self.TS), vacuum=False)
        vac = parse_turbospectrum_linelist(_require(self.TS), vacuum=True)
        for la, lv in zip(air, vac):
            assert np.isclose(la.wl, float(air_to_vacuum(lv.wl)), rtol=1e-14)

    def test_isotope_fudge_factor_and_abo(self, ref):
        """
        The third line uses a positive fdamp (2.5), i.e. an Unsöld fudge factor,
        while the first two use a negative fdamp (log10 γ_vdW).
        """
        from korg.linelist import parse_turbospectrum_linelist

        lines = parse_turbospectrum_linelist(_require(self.TS), vacuum=False)
        assert np.isclose(lines[0].vdW[0], 10.0 ** -6.25, rtol=1e-13)
        assert lines[0].vdW[1] == -1.0
        # the fudge-factor line has 2.5 × the Unsöld value -> strictly larger
        assert lines[2].vdW[0] > lines[0].vdW[0]

    def test_gamma_rad_zero_triggers_approximation(self, tmp_path):
        from korg.linelist import approximate_radiative_gamma, parse_turbospectrum_linelist

        p = tmp_path / "ts"
        p.write_text(
            "' 3.0000             '    1         1\n"
            "'this is Li I '\n"
            " 15982.629  4.521  -2.752     -6.25    6.0  0   -1.75\n"
        )
        lines = parse_turbospectrum_linelist(str(p), vacuum=True)
        expected = float(approximate_radiative_gamma(lines[0].wl, lines[0].log_gf))
        assert np.isclose(lines[0].gamma_rad, expected, rtol=1e-14)

    def test_gamma_rad_one_triggers_approximation(self, tmp_path):
        """Korg.jl treats gamma_rad == 1 as a placeholder too."""
        from korg.linelist import approximate_radiative_gamma, parse_turbospectrum_linelist

        p = tmp_path / "ts"
        p.write_text(
            "' 3.0000             '    1         1\n"
            "'this is Li I '\n"
            " 15982.629  4.521  -2.752     -6.25    6.0  1   -1.75\n"
        )
        lines = parse_turbospectrum_linelist(str(p), vacuum=True)
        expected = float(approximate_radiative_gamma(lines[0].wl, lines[0].log_gf))
        assert np.isclose(lines[0].gamma_rad, expected, rtol=1e-14)

    def test_gamma_stark_zero_triggers_approximation(self, tmp_path):
        """A 0 in column 7 means "no data" (tentotheOrMissing), not 10^0 = 1."""
        from korg.linelist import approximate_gammas, parse_turbospectrum_linelist
        from korg.species import Species

        p = tmp_path / "ts"
        p.write_text(
            "' 3.0000             '    1         1\n"
            "'this is Li I '\n"
            " 15982.629  4.521  -2.752     -6.25    6.0  5.25e+06  0.00\n"
        )
        lines = parse_turbospectrum_linelist(str(p), vacuum=True)
        expected, _ = approximate_gammas(lines[0].wl, Species("Li I"), 4.521)
        assert np.isclose(lines[0].gamma_stark, float(expected), rtol=1e-13)

    def test_non_numeric_stark_column_treated_as_missing(self, tmp_path):
        """
        If column 7 is an orbital-angular-momentum letter rather than a number,
        Korg.jl's ``tryparse`` returns nothing and the Stark width is approximated.
        """
        from korg.linelist import approximate_gammas, parse_turbospectrum_linelist
        from korg.species import Species

        p = tmp_path / "ts"
        p.write_text(
            "' 3.0000             '    1         1\n"
            "'this is Li I '\n"
            " 15982.629  4.521  -2.752     -6.25    6.0  5.25e+06  'p' 'x'\n"
        )
        lines = parse_turbospectrum_linelist(str(p), vacuum=True)
        expected, _ = approximate_gammas(lines[0].wl, Species("Li I"), 4.521)
        assert np.isclose(lines[0].gamma_stark, float(expected), rtol=1e-13)

    def test_six_column_transition(self, tmp_path):
        """The minimum legal transition row has 6 columns (no Stark column)."""
        from korg.linelist import parse_turbospectrum_linelist

        p = tmp_path / "ts"
        p.write_text(
            "' 3.0000             '    1         1\n"
            "'this is Li I '\n"
            " 15982.629  4.521  -2.752     -6.25    6.0  5.25e+06\n"
        )
        lines = parse_turbospectrum_linelist(str(p), vacuum=True)
        assert len(lines) == 1
        assert lines[0].gamma_stark > 0.0  # approximated

    def test_molecular_species_header(self, tmp_path):
        """A molecular species code (>99) is unpacked into a molecule."""
        from korg.linelist import parse_turbospectrum_linelist

        p = tmp_path / "ts"
        p.write_text(
            "' 108.000000         '    1         1\n"
            "'this is OH '\n"
            " 15982.629  4.521  -2.752     -6.25    6.0  5.25e+06  -1.75\n"
        )
        lines = parse_turbospectrum_linelist(str(p), vacuum=True)
        assert [int(a) for a in lines[0].species.formula.atoms if a != 0] == [1, 8]

    def test_isotopic_correction_from_header(self, tmp_path):
        """The 3-digit isotope field in the species header shifts log_gf."""
        from korg.linelist import isotopic_abundances, parse_turbospectrum_linelist

        p = tmp_path / "ts"
        p.write_text(
            "' 26.056             '    1         1\n"
            "'this is Fe I '\n"
            " 5000.000  1.0  -1.000     -6.25    6.0  5.25e+06  -1.75\n"
        )
        lines = parse_turbospectrum_linelist(str(p), vacuum=True)
        assert np.isclose(lines[0].log_gf,
                          -1.0 + math.log10(isotopic_abundances[26][56]), rtol=1e-13)

    def test_zero_isotope_field_leaves_log_gf(self, tmp_path):
        from korg.linelist import parse_turbospectrum_linelist

        p = tmp_path / "ts"
        p.write_text(
            "' 26.000             '    1         1\n"
            "'this is Fe I '\n"
            " 5000.000  1.0  -1.000     -6.25    6.0  5.25e+06  -1.75\n"
        )
        lines = parse_turbospectrum_linelist(str(p), vacuum=True)
        assert lines[0].log_gf == -1.0

    def test_header_without_species_block_yields_nothing(self, tmp_path):
        """
        Korg.jl only recognises a species block when two consecutive lines start
        with a quote.  ``requires_default_gammas`` has a single quoted line, so
        both implementations return an empty linelist.
        """
        from korg.linelist import parse_turbospectrum_linelist

        path = DATA_DIR / "Turbospectrum" / "requires_default_gammas"
        assert parse_turbospectrum_linelist(_require(path), vacuum=True) == []

    def test_malformed_species_header_skipped(self, tmp_path):
        from korg.linelist import parse_turbospectrum_linelist

        p = tmp_path / "ts"
        p.write_text("'not a species header'\n'second quoted line'\n 5000.0 1.0 -1.0 -6.25 6.0 0\n")
        assert parse_turbospectrum_linelist(str(p), vacuum=True) == []

    def test_short_transition_rows_skipped(self, tmp_path):
        from korg.linelist import parse_turbospectrum_linelist

        p = tmp_path / "ts"
        p.write_text(
            "' 3.0000             '    1         2\n"
            "'this is Li I '\n"
            " 15982.629  4.521  -2.752\n"                              # too few columns
            " 15982.629  4.521  -2.752  -6.25  6.0  5.25e+06  -1.75\n"
        )
        lines = parse_turbospectrum_linelist(str(p), vacuum=True)
        assert len(lines) == 1

    def test_unparseable_transition_row_skipped(self, tmp_path):
        from korg.linelist import parse_turbospectrum_linelist

        p = tmp_path / "ts"
        p.write_text(
            "' 3.0000             '    1         2\n"
            "'this is Li I '\n"
            " xxxx  4.521  -2.752  -6.25  6.0  5.25e+06  -1.75\n"
            " 15982.629  4.521  -2.752  -6.25  6.0  5.25e+06  -1.75\n"
        )
        lines = parse_turbospectrum_linelist(str(p), vacuum=True)
        assert len(lines) == 1

    def test_quoted_line_inside_a_block_ends_it(self, tmp_path):
        """A stray quoted line terminates the current species' transition list."""
        from korg.linelist import parse_turbospectrum_linelist

        p = tmp_path / "ts"
        p.write_text(
            "' 26.000             '    1         2\n"
            "'Fe I '\n"
            " 5000.000  1.0  -1.000  -6.25  6.0  5.25e+06  -1.75\n"
            "'a trailing comment that is not a species header'\n"
        )
        lines = parse_turbospectrum_linelist(str(p), vacuum=True)
        assert len(lines) == 1

    def test_short_isotope_field_is_ignored(self, tmp_path):
        """Fewer than 3 digits per atom means the isotope field is unusable."""
        from korg.linelist import parse_turbospectrum_linelist

        p = tmp_path / "ts"
        p.write_text(
            "' 108.00             '    1         1\n"
            "'OH '\n"
            " 5000.000  1.0  -1.000  -6.25  6.0  5.25e+06  -1.75\n"
        )
        lines = parse_turbospectrum_linelist(str(p), vacuum=True)
        assert lines[0].log_gf == -1.0

    def test_unknown_isotope_in_header_contributes_nothing(self, tmp_path):
        from korg.linelist import parse_turbospectrum_linelist

        p = tmp_path / "ts"
        p.write_text(
            "' 108.999999         '    1         1\n"
            "'OH '\n"
            " 5000.000  1.0  -1.000  -6.25  6.0  5.25e+06  -1.75\n"
        )
        lines = parse_turbospectrum_linelist(str(p), vacuum=True)
        assert lines[0].log_gf == -1.0

    def test_custom_isotopic_abundances_forwarded(self):
        from korg.linelist import isotopic_abundances, parse_turbospectrum_linelist

        custom = {Z: dict(d) for Z, d in isotopic_abundances.items()}
        lines = parse_turbospectrum_linelist(_require(self.TS), isotopic_abundances=custom,
                                             vacuum=True)
        assert len(lines) == 3

    def test_two_species_blocks_are_merged_and_sorted(self, tmp_path):
        from korg.linelist import parse_turbospectrum_linelist

        p = tmp_path / "ts"
        p.write_text(
            "' 26.000             '    1         1\n"
            "'Fe I '\n"
            " 6000.000  1.0  -1.000  -6.25  6.0  5.25e+06  -1.75\n"
            "' 20.000             '    1         1\n"
            "'Ca I '\n"
            " 5000.000  1.0  -1.000  -6.25  6.0  5.25e+06  -1.75\n"
        )
        lines = parse_turbospectrum_linelist(str(p), vacuum=True)
        assert len(lines) == 2
        assert [str(l.species) for l in lines] == ["Ca I", "Fe I"]
        assert lines[0].wl < lines[1].wl


# ===========================================================================
# Korg HDF5 linelists
# ===========================================================================

class TestKorgHDF5Linelist:
    """``save_linelist`` / ``read_korg_linelist``."""

    H5 = DATA_DIR / "korg_roundtrip.h5"

    def test_reads_julia_written_file(self, ref):
        """
        Regression test for the transposed ``formula`` dataset.

        Korg.jl writes ``reduce(hcat, formula.atoms)``, which h5py sees with
        shape (n_lines, 6).  The reader used to index it as [:, i], mixing up
        every species.
        """
        from korg.linelist import read_korg_linelist

        lines = read_korg_linelist(_require(self.H5))
        assert_lines_match(lines, ref["korg_h5_roundtrip"]["lines"], context="korg h5")

    def test_formula_dataset_layout_matches_julia(self, tmp_path):
        """A Python-written file must be byte-compatible with Korg.jl's layout."""
        from korg.linelist import create_line, save_linelist
        from korg.species import MAX_ATOMS_PER_MOLECULE

        lines = [create_line(5000.0, -1.5, "Fe I", 1.0),
                 create_line(5001.0, -1.0, "CN", 0.5)]
        out = tmp_path / "ll.h5"
        save_linelist(str(out), lines)
        with h5py.File(out, "r") as f:
            assert f["formula"].shape == (len(lines), MAX_ATOMS_PER_MOLECULE)
            # zero-padded at the *front*, like Korg.jl's Formula.atoms
            assert list(f["formula"][0]) == [0, 0, 0, 0, 0, 26]
            assert list(f["formula"][1]) == [0, 0, 0, 0, 6, 7]
            assert f.attrs["version"] == "2024-12-18"
            assert [s.decode() if isinstance(s, bytes) else s
                    for s in f["species"][:]] == ["Fe I", "CN"]

    def test_round_trip_is_exact(self, tmp_path):
        from korg.linelist import create_line, read_korg_linelist, save_linelist

        lines = [
            create_line(5000.0, -1.5, "Fe I", 1.01),
            create_line(5001.0, -0.5, "Ca II", 3.15),
            create_line(5002.0, 0.25, "CaH", 0.5),
            create_line(5003.0, -2.0, "Fe I", 2.0, gamma_rad=1e8,
                        gamma_stark=1e-5, vdW=-7.5),
            create_line(5004.0, -2.0, "Fe I", 2.0, gamma_rad=1e8,
                        gamma_stark=1e-5, vdW=234.23),
        ]
        out = tmp_path / "rt.h5"
        save_linelist(str(out), lines)
        back = read_korg_linelist(str(out))
        assert len(back) == len(lines)
        for a, b in zip(lines, back):
            assert a.wl == b.wl
            assert a.log_gf == b.log_gf
            assert a.E_lower == b.E_lower
            assert a.gamma_rad == b.gamma_rad
            assert a.gamma_stark == b.gamma_stark
            assert a.vdW == b.vdW
            assert a.species == b.species

    def test_empty_linelist_round_trip(self, tmp_path):
        from korg.linelist import read_korg_linelist, save_linelist

        out = tmp_path / "empty.h5"
        save_linelist(str(out), [])
        assert read_korg_linelist(str(out)) == []

    def test_single_line_round_trip(self, tmp_path):
        from korg.linelist import create_line, read_korg_linelist, save_linelist

        out = tmp_path / "one.h5"
        save_linelist(str(out), [create_line(5000.0, -1.5, "Fe I", 1.01)])
        assert len(read_korg_linelist(str(out))) == 1

    def test_all_zero_formula_row_is_skipped(self, tmp_path):
        """A corrupt row with no atoms is dropped rather than crashing."""
        from korg.linelist import create_line, read_korg_linelist, save_linelist

        out = tmp_path / "bad.h5"
        save_linelist(str(out), [create_line(5000.0, -1.5, "Fe I", 1.01),
                                 create_line(5001.0, -1.5, "Fe I", 1.01)])
        with h5py.File(out, "r+") as f:
            f["formula"][0] = 0
        assert len(read_korg_linelist(str(out))) == 1


# ===========================================================================
# read_linelist dispatch
# ===========================================================================

class TestReadLinelistDispatch:

    def test_default_format_from_h5_extension(self, tmp_path):
        from korg.linelist import create_line, read_linelist, save_linelist

        out = tmp_path / "ll.h5"
        save_linelist(str(out), [create_line(5000.0, -1.5, "Fe I", 1.01)])
        assert len(read_linelist(str(out))) == 1

    def test_explicit_korg_format(self):
        from korg.linelist import read_linelist

        assert len(read_linelist(_require(TestKorgHDF5Linelist.H5), format="korg")) == 5

    def test_vald_default_format(self):
        from korg.linelist import read_linelist

        lines = read_linelist(_require(DATA_DIR / "5000-5005.vald"))
        assert len(lines) > 0

    @pytest.mark.parametrize("fmt,path,expected_n", [
        ("moog", DATA_DIR / "s5eqw_short.moog", 6),
        ("moog_air", DATA_DIR / "s5eqw_short.moog", 6),
        ("turbospectrum", DATA_DIR / "Turbospectrum" / "goodlist", 3),
        ("turbospectrum_vac", DATA_DIR / "Turbospectrum" / "goodlist", 3),
    ])
    def test_text_formats(self, fmt, path, expected_n):
        from korg.linelist import read_linelist

        assert len(read_linelist(_require(path), format=fmt)) == expected_n

    def test_moog_air_differs_from_moog(self):
        from korg.linelist import read_linelist

        vac = read_linelist(_require(DATA_DIR / "s5eqw_short.moog"), format="moog")
        air = read_linelist(_require(DATA_DIR / "s5eqw_short.moog"), format="moog_air")
        assert all(a.wl > v.wl for a, v in zip(air, vac))

    def test_unknown_format_raises(self):
        from korg.linelist import read_linelist

        with pytest.raises(ValueError, match="Unknown linelist format"):
            read_linelist("whatever.txt", format="sme")

    def test_isotopic_abundances_forwarded(self):
        """``isotopic_abundances`` must reach the MOOG parser."""
        from korg.linelist import isotopic_abundances, read_linelist

        custom = {Z: dict(d) for Z, d in isotopic_abundances.items()}
        custom[25][55] = 0.25
        lines = read_linelist(_require(DATA_DIR / "s5eqw_short.moog"), format="moog",
                              isotopic_abundances=custom)
        mn = [l for l in lines if str(l.species) == "Mn I"][0]
        assert np.isclose(mn.log_gf, -3.363 + math.log10(0.25), rtol=1e-13)


# ===========================================================================
# The simplified in-module VALD reader
# ===========================================================================

class TestSimpleVALDReader:
    """``korg.linelist.read_vald_linelist`` (the reader used by the public API)."""

    def test_parses_the_bundled_solar_linelist(self):
        from korg.linelist import get_VALD_solar_linelist

        lines = get_VALD_solar_linelist()
        assert len(lines) > 1000
        assert all(l.wl > 0 for l in lines)

    def test_agrees_with_the_full_parser_on_the_solar_linelist(self):
        """
        ``korg.linelist.read_vald_linelist`` and ``korg.vald_parser.read_vald_linelist``
        are two implementations of the same thing.  On the bundled short
        "extract stellar" file they must agree on wavelength and log gf.
        """
        from korg.linelist import get_VALD_solar_linelist
        from korg.species import Species
        from korg.vald_parser import read_vald_linelist as full_reader
        from korg.data_loader import _DATA_DIR

        simple = get_VALD_solar_linelist()
        path = os.path.join(_DATA_DIR, "linelists",
                            "vald_extract_stellar_solar_threshold001.vald")
        full = full_reader(path)
        # the full reader additionally drops H I and charge > 2
        simple_kept = [l for l in simple
                       if 0 <= l.species.charge <= 2 and l.species != Species("H I")]
        assert len(simple_kept) == len(full)
        for a, b in zip(sorted(simple_kept, key=lambda l: l.wl), full):
            assert np.isclose(a.wl, b.wl, rtol=1e-12)
            assert np.isclose(a.log_gf, b.log_gf, rtol=1e-12, atol=1e-12)
            assert a.species == b.species

    def test_skips_rows_with_too_few_fields(self, tmp_path):
        from korg.linelist import read_vald_linelist

        p = tmp_path / "x.vald"
        p.write_text("'Fe 1', 5000.0, 1.0\n")
        assert read_vald_linelist(str(p)) == []

    def test_skips_rows_with_unparseable_species(self, tmp_path):
        from korg.linelist import read_vald_linelist

        p = tmp_path / "x.vald"
        p.write_text(
            "'???', 5000.0, 1.0, 1.0, -1.0, 7.0, -5.0, -7.5, 1.0, 0.1, 'ref'\n"
        )
        assert read_vald_linelist(str(p)) == []

    def test_skips_rows_with_unparseable_numbers(self, tmp_path):
        from korg.linelist import read_vald_linelist

        p = tmp_path / "x.vald"
        p.write_text(
            "'Fe 1', not_a_number, 1.0, 1.0, -1.0, 7.0, -5.0, -7.5, 1.0, 0.1, 'ref'\n"
        )
        assert read_vald_linelist(str(p)) == []

    def test_zero_vdW_column_is_approximated(self, tmp_path):
        """A 0.0 in the Waals column means "no data", so Unsöld is used."""
        from korg.linelist import approximate_gammas, read_vald_linelist
        from korg.species import Species

        p = tmp_path / "x.vald"
        p.write_text(
            "'Fe 1', 5000.0, 1.0, 1.0, -1.0, 7.0, -5.0, 0.0, 1.0, 0.1, 'ref'\n"
        )
        lines = read_vald_linelist(str(p))
        assert len(lines) == 1
        _, log_vdW = approximate_gammas(5e-5, Species("Fe I"), 1.0)
        assert np.isclose(lines[0].vdW[0], 10 ** float(log_vdW), rtol=1e-12)
        assert lines[0].vdW[1] == -1.0

    def test_blank_broadening_columns_are_approximated(self, tmp_path):
        from korg.linelist import read_vald_linelist

        p = tmp_path / "x.vald"
        p.write_text("'Fe 1', 5000.0, 1.0, 1.0, -1.0, , , , 1.0, 0.1, 'ref'\n")
        lines = read_vald_linelist(str(p))
        assert len(lines) == 1
        assert lines[0].gamma_rad > 0
        assert lines[0].gamma_stark > 0

    def test_non_data_lines_ignored(self, tmp_path):
        from korg.linelist import read_vald_linelist

        p = tmp_path / "x.vald"
        p.write_text(
            "some header\n"
            "'Fe 1', 5000.0, 1.0, 1.0, -1.0, 7.0, -5.0, -7.5, 1.0, 0.1, 'ref'\n"
            "  References:\n"
        )
        assert len(read_vald_linelist(str(p))) == 1

    def test_roman_numeral_ionization_stages(self, tmp_path):
        from korg.linelist import read_vald_linelist

        p = tmp_path / "x.vald"
        p.write_text(
            "'Fe 1', 5000.0, 1.0, 1.0, -1.0, 7.0, -5.0, -7.5, 1.0, 0.1, 'ref'\n"
            "'Ca 2', 5001.0, 1.0, 1.0, -1.0, 7.0, -5.0, -7.5, 1.0, 0.1, 'ref'\n"
            "'Ti 3', 5002.0, 1.0, 1.0, -1.0, 7.0, -5.0, -7.5, 1.0, 0.1, 'ref'\n"
        )
        lines = read_vald_linelist(str(p))
        assert [l.species.charge for l in lines] == [0, 1, 2]


# ===========================================================================
# ExoMol
# ===========================================================================

class TestExoMolLinelist:

    STATES = DATA_DIR / "ExoMol" / "40Ca-1H__XAB_abridged.states"
    TRANS = DATA_DIR / "ExoMol" / "40Ca-1H__XAB_abridged.trans"

    def test_matches_julia_6800_6810(self, ref):
        from korg.linelist import load_ExoMol_linelist

        lines = load_ExoMol_linelist("CaH", _require(self.STATES), _require(self.TRANS),
                                     6800, 6810, verbose=False)
        assert_lines_match(lines, ref["exomol"]["default_6800_6810"], context="exomol 6800-6810")

    def test_matches_julia_narrow_window(self, ref):
        from korg.linelist import load_ExoMol_linelist

        lines = load_ExoMol_linelist("CaH", _require(self.STATES), _require(self.TRANS),
                                     6800, 6804, verbose=False)
        assert_lines_match(lines, ref["exomol"]["default_6800_6804"], context="exomol 6800-6804")
        assert 0 < len(lines) < len(ref["exomol"]["default_6800_6810"])

    def test_explicit_isotopes_match_default(self, ref):
        from korg.linelist import load_ExoMol_linelist

        lines = load_ExoMol_linelist("CaH", _require(self.STATES), _require(self.TRANS),
                                     6800, 6810, isotopes=[(20, 40), (1, 1)], verbose=False)
        assert_lines_match(lines, ref["exomol"]["explicit_isotopes_6800_6810"],
                           context="exomol explicit isotopes")

    def test_deuterated_isotopes_shift_log_gf(self, ref):
        """
        With (1, 2) instead of (1, 1) the isotopic correction changes by
        log10(abundance) - log10(nuclear spin degeneracy), and most lines fall
        below the strength cutoff.
        """
        from korg.linelist import load_ExoMol_linelist

        lines = load_ExoMol_linelist("CaH", _require(self.STATES), _require(self.TRANS),
                                     6800, 6810, isotopes=[(20, 40), (1, 2)], verbose=False)
        assert_lines_match(lines, ref["exomol"]["deuterated_6800_6810"],
                           context="exomol deuterated")
        assert len(lines) < len(ref["exomol"]["default_6800_6810"])

    def test_empty_window(self, ref):
        from korg.linelist import load_ExoMol_linelist

        lines = load_ExoMol_linelist("CaH", _require(self.STATES), _require(self.TRANS),
                                     5500, 6000, verbose=False)
        assert lines == []
        assert ref["exomol"]["n_empty_5500_6000"] == 0

    def test_verbose_on_empty_window_does_not_crash(self, capsys):
        from korg.linelist import load_ExoMol_linelist

        lines = load_ExoMol_linelist("CaH", _require(self.STATES), _require(self.TRANS),
                                     5500, 6000, verbose=True)
        assert lines == []
        assert "Loading ExoMol linelist" in capsys.readouterr().out

    def test_verbose_prints_removal_summary(self, capsys):
        from korg.linelist import load_ExoMol_linelist

        load_ExoMol_linelist("CaH", _require(self.STATES), _require(self.TRANS),
                             6800, 6810, verbose=True)
        out = capsys.readouterr().out
        assert "Assuming the most abundant isotope" in out
        assert "Removed" in out

    def test_species_object_accepted(self):
        from korg.linelist import load_ExoMol_linelist
        from korg.species import Species

        lines = load_ExoMol_linelist(Species("CaH"), _require(self.STATES), _require(self.TRANS),
                                     6800, 6801, verbose=False)
        assert all(l.species == Species("CaH") for l in lines)

    def test_line_strength_cutoff_filters(self):
        from korg.linelist import load_ExoMol_linelist

        loose = load_ExoMol_linelist("CaH", _require(self.STATES), _require(self.TRANS),
                                     6800, 6810, line_strength_cutoff=-np.inf, verbose=False)
        tight = load_ExoMol_linelist("CaH", _require(self.STATES), _require(self.TRANS),
                                     6800, 6810, line_strength_cutoff=-8.0, verbose=False)
        assert len(tight) < len(loose)

    def test_sorted_by_wavelength(self):
        from korg.linelist import load_ExoMol_linelist

        lines = load_ExoMol_linelist("CaH", _require(self.STATES), _require(self.TRANS),
                                     6800, 6810, verbose=False)
        assert [l.wl for l in lines] == sorted(l.wl for l in lines)

    def test_unknown_isotope_falls_back_to_no_correction(self, tmp_path):
        """An isotope missing from the tables leaves the log gf unchanged."""
        from korg.linelist import load_ExoMol_linelist

        states = tmp_path / "x.states"
        states.write_text("1 0.0 1\n2 14700.0 3\n")
        trans = tmp_path / "x.trans"
        trans.write_text("2 1 1.0e6\n")
        good = load_ExoMol_linelist("CaH", str(states), str(trans), 6790, 6810,
                                    isotopes=[(20, 40), (1, 1)],
                                    line_strength_cutoff=-np.inf, verbose=False)
        bad = load_ExoMol_linelist("CaH", str(states), str(trans), 6790, 6810,
                                   isotopes=[(20, 999), (1, 1)],
                                   line_strength_cutoff=-np.inf, verbose=False)
        assert len(good) == len(bad) == 1
        assert bad[0].log_gf > good[0].log_gf  # no negative isotopic correction applied

    def test_malformed_rows_skipped(self, tmp_path):
        from korg.linelist import load_ExoMol_linelist

        states = tmp_path / "x.states"
        states.write_text("1 0.0\n1 0.0 1\n2 20000.0 3\n\n")
        trans = tmp_path / "x.trans"
        trans.write_text("2 1\n2 1 1.0e8\n\n")
        lines = load_ExoMol_linelist("CaH", str(states), str(trans), 4990, 5010,
                                     line_strength_cutoff=-np.inf, verbose=False)
        assert len(lines) == 1

    def test_unmapped_state_ids_skipped(self, tmp_path):
        from korg.linelist import load_ExoMol_linelist

        states = tmp_path / "x.states"
        states.write_text("1 0.0 1\n2 20000.0 3\n")
        trans = tmp_path / "x.trans"
        trans.write_text("3 1 1.0e8\n2 1 1.0e8\n2 9 1.0e8\n")
        lines = load_ExoMol_linelist("CaH", str(states), str(trans), 4990, 5010,
                                     line_strength_cutoff=-np.inf, verbose=False)
        assert len(lines) == 1

    def test_non_positive_wavenumber_skipped(self, tmp_path):
        from korg.linelist import load_ExoMol_linelist

        states = tmp_path / "x.states"
        states.write_text("1 0.0 1\n2 20000.0 3\n")
        trans = tmp_path / "x.trans"
        trans.write_text("1 2 1.0e8\n1 1 1.0e8\n2 1 1.0e8\n")
        lines = load_ExoMol_linelist("CaH", str(states), str(trans), 4990, 5010,
                                     line_strength_cutoff=-np.inf, verbose=False)
        assert len(lines) == 1

    def test_zero_einstein_A_skipped(self, tmp_path):
        from korg.linelist import load_ExoMol_linelist

        states = tmp_path / "x.states"
        states.write_text("1 0.0 1\n2 20000.0 3\n")
        trans = tmp_path / "x.trans"
        trans.write_text("2 1 0.0\n")
        lines = load_ExoMol_linelist("CaH", str(states), str(trans), 4990, 5010,
                                     line_strength_cutoff=-np.inf, verbose=False)
        assert lines == []

    def test_all_lines_removed_by_cutoff(self, tmp_path, capsys):
        from korg.linelist import load_ExoMol_linelist

        states = tmp_path / "x.states"
        states.write_text("1 0.0 1\n2 20000.0 3\n")
        trans = tmp_path / "x.trans"
        trans.write_text("2 1 1.0e-30\n")
        lines = load_ExoMol_linelist("CaH", str(states), str(trans), 4990, 5010,
                                     line_strength_cutoff=0.0, verbose=True)
        assert lines == []
        assert "Removed 1 lines" in capsys.readouterr().out


# ===========================================================================
# get_GES_linelist / get_APOGEE_DR17_linelist (synthetic data)
# ===========================================================================

def _write_ges_h5(path):
    """Write a miniature Heiter et al. 2021 artifact file."""
    species = ["Fe I", "Ca II", "CN", "CH", "CH"]
    wls_air_cm = np.array([5000.0, 5001.0, 5002.0, 5003.0, 5004.0]) * 1e-8
    with h5py.File(path, "w") as f:
        f.create_dataset("species", data=np.array(species, dtype=h5py.special_dtype(vlen=str)))
        f.create_dataset("wl", data=wls_air_cm)
        f.create_dataset("log_gf", data=np.array([-1.5, -0.5, -2.0, -3.0, -1.0]))
        f.create_dataset("E_lower", data=np.array([1.0, 3.15, 0.5, 0.4, 0.4]))
        # 0 and NaN both mean "missing"
        f.create_dataset("gamma_rad", data=np.array([8.0, 0.0, np.nan, 8.0, 8.0]))
        f.create_dataset("gamma_stark", data=np.array([-5.0, 0.0, np.nan, -5.0, -5.0]))
        f.create_dataset("vdW", data=np.array([-7.5, np.nan, np.nan, -7.5, -7.5]))


class TestGESLinelist:

    @pytest.fixture
    def patched_glob(self, tmp_path, monkeypatch):
        import glob as glob_mod

        path = tmp_path / "Heiter_et_al_2021.h5"
        _write_ges_h5(path)
        monkeypatch.setattr(glob_mod, "glob", lambda pattern: [str(path)])
        return path

    def test_includes_molecules_by_default(self, patched_glob):
        from korg.linelist import get_GES_linelist

        from korg.species import Species

        lines = get_GES_linelist()
        specs = [l.species for l in lines]
        assert Species("CN") in specs
        # the CH line with log_gf = -1.0 > -1.9 is dropped (Korg.jl issue #356)
        assert specs.count(Species("CH")) == 1

    def test_exclude_molecules(self, patched_glob):
        from korg.linelist import get_GES_linelist

        lines = get_GES_linelist(include_molecules=False)
        assert all(not l.species.formula.is_molecule() for l in lines)
        assert len(lines) == 2

    def test_air_to_vacuum_applied(self, patched_glob):
        from korg.linelist import get_GES_linelist
        from korg.utils import air_to_vacuum

        lines = get_GES_linelist(include_molecules=False)
        assert np.isclose(lines[0].wl * 1e8, float(air_to_vacuum(5000.0)), rtol=1e-12)

    def test_missing_broadening_is_approximated(self, patched_glob):
        from korg.linelist import approximate_radiative_gamma, get_GES_linelist

        lines = get_GES_linelist(include_molecules=False)
        ca = lines[1]
        assert np.isclose(ca.gamma_rad,
                          float(approximate_radiative_gamma(ca.wl, ca.log_gf)), rtol=1e-12)
        fe = lines[0]
        assert np.isclose(fe.gamma_rad, 10.0 ** 8.0, rtol=1e-12)
        assert np.isclose(fe.gamma_stark, 10.0 ** -5.0, rtol=1e-12)

    def test_missing_artifact_raises(self, monkeypatch):
        import glob as glob_mod

        from korg.linelist import get_GES_linelist

        monkeypatch.setattr(glob_mod, "glob", lambda pattern: [])
        with pytest.raises(FileNotFoundError, match="GES linelist not found"):
            get_GES_linelist()


class TestGALAHLinelistEdgeCases:
    """
    ``get_GALAH_DR3_linelist`` against a miniature synthetic file, so that the
    sentinel-value and empty-formula branches (which the real file never
    exercises) are covered.
    """

    @pytest.fixture
    def patched_data_dir(self, tmp_path, monkeypatch):
        from korg import data_loader

        d = tmp_path / "linelists" / "GALAH_DR3"
        d.mkdir(parents=True)
        with h5py.File(d / "galah_dr3_linelist.h5", "w") as f:
            # rows: Fe I, TiO (molecule), H I (filtered out), an empty formula
            f.create_dataset("wl", data=np.array([5000.0, 5001.0, 6563.0, 5002.0]))
            f.create_dataset("log_gf", data=np.array([-1.5, -2.0, 0.71, -1.0]))
            f.create_dataset("E_lo", data=np.array([1.0, 0.5, 10.2, 1.0]))
            f.create_dataset("formula", data=np.array([[26, 0, 0],
                                                       [22, 8, 0],
                                                       [1, 0, 0],
                                                       [0, 0, 0]], dtype=np.uint8))
            f.create_dataset("ionization", data=np.array([1, 1, 1, 1], dtype=np.int16))
            # -999, 0 and NaN are all "no data" sentinels
            f.create_dataset("gamma_rad", data=np.array([8.0, -999.0, 8.0, 0.0]))
            f.create_dataset("gamma_stark", data=np.array([-5.0, 0.0, -5.0, np.nan]))
            f.create_dataset("vdW", data=np.array([-7.5, -999.0, -7.5, np.nan]))
        monkeypatch.setattr(data_loader, "_DATA_DIR", str(tmp_path))
        return d

    def test_sentinels_hydrogen_and_empty_formula(self, patched_data_dir):
        from korg.linelist import approximate_radiative_gamma, get_GALAH_DR3_linelist
        from korg.species import Species

        lines = get_GALAH_DR3_linelist()
        # H I dropped, the all-zero formula row dropped
        assert len(lines) == 2
        assert [str(l.species) for l in lines] == ["Fe I", "OTi"]
        assert np.isclose(lines[0].gamma_rad, 1e8, rtol=1e-12)
        assert np.isclose(lines[0].gamma_stark, 1e-5, rtol=1e-12)
        # the molecule's -999 sentinels fall back to the approximations
        assert np.isclose(lines[1].gamma_rad,
                          float(approximate_radiative_gamma(lines[1].wl, -2.0)),
                          rtol=1e-12)
        assert lines[1].species == Species("TiO")


class TestAPOGEELinelist:

    @pytest.fixture
    def patched_data_dir(self, tmp_path, monkeypatch):
        from korg import data_loader

        d = tmp_path / "linelists" / "APOGEE_DR17"
        d.mkdir(parents=True)
        (d / "turbospec.20180901t20.atoms_no_ba").write_text(
            "' 26.000             '    1         1\n"
            "'Fe I '\n"
            " 16000.000  1.0  -1.000  -6.25  6.0  5.25e+06  -1.75\n"
        )
        (d / "turbospec.20180901t20.molec").write_text(
            "' 108.000000         '    1         1\n"
            "'OH '\n"
            " 15500.000  1.0  -2.000  -6.25  6.0  5.25e+06  -1.75\n"
        )
        with h5py.File(d / "pokazatel_water_lines.h5", "w") as f:
            f.create_dataset("wl", data=np.array([1.6e-4]))
            f.create_dataset("log_gf", data=np.array([-4.0]))
            f.create_dataset("E_lower", data=np.array([0.3]))
            f.create_dataset("gamma_rad", data=np.array([1e7]))
        monkeypatch.setattr(data_loader, "_DATA_DIR", str(tmp_path))
        return d

    def test_includes_water_by_default(self, patched_data_dir):
        from korg.linelist import get_APOGEE_DR17_linelist

        lines = get_APOGEE_DR17_linelist()
        assert len(lines) == 3
        assert "H2O" in [str(l.species) for l in lines]
        assert [l.wl for l in lines] == sorted(l.wl for l in lines)

    def test_exclude_water(self, patched_data_dir):
        from korg.linelist import get_APOGEE_DR17_linelist

        lines = get_APOGEE_DR17_linelist(include_water=False)
        assert len(lines) == 2
        assert "H2O" not in [str(l.species) for l in lines]

    def test_missing_water_file_tolerated(self, patched_data_dir):
        from korg.linelist import get_APOGEE_DR17_linelist

        (patched_data_dir / "pokazatel_water_lines.h5").unlink()
        assert len(get_APOGEE_DR17_linelist(include_water=True)) == 2

    def test_missing_data_dir_raises(self, tmp_path, monkeypatch):
        from korg import data_loader
        from korg.linelist import get_APOGEE_DR17_linelist

        monkeypatch.setattr(data_loader, "_DATA_DIR", str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError, match="APOGEE DR17 linelist data not found"):
            get_APOGEE_DR17_linelist()


# ===========================================================================
# create_line / approximate_gammas edge cases (Julia reference)
# ===========================================================================

class TestApproximateGammasEdgeCases:

    def test_all_edge_cases_match_julia(self, ref):
        from korg.linelist import approximate_gammas
        from korg.species import Species

        block = ref["approximate_gammas_edge_cases"]
        for (wl, species_str, E_lower), (j_stark, j_vdW) in zip(block["inputs"],
                                                                block["outputs"]):
            stark, log_vdW = approximate_gammas(wl, Species(species_str), E_lower)
            assert np.isclose(float(stark), j_stark, rtol=1e-13, atol=1e-300), (
                f"gamma_stark for {species_str} at wl={wl}, E={E_lower}"
            )
            assert np.isclose(float(log_vdW), j_vdW, rtol=1e-13, atol=1e-300), (
                f"log_gamma_vdW for {species_str} at wl={wl}, E={E_lower}"
            )

    def test_autoionizing_line_gives_zero_vdW(self, ref):
        """E_upper > χ -> log_gamma_vdW is set to 0 (interpreted as γ, not log γ)."""
        from korg.linelist import approximate_gammas
        from korg.species import Species

        stark, log_vdW = approximate_gammas(5e-5, Species("Fe I"), 7.5)
        assert log_vdW == 0.0
        assert float(stark) > 0.0

    def test_highly_ionized_returns_zeros(self):
        from korg.linelist import approximate_gammas
        from korg.species import Species

        assert approximate_gammas(5e-5, Species("Fe IV"), 1.0) == (0.0, 0.0)

    def test_custom_ionization_energies_used(self):
        from korg.data_loader import ionization_energies
        from korg.linelist import approximate_gammas
        from korg.species import Species

        custom = {Z: list(v) for Z, v in ionization_energies.items()}
        custom[26] = [20.0, custom[26][1], custom[26][2]]
        default_stark, _ = approximate_gammas(5e-5, Species("Fe I"), 1.0)
        custom_stark, _ = approximate_gammas(5e-5, Species("Fe I"), 1.0,
                                             ionization_energies_dict=custom)
        assert float(custom_stark) < float(default_stark)

    def test_empty_formula_raises(self):
        """The ``no atoms in formula`` guard is reachable only via a corrupt Formula."""
        from korg.linelist import approximate_gammas
        from korg.species import Formula, Species

        formula = Formula([26])
        object.__setattr__(formula, "atoms", tuple(np.uint8([0] * 6)))
        spec = Species(formula, charge=0)
        with pytest.raises(ValueError, match="no atoms in formula"):
            approximate_gammas(5e-5, spec, 1.0)


class TestCreateLineVdWDecoding:

    def test_all_scalar_vdW_branches_match_julia(self, ref):
        from korg.linelist import create_line

        block = ref["line_vdW_decoding"]
        for vdW_in, (j0, j1) in zip(block["inputs"], block["outputs"]):
            line = create_line(5000.0, -1.5, "Fe I", 1.01, vdW=vdW_in)
            assert np.isclose(line.vdW[0], j0, rtol=1e-13, atol=1e-300), f"vdW={vdW_in}"
            assert np.isclose(line.vdW[1], j1, rtol=1e-13, atol=1e-300), f"vdW={vdW_in}"

    def test_tuple_vdW_stored_verbatim(self, ref):
        from korg.linelist import create_line

        expected = ref["line_vdW_decoding"]["tuple_input"]
        line = create_line(5000.0, -1.5, "Fe I", 1.01, gamma_stark=1e-5, vdW=(1e-14, 0.3))
        assert line.vdW == (expected[0], expected[1])

    def test_list_vdW_normalised_to_tuple(self):
        from korg.linelist import create_line

        line = create_line(5000.0, -1.5, "Fe I", 1.01, gamma_stark=1e-5, vdW=[1e-14, 0.3])
        assert line.vdW == (1e-14, 0.3)
        assert isinstance(line.vdW, tuple)

    def test_wavelength_in_cm_is_not_rescaled(self):
        from korg.linelist import create_line

        assert create_line(5e-5, -1.5, "Fe I", 1.01).wl == 5e-5

    def test_species_object_accepted(self):
        from korg.linelist import create_line
        from korg.species import Species

        assert create_line(5000.0, -1.5, Species("Fe I"), 1.01).species == Species("Fe I")

    def test_custom_ionization_energies_forwarded(self):
        """``ionization_energies_dict`` must reach approximate_gammas."""
        from korg.data_loader import ionization_energies
        from korg.linelist import create_line

        custom = {Z: list(v) for Z, v in ionization_energies.items()}
        custom[26] = [20.0, custom[26][1], custom[26][2]]
        default = create_line(5000.0, -1.5, "Fe I", 1.01)
        modified = create_line(5000.0, -1.5, "Fe I", 1.01,
                               ionization_energies_dict=custom)
        assert modified.gamma_stark < default.gamma_stark
        assert modified.vdW[0] != default.vdW[0]

    def test_nan_broadening_is_approximated(self):
        from korg.linelist import create_line

        line = create_line(5000.0, -1.5, "Fe I", 1.01,
                           gamma_rad=float("nan"), gamma_stark=float("nan"),
                           vdW=float("nan"))
        assert np.isfinite(line.gamma_rad) and line.gamma_rad > 0
        assert np.isfinite(line.gamma_stark) and line.gamma_stark > 0
        assert line.vdW[1] == -1.0

    def test_repr_is_readable(self):
        from korg.linelist import create_line

        text = repr(create_line(5000.0, -1.5, "Fe I", 1.01))
        assert "Fe I" in text and "5000.000000" in text and "log gf" in text


# ===========================================================================
# approximate_line_strength (Julia reference)
# ===========================================================================

class TestApproximateLineStrength:

    def test_matches_julia(self, ref):
        from korg.linelist import approximate_line_strength, create_line

        block = ref["approximate_line_strength"]
        for (wl, log_gf, species_str, E_lower, T), expected in zip(block["inputs"],
                                                                   block["outputs"]):
            line = create_line(wl, log_gf, species_str, E_lower)
            got = approximate_line_strength(line, T)
            assert np.isclose(got, expected, rtol=1e-13), (
                f"{species_str} at {wl} Å, T={T}: {got} != {expected}"
            )

    def test_increases_with_log_gf(self):
        from korg.linelist import approximate_line_strength, create_line

        weak = create_line(5000.0, -3.0, "Fe I", 1.0)
        strong = create_line(5000.0, 0.0, "Fe I", 1.0)
        assert (approximate_line_strength(strong, 5000.0)
                > approximate_line_strength(weak, 5000.0))

    def test_decreases_with_excitation_potential(self):
        from korg.linelist import approximate_line_strength, create_line

        low = create_line(5000.0, -1.0, "Fe I", 0.0)
        high = create_line(5000.0, -1.0, "Fe I", 4.0)
        assert (approximate_line_strength(high, 5000.0)
                < approximate_line_strength(low, 5000.0))


# ===========================================================================
# 3. Autodiff
# ===========================================================================

def _central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


class TestAutodiff:

    def test_radiative_gamma_grad_wrt_wavelength(self):
        from korg.linelist import approximate_radiative_gamma

        f = lambda wl: approximate_radiative_gamma(wl, -1.5)
        g = float(jax.grad(f)(5e-5))
        assert np.isfinite(g)
        assert g < 0.0, "gamma_rad ∝ λ^-2, so dγ/dλ must be negative"
        fd = float(_central_difference(f, 5e-5, 5e-5 * 1e-5))
        assert np.isclose(g, fd, rtol=1e-5)

    def test_radiative_gamma_grad_wrt_log_gf(self):
        from korg.linelist import approximate_radiative_gamma

        f = lambda log_gf: approximate_radiative_gamma(5e-5, log_gf)
        g = float(jax.grad(f)(-1.5))
        assert np.isfinite(g) and g > 0.0
        fd = float(_central_difference(f, -1.5, 1e-5))
        assert np.isclose(g, fd, rtol=1e-5)
        # analytic: dγ/dlog_gf = γ ln(10)
        assert np.isclose(g, float(f(-1.5)) * math.log(10.0), rtol=1e-12)

    def test_approximate_gammas_stark_grad(self):
        from korg.linelist import approximate_gammas
        from korg.species import Species

        sp = Species("Fe I")
        f = lambda wl: approximate_gammas(wl, sp, 1.01)[0]
        g = float(jax.grad(f)(5e-5))
        assert np.isfinite(g) and g != 0.0
        fd = float(_central_difference(f, 5e-5, 5e-5 * 1e-5))
        assert np.isclose(g, fd, rtol=1e-5)

    def test_approximate_gammas_log_vdW_grad(self):
        from korg.linelist import approximate_gammas
        from korg.species import Species

        sp = Species("Fe I")
        f = lambda E: approximate_gammas(5e-5, sp, E)[1]
        g = float(jax.grad(f)(1.01))
        assert np.isfinite(g) and g != 0.0
        fd = float(_central_difference(f, 1.01, 1e-6))
        assert np.isclose(g, fd, rtol=1e-5)

    def test_approximate_gammas_grad_wrt_wavelength_for_log_vdW(self):
        from korg.linelist import approximate_gammas
        from korg.species import Species

        sp = Species("Ca II")
        f = lambda wl: approximate_gammas(wl, sp, 3.15)[1]
        g = float(jax.grad(f)(5e-5))
        assert np.isfinite(g) and g != 0.0
        fd = float(_central_difference(f, 5e-5, 5e-5 * 1e-5))
        assert np.isclose(g, fd, rtol=1e-5)

    def test_approximate_line_strength_grad_wrt_temperature(self):
        from korg.linelist import approximate_line_strength, create_line

        line = create_line(5000.0, -1.5, "Fe I", 1.01)
        f = lambda T: approximate_line_strength(line, T)
        g = float(jax.grad(f)(3500.0))
        assert np.isfinite(g)
        assert g > 0.0, "raising T reduces the Boltzmann penalty, so strength grows"
        fd = float(_central_difference(f, 3500.0, 3500.0 * 1e-5))
        assert np.isclose(g, fd, rtol=1e-5)

    def test_gradients_are_finite_over_a_wavelength_sweep(self):
        from korg.linelist import approximate_radiative_gamma

        grads = jax.vmap(jax.grad(lambda wl: approximate_radiative_gamma(wl, 0.0)))(
            jnp.linspace(3e-5, 9e-5, 25)
        )
        assert np.all(np.isfinite(np.asarray(grads)))
        assert np.all(np.asarray(grads) < 0.0)

    def test_no_nan_gradient_from_molecular_branch(self):
        """
        The molecule branch returns Python 0.0 constants; differentiating through
        it must give an exact zero cotangent, not NaN.
        """
        from korg.linelist import approximate_gammas
        from korg.species import Species

        sp = Species("CN")
        g = float(jax.grad(lambda wl: jnp.asarray(approximate_gammas(wl, sp, 0.5)[0]))(5e-5))
        assert g == 0.0


# ===========================================================================
# 4. jit-tracing
# ===========================================================================

class TestJIT:

    def test_approximate_radiative_gamma_jits(self):
        from korg.linelist import approximate_radiative_gamma

        jitted = jax.jit(approximate_radiative_gamma)
        assert np.isclose(float(jitted(5e-5, -1.5)),
                          float(approximate_radiative_gamma(5e-5, -1.5)), rtol=1e-14)

    def test_approximate_line_strength_jits_in_temperature(self):
        from korg.linelist import approximate_line_strength, create_line

        line = create_line(5000.0, -1.5, "Fe I", 1.01)
        jitted = jax.jit(lambda T: approximate_line_strength(line, T))
        assert np.isclose(float(jitted(3500.0)),
                          approximate_line_strength(line, 3500.0), rtol=1e-14)

    def test_approximate_gammas_cannot_be_jitted(self):
        """
        ``approximate_gammas`` branches on ``chi < E_upper`` (autoionizing test),
        a Python ``if`` on a value derived from the traced wavelength.  This is
        not traceable and there is no jnp.where formulation that preserves the
        "0.0 means γ, not log γ" convention, so pin the failure.
        """
        from korg.linelist import approximate_gammas
        from korg.species import Species

        sp = Species("Fe I")
        with pytest.raises(jax.errors.TracerBoolConversionError):
            jax.jit(lambda wl, E: approximate_gammas(wl, sp, E))(5e-5, 1.01)

    def test_create_line_cannot_be_jitted(self):
        """
        ``create_line`` branches on the traced wavelength (``if wl >= 1``) and on
        the vdW value, and finally calls ``float()`` to build a frozen dataclass.
        None of that is traceable — Line objects are host-side data structures.
        """
        from korg.linelist import create_line

        with pytest.raises(jax.errors.TracerBoolConversionError):
            jax.jit(lambda wl: create_line(wl, -1.5, "Fe I", 1.01))(5000.0)

    def test_parsers_cannot_be_jitted(self):
        """File I/O and Python string handling are not traceable by design."""
        from korg.linelist import parse_moog_linelist

        with pytest.raises(Exception):
            jax.jit(lambda x: parse_moog_linelist(_require(DATA_DIR / "s5eqw_short.moog")))(1.0)

    def test_vmap_over_radiative_gamma(self):
        from korg.linelist import approximate_radiative_gamma

        wls = jnp.linspace(3e-5, 9e-5, 7)
        out = jax.jit(jax.vmap(lambda wl: approximate_radiative_gamma(wl, -1.0)))(wls)
        assert out.shape == (7,)
        assert np.all(np.asarray(out) > 0)


# ===========================================================================
# Duplicate-implementation guards
# ===========================================================================

class TestNoDuplicateImplementations:

    def test_air_to_vacuum_is_the_single_utils_implementation(self):
        """``korg.linelist`` must re-export ``korg.utils.air_to_vacuum``, not shadow it."""
        import korg
        import korg.linelist as linelist_mod
        import korg.utils as utils_mod
        import korg.wavelengths as wavelengths_mod

        assert linelist_mod.air_to_vacuum is utils_mod.air_to_vacuum
        assert linelist_mod.vacuum_to_air is utils_mod.vacuum_to_air
        assert wavelengths_mod.air_to_vacuum is utils_mod.air_to_vacuum
        assert korg.air_to_vacuum is utils_mod.air_to_vacuum

    def test_air_to_vacuum_uses_full_precision_constants(self):
        """
        Recompute Birch & Downs (1994) with the full-precision constants Korg.jl
        uses.  A truncated-constant copy of this function (one existed in this
        codebase) disagrees well above 1e-12 relative.
        """
        from korg.utils import air_to_vacuum

        lam = 5000.0
        s = 1e4 / lam
        n = (1 + 0.00008336624212083
             + 0.02408926869968 / (130.1065924522 - s ** 2)
             + 0.0001599740894897 / (38.92568793293 - s ** 2))
        assert np.isclose(float(air_to_vacuum(lam)), lam * n, rtol=1e-15)
        # the correction is ~0.28 Å at 5000 Å, so a truncated copy is detectable
        assert 5001.39 < float(air_to_vacuum(lam)) < 5001.40

    def test_vald_parser_uses_the_shared_air_to_vacuum(self):
        import korg.utils as utils_mod
        import korg.vald_parser as vald_mod

        assert vald_mod.air_to_vacuum is utils_mod.air_to_vacuum
