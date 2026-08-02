"""
Functional behaviour and error paths for Korg.px's utility modules.

Covers ``korg.species``, ``korg.wavelengths``, ``korg.cubic_splines``,
``korg.abundances``, ``korg.artifacts``, ``korg.simple_ionization`` and the
``korg`` package ``__init__`` fallback.

Deliberate choices:

* No network access.  ``urllib.request.urlretrieve`` is replaced with a local
  copy for every artifact test; a test that reached S3 would be a flaky test.
* No silent skips.  Fixtures raise when their inputs are missing, so a broken
  checkout fails loudly instead of shrinking the suite.
"""

import hashlib
import os
import subprocess
import sys
import tarfile
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest

from korg import artifacts as artifacts_module
from korg.abundances import (
    BERGEMANN_2025_SOLAR_ABUNDANCES,
    format_A_X,
    get_alpha_H,
    get_metals_H,
    get_solar_abundances,
    A_X_to_absolute,
)
from korg.cubic_splines import cubic_spline
from korg.species import (
    MAX_ATOMS_PER_MOLECULE,
    Formula,
    Species,
    all_atomic_species,
)
from korg.wavelengths import Wavelengths


# ===========================================================================
# Formula
# ===========================================================================


class TestFormulaConstruction:
    """Every input type ``Formula`` accepts, and every rejection."""

    def test_from_atomic_number(self):
        assert str(Formula(26)) == "Fe"
        assert Formula(26).n_atoms() == 1
        assert Formula(26).is_molecule() is False

    def test_from_list_and_tuple(self):
        assert str(Formula([1, 8])) == "HO"
        assert str(Formula((8, 1))) == "HO", "input order must not matter"
        assert str(Formula([1, 1, 8])) == "H2O"

    def test_copy_constructor(self):
        original = Formula("H2O")
        copy = Formula(original)
        assert copy == original
        assert copy.atoms == original.atoms
        assert hash(copy) == hash(original)

    def test_from_string_classmethod(self):
        assert Formula.from_string("FeH") == Formula("FeH")

    def test_six_atom_molecule_is_the_limit(self):
        """MAX_ATOMS_PER_MOLECULE atoms is allowed; one more is not."""
        six = Formula([1] * MAX_ATOMS_PER_MOLECULE)
        assert six.n_atoms() == MAX_ATOMS_PER_MOLECULE
        assert six.is_molecule() is True
        with pytest.raises(ValueError, match="Up to 6 atoms"):
            Formula([1] * (MAX_ATOMS_PER_MOLECULE + 1))

    @pytest.mark.parametrize("bad", [0, -1, 93, 1000])
    def test_atomic_number_out_of_range(self, bad):
        with pytest.raises(ValueError, match="between 1 and 92"):
            Formula(bad)

    def test_empty_inputs_rejected(self):
        with pytest.raises(ValueError, match="empty Formula"):
            Formula([])
        with pytest.raises(ValueError, match="empty Formula"):
            Formula(())

    def test_list_with_out_of_range_atom_rejected(self):
        with pytest.raises(ValueError, match="atomic numbers must be"):
            Formula([1, 200])

    def test_wrong_type_rejected(self):
        with pytest.raises(TypeError, match="Invalid input type"):
            Formula(3.5)
        with pytest.raises(TypeError, match="Invalid input type"):
            Formula(None)

    def test_unknown_element_symbol_rejected(self):
        with pytest.raises(ValueError, match="Unknown element symbol"):
            Formula("Xx")

    def test_unparseable_formula_rejected(self):
        """A code with no uppercase letter yields no atoms at all."""
        with pytest.raises(ValueError, match="Could not parse formula"):
            Formula("fe")

    def test_odd_length_numeric_code_is_zero_padded(self):
        """A 5-digit code is padded to 6 before being split into pairs."""
        assert str(Formula("10608")) == "HCO"
        assert list(Formula("10608").get_atoms()) == [1, 6, 8]

    def test_too_many_nuclei_in_numeric_code_rejected(self):
        with pytest.raises(ValueError, match="up to 6 nuclei"):
            Formula("0101010101010101")

    def test_leading_junk_characters_are_skipped(self):
        """Non-uppercase, non-digit leading characters are stepped over."""
        assert Formula("-Fe") == Formula("Fe")


class TestFormulaAccessors:
    def test_get_atoms_excludes_padding(self):
        assert list(Formula("H2O").get_atoms()) == [1, 1, 8]
        assert list(Formula("Fe").get_atoms()) == [26]

    def test_get_atom_on_atom(self):
        atom = Formula("Fe").get_atom()
        assert atom == 26
        assert isinstance(atom, int), (
            "must be a Python int: a np.uint8 wraps around under NEP 50"
        )
        assert atom * 100 == 2600

    def test_get_atom_on_molecule_raises(self):
        with pytest.raises(ValueError, match="Can't get the atomic number"):
            Formula("H2O").get_atom()

    def test_get_mass_is_the_sum_of_atomic_masses(self):
        from korg.atomic_data import atomic_masses
        assert Formula("H2O").get_mass() == pytest.approx(
            2 * atomic_masses[0] + atomic_masses[7], rel=1e-15)

    def test_repr_round_trips(self):
        for code in ("Fe", "H2O", "CO", "C2"):
            assert eval(repr(Formula(code))) == Formula(code)  # noqa: S307

    def test_all_zero_formula_yields_no_atoms(self):
        """The all-padding state is unreachable via the constructor.

        ``get_atoms`` has a defensive branch for it; construct the state
        directly so the branch is actually executed rather than merely
        believed to be correct.
        """
        degenerate = Formula.__new__(Formula)
        object.__setattr__(degenerate, "atoms", tuple(np.uint8([0] * 6)))
        assert list(degenerate.get_atoms()) == []
        assert str(degenerate) == ""


class TestFormulaOrdering:
    """``Formula`` is used as a sort key and a dict key."""

    def test_equality_against_other_types(self):
        assert Formula("Fe") != "Fe"
        assert Formula("Fe") != 26
        assert (Formula("Fe") == object()) is False

    def test_total_ordering(self):
        h, fe = Formula("H"), Formula("Fe")
        assert h < fe and h <= fe
        assert fe > h and fe >= h
        assert h <= Formula("H") and h >= Formula("H")

    def test_comparison_with_other_type_is_not_implemented(self):
        for op in ("__lt__", "__le__", "__gt__", "__ge__"):
            assert getattr(Formula("H"), op)("H") is NotImplemented

    def test_sorting_a_mixed_list(self):
        """Ordering is lexicographic on the zero-padded atom vector.

        That puts every single atom before every molecule, because a molecule
        has a non-zero entry earlier in the vector.
        """
        formulas = [Formula("Fe"), Formula("H"), Formula("H2O"), Formula("He")]
        assert [str(f) for f in sorted(formulas)] == ["H", "He", "Fe", "H2O"]

    def test_usable_as_dict_key(self):
        d = {Formula("Fe"): 1, Formula("H"): 2}
        assert d[Formula(26)] == 1


# ===========================================================================
# Species
# ===========================================================================


class TestSpeciesConstruction:
    def test_from_formula_and_charge(self):
        species = Species(Formula(26), 1)
        assert str(species) == "Fe II"
        assert species.charge == 1

    def test_from_atomic_number_and_charge(self):
        assert str(Species(26, 2)) == "Fe III"

    def test_default_charge_is_neutral(self):
        assert Species(Formula(26)).charge == 0

    def test_copy_constructor(self):
        original = Species("Fe II")
        copy = Species(original)
        assert copy == original
        assert hash(copy) == hash(original)

    def test_from_string_classmethod(self):
        assert Species.from_string("Ca II") == Species("Ca II")

    def test_explicit_charge_overrides_parsed_charge(self):
        assert Species("Fe II", charge=0) == Species("Fe I")

    @pytest.mark.parametrize("ctor", [
        lambda: Species(Formula(26), -2),
        lambda: Species("Fe", charge=-2),
        lambda: Species(26, -5),
    ])
    def test_charge_below_minus_one_rejected(self, ctor):
        """Korg.jl validates in the inner constructor, i.e. on every path.

        The string path used to skip the check, so ``Species("Fe", charge=-3)``
        silently produced a species with three extra electrons.
        """
        with pytest.raises(ValueError, match="charge < -1"):
            ctor()

    def test_charge_minus_one_is_allowed(self):
        assert Species(Formula(1), -1).charge == -1
        assert str(Species(Formula(1), -1)) == "H-"

    @pytest.mark.parametrize("code", ["Fe I II", "H.1.2", "1.2.3"])
    def test_too_many_tokens_rejected(self, code):
        with pytest.raises(ValueError, match="isn't a valid species code"):
            Species(code)

    @pytest.mark.parametrize("code", ["0", "  ", "000", ".", "._"])
    def test_degenerate_codes_rejected(self, code):
        """Codes that strip down to nothing must raise, not IndexError."""
        with pytest.raises(ValueError, match="isn't a valid species code"):
            Species(code)

    def test_high_charge_uses_numeric_tag_in_string(self):
        """Beyond X (charge 9) Korg.jl prints the number, not a numeral."""
        assert str(Species(Formula(26), 12)) == "Fe 12"
        assert str(Species(Formula(26), 9)) == "Fe X"


class TestSpeciesAccessors:
    def test_delegates_to_formula(self):
        species = Species("H2O")
        assert list(species.get_atoms()) == [1, 1, 8]
        assert species.n_atoms() == 3
        assert species.is_molecule() is True
        assert species.get_mass() == Formula("H2O").get_mass()

    def test_get_atom(self):
        assert Species("Fe II").get_atom() == 26

    def test_repr_round_trips(self):
        for code in ("Fe I", "H2O", "H-", "OH+"):
            assert eval(repr(Species(code))) == Species(code)  # noqa: S307

    def test_neutral_molecule_has_no_charge_tag(self):
        assert str(Species("CO")) == "CO"
        assert str(Species("H2O")) == "H2O"

    def test_molecular_cation_uses_plus(self):
        assert str(Species(Formula("OH"), 1)) == "HO+"


class TestSpeciesOrdering:
    def test_equality_against_other_types(self):
        assert Species("Fe I") != "Fe I"
        assert (Species("Fe I") == Formula("Fe")) is False

    def test_charge_breaks_ties(self):
        assert Species("Fe I") < Species("Fe II")
        assert Species("Fe II") > Species("Fe I")
        assert Species("Fe I") <= Species("Fe I")
        assert Species("Fe I") >= Species("Fe I")

    def test_comparison_with_other_type_is_not_implemented(self):
        for op in ("__lt__", "__le__", "__gt__", "__ge__"):
            assert getattr(Species("H I"), op)("H I") is NotImplemented

    def test_usable_as_dict_key(self):
        d = {Species("Fe I"): "a", Species("Fe II"): "b"}
        assert d[Species(Formula(26), 0)] == "a"
        assert d[Species(Formula(26), 1)] == "b"

    def test_all_atomic_species_shape(self):
        species = list(all_atomic_species())
        assert len(species) == 275
        assert Species("H I") in species
        assert Species("H II") in species
        # H cannot be doubly ionised
        assert Species(Formula(1), 2) not in species
        assert Species(Formula(2), 2) in species


# ===========================================================================
# Wavelengths
# ===========================================================================


class TestWavelengthsConstruction:
    def test_two_element_tuple_uses_default_step(self):
        wls = Wavelengths((5000, 5001))
        assert len(wls) == 101, "the default step is 0.01 A"

    def test_cm_inputs_use_a_cm_default_step(self):
        """Values < 1 are cm, and then the default step is 1e-10 cm."""
        wls = Wavelengths((5.000e-5, 5.001e-5))
        assert len(wls) == 101
        assert float(wls[0]) == pytest.approx(5.0e-5, rel=1e-15)

    def test_cm_inputs_with_explicit_step(self):
        # 5.0e-5 cm to 5.01e-5 cm is 1e-7 cm, i.e. 100 steps of 1e-9 cm
        wls = Wavelengths((5.0e-5, 5.01e-5, 1e-9))
        assert len(wls) == 101
        assert float(wls[-1]) == pytest.approx(5.01e-5, rel=1e-12)

    def test_list_of_tuples(self):
        wls = Wavelengths([(5000, 5010, 1.0), (6000, 6010, 1.0)])
        assert len(wls) == 22
        assert len(wls.wl_ranges) == 2

    def test_from_explicit_array(self):
        grid = np.linspace(5000.0, 5010.0, 11)
        wls = Wavelengths(grid)
        np.testing.assert_allclose(wls.wavelengths_angstrom, grid, rtol=1e-12)

    def test_from_plain_list_of_values(self):
        wls = Wavelengths([5000.0, 5001.0, 5002.0])
        assert len(wls) == 3

    def test_single_value_array(self):
        wls = Wavelengths(np.array([5000.0]))
        assert len(wls) == 1
        assert float(wls[0]) == pytest.approx(5.0e-5, rel=1e-15)

    def test_copy_constructor(self):
        original = Wavelengths([(5000, 5010, 1.0), (6000, 6010, 1.0)])
        copy = Wavelengths(original)
        assert copy == original
        assert copy is not original
        np.testing.assert_array_equal(copy.all_wls, original.all_wls)
        np.testing.assert_array_equal(copy.all_freqs, original.all_freqs)
        # a real copy, not an alias
        copy.all_wls[0] = 0.0
        assert original.all_wls[0] != 0.0

    def test_copy_constructor_with_air_conversion(self):
        """``Wavelengths(wls, air_wavelengths=True)`` re-derives from the raw ranges."""
        air = Wavelengths((5000, 5010, 1.0))
        vacuum = Wavelengths(air, air_wavelengths=True)
        assert len(vacuum) == len(air)
        assert float(vacuum[0]) > float(air[0]), "vacuum wavelengths are longer"
        assert float(vacuum[0]) == pytest.approx(5.00139484863807e-05, rel=1e-13)

    def test_empty_array_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            Wavelengths(np.array([]))

    def test_empty_list_of_ranges_rejected(self):
        with pytest.raises(ValueError):
            Wavelengths([])

    def test_non_linearly_spaced_array_rejected(self):
        with pytest.raises(ValueError, match="not linearly spaced"):
            Wavelengths(np.array([5000.0, 5001.0, 5003.0]))

    @pytest.mark.parametrize("bad", [(5000,), (5000, 5010, 1.0, 2.0)])
    def test_wrong_tuple_length_rejected(self, bad):
        with pytest.raises(ValueError, match="must be specified as"):
            Wavelengths([bad])

    def test_overlapping_windows_rejected(self):
        with pytest.raises(ValueError, match="sorted and non-overlapping"):
            Wavelengths([(5000, 5100, 1.0), (5050, 5150, 1.0)])

    def test_descending_windows_rejected(self):
        with pytest.raises(ValueError, match="sorted and non-overlapping"):
            Wavelengths([(6000, 6010, 1.0), (5000, 5010, 1.0)])


class TestWavelengthsAirConversion:
    def test_air_conversion_shifts_the_grid(self):
        air = Wavelengths((5000, 5010, 1.0))
        vac = Wavelengths((5000, 5010, 1.0), air_wavelengths=True)
        assert np.all(vac.all_wls > air.all_wls)

    def test_wide_range_exceeds_the_linearity_threshold(self):
        """A linear air range cannot be a linear vacuum range over 7000 A."""
        with pytest.raises(ValueError, match="can't be approximated exactly"):
            Wavelengths((3000, 10000, 1.0), air_wavelengths=True)

    def test_threshold_can_be_relaxed(self):
        """The error is a threshold, not a hard limit."""
        wls = Wavelengths((3000, 10000, 1.0), air_wavelengths=True,
                          wavelength_conversion_warn_threshold=1.0)
        assert len(wls) == 7001


class TestWavelengthsInterface:
    @pytest.fixture
    def wls(self):
        return Wavelengths([(5000, 5010, 1.0), (6000, 6010, 1.0)])

    def test_len_getitem_iter(self, wls):
        assert len(wls) == 22
        assert float(wls[0]) == pytest.approx(5.0e-5, rel=1e-15)
        assert len(list(iter(wls))) == 22
        np.testing.assert_array_equal(wls[0:3], wls.all_wls[0:3])

    def test_repr_lists_the_windows(self, wls):
        assert repr(wls) == "Wavelengths(5000-5010 Å, 6000-6010 Å)"

    def test_equality(self, wls):
        assert wls == Wavelengths([(5000, 5010, 1.0), (6000, 6010, 1.0)])
        assert wls != Wavelengths((5000, 5010, 1.0))
        assert (wls == "not a Wavelengths") is False

    def test_eachwindow(self, wls):
        windows = list(wls.eachwindow())
        assert len(windows) == 2
        assert windows[0][0] == pytest.approx(5.0e-5, rel=1e-15)
        assert windows[1][1] == pytest.approx(6.01e-5, rel=1e-15)

    def test_eachfreq_is_all_freqs(self, wls):
        assert wls.eachfreq() is wls.all_freqs
        assert np.all(np.diff(wls.eachfreq()) > 0)

    def test_subspectrum_indices_tile_the_grid(self, wls):
        indices = wls.subspectrum_indices()
        assert indices == [(0, 11), (11, 22)]
        assert indices[-1][1] == len(wls)

    def test_searchsorted_accepts_angstroms(self, wls):
        """A value strictly between two grid points brackets them.

        Deliberately off-grid: ``lam * 1e-8`` and the ``linspace`` grid value
        for the same wavelength can differ in the last ulp, so an exactly
        on-grid query is a coin toss between the two neighbouring indices.
        """
        first = wls.searchsortedfirst(5005.5)
        last = wls.searchsortedlast(5005.5)
        assert last == first - 1
        assert float(wls[last]) < 5005.5e-8 < float(wls[first])
        assert float(wls[first]) == pytest.approx(5006e-8, rel=1e-12)

    def test_searchsorted_accepts_cm(self, wls):
        """Values < 1 are taken as cm and used unconverted."""
        assert wls.searchsortedfirst(5005.5e-8) == wls.searchsortedfirst(5005.5)
        assert wls.searchsortedlast(5005.5e-8) == wls.searchsortedlast(5005.5)

    def test_searchsorted_spans_the_window_gap(self, wls):
        """A wavelength inside the gap between two windows."""
        assert wls.searchsortedfirst(5500.0) == 11
        assert wls.searchsortedlast(5500.0) == 10

    def test_searchsorted_out_of_range(self, wls):
        assert wls.searchsortedfirst(9000.0) == len(wls)
        assert wls.searchsortedlast(1000.0) == -1

    def test_wavelength_unit_properties(self, wls):
        np.testing.assert_allclose(wls.wavelengths_angstrom,
                                   wls.all_wls * 1e8, rtol=0)
        assert wls.wavelengths_cm is wls.all_wls


# ===========================================================================
# Cubic splines
# ===========================================================================


class TestCubicSplineFunctional:
    T = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    U = np.array([0.0, 1.0, 4.0, 9.0, 16.0])

    def test_passes_through_every_knot(self):
        spline = cubic_spline(self.T, self.U)
        for t, u in zip(self.T, self.U):
            assert float(spline(float(t))) == pytest.approx(float(u), rel=1e-12)

    def test_out_of_bounds_raises_when_not_extrapolating(self):
        spline = cubic_spline(self.T, self.U, extrapolate=False)
        with pytest.raises(ValueError, match="Out-of-bounds value"):
            spline(-1.0)
        with pytest.raises(ValueError, match="Out-of-bounds value"):
            spline(5.0)

    def test_out_of_bounds_message_reports_the_domain(self):
        spline = cubic_spline(self.T, self.U, extrapolate=False)
        with pytest.raises(ValueError, match=r"between 0\.0 and 4\.0"):
            spline(99.0)

    def test_array_evaluation_checks_every_element(self):
        """One out-of-bounds element in an array is enough to raise."""
        spline = cubic_spline(self.T, self.U, extrapolate=False)
        with pytest.raises(ValueError, match="Out-of-bounds value"):
            spline(np.array([1.0, 2.0, 99.0]))

    def test_flat_extrapolation_returns_the_edge_values(self):
        spline = cubic_spline(self.T, self.U, extrapolate=True)
        assert float(spline(-10.0)) == pytest.approx(float(self.U[0]), abs=1e-12)
        assert float(spline(99.0)) == pytest.approx(float(self.U[-1]), rel=1e-12)

    def test_numpy_eval_matches_jax_eval(self):
        spline = cubic_spline(self.T, self.U, extrapolate=True)
        xs = np.linspace(-1.0, 5.0, 25)
        np.testing.assert_allclose(
            np.asarray(spline.numpy_eval(xs)),
            np.array([float(spline(float(x))) for x in xs]),
            rtol=1e-13, atol=1e-14,
        )

    def test_numpy_eval_without_extrapolation_skips_the_clip(self):
        """``numpy_eval`` has no bounds check; the non-clipping branch runs."""
        spline = cubic_spline(self.T, self.U, extrapolate=False)
        inside = np.array([0.5, 1.5, 3.5])
        np.testing.assert_allclose(
            np.asarray(spline.numpy_eval(inside)),
            np.array([float(spline(float(x))) for x in inside]),
            rtol=1e-13,
        )

    def test_cumulative_integral_is_monotonic_for_positive_data(self):
        spline = cubic_spline(self.T, self.U)
        out = np.asarray(spline.cumulative_integral(0.0, 4.0))
        assert out[0] == 0.0
        assert np.all(np.diff(out) > 0)
        # x^2 integrated over [0, 4] is 64/3
        assert out[-1] == pytest.approx(64.0 / 3.0, rel=2e-2)

    def test_cumulative_integral_over_a_sub_interval(self):
        spline = cubic_spline(self.T, self.U)
        out = np.asarray(spline.cumulative_integral(0.5, 2.5))
        assert out[0] == 0.0
        assert out[-1] == pytest.approx(2.5**3 / 3 - 0.5**3 / 3, rel=2e-2)

    def test_cumulative_integral_stopping_exactly_on_a_knot(self):
        """``t2`` on a knot steps the upper interval index back by one."""
        spline = cubic_spline(self.T, self.U)
        out = np.asarray(spline.cumulative_integral(0.0, 3.0))
        assert out[-1] == pytest.approx(9.0, rel=3e-2)
        assert np.all(np.isfinite(out))

    def test_cumulative_integral_within_a_single_interval(self):
        spline = cubic_spline(self.T, self.U)
        out = np.asarray(spline.cumulative_integral(1.2, 1.8))
        assert np.all(np.isfinite(out))
        assert out[-1] > 0

    def test_two_knot_spline_is_a_straight_line(self):
        """The smallest spline the tridiagonal solve can handle."""
        spline = cubic_spline(np.array([0.0, 1.0]), np.array([0.0, 2.0]))
        assert float(spline(0.5)) == pytest.approx(1.0, rel=1e-12)


# ===========================================================================
# Abundances
# ===========================================================================


class TestFormatAXFunctional:
    def test_hydrogen_is_always_twelve(self):
        for kwargs in ({}, dict(default_metals_H=-3.0),
                       dict(abundances={"Fe": 2.0})):
            assert format_A_X(**kwargs)[0] == 12.0

    def test_helium_tracks_the_solar_value_not_the_metallicity(self):
        assert format_A_X(default_metals_H=-2.0)[1] == format_A_X()[1]

    def test_absolute_abundances(self):
        A_X = format_A_X(abundances={"Fe": 7.0}, solar_relative=False)
        assert A_X[25] == 7.0

    def test_absolute_abundances_by_atomic_number(self):
        A_X = format_A_X(abundances={26: 7.0}, solar_relative=False)
        assert A_X[25] == 7.0

    def test_numpy_integer_keys_are_accepted(self):
        """Julia's check is ``el isa Integer``; np.int64 must not be rejected."""
        A_X = format_A_X(abundances={np.int64(26): -0.5})
        assert A_X[25] == pytest.approx(
            BERGEMANN_2025_SOLAR_ABUNDANCES[25] - 0.5, rel=1e-15)

    def test_explicit_solar_hydrogen_is_allowed(self):
        assert format_A_X(abundances={1: 0.0})[0] == 12.0
        assert format_A_X(abundances={"H": 12.0}, solar_relative=False)[0] == 12.0

    def test_unknown_symbol_rejected(self):
        with pytest.raises(ValueError, match="isn't a valid atomic symbol"):
            format_A_X(abundances={"Xx": 0.0})

    def test_duplicate_element_rejected(self):
        with pytest.raises(ValueError, match="both atomic number and atomic symbol"):
            format_A_X(abundances={26: -0.5, "Fe": -0.5})

    @pytest.mark.parametrize("Z", [0, -1, 93, 200])
    def test_out_of_range_atomic_number_rejected(self, Z):
        with pytest.raises(ValueError, match="not a supported atomic number"):
            format_A_X(abundances={Z: 0.0})

    def test_non_element_key_rejected(self):
        with pytest.raises(ValueError, match="isn't a valid element"):
            format_A_X(abundances={1.5: 0.0})

    def test_setting_hydrogen_to_a_silly_value_rejected(self):
        with pytest.raises(ValueError, match=r"\[H/H\] set"):
            format_A_X(abundances={1: -1.0})
        with pytest.raises(ValueError, match=r"A\(H\) set"):
            format_A_X(abundances={1: 11.0}, solar_relative=False)

    def test_custom_alpha_elements(self):
        A_X = format_A_X(default_metals_H=-1.0, default_alpha_H=0.0,
                         alpha_elements=[26])
        assert A_X[25] == pytest.approx(BERGEMANN_2025_SOLAR_ABUNDANCES[25],
                                        rel=1e-15)
        assert A_X[7] == pytest.approx(BERGEMANN_2025_SOLAR_ABUNDANCES[7] - 1.0,
                                       rel=1e-15)

    def test_custom_solar_abundances(self):
        table = np.full(92, 5.0)
        table[0] = 12.0
        A_X = format_A_X(solar_abundances=table)
        assert A_X[25] == 5.0


class TestMetalsAlphaAndAbsolute:
    def test_round_trip_metals(self):
        for mh in (-2.0, -0.5, 0.0, 0.4):
            assert get_metals_H(format_A_X(default_metals_H=mh)) == pytest.approx(
                mh, abs=1e-12)

    def test_round_trip_alpha(self):
        A_X = format_A_X(default_metals_H=-1.0, default_alpha_H=-0.4)
        assert get_alpha_H(A_X) == pytest.approx(-0.4, abs=1e-12)
        assert get_metals_H(A_X) == pytest.approx(-1.0, abs=1e-12)

    def test_ignore_alpha_changes_the_answer(self):
        A_X = format_A_X(default_metals_H=-1.0, default_alpha_H=0.0)
        assert get_metals_H(A_X, ignore_alpha=True) == pytest.approx(-1.0, abs=1e-12)
        assert get_metals_H(A_X, ignore_alpha=False) > -1.0

    def test_custom_alpha_elements_are_honoured(self):
        A_X = format_A_X(default_metals_H=-1.0, default_alpha_H=-0.2,
                         alpha_elements=[8, 12])
        assert get_alpha_H(A_X, alpha_elements=[8, 12]) == pytest.approx(
            -0.2, abs=1e-12)
        # ...and get_metals_H must use the same definition when told to
        assert get_metals_H(A_X, alpha_elements=[8, 12]) == pytest.approx(
            -1.0, abs=1e-12)

    def test_custom_solar_abundances_are_honoured(self):
        from korg.abundances import ASPLUND_2009_SOLAR_ABUNDANCES as A09
        A_X = format_A_X(default_metals_H=-0.5, solar_abundances=A09)
        assert get_metals_H(A_X, solar_abundances=A09) == pytest.approx(
            -0.5, abs=1e-12)
        assert get_alpha_H(A_X, solar_abundances=A09) == pytest.approx(
            -0.5, abs=1e-12)

    def test_every_keyword_supplied_at_once(self):
        """Both defaults overridden on both functions, in one call each."""
        from korg.abundances import GREVESSE_2007_SOLAR_ABUNDANCES as G07
        alphas = [8, 12, 14]
        A_X = format_A_X(default_metals_H=-0.8, default_alpha_H=-0.3,
                         solar_abundances=G07, alpha_elements=alphas)
        assert get_metals_H(A_X, solar_abundances=G07, alpha_elements=alphas,
                            ignore_alpha=True) == pytest.approx(-0.8, abs=1e-12)
        assert get_alpha_H(A_X, solar_abundances=G07,
                           alpha_elements=alphas) == pytest.approx(-0.3,
                                                                   abs=1e-12)

    def test_A_X_to_absolute_is_normalised(self):
        fractions = A_X_to_absolute(format_A_X())
        assert fractions.sum() == pytest.approx(1.0, rel=1e-14)
        assert fractions[0] > 0.9, "hydrogen dominates by number"
        assert np.all(fractions > 0)


class TestGetSolarAbundances:
    @pytest.mark.parametrize("source", ["bergemann_2025", "asplund_2020",
                                        "asplund_2009", "grevesse_2007"])
    def test_each_source(self, source):
        A_X = get_solar_abundances(source)
        assert A_X.shape == (92,)
        assert A_X[0] == 12.0

    def test_case_insensitive(self):
        np.testing.assert_array_equal(get_solar_abundances("ASPLUND_2009"),
                                      get_solar_abundances("asplund_2009"))

    def test_returns_a_copy(self):
        """Mutating the result must not corrupt the module-level table."""
        A_X = get_solar_abundances("asplund_2009")
        A_X[25] = -99.0
        assert get_solar_abundances("asplund_2009")[25] != -99.0

    def test_unknown_source_rejected(self):
        with pytest.raises(ValueError, match="Unknown source"):
            get_solar_abundances("kurucz_1970")


# ===========================================================================
# Artifacts
# ===========================================================================


FAKE_ARTIFACT = "unit_test_artifact"


def _make_tarball(path: Path, arcnames_and_sizes):
    """Build a gzipped tarball with the given members."""
    staging = path.parent / "staging"
    staging.mkdir(exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        for arcname, size in arcnames_and_sizes:
            member = staging / Path(arcname).name
            member.write_bytes(b"x" * size)
            tar.add(member, arcname=arcname)
    return path


@pytest.fixture
def artifact_env(tmp_path, monkeypatch):
    """An isolated ~/.korg, a registered fake artifact and a mocked network.

    The mocked ``urlretrieve`` copies a locally built tarball; nothing in this
    module ever opens a socket. It raises if the test forgot to provide a
    tarball, so a test that silently downloads nothing cannot pass.
    """
    data_dir = tmp_path / "korg_data"
    monkeypatch.setenv("KORG_DATA_DIR", str(data_dir))
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    state = {"tarball": None, "calls": [], "reporthook_calls": []}

    def fake_urlretrieve(url, filename, reporthook=None):
        state["calls"].append(url)
        source = state["tarball"]
        if source is None:
            raise AssertionError(
                "the test did not stage a tarball; a real download was attempted"
            )
        data = Path(source).read_bytes()
        Path(filename).write_bytes(data)
        if reporthook is not None:
            reporthook(0, 4096, len(data))
            reporthook(1, 4096, len(data))
            # a totalsize of 0 is what servers without Content-Length report
            reporthook(1, 4096, 0)
            state["reporthook_calls"].append(len(data))
        return filename, None

    monkeypatch.setattr(artifacts_module.urllib.request, "urlretrieve",
                        fake_urlretrieve)

    entry = {
        "url": "https://example.invalid/unit_test_artifact.tar.gz",
        "sha256": "0" * 64,
        "git_tree_sha1": "0123456789abcdef0123456789abcdef01234567",
        "extract_dir": "unit_test_dir",
        "files": ["payload.h5"],
    }
    monkeypatch.setitem(artifacts_module.ARTIFACTS, FAKE_ARTIFACT, entry)

    def stage(members=(("unit_test_dir/payload.h5", 4096),)):
        tarball = tmp_path / "artifact.tar.gz"
        _make_tarball(tarball, members)
        entry["sha256"] = hashlib.sha256(tarball.read_bytes()).hexdigest()
        state["tarball"] = tarball
        return tarball

    state["stage"] = stage
    state["entry"] = entry
    state["data_dir"] = data_dir
    return state


class TestKorgDataDir:
    def test_env_var_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KORG_DATA_DIR", str(tmp_path))
        assert artifacts_module.get_korg_data_dir() == tmp_path

    def test_default_is_dot_korg_in_home(self, monkeypatch):
        monkeypatch.delenv("KORG_DATA_DIR", raising=False)
        assert artifacts_module.get_korg_data_dir() == Path.home() / ".korg"

    def test_empty_env_var_falls_back_to_home(self, monkeypatch):
        monkeypatch.setenv("KORG_DATA_DIR", "")
        assert artifacts_module.get_korg_data_dir() == Path.home() / ".korg"


class TestArtifactHelpers:
    def test_compute_sha256_matches_hashlib(self, tmp_path):
        # larger than the 4096-byte read chunk, so the loop really loops
        payload = os.urandom(20000)
        target = tmp_path / "blob.bin"
        target.write_bytes(payload)
        assert (artifacts_module.compute_sha256(target)
                == hashlib.sha256(payload).hexdigest())

    def test_is_placeholder_file(self, tmp_path):
        missing = tmp_path / "missing"
        small = tmp_path / "small"
        big = tmp_path / "big"
        small.write_bytes(b"")
        big.write_bytes(b"y" * 2048)
        assert artifacts_module.is_placeholder_file(missing) is False
        assert artifacts_module.is_placeholder_file(small) is True
        assert artifacts_module.is_placeholder_file(big) is False


class TestDownloadArtifact:
    def test_unknown_artifact_rejected(self, artifact_env):
        with pytest.raises(ValueError, match="Unknown artifact"):
            artifacts_module.download_artifact("no_such_artifact")

    def test_successful_download_and_extract(self, artifact_env, capsys):
        artifact_env["stage"]()
        path = artifacts_module.download_artifact(FAKE_ARTIFACT)
        payload = path / "unit_test_dir" / "payload.h5"
        assert payload.exists()
        assert payload.stat().st_size == 4096
        assert artifact_env["calls"] == [artifact_env["entry"]["url"]]
        # the tarball is cleaned up
        assert not (artifact_env["data_dir"] /
                    f"{FAKE_ARTIFACT}.tar.gz").exists()
        out = capsys.readouterr().out
        assert "Downloading" in out and "Hash verified" in out

    def test_quiet_download(self, artifact_env, capsys):
        artifact_env["stage"]()
        artifacts_module.download_artifact(FAKE_ARTIFACT, show_progress=False)
        assert capsys.readouterr().out == ""

    def test_second_call_short_circuits(self, artifact_env):
        artifact_env["stage"]()
        artifacts_module.download_artifact(FAKE_ARTIFACT)
        artifacts_module.download_artifact(FAKE_ARTIFACT)
        assert len(artifact_env["calls"]) == 1, "the second call re-downloaded"

    def test_force_redownloads(self, artifact_env):
        artifact_env["stage"]()
        artifacts_module.download_artifact(FAKE_ARTIFACT)
        artifacts_module.download_artifact(FAKE_ARTIFACT, force=True)
        assert len(artifact_env["calls"]) == 2

    def test_hash_mismatch_is_fatal_and_cleans_up(self, artifact_env):
        artifact_env["stage"]()
        artifact_env["entry"]["sha256"] = "f" * 64
        with pytest.raises(RuntimeError, match="SHA256 hash mismatch"):
            artifacts_module.download_artifact(FAKE_ARTIFACT)
        assert not (artifact_env["data_dir"] /
                    f"{FAKE_ARTIFACT}.tar.gz").exists()

    def test_path_traversal_in_tarball_is_refused(self, artifact_env):
        artifact_env["stage"]([("../escaped.h5", 4096)])
        with pytest.raises(RuntimeError, match="Unsafe path in tarball"):
            artifacts_module.download_artifact(FAKE_ARTIFACT)
        assert not (artifact_env["data_dir"].parent / "escaped.h5").exists()

    def test_incomplete_extraction_is_reported(self, artifact_env):
        artifact_env["stage"]([("unit_test_dir/something_else.h5", 4096)])
        with pytest.raises(RuntimeError, match="Extraction incomplete"):
            artifacts_module.download_artifact(FAKE_ARTIFACT)

    def test_network_failure_is_wrapped(self, artifact_env, monkeypatch):
        def boom(url, filename, reporthook=None):
            raise OSError("connection reset")
        monkeypatch.setattr(artifacts_module.urllib.request, "urlretrieve", boom)
        with pytest.raises(RuntimeError, match="Failed to download"):
            artifacts_module.download_artifact(FAKE_ARTIFACT,
                                               show_progress=False)

    def test_placeholder_in_ci_short_circuits_with_a_warning(self, artifact_env,
                                                             monkeypatch):
        artifacts_module.create_placeholder_artifact(FAKE_ARTIFACT)
        monkeypatch.setenv("CI", "true")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            path = artifacts_module.download_artifact(FAKE_ARTIFACT,
                                                      show_progress=False)
        assert artifact_env["calls"] == [], "CI placeholder still downloaded"
        assert path.name == artifact_env["entry"]["git_tree_sha1"]
        assert any("placeholder" in str(w.message).lower() for w in caught)

    def test_placeholder_outside_ci_is_accepted_as_valid(self, artifact_env):
        """Known wart: outside CI a 0-byte placeholder passes the "is it
        already there?" check.

        ``download_artifact`` only treats a placeholder specially when ``CI``
        or ``GITHUB_ACTIONS`` is set. Outside CI it falls through to the
        "do all the expected files exist?" test, which a 0-byte file passes,
        so the artifact is reported as installed and never repaired. Pass
        ``force=True`` to actually replace it. This test pins that behaviour
        so a change to it is a deliberate one.
        """
        artifacts_module.create_placeholder_artifact(FAKE_ARTIFACT)
        artifact_env["stage"]()
        path = artifacts_module.download_artifact(FAKE_ARTIFACT,
                                                  show_progress=False)
        assert artifact_env["calls"] == []
        payload = path / "unit_test_dir" / "payload.h5"
        assert payload.stat().st_size == 0, "still the placeholder"

        # force=True is the escape hatch
        artifacts_module.download_artifact(FAKE_ARTIFACT, force=True,
                                           show_progress=False)
        assert len(artifact_env["calls"]) == 1
        assert payload.stat().st_size == 4096


class TestGetArtifactPath:
    def test_unknown_artifact_rejected(self, artifact_env):
        with pytest.raises(ValueError, match="Unknown artifact"):
            artifacts_module.get_artifact_path("no_such_artifact")

    def test_missing_without_auto_download_returns_none(self, artifact_env):
        assert artifacts_module.get_artifact_path(FAKE_ARTIFACT,
                                                  auto_download=False) is None
        assert artifact_env["calls"] == []

    def test_missing_with_auto_download_downloads(self, artifact_env):
        artifact_env["stage"]()
        path = artifacts_module.get_artifact_path(FAKE_ARTIFACT)
        assert (path / "unit_test_dir" / "payload.h5").exists()
        assert len(artifact_env["calls"]) == 1

    def test_present_artifact_is_returned_without_download(self, artifact_env):
        artifact_env["stage"]()
        artifacts_module.download_artifact(FAKE_ARTIFACT, show_progress=False)
        artifact_env["calls"].clear()
        path = artifacts_module.get_artifact_path(FAKE_ARTIFACT)
        assert path.exists()
        assert artifact_env["calls"] == []

    def test_placeholder_in_ci_returns_the_path_with_a_warning(self,
                                                              artifact_env,
                                                              monkeypatch):
        artifacts_module.create_placeholder_artifact(FAKE_ARTIFACT)
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            path = artifacts_module.get_artifact_path(FAKE_ARTIFACT)
        assert path.name == artifact_env["entry"]["git_tree_sha1"]
        assert artifact_env["calls"] == []
        assert any("placeholder" in str(w.message).lower() for w in caught)

    def test_placeholder_outside_ci_is_accepted(self, artifact_env):
        """Mirrors ``download_artifact``: a placeholder passes outside CI."""
        artifacts_module.create_placeholder_artifact(FAKE_ARTIFACT)
        artifact_env["stage"]()
        path = artifacts_module.get_artifact_path(FAKE_ARTIFACT)
        assert artifact_env["calls"] == []
        assert (path / "unit_test_dir" / "payload.h5").stat().st_size == 0

    def test_partially_extracted_artifact_is_redownloaded(self, artifact_env):
        """The directory exists but the expected file does not."""
        entry = artifact_env["entry"]
        (artifact_env["data_dir"] / entry["git_tree_sha1"] /
         entry["extract_dir"]).mkdir(parents=True)
        artifact_env["stage"]()
        artifacts_module.get_artifact_path(FAKE_ARTIFACT)
        assert len(artifact_env["calls"]) == 1


class TestPlaceholderAndListing:
    def test_create_placeholder(self, artifact_env, capsys):
        path = artifacts_module.create_placeholder_artifact(FAKE_ARTIFACT)
        payload = path / "unit_test_dir" / "payload.h5"
        assert payload.exists() and payload.stat().st_size == 0
        assert artifacts_module.is_placeholder_file(payload) is True
        assert "Created placeholder" in capsys.readouterr().out

    def test_create_placeholder_unknown_artifact_rejected(self, artifact_env):
        with pytest.raises(ValueError, match="Unknown artifact"):
            artifacts_module.create_placeholder_artifact("no_such_artifact")

    def test_list_artifacts_reports_absent(self, artifact_env):
        status = artifacts_module.list_artifacts()
        assert status[FAKE_ARTIFACT] is False
        assert set(status) == set(artifacts_module.ARTIFACTS)

    def test_list_artifacts_reports_placeholder_as_absent(self, artifact_env):
        artifacts_module.create_placeholder_artifact(FAKE_ARTIFACT)
        assert artifacts_module.list_artifacts()[FAKE_ARTIFACT] is False

    def test_list_artifacts_reports_real_data_as_present(self, artifact_env):
        artifact_env["stage"]()
        artifacts_module.download_artifact(FAKE_ARTIFACT, show_progress=False)
        assert artifacts_module.list_artifacts()[FAKE_ARTIFACT] is True

    def test_list_artifacts_with_an_empty_extract_dir(self, artifact_env):
        entry = artifact_env["entry"]
        (artifact_env["data_dir"] / entry["git_tree_sha1"] /
         entry["extract_dir"]).mkdir(parents=True)
        assert artifacts_module.list_artifacts()[FAKE_ARTIFACT] is False

    def test_real_registry_entries_are_well_formed(self):
        """The shipped registry must not drift out of shape."""
        for name, entry in artifacts_module.ARTIFACTS.items():
            if name == FAKE_ARTIFACT:
                continue
            assert set(entry) == {"url", "sha256", "git_tree_sha1",
                                  "extract_dir", "files"}, name
            assert entry["url"].startswith("https://"), name
            assert len(entry["sha256"]) == 64, name
            assert len(entry["git_tree_sha1"]) == 40, name
            assert entry["files"], name


# ===========================================================================
# Package __init__
# ===========================================================================


class TestPackageInit:
    def test_public_names_are_importable(self):
        import korg
        missing = [name for name in korg.__all__ if not hasattr(korg, name)]
        assert missing == [], f"korg.__all__ advertises missing names: {missing}"

    def test_x64_is_enabled(self):
        import jax
        assert jax.config.jax_enable_x64 is True, (
            "spectral synthesis needs float64; float32 silently loses ~7 digits"
        )

    def test_scattering_module_is_gone(self):
        """``korg.scattering`` was a dead duplicate of the live sibling.

        ``continuum_absorption/__init__.py`` does ``from .scattering import
        rayleigh, electron_scattering``, which is a *relative* import and
        resolves to ``continuum_absorption/scattering.py``. The top-level
        ``korg/scattering.py`` was never reachable and had a different
        ``rayleigh`` signature. This test fails if it is ever reintroduced.
        """
        import importlib
        import korg.continuum_absorption as ca

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("korg.scattering")

        assert ca.rayleigh.__module__ == "korg.continuum_absorption.scattering"
        assert (ca.electron_scattering.__module__
                == "korg.continuum_absorption.scattering")

    def test_synthesis_import_failure_degrades_gracefully(self):
        """``import korg`` must still work when a synthesis data file is missing.

        Run in a subprocess, because the fallback runs at import time and cannot
        be triggered inside an already-imported package.

        ``korg.synthesis_preparation`` is the module blocked here because it is
        the only one of the three in the guarded ``try`` that nothing outside
        ``korg/__init__`` imports. Blocking ``korg.synthesis`` or
        ``korg.marcs_interpolation`` instead does *not* exercise the fallback:
        ``korg/fit.py`` imports both unconditionally, several lines below the
        ``try``, so the exception is simply re-raised and ``import korg`` fails
        outright. In other words the guard only protects against a narrower set
        of failures than it looks like it does.
        """
        script = textwrap.dedent(
            """
            import sys, warnings

            # Measure this child too, when the parent is running under coverage,
            # so that the fallback branch is not reported as untested purely
            # because it can only be reached in a fresh interpreter.
            try:
                import coverage
                coverage.process_startup()
            except ImportError:
                pass

            class Blocker:
                def find_module(self, name, path=None):
                    return self.find_spec(name, path)
                def find_spec(self, name, path=None, target=None):
                    if name == "korg.synthesis_preparation":
                        raise FileNotFoundError("simulated missing data file")
                    return None

            sys.meta_path.insert(0, Blocker())

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                import korg

            assert any("Could not import synthesis" in str(w.message)
                       for w in caught), [str(w.message) for w in caught]
            # Every name the fallback nulls out is None, including `synthesize`
            # and `interpolate_marcs`, which imported perfectly well -- the
            # except block clobbers them regardless of which import failed.
            assert korg.synthesize is None, korg.synthesize
            assert korg.load_synthesis_data is None
            assert korg.save_synthesis_data is None
            assert korg.interpolate_marcs is None
            # the rest of the package is still usable
            assert korg.format_A_X()[0] == 12.0
            assert str(korg.Species("Fe II")) == "Fe II"
            assert korg.read_linelist is not None
            print("OK")
            """
        )
        repo_root = Path(__file__).parent.parent
        env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true")
        try:
            import coverage
            if coverage.Coverage.current() is not None:
                env["COVERAGE_PROCESS_START"] = str(repo_root / "pyproject.toml")
        except ImportError:
            pass
        result = subprocess.run([sys.executable, "-c", script],
                                capture_output=True, text=True, env=env,
                                cwd=str(repo_root))
        assert result.returncode == 0, result.stderr[-4000:]
        assert "OK" in result.stdout
