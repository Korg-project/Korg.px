"""
Tests for synthesis_preparation utilities:
  prepare_wavelength_grid, preprocess_linelist, prepare_atmosphere.

All tests are self-contained (no MARCS grid or Julia reference required).
They use the solar atmosphere file (tests/data/sun.mod) for prepare_atmosphere
tests; those tests are skipped if the file is absent.
"""

from pathlib import Path

import numpy as np
import pytest

import korg
from korg.synthesis_preparation import (
    prepare_wavelength_grid,
    preprocess_linelist,
    prepare_atmosphere,
    PreparedLinelist,
    AtmosphereArrays,
    _vdW_to_sigma_alpha,
)
from korg.linelist import create_line, Line
from korg.species import Species
from korg.atmosphere import PlanarAtmosphere, ShellAtmosphere

ATM_FILE = Path(__file__).parent / "data" / "sun.mod"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lines(*specs):
    """
    Build a list of Lines from (wl_ang, log_gf, species_str, E_lower) tuples.
    """
    return [create_line(wl, lgf, sp, el) for wl, lgf, sp, el in specs]


# ---------------------------------------------------------------------------
# prepare_wavelength_grid
# ---------------------------------------------------------------------------

class TestPrepareWavelengthGrid:

    def test_step_mode_shape(self):
        wls_ang, wls_cm = prepare_wavelength_grid(5000.0, 5010.0, wl_step=0.01)
        assert wls_ang.shape == wls_cm.shape
        assert len(wls_ang) == 1001   # 5000, 5000.01, ..., 5010

    def test_n_points_mode_shape(self):
        wls_ang, wls_cm = prepare_wavelength_grid(5000.0, 5010.0, n_points=500)
        assert len(wls_ang) == 500
        assert len(wls_cm) == 500

    def test_n_points_overrides_step(self):
        _, wls_cm = prepare_wavelength_grid(5000.0, 5010.0,
                                            n_points=100, wl_step=0.001)
        assert len(wls_cm) == 100

    def test_unit_conversion(self):
        wls_ang, wls_cm = prepare_wavelength_grid(5000.0, 5010.0, wl_step=1.0)
        np.testing.assert_allclose(wls_cm, wls_ang * 1e-8)

    def test_endpoints(self):
        wls_ang, _ = prepare_wavelength_grid(4999.0, 5003.0, wl_step=1.0)
        assert wls_ang[0] == pytest.approx(4999.0)
        assert wls_ang[-1] == pytest.approx(5003.0)

    def test_dtype_float64(self):
        wls_ang, wls_cm = prepare_wavelength_grid(5000.0, 5010.0, wl_step=0.1)
        assert wls_ang.dtype == np.float64
        assert wls_cm.dtype == np.float64

    def test_monotonically_increasing(self):
        wls_ang, _ = prepare_wavelength_grid(5000.0, 5100.0, wl_step=0.05)
        assert np.all(np.diff(wls_ang) > 0)

    def test_error_reversed_range(self):
        with pytest.raises(ValueError, match="wl_start"):
            prepare_wavelength_grid(5010.0, 5000.0)

    def test_error_equal_endpoints(self):
        with pytest.raises(ValueError, match="wl_start"):
            prepare_wavelength_grid(5000.0, 5000.0)

    def test_error_nonpositive_step(self):
        with pytest.raises(ValueError, match="wl_step"):
            prepare_wavelength_grid(5000.0, 5010.0, wl_step=0.0)

    def test_error_n_points_too_small(self):
        with pytest.raises(ValueError, match="n_points"):
            prepare_wavelength_grid(5000.0, 5010.0, n_points=1)

    def test_single_point_n_points_2(self):
        wls_ang, _ = prepare_wavelength_grid(5000.0, 5010.0, n_points=2)
        assert len(wls_ang) == 2
        assert wls_ang[0] == pytest.approx(5000.0)
        assert wls_ang[1] == pytest.approx(5010.0)

    def test_cm_values_in_range(self):
        _, wls_cm = prepare_wavelength_grid(5000.0, 5010.0, wl_step=1.0)
        assert wls_cm[0] == pytest.approx(5e-5)
        assert wls_cm[-1] == pytest.approx(5.01e-5)


# ---------------------------------------------------------------------------
# preprocess_linelist
# ---------------------------------------------------------------------------

class TestPreprocessLinelist:

    @pytest.fixture
    def wls_cm(self):
        _, cm = prepare_wavelength_grid(5000.0, 5010.0, wl_step=0.01)
        return cm

    @pytest.fixture
    def simple_lines(self):
        return _make_lines(
            (5003.0, -1.0, "Fe I", 1.0),   # inside
            (5007.5, -0.5, "Fe I", 2.0),   # inside
            (5200.0, -2.0, "Ca I", 0.5),   # way outside
        )

    def test_filters_out_of_range_lines(self, wls_cm, simple_lines):
        pl = preprocess_linelist(simple_lines, wls_cm, line_buffer_cm=0.0)
        assert pl.n_lines == 2

    def test_line_buffer_extends_range(self, wls_cm):
        # A line at 5012 Å is 2 Å outside 5010 but within a 5 Å buffer
        lines = _make_lines(
            (5003.0, -1.0, "Fe I", 1.0),   # always in
            (5012.0, -1.0, "Fe I", 1.0),   # 2 Å outside range
        )
        pl_no_buf = preprocess_linelist(lines, wls_cm, line_buffer_cm=0.0)
        pl_with_buf = preprocess_linelist(lines, wls_cm, line_buffer_cm=5e-8)
        assert pl_no_buf.n_lines == 1
        assert pl_with_buf.n_lines == 2

    def test_empty_linelist_returns_zero_lines(self, wls_cm):
        pl = preprocess_linelist([], wls_cm)
        assert pl.n_lines == 0
        assert isinstance(pl, PreparedLinelist)

    def test_no_lines_in_range_returns_zero(self, wls_cm):
        lines = _make_lines((6000.0, -1.0, "Fe I", 1.0))
        pl = preprocess_linelist(lines, wls_cm, line_buffer_cm=0.0)
        assert pl.n_lines == 0

    def test_output_sorted_by_wavelength(self, wls_cm):
        # Provide lines in reverse wavelength order
        lines = _make_lines(
            (5009.0, -1.0, "Fe I", 1.0),
            (5003.0, -1.5, "Fe I", 2.0),
            (5006.0, -2.0, "Ca I", 0.5),
        )
        pl = preprocess_linelist(lines, wls_cm)
        assert np.all(np.diff(pl.wl) >= 0), "wl array must be non-decreasing"

    def test_wl_values_in_cm(self, wls_cm):
        lines = _make_lines((5005.0, -1.0, "Fe I", 1.0))
        pl = preprocess_linelist(lines, wls_cm)
        assert pl.n_lines == 1
        # 5005 Å in cm
        np.testing.assert_allclose(pl.wl[0], 5005e-8, rtol=1e-10)

    def test_log_gf_preserved(self, wls_cm):
        lines = _make_lines((5005.0, -1.234, "Fe I", 1.0))
        pl = preprocess_linelist(lines, wls_cm)
        assert pl.log_gf[0] == pytest.approx(-1.234)

    def test_E_lower_preserved(self, wls_cm):
        lines = _make_lines((5005.0, -1.0, "Fe I", 2.345))
        pl = preprocess_linelist(lines, wls_cm)
        assert pl.E_lower[0] == pytest.approx(2.345)

    def test_dtype_float64(self, wls_cm):
        lines = _make_lines((5005.0, -1.0, "Fe I", 1.0))
        pl = preprocess_linelist(lines, wls_cm)
        for arr in [pl.wl, pl.log_gf, pl.E_lower, pl.gamma_rad,
                    pl.gamma_stark, pl.vdW_sigma, pl.vdW_alpha, pl.mass]:
            assert arr.dtype == np.float64, f"expected float64, got {arr.dtype}"

    def test_species_id_dtype_int32(self, wls_cm):
        lines = _make_lines((5005.0, -1.0, "Fe I", 1.0))
        pl = preprocess_linelist(lines, wls_cm)
        assert pl.species_id.dtype == np.int32

    def test_multi_species_unique_ids(self, wls_cm):
        lines = _make_lines(
            (5003.0, -1.0, "Fe I", 1.0),
            (5005.0, -1.0, "Ca I", 0.5),
            (5008.0, -1.5, "Fe I", 2.0),   # same species as first line
        )
        pl = preprocess_linelist(lines, wls_cm)
        assert pl.n_lines == 3
        # Fe I and Ca I → two unique species
        assert len(pl.species_list) == 2
        # Both Fe I lines share the same species_id
        fe_id  = pl.species_id[np.isclose(pl.wl, 5003e-8)]
        fe_id2 = pl.species_id[np.isclose(pl.wl, 5008e-8)]
        assert fe_id[0] == fe_id2[0]

    def test_species_list_length_matches_unique_count(self, wls_cm):
        lines = _make_lines(
            (5003.0, -1.0, "Fe I", 1.0),
            (5005.0, -1.5, "Fe II", 1.5),
            (5007.0, -2.0, "Ca I", 0.5),
        )
        pl = preprocess_linelist(lines, wls_cm)
        assert len(pl.species_list) == 3

    def test_h_i_lines_excluded(self, wls_cm):
        # H I lines must be excluded — hydrogen opacity is handled separately
        fe_line = create_line(5005.0, -1.0, "Fe I", 1.0)
        h_line = create_line(5005.0, 0.0, "H I", 10.0)
        pl = preprocess_linelist([fe_line, h_line], wls_cm)
        # H I excluded, only Fe I survives
        assert pl.n_lines == 1
        assert pl.species_list[0] == Species("Fe_I")

    def test_mass_positive(self, wls_cm):
        lines = _make_lines(
            (5003.0, -1.0, "Fe I", 1.0),
            (5007.0, -1.0, "Ca I", 0.5),
        )
        pl = preprocess_linelist(lines, wls_cm)
        assert np.all(pl.mass > 0)

    def test_is_molecule_false_for_atomic_lines(self, wls_cm):
        lines = _make_lines((5005.0, -1.0, "Fe I", 1.0))
        pl = preprocess_linelist(lines, wls_cm)
        assert not pl.is_molecule[0]

    def test_gamma_rad_non_negative(self, wls_cm):
        lines = _make_lines((5005.0, -1.0, "Fe I", 1.0))
        pl = preprocess_linelist(lines, wls_cm)
        assert pl.gamma_rad[0] >= 0.0

    def test_all_arrays_same_length(self, wls_cm):
        lines = _make_lines(
            (5003.0, -1.0, "Fe I", 1.0),
            (5007.0, -1.5, "Ca I", 0.5),
        )
        pl = preprocess_linelist(lines, wls_cm)
        n = pl.n_lines
        for field, arr in [
            ("wl", pl.wl), ("log_gf", pl.log_gf), ("E_lower", pl.E_lower),
            ("gamma_rad", pl.gamma_rad), ("gamma_stark", pl.gamma_stark),
            ("vdW_sigma", pl.vdW_sigma), ("vdW_alpha", pl.vdW_alpha),
            ("mass", pl.mass), ("is_molecule", pl.is_molecule),
            ("species_id", pl.species_id),
        ]:
            assert len(arr) == n, f"field '{field}' has length {len(arr)}, expected {n}"

    def test_as_jax_converts_arrays(self, wls_cm):
        import jax.numpy as jnp
        lines = _make_lines((5005.0, -1.0, "Fe I", 1.0))
        pl = preprocess_linelist(lines, wls_cm)
        jpl = pl.as_jax()
        assert isinstance(jpl.wl, jnp.ndarray)
        assert isinstance(jpl.log_gf, jnp.ndarray)

    def test_as_jax_preserves_values(self, wls_cm):
        lines = _make_lines((5005.0, -1.234, "Fe I", 2.345))
        pl = preprocess_linelist(lines, wls_cm)
        jpl = pl.as_jax()
        np.testing.assert_allclose(np.asarray(jpl.wl), pl.wl)
        np.testing.assert_allclose(np.asarray(jpl.log_gf), pl.log_gf)


# ---------------------------------------------------------------------------
# vdW packing helper
# ---------------------------------------------------------------------------

class TestVdWToSigmaAlpha:

    def test_tuple_passthrough(self):
        sigma, alpha = _vdW_to_sigma_alpha((0.278, 0.227))
        assert sigma == pytest.approx(0.278)
        assert alpha == pytest.approx(0.227)

    def test_negative_scalar_log_c6(self):
        # Negative scalar is log10(gamma_vdW); converted to linear like _vdW_to_tuple
        sigma, alpha = _vdW_to_sigma_alpha(-7.5)
        assert sigma == pytest.approx(10**-7.5)
        assert alpha == pytest.approx(-1.0)

    def test_none_returns_no_broadening(self):
        sigma, alpha = _vdW_to_sigma_alpha(None)
        assert sigma == pytest.approx(0.0)
        assert alpha == pytest.approx(-1.0)

    def test_nan_returns_no_broadening(self):
        sigma, alpha = _vdW_to_sigma_alpha(float('nan'))
        assert sigma == pytest.approx(0.0)
        assert alpha == pytest.approx(-1.0)

    def test_zero_tuple_sentinel(self):
        sigma, alpha = _vdW_to_sigma_alpha((0.0, -1.0))
        assert sigma == pytest.approx(0.0)
        assert alpha == pytest.approx(-1.0)

    def test_a_small_positive_scalar_is_taken_as_a_direct_coefficient(self):
        """A positive scalar is a fudge factor, not a log; it passes through."""
        sigma, alpha = _vdW_to_sigma_alpha(1.5e-31)
        assert sigma == pytest.approx(1.5e-31, rel=0)
        assert alpha == pytest.approx(-1.0)

    def test_a_list_is_accepted_like_a_tuple(self):
        sigma, alpha = _vdW_to_sigma_alpha([0.278, 0.227])
        assert sigma == pytest.approx(0.278)
        assert alpha == pytest.approx(0.227)


class TestPrepareWavelengthGridUnreachableGuard:
    """``prepare_wavelength_grid`` has a guard that cannot fire.

    ``wl_start < wl_end`` is validated first, and both the ``n_points`` branch
    (``n_points >= 2``) and the ``np.arange`` branch then always produce at
    least one sample — so the ``"Grid is empty"`` ``ValueError`` is dead code.
    Recorded here rather than left as an unexplained coverage hole.
    """

    @pytest.mark.parametrize("kwargs", [
        {"wl_step": 1e6},          # step far wider than the range
        {"n_points": 2},
        {"wl_step": 1e-6},
    ])
    def test_the_grid_is_never_empty(self, kwargs):
        ang, cm = prepare_wavelength_grid(5000.0, 5000.001, **kwargs)
        assert len(ang) >= 1
        np.testing.assert_allclose(cm, ang * 1e-8, rtol=1e-15)


# ---------------------------------------------------------------------------
# prepare_atmosphere
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def solar_atm():
    if not ATM_FILE.exists():
        pytest.skip(f"Solar atmosphere not found: {ATM_FILE}")
    return korg.read_model_atmosphere(str(ATM_FILE))


class TestPrepareAtmosphere:

    def test_returns_atmosphere_arrays(self, solar_atm):
        aa = prepare_atmosphere(solar_atm)
        assert isinstance(aa, AtmosphereArrays)

    def test_all_arrays_same_length(self, solar_atm):
        aa = prepare_atmosphere(solar_atm)
        n = aa.n_layers
        for field, arr in [
            ("T", aa.T), ("ne", aa.ne), ("n_total", aa.n_total),
            ("log_tau_ref", aa.log_tau_ref), ("z", aa.z),
        ]:
            assert len(arr) == n, f"field '{field}' length {len(arr)} ≠ {n}"

    def test_dtype_float64(self, solar_atm):
        aa = prepare_atmosphere(solar_atm)
        for field, arr in [("T", aa.T), ("ne", aa.ne), ("n_total", aa.n_total),
                           ("log_tau_ref", aa.log_tau_ref), ("z", aa.z)]:
            assert arr.dtype == np.float64, f"field '{field}' has dtype {arr.dtype}"

    def test_temperature_positive(self, solar_atm):
        aa = prepare_atmosphere(solar_atm)
        assert np.all(aa.T > 0), "All temperatures must be positive"

    def test_electron_density_positive(self, solar_atm):
        aa = prepare_atmosphere(solar_atm)
        assert np.all(aa.ne > 0)

    def test_total_density_positive(self, solar_atm):
        aa = prepare_atmosphere(solar_atm)
        assert np.all(aa.n_total > 0)

    def test_solar_n_layers(self, solar_atm):
        aa = prepare_atmosphere(solar_atm)
        # Standard MARCS solar model has 56 layers
        assert aa.n_layers == 56

    def test_planar_not_spherical(self, solar_atm):
        if not isinstance(solar_atm, PlanarAtmosphere):
            pytest.skip("sun.mod is not planar")
        aa = prepare_atmosphere(solar_atm)
        assert aa.spherical is False

    def test_log_tau_ref_finite(self, solar_atm):
        aa = prepare_atmosphere(solar_atm)
        assert np.all(np.isfinite(aa.log_tau_ref))

    def test_log_tau_ref_mostly_increasing(self, solar_atm):
        # tau_ref increases with depth (inward layers have larger optical depth)
        aa = prepare_atmosphere(solar_atm)
        # Allow one non-monotone step (surface boundary artefact)
        diffs = np.diff(aa.log_tau_ref)
        n_decreasing = int(np.sum(diffs < 0))
        assert n_decreasing <= 1, f"{n_decreasing} non-monotone steps in log_tau_ref"

    def test_as_jax_converts(self, solar_atm):
        import jax.numpy as jnp
        aa = prepare_atmosphere(solar_atm).as_jax()
        assert isinstance(aa.T, jnp.ndarray)
        assert isinstance(aa.ne, jnp.ndarray)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="PlanarAtmosphere or ShellAtmosphere"):
            prepare_atmosphere({"T": [5000.0]})

    def test_matches_direct_attribute_access(self, solar_atm):
        aa = prepare_atmosphere(solar_atm)
        np.testing.assert_array_equal(aa.T, np.asarray(solar_atm.T))
        np.testing.assert_array_equal(aa.ne, np.asarray(solar_atm.ne))


# ---------------------------------------------------------------------------
# Integration: preprocess_linelist feeds into synthesize
# ---------------------------------------------------------------------------

class TestPreprocessLinkedToSynthesize:
    """
    Smoke tests confirming that preprocess_linelist output is consistent with
    what synthesize() would compute (same number of lines, same wavelengths).
    """

    @pytest.fixture(scope="class")
    def setup(self):
        if not ATM_FILE.exists():
            pytest.skip(f"Solar atmosphere not found: {ATM_FILE}")
        atm = korg.read_model_atmosphere(str(ATM_FILE))
        A_X = korg.format_A_X()
        lines = _make_lines(
            (5003.0, -1.0, "Fe I", 1.0),
            (5007.0, -0.5, "Fe I", 2.0),
            (5200.0, -2.0, "Ca I", 0.5),   # outside range
        )
        wls_ang, wls_cm = prepare_wavelength_grid(5000.0, 5010.0, wl_step=0.05)
        return {"atm": atm, "A_X": A_X, "lines": lines,
                "wls_ang": wls_ang, "wls_cm": wls_cm}

    def test_filtered_count_matches_manual(self, setup):
        pl = preprocess_linelist(setup["lines"], setup["wls_cm"],
                                  line_buffer_cm=10e-8)
        # The 5200 Å Ca line is > 10 Å outside [5000, 5010], so filtered
        assert pl.n_lines == 2

    def test_wl_grid_used_directly_in_synthesize(self, setup):
        flux, _ = korg.synthesize(
            setup["atm"], setup["lines"][:2], setup["wls_ang"], setup["A_X"],
            hydrogen_lines=False,
        )
        assert len(flux) == len(setup["wls_ang"])

    def test_preprocess_wl_matches_synthesize_output_grid(self, setup):
        # Wavelengths produced by prepare_wavelength_grid should be identical to
        # the grid synthesize reports back through ``synth``.
        wls, flux, _ = korg.synth(
            setup["atm"], setup["lines"][:2], setup["wls_ang"], setup["A_X"],
            hydrogen_lines=False,
        )
        np.testing.assert_array_equal(wls, setup["wls_ang"])
        assert len(flux) == len(setup["wls_ang"])
