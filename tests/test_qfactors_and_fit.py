"""Tests for qfactors and fit modules."""

import numpy as np
import pytest

from korg.qfactors import Qfactor, RV_prec_from_Q, RV_prec_from_noise
from korg.fit import (
    _tan_scale, _tan_unscale, _scale_params, _unscale_params,
    validate_params, _merge_windows, _numerical_dp_dscaled,
)
from korg.constants import c_cgs


class TestQfactors:
    """Tests for Q-factor and RV precision functions."""

    def _make_lsf(self, n_obs, n_synth):
        """Simple identity-like LSF: each obs pixel is one synth pixel."""
        step = n_synth // n_obs
        mat = np.zeros((n_obs, n_synth))
        for i in range(n_obs):
            mat[i, i * step] = 1.0
        return mat

    def test_Qfactor_flat_spectrum_is_zero(self):
        """A featureless flat spectrum carries no RV information → Q=0."""
        n = 100
        LSF = np.eye(n)
        wl = np.linspace(5000, 5100, n)
        Q = Qfactor(np.ones(n), wl, wl, LSF)
        assert Q == 0.0

    def test_Qfactor_positive_for_lined_spectrum(self):
        """A spectrum with line features should have Q > 0."""
        n = 200
        wl = np.linspace(5000, 5100, n)
        flux = 1.0 - 0.5 * np.exp(-((wl - 5050.0) ** 2) / 0.5 ** 2)
        LSF = np.eye(n)
        Q = Qfactor(flux, wl, wl, LSF)
        assert Q > 0.0

    def test_Qfactor_with_mask(self):
        """Masking half the spectrum should give a different (non-zero) Q."""
        n = 200
        wl = np.linspace(5000, 5100, n)
        flux = 1.0 - 0.5 * np.exp(-((wl - 5050.0) ** 2) / 0.5 ** 2)
        LSF = np.eye(n)
        mask = np.zeros(n, dtype=bool)
        mask[50:150] = True
        Q_full = Qfactor(flux, wl, wl, LSF)
        Q_masked = Qfactor(flux, wl, wl, LSF, obs_mask=mask)
        assert Q_full > 0.0
        assert Q_masked > 0.0
        assert Q_full != Q_masked

    def test_RV_prec_from_Q_formula(self):
        """Check RV_prec_from_Q against the analytical formula."""
        Q, SNR, N = 3000.0, 100.0, 50000.0
        expected = (c_cgs * 1e-2) / (Q * np.sqrt(N) * SNR)
        assert abs(RV_prec_from_Q(Q, SNR, N) - expected) < 1e-10

    def test_RV_prec_from_noise_positive(self):
        """RV precision from noise should be a positive finite number."""
        n = 100
        wl = np.linspace(5000, 5100, n)
        flux = 1.0 - 0.5 * np.exp(-((wl - 5050.0) ** 2) / 1.0 ** 2)
        LSF = np.eye(n)
        err = np.full(n, 0.01)
        rv = RV_prec_from_noise(flux, wl, wl, LSF, err)
        assert np.isfinite(rv)
        assert rv > 0.0

    def test_RV_prec_from_noise_with_mask(self):
        """Masking affects RV precision from noise."""
        n = 100
        wl = np.linspace(5000, 5100, n)
        flux = 1.0 - 0.5 * np.exp(-((wl - 5050.0) ** 2) / 1.0 ** 2)
        LSF = np.eye(n)
        err = np.full(n, 0.01)
        mask = np.ones(n, dtype=bool)
        rv_full = RV_prec_from_noise(flux, wl, wl, LSF, err)
        rv_masked = RV_prec_from_noise(flux, wl, wl, LSF, err, obs_mask=mask)
        assert abs(rv_full - rv_masked) < 1e-10  # mask=all should equal full


class TestParamScaling:
    """Tests for parameter scaling/unscaling."""

    def test_tan_scale_roundtrip(self):
        """Scaling and unscaling should recover the original value."""
        for lo, hi in [(-5, 1), (2800, 8000), (-0.5, 5.5)]:
            for p in np.linspace(lo, hi, 7)[1:-1]:
                p2 = _tan_unscale(_tan_scale(p, lo, hi), lo, hi)
                assert abs(p2 - p) < 1e-10, f"roundtrip failed for {lo},{hi},{p}"

    def test_tan_scale_midpoint_is_zero(self):
        """The midpoint of the range should map to 0."""
        lo, hi = 2800.0, 8000.0
        mid = (lo + hi) / 2
        assert abs(_tan_scale(mid, lo, hi)) < 1e-10

    def test_tan_scale_out_of_range_raises(self):
        with pytest.raises(ValueError):
            _tan_scale(9000, 2800, 8000)

    def test_scale_unscale_params_roundtrip(self):
        """Full dict roundtrip for common physical parameters."""
        params = {"Teff": 5777.0, "logg": 4.44, "M_H": -0.1,
                  "vmic": 1.5, "vsini": 10.0, "epsilon": 0.7}
        scaled = _scale_params(params)
        unscaled = _unscale_params(scaled)
        for k in params:
            assert abs(unscaled[k] - params[k]) < 1e-8, f"roundtrip failed for {k}"

    def test_scale_individual_elements(self):
        """Element abundances (e.g. Fe) should be scalable."""
        params = {"Teff": 5777.0, "logg": 4.44, "Fe": 0.3}
        scaled = _scale_params(params)
        assert "Fe" in scaled
        unscaled = _unscale_params(scaled)
        assert abs(unscaled["Fe"] - 0.3) < 1e-8


class TestValidateParams:
    """Tests for validate_params."""

    def test_missing_required_raises(self):
        with pytest.raises(ValueError, match="Teff"):
            validate_params({"logg": 4.44})

    def test_unknown_param_raises(self):
        with pytest.raises(ValueError, match="bad_param"):
            validate_params({"Teff": 5777, "logg": 4.44, "bad_param": 1.0})

    def test_param_in_both_raises(self):
        with pytest.raises(ValueError):
            validate_params({"Teff": 5777}, {"Teff": 5800, "logg": 4.44})

    def test_defaults_filled_in(self):
        ig, fp = validate_params({"Teff": 5777, "logg": 4.44})
        assert "M_H" in fp
        assert fp["M_H"] == 0.0
        assert "vmic" in fp
        assert fp["vmic"] == 1.0

    def test_fixed_param_not_overwritten_by_default(self):
        ig, fp = validate_params({"Teff": 5777, "logg": 4.44}, {"M_H": -1.0})
        assert fp["M_H"] == -1.0

    def test_element_abundance_allowed(self):
        ig, fp = validate_params({"Teff": 5777, "logg": 4.44}, {"Fe": 0.2})
        assert fp["Fe"] == pytest.approx(0.2)

    def test_returns_float_values(self):
        ig, fp = validate_params({"Teff": 5777, "logg": 4}, {})
        assert isinstance(ig["Teff"], float)
        assert isinstance(ig["logg"], float)


class TestMergeWindows:
    """Tests for the window merging helper."""

    def test_single_window(self):
        result = _merge_windows([(5000, 5100)], buffer=1.0)
        assert len(result) == 1
        assert abs(result[0][0] - 4999.0) < 1e-10
        assert abs(result[0][1] - 5101.0) < 1e-10

    def test_overlapping_merged(self):
        result = _merge_windows([(5000, 5100), (5050, 5200)], buffer=0.0)
        assert len(result) == 1
        assert result[0] == (5000.0, 5200.0)

    def test_non_overlapping_kept_separate(self):
        result = _merge_windows([(5000, 5050), (5200, 5300)], buffer=0.0)
        assert len(result) == 2

    def test_buffer_causes_merge(self):
        """Windows 5000-5050 and 5100-5200 with buffer=100 should merge."""
        result = _merge_windows([(5000, 5050), (5100, 5200)], buffer=100.0)
        assert len(result) == 1
