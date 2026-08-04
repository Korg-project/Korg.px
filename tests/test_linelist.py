"""
Tests for linelist parsing and wrapper functions.
"""

import pytest
import numpy as np


class TestVALDLinelist:
    """Test VALD linelist parsing."""

    def test_get_VALD_solar_linelist_loads(self):
        """get_VALD_solar_linelist should load without errors."""
        try:
            from korg.linelist import get_VALD_solar_linelist
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        try:
            linelist = get_VALD_solar_linelist()
        except FileNotFoundError as e:
            pytest.skip(f"VALD linelist file not found: {e}")

        # Basic sanity checks
        assert isinstance(linelist, list), "Should return a list"
        assert len(linelist) > 0, "Should contain lines"

        # Check first line has required attributes
        line = linelist[0]
        assert hasattr(line, 'wl'), "Line should have wavelength"
        assert hasattr(line, 'log_gf'), "Line should have log_gf"
        assert hasattr(line, 'species'), "Line should have species"
        assert hasattr(line, 'E_lower'), "Line should have E_lower"
        assert hasattr(line, 'gamma_rad'), "Line should have gamma_rad"
        assert hasattr(line, 'gamma_stark'), "Line should have gamma_stark"
        assert hasattr(line, 'vdW'), "Line should have vdW"

        # Check values are reasonable
        assert line.wl > 0, "Wavelength should be positive"
        assert line.wl < 1e-4, "Wavelength should be in cm (< 10000 Angstroms)"
        assert -10 < line.log_gf < 5, "log_gf should be reasonable"
        assert 0 < line.E_lower < 50, "E_lower should be in eV and reasonable"

    def test_get_VALD_solar_linelist_line_count(self):
        """get_VALD_solar_linelist should return expected number of lines."""
        try:
            from korg.linelist import get_VALD_solar_linelist
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        try:
            linelist = get_VALD_solar_linelist()
        except FileNotFoundError as e:
            pytest.skip(f"VALD linelist file not found: {e}")

        # VALD solar linelist should have many lines (thousands)
        assert len(linelist) > 100, "Should have at least 100 lines"

    def test_get_VALD_solar_linelist_sorted(self):
        """get_VALD_solar_linelist lines should be sorted by wavelength."""
        try:
            from korg.linelist import get_VALD_solar_linelist
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        try:
            linelist = get_VALD_solar_linelist()
        except FileNotFoundError as e:
            pytest.skip(f"VALD linelist file not found: {e}")

        # Check if wavelengths are sorted
        wavelengths = [line.wl for line in linelist]
        # Allow small violations for numerical precision
        for i in range(1, min(100, len(wavelengths))):
            assert wavelengths[i] >= wavelengths[i-1] * 0.9999, \
                f"Wavelengths should be roughly sorted: {wavelengths[i-1]} > {wavelengths[i]}"


class TestGALAHLinelist:
    """Test GALAH linelist parsing."""

    def test_get_GALAH_DR3_linelist_loads(self):
        """get_GALAH_DR3_linelist should load without errors."""
        try:
            from korg.linelist import get_GALAH_DR3_linelist
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        try:
            linelist = get_GALAH_DR3_linelist()
        except FileNotFoundError as e:
            pytest.skip(f"GALAH linelist file not found: {e}")

        # Basic sanity checks
        assert isinstance(linelist, list), "Should return a list"
        assert len(linelist) > 0, "Should contain lines"

        # Check first line has required attributes
        line = linelist[0]
        assert hasattr(line, 'wl'), "Line should have wavelength"
        assert hasattr(line, 'log_gf'), "Line should have log_gf"
        assert hasattr(line, 'species'), "Line should have species"
        assert hasattr(line, 'E_lower'), "Line should have E_lower"

        # Check values are reasonable
        assert line.wl > 0, "Wavelength should be positive"
        assert -10 < line.log_gf < 5, "log_gf should be reasonable"
        assert 0 < line.E_lower < 50, "E_lower should be in eV and reasonable"

    def test_get_GALAH_DR3_linelist_wavelength_range(self):
        """GALAH DR3 linelist should cover expected wavelength range."""
        try:
            from korg.linelist import get_GALAH_DR3_linelist
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        try:
            linelist = get_GALAH_DR3_linelist()
        except FileNotFoundError as e:
            pytest.skip(f"GALAH linelist file not found: {e}")

        # Get wavelengths in Angstroms
        wavelengths = [line.wl * 1e8 for line in linelist]

        # GALAH DR3 ranges from roughly 4,675 Å to 7,930 Å
        min_wl = min(wavelengths)
        max_wl = max(wavelengths)

        assert 4000 < min_wl < 5000, f"Min wavelength should be ~4675 Å, got {min_wl}"
        assert 7000 < max_wl < 9000, f"Max wavelength should be ~7930 Å, got {max_wl}"

    def test_get_GALAH_DR3_linelist_no_hydrogen(self):
        """GALAH DR3 linelist should filter out hydrogen lines."""
        try:
            from korg.linelist import get_GALAH_DR3_linelist
            from korg.species import Species
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        try:
            linelist = get_GALAH_DR3_linelist()
        except FileNotFoundError as e:
            pytest.skip(f"GALAH linelist file not found: {e}")

        # Check no H I lines are present
        H_I = Species("H_I")
        for line in linelist:
            assert line.species != H_I, "Should not contain H I lines"

    def test_get_GALAH_DR3_linelist_line_count(self):
        """get_GALAH_DR3_linelist should return expected number of lines."""
        try:
            from korg.linelist import get_GALAH_DR3_linelist
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        try:
            linelist = get_GALAH_DR3_linelist()
        except FileNotFoundError as e:
            pytest.skip(f"GALAH linelist file not found: {e}")

        # GALAH DR3 should have hundreds or thousands of lines
        assert len(linelist) > 100, "Should have at least 100 lines"


class TestAirVacuumConversion:
    """Test air/vacuum wavelength conversion functions."""

    def test_air_to_vacuum_basic(self):
        """air_to_vacuum should convert wavelengths correctly."""
        try:
            from korg.linelist import air_to_vacuum
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        # Test at 5000 Angstroms
        wl_air = 5000.0
        wl_vac = air_to_vacuum(wl_air)

        # Vacuum wavelength should be slightly longer
        assert wl_vac > wl_air, "Vacuum wavelength should be longer"
        assert wl_vac < wl_air * 1.001, "Difference should be small (~0.1%)"

        # Check approximate value (from Edlén formula)
        assert 5001 < wl_vac < 5002, f"Expected ~5001.4, got {wl_vac}"

    def test_vacuum_to_air_basic(self):
        """vacuum_to_air should convert wavelengths correctly."""
        try:
            from korg.linelist import vacuum_to_air
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        # Test at 5000 Angstroms
        wl_vac = 5000.0
        wl_air = vacuum_to_air(wl_vac)

        # Air wavelength should be slightly shorter
        assert wl_air < wl_vac, "Air wavelength should be shorter"
        assert wl_air > wl_vac * 0.999, "Difference should be small (~0.1%)"

    def test_air_vacuum_roundtrip(self):
        """air_to_vacuum and vacuum_to_air should be inverses."""
        try:
            from korg.linelist import air_to_vacuum, vacuum_to_air
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        # Test at several wavelengths
        for wl in [3000.0, 5000.0, 7000.0, 10000.0]:
            wl_vac = air_to_vacuum(wl)
            wl_roundtrip = vacuum_to_air(wl_vac)

            assert np.isclose(wl, wl_roundtrip, rtol=1e-6), \
                f"Roundtrip failed at {wl} Å: {wl} -> {wl_vac} -> {wl_roundtrip}"


class TestFilterLinelist:
    """Tests for filter_linelist (Priority 1)."""

    def make_lines(self, wavelengths_angstrom):
        """Create fake Line-like objects at given wavelengths (in Å)."""
        from korg.linelist import create_line, Species
        spec = Species('Fe I')
        return [create_line(wl, -1.0, spec, 1.0) for wl in wavelengths_angstrom]

    def test_filter_basic_range(self):
        """filter_linelist keeps only lines within range + buffer."""
        from korg.synthesis import filter_linelist

        lines = self.make_lines([4990, 4995, 5000, 5005, 5010])
        wls = np.array([4997.0, 5003.0])
        filtered = filter_linelist(lines, wls, 3.0)
        wl_angstrom = [l.wl * 1e8 for l in filtered]
        assert all(4994 <= w <= 5006 for w in wl_angstrom), \
            f"Expected lines in [4994, 5006] Å, got {wl_angstrom}"
        assert len(filtered) == 3  # 4995, 5000, 5005

    def test_filter_empty_linelist(self):
        """filter_linelist handles empty linelist."""
        from korg.synthesis import filter_linelist
        result = filter_linelist([], np.array([5000.0, 5010.0]), 10.0)
        assert result == []

    def test_filter_no_warn_on_empty_input(self):
        """filter_linelist should not warn when input is already empty."""
        import warnings
        from korg.synthesis import filter_linelist
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            filter_linelist([], np.array([5000.0]), 10.0)

    def test_filter_sorted_result(self):
        """filter_linelist result should preserve sort order."""
        from korg.synthesis import filter_linelist
        lines = self.make_lines([4980, 4990, 5000, 5010, 5020])
        wls = np.array([4985.0, 5015.0])
        filtered = filter_linelist(lines, wls, 5.0)
        wl_vals = [l.wl for l in filtered]
        assert wl_vals == sorted(wl_vals)


class TestGetReferenceWavelengthLinelist:
    """Tests for get_reference_wavelength_linelist (Priority 1)."""

    def test_empty_linelist_uses_fallback(self):
        """With empty linelist, should return built-in alpha_5000 lines."""
        from korg.synthesis import get_reference_wavelength_linelist
        result = get_reference_wavelength_linelist([], reference_wavelength_cm=5e-5)
        assert len(result) > 0, "Should return fallback lines"
        wls = [l.wl * 1e8 for l in result]
        assert min(wls) < 5000 < max(wls), "Fallback lines should span 5000 Å"

    def test_non_5000_no_fallback_raises(self):
        """Non-5000 Å reference wavelength with no lines should raise."""
        from korg.synthesis import get_reference_wavelength_linelist
        with pytest.raises((ValueError, Exception)):
            get_reference_wavelength_linelist([], reference_wavelength_cm=6e-5)

    def test_fallback_disabled_still_fills_an_empty_linelist(self):
        """``use_internal_reference_linelist=False`` does not mean "return nothing".

        Korg.jl v1.2.1 ``get_reference_wavelength_linelist`` only short-circuits to the
        built-in list when the flag is *on*; with it off it filters the user's lines to
        within 21 Å of 5000 Å and then, if that leaves nothing, still falls back to
        ``_alpha_5000_default_linelist``.  The flag chooses whether the user's lines are
        *preferred*, not whether alpha_5000 may be computed from a one-sided linelist.
        (This test previously asserted an empty result, which was the port's behaviour
        before the merge logic was implemented, not Korg's.)
        """
        from korg.synthesis import get_reference_wavelength_linelist
        from korg.data_loader import load_default_linelist
        result = get_reference_wavelength_linelist(
            [], reference_wavelength_cm=5e-5,
            use_internal_reference_linelist=False
        )
        assert len(result) == len(load_default_linelist(5e-5))

    def test_fallback_disabled_prefers_user_lines_that_span_5000(self):
        """When the user's lines do straddle 5000 Å, the flag does take effect."""
        from korg.synthesis import get_reference_wavelength_linelist
        from korg.linelist import create_line, Species
        spec = Species('Fe I')
        lines = [create_line(wl, -1.0, spec, 1.0) for wl in (4995, 5000, 5005)]
        result = get_reference_wavelength_linelist(
            lines, reference_wavelength_cm=5e-5,
            use_internal_reference_linelist=False
        )
        assert len(result) == 3

    def test_linelist_spanning_5000(self):
        """Linelist that spans 5000 Å should be returned as-is."""
        from korg.synthesis import get_reference_wavelength_linelist
        from korg.data_loader import load_default_linelist
        builtin = load_default_linelist(5e-5)
        result = get_reference_wavelength_linelist(builtin, reference_wavelength_cm=5e-5)
        assert len(result) == len(builtin)


class TestDefaultLinelist:
    """Tests for load_default_linelist (Priority 1 support)."""

    def test_loads_5000_angstrom(self):
        """load_default_linelist should return lines around 5000 Å."""
        from korg.data_loader import load_default_linelist
        lines = load_default_linelist(5e-5)
        assert len(lines) > 100, "Should return many lines"
        wls = [l.wl * 1e8 for l in lines]
        assert min(wls) < 5000 < max(wls), "Lines should span 5000 Å"

    def test_other_wavelength_returns_empty(self):
        """load_default_linelist only supports 5000 Å."""
        from korg.data_loader import load_default_linelist
        result = load_default_linelist(6e-5)
        assert result == []

    def test_lines_are_sorted(self):
        """Returned lines should be sorted by wavelength."""
        from korg.data_loader import load_default_linelist
        lines = load_default_linelist(5e-5)
        wls = [l.wl for l in lines]
        assert wls == sorted(wls)


class TestExoMolLinelist:
    """Tests for load_ExoMol_linelist (Priority 2)."""

    def test_import(self):
        """load_ExoMol_linelist should be importable."""
        from korg.linelist import load_ExoMol_linelist
        assert callable(load_ExoMol_linelist)

    def test_signature(self):
        """load_ExoMol_linelist should have the expected signature."""
        import inspect
        from korg.linelist import load_ExoMol_linelist
        sig = inspect.signature(load_ExoMol_linelist)
        params = list(sig.parameters.keys())
        assert 'spec' in params
        assert 'states_file' in params
        assert 'transitions_file' in params
        assert 'lower_wavelength' in params
        assert 'upper_wavelength' in params

    def test_with_synthetic_data(self, tmp_path):
        """load_ExoMol_linelist should parse simple synthetic ExoMol files."""
        from korg.linelist import load_ExoMol_linelist

        # Write minimal states file: id, E_wavenumber, g
        states = tmp_path / "test.states"
        states.write_text(
            "1 0.0 1\n"     # ground state
            "2 20000.0 3\n" # upper state at 20000 cm⁻¹ → 5000 Å
        )
        # Write minimal transitions file: id_upper, id_lower, A
        trans = tmp_path / "test.trans"
        trans.write_text("2 1 1.0e8\n")  # A = 1e8 s⁻¹

        lines = load_ExoMol_linelist(
            'MgH', str(states), str(trans),
            lower_wavelength=4990.0, upper_wavelength=5010.0,
            verbose=False
        )
        # The 1→2 transition at 5000 Å should be in range
        assert len(lines) >= 1, "Should find the line in range"
        assert abs(lines[0].wl * 1e8 - 5000.0) < 1.0, "Line should be near 5000 Å"
