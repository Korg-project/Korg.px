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
