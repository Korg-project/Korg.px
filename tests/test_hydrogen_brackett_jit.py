"""
Tests for JIT-compatible hydrogen line absorption (Brackett series only).

These tests don't require the Stehlé profile data file.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from korg.hydrogen_line_absorption import hydrogen_line_absorption
from korg.constants import c_cgs, hplanck_eV
from korg.statmech import translational_U


class TestHydrogenLineAbsorptionBrackett:
    """Test Brackett series hydrogen line absorption."""

    def test_brackett_basic_no_jit(self):
        """Test basic Brackett series absorption without JIT."""
        # Setup - use wavelengths around Brackett alpha (4.05 microns)
        lambda_center = 4.05e-4  # cm
        wavelengths = np.linspace(lambda_center - 2e-5, lambda_center + 2e-5, 100)
        T = 10000.0  # K
        ne = 1e15  # cm^-3
        nH_I = 1e16  # cm^-3
        nHe_I = 1e15  # cm^-3
        UH_I = translational_U(1.008, T)
        xi = 2e5  # cm/s
        window_size = 3e-5  # cm

        # Calculate absorption (will use Brackett series only, no Stehlé profiles needed)
        alphas = hydrogen_line_absorption(
            wavelengths, T, ne, nH_I, nHe_I, UH_I, xi, window_size,
            use_jit=False, stark_profiles={}  # Empty dict to skip Stehlé profiles
        )

        # Check output
        assert alphas.shape == wavelengths.shape
        assert np.all(np.isfinite(alphas))
        assert np.all(alphas >= 0)

        # Should have some absorption from Brackett lines
        assert np.any(alphas > 0), "Expected non-zero absorption from Brackett lines"

    def test_brackett_jit_vs_no_jit(self):
        """Test that JIT and non-JIT produce same results for Brackett series."""
        # Setup
        lambda_center = 4.05e-4  # cm, Brackett alpha
        wavelengths = np.linspace(lambda_center - 1e-5, lambda_center + 1e-5, 50)
        T = 8000.0
        ne = 5e14
        nH_I = 5e15
        nHe_I = 5e14
        UH_I = translational_U(1.008, T)
        xi = 1e5
        window_size = 2e-5

        # Calculate with and without JIT
        alphas_no_jit = hydrogen_line_absorption(
            wavelengths, T, ne, nH_I, nHe_I, UH_I, xi, window_size,
            use_jit=False, stark_profiles={}
        )

        alphas_jit = hydrogen_line_absorption(
            wavelengths, T, ne, nH_I, nHe_I, UH_I, xi, window_size,
            use_jit=True, stark_profiles={}
        )

        # Should produce very similar results
        np.testing.assert_allclose(alphas_jit, alphas_no_jit, rtol=1e-6, atol=1e-30)

    def test_brackett_temperature_dependence(self):
        """Test temperature dependence of Brackett absorption."""
        lambda_center = 4.05e-4
        wavelengths = np.linspace(lambda_center - 1e-5, lambda_center + 1e-5, 50)
        ne = 1e15
        nH_I = 1e16
        nHe_I = 1e15
        xi = 2e5
        window_size = 2e-5

        T_low = 6000.0
        T_high = 12000.0

        UH_I_low = translational_U(1.008, T_low)
        UH_I_high = translational_U(1.008, T_high)

        alphas_low = hydrogen_line_absorption(
            wavelengths, T_low, ne, nH_I, nHe_I, UH_I_low, xi, window_size,
            use_jit=True, stark_profiles={}
        )

        alphas_high = hydrogen_line_absorption(
            wavelengths, T_high, ne, nH_I, nHe_I, UH_I_high, xi, window_size,
            use_jit=True, stark_profiles={}
        )

        # Both should be finite and non-negative
        assert np.all(np.isfinite(alphas_low)) and np.all(alphas_low >= 0)
        assert np.all(np.isfinite(alphas_high)) and np.all(alphas_high >= 0)

        # At least one should have significant absorption
        assert (alphas_low.max() > 1e-80) or (alphas_high.max() > 1e-80)

        # Temperature dependence: values should scale differently (by orders of magnitude)
        # The ratios should differ significantly where both are non-zero
        non_zero_mask = (alphas_low > 1e-100) & (alphas_high > 1e-100)
        if np.any(non_zero_mask):
            ratios = alphas_high[non_zero_mask] / alphas_low[non_zero_mask]
            ratio_range = np.ptp(ratios)
            assert ratio_range > 1.0, "Expected temperature-dependent scaling"

    def test_brackett_multiple_lines(self):
        """Test that multiple Brackett lines contribute."""
        # Wide wavelength range covering Br-alpha, Br-beta, Br-gamma, etc.
        wavelengths = np.linspace(1.5e-4, 5e-4, 200)  # 1.5 to 5 microns
        T = 10000.0
        ne = 1e15
        nH_I = 1e16
        nHe_I = 1e15
        UH_I = translational_U(1.008, T)
        xi = 2e5
        window_size = 3e-5  # Larger window to capture more lines

        alphas = hydrogen_line_absorption(
            wavelengths, T, ne, nH_I, nHe_I, UH_I, xi, window_size,
            use_jit=True, stark_profiles={}
        )

        # Should have some absorption (Brackett lines present)
        assert alphas.max() > 0, "Expected non-zero absorption from Brackett lines"
        assert np.any(alphas > 1e-80), "Expected significant absorption from at least one line"

        # Check that absorption is localized (not uniform across all wavelengths)
        # Most wavelengths should have very low absorption
        low_absorption = np.sum(alphas < alphas.max() * 0.01)
        assert low_absorption > len(wavelengths) * 0.5, "Expected localized line absorption"

    def test_can_jit_compile_brackett(self):
        """Test that Brackett-only version can be JIT compiled."""
        wavelengths_jax = jnp.linspace(4.0e-4, 4.1e-4, 30)
        T = 10000.0
        ne = 1e15
        nH_I = 1e16
        nHe_I = 1e15
        UH_I = translational_U(1.008, T)
        xi = 2e5
        window_size = 2e-5

        # Define a function that can be JIT compiled
        def compute_absorption(wls):
            # Note: stark_profiles={} must be passed outside JIT boundary
            return hydrogen_line_absorption(
                wls, T, ne, nH_I, nHe_I, UH_I, xi, window_size,
                use_jit=True, stark_profiles={}
            )

        # Call it once (will be JIT compiled internally)
        result = compute_absorption(wavelengths_jax)

        assert result.shape == wavelengths_jax.shape
        assert jnp.all(jnp.isfinite(result))
        assert jnp.any(result > 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
