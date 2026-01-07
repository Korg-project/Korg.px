"""
Tests for fully JIT-compatible hydrogen_line_absorption function.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from korg.hydrogen_line_absorption import hydrogen_line_absorption
from korg.hydrogen_stark_data import hline_stark_profiles
from korg.constants import c_cgs, hplanck_eV
from korg.statmech import translational_U


@pytest.mark.skipif(len(hline_stark_profiles) == 0,
                    reason="Stark profile data not available")
class TestHydrogenLineAbsorptionFull:
    """Test full hydrogen line absorption including Stehlé profiles."""

    def test_basic_functionality_no_jit(self):
        """Test basic functionality without JIT."""
        # Setup
        wavelengths = np.linspace(6540e-8, 6580e-8, 100)  # cm, around H-alpha
        T = 10000.0  # K
        ne = 1e15  # cm^-3
        nH_I = 1e16  # cm^-3
        nHe_I = 1e15  # cm^-3
        UH_I = translational_U(1.008, T)
        xi = 2e5  # cm/s
        window_size = 20e-8  # cm

        # Calculate absorption
        alphas = hydrogen_line_absorption(
            wavelengths, T, ne, nH_I, nHe_I, UH_I, xi, window_size,
            use_jit=False
        )

        # Check output
        assert alphas.shape == wavelengths.shape
        assert np.all(np.isfinite(alphas))
        assert np.all(alphas >= 0)

        # Should have some absorption (not all zeros)
        assert np.any(alphas > 0)

    def test_jit_vs_no_jit_match(self):
        """Test that JIT and non-JIT versions produce same results."""
        # Setup
        wavelengths = np.linspace(6540e-8, 6580e-8, 50)  # cm, around H-alpha
        T = 8000.0  # K
        ne = 5e14  # cm^-3
        nH_I = 5e15  # cm^-3
        nHe_I = 5e14  # cm^-3
        UH_I = translational_U(1.008, T)
        xi = 1e5  # cm/s
        window_size = 15e-8  # cm

        # Calculate with and without JIT
        alphas_no_jit = hydrogen_line_absorption(
            wavelengths, T, ne, nH_I, nHe_I, UH_I, xi, window_size,
            use_jit=False
        )

        alphas_jit = hydrogen_line_absorption(
            wavelengths, T, ne, nH_I, nHe_I, UH_I, xi, window_size,
            use_jit=True
        )

        # Should produce similar results (within numerical tolerance)
        # Allow 1% relative difference due to interpolation differences
        np.testing.assert_allclose(alphas_jit, alphas_no_jit, rtol=1e-2, atol=1e-30)

    def test_temperature_dependence(self):
        """Test that absorption depends on temperature."""
        wavelengths = np.linspace(6540e-8, 6580e-8, 50)
        ne = 1e15
        nH_I = 1e16
        nHe_I = 1e15
        xi = 2e5
        window_size = 20e-8

        T_low = 6000.0
        T_high = 12000.0

        UH_I_low = translational_U(1.008, T_low)
        UH_I_high = translational_U(1.008, T_high)

        alphas_low = hydrogen_line_absorption(
            wavelengths, T_low, ne, nH_I, nHe_I, UH_I_low, xi, window_size,
            use_jit=True
        )

        alphas_high = hydrogen_line_absorption(
            wavelengths, T_high, ne, nH_I, nHe_I, UH_I_high, xi, window_size,
            use_jit=True
        )

        # Profiles should differ with temperature
        assert not np.allclose(alphas_low, alphas_high)

    def test_brackett_only(self):
        """Test Brackett series contribution (no Stehlé profiles)."""
        # Use wavelengths where only Brackett lines contribute
        wavelengths = np.linspace(4e-5, 5e-5, 50)  # cm, far from Balmer
        T = 10000.0
        ne = 1e15
        nH_I = 1e16
        nHe_I = 1e15
        UH_I = translational_U(1.008, T)
        xi = 2e5
        window_size = 1e-6  # Small window

        alphas = hydrogen_line_absorption(
            wavelengths, T, ne, nH_I, nHe_I, UH_I, xi, window_size,
            use_jit=True
        )

        # Should have some Brackett absorption
        assert alphas.shape == wavelengths.shape
        assert np.all(np.isfinite(alphas))
        assert np.all(alphas >= 0)


@pytest.mark.skipif(len(hline_stark_profiles) == 0,
                    reason="Stark profile data not available")
class TestHydrogenLineAbsorptionJIT:
    """Test JIT compilation of hydrogen_line_absorption."""

    def test_can_jit_compile_wrapper(self):
        """Test that we can JIT compile a wrapper function."""
        wavelengths = jnp.linspace(6540e-8, 6580e-8, 30)
        T = 10000.0
        ne = 1e15
        nH_I = 1e16
        nHe_I = 1e15
        UH_I = translational_U(1.008, T)
        xi = 2e5
        window_size = 20e-8

        # Create a JIT-compiled wrapper
        @jax.jit
        def compute_absorption(wls):
            return hydrogen_line_absorption(
                wls, T, ne, nH_I, nHe_I, UH_I, xi, window_size,
                use_jit=True
            )

        # This should work
        result = compute_absorption(wavelengths)

        assert result.shape == wavelengths.shape
        assert jnp.all(jnp.isfinite(result))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
