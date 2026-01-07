"""
Tests for JIT compatibility of hydrogen line absorption functions.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from korg.hydrogen_line_absorption import (
    brackett_line_stark_profiles,
    brackett_oscillator_strength,
    holtsmark_profile,
    hummer_mihalas_w,
    exponential_integral_1,
    bracket_line_interpolator,
)
from korg.constants import c_cgs, hplanck_eV


class TestBrackettStarkProfilesJIT:
    """Test JIT compilation of Brackett Stark profiles."""

    def test_brackett_stark_profiles_basic(self):
        """Test basic Brackett Stark profile calculation."""
        # Setup
        m = 5  # Brackett alpha
        n = 4
        E = 13.6 * (1 / n**2 - 1 / m**2)  # eV
        lambda0 = hplanck_eV * c_cgs / E  # cm

        wavelengths = jnp.linspace(lambda0 - 1e-6, lambda0 + 1e-6, 101)
        T = 10000.0  # K
        ne = 1e16  # cm^-3

        # Calculate profiles
        impact, quasistatic = brackett_line_stark_profiles(m, wavelengths, lambda0, T, ne)

        # Check outputs
        assert impact.shape == wavelengths.shape
        assert quasistatic.shape == wavelengths.shape
        assert jnp.all(jnp.isfinite(impact))
        assert jnp.all(jnp.isfinite(quasistatic))
        assert jnp.all(impact >= 0)
        assert jnp.all(quasistatic >= 0)

        # Profiles should be symmetric around line center
        mid = len(wavelengths) // 2
        # Check that profiles have maximum near center
        assert impact[mid - 5:mid + 5].max() > impact[0]
        assert quasistatic[mid - 5:mid + 5].max() > quasistatic[0]

    def test_brackett_stark_profiles_jit(self):
        """Test that brackett_line_stark_profiles is JIT-compatible."""
        # Setup
        m = 6
        n = 4
        E = 13.6 * (1 / n**2 - 1 / m**2)
        lambda0 = hplanck_eV * c_cgs / E

        wavelengths = jnp.linspace(lambda0 - 5e-7, lambda0 + 5e-7, 51)
        T = 8000.0
        ne = 5e15

        # JIT compile
        jitted_profiles = jax.jit(brackett_line_stark_profiles, static_argnums=(0,))

        # Call JIT version
        impact_jit, quasistatic_jit = jitted_profiles(m, wavelengths, lambda0, T, ne)

        # Call non-JIT version
        impact_nojit, quasistatic_nojit = brackett_line_stark_profiles(m, wavelengths, lambda0, T, ne)

        # Should produce identical results
        np.testing.assert_allclose(impact_jit, impact_nojit, rtol=1e-10)
        np.testing.assert_allclose(quasistatic_jit, quasistatic_nojit, rtol=1e-10)

    def test_brackett_stark_profiles_temperature_dependence(self):
        """Test temperature dependence of Stark profiles."""
        m = 5
        n = 4
        E = 13.6 * (1 / n**2 - 1 / m**2)
        lambda0 = hplanck_eV * c_cgs / E

        wavelengths = jnp.linspace(lambda0 - 1e-6, lambda0 + 1e-6, 101)
        ne = 1e16

        # Test at different temperatures
        T_low = 5000.0
        T_high = 15000.0

        impact_low, quasistatic_low = brackett_line_stark_profiles(m, wavelengths, lambda0, T_low, ne)
        impact_high, quasistatic_high = brackett_line_stark_profiles(m, wavelengths, lambda0, T_high, ne)

        # Profiles should differ with temperature
        assert not jnp.allclose(impact_low, impact_high)
        assert not jnp.allclose(quasistatic_low, quasistatic_high)

        # All should be finite and non-negative
        assert jnp.all(jnp.isfinite(impact_low)) and jnp.all(impact_low >= 0)
        assert jnp.all(jnp.isfinite(impact_high)) and jnp.all(impact_high >= 0)


class TestBracketLineInterpolatorJIT:
    """Test JIT compatibility of bracket line interpolator."""

    def test_bracket_line_interpolator_basic(self):
        """Test basic bracket line interpolator."""
        m = 5
        n = 4
        E = 13.6 * (1 / n**2 - 1 / m**2)
        lambda0 = hplanck_eV * c_cgs / E

        T = 10000.0
        ne = 1e16
        xi = 2e5  # 2 km/s microturbulence

        # Create interpolator
        interp_func, window = bracket_line_interpolator(m, lambda0, T, ne, xi)

        # Test interpolator at a few points
        test_wls = jnp.array([lambda0 - 1e-7, lambda0, lambda0 + 1e-7])
        results = jax.vmap(interp_func)(test_wls)

        # Check results
        assert results.shape == test_wls.shape
        assert jnp.all(jnp.isfinite(results))
        assert jnp.all(results >= 0)

        # Profile should peak near line center
        assert results[1] >= results[0]
        assert results[1] >= results[2]

    def test_bracket_line_interpolator_jit_compatible(self):
        """Test that the interpolator function returned is JIT-compatible."""
        m = 6
        n = 4
        E = 13.6 * (1 / n**2 - 1 / m**2)
        lambda0 = hplanck_eV * c_cgs / E

        T = 8000.0
        ne = 5e15
        xi = 1e5

        interp_func, window = bracket_line_interpolator(m, lambda0, T, ne, xi)

        # JIT compile a function that uses the interpolator
        @jax.jit
        def evaluate_profile(wl):
            return interp_func(wl)

        # Test on a single wavelength
        result = evaluate_profile(lambda0)

        assert jnp.isfinite(result)
        assert result >= 0

    def test_bracket_line_interpolator_vectorized(self):
        """Test that bracket line interpolator works with vectorized calls."""
        m = 5
        n = 4
        E = 13.6 * (1 / n**2 - 1 / m**2)
        lambda0 = hplanck_eV * c_cgs / E

        T = 10000.0
        ne = 1e16
        xi = 2e5

        interp_func, window = bracket_line_interpolator(m, lambda0, T, ne, xi)

        # Create wavelength array
        wls = jnp.linspace(lambda0 - window/2, lambda0 + window/2, 50)

        # Vectorized evaluation
        results = jax.vmap(interp_func)(wls)

        assert results.shape == wls.shape
        assert jnp.all(jnp.isfinite(results))


class TestHelperFunctionsJIT:
    """Test that helper functions are JIT-compatible."""

    def test_exponential_integral_1_jit(self):
        """Test E1 function JIT compatibility."""
        @jax.jit
        def compute_e1(x):
            return exponential_integral_1(x)

        # Test at various points
        x_vals = jnp.array([0.01, 0.1, 1.0, 10.0, 30.0])
        results = jax.vmap(compute_e1)(x_vals)

        assert results.shape == x_vals.shape
        assert jnp.all(jnp.isfinite(results))

    def test_brackett_oscillator_strength_jit(self):
        """Test oscillator strength JIT compatibility."""
        @jax.jit
        def compute_f(m):
            return brackett_oscillator_strength(4, m)

        # Test for Brackett alpha, beta, gamma
        result_5 = compute_f(5)
        result_6 = compute_f(6)
        result_7 = compute_f(7)

        assert jnp.isfinite(result_5)
        assert jnp.isfinite(result_6)
        assert jnp.isfinite(result_7)

        # Oscillator strengths should be positive and decrease with m
        assert result_5 > result_6 > result_7 > 0

    def test_holtsmark_profile_jit(self):
        """Test Holtsmark profile JIT compatibility."""
        @jax.jit
        def compute_holtsmark(beta, P):
            return holtsmark_profile(beta, P)

        # Test at various beta values
        betas = jnp.array([1.0, 10.0, 30.0, 100.0, 600.0])
        P = 0.5

        results = jax.vmap(lambda b: compute_holtsmark(b, P))(betas)

        assert results.shape == betas.shape
        assert jnp.all(jnp.isfinite(results))
        assert jnp.all(results > 0)

    def test_hummer_mihalas_w_jit(self):
        """Test H&M occupation probability JIT compatibility."""
        @jax.jit
        def compute_w(T, n_eff, nH, nHe, ne):
            return hummer_mihalas_w(T, n_eff, nH, nHe, ne, use_hubeny_generalization=False)

        T = 10000.0
        n_eff = 5.0
        nH = 1e17
        nHe = 1e16
        ne = 1e14

        result = compute_w(T, n_eff, nH, nHe, ne)

        assert jnp.isfinite(result)
        assert 0 < result <= 1  # Occupation probability should be between 0 and 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
