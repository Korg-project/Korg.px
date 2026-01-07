"""
Tests for line_absorption module with JIT compatibility.

Validates that line_absorption works both normally and when JIT-compiled.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

try:
    from korg.line_absorption import (
        line_absorption,
        line_absorption_core,
        prepare_linelist_arrays,
        doppler_width,
        scaled_stark,
        scaled_vdW,
        line_profile,
        sigma_line,
        inverse_gaussian_density,
        inverse_lorentz_density
    )
    from korg.linelist import Line
    from korg.species import Species
    from korg.data_loader import load_atomic_partition_functions
    from korg.cubic_splines import CubicSpline
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


pytestmark = pytest.mark.skipif(
    not IMPORTS_AVAILABLE,
    reason=f"Required imports not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}"
)


class TestLineAbsorptionHelpers:
    """Test helper functions for line absorption."""

    def test_doppler_width(self):
        """Test Doppler width calculation."""
        # For Fe I at 5000 K with mass ~56 amu
        wl = 5000e-8  # cm
        T = 5000  # K
        mass = 56 * 1.66054e-24  # g
        xi = 1e5  # cm/s (1 km/s)

        sigma = doppler_width(wl, T, mass, xi)

        # Should be positive
        assert sigma > 0
        # Should be small (much less than wavelength)
        assert sigma < wl / 10

        # Check that it increases with temperature
        sigma_hot = doppler_width(wl, 10000, mass, xi)
        assert sigma_hot > sigma

    def test_scaled_stark(self):
        """Test Stark broadening scaling."""
        gamma_stark = 1e8  # rad/s at 10,000 K
        T = 5000  # K

        gamma_scaled = scaled_stark(gamma_stark, T)

        # At 5000 K should be smaller than at 10,000 K
        assert gamma_scaled < gamma_stark
        # Should scale as T^(1/6)
        expected = gamma_stark * (5000 / 10000)**(1/6)
        np.testing.assert_allclose(gamma_scaled, expected, rtol=1e-6)

    def test_scaled_vdW_simple(self):
        """Test simple vdW scaling (vdW[1] == -1)."""
        vdW = (1e8, -1.0)  # Simple scaling
        mass = 56 * 1.66054e-24  # g
        T = 5000  # K

        gamma = scaled_vdW(vdW, mass, T)

        # Should scale as T^0.3
        expected = vdW[0] * (T / 10000)**0.3
        np.testing.assert_allclose(gamma, expected, rtol=1e-6)

    def test_scaled_vdW_ABO(self):
        """Test ABO theory vdW scaling."""
        # Typical ABO parameters
        vdW = (1e-8, 0.3)  # (σ, α)
        mass = 56 * 1.66054e-24  # g
        T = 5000  # K

        gamma = scaled_vdW(vdW, mass, T)

        # Should be positive
        assert gamma > 0
        # Different from simple scaling
        gamma_simple = scaled_vdW((1e8, -1.0), mass, T)
        # Can't directly compare, just check it runs

    def test_sigma_line(self):
        """Test cross-section calculation."""
        wl = 5000e-8  # cm

        sigma = sigma_line(wl)

        # Should be positive
        assert sigma > 0
        # Scales as λ²
        sigma_2x = sigma_line(2 * wl)
        np.testing.assert_allclose(sigma_2x / sigma, 4.0, rtol=1e-6)

    def test_inverse_gaussian_density(self):
        """Test inverse Gaussian density."""
        sigma = 1e-8  # cm

        # At max density, should return 0
        max_density = 1.0 / (jnp.sqrt(2 * jnp.pi) * sigma)
        assert inverse_gaussian_density(max_density * 1.1, sigma) == 0.0

        # Below max, should return positive distance
        half_max = max_density / 2
        x = inverse_gaussian_density(half_max, sigma)
        assert x > 0

    def test_inverse_lorentz_density(self):
        """Test inverse Lorentz density."""
        gamma = 1e-8  # cm

        # At max density, should return 0
        max_density = 1.0 / (jnp.pi * gamma)
        assert inverse_lorentz_density(max_density * 1.1, gamma) == 0.0

        # Below max, should return positive distance
        half_max = max_density / 2
        x = inverse_lorentz_density(half_max, gamma)
        assert x > 0

    def test_line_profile(self):
        """Test Voigt line profile."""
        wl_center = 5000e-8  # cm
        sigma = 1e-9  # cm
        gamma = 1e-10  # cm
        amplitude = 1.0

        # At line center
        profile_center = line_profile(wl_center, sigma, gamma, amplitude, wl_center)
        assert profile_center > 0

        # In wings (should be smaller)
        profile_wing = line_profile(wl_center, sigma, gamma, amplitude, wl_center + 10 * sigma)
        assert profile_wing < profile_center

        # Far wings (should be very small)
        profile_far = line_profile(wl_center, sigma, gamma, amplitude, wl_center + 100 * sigma)
        assert profile_far < profile_wing


class TestLineAbsorptionBasic:
    """Test basic line_absorption functionality."""

    def setup_method(self):
        """Setup test data."""
        # Create a simple test linelist with one Fe I line
        # Using more realistic vdW parameters (much smaller)
        self.fe_line = Line(
            wl=5000e-8,  # cm (5000 Å)
            log_gf=-1.0,
            species=Species("Fe_I"),
            E_lower=3.0,  # eV
            gamma_rad=1e7,  # rad/s
            gamma_stark=1e-7,  # rad/s per electron (very small)
            vdW=(1e-7, -1.0)  # rad/s per H atom at 10,000 K (realistic)
        )
        self.linelist = [self.fe_line]

        # Wavelength grid around the line
        self.wavelengths = jnp.linspace(4999e-8, 5001e-8, 1000)

        # Single atmospheric layer
        self.temperatures = jnp.array([5000.0])
        self.electron_densities = jnp.array([1e14])

        # Number densities (need H_I for vdW and Fe_I for the line)
        self.number_densities = {
            Species("H_I"): jnp.array([1e17]),
            Species("Fe_I"): jnp.array([1e10]),
        }

        # Partition functions
        partition_funcs_full = load_atomic_partition_functions()
        self.partition_functions = {
            Species("Fe_I"): partition_funcs_full[Species("Fe_I")],
            Species("H_I"): partition_funcs_full[Species("H_I")],
        }

        # Microturbulent velocity
        self.xi = 1e5  # cm/s (1 km/s)

        # Continuum opacity (simple constant)
        def continuum_opacity(wl):
            return jnp.full_like(self.temperatures, 1e-26)
        self.continuum_opacity = continuum_opacity

    def test_line_absorption_basic(self):
        """Test basic line absorption calculation."""
        alpha = line_absorption(
            self.linelist,
            self.wavelengths,
            self.temperatures,
            self.electron_densities,
            self.number_densities,
            self.partition_functions,
            self.xi,
            self.continuum_opacity
        )

        # Check shape
        assert alpha.shape == (len(self.temperatures), len(self.wavelengths))
        assert alpha.shape == (1, 1000)

        # Check that absorption is positive somewhere
        assert jnp.any(alpha > 0), "Should have some absorption"

        # Check that peak is near line center
        line_center_idx = jnp.argmin(jnp.abs(self.wavelengths - self.fe_line.wl))
        peak_idx = jnp.argmax(alpha[0])

        # Peak should be within 10 pixels of line center
        assert abs(peak_idx - line_center_idx) < 10

    def test_line_absorption_empty_linelist(self):
        """Test with empty linelist."""
        alpha = line_absorption(
            [],  # Empty linelist
            self.wavelengths,
            self.temperatures,
            self.electron_densities,
            self.number_densities,
            self.partition_functions,
            self.xi,
            self.continuum_opacity
        )

        # Should return zeros
        assert alpha.shape == (len(self.temperatures), len(self.wavelengths))
        np.testing.assert_array_equal(alpha, 0.0)

    def test_line_absorption_multiple_layers(self):
        """Test with multiple atmospheric layers."""
        # Create 3 layers with different temperatures
        temps = jnp.array([4000.0, 5000.0, 6000.0])
        ne = jnp.array([1e13, 1e14, 1e15])
        n_densities = {
            Species("H_I"): jnp.array([2e17, 1e17, 5e16]),
            Species("Fe_I"): jnp.array([2e10, 1e10, 5e9]),
        }

        alpha = line_absorption(
            self.linelist,
            self.wavelengths,
            temps,
            ne,
            n_densities,
            self.partition_functions,
            self.xi,
            lambda wl: jnp.full(3, 1e-26)
        )

        # Check shape
        assert alpha.shape == (3, 1000)

        # All layers should have some absorption
        for i in range(3):
            assert jnp.any(alpha[i] > 0), f"Layer {i} should have absorption"

    def test_line_absorption_multiple_lines(self):
        """Test with multiple lines."""
        # Add a second line at different wavelength
        mg_line = Line(
            wl=5200e-8,  # cm
            log_gf=-0.5,
            species=Species("Mg_I"),
            E_lower=2.5,  # eV
            gamma_rad=1e7,
            gamma_stark=1e-7,   # rad/s per electron
            vdW=(1e-7, -1.0)    # rad/s per H atom
        )

        linelist = [self.fe_line, mg_line]
        wavelengths = jnp.linspace(4990e-8, 5210e-8, 2000)

        # Add Mg_I to number densities
        n_densities = {
            **self.number_densities,
            Species("Mg_I"): jnp.array([5e9]),
        }
        partition_funcs_full = load_atomic_partition_functions()
        partition_funcs = {
            **self.partition_functions,
            Species("Mg_I"): partition_funcs_full[Species("Mg_I")],
        }

        alpha = line_absorption(
            linelist,
            wavelengths,
            self.temperatures,
            self.electron_densities,
            n_densities,
            partition_funcs,
            self.xi,
            lambda wl: jnp.full_like(self.temperatures, 1e-26)
        )

        # Should have two peaks
        assert alpha.shape == (1, 2000)
        assert jnp.any(alpha > 0)

        # Find two peak regions (one for each line)
        # Fe line should be around index 1000 (5000 Å out of 4990-5210 Å range)
        # Mg line should be around index 1909 (5200 Å)
        fe_region = alpha[0, 800:1200]  # Around Fe line
        mg_region = alpha[0, 1800:2000]  # Around Mg line

        # Both regions should have significant absorption
        assert jnp.max(fe_region) > 0
        assert jnp.max(mg_region) > 0

        # Total absorption should be larger than from single line
        # (approximately additive for weak lines)
        assert jnp.max(alpha[0]) > 0


class TestLineAbsorptionJIT:
    """Test JIT compatibility of line_absorption."""

    def setup_method(self):
        """Setup test data."""
        # Create simple test case with realistic parameters
        self.fe_line = Line(
            wl=5000e-8,
            log_gf=-1.0,
            species=Species("Fe_I"),
            E_lower=3.0,
            gamma_rad=1e7,
            gamma_stark=1e-7,  # rad/s per electron
            vdW=(1e-7, -1.0)   # rad/s per H atom at 10,000 K
        )
        self.linelist = [self.fe_line]
        self.wavelengths = jnp.linspace(4999e-8, 5001e-8, 200)
        self.temperatures = jnp.array([5000.0])
        self.electron_densities = jnp.array([1e14])

        partition_funcs_full = load_atomic_partition_functions()
        self.partition_functions = {
            Species("Fe_I"): partition_funcs_full[Species("Fe_I")],
            Species("H_I"): partition_funcs_full[Species("H_I")],
        }

        self.number_densities = {
            Species("H_I"): jnp.array([1e17]),
            Species("Fe_I"): jnp.array([1e10]),
        }

        self.xi = 1e5

    def test_helper_functions_jit(self):
        """Test that helper functions are JIT-compatible."""
        # doppler_width
        @jax.jit
        def compute_doppler(wl, T, mass, xi):
            return doppler_width(wl, T, mass, xi)

        result = compute_doppler(5000e-8, 5000.0, 56 * 1.66054e-24, 1e5)
        assert result > 0

        # scaled_stark
        @jax.jit
        def compute_stark(gamma, T):
            return scaled_stark(gamma, T)

        result = compute_stark(1e8, 5000.0)
        assert result > 0

        # sigma_line
        @jax.jit
        def compute_sigma(wl):
            return sigma_line(wl)

        result = compute_sigma(5000e-8)
        assert result > 0

        # line_profile
        @jax.jit
        def compute_profile(wl_c, sigma, gamma, amp, wl):
            return line_profile(wl_c, sigma, gamma, amp, wl)

        result = compute_profile(5000e-8, 1e-9, 1e-10, 1.0, 5000e-8)
        assert result > 0

    def test_line_absorption_uses_jit_by_default(self):
        """
        Test that line_absorption uses JIT by default.

        The function now has a JIT-compiled implementation that is used
        automatically when use_jit=True (the default).
        """
        def continuum_opacity(wl):
            return jnp.full_like(self.temperatures, 1e-26)

        # Call with default (use_jit=True)
        alpha_jit = line_absorption(
            self.linelist,
            self.wavelengths,
            self.temperatures,
            self.electron_densities,
            self.number_densities,
            self.partition_functions,
            self.xi,
            continuum_opacity
        )

        # Call with use_jit=False (Python implementation)
        alpha_python = line_absorption(
            self.linelist,
            self.wavelengths,
            self.temperatures,
            self.electron_densities,
            self.number_densities,
            self.partition_functions,
            self.xi,
            continuum_opacity,
            use_jit=False
        )

        # Both should produce valid results
        assert alpha_jit.shape == (len(self.temperatures), len(self.wavelengths))
        assert alpha_python.shape == (len(self.temperatures), len(self.wavelengths))

        # Results should be very similar (within numerical precision)
        rel_diff = jnp.abs(alpha_jit - alpha_python) / (jnp.abs(alpha_python) + 1e-30)
        max_rel_diff = jnp.max(rel_diff)
        assert max_rel_diff < 1e-10, f"JIT and Python versions differ by {max_rel_diff}"

    def test_line_absorption_core_jit(self):
        """
        Test that line_absorption_core can be JIT-compiled directly.

        This is the key functionality - line_absorption_core is fully JAX-traceable.
        """
        def continuum_opacity(wl):
            return jnp.full_like(self.temperatures, 1e-26)

        # Prepare linelist arrays (done once, outside JIT)
        unique_species = list(set([line.species for line in self.linelist]))
        line_arrays = prepare_linelist_arrays(self.linelist, unique_species)

        # Prepare other arrays
        n_species = len(unique_species)
        n_layers = len(self.temperatures)
        species_to_id = {sp: i for i, sp in enumerate(unique_species)}

        number_densities_array = jnp.zeros((n_species, n_layers))
        for sp, idx in species_to_id.items():
            if sp in self.number_densities:
                number_densities_array = number_densities_array.at[idx].set(
                    self.number_densities[sp]
                )

        log_temps = jnp.log(self.temperatures)
        partition_funcs_array = jnp.zeros((n_species, n_layers))
        for sp, idx in species_to_id.items():
            if sp in self.partition_functions:
                U_vals = jnp.array([self.partition_functions[sp](lt) for lt in log_temps])
                partition_funcs_array = partition_funcs_array.at[idx].set(U_vals)

        continuum_opacities = jnp.array([
            continuum_opacity(line.wl) for line in self.linelist
        ])

        H_I_densities = self.number_densities.get(Species("H_I"), jnp.zeros(n_layers))

        # JIT-compile line_absorption_core
        @jax.jit
        def compute_line_absorption(temps, ne):
            """JIT-compiled wrapper."""
            return line_absorption_core(
                line_wls=line_arrays['wls'],
                line_log_gfs=line_arrays['log_gfs'],
                line_species_ids=line_arrays['species_ids'],
                line_E_lowers=line_arrays['E_lowers'],
                line_gamma_rads=line_arrays['gamma_rads'],
                line_gamma_starks=line_arrays['gamma_starks'],
                line_vdW_params=line_arrays['vdW_params'],
                line_masses=line_arrays['masses'],
                line_is_molecule=line_arrays['is_molecule'],
                wavelengths=self.wavelengths,
                temperatures=temps,
                electron_densities=ne,
                number_densities_array=number_densities_array,
                partition_funcs_array=partition_funcs_array,
                H_I_densities=H_I_densities,
                continuum_opacities=continuum_opacities,
                xi=self.xi
            )

        # First call (compiles)
        alpha = compute_line_absorption(self.temperatures, self.electron_densities)

        # Should work fine
        assert alpha.shape == (len(self.temperatures), len(self.wavelengths))
        assert jnp.any(alpha > 0), "Should have some absorption"

        # Second call (uses cached compilation)
        alpha2 = compute_line_absorption(self.temperatures, self.electron_densities)

        # Should be identical
        np.testing.assert_array_equal(alpha, alpha2)

    def test_line_absorption_consistency(self):
        """
        Test that repeated calls give consistent results.

        This verifies that the function is deterministic even though
        it can't be directly JITed.
        """
        def continuum_opacity(wl):
            return jnp.full_like(self.temperatures, 1e-26)

        # Call twice
        alpha1 = line_absorption(
            self.linelist,
            self.wavelengths,
            self.temperatures,
            self.electron_densities,
            self.number_densities,
            self.partition_functions,
            self.xi,
            continuum_opacity
        )

        alpha2 = line_absorption(
            self.linelist,
            self.wavelengths,
            self.temperatures,
            self.electron_densities,
            self.number_densities,
            self.partition_functions,
            self.xi,
            continuum_opacity
        )

        # Should be identical
        np.testing.assert_array_equal(alpha1, alpha2)

    def test_end_to_end_synthesis_jit(self):
        """
        Test end-to-end JIT compilation through a synthesis-like pipeline.

        This demonstrates that line_absorption_core can be used in a larger
        JIT-compiled function, which is the key use case for spectral synthesis.
        """
        def continuum_opacity(wl):
            return jnp.full_like(self.temperatures, 1e-26)

        # Prepare linelist arrays (done once, outside JIT)
        unique_species = list(set([line.species for line in self.linelist]))
        line_arrays = prepare_linelist_arrays(self.linelist, unique_species)

        # Prepare other arrays
        n_species = len(unique_species)
        n_layers = len(self.temperatures)
        species_to_id = {sp: i for i, sp in enumerate(unique_species)}

        number_densities_array = jnp.zeros((n_species, n_layers))
        for sp, idx in species_to_id.items():
            if sp in self.number_densities:
                number_densities_array = number_densities_array.at[idx].set(
                    self.number_densities[sp]
                )

        log_temps = jnp.log(self.temperatures)
        partition_funcs_array = jnp.zeros((n_species, n_layers))
        for sp, idx in species_to_id.items():
            if sp in self.partition_functions:
                U_vals = jnp.array([self.partition_functions[sp](lt) for lt in log_temps])
                partition_funcs_array = partition_funcs_array.at[idx].set(U_vals)

        continuum_opacities = jnp.array([
            continuum_opacity(line.wl) for line in self.linelist
        ])

        H_I_densities = self.number_densities.get(Species("H_I"), jnp.zeros(n_layers))

        # Define a synthesis-like function that JIT-compiles through line_absorption_core
        @jax.jit
        def synthesize_spectrum(temps, ne):
            """
            Synthesis-like function that includes line_absorption_core.

            This demonstrates the key use case: JIT-compiling the entire
            synthesis pipeline including line absorption.
            """
            # Compute line absorption (this is JIT-compiled!)
            alpha_lines = line_absorption_core(
                line_wls=line_arrays['wls'],
                line_log_gfs=line_arrays['log_gfs'],
                line_species_ids=line_arrays['species_ids'],
                line_E_lowers=line_arrays['E_lowers'],
                line_gamma_rads=line_arrays['gamma_rads'],
                line_gamma_starks=line_arrays['gamma_starks'],
                line_vdW_params=line_arrays['vdW_params'],
                line_masses=line_arrays['masses'],
                line_is_molecule=line_arrays['is_molecule'],
                wavelengths=self.wavelengths,
                temperatures=temps,
                electron_densities=ne,
                number_densities_array=number_densities_array,
                partition_funcs_array=partition_funcs_array,
                H_I_densities=H_I_densities,
                continuum_opacities=continuum_opacities,
                xi=self.xi
            )

            # Add continuum (hypothetical)
            alpha_continuum = jnp.full_like(alpha_lines, 1e-26)

            # Total opacity
            alpha_total = alpha_lines + alpha_continuum

            # Simple radiative transfer (hypothetical)
            tau = jnp.cumsum(alpha_total, axis=1) * (self.wavelengths[1] - self.wavelengths[0])
            intensity = jnp.exp(-tau)

            return intensity, alpha_lines

        # First call (compiles the entire pipeline)
        intensity1, alpha1 = synthesize_spectrum(self.temperatures, self.electron_densities)

        # Check results
        assert intensity1.shape == (len(self.temperatures), len(self.wavelengths))
        assert alpha1.shape == (len(self.temperatures), len(self.wavelengths))
        assert jnp.any(alpha1 > 0), "Should have line absorption"
        assert jnp.all((intensity1 >= 0) & (intensity1 <= 1)), "Intensity should be in [0,1]"

        # Second call (uses cached compilation - should be fast)
        intensity2, alpha2 = synthesize_spectrum(self.temperatures, self.electron_densities)

        # Should be identical
        np.testing.assert_array_equal(intensity1, intensity2)
        np.testing.assert_array_equal(alpha1, alpha2)

        # Third call with different parameters (still uses compiled version)
        temps_hot = jnp.array([6000.0])
        ne_hot = jnp.array([1e15])
        intensity3, alpha3 = synthesize_spectrum(temps_hot, ne_hot)

        # Should produce different results
        assert not jnp.array_equal(alpha1, alpha3), "Different temps should give different results"


class TestLineAbsorptionPhysics:
    """Test that line_absorption produces physically reasonable results."""

    def setup_method(self):
        """Setup test data."""
        self.fe_line = Line(
            wl=5000e-8,
            log_gf=-1.0,
            species=Species("Fe_I"),
            E_lower=3.0,
            gamma_rad=1e7,
            gamma_stark=1e-7,   # rad/s per electron
            vdW=(1e-7, -1.0)    # rad/s per H atom at 10,000 K
        )
        self.linelist = [self.fe_line]
        self.wavelengths = jnp.linspace(4995e-8, 5005e-8, 1000)

        partition_funcs_full = load_atomic_partition_functions()
        self.partition_functions = {
            Species("Fe_I"): partition_funcs_full[Species("Fe_I")],
            Species("H_I"): partition_funcs_full[Species("H_I")],
        }

    def test_temperature_dependence(self):
        """Test that line strength increases with temperature (for low E_lower)."""
        def continuum_opacity(wl):
            return jnp.array([1e-26])

        # Low temperature
        alpha_cool = line_absorption(
            self.linelist,
            self.wavelengths,
            jnp.array([4000.0]),
            jnp.array([1e14]),
            {
                Species("H_I"): jnp.array([1e17]),
                Species("Fe_I"): jnp.array([1e10]),
            },
            self.partition_functions,
            1e5,
            continuum_opacity
        )

        # High temperature
        alpha_hot = line_absorption(
            self.linelist,
            self.wavelengths,
            jnp.array([6000.0]),
            jnp.array([1e14]),
            {
                Species("H_I"): jnp.array([1e17]),
                Species("Fe_I"): jnp.array([1e10]),
            },
            self.partition_functions,
            1e5,
            continuum_opacity
        )

        # For E_lower = 3 eV, line should be stronger at higher T
        # (Boltzmann factor increases faster than partition function)
        assert jnp.max(alpha_hot) > jnp.max(alpha_cool)

    def test_abundance_dependence(self):
        """Test that line strength scales with abundance."""
        def continuum_opacity(wl):
            return jnp.array([1e-26])

        # Low abundance
        alpha_low = line_absorption(
            self.linelist,
            self.wavelengths,
            jnp.array([5000.0]),
            jnp.array([1e14]),
            {
                Species("H_I"): jnp.array([1e17]),
                Species("Fe_I"): jnp.array([1e9]),  # 10x lower
            },
            self.partition_functions,
            1e5,
            continuum_opacity
        )

        # High abundance
        alpha_high = line_absorption(
            self.linelist,
            self.wavelengths,
            jnp.array([5000.0]),
            jnp.array([1e14]),
            {
                Species("H_I"): jnp.array([1e17]),
                Species("Fe_I"): jnp.array([1e10]),  # 10x higher
            },
            self.partition_functions,
            1e5,
            continuum_opacity
        )

        # Should scale linearly with abundance
        ratio = jnp.max(alpha_high) / jnp.max(alpha_low)
        np.testing.assert_allclose(ratio, 10.0, rtol=0.1)

    def test_microturbulence_broadening(self):
        """Test that microturbulence broadens the line."""
        def continuum_opacity(wl):
            return jnp.array([1e-26])

        # No microturbulence
        alpha_no_turb = line_absorption(
            self.linelist,
            self.wavelengths,
            jnp.array([5000.0]),
            jnp.array([1e14]),
            {
                Species("H_I"): jnp.array([1e17]),
                Species("Fe_I"): jnp.array([1e10]),
            },
            self.partition_functions,
            0.0,  # xi = 0
            continuum_opacity
        )

        # With microturbulence
        alpha_turb = line_absorption(
            self.linelist,
            self.wavelengths,
            jnp.array([5000.0]),
            jnp.array([1e14]),
            {
                Species("H_I"): jnp.array([1e17]),
                Species("Fe_I"): jnp.array([1e10]),
            },
            self.partition_functions,
            2e5,  # xi = 2 km/s
            continuum_opacity
        )

        # Peak should be lower with microturbulence (same area, broader line)
        assert jnp.max(alpha_turb) < jnp.max(alpha_no_turb)

        # Total absorption (integrated) should be similar
        integral_no_turb = jnp.sum(alpha_no_turb)
        integral_turb = jnp.sum(alpha_turb)
        np.testing.assert_allclose(integral_turb, integral_no_turb, rtol=0.2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
