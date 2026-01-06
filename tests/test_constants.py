"""Tests for physical constants - verify they match Julia values."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import korg
from korg import constants
from korg import atomic_data


class TestPhysicalConstants:
    """Test physical constants match expected values."""

    def test_speed_of_light(self):
        """Speed of light in cm/s."""
        assert np.isclose(constants.c_cgs, 2.99792458e10, rtol=1e-10)

    def test_planck_constant_cgs(self):
        """Planck constant in erg*s."""
        assert np.isclose(constants.hplanck_cgs, 6.62607015e-27, rtol=1e-10)

    def test_planck_constant_eV(self):
        """Planck constant in eV*s."""
        assert np.isclose(constants.hplanck_eV, 4.135667696e-15, rtol=1e-10)

    def test_boltzmann_constant_cgs(self):
        """Boltzmann constant in erg/K."""
        assert np.isclose(constants.kboltz_cgs, 1.380649e-16, rtol=1e-10)

    def test_boltzmann_constant_eV(self):
        """Boltzmann constant in eV/K."""
        assert np.isclose(constants.kboltz_eV, 8.617333262e-5, rtol=1e-10)

    def test_electron_mass(self):
        """Electron mass in g."""
        assert np.isclose(constants.electron_mass_cgs, 9.1093837015e-28, rtol=1e-10)

    def test_electron_charge(self):
        """Electron charge in esu."""
        assert np.isclose(constants.electron_charge_cgs, 4.80320425e-10, rtol=1e-10)

    def test_amu(self):
        """Atomic mass unit in g."""
        assert np.isclose(constants.amu_cgs, 1.6605390666e-24, rtol=1e-10)

    def test_rydberg_energy(self):
        """Rydberg energy in eV."""
        assert np.isclose(constants.Rydberg_eV, 13.605693122994, rtol=1e-10)

    def test_max_atomic_number(self):
        """Maximum supported atomic number."""
        assert constants.MAX_ATOMIC_NUMBER == 92


class TestAtomicData:
    """Test atomic data."""

    def test_atomic_symbols_length(self):
        """Should have 92 elements (H through U)."""
        assert len(atomic_data.atomic_symbols) == 92

    def test_atomic_symbols_first(self):
        """First element is hydrogen."""
        assert atomic_data.atomic_symbols[0] == "H"

    def test_atomic_symbols_last(self):
        """Last element is uranium."""
        assert atomic_data.atomic_symbols[91] == "U"

    def test_atomic_numbers_mapping(self):
        """Symbol to Z mapping."""
        assert atomic_data.atomic_numbers["H"] == 1
        assert atomic_data.atomic_numbers["He"] == 2
        assert atomic_data.atomic_numbers["Fe"] == 26
        assert atomic_data.atomic_numbers["U"] == 92

    def test_atomic_masses_shape(self):
        """Should have 92 masses."""
        assert len(atomic_data.atomic_masses) == 92

    def test_atomic_masses_hydrogen(self):
        """Hydrogen mass ~ 1 amu."""
        expected = 1.008 * constants.amu_cgs
        assert np.isclose(atomic_data.atomic_masses[0], expected, rtol=1e-6)

    def test_ionization_energies_hydrogen(self):
        """Hydrogen first ionization energy."""
        if not hasattr(atomic_data, 'ionization_energies'):
            pytest.skip("ionization_energies not yet implemented in atomic_data")
        chi_I, chi_II, chi_III = atomic_data.ionization_energies[1]
        assert np.isclose(chi_I, 13.5984, rtol=1e-6)
        assert chi_II == -1.0  # Not available

    def test_ionization_energies_iron(self):
        """Iron ionization energies."""
        if not hasattr(atomic_data, 'ionization_energies'):
            pytest.skip("ionization_energies not yet implemented in atomic_data")
        chi_I, chi_II, chi_III = atomic_data.ionization_energies[26]
        assert np.isclose(chi_I, 7.9025, rtol=1e-4)
        assert np.isclose(chi_II, 16.199, rtol=1e-4)


class TestSolarAbundances:
    """Test solar abundance arrays."""

    def test_abundances_shape(self):
        """All abundance arrays should have 92 elements."""
        assert len(atomic_data.asplund_2009_solar_abundances) == 92
        assert len(atomic_data.asplund_2020_solar_abundances) == 92
        assert len(atomic_data.grevesse_2007_solar_abundances) == 92
        assert len(atomic_data.bergemann_2025_solar_abundances) == 92

    def test_hydrogen_abundance(self):
        """Hydrogen is always 12.0 by definition."""
        assert atomic_data.asplund_2009_solar_abundances[0] == 12.0
        assert atomic_data.asplund_2020_solar_abundances[0] == 12.0
        assert atomic_data.grevesse_2007_solar_abundances[0] == 12.0
        assert atomic_data.bergemann_2025_solar_abundances[0] == 12.0

    def test_iron_abundance_range(self):
        """Iron abundance should be around 7.5."""
        fe_idx = 25  # Z=26, index 25
        for abund in [
            atomic_data.asplund_2009_solar_abundances,
            atomic_data.asplund_2020_solar_abundances,
            atomic_data.grevesse_2007_solar_abundances,
            atomic_data.bergemann_2025_solar_abundances,
        ]:
            assert 7.4 < abund[fe_idx] < 7.6

    def test_default_is_bergemann(self):
        """Default solar abundances should be Bergemann 2025."""
        assert jnp.allclose(
            atomic_data.default_solar_abundances,
            atomic_data.bergemann_2025_solar_abundances
        )


class TestAlphaElements:
    """Test alpha element definitions."""

    def test_alpha_elements_count(self):
        """Should have 8 alpha elements."""
        if not hasattr(atomic_data, 'default_alpha_elements'):
            # Check if it's in abundances module instead
            from korg.abundances import DEFAULT_ALPHA_ELEMENTS
            assert len(DEFAULT_ALPHA_ELEMENTS) == 8
        else:
            assert len(atomic_data.default_alpha_elements) == 8

    def test_alpha_elements_values(self):
        """Alpha elements are O, Ne, Mg, Si, S, Ar, Ca, Ti."""
        expected = [8, 10, 12, 14, 16, 18, 20, 22]
        if not hasattr(atomic_data, 'default_alpha_elements'):
            # Check if it's in abundances module instead
            from korg.abundances import DEFAULT_ALPHA_ELEMENTS
            assert DEFAULT_ALPHA_ELEMENTS == expected
        else:
            assert list(atomic_data.default_alpha_elements) == expected


class TestJITCompatibility:
    """Test that constants work with JAX JIT."""

    def test_constants_in_jit(self):
        """Constants should be usable in JIT-compiled functions."""
        @jax.jit
        def compute_energy(wavelength_cm):
            return constants.hplanck_cgs * constants.c_cgs / wavelength_cm

        # 5000 Angstroms = 5e-5 cm
        energy = compute_energy(5e-5)
        expected = 6.62607015e-27 * 2.99792458e10 / 5e-5
        assert np.isclose(float(energy), expected, rtol=1e-10)

    def test_abundances_in_jit(self):
        """Abundance arrays should work in JIT-compiled functions."""
        @jax.jit
        def get_fe_abundance(abundances):
            return abundances[25]  # Iron

        fe_abund = get_fe_abundance(atomic_data.default_solar_abundances)
        assert np.isclose(float(fe_abund), 7.51, rtol=1e-3)
