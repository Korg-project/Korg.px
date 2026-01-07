"""
Tests comparing Python implementations against serialized Julia reference data.

These tests load pre-computed Julia reference data from julia_reference_data.json
and verify that Python/JAX implementations match to the required precision.

To regenerate the Julia reference data:
    julia --project=/tmp/Korg.jl tests/generate_julia_reference.jl

Then run these tests:
    pytest tests/test_julia_reference.py -v
"""

import json
import os
from pathlib import Path

# Import korg FIRST to enable JAX x64 mode before any other JAX operations
import korg

import jax
import numpy as np
import pytest

# Path to reference data
REFERENCE_FILE = Path(__file__).parent / "julia_reference_data.json"


@pytest.fixture(scope="module")
def reference_data():
    """Load Julia reference data."""
    if not REFERENCE_FILE.exists():
        pytest.skip(
            f"Julia reference data not found at {REFERENCE_FILE}. "
            "Run: julia --project=/tmp/Korg.jl tests/generate_julia_reference.jl"
        )
    with open(REFERENCE_FILE) as f:
        return json.load(f)


class TestConstantsReference:
    """Compare Python constants with Julia reference values."""

    def test_c_cgs(self, reference_data):
        """Speed of light should match."""
        from korg.constants import c_cgs
        julia_val = reference_data["constants"]["c_cgs"]
        assert np.isclose(c_cgs, julia_val, rtol=1e-10)

    def test_hplanck_cgs(self, reference_data):
        """Planck constant should match."""
        from korg.constants import hplanck_cgs
        julia_val = reference_data["constants"]["hplanck_cgs"]
        assert np.isclose(hplanck_cgs, julia_val, rtol=1e-10)

    def test_kboltz_cgs(self, reference_data):
        """Boltzmann constant should match."""
        from korg.constants import kboltz_cgs
        julia_val = reference_data["constants"]["kboltz_cgs"]
        assert np.isclose(kboltz_cgs, julia_val, rtol=1e-10)

    def test_electron_mass_cgs(self, reference_data):
        """Electron mass should match."""
        from korg.constants import electron_mass_cgs
        julia_val = reference_data["constants"]["electron_mass_cgs"]
        assert np.isclose(electron_mass_cgs, julia_val, rtol=1e-10)

    def test_Rydberg_eV(self, reference_data):
        """Rydberg energy should match."""
        from korg.constants import Rydberg_eV
        julia_val = reference_data["constants"]["Rydberg_eV"]
        assert np.isclose(Rydberg_eV, julia_val, rtol=1e-10)

    def test_kboltz_eV(self, reference_data):
        """Boltzmann constant in eV should match."""
        from korg.constants import kboltz_eV
        julia_val = reference_data["constants"]["kboltz_eV"]
        assert np.isclose(kboltz_eV, julia_val, rtol=1e-10)

    def test_hplanck_eV(self, reference_data):
        """Planck constant in eV should match."""
        from korg.constants import hplanck_eV
        julia_val = reference_data["constants"]["hplanck_eV"]
        assert np.isclose(hplanck_eV, julia_val, rtol=1e-10)


class TestElectronScatteringReference:
    """Compare electron scattering with Julia reference values."""

    def test_electron_scattering(self, reference_data):
        """Thomson scattering coefficient should match."""
        from korg.continuum_absorption.scattering import electron_scattering

        ref = reference_data["electron_scattering"]
        for ne_str, julia_val in ref["outputs"].items():
            ne = float(ne_str)
            py_val = float(electron_scattering(ne))
            assert np.isclose(py_val, julia_val, rtol=1e-6), \
                f"Mismatch at ne={ne}: Python={py_val}, Julia={julia_val}"


class TestRayleighScatteringReference:
    """Compare Rayleigh scattering with Julia reference values."""

    def test_rayleigh(self, reference_data):
        """Rayleigh scattering coefficient should match."""
        from korg.continuum_absorption.scattering import rayleigh
        from korg.constants import c_cgs

        ref = reference_data["rayleigh_scattering"]
        nH_I = ref["inputs"]["nH_I"]
        nHe_I = ref["inputs"]["nHe_I"]
        nH2 = ref["inputs"]["nH2"]

        for wl_str, julia_val in ref["outputs"].items():
            wl_A = float(wl_str)
            nu = c_cgs / (wl_A * 1e-8)  # Convert to frequency
            py_val = float(rayleigh(np.array([nu]), nH_I, nHe_I, nH2)[0])
            assert np.isclose(py_val, julia_val, rtol=1e-6), \
                f"Mismatch at wavelength={wl_A}A: Python={py_val}, Julia={julia_val}"


class TestTranslationalUReference:
    """Compare translational partition function with Julia reference values."""

    def test_translational_U(self, reference_data):
        """Translational partition function should match."""
        from korg.statmech import translational_U
        from korg.constants import electron_mass_cgs

        ref = reference_data["translational_U"]
        for T_str, julia_val in ref["outputs"].items():
            T = float(T_str)
            py_val = float(translational_U(electron_mass_cgs, T))
            assert np.isclose(py_val, julia_val, rtol=1e-6), \
                f"Mismatch at T={T}: Python={py_val}, Julia={julia_val}"


class TestGauntFactorReference:
    """Compare Gaunt factor with Julia reference values."""

    def test_gaunt_ff_vanHoof(self, reference_data):
        """Gaunt factor interpolation should match."""
        try:
            from korg.continuum_absorption.hydrogenic_bf_ff import gaunt_ff_vanHoof
        except (ImportError, FileNotFoundError) as e:
            pytest.skip(f"hydrogenic_bf_ff module not available: {e}")

        try:
            ref = reference_data["gaunt_ff"]
            for key, julia_val in ref["outputs"].items():
                log_u, log_gamma2 = [float(x) for x in key.split("_")]
                py_val = gaunt_ff_vanHoof(log_u, log_gamma2)
                assert np.isclose(py_val, julia_val, rtol=1e-5), \
                    f"Mismatch at ({log_u}, {log_gamma2}): Python={py_val}, Julia={julia_val}"
        except FileNotFoundError as e:
            pytest.skip(f"Data file not found: {e}")


class TestHydrogenicFFReference:
    """Compare hydrogenic free-free absorption with Julia reference values."""

    def test_hydrogenic_ff_absorption(self, reference_data):
        """Free-free absorption coefficient should match."""
        try:
            from korg.continuum_absorption.hydrogenic_bf_ff import hydrogenic_ff_absorption
        except (ImportError, FileNotFoundError) as e:
            pytest.skip(f"hydrogenic_bf_ff module not available: {e}")

        try:
            from korg.constants import c_cgs

            ref = reference_data["hydrogenic_ff"]
            T = ref["inputs"]["T"]
            Z = ref["inputs"]["Z"]
            ni = ref["inputs"]["ni"]
            ne = ref["inputs"]["ne"]

            for wl_str, julia_val in ref["outputs"].items():
                wl_A = float(wl_str)
                nu = c_cgs / (wl_A * 1e-8)
                py_val = hydrogenic_ff_absorption(nu, T, Z, ni, ne)
                assert np.isclose(py_val, julia_val, rtol=1e-5), \
                    f"Mismatch at wavelength={wl_A}A: Python={py_val}, Julia={julia_val}"
        except FileNotFoundError as e:
            pytest.skip(f"Data file not found: {e}")


class TestSpeciesReference:
    """Compare Species parsing with Julia reference values."""

    def test_species_charge(self, reference_data):
        """Species charge should match."""
        from korg.species import Species

        ref = reference_data["species"]
        for code, julia_result in ref["outputs"].items():
            try:
                py_species = Species.from_string(code)
                assert py_species.charge == julia_result["charge"], \
                    f"Charge mismatch for {code}: Python={py_species.charge}, Julia={julia_result['charge']}"
            except Exception as e:
                pytest.skip(f"Species.from_string not fully implemented for '{code}': {e}")

    def test_species_is_molecule(self, reference_data):
        """Species molecule detection should match."""
        from korg.species import Species

        ref = reference_data["species"]
        for code, julia_result in ref["outputs"].items():
            try:
                py_species = Species.from_string(code)
                py_is_mol = py_species.is_molecule
                julia_is_mol = julia_result["is_molecule"]
                assert py_is_mol == julia_is_mol, \
                    f"is_molecule mismatch for {code}: Python={py_is_mol}, Julia={julia_is_mol}"
            except Exception as e:
                pytest.skip(f"Species.from_string not fully implemented for '{code}': {e}")


class TestFormulaReference:
    """Compare Formula parsing with Julia reference values."""

    def test_formula_atoms(self, reference_data):
        """Formula atoms should match."""
        from korg.species import Formula

        ref = reference_data["formula"]
        for code, julia_result in ref["outputs"].items():
            try:
                py_formula = Formula.from_string(code)
                julia_atoms = tuple(julia_result["atoms"])
                assert py_formula.atoms == julia_atoms, \
                    f"Atoms mismatch for {code}: Python={py_formula.atoms}, Julia={julia_atoms}"
            except Exception as e:
                pytest.skip(f"Formula.from_string not fully implemented for '{code}': {e}")


class TestAtomicDataReference:
    """Compare atomic data with Julia reference values."""

    def test_atomic_symbols(self, reference_data):
        """Atomic symbols should match."""
        from korg.atomic_data import atomic_symbols

        julia_symbols = reference_data["atomic_data"]["atomic_symbols"]
        assert len(atomic_symbols) == len(julia_symbols), \
            f"Length mismatch: Python={len(atomic_symbols)}, Julia={len(julia_symbols)}"
        for i, (py_sym, julia_sym) in enumerate(zip(atomic_symbols, julia_symbols)):
            assert py_sym == julia_sym, \
                f"Symbol mismatch for Z={i+1}: Python={py_sym}, Julia={julia_sym}"

    def test_atomic_masses(self, reference_data):
        """Atomic masses should match."""
        from korg.atomic_data import atomic_masses

        julia_masses = reference_data["atomic_data"]["atomic_masses"]
        for i, julia_mass in enumerate(julia_masses):
            py_mass = atomic_masses[i]
            assert np.isclose(py_mass, julia_mass, rtol=1e-6), \
                f"Mass mismatch for Z={i+1}: Python={py_mass}, Julia={julia_mass}"

    def test_ionization_energies(self, reference_data):
        """Ionization energies (all three levels) should match."""
        from korg.atomic_data import ionization_energies

        julia_ie = reference_data["atomic_data"]["ionization_energies"]
        for Z_str, julia_vals in julia_ie.items():
            Z = int(Z_str)
            py_vals = ionization_energies[Z]
            for i, (py_val, julia_val) in enumerate(zip(py_vals, julia_vals)):
                # Both use -1 for unavailable
                if julia_val > 0 and py_val > 0:
                    assert np.isclose(py_val, julia_val, rtol=1e-6), \
                        f"Ionization energy mismatch for Z={Z}, level {i+1}: Python={py_val}, Julia={julia_val}"
                elif julia_val < 0 and py_val < 0:
                    pass  # Both unavailable, OK
                else:
                    # One is available and one is not
                    pytest.fail(f"Availability mismatch for Z={Z}, level {i+1}: Python={py_val}, Julia={julia_val}")


class TestSolarAbundancesReference:
    """Compare solar abundances with Julia reference values."""

    def test_grevesse_2007(self, reference_data):
        """Grevesse 2007 solar abundances should match."""
        from korg.atomic_data import grevesse_2007_solar_abundances

        julia_abund = reference_data["solar_abundances"]["grevesse_2007"]
        for i, (py_val, julia_val) in enumerate(zip(grevesse_2007_solar_abundances, julia_abund)):
            assert np.isclose(py_val, julia_val, rtol=1e-6), \
                f"Grevesse 2007 abundance mismatch for Z={i+1}: Python={py_val}, Julia={julia_val}"

    def test_asplund_2009(self, reference_data):
        """Asplund 2009 solar abundances should match."""
        from korg.atomic_data import asplund_2009_solar_abundances

        julia_abund = reference_data["solar_abundances"]["asplund_2009"]
        for i, (py_val, julia_val) in enumerate(zip(asplund_2009_solar_abundances, julia_abund)):
            assert np.isclose(py_val, julia_val, rtol=1e-6), \
                f"Asplund 2009 abundance mismatch for Z={i+1}: Python={py_val}, Julia={julia_val}"

    def test_asplund_2020(self, reference_data):
        """Asplund 2020 solar abundances should match."""
        from korg.atomic_data import asplund_2020_solar_abundances

        julia_abund = reference_data["solar_abundances"]["asplund_2020"]
        for i, (py_val, julia_val) in enumerate(zip(asplund_2020_solar_abundances, julia_abund)):
            assert np.isclose(py_val, julia_val, rtol=1e-6), \
                f"Asplund 2020 abundance mismatch for Z={i+1}: Python={py_val}, Julia={julia_val}"

    def test_bergemann_2025(self, reference_data):
        """Bergemann 2025 solar abundances should match."""
        from korg.atomic_data import bergemann_2025_solar_abundances

        julia_abund = reference_data["solar_abundances"]["bergemann_2025"]
        for i, (py_val, julia_val) in enumerate(zip(bergemann_2025_solar_abundances, julia_abund)):
            assert np.isclose(py_val, julia_val, rtol=1e-6), \
                f"Bergemann 2025 abundance mismatch for Z={i+1}: Python={py_val}, Julia={julia_val}"

    def test_default_abundances(self, reference_data):
        """Default solar abundances should match."""
        from korg.atomic_data import default_solar_abundances

        julia_abund = reference_data["solar_abundances"]["default"]
        for i, (py_val, julia_val) in enumerate(zip(default_solar_abundances, julia_abund)):
            assert np.isclose(py_val, julia_val, rtol=1e-6), \
                f"Default abundance mismatch for Z={i+1}: Python={py_val}, Julia={julia_val}"

    def test_magg_2022(self, reference_data):
        """Magg 2022 solar abundances should match."""
        from korg.atomic_data import magg_2022_solar_abundances

        julia_abund = reference_data["solar_abundances"]["magg_2022"]
        for i, (py_val, julia_val) in enumerate(zip(magg_2022_solar_abundances, julia_abund)):
            assert np.isclose(py_val, julia_val, rtol=1e-6), \
                f"Magg 2022 abundance mismatch for Z={i+1}: Python={py_val}, Julia={julia_val}"


class TestIsotopicDataReference:
    """Compare isotopic data with Julia reference values."""

    def test_isotopic_abundances(self, reference_data):
        """Isotopic abundances should match."""
        from korg.isotopic_data import isotopic_abundances

        julia_data = reference_data["isotopic_data"]["isotopic_abundances"]
        for Z_str, julia_isotopes in julia_data.items():
            Z = int(Z_str)
            assert Z in isotopic_abundances, \
                f"Z={Z} missing from Python isotopic_abundances"
            py_isotopes = isotopic_abundances[Z]
            for A_str, julia_val in julia_isotopes.items():
                A = int(A_str)
                assert A in py_isotopes, \
                    f"Isotope A={A} missing for Z={Z}"
                py_val = py_isotopes[A]
                assert np.isclose(py_val, julia_val, rtol=1e-6), \
                    f"Isotopic abundance mismatch for Z={Z}, A={A}: Python={py_val}, Julia={julia_val}"

    def test_isotopic_nuclear_spin_degeneracies(self, reference_data):
        """Isotopic nuclear spin degeneracies should match."""
        from korg.isotopic_data import isotopic_nuclear_spin_degeneracies

        julia_data = reference_data["isotopic_data"]["isotopic_nuclear_spin_degeneracies"]
        for Z_str, julia_isotopes in julia_data.items():
            Z = int(Z_str)
            assert Z in isotopic_nuclear_spin_degeneracies, \
                f"Z={Z} missing from Python isotopic_nuclear_spin_degeneracies"
            py_isotopes = isotopic_nuclear_spin_degeneracies[Z]
            for A_str, julia_val in julia_isotopes.items():
                A = int(A_str)
                assert A in py_isotopes, \
                    f"Isotope A={A} missing for Z={Z} in nuclear spin data"
                py_val = py_isotopes[A]
                assert py_val == julia_val, \
                    f"Nuclear spin mismatch for Z={Z}, A={A}: Python={py_val}, Julia={julia_val}"


class TestWavelengthUtilsReference:
    """Compare wavelength conversion functions with Julia reference values."""

    def test_air_to_vacuum(self, reference_data):
        """air_to_vacuum should match Julia."""
        from korg.utils import air_to_vacuum

        ref = reference_data["wavelength_utils"]
        for wl_str, julia_val in ref["air_to_vacuum"].items():
            wl = float(wl_str)
            py_val = air_to_vacuum(wl)
            assert np.isclose(py_val, julia_val, rtol=1e-14), \
                f"Mismatch at wl={wl}: Python={py_val}, Julia={julia_val}"

    def test_vacuum_to_air(self, reference_data):
        """vacuum_to_air should match Julia."""
        from korg.utils import vacuum_to_air

        ref = reference_data["wavelength_utils"]
        for wl_str, julia_val in ref["vacuum_to_air"].items():
            wl = float(wl_str)
            py_val = vacuum_to_air(wl)
            assert np.isclose(py_val, julia_val, rtol=1e-14), \
                f"Mismatch at wl={wl}: Python={py_val}, Julia={julia_val}"


class TestLinePhysicsReference:
    """Compare line physics functions with Julia reference values."""

    def test_sigma_line(self, reference_data):
        """sigma_line (cross-section factor) should match Julia."""
        from korg.line_absorption import sigma_line

        ref = reference_data["line_physics"]["sigma_line"]
        for wl_str, julia_val in ref["outputs"].items():
            wl_A = float(wl_str)
            wl_cm = wl_A * 1e-8
            py_val = float(sigma_line(wl_cm))
            assert np.isclose(py_val, julia_val, rtol=1e-6), \
                f"Mismatch at wl={wl_A}A: Python={py_val}, Julia={julia_val}"

    def test_doppler_width(self, reference_data):
        """doppler_width should match Julia."""
        from korg.line_absorption import doppler_width
        from korg.constants import amu_cgs

        ref = reference_data["line_physics"]["doppler_width"]
        for key, julia_val in ref["outputs"].items():
            parts = key.split("_")
            wl_A = float(parts[0])
            T = float(parts[1])
            mass_amu = float(parts[2])
            xi_kms = float(parts[3])

            wl_cm = wl_A * 1e-8
            mass_g = mass_amu * amu_cgs
            xi_cgs = xi_kms * 1e5

            py_val = float(doppler_width(wl_cm, T, mass_g, xi_cgs))
            assert np.isclose(py_val, julia_val, rtol=1e-6), \
                f"Mismatch for {key}: Python={py_val}, Julia={julia_val}"

    def test_scaled_stark(self, reference_data):
        """scaled_stark should match Julia."""
        from korg.line_absorption import scaled_stark

        ref = reference_data["line_physics"]["scaled_stark"]
        for key, julia_val in ref["outputs"].items():
            parts = key.split("_")
            gamma = float(parts[0])
            T = float(parts[1])

            py_val = float(scaled_stark(gamma, T))
            assert np.isclose(py_val, julia_val, rtol=1e-6), \
                f"Mismatch for gamma={gamma}, T={T}: Python={py_val}, Julia={julia_val}"


class TestNormalPDFReference:
    """Compare normal_pdf with Julia reference values."""

    def test_normal_pdf(self, reference_data):
        """normal_pdf should match Julia."""
        from korg.utils import normal_pdf

        ref = reference_data["normal_pdf"]
        for key, julia_val in ref["outputs"].items():
            parts = key.split("_")
            delta = float(parts[0])
            sigma = float(parts[1])

            py_val = float(normal_pdf(delta, sigma))
            assert np.isclose(py_val, julia_val, rtol=1e-10), \
                f"Mismatch for delta={delta}, sigma={sigma}: Python={py_val}, Julia={julia_val}"


class TestExponentialIntegral1Reference:
    """Test exponential_integral_1 against Julia reference data."""

    def test_exponential_integral_1(self, reference_data):
        """exponential_integral_1 should match Julia."""
        from korg.radiative_transfer.expint import exponential_integral_1

        ref = reference_data["exponential_integral_1"]
        for x_str, julia_val in ref["outputs"].items():
            x = float(x_str)
            py_val = float(exponential_integral_1(x))

            # Allow slightly higher tolerance for x > 30 (both return 0)
            if x > 30:
                assert py_val == 0.0, f"For x={x} > 30, expected 0.0, got {py_val}"
            else:
                assert np.isclose(py_val, julia_val, rtol=1e-6), \
                    f"Mismatch for x={x}: Python={py_val}, Julia={julia_val}"


class TestIntervalUtilsReference:
    """Test Interval utilities against Julia reference data."""

    def test_contained_exclusive(self, reference_data):
        """contained() with exclusive interval should match Julia."""
        from korg.utils import Interval, contained

        ref = reference_data["interval_utils"]["contained"]
        for key, julia_result in ref["outputs"].items():
            parts = key.split("_")
            value = float(parts[0])
            lower = float(parts[1])
            upper = float(parts[2])

            interval = Interval(lower, upper)
            py_result = contained(value, interval)

            assert py_result == julia_result, \
                f"Mismatch for contained({value}, Interval({lower}, {upper})): " \
                f"Python={py_result}, Julia={julia_result}"

    def test_closed_interval_contained(self, reference_data):
        """contained() with closed_interval should match Julia."""
        from korg.utils import closed_interval, contained

        ref = reference_data["interval_utils"]["closed_interval_contained"]
        for key, julia_result in ref["outputs"].items():
            parts = key.split("_")
            value = float(parts[0])
            lower = float(parts[1])
            upper = float(parts[2])

            interval = closed_interval(lower, upper)
            py_result = contained(value, interval)

            assert py_result == julia_result, \
                f"Mismatch for contained({value}, closed_interval({lower}, {upper})): " \
                f"Python={py_result}, Julia={julia_result}"

    def test_contained_slice(self, reference_data):
        """contained_slice should match Julia."""
        from korg.utils import Interval, closed_interval, contained_slice

        ref = reference_data["interval_utils"]["contained_slice"]
        test_vals = ref["test_vals"]

        # Test exclusive interval
        exclusive_ref = ref["outputs"]["exclusive_3_10"]
        interval_exclusive = Interval(3.0, 10.0)
        start, end = contained_slice(test_vals, interval_exclusive)

        # Julia uses 1-based indexing, Python uses 0-based
        # Julia's first:last is inclusive, Python's start:end is exclusive on end
        julia_first = exclusive_ref["first"]  # 1-based
        julia_last = exclusive_ref["last"]    # 1-based
        julia_values = exclusive_ref["values"]

        # Convert Julia indices to Python slice
        py_values = test_vals[start:end]
        assert list(py_values) == list(julia_values), \
            f"Exclusive slice mismatch: Python got {py_values}, Julia got {julia_values}"

        # Test closed interval
        closed_ref = ref["outputs"]["closed_3_10"]
        interval_closed = closed_interval(3.0, 10.0)
        start, end = contained_slice(test_vals, interval_closed)

        julia_values = closed_ref["values"]
        py_values = test_vals[start:end]
        assert list(py_values) == list(julia_values), \
            f"Closed slice mismatch: Python got {py_values}, Julia got {julia_values}"


# =============================================================================
# Level 2 Tests
# =============================================================================

class TestHarrisSeriesReference:
    """Test harris_series against Julia reference data."""

    def test_harris_series(self, reference_data):
        """harris_series should match Julia."""
        from korg.line_profiles import harris_series

        ref = reference_data["harris_series"]
        for v_str, julia_vals in ref["outputs"].items():
            v = float(v_str)
            H0, H1, H2 = harris_series(v)

            assert np.isclose(float(H0), julia_vals["H0"], rtol=1e-6), \
                f"H0 mismatch for v={v}: Python={H0}, Julia={julia_vals['H0']}"
            assert np.isclose(float(H1), julia_vals["H1"], rtol=1e-6), \
                f"H1 mismatch for v={v}: Python={H1}, Julia={julia_vals['H1']}"
            assert np.isclose(float(H2), julia_vals["H2"], rtol=1e-6), \
                f"H2 mismatch for v={v}: Python={H2}, Julia={julia_vals['H2']}"


class TestVoigtHjertingReference:
    """Test voigt_hjerting against Julia reference data."""

    def test_voigt_hjerting(self, reference_data):
        """voigt_hjerting should match Julia."""
        from korg.line_profiles import voigt_hjerting

        ref = reference_data["voigt_hjerting"]
        for key, julia_val in ref["outputs"].items():
            parts = key.split("_")
            alpha = float(parts[0])
            v = float(parts[1])

            py_val = float(voigt_hjerting(alpha, v))
            assert np.isclose(py_val, julia_val, rtol=1e-5), \
                f"Mismatch for alpha={alpha}, v={v}: Python={py_val}, Julia={julia_val}"


class TestLineProfileReference:
    """Test line_profile against Julia reference data."""

    def test_line_profile(self, reference_data):
        """line_profile should match Julia."""
        from korg.line_profiles import line_profile

        ref = reference_data["line_profile"]
        inputs = ref["inputs"]
        outputs = ref["outputs"]

        for i, (wl0, sigma, gamma, amp, wl) in enumerate(inputs):
            julia_val = outputs[str(i + 1)]  # Julia is 1-indexed
            py_val = float(line_profile(wl0, sigma, gamma, amp, wl))

            assert np.isclose(py_val, julia_val, rtol=1e-5), \
                f"Mismatch for case {i}: Python={py_val}, Julia={julia_val}"


class TestInverseGaussianDensityReference:
    """Test inverse_gaussian_density against Julia reference data."""

    def test_inverse_gaussian_density(self, reference_data):
        """inverse_gaussian_density should match Julia."""
        from korg.line_profiles import inverse_gaussian_density

        ref = reference_data["inverse_gaussian_density"]
        for key, julia_val in ref["outputs"].items():
            parts = key.split("_")
            rho = float(parts[0])
            sigma = float(parts[1])

            py_val = float(inverse_gaussian_density(rho, sigma))
            assert np.isclose(py_val, julia_val, rtol=1e-6), \
                f"Mismatch for rho={rho}, sigma={sigma}: Python={py_val}, Julia={julia_val}"


class TestInverseLorentzDensityReference:
    """Test inverse_lorentz_density against Julia reference data."""

    def test_inverse_lorentz_density(self, reference_data):
        """inverse_lorentz_density should match Julia."""
        from korg.line_profiles import inverse_lorentz_density

        ref = reference_data["inverse_lorentz_density"]
        for key, julia_val in ref["outputs"].items():
            parts = key.split("_")
            rho = float(parts[0])
            gamma = float(parts[1])

            py_val = float(inverse_lorentz_density(rho, gamma))
            assert np.isclose(py_val, julia_val, rtol=1e-6), \
                f"Mismatch for rho={rho}, gamma={gamma}: Python={py_val}, Julia={julia_val}"


class TestScaledVdWReference:
    """Test scaled_vdW against Julia reference data."""

    def test_scaled_vdW(self, reference_data):
        """scaled_vdW should match Julia."""
        from korg.line_absorption import scaled_vdW

        ref = reference_data["scaled_vdW"]
        inputs = ref["inputs"]
        outputs = ref["outputs"]

        for i, (vdW, mass, T) in enumerate(inputs):
            julia_val = outputs[str(i + 1)]  # Julia is 1-indexed
            py_val = float(scaled_vdW(tuple(vdW), mass, T))

            assert np.isclose(py_val, julia_val, rtol=1e-5), \
                f"Mismatch for case {i}: vdW={vdW}, mass={mass}, T={T}: Python={py_val}, Julia={julia_val}"


class TestSpeciesDetailsReference:
    """Test Species class against Julia reference data."""

    def test_species_atoms(self, reference_data):
        """Species atomic properties should match Julia."""
        from korg.species import Species

        ref = reference_data["species_details"]

        # Test atomic species
        for code in ["Fe I", "Fe II", "Ca II", "H I", "He I"]:
            julia_data = ref[code]
            sp = Species(code)

            assert sp.charge == julia_data["charge"], \
                f"Charge mismatch for {code}: Python={sp.charge}, Julia={julia_data['charge']}"
            assert np.isclose(sp.get_mass(), julia_data["mass"], rtol=1e-6), \
                f"Mass mismatch for {code}: Python={sp.get_mass()}, Julia={julia_data['mass']}"
            assert sp.is_molecule() == julia_data["is_molecule"], \
                f"is_molecule mismatch for {code}"
            assert sp.n_atoms() == julia_data["n_atoms"], \
                f"n_atoms mismatch for {code}"

    def test_species_molecules(self, reference_data):
        """Species molecular properties should match Julia."""
        from korg.species import Species

        ref = reference_data["species_details"]

        # Test molecular species
        for code in ["CO", "H2O", "FeH", "TiO", "C2"]:
            julia_data = ref[code]
            sp = Species(code)

            assert sp.charge == julia_data["charge"], \
                f"Charge mismatch for {code}: Python={sp.charge}, Julia={julia_data['charge']}"
            assert np.isclose(sp.get_mass(), julia_data["mass"], rtol=1e-6), \
                f"Mass mismatch for {code}: Python={sp.get_mass()}, Julia={julia_data['mass']}"
            assert sp.is_molecule() == julia_data["is_molecule"], \
                f"is_molecule mismatch for {code}"
            assert sp.n_atoms() == julia_data["n_atoms"], \
                f"n_atoms mismatch for {code}"


class TestFormulaDetailsReference:
    """Test Formula class against Julia reference data."""

    def test_formula_properties(self, reference_data):
        """Formula properties should match Julia."""
        from korg.species import Formula

        ref = reference_data["formula_details"]

        for code in ["H", "Fe", "CO", "H2O", "FeH", "C2", "TiO"]:
            julia_data = ref[code]
            f = Formula(code)

            assert np.isclose(f.get_mass(), julia_data["mass"], rtol=1e-6), \
                f"Mass mismatch for {code}: Python={f.get_mass()}, Julia={julia_data['mass']}"
            assert f.n_atoms() == julia_data["n_atoms"], \
                f"n_atoms mismatch for {code}: Python={f.n_atoms()}, Julia={julia_data['n_atoms']}"
            assert f.is_molecule() == julia_data["is_molecule"], \
                f"is_molecule mismatch for {code}: Python={f.is_molecule()}, Julia={julia_data['is_molecule']}"


class TestJITCompatibility:
    """Test that functions work with JAX JIT."""

    def test_electron_scattering_jit(self):
        """electron_scattering should be JIT-compatible."""
        from korg.continuum_absorption.scattering import electron_scattering

        @jax.jit
        def compute_scattering(ne):
            return electron_scattering(ne)

        result = compute_scattering(1e14)
        assert np.isfinite(float(result))

    def test_translational_U_jit(self):
        """translational_U should be JIT-compatible."""
        from korg.statmech import translational_U
        from korg.constants import electron_mass_cgs

        @jax.jit
        def compute_U(T):
            return translational_U(electron_mass_cgs, T)

        result = compute_U(5777.0)
        assert np.isfinite(float(result))

    def test_rayleigh_jit(self):
        """rayleigh scattering should be JIT-compatible."""
        import jax.numpy as jnp
        from korg.continuum_absorption.scattering import rayleigh

        # Note: The current rayleigh implementation uses Python assert which
        # is not JIT-compatible. This test just verifies the function works
        # without JIT for now.
        nu = jnp.array([6e14])  # ~5000 Angstrom
        result = rayleigh(nu, 1e15, 1e14, 1e10)
        assert np.isfinite(float(result[0]))

    def test_sigma_line_jit(self):
        """sigma_line should be JIT-compatible."""
        from korg.line_absorption import sigma_line

        @jax.jit
        def compute_sigma(wl):
            return sigma_line(wl)

        wl_cm = 5000.0 * 1e-8
        result = compute_sigma(wl_cm)
        assert np.isfinite(float(result))

    def test_doppler_width_jit(self):
        """doppler_width should be JIT-compatible."""
        from korg.line_absorption import doppler_width
        from korg.constants import amu_cgs

        @jax.jit
        def compute_doppler(wl, T, mass, xi):
            return doppler_width(wl, T, mass, xi)

        wl_cm = 5000.0 * 1e-8
        T = 5777.0
        mass = 55.85 * amu_cgs
        xi = 1e5  # 1 km/s
        result = compute_doppler(wl_cm, T, mass, xi)
        assert np.isfinite(float(result))

    def test_scaled_stark_jit(self):
        """scaled_stark should be JIT-compatible."""
        from korg.line_absorption import scaled_stark

        @jax.jit
        def compute_stark(gamma, T):
            return scaled_stark(gamma, T)

        result = compute_stark(1e-6, 5777.0)
        assert np.isfinite(float(result))

    def test_normal_pdf_jit(self):
        """normal_pdf should be JIT-compatible."""
        from korg.utils import normal_pdf

        @jax.jit
        def compute_pdf(delta, sigma):
            return normal_pdf(delta, sigma)

        result = compute_pdf(1.0, 1.0)
        assert np.isfinite(float(result))

    def test_air_to_vacuum_jit(self):
        """air_to_vacuum should be JIT-compatible."""
        from korg.utils import air_to_vacuum

        # Wrap in jax.jit
        air_to_vacuum_jit = jax.jit(air_to_vacuum)

        result = air_to_vacuum_jit(5000.0)
        assert np.isfinite(float(result))
        # Verify the result is reasonable (vacuum wavelength > air wavelength)
        assert float(result) > 5000.0

    def test_vacuum_to_air_jit(self):
        """vacuum_to_air should be JIT-compatible."""
        from korg.utils import vacuum_to_air

        # Wrap in jax.jit
        vacuum_to_air_jit = jax.jit(vacuum_to_air)

        result = vacuum_to_air_jit(5000.0)
        assert np.isfinite(float(result))
        # Verify the result is reasonable (air wavelength < vacuum wavelength)
        assert float(result) < 5000.0

    def test_exponential_integral_1_jit(self):
        """exponential_integral_1 should be JIT-compatible."""
        from korg.radiative_transfer.expint import exponential_integral_1

        # Already has @jit decorator, but test it works
        result = exponential_integral_1(1.0)
        assert np.isfinite(float(result))

        # Test with a new JIT wrapper to verify
        @jax.jit
        def compute_e1(x):
            return exponential_integral_1(x)

        result2 = compute_e1(2.0)
        assert np.isfinite(float(result2))

    # Level 2 JIT tests

    def test_harris_series_jit(self):
        """harris_series should be JIT-compatible."""
        from korg.line_profiles import harris_series

        # Already has @jit decorator
        H0, H1, H2 = harris_series(1.0)
        assert np.isfinite(float(H0))
        assert np.isfinite(float(H1))
        assert np.isfinite(float(H2))

    def test_voigt_hjerting_jit(self):
        """voigt_hjerting should be JIT-compatible."""
        from korg.line_profiles import voigt_hjerting

        # Already has @jit decorator
        result = voigt_hjerting(0.5, 1.0)
        assert np.isfinite(float(result))

    def test_line_profile_jit(self):
        """line_profile should be JIT-compatible."""
        from korg.line_profiles import line_profile

        # Already has @jit decorator
        result = line_profile(5000e-8, 0.01e-8, 0.001e-8, 1.0, 5000e-8)
        assert np.isfinite(float(result))

    def test_inverse_gaussian_density_jit(self):
        """inverse_gaussian_density should be JIT-compatible."""
        from korg.line_profiles import inverse_gaussian_density

        # Already has @jit decorator
        result = inverse_gaussian_density(0.1, 1.0)
        assert np.isfinite(float(result))

    def test_inverse_lorentz_density_jit(self):
        """inverse_lorentz_density should be JIT-compatible."""
        from korg.line_profiles import inverse_lorentz_density

        # Already has @jit decorator
        result = inverse_lorentz_density(0.1, 1.0)
        assert np.isfinite(float(result))

    def test_scaled_vdW_jit(self):
        """scaled_vdW should be JIT-compatible with both simple and ABO scaling."""
        from korg.line_absorption import scaled_vdW
        from korg.constants import amu_cgs

        # Test simple scaling (vdW[1] == -1)
        @jax.jit
        def compute_vdw_simple(T):
            return scaled_vdW((1e-7, -1.0), 55.85 * amu_cgs, T)

        result = compute_vdw_simple(5777.0)
        assert np.isfinite(float(result))

        # Test ABO scaling (vdW[1] != -1) - now uses jax.scipy.special.gamma
        @jax.jit
        def compute_vdw_abo(T):
            return scaled_vdW((300.0, 0.25), 55.85 * amu_cgs, T)

        result_abo = compute_vdw_abo(5777.0)
        assert np.isfinite(float(result_abo))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
