"""Extended statistical mechanics tests with Julia reference comparisons and JIT compatibility.

These tests complement tests/test_julia_reference.py (TestSahaEquation, TestGetLogNK)
with additional coverage of physics edge cases, cross-element comparisons, JIT behaviour,
and multi-molecule equilibrium constant consistency.
"""

import json
import os

import numpy as np
import pytest
import jax
import jax.numpy as jnp


@pytest.fixture(scope="module")
def reference_data():
    data_file = os.path.join(os.path.dirname(__file__), "julia_reference_data.json")
    if not os.path.exists(data_file):
        return {}
    with open(data_file) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def ionization_data():
    """Load ionization energies and partition functions; skip if unavailable."""
    try:
        from korg.data_loader import load_ionization_energies, load_atomic_partition_functions
    except ImportError as e:
        pytest.skip(f"Required modules not available: {e}")

    try:
        ie = load_ionization_energies()
        pf = load_atomic_partition_functions()
    except FileNotFoundError as e:
        pytest.skip(f"Data files not found: {e}")

    return ie, pf


@pytest.fixture(scope="module")
def equilibrium_constants():
    """Load Barklem-Collet equilibrium constants; skip if unavailable."""
    try:
        from korg.data_loader import load_barklem_collet_equilibrium_constants
    except ImportError as e:
        pytest.skip(f"Required modules not available: {e}")

    try:
        ec = load_barklem_collet_equilibrium_constants()
    except FileNotFoundError as e:
        pytest.skip(f"Data files not found: {e}")

    return ec


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _saha(T, ne, Z, ionization_data):
    try:
        from korg.statmech import saha_ion_weights
    except ImportError as e:
        pytest.skip(f"saha_ion_weights not importable: {e}")

    ie, pf = ionization_data
    return saha_ion_weights(T, ne, Z, ie, pf)


# ---------------------------------------------------------------------------
# Class 1 – Physics tests not covered by TestSahaEquation
# ---------------------------------------------------------------------------

class TestSahaIonWeightsPhysics:
    """Physics-level checks on saha_ion_weights beyond the basic hydrogen/iron tests."""

    def test_hydrogen_mostly_neutral_cool(self, ionization_data):
        """At T=5000 K, ne=1e13: hydrogen is almost entirely neutral (wII << 1)."""
        wII, wIII = _saha(5000.0, 1e13, 1, ionization_data)
        assert float(wII) < 1e-3, f"Expected wII << 1 for cool H, got {float(wII):.4e}"
        assert float(wIII) == 0.0, "Hydrogen has no wIII (cannot be doubly ionized)"

    def test_hydrogen_mostly_ionized_hot(self, ionization_data):
        """At T=20000 K, ne=1e14: hydrogen is mostly ionized (wII >> 1)."""
        wII, wIII = _saha(20000.0, 1e14, 1, ionization_data)
        assert float(wII) > 10.0, f"Expected wII >> 1 for hot H, got {float(wII):.4e}"
        assert float(wIII) == 0.0

    def test_wIII_always_less_than_wII_cool_star(self, ionization_data):
        """For Z>=2 at cool-to-solar temperatures, wIII << wII (second ionization rare)."""
        # Test a few elements at solar conditions
        ne = 1e14
        T = 5777.0
        for Z in (2, 12, 20, 26):  # He, Mg, Ca, Fe
            wII, wIII = _saha(T, ne, Z, ionization_data)
            assert float(wIII) < float(wII), \
                f"Z={Z}: expected wIII < wII but wII={float(wII):.4e}, wIII={float(wIII):.4e}"

    def test_increasing_T_increases_wII(self, ionization_data):
        """Ionization fraction wII increases monotonically with T at fixed ne."""
        ne = 1e14
        temps = [3000.0, 5000.0, 7000.0, 10000.0]
        prev_wII = -1.0
        for T in temps:
            wII, _ = _saha(T, ne, 26, ionization_data)  # Fe
            assert float(wII) > prev_wII, \
                f"wII did not increase going from previous T to {T}: wII={float(wII):.4e}"
            prev_wII = float(wII)

    def test_increasing_ne_decreases_wII(self, ionization_data):
        """Higher electron density suppresses ionization (Saha equation)."""
        T = 5777.0
        ne_values = [1e11, 1e13, 1e15]
        prev_wII = jnp.inf
        for ne in ne_values:
            wII, _ = _saha(T, ne, 26, ionization_data)  # Fe
            assert float(wII) < float(prev_wII), \
                f"wII did not decrease at ne={ne:.0e}: wII={float(wII):.4e}"
            prev_wII = wII

    def test_calcium_ionizes_more_easily_than_iron_solar(self, ionization_data):
        """Ca (Z=20) ionizes more easily than Fe (Z=26) at solar conditions.

        Ca has a lower first ionization potential (~6.1 eV) than Fe (~7.9 eV).
        """
        T, ne = 5777.0, 1e14
        wII_ca, _ = _saha(T, ne, 20, ionization_data)
        wII_fe, _ = _saha(T, ne, 26, ionization_data)
        assert float(wII_ca) > float(wII_fe), (
            f"Expected Ca more ionized than Fe: wII_Ca={float(wII_ca):.4e}, "
            f"wII_Fe={float(wII_fe):.4e}"
        )

    def test_wII_wIII_nonzero_for_metals(self, ionization_data):
        """wII and wII*wIII should both be nonzero for heavy elements at high T."""
        T, ne = 8000.0, 1e14
        wII, wIII = _saha(T, ne, 26, ionization_data)  # Fe hot
        assert float(wII) > 0.0
        assert float(wII * wIII) > 0.0


# ---------------------------------------------------------------------------
# Class 2 – JIT compatibility of saha_ion_weights
# ---------------------------------------------------------------------------

class TestSahaIonWeightsJITCompatibility:
    """Verify that the core arithmetic of saha_ion_weights is JIT-traceable."""

    def test_core_math_jit_compatible(self, ionization_data):
        """Wrap numeric-only calculations in jax.jit and confirm they compile."""
        try:
            from korg.statmech import translational_U
            from korg.constants import kboltz_eV, electron_mass_cgs
        except ImportError as e:
            pytest.skip(f"Required module not importable: {e}")

        @jax.jit
        def saha_core(T, ne, chi_I, UI, UII):
            trans = translational_U(electron_mass_cgs, T)
            return 2.0 / ne * (UII / UI) * trans * jnp.exp(-chi_I / (kboltz_eV * T))

        result = saha_core(5777.0, 1e14, 7.902, 25.0, 20.0)
        assert jnp.isfinite(result), "JIT-compiled saha core returned non-finite result"
        assert result > 0.0

    def test_jit_result_matches_eager_result(self, ionization_data):
        """JIT and eager mode must agree for the core Saha calculation."""
        try:
            from korg.statmech import translational_U
            from korg.constants import kboltz_eV, electron_mass_cgs
        except ImportError as e:
            pytest.skip(f"Required module not importable: {e}")

        def saha_core(T, ne, chi_I, UI, UII):
            trans = translational_U(electron_mass_cgs, T)
            return 2.0 / ne * (UII / UI) * trans * jnp.exp(-chi_I / (kboltz_eV * T))

        saha_jit = jax.jit(saha_core)

        params = (5777.0, 1e14, 7.902, 25.0, 20.0)
        eager_val = float(saha_core(*params))
        jit_val = float(saha_jit(*params))

        assert np.isclose(eager_val, jit_val, rtol=1e-6), \
            f"JIT vs eager mismatch: JIT={jit_val}, eager={eager_val}"

    def test_jit_vectorised_over_temperatures(self, ionization_data):
        """vmap over temperatures should produce finite, monotone ionization fractions."""
        try:
            from korg.statmech import translational_U
            from korg.constants import kboltz_eV, electron_mass_cgs
        except ImportError as e:
            pytest.skip(f"Required module not importable: {e}")

        def saha_core(T):
            trans = translational_U(electron_mass_cgs, T)
            return 2.0 / 1e14 * (20.0 / 25.0) * trans * jnp.exp(-7.902 / (kboltz_eV * T))

        temps = jnp.array([3000.0, 5000.0, 7000.0, 10000.0, 15000.0])
        wII_arr = jax.vmap(jax.jit(saha_core))(temps)

        assert jnp.all(jnp.isfinite(wII_arr)), "Some vmap results are non-finite"
        # Should be monotonically increasing
        diffs = jnp.diff(wII_arr)
        assert jnp.all(diffs > 0), "wII is not monotonically increasing with T"

    def test_full_function_returns_finite_results(self, ionization_data):
        """The full saha_ion_weights function returns finite, positive values."""
        wII, wIII = _saha(5777.0, 1e14, 26, ionization_data)
        assert jnp.isfinite(wII), f"wII is not finite: {wII}"
        assert jnp.isfinite(wIII), f"wIII is not finite: {wIII}"
        assert wII > 0, "wII must be positive"
        assert wIII >= 0, "wIII must be non-negative"


# ---------------------------------------------------------------------------
# Class 3 – get_log_nK: JIT and multi-molecule tests
# ---------------------------------------------------------------------------

class TestGetLogNKJITCompatibility:
    """JIT compatibility and per-molecule checks for get_log_nK."""

    def test_finite_for_co_h2_oh_at_5000k(self, equilibrium_constants):
        """get_log_nK returns finite values for CO, H2, OH at 5000 K."""
        try:
            from korg.statmech import get_log_nK
            from korg.species import Species
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        T = 5000.0
        for mol_str in ("CO", "H2", "OH"):
            try:
                mol = Species(mol_str)
            except Exception as e:
                pytest.skip(f"Cannot create Species('{mol_str}'): {e}")

            if mol not in equilibrium_constants:
                pytest.skip(f"{mol_str} not in equilibrium constants")

            val = get_log_nK(mol, T, equilibrium_constants)
            assert np.isfinite(float(val)), f"Non-finite log_nK for {mol_str} at T={T}"

    def test_co_bond_stronger_than_h2_at_solar_T(self, equilibrium_constants):
        """CO dissociation equilibrium constant should be lower than H2 at solar T.

        CO has a stronger bond (~11 eV) than H2 (~4.5 eV), so K_diss(CO) < K_diss(H2),
        i.e. log_nK(CO) < log_nK(H2) at the same temperature.
        """
        try:
            from korg.statmech import get_log_nK
            from korg.species import Species
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        T = 5777.0
        co, h2 = Species("CO"), Species("H2")
        if co not in equilibrium_constants or h2 not in equilibrium_constants:
            pytest.skip("CO or H2 not in equilibrium constants")

        log_nK_co = float(get_log_nK(co, T, equilibrium_constants))
        log_nK_h2 = float(get_log_nK(h2, T, equilibrium_constants))

        assert log_nK_co < log_nK_h2, (
            f"Expected log_nK(CO) < log_nK(H2) at T={T}: "
            f"CO={log_nK_co:.4f}, H2={log_nK_h2:.4f}"
        )

    def test_co_increases_with_temperature(self, equilibrium_constants):
        """CO dissociation equilibrium constant increases with temperature."""
        try:
            from korg.statmech import get_log_nK
            from korg.species import Species
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        co = Species("CO")
        if co not in equilibrium_constants:
            pytest.skip("CO not in equilibrium constants")

        temps = [4000.0, 5777.0, 7000.0, 8000.0]
        values = [float(get_log_nK(co, T, equilibrium_constants)) for T in temps]

        for i in range(len(values) - 1):
            assert values[i + 1] > values[i], (
                f"log_nK(CO) did not increase from T={temps[i]} to T={temps[i+1]}: "
                f"{values[i]:.4f} -> {values[i+1]:.4f}"
            )

    def test_julia_reference_co(self, reference_data, equilibrium_constants):
        """get_log_nK for CO must match Julia reference values (rtol=1e-5)."""
        if "get_log_nK" not in reference_data:
            pytest.skip("get_log_nK not in Julia reference data")

        try:
            from korg.statmech import get_log_nK
            from korg.species import Species
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        co = Species("CO")
        if co not in equilibrium_constants:
            pytest.skip("CO not in equilibrium constants")

        ref_co = reference_data["get_log_nK"]["outputs"].get("CO", {})
        for T_str, julia_val in ref_co.items():
            T = float(T_str)
            py_val = float(get_log_nK(co, T, equilibrium_constants))
            assert np.isclose(py_val, julia_val, rtol=1e-5), (
                f"CO log_nK mismatch at T={T}: Python={py_val:.8f}, Julia={julia_val:.8f}"
            )

    def test_julia_reference_h2(self, reference_data, equilibrium_constants):
        """get_log_nK for H2 must match Julia reference values (rtol=1e-5)."""
        if "get_log_nK" not in reference_data:
            pytest.skip("get_log_nK not in Julia reference data")

        try:
            from korg.statmech import get_log_nK
            from korg.species import Species
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        h2 = Species("H2")
        if h2 not in equilibrium_constants:
            pytest.skip("H2 not in equilibrium constants")

        ref_h2 = reference_data["get_log_nK"]["outputs"].get("H2", {})
        for T_str, julia_val in ref_h2.items():
            T = float(T_str)
            py_val = float(get_log_nK(h2, T, equilibrium_constants))
            assert np.isclose(py_val, julia_val, rtol=1e-5), (
                f"H2 log_nK mismatch at T={T}: Python={py_val:.8f}, Julia={julia_val:.8f}"
            )

    def test_julia_reference_oh(self, reference_data, equilibrium_constants):
        """get_log_nK for OH must match Julia reference values (rtol=1e-5)."""
        if "get_log_nK" not in reference_data:
            pytest.skip("get_log_nK not in Julia reference data")

        try:
            from korg.statmech import get_log_nK
            from korg.species import Species
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        oh = Species("OH")
        if oh not in equilibrium_constants:
            pytest.skip("OH not in equilibrium constants")

        ref_oh = reference_data["get_log_nK"]["outputs"].get("OH", {})
        for T_str, julia_val in ref_oh.items():
            T = float(T_str)
            py_val = float(get_log_nK(oh, T, equilibrium_constants))
            assert np.isclose(py_val, julia_val, rtol=1e-5), (
                f"OH log_nK mismatch at T={T}: Python={py_val:.8f}, Julia={julia_val:.8f}"
            )


# ---------------------------------------------------------------------------
# Class 4 – Cross-checks between saha_ion_weights and equilibrium interpretation
# ---------------------------------------------------------------------------

class TestSahaEquilibriumConsistency:
    """Consistency tests between ionization ratios and physical interpretation."""

    def test_hydrogen_always_has_zero_wIII(self, ionization_data):
        """wIII for hydrogen is exactly 0 at all temperatures (H has only 1 electron)."""
        ne = 1e14
        for T in (3000.0, 5777.0, 10000.0, 20000.0):
            _, wIII = _saha(T, ne, 1, ionization_data)
            assert float(wIII) == 0.0, f"H wIII should be 0 at T={T}, got {float(wIII)}"

    def test_high_T_hydrogen_dominant_ionized(self, ionization_data):
        """At T > 15000 K, hydrogen wII >> 1 (ionized dominates neutral)."""
        wII, _ = _saha(15000.0, 1e14, 1, ionization_data)
        assert float(wII) > 1.0, \
            f"At T=15000 K, H wII should exceed 1, got {float(wII):.4e}"

    def test_iron_solar_mostly_neutral_wII_order_unity(self, ionization_data):
        """At solar T and ne, iron wII ~ a few (singly-ionized and neutral both present).

        Julia reference gives wII ≈ 4.1 for Fe at T=5777, ne=1e14.
        We check it is between 0.1 and 100 (order-unity regime).
        """
        wII, wIII = _saha(5777.0, 1e14, 26, ionization_data)
        assert 0.1 < float(wII) < 100.0, \
            f"Fe wII at solar T/ne expected order unity, got {float(wII):.4e}"
        assert float(wIII) < 1e-4, \
            f"Fe wIII at solar T/ne should be tiny, got {float(wIII):.4e}"

    def test_magnesium_highly_ionized_at_solar_conditions(self, ionization_data):
        """Mg (Z=12) has low first ionization energy (~7.6 eV) so wII >> 1 at solar T.

        Julia reference: T=4500, ne=1e12, Z=12 → wII ≈ 7.9.
        """
        wII, _ = _saha(4500.0, 1e12, 12, ionization_data)
        assert float(wII) > 1.0, \
            f"Mg at T=4500/ne=1e12: expected wII > 1, got {float(wII):.4e}"

    def test_julia_reference_additional_entries(self, reference_data, ionization_data):
        """Spot-check a subset of Julia reference entries not used by TestSahaEquation.

        TestSahaEquation already iterates over all entries; here we explicitly assert
        specific entries that exercise the Mg, hot-H, and hot-Fe cases.
        """
        if "saha_ion_weights" not in reference_data:
            pytest.skip("saha_ion_weights not in Julia reference data")

        try:
            from korg.statmech import saha_ion_weights
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

        ie, pf = ionization_data
        # Entries of specific physical interest beyond what's in the existing tests
        spot_checks = {
            "4500.0_1.0e12_12": "Mg at T=4500, ne=1e12",   # highly ionised Mg
            "8000.0_1.0e14_26": "Fe at T=8000, ne=1e14",   # hot Fe (wII~479, wIII~0.46)
            "10000.0_1.0e15_1": "H at T=10000, ne=1e15",   # moderately ionized H
        }
        ref = reference_data["saha_ion_weights"]["outputs"]

        for key, description in spot_checks.items():
            if key not in ref:
                continue
            parts = key.split("_")
            T = float(parts[0])
            ne = float(parts[1])
            Z = int(parts[2])
            julia = ref[key]

            wII, wIII = saha_ion_weights(T, ne, Z, ie, pf)

            assert np.isclose(float(wII), julia["wII"], rtol=1e-5), \
                f"wII mismatch [{description}]: Python={float(wII):.6e}, Julia={julia['wII']:.6e}"
            assert np.isclose(float(wIII), julia["wIII"], rtol=1e-5), \
                f"wIII mismatch [{description}]: Python={float(wIII):.6e}, Julia={julia['wIII']:.6e}"

    def test_saha_ratio_wII_wIII_second_ionization_harder(self, ionization_data):
        """For any element, the second ionization is harder than the first.

        This means n(X III) / n(X II) < n(X II) / n(X I) for stellar conditions,
        i.e. wIII < wII^2 (since wIII = wII * ratio_23 and ratio_23 < wII for cool stars).
        """
        T, ne = 5777.0, 1e14
        for Z in (6, 12, 20, 26):  # C, Mg, Ca, Fe
            wII, wIII = _saha(T, ne, Z, ionization_data)
            # wIII/wII gives the II->III ratio, while wII gives the I->II ratio
            # For the second ionization to be harder: wIII/wII < wII
            if float(wII) > 0 and float(wIII) > 0:
                ratio_23 = float(wIII) / float(wII)
                assert ratio_23 < float(wII), (
                    f"Z={Z}: ratio_23={ratio_23:.4e} should be < wII={float(wII):.4e} "
                    f"(second ionization should be harder than first)"
                )
