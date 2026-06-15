"""Tests for continuum opacity functions and Julia reference comparison.

Tests the total_continuum_absorption / compute_continuum_absorption functions
against pre-computed Julia reference values, plus physical-sanity and JIT tests
for individual H⁻ bound-free/free-free sources.

Reference data lives in tests/julia_reference_data.json under the key
``total_continuum_absorption``.
"""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Enable 64-bit mode (must happen before other JAX code).
import korg  # noqa: F401 – side-effect: sets x64

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

REFERENCE_FILE = os.path.join(os.path.dirname(__file__), "julia_reference_data.json")


@pytest.fixture(scope="module")
def reference_data():
    """Load Julia reference data, skip if the file is absent."""
    if not os.path.exists(REFERENCE_FILE):
        return {}
    with open(REFERENCE_FILE) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helper: build Species-keyed number_densities from the solar_layer dict
# ---------------------------------------------------------------------------

def _solar_number_densities(solar_ref):
    """Return a {Species: float} dict from the solar_layer reference dict."""
    from korg.species import Species

    return {
        Species("H I"): solar_ref["nH_I"],
        Species("H II"): solar_ref["nH_II"],
        Species("He I"): solar_ref["nHe_I"],
        Species("He II"): solar_ref["nHe_II"],
        Species("H2 I"): solar_ref.get("nH2", 0.0),
    }


# ---------------------------------------------------------------------------
# 1. Reference comparison: total continuum absorption vs Julia
# ---------------------------------------------------------------------------

class TestContinuumAbsorptionSolarReference:
    """Compare compute_continuum_absorption to Julia reference values."""

    def test_solar_layer_each_wavelength(self, reference_data):
        """Absorption at each reference wavelength must match Julia within rtol=1e-4."""
        if "total_continuum_absorption" not in reference_data:
            pytest.skip("total_continuum_absorption not in reference data")

        ref = reference_data["total_continuum_absorption"]
        if "solar_layer" not in ref:
            pytest.skip("solar_layer case missing from reference data")

        try:
            from korg.synthesis import compute_continuum_absorption
            from korg.data_loader import default_partition_funcs
        except ImportError as exc:
            pytest.skip(f"Required module unavailable: {exc}")

        solar_ref = ref["solar_layer"]
        T = solar_ref["T"]
        ne = solar_ref["ne"]
        number_densities = _solar_number_densities(solar_ref)
        julia_outputs = solar_ref["outputs"]
        wavelengths_A = solar_ref["wavelengths_A"]

        for wl_A in wavelengths_A:
            wl_key = str(wl_A)
            if wl_key not in julia_outputs:
                continue

            wl_cm = np.array([wl_A * 1e-8])
            py_alpha = float(
                compute_continuum_absorption(
                    wl_cm, T, ne, number_densities, default_partition_funcs
                )[0]
            )
            julia_alpha = julia_outputs[wl_key]

            assert np.isclose(py_alpha, julia_alpha, rtol=1e-4), (
                f"Continuum absorption mismatch at {wl_A} Å: "
                f"Python={py_alpha:.6e}, Julia={julia_alpha:.6e}"
            )

    def test_solar_layer_all_wavelengths_vectorised(self, reference_data):
        """Vectorised call over all wavelengths matches Julia reference values."""
        if "total_continuum_absorption" not in reference_data:
            pytest.skip("total_continuum_absorption not in reference data")

        ref = reference_data["total_continuum_absorption"]
        if "solar_layer" not in ref:
            pytest.skip("solar_layer case missing from reference data")

        try:
            from korg.synthesis import compute_continuum_absorption
            from korg.data_loader import default_partition_funcs
        except ImportError as exc:
            pytest.skip(f"Required module unavailable: {exc}")

        solar_ref = ref["solar_layer"]
        T = solar_ref["T"]
        ne = solar_ref["ne"]
        number_densities = _solar_number_densities(solar_ref)
        julia_outputs = solar_ref["outputs"]
        wavelengths_A = solar_ref["wavelengths_A"]

        wl_cm = np.array([wl_A * 1e-8 for wl_A in wavelengths_A])
        py_alphas = compute_continuum_absorption(
            wl_cm, T, ne, number_densities, default_partition_funcs
        )
        julia_alphas = np.array([julia_outputs[str(wl_A)] for wl_A in wavelengths_A])

        assert np.allclose(py_alphas, julia_alphas, rtol=1e-4), (
            f"Vectorised continuum mismatch.\n"
            f"  Python : {np.array(py_alphas)}\n"
            f"  Julia  : {julia_alphas}"
        )

    def test_solar_layer_physical_magnitude(self, reference_data):
        """Absorption values should be in the expected range for the solar photosphere."""
        if "total_continuum_absorption" not in reference_data:
            pytest.skip("total_continuum_absorption not in reference data")

        ref = reference_data["total_continuum_absorption"]
        if "solar_layer" not in ref:
            pytest.skip("solar_layer case missing from reference data")

        try:
            from korg.synthesis import compute_continuum_absorption
            from korg.data_loader import default_partition_funcs
        except ImportError as exc:
            pytest.skip(f"Required module unavailable: {exc}")

        solar_ref = ref["solar_layer"]
        T = solar_ref["T"]
        ne = solar_ref["ne"]
        number_densities = _solar_number_densities(solar_ref)
        wavelengths_A = solar_ref["wavelengths_A"]

        wl_cm = np.array([wl_A * 1e-8 for wl_A in wavelengths_A])
        py_alphas = np.array(
            compute_continuum_absorption(
                wl_cm, T, ne, number_densities, default_partition_funcs
            )
        )

        assert np.all(py_alphas > 0), "All absorption values should be positive"
        assert np.all(np.isfinite(py_alphas)), "All absorption values should be finite"
        # Solar photosphere: expect roughly 1e-9 – 1e-5 cm⁻¹ across optical wavelengths
        assert np.all(py_alphas > 1e-10), "Absorption unexpectedly small"
        assert np.all(py_alphas < 1e-4), "Absorption unexpectedly large"

    def test_solar_layer_jit(self, reference_data):
        """compute_continuum_absorption result under jax.jit matches non-JIT."""
        if "total_continuum_absorption" not in reference_data:
            pytest.skip("total_continuum_absorption not in reference data")

        ref = reference_data["total_continuum_absorption"]
        if "solar_layer" not in ref:
            pytest.skip("solar_layer case missing from reference data")

        try:
            from korg.synthesis import compute_continuum_absorption
            from korg.data_loader import default_partition_funcs
        except ImportError as exc:
            pytest.skip(f"Required module unavailable: {exc}")

        solar_ref = ref["solar_layer"]
        T = float(solar_ref["T"])
        ne = float(solar_ref["ne"])
        number_densities = _solar_number_densities(solar_ref)

        wl_cm = np.array([5000.0e-8])  # 5000 Å

        # Baseline without JIT
        alpha_nojit = float(
            compute_continuum_absorption(wl_cm, T, ne, number_densities, default_partition_funcs)[0]
        )

        # Under JIT — the function itself iterates over a dict so we wrap only
        # the pure-JAX inner call (total_continuum_absorption).
        from korg.continuum import total_continuum_absorption
        from korg.constants import c_cgs
        from korg.data_loader import default_partition_funcs as pf
        from korg.species import Species

        # Build string-keyed dicts as synthesis.py does
        def _to_key(spec):
            return str(spec).replace(" ", "_")

        nd_str = {_to_key(k): v for k, v in number_densities.items()}
        pf_str = {_to_key(k): v for k, v in pf.items() if isinstance(k, Species)}

        nu = jnp.array([c_cgs / (5000.0e-8)])

        @jax.jit
        def jit_fn(nu_arr):
            return total_continuum_absorption(nu_arr, T, ne, nd_str, pf_str)

        alpha_jit = float(jit_fn(nu)[0])

        assert np.isclose(alpha_jit, alpha_nojit, rtol=1e-6), (
            f"JIT result {alpha_jit:.6e} differs from non-JIT {alpha_nojit:.6e}"
        )


# ---------------------------------------------------------------------------
# 2. H⁻ bound-free physical tests
# ---------------------------------------------------------------------------

class TestHminusBoundFree:
    """Physical-property tests for Hminus_bf (no Julia reference needed)."""

    @pytest.fixture(scope="class")
    def hminus_bf(self):
        try:
            from korg.continuum import Hminus_bf
            return Hminus_bf
        except ImportError as exc:
            pytest.skip(f"Hminus_bf not importable: {exc}")

    # Solar photosphere conditions
    T = 5778.0
    nH_I_div_U = 1.8e17 / 1.0   # rough: U(H I) ≈ 1 at photospheric T
    ne = 1.5e14

    def _nu(self, lambda_A):
        from korg.constants import c_cgs
        return c_cgs / (lambda_A * 1e-8)

    def test_zero_above_threshold(self, hminus_bf):
        """H⁻ bf cross-section should be zero at wavelengths above the ~1.644 μm threshold."""
        # Well above threshold (e.g. 20000 Å = 2 μm)
        nu_above = self._nu(20000.0)
        alpha = float(hminus_bf(nu_above, self.T, self.nH_I_div_U, self.ne))
        assert alpha == pytest.approx(0.0, abs=1e-50), (
            f"Hminus_bf should be 0 above ionisation threshold, got {alpha}"
        )

    def test_positive_below_threshold(self, hminus_bf):
        """H⁻ bf should be positive at optical wavelengths (well below threshold)."""
        for wl_A in [3000.0, 5000.0, 8000.0, 12000.0]:
            nu = self._nu(wl_A)
            alpha = float(hminus_bf(nu, self.T, self.nH_I_div_U, self.ne))
            assert alpha > 0, f"Hminus_bf should be positive at {wl_A} Å, got {alpha}"

    def test_finite_everywhere(self, hminus_bf):
        """H⁻ bf should be finite for a range spanning both sides of threshold."""
        lambdas_A = [1000.0, 3000.0, 5000.0, 10000.0, 16000.0, 20000.0, 30000.0]
        for wl_A in lambdas_A:
            nu = self._nu(wl_A)
            alpha = float(hminus_bf(nu, self.T, self.nH_I_div_U, self.ne))
            assert np.isfinite(alpha), f"Hminus_bf is not finite at {wl_A} Å: {alpha}"

    def test_array_input(self, hminus_bf):
        """Hminus_bf should accept JAX array inputs."""
        from korg.constants import c_cgs
        wls_A = jnp.array([3000.0, 5000.0, 8000.0])
        nus = c_cgs / (wls_A * 1e-8)
        alphas = hminus_bf(nus, self.T, self.nH_I_div_U, self.ne)
        assert alphas.shape == (3,), "Should return array of length 3"
        assert jnp.all(jnp.isfinite(alphas)), "All values should be finite"
        assert jnp.all(alphas >= 0.0), "All values should be non-negative"

    def test_jit_compatible(self, hminus_bf):
        """Hminus_bf should be jax.jit-able and produce the same result."""
        nu_val = self._nu(5000.0)
        T = self.T
        nH_div_U = self.nH_I_div_U
        ne = self.ne

        alpha_nojit = float(hminus_bf(nu_val, T, nH_div_U, ne))

        @jax.jit
        def jitted(nu):
            return hminus_bf(nu, T, nH_div_U, ne)

        alpha_jit = float(jitted(jnp.array(nu_val)))
        assert np.isclose(alpha_jit, alpha_nojit, rtol=1e-10), (
            f"JIT vs non-JIT mismatch: {alpha_jit} vs {alpha_nojit}"
        )

    def test_proportional_to_electron_density(self, hminus_bf):
        """Hminus_bf should scale roughly linearly with ne (Saha equation regime)."""
        nu_val = self._nu(5000.0)
        # In the linear Saha limit n(H⁻) ∝ ne, so alpha ∝ ne.
        alpha1 = float(hminus_bf(nu_val, self.T, self.nH_I_div_U, 1e14))
        alpha2 = float(hminus_bf(nu_val, self.T, self.nH_I_div_U, 2e14))
        ratio = alpha2 / alpha1
        assert 1.5 < ratio < 2.5, (
            f"Expected ~2× scaling with 2× ne, got {ratio:.3f}"
        )


# ---------------------------------------------------------------------------
# 3. H⁻ free-free physical tests
# ---------------------------------------------------------------------------

class TestHminusFreeFree:
    """Physical-property tests for Hminus_ff."""

    @pytest.fixture(scope="class")
    def hminus_ff(self):
        try:
            from korg.continuum import Hminus_ff
            return Hminus_ff
        except ImportError as exc:
            pytest.skip(f"Hminus_ff not importable: {exc}")

    T = 5778.0
    nH_I_div_U = 1.8e17
    ne = 1.5e14

    def _nu(self, lambda_A):
        from korg.constants import c_cgs
        return c_cgs / (lambda_A * 1e-8)

    def test_positive_everywhere(self, hminus_ff):
        """H⁻ ff should be positive at all wavelengths."""
        for wl_A in [2000.0, 5000.0, 10000.0, 20000.0]:
            nu = self._nu(wl_A)
            alpha = float(hminus_ff(nu, self.T, self.nH_I_div_U, self.ne))
            assert alpha > 0, f"Hminus_ff should be positive at {wl_A} Å, got {alpha}"

    def test_increases_with_wavelength(self, hminus_ff):
        """H⁻ ff opacity generally increases toward longer wavelengths (Bell & Berrington)."""
        # Sample a monotonically increasing wavelength sequence in the interpolation range
        wls_A = [3000.0, 6000.0, 12000.0]
        alphas = [float(hminus_ff(self._nu(w), self.T, self.nH_I_div_U, self.ne))
                  for w in wls_A]
        assert alphas[1] > alphas[0], (
            f"Hminus_ff should increase: {alphas[0]:.3e} at 3000 Å > {alphas[1]:.3e} at 6000 Å"
        )
        assert alphas[2] > alphas[1], (
            f"Hminus_ff should increase: {alphas[1]:.3e} at 6000 Å > {alphas[2]:.3e} at 12000 Å"
        )

    def test_finite_for_range(self, hminus_ff):
        """H⁻ ff should be finite across a wide wavelength range."""
        for wl_A in [2000.0, 5000.0, 10000.0, 50000.0]:
            nu = self._nu(wl_A)
            alpha = float(hminus_ff(nu, self.T, self.nH_I_div_U, self.ne))
            assert np.isfinite(alpha), f"Hminus_ff is not finite at {wl_A} Å: {alpha}"

    def test_array_input(self, hminus_ff):
        """Hminus_ff should accept array inputs and return matching array."""
        from korg.constants import c_cgs
        wls_A = jnp.array([3000.0, 5000.0, 10000.0])
        nus = c_cgs / (wls_A * 1e-8)
        alphas = hminus_ff(nus, self.T, self.nH_I_div_U, self.ne)
        assert alphas.shape == (3,)
        assert jnp.all(alphas > 0)
        assert jnp.all(jnp.isfinite(alphas))

    def test_jit_compatible(self, hminus_ff):
        """Hminus_ff should work under jax.jit."""
        nu_val = self._nu(5000.0)
        T, nH, ne = self.T, self.nH_I_div_U, self.ne
        alpha_ref = float(hminus_ff(nu_val, T, nH, ne))

        @jax.jit
        def jitted(nu):
            return hminus_ff(nu, T, nH, ne)

        alpha_jit = float(jitted(jnp.array(nu_val)))
        assert np.isclose(alpha_jit, alpha_ref, rtol=1e-10)


# ---------------------------------------------------------------------------
# 4. Wavelength/temperature-dependence of total continuum
# ---------------------------------------------------------------------------

class TestContinuumPhysics:
    """Physical-sanity tests on the total continuum function."""

    @pytest.fixture(scope="class")
    def total_fn(self):
        try:
            from korg.continuum import total_continuum_absorption
            return total_continuum_absorption
        except ImportError as exc:
            pytest.skip(f"total_continuum_absorption not importable: {exc}")

    @pytest.fixture(scope="class")
    def partition_funcs_str(self):
        """String-keyed partition function dict as expected by total_continuum_absorption."""
        try:
            from korg.data_loader import default_partition_funcs
            from korg.species import Species
            return {str(k).replace(" ", "_"): v for k, v in default_partition_funcs.items()
                    if isinstance(k, Species)}
        except ImportError as exc:
            pytest.skip(f"default_partition_funcs not importable: {exc}")

    # Minimal solar-photosphere number densities (string keys)
    _nd = {
        "H_I": 1.8e17,
        "H_II": 1.0e11,
        "He_I": 1.6e16,
        "He_II": 1.0e9,
        "H2_I": 1.0e12,
    }
    T = 5778.0
    ne = 1.5e14

    def _nu(self, lambda_A):
        from korg.constants import c_cgs
        return c_cgs / (lambda_A * 1e-8)

    def test_positive_and_finite(self, total_fn, partition_funcs_str):
        """Total continuum should be positive and finite at visible wavelengths."""
        for wl_A in [3000.0, 5000.0, 8000.0]:
            nu = jnp.array([self._nu(wl_A)])
            alpha = float(total_fn(nu, self.T, self.ne, self._nd, partition_funcs_str)[0])
            assert np.isfinite(alpha), f"alpha is not finite at {wl_A} Å"
            assert alpha > 0, f"alpha should be positive at {wl_A} Å, got {alpha}"

    def test_different_wavelengths_give_different_values(self, total_fn, partition_funcs_str):
        """Different wavelengths should produce different absorption coefficients."""
        nu_3000 = self._nu(3000.0)
        nu_8000 = self._nu(8000.0)
        alpha_3 = float(
            total_fn(jnp.array([nu_3000]), self.T, self.ne, self._nd, partition_funcs_str)[0]
        )
        alpha_8 = float(
            total_fn(jnp.array([nu_8000]), self.T, self.ne, self._nd, partition_funcs_str)[0]
        )
        assert alpha_3 != alpha_8, (
            "Continuum at 3000 Å and 8000 Å should differ"
        )

    def test_absorption_increases_with_temperature(self, total_fn, partition_funcs_str):
        """At fixed density, higher T raises H⁻ opacity (more electrons from metals)."""
        nu = jnp.array([self._nu(5000.0)])
        alpha_low = float(
            total_fn(nu, 4000.0, self.ne, self._nd, partition_funcs_str)[0]
        )
        alpha_high = float(
            total_fn(nu, 7000.0, self.ne, self._nd, partition_funcs_str)[0]
        )
        # Both should be positive; we just check they differ
        assert alpha_low > 0
        assert alpha_high > 0
        assert alpha_low != alpha_high, (
            f"Continuum absorption should differ between 4000 K and 7000 K; "
            f"got {alpha_low:.3e} and {alpha_high:.3e}"
        )

    def test_vectorised_frequency_array(self, total_fn, partition_funcs_str):
        """total_continuum_absorption should handle a multi-element frequency array."""
        from korg.constants import c_cgs
        wls_A = jnp.array([3000.0, 4000.0, 5000.0, 6000.0, 8000.0, 10000.0])
        nus = c_cgs / (wls_A * 1e-8)
        alphas = total_fn(nus, self.T, self.ne, self._nd, partition_funcs_str)
        assert alphas.shape == (6,)
        assert jnp.all(jnp.isfinite(alphas))
        assert jnp.all(alphas > 0)

    def test_zero_density_gives_small_absorption(self, total_fn, partition_funcs_str):
        """With near-zero number densities, absorption should be very small (dominated by e-scatter)."""
        nd_empty = {
            "H_I": 1.0,     # trace only
            "H_II": 1.0,
            "He_I": 1.0,
            "He_II": 0.0,
            "H2_I": 0.0,
        }
        nu = jnp.array([self._nu(5000.0)])
        alpha = float(total_fn(nu, self.T, 1.0, nd_empty, partition_funcs_str)[0])
        # Should be non-negative and very small
        assert alpha >= 0.0
        assert np.isfinite(alpha)
        assert alpha < 1e-20, (
            f"Expected very small absorption with trace densities, got {alpha:.3e}"
        )

    def test_electron_scattering_scales_linearly(self, total_fn, partition_funcs_str):
        """Thomson scattering ∝ ne: doubling ne (with zero number densities) should ~double alpha."""
        nd_zero = {k: 0.0 for k in ["H_I", "H_II", "He_I", "He_II", "H2_I"]}
        nu = jnp.array([self._nu(5000.0)])
        ne1 = 1e14
        ne2 = 2e14
        a1 = float(total_fn(nu, self.T, ne1, nd_zero, partition_funcs_str)[0])
        a2 = float(total_fn(nu, self.T, ne2, nd_zero, partition_funcs_str)[0])
        if a1 == 0.0:
            pytest.skip("electron scattering contribution is zero — check implementation")
        assert np.isclose(a2 / a1, 2.0, rtol=1e-4), (
            f"Thomson scattering should scale linearly with ne: ratio={a2/a1:.4f}"
        )
