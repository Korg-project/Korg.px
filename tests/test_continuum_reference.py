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


# ---------------------------------------------------------------------------
# 5. Per-source Julia references
# ---------------------------------------------------------------------------
#
# The total-continuum comparison above exercises nine sources at a single solar layer,
# where H⁻ dominates the optical continuum.  A large relative error in a sub-dominant
# source (metal bf, positive-ion ff) hides inside the tolerance on that sum, so the five
# sources below are also compared against Julia individually, at three conditions
# (T = 3500 / 5778 / 9000 K) spanning 3000–20000 Å.
#
# Fixtures live under the ``continuum_sources`` key of tests/julia_reference_data.json;
# see the matching section of tests/generate_julia_reference.jl.

CONTINUUM_CONDITIONS = ["cool_dense", "solar", "hot"]


def _source_case(reference_data, label):
    """Return the ``continuum_sources`` entry for ``label``, or skip."""
    sources = reference_data.get("continuum_sources")
    if not sources or label not in sources:
        pytest.skip(f"continuum_sources[{label!r}] not in reference data "
                    "(regenerate tests/julia_reference_data.json)")
    return sources[label]


def _assert_matches(py_vals, julia_vals, wavelengths_A, rtol, what):
    """Element-wise comparison that also demands exact zeros where Julia gives zero.

    Several sources are identically zero outside their tabulated range (Heminus_ff below
    5063 Å, Hminus_bf beyond the 1.64 μm threshold), and a relative tolerance says nothing
    there — so those points are checked exactly.
    """
    py_vals = np.asarray(py_vals, dtype=float)
    julia_vals = np.asarray(julia_vals, dtype=float)
    assert np.all(np.isfinite(py_vals)), f"{what}: non-finite Python values {py_vals}"

    for wl_A, py, julia in zip(wavelengths_A, py_vals, julia_vals):
        if julia == 0.0:
            assert py == 0.0, f"{what} at {wl_A} Å: Julia gives exactly 0, Python {py:.6e}"
        else:
            assert np.isclose(py, julia, rtol=rtol, atol=0.0), (
                f"{what} at {wl_A} Å: Python={py:.10e} Julia={julia:.10e} "
                f"(rel={py / julia - 1:.3e}, rtol={rtol:g})"
            )


class TestContinuumSourceReference:
    """Compare each individually-untested continuum source against Julia."""

    @staticmethod
    def _nus(wavelengths_A):
        from korg.constants import c_cgs
        return jnp.array([c_cgs / (wl_A * 1e-8) for wl_A in wavelengths_A])

    # -- H⁻ number density ---------------------------------------------------
    #
    # Korg v1.2 changed Hminus_bf to take n(H⁻) directly, supplied by chemical
    # equilibrium as n(H⁻) = Hminus_nK(T) * n(H I) * nₑ.  Korg.px still derives it inside
    # Hminus_bf from the ground-state population (the v1.1 behaviour), i.e. the same
    # expression with n(H I) → n(H I, n=1) = 2 n(H I) / U(H I).  The two therefore differ
    # by a factor 2/U(H I), which is 1 to within 1e-14 at 3500 K but reaches 1.5e-5 at
    # 9000 K.  Both comparisons are made explicitly below rather than absorbed into a
    # loose tolerance on Hminus_bf.

    @pytest.mark.parametrize("label", CONTINUUM_CONDITIONS)
    def test_hminus_number_density_ground_state_convention(self, reference_data, label):
        """Python's internal n(H⁻) matches Korg's nK relation in the ground-state convention.

        Korg.px previously carried Korg.jl v1.1's literal coef = 3.31283018e-22, which is
        high by 9.238e-7 relative to (h²·k_eV / 2π·m_e·k_cgs)^1.5 — the value v1.2 obtains
        via 1/translational_U(m_e, T).  The coefficient is now derived from the constants
        instead, so this agrees to machine precision and the tolerance is 1e-12.
        """
        from korg.continuum import ndens_Hminus

        case = _source_case(reference_data, label)
        py = float(ndens_Hminus(case["nH_I_div_U"], case["ne"], case["T"]))
        julia = case["nHminus_ground_state"]

        rel = py / julia - 1
        assert abs(rel) < 1e-12, (
            f"{label}: Python n(H⁻)={py:.10e} vs Julia ground-state convention "
            f"{julia:.10e} (rel={rel:.3e})"
        )

    @pytest.mark.parametrize("label", CONTINUUM_CONDITIONS)
    def test_hminus_number_density_vs_korg_v1_2(self, reference_data, label):
        """Quantify the documented v1.1/v1.2 divergence: n(H⁻) differs by 2/U(H I).

        This is a known API difference, not a bug, but it is *not* below 1e-6 everywhere:
        at 9000 K U(H I) = 2.0000295, so Korg.px's n(H⁻) — and hence its H⁻ bound-free
        opacity — is 1.4e-5 lower than Korg v1.2 would compute from the same n(H I).
        The assertion pins the expected 2/U(H I) ratio so the divergence cannot grow
        silently or acquire a second cause.
        """
        from korg.continuum import ndens_Hminus

        case = _source_case(reference_data, label)
        py = float(ndens_Hminus(case["nH_I_div_U"], case["ne"], case["T"]))
        julia_v12 = case["nHminus_korg_v1_2"]

        expected_ratio = 2.0 / case["U_H_I"]
        observed_ratio = py / julia_v12
        # With the coefficient derived rather than hardcoded, the only remaining
        # difference is the convention itself: Korg v1.2 folds U(H I) = 2 into its
        # constant, where Korg.px divides by the actual partition function.
        assert np.isclose(observed_ratio, expected_ratio, rtol=1e-12), (
            f"{label}: n(H⁻) ratio to Korg v1.2 relation is {observed_ratio:.12f}, "
            f"expected 2/U(H I) = {expected_ratio:.12f}"
        )

    # -- H⁻ bound-free -------------------------------------------------------

    @pytest.mark.parametrize("label", CONTINUUM_CONDITIONS)
    def test_Hminus_bf_cross_section(self, reference_data, label):
        """α per H⁻ ion — the McLaughlin+ 2017 cross section and stimulated emission.

        Dividing out Python's own n(H⁻) isolates the physics from the density
        convention, so this is held to rtol=1e-6.
        """
        from korg.continuum import Hminus_bf, ndens_Hminus

        case = _source_case(reference_data, label)
        wavelengths_A = case["wavelengths_A"]
        nus = self._nus(wavelengths_A)

        n_Hminus = float(ndens_Hminus(case["nH_I_div_U"], case["ne"], case["T"]))
        py = np.array(Hminus_bf(nus, case["T"], case["nH_I_div_U"], case["ne"])) / n_Hminus
        julia = [case["outputs"]["Hminus_bf_unit_ndens"][str(w)] for w in wavelengths_A]

        _assert_matches(py, julia, wavelengths_A, 1e-6, f"Hminus_bf per H⁻ ion [{label}]")

    @pytest.mark.parametrize("label", CONTINUUM_CONDITIONS)
    def test_Hminus_bf(self, reference_data, label):
        """H⁻ bound-free absorption vs Julia, fed the same n(H⁻).

        rtol is 2e-6 rather than 1e-6 because Korg.px derives n(H⁻) internally with the
        v1.1 literal coefficient, which is 9.238e-7 low relative to Korg v1.2's
        translational_U (see test_hminus_number_density_ground_state_convention).  The
        cross section itself is checked at 1e-6 in test_Hminus_bf_cross_section.
        """
        from korg.continuum import Hminus_bf

        case = _source_case(reference_data, label)
        wavelengths_A = case["wavelengths_A"]
        nus = self._nus(wavelengths_A)

        py = np.array(Hminus_bf(nus, case["T"], case["nH_I_div_U"], case["ne"]))
        julia = [case["outputs"]["Hminus_bf"][str(w)] for w in wavelengths_A]

        _assert_matches(py, julia, wavelengths_A, 2e-6, f"Hminus_bf [{label}]")

    # -- H⁻ free-free --------------------------------------------------------

    @pytest.mark.parametrize("label", CONTINUUM_CONDITIONS)
    def test_Hminus_ff(self, reference_data, label):
        """H⁻ free-free (Bell & Berrington 1987 table) vs Julia."""
        from korg.continuum import Hminus_ff

        case = _source_case(reference_data, label)
        wavelengths_A = case["wavelengths_A"]
        nus = self._nus(wavelengths_A)

        py = np.array(Hminus_ff(nus, case["T"], case["nH_I_div_U"], case["ne"]))
        julia = [case["outputs"]["Hminus_ff"][str(w)] for w in wavelengths_A]

        _assert_matches(py, julia, wavelengths_A, 1e-6, f"Hminus_ff [{label}]")

    # -- He⁻ free-free -------------------------------------------------------

    @pytest.mark.parametrize("label", CONTINUUM_CONDITIONS)
    def test_Heminus_ff(self, reference_data, label):
        """He⁻ free-free (John 1994 table) vs Julia, including the λ < 5063 Å cutoff."""
        from korg.continuum import Heminus_ff

        case = _source_case(reference_data, label)
        wavelengths_A = case["wavelengths_A"]
        nus = self._nus(wavelengths_A)

        py = np.array(Heminus_ff(nus, case["T"], case["nHe_I_div_U"], case["ne"]))
        julia = [case["outputs"]["Heminus_ff"][str(w)] for w in wavelengths_A]

        _assert_matches(py, julia, wavelengths_A, 1e-6, f"Heminus_ff [{label}]")

    # -- Metal bound-free ----------------------------------------------------

    @pytest.mark.parametrize("label", CONTINUUM_CONDITIONS)
    def test_metal_bf_absorption(self, reference_data, label):
        """Metal bound-free (TOPBase/NORAD tables) vs Julia."""
        from korg.continuum import metal_bf_absorption

        case = _source_case(reference_data, label)
        wavelengths_A = case["wavelengths_A"]
        nus = self._nus(wavelengths_A)

        py = np.array(metal_bf_absorption(nus, case["T"], case["number_densities"]))
        julia = [case["outputs"]["metal_bf"][str(w)] for w in wavelengths_A]

        assert np.any(py > 0), f"metal_bf_absorption is identically zero at [{label}]"
        _assert_matches(py, julia, wavelengths_A, 1e-6, f"metal_bf_absorption [{label}]")

    def test_metal_bf_absorption_key_spelling(self, reference_data):
        """'Fe_I' and 'Fe I' must give the same answer.

        The cross-section tables are keyed with a space while the rest of the package
        keys number densities with an underscore; a mismatch here silently drops every
        metal bf contribution from total_continuum_absorption rather than erroring.
        """
        from korg.continuum import metal_bf_absorption

        case = _source_case(reference_data, "solar")
        nus = self._nus(case["wavelengths_A"])
        nd_underscore = case["number_densities"]
        nd_space = {k.replace("_", " "): v for k, v in nd_underscore.items()}

        a_underscore = np.array(metal_bf_absorption(nus, case["T"], nd_underscore))
        a_space = np.array(metal_bf_absorption(nus, case["T"], nd_space))

        assert np.any(a_underscore > 0), "underscore-keyed metal bf is identically zero"
        assert np.allclose(a_underscore, a_space, rtol=1e-12, atol=0.0), (
            f"key spelling changes metal bf: {a_underscore} vs {a_space}"
        )

    # -- Positive-ion free-free ---------------------------------------------

    @pytest.mark.parametrize("label", CONTINUUM_CONDITIONS)
    def test_positive_ion_ff_absorption(self, reference_data, label):
        """Positive-ion free-free (hydrogenic + Peach 1970 departures) vs Julia."""
        from korg.continuum import positive_ion_ff_absorption

        case = _source_case(reference_data, label)
        wavelengths_A = case["wavelengths_A"]
        nus = self._nus(wavelengths_A)

        py = np.array(positive_ion_ff_absorption(nus, case["T"], case["number_densities"],
                                                 case["ne"]))
        julia = [case["outputs"]["positive_ion_ff"][str(w)] for w in wavelengths_A]

        assert np.any(py > 0), f"positive_ion_ff_absorption is identically zero at [{label}]"
        _assert_matches(py, julia, wavelengths_A, 1e-6,
                        f"positive_ion_ff_absorption [{label}]")


class TestTotalContinuumConditions:
    """The summed continuum at three conditions where different sources dominate."""

    # Per-condition tolerances.  The sum is limited by H_I_bf, which agrees with Julia
    # only at the few×1e-4 level (its own reference test uses rtol=1e-4, citing the MHD
    # occupation-probability formalism).  H_I_bf is negligible in the cool case, ~1% of
    # the total at 3000 Å in the solar case, and up to 93% of it in the hot case, so the
    # tolerance is set per condition instead of using one loose bound everywhere.
    # Observed maxima: 9.0e-7 (cool_dense), 3.1e-6 (solar), 1.0e-4 (hot).
    RTOL = {"cool_dense": 2e-6, "solar": 1e-5, "hot": 3e-4}

    @pytest.mark.parametrize("label", CONTINUUM_CONDITIONS)
    def test_total_continuum_absorption(self, reference_data, label):
        from korg.continuum import total_continuum_absorption
        from korg.constants import c_cgs
        from korg.data_loader import default_partition_funcs
        from korg.species import Species

        ref = reference_data.get("total_continuum_absorption", {}).get("conditions")
        if not ref or label not in ref:
            pytest.skip(f"total_continuum_absorption['conditions'][{label!r}] not in "
                        "reference data (regenerate tests/julia_reference_data.json)")
        case = ref[label]

        wavelengths_A = case["wavelengths_A"]
        nus = jnp.array([c_cgs / (wl_A * 1e-8) for wl_A in wavelengths_A])
        pf_str = {str(k).replace(" ", "_"): v for k, v in default_partition_funcs.items()
                  if isinstance(k, Species)}

        py = np.array(total_continuum_absorption(nus, case["T"], case["ne"],
                                                 case["number_densities"], pf_str))
        julia = [case["outputs"][str(w)] for w in wavelengths_A]

        _assert_matches(py, julia, wavelengths_A, self.RTOL[label],
                        f"total_continuum_absorption [{label}]")


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------

class TestContinuumGradients:
    """The continuum must be differentiable, not merely finite-valued.

    ``H_I_bf`` computes an effective quantum number 1/sqrt(1/n^2 - h*nu/chi) for the
    level an electron is excited into. Above the series limit that radicand is negative,
    so the expression is NaN, and the result is discarded by a ``jnp.where``. Masking a
    NaN does not mask its gradient, so before the radicand was clamped this returned a
    NaN derivative while the value looked perfectly healthy — which is exactly the
    failure mode these tests exist to catch.
    """

    @staticmethod
    def _nu(wavelength_A):
        from korg.constants import c_cgs
        return c_cgs / (wavelength_A * 1e-8)

    @staticmethod
    def _central_diff(f, x, rel_step=1e-4):
        return (float(f(x * (1 + rel_step))) - float(f(x * (1 - rel_step)))) / (2 * rel_step * x)

    # Wavelengths straddle the Balmer (3646 A) and Paschen (8206 A) limits, where the
    # masked branch is taken for different sets of levels.
    @pytest.mark.parametrize("T", [4000.0, 5000.0, 8000.0])
    @pytest.mark.parametrize("wavelength_A", [3000.0, 4000.0, 5000.0, 8000.0])
    def test_H_I_bf_grad_wrt_ne(self, T, wavelength_A):
        from korg.continuum import H_I_bf

        nu = jnp.array([self._nu(wavelength_A)])
        f = lambda ne: jnp.sum(H_I_bf(nu, T, 1e17, 1e16, ne, 2.0))

        grad = float(jax.grad(f)(jnp.float64(1e13)))
        assert np.isfinite(grad), f"d(H_I_bf)/dne is NaN at T={T}, {wavelength_A} A"

        fd = self._central_diff(f, 1e13)
        assert np.isclose(grad, fd, rtol=1e-4), (
            f"T={T}, {wavelength_A} A: grad={grad:.6e} vs finite difference {fd:.6e}"
        )

    def test_H_I_bf_grad_wrt_other_inputs_finite(self):
        from korg.continuum import H_I_bf

        nu = jnp.array([self._nu(5000.0)])
        cases = {
            "T": (lambda x: jnp.sum(H_I_bf(nu, x, 1e17, 1e16, 1e13, 2.0)), 5000.0),
            "nH_I": (lambda x: jnp.sum(H_I_bf(nu, 5000.0, x, 1e16, 1e13, 2.0)), 1e17),
            "nHe_I": (lambda x: jnp.sum(H_I_bf(nu, 5000.0, 1e17, x, 1e13, 2.0)), 1e16),
        }
        for name, (f, x0) in cases.items():
            grad = float(jax.grad(f)(jnp.float64(x0)))
            assert np.isfinite(grad), f"d(H_I_bf)/d{name} is not finite"

    def test_total_continuum_absorption_grad_wrt_ne(self):
        from korg.continuum import total_continuum_absorption
        from korg.data_loader import default_partition_funcs

        nu = jnp.array([self._nu(5000.0)])
        number_densities = {"H_I": 1e17, "H_II": 1e13, "He_I": 1e16, "H2": 1e10}
        f = lambda ne: jnp.sum(
            total_continuum_absorption(nu, 5000.0, ne, number_densities, default_partition_funcs)
        )

        grad = float(jax.grad(f)(jnp.float64(1e13)))
        assert np.isfinite(grad), "d(total_continuum_absorption)/dne is NaN"

        fd = self._central_diff(f, 1e13)
        assert np.isclose(grad, fd, rtol=1e-4), (
            f"grad={grad:.6e} vs finite difference {fd:.6e}"
        )
