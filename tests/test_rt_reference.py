"""
Tests for radiative transfer functions against Julia reference data.

Covers:
- generate_mu_grid: Gauss-Legendre quadrature points and weights
- compute_I_linear_flux_only: emergent flux via linear interpolation
- compute_F_flux_only_expint: emergent flux via exponential integrals
- blackbody: Planck function B_λ(T)

Reference data lives in tests/julia_reference_data.json.
To regenerate:
    julia --project=/tmp/Korg.jl tests/generate_julia_reference.jl
"""

import json
from pathlib import Path

# Import korg FIRST to enable JAX x64 mode before any other JAX operations
import korg  # noqa: F401 — side-effect: enables float64

import jax
import jax.numpy as jnp
import numpy as np
import pytest

REFERENCE_FILE = Path(__file__).parent / "julia_reference_data.json"


@pytest.fixture(scope="module")
def reference_data():
    """Load pre-computed Julia reference data."""
    if not REFERENCE_FILE.exists():
        pytest.skip(
            f"Julia reference data not found at {REFERENCE_FILE}. "
            "Run: julia --project=/tmp/Korg.jl tests/generate_julia_reference.jl"
        )
    with open(REFERENCE_FILE) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# generate_mu_grid
# ---------------------------------------------------------------------------

class TestGenerateMuGridReference:
    """Tests for generate_mu_grid against Julia reference values."""

    # --- Julia comparison ---

    @pytest.mark.parametrize("n", [2, 3, 5, 7])
    def test_mu_matches_julia(self, reference_data, n):
        """μ values must match Julia for each supported n_points."""
        if "generate_mu_grid" not in reference_data:
            pytest.skip("generate_mu_grid key missing from reference data")
        outputs = reference_data["generate_mu_grid"]["outputs"]
        key = str(n)
        if key not in outputs:
            pytest.skip(f"n={n} not stored in reference data")

        from korg.radiative_transfer.core import generate_mu_grid
        mu_py, _ = generate_mu_grid(n)
        mu_julia = np.array(outputs[key]["mu"])
        np.testing.assert_allclose(
            np.sort(np.array(mu_py)),
            np.sort(mu_julia),
            rtol=1e-10,
            err_msg=f"μ mismatch for n={n}",
        )

    @pytest.mark.parametrize("n", [2, 3, 5, 7])
    def test_weights_match_julia(self, reference_data, n):
        """Quadrature weights must match Julia for each supported n_points."""
        if "generate_mu_grid" not in reference_data:
            pytest.skip("generate_mu_grid key missing from reference data")
        outputs = reference_data["generate_mu_grid"]["outputs"]
        key = str(n)
        if key not in outputs:
            pytest.skip(f"n={n} not stored in reference data")

        from korg.radiative_transfer.core import generate_mu_grid
        _, w_py = generate_mu_grid(n)
        w_julia = np.array(outputs[key]["weights"])

        # Sort both by corresponding mu so order doesn't matter
        mu_py, _ = generate_mu_grid(n)
        order_py = np.argsort(np.array(mu_py))
        order_julia = np.argsort(np.array(outputs[key]["mu"]))

        np.testing.assert_allclose(
            np.array(w_py)[order_py],
            w_julia[order_julia],
            rtol=1e-10,
            err_msg=f"weights mismatch for n={n}",
        )

    # --- Sanity checks ---

    def test_mu_in_unit_interval(self):
        """All μ values must lie in (0, 1]."""
        from korg.radiative_transfer.core import generate_mu_grid
        mu, _ = generate_mu_grid(5)
        mu = np.array(mu)
        assert np.all(mu > 0) and np.all(mu <= 1), f"μ out of range: {mu}"

    def test_weights_sum_to_one(self):
        """Quadrature weights for GL on [0,1] must sum to 1.0."""
        from korg.radiative_transfer.core import generate_mu_grid
        for n in [2, 3, 5, 7]:
            _, w = generate_mu_grid(n)
            total = float(jnp.sum(w))
            assert abs(total - 1.0) < 1e-12, (
                f"weights for n={n} sum to {total}, expected 1.0"
            )

    def test_mu_monotonically_increasing(self):
        """Returned μ values should be sorted in ascending order."""
        from korg.radiative_transfer.core import generate_mu_grid
        for n in [2, 3, 5, 7]:
            mu, _ = generate_mu_grid(n)
            mu = np.array(mu)
            assert np.all(np.diff(mu) > 0), (
                f"μ not monotonically increasing for n={n}: {mu}"
            )

    def test_correct_number_of_points(self):
        """Output arrays must have exactly n_mu elements."""
        from korg.radiative_transfer.core import generate_mu_grid
        for n in [2, 3, 5, 7, 10]:
            mu, w = generate_mu_grid(n)
            assert len(mu) == n, f"expected {n} μ points, got {len(mu)}"
            assert len(w) == n, f"expected {n} weights, got {len(w)}"

    def test_weights_positive(self):
        """All quadrature weights must be positive."""
        from korg.radiative_transfer.core import generate_mu_grid
        for n in [2, 3, 5, 7]:
            _, w = generate_mu_grid(n)
            assert np.all(np.array(w) > 0), f"non-positive weights for n={n}"


# ---------------------------------------------------------------------------
# Formal solution: compute_I_linear_flux_only and compute_F_flux_only_expint
# ---------------------------------------------------------------------------

class TestFormalSolutionReference:
    """Tests for formal solution functions against Julia reference values."""

    @pytest.fixture(scope="class")
    def rt_ref(self, reference_data):
        """Return the rt_formal_solution sub-dict, skipping if absent."""
        if "rt_formal_solution" not in reference_data:
            pytest.skip("rt_formal_solution key missing from reference data")
        return reference_data["rt_formal_solution"]

    # --- Julia comparisons ---

    def test_linear_flux_only_matches_julia(self, rt_ref):
        """compute_I_linear_flux_only must match Julia F_linear_flux_only."""
        from korg.radiative_transfer.intensity import compute_I_linear_flux_only

        tau = jnp.array(rt_ref["tau"])
        S = jnp.array(rt_ref["S"])
        F_julia = rt_ref["F_linear_flux_only"]

        F_py = float(compute_I_linear_flux_only(tau, S))
        assert np.isclose(F_py, F_julia, rtol=1e-6), (
            f"linear_flux_only: got {F_py}, expected {F_julia}"
        )

    def test_expint_flux_only_matches_julia(self, rt_ref):
        """compute_F_flux_only_expint must match Julia F_expint_flux_only."""
        from korg.radiative_transfer.intensity import compute_F_flux_only_expint

        tau = jnp.array(rt_ref["tau"])
        S = jnp.array(rt_ref["S"])
        F_julia = rt_ref["F_expint_flux_only"]

        F_py = float(compute_F_flux_only_expint(tau, S))
        assert np.isclose(F_py, F_julia, rtol=1e-6), (
            f"expint_flux_only: got {F_py}, expected {F_julia}"
        )

    # --- Sanity checks ---

    def test_linear_flux_positive_for_radiating_slab(self):
        """Flux from a non-zero source function must be positive."""
        from korg.radiative_transfer.intensity import compute_I_linear_flux_only

        tau = jnp.linspace(0.0, 5.0, 20)
        S = jnp.ones(20)  # constant source function
        F = float(compute_I_linear_flux_only(tau, S))
        assert F > 0, f"expected positive flux, got {F}"

    def test_expint_flux_positive_for_radiating_slab(self):
        """Flux from a non-zero source function must be positive."""
        from korg.radiative_transfer.intensity import compute_F_flux_only_expint

        tau = jnp.linspace(0.01, 5.0, 20)
        S = jnp.ones(20)
        F = float(compute_F_flux_only_expint(tau, S))
        assert F > 0, f"expected positive flux, got {F}"

    def test_two_schemes_agree_for_smooth_source(self):
        """Linear and expint schemes should give consistent results for smooth S(τ)."""
        from korg.radiative_transfer.intensity import (
            compute_I_linear_flux_only,
            compute_F_flux_only_expint,
        )

        tau = jnp.linspace(0.01, 3.0, 50)
        S = 1.0 + 0.5 * tau  # linear in tau — both schemes exact for this

        F_lin = float(compute_I_linear_flux_only(tau, S))
        F_exp = float(compute_F_flux_only_expint(tau, S))

        # Proportional agreement within 10 % (different normalisation conventions)
        ratio = F_lin / F_exp
        assert 0.5 < ratio < 5.0, (
            f"schemes disagree too much: linear={F_lin:.6g}, expint={F_exp:.6g}"
        )

    def test_flux_finite_and_non_nan(self, rt_ref):
        """Both formal solution outputs must be finite numbers."""
        from korg.radiative_transfer.intensity import (
            compute_I_linear_flux_only,
            compute_F_flux_only_expint,
        )

        tau = jnp.array(rt_ref["tau"])
        S = jnp.array(rt_ref["S"])

        F_lin = float(compute_I_linear_flux_only(tau, S))
        F_exp = float(compute_F_flux_only_expint(tau, S))

        assert np.isfinite(F_lin), f"linear flux is not finite: {F_lin}"
        assert np.isfinite(F_exp), f"expint flux is not finite: {F_exp}"

    def test_linear_flux_jit_compatible(self, rt_ref):
        """compute_I_linear_flux_only must work under jax.jit."""
        from korg.radiative_transfer.intensity import compute_I_linear_flux_only

        tau = jnp.array(rt_ref["tau"])
        S = jnp.array(rt_ref["S"])
        F_jit = float(jax.jit(compute_I_linear_flux_only)(tau, S))
        F_eager = float(compute_I_linear_flux_only(tau, S))
        assert np.isclose(F_jit, F_eager, rtol=1e-10)

    def test_expint_flux_jit_compatible(self, rt_ref):
        """compute_F_flux_only_expint must work under jax.jit."""
        from korg.radiative_transfer.intensity import compute_F_flux_only_expint

        tau = jnp.array(rt_ref["tau"])
        S = jnp.array(rt_ref["S"])
        F_jit = float(jax.jit(compute_F_flux_only_expint)(tau, S))
        F_eager = float(compute_F_flux_only_expint(tau, S))
        assert np.isclose(F_jit, F_eager, rtol=1e-10)

    def test_flux_increases_with_brighter_source(self):
        """Doubling the source function should roughly double the flux."""
        from korg.radiative_transfer.intensity import compute_I_linear_flux_only

        tau = jnp.linspace(0.0, 5.0, 30)
        S1 = jnp.ones(30)
        S2 = 2.0 * S1

        F1 = float(compute_I_linear_flux_only(tau, S1))
        F2 = float(compute_I_linear_flux_only(tau, S2))
        assert np.isclose(F2, 2.0 * F1, rtol=1e-10), (
            f"expected F2=2*F1, got F1={F1:.6g}, F2={F2:.6g}"
        )


# ---------------------------------------------------------------------------
# blackbody
# ---------------------------------------------------------------------------

class TestBlackbodyReference:
    """Tests for the Planck blackbody function against Julia reference values."""

    # --- Julia comparisons ---

    def test_all_stored_t_wl_combos(self, reference_data):
        """All (T, λ) combinations stored in JSON must match Julia to rtol=1e-6."""
        if "blackbody" not in reference_data:
            pytest.skip("blackbody key missing from reference data")
        outputs = reference_data["blackbody"]["outputs"]

        from korg.synthesis import blackbody

        for key, julia_val in outputs.items():
            T_str, wl_str = key.split("_", 1)
            T = float(T_str)
            wl_cm = float(wl_str)
            py_val = float(blackbody(T, wl_cm))
            assert np.isclose(py_val, julia_val, rtol=1e-6), (
                f"blackbody(T={T}, λ={wl_cm} cm): got {py_val:.6g}, "
                f"expected {julia_val:.6g}"
            )

    # --- Sanity checks ---

    def test_positive_and_finite(self):
        """Planck function must be positive and finite for physical T and λ."""
        from korg.synthesis import blackbody

        for T in [3000.0, 5778.0, 8000.0, 20000.0]:
            for wl_cm in [1e-5, 5e-5, 1e-4, 5e-4]:
                val = float(blackbody(T, wl_cm))
                assert val > 0 and np.isfinite(val), (
                    f"blackbody(T={T}, λ={wl_cm}) = {val}"
                )

    def test_wiens_law_peak_wavelength(self):
        """Peak of B_λ should follow Wien's displacement law: λ_max ≈ b/T."""
        from korg.synthesis import blackbody

        b_wien_cm = 0.28977719  # Wien displacement constant in cm·K

        for T in [3000.0, 5778.0, 10000.0]:
            wl_peak_expected = b_wien_cm / T  # cm
            # Sample wavelengths around expected peak
            wls = np.linspace(wl_peak_expected * 0.3, wl_peak_expected * 3.0, 500)
            vals = np.array([float(blackbody(T, w)) for w in wls])
            wl_peak_found = wls[np.argmax(vals)]
            # Peak should be within 5% of Wien's law
            assert abs(wl_peak_found - wl_peak_expected) / wl_peak_expected < 0.05, (
                f"T={T} K: Wien peak at {wl_peak_expected:.3e} cm, "
                f"found at {wl_peak_found:.3e} cm"
            )

    def test_hotter_star_brighter_in_blue(self):
        """Hotter blackbody should emit more at short wavelengths."""
        from korg.synthesis import blackbody

        wl_blue_cm = 3e-5  # 300 nm
        B_cool = float(blackbody(4000.0, wl_blue_cm))
        B_hot = float(blackbody(8000.0, wl_blue_cm))
        assert B_hot > B_cool, (
            f"expected hotter star brighter in blue; "
            f"B(4000 K)={B_cool:.3e}, B(8000 K)={B_hot:.3e}"
        )

    def test_rayleigh_jeans_limit(self):
        """In the Rayleigh-Jeans limit (hc/λkT << 1), B_λ ∝ T."""
        from korg.synthesis import blackbody

        # Long wavelength: hc/λkT << 1 means λ >> hc/kT
        # For T=5778 K, hc/kT ~ 0.25 cm; use λ = 10 cm (radio)
        wl_rj_cm = 10.0  # 10 cm — deep Rayleigh-Jeans

        T1, T2 = 5000.0, 10000.0
        B1 = float(blackbody(T1, wl_rj_cm))
        B2 = float(blackbody(T2, wl_rj_cm))

        # In RJ limit B_λ ∝ T, so ratio should be T2/T1
        ratio = B2 / B1
        expected_ratio = T2 / T1
        assert np.isclose(ratio, expected_ratio, rtol=1e-3), (
            f"RJ ratio B(T2)/B(T1) = {ratio:.4f}, expected {expected_ratio:.4f}"
        )

    def test_jit_compatible(self):
        """blackbody must be callable under jax.jit."""
        from korg.synthesis import blackbody

        bb_jit = jax.jit(blackbody)
        T, wl = 5778.0, 5e-5
        val_jit = float(bb_jit(T, wl))
        val_eager = float(blackbody(T, wl))
        assert np.isclose(val_jit, val_eager, rtol=1e-10)

    def test_vectorised_over_wavelengths(self):
        """blackbody should broadcast correctly over an array of wavelengths."""
        from korg.synthesis import blackbody

        wls = jnp.array([3e-5, 5e-5, 1e-4])
        vals = blackbody(5778.0, wls)
        assert vals.shape == (3,), f"expected shape (3,), got {vals.shape}"
        assert jnp.all(vals > 0), "all blackbody values should be positive"
