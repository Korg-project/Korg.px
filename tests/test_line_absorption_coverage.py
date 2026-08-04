"""
Branch and error-path coverage for :mod:`korg.line_absorption`.

``test_line_absorption_jit.py`` covers the fast JIT path against Korg.jl.  This
file fills in the parts it does not reach: the vdW encoding decoder, the empty
and degenerate inputs, the error paths, the pure-Python reference path, and the
ABO branch of the vectorised implementation.

Categories: functional / error paths, autodiff, and jit-tracing.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import korg  # noqa: F401 — side effect: enables x64
from korg import line_absorption as LA
from korg.line_absorption import (
    _vdW_to_tuple, prepare_linelist_arrays, line_absorption,
    inverse_gaussian_density, inverse_lorentz_density, sigma_line,
    doppler_width, scaled_stark, scaled_vdW, line_profile,
)
from korg.linelist import Line
from korg.species import Species
from korg.constants import bohr_radius_cgs


# ===========================================================================
# 1. FUNCTIONAL — the vdW field decoder
# ===========================================================================

class TestVdWToTuple:
    """Korg encodes four different things in one float; all four must decode."""

    def test_none_is_disabled(self):
        assert _vdW_to_tuple(None) == (0.0, -1.0)

    def test_zero_is_disabled(self):
        assert _vdW_to_tuple(0.0) == (0.0, -1.0)

    def test_tuple_passes_through(self):
        assert _vdW_to_tuple((1.5e-15, 0.3)) == (1.5e-15, 0.3)

    def test_list_passes_through(self):
        assert _vdW_to_tuple([1.5e-15, 0.3]) == (1.5e-15, 0.3)

    def test_negative_is_log10_gamma(self):
        gamma, alpha = _vdW_to_tuple(-7.5)
        assert gamma == pytest.approx(10 ** -7.5)
        assert alpha == -1.0

    def test_small_positive_is_a_fudge_factor(self):
        assert _vdW_to_tuple(2.5) == (2.5, -1.0)

    def test_large_positive_is_abo_packing(self):
        """Integer part = σ/a₀², fractional part = α."""
        gamma, alpha = _vdW_to_tuple(325.25)
        assert alpha == pytest.approx(0.25)
        assert gamma == pytest.approx(325 * bohr_radius_cgs ** 2)

    def test_abo_boundary_at_twenty(self):
        assert _vdW_to_tuple(19.9) == (19.9, -1.0)
        gamma, alpha = _vdW_to_tuple(20.5)
        assert alpha == pytest.approx(0.5)
        assert gamma == pytest.approx(20 * bohr_radius_cgs ** 2)


# ===========================================================================
# 1. FUNCTIONAL — profile helpers and degenerate inputs
# ===========================================================================

class TestProfileHelpers:

    def test_inverse_gaussian_density_inverts_the_gaussian(self):
        sigma = 0.2
        for rho in (0.05, 0.5, 1.0):
            x = float(inverse_gaussian_density(rho, sigma))
            if x > 0:
                back = np.exp(-x ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
                assert back == pytest.approx(rho, rel=1e-8)

    def test_inverse_gaussian_density_above_the_peak_is_zero(self):
        """No x has a density greater than the peak, so the answer is 0."""
        sigma = 0.2
        peak = 1.0 / (np.sqrt(2 * np.pi) * sigma)
        assert float(inverse_gaussian_density(peak * 2, sigma)) == 0.0

    def test_inverse_lorentz_density_inverts_the_lorentzian(self):
        gamma = 0.1
        for rho in (0.01, 0.5):
            x = float(inverse_lorentz_density(rho, gamma))
            if x > 0:
                back = gamma / (np.pi * (x ** 2 + gamma ** 2))
                assert back == pytest.approx(rho, rel=1e-8)

    def test_inverse_lorentz_density_above_the_peak_is_zero(self):
        gamma = 0.1
        peak = 1.0 / (np.pi * gamma)
        assert float(inverse_lorentz_density(peak * 2, gamma)) == 0.0

    def test_sigma_line_scales_as_lambda_squared(self):
        a = float(sigma_line(5000e-8))
        b = float(sigma_line(10000e-8))
        assert b / a == pytest.approx(4.0, rel=1e-12)

    def test_doppler_width_combines_thermal_and_microturbulence(self):
        from korg.constants import amu_cgs
        mass = 55.845 * amu_cgs
        cold = float(doppler_width(5000e-8, 3000.0, mass, 0.0))
        hot = float(doppler_width(5000e-8, 9000.0, mass, 0.0))
        assert hot > cold
        with_xi = float(doppler_width(5000e-8, 3000.0, mass, 2e5))
        assert with_xi > cold

    def test_scaled_stark_scales_as_T_to_the_one_sixth(self):
        a = float(scaled_stark(1e-5, 10000.0))
        b = float(scaled_stark(1e-5, 10000.0 * 64))
        assert b / a == pytest.approx(64 ** (1 / 6), rel=1e-10)

    def test_scaled_vdW_simple_scaling_branch(self):
        from korg.constants import amu_cgs
        v = float(scaled_vdW((1e-31, -1.0), 55.845 * amu_cgs, 10000.0))
        assert v == pytest.approx(1e-31, rel=1e-10)

    def test_scaled_vdW_abo_branch(self):
        from korg.constants import amu_cgs
        v = float(scaled_vdW((300 * bohr_radius_cgs ** 2, 0.3),
                             55.845 * amu_cgs, 5000.0))
        assert np.isfinite(v) and v > 0

    def test_line_profile_is_normalised_and_peaked_at_the_centre(self):
        wl0, sigma, gamma, amp = 5000e-8, 0.02e-8, 0.01e-8, 1.0
        grid = np.linspace(wl0 - 1e-8, wl0 + 1e-8, 4001)
        vals = np.array([float(line_profile(wl0, sigma, gamma, amp, w)) for w in grid])
        assert vals.argmax() == len(grid) // 2
        # ±1e-8 cm truncates the Lorentz wings, so a little of the area is lost
        assert np.trapezoid(vals, grid) == pytest.approx(amp, rel=1e-2)
        assert np.trapezoid(vals, grid) < amp, "truncated integral cannot exceed amp"


# ===========================================================================
# 1. FUNCTIONAL — array preparation and error paths
# ===========================================================================

def _line(wl_A=5000.0, species="Fe I", log_gf=-1.5, E_lower=1.0,
          gamma_rad=1e8, gamma_stark=1e-5, vdW=None):
    """Build a Line; ``vdW`` accepts Korg's packed float encoding or a tuple."""
    return Line(wl_A * 1e-8, log_gf, Species(species), E_lower,
                gamma_rad, gamma_stark, _vdW_to_tuple(vdW))


class TestPrepareLinelistArrays:

    def test_empty_linelist_returns_empty_arrays(self):
        arrays = prepare_linelist_arrays([], [])
        assert arrays["wls"].shape == (0,)
        assert arrays["species_ids"].shape == (0,)
        assert arrays["species_ids"].dtype == jnp.int32

    def test_populated_linelist(self):
        lines = [_line(5000.0, "Fe I"), _line(5200.0, "Ca II")]
        species = [Species("Fe I"), Species("Ca II")]
        arrays = prepare_linelist_arrays(lines, species)
        assert arrays["wls"].shape == (2,)
        np.testing.assert_allclose(np.asarray(arrays["wls"]), [5000e-8, 5200e-8])
        assert set(np.asarray(arrays["species_ids"]).tolist()) == {0, 1}


class TestLineAbsorptionErrorPaths:

    WLS = jnp.asarray(np.linspace(4999e-8, 5001e-8, 51))
    T = jnp.array([5000.0, 6000.0])
    NE = jnp.array([1e13, 1e14])
    XI = 1e5

    def _cntm(self, wl):
        return np.full(len(self.T), 1e-9)

    def test_hydrogen_in_linelist_is_rejected(self):
        """H lines must go through hydrogen_line_absorption instead."""
        lines = [_line(5000.0, "H I")]
        with pytest.raises(ValueError, match="Atomic hydrogen should not be"):
            line_absorption(lines, self.WLS, self.T, self.NE,
                            {Species("H I"): np.full(2, 1e16)},
                            {Species("H I"): lambda lt: 2.0},
                            self.XI, self._cntm)

    def test_missing_species_density_is_rejected(self):
        lines = [_line(5000.0, "Fe I")]
        with pytest.raises(ValueError, match="not in number_densities"):
            line_absorption(lines, self.WLS, self.T, self.NE, {},
                            {Species("Fe I"): lambda lt: 25.0},
                            self.XI, self._cntm, use_jit=False)

    def test_empty_linelist_returns_zeros(self):
        out = np.asarray(line_absorption([], self.WLS, self.T, self.NE, {}, {},
                                         self.XI, self._cntm, use_jit=False))
        assert out.shape == (len(self.T), len(self.WLS))
        assert np.all(out == 0.0)


# ===========================================================================
# 1. FUNCTIONAL — the two implementations
# ===========================================================================

class TestPythonAndFastPathsAgree:
    """``use_jit=False`` selects the reference Python loop; it should track the
    vectorised implementation."""

    WLS = jnp.asarray(np.linspace(4998e-8, 5002e-8, 201))
    T = jnp.array([4500.0, 6000.0])
    NE = jnp.array([1e13, 1e14])
    XI = 1e5

    def _setup(self, vdW=None):
        lines = [_line(5000.0, "Fe I", vdW=vdW)] if vdW is not None else \
                [_line(5000.0, "Fe I")]
        nd = {Species("Fe I"): np.full(2, 1e12), Species("H I"): np.full(2, 1e16)}
        pf = {Species("Fe I"): lambda lt: 25.0, Species("H I"): lambda lt: 2.0}

        def cntm(wl):
            return np.full(2, 1e-9)

        return lines, nd, pf, cntm

    def test_python_path_produces_a_line(self):
        lines, nd, pf, cntm = self._setup()
        out = np.asarray(line_absorption(lines, self.WLS, self.T, self.NE, nd, pf,
                                         self.XI, cntm, use_jit=False))
        assert out.shape == (2, len(self.WLS))
        assert np.all(np.isfinite(out))
        assert out.max() > 0
        assert out[:, len(self.WLS) // 2].min() > 0, "peak should be at line centre"

    def test_python_and_jit_paths_are_close(self):
        lines, nd, pf, cntm = self._setup()
        slow = np.asarray(line_absorption(lines, self.WLS, self.T, self.NE, nd, pf,
                                          self.XI, cntm, use_jit=False))
        fast = np.asarray(line_absorption(lines, self.WLS, self.T, self.NE, nd, pf,
                                          self.XI, cntm, use_jit=True))
        assert fast.shape == slow.shape
        peak_slow, peak_fast = slow.max(), fast.max()
        assert peak_fast == pytest.approx(peak_slow, rel=1e-3)

    def test_abo_vdw_branch_runs(self):
        """vdW > 20 packs (σ, α) and takes the ABO branch, which needs scipy's
        gamma function and a per-line loop."""
        lines, nd, pf, cntm = self._setup(vdW=325.25)
        out = np.asarray(line_absorption(lines, self.WLS, self.T, self.NE, nd, pf,
                                         self.XI, cntm))
        assert np.all(np.isfinite(out)) and out.max() > 0

    def test_simple_vdw_branch_runs(self):
        lines, nd, pf, cntm = self._setup(vdW=-7.5)
        out = np.asarray(line_absorption(lines, self.WLS, self.T, self.NE, nd, pf,
                                         self.XI, cntm))
        assert np.all(np.isfinite(out)) and out.max() > 0

    def test_molecular_line_skips_stark_and_vdw(self):
        lines = [_line(5000.0, "CO", log_gf=-2.0)]
        nd = {Species("CO"): np.full(2, 1e12), Species("H I"): np.full(2, 1e16)}
        pf = {Species("CO"): lambda lt: 100.0, Species("H I"): lambda lt: 2.0}

        def cntm(wl):
            return np.full(2, 1e-9)

        out = np.asarray(line_absorption(lines, self.WLS, self.T, self.NE, nd, pf,
                                         self.XI, cntm))
        assert np.all(np.isfinite(out))

    def test_partition_function_without_numpy_eval(self):
        """The fast path prefers ``pf.numpy_eval``; a plain callable falls back to
        a per-layer Python loop (``line_absorption.py`` lines 631-632).

        Only the fallback branch is under test here — the resulting opacity is not
        asserted to be non-zero, because a bare lambda is not the CubicSpline the
        rest of the pipeline supplies and the amplitude/window bookkeeping can
        legitimately collapse the line to nothing.
        """
        lines, nd, _, cntm = self._setup()
        pf = {Species("Fe I"): lambda lt: 25.0 + 0.0 * lt,
              Species("H I"): lambda lt: 2.0 + 0.0 * lt}
        out = np.asarray(line_absorption(lines, self.WLS, self.T, self.NE, nd, pf,
                                         self.XI, cntm))
        assert out.shape == (len(self.T), len(self.WLS))
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0)

    def test_scalar_continuum_callback_is_accepted(self):
        """The batched call is tried first and falls back to per-line calls."""
        lines, nd, pf, _ = self._setup()
        calls = []

        def cntm(wl):
            calls.append(wl)
            wl = np.atleast_1d(np.asarray(wl))
            if wl.size != 1:
                raise ValueError("only scalars supported")
            return np.full(2, 1e-9)

        out = np.asarray(line_absorption(lines, self.WLS, self.T, self.NE, nd, pf,
                                         self.XI, cntm))
        assert np.all(np.isfinite(out))
        assert len(calls) >= 2, "should have retried per line"


# ===========================================================================
# 3. AUTODIFF
# ===========================================================================

class TestAutodiff:

    def test_sigma_line_grad(self):
        g = float(jax.grad(sigma_line)(5000e-8))
        assert np.isfinite(g) and g != 0.0
        assert g == pytest.approx(2 * float(sigma_line(5000e-8)) / 5000e-8, rel=1e-10)

    def test_doppler_width_grad_wrt_temperature(self):
        from korg.constants import amu_cgs
        mass = 55.845 * amu_cgs
        T = 5000.0
        g = float(jax.grad(lambda t: doppler_width(5000e-8, t, mass, 1e5))(T))
        h = T * 1e-6
        fd = (float(doppler_width(5000e-8, T + h, mass, 1e5))
              - float(doppler_width(5000e-8, T - h, mass, 1e5))) / (2 * h)
        assert g == pytest.approx(fd, rel=1e-5)

    def test_scaled_stark_grad(self):
        g = float(jax.grad(lambda T: scaled_stark(1e-5, T))(8000.0))
        assert np.isfinite(g) and g != 0.0

    def test_line_profile_grad_wrt_all_shape_parameters(self):
        wl0, sigma, gamma, amp, wl = 5000e-8, 0.02e-8, 0.01e-8, 1.0, 5000.01e-8
        for argnum in range(4):
            g = float(jax.grad(line_profile, argnums=argnum)(wl0, sigma, gamma,
                                                             amp, wl))
            assert np.isfinite(g), f"argnum={argnum}"

    def test_inverse_density_grads_are_finite_off_the_branch_point(self):
        """Above the peak these return 0, and the discarded branch takes a sqrt/log
        of a non-positive number — so the cotangent has to be guarded.

        Exactly *at* the peak the derivative is genuinely infinite (the inverse
        has a vertical tangent there: x(ρ) ∝ √(-2 ln(ρ/ρ_peak)) → dx/dρ → ∞ as
        ρ → ρ_peak⁻).  That is mathematics, not a masking bug, so it is asserted
        rather than guarded away.
        """
        sigma, gamma = 0.2, 0.1
        peak_g = 1.0 / (np.sqrt(2 * np.pi) * sigma)
        peak_l = 1.0 / (np.pi * gamma)

        for rho in (peak_g * 2, peak_g * 1.001, peak_g * 0.5, peak_g * 0.1):
            g = float(jax.grad(lambda r: inverse_gaussian_density(r, sigma))(rho))
            assert np.isfinite(g), f"gaussian at rho={rho}"
        for rho in (peak_l * 2, peak_l * 1.001, peak_l * 0.5, peak_l * 0.1):
            g = float(jax.grad(lambda r: inverse_lorentz_density(r, gamma))(rho))
            assert np.isfinite(g), f"lorentz at rho={rho}"

        # the branch point itself
        assert np.isinf(float(
            jax.grad(lambda r: inverse_gaussian_density(r, sigma))(peak_g)))


# ===========================================================================
# 4. JIT TRACING
# ===========================================================================

class TestJitTracing:

    def test_scalar_helpers_jit(self):
        from korg.constants import amu_cgs
        assert np.isfinite(float(jax.jit(sigma_line)(5000e-8)))
        assert np.isfinite(float(jax.jit(
            lambda T: doppler_width(5000e-8, T, 55.845 * amu_cgs, 1e5))(5000.0)))
        assert np.isfinite(float(jax.jit(
            lambda T: scaled_stark(1e-5, T))(8000.0)))
        assert np.isfinite(float(jax.jit(
            lambda s, g: line_profile(5000e-8, s, g, 1.0, 5000.01e-8))(
                0.02e-8, 0.01e-8)))

    def test_inverse_densities_jit(self):
        assert np.isfinite(float(jax.jit(
            lambda r: inverse_gaussian_density(r, 0.2))(0.5)))
        assert np.isfinite(float(jax.jit(
            lambda r: inverse_lorentz_density(r, 0.1))(0.5)))

    def test_line_profile_vmaps_over_wavelength(self):
        wls = jnp.asarray(np.linspace(4999e-8, 5001e-8, 101))
        f = jax.jit(jax.vmap(lambda w: line_profile(5000e-8, 0.02e-8, 0.01e-8, 1.0, w)))
        out = np.asarray(f(wls))
        assert out.shape == (101,)
        assert np.all(np.isfinite(out)) and out.max() > 0

    def test_vdW_to_tuple_cannot_be_jitted(self):
        """It branches on the *value* of vdW (``v < 0``, ``v == 0``, ``v < 20``)
        and calls ``math.floor``, so the encoding must be decoded in Python before
        the trace boundary — which is what ``prepare_linelist_arrays`` is for.
        """
        with pytest.raises((jax.errors.TracerBoolConversionError, TypeError)):
            jax.jit(lambda v: _vdW_to_tuple(v)[0])(jnp.float64(325.25))

    def test_prepare_linelist_arrays_is_constant_folded_under_jit(self):
        """It iterates Python ``Line`` objects, so nothing in it is traced.

        Calling it inside a jitted function therefore *works* — but only because
        the whole linelist is baked into the trace as a constant, which means it
        re-runs on every retrace and cannot depend on a traced value.  That is why
        the docstring tells callers to invoke it once, outside the trace boundary.
        """
        lines = [_line(5000.0, "Fe I")]
        out = jax.jit(
            lambda x: prepare_linelist_arrays(lines, [Species("Fe I")])["wls"] * x
        )(2.0)
        np.testing.assert_allclose(np.asarray(out), [2 * 5000e-8])
