"""
Autodiff and jit-tracing behaviour for Korg.px's utility modules.

Three things are pinned here:

1. Where gradients exist, ``jax.grad`` must be finite, non-zero and agree with
   central finite differences to ~1e-5 relative.
2. Where ``jax.jit`` works, it must keep working.
3. Where ``jax.jit`` *cannot* work, the failure is pinned with
   ``pytest.raises`` so that it is a documented boundary rather than a
   surprise.

The spline gradient quirk at the clamped domain endpoints (JAX's tie-splitting
makes ``grad`` exactly half the one-sided interior derivative) is pinned in
``test_autodiff_wavelengths_utils.py`` and is deliberately not duplicated here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from korg.cubic_splines import CubicSpline, cubic_spline
from korg.radiative_transfer import core as rt_core

jax.config.update("jax_enable_x64", True)


def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)


def assert_finite(value, what):
    array = np.asarray(value)
    assert np.all(np.isfinite(array)), f"{what} is not finite: {array}"


# ===========================================================================
# Cubic splines
# ===========================================================================


@pytest.fixture(scope="module")
def spline_data():
    t = np.linspace(0.0, 5.0, 9)
    return t, t**2


class TestCubicSplineAutodiffAndJit:
    def test_gradient_with_respect_to_knot_values(self, spline_data):
        """d(spline)/du: the spline is linear in ``u``, so this is exact.

        ``cubic_spline`` builds the tridiagonal solve in NumPy/SciPy, so the
        construction is not traceable; what *is* traceable is evaluating a
        ``CubicSpline`` whose ``u`` and ``z`` are JAX arrays. Differentiating
        through both together is what a downstream fit would need.
        """
        t, u = spline_data
        base = cubic_spline(t, u, extrapolate=True)

        def evaluate(scale):
            spline = CubicSpline(base.t, base.u * scale, base.h,
                                 base.z * scale, extrapolate=True)
            return spline(2.3)

        grad = float(jax.grad(evaluate)(1.0))
        assert_finite(grad, "d(spline)/d(scale)")
        assert grad != 0.0
        fd = float(central_difference(evaluate, 1.0, 1e-6))
        assert grad == pytest.approx(fd, rel=1e-5)
        # linear in the scale, so the gradient is the value itself
        assert grad == pytest.approx(float(evaluate(1.0)), rel=1e-12)

    @pytest.mark.parametrize("x", [0.4, 1.7, 2.5, 3.3, 4.8])
    def test_gradient_matches_finite_differences(self, spline_data, x):
        t, u = spline_data
        spline = cubic_spline(t, u, extrapolate=True)
        grad = float(jax.grad(lambda z: spline(z))(x))
        fd = float(central_difference(lambda z: float(spline(z)), x, 1e-6))
        assert_finite(grad, f"d(spline)/dx at {x}")
        assert grad != 0.0
        assert grad == pytest.approx(fd, rel=1e-5)

    def test_gradient_of_numpy_eval_is_not_available(self, spline_data):
        """``numpy_eval`` is a NumPy fast path; it is opaque to autodiff."""
        t, u = spline_data
        spline = cubic_spline(t, u, extrapolate=True)
        with pytest.raises((TypeError, jax.errors.TracerArrayConversionError)):
            jax.grad(lambda x: spline.numpy_eval(x))(2.5)

    def test_jit_works_when_extrapolating(self, spline_data):
        t, u = spline_data
        spline = cubic_spline(t, u, extrapolate=True)
        jitted = jax.jit(lambda x: spline(x))
        for x in (0.0, 2.5, 5.0, -1.0, 7.0):
            assert float(jitted(x)) == pytest.approx(float(spline(x)),
                                                     rel=1e-12)

    def test_jit_is_impossible_without_extrapolation(self, spline_data):
        """``extrapolate=False`` does a Python ``if`` on a traced comparison.

        ``jnp.any(...)`` under ``jit`` is a tracer, and ``if <tracer>`` cannot
        be resolved at trace time. The bounds check is the point of
        ``extrapolate=False``, so this is a real constraint on the API and not
        something to work around silently.
        """
        t, u = spline_data
        spline = cubic_spline(t, u, extrapolate=False)
        with pytest.raises(jax.errors.TracerBoolConversionError):
            jax.jit(lambda x: spline(x))(2.5)

    def test_eager_grad_still_works_without_extrapolation(self, spline_data):
        """Eager ``grad`` is fine: the tracer there wraps a concrete value."""
        t, u = spline_data
        spline = cubic_spline(t, u, extrapolate=False)
        grad = float(jax.grad(lambda x: spline(x))(2.5))
        assert_finite(grad, "d(bounded spline)/dx")
        assert grad == pytest.approx(2 * 2.5, rel=1e-3)

    def test_vmap_over_evaluation_points(self, spline_data):
        t, u = spline_data
        spline = cubic_spline(t, u, extrapolate=True)
        xs = jnp.array([0.4, 1.7, 2.5, 3.3, 4.8])
        batched = jax.vmap(lambda x: spline(x))(xs)
        np.testing.assert_allclose(np.asarray(batched),
                                   np.asarray(spline(xs)), rtol=1e-13)

    def test_cumulative_integral_is_traceable_in_its_endpoints(self,
                                                               spline_data):
        """The integral bounds are concrete Python floats by construction.

        ``cumulative_integral`` calls ``int(idx1)``/``range(...)``, so the
        interval indices must be concrete. That makes it usable eagerly but not
        under ``jit``; pin both halves.
        """
        t, u = spline_data
        spline = cubic_spline(t, u)
        out = np.asarray(spline.cumulative_integral(0.3, 4.4))
        assert_finite(out, "cumulative_integral")
        with pytest.raises(Exception):
            jax.jit(lambda a: spline.cumulative_integral(a, 4.4))(0.3)


# ===========================================================================
# Wavelengths / abundances: the NumPy boundary
# ===========================================================================


class TestNumpyOnlyUtilities:
    """These are grid- and table-building helpers, not differentiable physics."""

    def test_wavelength_grid_construction_is_not_traceable(self):
        """``Wavelengths`` builds a NumPy grid, so it cannot be jitted."""
        from korg.wavelengths import Wavelengths

        with pytest.raises(Exception):
            jax.jit(lambda start: len(Wavelengths((start, 5010.0, 1.0))))(5000.0)

    def test_wavelength_grid_is_finite_and_strictly_increasing(self):
        from korg.wavelengths import Wavelengths

        wls = Wavelengths([(5000, 5010, 0.5), (6000, 6010, 0.5)])
        assert_finite(wls.all_wls, "all_wls")
        assert_finite(wls.all_freqs, "all_freqs")
        assert np.all(np.diff(wls.all_wls) > 0)
        assert np.all(wls.all_wls > 0), "a zero wavelength divides by zero later"

    def test_format_A_X_is_not_traceable_but_is_finite(self):
        """Abundances are a fitted *input*, but the table build is NumPy."""
        from korg.abundances import format_A_X, get_metals_H

        with pytest.raises(Exception):
            jax.jit(lambda mh: format_A_X(mh)[25])(-1.0)

        A_X = format_A_X(-1.0)
        assert_finite(A_X, "format_A_X")
        assert_finite(get_metals_H(A_X), "get_metals_H")

    def test_metallicity_recovery_is_smooth_in_the_metallicity(self):
        """A numerical derivative of [M/H] w.r.t. the requested value is 1."""
        from korg.abundances import format_A_X, get_metals_H

        def f(mh):
            return get_metals_H(format_A_X(default_metals_H=mh))

        derivative = central_difference(f, -1.0, 1e-5)
        assert float(derivative) == pytest.approx(1.0, rel=1e-6)


# ===========================================================================
# Radiative transfer core
# ===========================================================================


@pytest.fixture(scope="module")
def atmosphere():
    """A smooth, monotonic 40-layer plane-parallel test atmosphere.

    Chosen so that the reference optical depth spans 1e-4 to 1e2 (the usual
    MARCS range) and the source function increases inward, which is what makes
    the emergent flux positive and the schemes comparable.
    """
    n_layers = 40
    log_tau_ref = np.linspace(-4.0, 2.0, n_layers)
    T = np.linspace(4200.0, 8500.0, n_layers)
    S = 1e-5 * (T / 5000.0) ** 4
    alpha_ref = np.geomspace(1e-9, 1e-7, n_layers)
    alpha = alpha_ref * 1.3
    spatial_coord = np.linspace(3e7, 0.0, n_layers)
    return dict(alpha=jnp.asarray(alpha), S=jnp.asarray(S),
                spatial_coord=jnp.asarray(spatial_coord),
                log_tau_ref=jnp.asarray(log_tau_ref),
                alpha_ref=jnp.asarray(alpha_ref))


class TestGenerateMuGrid:
    @pytest.mark.parametrize("n_mu", [2, 5, 7, 20])
    def test_gauss_legendre_grid(self, n_mu):
        mu, weights = rt_core.generate_mu_grid(n_mu)
        assert mu.shape == (n_mu,)
        assert np.all((np.asarray(mu) >= 0.0) & (np.asarray(mu) <= 1.0))
        assert float(jnp.sum(weights)) == pytest.approx(1.0, rel=1e-12)

    def test_quadrature_is_exact_for_low_order_polynomials(self):
        """5-point Gauss-Legendre integrates degree <= 9 exactly on [0, 1]."""
        mu, weights = rt_core.generate_mu_grid(5)
        for power in range(10):
            got = float(jnp.sum(weights * mu**power))
            assert got == pytest.approx(1.0 / (power + 1), rel=1e-12)

    def test_explicit_mu_values_use_trapezoid_weights(self):
        """Korg.jl's ``generate_mu_grid(mu_values)`` method."""
        mus = np.array([0.1, 0.4, 0.7, 1.0])
        mu, weights = rt_core.generate_mu_grid(mus)
        np.testing.assert_allclose(np.asarray(mu), mus, rtol=0)
        assert float(jnp.sum(weights)) == pytest.approx(mus[-1] - mus[0],
                                                        rel=1e-12)
        # a linear function integrates exactly under the trapezoid rule
        assert float(jnp.sum(weights * mu)) == pytest.approx(
            0.5 * (1.0**2 - 0.1**2), rel=1e-12)

    def test_single_explicit_mu_value(self):
        mu, weights = rt_core.generate_mu_grid(np.array([0.7]))
        assert float(mu[0]) == 0.7
        assert float(weights[0]) == 1.0

    def test_leggauss_nodes_match_numpy(self):
        """``leggauss`` is an eigenvalue solve; check it against NumPy's."""
        for n in (3, 5, 12):
            nodes, weights = rt_core.leggauss(n)
            np_nodes, np_weights = np.polynomial.legendre.leggauss(n)
            np.testing.assert_allclose(np.asarray(nodes), np_nodes, atol=1e-12)
            np.testing.assert_allclose(np.asarray(weights), np_weights,
                                       atol=1e-12)


class TestRadiativeTransferSingleWavelength:
    def test_anchored_expint_flux(self, atmosphere):
        flux, intensity = rt_core.radiative_transfer_single_wavelength(
            **atmosphere)
        assert intensity is None
        assert_finite(flux, "flux")
        assert float(flux) > 0

    def test_anchored_without_expint(self, atmosphere):
        flux, intensity = rt_core.radiative_transfer_single_wavelength(
            **atmosphere, use_expint_flux=False)
        assert intensity is None
        assert float(flux) > 0

    def test_expint_and_vertical_ray_fluxes_are_comparable(self, atmosphere):
        """Both approximate the same integral, so they must not differ wildly."""
        expint, _ = rt_core.radiative_transfer_single_wavelength(**atmosphere)
        eddington, _ = rt_core.radiative_transfer_single_wavelength(
            **atmosphere, use_expint_flux=False)
        assert float(expint) == pytest.approx(float(eddington), rel=0.5)

    @pytest.mark.parametrize("n_mu", [5, 9])
    def test_linear_intensity_scheme_cannot_run(self, atmosphere, n_mu):
        """``intensity_scheme="linear"`` has never executed. Pin that.

        ``radiative_transfer_single_wavelength`` reaches
        ``compute_I_linear(tau, S, mu)``, which is decorated with ``@jit`` in
        ``radiative_transfer/intensity.py`` and contains
        ``if delta_tau <= 0: continue`` inside its layer loop. Under the
        decorator's trace, ``delta_tau`` is a tracer, so the ``if`` raises
        ``TracerBoolConversionError`` on the first layer, every time, for every
        input. The branch is dead on arrival.

        ``intensity.py`` is out of scope for this change, so this test records
        the failure rather than fixing it. When ``compute_I_linear`` is
        rewritten with ``jnp.where``/``lax.cond``, replace this with the
        obvious positive test: flux > 0 and ``intensity.shape == (n_mu,)``.
        """
        with pytest.raises(jax.errors.TracerBoolConversionError):
            rt_core.radiative_transfer_single_wavelength(
                **atmosphere, intensity_scheme="linear", n_mu=n_mu)

    def test_bezier_intensity_scheme_is_the_working_angle_resolved_path(
            self, atmosphere):
        """Contrast with the above: the Bezier intensity solver does run."""
        flux, intensity = rt_core.radiative_transfer_single_wavelength(
            **atmosphere, intensity_scheme="bezier", n_mu=5)
        assert intensity is not None
        assert intensity.shape == (5,)
        assert_finite(intensity, "intensity")
        assert float(flux) > 0

    def test_bezier_intensity_scheme(self, atmosphere):
        flux, intensity = rt_core.radiative_transfer_single_wavelength(
            **atmosphere, intensity_scheme="bezier", n_mu=5)
        assert intensity.shape == (5,)
        assert_finite(flux, "bezier flux")
        assert float(flux) > 0

    def test_bezier_tau_scheme(self, atmosphere):
        flux, _ = rt_core.radiative_transfer_single_wavelength(
            **atmosphere, tau_scheme="bezier")
        assert_finite(flux, "bezier-tau flux")

    def test_anchored_requires_alpha_ref(self, atmosphere):
        kwargs = dict(atmosphere)
        kwargs["alpha_ref"] = None
        with pytest.raises(ValueError, match="requires alpha_ref"):
            rt_core.radiative_transfer_single_wavelength(**kwargs)

    def test_unknown_tau_scheme_rejected(self, atmosphere):
        with pytest.raises(ValueError, match="Unknown tau_scheme"):
            rt_core.radiative_transfer_single_wavelength(**atmosphere,
                                                         tau_scheme="magic")

    def test_unknown_intensity_scheme_rejected(self, atmosphere):
        with pytest.raises(ValueError, match="Unknown intensity_scheme"):
            rt_core.radiative_transfer_single_wavelength(
                **atmosphere, intensity_scheme="magic")

    def test_spherical_branch(self, atmosphere):
        """``spherical=True`` delegates to the shell solver and unwraps it."""
        radii = jnp.linspace(7.0e10, 6.9e10, len(atmosphere["alpha"]))
        flux, intensity = rt_core.radiative_transfer_single_wavelength(
            atmosphere["alpha"], atmosphere["S"], radii,
            atmosphere["log_tau_ref"], atmosphere["alpha_ref"],
            spherical=True, n_mu=8)
        assert np.ndim(flux) == 0
        assert intensity.shape == (8,)
        assert float(flux) > 0

    def test_flux_scales_with_the_source_function(self, atmosphere):
        """The transfer equation is linear in S."""
        base, _ = rt_core.radiative_transfer_single_wavelength(**atmosphere)
        kwargs = dict(atmosphere)
        kwargs["S"] = atmosphere["S"] * 3.0
        scaled, _ = rt_core.radiative_transfer_single_wavelength(**kwargs)
        assert float(scaled) == pytest.approx(3.0 * float(base), rel=1e-10)


class TestRadiativeTransferMultiWavelength:
    @pytest.fixture
    def grids(self, atmosphere):
        n_wl = 4
        scales = jnp.array([0.5, 1.0, 2.0, 4.0])[:, None]
        return dict(
            alpha_grid=atmosphere["alpha"][None, :] * scales,
            S_grid=jnp.tile(atmosphere["S"], (n_wl, 1)),
            spatial_coord=atmosphere["spatial_coord"],
            log_tau_ref=atmosphere["log_tau_ref"],
            alpha_ref=atmosphere["alpha_ref"],
        )

    def test_flux_only_returns_no_intensities(self, grids):
        fluxes, intensities = rt_core.radiative_transfer(**grids)
        assert fluxes.shape == (4,)
        assert intensities is None
        assert np.all(np.asarray(fluxes) > 0)

    def test_matches_the_single_wavelength_solver(self, grids):
        fluxes, _ = rt_core.radiative_transfer(**grids)
        for i in range(grids["alpha_grid"].shape[0]):
            one, _ = rt_core.radiative_transfer_single_wavelength(
                grids["alpha_grid"][i], grids["S_grid"][i],
                grids["spatial_coord"], grids["log_tau_ref"],
                grids["alpha_ref"])
            assert float(fluxes[i]) == pytest.approx(float(one), rel=1e-12)

    def test_bezier_scheme_collects_intensities(self, grids):
        """The multi-wavelength loop accumulates one intensity row per point."""
        fluxes, intensities = rt_core.radiative_transfer(
            **grids, intensity_scheme="bezier", n_mu=6)
        assert fluxes.shape == (4,)
        assert intensities.shape == (4, 6)
        assert_finite(intensities, "intensities")

    def test_linear_scheme_cannot_run(self, grids):
        """Same dead branch as the single-wavelength solver; see that test."""
        with pytest.raises(jax.errors.TracerBoolConversionError):
            rt_core.radiative_transfer(**grids, intensity_scheme="linear",
                                       n_mu=6)

    def test_more_opacity_gives_less_flux(self, grids):
        fluxes, _ = rt_core.radiative_transfer(**grids)
        assert np.all(np.diff(np.asarray(fluxes)) < 0), (
            "increasing alpha at fixed alpha_ref must lower the emergent flux"
        )

    def test_spherical_branch(self, atmosphere, grids):
        radii = jnp.linspace(7.0e10, 6.9e10, len(atmosphere["alpha"]))
        fluxes, intensities = rt_core.radiative_transfer(
            grids["alpha_grid"], grids["S_grid"], radii,
            grids["log_tau_ref"], grids["alpha_ref"], spherical=True, n_mu=8)
        assert fluxes.shape == (4,)
        assert intensities.shape == (4, 8)

    def test_photosphere_correction_rescales_the_flux(self, atmosphere, grids):
        radii = jnp.linspace(7.0e10, 6.9e10, len(atmosphere["alpha"]))
        base, _ = rt_core.radiative_transfer_spherical(
            grids["alpha_grid"], grids["S_grid"], radii, grids["log_tau_ref"],
            grids["alpha_ref"], n_mu=8)
        corrected, _ = rt_core.radiative_transfer_spherical(
            grids["alpha_grid"], grids["S_grid"], radii, grids["log_tau_ref"],
            grids["alpha_ref"], n_mu=8, R_photosphere=6.95e10)
        expected = np.asarray(base) * (float(radii[0]) / 6.95e10) ** 2
        np.testing.assert_allclose(np.asarray(corrected), expected, rtol=1e-12)

    def test_spherical_accepts_an_explicit_mu_grid(self, atmosphere, grids):
        """``mu_grid`` without weights: the weights are derived from it."""
        radii = jnp.linspace(7.0e10, 6.9e10, len(atmosphere["alpha"]))
        mus = np.linspace(0.05, 1.0, 9)
        with_values, _ = rt_core.radiative_transfer_spherical(
            grids["alpha_grid"], grids["S_grid"], radii, grids["log_tau_ref"],
            grids["alpha_ref"], mu_grid=mus)
        assert with_values.shape == (4,)
        assert np.all(np.asarray(with_values) > 0)

    def test_spherical_accepts_explicit_mu_grid_and_weights(self, atmosphere,
                                                            grids):
        """Both given: neither ``generate_mu_grid`` branch is taken."""
        radii = jnp.linspace(7.0e10, 6.9e10, len(atmosphere["alpha"]))
        mus, weights = rt_core.generate_mu_grid(9)
        both, _ = rt_core.radiative_transfer_spherical(
            grids["alpha_grid"], grids["S_grid"], radii, grids["log_tau_ref"],
            grids["alpha_ref"], mu_grid=mus, mu_weights=weights)
        from_n_mu, _ = rt_core.radiative_transfer_spherical(
            grids["alpha_grid"], grids["S_grid"], radii, grids["log_tau_ref"],
            grids["alpha_ref"], n_mu=9)
        np.testing.assert_allclose(np.asarray(both), np.asarray(from_n_mu),
                                   rtol=1e-13)


class TestRadiativeTransferJitPath:
    def test_anchored_tau_matches_a_hand_trapezoid(self, atmosphere):
        tau = rt_core._compute_tau_anchored_planar(
            atmosphere["alpha"], atmosphere["log_tau_ref"],
            atmosphere["alpha_ref"])
        integrand = np.asarray(atmosphere["alpha"]) * \
            10.0 ** np.asarray(atmosphere["log_tau_ref"]) / \
            np.asarray(atmosphere["alpha_ref"])
        expected = np.concatenate([[0.0], np.cumsum(
            0.5 * (integrand[:-1] + integrand[1:]) *
            np.diff(np.asarray(atmosphere["log_tau_ref"])) * np.log(10.0))])
        np.testing.assert_allclose(np.asarray(tau), expected, rtol=1e-13)
        assert np.all(np.diff(np.asarray(tau)) > 0)

    def test_single_wavelength_jit_matches_the_eager_solver(self, atmosphere):
        jit_flux = rt_core.radiative_transfer_single_wavelength_jit(
            atmosphere["alpha"], atmosphere["S"], atmosphere["log_tau_ref"],
            atmosphere["alpha_ref"])
        eager, _ = rt_core.radiative_transfer_single_wavelength(**atmosphere)
        assert float(jit_flux) == pytest.approx(float(eager), rel=1e-8)

    def test_batched_jit_matches_the_scalar_jit(self, atmosphere):
        alpha_grid = jnp.stack([atmosphere["alpha"],
                                atmosphere["alpha"] * 2.0])
        S_grid = jnp.stack([atmosphere["S"], atmosphere["S"]])
        fluxes, intensities = rt_core.radiative_transfer_jit(
            alpha_grid, S_grid, atmosphere["spatial_coord"],
            atmosphere["log_tau_ref"], atmosphere["alpha_ref"])
        assert intensities is None
        assert fluxes.shape == (2,)
        for i in range(2):
            one = rt_core.radiative_transfer_single_wavelength_jit(
                alpha_grid[i], S_grid[i], atmosphere["log_tau_ref"],
                atmosphere["alpha_ref"])
            assert float(fluxes[i]) == pytest.approx(float(one), rel=1e-12)

    def test_gauss_legendre_flux_helper_agrees_with_expint(self, atmosphere):
        """``_compute_F_gl`` is an unused second implementation of the flux.

        It is not called anywhere in the package -- ``radiative_transfer_
        single_wavelength_jit`` uses ``compute_F_flux_only_expint`` instead --
        but it is a genuine independent 20-point Gauss-Legendre evaluation of
        the same integral, so testing it cross-checks the expint path.
        """
        tau = rt_core._compute_tau_anchored_planar(
            atmosphere["alpha"], atmosphere["log_tau_ref"],
            atmosphere["alpha_ref"])
        gl = float(rt_core._compute_F_gl(tau, atmosphere["S"]))
        from korg.radiative_transfer.intensity import compute_F_flux_only_expint
        expint = float(compute_F_flux_only_expint(tau, atmosphere["S"]))
        assert_finite(gl, "_compute_F_gl")
        assert gl > 0
        assert gl == pytest.approx(expint, rel=2e-3)

    def test_gl_helper_handles_repeated_tau_values(self, atmosphere):
        """A zero-width layer must not produce a NaN via 0/0.

        ``_compute_F_gl`` guards the slope with
        ``jnp.where(delta_tau > 0, delta_tau, 1.0)``. The substitute is
        strictly positive, which is what keeps the cotangent finite too --
        clamping to zero would give a NaN gradient.
        """
        tau = rt_core._compute_tau_anchored_planar(
            atmosphere["alpha"], atmosphere["log_tau_ref"],
            atmosphere["alpha_ref"])
        tau = tau.at[5].set(tau[4])  # a duplicated optical depth
        value = rt_core._compute_F_gl(tau, atmosphere["S"])
        assert_finite(value, "_compute_F_gl with a repeated tau")

        grad = jax.grad(lambda S: rt_core._compute_F_gl(tau, S))(
            atmosphere["S"])
        assert_finite(grad, "d(_compute_F_gl)/dS with a repeated tau")

    def test_jit_of_the_jit_entry_point(self, atmosphere):
        """Already decorated with @jit; wrapping again must still compile."""
        alpha_grid = atmosphere["alpha"][None, :]
        S_grid = atmosphere["S"][None, :]
        wrapped = jax.jit(rt_core.radiative_transfer_jit)
        fluxes, _ = wrapped(alpha_grid, S_grid, atmosphere["spatial_coord"],
                            atmosphere["log_tau_ref"], atmosphere["alpha_ref"])
        assert_finite(fluxes, "jit(radiative_transfer_jit)")


class TestRadiativeTransferAutodiff:
    def test_gradient_with_respect_to_opacity(self, atmosphere):
        def flux(alpha):
            return rt_core.radiative_transfer_single_wavelength_jit(
                alpha, atmosphere["S"], atmosphere["log_tau_ref"],
                atmosphere["alpha_ref"])

        grad = jax.grad(flux)(atmosphere["alpha"])
        assert_finite(grad, "d(flux)/d(alpha)")
        assert np.any(np.asarray(grad) != 0.0)
        assert np.all(np.asarray(grad) <= 0.0), (
            "more opacity at fixed alpha_ref can only reduce the flux"
        )

    def test_gradient_with_respect_to_opacity_matches_finite_differences(
            self, atmosphere):
        def flux(scale):
            return rt_core.radiative_transfer_single_wavelength_jit(
                atmosphere["alpha"] * scale, atmosphere["S"],
                atmosphere["log_tau_ref"], atmosphere["alpha_ref"])

        grad = float(jax.grad(flux)(1.0))
        fd = float(central_difference(flux, 1.0, 1e-5))
        assert_finite(grad, "d(flux)/d(scale)")
        assert grad != 0.0
        assert grad == pytest.approx(fd, rel=1e-5)

    def test_gradient_with_respect_to_the_source_function(self, atmosphere):
        def flux(S):
            return rt_core.radiative_transfer_single_wavelength_jit(
                atmosphere["alpha"], S, atmosphere["log_tau_ref"],
                atmosphere["alpha_ref"])

        grad = jax.grad(flux)(atmosphere["S"])
        assert_finite(grad, "d(flux)/dS")
        assert np.all(np.asarray(grad) >= 0.0), "S can only add flux"
        # linear in S, so grad . S reproduces the flux itself
        assert float(jnp.dot(grad, atmosphere["S"])) == pytest.approx(
            float(flux(atmosphere["S"])), rel=1e-10)

    def test_gradient_through_the_batched_entry_point(self, atmosphere):
        alpha_grid = jnp.stack([atmosphere["alpha"],
                                atmosphere["alpha"] * 1.5])
        S_grid = jnp.stack([atmosphere["S"], atmosphere["S"]])

        def total_flux(alphas):
            fluxes, _ = rt_core.radiative_transfer_jit(
                alphas, S_grid, atmosphere["spatial_coord"],
                atmosphere["log_tau_ref"], atmosphere["alpha_ref"])
            return jnp.sum(fluxes)

        grad = jax.grad(total_flux)(alpha_grid)
        assert grad.shape == alpha_grid.shape
        assert_finite(grad, "d(sum flux)/d(alpha_grid)")
        assert np.any(np.asarray(grad) != 0.0)

    def test_gradient_of_anchored_tau(self, atmosphere):
        def total_tau(alpha_ref):
            return jnp.sum(rt_core._compute_tau_anchored_planar(
                atmosphere["alpha"], atmosphere["log_tau_ref"], alpha_ref))

        grad = jax.grad(total_tau)(atmosphere["alpha_ref"])
        assert_finite(grad, "d(tau)/d(alpha_ref)")
        fd_index = 7
        eps = 1e-6 * float(atmosphere["alpha_ref"][fd_index])
        plus = atmosphere["alpha_ref"].at[fd_index].add(eps)
        minus = atmosphere["alpha_ref"].at[fd_index].add(-eps)
        fd = (float(total_tau(plus)) - float(total_tau(minus))) / (2 * eps)
        assert float(grad[fd_index]) == pytest.approx(fd, rel=1e-5)

    def test_anchored_tau_gradient_survives_a_zero_reference_opacity(self):
        """The ``jnp.clip(alpha_ref, 1e-30, inf)`` guard must not leak a NaN."""
        log_tau_ref = jnp.linspace(-4.0, 2.0, 12)
        alpha = jnp.full(12, 1e-8)
        alpha_ref = jnp.full(12, 1e-8).at[3].set(0.0)

        def total_tau(a):
            return jnp.sum(rt_core._compute_tau_anchored_planar(
                a, log_tau_ref, alpha_ref))

        value = total_tau(alpha)
        assert_finite(value, "tau with a zero alpha_ref")
        assert_finite(jax.grad(total_tau)(alpha),
                      "d(tau)/d(alpha) with a zero alpha_ref")


# ===========================================================================
# simple_ionization
# ===========================================================================


@pytest.fixture(scope="module")
def statmech_data():
    """Ionization energies and partition functions, or a hard failure.

    Raising rather than skipping is deliberate: these ship with the package,
    so their absence is a broken install, and a skipped test here would be
    indistinguishable from a passing one in the summary line.
    """
    from korg.data_loader import ionization_energies, load_atomic_partition_functions
    partition_funcs = load_atomic_partition_functions()
    if not partition_funcs:
        raise RuntimeError("no atomic partition functions were loaded")
    if not ionization_energies:
        raise RuntimeError("no ionization energies were loaded")
    return ionization_energies, partition_funcs


class TestSimpleIonization:
    """``simple_ionization`` partitions a known total density with Saha.

    Note: this module is not imported by the package itself and is not part of
    Korg.jl's API. See the report accompanying these tests -- the recommendation
    is that it is superseded by ``korg.statmech.chemical_equilibrium``.
    """

    N_LAYERS = 4

    @pytest.fixture
    def inputs(self):
        from korg.abundances import format_A_X
        from korg.atomic_data import atomic_symbols
        return dict(
            temperatures=np.linspace(4500.0, 7500.0, self.N_LAYERS),
            electron_densities=np.geomspace(1e11, 1e14, self.N_LAYERS),
            total_number_densities=np.geomspace(1e15, 1e17, self.N_LAYERS),
            abundances_A_X=format_A_X(),
            atomic_symbols=list(atomic_symbols),
        )

    def test_species_keys_use_the_right_element(self, inputs, statmech_data):
        """Regression guard for an off-by-one in the element loop.

        ``atomic_symbols`` is indexed by Z-1, so the atomic number handed to
        ``saha_ion_weights`` must be ``i + 1``. The original loop used ``i``
        and skipped index 0, so hydrogen was absent entirely and every other
        element was ionised with its neighbour's ionization energies.
        """
        from korg.simple_ionization import compute_ionization_states
        from korg.species import Species

        densities = compute_ionization_states(
            **inputs, ionization_energies=statmech_data[0],
            partition_functions=statmech_data[1])

        assert Species("H I") in densities, "hydrogen went missing"
        assert Species("Fe I") in densities
        assert Species("U III") in densities
        assert len(densities) == 3 * 92

    def test_totals_are_conserved(self, inputs, statmech_data):
        """n_I + n_II + n_III must be the element's total number density."""
        from korg.simple_ionization import compute_ionization_states
        from korg.species import Species

        densities = compute_ionization_states(
            **inputs, ionization_energies=statmech_data[0],
            partition_functions=statmech_data[1])

        for Z, symbol in ((1, "H"), (26, "Fe"), (20, "Ca")):
            total = sum(densities[Species(symbol, charge)] for charge in (0, 1, 2))
            expected = (inputs["total_number_densities"] *
                        10 ** (inputs["abundances_A_X"][Z - 1] - 12))
            np.testing.assert_allclose(total, expected, rtol=1e-12)

    def test_agrees_with_saha_ion_weights_directly(self, inputs, statmech_data):
        """Cross-check against ``statmech.saha_ion_weights`` layer by layer."""
        from korg.simple_ionization import compute_ionization_states
        from korg.species import Species
        from korg.statmech import saha_ion_weights

        ionization_energies, partition_funcs = statmech_data
        densities = compute_ionization_states(
            **inputs, ionization_energies=ionization_energies,
            partition_functions=partition_funcs)

        for Z, symbol in ((1, "H"), (26, "Fe")):
            n_total = (inputs["total_number_densities"] *
                       10 ** (inputs["abundances_A_X"][Z - 1] - 12))
            for layer in range(self.N_LAYERS):
                wII, wIII = saha_ion_weights(
                    inputs["temperatures"][layer],
                    inputs["electron_densities"][layer],
                    Z, ionization_energies, partition_funcs)
                expected_I = n_total[layer] / (1.0 + wII + wIII)
                assert densities[Species(symbol, 0)][layer] == pytest.approx(
                    expected_I, rel=1e-12)
                assert densities[Species(symbol, 1)][layer] == pytest.approx(
                    expected_I * wII, rel=1e-12)
                assert densities[Species(symbol, 2)][layer] == pytest.approx(
                    expected_I * wIII, rel=1e-12)

    def test_hydrogen_is_never_doubly_ionised(self, inputs, statmech_data):
        from korg.simple_ionization import compute_ionization_states
        from korg.species import Species

        densities = compute_ionization_states(
            **inputs, ionization_energies=statmech_data[0],
            partition_functions=statmech_data[1])
        assert np.all(densities[Species("H", 2)] == 0.0)

    def test_ionization_increases_with_temperature(self, inputs, statmech_data):
        """At fixed electron density, Saha must ionise more when it is hotter."""
        from korg.simple_ionization import compute_ionization_states
        from korg.species import Species

        isothermal_ne = dict(inputs)
        isothermal_ne["electron_densities"] = np.full(self.N_LAYERS, 1e13)
        densities = compute_ionization_states(
            **isothermal_ne, ionization_energies=statmech_data[0],
            partition_functions=statmech_data[1])
        ratio = densities[Species("Fe", 1)] / densities[Species("Fe", 0)]
        assert np.all(np.diff(ratio) > 0), (
            "Fe II / Fe I must rise with temperature at fixed n_e"
        )

    def test_ionization_decreases_with_electron_density(self, inputs,
                                                        statmech_data):
        """At fixed temperature, more free electrons push Saha back to neutral."""
        from korg.simple_ionization import compute_ionization_states
        from korg.species import Species

        isothermal_T = dict(inputs)
        isothermal_T["temperatures"] = np.full(self.N_LAYERS, 6000.0)
        densities = compute_ionization_states(
            **isothermal_T, ionization_energies=statmech_data[0],
            partition_functions=statmech_data[1])
        ratio = densities[Species("Fe", 1)] / densities[Species("Fe", 0)]
        assert np.all(np.diff(ratio) < 0)

    def test_missing_data_falls_back_to_all_neutral(self, inputs):
        """The KeyError/IndexError fallback assumes the element is neutral."""
        from korg.simple_ionization import compute_ionization_states
        from korg.species import Species

        densities = compute_ionization_states(
            **inputs, ionization_energies={}, partition_functions={})
        expected = (inputs["total_number_densities"] *
                    10 ** (inputs["abundances_A_X"][25] - 12))
        np.testing.assert_allclose(densities[Species("Fe", 0)], expected,
                                   rtol=1e-12)
        assert np.all(densities[Species("Fe", 1)] == 0.0)
        assert np.all(densities[Species("Fe", 2)] == 0.0)

    def test_short_abundance_vector_truncates_the_element_list(self, inputs,
                                                               statmech_data):
        from korg.simple_ionization import compute_ionization_states
        from korg.species import Species

        truncated = dict(inputs)
        truncated["abundances_A_X"] = inputs["abundances_A_X"][:3]
        densities = compute_ionization_states(
            **truncated, ionization_energies=statmech_data[0],
            partition_functions=statmech_data[1])
        assert Species("Li I") in densities
        assert Species("Be I") not in densities
        assert len(densities) == 9


class TestElectronDensityConsistency:
    def test_self_consistent_case(self):
        from korg.simple_ionization import check_electron_density_consistency
        from korg.species import Species

        ne = np.array([100.0, 200.0])
        densities = {
            Species("H I"): np.array([1000.0, 1000.0]),
            Species("H II"): np.array([100.0, 200.0]),
        }
        ok, implied, error = check_electron_density_consistency(densities, ne)
        assert ok is True or ok == np.True_
        np.testing.assert_allclose(implied, ne, rtol=1e-12)
        np.testing.assert_allclose(error, 0.0, atol=1e-12)

    def test_doubly_ionised_species_contribute_two_electrons(self):
        from korg.simple_ionization import check_electron_density_consistency
        from korg.species import Species

        ne = np.array([300.0])
        densities = {
            Species("Ca II"): np.array([100.0]),
            Species("Ca III"): np.array([100.0]),
        }
        _, implied, _ = check_electron_density_consistency(densities, ne)
        assert float(implied[0]) == pytest.approx(100.0 + 2 * 100.0, rel=1e-12)

    def test_inconsistent_case_is_reported(self):
        from korg.simple_ionization import check_electron_density_consistency
        from korg.species import Species

        ne = np.array([100.0, 100.0])
        densities = {Species("H II"): np.array([100.0, 300.0])}
        ok, implied, error = check_electron_density_consistency(densities, ne)
        assert not ok
        assert float(error[1]) == pytest.approx(2.0, rel=1e-12)

    def test_tolerance_is_honoured(self):
        from korg.simple_ionization import check_electron_density_consistency
        from korg.species import Species

        ne = np.array([100.0])
        densities = {Species("H II"): np.array([105.0])}
        assert check_electron_density_consistency(densities, ne,
                                                  tolerance=0.1)[0]
        assert not check_electron_density_consistency(densities, ne,
                                                      tolerance=0.01)[0]

    def test_zero_electron_density_does_not_produce_nan(self):
        """A 0/0 in the fractional error is replaced with 0, not NaN."""
        from korg.simple_ionization import check_electron_density_consistency
        from korg.species import Species

        ne = np.array([0.0, 100.0])
        densities = {Species("H II"): np.array([0.0, 100.0])}
        ok, implied, error = check_electron_density_consistency(densities, ne)
        assert np.all(np.isfinite(error))
        assert ok

    def test_neutral_species_contribute_nothing(self):
        from korg.simple_ionization import check_electron_density_consistency
        from korg.species import Species

        ne = np.array([0.0])
        densities = {Species("H I"): np.array([1e10])}
        _, implied, _ = check_electron_density_consistency(densities, ne)
        assert float(implied[0]) == 0.0

    def test_anions_remove_electrons(self):
        """H- has charge -1 and so subtracts from the implied electron count."""
        from korg.simple_ionization import check_electron_density_consistency
        from korg.species import Species

        ne = np.array([100.0])
        densities = {
            Species("H II"): np.array([100.0]),
            Species("H-"): np.array([10.0]),
        }
        _, implied, _ = check_electron_density_consistency(densities, ne)
        assert float(implied[0]) == pytest.approx(90.0, rel=1e-12)
