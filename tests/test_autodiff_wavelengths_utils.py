"""
Automatic-differentiation tests for the wavelength and utility layer.

The purpose of this port is that spectral synthesis be differentiable with JAX,
so every function that maps floats to floats needs its *gradient* guarded, not
only its value. Value-level tests are blind to the failure mode that has bitten
this codebase repeatedly::

    y = jnp.where(cond, safe_expression, dangerous_expression)

If ``dangerous_expression`` evaluates to NaN or inf, the ``where`` masks the
value but not the cotangent: reverse-mode AD still pushes a gradient through the
dead branch and the result is a NaN derivative sitting behind a perfectly
healthy-looking number. The remedy is always to make the dangerous branch finite
*before* the select (the "double where" pattern, e.g.
``jnp.sqrt(jnp.maximum(x, eps))``).

These tests cover:

- ``korg.utils.air_to_vacuum`` / ``vacuum_to_air``
- ``korg.utils.normal_pdf``
- ``korg.utils.translational_U``
- ``korg.cubic_splines.CubicSpline.__call__`` and ``CubicSpline._integral``
- ``korg.utils.apply_LSF``, ``apply_rotation`` and ``compute_LSF_matrix``,
  with respect to the flux vector *and* to the two shape parameters
  ``korg.fit`` optimises, ``R`` and ``vsini``

and document, with executable checks, the parts of the layer that are
deliberately not differentiable (wavelength grids, window indices, interval
membership and spline construction).

Every gradient assertion here checks three things: that the derivative is
finite, that it is non-zero wherever physics says it must be, and that it agrees
with a central finite difference.
"""

# Import korg FIRST: it enables JAX x64 mode as a side effect, and every
# tolerance below assumes float64.
import korg

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from korg.constants import c_cgs, hplanck_cgs, kboltz_cgs
from korg.cubic_splines import CubicSpline, cubic_spline
from korg.utils import (
    air_to_vacuum,
    apply_LSF,
    apply_rotation,
    compute_LSF_matrix,
    normal_pdf,
    translational_U,
    vacuum_to_air,
)


# Exceptions raised when a NumPy-only implementation is handed a JAX tracer.
# ``np.asarray``/``float()`` on a tracer give TracerArrayConversionError; a
# Python ``if`` on a traced predicate gives TracerBoolConversionError (both
# subclass ConcretizationTypeError); assigning a tracer into a preallocated
# ``np.ndarray`` gives a bare ValueError or TypeError from NumPy itself.
_NOT_TRACEABLE = (
    jax.errors.TracerArrayConversionError,
    jax.errors.ConcretizationTypeError,
    TypeError,
    ValueError,
)


def central_difference(f, x, h):
    """
    Central finite difference of a scalar function.

    Parameters
    ----------
    f : callable
        Function of a single float, returning a float.
    x : float
        Point at which to differentiate.
    h : float
        Step size. Should be chosen ~1e-5 to 1e-6 relative to the scale of `x`
        so that truncation and round-off error are both near their float64
        minimum of a few times 1e-6 relative.

    Returns
    -------
    float
        Approximation to ``df/dx``.
    """
    return (float(f(x + h)) - float(f(x - h))) / (2.0 * h)


def assert_finite(value, name):
    """
    Assert that every element of `value` is finite.

    Parameters
    ----------
    value : float or array_like
        Quantity to check, typically a gradient.
    name : str
        Human-readable name used in the failure message.
    """
    arr = np.asarray(value, dtype=np.float64)
    assert np.all(np.isfinite(arr)), (
        f"{name} is not finite (NaN or inf): {arr}. This is the signature of a "
        "`jnp.where` whose dead branch evaluates to NaN/inf -- the value is "
        "masked but the cotangent is not. Make the dangerous branch finite "
        "before the select (double-where)."
    )


# =============================================================================
# Air <-> vacuum conversion
# =============================================================================

# Birch & Downs (1994) is quoted as valid above 2000 A; the refractive-index
# poles sit near 880 A and 1600 A, far below any wavelength used here.
AIR_VACUUM_WAVELENGTHS = [2000.0, 3000.0, 3933.66, 5500.0, 6562.8, 10000.0, 50000.0]


class TestAirVacuumAutodiff:
    """Gradients of the air/vacuum refractive-index conversions."""

    @pytest.mark.parametrize("func", [air_to_vacuum, vacuum_to_air])
    @pytest.mark.parametrize("lambda_angstrom", AIR_VACUUM_WAVELENGTHS)
    def test_gradient_is_finite_and_near_unity(self, func, lambda_angstrom):
        """d(lambda_out)/d(lambda_in) must be finite and within 1e-3 of one."""
        g = jax.grad(func)(lambda_angstrom)
        assert_finite(g, f"d{func.__name__}/dlambda at {lambda_angstrom} A")
        # n - 1 ~ 2.8e-4 in the optical, so the derivative is 1 to within 1e-3.
        assert abs(float(g) - 1.0) < 1e-3, (
            f"{func.__name__} derivative {float(g)} is implausible; the "
            "refractive index only differs from 1 at the 1e-4 level."
        )
        assert float(g) > 0.0, "the conversion must be monotonically increasing"

    @pytest.mark.parametrize("func", [air_to_vacuum, vacuum_to_air])
    @pytest.mark.parametrize("lambda_angstrom", AIR_VACUUM_WAVELENGTHS)
    def test_gradient_matches_finite_difference(self, func, lambda_angstrom):
        """AD and central differences must agree to 1e-6 relative."""
        g = float(jax.grad(func)(lambda_angstrom))
        fd = central_difference(func, lambda_angstrom, h=1e-6 * lambda_angstrom)
        assert g == pytest.approx(fd, rel=1e-6)

    def test_roundtrip_derivative_is_one(self):
        """vacuum_to_air(air_to_vacuum(x)) is the identity, so its slope is 1."""
        g = float(jax.grad(lambda x: vacuum_to_air(air_to_vacuum(x)))(5500.0))
        assert_finite(g, "roundtrip derivative")
        # The two formulae are different fits, so the round trip is the identity
        # only to ~1e-8; the derivative inherits that level of agreement.
        assert g == pytest.approx(1.0, abs=1e-7)

    def test_jacfwd_over_wavelength_array_is_diagonal(self):
        """Element-wise conversion must have a diagonal, finite Jacobian."""
        lams = jnp.array([4000.0, 5000.0, 6000.0, 8000.0])
        jac = np.asarray(jax.jacfwd(air_to_vacuum)(lams))
        assert_finite(jac, "air_to_vacuum Jacobian")
        off_diagonal = jac - np.diag(np.diag(jac))
        assert np.all(off_diagonal == 0.0), "conversion must act element-wise"
        assert np.all(np.diag(jac) > 0.0)

    def test_second_derivative_is_finite(self):
        """Curvature must also be free of NaN (guards nested differentiation)."""
        g2 = jax.grad(jax.grad(air_to_vacuum))(5500.0)
        assert_finite(g2, "d2(air_to_vacuum)/dlambda2")
        assert float(g2) != 0.0, "the dispersion relation is not exactly linear"


# =============================================================================
# normal_pdf
# =============================================================================

# sigma values spanning the range actually used in Korg: ~1e-11 cm (a 1e-3 A
# LSF sigma) up to order-unity values in the dimensionless tests.
NORMAL_PDF_SIGMAS = [1e-11, 1e-8, 1e-4, 1.0, 10.0, 1e4]


class TestNormalPdfAutodiff:
    """Gradients of the Gaussian probability density."""

    @pytest.mark.parametrize("sigma", NORMAL_PDF_SIGMAS)
    @pytest.mark.parametrize("n_sigma", [0.0, 0.5, 1.0, 4.0, 40.0])
    def test_gradients_are_finite_over_the_whole_kernel(self, sigma, n_sigma):
        """No NaN in d/d(delta) or d/d(sigma), including in the far wings.

        The far wings are the dangerous region: ``exp(-0.5 (delta/sigma)^2)``
        underflows to exactly zero there while its derivative factor
        ``delta**2 / sigma**3`` grows without bound, and ``0 * inf`` is NaN.
        """
        delta = n_sigma * sigma
        g_delta, g_sigma = jax.grad(normal_pdf, argnums=(0, 1))(delta, sigma)
        assert_finite(g_delta, f"d(normal_pdf)/d(delta) at {n_sigma} sigma")
        assert_finite(g_sigma, f"d(normal_pdf)/d(sigma) at {n_sigma} sigma")

    @pytest.mark.parametrize("sigma", [1e-8, 1.0, 10.0])
    @pytest.mark.parametrize("n_sigma", [0.5, 2.5, 4.0])
    def test_gradient_matches_finite_difference(self, sigma, n_sigma):
        """AD must reproduce central differences in both arguments.

        |delta| = sigma is deliberately excluded: d(pdf)/d(sigma) is exactly
        zero there, so a relative comparison would only be comparing round-off.
        That point gets its own analytic test below.
        """
        delta = n_sigma * sigma
        g_delta, g_sigma = jax.grad(normal_pdf, argnums=(0, 1))(delta, sigma)

        fd_delta = central_difference(
            lambda d: normal_pdf(d, sigma), delta, h=1e-6 * sigma
        )
        fd_sigma = central_difference(
            lambda s: normal_pdf(delta, s), sigma, h=1e-6 * sigma
        )
        assert float(g_delta) == pytest.approx(fd_delta, rel=1e-5)
        assert float(g_sigma) == pytest.approx(fd_sigma, rel=1e-5)

    @pytest.mark.parametrize("sigma", NORMAL_PDF_SIGMAS)
    def test_peak_is_a_stationary_maximum(self, sigma):
        """At delta = 0 the first derivative vanishes and the second is negative."""
        g_delta = jax.grad(normal_pdf, argnums=0)(0.0, sigma)
        assert_finite(g_delta, "d(normal_pdf)/d(delta) at the peak")
        assert float(g_delta) == 0.0, "the Gaussian peak must be stationary"

        g2 = jax.grad(jax.grad(normal_pdf, argnums=0), argnums=0)(0.0, sigma)
        assert_finite(g2, "d2(normal_pdf)/d(delta)2 at the peak")
        assert float(g2) < 0.0, "delta = 0 must be a maximum, not a minimum"

    @pytest.mark.parametrize("sigma", NORMAL_PDF_SIGMAS)
    def test_width_gradient_is_nonzero_and_correctly_signed(self, sigma):
        """Widening the Gaussian must lower the peak and raise the far wing."""
        g_peak = float(jax.grad(normal_pdf, argnums=1)(0.0, sigma))
        assert_finite(g_peak, "d(normal_pdf)/d(sigma) at the peak")
        assert g_peak < 0.0, "the peak of a normalised Gaussian falls as sigma grows"

        g_wing = float(jax.grad(normal_pdf, argnums=1)(2.0 * sigma, sigma))
        assert_finite(g_wing, "d(normal_pdf)/d(sigma) at 2 sigma")
        assert g_wing > 0.0, "beyond 1 sigma the density rises as sigma grows"

    @pytest.mark.parametrize("sigma", NORMAL_PDF_SIGMAS)
    @pytest.mark.parametrize("sign", [-1.0, 1.0])
    def test_width_gradient_vanishes_at_one_sigma(self, sigma, sign):
        """d(pdf)/d(sigma) = pdf * (delta^2/sigma^3 - 1/sigma) is zero at |delta| = sigma.

        This is the crossover between the falling peak and the rising wing. It
        is an exact analytic zero, so it pins the shape of the sigma derivative
        far more tightly than a finite difference could.
        """
        g = float(jax.grad(normal_pdf, argnums=1)(sign * sigma, sigma))
        assert_finite(g, "d(normal_pdf)/d(sigma) at |delta| = sigma")
        # Scale of a "typical" non-zero value of this derivative, for context.
        scale = float(normal_pdf(0.0, sigma)) / sigma
        assert abs(g) < 1e-12 * scale

    def test_normalisation_is_insensitive_to_sigma(self):
        """d/d(sigma) of the integrated PDF is zero, because it always equals 1."""
        grid = jnp.linspace(-12.0, 12.0, 4001)
        dx = float(grid[1] - grid[0])

        def integral(sigma):
            return jnp.sum(normal_pdf(grid, sigma)) * dx

        assert float(integral(1.0)) == pytest.approx(1.0, rel=1e-10)
        g = jax.grad(integral)(1.0)
        assert_finite(g, "d(integral of normal_pdf)/d(sigma)")
        assert abs(float(g)) < 1e-8, (
            "the Gaussian is normalised for every sigma, so the integral must "
            "have zero derivative"
        )

    def test_jacfwd_over_delta_array_is_diagonal(self):
        """normal_pdf must act element-wise on the deviation array."""
        deltas = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        jac = np.asarray(jax.jacfwd(normal_pdf, argnums=0)(deltas, 1.0))
        assert_finite(jac, "normal_pdf Jacobian w.r.t. delta")
        assert np.all(jac - np.diag(np.diag(jac)) == 0.0)


# =============================================================================
# translational_U
# =============================================================================

class TestTranslationalUAutodiff:
    """Gradients of the translational partition function (2 pi m k T / h^2)^1.5."""

    @pytest.mark.parametrize("mass_g", [9.1093837e-28, 1.6737e-24, 1.0e-22])
    @pytest.mark.parametrize("temperature", [1000.0, 3500.0, 5777.0, 20000.0])
    def test_gradients_are_finite_and_positive(self, mass_g, temperature):
        """U grows with both mass and temperature, with finite derivatives."""
        g_m, g_T = jax.grad(translational_U, argnums=(0, 1))(mass_g, temperature)
        assert_finite(g_m, "d(translational_U)/dm")
        assert_finite(g_T, "d(translational_U)/dT")
        assert float(g_m) > 0.0
        assert float(g_T) > 0.0

    @pytest.mark.parametrize("temperature", [1000.0, 5777.0, 20000.0])
    def test_log_derivatives_are_exactly_three_halves(self, temperature):
        """U is a pure 1.5 power law, so dlog(U)/dlog(x) = 1.5 in both arguments.

        This is a much sharper check than a finite difference: it pins the
        analytic form of the derivative, not just its numerical value.
        """
        mass_g = 1.6737e-24

        dlogU_dlogT = jax.grad(
            lambda logT: jnp.log(translational_U(mass_g, jnp.exp(logT)))
        )(float(np.log(temperature)))
        dlogU_dlogm = jax.grad(
            lambda logm: jnp.log(translational_U(jnp.exp(logm), temperature))
        )(float(np.log(mass_g)))

        assert_finite(dlogU_dlogT, "dlogU/dlogT")
        assert_finite(dlogU_dlogm, "dlogU/dlogm")
        assert float(dlogU_dlogT) == pytest.approx(1.5, rel=1e-12)
        assert float(dlogU_dlogm) == pytest.approx(1.5, rel=1e-12)

    @pytest.mark.parametrize("temperature", [1000.0, 5777.0, 20000.0])
    def test_temperature_gradient_matches_finite_difference(self, temperature):
        """AD and central differences must agree to 1e-6 relative."""
        mass_g = 1.6737e-24
        g = float(jax.grad(translational_U, argnums=1)(mass_g, temperature))
        fd = central_difference(
            lambda T: translational_U(mass_g, T), temperature, h=1e-6 * temperature
        )
        assert g == pytest.approx(fd, rel=1e-6)

    def test_gradient_matches_closed_form(self):
        """dU/dT = 1.5 * (2 pi m k / h^2)^1.5 * T^0.5."""
        mass_g, temperature = 1.6737e-24, 5777.0
        prefactor = (2 * np.pi * mass_g * kboltz_cgs / hplanck_cgs**2) ** 1.5
        expected = 1.5 * prefactor * np.sqrt(temperature)
        g = float(jax.grad(translational_U, argnums=1)(mass_g, temperature))
        assert g == pytest.approx(expected, rel=1e-12)

    def test_zero_temperature_is_not_nan(self):
        """T = 0 is unphysical but must not poison a gradient with NaN."""
        g_m, g_T = jax.grad(translational_U, argnums=(0, 1))(1.6737e-24, 0.0)
        assert_finite(g_m, "d(translational_U)/dm at T=0")
        assert_finite(g_T, "d(translational_U)/dT at T=0")


# =============================================================================
# Cubic splines
# =============================================================================

@pytest.fixture(scope="module")
def quadratic_knots():
    """Knots of y = x^2 on [0, 5], the fixture used by all spline tests."""
    t = jnp.linspace(0.0, 5.0, 6)
    u = t**2
    return t, u


@pytest.fixture(scope="module")
def spline_extrapolating(quadratic_knots):
    """Cubic spline through y = x^2 with flat extrapolation enabled."""
    t, u = quadratic_knots
    return cubic_spline(t, u, extrapolate=True)


@pytest.fixture(scope="module")
def spline_bounded(quadratic_knots):
    """Cubic spline through y = x^2 that rejects out-of-bounds evaluation."""
    t, u = quadratic_knots
    return cubic_spline(t, u, extrapolate=False)


class TestCubicSplineAutodiff:
    """Gradients of spline evaluation with respect to the evaluation point."""

    # Interior points, points immediately either side of a knot, and points
    # exactly on interior knots (where searchsorted switches interval).
    INTERIOR_POINTS = [0.37, 1.0, 1.0 - 1e-9, 1.0 + 1e-9, 2.5, 3.0, 4.62, 4.999]

    @pytest.mark.parametrize("t_eval", INTERIOR_POINTS)
    def test_interior_gradient_is_finite_and_nonzero(
        self, spline_extrapolating, t_eval
    ):
        """dy/dx of a spline through y = x^2 must be finite and positive."""
        g = jax.grad(spline_extrapolating)(t_eval)
        assert_finite(g, f"d(spline)/dt at {t_eval}")
        assert float(g) > 0.0, "y = x^2 is increasing on [0, 5]"

    @pytest.mark.parametrize("t_eval", INTERIOR_POINTS)
    def test_interior_gradient_matches_finite_difference(
        self, spline_extrapolating, t_eval
    ):
        """AD must reproduce central differences inside the domain.

        The step is 1e-5, small enough that the difference stays within a
        single cubic piece for the points that are not exactly on a knot, and
        small enough that the cubic truncation error stays below 1e-9 relative.
        Points sitting exactly on a knot straddle two pieces, but a natural
        cubic spline is C2 there, so the finite difference remains valid.
        """
        g = float(jax.grad(spline_extrapolating)(t_eval))
        fd = central_difference(spline_extrapolating, t_eval, h=1e-5)
        assert g == pytest.approx(fd, rel=1e-6)

    @pytest.mark.parametrize("t_eval", [-3.0, -1e-9, 5.0 + 1e-9, 12.0])
    def test_flat_extrapolation_has_exactly_zero_gradient(
        self, spline_extrapolating, t_eval
    ):
        """Outside the knots the spline is constant, so its gradient is zero.

        This is the check that a ``jnp.clip``-based guard is doing its job: a
        naive implementation that evaluated the cubic at the raw ``t_eval`` and
        then selected the clamped value would return a finite value here but a
        large (or NaN) cotangent.
        """
        g = jax.grad(spline_extrapolating)(t_eval)
        assert_finite(g, f"d(spline)/dt outside the domain at {t_eval}")
        assert float(g) == 0.0, (
            "flat extrapolation must have identically zero derivative outside "
            "the knot range"
        )

    @pytest.mark.parametrize("t_eval", [0.0, 5.0])
    def test_domain_endpoint_gradient_is_finite_subgradient(
        self, spline_extrapolating, spline_bounded, t_eval
    ):
        """At the clamp boundary the gradient is half the interior derivative.

        With ``extrapolate=True`` the spline is a kink at ``t[0]`` and
        ``t[-1]``: zero slope outside, the cubic's slope inside. JAX's
        ``clip``/``maximum`` convention splits ties evenly, so the reported
        subgradient is exactly half the one-sided interior derivative. That is a
        legitimate choice, but it is a silent factor-of-two, so pin it here: if
        the guard is ever replaced by something that leaks NaN, or by something
        that returns the full one-sided slope, this test says so.
        """
        g = float(jax.grad(spline_extrapolating)(t_eval))
        interior = float(jax.grad(spline_bounded)(t_eval))
        assert_finite(g, f"d(spline)/dt at the domain endpoint {t_eval}")
        assert interior != 0.0
        assert g == pytest.approx(0.5 * interior, rel=1e-12)

    @pytest.mark.parametrize("t_eval", [0.37, 2.5, 4.62])
    def test_bounded_spline_gradient_matches_extrapolating_spline(
        self, spline_extrapolating, spline_bounded, t_eval
    ):
        """extrapolate=False must not change derivatives strictly inside."""
        g_bounded = float(jax.grad(spline_bounded)(t_eval))
        g_extrap = float(jax.grad(spline_extrapolating)(t_eval))
        assert_finite(g_bounded, "d(bounded spline)/dt")
        assert g_bounded == pytest.approx(g_extrap, rel=1e-12)

    def test_second_derivative_is_finite(self, spline_extrapolating):
        """Nested differentiation must stay finite (needed for Hessians)."""
        for t_eval in (0.9, 2.3, 4.1):
            g2 = jax.grad(jax.grad(spline_extrapolating))(t_eval)
            assert_finite(g2, f"d2(spline)/dt2 at {t_eval}")
            assert float(g2) > 0.0, "y = x^2 is convex"

    def test_jacfwd_over_array_is_diagonal(self, spline_extrapolating):
        """Vector evaluation must be element-wise with a finite Jacobian."""
        pts = jnp.array([0.7, 1.9, 3.3, 4.4])
        jac = np.asarray(jax.jacfwd(spline_extrapolating)(pts))
        assert_finite(jac, "spline Jacobian")
        assert np.all(jac - np.diag(np.diag(jac)) == 0.0)
        assert np.all(np.diag(jac) > 0.0)

    def test_gradient_is_stable_under_jit(self, spline_extrapolating):
        """jit(grad(...)) must compile and agree with the eager gradient.

        ``extrapolate=True`` is required for this: the ``extrapolate=False``
        path does a Python ``if`` on a traced bounds check, which is fine under
        eager ``grad`` (the tracer wraps a concrete value) but raises
        ``TracerBoolConversionError`` under ``jit``. That asymmetry is checked
        in TestKnownNonDifferentiableCode.
        """
        eager = float(jax.grad(spline_extrapolating)(2.5))
        jitted = float(jax.jit(jax.grad(spline_extrapolating))(2.5))
        assert_finite(jitted, "jit(grad(spline))")
        assert jitted == pytest.approx(eager, rel=1e-12)

    @pytest.mark.parametrize("idx, t_eval", [(0, 0.4), (1, 1.7), (2, 2.2), (3, 3.2)])
    def test_integral_derivative_recovers_the_spline(
        self, spline_extrapolating, idx, t_eval
    ):
        """d/dt of the antiderivative must return the spline itself.

        The fundamental theorem of calculus is a far stronger statement about
        ``_integral`` than any finite difference: it ties the analytic
        polynomial coefficients to the interpolant they are supposed to
        integrate.
        """
        g = jax.grad(lambda x: spline_extrapolating._integral(idx, x))(t_eval)
        assert_finite(g, f"d(_integral)/dt in interval {idx}")
        assert float(g) == pytest.approx(float(spline_extrapolating(t_eval)), rel=1e-12)

    def test_gradient_with_respect_to_knot_ordinates_is_finite(
        self, spline_extrapolating
    ):
        """The evaluation kernel is differentiable in u, even though setup is not.

        ``cubic_spline`` solves for the second derivatives ``z`` with SciPy, so
        ``z`` is a constant as far as JAX is concerned and this is only a
        partial derivative. It is still worth guarding: it is the derivative
        that a future JAX-native spline constructor would have to reproduce,
        and it must not be NaN today.
        """

        def evaluate(u):
            spline = CubicSpline(
                spline_extrapolating.t,
                u,
                spline_extrapolating.h,
                spline_extrapolating.z,
                True,
            )
            return spline(2.5)

        g = jax.grad(evaluate)(spline_extrapolating.u)
        assert_finite(g, "d(spline)/du")
        # 2.5 sits inside interval [2, 3], so only u[2] and u[3] contribute.
        assert np.all(np.asarray(g)[[0, 1, 4, 5]] == 0.0)
        assert np.all(np.asarray(g)[[2, 3]] > 0.0)


# =============================================================================
# Post-processing: apply_LSF / apply_rotation / compute_LSF_matrix
# =============================================================================

def _synthetic_spectrum(n=101):
    """
    A normalised spectrum with a single absorption line, for broadening tests.

    Parameters
    ----------
    n : int, optional
        Number of pixels. Default: 101.

    Returns
    -------
    tuple
        ``(wl_spec, flux)`` where `wl_spec` is a ``(start, stop, step)`` tuple
        in Angstroms and `flux` is a float64 array.
    """
    wl_spec = (5000.0, 5000.0 + 0.01 * (n - 1), 0.01)
    x = np.linspace(0.0, 1.0, n)
    flux = 1.0 - 0.5 * np.exp(-(((x - 0.5) / 0.05) ** 2))
    return wl_spec, flux


def _grad(func, x, description):
    """
    Differentiate `func` at `x`, failing loudly if it is not traceable.

    The post-processing functions used to be pure NumPy (preallocated output
    arrays, ``bisect``-based window bounds, Python ``for`` loops), so these
    checks were marked ``xfail``. They are now written in JAX with fixed-size,
    masked convolution windows, so the gradient must exist -- a regression back
    to NumPy has to fail here rather than quietly skip.

    Parameters
    ----------
    func : callable
        Scalar-valued function of a single argument.
    x : float or array
        Point at which to differentiate.
    description : str
        Name of the function under test, used in the failure message.

    Returns
    -------
    array
        The gradient.
    """
    try:
        return jax.grad(func)(x)
    except _NOT_TRACEABLE as exc:  # pragma: no cover - regression guard
        raise AssertionError(
            f"{description} is no longer differentiable: {type(exc).__name__}: "
            f"{exc}. R and vsini are fitted parameters, so this code has to "
            "stay traceable by JAX."
        ) from exc


class TestApplyLSFDifferentiability:
    """LSF convolution: gradients w.r.t. flux and w.r.t. the resolving power R."""

    def test_values_are_finite(self):
        """A healthy value is the precondition for a meaningful gradient."""
        wl_spec, flux = _synthetic_spectrum()
        out = apply_LSF(flux, wl_spec, R=20000.0)
        assert_finite(out, "apply_LSF output")

    def test_gradient_with_respect_to_flux(self):
        """d(sum of convolved flux)/d(flux) should be the kernel column sums."""
        wl_spec, flux = _synthetic_spectrum()
        g = _grad(
            lambda f: jnp.sum(apply_LSF(f, wl_spec, 20000.0)),
            jnp.asarray(flux),
            "apply_LSF (w.r.t. flux)",
        )
        assert_finite(g, "d(apply_LSF)/d(flux)")
        assert np.all(np.asarray(g) > 0.0), (
            "convolution weights are strictly positive, so every input pixel "
            "must influence the output"
        )

    @pytest.mark.parametrize("R", [3000.0, 20000.0, 100000.0])
    def test_gradient_with_respect_to_R(self, R):
        """R is a fitted parameter, so d(flux)/dR must exist and be finite."""
        wl_spec, flux = _synthetic_spectrum()
        flux_j = jnp.asarray(flux)

        def line_depth(resolving_power):
            return jnp.min(apply_LSF(flux_j, wl_spec, resolving_power))

        g = _grad(line_depth, R, "apply_LSF (w.r.t. R)")
        assert_finite(g, "d(apply_LSF)/dR")
        fd = central_difference(line_depth, R, h=1e-5 * R)
        assert float(g) == pytest.approx(fd, rel=1e-5)

    @pytest.mark.parametrize("R", [3000.0, 20000.0])
    def test_finite_difference_sensitivity_to_R_is_finite_and_nonzero(self, R):
        """Value-level cross-check: the response to R must be finite and real.

        This is the value-level twin of the AD test above, and catches a kernel
        that silently returns NaN or that has no dependence on R at all --
        including the case where the fixed-size window mask has been made so
        wide (or so narrow) that R stops mattering.
        """
        wl_spec, flux = _synthetic_spectrum()
        h = 1e-5 * R
        response = (
            apply_LSF(flux, wl_spec, R + h) - apply_LSF(flux, wl_spec, R - h)
        ) / (2.0 * h)
        assert_finite(response, "d(apply_LSF)/dR by finite difference")
        assert np.max(np.abs(response)) > 0.0, "the LSF must actually depend on R"

    def test_infinite_R_is_the_identity(self):
        """R = inf short-circuits; the result must be bitwise the input."""
        wl_spec, flux = _synthetic_spectrum()
        out = apply_LSF(flux, wl_spec, R=np.inf)
        assert np.array_equal(out, flux)


class TestApplyRotationDifferentiability:
    """Rotational broadening: gradients w.r.t. flux and w.r.t. vsini."""

    @pytest.mark.parametrize("vsini", [0.0, 1e-6, 0.5, 5.0, 25.0])
    def test_values_are_finite(self, vsini):
        """The rotation kernel involves sqrt(1 - x^2) and arcsin(x) at x = +-1.

        Those are exactly the expressions that produce NaN one ulp outside the
        unit interval, so check the whole vsini sweep, including the vsini = 0
        boundary where the implementation short-circuits.
        """
        wl_spec, flux = _synthetic_spectrum()
        out = apply_rotation(flux, wl_spec, vsini)
        assert_finite(out, f"apply_rotation output at vsini={vsini}")

    def test_zero_vsini_is_the_identity(self):
        """vsini = 0 must return the input exactly, not approximately."""
        wl_spec, flux = _synthetic_spectrum()
        out = apply_rotation(flux, wl_spec, vsini=0.0)
        assert np.array_equal(out, flux)

    @pytest.mark.parametrize("vsini", [1e-9, 1e-6, 1e-3])
    def test_values_are_continuous_across_the_vsini_zero_boundary(self, vsini):
        """Approaching vsini = 0 from above must reproduce the identity branch.

        A discontinuity here would mean the short-circuit at vsini = 0 is
        papering over a different limit, which is the value-level analogue of a
        masked NaN.
        """
        wl_spec, flux = _synthetic_spectrum()
        out = apply_rotation(flux, wl_spec, vsini)
        assert_finite(out, f"apply_rotation output at vsini={vsini}")
        np.testing.assert_allclose(out, flux, rtol=0.0, atol=1e-12)

    @pytest.mark.parametrize("vsini", [5.0, 25.0])
    def test_gradient_with_respect_to_vsini(self, vsini):
        """vsini is a fitted parameter, so d(flux)/d(vsini) must be finite.

        vsini is sampled well above the 0.3 km/s at which the rotational
        kernel first becomes wider than one 0.01 A pixel; below that the
        discrete response is identically zero for grid reasons, not physics.
        """
        wl_spec, flux = _synthetic_spectrum()
        flux_j = jnp.asarray(flux)

        def line_depth(v):
            return jnp.min(apply_rotation(flux_j, wl_spec, v))

        g = _grad(line_depth, vsini, "apply_rotation (w.r.t. vsini)")
        assert_finite(g, "d(apply_rotation)/d(vsini)")
        assert float(g) != 0.0, "rotation must fill in the line core"
        fd = central_difference(line_depth, vsini, h=1e-4)
        assert float(g) == pytest.approx(fd, rel=1e-5)

    def test_gradient_with_respect_to_vsini_at_zero(self):
        """vsini = 0 is the boundary a `where` guard would hide a NaN behind."""
        wl_spec, flux = _synthetic_spectrum()
        flux_j = jnp.asarray(flux)
        g = _grad(
            lambda v: jnp.sum(apply_rotation(flux_j, wl_spec, v)),
            0.0,
            "apply_rotation (w.r.t. vsini at zero)",
        )
        assert_finite(g, "d(apply_rotation)/d(vsini) at vsini = 0")

    @pytest.mark.parametrize("vsini", [5.0, 25.0])
    def test_finite_difference_sensitivity_to_vsini_is_finite_and_nonzero(
        self, vsini
    ):
        """Value-level twin of the AD check on vsini."""
        wl_spec, flux = _synthetic_spectrum()
        h = 1e-4
        response = (
            apply_rotation(flux, wl_spec, vsini + h)
            - apply_rotation(flux, wl_spec, vsini - h)
        ) / (2.0 * h)
        assert_finite(response, "d(apply_rotation)/d(vsini) by finite difference")
        assert np.max(np.abs(response)) > 1e-6

    @pytest.mark.parametrize("vsini", [1.0, 5.0, 25.0])
    def test_kernel_is_normalised(self, vsini):
        """Broadening a flat continuum must return the same flat continuum.

        This is the sharpest available check on ``_rotation_kernel_integral``'s
        handling of ``|detuning| = delta_lambda_rot``, where the closed form
        contains ``sqrt(1 - ratio**2)`` and ``arcsin(ratio)`` evaluated exactly
        at the unit-circle boundary. The special case there returns
        ``sign(detuning) * 0.5``; if that value were wrong, or if the guard
        stopped catching the endpoint, the kernel weights would no longer sum
        to one and a flat spectrum would develop structure.
        """
        wl_spec, _ = _synthetic_spectrum(n=501)
        ones = np.ones(501)
        out = apply_rotation(ones, wl_spec, vsini)
        assert_finite(out, "apply_rotation of a flat continuum")
        np.testing.assert_allclose(out, ones, rtol=0.0, atol=1e-13)

    @pytest.mark.parametrize("vsini", [5.0, 25.0])
    def test_equivalent_width_is_conserved(self, vsini):
        """Rotation redistributes flux, so the integrated line must not move.

        Equivalently, d(sum flux)/d(vsini) = 0. The sum is taken over the
        interior of the grid only: pixels within one ``delta_lambda_rot`` of an
        edge exchange flux with wavelengths that were never synthesised, which
        is a boundary effect rather than a property of the kernel. The margin
        below is generous enough to cover ``delta_lambda_rot`` at 25 km/s.
        """
        n = 501
        wl_spec, flux = _synthetic_spectrum(n=n)
        # delta_lambda_rot = lambda * vsini / c; convert to a pixel count on the
        # 0.01 A grid and add slack.
        delta_lambda_rot_angstrom = 5000.0 * vsini * 1e5 / c_cgs
        margin = int(np.ceil(delta_lambda_rot_angstrom / 0.01)) + 2

        h = 1e-4
        hi = apply_rotation(flux, wl_spec, vsini + h)
        lo = apply_rotation(flux, wl_spec, vsini - h)
        d_sum = (np.sum(hi[margin:-margin]) - np.sum(lo[margin:-margin])) / (2.0 * h)

        equivalent_width = np.sum(1.0 - flux)
        assert np.isfinite(d_sum)
        assert abs(d_sum) < 1e-6 * equivalent_width


class TestComputeLSFMatrixDifferentiability:
    """LSF matrix construction: gradient of the matrix entries w.r.t. R."""

    def test_matrix_is_finite_and_normalised(self):
        """Each row must be a finite, unit-sum convolution kernel."""
        synth_wls = (5000.0, 5002.0, 0.01)
        obs_wls = np.linspace(5000.5, 5001.5, 11)
        matrix = compute_LSF_matrix(synth_wls, obs_wls, R=20000.0, verbose=False)
        assert_finite(matrix, "compute_LSF_matrix output")
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0, rtol=1e-12)

    @pytest.mark.parametrize("R", [1e4, 1e5, 1e6, 5e6, 1e7, 5e7, 1e8])
    def test_every_row_sums_to_one_at_any_resolving_power(self, R):
        """No row may be silently zero, however narrow the LSF gets.

        The failure this pins used to be invisible: once ``window_size * sigma``
        drops below half the synthesis grid spacing, an observed wavelength
        sitting between two synthesis pixels has *no* synthesis pixel inside its
        kernel window. The old implementation produced an empty slice, never
        wrote the row, and left it all zeros -- so the convolved flux became
        exactly zero with no warning, no NaN, and nothing a value test would
        trip on. The physically sensible answer is nearest-neighbour
        interpolation: you cannot resolve an LSF finer than your synthesis
        sampling.

        The observed wavelengths below are deliberately offset by half a
        synthesis pixel (5000.005, 5000.505) so that they fall exactly between
        grid points, which is the configuration that triggered the bug.
        """
        synth_wls = (5000.0, 5001.0, 0.01)
        obs_wls = np.array([5000.005, 5000.505])
        matrix = compute_LSF_matrix(synth_wls, obs_wls, R, verbose=False)

        assert_finite(matrix, f"compute_LSF_matrix output at R={R:g}")
        assert np.all(np.asarray(matrix) >= 0.0), "weights must be non-negative"
        np.testing.assert_allclose(
            np.asarray(matrix).sum(axis=1), 1.0, rtol=1e-12,
            err_msg=(
                f"LSF matrix rows are not normalised at R={R:g}. A row sum of "
                "zero means the kernel window fell between two synthesis "
                "pixels and the row was never written."
            ),
        )

    def test_a_kernel_narrower_than_the_grid_warns(self):
        """The zero-row fallback means the synthesis grid is too coarse; say so."""
        synth_wls = (5000.0, 5001.0, 0.01)
        obs_wls = np.array([5000.005, 5000.505])
        with pytest.warns(UserWarning, match="narrower than the synthesis grid"):
            compute_LSF_matrix(synth_wls, obs_wls, R=5e7, verbose=True)

    def test_narrow_kernel_falls_back_to_the_nearest_pixel(self):
        """The fallback must be nearest-neighbour, not an arbitrary pixel."""
        synth_wls = (5000.0, 5001.0, 0.01)
        # 5000.0040 is nearest synthesis pixel 0 (5000.00); 5000.5070 is
        # nearest pixel 51 (5000.51).
        obs_wls = np.array([5000.0040, 5000.5070])
        matrix = np.asarray(
            compute_LSF_matrix(synth_wls, obs_wls, R=5e7, verbose=False)
        )
        assert np.argmax(matrix[0]) == 0
        assert np.argmax(matrix[1]) == 51
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0, rtol=1e-12)

    @pytest.mark.parametrize("R", [5000.0, 20000.0])
    def test_gradient_of_convolved_flux_with_respect_to_R(self, R):
        """The matrix is only useful if the fit can differentiate through R."""
        synth_wls = (5000.0, 5002.0, 0.01)
        obs_wls = np.linspace(5000.5, 5001.5, 11)
        n_synth = 201
        x = np.linspace(0.0, 1.0, n_synth)
        flux = jnp.asarray(1.0 - 0.5 * np.exp(-(((x - 0.5) / 0.05) ** 2)))

        def core_depth(resolving_power):
            matrix = compute_LSF_matrix(
                synth_wls, obs_wls, resolving_power, verbose=False
            )
            return jnp.min(matrix @ flux)

        g = _grad(core_depth, R, "compute_LSF_matrix (w.r.t. R)")
        assert_finite(g, "d(LSF matrix @ flux)/dR")
        fd = central_difference(core_depth, R, h=1e-5 * R)
        assert float(g) == pytest.approx(fd, rel=1e-5)

    @pytest.mark.parametrize("R", [5000.0, 20000.0])
    def test_finite_difference_sensitivity_to_R_is_finite_and_nonzero(self, R):
        """Value-level twin of the AD check on R."""
        synth_wls = (5000.0, 5002.0, 0.01)
        obs_wls = np.linspace(5000.5, 5001.5, 11)
        h = 1e-5 * R
        hi = compute_LSF_matrix(synth_wls, obs_wls, R + h, verbose=False)
        lo = compute_LSF_matrix(synth_wls, obs_wls, R - h, verbose=False)
        response = (hi - lo) / (2.0 * h)
        assert_finite(response, "d(LSF matrix)/dR by finite difference")
        assert np.max(np.abs(response)) > 0.0


# =============================================================================
# Code that is deliberately not differentiable
# =============================================================================

class TestKnownNonDifferentiableCode:
    """Pin the boundary between the differentiable and non-differentiable code.

    Nothing here is a bug: wavelength grids, window indices and interval
    membership are integer- or boolean-valued by construction. The tests exist
    so that the boundary is explicit, and so that anyone who moves a function
    across it is forced to notice.
    """

    def test_air_vacuum_conversion_has_exactly_one_implementation(self):
        """Every import path must resolve to the same ``korg.utils`` function.

        There used to be three copies: ``korg.utils`` (correct constants, A
        only), ``korg.wavelengths`` (correct constants plus ``cgs`` handling,
        but NumPy-based and therefore untraceable) and ``korg.linelist``
        (truncated constants, mislabelled as Edlen 1966, and the copy
        re-exported as ``korg.air_to_vacuum``). Korg.jl defines these once, in
        utils.jl, so a divergence here is a physics fork rather than a style
        issue.
        """
        import korg
        import korg.linelist as linelist_module
        import korg.wavelengths as wavelengths_module

        for module in (korg, wavelengths_module, linelist_module):
            assert module.air_to_vacuum is air_to_vacuum
            assert module.vacuum_to_air is vacuum_to_air

        # The surviving copy is the traceable one, and carries Korg.jl's
        # ``cgs=λ<1`` auto-detection: A and cm inputs agree exactly.
        assert_finite(jax.grad(wavelengths_module.air_to_vacuum)(5000.0),
                      "d(air_to_vacuum)/dlambda")
        assert float(air_to_vacuum(5000.0e-8)) * 1e8 == pytest.approx(
            float(air_to_vacuum(5000.0)), rel=1e-12
        )
        assert float(air_to_vacuum(5000.0e-8, cgs=True)) * 1e8 == pytest.approx(
            float(air_to_vacuum(5000.0, cgs=False)), rel=1e-12
        )

    def test_wavelength_grid_is_a_finite_constant(self):
        """``Wavelengths`` builds a NumPy grid; it is a constant, not a variable.

        Wavelength grids are chosen by the user, never fitted, so there is
        nothing to differentiate. What matters downstream is that the grid and
        the derived frequencies are finite and strictly monotonic, since they
        divide into other expressions.
        """
        from korg.wavelengths import Wavelengths

        wls = Wavelengths([(5000.0, 5010.0, 0.01), (6000.0, 6010.0, 0.01)])
        assert_finite(wls.all_wls, "Wavelengths.all_wls")
        assert_finite(wls.all_freqs, "Wavelengths.all_freqs")
        assert np.all(np.diff(wls.all_wls) > 0.0)
        assert np.all(np.diff(wls.all_freqs) > 0.0)
        assert np.all(wls.all_wls > 0.0), "a zero wavelength would divide by zero"
        np.testing.assert_allclose(
            wls.all_freqs, c_cgs / wls.all_wls[::-1], rtol=1e-15
        )

    def test_search_helpers_return_integers(self):
        """``searchsortedfirst``/``searchsortedlast`` are index maps, not floats."""
        from korg.wavelengths import Wavelengths

        wls = Wavelengths((5000.0, 5001.0, 0.01))
        assert isinstance(wls.searchsortedfirst(5000.5), int)
        assert isinstance(wls.searchsortedlast(5000.5), int)
        assert isinstance(wls.subspectrum_indices()[0][0], int)

    def test_interval_helpers_are_boolean_and_integer_valued(self):
        """Interval membership and slicing are discrete by design."""
        from korg.utils import Interval, closed_interval, contained, contained_slice

        interval = Interval(3.0, 10.0)
        # NumPy chained comparison returns np.bool_, so compare by value.
        assert bool(contained(5.0, interval)) is True
        assert bool(contained(3.0, interval)) is False
        assert bool(contained(3.0, closed_interval(3.0, 10.0))) is True

        start, end = contained_slice([1.0, 2.0, 5.0, 8.0, 12.0], interval)
        assert isinstance(start, int) and isinstance(end, int)
        assert (start, end) == (2, 4)

    def test_bounded_spline_cannot_be_jitted(self):
        """``extrapolate=False`` branches on a traced bounds check.

        Eager ``jax.grad`` survives this because its tracers wrap concrete
        values, but ``jit`` does not. Use ``extrapolate=True`` inside jitted
        code. Recorded here so the asymmetry is not rediscovered the hard way.
        """
        t = jnp.linspace(0.0, 5.0, 6)
        spline = cubic_spline(t, t**2, extrapolate=False)
        with pytest.raises(jax.errors.ConcretizationTypeError):
            jax.jit(jax.grad(spline))(2.5)

    def test_spline_construction_is_not_traceable(self):
        """``cubic_spline`` solves for z with SciPy, so knots cannot be fitted.

        Every spline in Korg is built from a fixed data table (partition
        functions, equilibrium constants, Stark profiles), so this is by design.
        It does mean that the derivative of a spline w.r.t. its knot ordinates
        is incomplete; see the corresponding test in TestCubicSplineAutodiff.
        """
        t = jnp.linspace(0.0, 5.0, 6)
        with pytest.raises(_NOT_TRACEABLE):
            jax.grad(lambda u: cubic_spline(t, u, extrapolate=True)(2.5))(t**2)
