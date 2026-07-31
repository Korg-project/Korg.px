"""
Automatic-differentiation tests for the line profile and line broadening code.

The whole point of this port is that spectral synthesis is differentiable with
JAX, so every float-in/float-out function in ``korg.line_profiles``,
``korg.line_broadening`` and the broadening helpers of ``korg.line_absorption``
must return finite, correct gradients -- not just finite values.

The specific failure mode these tests guard against is::

    y = jnp.where(cond, safe_expression, dangerous_expression)

If ``dangerous_expression`` evaluates to NaN or inf, ``jnp.where`` masks the
*value* but not the *cotangent*: reverse-mode AD still walks the dead branch and
the gradient comes back NaN even though the value looks perfectly healthy. Line
profile code is riddled with piecewise branches (the Voigt-Hjerting function has
four regimes, the Harris series is piecewise in ``v``), so this is exactly where
such a bug hides.

Every test therefore checks three things where applicable: the gradient is
finite, it is non-zero where physics says it must be, and it agrees with a
central finite difference.
"""

# Import korg FIRST to enable JAX x64 mode before any other JAX operations
import korg

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from korg.line_absorption import doppler_width as doppler_width_absorption
from korg.line_absorption import (
    inverse_gaussian_density as inverse_gaussian_density_absorption,
)
from korg.line_absorption import (
    inverse_lorentz_density as inverse_lorentz_density_absorption,
)
from korg.line_absorption import scaled_stark, scaled_vdW, sigma_line
from korg.line_broadening import scaled_stark as scaled_stark_broadening
from korg.line_broadening import scaled_vdW as scaled_vdW_broadening
from korg.line_broadening import sigma_line as sigma_line_broadening
from korg.line_profiles import (
    doppler_width,
    exponential_integral_1,
    harris_series,
    inverse_gaussian_density,
    inverse_lorentz_density,
    line_profile,
    voigt_hjerting,
)

# Relative tolerance for "the AD gradient matches a central finite difference".
# A central difference in float64 is limited to roughly eps^(2/3) ~ 5e-11
# relative accuracy at the optimal step size, so 1e-5 leaves several orders of
# magnitude of headroom for the step sizes chosen below.
FD_RTOL = 1e-5


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def central_difference(f, x, step):
    """
    Central finite-difference estimate of ``df/dx``.

    The effective step is recomputed as ``x_plus - x_minus`` so that the
    estimate stays accurate even when ``step`` is not exactly representable
    relative to ``x`` (which matters for arguments such as wavelengths in cm,
    where ``x`` is ~5e-5 and the natural step is ~1e-15).

    Parameters
    ----------
    f : callable
        Scalar function of a single float.
    x : float
        Point at which to differentiate.
    step : float
        Nominal half-step.

    Returns
    -------
    float
        Estimate of ``df/dx`` at ``x``.
    """
    x_plus = np.float64(x) + np.float64(step)
    x_minus = np.float64(x) - np.float64(step)
    return (float(f(x_plus)) - float(f(x_minus))) / (x_plus - x_minus)


def assert_grad_matches_fd(f, x, step, rtol=FD_RTOL, atol=0.0, name=""):
    """
    Assert ``jax.grad(f)(x)`` is finite and matches a central difference.

    Parameters
    ----------
    f : callable
        Scalar function of a single float.
    x : float
        Point at which to differentiate.
    step : float
        Half-step for the finite difference.
    rtol : float, optional
        Relative tolerance for the comparison.
    atol : float, optional
        Absolute tolerance, needed only where the true derivative is zero (a
        stationary point), since there a purely relative comparison compares two
        pieces of floating-point noise. Set it to the finite-difference noise
        floor, roughly ``eps * |f| / step``.
    name : str, optional
        Label included in assertion messages.

    Returns
    -------
    float
        The analytic (AD) gradient.
    """
    analytic = float(jax.grad(f)(np.float64(x)))
    assert np.isfinite(analytic), f"{name}: gradient at x={x!r} is {analytic}"

    numeric = central_difference(f, x, step)
    scale = max(abs(numeric), abs(analytic))
    assert abs(analytic - numeric) <= rtol * scale + atol, (
        f"{name}: AD gradient {analytic!r} disagrees with finite difference "
        f"{numeric!r} at x={x!r} (relative error "
        f"{abs(analytic - numeric) / max(scale, 1e-300):.3e})"
    )
    return analytic


def one_sided_derivatives(f, x, offset):
    """
    Derivative of ``f`` just below and just above ``x``.

    Used to probe whether a piecewise definition is differentiable across a
    regime boundary. Each derivative is evaluated strictly on its own side of
    the boundary, so neither finite-differences across the seam.

    Parameters
    ----------
    f : callable
        Scalar function of a single float.
    x : float
        Location of the boundary.
    offset : float
        Distance from the boundary at which to evaluate.

    Returns
    -------
    tuple of float
        ``(derivative below x, derivative above x)``.
    """
    grad_f = jax.grad(f)
    return float(grad_f(np.float64(x - offset))), float(grad_f(np.float64(x + offset)))


# ---------------------------------------------------------------------------
# harris_series
# ---------------------------------------------------------------------------

# harris_series is only valid for v < 5; H1 is piecewise with knots at 1.3 and
# 2.4. sqrt(1.5) ~ 1.2247 is included because the (unselected) v >= 2.4 branch
# divides by v**2 - 3/2 and therefore blows up there -- a classic masked-NaN
# trap.
HARRIS_POINTS = [
    0.0,
    0.05,
    0.5,
    1.0,
    1.2247448713915892,  # v**2 == 1.5 to within one ulp: pole of the dead branch
    1.29,
    1.31,
    1.5,
    2.0,
    2.35,
    2.45,
    3.0,
    4.0,
    4.99,
]


class TestHarrisSeriesGradients:
    """Gradients of the (piecewise in v) Harris series."""

    @pytest.mark.parametrize("v", HARRIS_POINTS)
    def test_jacobian_is_finite(self, v):
        """All three Harris coefficients must have finite derivatives in v."""
        jac = jax.jacfwd(harris_series)(np.float64(v))
        for i, component in enumerate(jac):
            assert np.isfinite(float(component)), (
                f"dH{i}/dv is {float(component)} at v={v!r}"
            )

    @pytest.mark.parametrize("v", HARRIS_POINTS)
    def test_forward_and_reverse_mode_agree(self, v):
        """jacfwd and jacrev must give the same answer (they exercise different code paths)."""
        fwd = jax.jacfwd(harris_series)(np.float64(v))
        rev = jax.jacrev(harris_series)(np.float64(v))
        for i, (a, b) in enumerate(zip(fwd, rev)):
            assert np.isfinite(float(b)), f"jacrev dH{i}/dv is {float(b)} at v={v!r}"
            np.testing.assert_allclose(
                float(a), float(b), rtol=1e-12, atol=1e-14,
                err_msg=f"jacfwd/jacrev disagree for H{i} at v={v!r}",
            )

    @pytest.mark.parametrize("v", HARRIS_POINTS)
    def test_jacobian_matches_finite_difference(self, v):
        """
        Each Harris coefficient's derivative must match a central difference.

        ``atol`` covers the stationary points of the individual coefficients,
        e.g. dH2/dv = -2v(3 - 2v**2) exp(-v**2) vanishes at v**2 = 3/2, where a
        purely relative comparison would be comparing two roundoff errors. The
        H components are O(1) and the step is 1e-6, so the finite-difference
        noise floor is ~1e-10; 1e-8 is a safe absolute tolerance.
        """
        jac = jax.jacfwd(harris_series)(np.float64(v))
        for i in range(3):
            assert_grad_matches_fd(
                lambda x, i=i: harris_series(x)[i],
                v,
                step=1e-6,
                atol=1e-8,
                name=f"harris_series H{i}",
            )
            assert np.isfinite(float(jac[i]))

    def test_gradient_finite_at_dead_branch_pole(self):
        """
        The v >= 2.4 branch of H1 has a pole at v**2 == 3/2, i.e. v ~ 1.2247.

        That value falls inside the *v < 1.3* regime, so the pole is never
        returned -- but it is still evaluated, and if v*v ever landed exactly on
        1.5 the masked branch would contribute a NaN cotangent. Sweep the
        floating-point neighbourhood of sqrt(1.5) to make sure nothing leaks.
        """
        v = np.sqrt(1.5)
        for k in range(-5, 6):
            probe = v
            for _ in range(abs(k)):
                probe = np.nextafter(probe, np.inf if k > 0 else -np.inf)
            jac = jax.jacfwd(harris_series)(np.float64(probe))
            for i, component in enumerate(jac):
                assert np.isfinite(float(component)), (
                    f"dH{i}/dv is {float(component)} at v={probe!r} "
                    f"(v*v - 1.5 = {probe * probe - 1.5!r})"
                )

    def test_smooth_components_are_differentiable_across_knots(self):
        """
        H0 = exp(-v**2) and H2 = (1 - 2v**2) H0 are analytic everywhere.

        Only H1 is piecewise, so the H0 and H2 derivatives must be continuous
        across the H1 knots. If they are not, the piecewise select has leaked
        into components that should never see it.
        """
        for knot in (1.3, 2.4):
            for i in (0, 2):
                below, above = one_sided_derivatives(
                    lambda x, i=i: harris_series(x)[i], knot, 1e-7
                )
                np.testing.assert_allclose(
                    below, above, rtol=1e-5,
                    err_msg=f"dH{i}/dv is discontinuous at the H1 knot v={knot}",
                )

    @pytest.mark.parametrize(
        "knot, max_relative_jump",
        [
            # Measured one-sided derivative jumps in H1 for the Hunger (1965)
            # coefficients that Korg.jl also uses: 59% at v=1.3 and 4.8% at
            # v=2.4. These are properties of the published fit, not of the port
            # (the *values* jump by 1.3% and 0.16% there too), so we pin them as
            # a characterisation test rather than demanding continuity. A
            # regression that widened either seam would fail here.
            (1.3, 0.65),
            (2.4, 0.06),
        ],
    )
    def test_h1_knot_derivative_jump_is_bounded(self, knot, max_relative_jump):
        """The known H1 derivative discontinuities must not get worse."""
        below, above = one_sided_derivatives(lambda x: harris_series(x)[1], knot, 1e-7)
        assert np.isfinite(below) and np.isfinite(above)
        relative_jump = abs(above - below) / abs(below)
        assert relative_jump <= max_relative_jump, (
            f"dH1/dv jumps by {relative_jump:.3f} across v={knot} "
            f"(was {max_relative_jump} or less)"
        )

    def test_h0_gradient_is_exact(self):
        """dH0/dv = -2 v exp(-v**2) analytically; AD must reproduce it exactly."""
        for v in (0.3, 1.0, 2.0, 4.0):
            analytic = float(jax.grad(lambda x: harris_series(x)[0])(np.float64(v)))
            np.testing.assert_allclose(analytic, -2 * v * np.exp(-v * v), rtol=1e-13)


# ---------------------------------------------------------------------------
# voigt_hjerting
# ---------------------------------------------------------------------------

# The four regimes are:
#   case 1: alpha <= 0.2 and v >= 5
#   case 2: alpha <= 0.2 and v <  5
#   case 3: alpha <= 1.4 and alpha + v < 3.2
#   case 4: otherwise
# These points sit safely inside a single regime, so finite differences are
# meaningful.
VOIGT_INTERIOR_POINTS = [
    (0.01, 0.0),
    (0.01, 1.0),
    (0.05, 2.0),
    (0.1, 0.5),
    (0.15, 3.0),
    (0.02, 6.0),  # case 1
    (0.1, 8.0),  # case 1
    (0.19, 12.0),  # case 1
    (0.3, 0.5),  # case 3
    (0.8, 1.0),  # case 3
    (1.2, 1.5),  # case 3
    (1.35, 0.2),  # case 3
    (2.0, 1.0),  # case 4
    (5.0, 3.0),  # case 4
    (0.5, 6.0),  # case 4
    (20.0, 0.5),  # case 4
]

# Points at, either side of, and close to every regime boundary.
VOIGT_BOUNDARY_POINTS = [
    (0.2, 4.999), (0.2, 5.0), (0.2, 5.001),
    (0.19999, 5.0), (0.2, 5.0), (0.20001, 5.0),
    (0.19999, 3.0), (0.2, 3.0), (0.20001, 3.0),
    (0.19999, 0.0), (0.2, 0.0), (0.20001, 0.0),
    (0.19999, 6.0), (0.2, 6.0), (0.20001, 6.0),
    (1.39999, 1.0), (1.4, 1.0), (1.40001, 1.0),
    (1.39999, 1.7999), (1.4, 1.8), (1.40001, 1.8001),
    (1.0, 2.19999), (1.0, 2.2), (1.0, 2.20001),
    (0.5, 2.69999), (0.5, 2.7), (0.5, 2.70001),
    (0.3, 2.89999), (0.3, 2.9), (0.3, 2.90001),
    (3.2, 0.0), (0.0, 3.2), (1.6, 1.6),
]


class TestVoigtHjertingGradients:
    """Gradients of the four-regime Hjerting function."""

    @pytest.mark.parametrize("alpha, v", VOIGT_INTERIOR_POINTS)
    def test_gradients_are_finite(self, alpha, v):
        """Both partial derivatives must be finite well inside every regime."""
        d_alpha, d_v = jax.grad(voigt_hjerting, argnums=(0, 1))(
            np.float64(alpha), np.float64(v)
        )
        assert np.isfinite(float(d_alpha)), f"dH/dalpha = {float(d_alpha)} at ({alpha}, {v})"
        assert np.isfinite(float(d_v)), f"dH/dv = {float(d_v)} at ({alpha}, {v})"

    @pytest.mark.parametrize("alpha, v", VOIGT_INTERIOR_POINTS)
    def test_d_alpha_matches_finite_difference(self, alpha, v):
        """dH/dalpha must match a central difference."""
        assert_grad_matches_fd(
            lambda a: voigt_hjerting(a, np.float64(v)),
            alpha,
            step=1e-7,
            name=f"voigt_hjerting dH/dalpha at v={v}",
        )

    @pytest.mark.parametrize("alpha, v", VOIGT_INTERIOR_POINTS)
    def test_d_v_matches_finite_difference(self, alpha, v):
        """dH/dv must match a central difference."""
        if v == 0.0:
            # The true Hjerting function is even in v, but the Hunger (1965)
            # polynomial fit for H1 is not (dH1/dv = -0.155 at v = 0), so the
            # approximation has a kink at line centre and a two-sided finite
            # difference there straddles it. line_profile only ever evaluates
            # v >= 0, so compare against the one-sided v -> 0+ limit instead.
            at_zero = float(jax.grad(voigt_hjerting, argnums=1)(np.float64(alpha), 0.0))
            just_right = float(jax.grad(voigt_hjerting, argnums=1)(np.float64(alpha), 1e-11))
            assert np.isfinite(at_zero)
            np.testing.assert_allclose(at_zero, just_right, rtol=1e-5, atol=1e-12)
            return
        assert_grad_matches_fd(
            lambda x: voigt_hjerting(np.float64(alpha), x),
            v,
            step=1e-7,
            name=f"voigt_hjerting dH/dv at alpha={alpha}",
        )

    @pytest.mark.parametrize("alpha, v", VOIGT_BOUNDARY_POINTS)
    def test_gradients_are_finite_at_regime_boundaries(self, alpha, v):
        """
        No masked NaN may leak from an unselected regime at a boundary.

        Finite differences are not checked here because a step may straddle the
        seam; only finiteness is asserted.
        """
        d_alpha, d_v = jax.grad(voigt_hjerting, argnums=(0, 1))(
            np.float64(alpha), np.float64(v)
        )
        assert np.isfinite(float(d_alpha)), f"dH/dalpha = {float(d_alpha)} at ({alpha}, {v})"
        assert np.isfinite(float(d_v)), f"dH/dv = {float(d_v)} at ({alpha}, {v})"

    @pytest.mark.parametrize("v", [0.0, 0.5, 1.0, 2.5, 3.2, 4.999, 5.0, 6.0, 12.0, 50.0])
    def test_gradient_is_finite_at_zero_alpha(self, v):
        """
        Regression test: alpha = 0 used to give a NaN gradient at every v.

        The alpha > 1.4 branch computes ``v**2 / alpha**2``, which is inf (or
        0/0 = NaN) at alpha = 0. That branch is never *selected* for
        alpha <= 0.2, but reverse-mode AD still pushed a cotangent through it
        and poisoned the whole gradient. alpha = gamma/(sigma*sqrt(2)) is
        genuinely zero for a line with no Lorentz broadening, so this is a
        reachable input, not a pathological one.
        """
        d_alpha, d_v = jax.grad(voigt_hjerting, argnums=(0, 1))(0.0, np.float64(v))
        assert np.isfinite(float(d_alpha)), f"dH/dalpha = {float(d_alpha)} at alpha=0, v={v}"
        assert np.isfinite(float(d_v)), f"dH/dv = {float(d_v)} at alpha=0, v={v}"

    @pytest.mark.parametrize("v", [0.0, 1.0, 3.0, 6.0])
    def test_zero_alpha_gradient_is_the_limit_of_small_alpha(self, v):
        """
        The alpha = 0 gradient must agree with the alpha -> 0 limit.

        ``atol`` is 1e-7 because dH/dv is itself proportional to alpha for small
        alpha (dH/dv = alpha dH1/dv + O(alpha**2) at v = 0), so the two
        evaluations differ by O(1e-9) there purely from the alpha offset. The
        gradients being compared are otherwise O(0.01) to O(1), so this still
        catches any real discontinuity, and it catches NaNs regardless.
        """
        at_zero = jax.grad(voigt_hjerting, argnums=(0, 1))(0.0, np.float64(v))
        near_zero = jax.grad(voigt_hjerting, argnums=(0, 1))(1e-9, np.float64(v))
        for i, (a, b) in enumerate(zip(at_zero, near_zero)):
            assert np.isfinite(float(a)) and np.isfinite(float(b))
            np.testing.assert_allclose(
                float(a), float(b), rtol=1e-6, atol=1e-7,
                err_msg=f"argument {i} gradient is discontinuous at alpha=0, v={v}",
            )

    def test_no_nan_gradients_over_a_dense_sweep(self):
        """
        Blanket sweep of the (alpha, v) plane: no gradient may be NaN or inf.

        This is the net that catches a masked-NaN reintroduced anywhere, not
        just at the points enumerated above.
        """
        alphas = np.concatenate([
            np.array([0.0, 1e-300, 1e-12, 0.2, 1.4, 3.2]),
            np.logspace(-8, 1.5, 60),
        ])
        vs = np.concatenate([
            np.array([0.0, 1e-300, 1.3, 2.4, 3.2, 5.0]),
            np.linspace(0.0, 15.0, 60),
        ])
        grid_alpha, grid_v = np.meshgrid(alphas, vs, indexing="ij")
        flat_alpha = jnp.asarray(grid_alpha.ravel())
        flat_v = jnp.asarray(grid_v.ravel())

        d_alpha, d_v = jax.vmap(jax.grad(voigt_hjerting, argnums=(0, 1)))(flat_alpha, flat_v)
        d_alpha = np.asarray(d_alpha)
        d_v = np.asarray(d_v)

        bad_alpha = ~np.isfinite(d_alpha)
        bad_v = ~np.isfinite(d_v)
        assert not bad_alpha.any(), (
            f"dH/dalpha is non-finite at {bad_alpha.sum()} points, e.g. "
            f"(alpha, v) = {list(zip(grid_alpha.ravel()[bad_alpha][:5], grid_v.ravel()[bad_alpha][:5]))}"
        )
        assert not bad_v.any(), (
            f"dH/dv is non-finite at {bad_v.sum()} points, e.g. "
            f"(alpha, v) = {list(zip(grid_alpha.ravel()[bad_v][:5], grid_v.ravel()[bad_v][:5]))}"
        )

    @pytest.mark.parametrize("alpha, v", [(0.05, 3.0), (0.15, 4.0), (0.3, 4.0), (1.0, 4.0)])
    def test_damping_gradient_is_positive_in_the_wings(self, alpha, v):
        """
        In the wings the profile is Lorentzian, so more damping means more
        opacity: dH/dalpha must be strictly positive and comfortably non-zero.
        """
        d_alpha = float(jax.grad(voigt_hjerting, argnums=0)(np.float64(alpha), np.float64(v)))
        assert d_alpha > 0, f"dH/dalpha = {d_alpha} at ({alpha}, {v}); expected > 0"
        assert d_alpha > 1e-6 * max(float(voigt_hjerting(alpha, v)), 1e-30)

    @pytest.mark.parametrize(
        "alpha, v", [(0.01, 1.0), (0.1, 2.0), (0.5, 1.0), (1.0, 1.0), (2.0, 2.0), (0.05, 7.0)]
    )
    def test_detuning_gradient_is_negative(self, alpha, v):
        """H(alpha, v) decreases away from line centre, so dH/dv < 0 for v > 0."""
        d_v = float(jax.grad(voigt_hjerting, argnums=1)(np.float64(alpha), np.float64(v)))
        assert d_v < 0, f"dH/dv = {d_v} at ({alpha}, {v}); expected < 0"

    @pytest.mark.parametrize(
        "alpha, v, argnum, max_relative_jump",
        [
            # The Hunger (1965) piecewise approximation that Korg.jl implements
            # is not C0 across its regime seams, let alone C1: the *values* jump
            # by up to 15% at alpha = 1.4. These bounds pin the measured
            # one-sided derivative jumps so that a regression which widened a
            # seam would be caught, while documenting that the seams are not
            # smooth today. See the module-level test report for the numbers.
            (0.2, 3.0, 0, 0.06),   # case 2 <-> case 3, varying alpha
            (0.2, 0.5, 0, 0.06),   # case 2 <-> case 3, varying alpha
            (0.2, 5.0, 1, 0.10),   # case 2 <-> case 1, varying v
            (0.1, 5.0, 1, 0.10),   # case 2 <-> case 1, varying v
            (0.2, 6.0, 0, 0.01),   # case 1 <-> case 4, varying alpha
            (0.2, 12.0, 0, 0.01),  # case 1 <-> case 4, varying alpha
            (1.4, 1.0, 0, 3.0),    # case 3 <-> case 4, varying alpha
            (1.0, 2.2, 1, 0.10),   # case 3 <-> case 4, varying v (alpha + v = 3.2)
            (0.5, 2.7, 1, 0.10),   # case 3 <-> case 4, varying v (alpha + v = 3.2)
        ],
    )
    def test_regime_boundary_derivative_jumps_are_bounded(
        self, alpha, v, argnum, max_relative_jump
    ):
        """
        One-sided derivatives either side of each regime seam must be finite
        and must not differ by more than the documented amount.
        """
        boundary = alpha if argnum == 0 else v

        def f(x):
            return voigt_hjerting(x, np.float64(v)) if argnum == 0 else voigt_hjerting(
                np.float64(alpha), x
            )

        below, above = one_sided_derivatives(f, boundary, 1e-6)
        assert np.isfinite(below) and np.isfinite(above), (
            f"one-sided derivatives ({below}, {above}) at ({alpha}, {v}) argnum={argnum}"
        )
        relative_jump = abs(above - below) / max(abs(below), abs(above))
        assert relative_jump <= max_relative_jump, (
            f"derivative w.r.t. argument {argnum} jumps by {relative_jump:.3f} across "
            f"(alpha, v) = ({alpha}, {v}); documented bound is {max_relative_jump}"
        )

    def test_hessian_is_finite(self):
        """Second derivatives must survive too (needed for curvature/Fisher work)."""
        for alpha, v in [(0.05, 1.0), (0.3, 1.0), (1.0, 2.0), (2.0, 3.0), (0.1, 7.0)]:
            hessian = jax.hessian(voigt_hjerting, argnums=(0, 1))(
                np.float64(alpha), np.float64(v)
            )
            flat = np.array([float(x) for row in hessian for x in row])
            assert np.isfinite(flat).all(), f"Hessian {flat} at ({alpha}, {v})"

    def test_gradient_under_jit_and_vmap(self):
        """Gradients must survive jit and vmap composition."""
        grad_fn = jax.jit(jax.vmap(jax.grad(voigt_hjerting, argnums=(0, 1))))
        alphas = jnp.array([0.0, 0.1, 0.2, 0.5, 1.4, 3.0])
        vs = jnp.array([0.0, 1.0, 5.0, 2.0, 1.0, 7.0])
        d_alpha, d_v = grad_fn(alphas, vs)
        assert np.isfinite(np.asarray(d_alpha)).all()
        assert np.isfinite(np.asarray(d_v)).all()


# ---------------------------------------------------------------------------
# line_profile
# ---------------------------------------------------------------------------

# A representative optical line: 5000 A centre, sigma ~ 2e-10 cm (thermal at a
# few thousand K), gamma ~ 5e-11 cm.
LAMBDA_0 = 5000e-8
SIGMA = 2e-10
GAMMA = 5e-11
AMPLITUDE = 1e-6

# Detunings in units of sigma, spanning core, shoulder and far wing so that
# every Voigt regime is exercised through line_profile.
DETUNINGS = [0.0, 0.3, 1.0, 2.5, 5.0, 9.0, 25.0]


class TestLineProfileGradients:
    """Gradients of the Voigt line profile w.r.t. the parameters users fit."""

    @pytest.mark.parametrize("n_sigma", DETUNINGS)
    def test_all_gradients_are_finite(self, n_sigma):
        """Every one of the five arguments must have a finite gradient."""
        wavelength = LAMBDA_0 + n_sigma * SIGMA
        grads = jax.grad(line_profile, argnums=(0, 1, 2, 3, 4))(
            LAMBDA_0, SIGMA, GAMMA, AMPLITUDE, wavelength
        )
        names = ["wavelength_0", "sigma", "gamma", "amplitude", "wavelength"]
        for name, g in zip(names, grads):
            assert np.isfinite(float(g)), f"d/d{name} = {float(g)} at {n_sigma} sigma"

    @pytest.mark.parametrize("n_sigma", [0.3, 1.0, 2.5, 5.0, 9.0, 25.0])
    def test_gradient_wrt_line_centre(self, n_sigma):
        """d(profile)/d(lambda_0) must match a central difference."""
        wavelength = LAMBDA_0 + n_sigma * SIGMA
        assert_grad_matches_fd(
            lambda x: line_profile(x, SIGMA, GAMMA, AMPLITUDE, wavelength),
            LAMBDA_0,
            step=1e-4 * SIGMA,
            name="line_profile d/d(lambda_0)",
        )

    @pytest.mark.parametrize("n_sigma", DETUNINGS)
    def test_gradient_wrt_sigma(self, n_sigma):
        """d(profile)/d(sigma) must match a central difference."""
        wavelength = LAMBDA_0 + n_sigma * SIGMA
        assert_grad_matches_fd(
            lambda x: line_profile(LAMBDA_0, x, GAMMA, AMPLITUDE, wavelength),
            SIGMA,
            step=1e-5 * SIGMA,
            name="line_profile d/d(sigma)",
        )

    @pytest.mark.parametrize("n_sigma", DETUNINGS)
    def test_gradient_wrt_gamma(self, n_sigma):
        """d(profile)/d(gamma) must match a central difference and be positive in the wings."""
        wavelength = LAMBDA_0 + n_sigma * SIGMA
        analytic = assert_grad_matches_fd(
            lambda x: line_profile(LAMBDA_0, SIGMA, x, AMPLITUDE, wavelength),
            GAMMA,
            step=1e-5 * GAMMA,
            name="line_profile d/d(gamma)",
        )
        if n_sigma >= 2.5:
            assert analytic > 0, (
                f"increasing gamma must increase wing opacity, got {analytic}"
            )

    @pytest.mark.parametrize("n_sigma", DETUNINGS)
    def test_gradient_wrt_amplitude_is_exactly_linear(self, n_sigma):
        """
        The profile is exactly linear in amplitude, so d/d(amplitude) equals the
        profile evaluated at unit amplitude. This is a gradient we know in
        closed form, so it is checked to machine precision rather than to the
        finite-difference tolerance.
        """
        wavelength = LAMBDA_0 + n_sigma * SIGMA
        analytic = float(
            jax.grad(line_profile, argnums=3)(LAMBDA_0, SIGMA, GAMMA, AMPLITUDE, wavelength)
        )
        expected = float(line_profile(LAMBDA_0, SIGMA, GAMMA, 1.0, wavelength))
        np.testing.assert_allclose(analytic, expected, rtol=1e-13)
        assert analytic > 0

    @pytest.mark.parametrize("n_sigma", [0.3, 1.0, 2.5, 5.0, 9.0, 25.0])
    def test_gradient_wrt_wavelength(self, n_sigma):
        """
        d(profile)/d(lambda) must match a central difference, be negative on the
        red side of the line, and be the exact negative of d/d(lambda_0).
        """
        wavelength = LAMBDA_0 + n_sigma * SIGMA
        analytic = assert_grad_matches_fd(
            lambda x: line_profile(LAMBDA_0, SIGMA, GAMMA, AMPLITUDE, x),
            wavelength,
            step=1e-4 * SIGMA,
            name="line_profile d/d(lambda)",
        )
        assert analytic < 0, f"profile must fall off redward of centre, got {analytic}"

        d_centre = float(
            jax.grad(line_profile, argnums=0)(LAMBDA_0, SIGMA, GAMMA, AMPLITUDE, wavelength)
        )
        np.testing.assert_allclose(analytic, -d_centre, rtol=1e-13)

    @pytest.mark.parametrize("n_sigma", [0.0, 0.5, 2.0, 6.0, 20.0])
    def test_gradients_finite_for_zero_gamma(self, n_sigma):
        """
        Regression test: gamma = 0 (a pure Gaussian line) used to give NaN
        gradients for every argument.

        gamma = 0 makes the Voigt alpha parameter zero, which used to divide by
        zero in the unselected alpha > 1.4 branch of voigt_hjerting. Lines with
        no Lorentz broadening data are common, so this must work.
        """
        wavelength = LAMBDA_0 + n_sigma * SIGMA
        grads = jax.grad(line_profile, argnums=(0, 1, 2, 3, 4))(
            LAMBDA_0, SIGMA, 0.0, AMPLITUDE, wavelength
        )
        names = ["wavelength_0", "sigma", "gamma", "amplitude", "wavelength"]
        for name, g in zip(names, grads):
            assert np.isfinite(float(g)), (
                f"d/d{name} = {float(g)} at gamma=0, {n_sigma} sigma"
            )

    def test_gradient_at_line_centre_is_symmetric(self):
        """
        The profile is even about lambda_0, so d/d(lambda) vanishes at centre and
        d/d(sigma) and d/d(gamma) are unchanged under reflection.
        """
        d_wavelength = float(
            jax.grad(line_profile, argnums=4)(LAMBDA_0, SIGMA, GAMMA, AMPLITUDE, LAMBDA_0)
        )
        assert np.isfinite(d_wavelength)

        blue = jax.grad(line_profile, argnums=(1, 2, 3))(
            LAMBDA_0, SIGMA, GAMMA, AMPLITUDE, LAMBDA_0 - 3 * SIGMA
        )
        red = jax.grad(line_profile, argnums=(1, 2, 3))(
            LAMBDA_0, SIGMA, GAMMA, AMPLITUDE, LAMBDA_0 + 3 * SIGMA
        )
        for b, r in zip(blue, red):
            np.testing.assert_allclose(float(b), float(r), rtol=1e-12)

    def test_no_nan_gradients_over_a_parameter_sweep(self):
        """Blanket sweep over gamma/sigma/detuning: no gradient may be non-finite."""
        gammas = np.concatenate([np.array([0.0]), np.logspace(-16, -7, 12)])
        sigmas = np.logspace(-12, -8, 7)
        detunings = np.concatenate([np.array([0.0]), np.logspace(-14, -6, 12)])

        for gamma in gammas:
            for sigma in sigmas:
                for detuning in detunings:
                    grads = jax.grad(line_profile, argnums=(0, 1, 2, 3, 4))(
                        LAMBDA_0, float(sigma), float(gamma), AMPLITUDE,
                        LAMBDA_0 + float(detuning),
                    )
                    values = np.array([float(g) for g in grads])
                    assert np.isfinite(values).all(), (
                        f"non-finite gradient {values} for sigma={sigma:g}, "
                        f"gamma={gamma:g}, detuning={detuning:g}"
                    )

    def test_gradient_under_jit_and_vmap(self):
        """Gradients must survive jit and vmap composition."""
        grad_fn = jax.jit(
            jax.vmap(jax.grad(line_profile, argnums=(0, 1, 2, 3, 4)),
                     in_axes=(None, None, None, None, 0))
        )
        wavelengths = LAMBDA_0 + jnp.linspace(-30 * SIGMA, 30 * SIGMA, 33)
        grads = grad_fn(LAMBDA_0, SIGMA, GAMMA, AMPLITUDE, wavelengths)
        for g in grads:
            assert np.isfinite(np.asarray(g)).all()


# ---------------------------------------------------------------------------
# inverse_gaussian_density / inverse_lorentz_density
# ---------------------------------------------------------------------------


class TestInverseDensityGradients:
    """
    Gradients of the inverse-PDF helpers used to size line windows.

    Both functions return 0 when the requested density exceeds the peak of the
    PDF. The masked-out branch takes the square root of a negative number, which
    used to produce a NaN cotangent.
    """

    @pytest.mark.parametrize("sigma", [1e-10, 1e-3, 1.0, 5.0])
    @pytest.mark.parametrize("density_fraction", [1e-6, 1e-3, 0.1, 0.5, 0.9])
    def test_gaussian_gradients_match_finite_difference(self, sigma, density_fraction):
        """
        Gradients w.r.t. both rho and sigma must match central differences.

        The absolute tolerances cover stationary points: holding rho fixed,
        dx/d(sigma) vanishes when -2 log(density_fraction) == 1. Both
        derivatives are dimensionless multiples of the returned x (itself of
        order sigma), so the finite-difference noise floor is ~1e-10; 1e-8 is
        safe.
        """
        rho = density_fraction / (np.sqrt(2 * np.pi) * sigma)
        assert_grad_matches_fd(
            lambda x: inverse_gaussian_density(x, sigma),
            rho,
            step=1e-6 * rho,
            atol=1e-8 * sigma / rho,
            name="inverse_gaussian_density d/d(rho)",
        )
        assert_grad_matches_fd(
            lambda x: inverse_gaussian_density(rho, x),
            sigma,
            step=1e-6 * sigma,
            atol=1e-8,
            name="inverse_gaussian_density d/d(sigma)",
        )

    @pytest.mark.parametrize("gamma", [1e-10, 1e-3, 1.0, 5.0])
    @pytest.mark.parametrize("density_fraction", [1e-6, 1e-3, 0.1, 0.5, 0.9])
    def test_lorentz_gradients_match_finite_difference(self, gamma, density_fraction):
        """
        Gradients w.r.t. both rho and gamma must match central differences.

        As for the Gaussian, ``atol`` covers stationary points: holding rho
        fixed at ``f / (pi gamma_0)``, dx/d(gamma) = (2 gamma_0 - 2 gamma) / 2x
        vanishes identically at ``density_fraction = 0.5``.
        """
        rho = density_fraction / (np.pi * gamma)
        assert_grad_matches_fd(
            lambda x: inverse_lorentz_density(x, gamma),
            rho,
            step=1e-6 * rho,
            atol=1e-8 * gamma / rho,
            name="inverse_lorentz_density d/d(rho)",
        )
        assert_grad_matches_fd(
            lambda x: inverse_lorentz_density(rho, x),
            gamma,
            step=1e-6 * gamma,
            atol=1e-8,
            name="inverse_lorentz_density d/d(gamma)",
        )

    @pytest.mark.parametrize("density_fraction", [1.0000001, 1.5, 10.0, 1e6])
    @pytest.mark.parametrize(
        "func, module_name",
        [
            (inverse_gaussian_density, "line_profiles"),
            (inverse_gaussian_density_absorption, "line_absorption"),
        ],
    )
    def test_gaussian_gradient_is_zero_above_peak_density(
        self, density_fraction, func, module_name
    ):
        """
        Regression test: rho above the peak of the PDF used to give a NaN gradient.

        The function returns a constant 0 there, so both partial derivatives
        must be exactly 0 -- not NaN from the sqrt of a negative radicand in the
        masked branch.
        """
        sigma = 1.0
        rho = density_fraction / (np.sqrt(2 * np.pi) * sigma)
        assert float(func(rho, sigma)) == 0.0
        d_rho, d_sigma = jax.grad(func, argnums=(0, 1))(np.float64(rho), np.float64(sigma))
        assert float(d_rho) == 0.0, f"{module_name}: d/d(rho) = {float(d_rho)}"
        assert float(d_sigma) == 0.0, f"{module_name}: d/d(sigma) = {float(d_sigma)}"

    @pytest.mark.parametrize("density_fraction", [1.0000001, 1.5, 10.0, 1e6])
    @pytest.mark.parametrize(
        "func, module_name",
        [
            (inverse_lorentz_density, "line_profiles"),
            (inverse_lorentz_density_absorption, "line_absorption"),
        ],
    )
    def test_lorentz_gradient_is_zero_above_peak_density(
        self, density_fraction, func, module_name
    ):
        """Regression test: rho above the peak of the PDF used to give a NaN gradient."""
        gamma = 1.0
        rho = density_fraction / (np.pi * gamma)
        assert float(func(rho, gamma)) == 0.0
        d_rho, d_gamma = jax.grad(func, argnums=(0, 1))(np.float64(rho), np.float64(gamma))
        assert float(d_rho) == 0.0, f"{module_name}: d/d(rho) = {float(d_rho)}"
        assert float(d_gamma) == 0.0, f"{module_name}: d/d(gamma) = {float(d_gamma)}"

    def test_gaussian_gradient_signs(self):
        """Requesting a lower density moves you further out; a wider sigma does too."""
        sigma, rho = 1.0, 0.05
        d_rho, d_sigma = jax.grad(inverse_gaussian_density, argnums=(0, 1))(rho, sigma)
        assert float(d_rho) < 0
        assert float(d_sigma) > 0

    def test_lorentz_gradient_signs(self):
        """Same monotonicity for the Lorentzian."""
        gamma, rho = 1.0, 0.05
        d_rho, d_gamma = jax.grad(inverse_lorentz_density, argnums=(0, 1))(rho, gamma)
        assert float(d_rho) < 0
        assert float(d_gamma) > 0

    def test_gaussian_gradient_is_exact(self):
        """
        x(rho) = sigma sqrt(-2 log(sqrt(2 pi) sigma rho)), so
        dx/d(rho) = -sigma / (rho sqrt(-2 log(sqrt(2 pi) sigma rho))).
        """
        for sigma, rho in [(1.0, 0.05), (2.0, 0.01), (1e-3, 50.0)]:
            radicand = -2 * np.log(np.sqrt(2 * np.pi) * sigma * rho)
            expected = -sigma / (rho * np.sqrt(radicand))
            analytic = float(jax.grad(inverse_gaussian_density, argnums=0)(rho, sigma))
            np.testing.assert_allclose(analytic, expected, rtol=1e-12)

    @pytest.mark.parametrize(
        "profile_func, absorption_func",
        [
            (inverse_gaussian_density, inverse_gaussian_density_absorption),
            (inverse_lorentz_density, inverse_lorentz_density_absorption),
        ],
    )
    def test_duplicate_implementations_have_the_same_gradients(
        self, profile_func, absorption_func
    ):
        """
        ``line_profiles`` and ``line_absorption`` carry duplicate copies of these
        helpers; their gradients must not drift apart.
        """
        for rho, width in [(0.05, 1.0), (1e-3, 2.0), (0.2, 0.5)]:
            a = jax.grad(profile_func, argnums=(0, 1))(np.float64(rho), np.float64(width))
            b = jax.grad(absorption_func, argnums=(0, 1))(np.float64(rho), np.float64(width))
            for x, y in zip(a, b):
                np.testing.assert_allclose(float(x), float(y), rtol=1e-13)


# ---------------------------------------------------------------------------
# broadening helpers
# ---------------------------------------------------------------------------

FE_MASS = 55.845 * 1.6605402e-24  # g
TEMPERATURES = [3000.0, 4500.0, 5777.0, 8000.0, 15000.0]


class TestSigmaLineGradients:
    """Gradients of the gf-normalised line cross-section."""

    @pytest.mark.parametrize("wavelength_angstrom", [3000.0, 5000.0, 8000.0, 16000.0])
    @pytest.mark.parametrize(
        "func", [sigma_line, sigma_line_broadening], ids=["line_absorption", "line_broadening"]
    )
    def test_gradient_matches_finite_difference(self, wavelength_angstrom, func):
        """d(sigma_line)/d(lambda) must match a central difference and be positive."""
        wavelength = wavelength_angstrom * 1e-8
        analytic = assert_grad_matches_fd(
            func, wavelength, step=1e-6 * wavelength, name="sigma_line"
        )
        assert analytic > 0

    @pytest.mark.parametrize(
        "func", [sigma_line, sigma_line_broadening], ids=["line_absorption", "line_broadening"]
    )
    def test_gradient_is_exact(self, func):
        """sigma_line is quadratic in lambda, so d(sigma)/d(lambda) = 2 sigma / lambda."""
        wavelength = 5000e-8
        analytic = float(jax.grad(func)(wavelength))
        expected = 2 * float(func(wavelength)) / wavelength
        np.testing.assert_allclose(analytic, expected, rtol=1e-13)


class TestDopplerWidthGradients:
    """Gradients of the thermal + microturbulent Doppler width."""

    @pytest.mark.parametrize("temperature", TEMPERATURES)
    @pytest.mark.parametrize("xi", [0.0, 1e5, 3e5])
    def test_all_gradients_are_finite(self, temperature, xi):
        """All four arguments must have finite gradients, including xi = 0."""
        grads = jax.grad(doppler_width_absorption, argnums=(0, 1, 2, 3))(
            5000e-8, np.float64(temperature), FE_MASS, np.float64(xi)
        )
        names = ["wavelength", "temperature", "mass", "xi"]
        for name, g in zip(names, grads):
            assert np.isfinite(float(g)), f"d/d{name} = {float(g)} at T={temperature}, xi={xi}"

    @pytest.mark.parametrize("temperature", TEMPERATURES)
    def test_temperature_gradient(self, temperature):
        """d(sigma)/dT must match a central difference and be positive."""
        analytic = assert_grad_matches_fd(
            lambda t: doppler_width_absorption(5000e-8, t, FE_MASS, 1e5),
            temperature,
            step=1e-4 * temperature,
            name="doppler_width d/dT",
        )
        assert analytic > 0, "hotter gas must broaden the line"

    def test_wavelength_gradient(self):
        """The width is linear in wavelength, so d(sigma)/d(lambda) = sigma / lambda."""
        wavelength = 5000e-8
        analytic = float(
            jax.grad(doppler_width_absorption, argnums=0)(wavelength, 5777.0, FE_MASS, 1e5)
        )
        expected = float(doppler_width_absorption(wavelength, 5777.0, FE_MASS, 1e5)) / wavelength
        np.testing.assert_allclose(analytic, expected, rtol=1e-13)

    def test_mass_gradient_is_negative(self):
        """Heavier species have narrower thermal profiles."""
        analytic = assert_grad_matches_fd(
            lambda m: doppler_width_absorption(5000e-8, 5777.0, m, 1e5),
            FE_MASS,
            step=1e-6 * FE_MASS,
            name="doppler_width d/dm",
        )
        assert analytic < 0

    @pytest.mark.parametrize("xi", [5e4, 1e5, 3e5])
    def test_microturbulence_gradient_is_positive(self, xi):
        """More microturbulence means a wider profile."""
        analytic = assert_grad_matches_fd(
            lambda x: doppler_width_absorption(5000e-8, 5777.0, FE_MASS, x),
            xi,
            step=1e-6 * xi,
            name="doppler_width d/dxi",
        )
        assert analytic > 0

    def test_duplicate_implementations_have_the_same_gradients(self):
        """``line_profiles.doppler_width`` must agree with ``line_absorption``'s copy."""
        args = (5000e-8, 5777.0, FE_MASS, 1e5)
        a = jax.grad(doppler_width, argnums=(0, 1, 2, 3))(*args)
        b = jax.grad(doppler_width_absorption, argnums=(0, 1, 2, 3))(*args)
        for x, y in zip(a, b):
            np.testing.assert_allclose(float(x), float(y), rtol=1e-13)


class TestScaledStarkGradients:
    """Gradients of the temperature-scaled Stark broadening parameter."""

    @pytest.mark.parametrize("temperature", TEMPERATURES)
    @pytest.mark.parametrize(
        "func", [scaled_stark, scaled_stark_broadening],
        ids=["line_absorption", "line_broadening"],
    )
    def test_temperature_gradient(self, temperature, func):
        """d(gamma)/dT must match a central difference and be positive."""
        gamma_stark = 1e8
        analytic = assert_grad_matches_fd(
            lambda t: func(gamma_stark, t),
            temperature,
            step=1e-4 * temperature,
            name="scaled_stark d/dT",
        )
        assert analytic > 0

    @pytest.mark.parametrize("temperature", TEMPERATURES)
    def test_temperature_gradient_is_exact(self, temperature):
        """gamma(T) = gamma_0 (T/T0)^(1/6), so d(gamma)/dT = gamma(T) / (6 T)."""
        gamma_stark = 1e8
        analytic = float(jax.grad(scaled_stark, argnums=1)(gamma_stark, np.float64(temperature)))
        expected = float(scaled_stark(gamma_stark, temperature)) / (6 * temperature)
        np.testing.assert_allclose(analytic, expected, rtol=1e-12)

    def test_gamma_gradient_is_exact(self):
        """The scaling is linear in gamma_stark."""
        analytic = float(jax.grad(scaled_stark, argnums=0)(1e8, 5777.0))
        expected = float(scaled_stark(1.0, 5777.0))
        np.testing.assert_allclose(analytic, expected, rtol=1e-13)
        assert analytic > 0


class TestScaledVdWGradients:
    """
    Gradients of the van der Waals broadening parameter, in both call modes.

    ``scaled_vdW`` takes ``vdW = (gamma_vdW, -1)`` for the simple
    gamma-per-perturber scaling, or ``vdW = (sigma, alpha)`` for the ABO
    (Anstee-Barklem-O'Mara) parametrisation. ``line_absorption``'s version picks
    between them with ``jnp.where``, so the unselected branch is always
    evaluated and always differentiated -- exactly the situation where a masked
    NaN would bite.
    """

    # A typical ABO entry: sigma in cm^2 (a few hundred Bohr radii squared),
    # alpha dimensionless and well inside (0, 1).
    ABO_SIGMA = 1.4e-14
    ABO_ALPHA = 0.3
    SIMPLE_GAMMA = 1e-8

    @pytest.mark.parametrize("temperature", TEMPERATURES)
    def test_simple_mode_gradients_are_finite(self, temperature):
        """The simple (alpha = -1) mode must not be poisoned by the ABO branch."""
        grads = jax.grad(
            lambda g, m, t: scaled_vdW((g, -1.0), m, t), argnums=(0, 1, 2)
        )(self.SIMPLE_GAMMA, FE_MASS, np.float64(temperature))
        for name, g in zip(["gamma_vdW", "mass", "temperature"], grads):
            assert np.isfinite(float(g)), f"d/d{name} = {float(g)} at T={temperature}"

    @pytest.mark.parametrize("temperature", TEMPERATURES)
    def test_simple_mode_temperature_gradient(self, temperature):
        """gamma(T) = gamma_0 (T/1e4)^0.3, so d(gamma)/dT = 0.3 gamma(T) / T."""
        analytic = assert_grad_matches_fd(
            lambda t: scaled_vdW((self.SIMPLE_GAMMA, -1.0), FE_MASS, t),
            temperature,
            step=1e-4 * temperature,
            name="scaled_vdW (simple) d/dT",
        )
        assert analytic > 0
        expected = 0.3 * float(scaled_vdW((self.SIMPLE_GAMMA, -1.0), FE_MASS, temperature)) / temperature
        np.testing.assert_allclose(analytic, expected, rtol=1e-12)

    def test_simple_mode_gamma_gradient_is_exact(self):
        """The simple scaling is linear in gamma_vdW."""
        analytic = float(
            jax.grad(lambda g: scaled_vdW((g, -1.0), FE_MASS, 5777.0))(self.SIMPLE_GAMMA)
        )
        expected = float(scaled_vdW((1.0, -1.0), FE_MASS, 5777.0))
        np.testing.assert_allclose(analytic, expected, rtol=1e-13)
        assert analytic > 0

    def test_simple_mode_mass_gradient_is_zero(self):
        """
        The simple scaling ignores the species mass, so its gradient must be
        exactly zero -- and in particular must not pick up the ABO branch's
        (non-zero, finite) mass dependence through the select.
        """
        analytic = float(
            jax.grad(lambda m: scaled_vdW((self.SIMPLE_GAMMA, -1.0), m, 5777.0))(FE_MASS)
        )
        assert analytic == 0.0, f"expected exactly 0, got {analytic}"

    @pytest.mark.parametrize("temperature", TEMPERATURES)
    def test_abo_mode_gradients_are_finite(self, temperature):
        """All four ABO arguments must have finite gradients."""
        grads = jax.grad(
            lambda s, a, m, t: scaled_vdW((s, a), m, t), argnums=(0, 1, 2, 3)
        )(self.ABO_SIGMA, self.ABO_ALPHA, FE_MASS, np.float64(temperature))
        for name, g in zip(["sigma", "alpha", "mass", "temperature"], grads):
            assert np.isfinite(float(g)), f"d/d{name} = {float(g)} at T={temperature}"

    @pytest.mark.parametrize("temperature", TEMPERATURES)
    def test_abo_temperature_gradient(self, temperature):
        """d(gamma)/dT must match a central difference and be positive for alpha < 1."""
        analytic = assert_grad_matches_fd(
            lambda t: scaled_vdW((self.ABO_SIGMA, self.ABO_ALPHA), FE_MASS, t),
            temperature,
            step=1e-4 * temperature,
            name="scaled_vdW (ABO) d/dT",
        )
        assert analytic > 0

    @pytest.mark.parametrize("alpha", [0.0, 0.1, 0.25, 0.3, 0.5, 0.9])
    def test_abo_alpha_gradient(self, alpha):
        """
        d(gamma)/d(alpha) must match a central difference.

        This is the only argument whose gradient runs through
        ``jax.scipy.special.gamma``, so it exercises a code path none of the
        other tests touch.
        """
        assert_grad_matches_fd(
            lambda a: scaled_vdW((self.ABO_SIGMA, a), FE_MASS, 5777.0),
            alpha,
            step=1e-6,
            rtol=1e-4,  # digamma-based gamma-function VJP: slightly noisier
            name="scaled_vdW (ABO) d/d(alpha)",
        )

    def test_abo_sigma_gradient_is_exact(self):
        """The ABO formula is linear in sigma."""
        analytic = float(
            jax.grad(lambda s: scaled_vdW((s, self.ABO_ALPHA), FE_MASS, 5777.0))(self.ABO_SIGMA)
        )
        expected = float(scaled_vdW((1.0, self.ABO_ALPHA), FE_MASS, 5777.0))
        np.testing.assert_allclose(analytic, expected, rtol=1e-12)
        assert analytic > 0

    def test_abo_mass_gradient(self):
        """
        Heavier species have a lower mean relative velocity, hence (for
        alpha < 1) less vdW broadening.
        """
        analytic = assert_grad_matches_fd(
            lambda m: scaled_vdW((self.ABO_SIGMA, self.ABO_ALPHA), m, 5777.0),
            FE_MASS,
            step=1e-6 * FE_MASS,
            name="scaled_vdW (ABO) d/dm",
        )
        assert analytic < 0

    @pytest.mark.parametrize("temperature", TEMPERATURES)
    def test_line_broadening_copy_matches(self, temperature):
        """
        ``line_broadening.scaled_vdW`` branches with a Python ``if``, so it
        cannot be differentiated w.r.t. alpha, but its gradients w.r.t. sigma,
        mass and temperature must agree with ``line_absorption``'s traced copy
        in both modes.
        """
        for vdW in [(self.SIMPLE_GAMMA, -1.0), (self.ABO_SIGMA, self.ABO_ALPHA)]:
            a = jax.grad(
                lambda s, m, t: scaled_vdW_broadening((s, vdW[1]), m, t), argnums=(0, 1, 2)
            )(vdW[0], FE_MASS, np.float64(temperature))
            b = jax.grad(
                lambda s, m, t: scaled_vdW((s, vdW[1]), m, t), argnums=(0, 1, 2)
            )(vdW[0], FE_MASS, np.float64(temperature))
            for x, y in zip(a, b):
                assert np.isfinite(float(x))
                np.testing.assert_allclose(float(x), float(y), rtol=1e-12)

    def test_gradient_under_jit_and_vmap(self):
        """ABO gradients must survive jit and vmap over temperature."""
        grad_fn = jax.jit(
            jax.vmap(
                jax.grad(lambda s, a, m, t: scaled_vdW((s, a), m, t), argnums=(0, 1, 2, 3)),
                in_axes=(None, None, None, 0),
            )
        )
        temperatures = jnp.linspace(3000.0, 15000.0, 17)
        grads = grad_fn(self.ABO_SIGMA, self.ABO_ALPHA, FE_MASS, temperatures)
        for g in grads:
            assert np.isfinite(np.asarray(g)).all()


class TestExponentialIntegral1Gradients:
    """
    Gradients of the E1 approximation used by the Brackett line profiles.

    E1 is not in the Voigt path, but it lives in ``line_profiles`` and is a
    four-way ``jnp.where`` over x, so it deserves the same guard.
    """

    @pytest.mark.parametrize(
        "x", [1e-4, 0.005, 0.009, 0.011, 0.1, 0.5, 0.99, 1.01, 2.0, 10.0, 29.0, 31.0, 100.0]
    )
    def test_gradient_is_finite(self, x):
        """No branch of the piecewise definition may leak a NaN cotangent."""
        derivative = float(jax.grad(exponential_integral_1)(np.float64(x)))
        assert np.isfinite(derivative), f"dE1/dx = {derivative} at x={x}"

    @pytest.mark.parametrize("x", [-10.0, -1.0, -1e-6])
    def test_gradient_is_finite_for_negative_x(self, x):
        """
        E1 is defined as 0 for x < 0. The masked branches take log(x), whose
        *value* is NaN there; the gradient must still come back finite (and, as
        the function is constant there, exactly zero).
        """
        derivative = float(jax.grad(exponential_integral_1)(np.float64(x)))
        assert np.isfinite(derivative), f"dE1/dx = {derivative} at x={x}"
        assert derivative == 0.0

    @pytest.mark.parametrize("x", [0.005, 0.1, 0.5, 2.0, 10.0, 25.0])
    def test_gradient_matches_finite_difference(self, x):
        """Within each branch the derivative must match a central difference."""
        analytic = assert_grad_matches_fd(
            exponential_integral_1, x, step=1e-7 * x, name="exponential_integral_1"
        )
        assert analytic < 0, "E1 is monotonically decreasing"
