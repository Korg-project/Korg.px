"""
Automatic-differentiation tests for the radiative transfer code.

The whole point of this port is that spectral synthesis is differentiable with
JAX, so every float-in/float-out function in ``korg.radiative_transfer`` must
return finite, correct gradients -- not just finite values.

The specific failure mode these tests guard against is::

    y = jnp.where(cond, safe_expression, dangerous_expression)

If ``dangerous_expression`` evaluates to NaN or inf, ``jnp.where`` masks the
*value* but not the *cotangent*: reverse-mode AD still walks the dead branch and
the gradient comes back NaN even though the value looks perfectly healthy. Two
corollaries, both learned the hard way elsewhere in this package:

1. Clamping a dangerous denominator or radicand to zero is not enough -- the
   substituted value has to be strictly *safe* (positive, away from poles),
   because ``0 * inf`` is still NaN.
2. An unmasked infinite derivative is just as damaging as a masked one.

Radiative transfer is dense with such traps: the exponential integrals are
piecewise with a logarithmic pole at ``x = 0``, the accumulated optical depth is
identically zero at the top boundary, and the linear-source integrands have
removable ``0/0`` singularities as ``Δτ → 0``.

Two masked-NaN bugs found by these tests and fixed in ``src/korg``:

* ``exponential_integral_2`` returned a NaN gradient at ``x == 0`` (the value
  was a healthy 1.0). The dead ``_expint_small`` branch evaluates ``log(0)*0``
  and the dead ``_expint_large`` branch divides by zero, so the zero cotangent
  of the unselected branches gave ``0 * NaN``. This poisoned
  ``exponential_integral_3``, ``expint_transfer_integral_core`` and
  ``compute_F_flux_only_expint`` at the top of the atmosphere, where ``τ = 0``
  exactly. ``exponential_integral_1`` had the same problem at ``x = 0`` via the
  dead ``1/x`` in its ``x <= 30`` branch.
* ``fritsch_butland_C`` masked ``numerator / denominator`` but not the
  denominator itself, so ``d(num/den)/d(den) = -num/den**2`` blew up whenever
  the Fritsch-Butland denominator vanished -- which it does *identically* for a
  constant source function or a flat opacity column, and at any local extremum
  with symmetric slopes. Both Bezier schemes (``tau_scheme="bezier"`` and
  ``intensity_scheme="bezier"``) therefore returned finite fluxes with NaN
  gradients for an isothermal atmosphere.

Each fix uses the double-``where`` pattern with a strictly safe substitute, and
leaves every value bitwise unchanged (the substitute is only ever fed to a
branch that is not selected).

Notes
-----
``calculate_rays`` does not exist anywhere in this port -- the spherical ray
geometry of Korg.jl has not been ported, and ``spherical=True`` is accepted but
ignored by ``compute_tau_anchored``. It therefore has no gradient tests here.
"""

# Import korg FIRST to enable JAX x64 mode before any other JAX operations
import korg  # noqa: F401 — side-effect: enables float64

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from korg.radiative_transfer.core import (
    generate_mu_grid,
    leggauss,
    radiative_transfer,
    radiative_transfer_jit,
    radiative_transfer_single_wavelength,
    radiative_transfer_single_wavelength_jit,
)
from korg.radiative_transfer.expint import (
    exponential_integral_1,
    exponential_integral_2,
    exponential_integral_3,
)
from korg.radiative_transfer.intensity import (
    compute_flux_from_intensities,
    compute_F_flux_only_expint,
    compute_I_bezier,
    compute_I_linear,
    compute_I_linear_flux_only,
    expint_transfer_integral_core,
    fritsch_butland_C,
)
from korg.radiative_transfer.optical_depth import (
    compute_tau_anchored,
    compute_tau_bezier,
    compute_tau_direct,
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
    Central finite-difference estimate of ``df/dx`` for a scalar argument.

    The effective step is recomputed as ``x_plus - x_minus`` so that the
    estimate stays accurate even when ``step`` is not exactly representable
    relative to ``x``.

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
        Absolute tolerance, needed only where the true derivative is zero,
        since there a purely relative comparison compares two pieces of
        floating-point noise.
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


def directional_fd(f, x, direction, step):
    """
    Central finite difference of a scalar-valued ``f`` along ``direction``.

    Parameters
    ----------
    f : callable
        Function mapping an array to a scalar.
    x : array_like
        Point at which to differentiate.
    direction : array_like
        Perturbation direction, same shape as ``x``. Passing ``direction = x``
        gives a purely multiplicative perturbation, which is the only sane
        choice for quantities such as opacities that span many decades.
    step : float
        Half-step along ``direction``.

    Returns
    -------
    float
        Estimate of the directional derivative ``grad(f) . direction``.
    """
    x = np.asarray(x, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    plus = float(f(jnp.asarray(x + step * direction)))
    minus = float(f(jnp.asarray(x - step * direction)))
    return (plus - minus) / (2.0 * step)


def assert_array_grad_matches_fd(f, x, direction=None, step=1e-6, rtol=FD_RTOL,
                                 atol=0.0, name=""):
    """
    Assert the AD gradient of an array-input scalar-output ``f`` is correct.

    The comparison is made along a single direction rather than component by
    component, which keeps the number of function evaluations small while still
    exercising every component of the gradient.

    Parameters
    ----------
    f : callable
        Function mapping an array to a scalar.
    x : array_like
        Point at which to differentiate.
    direction : array_like, optional
        Perturbation direction. Defaults to ``x`` itself (multiplicative).
    step : float, optional
        Half-step along ``direction``.
    rtol, atol : float, optional
        Tolerances for the comparison.
    name : str, optional
        Label included in assertion messages.

    Returns
    -------
    numpy.ndarray
        The analytic (AD) gradient.
    """
    x = np.asarray(x, dtype=np.float64)
    if direction is None:
        direction = x
    direction = np.asarray(direction, dtype=np.float64)

    analytic = np.asarray(jax.grad(f)(jnp.asarray(x)), dtype=np.float64)
    assert np.all(np.isfinite(analytic)), (
        f"{name}: gradient contains non-finite entries: {analytic}"
    )

    directional_analytic = float(np.sum(analytic * direction))
    numeric = directional_fd(f, x, direction, step)
    scale = max(abs(numeric), abs(directional_analytic))
    assert abs(directional_analytic - numeric) <= rtol * scale + atol, (
        f"{name}: directional AD derivative {directional_analytic!r} disagrees "
        f"with finite difference {numeric!r} (relative error "
        f"{abs(directional_analytic - numeric) / max(scale, 1e-300):.3e})"
    )
    return analytic


def model_atmosphere(n_layers=20, isothermal=False, flat_opacity=False):
    """
    A small, smooth, solar-ish atmosphere for gradient tests.

    Parameters
    ----------
    n_layers : int, optional
        Number of atmospheric layers.
    isothermal : bool, optional
        If True, the source function is constant with depth. This is the case
        that makes the Fritsch-Butland denominator vanish identically.
    flat_opacity : bool, optional
        If True, the absorption coefficient is constant with depth.

    Returns
    -------
    dict
        Keys ``alpha``, ``S``, ``spatial_coord``, ``log_tau_ref``,
        ``alpha_ref``, each a ``jnp`` array of shape ``(n_layers,)``.
    """
    log_tau_ref = jnp.linspace(-4.5, 1.5, n_layers)
    spatial_coord = jnp.linspace(2.0e9, 0.0, n_layers)
    alpha_ref = jnp.exp(jnp.linspace(-20.0, -14.0, n_layers))
    alpha = alpha_ref * jnp.exp(jnp.linspace(0.3, -0.4, n_layers)) * 1.7
    S = jnp.linspace(2.0e-5, 9.0e-5, n_layers)

    if isothermal:
        S = jnp.full((n_layers,), 5.0e-5)
    if flat_opacity:
        alpha = jnp.full((n_layers,), 1.0e-8)

    return dict(alpha=alpha, S=S, spatial_coord=spatial_coord,
                log_tau_ref=log_tau_ref, alpha_ref=alpha_ref)


def neighbourhood(x, n=6):
    """
    The ``2n + 1`` floats nearest to ``x``, ``x`` itself included.

    Used to probe whether a pole in an unselected branch leaks into the
    gradient only when the argument lands exactly on it.

    Parameters
    ----------
    x : float
        Centre of the neighbourhood.
    n : int, optional
        Number of ulps to walk in each direction.

    Returns
    -------
    list of float
        Sorted floats spanning ``[x - n ulp, x + n ulp]``.
    """
    out = [np.float64(x)]
    for direction in (np.inf, -np.inf):
        probe = np.float64(x)
        for _ in range(n):
            probe = np.nextafter(probe, direction)
            out.append(probe)
    return sorted(out)


# ---------------------------------------------------------------------------
# exponential_integral_1
# ---------------------------------------------------------------------------

# E1 is piecewise with knots at 0.01, 1.0 and 30.0, is defined to be zero
# outside [0, 30], and has a logarithmic pole at x = 0. The two negative values
# are the roots of the x <= 30 branch's denominator, x*(x + 3.330657) + 1.681534,
# i.e. poles of a branch that is never selected there.
E1_DEAD_BRANCH_POLES = [-0.6204416419266338, -2.7102153580733664]

E1_INTERIOR_POINTS = [
    1e-8, 1e-4, 0.001, 0.005, 0.0099, 0.011, 0.05, 0.2, 0.5, 0.9, 0.999,
    1.001, 1.5, 3.0, 7.0, 15.0, 29.5, 29.999,
]


class TestExponentialIntegral1Gradients:
    """Gradients of E1(x), which is piecewise in x with a log pole at 0."""

    @pytest.mark.parametrize("x", E1_INTERIOR_POINTS)
    def test_gradient_is_finite_and_negative(self, x):
        """E1 is positive and strictly decreasing on (0, 30), so E1' < 0."""
        g = float(jax.grad(exponential_integral_1)(np.float64(x)))
        assert np.isfinite(g), f"dE1/dx is {g} at x={x!r}"
        assert g < 0.0, f"dE1/dx should be negative at x={x!r}, got {g}"

    @pytest.mark.parametrize("x", E1_INTERIOR_POINTS)
    def test_gradient_matches_finite_difference(self, x):
        """
        AD must agree with a central difference away from the knots.

        The step is multiplicative because E1 varies as ``-log(x)`` near the
        origin: an absolute step would step straight across the pole.
        """
        assert_grad_matches_fd(exponential_integral_1, x, step=1e-6 * abs(x),
                               name="exponential_integral_1")

    @pytest.mark.parametrize("x", [-1e-3, -0.5, -1.0, -5.0, -100.0, 30.5, 50.0, 1e3])
    def test_gradient_vanishes_outside_the_support(self, x):
        """
        E1 is defined as identically zero for x < 0 and x > 30.

        A constant has zero derivative; anything else means a dead branch has
        leaked. In particular the x < 0 region evaluates ``log`` of a negative
        number in two unselected branches, and ``1/x`` in a third.
        """
        g = float(jax.grad(exponential_integral_1)(np.float64(x)))
        assert g == 0.0, f"dE1/dx should be exactly 0 outside [0, 30], got {g} at x={x!r}"

    def test_gradient_at_the_log_pole_is_not_nan(self):
        """
        At x = 0 exactly, E1 = -log(0) - gamma = +inf and E1' = -inf.

        Regression test. The selected branch honestly diverges, but before the
        fix the *unselected* ``x <= 30`` branch contributed ``0 * inf`` from its
        ``1/x``, turning a signed infinity into a NaN. An infinity at least
        tells the caller which way the function is going; a NaN does not.
        """
        value = float(exponential_integral_1(np.float64(0.0)))
        g = float(jax.grad(exponential_integral_1)(np.float64(0.0)))
        assert np.isinf(value) and value > 0.0, f"E1(0) should be +inf, got {value}"
        assert not np.isnan(g), "dE1/dx at the log pole is NaN (masked dead branch)"
        assert np.isneginf(g), f"dE1/dx at x=0 should be -inf, got {g}"

    @pytest.mark.parametrize("pole", E1_DEAD_BRANCH_POLES)
    def test_dead_branch_denominator_poles_do_not_leak(self, pole):
        """
        The x <= 30 branch has poles at x ~ -0.620 and x ~ -2.710.

        Those arguments select the ``x < 0`` branch (which returns 0), so the
        poles are never returned -- but they are still evaluated, and a zero
        cotangent times an infinite derivative is NaN. Sweep the floating-point
        neighbourhood of each pole, since landing exactly on it is what breaks.
        """
        for probe in neighbourhood(pole):
            g = float(jax.grad(exponential_integral_1)(probe))
            assert np.isfinite(g), f"dE1/dx is {g} at x={probe!r} (dead-branch pole)"
            assert g == 0.0, f"dE1/dx should be 0 for x < 0, got {g} at x={probe!r}"

    def test_gradient_finite_just_above_the_log_pole(self):
        """
        Just to the right of x = 0 the gradient must be finite and huge.

        E1'(x) = -exp(-x)/x, so the derivative is ~ -1/x. This is the regime a
        line-wing evaluation actually lands in, and it must not be NaN.

        Regression: it was. For x below ~1e-150 the dead ``x <= 30`` branch
        computes a ratio divided by x, whose derivative carries a second factor
        of 1/x and overflows to inf, so the zero cotangent gave 0 * inf = NaN
        over the whole tail of the domain -- not just at the pole itself.
        """
        for x in [1e-300, 1e-100, 1e-30, 1e-12, 1e-6]:
            g = float(jax.grad(exponential_integral_1)(np.float64(x)))
            assert np.isfinite(g), f"dE1/dx is {g} at x={x!r}"
            np.testing.assert_allclose(g, -1.0 / x, rtol=1e-5)

    @pytest.mark.parametrize("knot,max_relative_jump", [(0.01, 1e-4), (1.0, 1e-3)])
    def test_derivative_is_nearly_continuous_across_internal_knots(self, knot,
                                                                   max_relative_jump):
        """
        The internal knots are seams between independent fits, not true joins.

        E1 is stitched together from a series expansion (x <= 0.01), a
        polynomial fit (x <= 1) and a rational fit (x <= 30), each accurate to
        its own tolerance, so the derivative genuinely jumps at 0.01 (by 3.0e-5
        relative) and at 1.0 (by 4.5e-4 relative). Those jumps are inherited
        from Korg.jl and cannot be removed without changing values; this test
        pins them so they cannot silently grow.
        """
        below, above = one_sided_derivatives(exponential_integral_1, knot, 1e-7)
        assert np.isfinite(below) and np.isfinite(above)
        relative_jump = abs(above - below) / max(abs(below), abs(above))
        assert relative_jump < max_relative_jump, (
            f"dE1/dx jumps by {relative_jump:.3e} across the knot at x={knot}"
        )

    def test_derivative_is_discontinuous_at_the_cutoff(self):
        """
        E1 is truncated to zero at x = 30, where it is still ~3e-15.

        This is a genuine step discontinuity in both value and derivative, not a
        masked-NaN bug: the derivative simply drops from -3.1e-15 to exactly 0.
        Documented here so that the jump is a recorded property rather than a
        surprise for anyone differentiating a Brackett-line wing.
        """
        below, above = one_sided_derivatives(exponential_integral_1, 30.0, 1e-7)
        assert above == 0.0
        assert -1e-14 < below < 0.0


# ---------------------------------------------------------------------------
# exponential_integral_2 / exponential_integral_3
# ---------------------------------------------------------------------------

# E2 is piecewise with knots at 1.1, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5 and 9.0, plus
# a special case at exactly x == 0.
E2_KNOTS = [1.1, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 9.0]

E2_INTERIOR_POINTS = [
    1e-12, 1e-6, 0.01, 0.3, 0.7, 1.05, 1.2, 2.0, 2.4, 2.6, 3.0, 4.0, 5.0,
    6.0, 7.0, 8.0, 8.9, 9.5, 12.0, 30.0, 100.0,
]

# E2(x) -> 1 as x -> 0, so for tiny x the whole variation of the function lives
# in the last few bits and no finite difference can resolve the derivative. Those
# points are covered instead by comparing against the analytic limit log(x) +
# gamma in TestExponentialIntegral2Gradients.test_gradient_just_above_zero.
E2_FD_POINTS = [x for x in E2_INTERIOR_POINTS if x >= 0.01]


class TestExponentialIntegral2Gradients:
    """Gradients of E2(x), which is piecewise in x with a special case at 0."""

    @pytest.mark.parametrize("x", E2_INTERIOR_POINTS)
    def test_gradient_is_finite_and_negative(self, x):
        """E2 is positive and strictly decreasing, so E2'(x) = -E1(x) < 0."""
        g = float(jax.grad(exponential_integral_2)(np.float64(x)))
        assert np.isfinite(g), f"dE2/dx is {g} at x={x!r}"
        assert g < 0.0, f"dE2/dx should be negative at x={x!r}, got {g}"

    @pytest.mark.parametrize("x", E2_FD_POINTS)
    def test_gradient_matches_finite_difference(self, x):
        """
        AD must agree with a central difference away from the knots.

        The step is multiplicative: E2's derivative is ``-E1(x) ~ log(x)`` near
        the origin, so an absolute step would be useless for the tiny arguments
        in the list.
        """
        assert_grad_matches_fd(exponential_integral_2, x, step=1e-6 * abs(x),
                               name="exponential_integral_2")

    @pytest.mark.parametrize("x", E2_INTERIOR_POINTS)
    def test_forward_and_reverse_mode_agree(self, x):
        """jacfwd and jacrev exercise different code paths and must agree."""
        fwd = float(jax.jacfwd(exponential_integral_2)(np.float64(x)))
        rev = float(jax.jacrev(exponential_integral_2)(np.float64(x)))
        assert np.isfinite(rev), f"jacrev dE2/dx is {rev} at x={x!r}"
        np.testing.assert_allclose(fwd, rev, rtol=1e-12, atol=1e-300)

    def test_gradient_at_zero_is_finite(self):
        """
        Regression: ``jax.grad(exponential_integral_2)(0.0)`` used to be NaN.

        ``x == 0`` selects the literal 1.0, but the unselected small-x branch
        evaluates ``log(0) * 0 = NaN`` and the unselected large-x branch divides
        by zero, so ``jnp.where`` masked the value while the zero cotangent of
        the dead branches gave ``0 * NaN = NaN``.

        The double-``where`` fix feeds the dead branches a strictly positive
        substitute, giving derivative 0 -- the same answer Julia's ForwardDiff
        gets by taking the ``x == 0`` branch of the equivalent if/elseif chain,
        and the answer that makes ``tau * E2(tau)`` differentiable at the top of
        the atmosphere.
        """
        assert float(exponential_integral_2(np.float64(0.0))) == 1.0
        g = float(jax.grad(exponential_integral_2)(np.float64(0.0)))
        assert not np.isnan(g), "dE2/dx at x=0 is NaN (masked dead branch)"
        assert g == 0.0, f"dE2/dx at x=0 should be 0.0, got {g}"

    @pytest.mark.parametrize("x", [1e-300, 1e-200, 1e-100, 5e-77, 1e-40, 1e-30,
                                   1e-14, 1e-8])
    def test_gradient_just_above_zero_is_finite(self, x):
        """
        Regression: tiny positive arguments used to give a NaN gradient.

        E2'(x) = -E1(x) ~ log(x) as x -> 0, so it diverges only
        logarithmically: at x = 1e-300 the answer is a perfectly representable
        -690. But the unselected ``x >= 9`` branch forms ``120 / x**4``, which
        overflows to inf below x ~ 5e-77 and then contributed ``0 * inf = NaN``.
        This one is subtler than the ``x == 0`` case because it needs no exact
        equality to trigger -- 77 decades of the domain were affected.

        The expected value is ``log(x) + gamma``, the derivative of the leading
        ``x (log(x) + gamma - 1)`` term.
        """
        g = float(jax.grad(exponential_integral_2)(np.float64(x)))
        assert np.isfinite(g), f"dE2/dx is {g} at x={x!r}"
        assert g < 0.0
        expected = np.log(x) + 0.5772156649015329
        np.testing.assert_allclose(g, expected, rtol=1e-6)

    def test_denormal_arguments_are_out_of_reach(self):
        """
        Below the smallest normal float the gradient silently flattens to zero.

        ``_expint_small`` differentiates ``x * log(x)`` as ``log(x) + x * (1/x)``
        and ``1/x`` overflows once x is denormal, so the ``x * (1/x)`` term is
        lost. Recorded rather than fixed: no optical depth in a stellar
        atmosphere is 1e-320, and the alternative is rewriting the fit.
        """
        assert float(jax.grad(exponential_integral_2)(np.float64(1e-320))) == 0.0

    def test_nan_input_still_propagates(self):
        """
        A NaN argument must stay NaN rather than being masked by a guard.

        The dead-branch substitutions use ``x < 9.0`` style comparisons, which
        are false for NaN, so a NaN argument still reaches the branch that
        propagates it. Silently turning a NaN input into a plausible number
        would hide upstream bugs.
        """
        assert np.isnan(float(exponential_integral_2(np.float64(np.nan))))
        assert np.isnan(float(exponential_integral_2(np.float64(-1.0)))), (
            "negative arguments are outside E2's domain and must stay NaN"
        )

    @pytest.mark.parametrize("knot", E2_KNOTS)
    def test_gradient_is_finite_on_both_sides_of_each_knot(self, knot):
        """Every piecewise seam must be finitely differentiable from both sides."""
        for probe in neighbourhood(knot, n=3):
            g = float(jax.grad(exponential_integral_2)(probe))
            assert np.isfinite(g), f"dE2/dx is {g} at x={probe!r} (knot {knot})"

    @pytest.mark.parametrize("knot,max_relative_jump", [
        (1.1, 3.0e-2), (2.5, 1.5e-2), (3.5, 8.0e-3), (4.5, 5.0e-3),
        (5.5, 4.0e-3), (6.5, 3.0e-3), (7.5, 3.0e-3), (9.0, 4.0e-2),
    ])
    def test_derivative_jump_across_each_knot_is_bounded(self, knot,
                                                         max_relative_jump):
        """
        E2's derivative is genuinely discontinuous at every internal knot.

        Each interval carries an independent minimax polynomial fit accurate to
        about 1%, so the fits disagree at the seams: the derivative jumps by
        1.9e-2 at x = 1.1, 7.1e-3 at 2.5, 3.2e-3 at 3.5, 2.1e-3 at 4.5,
        1.6e-3 at 5.5, 1.4e-3 at 6.5, 1.2e-3 at 7.5 and 2.7e-2 at 9.0
        (relative). The value itself jumps by up to 9.5e-3 at x = 1.1.

        This is a property of the approximation ported from Korg.jl, not a
        masked-NaN bug, and removing it would change values. It is pinned here
        because a *growing* jump would be a real regression, and because anyone
        optimising through ``compute_F_flux_only_expint`` should know the
        objective has ~1% kinks in its derivative at these optical depths.
        """
        below, above = one_sided_derivatives(exponential_integral_2, knot, 1e-7)
        assert np.isfinite(below) and np.isfinite(above)
        relative_jump = abs(above - below) / max(abs(below), abs(above))
        assert relative_jump < max_relative_jump, (
            f"dE2/dx jumps by {relative_jump:.3e} across the knot at x={knot}"
        )

    def test_gradient_equals_minus_e1(self):
        """
        E2'(x) = -E1(x) analytically; the two fits should roughly agree.

        Both are independent ~1% approximations, so this is a loose consistency
        check rather than an identity -- but a sign error or a factor of two
        would show up immediately.
        """
        for x in [0.5, 1.5, 3.0, 5.0, 8.0, 15.0]:
            g = float(jax.grad(exponential_integral_2)(np.float64(x)))
            e1 = float(exponential_integral_1(np.float64(x)))
            np.testing.assert_allclose(g, -e1, rtol=0.05,
                                       err_msg=f"E2'(x) != -E1(x) at x={x}")


class TestExponentialIntegral3Gradients:
    """Gradients of E3(x) = (exp(-x) - x E2(x)) / 2, which is built on E2."""

    @pytest.mark.parametrize("x", [0.01, 0.5, 1.0, 2.0, 5.0, 8.5, 20.0])
    def test_gradient_matches_finite_difference(self, x):
        """
        E3'(x) = -E2(x); check against a central difference.

        The identity is only satisfied to the accuracy of the E2 fit, since AD
        differentiates the piecewise polynomial rather than the true E2, so the
        second comparison is deliberately loose (the two disagree by 4e-3 at
        x = 1). The knot at x = 9 is excluded because a central difference
        across a seam of a discontinuous fit is meaningless.
        """
        g = assert_grad_matches_fd(exponential_integral_3, x, step=1e-7,
                                   name="exponential_integral_3")
        np.testing.assert_allclose(
            g, -float(exponential_integral_2(np.float64(x))), rtol=2e-2,
            err_msg=f"E3'(x) != -E2(x) at x={x}",
        )

    def test_gradient_at_zero_is_exact(self):
        """
        Regression: E3'(0) = -E2(0) = -1 exactly, and used to come back NaN.

        E3 calls E2, so it inherited the masked-NaN at x = 0. Unlike E1 and E2,
        E3 has a perfectly finite derivative there, which makes this the
        clearest demonstration that the bug destroyed real information rather
        than merely relabelling an infinity.
        """
        assert float(exponential_integral_3(np.float64(0.0))) == 0.5
        g = float(jax.grad(exponential_integral_3)(np.float64(0.0)))
        assert not np.isnan(g), "dE3/dx at x=0 is NaN (masked dead branch in E2)"
        assert g == -1.0, f"dE3/dx at x=0 should be exactly -1.0, got {g}"


# ---------------------------------------------------------------------------
# expint_transfer_integral_core
# ---------------------------------------------------------------------------

MB_CASES = [(1.0, 2.0), (-1.0, 2.0), (1.0, -2.0), (-1.0, -2.0), (0.0, 1.0), (1.0, 0.0)]


class TestExpintTransferIntegralCoreGradients:
    """
    Gradients of the antiderivative used by the exponential-integral flux.

    ``core(tau, m, b) = (tau E2(tau) (3b + 2 m tau) - exp(-tau)(3b + 2m(tau+1))) / 6``
    """

    @pytest.mark.parametrize("m,b", MB_CASES)
    @pytest.mark.parametrize("tau", [1e-3, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 12.0])
    def test_gradient_wrt_tau_matches_finite_difference(self, tau, m, b):
        """
        d(core)/d(tau) must be finite and match a central difference.

        ``atol`` covers the sign changes of the derivative -- for m = -1, b = 2
        it passes through zero near tau = 2, where a purely relative comparison
        would be comparing two pieces of floating-point noise. The
        antiderivative is O(1) and the step is 1e-7, so the finite-difference
        noise floor is ~1e-9.
        """
        assert_grad_matches_fd(
            lambda t: expint_transfer_integral_core(t, m, b), tau, step=1e-7,
            atol=1e-9, name=f"expint_transfer_integral_core (m={m}, b={b})",
        )

    @pytest.mark.parametrize("m,b", MB_CASES)
    @pytest.mark.parametrize("tau", [1e-14, 1e-10, 1e-6, 1e-4])
    def test_gradient_wrt_tau_is_finite_for_tiny_tau(self, tau, m, b):
        """
        Optically thin layers must not lose their gradient.

        A central difference is useless this close to the origin (the step
        would have to be smaller than the function's own rounding), so this
        checks finiteness plus convergence to the analytic tau -> 0 limit,
        which is ``b``.
        """
        g = float(jax.grad(expint_transfer_integral_core)(np.float64(tau), m, b))
        assert np.isfinite(g), f"d(core)/d(tau) is {g} at tau={tau}, m={m}, b={b}"
        # The leading correction to the limit is O(tau log tau).
        tolerance = 10.0 * (abs(m) + abs(b)) * tau * abs(np.log(tau)) + 1e-14
        assert abs(g - b) < tolerance, (
            f"d(core)/d(tau) = {g} does not approach its tau -> 0 limit {b} "
            f"at tau={tau} (tolerance {tolerance:.3e})"
        )

    @pytest.mark.parametrize("m,b", MB_CASES)
    def test_gradient_wrt_tau_at_zero_is_finite(self, m, b):
        """
        Regression: the tau -> 0 limit is finite but used to be NaN.

        ``tau E2(tau)`` has derivative ``E2(0) + 0 * E2'(0)``. E2'(0) diverges
        only logarithmically, so ``tau E2'(tau) -> 0`` and the true derivative
        at tau = 0 is simply ``b``. Before the E2 fix this evaluated as
        ``0 * NaN``. tau = 0 is not a hypothetical: it is the first entry of
        every optical depth array this package produces.
        """
        g = float(jax.grad(expint_transfer_integral_core)(np.float64(0.0), m, b))
        assert not np.isnan(g), f"d(core)/d(tau) at tau=0 is NaN for m={m}, b={b}"
        np.testing.assert_allclose(g, b, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize("m,b", MB_CASES)
    def test_gradient_wrt_tau_is_continuous_into_zero(self, m, b):
        """The tau = 0 gradient must be the limit of the gradient as tau -> 0."""
        at_zero = float(jax.grad(expint_transfer_integral_core)(np.float64(0.0), m, b))
        nearly = float(jax.grad(expint_transfer_integral_core)(np.float64(1e-9), m, b))
        np.testing.assert_allclose(at_zero, nearly, rtol=1e-6, atol=1e-7)

    @pytest.mark.parametrize("tau", [0.0, 1e-8, 0.5, 3.0, 10.0])
    def test_gradients_wrt_m_and_b_match_finite_differences(self, tau):
        """
        The antiderivative is linear in m and b, so these gradients are exact.

        They are also the ones that actually matter: in
        ``compute_F_flux_only_expint`` the source function enters only through
        m and b.
        """
        m0, b0 = 1.3, 2.7
        assert_grad_matches_fd(
            lambda m: expint_transfer_integral_core(np.float64(tau), m, b0),
            m0, step=1e-6, atol=1e-12, name=f"d(core)/dm at tau={tau}",
        )
        assert_grad_matches_fd(
            lambda b: expint_transfer_integral_core(np.float64(tau), m0, b),
            b0, step=1e-6, name=f"d(core)/db at tau={tau}",
        )

    def test_gradients_wrt_m_and_b_at_zero_are_exact(self):
        """
        At tau = 0 the antiderivative reduces to ``-(b/2 + m/3)``.

        So d/dm = -1/3 and d/db = -1/2 exactly, independent of everything else.
        """
        gm, gb = jax.grad(expint_transfer_integral_core, argnums=(1, 2))(
            np.float64(0.0), 1.0, 2.0)
        np.testing.assert_allclose(float(gm), -1.0 / 3.0, rtol=1e-14)
        np.testing.assert_allclose(float(gb), -0.5, rtol=1e-14)


# ---------------------------------------------------------------------------
# fritsch_butland_C
# ---------------------------------------------------------------------------


class TestFritschButlandGradients:
    """
    Gradients of the monotonicity-preserving Bezier control points.

    This helper is the shared foundation of ``compute_tau_bezier`` and
    ``compute_I_bezier``, so a NaN here poisons both Bezier schemes.
    """

    @staticmethod
    def _sum_C(x, y):
        """
        Scalarise ``fritsch_butland_C`` so ``jax.grad`` applies directly.

        The control points are weighted by their index rather than summed
        plainly, so that the scalar retains sensitivity to each control point
        individually instead of letting neighbouring ones cancel.
        """
        C = fritsch_butland_C(x, y)
        return jnp.sum(C * jnp.arange(1, C.shape[0] + 1))

    def test_jacobian_is_finite_for_a_generic_profile(self):
        """Baseline: a strictly monotone profile must differentiate cleanly."""
        x = jnp.linspace(0.0, 5.0, 8)
        y = jnp.linspace(1.0, 9.0, 8) ** 1.3
        for argnums in (0, 1):
            jac = np.asarray(jax.jacrev(fritsch_butland_C, argnums=argnums)(x, y))
            assert np.all(np.isfinite(jac)), (
                f"dC/d(arg {argnums}) is non-finite for a monotone profile"
            )

    def test_jacobian_is_finite_for_a_constant_profile(self):
        """
        Regression: a constant ``y`` used to give a NaN Jacobian.

        Every finite difference ``d[k]`` is zero, so the Fritsch-Butland
        denominator ``alpha d[k+1] + (1 - alpha) d[k]`` is identically zero. The
        old code masked ``numerator / denominator`` with ``jnp.where`` but left
        the denominator itself unguarded, so reverse-mode AD differentiated
        ``0/0`` and multiplied the resulting infinity by a zero cotangent.

        A constant source function is an isothermal atmosphere and a constant
        opacity is a flat continuum window -- neither is an exotic input.
        """
        x = jnp.linspace(0.0, 5.0, 8)
        y = jnp.full((8,), 3.0)
        for argnums in (0, 1):
            jac = np.asarray(jax.jacrev(fritsch_butland_C, argnums=argnums)(x, y))
            assert np.all(np.isfinite(jac)), (
                f"dC/d(arg {argnums}) is NaN for a constant profile "
                f"(masked 0/0 in the Fritsch-Butland denominator)"
            )

    def test_jacobian_is_finite_for_a_zigzag_profile(self):
        """
        A local extremum with symmetric slopes also zeroes the denominator.

        With ``d = (+1, -1, +1, ...)`` and equal spacing, alpha = 1/2 and the
        denominator is exactly ``(d[k+1] - d[k]) / 2 = 0``.
        """
        x = jnp.linspace(0.0, 5.0, 8)
        y = jnp.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
        for argnums in (0, 1):
            jac = np.asarray(jax.jacrev(fritsch_butland_C, argnums=argnums)(x, y))
            assert np.all(np.isfinite(jac)), (
                f"dC/d(arg {argnums}) is NaN at a symmetric local extremum"
            )

    def test_jacobian_is_finite_for_a_partially_flat_profile(self):
        """A profile that is flat over only part of its range must also work."""
        x = jnp.linspace(0.0, 5.0, 8)
        y = jnp.array([1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 6.0])
        for argnums in (0, 1):
            jac = np.asarray(jax.jacrev(fritsch_butland_C, argnums=argnums)(x, y))
            assert np.all(np.isfinite(jac)), (
                f"dC/d(arg {argnums}) is NaN for a partially flat profile"
            )

    def test_jacobian_matches_finite_difference(self):
        """
        The masked branch must not perturb the gradient where it is live.

        The abscissa is log-spaced, as a real optical depth grid is. The
        control points are invariant under affine reparametrisation of x (see
        the test below), so on a uniformly spaced grid a linearly ramped
        perturbation direction is annihilated exactly and the comparison would
        be a contest between two roundoff errors.
        """
        x = np.logspace(-2.0, 1.0, 8)
        y = np.linspace(1.0, 9.0, 8) ** 1.3
        assert_array_grad_matches_fd(
            lambda yy: self._sum_C(jnp.asarray(x), yy), y, step=1e-6,
            name="fritsch_butland_C d/dy",
        )
        assert_array_grad_matches_fd(
            lambda xx: self._sum_C(xx, jnp.asarray(y)), x,
            direction=np.linspace(0.1, 1.0, 8),
            step=1e-6, name="fritsch_butland_C d/dx",
        )

    @pytest.mark.parametrize("grid", ["uniform", "log"])
    def test_gradient_annihilates_affine_reparametrisations(self, grid):
        """
        The control points are invariant under ``x -> a + b x``.

        Translating the abscissa leaves the spacings alone, and dilating it
        scales ``h`` up and the derivative estimate ``yprime`` down by the same
        factor, so ``h * yprime`` -- the only place x enters -- is unchanged.
        The directional derivatives along the constant direction and along x
        itself must therefore both be exactly zero.

        This is a sharp, sign-sensitive check on dC/dx: an error in any single
        component would break the cancellation. It is also the reason the
        finite-difference test above uses a log-spaced grid.
        """
        x = (np.linspace(0.0, 5.0, 8) if grid == "uniform"
             else np.logspace(-2.0, 1.0, 8))
        y = np.linspace(1.0, 9.0, 8) ** 1.3
        grad = np.asarray(jax.grad(
            lambda xx: self._sum_C(xx, jnp.asarray(y))
        )(jnp.asarray(x)))
        scale = np.abs(grad).max() * max(1.0, np.abs(x).max())
        for name, direction in [("translation", np.ones_like(x)),
                                ("dilation", x)]:
            assert abs(float(np.sum(grad * direction))) < 1e-12 * scale, (
                f"dC/dx does not annihilate a {name} of the {grid} abscissa"
            )

    def test_constant_profile_gradient_is_the_expected_subgradient(self):
        """
        Where the denominator is masked, ``yprime`` is the constant 0.

        The control points then reduce to plain averages of neighbouring y
        values, so dC/dy must be an exact averaging matrix -- entries drawn from
        {0, 1/2, 1} with unit row sums -- rather than merely finite. This pins
        down *which* subgradient the mask picks, and confirms the mask is doing
        the interpolation rather than leaking a stray derivative of the dead
        0/0 branch.
        """
        x = jnp.linspace(0.0, 5.0, 6)
        y = jnp.full((6,), 3.0)
        jac = np.asarray(jax.jacrev(fritsch_butland_C, argnums=1)(x, y))
        assert set(np.unique(jac)).issubset({0.0, 0.5, 1.0}), (
            f"unexpected dC/dy for a constant profile: {jac}"
        )
        np.testing.assert_allclose(jac.sum(axis=1), np.ones(jac.shape[0]),
                                   rtol=1e-14)
        jac_x = np.asarray(jax.jacrev(fritsch_butland_C, argnums=0)(x, y))
        np.testing.assert_array_equal(jac_x, np.zeros_like(jac_x))


# ---------------------------------------------------------------------------
# compute_tau_anchored
# ---------------------------------------------------------------------------


class TestComputeTauAnchoredGradients:
    """
    Gradients of the anchored optical depth, the default tau scheme.

    The interesting boundary is the top of the atmosphere, where the
    accumulated tau is exactly zero, and the ``|alpha_ref| > 1e-30`` floor.
    """

    @staticmethod
    def _tau(atm, **overrides):
        """Evaluate ``compute_tau_anchored`` on a model atmosphere."""
        kwargs = dict(alpha=atm["alpha"], spatial_coord=atm["spatial_coord"],
                      log_tau_ref=atm["log_tau_ref"], alpha_ref=atm["alpha_ref"])
        kwargs.update(overrides)
        return compute_tau_anchored(kwargs["alpha"], kwargs["spatial_coord"],
                                    kwargs["log_tau_ref"], kwargs["alpha_ref"])

    def test_jacobian_wrt_alpha_is_finite_and_structured(self):
        """
        tau is a cumulative sum, so d(tau[i])/d(alpha[j]) is lower triangular.

        The first row must be identically zero because tau[0] = 0 by definition
        -- that is the top-boundary case, and it must be a clean zero rather
        than a NaN from an unguarded division.
        """
        atm = model_atmosphere()
        jac = np.asarray(jax.jacrev(lambda a: self._tau(atm, alpha=a))(atm["alpha"]))
        assert np.all(np.isfinite(jac)), "d(tau)/d(alpha) is not finite"
        np.testing.assert_array_equal(jac[0], np.zeros_like(jac[0]))
        assert np.all(jac[1:] >= 0.0), "increasing opacity must increase tau"
        assert np.any(jac[1:] > 0.0), "d(tau)/d(alpha) is identically zero"
        # strictly lower triangular in the sense that layer i is unaffected by
        # opacity deeper than i
        for i in range(jac.shape[0]):
            np.testing.assert_array_equal(jac[i, i + 1:], np.zeros(jac.shape[1] - i - 1))

    def test_gradient_wrt_alpha_matches_finite_difference(self):
        """AD against a multiplicative central difference on alpha."""
        atm = model_atmosphere()
        assert_array_grad_matches_fd(
            lambda a: jnp.sum(self._tau(atm, alpha=a)), atm["alpha"], step=1e-6,
            name="compute_tau_anchored d/d(alpha)",
        )

    def test_gradient_wrt_alpha_ref_matches_finite_difference(self):
        """
        alpha_ref enters as a reciprocal, so its gradient is the sensitive one.

        It is also the quantity most likely to be small, which is exactly where
        an unguarded ``1/alpha_ref`` would blow up.
        """
        atm = model_atmosphere()
        grad = assert_array_grad_matches_fd(
            lambda ar: jnp.sum(self._tau(atm, alpha_ref=ar)), atm["alpha_ref"],
            step=1e-6, name="compute_tau_anchored d/d(alpha_ref)",
        )
        assert np.all(grad[:-1] <= 0.0), "more reference opacity must lower tau"

    def test_gradient_wrt_log_tau_ref_matches_finite_difference(self):
        """
        log_tau_ref is both the integration variable and part of the integrand.

        Its gradient therefore mixes the ``10**log_tau_ref`` factor with the
        trapezoid widths, and is easy to get wrong.
        """
        atm = model_atmosphere()
        assert_array_grad_matches_fd(
            lambda lt: jnp.sum(self._tau(atm, log_tau_ref=lt)), atm["log_tau_ref"],
            step=1e-7, name="compute_tau_anchored d/d(log_tau_ref)",
        )

    def test_gradient_wrt_spatial_coord_is_identically_zero(self):
        """
        ``spatial_coord`` is accepted but never used by the anchored scheme.

        The anchored integration runs over log(tau_ref), so the spatial
        coordinate genuinely drops out. The gradient must be an exact zero
        (documenting the unused argument), not a NaN.
        """
        atm = model_atmosphere()
        grad = np.asarray(jax.grad(
            lambda sc: jnp.sum(self._tau(atm, spatial_coord=sc))
        )(atm["spatial_coord"]))
        np.testing.assert_array_equal(grad, np.zeros_like(grad))

    @pytest.mark.parametrize("floor_value", [1e-25, 1e-30, 1e-31, 1e-40, 0.0])
    def test_tiny_alpha_ref_gives_finite_gradients(self, floor_value):
        """
        ``alpha_ref`` is floored at 1e-30 to protect the division.

        The floor is applied to the denominator only (the safe pattern), so both
        sides of it must give finite gradients. Below the floor the derivative
        with respect to ``alpha_ref`` is zero, because a clamped constant no
        longer depends on its input.
        """
        atm = model_atmosphere()
        alpha_ref = atm["alpha_ref"].at[3].set(floor_value)
        grad = np.asarray(jax.grad(
            lambda ar: jnp.sum(self._tau(atm, alpha_ref=ar))
        )(alpha_ref))
        assert np.all(np.isfinite(grad)), (
            f"d(tau)/d(alpha_ref) is non-finite with alpha_ref[3]={floor_value}"
        )
        if floor_value <= 1e-30:
            assert grad[3] == 0.0, (
                "a clamped alpha_ref must have zero derivative, got "
                f"{grad[3]} for alpha_ref[3]={floor_value}"
            )
        grad_alpha = np.asarray(jax.grad(
            lambda a: jnp.sum(self._tau(atm, alpha=a, alpha_ref=alpha_ref))
        )(atm["alpha"]))
        assert np.all(np.isfinite(grad_alpha)), (
            "d(tau)/d(alpha) is non-finite with a clamped alpha_ref"
        )

    def test_top_boundary_gradient_is_exactly_zero(self):
        """
        tau[0] = 0 is a hard-coded constant, so all of its gradients vanish.

        This is the "optical depth vanishes at the top boundary" case: the
        concern is not that the derivative is zero (it should be), but that the
        zero row could be contaminated by NaN from the neighbouring machinery.
        """
        atm = model_atmosphere()
        for key in ("alpha", "alpha_ref", "log_tau_ref"):
            grad = np.asarray(jax.grad(
                lambda v, key=key: self._tau(atm, **{key: v})[0]
            )(atm[key]))
            np.testing.assert_array_equal(
                grad, np.zeros_like(grad),
                err_msg=f"d(tau[0])/d({key}) is not exactly zero",
            )

    def test_isothermal_flat_atmosphere_is_differentiable(self):
        """A completely degenerate atmosphere must still give finite gradients."""
        atm = model_atmosphere(isothermal=True, flat_opacity=True)
        for key in ("alpha", "alpha_ref", "log_tau_ref"):
            grad = np.asarray(jax.grad(
                lambda v, key=key: jnp.sum(self._tau(atm, **{key: v}))
            )(atm[key]))
            assert np.all(np.isfinite(grad)), f"d(tau)/d({key}) is non-finite"


# ---------------------------------------------------------------------------
# compute_tau_bezier
# ---------------------------------------------------------------------------


class TestComputeTauBezierGradients:
    """Gradients of the Bezier optical depth scheme."""

    def test_gradients_are_finite_for_a_generic_atmosphere(self):
        """Baseline: a smooth atmosphere must differentiate cleanly."""
        atm = model_atmosphere()
        for key in ("alpha", "spatial_coord"):
            jac = np.asarray(jax.jacrev(
                lambda v, key=key: compute_tau_bezier(
                    v if key == "alpha" else atm["alpha"],
                    v if key == "spatial_coord" else atm["spatial_coord"],
                )
            )(atm[key]))
            assert np.all(np.isfinite(jac)), f"d(tau)/d({key}) is non-finite"

    def test_gradient_wrt_alpha_matches_finite_difference(self):
        """AD against a multiplicative central difference on alpha."""
        atm = model_atmosphere()
        assert_array_grad_matches_fd(
            lambda a: jnp.sum(compute_tau_bezier(a, atm["spatial_coord"])),
            atm["alpha"], step=1e-6, name="compute_tau_bezier d/d(alpha)",
        )

    def test_gradient_wrt_spatial_coord_matches_finite_difference(self):
        """
        The Bezier scheme integrates over the spatial coordinate itself.

        Unlike the anchored scheme, ``spatial_coord`` is live here, so the
        gradient must be non-zero and correct.
        """
        atm = model_atmosphere()
        # The direction is scaled to the coordinate itself (~1e9 cm): an
        # absolute perturbation of order unity would be lost in the last bits
        # of the coordinate and the finite difference would return noise.
        grad = assert_array_grad_matches_fd(
            lambda sc: jnp.sum(compute_tau_bezier(atm["alpha"], sc)),
            atm["spatial_coord"],
            direction=np.linspace(1.0, 2.0, atm["spatial_coord"].size) * 1e9,
            step=1e-6, name="compute_tau_bezier d/d(spatial_coord)",
        )
        assert np.any(grad != 0.0), "d(tau)/d(spatial_coord) is identically zero"

    def test_flat_opacity_column_is_differentiable(self):
        """
        Regression: a constant ``alpha`` used to give NaN gradients.

        ``compute_tau_bezier`` builds its control points with
        ``fritsch_butland_C(spatial_coord, alpha)``, whose denominator vanishes
        identically when alpha is constant. The value was finite throughout, so
        nothing looked wrong until a gradient was requested.
        """
        atm = model_atmosphere(flat_opacity=True)
        tau = np.asarray(compute_tau_bezier(atm["alpha"], atm["spatial_coord"]))
        assert np.all(np.isfinite(tau))
        jac = np.asarray(jax.jacrev(
            lambda a: compute_tau_bezier(a, atm["spatial_coord"])
        )(atm["alpha"]))
        assert np.all(np.isfinite(jac)), (
            "d(tau)/d(alpha) is NaN for a flat opacity column "
            "(masked 0/0 in fritsch_butland_C)"
        )
        assert np.any(jac != 0.0), "d(tau)/d(alpha) is identically zero"

    def test_top_boundary_gradient_is_exactly_zero(self):
        """tau[0] is the hard-coded seed 1e-5, so it carries no gradient."""
        atm = model_atmosphere()
        grad = np.asarray(jax.grad(
            lambda a: compute_tau_bezier(a, atm["spatial_coord"])[0]
        )(atm["alpha"]))
        np.testing.assert_array_equal(grad, np.zeros_like(grad))


class TestComputeTauDirectGradients:
    """
    Gradients of the direct spatial integration of optical depth.

    ``compute_tau_direct`` is public but no ``tau_scheme`` selects it, so it is
    reachable only by calling it explicitly. It is covered here because it is a
    float-in/float-out function in the module, and because its ``dz / mu``
    divides by an angle cosine that can legitimately approach zero.
    """

    @pytest.mark.parametrize("spherical", [False, True])
    @pytest.mark.parametrize("mu", [1.0, 0.5, 1e-3])
    def test_jacobian_wrt_alpha_is_finite(self, spherical, mu):
        """Integration is linear in alpha, so the Jacobian is a constant matrix."""
        atm = model_atmosphere()
        jac = np.asarray(jax.jacrev(
            lambda a: compute_tau_direct(a, atm["spatial_coord"],
                                         spherical=spherical, mu=mu)
        )(atm["alpha"]))
        assert np.all(np.isfinite(jac)), "d(tau)/d(alpha) is non-finite"
        assert np.any(jac != 0.0)
        # The surface layer is the zero boundary condition.
        np.testing.assert_array_equal(jac[-1], np.zeros_like(jac[-1]))

    def test_gradients_match_finite_differences(self):
        """AD against central differences in alpha and in the spatial coordinate."""
        atm = model_atmosphere()
        assert_array_grad_matches_fd(
            lambda a: jnp.sum(compute_tau_direct(a, atm["spatial_coord"])),
            atm["alpha"], step=1e-6, name="compute_tau_direct d/d(alpha)",
        )
        assert_array_grad_matches_fd(
            lambda sc: jnp.sum(compute_tau_direct(atm["alpha"], sc)),
            atm["spatial_coord"],
            direction=np.linspace(1.0, 2.0, atm["spatial_coord"].size) * 1e9,
            step=1e-6, name="compute_tau_direct d/d(spatial_coord)",
        )

    @pytest.mark.parametrize("mu", [1.0, 0.5, 0.1, 1e-3])
    def test_gradient_wrt_mu_matches_finite_difference(self, mu):
        """
        The path length is ``dz / mu``, so ``d(tau)/d(mu) = -tau / mu``.

        The gradient therefore grows without bound for grazing rays, which is
        physically correct rather than a numerical failure.
        """
        atm = model_atmosphere()
        g = assert_grad_matches_fd(
            lambda m: jnp.sum(compute_tau_direct(atm["alpha"],
                                                 atm["spatial_coord"], mu=m)),
            mu, step=1e-7 * mu, rtol=1e-4, name="compute_tau_direct d/d(mu)",
        )
        assert g < 0.0

    def test_exactly_grazing_ray_diverges_rather_than_nan(self):
        """
        At mu = 0 the path length is infinite, and the gradient must say so.

        A signed infinity is the honest answer here; a NaN would mean a masked
        branch had leaked.
        """
        atm = model_atmosphere()
        g = float(jax.grad(
            lambda m: jnp.sum(compute_tau_direct(atm["alpha"],
                                                 atm["spatial_coord"], mu=m))
        )(np.float64(0.0)))
        assert np.isneginf(g), f"d(tau)/d(mu) at mu=0 should be -inf, got {g}"


# ---------------------------------------------------------------------------
# compute_I_linear_flux_only / compute_F_flux_only_expint / compute_I_bezier
# ---------------------------------------------------------------------------


def _tau_and_S(n_layers=20, dtau_scale=1.0):
    """
    A monotone optical depth grid starting at exactly zero, plus a source.

    Parameters
    ----------
    n_layers : int, optional
        Number of layers.
    dtau_scale : float, optional
        Multiplies the whole tau grid, so that ``dtau_scale -> 0`` drives every
        ``Δτ`` to zero simultaneously.

    Returns
    -------
    tuple of jax.Array
        ``(tau, S)``.
    """
    tau = jnp.concatenate([
        jnp.array([0.0]),
        jnp.logspace(-4.0, 1.0, n_layers - 1),
    ]) * dtau_scale
    S = jnp.linspace(2.0e-5, 9.0e-5, n_layers)
    return tau, S


INTENSITY_FUNCTIONS = [
    ("compute_I_linear_flux_only", compute_I_linear_flux_only),
    ("compute_F_flux_only_expint", compute_F_flux_only_expint),
]


class TestFluxOnlyIntensityGradients:
    """
    Gradients of the two flux-only formal solutions.

    Both assume a piecewise linear source function, so both divide by ``Δτ``:
    the removable ``0/0`` as ``Δτ → 0`` is the boundary of interest, along with
    the ``τ = 0`` top boundary that feeds ``E2(0)``.
    """

    @pytest.mark.parametrize("name,func", INTENSITY_FUNCTIONS)
    def test_gradient_wrt_S_is_finite_and_positive(self, name, func):
        """
        More source function anywhere means more emergent flux.

        Every component of dI/dS must therefore be strictly positive, which is a
        much stronger statement than mere finiteness.
        """
        tau, S = _tau_and_S()
        grad = np.asarray(jax.grad(lambda s: func(tau, s))(S))
        assert np.all(np.isfinite(grad)), f"{name}: dI/dS is non-finite: {grad}"
        assert np.all(grad > 0.0), f"{name}: dI/dS should be positive, got {grad}"

    @pytest.mark.parametrize("name,func", INTENSITY_FUNCTIONS)
    def test_gradient_wrt_S_matches_finite_difference(self, name, func):
        """AD against a multiplicative central difference on S."""
        tau, S = _tau_and_S()
        assert_array_grad_matches_fd(lambda s: func(tau, s), S, step=1e-6,
                                     name=f"{name} d/dS")

    @pytest.mark.parametrize("name,func", INTENSITY_FUNCTIONS)
    def test_gradient_wrt_tau_is_finite_at_the_top_boundary(self, name, func):
        """
        Regression: ``compute_F_flux_only_expint`` gave a NaN at ``tau[0] = 0``.

        The antiderivative evaluates ``E2(tau[0])`` with ``tau[0] = 0``
        exactly, which is precisely the argument at which E2's gradient was
        NaN. Note the value was finite and correct throughout -- only the
        gradient was poisoned, and only in its first component.
        """
        tau, S = _tau_and_S()
        grad = np.asarray(jax.grad(lambda t: func(t, S))(tau))
        assert np.all(np.isfinite(grad)), (
            f"{name}: dI/d(tau) is non-finite: {grad} "
            f"(component 0 corresponds to tau = {float(tau[0])})"
        )
        assert grad[0] < 0.0, (
            f"{name}: pushing the top boundary deeper must reduce the flux, "
            f"got dI/d(tau[0]) = {grad[0]}"
        )

    @pytest.mark.parametrize("name,func", INTENSITY_FUNCTIONS)
    def test_gradient_wrt_tau_matches_finite_difference(self, name, func):
        """
        AD against a central difference on the interior of the tau grid.

        The top boundary is held fixed because tau[0] = 0 is the edge of the
        domain and cannot be centrally differenced.
        """
        tau, S = _tau_and_S()
        direction = np.asarray(tau, dtype=np.float64).copy()
        direction[0] = 0.0
        assert_array_grad_matches_fd(
            lambda t: func(t, S), tau, direction=direction, step=1e-6,
            name=f"{name} d/d(tau)",
        )

    @pytest.mark.parametrize("name,func", INTENSITY_FUNCTIONS)
    @pytest.mark.parametrize("dtau", [1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13])
    def test_gradient_survives_the_vanishing_dtau_limit(self, name, func, dtau):
        """
        As ``Δτ → 0`` the linear-source integrand becomes ``0/0``.

        The slope ``m = ΔS/Δτ`` diverges while the interval it is integrated
        over shrinks; the product has a finite limit, but a naive evaluation can
        lose it. Check the gradient stays finite over twelve decades of ``Δτ``.
        Below ~1e-8 the *accuracy* degrades through cancellation in
        ``a = S - m tau`` (a floating-point issue, not an AD one), so only
        finiteness is asserted there.
        """
        tau = jnp.array([0.0, dtau])
        S = jnp.array([1.0, 1.0 + 3.0 * dtau])
        value = float(func(tau, S))
        assert np.isfinite(value), f"{name}: value is {value} at dtau={dtau}"

        grad_S = np.asarray(jax.grad(lambda s: func(tau, s))(S))
        grad_tau = np.asarray(jax.grad(lambda t: func(t, S))(tau))
        assert np.all(np.isfinite(grad_S)), f"{name}: dI/dS is {grad_S} at dtau={dtau}"
        assert np.all(np.isfinite(grad_tau)), (
            f"{name}: dI/d(tau) is {grad_tau} at dtau={dtau}"
        )

    @pytest.mark.parametrize("name,func", INTENSITY_FUNCTIONS)
    @pytest.mark.parametrize("dtau", [1e-1, 1e-3, 1e-5])
    def test_vanishing_dtau_gradient_matches_finite_difference(self, name, func, dtau):
        """
        In the shrinking-interval limit the gradient must still be *correct*.

        For a single layer of thickness ``Δτ`` starting at ``τ = 0``, the
        emergent intensity tends to ``S Δτ``, so the two components of
        ``dI/dS`` must sum to ``Δτ``. The comparison spans four decades of
        ``Δτ``; smaller values are excluded because there the finite
        difference becomes unreliable, not the AD.
        """
        tau = jnp.array([0.0, dtau])
        S = np.array([1.0, 1.0 + 3.0 * dtau])
        grad = assert_array_grad_matches_fd(
            lambda s: func(tau, s), S, step=1e-6,
            rtol=1e-4, name=f"{name} d/dS at dtau={dtau}",
        )
        # For a single layer of thickness dtau starting at tau = 0, both
        # schemes give I -> S * dtau (E2(0) = exp(-0) = 1), with a relative
        # correction of order dtau.
        total = float(grad.sum())
        assert abs(total - dtau) < 10.0 * dtau ** 2 + 1e-12, (
            f"{name}: sum(dI/dS) = {total} does not approach dtau = {dtau}"
        )

    @pytest.mark.parametrize("name,func", INTENSITY_FUNCTIONS)
    def test_uniform_shrinking_of_the_whole_tau_grid(self, name, func):
        """
        Squeeze every layer towards zero optical depth at once.

        This is the optically thin limit of a real atmosphere, where all
        ``Δτ`` vanish together rather than one at a time.
        """
        for scale in [1.0, 1e-3, 1e-6, 1e-9, 1e-12]:
            tau, S = _tau_and_S(dtau_scale=scale)
            grad = np.asarray(jax.grad(lambda s: func(tau, s))(S))
            assert np.all(np.isfinite(grad)), (
                f"{name}: dI/dS is non-finite with the tau grid scaled by {scale}"
            )

    @pytest.mark.parametrize("name,func", INTENSITY_FUNCTIONS)
    def test_isothermal_source_function_is_differentiable(self, name, func):
        """A constant S makes every layer's slope exactly zero."""
        tau, _ = _tau_and_S()
        S = jnp.full(tau.shape, 5.0e-5)
        grad = np.asarray(jax.grad(lambda s: func(tau, s))(S))
        assert np.all(np.isfinite(grad)), f"{name}: dI/dS is non-finite for constant S"
        assert np.all(grad > 0.0)

    @pytest.mark.parametrize("name,func", INTENSITY_FUNCTIONS)
    @pytest.mark.xfail(
        reason="exactly duplicated tau values give an unmasked 0/0: m = dS/dtau "
               "is inf/NaN and the value itself becomes NaN. This is a loud "
               "failure (the flux is visibly NaN), not a masked-cotangent bug, "
               "so it is documented rather than silently patched -- fixing it "
               "would change values in a regime that currently returns NaN.",
        strict=True,
    )
    def test_exactly_degenerate_layer_is_differentiable(self, name, func):
        """Two layers at identical optical depth: Δτ is exactly zero."""
        tau, S = _tau_and_S()
        tau = tau.at[5].set(float(tau[4]))
        assert np.isfinite(float(func(tau, S)))
        grad = np.asarray(jax.grad(lambda s: func(tau, s))(S))
        assert np.all(np.isfinite(grad))


class TestComputeIBezierGradients:
    """Gradients of the Bezier formal solution."""

    def test_gradient_wrt_S_matches_finite_difference(self):
        """AD against a multiplicative central difference on S."""
        tau, S = _tau_and_S()
        tau = tau + 1e-6  # the Bezier scheme needs a strictly increasing tau
        assert_array_grad_matches_fd(lambda s: compute_I_bezier(tau, s), S,
                                     step=1e-6, name="compute_I_bezier d/dS")

    def test_gradient_wrt_tau_is_finite(self):
        """The recursion exponentiates tau differences; all must stay finite."""
        tau, S = _tau_and_S()
        tau = tau + 1e-6
        grad = np.asarray(jax.grad(lambda t: compute_I_bezier(t, S))(tau))
        assert np.all(np.isfinite(grad)), f"dI/d(tau) is non-finite: {grad}"

    def test_isothermal_source_function_is_differentiable(self):
        """
        Regression: a constant S used to give a NaN dI/dS.

        ``compute_I_bezier`` builds control points with
        ``fritsch_butland_C(tau, S)``, so an isothermal atmosphere zeroed the
        Fritsch-Butland denominator and poisoned the gradient while leaving the
        intensity finite and correct.
        """
        tau, _ = _tau_and_S()
        tau = tau + 1e-6
        S = jnp.full(tau.shape, 5.0e-5)
        value = float(compute_I_bezier(tau, S))
        assert np.isfinite(value)
        grad = np.asarray(jax.grad(lambda s: compute_I_bezier(tau, s))(S))
        assert np.all(np.isfinite(grad)), (
            f"dI/dS is NaN for an isothermal atmosphere: {grad}"
        )
        assert np.any(grad != 0.0)

    @pytest.mark.parametrize("scale", [1.0, 1e-3, 1e-6, 1e-9])
    def test_vanishing_dtau_limit(self, scale):
        """
        The Bezier coefficients divide by ``Δτ**2``, guarded by a 1e-30 floor.

        Shrinking the whole grid drives ``Δτ**2`` below that floor, so this
        exercises both sides of the guard.
        """
        tau, S = _tau_and_S(dtau_scale=scale)
        tau = tau + 1e-8 * scale
        grad = np.asarray(jax.grad(lambda s: compute_I_bezier(tau, s))(S))
        assert np.all(np.isfinite(grad)), (
            f"dI/dS is non-finite with the tau grid scaled by {scale}"
        )


class TestComputeILinearGradients:
    """
    Gradients of the angle-resolved linear formal solution.

    ``compute_I_linear`` is decorated with ``@jit`` but branches on a traced
    value (``if delta_tau <= 0: continue``), so it cannot be traced at all: any
    call raises ``TracerBoolConversionError``, and so does
    ``radiative_transfer(..., intensity_scheme="linear")``. It *is* usable and
    differentiable under ``jax.disable_jit()``, where the branch sees concrete
    primals, so the gradient tests below run in that mode. This is a real defect
    but not an AD-masking one, and fixing it means rewriting the loop, so it is
    documented rather than patched here.
    """

    def test_cannot_be_traced_under_jit(self):
        """Document the data-dependent Python branch that blocks tracing."""
        tau, S = _tau_and_S(n_layers=6)
        with pytest.raises(jax.errors.TracerBoolConversionError):
            compute_I_linear(tau, S, 1.0)

    def test_intensity_scheme_linear_cannot_be_traced(self):
        """The defect propagates to the public entry point."""
        atm = model_atmosphere(n_layers=8)
        with pytest.raises(jax.errors.TracerBoolConversionError):
            radiative_transfer_single_wavelength(
                atm["alpha"], atm["S"], atm["spatial_coord"], atm["log_tau_ref"],
                alpha_ref=atm["alpha_ref"], intensity_scheme="linear",
            )

    @pytest.mark.parametrize("mu", [1.0, 0.95, 0.5, 0.1, 1e-2, 1e-4, 1e-8])
    def test_gradient_wrt_S_is_finite_at_every_mu(self, mu):
        """
        Every ray from disc centre (mu = 1) to grazing (mu -> 0) must work.

        The slant optical depth is ``tau / mu``, so small mu drives the
        exponentials to zero and the linear coefficients to huge values -- the
        classic place for an ``inf * 0``.
        """
        tau, S = _tau_and_S(n_layers=10)
        with jax.disable_jit():
            grad = np.asarray(jax.grad(
                lambda s: compute_I_linear(tau, s, np.float64(mu))
            )(S))
        assert np.all(np.isfinite(grad)), f"dI/dS is non-finite at mu={mu}: {grad}"
        assert np.any(grad > 0.0)

    @pytest.mark.parametrize("mu", [1.0, 0.95, 0.5, 0.1, 1e-2])
    def test_gradient_wrt_S_matches_finite_difference(self, mu):
        """AD against a multiplicative central difference on S."""
        tau, S = _tau_and_S(n_layers=10)
        with jax.disable_jit():
            assert_array_grad_matches_fd(
                lambda s: compute_I_linear(tau, s, np.float64(mu)), S,
                step=1e-6, name=f"compute_I_linear d/dS at mu={mu}",
            )

    @pytest.mark.parametrize("mu", [1.0, 0.95, 0.5, 0.1, 1e-2])
    def test_gradient_wrt_mu_matches_finite_difference(self, mu):
        """
        dI/dmu is the limb-darkening derivative and must be positive.

        Looking closer to disc centre sees deeper, hotter layers, so the
        intensity increases with mu for a source function that increases
        inwards.
        """
        tau, S = _tau_and_S(n_layers=10)
        with jax.disable_jit():
            g = assert_grad_matches_fd(
                lambda m: compute_I_linear(tau, S, m), mu,
                step=1e-7 * mu, rtol=1e-4, name=f"compute_I_linear d/dmu",
            )
        assert g > 0.0, f"dI/dmu should be positive, got {g} at mu={mu}"

    def test_exactly_grazing_ray_is_not_differentiable(self):
        """
        At mu = 0 exactly the slant optical depth is ``tau / 0``.

        The value is NaN, so this is a loud failure rather than a masked one.
        The quadrature grid never contains mu = 0 (Gauss-Legendre nodes are
        strictly interior), so it is documented rather than guarded.
        """
        tau, S = _tau_and_S(n_layers=6)
        with jax.disable_jit():
            value = float(compute_I_linear(tau, S, np.float64(0.0)))
        assert np.isnan(value), (
            "mu = 0 now produces a finite value; the grazing-ray note in this "
            "test is stale and should be updated"
        )


class TestComputeFluxFromIntensitiesGradients:
    """Gradients of the angle quadrature ``F = 2 pi sum w I mu``."""

    def test_gradients_are_exact(self):
        """The quadrature is bilinear, so its gradients are closed form."""
        mu_points, mu_weights = generate_mu_grid(5)
        intensities = jnp.linspace(1.0, 2.0, 5)

        grad_I = np.asarray(jax.grad(
            lambda i: compute_flux_from_intensities(i, mu_points, mu_weights)
        )(intensities))
        np.testing.assert_allclose(
            grad_I, 2.0 * np.pi * np.asarray(mu_weights) * np.asarray(mu_points),
            rtol=1e-14,
        )

        grad_mu = np.asarray(jax.grad(
            lambda m: compute_flux_from_intensities(intensities, m, mu_weights)
        )(mu_points))
        np.testing.assert_allclose(
            grad_mu,
            2.0 * np.pi * np.asarray(mu_weights) * np.asarray(intensities),
            rtol=1e-14,
        )
        assert np.all(np.isfinite(grad_I)) and np.all(np.isfinite(grad_mu))


# ---------------------------------------------------------------------------
# generate_mu_grid
# ---------------------------------------------------------------------------


class TestMuGridDifferentiability:
    """
    Is ``generate_mu_grid`` differentiable through its eigendecomposition?

    Short answer: the question does not arise for the function as written.
    ``generate_mu_grid(n_mu)`` takes a single Python integer and builds the
    Jacobi matrix from ``jnp.arange``, so its output is a compile-time constant
    with no float input to differentiate with respect to. There is nothing for
    ``jax.grad`` to attach to, and nothing in a synthesis that varies it.

    Longer answer: the ``jnp.linalg.eigh`` it uses is not itself an obstruction.
    eigh's JVP divides by eigenvalue *differences*, so it is differentiable
    exactly when the eigenvalues are distinct -- and the Gauss-Legendre nodes
    are the roots of a Legendre polynomial, which are always simple. The test
    below feeds a perturbation parameter into the Jacobi matrix to confirm that
    gradients do flow through the eigendecomposition, so if the mu grid ever
    becomes a function of a continuous parameter it will differentiate cleanly.

    ``calculate_rays`` is not implemented in this port, so there is nothing to
    test for it.
    """

    @pytest.mark.parametrize("n_mu", [2, 3, 5, 7, 20])
    def test_output_is_a_constant_with_no_float_input(self, n_mu):
        """The mu grid depends only on an integer, so it carries no gradient."""
        mu_points, mu_weights = generate_mu_grid(n_mu)
        assert mu_points.shape == (n_mu,)
        assert np.all(np.isfinite(np.asarray(mu_points)))
        assert np.all(np.isfinite(np.asarray(mu_weights)))
        # Gauss-Legendre nodes are strictly interior: no ray is exactly grazing
        # (mu = 0) or exactly vertical (mu = 1), which is what keeps the
        # tau / mu of the angle-resolved schemes finite.
        assert np.all(np.asarray(mu_points) > 0.0)
        assert np.all(np.asarray(mu_points) < 1.0)
        np.testing.assert_allclose(float(jnp.sum(mu_weights)), 1.0, rtol=1e-12)

    @pytest.mark.parametrize("n_mu", [2, 3, 5, 7, 20])
    def test_eigendecomposition_is_differentiable(self, n_mu):
        """
        Gradients flow through ``jnp.linalg.eigh`` for the Jacobi matrix.

        A scale factor is threaded into the off-diagonal so there is something
        to differentiate with respect to. eigh's JVP would return NaN for
        degenerate eigenvalues; Legendre roots are simple, so it does not.
        """
        def scaled_nodes_and_weights(scale):
            i = jnp.arange(1, n_mu)
            beta = (i / jnp.sqrt(4 * i ** 2 - 1)) * (1.0 + scale)
            T = jnp.diag(beta, -1) + jnp.diag(beta, 1)
            nodes, V = jnp.linalg.eigh(T)
            return jnp.sum(nodes ** 2) + jnp.sum(2 * V[0, :] ** 2)

        assert_grad_matches_fd(scaled_nodes_and_weights, 0.0, step=1e-6,
                               atol=1e-9, name=f"eigh (n_mu={n_mu})")

    def test_nodes_are_simple(self):
        """
        eigh is differentiable precisely because the nodes are distinct.

        Record the minimum gap so that the justification above is checked
        rather than asserted.
        """
        for n_mu in (2, 3, 5, 7, 20):
            nodes, _ = leggauss(n_mu)
            gaps = np.diff(np.sort(np.asarray(nodes)))
            assert np.all(gaps > 1e-3), (
                f"Gauss-Legendre nodes for n={n_mu} are nearly degenerate "
                f"(minimum gap {gaps.min():.3e}); eigh's JVP would be unstable"
            )


# ---------------------------------------------------------------------------
# radiative_transfer, end to end
# ---------------------------------------------------------------------------

# "linear" is excluded: it cannot be traced at all (see TestComputeILinear).
SCHEME_COMBINATIONS = [
    ("anchored", "linear_flux_only", True),
    ("anchored", "linear_flux_only", False),
    ("anchored", "bezier", True),
    ("bezier", "linear_flux_only", True),
    ("bezier", "linear_flux_only", False),
    ("bezier", "bezier", True),
]


def fd_settings(intensity_scheme):
    """
    Finite-difference step and tolerance appropriate to an intensity scheme.

    ``compute_I_bezier``'s integration coefficients are of the form
    ``(2 + d**2 - 2d - 2 exp(-d)) / d**2``, whose numerator is a difference of
    O(1) terms that cancels to O(d**3). For the thin layers at the top of an
    atmosphere this costs about eight significant digits, so the *value* of the
    Bezier flux carries ~5e-8 of relative noise. A central difference inherits
    that noise divided by the step, which caps the achievable agreement at
    ~3e-5 around a step of 1e-3 -- so a smaller step makes the comparison worse,
    not better. The AD gradient is the accurate one here; the finite difference
    is the limiting factor.

    Parameters
    ----------
    intensity_scheme : str
        The ``intensity_scheme`` passed to ``radiative_transfer``.

    Returns
    -------
    dict
        Keyword arguments for :func:`assert_array_grad_matches_fd`.
    """
    if intensity_scheme == "bezier":
        return dict(step=1e-3, rtol=1e-3)
    return dict(step=1e-6, rtol=FD_RTOL)


def _rt_flux(atm, tau_scheme, intensity_scheme, use_expint_flux, spherical,
             **overrides):
    """
    Total flux over a three-wavelength grid, as a differentiable scalar.

    Parameters
    ----------
    atm : dict
        Model atmosphere from :func:`model_atmosphere`.
    tau_scheme, intensity_scheme : str
        Scheme selectors passed straight through.
    use_expint_flux : bool
        Exponential-integral flux switch.
    spherical : bool
        Geometry switch.
    **overrides
        Replacement values for any of ``alpha_grid``, ``S_grid``, ``alpha_ref``,
        ``log_tau_ref`` or ``spatial_coord``.

    Returns
    -------
    jax.Array
        Scalar total flux.
    """
    alpha_grid = overrides.get(
        "alpha_grid",
        jnp.stack([atm["alpha"], atm["alpha"] * 1.4, atm["alpha"] * 0.6]),
    )
    S_grid = overrides.get(
        "S_grid", jnp.stack([atm["S"], atm["S"] * 1.05, atm["S"] * 0.95]),
    )
    fluxes, _ = radiative_transfer(
        alpha_grid,
        S_grid,
        overrides.get("spatial_coord", atm["spatial_coord"]),
        overrides.get("log_tau_ref", atm["log_tau_ref"]),
        alpha_ref=overrides.get("alpha_ref", atm["alpha_ref"]),
        spherical=spherical,
        tau_scheme=tau_scheme,
        intensity_scheme=intensity_scheme,
        use_expint_flux=use_expint_flux,
        n_mu=5,
    )
    return jnp.sum(fluxes)


class TestRadiativeTransferEndToEnd:
    """
    Gradients of the full solver across every traceable option combination.

    The differentiated quantities are the ones a synthesis actually varies: the
    absorption coefficient, the source function, the reference opacity, the
    reference optical depth scale and the spatial coordinate.
    """

    @pytest.mark.parametrize("spherical", [False, True])
    @pytest.mark.parametrize("tau_scheme,intensity_scheme,use_expint_flux",
                             SCHEME_COMBINATIONS)
    def test_gradient_wrt_alpha_is_finite_and_negative(
            self, tau_scheme, intensity_scheme, use_expint_flux, spherical):
        """
        More opacity means less emergent flux, everywhere in the atmosphere.

        A finite gradient is the minimum bar; the sign is the physics.
        """
        atm = model_atmosphere(n_layers=16)
        alpha_grid = jnp.stack([atm["alpha"], atm["alpha"] * 1.4, atm["alpha"] * 0.6])
        grad = np.asarray(jax.grad(
            lambda a: _rt_flux(atm, tau_scheme, intensity_scheme, use_expint_flux,
                               spherical, alpha_grid=a)
        )(alpha_grid))
        assert np.all(np.isfinite(grad)), f"d(flux)/d(alpha) is non-finite: {grad}"
        assert np.any(grad != 0.0), "d(flux)/d(alpha) is identically zero"
        assert np.all(grad <= 0.0), (
            f"increasing opacity must not increase the flux, got max {grad.max()}"
        )

    @pytest.mark.parametrize("spherical", [False, True])
    @pytest.mark.parametrize("tau_scheme,intensity_scheme,use_expint_flux",
                             SCHEME_COMBINATIONS)
    def test_gradient_wrt_S_is_finite_and_positive(
            self, tau_scheme, intensity_scheme, use_expint_flux, spherical):
        """More source function means more flux, at every layer."""
        atm = model_atmosphere(n_layers=16)
        S_grid = jnp.stack([atm["S"], atm["S"] * 1.05, atm["S"] * 0.95])
        grad = np.asarray(jax.grad(
            lambda s: _rt_flux(atm, tau_scheme, intensity_scheme, use_expint_flux,
                               spherical, S_grid=s)
        )(S_grid))
        assert np.all(np.isfinite(grad)), f"d(flux)/dS is non-finite: {grad}"
        assert np.all(grad >= 0.0), f"d(flux)/dS should be non-negative: {grad.min()}"
        assert np.any(grad > 0.0)

    @pytest.mark.parametrize("tau_scheme,intensity_scheme,use_expint_flux",
                             SCHEME_COMBINATIONS)
    def test_gradient_wrt_alpha_matches_finite_difference(
            self, tau_scheme, intensity_scheme, use_expint_flux):
        """AD against a multiplicative central difference on the opacity grid."""
        atm = model_atmosphere(n_layers=16)
        alpha_grid = np.asarray(
            jnp.stack([atm["alpha"], atm["alpha"] * 1.4, atm["alpha"] * 0.6]))
        assert_array_grad_matches_fd(
            lambda a: _rt_flux(atm, tau_scheme, intensity_scheme, use_expint_flux,
                               False, alpha_grid=a),
            alpha_grid,
            name=f"radiative_transfer d/d(alpha) [{tau_scheme}/{intensity_scheme}]",
            **fd_settings(intensity_scheme),
        )

    @pytest.mark.parametrize("tau_scheme,intensity_scheme,use_expint_flux",
                             SCHEME_COMBINATIONS)
    def test_gradient_wrt_S_matches_finite_difference(
            self, tau_scheme, intensity_scheme, use_expint_flux):
        """AD against a multiplicative central difference on the source grid."""
        atm = model_atmosphere(n_layers=16)
        S_grid = np.asarray(jnp.stack([atm["S"], atm["S"] * 1.05, atm["S"] * 0.95]))
        assert_array_grad_matches_fd(
            lambda s: _rt_flux(atm, tau_scheme, intensity_scheme, use_expint_flux,
                               False, S_grid=s),
            S_grid,
            name=f"radiative_transfer d/dS [{tau_scheme}/{intensity_scheme}]",
            **fd_settings(intensity_scheme),
        )

    @pytest.mark.parametrize("intensity_scheme", ["linear_flux_only", "bezier"])
    def test_gradient_wrt_alpha_ref_matches_finite_difference(self, intensity_scheme):
        """
        ``alpha_ref`` only enters through the anchored tau scheme.

        It is the denominator of the opacity ratio, so its gradient is the one
        that would blow up if the small-``alpha_ref`` guard were wrong.
        """
        atm = model_atmosphere(n_layers=16)
        grad = assert_array_grad_matches_fd(
            lambda ar: _rt_flux(atm, "anchored", intensity_scheme, True, False,
                                alpha_ref=ar),
            np.asarray(atm["alpha_ref"]),
            name=f"radiative_transfer d/d(alpha_ref) [{intensity_scheme}]",
            **fd_settings(intensity_scheme),
        )
        assert np.any(grad != 0.0)

    @pytest.mark.parametrize("intensity_scheme", ["linear_flux_only", "bezier"])
    def test_gradient_wrt_log_tau_ref_matches_finite_difference(self, intensity_scheme):
        """``log_tau_ref`` is the anchored scheme's integration variable."""
        atm = model_atmosphere(n_layers=16)
        grad = assert_array_grad_matches_fd(
            lambda lt: _rt_flux(atm, "anchored", intensity_scheme, True, False,
                                log_tau_ref=lt),
            np.asarray(atm["log_tau_ref"]),
            name=f"radiative_transfer d/d(log_tau_ref) [{intensity_scheme}]",
            **fd_settings(intensity_scheme),
        )
        assert np.any(grad != 0.0)

    @pytest.mark.parametrize("intensity_scheme", ["linear_flux_only", "bezier"])
    def test_gradient_wrt_spatial_coord(self, intensity_scheme):
        """
        The spatial coordinate is live for the Bezier tau scheme only.

        For the anchored scheme it is an unused argument and the gradient is an
        exact zero -- worth pinning, because a caller optimising stellar radius
        through the anchored scheme would otherwise get silence rather than an
        error.
        """
        atm = model_atmosphere(n_layers=16)
        anchored = np.asarray(jax.grad(
            lambda sc: _rt_flux(atm, "anchored", intensity_scheme, True, False,
                                spatial_coord=sc)
        )(atm["spatial_coord"]))
        np.testing.assert_array_equal(anchored, np.zeros_like(anchored))

        bezier = assert_array_grad_matches_fd(
            lambda sc: _rt_flux(atm, "bezier", intensity_scheme, True, False,
                                spatial_coord=sc),
            np.asarray(atm["spatial_coord"]),
            direction=np.linspace(1.0, 2.0, 16) * 1e9,
            name=f"radiative_transfer d/d(spatial_coord) [bezier/{intensity_scheme}]",
            **fd_settings(intensity_scheme),
        )
        assert np.any(bezier != 0.0)

    @pytest.mark.parametrize("tau_scheme,intensity_scheme,use_expint_flux",
                             SCHEME_COMBINATIONS)
    def test_isothermal_atmosphere_is_differentiable(
            self, tau_scheme, intensity_scheme, use_expint_flux):
        """
        Regression: an isothermal, flat-opacity atmosphere gave NaN gradients.

        Both Bezier schemes call ``fritsch_butland_C``, whose denominator is
        identically zero when its second argument is constant. The fluxes were
        finite and plausible; only the gradients were NaN. This is the
        end-to-end version of the ``fritsch_butland_C`` regression.
        """
        atm = model_atmosphere(n_layers=16, isothermal=True, flat_opacity=True)
        alpha_grid = jnp.stack([atm["alpha"], atm["alpha"], atm["alpha"]])
        S_grid = jnp.stack([atm["S"], atm["S"], atm["S"]])

        value = float(_rt_flux(atm, tau_scheme, intensity_scheme, use_expint_flux,
                               False, alpha_grid=alpha_grid, S_grid=S_grid))
        assert np.isfinite(value)

        grad_alpha = np.asarray(jax.grad(
            lambda a: _rt_flux(atm, tau_scheme, intensity_scheme, use_expint_flux,
                               False, alpha_grid=a, S_grid=S_grid)
        )(alpha_grid))
        grad_S = np.asarray(jax.grad(
            lambda s: _rt_flux(atm, tau_scheme, intensity_scheme, use_expint_flux,
                               False, alpha_grid=alpha_grid, S_grid=s)
        )(S_grid))
        assert np.all(np.isfinite(grad_alpha)), (
            f"d(flux)/d(alpha) is NaN for an isothermal atmosphere: {grad_alpha}"
        )
        assert np.all(np.isfinite(grad_S)), (
            f"d(flux)/dS is NaN for an isothermal atmosphere: {grad_S}"
        )

    @pytest.mark.parametrize("tau_scheme,intensity_scheme,use_expint_flux",
                             SCHEME_COMBINATIONS)
    def test_spherical_and_planar_gradients_agree(
            self, tau_scheme, intensity_scheme, use_expint_flux):
        """
        ``spherical=True`` is accepted but has no effect in this port.

        ``compute_tau_anchored`` takes the flag and ignores it, and the Bezier
        scheme never receives it, so both the values and the gradients are
        identical to the planar case. Pinned so that implementing spherical
        geometry cannot slip in unnoticed, and so nobody assumes the flag does
        something today.
        """
        atm = model_atmosphere(n_layers=16)
        alpha_grid = jnp.stack([atm["alpha"], atm["alpha"] * 1.4, atm["alpha"] * 0.6])
        grads = [
            np.asarray(jax.grad(
                lambda a, sph=sph: _rt_flux(atm, tau_scheme, intensity_scheme,
                                            use_expint_flux, sph, alpha_grid=a)
            )(alpha_grid))
            for sph in (False, True)
        ]
        assert np.all(np.isfinite(grads[0])) and np.all(np.isfinite(grads[1]))
        np.testing.assert_array_equal(grads[0], grads[1])

    @pytest.mark.parametrize("n_layers", [3, 5, 56])
    def test_gradients_are_finite_for_various_layer_counts(self, n_layers):
        """
        The Fritsch-Butland construction slices arrays four different ways.

        Three layers is the minimum for which it is defined and 56 is the size
        of a real MARCS model, so both ends are exercised.
        """
        atm = model_atmosphere(n_layers=n_layers)
        alpha_grid = jnp.stack([atm["alpha"], atm["alpha"] * 1.4])
        S_grid = jnp.stack([atm["S"], atm["S"] * 1.05])
        for tau_scheme, intensity_scheme, use_expint_flux in SCHEME_COMBINATIONS:
            grad = np.asarray(jax.grad(
                lambda a: _rt_flux(atm, tau_scheme, intensity_scheme,
                                   use_expint_flux, False, alpha_grid=a,
                                   S_grid=S_grid)
            )(alpha_grid))
            assert np.all(np.isfinite(grad)), (
                f"d(flux)/d(alpha) is non-finite for n_layers={n_layers}, "
                f"scheme {tau_scheme}/{intensity_scheme}"
            )


class TestRadiativeTransferJitGradients:
    """
    Gradients of the JIT/vmap path, which is what synthesis actually calls.

    This path always uses the anchored tau scheme plus the exponential-integral
    flux, so it depends on ``exponential_integral_2`` at ``tau = 0``.
    """

    def test_gradients_wrt_every_input_are_finite(self):
        """alpha, S, alpha_ref and log_tau_ref must all differentiate."""
        atm = model_atmosphere(n_layers=24)
        alpha_grid = jnp.stack([atm["alpha"], atm["alpha"] * 1.4, atm["alpha"] * 0.6])
        S_grid = jnp.stack([atm["S"], atm["S"] * 1.05, atm["S"] * 0.95])

        def flux(alpha_grid_, S_grid_, log_tau_ref_, alpha_ref_):
            fluxes, _ = radiative_transfer_jit(
                alpha_grid_, S_grid_, atm["spatial_coord"], log_tau_ref_, alpha_ref_)
            return jnp.sum(fluxes)

        args = (alpha_grid, S_grid, atm["log_tau_ref"], atm["alpha_ref"])
        for argnum, label in enumerate(["alpha", "S", "log_tau_ref", "alpha_ref"]):
            grad = np.asarray(jax.grad(flux, argnums=argnum)(*args))
            assert np.all(np.isfinite(grad)), f"d(flux)/d({label}) is non-finite"
            assert np.any(grad != 0.0), f"d(flux)/d({label}) is identically zero"

    def test_gradient_wrt_alpha_matches_finite_difference(self):
        """AD against a multiplicative central difference on the opacity grid."""
        atm = model_atmosphere(n_layers=24)
        alpha_grid = np.asarray(
            jnp.stack([atm["alpha"], atm["alpha"] * 1.4, atm["alpha"] * 0.6]))
        S_grid = jnp.stack([atm["S"], atm["S"] * 1.05, atm["S"] * 0.95])

        assert_array_grad_matches_fd(
            lambda a: jnp.sum(radiative_transfer_jit(
                a, S_grid, atm["spatial_coord"], atm["log_tau_ref"],
                atm["alpha_ref"])[0]),
            alpha_grid, step=1e-6, name="radiative_transfer_jit d/d(alpha)",
        )

    def test_single_wavelength_gradient_matches_finite_difference(self):
        """The scalar entry point that the vmap above is built on."""
        atm = model_atmosphere(n_layers=24)
        assert_array_grad_matches_fd(
            lambda a: radiative_transfer_single_wavelength_jit(
                a, atm["S"], atm["log_tau_ref"], atm["alpha_ref"]),
            np.asarray(atm["alpha"]), step=1e-6,
            name="radiative_transfer_single_wavelength_jit d/d(alpha)",
        )

    def test_isothermal_atmosphere_is_differentiable(self):
        """The default synthesis path on a fully degenerate atmosphere."""
        atm = model_atmosphere(n_layers=24, isothermal=True, flat_opacity=True)
        alpha_grid = jnp.stack([atm["alpha"], atm["alpha"]])
        S_grid = jnp.stack([atm["S"], atm["S"]])
        grad = np.asarray(jax.grad(
            lambda a: jnp.sum(radiative_transfer_jit(
                a, S_grid, atm["spatial_coord"], atm["log_tau_ref"],
                atm["alpha_ref"])[0])
        )(alpha_grid))
        assert np.all(np.isfinite(grad))

    def test_hessian_is_finite(self):
        """
        Second derivatives must survive too.

        A masked NaN that happens to cancel in the first derivative can still
        surface in the second, and any Newton-style fit will ask for one.
        """
        atm = model_atmosphere(n_layers=8)

        def flux(scale):
            fluxes, _ = radiative_transfer_jit(
                (atm["alpha"] * scale)[None], atm["S"][None], atm["spatial_coord"],
                atm["log_tau_ref"], atm["alpha_ref"])
            return jnp.sum(fluxes)

        second = float(jax.grad(jax.grad(flux))(1.0))
        assert np.isfinite(second), f"d2(flux)/d(scale)2 is {second}"
        assert second != 0.0
