"""
Automatic-differentiation tests for the hydrogen line absorption code.

The whole point of this port is that spectral synthesis is differentiable with
JAX, so the functions in ``korg.hydrogen_line_absorption`` and
``korg.hydrogen_stark_data`` must return finite, *correct* gradients -- not just
finite values.

The specific failure mode these tests guard against is::

    y = jnp.where(cond, safe_expression, dangerous_expression)

If ``dangerous_expression`` evaluates to NaN or inf, ``jnp.where`` masks the
*value* but not the *cotangent*: reverse-mode AD still walks the dead branch and
the gradient comes back NaN even though the value looks perfectly healthy. This
module is full of the right ingredients for that bug -- ``exponential_integral_1``
has four regimes, ``holtsmark_profile`` is piecewise in beta with ``1/beta**2``
asymptotics, and ``hummer_mihalas_w`` evaluates the entire Hubeny+ 1994
generalisation unconditionally before throwing it away.

Three of those were live bugs, all reachable at the *line centre* (``beta == 0``,
which really is on the wavelength grid that ``bracket_line_interpolator`` builds):

* ``exponential_integral_1(0.0)`` -- the unselected ``-log(x)`` and ``.../x``
  branches are infinite there;
* ``holtsmark_profile(0.0, P)`` -- the unselected ``1/sqrt(beta)``/``1/beta**2``
  wing branches are infinite there;
* ``brackett_line_stark_profiles`` -- ``sqrt(y1)`` in the quasistatic electron
  correction has an infinite derivative at ``y1 == 0``. This one is not even
  masked; it is live.

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

from korg.atomic_data import atomic_masses
from korg.constants import RydbergH_eV, bohr_radius_cgs, c_cgs, hplanck_eV
from korg.hydrogen_line_absorption import (
    _BALMER_ABO_PARAMS,
    _h_alpha_from_precomp_jit,
    _interp_linear_2d_jax,
    _interp_linear_3d_jax,
    _process_one_stehle_line,
    _process_stehle_line_all_layers_jit,
    autodiffable_conv,
    brackett_line_stark_profiles,
    brackett_oscillator_strength,
    exponential_integral_1,
    greim_1960_Knm,
    griem_1960_Knm,
    holtsmark_profile,
    hummer_mihalas_w,
    hydrogen_line_absorption,
    hydrogen_line_absorption_core,
    normal_pdf,
    precompute_hummer_ws,
    prepare_stark_profiles_for_jit,
)
from korg.hydrogen_stark_data import hline_stark_profiles

# The full Stehle table holds 84 transitions and every one of them is traced and
# JIT-compiled separately (their interpolation grids have line-dependent shapes,
# so hydrogen_line_absorption_core unrolls a Python loop over them). Restricting
# the driver tests to the three lower Balmer lines -- which are the ones that
# carry the ABO branch and the only ones inside the Halpha window used here --
# exercises exactly the same code path in a fraction of the time. A separate
# test runs the full table.
_BALMER_ONLY = {
    key: line
    for key, line in hline_stark_profiles.items()
    if line.lower == 2 and line.upper in (3, 4, 5)
}

# For the Brackett (n = 4) branch, n_max -- and therefore which Brackett lines
# are computed -- is taken from the largest upper level present in the supplied
# Stark table. This two-line subset keeps n_max at 20 (so Brackett-gamma, m = 7,
# is included) while leaving only two Stehle lines to compile.
_BRACKETT_ENABLING = {
    key: line
    for key, line in hline_stark_profiles.items()
    if line.lower == 3 and line.upper in (4, 20)
}

# Relative tolerance for "the AD gradient matches a central finite difference".
# A central difference in float64 is limited to roughly eps^(2/3) ~ 5e-11
# relative accuracy at the optimal step size, so 1e-5 leaves several orders of
# magnitude of headroom for the step sizes chosen below.
FD_RTOL = 1e-5

# Skip marker for everything that needs the Stehle & Hutcheon profile tables.
requires_stark_tables = pytest.mark.skipif(
    len(hline_stark_profiles) == 0,
    reason="Stehle-Hutchson-hydrogen-profiles.h5 not present in korg/data",
)


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
        pieces of floating-point noise.
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


def assert_all_finite(grads, name):
    """
    Assert every element of a (possibly nested) gradient container is finite.

    Parameters
    ----------
    grads : pytree
        Gradient structure returned by ``jax.grad``/``jax.jacfwd``.
    name : str
        Label included in the assertion message.
    """
    for i, leaf in enumerate(jax.tree_util.tree_leaves(grads)):
        arr = np.asarray(leaf)
        assert np.all(np.isfinite(arr)), (
            f"{name}: gradient leaf {i} is not finite "
            f"({arr[~np.isfinite(arr)][:5] if arr.ndim else arr})"
        )


def brackett_line_center(n, m):
    """
    Line centre wavelength [cm] of the hydrogen ``n -> m`` transition.

    Parameters
    ----------
    n : int
        Lower principal quantum number.
    m : int
        Upper principal quantum number.

    Returns
    -------
    float
        Line centre in cm.
    """
    return hplanck_eV * c_cgs / (RydbergH_eV * (1 / n**2 - 1 / m**2))


# ---------------------------------------------------------------------------
# exponential_integral_1
# ---------------------------------------------------------------------------

# The four regimes are x < 0, x <= 0.01, x <= 1.0, x <= 30, x > 30. The
# interesting points are the knots themselves and x == 0, where the *unselected*
# ``-log(x)`` and ``.../x`` branches are infinite. x == 0 is reached on every
# Brackett synthesis: y1 and y2 in brackett_line_stark_profiles are proportional
# to beta and beta**2, and beta == 0 at the line centre.
EXPINT_POINTS = [
    -3.0,
    -1e-8,
    0.0,
    1e-300,
    1e-30,
    1e-8,
    0.001,
    0.00999,
    0.01,
    0.0101,
    0.1,
    0.5,
    0.999,
    1.0,
    1.001,
    5.0,
    29.99,
    30.0,
    30.01,
    100.0,
]


class TestExponentialIntegral1Gradients:
    """Gradients of the piecewise Kurucz VCSE1F approximation to E1(x)."""

    @pytest.mark.parametrize("x", EXPINT_POINTS)
    def test_gradient_is_finite(self, x):
        """No regime, selected or not, may leak an infinite cotangent."""
        g = float(jax.grad(exponential_integral_1)(np.float64(x)))
        assert np.isfinite(g), f"dE1/dx is {g} at x={x!r}"

    def test_gradient_is_finite_at_zero(self):
        """
        Regression test: E1'(0) used to be NaN.

        At x == 0 the ``x <= 0.01`` branch is selected, but the ``x <= 1``
        branch evaluates ``-log(0) == inf`` and the ``x <= 30`` branch divides
        by zero. ``jnp.where`` hid those values and reverse-mode AD turned the
        masked zero cotangent into ``0 * inf == NaN``. This is not a corner
        case: ``brackett_line_stark_profiles`` calls ``E1(y1)`` with ``y1 == 0``
        at the line centre on every Brackett line it computes.
        """
        assert np.isfinite(float(jax.grad(exponential_integral_1)(np.float64(0.0))))

    @pytest.mark.parametrize("x", [-3.0, -1e-8, -1e-300])
    def test_gradient_is_exactly_zero_for_negative_x(self, x):
        """
        For x < 0 the function returns the constant 0, so the derivative is 0.

        This is the sharpest probe of the dead-branch hazard in this function:
        every other branch is still evaluated at a negative x, where ``log(x)``
        is NaN and ``exp(-x)`` overflows. Anything other than a hard 0 means a
        cotangent leaked.
        """
        assert float(jax.grad(exponential_integral_1)(np.float64(x))) == 0.0

    @pytest.mark.parametrize("x", [0.0, 1e-300, 1e-30, 1e-11])
    def test_gradient_below_the_log_floor_is_the_linear_term(self, x):
        """
        For 0 <= x < 1e-10 the selected branch is ``-log(max(x, 1e-10)) - c + x``.

        ``max`` routes the cotangent of the log to the constant, so all that
        survives is the derivative of the ``+ x`` term: exactly 1. The value is
        unaffected (the floor was already there before these tests) -- the point
        is that the answer is a clean, finite 1 rather than the NaN the
        unguarded ``-log(x)`` and ``.../x`` branches used to inject.
        """
        assert float(jax.grad(exponential_integral_1)(np.float64(x))) == 1.0

    @pytest.mark.parametrize(
        "x", [0.001, 0.005, 0.02, 0.1, 0.5, 0.99, 1.5, 5.0, 15.0, 29.5, 50.0]
    )
    def test_gradient_matches_finite_difference(self, x):
        """Inside each regime the AD derivative must match a central difference."""
        assert_grad_matches_fd(
            exponential_integral_1,
            x,
            step=max(1e-9, abs(x) * 1e-6),
            name="exponential_integral_1",
        )

    @pytest.mark.parametrize("x", [0.001, 0.02, 0.5, 5.0, 29.0])
    def test_gradient_is_negative(self, x):
        """E1 is monotonically decreasing on x > 0, so E1' < 0 there."""
        g = float(jax.grad(exponential_integral_1)(np.float64(x)))
        assert g < 0.0, f"E1'({x}) = {g}, expected negative"

    @pytest.mark.parametrize("knot", [0.01, 1.0])
    def test_derivative_is_continuous_across_the_interior_knots(self, knot):
        """
        The 0.01 and 1.0 knots join two approximations to the same function.

        They are separate polynomial/rational fits, so the derivative is not
        analytically continuous, but the fits agree well enough that the jump
        must stay below 1e-3 relative. A much larger jump would mean a branch
        was mis-selected.
        """
        below, above = one_sided_derivatives(exponential_integral_1, knot, 1e-9)
        assert np.isfinite(below) and np.isfinite(above)
        rel = abs(above - below) / max(abs(below), abs(above))
        assert rel < 1e-3, f"E1' jumps by {rel:.3e} across x = {knot}"

    def test_derivative_drops_to_zero_above_thirty(self):
        """
        Above x = 30 the approximation returns a hard 0, so the derivative is 0.

        Both the value and the derivative are genuinely discontinuous at x = 30
        in the original Kurucz approximation (E1(30) ~ 3e-15 there); this test
        pins that behaviour rather than claiming it is smooth.
        """
        below, above = one_sided_derivatives(exponential_integral_1, 30.0, 1e-9)
        assert below < 0.0 and np.isfinite(below)
        assert above == 0.0


# ---------------------------------------------------------------------------
# holtsmark_profile
# ---------------------------------------------------------------------------

# Regimes: beta > 500 (pure asymptotic), beta <= 25.12 (tabulated PROB7
# correction blended between the PR1/PR2 forms over 8 <= beta <= 10), and the
# medium band in between. beta == 0 is the line centre and is on the grid that
# brackett_line_stark_profiles evaluates.
HOLTSMARK_BETAS = [
    0.0,
    1e-30,
    1e-8,
    0.1,
    1.0,
    1.259,
    3.0,
    7.999,
    8.0,
    8.001,
    9.999,
    10.0,
    10.001,
    20.0,
    25.119,
    25.12,
    25.121,
    60.0,
    200.0,
    499.9,
    500.0,
    500.1,
    5000.0,
]

# The shielding parameter P selects a column of the PROB7 table via
# ``IM = clip(floor(5P + 1) - 1, 0, 3)``, so 0.2/0.4/0.6/0.8 are index switches.
HOLTSMARK_PS = [0.0, 0.1, 0.2, 0.37, 0.4, 0.6, 0.8, 1.0]


class TestHoltsmarkProfileGradients:
    """Gradients of the Griem 1960 / Kurucz SOFBET Holtsmark profile."""

    @pytest.mark.parametrize("beta", HOLTSMARK_BETAS)
    @pytest.mark.parametrize("P", HOLTSMARK_PS)
    def test_gradients_are_finite(self, beta, P):
        """Both partial derivatives must be finite in every regime."""
        b, p = jnp.float64(beta), jnp.float64(P)
        d_beta, d_P = jax.grad(holtsmark_profile, argnums=(0, 1))(b, p)
        assert np.isfinite(float(d_beta)), f"d/dbeta is {d_beta} at beta={beta}, P={P}"
        assert np.isfinite(float(d_P)), f"d/dP is {d_P} at beta={beta}, P={P}"

    @pytest.mark.parametrize("P", HOLTSMARK_PS)
    def test_gradient_is_finite_at_zero_beta(self, P):
        """
        Regression test: both derivatives at beta == 0 used to be NaN.

        ``large_beta_result``, ``PR2`` and ``medium_beta_result`` all evaluate
        ``1.5/sqrt(beta) + 27/beta**2`` divided by ``beta**2``, which is +inf at
        beta == 0. The ``beta <= 25.12`` branch is the one selected there, but
        ``jnp.where`` masks only the values, so AD produced ``0 * inf == NaN``.
        beta == 0 is not hypothetical: ``bracket_line_interpolator`` builds its
        wavelength grid with ``np.linspace`` around the line centre and that grid
        contains the centre exactly, so every Brackett line hits it.
        """
        b, p = jnp.float64(0.0), jnp.float64(P)
        d_beta, d_P = jax.grad(holtsmark_profile, argnums=(0, 1))(b, p)
        assert np.isfinite(float(d_beta))
        assert np.isfinite(float(d_P))

    @pytest.mark.parametrize(
        "beta", [1e-8, 0.5, 3.0, 7.5, 9.0, 12.0, 20.0, 30.0, 100.0, 400.0, 800.0]
    )
    @pytest.mark.parametrize("P", [0.0, 0.37, 0.8])
    def test_d_dbeta_matches_finite_difference(self, beta, P):
        """Inside each regime d/dbeta must match a central difference."""
        assert_grad_matches_fd(
            lambda x: holtsmark_profile(jnp.float64(x), jnp.float64(P)),
            beta,
            step=max(1e-7, beta * 1e-7),
            name=f"holtsmark_profile d/dbeta (P={P})",
        )

    @pytest.mark.parametrize("beta", [0.5, 5.0, 12.0, 40.0, 300.0])
    @pytest.mark.parametrize("P", [0.05, 0.25, 0.5, 0.75])
    def test_d_dP_matches_finite_difference(self, beta, P):
        """
        d/dP must match a central difference away from the table index switches.

        P values are chosen strictly inside a ``floor(5P + 1)`` cell so the
        finite difference does not straddle a table-column change. Above
        beta = 500 the profile does not depend on P at all, so those betas are
        excluded here and covered by ``test_dP_vanishes_above_500`` instead.
        """
        assert_grad_matches_fd(
            lambda x: holtsmark_profile(jnp.float64(beta), jnp.float64(x)),
            P,
            step=1e-7,
            name=f"holtsmark_profile d/dP (beta={beta})",
        )

    @pytest.mark.parametrize("P", [0.0, 0.37, 0.8])
    def test_dP_vanishes_above_500(self, P):
        """
        For beta > 500 the profile is the pure asymptotic form, independent of P.

        The P-dependent pieces (the PROB7 interpolation and the C7/D7 medium
        correction) are still *evaluated* there, so a leaking cotangent would
        show up as a non-zero -- or NaN -- d/dP. It must be exactly zero.
        """
        d_P = jax.grad(holtsmark_profile, argnums=1)(jnp.float64(2000.0), jnp.float64(P))
        assert float(d_P) == 0.0, f"d/dP = {float(d_P)} for beta > 500, expected 0"

    @pytest.mark.parametrize("beta", [0.5, 3.0, 9.0, 30.0, 300.0, 2000.0])
    def test_profile_decreases_with_beta(self, beta):
        """The Holtsmark profile falls off monotonically, so d/dbeta < 0."""
        d_beta = jax.grad(holtsmark_profile, argnums=0)(
            jnp.float64(beta), jnp.float64(0.37)
        )
        assert float(d_beta) < 0.0, f"d/dbeta = {float(d_beta)} at beta={beta}"

    @pytest.mark.parametrize("boundary", [8.0, 10.0, 25.12, 500.0])
    @pytest.mark.parametrize("P", [0.0, 0.37, 0.8])
    def test_derivative_across_each_regime_boundary(self, boundary, P):
        """
        The derivative must be finite, negative and of the same magnitude on
        both sides of every regime switch.

        Kurucz's SOFBET is genuinely only piecewise-C0: the 8 and 10 knots are
        the ends of a linear blend between the PR1 and PR2 forms and the
        derivative there jumps by up to ~16%, while the 25.12 and 500 switches
        jump by a few percent or less. So the assertion is that the two sides
        agree to within a factor of 2 -- enough to catch a mis-selected branch
        or a leaked infinity, without pretending the fit is smooth.
        """
        f = lambda x: holtsmark_profile(jnp.float64(x), jnp.float64(P))
        below, above = one_sided_derivatives(f, boundary, 1e-7)
        assert np.isfinite(below) and np.isfinite(above)
        assert below < 0 and above < 0
        assert 0.5 < below / above < 2.0, (
            f"d/dbeta jumps from {below!r} to {above!r} across beta = {boundary}"
        )

    @pytest.mark.parametrize("beta", [1e-30, 1e-20, 1e-10, 1e-3])
    def test_small_beta_gradient_is_stable(self, beta):
        """
        The wing expressions are floored at beta = 1e-30 to keep beta**4 finite.

        The floor must not perturb the *selected* small-beta branch: d/dbeta
        there is dominated by the PROB7 interpolation and tends to a finite,
        non-zero constant as beta -> 0.
        """
        d_beta = float(
            jax.grad(holtsmark_profile, argnums=0)(jnp.float64(beta), jnp.float64(0.37))
        )
        assert np.isfinite(d_beta)
        assert d_beta != 0.0


# ---------------------------------------------------------------------------
# brackett_oscillator_strength and greim_1960_Knm
# ---------------------------------------------------------------------------


class TestBrackettOscillatorStrengthGradients:
    """Gradients of the Peterson & Kurucz oscillator strength fit."""

    @pytest.mark.parametrize("m", [5.0, 6.0, 7.0, 10.0, 15.0, 20.0, 30.0])
    def test_gradient_wrt_m_matches_finite_difference(self, m):
        """d f_nm / dm must be finite and match a central difference."""
        assert_grad_matches_fd(
            lambda x: brackett_oscillator_strength(4, x),
            m,
            step=1e-6,
            name="brackett_oscillator_strength d/dm",
        )

    @pytest.mark.parametrize("n", [2.0, 3.0, 4.0, 5.0])
    def test_gradient_wrt_n_matches_finite_difference(self, n):
        """
        d f_nm / dn must be finite and match a central difference.

        ``n`` is a fixed integer (4) in every caller, but it appears as
        ``n**0.71``, ``2.4/n**3`` and ``(m - n)**1.2``, all of which are NaN or
        singular for n <= 0, so differentiating in n is a cheap check that no
        such expression has been written in a way that blows up.
        """
        assert_grad_matches_fd(
            lambda x: brackett_oscillator_strength(x, 12.0),
            n,
            step=1e-6,
            name="brackett_oscillator_strength d/dn",
        )

    @pytest.mark.parametrize("m", [5.0, 8.0, 20.0])
    def test_gradient_wrt_m_is_negative(self, m):
        """Brackett oscillator strengths fall off with upper level, so df/dm < 0."""
        g = float(jax.grad(lambda x: brackett_oscillator_strength(4, x))(np.float64(m)))
        assert g < 0.0, f"df/dm = {g} at m={m}"

    def test_gradient_finite_just_above_the_series_head(self):
        """
        ``(m - n)**1.2`` is NaN for m < n and 0 with an infinite derivative at
        m == n, so the m -> n+ limit is the dangerous one. Values just above the
        head of the series must still differentiate cleanly.
        """
        for m in (4.5, 4.1, 4.01, 4.001):
            g = float(
                jax.grad(lambda x: brackett_oscillator_strength(4, x))(np.float64(m))
            )
            assert np.isfinite(g), f"df/dm is {g} at m={m}"


class TestGriemKnmGradients:
    """Gradients of the Griem 1960 K_nm constants."""

    def test_alias_is_the_same_function(self):
        """``griem_1960_Knm`` is an alias for the (misspelled) ``greim_1960_Knm``."""
        assert griem_1960_Knm is greim_1960_Knm

    @pytest.mark.parametrize("m", [9.0, 12.0, 20.0, 30.0])
    def test_analytical_branch_gradient_matches_finite_difference(self, m):
        """
        For ``m - n > 3`` the Griem eqn 33 formula is used and is differentiable.

        The formula has a pole at ``m == n`` (from ``m**2 - n**2``) and another
        at ``m - n == -0.13`` (from the Kurucz ``1 + 0.13/(m - n)`` factor), both
        far outside the physical domain, but they are what makes this worth
        differentiating at all.
        """
        assert_grad_matches_fd(
            lambda x: greim_1960_Knm(4.0, x),
            m,
            step=1e-6,
            name="greim_1960_Knm d/dm",
        )

    @pytest.mark.parametrize("m", [9.0, 20.0])
    def test_analytical_branch_gradient_wrt_n(self, m):
        """d K_nm / dn on the analytical branch must be finite and match FD."""
        assert_grad_matches_fd(
            lambda x: greim_1960_Knm(x, m),
            5.0,
            step=1e-6,
            name="greim_1960_Knm d/dn",
        )

    @pytest.mark.parametrize("n,m", [(2, 3), (2, 5), (3, 4), (4, 5), (4, 6)])
    def test_tabulated_branch_returns_a_plain_constant(self, n, m):
        """
        For ``m - n <= 3`` and ``n <= 4`` the value comes from a lookup table.

        The Python-level ``if`` means this branch is not a differentiable
        function of ``n``/``m`` at all -- it returns a plain float. That is
        deliberate (n and m are always concrete integers), and this test pins it
        so nobody mistakes the constant for a silently-zeroed derivative.
        """
        value = greim_1960_Knm(n, m)
        assert isinstance(value, float)
        assert np.isfinite(value) and value > 0


# ---------------------------------------------------------------------------
# hummer_mihalas_w
# ---------------------------------------------------------------------------

# Solar photosphere-ish reference conditions.
HM_T = 5778.0
HM_NH = 1.8e17
HM_NHE = 1.6e16
HM_NE = 1.5e14

# Guard conditions, table switches and physical extremes. The (ne > 10) & (T > 10)
# guard on the Hubeny branch and the (n_eff > 3) switch on the QM correction K
# are the interesting discrete boundaries; the rest span cool dwarf atmospheres
# through hot star atmospheres.
HM_TEMPERATURES = [9.999, 10.0, 10.001, 2000.0, 3500.0, 5778.0, 1e4, 4e4, 1e5]
HM_ELECTRON_DENSITIES = [1e-5, 1.0, 9.999, 10.0, 10.001, 1e6, 1e10, 1.5e14, 1e18]
HM_N_EFFS = [1.0, 2.0, 2.999, 3.0, 3.001, 5.0, 10.0, 20.0, 40.0]


class TestHummerMihalasWGradients:
    """Gradients of the Hummer & Mihalas 1988 occupation probability."""

    @pytest.mark.parametrize("T", HM_TEMPERATURES)
    @pytest.mark.parametrize("ne", HM_ELECTRON_DENSITIES)
    @pytest.mark.parametrize("use_hubeny", [False, True])
    def test_gradients_finite_over_the_physical_grid(self, T, ne, use_hubeny):
        """
        All five partial derivatives must be finite across the physical (T, ne)
        plane, with and without the Hubeny generalisation.

        This is the sweep that answers the standing question about the
        unconditionally-evaluated Hubeny block: at every physically plausible
        point, with ``use_hubeny_generalization`` either way, nothing leaks.
        """
        grads = jax.grad(hummer_mihalas_w, argnums=(0, 1, 2, 3, 4))(
            jnp.float64(T),
            jnp.float64(10.0),
            jnp.float64(HM_NH),
            jnp.float64(HM_NHE),
            jnp.float64(ne),
            use_hubeny,
        )
        assert_all_finite(grads, f"hummer_mihalas_w(T={T}, ne={ne}, hubeny={use_hubeny})")

    @pytest.mark.parametrize("n_eff", HM_N_EFFS)
    @pytest.mark.parametrize("use_hubeny", [False, True])
    def test_gradients_finite_across_the_K_switch(self, n_eff, use_hubeny):
        """
        The QM correction ``K`` switches form at ``n_eff > 3``.

        ``BETAC`` in the Hubeny branch goes as ``K / n_eff**4``, so this is also
        the axis along which the Hubeny term becomes extreme.
        """
        grads = jax.grad(hummer_mihalas_w, argnums=(0, 1, 2, 3, 4))(
            jnp.float64(HM_T),
            jnp.float64(n_eff),
            jnp.float64(HM_NH),
            jnp.float64(HM_NHE),
            jnp.float64(HM_NE),
            use_hubeny,
        )
        assert_all_finite(grads, f"hummer_mihalas_w(n_eff={n_eff}, hubeny={use_hubeny})")

    @pytest.mark.parametrize(
        "arg,index,step",
        [
            ("T", 0, 1e-3),
            ("n_eff", 1, 1e-7),
            ("nH", 2, 1e10),
            ("nHe", 3, 1e9),
            ("ne", 4, 1e7),
        ],
    )
    def test_gradients_match_finite_difference(self, arg, index, step):
        """Every partial derivative must agree with a central difference."""
        base = [HM_T, 10.0, HM_NH, HM_NHE, HM_NE]

        def f(x):
            args = list(base)
            args[index] = x
            return hummer_mihalas_w(*[jnp.float64(a) for a in args])

        assert_grad_matches_fd(f, base[index], step=step, name=f"hummer_mihalas_w d/d{arg}")

    @pytest.mark.parametrize("n_eff", [2.0, 5.0, 10.0, 20.0])
    def test_gradients_have_the_right_signs(self, n_eff):
        """
        Adding perturbers can only *lower* the occupation probability.

        w = exp(-4pi/3 (neutral_term + charged_term)) with both terms strictly
        positive and increasing in nH, nHe and ne, so dw/dnH, dw/dnHe and dw/dne
        must all be negative; dw/dn_eff is negative too because the level radius
        grows as n_eff**2.
        """
        _, d_neff, d_nH, d_nHe, d_ne = jax.grad(hummer_mihalas_w, argnums=(0, 1, 2, 3, 4))(
            jnp.float64(HM_T),
            jnp.float64(n_eff),
            jnp.float64(HM_NH),
            jnp.float64(HM_NHE),
            jnp.float64(HM_NE),
        )
        assert float(d_nH) < 0.0
        assert float(d_nHe) < 0.0
        assert float(d_ne) < 0.0
        assert float(d_neff) < 0.0

    def test_default_branch_has_no_temperature_dependence(self):
        """
        Without the Hubeny generalisation, w does not depend on T at all.

        The whole Hubeny block -- which *is* a function of T -- is nonetheless
        evaluated. If its cotangent leaked, dw/dT would come back non-zero or
        NaN instead of an exact zero. This is the sharpest available probe of
        the dead-branch hazard.
        """
        d_T = jax.grad(hummer_mihalas_w, argnums=0)(
            jnp.float64(HM_T),
            jnp.float64(10.0),
            jnp.float64(HM_NH),
            jnp.float64(HM_NHE),
            jnp.float64(HM_NE),
        )
        assert float(d_T) == 0.0, f"dw/dT = {float(d_T)} with the H&M branch selected"

    @pytest.mark.parametrize(
        "T,ne,n_eff",
        [
            # ne <= 0: log(ne) is -inf (or NaN), so A, BETAC and F are all poisoned.
            (5778.0, 0.0, 10.0),
            (5778.0, -1.5e14, 10.0),
            # T <= 0: 1/sqrt(T) in A is inf (or NaN).
            (0.0, 1.5e14, 10.0),
            (-100.0, 1.5e14, 10.0),
            # ne far below anything physical: BETAC**3 overflows and F is inf/inf.
            (5778.0, 1e-300, 10.0),
            (5778.0, 1e-160, 10.0),
            # ne (or n_eff) far above anything physical: BETAC**3 underflows and
            # log(F) is -inf.
            (5778.0, 1e300, 10.0),
            (5778.0, 1.5e14, 1e30),
            # Right on the guard, where the branch flips.
            (10.0, 10.0, 10.0),
            (10.000001, 10.000001, 3.0),
        ],
    )
    def test_default_branch_immune_to_pathological_hubeny_inputs(self, T, ne, n_eff):
        """
        Regression test for the unconditionally-evaluated Hubeny+ 1994 branch.

        ``hummer_mihalas_w`` computes ``A``, ``X``, ``BETAC``, ``F`` and
        ``log(F/(1+F))`` before discarding them with
        ``jnp.where(use_hubeny_generalization, hubeny_term, hm_term)``. Because
        ``jnp.where`` does not mask cotangents, any non-finite intermediate in
        that block used to poison dw/dT, dw/dn_eff and dw/dne even with
        ``use_hubeny_generalization=False`` -- i.e. on the path every synthesis
        takes. These inputs are all *un*physical; the point is precisely that
        the default branch must not care what the dead branch was handed.
        """
        grads = jax.grad(hummer_mihalas_w, argnums=(0, 1, 2, 3, 4))(
            jnp.float64(T),
            jnp.float64(n_eff),
            jnp.float64(HM_NH),
            jnp.float64(HM_NHE),
            jnp.float64(ne),
            False,
        )
        assert_all_finite(grads, f"hummer_mihalas_w dead-branch leak (T={T}, ne={ne})")

    @pytest.mark.parametrize("ne", [0.0, 1e-300, 1e300])
    def test_hubeny_branch_immune_to_its_own_guard_being_false(self, ne):
        """
        With ``use_hubeny_generalization=True`` but ``ne <= 10``, the Hubeny term
        is masked to 0 by an inner ``jnp.where``.

        That inner mask has exactly the same cotangent problem as the outer one,
        so these inputs must also give finite gradients.
        """
        grads = jax.grad(hummer_mihalas_w, argnums=(0, 1, 2, 3, 4))(
            jnp.float64(5.0),
            jnp.float64(10.0),
            jnp.float64(HM_NH),
            jnp.float64(HM_NHE),
            jnp.float64(ne),
            True,
        )
        assert_all_finite(grads, f"hummer_mihalas_w hubeny guard (ne={ne})")

    def test_hubeny_branch_gradient_matches_finite_difference(self):
        """
        When the Hubeny branch really is live it must still be correct.

        Hardening a dead branch is only safe if it leaves the live branch alone,
        so this checks dw/dne and dw/dT against a central difference well inside
        the ``(ne > 10) & (T > 10)`` guard.
        """
        for index, x, step in ((0, HM_T, 1e-2), (4, HM_NE, 1e7)):
            base = [HM_T, 10.0, HM_NH, HM_NHE, HM_NE]

            def f(v, index=index, base=base):
                args = list(base)
                args[index] = v
                return hummer_mihalas_w(*[jnp.float64(a) for a in args], True)

            assert_grad_matches_fd(
                f, x, step=step, name=f"hummer_mihalas_w hubeny d/d[{index}]"
            )

    def test_vmapped_and_jitted_batch_gradients_are_finite(self):
        """
        ``precompute_hummer_ws`` vmaps the above over layers and n = 1..20.

        This is how synthesis actually calls it, so the batched, JIT-compiled
        path gets its own check. n = 1..20 spans the ``n_eff > 3`` switch and
        reaches the levels where w underflows to 0 (dw/dx = 0 there, which is
        correct, so only finiteness is asserted for the sum).
        """
        T = jnp.array([3500.0, 5778.0, 8000.0])
        nH = jnp.array([1e16, 1.8e17, 5e17])
        nHe = jnp.array([1e15, 1.6e16, 4e16])
        ne = jnp.array([1e11, 1.5e14, 1e15])

        grads = jax.grad(
            lambda t, a, b, c: jnp.sum(precompute_hummer_ws(t, a, b, c)),
            argnums=(0, 1, 2, 3),
        )(T, nH, nHe, ne)
        assert_all_finite(grads, "precompute_hummer_ws")
        # nH and ne genuinely move the answer for the lower levels.
        assert np.any(np.asarray(grads[1]) != 0.0)
        assert np.any(np.asarray(grads[3]) != 0.0)


# ---------------------------------------------------------------------------
# _interp_linear_2d_jax / _interp_linear_3d_jax
# ---------------------------------------------------------------------------


@requires_stark_tables
class TestStarkTableInterpolatorGradients:
    """
    Gradients of the bilinear/trilinear interpolators over the Stehle tables.

    Both interpolators clamp the *cell index* but not the interpolation weight,
    so they extrapolate linearly outside the grid. That is a deliberate match to
    Julia's flat-index behaviour; the tests below check that the extrapolated
    derivative is the edge-cell slope rather than a NaN or a silent zero.
    """

    @staticmethod
    def _line():
        return hline_stark_profiles["2_3"]  # H-alpha

    @pytest.mark.parametrize(
        "where_", ["low-corner", "high-corner", "interior", "below-grid", "above-grid"]
    )
    def test_2d_gradients_are_finite(self, where_):
        """d lambda0 / dT and d lambda0 / dne must be finite at and off the edges."""
        line = self._line()
        T_grid, ne_grid = line.temps, line.electron_number_densities
        lo_T, hi_T = float(T_grid[0]), float(T_grid[-1])
        lo_ne, hi_ne = float(ne_grid[0]), float(ne_grid[-1])
        point = {
            "low-corner": (lo_T, lo_ne),
            "high-corner": (hi_T, hi_ne),
            "interior": (0.5 * (lo_T + hi_T), 0.5 * (lo_ne + hi_ne)),
            "below-grid": (0.5 * lo_T, 0.5 * lo_ne),
            "above-grid": (2.0 * hi_T, 3.0 * hi_ne),
        }[where_]
        grads = jax.grad(_interp_linear_2d_jax, argnums=(0, 1))(
            jnp.float64(point[0]), jnp.float64(point[1]), T_grid, ne_grid, line.lambda0_data
        )
        assert_all_finite(grads, f"_interp_linear_2d_jax at {where_}")

    @pytest.mark.parametrize(
        "where_", ["low-corner", "high-corner", "interior", "below-grid", "above-grid"]
    )
    def test_3d_gradients_are_finite(self, where_):
        """
        d log(profile) / d(T, ne, log dnu) must be finite at and off the edges.

        The log(delta_nu) axis begins with a -1e308 sentinel standing in for
        delta_nu == 0, so the first cell is ~1e308 wide; the interpolation weight
        there is (x + 1e308)/1e308, which is finite but whose derivative
        underflows to zero. That is the intended behaviour (the profile is flat
        as delta_nu -> 0), not a masked NaN, and it is pinned separately below.
        """
        line = self._line()
        T_grid = line.temps
        ne_grid = line.electron_number_densities
        d_grid = line.log_delta_nu_grid
        lo_T, hi_T = float(T_grid[0]), float(T_grid[-1])
        lo_ne, hi_ne = float(ne_grid[0]), float(ne_grid[-1])
        lo_d, hi_d = float(d_grid[1]), float(d_grid[-1])  # d_grid[0] is the sentinel
        point = {
            "low-corner": (lo_T, lo_ne, lo_d),
            "high-corner": (hi_T, hi_ne, hi_d),
            "interior": (0.5 * (lo_T + hi_T), 0.5 * (lo_ne + hi_ne), 0.5 * (lo_d + hi_d)),
            "below-grid": (0.9 * lo_T, 0.5 * lo_ne, lo_d - 5.0),
            "above-grid": (1.5 * hi_T, 3.0 * hi_ne, hi_d + 5.0),
        }[where_]
        grads = jax.grad(_interp_linear_3d_jax, argnums=(0, 1, 2))(
            jnp.float64(point[0]),
            jnp.float64(point[1]),
            jnp.float64(point[2]),
            T_grid,
            ne_grid,
            d_grid,
            line.profile_data,
        )
        assert_all_finite(grads, f"_interp_linear_3d_jax at {where_}")

    def test_3d_gradient_matches_finite_difference_in_each_axis(self):
        """
        Each trilinear partial derivative must match a central difference.

        The point is placed strictly inside a single cell in all three axes so
        the finite difference never straddles a cell boundary, where the
        piecewise-linear interpolant is only C0.
        """
        line = self._line()
        T_grid = np.asarray(line.temps)
        ne_grid = np.asarray(line.electron_number_densities)
        d_grid = np.asarray(line.log_delta_nu_grid)
        # Cell centres well inside the grid.
        T0 = 0.5 * (T_grid[2] + T_grid[3])
        ne0 = 0.5 * (ne_grid[6] + ne_grid[7])
        d0 = 0.5 * (d_grid[10] + d_grid[11])

        def make(index, values):
            def f(x):
                args = list(values)
                args[index] = x
                return _interp_linear_3d_jax(
                    jnp.float64(args[0]),
                    jnp.float64(args[1]),
                    jnp.float64(args[2]),
                    line.temps,
                    line.electron_number_densities,
                    line.log_delta_nu_grid,
                    line.profile_data,
                )

            return f

        base = [T0, ne0, d0]
        for index, step in ((0, 1.0), (1, 1e9), (2, 1e-4)):
            assert_grad_matches_fd(
                make(index, base),
                base[index],
                step=step,
                name=f"_interp_linear_3d_jax d/d[{index}]",
            )

    def test_3d_gradient_wrt_log_delta_nu_is_non_zero_and_negative(self):
        """
        The Stark profile falls off with detuning, so d log(profile)/d log(dnu) < 0.

        A clamped interpolator that silently zeroed the derivative would pass a
        pure finiteness check; this one would not.
        """
        line = self._line()
        d_grid = np.asarray(line.log_delta_nu_grid)
        for frac in (0.3, 0.5, 0.8):
            z = d_grid[1] + frac * (d_grid[-1] - d_grid[1])
            g = jax.grad(_interp_linear_3d_jax, argnums=2)(
                jnp.float64(6000.0),
                jnp.float64(1e14),
                jnp.float64(z),
                line.temps,
                line.electron_number_densities,
                line.log_delta_nu_grid,
                line.profile_data,
            )
            assert float(g) < 0.0, f"d log(profile)/d log(dnu) = {float(g)} at z={z}"

    def test_3d_gradient_in_the_sentinel_cell_is_finite_and_flat(self):
        """
        Inside the ~1e308-wide sentinel cell the log(dnu) derivative underflows to 0.

        That is the correct answer (the tabulated profile is constant across the
        cell to within 1e-308 per unit), and the point of the test is that it is
        a clean zero rather than a NaN produced by the huge subtraction.
        """
        line = self._line()
        z = float(line.log_delta_nu_grid[1]) - 1.0  # strictly inside the sentinel cell
        gT, gne, gz = jax.grad(_interp_linear_3d_jax, argnums=(0, 1, 2))(
            jnp.float64(6000.0),
            jnp.float64(1e14),
            jnp.float64(z),
            line.temps,
            line.electron_number_densities,
            line.log_delta_nu_grid,
            line.profile_data,
        )
        assert np.isfinite(float(gT)) and np.isfinite(float(gne))
        assert float(gz) == 0.0

    def test_stark_profile_line_method_wrappers_are_differentiable(self):
        """
        ``StarkProfileLine.interpolate_lambda0_jax`` / ``interpolate_profile_jax``
        are the public entry points into the two interpolators.

        They just forward to the module-level functions, but they are what
        ``prepare_stark_profiles_for_jit`` calls, so they get their own check.
        """
        line = self._line()
        grads = jax.grad(lambda T, ne: line.interpolate_lambda0_jax(T, ne), argnums=(0, 1))(
            jnp.float64(6000.0), jnp.float64(1e14)
        )
        assert_all_finite(grads, "interpolate_lambda0_jax")

        grads = jax.grad(
            lambda T, ne, z: line.interpolate_profile_jax(T, ne, z), argnums=(0, 1, 2)
        )(jnp.float64(6000.0), jnp.float64(1e14), jnp.float64(25.0))
        assert_all_finite(grads, "interpolate_profile_jax")
        assert float(grads[2]) < 0.0


# ---------------------------------------------------------------------------
# normal_pdf and autodiffable_conv
# ---------------------------------------------------------------------------


class TestDopplerHelperGradients:
    """Gradients of the two small helpers used to build the convolved profiles."""

    @pytest.mark.parametrize("delta", [0.0, 1e-10, 1e-9, 5e-9])
    def test_normal_pdf_gradients_match_finite_difference(self, delta):
        """d/d(delta) and d/d(sigma) of the Gaussian must match a central difference."""
        sigma = 3e-9
        assert_grad_matches_fd(
            lambda x: normal_pdf(jnp.float64(x), jnp.float64(sigma)),
            delta,
            step=1e-13,
            # At delta == 0 the PDF is stationary in delta, so a purely relative
            # comparison would compare two pieces of roundoff. The PDF is ~1e8
            # and the step 1e-13, so the finite-difference noise floor is
            # ~eps*1e8/1e-13 ~ 2e11; 1e12 is a safe absolute floor.
            atol=1e12 if delta == 0.0 else 0.0,
            name="normal_pdf d/d(delta)",
        )
        assert_grad_matches_fd(
            lambda x: normal_pdf(jnp.float64(delta), jnp.float64(x)),
            sigma,
            step=1e-15,
            name="normal_pdf d/d(sigma)",
        )

    def test_autodiffable_conv_jacobian_is_finite_and_exact(self):
        """
        The convolution is linear, so its Jacobian is exactly the other operand.

        A finite, correct Jacobian here is what makes the Doppler convolution in
        ``bracket_line_interpolator`` differentiable at all.
        """
        f = jnp.array([1.0, 2.0, 3.0, 4.0])
        g = jnp.array([0.5, 1.0, 0.25])
        jac = jax.jacfwd(lambda a: autodiffable_conv(a, g))(f)
        assert np.all(np.isfinite(np.asarray(jac)))
        # d(conv)_i / df_j = g_{i-j}
        expected = np.zeros((len(f) + len(g) - 1, len(f)))
        for j in range(len(f)):
            expected[j : j + len(g), j] = np.asarray(g)
        np.testing.assert_allclose(np.asarray(jac), expected, rtol=1e-14, atol=1e-15)


# ---------------------------------------------------------------------------
# brackett_line_stark_profiles
# ---------------------------------------------------------------------------

# (T, ne) sampling the Brackett-relevant part of parameter space: cool giant
# through hot dwarf photospheres.
BRACKETT_CONDITIONS = [(3000.0, 1e10), (5778.0, 1e13), (8000.0, 1e12), (1e4, 1e16)]


class TestBrackettStarkProfileGradients:
    """Gradients of the Griem/Kurucz Brackett Stark profile."""

    @staticmethod
    def _profile_sum(m, wavelengths, lambda0):
        def f(T, ne):
            impact, quasistatic = brackett_line_stark_profiles(
                m, wavelengths, lambda0, T, ne
            )
            return jnp.sum(impact) + jnp.sum(quasistatic)

        return f

    @pytest.mark.parametrize("m", [5, 7, 10, 15])
    @pytest.mark.parametrize("T,ne", BRACKETT_CONDITIONS)
    def test_gradients_wrt_T_and_ne_are_finite_off_centre(self, m, T, ne):
        """Baseline: away from the line centre nothing should be pathological."""
        lambda0 = brackett_line_center(4, m)
        wavelengths = jnp.asarray(
            lambda0 + np.linspace(-3e-7, 3e-7, 41) + 1.7e-9  # deliberately off-centre
        )
        grads = jax.grad(self._profile_sum(m, wavelengths, lambda0), argnums=(0, 1))(
            jnp.float64(T), jnp.float64(ne)
        )
        assert_all_finite(grads, f"brackett_line_stark_profiles (m={m}, T={T}, ne={ne})")

    @pytest.mark.parametrize("m", [5, 7, 10, 15])
    @pytest.mark.parametrize("T,ne", BRACKETT_CONDITIONS)
    def test_gradients_finite_with_the_line_centre_on_the_grid(self, m, T, ne):
        """
        Regression test: the line centre is on the grid, and it used to NaN.

        ``bracket_line_interpolator`` samples the profile with
        ``np.linspace(lambda0 - w, lambda0 + w, 201)``, whose middle point *is*
        lambda0 exactly, so ``beta == 0``, ``y1 == 0`` and ``y2 == 0`` all occur
        on every real call. Three separate expressions were infinite there --
        the Holtsmark wings, the E1 branches, and the live ``sqrt(y1)`` in the
        quasistatic electron correction -- and every one of them turned d/dT and
        d/dne into NaN.
        """
        lambda0 = brackett_line_center(4, m)
        wavelengths = jnp.asarray(np.linspace(lambda0 - 3e-7, lambda0 + 3e-7, 41))
        assert np.any(np.asarray(wavelengths) == lambda0), "grid must contain the centre"

        grads = jax.grad(self._profile_sum(m, wavelengths, lambda0), argnums=(0, 1))(
            jnp.float64(T), jnp.float64(ne)
        )
        assert_all_finite(grads, f"brackett centre (m={m}, T={T}, ne={ne})")

    @pytest.mark.parametrize("T,ne", BRACKETT_CONDITIONS)
    def test_centre_gradient_is_continuous_with_its_neighbourhood(self, T, ne):
        """
        The repaired derivative at the centre must join up with the limit.

        Evaluating the single-wavelength profile at lambda0 and at
        lambda0 + 1e-15 cm must give essentially the same d/dT: if the fix had
        merely replaced NaN with some arbitrary number, this would show it.
        A 1e-15 cm offset is a vanishing fraction of even the narrowest Stark
        width sampled here, so the two must agree to ~1e-5 relative. Larger
        offsets are deliberately not used: at 1e-12 cm the coolest, thinnest
        case (T = 3000 K, ne = 1e10) really has moved by more than 1%, so
        agreement there would say nothing about the repair.
        """
        m = 7
        lambda0 = brackett_line_center(4, m)
        values = []
        for offset in (0.0, 1e-15, 1e-14):
            wavelengths = jnp.asarray(np.array([lambda0 + offset]))
            values.append(
                float(
                    jax.grad(self._profile_sum(m, wavelengths, lambda0), argnums=0)(
                        jnp.float64(T), jnp.float64(ne)
                    )
                )
            )
        assert all(np.isfinite(values)), values
        np.testing.assert_allclose(values[0], values[1], rtol=1e-5)
        np.testing.assert_allclose(values[0], values[2], rtol=1e-5)

    @pytest.mark.parametrize("T,ne", BRACKETT_CONDITIONS)
    def test_gradients_wrt_T_and_ne_match_finite_difference(self, T, ne):
        """
        d/dT and d/dne of the summed profile must match a central difference.

        The grid deliberately contains the line centre, so this checks the
        repaired code path and not just a benign wing.
        """
        m = 7
        lambda0 = brackett_line_center(4, m)
        wavelengths = jnp.asarray(np.linspace(lambda0 - 3e-7, lambda0 + 3e-7, 41))
        f = self._profile_sum(m, wavelengths, lambda0)
        assert_grad_matches_fd(
            lambda x: f(jnp.float64(x), jnp.float64(ne)),
            T,
            step=T * 1e-7,
            name=f"brackett d/dT (T={T}, ne={ne})",
        )
        assert_grad_matches_fd(
            lambda x: f(jnp.float64(T), jnp.float64(x)),
            ne,
            step=ne * 1e-7,
            name=f"brackett d/dne (T={T}, ne={ne})",
        )

    def test_jacobian_wrt_wavelength_is_finite_including_at_the_centre(self):
        """
        The profile must also be differentiable with respect to wavelength.

        ``betas`` is built from ``jnp.abs(wavelengths - lambda0)``, whose
        derivative JAX defines as 0 at the cusp; the requirement here is only
        that it be finite, since the profile genuinely has a cusp at the centre.
        """
        m = 7
        lambda0 = brackett_line_center(4, m)
        wavelengths = jnp.asarray(np.linspace(lambda0 - 3e-7, lambda0 + 3e-7, 21))

        def f(wl):
            impact, quasistatic = brackett_line_stark_profiles(
                m, wl, lambda0, jnp.float64(5778.0), jnp.float64(1e13)
            )
            return jnp.sum(impact) + jnp.sum(quasistatic)

        jac = jax.jacfwd(f)(wavelengths)
        assert_all_finite(jac, "brackett d/d(wavelength)")

    @pytest.mark.parametrize("m", [5, 10])
    def test_profile_broadens_with_electron_density(self, m):
        """
        Stark broadening grows with ne, so the far wing must strengthen with ne.

        A gradient that is finite but identically zero would mean the electron
        density had been quietly dropped from the profile.
        """
        lambda0 = brackett_line_center(4, m)
        wavelengths = jnp.asarray(np.array([lambda0 + 5e-7]))
        d_ne = jax.grad(self._profile_sum(m, wavelengths, lambda0), argnums=1)(
            jnp.float64(5778.0), jnp.float64(1e13)
        )
        assert np.isfinite(float(d_ne))
        assert float(d_ne) > 0.0, f"d(wing)/dne = {float(d_ne)}, expected positive"


# ---------------------------------------------------------------------------
# hydrogen_line_absorption_core and _process_one_stehle_line
# ---------------------------------------------------------------------------


@requires_stark_tables
class TestStehleLineGradients:
    """Gradients of the Stehle & Hutcheon Stark line evaluation."""

    @staticmethod
    def _hydrogen_alpha_setup():
        """Return (wavelengths, line, ABO params) for H-alpha."""
        line = hline_stark_profiles["2_3"]
        lambda0_abo, sigma_a0, alpha_abo = _BALMER_ABO_PARAMS[3]
        wavelengths = jnp.asarray(np.linspace(6555e-8, 6575e-8, 41))
        return wavelengths, line, (lambda0_abo, sigma_a0 * bohr_radius_cgs**2, alpha_abo)

    def _single_line(self, T, ne, nH_I, UH_I, xi):
        wavelengths, line, (lambda0_abo, sigma_abo, alpha_abo) = self._hydrogen_alpha_setup()
        ws = jnp.full(20, 0.9)
        return jnp.sum(
            _process_one_stehle_line(
                wavelengths,
                T,
                ne,
                nH_I,
                UH_I,
                jnp.float64(1.5e-6),
                xi,
                ws,
                2,
                3,
                jnp.float64(line.log_gf),
                jnp.float64(6.5647e-5),
                jnp.float64(lambda0_abo),
                jnp.float64(sigma_abo),
                jnp.float64(alpha_abo),
                jnp.float64(1.0),
                line.temps,
                line.electron_number_densities,
                line.log_delta_nu_grid,
                line.profile_data,
            )
        )

    @pytest.mark.parametrize(
        "T,ne", [(4000.0, 1e12), (5778.0, 1e13), (7500.0, 1e14), (1e4, 1e15)]
    )
    def test_single_line_gradients_are_finite(self, T, ne):
        """
        d/d(T, ne, nH_I, UH_I, xi) of one Stehle line must be finite.

        The line centre falls inside the wavelength window, so the
        ``max(scaled_delta_nu, tiny)`` clamp before ``log`` is exercised.
        """
        grads = jax.grad(self._single_line, argnums=(0, 1, 2, 3, 4))(
            jnp.float64(T),
            jnp.float64(ne),
            jnp.float64(1e17),
            jnp.float64(2.0),
            jnp.float64(1e5),
        )
        assert_all_finite(grads, f"_process_one_stehle_line (T={T}, ne={ne})")

    def test_single_line_gradients_are_non_zero(self):
        """Every one of the five inputs must actually move the answer."""
        grads = jax.grad(self._single_line, argnums=(0, 1, 2, 3, 4))(
            jnp.float64(5778.0),
            jnp.float64(1.3e13),
            jnp.float64(1e17),
            jnp.float64(2.0),
            jnp.float64(1e5),
        )
        for name, g in zip(("T", "ne", "nH_I", "UH_I", "xi"), grads):
            assert float(g) != 0.0, f"d(alpha)/d{name} is exactly zero"

    @pytest.mark.parametrize(
        "index,name,value,step",
        [
            (0, "T", 5778.0, 1e-3),
            (1, "ne", 1.3e13, 1e6),
            (2, "nH_I", 1e17, 1e10),
            (3, "UH_I", 2.0, 1e-7),
            (4, "xi", 1e5, 1e-2),
        ],
    )
    def test_single_line_gradients_match_finite_difference(self, index, name, value, step):
        """
        Each partial derivative must agree with a central difference.

        ne is 1.3e13 rather than 1e13 because 1e13 is a node of the tabulated
        electron-density grid: the trilinear interpolant is only C0 there, so a
        central difference would straddle two cells and average two different
        one-sided slopes. T = 5778 K sits strictly inside the [5000, 10000] cell.
        """
        base = [5778.0, 1.3e13, 1e17, 2.0, 1e5]

        def f(x):
            args = list(base)
            args[index] = x
            return self._single_line(*[jnp.float64(a) for a in args])

        assert_grad_matches_fd(
            f, value, step=step, name=f"_process_one_stehle_line d/d{name}"
        )

    def test_wavelength_jacobian_is_finite_at_the_line_centre(self):
        """
        The absorption must be differentiable in wavelength, centre included.

        ``scaled_delta_nu`` is clamped with ``jnp.maximum(..., tiny)`` before the
        ``log``; the clamp routes the cotangent to the constant, so the
        derivative at the exact centre is a finite zero rather than a NaN.
        """
        wavelengths, line, (lambda0_abo, sigma_abo, alpha_abo) = self._hydrogen_alpha_setup()
        wavelengths = jnp.asarray(
            np.concatenate([np.asarray(wavelengths), [lambda0_abo]])
        )

        def f(wl):
            ws = jnp.full(20, 0.9)
            return jnp.sum(
                _process_one_stehle_line(
                    wl,
                    jnp.float64(5778.0),
                    jnp.float64(1e13),
                    jnp.float64(1e17),
                    jnp.float64(2.0),
                    jnp.float64(1.5e-6),
                    jnp.float64(1e5),
                    ws,
                    2,
                    3,
                    jnp.float64(line.log_gf),
                    jnp.float64(6.5647e-5),
                    jnp.float64(lambda0_abo),
                    jnp.float64(sigma_abo),
                    jnp.float64(alpha_abo),
                    jnp.float64(1.0),
                    line.temps,
                    line.electron_number_densities,
                    line.log_delta_nu_grid,
                    line.profile_data,
                )
            )

        assert_all_finite(jax.jacfwd(f)(wavelengths), "_process_one_stehle_line d/dwl")

    def test_batched_all_layers_gradients_are_finite(self):
        """
        ``_process_stehle_line_all_layers_jit`` vmaps the above over layers.

        It also masks out-of-grid layers with ``jnp.where(valid_i, contrib, 0)``,
        which is exactly the construct that leaks cotangents, so an invalid layer
        is included in the batch on purpose.
        """
        wavelengths, line, (lambda0_abo, sigma_abo, alpha_abo) = self._hydrogen_alpha_setup()
        n_layers = 4
        T = jnp.array([1000.0, 4000.0, 5778.0, 9000.0])  # first layer is below the grid
        ne = jnp.array([1e8, 1e12, 1e13, 1e14])  # first layer is below the grid
        valid = jnp.array([False, True, True, True])

        def f(T_arr, ne_arr, nH_arr, UH_arr):
            return jnp.sum(
                _process_stehle_line_all_layers_jit(
                    wavelengths,
                    T_arr,
                    ne_arr,
                    nH_arr,
                    UH_arr,
                    jnp.full(n_layers, 6.5647e-5),
                    jnp.full(n_layers, lambda0_abo),
                    valid,
                    jnp.float64(1.5e-6),
                    jnp.float64(1e5),
                    jnp.full((n_layers, 20), 0.9),
                    2,
                    3,
                    jnp.float64(line.log_gf),
                    jnp.float64(sigma_abo),
                    jnp.float64(alpha_abo),
                    jnp.float64(1.0),
                    line.temps,
                    line.electron_number_densities,
                    line.log_delta_nu_grid,
                    line.profile_data,
                )
            )

        grads = jax.grad(f, argnums=(0, 1, 2, 3))(
            T,
            ne,
            jnp.full(n_layers, 1e17),
            jnp.full(n_layers, 2.0),
        )
        assert_all_finite(grads, "_process_stehle_line_all_layers_jit")
        # The masked layer must contribute exactly nothing.
        assert float(np.asarray(grads[0])[0]) == 0.0
        assert float(np.asarray(grads[2])[0]) == 0.0
        # The valid layers must contribute something.
        assert np.any(np.asarray(grads[2])[1:] != 0.0)

    @pytest.mark.parametrize(
        "T,ne", [(4000.0, 1e12), (5778.0, 1e13), (7500.0, 1e14)]
    )
    def test_core_driver_gradients_are_finite(self, T, ne):
        """
        ``hydrogen_line_absorption_core`` sums ~50 Stehle lines.

        It is the differentiable half of ``hydrogen_line_absorption``: the line
        selection is done outside it with concrete T/ne, and everything inside
        is traced. This is what a fit would differentiate through.
        """
        profile_data, valid = prepare_stark_profiles_for_jit(_BALMER_ONLY, T, ne)
        assert len(valid) > 0
        wavelengths = jnp.asarray(np.linspace(6540e-8, 6590e-8, 41))

        def f(T_, ne_, nH_I, UH_I, xi):
            ws = jax.vmap(
                lambda n_eff: hummer_mihalas_w(T_, n_eff, nH_I, jnp.float64(1.6e16), ne_)
            )(jnp.arange(1.0, 21.0))
            return jnp.sum(
                hydrogen_line_absorption_core(
                    wavelengths, T_, ne_, nH_I, UH_I, 1.5e-6, xi, ws, profile_data, len(valid)
                )
            )

        grads = jax.grad(f, argnums=(0, 1, 2, 3, 4))(
            jnp.float64(T),
            jnp.float64(ne),
            jnp.float64(1e17),
            jnp.float64(2.0),
            jnp.float64(1e5),
        )
        assert_all_finite(grads, f"hydrogen_line_absorption_core (T={T}, ne={ne})")
        for g in grads:
            assert float(g) != 0.0

    @pytest.mark.parametrize(
        "index,name,value,step",
        [
            (0, "T", 5778.0, 1e-3),
            (1, "ne", 1.3e13, 1e6),
            (2, "nH_I", 1e17, 1e10),
            (3, "UH_I", 2.0, 1e-7),
            (4, "xi", 1e5, 1e-2),
        ],
    )
    def test_core_driver_gradients_match_finite_difference(self, index, name, value, step):
        """
        Each partial derivative of the core driver must match a central difference.

        ``profile_data`` is built once, outside the differentiated function, so
        that perturbing T or ne does not change which lines are in bounds; a
        finite difference across such a change would be comparing two different
        sums.
        """
        profile_data, valid = prepare_stark_profiles_for_jit(_BALMER_ONLY, 5778.0, 1.3e13)
        wavelengths = jnp.asarray(np.linspace(6540e-8, 6590e-8, 41))
        base = [5778.0, 1.3e13, 1e17, 2.0, 1e5]

        def f(x):
            args = [jnp.float64(a) for a in base]
            args[index] = jnp.float64(x)
            T_, ne_, nH_I, UH_I, xi = args
            ws = jax.vmap(
                lambda n_eff: hummer_mihalas_w(T_, n_eff, nH_I, jnp.float64(1.6e16), ne_)
            )(jnp.arange(1.0, 21.0))
            return jnp.sum(
                hydrogen_line_absorption_core(
                    wavelengths, T_, ne_, nH_I, UH_I, 1.5e-6, xi, ws, profile_data, len(valid)
                )
            )

        assert_grad_matches_fd(
            f, value, step=step, name=f"hydrogen_line_absorption_core d/d{name}"
        )


# ---------------------------------------------------------------------------
# _h_alpha_from_precomp_jit (the fast synthesis path)
# ---------------------------------------------------------------------------


@requires_stark_tables
class TestPrecomputedHAlphaGradients:
    """
    Gradients of the precomputed-profile fast path used by ``synthesize``.

    ``_h_alpha_from_precomp_jit`` splits the 3-D Stark interpolation: the
    (T, ne) bilinear reduction is precomputed per layer, and only the 1-D
    interpolation in log(delta_nu) runs per call. It is a different code path
    from ``_process_one_stehle_line`` and needs its own coverage.
    """

    @staticmethod
    def _setup(T_arr, ne_arr):
        line = hline_stark_profiles["2_3"]
        n_delta = line.profile_data.shape[2]
        profiles_1d = jnp.stack(
            [
                jax.vmap(
                    lambda k: _interp_linear_2d_jax(
                        jnp.float64(T),
                        jnp.float64(ne),
                        line.temps,
                        line.electron_number_densities,
                        line.profile_data[:, :, k],
                    )
                )(jnp.arange(n_delta))
                for T, ne in zip(T_arr, ne_arr)
            ]
        )
        return line, profiles_1d

    def _alpha_sum(self, T_arr, nH_arr, UH_arr, line, profiles_1d, F0_arr, valid):
        lambda0_abo, sigma_a0, alpha_abo = _BALMER_ABO_PARAMS[3]
        n_layers = len(F0_arr)
        wavelengths = jnp.asarray(np.linspace(6558e-8, 6572e-8, 33))
        return jnp.sum(
            _h_alpha_from_precomp_jit(
                wavelengths,
                T_arr,
                nH_arr,
                UH_arr,
                jnp.full((n_layers, 20), 0.9),
                valid,
                profiles_1d,
                jnp.full(n_layers, 6.5647e-5),
                jnp.full(n_layers, lambda0_abo),
                F0_arr,
                line.log_delta_nu_grid,
                jnp.float64(1.5e-6),
                jnp.float64(line.log_gf),
                jnp.float64(sigma_a0 * bohr_radius_cgs**2),
                jnp.float64(alpha_abo),
                jnp.float64(1.0),
                jnp.float64(1e5),
                2,
                3,
            )
        )

    def test_gradients_are_finite_and_non_zero(self):
        """d/d(T, nH_I, UH_I) over all layers must be finite and actually non-zero."""
        T_arr = np.array([4000.0, 5778.0, 7000.0])
        ne_arr = np.array([1e12, 1e13, 1e14])
        line, profiles_1d = self._setup(T_arr, ne_arr)
        F0 = jnp.asarray(1.25e-9 * ne_arr ** (2 / 3))
        valid = jnp.array([True, True, True])

        grads = jax.grad(
            lambda T, nH, UH: self._alpha_sum(T, nH, UH, line, profiles_1d, F0, valid),
            argnums=(0, 1, 2),
        )(jnp.asarray(T_arr), jnp.full(3, 1e17), jnp.full(3, 2.0))
        assert_all_finite(grads, "_h_alpha_from_precomp_jit")
        for g in grads:
            assert np.all(np.asarray(g) != 0.0)

    def test_invalid_layers_contribute_a_hard_zero(self):
        """
        Out-of-grid layers are masked with ``jnp.where(valid_i, contrib, 0)``.

        The masked contribution is still computed, so a leak would show up as a
        NaN or a non-zero derivative for that layer.
        """
        T_arr = np.array([4000.0, 5778.0, 7000.0])
        ne_arr = np.array([1e12, 1e13, 1e14])
        line, profiles_1d = self._setup(T_arr, ne_arr)
        F0 = jnp.asarray(1.25e-9 * ne_arr ** (2 / 3))
        valid = jnp.array([False, True, False])

        grads = jax.grad(
            lambda T, nH, UH: self._alpha_sum(T, nH, UH, line, profiles_1d, F0, valid),
            argnums=(0, 1, 2),
        )(jnp.asarray(T_arr), jnp.full(3, 1e17), jnp.full(3, 2.0))
        assert_all_finite(grads, "_h_alpha_from_precomp_jit masked")
        for g in grads:
            arr = np.asarray(g)
            assert arr[0] == 0.0 and arr[2] == 0.0
            assert arr[1] != 0.0

    def test_gradient_wrt_temperature_matches_finite_difference(self):
        """
        d(alpha)/dT must match a central difference.

        ``profiles_1d`` is held fixed (it is precomputed outside the traced
        function in the real caller too), so this isolates the temperature
        dependence of the Boltzmann factor, the ABO Voigt profile and the
        Doppler width.
        """
        T_arr = np.array([4000.0, 5778.0, 7000.0])
        ne_arr = np.array([1e12, 1e13, 1e14])
        line, profiles_1d = self._setup(T_arr, ne_arr)
        F0 = jnp.asarray(1.25e-9 * ne_arr ** (2 / 3))
        valid = jnp.array([True, True, True])

        def f(x):
            T = jnp.asarray(T_arr).at[1].set(x)
            return self._alpha_sum(T, jnp.full(3, 1e17), jnp.full(3, 2.0), line, profiles_1d, F0, valid)

        assert_grad_matches_fd(
            lambda x: f(jnp.float64(x)),
            5778.0,
            step=1e-3,
            name="_h_alpha_from_precomp_jit d/dT",
        )
        # jax.grad returns the whole vector; pick out the perturbed layer.
        g = jax.grad(
            lambda T: self._alpha_sum(
                T, jnp.full(3, 1e17), jnp.full(3, 2.0), line, profiles_1d, F0, valid
            )
        )(jnp.asarray(T_arr))
        assert np.isfinite(float(np.asarray(g)[1]))


# ---------------------------------------------------------------------------
# hydrogen_line_absorption (top-level)
# ---------------------------------------------------------------------------


@requires_stark_tables
class TestHydrogenLineAbsorptionGradients:
    """Gradients of the top-level entry point."""

    WAVELENGTHS = np.linspace(6540e-8, 6590e-8, 41)

    def _alpha_sum(self, nH_I, nHe_I, UH_I, T=5778.0, ne=1e13, xi=1e5):
        return jnp.sum(
            hydrogen_line_absorption(
                self.WAVELENGTHS,
                T,
                ne,
                nH_I,
                nHe_I,
                UH_I,
                xi,
                1.5e-6,
                stark_profiles=_BALMER_ONLY,
            )
        )

    @pytest.mark.parametrize(
        "T,ne", [(4000.0, 1e12), (5778.0, 1e13), (7500.0, 1e14)]
    )
    def test_gradients_wrt_densities_are_finite(self, T, ne):
        """
        d/d(nH_I, nHe_I, UH_I) must be finite over the Balmer window.

        These are the three arguments that stay traced all the way through the
        top level (see ``test_temperature_is_not_traceable_at_the_top_level``).
        nHe_I only enters through the MHD occupation probabilities, so it is the
        one that exercises ``hummer_mihalas_w`` end to end.
        """
        grads = jax.grad(self._alpha_sum, argnums=(0, 1, 2))(
            jnp.float64(1e17), jnp.float64(1e16), jnp.float64(2.0), T=T, ne=ne
        )
        assert_all_finite(grads, f"hydrogen_line_absorption (T={T}, ne={ne})")
        for g in grads:
            assert float(g) != 0.0

    @pytest.mark.parametrize(
        "index,name,value,step",
        [(0, "nH_I", 1e17, 1e10), (1, "nHe_I", 1e16, 1e14), (2, "UH_I", 2.0, 1e-7)],
    )
    def test_gradients_match_finite_difference(self, index, name, value, step):
        """
        Each traceable partial derivative must agree with a central difference.

        The nHe_I step is a full 1% of nHe_I because neutral helium only enters
        through the MHD occupation probabilities, where it shifts alpha by
        ~1e-21 relative per cm^-3; a step small enough to be "safe" would leave
        the difference buried in float64 roundoff. w is exp(-c*nHe) with
        c*nHe ~ 0.02, so the truncation error of a 1% step is ~1e-8 relative --
        still three orders below the 1e-5 tolerance.
        """
        base = [1e17, 1e16, 2.0]

        def f(x):
            args = list(base)
            args[index] = x
            return self._alpha_sum(*[jnp.float64(a) for a in args])

        assert_grad_matches_fd(
            f, value, step=step, name=f"hydrogen_line_absorption d/d{name}"
        )

    def test_absorption_grows_with_neutral_hydrogen(self):
        """Balmer absorption is linear in nH_I, so d(alpha)/d(nH_I) > 0."""
        g = jax.grad(self._alpha_sum, argnums=0)(
            jnp.float64(1e17), jnp.float64(1e16), jnp.float64(2.0)
        )
        assert float(g) > 0.0

    def test_gradients_finite_in_a_brackett_window(self):
        """
        The Brackett branch is a completely different code path.

        It goes through ``brackett_line_stark_profiles`` and the Doppler
        convolution rather than the Stehle tables, so it needs its own check.
        Only nH_I/nHe_I/UH_I are traced (see below), but those are enough to
        exercise the amplitude and the MHD occupation probabilities.
        """
        wavelengths = np.linspace(21550e-8, 21750e-8, 41)  # Brackett-gamma
        grads = jax.grad(
            lambda a, b, c: jnp.sum(
                hydrogen_line_absorption(
                    wavelengths,
                    8000.0,
                    1e14,
                    a,
                    b,
                    c,
                    1e5,
                    3e-6,
                    stark_profiles=_BRACKETT_ENABLING,
                )
            ),
            argnums=(0, 1, 2),
        )(jnp.float64(1e17), jnp.float64(1e16), jnp.float64(2.0))
        assert_all_finite(grads, "hydrogen_line_absorption (Brackett)")
        assert float(grads[0]) != 0.0

    def test_temperature_is_not_traceable_at_the_top_level(self):
        """
        ``hydrogen_line_absorption`` cannot be differentiated in T or ne.

        ``prepare_stark_profiles_for_jit`` calls ``float(T)``/``float(ne)`` to
        decide, with concrete Python control flow, which tabulated lines are in
        bounds, and ``bracket_line_interpolator`` builds a concrete numpy
        wavelength grid. Both are deliberate (documented in the source), and the
        differentiable entry point for T and ne is
        ``hydrogen_line_absorption_core``, which is covered above. This test
        pins the limitation so it cannot regress silently into a *wrong* number
        instead of a loud error.
        """
        with pytest.raises(jax.errors.ConcretizationTypeError):
            jax.grad(
                lambda T: jnp.sum(
                    hydrogen_line_absorption(
                        self.WAVELENGTHS,
                        T,
                        1e13,
                        1e17,
                        1e16,
                        2.0,
                        1e5,
                        1.5e-6,
                        stark_profiles=_BALMER_ONLY,
                    )
                )
            )(jnp.float64(5778.0))
