"""
Tests for spherical radiative transfer.

Korg.jl solves the transfer equation along *rays* when the atmosphere is a
spherical shell rather than a plane-parallel slab.  Each surface direction
cosine μ selects a ray of impact parameter ``b = r[0] sqrt(1 - μ²)``; the
optical depth is integrated along that ray's own path-length coordinate
``s = sqrt(r² - b²)`` (so ``ds/dr = r/s``, not the plane-parallel ``1/μ``); rays
with ``b > r[-1]`` never reach the innermost shell and turn around at a tangent
point, where an inward ray seeds the intensity of the outward leg; and the
emergent flux is assembled from the rays.

This port previously accepted ``spherical=True`` and ignored it, returning
bitwise plane-parallel values *and* gradients.  These tests cover the four
things the package requires of any physics routine:

1. **Functional behaviour** -- the ray geometry, which layers a ray sees, and
   how the flux responds to sphericity.
2. **Agreement with Korg.jl** -- against fixtures in
   ``tests/julia_reference_data.json`` (regenerate with
   ``julia --project=. tests/generate_julia_reference.jl``).
3. **Autodiff** -- ``jax.grad`` wrt α, S and the radial coordinate must be
   finite, non-zero and match central differences.  The tangent point is a
   ``sqrt(0)`` waiting to happen: ``jnp.where(cond, safe, dangerous)`` masks a
   NaN value but not its cotangent, and clamping the radicand to zero is not
   enough because ``sqrt(0)`` has an infinite derivative and ``0 * inf`` is
   NaN.  The radicand is therefore floored at a strictly positive value.
4. **jit tracing** -- the number of layers a ray intersects is data dependent,
   so the geometry is carried as a fixed-size padded array plus a mask instead
   of Python control flow.
"""

import json
from pathlib import Path

# Import korg FIRST to enable JAX x64 mode before any other JAX operations
import korg  # noqa: F401 — side-effect: enables float64

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from korg.radiative_transfer.core import generate_mu_grid, radiative_transfer_spherical
from korg.radiative_transfer.rays import calculate_rays
from korg.radiative_transfer.spherical import spherical_ray_flux

REFERENCE_FILE = Path(__file__).parent / "julia_reference_data.json"

# Relative tolerance for the finite-difference comparisons.  A central
# difference is accurate to about (machine epsilon)^(2/3) ~ 4e-11 relative in
# the best case; 1e-5 leaves five orders of magnitude of headroom.
FD_RTOL = 1e-5


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def shell_atmosphere(thickness_over_radius=0.3, n_layers=25, R_inner=1.0e12):
    """
    A self-consistent spherical shell atmosphere.

    The anchored optical depth scheme assumes ``dr/d(ln τ_ref) = -τ_ref/α_ref``,
    so the radii are obtained by integrating that relation rather than by
    inventing an unrelated radial grid.  Scaling ``α_ref`` up shrinks the
    geometric thickness at fixed optical depth, which is how the thin limit is
    approached.  This mirrors the construction in
    ``tests/generate_julia_reference.jl`` exactly.

    Parameters
    ----------
    thickness_over_radius : float, optional
        Geometric thickness of the atmosphere divided by the innermost radius.
    n_layers : int, optional
        Number of layers.
    R_inner : float, optional
        Innermost radius [cm].

    Returns
    -------
    dict
        Keys ``radii``, ``log_tau_ref``, ``alpha_ref``, ``alpha`` (shape
        ``(2, n_layers)``) and ``S`` (shape ``(2, n_layers)``), all float64
        NumPy arrays.  ``radii`` is ordered outermost first, as Korg.jl orders
        model atmosphere layers.
    """
    tau_ref = 10.0 ** np.linspace(-5.0, 2.0, n_layers)
    log_tau_ref = np.log10(tau_ref)
    # The tau_ref**0.9 factor spreads the geometric thickness over the whole
    # optical depth range, as in a real extended model, instead of piling it
    # up in the deepest couple of layers.
    profile = (1.0 + 0.5 * np.sin(np.linspace(0.0, np.pi, n_layers))) * tau_ref ** 0.9

    g = tau_ref / profile
    dr = 0.5 * (g[:-1] + g[1:]) * np.diff(np.log(tau_ref))
    depth = np.concatenate([[0.0], np.cumsum(dr)])
    scale = depth[-1] / (thickness_over_radius * R_inner)

    alpha_ref = profile * scale
    depth = depth / scale
    radii = R_inner + depth[-1] - depth

    alpha = np.stack([
        alpha_ref * 1.5,
        alpha_ref * (1.0 + 0.8 * np.cos(np.linspace(0.0, 2.0 * np.pi, n_layers))),
    ])
    S_col = np.linspace(2.0e-5, 9.0e-5, n_layers)
    S = np.stack([S_col, S_col * 1.1])

    return dict(radii=radii, log_tau_ref=log_tau_ref, alpha_ref=alpha_ref,
                alpha=alpha, S=S)


def spherical_flux(atm, n_mu=20, **overrides):
    """
    Total emergent flux of *atm*, summed over wavelength, as a JAX scalar.

    Parameters
    ----------
    atm : dict
        Atmosphere from :func:`shell_atmosphere`.
    n_mu : int, optional
        Number of Gauss-Legendre μ points.
    **overrides
        Replacements for any of ``alpha``, ``S``, ``radii``, ``alpha_ref`` or
        ``log_tau_ref``.

    Returns
    -------
    jax.Array
        Scalar total flux.
    """
    fluxes, _ = radiative_transfer_spherical(
        overrides.get("alpha", atm["alpha"]),
        overrides.get("S", atm["S"]),
        overrides.get("radii", atm["radii"]),
        overrides.get("log_tau_ref", atm["log_tau_ref"]),
        overrides.get("alpha_ref", atm["alpha_ref"]),
        n_mu=n_mu,
    )
    return jnp.sum(fluxes)


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
        Perturbation direction, same shape as *x*.
    step : float
        Half-step along *direction*.

    Returns
    -------
    float
        Estimate of ``grad(f) . direction``.
    """
    x = np.asarray(x, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    plus = float(f(jnp.asarray(x + step * direction)))
    minus = float(f(jnp.asarray(x - step * direction)))
    return (plus - minus) / (2.0 * step)


def assert_array_grad_matches_fd(f, x, direction=None, step=1e-6, rtol=FD_RTOL,
                                 name=""):
    """
    Assert the AD gradient of an array-input, scalar-output ``f`` is correct.

    The comparison is directional rather than component by component, which
    keeps the number of function evaluations small while still exercising
    every component of the gradient.

    Parameters
    ----------
    f : callable
        Function mapping an array to a scalar.
    x : array_like
        Point at which to differentiate.
    direction : array_like, optional
        Perturbation direction.  Defaults to *x* itself (multiplicative),
        which is the only sane choice for quantities spanning many decades.
    step : float, optional
        Half-step along *direction*.
    rtol : float, optional
        Relative tolerance.
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
    assert abs(directional_analytic - numeric) <= rtol * scale, (
        f"{name}: directional AD derivative {directional_analytic!r} disagrees "
        f"with finite difference {numeric!r} (relative error "
        f"{abs(directional_analytic - numeric) / max(scale, 1e-300):.3e})"
    )
    return analytic


@pytest.fixture(scope="module")
def reference_data():
    """Load pre-computed Julia reference data."""
    if not REFERENCE_FILE.exists():
        pytest.skip(
            f"Julia reference data not found at {REFERENCE_FILE}. "
            "Run: julia --project=. tests/generate_julia_reference.jl"
        )
    with open(REFERENCE_FILE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def spherical_ref(reference_data):
    """The ``spherical_radiative_transfer`` fixture block."""
    if "spherical_radiative_transfer" not in reference_data:
        pytest.skip("spherical_radiative_transfer key missing from reference data")
    return reference_data["spherical_radiative_transfer"]


# ---------------------------------------------------------------------------
# calculate_rays: functional behaviour
# ---------------------------------------------------------------------------


class TestCalculateRaysGeometry:
    """The ray geometry itself, independent of any radiative transfer."""

    def test_shapes_and_mask_dtype(self):
        """Everything is (n_mu, n_layers) and the mask is boolean."""
        atm = shell_atmosphere()
        mu, _ = generate_mu_grid(7)
        path, dsdz, mask = calculate_rays(mu, atm["radii"])
        expected = (7, atm["radii"].size)
        assert path.shape == expected
        assert dsdz.shape == expected
        assert mask.shape == expected
        assert mask.dtype == bool

    def test_mask_is_a_prefix(self):
        """
        A ray enters at the top and stops: the layers it sees are contiguous.

        Korg.jl expresses this as ``1:lowest_layer_index``.  Because the radii
        decrease monotonically, ``r >= b`` is a leading run, and anything else
        would mean the padded representation had lost the ray's structure.
        """
        atm = shell_atmosphere()
        mu, _ = generate_mu_grid(20)
        _, _, mask = calculate_rays(mu, atm["radii"])
        mask = np.asarray(mask)
        for row in mask:
            n_true = int(row.sum())
            assert np.all(row[:n_true]), f"mask is not a prefix: {row}"
            assert not np.any(row[n_true:]), f"mask is not a prefix: {row}"

    def test_radial_ray_sees_every_layer(self):
        """μ = 1 gives b = 0, so the ray passes straight through the core."""
        atm = shell_atmosphere()
        path, dsdz, mask = calculate_rays(jnp.array([1.0]), atm["radii"])
        assert np.all(np.asarray(mask))
        # b = 0 means s = r and ds/dr = 1 exactly.
        np.testing.assert_allclose(np.asarray(path)[0], atm["radii"], rtol=1e-15)
        np.testing.assert_allclose(np.asarray(dsdz)[0], 1.0, rtol=1e-15)

    @pytest.mark.parametrize("thickness", [0.5, 0.3, 0.05])
    def test_tangent_rays_stop_where_the_radius_meets_the_impact_parameter(
            self, thickness):
        """The deepest intersected layer is the last one with ``r >= b``."""
        atm = shell_atmosphere(thickness_over_radius=thickness)
        mu, _ = generate_mu_grid(20)
        _, _, mask = calculate_rays(mu, atm["radii"])
        b = atm["radii"][0] * np.sqrt(1.0 - np.asarray(mu) ** 2)
        for j, row in enumerate(np.asarray(mask)):
            n_true = int(row.sum())
            assert n_true >= 1, "every ray must at least graze the surface layer"
            assert atm["radii"][n_true - 1] >= b[j]
            if n_true < atm["radii"].size:
                assert atm["radii"][n_true] < b[j]

    def test_grazing_rays_do_not_reach_the_core_but_radial_ones_do(self):
        """
        An extended atmosphere must produce both kinds of ray.

        If every ray reached the core the tangent-point machinery would never
        be exercised, and the test suite would pass without it.
        """
        atm = shell_atmosphere(thickness_over_radius=0.3)
        mu, _ = generate_mu_grid(20)
        _, _, mask = calculate_rays(mu, atm["radii"])
        reaches_core = np.asarray(mask)[:, -1]
        assert np.any(reaches_core), "no ray reaches the innermost shell"
        assert not np.all(reaches_core), "no tangent rays in an extended atmosphere"

    def test_path_length_is_monotonic_along_the_ray(self):
        """
        ``s`` decreases towards the tangent point, on padded layers too.

        The padded continuation is negative rather than clamped, which keeps
        the array strictly decreasing so the Bezier schemes' spacing never
        degenerates.
        """
        atm = shell_atmosphere()
        mu, _ = generate_mu_grid(20)
        path, _, _ = calculate_rays(mu, atm["radii"])
        assert np.all(np.diff(np.asarray(path), axis=1) < 0.0)

    def test_planar_mode_reproduces_the_plane_parallel_ray(self):
        """``spherical=False`` gives ``s = z/μ``, ``ds/dz = 1/μ``, no masking."""
        z = np.linspace(2.0e9, 0.0, 12)
        mu = np.array([0.2, 0.5, 1.0])
        path, dsdz, mask = calculate_rays(mu, z, spherical=False)
        np.testing.assert_allclose(np.asarray(path), z[None, :] / mu[:, None], rtol=1e-15)
        np.testing.assert_allclose(np.asarray(dsdz),
                                   np.broadcast_to(1.0 / mu[:, None], path.shape),
                                   rtol=1e-15)
        assert np.all(np.asarray(mask))

    def test_impact_parameter_scales_with_the_outermost_radius(self):
        """``b`` is built from ``r[0]``, so a uniform rescaling leaves ds/dr fixed."""
        atm = shell_atmosphere()
        mu, _ = generate_mu_grid(9)
        _, dsdz_a, mask_a = calculate_rays(mu, atm["radii"])
        _, dsdz_b, mask_b = calculate_rays(mu, atm["radii"] * 3.0)
        np.testing.assert_array_equal(np.asarray(mask_a), np.asarray(mask_b))
        np.testing.assert_allclose(np.asarray(dsdz_a), np.asarray(dsdz_b), rtol=1e-12)


# ---------------------------------------------------------------------------
# Agreement with Korg.jl
# ---------------------------------------------------------------------------


class TestCalculateRaysReference:
    """``calculate_rays`` against Korg.jl's own ray geometry."""

    @pytest.mark.parametrize("case", ["extended", "thin"])
    def test_atmosphere_construction_matches_julia(self, spherical_ref, case):
        """
        The Python and Julia atmosphere builders must agree first.

        Everything downstream compares against Korg.jl evaluated on *these*
        radii, so a mismatch here would silently weaken every later test.
        """
        ref = spherical_ref["cases"][case]
        atm = shell_atmosphere(
            thickness_over_radius=ref["thickness_over_radius"])
        np.testing.assert_allclose(atm["radii"], np.array(ref["radii"]), rtol=1e-13)
        np.testing.assert_allclose(atm["alpha_ref"], np.array(ref["alpha_ref"]),
                                   rtol=1e-13)

    @pytest.mark.parametrize("case", ["extended", "thin"])
    def test_layer_counts_match_julia(self, spherical_ref, case):
        """Each ray must intersect exactly the layers Korg.jl says it does."""
        ref = spherical_ref["cases"][case]
        radii = np.array(ref["radii"])
        mu, _ = generate_mu_grid(spherical_ref["inputs"]["mu_geom_n_points"])
        _, _, mask = calculate_rays(mu, radii)
        counts = np.asarray(mask).sum(axis=1)
        expected = np.array([r["n_layers"] for r in ref["rays"]])
        np.testing.assert_array_equal(counts, expected)

    @pytest.mark.parametrize("case", ["extended", "thin"])
    def test_path_and_dsdr_match_julia(self, spherical_ref, case):
        """``s`` and ``ds/dr`` on the intersected layers, to round-off."""
        ref = spherical_ref["cases"][case]
        radii = np.array(ref["radii"])
        mu, _ = generate_mu_grid(spherical_ref["inputs"]["mu_geom_n_points"])
        path, dsdz, _ = calculate_rays(mu, radii)
        for j, ray in enumerate(ref["rays"]):
            n = ray["n_layers"]
            np.testing.assert_allclose(np.asarray(path)[j, :n], np.array(ray["s"]),
                                       rtol=1e-13, err_msg=f"s mismatch for ray {j}")
            np.testing.assert_allclose(np.asarray(dsdz)[j, :n], np.array(ray["dsdr"]),
                                       rtol=1e-13, err_msg=f"ds/dr mismatch for ray {j}")


class TestSphericalFluxReference:
    """The spherical solver against Korg.jl's ``radiative_transfer``."""

    @staticmethod
    def _inputs(spherical_ref, case):
        ref = spherical_ref["cases"][case]
        return dict(
            alpha=np.array(ref["alpha"]),
            S=np.array(spherical_ref["inputs"]["S"]),
            radii=np.array(ref["radii"]),
            log_tau_ref=np.log10(np.array(spherical_ref["inputs"]["tau_ref"])),
            alpha_ref=np.array(ref["alpha_ref"]),
        )

    @pytest.mark.parametrize("case", ["extended", "thin"])
    @pytest.mark.parametrize("scheme", ["linear_flux_only", "linear"])
    @pytest.mark.parametrize("n_mu", [5, 20])
    def test_flux_matches_julia(self, spherical_ref, case, scheme, n_mu):
        """
        Emergent flux, to 1e-10 relative.

        Both of Korg.jl's linear intensity schemes are checked: shell
        atmospheres default to ``"linear"`` (the recursion that stores the
        intensity at every layer) while ``"linear_flux_only"`` sums the same
        integral segment by segment.
        """
        ref = spherical_ref["cases"][case]["solver"][f"{scheme}_{n_mu}"]
        args = self._inputs(spherical_ref, case)
        fluxes, _ = radiative_transfer_spherical(
            args["alpha"], args["S"], args["radii"], args["log_tau_ref"],
            args["alpha_ref"], n_mu=n_mu, intensity_scheme=scheme,
        )
        np.testing.assert_allclose(np.asarray(fluxes), np.array(ref["flux"]),
                                   rtol=1e-10)

    @pytest.mark.parametrize("case", ["extended", "thin"])
    @pytest.mark.parametrize("n_mu", [5, 20])
    def test_mu_grid_matches_julia(self, spherical_ref, case, n_mu):
        """The quadrature Korg.jl used is the quadrature we use."""
        ref = spherical_ref["cases"][case]["solver"][f"linear_flux_only_{n_mu}"]
        mu, weights = generate_mu_grid(n_mu)
        np.testing.assert_allclose(np.asarray(mu), np.array(ref["mu_grid"]), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(weights), np.array(ref["mu_weights"]),
                                   rtol=1e-12)

    @pytest.mark.parametrize("case", ["extended", "thin"])
    def test_surface_intensities_match_julia(self, spherical_ref, case):
        """
        Ray by ray, not just the flux.

        The flux is a weighted sum, so a compensating error in two rays could
        hide in it.  The tolerance is loosened for rays whose intensity is a
        tiny fraction of the disc-centre value: those are grazing rays that
        barely clip the outermost layers, where the seed and emission terms
        cancel to several digits.
        """
        ref = spherical_ref["cases"][case]["solver"]["linear_flux_only_20"]
        args = self._inputs(spherical_ref, case)
        _, intensities = radiative_transfer_spherical(
            args["alpha"], args["S"], args["radii"], args["log_tau_ref"],
            args["alpha_ref"], n_mu=20,
        )
        intensities = np.asarray(intensities)
        expected = np.array(ref["surface_I"]).T  # stored as (n_mu, n_wavelength)
        assert intensities.shape == expected.shape
        atol = 1e-8 * np.max(np.abs(expected))
        np.testing.assert_allclose(intensities, expected, rtol=1e-8, atol=atol)

    @pytest.mark.parametrize("case", ["extended", "thin"])
    def test_intensity_schemes_agree_with_each_other(self, spherical_ref, case):
        """
        ``"linear"`` and ``"linear_flux_only"`` are algebraically identical.

        They differ only in the order of the floating-point operations, so
        disagreement beyond round-off means one of them is wrong.
        """
        args = self._inputs(spherical_ref, case)
        fluxes = [
            np.asarray(radiative_transfer_spherical(
                args["alpha"], args["S"], args["radii"], args["log_tau_ref"],
                args["alpha_ref"], n_mu=20, intensity_scheme=scheme)[0])
            for scheme in ("linear_flux_only", "linear")
        ]
        np.testing.assert_allclose(fluxes[0], fluxes[1], rtol=1e-11)

    def test_photosphere_correction_matches_korgs_shell_wrapper(self, spherical_ref):
        """
        Korg.jl rescales a shell atmosphere's flux by ``(r[0]/R)²``.

        The low-level solver returns the flux at the outermost radius; the
        optional *R_photosphere* argument applies the same correction the
        ``ShellAtmosphere`` method of Korg.jl's ``radiative_transfer`` does.
        """
        args = self._inputs(spherical_ref, "extended")
        R = 1.0e12
        base, _ = radiative_transfer_spherical(
            args["alpha"], args["S"], args["radii"], args["log_tau_ref"],
            args["alpha_ref"], n_mu=20)
        scaled, _ = radiative_transfer_spherical(
            args["alpha"], args["S"], args["radii"], args["log_tau_ref"],
            args["alpha_ref"], n_mu=20, R_photosphere=R)
        np.testing.assert_allclose(
            np.asarray(scaled),
            np.asarray(base) * (args["radii"][0] / R) ** 2, rtol=1e-14)


# ---------------------------------------------------------------------------
# Spherical versus plane-parallel
# ---------------------------------------------------------------------------


class TestSphericalVersusPlanar:
    """Sphericity must change the answer, and must stop doing so as t/R -> 0."""

    def test_spherical_differs_from_planar(self, spherical_ref):
        """
        The regression this whole module exists for.

        ``spherical=True`` used to return bitwise plane-parallel values.
        """
        ref = spherical_ref["cases"]["extended"]
        args = TestSphericalFluxReference._inputs(spherical_ref, "extended")
        spherical, _ = radiative_transfer_spherical(
            args["alpha"], args["S"], args["radii"], args["log_tau_ref"],
            args["alpha_ref"], n_mu=20)
        planar = np.array(ref["planar"]["linear_20"])
        relative = np.abs(np.asarray(spherical) / planar - 1.0)
        assert np.all(relative > 1e-3), (
            "spherical geometry changed the flux by less than 0.1% in an "
            f"atmosphere {ref['thickness_over_radius']:.2f} R thick: {relative}"
        )

    @pytest.mark.parametrize("case", ["extended", "thin"])
    def test_plane_parallel_limit_matches_julia(self, spherical_ref, case):
        """
        The plane-parallel reference itself agrees with Korg.jl.

        The same ray machinery runs with ``spherical=False``, so this pins the
        comparison used by the convergence sweep below to Korg.jl rather than
        to another piece of this port.
        """
        ref = spherical_ref["cases"][case]
        args = TestSphericalFluxReference._inputs(spherical_ref, case)
        mu, weights = generate_mu_grid(20)
        fluxes, _ = spherical_ray_flux(
            args["alpha"], args["S"], args["radii"] - args["radii"][-1],
            args["log_tau_ref"], args["alpha_ref"], mu, weights,
            intensity_scheme="linear", spherical=False)
        np.testing.assert_allclose(np.asarray(fluxes),
                                   np.array(ref["planar"]["linear_20"]), rtol=1e-10)

    @pytest.mark.parametrize("thickness", [3e-1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    def test_thin_atmosphere_converges_to_plane_parallel(self, thickness):
        """
        Physical sanity: geometry stops mattering when the shell is thin.

        The comparison runs the *same* solver over plane-parallel rays, so the
        geometry is the only thing that differs.  The discrepancy must fall at
        least in proportion to the thickness.
        """
        atm = shell_atmosphere(thickness_over_radius=thickness)
        mu, weights = generate_mu_grid(20)
        spherical, _ = spherical_ray_flux(
            atm["alpha"], atm["S"], atm["radii"], atm["log_tau_ref"],
            atm["alpha_ref"], mu, weights)
        planar, _ = spherical_ray_flux(
            atm["alpha"], atm["S"], atm["radii"] - atm["radii"][-1],
            atm["log_tau_ref"], atm["alpha_ref"], mu, weights, spherical=False)
        relative = float(np.max(np.abs(np.asarray(spherical) / np.asarray(planar) - 1.0)))
        # First order in t/R, with a factor of a few of slack.
        assert relative < 5.0 * thickness, (
            f"spherical and plane-parallel differ by {relative:.3e} for an "
            f"atmosphere only {thickness:.0e} R thick"
        )

    def test_convergence_is_monotonic_in_thickness(self):
        """Thinner must mean closer, at every step of the sweep."""
        mu, weights = generate_mu_grid(20)
        errors = []
        for thickness in [3e-1, 1e-1, 1e-2, 1e-3, 1e-4]:
            atm = shell_atmosphere(thickness_over_radius=thickness)
            spherical, _ = spherical_ray_flux(
                atm["alpha"], atm["S"], atm["radii"], atm["log_tau_ref"],
                atm["alpha_ref"], mu, weights)
            planar, _ = spherical_ray_flux(
                atm["alpha"], atm["S"], atm["radii"] - atm["radii"][-1],
                atm["log_tau_ref"], atm["alpha_ref"], mu, weights, spherical=False)
            errors.append(float(np.max(np.abs(
                np.asarray(spherical) / np.asarray(planar) - 1.0))))
        assert all(later < earlier for earlier, later in zip(errors, errors[1:])), (
            f"error did not fall monotonically with thickness: {errors}"
        )


# ---------------------------------------------------------------------------
# Autodiff
# ---------------------------------------------------------------------------


class TestSphericalGradients:
    """
    ``jax.grad`` through the ray geometry.

    The tangent point is exactly the place where a masked NaN would hide: the
    path length vanishes there, ``sqrt`` has an infinite derivative at zero,
    and every padded layer has a negative radicand.
    """

    def test_gradient_wrt_alpha_is_finite_and_reduces_the_flux_overall(self):
        """
        More opacity, less flux -- but only *on balance* in spherical geometry.

        A plane-parallel ray always reaches optical depths where it saturates
        at the local source function, so extra opacity can only move the
        emitting surface outwards and reduce the flux: the gradient is
        negative everywhere.  A tangent ray is different.  It turns around at
        its own impact parameter, and if it never gets optically thick its
        emergent intensity is roughly ``S * τ_ray``, which *increases* with
        opacity.  So individual entries of the gradient are legitimately
        positive; what must be negative is the response to scaling the whole
        opacity column, which the disc-filling core rays dominate.
        """
        atm = shell_atmosphere()
        grad = np.asarray(jax.grad(
            lambda a: spherical_flux(atm, alpha=a))(jnp.asarray(atm["alpha"])))
        assert np.all(np.isfinite(grad)), f"d(flux)/d(alpha) is non-finite: {grad}"
        assert np.any(grad != 0.0), "d(flux)/d(alpha) is identically zero"
        scaling_response = float(np.sum(grad * atm["alpha"]))
        assert scaling_response < 0.0, (
            f"scaling the opacity column up increased the flux: {scaling_response}")

    def test_radial_ray_darkens_with_opacity_at_every_layer(self):
        """
        The μ = 1 ray passes through the core, so it behaves plane-parallel.

        This is the sign test the previous one cannot make globally: with
        ``b = 0`` the ray sees every layer and saturates, so extra opacity
        anywhere can only reduce its emergent intensity.
        """
        atm = shell_atmosphere()
        mu_grid = jnp.array([1.0])
        weights = jnp.array([1.0])

        def intensity(a):
            _, I = spherical_ray_flux(a, atm["S"], atm["radii"], atm["log_tau_ref"],
                                      atm["alpha_ref"], mu_grid, weights)
            return jnp.sum(I)

        grad = np.asarray(jax.grad(intensity)(jnp.asarray(atm["alpha"])))
        assert np.all(np.isfinite(grad))
        assert np.all(grad <= 0.0), (
            f"the radial ray brightened with opacity, max {grad.max()}")
        assert np.any(grad < 0.0)

    def test_gradient_wrt_S_is_finite_and_positive(self):
        """More source function, more flux."""
        atm = shell_atmosphere()
        grad = np.asarray(jax.grad(
            lambda s: spherical_flux(atm, S=s))(jnp.asarray(atm["S"])))
        assert np.all(np.isfinite(grad)), f"d(flux)/dS is non-finite: {grad}"
        assert np.all(grad >= 0.0), f"d(flux)/dS should be non-negative: {grad.min()}"
        assert np.any(grad > 0.0)

    def test_gradient_wrt_radii_is_finite_and_non_zero(self):
        """
        The radial coordinate is live now, which it was not before.

        In plane-parallel geometry the anchored scheme never touches the
        spatial coordinate; in spherical geometry it sets the impact parameter
        and every path length, so a zero gradient here would mean the geometry
        was still being ignored.
        """
        atm = shell_atmosphere()
        grad = np.asarray(jax.grad(
            lambda r: spherical_flux(atm, radii=r))(jnp.asarray(atm["radii"])))
        assert np.all(np.isfinite(grad)), f"d(flux)/d(radii) is non-finite: {grad}"
        assert np.any(grad != 0.0), (
            "d(flux)/d(radii) is identically zero -- the ray geometry is not "
            "connected to the flux"
        )

    def test_gradient_wrt_alpha_matches_finite_difference(self):
        """AD against a multiplicative central difference on the opacity."""
        atm = shell_atmosphere()
        assert_array_grad_matches_fd(
            lambda a: spherical_flux(atm, alpha=a), atm["alpha"],
            step=1e-6, name="spherical flux d/d(alpha)")

    def test_gradient_wrt_S_matches_finite_difference(self):
        """AD against a multiplicative central difference on the source function."""
        atm = shell_atmosphere()
        assert_array_grad_matches_fd(
            lambda s: spherical_flux(atm, S=s), atm["S"],
            step=1e-6, name="spherical flux d/dS")

    def test_gradient_wrt_radii_matches_finite_difference(self):
        """
        AD against a central difference in the radial coordinate.

        The perturbation direction is the height above the innermost shell:
        scaling *all* the radii uniformly rescales the impact parameter too and
        leaves ``ds/dr`` invariant, so that direction carries no derivative and
        would compare two pieces of noise.
        """
        atm = shell_atmosphere()
        assert_array_grad_matches_fd(
            lambda r: spherical_flux(atm, radii=r), atm["radii"],
            direction=atm["radii"] - atm["radii"][-1],
            step=1e-6, name="spherical flux d/d(radii)")

    def test_uniform_rescaling_carries_no_gradient(self):
        """
        Scaling the whole star leaves ``ds/dr`` -- and so the flux -- unchanged.

        A useful independent check that the geometry enters only through
        ratios, and a reminder of why the finite-difference direction above is
        chosen the way it is.
        """
        atm = shell_atmosphere()
        grad = np.asarray(jax.grad(
            lambda r: spherical_flux(atm, radii=r))(jnp.asarray(atm["radii"])))
        directional = float(np.sum(grad * atm["radii"]))
        flux = float(spherical_flux(atm))
        assert abs(directional) < 1e-8 * abs(flux), (
            f"a uniform rescaling changed the flux: d(flux)/d(scale) = {directional}")

    def test_gradient_is_finite_when_a_ray_is_exactly_tangent(self):
        """
        The failure mode this codebase has hit eight times.

        A μ chosen so that ``b`` lands exactly on a layer radius makes
        ``r² - b² = 0`` there: ``sqrt`` is finite but its derivative is not,
        and a masked branch would return a healthy value with a NaN cotangent.
        """
        atm = shell_atmosphere()
        radii = atm["radii"]
        # b = r[0] sqrt(1 - mu**2) = radii[k]  =>  mu = sqrt(1 - (radii[k]/radii[0])**2)
        mus = np.sqrt(1.0 - (radii[[5, 12, 18]] / radii[0]) ** 2)
        mu_grid = jnp.asarray(mus)
        weights = jnp.full((mus.size,), 1.0 / mus.size)

        path, _, _ = calculate_rays(mu_grid, radii)
        # The tangent layer's path length is the floored value, not a NaN.
        assert np.all(np.isfinite(np.asarray(path)))

        def flux(r):
            fluxes, _ = spherical_ray_flux(
                atm["alpha"], atm["S"], r, atm["log_tau_ref"], atm["alpha_ref"],
                mu_grid, weights)
            return jnp.sum(fluxes)

        value = float(flux(jnp.asarray(radii)))
        assert np.isfinite(value), f"exactly tangent ray gave flux {value}"
        grad = np.asarray(jax.grad(flux)(jnp.asarray(radii)))
        assert np.all(np.isfinite(grad)), (
            f"exactly tangent ray poisoned d(flux)/d(radii): {grad}")

    def test_gradient_is_finite_for_a_radial_ray(self):
        """μ = 1 gives b = 0, i.e. ``sqrt(1 - μ²)`` right on its own pole."""
        atm = shell_atmosphere()
        mu_grid = jnp.array([1.0])
        weights = jnp.array([1.0])

        def flux(r):
            fluxes, _ = spherical_ray_flux(
                atm["alpha"], atm["S"], r, atm["log_tau_ref"], atm["alpha_ref"],
                mu_grid, weights)
            return jnp.sum(fluxes)

        assert np.isfinite(float(flux(jnp.asarray(atm["radii"]))))
        grad = np.asarray(jax.grad(flux)(jnp.asarray(atm["radii"])))
        assert np.all(np.isfinite(grad)), f"radial ray gave a NaN gradient: {grad}"

    @pytest.mark.parametrize("thickness", [0.5, 0.3, 1e-2, 1e-5])
    def test_gradients_are_finite_across_the_thickness_sweep(self, thickness):
        """
        A very thin shell puts every ray's tangent point near the surface.

        This is where the floored radicand does its work: as ``t/R -> 0`` the
        smallest genuine ``r² - b²`` shrinks with it.
        """
        atm = shell_atmosphere(thickness_over_radius=thickness)
        for name, key, x in (("alpha", "alpha", atm["alpha"]),
                             ("S", "S", atm["S"]),
                             ("radii", "radii", atm["radii"])):
            grad = np.asarray(jax.grad(
                lambda v: spherical_flux(atm, **{key: v}))(jnp.asarray(x)))
            assert np.all(np.isfinite(grad)), (
                f"d(flux)/d({name}) is non-finite at t/R={thickness:g}: {grad}")

    def test_isothermal_flat_opacity_atmosphere_is_differentiable(self):
        """
        Constant S and constant α make several denominators vanish identically.

        The same degeneracy that produced NaN gradients from
        ``fritsch_butland_C`` elsewhere in this package.
        """
        atm = shell_atmosphere()
        alpha = jnp.full(atm["alpha"].shape, 1.0e-10)
        S = jnp.full(atm["S"].shape, 5.0e-5)
        value = float(spherical_flux(atm, alpha=alpha, S=S))
        assert np.isfinite(value)
        grad_alpha = np.asarray(jax.grad(
            lambda a: spherical_flux(atm, alpha=a, S=S))(alpha))
        grad_S = np.asarray(jax.grad(
            lambda s: spherical_flux(atm, alpha=alpha, S=s))(S))
        assert np.all(np.isfinite(grad_alpha)), grad_alpha
        assert np.all(np.isfinite(grad_S)), grad_S

    @pytest.mark.parametrize("scheme", ["linear_flux_only", "linear", "bezier"])
    def test_every_intensity_scheme_is_differentiable(self, scheme):
        """All three of Korg.jl's intensity schemes must carry a gradient."""
        atm = shell_atmosphere()

        def flux(a):
            fluxes, _ = radiative_transfer_spherical(
                a, atm["S"], atm["radii"], atm["log_tau_ref"], atm["alpha_ref"],
                n_mu=7, intensity_scheme=scheme)
            return jnp.sum(fluxes)

        grad = np.asarray(jax.grad(flux)(jnp.asarray(atm["alpha"])))
        assert np.all(np.isfinite(grad)), f"{scheme}: non-finite gradient {grad}"
        assert np.any(grad != 0.0), f"{scheme}: identically zero gradient"

    def test_bezier_tau_scheme_is_differentiable(self):
        """The Bezier optical depth runs along the ray's path coordinate."""
        atm = shell_atmosphere()

        def flux(a):
            fluxes, _ = radiative_transfer_spherical(
                a, atm["S"], atm["radii"], atm["log_tau_ref"], atm["alpha_ref"],
                n_mu=7, tau_scheme="bezier")
            return jnp.sum(fluxes)

        grad = np.asarray(jax.grad(flux)(jnp.asarray(atm["alpha"])))
        assert np.all(np.isfinite(grad)), f"non-finite gradient {grad}"
        assert np.any(grad != 0.0)


# ---------------------------------------------------------------------------
# jit tracing
# ---------------------------------------------------------------------------


class TestSphericalJit:
    """
    Fixed-size padded geometry means the whole solver traces.

    The number of layers a ray intersects depends on the data, so a
    ``lowest_layer_index`` used as an array bound would make this impossible.
    """

    def test_flux_is_jit_traceable(self):
        """``jax.jit`` over the flux computation must work at all."""
        atm = shell_atmosphere()
        jitted = jax.jit(lambda a, s, r, lt, ar: radiative_transfer_spherical(
            a, s, r, lt, ar, n_mu=20)[0])
        fluxes = np.asarray(jitted(jnp.asarray(atm["alpha"]), jnp.asarray(atm["S"]),
                                   jnp.asarray(atm["radii"]),
                                   jnp.asarray(atm["log_tau_ref"]),
                                   jnp.asarray(atm["alpha_ref"])))
        assert np.all(np.isfinite(fluxes)) and np.all(fluxes > 0.0)

    def test_jit_and_eager_agree(self):
        """Tracing must not change the answer."""
        atm = shell_atmosphere()

        def flux(a, s, r, lt, ar):
            return radiative_transfer_spherical(a, s, r, lt, ar, n_mu=20)[0]

        args = [jnp.asarray(atm[k]) for k in
                ("alpha", "S", "radii", "log_tau_ref", "alpha_ref")]
        # XLA is free to fuse and reassociate, so this is round-off agreement
        # rather than bitwise agreement.
        np.testing.assert_allclose(np.asarray(jax.jit(flux)(*args)),
                                   np.asarray(flux(*args)), rtol=1e-11)

    def test_calculate_rays_is_jit_traceable(self):
        """The geometry alone traces, masks and all."""
        atm = shell_atmosphere()
        mu, _ = generate_mu_grid(9)
        jitted = jax.jit(lambda m, r: calculate_rays(m, r))
        path, dsdz, mask = jitted(jnp.asarray(mu), jnp.asarray(atm["radii"]))
        path_ref, dsdz_ref, mask_ref = calculate_rays(mu, atm["radii"])
        # The mask is the part that must be exact: it is what makes the
        # data-dependent ray length representable at a fixed shape.  The
        # floating point values agree only to round-off because XLA fuses
        # ``r**2 - b**2`` differently under jit.
        np.testing.assert_array_equal(np.asarray(mask), np.asarray(mask_ref))
        np.testing.assert_allclose(np.asarray(path), np.asarray(path_ref), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(dsdz), np.asarray(dsdz_ref), rtol=1e-12)

    def test_ray_geometry_shapes_do_not_depend_on_the_data(self):
        """
        Two atmospheres with different ray structure must compile once.

        If the padded representation leaked the intersected-layer count into a
        shape, this would retrace (or fail) for the second atmosphere.
        """
        thick = shell_atmosphere(thickness_over_radius=0.5)
        thin = shell_atmosphere(thickness_over_radius=1e-4)
        mu, _ = generate_mu_grid(20)

        counts = [np.asarray(calculate_rays(mu, a["radii"])[2]).sum(axis=1)
                  for a in (thick, thin)]
        assert not np.array_equal(counts[0], counts[1]), (
            "the two atmospheres were supposed to have different ray structure")

        jitted = jax.jit(lambda a, s, r, lt, ar: radiative_transfer_spherical(
            a, s, r, lt, ar, n_mu=20)[0])
        for atm in (thick, thin):
            out = jitted(jnp.asarray(atm["alpha"]), jnp.asarray(atm["S"]),
                         jnp.asarray(atm["radii"]), jnp.asarray(atm["log_tau_ref"]),
                         jnp.asarray(atm["alpha_ref"]))
            assert np.all(np.isfinite(np.asarray(out)))
        assert jitted._cache_size() == 1, (
            "the spherical solver recompiled for a different ray structure")

    def test_grad_under_jit(self):
        """``jax.jit(jax.grad(...))`` -- the combination synthesis fitting uses."""
        atm = shell_atmosphere()
        grad_fn = jax.jit(jax.grad(lambda a: spherical_flux(atm, alpha=a)))
        grad = np.asarray(grad_fn(jnp.asarray(atm["alpha"])))
        eager = np.asarray(jax.grad(lambda a: spherical_flux(atm, alpha=a))(
            jnp.asarray(atm["alpha"])))
        assert np.all(np.isfinite(grad))
        # Reverse-mode accumulation amplifies the fusion differences of the
        # forward pass, so this is a round-off comparison, not a bitwise one.
        np.testing.assert_allclose(grad, eager, rtol=1e-7)

    def test_vmap_over_atmospheres(self):
        """The solver batches, which requires the geometry to be shape-stable."""
        atms = [shell_atmosphere(thickness_over_radius=t) for t in (0.3, 0.1, 0.01)]
        radii = jnp.stack([jnp.asarray(a["radii"]) for a in atms])
        alpha_ref = jnp.stack([jnp.asarray(a["alpha_ref"]) for a in atms])
        alpha = jnp.stack([jnp.asarray(a["alpha"]) for a in atms])

        batched = jax.vmap(lambda a, r, ar: radiative_transfer_spherical(
            a, jnp.asarray(atms[0]["S"]), r, jnp.asarray(atms[0]["log_tau_ref"]),
            ar, n_mu=20)[0])
        out = np.asarray(batched(alpha, radii, alpha_ref))
        assert out.shape == (3, 2)
        for i, atm in enumerate(atms):
            single, _ = radiative_transfer_spherical(
                atm["alpha"], atms[0]["S"], atm["radii"], atms[0]["log_tau_ref"],
                atm["alpha_ref"], n_mu=20)
            np.testing.assert_allclose(out[i], np.asarray(single), rtol=1e-12)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


class TestSphericalApi:
    """The entry points, and the error that replaced the silent no-op."""

    def test_compute_tau_anchored_rejects_spherical_without_geometry(self):
        """
        ``spherical=True`` with no ray geometry used to be silently ignored.

        It now raises, because a plane-parallel answer returned under a
        spherical flag is worse than no answer at all.
        """
        from korg.radiative_transfer.optical_depth import compute_tau_anchored
        atm = shell_atmosphere()
        with pytest.raises(ValueError, match="calculate_rays"):
            compute_tau_anchored(atm["alpha"][0], atm["radii"], atm["log_tau_ref"],
                                 atm["alpha_ref"], spherical=True)

    def test_radiative_transfer_dispatches_on_the_spherical_flag(self):
        """``radiative_transfer(..., spherical=True)`` reaches the ray solver."""
        from korg.radiative_transfer.core import radiative_transfer
        atm = shell_atmosphere()
        fluxes, intensities = radiative_transfer(
            atm["alpha"], atm["S"], atm["radii"], atm["log_tau_ref"],
            alpha_ref=atm["alpha_ref"], spherical=True, n_mu=20)
        expected, _ = radiative_transfer_spherical(
            atm["alpha"], atm["S"], atm["radii"], atm["log_tau_ref"],
            atm["alpha_ref"], n_mu=20)
        np.testing.assert_allclose(np.asarray(fluxes), np.asarray(expected), rtol=1e-14)
        assert intensities.shape == (atm["alpha"].shape[0], 20)

    def test_single_wavelength_entry_point(self):
        """The scalar-wavelength wrapper agrees with the batched solver."""
        from korg.radiative_transfer.core import radiative_transfer_single_wavelength
        atm = shell_atmosphere()
        flux, intensity = radiative_transfer_single_wavelength(
            atm["alpha"][0], atm["S"][0], atm["radii"], atm["log_tau_ref"],
            alpha_ref=atm["alpha_ref"], spherical=True, n_mu=20)
        expected, _ = radiative_transfer_spherical(
            atm["alpha"][:1], atm["S"][:1], atm["radii"], atm["log_tau_ref"],
            atm["alpha_ref"], n_mu=20)
        np.testing.assert_allclose(float(flux), float(expected[0]), rtol=1e-14)
        assert intensity.shape == (20,)

    def test_explicit_mu_values_use_trapezoid_weights(self):
        """
        ``generate_mu_grid`` accepts a μ vector, as Korg.jl's does.

        The weights are the trapezoid rule, and must sum to the width of the
        interval covered.
        """
        mu = np.array([0.1, 0.4, 0.7, 1.0])
        grid, weights = generate_mu_grid(mu)
        np.testing.assert_allclose(np.asarray(grid), mu)
        np.testing.assert_allclose(float(np.sum(np.asarray(weights))), 0.9, rtol=1e-14)

    def test_unknown_scheme_names_are_rejected(self):
        """A typo in a scheme name must not fall through to a default."""
        atm = shell_atmosphere()
        with pytest.raises(ValueError, match="tau_scheme"):
            radiative_transfer_spherical(
                atm["alpha"], atm["S"], atm["radii"], atm["log_tau_ref"],
                atm["alpha_ref"], tau_scheme="nonsense")
        with pytest.raises(ValueError, match="intensity_scheme"):
            radiative_transfer_spherical(
                atm["alpha"], atm["S"], atm["radii"], atm["log_tau_ref"],
                atm["alpha_ref"], intensity_scheme="nonsense")
