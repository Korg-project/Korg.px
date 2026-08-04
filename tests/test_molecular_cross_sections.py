"""
Tests for ``korg.molecular_cross_sections``.

1. Functional — construction, option combinations, interpolation at and between
   grid nodes, out-of-bounds behaviour, HDF5 round trip, error paths.
2. Julia agreement — the precomputed (vmic, log T, λ) grid and a set of off-grid
   interpolation queries are compared against Korg.jl v1.2.1
   (``tests/linelist_reference_data.json``).
3. Autodiff — the module holds no JAX kernel of its own; the differentiable
   quantity is the interpolation, which goes through SciPy.  That is asserted
   explicitly rather than left implicit.
4. jit-tracing — SciPy's RegularGridInterpolator is not a JAX primitive, so the
   cross-section cannot be traced; that is pinned with ``pytest.raises``.
"""

import json
from pathlib import Path

import korg  # noqa: F401 — enables JAX x64 mode

import h5py
import jax
import numpy as np
import pytest

REFERENCE_FILE = Path(__file__).parent / "linelist_reference_data.json"


@pytest.fixture(scope="module")
def mol_ref():
    if not REFERENCE_FILE.exists():
        pytest.fail(
            f"{REFERENCE_FILE} is missing. Regenerate with "
            "`julia --project=. tests/generate_linelist_reference.jl`."
        )
    with open(REFERENCE_FILE) as f:
        return json.load(f)["molecular_cross_section"]


@pytest.fixture(scope="module")
def cn_lines(mol_ref):
    from korg.linelist import create_line

    return [create_line(wl, log_gf, sp, E) for wl, log_gf, sp, E in mol_ref["lines"]]


@pytest.fixture(scope="module")
def cn_sigma(mol_ref, cn_lines):
    from korg.molecular_cross_sections import MolecularCrossSection

    return MolecularCrossSection(
        cn_lines,
        mol_ref["wavelengths_angstrom"],
        vmic_vals=mol_ref["vmic_vals"],
        log_temp_vals=mol_ref["log_temp_vals"],
    )


# ===========================================================================
# 2. Julia agreement
# ===========================================================================

class TestAgainstJulia:

    def test_grid_matches_julia(self, cn_sigma, mol_ref):
        julia = np.array(mol_ref["grid"])
        python = np.asarray(cn_sigma._grid)
        assert python.shape == julia.shape
        # Measured against each (vmic, log T) slice's peak absorption, which is
        # the physically meaningful scale: every point agrees to better than
        # 1e-10 of the line-core opacity.
        peak = np.max(python, axis=2, keepdims=True)
        assert np.all(np.abs(python - julia) <= 1e-10 * peak), (
            "worst deviation relative to the line-core opacity: "
            f"{np.max(np.abs(python - julia) / peak):.3e}"
        )
        # Point-by-point relative error.  Half the grid agrees to round-off.  The
        # residual sits in the far wings, ~5 dex below the line core, where the
        # rational approximation used by korg.line_absorption's Voigt kernel
        # differs from Korg.jl's in its last few digits.  That kernel belongs to
        # a different module, so bound it here rather than chase it.
        rel = np.abs(python - julia) / np.maximum(np.abs(julia), 1e-300)
        assert np.median(rel) < 1e-13, f"median relative error {np.median(rel):.3e}"
        assert np.quantile(rel, 0.90) < 1e-10, (
            f"90th percentile relative error {np.quantile(rel, 0.90):.3e}"
        )
        assert rel.max() < 1e-7, f"max relative error {rel.max():.3e}"

    def test_interpolation_queries_match_julia(self, cn_sigma, mol_ref):
        for query, expected in zip(mol_ref["queries"], mol_ref["query_values"]):
            got = cn_sigma(*query)
            assert np.isclose(got, expected, rtol=1e-10, atol=1e-300), (
                f"query {query}: {got} != {expected}"
            )

    def test_species_matches_julia(self, cn_sigma, mol_ref):
        assert str(cn_sigma.species) == mol_ref["species"]

    def test_out_of_bounds_query_is_zero(self, cn_sigma, mol_ref):
        """Korg.jl extrapolates with 0.0; so must we."""
        assert cn_sigma(1.0, 3.6, 4000.0) == 0.0
        assert cn_sigma(1.0, 3.6, 9000.0) == 0.0
        assert cn_sigma(99.0, 3.6, 5000.0) == 0.0


# ===========================================================================
# 1. Functional
# ===========================================================================

class TestConstruction:

    def test_empty_linelist_raises(self):
        from korg.molecular_cross_sections import MolecularCrossSection

        with pytest.raises(ValueError, match="linelist cannot be empty"):
            MolecularCrossSection([], np.linspace(5000, 5001, 5))

    def test_mixed_species_raises(self):
        from korg.linelist import create_line
        from korg.molecular_cross_sections import MolecularCrossSection

        lines = [create_line(5000.0, -1.0, "CN", 0.5),
                 create_line(5000.5, -1.0, "CH", 0.5)]
        with pytest.raises(ValueError, match="same species"):
            MolecularCrossSection(lines, np.linspace(5000, 5001, 5))

    def test_single_line(self):
        from korg.linelist import create_line
        from korg.molecular_cross_sections import MolecularCrossSection

        sigma = MolecularCrossSection([create_line(5000.0, -1.0, "CN", 0.5)],
                                      np.linspace(4999.5, 5000.5, 11),
                                      vmic_vals=[1.0], log_temp_vals=[3.5])
        assert sigma._grid.shape == (1, 1, 11)
        assert np.all(sigma._grid >= 0)
        assert sigma._grid.max() > 0

    def test_default_grids_match_korg(self):
        """
        Korg.jl's defaults are
        ``vmic_vals = [(0:1/3:1)...; 1.5; (2:2/3:16/3)...]`` and
        ``log_temp_vals = 3:0.04:5``.
        """
        from korg.linelist import create_line
        from korg.molecular_cross_sections import MolecularCrossSection

        sigma = MolecularCrossSection([create_line(5000.0, -1.0, "CN", 0.5)],
                                      np.array([5000.0]))
        assert np.allclose(sigma.vmic_vals,
                           [0.0, 1 / 3, 2 / 3, 1.0, 1.5, 2.0, 8 / 3, 10 / 3,
                            4.0, 14 / 3, 16 / 3], rtol=1e-14)
        assert len(sigma.log_temp_vals) == 51
        assert np.isclose(sigma.log_temp_vals[0], 3.0)
        assert np.isclose(sigma.log_temp_vals[-1], 5.0)
        assert np.isclose(np.diff(sigma.log_temp_vals).max(), 0.04)

    def test_cutoff_alpha_scaling_is_a_no_op_on_the_result(self):
        """
        ``cutoff_alpha`` only rescales the number density used internally; the
        returned cross-section must be (nearly) independent of it.
        """
        from korg.linelist import create_line
        from korg.molecular_cross_sections import MolecularCrossSection

        lines = [create_line(5000.0, -1.0, "CN", 0.5)]
        wls = np.linspace(4999.8, 5000.2, 9)
        a = MolecularCrossSection(lines, wls, cutoff_alpha=1e-32,
                                  vmic_vals=[1.0], log_temp_vals=[3.5])
        b = MolecularCrossSection(lines, wls, cutoff_alpha=1e-30,
                                  vmic_vals=[1.0], log_temp_vals=[3.5])
        assert np.allclose(a._grid, b._grid, rtol=1e-10, atol=0.0)

    def test_larger_vmic_broadens_the_line(self, cn_sigma, mol_ref):
        """The line core weakens and the wings strengthen as vmic grows."""
        grid = np.asarray(cn_sigma._grid)
        wls = np.asarray(cn_sigma.wavelengths_angstrom)
        core = int(np.argmax(grid[0, 1]))
        assert grid[0, 1, core] > grid[2, 1, core]
        wing = int(np.argmin(np.abs(wls - (wls[core] + 0.5))))
        assert grid[2, 1, wing] > grid[0, 1, wing]

    def test_repr(self, cn_sigma):
        text = repr(cn_sigma)
        assert "MolecularCrossSection for CN" in text
        assert "K ≤ T ≤" in text
        assert "km/s ≤ vmic ≤" in text


class TestInterpolation:

    def test_call_at_grid_node_returns_grid_value(self, cn_sigma):
        grid = np.asarray(cn_sigma._grid)
        for i, v in enumerate(cn_sigma.vmic_vals):
            for j, lt in enumerate(cn_sigma.log_temp_vals):
                for k in (0, len(cn_sigma.wavelengths_angstrom) // 2, -1):
                    got = cn_sigma(float(v), float(lt),
                                   float(cn_sigma.wavelengths_angstrom[k]))
                    assert np.isclose(got, grid[i, j, k], rtol=1e-14, atol=0.0)

    def test_interpolate_layer_matches_scalar_calls(self, cn_sigma):
        layer = cn_sigma.interpolate_layer(0.5, 3.5)
        assert layer.shape == cn_sigma.wavelengths_angstrom.shape
        for k in range(0, len(layer), 7):
            assert np.isclose(layer[k],
                              cn_sigma(0.5, 3.5, float(cn_sigma.wavelengths_angstrom[k])),
                              rtol=1e-14)

    def test_interpolate_layer_at_node_returns_grid_row(self, cn_sigma):
        grid = np.asarray(cn_sigma._grid)
        layer = cn_sigma.interpolate_layer(float(cn_sigma.vmic_vals[1]),
                                           float(cn_sigma.log_temp_vals[2]))
        assert np.allclose(layer, grid[1, 2], rtol=1e-14, atol=0.0)

    def test_bilinear_midpoint(self, cn_sigma):
        """Linear interpolation: the midpoint is the mean of the two nodes."""
        grid = np.asarray(cn_sigma._grid)
        v0, v1 = cn_sigma.vmic_vals[0], cn_sigma.vmic_vals[1]
        lt = float(cn_sigma.log_temp_vals[0])
        wl = float(cn_sigma.wavelengths_angstrom[10])
        mid = cn_sigma(float(0.5 * (v0 + v1)), lt, wl)
        assert np.isclose(mid, 0.5 * (grid[0, 0, 10] + grid[1, 0, 10]),
                          rtol=1e-12, atol=0.0)


class TestPersistence:

    def test_save_read_round_trip(self, cn_sigma, tmp_path):
        from korg.molecular_cross_sections import (read_molecular_cross_section,
                                                   save_molecular_cross_section)

        path = tmp_path / "cn.h5"
        save_molecular_cross_section(str(path), cn_sigma)
        back = read_molecular_cross_section(str(path))

        assert back.species == cn_sigma.species
        assert np.array_equal(back.wavelengths_angstrom, cn_sigma.wavelengths_angstrom)
        assert np.array_equal(back.vmic_vals, cn_sigma.vmic_vals)
        assert np.array_equal(back.log_temp_vals, cn_sigma.log_temp_vals)
        assert np.array_equal(back._grid, cn_sigma._grid)

    def test_round_trip_preserves_interpolation(self, cn_sigma, tmp_path, mol_ref):
        from korg.molecular_cross_sections import (read_molecular_cross_section,
                                                   save_molecular_cross_section)

        path = tmp_path / "cn.h5"
        save_molecular_cross_section(str(path), cn_sigma)
        back = read_molecular_cross_section(str(path))
        for query, expected in zip(mol_ref["queries"], mol_ref["query_values"]):
            assert np.isclose(back(*query), expected, rtol=1e-10, atol=1e-300)

    def test_saved_datasets(self, cn_sigma, tmp_path):
        from korg.molecular_cross_sections import save_molecular_cross_section

        path = tmp_path / "cn.h5"
        save_molecular_cross_section(str(path), cn_sigma)
        with h5py.File(path, "r") as f:
            assert set(f.keys()) == {"wls_angstrom", "vmic_vals", "log_temp_vals",
                                     "alpha_grid", "species"}
            assert f["alpha_grid"].shape == cn_sigma._grid.shape

    def test_species_stored_as_bytes_is_decoded(self, cn_sigma, tmp_path):
        """h5py may hand back ``bytes`` or ``str``; both must work."""
        from korg.molecular_cross_sections import read_molecular_cross_section
        from korg.species import Species

        path = tmp_path / "cn_bytes.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("wls_angstrom", data=cn_sigma.wavelengths_angstrom)
            f.create_dataset("vmic_vals", data=cn_sigma.vmic_vals)
            f.create_dataset("log_temp_vals", data=cn_sigma.log_temp_vals)
            f.create_dataset("alpha_grid", data=cn_sigma._grid)
            f.create_dataset("species", data=np.bytes_("CN"))
        assert read_molecular_cross_section(str(path)).species == Species("CN")


class TestInterpolateMolecularCrossSections:

    def test_no_cross_sections_is_a_no_op(self):
        from korg.molecular_cross_sections import interpolate_molecular_cross_sections

        alpha = np.ones((3, 4))
        out = interpolate_molecular_cross_sections(alpha, [], np.linspace(5000, 5001, 4),
                                                   np.array([4000.0, 5000.0, 6000.0]),
                                                   1.0, {})
        assert out is alpha
        assert np.array_equal(out, np.ones((3, 4)))

    def test_scalar_vmic(self, cn_sigma):
        from korg.molecular_cross_sections import interpolate_molecular_cross_sections
        from korg.species import Species

        wls = np.asarray(cn_sigma.wavelengths_angstrom)
        temps = np.array([4000.0, 5000.0])
        alpha = np.zeros((2, len(wls)))
        n = np.array([1e10, 2e10])
        out = interpolate_molecular_cross_sections(
            alpha, [cn_sigma], wls, temps, 1.0, {Species("CN"): n})
        assert out is alpha
        for i in range(2):
            expected = cn_sigma.interpolate_layer(1.0, np.log10(temps[i])) * n[i]
            assert np.allclose(out[i], expected, rtol=1e-14)

    def test_per_layer_vmic(self, cn_sigma):
        from korg.molecular_cross_sections import interpolate_molecular_cross_sections
        from korg.species import Species

        wls = np.asarray(cn_sigma.wavelengths_angstrom)
        temps = np.array([4000.0, 5000.0])
        vmic = np.array([0.5, 1.5])
        alpha = np.zeros((2, len(wls)))
        n = np.array([1e10, 1e10])
        interpolate_molecular_cross_sections(
            alpha, [cn_sigma], wls, temps, vmic, {Species("CN"): n})
        for i in range(2):
            expected = cn_sigma.interpolate_layer(vmic[i], np.log10(temps[i])) * n[i]
            assert np.allclose(alpha[i], expected, rtol=1e-14)

    def test_number_densities_keyed_by_string(self, cn_sigma):
        from korg.molecular_cross_sections import interpolate_molecular_cross_sections

        wls = np.asarray(cn_sigma.wavelengths_angstrom)
        temps = np.array([4000.0])
        alpha = np.zeros((1, len(wls)))
        interpolate_molecular_cross_sections(
            alpha, [cn_sigma], wls, temps, 1.0, {"CN": np.array([1e10])})
        assert alpha.max() > 0

    def test_missing_species_is_skipped(self, cn_sigma):
        from korg.molecular_cross_sections import interpolate_molecular_cross_sections

        wls = np.asarray(cn_sigma.wavelengths_angstrom)
        alpha = np.zeros((1, len(wls)))
        interpolate_molecular_cross_sections(
            alpha, [cn_sigma], wls, np.array([4000.0]), 1.0, {})
        assert np.all(alpha == 0.0)

    def test_contributions_are_additive(self, cn_sigma):
        from korg.molecular_cross_sections import interpolate_molecular_cross_sections
        from korg.species import Species

        wls = np.asarray(cn_sigma.wavelengths_angstrom)
        alpha_once = np.zeros((1, len(wls)))
        alpha_twice = np.zeros((1, len(wls)))
        nd = {Species("CN"): np.array([1e10])}
        interpolate_molecular_cross_sections(alpha_once, [cn_sigma], wls,
                                             np.array([4000.0]), 1.0, nd)
        interpolate_molecular_cross_sections(alpha_twice, [cn_sigma, cn_sigma], wls,
                                             np.array([4000.0]), 1.0, nd)
        assert np.allclose(alpha_twice, 2 * alpha_once, rtol=1e-14)


# ===========================================================================
# 3. Autodiff and 4. jit-tracing
# ===========================================================================

class TestAutodiffAndJIT:

    def test_cross_section_is_not_differentiable(self, cn_sigma):
        """
        ``MolecularCrossSection`` stores a SciPy ``RegularGridInterpolator``.
        SciPy is not a JAX primitive, so ``jax.grad`` cannot see through it;
        pinning this documents that a JAX-native interpolator would be needed to
        differentiate a synthesis that uses precomputed molecular opacities.
        """
        with pytest.raises(Exception):
            jax.grad(lambda T: cn_sigma(1.0, T, 5000.0))(3.5)

    def test_cross_section_cannot_be_jitted(self, cn_sigma):
        """Same reason: the interpolation is host-side SciPy, not a JAX op."""
        with pytest.raises(Exception):
            jax.jit(lambda T: cn_sigma(1.0, T, 5000.0))(3.5)

    def test_interpolate_layer_cannot_be_jitted(self, cn_sigma):
        with pytest.raises(Exception):
            jax.jit(lambda v: cn_sigma.interpolate_layer(v, 3.5))(1.0)

    def test_finite_difference_sensitivity_is_physical(self, cn_sigma):
        """
        Even without autodiff the interpolant must vary smoothly and monotonically
        in log T between grid nodes; check by finite differences.
        """
        lt0, lt1 = float(cn_sigma.log_temp_vals[0]), float(cn_sigma.log_temp_vals[1])
        wl = float(cn_sigma.wavelengths_angstrom[len(cn_sigma.wavelengths_angstrom) // 2])
        xs = np.linspace(lt0, lt1, 5)
        ys = np.array([cn_sigma(1.0, float(x), wl) for x in xs])
        d = np.diff(ys)
        assert np.all(np.isfinite(ys))
        assert np.allclose(d, d[0], rtol=1e-10), "linear interpolation must have a constant slope"
