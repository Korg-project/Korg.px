"""
Tests for Priority 1-5 implementations.

Covers the functions listed in progress.md Priorities 1-5 that were not
already tested in other test files.
"""

import pytest
import numpy as np


# =============================================================================
# Priority 1: Core Synthesis Helpers
# =============================================================================

class TestPruneLinelist:
    """Tests for prune_linelist and merge_close_lines (Priority 1)."""

    def test_merge_close_lines_import(self):
        """merge_close_lines should be importable."""
        from korg.prune_linelist import merge_close_lines
        assert callable(merge_close_lines)

    def test_merge_close_lines_empty(self):
        """merge_close_lines should handle empty linelist."""
        from korg.prune_linelist import merge_close_lines
        result = merge_close_lines([])
        assert result == []

    def test_merge_close_lines_basic(self):
        """merge_close_lines should merge nearby lines."""
        from korg.prune_linelist import merge_close_lines
        from korg.linelist import create_line, Species

        spec = Species('Fe I')
        # Two lines 0.05 Å apart (< default merge_distance of 0.2 Å)
        lines = [
            create_line(5000.0, -1.0, spec, 1.0),
            create_line(5000.05, -1.5, spec, 1.0),
            create_line(5010.0, -1.0, spec, 1.0),  # far apart
        ]
        merged = merge_close_lines(lines, merge_distance=0.2)
        # First two should merge into one, third stays
        assert len(merged) == 2

    def test_prune_linelist_import(self):
        """prune_linelist should be importable."""
        from korg.prune_linelist import prune_linelist
        assert callable(prune_linelist)


# =============================================================================
# Priority 3: Molecular Cross-Sections
# =============================================================================

class TestMolecularCrossSection:
    """Tests for molecular cross-section functions (Priority 3)."""

    def test_import(self):
        """All Priority 3 functions should be importable."""
        from korg.molecular_cross_sections import (
            MolecularCrossSection,
            interpolate_molecular_cross_sections,
            save_molecular_cross_section,
            read_molecular_cross_section,
        )
        assert callable(MolecularCrossSection)
        assert callable(interpolate_molecular_cross_sections)
        assert callable(save_molecular_cross_section)
        assert callable(read_molecular_cross_section)

    def test_molecular_cross_section_signature(self):
        """MolecularCrossSection should have expected constructor parameters."""
        import inspect
        from korg.molecular_cross_sections import MolecularCrossSection
        sig = inspect.signature(MolecularCrossSection.__init__)
        params = list(sig.parameters.keys())
        assert 'linelist' in params
        assert 'wavelengths_angstrom' in params
        assert 'cutoff_alpha' in params

    def test_save_read_roundtrip(self, tmp_path):
        """save and read molecular cross-section should roundtrip correctly."""
        from korg.molecular_cross_sections import (
            MolecularCrossSection, save_molecular_cross_section,
            read_molecular_cross_section
        )
        from korg.linelist import create_line, Species

        spec = Species('CO')
        # Create a minimal linelist with 2 lines
        lines = [
            create_line(5000.0, -1.0, spec, 0.5),
            create_line(5001.0, -2.0, spec, 0.8),
        ]
        wls = np.linspace(4990, 5010, 50)

        # Create cross-section object (this computes the grid)
        cs = MolecularCrossSection(lines, wls, log_temp_vals=np.array([3.5, 3.6, 3.7]))

        # Save and reload
        path = str(tmp_path / "test_cs.h5")
        save_molecular_cross_section(path, cs)
        cs2 = read_molecular_cross_section(path)

        # Verify basic properties preserved
        assert cs2.species == cs.species
        np.testing.assert_allclose(cs2.wavelengths_angstrom, cs.wavelengths_angstrom)
        np.testing.assert_allclose(cs2.log_temp_vals, cs.log_temp_vals)

    def test_interpolate_molecular_cross_sections_signature(self):
        """interpolate_molecular_cross_sections should have expected parameters."""
        import inspect
        from korg.molecular_cross_sections import interpolate_molecular_cross_sections
        params = list(inspect.signature(interpolate_molecular_cross_sections).parameters.keys())
        assert 'alpha' in params
        assert 'molecular_cross_sections' in params
        assert 'wavelengths_angstrom' in params
        assert 'temperatures' in params


# =============================================================================
# Priority 4: Bezier Radiative Transfer Solver
# =============================================================================

class TestBezierRT:
    """Tests for compute_tau_bezier (Priority 4)."""

    def test_import(self):
        """compute_tau_bezier should be importable."""
        from korg.radiative_transfer.optical_depth import compute_tau_bezier
        assert callable(compute_tau_bezier)

    def test_basic_computation(self):
        """compute_tau_bezier should produce monotonically increasing tau."""
        import jax.numpy as jnp
        from korg.radiative_transfer.optical_depth import compute_tau_bezier

        n_layers = 15
        alpha = jnp.ones(n_layers) * 1e-4
        z = jnp.linspace(0, 1e9, n_layers)  # spatial coordinate in cm

        tau = compute_tau_bezier(alpha, z)
        assert tau.shape == (n_layers,), f"Expected shape ({n_layers},), got {tau.shape}"
        # tau should increase with depth (larger index)
        assert jnp.all(jnp.diff(tau) >= 0), "tau should be non-decreasing"

    def test_tau_starts_near_zero(self):
        """compute_tau_bezier first layer should start at a small tau."""
        import jax.numpy as jnp
        from korg.radiative_transfer.optical_depth import compute_tau_bezier

        alpha = jnp.ones(10) * 1e-4
        z = jnp.linspace(0, 1e9, 10)
        tau = compute_tau_bezier(alpha, z)
        # The implementation uses 1e-5 as the initial surface optical depth
        assert float(tau[0]) < 1e-3, \
            f"First tau should be near zero (surface), got {float(tau[0])}"

    def test_larger_alpha_larger_tau(self):
        """Higher opacity should give larger optical depth."""
        import jax.numpy as jnp
        from korg.radiative_transfer.optical_depth import compute_tau_bezier

        z = jnp.linspace(0, 1e9, 10)
        tau_low = compute_tau_bezier(jnp.ones(10) * 1e-5, z)
        tau_high = compute_tau_bezier(jnp.ones(10) * 1e-3, z)
        assert float(tau_high[-1]) > float(tau_low[-1]), \
            "Higher alpha should give higher tau"

    def test_jit_compatible(self):
        """compute_tau_bezier should work under jax.jit."""
        import jax
        import jax.numpy as jnp
        from korg.radiative_transfer.optical_depth import compute_tau_bezier

        jitted = jax.jit(compute_tau_bezier)
        alpha = jnp.ones(8) * 1e-4
        z = jnp.linspace(0, 1e8, 8)
        tau = jitted(alpha, z)
        assert tau.shape == (8,)


# =============================================================================
# Priority 5: Atmosphere Readers
# =============================================================================

class TestAtmosphereReaders:
    """Tests for atmosphere reading functions (Priority 5)."""

    def test_read_phoenix_import(self):
        """_read_phoenix_model_atmosphere should be importable."""
        from korg.atmosphere import _read_phoenix_model_atmosphere
        assert callable(_read_phoenix_model_atmosphere)

    def test_read_phoenix_signature(self):
        """_read_phoenix_model_atmosphere should have fname parameter."""
        import inspect
        from korg.atmosphere import _read_phoenix_model_atmosphere
        params = list(inspect.signature(_read_phoenix_model_atmosphere).parameters.keys())
        assert 'fname' in params

    def test_lazy_multilinear_import(self):
        """lazy_multilinear_interpolation should be importable."""
        from korg.marcs_interpolation import lazy_multilinear_interpolation
        assert callable(lazy_multilinear_interpolation)

    def test_lazy_multilinear_basic(self):
        """lazy_multilinear_interpolation should interpolate over atmosphere grids."""
        from korg.marcs_interpolation import lazy_multilinear_interpolation

        # Grid has shape (n_layers, n_quant, *grid_dims) per the function spec
        # Simulate a 2-parameter grid with 3 layers, 2 quantities
        x_nodes = np.array([1.0, 2.0, 3.0])
        y_nodes = np.array([10.0, 20.0])

        # Grid shape: (n_layers=3, n_quant=2, len(x_nodes)=3, len(y_nodes)=2)
        grid = np.zeros((3, 2, 3, 2))
        for ix, x in enumerate(x_nodes):
            for iy, y in enumerate(y_nodes):
                grid[:, 0, ix, iy] = x + y   # quantity 0: x + y
                grid[:, 1, ix, iy] = x * y   # quantity 1: x * y

        nodes = [x_nodes, y_nodes]
        params = [1.5, 15.0]  # midpoint of both axes
        result = lazy_multilinear_interpolation(params, nodes, grid)
        # Expected: shape (n_layers=3, n_quant=2)
        assert result.shape == (3, 2), f"Expected shape (3, 2), got {result.shape}"
        # quantity 0 at (1.5, 15.0) = 1.5 + 15.0 = 16.5
        expected_q0 = 16.5
        assert np.allclose(result[:, 0], expected_q0, rtol=1e-5), \
            f"Expected {expected_q0}, got {result[:, 0]}"

    def test_lazy_multilinear_at_grid_points(self):
        """lazy_multilinear_interpolation should be exact at grid nodes."""
        from korg.marcs_interpolation import lazy_multilinear_interpolation

        x_nodes = np.array([0.0, 1.0, 2.0])
        y_nodes = np.array([0.0, 1.0])
        # Grid shape: (n_layers=2, n_quant=1, 3, 2)
        grid = np.zeros((2, 1, 3, 2))
        for ix, x in enumerate(x_nodes):
            for iy, y in enumerate(y_nodes):
                grid[:, 0, ix, iy] = x + y

        nodes = [x_nodes, y_nodes]
        # At grid point (1.0, 1.0): should give x+y = 2.0 for all layers
        result = lazy_multilinear_interpolation([1.0, 1.0], nodes, grid,
                                                perturb_at_grid_values=True)
        assert np.allclose(result[:, 0], 2.0, rtol=1e-5), \
            f"Expected 2.0, got {result[:, 0]}"
